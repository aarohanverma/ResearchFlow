"""ResearchFlow FastAPI application entry point.

Layer order: lifespan → middleware → routers → exception handlers.
No business logic here — just wiring.
"""

import logging
import os
import time
import uuid as _uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import router as v1_router
from app.core.config import settings
from app.scheduler.jobs import start_scheduler, stop_scheduler

_LOG_LEVEL = (
    logging.DEBUG
    if settings.debug
    else getattr(logging, settings.log_level, logging.INFO)
)
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
log = logging.getLogger(__name__)


def _seed_accounts_from_env() -> list[tuple[str, str, str, bool]]:
    """Resolve seed accounts strictly from settings — no hardcoded passwords.

    Returns a list of ``(email, password, display_name, is_admin)`` tuples
    for every seed slot whose ``email`` AND ``password`` are both non-empty
    in the environment. Empty password (the default) means "skip this
    seed entirely" — production deploys can simply leave the password
    vars unset and no demo accounts are created.
    """
    candidates: list[tuple[str, str, str, bool]] = [
        (settings.seed_guest_email, settings.seed_guest_password, "Guest Researcher", False),
        (settings.seed_admin_email, settings.seed_admin_password, "Admin",            True),
        (settings.seed_user_email,  settings.seed_user_password,  "Researcher",       False),
    ]
    return [(e.strip(), p, name, admin)
            for e, p, name, admin in candidates
            if e and e.strip() and p]


async def _ensure_seed_user() -> None:
    """Idempotently materialise the guest / admin / normal seed accounts.

    Runs on every boot (local *and* cloud — gated only on whether the
    operator configured ``SEED_*_PASSWORD`` env vars, blank-password
    means "skip this slot"). Designed to be a deterministic
    *self-heal* against partial state from prior boots: a previous
    crash that left only the guest, or a schema bug that rolled back
    half the inserts, gets repaired here.

    Three properties this routine guarantees post-commit, given the
    env is configured for all three slots:

    1. ``admin@…`` exists with ``is_admin = True``.
    2. ``test@…`` (guest) exists with ``is_admin = False``.
    3. ``user@…`` (normal) exists with ``is_admin = False``.

    Each seed account is created in its **own transaction** so a row
    failure for one account (e.g. an out-of-date provider_settings
    schema bringing one INSERT down) doesn't roll back the others —
    that exact bug is how an earlier install ended up with the guest
    promoted to admin and no separate admin account at all.

    A final *invariant pass* runs after all seed accounts have been
    upserted: it demotes every user whose email is NOT
    ``SEED_ADMIN_EMAIL``, and promotes ``SEED_ADMIN_EMAIL`` if it's
    present in the DB. This guarantees the admin invariant even when
    the legacy lockout-bootstrap promoted the wrong user on an
    earlier boot.

    Failure mode:
        Any single seed slot that crashes is logged and skipped; the
        remaining slots are still processed. The invariant pass is
        wrapped in its own try/except so it never blocks startup.
    """
    from sqlalchemy import select as _select
    from sqlalchemy import update as _update
    from app.db.session import async_session_factory
    from app.core.security import hash_password
    from app.models.user import User, UserProviderSettings, UserInterestProfile, ExpertiseLevel, Orientation
    from app.models.graph import NamespaceSubscription, SourceMapping

    _DEFAULT_NS = [
        ("cs.AI",  "arxiv_rss", "cs.AI"),
        ("cs.ML",  "arxiv_rss", "cs.LG"),
        ("cs.NLP", "arxiv_rss", "cs.CL"),
    ]

    seed_accounts = _seed_accounts_from_env()
    if not seed_accounts:
        log.info("seed accounts: none configured (set SEED_*_PASSWORD env vars to enable)")
        return

    # ── Per-account upsert. Each slot opens its own session/transaction
    # so a row-level failure (schema drift, FK conflict, etc.) cannot
    # bring down the rest of the seed batch.
    for email, password, display_name, is_admin in seed_accounts:
        try:
            async with async_session_factory() as db:
                row = await db.execute(_select(User).where(User.email == email))
                user = row.scalar_one_or_none()

                if not user:
                    user = User(
                        email=email,
                        hashed_password=hash_password(password),
                        display_name=display_name,
                        expertise_level=ExpertiseLevel.practitioner,
                        orientation=Orientation.both,
                        onboarding_complete=True,
                        is_admin=is_admin,
                    )
                    db.add(user)
                    await db.flush()
                    db.add(UserProviderSettings(user_id=user.id))
                    db.add(UserInterestProfile(user_id=user.id))
                    log.info("seed user created: %s (admin=%s)", email, is_admin)
                else:
                    # Sync the admin bit so a previously-misplaced admin
                    # flag is repaired automatically without intervention.
                    if user.is_admin != is_admin:
                        log.info("seed user admin bit corrected: %s %s → %s",
                                 email, user.is_admin, is_admin)
                        user.is_admin = is_admin
                    # Re-add provider_settings / interest_profile if a
                    # prior partial-rollback left them missing — both
                    # are required by downstream code paths.
                    ps_row = await db.execute(
                        _select(UserProviderSettings).where(UserProviderSettings.user_id == user.id)
                    )
                    if ps_row.scalar_one_or_none() is None:
                        db.add(UserProviderSettings(user_id=user.id))
                        log.info("seed user provider_settings repaired: %s", email)
                    ip_row = await db.execute(
                        _select(UserInterestProfile).where(UserInterestProfile.user_id == user.id)
                    )
                    if ip_row.scalar_one_or_none() is None:
                        db.add(UserInterestProfile(user_id=user.id))
                        log.info("seed user interest_profile repaired: %s", email)

                for ns_key, source_name, arxiv_cat in _DEFAULT_NS:
                    sub = await db.execute(
                        _select(NamespaceSubscription).where(
                            NamespaceSubscription.user_id == user.id,
                            NamespaceSubscription.namespace_key == ns_key,
                        )
                    )
                    if not sub.scalar_one_or_none():
                        db.add(NamespaceSubscription(user_id=user.id, namespace_key=ns_key))

                    mapping = await db.execute(
                        _select(SourceMapping).where(
                            SourceMapping.namespace_key == ns_key,
                            SourceMapping.source_name == source_name,
                        )
                    )
                    if not mapping.scalar_one_or_none():
                        db.add(SourceMapping(
                            namespace_key=ns_key,
                            source_name=source_name,
                            external_category_key=arxiv_cat,
                        ))

                await db.commit()
        except Exception as exc:
            log.warning("seed user upsert failed for %s: %s", email, exc)
            # Fall through to the next slot — other seeds should still
            # be created so we don't leave the install in a half-state.

    # ── Invariant pass: enforce admin precedence. Runs in its own
    # transaction; idempotent; protects against an earlier
    # lockout-bootstrap promotion that left guest/user as admin.
    admin_email = (settings.seed_admin_email or "").strip().lower()
    if not admin_email:
        return
    try:
        async with async_session_factory() as db:
            # 1) Demote any user with is_admin=True whose email is not
            #    the canonical admin email. This nukes accidental
            #    promotions across the board, not just for the seeded
            #    guest/user — operators who hand-promoted themselves
            #    via SQL should set SEED_ADMIN_EMAIL accordingly or
            #    flip is_admin AFTER startup via the admin panel.
            result = await db.execute(
                _update(User).where(
                    User.is_admin.is_(True),
                    User.email != admin_email,
                ).values(is_admin=False).returning(User.email)
            )
            demoted = [r[0] for r in result.fetchall()]
            if demoted:
                log.info("admin invariant: demoted non-canonical admin(s): %s", demoted)

            # 2) Promote the canonical admin if it exists and isn't admin.
            result = await db.execute(
                _update(User).where(
                    User.email == admin_email,
                    User.is_admin.is_(False),
                ).values(is_admin=True).returning(User.email)
            )
            promoted = [r[0] for r in result.fetchall()]
            if promoted:
                log.info("admin invariant: promoted canonical admin: %s", promoted)

            await db.commit()
    except Exception as exc:
        log.warning("admin invariant pass skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Startup: create DB tables (dev), start scheduler. Shutdown: stop scheduler."""
    log.info("ResearchFlow starting — environment=%s debug=%s", settings.environment, settings.debug)

    # Security guard: reject startup when JWT_SECRET is the insecure default.
    # A forged token gives attackers full access to every user account.
    _JWT_DEFAULT = "change-me-in-production"
    if settings.jwt_secret == _JWT_DEFAULT:
        if settings.environment != "local":
            # Hard abort — running with a forgeable JWT secret in production
            # is worse than being unavailable. The operator MUST set JWT_SECRET.
            raise RuntimeError(
                "FATAL: JWT_SECRET is set to the default insecure value in a "
                f"non-local environment ({settings.environment!r}). "
                "All tokens can be forged. "
                "Set JWT_SECRET to a cryptographically random 32+ char string "
                "and restart the service."
            )
        else:
            log.warning(
                "JWT_SECRET is set to the default insecure value. "
                "Set JWT_SECRET before deploying to staging or production."
            )

    if settings.environment == "local":
        from app.db.session import create_all_tables
        await create_all_tables()

    # Create search indexes idempotently — safe to run every startup.
    # GIN for full-text search, HNSW for vector similarity, composite for filtering.
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_papers_fts
                ON papers USING GIN (
                    to_tsvector('english',
                        COALESCE(title, '') || ' ' ||
                        COALESCE(tldr, '') || ' ' ||
                        COALESCE(abstract, '') || ' ' ||
                        COALESCE(array_to_string(key_concepts, ' '), '') || ' ' ||
                        COALESCE(array_to_string(methods_used, ' '), '')
                    )
                )
            """))
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_papers_external_id
                ON papers (external_id)
            """))
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_chunks_dim_provider
                ON paper_chunks (embedding_dim, embedding_provider)
            """))
            # HNSW index for fast approximate nearest-neighbour search.
            # Requires pgvector >= 0.5. Skipped silently if unavailable.
            try:
                await conn.execute(_text("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
                    ON paper_chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """))
            except Exception:
                pass  # pgvector HNSW not available — IVFFlat/exact scan fallback
        log.info("search indexes ensured")
    except Exception as exc:
        log.warning("search index creation skipped: %s", exc)

    # Enable pg_trgm for fuzzy keyword search (idempotent, silently skipped if unavailable)
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_papers_title_trgm
                ON papers USING GIN (title gin_trgm_ops)
            """))
        log.info("pg_trgm extension and trigram index ensured")
    except Exception as exc:
        log.warning("pg_trgm setup skipped (fuzzy search will degrade to ILIKE): %s", exc)

    # Schema migrations — ADD COLUMN IF NOT EXISTS is idempotent
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text("""
                ALTER TABLE idea_capsules
                ADD COLUMN IF NOT EXISTS source_mode VARCHAR(20) NOT NULL DEFAULT 'manual'
            """))
            await conn.execute(_text("""
                ALTER TABLE idea_capsules
                ADD COLUMN IF NOT EXISTS source_query TEXT
            """))
            # ``namespace_key`` stamped at capsule creation so the Genie
            # Ideas list can filter directly. Without it, combined ideas
            # whose seeds are other capsules (no Paper FK) leak across
            # namespaces — the list-filter previously had no signal to
            # discriminate them.
            await conn.execute(_text("""
                ALTER TABLE idea_capsules
                ADD COLUMN IF NOT EXISTS namespace_key VARCHAR(100)
            """))
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_idea_capsules_namespace
                ON idea_capsules (user_id, namespace_key)
            """))
            # One-shot backfill for rows created before the namespace_key
            # column shipped. Resolves the dominant namespace via two
            # parallel paths and writes back to the row so future list
            # queries hit the primary index instead of the seed-resolution
            # fallback. Idempotent — re-running picks up new rows without
            # touching already-stamped ones.
            #
            # Path A — paper-backed seeds (regular synthesized capsules):
            #   genie_elements.id ∈ seed_element_ids (a JSONB string array)
            #     ⟶ genie_elements.paper_id ⟶ papers.namespace_key
            # Path B — capsule-backed seeds (combined capsules):
            #   genie_elements.id ∈ seed_element_ids
            #     ⟶ genie_elements.idea_capsule_id ⟶ idea_capsules.namespace_key
            # Path A wins ties when both resolve. We take the modal
            # namespace per capsule via MIN() of grouped counts. Wrapped
            # in its own try/except so a malformed legacy row can't kill
            # the whole startup migration block.
            try:
                await conn.execute(_text("""
                    WITH seed_explode AS (
                        SELECT
                            c.id AS capsule_id,
                            (s.value #>> '{}')::uuid AS element_id
                        FROM idea_capsules c
                        CROSS JOIN LATERAL jsonb_array_elements(
                            COALESCE(c.seed_element_ids, '[]'::jsonb)
                        ) AS s(value)
                        WHERE c.namespace_key IS NULL
                    ),
                    paper_ns AS (
                        SELECT
                            se.capsule_id,
                            p.namespace_key,
                            COUNT(*) AS n
                        FROM seed_explode se
                        JOIN genie_elements ge ON ge.id = se.element_id
                        JOIN papers p ON p.id = ge.paper_id
                        WHERE p.namespace_key IS NOT NULL AND p.namespace_key <> ''
                        GROUP BY se.capsule_id, p.namespace_key
                    ),
                    capsule_ns AS (
                        SELECT
                            se.capsule_id,
                            parent.namespace_key,
                            COUNT(*) AS n
                        FROM seed_explode se
                        JOIN genie_elements ge ON ge.id = se.element_id
                        JOIN idea_capsules parent ON parent.id = ge.idea_capsule_id
                        WHERE parent.namespace_key IS NOT NULL AND parent.namespace_key <> ''
                        GROUP BY se.capsule_id, parent.namespace_key
                    ),
                    unified AS (
                        SELECT capsule_id, namespace_key, n FROM paper_ns
                        UNION ALL
                        SELECT capsule_id, namespace_key, n FROM capsule_ns
                    ),
                    dominant AS (
                        SELECT DISTINCT ON (capsule_id)
                            capsule_id,
                            namespace_key
                        FROM unified
                        ORDER BY capsule_id, n DESC, namespace_key
                    )
                    UPDATE idea_capsules ic
                    SET namespace_key = d.namespace_key
                    FROM dominant d
                    WHERE ic.id = d.capsule_id
                      AND ic.namespace_key IS NULL
                """))
                log.info("idea_capsules namespace_key backfill complete")
            except Exception as exc:
                log.warning("idea_capsules namespace_key backfill skipped: %s", exc)
        log.info("idea_capsules schema migration complete")
    except Exception as exc:
        log.warning("idea_capsules migration skipped: %s", exc)

    # ── Admin schema: is_admin column + app_settings table ────────────────
    # Both are idempotent so re-running on every startup is safe.
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE"
            ))
            await conn.execute(_text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS feature_overrides JSONB NOT NULL DEFAULT '{}'::jsonb"
            ))
            # RBAC scaffolding — ``role`` is forward-compatible (today
            # everyone reads ``is_admin``; the string column lets us
            # introduce editor / reviewer / etc. roles without another
            # migration). ``tier_slug`` joins to ``tiers.slug`` when
            # subscriptions ship. NULL today, fully optional.
            await conn.execute(_text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(40) NOT NULL DEFAULT 'member'"
            ))
            await conn.execute(_text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS tier_slug VARCHAR(40)"
            ))
            await conn.execute(_text(
                "CREATE INDEX IF NOT EXISTS idx_users_tier_slug ON users (tier_slug)"
            ))
            # ``UserProviderSettings.encrypted_wolfram_key`` was added to
            # the model after the table was first created in older installs.
            # An INSERT from /admin/users → create_user fails with
            # ``UndefinedColumnError`` until we backfill the column.
            # Idempotent — re-running on already-migrated installs is a no-op.
            await conn.execute(_text(
                "ALTER TABLE user_provider_settings ADD COLUMN IF NOT EXISTS encrypted_wolfram_key TEXT"
            ))
            # Admin precedence: the seeded admin account (configured via
            # SEED_ADMIN_EMAIL) is the canonical admin. Any user that was
            # previously promoted via the legacy earliest-user bootstrap
            # gets *demoted* here as long as the seeded admin exists,
            # otherwise the guest account ends up with admin powers it
            # shouldn't have.
            #
            # The seeded admin itself is (re)promoted in ``_ensure_seed_user``
            # — this query only fixes up users that were accidentally
            # promoted by the old bootstrap.
            admin_email = (settings.seed_admin_email or "").strip()
            if admin_email:
                await conn.execute(
                    _text(
                        """
                        UPDATE users
                        SET is_admin = FALSE
                        WHERE is_admin = TRUE
                          AND email <> :admin_email
                          AND email IN (:guest_email, :user_email)
                        """
                    ),
                    {
                        "admin_email": admin_email,
                        "guest_email": (settings.seed_guest_email or "").strip(),
                        "user_email": (settings.seed_user_email or "").strip(),
                    },
                )
            # Lockout safety: if there is genuinely NO admin in the DB
            # (e.g. a fresh boot where the seeded admin password is blank
            # and no human has signed up yet), promote the earliest user
            # so the install is not bricked. Once anyone becomes admin
            # this clause is a no-op.
            await conn.execute(_text(
                """
                UPDATE users SET is_admin = TRUE
                WHERE id = (
                    SELECT id FROM users
                    WHERE is_admin = FALSE
                    ORDER BY created_at ASC
                    LIMIT 1
                )
                AND NOT EXISTS (SELECT 1 FROM users WHERE is_admin = TRUE)
                """
            ))
        log.info("admin schema migration complete (is_admin column + bootstrap)")
    except Exception as exc:
        log.warning("admin schema migration skipped: %s", exc)

    # Ensure generated_artifacts table exists (idempotent via create_all on local)
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_generated_artifacts_user_source
                ON generated_artifacts (user_id, source_id, generation_type)
            """))
            # Composite index for cache-lookup query (user, source, type, status, created_at)
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_generated_artifacts_lookup
                ON generated_artifacts (user_id, source_id, generation_type, status, created_at DESC)
            """))
        log.info("generated_artifacts indexes ensured")
    except Exception as exc:
        log.warning("generated_artifacts index creation skipped: %s", exc)

    # Index for WorkflowRepository.should_run() — called on every nightly ingestion
    # to guard idempotency.  Without this, the query scans the whole workflow_runs
    # table; with this index it resolves the (name, scope, date, status) lookup
    # in O(log n).
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_workflow_runs_idempotency
                ON workflow_runs (workflow_name, scope_key, run_date, status)
            """))
        log.info("workflow_runs idempotency index ensured")
    except Exception as exc:
        log.warning("workflow_runs index creation skipped: %s", exc)

    # Add parser metadata columns to papers (idempotent)
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text("""
                ALTER TABLE papers ADD COLUMN IF NOT EXISTS parser_used VARCHAR(50)
            """))
            await conn.execute(_text("""
                ALTER TABLE papers ADD COLUMN IF NOT EXISTS parser_fallback_used BOOLEAN DEFAULT FALSE
            """))
            await conn.execute(_text("""
                ALTER TABLE papers ADD COLUMN IF NOT EXISTS parse_duration_ms INTEGER
            """))
            await conn.execute(_text("""
                ALTER TABLE papers ADD COLUMN IF NOT EXISTS parser_confidence DOUBLE PRECISION
            """))
        log.info("papers parser-metadata columns ensured")
    except Exception as exc:
        log.warning("papers parser-metadata migration skipped: %s", exc)

    # Add is_manually_imported flag to papers (idempotent)
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text("""
                ALTER TABLE papers
                ADD COLUMN IF NOT EXISTS is_manually_imported BOOLEAN NOT NULL DEFAULT FALSE
            """))
        log.info("papers.is_manually_imported column ensured")
    except Exception as exc:
        log.warning("papers.is_manually_imported migration skipped: %s", exc)

    # Create paper_namespace_hides table (idempotent) — stores per-user, per-namespace hide state
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text("""
                CREATE TABLE IF NOT EXISTS paper_namespace_hides (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                    namespace_key VARCHAR(100) NOT NULL,
                    hidden_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_paper_hide UNIQUE (user_id, paper_id, namespace_key)
                )
            """))
            await conn.execute(_text("""
                CREATE INDEX IF NOT EXISTS idx_paper_hides_user_ns
                ON paper_namespace_hides (user_id, namespace_key)
            """))
        log.info("paper_namespace_hides table ensured")
    except Exception as exc:
        log.warning("paper_namespace_hides table creation skipped: %s", exc)

    # Materialise the configured seed accounts (guest / admin / normal).
    # Runs on every boot, local and cloud — gated only on whether the
    # operator set ``SEED_*_PASSWORD`` env vars (blank password = skip
    # that slot, which is the right production default for unused
    # demo accounts). Idempotent and self-healing — see the docstring.
    try:
        await _ensure_seed_user()
    except Exception as exc:
        log.warning("seed user creation skipped: %s", exc)

    # Initialise the LangGraph PostgreSQL checkpoint store so the tables
    # exist before any workflow runs. The checkpointer is a module-level
    # singleton that slides and podcast workflows share.
    try:
        from app.db.checkpointer import get_checkpointer
        await get_checkpointer()
        log.info("LangGraph checkpoint store ready")
    except Exception as exc:
        log.warning("LangGraph checkpoint store init failed (workflows will run without checkpointing): %s", exc)

    # Add created_at to langgraph_checkpoints if it was created before this column existed.
    try:
        from app.db.session import engine
        from sqlalchemy import text as _text
        async with engine.begin() as conn:
            await conn.execute(_text("""
                ALTER TABLE langgraph_checkpoints
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            """))
        log.info("langgraph_checkpoints.created_at column ensured")
    except Exception as exc:
        log.warning("langgraph_checkpoints migration skipped: %s", exc)

    # Sweep any GeneratedArtifact rows left in ``running``/``queued`` state by
    # a previous worker crash. With checkpointing active, jobs that have partial
    # state are re-dispatched from the last completed node rather than restarted
    # from scratch — preventing token waste on already-completed LLM calls.
    try:
        from app.workflows._generation_runtime import recover_orphaned_artifacts
        recovered = await recover_orphaned_artifacts()
        if recovered:
            log.info("startup recovery: %d orphaned generation job(s) processed", recovered)
    except Exception as exc:
        log.warning("startup recovery skipped: %s", exc)

    # Reconcile any AssistantTask rows left as running/pending by a previous
    # worker crash. Recent + cancellable tasks are re-submitted (the
    # orchestrator's idempotent step replay skips completed steps); stale or
    # too-old tasks are marked failed so the UI doesn't show forever-spinners.
    try:
        # Importing the service registers the orchestrator with the scheduler,
        # which reconcile_orphans needs.
        import app.services.research_assistant  # noqa: F401
        from app.assistant.recovery import reconcile_orphans
        recovery_counts = await reconcile_orphans()
        if any(recovery_counts.values()):
            log.info(
                "assistant recovery: resumed=%d failed=%d cancelled=%d",
                recovery_counts["resumed"],
                recovery_counts["failed"],
                recovery_counts["cancelled"],
            )
    except Exception as exc:
        log.warning("assistant recovery skipped: %s", exc)

    start_scheduler()
    log.info("scheduler started")

    yield

    # ── Graceful shutdown ─────────────────────────────────────────────────
    # Stop accepting new scheduled jobs first, then drain pool/sockets so
    # we never leak connections back to the OS on container exit.
    try:
        stop_scheduler()
    except Exception as exc:
        log.warning("scheduler shutdown error: %s", exc)

    # Dispose the LangGraph checkpoint pool (asyncpg) so connections are
    # returned to the server cleanly. Skipped silently if the checkpointer
    # was never initialised — startup may have logged a warning above.
    try:
        from app.db.checkpointer import _checkpointer as _maybe_checkpointer
        if _maybe_checkpointer is not None:
            await _maybe_checkpointer.close()
            log.info("LangGraph checkpoint pool closed")
    except Exception as exc:
        log.warning("checkpoint pool close error: %s", exc)

    # Dispose the SQLAlchemy async engine — releases the connection pool
    # back to PostgreSQL. Without this, Azure PostgreSQL leaves the pool's
    # connections in the "idle in transaction" state for up to 30 min,
    # consuming server-side max_connections and producing intermittent
    # "remaining connection slots are reserved" errors on rolling deploys.
    try:
        from app.db.session import engine as _engine
        await _engine.dispose()
        log.info("SQLAlchemy engine disposed")
    except Exception as exc:
        log.warning("engine dispose error: %s", exc)

    log.info("ResearchFlow shutting down")


app = FastAPI(
    title="ResearchFlow API",
    description="AI-native research intelligence platform",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing + request-ID middleware ─────────────────────────────────────
@app.middleware("http")
async def request_instrumentation(request: Request, call_next):
    req_id = str(_uuid.uuid4())[:8]
    request.state.request_id = req_id
    t0 = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Request-Id"] = req_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

    log_fn = log.debug if settings.debug else log.info
    log_fn(
        "%s %s → %d  %sms  req_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        req_id,
    )
    if settings.debug and request.query_params:
        log.debug("  query=%s", dict(request.query_params))

    return response


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(v1_router)

# ── Static files (blob storage in local dev) ──────────────────────────────────
blob_dir = settings.blob_local_dir
os.makedirs(blob_dir, exist_ok=True)
app.mount("/blobs", StaticFiles(directory=blob_dir, check_dir=False), name="blobs")


# ── Global exception handler — never leak internals ───────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(
        "unhandled exception path=%s req_id=%s err=%s",
        request.url.path,
        getattr(request.state, "request_id", "?"),
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Something went wrong. We're on it.",
            "action": "Try again or contact support.",
        },
    )


# ── Health + debug endpoints ───────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Liveness probe — always returns 200 so the process is known to be up.

    For a deeper readiness check (DB connectivity) call ``/health/ready``.
    """
    return {"status": "ok", "environment": settings.environment}


@app.get("/health/ready")
async def health_ready():
    """Readiness probe — verifies DB connectivity before accepting traffic.

    Returns 200 when the database is reachable, 503 otherwise.
    Suitable for Kubernetes readinessProbe / load-balancer health checks.
    """
    from fastapi.responses import JSONResponse as _JSONResponse
    from app.db.session import engine
    from sqlalchemy import text as _text

    try:
        async with engine.connect() as conn:
            await conn.execute(_text("SELECT 1"))
        return {"status": "ready", "db": "ok"}
    except Exception as exc:
        # Log the full exception server-side (may contain connection strings).
        # Never surface raw exception text to callers — it can expose credentials.
        log.error("health_ready: DB connectivity check failed — %s", exc)
        return _JSONResponse(
            status_code=503,
            content={"status": "not_ready", "db": "unavailable"},
        )


@app.get("/debug/status", include_in_schema=settings.debug)
async def debug_status():
    """Runtime config snapshot — only accessible when DEBUG=true."""
    if not settings.debug:
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    def _mask(val: str) -> str:
        return val[:4] + "…" if len(val) > 8 else ("set" if val else "")

    return {
        "environment": settings.environment,
        "log_level": settings.log_level,
        "debug": settings.debug,
        "database_url": settings.database_url.split("@")[-1],  # hide credentials
        "cache_backend": settings.cache_backend,
        "blob_backend": settings.blob_backend,
        "ingestion_mode": settings.ingestion_mode,
        "default_llm_provider": settings.default_llm_provider,
        "default_embedding_provider": settings.default_embedding_provider,
        "default_embedding_model": settings.default_embedding_model,
        "openai_api_key": _mask(settings.openai_api_key),
        "google_api_key": _mask(settings.google_api_key),
        "anthropic_api_key": _mask(settings.anthropic_api_key),
        "langsmith_tracing": bool(settings.langsmith_api_key),
        "breakthrough_threshold": settings.breakthrough_threshold,
        "ingestion_cron": settings.ingestion_cron,
    }
