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


_GUEST_EMAIL = "test@researchflow.ai"
_GUEST_PASSWORD = "ResearchFlow2024!"


async def _ensure_seed_user() -> None:
    """Idempotently create the guest/test user and its SourceMappings on startup."""
    from sqlalchemy import select as _select
    from app.db.session import async_session_factory
    from app.core.security import hash_password
    from app.models.user import User, UserProviderSettings, UserInterestProfile, ExpertiseLevel, Orientation
    from app.models.graph import NamespaceSubscription, SourceMapping

    _DEFAULT_NS = [
        ("cs.AI",  "arxiv_rss", "cs.AI"),
        ("cs.ML",  "arxiv_rss", "cs.LG"),
        ("cs.NLP", "arxiv_rss", "cs.CL"),
    ]

    async with async_session_factory() as db:
        row = await db.execute(_select(User).where(User.email == _GUEST_EMAIL))
        user = row.scalar_one_or_none()

        if not user:
            user = User(
                email=_GUEST_EMAIL,
                hashed_password=hash_password(_GUEST_PASSWORD),
                display_name="Guest Researcher",
                expertise_level=ExpertiseLevel.practitioner,
                orientation=Orientation.both,
                onboarding_complete=True,
            )
            db.add(user)
            await db.flush()
            db.add(UserProviderSettings(user_id=user.id))
            db.add(UserInterestProfile(user_id=user.id))
            log.info("seed user created: %s", _GUEST_EMAIL)

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
        log.info("idea_capsules schema migration complete")
    except Exception as exc:
        log.warning("idea_capsules migration skipped: %s", exc)

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

    # Ensure the guest/test user always exists in local dev — idempotent
    if settings.environment == "local":
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

    stop_scheduler()
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
