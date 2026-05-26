"""Settings router — provider config, topic subscriptions, notifications, profile."""

from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm.attributes import flag_modified

from app.core.deps import CurrentUserID, DBSession
from app.models.assistant import AssistantSession, MemoryRevision
from app.models.workflow import TokenUsage
from app.repositories.user import UserRepository
from app.schemas import (
    NotificationSettingsRequest,
    OnboardingRequest,
    ProviderSettingsRequest,
)
from app.services.namespace import NAMESPACE_TO_ARXIV, NamespaceManager

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/subjects")
async def get_subjects():
    """Return available subjects and topics for onboarding/settings."""
    ns = NamespaceManager()
    return {"subjects": ns.subject_topics()}


@router.get("/namespaces")
async def get_namespaces():
    """Return all available namespace keys."""
    return {"namespaces": sorted(NAMESPACE_TO_ARXIV.keys())}


@router.post("/onboarding", status_code=200)
async def complete_onboarding(body: OnboardingRequest, user_id: CurrentUserID, db: DBSession):
    """Complete onboarding: save topics, expertise, notifications."""
    repo = UserRepository(db)
    ns_manager = NamespaceManager()

    namespace_keys: list[str] = []
    for topic_str in body.topics:
        if topic_str in NAMESPACE_TO_ARXIV:
            # Already a valid namespace key (new onboarding format)
            namespace_keys.append(topic_str)
        elif ":" in topic_str:
            # Legacy "Subject:Topic" format
            subject, topic = topic_str.split(":", 1)
            ns = ns_manager.resolve(subject, topic)
            if ns:
                namespace_keys.append(ns)

    await repo.set_namespace_subscriptions(user_id, namespace_keys)

    user = await repo.get_by_id(user_id)
    if user:
        user.expertise_level = body.expertise_level
        user.orientation = body.orientation
        user.notify_potd = body.notify_potd
        user.notify_digest = body.notify_digest
        user.notify_breakthrough = body.notify_breakthrough
        user.onboarding_complete = True

    # Seed SourceMappings for each namespace
    from app.models.graph import SourceMapping
    from sqlalchemy import select
    for ns_key in namespace_keys:
        existing = await db.execute(
            select(SourceMapping).where(SourceMapping.namespace_key == ns_key)
        )
        if not existing.scalar_one_or_none():
            arxiv_cat = ns_manager.arxiv_category(ns_key) or ns_key
            db.add(SourceMapping(
                namespace_key=ns_key,
                source_name="arxiv_rss",
                external_category_key=arxiv_cat,
            ))

    await db.commit()
    return {"namespaces": namespace_keys}


@router.get("/profile")
async def get_profile(user_id: CurrentUserID, db: DBSession):
    """Return the current user's display name, expertise level, and orientation.

    Args:
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        A dict with ``display_name``, ``expertise_level``, and
        ``orientation`` values, or an empty dict if the user is not found.
    """
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if not user:
        return {}
    return {
        "display_name": user.display_name,
        "expertise_level": user.expertise_level.value,
        "orientation": user.orientation.value,
    }


class ProfileUpdateRequest(BaseModel):
    """Request body for PATCH /settings/profile."""

    display_name: str | None = None
    expertise_level: str | None = None
    orientation: str | None = None


@router.patch("/profile", status_code=200)
async def update_profile(body: ProfileUpdateRequest, user_id: CurrentUserID, db: DBSession):
    """Update the current user's display name, expertise level, or orientation.

    Only fields present in the request body are updated; ``None`` values are
    ignored.

    Args:
        body: Partial profile update — all fields optional.
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        ``{"ok": True}`` on success.
    """
    from app.models.user import ExpertiseLevel, Orientation
    from fastapi import HTTPException
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user:
        if body.display_name is not None:
            user.display_name = body.display_name.strip() or user.display_name
        if body.expertise_level is not None:
            try:
                user.expertise_level = ExpertiseLevel(body.expertise_level)
            except ValueError:
                raise HTTPException(status_code=422, detail=f"Invalid expertise_level: '{body.expertise_level}'")
        if body.orientation is not None:
            try:
                user.orientation = Orientation(body.orientation)
            except ValueError:
                raise HTTPException(status_code=422, detail=f"Invalid orientation: '{body.orientation}'")
        await db.commit()
    return {"ok": True}


@router.get("/provider")
async def get_provider_settings(user_id: CurrentUserID, db: DBSession):
    """Return the effective LLM and embedding provider configuration.

    Always returns what the backend is actually running with:

    - User's saved DB row when present.
    - System defaults from config / .env.local otherwise.
    - Embedding provider reflects the runtime fallback (e.g. if Gemini is
      configured but no Google key is set, the actual OpenAI fallback is
      returned) so the UI never shows a provider that isn't being used.
    """
    from app.core.config import settings as _cfg
    from app.adapters.embedding import resolve_embedding_provider

    repo = UserRepository(db)
    s = await repo.get_provider_settings(user_id)

    # Determine the embedding provider and model that will actually run,
    # accounting for missing API keys and the fallback chain.
    preferred_embed = s.embedding_provider.value if s else _cfg.default_embedding_provider
    effective_embed_provider, effective_embed_model = resolve_embedding_provider(preferred_embed)
    # If the user explicitly saved a model for the effective provider, respect it.
    if s and s.embedding_provider.value == effective_embed_provider:
        effective_embed_model = s.embedding_model

    return {
        "llm_provider":       s.llm_provider.value  if s else _cfg.default_llm_provider,
        "cheap_model":        s.cheap_model          if s else _cfg.default_cheap_model,
        "quality_model":      s.quality_model        if s else _cfg.default_quality_model,
        "reasoning_model":    s.reasoning_model      if s else _cfg.default_reasoning_model,
        "embedding_provider": effective_embed_provider,
        "embedding_model":    effective_embed_model,
    }


@router.patch("/provider", status_code=200)
async def update_provider_settings(
    body: ProviderSettingsRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Update the user's LLM or embedding provider settings.

    Only non-None fields in the request body are applied.

    Args:
        body: Partial provider settings update.
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        A dict with ``updated`` listing the field names that were changed.
    """
    repo = UserRepository(db)
    updates = body.model_dump(exclude_none=True)
    await repo.update_provider_settings(user_id, updates)
    await db.commit()
    return {"updated": list(updates.keys())}


class ApiKeysRequest(BaseModel):
    """Request body for PATCH /settings/api-keys."""

    openai_key: str | None = None
    anthropic_key: str | None = None
    google_key: str | None = None
    wolfram_key: str | None = None


@router.get("/api-keys")
async def get_api_keys(user_id: CurrentUserID, db: DBSession):
    """Return masked API key status — which keys are set from env vs user override.

    Reads keys from every plausible source on each call so the UI never
    shows a stale "not set" for a deployment that exported its env vars
    after the Python process started:

      1. The cached ``Settings`` singleton (.env / .env.local at boot).
      2. A fresh ``Settings()`` instance — picks up env vars exported
         AFTER the process began (e.g. `export OPENAI_API_KEY=… ; uvicorn`
         where the export happened in the same shell session).
      3. Direct ``os.environ`` lookup as a final fallback.
    """
    import os

    from app.core.config import Settings as _SettingsCls
    from app.core.config import settings as _cfg

    repo = UserRepository(db)
    ps = await repo.get_provider_settings(user_id)

    def _mask(val: str | None) -> str:
        if not val:
            return ""
        visible = val[:4]
        return visible + "•" * min(len(val) - 4, 24)

    # Build a fresh Settings instance so newly-exported env vars get picked
    # up without restarting the process. Falls back silently on validation
    # error.
    try:
        live_cfg = _SettingsCls()
    except Exception:
        live_cfg = _cfg

    def _resolve(*candidates: str | None) -> str:
        for c in candidates:
            if c:
                return c
        return ""

    def _from_env(*names: str) -> str:
        for n in names:
            v = os.environ.get(n)
            if v:
                return v
        return ""

    env_keys = {
        "openai":    _resolve(_cfg.openai_api_key,    live_cfg.openai_api_key,    _from_env("OPENAI_API_KEY")),
        "anthropic": _resolve(_cfg.anthropic_api_key, live_cfg.anthropic_api_key, _from_env("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")),
        "google":    _resolve(_cfg.google_api_key,    live_cfg.google_api_key,    _from_env("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY")),
        "wolfram":   _resolve(_cfg.wolfram_alpha_app_id, live_cfg.wolfram_alpha_app_id, _from_env("WOLFRAM_ALPHA_APP_ID", "WOLFRAM_APP_ID")),
    }
    user_keys = {
        "openai":    ps.encrypted_openai_key    if ps else None,
        "anthropic": ps.encrypted_anthropic_key if ps else None,
        "google":    ps.encrypted_google_key    if ps else None,
        "wolfram":   ps.encrypted_wolfram_key   if ps else None,
    }

    result = {}
    for provider in ("openai", "anthropic", "google", "wolfram"):
        env_val  = env_keys[provider] or ""
        user_val = user_keys[provider] or ""
        is_overridden = bool(user_val)
        active_val    = user_val if is_overridden else env_val
        result[provider] = {
            "is_set":        bool(active_val),
            "from_env":      bool(env_val and not is_overridden),
            "is_overridden": is_overridden,
            "masked":        _mask(active_val),
        }
    return result


@router.patch("/api-keys", status_code=200)
async def update_api_keys(body: ApiKeysRequest, user_id: CurrentUserID, db: DBSession):
    """Save or clear user-supplied API key overrides (stored in provider settings)."""
    repo = UserRepository(db)
    updates: dict = {}
    if body.openai_key is not None:
        updates["encrypted_openai_key"]    = body.openai_key    or None
    if body.anthropic_key is not None:
        updates["encrypted_anthropic_key"] = body.anthropic_key or None
    if body.google_key is not None:
        updates["encrypted_google_key"]    = body.google_key    or None
    if body.wolfram_key is not None:
        updates["encrypted_wolfram_key"]   = body.wolfram_key   or None
    if updates:
        await repo.update_provider_settings(user_id, updates)
        await db.commit()
    return {"ok": True}


@router.get("/notifications")
async def get_notifications(user_id: CurrentUserID, db: DBSession):
    """Return the current user's notification preferences.

    Args:
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        A dict with ``notify_potd``, ``notify_digest``, and
        ``notify_breakthrough`` boolean flags, or an empty dict if not found.
    """
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if not user:
        return {}
    return {
        "notify_potd": user.notify_potd,
        "notify_digest": user.notify_digest,
        "notify_breakthrough": user.notify_breakthrough,
    }


@router.patch("/notifications", status_code=200)
async def update_notifications(
    body: NotificationSettingsRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Update the current user's notification preferences.

    Only non-None fields in the request body are applied.

    Args:
        body: Partial notification settings update.
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        ``{"ok": True}`` on success.
    """
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)
    if user:
        if body.notify_potd is not None:
            user.notify_potd = body.notify_potd
        if body.notify_digest is not None:
            user.notify_digest = body.notify_digest
        if body.notify_breakthrough is not None:
            user.notify_breakthrough = body.notify_breakthrough
        await db.commit()
    return {"ok": True}


@router.get("/subscriptions")
async def get_subscriptions(user_id: CurrentUserID, db: DBSession):
    """Return the namespace keys the current user is subscribed to.

    Args:
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        A dict with a ``namespaces`` list of namespace key strings.
    """
    repo = UserRepository(db)
    namespaces = await repo.get_namespace_subscriptions(user_id)
    return {"namespaces": namespaces}


class SubscriptionsUpdateRequest(BaseModel):
    """Request body for PATCH /settings/subscriptions."""

    namespace_keys: list[str]


@router.patch("/subscriptions", status_code=200)
async def update_subscriptions(
    body: SubscriptionsUpdateRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Replace user's namespace subscriptions and seed SourceMappings."""
    ns_manager = NamespaceManager()
    repo = UserRepository(db)

    # Filter to only valid namespaces
    valid = [ns for ns in body.namespace_keys if ns in NAMESPACE_TO_ARXIV]
    await repo.set_namespace_subscriptions(user_id, valid)

    # Seed SourceMappings for any new namespaces
    from app.models.graph import SourceMapping
    from sqlalchemy import select
    for ns_key in valid:
        existing = await db.execute(
            select(SourceMapping).where(SourceMapping.namespace_key == ns_key)
        )
        if not existing.scalar_one_or_none():
            arxiv_cat = ns_manager.arxiv_category(ns_key) or ns_key
            db.add(SourceMapping(
                namespace_key=ns_key,
                source_name="arxiv_rss",
                external_category_key=arxiv_cat,
            ))

    await db.commit()
    return {"namespaces": valid}


# ── Token usage dashboard ─────────────────────────────────────────────────────


def _parse_date(s: str | None) -> datetime | None:
    """Parse YYYY-MM-DD into a UTC midnight datetime; return None on bad input."""
    if not s:
        return None
    try:
        return datetime.combine(date.fromisoformat(s), datetime.min.time(), tzinfo=timezone.utc)
    except ValueError:
        return None


@router.get("/token-usage")
async def get_token_usage(
    user_id: CurrentUserID,
    db: DBSession,
    date_from: str | None = Query(default=None, alias="from", description="YYYY-MM-DD inclusive (UTC)"),
    date_to:   str | None = Query(default=None, alias="to",   description="YYYY-MM-DD inclusive (UTC)"),
):
    """Return aggregated LLM token usage for the authenticated user.

    When neither ``from`` nor ``to`` is supplied, defaults to the current UTC
    day. Otherwise both ends are inclusive and span ``[from 00:00 UTC, to 24:00 UTC)``.

    Returns:
        A dict with:
          * ``range``: ``{from, to}`` echoing the resolved window
          * ``totals``: ``{input_tokens, output_tokens, total_tokens, cost_usd, calls}``
          * ``by_day``:    list of ``{date, input_tokens, output_tokens, total_tokens, cost_usd}``
          * ``by_workflow``: list of ``{workflow, input_tokens, output_tokens, total_tokens, calls}``
          * ``by_model``:    list of ``{provider, model, input_tokens, output_tokens, total_tokens, calls}``
    """
    # Resolve date range — default to today (UTC)
    today_utc = datetime.now(timezone.utc).date()
    start = _parse_date(date_from) or datetime.combine(today_utc, datetime.min.time(), tzinfo=timezone.utc)
    end_day = _parse_date(date_to)
    end = (end_day + timedelta(days=1)) if end_day else (start + timedelta(days=1))
    if end <= start:
        end = start + timedelta(days=1)

    base = (
        select(TokenUsage)
        .where(
            TokenUsage.user_id == user_id,
            TokenUsage.created_at >= start,
            TokenUsage.created_at <  end,
        )
    )

    # Totals
    totals_row = (await db.execute(
        select(
            func.coalesce(func.sum(TokenUsage.input_tokens),  0).label("inp"),
            func.coalesce(func.sum(TokenUsage.output_tokens), 0).label("out"),
            func.coalesce(func.sum(TokenUsage.cost_usd),      0.0).label("cost"),
            func.count().label("calls"),
        ).where(
            TokenUsage.user_id == user_id,
            TokenUsage.created_at >= start,
            TokenUsage.created_at <  end,
        )
    )).one()

    # Per-day breakdown (UTC days)
    day_col = func.date_trunc("day", TokenUsage.created_at).label("day")
    by_day_rows = (await db.execute(
        select(
            day_col,
            func.coalesce(func.sum(TokenUsage.input_tokens),  0),
            func.coalesce(func.sum(TokenUsage.output_tokens), 0),
            func.coalesce(func.sum(TokenUsage.cost_usd),      0.0),
        ).where(
            TokenUsage.user_id == user_id,
            TokenUsage.created_at >= start,
            TokenUsage.created_at <  end,
        ).group_by(day_col).order_by(day_col)
    )).all()

    # Per-workflow breakdown
    by_wf_rows = (await db.execute(
        select(
            func.coalesce(TokenUsage.workflow, "").label("wf"),
            func.coalesce(func.sum(TokenUsage.input_tokens),  0),
            func.coalesce(func.sum(TokenUsage.output_tokens), 0),
            func.count().label("calls"),
        ).where(
            TokenUsage.user_id == user_id,
            TokenUsage.created_at >= start,
            TokenUsage.created_at <  end,
        ).group_by(TokenUsage.workflow)
        .order_by(func.sum(TokenUsage.input_tokens + TokenUsage.output_tokens).desc())
    )).all()

    # Per-model breakdown
    by_model_rows = (await db.execute(
        select(
            TokenUsage.provider,
            TokenUsage.model,
            func.coalesce(func.sum(TokenUsage.input_tokens),  0),
            func.coalesce(func.sum(TokenUsage.output_tokens), 0),
            func.count().label("calls"),
        ).where(
            TokenUsage.user_id == user_id,
            TokenUsage.created_at >= start,
            TokenUsage.created_at <  end,
        ).group_by(TokenUsage.provider, TokenUsage.model)
        .order_by(func.sum(TokenUsage.input_tokens + TokenUsage.output_tokens).desc())
    )).all()

    inp = int(totals_row.inp or 0)
    out = int(totals_row.out or 0)

    return {
        "range": {
            "from": start.date().isoformat(),
            "to":   (end - timedelta(days=1)).date().isoformat(),
        },
        "totals": {
            "input_tokens":  inp,
            "output_tokens": out,
            "total_tokens":  inp + out,
            "cost_usd":      round(float(totals_row.cost or 0.0), 4),
            "calls":         int(totals_row.calls or 0),
        },
        "by_day": [
            {
                "date":          (d.date() if hasattr(d, "date") else d).isoformat(),
                "input_tokens":  int(i or 0),
                "output_tokens": int(o or 0),
                "total_tokens":  int((i or 0) + (o or 0)),
                "cost_usd":      round(float(c or 0.0), 4),
            }
            for d, i, o, c in by_day_rows
        ],
        "by_workflow": [
            # Empty workflow label means a background pass that didn't tag its
            # context (e.g. legacy news-ingest jobs). Calling it "Unknown" was
            # confusing; "Background" is honest and matches what these rows
            # actually represent.
            {
                "workflow":      wf or "Background",
                "input_tokens":  int(i or 0),
                "output_tokens": int(o or 0),
                "total_tokens":  int((i or 0) + (o or 0)),
                "calls":         int(calls or 0),
            }
            for wf, i, o, calls in by_wf_rows
        ],
        "by_model": [
            {
                "provider":      prov,
                "model":         mdl,
                "input_tokens":  int(i or 0),
                "output_tokens": int(o or 0),
                "total_tokens":  int((i or 0) + (o or 0)),
                "calls":         int(calls or 0),
            }
            for prov, mdl, i, o, calls in by_model_rows
        ],
    }


# ── Long-term memory inspection & management ────────────────────────────────
#
# Users explicitly asked for transparency and control over what RA
# remembers across turns. The memory tiers RA writes into are:
#
#   * ``short``  — chat-scoped, stored in the current session's state.
#                  Auto-evicted on session end; intentionally NOT
#                  exposed here (it's not "long-term").
#   * ``medium`` — session-tree-scoped, stored in the *root* session's
#                  state. Survives branches.
#   * ``long``   — namespace-scoped, stored in the root session's state
#                  keyed by namespace.
#
# The inspect API surfaces medium + long tiers (the two persistent
# ones). All endpoints are user-scoped — the root session lookup
# filters by ``user_id``, so a user can never read or mutate another
# user's memory even if they guess a session id.
#
# Memory layout in ``AssistantSession.state``:
#   {
#     "tree_memory": { "<key>": {"value": ..., "type": ..., "ts": ..., ...}, ... },
#     "ns_memory":   { "<namespace>": {"<key>": {...}, ...}, ... },
#     ...
#   }


_MEMORY_TIER_LABELS = {
    "medium": "tree_memory",
    "long":   "ns_memory",
}


def _memory_entry_to_row(
    *,
    tier: str,
    namespace_key: str,
    key: str,
    entry: Any,
) -> dict[str, Any]:
    """Project a stored memory entry into the flat shape the UI renders.

    Carries the cognitive-class label (``semantic`` / ``episodic`` /
    ``procedural`` / ``preference`` / ``-``) derived from the entry
    type so the UI can group by the three-way memory taxonomy without
    re-deriving it client-side. Subject and topic are split from the
    namespace key for the same reason.

    Tolerant of legacy str-only entries (returned as ``{"value": ...,
    "type": "context"}``) so a long-lived dataset with mixed shapes
    surfaces cleanly without backfill.
    """
    from app.assistant.memory_revisions import derive_entry_status, split_subject_topic
    from app.assistant.tools.memory import memory_category

    subject, topic = split_subject_topic(namespace_key)
    if isinstance(entry, dict):
        entry_type = str(entry.get("type", "context"))
        # Status: ``superseded_by_key`` set by the supersession
        # detector overrides the freshness derivation. Surfacing it
        # in the projection means the UI can show the chain and the
        # planner (via _matches in memory.py) keeps filtering them
        # out of recall — both views stay in sync.
        if entry.get("superseded_by_key"):
            status = "superseded"
        else:
            status = derive_entry_status(entry)
        return {
            "tier": tier,
            "namespace_key": namespace_key,
            "subject": subject,
            "topic": topic,
            "key": key,
            "value": str(entry.get("value", "")),
            "type": entry_type,
            "memory_class": memory_category(entry_type),
            "ts": str(entry.get("ts", "")),
            "source": str(entry.get("source", "manual")),
            "ttl_days": entry.get("ttl_days"),
            "origin_session": entry.get("origin_session"),
            "status": status,
            "version": int(entry.get("version") or 1),
            "last_recalled_ts": entry.get("last_recalled_ts"),
            "superseded_by_key": entry.get("superseded_by_key"),
            "superseded_at": entry.get("superseded_at"),
            "superseded_similarity": entry.get("superseded_similarity"),
        }
    return {
        "tier": tier,
        "namespace_key": namespace_key,
        "subject": subject,
        "topic": topic,
        "key": key,
        "value": str(entry),
        "type": "context",
        "memory_class": memory_category("context"),
        "ts": "",
        "source": "manual",
        "ttl_days": None,
        "origin_session": None,
        "status": "active",
        "version": 1,
        "last_recalled_ts": None,
    }


async def _user_root_sessions(db, user_id: UUID) -> list[AssistantSession]:
    """Return every root session (no parent) the current user owns.

    Memory persists on the ROOT session of a tree, so listing roots
    is equivalent to listing the user's distinct memory "containers".
    The user_id filter is the load-bearing isolation boundary — no
    other endpoint should be able to cross it.
    """
    result = await db.execute(
        select(AssistantSession)
        .where(
            AssistantSession.user_id == user_id,
            AssistantSession.parent_session_id.is_(None),
        )
    )
    return list(result.scalars().all())


@router.get("/memory")
async def list_memory(
    user_id: CurrentUserID,
    db: DBSession,
    tier: str | None = Query(
        None, pattern="^(medium|long)$",
        description="Optional tier filter — 'medium' (session tree) or 'long' (namespace).",
    ),
    namespace_key: str | None = Query(
        None, max_length=120,
        description="Optional namespace filter — only returns 'long' entries scoped to this namespace.",
    ),
    subject: str | None = Query(
        None, max_length=60,
        description="Optional subject filter (e.g. 'cs') — derived from namespace_key prefix.",
    ),
    topic: str | None = Query(
        None, max_length=60,
        description="Optional topic filter (e.g. 'AI') — derived from namespace_key suffix.",
    ),
    memory_class: str | None = Query(
        None, pattern="^(semantic|episodic|procedural|preference|-)$",
        description=(
            "Optional cognitive-class filter. Maps to the three-way memory "
            "taxonomy (semantic / episodic / procedural) plus 'preference' "
            "for user-stated preferences and '-' for entries with no clean "
            "class mapping (hypothesis / context)."
        ),
    ),
) -> dict[str, Any]:
    """List every long-term memory entry the current user has stored.

    Returns medium-tier (session-tree) and long-tier (namespace) entries
    together so the user can see what RA might inject on future turns.
    Optional filters narrow the result by tier and/or namespace; the
    full unfiltered list is fine for normal account sizes (caps are
    80 medium + 120 long per root × small number of roots).

    Security:
        ``user_id`` comes from the auth dependency. The root-session
        lookup filters by ``user_id`` — a leaked session id from
        another user yields zero rows here. Verified by
        ``test_memory_api_user_isolation``.
    """
    roots = await _user_root_sessions(db, user_id)

    rows: list[dict[str, Any]] = []
    namespaces_seen: set[str] = set()
    counts = {"medium": 0, "long": 0}
    class_counts: dict[str, int] = {}
    subjects_seen: set[str] = set()
    topics_seen: set[str] = set()
    for root in roots:
        state = root.state or {}
        # ── Medium (tree) memory ──────────────────────────────────
        if tier in (None, "medium"):
            tree_mem = state.get("tree_memory") or {}
            if isinstance(tree_mem, dict):
                for key, entry in tree_mem.items():
                    counts["medium"] += 1
                    row = _memory_entry_to_row(
                        tier="medium", namespace_key="", key=key, entry=entry,
                    )
                    # The root session UUID is needed for per-entry
                    # delete because a single user owns multiple
                    # roots (one per top-level session); the UI must
                    # tell us WHICH root the entry lives in.
                    row["root_session_id"] = str(root.id)
                    rows.append(row)
        # ── Long (namespace) memory ──────────────────────────────
        if tier in (None, "long"):
            ns_mem = state.get("ns_memory") or {}
            if isinstance(ns_mem, dict):
                for ns_key, bucket in ns_mem.items():
                    if namespace_key and ns_key != namespace_key:
                        continue
                    if not isinstance(bucket, dict):
                        continue
                    namespaces_seen.add(ns_key)
                    for key, entry in bucket.items():
                        counts["long"] += 1
                        row = _memory_entry_to_row(
                            tier="long", namespace_key=ns_key, key=key, entry=entry,
                        )
                        row["root_session_id"] = str(root.id)
                        rows.append(row)

    # Subject / topic / class filters applied AFTER projection so the
    # counts above reflect the FULL set the user owns (the UI shows
    # them as headline totals) and the returned ``entries`` honour
    # the filter selection. Guard each filter on ``isinstance(..., str)
    # and len(...) > 0`` because FastAPI's ``Query(None, ...)`` default
    # is a sentinel object (not Python ``None``) when the function is
    # called directly — a truthy check would silently filter every
    # row out in tests / fallback paths.
    if isinstance(subject, str) and subject:
        rows = [r for r in rows if (r.get("subject") or "") == subject]
    if isinstance(topic, str) and topic:
        rows = [r for r in rows if (r.get("topic") or "") == topic]
    if isinstance(memory_class, str) and memory_class:
        rows = [r for r in rows if (r.get("memory_class") or "-") == memory_class]

    # Surface the available subject / topic / class facets so the UI
    # can render filter chips without re-querying. Counts are taken
    # over the UNFILTERED set so chips don't disappear after a
    # filter is applied — the user can always un-pick a chip.
    for r in rows:
        subjects_seen.add(r.get("subject") or "")
        topics_seen.add(r.get("topic") or "")
        cls = r.get("memory_class") or "-"
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Sort newest first by timestamp — empty ts sorts last so legacy
    # entries don't clutter the top of the list.
    rows.sort(key=lambda r: r.get("ts") or "", reverse=True)

    # Read the user's pause-injection toggle (global + per-namespace
    # overrides) so the UI surfaces effective state at the top of the
    # panel and per-namespace chip. Inline query — the User row is
    # small and called rarely on this endpoint.
    from app.models.user import User as _User
    user_row = await db.execute(select(_User).where(_User.id == user_id))
    user_obj = user_row.scalar_one_or_none()
    injection_enabled = bool(getattr(user_obj, "memory_injection_enabled", True))
    raw_overrides = getattr(user_obj, "memory_injection_overrides", None) or {}
    overrides: dict[str, bool] = {
        str(k): bool(v) for k, v in raw_overrides.items()
        if isinstance(raw_overrides, dict)
    }

    return {
        "entries": rows,
        "counts": counts,
        "namespaces": sorted(namespaces_seen),
        "subjects": sorted(s for s in subjects_seen if s),
        "topics": sorted(t for t in topics_seen if t),
        "class_counts": class_counts,
        "tiers": ["medium", "long"],
        "memory_classes": ["semantic", "episodic", "procedural", "preference", "-"],
        "injection_enabled": injection_enabled,
        "injection_overrides": overrides,
    }


# ── Memory revisions (history / restore) ───────────────────────────────────


@router.get("/memory/roots")
async def list_memory_roots(
    user_id: CurrentUserID,
    db: DBSession,
) -> dict[str, Any]:
    """List the user's root sessions so the Memory UI can target a
    manual write even when no entries exist yet.

    The Settings → Memory "Add memory" button needs to attach the
    new entry to a concrete root session bucket. When the user has
    no entries yet, ``list_memory`` returns an empty list and the
    UI can't infer a root from existing entries — this endpoint
    closes the gap by returning every root the user owns. Caller
    is the modal's namespace dropdown.
    """
    roots = await _user_root_sessions(db, user_id)
    return {
        "roots": [
            {
                "id": str(r.id),
                "title": r.title or "",
                "namespace_key": r.namespace_key or "",
                "topic_keys": list(r.topic_keys or []),
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in roots
        ],
    }


@router.get("/memory/revisions")
async def list_memory_revisions(
    user_id: CurrentUserID,
    db: DBSession,
    tier: str = Query(..., pattern="^(medium|long)$"),
    key: str = Query(..., min_length=1, max_length=200),
    namespace_key: str = Query(default="", max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Return the full revision history for one memory entry.

    Always user-scoped: the query filters by ``user_id`` first so a
    leaked tier+key combination from another user yields zero rows.
    Newest revisions first, with previous_value populated so the UI
    can render a diff between any two adjacent revisions.
    """
    result = await db.execute(
        select(MemoryRevision)
        .where(
            MemoryRevision.user_id == user_id,
            MemoryRevision.tier == tier,
            MemoryRevision.namespace_key == (namespace_key or ""),
            MemoryRevision.key == key,
        )
        .order_by(MemoryRevision.created_at.desc())
        .limit(limit)
    )
    revisions = list(result.scalars().all())
    return {
        "revisions": [
            {
                "id": str(r.id),
                "action": r.action,
                "status": r.status,
                "value": r.value,
                "previous_value": r.previous_value,
                "entry_type": r.entry_type,
                "source": r.source,
                "ttl_days": r.ttl_days,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "subject": r.subject,
                "topic": r.topic,
                "origin_session_id": str(r.origin_session_id) if r.origin_session_id else None,
                "root_session_id": str(r.root_session_id),
                "extras": r.extras or {},
            }
            for r in revisions
        ],
        "count": len(revisions),
    }


class MemoryRestoreRequest(BaseModel):
    """Restore body — revision_id is the only required field.

    We fetch the revision (validating ownership) and reapply its
    ``value`` + metadata to the live entry. A new ``restore`` revision
    is recorded so the audit trail is complete.
    """
    revision_id: UUID


@router.post("/memory/restore", status_code=200)
async def restore_memory_revision(
    body: MemoryRestoreRequest,
    user_id: CurrentUserID,
    db: DBSession,
) -> dict[str, Any]:
    """Restore a previous revision's value to the live memory entry.

    Workflow:
        1. Fetch the revision; validate ``user_id`` ownership.
        2. Fetch the root session that owned the bucket.
        3. Under the per-session lock, reapply the revision's value
           (or recreate the entry if it had been deleted).
        4. Append a new ``restore`` revision so the audit trail
           records the action.

    A restore of a ``delete`` revision recreates the entry at the
    deleted-from key. A restore of an ``update`` revision overwrites
    whatever is currently there with the revision's value.
    """
    from app.assistant.memory_revisions import record_revision
    from app.assistant.state_lock import session_state_lock
    from app.assistant.tools.memory import _SCOPE_TO_BUCKET
    from sqlalchemy.orm.attributes import flag_modified

    # 1) Fetch revision + ownership check
    rev_row = await db.execute(
        select(MemoryRevision).where(
            MemoryRevision.id == body.revision_id,
            MemoryRevision.user_id == user_id,
        )
    )
    revision = rev_row.scalar_one_or_none()
    if revision is None:
        raise HTTPException(status_code=404, detail="revision not found")

    # 2) Fetch root session (ownership-checked)
    root_row = await db.execute(
        select(AssistantSession).where(
            AssistantSession.id == revision.root_session_id,
            AssistantSession.user_id == user_id,
        )
    )
    root = root_row.scalar_one_or_none()
    if root is None:
        raise HTTPException(status_code=404, detail="root session not found")

    bucket_name = _SCOPE_TO_BUCKET.get(revision.tier)
    if not bucket_name:
        raise HTTPException(status_code=400, detail=f"cannot restore tier {revision.tier!r}")

    restored_value = revision.value or revision.previous_value or ""
    if not restored_value:
        raise HTTPException(
            status_code=400,
            detail="revision has no value to restore (was a pure delete with no prior value)",
        )

    async with session_state_lock(root.id):
        state = dict(root.state or {})
        if revision.tier == "long":
            ns_mem = dict(state.get(bucket_name) or {})
            ns_bucket = dict(ns_mem.get(revision.namespace_key) or {})
            prior_live = ns_bucket.get(revision.key)
            ns_bucket[revision.key] = {
                "value": restored_value,
                "type": revision.entry_type or "context",
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": "restore",
                "restored_from_revision": str(revision.id),
            }
            ns_mem[revision.namespace_key] = ns_bucket
            state[bucket_name] = ns_mem
        else:  # medium
            tree_mem = dict(state.get(bucket_name) or {})
            prior_live = tree_mem.get(revision.key)
            tree_mem[revision.key] = {
                "value": restored_value,
                "type": revision.entry_type or "context",
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": "restore",
                "restored_from_revision": str(revision.id),
            }
            state[bucket_name] = tree_mem
        root.state = state
        flag_modified(root, "state")

        prior_live_value: str | None = None
        if isinstance(prior_live, dict):
            prior_live_value = str(prior_live.get("value") or "")
        elif prior_live is not None:
            prior_live_value = str(prior_live)

        await record_revision(
            db,
            user_id=user_id,
            session_id=root.id,
            tier=revision.tier,
            key=revision.key,
            value=restored_value,
            action="restore",
            namespace_key=revision.namespace_key or "",
            entry_type=revision.entry_type,
            source="restore",
            previous_value=prior_live_value,
            extras={"restored_from_revision_id": str(revision.id)},
        )
        await db.commit()
    return {"restored": True, "key": revision.key, "tier": revision.tier}


class MemoryUpsertRequest(BaseModel):
    """User-initiated add or edit of a long-term memory entry.

    ``root_session_id`` is required so we know which memory bucket
    the entry belongs to — a user typically has several root sessions
    (one per top-level investigation), and a manual write must land
    in a specific one. The frontend resolves the root via the
    inspect-list response; on a fresh empty namespace it falls back
    to the user's earliest root.
    """
    tier: str = Field(..., pattern="^(medium|long)$")
    key: str = Field(..., min_length=1, max_length=200)
    value: str = Field(..., min_length=1, max_length=4000)
    entry_type: str = Field(
        default="context",
        pattern="^(finding|preference|concept|hypothesis|context|paper_note|episode|skill|procedure)$",
    )
    namespace_key: str = Field(default="", max_length=120)
    root_session_id: UUID
    ttl_days: int | None = Field(default=None, ge=1, le=365)


@router.post("/memory", status_code=200)
async def upsert_memory_entry(
    body: MemoryUpsertRequest,
    user_id: CurrentUserID,
    db: DBSession,
) -> dict[str, Any]:
    """Manually add a new long-term memory entry or update an existing one.

    Surfaces the same write path as the auto-memory consolidator, but
    initiated by the user from Settings → Memory. Ownership of the
    target ``root_session_id`` is validated before any mutation —
    cross-user writes return 404 (existence not leaked). The audit
    trail records ``action='create'`` or ``action='update'`` with the
    prior value so the user can roll back via the History modal.
    """
    from app.assistant.memory_revisions import record_revision
    from app.assistant.state_lock import session_state_lock
    from app.assistant.tools.memory import _SCOPE_TO_BUCKET, _normalize_key
    from sqlalchemy.orm.attributes import flag_modified

    # Ownership check.
    result = await db.execute(
        select(AssistantSession).where(
            AssistantSession.id == body.root_session_id,
            AssistantSession.user_id == user_id,
        )
    )
    root = result.scalar_one_or_none()
    if root is None:
        raise HTTPException(status_code=404, detail="root session not found")

    norm_key = _normalize_key(body.key)
    bucket_name = _SCOPE_TO_BUCKET[body.tier]
    now_iso = datetime.now(timezone.utc).isoformat()

    async with session_state_lock(root.id):
        state = dict(root.state or {})
        prior_value: str | None = None
        prior_type: str = body.entry_type
        if body.tier == "long":
            ns_mem = dict(state.get(bucket_name) or {})
            ns_bucket = dict(ns_mem.get(body.namespace_key) or {})
            prev = ns_bucket.get(norm_key)
            if isinstance(prev, dict):
                prior_value = str(prev.get("value") or "")
                prior_type = str(prev.get("type") or body.entry_type)
            elif prev is not None:
                prior_value = str(prev)
            new_entry: dict[str, Any] = {
                "value": body.value,
                "type": body.entry_type,
                "ts": now_iso,
                "source": "manual",
            }
            if body.ttl_days:
                new_entry["ttl_days"] = int(body.ttl_days)
            ns_bucket[norm_key] = new_entry
            ns_mem[body.namespace_key] = ns_bucket
            state[bucket_name] = ns_mem
        else:  # medium
            tree_mem = dict(state.get(bucket_name) or {})
            prev = tree_mem.get(norm_key)
            if isinstance(prev, dict):
                prior_value = str(prev.get("value") or "")
                prior_type = str(prev.get("type") or body.entry_type)
            elif prev is not None:
                prior_value = str(prev)
            new_entry = {
                "value": body.value,
                "type": body.entry_type,
                "ts": now_iso,
                "source": "manual",
            }
            if body.ttl_days:
                new_entry["ttl_days"] = int(body.ttl_days)
            tree_mem[norm_key] = new_entry
            state[bucket_name] = tree_mem
        root.state = state
        flag_modified(root, "state")

        await record_revision(
            db,
            user_id=user_id,
            session_id=root.id,
            tier=body.tier,
            key=norm_key,
            value=body.value,
            action=("update" if prior_value is not None else "create"),
            namespace_key=body.namespace_key if body.tier == "long" else "",
            entry_type=body.entry_type,
            source="manual",
            previous_value=prior_value,
            ttl_days=body.ttl_days,
        )
        await db.commit()
    return {"saved": True, "key": norm_key, "tier": body.tier}


class MemoryInjectionToggleRequest(BaseModel):
    """Pause / resume injection. ``namespace_key`` is optional — when
    supplied the toggle is scoped to that one namespace; when omitted
    it sets the user-global default. Per-namespace overrides shadow
    the global default at read time."""
    enabled: bool
    namespace_key: str | None = Field(default=None, max_length=120)


@router.post("/memory/injection", status_code=200)
async def toggle_memory_injection(
    body: MemoryInjectionToggleRequest,
    user_id: CurrentUserID,
    db: DBSession,
) -> dict[str, Any]:
    """Enable or disable RA's long-term memory injection.

    Two scopes:

    * **Global** (``namespace_key`` omitted) — flips the user-wide
      default. Stored memory is preserved either way.
    * **Per-namespace** (``namespace_key`` supplied) — adds (or
      updates) an override entry for that namespace. The override
      shadows the global default at read time, so a user can leave
      memory ON globally but PAUSE injection only for a specific
      namespace where the stored memory is noisy.

    Short-term chat memory is unaffected in both cases — it's the
    inline conversation context the model already sees.
    """
    from app.models.user import User as _User
    from sqlalchemy.orm.attributes import flag_modified

    row = await db.execute(select(_User).where(_User.id == user_id))
    user = row.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")

    nk = (body.namespace_key or "").strip()
    if nk:
        overrides = dict(user.memory_injection_overrides or {})
        overrides[nk] = bool(body.enabled)
        user.memory_injection_overrides = overrides
        flag_modified(user, "memory_injection_overrides")
        await db.commit()
        return {
            "scope": "namespace",
            "namespace_key": nk,
            "injection_enabled": bool(body.enabled),
            "global_injection_enabled": bool(user.memory_injection_enabled),
            "overrides": overrides,
        }

    user.memory_injection_enabled = bool(body.enabled)
    await db.commit()
    return {
        "scope": "global",
        "injection_enabled": bool(body.enabled),
        "overrides": dict(user.memory_injection_overrides or {}),
    }


@router.delete("/memory/injection", status_code=200)
async def clear_namespace_injection_override(
    user_id: CurrentUserID,
    db: DBSession,
    namespace_key: str = Query(..., min_length=1, max_length=120),
) -> dict[str, Any]:
    """Drop a per-namespace override so the namespace falls back to
    the global default. Idempotent — removing a key that isn't there
    is a no-op."""
    from app.models.user import User as _User
    from sqlalchemy.orm.attributes import flag_modified

    row = await db.execute(select(_User).where(_User.id == user_id))
    user = row.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    overrides = dict(user.memory_injection_overrides or {})
    if namespace_key in overrides:
        del overrides[namespace_key]
        user.memory_injection_overrides = overrides
        flag_modified(user, "memory_injection_overrides")
        await db.commit()
    return {
        "namespace_key": namespace_key,
        "removed": True,
        "overrides": overrides,
    }


class MemoryDeleteRequest(BaseModel):
    """Per-entry delete request body.

    ``root_session_id`` is required so we never have to guess WHICH
    root holds the entry — a user with multiple top-level sessions
    can have the same key in multiple roots, and silently picking
    one would delete the wrong fact.
    """
    tier: str = Field(..., pattern="^(medium|long)$")
    key: str = Field(..., min_length=1, max_length=200)
    namespace_key: str = Field(default="", max_length=120)
    root_session_id: UUID


@router.delete("/memory", status_code=200)
async def delete_memory_entry(
    body: MemoryDeleteRequest,
    user_id: CurrentUserID,
    db: DBSession,
) -> dict[str, Any]:
    """Delete a single long-term memory entry.

    The root session must belong to the calling user; otherwise this
    raises 404 (we don't distinguish "wrong owner" from "missing" so
    we don't leak existence information). The operation mutates the
    JSONB state column under :class:`session_state_lock` to serialise
    with the orchestrator's own writes.
    """
    from app.assistant.state_lock import session_state_lock

    # Fetch + ownership check in one query — never trust the
    # ``root_session_id`` from the request body alone.
    result = await db.execute(
        select(AssistantSession).where(
            AssistantSession.id == body.root_session_id,
            AssistantSession.user_id == user_id,
        )
    )
    root = result.scalar_one_or_none()
    if root is None:
        raise HTTPException(status_code=404, detail="memory entry not found")

    from app.assistant.memory_revisions import record_revision

    bucket_name = _MEMORY_TIER_LABELS[body.tier]
    removed = False
    prior_value: str = ""
    prior_type: str = "context"
    async with session_state_lock(root.id):
        state = dict(root.state or {})
        if body.tier == "long":
            ns_mem = dict(state.get(bucket_name) or {})
            ns_bucket = dict(ns_mem.get(body.namespace_key) or {})
            if body.key in ns_bucket:
                removed_entry = ns_bucket[body.key]
                if isinstance(removed_entry, dict):
                    prior_value = str(removed_entry.get("value") or "")
                    prior_type = str(removed_entry.get("type") or "context")
                elif removed_entry is not None:
                    prior_value = str(removed_entry)
                del ns_bucket[body.key]
                ns_mem[body.namespace_key] = ns_bucket
                state[bucket_name] = ns_mem
                removed = True
        else:
            tree_mem = dict(state.get(bucket_name) or {})
            if body.key in tree_mem:
                removed_entry = tree_mem[body.key]
                if isinstance(removed_entry, dict):
                    prior_value = str(removed_entry.get("value") or "")
                    prior_type = str(removed_entry.get("type") or "context")
                elif removed_entry is not None:
                    prior_value = str(removed_entry)
                del tree_mem[body.key]
                state[bucket_name] = tree_mem
                removed = True
        if removed:
            root.state = state
            flag_modified(root, "state")
            # Record the user-initiated delete in the audit trail so
            # it can be restored later. Failure to record never
            # blocks the delete itself.
            await record_revision(
                db,
                user_id=user_id,
                session_id=root.id,
                tier=body.tier,
                key=body.key,
                value="",
                action="delete",
                namespace_key=body.namespace_key if body.tier == "long" else "",
                entry_type=prior_type,
                source="manual",
                previous_value=prior_value,
                status="deleted",
            )
            await db.commit()
    if not removed:
        # Idempotent: a delete of an already-gone entry is fine —
        # tell the caller so the UI can refresh its view without
        # alarming the user.
        return {"removed": False, "message": "entry not found (already removed)"}
    return {"removed": True}


class MemoryClearRequest(BaseModel):
    """Bulk-clear request.

    Scope semantics:
      * ``tier="long"`` + ``namespace_key`` → clear that one namespace.
      * ``tier="long"`` + no namespace      → clear every namespace bucket.
      * ``tier="medium"``                   → clear the tree_memory bucket.
      * ``tier="all"``                      → clear medium AND long.
    Short-term (chat) memory is intentionally NOT clearable here — it
    lives on individual chat sessions and is auto-pruned on session
    end. The user spec explicitly says clearing long-term memory must
    NOT touch short-term or in-flight chat context.
    """
    tier: str = Field(..., pattern="^(medium|long|all)$")
    namespace_key: str | None = Field(default=None, max_length=120)


@router.post("/memory/clear", status_code=200)
async def clear_memory(
    body: MemoryClearRequest,
    user_id: CurrentUserID,
    db: DBSession,
) -> dict[str, Any]:
    """Bulk-clear long-term memory across the user's root sessions.

    Always user-scoped via ``_user_root_sessions``. Returns the per-tier
    count of removed entries so the UI can show "Cleared N memories"
    confirmation. Short-term chat memory is never touched, matching
    the user's explicit spec.
    """
    from app.assistant.state_lock import session_state_lock

    roots = await _user_root_sessions(db, user_id)
    removed = {"medium": 0, "long": 0}

    for root in roots:
        # Take the per-session lock individually so a long iteration
        # doesn't hold one global lock across every session.
        async with session_state_lock(root.id):
            state = dict(root.state or {})
            mutated = False
            if body.tier in ("medium", "all"):
                tree_mem = state.get("tree_memory") or {}
                if isinstance(tree_mem, dict) and tree_mem:
                    removed["medium"] += len(tree_mem)
                    state["tree_memory"] = {}
                    mutated = True
            if body.tier in ("long", "all"):
                ns_mem = state.get("ns_memory") or {}
                if isinstance(ns_mem, dict):
                    if body.namespace_key:
                        bucket = ns_mem.get(body.namespace_key)
                        if isinstance(bucket, dict) and bucket:
                            removed["long"] += len(bucket)
                            ns_mem = dict(ns_mem)
                            ns_mem[body.namespace_key] = {}
                            state["ns_memory"] = ns_mem
                            mutated = True
                    else:
                        for bucket in ns_mem.values():
                            if isinstance(bucket, dict):
                                removed["long"] += len(bucket)
                        if ns_mem:
                            state["ns_memory"] = {}
                            mutated = True
            if mutated:
                root.state = state
                flag_modified(root, "state")
                await db.commit()
    return {"removed": removed}
