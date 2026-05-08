"""Settings router — provider config, topic subscriptions, notifications, profile."""

from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import func, select

from app.core.deps import CurrentUserID, DBSession
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
    """Return the current user's LLM and embedding provider configuration.

    Args:
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        A dict with provider and model fields, or an empty dict if no
        provider settings row exists yet.
    """
    repo = UserRepository(db)
    s = await repo.get_provider_settings(user_id)
    if not s:
        return {}
    return {
        "llm_provider": s.llm_provider.value,
        "cheap_model": s.cheap_model,
        "quality_model": s.quality_model,
        "reasoning_model": s.reasoning_model,
        "embedding_provider": s.embedding_provider.value,
        "embedding_model": s.embedding_model,
        "embedding_dim": s.embedding_dim,
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


@router.get("/api-keys")
async def get_api_keys(user_id: CurrentUserID, db: DBSession):
    """Return masked API key status — which keys are set from env vs user override."""
    from app.core.config import settings as _cfg

    repo = UserRepository(db)
    ps = await repo.get_provider_settings(user_id)

    def _mask(val: str | None) -> str:
        """Mask an API key string, showing only the first 4 characters and replacing the rest with bullets."""
        if not val:
            return ""
        visible = val[:4]
        return visible + "•" * min(len(val) - 4, 24)

    env_keys = {
        "openai":    _cfg.openai_api_key,
        "anthropic": _cfg.anthropic_api_key,
        "google":    _cfg.google_api_key,
    }
    user_keys = {
        "openai":    ps.encrypted_openai_key    if ps else None,
        "anthropic": ps.encrypted_anthropic_key if ps else None,
        "google":    ps.encrypted_google_key    if ps else None,
    }

    result = {}
    for provider in ("openai", "anthropic", "google"):
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
            {
                "workflow":      wf or "(unknown)",
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
