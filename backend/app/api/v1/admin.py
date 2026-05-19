"""Admin panel API — feature flags, user management, basic analytics.

Endpoints under ``/api/v1/admin/*`` all require ``require_admin``. Non-admin
callers see 403 even on simple reads, so the panel is invisible without
the bit set.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from app.core.deps import AdminUserID, CurrentUserID, DBSession
from app.models.assistant import AssistantMessage, AssistantSession
from app.models.genie import IdeaCapsule
from app.models.paper import Bookmark, Paper
from app.models.user import User
from app.services.admin_settings import get_app_settings, set_app_settings
from app.services.feature_flags import (
    DEFAULTS as FEATURE_DEFAULTS,
    FEATURES,
    get_effective_features,
    set_user_overrides,
)

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Public settings probe ─────────────────────────────────────────────────────
# Non-admin endpoint that returns only the flags safe to leak to the frontend
# (so the UI can hide the Graph nav etc. without leaking the rest of admin
# config). Lives on the admin router for locality but skips the admin guard.

settings_router = APIRouter(prefix="/settings", tags=["settings"])


class PublicAppSettings(BaseModel):
    """Effective per-user feature view used by the frontend.

    Backwards compatible: the original ``graph_enabled`` field is still
    here, but the richer ``features`` map is the canonical surface — the
    frontend uses it to gate any nav / button / panel that's gated by a
    flag without needing a per-feature schema bump.
    """

    graph_enabled: bool = False
    features: dict[str, bool] = {}


@settings_router.get("/public", response_model=PublicAppSettings)
async def public_settings(user_id: CurrentUserID, db: DBSession) -> PublicAppSettings:
    """Return the effective feature map for the authenticated user."""
    eff = await get_effective_features(user_id, db)
    return PublicAppSettings(
        graph_enabled=bool(eff.get("graph_enabled", False)),
        features=eff,
    )


@settings_router.get("/features/catalog")
async def features_catalog(_uid: CurrentUserID) -> dict[str, dict]:
    """Public catalog of all feature flag keys + labels + descriptions."""
    return FEATURES


# ── Admin endpoints (require_admin) ───────────────────────────────────────────


@router.get("/me")
async def admin_me(_uid: AdminUserID) -> dict[str, Any]:
    """Trivial probe used by the frontend to confirm admin access."""
    return {"admin": True}


@router.get("/settings")
async def get_settings(_uid: AdminUserID, db: DBSession) -> dict[str, Any]:
    """Return the merged global settings (defaults overlaid with stored overrides)."""
    return await get_app_settings(db)


@router.get("/features")
async def get_global_features(_uid: AdminUserID, db: DBSession) -> dict[str, Any]:
    """Return the global feature map merged with admin overrides.

    Each key corresponds to a flag in
    :data:`app.services.feature_flags.FEATURES`. Useful for the global
    ``Feature flags`` tab on the admin panel.
    """
    stored = await get_app_settings(db)
    return {k: bool(stored.get(k, FEATURE_DEFAULTS[k])) for k in FEATURE_DEFAULTS}


@router.patch("/features")
async def patch_global_features(
    patch: dict[str, bool | None],
    _uid: AdminUserID,
) -> dict[str, Any]:
    """Update one or more global feature flags. Unknown keys are ignored."""
    sanitized: dict[str, Any] = {}
    for k, v in (patch or {}).items():
        if k in FEATURE_DEFAULTS and v is not None:
            sanitized[k] = bool(v)
    if not sanitized:
        return await get_app_settings()
    return await set_app_settings(sanitized)


class SettingsPatch(BaseModel):
    """Legacy single-flag patch — kept for backward compatibility."""
    graph_enabled: bool | None = None


@router.patch("/settings")
async def update_settings(body: SettingsPatch, _uid: AdminUserID) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    if body.graph_enabled is not None:
        patch["graph_enabled"] = bool(body.graph_enabled)
    if not patch:
        return await get_app_settings()
    return await set_app_settings(patch)


@router.get("/users/{user_id}/features")
async def get_user_features(user_id: UUID, _uid: AdminUserID, db: DBSession) -> dict[str, Any]:
    """Return both raw per-user overrides and effective merged features."""
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    eff = await get_effective_features(user_id, db)
    return {
        "overrides": dict(user.feature_overrides or {}),
        "effective": eff,
        "defaults": FEATURE_DEFAULTS,
    }


@router.patch("/users/{user_id}/features")
async def patch_user_features(
    user_id: UUID,
    patch: dict[str, bool | None],
    _uid: AdminUserID,
) -> dict[str, Any]:
    """Set per-user feature overrides. ``null`` clears the override."""
    try:
        eff = await set_user_overrides(user_id, patch or {})
    except LookupError:
        raise HTTPException(status_code=404, detail="User not found")
    return {"effective": eff}


class AdminUserItem(BaseModel):
    id: UUID
    email: str
    display_name: str
    is_active: bool
    is_admin: bool
    onboarding_complete: bool
    created_at: datetime
    # RBAC fields — forward-compatible. ``role`` is "admin"|"member" today;
    # ``tier_slug`` is None unless an admin has assigned a tier.
    role: str = "member"
    tier_slug: str | None = None

    model_config = {"from_attributes": True}


@router.get("/users", response_model=list[AdminUserItem])
async def list_users(_uid: AdminUserID, db: DBSession) -> list[AdminUserItem]:
    rows = await db.execute(select(User).order_by(User.created_at.desc()))
    return [AdminUserItem.model_validate(u, from_attributes=True) for u in rows.scalars()]


class AdminCreateUserRequest(BaseModel):
    """Body for ``POST /admin/users``.

    All fields except ``email`` and ``password`` are optional. ``role`` is
    forward-compatible with the upcoming RBAC layer — passing
    ``role='admin'`` is equivalent to ``is_admin=true`` while we run with
    the two-state model, and will start applying tier-driven feature
    overrides once subscription tiers are wired.
    """

    email: str
    password: str
    display_name: str | None = None
    is_admin: bool = False
    role: str | None = None  # reserved for RBAC tiers ("admin"|"member"|"premium"…)


@router.post("/users", response_model=AdminUserItem, status_code=201)
async def create_user(
    body: AdminCreateUserRequest,
    _uid: AdminUserID,
    db: DBSession,
) -> AdminUserItem:
    """Create a new user account. Admin-only.

    Mirrors the public ``/auth/register`` flow but skips the OPEN_REGISTRATION
    gate and lets the admin set ``is_admin`` / ``role`` up front. Useful
    for inviting collaborators without exposing public signup.
    """
    from app.core.security import hash_password
    from app.models.user import (
        ExpertiseLevel,
        Orientation,
        UserInterestProfile,
        UserProviderSettings,
    )

    email = (body.email or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email is required.")
    if not body.password or len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="A user with that email already exists.")

    role = (body.role or "").strip().lower()
    promote = bool(body.is_admin) or role == "admin"

    user = User(
        email=email,
        hashed_password=hash_password(body.password),
        display_name=(body.display_name or email.split("@")[0]).strip() or "Researcher",
        expertise_level=ExpertiseLevel.practitioner,
        orientation=Orientation.both,
        onboarding_complete=True,
        is_admin=promote,
    )
    db.add(user)
    await db.flush()
    db.add(UserProviderSettings(user_id=user.id))
    db.add(UserInterestProfile(user_id=user.id))
    await db.commit()
    await db.refresh(user)
    return AdminUserItem.model_validate(user, from_attributes=True)


@router.delete("/users/{user_id}")
async def delete_user(user_id: UUID, admin_id: AdminUserID, db: DBSession) -> dict[str, bool]:
    """Hard-delete a user (and their cascaded records). Admin-only.

    Refuses to delete the calling admin to prevent self-lockout. All
    foreign keys cascade on delete at the schema level so this leaves
    no orphan rows.
    """
    if user_id == admin_id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    await db.delete(user)
    await db.commit()
    # Bounce any cached "is_active" lookup for this user so an already
    # issued token can't outlive the row — the auth dep will then see
    # the missing row and 403 the request as deactivated.
    try:
        from app.core.deps import invalidate_active_cache
        invalidate_active_cache(user_id)
    except Exception:
        pass
    return {"deleted": True}


class AdminResetPasswordRequest(BaseModel):
    new_password: str


# ── Tiers (RBAC scaffolding — feature_flags layer 3) ──────────────────────────


class TierItem(BaseModel):
    slug: str
    name: str
    description: str = ""
    feature_set: dict[str, bool] = {}
    quotas: dict[str, Any] = {}
    is_default: bool = False

    model_config = {"from_attributes": True}


class TierUpsertRequest(BaseModel):
    slug: str
    name: str
    description: str = ""
    feature_set: dict[str, bool] = {}
    quotas: dict[str, Any] = {}
    is_default: bool = False


@router.get("/tiers", response_model=list[TierItem])
async def list_tiers(_uid: AdminUserID, db: DBSession) -> list[TierItem]:
    """List all configured subscription tiers."""
    from app.models.rbac import Tier

    result = await db.execute(select(Tier).order_by(Tier.created_at.asc()))
    return [TierItem.model_validate(t, from_attributes=True) for t in result.scalars()]


@router.post("/tiers", response_model=TierItem, status_code=201)
async def create_tier(body: TierUpsertRequest, _uid: AdminUserID, db: DBSession) -> TierItem:
    """Create or upsert a tier — slug is the unique key."""
    from app.models.rbac import Tier

    slug = (body.slug or "").strip().lower()
    if not slug:
        raise HTTPException(status_code=400, detail="slug required")
    existing = (await db.execute(select(Tier).where(Tier.slug == slug))).scalar_one_or_none()
    # Filter feature_set to known flags only — typos in admin input shouldn't
    # become silent always-off values.
    fs = {k: bool(v) for k, v in (body.feature_set or {}).items() if k in FEATURE_DEFAULTS}
    if existing is None:
        tier = Tier(
            slug=slug,
            name=body.name,
            description=body.description,
            feature_set=fs,
            quotas=dict(body.quotas or {}),
            is_default=bool(body.is_default),
        )
        db.add(tier)
    else:
        tier = existing
        tier.name = body.name or tier.name
        tier.description = body.description
        tier.feature_set = fs
        tier.quotas = dict(body.quotas or {})
        tier.is_default = bool(body.is_default)
    # Only one tier may be is_default=True at a time. If this row is being
    # promoted, demote every other tier in the same transaction.
    if tier.is_default:
        from sqlalchemy import update as sql_update
        await db.execute(
            sql_update(Tier).where(Tier.slug != slug).values(is_default=False)
        )
    await db.commit()
    await db.refresh(tier)
    return TierItem.model_validate(tier, from_attributes=True)


@router.delete("/tiers/{slug}")
async def delete_tier(slug: str, _uid: AdminUserID, db: DBSession) -> dict[str, bool]:
    """Delete a tier. Users assigned to it fall back to global defaults."""
    from app.models.rbac import Tier
    from sqlalchemy import update as sql_update

    tier = (await db.execute(select(Tier).where(Tier.slug == slug))).scalar_one_or_none()
    if tier is None:
        raise HTTPException(status_code=404, detail="Tier not found")
    # Detach users first so the FK-less ``tier_slug`` column doesn't carry a
    # dangling reference.
    await db.execute(sql_update(User).where(User.tier_slug == slug).values(tier_slug=None))
    await db.delete(tier)
    await db.commit()
    return {"deleted": True}


class AssignTierRequest(BaseModel):
    tier_slug: str | None = None


@router.patch("/users/{user_id}/tier", response_model=AdminUserItem)
async def assign_user_tier(
    user_id: UUID,
    body: AssignTierRequest,
    _uid: AdminUserID,
    db: DBSession,
) -> AdminUserItem:
    """Assign ``user_id`` to a tier (or clear with ``tier_slug=null``)."""
    from app.models.rbac import Tier

    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    new_slug = (body.tier_slug or "").strip().lower() or None
    if new_slug is not None:
        tier = (await db.execute(select(Tier).where(Tier.slug == new_slug))).scalar_one_or_none()
        if tier is None:
            raise HTTPException(status_code=404, detail=f"Tier '{new_slug}' not found")
    user.tier_slug = new_slug
    await db.commit()
    await db.refresh(user)
    return AdminUserItem.model_validate(user, from_attributes=True)


@router.post("/users/{user_id}/password", status_code=200)
async def reset_user_password(
    user_id: UUID,
    body: AdminResetPasswordRequest,
    _uid: AdminUserID,
    db: DBSession,
) -> dict[str, bool]:
    """Force-reset a user's password. Admin-only.

    Useful when a user has lost access and self-service reset isn't
    wired yet. Logs a deliberate audit message so the action is
    discoverable in server logs.
    """
    from app.core.security import hash_password

    if not body.new_password or len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user.hashed_password = hash_password(body.new_password)
    await db.commit()
    return {"reset": True}


class UserPatch(BaseModel):
    is_active: bool | None = None
    is_admin: bool | None = None


@router.patch("/users/{user_id}", response_model=AdminUserItem)
async def patch_user(user_id: UUID, body: UserPatch, admin_id: AdminUserID, db: DBSession) -> AdminUserItem:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    # Safety: never let an admin demote / deactivate themselves out of admin
    # — there must always be at least one admin who can flip flags.
    if user.id == admin_id and (body.is_admin is False or body.is_active is False):
        raise HTTPException(
            status_code=400,
            detail="You cannot remove admin or deactivate your own account.",
        )
    if body.is_active is not None:
        user.is_active = bool(body.is_active)
    if body.is_admin is not None:
        user.is_admin = bool(body.is_admin)
    await db.commit()
    await db.refresh(user)
    # Invalidate the deps-layer active cache so the user's NEXT request
    # in any worker sees the new state immediately (rather than waiting
    # for the 30-second TTL to expire). Idempotent and free when the
    # user wasn't cached yet.
    try:
        from app.core.deps import invalidate_active_cache
        invalidate_active_cache(user_id)
    except Exception:
        pass
    return AdminUserItem.model_validate(user, from_attributes=True)


@router.get("/analytics")
async def analytics(_uid: AdminUserID, db: DBSession) -> dict[str, Any]:
    """High-level counts useful for at-a-glance admin overview."""
    now = datetime.now(timezone.utc)
    last_7 = now - timedelta(days=7)
    last_30 = now - timedelta(days=30)

    async def _scalar(stmt) -> int:
        result = await db.execute(stmt)
        v = result.scalar_one_or_none()
        return int(v or 0)

    total_users = await _scalar(select(func.count(User.id)))
    active_users = await _scalar(select(func.count(User.id)).where(User.is_active.is_(True)))
    admin_users = await _scalar(select(func.count(User.id)).where(User.is_admin.is_(True)))
    new_users_7d = await _scalar(select(func.count(User.id)).where(User.created_at >= last_7))
    total_papers = await _scalar(select(func.count(Paper.id)))
    total_bookmarks = await _scalar(select(func.count(Bookmark.id)))
    total_ideas = await _scalar(select(func.count(IdeaCapsule.id)))
    total_sessions = await _scalar(select(func.count(AssistantSession.id)))
    msgs_7d = await _scalar(select(func.count(AssistantMessage.id)).where(AssistantMessage.created_at >= last_7))
    msgs_30d = await _scalar(select(func.count(AssistantMessage.id)).where(AssistantMessage.created_at >= last_30))

    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "admins": admin_users,
            "new_last_7_days": new_users_7d,
        },
        "content": {
            "papers": total_papers,
            "bookmarks": total_bookmarks,
            "ideas": total_ideas,
            "assistant_sessions": total_sessions,
        },
        "activity": {
            "assistant_messages_last_7_days": msgs_7d,
            "assistant_messages_last_30_days": msgs_30d,
        },
        "generated_at": now.isoformat(),
    }
