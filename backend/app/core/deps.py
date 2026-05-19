"""FastAPI dependency injection — DB sessions, current user, adapters."""

import time
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.core.tracking import current_user_id as _current_user_id_ctx
from app.db.session import async_session_factory


# ── Activity check cache ─────────────────────────────────────────────────────
# Tiny in-process TTL cache of ``user_id → is_active`` so we don't issue a
# DB roundtrip on every authenticated request. Admin de-/re-activations
# take effect within ``_ACTIVE_CACHE_TTL`` seconds across all workers
# (instant for the worker that handled the PATCH, since we proactively
# invalidate that user's slot on writes). The cache is bounded in size
# by an LRU eviction in ``_check_user_active`` so a hostile login burst
# can't exhaust memory.
_ACTIVE_CACHE: dict[UUID, tuple[bool, float]] = {}
_ACTIVE_CACHE_TTL = 30.0
_ACTIVE_CACHE_MAX = 5000


def invalidate_active_cache(user_id: UUID | None = None) -> None:
    """Drop a single user's cached is_active state, or wipe the whole cache.

    Called by admin endpoints whenever a user's ``is_active`` bit flips so
    the very next request from that user (or any worker) sees the new
    value without waiting for the TTL.
    """
    if user_id is None:
        _ACTIVE_CACHE.clear()
        return
    _ACTIVE_CACHE.pop(user_id, None)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_db() -> AsyncSession:
    """Yields a per-request async DB session — always closed after the request."""
    async with async_session_factory() as session:
        yield session


DBSession = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user_id(token: Annotated[str, Depends(oauth2_scheme)]) -> UUID:
    """Decodes JWT and returns the user UUID, enforcing ``is_active``.

    Beyond JWT validity, this also verifies the user's row still has
    ``is_active=True``. An admin who flips a user's active bit off via
    ``PATCH /admin/users/{id}`` must be able to revoke access without
    waiting for the JWT to expire — without this check, an already-issued
    7-day token would let a deactivated user keep working.

    The is_active lookup is cached for ``_ACTIVE_CACHE_TTL`` seconds with
    proactive invalidation from admin write paths, so the cost is one
    cheap PK lookup per user per TTL window.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    deactivated_exc = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Account is deactivated. Contact your administrator.",
    )
    try:
        payload = decode_access_token(token)
        user_id_str: str | None = payload.get("sub")
        if user_id_str is None:
            raise credentials_exc
        uid = UUID(user_id_str)
    except (JWTError, ValueError):
        raise credentials_exc

    if not await _check_user_active(uid):
        raise deactivated_exc

    # Stash on the request-local contextvar so token-usage tracking can
    # attribute every LLM call made during this request to this user.
    _current_user_id_ctx.set(uid)
    return uid


async def _check_user_active(user_id: UUID) -> bool:
    """Return True iff the user exists and has ``is_active=True``.

    Cached for ``_ACTIVE_CACHE_TTL`` seconds. A missing user row is
    treated as deactivated so a JWT issued before account deletion
    can't outlive the row. Failures opening a DB session fail-open
    (return True) so a transient DB outage doesn't lock everyone out.
    """
    now = time.monotonic()
    cached = _ACTIVE_CACHE.get(user_id)
    if cached and (now - cached[1]) < _ACTIVE_CACHE_TTL:
        return cached[0]

    try:
        from app.models.user import User as UserModel
        async with async_session_factory() as db:
            row = await db.get(UserModel, user_id)
            is_active = bool(row is not None and getattr(row, "is_active", True))
    except Exception:
        # Fail-open on transient DB error — a permanent outage will
        # surface elsewhere; we won't 403 every authenticated user
        # because the pool blipped.
        return True

    # Bounded cache — drop the oldest entry when we'd exceed the cap.
    if len(_ACTIVE_CACHE) >= _ACTIVE_CACHE_MAX:
        try:
            oldest = min(_ACTIVE_CACHE.items(), key=lambda kv: kv[1][1])[0]
            _ACTIVE_CACHE.pop(oldest, None)
        except ValueError:
            pass
    _ACTIVE_CACHE[user_id] = (is_active, now)
    return is_active


CurrentUserID = Annotated[UUID, Depends(get_current_user_id)]


async def require_admin(
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    db: DBSession,
) -> UUID:
    """Reject the request with 403 unless the authenticated user is an admin.

    Admin-only endpoints (settings panel, user management, graph rebuild)
    inject this dependency. Plays nicely with ``CurrentUserID`` — admins
    are still regular users, just with extra privileges.
    """
    from app.models.user import User as UserModel

    row = await db.get(UserModel, user_id)
    if row is None or not getattr(row, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin privileges required.")
    return user_id


AdminUserID = Annotated[UUID, Depends(require_admin)]


# ── Feature-flag dependency factory ──────────────────────────────────────────
#
# Single source of truth for gating every flagged feature. Attach as:
#
#     router = APIRouter(prefix="/foo", dependencies=[Depends(require_feature("foo_enabled"))])
#
# or per-endpoint:
#
#     @router.post("/bar", dependencies=[Depends(require_feature("foo_enabled"))])
#
# Resolution order matches the rest of the app (defaults → global admin →
# per-user override) so admins can disable a feature for everyone OR just
# for a single user, and every gated route reacts identically.
#
# Returns 404 instead of 403 so a disabled feature looks like it doesn't
# exist — keeps the response shape uniform with the route-not-mounted
# experience and prevents leaking feature catalogues to non-entitled users.
# When billing tiers ship later, mapping a tier → set of enabled features
# becomes a single helper in ``feature_flags.py``; every call site here
# stays unchanged.

def require_feature(feature: str):
    """Build a FastAPI dependency that 404s when ``feature`` is off for the user.

    Resolves the user via *either* the standard ``Authorization: Bearer``
    header *or* a ``token`` query parameter — SSE endpoints (EventSource)
    cannot set headers, so they pass the JWT via querystring. When neither
    source is present (e.g. an OPTIONS preflight gets past CORS), we fall
    back to the global flag value so we never 401 a legitimately
    unauthenticated request that the underlying route is itself happy
    to handle.
    """

    async def _check(
        request: Request,
        bearer_token: Annotated[str | None, Depends(_optional_bearer)] = None,
        token: str | None = Query(default=None),
    ) -> None:
        from app.services.feature_flags import is_feature_enabled, is_global_feature_enabled

        # Best-effort user resolution. We swallow JWT errors and fall back
        # to the global flag — the actual auth-enforcing dependency on the
        # endpoint will produce a clean 401 if the token is genuinely bad.
        resolved_uid: UUID | None = None
        raw_token = bearer_token or token
        if raw_token:
            try:
                payload = decode_access_token(raw_token)
                sub = payload.get("sub")
                if sub:
                    resolved_uid = UUID(sub)
            except (JWTError, ValueError, AttributeError):
                resolved_uid = None

        if resolved_uid is None:
            enabled = await is_global_feature_enabled(feature)
        else:
            enabled = await is_feature_enabled(feature, resolved_uid)

        if not enabled:
            raise HTTPException(status_code=404, detail=f"Feature '{feature}' is not available.")

    _check.__name__ = f"require_feature_{feature}"
    return _check


async def _optional_bearer(request: Request) -> str | None:
    """Return the bearer token from the ``Authorization`` header if present.

    Lives separately from ``get_current_user_id`` because we want soft
    failure — missing or malformed headers should propagate ``None``, not
    raise 401. The real auth dep on the endpoint will reject bad tokens
    after the feature flag check has passed.
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None
