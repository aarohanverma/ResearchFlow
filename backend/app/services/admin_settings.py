"""Read/write the singleton ``app_settings`` row keyed by ``"global"``.

Stored as a single JSONB document so we can add new feature flags without
schema migrations. Every flag has a documented default in :data:`DEFAULTS`,
and every consumer should read through ``get_app_settings`` (cached) — never
materialise its own defaults inline.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_factory
from app.models.admin import AppSetting

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    # When False, the entire Knowledge Graph feature is disabled — the
    # nav item is hidden, the /graph/* routes return 404, and assistant
    # tools that depend on it short-circuit. Default off because the
    # current graph builder doesn't scale well for large feeds.
    "graph_enabled": False,
}

_KEY = "global"
_CACHE: dict[str, Any] = {}
_CACHE_TS: float = 0.0
_TTL_SECONDS = 30.0


async def get_app_settings(db: AsyncSession | None = None) -> dict[str, Any]:
    """Return the merged settings dict (defaults overlaid with stored values).

    Cached for ``_TTL_SECONDS`` so hot paths can call this freely. The cache
    is invalidated on every successful ``set_app_setting`` write.
    """
    global _CACHE, _CACHE_TS
    if _CACHE and (time.monotonic() - _CACHE_TS) < _TTL_SECONDS:
        return dict(_CACHE)

    async def _fetch(session: AsyncSession) -> dict[str, Any]:
        result = await session.execute(select(AppSetting).where(AppSetting.key == _KEY))
        row = result.scalar_one_or_none()
        stored = dict((row.value if row else {}) or {})
        merged = {**DEFAULTS, **stored}
        return merged

    try:
        if db is not None:
            merged = await _fetch(db)
        else:
            async with async_session_factory() as session:
                merged = await _fetch(session)
    except Exception as exc:
        log.warning("get_app_settings failed, falling back to defaults: %s", exc)
        return dict(DEFAULTS)

    _CACHE = merged
    _CACHE_TS = time.monotonic()
    return dict(merged)


async def set_app_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """Merge ``patch`` into the stored settings row; returns the new merged state."""
    global _CACHE, _CACHE_TS
    async with async_session_factory() as session:
        result = await session.execute(select(AppSetting).where(AppSetting.key == _KEY))
        row = result.scalar_one_or_none()
        if row is None:
            row = AppSetting(key=_KEY, value={**patch})
            session.add(row)
        else:
            current = dict(row.value or {})
            current.update(patch)
            row.value = current
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(row, "value")
        await session.commit()

    _CACHE = {}
    _CACHE_TS = 0.0
    return await get_app_settings()


def invalidate_cache() -> None:
    """Drop the in-process cache — primarily for tests / startup migrations."""
    global _CACHE, _CACHE_TS
    _CACHE = {}
    _CACHE_TS = 0.0
