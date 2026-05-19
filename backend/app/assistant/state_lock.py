"""Per-session locks for ``AssistantSession.state`` read-modify-write.

Multiple concurrent paths inside one Python worker can mutate the same
session's ``state`` JSONB (auto-memory consolidation, branch-summary
roll-up, telemetry append, memory-embedding cache writes, prune passes).
SQLAlchemy ORM does a full-object UPDATE on ``state``, so concurrent
read-modify-writes lose updates without explicit serialisation.

This module hands out a two-layer lock:

* **Intra-process** — an ``asyncio.Lock`` keyed by session_id, stored
  in a ``WeakValueDictionary`` so unused locks are garbage-collected.
  Always active.

* **Inter-process** — a PostgreSQL session-scoped advisory lock keyed
  by a 64-bit hash of the session_id. Active when more than one worker
  could be serving the same session (anything other than a single-
  worker local dev process). Released on context exit.

Together they survive the local → cloud transition. Single-worker dev
keeps the cheap in-process path; multi-worker deployments automatically
get cross-process serialisation without code changes — just set
``ENVIRONMENT`` away from ``local`` (or set ``STATE_LOCK_PG_ADVISORY=1``
explicitly).

Usage
-----

    async with session_state_lock(session_id):
        # safe read-modify-write on session.state
        ...

The helper is a coroutine context manager so it composes cleanly with
``async with async_session_factory() as db`` blocks. Both layers have
bounded acquisition timeouts so a stuck holder surfaces as a logged
warning rather than a silent deadlock.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import weakref
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from sqlalchemy import text as _sql_text

from app.core.config import settings
from app.db.session import async_session_factory

log = logging.getLogger(__name__)


_LOCK_REGISTRY: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()
_REGISTRY_MUTEX = asyncio.Lock()
_ACQUIRE_TIMEOUT_S = 30.0
_PG_LOCK_TIMEOUT_MS = 30_000  # PostgreSQL lock_timeout in milliseconds


def _pg_advisory_enabled() -> bool:
    """Whether to engage the PostgreSQL advisory lock layer.

    Auto-on for non-local environments (cloud deployments typically
    spawn multiple workers). Operators can force it on/off explicitly
    via ``STATE_LOCK_PG_ADVISORY=1|0`` for ops/debugging.
    """
    forced = os.environ.get("STATE_LOCK_PG_ADVISORY", "").strip()
    if forced in {"1", "true", "True", "yes"}:
        return True
    if forced in {"0", "false", "False", "no"}:
        return False
    return settings.environment != "local"


def _key_for(session_id: Any) -> str:
    if isinstance(session_id, UUID):
        return str(session_id)
    return str(session_id or "")


def _pg_lock_key(session_id: Any) -> int:
    """Stable signed-64-bit advisory lock key for a session_id.

    ``pg_advisory_lock(bigint)`` requires a signed 64-bit integer, so we
    take the top 8 bytes of a SHA-256 of the UUID string and clamp into
    the signed range. Collision risk between distinct sessions is
    negligible (~2⁻⁶³) and bounded to "two unrelated sessions briefly
    serialise" — a perf nudge, never a correctness issue.
    """
    digest = hashlib.sha256(_key_for(session_id).encode("utf-8")).digest()[:8]
    n = int.from_bytes(digest, "big", signed=False)
    if n >= (1 << 63):
        n -= (1 << 64)
    return n


async def _get_lock(session_id: Any) -> asyncio.Lock:
    key = _key_for(session_id)
    async with _REGISTRY_MUTEX:
        lock = _LOCK_REGISTRY.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _LOCK_REGISTRY[key] = lock
    return lock


@asynccontextmanager
async def _pg_advisory_lock(session_id: Any):
    """Hold a session-scoped PostgreSQL advisory lock for cross-process serialisation.

    On infrastructure failure (DB unavailable, lock timeout) the manager
    logs and yields without the lock — the in-process layer remains in
    force, and the worst case (multi-worker concurrent writes during a
    DB hiccup) degrades to the documented last-writer-wins behaviour.
    """
    if not _pg_advisory_enabled():
        yield
        return

    key = _pg_lock_key(session_id)
    acquired_via: Any | None = None
    try:
        async with async_session_factory() as conn_session:
            # Bound the wait so a stuck holder can't deadlock the worker.
            try:
                await conn_session.execute(
                    _sql_text(f"SET LOCAL lock_timeout = {_PG_LOCK_TIMEOUT_MS}")
                )
                await conn_session.execute(
                    _sql_text("SELECT pg_advisory_lock(:k)"),
                    {"k": key},
                )
                acquired_via = conn_session
            except Exception as exc:
                log.warning(
                    "session_state_lock: pg advisory acquire failed key=%s: %s",
                    key, exc,
                )
                acquired_via = None
            try:
                yield
            finally:
                if acquired_via is not None:
                    try:
                        await conn_session.execute(
                            _sql_text("SELECT pg_advisory_unlock(:k)"),
                            {"k": key},
                        )
                    except Exception as exc:
                        log.debug(
                            "session_state_lock: pg advisory release failed key=%s: %s",
                            key, exc,
                        )
    except Exception as exc:
        # Catastrophic failure obtaining a DB session — degrade gracefully.
        log.warning("session_state_lock: pg layer unavailable: %s", exc)
        yield


@asynccontextmanager
async def session_state_lock(session_id: Any):
    """Acquire the per-session state lock (in-process + optional cross-process).

    On in-process timeout, logs and yields anyway so the caller's work
    doesn't stall indefinitely. The lost-update risk in that degenerate
    case is documented in ``STATE_OWNERSHIP.md``.
    """
    key = _key_for(session_id)
    if not key:
        # No session to lock — degenerate path; just yield.
        yield
        return
    lock = await _get_lock(session_id)
    acquired = False
    try:
        try:
            await asyncio.wait_for(lock.acquire(), timeout=_ACQUIRE_TIMEOUT_S)
            acquired = True
        except asyncio.TimeoutError:
            log.warning(
                "session_state_lock: timeout acquiring in-process lock for "
                "session=%s; proceeding without serialisation "
                "(lost-update risk this turn)",
                key,
            )
        async with _pg_advisory_lock(session_id):
            yield
    finally:
        if acquired:
            try:
                lock.release()
            except RuntimeError:
                log.debug("session_state_lock: double-release on %s", key)


__all__ = ["session_state_lock"]
