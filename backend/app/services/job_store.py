"""JobStore — thin abstraction over background job state.

The four media generation workflows previously each maintained their own
``_jobs: dict[str, dict]`` module-level state.  That violated DRY, made
horizontal scaling impossible (state isolated per worker), and lost all
in-flight jobs on worker restart.

This module replaces those per-workflow dicts with a single :class:`JobStore`
abstraction that has two adapters:

* :class:`InMemoryJobStore` — default for local dev, in-process only.
* :class:`RedisJobStore`    — cloud-friendly; enabled when ``CACHE_BACKEND=redis``.

The artifact's authoritative state lives in the :class:`GeneratedArtifact`
DB table; the JobStore is purely a fast-path notification cache and does
not need to be durable. On worker restart, callers consult the DB row
(which has ``running``/``failed`` status) rather than the JobStore.

DESIGN PRINCIPLE — the JobStore is OPTIONAL infrastructure: every consumer
must work correctly when the JobStore is empty or stale, falling back to
the DB.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings

log = logging.getLogger(__name__)


# ── Adapter ABC ───────────────────────────────────────────────────────────────


class JobStore(ABC):
    """Pluggable storage for background-job presence + lightweight progress data.

    Implementations must be thread-safe / async-safe.  Methods MUST never
    raise on the happy path; on infrastructure failure they should log and
    return safe defaults so generation workflows are not derailed by a
    Redis blip.
    """

    @abstractmethod
    async def put(self, job_id: str, payload: dict[str, Any]) -> None:
        """Store or replace a job record."""

    @abstractmethod
    async def get(self, job_id: str) -> dict[str, Any] | None:
        """Return a job record by ID, or ``None`` if not found."""

    @abstractmethod
    async def update(self, job_id: str, patch: dict[str, Any]) -> None:
        """Merge ``patch`` into the existing record. No-op if not found."""

    @abstractmethod
    async def list_by_user(self, user_id: str) -> list[dict[str, Any]]:
        """Return all known jobs for a user, newest-first."""

    @abstractmethod
    async def delete(self, job_id: str) -> None:
        """Remove a job record. Idempotent."""

    async def clear_all(self) -> None:
        """Remove every job record. Used by the dev reset endpoint."""


# ── In-memory adapter (default) ───────────────────────────────────────────────


class InMemoryJobStore(JobStore):
    """Process-local job store backed by a plain dict.

    Acceptable for single-worker local dev.  In production deployments use
    :class:`RedisJobStore` so workers share state and survives restarts via
    the DB-backed authoritative status.

    Eviction: completed and failed jobs older than ``_COMPLETED_TTL_S`` seconds
    are lazily evicted on every ``put`` call so the store never grows without
    bound in long-running processes.
    """

    _COMPLETED_TTL_S = 3600      # evict terminal jobs after 1 h
    _MAX_JOBS = 2000             # hard cap; evict oldest terminal jobs when exceeded

    def __init__(self) -> None:
        """Initialise an empty in-memory job store with an asyncio lock."""
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def _evict_stale(self) -> None:
        """Remove terminal (completed/failed) jobs older than _COMPLETED_TTL_S.

        Called inside the lock so callers don't need to worry about it.
        Also enforces _MAX_JOBS by removing the oldest terminal entries when
        the store is over capacity.
        """
        if len(self._jobs) < self._MAX_JOBS // 2:
            return  # nothing to do — avoid the iteration overhead

        cutoff = datetime.now(timezone.utc).timestamp() - self._COMPLETED_TTL_S
        terminal_statuses = {"completed", "failed", "cancelled", "done"}

        to_delete: list[str] = []
        for jid, job in self._jobs.items():
            status = job.get("status", "")
            if status not in terminal_statuses:
                continue
            finished = job.get("finished_at") or job.get("completed_at")
            if not finished:
                continue
            try:
                ts = datetime.fromisoformat(
                    finished.replace("Z", "+00:00")
                ).timestamp()
                if ts < cutoff:
                    to_delete.append(jid)
            except (ValueError, AttributeError):
                pass

        for jid in to_delete:
            del self._jobs[jid]

        # Hard cap: if still over limit, evict oldest terminal entries
        if len(self._jobs) > self._MAX_JOBS:
            terminal = sorted(
                (
                    (jid, j.get("created_at", ""))
                    for jid, j in self._jobs.items()
                    if j.get("status", "") in terminal_statuses
                ),
                key=lambda x: x[1],
            )
            for jid, _ in terminal[: len(self._jobs) - self._MAX_JOBS]:
                self._jobs.pop(jid, None)

    async def put(self, job_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            self._jobs[job_id] = dict(payload)
            self._evict_stale()

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async with self._lock:
            entry = self._jobs.get(job_id)
            return dict(entry) if entry else None

    async def update(self, job_id: str, patch: dict[str, Any]) -> None:
        async with self._lock:
            existing = self._jobs.get(job_id)
            if existing is None:
                return
            existing.update(patch)

    async def list_by_user(self, user_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            jobs = [dict(j) for j in self._jobs.values() if j.get("user_id") == user_id]
        return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            self._jobs.pop(job_id, None)

    async def clear_all(self) -> None:
        async with self._lock:
            self._jobs.clear()


# ── Redis adapter ─────────────────────────────────────────────────────────────


class RedisJobStore(JobStore):
    """Redis-backed job store — cloud-friendly, multi-worker safe.

    Keys used:

    * ``rf:jobs:{job_id}``        — JSON-encoded job payload (TTL 24h).
    * ``rf:jobs:user:{user_id}``  — Sorted set of ``job_id`` -> ``created_at_epoch``.

    All Redis operations are wrapped in try/except — on any infrastructure
    failure we log and fall back to silent no-op so the workflow proceeds
    using the DB as the authoritative store.

    Args:
        redis_url: redis:// connection string.  Defaults to ``settings.redis_url``.
        ttl_seconds: How long to retain finished jobs.  Defaults to 24h.
    """

    _KEY_PREFIX = "rf:jobs"
    _USER_INDEX_PREFIX = "rf:jobs:user"

    def __init__(self, redis_url: str | None = None, ttl_seconds: int = 86400) -> None:
        self._redis_url = redis_url or settings.redis_url
        self._ttl = ttl_seconds
        self._client: Any | None = None  # lazy

    async def _get_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            import redis.asyncio as redis_async
            self._client = redis_async.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
            return self._client
        except Exception as exc:  # noqa: BLE001 — degrade silently
            log.warning("RedisJobStore: cannot connect to %s — %s", self._redis_url, exc)
            return None

    def _key(self, job_id: str) -> str:
        return f"{self._KEY_PREFIX}:{job_id}"

    def _user_key(self, user_id: str) -> str:
        return f"{self._USER_INDEX_PREFIX}:{user_id}"

    async def put(self, job_id: str, payload: dict[str, Any]) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.setex(self._key(job_id), self._ttl, json.dumps(payload))
            user_id = str(payload.get("user_id") or "")
            if user_id:
                created_at = payload.get("created_at") or datetime.now(timezone.utc).isoformat()
                # Use timestamp for ordering — newer at higher score.
                score = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp() \
                    if isinstance(created_at, str) else 0
                await client.zadd(self._user_key(user_id), {job_id: score})
                await client.expire(self._user_key(user_id), self._ttl)
        except Exception as exc:  # noqa: BLE001 — non-fatal
            log.debug("RedisJobStore.put failed: %s", exc)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        client = await self._get_client()
        if client is None:
            return None
        try:
            raw = await client.get(self._key(job_id))
            return json.loads(raw) if raw else None
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisJobStore.get failed: %s", exc)
            return None

    async def update(self, job_id: str, patch: dict[str, Any]) -> None:
        existing = await self.get(job_id)
        if existing is None:
            return
        existing.update(patch)
        await self.put(job_id, existing)

    async def list_by_user(self, user_id: str) -> list[dict[str, Any]]:
        client = await self._get_client()
        if client is None:
            return []
        try:
            ids = await client.zrevrange(self._user_key(user_id), 0, 99)
            if not ids:
                return []
            # Bulk fetch all job payloads in one MGET round-trip instead of N GETs
            keys = [self._key(jid) for jid in ids]
            raw_values = await client.mget(*keys)
            results: list[dict[str, Any]] = []
            for raw in raw_values:
                if raw is not None:
                    try:
                        results.append(json.loads(raw))
                    except (json.JSONDecodeError, TypeError):
                        pass
            return results
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisJobStore.list_by_user failed: %s", exc)
            return []

    async def delete(self, job_id: str) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.delete(self._key(job_id))
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisJobStore.delete failed: %s", exc)

    async def clear_all(self) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            keys = await client.keys(f"{self._KEY_PREFIX}:*")
            if keys:
                await client.delete(*keys)
        except Exception as exc:  # noqa: BLE001
            log.debug("RedisJobStore.clear_all failed: %s", exc)


# ── Factory + module singleton ────────────────────────────────────────────────

_singleton: JobStore | None = None


def get_job_store() -> JobStore:
    """Return the configured global :class:`JobStore` singleton.

    Selection logic mirrors the cache backend:
    - ``CACHE_BACKEND=redis`` → :class:`RedisJobStore`
    - otherwise              → :class:`InMemoryJobStore`

    The first call instantiates the store; subsequent calls return the same
    instance to preserve the in-memory state across the process lifetime.
    """
    global _singleton
    if _singleton is None:
        if settings.cache_backend == "redis":
            _singleton = RedisJobStore()
            log.info("JobStore: using RedisJobStore at %s", settings.redis_url)
        else:
            _singleton = InMemoryJobStore()
            log.info("JobStore: using InMemoryJobStore (single-worker)")
    return _singleton


def reset_job_store_for_tests() -> None:
    """Reset the JobStore singleton (test-only helper)."""
    global _singleton
    _singleton = None
