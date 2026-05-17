"""Assistant scheduler — turn-execution worker abstraction.

Today the orchestrator runs in-process via ``asyncio.create_task``. A future
deployment may want a separate worker pool (Arq/Celery/Dramatiq) for
horizontal scaling. The scheduler module is the seam: callers always go
through ``submit(job_id)`` and ``cancel(job_id)``; swap the implementation
later without touching call sites.

Also owns the in-process task registry so cancellation can interrupt the
running coroutine and orphan-task reconciliation can find live work.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

log = logging.getLogger(__name__)


# Per-job asyncio.Task registry. Used both for cooperative cancellation and
# for the startup reconciliation pass to know which tasks survived a restart
# (none — the dict is empty on cold start; everything in DB needs handling).
_tasks: dict[str, asyncio.Task] = {}

# Bounded concurrency: the orchestrator can spawn long-running LLM calls; on
# a small backend container, unbounded parallel turns would exhaust the
# event-loop and saturate the LLM provider. The semaphore is a soft cap —
# turns queue waiting for a slot rather than failing.
_MAX_CONCURRENT_TURNS = 8
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_TURNS)

# The coroutine factory that actually executes a turn. Set by the
# orchestrator on import to break a circular dependency.
_turn_runner: Callable[[str], Awaitable[None]] | None = None


def register_runner(runner: Callable[[str], Awaitable[None]]) -> None:
    """Register the turn-execution coroutine factory.

    Called once during application import — usually by the orchestrator
    module after it constructs its singleton instance.
    """
    global _turn_runner
    _turn_runner = runner


def submit(job_id: str) -> asyncio.Task:
    """Queue a turn for execution. Returns the live asyncio.Task.

    The task is registered so :func:`cancel` can interrupt it and the
    finalizer pops it on completion. ``register_runner`` must have been
    called before the first ``submit``.

    Idempotent: if a live (non-done) task already exists for ``job_id``,
    the existing task is returned and no new coroutine is started. This
    prevents double-submit races (e.g. a recovery re-submit racing with a
    user re-submit) from producing two parallel runners that would step on
    each other's DB writes.
    """
    if _turn_runner is None:
        raise RuntimeError("scheduler.register_runner() not called yet")

    existing = _tasks.get(job_id)
    if existing is not None and not existing.done():
        log.info("scheduler.submit: job=%s already running, reusing existing task", job_id)
        return existing

    async def _wrapped() -> None:
        async with _semaphore:
            try:
                await _turn_runner(job_id)
            except asyncio.CancelledError:
                # Re-raise so the asyncio task records cancellation; the
                # orchestrator already wrote the cancelled state to the DB.
                raise
            except Exception:
                # Last-resort guard — orchestrator handles its own errors,
                # but we don't want stray exceptions to crash the loop.
                log.exception("assistant scheduler: unhandled error job=%s", job_id)

    task = asyncio.create_task(_wrapped(), name=f"ra:turn:{job_id}")
    _tasks[job_id] = task

    def _cleanup(t: asyncio.Task) -> None:
        # Only pop if the registry still references THIS task — guards against
        # a race where submit() re-registered the same job_id with a fresh task
        # before this callback ran (would otherwise drop the new task).
        if _tasks.get(job_id) is t:
            _tasks.pop(job_id, None)

    task.add_done_callback(_cleanup)
    return task


def cancel(job_id: str) -> bool:
    """Request cooperative cancellation of an in-flight turn.

    Returns ``True`` when a live task existed and was asked to cancel,
    ``False`` when the job is unknown to this worker (e.g. it ran on a
    different process or was already drained).
    """
    task = _tasks.get(job_id)
    if not task or task.done():
        return False
    task.cancel()
    return True


def is_running(job_id: str) -> bool:
    """Return ``True`` iff a live in-process task exists for the job."""
    task = _tasks.get(job_id)
    return bool(task and not task.done())


def active_job_ids() -> list[str]:
    """Snapshot of currently running job ids — used for orphan reconciliation."""
    return [j for j, t in _tasks.items() if not t.done()]
