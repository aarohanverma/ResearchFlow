"""Startup reconciliation for orphaned assistant turns.

When the backend process restarts mid-turn, asyncio tasks die but the DB
rows stay as ``running`` / ``pending`` forever. ``reconcile_orphans``
sweeps those rows on startup:

* If the user already requested cancellation, mark cancelled.
* If a recent task can plausibly be resumed, requeue it via the scheduler
  (the orchestrator's idempotent step-replay handles skipping completed
  steps; see :func:`Orchestrator._already_completed_steps`).
* Otherwise mark it failed with ``error="process restarted"`` so the
  notification panel doesn't keep spinning forever.

Runs once per process during the FastAPI lifespan startup event.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select

from app.assistant import scheduler
from app.db.session import async_session_factory
from app.models.assistant import AssistantTask, AssistantTaskStatus
from app.repositories.assistant import AssistantRepository
from app.services.job_store import get_job_store

log = logging.getLogger(__name__)


# How long after creation an in-flight task is still safe to resume. Older
# tasks are marked failed instead of replayed — they probably hit a real
# bug rather than just a process restart.
_RESUME_AGE_LIMIT = timedelta(hours=2)


async def reconcile_orphans() -> dict[str, int]:
    """Scan running/pending tasks at startup and either resume or fail them.

    Returns:
        Counts dict like ``{"resumed": int, "failed": int, "cancelled": int}``
        for the startup log.
    """
    counts = {"resumed": 0, "failed": 0, "cancelled": 0}
    async with async_session_factory() as db:
        result = await db.execute(
            select(AssistantTask).where(
                or_(
                    AssistantTask.status == AssistantTaskStatus.running,
                    AssistantTask.status == AssistantTaskStatus.pending,
                )
            )
        )
        orphans = list(result.scalars())

    if not orphans:
        return counts

    log.info("assistant recovery: %d orphaned task(s) to reconcile", len(orphans))
    now = datetime.now(timezone.utc)
    for task in orphans:
        # Honour pending cancellation even after a restart.
        if task.cancel_requested_at:
            await _mark_cancelled(task.job_id)
            counts["cancelled"] += 1
            continue

        created = task.created_at
        if created and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age = now - created if created else _RESUME_AGE_LIMIT
        if age > _RESUME_AGE_LIMIT:
            await _mark_failed(task.job_id, "Orphaned by process restart (too old to resume)")
            counts["failed"] += 1
            continue

        # Re-submit. The orchestrator skips completed steps on replay.
        # Seed a job store entry first so progress updates emitted by the
        # orchestrator (via get_job_store().update()) are not silently dropped.
        try:
            created_iso = (
                task.created_at.isoformat()
                if task.created_at
                else datetime.now(timezone.utc).isoformat()
            )
            await get_job_store().put(task.job_id, {
                "kind": "assistant",
                "job_id": task.job_id,
                "user_id": str(task.user_id),
                "session_id": str(task.session_id),
                "assistant_message_id": str(task.assistant_message_id) if task.assistant_message_id else None,
                "task_id": str(task.id),
                "title": task.title,
                "status": "running",
                "namespace_key": task.namespace_key,
                "created_at": created_iso,
                "completed_at": None,
                "summary": "Resumed after process restart",
            })
            scheduler.submit(task.job_id)
            counts["resumed"] += 1
            log.info("assistant recovery: resumed job=%s", task.job_id)
        except Exception:
            log.exception("assistant recovery: resume failed job=%s", task.job_id)
            await _mark_failed(task.job_id, "Failed to resume after process restart")
            counts["failed"] += 1
    return counts


async def _mark_cancelled(job_id: str) -> None:
    async with async_session_factory() as db:
        repo = AssistantRepository(db)
        await repo.update_task(
            job_id,
            status=AssistantTaskStatus.cancelled,
            progress={"stage": "cancelled", "percent": 100,
                      "summary": "Cancelled before process restart"},
            completed=True,
        )
        await db.commit()
    await get_job_store().update(job_id, {"status": "cancelled", "summary": "Cancelled"})


async def _mark_failed(job_id: str, reason: str) -> None:
    async with async_session_factory() as db:
        repo = AssistantRepository(db)
        await repo.update_task(
            job_id,
            status=AssistantTaskStatus.failed,
            progress={"stage": "failed", "percent": 100, "summary": reason},
            error=reason,
            completed=True,
        )
        await db.commit()
    await get_job_store().update(job_id, {"status": "failed", "summary": reason})
