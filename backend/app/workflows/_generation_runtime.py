"""Shared runtime helpers for media-generation LangGraph workflows.

Eliminates duplication across the podcast and slides workflows by
centralising:

* :func:`load_source_content` — content loading via :class:`ContentLoaderService`
* :func:`queue_generation_job` — job-store + asyncio task plumbing
* :func:`run_with_recovery` — invokes a graph and handles all failure paths
                              (DB updates, JobStore updates, graceful logging)

Workflows now only declare their LangGraph and call these helpers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.core.tracking import current_user_id as _ctx_uid, set_workflow_context
from app.db.session import async_session_factory
from app.repositories.artifact import ArtifactRepository
from app.services.content_loader import ContentLoaderService, LoadedContent
from app.services.job_store import get_job_store

log = logging.getLogger(__name__)

# ── Active task registry ──────────────────────────────────────────────────────
# Maps artifact_id (str) → asyncio.Task so cancel_artifact can stop the
# background coroutine immediately. Single-process only; in multi-worker
# deployments the cancel falls back to DB/JobStore flagging.
_ACTIVE_TASKS: dict[str, "asyncio.Task[None]"] = {}


def cancel_generation_task(artifact_id: str) -> bool:
    """Cancel the asyncio task for the given artifact if it is still running.

    Returns True if a live task was found and cancelled, False otherwise.
    """
    task = _ACTIVE_TASKS.get(artifact_id)
    if task and not task.done():
        task.cancel()
        return True
    return False


# ── Source content loading ────────────────────────────────────────────────────


async def load_source_content(
    *,
    source_type: str,
    source_id: str,
    user_id: str,
    paper_ids: list[str] | None = None,
) -> LoadedContent:
    """Load content for a generation workflow.

    Centralised so all four workflows share identical loading semantics and
    folder ownership checks.

    For ``source_type="folder"``, runs the full LangGraph consolidation
    pipeline (coherence analysis + deep synthesis of related papers) unless
    ``paper_ids`` is provided, in which case only those papers are used.

    Args:
        source_type: ``"paper" | "capsule" | "folder"``.
        source_id: UUID string of the source entity.
        user_id: UUID string of the requesting user (required for folders).
        paper_ids: Optional list of paper UUIDs to include (folder only, max 5).

    Returns:
        :class:`LoadedContent`. ``ok=False`` indicates an empty or
        not-found source — the caller should surface this as an error.
    """
    if source_type == "folder":
        return await _load_folder_consolidated(
            folder_id=source_id,
            user_id=user_id,
            paper_ids=paper_ids,
        )

    async with async_session_factory() as db:
        loader = ContentLoaderService(db)
        return await loader.load(
            source_type=source_type,
            source_id=UUID(source_id),
            user_id=UUID(user_id),
        )


async def _load_folder_consolidated(
    folder_id: str,
    user_id: str,
    paper_ids: list[str] | None = None,
) -> LoadedContent:
    """Run the folder consolidation LangGraph pipeline and return a LoadedContent.

    Uses the deep synthesis (cross-paper analysis, complementary findings,
    combined results) rather than a simple concatenation of abstracts.
    """
    from app.workflows.folder_consolidation import run_folder_analysis

    log.info(
        "load_source_content: running folder consolidation folder=%s paper_ids=%s",
        folder_id, paper_ids,
    )

    result = await run_folder_analysis(
        folder_id=folder_id,
        user_id=user_id,
        paper_ids_override=(paper_ids or None),
    )

    if result.get("error") and not result.get("consolidated_content"):
        return LoadedContent(
            title="(folder load failed)",
            content="",
            ok=False,
        )

    papers = result.get("papers", [])
    report = result.get("coherence_report", {})
    titles = [p.get("title", "") for p in papers if p.get("title")]

    folder_title = f"Folder — {report.get('main_theme', f'{len(papers)} papers')}"

    return LoadedContent(
        title=folder_title,
        content=result.get("consolidated_content", ""),
        source_summary=report.get("synthesis_summary", ""),
        paper_count=len(papers),
        ok=bool(result.get("consolidated_content")),
    )


# ── Background job orchestration ──────────────────────────────────────────────


async def run_with_recovery(
    *,
    job_id: str,
    artifact_id: UUID,
    user_id: UUID,
    graph_invoker: Callable[[], Awaitable[None]],
    workflow_name: str,
) -> None:
    """Run a generation graph with full recovery, status tracking, and timing.

    Wraps a coroutine that runs the LangGraph workflow. Updates both the DB
    artifact row and the JobStore through the entire lifecycle. Catches
    every exception so the caller (an asyncio task) cannot crash the worker.

    Args:
        job_id: UUID string identifying this job in the JobStore.
        artifact_id: UUID of the GeneratedArtifact row in the DB.
        user_id: UUID of the artifact owner.
        graph_invoker: Async callable that invokes ``graph.ainvoke(...)``.
            All workflow-specific state should be captured by the closure.
        workflow_name: Workflow identifier for logging and token attribution.
    """
    _ctx_uid.set(user_id)
    set_workflow_context(workflow_name, "")
    store = get_job_store()
    t0 = time.monotonic()

    # Authoritative status in DB — mark_running is a no-op if already failed/cancelled
    async with async_session_factory() as db:
        repo = ArtifactRepository(db)
        await repo.mark_running(artifact_id)
        await db.commit()
        artifact_check = await repo.get_by_id(artifact_id)

    # Abort early if the job was cancelled before we started
    from app.models.artifact import ArtifactStatus as _ArtifactStatus  # local import avoids cycle
    if artifact_check and artifact_check.status == _ArtifactStatus.failed:
        log.info("%s job cancelled before start artifact=%s", workflow_name, artifact_id)
        return

    await store.update(job_id, {"status": "running"})

    try:
        await graph_invoker()
        # Note: the workflow's save_artifact node is responsible for transitioning
        # the DB row to completed/failed. We only update the JobStore-side status here.
        # We must look up the actual final DB state to mirror it.
        async with async_session_factory() as db:
            repo = ArtifactRepository(db)
            artifact = await repo.get_by_id(artifact_id)
            final_status = artifact.status.value if artifact else "completed"
        await store.update(job_id, {
            "status": final_status,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int((time.monotonic() - t0) * 1000),
        })
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        log.exception("%s workflow failed job=%s err=%s", workflow_name, job_id, exc)

        try:
            async with async_session_factory() as db:
                repo = ArtifactRepository(db)
                await repo.mark_failed(artifact_id, error_message=str(exc)[:500])
                await db.commit()
        except Exception as inner_exc:  # noqa: BLE001
            log.error("%s failed-state DB update also failed: %s", workflow_name, inner_exc)

        await store.update(job_id, {
            "status": "failed",
            "error": str(exc)[:500],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int((time.monotonic() - t0) * 1000),
        })


def queue_generation_job(
    *,
    artifact_id: UUID,
    user_id: UUID,
    source_type: str,
    source_id: str,
    expertise_level: str,
    orientation: str,
    generation_type: str,
    title: str,
    runner: Callable[[str], Awaitable[None]],
) -> str:
    """Queue a generation job: register it, kick off the asyncio task, return job_id.

    Args:
        artifact_id: UUID of the pre-created GeneratedArtifact row.
        user_id: UUID of the requesting user.
        source_type: ``paper`` | ``capsule`` | ``folder``.
        source_id: UUID string of the source entity.
        expertise_level: User expertise tag baked into generation.
        orientation: User orientation tag baked into generation.
        generation_type: ``podcast`` | ``slides``.
        title: Display title used in the notification panel.
        runner: Async callable that takes the ``job_id`` and runs the workflow.

    Returns:
        Job UUID string for status polling.
    """
    job_id = str(uuid4())
    payload = {
        "job_id": job_id,
        "artifact_id": str(artifact_id),
        "user_id": str(user_id),
        "source_type": source_type,
        "source_id": source_id,
        "expertise_level": expertise_level,
        "orientation": orientation,
        "generation_type": generation_type,
        "title": title,
        "status": "queued",
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "duration_ms": None,
    }

    artifact_id_str = str(artifact_id)

    async def _bootstrap() -> None:
        try:
            await get_job_store().put(job_id, payload)
        except Exception as exc:  # noqa: BLE001 — non-fatal, DB row is authoritative
            log.debug("queue_generation_job: job_store.put failed (continuing): %s", exc)
        try:
            await runner(job_id)
        except asyncio.CancelledError:
            log.info("generation task cancelled artifact=%s", artifact_id_str)
        finally:
            _ACTIVE_TASKS.pop(artifact_id_str, None)

    task = asyncio.create_task(_bootstrap(), name=f"gen:{generation_type}:{artifact_id}")
    _ACTIVE_TASKS[artifact_id_str] = task
    return job_id


# ── Recovery on startup ───────────────────────────────────────────────────────


async def recover_orphaned_artifacts() -> int:
    """Resume or fail artifacts left in ``running``/``queued`` state after a crash.

    For each orphaned artifact:
    - If a LangGraph checkpoint exists for its UUID, re-dispatch the workflow so
      it resumes from the last completed node (no wasted tokens).
    - If no checkpoint exists (crashed before the first node saved state),
      mark it ``failed`` so the user gets a retryable error instead of a
      spinner that never resolves.

    Returns:
        Number of artifacts processed (resumed + failed).
    """
    from sqlalchemy import or_, select, update
    from app.models.artifact import ArtifactStatus, GeneratedArtifact

    try:
        async with async_session_factory() as db:
            res = await db.execute(
                select(GeneratedArtifact).where(
                    or_(
                        GeneratedArtifact.status == ArtifactStatus.running,
                        GeneratedArtifact.status == ArtifactStatus.queued,
                    )
                )
            )
            orphans = list(res.scalars())
    except Exception as exc:  # noqa: BLE001
        log.warning("recover_orphaned_artifacts: DB read failed — %s", exc)
        return 0

    if not orphans:
        return 0

    # Initialise checkpointer so we can check for existing state
    try:
        from app.db.checkpointer import get_checkpointer
        checkpointer = await get_checkpointer()
    except Exception as exc:  # noqa: BLE001
        log.warning("recover_orphaned_artifacts: checkpointer unavailable, marking all failed — %s", exc)
        checkpointer = None

    resumed = 0
    failed = 0

    for artifact in orphans:
        artifact_id_str = str(artifact.id)
        has_checkpoint = False

        if checkpointer:
            try:
                has_checkpoint = await checkpointer.has_checkpoint(artifact_id_str)
            except Exception:  # noqa: BLE001
                pass

        if has_checkpoint:
            # Re-dispatch the workflow — LangGraph will resume from the last
            # checkpoint node, skipping already-completed (and already-paid) nodes.
            try:
                await _redispatch_artifact(artifact)
                resumed += 1
                log.info(
                    "recover: resuming artifact=%s type=%s",
                    artifact.id, artifact.generation_type,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("recover: re-dispatch failed for artifact=%s — %s", artifact.id, exc)
                # Fall through to mark failed
                has_checkpoint = False

        if not has_checkpoint:
            try:
                async with async_session_factory() as db:
                    await db.execute(
                        update(GeneratedArtifact)
                        .where(GeneratedArtifact.id == artifact.id)
                        .values(
                            status=ArtifactStatus.failed,
                            error_message="Worker restarted before generation could complete. Please retry.",
                            completed_at=datetime.now(timezone.utc),
                        )
                    )
                    await db.commit()
                failed += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("recover: mark-failed failed for artifact=%s — %s", artifact.id, exc)

    log.info(
        "recover_orphaned_artifacts: resumed=%d failed=%d total=%d",
        resumed, failed, len(orphans),
    )
    return len(orphans)


async def _redispatch_artifact(artifact) -> None:
    """Re-queue a generation workflow for an orphaned artifact.

    The LangGraph checkpointer will skip any nodes that already completed
    before the crash, resuming from the first incomplete node.
    """
    from app.models.artifact import GenerationType

    gen_type = artifact.generation_type
    artifact_id = artifact.id
    user_id = artifact.user_id
    source_type = artifact.source_type.value
    source_id = str(artifact.source_id)
    expertise = artifact.expertise_level or "practitioner"
    orientation = artifact.orientation or "both"
    title = (artifact.artifact_metadata or {}).get("source_title", "")

    job_id = str(uuid4())

    if gen_type == GenerationType.slides:
        from app.workflows.slides import _get_slides_graph
        from app.workflows.slides import SlidesState

        async def runner(_: str) -> None:
            graph = await _get_slides_graph()
            initial_state: SlidesState = {
                "artifact_id": str(artifact_id),
                "user_id": str(user_id),
                "source_type": source_type,
                "source_id": source_id,
                "expertise_level": expertise,
                "orientation": orientation,
                "title": "", "paper_content": "",
                "slide_plan": {}, "marp_markdown": "",
                "slide_batches": [], "blob_path": None,
                "error_metadata": {}, "paper_ids": None,
            }
            config = {"configurable": {"thread_id": str(artifact_id)}}
            await graph.ainvoke(initial_state, config=config)

    elif gen_type == GenerationType.podcast:
        from app.workflows.podcast import _get_podcast_graph
        from app.workflows.podcast import PodcastState

        async def runner(_: str) -> None:
            graph = await _get_podcast_graph()
            initial_state: PodcastState = {
                "artifact_id": str(artifact_id),
                "user_id": str(user_id),
                "source_type": source_type,
                "source_id": source_id,
                "expertise_level": expertise,
                "orientation": orientation,
                "title": "", "paper_content": "",
                "episode_plan": {}, "script": "",
                "segment_scripts": [], "utterances": [],
                "audio_bytes": None, "blob_path": None,
                "error_metadata": {}, "paper_ids": None,
            }
            config = {"configurable": {"thread_id": str(artifact_id)}}
            await graph.ainvoke(initial_state, config=config)

    else:
        raise ValueError(f"Unknown generation type: {gen_type}")

    # Re-mark as running so the UI shows activity
    async with async_session_factory() as db:
        from app.repositories.artifact import ArtifactRepository
        repo = ArtifactRepository(db)
        await repo.mark_running(artifact_id)
        await db.commit()

    # Register the job in the store so the frontend notifications panel shows it
    payload = {
        "job_id": job_id,
        "artifact_id": str(artifact_id),
        "user_id": str(user_id),
        "source_type": source_type,
        "source_id": source_id,
        "expertise_level": expertise,
        "orientation": orientation,
        "generation_type": gen_type.value,
        "title": title,
        "status": "running",
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "duration_ms": None,
    }
    try:
        await get_job_store().put(job_id, payload)
    except Exception as exc:  # noqa: BLE001 — non-fatal, DB row is authoritative
        log.debug("_redispatch_artifact: job_store.put failed (continuing): %s", exc)

    # Dispatch as background task — startup must never block on recovery execution.
    # run_with_recovery is a total-failure safety net and will never raise.
    asyncio.create_task(
        run_with_recovery(
            job_id=job_id,
            artifact_id=artifact_id,
            user_id=user_id,
            graph_invoker=lambda: runner(job_id),
            workflow_name=gen_type.value,
        ),
        name=f"recover:{gen_type.value}:{artifact_id}",
    )
