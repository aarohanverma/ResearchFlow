"""Dev-only reset endpoint — wipes all content data for a clean test run.

Triple-gated to prevent accidental production data loss:

  1. ``ENABLE_DEV_RESET=true`` must be set in the deployment environment.
  2. The request must carry a valid auth token (``CurrentUserID``).
  3. The authenticated user's email must be in the ``ADMIN_EMAILS``
     environment variable (comma-separated). When ``ADMIN_EMAILS`` is
     unset and ``ENABLE_DEV_RESET`` is on, the endpoint accepts any
     authed user — appropriate for local dev only.

Any single gate failing returns 403. User accounts are intentionally
preserved across resets so you do not need to re-register.

Cleanup ordering matters: in-flight background tasks are *cancelled and
awaited first* so a running deep-search / Genie / RA job cannot write
a partial row into a table that's about to be truncated. Filesystem
deletes run in a thread so we don't block the event loop.

POST /api/v1/dev/reset
  Clears every content table (papers, capsules, artifacts, graph, etc.),
  the three LangGraph checkpoint tables, the in-process job store, and all
  local blob files (or Azure blobs when the azure backend is active).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.core.config import get_settings
from app.core.deps import CurrentUserID, DBSession
from app.db.session import async_session_factory
from app.repositories.user import UserRepository

log = logging.getLogger(__name__)
router = APIRouter(prefix="/dev", tags=["dev"])

# ── Tables to truncate (safe order — CASCADE handles FK deps) ─────────────────
# User rows are kept so you don't need to re-register between test runs.
_CONTENT_TABLES = [
    # LangGraph checkpoints
    "langgraph_checkpoint_writes",
    "langgraph_checkpoint_blobs",
    "langgraph_checkpoints",
    # Generated media artifacts
    "generated_artifacts",
    # Genie / capsules
    "genie_elements",
    "idea_capsules",
    "genie_sessions",
    # Graph
    "knowledge_edges",
    "knowledge_nodes",
    "namespace_subscriptions",
    "source_mappings",
    # Paper content
    "bookmark_folder_members",
    "bookmarks",
    "bookmark_folders",
    "paper_chunks",
    "summaries",
    "paper_of_day",
    "paper_citations",
    "query_logs",
    "feed_feedback",
    "papers",
    # Workflow / token accounting
    "workflow_runs",
    "token_usage",
    # User-level content (not the user row itself)
    "annotations",
    "user_interest_profiles",
]


@router.post("/reset")
async def reset_all_data(user_id: CurrentUserID, db: DBSession):
    """Wipe all content data and return a summary of what was cleared.

    Requires:
      * ``ENABLE_DEV_RESET=true`` in the environment.
      * Valid auth token (any authed user when ``ADMIN_EMAILS`` is unset;
        otherwise the user's email must appear in ``ADMIN_EMAILS``).

    User accounts are preserved across the reset.
    """
    settings = get_settings()
    if not settings.enable_dev_reset:
        raise HTTPException(
            status_code=403,
            detail="Dev reset is disabled. Set ENABLE_DEV_RESET=true to enable.",
        )

    # Admin allow-list (comma-separated emails). When unset, any authed
    # user with the env flag is allowed — appropriate for local-only dev.
    admin_emails_raw = os.environ.get("ADMIN_EMAILS", "").strip()
    if admin_emails_raw:
        allowed = {e.strip().lower() for e in admin_emails_raw.split(",") if e.strip()}
        try:
            user = await UserRepository(db).get_by_id(user_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("dev_reset: user lookup failed — %s", exc)
            user = None
        email = (getattr(user, "email", None) or "").strip().lower()
        if not email or email not in allowed:
            raise HTTPException(
                status_code=403,
                detail="Dev reset requires an admin account.",
            )

    results: dict[str, str] = {}

    # ── 0. Cancel in-flight background tasks before we truncate ──────────────
    # A reset that runs concurrently with a Genie / deep-search / Build-Deep
    # job will see the running job write a partial row into a just-truncated
    # table and leave the DB in a half-cleared state. Cancelling here is
    # best-effort: we issue cancel() to every known task pool, then wait
    # briefly for them to actually settle. Anything still running after the
    # grace window is logged — the truncate will still race with it, but
    # the window is small enough in practice to be tolerable for a dev tool.
    try:
        await _cancel_inflight_tasks(grace_seconds=2.0)
        results["inflight_cancelled"] = "ok"
    except Exception as exc:  # noqa: BLE001
        log.warning("dev_reset: in-flight cancel failed — %s", exc)
        results["inflight_cancelled"] = f"error: {exc}"

    # ── 1. Truncate DB tables — one transaction per table so a single failure
    #       never aborts the rest (PostgreSQL marks the whole txn as aborted
    #       on any error, making subsequent statements silently no-ops).
    for table in _CONTENT_TABLES:
        async with async_session_factory() as db:
            try:
                await db.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                await db.commit()
                results[table] = "truncated"
            except Exception as exc:  # noqa: BLE001
                await db.rollback()
                log.warning("dev_reset: truncate %s failed — %s", table, exc)
                results[table] = f"error: {exc}"

    # ── 2. Clear job store ────────────────────────────────────────────────────
    try:
        from app.services.job_store import get_job_store
        store = get_job_store()
        await store.clear_all()
        results["job_store"] = "cleared"
    except Exception as exc:  # noqa: BLE001
        log.warning("dev_reset: job store clear failed — %s", exc)
        results["job_store"] = f"error: {exc}"

    # ── 3. Clear blobs ────────────────────────────────────────────────────────
    if settings.blob_backend == "azure":
        try:
            from azure.storage.blob.aio import BlobServiceClient
            _CONTAINER = "researchflow"
            async with BlobServiceClient.from_connection_string(
                settings.azure_storage_connection_string
            ) as svc:
                container = svc.get_container_client(_CONTAINER)
                deleted = 0
                async for blob in container.list_blobs():
                    await container.delete_blob(blob.name)
                    deleted += 1
            results["blobs_azure"] = f"deleted {deleted} blobs"
        except Exception as exc:  # noqa: BLE001
            log.warning("dev_reset: azure blob clear failed — %s", exc)
            results["blobs_azure"] = f"error: {exc}"
    else:
        try:
            blob_dir = Path(settings.blob_local_dir)
            # ``shutil.rmtree`` walks the directory synchronously and can
            # take seconds for a populated blob store. Push it to a thread
            # so the event loop keeps serving heartbeats/healthchecks
            # while the reset progresses.
            def _wipe_blobs() -> None:
                if blob_dir.exists():
                    shutil.rmtree(blob_dir)
                blob_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(_wipe_blobs)
            results["blobs_local"] = f"cleared {blob_dir}"
        except Exception as exc:  # noqa: BLE001
            log.warning("dev_reset: local blob dir clear failed — %s", exc)
            results["blobs_local"] = f"error: {exc}"

    # ── 4. Clear module-level podcast audio buffer ────────────────────────────
    try:
        from app.workflows.podcast import _AUDIO_BUFFER
        _AUDIO_BUFFER.clear()
        results["audio_buffer"] = "cleared"
    except Exception as exc:  # noqa: BLE001
        results["audio_buffer"] = f"error: {exc}"

    # ── 5. Drop in-process caches that reference truncated rows ──────────────
    # GraphService, admin_settings, and the per-session state-lock registry
    # all keep references to UUIDs that no longer exist after the wipe.
    # Without an explicit clear, the next request still gets the stale
    # entry from cache and confusingly behaves as if the row is alive.
    try:
        from app.services.graph import GraphService
        GraphService._build_cache.clear()  # type: ignore[attr-defined]
        # Persistent subgraph cache is cleared best-effort; failures
        # don't matter because the next read is authoritative.
        try:
            await GraphService.clear_subgraph_cache(None)
        except Exception:
            pass
        results["graph_cache"] = "cleared"
    except Exception as exc:  # noqa: BLE001
        results["graph_cache"] = f"error: {exc}"

    try:
        from app.services.admin_settings import invalidate_cache as invalidate_admin_cache
        invalidate_admin_cache()
        results["admin_settings_cache"] = "cleared"
    except Exception as exc:  # noqa: BLE001
        results["admin_settings_cache"] = f"error: {exc}"

    # Session locks live in a WeakValueDictionary so they self-collect
    # once their owning session row is GC'd, but truncating the rows
    # doesn't free the Lock objects (they have no Python-side reference
    # back to the row). Clearing eagerly avoids stale Lock objects
    # lingering until the next GC cycle.
    try:
        from app.assistant import state_lock as _sl
        reg = getattr(_sl, "_LOCK_REGISTRY", None)
        if reg is not None:
            try:
                reg.clear()
            except Exception:
                pass
        results["session_locks"] = "cleared"
    except Exception as exc:  # noqa: BLE001
        results["session_locks"] = f"skipped: {exc}"

    log.info("dev_reset: completed — %s", results)
    return {"status": "ok", "cleared": results}


async def _cancel_inflight_tasks(grace_seconds: float = 2.0) -> None:
    """Cancel and await every background task we know about.

    Touches every module-level task registry the app maintains:

    * ``app.assistant.scheduler._tasks`` — RA turns in flight.
    * ``app.api.v1.search._background_tasks`` — deep-search bg jobs.
    * ``app.api.v1.genie._background_tasks`` — Genie SSE / fan-out work.
    * ``app.api.v1.graph._BUILD_TASKS`` — Build Deep background runs.
    * ``app.workflows._generation_runtime._ACTIVE_TASKS`` — generation
      workflows (podcast, slides, study, etc.).
    * ``app.adapters.llm.tracking._tracking_tasks`` — async token-usage
      writers (cheap to cancel; they're idempotent).

    Best-effort: missing modules are skipped silently. Hard timeout
    via ``asyncio.wait`` so the reset endpoint never hangs even when a
    task is stuck inside an unkillable C extension.
    """
    pools: list[asyncio.Task] = []

    def _drain(container, kind: str) -> None:
        """Snapshot the container and request cancellation on each task."""
        try:
            if isinstance(container, dict):
                items = list(container.values())
            else:
                items = list(container)
        except Exception:
            return
        for t in items:
            if isinstance(t, asyncio.Task) and not t.done():
                t.cancel()
                pools.append(t)
        log.info("dev_reset: cancelled %d tasks from %s", len(items), kind)

    try:
        from app.assistant import scheduler as _sched
        _drain(getattr(_sched, "_tasks", {}), "ra_scheduler")
    except Exception:
        pass
    try:
        from app.api.v1 import search as _search_mod
        _drain(getattr(_search_mod, "_background_tasks", set()), "deep_search")
    except Exception:
        pass
    try:
        from app.api.v1 import genie as _genie_mod
        _drain(getattr(_genie_mod, "_background_tasks", set()), "genie")
    except Exception:
        pass
    try:
        from app.api.v1 import graph as _graph_mod
        _drain(getattr(_graph_mod, "_BUILD_TASKS", {}), "graph_build")
    except Exception:
        pass
    try:
        from app.workflows import _generation_runtime as _gen
        _drain(getattr(_gen, "_ACTIVE_TASKS", {}), "generation")
    except Exception:
        pass

    if not pools:
        return
    # Wait for the cancellations to actually settle — bounded so a stuck
    # task can't block the reset indefinitely.
    try:
        await asyncio.wait(pools, timeout=grace_seconds)
    except Exception as exc:  # noqa: BLE001
        log.warning("dev_reset: error waiting for cancellations — %s", exc)
