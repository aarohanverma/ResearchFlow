"""Dev-only reset endpoint — wipes all content data for a clean test run.

Guarded by ENABLE_DEV_RESET=true in the environment. Returns 403 if the flag
is not set, so it is safe to deploy with the flag off.

POST /api/v1/dev/reset
  Clears every content table (papers, capsules, artifacts, graph, etc.),
  the three LangGraph checkpoint tables, the in-process job store, and all
  local blob files (or Azure blobs when the azure backend is active).
  User accounts are intentionally preserved so you do not need to re-register.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import async_session_factory

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
async def reset_all_data():
    """Wipe all content data and return a summary of what was cleared.

    Requires ENABLE_DEV_RESET=true in the environment. User accounts are
    preserved. Useful for a clean-slate test run without re-registering.
    """
    settings = get_settings()
    if not settings.enable_dev_reset:
        raise HTTPException(
            status_code=403,
            detail="Dev reset is disabled. Set ENABLE_DEV_RESET=true to enable.",
        )

    results: dict[str, str] = {}

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
            if blob_dir.exists():
                shutil.rmtree(blob_dir)
                blob_dir.mkdir(parents=True, exist_ok=True)
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

    log.info("dev_reset: completed — %s", results)
    return {"status": "ok", "cleared": results}
