"""Generation router — background media generation for papers, capsules, folders.

Endpoints:
  POST /{source_type}/{source_id}/{generation_type}   — trigger generation
  GET  /artifact/{artifact_id}                         — poll status / result
  GET  /{source_type}/{source_id}                      — list all artifacts for source
  GET  /jobs                                           — list user's in-progress jobs
  DELETE /artifact/{artifact_id}                       — delete artifact + blob

Cache strategy:
  Returns the most recent ``completed`` artifact when ALL of (source, type,
  expertise, orientation, provider, model, parser) match the current
  generation context. Otherwise creates a new artifact.
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.config import settings
from app.core.deps import CurrentUserID, DBSession
from app.models.artifact import ArtifactStatus, GenerationType, SourceType
from app.repositories.artifact import ArtifactRepository
from app.repositories.user import UserRepository
from app.services.job_store import get_job_store
from app.workflows._generation_runtime import cancel_generation_task

log = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generate"])


# ── Response schemas ──────────────────────────────────────────────────────────


class ArtifactResponse(BaseModel):
    """Public representation of a generated artifact."""

    id: str
    generation_type: str
    source_type: str
    source_id: str
    source_title: str          # human-readable name of the source entity
    status: str
    blob_path: str | None
    content: dict | None
    expertise_level: str | None
    orientation: str | None
    provider: str | None
    model_used: str | None
    parser_used: str | None
    input_tokens: int
    output_tokens: int
    generation_duration_ms: int
    error_message: str | None
    created_at: str
    completed_at: str | None

    model_config = {"from_attributes": True}


class TriggerResponse(BaseModel):
    """Returned when a generation job is successfully queued."""

    artifact_id: str
    job_id: str
    status: str = "queued"
    message: str
    source_title: str = ""     # human-readable source entity name for the notification panel


class JobsListResponse(BaseModel):
    """Listing of in-progress generation jobs for the current user."""

    jobs: list[dict]
    total: int


# ── Helpers ────────────────────────────────────────────────────────────────────

_VALID_SOURCE_TYPES = {"paper", "capsule"}
# The only user-facing media types.
_VALID_GEN_TYPES = {"podcast", "slides"}


def _to_gen_type_strict(value: str) -> GenerationType:
    """Strictly accept only the two active media types."""
    if value not in _VALID_GEN_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid generation_type '{value}'. Must be one of: {sorted(_VALID_GEN_TYPES)}",
        )
    return GenerationType(value)


def _to_source_type(value: str) -> SourceType:
    try:
        return SourceType(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid source_type '{value}'. Must be one of: {_VALID_SOURCE_TYPES}",
        ) from exc


def _to_gen_type(value: str) -> GenerationType:
    if value not in _VALID_GEN_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid generation_type '{value}'. Must be one of: {sorted(_VALID_GEN_TYPES)}",
        )
    try:
        return GenerationType(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid generation_type '{value}'. Must be one of: {_VALID_GEN_TYPES}",
        ) from exc


def _parse_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {field_name} UUID: {value}") from exc


def _models_for_generation(gen_type: GenerationType) -> tuple[str, str]:
    """Return ``(provider_id, model_used)`` for a generation type.

    Mirrors the model selection done inside each workflow's ``save_artifact``
    node. Used to build the cache lookup signature so cache hits only occur
    when the configured model is the same as what was used to generate the
    cached artifact.
    """
    provider = settings.default_llm_provider
    if gen_type == GenerationType.podcast:
        return provider, settings.default_quality_model
    if gen_type == GenerationType.slides:
        return provider, settings.default_reasoning_model
    return provider, settings.default_quality_model


def _artifact_to_response(artifact) -> ArtifactResponse:
    source_title = ""
    if isinstance(artifact.artifact_metadata, dict):
        source_title = artifact.artifact_metadata.get("source_title", "")
    return ArtifactResponse(
        id=str(artifact.id),
        generation_type=artifact.generation_type.value,
        source_type=artifact.source_type.value,
        source_id=str(artifact.source_id),
        source_title=source_title,
        status=artifact.status.value,
        blob_path=artifact.blob_path,
        content=artifact.content,
        expertise_level=artifact.expertise_level,
        orientation=artifact.orientation,
        provider=artifact.provider,
        model_used=artifact.model_used,
        parser_used=artifact.parser_used,
        input_tokens=artifact.input_tokens,
        output_tokens=artifact.output_tokens,
        generation_duration_ms=artifact.generation_duration_ms,
        error_message=artifact.error_message,
        created_at=artifact.created_at.isoformat(),
        completed_at=artifact.completed_at.isoformat() if artifact.completed_at else None,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post("/folder/{folder_id}/analyze")
async def analyze_folder_coherence(
    folder_id: str,
    user_id: CurrentUserID,
    paper_ids: str | None = Query(default=None, description="Comma-separated paper IDs to include (default: all)"),
):
    """Analyse thematic coherence of papers in a bookmark folder.

    Returns a :class:`CoherenceReport` identifying which papers are on-theme
    (related) and which are outliers.  The frontend shows this to the user
    before triggering media generation so they can deselect outlier papers.

    Args:
        folder_id: UUID of the bookmark folder (must be owned by the user).
        paper_ids: Optional comma-separated list of paper UUIDs to include.
            When omitted all papers in the folder are analysed.

    Returns:
        CoherenceReport dict with ``coherence_report``, ``consolidated_content``,
        and ``error`` keys.
    """
    folder_uuid = _parse_uuid(folder_id, "folder_id")
    paper_id_list = [pid.strip() for pid in (paper_ids or "").split(",") if pid.strip()] or None

    from app.workflows.folder_consolidation import run_folder_analysis

    result = await run_folder_analysis(
        folder_id=str(folder_uuid),
        user_id=str(user_id),
        paper_ids_override=paper_id_list,
    )

    return {
        "folder_id": folder_id,
        "coherence_report": result.get("coherence_report", {}),
        "paper_count": len(result.get("papers", [])),
        "error": result.get("error"),
    }


@router.get("/jobs", response_model=JobsListResponse)
async def list_jobs(user_id: CurrentUserID):
    """Return in-progress + recently-finished generation jobs for the current user.

    Reads from the JobStore (in-memory or Redis). Used by the frontend
    JobsPanel to surface generation jobs alongside Study/Genie/Graph jobs.
    """
    store = get_job_store()
    all_jobs = await store.list_by_user(str(user_id))
    # Filter to only media-generation jobs — the store is shared with assistant
    # tasks (kind="assistant") which must not appear here or the frontend will
    # try to render them as GenerationJob objects (missing generation_type →
    # "undefined generation started" toast).
    jobs = [j for j in all_jobs if j.get("generation_type") is not None]
    return JobsListResponse(jobs=jobs, total=len(jobs))


@router.post("/{source_type}/{source_id}/{generation_type}", response_model=TriggerResponse)
async def trigger_generation(
    source_type: str,
    source_id: str,
    generation_type: str,
    user_id: CurrentUserID,
    db: DBSession,
    force_regenerate: bool = Query(default=False, description="Re-generate even if matching completed artifact exists"),
    expertise_level_param: str | None = Query(default=None, alias="expertise_level", description="Override expertise level (defaults to user profile setting)"),
    orientation_param: str | None = Query(default=None, alias="orientation", description="Override orientation (defaults to user profile setting)"),
):
    """Queue a media generation job for a paper or capsule.

    Cache lookup uses the FULL signature required by spec:
    ``(user, source, type, expertise, orientation, provider, model)``.
    A cached completed artifact is returned only when *all* fields match.

    Args:
        source_type: ``paper`` | ``capsule``.
        source_id: UUID of the source entity.
        generation_type: ``podcast`` | ``slides``.
        force_regenerate: Skip cache check and always create new artifact.

    Returns:
        ``TriggerResponse`` with artifact_id and job_id.
    """
    st = _to_source_type(source_type)
    gt = _to_gen_type(generation_type)
    source_uuid = _parse_uuid(source_id, "source_id")

    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found.")

    expertise = expertise_level_param or (user.expertise_level.value if user.expertise_level else "practitioner")
    orientation = orientation_param or (user.orientation.value if user.orientation else "both")
    provider, model = _models_for_generation(gt)
    parser = settings.pdf_parser

    repo = ArtifactRepository(db)

    # Spec-compliant cache lookup — all dimensions must match
    if not force_regenerate:
        existing = await repo.get_latest_completed(
            user_id=user_id, source_id=source_uuid, generation_type=gt
        )
        if (
            existing
            and existing.expertise_level == expertise
            and existing.orientation == orientation
            and existing.provider == provider
            and existing.model_used == model
            # parser_used: only enforce match when the existing artifact recorded it
            and (existing.parser_used is None or existing.parser_used == parser)
        ):
            cached_title = (existing.artifact_metadata or {}).get("source_title", "")
            log.info("generate.trigger cache_hit artifact=%s type=%s", existing.id, gt)
            return TriggerResponse(
                artifact_id=str(existing.id),
                job_id="cached",
                status="completed",
                source_title=cached_title,
                message="Returning cached artifact. Use force_regenerate=true to regenerate.",
            )

    # Cross-user dedup: a deterministic generation already completed for some
    # other user with the same (source, type, expertise, orientation, provider,
    # model, parser) signature. Clone the heavy outputs into a new per-user row
    # so ownership/lifecycle remain user-scoped while saving regeneration cost.
    if not force_regenerate:
        reusable = await repo.find_reusable_completed_global(
            source_id=source_uuid,
            generation_type=gt,
            expertise_level=expertise,
            orientation=orientation,
            provider=provider,
            model_used=model,
            parser_used=parser,
        )
        if reusable is not None and reusable.user_id != user_id:
            cloned = await repo.create(
                user_id=user_id,
                generation_type=gt,
                source_type=st,
                source_id=source_uuid,
                expertise_level=expertise,
                orientation=orientation,
            )
            await repo.mark_completed(
                cloned.id,
                blob_path=reusable.blob_path,
                content=reusable.content,
                provider=reusable.provider,
                model_used=reusable.model_used,
                parser_used=reusable.parser_used,
                input_tokens=0,  # this user paid no tokens — they reused output
                output_tokens=0,
                duration_ms=0,
                metadata={
                    **(reusable.artifact_metadata or {}),
                    "reused_from_artifact": str(reusable.id),
                },
            )
            await db.commit()
            cached_title = (reusable.artifact_metadata or {}).get("source_title", "")
            log.info(
                "generate.trigger global_dedup artifact=%s reused_from=%s type=%s",
                cloned.id, reusable.id, gt,
            )
            return TriggerResponse(
                artifact_id=str(cloned.id),
                job_id="cached",
                status="completed",
                source_title=cached_title,
                message="Reused deterministic output from prior generation.",
            )

    # Create a new artifact row in queued state
    artifact = await repo.create(
        user_id=user_id,
        generation_type=gt,
        source_type=st,
        source_id=source_uuid,
        expertise_level=expertise,
        orientation=orientation,
    )
    # Pre-record provider/model/parser so cache lookups work even on partial generation.
    artifact.provider = provider
    artifact.model_used = model
    artifact.parser_used = parser
    await db.commit()

    # Resolve the display title from the source entity for the notification panel
    source_title = await _resolve_source_title(db, source_uuid, st)

    # Persist title in artifact metadata so the JobStore and frontend can show it
    artifact.artifact_metadata = {"source_title": source_title}
    await db.commit()

    job_id = _dispatch_job(
        generation_type=gt,
        artifact_id=artifact.id,
        user_id=user_id,
        source_type=source_type,
        source_id=source_id,
        expertise_level=expertise,
        orientation=orientation,
        title=source_title,
    )

    log.info(
        "generate.trigger queued artifact=%s type=%s source=%s/%s expertise=%s orientation=%s",
        artifact.id, gt, source_type, source_id, expertise, orientation,
    )

    return TriggerResponse(
        artifact_id=str(artifact.id),
        job_id=job_id,
        status="queued",
        source_title=source_title,
        message=f"{generation_type.title()} generation queued. Poll /generate/artifact/{artifact.id} for status.",
    )


async def _resolve_source_title(db, source_id: UUID, source_type: SourceType) -> str:
    """Look up the human-readable title for the source entity."""
    try:
        if source_type == SourceType.paper:
            from sqlalchemy import select
            from app.models.paper import Paper
            row = await db.execute(select(Paper.title).where(Paper.id == source_id))
            title = row.scalar_one_or_none()
            return title or "Untitled Paper"

        if source_type == SourceType.capsule:
            from sqlalchemy import select
            from app.models.genie import IdeaCapsule
            row = await db.execute(select(IdeaCapsule.title).where(IdeaCapsule.id == source_id))
            title = row.scalar_one_or_none()
            return title or "Untitled Idea"

        if source_type == SourceType.folder:
            from sqlalchemy import select
            from app.models.paper import BookmarkFolder
            row = await db.execute(select(BookmarkFolder.name).where(BookmarkFolder.id == source_id))
            name = row.scalar_one_or_none()
            return name or "Untitled Folder"
    except Exception as exc:
        log.debug("_resolve_source_title failed: %s", exc)
    return ""


def _dispatch_job(
    *,
    generation_type: GenerationType,
    artifact_id: UUID,
    user_id: UUID,
    source_type: str,
    source_id: str,
    expertise_level: str,
    orientation: str,
    title: str,
) -> str:
    """Route to the correct queue_* function based on generation type."""
    kwargs = dict(
        artifact_id=artifact_id,
        user_id=user_id,
        source_type=source_type,
        source_id=source_id,
        expertise_level=expertise_level,
        orientation=orientation,
        title=title,
    )

    if generation_type == GenerationType.podcast:
        from app.workflows.podcast import queue_podcast
        return queue_podcast(**kwargs)
    if generation_type == GenerationType.slides:
        from app.workflows.slides import queue_slides
        return queue_slides(**kwargs)

    raise HTTPException(
        status_code=422,
        detail=f"Generation type '{generation_type.value}' is no longer supported.",
    )


@router.get("/artifact/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(
    artifact_id: str,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Poll the status and result of a generated artifact."""
    art_uuid = _parse_uuid(artifact_id, "artifact_id")

    repo = ArtifactRepository(db)
    artifact = await repo.get_by_id(art_uuid)

    if not artifact or artifact.user_id != user_id:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    return _artifact_to_response(artifact)


@router.get("/{source_type}/{source_id}", response_model=list[ArtifactResponse])
async def list_artifacts(
    source_type: str,
    source_id: str,
    user_id: CurrentUserID,
    db: DBSession,
):
    """List artifacts for a source entity, newest first.

    Only the currently supported media types (podcast, slides) are returned.
    Any deprecated rows still in the DB from older deployments are filtered
    out at this layer so the UI never sees them.
    """
    _to_source_type(source_type)  # validate
    source_uuid = _parse_uuid(source_id, "source_id")

    repo = ArtifactRepository(db)
    artifacts = await repo.list_for_source(user_id=user_id, source_id=source_uuid)
    active = [
        a for a in artifacts
        if a.generation_type.value in _VALID_GEN_TYPES
    ]
    return [_artifact_to_response(a) for a in active]


@router.delete("/artifact/{artifact_id}", status_code=204)
async def delete_artifact(
    artifact_id: str,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Delete an artifact record and its blob (if any). Idempotent on missing blob."""
    art_uuid = _parse_uuid(artifact_id, "artifact_id")

    repo = ArtifactRepository(db)
    artifact = await repo.get_by_id(art_uuid)

    if not artifact or artifact.user_id != user_id:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    if artifact.blob_path:
        # Refcount: shared deterministic outputs (clones from global dedup) may
        # point at the same blob_path. Only delete the underlying blob when this
        # is the LAST referencing artifact — otherwise other users' rows would
        # be left dangling.
        try:
            other_refs = await repo.count_references_to_blob(
                blob_path=artifact.blob_path, exclude_artifact_id=artifact.id,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning("generate.delete_artifact refcount failed: %s", exc)
            other_refs = 1  # conservative: assume another user references it
        if other_refs == 0:
            try:
                from app.adapters.blob import get_blob_storage
                blob = get_blob_storage()
                await blob.delete(artifact.blob_path)
            except Exception as exc:  # noqa: BLE001 — non-fatal
                log.warning("generate.delete_artifact blob delete failed: %s", exc)
        else:
            log.info(
                "generate.delete_artifact id=%s preserved blob %s (%d other refs)",
                artifact.id, artifact.blob_path, other_refs,
            )

    # Remove checkpoint data so orphan rows don't accumulate
    try:
        from app.db.checkpointer import get_checkpointer
        cp = await get_checkpointer()
        await cp.delete_thread(artifact_id)
    except Exception as exc:  # noqa: BLE001 — non-fatal
        log.debug("generate.delete_artifact checkpoint cleanup skipped: %s", exc)

    await db.delete(artifact)
    await db.commit()


@router.post("/artifact/{artifact_id}/cancel", status_code=200)
async def cancel_artifact(
    artifact_id: str,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Mark an in-flight generation artifact as cancelled.

    Marks the DB row as ``failed`` with ``error_message="cancelled"`` so the
    UI immediately shows a terminal state. Currently we cannot abort the
    underlying ``asyncio.create_task`` from here (the running graph holds
    no cancel handle), so the worker will keep going until its current
    LLM/TTS call completes. The user-facing job, however, is treated as
    cancelled — no notification when it finishes, no cached artifact.

    Returns:
        ``{"artifact_id": ..., "status": "cancelled"}``.
    """
    art_uuid = _parse_uuid(artifact_id, "artifact_id")

    repo = ArtifactRepository(db)
    artifact = await repo.get_by_id(art_uuid)

    if not artifact or artifact.user_id != user_id:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    if artifact.status in (ArtifactStatus.completed, ArtifactStatus.failed):
        # Already terminal — return current state.
        return {"artifact_id": str(art_uuid), "status": artifact.status.value}

    # Stop the asyncio task immediately (single-process; no-op in multi-worker).
    cancel_generation_task(str(art_uuid))

    await repo.mark_failed(art_uuid, error_message="cancelled")
    await db.commit()

    # Mirror the cancellation in the JobStore so polling doesn't re-surface
    # the job as "running" after the local dismissedArtifactIds is cleared.
    store = get_job_store()
    try:
        user_jobs = await store.list_by_user(str(user_id))
        for job in user_jobs:
            if job.get("artifact_id") == str(art_uuid):
                await store.update(job["job_id"], {"status": "failed"})
                break
    except Exception as exc:  # noqa: BLE001 — non-fatal
        log.debug("generate.cancel: job_store update skipped: %s", exc)

    log.info("generate.cancel artifact=%s user=%s", art_uuid, user_id)
    return {"artifact_id": str(art_uuid), "status": "cancelled"}
