"""Bookmarks router — reading list with named folders (multi-folder), notes, and RAG chat."""

import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete as sa_delete
from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.deps import CurrentUserID, DBSession
from app.models.genie import ElementType, GenieElement
from app.models.paper import Bookmark, BookmarkFolder, BookmarkFolderMember, PaperChunk
from app.repositories.paper import PaperRepository
from app.schemas import BookmarkRequest, BookmarkResponse, FolderResponse, PaperResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/bookmarks", tags=["bookmarks"])


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _folder_ids_for_bookmark(db, bookmark_id: UUID) -> list[UUID]:
    """Return all folder UUIDs that contain the given bookmark."""
    rows = await db.execute(
        select(BookmarkFolderMember.folder_id).where(BookmarkFolderMember.bookmark_id == bookmark_id)
    )
    return list(rows.scalars().all())


async def _set_bookmark_folders(db, bookmark_id: UUID, user_id: UUID, folder_ids: list[UUID]) -> None:
    """Replace all folder memberships for a bookmark atomically."""
    await db.execute(
        sa_delete(BookmarkFolderMember).where(BookmarkFolderMember.bookmark_id == bookmark_id)
    )
    for fid in folder_ids:
        folder_check = await db.execute(
            select(BookmarkFolder).where(BookmarkFolder.id == fid, BookmarkFolder.user_id == user_id)
        )
        if folder_check.scalar_one_or_none():
            db.add(BookmarkFolderMember(bookmark_id=bookmark_id, folder_id=fid))


async def _index_paper_background(paper_id: UUID, user_id: UUID) -> None:
    """Embed abstract + build graph node + Genie element for a newly bookmarked paper."""
    from app.adapters.embedding import get_embedding_adapter
    from app.db.session import async_session_factory
    from app.services.graph import GraphService

    try:
        async with async_session_factory() as db:
            paper_repo = PaperRepository(db)
            paper = await paper_repo.get_by_id(paper_id)
            if not paper:
                return

            existing_chunks = await paper_repo.get_chunks(paper_id)
            # Find existing abstract chunk — may exist with embedding=None from a previous failed attempt
            abstract_chunk = next((c for c in existing_chunks if c.section_type == "abstract"), None)

            if abstract_chunk is None or abstract_chunk.embedding is None:
                text = (paper.abstract or paper.title or "").strip()
                if abstract_chunk is None:
                    abstract_chunk = PaperChunk(
                        paper_id=paper_id,
                        chunk_index=0,
                        section_type="abstract",
                        content=text,
                        embedding=None,
                        embedding_dim=768,
                        embedding_provider="gemini",
                    )
                    db.add(abstract_chunk)
                    await db.flush()

                try:
                    embed = get_embedding_adapter()
                    vectors = await embed.embed_texts([text], task_type="RETRIEVAL_DOCUMENT")
                    abstract_chunk.embedding = vectors[0]
                    abstract_chunk.embedding_dim = embed.dimensions
                    abstract_chunk.embedding_provider = embed.provider_id
                except Exception as emb_exc:
                    log.warning("bookmark indexing: embedding failed paper=%s err=%s", paper_id, emb_exc)

                await db.commit()
    except Exception as exc:
        log.warning("bookmark indexing: stage-1 failed paper=%s err=%s", paper_id, exc)

    # Skip the graph-assignment stage entirely when the feature is off.
    # Bookmarking still works (embedding + Genie element creation below);
    # only the graph node + cache invalidation are suppressed.
    try:
        from app.services.admin_settings import get_app_settings
        _settings = await get_app_settings()
        _graph_on = bool(_settings.get("graph_enabled", False))
    except Exception:
        _graph_on = False

    try:
        async with async_session_factory() as db:
            paper_repo = PaperRepository(db)
            paper = await paper_repo.get_by_id(paper_id)
            if not paper:
                return

            try:
                if _graph_on:
                    svc = GraphService(db)
                    await svc.add_paper_node(paper)
                    # Invalidate in-memory build cache so next Build Deep includes this paper
                    GraphService._build_cache.pop(paper.namespace_key or "cs.AI", None)
                    GraphService._build_cache.pop(None, None)
                    # Clear the persistent subgraph cache so the graph page shows the new node
                    await GraphService.clear_subgraph_cache(paper.namespace_key)
                    await GraphService.clear_subgraph_cache(None)
            except Exception as graph_exc:
                log.warning("bookmark indexing: graph node failed paper=%s err=%s", paper_id, graph_exc)
                await db.rollback()

                async with async_session_factory() as db2:
                    existing = await db2.execute(
                        select(GenieElement).where(
                            GenieElement.user_id == user_id,
                            GenieElement.paper_id == paper_id,
                        )
                    )
                    if not existing.scalar_one_or_none():
                        db2.add(GenieElement(
                            user_id=user_id,
                            element_type=ElementType.paper,
                            label=paper.title[:500],
                            paper_id=paper_id,
                        ))
                    await db2.commit()
                return

            existing = await db.execute(
                select(GenieElement).where(
                    GenieElement.user_id == user_id,
                    GenieElement.paper_id == paper_id,
                )
            )
            if not existing.scalar_one_or_none():
                db.add(GenieElement(
                    user_id=user_id,
                    element_type=ElementType.paper,
                    label=paper.title[:500],
                    paper_id=paper_id,
                ))
            await db.commit()
    except Exception as exc:
        log.warning("bookmark indexing: stage-2 failed paper=%s err=%s", paper_id, exc)


# ── Folder endpoints ──────────────────────────────────────────────────────────

class FolderCreateRequest(BaseModel):
    """Request body for creating or renaming a bookmark folder."""

    name: str = Field(min_length=1, max_length=200)
    color: str | None = None


@router.get("/folders", response_model=list[FolderResponse])
async def list_folders(user_id: CurrentUserID, db: DBSession):
    """Return all bookmark folders for the current user with bookmark counts.

    Uses a single aggregated query (GROUP BY) instead of one COUNT per folder.

    Args:
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        A list of ``FolderResponse`` objects ordered by creation time.
    """
    result = await db.execute(
        select(BookmarkFolder).where(BookmarkFolder.user_id == user_id).order_by(BookmarkFolder.created_at)
    )
    folders = list(result.scalars().all())

    if not folders:
        return []

    # Single query: count members for ALL folders at once
    folder_ids = [f.id for f in folders]
    counts_result = await db.execute(
        select(
            BookmarkFolderMember.folder_id,
            sa_func.count(BookmarkFolderMember.bookmark_id).label("cnt"),
        )
        .where(BookmarkFolderMember.folder_id.in_(folder_ids))
        .group_by(BookmarkFolderMember.folder_id)
    )
    count_map: dict = {row.folder_id: row.cnt for row in counts_result.fetchall()}

    return [
        FolderResponse(
            id=f.id,
            name=f.name,
            color=f.color,
            created_at=f.created_at,
            bookmark_count=count_map.get(f.id, 0),
        )
        for f in folders
    ]


@router.post("/folders", response_model=FolderResponse, status_code=201)
async def create_folder(body: FolderCreateRequest, user_id: CurrentUserID, db: DBSession):
    """Create a new named bookmark folder for the current user.

    Args:
        body: Folder name (required) and optional colour.
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        The newly created ``FolderResponse``.

    Raises:
        HTTPException: 409 if a folder with that name already exists.
    """
    try:
        folder = BookmarkFolder(user_id=user_id, name=body.name.strip(), color=body.color)
        db.add(folder)
        await db.flush()
        await db.commit()
        return FolderResponse(id=folder.id, name=folder.name, color=folder.color,
                               created_at=folder.created_at, bookmark_count=0)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail=f"Folder '{body.name}' already exists")


@router.patch("/folders/{folder_id}")
async def rename_folder(
    folder_id: UUID,
    body: FolderCreateRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Rename a bookmark folder (and optionally update its colour).

    Args:
        folder_id: UUID of the folder to update.
        body: New name (required) and optional new colour.
        user_id: UUID of the authenticated user (must own the folder).
        db: Injected async database session.

    Returns:
        A dict with the updated ``id``, ``name``, and ``color`` fields.

    Raises:
        HTTPException: 404 if the folder is not found or not owned by the user.
    """
    result = await db.execute(
        select(BookmarkFolder).where(BookmarkFolder.id == folder_id, BookmarkFolder.user_id == user_id)
    )
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    folder.name = body.name.strip()
    if body.color is not None:
        folder.color = body.color
    await db.commit()
    return {"id": str(folder.id), "name": folder.name, "color": folder.color}


@router.delete("/folders/{folder_id}", status_code=204)
async def delete_folder(folder_id: UUID, user_id: CurrentUserID, db: DBSession):
    """Delete a bookmark folder and all its membership records.

    Args:
        folder_id: UUID of the folder to delete.
        user_id: UUID of the authenticated user (must own the folder).
        db: Injected async database session.

    Raises:
        HTTPException: 404 if the folder is not found or not owned by the user.
    """
    result = await db.execute(
        select(BookmarkFolder).where(BookmarkFolder.id == folder_id, BookmarkFolder.user_id == user_id)
    )
    folder = result.scalar_one_or_none()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    await db.delete(folder)
    await db.commit()


# ── Bookmark endpoints ────────────────────────────────────────────────────────

@router.get("/indexing-status")
async def indexing_status(
    user_id: CurrentUserID,
    db: DBSession,
    namespace_keys: str | None = Query(default=None),
):
    """Return how many of the user's bookmarked papers have been embedded.

    Uses batch queries (one for papers, one for chunk existence) instead of
    per-bookmark DB round-trips.

    Args:
        user_id: UUID of the authenticated user.
        db: Injected async database session.
        namespace_keys: Optional comma-separated namespace filter.

    Returns:
        A dict with ``total`` bookmarks, ``indexed`` count (papers with at
        least one embedding chunk), and a ``ready`` boolean.
    """
    from app.models.paper import Paper as _Paper
    from sqlalchemy import distinct as sa_distinct

    repo = PaperRepository(db)
    bookmarks = await repo.get_bookmarks(user_id)
    if not bookmarks:
        return {"total": 0, "indexed": 0, "ready": True}

    paper_ids = [bm.paper_id for bm in bookmarks]

    allowed_ns: set[str] | None = None
    if namespace_keys:
        allowed_ns = {k.strip() for k in namespace_keys.split(",") if k.strip()}

    # Batch load papers to apply namespace filter
    if allowed_ns:
        ns_result = await db.execute(
            select(_Paper.id).where(_Paper.id.in_(paper_ids), _Paper.namespace_key.in_(allowed_ns))
        )
        paper_ids = [r[0] for r in ns_result.fetchall()]

    total = len(paper_ids)
    if total == 0:
        return {"total": 0, "indexed": 0, "ready": True}

    # Single query: which paper_ids have at least one embedded chunk
    embedded_result = await db.execute(
        select(sa_distinct(PaperChunk.paper_id)).where(
            PaperChunk.paper_id.in_(paper_ids),
            PaperChunk.embedding.isnot(None),
        )
    )
    indexed_ids = {r[0] for r in embedded_result.fetchall()}
    indexed = len(indexed_ids)

    return {"total": total, "indexed": indexed, "ready": total == 0 or indexed == total}


@router.get("", response_model=list[BookmarkResponse])
async def list_bookmarks(
    user_id: CurrentUserID,
    db: DBSession,
    namespace_keys: str | None = Query(default=None),
    folder_id: UUID | None = Query(default=None, description="Filter to bookmarks in this folder"),
):
    """Return all bookmarks for the current user, with optional filters.

    Uses batch queries — one for papers, one for folder memberships — instead of
    one query per bookmark (eliminating the previous O(2N) pattern).

    Args:
        user_id: UUID of the authenticated user.
        db: Injected async database session.
        namespace_keys: Optional comma-separated namespace filter.
        folder_id: When provided, only bookmarks in this folder are returned.

    Returns:
        A list of ``BookmarkResponse`` objects including folder memberships
        and embedded paper data.
    """
    from app.models.paper import Paper as _Paper

    repo = PaperRepository(db)
    bookmarks = await repo.get_bookmarks(user_id)
    if not bookmarks:
        return []

    allowed_ns: set[str] | None = None
    if namespace_keys:
        allowed_ns = {k.strip() for k in namespace_keys.split(",") if k.strip()}

    # Folder filter — get bookmark IDs in that folder (single query)
    folder_bm_ids: set[UUID] | None = None
    if folder_id is not None:
        rows = await db.execute(
            select(BookmarkFolderMember.bookmark_id).where(BookmarkFolderMember.folder_id == folder_id)
        )
        folder_bm_ids = set(rows.scalars().all())

    # Apply folder filter before DB trips
    if folder_bm_ids is not None:
        bookmarks = [bm for bm in bookmarks if bm.id in folder_bm_ids]
    if not bookmarks:
        return []

    # Batch load all papers in ONE query
    paper_ids = [bm.paper_id for bm in bookmarks]
    papers_result = await db.execute(
        select(_Paper).where(_Paper.id.in_(paper_ids))
    )
    paper_map: dict[UUID, _Paper] = {p.id: p for p in papers_result.scalars()}

    # Apply namespace filter using the preloaded paper map
    if allowed_ns:
        bookmarks = [
            bm for bm in bookmarks
            if (p := paper_map.get(bm.paper_id)) and p.namespace_key in allowed_ns
        ]
    if not bookmarks:
        return []

    # Batch load all folder memberships in ONE query
    bm_ids = [bm.id for bm in bookmarks]
    members_result = await db.execute(
        select(BookmarkFolderMember.bookmark_id, BookmarkFolderMember.folder_id)
        .where(BookmarkFolderMember.bookmark_id.in_(bm_ids))
    )
    folders_map: dict[UUID, list[UUID]] = {}
    for bm_id, fid in members_result.fetchall():
        folders_map.setdefault(bm_id, []).append(fid)

    return [
        BookmarkResponse(
            id=bm.id,
            paper_id=bm.paper_id,
            folder_ids=folders_map.get(bm.id, []),
            note=bm.note,
            created_at=bm.created_at,
            paper=PaperResponse.model_validate(paper_map[bm.paper_id]) if bm.paper_id in paper_map else None,
        )
        for bm in bookmarks
    ]


@router.post("", status_code=201)
async def add_bookmark(
    body: BookmarkRequest,
    user_id: CurrentUserID,
    db: DBSession,
    bg: BackgroundTasks,
):
    """Bookmark a paper for the current user and trigger background indexing.

    Creates the bookmark record, assigns it to the requested folders (if any),
    and queues a background task to embed the abstract, add a knowledge graph
    node, and create a Genie element. If the bookmark already exists, folder
    memberships are updated instead.

    Args:
        body: Bookmark payload with ``paper_id``, optional ``note``, and
            optional ``folder_ids``.
        user_id: UUID of the authenticated user.
        db: Injected async database session.
        bg: FastAPI background task queue.

    Returns:
        A dict with the bookmark ``id`` and its ``folder_ids``.
    """
    repo = PaperRepository(db)
    try:
        bm = await repo.add_bookmark(user_id, body.paper_id, body.note)
        if body.folder_ids:
            await _set_bookmark_folders(db, bm.id, user_id, body.folder_ids)
        await db.commit()
        bg.add_task(_index_paper_background, body.paper_id, user_id)
        fids = await _folder_ids_for_bookmark(db, bm.id)
        return {"id": str(bm.id), "folder_ids": [str(f) for f in fids]}
    except IntegrityError:
        await db.rollback()
        result = await db.execute(
            select(Bookmark).where(Bookmark.user_id == user_id, Bookmark.paper_id == body.paper_id)
        )
        existing = result.scalar_one_or_none()
        if existing and body.folder_ids:
            await _set_bookmark_folders(db, existing.id, user_id, body.folder_ids)
            await db.commit()
        fids = await _folder_ids_for_bookmark(db, existing.id) if existing else []
        return {"id": str(existing.id) if existing else "", "folder_ids": [str(f) for f in fids]}


@router.put("/{paper_id}/folders")
async def set_paper_folders(
    paper_id: UUID,
    user_id: CurrentUserID,
    db: DBSession,
    body: dict,
):
    """Replace all folder memberships for a bookmarked paper.
    Body: { "folder_ids": ["uuid", ...] }
    """
    result = await db.execute(
        select(Bookmark).where(Bookmark.user_id == user_id, Bookmark.paper_id == paper_id)
    )
    bm = result.scalar_one_or_none()
    if not bm:
        raise HTTPException(status_code=404, detail="Bookmark not found")

    folder_ids = [UUID(f) for f in (body.get("folder_ids") or [])]
    await _set_bookmark_folders(db, bm.id, user_id, folder_ids)
    await db.commit()
    fids = await _folder_ids_for_bookmark(db, bm.id)
    return {"paper_id": str(paper_id), "folder_ids": [str(f) for f in fids]}


@router.delete("/{paper_id}/folders/{folder_id}", status_code=204)
async def remove_from_folder(
    paper_id: UUID,
    folder_id: UUID,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Remove a bookmark from one specific folder without deleting the bookmark."""
    result = await db.execute(
        select(Bookmark).where(Bookmark.user_id == user_id, Bookmark.paper_id == paper_id)
    )
    bm = result.scalar_one_or_none()
    if not bm:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    await db.execute(
        sa_delete(BookmarkFolderMember).where(
            BookmarkFolderMember.bookmark_id == bm.id,
            BookmarkFolderMember.folder_id == folder_id,
        )
    )
    await db.commit()


@router.post("/reindex", status_code=202)
async def reindex_bookmarks(user_id: CurrentUserID, db: DBSession, bg: BackgroundTasks):
    """Queue background embedding for all bookmarks that lack an abstract chunk.

    Args:
        user_id: UUID of the authenticated user.
        db: Injected async database session.
        bg: FastAPI background task queue.

    Returns:
        A dict with ``queued`` (number of tasks enqueued) and ``total``
        (total number of bookmarks).
    """
    repo = PaperRepository(db)
    bookmarks = await repo.get_bookmarks(user_id)
    queued = 0
    for bm in bookmarks:
        chunks = await repo.get_chunks(bm.paper_id)
        if not any(c.section_type == "abstract" and c.embedding is not None for c in chunks):
            bg.add_task(_index_paper_background, bm.paper_id, user_id)
            queued += 1
    return {"queued": queued, "total": len(bookmarks)}


@router.delete("/{paper_id}", status_code=204)
async def remove_bookmark(paper_id: UUID, user_id: CurrentUserID, db: DBSession):
    """Remove a bookmark and its associated Genie element.

    Args:
        paper_id: UUID of the paper whose bookmark should be deleted.
        user_id: UUID of the authenticated user.
        db: Injected async database session.
    """
    repo = PaperRepository(db)
    await repo.remove_bookmark(user_id, paper_id)
    await db.execute(
        sa_delete(GenieElement).where(
            GenieElement.user_id == user_id,
            GenieElement.paper_id == paper_id,
        )
    )
    await db.commit()


# ── RAG Chat (folder-scoped) ──────────────────────────────────────────────────

class BookmarkChatRequest(BaseModel):
    """Request body for POST /bookmarks/chat."""

    message: str
    expertise_level: str = "practitioner"
    history: list[dict] = []
    namespace_keys: list[str] | None = None
    folder_id: str | None = None


@router.post("/chat", response_class=StreamingResponse)
async def chat_bookmarks(body: BookmarkChatRequest, user_id: CurrentUserID, db: DBSession):
    """Stream RAG chat. When folder_id is set, only that folder's papers are context."""
    from app.workflows.study import run_bookmarks_chat

    paper_ids: list[str] | None = None
    if body.folder_id:
        try:
            fid = UUID(body.folder_id)
            # Get all bookmark IDs in this folder, then get their paper IDs
            rows = await db.execute(
                select(BookmarkFolderMember.bookmark_id).where(BookmarkFolderMember.folder_id == fid)
            )
            bm_ids = list(rows.scalars().all())
            if bm_ids:
                paper_rows = await db.execute(
                    select(Bookmark.paper_id).where(
                        Bookmark.id.in_(bm_ids),
                        Bookmark.user_id == user_id,
                    )
                )
                paper_ids = [str(r) for r in paper_rows.scalars()]
        except Exception:
            pass

    ns_list = body.namespace_keys if body.namespace_keys else None

    async def event_generator():
        """Yield SSE chunks from the bookmark RAG chat workflow."""
        async for chunk in run_bookmarks_chat(
            user_id, body.expertise_level, body.message, body.history,
            namespace_keys=ns_list,
            paper_ids=paper_ids,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
