"""Knowledge Graph router — subgraph, expand node, serendipity."""

import asyncio
import logging
import uuid as _uuid
from uuid import UUID

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.adapters.cache import get_cache
from app.core.deps import CurrentUserID, DBSession
from app.models.paper import Paper as PaperModel
from app.schemas import GraphResponse
from app.services.graph import GraphService

log = logging.getLogger(__name__)

_BUILD_CACHE_TTL = 7_200  # 2 h — keeps job status accessible after completion
_BUILD_TASKS: dict[str, asyncio.Task] = {}

# Limit concurrent deep-build tasks to avoid exhausting the LLM rate limit
# and saturating the DB connection pool when many namespaces are selected.
_BUILD_SEMAPHORE = asyncio.Semaphore(2)

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("", response_model=GraphResponse)
async def get_subgraph(
    db: DBSession,
    user_id: CurrentUserID,
    namespace_key: str | None = Query(default=None),
    namespace_keys: str | None = Query(default=None, description="Comma-separated namespace keys to filter"),
    depth: int = Query(default=2, le=3),
    bookmarks_only: bool = Query(default=False, description="Only show bookmarked papers and their parent nodes"),
):
    """Return a knowledge graph subgraph, optionally scoped by namespace or bookmarks.

    When ``bookmarks_only`` is ``True``, only the user's bookmarked paper nodes
    and their ancestors (upward BFS) are included. Missing paper nodes are
    auto-healed by creating them on demand. When a namespace filter is active,
    nodes outside the filter are stripped before returning.

    Args:
        db: Injected async database session.
        user_id: UUID of the authenticated user.
        namespace_key: Single namespace to filter (e.g. ``"cs.AI"``).
        namespace_keys: Comma-separated namespaces; takes priority over
            ``namespace_key`` when both are provided.
        depth: Graph traversal depth (max 3).
        bookmarks_only: When ``True``, restrict nodes to bookmarked papers
            and their parent hierarchy.

    Returns:
        A ``GraphResponse`` with ``nodes`` and ``edges`` lists.
    """
    # Resolve active namespace filter (multi takes priority over single)
    ns_filter: set[str] | None = None
    if namespace_keys:
        ns_filter = {k.strip() for k in namespace_keys.split(",") if k.strip()}
    elif namespace_key:
        ns_filter = {namespace_key}

    svc = GraphService(db)
    # Bookmark-scoped graphs are user-specific — skip the shared feed cache.
    # Feed-scoped graphs are served from cache when available.
    graph = await svc.get_subgraph(None, depth, use_cache=not bookmarks_only)

    if not bookmarks_only:
        if ns_filter:
            # Step 1: collect every node whose namespace_key is in the filter.
            # These are SUBTOPIC, CONCEPT, PAPER, and METHOD nodes that belong
            # to the requested subjects.
            ns_nodes = {n["id"] for n in graph["nodes"] if n.get("namespace_key") in ns_filter}

            # Step 2: walk UPWARD through namespace_key=None TOPIC nodes
            # (Subject and Domain nodes) to find only the ancestors that are
            # actually connected to the filtered nodes.  Including ALL None-keyed
            # nodes unconditionally caused deselected subjects (e.g. Mathematics)
            # to appear as orphaned subject bubbles even when no math namespaces
            # were selected.
            topic_ids = {n["id"] for n in graph["nodes"] if n.get("namespace_key") is None}
            target_to_sources: dict[str, set[str]] = {}
            for e in graph["edges"]:
                target_to_sources.setdefault(e["target"], set()).add(e["source"])

            reachable = set(ns_nodes)
            frontier = set(ns_nodes)
            while frontier:
                next_frontier: set[str] = set()
                for nid in frontier:
                    for parent in target_to_sources.get(nid, set()):
                        if parent not in reachable and parent in topic_ids:
                            reachable.add(parent)
                            next_frontier.add(parent)
                frontier = next_frontier

            nodes = [n for n in graph["nodes"] if n["id"] in reachable]
            node_ids = {n["id"] for n in nodes}
            edges = [e for e in graph["edges"] if e["source"] in node_ids and e["target"] in node_ids]
            return {"nodes": nodes, "edges": edges}
        return graph

    # ── bookmarks_only path ──────────────────────────────────────────────────
    from app.repositories.paper import PaperRepository
    paper_repo = PaperRepository(db)
    bookmarks = await paper_repo.get_bookmarks(user_id)

    # Apply namespace filter to bookmarks
    if ns_filter:
        filtered_bookmarks = []
        for bm in bookmarks:
            paper = await paper_repo.get_by_id(bm.paper_id)
            if paper and paper.namespace_key in ns_filter:
                filtered_bookmarks.append(bm)
        bookmarks = filtered_bookmarks

    bookmarked_paper_ids = {str(bm.paper_id) for bm in bookmarks}
    if not bookmarked_paper_ids:
        return {"nodes": [], "edges": []}

    nodes = graph["nodes"]
    edges = graph["edges"]

    # Find existing PAPER graph nodes for bookmarked papers
    bookmarked_node_ids: set[str] = set()
    for n in nodes:
        if n["type"] == "PAPER" and n.get("paper_id") in bookmarked_paper_ids:
            bookmarked_node_ids.add(n["id"])

    # Auto-heal: create graph nodes for bookmarked papers that have none
    existing_graph_paper_ids = {n.get("paper_id") for n in nodes if n["type"] == "PAPER"}
    missing_ids = bookmarked_paper_ids - existing_graph_paper_ids
    if missing_ids:
        try:
            missing_res = await db.execute(
                select(PaperModel).where(PaperModel.id.in_([UUID(pid) for pid in missing_ids]))
            )
            missing_papers = list(missing_res.scalars())
            for paper in missing_papers:
                await svc.add_paper_node(paper)
            await db.commit()
            # Reload so newly created nodes are included
            graph = await svc.get_subgraph(None, depth)
            nodes = graph["nodes"]
            edges = graph["edges"]
            bookmarked_node_ids = {
                n["id"] for n in nodes
                if n["type"] == "PAPER" and n.get("paper_id") in bookmarked_paper_ids
            }
        except Exception as exc:
            log.warning("get_subgraph auto-heal failed: %s", exc)
            await db.rollback()

    if not bookmarked_node_ids:
        return {"nodes": [], "edges": []}

    # Build upward adjacency only: child → parents
    target_to_sources: dict[str, set[str]] = {}
    for e in edges:
        target_to_sources.setdefault(e["target"], set()).add(e["source"])

    # BFS upward only — never descend into siblings of bookmarked papers
    reachable: set[str] = set(bookmarked_node_ids)
    frontier = set(bookmarked_node_ids)
    while frontier:
        next_frontier: set[str] = set()
        for nid in frontier:
            for parent in target_to_sources.get(nid, set()):
                if parent not in reachable:
                    reachable.add(parent)
                    next_frontier.add(parent)
        frontier = next_frontier

    filtered_nodes = [n for n in nodes if n["id"] in reachable]
    filtered_edges = [
        e for e in edges
        if e["source"] in reachable and e["target"] in reachable
    ]

    return {"nodes": filtered_nodes, "edges": filtered_edges}


@router.get("/expand/{node_id}")
async def expand_node(node_id: UUID, db: DBSession):
    """Return the immediate neighbours of a single knowledge graph node.

    Args:
        node_id: UUID of the node to expand.
        db: Injected async database session.

    Returns:
        A dict with ``nodes`` and ``edges`` representing the one-hop
        neighbourhood of the given node.
    """
    svc = GraphService(db)
    return await svc.expand_node(node_id)


@router.post("/rebuild-hierarchy", status_code=200)
async def rebuild_hierarchy(db: DBSession, user_id: CurrentUserID):
    """Backfill TOPIC → SUBTOPIC → PAPER edges for all existing paper nodes.

    Safe to call multiple times — idempotent edge creation means re-running
    it on an already-wired graph is a no-op.
    """
    svc = GraphService(db)
    wired = await svc.rebuild_hierarchy()
    clustered = await svc.rebuild_clusters()
    return {
        "wired": wired,
        "clustered": clustered,
        "message": f"Hierarchy rebuilt: {wired} orphaned nodes wired; {clustered} concept cluster edges created.",
    }


@router.post("/build-deep", status_code=200)
async def build_deep_graph(
    db: DBSession,
    user_id: CurrentUserID,
    namespace_key: str | None = Query(default=None, description="Scope to a single namespace; omit for all"),
):
    """Generate a deep LLM-powered taxonomy: TOPIC → SUBTOPIC (area) → CONCEPT (cluster) → PAPER.

    The LLM reads paper abstracts and key concepts to produce a 2-level taxonomy
    of research areas and thematic clusters. Cluster and area names are shaped by
    the user's orientation setting (research → academic terminology;
    production → application-focused terminology). Idempotent — re-running merges
    into the existing graph without duplicates.
    """
    from app.repositories.user import UserRepository
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    orientation = user.orientation.value if user else "both"

    svc = GraphService(db)
    result = await svc.build_deep_graph(namespace_key, orientation=orientation)
    if result.get("already_up_to_date"):
        return {**result, "message": "Graph is already up to date — no new papers since last build."}
    return {
        **result,
        "message": (
            f"Deep graph built: {result['areas_created']} areas, "
            f"{result['clusters_created']} clusters, "
            f"{result['papers_mapped']} papers mapped "
            f"(of {result['total_papers_processed']} processed)."
        ),
    }


# ── Background build-deep ──────────────────────────────────────────────────────

async def _run_build_deep_background(
    job_id: str,
    namespace_key: str | None,
    orientation: str,
    lock_key: str | None = None,
) -> None:
    """Run build_deep_graph in the background and write result to cache.

    Follows the same fire-and-forget pattern as Genie's synthesize-bg and
    Deep Search's deep-bg.  The job status is queryable via
    ``GET /graph/build-deep/status/{job_id}``.
    """
    from app.db.session import async_session_factory

    cache = get_cache()
    try:
        async with _BUILD_SEMAPHORE:
            async with async_session_factory() as db:
                svc = GraphService(db)

                async def _should_cancel() -> bool:
                    data = await cache.get(f"graph:build:{job_id}")
                    return bool(data and data.get("cancel_requested"))

                result = await svc.build_deep_graph(
                    namespace_key,
                    orientation=orientation,
                    should_cancel=_should_cancel,
                )

        if result.get("already_up_to_date"):
            msg = "Graph is already up to date — no new papers since last build."
        else:
            msg = (
                f"Deep graph built: {result.get('areas_created', 0)} areas, "
                f"{result.get('sub_areas_created', 0)} sub-areas, "
                f"{result.get('clusters_created', 0)} clusters, "
                f"{result.get('papers_mapped', 0)} papers mapped "
                f"(of {result.get('total_papers_processed', 0)} processed), "
                f"{result.get('related_edges_added', 0)} related-to edges added."
            )

        # Ensure subgraph cache is clear before writing "done" so the frontend's
        # loadGraph() call (triggered by status transition) always hits fresh DB data.
        await GraphService.clear_subgraph_cache(namespace_key)
        await GraphService.clear_subgraph_cache(None)

        await cache.set(
            f"graph:build:{job_id}",
            {"status": "done", "namespace_key": namespace_key, "result": result, "message": msg},
            ttl_seconds=_BUILD_CACHE_TTL,
        )
        log.info("build_deep_background: job=%s done namespace=%s", job_id, namespace_key)

    except asyncio.CancelledError:
        msg = "Graph build cancelled. Partial taxonomy committed before cancellation remains available."
        await cache.set(
            f"graph:build:{job_id}",
            {
                "status": "cancelled",
                "namespace_key": namespace_key,
                "message": msg,
                "cancel_requested": True,
            },
            ttl_seconds=_BUILD_CACHE_TTL,
        )
        log.info("build_deep_background: job=%s cancelled namespace=%s", job_id, namespace_key)
        raise
    except Exception as exc:
        log.exception("build_deep_background: job=%s FAILED: %s", job_id, exc)
        await cache.set(
            f"graph:build:{job_id}",
            {"status": "failed", "namespace_key": namespace_key, "error": str(exc), "message": str(exc)},
            ttl_seconds=_BUILD_CACHE_TTL,
        )
    finally:
        # Release the per-namespace lock so a new build can be started
        if lock_key:
            try:
                await cache.delete(lock_key)
            except Exception:
                pass


@router.post("/build-deep-bg", status_code=202)
async def build_deep_graph_background(
    user_id: CurrentUserID,
    db: DBSession,
    namespace_key: str | None = Query(default=None, description="Scope to a single namespace; omit for all"),
):
    """Queue Build Deep as a background job and return a job ID immediately.

    The full LLM taxonomy build + related-edge computation can take several
    minutes for large namespaces.  This endpoint fires the work off
    asynchronously and returns a ``job_id`` so the caller can poll
    ``GET /graph/build-deep/status/{job_id}`` for progress.

    Returns 202 Accepted with ``{job_id, status: "running"}``.
    """
    from app.repositories.user import UserRepository
    from fastapi import HTTPException as _HTTPException

    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    orientation = user.orientation.value if user else "both"

    cache = get_cache()

    # Idempotency: reject if a build for this namespace is already running.
    # Key: "graph:build:lock:{ns_hash}" exists only while a job is in flight.
    import hashlib as _hl
    ns_hash = _hl.sha256((namespace_key or "__all__").encode()).hexdigest()[:12]
    lock_key = f"graph:build:lock:{ns_hash}"
    existing_job_id = await cache.get(lock_key)
    if existing_job_id:
        return {
            "job_id": existing_job_id,
            "status": "running",
            "namespace_key": namespace_key,
            "message": "A build for this namespace is already in progress.",
        }

    job_id = str(_uuid.uuid4())
    # Set the per-namespace lock (TTL slightly longer than the expected max build time)
    await cache.set(lock_key, job_id, ttl_seconds=_BUILD_CACHE_TTL)
    await cache.set(
        f"graph:build:{job_id}",
        {"status": "running", "namespace_key": namespace_key, "lock_key": lock_key, "cancel_requested": False},
        ttl_seconds=_BUILD_CACHE_TTL,
    )

    task = asyncio.create_task(
        _run_build_deep_background(job_id, namespace_key, orientation, lock_key)
    )
    _BUILD_TASKS[job_id] = task
    task.add_done_callback(lambda _: _BUILD_TASKS.pop(job_id, None))

    return {
        "job_id": job_id,
        "status": "running",
        "namespace_key": namespace_key,
        "message": "Build Deep started in background. Poll /graph/build-deep/status/{job_id} for progress.",
    }


@router.get("/build-deep/status/{job_id}")
async def build_deep_status(job_id: str, user_id: CurrentUserID):
    """Poll the status of a background Build Deep job.

    Returns ``{status, message, result}`` where ``status`` is one of
    ``"running"``, ``"done"``, ``"failed"``, or ``"cancelled"``.
    """
    cache = get_cache()
    data = await cache.get(f"graph:build:{job_id}")
    if data is None:
        return {"status": "not_found", "message": "Job not found or expired (TTL 2 h)."}
    return data


@router.post("/build-deep/{job_id}/cancel", status_code=200)
async def cancel_build_deep(job_id: str, user_id: CurrentUserID):
    """Cancel a running Build Deep task and release its namespace lock."""
    cache = get_cache()
    data = await cache.get(f"graph:build:{job_id}")
    if data is None:
        return {"status": "not_found", "message": "Job not found or expired."}

    lock_key = data.get("lock_key")
    task = _BUILD_TASKS.get(job_id)
    if task and not task.done():
        task.cancel()

    data.update({
        "status": "cancelled",
        "cancel_requested": True,
        "message": "Graph build cancellation requested. Partial graph state was preserved.",
    })
    await cache.set(f"graph:build:{job_id}", data, ttl_seconds=_BUILD_CACHE_TTL)
    if lock_key:
        await cache.delete(lock_key)
    return {"status": "cancelled", "job_id": job_id}


@router.post("/deduplicate", status_code=200)
async def deduplicate_nodes(db: DBSession, user_id: CurrentUserID):
    """Merge duplicate graph nodes that share the same (label, node_type, namespace_key).

    Caused by a race condition when multiple Build Deep jobs run simultaneously and
    both try to create the same subject/topic node. Idempotent — safe to call repeatedly.
    """
    from app.models.graph import KnowledgeNode, KnowledgeEdge, NodeType
    from sqlalchemy import select, func, delete as sa_delete, update as sa_update

    # Find all (label, node_type, namespace_key) groups with more than one node
    dup_check = await db.execute(
        select(
            KnowledgeNode.label,
            KnowledgeNode.node_type,
            KnowledgeNode.namespace_key,
            func.count(KnowledgeNode.id).label("cnt"),
            func.min(KnowledgeNode.id).label("keep_id"),
        )
        .group_by(KnowledgeNode.label, KnowledgeNode.node_type, KnowledgeNode.namespace_key)
        .having(func.count(KnowledgeNode.id) > 1)
    )
    duplicates = dup_check.fetchall()

    merged = 0
    for row in duplicates:
        keep_id = row.keep_id
        # Find all IDs to drop
        victims = await db.execute(
            select(KnowledgeNode.id).where(
                KnowledgeNode.label == row.label,
                KnowledgeNode.node_type == row.node_type,
                KnowledgeNode.namespace_key == row.namespace_key,
                KnowledgeNode.id != keep_id,
            )
        )
        victim_ids = [r[0] for r in victims.fetchall()]
        for vid in victim_ids:
            # Re-point all edges that reference the victim to the kept node
            await db.execute(
                sa_update(KnowledgeEdge)
                .where(KnowledgeEdge.source_id == vid)
                .values(source_id=keep_id)
            )
            await db.execute(
                sa_update(KnowledgeEdge)
                .where(KnowledgeEdge.target_id == vid)
                .values(target_id=keep_id)
            )
            # Delete any self-loops created by the re-pointing
            await db.execute(
                sa_delete(KnowledgeEdge).where(
                    KnowledgeEdge.source_id == KnowledgeEdge.target_id
                )
            )
            # Remove duplicate edges (same source/target/type, keep one)
            dup_edges = await db.execute(
                select(KnowledgeEdge.id).where(
                    KnowledgeEdge.source_id == keep_id,
                    KnowledgeEdge.target_id == keep_id,
                )
            )
            # Delete the victim node
            await db.execute(sa_delete(KnowledgeNode).where(KnowledgeNode.id == vid))
            merged += 1

    await db.commit()
    # Clear cache so next graph load reflects the merged state
    await GraphService.clear_subgraph_cache(None)

    return {"merged": merged, "duplicate_groups": len(duplicates), "message": f"Merged {merged} duplicate node(s)."}


@router.post("/clear", status_code=200)
async def clear_graph(
    db: DBSession,
    user_id: CurrentUserID,
    namespace_keys: str | None = Query(default=None, description="Comma-separated namespaces to clear. Omit to clear all."),
):
    """Delete graph nodes/edges and caches scoped to the given namespaces (or all if omitted).

    Race-safety: any in-flight Build Deep tasks targeting an affected namespace
    are cancelled BEFORE the DELETE statements run. Without this, a running
    builder would continue inserting nodes/edges into a now-empty graph and
    leave an inconsistent partial state behind.
    """
    from app.models.graph import KnowledgeNode, KnowledgeEdge
    from sqlalchemy import delete, or_
    import hashlib as _hl

    ns_list: list[str] | None = None
    if namespace_keys:
        ns_list = [k.strip() for k in namespace_keys.split(",") if k.strip()]

    # Cancel any in-flight Build Deep job that targets a cleared namespace.
    # When no namespace filter is supplied we cancel every running build.
    cache = get_cache()
    cancelled_jobs: list[str] = []
    for job_id, task in list(_BUILD_TASKS.items()):
        if task.done():
            continue
        data = await cache.get(f"graph:build:{job_id}") or {}
        target_ns = data.get("namespace_key")
        if ns_list is not None and target_ns not in ns_list:
            continue
        task.cancel()
        data["cancel_requested"] = True
        data["status"] = "cancelled"
        await cache.set(f"graph:build:{job_id}", data, ttl_seconds=_BUILD_CACHE_TTL)
        lock_key = data.get("lock_key")
        if lock_key:
            await cache.delete(lock_key)
        cancelled_jobs.append(job_id)
    if cancelled_jobs:
        # Give cancellation a brief moment to land — short enough not to block
        # the request, long enough that most in-flight INSERTs return first.
        import asyncio as _a
        await _a.sleep(0.1)

    if ns_list:
        # Scoped clear: remove nodes whose namespace_key is in the list
        # (TOPIC nodes with namespace_key=None are shared — leave them unless all namespaces are cleared)
        node_ids_res = await db.execute(
            select(KnowledgeNode.id).where(KnowledgeNode.namespace_key.in_(ns_list))
        )
        scoped_ids = [r[0] for r in node_ids_res.fetchall()]
        if scoped_ids:
            await db.execute(
                delete(KnowledgeEdge).where(
                    or_(
                        KnowledgeEdge.source_id.in_(scoped_ids),
                        KnowledgeEdge.target_id.in_(scoped_ids),
                    )
                )
            )
            await db.execute(delete(KnowledgeNode).where(KnowledgeNode.id.in_(scoped_ids)))
        for ns in ns_list:
            GraphService._build_cache.pop(ns, None)
    else:
        # Full clear
        await db.execute(delete(KnowledgeEdge))
        await db.execute(delete(KnowledgeNode))
        GraphService._build_cache.clear()

    await db.commit()

    # Clear subgraph cache + build locks only for the affected namespaces
    targets = ns_list if ns_list else [None]  # None → global "__all__" hash
    for ns in targets:
        await GraphService.clear_subgraph_cache(ns)
        ns_hash = _hl.sha256((ns or "__all__").encode()).hexdigest()[:12]
        await cache.delete(f"graph:build:lock:{ns_hash}")
    # Always clear the global aggregated cache since any namespace change affects it
    await GraphService.clear_subgraph_cache(None)

    scope = ", ".join(ns_list) if ns_list else "all namespaces"
    msg = f"Graph cleared for {scope}. Run Build Deep to regenerate."
    if cancelled_jobs:
        msg += f" Cancelled {len(cancelled_jobs)} in-flight build(s)."
    return {"message": msg, "cancelled_jobs": cancelled_jobs}


@router.post("/cleanup", status_code=200)
async def cleanup_graph(
    db: DBSession,
    user_id: CurrentUserID,
    namespace_key: str | None = Query(default=None),
):
    """Remove stale SUBTOPIC nodes whose namespace_key is not a known arXiv category,
    plus any isolated CONCEPT/METHOD nodes with no paper connections. Then rewire.

    This fixes stray nodes like 'AI Systems' that appear when papers carry
    unusual namespace_keys not in the standard _NS_LABEL mapping.
    """
    from app.models.graph import KnowledgeNode, KnowledgeEdge, NodeType
    from sqlalchemy import delete

    svc = GraphService(db)

    # All valid namespace keys — derived from the curated _NS_LABEL mapping which
    # covers every arXiv namespace the service supports.  The old hardcoded 21-key
    # set was a subset and incorrectly removed valid subtopics like quant-ph, math.AG,
    # astro-ph.CO, hep-th, etc. when a user subscribed to those namespaces.
    from app.services.graph import _NS_LABEL
    known_ns = set(_NS_LABEL.keys())

    # Find SUBTOPIC nodes with unknown namespace_keys
    result = await db.execute(
        select(KnowledgeNode).where(
            KnowledgeNode.node_type == NodeType.subtopic,
            KnowledgeNode.namespace_key.isnot(None),
            KnowledgeNode.namespace_key.notin_(known_ns),
        )
    )
    stale_subtopics = list(result.scalars())
    stale_ids = [n.id for n in stale_subtopics]

    removed = 0
    if stale_ids:
        # Delete edges connected to stale subtopics
        await db.execute(
            delete(KnowledgeEdge).where(
                (KnowledgeEdge.source_id.in_(stale_ids)) |
                (KnowledgeEdge.target_id.in_(stale_ids))
            )
        )
        # Delete the stale subtopic nodes themselves
        await db.execute(
            delete(KnowledgeNode).where(KnowledgeNode.id.in_(stale_ids))
        )
        removed = len(stale_ids)

    # Clear the deep build cache so next Build Deep runs fresh
    GraphService._build_cache.clear()

    # Rewire orphaned paper nodes back into the correct hierarchy
    wired = await svc.rebuild_hierarchy()

    await db.commit()
    return {
        "removed_stale_subtopics": removed,
        "rewired_papers": wired,
        "message": (
            f"Cleaned up {removed} stale subtopic node(s) and rewired {wired} paper(s). "
            "Click 'Build Deep' to regenerate the deep taxonomy."
        ) if removed else "Graph is clean — no stale nodes found.",
    }
