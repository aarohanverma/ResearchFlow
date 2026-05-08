"""Feed router — personalized paper feed with scoring, feedback, and manual refresh."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.core.config import settings
from app.core.deps import CurrentUserID, DBSession
from app.repositories.graph import GraphRepository
from app.repositories.paper import PaperRepository
from app.repositories.user import UserRepository
from app.schemas import FeedbackRequest, FeedPaperResponse, FeedResponse, PaperResponse
from app.services.scoring import ScoringService

router = APIRouter(prefix="/feed", tags=["feed"])


@router.get("/suggested")
async def get_suggested(
    user_id: CurrentUserID,
    db: DBSession,
    limit: int = Query(default=12, le=30),
    namespace_keys: str | None = Query(default=None, description="Comma-separated namespace keys to restrict results"),
):
    """Recommend papers based on the user's liked + bookmarked paper concepts."""
    from collections import Counter

    paper_repo = PaperRepository(db)
    bookmarks = await paper_repo.get_bookmarks(user_id)
    liked_ids = await paper_repo.get_liked_paper_ids(user_id)

    sourced_ids: set[str] = set()
    all_concepts: list[str] = []

    for bm in bookmarks[:20]:
        sourced_ids.add(str(bm.paper_id))
        paper = await paper_repo.get_by_id(bm.paper_id)
        if paper:
            all_concepts.extend(paper.key_concepts or [])

    for pid_str in liked_ids[:20]:
        from uuid import UUID
        sourced_ids.add(pid_str)
        try:
            paper = await paper_repo.get_by_id(UUID(pid_str))
            if paper:
                all_concepts.extend(paper.key_concepts or [])
        except Exception:
            pass

    _STOP = {"a", "an", "the", "of", "in", "on", "for", "and", "or", "with", "to", "from", "by", "is", "are", "using", "based", "via"}

    if not all_concepts:
        # Enrichment hasn't run yet — extract meaningful keywords from bookmarked titles
        title_terms: list[str] = []
        for bm in bookmarks[:10]:
            paper = await paper_repo.get_by_id(bm.paper_id)
            if paper:
                words = [w.strip(":.,-()") for w in paper.title.split()
                         if len(w) > 3 and w.lower().strip(":.,-()") not in _STOP]
                title_terms.extend(words[:5])
        if not title_terms:
            return {"suggestions": [], "based_on": []}
        # Deduplicate preserving order
        seen: set[str] = set()
        deduped = [w for w in title_terms if not (w.lower() in seen or seen.add(w.lower()))]  # type: ignore[func-returns-value]
        top_concepts = deduped[:8]
        query = " ".join(top_concepts[:6])
    else:
        top_concepts = [c for c, _ in Counter(all_concepts).most_common(8)]
        query = " ".join(top_concepts[:6])

    ns_list: list[str] | None = None
    if namespace_keys:
        ns_list = [k.strip() for k in namespace_keys.split(",") if k.strip()]

    from app.repositories.search import SearchRepository
    search_repo = SearchRepository(db)

    try:
        from app.adapters.embedding import get_embedding_adapter
        embed = get_embedding_adapter()
        vec = await embed.embed_query(query)
        results = await search_repo.hybrid_search(
            query,
            query_vector=vec,
            embedding_dim=embed.dimensions,
            embedding_provider=embed.provider_id,
            namespace_keys=ns_list,
            limit=limit + len(sourced_ids) + 10,
        )
    except Exception:
        results = await search_repo.hybrid_search(query, namespace_keys=ns_list, limit=limit + len(sourced_ids) + 10)

    suggestions = [r for r in results if str(r["paper_id"]) not in sourced_ids][:limit]
    return {"suggestions": suggestions, "based_on": top_concepts}


@router.get("", response_model=FeedResponse)
async def get_feed(
    user_id: CurrentUserID,
    db: DBSession,
    namespace_key: str = Query(..., description="e.g. cs.AI"),
    limit: int = Query(default=30, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Return the personalised paper feed for a namespace.

    Scores papers using the user's interest profile (hot/cold subtopics and
    orientation) via ``ScoringService``, then slices with ``offset``/``limit``.

    Args:
        user_id: UUID of the authenticated user.
        db: Injected async database session.
        namespace_key: arXiv namespace to fetch (e.g. ``"cs.AI"``).
        limit: Maximum number of papers to return (max 100).
        offset: Number of papers to skip before returning results.

    Returns:
        A ``FeedResponse`` containing scored papers and total count.
    """
    user_repo = UserRepository(db)
    user = await user_repo.get_by_id(user_id)
    if user is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="User not found — please log in again.")
    profile = await user_repo.get_interest_profile(user_id)

    scoring = ScoringService(db)
    scored = await scoring.score_papers_for_user(
        user_id=user_id,
        namespace_key=namespace_key,
        orientation=user.orientation,
        hot_subtopics=profile.hot_subtopics if profile else [],
        cold_subtopics=profile.cold_subtopics if profile else [],
        limit=limit + offset,  # over-fetch so offset slice is valid
    )
    scored = scored[offset:]  # apply offset after scoring

    papers = [
        FeedPaperResponse(
            paper=PaperResponse.model_validate(item["paper"]),
            score=item["score"],
            why_tag=item["why_tag"],
        )
        for item in scored
    ]

    return FeedResponse(papers=papers, total=len(papers), namespace_key=namespace_key)


@router.get("/liked-ids", response_model=list[str])
async def get_liked_paper_ids(user_id: CurrentUserID, db: DBSession):
    """Return paper IDs the current user has liked."""
    paper_repo = PaperRepository(db)
    return await paper_repo.get_liked_paper_ids(user_id)


@router.delete("/liked/{paper_id}", status_code=204)
async def unlike_paper(paper_id: str, user_id: CurrentUserID, db: DBSession):
    """Remove a like for the given paper."""
    from uuid import UUID
    paper_repo = PaperRepository(db)
    await paper_repo.remove_feedback(user_id, UUID(paper_id), "like")
    await db.commit()


@router.post("/feedback", status_code=204)
async def submit_feedback(body: FeedbackRequest, user_id: CurrentUserID, db: DBSession):
    """Record a feed signal and update the user's interest profile.

    Persists the signal (``like``, ``dismiss``, or ``more_like_this``) to
    ``feed_feedback``, then adjusts the user's ``hot_subtopics`` /
    ``cold_subtopics`` based on the paper's ``key_concepts``. Both lists are
    capped at 60 entries to prevent unbounded growth.

    Args:
        body: Feedback payload containing ``paper_id`` and ``signal``.
        user_id: UUID of the authenticated user.
        db: Injected async database session.
    """
    paper_repo = PaperRepository(db)
    await paper_repo.add_feedback(user_id, body.paper_id, body.signal)

    # Update interest profile so relevance scores drift toward user taste
    paper = await paper_repo.get_by_id(body.paper_id)
    if paper:
        user_repo = UserRepository(db)
        profile = await user_repo.get_interest_profile(user_id)
        if profile:
            hot: list[str] = list(profile.hot_subtopics or [])
            cold: list[str] = list(profile.cold_subtopics or [])
            signals = paper.key_concepts or []

            if body.signal in ("like", "save"):
                for concept in signals:
                    if concept not in hot:
                        hot.append(concept)
                    if concept in cold:
                        cold.remove(concept)
            elif body.signal == "dismiss":
                for concept in signals:
                    if concept not in cold:
                        cold.append(concept)
                    if concept in hot:
                        hot.remove(concept)

            # Keep lists bounded to avoid runaway growth
            await user_repo.update_interest_profile(user_id, hot[-60:], cold[-60:])

    await db.commit()


@router.get("/papers/{paper_id}/related")
async def get_related_papers(
    paper_id: str,
    db: DBSession,
    user_id: CurrentUserID,
    limit: int = Query(default=5, le=10),
    namespace_keys: str | None = Query(default=None, description="Comma-separated namespace keys to restrict results"),
):
    """Return papers semantically similar to the given paper (max 5).

    Uses pure cosine similarity on abstract embeddings — no keyword mixing —
    so results are domain-specific rather than generic cs.AI matches.
    """
    from uuid import UUID as _UUID
    from sqlalchemy import select as _select
    from app.models.paper import PaperChunk as _Chunk
    from app.repositories.search import SearchRepository

    try:
        pid = _UUID(paper_id)
    except ValueError:
        return []

    paper_repo = PaperRepository(db)
    paper = await paper_repo.get_by_id(pid)
    if not paper:
        return []

    ns_list: list[str] | None = None
    if namespace_keys:
        ns_list = [k.strip() for k in namespace_keys.split(",") if k.strip()]

    search_repo = SearchRepository(db)
    MIN_SIM = 0.35  # minimum cosine similarity for related papers (lowered for better recall)

    # Prefer the paper's own stored embedding chunk for exact similarity
    chunk_q = await db.execute(
        _select(_Chunk).where(
            _Chunk.paper_id == pid,
            _Chunk.embedding.is_not(None),
        ).limit(1)
    )
    chunk = chunk_q.scalar_one_or_none()

    if chunk and chunk.embedding is not None:
        provider = chunk.embedding_provider.value if hasattr(chunk.embedding_provider, "value") else str(chunk.embedding_provider)
        sem_results = await search_repo._semantic_search(
            list(chunk.embedding),
            embedding_dim=chunk.embedding_dim,
            embedding_provider=provider,
            namespace_keys=ns_list,
        )
        out = []
        for r in sem_results:
            if str(r["paper_id"]) == paper_id:
                continue
            if (r.get("sem_score") or 0) < MIN_SIM:
                break  # sorted desc, safe to stop early
            out.append(r)
            if len(out) >= limit:
                break
        return out

    # Fallback: no embedding stored yet — keyword search on title + concepts only
    concepts = (paper.key_concepts or [])[:5]
    text_q = " ".join(filter(None, [paper.title, " ".join(concepts)]))
    results = await search_repo.hybrid_search(text_q, namespace_keys=ns_list, limit=limit + 5)
    return [r for r in results if str(r["paper_id"]) != paper_id][:limit]


@router.post("/refresh")
async def refresh_feed(
    user_id: CurrentUserID,
    db: DBSession,
    background_tasks: BackgroundTasks,
    namespace_key: str = Query(..., description="Namespace to refresh, e.g. cs.AI"),
):
    """Manually trigger arXiv RSS ingestion for a namespace.

    Runs in the background so the response returns immediately.
    New papers appear in the feed within 30–60 seconds for typical namespaces.
    """
    from app.services.namespace import NAMESPACE_TO_ARXIV

    if namespace_key not in NAMESPACE_TO_ARXIV:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown namespace '{namespace_key}'. Valid: {sorted(NAMESPACE_TO_ARXIV)}",
        )

    # Ensure a SourceMapping exists for this namespace
    graph_repo = GraphRepository(db)
    mappings = await graph_repo.get_source_mappings(namespace_key)

    if not mappings:
        from app.models.graph import SourceMapping

        arxiv_cat = NAMESPACE_TO_ARXIV[namespace_key]
        db.add(SourceMapping(
            namespace_key=namespace_key,
            source_name="arxiv_rss" if settings.ingestion_mode == "rss" else "arxiv_mcp",
            external_category_key=arxiv_cat,
        ))
        await db.commit()

    async def _trigger():
        """Background task: run the ingestion workflow for the requested namespace."""
        from app.workflows.ingestion import run_ingestion
        try:
            await run_ingestion(namespace_key)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("manual refresh failed ns=%s err=%s", namespace_key, exc)

    background_tasks.add_task(_trigger)

    return {
        "triggered": True,
        "namespace_key": namespace_key,
        "message": f"Ingestion started for {namespace_key}. New papers appear in 30–60s.",
    }
