"""Papers router — detail, recommendations, annotations."""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.core.deps import CurrentUserID, DBSession
from app.repositories.paper import PaperRepository
from app.adapters.embedding import get_embedding_adapter
from app.schemas import AnnotationRequest, PaperResponse

router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(paper_id: UUID, db: DBSession):
    """Return the full detail of a single paper by its UUID.

    Args:
        paper_id: UUID of the paper to retrieve.
        db: Injected async database session.

    Returns:
        A ``PaperResponse`` with all enrichment fields populated.

    Raises:
        HTTPException: 404 if no paper exists with the given ID.
    """
    repo = PaperRepository(db)
    paper = await repo.get_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperResponse.model_validate(paper)


@router.get("/{paper_id}/related", response_model=list[PaperResponse])
async def get_related_papers(paper_id: UUID, db: DBSession):
    """Return up to 5 papers semantically similar to the given paper.

    Uses pure cosine similarity on abstract embeddings (minimum 0.50) with a
    keyword fallback when no embedding is stored. The source paper itself is
    always excluded from results.

    Args:
        paper_id: UUID of the reference paper.
        db: Injected async database session.

    Returns:
        A list of up to 5 ``PaperResponse`` objects ordered by similarity.

    Raises:
        HTTPException: 404 if the reference paper is not found.
    """
    from app.repositories.search import SearchRepository
    paper_repo = PaperRepository(db)
    paper = await paper_repo.get_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    search_repo = SearchRepository(db)

    # Use the full abstract as the semantic query — this captures domain-specific
    # meaning far better than keyword matching on title/concepts, which picks up
    # generic ML terms and returns unrelated papers.
    sem_text = paper.abstract or paper.title

    import logging as _log
    _logger = _log.getLogger(__name__)
    related_papers: list = []

    try:
        embed = get_embedding_adapter()
        query_vec = await embed.embed_query(sem_text)
        # Pure semantic search: bypass RRF/keyword entirely so unrelated papers
        # with matching generic terms (e.g. "model", "neural") don't rank high.
        sem_results = await search_repo.semantic_search(
            query_vec,
            embedding_dim=embed.dimensions,
            embedding_provider=embed.provider_id,
        )
        # Filter: exclude the paper itself and apply a minimum similarity floor.
        MIN_SIM = 0.35
        candidate_ids = []
        for r in sem_results:
            if str(r["paper_id"]) == str(paper_id):
                continue
            if (r.get("sem_score") or 0) < MIN_SIM:
                break  # sorted desc, safe to stop early
            candidate_ids.append(r["paper_id"])
            if len(candidate_ids) >= 5:
                break

        # Batch-load all candidate papers in one query
        if candidate_ids:
            from sqlalchemy import select as _sel
            from app.models.paper import Paper as _Paper
            import uuid as _uuid
            batch = await db.execute(
                _sel(_Paper).where(_Paper.id.in_([
                    _uuid.UUID(str(cid)) for cid in candidate_ids
                ]))
            )
            paper_map = {str(p.id): p for p in batch.scalars()}
            for cid in candidate_ids:
                p = paper_map.get(str(cid))
                if p:
                    related_papers.append(PaperResponse.model_validate(p))

    except Exception as exc:
        _logger.warning("related papers semantic failed paper=%s: %s", paper_id, exc)
        # Keyword fallback using title + concepts only (no namespace bias)
        concepts = (paper.key_concepts or [])[:5]
        text_q = " ".join(filter(None, [paper.title, " ".join(concepts)]))
        results = await search_repo.hybrid_search(text_q, limit=10)
        fb_ids = [
            r["paper_id"] for r in results
            if str(r["paper_id"]) != str(paper_id)
        ][:5]
        if fb_ids:
            from sqlalchemy import select as _sel
            from app.models.paper import Paper as _Paper
            import uuid as _uuid
            batch = await db.execute(
                _sel(_Paper).where(_Paper.id.in_([_uuid.UUID(str(i)) for i in fb_ids]))
            )
            paper_map = {str(p.id): p for p in batch.scalars()}
            for i in fb_ids:
                p = paper_map.get(str(i))
                if p:
                    related_papers.append(PaperResponse.model_validate(p))

    return related_papers


@router.get("/{paper_id}/tldr")
async def get_paper_tldr(paper_id: UUID, db: DBSession):
    """Return cached TLDR or generate + save using the cheapest model."""
    repo = PaperRepository(db)
    paper = await repo.get_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    if paper.tldr:
        return {"tldr": paper.tldr}

    from app.adapters.llm import get_llm_adapter
    llm = get_llm_adapter()
    result = await llm.complete(
        messages=[{
            "role": "user",
            "content": (
                f"Summarize in ONE sentence (max 30 words, plain English, no jargon): "
                f"{paper.title}. Abstract: {paper.abstract[:600]}"
            ),
        }],
        model=llm.cheap_model,
        max_tokens=80,
        temperature=0.2,
    )
    tldr = result.text.strip().strip('"')
    await repo.update_enrichment(paper_id, {"tldr": tldr})
    await db.commit()
    return {"tldr": tldr}


@router.post("/generate-tldrs")
async def batch_generate_tldrs(db: DBSession, user_id: CurrentUserID, limit: int = 50):
    """Batch-generate TLDRs for papers that don't have one yet."""
    from sqlalchemy import select
    from app.models.paper import Paper
    from app.adapters.llm import get_llm_adapter
    import asyncio

    result = await db.execute(
        select(Paper).where(Paper.tldr.is_(None)).limit(limit)
    )
    papers = list(result.scalars())
    if not papers:
        return {"generated": 0, "message": "All papers already have TLDRs"}

    llm = get_llm_adapter()
    repo = PaperRepository(db)

    async def _gen(p: Paper) -> None:
        """Generate and persist a one-sentence TL;DR for a single paper."""
        try:
            r = await llm.complete(
                messages=[{
                    "role": "user",
                    "content": (
                        f"Summarize in ONE sentence (max 30 words, plain English, no jargon): "
                        f"{p.title}. Abstract: {p.abstract[:600]}"
                    ),
                }],
                model=llm.cheap_model,
                max_tokens=80,
                temperature=0.2,
            )
            await repo.update_enrichment(p.id, {"tldr": r.text.strip().strip('"')})
        except Exception:
            pass

    # Run in small concurrent batches to avoid rate limits
    for i in range(0, len(papers), 5):
        await asyncio.gather(*[_gen(p) for p in papers[i:i + 5]])

    await db.commit()
    return {"generated": len(papers)}


@router.post("/{paper_id}/annotate", status_code=201)
async def add_annotation(
    paper_id: UUID,
    body: AnnotationRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Save a highlighted text annotation for a paper.

    Args:
        paper_id: UUID of the paper being annotated.
        body: Annotation payload with ``highlighted_text`` and optional ``note``.
        user_id: UUID of the authenticated user.
        db: Injected async database session.

    Returns:
        A dict containing the ``id`` of the newly created annotation.
    """
    from app.repositories.user import UserRepository
    repo = UserRepository(db)
    ann = await repo.add_annotation(user_id, paper_id, body.highlighted_text, body.note)
    await db.commit()
    return {"id": str(ann.id)}
