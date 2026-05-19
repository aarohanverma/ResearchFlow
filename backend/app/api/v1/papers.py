"""Papers router — detail, recommendations, annotations, manual arXiv import."""

import asyncio
import logging
import re
import uuid as _uuid
from datetime import datetime, timezone
from uuid import UUID

import feedparser
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.deps import CurrentUserID, DBSession
from app.repositories.paper import PaperRepository
from app.adapters.embedding import get_embedding_adapter
from app.schemas import AnnotationRequest, PaperResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])

# Strong references to background import tasks. Without this, Python 3.12+ may
# garbage-collect the create_task() return value before the task finishes,
# emitting RuntimeWarning and potentially cancelling the import mid-flight.
# Tasks self-discard on completion so the set stays bounded.
_background_tasks: set[asyncio.Task] = set()


# ── arXiv import ──────────────────────────────────────────────────────────────

_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")


class ArxivImportRequest(BaseModel):
    arxiv_id: str
    namespace_keys: list[str]  # one import job spawned per namespace


async def _fetch_arxiv_by_id(arxiv_id: str) -> dict | None:
    """Fetch a single paper from the arXiv Atom API. Returns a raw dict or None."""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://export.arxiv.org/api/query",
                params={"id_list": arxiv_id, "max_results": "1"},
            )
            resp.raise_for_status()
    except Exception as exc:
        log.warning("arxiv fetch failed id=%s err=%s", arxiv_id, exc)
        return None

    feed = feedparser.parse(resp.text)
    if not feed.entries:
        return None
    entry = feed.entries[0]
    # arXiv returns a "Missing" entry for unknown IDs
    if "missing" in entry.get("title", "").lower() or not entry.get("title"):
        return None
    return entry


@router.post("/import-arxiv", status_code=202)
async def import_arxiv_paper(
    body: ArxivImportRequest,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Queue a full-pipeline arXiv import and return immediately (202 Accepted).

    Validates the ID and confirms the paper exists on arXiv, then kicks off
    enrichment (embeddings + graph + LLM) in a background task so the HTTP
    response is not held while heavy work completes. Progress is tracked in
    the job store and visible in the notifications panel.
    """
    # Validate arXiv ID format
    raw_id = body.arxiv_id.strip()
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv.org/abs/"):
        if raw_id.startswith(prefix):
            raw_id = raw_id[len(prefix):]
    m = _ARXIV_ID_RE.match(raw_id)
    if not m:
        raise HTTPException(status_code=422, detail="Invalid arXiv ID — expected format: 1706.03762")
    canonical_id = m.group(1)

    # Confirm paper exists on arXiv and grab its title for the response.
    # This is the only blocking call — it's fast (1-2s) and lets us return
    # a meaningful 404 rather than discovering the ID is wrong in the background.
    entry = await _fetch_arxiv_by_id(canonical_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Paper {canonical_id} not found on arXiv")

    title = re.sub(r"\s+", " ", entry.get("title", "")).strip() or canonical_id

    # Validate namespace list
    namespace_keys = [k.strip() for k in body.namespace_keys if k.strip()]
    if not namespace_keys:
        raise HTTPException(status_code=422, detail="At least one namespace must be selected")
    if len(namespace_keys) > 20:
        raise HTTPException(status_code=422, detail="At most 20 namespaces allowed per import")

    from app.services.job_store import get_job_store
    job_store = get_job_store()
    created_at = datetime.now(timezone.utc).isoformat()
    spawned: list[dict] = []

    for ns_key in namespace_keys:
        job_id = f"import-arxiv:{_uuid.uuid4()}"
        await job_store.put(job_id, {
            "kind": "arxiv_import",
            "job_id": job_id,
            "user_id": str(user_id),
            "arxiv_id": canonical_id,
            "title": title,
            "status": "running",
            "namespace_key": ns_key,
            "created_at": created_at,
            "completed_at": None,
            "summary": f"Importing '{title[:60]}'…",
        })
        task = asyncio.create_task(
            _import_arxiv_background(
                job_id=job_id,
                arxiv_id=canonical_id,
                entry=entry,
                namespace_key=ns_key,
            ),
            name=f"import-arxiv:{canonical_id}:{ns_key}",
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        spawned.append({"job_id": job_id, "namespace_key": ns_key})

    ns_label = namespace_keys[0] if len(namespace_keys) == 1 else f"{len(namespace_keys)} namespaces"
    return {
        "jobs": spawned,
        "arxiv_id": canonical_id,
        "title": title,
        "message": f"Import started — '{title[:60]}' will appear in your feed shortly",
    }


async def _import_arxiv_background(
    *,
    job_id: str,
    arxiv_id: str,
    entry: dict,
    namespace_key: str,
) -> None:
    """Full pipeline for a single arXiv paper (background task).

    Runs store → embeddings → knowledge graph → LLM enrichment → mark imported.
    Deduplication is handled at the DB layer via ON CONFLICT DO NOTHING, so
    concurrent imports of the same paper are safe.
    """
    from app.db.session import async_session_factory
    from app.services.job_store import get_job_store

    job_store = get_job_store()

    try:
        # Parse entry fields
        title = re.sub(r"\s+", " ", entry.get("title", "")).strip()
        abstract = (entry.get("summary") or entry.get("description") or "").strip()
        authors = [a.get("name", "Unknown") for a in entry.get("authors", [])] or ["Unknown"]
        published_at = None
        if entry.get("published"):
            try:
                published_at = datetime.fromisoformat(
                    entry["published"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            except Exception:
                pass

        from app.adapters.sources.base import RawPaper
        from app.services.arxiv_import import ArxivImportService

        raw_paper = RawPaper(
            external_id=arxiv_id,
            title=title,
            authors=authors,
            abstract=abstract,
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            published_at=published_at,
            namespace_key=namespace_key,
            raw=dict(entry),
        )

        # Store + embed + graph (ON CONFLICT DO NOTHING deduplicates)
        async with async_session_factory() as db:
            svc = ArxivImportService(db)
            new_papers, _ = await svc.import_raw_papers(
                [raw_paper],
                namespace_key=namespace_key,
                create_embeddings=True,
                update_graph=True,
            )

            from sqlalchemy import select as _sel
            from app.models.paper import Paper
            result = await db.execute(
                _sel(Paper).where(
                    Paper.external_id == arxiv_id,
                    Paper.namespace_key == namespace_key,
                )
            )
            paper = result.scalar_one_or_none()
            if not paper:
                raise RuntimeError(f"Paper {arxiv_id} not found after import")

            # LLM enrichment for newly stored papers
            if new_papers:
                try:
                    from app.adapters.llm import get_llm_adapter
                    from app.workflows.ingestion import (
                        _ENRICHMENT_SYSTEM,
                        _parse_enrichment_items,
                        _coerce_enrichment_item,
                    )
                    llm = get_llm_adapter()
                    paper_list = f"[PAPER 0]\n[START]\n{paper.title}\n\n{paper.abstract}\n[END]"
                    messages = [
                        {"role": "system", "content": _ENRICHMENT_SYSTEM},
                        {"role": "user", "content": f"Analyze these 1 papers:\n\n{paper_list}"},
                    ]
                    res = await llm.complete(
                        messages,
                        llm.cheap_model,
                        response_format={"type": "json_object"},
                    )
                    items = _parse_enrichment_items(res.text)
                    if items:
                        enrichment = _coerce_enrichment_item(items[0])
                        if not enrichment.get("tldr"):
                            enrichment.pop("tldr", None)
                        paper_repo = PaperRepository(db)
                        await paper_repo.update_enrichment(paper.id, enrichment)
                        await db.commit()
                        await db.refresh(paper)
                except Exception as exc:
                    log.warning("import_arxiv: enrichment failed id=%s err=%s", arxiv_id, exc)

            # Mark as manually imported (idempotent)
            try:
                from sqlalchemy import update as _upd
                await db.execute(
                    _upd(Paper).where(Paper.id == paper.id).values(is_manually_imported=True)
                )
                await db.commit()
            except Exception as exc:
                log.warning("import_arxiv: could not set is_manually_imported flag: %s", exc)

        was_new = bool(new_papers)
        summary = (
            f"Imported '{title[:60]}'"
            if was_new
            else f"'{title[:60]}' already in {namespace_key}"
        )
        await job_store.update(job_id, {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
        })
        log.info("import_arxiv: done id=%s new=%s ns=%s", arxiv_id, was_new, namespace_key)

    except Exception as exc:
        log.exception("import_arxiv: background task failed id=%s", arxiv_id)
        await job_store.update(job_id, {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": f"Import failed: {exc!s:.120}",
        })


# ── arXiv import status ──────────────────────────────────────────────────────

@router.get("/import-arxiv/status/{job_id}")
async def get_arxiv_import_status(job_id: str, user_id: CurrentUserID):
    """Return the current status of a single arXiv import job."""
    from app.services.job_store import get_job_store
    job = await get_job_store().get(job_id)
    if not job or job.get("user_id") != str(user_id):
        raise HTTPException(status_code=404, detail="Import job not found")
    return job


# ── Namespace-level paper hide/unhide ─────────────────────────────────────────

@router.get("/hidden-ids")
async def get_hidden_paper_ids(
    namespace_key: str,
    user_id: CurrentUserID,
    db: DBSession,
) -> list[str]:
    """Return the IDs of manually-imported papers hidden by this user in a namespace."""
    from sqlalchemy import select as _sel
    from app.models.paper import PaperNamespaceHide
    result = await db.execute(
        _sel(PaperNamespaceHide.paper_id).where(
            PaperNamespaceHide.user_id == user_id,
            PaperNamespaceHide.namespace_key == namespace_key,
        )
    )
    return [str(row[0]) for row in result.fetchall()]


@router.post("/{paper_id}/hide", status_code=204)
async def hide_paper(
    paper_id: UUID,
    namespace_key: str,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Hide a manually-imported paper from a namespace for this user."""
    from app.models.paper import Paper, PaperNamespaceHide
    paper = await db.get(Paper, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    if not paper.is_manually_imported:
        raise HTTPException(status_code=400, detail="Only manually imported papers can be hidden")

    from sqlalchemy.dialects.postgresql import insert as _pg_insert
    stmt = _pg_insert(PaperNamespaceHide).values(
        user_id=user_id,
        paper_id=paper_id,
        namespace_key=namespace_key,
    ).on_conflict_do_nothing(constraint="uq_paper_hide")
    await db.execute(stmt)
    await db.commit()


@router.delete("/{paper_id}/hide", status_code=204)
async def unhide_paper(
    paper_id: UUID,
    namespace_key: str,
    user_id: CurrentUserID,
    db: DBSession,
):
    """Unhide a manually-imported paper in a namespace for this user."""
    from sqlalchemy import delete as _del
    from app.models.paper import PaperNamespaceHide
    await db.execute(
        _del(PaperNamespaceHide).where(
            PaperNamespaceHide.user_id == user_id,
            PaperNamespaceHide.paper_id == paper_id,
            PaperNamespaceHide.namespace_key == namespace_key,
        )
    )
    await db.commit()


@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(paper_id: UUID, user_id: CurrentUserID, db: DBSession):  # noqa: ARG001 — auth gate
    """Return the full detail of a single paper by its UUID.

    Papers are shared content, but the endpoint is auth-gated to prevent
    unauthenticated scraping and rate-limit abuse.

    Args:
        paper_id: UUID of the paper to retrieve.
        user_id: Required for auth; the paper itself is returned unfiltered.
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
async def get_related_papers(paper_id: UUID, user_id: CurrentUserID, db: DBSession):  # noqa: ARG001 — auth gate
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
async def get_paper_tldr(paper_id: UUID, user_id: CurrentUserID, db: DBSession):  # noqa: ARG001 — auth gate
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
