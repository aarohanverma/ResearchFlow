"""Chunk-level provenance evidence.

The existing :mod:`app.assistant.provenance_verification` checks each
claim against the cited paper's *title + abstract + tldr* via lexical
overlap. That catches the loud failure mode ("cited paper has nothing
to do with the topic"), but it stops at paper level.

This module deepens provenance to **chunk level**. For every supported
or unverified claim that points at a corpus paper, we:

  1. Pull the paper's :class:`PaperChunk` rows (we have these in
     Postgres with embeddings already populated during ingestion).
  2. Embed the claim sentence via the existing embedding adapter.
  3. Rank chunks by cosine similarity.
  4. Return the top-2 chunks as :class:`ChunkEvidence` records
     containing the chunk id, paper id, similarity score, and the
     exact text excerpt that scores highest.

The result is what the UI surfaces as "the specific paragraph that
supports this claim" — clickable from the inline ``[N]`` marker
to the chunk text.

Cost discipline:

* **One embedding batch per turn**. We collect all claim sentences
  that need lookup, batch-embed them once, then run the per-paper
  similarity locally. This bounds the LLM cost to a single
  ``embed_texts`` call regardless of how many claims the answer
  contains.
* **Cap per turn** via ``MAX_CLAIMS_PER_TURN`` so a 30-citation
  answer doesn't blow the budget. The cap drops the *unsupported*
  / *unverified* claims first; supported claims always get evidence
  pointers when possible.
* **Cache-friendly** — chunks are pulled per (paper_id) once per
  turn even when multiple claims cite the same paper.

The chunk evidence lands on the synthesizer's ``output``
side-channel as ``output["provenance"]["chunk_evidence"]`` so the
orchestrator can persist it into ``payload.provenance`` for the UI
without an additional plumbing pass.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant.provenance_verification import ClaimVerdict
from app.models.paper import PaperChunk

log = logging.getLogger(__name__)


# ── Tuneables ────────────────────────────────────────────────────────────────


# Cap claims-per-turn for chunk lookup. A typical 600-word answer has
# 8-15 citation markers; we want headroom but not unbounded growth.
MAX_CLAIMS_PER_TURN = 20

# Top-K chunks to return per claim. 2 is the inflection point: one
# chunk is often a single sentence the model could clip; two gives
# the user enough surrounding context without bloating the UI.
TOP_K_CHUNKS = 2

# Skip the lookup when there's not enough overlap signal. The
# verifier already produced a similarity-ish overlap_score; below
# this floor the chunk lookup is unlikely to find anything useful
# anyway — save the embedding call.
MIN_OVERLAP_FOR_LOOKUP = 0.10

# Hard cap on chunks fetched per paper. Large papers have 30+ chunks;
# embedding-comparing all of them is fine, but we only need the
# best ones. The DB query keeps everything cheap with pgvector index.
_CHUNK_FETCH_LIMIT_PER_PAPER = 60


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class ChunkEvidence:
    """One paper chunk that grounds a specific claim.

    The synthesizer attaches these to claims in ``output["provenance"]
    ["chunk_evidence"]``. The frontend renders each chunk as a hover-
    preview / click-through from the corresponding inline ``[N]``
    marker so the user can audit "did the cited paper actually say
    this?" in one click.
    """

    paper_id: str
    chunk_id: str
    chunk_index: int
    section_type: str
    similarity: float
    excerpt: str            # the chunk's text, truncated for display


@dataclass
class ClaimChunkLink:
    """One (claim, [chunks]) record — the answer the verifier returns."""

    marker: str
    claim: str
    paper_id: str
    paper_title: str
    chunks: list[ChunkEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "claim": self.claim[:300],
            "paper_id": self.paper_id,
            "paper_title": self.paper_title[:160],
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "paper_id": c.paper_id,
                    "chunk_index": c.chunk_index,
                    "section_type": c.section_type,
                    "similarity": round(c.similarity, 3),
                    "excerpt": c.excerpt[:480],
                }
                for c in self.chunks
            ],
        }


# ── Public API ───────────────────────────────────────────────────────────────


async def attach_chunk_evidence(
    *,
    db: AsyncSession,
    claim_verdicts: list[ClaimVerdict],
    papers: list[dict],
) -> list[ClaimChunkLink]:
    """For every (claim, paper) verdict, find the top-K supporting chunks.

    Skips claims whose verdict is ``"unsupported"`` AND
    ``overlap_score`` < ``MIN_OVERLAP_FOR_LOOKUP`` — for those the
    paper genuinely isn't the right citation and chunk-level lookup
    would surface arbitrary near-misses.

    Returns a list of :class:`ClaimChunkLink`, one per claim that
    received at least one chunk pointer. Claims with no chunks (paper
    not ingested with embeddings, no semantic match above the floor)
    are omitted rather than returned with an empty list.

    Failures are swallowed — chunk-level provenance is best-effort.
    The deterministic + LLM-escalation passes have already verified
    the claims; the chunk pointers are an additional UX layer.
    """
    if not claim_verdicts or db is None:
        return []

    # Filter to claims worth looking up + sort so supported claims
    # win the budget when we exceed the per-turn cap.
    eligible = [
        v for v in claim_verdicts
        if v.paper_id
        and not (v.verdict == "unsupported" and v.overlap_score < MIN_OVERLAP_FOR_LOOKUP)
    ]
    eligible.sort(key=_priority_key)
    eligible = eligible[:MAX_CLAIMS_PER_TURN]
    if not eligible:
        return []

    paper_lookup = {str(p.get("paper_id") or ""): p for p in (papers or [])}
    unique_paper_ids: list[str] = []
    seen: set[str] = set()
    for v in eligible:
        if v.paper_id not in seen:
            seen.add(v.paper_id)
            unique_paper_ids.append(v.paper_id)

    chunks_by_paper = await _load_paper_chunks(db, unique_paper_ids)
    if not chunks_by_paper:
        log.debug("attach_chunk_evidence: no chunks found for any cited paper")
        return []

    # Batch-embed every distinct claim text in one shot.
    claim_texts = [v.claim for v in eligible]
    embeddings = await _embed_batch(claim_texts)
    if not embeddings or len(embeddings) != len(eligible):
        log.debug("attach_chunk_evidence: embedding batch failed; skipping")
        return []

    out: list[ClaimChunkLink] = []
    for verdict, claim_vec in zip(eligible, embeddings):
        chunks = chunks_by_paper.get(verdict.paper_id) or []
        if not chunks or not claim_vec:
            continue
        ranked = _rank_chunks_by_similarity(claim_vec, chunks)
        top = ranked[:TOP_K_CHUNKS]
        if not top:
            continue
        paper = paper_lookup.get(verdict.paper_id, {})
        out.append(ClaimChunkLink(
            marker=verdict.marker,
            claim=verdict.claim,
            paper_id=verdict.paper_id,
            paper_title=str(paper.get("title") or verdict.paper_title or ""),
            chunks=top,
        ))
    return out


# ── Internals ────────────────────────────────────────────────────────────────


def _priority_key(verdict: ClaimVerdict) -> tuple[int, float]:
    """Sort eligible claims so we spend chunk-lookup budget on the
    most informative ones. Supported claims first (the answer's
    backbone), then unverified, then high-overlap unsupported
    (potential mis-cites worth surfacing). Within each tier, higher
    overlap wins."""
    tier = {"supported": 0, "unverified": 1, "unsupported": 2}.get(verdict.verdict, 3)
    return (tier, -verdict.overlap_score)


async def _load_paper_chunks(
    db: AsyncSession,
    paper_ids: list[str],
) -> dict[str, list[PaperChunk]]:
    """Pull chunks for every cited paper in one query. Returns a
    paper_id → [chunk] dict (paper_id stringified for cross-type
    safety since marker → paper_id resolution may yield strings)."""
    if not paper_ids:
        return {}
    import uuid as _uuid

    uuid_keys: list[_uuid.UUID] = []
    str_to_uuid: dict[str, _uuid.UUID] = {}
    for pid in paper_ids:
        try:
            u = _uuid.UUID(str(pid))
        except (ValueError, TypeError):
            continue
        uuid_keys.append(u)
        str_to_uuid[str(pid)] = u
    if not uuid_keys:
        return {}

    stmt = (
        select(PaperChunk)
        .where(PaperChunk.paper_id.in_(uuid_keys))
        .order_by(PaperChunk.paper_id, PaperChunk.chunk_index)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    grouped: dict[str, list[PaperChunk]] = {}
    for chunk in rows:
        key = str(chunk.paper_id)
        grouped.setdefault(key, []).append(chunk)
    # Cap per paper so a pathologically chunked PDF doesn't dominate
    # the per-claim ranking budget.
    for key, chunks in grouped.items():
        if len(chunks) > _CHUNK_FETCH_LIMIT_PER_PAPER:
            grouped[key] = chunks[:_CHUNK_FETCH_LIMIT_PER_PAPER]
    return grouped


async def _embed_batch(texts: list[str]) -> list[list[float] | None]:
    """Embed a batch of claim sentences via the existing adapter.

    Returns ``[None, ...]`` on failure rather than raising — the
    chunk-evidence pass is best-effort.
    """
    try:
        from app.adapters.embedding import get_embedding_adapter
        adapter = get_embedding_adapter()
        vectors = await adapter.embed_texts(
            [t[:1000] for t in texts],
            task_type="RETRIEVAL_QUERY",
        )
        if not vectors or len(vectors) != len(texts):
            return [None] * len(texts)
        return vectors
    except Exception as exc:  # noqa: BLE001 — chunk evidence must never raise
        log.debug("provenance_evidence: embedding batch failed: %s", exc)
        return [None] * len(texts)


def _rank_chunks_by_similarity(
    claim_vec: list[float],
    chunks: list[PaperChunk],
) -> list[ChunkEvidence]:
    """Rank chunks by cosine similarity to the claim embedding.

    Chunks without an embedding stay out of the ranking. The
    similarity is normalised cosine in [-1, 1]; we filter to
    non-trivial positive matches (≥0.30) so the UI doesn't render
    pointers for chunks that aren't actually about the claim's topic.
    """
    if not claim_vec:
        return []
    claim_norm = _l2_norm(claim_vec)
    if claim_norm == 0:
        return []

    scored: list[tuple[float, PaperChunk]] = []
    for chunk in chunks:
        # ``chunk.embedding`` is a pgvector column — depending on the
        # pgvector Python binding it can come back as a ``numpy.ndarray``.
        # ``not numpy_array`` raises ``ValueError: ambiguous truth value``,
        # so use an explicit ``is None`` check + length probe instead of a
        # truthy test.
        if chunk.embedding is None:
            continue
        try:
            if len(chunk.embedding) == 0:
                continue
        except TypeError:
            # Non-sized type — treat as missing rather than raising.
            continue
        chunk_norm = _l2_norm(chunk.embedding)
        if chunk_norm == 0:
            continue
        sim = _dot(claim_vec, chunk.embedding) / (claim_norm * chunk_norm)
        if sim < 0.30:
            continue
        scored.append((sim, chunk))

    scored.sort(key=lambda x: -x[0])
    return [
        ChunkEvidence(
            paper_id=str(chunk.paper_id),
            chunk_id=str(chunk.id),
            chunk_index=int(chunk.chunk_index),
            section_type=str(chunk.section_type),
            similarity=float(sim),
            excerpt=(chunk.content or "").strip()[:600],
        )
        for sim, chunk in scored
    ]


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _dot(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))
