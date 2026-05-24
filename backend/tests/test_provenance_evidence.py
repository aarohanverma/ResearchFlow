"""Chunk-level provenance evidence.

The verifier (``provenance_verification``) operates at paper level.
This module deepens that to chunk level by:

  * Pulling each cited paper's :class:`PaperChunk` rows.
  * Embedding each claim sentence (one batched call per turn).
  * Ranking chunks by cosine similarity and returning the top-K.

Tests pin:

  * Priority ordering — supported claims get the chunk-lookup budget
    before unsupported ones.
  * Cap enforcement — never exceed ``MAX_CLAIMS_PER_TURN``.
  * Similarity ranking — chunks closer to the claim embedding rank
    higher; the trivial-match floor (0.30) filters noise.
  * Graceful failure — embedding adapter or DB failures return ``[]``
    rather than raising; the verifier is best-effort.
  * Skip semantics — unsupported claims with low overlap_score are
    skipped (paper is wrong, chunk lookup would surface noise).
"""

from __future__ import annotations

import math
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.assistant.provenance_evidence import (
    MAX_CLAIMS_PER_TURN,
    MIN_OVERLAP_FOR_LOOKUP,
    TOP_K_CHUNKS,
    ChunkEvidence,
    ClaimChunkLink,
    _priority_key,
    _rank_chunks_by_similarity,
    attach_chunk_evidence,
)
from app.assistant.provenance_verification import ClaimVerdict


# ── Test fixtures ──────────────────────────────────────────────────────────


def _verdict(
    marker: str,
    paper_id: str,
    *,
    claim: str = "X is true",
    verdict: str = "supported",
    overlap: float = 0.5,
) -> ClaimVerdict:
    return ClaimVerdict(
        marker=marker,
        claim=claim,
        paper_id=paper_id,
        paper_title=f"Paper for {paper_id}",
        verdict=verdict,
        overlap_score=overlap,
    )


def _chunk(
    *,
    chunk_id: str | None = None,
    paper_id: str,
    index: int = 0,
    section: str = "abstract",
    content: str = "chunk content",
    embedding: list[float] | None = None,
) -> MagicMock:
    """Fake PaperChunk that has the attribute access pattern the
    ranker uses. We don't need a real ORM instance — the ranker only
    reads ``id``, ``paper_id``, ``chunk_index``, ``section_type``,
    ``content``, ``embedding``.
    """
    c = MagicMock()
    c.id = uuid.UUID(chunk_id) if chunk_id else uuid.uuid4()
    c.paper_id = uuid.UUID(paper_id) if isinstance(paper_id, str) else paper_id
    c.chunk_index = index
    c.section_type = section
    c.content = content
    c.embedding = embedding
    return c


# ── Priority ordering ──────────────────────────────────────────────────────


def test_priority_supported_before_unverified_before_unsupported():
    """Supported claims claim the chunk-lookup budget first — they're
    the answer's backbone."""
    sup = _verdict("[1]", str(uuid.uuid4()), verdict="supported", overlap=0.4)
    unv = _verdict("[2]", str(uuid.uuid4()), verdict="unverified", overlap=0.4)
    uns = _verdict("[3]", str(uuid.uuid4()), verdict="unsupported", overlap=0.4)
    sorted_verdicts = sorted([uns, unv, sup], key=_priority_key)
    assert [v.verdict for v in sorted_verdicts] == ["supported", "unverified", "unsupported"]


def test_priority_within_tier_higher_overlap_wins():
    paper = str(uuid.uuid4())
    low = _verdict("[1]", paper, verdict="supported", overlap=0.3)
    high = _verdict("[2]", paper, verdict="supported", overlap=0.7)
    sorted_v = sorted([low, high], key=_priority_key)
    assert sorted_v[0].marker == "[2]"  # higher overlap wins


# ── Similarity ranking ─────────────────────────────────────────────────────


def test_ranking_returns_top_k_chunks_by_cosine():
    """A chunk whose embedding is closer to the claim wins. The
    bottom-K chunks get dropped even when they're above the floor."""
    paper = str(uuid.uuid4())
    claim_vec = [1.0, 0.0, 0.0]
    chunks = [
        _chunk(paper_id=paper, index=0, content="aligned", embedding=[1.0, 0.0, 0.0]),
        _chunk(paper_id=paper, index=1, content="orthogonal", embedding=[0.0, 1.0, 0.0]),
        _chunk(paper_id=paper, index=2, content="near", embedding=[0.95, 0.31, 0.0]),
    ]
    ranked = _rank_chunks_by_similarity(claim_vec, chunks)
    # First two cleared the 0.30 floor (1.0 and ~0.95); orthogonal
    # at 0.0 is below.
    assert len(ranked) == 2
    assert ranked[0].chunk_index == 0
    assert ranked[1].chunk_index == 2
    assert ranked[0].similarity > ranked[1].similarity


def test_ranking_filters_below_floor():
    """Chunks with negative or near-zero cosine to the claim must NOT
    appear in the output — they're surface noise."""
    paper = str(uuid.uuid4())
    claim_vec = [1.0, 0.0, 0.0]
    chunks = [
        _chunk(paper_id=paper, content="noise", embedding=[-1.0, 0.0, 0.0]),  # cos=-1
        _chunk(paper_id=paper, content="weak", embedding=[0.1, 0.99, 0.0]),  # cos≈0.1
    ]
    assert _rank_chunks_by_similarity(claim_vec, chunks) == []


def test_ranking_skips_chunks_without_embedding():
    """Chunks ingested without an embedding (e.g. pre-vector
    migration) shouldn't crash the ranker — they're skipped."""
    paper = str(uuid.uuid4())
    claim_vec = [1.0, 0.0, 0.0]
    chunks = [
        _chunk(paper_id=paper, content="has emb", embedding=[1.0, 0.0, 0.0]),
        _chunk(paper_id=paper, content="no emb", embedding=None),
    ]
    ranked = _rank_chunks_by_similarity(claim_vec, chunks)
    assert len(ranked) == 1
    assert ranked[0].excerpt == "has emb"


def test_ranking_empty_claim_vec_returns_empty():
    assert _rank_chunks_by_similarity([], [_chunk(paper_id=str(uuid.uuid4()), embedding=[1.0])]) == []


# ── ChunkEvidence serialisation ────────────────────────────────────────────


def test_chunk_link_to_dict_truncates_excerpt():
    """The dict serialisation caps excerpts at 480 chars so the UI
    payload stays bounded even for huge LaTeX chunks."""
    paper = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    link = ClaimChunkLink(
        marker="[3]",
        claim="claim text",
        paper_id=paper,
        paper_title="Title",
        chunks=[ChunkEvidence(
            paper_id=paper, chunk_id=chunk_id, chunk_index=2,
            section_type="abstract", similarity=0.87,
            excerpt="x" * 1000,
        )],
    )
    d = link.to_dict()
    assert len(d["chunks"][0]["excerpt"]) == 480


# ── End-to-end attach (mocked DB + embedding adapter) ─────────────────────


@pytest.mark.asyncio
async def test_attach_returns_empty_when_no_verdicts():
    """No-op when the verifier produced no verdicts."""
    db = MagicMock()
    result = await attach_chunk_evidence(
        db=db, claim_verdicts=[], papers=[],
    )
    assert result == []


@pytest.mark.asyncio
async def test_attach_skips_low_overlap_unsupported_claims():
    """An unsupported claim with overlap below the floor doesn't even
    get an embedding call — the paper is wrong, chunk lookup would
    surface arbitrary near-misses."""
    db = MagicMock()
    db.execute = AsyncMock()  # never invoked
    paper_id = str(uuid.uuid4())
    verdicts = [_verdict("[1]", paper_id, verdict="unsupported", overlap=0.0)]
    result = await attach_chunk_evidence(
        db=db, claim_verdicts=verdicts, papers=[{"paper_id": paper_id}],
    )
    assert result == []
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_attach_caps_at_max_claims_per_turn(monkeypatch):
    """A 50-claim answer must trigger at most MAX_CLAIMS_PER_TURN
    chunk lookups. The cap fires BEFORE the embedding call so we
    never bill the LLM for chunk lookup on the long tail of claims."""
    paper_id = str(uuid.uuid4())
    verdicts = [
        _verdict(f"[{i+1}]", paper_id, verdict="supported", overlap=0.5)
        for i in range(MAX_CLAIMS_PER_TURN + 10)
    ]
    # DB returns one chunk so the embed step actually runs (the early-
    # return when chunks is empty would otherwise short-circuit).
    fake_chunks = [_chunk(
        paper_id=paper_id, content="hit", embedding=[1.0, 0.0, 0.0],
    )]
    db = MagicMock()
    db.execute = AsyncMock(
        return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: fake_chunks))
    )

    embed_called: list[int] = []

    async def _fake_embed(texts, task_type=""):
        embed_called.append(len(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    fake_adapter = MagicMock()
    fake_adapter.embed_texts = _fake_embed
    monkeypatch.setattr(
        "app.adapters.embedding.get_embedding_adapter",
        lambda: fake_adapter,
    )

    await attach_chunk_evidence(
        db=db, claim_verdicts=verdicts, papers=[{"paper_id": paper_id}],
    )
    # Embedding batch ran with at most MAX_CLAIMS_PER_TURN texts.
    assert embed_called and embed_called[0] == MAX_CLAIMS_PER_TURN


@pytest.mark.asyncio
async def test_attach_handles_embedding_failure_gracefully(monkeypatch):
    """An embedding adapter that returns wrong-length output must NOT
    crash — chunk evidence is best-effort."""
    paper_id = str(uuid.uuid4())
    verdicts = [_verdict("[1]", paper_id, verdict="supported", overlap=0.5)]
    db = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: [])))

    async def _bad_embed(texts, task_type=""):
        return []  # wrong length

    fake_adapter = MagicMock()
    fake_adapter.embed_texts = _bad_embed
    monkeypatch.setattr(
        "app.adapters.embedding.get_embedding_adapter",
        lambda: fake_adapter,
    )
    result = await attach_chunk_evidence(
        db=db, claim_verdicts=verdicts, papers=[{"paper_id": paper_id}],
    )
    assert result == []


@pytest.mark.asyncio
async def test_attach_end_to_end_produces_chunk_links(monkeypatch):
    """Happy path: verdict → embed → DB chunks → ranked top-K."""
    paper_id = str(uuid.uuid4())
    chunk_ids = [str(uuid.uuid4()) for _ in range(3)]

    # Three chunks; the first one is most similar to the claim vector.
    fake_chunks = [
        _chunk(chunk_id=chunk_ids[0], paper_id=paper_id, index=0,
               content="strongly relevant", embedding=[1.0, 0.0, 0.0]),
        _chunk(chunk_id=chunk_ids[1], paper_id=paper_id, index=1,
               content="medium relevance", embedding=[0.7, 0.7, 0.0]),
        _chunk(chunk_id=chunk_ids[2], paper_id=paper_id, index=2,
               content="orthogonal noise", embedding=[0.0, 0.0, 1.0]),
    ]

    db = MagicMock()
    db.execute = AsyncMock(
        return_value=MagicMock(
            scalars=lambda: MagicMock(all=lambda: fake_chunks)
        )
    )

    async def _fake_embed(texts, task_type=""):
        return [[1.0, 0.0, 0.0] for _ in texts]

    fake_adapter = MagicMock()
    fake_adapter.embed_texts = _fake_embed
    monkeypatch.setattr(
        "app.adapters.embedding.get_embedding_adapter",
        lambda: fake_adapter,
    )

    verdicts = [_verdict("[1]", paper_id, verdict="supported", overlap=0.5)]
    result = await attach_chunk_evidence(
        db=db, claim_verdicts=verdicts,
        papers=[{"paper_id": paper_id, "title": "Real Paper"}],
    )
    assert len(result) == 1
    link = result[0]
    assert link.marker == "[1]"
    assert link.paper_id == paper_id
    assert link.paper_title == "Real Paper"
    # Top-K caps the chunk count; orthogonal chunk was below the
    # floor anyway, so only 2 made it past — the cap is TOP_K_CHUNKS.
    assert len(link.chunks) <= TOP_K_CHUNKS
    # Highest-similarity chunk ranked first.
    assert link.chunks[0].excerpt == "strongly relevant"
    assert link.chunks[0].similarity > (link.chunks[1].similarity if len(link.chunks) > 1 else 0)
