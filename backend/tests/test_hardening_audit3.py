"""Regression tests for hardening audit #3 (Opus 4.7, May 2026).

Each test pins one specific bug fixed in this audit:

  * ``_rank_chunks_by_similarity`` now tolerates numpy-array embeddings
    coming back from pgvector instead of raising on the truthy probe.
  * ``_normalized_rank_distance`` was computed over positionally-
    misaligned score lists when papers had partial scores; the fix
    extracts paired (raw_score, rerank_score) tuples per paper so
    rank-distance refers to the same paper on both sides.
  * ``InvestigationPlan.apply_operations`` now bumps
    ``last_updated_iteration`` even when the payload is malformed, so
    stuck-in-progress tracking stays anchored to the current iteration.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from app.assistant.provenance_evidence import _rank_chunks_by_similarity
from app.assistant.react.investigation_plan import InvestigationPlan
from app.assistant.retrieval_observability import (
    RetrievalObservability,
    _paired_scores,
)
from app.assistant.tools.base import ToolResult


# ── chunk-evidence ranker tolerates numpy-array embeddings ─────────────────


class _NumpyLikeArray:
    """Stand-in for ``numpy.ndarray`` that raises on ``bool(self)``.

    The real failure pattern: pgvector via SQLAlchemy returns
    ``numpy.ndarray`` for the ``embedding`` column when numpy is
    installed in the environment. ``not array`` raises
    ``ValueError: The truth value of an array with more than one
    element is ambiguous`` — the same shape this stand-in exhibits.
    """

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __bool__(self) -> bool:
        raise ValueError(
            "truth value of an array with more than one element is ambiguous"
        )

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, idx):
        return self._values[idx]


def _chunk(*, embedding) -> MagicMock:
    c = MagicMock()
    c.id = uuid.uuid4()
    c.paper_id = uuid.uuid4()
    c.chunk_index = 0
    c.section_type = "abstract"
    c.content = "content"
    c.embedding = embedding
    return c


def test_ranker_tolerates_numpy_array_embeddings():
    """Regression: the ranker used to truthy-test ``chunk.embedding``,
    which raises on numpy arrays. It must now route through ``is None``
    + ``len()`` instead so pgvector-backed runs don't blow up."""
    claim_vec = [1.0, 0.0, 0.0]
    chunks = [
        _chunk(embedding=_NumpyLikeArray([1.0, 0.0, 0.0])),
        _chunk(embedding=_NumpyLikeArray([0.0, 1.0, 0.0])),
    ]
    # Must not raise — used to crash with ValueError on the truthy check.
    ranked = _rank_chunks_by_similarity(claim_vec, chunks)
    # First chunk is perfectly aligned; the orthogonal one is filtered
    # at the 0.30 floor.
    assert len(ranked) == 1
    assert ranked[0].excerpt == "content"


def test_ranker_skips_none_embedding_after_numpy_fix():
    """Sanity check: the ``is None`` short-circuit still kicks in for
    chunks that genuinely lack an embedding."""
    claim_vec = [1.0, 0.0, 0.0]
    chunks = [
        _chunk(embedding=None),
        _chunk(embedding=_NumpyLikeArray([1.0, 0.0, 0.0])),
    ]
    ranked = _rank_chunks_by_similarity(claim_vec, chunks)
    assert len(ranked) == 1


# ── retrieval observability rank-distance alignment ─────────────────────────


def _result(papers: list[dict]) -> ToolResult:
    return ToolResult(output={"papers": papers}, summary="")


def test_paired_scores_skips_papers_missing_either_score():
    """Only papers that have BOTH raw and rerank scores enter the
    pairing; anything else is dropped so rank-distance never compares
    positionally-misaligned scores."""
    papers = [
        {"paper_id": "p1", "raw_score": 0.9, "search_score": 0.2},
        {"paper_id": "p2", "raw_score": 0.7},                       # no rerank
        {"paper_id": "p3", "search_score": 0.8},                    # no raw
        {"paper_id": "p4", "raw_score": 0.5, "search_score": 0.5},
    ]
    paired = _paired_scores(papers)
    assert paired == [(0.9, 0.2), (0.5, 0.5)]


def test_rerank_disagreement_aligned_when_scores_are_sparse():
    """Regression: when some papers had a ``raw_score`` but no
    ``search_score`` (or vice versa), the pre-fix metric compared rank
    of paper A's raw_score against rank of paper B's rerank_score —
    nonsense. The fix uses paired extraction so rank-distance is only
    computed over papers that have BOTH scores.

    Concretely: with three fully-scored papers in reversed order and
    one paper missing the rerank score, the metric should still report
    a sharp disagreement (the three paired papers ARE reversed) and
    must not raise.
    """
    obs = RetrievalObservability()
    papers = [
        # Three paired papers, reversed order.
        {"paper_id": "p1", "raw_score": 0.90, "search_score": 0.10},
        {"paper_id": "p2", "raw_score": 0.60, "search_score": 0.40},
        {"paper_id": "p3", "raw_score": 0.30, "search_score": 0.90},
        # Spoiler paper — has raw but no rerank. Pre-fix this would
        # silently mis-pair against p3's position in scores.
        {"paper_id": "p4", "raw_score": 0.20},
    ]
    snap = obs.record("deep_search", {"limit": 4}, _result(papers))
    assert snap is not None
    # The three paired papers are reversed → rank-distance should be
    # at or near the maximum (≥0.6 leaves plenty of headroom regardless
    # of the exact normalisation).
    assert snap.rerank_disagreement is not None
    assert snap.rerank_disagreement >= 0.6


def test_rerank_disagreement_none_when_fewer_than_three_paired_papers():
    """With only two paired papers the metric is statistically
    uninformative — return ``None`` instead of a spurious 1.0."""
    obs = RetrievalObservability()
    papers = [
        {"paper_id": "p1", "raw_score": 0.9, "search_score": 0.1},
        {"paper_id": "p2", "raw_score": 0.3, "search_score": 0.9},
        # Two more papers, neither paired.
        {"paper_id": "p3", "raw_score": 0.5},
        {"paper_id": "p4", "search_score": 0.5},
    ]
    snap = obs.record("deep_search", {"limit": 4}, _result(papers))
    assert snap.rerank_disagreement is None


# ── InvestigationPlan keeps iteration cursor on malformed batch ────────────


def test_apply_operations_bumps_iteration_on_non_list_payload():
    """Regression: a malformed write_todos payload used to return early
    WITHOUT updating ``last_updated_iteration``. That left the stuck-
    in-progress tracker anchored to an old iteration so it would
    surface true-stuck items as fresh."""
    plan = InvestigationPlan()
    plan.apply_operations([{"kind": "add", "text": "task one"}], iteration=1)
    plan.apply_operations([{"kind": "update", "id": "t1", "status": "in_progress"}], iteration=2)
    assert plan.last_updated_iteration == 2

    # Now feed a malformed batch — the cursor must still move.
    notes = plan.apply_operations("not a list", iteration=5)  # type: ignore[arg-type]
    assert notes == ["write_todos payload was not a list — ignored"]
    assert plan.last_updated_iteration == 5

    # Stuck tracker uses last_updated_iteration as the anchor. The
    # in-progress todo from iteration 2 IS stuck at iteration 5
    # (gap of 3 ≥ slack 2). Without the bump the cursor would still
    # read iteration 2 and the same todo would not register as stuck.
    stuck = plan.stuck_in_progress(current_iteration=5, slack=2)
    assert any(t.id == "t1" for t in stuck)


def test_apply_operations_normal_path_still_works():
    """Sanity: the iteration-cursor fix didn't break the well-formed
    path."""
    plan = InvestigationPlan()
    notes = plan.apply_operations(
        [{"kind": "add", "text": "find counter-evidence"}], iteration=1,
    )
    assert any("added" in n for n in notes)
    assert plan.last_updated_iteration == 1
    assert len(plan.todos) == 1
