"""Retrieval observability metrics."""

from __future__ import annotations

from app.assistant.retrieval_observability import RetrievalObservability
from app.assistant.tools.base import ToolResult


def _result(papers: list[dict], summary: str = "") -> ToolResult:
    return ToolResult(output={"papers": papers}, summary=summary)


def test_records_only_retrieval_class_tools():
    obs = RetrievalObservability()
    snap = obs.record("compare_papers", {}, _result([]))   # not retrieval
    assert snap is None
    snap = obs.record("deep_search", {"limit": 8}, _result([]))
    assert snap is not None
    assert len(obs.snapshots) == 1


def test_coverage_ratio_flags_thin_retrieval():
    """1/8 of asked-for papers must round to a low coverage_ratio so
    the loop can react to it and broaden the next query."""
    obs = RetrievalObservability()
    snap = obs.record(
        "deep_search",
        {"limit": 8},
        _result([{"paper_id": "p1", "search_score": 0.4}]),
    )
    assert snap is not None
    assert snap.coverage_ratio < 0.2
    assert snap.note == "thin"


def test_coverage_saturated_when_returned_equals_asked():
    obs = RetrievalObservability()
    papers = [{"paper_id": f"p{i}", "search_score": 0.7} for i in range(5)]
    snap = obs.record("deep_search", {"limit": 5}, _result(papers))
    assert snap.coverage_ratio == 1.0
    assert snap.note == "saturated"


def test_dispersion_high_when_top_dominates_tail():
    """Score dispersion (CV) should be HIGH when the top-1 paper scores
    way above the long tail (mixed-quality retrieval)."""
    obs = RetrievalObservability()
    papers = [
        {"paper_id": "p1", "search_score": 0.95},
        {"paper_id": "p2", "search_score": 0.30},
        {"paper_id": "p3", "search_score": 0.25},
        {"paper_id": "p4", "search_score": 0.20},
    ]
    snap = obs.record("deep_search", {"limit": 4}, _result(papers))
    assert snap.score_dispersion is not None
    assert snap.score_dispersion > 0.5


def test_dispersion_low_when_scores_are_uniform():
    obs = RetrievalObservability()
    papers = [
        {"paper_id": f"p{i}", "search_score": 0.80} for i in range(4)
    ]
    snap = obs.record("deep_search", {"limit": 4}, _result(papers))
    assert snap.score_dispersion is not None
    assert snap.score_dispersion < 0.05


def test_rerank_disagreement_high_when_reranker_reverses_order():
    """When the raw retrieval order and the post-rerank order disagree
    sharply, the disagreement metric should land above the high-water
    threshold so the prompt can flag a rerank-rescued retrieval."""
    obs = RetrievalObservability()
    # Raw scores rank p1>p2>p3>p4; rerank flips to p4>p3>p2>p1.
    papers = [
        {"paper_id": "p1", "search_score": 0.10, "raw_score": 0.90},
        {"paper_id": "p2", "search_score": 0.30, "raw_score": 0.70},
        {"paper_id": "p3", "search_score": 0.70, "raw_score": 0.30},
        {"paper_id": "p4", "search_score": 0.90, "raw_score": 0.10},
    ]
    snap = obs.record("deep_search", {"limit": 4}, _result(papers))
    assert snap.rerank_disagreement is not None
    assert snap.rerank_disagreement > 0.7


def test_rerank_disagreement_none_when_no_raw_baseline():
    """No ``raw_score`` baseline → we can't compute disagreement, so
    we return None rather than fabricating a number."""
    obs = RetrievalObservability()
    papers = [{"paper_id": "p1", "search_score": 0.9}]
    snap = obs.record("deep_search", {"limit": 1}, _result(papers))
    assert snap.rerank_disagreement is None


def test_thin_evidence_signal_aggregates_across_calls():
    obs = RetrievalObservability()
    obs.record("deep_search", {"limit": 8}, _result([
        {"paper_id": "p1", "search_score": 0.4},
    ]))
    obs.record("arxiv_search", {"limit": 6}, _result([]))
    # Two thin calls in one turn → aggregate signal fires.
    assert obs.thin_evidence_signal() is True


def test_agent_notes_serialisation_carries_weakest():
    obs = RetrievalObservability()
    obs.record("deep_search", {"limit": 8}, _result([
        {"paper_id": "p1", "search_score": 0.5},
    ]))
    notes = obs.to_agent_notes()
    assert notes["thin_calls"] >= 1
    assert "deep_search" in notes["weakest"]
