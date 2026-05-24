"""Repair-pass drift detector."""

from __future__ import annotations

from app.assistant.repair_drift import detect_repair_drift


def test_no_drift_when_only_prose_changed():
    """Pure rewording — same claims, same citations, different
    surface words. Drift detector must NOT flag this."""
    pre = "Transformers achieve SOTA on benchmarks [1]. Retrieval helps [2]."
    post = "Transformers reach the state-of-the-art on benchmarks [1]. Retrieval is useful [2]."
    rep = detect_repair_drift(pre=pre, post=post)
    assert rep.has_drift is False
    # The "changed" set may capture these as same-claim/different-prose
    # but neither has_drift nor the loud signals should fire.
    assert not rep.new_markers
    assert not rep.new_claims


def test_drift_when_repair_adds_new_citation_marker():
    """Repair introduces ``[3]`` that wasn't in the pre answer — load-
    bearing signal that the repair LLM hallucinated a new source."""
    pre = "Transformers achieve SOTA [1]."
    post = "Transformers achieve SOTA [1] and outperform RNNs decisively [3]."
    rep = detect_repair_drift(pre=pre, post=post)
    assert rep.has_drift is True
    assert "[3]" in rep.new_markers


def test_drift_when_repair_adds_new_citation_bearing_claim():
    pre = "Transformers achieve SOTA on GLUE [1]."
    post = (
        "Transformers achieve SOTA on GLUE [1]. They also dominate on "
        "long-context benchmarks across every reported setting [1]."
    )
    rep = detect_repair_drift(pre=pre, post=post)
    assert rep.has_drift is True
    assert rep.new_claims


def test_drift_changed_claim_when_prose_diverges_substantively():
    """Same citation, but the claim around it changed substantially —
    capture as ``changed_claims`` so the synth can surface it."""
    pre = "BERT achieves 88% accuracy on MNLI [1]."
    post = (
        "BERT achieves 99% perfect accuracy across every NLU benchmark, "
        "including completely unrelated reasoning suites [1]."
    )
    rep = detect_repair_drift(pre=pre, post=post)
    # The Jaccard heuristic may classify this as either changed or
    # entirely new; either way the drift report must fire.
    assert rep.has_drift is True


def test_no_drift_when_post_is_empty():
    rep = detect_repair_drift(pre="A [1].", post="")
    assert rep.has_drift is False


def test_render_contains_drift_summary():
    pre = "Original [1]."
    post = "Original [1]. Hallucinated extra fact [4]."
    rep = detect_repair_drift(pre=pre, post=post)
    lines = rep.render_for_agent_notes()
    assert any("[4]" in l for l in lines)
