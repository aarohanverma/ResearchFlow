"""Tests for the strong-claim detector + ledger that powers the
full-paper verification middleware.

The detector is a syntactic heuristic — these tests pin the patterns we
expect to fire (numeric performance, SOTA, causal, comparative) and
the ones we expect NOT to (routine metadata, short snippets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.assistant.claim_ledger import (
    ClaimLedger,
    SOURCE_ABSTRACT,
    SOURCE_CHUNK,
    SOURCE_SNIPPET,
    StrongClaim,
    detect_strong_spans,
    extract_claims_from_result,
    resolve_paper_qa_verdict,
)


@dataclass
class _FakeResult:
    """Minimal stand-in for ToolResult — the ledger only reads .output."""
    output: dict[str, Any]


# ── Detector ────────────────────────────────────────────────────────────────


def test_numeric_perf_pattern_fires():
    text = (
        "Our model achieves 92.4% accuracy on ImageNet — a 3.1 point improvement "
        "over the prior state of the art baseline."
    )
    spans = detect_strong_spans(text)
    assert spans, "expected a numeric/SOTA span to fire"
    assert any("92.4" in s for s in spans)


def test_sota_pattern_fires():
    text = "We present the first method to outperform every prior baseline on COCO."
    spans = detect_strong_spans(text)
    assert spans
    assert any("outperform" in s.lower() or "first" in s.lower() for s in spans)


def test_causal_pattern_fires():
    text = "Removing the residual connection collapses training within 5 epochs."
    spans = detect_strong_spans(text)
    assert spans
    assert "collapses" in spans[0].lower()


def test_routine_metadata_does_not_fire():
    text = "This paper studies generative models trained on web-scale data."
    assert detect_strong_spans(text) == []


def test_short_text_is_skipped():
    assert detect_strong_spans("") == []
    assert detect_strong_spans("Short.") == []


def test_max_spans_caps_extraction():
    # Build text with many strong claims; verify the cap holds.
    text = " ".join([
        "Our model achieves 95% accuracy on dataset A.",
        "It is the first to beat baseline B by 4 BLEU.",
        "It surpasses all prior work on benchmark C.",
        "Removing component D collapses performance.",
        "It outperforms every prior baseline on benchmark E.",
    ])
    spans = detect_strong_spans(text, max_spans=2)
    assert len(spans) == 2


# ── extract_claims_from_result ──────────────────────────────────────────────


def test_extract_from_retrieval_tags_source_abstract():
    result = _FakeResult(output={"papers": [
        {
            "paper_id": "p1", "title": "Model X",
            "abstract": "We achieve 95.2% accuracy, improving over the prior SOTA by 4 points.",
        },
    ]})
    claims = extract_claims_from_result(action="deep_search", result=result, iteration=1)
    assert claims
    assert all(c.paper_id == "p1" for c in claims)
    assert all(c.source_field == SOURCE_ABSTRACT for c in claims)
    assert all(c.verdict == "provisional" for c in claims)


def test_extract_from_paper_qa_tags_verified_chunk():
    result = _FakeResult(output={
        "paper_id": "p1", "paper_title": "Model X",
        "answer": "The paper reports 92.4% accuracy and outperforms all prior baselines.",
    })
    claims = extract_claims_from_result(action="paper_qa", result=result, iteration=2)
    assert claims
    assert all(c.source_field == SOURCE_CHUNK for c in claims)
    assert all(c.verdict == "verified" for c in claims)
    assert all(c.verified_at_iteration == 2 for c in claims)


def test_extract_handles_empty_and_unknown_shapes():
    # No papers list
    assert extract_claims_from_result(
        action="deep_search", result=_FakeResult(output={}), iteration=1,
    ) == []
    # paper_qa without answer
    assert extract_claims_from_result(
        action="paper_qa", result=_FakeResult(output={"paper_id": "p1"}), iteration=1,
    ) == []


# ── Ledger semantics ────────────────────────────────────────────────────────


def test_ledger_dedup_by_paper_and_head():
    ledger = ClaimLedger()
    c1 = StrongClaim(
        span="Our model achieves 95% accuracy on benchmark A.",
        paper_id="p1", paper_title="X", source_field=SOURCE_ABSTRACT,
        iteration_seen=1,
    )
    c2 = StrongClaim(
        span="Our model achieves 95% accuracy on benchmark A.",  # duplicate head
        paper_id="p1", paper_title="X", source_field=SOURCE_ABSTRACT,
        iteration_seen=2,
    )
    assert ledger.add(c1) is True
    assert ledger.add(c2) is False  # duplicate
    assert len(ledger.by_key) == 1


def test_unverified_excludes_chunk_sourced_claims():
    ledger = ClaimLedger()
    abstract_claim = StrongClaim(
        span="We achieve 95% accuracy and outperform all baselines.",
        paper_id="p1", paper_title="X",
        source_field=SOURCE_ABSTRACT, iteration_seen=1,
    )
    chunk_claim = StrongClaim(
        span="The full body confirms 95% accuracy on benchmark A.",
        paper_id="p2", paper_title="Y",
        source_field=SOURCE_CHUNK, iteration_seen=1, verdict="verified",
    )
    ledger.add(abstract_claim)
    ledger.add(chunk_claim)
    unverified = ledger.unverified()
    assert len(unverified) == 1
    assert unverified[0].paper_id == "p1"


def test_summarize_counts_buckets():
    ledger = ClaimLedger()
    ledger.add(StrongClaim(
        span="A: We achieve 99% accuracy in the introduction.",
        paper_id="p1", paper_title="X", source_field=SOURCE_ABSTRACT,
        iteration_seen=1,
    ))
    ledger.add(StrongClaim(
        span="B: The full body confirms 95% accuracy on the test set.",
        paper_id="p2", paper_title="Y", source_field=SOURCE_CHUNK,
        iteration_seen=1, verdict="verified",
    ))
    ledger.add(StrongClaim(
        span="C: The headline beats baselines by 10 BLEU points.",
        paper_id="p3", paper_title="Z", source_field=SOURCE_SNIPPET,
        iteration_seen=1, verdict="contradicted",
    ))
    summary = ledger.summarize()
    assert summary["total"] == 3
    assert summary["verified_count"] == 1
    assert summary["contradicted_count"] == 1
    assert summary["provisional_count"] == 1
    assert len(summary["verified"]) == 1
    assert len(summary["contradicted"]) == 1
    assert len(summary["provisional"]) == 1


def test_render_for_prompt_caps_lines():
    ledger = ClaimLedger()
    for i in range(20):
        ledger.add(StrongClaim(
            span=f"Claim {i}: We achieve {90 + i}% accuracy on benchmark {i}.",
            paper_id=f"p{i}", paper_title=f"Paper {i}",
            source_field=SOURCE_ABSTRACT, iteration_seen=1,
        ))
    rendered = ledger.render_for_prompt(limit=5)
    # 5 listed + truncation footer
    assert rendered.count("\n") == 5
    assert "and 15 more" in rendered


# ── Refutation-aware paper_qa extraction ────────────────────────────────────


def test_paper_qa_refutation_does_not_seed_verified_claims():
    """A paper_qa answer that REFUTES the asked claim must not be
    mined for "verified" spans — otherwise sentences like "the paper
    does NOT achieve 95% accuracy" land "95% accuracy" in the ledger
    as a verified claim because the regex matches the numeric span and
    loses the negation context."""
    result = _FakeResult(output={
        "paper_id": "p1", "paper_title": "Model X", "found": True,
        "answer": (
            "The paper does not actually support the claim. The reported "
            "95% accuracy in the abstract is contradicted by the "
            "experimental section, which shows a 78% accuracy under the "
            "evaluation protocol."
        ),
    })
    claims = extract_claims_from_result(action="paper_qa", result=result, iteration=2)
    assert claims == [], "refutation answers must not seed verified claims"


def test_paper_qa_not_found_returns_empty():
    """When paper_qa reports ``found=False`` the synthetic placeholder
    answer must not be mined for strong claims."""
    result = _FakeResult(output={
        "paper_id": "", "paper_title": "", "found": False,
        "answer": "",
    })
    assert extract_claims_from_result(
        action="paper_qa", result=result, iteration=1,
    ) == []


def test_paper_qa_affirmative_with_inner_negation_skipped():
    """A largely-affirmative paper_qa answer may still contain a single
    sentence that denies a specific claim; the extractor must drop
    that span rather than mine it as verified."""
    result = _FakeResult(output={
        "paper_id": "p1", "paper_title": "Model X", "found": True,
        "answer": (
            "Yes, the paper confirms its training procedure. "
            "However, the model does not actually achieve the 99% accuracy "
            "mentioned in the abstract on benchmark Z."
        ),
    })
    claims = extract_claims_from_result(action="paper_qa", result=result, iteration=2)
    # Any extracted claim must not be the negated 99% sentence.
    for c in claims:
        assert "99%" not in c.span and "does not" not in c.span.lower()


# ── resolve_paper_qa_verdict ────────────────────────────────────────────────


def test_resolve_paper_qa_verdict_classifies():
    v, _ = resolve_paper_qa_verdict("Yes, the paper supports the claim that X holds.")
    assert v == "verified"

    v, _ = resolve_paper_qa_verdict("The paper does not support this — the body shows the opposite.")
    assert v == "contradicted"

    v, _ = resolve_paper_qa_verdict("It is possible to interpret the text either way.")
    assert v == "unverifiable"

    v, _ = resolve_paper_qa_verdict("")
    assert v == "unverifiable"


def test_resolve_paper_qa_verdict_refutation_wins_over_affirmation():
    """When both refutation and affirmation cues appear, the refutation
    wins — a paper that "states X but does NOT actually achieve X" is
    still a refutation of the original claim under verification."""
    text = (
        "The paper states X. However, the experimental section does not "
        "support the claim — the metrics tell a different story."
    )
    v, _ = resolve_paper_qa_verdict(text)
    assert v == "contradicted"


# ── ClaimLedger.find_pending ────────────────────────────────────────────────


def test_find_pending_returns_inflight_target_by_paper_and_head():
    ledger = ClaimLedger()
    target = StrongClaim(
        span="Our model achieves 95% accuracy on benchmark Q.",
        paper_id="p1", paper_title="X",
        source_field=SOURCE_ABSTRACT, iteration_seen=1,
    )
    assert ledger.add(target) is True
    # Simulate gate_finalize stamping it as in-flight.
    target.verdict = "in_flight"

    found = ledger.find_pending("p1", target.span)
    assert found is target

    # Different paper id → no match.
    assert ledger.find_pending("p2", target.span) is None

    # Same paper id but different span → no match.
    assert ledger.find_pending("p1", "Some other claim entirely.") is None


def test_find_pending_skips_non_inflight_verdicts():
    """``find_pending`` only resolves claims the gate has actively
    dispatched a paper_qa for — provisional claims that the model
    happened to also paper_qa on their own must not get auto-resolved
    via the in-flight resolver path."""
    ledger = ClaimLedger()
    claim = StrongClaim(
        span="Our model achieves 95% accuracy on benchmark Q.",
        paper_id="p1", paper_title="X",
        source_field=SOURCE_ABSTRACT, iteration_seen=1,
    )
    ledger.add(claim)
    # Claim stays in default "provisional" verdict.
    assert ledger.find_pending("p1", claim.span) is None


def test_summarize_treats_inflight_as_provisional():
    """An ``in_flight`` claim whose paper_qa never resolved (cancel,
    crash) should surface to the synthesizer alongside provisional
    claims so the answer still hedges on it."""
    ledger = ClaimLedger()
    ledger.add(StrongClaim(
        span="A: 99% accuracy on dataset A.",
        paper_id="p1", paper_title="X",
        source_field=SOURCE_ABSTRACT, iteration_seen=1,
    ))
    stranded = StrongClaim(
        span="B: 99% accuracy on dataset B.",
        paper_id="p2", paper_title="Y",
        source_field=SOURCE_ABSTRACT, iteration_seen=1,
    )
    ledger.add(stranded)
    stranded.verdict = "in_flight"

    summary = ledger.summarize()
    # Both rows are still "not verified" — in_flight rolls up into provisional.
    assert summary["total"] == 2
    assert summary["provisional_count"] == 2
    assert summary["verified_count"] == 0
