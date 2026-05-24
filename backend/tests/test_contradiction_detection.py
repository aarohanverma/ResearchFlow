"""Active contradiction detection + adaptive counter-search policy."""

from __future__ import annotations

import pytest

from app.assistant.contradiction import (
    ContradictionLedger,
    detect_contradictions_in_results,
)
from app.assistant.tools.base import ToolResult


def _result(summary: str = "", **out) -> ToolResult:
    return ToolResult(output=out, summary=summary)


# ── Detector ───────────────────────────────────────────────────────────


def test_detects_explicit_lexical_contradiction():
    sigs = detect_contradictions_in_results(
        {"deep_search": _result(
            "Recent work contradicts earlier claims about scaling laws.",
            papers=[],
        )},
        iteration=1,
    )
    assert any(s.kind == "lexical" for s in sigs)
    assert any("contradict" in s.span.lower() for s in sigs)


def test_lexical_confidence_high_on_replication_marker():
    sigs = detect_contradictions_in_results(
        {"lit": _result(
            "These findings fail to replicate on out-of-domain benchmarks.",
        )},
        iteration=1,
    )
    assert sigs
    assert sigs[0].confidence >= 0.7


def test_lexical_confidence_medium_on_soft_marker():
    """A single soft marker ("inconsistent with") should sit BELOW the
    auto-force threshold so the loop surfaces it but doesn't burn an
    iteration on a single weak signal."""
    sigs = detect_contradictions_in_results(
        {"lit": _result(
            "This result is inconsistent with the original paper's setup.",
        )},
        iteration=1,
    )
    assert sigs
    assert sigs[0].confidence < ContradictionLedger.FORCE_CONFIDENCE_THRESHOLD


def test_detects_numeric_disagreement_between_papers():
    sigs = detect_contradictions_in_results(
        {"deep_search": _result(
            "",
            papers=[
                {"title": "Paper A", "abstract": "We achieve accuracy 92.0% on MMLU."},
                {"title": "Paper B", "abstract": "On the same MMLU split, accuracy 71.2%."},
            ],
        )},
        iteration=1,
    )
    assert any(s.kind == "numeric" for s in sigs)


def test_no_false_positive_on_aligned_numbers():
    """Same metric, similar values (within epsilon) should NOT trip
    the detector — otherwise every benchmark table would be flagged."""
    sigs = detect_contradictions_in_results(
        {"deep_search": _result(
            "",
            papers=[
                {"title": "A", "abstract": "accuracy 89.1% on GLUE"},
                {"title": "B", "abstract": "accuracy 91.4% on GLUE"},
            ],
        )},
        iteration=1,
    )
    assert not any(s.kind == "numeric" for s in sigs)


# ── Ledger policy ──────────────────────────────────────────────────────


def test_ledger_dedupes_same_span_from_multiple_sources():
    led = ContradictionLedger()
    from app.assistant.contradiction import ContradictionSignal
    s1 = ContradictionSignal(kind="lexical", span="X contradicts Y", sources=["a"], confidence=0.8)
    s2 = ContradictionSignal(kind="lexical", span="X contradicts Y", sources=["b"], confidence=0.8)
    assert led.add(s1) is True
    assert led.add(s2) is False
    assert len(led.signals) == 1
    assert set(led.signals[0].sources) == {"a", "b"}


def test_force_policy_blocks_low_confidence():
    """A soft contradiction (confidence < 0.65) must NOT trigger an
    auto-force, no matter how many iterations are left. The model
    should make the call itself based on the rendered prompt."""
    led = ContradictionLedger()
    from app.assistant.contradiction import ContradictionSignal
    led.add(ContradictionSignal(
        kind="lexical", span="however earlier work disagreed",
        sources=["deep_search"], confidence=0.45,
    ))
    assert led.next_to_force(iterations_remaining=4) is None


def test_force_policy_skips_when_budget_exhausted():
    led = ContradictionLedger()
    from app.assistant.contradiction import ContradictionSignal
    led.add(ContradictionSignal(
        kind="lexical", span="strong refutation here",
        sources=["deep_search"], confidence=0.9,
    ))
    # Only 1 iteration left — forcing now would land too late for the
    # model to read the result before finalizing.
    assert led.next_to_force(iterations_remaining=1) is None
    # 2+ iterations gives us room to dispatch + observe.
    assert led.next_to_force(iterations_remaining=2) is not None


def test_force_policy_only_fires_once_per_turn():
    led = ContradictionLedger()
    from app.assistant.contradiction import ContradictionSignal
    led.add(ContradictionSignal(
        kind="lexical", span="X contradicts Y", sources=["a"], confidence=0.9,
    ))
    assert led.next_to_force(iterations_remaining=4) is not None
    led.record_forced()
    # Even though the contradiction is still open, the policy caps
    # forced counter-searches at one per turn — further follow-up is
    # the model's job via the rendered prompt.
    assert led.next_to_force(iterations_remaining=4) is None


def test_addressed_marker_uses_token_overlap():
    led = ContradictionLedger()
    from app.assistant.contradiction import ContradictionSignal
    led.add(ContradictionSignal(
        kind="lexical", span="scaling laws contradict earlier results",
        sources=["deep_search"], confidence=0.9,
    ))
    # A subsequent action whose params mention the span's tokens marks
    # the contradiction addressed so the policy doesn't keep forcing
    # the same counter-search.
    marked = led.mark_addressed("scaling laws review")
    assert marked == 1
    assert led.signals[0].addressed is True


def test_render_orders_unaddressed_first():
    led = ContradictionLedger()
    from app.assistant.contradiction import ContradictionSignal
    led.add(ContradictionSignal(
        kind="lexical", span="addressed claim", sources=["a"],
        confidence=0.9, addressed=True,
    ))
    led.add(ContradictionSignal(
        kind="lexical", span="open claim", sources=["b"], confidence=0.9,
    ))
    rendered = led.render_for_prompt(limit=4)
    # Unaddressed signals must appear first so the model's attention
    # lands on what still needs counter-evidence.
    assert rendered.index("open claim") < rendered.index("addressed claim")
