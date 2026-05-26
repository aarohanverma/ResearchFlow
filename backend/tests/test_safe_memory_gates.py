"""Tests for the safe-memory programmatic gates.

The librarian prompt already tells the cheap model to be conservative,
but it occasionally drifts into saving speculative / unverified /
transient content. ``_is_unsafe_memory_value`` is the deterministic
safety net that rejects writes whose VALUE reads as a hedge, a tool-
run trace, or an unresolved template variable — the four categories
the user explicitly told us never to persist.
"""

from __future__ import annotations

import pytest

from app.assistant.auto_memory import _is_unsafe_memory_value


# ── Hedged / speculative values get refused ─────────────────────────────────


@pytest.mark.parametrize("value", [
    "This might be the case but we're not sure.",
    "Possibly the model scales better — speculative.",
    "Could be related to the gradient signal in the residual stream.",
    "I think this is a promising direction.",
    "Unverified: the paper does not show experimental results.",
    "Tentative finding: needs replication.",
    "Perhaps the bottleneck is attention not MLPs.",
])
def test_hedged_value_refused(value):
    unsafe, reason = _is_unsafe_memory_value(value, "finding")
    assert unsafe, f"expected refusal for hedged value: {value!r}"
    assert "hedged" in reason


# ── Transient / run-state values get refused ────────────────────────────────


@pytest.mark.parametrize("value", [
    "Search returned 3 papers about robot manipulation.",
    "Tool failed with timeout — retry later.",
    "The loop ran 4 iterations and produced this answer.",
    "Paper_qa returned ambiguous results for this claim.",
    "Synthesis failed: insufficient context.",
])
def test_transient_value_refused(value):
    unsafe, reason = _is_unsafe_memory_value(value, "finding")
    assert unsafe, f"expected refusal for transient value: {value!r}"
    assert "transient" in reason


# ── Template-placeholder leaks get refused ──────────────────────────────────


def test_template_placeholder_refused():
    unsafe, reason = _is_unsafe_memory_value(
        "Paper {{best_supporting_paper_id}} confirms the claim.",
        "paper_note",
    )
    assert unsafe
    assert "template placeholder" in reason


def test_js_template_literal_refused():
    unsafe, reason = _is_unsafe_memory_value(
        "Reference: ${paperId}",
        "paper_note",
    )
    assert unsafe


# ── Empty / whitespace values get refused ───────────────────────────────────


def test_empty_value_refused():
    unsafe, _ = _is_unsafe_memory_value("", "finding")
    assert unsafe
    unsafe, _ = _is_unsafe_memory_value("   ", "finding")
    assert unsafe


# ── Real durable facts pass through ─────────────────────────────────────────


@pytest.mark.parametrize("value", [
    # Durable user preference
    "User prefers technical responses with paper citations and pseudocode.",
    # Concrete research finding (no hedging)
    "BERT-large achieves 92.4% accuracy on SQuAD 2.0 in the original paper.",
    # Definition / concept
    "Embodied AI refers to agents that learn through physical interaction with environments.",
    # User identity / role
    "Aarohan is a full-stack engineer building ResearchFlow.",
])
def test_durable_value_passes(value):
    unsafe, reason = _is_unsafe_memory_value(value, "finding")
    assert not unsafe, f"false positive on durable value: {value!r} (reason={reason!r})"


# ── Hypothesis-typed entries get an extra bar ───────────────────────────────


def test_open_question_rejected_as_hypothesis():
    """A value framed as an open question is by definition NOT a
    durable hypothesis — auto-stored hypotheses should describe a
    proposal, not a question to investigate later."""
    unsafe, reason = _is_unsafe_memory_value(
        "What if attention heads encode position implicitly?",
        "hypothesis",
    )
    assert unsafe
    assert "open question" in reason


def test_open_question_allowed_as_context():
    """The same value typed as ``context`` (not ``hypothesis``) is
    fine — context entries are catch-alls and the user may genuinely
    want to remember an open question they're exploring."""
    unsafe, _ = _is_unsafe_memory_value(
        "What if attention heads encode position implicitly?",
        "context",
    )
    assert not unsafe


def test_none_value_handled_safely():
    """``None`` input must not crash the checker — defensive against
    a librarian decision that somehow lands with a null value field."""
    unsafe, reason = _is_unsafe_memory_value(None, "finding")  # type: ignore[arg-type]
    assert unsafe
    assert reason == "empty value"


def test_non_string_value_coerced_and_evaluated():
    """An int or other non-string value gets stringified and
    evaluated — keeps the checker robust against malformed librarian
    output rather than crashing the auto-memory pipeline."""
    unsafe, _ = _is_unsafe_memory_value(12345, "finding")  # type: ignore[arg-type]
    # "12345" is < 6 chars, doesn't match patterns, but is short — the
    # *length-6 guard at the call site* would reject; here we just
    # confirm we don't raise.
    assert isinstance(unsafe, bool)


# ── Regression: hedge filter must use word boundaries ───────────────────────
#
# The earlier implementation matched hedges as plain substrings, which
# rejected legitimate facts whose text happened to overlap a hedge word.
# The most common false positives are month names (``"may 2026"``
# tripping ``"may "``) and surnames (``"Mayer"``, ``"Liklewood"``).
# These are exactly the kind of long-lived facts the librarian SHOULD
# be allowed to save — losing them silently corrupts the user's
# memory over time.


@pytest.mark.parametrize("value", [
    # Month names — extremely common in episode / finding values.
    "We met in May 2026 at the conference.",
    "Aarohan shipped the news pipeline in may 2026.",
    # Surname overlaps with a hedge word — collide on substring.
    "Dr. Mayer authored the seminal 2019 paper on quantization.",
    "Author Liklewood's 2023 paper is the canonical reference.",
    # Domain phrasing that contains a hedge word as a noun, not a hedge.
    # ``likelihood`` is a stats term (maximum likelihood, likelihood
    # ratio, etc.) — must not be conflated with the modal "likely".
    "Aarohan prefers Likelihood-based methods over Bayesian sampling.",
    # ``unlikely-event sampling`` is a real ML technique name.
    "BERT-base recall improves to 92.4% under unlikely-event sampling.",
])
def test_durable_value_with_hedge_substring_not_rejected(value):
    """Hedge words embedded in dates, surnames, or technical terms must
    NOT trigger a hedge rejection. Word-boundary matching is the
    surgical fix — substring matching false-rejects too much."""
    unsafe, reason = _is_unsafe_memory_value(value, "finding")
    assert not unsafe, (
        f"hedge filter false-positive on durable value: {value!r} "
        f"(reason={reason!r})"
    )


def test_real_hedge_at_start_still_rejected():
    """The fix must not loosen the hedge guard for real hedges."""
    # Each of these leads with a real hedge — word-boundary regex
    # should still catch them. (We dropped ``likely``/``unlikely``
    # because they're stats terms in our domain — covered by
    # ``maybe``/``perhaps``/``might`` instead.)
    for value in (
        "Maybe BERT scales better — speculative.",
        "Perhaps the attention heads encode position implicitly.",
        "Might that this approach generalises to vision tasks — uncertain.",
        "Tbd: whether the gradient flow improves downstream.",
    ):
        unsafe, reason = _is_unsafe_memory_value(value, "finding")
        assert unsafe, f"real hedge slipped through: {value!r}"
        assert "hedged" in reason
