"""Tests for ``_detect_output_quality_issue`` — the synthesizer's
post-generation safeguard that prevents empty / truncated / corrupted
answers from ever reaching the user.

The user's spec is explicit: "RA must never output empty or corrupted
content. Output must always be complete and never truncated." This
safeguard is the load-bearing enforcement of that promise.
"""

from __future__ import annotations

import pytest

from app.assistant.synthesizer import _detect_output_quality_issue


# ── Empty / whitespace cases ────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["", "   ", "\n\n\t  ", None])
def test_empty_answer_caught(value):
    issue = _detect_output_quality_issue(value)
    assert issue is not None
    assert "empty" in issue.lower() or "not a string" in issue.lower()


# ── Length floor ────────────────────────────────────────────────────────────


def test_very_short_answer_caught_as_truncation():
    """An answer under 24 chars on a research turn is almost
    certainly a truncated generation."""
    issue = _detect_output_quality_issue("Yes.")
    assert issue is not None
    assert "short" in issue


def test_short_unterminated_answer_caught():
    """A 24-80 char answer that doesn't end with sentence punctuation
    is treated as a fragment."""
    answer = "The paper compares three architectures including"
    issue = _detect_output_quality_issue(answer)
    assert issue is not None
    assert "short" in issue or "truncated" in issue


def test_short_complete_answer_passes():
    """A complete short sentence WITH terminator is legitimate and
    must NOT trip the check — fixes the false-positive on terse
    replies."""
    answer = "Yes. The transformer introduced multi-head self-attention."
    assert _detect_output_quality_issue(answer) is None


def test_just_over_threshold_passes():
    answer = "x" * 90 + ". And this completes a sentence."
    assert _detect_output_quality_issue(answer) is None


# ── Template placeholder leakage ────────────────────────────────────────────


@pytest.mark.parametrize("answer", [
    "The paper by {{author_name}} shows that XYZ is the most promising approach for the task at hand.",
    "Looking at ${best_supporting_paper_id} we can see the results clearly demonstrate the claim.",
])
def test_template_placeholder_leak_caught(answer):
    issue = _detect_output_quality_issue(answer)
    assert issue is not None
    assert "placeholder" in issue.lower()


def test_todo_marker_caught():
    """A leaked TODO/FIXME marker is unambiguous corruption — the
    model is supposed to PRODUCE content there, not annotate
    intentions."""
    answer = (
        "This is a thorough analysis of the paper's methodology, "
        "covering the architecture and the training objective. "
        "<TODO: add experimental comparison against the baseline reported in 2024.>"
    )
    issue = _detect_output_quality_issue(answer)
    assert issue is not None
    assert "todo" in issue.lower() or "placeholder" in issue.lower()


# ── Provider error markers ─────────────────────────────────────────────────


@pytest.mark.parametrize("answer", [
    "[ERROR] Provider returned 502 on the synthesis call. Please retry in a few moments now okay?",
    "Error: rate limit exceeded for this token bucket. Please try again later when the limit resets.",
    "[BLOCKED] Content policy violation flagged on this query — abort. Please revise the prompt now.",
])
def test_provider_error_markers_caught(answer):
    issue = _detect_output_quality_issue(answer)
    assert issue is not None
    assert "error" in issue.lower()


# ── Unbalanced code fence / LaTeX ───────────────────────────────────────────


def test_unbalanced_code_fence_caught():
    answer = (
        "Here is an example implementation showing the approach:\n\n"
        "```python\ndef compute(x):\n    return x + 1"
    )
    issue = _detect_output_quality_issue(answer)
    assert issue is not None
    assert "code" in issue.lower()


def test_balanced_code_fence_passes():
    answer = (
        "Here is an example implementation showing the approach:\n\n"
        "```python\ndef compute(x):\n    return x + 1\n```"
    )
    assert _detect_output_quality_issue(answer) is None


def test_unbalanced_latex_caught():
    answer = "The result depends on the integral $\\int_0^\\infty f(x) dx that needs to evaluate cleanly."
    issue = _detect_output_quality_issue(answer)
    assert issue is not None
    assert "latex" in issue.lower()


# ── Mid-sentence truncation tells ──────────────────────────────────────────


@pytest.mark.parametrize("answer", [
    "The paper compares three methods: BERT, RoBERTa, and ALBERT. The strongest empirical result holds because of",
    "We surveyed the literature on retrieval-augmented generation. The dominant pattern across all the work is to",
    "Three papers converge on the same conclusion. The shared mechanism across them is that the embedding step and",
])
def test_trailing_connective_caught(answer):
    """An answer ending on ``because`` / ``to`` / ``and`` without
    punctuation is a textbook generation cutoff."""
    issue = _detect_output_quality_issue(answer)
    assert issue is not None
    assert "mid-sentence" in issue.lower()


def test_trailing_connective_with_period_passes():
    """Same word inside a complete sentence with proper end punctuation
    must NOT trip the check."""
    answer = (
        "Three papers converge on the same conclusion. "
        "The shared mechanism across them is the embedding step that runs before retrieval, and."
    )
    # Wait — ``ends with .`` so trailing-connective check is bypassed by
    # design. Confirm.
    assert _detect_output_quality_issue(answer) is None


# ── Happy path passes through ──────────────────────────────────────────────


def test_realistic_grounded_answer_passes():
    """A normal grounded research answer must NOT trip any check."""
    answer = (
        "Three recent papers converge on retrieval-augmented generation as the "
        "default architecture for grounded research assistants. [1] introduces "
        "the canonical retrieve-then-read pattern; [2] extends it with iterative "
        "refinement; [3] shows that explicit citation grounding improves "
        "factual accuracy by 12.4% on the MMLU benchmark. The shared insight "
        "is that grounding via retrieval beats parametric recall for "
        "knowledge-intensive tasks."
    )
    assert _detect_output_quality_issue(answer) is None
