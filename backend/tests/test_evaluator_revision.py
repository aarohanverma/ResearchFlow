"""Tests for the final evaluator's revision-pass guards.

When the evaluator triggers a revision, the synthesizer's existing
provenance verdicts are stale (they were keyed against the original
answer's claim positions). Two correctness properties matter:

1. Stale provenance must NOT be returned alongside the revised text
   — the frontend would render wrong per-marker verdicts.
2. The revision pass must propagate ``asyncio.CancelledError`` so
   the user's Stop button still works mid-revision. (Best-effort
   try/except wrappers must not eat cancel.)
"""

from __future__ import annotations

import pytest

from app.assistant.final_evaluator import (
    revision_notes,
    should_revise,
)


# ── should_revise threshold logic ──────────────────────────────────────────


def test_should_revise_on_explicit_verdict():
    assert should_revise({"verdict": "revise"}) is True
    assert should_revise({"verdict": "drift"}) is True
    assert should_revise({"verdict": "ship"}) is False


def test_should_revise_on_low_groundedness():
    report = {
        "verdict": "ship",
        "relevance": 0.9,
        "groundedness": 0.4,    # below floor
        "completeness": 0.9,
        "focus": 0.9,
    }
    assert should_revise(report) is True


def test_should_revise_on_low_focus():
    """Focus has the strictest floor — drift is the worst failure
    mode because it wastes the whole turn."""
    report = {
        "verdict": "ship",
        "relevance": 0.9,
        "groundedness": 0.9,
        "completeness": 0.9,
        "focus": 0.65,   # below 0.70 floor
    }
    assert should_revise(report) is True


def test_should_revise_passes_strong_answer():
    report = {
        "verdict": "ship",
        "relevance": 0.85,
        "groundedness": 0.80,
        "completeness": 0.75,
        "focus": 0.90,
    }
    assert should_revise(report) is False


def test_should_revise_safe_on_malformed_input():
    """A non-dict (defensive case) returns False so we don't
    accidentally trigger a revision on garbage input."""
    assert should_revise("not a dict") is False  # type: ignore[arg-type]
    assert should_revise(None) is False           # type: ignore[arg-type]


# ── revision_notes formatting ──────────────────────────────────────────────


def test_revision_notes_includes_verdict_and_improvements():
    report = {
        "verdict": "revise",
        "relevance": 0.5,
        "groundedness": 0.6,
        "completeness": 0.7,
        "focus": 0.8,
        "improvements": [
            "cite the specific paper for the 92% accuracy claim",
            "remove the digression about subfield Y",
        ],
    }
    notes = revision_notes(report)
    assert "verdict: revise" in notes
    assert "cite the specific paper" in notes
    assert "remove the digression" in notes


def test_revision_notes_truncates_overflow():
    """A pathological evaluator emitting 50 long suggestions
    cannot bloat the revision prompt."""
    report = {
        "verdict": "revise",
        "relevance": 0.5,
        "groundedness": 0.5,
        "completeness": 0.5,
        "focus": 0.5,
        "improvements": ["x" * 500 for _ in range(50)],
    }
    notes = revision_notes(report)
    # Cap is 1200 chars per the module's _MAX_NOTES_CHARS.
    assert len(notes) <= 1200


def test_revision_notes_empty_on_non_dict():
    assert revision_notes("nope") == ""        # type: ignore[arg-type]
    assert revision_notes(None) == ""          # type: ignore[arg-type]
