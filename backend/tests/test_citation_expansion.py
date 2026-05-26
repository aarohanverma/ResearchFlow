"""Tests for the citation-marker post-processing in the synthesizer.

Two cases the spec explicitly calls out:

1. **Comma-joined markers must expand**: ``_strip_unresolvable_citations``
   used to emit ``[1,2,3]`` from a range like ``[1-3]``. The frontend's
   marker regex only matches single-number brackets, so the comma form
   rendered as plain text — broken citations the user couldn't click.

2. **Adjacent-marker ranges must expand**: the LLM regularly emits
   ``[1]-[7]`` (or ``[1] – [7]``, ``[1]_[7]``, etc.) to compress a
   contiguous citation. The user's spec is unambiguous: each citation
   must appear individually so each chip is clickable and auditable
   in the citation table.
"""

from __future__ import annotations

import pytest

from app.assistant.synthesizer import (
    _expand_adjacent_marker_ranges,
    _strip_unresolvable_citations,
)


def _papers(n: int) -> list[dict]:
    return [{"paper_id": f"p{i}"} for i in range(n)]


def _arxiv(n: int) -> list[dict]:
    return [{"external_id": f"2401.{i:05d}"} for i in range(n)]


# ── Adjacent-marker range expansion ─────────────────────────────────────────


def test_corpus_adjacent_range_expands_to_individual_markers():
    out = _expand_adjacent_marker_ranges("see [1]-[5] for evidence", 10, 0, set())
    assert out == "see [1] [2] [3] [4] [5] for evidence"


@pytest.mark.parametrize("sep", ["-", "–", "—", "_", " - ", " – ", " _ "])
def test_corpus_range_handles_various_separators(sep):
    text = f"refs [2]{sep}[6] discuss this"
    out = _expand_adjacent_marker_ranges(text, 10, 0, set())
    assert "[2] [3] [4] [5] [6]" in out
    # And the dash separator itself is gone.
    assert sep.strip() not in out.replace("[", "").replace("]", "")


def test_arxiv_adjacent_range_expands_too():
    out = _expand_adjacent_marker_ranges(
        "external sources [A1]-[A4]",
        0, 10, {1, 2, 3, 4},
    )
    assert out == "external sources [A1] [A2] [A3] [A4]"


def test_arxiv_range_drops_unresolvable_indices():
    """When some indices in the expanded range can't be linked (no
    external_id and no source_url), they must NOT appear in the
    expansion — the chip would be inert."""
    out = _expand_adjacent_marker_ranges(
        "see [A1]-[A4]",
        0, 10, {1, 3},  # 2 and 4 are unresolvable
    )
    assert out == "see [A1] [A3]"


def test_reversed_range_left_intact():
    """``[5]-[2]`` is not a valid range — leave the original text
    alone rather than silently swap or drop."""
    text = "weird [5]-[2] backwards"
    out = _expand_adjacent_marker_ranges(text, 10, 0, set())
    assert out == text


def test_oversized_range_left_intact():
    """A pathological [1]-[10000] expansion would be a wall of chips;
    the safety cap leaves it untouched."""
    text = "huge [1]-[10000] range"
    out = _expand_adjacent_marker_ranges(text, 0, 0, set())
    assert out == text


def test_chained_ranges_resolve_iteratively():
    """``[1]-[3] [4]-[6]`` should fully expand in one call thanks to
    the iterative passes."""
    out = _expand_adjacent_marker_ranges(
        "see [1]-[3] and [4]-[6]", 10, 0, set(),
    )
    assert "[1] [2] [3]" in out
    assert "[4] [5] [6]" in out


def test_no_range_no_change():
    text = "regular [1] and [2] citations, nothing to expand"
    out = _expand_adjacent_marker_ranges(text, 10, 0, set())
    assert out == text


# ── Integration through the full strip+expand pipeline ─────────────────────


def test_strip_emits_space_separated_markers_for_internal_range():
    """``[1-3]`` (range INSIDE brackets) must emit
    ``[1] [2] [3]`` — the frontend regex matches single-number
    markers only; the legacy ``[1,2,3]`` form rendered as plain text.
    """
    out = _strip_unresolvable_citations(
        "see [1-3] for evidence",
        papers=_papers(5),
        arxiv_results=[],
    )
    assert "[1] [2] [3]" in out


def test_strip_expands_adjacent_range_through_pipeline():
    """End-to-end: ``[1]-[5]`` should land as five individual
    markers after the strip pass runs the expander."""
    out = _strip_unresolvable_citations(
        "evidence [1]-[5] supports this",
        papers=_papers(8),
        arxiv_results=[],
    )
    assert "[1] [2] [3] [4] [5]" in out
    # And the dash is gone — no "[1]-[5]" residue.
    assert "[1]-[5]" not in out


def test_strip_clamps_range_at_paper_ceiling():
    """``[2]-[20]`` with only 6 papers expands to [2..6] — anything
    above the ceiling silently clamps to the available count rather
    than rendering broken citations."""
    out = _strip_unresolvable_citations(
        "look at [2]-[20]",
        papers=_papers(6),
        arxiv_results=[],
    )
    assert "[2] [3] [4] [5] [6]" in out
    assert "[7]" not in out


def test_strip_keeps_existing_individual_markers_intact():
    """Plain markers that aren't part of a range survive untouched
    through both passes."""
    out = _strip_unresolvable_citations(
        "see [1] then [3]",
        papers=_papers(5),
        arxiv_results=[],
    )
    assert "[1]" in out and "[3]" in out
