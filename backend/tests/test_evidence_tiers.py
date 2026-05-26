"""Tests for the evidence-tier classifier.

The classifier uses the paper's own STRUCTURE — where in the document
the answer-grounding chunks lie — rather than hardcoded section-name
vocabularies. A long hardcoded cue list only ever covered the venues
we happened to think of; chunk position works across every venue
because the canonical paper structure (abstract → introduction →
method → results → discussion → conclusion) is universal.
"""

from __future__ import annotations

import pytest

from app.assistant.claim_ledger import (
    SOURCE_ABSTRACT,
    ClaimLedger,
    StrongClaim,
    evidence_tier_from_sections,
    evidence_tier_from_structure,
)
from app.assistant.react.middlewares.full_paper_gate import (
    FullPaperVerificationMiddleware,
)
from app.assistant.tools.base import ToolResult

from tests.test_full_paper_gate import _make_state, _provisional_claim


# ── Position-based primary signal ───────────────────────────────────────────


@pytest.mark.parametrize("positions,expected", [
    # Late chunks (≥ 0.55) → experiment-verified
    ([0.85], "experiment-verified"),
    ([0.55], "experiment-verified"),
    ([0.10, 0.60], "experiment-verified"),  # max-pos wins
    ([0.20, 0.80], "experiment-verified"),
    # Middle chunks (≥ 0.20 and < 0.55) → method-verified
    ([0.30], "method-verified"),
    ([0.20], "method-verified"),
    ([0.45], "method-verified"),
    ([0.10, 0.40], "method-verified"),
    # Early chunks (< 0.20) → abstract-only
    ([0.0], "abstract-only"),
    ([0.05], "abstract-only"),
    ([0.0, 0.15], "abstract-only"),
])
def test_position_based_classification(positions, expected):
    assert evidence_tier_from_structure(chunk_positions=positions) == expected


def test_empty_positions_defaults_to_abstract_only():
    """No structural signal → conservative default. We refuse to
    guess a stronger tier without evidence."""
    assert evidence_tier_from_structure(chunk_positions=None) == "abstract-only"
    assert evidence_tier_from_structure(chunk_positions=[]) == "abstract-only"


def test_abstract_only_demotion_overrides_position():
    """When every chunk the parser stamped as ``abstract``, even a
    late position cannot upgrade the tier — the answer literally
    came from the abstract."""
    assert evidence_tier_from_structure(
        chunk_positions=[0.99], section_types=["abstract"],
    ) == "abstract-only"
    assert evidence_tier_from_structure(
        chunk_positions=[0.10, 0.99], section_types=["abstract", "abstract"],
    ) == "abstract-only"


def test_non_abstract_section_does_not_demote():
    """A chunk in a section the parser didn't tag ``abstract``
    (regardless of what the tag says) lets position drive the tier."""
    assert evidence_tier_from_structure(
        chunk_positions=[0.7], section_types=["custom-results-section"],
    ) == "experiment-verified"
    assert evidence_tier_from_structure(
        chunk_positions=[0.4], section_types=["whatever"],
    ) == "method-verified"


# ── Namespace-agnostic: works across all paper types ───────────────────────


@pytest.mark.parametrize("description,positions", [
    ("CS/ML — paper with method @ idx 4, results @ idx 7 of 10", [0.4, 0.7]),
    ("Math — proof at idx 6 of 8", [0.75]),
    ("Physics — measurement at idx 7 of 12", [0.58]),
    ("Biology — primary endpoint at idx 9 of 15", [0.60]),
    ("Economics — robustness check at idx 8 of 14", [0.57]),
    ("Clinical — trial outcomes at idx 11 of 18", [0.61]),
    ("Chemistry — yield section at idx 4 of 7", [0.57]),
])
def test_classifier_namespace_agnostic_via_position(description, positions):
    """The structural signal is universal: any paper, any discipline,
    any section name — chunks in the back half of the paper classify
    as experiment-verified."""
    assert evidence_tier_from_structure(chunk_positions=positions) == "experiment-verified", (
        f"{description}: expected experiment-verified, got "
        f"{evidence_tier_from_structure(chunk_positions=positions)!r}"
    )


def test_single_chunk_paper_collapses_to_abstract_only():
    """A paper indexed with only an abstract row produces a single
    chunk at position 0.0; tier must collapse to abstract-only so
    answers from abstract-only papers honestly hedge."""
    assert evidence_tier_from_structure(chunk_positions=[0.0]) == "abstract-only"


# ── Legacy shim ─────────────────────────────────────────────────────────────


def test_legacy_sections_only_shim_returns_abstract_only():
    """The old ``evidence_tier_from_sections`` API is kept as a shim
    for backward compat — without positions it has nothing to do but
    return the conservative default."""
    assert evidence_tier_from_sections(["method"]) == "abstract-only"
    assert evidence_tier_from_sections(["results"]) == "abstract-only"
    assert evidence_tier_from_sections([]) == "abstract-only"
    assert evidence_tier_from_sections(None) == "abstract-only"


# ── After-tool tier upgrade in full_paper_gate (now position-based) ────────


@pytest.mark.asyncio
async def test_after_tool_upgrades_tier_via_chunk_position():
    """A verified claim whose paper_qa hit drew from a chunk near the
    back of the paper (position ≥ 0.55) must be tagged
    ``experiment-verified`` — no hardcoded section names involved."""
    claim = _provisional_claim()
    claim.verdict = "in_flight"
    ledger = ClaimLedger()
    ledger.add(claim)
    state = _make_state(claim_ledger=ledger)
    state._fpg_inflight = {claim.paper_id: claim.span}  # type: ignore[attr-defined]

    result = ToolResult(
        output={
            "paper_id": claim.paper_id, "paper_title": "X",
            "answer": "Yes, the paper confirms the 95% accuracy on benchmark Q.",
            "found": True, "chunks_used": 2,
            "sections_used": ["abstract", "unknown_section"],
            "chunk_positions": [0.05, 0.75],
            "total_chunks": 20,
        },
        summary="ok",
    )
    mw = FullPaperVerificationMiddleware()
    await mw.after_tool(state, "paper_qa", {"paper_id": claim.paper_id}, result)
    assert claim.verdict == "verified"
    assert claim.evidence_tier == "experiment-verified"


@pytest.mark.asyncio
async def test_after_tool_assigns_method_tier_via_position():
    claim = _provisional_claim()
    claim.verdict = "in_flight"
    ledger = ClaimLedger()
    ledger.add(claim)
    state = _make_state(claim_ledger=ledger)
    state._fpg_inflight = {claim.paper_id: claim.span}  # type: ignore[attr-defined]

    result = ToolResult(
        output={
            "paper_id": claim.paper_id, "paper_title": "X",
            "answer": "Yes, the paper describes its model.",
            "found": True, "chunks_used": 1,
            "sections_used": ["custom"],
            "chunk_positions": [0.35],
            "total_chunks": 10,
        },
        summary="ok",
    )
    mw = FullPaperVerificationMiddleware()
    await mw.after_tool(state, "paper_qa", {"paper_id": claim.paper_id}, result)
    assert claim.verdict == "verified"
    assert claim.evidence_tier == "method-verified"


@pytest.mark.asyncio
async def test_after_tool_abstract_demotion_via_section_tag():
    """When every chunk the parser used is tagged exactly
    ``abstract``, the tier collapses to abstract-only even if
    chunk_positions would otherwise upgrade — abstract-only papers
    (no body indexed) must surface honestly."""
    claim = _provisional_claim()
    claim.verdict = "in_flight"
    ledger = ClaimLedger()
    ledger.add(claim)
    state = _make_state(claim_ledger=ledger)
    state._fpg_inflight = {claim.paper_id: claim.span}  # type: ignore[attr-defined]

    result = ToolResult(
        output={
            "paper_id": claim.paper_id, "paper_title": "X",
            "answer": "Yes, the paper supports the claim.",
            "found": True, "chunks_used": 1,
            "sections_used": ["abstract"],
            "chunk_positions": [0.0],
            "total_chunks": 1,
        },
        summary="ok",
    )
    mw = FullPaperVerificationMiddleware()
    await mw.after_tool(state, "paper_qa", {"paper_id": claim.paper_id}, result)
    assert claim.verdict == "verified"
    assert claim.evidence_tier == "abstract-only"


@pytest.mark.asyncio
async def test_after_tool_keeps_unverified_tier_on_failed_paper_qa():
    claim = _provisional_claim()
    claim.verdict = "in_flight"
    ledger = ClaimLedger()
    ledger.add(claim)
    state = _make_state(claim_ledger=ledger)
    state._fpg_inflight = {claim.paper_id: claim.span}  # type: ignore[attr-defined]

    result = ToolResult(
        output={
            "paper_id": "", "paper_title": "", "found": False,
            "answer": "", "sections_used": [], "chunk_positions": [],
            "total_chunks": 0,
        },
        summary="not found",
    )
    mw = FullPaperVerificationMiddleware()
    await mw.after_tool(state, "paper_qa", {"paper_id": claim.paper_id}, result)
    assert claim.verdict == "unverifiable"
    assert claim.evidence_tier == "unverified"


# ── ClaimLedger.summarize includes tier counts ──────────────────────────────


def test_summarize_emits_evidence_tier_buckets():
    ledger = ClaimLedger()
    c1 = StrongClaim(
        span="A: 99% accuracy on dataset A.",
        paper_id="p1", paper_title="X", source_field=SOURCE_ABSTRACT,
        iteration_seen=1, verdict="verified", evidence_tier="experiment-verified",
    )
    c2 = StrongClaim(
        span="B: SOTA method described in the paper.",
        paper_id="p2", paper_title="Y", source_field=SOURCE_ABSTRACT,
        iteration_seen=1, verdict="verified", evidence_tier="method-verified",
    )
    c3 = StrongClaim(
        span="C: untouched provisional claim.",
        paper_id="p3", paper_title="Z", source_field=SOURCE_ABSTRACT,
        iteration_seen=1,
    )
    ledger.add(c1); ledger.add(c2); ledger.add(c3)

    s = ledger.summarize()
    tiers = s["by_evidence_tier"]
    assert tiers["experiment-verified"] == 1
    assert tiers["method-verified"] == 1
    assert tiers["abstract-only"] == 1
    assert tiers["unverified"] == 0
    assert any(item["evidence_tier"] == "experiment-verified" for item in s["verified"])
