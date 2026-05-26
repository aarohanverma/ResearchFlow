"""Tests for the full-paper verification middleware.

These cover the verdict-resolution loop introduced when the gate
started actively interpreting the ``paper_qa`` answer it forced —
without those changes a forced paper_qa would leave the original
provisional claim un-resolved on the ledger, so the synthesizer would
always mass-label otherwise-verified strong claims as ``unverifiable``.

The tests exercise the middleware against a minimal LoopState fixture;
the chain composition tests in ``test_react_middleware_chain.py`` cover
how this middleware interacts with the rest of the chain.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.assistant.claim_ledger import (
    SOURCE_ABSTRACT,
    ClaimLedger,
    StrongClaim,
)
from app.assistant.react.middleware import (
    FinalizeAllow,
    FinalizeForceAction,
)
from app.assistant.react.middlewares.full_paper_gate import (
    FullPaperVerificationMiddleware,
)
from app.assistant.react.state import LoopState
from app.assistant.react_loop import PaperLedger, ReactConfig
from app.assistant.scratchpad import Scratchpad
from app.assistant.tools.base import ToolResult


def _make_state(*, max_iterations: int = 5, claim_ledger: ClaimLedger | None = None) -> LoopState:
    config = ReactConfig(max_iterations=max_iterations, deadline_seconds=10.0)
    state = LoopState(
        query="test query",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        active_context=None,
        ctx=None,
        ctx_factory=None,
        should_cancel=None,
        publish=None,
        config=config,
        deadline=time.monotonic() + 10.0,
        pad=Scratchpad(),
    )
    state.ledger = PaperLedger()
    if claim_ledger is not None:
        state.claim_ledger = claim_ledger
    return state


def _provisional_claim(
    paper_id: str = "p1", span: str = "Our model achieves 95% accuracy on benchmark Q.",
) -> StrongClaim:
    return StrongClaim(
        span=span, paper_id=paper_id, paper_title="Paper X",
        source_field=SOURCE_ABSTRACT, iteration_seen=1,
    )


# ── gate_finalize transitions the target to in_flight ───────────────────────


@pytest.mark.asyncio
async def test_gate_finalize_marks_target_in_flight():
    """Once gate_finalize emits a FinalizeForceAction the target claim
    must be ``in_flight`` so the next finalize pass doesn't pick the
    same span again and burn the per-turn budget on one claim."""
    state = _make_state()
    state.claim_ledger.add(_provisional_claim())
    mw = FullPaperVerificationMiddleware()

    with patch(
        "app.assistant.react.middlewares.full_paper_gate.get_tool",
        return_value=MagicMock(),
    ):
        gate = await mw.gate_finalize(state)

    assert isinstance(gate, FinalizeForceAction)
    assert gate.action == "paper_qa"
    target = next(iter(state.claim_ledger.by_key.values()))
    assert target.verdict == "in_flight"
    # The mapping that after_tool reads is stamped on state.
    assert getattr(state, "_fpg_inflight", None) == {"p1": target.span}
    # needs_verification must be False now so a re-entry picks a
    # different claim (or yields).
    assert target.needs_verification() is False
    assert state.forced_paper_qa == 1


@pytest.mark.asyncio
async def test_gate_finalize_yields_when_paper_qa_unavailable():
    """When paper_qa isn't registered the middleware must yield and
    label any unverified claims as ``unverifiable`` rather than emit a
    forced action that the loop couldn't dispatch."""
    state = _make_state()
    state.claim_ledger.add(_provisional_claim())
    mw = FullPaperVerificationMiddleware()

    with patch(
        "app.assistant.react.middlewares.full_paper_gate.get_tool",
        return_value=None,
    ):
        gate = await mw.gate_finalize(state)

    assert isinstance(gate, FinalizeAllow)
    target = next(iter(state.claim_ledger.by_key.values()))
    assert target.verdict == "unverifiable"


# ── after_tool resolves the in-flight verdict ───────────────────────────────


@pytest.mark.asyncio
async def test_after_tool_resolves_in_flight_to_verified():
    """An affirmative paper_qa answer must transition the in-flight
    claim to ``verified`` so the synthesizer can quote it firmly."""
    claim = _provisional_claim()
    claim.verdict = "in_flight"
    claim.verification_note = "paper_qa forced at finalize"
    ledger = ClaimLedger()
    ledger.add(claim)

    state = _make_state(claim_ledger=ledger)
    state._fpg_inflight = {claim.paper_id: claim.span}  # type: ignore[attr-defined]
    state.iteration_count = 3

    result = ToolResult(
        output={
            "paper_id": claim.paper_id, "paper_title": "Paper X",
            "answer": "Yes, the paper confirms the 95% accuracy on benchmark Q.",
            "found": True, "chunks_used": 6,
        },
        summary="paper_qa ok",
    )
    mw = FullPaperVerificationMiddleware()
    await mw.after_tool(state, "paper_qa", {"paper_id": claim.paper_id}, result)

    assert claim.verdict == "verified"
    assert claim.verified_at_iteration == 3
    # In-flight map cleared so the same target can't be resolved twice.
    assert claim.paper_id not in (state._fpg_inflight or {})  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_after_tool_resolves_in_flight_to_contradicted():
    claim = _provisional_claim()
    claim.verdict = "in_flight"
    ledger = ClaimLedger()
    ledger.add(claim)

    state = _make_state(claim_ledger=ledger)
    state._fpg_inflight = {claim.paper_id: claim.span}  # type: ignore[attr-defined]

    result = ToolResult(
        output={
            "paper_id": claim.paper_id, "paper_title": "Paper X",
            "answer": "The paper does not support the claim — the experimental section reports 78% instead.",
            "found": True, "chunks_used": 4,
        },
        summary="paper_qa contradiction",
    )
    mw = FullPaperVerificationMiddleware()
    await mw.after_tool(state, "paper_qa", {"paper_id": claim.paper_id}, result)

    assert claim.verdict == "contradicted"


@pytest.mark.asyncio
async def test_after_tool_unverifiable_when_paper_qa_found_false():
    """A paper_qa call that couldn't resolve the paper must collapse
    the in-flight target to ``unverifiable`` so it surfaces honestly
    in the synthesizer's agent_notes."""
    claim = _provisional_claim()
    claim.verdict = "in_flight"
    ledger = ClaimLedger()
    ledger.add(claim)

    state = _make_state(claim_ledger=ledger)
    state._fpg_inflight = {claim.paper_id: claim.span}  # type: ignore[attr-defined]

    result = ToolResult(
        output={
            "paper_id": "", "paper_title": "", "found": False, "answer": "",
        },
        summary="paper_qa not found",
    )
    mw = FullPaperVerificationMiddleware()
    # found=False keeps paper_id empty; pass the in-flight paper_id via params.
    await mw.after_tool(state, "paper_qa", {"paper_id": claim.paper_id}, result)

    assert claim.verdict == "unverifiable"
    assert "could not resolve" in claim.verification_note


@pytest.mark.asyncio
async def test_after_tool_skips_model_initiated_paper_qa():
    """When the model itself calls paper_qa (no forced in-flight target
    on state), the after_tool resolver must NOT touch existing
    provisional claims — that would side-effect random verdicts onto
    unrelated claims."""
    claim = _provisional_claim()
    ledger = ClaimLedger()
    ledger.add(claim)

    state = _make_state(claim_ledger=ledger)
    # No state._fpg_inflight set — model-initiated call.

    result = ToolResult(
        output={
            "paper_id": claim.paper_id, "paper_title": "Paper X",
            "answer": "Yes, the paper supports this.",
            "found": True, "chunks_used": 6,
        },
        summary="paper_qa ok",
    )
    mw = FullPaperVerificationMiddleware()
    await mw.after_tool(state, "paper_qa", {"paper_id": claim.paper_id}, result)

    # Verdict untouched — the gate only resolves claims it explicitly
    # forced. Extraction of new "verified" spans from the answer is the
    # separate code path tested in test_claim_ledger.
    assert claim.verdict == "provisional"


@pytest.mark.asyncio
async def test_yield_marking_unverifiable_sweeps_stranded_in_flight():
    """An ``in_flight`` claim whose forced paper_qa never returned
    (cancel, crash) must be collapsed to ``unverifiable`` so the
    synthesizer doesn't see a claim stuck in transition."""
    claim = _provisional_claim()
    claim.verdict = "in_flight"
    claim.verification_note = "paper_qa forced at finalize"
    ledger = ClaimLedger()
    ledger.add(claim)

    state = _make_state(claim_ledger=ledger)
    # Drive forced_paper_qa to the module's per-turn cap, whatever it is.
    from app.assistant.react.middlewares.full_paper_gate import (
        _MAX_FORCED_PAPER_QA_PER_TURN,
    )
    state.forced_paper_qa = _MAX_FORCED_PAPER_QA_PER_TURN
    mw = FullPaperVerificationMiddleware()

    with patch(
        "app.assistant.react.middlewares.full_paper_gate.get_tool",
        return_value=MagicMock(),
    ):
        gate = await mw.gate_finalize(state)

    assert isinstance(gate, FinalizeAllow)
    assert claim.verdict == "unverifiable"
    assert "did not resolve" in claim.verification_note


# ── Type-aware verification question ────────────────────────────────────────


def test_numeric_claim_question_targets_results_section():
    """A numeric claim must produce a question that asks for the
    table / figure / experimental passage — not a generic 'does the
    paper support this?' prompt that the in-paper retriever has no
    anchor for."""
    from app.assistant.react.middlewares.full_paper_gate import (
        _build_verification_question,
    )
    claim = _provisional_claim(span="Our model achieves 95% accuracy on benchmark Q.")
    q = _build_verification_question(claim)
    assert "RESULTS" in q or "EXPERIMENTS" in q
    assert "quote" in q.lower()
    assert "95%" in q


def test_causal_claim_question_asks_for_ablation():
    """SOTA / causal claims must push the in-paper retriever to look
    for the controlled comparison, not just the abstract's repetition
    of the claim."""
    from app.assistant.react.middlewares.full_paper_gate import (
        _build_verification_question,
    )
    claim = _provisional_claim(span="Attention causes the model to outperform all baselines.")
    q = _build_verification_question(claim)
    assert "ABLATION" in q or "ablation" in q.lower()
    assert "baseline" in q.lower()


def test_comparative_claim_question_asks_for_head_to_head():
    from app.assistant.react.middlewares.full_paper_gate import (
        _build_verification_question,
    )
    claim = _provisional_claim(span="Our system outperforms GPT-4 on this task.")
    q = _build_verification_question(claim)
    assert "head-to-head" in q.lower()
    assert "matched" in q.lower() or "fair" in q.lower()


def test_generic_claim_falls_back_to_default_prompt():
    from app.assistant.react.middlewares.full_paper_gate import (
        _build_verification_question,
    )
    claim = _provisional_claim(span="The proposed approach handles edge cases gracefully.")
    q = _build_verification_question(claim)
    assert "full text" in q.lower()
    assert "body" in q.lower()


def test_priority_abstract_sourced_claims_first():
    """A strong claim sourced from an abstract is more dangerous than
    one sourced from a paper_qa chunk, so the priority ordering must
    surface the abstract-sourced claim first when both are otherwise
    equivalent."""
    from app.assistant.react.middlewares.full_paper_gate import _priority
    from app.assistant.claim_ledger import SOURCE_ABSTRACT, SOURCE_CHUNK, StrongClaim
    abs_claim = StrongClaim(
        span="Achieves 95% on benchmark Q.", paper_id="p1", paper_title="X",
        source_field=SOURCE_ABSTRACT, iteration_seen=1,
    )
    chunk_claim = StrongClaim(
        span="Achieves 95% on benchmark Q.", paper_id="p2", paper_title="Y",
        source_field=SOURCE_CHUNK, iteration_seen=1,
    )
    ordered = sorted([chunk_claim, abs_claim], key=_priority)
    assert ordered[0] is abs_claim
