"""Per-middleware unit tests.

Each middleware is exercised in isolation against a minimal LoopState
fixture. The chain composition tests in
``test_react_middleware_chain.py`` cover how they interact; these tests
pin each middleware's own contract.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.assistant.contradiction import ContradictionLedger, ContradictionSignal
from app.assistant.react.middleware import (
    AbortDispatch,
    DispatchOverride,
    FinalizeAllow,
    FinalizeForceAction,
    FinalizeForceCritique,
)
from app.assistant.react.middlewares.contradiction_mw import ContradictionMiddleware
from app.assistant.react.middlewares.critic_gate import CriticGateMiddleware
from app.assistant.react.middlewares.diminishing_returns import DiminishingReturnsMiddleware
from app.assistant.react.middlewares.observability_mw import RetrievalObservabilityMiddleware
from app.assistant.react.middlewares.paper_ledger import PaperLedgerMiddleware
from app.assistant.react.middlewares.param_preflight import ParamPreflightMiddleware
from app.assistant.react.middlewares.tool_ban import ToolBanMiddleware
from app.assistant.react.state import LoopState
from app.assistant.react_loop import PaperLedger, ReactConfig
from app.assistant.retrieval_observability import RetrievalObservability
from app.assistant.scratchpad import Scratchpad
from app.assistant.tools.base import ToolResult


def _make_state(*, query: str = "test query", max_iterations: int = 5) -> LoopState:
    """Minimal LoopState fixture for middleware unit tests."""
    config = ReactConfig(max_iterations=max_iterations, deadline_seconds=10.0)
    state = LoopState(
        query=query,
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
    return state


# ── ToolBan ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_ban_blocks_disallowed_from_loop():
    """memory_write / memory_delete must never run from inside the loop —
    durable memory writes are the post-turn pass's job."""
    state = _make_state()
    mw = ToolBanMiddleware()
    result = await mw.before_tool(state, "memory_write", {"key": "x", "value": "y"})
    assert isinstance(result, AbortDispatch)
    assert "disallowed" in result.reason


@pytest.mark.asyncio
async def test_tool_ban_blocks_banned_tool():
    state = _make_state()
    state.banned_tools.add("deep_search")
    mw = ToolBanMiddleware()
    result = await mw.before_tool(state, "deep_search", {"query": "x"})
    assert isinstance(result, AbortDispatch)
    assert result.error == "tool_banned"


@pytest.mark.asyncio
async def test_tool_ban_on_error_increments_counter():
    state = _make_state()
    mw = ToolBanMiddleware()
    await mw.on_tool_error(state, "deep_search", {}, RuntimeError("boom"))
    assert state.tool_failures == 1
    assert state.tool_fail_counts["deep_search"] == 1


@pytest.mark.asyncio
async def test_tool_ban_promotes_to_banned_set_after_cap():
    """Two failures of the same tool must add it to banned_tools — the
    third dispatch in the same turn would otherwise burn an iteration
    on a known-broken tool."""
    state = _make_state()
    mw = ToolBanMiddleware()
    await mw.on_tool_error(state, "deep_search", {}, RuntimeError("boom1"))
    await mw.on_tool_error(state, "deep_search", {}, RuntimeError("boom2"))
    assert "deep_search" in state.banned_tools


# ── PaperLedger ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_paper_ledger_collects_ids_from_result():
    state = _make_state()
    mw = PaperLedgerMiddleware()
    result = ToolResult(
        output={"papers": [
            {"paper_id": "p1", "title": "A"},
            {"paper_id": "p2", "title": "B"},
        ]},
        summary="found 2",
    )
    await mw.after_tool(state, "deep_search", {}, result)
    assert state.ledger.ids() == ["p1", "p2"]
    assert state.successful_retrievals == 1


@pytest.mark.asyncio
async def test_paper_ledger_skips_non_retrieval_tools():
    """compare_papers / paper_qa results may not have paper IDs in the
    'papers' shape — the ledger gracefully returns 0 additions and we
    do NOT bump the retrieval counter."""
    state = _make_state()
    mw = PaperLedgerMiddleware()
    result = ToolResult(output={"rows": []}, summary="compared")
    await mw.after_tool(state, "compare_papers", {}, result)
    assert state.successful_retrievals == 0


# ── RetrievalObservability ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_observability_records_snapshot_for_retrieval_tool():
    state = _make_state()
    mw = RetrievalObservabilityMiddleware()
    result = ToolResult(
        output={"papers": [{"paper_id": f"p{i}", "search_score": 0.7} for i in range(5)]},
        summary="found 5",
    )
    await mw.after_tool(state, "deep_search", {"limit": 5}, result)
    assert len(state.retrieval_obs.snapshots) == 1


@pytest.mark.asyncio
async def test_observability_warns_on_thin_retrieval():
    state = _make_state()
    mw = RetrievalObservabilityMiddleware()
    # 1 paper for a limit of 8 → coverage 0.125, below thin threshold.
    result = ToolResult(
        output={"papers": [{"paper_id": "p1", "search_score": 0.4}]},
        summary="found 1",
    )
    await mw.after_tool(state, "deep_search", {"limit": 8}, result)
    # Scratchpad got a thin-retrieval warning.
    assert any(
        "Retrieval quality warning" in t.text
        for t in state.pad.thoughts()
    )


# ── DiminishingReturns ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_diminishing_returns_after_tool_sets_flag_on_saturation():
    """A retrieval that adds only papers we've already seen should mark
    ``state._diminishing_returns_hit`` so the loop driver exits."""
    state = _make_state()
    # Seed the ledger so the new result's papers are all duplicates.
    seed = ToolResult(
        output={"papers": [{"paper_id": "p1"}, {"paper_id": "p2"}]},
        summary="seed",
    )
    state.prior_results["deep_search"] = seed
    state.ledger.add_from_result(seed)

    mw = DiminishingReturnsMiddleware()
    redundant = ToolResult(
        output={"papers": [{"paper_id": "p1"}, {"paper_id": "p2"}]},
        summary="redundant",
    )
    await mw.after_tool(state, "deep_search", {"query": "different"}, redundant)
    assert getattr(state, "_diminishing_returns_hit", False) is True
    assert state.completed_normally is True


# ── ParamPreflight ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_param_preflight_repairs_placeholder(monkeypatch):
    """Empty/placeholder query gets auto-filled from the user query."""
    from app.assistant.tools.registry import register_tool
    from pydantic import BaseModel, Field

    class _Input(BaseModel):
        query: str = Field(min_length=1)

    class _Out(BaseModel):
        ok: bool = True

    fake = MagicMock()
    fake.name = "tst_preflight"
    fake.summary = "test"
    fake.cost_class = "cheap"
    fake.side_effects = False
    fake.cancellable = False
    fake.streamable = False
    fake.input_schema = _Input
    fake.output_schema = _Out
    register_tool(fake)

    state = _make_state(query="What is BERT?")
    mw = ParamPreflightMiddleware()
    result = await mw.before_tool(state, "tst_preflight", {"query": "__to_fill__"})
    assert isinstance(result, DispatchOverride)
    assert result.params["query"] == "What is BERT?"


# ── ContradictionMiddleware ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_contradiction_detects_lexical_marker_on_result():
    state = _make_state()
    mw = ContradictionMiddleware(enable_semantic_llm=False)
    result = ToolResult(
        output={"papers": [{"title": "X", "abstract": "This refutes prior claims."}]},
        summary="ran",
    )
    await mw.after_tool(state, "deep_search", {"query": "x"}, result)
    assert any(s.kind == "lexical" for s in state.contradictions.signals)


@pytest.mark.asyncio
async def test_contradiction_gate_forces_counter_search_on_open_signal():
    state = _make_state(max_iterations=5)
    state.iteration_count = 1
    state.contradictions.add(ContradictionSignal(
        kind="lexical",
        span="strong refutation here",
        sources=["deep_search"],
        confidence=0.9,
    ))
    mw = ContradictionMiddleware(enable_semantic_llm=False)
    gate = await mw.gate_finalize(state)
    assert isinstance(gate, FinalizeForceAction)
    assert "claim" in gate.params or "query" in gate.params


@pytest.mark.asyncio
async def test_contradiction_gate_allows_finalize_when_no_open_signal():
    state = _make_state(max_iterations=5)
    state.iteration_count = 2
    mw = ContradictionMiddleware(enable_semantic_llm=False)
    gate = await mw.gate_finalize(state)
    assert isinstance(gate, FinalizeAllow)


@pytest.mark.asyncio
async def test_contradiction_gate_skips_on_last_iteration():
    """Iteration cap is the hard ceiling — never blow past it on a
    contradiction signal."""
    state = _make_state(max_iterations=3)
    state.iteration_count = 3
    state.is_last_iteration = True
    state.contradictions.add(ContradictionSignal(
        kind="lexical", span="claim", sources=["a"], confidence=0.95,
    ))
    mw = ContradictionMiddleware(enable_semantic_llm=False)
    gate = await mw.gate_finalize(state)
    assert isinstance(gate, FinalizeAllow)


# ── CriticGate ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_critic_gate_forces_critique_on_too_early_finalize():
    state = _make_state(max_iterations=5)
    state.iteration_count = 1  # Way below MIN_ITERS
    mw = CriticGateMiddleware()
    gate = await mw.gate_finalize(state)
    assert isinstance(gate, FinalizeForceCritique)


@pytest.mark.asyncio
async def test_critic_gate_allows_when_critique_already_present():
    state = _make_state(max_iterations=5)
    state.iteration_count = 1
    state.pad.critique(
        groundedness=0.8, completeness=0.8, memory_faithfulness=1.0,
        issues=[], verdict="ship",
    )
    mw = CriticGateMiddleware()
    gate = await mw.gate_finalize(state)
    assert isinstance(gate, FinalizeAllow)


@pytest.mark.asyncio
async def test_critic_gate_fires_only_once_per_turn():
    state = _make_state(max_iterations=8)
    state.iteration_count = 1
    mw = CriticGateMiddleware()
    first = await mw.gate_finalize(state)
    assert isinstance(first, FinalizeForceCritique)
    # The state counter should now block a second force this turn.
    second = await mw.gate_finalize(state)
    assert isinstance(second, FinalizeAllow)


@pytest.mark.asyncio
async def test_critic_gate_allows_on_last_iteration():
    state = _make_state(max_iterations=3)
    state.iteration_count = 3
    state.is_last_iteration = True
    mw = CriticGateMiddleware()
    gate = await mw.gate_finalize(state)
    assert isinstance(gate, FinalizeAllow)
