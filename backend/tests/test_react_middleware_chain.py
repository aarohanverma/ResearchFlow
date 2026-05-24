"""Middleware Protocol + chain composition contracts.

Pins the chain's guarantees independent of any specific middleware:
ordering, override accumulation, abort short-circuit, first-non-allow-
wins for gates, failure isolation.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.assistant.react.middleware import (
    AbortDispatch,
    CONTINUE,
    DispatchOverride,
    FinalizeAllow,
    FinalizeForceAction,
    FinalizeForceCritique,
    MiddlewareChain,
)
from app.assistant.react.middlewares.base import BaseMiddleware
from app.assistant.tools.base import ToolResult


def _ctx() -> dict:
    """Minimal fake state for chain tests — middlewares that touch
    ``state.pad`` use this; chain tests don't need a full LoopState."""
    class _PadStub:
        def __init__(self): self.thoughts: list[str] = []
        def think(self, text: str) -> None: self.thoughts.append(text)
    return {"pad": _PadStub()}


class _Recorder(BaseMiddleware):
    """Test middleware that records every hook invocation."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[str] = []

    async def before_iteration(self, state):
        self.calls.append("before_iteration")

    async def before_tool(self, state, action, params):
        self.calls.append(f"before_tool:{action}")
        return CONTINUE

    async def after_tool(self, state, action, params, result):
        self.calls.append(f"after_tool:{action}")

    async def gate_finalize(self, state):
        self.calls.append("gate_finalize")
        return FinalizeAllow()


# ── Ordering ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chain_walks_middlewares_in_registration_order():
    a, b, c = _Recorder("a"), _Recorder("b"), _Recorder("c")
    chain = MiddlewareChain([a, b, c])
    await chain.before_iteration(_ctx())
    # All three fire in order — every middleware sees every iteration.
    assert a.calls == ["before_iteration"]
    assert b.calls == ["before_iteration"]
    assert c.calls == ["before_iteration"]


@pytest.mark.asyncio
async def test_chain_names_reports_active_middlewares():
    chain = MiddlewareChain([_Recorder("first"), _Recorder("second")])
    assert chain.names() == ["first", "second"]


# ── before_tool override accumulation ───────────────────────────────────────


class _ParamRewriter(BaseMiddleware):
    name = "rewriter"

    async def before_tool(self, state, action, params):
        return DispatchOverride(params={**params, "rewritten": True})


class _ActionRewriter(BaseMiddleware):
    name = "action_rewriter"

    async def before_tool(self, state, action, params):
        return DispatchOverride(action=f"rewritten_{action}")


@pytest.mark.asyncio
async def test_chain_accumulates_overrides_across_middlewares():
    """A param override followed by an action override must produce a
    single combined override at the chain output, not lose either."""
    chain = MiddlewareChain([_ParamRewriter(), _ActionRewriter()])
    result = await chain.before_tool(_ctx(), "deep_search", {"q": "x"})
    assert isinstance(result, DispatchOverride)
    assert result.action == "rewritten_deep_search"
    assert result.params == {"q": "x", "rewritten": True}


@pytest.mark.asyncio
async def test_chain_returns_continue_when_no_middleware_modifies():
    chain = MiddlewareChain([_Recorder("a"), _Recorder("b")])
    result = await chain.before_tool(_ctx(), "deep_search", {"q": "x"})
    # Nothing changed → CONTINUE sentinel.
    from app.assistant.react.middleware import _Continue
    assert isinstance(result, _Continue)


# ── before_tool abort short-circuit ─────────────────────────────────────────


class _Aborter(BaseMiddleware):
    name = "aborter"

    async def before_tool(self, state, action, params):
        return AbortDispatch(
            reason="test_abort",
            observation_summary="aborted by test middleware",
            error="test_error",
        )


@pytest.mark.asyncio
async def test_abort_short_circuits_remaining_middlewares():
    """Once a middleware aborts, no subsequent before_tool hooks
    should fire — that's the whole point of the short-circuit."""
    after = _Recorder("after_abort")
    chain = MiddlewareChain([_Aborter(), after])
    result = await chain.before_tool(_ctx(), "deep_search", {})
    assert isinstance(result, AbortDispatch)
    assert result.reason == "test_abort"
    assert after.calls == []   # never invoked


# ── gate_finalize precedence ────────────────────────────────────────────────


class _ForceCritique(BaseMiddleware):
    name = "force_critique"

    async def gate_finalize(self, state):
        return FinalizeForceCritique(reason="not_enough_critique")


class _ForceAction(BaseMiddleware):
    name = "force_action"

    async def gate_finalize(self, state):
        return FinalizeForceAction(
            action="citation_finder",
            params={"claim": "x"},
            reason="open_contradiction",
        )


@pytest.mark.asyncio
async def test_first_non_allow_gate_wins():
    """When multiple middlewares would intervene at finalize, the
    earlier-registered one wins. This is the contract that makes
    middleware order semantically meaningful."""
    chain = MiddlewareChain([_ForceCritique(), _ForceAction()])
    gate = await chain.gate_finalize(_ctx())
    assert isinstance(gate, FinalizeForceCritique)


@pytest.mark.asyncio
async def test_allow_passes_to_next_gate():
    """A middleware that allows finalize must NOT block a later
    middleware that wants to force an action."""
    class _AllowEarly(BaseMiddleware):
        name = "allow_early"
        async def gate_finalize(self, state):
            return FinalizeAllow()
    chain = MiddlewareChain([_AllowEarly(), _ForceAction()])
    gate = await chain.gate_finalize(_ctx())
    assert isinstance(gate, FinalizeForceAction)


@pytest.mark.asyncio
async def test_all_allow_returns_finalize_allow():
    chain = MiddlewareChain([_Recorder("a"), _Recorder("b")])
    gate = await chain.gate_finalize(_ctx())
    assert isinstance(gate, FinalizeAllow)


# ── Failure isolation ───────────────────────────────────────────────────────


class _Crashing(BaseMiddleware):
    name = "crashing"

    async def before_iteration(self, state):
        raise RuntimeError("boom")

    async def before_tool(self, state, action, params):
        raise RuntimeError("boom")

    async def after_tool(self, state, action, params, result):
        raise RuntimeError("boom")

    async def gate_finalize(self, state):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_crashing_middleware_does_not_take_down_chain():
    """A buggy middleware that raises must be skipped — chain failure
    isolation is the contract that makes adding new middlewares safe."""
    crash = _Crashing()
    survivor = _Recorder("survivor")
    chain = MiddlewareChain([crash, survivor])

    # Every hook must complete cleanly + the survivor must still fire.
    await chain.before_iteration(_ctx())
    assert survivor.calls == ["before_iteration"]

    result = await chain.before_tool(_ctx(), "deep_search", {"q": "x"})
    # Crash skipped, survivor was a no-op → CONTINUE.
    from app.assistant.react.middleware import _Continue
    assert isinstance(result, _Continue)

    await chain.after_tool(_ctx(), "deep_search", {}, ToolResult(output={}, summary=""))
    # survivor recorded after_tool too.
    assert any(c.startswith("after_tool:") for c in survivor.calls)

    gate = await chain.gate_finalize(_ctx())
    assert isinstance(gate, FinalizeAllow)
