"""Tests for the per-tool wall-clock guardrail in the ReAct loop.

The loop's outer deadline only fires between iterations. Without a
per-tool timeout a single hanging tool (slow provider, infinite
poll, unresponsive upstream) would block the loop past its deadline
indefinitely, defeating both the cap and the user's Stop button.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.assistant.react_loop import (
    PaperLedger,
    ReactConfig,
    run_react_loop,
)
from app.assistant.tools.base import ToolContext, ToolResult


class _HangingTool:
    """Stand-in tool whose ``run`` awaits forever — exercises the
    per-tool timeout guardrail."""

    name = "hanging_tool"
    summary = "test tool that hangs"
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False

    class _Schema:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def model_json_schema():
            return {"properties": {"q": {"type": "string"}}, "required": []}

    input_schema = _Schema
    output_schema = _Schema

    async def run(self, ctx: Any, params: Any) -> ToolResult:
        # Block well past any conceivable wall-clock deadline. Should
        # never actually finish — the timeout guardrail must intercept.
        await asyncio.sleep(120.0)
        return ToolResult(output={}, summary="should not happen")


class _FakeCtx:
    """Minimal ToolContext-shaped stub for the dispatch path."""
    db = None
    user_id = None
    session_id = None
    job_id = "test-job"
    namespace_key = ""

    async def emit_progress(self, *_args, **_kwargs) -> None:
        return None

    async def should_cancel(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_hanging_tool_times_out_within_loop_deadline(monkeypatch):
    """A tool that awaits forever must be interrupted by the
    per-tool wait_for guardrail and surface as a tool failure
    observation, not deadlock the loop past its wall-clock deadline."""
    from app.assistant.tools.registry import register_tool, get_tool

    # Register the hanging tool only for the duration of this test.
    hanging = _HangingTool()
    register_tool(hanging)
    try:
        assert get_tool("hanging_tool") is hanging

        # Decision step returns ONE call to the hanging tool then
        # finalize. The dispatch must time out and the loop must
        # complete cleanly (not hang).
        decisions = iter([
            {"thought": "call the hanging tool", "action": "hanging_tool", "params": {"q": "x"}},
            {"thought": "now finalize", "action": "finalize", "params": {}},
        ])

        async def _fake_decide(*_args, **_kwargs):
            try:
                return next(decisions)
            except StopIteration:
                return {"thought": "done", "action": "finalize", "params": {}}

        monkeypatch.setattr(
            "app.assistant.react_loop._decide_next_action",
            _fake_decide,
        )

        cfg = ReactConfig(
            max_iterations=3,
            # Very short deadline so the wait_for guardrail's
            # ``state.time_remaining() + 2`` cap fires quickly during
            # the test. The hanging tool sleeps 120s — without the
            # guard the loop would block for ~120s.
            deadline_seconds=2.0,
        )
        ctx = _FakeCtx()

        t0 = time.monotonic()
        outcome = await asyncio.wait_for(
            run_react_loop(
                query="hang test",
                initial_plan_actions=[],
                prior_results={},
                memory_view={},
                research_brief_text="",
                ctx=ctx,
                config=cfg,
            ),
            # Hard cap on the test itself — guard against the
            # guardrail being broken and the loop actually hanging.
            timeout=15.0,
        )
        elapsed = time.monotonic() - t0

        # The loop must finish in roughly its deadline + dispatch
        # timeout grace, not in the hanging tool's full 120s sleep.
        assert elapsed < 10.0, f"loop hung for {elapsed:.1f}s — guardrail did not fire"

        # The hanging tool's failure must show up on the scratchpad
        # as an explicit timeout observation.
        observations = [
            e for e in outcome.scratchpad.entries
            if getattr(e, "kind", "") == "observation"
        ]
        timed_out = [o for o in observations if getattr(o, "error", "") == "tool_timeout"]
        assert timed_out, "expected an observation with error='tool_timeout'"

        # The timeout MUST also bump the per-tool failure counter —
        # the ToolBanMiddleware's ``on_tool_error`` hook should run
        # via the timeout's chain.on_tool_error call. Without this,
        # middleware observability silently misses every tool
        # timeout (regression guard for the original bug where the
        # timeout path skipped on_tool_error).
        assert outcome.tool_failures >= 1, (
            "tool_failures should be incremented for timeouts via "
            "the middleware on_tool_error hook"
        )
    finally:
        # Tool registry is process-global; remove our test tool so
        # downstream tests don't see it. The registry's private
        # mapping is the cleanest way to back out.
        from app.assistant.tools import registry as _reg
        _reg._TOOLS.pop("hanging_tool", None)  # noqa: SLF001 — test cleanup


@pytest.mark.asyncio
async def test_timeout_invokes_middleware_on_tool_error(monkeypatch):
    """Custom middleware's ``on_tool_error`` hook must fire on a tool
    timeout. Regression for a silent bug where the timeout branch
    bypassed the middleware error chain — observability middlewares
    (retrieval metrics, contradiction signals, per-tool ban counters)
    never saw timeouts.
    """
    from app.assistant.tools.registry import register_tool
    from app.assistant.react.middlewares import default_chain_factory
    from app.assistant.react.middlewares.base import BaseMiddleware

    hanging = _HangingTool()
    register_tool(hanging)

    error_records: list[tuple[str, type]] = []

    class _RecordingMiddleware(BaseMiddleware):
        name = "_recording_for_test"

        async def on_tool_error(
            self, state, action: str, params: dict, exc: BaseException,
        ) -> None:
            error_records.append((action, type(exc)))

    # Splice the recording middleware in front of the default chain
    # so it observes the error first. monkeypatching the factory
    # keeps the patch surgical.
    real_factory = default_chain_factory

    def _patched_factory(*args, **kwargs):
        return [_RecordingMiddleware()] + list(real_factory(*args, **kwargs))

    monkeypatch.setattr(
        "app.assistant.react.middlewares.default_chain_factory",
        _patched_factory,
    )
    # The loop imports default_chain_factory via the middlewares
    # package init lazily — patch the symbol it actually binds.
    import app.assistant.react.middlewares as _mws_pkg
    monkeypatch.setattr(_mws_pkg, "default_chain_factory", _patched_factory)

    decisions = iter([
        {"thought": "call the hanging tool", "action": "hanging_tool", "params": {"q": "x"}},
        {"thought": "finalize now", "action": "finalize", "params": {}},
    ])

    async def _fake_decide(*_args, **_kwargs):
        try:
            return next(decisions)
        except StopIteration:
            return {"thought": "done", "action": "finalize", "params": {}}

    monkeypatch.setattr(
        "app.assistant.react_loop._decide_next_action",
        _fake_decide,
    )

    try:
        outcome = await asyncio.wait_for(
            run_react_loop(
                query="middleware on_tool_error timeout regression",
                initial_plan_actions=[],
                prior_results={},
                memory_view={},
                research_brief_text="",
                ctx=_FakeCtx(),
                config=ReactConfig(max_iterations=3, deadline_seconds=2.0),
            ),
            timeout=15.0,
        )
        assert any(
            action == "hanging_tool" and issubclass(et, asyncio.TimeoutError)
            for action, et in error_records
        ), (
            f"middleware on_tool_error was not invoked for the timeout — "
            f"recorded: {error_records!r}"
        )
        # And one (not two) increments — the timeout path no longer
        # double-counts after going through the chain.
        assert outcome.tool_failures == 1, (
            f"timeout should increment tool_failures exactly once via the "
            f"middleware; got {outcome.tool_failures}"
        )
    finally:
        from app.assistant.tools import registry as _reg
        _reg._TOOLS.pop("hanging_tool", None)  # noqa: SLF001
