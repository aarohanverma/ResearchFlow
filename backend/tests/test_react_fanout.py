"""Multi-branch parallel fanout in the ReAct loop."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest

from app.assistant import react_loop as rl
from app.assistant.tools.base import ToolResult


def _ctx_factory():
    from contextlib import asynccontextmanager
    from app.assistant.tools.base import ToolContext

    @asynccontextmanager
    async def _factory():
        async def _never(): return False
        async def _noop(*_a, **_k): return None
        yield ToolContext(
            user_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            namespace_key="cs.AI",
            namespace_keys=["cs.AI"],
            orientation="both",
            expertise_level="practitioner",
            job_id="test",
            parent_message_id=uuid.uuid4(),
            db=MagicMock(),
            should_cancel=_never,
            emit_progress=_noop,
        )
    return _factory


@pytest.mark.asyncio
async def test_fanout_dispatches_branches_concurrently(monkeypatch):
    """Two fanout branches must run as one iteration AND both results
    must land in ``new_results`` — that's the whole point of the
    multi-branch action."""
    from pydantic import BaseModel, Field

    class _In(BaseModel):
        query: str = Field(min_length=1, max_length=400)

    class _Out(BaseModel):
        ok: bool = True

    started = asyncio.Event()
    started_count = {"n": 0}

    async def _tool_a_run(_ctx, params):
        started_count["n"] += 1
        if started_count["n"] >= 2:
            started.set()
        else:
            await asyncio.wait_for(started.wait(), timeout=2.0)
        return ToolResult(
            output={"papers": [{"paper_id": "pa", "title": "A"}]},
            summary="branch A done",
        )

    async def _tool_b_run(_ctx, params):
        started_count["n"] += 1
        if started_count["n"] >= 2:
            started.set()
        else:
            await asyncio.wait_for(started.wait(), timeout=2.0)
        return ToolResult(
            output={"papers": [{"paper_id": "pb", "title": "B"}]},
            summary="branch B done",
        )

    tool_a = MagicMock(); tool_a.input_schema = _In; tool_a.output_schema = _Out
    tool_a.run = _tool_a_run
    tool_b = MagicMock(); tool_b.input_schema = _In; tool_b.output_schema = _Out
    tool_b.run = _tool_b_run

    monkeypatch.setattr(rl, "get_tool", lambda n: {"tool_a": tool_a, "tool_b": tool_b}.get(n))

    decisions = [
        {
            "thought": "two independent sub-questions",
            "action": "fanout",
            "branches": [
                {"tool": "tool_a", "params": {"query": "alpha"}, "rationale": "first branch"},
                {"tool": "tool_b", "params": {"query": "beta"}, "rationale": "second branch"},
            ],
        },
        {"thought": "done", "action": "finalize"},
    ]
    forced_critique = {"hit": False}

    async def _decide(**_kw):
        # The forced-critique gate runs an extra iteration before
        # accepting a finalize. We tolerate that by repeating the
        # last decision once we've exhausted the scripted list.
        return decisions.pop(0) if decisions else {"thought": "done", "action": "finalize"}

    async def _fake_critique(**kwargs):
        # Forced critique runs once on the way to finalize because the
        # min-iters guard kicks in. We accept it as a no-op so the
        # second iteration's finalize succeeds.
        forced_critique["hit"] = True
        pad = kwargs["pad"]
        pad.critique(
            groundedness=0.8, completeness=0.8, memory_faithfulness=1.0,
            issues=[], verdict="ship",
        )

    monkeypatch.setattr(rl, "_decide_next_action", _decide)
    monkeypatch.setattr(rl, "_run_self_critique", _fake_critique)

    outcome = await rl.run_react_loop(
        query="alpha vs beta comparison",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx_factory=_ctx_factory(),
        config=rl.ReactConfig(max_iterations=4, deadline_seconds=10),
    )

    # Both branches must have run and landed in new_results.
    assert "tool_a" in outcome.new_results
    assert "tool_b" in outcome.new_results
    # The fanout counted as one iteration (the model's "fanout"
    # decision), then a forced critique, then finalize. So iterations
    # <= 3 even though three tool calls happened.
    assert outcome.iterations <= 3
    assert outcome.completed_normally is True


@pytest.mark.asyncio
async def test_fanout_caps_at_max_branches(monkeypatch):
    """A model that emits 8 branches must still dispatch at most
    ``_MAX_FANOUT_BRANCHES``."""
    from pydantic import BaseModel, Field

    class _In(BaseModel):
        query: str = Field(min_length=1, max_length=400)

    fake_tool = MagicMock()
    fake_tool.input_schema = _In

    runs = {"n": 0}

    async def _run(_ctx, _params):
        runs["n"] += 1
        return ToolResult(output={"papers": []}, summary="ran")

    fake_tool.run = _run

    monkeypatch.setattr(rl, "get_tool", lambda n: fake_tool)

    branches = [{"tool": "t", "params": {"query": f"q{i}"}} for i in range(8)]
    decisions = [
        {"thought": "huge fanout", "action": "fanout", "branches": branches},
        {"thought": "done", "action": "finalize"},
    ]

    async def _decide(**_kw):
        # The forced-critique gate runs an extra iteration before
        # accepting a finalize. We tolerate that by repeating the
        # last decision once we've exhausted the scripted list.
        return decisions.pop(0) if decisions else {"thought": "done", "action": "finalize"}

    async def _fake_critique(**kwargs):
        kwargs["pad"].critique(
            groundedness=0.8, completeness=0.8, memory_faithfulness=1.0,
            issues=[], verdict="ship",
        )

    monkeypatch.setattr(rl, "_decide_next_action", _decide)
    monkeypatch.setattr(rl, "_run_self_critique", _fake_critique)

    await rl.run_react_loop(
        query="x",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx_factory=_ctx_factory(),
        config=rl.ReactConfig(max_iterations=4, deadline_seconds=10),
    )
    assert runs["n"] <= rl._MAX_FANOUT_BRANCHES


@pytest.mark.asyncio
async def test_fanout_isolates_branch_failure(monkeypatch):
    """One branch raising must not abort the others. The good branch's
    result still lands in ``new_results``; the bad one logs an error
    observation."""
    from pydantic import BaseModel, Field

    class _In(BaseModel):
        query: str = Field(min_length=1, max_length=400)

    good = MagicMock(); good.input_schema = _In
    bad = MagicMock(); bad.input_schema = _In

    async def _good_run(_c, _p):
        return ToolResult(output={"papers": [{"paper_id": "g", "title": "G"}]}, summary="good")

    async def _bad_run(_c, _p):
        raise RuntimeError("branch exploded")

    good.run = _good_run
    bad.run = _bad_run

    monkeypatch.setattr(rl, "get_tool", lambda n: {"good": good, "bad": bad}.get(n))

    decisions = [
        {
            "thought": "two branches",
            "action": "fanout",
            "branches": [
                {"tool": "good", "params": {"query": "g"}},
                {"tool": "bad",  "params": {"query": "b"}},
            ],
        },
        {"thought": "done", "action": "finalize"},
    ]

    async def _decide(**_kw):
        # The forced-critique gate runs an extra iteration before
        # accepting a finalize. We tolerate that by repeating the
        # last decision once we've exhausted the scripted list.
        return decisions.pop(0) if decisions else {"thought": "done", "action": "finalize"}

    async def _fake_critique(**kwargs):
        kwargs["pad"].critique(
            groundedness=0.8, completeness=0.8, memory_faithfulness=1.0,
            issues=[], verdict="ship",
        )

    monkeypatch.setattr(rl, "_decide_next_action", _decide)
    monkeypatch.setattr(rl, "_run_self_critique", _fake_critique)

    outcome = await rl.run_react_loop(
        query="x",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx_factory=_ctx_factory(),
        config=rl.ReactConfig(max_iterations=4, deadline_seconds=10),
    )

    # Good branch survived; bad branch logged but didn't abort.
    assert "good" in outcome.new_results
    assert "bad" not in outcome.new_results
    # Failure incremented the fail counter.
    assert outcome.tool_failures >= 1
