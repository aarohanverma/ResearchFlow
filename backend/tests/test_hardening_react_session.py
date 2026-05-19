"""Regression tests for the three audit-flagged production-stability bugs
introduced with the ReAct loop integration:

* The loop must open a fresh DB session per tool action (idle-in-tx kill
  risk + flush-without-commit data loss when one session is shared
  across iterations).
* The loop must honour an external cancellation signal between
  iterations, not only the wall-clock deadline.
* Active context must reach the loop's decision prompt — the inventory
  is stored on the orchestrator's ``task`` object in
  ``_load_session_context`` and the loop reads it from there.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest


def _make_ctx_factory_recorder():
    """Build a ctx_factory that records how many times it was opened.

    Each call to the factory must return a fresh contextmanager so
    SQLAlchemy session lifetimes are tied to the single action, not
    the whole loop. We don't need a real DB here; we just observe
    that the loop opens a new context per tool invocation.
    """
    from contextlib import asynccontextmanager
    from app.assistant.tools.base import ToolContext

    calls: dict = {"open_count": 0, "exit_count": 0}

    @asynccontextmanager
    async def _factory():
        calls["open_count"] += 1
        # Fresh ToolContext per call — mirrors the per-action session
        # pattern the production orchestrator uses.
        async def _never_cancel() -> bool: return False
        async def _noop(*_a, **_k): return None
        ctx = ToolContext(
            user_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            namespace_key="cs.AI",
            namespace_keys=["cs.AI"],
            orientation="both",
            expertise_level="practitioner",
            job_id="test-job",
            parent_message_id=uuid.uuid4(),
            db=MagicMock(),
            should_cancel=_never_cancel,
            emit_progress=_noop,
        )
        try:
            yield ctx
        finally:
            calls["exit_count"] += 1

    return _factory, calls


@pytest.mark.asyncio
async def test_react_loop_opens_fresh_ctx_per_tool_call(monkeypatch):
    """Two tool calls inside one loop must open two ctx_factory
    contexts. A single shared session would have just one open."""
    from app.assistant import react_loop as rl
    from app.assistant.tools.base import ToolResult

    factory, calls = _make_ctx_factory_recorder()

    decisions = [
        {"thought": "first", "action": "fake_tool", "params": {}},
        {"thought": "second", "action": "fake_tool", "params": {"q": "x"}},  # different params so dedup skip doesn't fire
        {"thought": "done", "action": "finalize"},
    ]

    async def _decide(**_kw):
        return decisions.pop(0)

    fake_tool = MagicMock()
    fake_tool.input_schema = lambda **kw: kw

    async def _fake_run(_ctx, _params):
        return ToolResult(output={"answer": "ok"}, summary="ran")

    fake_tool.run = _fake_run

    monkeypatch.setattr(rl, "_decide_next_action", _decide)
    monkeypatch.setattr(rl, "get_tool", lambda name: fake_tool if name == "fake_tool" else None)

    outcome = await rl.run_react_loop(
        query="test",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx_factory=factory,
        config=rl.ReactConfig(max_iterations=4, deadline_seconds=10),
    )
    assert calls["open_count"] == 2, (
        f"expected 2 ctx_factory opens (one per tool action), got {calls['open_count']}"
    )
    assert calls["exit_count"] == 2, "every opened ctx must also be closed (no leaks)"
    assert outcome.completed_normally is True


@pytest.mark.asyncio
async def test_react_loop_honours_should_cancel_between_iterations(monkeypatch):
    """When the orchestrator's cancel signal flips True, the loop must
    stop on the *next* iteration boundary regardless of the deadline."""
    from app.assistant import react_loop as rl

    cancel_flipped = {"value": False}

    async def _should_cancel() -> bool:
        return cancel_flipped["value"]

    iteration_seen: list[int] = []

    async def _decide(**kw):
        iteration_seen.append(kw["pad"].iteration)
        # Flip cancel AFTER the first iteration completes — the loop
        # must then exit on the second iteration's cancel check, not
        # invoke the LLM decision step again.
        if len(iteration_seen) == 1:
            cancel_flipped["value"] = True
        return {"thought": "keep going", "action": "no_such_tool", "params": {}}

    monkeypatch.setattr(rl, "_decide_next_action", _decide)

    outcome = await rl.run_react_loop(
        query="test",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx_factory=_make_ctx_factory_recorder()[0],
        should_cancel=_should_cancel,
        config=rl.ReactConfig(max_iterations=10, deadline_seconds=60),
    )
    # Exactly one iteration's decision LLM call was made — the cancel
    # check on iteration #2 short-circuited before _decide_next_action.
    assert len(iteration_seen) == 1, f"loop should stop on cancel; ran {len(iteration_seen)} decisions"
    # Final scratchpad records the cancel signal.
    thought_texts = [t.text.lower() for t in outcome.scratchpad.thoughts()]
    assert any("cancellation requested" in t for t in thought_texts)


@pytest.mark.asyncio
async def test_react_decision_prompt_pulls_active_context_via_task(monkeypatch):
    """End-to-end check that the active-context dict flows through to
    the decision LLM's prompt. The orchestrator stores it on the task;
    the loop passes it to ``_decide_next_action``; the decision prompt
    text contains the inventory summary.
    """
    from app.assistant import react_loop as rl

    captured: list[list[dict]] = []

    class _FakeLLM:
        cheap_model = "gpt-4o-mini"

        async def complete_structured(self, messages, _model, _schema):
            captured.append(messages)
            return {"thought": "done", "action": "finalize"}

    monkeypatch.setattr("app.adapters.llm.get_llm_adapter", lambda: _FakeLLM())

    factory, _ = _make_ctx_factory_recorder()
    await rl.run_react_loop(
        query="what's in the uploaded files?",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        active_context={"total": 4, "kinds": {"pdf": 2, "note": 2}, "labels": ["a.pdf", "b.md"]},
        ctx_factory=factory,
        config=rl.ReactConfig(max_iterations=1, deadline_seconds=5),
    )
    user_text = captured[0][-1]["content"]
    assert "ACTIVE CONTEXT" in user_text
    assert "total=4" in user_text
    assert "pdf=2" in user_text or "note=2" in user_text


@pytest.mark.asyncio
async def test_react_loop_ctx_factory_cleanup_runs_on_tool_error(monkeypatch):
    """If a tool raises mid-loop, the per-action ctx_factory's
    contextmanager must still close cleanly — no leaked sessions
    when a tool blows up.
    """
    from app.assistant import react_loop as rl

    factory, calls = _make_ctx_factory_recorder()

    async def _decide(**_kw):
        return {"thought": "try", "action": "bad_tool", "params": {}}

    bad_tool = MagicMock()
    bad_tool.input_schema = lambda **kw: kw

    async def _crashes(_ctx, _params):
        raise RuntimeError("boom")

    bad_tool.run = _crashes

    monkeypatch.setattr(rl, "_decide_next_action", _decide)
    monkeypatch.setattr(rl, "get_tool", lambda name: bad_tool if name == "bad_tool" else None)

    await rl.run_react_loop(
        query="test",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx_factory=factory,
        config=rl.ReactConfig(max_iterations=2, deadline_seconds=10),
    )
    # Even though the tool raised, the ctx_factory's __aexit__ must
    # have fired (else we'd be leaking sessions in production).
    assert calls["open_count"] >= 1
    assert calls["exit_count"] == calls["open_count"], (
        "every opened ctx_factory must be closed even when the tool raises"
    )
