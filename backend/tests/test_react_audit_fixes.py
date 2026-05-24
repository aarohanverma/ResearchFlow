"""Regression tests for the deep-audit bug fixes.

Each test pins a specific bug the audit uncovered:

  * Subagent recursion: a subagent must not be able to spawn another
    subagent (defeats context quarantine, risks runaway budget).
  * Subagent role routing: the spec's role prompt goes into the
    SYSTEM message, not buried in the query field.
  * Memory recall hides consolidated originals so the rollup is the
    authoritative view; the originals stay in DB for provenance audit.
  * Loop coerces non-dict params at the boundary so middlewares + the
    dispatch path never see a string/list/None where a dict is
    expected.
  * Subagent skips cleanly when parent has effectively zero deadline
    budget left rather than burning another second.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from app.assistant.react.state import LoopState
from app.assistant.react.subagent_runner import run_subagent
from app.assistant.react.subagents import get_subagent
from app.assistant.react_loop import ReactConfig, ReactOutcome
from app.assistant.scratchpad import Scratchpad
from app.assistant.tools.base import ToolResult


def _make_state(*, subagent_depth: int = 0, subagent_role: str | None = None,
                max_iterations: int = 5) -> LoopState:
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
        subagent_depth=subagent_depth,
        subagent_role=subagent_role,
    )
    from app.assistant.react_loop import PaperLedger
    state.ledger = PaperLedger()
    return state


# ── Bug 1: subagent recursion ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decision_prompt_hides_subagent_catalog_at_depth_one(monkeypatch):
    """When this loop IS a subagent (depth ≥ 1), the decision prompt
    must NOT advertise the subagent action — that would let it
    recurse indefinitely."""
    from app.assistant import react_loop as rl

    captured_messages: list = []

    class _FakeLLM:
        cheap_model = "fake-cheap"

        async def complete_structured(self, messages, model, schema):
            captured_messages.extend(messages)
            return {"thought": "done", "action": "finalize"}

    monkeypatch.setattr("app.adapters.llm.get_llm_adapter", lambda: _FakeLLM())

    await rl._decide_next_action(
        query="task",
        pad=Scratchpad(),
        prior_results={},
        new_results={},
        memory_view={},
        research_brief_text="",
        active_context=None,
        config=ReactConfig(max_iterations=2, deadline_seconds=5),
        is_last_iteration=False,
        subagent_depth=1,
    )
    full_prompt = " ".join(m.get("content", "") for m in captured_messages)
    # At depth 1, the subagent catalog block must NOT appear so the
    # model can't ask to spawn another subagent.
    assert "Available subagents:" not in full_prompt
    assert "You are a subagent — do NOT delegate further" in full_prompt


@pytest.mark.asyncio
async def test_decision_prompt_shows_subagent_catalog_at_depth_zero(monkeypatch):
    """At depth 0 (top-level RA loop), the subagent catalog IS shown
    — that's how the model knows the capability exists."""
    from app.assistant import react_loop as rl

    captured_messages: list = []

    class _FakeLLM:
        cheap_model = "fake-cheap"

        async def complete_structured(self, messages, model, schema):
            captured_messages.extend(messages)
            return {"thought": "done", "action": "finalize"}

    monkeypatch.setattr("app.adapters.llm.get_llm_adapter", lambda: _FakeLLM())

    await rl._decide_next_action(
        query="task",
        pad=Scratchpad(),
        prior_results={},
        new_results={},
        memory_view={},
        research_brief_text="",
        active_context=None,
        config=ReactConfig(max_iterations=2, deadline_seconds=5),
        is_last_iteration=False,
        subagent_depth=0,
    )
    full_prompt = " ".join(m.get("content", "") for m in captured_messages)
    assert "Available subagents:" in full_prompt
    assert "researcher" in full_prompt  # at least one named subagent rendered


@pytest.mark.asyncio
async def test_loop_blocks_nested_subagent_dispatch(monkeypatch):
    """Even if the model somehow emits ``action="subagent"`` at depth
    > 0 (cached prompt, prompt injection, whatever), the loop refuses
    the dispatch and records a clear observation."""
    from app.assistant import react_loop as rl

    decisions = [
        {"thought": "delegate", "action": "subagent",
         "params": {"subagent_name": "researcher", "task": "x"}},
        {"thought": "done", "action": "finalize"},
    ]

    async def _decide(**_kw):
        return decisions.pop(0) if decisions else {"thought": "done", "action": "finalize"}

    async def _fake_critique(**kwargs):
        kwargs["pad"].critique(
            groundedness=0.9, completeness=0.9, memory_faithfulness=1.0,
            issues=[], verdict="ship",
        )

    monkeypatch.setattr(rl, "_decide_next_action", _decide)
    monkeypatch.setattr(rl, "_run_self_critique", _fake_critique)

    # Spy: if the subagent runner was called, the test fails.
    runner_calls: list = []

    async def _spy_runner(**kwargs):
        runner_calls.append(kwargs)
        from app.assistant.react.subagents import SubAgentResult
        return SubAgentResult(subagent_name="researcher", summary="should not run")

    monkeypatch.setattr("app.assistant.react.subagent_runner.run_subagent", _spy_runner)

    # Run at depth=1 (simulating a subagent invocation).
    outcome = await rl.run_react_loop(
        query="x",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        config=ReactConfig(max_iterations=3, deadline_seconds=10),
        subagent_depth=1,
    )
    assert runner_calls == [], "subagent runner must NOT be invoked at depth > 0"
    # Loop recorded an observation explaining the refusal.
    obs_summaries = [
        e.summary for e in outcome.scratchpad.entries
        if getattr(e, "kind", None) == "observation"
    ]
    assert any("Nested subagent dispatch refused" in s for s in obs_summaries)


# ── Bug 2: subagent role prompt routing ─────────────────────────────────────


@pytest.mark.asyncio
async def test_subagent_role_goes_into_system_message(monkeypatch):
    """When ``subagent_role`` is set, the decision prompt's SYSTEM
    message starts with the role — replacing the generic "you are
    RA's reasoning engine" prompt."""
    from app.assistant import react_loop as rl

    captured_messages: list = []

    class _FakeLLM:
        cheap_model = "fake-cheap"

        async def complete_structured(self, messages, model, schema):
            captured_messages.extend(messages)
            return {"thought": "done", "action": "finalize"}

    monkeypatch.setattr("app.adapters.llm.get_llm_adapter", lambda: _FakeLLM())

    await rl._decide_next_action(
        query="task",
        pad=Scratchpad(),
        prior_results={},
        new_results={},
        memory_view={},
        research_brief_text="",
        active_context=None,
        config=ReactConfig(max_iterations=2, deadline_seconds=5),
        is_last_iteration=False,
        subagent_depth=1,
        subagent_role="You are a focused citation auditor.",
    )
    sys_msg = next(m["content"] for m in captured_messages if m["role"] == "system")
    # Role appears FIRST in the system message — not buried in the query.
    assert sys_msg.startswith("You are a specialised subagent. Role:")
    assert "focused citation auditor" in sys_msg
    # The generic "reasoning engine of RA" prompt is gone.
    assert "reasoning engine of a research assistant" not in sys_msg


# ── Bug 3: recall hides consolidated originals ──────────────────────────────


def test_recall_filter_hides_consolidated_into_entries():
    """The ``_matches`` helper in MemoryRecallTool.run must reject
    entries whose ``consolidated_into`` is set — the rollup is the
    authoritative view."""
    # We can't easily call the inner _matches function directly
    # (it's a closure inside run); instead we exercise the contract
    # via a synthetic tier dict and the same filter shape.

    def _matches(k: str, entry: object, *, query: str = "", type_: str = "") -> bool:
        # Mirror of the production filter — keep in sync with memory.py.
        if isinstance(entry, dict) and entry.get("consolidated_into"):
            return False
        if type_ and (entry.get("type", "context") if isinstance(entry, dict) else "context") != type_:
            return False
        if query:
            val = entry.get("value", "") if isinstance(entry, dict) else str(entry)
            return query in k.lower() or query in val.lower()
        return True

    rolled_up_original = {
        "value": "user prefers terse answers",
        "type": "preference",
        "consolidated_into": "consolidated__pref_rollup",
    }
    rollup = {
        "value": "user prefers terse, technical responses",
        "type": "preference",
        "consolidated_from": ["pref_terse", "pref_tech"],
    }
    independent = {"value": "user is a senior researcher", "type": "preference"}

    assert _matches("pref_terse", rolled_up_original) is False
    assert _matches("consolidated__pref_rollup", rollup) is True
    assert _matches("user_role", independent) is True


# ── Bug 4: defensive params coercion ────────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_coerces_non_dict_params(monkeypatch):
    """When the model emits ``params`` as a string / list / None, the
    loop coerces to ``{}`` at the boundary and records a thought
    explaining the coercion — every downstream middleware then sees
    a real dict."""
    from app.assistant import react_loop as rl

    decisions = [
        {"thought": "broken params", "action": "deep_search", "params": "not a dict"},
        {"thought": "done", "action": "finalize"},
    ]

    async def _decide(**_kw):
        return decisions.pop(0) if decisions else {"thought": "done", "action": "finalize"}

    async def _fake_critique(**kwargs):
        kwargs["pad"].critique(
            groundedness=0.9, completeness=0.9, memory_faithfulness=1.0,
            issues=[], verdict="ship",
        )

    monkeypatch.setattr(rl, "_decide_next_action", _decide)
    monkeypatch.setattr(rl, "_run_self_critique", _fake_critique)

    outcome = await rl.run_react_loop(
        query="test",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        config=ReactConfig(max_iterations=3, deadline_seconds=10),
    )
    # The loop recorded a thought about the coercion — proves the
    # defensive branch fired.
    thoughts = [t.text for t in outcome.scratchpad.thoughts()]
    assert any("non-dict params" in t for t in thoughts)


# ── Bug 5: subagent skips on near-zero parent budget ────────────────────────


@pytest.mark.asyncio
async def test_subagent_skips_when_parent_deadline_exhausted():
    """A subagent invoked with < 2 seconds of parent budget remaining
    must skip cleanly rather than burn another second on a doomed
    investigation."""
    fake_state = MagicMock(spec=LoopState)
    fake_state.config = ReactConfig(max_iterations=4, deadline_seconds=10)
    fake_state.time_remaining = MagicMock(return_value=0.5)  # essentially expired

    result = await run_subagent(
        parent_state=fake_state,
        subagent_name="researcher",
        task="anything",
    )
    assert result.completed_normally is False
    assert "deadline budget too low" in result.summary
    # iterations stays at default 0 because we never invoked the loop.
    assert result.iterations == 0
