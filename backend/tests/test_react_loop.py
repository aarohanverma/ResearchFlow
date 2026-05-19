"""Regression tests for the ReAct mid-turn loop + scratchpad + provenance.

Locks in the contracts that make RA depth-driven and inspectable:

* Scratchpad round-trips through JSON without losing entries or types.
* The loop terminates when the model emits ``finalize``.
* The loop terminates on its iteration cap even if the model keeps
  picking new actions — no path runs forever.
* Memory-write / memory-delete tools are refused inside the loop so
  durable memory consolidation stays on the post-turn pass (one
  writer per tier).
* Provenance extraction maps inline ``[N]`` markers to the right
  papers, including ``[N,M]`` and ``[N-M]`` forms.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Scratchpad ────────────────────────────────────────────────────────────────


def test_scratchpad_round_trips_through_json():
    from app.assistant.scratchpad import Scratchpad

    pad = Scratchpad()
    pad.next_iteration()
    pad.think("First reasoning step")
    pad.act("deep_search", {"query": "agent memory"}, rationale="initial broad scan")
    pad.observe("deep_search", "Returned 5 papers", "deep_search", error=None)
    pad.next_iteration()
    pad.think("Need verification")
    pad.act("wikipedia", {"topic": "ReAct"}, rationale="background context")
    pad.observe("wikipedia", "Found ReAct paper", "wikipedia", error=None)
    pad.critique(groundedness=0.8, completeness=0.6, memory_faithfulness=1.0, issues=[], verdict="ship")
    pad.provenance("The agent loop works", ["p1", "p2"], "[1,2]")
    pad.finish()

    serialised = pad.to_dict()
    revived = Scratchpad.from_dict(serialised)

    assert len(revived.entries) == len(pad.entries)
    assert revived.iteration == pad.iteration
    assert revived.finished_at is not None
    # Types survive the round-trip.
    kinds = [e.kind for e in revived.entries]
    assert kinds == ["thought", "action", "observation", "thought", "action", "observation", "critique", "provenance"]


def test_scratchpad_render_for_prompt_is_compact_and_safe():
    from app.assistant.scratchpad import Scratchpad

    pad = Scratchpad()
    pad.next_iteration()
    pad.think("x" * 5000)
    pad.act("deep_search", {"query": "y" * 5000}, rationale="z" * 5000)

    rendered = pad.render_for_prompt(max_entries=10)
    # Each entry is truncated for prompt frugality.
    assert len(rendered) < 4000


# ── ReAct loop control flow ───────────────────────────────────────────────────


def _make_ctx():
    """Build a minimal ToolContext-shaped object for tool execution."""
    import uuid as _uuid
    from app.assistant.tools.base import ToolContext

    async def _noop(*_a, **_k): return None
    async def _never_cancel(): return False

    return ToolContext(
        user_id=_uuid.uuid4(),
        session_id=_uuid.uuid4(),
        namespace_key="cs.AI",
        namespace_keys=["cs.AI"],
        orientation="both",
        expertise_level="practitioner",
        job_id="test-job",
        parent_message_id=_uuid.uuid4(),
        db=MagicMock(),
        should_cancel=_never_cancel,
        emit_progress=_noop,
    )


@pytest.mark.asyncio
async def test_react_loop_finalizes_on_first_decision(monkeypatch):
    """If the model says 'finalize' immediately, the loop ends cleanly."""
    from app.assistant import react_loop as rl

    # Patch the decision step to always return 'finalize'.
    async def _decide(**_kw):
        return {"thought": "Initial results are sufficient.", "action": "finalize"}

    monkeypatch.setattr(rl, "_decide_next_action", _decide)

    outcome = await rl.run_react_loop(
        query="why does ReAct work?",
        initial_plan_actions=["Deep Search"],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx=_make_ctx(),
        config=rl.ReactConfig(max_iterations=4, deadline_seconds=10),
    )
    assert outcome.completed_normally is True
    assert outcome.iterations == 1
    assert outcome.new_results == {}
    # First iteration's THOUGHT is recorded on the pad.
    thoughts = outcome.scratchpad.thoughts()
    assert any("sufficient" in t.text.lower() for t in thoughts)


@pytest.mark.asyncio
async def test_react_loop_respects_iteration_cap(monkeypatch):
    """Model that never says finalize must still terminate at the cap."""
    from app.assistant import react_loop as rl

    call_count = {"n": 0}

    async def _decide(**_kw):
        call_count["n"] += 1
        # Always pick an unknown tool; the loop treats it as a no-op
        # observation and continues.
        return {"thought": f"iteration {call_count['n']}", "action": "no_such_tool", "params": {}}

    monkeypatch.setattr(rl, "_decide_next_action", _decide)

    outcome = await rl.run_react_loop(
        query="test cap",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx=_make_ctx(),
        config=rl.ReactConfig(max_iterations=3, deadline_seconds=10),
    )
    assert outcome.iterations == 3, f"expected 3 iterations, got {outcome.iterations}"
    assert outcome.completed_normally is False


@pytest.mark.asyncio
async def test_react_loop_refuses_disallowed_tools(monkeypatch):
    """memory_write / memory_delete must be silently rejected from the loop.

    Durable consolidation belongs to the post-turn auto-memory pass —
    letting the loop write durable memory mid-turn invites premature
    commits and double-writes.
    """
    from app.assistant import react_loop as rl

    calls = []

    async def _decide(**kw):
        # First call: pick memory_write. Second call: finalize.
        calls.append(kw)
        if len(calls) == 1:
            return {"thought": "I want to save this fact", "action": "memory_write", "params": {"value": "x"}}
        return {"thought": "stopping", "action": "finalize"}

    monkeypatch.setattr(rl, "_decide_next_action", _decide)

    outcome = await rl.run_react_loop(
        query="test refusal",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx=_make_ctx(),
        config=rl.ReactConfig(max_iterations=3, deadline_seconds=10),
    )
    # No tool result for memory_write should be present — the loop
    # refused to call it.
    assert "memory_write" not in outcome.new_results
    # The pad should record that we declined.
    thoughts_text = " ".join(t.text.lower() for t in outcome.scratchpad.thoughts())
    assert "memory_write" in thoughts_text or "not callable" in thoughts_text or "finalize" in thoughts_text


# ── Provenance extraction ─────────────────────────────────────────────────────


def test_extract_provenance_handles_simple_marker():
    from app.assistant.orchestrator import _extract_provenance

    papers = [{"paper_id": "p1", "title": "A"}, {"paper_id": "p2", "title": "B"}]
    answer = "The model achieves 92% accuracy on the benchmark [1]. Earlier work used a simpler baseline [2]."
    prov = _extract_provenance(answer, papers)
    assert len(prov) == 2
    by_marker = {p["marker"]: p for p in prov}
    assert by_marker["[1]"]["sources"] == ["p1"]
    assert by_marker["[2]"]["sources"] == ["p2"]
    # Claim span should contain the citing sentence (not the entire answer).
    assert "92%" in by_marker["[1]"]["claim_span"]
    assert "baseline" in by_marker["[2]"]["claim_span"]


def test_extract_provenance_handles_compound_markers():
    from app.assistant.orchestrator import _extract_provenance

    papers = [
        {"paper_id": "p1", "title": "A"},
        {"paper_id": "p2", "title": "B"},
        {"paper_id": "p3", "title": "C"},
        {"paper_id": "p4", "title": "D"},
    ]
    answer = "Multiple groups confirm this finding [1, 2, 3]. The bound is tight [2-4]."
    prov = _extract_provenance(answer, papers)
    by_marker = {p["marker"]: p for p in prov}
    assert "[1, 2, 3]" in by_marker or "[1,2,3]" in by_marker  # tolerate either spacing
    multi = next(p for p in prov if p["marker"] in {"[1, 2, 3]", "[1,2,3]"})
    assert multi["sources"] == ["p1", "p2", "p3"]
    rng = next(p for p in prov if p["marker"] in {"[2-4]", "[2 - 4]"})
    assert rng["sources"] == ["p2", "p3", "p4"]


def test_extract_provenance_skips_out_of_range_markers():
    from app.assistant.orchestrator import _extract_provenance

    papers = [{"paper_id": "p1", "title": "A"}]
    # Marker [5] references a paper that doesn't exist — must be dropped,
    # not silently mapped to a wrong source.
    answer = "Some claim [5]."
    prov = _extract_provenance(answer, papers)
    assert prov == []


def test_extract_provenance_handles_empty_inputs():
    from app.assistant.orchestrator import _extract_provenance

    assert _extract_provenance("", []) == []
    assert _extract_provenance("no citations here", [{"paper_id": "p1", "title": "A"}]) == []
