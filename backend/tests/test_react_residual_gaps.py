"""Regression tests for the two residual gaps closed in this pass.

Gap A: scratchpad insights reach the synthesizer.
* ``_distill_agent_notes`` produces ``None`` when ReAct didn't run.
* It surfaces the latest critique's verdict + scores + issues.
* ``thin_evidence`` flag fires when iterations > 0 AND total papers < 2,
  OR when the latest critique verdict is "revise" — even if the model
  itself didn't flag it, the deterministic heuristic does.
* ``_render_agent_notes`` produces a compact ``<agent_notes>`` block,
  is silent when there's nothing to say, and surfaces the THIN-evidence
  instruction in plain text the synthesizer prompt can act on.

Gap B: ReAct decision step sees the active context inventory.
* ``run_react_loop`` accepts an ``active_context`` kwarg and forwards
  it to ``_decide_next_action``.
* The block rendered into the LLM prompt mentions attachment counts
  and labels when present, and degrades to a clear "(none)" line when
  the session has no attachments — so the prompt is never confusing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── Agent notes distillation ──────────────────────────────────────────────────


def test_distill_returns_none_when_react_did_not_run():
    from app.assistant.orchestrator import _distill_agent_notes

    assert _distill_agent_notes(scratchpad=None, iterations=0, papers=[], arxiv_results=[]) is None


def test_distill_surfaces_latest_critique():
    from app.assistant.orchestrator import _distill_agent_notes
    from app.assistant.scratchpad import Scratchpad

    pad = Scratchpad()
    pad.next_iteration()
    pad.critique(
        groundedness=0.4, completeness=0.5, memory_faithfulness=1.0,
        issues=["thin coverage on benchmark X"], verdict="revise",
    )
    notes = _distill_agent_notes(
        scratchpad=pad, iterations=2,
        papers=[{"paper_id": "p1"}], arxiv_results=[],
    )
    assert notes is not None
    assert notes["iterations"] == 2
    assert notes["critique"]["verdict"] == "revise"
    assert notes["critique"]["groundedness"] == 0.4
    assert "thin coverage on benchmark X" in notes["critique"]["issues"]
    # Thin: 1 paper + 0 arxiv = 1 evidence item, iterations > 0 → thin.
    assert notes["thin_evidence"] is True


def test_distill_thin_flag_fires_on_revise_verdict_even_with_evidence():
    from app.assistant.orchestrator import _distill_agent_notes
    from app.assistant.scratchpad import Scratchpad

    pad = Scratchpad()
    pad.critique(
        groundedness=0.7, completeness=0.5, memory_faithfulness=1.0,
        issues=[], verdict="revise",
    )
    notes = _distill_agent_notes(
        scratchpad=pad, iterations=1,
        papers=[{"paper_id": str(i)} for i in range(5)],  # plenty of papers
        arxiv_results=[],
    )
    assert notes["thin_evidence"] is True


def test_distill_no_thin_flag_when_critique_ships_and_evidence_is_solid():
    from app.assistant.orchestrator import _distill_agent_notes
    from app.assistant.scratchpad import Scratchpad

    pad = Scratchpad()
    pad.critique(
        groundedness=0.9, completeness=0.9, memory_faithfulness=1.0,
        issues=[], verdict="ship",
    )
    notes = _distill_agent_notes(
        scratchpad=pad, iterations=1,
        papers=[{"paper_id": str(i)} for i in range(5)],
        arxiv_results=[],
    )
    assert notes is not None
    assert notes.get("thin_evidence") is not True


def test_render_agent_notes_empty_input_yields_empty_string():
    from app.assistant.synthesizer import _render_agent_notes

    assert _render_agent_notes(None) == ""
    assert _render_agent_notes({}) == ""


def test_render_agent_notes_emits_thin_evidence_instruction():
    from app.assistant.synthesizer import _render_agent_notes

    block = _render_agent_notes({
        "iterations": 3,
        "critique": {"verdict": "revise", "groundedness": 0.3, "completeness": 0.4, "issues": ["X", "Y"]},
        "thin_evidence": True,
    })
    assert "<agent_notes>" in block
    assert "</agent_notes>" in block
    assert "THIN" in block
    assert "honest about uncertainty" in block.lower()
    assert "verdict: revise" in block.lower()


def test_render_agent_notes_omits_thin_block_when_solid():
    from app.assistant.synthesizer import _render_agent_notes

    block = _render_agent_notes({
        "iterations": 1,
        "critique": {"verdict": "ship", "groundedness": 0.9, "completeness": 0.9, "issues": []},
    })
    assert "<agent_notes>" in block
    assert "thin" not in block.lower()


# ── Active context plumbing ───────────────────────────────────────────────────


def _make_ctx():
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
async def test_react_loop_forwards_active_context_to_decision(monkeypatch):
    """The decision step must receive the active-context dict so its prompt
    can mention attached files. The loop calls ``_decide_next_action``
    with ``active_context=...``; we capture the kwargs to verify.
    """
    from app.assistant import react_loop as rl

    captured: list[dict] = []

    async def _decide(**kw):
        captured.append(kw)
        return {"thought": "stop", "action": "finalize"}

    monkeypatch.setattr(rl, "_decide_next_action", _decide)

    ac = {"total": 3, "kinds": {"note": 2, "url": 1}, "labels": ["my notes.md", "https://arxiv.org/abs/...", "background"]}
    await rl.run_react_loop(
        query="any query",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        active_context=ac,
        ctx=_make_ctx(),
        config=rl.ReactConfig(max_iterations=1, deadline_seconds=5),
    )
    assert captured, "_decide_next_action must have been called"
    assert captured[0].get("active_context") is ac, "active_context was not forwarded"


@pytest.mark.asyncio
async def test_react_decision_prompt_mentions_attachments_when_present(monkeypatch):
    """The user-message text passed to the LLM must include the
    ACTIVE CONTEXT block with counts when attachments exist."""
    from app.assistant import react_loop as rl

    captured_messages: list[list[dict]] = []

    class _FakeLLM:
        cheap_model = "gpt-4o-mini"

        async def complete_structured(self, messages, _model, _schema):
            captured_messages.append(messages)
            return {"thought": "ok", "action": "finalize"}

    monkeypatch.setattr("app.adapters.llm.get_llm_adapter", lambda: _FakeLLM())

    ac = {"total": 2, "kinds": {"note": 1, "pdf": 1}, "labels": ["sketch.md", "paper.pdf"]}
    await rl.run_react_loop(
        query="what do these documents say?",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        active_context=ac,
        ctx=_make_ctx(),
        config=rl.ReactConfig(max_iterations=1, deadline_seconds=5),
    )
    assert captured_messages, "decision LLM must have been called"
    user_text = captured_messages[0][-1]["content"]
    assert "ACTIVE CONTEXT" in user_text
    assert "total=2" in user_text
    assert "note=1" in user_text or "pdf=1" in user_text


@pytest.mark.asyncio
async def test_react_decision_prompt_says_none_when_no_attachments(monkeypatch):
    from app.assistant import react_loop as rl

    captured_messages: list[list[dict]] = []

    class _FakeLLM:
        cheap_model = "gpt-4o-mini"

        async def complete_structured(self, messages, _model, _schema):
            captured_messages.append(messages)
            return {"thought": "ok", "action": "finalize"}

    monkeypatch.setattr("app.adapters.llm.get_llm_adapter", lambda: _FakeLLM())

    await rl.run_react_loop(
        query="test",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        active_context=None,
        ctx=_make_ctx(),
        config=rl.ReactConfig(max_iterations=1, deadline_seconds=5),
    )
    user_text = captured_messages[0][-1]["content"]
    assert "ACTIVE CONTEXT" in user_text
    assert "(none" in user_text.lower()
