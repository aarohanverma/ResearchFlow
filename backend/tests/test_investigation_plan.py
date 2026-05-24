"""Investigation plan + write_todos pseudo-action.

The plan is the model's durable mid-loop task list. These tests pin:

  * Operations contract: add / update / complete / cancel / clear all
    behave as expected, malformed ops are skipped not crashed.
  * Persistence: entries survive across iterations and a follow-up
    write_todos call sees them.
  * Eviction: open work is NEVER evicted; only completed/cancelled
    entries get trimmed when the cap fills.
  * Stuck-in-progress surfacing: entries left in_progress for too
    many iterations land in the synthesizer's agent_notes.
  * Loop integration: the write_todos pseudo-action lands ops on the
    state's plan and the next decision sees them.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from app.assistant.react.investigation_plan import InvestigationPlan, Todo
from app.assistant.react.state import LoopState
from app.assistant.react_loop import ReactConfig
from app.assistant.scratchpad import Scratchpad


# ── Operations contract ─────────────────────────────────────────────────────


def test_add_op_assigns_stable_slug():
    plan = InvestigationPlan()
    notes = plan.apply_operations(
        [{"kind": "add", "text": "compare RAG vs long-context"}],
        iteration=1,
    )
    assert len(plan.todos) == 1
    assert plan.todos[0].id == "t1"
    assert plan.todos[0].text == "compare RAG vs long-context"
    assert plan.todos[0].status == "pending"
    assert any("added t1" in n for n in notes)


def test_add_op_auto_increments_slug_across_calls():
    plan = InvestigationPlan()
    plan.apply_operations([{"kind": "add", "text": "first"}], iteration=1)
    plan.apply_operations([{"kind": "add", "text": "second"}], iteration=2)
    plan.apply_operations([{"kind": "add", "text": "third"}], iteration=2)
    assert [t.id for t in plan.todos] == ["t1", "t2", "t3"]


def test_update_op_modifies_existing_entry():
    plan = InvestigationPlan()
    plan.apply_operations([{"kind": "add", "text": "initial"}], iteration=1)
    plan.apply_operations(
        [{"kind": "update", "id": "t1", "text": "revised", "status": "in_progress"}],
        iteration=2,
    )
    t1 = plan.by_id("t1")
    assert t1 is not None
    assert t1.text == "revised"
    assert t1.status == "in_progress"
    assert t1.iteration == 2


def test_complete_op_marks_done_and_attaches_evidence():
    plan = InvestigationPlan()
    plan.apply_operations([{"kind": "add", "text": "investigate moe"}], iteration=1)
    plan.apply_operations(
        [{"kind": "complete", "id": "t1", "evidence": ["paper-a", "paper-b"]}],
        iteration=2,
    )
    t1 = plan.by_id("t1")
    assert t1.status == "completed"
    assert t1.evidence == ["paper-a", "paper-b"]


def test_complete_op_dedupes_evidence_on_update():
    """Evidence pointers must dedupe so calling complete twice with
    overlapping evidence sets doesn't bloat the list."""
    plan = InvestigationPlan()
    plan.apply_operations(
        [{"kind": "add", "text": "task", "evidence": ["a", "b"]}], iteration=1,
    )
    plan.apply_operations(
        [{"kind": "complete", "id": "t1", "evidence": ["b", "c"]}], iteration=2,
    )
    t1 = plan.by_id("t1")
    assert t1.evidence == ["a", "b", "c"]


def test_cancel_op_marks_cancelled():
    plan = InvestigationPlan()
    plan.apply_operations([{"kind": "add", "text": "x"}], iteration=1)
    plan.apply_operations([{"kind": "cancel", "id": "t1"}], iteration=2)
    assert plan.by_id("t1").status == "cancelled"


def test_clear_op_cancels_all_open_keeps_completed():
    plan = InvestigationPlan()
    plan.apply_operations(
        [
            {"kind": "add", "text": "a"},
            {"kind": "add", "text": "b"},
            {"kind": "add", "text": "c"},
        ],
        iteration=1,
    )
    plan.apply_operations(
        [{"kind": "complete", "id": "t2"}], iteration=2,
    )
    plan.apply_operations([{"kind": "clear"}], iteration=3)
    # t1 + t3 → cancelled, t2 stays completed (audit preserved).
    assert plan.by_id("t1").status == "cancelled"
    assert plan.by_id("t2").status == "completed"
    assert plan.by_id("t3").status == "cancelled"


# ── Robustness: malformed ops skipped, never crash ──────────────────────────


def test_non_list_payload_returns_skip_note():
    plan = InvestigationPlan()
    notes = plan.apply_operations("not a list", iteration=1)  # type: ignore[arg-type]
    assert any("not a list" in n for n in notes)
    assert plan.todos == []


def test_unknown_kind_recorded_but_not_crashed():
    plan = InvestigationPlan()
    notes = plan.apply_operations(
        [{"kind": "delete_everything"}], iteration=1,
    )
    assert any("unknown op kind" in n for n in notes)


def test_update_unknown_id_does_not_crash():
    plan = InvestigationPlan()
    notes = plan.apply_operations(
        [{"kind": "update", "id": "missing", "text": "x"}], iteration=1,
    )
    assert any("unknown id" in n for n in notes)


def test_invalid_status_falls_back_to_pending():
    plan = InvestigationPlan()
    plan.apply_operations(
        [{"kind": "add", "text": "task", "status": "weird"}], iteration=1,
    )
    assert plan.by_id("t1").status == "pending"


# ── Eviction never drops open work ──────────────────────────────────────────


def test_open_work_never_evicted_when_cap_reached():
    """Open todos are durable. The cap is enforced by evicting
    finished entries first; when the model genuinely has > 20 OPEN
    todos, the plan accepts the overflow rather than silently dropping
    the model's stated intentions. The synthesizer's stuck-in-progress
    surfacing handles abuse cases."""
    plan = InvestigationPlan()
    for i in range(25):
        plan.apply_operations(
            [{"kind": "add", "text": f"task {i}"}], iteration=1,
        )
    # All 25 open todos kept — open work never evicted even when over cap.
    open_ = plan.open_todos()
    assert len(open_) == 25
    assert all(t.status == "pending" for t in plan.todos)


def test_eviction_prefers_finished_entries_when_cap_reached():
    """Eviction must drop finished entries before open ones."""
    plan = InvestigationPlan()
    # Add 21 todos, complete the first 10.
    for i in range(21):
        plan.apply_operations(
            [{"kind": "add", "text": f"task {i}"}], iteration=1,
        )
    for i in range(1, 11):
        plan.apply_operations(
            [{"kind": "complete", "id": f"t{i}"}], iteration=2,
        )
    # Now we have 11 open + 10 completed = 21 entries, cap is 20.
    # After eviction, all 11 open entries survive; 9 of the 10
    # completed entries survive (oldest completed got dropped).
    assert len(plan.todos) == 20
    assert len(plan.open_todos()) == 11


# ── Stuck-in-progress surfacing ─────────────────────────────────────────────


def test_stuck_in_progress_surfaces_after_slack_iterations():
    plan = InvestigationPlan()
    plan.apply_operations(
        [{"kind": "add", "text": "investigate x"}], iteration=1,
    )
    plan.apply_operations(
        [{"kind": "update", "id": "t1", "status": "in_progress"}], iteration=2,
    )
    # Two iterations later, still in_progress → stuck.
    stuck = plan.stuck_in_progress(current_iteration=5, slack=2)
    assert len(stuck) == 1
    assert stuck[0].text == "investigate x"


def test_recently_touched_in_progress_not_stuck():
    plan = InvestigationPlan()
    plan.apply_operations(
        [{"kind": "add", "text": "investigate x", "status": "in_progress"}],
        iteration=5,
    )
    stuck = plan.stuck_in_progress(current_iteration=6, slack=2)
    assert stuck == []


# ── Rendering for prompt + summary ──────────────────────────────────────────


def test_render_for_prompt_orders_open_first():
    plan = InvestigationPlan()
    plan.apply_operations(
        [
            {"kind": "add", "text": "open one"},
            {"kind": "add", "text": "open two"},
        ],
        iteration=1,
    )
    plan.apply_operations(
        [{"kind": "complete", "id": "t1"}], iteration=2,
    )
    rendered = plan.render_for_prompt()
    # "open two" (still pending) appears before "open one" (completed)
    # in the prompt rendering.
    assert rendered.index("open two") < rendered.index("open one")


def test_summarize_for_synth_splits_completed_open_cancelled():
    plan = InvestigationPlan()
    plan.apply_operations(
        [
            {"kind": "add", "text": "a"},
            {"kind": "add", "text": "b"},
            {"kind": "add", "text": "c"},
        ],
        iteration=1,
    )
    plan.apply_operations(
        [{"kind": "complete", "id": "t1"}, {"kind": "cancel", "id": "t2"}],
        iteration=2,
    )
    summary = plan.summarize_for_synth()
    assert summary["total"] == 3
    assert summary["completed"] == ["a"]
    assert summary["open"] == ["c"]
    assert summary["cancelled"] == ["b"]


def test_empty_plan_renders_helpful_hint():
    """The empty-plan prompt block hints at how to use the action —
    teaches the model that ``write_todos`` is available without
    forcing it to learn from the action menu alone."""
    plan = InvestigationPlan()
    rendered = plan.render_for_prompt()
    assert "write_todos" in rendered


# ── Loop integration: write_todos lands ops on state.plan ──────────────────


@pytest.mark.asyncio
async def test_write_todos_action_applies_to_state_plan(monkeypatch):
    """End-to-end: the model emits ``action="write_todos"`` with a
    todos payload; the loop applies the ops to ``state.plan`` and
    records an Observation so the model sees the result."""
    from app.assistant import react_loop as rl

    decisions = [
        {
            "thought": "drafting plan",
            "action": "write_todos",
            "todos": [
                {"kind": "add", "text": "find baseline papers"},
                {"kind": "add", "text": "compare against MoE results"},
            ],
        },
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
    # The plan landed on the outcome — two open todos.
    plan_summary = outcome.investigation_plan
    assert plan_summary["total"] == 2
    assert plan_summary["open"] == ["find baseline papers", "compare against MoE results"]
    # Scratchpad recorded the plan update observation.
    obs = [
        e.summary for e in outcome.scratchpad.entries
        if getattr(e, "kind", None) == "observation"
    ]
    assert any("Plan updated" in s for s in obs)


@pytest.mark.asyncio
async def test_write_todos_malformed_payload_recorded(monkeypatch):
    """A write_todos call with a non-list ``todos`` field must record
    a clear observation and continue rather than crashing the loop."""
    from app.assistant import react_loop as rl

    decisions = [
        {"thought": "x", "action": "write_todos", "todos": "this is not a list"},
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
    obs = [
        e.summary for e in outcome.scratchpad.entries
        if getattr(e, "kind", None) == "observation"
    ]
    assert any("missing or non-list" in s for s in obs)
    assert outcome.investigation_plan["total"] == 0
