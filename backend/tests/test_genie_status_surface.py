"""Tests for the synthesizer's Genie-status directive.

The orchestrator surfaces the genie_synthesize tool's actual outcome
(done / running / queued / timeout / failed / cancelled) onto
``agent_notes['genie_status']``. The synthesizer's ``_render_agent_notes``
turns that into a clear directive so the answer narrates Genie honestly
instead of describing a still-running synthesis as if it had completed.
"""

from __future__ import annotations

from app.assistant.orchestrator import Orchestrator
from app.assistant.synthesizer import _render_agent_notes
from app.assistant.tools.base import ToolResult


# ── Orchestrator surfacer ───────────────────────────────────────────────────


def test_genie_status_from_results_returns_none_when_not_run():
    assert Orchestrator._genie_status_from_results({}) is None


def test_genie_status_from_results_captures_done_with_capsule():
    results = {"genie_synthesize": ToolResult(
        output={
            "genie_session_id": "sess-1",
            "seed_count": 3,
            "status": "done",
            "capsule": {"id": "cap-1", "title": "A novel hybrid approach"},
        },
        summary="genie ok",
    )}
    status = Orchestrator._genie_status_from_results(results)
    assert status == {
        "session_id": "sess-1",
        "status": "done",
        "seed_count": 3,
        "capsule_id": "cap-1",
        "capsule_title": "A novel hybrid approach",
    }


def test_genie_status_from_results_captures_timeout_without_capsule():
    results = {"genie_synthesize": ToolResult(
        output={
            "genie_session_id": "sess-2",
            "seed_count": 3,
            "status": "timeout",
            "capsule": None,
        },
        summary="genie still running",
    )}
    status = Orchestrator._genie_status_from_results(results)
    assert status["status"] == "timeout"
    assert status["capsule_title"] is None
    assert status["capsule_id"] is None


# ── Synthesizer agent_notes rendering ───────────────────────────────────────


def test_render_agent_notes_emits_running_directive_for_timeout():
    notes = {"genie_status": {"status": "timeout", "capsule_title": None}}
    rendered = _render_agent_notes(notes)
    assert "GENIE STATUS: timeout" in rendered
    assert "STILL being synthesized" in rendered
    assert "DO NOT describe it as completed" in rendered


def test_render_agent_notes_emits_done_directive_with_title():
    notes = {"genie_status": {"status": "done", "capsule_title": "A novel hybrid"}}
    rendered = _render_agent_notes(notes)
    assert "GENIE STATUS: done" in rendered
    assert "A novel hybrid" in rendered


def test_render_agent_notes_emits_failed_directive():
    notes = {"genie_status": {"status": "failed", "capsule_title": None}}
    rendered = _render_agent_notes(notes)
    assert "GENIE STATUS: failed" in rendered
    assert "did NOT produce" in rendered


def test_render_agent_notes_empty_when_no_genie_status():
    """A turn that didn't invoke Genie must not get a Genie directive
    in the agent_notes block."""
    assert "GENIE STATUS" not in _render_agent_notes({})
    assert "GENIE STATUS" not in _render_agent_notes({"iterations": 2})
