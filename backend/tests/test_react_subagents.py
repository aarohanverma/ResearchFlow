"""Subagent registry + nested dispatch tests.

The subagent system is the real context-quarantine primitive — these
tests pin its contract:

  * Registry lookups are predictable + spec immutable.
  * Tool scoping: a subagent's allowed_tools is the catalog cap; the
    nested loop never sees tools outside that set.
  * Structured response: the spec's response_schema shapes the output
    the parent receives.
  * Tool-hide prefix integration: a parent's restricted catalog
    propagates correctly to the subagent.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from app.assistant.react.subagent_runner import _distill_outcome, _hide_tools, run_subagent
from app.assistant.react.subagents import (
    SUBAGENT_REGISTRY,
    SubAgentResult,
    SubAgentSpec,
    describe_subagents_for_prompt,
    get_subagent,
)
from app.assistant.scratchpad import Scratchpad
from app.assistant.tools.base import ToolResult


# ── Registry ────────────────────────────────────────────────────────────────


def test_registry_has_expected_research_subagents():
    """Specific subagents the loop's prompt references must exist."""
    for name in ("researcher", "comparator", "critic", "baseline_finder", "contradiction_hunter"):
        spec = get_subagent(name)
        assert spec is not None, f"missing subagent: {name}"
        assert isinstance(spec, SubAgentSpec)


def test_unknown_subagent_returns_none():
    assert get_subagent("does_not_exist") is None


def test_subagent_specs_are_immutable_frozen_dataclasses():
    """Specs must be hashable + immutable so the registry can be shared
    across concurrent loops without defensive-copying."""
    spec = get_subagent("researcher")
    with pytest.raises(Exception):  # frozen=True → FrozenInstanceError or AttributeError
        spec.max_iterations = 99


def test_subagent_catalog_renders_for_prompt():
    rendered = describe_subagents_for_prompt()
    assert "researcher" in rendered
    assert "comparator" in rendered
    # Must be ``-`` bullets so it embeds cleanly into the existing
    # action-list block of the decision prompt.
    assert rendered.lstrip().startswith("- ")


# ── Tool scoping ────────────────────────────────────────────────────────────


def test_hide_tools_uses_tool_prefix():
    """The tool-hide mechanism encodes a per-call blocklist into the
    ``disabled_features`` set via ``tool:<name>`` keys. The catalog
    decoder recognises the prefix; feature-flag handlers don't."""
    existing = {"graph_enabled"}
    hidden = _hide_tools(existing, {"deep_search", "arxiv_import"})
    assert "graph_enabled" in hidden
    assert "tool:deep_search" in hidden
    assert "tool:arxiv_import" in hidden


def test_tool_hide_prefix_filters_catalog():
    """Integration check: a ``tool:<name>`` flag in disabled_features
    hides that tool from describe_for_planner output."""
    from app.assistant.tools.registry import describe_for_planner, register_tool
    from pydantic import BaseModel

    class _In(BaseModel):
        q: str = ""

    class _Out(BaseModel):
        ok: bool = True

    fake = MagicMock()
    fake.name = "tst_hideme"
    fake.summary = "test"
    fake.cost_class = "cheap"
    fake.side_effects = False
    fake.cancellable = False
    fake.streamable = False
    fake.input_schema = _In
    fake.output_schema = _Out
    register_tool(fake)

    # Visible by default.
    catalog = describe_for_planner()
    assert any(t["name"] == "tst_hideme" for t in catalog)

    # Hidden when ``tool:<name>`` is in disabled_features.
    hidden_catalog = describe_for_planner(disabled_features={"tool:tst_hideme"})
    assert all(t["name"] != "tst_hideme" for t in hidden_catalog)


# ── SubAgentResult shape ────────────────────────────────────────────────────


def test_subagent_result_projects_to_tool_result_with_papers():
    """A subagent that surfaced paper IDs must produce a ToolResult
    the parent's paper_ledger middleware can pick up."""
    result = SubAgentResult(
        subagent_name="researcher",
        summary="Found 3 papers on retrieval-augmented generation.",
        paper_ids_surfaced=["p1", "p2", "p3"],
        iterations=3,
        completed_normally=True,
    )
    tr = result.to_tool_result()
    assert tr.summary.startswith("Found 3 papers")
    # Mirror retrieval-tool shape so paper_ledger middleware reuses
    # its existing add_from_result extraction logic.
    assert "papers" in tr.output
    assert len(tr.output["papers"]) == 3
    assert tr.output["papers"][0]["paper_id"] == "p1"


def test_subagent_result_without_papers_omits_papers_key():
    result = SubAgentResult(subagent_name="critic", summary="No counter-evidence found.")
    tr = result.to_tool_result()
    assert "papers" not in tr.output


# ── Nested dispatch (end-to-end with mocked loop) ──────────────────────────


@pytest.mark.asyncio
async def test_run_subagent_returns_unknown_for_typo():
    """Typo'd subagent names must produce a clear unknown-name result
    rather than crash — the model can then retry with the right name."""
    from app.assistant.react.state import LoopState
    from app.assistant.react_loop import ReactConfig

    fake_state = MagicMock(spec=LoopState)
    fake_state.config = ReactConfig(max_iterations=4, deadline_seconds=10)
    fake_state.time_remaining = MagicMock(return_value=10.0)

    result = await run_subagent(
        parent_state=fake_state,
        subagent_name="researcherr_typo",
        task="anything",
    )
    assert result.completed_normally is False
    assert "Unknown subagent" in result.summary
    assert "researcher" in result.summary  # mentions available names


@pytest.mark.asyncio
async def test_run_subagent_invokes_nested_loop(monkeypatch):
    """A known subagent must invoke run_react_loop with the spec's
    role prompt baked into the focused query + the spec's iteration
    cap."""
    from app.assistant.react.state import LoopState
    from app.assistant.react_loop import ReactConfig, ReactOutcome

    captured: dict = {}

    async def _fake_loop(**kwargs):
        captured.update(kwargs)
        return ReactOutcome(
            scratchpad=Scratchpad(),
            new_results={
                "deep_search": ToolResult(
                    output={"papers": [{"paper_id": "p1", "title": "X"}]},
                    summary="found 1",
                ),
            },
            completed_normally=True,
            iterations=2,
            successful_retrievals=1,
            paper_ledger_size=1,
        )

    import app.assistant.react_loop as rl_mod
    monkeypatch.setattr(rl_mod, "run_react_loop", _fake_loop)

    fake_state = MagicMock(spec=LoopState)
    fake_state.config = ReactConfig(max_iterations=8, deadline_seconds=90)
    fake_state.time_remaining = MagicMock(return_value=60.0)
    fake_state.ctx_factory = None
    fake_state.ctx = None
    fake_state.should_cancel = None
    # MagicMock auto-fakes ``subagent_depth`` as a mock object whose
    # ``__int__`` returns 1; pin it to 0 so the runner's depth+1
    # calculation produces the expected 1 (not 2).
    fake_state.subagent_depth = 0

    result = await run_subagent(
        parent_state=fake_state,
        subagent_name="researcher",
        task="find papers on retrieval-augmented generation",
    )
    # Role-prompt routing: the spec's role goes into the SYSTEM
    # message (via subagent_role), NOT into the query. The query
    # carries only the task so the model sees one coherent role
    # instruction in the system prompt.
    assert captured["query"] == "find papers on retrieval-augmented generation"
    assert captured.get("subagent_role")
    assert "focused literature researcher" in captured["subagent_role"]
    # Recursion guard: nested loop's depth is parent+1.
    assert captured.get("subagent_depth") == 1
    # Iteration cap respects spec (4 for researcher, never exceeds
    # parent's budget).
    assert captured["config"].max_iterations <= 4
    # Result carries the surfaced paper ID + the iteration count.
    assert result.paper_ids_surfaced == ["p1"]
    assert result.iterations == 2
    assert result.completed_normally is True


@pytest.mark.asyncio
async def test_subagent_distillation_extracts_final_thought():
    """When the nested loop produced a meaningful Thought entry near
    the end (the model's "I have enough; finalizing" reasoning), the
    distillation surface it in the summary."""
    from app.assistant.react_loop import ReactOutcome

    pad = Scratchpad()
    pad.think("Auto-repaired params for deep_search: filled query")  # filtered
    pad.think("Found three highly-relevant papers on RAG production patterns")  # kept
    pad.think("Loop finalized — handing off")  # filtered (not informative)

    outcome = ReactOutcome(
        scratchpad=pad,
        new_results={},
        completed_normally=True,
        iterations=2,
    )
    spec = get_subagent("researcher")
    result = _distill_outcome(spec=spec, outcome=outcome)
    assert "highly-relevant papers on RAG" in result.summary
    # Filtered "Auto-repaired" entries don't pollute the summary.
    assert "Auto-repaired" not in result.summary
