"""Regression tests for the ReAct loop's tool-knowledge hardening.

The audit screenshots showed the model emitting ``params={}`` and
``__to_fill_from_retrieval__*`` to tools that required real values
(``query``, ``paper_id``, ``paper_ids``) and the loop then logging a
pydantic ``query field required`` error and giving up. This file pins
the contract for the fix:

  * the placeholder detector catches every shape the model has been
    seen to emit (underscore-bounded, angle-bracketed, brace/bracket
    placeholders, raw "tbd" / "null" / empty strings),
  * the catalog renderer surfaces required + optional params with
    types and descriptions so the model can fill them correctly the
    first time,
  * the paper-id ledger collects ids from retrieval results and
    surfaces them in the decision prompt for compare/paper_qa/
    genie_synthesize,
  * the preflight repair fills missing required fields from the user
    query and the ledger before pydantic validation fires,
  * the loop's failure handling bans a tool after repeated failures
    and tells the next decision step which tools are banned.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest

from app.assistant import react_loop as rl
from app.assistant.tools.base import ToolResult


# ── Placeholder detection ─────────────────────────────────────────────────


def test_placeholder_detector_catches_underscore_form():
    assert rl._looks_like_placeholder("__to_fill_from_retrieval__")
    assert rl._looks_like_placeholder("__fill__")
    assert rl._looks_like_placeholder("__to_fill_from_retrieval__best")


def test_placeholder_detector_catches_angle_and_brace_forms():
    assert rl._looks_like_placeholder("<TODO>")
    assert rl._looks_like_placeholder("<<fill>>")
    assert rl._looks_like_placeholder("{placeholder}")
    assert rl._looks_like_placeholder("[tbd]")


def test_placeholder_detector_catches_raw_keywords():
    assert rl._looks_like_placeholder("null")
    assert rl._looks_like_placeholder("None")
    assert rl._looks_like_placeholder("tbd")
    assert rl._looks_like_placeholder("undefined")
    assert rl._looks_like_placeholder("???")
    assert rl._looks_like_placeholder("")
    assert rl._looks_like_placeholder("   ")


def test_placeholder_detector_accepts_real_values():
    assert not rl._looks_like_placeholder("transformer attention mechanisms")
    assert not rl._looks_like_placeholder("550e8400-e29b-41d4-a716-446655440000")
    assert not rl._looks_like_placeholder(0)
    assert not rl._looks_like_placeholder(False)
    assert not rl._looks_like_placeholder(["abc", "def"])  # real list
    assert not rl._looks_like_placeholder([])              # empty list passes


def test_placeholder_detector_recursive_on_lists():
    assert rl._looks_like_placeholder(["__fill__", "<TODO>"])
    # Mixed → not a placeholder (real value present)
    assert not rl._looks_like_placeholder(["__fill__", "real-id"])


# ── Paper ledger ──────────────────────────────────────────────────────────


def test_paper_ledger_collects_ids_from_papers_list():
    ledger = rl.PaperLedger()
    result = ToolResult(
        output={
            "papers": [
                {"paper_id": "pid-1", "title": "First paper", "namespace_key": "cs.AI"},
                {"paper_id": "pid-2", "title": "Second paper"},
            ],
        },
        summary="two papers",
    )
    added = ledger.add_from_result(result)
    assert added == 2
    assert ledger.ids() == ["pid-1", "pid-2"]


def test_paper_ledger_deduplicates_across_retrievals():
    ledger = rl.PaperLedger()
    ledger.add_from_result(ToolResult(
        output={"papers": [{"paper_id": "pid-1", "title": "A"}]},
        summary="",
    ))
    added2 = ledger.add_from_result(ToolResult(
        output={"papers": [{"paper_id": "pid-1", "title": "A"},
                           {"paper_id": "pid-2", "title": "B"}]},
        summary="",
    ))
    assert added2 == 1  # only the new pid counts
    assert ledger.ids() == ["pid-1", "pid-2"]


def test_paper_ledger_renders_for_prompt():
    ledger = rl.PaperLedger()
    ledger.add_from_result(ToolResult(
        output={"papers": [
            {"paper_id": "abc", "title": "Attention is all you need", "namespace_key": "cs.LG"},
        ]},
        summary="",
    ))
    rendered = ledger.render_for_prompt()
    assert "abc" in rendered
    assert "Attention is all you need" in rendered
    assert "cs.LG" in rendered


def test_paper_ledger_empty_message_warns_model():
    """Empty ledger renders a message that tells the model what to do —
    so it doesn't try to invoke compare_papers with placeholders."""
    ledger = rl.PaperLedger()
    rendered = ledger.render_for_prompt()
    assert "no papers retrieved yet" in rendered.lower()
    assert "retrieval" in rendered.lower()


# ── Catalog rendering ────────────────────────────────────────────────────


def test_catalog_renders_required_params_with_descriptions():
    """The model used to see only ``name: summary``. The catalog must
    now expose every required field by name + type so the model can
    fill it correctly the first time."""
    catalog = [
        {
            "name": "deep_search",
            "summary": "Search the corpus.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
    ]
    rendered = rl._render_tool_catalog(catalog)
    assert "deep_search" in rendered
    assert "query" in rendered
    assert "required" in rendered
    assert "Search query" in rendered
    assert "limit" in rendered
    assert "default=8" in rendered


def test_catalog_handles_array_fields():
    catalog = [
        {
            "name": "compare_papers",
            "summary": "Compare papers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                },
                "required": ["paper_ids"],
            },
        },
    ]
    rendered = rl._render_tool_catalog(catalog)
    assert "paper_ids" in rendered
    assert "required" in rendered


# ── Preflight repair ─────────────────────────────────────────────────────


def test_preflight_fills_missing_query_from_user_query():
    schema = {
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    params, notes = rl._preflight_and_repair_params(
        "deep_search", {}, schema,
        query="transformer attention mechanisms",
        ledger=rl.PaperLedger(),
    )
    assert params.get("query") == "transformer attention mechanisms"
    assert any("auto-filled" in n and "query" in n for n in notes)


def test_preflight_strips_underscore_placeholder():
    schema = {
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    params, notes = rl._preflight_and_repair_params(
        "deep_search",
        {"query": "__to_fill_from_retrieval__"},
        schema,
        query="moe routing",
        ledger=rl.PaperLedger(),
    )
    # Placeholder removed, then refilled from the user query.
    assert params["query"] == "moe routing"
    assert any("removed placeholder" in n for n in notes)
    assert any("auto-filled" in n for n in notes)


def test_preflight_fills_paper_ids_from_ledger():
    """compare_papers requires ≥2 paper_ids. When the model leaves it
    empty but the ledger has 2+ retrieved papers, the preflight pulls
    them in so the call goes through instead of dying with
    ``paper_ids field required``.
    """
    ledger = rl.PaperLedger()
    ledger.add_from_result(ToolResult(
        output={"papers": [
            {"paper_id": "pid-1", "title": "A"},
            {"paper_id": "pid-2", "title": "B"},
            {"paper_id": "pid-3", "title": "C"},
        ]},
        summary="",
    ))
    schema = {
        "properties": {"paper_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2}},
        "required": ["paper_ids"],
    }
    params, notes = rl._preflight_and_repair_params(
        "compare_papers", {}, schema,
        query="anything",
        ledger=ledger,
    )
    assert params.get("paper_ids") and len(params["paper_ids"]) >= 2
    assert params["paper_ids"][0] == "pid-1"
    assert any("ledger" in n for n in notes)


def test_preflight_fills_paper_id_singular_from_ledger():
    """paper_qa wants a single paper_id; the ledger's top item is the
    most relevant retrieval so we pick it as the default repair."""
    ledger = rl.PaperLedger()
    ledger.add_from_result(ToolResult(
        output={"papers": [{"paper_id": "pid-top", "title": "Top"}]},
        summary="",
    ))
    schema = {
        "properties": {"paper_id": {"type": "string"}, "question": {"type": "string"}},
        "required": ["paper_id", "question"],
    }
    params, notes = rl._preflight_and_repair_params(
        "paper_qa", {"question": "what is X?"}, schema,
        query="what is X?",
        ledger=ledger,
    )
    assert params["paper_id"] == "pid-top"
    assert params["question"] == "what is X?"


def test_preflight_leaves_unfillable_required_fields_for_pydantic():
    """When neither the user query nor the ledger can supply a value,
    the preflight does NOT fabricate one — let pydantic surface the
    missing-field error and have the loop record a clear observation
    instead of silently dispatching garbage."""
    schema = {
        "properties": {"weird_required_field": {"type": "string"}},
        "required": ["weird_required_field"],
    }
    params, _notes = rl._preflight_and_repair_params(
        "some_tool", {}, schema,
        query="x",
        ledger=rl.PaperLedger(),
    )
    assert "weird_required_field" not in params


# ── End-to-end: repair + ban flow inside the loop ────────────────────────


def _make_ctx_factory():
    """Minimal ctx_factory recorder copied from the existing hardening test."""
    from contextlib import asynccontextmanager
    from app.assistant.tools.base import ToolContext

    @asynccontextmanager
    async def _factory():
        async def _never(): return False
        async def _noop(*_a, **_k): return None
        ctx = ToolContext(
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
        yield ctx
    return _factory


@pytest.mark.asyncio
async def test_loop_repairs_empty_params_with_ledger_and_query(monkeypatch):
    """End-to-end: model emits ``{}`` to a tool that requires ``query``;
    the loop must repair the call rather than skip the action.
    """
    from pydantic import BaseModel, Field

    class _FakeInput(BaseModel):
        query: str = Field(min_length=1, max_length=400)

    class _FakeOutput(BaseModel):
        ok: bool = True

    fake_tool = MagicMock()
    fake_tool.input_schema = _FakeInput
    fake_tool.output_schema = _FakeOutput

    seen_params: list[_FakeInput] = []

    async def _run(_ctx, params):
        seen_params.append(params)
        return ToolResult(
            output={"papers": [{"paper_id": "p-1", "title": "T"}]},
            summary="ok",
        )
    fake_tool.run = _run

    decisions = [
        {"thought": "go", "action": "the_tool", "params": {}},
        {"thought": "done", "action": "finalize"},
    ]

    async def _decide(**_kw):
        return decisions.pop(0)

    monkeypatch.setattr(rl, "_decide_next_action", _decide)
    monkeypatch.setattr(rl, "get_tool", lambda n: fake_tool if n == "the_tool" else None)

    outcome = await rl.run_react_loop(
        query="moe routing limits",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx_factory=_make_ctx_factory(),
        config=rl.ReactConfig(max_iterations=3, deadline_seconds=10),
    )

    assert len(seen_params) == 1
    # ``query`` got auto-filled from the user query, not left as ``""``.
    assert seen_params[0].query == "moe routing limits"
    assert outcome.tool_failures == 0
    # Loop produced a result so the ledger grew.
    assert outcome.paper_ledger_size >= 1


@pytest.mark.asyncio
async def test_loop_bans_tool_after_repeated_failures(monkeypatch):
    """A tool that crashes twice gets banned for the remainder of the
    turn so the model can't burn every remaining iteration on it."""

    from pydantic import BaseModel

    class _Input(BaseModel):
        query: str = ""

    bad_tool = MagicMock()
    bad_tool.input_schema = _Input
    async def _crash(_ctx, _p):
        raise RuntimeError("kaboom")
    bad_tool.run = _crash

    # Track what was offered to the decision step on each iteration so
    # we can verify the ban shows up.
    catalog_calls: list[list[str]] = []

    async def _decide(**kw):
        # Inspect the catalog the prompt builder will render — proxy
        # for what the model sees. We use the same registry view the
        # production code uses.
        from app.assistant.tools.registry import describe_for_planner
        names = [t["name"] for t in describe_for_planner()]
        catalog_calls.append(names)
        return {"thought": "try again", "action": "bad_tool", "params": {"query": "x"}}

    monkeypatch.setattr(rl, "_decide_next_action", _decide)
    monkeypatch.setattr(rl, "get_tool", lambda n: bad_tool if n == "bad_tool" else None)

    outcome = await rl.run_react_loop(
        query="x",
        initial_plan_actions=[],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx_factory=_make_ctx_factory(),
        config=rl.ReactConfig(max_iterations=5, deadline_seconds=10),
    )

    # Bad tool ran at most twice — third attempt must hit the ban path
    # and short-circuit before dispatch.
    assert outcome.tool_failures >= 2
    # Scratchpad reflects the ban in at least one observation.
    obs_texts = [
        e.summary.lower()
        for e in outcome.scratchpad.entries
        if getattr(e, "kind", None) == "observation"
    ]
    assert any("banned" in t for t in obs_texts), obs_texts
