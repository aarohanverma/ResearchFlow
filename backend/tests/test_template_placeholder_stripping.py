"""Tests for the placeholder detector's coverage of template-variable syntax.

The planner LLM regularly emits Mustache / Jinja / Handlebars / Liquid
style template variables like ``{{best_supporting_paper_id}}`` for
cross-step references, expecting some substitution layer to fill them
in. There is no such layer — without an explicit detector entry the
placeholder reaches the tool unchanged and produces traces like
``"Paper not found: {{best_supporting_paper_id}}"``.

Coverage:

* ``{{var}}`` / ``{{ var }}`` — Mustache / Jinja / Handlebars / Liquid
* ``${var}``                 — JavaScript template literal
* ``<%= var %>`` / ``<% var %>`` — ERB / EJS
"""

from __future__ import annotations

import pytest

from app.assistant.react_loop import _looks_like_placeholder, _preflight_and_repair_params, PaperLedger


# ── Detector coverage ──────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [
    "{{best_supporting_paper_id}}",
    "{{ best_supporting_paper_id }}",
    "{{paperId}}",
    "{{paper_id}}",
    "{{ user.query }}",
    "${var_name}",
    "${ paperId }",
    "<%= paper_id %>",
    "<% paper_id %>",
    # LangChain-style cross-step references the planner LLM hallucinates
    # from training data. The orchestrator has no such substitution
    # layer, so these MUST be caught at preflight or the literal string
    # lands at paper_qa / compare_papers as a paper id.
    "$STEP2.paper_ids[0]",
    "$STEP2.paper_ids[1]",
    "$STEP2.paper_ids[2]",
    "${STEP_2.paper_ids}",
    "STEP1.paper_id",
    "step3.results[2]",
    "<<step_2.output>>",
    "[step1.id]",
    "output of step 2",
    "id from step 1",
])
def test_template_placeholder_recognised(value):
    assert _looks_like_placeholder(value), f"expected placeholder: {value!r}"


@pytest.mark.parametrize("value", [
    "5e2b3c7c-9a4f-4f8d-9b3d-1234567890ab",   # real UUID
    "2401.12345",                              # real arXiv id
    "What is mechanistic interpretability?",   # real query
    "Lost in the Middle: How Language Models Use Long Contexts",  # real title
    # Real strings containing braces in non-template positions must NOT
    # be mistaken for placeholders:
    "Paper {1}: a study (1995)",
    "$100 worth of GPU credits",
    "a{2}b",
])
def test_real_values_pass_through(value):
    assert not _looks_like_placeholder(value), f"false positive on: {value!r}"


# ── End-to-end: preflight strips the template placeholder ──────────────────


def test_preflight_strips_double_brace_paper_id():
    """The ReAct loop / planner preflight must strip a template-style
    paper_id so the tool doesn't receive the raw template text."""
    schema = {
        "properties": {
            "paper_id": {"type": "string"},
            "question": {"type": "string"},
        },
        "required": ["question"],
    }
    raw = {"paper_id": "{{best_supporting_paper_id}}", "question": "Does X hold?"}
    repaired, notes = _preflight_and_repair_params(
        "paper_qa", raw, schema, query="Does X hold?", ledger=PaperLedger(),
    )
    assert "paper_id" not in repaired or repaired.get("paper_id") == ""
    assert any("placeholder paper_id" in n for n in notes)


@pytest.mark.asyncio
async def test_paper_qa_refuses_empty_id_and_title():
    """When BOTH paper_id and paper_title are empty after preflight,
    paper_qa must refuse to run with a clear summary — without this
    the tool produces a misleading 'Paper not found:' trace with no
    subject, the exact bug behind the screenshot."""
    from unittest.mock import AsyncMock
    from app.assistant.tools.paper_qa import paper_qa_tool, PaperQAInput

    ctx = AsyncMock()
    ctx.emit_progress = AsyncMock()
    ctx.db = AsyncMock()

    params = PaperQAInput(question="Does X hold?", paper_id="", paper_title="")
    result = await paper_qa_tool.run(ctx, params)

    assert result.output["found"] is False
    assert result.output["chunks_used"] == 0
    assert "neither paper_id nor paper_title" in result.summary
    # ctx.db.execute must NOT be called — we refused before any DB work.
    ctx.db.execute.assert_not_called()


def test_preflight_strips_stepn_template_and_autofills_from_ledger():
    """The screenshot bug: planner emits ``$STEP2.paper_ids[0]`` as
    the paper_id, the literal string survives preflight, and paper_qa
    reports ``Paper not found: $STEP2.paper_ids[0]``. After this fix
    the placeholder is stripped AND the ledger fills paper_id from
    real retrieval results."""
    schema = {
        "properties": {
            "paper_id": {"type": "string"},
            "paper_title": {"type": "string"},
            "question": {"type": "string"},
        },
        "required": ["question"],
    }
    raw = {
        "paper_id": "$STEP2.paper_ids[0]",
        "paper_title": "",
        "question": "What method does the top paper use?",
    }
    ledger = PaperLedger()
    ledger.by_id["abc-123-real-uuid"] = {"title": "Real Paper", "ns": "ai-ml"}
    repaired, notes = _preflight_and_repair_params(
        "paper_qa", raw, schema,
        query="What method does the top paper use?", ledger=ledger,
    )
    assert repaired["paper_id"] == "abc-123-real-uuid"
    assert any("placeholder paper_id" in n for n in notes)
    assert any("auto-filled 'paper_id' from ledger for paper_qa" in n for n in notes)


def test_preflight_strips_stepn_in_paper_ids_list_and_refills_for_compare_papers():
    """compare_papers takes ``paper_ids: list[str]`` (required, min_length=2).
    When every element is a ``$STEP2.paper_ids[i]`` template, the list
    must be detected as a placeholder list, dropped, and refilled from
    the ledger so the comparison can run instead of being skipped."""
    schema = {
        "properties": {
            "paper_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "focus": {"type": "string"},
        },
        "required": ["paper_ids"],
    }
    raw = {
        "paper_ids": [
            "$STEP2.paper_ids[0]",
            "$STEP2.paper_ids[1]",
            "$STEP2.paper_ids[2]",
        ],
        "focus": "data efficiency",
    }
    ledger = PaperLedger()
    ledger.by_id["id-1"] = {"title": "Paper One", "ns": "ai-ml"}
    ledger.by_id["id-2"] = {"title": "Paper Two", "ns": "ai-ml"}
    ledger.by_id["id-3"] = {"title": "Paper Three", "ns": "ai-ml"}
    repaired, notes = _preflight_and_repair_params(
        "compare_papers", raw, schema,
        query="compare these papers", ledger=ledger,
    )
    assert isinstance(repaired.get("paper_ids"), list)
    assert len(repaired["paper_ids"]) >= 2
    assert all(pid.startswith("id-") for pid in repaired["paper_ids"])
    assert any("placeholder paper_ids" in n for n in notes)
    assert any("auto-filled 'paper_ids' from ledger" in n for n in notes)


def test_preflight_autofills_query_when_brace_template_stripped():
    """When the planner emitted ``query="{{user_query}}"`` we expect
    the placeholder to be removed AND the required field auto-filled
    from the actual user query — not left empty."""
    schema = {
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    }
    raw = {"query": "{{user_query}}"}
    repaired, notes = _preflight_and_repair_params(
        "deep_search", raw, schema,
        query="What is mechanistic interpretability?", ledger=PaperLedger(),
    )
    assert repaired["query"] == "What is mechanistic interpretability?"
    assert any("auto-filled 'query'" in n for n in notes)
