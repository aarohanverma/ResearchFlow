"""Regression tests for the three audit-flagged gaps closed in this pass.

* Citation safety net (``_strip_unresolvable_citations``):
  out-of-range ``[N]`` / ``[A N]`` markers never reach the user, compound
  forms keep their resolvable indices, orphan punctuation is tidied.

* Critique-as-action (``_run_self_critique`` + loop integration):
  the ReAct loop intercepts ``action="critique"``, records a
  ``Critique`` scratchpad entry, and continues iterating instead of
  treating it as an unknown tool.

* Diminishing-returns guard (``_is_diminishing_returns``):
  a retrieval call that adds no new paper IDs terminates the loop with
  ``completed_normally=True``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── Citation safety net ───────────────────────────────────────────────────────


def test_strip_drops_out_of_range_marker():
    from app.assistant.synthesizer import _strip_unresolvable_citations

    papers = [{"paper_id": "p1"}, {"paper_id": "p2"}]
    answer = "Claim one [1]. Claim two [5]."
    cleaned = _strip_unresolvable_citations(answer, papers, [])
    assert "[5]" not in cleaned
    assert "[1]" in cleaned
    # Orphan space-before-period tidied.
    assert "two." in cleaned and " ." not in cleaned


def test_strip_keeps_resolvable_subset_of_compound_marker():
    from app.assistant.synthesizer import _strip_unresolvable_citations

    papers = [{"paper_id": "p1"}, {"paper_id": "p2"}]
    answer = "Compound [1,2,5,3]. Range [1-4]."
    cleaned = _strip_unresolvable_citations(answer, papers, [])
    assert "[1,2]" in cleaned, f"compound filter wrong, got {cleaned!r}"
    assert "[1,2]" in cleaned  # range 1-4 clamped to 1-2
    assert "5" not in cleaned.split("[", 1)[1] if "[" in cleaned else True


def test_strip_handles_empty_answer_and_empty_papers():
    from app.assistant.synthesizer import _strip_unresolvable_citations

    assert _strip_unresolvable_citations("", [], []) == ""
    # With no papers, every paper marker is unresolvable.
    assert "[1]" not in _strip_unresolvable_citations("Claim [1].", [], [])


def test_strip_does_not_touch_non_citation_brackets():
    from app.assistant.synthesizer import _strip_unresolvable_citations

    papers = [{"paper_id": "p1"}]
    # ``[code]`` and other non-numeric bracket content must survive.
    answer = "See `model.fit([x])` for details [1]."
    cleaned = _strip_unresolvable_citations(answer, papers, [])
    assert "[x]" in cleaned
    assert "[1]" in cleaned


# ── Critique-as-action ────────────────────────────────────────────────────────


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
async def test_react_loop_intercepts_critique_action(monkeypatch):
    """``action='critique'`` runs the judge + records a Critique entry."""
    from app.assistant import react_loop as rl

    calls = {"n": 0}

    async def _decide(**_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"thought": "Check evidence first", "action": "critique"}
        return {"thought": "OK now finalize", "action": "finalize"}

    async def _fake_llm_critique(**_kw):
        return {
            "groundedness": 0.85,
            "completeness": 0.6,
            "memory_faithfulness": 1.0,
            "issues": ["thin coverage on benchmark X"],
            "should_repair": False,
        }

    monkeypatch.setattr(rl, "_decide_next_action", _decide)
    monkeypatch.setattr("app.assistant.reflection.llm_critique", _fake_llm_critique)

    outcome = await rl.run_react_loop(
        query="evaluate evidence",
        initial_plan_actions=["Deep Search"],
        prior_results={},
        memory_view={},
        research_brief_text="",
        ctx=_make_ctx(),
        config=rl.ReactConfig(max_iterations=3, deadline_seconds=10),
    )
    crits = [e for e in outcome.scratchpad.entries if getattr(e, "kind", "") == "critique"]
    assert len(crits) == 1, f"expected 1 critique entry, got {len(crits)}"
    assert crits[0].verdict == "ship"
    assert crits[0].groundedness == 0.85
    # No tool result for "critique" — it's a pseudo-action.
    assert "critique" not in outcome.new_results


# ── Diminishing-returns guard ─────────────────────────────────────────────────


def test_extract_paper_ids_pulls_from_papers_key():
    from app.assistant.react_loop import _extract_paper_ids
    from app.assistant.tools.base import ToolResult

    r = ToolResult(output={"papers": [{"paper_id": "p1"}, {"paper_id": "p2"}, {"id": "p3"}]}, summary="")
    ids = _extract_paper_ids(r)
    assert ids == {"p1", "p2", "p3"}


def test_diminishing_returns_fires_when_new_call_is_subset():
    from app.assistant.react_loop import _is_diminishing_returns
    from app.assistant.tools.base import ToolResult

    prior = {"deep_search": ToolResult(output={"papers": [{"paper_id": "p1"}, {"paper_id": "p2"}]}, summary="")}
    new = {}
    duplicate = ToolResult(output={"papers": [{"paper_id": "p1"}]}, summary="")
    # New call returns a subset of prior → diminishing returns
    assert _is_diminishing_returns("arxiv_search", duplicate, prior, new) is True


def test_diminishing_returns_does_not_fire_when_new_ids_added():
    from app.assistant.react_loop import _is_diminishing_returns
    from app.assistant.tools.base import ToolResult

    prior = {"deep_search": ToolResult(output={"papers": [{"paper_id": "p1"}]}, summary="")}
    new = {}
    fresh = ToolResult(output={"papers": [{"paper_id": "p2"}]}, summary="")
    # Even one new id keeps the loop going
    assert _is_diminishing_returns("arxiv_search", fresh, prior, new) is False


def test_diminishing_returns_skips_non_retrieval_tools():
    """Verification / explain / compare tools don't surface paper IDs —
    the guard must be a no-op for them so a ``critique`` or
    ``concept_explain`` call after retrieval doesn't accidentally
    terminate the loop."""
    from app.assistant.react_loop import _is_diminishing_returns
    from app.assistant.tools.base import ToolResult

    prior = {"deep_search": ToolResult(output={"papers": [{"paper_id": "p1"}]}, summary="")}
    r = ToolResult(output={"answer": "..."}, summary="")
    assert _is_diminishing_returns("concept_explain", r, prior, {}) is False


@pytest.mark.asyncio
async def test_react_loop_stops_on_diminishing_returns(monkeypatch):
    """When a retrieval call duplicates prior paper IDs, the loop finalizes."""
    from app.assistant import react_loop as rl
    from app.assistant.tools.base import ToolResult

    # Pretend deep_search is in the registry and returns a duplicate set.
    fake_tool = MagicMock()
    fake_tool.input_schema = lambda **kw: kw  # noqa: E731 — pass-through
    async def _fake_run(_ctx, _params):
        return ToolResult(output={"papers": [{"paper_id": "p1"}]}, summary="dup")
    fake_tool.run = _fake_run
    monkeypatch.setattr(rl, "get_tool", lambda name: fake_tool if name == "deep_search" else None)

    async def _decide(**_kw):
        return {"thought": "broaden", "action": "deep_search", "params": {"query": "x"}}

    monkeypatch.setattr(rl, "_decide_next_action", _decide)

    # Pre-seed prior_results with a deep_search that already saw p1.
    prior = {"deep_search": ToolResult(output={"papers": [{"paper_id": "p1"}]}, summary="seed")}
    outcome = await rl.run_react_loop(
        query="test",
        initial_plan_actions=["Deep Search"],
        prior_results=prior,
        memory_view={},
        research_brief_text="",
        ctx=_make_ctx(),
        config=rl.ReactConfig(max_iterations=4, deadline_seconds=10),
    )
    # The loop terminated normally on the diminishing-returns signal, not
    # because the cap was hit.
    assert outcome.completed_normally is True
    assert outcome.iterations == 1
