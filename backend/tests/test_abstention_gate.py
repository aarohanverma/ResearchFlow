"""Low-grounding informational footer.

This is NOT an abstention gate — RA always answers the user's
question. The detector + footer exist so the user is *informed*
about evidence weakness without the answer being replaced or
prefixed with a warning. The polished response stays foregrounded.
"""

from __future__ import annotations

from app.assistant.synthesizer import (
    _has_low_grounding_signal,
    _low_grounding_notice,
)


def _papers(n: int) -> list[dict]:
    return [{"paper_id": f"p{i}", "title": f"Paper {i}"} for i in range(n)]


# ── False-positive guard: clean answers must NOT trip the footer ────────────


def test_no_footer_when_evidence_is_strong():
    """A turn with abundant evidence + high groundedness must pass
    through cleanly. Appending an "evidence is thin" note on a
    strong answer would itself be a regression."""
    assert _has_low_grounding_signal(
        answer="Solid answer.",
        papers=_papers(6),
        arxiv_results=_papers(2),
        agent_notes={
            "critique": {"groundedness": 0.85, "completeness": 0.8},
            "tool_failures": 0,
            "successful_retrievals": 2,
            "retrieval": {"thin_calls": 0},
        },
        output={"provenance": {"total": 5, "supported": 5, "flagged": []}},
    ) is False


def test_no_footer_when_no_signals_present():
    """When the agent didn't run ReAct at all (trivial-tier turn) the
    detector must stay quiet — the absence of signals is not itself
    a signal."""
    assert _has_low_grounding_signal(
        answer="Short answer.",
        papers=_papers(2),
        arxiv_results=[],
        agent_notes=None,
        output=None,
    ) is False


# ── True-positive cases: each individual signal triggers a footer ───────────


def test_footer_on_low_groundedness_and_thin_evidence():
    """The single most important honesty signal: the agent's own
    critique flagged low groundedness AND the evidence base is tiny.
    We surface this to the user but the answer itself stays."""
    assert _has_low_grounding_signal(
        answer="Confident-sounding answer.",
        papers=_papers(1),
        arxiv_results=[],
        agent_notes={"critique": {"groundedness": 0.25}},
        output=None,
    ) is True


def test_footer_on_evidence_expansion_failure():
    """Tool retries burned without recovery — the agent tried to
    expand evidence and couldn't. The user deserves to know that
    the answer rests on the initial-plan evidence alone."""
    assert _has_low_grounding_signal(
        answer="x",
        papers=_papers(2),
        arxiv_results=[],
        agent_notes={"tool_failures": 3, "successful_retrievals": 0},
        output=None,
    ) is True


def test_footer_on_multiple_thin_retrievals():
    """Two or more thin retrieval calls in one turn — the corpus
    didn't serve the query well; user should know to follow up."""
    assert _has_low_grounding_signal(
        answer="x",
        papers=_papers(3),
        arxiv_results=[],
        agent_notes={"retrieval": {"thin_calls": 2}},
        output=None,
    ) is True


def test_footer_on_unverified_citations_majority():
    """Provenance verifier reports <40% citations supported with at
    least 3 flagged. The answer's citations don't hold up."""
    assert _has_low_grounding_signal(
        answer="x",
        papers=_papers(4),
        arxiv_results=[],
        agent_notes={},
        output={"provenance": {
            "total": 5, "supported": 1,
            "flagged": [{"marker": "[1]"}, {"marker": "[2]"}, {"marker": "[3]"}, {"marker": "[4]"}],
        }},
    ) is True


def test_footer_quiet_below_provenance_minimum():
    """The provenance signal needs at least 3 flagged AND <40%
    supported — a single flagged citation on its own does not
    cascade into the footer."""
    assert _has_low_grounding_signal(
        answer="x",
        papers=_papers(4),
        arxiv_results=[],
        agent_notes={"critique": {"groundedness": 0.8}},
        output={"provenance": {
            "total": 5, "supported": 4,
            "flagged": [{"marker": "[2]"}],
        }},
    ) is False


# ── Footer rendering — informational tone, appended, no warning chrome ──────


def test_notice_names_specific_reason_groundedness():
    notice = _low_grounding_notice(
        papers=_papers(1), arxiv_results=[],
        agent_notes={"critique": {"groundedness": 0.20}},
        output=None,
    )
    assert "groundedness" in notice.lower()
    assert "0.20" in notice


def test_notice_names_specific_reason_thin_retrievals():
    notice = _low_grounding_notice(
        papers=_papers(3), arxiv_results=[],
        agent_notes={"retrieval": {"thin_calls": 3}},
        output=None,
    )
    assert "thin coverage" in notice.lower()
    assert "3 retrieval" in notice


def test_notice_names_specific_reason_evidence_expansion():
    notice = _low_grounding_notice(
        papers=_papers(2), arxiv_results=[],
        agent_notes={"tool_failures": 2, "successful_retrievals": 0},
        output=None,
    )
    assert "errored" in notice.lower() or "expansion" in notice.lower()


def test_notice_names_specific_reason_provenance():
    notice = _low_grounding_notice(
        papers=_papers(4), arxiv_results=[],
        agent_notes={},
        output={"provenance": {
            "total": 5, "supported": 1,
            "flagged": [{}, {}, {}, {}],
        }},
    )
    assert "1/5" in notice
    assert "verified" in notice.lower()


def test_notice_is_informational_not_warning():
    """No alarm chrome — the footer must read as informational
    context, not as a warning that hijacks the answer's framing.
    The polished response stays foregrounded; the footer follows."""
    notice = _low_grounding_notice(
        papers=_papers(1), arxiv_results=[],
        agent_notes={"critique": {"groundedness": 0.10}},
        output=None,
    )
    assert "⚠" not in notice
    assert "tentative" not in notice.lower()
    assert "provisional" not in notice.lower()
    # Should be a markdown italic footer separated by a horizontal rule,
    # not a leading blockquote banner.
    assert notice.lstrip().startswith("---")
    assert "Heads-up" in notice or "heads-up" in notice.lower()


def test_notice_empty_string_when_no_reasons():
    """If somehow the detector fires but no specific reasons can be
    enumerated, return an empty string rather than a boilerplate
    footer — never inform the user about nothing."""
    notice = _low_grounding_notice(
        papers=_papers(10), arxiv_results=_papers(5),
        agent_notes={"critique": {"groundedness": 0.9}},
        output={"provenance": {"total": 10, "supported": 10, "flagged": []}},
    )
    assert notice == ""
