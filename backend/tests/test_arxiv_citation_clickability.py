"""Tests for the [A*] external-citation clickability fix.

The synthesizer's strip pass must remove ``[A*]`` markers whose
corresponding ``arxiv_results`` entry has no usable destination
(neither ``external_id`` nor ``source_url``) — otherwise the
frontend renders a styled-but-inert citation chip. The orchestrator's
``_arxiv_results_from_results`` normaliser is the upstream source of
the ``source_url`` fallback, so the strip pass and the normaliser
contract must stay aligned.
"""

from __future__ import annotations

from app.assistant.orchestrator import Orchestrator
from app.assistant.synthesizer import _strip_unresolvable_citations
from app.assistant.tools.base import ToolResult


# ── _strip_unresolvable_citations: [A*] dest filter ─────────────────────────


def test_strip_keeps_arxiv_marker_when_external_id_present():
    answer = "Recent work surveyed RAG approaches [A1]."
    arxiv = [{"external_id": "2401.12345", "title": "RAG Survey"}]
    assert _strip_unresolvable_citations(answer, [], arxiv) == answer


def test_strip_keeps_arxiv_marker_when_source_url_present():
    """A non-arXiv preprint (DOI, publisher page) must keep its marker
    as long as ``source_url`` resolves to a real destination."""
    answer = "An earlier study showed similar results [A1]."
    arxiv = [{
        "external_id": "",
        "source_url": "https://doi.org/10.1000/xyz",
        "title": "Earlier Study",
    }]
    assert _strip_unresolvable_citations(answer, [], arxiv) == answer


def test_strip_removes_marker_when_no_usable_destination():
    """A candidate with neither external_id nor source_url is inert in
    the UI (chip styled as a link but click does nothing) — the
    marker must be stripped so on-screen state stays honest."""
    answer = "The benchmark was introduced earlier [A1]."
    arxiv = [{"external_id": "", "source_url": "", "title": "Mystery Paper"}]
    assert _strip_unresolvable_citations(answer, [], arxiv) == "The benchmark was introduced earlier."


def test_strip_partials_compound_marker_against_resolvable_set():
    """``[A1,2,3]`` where only A1 has a destination should collapse
    to ``[A1]`` rather than vanish entirely or stay broken — the
    indices inside one ``[A...]`` marker share the ``A`` prefix per
    the synthesizer prompt's citation syntax."""
    answer = "Multiple surveys cover this ground [A1,2,3]."
    arxiv = [
        {"external_id": "2401.0001", "title": "Resolvable"},
        {"external_id": "", "source_url": "", "title": "Inert A2"},
        {"external_id": "", "source_url": "", "title": "Inert A3"},
    ]
    out = _strip_unresolvable_citations(answer, [], arxiv)
    assert out == "Multiple surveys cover this ground [A1]."


def test_strip_independent_markers_filtered_by_resolvability():
    """Independent ``[A1]`` / ``[A2]`` / ``[A3]`` markers must each
    be filtered against the resolvable set — only the resolvable
    ones survive, with whitespace and orphan punctuation cleaned."""
    answer = "Survey [A1], earlier work [A2], and a follow-up [A3] cover this."
    arxiv = [
        {"external_id": "2401.0001", "title": "Resolvable"},
        {"external_id": "", "source_url": "", "title": "Inert"},
        {"external_id": "", "source_url": "https://doi.org/x", "title": "Resolvable via DOI"},
    ]
    out = _strip_unresolvable_citations(answer, [], arxiv)
    # A1 kept; A2 stripped (and its leading "earlier work " kept);
    # A3 kept (resolvable via DOI fallback).
    assert "[A1]" in out
    assert "[A2]" not in out
    assert "[A3]" in out


# ── _arxiv_results_from_results: source_url fallback ────────────────────────


def _wrap(name: str, output: dict) -> dict:
    """Build a one-tool result map keyed by tool name."""
    return {name: ToolResult(output=output, summary="ok")}


def test_arxiv_normaliser_preserves_source_url_from_frontier_scan():
    """frontier_scan emits ``source_url``/``pdf_url`` — both must
    survive into the normalised candidate so the frontend's citation
    map can fall back to them when external_id is empty."""
    results = _wrap("frontier_scan", {"papers": [
        {
            "external_id": "",
            "title": "Frontier Paper",
            "authors": ["Doe"],
            "source_url": "https://example.org/paper",
            "pdf_url": "https://example.org/paper.pdf",
        }
    ]})
    out = Orchestrator._arxiv_results_from_results(results)
    assert len(out) == 1
    assert out[0]["source_url"] == "https://example.org/paper"


def test_arxiv_normaliser_derives_arxiv_url_from_external_id():
    """When the upstream tool gave us an arXiv id but no explicit
    ``source_url``, the normaliser must derive the abs URL itself —
    keeps the [A*] chip resolvable end-to-end."""
    results = _wrap("arxiv_search", {"results": [
        {"external_id": "2401.99999", "title": "Inferred URL"},
    ]})
    out = Orchestrator._arxiv_results_from_results(results)
    assert out[0]["external_id"] == "2401.99999"
    assert out[0]["source_url"] == "https://arxiv.org/abs/2401.99999"


def test_arxiv_normaliser_derives_doi_url_from_external_id():
    """A citation_finder hit whose external_id looks like a DOI must
    resolve to doi.org rather than arxiv.org/abs."""
    results = _wrap("citation_finder", {"papers": [
        {"external_id": "10.1000/xyz", "title": "DOI Paper"},
    ]})
    out = Orchestrator._arxiv_results_from_results(results)
    assert out[0]["source_url"] == "https://doi.org/10.1000/xyz"


def test_arxiv_normaliser_leaves_url_empty_when_nothing_resolvable():
    """No external_id, no source_url, no pdf_url → the normaliser
    must leave ``source_url`` empty rather than fabricate one. The
    strip pass then drops the marker."""
    results = _wrap("citation_finder", {"papers": [
        {"title": "Ghost Paper"},
    ]})
    out = Orchestrator._arxiv_results_from_results(results)
    assert out[0]["source_url"] == ""


def test_arxiv_normaliser_handles_full_url_in_external_id():
    """When the upstream tool jammed a full URL into ``external_id``
    (citation_finder is the usual culprit), the normaliser must
    surface it directly rather than concatenating it onto an
    ``arxiv.org/abs/`` prefix — without this guard the frontend
    renders a doubled-prefix URL that 404s."""
    results = _wrap("citation_finder", {"papers": [
        {
            "external_id": "https://example.org/preprint/12345",
            "title": "Full URL Paper",
        },
    ]})
    out = Orchestrator._arxiv_results_from_results(results)
    assert out[0]["source_url"] == "https://example.org/preprint/12345"
