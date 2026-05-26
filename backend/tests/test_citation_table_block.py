"""Tests for the consolidated citation table block.

The synthesizer emits a ``citation_table`` block at the end of every
grounded message so the user can audit every cited source at a glance:
marker, title, evidence tier, verification verdict, and clickable
destination. Rows that cannot be resolved to a real source must
surface as ``verdict="unresolved"`` rather than silently look
grounded.
"""

from __future__ import annotations

from app.assistant.synthesizer import (
    _build_citation_rows,
    build_message_blocks,
)


# ── _build_citation_rows: corpus / external row composition ─────────────────


def test_rows_emitted_for_corpus_papers_with_evidence_tier_from_ledger():
    papers = [
        {
            "paper_id": "uuid-1",
            "title": "Verified Paper",
            "authors": ["Doe"],
            "namespace_key": "cs.LG",
            "source_url": "https://example.org/p1",
        },
    ]
    agent_notes = {"claim_ledger": {
        "total": 1,
        "verified": [{
            "paper_id": "uuid-1",
            "span": "X achieves Y",
            "evidence_tier": "experiment-verified",
        }],
    }}
    rows = _build_citation_rows(papers=papers, arxiv_results=[], agent_notes=agent_notes)
    assert len(rows) == 1
    row = rows[0]
    assert row["marker"] == "1"
    assert row["is_external"] is False
    assert row["paper_id"] == "uuid-1"
    assert row["evidence_tier"] == "experiment-verified"
    assert row["verdict"] == "verified"


def test_rows_pick_strongest_tier_when_paper_has_multiple_claims():
    papers = [{"paper_id": "uuid-1", "title": "P", "authors": []}]
    agent_notes = {"claim_ledger": {
        "total": 2,
        "verified": [
            {"paper_id": "uuid-1", "span": "method claim", "evidence_tier": "method-verified"},
            {"paper_id": "uuid-1", "span": "results claim", "evidence_tier": "experiment-verified"},
        ],
    }}
    rows = _build_citation_rows(papers=papers, arxiv_results=[], agent_notes=agent_notes)
    assert rows[0]["evidence_tier"] == "experiment-verified"


def test_external_row_with_arxiv_id_resolves_to_abs_url():
    arxiv = [{"external_id": "2401.12345", "title": "External", "authors": ["Doe"]}]
    rows = _build_citation_rows(papers=[], arxiv_results=arxiv, agent_notes={})
    assert rows[0]["marker"] == "A1"
    assert rows[0]["is_external"] is True
    assert rows[0]["url"] == "https://arxiv.org/abs/2401.12345"
    assert rows[0]["verdict"] == "unverified"  # external default until inspected


def test_external_row_with_doi_external_id_resolves_to_doi_url():
    arxiv = [{"external_id": "10.1000/xyz", "title": "DOI Paper", "authors": []}]
    rows = _build_citation_rows(papers=[], arxiv_results=arxiv, agent_notes={})
    assert rows[0]["url"] == "https://doi.org/10.1000/xyz"


def test_external_row_falls_back_to_source_url_when_external_id_empty():
    arxiv = [{
        "external_id": "",
        "source_url": "https://example.org/preprint",
        "title": "No ID Paper",
        "authors": [],
    }]
    rows = _build_citation_rows(papers=[], arxiv_results=arxiv, agent_notes={})
    assert rows[0]["url"] == "https://example.org/preprint"


def test_unresolvable_external_row_marked_unresolved():
    """A candidate with neither external_id nor source_url must
    surface as ``verdict='unresolved'`` so the UI can highlight it —
    the user explicitly asked us not to silently treat unresolved
    citations as grounded."""
    arxiv = [{"external_id": "", "source_url": "", "title": "Mystery", "authors": []}]
    rows = _build_citation_rows(papers=[], arxiv_results=arxiv, agent_notes={})
    assert rows[0]["verdict"] == "unresolved"
    assert rows[0]["url"] == ""


def test_provenance_flagged_marker_downgrades_to_unverified():
    """When the provenance verifier flagged a corpus marker as
    unsupported, the table row must surface it as ``unverified`` even
    if the ledger left it as ``provisional``."""
    papers = [{"paper_id": "uuid-1", "title": "P", "authors": []}]
    agent_notes = {
        "claim_ledger": {"total": 0},
        "provenance": {"flagged": [
            {"marker": "1", "verdict": "unverified"},
        ]},
    }
    rows = _build_citation_rows(papers=papers, arxiv_results=[], agent_notes=agent_notes)
    assert rows[0]["verdict"] == "unverified"


# ── build_message_blocks emits citation_table at the end ────────────────────


def test_build_message_blocks_appends_citation_table():
    papers = [{"paper_id": "uuid-1", "title": "P", "authors": ["Doe"]}]
    blocks = build_message_blocks(
        answer="Some answer [1].",
        papers=papers,
        arxiv_results=[],
        imported_count=0,
        graph_result=None,
        genie_session_id=None,
        suggestions=[],
        actions=[],
    )
    kinds = [b["kind"] for b in blocks]
    assert kinds[-1] == "citation_table"  # always at the end
    table = blocks[-1]
    assert len(table["rows"]) == 1
    assert table["rows"][0]["marker"] == "1"


def test_build_message_blocks_omits_citation_table_when_no_sources():
    """An answer with no papers / external candidates should not
    emit an empty citation table (the block would add visual noise
    with nothing to show)."""
    blocks = build_message_blocks(
        answer="A free-reasoning answer with no citations.",
        papers=[],
        arxiv_results=[],
        imported_count=0,
        graph_result=None,
        genie_session_id=None,
        suggestions=[],
        actions=[],
    )
    assert all(b["kind"] != "citation_table" for b in blocks)


def test_build_message_blocks_citation_table_pinned_truly_last():
    """When many tail blocks (suggestions, actions, web_results) are
    present, the citation_table must still be the FINAL block — the
    user spec is unambiguous, so the helper enforces this with a
    defensive sort regardless of the order helpers below it appended
    in."""
    papers = [{"paper_id": "uuid-1", "title": "P", "authors": ["Doe"]}]
    blocks = build_message_blocks(
        answer="ans [1].",
        papers=papers,
        arxiv_results=[{"external_id": "2401.1", "title": "Ext", "authors": []}],
        imported_count=2,
        graph_result=None,
        genie_session_id="sess-1",
        suggestions=[{"label": "Try X"}],
        actions=["did stuff"],
        web_results=[{"title": "W", "url": "https://example.org", "snippet": "s"}],
    )
    # Many blocks emitted, citation_table is still LAST.
    assert blocks[-1]["kind"] == "citation_table"
    assert sum(1 for b in blocks if b["kind"] == "citation_table") == 1


def test_build_citation_rows_handles_malformed_agent_notes():
    """A non-dict ``claim_ledger`` (some future serialisation path
    that lands a list here) must not crash the builder — the row
    output silently falls back to default tier/verdict labels."""
    from app.assistant.synthesizer import _build_citation_rows

    papers = [{"paper_id": "uuid-1", "title": "P", "authors": []}]
    # claim_ledger is a list (malformed); provenance.flagged is None
    agent_notes = {"claim_ledger": ["not-a-dict"], "provenance": None}
    rows = _build_citation_rows(papers=papers, arxiv_results=[], agent_notes=agent_notes)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "provisional"
    assert rows[0]["evidence_tier"] == "abstract-only"


def test_build_citation_rows_handles_non_dict_bucket_items():
    """When ledger buckets contain non-dict entries (LLM emitted
    malformed JSON), the builder must skip them gracefully."""
    from app.assistant.synthesizer import _build_citation_rows

    papers = [{"paper_id": "uuid-1", "title": "P", "authors": []}]
    agent_notes = {"claim_ledger": {
        "verified": ["scrambled-non-dict", {"paper_id": "uuid-1", "evidence_tier": "method-verified"}],
    }}
    rows = _build_citation_rows(papers=papers, arxiv_results=[], agent_notes=agent_notes)
    assert rows[0]["evidence_tier"] == "method-verified"
    assert rows[0]["verdict"] == "verified"
