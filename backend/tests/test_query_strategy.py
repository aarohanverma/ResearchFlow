"""Adaptive query-shape strategy router."""

from __future__ import annotations

from app.assistant.query_strategy import classify_query


def test_definition_query_picks_concept_first():
    s = classify_query("What is BERT?")
    assert s.shape == "definition"
    assert "concept_explain" in s.preferred_tools
    assert "literature_survey" in s.avoid_tools
    assert s.retrieval_limit <= 5


def test_comparison_query_bumps_retrieval_limit():
    s = classify_query("Compare RAG vs long-context for production research")
    assert s.shape == "comparison"
    assert s.retrieval_limit >= 10
    assert s.rerank_intensity == "heavy"
    assert "compare_papers" in s.preferred_tools


def test_survey_query_routes_to_literature_survey():
    s = classify_query("Give me a literature review of mechanistic interpretability")
    assert s.shape == "survey"
    assert "literature_survey" in s.preferred_tools
    assert s.retrieval_limit >= 12


def test_synthesis_query_routes_to_genie():
    s = classify_query("Synthesize a novel architecture combining MoE and RAG")
    assert s.shape == "synthesis"
    assert "genie_synthesize" in s.preferred_tools
    # genie_read of stale capsules is exactly what we want to AVOID.
    assert "genie_read" in s.avoid_tools


def test_recency_query_routes_to_frontier():
    s = classify_query("What are the latest papers on diffusion transformers this month?")
    assert s.shape == "frontier"
    assert "frontier_scan" in s.preferred_tools


def test_identifier_query_routes_to_paper_qa():
    s = classify_query("Explain the contribution of arXiv:2304.12345")
    assert s.shape == "identifier_lookup"
    assert s.retrieval_limit <= 5
    assert "literature_survey" in s.avoid_tools


def test_followup_query_preserves_narrow_retrieval():
    s = classify_query(
        "Can you elaborate on that earlier point?",
        history=[
            {"role": "user", "content": "What is attention?"},
            {"role": "assistant", "content": "Attention is a mechanism that..."},
        ],
    )
    assert s.shape == "followup"
    assert "literature_survey" in s.avoid_tools


def test_default_exploratory_is_balanced():
    """A vanilla research question should land in the 'exploratory'
    bucket — neither the lookup nor the survey extreme."""
    s = classify_query("How do transformers handle long sequences?")
    assert s.shape in {"exploratory", "explanation"}
    # The exploratory / explanation defaults should NOT lock the
    # planner into a survey or a single-tool answer.
    assert "literature_survey" not in (s.avoid_tools or [])
    assert s.retrieval_limit >= 6


def test_empty_query_falls_back_to_exploratory_default():
    s = classify_query("")
    assert s.shape == "exploratory"
