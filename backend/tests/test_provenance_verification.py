"""Claim-level provenance verification."""

from __future__ import annotations

from app.assistant.provenance_verification import verify_claims


def _paper(idx: int, title: str, abstract: str) -> dict:
    return {
        "paper_id": f"p-{idx}",
        "title": title,
        "abstract": abstract,
    }


def test_supported_claim_passes_verification():
    """A claim whose content words clearly overlap with the cited
    paper's text should be marked ``supported``."""
    papers = [_paper(
        1,
        "Mixture-of-Experts at Scale",
        "We train sparse Mixture-of-Experts (MoE) models with up to 1.6T parameters "
        "and observe sample-efficient gains on language modelling benchmarks.",
    )]
    answer = (
        "Sparse Mixture-of-Experts models scale to trillion-parameter regimes [1]."
    )
    rep = verify_claims(answer=answer, papers=papers)
    assert rep.total == 1
    assert rep.supported == 1
    assert rep.claims[0].verdict == "supported"


def test_unsupported_claim_flagged_when_paper_text_unrelated():
    """A claim about MoE cited to a paper about retrieval should land
    in the ``unsupported`` bucket because the salient terms don't
    appear in the paper text at all."""
    papers = [_paper(
        1,
        "Retrieval-Augmented Generation at Scale",
        "We study dense retrieval pipelines for question answering, "
        "combining BM25 + dense encoders + reranking for production RAG.",
    )]
    answer = (
        "Sparse Mixture-of-Experts and FlashAttention dominate the MoE regime [1]."
    )
    rep = verify_claims(answer=answer, papers=papers)
    assert rep.total == 1
    assert rep.claims[0].verdict == "unsupported"
    # The detector should specifically flag the salient terms that the
    # paper text didn't contain so the synthesizer can caveat them.
    assert any(
        s.lower() in {"mixture-of-experts", "flashattention", "moe"}
        for s in rep.claims[0].missing_salient
    )


def test_unverified_claim_when_overlap_is_borderline():
    """When only one content token overlaps and salient nouns are
    missing, the verifier should NOT land in ``supported``."""
    papers = [_paper(
        1,
        "Token Embeddings for Search",
        "Standard token embeddings learnt with contrastive objectives.",
    )]
    # Claim shares only "embeddings" with the paper; salient nouns
    # ("FlashAttention", "GPU") are absent from the paper text.
    answer = "FlashAttention kernels accelerate token embeddings on GPU [1]."
    rep = verify_claims(answer=answer, papers=papers)
    assert rep.total == 1
    assert rep.claims[0].verdict in ("unverified", "unsupported")


def test_arxiv_marker_resolves_to_arxiv_list():
    """``[A1]`` must look up ``arxiv_results[0]``, not corpus papers."""
    papers = [_paper(1, "Unrelated Corpus Paper", "Nothing about retrieval here.")]
    arxiv = [{
        "external_id": "2304.12345",
        "title": "Switch Transformer Routing at Scale",
        "abstract": (
            "Switch transformer routing at scale: we route tokens through "
            "expert subnetworks to scale model capacity efficiently."
        ),
    }]
    # Claim shares "Switch transformer routing" with the paper title +
    # abstract — strong supported signal.
    answer = "Switch transformer routing scales model capacity efficiently [A1]."
    rep = verify_claims(answer=answer, papers=papers, arxiv_results=arxiv)
    assert rep.total == 1
    assert rep.claims[0].marker == "[A1]"
    assert rep.claims[0].verdict == "supported"


def test_out_of_range_marker_silently_skipped():
    """A ``[7]`` marker when only 2 papers exist must not crash the
    verifier (the citation-strip pass cleans those up upstream)."""
    papers = [_paper(1, "Paper One", "content"), _paper(2, "Paper Two", "more content")]
    answer = "Some claim [7]."
    rep = verify_claims(answer=answer, papers=papers)
    assert rep.total == 0


def test_compound_marker_resolves_each_index():
    """``[1,2]`` resolves to BOTH papers — verification runs per index."""
    papers = [
        _paper(1, "Paper About RAG", "retrieval augmented generation methods"),
        _paper(2, "Paper About MoE", "mixture of experts routing techniques"),
    ]
    answer = "Both retrieval and routing matter [1,2]."
    rep = verify_claims(answer=answer, papers=papers)
    assert rep.total == 2


def test_report_render_includes_flagged_pairs():
    papers = [_paper(1, "Some Paper", "completely unrelated topic xyz")]
    answer = "MoE scaling works [1]."
    rep = verify_claims(answer=answer, papers=papers)
    lines = rep.render_for_agent_notes()
    assert any("UNSUPPORTED" in l or "UNVERIFIED" in l for l in lines)


def test_report_verified_ratio_is_one_when_all_supported():
    """Sanity check: a single clearly-supported claim → 100% verified."""
    papers = [_paper(
        1,
        "Mixture-of-Experts Scaling Laws",
        "Sparse Mixture-of-Experts scaling laws across model sizes and "
        "compute budgets in language modelling regimes.",
    )]
    answer = (
        "Sparse Mixture-of-Experts scaling laws hold across model sizes [1]."
    )
    rep = verify_claims(answer=answer, papers=papers)
    assert rep.verified_ratio == 1.0
