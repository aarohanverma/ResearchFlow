"""Concept explanation tool — RAG-grounded definition + context for a term.

Distinct from deep_search: optimized for "what does X mean" / "explain Y"
intents where the user wants an unpacked definition with grounded examples
rather than a list of papers. Pulls 3-5 chunks from the user's corpus,
asks the quality LLM to write an expertise-tuned explanation, returns the
explanation + the supporting paper references.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult

log = logging.getLogger(__name__)


_EXPERTISE_HINT = {
    "newcomer": "Explain like I'm new — define jargon, give one concrete example, avoid math unless essential.",
    "practitioner": "Balance precision with intuition. Mention the canonical formulation if relevant.",
    "expert": "Be terse and technical; assume background. Highlight subtle distinctions and recent debates.",
}


class ConceptExplainInput(BaseModel):
    concept: str = Field(min_length=2, max_length=240,
                         description="The term/concept to explain (e.g. 'mixture of experts', 'instruction tuning').")
    namespace_keys: list[str] = Field(default_factory=list)
    context: str = Field(default="", max_length=500,
                         description="Optional surrounding context — e.g. why the user is asking.")


class ConceptExplainOutput(BaseModel):
    concept: str
    explanation: str
    supporting_paper_ids: list[str]
    supporting_papers: list[dict]
    mermaid: str | None = None      # Small concept map: central node → 3-5 related concepts


class ConceptExplainTool:
    """RAG-grounded concept explanation tuned to user expertise level."""

    name = "concept_explain"
    summary = (
        "Grounded explanation of a single concept/term/method using snippets from "
        "the user's corpus. Tuned to expertise level (newcomer/practitioner/expert). "
        "Use when the user asks 'what is X', 'explain Y', 'define Z' or wants a "
        "concept unpacked rather than a paper list. Returns the explanation plus "
        "the underlying paper references."
    )
    cost_class = "moderate"
    side_effects = False
    cancellable = True
    streamable = True
    input_schema = ConceptExplainInput
    output_schema = ConceptExplainOutput

    async def run(self, ctx: ToolContext, params: ConceptExplainInput) -> ToolResult:
        # Reuse deep_search's hybrid retrieval — it already covers keyword +
        # semantic + graph expansion + LLM rerank, which is overkill for a
        # 3-5 chunk RAG but gives strong grounding.
        from app.api.v1.search import _run_deep_search
        import uuid as _uuid

        ns_keys = params.namespace_keys or ctx.namespace_keys or [ctx.namespace_key]
        await ctx.emit_progress(20, f"Retrieving context for '{params.concept}'")
        try:
            res = await _run_deep_search(
                job_id=f"assistant-cx:{_uuid.uuid4()}",
                query=params.concept,
                namespace_keys=ns_keys,
                limit=5,
                db=ctx.db,
                include_arxiv_mcp=False,
                arxiv_max_results=0,
            )
        except Exception as exc:
            log.warning("concept_explain retrieval failed: %s", exc)
            res = None

        rows = (res.results if res else None) or []
        papers = [r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r) for r in rows[:5]]

        await ctx.emit_progress(60, "Composing grounded explanation")
        explanation = await _llm_explain(
            concept=params.concept,
            context=params.context,
            papers=papers,
            expertise=ctx.expertise_level,
            orientation=ctx.orientation,
        )
        paper_ids = [str(p.get("paper_id") or "") for p in papers if p.get("paper_id")]
        # Build a small concept map from the supporting papers' key_concepts.
        # Pure composition over already-extracted enrichment fields — no
        # extra LLM call so we can keep this cheap and always-on when
        # there are enough sources to make a meaningful diagram.
        await ctx.emit_progress(85, "Drafting concept map")
        mermaid = _build_concept_map(params.concept, papers)
        await ctx.emit_progress(100, f"Explained '{params.concept}' with {len(paper_ids)} sources")

        return ToolResult(
            output={
                "concept": params.concept,
                "explanation": explanation,
                "supporting_paper_ids": paper_ids,
                "supporting_papers": papers,
                "mermaid": mermaid,
            },
            summary=f"Explained '{params.concept}' grounded in {len(paper_ids)} paper(s)",
            citations=paper_ids,
        )


async def _llm_explain(
    *,
    concept: str,
    context: str,
    papers: list[dict],
    expertise: str,
    orientation: str,
) -> str:
    """LLM-generated explanation. Falls back to a short structural note when offline."""
    if not papers:
        return (
            f"I don't have grounded snippets for '{concept}' in your current corpus. "
            f"Try importing recent papers on the topic or broadening the namespace, "
            f"then ask again."
        )
    paper_block = "\n\n".join(
        f"[{i + 1}] {p.get('title')}\n{p.get('tldr') or (p.get('abstract') or '')[:600]}"
        for i, p in enumerate(papers)
    )
    hint = _EXPERTISE_HINT.get(expertise, _EXPERTISE_HINT["practitioner"])
    try:
        from app.adapters.llm import get_llm_adapter

        llm = get_llm_adapter()
        prompt = (
            f"Concept: {concept}\n"
            f"User profile: expertise={expertise}, orientation={orientation}. {hint}\n"
            f"Optional context: {context or '(none)'}\n\n"
            "Write a grounded explanation in 4-7 short paragraphs (or fewer for "
            "experts). Cite sources inline as [1], [2], etc. Distinguish "
            "'verified from sources' vs 'general background knowledge' explicitly. "
            "End with a 1-2 sentence 'common pitfalls' note when relevant.\n\n"
            f"Source snippets:\n{paper_block}"
        )
        res = await llm.complete(
            [{"role": "user", "content": prompt}],
            llm.quality_model,
            max_tokens=900,
            temperature=0.2,
        )
        return res.text.strip() or _fallback_explain(concept, papers)
    except Exception as exc:
        log.warning("concept_explain LLM fell back: %s", exc)
        return _fallback_explain(concept, papers)


def _fallback_explain(concept: str, papers: list[dict]) -> str:
    bullets = "\n".join(f"- [{i + 1}] {p.get('title')}" for i, p in enumerate(papers[:5]))
    return (
        f"### {concept}\n\n"
        f"Grounded sources retrieved (LLM synthesis unavailable):\n{bullets}\n\n"
        f"Open the papers above for the canonical definitions."
    )


def _build_concept_map(central: str, papers: list[dict]) -> str | None:
    """Compose a small mermaid concept map from extracted key_concepts.

    Centers ``central`` (the explained concept) and links it to the most
    common neighboring concepts across the supporting papers. Returns None
    when there's not enough signal to make a meaningful diagram (avoids
    rendering trivial single-node graphs that add noise).
    """
    from collections import Counter

    central_clean = (central or "").strip()
    if not central_clean or not papers:
        return None

    # Count how often each non-central concept co-occurs across the source
    # papers. Lowercase-fold for matching but keep first-seen casing for
    # display so titles like "Mixture of Experts" don't become "mixture of experts".
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for p in papers:
        for c in (p.get("key_concepts") or [])[:8]:
            label = str(c).strip()
            if not label or label.lower() == central_clean.lower():
                continue
            key = label.lower()
            counts[key] += 1
            display.setdefault(key, label[:40])
    top = [display[k] for k, _ in counts.most_common(6)]
    if len(top) < 2:
        # A two-edge minimum keeps the diagram from looking like a stub.
        return None

    safe_central = _mermaid_safe(central_clean[:40])
    lines = ["graph LR", f'  C(("{safe_central}"))']
    for i, label in enumerate(top, start=1):
        safe = _mermaid_safe(label)
        lines.append(f'  N{i}["{safe}"]')
        lines.append(f"  C --> N{i}")
    return "\n".join(lines)


def _mermaid_safe(text: str) -> str:
    """Escape characters mermaid's parser dislikes inside node labels."""
    return text.replace('"', "'").replace("\\", "/").replace("\n", " ").strip()
