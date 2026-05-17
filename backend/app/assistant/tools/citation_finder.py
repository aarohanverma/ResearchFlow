"""Citation Finder tool — find the best papers to cite for a specific claim.

Given a claim or assertion, searches the corpus and arXiv to find papers
that best support, refute, or contextualize it. Returns ranked papers with
relevance notes. Useful when writing and the user needs to know what to cite.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)


class CitationFinderInput(BaseModel):
    claim: str = Field(min_length=5, max_length=500, description="The claim, assertion, or statement that needs to be cited")
    relationship: str = Field(
        default="support",
        description="'support' (papers that back the claim), 'refute' (papers that challenge it), or 'context' (papers that provide background)",
    )
    limit: int = Field(default=6, ge=1, le=12)
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class CitationFinderOutput(BaseModel):
    papers: list[dict]
    claim: str
    total: int


class CitationFinderTool:
    """Find best papers to cite for a specific claim or assertion."""

    name = "citation_finder"
    summary = (
        "Find the most relevant papers to cite for a specific claim or statement. "
        "Searches the user's corpus + arXiv for papers that support, refute, or "
        "contextualize the claim. Use when: 'What should I cite for X?', "
        "'Find references for the claim that Y', 'which papers support Z?', "
        "'I need citations for [assertion]'. Returns papers ranked by relevance "
        "with a note on why each is a good citation for the claim."
    )
    cost_class = "moderate"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = CitationFinderInput
    output_schema = CitationFinderOutput

    async def run(self, ctx: ToolContext, params: CitationFinderInput) -> ToolResult:
        await ctx.emit_progress(15, f"Finding citations for: {params.claim[:60]}")

        # Step 1: Search corpus via deep_search-style retrieval
        corpus_papers: list[dict] = []
        try:
            from app.api.v1.search import _run_deep_search
            from app.db.session import async_session_factory

            async with async_session_factory() as db:
                ds_result = await _run_deep_search(
                    query=params.claim,
                    user_id=ctx.user_id,
                    namespace_keys=[ctx.namespace_key] if ctx.namespace_key else None,
                    include_arxiv_mcp=True,
                    arxiv_max_results=8,
                    db=db,
                )
                for p in (ds_result.results or [])[:params.limit]:
                    paper_dict = p.model_dump() if hasattr(p, "model_dump") else dict(p)
                    corpus_papers.append(paper_dict)
        except Exception as exc:
            log.warning("citation_finder: corpus search failed: %s", exc)

        if not corpus_papers:
            return ToolResult(
                output={"papers": [], "claim": params.claim, "total": 0},
                summary=f"No citation candidates found for: {params.claim[:60]}",
            )

        await ctx.emit_progress(65, f"Ranking {len(corpus_papers)} candidate(s)…")

        # Step 2: LLM ranking — pick best citations and explain why
        candidates_text = "\n\n".join(
            f"[{i+1}] Title: {p.get('title', '')}\n"
            f"Authors: {', '.join((p.get('authors') or [])[:3])}\n"
            f"Year: {p.get('year', 'N/A')}\n"
            f"Abstract: {(p.get('tldr') or p.get('abstract') or '')[:400]}"
            for i, p in enumerate(corpus_papers)
        )

        rel_instruction = {
            "support": "papers that provide evidence FOR the claim",
            "refute": "papers that challenge or contradict the claim",
            "context": "papers that provide relevant background or context for the claim",
        }.get(params.relationship, "papers most relevant to the claim")

        prompt = (
            f"You are helping a researcher find citations. Select the best papers ({rel_instruction}).\n\n"
            f"CLAIM: \"{params.claim}\"\n\n"
            f"CANDIDATE PAPERS:\n{candidates_text}\n\n"
            f"Instructions:\n"
            f"• Select up to {min(params.limit, len(corpus_papers))} papers that best serve as citations for this claim.\n"
            f"• For each selected paper, output:\n"
            f"  INDEX: [number]\n"
            f"  RELEVANCE: [1-2 sentences explaining WHY this paper is a good citation for the claim]\n"
            f"• Rank from most to least relevant.\n"
            f"• If a paper is not relevant at all, skip it.\n\n"
            f"Selected citations:"
        )

        ranked_indices: list[int] = []
        relevance_notes: dict[int, str] = {}

        try:
            from app.adapters.llm import get_llm_adapter
            llm = get_llm_adapter()
            res = await llm.complete(
                [{"role": "user", "content": prompt}],
                llm.quality_model,
                max_tokens=800,
                temperature=0.0,
            )
            raw = res.text or ""
            # Parse "INDEX: N" lines
            import re
            current_idx: int | None = None
            for line in raw.splitlines():
                line = line.strip()
                idx_match = re.match(r"INDEX:\s*\[?(\d+)\]?", line, re.IGNORECASE)
                rel_match = re.match(r"RELEVANCE:\s*(.+)", line, re.IGNORECASE)
                if idx_match:
                    current_idx = int(idx_match.group(1)) - 1  # 0-based
                    if 0 <= current_idx < len(corpus_papers):
                        ranked_indices.append(current_idx)
                elif rel_match and current_idx is not None:
                    relevance_notes[current_idx] = rel_match.group(1).strip()
        except Exception as exc:
            log.warning("citation_finder: LLM ranking failed: %s — returning unranked", exc)
            ranked_indices = list(range(len(corpus_papers)))

        # Fall back to original order if parsing produced nothing
        if not ranked_indices:
            ranked_indices = list(range(len(corpus_papers)))

        # Deduplicate and assemble results
        seen: set[int] = set()
        papers_out: list[dict] = []
        for idx in ranked_indices:
            if idx in seen or idx >= len(corpus_papers):
                continue
            seen.add(idx)
            p = corpus_papers[idx]
            papers_out.append({
                "title": p.get("title", ""),
                "authors": p.get("authors", [])[:4],
                "year": p.get("year"),
                "abstract": (p.get("tldr") or p.get("abstract") or "")[:400],
                "paper_id": p.get("paper_id") or p.get("id"),
                "arxiv_id": p.get("arxiv_id") or p.get("external_id"),
                "relevance_note": relevance_notes.get(idx, ""),
                "source": "corpus",
            })

        await ctx.emit_progress(100, f"Found {len(papers_out)} citation candidates")

        if not papers_out:
            return ToolResult(
                output={"papers": [], "claim": params.claim, "total": 0},
                summary=f"No relevant citations found for: {params.claim[:60]}",
            )

        return ToolResult(
            output={"papers": papers_out, "claim": params.claim, "total": len(papers_out)},
            summary=(
                f"{len(papers_out)} citation candidates for '{params.claim[:60]}' "
                f"(top: '{papers_out[0]['title'][:50]}')"
            ),
        )


citation_finder_tool = CitationFinderTool()
