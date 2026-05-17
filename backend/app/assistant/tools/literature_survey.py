"""Literature survey tool — structured survey of a research area.

Combines corpus deep search, arXiv frontier scan, and LLM synthesis to
produce a structured literature survey: key themes, leading methods,
open problems, contradictions, and research gaps. Grounded in actual papers.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult

log = logging.getLogger(__name__)


class LiteratureSurveyInput(BaseModel):
    query: str = Field(min_length=3, max_length=500, description="Research area or question to survey")
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)
    depth: str = Field(default="medium", description="shallow | medium | deep")


class LiteratureSurveyOutput(BaseModel):
    survey: str
    themes: list[str]
    open_problems: list[str]
    paper_count: int


class LiteratureSurveyTool:
    """Produce a structured literature survey combining corpus + arXiv search + LLM synthesis."""

    name = "literature_survey"
    summary = (
        "Produce a structured literature survey of a research area. Searches the user's "
        "corpus and arXiv, then synthesizes key themes, leading methods, open problems, "
        "contradictions, and research gaps into a coherent report with citations. Use when "
        "the user wants a comprehensive overview of a field, 'what's the state of the art', "
        "or a 'related work'-style survey."
    )
    cost_class = "heavy"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = LiteratureSurveyInput
    output_schema = LiteratureSurveyOutput

    async def run(self, ctx: ToolContext, params: LiteratureSurveyInput) -> ToolResult:
        await ctx.emit_progress(10, f"Starting literature survey: {params.query[:60]}")
        ns_keys = params.namespace_keys or ([params.namespace_key] if params.namespace_key else [])

        # Step 1: search corpus via deep search
        corpus_papers: list[dict] = []
        try:
            import uuid as _uuid
            from app.api.v1.search import _run_deep_search
            from app.db.session import async_session_factory
            async with async_session_factory() as _db:
                ds_result = await _run_deep_search(
                    job_id=str(_uuid.uuid4()),
                    query=params.query,
                    namespace_keys=ns_keys or None,
                    limit=15,
                    db=_db,
                    include_arxiv_mcp=False,
                )
            corpus_papers = [
                {
                    "title": p.title if hasattr(p, "title") else (p.get("title") if isinstance(p, dict) else ""),
                    "abstract": (
                        (p.tldr if hasattr(p, "tldr") else p.get("tldr") or "") or
                        (p.abstract if hasattr(p, "abstract") else p.get("abstract") or "")
                    )[:600],
                    "authors": (p.authors if hasattr(p, "authors") else p.get("authors")) or [],
                    "year": p.year if hasattr(p, "year") else p.get("year"),
                }
                for p in (ds_result.results or [])
            ]
        except Exception as exc:
            log.warning("literature_survey corpus search failed: %s", exc)

        await ctx.emit_progress(40, f"Corpus search done ({len(corpus_papers)} papers)")

        if await ctx.should_cancel():
            return ToolResult(output={"survey": "", "themes": [], "open_problems": [], "paper_count": 0},
                              summary="Cancelled")

        # Step 2: arXiv search for frontier papers
        arxiv_papers: list[dict] = []
        try:
            from app.adapters.sources.arxiv_mcp import ArXivMcpSource
            results = await ArXivMcpSource().search(params.query, max_results=8)
            arxiv_papers = [
                {
                    "title": r.title,
                    "abstract": (r.abstract or "")[:600],
                    "authors": r.authors or [],
                    "arxiv_id": r.external_id,
                    "source_url": r.source_url,
                }
                for r in (results or [])
            ]
        except Exception as exc:
            log.debug("literature_survey arXiv search failed: %s", exc)

        await ctx.emit_progress(60, "Synthesizing survey…")

        all_papers = corpus_papers + arxiv_papers
        # If no papers from either source, fall through to LLM synthesis using
        # training knowledge — return a clearly labelled knowledge-only survey
        # rather than an unhelpful "no papers found" message.
        knowledge_only = not all_papers

        # Step 3: LLM synthesis
        depth_instruction = {
            "shallow": "Produce a 3-4 paragraph high-level overview. Keep it concise.",
            "medium": "Produce a 6-8 paragraph structured survey covering themes, methods, contradictions, and gaps.",
            "deep": "Produce a thorough multi-section survey (10+ paragraphs) covering all themes, key debates, methodological comparison, open problems, and future directions.",
        }.get(params.depth, "medium")

        corpus_blob = "\n\n".join(
            f"[{i+1}] {p['title']}\nAuthors: {', '.join(p['authors'])}\nAbstract: {p['abstract']}"
            for i, p in enumerate(corpus_papers[:12])
        )
        arxiv_blob = "\n\n".join(
            f"[A{i+1}] {p['title']}\nAuthors: {', '.join(p['authors'])}\nAbstract: {p['abstract']}"
            for i, p in enumerate(arxiv_papers[:6])
        )

        if knowledge_only:
            prompt = (
                f"You are conducting a structured literature survey on: {params.query}\n\n"
                "NOTE: No papers were retrieved from the external corpus or arXiv for this query. "
                "Write the survey based on your training knowledge. Clearly state at the top that "
                "this survey is based on training knowledge and no external papers were retrieved. "
                "Be accurate and analytical; clearly mark speculative or uncertain claims.\n\n"
                f"{depth_instruction}\n\n"
                "Structure the survey as:\n"
                "1. **Overview** — what this area is about and why it matters\n"
                "2. **Key Themes & Approaches** — cluster methods by paradigm\n"
                "3. **Leading Methods** — compare dominant approaches with trade-offs\n"
                "4. **Contradictions & Debates** — where the literature disagrees\n"
                "5. **Open Problems & Gaps** — what remains unsolved\n"
                "6. **Frontier & Emerging Directions** — most recent developments\n\n"
                "Write the survey now:"
            )
        else:
            prompt = (
                f"You are conducting a structured literature survey on: {params.query}\n\n"
                f"CORPUS PAPERS (indexed, primary evidence):\n{corpus_blob or '(none)'}\n\n"
                f"ARXIV CANDIDATES (frontier, lightly vetted):\n{arxiv_blob or '(none)'}\n\n"
                f"{depth_instruction}\n\n"
                "Structure the survey as:\n"
                "1. **Overview** — what this area is about and why it matters\n"
                "2. **Key Themes & Approaches** — cluster papers by methodology or paradigm\n"
                "3. **Leading Methods** — compare dominant approaches with trade-offs\n"
                "4. **Contradictions & Debates** — where the literature disagrees\n"
                "5. **Open Problems & Gaps** — what remains unsolved\n"
                "6. **Frontier & Emerging Directions** — most recent developments\n\n"
                "Cite papers inline as [1], [2], ... (corpus) or [A1], [A2], ... (arXiv).\n"
                "Every claim must be tied to a citation. Be analytical, not just descriptive.\n"
                "Write the survey now:"
            )

        try:
            from app.adapters.llm import get_llm_adapter
            llm = get_llm_adapter()
            max_tok = {"shallow": 2000, "medium": 4000, "deep": 6000}.get(params.depth, 4000)
            res = await llm.complete(
                [{"role": "user", "content": prompt}],
                llm.reasoning_model,
                max_tokens=max_tok,
                temperature=None,
                reasoning_effort="low",
            )
            survey_text = res.text.strip()
        except Exception as exc:
            log.warning("literature_survey LLM failed: %s", exc)
            survey_text = f"Survey synthesis failed: {exc}"

        # Extract themes and open problems (simple heuristic — parse bold/bullet headings)
        themes: list[str] = []
        open_problems: list[str] = []
        for line in survey_text.splitlines():
            stripped = line.strip().lstrip("•-*#").strip()
            if "open problem" in line.lower() or "gap" in line.lower() or "unsolved" in line.lower():
                if stripped and len(stripped) < 120:
                    open_problems.append(stripped[:120])
            elif "approach" in line.lower() or "method" in line.lower() or "paradigm" in line.lower():
                if stripped and len(stripped) < 120:
                    themes.append(stripped[:120])

        await ctx.emit_progress(100, f"Literature survey complete ({len(all_papers)} papers)")
        survey_summary = (
            "Literature survey: training-knowledge-based (no external papers retrieved)"
            if knowledge_only
            else f"Literature survey: {len(corpus_papers)} corpus + {len(arxiv_papers)} arXiv papers synthesized"
        )
        return ToolResult(
            output={
                "survey": survey_text,
                "themes": themes[:8],
                "open_problems": open_problems[:6],
                "paper_count": len(all_papers),
            },
            summary=survey_summary,
        )


literature_survey_tool = LiteratureSurveyTool()
