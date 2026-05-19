"""Literature survey tool — structured survey of a research area.

Design (post-redesign May 2026)
-------------------------------

Earlier versions of this tool ran deep_search for corpus retrieval but
fell back to a raw ``ArXivMcpSource().search()`` keyword scan for the
"frontier" arm. arXiv's keyword search is recall-heavy and recency-
ordered — it returned visually-recent-but-topically-off papers (e.g.
an "agentic AI" query surfacing cs.RO/cs.DC papers that shared only
the words "agent" or "tool"). The synthesizer then dutifully cited
those off-topic papers, dragging the whole survey off track.

The new design treats deep_search as the canonical retrieval pipeline
(which already does validate → LLM rewrite → exact/fuzzy cache →
semantic + keyword + graph-concept retrieval → RRF fusion → LLM rerank)
and only uses arXiv import as a corpus-growth step when deep_search
returns too few results. arXiv MCP is never used as a synthesis source
directly anymore — anything from arXiv first lands in the corpus, gets
embedded + indexed + scored, and is then retrieved through the same
deep_search pipeline as anything else.

Pipeline
--------

    Step 1.  deep_search(query) on the existing corpus, scoped to the
             active namespace(s).
    Step 2.  If deep_search returned <3 papers, run arxiv_import to
             grow the corpus, then deep_search again.
    Step 3.  Semantic relevance gate against the *literal* user query
             — drops anything below the cosine-similarity threshold.
             A short LLM-judge pass then verifies topical fit for the
             surviving candidates and removes hard off-topic outliers.
    Step 4.  LLM synthesis on the surviving papers only. The synthesis
             prompt uses one citation scheme [1]..[N] and is instructed
             to never invent claims that the cited paper doesn't make.

Output
------

``papers`` carries the surviving filtered candidates so the orchestrator
renders them in the Grounded grid, matching what the synthesis actually
cited.
"""

from __future__ import annotations

import logging
import math
import uuid as _uuid

from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult

log = logging.getLogger(__name__)

# Semantic similarity threshold (query ↔ paper) below which a candidate
# is treated as off-topic. Raised from 0.32 → 0.40 after the redesign:
# now that retrieval all flows through deep_search, the surviving tail
# is already higher-quality, so the filter can afford to be stricter.
_MIN_SURVEY_PAPER_SIMILARITY = 0.40
_MIN_KEEP_AFTER_FILTER = 2          # never collapse to zero
_THIN_CORPUS_TRIGGER = 3            # below this, grow corpus via arxiv_import
_MAX_PAPERS_FOR_SYNTHESIS = 12      # cap context size for the synthesis LLM
_DEEP_SEARCH_LIMIT = 15             # initial deep_search recall budget


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


async def _semantic_relevance_filter(query: str, papers: list[dict]) -> list[dict]:
    """Drop papers whose title+abstract is semantically far from the query.

    Falls back to the input list when the embedding adapter is unavailable
    so the survey path never breaks if embeddings are offline.
    """
    if not papers or not (query or "").strip():
        return papers
    try:
        from app.adapters.embedding import get_embedding_adapter
        embed = get_embedding_adapter()
        q_vec = await embed.embed_query(query)
        texts = [
            (f"{(p.get('title') or '').strip()}. "
             f"{(p.get('abstract') or p.get('tldr') or '').strip()}")[:1200]
            or (p.get('title') or 'untitled')
            for p in papers
        ]
        vecs = await embed.embed_texts(texts, task_type="SEMANTIC_SIMILARITY")
    except Exception as exc:
        log.debug("literature_survey: relevance filter unavailable: %s", exc)
        return papers
    scored: list[tuple[float, dict]] = []
    for p, v in zip(papers, vecs or []):
        if not v:
            continue
        sim = _cosine(q_vec, v)
        annotated = dict(p)
        annotated["query_similarity"] = round(sim, 4)
        scored.append((sim, annotated))
    if not scored:
        return papers
    scored.sort(key=lambda t: t[0], reverse=True)
    kept = [p for sim, p in scored if sim >= _MIN_SURVEY_PAPER_SIMILARITY]
    if len(kept) < _MIN_KEEP_AFTER_FILTER:
        kept = [p for _sim, p in scored[:_MIN_KEEP_AFTER_FILTER]]
    log.info(
        "literature_survey: semantic filter kept %d/%d (top_sim=%.2f, min=%.2f)",
        len(kept), len(papers),
        scored[0][0] if scored else 0.0,
        _MIN_SURVEY_PAPER_SIMILARITY,
    )
    return kept


async def _llm_topic_judge(query: str, papers: list[dict]) -> list[dict]:
    """Cheap-model gate that removes hard off-topic outliers.

    Even after the embedding filter, occasional adjacent-vocabulary papers
    sneak through. We ask the cheap model: for each paper, is it directly
    about the query topic (yes) or only tangentially related (no)? Anything
    rated "no" is dropped. We always keep at least the top-N by semantic
    similarity so the survey never collapses if the judge is over-strict.
    """
    if len(papers) <= 2:
        return papers
    try:
        import json
        from app.adapters.llm import get_llm_adapter
        llm = get_llm_adapter()
        listing = "\n".join(
            f"[{i+1}] {(p.get('title') or 'untitled')[:200]} — "
            f"{(p.get('abstract') or p.get('tldr') or '')[:280]}"
            for i, p in enumerate(papers)
        )
        prompt = (
            "You are filtering candidate papers for a literature survey.\n\n"
            f"Survey topic / query: {query!r}\n\n"
            "For each paper below, decide if it is DIRECTLY about the survey "
            "topic (the paper's central contribution or core method is in this "
            "area) versus only SHALLOWLY related (shares some vocabulary but is "
            "really about a different problem).\n\n"
            f"PAPERS:\n{listing}\n\n"
            "Return strict JSON: an array of objects, one per paper, in input "
            "order, each like: {\"i\": <1-based index>, \"keep\": true|false, "
            "\"reason\": \"<≤12 words>\"}. Be strict: when in doubt, KEEP — only "
            "drop papers that are unmistakably off-topic."
        )
        res = await llm.complete(
            [{"role": "user", "content": prompt}],
            llm.cheap_model,
            max_tokens=600,
            temperature=0.1,
        )
        raw = (res.text or "").strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[:-3]
        # Tolerate object-wrapped responses: {"results": [...]}
        parsed = json.loads(raw) if raw.startswith("[") else json.loads(raw).get("results", [])
        keep_idx = {
            int(item["i"]) - 1
            for item in parsed
            if isinstance(item, dict) and item.get("keep") is True
        }
    except Exception as exc:
        log.debug("literature_survey: topic-judge unavailable: %s", exc)
        return papers
    if not keep_idx:
        # Judge said nothing was relevant — fall back to top-2 by similarity
        # rather than emit an empty survey.
        return papers[:2]
    kept = [p for i, p in enumerate(papers) if i in keep_idx]
    log.info("literature_survey: topic-judge kept %d/%d", len(kept), len(papers))
    return kept


def _normalize_paper(p) -> dict:
    """Coerce a deep_search result (DeepSearchResult or dict) to a uniform dict."""
    if isinstance(p, dict):
        d = dict(p)
    else:
        # Pydantic-style object — pull common fields explicitly so we don't
        # depend on .model_dump being a valid coroutine in some adapters.
        d = {
            "paper_id": getattr(p, "paper_id", None),
            "title": getattr(p, "title", "") or "",
            "abstract": getattr(p, "abstract", "") or "",
            "tldr": getattr(p, "tldr", "") or "",
            "authors": list(getattr(p, "authors", None) or []),
            "year": getattr(p, "year", None),
            "arxiv_id": getattr(p, "arxiv_id", None) or getattr(p, "external_id", None),
            "source_url": getattr(p, "source_url", None),
            "search_score": getattr(p, "search_score", None),
            "relevance_score": getattr(p, "relevance_score", None),
            "namespace_key": getattr(p, "namespace_key", None),
            "key_concepts": list(getattr(p, "key_concepts", None) or []),
        }
    # Truncate any oversized abstracts so the synthesis prompt stays bounded.
    if d.get("abstract"):
        d["abstract"] = d["abstract"][:800]
    return d


class LiteratureSurveyInput(BaseModel):
    query: str = Field(
        min_length=3, max_length=500,
        description="Research area or question to survey. Should be a focused "
                    "topic (e.g. 'sample-efficient RL for robotics manipulation') "
                    "rather than an open-ended question.",
    )
    namespace_key: str = Field(
        default="",
        description="Single namespace to scope retrieval (e.g. 'cs.AI'). Use "
                    "namespace_keys when scoping to multiple.",
    )
    namespace_keys: list[str] = Field(
        default_factory=list,
        description="Multi-namespace scope. Empty means 'use namespace_key' "
                    "or fall back to all indexed papers.",
    )
    depth: str = Field(
        default="medium",
        description="shallow (3-4 paragraphs) | medium (6-8 paragraphs) | "
                    "deep (10+ paragraphs)",
    )


class LiteratureSurveyOutput(BaseModel):
    survey: str = Field(description="The synthesized markdown survey")
    themes: list[str] = Field(default_factory=list)
    open_problems: list[str] = Field(default_factory=list)
    paper_count: int = Field(description="Number of papers the survey is grounded on")
    # Papers that survived all relevance gates — surfaced so the orchestrator
    # renders them in the Grounded grid alongside the survey text.
    papers: list[dict] = Field(default_factory=list)


class LiteratureSurveyTool:
    """Structured literature survey grounded in retrieval + LLM synthesis.

    Retrieval flows through ``deep_search`` (the same pipeline used for
    on-demand search) so the surveyed corpus is always the highest-quality
    available subset for the query. If the local corpus is thin,
    ``arxiv_import`` grows it before re-running deep_search — frontier
    coverage and corpus indexing share a single path.
    """

    name = "literature_survey"
    summary = (
        "Produce a structured literature survey of a focused research area. "
        "Retrieves via the deep_search pipeline (semantic + keyword + graph + "
        "LLM rerank), grows the corpus from arXiv when too few results are "
        "indexed, applies a two-stage relevance gate (embedding + LLM topic "
        "judge), then synthesizes themes, methods, contradictions, gaps, and "
        "frontier directions with strict per-claim citations.\n\n"
        "USE WHEN: the user wants a structured overview / state-of-the-art / "
        "'related work'-style report on a SPECIFIC area.\n"
        "DO NOT USE WHEN: the user wants a single concept explained "
        "(use concept_explain), wants raw retrieval (use deep_search), or "
        "wants frontier-only recency (use frontier_scan).\n\n"
        "Inputs: query (focused topic, 3-500 chars), namespace scope, depth. "
        "Outputs: markdown survey, themes list, open_problems list, the "
        "grounded papers list (one citation index per surviving paper)."
    )
    cost_class = "heavy"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = LiteratureSurveyInput
    output_schema = LiteratureSurveyOutput

    async def run(self, ctx: ToolContext, params: LiteratureSurveyInput) -> ToolResult:
        await ctx.emit_progress(5, f"Literature survey: {params.query[:60]}")
        ns_keys = params.namespace_keys or (
            [params.namespace_key] if params.namespace_key else []
        )

        from app.api.v1.search import _run_deep_search
        from app.db.session import async_session_factory

        # ── Step 1: deep_search on existing corpus ─────────────────────
        await ctx.emit_progress(15, "Searching indexed corpus…")
        ds_results: list[dict] = []
        rewritten_query = params.query
        try:
            async with async_session_factory() as db:
                ds = await _run_deep_search(
                    job_id=str(_uuid.uuid4()),
                    query=params.query,
                    namespace_keys=ns_keys or None,
                    limit=_DEEP_SEARCH_LIMIT,
                    db=db,
                    include_arxiv_mcp=False,
                )
                rewritten_query = ds.rewritten_query or params.query
                ds_results = [_normalize_paper(p) for p in (ds.results or [])]
        except Exception as exc:
            log.warning("literature_survey: initial deep_search failed: %s", exc)

        if await ctx.should_cancel():
            return ToolResult(
                output={"survey": "", "themes": [], "open_problems": [],
                        "paper_count": 0, "papers": []},
                summary="Cancelled",
            )

        # ── Step 2: corpus-growth via arxiv_import if thin ─────────────
        grew = False
        if len(ds_results) < _THIN_CORPUS_TRIGGER:
            await ctx.emit_progress(
                30,
                f"Corpus is thin ({len(ds_results)}) — importing from arXiv…",
            )
            try:
                from app.assistant.tools.arxiv_import import (
                    ArxivImportInput, ArxivImportTool,
                )
                import_tool = ArxivImportTool()
                # Use the deep_search-rewritten query for arXiv import so we
                # benefit from the same query expansion deep_search did.
                import_result = await import_tool.run(
                    ctx,
                    ArxivImportInput(
                        query=rewritten_query,
                        namespace_key=ns_keys[0] if ns_keys else None,
                        # Empty list = cross-arXiv; matches arxiv_import default.
                        namespace_keys=[],
                        max_results=12,
                    ),
                )
                imported_n = int(import_result.output.get("imported", 0))
                grew = imported_n > 0
                if grew:
                    await ctx.emit_progress(
                        50, f"Imported {imported_n} arXiv papers; re-searching…",
                    )
                    # Re-run deep_search now that the corpus has grown. Use a
                    # fresh DB session so cached writes from import are visible.
                    async with async_session_factory() as db:
                        ds2 = await _run_deep_search(
                            job_id=str(_uuid.uuid4()),
                            query=params.query,
                            namespace_keys=ns_keys or None,
                            limit=_DEEP_SEARCH_LIMIT,
                            db=db,
                            include_arxiv_mcp=False,
                        )
                        ds_results = [
                            _normalize_paper(p) for p in (ds2.results or [])
                        ]
            except Exception as exc:
                log.warning("literature_survey: corpus-growth path failed: %s", exc)

        await ctx.emit_progress(
            65, f"Retrieval done ({len(ds_results)} candidates)…",
        )

        # ── Step 3: two-stage relevance gate ────────────────────────────
        gated = await _semantic_relevance_filter(params.query, ds_results)
        if len(gated) > 3:
            gated = await _llm_topic_judge(params.query, gated)
        gated = gated[:_MAX_PAPERS_FOR_SYNTHESIS]

        await ctx.emit_progress(80, "Synthesizing survey…")

        knowledge_only = not gated

        # ── Step 4: LLM synthesis ───────────────────────────────────────
        depth_instruction = {
            "shallow": "Produce a 3-4 paragraph high-level overview. Keep it concise.",
            "medium": "Produce a 6-8 paragraph structured survey covering themes, methods, contradictions, and gaps.",
            "deep": "Produce a thorough multi-section survey (10+ paragraphs) covering all themes, key debates, methodological comparison, open problems, and future directions.",
        }.get(params.depth, "medium")

        paper_blob = "\n\n".join(
            f"[{i+1}] {(p.get('title') or '').strip()}\n"
            f"Authors: {', '.join((p.get('authors') or [])[:6])}\n"
            f"Abstract: {(p.get('abstract') or p.get('tldr') or '').strip()}"
            for i, p in enumerate(gated)
        )

        if knowledge_only:
            prompt = (
                f"You are conducting a structured literature survey on: {params.query}\n\n"
                "NOTE: No relevant papers were retrieved from the corpus or arXiv. "
                "Write the survey based on your training knowledge. State clearly "
                "at the top that no external papers were retrieved. Be analytical; "
                "explicitly mark speculative claims with 'tentatively' or "
                "'commonly reported'.\n\n"
                f"{depth_instruction}\n\n"
                "Structure:\n"
                "1. **Overview** — what this area is and why it matters\n"
                "2. **Key Themes & Approaches**\n"
                "3. **Leading Methods**\n"
                "4. **Contradictions & Debates**\n"
                "5. **Open Problems & Gaps**\n"
                "6. **Frontier & Emerging Directions**\n\n"
                "FORMATTING:\n"
                "- Plain markdown only. No LaTeX (\\rightarrow, \\text, $...$). "
                "Use Unicode arrows (→) and inline code for symbols where useful.\n"
                "- No tables of citation markers since there are no papers.\n\n"
                "Write the survey now:"
            )
        else:
            prompt = (
                f"You are conducting a structured literature survey on: {params.query}\n\n"
                "PAPERS (the ONLY allowed citation pool — do not invent or cite "
                "anything outside this list):\n"
                f"{paper_blob}\n\n"
                f"{depth_instruction}\n\n"
                "Structure:\n"
                "1. **Overview** — what this area is and why it matters\n"
                "2. **Key Themes & Approaches** — cluster papers by methodology\n"
                "3. **Leading Methods** — compare with trade-offs\n"
                "4. **Contradictions & Debates** — where the cited papers disagree\n"
                "5. **Open Problems & Gaps** — what the cited papers explicitly leave unsolved\n"
                "6. **Frontier & Emerging Directions** — most recent developments in the cited set\n\n"
                "STRICT CITATION RULES:\n"
                "- Cite as [1], [2], … using ONLY the indices above.\n"
                "- Every non-trivial factual claim must end with at least one citation.\n"
                "- Never fabricate authors, titles, or results not supported by the "
                "abstracts above. If something isn't supported, say so explicitly "
                "instead of guessing.\n\n"
                "FORMATTING RULES:\n"
                "- Plain markdown only. NO LaTeX commands. Do NOT write `$\\rightarrow$`, "
                "`\\text{}`, `$...$`, or any backslash-prefixed sequences. Use Unicode "
                "arrows (→, ⇒) and plain words instead.\n"
                "- Headings use '##' (or '**bold**' inline). Bullets use '-'. "
                "Code spans use backticks.\n"
                "- Numerical results from the abstracts are fine; mark them with the "
                "appropriate citation.\n\n"
                "Write the survey now:"
            )

        survey_text = ""
        try:
            from app.adapters.llm import get_llm_adapter
            llm = get_llm_adapter()
            max_tok = {"shallow": 2000, "medium": 4000, "deep": 6000}.get(
                params.depth, 4000,
            )
            res = await llm.complete(
                [{"role": "user", "content": prompt}],
                llm.reasoning_model,
                max_tokens=max_tok,
                temperature=None,
                reasoning_effort="low",
            )
            survey_text = (res.text or "").strip()
        except Exception as exc:
            log.warning("literature_survey: synthesis LLM failed: %s", exc)
            survey_text = f"Survey synthesis failed: {exc}"

        # Lightweight theme / open-problem extraction so the orchestrator
        # can show structured chips alongside the prose.
        themes: list[str] = []
        open_problems: list[str] = []
        for line in survey_text.splitlines():
            stripped = line.strip().lstrip("•-*#").strip()
            if not stripped or len(stripped) > 120:
                continue
            ll = line.lower()
            if any(k in ll for k in ("open problem", "gap", "unsolved", "remains")):
                open_problems.append(stripped[:120])
            elif any(k in ll for k in ("approach", "method", "paradigm")):
                themes.append(stripped[:120])

        await ctx.emit_progress(
            100, f"Literature survey complete ({len(gated)} papers)",
        )

        if knowledge_only:
            survey_summary = (
                "Literature survey: training-knowledge-based "
                "(no relevant papers found in corpus/arXiv)"
            )
        else:
            survey_summary = (
                f"Literature survey: {len(gated)} papers synthesized "
                f"(corpus{'+arxiv' if grew else ''})"
            )

        surfaced: list[dict] = []
        for p in gated[:8]:
            entry = dict(p)
            # Make sure downstream sorts have something monotonic to read.
            if entry.get("search_score") is None and entry.get("relevance_score") is None:
                entry["relevance_score"] = float(entry.get("query_similarity") or 1.0)
            surfaced.append(entry)

        return ToolResult(
            output={
                "survey": survey_text,
                "themes": themes[:8],
                "open_problems": open_problems[:6],
                "paper_count": len(gated),
                "papers": surfaced,
            },
            summary=survey_summary,
            citations=[str(p.get("paper_id")) for p in surfaced if p.get("paper_id")],
        )


literature_survey_tool = LiteratureSurveyTool()
