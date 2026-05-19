"""arXiv import tool — search + persist + embed + index."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.services.arxiv_import import ArxivImportService

log = logging.getLogger(__name__)


class ArxivImportInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    # Where new papers land in the user's feed (defaults to active namespace).
    namespace_key: str | None = None
    # Search-side scope. Empty = cross-arXiv; only set when the user explicitly
    # scopes to specific categories like "scoped to cs.LG and stat.ML".
    namespace_keys: list[str] = Field(default_factory=list)
    max_results: int = Field(default=6, ge=1, le=20)


class ArxivImportOutput(BaseModel):
    imported: int
    skipped: int
    paper_ids: list[str]
    arxiv_results: list[dict]
    widened: bool = False


class ArxivImportTool:
    """Import arXiv papers into the active namespace (DB write + embeddings + graph)."""

    name = "arxiv_import"
    summary = (
        "Persisting arXiv search: searches arXiv via MCP, IMPORTS new "
        "candidates into the active namespace, EMBEDS abstracts, and "
        "INDEXES them in the knowledge graph. Side-effecting — grows the "
        "user's corpus.\n\n"
        "USE WHEN: the user wants to GROW their corpus ('add the latest "
        "papers on X to my feed', 'pull recent work on Y'), or when the "
        "orchestrator decides the corpus is too thin for synthesis.\n"
        "DO NOT USE WHEN: the user only wants to browse / preview "
        "(use arxiv_search — no persistence), already has enough corpus "
        "for a question (use deep_search), or wants a structured survey "
        "(literature_survey will call arxiv_import internally if needed).\n\n"
        "Pass namespace_keys=[] (default) for cross-arXiv search. Inputs: "
        "query, namespace_key (where new papers land), max_results. "
        "Output: imported count + paper_ids + arxiv_results — surfaced in "
        "the Grounded grid via the coverage-guard fallback."
    )
    cost_class = "moderate"
    side_effects = True
    cancellable = True
    streamable = True
    input_schema = ArxivImportInput
    output_schema = ArxivImportOutput

    async def run(self, ctx: ToolContext, params: ArxivImportInput) -> ToolResult:
        target_ns = params.namespace_key or ctx.namespace_key
        # Honour explicit search-scope; otherwise search all of arXiv so
        # interdisciplinary queries (e.g. molecular GNNs in q-bio + cs.LG)
        # aren't filtered out by a single-category lock-in.
        ns_keys = params.namespace_keys or None
        await ctx.emit_progress(20, "Searching arXiv MCP for candidates")
        svc = ArxivImportService(ctx.db)
        new_papers, skipped, arxiv_results = await svc.import_search_results(
            params.query,
            namespace_key=target_ns,
            namespace_keys=ns_keys,
            max_results=params.max_results,
        )
        widened = False
        # If a narrow scope returned nothing, widen automatically rather than
        # making the user re-prompt with an unscoped query.
        if not arxiv_results and ns_keys:
            await ctx.emit_progress(55, "No matches in scope — widening to all arXiv")
            new_papers, skipped, arxiv_results = await svc.import_search_results(
                params.query,
                namespace_key=target_ns,
                namespace_keys=None,
                max_results=params.max_results,
            )
            widened = bool(arxiv_results)
        paper_ids = [str(p.id) for p in new_papers]
        suffix = " (widened to all arXiv)" if widened else ""
        await ctx.emit_progress(100, f"Imported {len(new_papers)}; skipped {skipped} duplicate(s){suffix}")
        return ToolResult(
            output={
                "imported": len(new_papers),
                "skipped": skipped,
                "paper_ids": paper_ids,
                "arxiv_results": arxiv_results[:8],
                "widened": widened,
            },
            summary=f"Imported {len(new_papers)} arXiv papers ({skipped} dupes skipped){suffix}",
            citations=paper_ids,
        )
