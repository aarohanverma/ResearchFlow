"""arXiv search tool — read-only MCP search, no DB writes."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.services.arxiv_import import ArxivImportService

log = logging.getLogger(__name__)


class ArxivSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    # Empty list = cross-namespace search (any arXiv category). The
    # planner / RA defaults to this so molecular-GNN papers that live in
    # q-bio or physics.chem-ph aren't filtered out by a cs.AI-only scope.
    namespace_keys: list[str] = Field(default_factory=list)
    max_results: int = Field(default=10, ge=1, le=30)


class ArxivSearchOutput(BaseModel):
    results: list[dict]
    total: int
    widened: bool = False


class ArxivSearchTool:
    """Search arXiv via the official MCP server without importing anything."""

    name = "arxiv_search"
    summary = (
        "Search arXiv via the MCP server (with Atom-API fallback). Returns raw "
        "RawPaper rows; does not persist or embed. Use when the user wants to "
        "discover candidate papers before deciding what to import. "
        "Pass namespace_keys=[] (default) for cross-arXiv search — only set "
        "namespace_keys when the user explicitly scopes to specific categories."
    )
    cost_class = "moderate"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = ArxivSearchInput
    output_schema = ArxivSearchOutput

    async def run(self, ctx: ToolContext, params: ArxivSearchInput) -> ToolResult:
        await ctx.emit_progress(20, "Querying arXiv MCP")
        # Honour explicit namespace constraints; otherwise search all of arXiv.
        ns_keys = params.namespace_keys or None
        svc = ArxivImportService(ctx.db)
        results = await svc.search(
            params.query,
            namespace_keys=ns_keys,
            max_results=params.max_results,
        )
        widened = False
        # Fallback: when an explicit namespace returned nothing, widen
        # automatically so the user isn't punished for category guesswork.
        if not results and ns_keys:
            await ctx.emit_progress(60, "No matches in scope — widening to all arXiv")
            results = await svc.search(
                params.query,
                namespace_keys=None,
                max_results=params.max_results,
            )
            widened = bool(results)
        suffix = " (widened to all arXiv)" if widened else ""
        await ctx.emit_progress(100, f"arXiv returned {len(results)} candidates{suffix}")
        return ToolResult(
            output={"results": results, "total": len(results), "widened": widened},
            summary=f"arXiv MCP found {len(results)} candidate papers{suffix}",
        )
