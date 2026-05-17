"""Web search tool — external grounding via Tavily/DuckDuckGo.

Wraps the platform's existing WebSearchAdapter so the assistant can pull
context that isn't yet in the user's paper corpus (release notes,
documentation, breaking research news, repository links). Results are
labelled as low-trust in the synthesizer prompt to keep grounding honest.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.adapters.web_search import get_web_search_adapter
from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult

log = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)


class WebSearchOutput(BaseModel):
    results: list[dict]
    total: int
    provider: str


class WebSearchTool:
    """External web search for context outside the paper corpus."""

    name = "web_search"
    summary = (
        "External web search via Tavily (when key configured) or DuckDuckGo "
        "(free fallback). Returns titles + URLs + snippets — does not persist "
        "anything. Use when the user needs context that lives outside arXiv "
        "(release notes, documentation, repos, blog posts) or when the corpus "
        "doesn't have enough material to answer. Treat results as low-trust."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = WebSearchInput
    output_schema = WebSearchOutput

    async def run(self, ctx: ToolContext, params: WebSearchInput) -> ToolResult:
        adapter = get_web_search_adapter()
        await ctx.emit_progress(30, f"Searching the web via {adapter.provider_id}")
        try:
            results = await adapter.search(params.query, max_results=params.max_results)
        except Exception as exc:
            log.warning("web_search failed: %s", exc)
            return ToolResult(
                output={"results": [], "total": 0, "provider": adapter.provider_id},
                summary=f"Web search failed ({type(exc).__name__})",
            )
        rows = [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in results
        ]
        await ctx.emit_progress(100, f"Found {len(rows)} web results")
        return ToolResult(
            output={"results": rows, "total": len(rows), "provider": adapter.provider_id},
            summary=f"Web search returned {len(rows)} results via {adapter.provider_id}",
        )
