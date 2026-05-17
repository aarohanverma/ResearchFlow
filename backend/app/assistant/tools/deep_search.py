"""Deep Search tool — wraps app.api.v1.search._run_deep_search.

Owns no logic of its own. Adapts the platform's deep-search pipeline (RRF +
graph expansion + LLM rerank + optional self-RAG check) to the assistant
tool contract so the orchestrator can compose it with other capabilities.
"""

from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult

log = logging.getLogger(__name__)


class DeepSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    namespace_keys: list[str] = Field(default_factory=list)
    limit: int = Field(default=8, ge=1, le=30)
    include_arxiv_mcp: bool = Field(
        default=True,
        description="Also surface arXiv candidates not yet in the corpus. Enables the orchestrator's coverage guard to import them if corpus results are thin.",
    )
    arxiv_max_results: int = Field(default=6, ge=0, le=20)


class DeepSearchOutput(BaseModel):
    papers: list[dict]
    total: int


class DeepSearchTool:
    """Hybrid retrieval over the user's corpus, optionally augmented with arXiv MCP."""

    name = "deep_search"
    summary = (
        "Hybrid retrieval (keyword + semantic + graph + LLM rerank) over the user's "
        "indexed paper corpus, augmented with arXiv MCP candidates not yet imported. "
        "Best first tool for any research question — searches beyond the current feed. "
        "If corpus results are sparse, the orchestrator auto-imports the top arXiv matches."
    )
    cost_class = "heavy"
    side_effects = False
    cancellable = True
    streamable = True
    input_schema = DeepSearchInput
    output_schema = DeepSearchOutput

    async def run(self, ctx: ToolContext, params: DeepSearchInput) -> ToolResult:
        from app.api.v1.search import (
            _apply_orientation_nudge,
            _dedup_results_by_external_id,
            _run_deep_search,
        )

        await ctx.emit_progress(15, "Running hybrid retrieval")
        ns_keys = params.namespace_keys or ctx.namespace_keys or [ctx.namespace_key]
        result = await _run_deep_search(
            job_id=f"assistant-ds:{uuid.uuid4()}",
            query=params.query,
            namespace_keys=ns_keys,
            limit=params.limit,
            db=ctx.db,
            include_arxiv_mcp=params.include_arxiv_mcp,
            arxiv_max_results=params.arxiv_max_results,
        )
        rows = result.results or []
        raw = [r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r) for r in rows]
        await ctx.emit_progress(75, "Applying orientation nudge")
        raw = await _apply_orientation_nudge(raw, ctx.user_id, ctx.db, preserve_order=True)
        # Collapse multi-topic duplicates so the synthesizer sees one entry
        # per logical paper (with aggregated topic memberships) instead of
        # three near-identical rows for cross-listed arXiv papers.
        raw = _dedup_results_by_external_id(raw, scope=ns_keys)
        papers = [_normalize(r) for r in raw[: params.limit]]

        await ctx.emit_progress(100, f"Retrieved {len(papers)} papers")
        return ToolResult(
            output={"papers": papers, "total": len(papers)},
            summary=f"Deep Search returned {len(papers)} grounded papers",
            citations=[p["paper_id"] for p in papers if p.get("paper_id")],
        )


def _normalize(row: dict) -> dict:
    """Coerce a search-result row into a stable, JSON-safe shape."""
    return {
        "paper_id": str(row.get("paper_id") or ""),
        "title": row.get("title") or "",
        "abstract": row.get("abstract") or "",
        "authors": row.get("authors") or [],
        "namespace_key": row.get("namespace_key") or "",
        "namespace_keys": row.get("namespace_keys") or [row.get("namespace_key") or ""],
        "source_url": row.get("source_url") or "",
        "pdf_url": row.get("pdf_url"),
        "published_at": row.get("published_at"),
        "tldr": row.get("tldr"),
        "key_concepts": row.get("key_concepts") or [],
        "methods_used": row.get("methods_used") or [],
        "novelty_score": row.get("novelty_score") or 0.0,
        "relevance_score": row.get("relevance_score") or 0.0,
        "search_score": row.get("search_score") or 0.0,
        "match_type": row.get("match_type") or "deep",
    }
