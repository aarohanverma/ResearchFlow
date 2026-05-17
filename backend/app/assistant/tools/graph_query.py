"""Graph query tool — READ-ONLY consumption of an existing knowledge graph.

The Research Assistant must NEVER build or refresh the graph (that's the
dedicated /graph page workflow). This tool only consumes existing graph
state, returning concept/paper neighborhoods so the assistant can ground
its answers in the user's curated taxonomy.

If no graph exists for the requested namespace yet, the tool returns an
empty result with ``has_graph=False`` so the synthesizer can tell the user
to build one from the graph page rather than silently producing nothing.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.services.graph import GraphService

log = logging.getLogger(__name__)


class GraphQueryInput(BaseModel):
    namespace_key: str | None = None
    depth: int = Field(default=2, ge=1, le=3)
    max_nodes: int = Field(default=80, ge=10, le=300)


class GraphQueryOutput(BaseModel):
    has_graph: bool
    node_count: int
    edge_count: int
    sample_concepts: list[str]
    sample_papers: list[dict]
    summary: dict


class GraphQueryTool:
    """Inspect existing graph nodes and edges for a namespace (READ-ONLY)."""

    name = "graph_query"
    summary = (
        "Read-only inspection of an existing knowledge graph for a namespace. "
        "Returns node/edge counts, sample concept names, and a few representative "
        "papers with their connected concepts. Use when the user asks about "
        "concept relationships, taxonomy, or 'what does the graph say'. "
        "Does NOT build or refresh anything — graph construction is the "
        "/graph page's job. If no graph exists, returns has_graph=False."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = False
    streamable = False
    input_schema = GraphQueryInput
    output_schema = GraphQueryOutput

    async def run(self, ctx: ToolContext, params: GraphQueryInput) -> ToolResult:
        target_ns = params.namespace_key or ctx.namespace_key
        await ctx.emit_progress(30, f"Reading existing graph for {target_ns or 'all namespaces'}")

        svc = GraphService(ctx.db)
        try:
            sub = await svc.get_subgraph(target_ns, depth=params.depth)
        except Exception as exc:
            log.warning("graph_query: get_subgraph failed: %s", exc)
            sub = {"nodes": [], "edges": []}

        nodes = sub.get("nodes") or []
        edges = sub.get("edges") or []

        if not nodes:
            await ctx.emit_progress(100, "No graph found — recommend building one")
            return ToolResult(
                output={
                    "has_graph": False,
                    "node_count": 0,
                    "edge_count": 0,
                    "sample_concepts": [],
                    "sample_papers": [],
                    "summary": {"namespace": target_ns},
                },
                summary="No graph exists for this namespace yet — open /graph to build one",
            )

        concept_nodes = [n for n in nodes if str(n.get("type", "")).lower() in ("concept", "method")]
        paper_nodes = [n for n in nodes if str(n.get("type", "")).lower() == "paper"]
        sample_concepts = [str(n.get("label") or "")[:60] for n in concept_nodes[:30] if n.get("label")]
        sample_papers = [
            {
                "node_id": str(n.get("id") or ""),
                "title": str(n.get("label") or ""),
                "paper_id": str(n.get("paper_id") or ""),
                "namespace_key": n.get("namespace_key"),
                "tldr": n.get("description"),
                "source_url": n.get("source_url"),
            }
            for n in paper_nodes[: params.max_nodes // 8]
        ]

        await ctx.emit_progress(100, f"Graph: {len(nodes)} nodes, {len(edges)} edges")
        return ToolResult(
            output={
                "has_graph": True,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "sample_concepts": sample_concepts[:30],
                "sample_papers": sample_papers,
                "summary": {
                    "namespace": target_ns,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "concept_count": len(concept_nodes),
                    "paper_count": len(paper_nodes),
                },
            },
            summary=f"Graph has {len(nodes)} nodes and {len(edges)} edges in scope",
        )
