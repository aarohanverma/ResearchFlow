"""Knowledge graph build tool — wraps GraphService.build_deep_graph."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.services.graph import GraphService

log = logging.getLogger(__name__)


class GraphBuildInput(BaseModel):
    namespace_key: str | None = None
    orientation: str | None = None


class GraphBuildOutput(BaseModel):
    result: dict


class GraphBuildTool:
    """Build/refresh the deep knowledge graph for a namespace."""

    name = "graph_build"
    summary = (
        "Build or refresh the deep knowledge-graph taxonomy (concepts, methods, "
        "papers, edges) for a namespace. Heavy: uses LLM-driven concept extraction. "
        "Use when the user asks about landscape, taxonomy, concept maps, or graph state."
    )
    cost_class = "heavy"
    side_effects = True
    cancellable = True
    streamable = True
    input_schema = GraphBuildInput
    output_schema = GraphBuildOutput

    async def run(self, ctx: ToolContext, params: GraphBuildInput) -> ToolResult:
        target_ns = params.namespace_key or ctx.namespace_key
        orientation = params.orientation or ctx.orientation or "both"
        await ctx.emit_progress(20, f"Building graph for {target_ns}")
        svc = GraphService(ctx.db)
        result = await svc.build_deep_graph(
            target_ns,
            orientation=orientation,
            should_cancel=ctx.should_cancel,
        )
        await ctx.emit_progress(100, "Graph taxonomy updated")
        return ToolResult(
            output={"result": result or {}},
            summary="Knowledge graph taxonomy updated",
            artifacts=[{
                "kind": "graph_snapshot",
                "ref_id": target_ns,
                "title": f"Graph: {target_ns}",
                "href": "/graph",
                "preview": {"namespace": target_ns, "summary": result or {}},
            }],
        )
