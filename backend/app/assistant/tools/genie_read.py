"""Genie-read tool — surface existing Genie idea capsules into RA context.

Fetches the user's most recent saved/draft IdeaCapsules and returns them as
structured data. The synthesizer then wraps these in a clearly-marked
<genie_research_ideas> block so the reasoning model knows they are
AI-generated hypotheses, not validated facts.
"""

from __future__ import annotations

import logging
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.models.genie import IdeaCapsule

log = logging.getLogger(__name__)


class GenieReadInput(BaseModel):
    query: str = Field(default="", max_length=500, description="Optional topic filter (keyword match on title/hypothesis)")
    limit: int = Field(default=5, ge=1, le=15)
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class GenieReadOutput(BaseModel):
    ideas: list[dict]
    total_found: int


class GenieReadTool:
    """Retrieve the user's existing Genie idea capsules for context."""

    name = "genie_read"
    summary = (
        "Retrieve the user's existing Genie idea capsules (AI-generated, unvalidated "
        "research hypotheses) into the assistant's context. Use when: the user asks "
        "'what ideas have I generated', 'show my Genie hypotheses', wants to discuss "
        "or build on a prior synthesis, or the conversation is exploring research "
        "directions where prior ideas are relevant. Never treat capsule content as "
        "established fact — always frame as unvalidated hypotheses."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = GenieReadInput
    output_schema = GenieReadOutput

    async def run(self, ctx: ToolContext, params: GenieReadInput) -> ToolResult:
        await ctx.emit_progress(20, "Reading Genie idea capsules")

        stmt = (
            select(IdeaCapsule)
            .where(
                IdeaCapsule.user_id == ctx.user_id,
                IdeaCapsule.status.in_(["saved", "draft"]),
            )
            .order_by(desc(IdeaCapsule.created_at))
            .limit(params.limit * 3)  # over-fetch for keyword filtering
        )
        result = await ctx.db.execute(stmt)
        rows = result.scalars().all()

        # Keyword filter on title + hypothesis when a query is provided
        q_lower = (params.query or "").lower().strip()
        if q_lower:
            keywords = q_lower.split()
            filtered = [
                r for r in rows
                if any(
                    kw in (r.title or "").lower() or kw in (r.hypothesis or "").lower()
                    for kw in keywords
                )
            ]
            rows = filtered or rows  # fall back to unfiltered if nothing matched

        rows = rows[: params.limit]

        ideas = []
        for c in rows:
            ideas.append({
                "id": str(c.id),
                "title": c.title,
                "hypothesis": c.hypothesis,
                "rationale": c.rationale[:600] if c.rationale else "",
                "mechanism": (c.mechanism or "")[:400],
                "predicted_outcome": (c.predicted_outcome or "")[:300],
                "experimental_design": (c.experimental_design or "")[:400],
                "risks_and_limitations": (c.risks_and_limitations or "")[:300],
                "novelty_score": float(c.novelty_score or 0.0),
                "feasibility_score": float(c.feasibility_score or 0.0),
                "impact_score": float(c.impact_score or 0.0),
                "status": c.status,
                "source_mode": c.source_mode,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "href": f"/genie/idea/{c.id}",
            })

        if not ideas:
            return ToolResult(
                output={"ideas": [], "total_found": 0},
                summary="No Genie idea capsules found for this user",
            )

        await ctx.emit_progress(100, f"Loaded {len(ideas)} Genie idea capsule(s)")
        return ToolResult(
            output={"ideas": ideas, "total_found": len(ideas)},
            summary=f"{len(ideas)} Genie hypothesis capsule(s) retrieved (unvalidated AI-generated ideas)",
        )


genie_read_tool = GenieReadTool()
