"""Idea-combine tool — fuse two existing IdeaCapsules into a hybrid hypothesis.

Wraps :func:`app.workflows.genie_combine.run_capsule_combine` so the RA can
drive capsule fusion from inside a session. The tool accepts either a pair of
capsule UUIDs or a pair of free-text labels that the orchestrator resolves to
capsules via the user's library (label-based resolution lets the user ask
something like *"combine the Mixture-of-Tokens idea with the Speculative
Decoding idea"* without having to copy UUIDs into the chat).

The feasibility judge runs first; an infeasible pair returns a structured
"declined" result instead of a hard error so the synthesizer can explain
why to the user.
"""

from __future__ import annotations

import logging
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import desc, or_, select

from app.assistant.tools.base import ToolContext, ToolResult
from app.models.genie import IdeaCapsule

log = logging.getLogger(__name__)


class GenieCombineInput(BaseModel):
    """Identify the two parent capsules to combine.

    At least one of the ``*_id`` / ``*_label`` pair must be supplied for each
    parent. UUIDs win when both are given.
    """

    capsule_a_id: str | None = Field(default=None, description="UUID of parent A.")
    capsule_b_id: str | None = Field(default=None, description="UUID of parent B.")
    capsule_a_label: str | None = Field(
        default=None,
        max_length=240,
        description=(
            "Free-text identifier for parent A — usually a substring of the title "
            "or a distinctive phrase from the hypothesis. Resolved against the "
            "user's capsule library when ``capsule_a_id`` is not provided."
        ),
    )
    capsule_b_label: str | None = Field(
        default=None,
        max_length=240,
        description="Free-text identifier for parent B (see ``capsule_a_label``).",
    )


class GenieCombineOutput(BaseModel):
    status: str            # "created" | "infeasible" | "missing" | "error"
    capsule_id: str | None
    parent_ids: list[str]
    feasibility: dict
    reason: str
    title: str | None = None
    hypothesis: str | None = None


# ── Label-based resolution ────────────────────────────────────────────────────

async def _resolve_capsule(
    db,
    user_id: UUID,
    capsule_id: str | None,
    label: str | None,
) -> IdeaCapsule | None:
    """Resolve a parent capsule by id-or-label, scoped to the user.

    Strategy: try UUID first. If it doesn't parse or no row matches, fall back
    to an ILIKE search on title (then hypothesis) and pick the most-recently
    created match — capsule titles aren't unique, so newer-first is the most
    intuitive default when the user is loose with phrasing.
    """
    if capsule_id:
        try:
            uid = UUID(capsule_id)
            res = await db.execute(
                select(IdeaCapsule).where(
                    IdeaCapsule.id == uid, IdeaCapsule.user_id == user_id
                )
            )
            cap = res.scalar_one_or_none()
            if cap:
                return cap
        except ValueError:
            pass

    if not label:
        return None
    pat = f"%{label.strip()}%"
    res = await db.execute(
        select(IdeaCapsule)
        .where(
            IdeaCapsule.user_id == user_id,
            or_(IdeaCapsule.title.ilike(pat), IdeaCapsule.hypothesis.ilike(pat)),
        )
        .order_by(desc(IdeaCapsule.created_at))
        .limit(1)
    )
    return res.scalar_one_or_none()


class GenieCombineTool:
    """Combine two saved Idea Capsules into a new hybrid hypothesis.

    The workflow checks combinability (complementarity + conceptual distance)
    before spending reasoning-tier tokens, and writes a new capsule with
    ``source_mode="combined"`` and provenance back to both parents via
    :class:`GenieElement` rows of type ``idea``.
    """

    name = "genie_combine"
    summary = (
        "Fuse two previously synthesized idea capsules into a new hybrid "
        "hypothesis grounded in both ideas' deep-dive content. Runs a "
        "feasibility judge first — declines redundant or fully disjoint "
        "pairs with an explanation. Use when the user says things like "
        "'combine these two ideas', 'merge X with Y', 'what if both X and Y "
        "were true at the same time'. Identify parents by UUID when possible; "
        "fall back to a distinctive label (title fragment) when not."
    )
    cost_class = "heavy"
    side_effects = True   # creates a new IdeaCapsule + linked elements
    cancellable = False
    streamable = False
    input_schema = GenieCombineInput
    output_schema = GenieCombineOutput

    async def run(self, ctx: ToolContext, params: GenieCombineInput) -> ToolResult:
        from app.workflows.genie_combine import run_capsule_combine

        await ctx.emit_progress(10, "Resolving parent capsules")
        a = await _resolve_capsule(ctx.db, ctx.user_id, params.capsule_a_id, params.capsule_a_label)
        b = await _resolve_capsule(ctx.db, ctx.user_id, params.capsule_b_id, params.capsule_b_label)

        if a is None or b is None:
            missing = []
            if a is None:
                missing.append(params.capsule_a_id or params.capsule_a_label or "A")
            if b is None:
                missing.append(params.capsule_b_id or params.capsule_b_label or "B")
            return ToolResult(
                output={
                    "status": "missing",
                    "capsule_id": None,
                    "parent_ids": [str(a.id) if a else "", str(b.id) if b else ""],
                    "feasibility": {},
                    "reason": f"Could not find capsule(s): {', '.join(missing)}",
                    "title": None,
                    "hypothesis": None,
                },
                summary=f"Could not resolve parent capsule(s): {', '.join(missing)}",
            )

        if a.id == b.id:
            return ToolResult(
                output={
                    "status": "infeasible",
                    "capsule_id": None,
                    "parent_ids": [str(a.id), str(b.id)],
                    "feasibility": {},
                    "reason": "Both inputs resolved to the same capsule.",
                    "title": None,
                    "hypothesis": None,
                },
                summary="Both inputs resolved to the same capsule.",
            )

        await ctx.emit_progress(35, "Checking feasibility")
        result = await run_capsule_combine(
            user_id=ctx.user_id,
            capsule_ids=[a.id, b.id],
        )

        if result["status"] != "created":
            await ctx.emit_progress(100, result["reason"])
            return ToolResult(
                output={
                    "status": result["status"],
                    "capsule_id": result.get("capsule_id"),
                    "parent_ids": result["parent_ids"],
                    "feasibility": result.get("feasibility", {}),
                    "reason": result.get("reason", ""),
                    "title": None,
                    "hypothesis": None,
                },
                summary=f"Capsule combine declined: {result['reason']}",
            )

        # Re-load the new capsule to surface its title/hypothesis in the block.
        from sqlalchemy import select as _sel
        new_id = UUID(result["capsule_id"])
        row = await ctx.db.execute(_sel(IdeaCapsule).where(IdeaCapsule.id == new_id))
        new_cap = row.scalar_one_or_none()
        title = new_cap.title if new_cap else "Hybrid Hypothesis"
        hypothesis = (new_cap.hypothesis if new_cap else "") or ""

        await ctx.emit_progress(100, f"Hybrid hypothesis created: {title[:60]}")
        return ToolResult(
            output={
                "status": "created",
                "capsule_id": result["capsule_id"],
                "parent_ids": result["parent_ids"],
                "feasibility": result.get("feasibility", {}),
                "reason": result.get("reason", ""),
                "title": title,
                "hypothesis": hypothesis,
            },
            summary=f"Combined '{(a.title or '')[:40]}' × '{(b.title or '')[:40]}' → '{title[:60]}'",
            citations=result["parent_ids"],
            artifacts=[{
                "kind": "idea_capsule",
                "ref_id": result["capsule_id"],
                "title": title[:120],
                "href": f"/genie/idea/{result['capsule_id']}",
                "preview": {
                    "source_mode": "combined",
                    "parents": result["parent_ids"],
                    "novelty": new_cap.novelty_score if new_cap else None,
                    "feasibility": new_cap.feasibility_score if new_cap else None,
                    "impact": new_cap.impact_score if new_cap else None,
                },
            }],
        )


genie_combine_tool = GenieCombineTool()
