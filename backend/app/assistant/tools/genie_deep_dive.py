"""Genie deep dive tool — access or trigger a comprehensive deep-dive article for a Genie idea.

When the user wants to explore a Genie-synthesized hypothesis in depth, this tool:
  1. Finds the target capsule (by ID or by keyword match on title/hypothesis).
  2. If a deep dive has already been generated (status == "done"), returns the full
     content for the synthesizer to reason over.
  3. If no deep dive exists yet, triggers background generation and returns a
     "generating" status with a link — RA can then inform the user it's in progress.

The deep dive itself is a publication-quality technical article written by the
Opus reasoning model using all source papers as context. It goes far beyond the
brief capsule hypothesis to cover mechanisms, prior work, experimental design,
risks, and open questions.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from app.assistant.tools.base import ToolContext, ToolResult

# Strong references to deep-dive background tasks against Python 3.12+ GC.
# A GC'd task can be cancelled mid-generation and leave deep_dive_status="generating"
# forever in the DB, so the user sees a permanent spinner.
_DEEP_DIVE_BG_TASKS: set[asyncio.Task] = set()
from app.models.genie import IdeaCapsule

log = logging.getLogger(__name__)

_DD_EXCERPT_CHARS = 6000   # How many chars of deep dive to pass to the synthesizer


class GenieDeepDiveInput(BaseModel):
    capsule_id: str = Field(
        default="",
        description=(
            "UUID of the specific IdeaCapsule to deep-dive. "
            "Leave empty to search by keyword instead."
        ),
    )
    query: str = Field(
        default="",
        max_length=300,
        description="Keyword to find the most relevant idea when capsule_id is unknown.",
    )
    trigger_if_missing: bool = Field(
        default=True,
        description=(
            "If True and no deep dive exists yet, start background generation. "
            "Set False to only read existing content."
        ),
    )
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class GenieDeepDiveOutput(BaseModel):
    capsule_id: str
    title: str
    hypothesis: str
    deep_dive_excerpt: str
    deep_dive_status: str   # "done" | "generating" | "none" | "failed" | "not_found"
    href: str
    chars_total: int


class GenieDeepDiveTool:
    """Deep-dive into a Genie idea — read or generate the comprehensive technical article."""

    name = "genie_deep_dive"
    summary = (
        "Access the comprehensive technical deep-dive article for a Genie-synthesized "
        "hypothesis. The deep dive covers mechanism, related work, experimental design, "
        "risks, and open questions — written by Opus with full paper context. "
        "Use when: the user wants to deeply explore a specific Genie idea, asks 'tell me "
        "more about [idea title]', or wants research directions beyond the brief capsule "
        "summary. If no deep dive exists, this tool triggers background generation "
        "and the user is informed it's in progress. "
        "Provide capsule_id if known; otherwise provide a query keyword to find the idea."
    )
    cost_class = "moderate"
    side_effects = True   # may trigger background deep dive generation
    cancellable = False
    streamable = False
    input_schema = GenieDeepDiveInput
    output_schema = GenieDeepDiveOutput

    async def run(self, ctx: ToolContext, params: GenieDeepDiveInput) -> ToolResult:
        await ctx.emit_progress(15, "Looking up Genie idea capsule…")

        capsule: IdeaCapsule | None = None

        # Resolve capsule: explicit ID first, then keyword search
        if params.capsule_id.strip():
            try:
                cid = UUID(params.capsule_id.strip())
                result = await ctx.db.execute(
                    select(IdeaCapsule).where(
                        IdeaCapsule.id == cid,
                        IdeaCapsule.user_id == ctx.user_id,
                    )
                )
                capsule = result.scalar_one_or_none()
            except (ValueError, Exception) as exc:
                log.warning("genie_deep_dive: bad capsule_id %s: %s", params.capsule_id, exc)

        if capsule is None and params.query.strip():
            q_lower = params.query.strip().lower()
            keywords = q_lower.split()
            rows_result = await ctx.db.execute(
                select(IdeaCapsule)
                .where(
                    IdeaCapsule.user_id == ctx.user_id,
                    IdeaCapsule.status.in_(["saved", "draft"]),
                )
                .order_by(desc(IdeaCapsule.created_at))
                .limit(30)
            )
            rows = rows_result.scalars().all()
            # Rank by keyword overlap
            def _score(cap: IdeaCapsule) -> int:
                text = ((cap.title or "") + " " + (cap.hypothesis or "")).lower()
                return sum(1 for kw in keywords if kw in text)
            scored = [(c, _score(c)) for c in rows if _score(c) > 0]
            if scored:
                capsule = max(scored, key=lambda x: x[1])[0]

        if capsule is None:
            return ToolResult(
                output={
                    "capsule_id": "",
                    "title": "",
                    "hypothesis": "",
                    "deep_dive_excerpt": "",
                    "deep_dive_status": "not_found",
                    "href": "/genie",
                    "chars_total": 0,
                },
                summary="No matching Genie idea capsule found",
            )

        capsule_id_str = str(capsule.id)
        href = f"/genie/idea/{capsule_id_str}"
        status = capsule.deep_dive_status or "none"

        # Case 1: Deep dive already generated — return excerpt
        if status == "done" and capsule.deep_dive_content:
            content = capsule.deep_dive_content
            excerpt = content[:_DD_EXCERPT_CHARS]
            if len(content) > _DD_EXCERPT_CHARS:
                excerpt += "\n\n[… content continues — see full article at the idea page …]"
            await ctx.emit_progress(100, f"Deep dive loaded: {len(content):,} chars")
            return ToolResult(
                output={
                    "capsule_id": capsule_id_str,
                    "title": capsule.title or "",
                    "hypothesis": capsule.hypothesis or "",
                    "deep_dive_excerpt": excerpt,
                    "deep_dive_status": "done",
                    "href": href,
                    "chars_total": len(content),
                },
                summary=(
                    f"Deep dive loaded for '{capsule.title}' "
                    f"({len(content):,} chars) — full article at {href}"
                ),
            )

        # Case 2: Already generating — just report status
        if status == "generating":
            await ctx.emit_progress(100, "Deep dive generation already in progress")
            return ToolResult(
                output={
                    "capsule_id": capsule_id_str,
                    "title": capsule.title or "",
                    "hypothesis": capsule.hypothesis or "",
                    "deep_dive_excerpt": "",
                    "deep_dive_status": "generating",
                    "href": href,
                    "chars_total": 0,
                },
                summary=(
                    f"Deep dive for '{capsule.title}' is already generating — "
                    f"check the idea page at {href}"
                ),
            )

        # Case 3: Not yet generated (or failed) — trigger background generation if allowed
        if params.trigger_if_missing:
            await ctx.emit_progress(60, f"Triggering deep dive generation for '{capsule.title}'…")
            try:
                # Mark as generating in DB before spawning task
                capsule.deep_dive_status = "generating"
                await ctx.db.commit()

                # Spawn as a fire-and-forget asyncio task — same pattern as the
                # API endpoint. Rooted in _DEEP_DIVE_BG_TASKS so Python 3.12+
                # cannot GC it before the article is committed (which would
                # leave deep_dive_status="generating" stuck in the DB).
                from app.workflows.genie import run_deep_dive_background
                t = asyncio.create_task(
                    run_deep_dive_background(capsule_id_str, str(ctx.user_id)),
                    name=f"ra:deep_dive:{capsule_id_str}",
                )
                _DEEP_DIVE_BG_TASKS.add(t)
                t.add_done_callback(_DEEP_DIVE_BG_TASKS.discard)
                await ctx.emit_progress(100, "Deep dive generation started")
            except Exception as exc:
                log.warning("genie_deep_dive: failed to trigger generation: %s", exc)
                return ToolResult(
                    output={
                        "capsule_id": capsule_id_str,
                        "title": capsule.title or "",
                        "hypothesis": capsule.hypothesis or "",
                        "deep_dive_excerpt": "",
                        "deep_dive_status": "failed",
                        "href": href,
                        "chars_total": 0,
                    },
                    summary=f"Could not start deep dive generation: {exc}",
                )

            return ToolResult(
                output={
                    "capsule_id": capsule_id_str,
                    "title": capsule.title or "",
                    "hypothesis": (capsule.hypothesis or "")[:500],
                    "deep_dive_excerpt": "",
                    "deep_dive_status": "generating",
                    "href": href,
                    "chars_total": 0,
                },
                summary=(
                    f"Deep dive generation started for '{capsule.title}'. "
                    f"It will be ready in ~60–90 seconds at {href}"
                ),
            )

        # trigger_if_missing=False and no content
        await ctx.emit_progress(100, "No deep dive content available")
        return ToolResult(
            output={
                "capsule_id": capsule_id_str,
                "title": capsule.title or "",
                "hypothesis": (capsule.hypothesis or "")[:500],
                "deep_dive_excerpt": "",
                "deep_dive_status": status,
                "href": href,
                "chars_total": 0,
            },
            summary=f"No deep dive yet for '{capsule.title}' (status: {status})",
        )


genie_deep_dive_tool = GenieDeepDiveTool()
