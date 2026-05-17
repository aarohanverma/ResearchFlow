"""Genie synthesis tool — seeds a Genie session and dispatches the workflow."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.db.session import async_session_factory
from app.models.genie import ElementType, GenieElement, GenieSession, IdeaCapsule

log = logging.getLogger(__name__)


# Strong references to queued-mode background tasks so Python 3.12+ does not
# garbage-collect the run_genie_background coroutine before it commits the
# IdeaCapsule. Tasks self-discard on completion.
_GENIE_BG_TASKS: set[asyncio.Task] = set()


# Inline budget — wait this long for the Genie workflow to finish before
# falling back to "queued, results in /genie tab" mode. The RA orchestrator
# itself is already async/background, so users keep interacting freely
# while we wait — there's no UI block. Generous budget so most synthesis
# runs land inline; a long-tail run still falls through to the queued
# artifact link without breaking the experience.
_INLINE_TIMEOUT_SEC = 600.0   # 10 minutes — covers ~99% of Genie wall times
_POLL_INTERVAL_SEC = 2.0


class GenieSynthesizeInput(BaseModel):
    paper_ids: list[str] = Field(default_factory=list, description="UUIDs of seed papers (≥2 required)")
    paper_titles: list[str] = Field(default_factory=list, description="Optional titles parallel to paper_ids")
    query: str = Field(default="", max_length=1000)
    inline: bool = Field(
        default=True,
        description="When true (default), wait up to 75s for the synthesis to "
                    "complete and return the produced idea capsule inline. "
                    "When false, return immediately with just a session id.",
    )


class GenieSynthesizeOutput(BaseModel):
    genie_session_id: str | None = None
    seed_count: int = 0
    status: str = "queued"           # queued | running | done | done_empty | failed | timeout
    capsule: dict | None = None      # Populated when status=='done'


class GenieSynthesizeTool:
    """Run a Genie hypothesis-synthesis workflow inline, with queued fallback."""

    name = "genie_synthesize"
    summary = (
        "Seed a Genie session with selected papers and run the transparent "
        "hypothesis-synthesis workflow (gather→bridges→viability→hypothesize→"
        "critique→elaborate→diagram→poc). Returns the produced idea capsule "
        "inline (waits up to ~75s) so the user can see the hypothesis, "
        "rationale, and experimental design without leaving the assistant. "
        "Falls back to a queued session id if the workflow doesn't complete "
        "in time. Requires ≥2 paper ids."
    )
    cost_class = "heavy"
    side_effects = True
    cancellable = False  # detaches into a separate background task
    streamable = True
    input_schema = GenieSynthesizeInput
    output_schema = GenieSynthesizeOutput

    async def run(self, ctx: ToolContext, params: GenieSynthesizeInput) -> ToolResult:
        from app.workflows.genie import run_genie_background

        seed_uuids: list[UUID] = []
        title_map: dict[str, str] = {
            pid: title for pid, title in zip(params.paper_ids, params.paper_titles)
        }
        for pid in params.paper_ids:
            try:
                seed_uuids.append(UUID(str(pid)))
            except ValueError:
                continue

        if len(seed_uuids) < 2:
            return ToolResult(
                output={"genie_session_id": None, "seed_count": len(seed_uuids)},
                summary="Skipped — Genie needs ≥2 valid paper ids",
            )

        await ctx.emit_progress(30, f"Seeding Genie with {len(seed_uuids)} papers")
        seed_element_ids: list[str] = []
        for pid in seed_uuids:
            existing = await ctx.db.execute(
                select(GenieElement).where(
                    GenieElement.user_id == ctx.user_id,
                    GenieElement.paper_id == pid,
                )
            )
            el = existing.scalar_one_or_none()
            if not el:
                el = GenieElement(
                    user_id=ctx.user_id,
                    element_type=ElementType.paper,
                    label=str(title_map.get(str(pid), ""))[:500],
                    paper_id=pid,
                )
                ctx.db.add(el)
                await ctx.db.flush()
            seed_element_ids.append(str(el.id))

        if len(seed_element_ids) < 2:
            return ToolResult(
                output={"genie_session_id": None, "seed_count": len(seed_element_ids)},
                summary="Skipped — could not resolve enough Genie seed elements",
            )

        session = GenieSession(
            user_id=ctx.user_id,
            seed_element_ids=seed_element_ids,
            status="running",
        )
        ctx.db.add(session)
        await ctx.db.flush()
        await ctx.db.commit()
        gid = str(session.id)
        await ctx.emit_progress(35, "Dispatching Genie synthesis")

        # Genie is its own long-running pipeline. We always launch it as a
        # detached task so the user's tab in /genie keeps the canonical
        # streaming UI working — then optionally wait here for completion.
        # Root the task in _GENIE_BG_TASKS so queued-mode (inline=False) doesn't
        # let Python 3.12+ GC it before the synthesis commits.
        bg = asyncio.create_task(
            run_genie_background(
                ctx.user_id,
                gid,
                seed_element_ids,
                ctx.namespace_key,
                source_mode="query",
                source_query=params.query,
            ),
            name=f"ra:genie:{gid}",
        )
        _GENIE_BG_TASKS.add(bg)
        bg.add_done_callback(_GENIE_BG_TASKS.discard)

        if not params.inline:
            await ctx.emit_progress(100, "Genie synthesis queued")
            return ToolResult(
                output={
                    "genie_session_id": gid,
                    "seed_count": len(seed_element_ids),
                    "status": "queued",
                    "capsule": None,
                },
                summary=f"Queued Genie synthesis from {len(seed_element_ids)} papers",
                artifacts=[_genie_artifact(gid, len(seed_element_ids))],
            )

        # Inline mode: poll the Genie session row until status leaves the
        # in-flight states OR the wall-clock budget is exhausted. Cancellation
        # propagates from the orchestrator through ctx.should_cancel.
        await ctx.emit_progress(45, "Waiting for hypothesis synthesis")
        capsule_row, final_status = await _await_capsule(ctx, gid, bg)
        if capsule_row:
            await ctx.emit_progress(100, "Hypothesis ready")
            capsule_dict = _capsule_to_dict(capsule_row)
            return ToolResult(
                output={
                    "genie_session_id": gid,
                    "seed_count": len(seed_element_ids),
                    "status": final_status,
                    "capsule": capsule_dict,
                },
                summary=f"Genie synthesized hypothesis · {capsule_row.title[:60]}",
                artifacts=[
                    _genie_artifact(gid, len(seed_element_ids)),
                    {
                        "kind": "idea_capsule",
                        "ref_id": str(capsule_row.id),
                        "title": capsule_row.title[:120],
                        "href": f"/genie/idea/{capsule_row.id}",
                        "preview": {
                            "novelty": capsule_row.novelty_score,
                            "feasibility": capsule_row.feasibility_score,
                            "impact": capsule_row.impact_score,
                        },
                    },
                ],
            )

        await ctx.emit_progress(100, f"Genie still running — open /genie when ready ({final_status})")
        return ToolResult(
            output={
                "genie_session_id": gid,
                "seed_count": len(seed_element_ids),
                "status": final_status,
                "capsule": None,
            },
            summary=f"Genie still in flight ({final_status}) — see /genie tab when ready",
            artifacts=[_genie_artifact(gid, len(seed_element_ids))],
        )


def _genie_artifact(session_id: str, seed_count: int) -> dict:
    """The session-level artifact card surfaced in the message blocks list."""
    return {
        "kind": "genie_session",
        "ref_id": session_id,
        "title": "Genie synthesis",
        "href": "/genie?tab=discoveries",
        "preview": {"seed_count": seed_count},
    }


def _capsule_to_dict(c: IdeaCapsule) -> dict:
    """Project an ``IdeaCapsule`` row into a JSON-safe dict for the message payload."""
    return {
        "id": str(c.id),
        "title": c.title,
        "hypothesis": c.hypothesis,
        "rationale": c.rationale,
        "mechanism": c.mechanism,
        "predicted_outcome": c.predicted_outcome,
        "experimental_design": c.experimental_design,
        "anti_finding": c.anti_finding,
        "risks_and_limitations": c.risks_and_limitations,
        "open_questions": c.open_questions,
        "novelty_score": float(c.novelty_score or 0.0),
        "feasibility_score": float(c.feasibility_score or 0.0),
        "impact_score": float(c.impact_score or 0.0),
        "citation_paper_ids": list(c.citation_paper_ids or []),
        "diagrams": list(c.diagrams or []),
        "poc_code": c.poc_code,
    }


async def _await_capsule(
    ctx: ToolContext,
    session_id: str,
    bg_task: asyncio.Task,
) -> tuple[IdeaCapsule | None, str]:
    """Poll the Genie session until completion or the budget is reached.

    Returns ``(capsule_or_None, final_status)``. ``final_status`` is one of
    ``done``, ``done_empty``, ``failed``, ``timeout``, or ``cancelled`` so
    the caller can communicate the outcome to the user with precision.
    """
    deadline = asyncio.get_event_loop().time() + _INLINE_TIMEOUT_SEC
    last_emit = 45
    while True:
        if asyncio.get_event_loop().time() >= deadline:
            return None, "timeout"
        try:
            if await ctx.should_cancel():
                bg_task.cancel()
                return None, "cancelled"
        except Exception:
            # should_cancel() should never raise, but guard anyway.
            pass

        async with async_session_factory() as db:
            res = await db.execute(
                select(GenieSession).where(GenieSession.id == session_id)
            )
            session = res.scalar_one_or_none()
            if session and session.status in {"done", "done_empty", "failed"}:
                if session.status == "done" and session.result_capsule_id:
                    cap_res = await db.execute(
                        select(IdeaCapsule).where(IdeaCapsule.id == session.result_capsule_id)
                    )
                    capsule = cap_res.scalar_one_or_none()
                    if capsule:
                        return capsule, "done"
                return None, session.status

        # Emit gentle progress every ~9s of waiting so the user sees motion
        # in the reasoning strip while we wait on Genie.
        if last_emit < 95:
            last_emit = min(95, last_emit + 5)
            try:
                await ctx.emit_progress(last_emit, "Genie still composing")
            except Exception:
                pass
        await asyncio.sleep(_POLL_INTERVAL_SEC)
