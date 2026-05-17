"""Study-mode tool — generate or fetch the cached Study Mode walkthrough.

Wraps the Study workflow (``app.workflows.study``) as an :class:`AssistantTool`
so the RA orchestrator can drive the same expertise-adapted deep walkthrough
the dedicated Study Mode page produces.

Behaviour:
    * If a cached :class:`Summary` exists for ``(paper_id, expertise_level)``
      and the prompt-hash / model match the current generation pipeline, the
      cached content is returned immediately (no LLM cost).
    * Otherwise the Study LangGraph is queued in the background via
      :func:`queue_study` and the tool returns a ``status="generating"`` payload
      with the job id. The user can be informed that the walkthrough is being
      prepared and the result will appear in the Saved page / notifications.

This makes Study Mode reachable from any RA session — e.g.,
"Walk me through Attention Is All You Need at an expert level"
— without leaving the chat surface.
"""

from __future__ import annotations

import logging
from uuid import UUID

from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)


class StudyPaperInput(BaseModel):
    paper_id: str = Field(
        min_length=1,
        description=(
            "UUID of the paper to study. Must already be ingested into the "
            "user's corpus (use `paper_import` or `arxiv_import` first if not)."
        ),
    )
    expertise_level: str = Field(
        default="practitioner",
        description=(
            "Walkthrough depth: 'newcomer' (high-level intuition), "
            "'practitioner' (default — methods + tradeoffs), or "
            "'expert' (technical depth + critical analysis)."
        ),
    )


class StudyPaperOutput(BaseModel):
    paper_id: str
    expertise_level: str
    status: str           # "cached" | "generating" | "missing"
    paper_title: str
    job_id: str | None = None
    summary: dict | None = None  # the cached Summary.content when status="cached"


class StudyPaperTool:
    """Generate or retrieve the cached Study Mode walkthrough for a paper.

    Cache-first: returns the existing Summary row when one matches the current
    generation pipeline; otherwise queues a background Study workflow run and
    returns the job id.
    """

    name = "study_paper"
    summary = (
        "Generate (or fetch from cache) the Study Mode deep walkthrough for a "
        "paper at the user's expertise level. Returns either the cached "
        "structured summary immediately or a generating-job id when the "
        "walkthrough hasn't been produced yet. Use when the user wants a "
        "section-by-section explanation, intuition for the method, or a "
        "critical read of a specific paper."
    )
    cost_class = "moderate"
    side_effects = True   # may queue a background generation job
    cancellable = False
    streamable = False
    input_schema = StudyPaperInput
    output_schema = StudyPaperOutput

    async def run(self, ctx: ToolContext, params: StudyPaperInput) -> ToolResult:
        from sqlalchemy import select
        from app.models.paper import Paper
        from app.repositories.paper import PaperRepository
        from app.workflows.study import queue_study

        try:
            paper_uuid = UUID(params.paper_id)
        except ValueError:
            await ctx.emit_progress(100, "Invalid paper id")
            return ToolResult(
                output={
                    "paper_id": params.paper_id,
                    "expertise_level": params.expertise_level,
                    "status": "missing",
                    "paper_title": "",
                    "job_id": None,
                    "summary": None,
                },
                summary=f"Invalid paper id: {params.paper_id}",
            )

        # Verify the paper exists
        await ctx.emit_progress(15, "Looking up paper")
        result = await ctx.db.execute(select(Paper).where(Paper.id == paper_uuid))
        paper = result.scalar_one_or_none()
        if not paper:
            return ToolResult(
                output={
                    "paper_id": params.paper_id,
                    "expertise_level": params.expertise_level,
                    "status": "missing",
                    "paper_title": "",
                    "job_id": None,
                    "summary": None,
                },
                summary="Paper not found — import it via paper_import first.",
            )

        # Try cache
        await ctx.emit_progress(45, "Checking cache")
        repo = PaperRepository(ctx.db)
        cached = await repo.get_summary(paper_uuid, params.expertise_level)
        if cached is not None and cached.content:
            await ctx.emit_progress(100, "Study walkthrough cached")
            return ToolResult(
                output={
                    "paper_id": params.paper_id,
                    "expertise_level": params.expertise_level,
                    "status": "cached",
                    "paper_title": paper.title or "",
                    "job_id": None,
                    "summary": cached.content,
                },
                summary=f"Study walkthrough cached for '{paper.title[:60]}'",
                artifacts=[{
                    "kind": "study_summary",
                    "ref_id": params.paper_id,
                    "title": paper.title or params.paper_id,
                    "href": f"/study/{params.paper_id}",
                    "preview": {"expertise_level": params.expertise_level},
                }],
                citations=[params.paper_id],
            )

        # Queue background generation — Study runs through LangGraph and persists
        # its result to the Summary table, so the next call hits the cache.
        await ctx.emit_progress(70, "Queueing Study generation")
        try:
            job_id = queue_study(
                paper_id=paper_uuid,
                expertise_level=params.expertise_level,
                user_id=ctx.user_id,
                paper_title=paper.title or "",
            )
        except Exception as exc:
            log.warning("study_paper: queue failed paper=%s err=%s", params.paper_id, exc)
            return ToolResult(
                output={
                    "paper_id": params.paper_id,
                    "expertise_level": params.expertise_level,
                    "status": "missing",
                    "paper_title": paper.title or "",
                    "job_id": None,
                    "summary": None,
                },
                summary=f"Failed to queue Study generation: {exc!s:.120}",
            )

        await ctx.emit_progress(100, "Study generation queued — open the Study page to follow along")
        return ToolResult(
            output={
                "paper_id": params.paper_id,
                "expertise_level": params.expertise_level,
                "status": "generating",
                "paper_title": paper.title or "",
                "job_id": job_id,
                "summary": None,
            },
            summary=f"Study walkthrough queued for '{(paper.title or '')[:60]}'",
            artifacts=[{
                "kind": "study_summary",
                "ref_id": params.paper_id,
                "title": paper.title or params.paper_id,
                "href": f"/study/{params.paper_id}",
                "preview": {"expertise_level": params.expertise_level, "status": "generating"},
            }],
            citations=[params.paper_id],
        )


study_paper_tool = StudyPaperTool()
