"""Media generation tool — trigger podcast or slide deck creation from selected papers.

STRICT HITL TOOL — only activates when:
  1. User explicitly requests media generation in this message.
  2. User has provided specific paper IDs to generate from.
  3. Conversation has enough research context to produce quality media.

Routes to the existing podcast/slides generation workflow (app.api.v1.generate)
which handles queuing, artifact creation, and job tracking.

Returns immediately with a job_id and a link to track progress — generation
happens asynchronously (typically 2-5 minutes for podcast, 1-3 minutes for slides).
"""

from __future__ import annotations

import logging
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)


class MediaGenerateInput(BaseModel):
    media_type: str = Field(
        description="'podcast' or 'slides'",
        pattern="^(podcast|slides)$",
    )
    paper_ids: list[str] = Field(
        min_length=1,
        max_length=10,
        description=(
            "List of paper UUIDs to generate from. "
            "User must have explicitly selected these papers. "
            "Do NOT infer — only pass IDs the user has named or selected."
        ),
    )
    expertise_level: str = Field(
        default="practitioner",
        description="'newcomer', 'practitioner', or 'expert' — adapts content depth.",
    )
    orientation: str = Field(
        default="both",
        description="'research', 'production', or 'both'.",
    )
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class MediaGenerateOutput(BaseModel):
    media_type: str
    job_id: str
    artifact_id: str
    paper_count: int
    href: str
    status: str


class MediaGenerateTool:
    """Trigger podcast or slide deck generation from user-selected papers (HITL)."""

    name = "media_generate"
    summary = (
        "⚠️  HUMAN-IN-THE-LOOP ONLY. Generate a podcast episode or slide deck from "
        "user-selected papers. Use ONLY when: "
        "(1) The user has EXPLICITLY asked to generate a podcast or slides in this message. "
        "(2) The user has named or selected specific papers (you have their UUIDs from context). "
        "(3) The conversation has sufficient research depth. "
        "NEVER trigger speculatively or based on vague requests. "
        "Media types: 'podcast' = conversational audio episode transcript, "
        "'slides' = Marp-formatted slide deck markdown. "
        "Generation is async (~2-5 min). Returns job_id + tracking link immediately."
    )
    cost_class = "heavy"
    side_effects = True
    cancellable = False
    streamable = False
    input_schema = MediaGenerateInput
    output_schema = MediaGenerateOutput

    async def run(self, ctx: ToolContext, params: MediaGenerateInput) -> ToolResult:
        from app.models.paper import Paper

        await ctx.emit_progress(15, f"Validating {len(params.paper_ids)} paper(s) for {params.media_type} generation…")

        # Validate all paper IDs belong to this user
        valid_paper_ids: list[str] = []
        paper_titles: list[str] = []
        for pid_str in params.paper_ids:
            try:
                pid = UUID(pid_str.strip())
                result = await ctx.db.execute(
                    select(Paper.id, Paper.title).where(
                        Paper.id == pid,
                        Paper.user_id == ctx.user_id,
                    )
                )
                row = result.one_or_none()
                if row:
                    valid_paper_ids.append(str(row[0]))
                    paper_titles.append(row[1] or "Untitled")
            except Exception as exc:
                log.warning("media_generate: invalid paper_id %s: %s", pid_str, exc)

        if not valid_paper_ids:
            return ToolResult(
                output={
                    "media_type": params.media_type,
                    "job_id": "",
                    "artifact_id": "",
                    "paper_count": 0,
                    "href": "/generate",
                    "status": "error",
                },
                summary=f"No valid papers found for media generation. Paper IDs provided: {params.paper_ids}",
            )

        await ctx.emit_progress(40, f"Queuing {params.media_type} generation for {len(valid_paper_ids)} paper(s)…")

        # For multi-paper generation, we use the first paper as the source
        # (the existing workflow takes a single paper or folder; multi-paper
        # requires creating a folder or using the paper directly).
        # Use the first valid paper as the primary source.
        source_paper_id = valid_paper_ids[0]
        source_type = "paper"

        artifact_id = ""
        job_id = ""
        href = "/generate"

        try:
            from app.models.artifact import Artifact, ArtifactStatus, GenerationType, SourceType
            from app.db.session import async_session_factory
            from app.services.job_store import get_job_store

            gen_type = GenerationType(params.media_type)
            src_type = SourceType.paper

            # Look up source paper title
            title = paper_titles[0] if paper_titles else "Selected Paper"

            # Resolve user provider settings for model selection
            from app.core.config import get_settings
            settings = get_settings()
            provider = settings.default_llm_provider
            model_used = (
                settings.default_quality_model
                if params.media_type == "podcast"
                else settings.default_reasoning_model
            )

            async with async_session_factory() as db:
                # Create an Artifact row (mirrors trigger_generation endpoint)
                artifact = Artifact(
                    user_id=ctx.user_id,
                    generation_type=gen_type,
                    source_type=src_type,
                    source_id=UUID(source_paper_id),
                    status=ArtifactStatus.queued,
                    expertise_level=params.expertise_level,
                    orientation=params.orientation,
                    provider=provider,
                    model_used=model_used,
                    artifact_metadata={"source_title": title, "paper_ids": valid_paper_ids},
                )
                db.add(artifact)
                await db.commit()
                await db.refresh(artifact)
                artifact_id = str(artifact.id)

            # Dispatch the generation job
            from app.api.v1.generate import _dispatch_job
            job_id = _dispatch_job(
                generation_type=gen_type,
                artifact_id=UUID(artifact_id),
                user_id=ctx.user_id,
                source_type=src_type.value,
                source_id=source_paper_id,
                expertise_level=params.expertise_level,
                orientation=params.orientation,
                title=title,
            )

            href = f"/generate?artifact={artifact_id}"

        except Exception as exc:
            log.warning("media_generate: dispatch failed: %s", exc)
            return ToolResult(
                output={
                    "media_type": params.media_type,
                    "job_id": "",
                    "artifact_id": artifact_id,
                    "paper_count": len(valid_paper_ids),
                    "href": "/generate",
                    "status": "error",
                },
                summary=f"Media generation dispatch failed: {exc}",
            )

        await ctx.emit_progress(100, f"{params.media_type.capitalize()} generation queued")

        titles_preview = "; ".join(paper_titles[:3])
        if len(paper_titles) > 3:
            titles_preview += f" +{len(paper_titles) - 3} more"

        return ToolResult(
            output={
                "media_type": params.media_type,
                "job_id": job_id,
                "artifact_id": artifact_id,
                "paper_count": len(valid_paper_ids),
                "href": href,
                "status": "queued",
            },
            summary=(
                f"{params.media_type.capitalize()} generation queued for {len(valid_paper_ids)} paper(s) "
                f"({titles_preview}). Track progress at {href}"
            ),
        )


media_generate_tool = MediaGenerateTool()
