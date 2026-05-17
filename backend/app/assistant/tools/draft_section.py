"""Draft a publication-grade section from the user's session corpus.

Produces a Related Work, Background, Methodology, Discussion, or
Introduction draft grounded in 5-15 papers the user has surfaced in the
session. Every claim carries an inline citation; the LLM is forbidden
from introducing un-cited content. Intended as a *starting point* for the
user to edit, not a final submission — the assistant always reminds them
that human review and revision is required for publishable work.

This tool is the bridge between "I've explored the literature" and
"I'm writing the paper". Use cases:

* Newcomers: scaffold a Related Work section so they understand the lay
  of the land in formal academic prose.
* Experts: produce a fast first-pass draft they can then heavily revise.
* Mid-stream: clarify a gap that justifies a contribution.
"""

from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.models.paper import Paper

log = logging.getLogger(__name__)


SectionKind = Literal[
    "related_work", "background", "introduction", "methodology",
    "discussion", "limitations", "future_work",
]


_SECTION_GUIDES = {
    "related_work": (
        "Survey the prior work in this area. Cluster papers by approach. "
        "Identify what each line of work contributes and what it misses. "
        "End with a 1-2 sentence statement of where the gap lies."
    ),
    "background": (
        "Provide the technical background a reader needs to understand the "
        "rest of the paper. Define terms, sketch the canonical formulation, "
        "and cite the foundational sources for each construct."
    ),
    "introduction": (
        "Motivate the problem in 2-3 paragraphs: why does it matter, what "
        "has been tried, what's missing, and what this contribution targets. "
        "End with a bulleted list of contributions."
    ),
    "methodology": (
        "Describe the methodological approach, grounding choices in cited "
        "prior work. Be precise about what is novel vs. adapted from "
        "existing methods. Include data, training, evaluation specifics."
    ),
    "discussion": (
        "Interpret the (claimed/observed) results in light of the "
        "literature. What does this confirm, contradict, extend? What new "
        "questions does it raise?"
    ),
    "limitations": (
        "Honestly enumerate the limitations. Distinguish methodological "
        "limitations from scope limitations. Acknowledge confounds the "
        "literature has previously flagged."
    ),
    "future_work": (
        "Propose 3-5 concrete future directions, each tied to specific "
        "open questions in the literature."
    ),
}


class DraftSectionInput(BaseModel):
    section: SectionKind = Field(description="Which paper section to draft.")
    paper_ids: list[str] = Field(
        default_factory=list,
        description="Optional explicit paper IDs to cite; otherwise the latest "
                    "deep_search results from the session are used.",
    )
    topic: str = Field(
        default="", max_length=500,
        description="The topic / contribution of the user's paper. Critical for "
                    "framing — without it the draft will be generic.",
    )
    venue: str = Field(
        default="", max_length=80,
        description="Optional target venue (e.g. 'NeurIPS', 'CHI', 'ACL'). "
                    "Used for tone/format calibration.",
    )


class DraftSectionOutput(BaseModel):
    section: str
    markdown: str
    cited_paper_ids: list[str]
    notes: str


class DraftSectionTool:
    """Compose a publication-grade section draft grounded in cited papers."""

    name = "draft_section"
    summary = (
        "Draft a Related Work, Background, Introduction, Methodology, "
        "Discussion, Limitations, or Future Work section grounded in the "
        "user's session corpus. Every claim carries an inline citation. "
        "Use when the user is moving from exploration to writing — e.g. "
        "'draft a related work section', 'help me write the introduction', "
        "'what would the methodology section look like'. Always reminds "
        "the user that human review is required before publication."
    )
    cost_class = "heavy"
    side_effects = False
    cancellable = True
    streamable = True
    input_schema = DraftSectionInput
    output_schema = DraftSectionOutput

    async def run(self, ctx: ToolContext, params: DraftSectionInput) -> ToolResult:
        await ctx.emit_progress(15, f"Loading sources for {params.section}")
        papers = await self._resolve_papers(ctx, params.paper_ids)
        if len(papers) < 2:
            return ToolResult(
                output={
                    "section": params.section,
                    "markdown": (
                        f"### Cannot draft {params.section}\n\n"
                        "I need at least 2 cited papers from your session to ground a draft. "
                        "Run a Deep Search first or attach the papers you want to cite."
                    ),
                    "cited_paper_ids": [],
                    "notes": "Need >= 2 papers.",
                },
                summary=f"Skipped — need >= 2 papers, got {len(papers)}",
            )

        await ctx.emit_progress(60, "Composing draft")
        markdown = await _compose_draft(
            section=params.section,
            topic=params.topic,
            venue=params.venue,
            papers=papers,
            expertise=ctx.expertise_level,
            orientation=ctx.orientation,
        )
        cited_ids = [str(p.id) for p in papers]
        await ctx.emit_progress(100, f"Drafted {params.section} citing {len(cited_ids)} papers")

        return ToolResult(
            output={
                "section": params.section,
                "markdown": markdown,
                "cited_paper_ids": cited_ids,
                "notes": (
                    "First-pass draft for human revision. Verify every citation, "
                    "tighten the argument, and align with the target venue's style "
                    "guide before submission."
                ),
            },
            summary=f"Drafted {params.section} ({len(cited_ids)} citations)",
            citations=cited_ids,
            artifacts=[{
                "kind": "section_draft",
                "ref_id": f"{params.section}:{ctx.session_id}",
                "title": f"Draft · {params.section.replace('_', ' ').title()}",
                "preview": {"section": params.section, "citation_count": len(cited_ids)},
            }],
        )

    async def _resolve_papers(self, ctx: ToolContext, explicit_ids: list[str]) -> list[Paper]:
        """Use explicit IDs if provided; otherwise pull recent deep-search results."""
        ids: list[UUID] = []
        for pid in explicit_ids:
            try:
                ids.append(UUID(str(pid)))
            except ValueError:
                continue
        if not ids:
            # Fallback: scan recent assistant steps in the session for any
            # tool that produced cited papers, and reuse those IDs.
            from app.repositories.assistant import AssistantRepository

            repo = AssistantRepository(ctx.db)
            steps = await repo.list_steps_for_session(ctx.session_id, limit=20)
            for s in steps:
                output = s.output or {}
                for p in (output.get("papers") or [])[:8]:
                    pid = str(p.get("paper_id") or "")
                    try:
                        u = UUID(pid)
                    except ValueError:
                        continue
                    if u not in ids:
                        ids.append(u)
                if len(ids) >= 12:
                    break
        if not ids:
            return []
        result = await ctx.db.execute(select(Paper).where(Paper.id.in_(ids)))
        return list(result.scalars())[:15]


async def _compose_draft(
    *,
    section: str,
    topic: str,
    venue: str,
    papers: list[Paper],
    expertise: str,
    orientation: str,
) -> str:
    """Ask the quality model to compose a grounded section draft."""
    guide = _SECTION_GUIDES.get(section, _SECTION_GUIDES["related_work"])
    paper_block = "\n\n".join(
        f"[{i + 1}] {p.title}\nAuthors: {', '.join(p.authors or [])}\n"
        f"Venue/Date: {p.published_at.isoformat() if p.published_at else 'unknown'}\n"
        f"TLDR: {p.tldr or ''}\n"
        f"Abstract: {(p.abstract or '')[:1500]}"
        for i, p in enumerate(papers)
    )
    venue_clause = (
        f"Target venue: {venue}. Calibrate tone and length to that venue's norms."
        if venue else "No target venue specified — use neutral academic prose."
    )
    topic_clause = (
        f"The user's contribution / paper topic: {topic}"
        if topic else "(The user did not specify their contribution. Frame the section "
                      "as a survey grounded in the cited papers.)"
    )

    fallback = _fallback_draft(section, papers)
    try:
        from app.adapters.llm import get_llm_adapter

        llm = get_llm_adapter()
        prompt = (
            f"Draft the {section.replace('_', ' ').upper()} section of an "
            "academic paper, in markdown.\n\n"
            f"SECTION GUIDE: {guide}\n"
            f"{topic_clause}\n"
            f"{venue_clause}\n"
            f"User profile: expertise={expertise}, orientation={orientation}.\n\n"
            "STRICT RULES:\n"
            "1. EVERY claim must cite at least one of the supplied papers using "
            "[1], [2], etc. Uncited prose is forbidden.\n"
            "2. Do NOT invent papers, authors, methods, or results. Stay strictly "
            "within the supplied source material.\n"
            "3. Use academic register: precise, sober, no marketing language, "
            "no superlatives like 'revolutionary' or 'breakthrough' unless quoting "
            "from a source.\n"
            "4. Open with a 1-sentence framing of what this section will cover. "
            "Close with a 1-sentence transition that frames what comes next.\n"
            "5. Keep length proportional to the section: Related Work ~400-700 "
            "words, Methodology ~600-1000 words, Limitations ~150-300 words.\n"
            "6. End with a SOURCES section listing each cited paper as "
            "'[N] Title — Authors'.\n\n"
            f"Cited source papers:\n{paper_block}"
        )
        res = await llm.complete(
            [{"role": "user", "content": prompt}],
            llm.quality_model,
            max_tokens=2400,
            temperature=0.25,
        )
        text = (res.text or "").strip()
        return text or fallback
    except Exception as exc:
        log.warning("draft_section LLM fell back: %s", exc)
        return fallback


def _fallback_draft(section: str, papers: list[Paper]) -> str:
    """Skeleton draft when the LLM is unavailable — never empty."""
    bullets = "\n".join(f"- [{i + 1}] {p.title} ({p.namespace_key})"
                        for i, p in enumerate(papers))
    return (
        f"### {section.replace('_', ' ').title()}\n\n"
        "*(LLM synthesis unavailable — skeleton only. Open the cited papers "
        "to draft this section manually.)*\n\n"
        f"Cited sources:\n{bullets}\n"
    )
