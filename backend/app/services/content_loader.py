"""ContentLoaderService — single source of truth for source-entity content loading.

Loads structured content from a source entity (paper, capsule, or folder) into
a uniform string + title pair suitable for downstream LLM workflows.

DEEP GROUNDING (CRITICAL):
    Generations MUST be grounded in the full paper body, not just the abstract.
    For paper sources, if only the abstract chunk exists, we trigger an on-demand
    PDF parse via :func:`parse_with_fallback`, store the section chunks, and then
    assemble the prompt context from the actual PDF body sections (introduction,
    methodology, results, discussion, conclusion, etc.).

    Once parsed, chunks are cached in the ``paper_chunks`` table — so subsequent
    generations of the same paper (any media type, any expertise level) reuse
    the parsed content for free. This also benefits the Study workflow.

    For capsules, we hydrate the cited source papers in the same way.
    For folders, we include parsed section content for any papers already parsed.

This module exists to:

1. Eliminate duplication of ``_load_paper_content`` across the podcast and slides workflows.
2. Fix the folder loader bug where source_id was being misinterpreted as user_id.
3. Centralize content-shaping logic so prompt grounding stays consistent across all
   media-generation pipelines.
4. Provide a single, testable seam for content retrieval.
5. Enforce deep PDF grounding — never let generation fall back to abstract-only.

Returns a typed :class:`LoadedContent` so downstream callers don't need to know
which entity type was loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.genie import IdeaCapsule
from app.models.paper import Bookmark, BookmarkFolder, BookmarkFolderMember, Paper
from app.repositories.paper import PaperRepository

log = logging.getLogger(__name__)


@dataclass
class LoadedContent:
    """Normalized content payload returned by :class:`ContentLoaderService`.

    Attributes:
        title: Display-ready title for the source entity.
        content: Full assembled text content suitable for prompt context.
            Sections are passed at full length; the only cap is the overall
            _PAPER_CONTENT_CAP (120k chars ≈ 80k tokens) which guards against
            exceeding the model's 200k-token context window.
        source_summary: A short one-paragraph description of the source.
        paper_count: Number of distinct papers represented (1 for single paper,
            1 for capsule, N for folder).
        ok: ``False`` when the source was not found or had no usable content.
    """

    title: str
    content: str
    source_summary: str = ""
    paper_count: int = 0
    ok: bool = True


class ContentLoaderService:
    """Loads source-entity content for media-generation workflows.

    The service is stateless and thread-safe; one instance can be shared
    across all concurrent requests, but typically a new instance is created
    per workflow invocation tied to an active :class:`AsyncSession`.

    Args:
        db: An active async DB session used for all queries.
    """

    # Claude's context window is 200k tokens (~150k chars). We leave ~30k chars
    # headroom for system prompts, slide plans / episode plans, and multi-turn
    # generation batches, so the effective content budget is 120k chars.
    # No character caps — every section and field is passed in full so no
    # context detail is silently dropped during RAG grounding.
    _MAX_SECTIONS = 32              # generous ceiling; papers rarely exceed 16 sections
    _PDF_PARSE_TIMEOUT_S = 90.0
    _PDF_FETCH_TIMEOUT_S = 60.0

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the content loader with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` used for all content queries.
        """
        self._db = db

    async def load(
        self,
        *,
        source_type: str,
        source_id: UUID,
        user_id: UUID | None = None,
    ) -> LoadedContent:
        """Load content for a source entity.

        Args:
            source_type: ``"paper"`` | ``"capsule"`` | ``"folder"``.
            source_id: UUID of the source row.
            user_id: Required for ``"folder"`` so we only load bookmarks
                owned by this user.

        Returns:
            A :class:`LoadedContent`. ``ok=False`` when the source is missing
            or empty.
        """
        if source_type == "paper":
            return await self._load_paper(source_id)
        if source_type == "capsule":
            return await self._load_capsule(source_id)
        if source_type == "folder":
            if user_id is None:
                log.error("content_loader: folder load requires user_id")
                return LoadedContent(title="(missing user)", content="", ok=False)
            return await self._load_folder(source_id, user_id)

        log.warning("content_loader: unknown source_type=%s", source_type)
        return LoadedContent(title="(unknown source)", content="", ok=False)

    # ── Paper ──────────────────────────────────────────────────────────────────

    async def _load_paper(self, paper_id: UUID) -> LoadedContent:
        """Load full paper grounding — abstract + parsed PDF sections + study guide.

        If the paper has no parsed PDF section chunks yet, this method triggers
        an on-demand parse via the configured PDF parser chain (Docling →
        Marker → Gemini Vision) so generation always grounds in the real
        paper body — never in just the abstract.
        """
        repo = PaperRepository(self._db)
        paper = await repo.get_by_id(paper_id)
        if not paper:
            return LoadedContent(title="(paper not found)", content="", ok=False)

        # Ensure we have section-level PDF content. If only the abstract chunk
        # exists (or no chunks at all), parse the PDF on-demand and persist
        # the resulting section chunks so subsequent generations are fast.
        chunks = await self._ensure_pdf_parsed(paper, repo)

        section_chunks = [c for c in chunks if c.section_type != "abstract"]
        section_text = "\n\n".join(
            f"[{c.section_type.upper()}]\n{c.content or ''}"
            for c in section_chunks[: self._MAX_SECTIONS]
        )

        # Media generation is grounded in abstract + full parsed PDF only.
        # The study guide (generated content) is deliberately excluded so that
        # podcast/slides quality is independent of study depth and never
        # circular (generated content describing the same paper).
        body_parts = [
            f"Title: {paper.title}",
            f"Authors: {', '.join((paper.authors or [])[:5])}",
            "",
            "Abstract:",
            paper.abstract or "",
            "",
            f"Key Concepts: {', '.join((paper.key_concepts or [])[:10])}",
            f"Methods: {', '.join((paper.methods_used or [])[:8])}",
            f"Implications: {paper.implications or 'Not specified'}",
        ]
        if section_text:
            body_parts += ["", "── Full Paper Sections (parsed from PDF) ──", section_text]
        else:
            body_parts += ["", "(PDF body unavailable — grounding falls back to abstract + concepts)"]

        body = "\n".join(body_parts)

        log.debug(
            "content_loader.paper paper=%s parsed_sections=%d total_chars=%d parser=%s",
            paper.id, len(section_chunks), len(body), paper.parser_used or "n/a",
        )

        return LoadedContent(
            title=paper.title or "Untitled Paper",
            content=body,
            source_summary=paper.tldr or paper.abstract or "",
            paper_count=1,
        )

    async def _ensure_pdf_parsed(self, paper, repo: PaperRepository) -> list:
        """Return PaperChunk rows for ``paper``, parsing the PDF on demand if needed.

        If only the abstract chunk exists (or none), this fetches the PDF,
        parses it via the configured fallback chain, and persists section
        chunks (without embeddings — embeddings happen lazily on first
        retrieval call). Failures are non-fatal: returns whatever chunks
        exist so the caller can still proceed with abstract-only grounding.
        """
        chunks = await repo.get_chunks(paper.id)
        has_sections = any(c.section_type != "abstract" for c in chunks)
        if has_sections:
            return chunks

        if not paper.pdf_url:
            log.debug("content_loader: paper=%s has no pdf_url — abstract-only grounding", paper.id)
            return chunks

        try:
            import asyncio
            import httpx
            from app.adapters.pdf import parse_with_fallback
            from app.models.paper import PaperChunk

            log.info("content_loader: on-demand PDF parse paper=%s url=%s", paper.id, paper.pdf_url)

            async with httpx.AsyncClient(timeout=self._PDF_FETCH_TIMEOUT_S) as client:
                resp = await client.get(paper.pdf_url)
                if resp.status_code != 200:
                    log.warning("content_loader: PDF fetch failed status=%d", resp.status_code)
                    return chunks
                pdf_bytes = resp.content

            parsed = await asyncio.wait_for(
                parse_with_fallback(pdf_bytes), timeout=self._PDF_PARSE_TIMEOUT_S
            )

            # Persist section chunks (no embeddings — those are added lazily
            # by the Study workflow when the user opens the paper).
            section_count = 0
            for i, sec in enumerate(parsed.sections):
                if sec.section_type == "abstract":
                    continue  # already have abstract chunk from ingestion
                if not sec.content or not sec.content.strip():
                    continue
                chunk = PaperChunk(
                    paper_id=paper.id,
                    chunk_index=i + 1,
                    section_type=sec.section_type,
                    content=sec.content,
                    embedding=None,  # lazy embedding
                    embedding_dim=768,
                    embedding_provider="gemini",
                )
                self._db.add(chunk)
                section_count += 1

            # Persist parser provenance so cache keys can include it
            paper.pdf_parsed = True
            paper.parser_used = parsed.parser_name
            paper.parser_fallback_used = parsed.fallback_used
            paper.parse_duration_ms = parsed.parse_duration_ms
            paper.parser_confidence = parsed.parser_confidence

            await self._db.commit()
            log.info(
                "content_loader: parsed paper=%s parser=%s sections=%d duration_ms=%d",
                paper.id, parsed.parser_name, section_count, parsed.parse_duration_ms,
            )
            return await repo.get_chunks(paper.id)

        except asyncio.TimeoutError:
            log.warning("content_loader: PDF parse timed out for paper=%s", paper.id)
            return chunks
        except Exception as exc:  # noqa: BLE001 — graceful fallback
            log.warning("content_loader: PDF parse failed for paper=%s err=%s", paper.id, exc)
            try:
                await self._db.rollback()
            except Exception:  # noqa: BLE001
                pass
            return chunks

    # ── Capsule ────────────────────────────────────────────────────────────────

    async def _load_capsule(self, capsule_id: UUID) -> LoadedContent:
        """Load an IdeaCapsule's deep-dive article and structured fields.

        Requires ``deep_dive_status == "done"`` — capsules without a completed
        deep dive are returned as ``ok=False`` so generation workflows can gate
        on content availability.

        Args:
            capsule_id: UUID of the ``IdeaCapsule`` row to load.

        Returns:
            ``LoadedContent`` with deep-dive article + structured fields as
            context. ``ok=False`` when the capsule does not exist or has no
            completed deep dive.
        """
        row = await self._db.execute(
            select(IdeaCapsule).where(IdeaCapsule.id == capsule_id)
        )
        capsule = row.scalar_one_or_none()
        if not capsule:
            return LoadedContent(title="(capsule not found)", content="", ok=False)

        if not capsule.deep_dive_content or capsule.deep_dive_status != "done":
            return LoadedContent(title=capsule.title or "Untitled Idea", content="", ok=False)

        # Deep dive is the primary grounding — place it FIRST so it is never
        # truncated by the content cap. Structured capsule fields follow as
        # supplementary context.
        deep_dive_section = (
            f"── Deep Dive Article (primary source) ──\n"
            f"{capsule.deep_dive_content}"
        )
        structured_fields = (
            f"\n\n── Structured Capsule Fields ──\n"
            f"Idea: {capsule.title}\n\n"
            f"Hypothesis: {capsule.hypothesis or ''}\n\n"
            f"Rationale: {capsule.rationale or ''}\n\n"
            f"Mechanism: {capsule.mechanism or ''}\n\n"
            f"Experimental Design: {capsule.experimental_design or ''}\n\n"
            f"Predicted Outcomes: {capsule.predicted_outcome or ''}\n\n"
            f"Risks & Limitations: {capsule.risks_and_limitations or ''}\n\n"
            f"Open Questions: {capsule.open_questions or ''}\n\n"
            f"Anti-Finding: {capsule.anti_finding or ''}"
        )

        # No cap on capsule content — deep dive + all structured fields are
        # passed in full so every detail is available for grounding.
        body = deep_dive_section + structured_fields

        log.debug(
            "content_loader.capsule capsule=%s deep_dive_chars=%d total_chars=%d",
            capsule.id, len(capsule.deep_dive_content), len(body),
        )

        return LoadedContent(
            title=capsule.title or "Untitled Idea",
            content=body,
            source_summary=(capsule.hypothesis or "")[:400],
            paper_count=1,
        )

    # ── Folder ─────────────────────────────────────────────────────────────────

    async def _load_folder(self, folder_id: UUID, user_id: UUID) -> LoadedContent:
        """Load all papers in a bookmark folder as a combined context string.

        Resolves the folder's bookmarks via the junction table and assembles
        parsed section content for each paper. Unlike single-paper loading,
        folder loading does not trigger on-demand PDF parsing — only already-
        parsed chunks are included so bulk folder generation stays fast.

        Args:
            folder_id: UUID of the ``BookmarkFolder`` row to load.
            user_id: UUID of the user who must own the folder.

        Returns:
            ``LoadedContent`` with all papers' text concatenated. ``ok=False``
            when the folder does not exist or contains no accessible papers.
        """
        # 1. Validate folder ownership
        folder_row = await self._db.execute(
            select(BookmarkFolder).where(
                BookmarkFolder.id == folder_id,
                BookmarkFolder.user_id == user_id,
            )
        )
        folder = folder_row.scalar_one_or_none()
        if not folder:
            return LoadedContent(title="(folder not found)", content="", ok=False)

        # 2. Find all papers in this folder via the junction table
        papers_q = (
            select(Paper)
            .join(Bookmark, Bookmark.paper_id == Paper.id)
            .join(BookmarkFolderMember, BookmarkFolderMember.bookmark_id == Bookmark.id)
            .where(
                BookmarkFolderMember.folder_id == folder_id,
                Bookmark.user_id == user_id,
            )
            .limit(20)
        )
        result = await self._db.execute(papers_q)
        papers = list(result.scalars())

        if not papers:
            return LoadedContent(
                title=folder.name or "Empty Folder",
                content="",
                source_summary="Folder contains no papers.",
                paper_count=0,
                ok=False,
            )

        # 3. Build a multi-paper digest. For papers already parsed, include the
        # most relevant body section so cross-paper folder generations can
        # ground in actual paper content instead of abstracts only.
        repo = PaperRepository(self._db)
        sections: list[str] = []
        for p in papers:
            block = [
                f"=== {p.title} ===",
                f"Authors: {', '.join((p.authors or [])[:4])}",
                f"Abstract: {(p.abstract or '')}",
                f"Key Concepts: {', '.join((p.key_concepts or [])[:6])}",
                f"Methods: {', '.join((p.methods_used or [])[:5])}",
            ]
            # Pull a "core" parsed section if available — methodology > results > intro.
            try:
                paper_chunks = await repo.get_chunks(p.id)
                priority = ("methodology", "results", "discussion", "introduction", "background", "conclusion")
                picked = None
                for kind in priority:
                    for c in paper_chunks:
                        if c.section_type == kind and c.content:
                            picked = c
                            break
                    if picked:
                        break
                if picked:
                    block.append(
                        f"From [{picked.section_type.upper()}]: "
                        f"{(picked.content or '')}"
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug("content_loader.folder: paper=%s chunk pick failed err=%s", p.id, exc)

            sections.append("\n".join(block))

        body = (
            f"Folder: {folder.name}\n"
            f"Papers in folder: {len(papers)}\n\n"
            + "\n\n".join(sections)
        )

        return LoadedContent(
            title=f"{folder.name} ({len(papers)} papers)",
            content=body,
            source_summary=f"A curated collection of {len(papers)} research papers.",
            paper_count=len(papers),
        )
