"""Context Parser tool — reads files, notes, and URLs from active context.

The RA's active context contains attachments (PDFs, notes, images, URLs,
paper refs, or any uploaded file) that the user wants the assistant to
reason over. This tool materialises their text content so the synthesizer
can include them as grounded primary evidence.

Supports every attachment kind stored in AssistantAttachment.kind:
  note       — inline text written in the UI
  url        — fetches the URL and extracts readable text
  pdf        — extracted text is already stored in att.content at upload time
  image      — vision caption/OCR is stored in att.content at upload time
  file       — any uploaded file whose text was extracted at upload time
  paper_ref  — loads the paper's abstract + key concepts from the DB
"""

from __future__ import annotations

import logging
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.models.assistant import AssistantAttachment

log = logging.getLogger(__name__)

_MAX_CHARS_PER_ITEM = 8000  # generous but bounded token budget per attachment


class ParseContextInput(BaseModel):
    session_id: str = Field(default="", description="Session ID (injected automatically by orchestrator).")
    query: str = Field(default="", max_length=500, description="Optional keyword to focus extraction.")
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class ParseContextOutput(BaseModel):
    items: list[dict]
    total: int


class ParseContextTool:
    """Extract text from attachments (PDFs, notes, URLs, images, files) in the active context."""

    name = "parse_context"
    summary = (
        "Read and extract text content from files, notes, URLs, images, and papers "
        "attached to the current session. Returns structured text chunks usable as "
        "grounded evidence — equivalent to indexing the material but without persisting "
        "it. Use whenever the user references an attachment ('the PDF I uploaded', "
        "'the note I added', 'the URL I shared') or when active context items are "
        "likely relevant to the query."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = False
    streamable = False
    input_schema = ParseContextInput
    output_schema = ParseContextOutput

    async def run(self, ctx: ToolContext, params: ParseContextInput) -> ToolResult:
        await ctx.emit_progress(10, "Loading active context attachments")
        try:
            result = await ctx.db.execute(
                select(AssistantAttachment).where(
                    AssistantAttachment.session_id == ctx.session_id
                )
            )
            attachments = list(result.scalars())
        except Exception as exc:
            log.warning("parse_context: DB query failed: %s", exc)
            return ToolResult(output={"items": [], "total": 0}, summary="Context load failed")

        if not attachments:
            return ToolResult(
                output={"items": [], "total": 0},
                summary="No attachments in active context",
            )

        await ctx.emit_progress(30, f"Parsing {len(attachments)} attachment(s)")
        items: list[dict] = []
        for att in attachments:
            kind = att.kind
            label = att.label or att.url or kind
            text = await self._extract_text(att, ctx)

            if not text:
                continue

            # Keyword filter: if query given and keyword absent, still include but truncate.
            if params.query and params.query.lower() not in text.lower():
                text = text[:1000]

            items.append({
                "id": str(att.id),
                "kind": kind,
                "label": label,
                "text": text[:_MAX_CHARS_PER_ITEM],
                "chars": min(len(text), _MAX_CHARS_PER_ITEM),
            })

        await ctx.emit_progress(100, f"Parsed {len(items)} context item(s)")
        return ToolResult(
            output={"items": items, "total": len(items)},
            summary=f"Extracted text from {len(items)} context attachment(s)",
        )

    async def _extract_text(self, att: AssistantAttachment, ctx: ToolContext) -> str:
        """Return the best text representation for any attachment kind."""
        kind = att.kind

        # note / pdf / image / file — text was extracted at upload time and stored in content
        if kind in ("note", "pdf", "image", "file"):
            return att.content or ""

        # url — fetch the page and extract readable text
        if kind == "url":
            url = att.url
            if not url:
                return att.content or ""
            # Return cached content if already extracted
            if att.content:
                return att.content
            return await self._fetch_url_text(url)

        # paper_ref — load abstract + concepts from DB
        if kind == "paper_ref":
            if att.paper_id:
                return await self._load_paper_text(att.paper_id, ctx)
            return att.content or ""

        # Any other kind (future-proof): fall back to content field
        return att.content or ""

    async def _fetch_url_text(self, url: str) -> str:
        """Fetch a URL and return cleaned readable text (best-effort)."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "ResearchFlow/1.0"})
                if resp.status_code != 200:
                    log.warning("parse_context: URL fetch %s returned %d", url, resp.status_code)
                    return ""
                raw = resp.text
        except Exception as exc:
            log.warning("parse_context: URL fetch failed for %s: %s", url, exc)
            return ""

        return _strip_html(raw)

    async def _load_paper_text(self, paper_id: UUID, ctx: ToolContext) -> str:
        """Load abstract + key concepts for a paper_ref attachment."""
        try:
            from app.models.paper import Paper
            result = await ctx.db.execute(
                select(Paper).where(Paper.id == paper_id)
            )
            paper = result.scalar_one_or_none()
            if not paper:
                return ""
            parts = []
            if paper.title:
                parts.append(f"Title: {paper.title}")
            if paper.authors:
                parts.append(f"Authors: {', '.join((paper.authors or [])[:6])}")
            if paper.abstract:
                parts.append(f"\nAbstract:\n{paper.abstract}")
            if paper.key_concepts:
                parts.append(f"\nKey Concepts: {', '.join((paper.key_concepts or [])[:10])}")
            if paper.tldr:
                parts.append(f"\nTL;DR: {paper.tldr}")
            return "\n".join(parts)
        except Exception as exc:
            log.warning("parse_context: paper_ref load failed paper_id=%s: %s", paper_id, exc)
            return ""


def _strip_html(html: str) -> str:
    """Very lightweight HTML-to-text: strip tags, collapse whitespace."""
    import re
    # Remove script/style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip all remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    html = (html
        .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " "))
    # Collapse whitespace
    html = re.sub(r"\s{3,}", "\n\n", html)
    return html.strip()


parse_context_tool = ParseContextTool()
