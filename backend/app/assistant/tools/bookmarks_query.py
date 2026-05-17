"""Bookmarks-query tool — Q&A grounded in the user's bookmarked papers.

Uses the existing ``run_bookmarks_chat`` workflow which retrieves chunks
from bookmarked papers, runs the RAG pipeline, and returns a grounded
answer with citations. Wrapped here so the Research Assistant planner
can pick it when the user references "my bookmarks", "saved papers",
or wants synthesis over their curated subset.
"""

from __future__ import annotations

import logging
from uuid import UUID

from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.repositories.paper import PaperRepository

log = logging.getLogger(__name__)


class BookmarksQueryInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    folder_id: str | None = None


class BookmarksQueryOutput(BaseModel):
    answer: str
    citations: list[str]
    bookmark_count: int


class BookmarksQueryTool:
    """Run a Q&A grounded in the user's bookmarked papers."""

    name = "bookmarks_query"
    summary = (
        "Answer a question grounded ONLY in the user's bookmarked papers (their "
        "curated subset of the corpus). Use when the user references 'my "
        "bookmarks', 'saved papers', 'my reading list', or wants synthesis over "
        "their personal selection rather than the global corpus. Returns a "
        "cited answer plus the paper IDs used as evidence."
    )
    cost_class = "moderate"
    side_effects = False
    cancellable = True
    streamable = True
    input_schema = BookmarksQueryInput
    output_schema = BookmarksQueryOutput

    async def run(self, ctx: ToolContext, params: BookmarksQueryInput) -> ToolResult:
        # Quick guard: tell the user when there's nothing bookmarked yet so
        # we don't run a heavy workflow over an empty set.
        repo = PaperRepository(ctx.db)
        bookmarks = await repo.get_bookmarks(ctx.user_id)
        if not bookmarks:
            return ToolResult(
                output={
                    "answer": (
                        "You don't have any bookmarked papers yet. Bookmark a few "
                        "from the feed and I can synthesize across them."
                    ),
                    "citations": [],
                    "bookmark_count": 0,
                },
                summary="bookmarks_query skipped — user has no bookmarks",
            )

        await ctx.emit_progress(20, f"Querying across {len(bookmarks)} bookmarked paper(s)")

        from app.workflows.study import run_bookmarks_chat

        # run_bookmarks_chat is an async generator that streams SSE strings.
        # We accumulate chunks into a single answer text + capture citations
        # from the meta event.
        answer_chunks: list[str] = []
        citations: list[str] = []
        try:
            folder_uuid: UUID | None = None
            if params.folder_id:
                try:
                    folder_uuid = UUID(str(params.folder_id))
                except ValueError:
                    folder_uuid = None
            async for raw in run_bookmarks_chat(ctx.user_id, params.query, folder_id=folder_uuid):
                # raw lines are SSE-formatted: "data: {json}\n\n"
                for line in (raw or "").split("\n"):
                    if not line.startswith("data: "):
                        continue
                    try:
                        import json as _json
                        ev = _json.loads(line[6:])
                    except Exception:
                        continue
                    if ev.get("type") == "chunk":
                        answer_chunks.append(str(ev.get("text") or ""))
                    elif ev.get("type") == "meta":
                        cits = ev.get("citations") or ev.get("paper_ids") or []
                        if isinstance(cits, list):
                            citations = [str(c) for c in cits]
        except Exception as exc:
            log.warning("bookmarks_query: workflow failed: %s", exc)
            return ToolResult(
                output={
                    "answer": "Bookmarks Q&A failed mid-stream — try a simpler question or check the bookmarks page.",
                    "citations": [],
                    "bookmark_count": len(bookmarks),
                },
                summary=f"bookmarks_query failed: {type(exc).__name__}",
            )

        await ctx.emit_progress(100, f"Bookmarks answer composed ({len(citations)} citations)")
        answer = "".join(answer_chunks).strip() or "(no answer produced)"
        return ToolResult(
            output={
                "answer": answer,
                "citations": citations,
                "bookmark_count": len(bookmarks),
            },
            summary=f"Bookmarks Q&A grounded in {len(citations)} paper(s)",
            citations=citations,
        )
