"""Paper Q&A tool — targeted question answering over a single paper's full text.

Uses the paper's indexed chunks (DB retrieval) to answer specific questions about
what the paper says, defines, claims, or measures. Goes deeper than deep_search
by focusing all retrieval attention on one paper's content.

Useful when the user wants to drill into a specific paper: "What method does
paper X use?", "Find the exact definition of Y in [paper]", "What results does
[paper] report for benchmark Z?".
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_MAX_CHUNKS = 12          # chunks fed to LLM for synthesis
_CHUNK_CHARS = 800        # chars per chunk passed to LLM


class PaperQAInput(BaseModel):
    question: str = Field(min_length=5, max_length=500, description="Specific question to answer using the paper's content")
    paper_title: str = Field(default="", max_length=300, description="Paper title (partial match OK) to locate the paper")
    paper_id: str = Field(default="", description="Exact paper UUID if known (faster than title search)")
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class PaperQAOutput(BaseModel):
    answer: str
    paper_title: str
    paper_id: str
    chunks_used: int
    found: bool


class PaperQATool:
    """Ask a targeted question about a specific paper using its full indexed text."""

    name = "paper_qa"
    summary = (
        "Ask a specific question about a single paper's content using its full indexed text. "
        "Retrieves the most relevant chunks from the paper and synthesizes a precise answer. "
        "Use when: 'What does [paper] say about X?', 'Find the definition of Y in [paper title]', "
        "'What method/results/benchmark does [paper] report?', user wants to drill into one paper. "
        "Provide paper_id (UUID) for exact lookup, or paper_title for fuzzy search. "
        "Do NOT use for multi-paper questions — use deep_search instead."
    )
    cost_class = "moderate"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = PaperQAInput
    output_schema = PaperQAOutput

    async def run(self, ctx: ToolContext, params: PaperQAInput) -> ToolResult:
        from app.models.paper import Paper, PaperChunk
        from uuid import UUID

        await ctx.emit_progress(15, f"Locating paper: {params.paper_title or params.paper_id}")

        # Step 1: Resolve paper
        paper = None
        if params.paper_id.strip():
            try:
                pid = UUID(params.paper_id.strip())
                result = await ctx.db.execute(
                    select(Paper).where(
                        Paper.id == pid,
                        Paper.user_id == ctx.user_id,
                    )
                )
                paper = result.scalar_one_or_none()
            except Exception as exc:
                log.warning("paper_qa: invalid paper_id %s: %s", params.paper_id, exc)

        if paper is None and params.paper_title.strip():
            title_lower = params.paper_title.strip().lower()
            words = title_lower.split()[:4]
            result = await ctx.db.execute(
                select(Paper).where(
                    Paper.user_id == ctx.user_id,
                ).limit(200)
            )
            all_papers = result.scalars().all()
            # Score by word overlap with title
            def _score(p: Paper) -> int:
                t = (p.title or "").lower()
                return sum(1 for w in words if w in t)
            scored = [(p, _score(p)) for p in all_papers if _score(p) > 0]
            if scored:
                paper = max(scored, key=lambda x: x[1])[0]

        if paper is None:
            return ToolResult(
                output={"answer": "", "paper_title": "", "paper_id": "", "chunks_used": 0, "found": False},
                summary=f"Paper not found: {params.paper_title or params.paper_id}",
            )

        await ctx.emit_progress(35, f"Retrieved paper: {paper.title}")

        # Step 2: Fetch paper chunks (all, ordered)
        chunks_result = await ctx.db.execute(
            select(PaperChunk.content, PaperChunk.chunk_index).where(
                PaperChunk.paper_id == paper.id,
                PaperChunk.content.isnot(None),
            ).order_by(PaperChunk.chunk_index)
        )
        all_chunks = [(row[0] or "") for row in chunks_result.fetchall() if row[0]]

        if not all_chunks:
            # Fall back to abstract
            all_chunks = [paper.abstract or ""]

        # Step 3: Score chunks by keyword relevance to the question
        q_words = set(params.question.lower().split())
        def _chunk_score(c: str) -> int:
            c_lower = c.lower()
            return sum(1 for w in q_words if len(w) > 3 and w in c_lower)

        ranked = sorted(enumerate(all_chunks), key=lambda x: _chunk_score(x[1]), reverse=True)
        top_chunks = [c for _, c in ranked[:_MAX_CHUNKS]]
        # Re-order by original chunk index for coherence
        selected_indices = {ranked[i][0] for i in range(min(_MAX_CHUNKS, len(ranked)))}
        top_chunks = [all_chunks[i] for i in sorted(selected_indices)]

        await ctx.emit_progress(60, f"Synthesizing answer from {len(top_chunks)} chunk(s)…")

        # Step 4: LLM synthesis
        paper_context = "\n\n---\n\n".join(
            f"[Chunk {i+1}]:\n{c[:_CHUNK_CHARS]}" for i, c in enumerate(top_chunks)
        )
        meta = (
            f"Title: {paper.title}\n"
            f"Authors: {', '.join((paper.authors or [])[:4])}\n"
            f"Year: {paper.published_at.year if paper.published_at else 'N/A'}\n"
        )

        prompt = (
            f"You are answering a specific question about a research paper using its indexed text.\n\n"
            f"PAPER:\n{meta}\n"
            f"QUESTION: {params.question}\n\n"
            f"PAPER EXCERPTS:\n{paper_context}\n\n"
            f"Instructions:\n"
            f"• Answer the question precisely using only the paper's content.\n"
            f"• Quote or paraphrase specific passages where relevant.\n"
            f"• If the question cannot be answered from the provided excerpts, say so clearly.\n"
            f"• Be concise but complete. No padding.\n\n"
            f"Answer:"
        )

        answer = ""
        try:
            from app.adapters.llm import get_llm_adapter
            llm = get_llm_adapter()
            res = await llm.complete(
                [{"role": "user", "content": prompt}],
                llm.quality_model,
                max_tokens=1500,
                temperature=0.1,
            )
            answer = (res.text or "").strip()
        except Exception as exc:
            log.warning("paper_qa: LLM synthesis failed: %s", exc)
            answer = f"[Synthesis failed: {exc}] Found {len(top_chunks)} relevant chunks."

        await ctx.emit_progress(100, f"Paper Q&A complete: {paper.title[:50]}")
        return ToolResult(
            output={
                "answer": answer,
                "paper_title": paper.title or "",
                "paper_id": str(paper.id),
                "chunks_used": len(top_chunks),
                "found": True,
            },
            summary=f"Paper Q&A: '{paper.title[:60]}' — {len(top_chunks)} chunks used",
        )


paper_qa_tool = PaperQATool()
