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
    # Distinct ``section_type`` values for the chunks that backed the
    # answer (e.g. ["method", "results"]). The full-paper verification
    # middleware reads this together with ``chunk_positions`` to label
    # evidence quality honestly. Section names alone are venue-
    # specific (math papers don't have "method" sections, biology
    # papers say "materials and methods", etc.) so the structural
    # signal below carries the load; ``sections_used`` is kept for the
    # one canonical case the parser stamps consistently: chunks
    # tagged ``abstract`` always count as abstract-only regardless of
    # their position.
    sections_used: list[str] = []
    # Relative position of each grounding chunk inside the paper
    # (chunk_index / total_chunks, in ``[0.0, 1.0]``). This is the
    # primary evidence-tier signal — namespace-agnostic by design,
    # because the canonical paper structure (abstract → introduction
    # → method → results → discussion → conclusion) is consistent
    # across CS, physics, biology, math, economics, and clinical
    # venues. Empty when the paper has only an abstract row indexed.
    chunk_positions: list[float] = []
    # Total chunk count in the paper. Useful telemetry — a paper with
    # 3 chunks is structurally weaker evidence than one with 60.
    total_chunks: int = 0


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

        # Reject empty / placeholder identifiers up front. We also
        # reject obvious placeholder patterns that survived the LLM
        # preflight (TBD, PAPER_ID, all-caps stubs, etc.) so the
        # planner sees a useful diagnostic instead of a "Paper not
        # found:" trailing-colon trace. The list is conservative —
        # a UUID-shaped or arXiv-id-shaped input always passes
        # through to real lookup.
        id_in = (params.paper_id or "").strip()
        title_in = (params.paper_title or "").strip()

        def _looks_like_placeholder(v: str) -> bool:
            if not v:
                return False
            low = v.lower()
            if low in {"tbd", "todo", "placeholder", "paper_id", "id", "n/a", "na",
                       "none", "null", "undefined", "fill_me", "fill_in", "?", "??"}:
                return True
            # Template-variable shapes that survived the preflight repair —
            # e.g. ``{{best_supporting_paper_id}}`` or ``${paper_id}``.
            if "{{" in v or "}}" in v or v.startswith("${") and v.endswith("}"):
                return True
            # All-uppercase identifier-like stubs (``PAPER_ID``,
            # ``BEST_PAPER``) are almost always placeholders — real
            # arXiv ids contain digits + dots and UUIDs contain hex.
            stripped = v.replace("_", "").replace("-", "")
            if stripped.isalpha() and stripped.isupper() and len(stripped) <= 24:
                return True
            return False

        id_is_placeholder = _looks_like_placeholder(id_in)
        title_is_placeholder = _looks_like_placeholder(title_in)
        if (not id_in or id_is_placeholder) and (not title_in or title_is_placeholder):
            # Surface a structured ``recoverable_hint`` so the loop's
            # middleware chain can decide to schedule a retrieval
            # tool before the next paper_qa attempt.
            reason = (
                "both paper_id and paper_title are placeholders"
                if (id_is_placeholder or title_is_placeholder)
                else "neither paper_id nor paper_title was supplied"
            )
            return ToolResult(
                output={
                    "answer": "",
                    "paper_title": "",
                    "paper_id": "",
                    "chunks_used": 0,
                    "found": False,
                    "sections_used": [],
                    "chunk_positions": [],
                    "total_chunks": 0,
                    "recoverable_hint": "retrieve_then_retry",
                },
                summary=(
                    f"paper_qa skipped — {reason}. Run a retrieval tool "
                    "(deep_search / arxiv_import / literature_survey) first to "
                    "populate the paper ledger, then call paper_qa with the "
                    "concrete paper_id surfaced by retrieval."
                ),
            )

        await ctx.emit_progress(15, f"Locating paper: {params.paper_title or params.paper_id}")

        # Step 1: Resolve paper.
        #
        # Papers are global (no per-user ownership column on the Paper
        # model) — access is gated upstream by namespace subscription
        # rather than row-level ownership. Previously this query filtered
        # by ``Paper.user_id == ctx.user_id``, which raised
        # ``AttributeError`` at column access (the attribute simply does
        # not exist); the loop swallowed the exception as a tool failure,
        # so paper_qa silently never returned a paper. The full-paper
        # verification gate forces this tool, so the silent failure also
        # blocked every forced strong-claim verification round.
        paper = None
        if params.paper_id.strip():
            try:
                pid = UUID(params.paper_id.strip())
                result = await ctx.db.execute(
                    select(Paper).where(Paper.id == pid)
                )
                paper = result.scalar_one_or_none()
            except Exception as exc:
                log.warning("paper_qa: invalid paper_id %s: %s", params.paper_id, exc)

        if paper is None and params.paper_title.strip():
            title_lower = params.paper_title.strip().lower()
            words = title_lower.split()[:4]
            # Title fallback: bound the candidate scan to a recent slice
            # so a global title search stays cheap. The lookup is only
            # used when the model could not produce a paper_id from the
            # ledger.
            result = await ctx.db.execute(
                select(Paper).order_by(Paper.ingested_at.desc()).limit(200)
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
                output={
                    "answer": "", "paper_title": "", "paper_id": "",
                    "chunks_used": 0, "found": False, "sections_used": [],
                    "chunk_positions": [], "total_chunks": 0,
                },
                summary=f"Paper not found: {params.paper_title or params.paper_id}",
            )

        await ctx.emit_progress(35, f"Retrieved paper: {paper.title}")

        # Step 2: Fetch paper chunks (all, ordered). We also carry the
        # per-chunk ``section_type`` (abstract / introduction / method /
        # results / discussion / …) so the full-paper verification
        # middleware can later tag the strong claim with the actual
        # evidence tier the answer drew from. A claim verified against
        # the methods/experiments section is materially stronger than
        # one only echoed from the abstract.
        chunks_result = await ctx.db.execute(
            select(PaperChunk.content, PaperChunk.chunk_index, PaperChunk.section_type).where(
                PaperChunk.paper_id == paper.id,
                PaperChunk.content.isnot(None),
            ).order_by(PaperChunk.chunk_index)
        )
        all_rows = [
            (row[0] or "", (row[2] or "abstract"))
            for row in chunks_result.fetchall()
            if row[0]
        ]
        all_chunks = [r[0] for r in all_rows]
        chunk_sections = [r[1] for r in all_rows]

        if not all_chunks:
            # Fall back to abstract — single-row case, section_type
            # collapses to "abstract" by definition.
            all_chunks = [paper.abstract or ""]
            chunk_sections = ["abstract"]

        # Step 3: Score chunks by keyword relevance to the question
        q_words = set(params.question.lower().split())
        def _chunk_score(c: str) -> int:
            c_lower = c.lower()
            return sum(1 for w in q_words if len(w) > 3 and w in c_lower)

        ranked = sorted(enumerate(all_chunks), key=lambda x: _chunk_score(x[1]), reverse=True)
        # Re-order by original chunk index for coherence and capture
        # the section_type set the answer is grounded in.
        selected_indices = {ranked[i][0] for i in range(min(_MAX_CHUNKS, len(ranked)))}
        top_chunks = [all_chunks[i] for i in sorted(selected_indices)]
        sections_used = sorted({
            (chunk_sections[i] or "abstract").lower()
            for i in selected_indices
        })
        # Structural position signal. The evidence-tier classifier in
        # claim_ledger uses these positions — namespace-agnostic, so
        # math / biology / physics / economics papers all get tier
        # labels without needing per-discipline section-name
        # vocabularies. Guard the division: a single-chunk paper
        # collapses to a single position 0.0 (abstract-only).
        total = max(len(all_chunks), 1)
        chunk_positions = sorted({
            (float(i) / float(total)) if total > 1 else 0.0
            for i in selected_indices
        })

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
                "sections_used": sections_used,
                "chunk_positions": chunk_positions,
                "total_chunks": total,
                "answer": answer,
                "paper_title": paper.title or "",
                "paper_id": str(paper.id),
                "chunks_used": len(top_chunks),
                "found": True,
            },
            summary=f"Paper Q&A: '{paper.title[:60]}' — {len(top_chunks)} chunks used",
        )


paper_qa_tool = PaperQATool()
