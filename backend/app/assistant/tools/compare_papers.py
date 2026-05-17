"""Compare-papers tool — side-by-side comparison along researcher dimensions.

Pulls 2-5 papers from the user's corpus and produces a structured comparison
matrix (problem framing, methods, datasets, key results, limitations,
practical maturity). Pure composition over Paper rows + a quality LLM call;
no external API hits.
"""

from __future__ import annotations

import json
import logging
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.models.paper import Paper

log = logging.getLogger(__name__)


_DIMENSIONS = (
    "problem_framing",
    "methodology",
    "datasets_or_settings",
    "key_results",
    "limitations",
    "practical_maturity",
)


class ComparePapersInput(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=5)
    focus: str = Field(default="", max_length=240,
                       description="Optional comparison focus, e.g. 'data efficiency' or 'safety'")


class ComparePapersOutput(BaseModel):
    columns: list[dict]            # one entry per paper: {paper_id, title, authors, namespace_key, source_url}
    rows: list[dict]               # one entry per dimension: {dimension, cells: {paper_id: text}}
    notes: str                     # free-text caveats / cross-cutting observations


class ComparePapersTool:
    """Side-by-side structured comparison of 2-5 papers."""

    name = "compare_papers"
    summary = (
        "Build a structured comparison matrix over 2-5 papers along research "
        "dimensions: problem framing, methodology, datasets, key results, "
        "limitations, practical maturity. Use when the user asks 'compare X "
        "vs Y', 'how do these papers differ', or wants to evaluate trade-offs. "
        "Returns a row-per-dimension, column-per-paper matrix."
    )
    cost_class = "moderate"
    side_effects = False
    cancellable = True
    streamable = True
    input_schema = ComparePapersInput
    output_schema = ComparePapersOutput

    async def run(self, ctx: ToolContext, params: ComparePapersInput) -> ToolResult:
        paper_uuids: list[UUID] = []
        for pid in params.paper_ids:
            try:
                paper_uuids.append(UUID(str(pid)))
            except ValueError:
                continue
        if len(paper_uuids) < 2:
            return ToolResult(
                output={"columns": [], "rows": [], "notes": "Need at least 2 valid paper ids."},
                summary="compare_papers skipped — fewer than 2 valid paper ids",
            )

        await ctx.emit_progress(20, f"Loading {len(paper_uuids)} papers")
        result = await ctx.db.execute(select(Paper).where(Paper.id.in_(paper_uuids)))
        papers = list(result.scalars())
        # Preserve the requested order rather than DB order.
        order = {str(pid): i for i, pid in enumerate(paper_uuids)}
        papers.sort(key=lambda p: order.get(str(p.id), 999))

        if len(papers) < 2:
            return ToolResult(
                output={"columns": [], "rows": [], "notes": "Could not resolve enough papers."},
                summary="compare_papers skipped — could not load enough papers",
            )

        columns = [
            {
                "paper_id": str(p.id),
                "title": p.title,
                "authors": list(p.authors or []),
                "namespace_key": p.namespace_key,
                "source_url": p.source_url,
                "tldr": p.tldr,
            }
            for p in papers
        ]

        await ctx.emit_progress(55, "Composing structured comparison")
        rows = await _llm_comparison(papers, params.focus)
        notes = (
            f"Comparison generated from each paper's TL;DR + abstract. "
            f"Focus: {params.focus or 'general comparison'}."
        )
        await ctx.emit_progress(100, f"Compared {len(papers)} papers across {len(rows)} dimensions")

        return ToolResult(
            output={"columns": columns, "rows": rows, "notes": notes},
            summary=f"Compared {len(papers)} papers across {len(rows)} dimensions",
            citations=[c["paper_id"] for c in columns],
            artifacts=[{
                "kind": "comparison",
                "ref_id": ",".join(c["paper_id"] for c in columns),
                "title": f"Comparison · {len(papers)} papers",
                "preview": {"focus": params.focus, "dimensions": [r["dimension"] for r in rows]},
            }],
        )


async def _llm_comparison(papers: list[Paper], focus: str) -> list[dict]:
    """Ask the quality LLM to fill a JSON comparison matrix.

    Falls back to a deterministic abstract-snippet matrix when the LLM call
    fails so the user always sees *something* useful.
    """
    matrix_block = _matrix_fallback(papers)
    try:
        from app.adapters.llm import get_llm_adapter

        llm = get_llm_adapter()
        paper_block = "\n\n".join(
            f"[P{i + 1}] id={p.id} title={p.title}\nAuthors: {', '.join(p.authors or [])}\n"
            f"TLDR: {p.tldr or ''}\nAbstract: {(p.abstract or '')[:1200]}"
            for i, p in enumerate(papers)
        )
        focus_clause = f"Comparison focus: {focus}\n" if focus else ""
        prompt = (
            "Return ONLY a JSON object that compares the supplied papers along "
            "these dimensions: problem_framing, methodology, datasets_or_settings, "
            "key_results, limitations, practical_maturity. For each cell, write 1-2 "
            "concise sentences grounded in the paper's text. If a paper does not "
            "address a dimension, write \"not addressed\".\n\n"
            "JSON shape: {\"matrix\": {\"<dimension>\": {\"<paper_id>\": \"<text>\"}}}\n\n"
            f"{focus_clause}"
            f"Papers:\n{paper_block}"
        )
        raw = await llm.complete_structured(
            [{"role": "user", "content": prompt}],
            llm.quality_model,
            {
                "type": "object",
                "properties": {
                    "matrix": {"type": "object"},
                },
                "required": ["matrix"],
            },
        )
        matrix = (raw or {}).get("matrix") or {}
        if isinstance(matrix, dict) and matrix:
            rows = []
            for dim in _DIMENSIONS:
                cells = matrix.get(dim) or {}
                if not isinstance(cells, dict):
                    continue
                rows.append({"dimension": dim, "cells": {str(k): str(v) for k, v in cells.items()}})
            if rows:
                return rows
        log.warning("compare_papers: LLM returned empty matrix; using fallback")
    except Exception as exc:
        log.warning("compare_papers LLM fell back: %s", exc)
    return matrix_block


def _matrix_fallback(papers: list[Paper]) -> list[dict]:
    """Deterministic comparison from TL;DR + abstract snippets."""
    rows = []
    snippets = {
        str(p.id): (p.tldr or (p.abstract or "")[:400]).strip() or "—"
        for p in papers
    }
    rows.append({"dimension": "problem_framing", "cells": snippets})
    rows.append({"dimension": "methodology", "cells": {
        str(p.id): ", ".join((p.methods_used or [])[:6]) or "—" for p in papers
    }})
    rows.append({"dimension": "key_concepts", "cells": {
        str(p.id): ", ".join((p.key_concepts or [])[:6]) or "—" for p in papers
    }})
    return rows
