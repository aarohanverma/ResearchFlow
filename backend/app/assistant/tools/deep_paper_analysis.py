"""Deep-paper-analysis tool — comprehensive multi-aspect read of one paper.

Where ``paper_qa`` answers ONE focused question against a paper, this
tool runs four targeted ``paper_qa`` rounds in sequence over the same
paper to produce a structured deep-dive covering:

1. **methods** — the technical approach, architecture, training setup
2. **results** — the headline numbers and what beats what
3. **limitations** — caveats, failure modes, what the paper itself
   admits doesn't work
4. **ablations** — which design choices the paper isolates and what
   each contributes

Returns a structured dict keyed by aspect so the synthesizer can
render the deep-dive as labelled sections rather than free prose.

The user's spec: "RA should be able to dive deep into full paper as
per need or requirement." This tool is the explicit "deep dive"
channel — the planner reaches for it when the user asks for a
thorough read, when a critique demands deeper evidence, or when the
synthesizer needs full-paper grounding instead of abstract-only.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult

log = logging.getLogger(__name__)

# Per the user spec: deep_paper_analysis runs unbounded. The four
# aspects can each take as long as the underlying paper_qa needs.
# The outer ReAct loop / task cancellation still apply, so a runaway
# call isn't truly forever — ``should_cancel`` will trip from the
# orchestrator's wall-clock deadline. We just don't impose our own
# per-tool cap on top, because the user wants the full analysis to
# complete rather than be cut short.


# Aspects we probe in order. Each aspect maps to a focused question
# template that ``paper_qa`` answers against the paper body. Order
# matters: methods → results → limitations → ablations is the
# canonical reading order for ML / scientific papers, so the
# returned report reads coherently.
_ASPECTS: tuple[tuple[str, str], ...] = (
    ("methods", (
        "What is the technical approach? Describe the architecture, "
        "training setup, key algorithmic choices, and any new mechanisms "
        "introduced. Cite specific sections."
    )),
    ("results", (
        "What are the headline experimental results? List the main "
        "benchmarks / metrics, the numbers achieved, and what each "
        "comparison beats. Cite specific tables or figures when possible."
    )),
    ("limitations", (
        "What does this paper acknowledge as its OWN limitations, "
        "caveats, or failure modes? Look in the limitations / discussion "
        "sections specifically. Be honest — what doesn't this paper claim "
        "to solve?"
    )),
    ("ablations", (
        "What ablation studies does this paper run? For each ablation, "
        "what design choice is being isolated and what does the paper "
        "show that component contributes? Cite the ablation table or "
        "section. If there are no ablations, say so explicitly."
    )),
)


class DeepPaperAnalysisInput(BaseModel):
    paper_id: str = Field(
        default="",
        description=(
            "UUID of the paper to deep-dive. Must be a concrete paper id "
            "from the paper ledger (or an arXiv id resolved via arxiv_import / "
            "deep_search). Never pass placeholder text — call a retrieval "
            "tool first if you don't have a real id."
        ),
    )
    paper_title: str = Field(
        default="",
        description=(
            "Optional title fallback when paper_id is unavailable. The "
            "tool resolves the paper by title-substring; pass the EXACT "
            "title from a retrieval result for best results."
        ),
    )
    aspects: list[str] = Field(
        default_factory=lambda: [a for a, _ in _ASPECTS],
        description=(
            "Which aspects to probe. Defaults to all four. Pass a "
            "subset like ``['methods', 'limitations']`` to scope the "
            "deep dive when only certain angles matter."
        ),
    )


class DeepPaperAnalysisOutput(BaseModel):
    paper_id: str
    paper_title: str
    aspects: dict       # {aspect_name: {answer, found, chunks_used, ...}}
    overall_found: bool


class DeepPaperAnalysisTool:
    """Run a four-aspect deep dive on a single paper."""

    name = "deep_paper_analysis"
    summary = (
        "Comprehensive multi-aspect deep-dive on a specific paper. Runs "
        "focused paper_qa rounds over methods, results, limitations, and "
        "ablations, returning a structured report. Use when the user "
        "explicitly asks to 'go deeper' on a paper, when critique flags "
        "abstract-only grounding, or when comparing two papers requires "
        "real technical detail (not just abstracts). Requires the paper "
        "to already be in the corpus — call arxiv_import / deep_search "
        "first if needed."
    )
    cost_class = "expensive"   # 4× paper_qa calls
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = DeepPaperAnalysisInput
    output_schema = DeepPaperAnalysisOutput

    async def run(self, ctx: ToolContext, params: DeepPaperAnalysisInput) -> ToolResult:
        from app.assistant.tools.paper_qa import PaperQATool, PaperQAInput

        # Reject placeholders up front — same rule paper_qa enforces,
        # surfaced here so the cost gate doesn't fire four times for
        # an unresolvable paper.
        id_in = (params.paper_id or "").strip()
        title_in = (params.paper_title or "").strip()
        if not id_in and not title_in:
            return ToolResult(
                output={
                    "paper_id": "",
                    "paper_title": "",
                    "aspects": {},
                    "overall_found": False,
                },
                summary=(
                    "deep_paper_analysis skipped — no paper_id or paper_title "
                    "provided. Run deep_search / arxiv_import first to get a "
                    "concrete id, then call deep_paper_analysis."
                ),
            )

        # Filter requested aspects to the ones we know how to probe.
        # An unknown aspect is silently dropped rather than rejected —
        # the planner sometimes invents new names; we degrade
        # gracefully.
        requested = {a.lower() for a in (params.aspects or [])}
        targets = [
            (name, q) for name, q in _ASPECTS
            if not requested or name in requested
        ]
        if not targets:
            # Default to the full set when filtering left nothing.
            targets = list(_ASPECTS)

        await ctx.emit_progress(
            5,
            f"Deep dive starting ({len(targets)} aspects)",
        )

        results: dict = {}
        any_found = False
        resolved_title = ""
        resolved_id = ""
        paper_qa = PaperQATool()
        for i, (aspect, question) in enumerate(targets):
            pct = 10 + int(80 * i / max(1, len(targets)))
            await ctx.emit_progress(pct, f"Deep dive · {aspect}")
            try:
                # Unbounded per-aspect time per user spec. The outer
                # ReAct loop / task cancellation is the only ceiling.
                aspect_result = await paper_qa.run(
                    ctx,
                    PaperQAInput(
                        paper_id=id_in,
                        paper_title=title_in,
                        question=question,
                    ),
                )
                out = aspect_result.output or {}
                results[aspect] = {
                    "answer": out.get("answer") or "",
                    "found": bool(out.get("found")),
                    "chunks_used": int(out.get("chunks_used") or 0),
                    "sections_used": out.get("sections_used") or [],
                    "chunk_positions": out.get("chunk_positions") or [],
                }
                if out.get("found"):
                    any_found = True
                    # Capture the resolved title/id once — paper_qa
                    # returns them on every call but we only need the
                    # first successful resolution.
                    if not resolved_id:
                        resolved_id = str(out.get("paper_id") or "")
                    if not resolved_title:
                        resolved_title = str(out.get("paper_title") or "")
            except asyncio.CancelledError:
                # Cancellation from the outer loop / user Stop button
                # must propagate immediately — the partial report
                # whatever was already collected can still be returned
                # by the caller, but we don't try to swallow the cancel
                # and continue.
                raise
            except Exception as exc:  # noqa: BLE001 — one aspect failing must not kill the deep dive
                log.warning("deep_paper_analysis: aspect %s failed: %s", aspect, exc)
                results[aspect] = {
                    "answer": "",
                    "found": False,
                    "chunks_used": 0,
                    "sections_used": [],
                    "chunk_positions": [],
                    "error": str(exc)[:200],
                }

        await ctx.emit_progress(100, "Deep dive complete")
        # Top-of-summary line. When nothing resolved, the summary tells
        # the planner to retrieve first — same recoverable hint pattern
        # as paper_qa.
        if not any_found:
            return ToolResult(
                output={
                    "paper_id": id_in,
                    "paper_title": title_in,
                    "aspects": results,
                    "overall_found": False,
                    "recoverable_hint": "retrieve_then_retry",
                },
                summary=(
                    f"deep_paper_analysis could not resolve paper "
                    f"'{title_in or id_in}'. Retrieve via deep_search / "
                    "arxiv_import and retry with the surfaced id."
                ),
            )
        completed_aspects = [a for a, r in results.items() if r.get("found")]
        return ToolResult(
            output={
                "paper_id": resolved_id or id_in,
                "paper_title": resolved_title or title_in,
                "aspects": results,
                "overall_found": True,
            },
            summary=(
                f"Deep dive on '{resolved_title[:60] or id_in}' — "
                f"{len(completed_aspects)}/{len(targets)} aspects covered "
                f"({', '.join(completed_aspects)})"
            ),
        )


deep_paper_analysis_tool = DeepPaperAnalysisTool()
