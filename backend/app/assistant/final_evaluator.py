"""Final-answer evaluator — strong-model audit of the synthesized reply.

Where the synthesizer produces the answer, this module ASSESSES it:
is it relevant to the user's actual query, grounded in the cited
sources, complete enough to satisfy the ask, and free of drift?

The user spec is explicit: "Add a final evaluator to RA to check for
contextual relevancy, groundedness and that user query is answered
in satisfactory manner. It should suggest improvements, additions if
anything is missing and should also point out if the output is very
drifted and irrelevant in any manner. RA should then attempt to fix
all the shortcomings."

Design:

  * **Strong model.** Uses the reasoning_model so the audit is at
    least as careful as the synthesizer that produced the answer.
    Cheap-model evaluation produces shallow critiques that miss the
    real failure modes.
  * **Structured output.** Returns four scores in [0, 1]
    (relevance, groundedness, completeness, focus) plus a list of
    actionable improvement notes the synthesizer can act on. Scores
    below a soft threshold trigger ONE re-synth attempt with the
    notes spliced into the prompt.
  * **Bounded.** Runs at most ONCE per turn. The re-synth uses the
    same model + context but with the evaluator's suggestions
    appended. If the second pass still fails the same checks, we
    surface the answer plus a transparency footer rather than loop.
  * **Best-effort.** A provider error / malformed output collapses
    to "evaluator unavailable" and the original answer ships
    unchanged. The eval is advisory, not load-bearing — it should
    NEVER prevent shipping a reasonable answer the user is waiting
    for.

The evaluator is independent of the existing
``_detect_output_quality_issue`` post-pass (which catches mechanical
corruption — empty, truncated, placeholder leaks) and the provenance
verifier (which checks per-marker citation faithfulness). This new
layer covers SEMANTIC adequacy: did we ANSWER what was asked?
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# Score thresholds below which we trigger a re-synth. Calibrated so
# the evaluator only intervenes on clearly subpar answers — fixing a
# 0.85 answer is rarely worth the latency hit. The "drift" check is
# stricter because off-topic drift wastes the entire turn.
_RELEVANCE_FLOOR = 0.65
_GROUNDEDNESS_FLOOR = 0.55
_COMPLETENESS_FLOOR = 0.55
_FOCUS_FLOOR = 0.70   # drift detector — higher floor

# Cap on improvement-notes string length so a runaway evaluator can't
# bloat the re-synth prompt.
_MAX_NOTES_CHARS = 1200


_EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevance":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "groundedness":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "completeness":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "focus":         {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "drifted":       {"type": "boolean"},
        "answers_query": {"type": "boolean"},
        "improvements": {
            "type": "array",
            "items": {"type": "string"},
        },
        "verdict": {
            "type": "string",
            "enum": ["ship", "revise", "drift"],
        },
    },
    "required": ["relevance", "groundedness", "completeness", "focus", "verdict"],
}


_EVAL_SYSTEM = (
    "You are the final evaluator for a Research Assistant. Your job: "
    "audit an answer that's about to ship to the user and decide "
    "whether it's good enough or needs one revision pass.\n\n"
    "Score four dimensions, each in [0.0, 1.0]:\n\n"
    "  • relevance — does the answer address the user's ACTUAL question? "
    "An answer that's well-written but answers a slightly different "
    "question gets a LOW relevance score.\n"
    "  • groundedness — are the load-bearing claims supported by the "
    "papers cited in the answer? Are the citation markers attached to "
    "claims those papers actually make?\n"
    "  • completeness — does the answer cover the major angles the user "
    "would expect, OR honestly acknowledge what it doesn't cover?\n"
    "  • focus — does the answer stay on the user's main topic, or drift "
    "into adjacent subfields that don't change the conclusion?\n\n"
    "Then set:\n"
    "  • drifted = true when focus < 0.5 OR the answer's main thread is "
    "noticeably off the user's question.\n"
    "  • answers_query = true when the user could reasonably say "
    "'yes, this answered what I asked'.\n"
    "  • verdict = 'ship' (good enough — minor or no issues), "
    "'revise' (clear gap or weakness, ONE more synthesis pass would "
    "materially improve it), 'drift' (the answer is off-topic in a way "
    "that requires re-planning, not just re-synthesis).\n\n"
    "improvements — array of CONCRETE, actionable notes for the next "
    "synthesis pass. Example: 'cite the specific paper for the 92% "
    "accuracy claim', 'add a brief comparison to method X which the "
    "user named in the query', 'remove the digression about adjacent "
    "subfield Y'. Empty list when verdict is 'ship'.\n\n"
    "Be honest, not generous. A polished answer that misses the user's "
    "real question is still a bad answer.\n\n"
    "Return STRICT JSON matching the schema. No prose."
)


async def evaluate_final_answer(
    *,
    query: str,
    answer: str,
    papers: list[dict],
    arxiv_results: list[dict],
) -> dict[str, Any] | None:
    """Run one evaluation pass against the synthesized answer.

    Returns the evaluator's dict on success or ``None`` when the
    evaluator was unavailable (no LLM, malformed output, etc.). The
    caller treats ``None`` as "no signal — ship the original
    answer".

    Scoring scale:
        relevance, groundedness, completeness, focus ∈ [0.0, 1.0].
        verdict ∈ {ship, revise, drift}.

    Side effects:
        None. The function only reads the LLM adapter.

    Safety:
        Wrapped in try/except — never raises. The evaluator is
        advisory: a failure here must not prevent shipping the
        answer the user is waiting for.
    """
    if not query or not answer:
        return None
    try:
        from app.adapters.llm import get_llm_adapter
        llm = get_llm_adapter()
    except Exception as exc:  # noqa: BLE001 — evaluator must never crash the turn
        log.debug("final_evaluator: llm adapter unavailable: %s", exc)
        return None

    # Compact paper inventory for the evaluator — title + tldr is
    # enough to judge whether claims are grounded. The full answer is
    # always passed verbatim.
    paper_lines: list[str] = []
    for i, p in enumerate(papers or [], start=1):
        title = (p.get("title") or "")[:140]
        tldr = (p.get("tldr") or p.get("abstract") or "")[:280]
        if title:
            paper_lines.append(f"[{i}] {title}\n    {tldr}")
    for i, p in enumerate(arxiv_results or [], start=1):
        title = (p.get("title") or "")[:140]
        tldr = (p.get("tldr") or p.get("abstract") or "")[:280]
        if title:
            paper_lines.append(f"[A{i}] {title}\n    {tldr}")
    papers_block = "\n".join(paper_lines) if paper_lines else "(none surfaced)"

    user_msg = (
        f"USER QUERY:\n{query[:2000]}\n\n"
        f"PAPERS CITED IN THE ANSWER:\n{papers_block}\n\n"
        f"DRAFT ANSWER:\n{answer[:8000]}\n\n"
        "Audit the draft answer. Return strict JSON."
    )

    try:
        raw = await llm.complete_structured(
            [
                {"role": "system", "content": _EVAL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            llm.reasoning_model,
            _EVAL_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("final_evaluator: LLM call failed: %s", exc)
        return None
    if not isinstance(raw, dict):
        return None

    # Normalise / clamp
    def _clamp01(v: Any, default: float = 0.5) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, f))

    return {
        "relevance": _clamp01(raw.get("relevance")),
        "groundedness": _clamp01(raw.get("groundedness")),
        "completeness": _clamp01(raw.get("completeness")),
        "focus": _clamp01(raw.get("focus")),
        "drifted": bool(raw.get("drifted")),
        "answers_query": bool(raw.get("answers_query", True)),
        "verdict": str(raw.get("verdict") or "ship"),
        "improvements": [
            str(s)[:400]
            for s in (raw.get("improvements") or [])
            if isinstance(s, str) and s.strip()
        ][:8],
    }


def should_revise(report: dict[str, Any]) -> bool:
    """Apply the soft thresholds to decide if a re-synth is worth it.

    Returns True when ANY score is below its floor OR the verdict is
    explicitly ``revise``/``drift``. We deliberately don't auto-revise
    on score alone — the verdict carries the evaluator's holistic
    judgment.
    """
    if not isinstance(report, dict):
        return False
    if str(report.get("verdict") or "").lower() in {"revise", "drift"}:
        return True
    if report.get("relevance", 1.0) < _RELEVANCE_FLOOR:
        return True
    if report.get("groundedness", 1.0) < _GROUNDEDNESS_FLOOR:
        return True
    if report.get("completeness", 1.0) < _COMPLETENESS_FLOOR:
        return True
    if report.get("focus", 1.0) < _FOCUS_FLOOR:
        return True
    return False


def revision_notes(report: dict[str, Any]) -> str:
    """Format the evaluator's improvement notes into a prompt-ready
    block the synthesizer can splice in. Returns the empty string
    when there's nothing actionable.
    """
    if not isinstance(report, dict):
        return ""
    parts: list[str] = []
    verdict = str(report.get("verdict") or "")
    if verdict:
        parts.append(f"verdict: {verdict}")
    scores = (
        f"relevance={report.get('relevance', '?')}, "
        f"groundedness={report.get('groundedness', '?')}, "
        f"completeness={report.get('completeness', '?')}, "
        f"focus={report.get('focus', '?')}"
    )
    parts.append(scores)
    improvements = report.get("improvements") or []
    if improvements:
        parts.append("improvements:")
        for s in improvements:
            parts.append(f"  - {s}")
    blob = "\n".join(parts)
    return blob[:_MAX_NOTES_CHARS]


__all__ = [
    "evaluate_final_answer",
    "revision_notes",
    "should_revise",
]
