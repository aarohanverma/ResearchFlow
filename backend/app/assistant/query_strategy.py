"""Adaptive retrieval/synthesis strategy router.

The planner and the ReAct loop both used to treat every query the same
way: dispatch the standard tool catalog, let the LLM pick. That's a
static pipeline. For a query like ``"what is BERT"`` you don't want a
3-tool literature survey; for ``"compare RAG vs long-context for
production research workflows"`` a single deep_search call is the
wrong move.

This module is the per-query strategy hint. ``classify_query`` returns
a :class:`QueryStrategy` describing the query *shape* (factual lookup,
identifier, comparison, survey, exploratory, follow-up, etc.) along
with the recommended tool ordering, retrieval limits, and reranking
intensity.

The hint is *advisory*. Two places consume it:

* The planner ``_build_prompt`` injects the strategy as a "Strategy
  hint" block — the LLM can deviate when the conversation context
  argues for it.
* The ReAct loop's decision prompt mirrors the same hint so mid-turn
  decisions stay coherent with the planner's intent.

We never hard-route — that would freeze the system into the same
heuristic pipelines the user explicitly asked us to escape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Public types ─────────────────────────────────────────────────────────────


@dataclass
class QueryStrategy:
    """Strategy hint surfaced to planner + ReAct loop.

    Attributes:
        shape: Short label for the query class (used in prompts).
        preferred_tools: Ordered list of tool names worth trying first.
        avoid_tools: Tools that are usually a poor fit for this shape.
        retrieval_limit: Recommended top-N for retrieval calls. Larger
            for survey/comparison, smaller for factual lookups.
        rerank_intensity: ``"light"`` | ``"standard"`` | ``"heavy"``.
            Lookups need almost no rerank; comparisons / surveys
            benefit from heavy rerank because the long-tail matters.
        max_iterations: Recommended ReAct iteration cap override.
        rationale: One-line explanation for the human reader.
    """

    shape: str
    preferred_tools: list[str] = field(default_factory=list)
    avoid_tools: list[str] = field(default_factory=list)
    retrieval_limit: int = 8
    rerank_intensity: str = "standard"
    max_iterations: int = 6
    rationale: str = ""

    def render_for_prompt(self) -> str:
        bits = [f"shape={self.shape}"]
        if self.preferred_tools:
            bits.append("prefer=[" + ", ".join(self.preferred_tools[:6]) + "]")
        if self.avoid_tools:
            bits.append("avoid=[" + ", ".join(self.avoid_tools[:6]) + "]")
        bits.append(f"retrieval_limit={self.retrieval_limit}")
        bits.append(f"rerank={self.rerank_intensity}")
        bits.append(f"max_iters={self.max_iterations}")
        line = "; ".join(bits)
        return f"{line}\n  reason: {self.rationale[:240]}" if self.rationale else line


# ── Classifier ───────────────────────────────────────────────────────────────


_IDENTIFIER_RE = re.compile(
    r"\b("
    r"\d{4}\.\d{4,5}(?:v\d+)?"            # arxiv id (legacy)
    r"|arxiv:\s*\d{4}\.\d{4,5}"
    r"|10\.\d{4,9}/\S+"                   # DOI
    r"|[a-z]{2}\.[A-Z]{2,3}/\d+"          # old arxiv subject id
    r"|nips\d{4}|iclr\d{4}|neurips\d{4}|cvpr\d{4}|icml\d{4}|acl\d{4}"
    r")\b",
    re.IGNORECASE,
)

_DEFINITION_RE = re.compile(
    r"^\s*(what\s+is|what\s+are|define|definition\s+of|tldr|meaning\s+of|"
    r"who\s+is|who\s+was|when\s+was|how\s+many)\b",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b(compare|comparison|versus|\bvs\b|contrast|side[-\s]?by[-\s]?side|"
    r"differences?\s+between|trade[-\s]?offs?)\b",
    re.IGNORECASE,
)

_SURVEY_RE = re.compile(
    r"\b(survey|literature\s+review|state\s+of\s+the\s+art|sota|"
    r"overview\s+of|landscape\s+of|comprehensive\s+(?:review|analysis)|"
    r"what(?:'s|\s+is)\s+been\s+done)\b",
    re.IGNORECASE,
)

_RECENCY_RE = re.compile(
    r"\b(latest|recent|new(?:est)?|frontier|cutting[-\s]?edge|just\s+published|"
    r"this\s+(?:week|month|year)|past\s+(?:week|month|year))\b",
    re.IGNORECASE,
)

_EXPLAIN_RE = re.compile(
    r"\b(explain|how\s+does|how\s+do|why\s+does|why\s+do|walk\s+me\s+through|"
    r"intuition\s+behind|mechanism\s+of)\b",
    re.IGNORECASE,
)

_FOLLOWUP_HINTS = (
    "that", "those", "these", "it ", "they", "as before", "the same",
    "we discussed", "you mentioned", "earlier", "previously", "again",
    "proceed", "continue", "your suggestion",
)

_SYNTHESIS_RE = re.compile(
    r"\b(synthesi[sz]e|combine\s+(?:the\s+)?(?:above|these)\s+papers|"
    r"new\s+hypothesis|propose\s+(?:a\s+)?(?:novel|new)|brainstorm|"
    r"research\s+direction|run\s+genie)\b",
    re.IGNORECASE,
)


def classify_query(query: str, *, history: list[dict] | None = None) -> QueryStrategy:
    """Return the recommended strategy for ``query``.

    Pure-heuristic + cheap. The signals are deliberately conservative —
    when in doubt we return a sensible "exploratory" default rather
    than route into a narrow lane the user didn't ask for. Hard routing
    is exactly what the user told us to avoid.
    """
    q = (query or "").strip()
    if not q:
        return _exploratory(rationale="empty query")

    # 1. Identifier-shaped queries route to exact lookup first.
    if _IDENTIFIER_RE.search(q):
        return QueryStrategy(
            shape="identifier_lookup",
            preferred_tools=["arxiv_import", "paper_qa", "study_paper"],
            avoid_tools=["literature_survey", "frontier_scan"],
            retrieval_limit=3,
            rerank_intensity="light",
            max_iterations=3,
            rationale="query contains an arXiv id / DOI / canonical paper handle; resolve directly before any survey work",
        )

    # 2. Synthesis-style queries hit the Genie flow.
    if _SYNTHESIS_RE.search(q):
        return QueryStrategy(
            shape="synthesis",
            preferred_tools=["deep_search", "genie_synthesize", "genie_deep_dive"],
            avoid_tools=["genie_read"],
            retrieval_limit=10,
            rerank_intensity="heavy",
            max_iterations=8,
            rationale="user asked to synthesize / propose a novel direction; retrieve evidence then create a new capsule, do NOT just read stale capsules",
        )

    # 3. Comparison queries need both sides of the evidence — bump the
    #    retrieval limit, enable heavy rerank so the long tail matters,
    #    and prefer compare_papers downstream.
    if _COMPARISON_RE.search(q):
        return QueryStrategy(
            shape="comparison",
            preferred_tools=["deep_search", "literature_survey", "compare_papers"],
            avoid_tools=["wikipedia"],
            retrieval_limit=12,
            rerank_intensity="heavy",
            max_iterations=8,
            rationale="comparative question; retrieve a broad candidate set so both sides are represented before reranking",
        )

    # 4. Survey / SOTA queries.
    if _SURVEY_RE.search(q):
        return QueryStrategy(
            shape="survey",
            preferred_tools=["literature_survey", "deep_search", "research_trends"],
            avoid_tools=["wikipedia", "concept_explain"],
            retrieval_limit=16,
            rerank_intensity="heavy",
            max_iterations=8,
            rationale="structured survey requested; use literature_survey as the primary driver",
        )

    # 5. Recency-flavoured queries.
    if _RECENCY_RE.search(q):
        return QueryStrategy(
            shape="frontier",
            preferred_tools=["frontier_scan", "arxiv_import", "deep_search"],
            avoid_tools=["wikipedia"],
            retrieval_limit=10,
            rerank_intensity="standard",
            max_iterations=6,
            rationale="user wants recent / frontier work; prefer arXiv ingestion + frontier_scan",
        )

    # 6. Definitional / single-concept lookups.
    if _DEFINITION_RE.search(q) and len(q) < 120:
        return QueryStrategy(
            shape="definition",
            preferred_tools=["concept_explain", "wikipedia", "deep_search"],
            avoid_tools=["literature_survey", "compare_papers", "frontier_scan"],
            retrieval_limit=4,
            rerank_intensity="light",
            max_iterations=3,
            rationale="single-concept definition; one focused lookup + concept_explain is sufficient",
        )

    # 7. Explanation queries.
    if _EXPLAIN_RE.search(q) and len(q) < 160:
        return QueryStrategy(
            shape="explanation",
            preferred_tools=["concept_explain", "deep_search", "paper_qa"],
            avoid_tools=["frontier_scan"],
            retrieval_limit=6,
            rerank_intensity="standard",
            max_iterations=5,
            rationale="mechanistic / how-does explanation; ground in retrieval then explain",
        )

    # 8. Follow-up turn — borrow the prior strategy hint where possible
    #    and bias toward continuity instead of re-bootstrapping a survey.
    q_low = q.lower()
    if history and any(h.get("role") == "assistant" for h in history) and any(c in q_low for c in _FOLLOWUP_HINTS):
        return QueryStrategy(
            shape="followup",
            preferred_tools=["deep_search", "paper_qa", "compare_papers"],
            avoid_tools=["literature_survey"],
            retrieval_limit=6,
            rerank_intensity="standard",
            max_iterations=5,
            rationale="follow-up turn referencing prior context; resolve referent then retrieve narrowly",
        )

    # 9. Default — exploratory research question. Multi-tool retrieval
    #    with standard rerank, generous iteration budget.
    return _exploratory()


def _exploratory(rationale: str = "open-ended exploratory question; broad retrieval with adaptive reranking") -> QueryStrategy:
    return QueryStrategy(
        shape="exploratory",
        preferred_tools=["deep_search", "literature_survey", "concept_explain"],
        avoid_tools=[],
        retrieval_limit=8,
        rerank_intensity="standard",
        max_iterations=6,
        rationale=rationale,
    )
