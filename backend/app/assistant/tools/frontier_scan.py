"""Frontier scan tool — composes existing ranking signals for discovery.

Builds a ranked list of papers optimized for "what's emerging" rather than
"what's most relevant". Combines novelty score, recency decay, breakthrough
flag, and personal-interest overlap into a single ``frontier_score`` and
attaches a ``why_surfaced`` breakdown so the UI can render explainable badges.

This tool wraps existing data — it does NOT call the LLM or external APIs,
making it cheap to run repeatedly (unlike ``deep_search`` or ``arxiv_import``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult
from app.models.paper import Paper
from app.models.user import UserInterestProfile

log = logging.getLogger(__name__)


class FrontierScanInput(BaseModel):
    namespace_keys: list[str] = Field(default_factory=list)
    limit: int = Field(default=8, ge=1, le=30)
    min_novelty: float = Field(default=0.7, ge=0.0, le=1.0)
    days_recent: int = Field(default=30, ge=1, le=365)


class FrontierScanOutput(BaseModel):
    papers: list[dict]
    total: int
    widened: bool = False


class FrontierScanTool:
    """Cheap ranking pass that surfaces emerging/high-novelty papers."""

    name = "frontier_scan"
    summary = (
        "Ranking pass over already-indexed papers using existing signals "
        "(novelty score, recency decay, breakthrough flag, personal interest "
        "overlap). Each paper carries a why_surfaced breakdown so the UI can "
        "render explainable badges.\n\n"
        "USE WHEN: the user asks 'what's new', 'frontier', 'recent "
        "breakthroughs', or 'what should I look at this week' — they want a "
        "novelty-ordered scan of what's ALREADY in the corpus.\n"
        "DO NOT USE WHEN: the user asks a specific topic-grounded question "
        "(use deep_search), wants new arXiv papers not yet indexed "
        "(use arxiv_search or arxiv_import), or wants a structured field "
        "overview (use literature_survey).\n\n"
        "Default scope is cross-namespace; pass namespace_keys only when the "
        "user explicitly scopes. Cheap: no LLM calls, only DB + ranking."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = False
    streamable = False
    input_schema = FrontierScanInput
    output_schema = FrontierScanOutput

    async def run(self, ctx: ToolContext, params: FrontierScanInput) -> ToolResult:
        # Cross-namespace by default — frontier work is interdisciplinary by
        # nature, and scoping to the user's active topic alone routinely
        # returns 0 candidates for legitimate questions.
        ns_keys = params.namespace_keys or None
        scope_label = ", ".join(ns_keys) if ns_keys else "all subscribed namespaces"
        await ctx.emit_progress(20, f"Scanning frontier across {scope_label}")

        rows = await self._candidates(ctx, ns_keys, params.min_novelty)
        widened = False
        # Two-step relaxation when the narrow filter starves the result set:
        # 1. Drop the namespace filter.
        # 2. Lower the novelty floor so a sparser corpus still surfaces work.
        if not rows and ns_keys:
            await ctx.emit_progress(50, "No matches in scope — widening to all namespaces")
            rows = await self._candidates(ctx, None, params.min_novelty)
            widened = bool(rows)
        if not rows:
            await ctx.emit_progress(70, "Relaxing novelty floor for a sparser corpus")
            relaxed = max(0.4, params.min_novelty - 0.2)
            rows = await self._candidates(ctx, None, relaxed)
            widened = widened or bool(rows)

        # Soft-bias by user interest concepts when available. The schema
        # stores ``concept_affinity`` as ``{concept_label: weight}`` — we
        # treat anything with weight >= 0.5 as a "hot" concept for the
        # purpose of frontier ranking.
        interest = await ctx.db.execute(
            select(UserInterestProfile).where(UserInterestProfile.user_id == ctx.user_id)
        )
        profile = interest.scalar_one_or_none()
        affinity = (profile.concept_affinity or {}) if profile else {}
        hot = {str(k) for k, v in affinity.items() if isinstance(v, (int, float)) and v >= 0.5}

        now = datetime.now(timezone.utc)
        # Methodological-diversity factor — for each candidate, measure how
        # different its methods_used set is from the rest of the candidate
        # pool. Pure set algebra over already-extracted enrichment fields,
        # so this stays cheap (no embedding fetch, no LLM call).
        diversity_by_id = _diversity_factors(rows)

        scored: list[tuple[float, dict]] = []
        for p in rows:
            row, score = _score_paper(
                p,
                hot=hot,
                now=now,
                days_recent=params.days_recent,
                method_diversity=diversity_by_id.get(p.id, 0.0),
            )
            scored.append((score, row))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = [r for _, r in scored[: params.limit]]
        suffix = " (widened search)" if widened else ""
        await ctx.emit_progress(100, f"Surfaced {len(top)} frontier candidates{suffix}")
        return ToolResult(
            output={"papers": top, "total": len(top), "widened": widened},
            summary=f"Frontier scan surfaced {len(top)} emerging papers{suffix}",
        )

    @staticmethod
    async def _candidates(ctx: ToolContext, ns_keys: list[str] | None, min_novelty: float):
        """Pull a generous candidate set with the given filters; in-process re-rank."""
        stmt = select(Paper).where(Paper.novelty_score >= min_novelty)
        if ns_keys:
            stmt = stmt.where(Paper.namespace_key.in_(ns_keys))
        stmt = stmt.order_by(Paper.published_at.desc().nulls_last()).limit(120)
        result = await ctx.db.execute(stmt)
        return list(result.scalars())


def _score_paper(
    p: Paper,
    *,
    hot: set[str],
    now: datetime,
    days_recent: int,
    method_diversity: float = 0.0,
) -> tuple[dict, float]:
    """Return ``(public_row, frontier_score)`` for ``p``.

    Score components — composed weights, all normalized to [0, 1]:
        novelty           — paper.novelty_score                       (0.35)
        recency           — linear decay over ``days_recent``         (0.25)
        breakthrough      — +0.20 if the breakthrough flag is set
        personal_fit      — concept overlap with hot interests        (0.15)
        method_diversity  — methodological-novelty among candidates   (0.15)
    """
    novelty = float(p.novelty_score or 0.0)
    recency = _recency_factor(p.published_at, now, days_recent)
    breakthrough = 0.2 if getattr(p, "is_breakthrough", False) else 0.0
    overlap = _concept_overlap(p.key_concepts or [], hot)
    diversity = max(0.0, min(1.0, float(method_diversity)))

    score = (
        (novelty * 0.35)
        + (recency * 0.25)
        + breakthrough
        + (overlap * 0.15)
        + (diversity * 0.15)
    )

    why: list[dict[str, Any]] = []
    if novelty >= 0.78:
        why.append({"signal": "novelty", "label": "high novelty", "weight": round(novelty * 0.35, 2)})
    if recency >= 0.5:
        why.append({"signal": "recency", "label": "recent", "weight": round(recency * 0.25, 2)})
    if breakthrough:
        why.append({"signal": "breakthrough", "label": "breakthrough", "weight": breakthrough})
    if overlap > 0:
        why.append({"signal": "personal_fit", "label": "matches your interests", "weight": round(overlap * 0.15, 2)})
    if diversity >= 0.5:
        why.append({"signal": "method_diversity", "label": "methodologically distinct",
                    "weight": round(diversity * 0.15, 2)})

    row = {
        "paper_id": str(p.id),
        "title": p.title,
        "abstract": p.abstract or "",
        "authors": list(p.authors or []),
        "namespace_key": p.namespace_key,
        "source_url": p.source_url,
        "pdf_url": p.pdf_url,
        "published_at": p.published_at.isoformat() if p.published_at else None,
        "tldr": p.tldr,
        "key_concepts": list(p.key_concepts or []),
        "methods_used": list(p.methods_used or []),
        "novelty_score": novelty,
        "relevance_score": float(p.relevance_score or 0.0),
        "search_score": round(score, 4),
        "match_type": "frontier",
        "why_surfaced": why,
    }
    return row, score


def _recency_factor(published_at: datetime | None, now: datetime, days_recent: int) -> float:
    """Linear decay over ``days_recent``; 1.0 today, 0.0 at ``days_recent`` days old."""
    if not published_at:
        return 0.0
    pub = published_at if published_at.tzinfo else published_at.replace(tzinfo=timezone.utc)
    age_days = (now - pub).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return max(0.0, 1.0 - (age_days / float(days_recent)))


def _concept_overlap(concepts: list[str], hot: set[str]) -> float:
    """Jaccard-style overlap clamped to [0, 1]."""
    if not concepts or not hot:
        return 0.0
    matched = sum(1 for c in concepts if c in hot)
    return min(1.0, matched / max(1, len(concepts)))


def _diversity_factors(papers: list[Paper]) -> dict:
    """Per-paper methodological-distinctness in [0, 1] vs the candidate pool.

    For each paper, computes the mean Jaccard *distance* between its
    ``methods_used`` set and the methods_used of every other candidate.
    Papers using methods rarely seen in the pool score higher; papers
    using the same method everyone else uses score lower. Returns an
    empty dict if no paper has methods_used (gracefully degrades).
    """
    by_id: dict = {}
    methods: list[tuple[object, set[str]]] = [
        (p.id, {m.lower().strip() for m in (p.methods_used or []) if m})
        for p in papers
    ]
    if not methods:
        return by_id
    informative = [m for m in methods if m[1]]
    if len(informative) < 2:
        # Without at least 2 informative papers we have no signal; assign
        # a neutral 0.5 rather than 0 so this factor doesn't unfairly
        # punish single-method papers.
        return {pid: 0.5 for pid, _ in methods}

    for pid, this_set in methods:
        if not this_set:
            by_id[pid] = 0.5
            continue
        distances: list[float] = []
        for other_pid, other_set in informative:
            if other_pid == pid or not other_set:
                continue
            inter = len(this_set & other_set)
            union = len(this_set | other_set) or 1
            distances.append(1.0 - (inter / union))
        by_id[pid] = (sum(distances) / len(distances)) if distances else 0.5
    return by_id
