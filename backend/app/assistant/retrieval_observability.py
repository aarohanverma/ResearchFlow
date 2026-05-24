"""Retrieval-quality observability for the ReAct loop.

Until now the loop only knew whether a tool *ran*; it had no read on
whether the retrieval was *good*. That meant:

* Thin retrievals (1-2 papers when 8 were asked for) silently slipped
  through and the synthesizer was none the wiser.
* When the reranker reordered the candidate set dramatically (a sign
  that the raw retrieval was poor and the reranker had to rescue it),
  there was no signal that this happened — only the post-rerank top-N
  reached the synth.
* Score dispersion was invisible: a retrieval where the top-1 result
  scored 0.92 and the top-8 scored 0.31 was indistinguishable from one
  where every result scored ~0.88. The first case warns that the
  evidence base is thin or off-topic; the second means it's actually
  on-topic.

This module computes three lightweight metrics per retrieval call:

  * ``coverage_ratio``     — returned / asked-for (caps at 1.0). When
    a tool returns 0-1 papers for a `limit=8` request, this drops
    sharply and the loop knows the corpus didn't satisfy the query.
  * ``score_dispersion``   — stdev of ``search_score`` across the
    returned papers, normalized to the mean (CV). High CV means the
    top result is much better than the tail, which usually means
    a thin but-on-topic match. Low CV with a low mean means everything
    is uniformly mediocre.
  * ``rerank_disagreement`` — Kendall-style rank-distance between the
    raw retrieval order (sorted by ``raw_score`` when present) and the
    final reranker order. High values mean the reranker did most of
    the work — a flag that the raw retrieval is weak.

We do NOT call any LLM here. Everything is computed from the
``ToolResult.output`` dict in O(papers) time. The result is rolled
into the per-turn ``RetrievalObservability`` dataclass and rendered
into both the next decision prompt (so the model can react to a thin
retrieval) and the synthesizer's ``<agent_notes>`` (so the answer
honestly caveats when the evidence base was weak).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.assistant.tools.base import ToolResult


_RETRIEVAL_TOOLS: frozenset[str] = frozenset({
    "deep_search", "arxiv_search", "arxiv_import", "frontier_scan",
    "literature_survey", "citation_finder",
    "pubmed", "inspire_hep", "nasa_ads", "semantic_scholar",
    "huggingface_search", "github_search", "papers_with_code",
})


@dataclass
class RetrievalSnapshot:
    """Per-call quality metrics for one retrieval-class tool invocation."""

    tool: str
    returned: int
    asked: int
    coverage_ratio: float
    score_dispersion: float | None   # None when no scores available
    rerank_disagreement: float | None  # None when no raw_score baseline
    top_score: float | None
    mean_score: float | None
    note: str = ""

    def render(self) -> str:
        bits = [f"{self.tool}: {self.returned}/{self.asked} papers"]
        if self.coverage_ratio < 0.5:
            bits.append("LOW coverage")
        if self.score_dispersion is not None:
            bits.append(f"dispersion={self.score_dispersion:.2f}")
        if self.rerank_disagreement is not None and self.rerank_disagreement > 0.4:
            bits.append(f"rerank-disagreement={self.rerank_disagreement:.2f}")
        if self.top_score is not None:
            bits.append(f"top={self.top_score:.2f}")
        if self.note:
            bits.append(f"({self.note})")
        return "; ".join(bits)


@dataclass
class RetrievalObservability:
    """Running collection of every retrieval snapshot this turn."""

    snapshots: list[RetrievalSnapshot] = field(default_factory=list)

    def record(self, tool: str, params: dict[str, Any] | None, result: ToolResult) -> RetrievalSnapshot | None:
        """Compute + append a snapshot for ``result``. Returns the snapshot
        (or ``None`` if the tool isn't retrieval-class)."""
        if tool not in _RETRIEVAL_TOOLS:
            return None
        snap = _compute_snapshot(tool, params or {}, result)
        self.snapshots.append(snap)
        return snap

    # ── views ──────────────────────────────────────────────────────────

    def thin_retrieval_count(self, *, coverage_threshold: float = 0.4) -> int:
        return sum(1 for s in self.snapshots if s.coverage_ratio < coverage_threshold)

    def high_rerank_disagreement_count(self, *, threshold: float = 0.45) -> int:
        return sum(
            1 for s in self.snapshots
            if s.rerank_disagreement is not None and s.rerank_disagreement > threshold
        )

    def thin_evidence_signal(self) -> bool:
        """True when the aggregate retrieval picture is poor.

        Two failure modes worth surfacing:
          - More than one thin retrieval this turn, OR
          - A single retrieval where coverage AND top_score are both low.
        """
        if self.thin_retrieval_count() >= 2:
            return True
        for s in self.snapshots:
            if s.coverage_ratio < 0.35 and (s.top_score or 1.0) < 0.4:
                return True
        return False

    def render_for_prompt(self, limit: int = 6) -> str:
        if not self.snapshots:
            return "(no retrievals yet)"
        lines = [f"  - {s.render()}" for s in self.snapshots[-limit:]]
        if len(self.snapshots) > limit:
            lines.append(f"  ... and {len(self.snapshots) - limit} earlier")
        return "\n".join(lines)

    def to_agent_notes(self) -> dict[str, Any]:
        """Serialise into ``agent_notes['retrieval']`` for the synthesizer.

        Returns ``{}`` when nothing useful to surface; otherwise a small
        dict the synthesizer renders into ``<agent_notes>``.
        """
        if not self.snapshots:
            return {}
        return {
            "calls": len(self.snapshots),
            "thin_calls": self.thin_retrieval_count(),
            "rerank_heavy_calls": self.high_rerank_disagreement_count(),
            "weakest": min(self.snapshots, key=lambda s: s.coverage_ratio).render(),
            "thin_evidence": self.thin_evidence_signal(),
        }


# ── Internals ────────────────────────────────────────────────────────────────


def _compute_snapshot(tool: str, params: dict, result: ToolResult) -> RetrievalSnapshot:
    out = result.output or {}
    papers = _papers_from_output(out)
    returned = len(papers)
    asked = int(
        params.get("limit")
        or params.get("max_results")
        or params.get("arxiv_max_results")
        or 8
    )
    coverage_ratio = min(1.0, returned / max(1, asked))

    # Score dispersion: coefficient of variation across search_score
    # (falls back to ``score`` / ``relevance_score``). Returns None
    # when no numeric scores are present.
    scores = _scores(papers)
    if scores:
        mean = sum(scores) / len(scores)
        if mean > 0 and len(scores) > 1:
            variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
            dispersion = math.sqrt(variance) / mean
        else:
            dispersion = 0.0
        top_score = max(scores)
        mean_score: float | None = mean
    else:
        dispersion = None
        top_score = None
        mean_score = None

    # Rerank disagreement: only computed when both ``raw_score`` (pre-
    # rerank baseline) and ``search_score`` (post-rerank) are present
    # for the SAME papers. Previously ``_scores`` and ``_raw_scores`` were
    # extracted independently — when some papers had one score but not the
    # other, the returned lists were positionally misaligned and the
    # rank-distance computation compared paper A's raw rank against paper
    # B's rerank rank. Use the paired extractor so the ranks always refer
    # to the same paper at the same index.
    paired = _paired_scores(papers)
    if len(paired) >= 3:
        paired_raw = [r for r, _ in paired]
        paired_rerank = [s for _, s in paired]
        rerank_disagreement = _normalized_rank_distance(paired_raw, paired_rerank)
    else:
        rerank_disagreement = None

    note = ""
    if returned == 0:
        note = "empty"
    elif returned == asked:
        note = "saturated"
    elif coverage_ratio < 0.4:
        note = "thin"

    return RetrievalSnapshot(
        tool=tool,
        returned=returned,
        asked=asked,
        coverage_ratio=coverage_ratio,
        score_dispersion=dispersion,
        rerank_disagreement=rerank_disagreement,
        top_score=top_score,
        mean_score=mean_score,
        note=note,
    )


def _papers_from_output(out: dict) -> list[dict]:
    for key in ("papers", "results", "items", "candidates"):
        v = out.get(key)
        if isinstance(v, list):
            return [c for c in v if isinstance(c, dict)]
    return []


def _scores(papers: list[dict]) -> list[float]:
    vals: list[float] = []
    for p in papers:
        for k in ("search_score", "score", "relevance_score", "rerank_score"):
            v = p.get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
                break
    return vals


def _raw_scores(papers: list[dict]) -> list[float]:
    vals: list[float] = []
    for p in papers:
        v = p.get("raw_score") or p.get("retrieval_score") or p.get("rrf_score")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    # Only treat as a valid baseline if we have a score for most papers.
    if len(vals) < max(2, len(papers) // 2):
        return []
    return vals


def _paired_scores(papers: list[dict]) -> list[tuple[float, float]]:
    """Per-paper (raw_score, rerank_score) pairs — only when BOTH are set.

    The rank-distance metric needs positionally-aligned scores; extracting
    each side independently with :func:`_scores` and :func:`_raw_scores`
    produces lists that can correspond to different paper subsets when
    some papers are missing one score or the other. Walking papers once
    and only keeping the pairs where both scores are present guarantees
    that ``paired[i]`` refers to the same paper on both sides.
    """
    out: list[tuple[float, float]] = []
    for p in papers:
        raw_v = p.get("raw_score") or p.get("retrieval_score") or p.get("rrf_score")
        if not isinstance(raw_v, (int, float)):
            continue
        rerank_v: float | None = None
        for k in ("search_score", "score", "relevance_score", "rerank_score"):
            v = p.get(k)
            if isinstance(v, (int, float)):
                rerank_v = float(v)
                break
        if rerank_v is None:
            continue
        out.append((float(raw_v), rerank_v))
    return out


def _normalized_rank_distance(raw_scores: list[float], rerank_scores: list[float]) -> float:
    """Spearman footrule normalised to 0..1. Returns the average
    absolute rank shift between the two orderings, divided by the
    maximum possible shift for the list length.

    0.0 = identical orders (reranker preserved retrieval)
    ~0.5 = substantial reordering (reranker did real work)
    ~1.0 = reversed order (reranker had to fight the retrieval)
    """
    n = min(len(raw_scores), len(rerank_scores))
    if n < 2:
        return 0.0
    raw_order = sorted(range(n), key=lambda i: -raw_scores[i])
    rerank_order = sorted(range(n), key=lambda i: -rerank_scores[i])
    raw_rank = {idx: rank for rank, idx in enumerate(raw_order)}
    rerank_rank = {idx: rank for rank, idx in enumerate(rerank_order)}
    shift = sum(abs(raw_rank[i] - rerank_rank[i]) for i in range(n))
    max_shift = (n * n) // 2 if n % 2 == 0 else (n * n - 1) // 2
    return shift / max(1, max_shift)
