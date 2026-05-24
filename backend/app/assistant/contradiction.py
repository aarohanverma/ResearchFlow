"""Active contradiction detection across ReAct observations.

The ReAct loop's only protection against confirmation bias used to be a
prose nudge in the decision prompt asking the model to "look for
counter-evidence." That's not a mechanism — the model can read the
guidance and still polish past a one-sided evidence base.

This module is the mechanism:

* :func:`detect_contradictions_in_results` scans the per-tool ``ToolResult``
  outputs that have landed this turn (initial plan + ReAct iterations)
  and looks for explicit disagreement markers in both *prose*
  (summaries, abstracts, comparison cells, survey text) and *numbers*
  (mismatched scalar claims about the same metric across papers).

* :class:`ContradictionSignal` is the structured row the loop appends
  to its scratchpad and renders into the next decision prompt so the
  model is told **which claim** is contested **and where** before it
  picks the next ACTION.

* :func:`should_force_counter_search` returns ``True`` when at least one
  un-investigated contradiction is on record AND the model is trying to
  finalize without having addressed it. The loop intercepts that case
  and forces a targeted ``citation_finder`` / retrieval call.

The detector is deliberately *high precision over high recall*: a false
positive forces a wasted iteration and a worse user experience, while a
false negative just degrades to the old prose-nudge behaviour. We only
flag when the contradiction signal is unambiguous — explicit lexical
markers, or a numeric disagreement larger than a configurable epsilon
on the same key — never on speculative semantic inference.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.assistant.tools.base import ToolResult

log = logging.getLogger(__name__)


# Lexical signals that one source is disagreeing with another. Kept
# conservative — generic words like "however" or "but" alone would
# produce too many false positives, so we require a sharper marker.
_CONTRADICTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcontradict(?:s|ed|ion|ory)\b", re.IGNORECASE),
    re.compile(r"\bfail(?:ed|s)?\s+to\s+replicate\b", re.IGNORECASE),
    re.compile(r"\bcould\s+not\s+(?:replicate|reproduce|confirm)\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+(?:replicate|reproduce|hold|generalize)\b", re.IGNORECASE),
    re.compile(r"\bin\s+contrast\s+to\b", re.IGNORECASE),
    re.compile(r"\binconsistent\s+with\b", re.IGNORECASE),
    re.compile(r"\bopposite\s+conclusion\b", re.IGNORECASE),
    re.compile(r"\bdisagree(?:s|d|ment)?\s+with\b", re.IGNORECASE),
    re.compile(r"\bovers?(?:tate|estimate|claim)(?:s|d)?\b", re.IGNORECASE),
    re.compile(r"\bunderperforms?\b", re.IGNORECASE),
    re.compile(r"\bweaker\s+than\s+(?:the\s+)?(?:reported|claimed|baseline)\b", re.IGNORECASE),
    re.compile(r"\brefut(?:e|es|ed|ation)\b", re.IGNORECASE),
    re.compile(r"\bnegative\s+result(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bbut\s+(?:other\s+work|prior\s+work|recent\s+work)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+supported\s+by\b", re.IGNORECASE),
]

# Numeric disagreement: same metric mentioned with very different
# scalar values across two papers. Currently we extract simple
# ``key: number`` patterns; richer parsing (e.g. units, ranges) is left
# for a follow-up — this catches the obvious cases like ``accuracy 92%``
# vs ``accuracy 71%`` for the same task.
_METRIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?P<key>accuracy|f1|recall|precision|exact[-_ ]match|bleu|rouge|"
        r"perplexity|loss|win[-_ ]?rate|score|throughput|latency|cost)\b"
        r"[\s:=]+(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>%|ms|s|x|×)?",
        re.IGNORECASE,
    ),
]


@dataclass
class ContradictionSignal:
    """One contested-claim row the loop surfaces to the decision prompt.

    Two flavours:
      * ``kind='lexical'`` — an explicit "X contradicts Y" phrase was
        found in one source's text; ``span`` is the offending sentence.
      * ``kind='numeric'`` — two sources reported very different scalar
        values for the same metric; ``span`` describes the disagreement.

    ``addressed`` is set to True once the loop runs at least one
    counter-search action whose tool target mentions a substring of the
    contradiction span — a coarse but workable "did we look?" gate that
    keeps the loop from forcing the same counter-search every iteration.

    ``confidence`` is a 0..1 score driving the *adaptive* counter-search
    policy. Soft signals like a single "however" stay below 0.6 and just
    surface in the prompt; sharp signals like an explicit "fails to
    replicate" or a numeric gap that's 2× the metric epsilon clear 0.6
    and trigger a forced counter-search if the model still tries to
    finalize. This is the difference between *nudging* the model and
    *blocking* it — we only block when we're confident the gap is real.
    """

    kind: str            # 'lexical' | 'numeric'
    span: str
    sources: list[str]   # ToolResult keys that surfaced the disagreement
    addressed: bool = False
    iteration: int = 0   # ReAct iteration the signal was first seen on
    confidence: float = 0.5

    def render(self, max_chars: int = 240) -> str:
        flag = "✓ counter-searched" if self.addressed else "✗ NOT YET addressed"
        spread = ",".join(self.sources[:3]) + ("..." if len(self.sources) > 3 else "")
        return (
            f"[{self.kind}|conf={self.confidence:.2f}] "
            f"{self.span[:max_chars]} (sources={spread}) — {flag}"
        )


# Lexical patterns rated by signal strength. Hits in the HIGH set
# clear the auto-force threshold; MED hits surface in the prompt but
# don't block finalize on their own.
_HIGH_CONFIDENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcontradict(?:s|ed|ion)\b", re.IGNORECASE),
    re.compile(r"\bfail(?:ed|s)?\s+to\s+replicate\b", re.IGNORECASE),
    re.compile(r"\bcould\s+not\s+(?:replicate|reproduce)\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+(?:replicate|reproduce|hold)\b", re.IGNORECASE),
    re.compile(r"\brefut(?:e|es|ed|ation)\b", re.IGNORECASE),
    re.compile(r"\bopposite\s+conclusion\b", re.IGNORECASE),
    re.compile(r"\bnegative\s+result(?:s)?\b", re.IGNORECASE),
]


def _lexical_confidence(span: str) -> float:
    """Score the lexical contradiction strength for ``span``.

    HIGH_CONFIDENCE markers score 0.8; everything else 0.45 — soft enough
    that the loop surfaces it but doesn't auto-force a counter-search
    when the model would rather finalize. Two markers anywhere in the
    same span clamp to 0.95.
    """
    hits = sum(1 for p in _HIGH_CONFIDENCE_PATTERNS if p.search(span))
    if hits >= 2:
        return 0.95
    if hits == 1:
        return 0.8
    return 0.45


def _numeric_confidence(values: list[float], epsilon: float) -> float:
    """Score a numeric contradiction by how far it exceeds the epsilon
    threshold. Tiny gaps stay soft (≤0.5), gaps ≥3× epsilon clamp at
    0.9."""
    if not values:
        return 0.0
    spread = max(values) - min(values)
    if epsilon <= 0:
        return 0.7
    ratio = spread / epsilon
    if ratio < 1.2:
        return 0.45
    if ratio < 2.0:
        return 0.7
    if ratio < 3.0:
        return 0.85
    return 0.9


@dataclass
class ContradictionLedger:
    """Append-only set of unique contradictions discovered this turn.

    Dedupes by ``(kind, normalized span)`` so a contradiction that
    appears in multiple observations doesn't get re-rendered three
    times in the next decision prompt.

    ``forced_counter_searches`` keeps the loop honest about its own
    interventions: the adaptive policy is "force at most one counter-
    search per turn, only for a high-confidence un-addressed signal,
    only if we have iteration budget left." Anything beyond that is
    left to the model's own judgement based on the prompt-rendered
    signals, not a hardcoded retry.
    """

    signals: list[ContradictionSignal] = field(default_factory=list)
    forced_counter_searches: int = 0
    FORCE_CONFIDENCE_THRESHOLD: float = 0.65
    MAX_FORCED_PER_TURN: int = 1

    def next_to_force(self, *, iterations_remaining: int) -> ContradictionSignal | None:
        """Return the contradiction the loop should auto-counter-search now.

        ``None`` when the policy declines — either because every signal
        is already addressed, the highest-confidence open signal is below
        the threshold, we've already forced a counter-search this turn,
        or we don't have enough iteration budget left to act on it. The
        loop then falls through to its normal finalize handling, trusting
        the model's own decision in light of the rendered signals.
        """
        if self.forced_counter_searches >= self.MAX_FORCED_PER_TURN:
            return None
        if iterations_remaining < 2:
            # Need at least one iteration to dispatch the counter-search
            # AND one for the model to read the new observation before
            # finalizing. Otherwise the forced search lands too late.
            return None
        open_signals = [s for s in self.signals if not s.addressed]
        if not open_signals:
            return None
        open_signals.sort(key=lambda s: s.confidence, reverse=True)
        top = open_signals[0]
        if top.confidence < self.FORCE_CONFIDENCE_THRESHOLD:
            return None
        return top

    def record_forced(self) -> None:
        self.forced_counter_searches += 1

    def add(self, sig: ContradictionSignal) -> bool:
        norm = _norm(sig.span)
        for existing in self.signals:
            if existing.kind == sig.kind and _norm(existing.span) == norm:
                # Merge: union of source tool names so the prompt shows
                # the full provenance even when the same claim is
                # surfaced by multiple tools.
                for s in sig.sources:
                    if s not in existing.sources:
                        existing.sources.append(s)
                return False
        self.signals.append(sig)
        return True

    def unaddressed(self) -> list[ContradictionSignal]:
        return [s for s in self.signals if not s.addressed]

    def mark_addressed(self, query_text: str) -> int:
        """Mark any signal whose span shares a token with ``query_text`` as
        addressed. Returns the count we marked — useful for tests.
        """
        n = 0
        toks = _tokens(query_text)
        if not toks:
            return 0
        for s in self.signals:
            if s.addressed:
                continue
            if toks & _tokens(s.span):
                s.addressed = True
                n += 1
        return n

    def render_for_prompt(self, limit: int = 4) -> str:
        if not self.signals:
            return "(no contradictions detected)"
        lines: list[str] = []
        # Unaddressed first so the model's attention lands on what
        # still needs counter-evidence.
        ordered = sorted(self.signals, key=lambda s: s.addressed)
        for s in ordered[:limit]:
            lines.append("  - " + s.render())
        more = len(self.signals) - limit
        if more > 0:
            lines.append(f"  ... and {more} more")
        return "\n".join(lines)


# Bounded-cost LLM semantic-contradiction budget per turn. We only
# escalate when (a) two or more papers are present and (b) at least
# one pair has high topic overlap (so they're plausibly making
# competing claims, not just talking past each other). The
# deterministic lexical / numeric detector handles the high-precision
# cases; the LLM handles the cases where disagreement is implicit.
_SEMANTIC_LLM_MAX_PAIRS = 4
_SEMANTIC_TOPIC_OVERLAP_FLOOR = 0.25  # soft prior: pairs below this don't compete


async def detect_semantic_contradictions(
    *,
    query: str,
    results: dict[str, ToolResult],
    existing: "ContradictionLedger",
    max_pairs: int = _SEMANTIC_LLM_MAX_PAIRS,
) -> list["ContradictionSignal"]:
    """LLM-backed semantic contradiction detector.

    The lexical / numeric detector catches explicit disagreement
    ("paper A contradicts paper B", "accuracy 92% vs 71%"). It misses
    the subtler pattern: paper A says "X improves throughput", paper
    B says "X has no measurable effect on throughput" — both with
    confident assertion, neither with an explicit comparison marker.

    Approach:

    * Walk every retrieval result's top papers, build (title, claim)
      pairs from each paper's title + tldr + first 300 chars of
      abstract.
    * Cluster pairs by topic overlap (content-token Jaccard) so we
      only ask the LLM about papers that are plausibly making
      competing claims.
    * Send up to ``max_pairs`` high-overlap pairs to a cheap-model
      structured prompt asking ``"do these disagree, partially
      disagree, or agree?"``.
    * For each LLM-flagged disagreement, build a :class:`ContradictionSignal`
      with ``kind='semantic'`` and a confidence derived from the
      LLM's own confidence level. Pairs already covered by an existing
      signal (same span tokens) are deduped.

    Cost is bounded by ``max_pairs`` * one cheap LLM call. Returns the
    new signals so the caller can fold them into a ledger; never
    raises — any failure returns an empty list.
    """
    try:
        papers = _collect_paper_facts(results)
    except Exception:
        return []
    if len(papers) < 2:
        return []
    # Pair generation: every unordered pair, scored by content-token
    # Jaccard. We only consider pairs above the topic-overlap floor —
    # below that, the papers aren't talking about the same thing and a
    # "disagreement" is just non-overlap, not contradiction.
    pairs: list[tuple[float, dict, dict]] = []
    for i, p_a in enumerate(papers):
        toks_a = _tokens(p_a["text"])
        if not toks_a:
            continue
        for p_b in papers[i + 1:]:
            toks_b = _tokens(p_b["text"])
            if not toks_b:
                continue
            inter = len(toks_a & toks_b)
            union = len(toks_a | toks_b)
            if union == 0:
                continue
            jaccard = inter / union
            if jaccard < _SEMANTIC_TOPIC_OVERLAP_FLOOR:
                continue
            pairs.append((jaccard, p_a, p_b))
    if not pairs:
        return []
    pairs.sort(key=lambda x: x[0], reverse=True)
    pairs = pairs[:max_pairs]

    try:
        from app.adapters.llm import get_llm_adapter
    except Exception:
        return []
    items_payload = [
        {
            "id": i,
            "a_title": p_a["title"][:160],
            "a_claim": p_a["text"][:600],
            "b_title": p_b["title"][:160],
            "b_claim": p_b["text"][:600],
            "topic_overlap": round(jc, 3),
        }
        for i, (jc, p_a, p_b) in enumerate(pairs)
    ]
    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "verdict": {
                            "type": "string",
                            "enum": ["agree", "partial_disagreement", "disagree", "unrelated"],
                        },
                        "summary": {"type": "string", "maxLength": 240},
                        "confidence": {"type": "number"},
                    },
                    "required": ["id", "verdict"],
                },
            },
        },
        "required": ["results"],
    }
    sys_msg = (
        "You are a research-evidence auditor looking for semantic "
        "disagreements between paper pairs. Given two paper claims on "
        "an overlapping topic, decide whether they 'agree', "
        "'partial_disagreement' (overlap mostly but diverge on a key "
        "point), 'disagree' (make incompatible claims), or 'unrelated' "
        "(actually discussing different things despite vocabulary "
        "overlap). Be strict: 'disagree' should only fire when the "
        "claims are genuinely incompatible, not merely different in "
        "focus."
    )
    user_msg = (
        f"User's research question (for relevance scoring): {query[:600]}\n\n"
        "For each pair, return a verdict id, a one-sentence summary "
        "of the disagreement when there is one, and your confidence "
        "(0..1).\n\n"
        + "\n\n".join(
            f"PAIR {it['id']} (topic overlap {it['topic_overlap']:.2f}):\n"
            f"  A: {it['a_title']}\n     {it['a_claim']}\n"
            f"  B: {it['b_title']}\n     {it['b_claim']}"
            for it in items_payload
        )
    )
    try:
        llm = get_llm_adapter()
        raw = await llm.complete_structured(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": user_msg}],
            llm.cheap_model,
            schema,
        )
    except Exception as exc:
        log.debug("semantic contradiction LLM call skipped: %s", exc)
        return []

    new_signals: list[ContradictionSignal] = []
    items = (raw or {}).get("results") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []
    for r in items:
        if not isinstance(r, dict):
            continue
        try:
            pair_id = int(r.get("id"))
        except (TypeError, ValueError):
            continue
        verdict = str(r.get("verdict") or "").lower()
        if verdict not in {"disagree", "partial_disagreement"}:
            continue
        if pair_id < 0 or pair_id >= len(pairs):
            continue
        _jc, p_a, p_b = pairs[pair_id]
        summary = str(r.get("summary") or "").strip() or (
            f"semantic disagreement between {p_a['title'][:60]} and {p_b['title'][:60]}"
        )
        # Confidence: trust the LLM's number when present; otherwise
        # use a tier-based default (disagree → 0.7, partial → 0.5).
        try:
            conf = float(r.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf <= 0.0:
            conf = 0.7 if verdict == "disagree" else 0.5
        sig = ContradictionSignal(
            kind="semantic",
            span=summary,
            sources=[p_a["tool"], p_b["tool"]],
            confidence=max(0.0, min(1.0, conf)),
        )
        # Dedupe against already-known signals.
        if any(_norm(existing_sig.span) == _norm(sig.span) for existing_sig in existing.signals):
            continue
        new_signals.append(sig)
    return new_signals


def _collect_paper_facts(results: dict[str, ToolResult]) -> list[dict]:
    """Flatten retrieval results into ``{tool, title, text}`` records
    we can pair-wise compare. We pick at most a few papers per tool to
    keep the pair count bounded; the topic-overlap filter then trims
    further."""
    out: list[dict] = []
    for tool_name, r in (results or {}).items():
        out_dict = r.output or {}
        for key in ("papers", "results", "items", "candidates"):
            col = out_dict.get(key)
            if not isinstance(col, list):
                continue
            for paper in col[:6]:
                if not isinstance(paper, dict):
                    continue
                title = (paper.get("title") or "").strip()
                tldr = (paper.get("tldr") or "").strip()
                abstract = (paper.get("abstract") or "").strip()
                text = (tldr or abstract[:600]).strip()
                if not title or not text:
                    continue
                out.append({"tool": tool_name, "title": title, "text": text})
            break
    return out


def detect_contradictions_in_results(
    results: dict[str, ToolResult],
    iteration: int,
) -> list[ContradictionSignal]:
    """Scan a per-tool ToolResult dict for contradictions.

    Surface text comes from each ``ToolResult.summary`` and from
    structured prose fields when present (paper abstracts/tldrs,
    comparison cells, survey markdown). We never reach into the full
    output blob — that would be slow and noisy — only the fields that
    typically carry human-readable claims.
    """
    signals: list[ContradictionSignal] = []
    for tool_name, r in (results or {}).items():
        for span in _extract_text_spans(r):
            for pat in _CONTRADICTION_PATTERNS:
                m = pat.search(span)
                if m:
                    snippet = _surrounding_sentence(span, m.start(), m.end())
                    signals.append(ContradictionSignal(
                        kind="lexical",
                        span=snippet,
                        sources=[tool_name],
                        iteration=iteration,
                        confidence=_lexical_confidence(snippet),
                    ))
                    break  # one contradiction per span is enough
    # Cross-paper numeric disagreement.
    metric_map = _collect_metric_values(results)
    for key, hits in metric_map.items():
        if len(hits) < 2:
            continue
        vals = [v for _src, v, _u in hits]
        eps = _epsilon_for(key)
        if max(vals) - min(vals) <= eps:
            continue
        spread = ", ".join(f"{v:g}{u or ''}" for _s, v, u in hits[:4])
        sources = list({src for src, _, _ in hits})
        signals.append(ContradictionSignal(
            kind="numeric",
            span=f"{key} reported as {spread} across sources",
            sources=sources,
            iteration=iteration,
            confidence=_numeric_confidence(vals, eps),
        ))
    return signals


# ── Internals ────────────────────────────────────────────────────────────────


def _extract_text_spans(r: ToolResult) -> list[str]:
    """Pull every human-readable span we should scan for contradictions."""
    spans: list[str] = []
    if r.summary:
        spans.append(r.summary)
    out = r.output or {}
    # Common structured surfaces from our retrieval / synthesis tools.
    for key in ("survey", "answer", "explanation", "notes"):
        v = out.get(key)
        if isinstance(v, str) and v:
            spans.append(v[:4000])
    # Paper-like collections: scan title + abstract + tldr.
    for col_key in ("papers", "results", "items", "candidates", "ideas"):
        col = out.get(col_key)
        if not isinstance(col, list):
            continue
        for c in col[:12]:
            if not isinstance(c, dict):
                continue
            for f in ("title", "abstract", "tldr", "summary"):
                v = c.get(f)
                if isinstance(v, str) and v:
                    spans.append(v[:2000])
    # compare_papers rows / cells.
    rows = out.get("rows")
    if isinstance(rows, list):
        for row in rows[:8]:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if isinstance(cells, dict):
                for v in cells.values():
                    if isinstance(v, str) and v:
                        spans.append(v[:1200])
    return spans


def _surrounding_sentence(text: str, start: int, end: int, pad: int = 140) -> str:
    """Return the sentence containing ``[start:end]`` within ``text``.

    Falls back to a ±``pad`` character window if no sentence boundary is
    detected (e.g. for un-punctuated tool summaries).
    """
    # Sentence boundaries: ``.`` / ``!`` / ``?`` / newline.
    left = max(text.rfind(c, 0, start) for c in ".!?\n")
    right_candidates = [text.find(c, end) for c in ".!?\n"]
    right_candidates = [r for r in right_candidates if r != -1]
    right = min(right_candidates) if right_candidates else -1
    s = left + 1 if left >= 0 else max(0, start - pad)
    e = right + 1 if right >= 0 else min(len(text), end + pad)
    return text[s:e].strip()


def _collect_metric_values(results: dict[str, ToolResult]) -> dict[str, list[tuple[str, float, str]]]:
    """Walk every paper-like row, extract ``metric: value`` pairs."""
    out: dict[str, list[tuple[str, float, str]]] = {}
    for tool_name, r in (results or {}).items():
        for span in _extract_text_spans(r):
            for pat in _METRIC_PATTERNS:
                for m in pat.finditer(span):
                    key = m.group("key").lower().replace(" ", "_")
                    try:
                        val = float(m.group("val"))
                    except (TypeError, ValueError):
                        continue
                    unit = (m.group("unit") or "").lower()
                    out.setdefault(key, []).append((tool_name, val, unit))
    return out


def _epsilon_for(metric_key: str) -> float:
    """Per-metric "different enough to count as contradiction" threshold.

    Conservative defaults; tuned to flag obvious disagreements while
    ignoring noisy benchmark numbers that bounce within a couple of
    points across reruns.
    """
    if metric_key in {"accuracy", "f1", "exact_match", "precision", "recall",
                      "bleu", "rouge", "win_rate", "score"}:
        return 5.0   # percentage / unit-less score points
    if metric_key in {"perplexity"}:
        return 4.0
    if metric_key in {"loss"}:
        return 0.5
    if metric_key in {"latency"}:
        return 100.0  # ms
    if metric_key in {"throughput"}:
        return 1.0    # x / ratio
    if metric_key in {"cost"}:
        return 0.5
    return 5.0


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _tokens(s: str) -> set[str]:
    """Lowercase content-tokens used for the "did the next action target
    this contradiction?" gate. Stop-words are stripped because matching
    on ``the`` or ``and`` would mark everything addressed."""
    raw = {m.group(0).lower() for m in _TOKEN_RE.finditer(s)}
    return raw - _STOPWORDS


_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "but", "for", "are", "was", "were", "with", "from",
    "this", "that", "these", "those", "than", "then", "into", "onto",
    "over", "under", "more", "less", "much", "many", "some", "any",
    "all", "very", "such", "also", "only", "even", "still", "again",
    "have", "has", "had", "been", "being", "their", "there", "where",
    "when", "what", "which", "who", "how", "why", "your", "our", "its",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "paper", "papers", "result", "results", "study", "studies",
    "report", "reported", "show", "shows", "shown", "find", "found",
})
