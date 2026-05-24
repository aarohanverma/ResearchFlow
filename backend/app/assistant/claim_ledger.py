"""Strong-claim ledger and detector for full-paper verification.

The ReAct loop accumulates retrieval results across iterations. Most
of what each result contains is descriptive (titles, authors, taglines,
short summaries) and only a fraction of the text contains *strong
claims* — empirical statements that, if cited in the final answer,
deserve verification against the cited paper's actual content rather
than the abstract or a snippet.

A "strong claim" is, very deliberately, a syntactic heuristic — not
an LLM judgement. We want fast, deterministic detection so the
middleware can fire dozens of times per turn without burning a model
call. The heuristics target the patterns that most often show up in
RA hallucinations / over-claims:

* **Numeric performance claims** — "achieves 92.4% accuracy",
  "reduces latency by 35%", "improves F1 by 4 points".
* **SOTA / superlative claims** — "state-of-the-art", "outperforms
  every baseline", "the first to demonstrate".
* **Causal claims** — "X causes Y", "removing the residual connection
  collapses training".
* **Comparative claims with named methods** — "X beats Method-Y on
  Benchmark-Z".

For each detected claim we record (a) the verbatim span, (b) the
source paper id it was scraped from, (c) the source field (abstract
vs chunk content). The middleware then:

* Inspects each claim at finalize.
* For STRONG claims sourced from an ABSTRACT (or from a snippet
  short enough that we can't tell if the full paper actually
  supports it), it forces a ``paper_qa`` round to verify against
  the paper's chunk-indexed body.
* The middleware annotates the claim's verdict: ``verified`` (paper
  body confirms), ``contradicted`` (paper body says otherwise),
  ``provisional`` (paper_qa couldn't be run, or returned an
  ambiguous answer — synth must label as abstract-only).

This file owns the data structures + heuristics. The middleware
wiring lives in
:mod:`app.assistant.react.middlewares.full_paper_gate`.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ── Heuristic patterns ───────────────────────────────────────────────────────
#
# Each pattern is conservative on purpose — false positives waste
# paper_qa calls (cheap but not free) while false negatives leak
# unverified claims into the final answer. We tuned to favour
# precision over recall on a corpus of RA outputs; the loop's
# critique pass catches the recall miss anyway.

# Numeric performance: "92.4% accuracy", "35% reduction", "4.5 BLEU"
_NUMERIC_PERF = re.compile(
    r"""
    (?P<num>\d+(?:\.\d+)?)\s*
    (?:%|percent|\bpoints?\b|\bpts?\b|x\b|\bms\b|\bs\b|\bGB\b|\bMB\b)?\s*
    (?:
        (?P<verb>improvement|improves|outperforms|reduces|increases|gains?|drops?|beats?)
        |
        (?P<metric>accuracy|f1|bleu|rouge|map|ndcg|precision|recall|exact[-\s]match
          |perplexity|latency|throughput|speedup|win[-\s]rate)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# SOTA / superlative claims
_SOTA = re.compile(
    r"""
    \b(?:
        state[-\s]of[-\s]the[-\s]art | sota
        | first\s+to\s+(?:show|demonstrate|achieve|introduce|propose)
        | best[-\s]known | top[-\s]performing | record[-\s]breaking
        | outperforms?\s+(?:all|every|prior|previous)
        | surpass(?:es|ed)?\s+(?:all|every|prior|previous|the\s+best)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Causal claims. The "removing/ablating … collapses/breaks/…" form
# uses ``(?:\w+\s+){1,5}`` so a real-world noun phrase like
# "the residual connection" sits between the verb and the consequence
# without breaking the match.
_CAUSAL = re.compile(
    r"""
    \b(?:
        causes? | drives? | leads?\s+to | results?\s+in
        | (?:is|are)\s+caused\s+by
        | removing\s+(?:\w+\s+){1,5}(?:collapses|breaks|degrades|hurts?|drops?|reduces?)
        | ablating\s+(?:\w+\s+){1,5}(?:collapses|breaks|degrades|hurts?|drops?|reduces?)
        | because\s+of | due\s+to
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Comparative claims with method-vs-baseline shape
_COMPARATIVE = re.compile(
    r"""
    \b(?:
        compared\s+(?:to|with) | versus | vs\.?
        | over\s+baseline | over\s+the\s+baseline
        | (?:beats?|defeats?|exceeds?)\s+(?:[A-Z][\w-]+)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Source-field tags. Chunks indexed at section level are "chunk"; raw
# abstract / tldr are "abstract"; snippets from web search / external
# APIs are "snippet". The middleware only forces verification when
# the source is abstract / snippet — chunk-derived claims are
# already grounded in the paper body.
SOURCE_ABSTRACT = "abstract"
SOURCE_CHUNK = "chunk"
SOURCE_SNIPPET = "snippet"


def detect_strong_spans(text: str, *, max_spans: int = 3) -> list[str]:
    """Return up to ``max_spans`` verbatim spans from ``text`` that look
    like strong claims.

    Each span is a single sentence (split on ``.``/``!``/``?``) that
    contains at least one of the strong-claim regexes. Returned in
    document order so the synth can preserve narrative flow. Empty
    list when nothing matches — the common case for routine paper
    metadata.

    The spans are not deduplicated globally (different papers can
    make similar claims, and we want each tagged separately); the
    ledger handles dedup with paper-id keys.
    """
    if not text or len(text) < 30:
        return []
    # Split on sentence boundaries. Cheap split — not perfect, but
    # we don't need NLP-grade tokenisation for a heuristic.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    out: list[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) < 30 or len(s) > 500:
            continue
        if (
            _NUMERIC_PERF.search(s)
            or _SOTA.search(s)
            or _CAUSAL.search(s)
            or _COMPARATIVE.search(s)
        ):
            out.append(s)
            if len(out) >= max_spans:
                break
    return out


# ── Ledger entry + container ─────────────────────────────────────────────────


@dataclass
class StrongClaim:
    """One detected strong claim attached to a source paper.

    The verdict starts as ``provisional`` and transitions through
    ``in_flight`` (a forced ``paper_qa`` round has been dispatched but
    not yet resolved) to one of ``verified`` / ``contradicted`` /
    ``unverifiable`` after the full-paper verification middleware
    interprets the ``paper_qa`` answer. The ``in_flight`` marker exists
    so the gate does not re-pick the same claim on its next finalize
    pass — without it the loop would burn the per-turn forced-paper_qa
    budget on the same claim over and over, leaving every other strong
    claim unverified.
    """

    span: str
    paper_id: str
    paper_title: str
    source_field: str            # SOURCE_ABSTRACT | SOURCE_CHUNK | SOURCE_SNIPPET
    iteration_seen: int
    # Verification state — filled by the middleware.
    verdict: str = "provisional"  # provisional | in_flight | verified | contradicted | unverifiable
    verification_note: str = ""
    verified_at_iteration: int | None = None

    def needs_verification(self) -> bool:
        """True when the claim hasn't been verified and the source
        field is one we can deepen (abstract / snippet, not chunk).

        ``in_flight`` claims are NOT considered candidates — a forced
        verification round has already been dispatched and the gate is
        waiting on the ``after_tool`` hook to resolve the verdict.
        """
        if self.verdict != "provisional":
            return False
        return self.source_field in (SOURCE_ABSTRACT, SOURCE_SNIPPET)


@dataclass
class ClaimLedger:
    """Running inventory of strong claims across the turn.

    Keyed by ``(paper_id, span_hash)`` so repeated retrievals of the
    same claim don't bloat the ledger. The middleware iterates over
    ``needs_verification()`` entries at finalize time.
    """

    by_key: "OrderedDict[str, StrongClaim]" = field(default_factory=OrderedDict)

    def add(self, claim: StrongClaim) -> bool:
        """Register a new claim. Returns True when stored, False on
        duplicate (same paper + same first-80-chars of span)."""
        key = self._key(claim)
        if key in self.by_key:
            return False
        self.by_key[key] = claim
        return True

    def _key(self, claim: StrongClaim) -> str:
        head = (claim.span or "")[:80].lower().strip()
        return f"{claim.paper_id}::{hash(head)}"

    def unverified(self) -> list[StrongClaim]:
        """Claims that still need a full-paper verification pass."""
        return [c for c in self.by_key.values() if c.needs_verification()]

    def by_paper(self, paper_id: str) -> list[StrongClaim]:
        return [c for c in self.by_key.values() if c.paper_id == str(paper_id)]

    def find_pending(self, paper_id: str, claim_span: str) -> "StrongClaim | None":
        """Locate the in-flight provisional claim that a forced
        ``paper_qa`` round was dispatched to verify.

        Match key is ``(paper_id, lowercased-stripped first 80 chars of
        the span)`` — the same key the ledger uses for dedup. Returns
        ``None`` when no in-flight claim matches; the after_tool hook
        treats that as "this paper_qa was user-initiated, not
        gate-forced" and skips verdict resolution.
        """
        if not paper_id or not claim_span:
            return None
        head = (claim_span or "")[:80].lower().strip()
        if not head:
            return None
        target_key = f"{str(paper_id)}::{hash(head)}"
        candidate = self.by_key.get(target_key)
        if candidate is None or candidate.verdict != "in_flight":
            return None
        return candidate

    def summarize(self) -> dict[str, Any]:
        """Compact view for the synthesizer's agent_notes block.

        Lists the unverified strong claims (so the answer can label
        them as provisional / abstract-only) and the verified ones
        (so the answer can quote them with confidence). Capped to
        keep the payload bounded.
        """
        total = len(self.by_key)
        verified = [c for c in self.by_key.values() if c.verdict == "verified"]
        contradicted = [c for c in self.by_key.values() if c.verdict == "contradicted"]
        unverifiable = [c for c in self.by_key.values() if c.verdict == "unverifiable"]
        provisional = [
            c
            for c in self.by_key.values()
            # ``in_flight`` claims are mid-verification — surface them
            # alongside provisional so the synthesizer still hedges on
            # them when the forced paper_qa never lands (worker crash,
            # ctx cancel) and the verdict stays unresolved.
            if c.verdict in ("provisional", "in_flight")
        ]
        return {
            "total": total,
            "verified_count": len(verified),
            "contradicted_count": len(contradicted),
            "unverifiable_count": len(unverifiable),
            "provisional_count": len(provisional),
            "verified": [
                {"paper_id": c.paper_id, "span": c.span[:240], "note": c.verification_note[:200]}
                for c in verified[:6]
            ],
            "contradicted": [
                {"paper_id": c.paper_id, "span": c.span[:240], "note": c.verification_note[:200]}
                for c in contradicted[:6]
            ],
            "provisional": [
                {"paper_id": c.paper_id, "span": c.span[:240], "source": c.source_field}
                for c in (provisional + unverifiable)[:6]
            ],
        }

    def render_for_prompt(self, limit: int = 8) -> str:
        """One-line-per-claim view for the ReAct decision prompt so the
        model sees which strong claims have been verified vs still
        pending."""
        if not self.by_key:
            return "(no strong claims detected yet)"
        lines: list[str] = []
        for c in list(self.by_key.values())[:limit]:
            tag = c.verdict.upper()
            lines.append(
                f"  [{tag}] paper={c.paper_id[:12]} src={c.source_field} "
                f"claim={c.span[:140]!r}"
            )
        if len(self.by_key) > limit:
            lines.append(f"  ... and {len(self.by_key) - limit} more")
        return "\n".join(lines)


# ── Affirmative / refutation cues for paper_qa answers ─────────────────────
#
# These detectors are tied to ``paper_qa``'s strict instruction prompt
# ("answer the question precisely; if the question cannot be answered
# from the provided excerpts, say so clearly"). They are intentionally
# conservative: anything ambiguous defaults to "unverifiable", which the
# synthesizer treats as still-provisional rather than overstating.

_REFUTATION_PHRASES: tuple[str, ...] = (
    "not supported by", "does not support", "is not supported",
    "no evidence", "no support",
    "the paper does not", "the paper does not actually", "the paper does not state",
    "the paper does not claim", "the paper does not show", "the paper does not contain",
    "the paper does not address", "the paper does not provide",
    "cannot be answered", "cannot answer", "can't be answered",
    "is not addressed", "is not in the paper", "is not present",
    "contradicts", "contrary to", "refutes",
    "[synthesis failed",  # paper_qa's own LLM-failure stub
)

_AFFIRMATIVE_PHRASES: tuple[str, ...] = (
    "yes,", "yes.", "indeed", "the paper states", "the paper reports",
    "the paper claims", "the paper shows", "the paper demonstrates",
    "the paper confirms", "supports the claim", "is supported by",
    "supports this", "confirms the", "directly supports",
    "the paper does support",
)


def _looks_like_refutation(text: str) -> bool:
    """Return True when the paper_qa answer explicitly says the asked
    claim is NOT supported by the paper body, OR when synthesis itself
    failed.

    Used to suppress the "extract spans → verified" path so a refutation
    answer doesn't accidentally seed the ledger with verified-looking
    claims drawn from its own negation phrasing.
    """
    if not text:
        return True
    low = text[:600].lower()
    return any(phrase in low for phrase in _REFUTATION_PHRASES)


def _span_has_negation(span: str) -> bool:
    """Sentence-local negation guard for individual extracted spans.

    A surrounding paragraph may be supportive but a single sentence
    can still deny what we'd otherwise mistakenly mine as a verified
    claim ("but the model does not actually achieve 95%"). Returns True
    when the span itself carries a clear negation cue.
    """
    if not span:
        return False
    low = span.lower()
    return (
        " not " in low
        or low.startswith("not ")
        or "n't" in low
        or " no " in low
        or "fail" in low
    )


def resolve_paper_qa_verdict(answer: str) -> tuple[str, str]:
    """Classify a ``paper_qa`` answer as ``verified`` / ``contradicted``
    / ``unverifiable`` against the asked claim.

    Returns ``(verdict, note)``. ``note`` is a short audit string the
    middleware stamps onto the claim so the synthesizer can render the
    reason in the agent_notes block.

    Defaults to ``unverifiable`` on ambiguous answers — the synthesizer
    treats that as "we tried, we couldn't conclude", which is honest
    rather than overconfident. Refutations dominate affirmations when
    both phrasings appear, because a paper that "states X but does NOT
    actually achieve X" is still a refutation of the original claim.
    """
    if not answer:
        return "unverifiable", "paper_qa returned empty answer"
    low = answer[:1200].lower()
    if any(phrase in low for phrase in _REFUTATION_PHRASES):
        return "contradicted", "paper_qa answer explicitly refuted the claim"
    if any(phrase in low for phrase in _AFFIRMATIVE_PHRASES):
        return "verified", "paper_qa answer affirms the claim against paper body"
    return "unverifiable", "paper_qa answer was ambiguous about the claim"


# ── Extraction from tool results ─────────────────────────────────────────────


def extract_claims_from_result(
    *, action: str, result: Any, iteration: int,
) -> list[StrongClaim]:
    """Scan a ToolResult for strong claims and tag each with its source paper.

    ``action`` is the tool name; we use it to decide how to walk the
    output shape. Retrieval-class tools have a ``papers`` / ``results``
    list with abstract/tldr; ``paper_qa`` returns chunk-grounded text
    that we tag as ``SOURCE_CHUNK`` — but only when the answer text is
    affirmative. ``paper_qa`` answers that explicitly refute the asked
    question are NOT mined for "verified" spans; otherwise a sentence
    like "The paper does NOT achieve 95% accuracy" would land "95%
    accuracy" in the ledger as a verified claim (the regex matches the
    span; the negation context is lost).

    Returns an empty list when the output shape isn't recognised or the
    tool reported ``found=False`` — the middleware just keeps moving in
    that case.
    """
    try:
        out = (getattr(result, "output", None) or {})
    except Exception:
        return []

    claims: list[StrongClaim] = []

    # paper_qa: answer is chunk-grounded, tag SOURCE_CHUNK so we
    # don't re-verify it.
    if action == "paper_qa":
        # ``found=False`` (no paper resolved, no chunks indexed) gives
        # us a synthetic placeholder answer like "Paper not found"; we
        # must not mine that for strong claims.
        if out.get("found") is False:
            return claims
        ans = (out.get("answer") or "")
        pid = str(out.get("paper_id") or "")
        title = str(out.get("paper_title") or "")[:160]
        if pid and ans and not _looks_like_refutation(ans):
            for span in detect_strong_spans(ans, max_spans=4):
                # Skip spans that themselves carry negation cues — even
                # in an affirmative-overall answer, a single sentence
                # may be denying support for the asked claim.
                if _span_has_negation(span):
                    continue
                claims.append(StrongClaim(
                    span=span, paper_id=pid, paper_title=title,
                    source_field=SOURCE_CHUNK, iteration_seen=iteration,
                    verdict="verified",
                    verification_note="extracted from paper_qa chunk synthesis",
                    verified_at_iteration=iteration,
                ))
        return claims

    # Retrieval shapes — papers / results / items / candidates list
    candidates: list = []
    for key in ("papers", "results", "items", "candidates"):
        v = out.get(key)
        if isinstance(v, list):
            candidates = v
            break

    for c in candidates[:20]:  # cap per result so a 500-paper dump doesn't OOM the ledger
        if not isinstance(c, dict):
            continue
        pid = str(c.get("paper_id") or c.get("id") or c.get("external_id") or "")
        if not pid:
            continue
        title = (c.get("title") or "")[:160]
        # Try abstract/tldr first (the most common "strong claim" carrier)
        text_field = c.get("abstract") or c.get("tldr") or c.get("snippet") or c.get("summary") or ""
        if not isinstance(text_field, str):
            continue
        source = SOURCE_ABSTRACT if (c.get("abstract") or c.get("tldr")) else SOURCE_SNIPPET
        for span in detect_strong_spans(text_field, max_spans=2):
            claims.append(StrongClaim(
                span=span, paper_id=pid, paper_title=title,
                source_field=source, iteration_seen=iteration,
            ))

    return claims


__all__ = [
    "ClaimLedger",
    "SOURCE_ABSTRACT",
    "SOURCE_CHUNK",
    "SOURCE_SNIPPET",
    "StrongClaim",
    "detect_strong_spans",
    "extract_claims_from_result",
    "resolve_paper_qa_verdict",
]
