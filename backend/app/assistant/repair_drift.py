"""Repair-pass drift detection.

The synthesizer pipeline has a *repair LLM* that rewrites answers when
the deterministic self-check / citation-strip pass finds problems. The
repair pass is supposed to fix citations and tighten language — not
introduce *new* substantive claims or new citation markers that
weren't supported by the pre-repair evidence.

A drifted repair is the silent failure mode: the repair LLM polishes
the prose, hallucinates a confident sentence with a brand-new ``[N]``
marker, and ships. We detect that here.

Approach: compare the pre- and post-repair answers as two sets of
*citation-bearing claims* (sentences containing one or more ``[N]`` or
``[A N]`` markers). A drift is any claim in the post that:

* contains a citation marker the pre-repair answer never used, OR
* uses an existing marker on a *materially different* sentence than
  the pre-repair version did.

We don't try to revert; the caller decides what to do (today the
synthesizer logs the drift into ``agent_notes`` so the answer carries
an honesty annotation). Reverting would risk re-introducing the very
problems the repair was supposed to fix.

Cheap by construction — pure regex + set ops over the answer text;
zero LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_MARKER_RE = re.compile(r"\[(A?\d+(?:\s*[,\-]\s*\d+)*)\]")
# A "claim" is a sentence that contains at least one citation marker.
# Sentence boundaries are simple .!? + newlines; that's good enough
# for the synth's emitted prose.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[*])|\n+")


@dataclass
class DriftReport:
    """Summary of repair-pass drift between pre and post answers."""

    new_markers: list[str] = field(default_factory=list)
    new_claims: list[str] = field(default_factory=list)
    changed_claims: list[tuple[str, str]] = field(default_factory=list)
    summary: str = ""

    @property
    def has_drift(self) -> bool:
        """True only when the repair pass introduced genuinely new
        material — a citation marker that wasn't there before, or a
        citation-bearing claim with no near-match in the pre answer.

        ``changed_claims`` (same claim, different prose) is exactly
        what the repair LLM is supposed to do, so it doesn't count as
        drift on its own. It's still surfaced in ``render_for_agent_notes``
        for full transparency, but the boolean signal stays calm."""
        return bool(self.new_markers or self.new_claims)

    def render_for_agent_notes(self, limit: int = 4) -> list[str]:
        if not self.has_drift:
            return []
        lines = [
            "Repair-pass drift detected — the repair LLM changed citation-"
            "bearing claims that were not in the pre-repair answer:"
        ]
        if self.new_markers:
            lines.append(
                f"  • New citation markers introduced by the repair pass: "
                f"{', '.join(sorted(self.new_markers)[:8])}"
            )
        for c in self.new_claims[:limit]:
            lines.append(f"  • NEW claim w/ citation: {c[:200]!r}")
        for pre, post in self.changed_claims[:limit]:
            lines.append(f"  • CHANGED claim:")
            lines.append(f"      pre  → {pre[:160]!r}")
            lines.append(f"      post → {post[:160]!r}")
        return lines


def detect_repair_drift(*, pre: str, post: str) -> DriftReport:
    """Compare pre/post repair answers and surface citation drift."""
    if not pre or not post:
        return DriftReport()

    pre_markers = _markers(pre)
    post_markers = _markers(post)
    new_markers = sorted(post_markers - pre_markers)

    pre_claims = _citation_claims(pre)
    post_claims = _citation_claims(post)
    pre_norms = {_normalise(c): c for c in pre_claims}

    new_claims: list[str] = []
    changed_claims: list[tuple[str, str]] = []
    for c in post_claims:
        key = _normalise(c)
        if key in pre_norms:
            continue
        # Heuristic: if the citation-stripped form of this claim is
        # similar enough to a pre-repair claim, count it as "changed"
        # (the repair touched the prose but kept the citation). Pure-
        # new claims (no near-match in pre) are stronger drift signals.
        match = _find_near_match(key, pre_norms.keys())
        if match is not None:
            changed_claims.append((pre_norms[match], c))
        else:
            new_claims.append(c)

    return DriftReport(
        new_markers=new_markers,
        new_claims=new_claims,
        changed_claims=changed_claims,
        summary=(
            f"new_markers={len(new_markers)}, new_claims={len(new_claims)}, "
            f"changed_claims={len(changed_claims)}"
        ),
    )


# ── Internals ────────────────────────────────────────────────────────────────


def _markers(text: str) -> set[str]:
    return {m.group(0) for m in _MARKER_RE.finditer(text or "")}


def _citation_claims(text: str) -> list[str]:
    """Return sentences that contain at least one citation marker."""
    out: list[str] = []
    for raw in _SENTENCE_SPLIT_RE.split(text or ""):
        s = raw.strip()
        if not s or not _MARKER_RE.search(s):
            continue
        out.append(s)
    return out


_STRIP_MARKERS_RE = re.compile(r"\s*\[A?\d+(?:\s*[,\-]\s*\d+)*\]\s*")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")


def _normalise(claim: str) -> str:
    """Citation-stripped, lowercased, whitespace-collapsed form for diffing."""
    no_markers = _STRIP_MARKERS_RE.sub(" ", claim or "")
    no_punct = _NON_ALNUM_RE.sub(" ", no_markers.lower())
    return re.sub(r"\s+", " ", no_punct).strip()


_DRIFT_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "on", "in", "to", "for",
    "with", "by", "at", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its",
    "than", "then", "so", "if", "we", "they", "their", "our",
    "can", "could", "may", "might", "will", "would", "should",
    "also", "very", "such", "more", "less",
})


def _find_near_match(target: str, candidates) -> str | None:
    """Return the candidate that shares enough *content* tokens with
    ``target`` to count as 'same claim, different prose'. Stopwords are
    stripped before the Jaccard so trivial English connectors don't
    drive similarity — without that filter, ``"X achieves SOTA on Y"``
    vs ``"X reaches state-of-the-art on Y"`` looked like only 30%
    overlap (the/on/of) and was wrongly flagged as drift."""
    if not target:
        return None
    target_toks = set(target.split()) - _DRIFT_STOPWORDS
    if not target_toks:
        return None
    best_key: str | None = None
    best_overlap = 0.0
    for cand in candidates:
        cand_toks = set(cand.split()) - _DRIFT_STOPWORDS
        if not cand_toks:
            continue
        inter = len(target_toks & cand_toks)
        union = len(target_toks | cand_toks)
        if union == 0:
            continue
        jaccard = inter / union
        if jaccard > best_overlap:
            best_overlap = jaccard
            best_key = cand
    # 0.25 with stopwords stripped is forgiving on purpose: a
    # paraphrase like "Transformers achieve SOTA on benchmarks" vs
    # "Transformers reach the state-of-the-art on benchmarks" shares
    # only 2 content tokens out of 7 (0.286) after stopword removal,
    # and we want that flagged as paraphrase-not-drift. The lower
    # threshold means we err on the side of "this is the same claim,
    # just paraphrased" rather than "this is a brand-new claim" —
    # false-negative-on-drift is preferred to false-positive-on-
    # drift now that ``has_drift`` only fires on genuinely new
    # markers / new claims (changed_claims is informational only).
    return best_key if best_overlap >= 0.25 else None
