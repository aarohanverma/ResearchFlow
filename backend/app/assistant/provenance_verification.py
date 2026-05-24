"""Claim-level provenance verification.

The synthesizer emits ``[N]`` citation markers tying each claim to a
specific paper. Until now we trusted those markers — if the model
wrote ``"X improves Y by 20% [3]"``, we accepted that paper [3] really
supports that claim. The auditor's job is to verify it.

This module is the deterministic-first verifier. For every (claim,
cited_paper) pair extracted from the answer, it checks whether the
paper's title + abstract + tldr genuinely overlap with the claim. The
goal isn't *strict entailment* — that needs an LLM call per claim and
costs scale linearly with answer length. The goal is to catch the
**obvious** failure mode: cited paper N has nothing to do with the
claim's topic. That's the high-frequency hallucination.

Three signal layers, cheapest first:

* **Lexical overlap** — content tokens (≥3 chars, non-stopword) shared
  between the claim sentence and the paper text. ≥2 shared content
  tokens or one shared bigram = supported.
* **Salient noun overlap** — capitalised tokens, acronyms, and
  hyphenated technical terms get extra weight; a missing salient
  noun is a strong "claim refers to something this paper never
  mentioned" signal.
* **Fallback verdict** — when overlap is borderline we mark the
  claim ``"unverified"`` rather than ``"unsupported"`` so the
  synthesizer can caveat it instead of stripping a possibly-correct
  citation. False positives here cost more than false negatives.

The verifier returns a :class:`VerificationReport` aggregating
per-claim verdicts, the overall verified-claim ratio, and a list of
specific unsupported (marker, claim, paper_id) triples the synthesizer
renders into ``<agent_notes>`` so the answer either drops or caveats
them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ── Public types ─────────────────────────────────────────────────────────────


@dataclass
class ClaimVerdict:
    marker: str            # e.g. "[3]" or "[A2]"
    claim: str             # the sentence the marker sits in
    paper_id: str          # the paper UUID / external id the marker resolves to
    paper_title: str
    verdict: str           # 'supported' | 'unverified' | 'unsupported'
    overlap_score: float   # 0..1 — fraction of claim content tokens found in paper text
    matched_tokens: list[str] = field(default_factory=list)
    missing_salient: list[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """Aggregate result the synthesizer reads after verification."""

    claims: list[ClaimVerdict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(1 for c in self.claims if c.verdict == "supported")

    @property
    def unsupported(self) -> int:
        return sum(1 for c in self.claims if c.verdict == "unsupported")

    @property
    def unverified(self) -> int:
        return sum(1 for c in self.claims if c.verdict == "unverified")

    @property
    def verified_ratio(self) -> float:
        return self.supported / self.total if self.total else 1.0

    def render_for_agent_notes(self, limit: int = 6) -> list[str]:
        """Compact lines the synthesizer can embed in ``<agent_notes>``."""
        out = [
            f"Provenance verification: {self.supported}/{self.total} claims "
            f"supported by the cited paper text "
            f"(unverified={self.unverified}, unsupported={self.unsupported}).",
        ]
        bad = [c for c in self.claims if c.verdict in ("unsupported", "unverified")]
        if not bad:
            return out
        out.append("  Flagged claim/paper pairs:")
        for c in bad[:limit]:
            short_claim = c.claim[:140].rsplit(" ", 1)[0]
            missing = ", ".join(c.missing_salient[:3]) if c.missing_salient else ""
            tail = f" (paper missing: {missing})" if missing else ""
            out.append(
                f"    • {c.verdict.upper()} {c.marker} → {c.paper_title[:60]!r}: "
                f"{short_claim}{tail}"
            )
        if len(bad) > limit:
            out.append(f"    ... and {len(bad) - limit} more")
        return out


# ── Public API ──────────────────────────────────────────────────────────────


# Bounded-cost LLM escalation budget per turn. The deterministic
# verifier handles the bulk of claims for free; the LLM only sees the
# subset that landed on the fence (``unverified``). Eight is enough to
# resolve a typical long-form answer's ambiguous citations while
# keeping the latency cost bounded — at the cheap model that's roughly
# one extra second per turn in the worst case.
_LLM_ESCALATION_BUDGET = 8


async def escalate_unverified_with_llm(
    *,
    report: "VerificationReport",
    papers: list[dict],
    arxiv_results: list[dict] | None = None,
    budget: int = _LLM_ESCALATION_BUDGET,
) -> "VerificationReport":
    """LLM-backed entailment check for claims the deterministic verifier
    couldn't decide.

    The deterministic pass catches the obvious "cited paper has nothing
    to do with this claim" failures — that's roughly 80% of real
    citation hallucinations. The remaining 20% are subtler: the paper
    is topically right, the claim shares enough vocabulary, but the
    paper actually says something *different* from what the claim
    attributes to it. This is where an LLM entailment check is
    worth the cost.

    Design points:

    * **Only escalates ``unverified``.** Claims already marked
      ``supported`` or ``unsupported`` by the deterministic pass are
      left alone — escalating those would burn budget without
      changing verdicts.
    * **Single batched LLM call.** All unverified claims (up to
      ``budget``) are sent in one structured request. One round-trip,
      one cache slot, one prompt-cache hit on the second turn.
    * **Trusts the LLM's verdict verbatim.** We don't post-process the
      verdict back through a numeric threshold (which would be its
      own overfitting target); we just take ``supported`` /
      ``unsupported`` / ``partial`` and rewrite the entry. ``partial``
      becomes ``unverified`` (caveat in the answer) so we err on
      transparency rather than confident either-way claims.
    * **Graceful fallback.** Any failure — LLM unavailable, schema
      mismatch, missing paper text — leaves the deterministic verdict
      in place. Verification must never block the answer.

    Returns the same :class:`VerificationReport` instance, mutated in
    place.
    """
    pending = [c for c in report.claims if c.verdict == "unverified"][:budget]
    if not pending:
        return report
    try:
        from app.adapters.llm import get_llm_adapter
    except Exception:
        return report
    arxiv_results = arxiv_results or []

    items_payload: list[dict] = []
    for i, c in enumerate(pending):
        paper = _resolve_paper(c, papers, arxiv_results)
        if paper is None:
            continue
        items_payload.append({
            "id": i,
            "marker": c.marker,
            "claim": c.claim[:600],
            "paper_title": (paper.get("title") or "")[:200],
            "paper_text": _paper_text_blob(paper)[:1400],
        })
    if not items_payload:
        return report

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
                            "enum": ["supported", "partial", "unsupported"],
                        },
                        "rationale": {"type": "string", "maxLength": 400},
                    },
                    "required": ["id", "verdict"],
                },
            },
        },
        "required": ["results"],
    }
    sys_msg = (
        "You are a citation auditor. For each (claim, paper_text) pair, "
        "decide whether the paper genuinely supports the claim. "
        "'supported' = the paper makes (or directly implies) the claim. "
        "'partial' = the paper covers the topic but the claim adds an "
        "interpretation the paper doesn't make. "
        "'unsupported' = the paper is about something different. "
        "Be strict — when in doubt, choose 'partial' or 'unsupported'. "
        "Return one verdict per item id."
    )
    user_msg = (
        "Audit each item. Each item gives a claim sentence and the cited "
        "paper's title + relevant text. Decide whether the claim is "
        "supported by the paper.\n\n"
        + "\n\n".join(
            f"ITEM {it['id']} (marker {it['marker']}):\n"
            f"  CLAIM: {it['claim']}\n"
            f"  PAPER: {it['paper_title']}\n"
            f"  PAPER TEXT: {it['paper_text']}"
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
        log.debug("provenance LLM escalation skipped: %s", exc)
        return report

    results = (raw or {}).get("results") if isinstance(raw, dict) else None
    if not isinstance(results, list):
        return report
    # Index by id so the LLM can return results in any order.
    by_id: dict[int, dict] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            by_id[int(r.get("id"))] = r
        except (TypeError, ValueError):
            continue
    for i, c in enumerate(pending):
        r = by_id.get(i)
        if not r:
            continue
        verdict = str(r.get("verdict") or "").lower()
        if verdict == "supported":
            c.verdict = "supported"
        elif verdict == "unsupported":
            c.verdict = "unsupported"
        elif verdict == "partial":
            # Keep unverified — partial support warrants a caveat in
            # the answer, not a strip.
            c.verdict = "unverified"
    return report


def _resolve_paper(claim: ClaimVerdict, papers: list[dict], arxiv_results: list[dict]) -> dict | None:
    """Find the paper this claim's marker points to. Defensive — the
    deterministic verifier already validated indices, but we re-check
    by paper_id since the LLM escalator runs separately."""
    pid = claim.paper_id
    if not pid:
        return None
    for p in papers or []:
        if str(p.get("paper_id") or "") == pid:
            return p
    for p in arxiv_results or []:
        if str(p.get("external_id") or p.get("paper_id") or "") == pid:
            return p
    return None


def verify_claims(
    *,
    answer: str,
    papers: list[dict],
    arxiv_results: list[dict] | None = None,
) -> VerificationReport:
    """Verify every ``[N]`` / ``[A N]`` marker in ``answer`` against its paper.

    ``papers`` is the 1-indexed corpus paper list; ``arxiv_results`` is
    the 1-indexed external arXiv list. The marker numbering matches
    the synthesizer's emission contract:

      * ``[1]…[N]`` index into ``papers``
      * ``[A1]…[An]`` index into ``arxiv_results``

    Returns a :class:`VerificationReport`. Never raises — verification
    is best-effort and failure must not block the answer.
    """
    report = VerificationReport()
    if not answer:
        return report
    papers = papers or []
    arxiv_results = arxiv_results or []

    for marker, kind, idx_list, claim_span in _extract_claim_citations(answer):
        sources = arxiv_results if kind == "arxiv" else papers
        for n in idx_list:
            if n < 1 or n > len(sources):
                continue
            paper = sources[n - 1]
            verdict, overlap, matched, missing = _verify_one_pair(
                claim=claim_span,
                paper=paper,
            )
            report.claims.append(ClaimVerdict(
                marker=marker,
                claim=claim_span,
                paper_id=str(paper.get("paper_id") or paper.get("external_id") or ""),
                paper_title=str(paper.get("title") or ""),
                verdict=verdict,
                overlap_score=overlap,
                matched_tokens=matched,
                missing_salient=missing,
            ))
    return report


# ── Internals ────────────────────────────────────────────────────────────────


# Matches a sentence containing one or more citation markers. We capture
# the sentence (up to the previous sentence boundary or ~250 chars
# back) so the verifier sees the *claim*, not just the marker.
_CITATION_MARKER_RE = re.compile(
    r"\[(A?)(\d+(?:\s*,\s*\d+)*(?:\s*-\s*\d+)?)\]"
)


def _extract_claim_citations(answer: str) -> list[tuple[str, str, list[int], str]]:
    """Yield ``(marker_text, kind, [resolved_indices], claim_sentence)``.

    Sentence boundary detection is greedy + conservative — we look back
    to the previous ``.``/``!``/``?``/``\\n`` and forward to the next
    one to extract the claim that sits AROUND the marker.
    """
    out: list[tuple[str, str, list[int], str]] = []
    for m in _CITATION_MARKER_RE.finditer(answer):
        is_arxiv = m.group(1) == "A"
        idx_list = _parse_index_range(m.group(2))
        if not idx_list:
            continue
        start, end = m.span()
        left_bound = max(
            (answer.rfind(c, 0, start) for c in ".!?\n"),
            default=-1,
        )
        right_candidates = [answer.find(c, end) for c in ".!?\n"]
        right_candidates = [r for r in right_candidates if r != -1]
        right_bound = min(right_candidates) if right_candidates else len(answer)
        claim = answer[left_bound + 1: right_bound + 1].strip()
        marker_text = f"[{'A' if is_arxiv else ''}{','.join(str(i) for i in idx_list)}]"
        out.append((marker_text, "arxiv" if is_arxiv else "corpus", idx_list, claim))
    return out


def _parse_index_range(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a, b = (int(x.strip()) for x in part.split("-", 1))
            except ValueError:
                continue
            if 0 < a <= b and b - a < 30:
                out.extend(range(a, b + 1))
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out


# Thresholds for the verdict ladder. Borderline cases mark "unverified"
# (keep the citation, caveat in agent_notes) rather than "unsupported"
# (drop the citation outright).
_SUPPORTED_THRESHOLD = 0.30
_UNVERIFIED_FLOOR = 0.12


def _verify_one_pair(*, claim: str, paper: dict) -> tuple[str, float, list[str], list[str]]:
    """Compare ``claim`` against ``paper``'s text and return a verdict."""
    paper_text = _paper_text_blob(paper)
    paper_tokens = _content_tokens(paper_text)
    paper_bigrams = _bigrams(paper_text)
    claim_tokens = _content_tokens(claim)
    if not claim_tokens or not paper_tokens:
        return "unverified", 0.0, [], []

    matched = [t for t in claim_tokens if t in paper_tokens]
    overlap_score = len(matched) / max(1, len(claim_tokens))

    # Salient nouns (capitalised words, acronyms, hyphenated technical
    # terms) carry more weight — a paper that talks about
    # "retrieval-augmented generation" but never mentions "MoE" is
    # almost certainly NOT the right citation for a claim about MoE.
    claim_salient = _salient_terms(claim)
    matched_salient = [t for t in claim_salient if t.lower() in paper_text.lower()]
    missing_salient = [t for t in claim_salient if t not in matched_salient]

    # Bigram match is a tie-breaker: a shared 2-word phrase is much
    # stronger evidence than two independently-matched words.
    claim_bigrams = _bigrams(claim)
    bigram_hit = any(bg in paper_bigrams for bg in claim_bigrams)

    # Salient-noun veto: when the claim mentions specific named
    # entities / acronyms / hyphenated technical terms and NONE of
    # them appear in the paper text, the citation almost certainly
    # belongs to a different paper. Partial salient matches (some
    # named entities in the paper, some not) are tolerated — papers
    # routinely use "1.6T parameters" where a claim says "trillion-
    # parameter regimes", and "language modelling" where a claim
    # says "NLP". We caveat (``unverified``) when the missing
    # salient nouns add interpretive content beyond the paper's
    # exact wording, but only HARD-veto when *no* salient noun
    # bridges the claim and the paper.
    # Hard veto only when the claim contains ≥2 salient terms and
    # NONE of them appear in the paper. A single missing salient
    # term (often a 2-3 letter field acronym like "NLP" or "ML") is
    # too noisy to veto on — those routinely appear in claims about
    # papers that don't literally spell them out. With ≥2 missing,
    # the citation almost certainly belongs to a different paper.
    hard_salient_veto = (
        bool(claim_salient)
        and not matched_salient
        and len(claim_salient) >= 2
    )
    soft_salient_caveat = (
        bool(claim_salient)
        and matched_salient
        and len(matched_salient) < len(claim_salient)
    )

    if hard_salient_veto:
        return "unsupported", overlap_score, matched, missing_salient

    strong_match = (
        overlap_score >= _SUPPORTED_THRESHOLD
        or bigram_hit
        or (claim_salient and len(matched_salient) == len(claim_salient))
    )
    if strong_match:
        # Partial salient overlap is still supported but flag the
        # missing terms so the synthesizer can caveat them.
        if soft_salient_caveat and overlap_score < _SUPPORTED_THRESHOLD * 1.5:
            return "unverified", overlap_score, matched, missing_salient
        return "supported", overlap_score, matched, missing_salient
    if overlap_score >= _UNVERIFIED_FLOOR:
        return "unverified", overlap_score, matched, missing_salient
    return "unsupported", overlap_score, matched, missing_salient


def _paper_text_blob(paper: dict) -> str:
    pieces = [
        paper.get("title") or "",
        paper.get("abstract") or "",
        paper.get("tldr") or "",
        " ".join(paper.get("key_concepts") or []),
        " ".join(paper.get("methods_used") or []),
    ]
    return " ".join(str(p) for p in pieces if p)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")


def _content_tokens(text: str) -> set[str]:
    """Lowercase content-word tokens, stopwords removed."""
    raw = {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}
    return raw - _STOPWORDS


def _bigrams(text: str) -> set[tuple[str, str]]:
    toks = [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]
    return {(a, b) for a, b in zip(toks, toks[1:]) if a not in _STOPWORDS and b not in _STOPWORDS}


_SALIENT_RE = re.compile(
    r"\b("
    r"[A-Z]{2,}(?:-[A-Z0-9]+)*"      # acronym e.g. RAG, BERT, MoE
    r"|[A-Z][a-z]+(?:[A-Z][a-z]+)+"  # CamelCase
    r"|[a-z]+(?:-[a-z0-9]+){1,}"     # hyphenated technical term
    r"|[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}"  # Title-Case multi-word
    r")\b"
)


def _salient_terms(text: str) -> list[str]:
    """Pull out probable named-entity / technical-term tokens from a claim.

    Conservative — we'd rather miss a salient term than over-flag
    every adjective as a missing reference.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _SALIENT_RE.finditer(text or ""):
        t = m.group(0).strip()
        if len(t) < 3 or t.lower() in _STOPWORDS:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:12]


# Stopwords + filler words that shouldn't count as content overlap.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "was", "were", "with", "from",
    "this", "that", "these", "those", "than", "then", "into", "onto",
    "over", "under", "more", "less", "much", "many", "some", "any",
    "all", "very", "such", "also", "only", "even", "still", "again",
    "have", "has", "had", "been", "being", "their", "there", "where",
    "when", "what", "which", "who", "how", "why", "your", "our", "its",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "use", "used", "using", "uses",
    "but", "yet", "though", "however", "therefore", "thus", "because",
    "while", "during", "across", "between", "among", "above", "below",
    "paper", "papers", "result", "results", "study", "studies",
    "report", "reported", "show", "shows", "shown", "find", "found",
    "approach", "approaches", "method", "methods", "model", "models",
    "system", "systems", "based", "via", "using", "way", "case",
    "high", "low", "good", "bad", "new", "old", "first", "last",
    "say", "said", "saying", "make", "made", "making",
    "give", "given", "giving", "see", "seen", "seeing",
})
