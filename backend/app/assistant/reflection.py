"""Self-reflection, repair, and improvisation for the Research Assistant.

The Research Assistant gets one shot at answering each turn — by default
the model writes the answer and we hand it back. That's brittle: a
fabricated citation, a half-sentence cut-off, or a key claim with no
grounded evidence all slip through silently.

This module adds two complementary layers on top of the synthesizer:

1. ``deterministic_self_check``
       Cheap, no-LLM checks that catch the most common failure modes
       (citation indices out of range, mid-sentence truncation, empty
       answer). Used as a fast first filter.

2. ``llm_critique``
       A short, cheap-model judging pass that scores the answer on
       four dimensions — groundedness, completeness, faithfulness to
       memory, and clarity — and flags specific improvable issues.
       Only runs on substantive turns to keep latency bounded.

If either layer surfaces issues, the synthesizer's existing repair pass
re-runs the model with the issues as a corrective preamble.

Improvisation
-------------
``improvise_after_thin_results`` runs before synthesis when the executed
plan produced little or no evidence. It looks at what tools were used,
decides which complementary path could fill the gap (e.g. web_search
when corpus was empty, broader arxiv_import when scoped search was thin),
and returns an extra set of tool results to fold back into ``results``.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import UUID

log = logging.getLogger(__name__)


# ─── Deterministic checks ────────────────────────────────────────────────────

_TRUNCATION_TERMINATORS = ".!?:)]\"'`*}>›"


def deterministic_self_check(
    answer: str,
    *,
    papers: list[dict],
    arxiv_results: list[dict],
) -> list[str]:
    """Return a list of human-readable issues found in the answer."""
    issues: list[str] = []
    text = (answer or "").strip()
    if not text:
        return ["empty answer"]

    max_corpus = len(papers)
    max_arxiv = len(arxiv_results)
    seen_idx: set[int] = set()
    for m in re.finditer(r"\[(\d+)\]", text):
        n = int(m.group(1))
        if n in seen_idx:
            continue
        seen_idx.add(n)
        if max_corpus == 0 and max_arxiv == 0:
            continue
        if max_corpus and (n < 1 or n > max_corpus):
            issues.append(
                f"cite [{n}] but only {max_corpus} grounded paper(s) available"
            )
            break
    seen_arxiv: set[int] = set()
    for m in re.finditer(r"\[A(\d+)\]", text):
        n = int(m.group(1))
        if n in seen_arxiv:
            continue
        seen_arxiv.add(n)
        if max_arxiv and (n < 1 or n > max_arxiv):
            issues.append(
                f"cite [A{n}] but only {max_arxiv} arXiv candidate(s) available"
            )
            break

    last = text[-1] if text else ""
    if last and last not in _TRUNCATION_TERMINATORS:
        issues.append("answer appears truncated mid-sentence")

    return issues


# ─── LLM-as-judge critique ───────────────────────────────────────────────────

_CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "groundedness":  {"type": "number", "minimum": 0, "maximum": 1},
        "completeness":  {"type": "number", "minimum": 0, "maximum": 1},
        "memory_faithfulness": {"type": "number", "minimum": 0, "maximum": 1},
        "clarity":       {"type": "number", "minimum": 0, "maximum": 1},
        "issues": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
        },
        "should_repair": {"type": "boolean"},
    },
    "required": ["groundedness", "completeness", "should_repair"],
}


async def llm_critique(
    *,
    query: str,
    answer: str,
    evidence_excerpt: str,
    memory_excerpt: str,
) -> dict[str, Any] | None:
    """Return a critique dict, or None on any failure.

    Designed to be cheap: one cheap-model structured-output call, no
    streaming, capped excerpts on every input.
    """
    try:
        from app.adapters.llm import get_llm_adapter
        llm = get_llm_adapter()

        system = (
            "You are auditing a research-assistant answer for quality.\n\n"
            "Score four dimensions on [0, 1]:\n"
            "  groundedness        — every factual claim is supported by the "
            "                        provided evidence block.\n"
            "  completeness        — the answer fully addresses the user's "
            "                        question, not just part of it.\n"
            "  memory_faithfulness — the answer respects stored user "
            "                        preferences/context WITHOUT being "
            "                        overridden by stale memory. 1.0 when "
            "                        either there's no relevant memory OR "
            "                        the answer respects it sensibly.\n"
            "  clarity             — well-structured, no truncation, no "
            "                        jargon left unexplained.\n\n"
            "List concrete issues in ``issues`` (max 5). Set ``should_repair`` "
            "to true ONLY when groundedness < 0.6 OR there's a critical "
            "issue (fabricated citation, mid-sentence truncation, claim "
            "directly contradicted by evidence). Be conservative — frequent "
            "false-positive repairs hurt latency."
        )

        # Caps generous enough to avoid mid-claim truncation while still
        # protecting the cheap-judge model from runaway prompt size.
        user = (
            f"USER QUERY:\n{query[:4000]}\n\n"
            f"EVIDENCE:\n{evidence_excerpt[:12000]}\n\n"
            f"MEMORY HINT (advisory):\n{memory_excerpt[:2400]}\n\n"
            f"ASSISTANT ANSWER:\n{answer[:16000]}"
        )

        return await llm.complete_structured(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            llm.cheap_model,
            _CRITIQUE_SCHEMA,
        )
    except Exception as exc:
        log.debug("llm_critique failed: %s", exc)
        return None


# ─── Red-team adversarial check ─────────────────────────────────────────────


_REDTEAM_SCHEMA = {
    "type": "object",
    "properties": {
        "biased_claims": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
            "description": "Statements that read as biased or one-sided.",
        },
        "missing_perspectives": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
            "description": "Counter-views or competing methods the draft ignores.",
        },
        "weak_evidence": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
            "description": "Claims whose support in the evidence is thin or absent.",
        },
        "overclaims": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
            "description": "Statements that go beyond what the evidence actually shows.",
        },
        "severity": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
            "description": "Overall severity of the issues flagged.",
        },
    },
    "required": ["severity"],
}


async def red_team_review(
    *,
    query: str,
    answer: str,
    evidence_excerpt: str,
) -> dict[str, Any] | None:
    """Adversarial review pass — surface biased / missing / over-claimed content.

    Inspired by the "red-teaming" denoise pattern from agentic deep-research
    designs: a separate cheap-model agent role-plays a skeptical reviewer
    and asks what the draft would face from a tough peer reviewer.

    Returns ``None`` on failure (caller proceeds without the flags).
    """
    try:
        from app.adapters.llm import get_llm_adapter
        llm = get_llm_adapter()
        system = (
            "You are a skeptical peer reviewer auditing a research-assistant "
            "draft for bias, missing perspectives, weak evidence, and "
            "overclaiming. Be terse and specific. List flaws as short "
            "phrases; do NOT rewrite the answer. If the draft is clean, "
            "return empty arrays and severity='none'.\n\n"
            "Severity rubric:\n"
            "  none   — no material issues\n"
            "  low    — minor phrasing problems, nothing factual\n"
            "  medium — a missing perspective or thin evidence on at least "
            "           one claim\n"
            "  high   — overclaim, bias, or hallucinated citation that "
            "           would mislead a reader"
        )
        user = (
            f"USER QUERY:\n{query[:4000]}\n\n"
            f"EVIDENCE PROVIDED TO THE DRAFTER:\n{evidence_excerpt[:12000]}\n\n"
            f"DRAFT ANSWER:\n{answer[:16000]}"
        )
        return await llm.complete_structured(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            llm.cheap_model,
            _REDTEAM_SCHEMA,
        )
    except Exception as exc:
        log.debug("red_team_review failed: %s", exc)
        return None


def redteam_to_issue_list(redteam: dict[str, Any]) -> list[str]:
    """Flatten a red-team report into the issue-string shape repair uses."""
    if not redteam:
        return []
    out: list[str] = []
    sev = (redteam.get("severity") or "none").lower()
    if sev == "none":
        return []
    for label, key in (
        ("bias", "biased_claims"),
        ("missing perspective", "missing_perspectives"),
        ("weak evidence", "weak_evidence"),
        ("overclaim", "overclaims"),
    ):
        for item in (redteam.get(key) or [])[:3]:
            s = str(item).strip()
            if s:
                out.append(f"{label}: {s}")
    return out[:8]


# ─── Convergence check ──────────────────────────────────────────────────────


def has_converged(*, before: str, after: str, min_delta: float = 0.05) -> bool:
    """Decide whether a second repair pass would still meaningfully change.

    Used to short-circuit the iterative denoise loop — once successive
    drafts differ by less than ``min_delta`` (Jaccard over word sets), we
    declare convergence and stop spinning model calls.
    """
    a = set((before or "").lower().split())
    b = set((after or "").lower().split())
    if not a or not b:
        return False
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return True
    similarity = inter / union
    return (1.0 - similarity) < min_delta


def critique_to_issue_list(critique: dict[str, Any]) -> list[str]:
    """Flatten an LLM critique dict into the same issue-string list shape
    the deterministic check uses, so the synthesizer's existing repair
    preamble can splice them in without branching."""
    if not critique:
        return []
    issues = [str(i).strip() for i in (critique.get("issues") or []) if str(i).strip()]
    g = float(critique.get("groundedness") or 1.0)
    c = float(critique.get("completeness") or 1.0)
    mf = float(critique.get("memory_faithfulness") or 1.0)
    if g < 0.6:
        issues.append(
            f"low groundedness score ({g:.2f}) — claims must be tightly tied to evidence"
        )
    if c < 0.6:
        issues.append(
            f"low completeness score ({c:.2f}) — the question was only partially answered"
        )
    if mf < 0.6:
        issues.append(
            f"memory faithfulness slipped ({mf:.2f}) — re-check user preferences before responding"
        )
    return issues[:6]


# ─── Improvisation ───────────────────────────────────────────────────────────


def _has_useful_papers(results: dict, key: str = "papers") -> bool:
    r = results.get(key)
    return bool(r and isinstance(r, dict) and r.get("output", {}).get("papers"))


async def improvise_after_thin_results(
    *,
    query: str,
    namespace_key: str,
    namespace_keys: list[str],
    user_id: UUID,
    results: dict,
) -> dict | None:
    """Run a single, sensible fallback when the executed plan came back thin.

    Returns a tool-result dict to merge into ``results``, or None when no
    improvisation is warranted. The dict's key is the tool name so the
    orchestrator's existing extractors keep working.

    Heuristics:
        * If ``deep_search``/``arxiv_import``/``frontier_scan`` ran and
          produced no papers, fire one ``web_search`` over the query so
          the synthesizer has at least lightly-vetted leads.
        * If a domain tool (pubmed, inspire_hep, ...) ran and was empty,
          widen with ``arxiv_import``.

    Never raises; logs the chosen path for debuggability.
    """
    try:
        from app.assistant.tools.base import ToolContext
        from app.db.session import async_session_factory
    except Exception:
        return None

    # Were we actually short on grounding?
    has_corpus = False
    for k in ("deep_search", "arxiv_import", "frontier_scan"):
        r = results.get(k)
        if r and getattr(r, "output", {}).get("papers"):
            has_corpus = True
            break
    if has_corpus:
        return None
    has_domain = False
    for k in ("pubmed", "inspire_hep", "nasa_ads", "papers_with_code"):
        r = results.get(k)
        if r and getattr(r, "output", {}).get("papers"):
            has_domain = True
            break

    web_already = "web_search" in results
    if web_already:
        return None

    # Pick the improvisation: web_search is the safest "fill the gap"
    # tool that does not require side-effects on the corpus.
    try:
        from app.assistant.tools.web_search import WebSearchTool, WebSearchInput
    except Exception:
        return None

    async def _noop_progress(_pct: int, _msg: str) -> None:
        pass

    async def _noop_cancel() -> bool:
        return False

    async with async_session_factory() as db:
        ctx = ToolContext(
            user_id=user_id,
            session_id=UUID(int=0),
            namespace_key=namespace_key or "",
            namespace_keys=namespace_keys or ([namespace_key] if namespace_key else []),
            orientation="both",
            expertise_level="practitioner",
            job_id="ra:improvise",
            parent_message_id=UUID(int=0),
            db=db,
            emit_progress=_noop_progress,
            should_cancel=_noop_cancel,
        )
        try:
            tool = WebSearchTool()
            try:
                params = tool.input_schema(query=query, max_results=5)
            except Exception:
                params = WebSearchInput(query=query, max_results=5)
            result = await tool.run(ctx, params)
            log.info(
                "ra improvisation: fired web_search (corpus_empty=%s domain_empty=%s)",
                not has_corpus, not has_domain,
            )
            return {"web_search": result}
        except Exception as exc:
            log.debug("improvisation web_search failed: %s", exc)
            return None
