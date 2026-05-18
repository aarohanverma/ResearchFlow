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


# ─── Evidence-gap extraction ────────────────────────────────────────────────


_GAP_SCHEMA = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": (
                            "The specific claim or topic in the draft that "
                            "lacks support or coverage."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "A focused retrieval query that, if answered, "
                            "would close the gap."
                        ),
                    },
                    "source_hint": {
                        "type": "string",
                        "enum": ["papers", "web", "either"],
                        "description": (
                            "Where the missing evidence likely lives: "
                            "'papers' for arXiv-style scholarly material, "
                            "'web' for current-events / industry / docs, "
                            "'either' when unclear."
                        ),
                    },
                },
                "required": ["claim", "query"],
            },
            "maxItems": 2,
        }
    },
    "required": ["gaps"],
}


async def extract_evidence_gaps(
    *,
    query: str,
    answer: str,
    issues: list[str],
    evidence_excerpt: str,
) -> list[dict[str, str]]:
    """Turn critique/red-team issue strings into actionable retrieval gaps.

    Returns a list of ``{claim, query, source_hint}`` dicts (max 2). On
    any failure returns an empty list — caller falls back to rewording-
    only repair. We deliberately cap at 2 so the worst case is two extra
    retrieval calls per turn.
    """
    if not issues:
        return []
    try:
        from app.adapters.llm import get_llm_adapter
        llm = get_llm_adapter()

        system = (
            "You convert reviewer issues about a research-assistant draft "
            "into concrete RETRIEVAL gaps that, if filled, would close the "
            "issues. Each gap is a tight focused search query that a "
            "downstream retrieval tool can run.\n\n"
            "Rules:\n"
            "  • Only return a gap when more evidence (paper or web) would "
            "    materially help. If an issue is purely stylistic (voice, "
            "    structure, truncation), do not return a gap for it.\n"
            "  • Each query must be a self-contained, search-engine-style "
            "    phrase — not a question.\n"
            "  • Pick the right source_hint: 'papers' for scholarly content, "
            "    'web' for industry / news / docs / benchmarks, 'either' "
            "    only when genuinely unclear.\n"
            "  • Hard cap of 2 gaps. Pick the highest-leverage ones.\n"
            "  • Return strict JSON matching the schema."
        )
        user_msg = (
            f"USER QUERY:\n{query[:4000]}\n\n"
            f"ASSISTANT DRAFT:\n{answer[:12000]}\n\n"
            f"EVIDENCE ALREADY AVAILABLE:\n{evidence_excerpt[:6000]}\n\n"
            f"REVIEWER ISSUES:\n"
            + "\n".join(f"  - {s}" for s in issues)
        )
        raw = await llm.complete_structured(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            llm.cheap_model,
            _GAP_SCHEMA,
        )
        if not isinstance(raw, dict):
            return []
        gaps_in = raw.get("gaps") or []
        out: list[dict[str, str]] = []
        for g in gaps_in[:2]:
            if not isinstance(g, dict):
                continue
            claim = str(g.get("claim") or "").strip()
            q = str(g.get("query") or "").strip()
            if not q:
                continue
            hint = str(g.get("source_hint") or "either").strip()
            if hint not in {"papers", "web", "either"}:
                hint = "either"
            out.append({"claim": claim[:200], "query": q[:200], "source_hint": hint})
        return out
    except Exception as exc:
        log.debug("extract_evidence_gaps failed: %s", exc)
        return []


async def fetch_gap_evidence(
    *,
    user_id,
    namespace_key: str,
    namespace_keys: list[str],
    gap: dict[str, str],
) -> dict | None:
    """Run one focused retrieval call against the gap.

    Returns a small dict describing the fresh evidence we found, suitable
    for splicing into the synthesizer's extra_context block. ``None`` on
    failure so the caller can move on.
    """
    try:
        hint = gap.get("source_hint", "either")
        if hint == "web":
            return await _fetch_web(user_id, namespace_key, namespace_keys, gap["query"])
        if hint == "papers":
            return await _fetch_papers(user_id, namespace_key, namespace_keys, gap["query"])
        # 'either' — try papers first; if that comes back empty, try web.
        result = await _fetch_papers(user_id, namespace_key, namespace_keys, gap["query"])
        if result:
            return result
        return await _fetch_web(user_id, namespace_key, namespace_keys, gap["query"])
    except Exception as exc:
        log.debug("fetch_gap_evidence failed (gap=%s): %s", gap.get("query"), exc)
        return None


async def _make_ctx(user_id, namespace_key: str, namespace_keys: list[str], db):
    """Construct a minimal ToolContext for one-off tool calls inside repair."""
    from uuid import UUID
    from app.assistant.tools.base import ToolContext

    async def _np(_pct: int, _msg: str) -> None:
        pass

    async def _nc() -> bool:
        return False

    return ToolContext(
        user_id=user_id,
        session_id=UUID(int=0),
        namespace_key=namespace_key or "",
        namespace_keys=namespace_keys or ([namespace_key] if namespace_key else []),
        orientation="both",
        expertise_level="practitioner",
        job_id="ra:repair_requery",
        parent_message_id=UUID(int=0),
        db=db,
        emit_progress=_np,
        should_cancel=_nc,
    )


async def _fetch_papers(user_id, namespace_key, namespace_keys, query: str) -> dict | None:
    """Targeted paper retrieval. Caller treats output as advisory evidence."""
    try:
        from app.assistant.tools.deep_search import DeepSearchInput, DeepSearchTool
        from app.db.session import async_session_factory
        async with async_session_factory() as db:
            ctx = await _make_ctx(user_id, namespace_key, namespace_keys, db)
            tool = DeepSearchTool()
            params = DeepSearchInput(
                query=query,
                namespace_keys=namespace_keys or [],
                limit=4,
                include_arxiv_mcp=True,
                arxiv_max_results=3,
            )
            res = await tool.run(ctx, params)
            papers = list(res.output.get("papers") or [])
            if not papers:
                return None
            return {"kind": "papers", "query": query, "papers": papers[:5]}
    except Exception as exc:
        log.debug("repair re-query papers failed: %s", exc)
        return None


async def _fetch_web(user_id, namespace_key, namespace_keys, query: str) -> dict | None:
    """Targeted web retrieval — used for industry, news, benchmark gaps."""
    try:
        from app.assistant.tools.web_search import WebSearchInput, WebSearchTool
        from app.db.session import async_session_factory
        async with async_session_factory() as db:
            ctx = await _make_ctx(user_id, namespace_key, namespace_keys, db)
            tool = WebSearchTool()
            try:
                params = tool.input_schema(query=query, max_results=4)
            except Exception:
                params = WebSearchInput(query=query, max_results=4)
            res = await tool.run(ctx, params)
            results = list(res.output.get("results") or [])
            if not results:
                return None
            return {"kind": "web", "query": query, "results": results[:5]}
    except Exception as exc:
        log.debug("repair re-query web failed: %s", exc)
        return None


def render_gap_evidence(gap_evidence: list[dict]) -> str:
    """Render gap-evidence blocks as an XML-tagged appendix for the synth prompt."""
    if not gap_evidence:
        return ""
    parts: list[str] = []
    for block in gap_evidence:
        if not isinstance(block, dict):
            continue
        if block.get("kind") == "papers":
            paper_lines: list[str] = []
            for i, p in enumerate(block.get("papers") or [], start=1):
                title = (p.get("title") or "").strip()
                authors = ", ".join((p.get("authors") or [])[:3])
                tldr = (p.get("tldr") or p.get("abstract") or "")[:400]
                paper_lines.append(
                    f"  [{i}] {title}\n      Authors: {authors}\n      {tldr}"
                )
            parts.append(
                f"<gap_evidence kind=\"papers\" query=\"{block.get('query', '')[:120]}\">\n"
                + "\n".join(paper_lines)
                + "\n</gap_evidence>"
            )
        elif block.get("kind") == "web":
            web_lines: list[str] = []
            for r in (block.get("results") or [])[:5]:
                title = (r.get("title") or "").strip()
                url = (r.get("url") or "").strip()
                snippet = (r.get("snippet") or "")[:400]
                web_lines.append(f"  - {title} — {snippet} ({url})")
            parts.append(
                f"<gap_evidence kind=\"web\" query=\"{block.get('query', '')[:120]}\">\n"
                + "\n".join(web_lines)
                + "\n</gap_evidence>"
            )
    return "\n\n".join(parts)


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
