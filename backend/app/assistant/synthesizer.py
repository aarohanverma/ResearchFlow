"""Synthesizer — composes the final assistant message from step results.

Pure composition: never invents facts. Reads the orchestrator's step outputs,
picks the relevant slices (papers, arXiv results, graph result, genie session),
and asks the quality LLM to write a grounded answer with inline citations.

Also emits a structured ``blocks`` list that the M2 frontend renders as
heterogeneous UI elements (paper grids, step progress, suggestion chips,
artifact links) instead of a wall of markdown.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)


_EXPERTISE_HINTS = {
    "newcomer": (
        "The user is new to this area. Define every technical term the first "
        "time it appears. Scaffold concepts from familiar ground up. Suggest "
        "one foundational paper to read first if relevant. Avoid implying "
        "they should already know things."
    ),
    "practitioner": (
        "The user is a working researcher / engineer. Balance clarity with "
        "implementation and methodology detail. Skip definitions of standard "
        "terminology. Be specific about tradeoffs and gotchas."
    ),
    "expert": (
        "The user is a domain expert. Be terse and technical; assume "
        "background. Highlight subtle distinctions, contested interpretations, "
        "and frontier debates. Skip preamble entirely."
    ),
}
_ORIENTATION_HINTS = {
    "research": "Emphasize novelty, evidence, gaps, methodological tradeoffs, and openness of the question.",
    "production": "Emphasize implementation maturity, validation, constraints, deployment risks, and reproducibility.",
    "both": "Balance research implications (novelty, gaps) with practical execution (implementation, validation).",
}

# End-to-end research lifecycle scaffolding. Even when the user doesn't ask
# for it, the synthesizer should tee up the *next* stage so beginners get
# pulled forward and seasoned researchers can shortcut to where they need
# to be. The "Next moves" section anchors this — see prompt rule 6.
_LIFECYCLE_GUIDANCE = (
    "RESEARCH LIFECYCLE AWARENESS: The user may be at any stage — "
    "discovery, learning, understanding, exploration, synthesis, ideation, "
    "validation, experimentation, implementation, or publication. Infer the "
    "stage from their question and the conversation history. Always close "
    "with one concrete, stage-appropriate next step (e.g. 'read foundational "
    "paper X', 'compare methods Y vs Z', 'run a small ablation that controls "
    "for W', 'draft a related-work section grounded in [1], [3]'). For "
    "complete beginners, scaffold the path forward; for experts, surface "
    "leverage points and unresolved debates."
)

# Output format guidance — injected into the prompt to break the rigid
# Takeaway/Evidence/Gaps/Next-moves template and let the synthesizer choose
# the best representation for the query.
_FORMAT_GUIDANCE = (
    "OUTPUT FORMAT — ADAPTIVE:\n"
    "Choose the format that best serves this specific query and user stage. "
    "Do NOT force a fixed section template. Examples:\n"
    "• Conceptual question → 1-2 sentence core idea, then layered explanation with cited evidence\n"
    "• Discovery / mapping → short synthesis paragraph + ranked paper list with why-surfaced notes\n"
    "• Comparison request → a concise comparison table or side-by-side prose, then verdict\n"
    "• Research roadmap → numbered learning path from foundations to frontier\n"
    "• Narrow technical question → direct answer first, cited evidence, one follow-up\n"
    "• Hypothesis / ideation → premise → evidence → gap → proposed direction\n"
    "• Teaching / beginner → accessible explanation → example → key paper to read next\n"
    "\n"
    "FORMATTING RULES (the renderer supports full GitHub-flavored markdown):\n"
    "• Headings — use `##` (h2) for top-level sections of a multi-part answer; "
    "use `###` (h3) for sub-sections. Never put a heading at the very top — start "
    "with a single TL;DR paragraph instead.\n"
    "• Paragraphs — separate with one blank line. Never use line-break HTML like `<br>`.\n"
    "• Lists — use `- ` for bullets and `1. ` for ordered steps. ONE blank line "
    "before the first list item, no blank lines between items, ONE blank line after.\n"
    "• Emphasis — **bold** for key terms and definitions; *italic* for paper names "
    "and proper-noun emphasis; `code` (inline backticks) for symbols, identifiers, "
    "numeric thresholds, and file paths.\n"
    "• Math — inline math with `$...$`, display math with `$$...$$`. Reserve "
    "LaTeX for genuine math (equations, fractions, sums, integrals, subscripts). "
    "For typographic arrows and operators in PROSE (NOT equations), use Unicode "
    "directly: → ⇒ ↔ × · ± ≤ ≥ ≠ ≈ — never write `$\\rightarrow$` or "
    "`$\\to$` in a sentence. Save `$...$` for symbols that genuinely need math "
    "typesetting.\n"
    "• Code blocks — triple-backticks with a language tag (` ```python ` etc.) "
    "for runnable snippets, pseudocode, or shell commands. Never paste tables of "
    "numbers inside code blocks — use a markdown table instead.\n"
    "• Tables — pipe-separated; use them whenever you are presenting more than three "
    "row-aligned facts (comparison, benchmark numbers, parameter sweeps).\n"
    "• Quotes / callouts — use `> ` blockquotes for direct quotes from a paper.\n"
    "• Mermaid diagrams — when a concept benefits from a flow / state diagram, emit "
    "a ` ```mermaid ` fenced block. Keep node labels short. Validate the syntax in "
    "your head before emitting (no trailing semicolons, no smart quotes in labels).\n"
    "• Math expressions — wrap inline math in `$...$` (e.g. `$r_t < \\tau$`, "
    "`$\\pi_\\theta(a \\mid s)$`) so the frontend renders it via KaTeX. NEVER wrap "
    "math in backticks (`` ` ``) — that's code formatting and the math will render "
    "as a flat literal string instead of typeset math. Use display math `$$...$$` "
    "on its own line for important formulas.\n"
    "• Citations — every factual sentence must carry at least one citation in the "
    "form `[1]`, `[2]`, or `[A1]` for arXiv candidates. NEVER compress citation "
    "ranges. Write `[2] [3] [4] [5] [6]` (each marker individually) — NOT "
    "`[2]-[6]`, NOT `[2]_[6]`, NOT `[2]–[6]`. Each marker must be its own "
    "bracketed token so the frontend can render every citation as its own "
    "clickable chip.\n"
    "\n"
    "STRUCTURE — ALWAYS:\n"
    "1. Open with a 1–3 sentence answer-first TL;DR (no heading).\n"
    "2. Then layered detail in sections that match the query type from the examples above.\n"
    "3. End with a short *Next steps* or *Open questions* line when the topic warrants it.\n"
    "\n"
    "NEVER produce a wall of dense paragraphs without visual structure. NEVER chain "
    "more than two `#`-headings without intervening prose. Prefer clarity over volume."
)


def _self_check_answer(answer: str, *, papers: list[dict], arxiv_results: list[dict]) -> list[str]:
    """Deterministic self-check, delegating to ``reflection`` for the impl."""
    from app.assistant.reflection import deterministic_self_check
    return deterministic_self_check(answer, papers=papers, arxiv_results=arxiv_results)


def _render_agent_notes(agent_notes: dict | None) -> str:
    """Render scratchpad-derived notes as an ``<agent_notes>`` XML block.

    The orchestrator extracts a small dict from the ReAct scratchpad and
    passes it here. Fields we care about today:

    * ``critique`` — the latest mid-turn critique entry, if any. Tells
      the synthesizer whether the agent itself judged the evidence
      sufficient, and which specific gaps it identified.
    * ``iterations`` — how many ReAct rounds ran. ``0`` means the
      initial plan was deemed sufficient; higher counts mean the
      agent worked harder to fill gaps.
    * ``thin_evidence`` — boolean derived from
      ``len(papers) + len(arxiv_results) < 2`` AND ``iterations > 0``.
      A blunt signal that the agent tried and the evidence base is
      still weak; primary trigger for honest-uncertainty language
      in the answer.

    Returns the empty string when there is nothing useful to say, so
    the synthesizer's prompt stays compact on the fast path. The
    synthesizer-prompt instructions in :func:`_build_prompt` already
    tell the model how to interpret this block.
    """
    if not agent_notes:
        return ""
    parts: list[str] = []
    critique = (agent_notes or {}).get("critique") or {}
    iters = int((agent_notes or {}).get("iterations") or 0)
    thin = bool((agent_notes or {}).get("thin_evidence"))
    # Evidence-expansion failure signals — set by the ReAct loop when
    # tool dispatches errored / returned no new papers. We want the
    # synthesizer to caveat the answer instead of polishing past the
    # fact that the agent tried to expand evidence and couldn't.
    tool_failures = int((agent_notes or {}).get("tool_failures") or 0)
    successful_retrievals = int(
        (agent_notes or {}).get("successful_retrievals") or 0
    )
    paper_ledger_size = int((agent_notes or {}).get("paper_ledger_size") or 0)
    evidence_expansion_failed = (
        tool_failures >= 2 and successful_retrievals == 0 and iters > 0
    )

    # Genie status — surfaces the actual tool outcome so the answer
    # narrates honestly. Without this directive the synthesizer would
    # describe a still-running synthesis as if it had completed,
    # because the only signal it sees is "genie_session_id exists".
    genie_status = (agent_notes or {}).get("genie_status") or {}
    if isinstance(genie_status, dict) and genie_status.get("status"):
        st = str(genie_status.get("status") or "").lower()
        title = (genie_status.get("capsule_title") or "").strip()
        if st in {"done"}:
            if title:
                parts.append(
                    f"- GENIE STATUS: done — capsule '{title}' is ready. The user "
                    "can open it from the Genie tab; you may reference its "
                    "hypothesis in the answer."
                )
            else:
                parts.append(
                    "- GENIE STATUS: done — capsule is ready on the Genie tab."
                )
        elif st in {"queued", "running", "timeout"}:
            parts.append(
                f"- GENIE STATUS: {st} — the idea is STILL being synthesized in "
                "the background. DO NOT describe it as completed. Tell the "
                "user it's in progress and will appear on the Genie tab when "
                "ready; link there rather than narrating a finished hypothesis."
            )
        elif st in {"failed", "done_empty"}:
            parts.append(
                f"- GENIE STATUS: {st} — the synthesis did NOT produce a "
                "publishable hypothesis. Acknowledge this honestly in the "
                "answer; do not pretend a capsule was created."
            )
        elif st == "cancelled":
            parts.append(
                "- GENIE STATUS: cancelled — the user (or the orchestrator) "
                "stopped the synthesis. Do not narrate a result."
            )

    if iters > 0:
        parts.append(f"- The agent ran {iters} adaptive iteration(s) after the initial plan.")
    if tool_failures > 0:
        parts.append(
            f"- The agent attempted {tool_failures} tool call(s) that FAILED "
            f"(validation errors / runtime errors / banned-after-repeat-failure). "
            f"Successful retrievals during the loop: {successful_retrievals}. "
            f"Paper ledger size: {paper_ledger_size}."
        )
    if critique:
        v = critique.get("verdict")
        g = critique.get("groundedness")
        c = critique.get("completeness")
        issues = critique.get("issues") or []
        if v or g is not None or c is not None:
            scores = []
            if g is not None:
                scores.append(f"groundedness={g:.2f}")
            if c is not None:
                scores.append(f"completeness={c:.2f}")
            head = f"- The agent self-critiqued the evidence ({', '.join(scores) or 'no scores'})."
            if v:
                head += f" Verdict: {v}."
            parts.append(head)
        if issues:
            parts.append("- Gaps the agent identified:")
            for issue in issues[:4]:
                parts.append(f"    • {str(issue)[:240]}")
    if thin:
        parts.append(
            "- Evidence base is THIN. Be honest about uncertainty — say what is and "
            "isn't supported, and recommend follow-up retrieval rather than over-claim."
        )
    if evidence_expansion_failed:
        parts.append(
            "- EVIDENCE EXPANSION FAILED: the agent tried to expand evidence "
            "mid-turn but multiple tools errored and no new papers landed. "
            "Treat the existing evidence as the WHOLE evidence base — do not "
            "imply you investigated angles you couldn't actually retrieve. "
            "Label speculative parts of the answer as speculative."
        )
    # Retrieval observability — surface aggregate quality so the answer
    # can honestly caveat thin / rerank-rescued retrievals instead of
    # presenting them with full confidence.
    retrieval = (agent_notes or {}).get("retrieval") or {}
    if retrieval:
        thin = int(retrieval.get("thin_calls") or 0)
        rerank_heavy = int(retrieval.get("rerank_heavy_calls") or 0)
        weakest = str(retrieval.get("weakest") or "").strip()
        if thin or rerank_heavy or retrieval.get("thin_evidence"):
            parts.append(
                f"- Retrieval-quality flags: "
                f"thin_calls={thin}, rerank_heavy_calls={rerank_heavy}. "
                + (f"Weakest call: {weakest}. " if weakest else "")
                + "Caveat any claim whose support comes from a thin call; "
                "do not present rerank-rescued retrievals as if they were "
                "strong first-stage matches."
            )
    # Contradictions worth surfacing to the synth (already filtered to
    # confidence ≥ 0.6 in ``_distill_agent_notes``).
    contras = (agent_notes or {}).get("contradictions") or []
    if contras:
        parts.append(
            "- CONTRADICTIONS detected across retrieved evidence — call them "
            "out in the answer rather than picking a side silently. RESOLUTION "
            "RULES (the user explicitly asks for this):\n"
            "    • If a contradiction was investigated and the evidence now "
            "favours one side, state the resolved position and briefly note "
            "what was contested.\n"
            "    • If a contradiction is still UN-investigated, you MUST "
            "DOWNGRADE the affected conclusion — present it as contested / "
            "provisional (hedge the wording, e.g. \"evidence is mixed\"), never "
            "as a settled fact. Do not silently pick the side that fits the "
            "narrative."
        )
        for c in contras[:4]:
            span = str(c.get("span") or "").strip()
            srcs = ", ".join(map(str, c.get("sources") or []))[:120]
            flag = " (addressed by counter-search)" if c.get("addressed") else " (UN-investigated — downgrade the affected claim)"
            parts.append(f"    • {span[:220]}{flag} [{srcs}]")
    # Strong-claim ledger — full-paper verification verdicts. The
    # synthesizer reads this to distinguish "verified against paper
    # body" claims (safe to quote firmly) from "provisional / abstract-
    # only" claims (must be labelled). This block is the structural
    # anchor for the user's hard requirement: RA must not lean on
    # abstracts for strong claims without flagging them.
    claims = (agent_notes or {}).get("claim_ledger") or {}
    if isinstance(claims, dict) and int(claims.get("total") or 0) > 0:
        v = int(claims.get("verified_count") or 0)
        c = int(claims.get("contradicted_count") or 0)
        u = int(claims.get("unverifiable_count") or 0)
        p = int(claims.get("provisional_count") or 0)
        tiers = claims.get("by_evidence_tier") or {}
        exp = int(tiers.get("experiment-verified") or 0)
        meth = int(tiers.get("method-verified") or 0)
        abs_only = int(tiers.get("abstract-only") or 0)
        unver = int(tiers.get("unverified") or 0)
        parts.append(
            f"- STRONG-CLAIM LEDGER: {claims['total']} strong claim(s) tracked — "
            f"{v} verified against the paper body, {c} contradicted, "
            f"{u} unverifiable (full-paper check unavailable), {p} still provisional.\n"
            f"    Evidence tiers: experiment-verified={exp}, method-verified={meth}, "
            f"abstract-only={abs_only}, unverified={unver}.\n"
            "    EVIDENCE-QUALITY LABELLING RULES (load-bearing — the user explicitly asks for this):\n"
            "    • An ``experiment-verified`` claim may be stated firmly. "
            "Optionally label it inline as \"(experiment-verified)\" when the precision matters.\n"
            "    • A ``method-verified`` claim should be quoted with a light hedge — "
            "say \"according to the paper's method section\" or label it \"(method-verified)\".\n"
            "    • An ``abstract-only`` claim MUST be explicitly labelled \"(abstract-only)\" — "
            "do not present it as if the full paper confirmed it.\n"
            "    • An ``unverified`` claim MUST be labelled \"(unverified)\" or \"(provisional)\" — "
            "the answer's tone around it should be tentative, not declarative."
        )
        if claims.get("contradicted"):
            parts.append("    Contradicted strong claims (DO NOT repeat without flagging):")
            for item in claims["contradicted"][:4]:
                parts.append(
                    f"      • paper={str(item.get('paper_id',''))[:12]} "
                    f"tier={item.get('evidence_tier','abstract-only')} "
                    f"claim={str(item.get('span',''))[:200]!r}"
                )
        if claims.get("provisional"):
            parts.append("    Provisional / abstract-only claims (label clearly in the answer):")
            for item in claims["provisional"][:4]:
                parts.append(
                    f"      • paper={str(item.get('paper_id',''))[:12]} "
                    f"src={item.get('source','?')} "
                    f"tier={item.get('evidence_tier','abstract-only')} "
                    f"claim={str(item.get('span',''))[:200]!r}"
                )
        if claims.get("verified"):
            parts.append("    Verified strong claims (safe to quote firmly):")
            for item in claims["verified"][:4]:
                parts.append(
                    f"      • paper={str(item.get('paper_id',''))[:12]} "
                    f"tier={item.get('evidence_tier','abstract-only')} "
                    f"claim={str(item.get('span',''))[:200]!r}"
                )

    # Investigation plan — the model's own mid-loop todo list. We
    # surface OPEN + STUCK items so the answer honestly acknowledges
    # unfinished investigation rather than pretending the loop
    # closed every sub-question.
    plan = (agent_notes or {}).get("investigation_plan") or {}
    if isinstance(plan, dict) and int(plan.get("total") or 0) > 0:
        open_items = list(plan.get("open") or [])
        stuck_items = list(plan.get("stuck_in_progress") or [])
        completed_items = list(plan.get("completed") or [])
        if open_items or stuck_items:
            parts.append(
                "- INVESTIGATION PLAN had unfinished work at finalize — "
                "the answer must acknowledge these gaps honestly rather "
                "than imply the question was fully resolved:"
            )
            for item in (stuck_items + open_items)[:5]:
                parts.append(f"    • {str(item)[:220]} (NOT resolved)")
            if completed_items:
                parts.append(
                    f"    (Completed in-loop: {len(completed_items)} item(s).)"
                )
    if not parts:
        return ""
    return "<agent_notes>\n" + "\n".join(parts) + "\n</agent_notes>"


def _detect_output_quality_issue(answer: str) -> str | None:
    """Return a short reason string when ``answer`` looks broken,
    else ``None`` when the answer is fit to ship.

    The user's spec is explicit: "RA must never output empty or
    corrupted content." We check for the failure modes we've actually
    seen in the wild:

    * Empty / whitespace-only output.
    * Suspiciously short output relative to a research turn (under
      ~80 chars including markdown formatting is almost always a
      truncated generation, not a real answer).
    * Truncated mid-token: ends with an unmatched code-fence,
      unbalanced parenthesis run, a dangling ``[`` citation marker,
      or an unmatched ``$`` LaTeX block.
    * Template-placeholder leakage: ``{{var}}`` / ``${var}`` /
      ``<TODO>`` markers that escaped the strip pass.
    * Provider error markers the adapter occasionally bubbles up
      (``[ERROR]``, ``[BLOCKED]``, ``RATE_LIMIT``).

    Returns the reason on failure so the caller can log it; returns
    ``None`` on the happy path so the safeguard is a cheap no-op for
    well-formed answers.
    """
    if not isinstance(answer, str):
        return "answer is not a string"
    stripped = answer.strip()
    if not stripped:
        return "empty answer"
    # Length floor with a "sentence-completed" carve-out. The
    # original 80-char floor false-positived on legitimate short
    # replies — a complete one-sentence answer like
    # ``"Yes. The transformer paper introduced multi-head attention,
    #    not the original attention mechanism."`` is fine but would
    # have tripped the 80-char check. The refined rule:
    #
    #   * Under 24 chars → almost certainly truncated.
    #   * 24..80 chars AND ends without sentence-ending punctuation
    #     → likely truncated mid-sentence. We let the trailing-
    #     connective check downstream catch the rest of these,
    #     but here we cover the case where the model emits a
    #     fragment with no terminator at all.
    #   * 24..80 chars ending with ``.!?`` (or close-quote/paren
    #     variants) is treated as a legitimate short reply.
    SENT_END = ".!?\"')]`"
    if len(stripped) < 24:
        return f"answer too short ({len(stripped)} chars) — likely truncated"
    if len(stripped) < 80 and stripped[-1] not in SENT_END:
        return (
            f"answer short ({len(stripped)} chars) and lacks sentence-ending "
            "punctuation — likely truncated"
        )
    # Template-placeholder leak.
    import re as _re_q
    if _re_q.search(r"\{\{\s*[A-Za-z_][\w.\-]*\s*\}\}|\$\{\s*[A-Za-z_][\w.\-]*\s*\}", stripped):
        return "answer contains unresolved template placeholder"
    if _re_q.search(r"<\s*(?:TODO|FIXME|PLACEHOLDER|TBD)\b", stripped, _re_q.IGNORECASE):
        return "answer contains TODO/FIXME placeholder marker"
    # Provider error markers — the adapter normally raises, but some
    # paths surface a string instead. Catch the obvious cases.
    low = stripped.lower()
    if low.startswith(("[error]", "[blocked]", "[rate_limit]", "error:", "blocked:")):
        return "answer starts with a provider error marker"
    # Unbalanced code fence — ``len(matches) % 2 != 0`` means the
    # final block was opened but never closed.
    if stripped.count("```") % 2 != 0:
        return "answer ends mid code-block (unbalanced ```)"
    # Unmatched single-line LaTeX — ``$...$`` count must be even.
    # We deliberately don't check ``$$...$$`` because that's an
    # acceptable display-math block and balanced rendering is
    # frontend-tolerant.
    if stripped.count("$") % 2 != 0 and "$$" not in stripped:
        return "answer ends mid LaTeX expression (unmatched $)"
    # Dangling citation marker — ``[`` at end without closing ``]``.
    # We only check the LAST 80 chars so a long answer with an
    # internal ``[`` in code (e.g. Python list literal) doesn't
    # false-positive.
    tail = stripped[-80:]
    open_brackets = tail.count("[")
    close_brackets = tail.count("]")
    if open_brackets > close_brackets and stripped.endswith(("[", "[A", "[A1")):
        return "answer ends mid citation marker"
    # Ends with a hanging conjunction / preposition / comma — a
    # strong truncation tell. We require the LAST WORD to be one of
    # these AND the answer not end with a period/question/exclamation,
    # so a sentence that closes correctly even with a trailing
    # connective doesn't trip the check.
    last_char = stripped[-1]
    if last_char not in ".!?\"')]`":
        last_word = stripped.rsplit(None, 1)[-1].lower() if stripped.split() else ""
        if last_word in {
            "and", "but", "or", "the", "a", "an", "of", "to", "in", "on", "for",
            "with", "as", "by", "from", "that", "which", "while", "where", "because",
        }:
            return f"answer ends mid-sentence (trailing connective {last_word!r})"
    return None


def _has_low_grounding_signal(
    *,
    answer: str,
    papers: list[dict],
    arxiv_results: list[dict],
    agent_notes: dict | None,
    output: dict | None,
) -> bool:
    """Detect when the evidence base behind this answer was meaningfully
    weak, so we can inform the user.

    This is NOT an abstention gate — RA always answers the user's
    question to the best of its ability. The detector exists so we
    can *append* a short footer naming the specific signals (thin
    retrieval, low critique score, unverified citations, expansion
    failure). The polished answer is unchanged; the footer is a
    note, not a warning.

    Fires when ANY of:
      * Critique groundedness < 0.4 AND total evidence ≤ 3 papers,
      * Provenance verifier reports < 40% claims supported with at
        least 3 flagged citations,
      * ReAct flagged ``evidence_expansion_failed`` (tool retries
        burned, zero new retrievals),
      * Retrieval observability reports two or more thin calls.

    Conservative on purpose — a single weak signal triggers the
    footer; a clean turn passes through silently.
    """
    notes = agent_notes or {}
    n_evidence = len(papers or []) + len(arxiv_results or [])
    crit = (notes.get("critique") or {}) if isinstance(notes, dict) else {}
    g = float(crit.get("groundedness") or 1.0) if isinstance(crit, dict) else 1.0
    retrieval = (notes.get("retrieval") or {}) if isinstance(notes, dict) else {}
    thin_calls = int((retrieval or {}).get("thin_calls") or 0)
    expansion_failed = bool((notes or {}).get("evidence_expansion_failed")) or (
        int((notes or {}).get("tool_failures") or 0) >= 2
        and int((notes or {}).get("successful_retrievals") or 0) == 0
    )
    prov = (output or {}).get("provenance") if isinstance(output, dict) else None
    prov_low = False
    if isinstance(prov, dict):
        total = int(prov.get("total") or 0)
        supported = int(prov.get("supported") or 0)
        flagged = len(prov.get("flagged") or [])
        if total >= 3 and supported / max(1, total) < 0.40 and flagged >= 3:
            prov_low = True

    signals = sum([
        g < 0.4 and n_evidence <= 3,
        thin_calls >= 2,
        expansion_failed,
        prov_low,
    ])
    return signals >= 1


def _low_grounding_notice(
    *,
    papers: list[dict],
    arxiv_results: list[dict],
    agent_notes: dict | None,
    output: dict | None,
) -> str:
    """Compose a short informational footer naming the specific
    signals that flagged this turn as low-grounding.

    Tone: informational, not warning. The footer goes at the END of
    the answer — the user has already read RA's best-effort response.
    The footer just gives them ground truth about how thin the
    foundation actually was, so they can decide whether the answer
    is solid enough for their use or whether a follow-up would help.
    """
    reasons: list[str] = []
    notes = agent_notes or {}
    n_evidence = len(papers or []) + len(arxiv_results or [])
    if n_evidence <= 3:
        reasons.append(f"only {n_evidence} grounded paper(s) reached synthesis")
    crit = notes.get("critique") if isinstance(notes, dict) else None
    if isinstance(crit, dict) and float(crit.get("groundedness") or 1.0) < 0.4:
        reasons.append(f"the agent's own groundedness score was {crit.get('groundedness'):.2f}")
    retrieval = notes.get("retrieval") if isinstance(notes, dict) else None
    if isinstance(retrieval, dict) and int(retrieval.get("thin_calls") or 0) >= 2:
        reasons.append(f"{int(retrieval['thin_calls'])} retrieval calls returned thin coverage")
    if int((notes or {}).get("tool_failures") or 0) >= 2 and int((notes or {}).get("successful_retrievals") or 0) == 0:
        reasons.append("evidence-expansion tools errored without recovery")
    prov = (output or {}).get("provenance") if isinstance(output, dict) else None
    if isinstance(prov, dict):
        total = int(prov.get("total") or 0)
        supported = int(prov.get("supported") or 0)
        if total >= 3 and supported / max(1, total) < 0.40:
            reasons.append(
                f"only {supported}/{total} citations could be verified against the cited paper text"
            )
    if not reasons:
        return ""
    return (
        "\n---\n"
        "*Heads-up on the evidence base behind this answer: "
        + "; ".join(reasons)
        + ". The response above is the best read of what was retrievable; "
        "a more specific follow-up query or a broader corpus would likely "
        "strengthen it.*"
    )


def _strip_unresolvable_citations(
    answer: str,
    papers: list[dict],
    arxiv_results: list[dict],
) -> str:
    """Remove ``[N]`` / ``[A N]`` markers whose index doesn't resolve.

    Deterministic safety net that runs after every other repair pass.
    The model is told to cite within the available context; the
    repair LLM is asked to fix bad citations; but neither is
    guaranteed. This pass enforces the invariant: ``[N]`` only
    survives in the answer when ``N`` is a real index into
    ``papers``, and ``[A N]`` only when ``N`` is a real index into
    ``arxiv_results``.

    Compound forms are simplified, not nuked:

        Input answer  : "The model achieves SOTA [2-5]. Earlier
                         work used a simpler baseline [6]."
        Available    : 3 papers
        Output answer: "The model achieves SOTA [2,3]. Earlier
                        work used a simpler baseline."

    A marker that contains *no* resolvable indices is dropped along
    with any trailing whitespace and orphan punctuation so the
    sentence reads cleanly.
    """
    if not answer:
        return answer

    n_papers = len(papers or [])
    n_arxiv = len(arxiv_results or [])

    # ``[A*]`` markers must point at an arxiv_results entry that the
    # frontend can actually open — either ``external_id`` (arXiv abs
    # URL) or ``source_url`` (DOI / publisher / Semantic Scholar
    # fallback). An entry with neither is functionally inert in the
    # UI (chip styled as a link but click does nothing), so we treat
    # those indices as unresolvable and strip the marker. This makes
    # the on-screen state honest: a citation chip survives only when
    # clicking it will navigate.
    resolvable_arxiv: set[int] = set()
    for idx, item in enumerate(arxiv_results or []):
        if not isinstance(item, dict):
            continue
        has_dest = bool(
            (item.get("external_id") or "").strip()
            or (item.get("source_url") or item.get("url") or "").strip()
        )
        if has_dest:
            resolvable_arxiv.add(idx + 1)  # 1-based marker index

    def _filter_indices(raw: str, ceiling: int, *, allowed: set[int] | None = None) -> list[int]:
        out: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                try:
                    a, b = (int(x) for x in part.split("-", 1))
                except ValueError:
                    continue
                if a <= b and 0 < a and b - a < 50:
                    out.extend(i for i in range(a, b + 1) if 1 <= i <= ceiling)
            else:
                try:
                    n = int(part)
                except ValueError:
                    continue
                if 1 <= n <= ceiling:
                    out.append(n)
        if allowed is not None:
            out = [i for i in out if i in allowed]
        # Preserve order, drop dups
        seen: set[int] = set()
        ordered: list[int] = []
        for n in out:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
        return ordered

    def _replace_paper(m: re.Match) -> str:
        idxs = _filter_indices(m.group(1), n_papers)
        if not idxs:
            return ""  # drop the whole marker
        # Emit one bracketed marker per index, space-separated, so the
        # frontend's single-number marker regex (``[A?\d+]``) renders
        # each citation as its own clickable chip. The earlier
        # comma-joined ``[1,2,3]`` form rendered as plain text in the
        # UI — broken citations the user couldn't click. Expanding to
        # ``[1] [2] [3]`` also matches the user's explicit spec
        # ("never collapse to a range").
        return " ".join(f"[{i}]" for i in idxs)

    def _replace_arxiv(m: re.Match) -> str:
        idxs = _filter_indices(m.group(1), n_arxiv, allowed=resolvable_arxiv)
        if not idxs:
            return ""
        return " ".join(f"[A{i}]" for i in idxs)

    # Expand adjacency ranges FIRST — the LLM emits ``[1] - [7]``
    # (two separate markers joined by a dash) more often than the
    # internal-range form ``[1-7]``. Running expansion before the
    # per-marker filter means each expanded index is then validated
    # by the same pipeline as any other inline marker — out-of-range
    # indices get clamped here, then re-validated below. The user's
    # spec is explicit: never collapse a contiguous citation range;
    # render every marker individually so each chip is clickable and
    # auditable in the citation table.
    cleaned = _expand_adjacent_marker_ranges(answer, n_papers, n_arxiv, resolvable_arxiv)
    # Paper markers — ``[A N]`` is more specific so we run it before
    # the broad ``[N]`` pattern.
    cleaned = re.sub(r"\[A\s*(\d+(?:\s*[-,]\s*\d+)*)\]", _replace_arxiv, cleaned)
    cleaned = re.sub(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]", _replace_paper, cleaned)
    # Tidy: orphan punctuation left behind when we dropped a marker
    # (``"...baseline ." → "...baseline."``, ``"...sentence  ;" → "...sentence;"``).
    # Use ``[^\S\n]+`` (horizontal whitespace only) rather than ``\s+`` so a
    # dropped marker at a line end can't pull the next line's punctuation up
    # and collapse an intended line break — the final answer must stay
    # correctly newlined for readability.
    cleaned = re.sub(r"[^\S\n]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


# Separators the LLM uses to "compress" a citation range across two
# adjacent markers. Matches a literal hyphen, en-dash, em-dash, minus
# sign, or underscore — optionally surrounded by whitespace. Does NOT
# match a comma (that's a list, not a range) or "to" (too ambiguous in
# prose).
_RANGE_SEP = r"\s*[-‐‑‒–—−_]\s*"
_CORPUS_RANGE_PATTERN = re.compile(rf"\[(\d+)\]{_RANGE_SEP}\[(\d+)\]")
_ARXIV_RANGE_PATTERN = re.compile(rf"\[A(\d+)\]{_RANGE_SEP}\[A(\d+)\]")
# A safety cap so a degenerate ``[1]-[10000]`` doesn't expand into a
# 10k-marker wall; 50 mirrors the per-marker range cap inside
# ``_filter_indices``.
_MAX_RANGE_EXPANSION = 50


def _expand_adjacent_marker_ranges(
    text: str,
    n_papers: int,
    n_arxiv: int,
    resolvable_arxiv: set[int],
) -> str:
    """Expand adjacent-marker range syntax into individual markers.

    Turns ``"[2]-[6]"`` and ``"[2] _ [6]"`` and ``"[2]–[6]"`` into
    ``"[2] [3] [4] [5] [6]"`` (corpus) and ``"[A2]-[A6]"`` into
    ``"[A2] [A3] [A4] [A5] [A6]"`` (arXiv). Reversed ranges (``[6]-[2]``)
    or excessive spans (``[1]-[10000]``) collapse safely back to the
    original text — the goal is precision, not aggressive rewrites.

    The expansion runs iteratively so chains like ``[1]-[3]-[5]`` resolve
    in two passes (``[1]-[3]`` first, then the result merges with
    ``-[5]``). Hard-capped at three passes so a pathological input can
    never blow the synthesizer.
    """
    if not text:
        return text

    def _expand_corpus(match: re.Match) -> str:
        a = int(match.group(1))
        b = int(match.group(2))
        expanded = _expand_range(a, b, n_papers, prefix="")
        # Invalid range (reversed, oversized, both below 1) → keep
        # the original text intact rather than silently deleting it.
        # The user can re-read and decide; we never want this helper
        # to make the answer worse.
        return expanded or match.group(0)

    def _expand_arxiv(match: re.Match) -> str:
        a = int(match.group(1))
        b = int(match.group(2))
        expanded = _expand_range(a, b, n_arxiv, prefix="A")
        if not expanded:
            return match.group(0)
        # Drop indices the frontend can't actually link — same gate the
        # per-marker filter uses for arXiv rows.
        if resolvable_arxiv:
            kept = [
                tok for tok in expanded.split(" ")
                if int(tok[2:-1]) in resolvable_arxiv  # ``[A12]`` → 12
            ]
            if not kept:
                # Nothing in the range is resolvable — leave the
                # original text so the strip pass below can decide
                # whether to drop or keep it.
                return match.group(0)
            return " ".join(kept)
        return expanded

    out = text
    for _ in range(3):
        prev = out
        out = _ARXIV_RANGE_PATTERN.sub(_expand_arxiv, out)
        out = _CORPUS_RANGE_PATTERN.sub(_expand_corpus, out)
        if out == prev:
            break
    return out


def _expand_range(a: int, b: int, ceiling: int, *, prefix: str) -> str:
    """Return the space-joined marker sequence for [a..b], or the
    empty string when the range is invalid. The caller decides whether
    to substitute or leave the original text intact.
    """
    if a < 1 or b < 1:
        return ""
    if a > b:
        return ""
    if b - a + 1 > _MAX_RANGE_EXPANSION:
        return ""
    if ceiling > 0:
        b = min(b, ceiling)
        if a > b:
            return ""
    return " ".join(f"[{prefix}{i}]" for i in range(a, b + 1))


async def synthesize_answer(
    *,
    query: str,
    papers: list[dict],
    arxiv_results: list[dict],
    imported_count: int,
    graph_result: dict | None,
    genie_session_id: str | None,
    orientation: str,
    expertise: str,
    complexity: str = "medium",
    actions: list[str],
    extra_results: dict | None = None,
    memory: dict | None = None,
    intent_hint: str = "",
    shape_hint: str = "",
    namespace_key: str = "",
    namespace_keys: list[str] | None = None,
    user_id: Any | None = None,
    skip_reflection: bool = False,
    agent_notes: dict | None = None,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
    output: dict | None = None,
) -> str:
    """Return a grounded research-workspace answer or a deterministic fallback.

    ``output`` is an optional dict the caller passes in; on return it
    carries side-channel reports the synthesizer produced — today
    ``provenance`` (claim-level verification) and ``drift`` (repair-
    pass drift). The orchestrator forwards both into ``agent_notes``
    on the next turn so the UI can render them.
    """
    context = _build_paper_context(papers, arxiv_results)
    extra_context = _build_extra_context(extra_results or {})
    # Conditional memory injection — only when memory is likely to matter.
    # Forcing the memory hint into every turn risked drifting answers when
    # the stored entries were stale or off-topic; the synthesizer now only
    # sees memory when (a) the planner explicitly recalled it this turn, or
    # (b) at least one preference is stored (preferences shape voice/depth
    # and are universally relevant).
    memory_block = ""
    if memory:
        mem_d = memory if isinstance(memory, dict) else {}
        results_dict = extra_results or {}
        explicit_recall = "memory_recall" in results_dict
        def _has_pref(d: dict) -> bool:
            for v in (d or {}).values():
                if isinstance(v, dict) and v.get("type") == "preference":
                    return True
            return False
        any_pref = (
            _has_pref(mem_d.get("short") or {})
            or _has_pref(mem_d.get("medium") or {})
            or _has_pref(mem_d.get("long") or {})
        )
        if explicit_recall or any_pref:
            memory_block = _build_memory_block(mem_d)
    if memory_block:
        extra_context = (memory_block + "\n\n" + extra_context).strip() if extra_context else memory_block
    # Splice the inferred intent + response shape as soft advisory blocks.
    # Both are advisory — the synthesizer is free to deviate when the
    # content benefits.
    if intent_hint:
        extra_context = (intent_hint + "\n\n" + extra_context).strip() if extra_context else intent_hint
    if shape_hint:
        extra_context = (shape_hint + "\n\n" + extra_context).strip() if extra_context else shape_hint
    # ── Agent notes from the ReAct loop ──────────────────────────────
    # When the mid-turn ReAct loop ran a self-critique or noticed thin
    # evidence, surface those findings to the synthesizer so the
    # answer's tone reflects the agent's own assessment instead of
    # confabulating confidence. Block is intentionally short — the
    # synthesizer needs a hint, not a transcript.
    notes_block = _render_agent_notes(agent_notes)
    if notes_block:
        extra_context = (notes_block + "\n\n" + extra_context).strip() if extra_context else notes_block
    fallback = _fallback_answer(query, papers, imported_count, graph_result, genie_session_id)

    grounded = bool(papers)
    # Pure-reasoning mode: no tool evidence — answer from general knowledge.
    # Still call the LLM (not the deterministic fallback) so the user gets a
    # real answer for definitional, conversational, or simple knowledge queries.
    pure_reasoning = not context and not extra_context

    try:
        from app.adapters.llm import get_llm_adapter

        llm = get_llm_adapter()
        prompt = _build_prompt(
            query=query,
            context=context,
            extra_context=extra_context,
            actions=actions,
            imported_count=imported_count,
            graph_result=graph_result,
            genie_session_id=genie_session_id,
            orientation=orientation,
            expertise=expertise,
            grounded=grounded,
            pure_reasoning=pure_reasoning,
        )
        # Pure-reasoning: quality model is faster and perfectly adequate.
        # Grounded evidence: use reasoning model for deep synthesis.
        messages = [{"role": "user", "content": prompt}]
        if pure_reasoning:
            if on_delta:
                chunks: list[str] = []
                async for chunk in llm.stream(messages, llm.quality_model):
                    chunks.append(chunk)
                    await on_delta(chunk)
                return "".join(chunks).strip() or fallback
            res = await llm.complete(messages, llm.quality_model, max_tokens=None, temperature=0.3)
        else:
            effort_map = {"simple": "low", "medium": "medium", "complex": "high"}
            r_effort = effort_map.get(complexity, "medium")
            if on_delta:
                chunks = []
                async for chunk in llm.stream(
                    messages, llm.reasoning_model, reasoning_effort=r_effort
                ):
                    chunks.append(chunk)
                    await on_delta(chunk)
                # Streaming path can't safely re-stream — return what we have.
                return "".join(chunks).strip() or fallback
            res = await llm.complete(
                messages, llm.reasoning_model, max_tokens=None,
                temperature=None, reasoning_effort=r_effort,
            )
        answer = (res.text or "").strip() or fallback

        # Self-reflection / red-team / convergence pipeline.
        #
        # Layer 1: deterministic checks (no LLM) — citation indices,
        # truncation. Always run.
        # Layer 2: LLM-as-judge critique — scores groundedness, completeness,
        # memory_faithfulness, clarity. Runs on substantive turns only.
        # Layer 3: red-team adversarial review — flags bias / weak evidence /
        # overclaims. Runs on complex grounded answers.
        # All three feed into the same repair preamble below.
        # Streaming is skipped because we can't safely re-stream a corrected
        # answer without the UI seeing the original and the repaired version.
        from app.assistant.reflection import (
            critique_to_issue_list,
            extract_evidence_gaps,
            fetch_gap_evidence,
            has_converged,
            llm_critique,
            red_team_review,
            redteam_to_issue_list,
            render_gap_evidence,
        )

        evidence_for_judge = (context + "\n\n" + extra_context)
        issues = _self_check_answer(answer, papers=papers, arxiv_results=arxiv_results)
        # Skip every reflection layer when the caller flagged this as a
        # trivial / single-cycle turn that doesn't earn the extra latency.
        run_reflection = (not skip_reflection)
        if run_reflection and not issues and (grounded or complexity == "complex"):
            try:
                critique = await llm_critique(
                    query=query,
                    answer=answer,
                    evidence_excerpt=evidence_for_judge,
                    memory_excerpt=memory_block,
                )
                if critique and critique.get("should_repair"):
                    issues = critique_to_issue_list(critique)
                    log.info(
                        "llm critique flagged repair (g=%.2f c=%.2f mf=%.2f)",
                        float(critique.get("groundedness") or 0),
                        float(critique.get("completeness") or 0),
                        float(critique.get("memory_faithfulness") or 0),
                    )
            except Exception as exc:
                log.debug("llm critique skipped: %s", exc)

        # Red-team only when grounded AND complex — burning a third
        # cheap-model pass on simple lookups is not worth the latency.
        if run_reflection and grounded and complexity == "complex":
            try:
                redteam = await red_team_review(
                    query=query,
                    answer=answer,
                    evidence_excerpt=evidence_for_judge,
                )
                if redteam and (redteam.get("severity") in {"medium", "high"}):
                    rt_issues = redteam_to_issue_list(redteam)
                    if rt_issues:
                        issues = (issues or []) + rt_issues
                        log.info("red-team flagged %d issue(s) at severity=%s",
                                 len(rt_issues), redteam.get("severity"))
            except Exception as exc:
                log.debug("red-team review skipped: %s", exc)

        if run_reflection and issues:
            log.info("synth self-check found issues — repairing: %s", "; ".join(issues))
            try:
                # ── Gap-driven re-querying (deep cycle only) ─────────────
                # When the issues describe evidential gaps and we have the
                # tooling to chase them, fire 1–2 targeted retrievals first,
                # splice the new material into the prompt, and let the
                # model write a NEW answer with stronger grounding before
                # the rewording-only repair runs as a fallback.
                gap_evidence: list[dict] = []
                if user_id is not None and complexity == "complex" and not pure_reasoning:
                    try:
                        gaps = await extract_evidence_gaps(
                            query=query,
                            answer=answer,
                            issues=issues,
                            evidence_excerpt=evidence_for_judge,
                        )
                        for gap in gaps:
                            block = await fetch_gap_evidence(
                                user_id=user_id,
                                namespace_key=namespace_key,
                                namespace_keys=namespace_keys or [],
                                gap=gap,
                            )
                            if block:
                                gap_evidence.append(block)
                    except Exception as exc:
                        log.debug("gap re-querying skipped: %s", exc)

                if gap_evidence:
                    log.info("synth repair: re-querying surfaced %d new evidence block(s)", len(gap_evidence))
                    fresh_block = render_gap_evidence(gap_evidence)
                    augmented_extra = (
                        (extra_context + "\n\n" if extra_context else "")
                        + "── Additional evidence from gap re-querying ──\n"
                        + fresh_block
                    )
                    augmented_prompt = _build_prompt(
                        query=query, context=context, extra_context=augmented_extra,
                        actions=actions, imported_count=imported_count,
                        graph_result=graph_result, genie_session_id=genie_session_id,
                        orientation=orientation, expertise=expertise,
                        grounded=grounded, pure_reasoning=pure_reasoning,
                    )
                    repair_msg = (
                        "Your previous draft had these issues:\n  - "
                        + "\n  - ".join(issues)
                        + "\n\nFresh evidence has been added below the original evidence "
                        "block — under '── Additional evidence from gap re-querying ──'. "
                        "Use it to write a CORRECTED answer that closes the gaps. "
                        "Cite only what exists in the merged evidence. Complete every "
                        "sentence. Do not invent claims beyond the evidence."
                    )
                    repair_messages = [
                        {"role": "user", "content": augmented_prompt},
                        {"role": "assistant", "content": answer},
                        {"role": "user", "content": repair_msg},
                    ]
                else:
                    repair_msg = (
                        "Your previous draft had these issues:\n  - "
                        + "\n  - ".join(issues)
                        + "\n\nProduce a corrected answer. Use ONLY citation indices that "
                        "actually exist in the evidence block. Complete every sentence. "
                        "Do not add new claims you can't ground in the evidence."
                    )
                    repair_messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                        {"role": "user", "content": repair_msg},
                    ]

                if pure_reasoning:
                    res2 = await llm.complete(
                        repair_messages, llm.quality_model,
                        max_tokens=None, temperature=0.2,
                    )
                else:
                    res2 = await llm.complete(
                        repair_messages, llm.reasoning_model,
                        max_tokens=None, temperature=None,
                        reasoning_effort=effort_map.get(complexity, "medium"),
                    )
                repaired = (res2.text or "").strip()
                if repaired and not _self_check_answer(
                    repaired, papers=papers, arxiv_results=arxiv_results
                ):
                    # Convergence guard: when the repaired draft is barely
                    # different from the original (Jaccard ~ identical) AND
                    # we did NOT re-query (no genuinely new evidence), the
                    # repair didn't add value — keep the original. When new
                    # evidence was added, always accept the repair so the
                    # extra citations make it through.
                    if not gap_evidence and has_converged(before=answer, after=repaired):
                        log.debug("repair converged with no material change — keeping original")
                    else:
                        # Drift detection: log any new citation markers /
                        # newly-cited claims the repair pass introduced.
                        # We accept the repair but surface the drift into
                        # ``output['drift']`` so the orchestrator can
                        # render it in agent_notes; the synth answer
                        # itself is unchanged. Reverting risks
                        # re-introducing the very issue the repair fixed.
                        try:
                            from app.assistant.repair_drift import detect_repair_drift
                            drift = detect_repair_drift(pre=answer, post=repaired)
                            if drift.has_drift and isinstance(output, dict):
                                output["drift"] = {
                                    "new_markers": drift.new_markers,
                                    "new_claims": drift.new_claims[:6],
                                    "changed_claims": drift.changed_claims[:6],
                                    "summary": drift.summary,
                                }
                        except Exception as exc:  # noqa: BLE001
                            log.debug("repair drift detector skipped: %s", exc)
                        answer = repaired
            except Exception as exc:
                log.warning("synthesizer repair pass failed: %s", exc)

        # ── Citation safety net ──────────────────────────────────────
        # Deterministic final pass. Even when the repair LLM is asked
        # to fix out-of-range citations, it sometimes leaves them in —
        # the model judges the issue resolved when the prose changed
        # but the marker survived. We strip any ``[N]`` marker that
        # cannot resolve to a paper in the synthesizer's context.
        # Resolvable markers are preserved verbatim, including
        # compound forms (``[2,3]``, ``[2-4]``) where only some inner
        # indices were out of range (we keep the in-range ones).
        try:
            answer = _strip_unresolvable_citations(answer, papers, arxiv_results)
        except Exception as exc:  # noqa: BLE001 — safety net must never raise
            log.debug("citation strip pass skipped: %s", exc)

        # ── Claim-level provenance verification ───────────────────────
        # For every surviving ``[N]`` / ``[A N]`` marker, check that the
        # cited paper's text actually overlaps with the claim it's
        # attached to. Deterministic + cheap; flagged claims land in
        # ``output['provenance']`` so the orchestrator/synthesizer can
        # caveat unverified citations honestly in the final answer.
        try:
            from app.assistant.provenance_verification import (
                escalate_unverified_with_llm,
                verify_claims,
            )
            report = verify_claims(
                answer=answer, papers=papers, arxiv_results=arxiv_results,
            )
            # LLM escalation on borderline cases. Only the
            # ``unverified`` subset gets escalated — typically a small
            # fraction of total claims, so the cost stays bounded.
            # Failures (LLM unavailable, schema mismatch, etc.) leave
            # the deterministic verdicts intact.
            if any(c.verdict == "unverified" for c in report.claims):
                try:
                    report = await escalate_unverified_with_llm(
                        report=report, papers=papers,
                        arxiv_results=arxiv_results,
                    )
                except Exception as _exc:
                    log.debug("provenance LLM escalation skipped: %s", _exc)
            if isinstance(output, dict) and report.total:
                output["provenance"] = {
                    "total": report.total,
                    "supported": report.supported,
                    "unverified": report.unverified,
                    "unsupported": report.unsupported,
                    "verified_ratio": report.verified_ratio,
                    "flagged": [
                        {
                            "marker": c.marker, "verdict": c.verdict,
                            "paper_title": c.paper_title[:120],
                            "claim": c.claim[:240],
                            "missing_salient": c.missing_salient[:4],
                            "overlap_score": c.overlap_score,
                        }
                        for c in report.claims
                        if c.verdict in ("unsupported", "unverified")
                    ][:10],
                }
        except Exception as exc:  # noqa: BLE001 — verification must never raise
            log.debug("claim verification skipped: %s", exc)

        # ── Chunk-level provenance evidence ──────────────────────────
        # Deepen the per-claim verdict from "supported by paper [3]"
        # to "supported by paper [3], chunk #4: <excerpt>". The
        # frontend renders chunks as hover-previews / click-through
        # from inline citation markers so the user can audit "did the
        # cited paper actually say this?" in one click. Bounded by
        # MAX_CLAIMS_PER_TURN; supported claims win the budget.
        try:
            from app.assistant.provenance_evidence import attach_chunk_evidence
            from app.db.session import async_session_factory

            # We need a DB session — the synthesizer's caller hasn't
            # threaded one in, so open a short-lived one here. Read-
            # only; we never commit.
            async with async_session_factory() as _evidence_db:
                chunk_links = await attach_chunk_evidence(
                    db=_evidence_db,
                    claim_verdicts=report.claims,
                    papers=papers,
                )
            if chunk_links and isinstance(output, dict):
                # Stash on the existing ``provenance`` dict if present
                # so the orchestrator persists it in one payload pass.
                prov_block = output.setdefault("provenance", {})
                prov_block["chunk_evidence"] = [link.to_dict() for link in chunk_links]
        except Exception as exc:  # noqa: BLE001 — chunk evidence is best-effort
            log.debug("chunk-level provenance skipped: %s", exc)

        # ── Low-grounding informational notice (NOT abstention) ──────
        # When the evidence base is weak (thin retrieval AND low
        # groundedness OR critique flagged unresolved issues), we
        # APPEND a short informational footer naming the specific
        # signals that triggered it. We do NOT prepend a warning,
        # we do NOT replace the polished answer, and we do NOT
        # caveat the prose itself — the user still gets RA's best
        # effort at answering the question. The footer just tells
        # them HOW thin the foundation is so they can decide whether
        # to follow up.
        try:
            if _has_low_grounding_signal(
                answer=answer, papers=papers, arxiv_results=arxiv_results,
                agent_notes=agent_notes, output=output,
            ):
                notice = _low_grounding_notice(
                    papers=papers, arxiv_results=arxiv_results,
                    agent_notes=agent_notes, output=output,
                )
                if notice and notice not in answer:
                    answer = answer.rstrip() + "\n\n" + notice
                if isinstance(output, dict):
                    output["low_grounding"] = True
        except Exception as exc:  # noqa: BLE001
            log.debug("low-grounding notice skipped: %s", exc)

        # ── Semantic-adequacy evaluator ─────────────────────────────
        # Strong-model audit of the answer for relevance,
        # groundedness, completeness, and drift. Distinct from the
        # mechanical-corruption check below — this catches answers
        # that are well-formed but don't actually address what the
        # user asked. When the evaluator flags revisable issues,
        # we run ONE re-synth pass with the evaluator's notes
        # spliced in. Bounded (one revision max) so we never loop.
        # Best-effort: a failure here ships the original answer.
        try:
            from app.assistant.final_evaluator import (
                evaluate_final_answer,
                revision_notes,
                should_revise,
            )
            eval_report = await evaluate_final_answer(
                query=query,
                answer=answer,
                papers=papers,
                arxiv_results=arxiv_results,
            )
            if eval_report and isinstance(output, dict):
                output["final_evaluation"] = eval_report
            if eval_report and should_revise(eval_report):
                notes = revision_notes(eval_report)
                log.info(
                    "final_evaluator: verdict=%s — running one revision pass",
                    eval_report.get("verdict"),
                )
                try:
                    from app.adapters.llm import get_llm_adapter
                    _llm = get_llm_adapter()
                    revise_prompt = (
                        f"Below is your DRAFT answer to the user. A reviewer flagged "
                        f"the following issues:\n\n{notes}\n\n"
                        f"ORIGINAL QUERY:\n{query}\n\n"
                        f"DRAFT ANSWER:\n{answer}\n\n"
                        "Produce a REVISED answer that addresses every flagged "
                        "issue while preserving everything the draft got right. "
                        "Keep the same citation style and the same evidence-tier "
                        "labelling. Do NOT introduce claims you can't cite. If a "
                        "flagged improvement asks for content you don't have "
                        "evidence for, acknowledge the gap explicitly rather than "
                        "fabricating."
                    )
                    revised = await _llm.complete(
                        [{"role": "user", "content": revise_prompt}],
                        _llm.reasoning_model,
                        max_tokens=None,
                        temperature=0.1,
                    )
                    revised_text = (revised.text or "").strip()
                    # Apply the same citation strip the original got
                    # so the revision's markers stay valid.
                    if revised_text:
                        try:
                            revised_text = _strip_unresolvable_citations(
                                revised_text, papers, arxiv_results,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        # Only adopt the revision when it didn't
                        # introduce a quality issue of its own.
                        rev_issue = _detect_output_quality_issue(revised_text)
                        if rev_issue is None:
                            answer = revised_text
                            if isinstance(output, dict):
                                output["revised_after_evaluation"] = True
                                # Stale provenance verdicts were keyed
                                # against the ORIGINAL answer's claim
                                # positions. Re-run the deterministic
                                # verifier on the revised text so the
                                # frontend renders correct per-marker
                                # verdicts. We skip the LLM-escalation
                                # step here to keep latency bounded —
                                # any unverified claims remain flagged
                                # exactly as the cheap path scored
                                # them. ``chunk_evidence`` is also
                                # cleared because its claim-position
                                # links no longer apply.
                                try:
                                    from app.assistant.provenance_verification import (
                                        verify_claims as _verify_claims,
                                    )
                                    rev_report = _verify_claims(
                                        answer=answer,
                                        papers=papers,
                                        arxiv_results=arxiv_results,
                                    )
                                    if rev_report.total:
                                        output["provenance"] = {
                                            "total": rev_report.total,
                                            "supported": rev_report.supported,
                                            "unverified": rev_report.unverified,
                                            "unsupported": rev_report.unsupported,
                                            "verified_ratio": rev_report.verified_ratio,
                                            "flagged": [
                                                {
                                                    "marker": c.marker, "verdict": c.verdict,
                                                    "paper_title": c.paper_title[:120],
                                                    "claim": c.claim[:240],
                                                    "missing_salient": c.missing_salient[:4],
                                                    "overlap_score": c.overlap_score,
                                                }
                                                for c in rev_report.claims
                                                if c.verdict in ("unsupported", "unverified")
                                            ][:10],
                                        }
                                    else:
                                        # No citations to verify —
                                        # clear any stale provenance.
                                        output.pop("provenance", None)
                                except Exception as _exc:  # noqa: BLE001
                                    # If re-verification fails, scrub
                                    # the stale provenance so the
                                    # frontend doesn't show wrong
                                    # verdicts.
                                    output.pop("provenance", None)
                                    log.debug(
                                        "post-revision provenance re-verify failed: %s",
                                        _exc,
                                    )
                except Exception as exc:  # noqa: BLE001 — revision is best-effort
                    log.debug("final_evaluator revision skipped: %s", exc)
        except Exception as exc:  # noqa: BLE001 — evaluator must never crash the turn
            log.debug("final_evaluator skipped: %s", exc)

        # ── Output-quality safeguard ─────────────────────────────────
        # Catch empty / truncated / corrupted answers BEFORE returning.
        # The user's spec is unambiguous: "RA must never output empty
        # or corrupted content." When the quality check trips, raise
        # so the existing fallback retry path runs — one retry with
        # the quality model. If that still produces garbage we
        # surface ``fallback`` rather than show the user a broken
        # answer.
        quality_issue = _detect_output_quality_issue(answer)
        if quality_issue is not None:
            log.warning(
                "assistant synthesis quality safeguard tripped: %s — retrying",
                quality_issue,
            )
            raise RuntimeError(f"synthesis quality safeguard: {quality_issue}")

        return answer
    except Exception as exc:
        log.warning("assistant synthesis fallback (reasoning model): %s — retrying with quality model", exc)
        try:
            from app.adapters.llm import get_llm_adapter
            llm2 = get_llm_adapter()
            prompt2 = _build_prompt(
                query=query, context=context, extra_context=extra_context,
                actions=actions, imported_count=imported_count,
                graph_result=graph_result, genie_session_id=genie_session_id,
                orientation=orientation, expertise=expertise, grounded=grounded,
                pure_reasoning=pure_reasoning,
            )
            msgs2 = [{"role": "user", "content": prompt2}]
            if on_delta:
                chunks2: list[str] = []
                async for chunk in llm2.stream(msgs2, llm2.quality_model):
                    chunks2.append(chunk)
                    await on_delta(chunk)
                retry_answer = "".join(chunks2).strip() or fallback
            else:
                res2 = await llm2.complete(msgs2, llm2.quality_model, max_tokens=None, temperature=0.2)
                retry_answer = res2.text.strip() or fallback
            # Apply the same quality safeguard to the retry — a broken
            # retry must NEVER reach the user. If the retry is also
            # corrupt, fall back to the canned ``fallback`` message
            # which is at least intelligible.
            retry_issue = _detect_output_quality_issue(retry_answer)
            if retry_issue is not None:
                log.warning(
                    "assistant synthesis retry also failed quality check: %s — "
                    "returning fallback",
                    retry_issue,
                )
                return fallback
            return retry_answer
        except Exception as exc2:
            log.warning("assistant synthesis fallback: %s", exc2)
            return fallback


def _build_memory_block(memory: dict) -> str:
    """Build a short, low-trust hint string from chat/tree/namespace memory.

    Memory can drift, get stale, or contain mis-typed planner writes — it
    should *nudge* the answer (tone, preferences, prior findings worth
    referencing) but never override grounded evidence. The block is
    deliberately short, capped, and clearly framed as advisory so the LLM
    treats it accordingly.
    """
    if not memory:
        return ""
    short = (memory.get("short") or {}) if isinstance(memory, dict) else {}
    medium = (memory.get("medium") or {}) if isinstance(memory, dict) else {}
    long_mem = (memory.get("long") or {}) if isinstance(memory, dict) else {}

    def _entry(v: object) -> tuple[str, str]:
        if isinstance(v, dict):
            return str(v.get("type") or "context"), str(v.get("value") or "")
        return "context", str(v or "")

    lines: list[str] = []
    # Surface preferences first across all tiers — they steer voice/depth and
    # are the most reliable / least likely to be stale.
    for source_label, mem_dict in (("chat", short), ("tree", medium), ("namespace", long_mem)):
        prefs = [(k, _entry(v)) for k, v in mem_dict.items() if _entry(v)[0] == "preference"]
        for k, (t, val) in prefs[:3]:
            lines.append(f"  [{source_label}/{t}] {k}: {val[:240]}")
    # Then a couple of findings/context entries as background.
    for source_label, mem_dict in (("chat", short), ("tree", medium), ("namespace", long_mem)):
        others = [(k, _entry(v)) for k, v in mem_dict.items() if _entry(v)[0] != "preference"]
        for k, (t, val) in others[:2]:
            lines.append(f"  [{source_label}/{t}] {k}: {val[:240]}")

    if not lines:
        return ""
    # Cap raw size as a final safety net.
    body = "\n".join(lines)[:1600]
    return (
        "<memory_hint trust=\"soft — may be stale, advisory only\">\n"
        + body
        + "\n</memory_hint>"
    )


def _build_extra_context(results: dict) -> str:
    """Build a combined context string from all non-paper tool outputs.

    Section labels use angle-bracket XML style so the model treats them as
    structural delimiters, not text to be quoted in the answer.
    """
    parts: list[str] = []

    # Wolfram Alpha — precise computation results
    wa = results.get("wolfram_alpha")
    if wa and wa.output.get("answer"):
        pods = wa.output.get("pods") or []
        pod_text = "\n".join(f"  {p['title']}: {p['text']}" for p in pods[:4]) if pods else ""
        parts.append(
            f"<wolfram_computation>\n"
            f"{wa.output['answer']}"
            + (f"\nPods:\n{pod_text}" if pod_text else "")
            + "\n</wolfram_computation>"
        )

    # Parse context — user-attached files, notes, URLs
    pc = results.get("parse_context")
    if pc and pc.output.get("items"):
        items = pc.output["items"]
        item_texts = []
        for item in items[:5]:
            label = item.get("label") or item.get("kind") or "Attachment"
            text = (item.get("text") or "")[:2000]
            item_texts.append(f"  <item label=\"{label}\">\n  {text}\n  </item>")
        parts.append("<active_context>\n" + "\n\n".join(item_texts) + "\n</active_context>")

    # Concept explain — teaching/explanation output
    ce = results.get("concept_explain")
    if ce and ce.output.get("explanation"):
        parts.append(f"<concept_background>\n{ce.output['explanation'][:3000]}\n</concept_background>")

    # Memory recall — prior session facts (typed: finding/preference/concept/hypothesis/paper_note/context)
    mr = results.get("memory_recall")
    if mr:
        med = mr.output.get("medium") or {}
        lng = mr.output.get("long") or {}
        brs = mr.output.get("branches") or {}
        # Support both new typed format {value, type, ts} and legacy string format
        def _fmt_entry(k: str, v: object) -> str:
            if isinstance(v, dict):
                vtype = v.get("type", "context")
                vval = v.get("value", "")
                return f"  [{vtype}] {k}: {vval}"
            return f"  {k}: {v}"
        if med or lng or brs:
            lines: list[str] = []
            if med:
                lines.append("Session memory:")
                lines.extend(_fmt_entry(k, v) for k, v in list(med.items())[:12])
            if lng:
                lines.append("Namespace memory (persists across sessions):")
                lines.extend(_fmt_entry(k, v) for k, v in list(lng.items())[:8])
            if brs:
                lines.append("Branch / nested-chat progress (parent + siblings):")
                for entry in list(brs.values())[:8]:
                    title = (entry.get("title") or "Branch").strip()
                    summary = (entry.get("summary") or "").strip()
                    if summary:
                        lines.append(f"  • {title}: {summary}")
            parts.append(f"<session_memory>\n" + "\n".join(lines) + "\n</session_memory>")

    # Frontier scan — recent papers
    fs = results.get("frontier_scan")
    if fs and fs.output.get("papers"):
        fps = fs.output["papers"][:5]
        paper_lines = "\n".join(
            f"  [{i+1}] {p.get('title')} ({p.get('year', '')}) — {p.get('tldr') or p.get('abstract', '')[:200]}"
            for i, p in enumerate(fps)
        )
        parts.append(f"<frontier_papers>\n{paper_lines}\n</frontier_papers>")

    # Bookmarks answer — user's saved papers (arXiv-sourced)
    bq = results.get("bookmarks_query")
    if bq and bq.output.get("answer"):
        parts.append(f"<bookmarks_answer>\n{bq.output['answer'][:1500]}\n</bookmarks_answer>")

    # Web search — external context (use for docs, news, repos; prefer arXiv for academic claims)
    ws = results.get("web_search")
    if ws and ws.output.get("results"):
        web_items = ws.output["results"][:5]
        web_lines = "\n".join(
            f"  {r.get('title')} — {r.get('snippet', '')[:300]} ({r.get('url', '')})"
            for r in web_items
        )
        parts.append(f"<web_search_results>\n{web_lines}\n</web_search_results>")

    # Genie deep dive — comprehensive Opus-generated technical article for a specific idea
    gdd = results.get("genie_deep_dive")
    if gdd:
        dd_status = gdd.output.get("deep_dive_status", "none")
        dd_title = gdd.output.get("title", "")
        dd_href = gdd.output.get("href", "")
        if dd_status == "done" and gdd.output.get("deep_dive_excerpt"):
            hypothesis = gdd.output.get("hypothesis", "")
            excerpt = gdd.output.get("deep_dive_excerpt", "")
            parts.append(
                f"<genie_deep_dive idea=\"{dd_title}\">\n"
                f"IMPORTANT: This is an AI-generated hypothesis and technical exploration, "
                f"NOT an experimentally validated finding. Treat as a creative research direction.\n\n"
                f"Hypothesis: {hypothesis[:400]}\n\n"
                f"Full technical deep dive:\n{excerpt}\n"
                f"</genie_deep_dive>"
            )
        elif dd_status == "generating":
            parts.append(
                f"<genie_deep_dive_status>\n"
                f"Deep dive for '{dd_title}' is generating now (~60-90 seconds). "
                f"The user can view it at: {dd_href}\n"
                f"</genie_deep_dive_status>"
            )

    # Genie ideas (research hypotheses, clearly labelled as unvalidated)
    gi = results.get("genie_read")
    if gi and gi.output.get("ideas"):
        ideas = gi.output["ideas"][:3]
        idea_lines = "\n".join(
            f"  Idea {i+1}: {idea.get('title')} — {(idea.get('hypothesis') or '')[:300]}"
            f"\n    Note: Novelty={idea.get('novelty_score',0):.0%} Feasibility={idea.get('feasibility_score',0):.0%}"
            for i, idea in enumerate(ideas)
        )
        parts.append(
            f"<genie_research_ideas>\n"
            f"IMPORTANT: These are AI-generated hypotheses that have NOT been experimentally validated.\n"
            f"Treat them as creative directions or starting points, not established facts.\n"
            f"{idea_lines}\n"
            f"</genie_research_ideas>"
        )

    # PubMed — biomedical / life sciences literature
    pm = results.get("pubmed")
    if pm and pm.output.get("papers"):
        pm_papers = pm.output["papers"][:5]
        pm_db = pm.output.get("database", "PubMed")
        pm_lines = "\n".join(
            f"  [{i+1}] {p.get('title', '')} ({p.get('year', '')}) — "
            f"PMID: {p.get('pmid', '')} — "
            f"{(p.get('abstract') or '')[:300]}"
            for i, p in enumerate(pm_papers)
        )
        parts.append(f"<pubmed_papers database=\"{pm_db}\">\n{pm_lines}\n</pubmed_papers>")

    # Unpaywall — open-access PDF availability
    uw = results.get("unpaywall")
    if uw and uw.output.get("is_oa") and uw.output.get("best_oa_url"):
        parts.append(
            f"<unpaywall doi=\"{uw.output.get('doi', '')}\">\n"
            f"  Open-access: {uw.output.get('best_oa_version', '')} — {uw.output.get('best_oa_url', '')}\n"
            f"  License: {uw.output.get('license', 'unknown')}\n"
            f"</unpaywall>"
        )

    # NVD CVE — security vulnerability data
    nvd = results.get("nvd_cve")
    if nvd and nvd.output.get("vulnerabilities"):
        vulns = nvd.output["vulnerabilities"][:5]
        vuln_lines = "\n".join(
            f"  [{i+1}] {v.get('id', '')} — CVSS: {v.get('cvss_score', 'N/A')} ({v.get('severity', '')}) — "
            f"{(v.get('description') or '')[:250]}"
            for i, v in enumerate(vulns)
        )
        total = nvd.output.get("total_results", len(vulns))
        parts.append(f"<nvd_cve total=\"{total}\">\n{vuln_lines}\n</nvd_cve>")

    # ClinicalTrials.gov — registered clinical studies
    ct = results.get("clinicaltrials")
    if ct and ct.output.get("studies"):
        studies = ct.output["studies"][:5]
        study_lines = "\n".join(
            f"  [{i+1}] {s.get('title', '')} ({s.get('nct_id', '')}) — "
            f"Phase: {s.get('phase', '?')} | Status: {s.get('status', '?')} — "
            f"Conditions: {', '.join((s.get('conditions') or [])[:3])}"
            for i, s in enumerate(studies)
        )
        total = ct.output.get("total_found", len(studies))
        parts.append(f"<clinical_trials total=\"{total}\">\n{study_lines}\n</clinical_trials>")

    # FRED — macroeconomic / financial time series
    fred = results.get("fred")
    if fred and fred.output.get("series"):
        series_list = fred.output["series"][:4]
        series_lines = []
        for s in series_list:
            obs = s.get("observations") or []
            recent_obs = obs[-3:] if obs else []
            obs_str = ", ".join(f"{o.get('date','')}: {o.get('value','')}" for o in recent_obs)
            series_lines.append(
                f"  {s.get('id','')}: {s.get('title','')} ({s.get('units','')}) — "
                f"Recent: {obs_str or 'no data'}"
            )
        parts.append(f"<fred_data>\n" + "\n".join(series_lines) + "\n</fred_data>")

    # NASA ADS — astronomy / astrophysics papers
    ads = results.get("nasa_ads")
    if ads and ads.output.get("papers"):
        ads_papers = ads.output["papers"][:5]
        ads_lines = "\n".join(
            f"  [{i+1}] {p.get('title', '')} ({p.get('year', '')}) — "
            f"bibcode: {p.get('bibcode', '')} — citations: {p.get('citation_count', 0)} — "
            f"{(p.get('abstract') or '')[:250]}"
            for i, p in enumerate(ads_papers)
        )
        total = ads.output.get("total_found", len(ads_papers))
        parts.append(f"<nasa_ads_papers total=\"{total}\">\n{ads_lines}\n</nasa_ads_papers>")

    # INSPIRE HEP — particle/high-energy physics literature
    ihep = results.get("inspire_hep")
    if ihep and ihep.output.get("papers"):
        ihep_papers = ihep.output["papers"][:5]
        ihep_lines = "\n".join(
            f"  [{i+1}] {p.get('title', '')} ({p.get('year', '')}) — "
            f"citations: {p.get('citation_count', 0)} — "
            f"{(p.get('abstract') or '')[:250]}"
            for i, p in enumerate(ihep_papers)
        )
        total = ihep.output.get("total_found", len(ihep_papers))
        parts.append(f"<inspire_hep_papers total=\"{total}\">\n{ihep_lines}\n</inspire_hep_papers>")

    # OEIS — integer sequences (math namespace)
    oeis = results.get("oeis")
    if oeis and oeis.output.get("sequences"):
        seqs = oeis.output["sequences"][:4]
        seq_lines = "\n".join(
            f"  {s.get('id','')}: {s.get('name','')} — "
            f"values: {', '.join(str(v) for v in (s.get('sample_values') or [])[:8])}"
            + (f"\n    Python: {s.get('python_code','')[:200]}" if s.get('python_code') else "")
            for s in seqs
        )
        parts.append(f"<oeis_sequences>\n{seq_lines}\n</oeis_sequences>")

    # GitHub search — code repositories
    gh = results.get("github_search")
    if gh and gh.output.get("repositories"):
        repos = gh.output["repositories"][:5]
        repo_lines = "\n".join(
            f"  {r.get('full_name', r.get('name',''))} ★{r.get('stars',0)} "
            f"[{r.get('language','?')}] — {r.get('description','')[:200]} — {r.get('url','')}"
            for r in repos
        )
        total = gh.output.get("total_count", len(repos))
        parts.append(f"<github_repos total=\"{total}\">\n{repo_lines}\n</github_repos>")

    # HuggingFace — models and datasets
    hf = results.get("huggingface_search")
    if hf and hf.output.get("results"):
        hf_items = hf.output["results"][:5]
        hf_type = hf.output.get("search_type", "models")
        hf_lines = "\n".join(
            f"  {item.get('id', item.get('name',''))} — "
            f"downloads: {item.get('downloads',0):,} — "
            f"likes: {item.get('likes',0):,} — "
            f"tags: {', '.join((item.get('tags') or [])[:4])}"
            for item in hf_items
        )
        parts.append(f"<huggingface_{hf_type}>\n{hf_lines}\n</huggingface_{hf_type}>")

    # Papers with Code — benchmarks, SoTA results
    pwc = results.get("papers_with_code")
    if pwc and pwc.output.get("results"):
        pwc_items = pwc.output["results"][:5]
        pwc_type = pwc.output.get("search_type", "papers")
        pwc_lines = []
        for item in pwc_items:
            title = item.get("title") or item.get("name") or ""
            benchmarks = item.get("benchmarks") or []
            bm_str = " | ".join(
                f"{b.get('dataset','')}/{b.get('task','')}: {b.get('sota_result','')}"
                for b in benchmarks[:2]
            ) if benchmarks else ""
            pwc_lines.append(f"  {title}" + (f" — SoTA: {bm_str}" if bm_str else ""))
        parts.append(f"<papers_with_code type=\"{pwc_type}\">\n" + "\n".join(pwc_lines) + "\n</papers_with_code>")

    # Wikipedia — authoritative background knowledge
    wp = results.get("wikipedia")
    if wp and wp.output.get("found") and wp.output.get("summary"):
        parts.append(
            f"<wikipedia_background title=\"{wp.output.get('title', '')}\">\n"
            f"{wp.output['summary'][:2000]}\n"
            f"</wikipedia_background>"
        )

    # CrossRef — verified bibliographic metadata
    cr = results.get("crossref")
    if cr and cr.output.get("works"):
        cr_works = cr.output["works"][:4]
        cr_lines = "\n".join(
            f"  [{i+1}] {w.get('title', '')} — {w.get('journal', '')} ({w.get('year', '')}) "
            f"DOI: {w.get('doi', '')} — {(w.get('abstract') or '')[:200]}"
            for i, w in enumerate(cr_works)
        )
        parts.append(f"<crossref_metadata>\n{cr_lines}\n</crossref_metadata>")

    # Research trends — publication growth data
    rt = results.get("research_trends")
    if rt and rt.output.get("yearly_counts"):
        trend = rt.output.get("trend", "unknown")
        peak_year = rt.output.get("peak_year")
        topic = rt.output.get("topic", "")
        yearly = rt.output.get("yearly_counts", [])
        trend_lines = "  " + ", ".join(f"{y['year']}: {y['count']:,}" for y in yearly[-8:])
        venues = rt.output.get("top_venues", [])
        venues_str = ", ".join(v["name"] for v in venues[:4]) if venues else "N/A"
        parts.append(
            f"<research_trends topic=\"{topic}\">\n"
            f"  Trend: {trend} (peak year: {peak_year})\n"
            f"  Publication counts: {trend_lines}\n"
            f"  Top venues (recent): {venues_str}\n"
            f"</research_trends>"
        )

    # Author network — key researchers in the field
    an = results.get("author_network")
    if an and an.output.get("authors"):
        authors = an.output["authors"][:5]
        author_lines = "\n".join(
            f"  {a.get('name', '')} — h-index: {a.get('h_index', '?')}, "
            f"{a.get('citation_count', 0):,} citations, {a.get('paper_count', 0)} papers"
            + (f" [{', '.join(a.get('affiliations', [])[:2])}]" if a.get('affiliations') else "")
            for a in authors
        )
        parts.append(f"<key_researchers>\n{author_lines}\n</key_researchers>")

    # Paper Q&A — targeted answer about a specific paper's content
    pqa = results.get("paper_qa")
    if pqa and pqa.output.get("found") and pqa.output.get("answer"):
        paper_title = pqa.output.get("paper_title", "")
        parts.append(
            f"<paper_qa paper=\"{paper_title}\">\n"
            f"{pqa.output['answer'][:3000]}\n"
            f"</paper_qa>"
        )

    # Citation finder — ranked papers for citing a claim
    cf = results.get("citation_finder")
    if cf and cf.output.get("papers"):
        claim = cf.output.get("claim", "")
        cf_papers = cf.output["papers"][:6]
        cf_lines = "\n".join(
            f"  [{i+1}] {p.get('title', '')} ({p.get('year', '')}) — "
            f"{p.get('relevance_note') or (p.get('abstract') or '')[:200]}"
            for i, p in enumerate(cf_papers)
        )
        parts.append(
            f"<citation_candidates claim=\"{claim[:100]}\">\n"
            f"{cf_lines}\n"
            f"</citation_candidates>"
        )

    # LaTeX parse — document structure and equations
    lp = results.get("latex_parse")
    if lp and (lp.output.get("title") or lp.output.get("sections") or lp.output.get("equations")):
        lp_title = lp.output.get("title", "")
        lp_authors = ", ".join(lp.output.get("authors", [])[:4])
        lp_abstract = lp.output.get("abstract", "")[:600]
        lp_sections = lp.output.get("sections", [])
        sec_lines = "\n".join(
            f"  {'  ' * (s.get('level', 1) - 1)}{'#' * s.get('level', 1)} {s.get('title', '')}"
            for s in lp_sections[:12]
        )
        lp_equations = lp.output.get("equations", [])[:5]
        eq_lines = "\n".join(f"  {eq[:200]}" for eq in lp_equations)
        parts.append(
            f"<latex_document title=\"{lp_title}\">\n"
            f"Authors: {lp_authors}\n"
            f"Abstract: {lp_abstract}\n"
            + (f"Sections:\n{sec_lines}\n" if sec_lines else "")
            + (f"Key equations:\n{eq_lines}\n" if eq_lines else "")
            + f"</latex_document>"
        )

    # Media generation — job queued
    mg = results.get("media_generate")
    if mg and mg.output.get("status") == "queued":
        mg_type = mg.output.get("media_type", "")
        mg_href = mg.output.get("href", "")
        mg_count = mg.output.get("paper_count", 0)
        parts.append(
            f"<media_generation_queued>\n"
            f"{mg_type.capitalize()} generation started for {mg_count} paper(s). "
            f"Track progress at: {mg_href}\n"
            f"</media_generation_queued>"
        )

    return "\n\n".join(parts)


def _build_paper_context(papers: list[dict], arxiv_results: list[dict]) -> str:
    """Build the per-paper context block, neutralising prompt-injection
    attempts in the retrieved prose.

    Paper titles + abstracts are external untrusted data. A paper whose
    abstract contains an "ignore previous instructions" stanza would
    otherwise be obeyed by the synthesizer. :func:`sanitize_untrusted`
    quote-escapes those phrases so the model still sees the text but
    can't be steered by it.
    """
    from app.assistant.prompt_safety import sanitize_untrusted

    sections: list[str] = []
    if papers:
        sections.append("\n\n".join(
            f"[{i + 1}] {sanitize_untrusted(p.get('title'))}\n"
            f"Authors: {sanitize_untrusted(', '.join(p.get('authors') or []))}\n"
            f"Namespace: {p.get('namespace_key', '')}\n"
            f"Abstract/TLDR: {sanitize_untrusted((p.get('tldr') or p.get('abstract', ''))[:900])}"
            for i, p in enumerate(papers[:8])
        ))
    if arxiv_results:
        # Deduplicate against corpus papers already included above
        corpus_titles = {p.get("title", "").lower() for p in papers}
        deduped = [p for p in arxiv_results if p.get("title", "").lower() not in corpus_titles]
        if deduped:
            sections.append("\n\n".join(
                f"[A{i + 1}] {sanitize_untrusted(p.get('title'))}\n"
                f"Authors: {sanitize_untrusted(', '.join(p.get('authors') or []))}\n"
                f"Abstract: {sanitize_untrusted(p.get('abstract', '')[:900])}"
                for i, p in enumerate(deduped[:6])
            ))
    return "\n\n".join(sections)


def _build_prompt(
    *,
    query: str,
    context: str,
    extra_context: str,
    actions: list[str],
    imported_count: int,
    graph_result: dict | None,
    genie_session_id: str | None,
    orientation: str,
    expertise: str,
    grounded: bool,
    pure_reasoning: bool = False,
) -> str:
    expertise_hint = _EXPERTISE_HINTS.get(expertise, "")
    orientation_hint = _ORIENTATION_HINTS.get(orientation, "")

    if pure_reasoning:
        # No retrieval evidence — direct knowledge answer.
        return (
            "You are ResearchFlow's AI-native research collaborator — brilliant, warm, precise, and deeply useful.\n\n"
            f"USER QUERY: {query}\n"
            f"USER PROFILE: expertise={expertise}; orientation={orientation}.\n"
            f"{expertise_hint} {orientation_hint}\n\n"
            f"{_FORMAT_GUIDANCE}\n\n"
            "No retrieval tools were needed for this query — answer directly from your knowledge.\n"
            "RESPONSE RULES:\n"
            "• Be accurate, clear, and appropriately detailed for the user's expertise level.\n"
            "• Use proper markdown formatting: bold key terms, headers for multi-section answers, bullet points where appropriate.\n"
            "• Do NOT fabricate citations. Only cite sources if you know them with certainty.\n"
            "• Write a complete answer — never truncate.\n\n"
            "Write your response now:"
        )

    grounding_clause = (
        "Papers marked [1], [2], … are from the user's INDEXED CORPUS — primary evidence."
        if grounded
        else "Papers marked [A1], [A2], … are unindexed arXiv CANDIDATES — treat as lightly-vetted leads, not verified evidence."
    )

    # Build the full evidence section combining papers + all other tool outputs
    evidence_parts: list[str] = []
    if context:
        evidence_parts.append(f"── Corpus papers ──\n{context}")
    if extra_context:
        evidence_parts.append(f"── Additional tool outputs ──\n{extra_context}")
    if graph_result:
        evidence_parts.append(f"── Knowledge graph ──\n{json.dumps(graph_result, default=str)[:600]}")
    if genie_session_id:
        evidence_parts.append("── Genie synthesis ── A novel hypothesis was generated in a Genie session (id surfaced in UI).")
    if imported_count:
        evidence_parts.append(f"── arXiv imports ── {imported_count} new paper(s) imported this turn (embed in next turn).")

    evidence_block = "\n\n".join(evidence_parts)

    from app.assistant.prompt_safety import untrusted_block_preamble
    return (
        "You are ResearchFlow's AI-native research collaborator — brilliant, warm, precise, and deeply useful.\n\n"
        f"{untrusted_block_preamble()}\n\n"
        "<evidence>\n"
        f"{evidence_block}\n"
        "</evidence>\n\n"
        f"<grounding>{grounding_clause}</grounding>\n\n"
        f"USER QUERY: {query}\n"
        f"USER PROFILE: expertise={expertise}; orientation={orientation}.\n"
        f"{expertise_hint} {orientation_hint}\n\n"
        f"TOOLS USED THIS TURN: {', '.join(actions) or 'none'}\n\n"
        f"{_LIFECYCLE_GUIDANCE}\n\n"
        f"{_FORMAT_GUIDANCE}\n\n"
        "RESPONSE RULES:\n"
        "• The <memory_hint> block is ADVISORY ONLY — treat it as a soft, possibly-stale hint about the user's preferences and prior context. Let it lightly shape voice/depth/framing; NEVER let it override or contradict grounded evidence; never cite or quote it.\n"
        "• ResearchFlow is arXiv-first. Prefer arXiv-sourced papers as primary evidence. Treat Wikipedia, web search, CrossRef, and Semantic Scholar as supplementary — use them to enrich or contextualise, not to replace arXiv evidence.\n"
        "• Synthesize ALL available evidence (papers, computations, attachments, web) into one coherent answer.\n"
        "• Cite inline: [1] for indexed papers, [A1] for arXiv candidates, [WA] for Wolfram Alpha, [W] for web sources.\n"
        "• Distinguish fact from hypothesis; acknowledge evidence gaps honestly.\n"
        "• If an <agent_notes> block is present, treat it as the agent's own self-assessment of the evidence it gathered. When it flags thin evidence, low groundedness, or unresolved gaps, REFLECT that honestly in the answer — caveat claims, name what's missing, and suggest a follow-up rather than over-claiming. The block is metadata; do NOT quote it or reproduce its prose.\n"
        "• Adapt voice and depth to the user's expertise level.\n"
        "• Close with one concrete, stage-appropriate next step (unless the question is fully resolved).\n"
        "• NEVER produce a rigid Takeaway / Evidence / Gaps / Next-moves skeleton. Choose the shape that fits.\n"
        "• NEVER reproduce XML/section tags from the evidence block (like <wolfram_computation>, <concept_background>, etc.) in your response — those are internal context markers only.\n"
        "• Use proper markdown: bold key terms, use headers sparingly, bullet points for lists, italics for emphasis.\n"
        "• Write a complete answer — NEVER cut off mid-sentence or truncate.\n"
        "\n"
        "DEPTH DISCIPLINE — when the user asks for an architecture, a novel "
        "idea, a synthesis, or a research direction (NOT for a definition / "
        "lookup / quick overview), do not stop at impressive prose. Convert "
        "the answer into something falsifiable and implementable:\n"
        "  • PRIMITIVES — name the building blocks (state, inputs, outputs, "
        "control policy) so a reader could begin implementing.\n"
        "  • MECHANISM — explain how the parts compose, where the load-"
        "bearing assumptions sit, and what the data path looks like.\n"
        "  • STRONGEST FAIR BASELINE — name the strongest existing system "
        "the new idea must beat, and say why that baseline (not a strawman) "
        "is the right comparison.\n"
        "  • NOVELTY VS BORROWED — for each major component, label it "
        "borrowed (from which prior work) or genuinely new. Mark anything "
        "you can't place as 'unattributed' rather than implying novelty.\n"
        "  • EVIDENCE TIER — separate established results, justified "
        "inference, and speculation. Don't promote inferences to facts.\n"
        "  • FALSIFIERS / MINIMAL EXPERIMENT — name at least one concrete "
        "ablation or experiment whose negative result would invalidate the "
        "idea. If you can't name one, the idea is not yet falsifiable.\n"
        "These sub-points should NOT become rigid section headers — weave "
        "them into the prose where they fit the question's shape. The goal "
        "is for the reader to know what to implement and what would prove "
        "the design wrong.\n\n"
        "EVIDENCE vs INFERENCE LABELLING — the reader must always be able "
        "to tell which is which. Use one of these inline markers when the "
        "distinction is non-obvious:\n"
        "  • \"Directly shown by [N]:\" — a claim the cited paper states.\n"
        "  • \"Reasonable inference from [N], [M]:\" — your synthesis from "
        "    cited evidence, not a direct quote.\n"
        "  • \"RA hypothesis (no direct citation):\" — your own speculation "
        "    beyond what sources show.\n"
        "  • \"Uncertain:\" — the evidence is mixed / thin and you can't "
        "    confidently land on either side.\n"
        "Do not bury a hypothesis inside source-summary prose. A reader "
        "skimming the answer should be able to identify in two seconds "
        "what is sourced vs what is RA's own read.\n\n"
        "PROVISIONAL CLAIMS (full-paper verification gate — read <agent_notes> "
        "STRONG-CLAIM LEDGER block when present):\n"
        "  • If a claim is in the PROVISIONAL or UNVERIFIABLE list, do NOT "
        "    state it as if the cited paper's full body confirms it. Label "
        "    it: \"(abstract-only)\" or \"(provisional — full-paper "
        "    verification unavailable)\". The full sentence stays — the "
        "    label tells the reader the strength of the support.\n"
        "  • If a claim is in the CONTRADICTED list, do not repeat the "
        "    original claim without flagging the contradiction; either "
        "    drop the sentence or write it as \"X was claimed but the full "
        "    paper's body did not support it\".\n"
        "  • If a claim is in the VERIFIED list, you may state it firmly.\n\n"
        "COMPETING EXPLANATIONS — when the evidence surfaced multiple "
        "candidate bottlenecks / explanations / methods, RANK them by "
        "EXPLICIT criteria, not by how many papers mention each:\n"
        "  • Evidence strength (direct + ablation > suggestive + indirect).\n"
        "  • Causal plausibility (mechanism articulated vs hand-wavy).\n"
        "  • Practical impact (degree of improvement on realistic tasks).\n"
        "  • Generality (works across domains / settings vs narrow).\n"
        "  • Testability (cleanly falsifiable vs vague).\n"
        "  • Production relevance (only when the question is applied).\n"
        "State the winning candidate and WHY it ranks above the others on "
        "these axes. Popularity in retrieved papers is NOT a criterion.\n\n"
        "PRODUCTION-AWARENESS (only when relevant — DO NOT bolt on a generic "
        "deployment checklist):\n"
        "  • If the question is about an applied / deployed AI system "
        "(serving in production, user-facing, regulated, or under "
        "operational constraints), discuss the relevant deployment "
        "concerns: latency, compute, data collection cost, safety, "
        "monitoring, human override, failure recovery, maintainability, "
        "evaluation drift — but ONLY the ones that actually matter for "
        "THIS method on THIS use case. Be specific.\n"
        "  • If the question is pure research / theory / explanation / "
        "literature, do NOT inject a production checklist. It is "
        "off-topic and dilutes the answer.\n"
        "  • When the user's orientation hint says 'research', lean "
        "further away from production framing. When it says "
        "'production', lean further toward it. 'Both' splits the "
        "difference based on the question shape itself.\n\n"
        "Write your response now:"
    )


def build_message_blocks(
    *,
    answer: str,
    papers: list[dict],
    arxiv_results: list[dict],
    imported_count: int,
    graph_result: dict | None,
    genie_session_id: str | None,
    suggestions: list[dict],
    actions: list[str],
    web_results: list[dict] | None = None,
    comparison: dict | None = None,
    bookmarks_answer: str | None = None,
    mermaid: tuple[str, str] | None = None,
    domain_papers: list[dict] | None = None,
    nvd_results: list[dict] | None = None,
    fred_series: list[dict] | None = None,
    trials_results: list[dict] | None = None,
    code_results: list[dict] | None = None,
    agent_notes: dict | None = None,
) -> list[dict[str, Any]]:
    """Assemble the structured ``payload.blocks`` list rendered by the UI.

    Block kinds: ``text``, ``paper_grid``, ``arxiv_grid``, ``source_papers``,
    ``graph_summary``, ``web_results``, ``comparison_table``, ``bookmarks_answer``,
    ``artifact_link``, ``suggestion_chips``, ``actions_taken``,
    ``nvd_results``, ``fred_data``, ``trials_results``, ``code_results``,
    ``citation_table``.
    Frontend dispatches per-block so the assistant message is never a wall of markdown.

    ``agent_notes`` carries the ReAct loop's claim ledger and the
    provenance verifier's per-citation report when available. The
    ``citation_table`` block — emitted at the END of the message — joins
    those reports against the actual ``papers`` + ``arxiv_results`` lists
    to give the user one consolidated, auditable view of every cited
    source: marker, title, evidence tier, verification verdict, and
    clickable URL. The user asked for this so they can verify every
    citation at a glance instead of having to cross-reference inline
    markers with separate grids.
    """
    blocks: list[dict[str, Any]] = []
    if answer:
        blocks.append({"kind": "text", "content": answer})
    if comparison and comparison.get("rows"):
        blocks.append({
            "kind": "comparison_table",
            "title": "Side-by-side comparison",
            "columns": comparison.get("columns") or [],
            "rows": comparison.get("rows") or [],
            "notes": comparison.get("notes") or "",
        })
    if bookmarks_answer:
        blocks.append({
            "kind": "bookmarks_answer",
            "title": "From your bookmarked papers",
            "content": bookmarks_answer,
        })
    if mermaid:
        title, code = mermaid
        # Only include if the code is non-trivial (real diagram, not empty or error)
        if code and len(code.strip()) > 40 and not code.strip().startswith("error"):
            blocks.append({
                "kind": "mermaid",
                "title": title,
                "code": code,
            })
    if papers:
        blocks.append({
            "kind": "paper_grid",
            "title": "Grounded papers",
            "papers": papers[:10],
        })
    # External arXiv references — always surfaced when present, regardless
    # of whether grounded corpus papers exist. The answer text may emit
    # ``[A1]…[A20]`` citation markers referencing these external papers
    # alongside ``[1]…[N]`` for the corpus; the frontend's citation map
    # needs both blocks to render either kind of citation as a clickable
    # link (corpus → Paper Panel, arXiv → external URL). Suppressing this
    # block when corpus papers exist left every ``[A*]`` marker visually
    # styled as a link but functionally inert.
    if arxiv_results:
        if papers:
            title = "External references (arXiv)"
        elif imported_count:
            title = f"arXiv candidates ({imported_count} imported)"
        else:
            title = "arXiv candidates (browse only)"
        blocks.append({
            "kind": "arxiv_grid",
            "title": title,
            "papers": arxiv_results[:20],
            "imported_count": imported_count,
        })
    # Domain-specific paper results (pubmed, inspire_hep, nasa_ads, papers_with_code)
    if domain_papers:
        blocks.append({
            "kind": "source_papers",
            "title": "Domain literature",
            "papers": domain_papers[:12],
        })
    # Security vulnerability data (nvd_cve)
    if nvd_results:
        blocks.append({
            "kind": "nvd_results",
            "title": f"Security vulnerabilities ({len(nvd_results)} found)",
            "vulnerabilities": nvd_results[:8],
        })
    # Clinical trials (clinicaltrials.gov)
    if trials_results:
        blocks.append({
            "kind": "trials_results",
            "title": f"Clinical trials ({len(trials_results)} found)",
            "studies": trials_results[:8],
        })
    # FRED macroeconomic time series
    if fred_series:
        blocks.append({
            "kind": "fred_data",
            "title": "FRED economic data",
            "series": fred_series[:4],
        })
    # Code repositories and models (github, huggingface)
    if code_results:
        blocks.append({
            "kind": "code_results",
            "title": "Code & models",
            "items": code_results[:10],
        })
    if web_results:
        blocks.append({
            "kind": "web_results",
            "title": "Web context (secondary)",
            "results": web_results[:6],
        })
    # Only show graph block when the graph has actual renderable content
    if graph_result and _graph_has_content(graph_result):
        blocks.append({
            "kind": "graph_summary",
            "title": "Knowledge graph",
            "summary": graph_result,
            "href": "/graph",
        })
    if genie_session_id:
        blocks.append({
            "kind": "artifact_link",
            "title": "Genie synthesis queued",
            "kind_label": "genie",
            "href": "/genie?tab=discoveries",
            "ref_id": genie_session_id,
        })
    if suggestions:
        blocks.append({
            "kind": "suggestion_chips",
            "title": "Next moves",
            "suggestions": suggestions,
        })
    if actions:
        blocks.append({
            "kind": "actions_taken",
            "actions": actions,
        })
    # Consolidated citation table — emitted at the END so it reads
    # like a References section. Joins the paper / arxiv lists against
    # the claim ledger + provenance verifier so each row shows the
    # marker the answer used, the title, the evidence tier, the
    # verification verdict, and the clickable URL. Skipped on turns
    # with no citations at all (free-reasoning answers).
    citation_rows = _build_citation_rows(
        papers=papers,
        arxiv_results=arxiv_results,
        agent_notes=agent_notes,
    )
    if citation_rows:
        blocks.append({
            "kind": "citation_table",
            "title": "Citations",
            "rows": citation_rows,
        })
    # Defensive guard — keep the citation table pinned to the END of
    # the block list regardless of which paths above appended what.
    # The user's spec is unambiguous ("Citation table should be shown
    # at the end of the output"); a future contributor adding a new
    # block kind after this point would otherwise silently displace
    # the table. Centralising the invariant here means any future
    # blocks land in their natural position and the table still ships
    # last.
    return _pin_citation_table_last(blocks)


def _pin_citation_table_last(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Move every ``citation_table`` block to the tail of the list.

    Idempotent: calling on a list where the table is already last is a
    no-op. Multiple tables collapse to a single trailing run (the
    synthesizer only ever emits one, but the helper handles arbitrary
    counts so downstream merges stay safe).
    """
    if not blocks:
        return blocks
    head: list[dict[str, Any]] = []
    tail: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, dict) and b.get("kind") == "citation_table":
            tail.append(b)
        else:
            head.append(b)
    return head + tail


def _build_citation_rows(
    *,
    papers: list[dict],
    arxiv_results: list[dict],
    agent_notes: dict | None,
) -> list[dict]:
    """Compose the consolidated citation table the UI renders at the
    bottom of every grounded message.

    Each row carries:

    * ``marker``         — the inline marker the answer uses (``1`` for
                           corpus, ``A1`` for external).
    * ``title``          — the paper title.
    * ``authors``        — first three authors, comma-separated.
    * ``namespace_key``  — corpus only; empty for external rows.
    * ``paper_id``       — corpus UUID, used by the frontend to open
                           the Paper Panel.
    * ``url``            — external URL when available (DOI / arXiv /
                           publisher / Semantic Scholar). Empty when
                           the row is corpus-only — those open through
                           ``paper_id`` instead.
    * ``evidence_tier``  — ``experiment-verified`` / ``method-verified``
                           / ``abstract-only`` / ``unverified``, sourced
                           from the strong-claim ledger when the paper
                           was the subject of a forced verification
                           round; otherwise defaults to the row's
                           inherent tier (corpus rows: abstract-only;
                           arXiv rows: unverified until inspected).
    * ``verdict``        — ``verified`` / ``contradicted`` / ``provisional``
                           / ``unverified`` / ``unresolved``. ``unresolved``
                           specifically marks rows the answer cited but
                           where the URL / paper_id couldn't be resolved
                           — the user explicitly asked us to surface
                           these rather than silently treat them as
                           grounded.

    Empty list when no papers or external candidates were surfaced this
    turn — the citation table block is then suppressed entirely.
    """
    rows: list[dict] = []

    # Build a paper_id → verdict / tier map from the claim ledger so
    # rows can carry per-paper verification state. Defensive against
    # malformed agent_notes — a non-dict ``claim_ledger`` (some future
    # serialisation path that lands a list here) silently collapses to
    # empty maps rather than crashing the table builder.
    raw_ledger = (agent_notes or {}).get("claim_ledger")
    ledger: dict = raw_ledger if isinstance(raw_ledger, dict) else {}
    by_paper_tier: dict[str, str] = {}
    by_paper_verdict: dict[str, str] = {}
    for bucket_name, default_verdict in (
        ("verified", "verified"),
        ("contradicted", "contradicted"),
        ("provisional", "provisional"),
    ):
        bucket = ledger.get(bucket_name)
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("paper_id") or "")
            if not pid:
                continue
            # Strongest tier wins; the same paper may carry multiple
            # claims at different tiers.
            tier = str(item.get("evidence_tier") or "abstract-only")
            current = by_paper_tier.get(pid)
            if _tier_rank(tier) > _tier_rank(current or "abstract-only"):
                by_paper_tier[pid] = tier
            elif current is None:
                by_paper_tier[pid] = tier
            # First-write wins for verdict (the ledger orders strongest
            # bucket first).
            by_paper_verdict.setdefault(pid, default_verdict)

    # Provenance verifier per-marker verdicts, when available. Same
    # defensive shape guard: a non-dict provenance silently collapses
    # to an empty flagged set rather than crashing the builder.
    raw_prov = (agent_notes or {}).get("provenance")
    prov: dict = raw_prov if isinstance(raw_prov, dict) else {}
    flagged_raw = prov.get("flagged")
    flagged_iter = flagged_raw if isinstance(flagged_raw, list) else []
    flagged_markers: set[str] = {
        str(f.get("marker") or "")
        for f in flagged_iter
        if isinstance(f, dict) and f.get("verdict") in {"unverified", "unsupported"}
    }

    # Corpus papers — markers [1]..[N].
    for idx, p in enumerate(papers or [], start=1):
        pid = str(p.get("paper_id") or "")
        marker = str(idx)
        tier = by_paper_tier.get(pid, "abstract-only")
        verdict = by_paper_verdict.get(pid, "provisional")
        # A corpus row whose marker the provenance verifier flagged
        # gets a softer "unverified" verdict so the UI can highlight
        # the misalignment rather than imply the citation is grounded.
        if marker in flagged_markers and verdict == "provisional":
            verdict = "unverified"
        rows.append({
            "marker": marker,
            "is_external": False,
            "title": (p.get("title") or "")[:280],
            "authors": [(a or "") for a in (p.get("authors") or [])[:3]],
            "namespace_key": p.get("namespace_key") or "",
            "paper_id": pid,
            "url": p.get("source_url") or "",
            "evidence_tier": tier,
            "verdict": verdict,
        })

    # External candidates — markers [A1]..[A20].
    for idx, p in enumerate(arxiv_results or [], start=1):
        marker = f"A{idx}"
        ext_id = str(p.get("external_id") or "").strip()
        src_url = str(p.get("source_url") or p.get("url") or "").strip()
        url = ""
        if ext_id and not ext_id.lower().startswith(("http://", "https://")):
            url = (
                f"https://doi.org/{ext_id}"
                if "/" in ext_id and not ext_id.lower().startswith("arxiv:")
                else f"https://arxiv.org/abs/{ext_id}"
            )
        elif src_url:
            url = src_url
        verdict = "unverified"
        if not url:
            # Unresolved — the answer cited a candidate we cannot link
            # back to a real source. Surface explicitly per user spec.
            verdict = "unresolved"
        if marker in flagged_markers and verdict != "unresolved":
            verdict = "unverified"
        rows.append({
            "marker": marker,
            "is_external": True,
            "title": (p.get("title") or "")[:280],
            "authors": [(a or "") for a in (p.get("authors") or [])[:3]],
            "namespace_key": "",
            "paper_id": "",
            "url": url,
            "evidence_tier": "abstract-only" if url else "unverified",
            "verdict": verdict,
        })

    return rows


_TIER_ORDER = {
    "unverified": 0,
    "abstract-only": 1,
    "method-verified": 2,
    "experiment-verified": 3,
}


def _tier_rank(tier: str) -> int:
    """Return a sortable rank for an evidence tier; unknown tiers
    sort below ``unverified`` so the strongest tier always wins the
    "strongest tier wins" merge."""
    return _TIER_ORDER.get(tier or "", -1)


def _graph_has_content(graph_result: dict) -> bool:
    """Return True only when the graph has actual nodes/edges worth rendering."""
    if not graph_result:
        return False
    nodes = graph_result.get("nodes") or graph_result.get("node_count") or []
    edges = graph_result.get("edges") or graph_result.get("edge_count") or []
    n = len(nodes) if isinstance(nodes, (list, dict)) else int(nodes or 0)
    e = len(edges) if isinstance(edges, (list, dict)) else int(edges or 0)
    return n > 0 or e > 0


def _fallback_answer(
    query: str,
    papers: list[dict],
    imported_count: int,
    graph_result: dict | None,
    genie_session_id: str | None,
) -> str:
    if not papers:
        # Honest, actionable empty state — never claim something we don't have.
        # The block renderer surfaces arxiv_results separately so the user
        # still sees candidates without us pretending they're "grounded".
        ns_msg = (
            f"I imported {imported_count} new paper(s) this turn but they "
            "haven't been embedded into the searchable corpus yet — "
            "try the same question again in a moment."
            if imported_count > 0
            else "No indexed papers matched in the current scope."
        )
        return (
            "I don't have grounded evidence to answer this question right now — "
            "I'm flagging that honestly rather than improvising.\n\n"
            f"{ns_msg}\n\n"
            "**What to try next:**\n"
            "- Rephrase the question with broader terminology\n"
            "- Enable arXiv import so fresh candidates land in your feed\n"
            "- Try a follow-up like \"search across all namespaces\""
        )
    lines = [f"I found {len(papers)} relevant paper(s) for your question.\n"]
    for i, p in enumerate(papers[:5], start=1):
        ns = p.get("namespace_key") or ""
        lines.append(f"- **[{i}]** {p.get('title')}{f' ({ns})' if ns else ''}")
    if genie_session_id:
        lines.append(
            "\n*A Genie synthesis workflow has been queued — it will appear "
            "in Genie discoveries when complete.*"
        )
    lines.append(
        "\n*(Synthesis model unavailable — this is a structured fallback without "
        "LLM-composed analysis.)*"
    )
    return "\n".join(lines)
