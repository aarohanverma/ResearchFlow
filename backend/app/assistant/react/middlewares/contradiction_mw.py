"""Contradiction detection middleware — three signal sources, one policy.

Sources, run in increasing cost order:

  1. **Lexical** — keyword markers like ``contradicts`` / ``fails to
     replicate`` / ``refutes`` in any text span from a tool result.
     Cheap, deterministic, fires per-result.
  2. **Numeric** — same metric (``accuracy``, ``F1``, ``latency``)
     reported with values that diverge by more than the metric-
     specific epsilon across results. Also cheap, fires per-result.
  3. **Semantic LLM pair check** — one cheap-model structured call
     over the top-N highest-topic-overlap paper pairs. Misses what
     lexical/numeric catch but picks up "paper A says X improves
     throughput, paper B says X has no effect" without explicit
     markers. Bounded to one call per turn after the ledger has
     enough material.

All three feed the same :class:`ContradictionLedger`. The middleware
also enforces the **adaptive counter-search policy**: when the model
tries to finalize but a high-confidence un-investigated contradiction
is open AND iteration budget remains, ``gate_finalize`` returns a
:class:`FinalizeForceAction` that injects a targeted
``citation_finder`` (or ``deep_search`` fallback) call.

Soft signals (confidence < threshold) are rendered into the prompt but
never auto-force — that's the user's "do not force unnecessarily"
guidance baked into the policy.
"""

from __future__ import annotations

import logging
from typing import Any

from app.assistant.contradiction import (
    detect_contradictions_in_results,
    detect_semantic_contradictions,
)
from app.assistant.react.middleware import (
    FinalizeAllow,
    FinalizeForceAction,
    FinalizeGate,
)
from app.assistant.react.middlewares.base import BaseMiddleware
from app.assistant.tools.base import ToolResult
from app.assistant.tools.registry import get_tool

log = logging.getLogger(__name__)


# When the ledger has at least this many papers, the semantic LLM
# detector becomes worth its single cheap-model call. Below it, the
# pair-overlap heuristic doesn't have enough candidates to be useful.
_SEMANTIC_LLM_MIN_LEDGER = 4


# Tokens we treat as too generic to count when checking if a
# contradiction touches the main topic. Without this filter, common
# research vocabulary ("model", "method", "result") would make every
# contradiction look on-topic and the guard would never fire.
_STOPWORDS_TOPIC: frozenset[str] = frozenset({
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
    "is", "are", "was", "were", "be", "been", "by", "with", "as",
    "from", "that", "this", "these", "those", "it", "its", "their",
    "model", "models", "method", "methods", "result", "results",
    "paper", "papers", "study", "studies", "research", "approach",
    "approaches", "system", "systems", "data", "task", "tasks",
    "show", "shows", "shown", "find", "finds", "found", "use", "uses",
    "using", "used", "we", "they", "our", "ours",
})


def _topic_tokens(text: str) -> set[str]:
    """Return the lowercased content-bearing tokens of ``text``.

    Drops stopwords and very short tokens. Used to detect overlap
    between a contradiction signal and the user's main query — a
    contradiction whose span shares zero meaningful vocabulary with
    the query is almost certainly about an adjacent subfield, not
    the main thesis.
    """
    import re as _re
    if not text:
        return set()
    raw = _re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+", text.lower())
    return {w for w in raw if len(w) >= 4 and w not in _STOPWORDS_TOPIC}


_MIN_SHARED_TOKENS = 2
_MIN_OVERLAP_RATIO = 0.20


def _contradiction_touches_main_topic(target: Any, *, query: str) -> bool:
    """Return True when the contradiction's span materially overlaps
    with the user's query.

    The earlier "≥1 shared token" rule let through too many adjacent-
    subfield signals (e.g. a query about "retrieval-augmented LLMs"
    and a contradiction span about "retrieval indices in databases"
    share "retrieval" but are unrelated). The tightened rule requires
    BOTH:

      * ≥ ``_MIN_SHARED_TOKENS`` (=2) shared content-bearing tokens, AND
      * the overlap covers ≥ ``_MIN_OVERLAP_RATIO`` (=20%) of the
        query's content tokens.

    A short query (1-2 tokens after stopwording) defaults to "on
    topic" because we can't make a confident off-topic call. A span
    with no extractable tokens (very short markers, numeric-only)
    also defaults to "on topic" — those tend to be legitimate
    numeric-divergence signals.
    """
    q_tokens = _topic_tokens(query)
    if len(q_tokens) < 2:
        return True   # query too thin to filter against
    span = ""
    try:
        span = str(getattr(target, "span", "") or "")
        if not span:
            span = str(getattr(target, "render", lambda: "")() or "")
    except Exception:
        span = ""
    s_tokens = _topic_tokens(span)
    if not s_tokens:
        return True
    shared = q_tokens & s_tokens
    if len(shared) < _MIN_SHARED_TOKENS:
        return False
    return (len(shared) / max(len(q_tokens), 1)) >= _MIN_OVERLAP_RATIO


class ContradictionMiddleware(BaseMiddleware):
    """Detect contradictions, surface them, force counter-search when
    the signal is high-confidence + open + budget allows."""

    name = "contradiction"

    def __init__(self, *, enable_semantic_llm: bool = True) -> None:
        self.enable_semantic_llm = enable_semantic_llm

    async def after_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None:
        # ── Lexical + numeric, per-result ───────────────────────────
        try:
            for sig in detect_contradictions_in_results(
                {action: result}, iteration=state.pad.iteration,
            ):
                if state.contradictions.add(sig):
                    state.pad.think(
                        f"⚠ Contradiction detected: {sig.render()[:240]} — "
                        "the loop will require counter-evidence before finalizing."
                    )
        except Exception as exc:  # noqa: BLE001
            log.debug("contradiction_mw: lexical/numeric scan failed: %s", exc)

        # ── Address resolution: did THIS action target an open signal? ─
        try:
            query_text = " ".join(
                str(v) for v in (params or {}).values() if isinstance(v, str)
            )
            if query_text:
                state.contradictions.mark_addressed(query_text)
        except Exception:  # noqa: BLE001
            pass

        # ── Semantic LLM sweep — at most one per turn ───────────────
        if (
            self.enable_semantic_llm
            and not state.semantic_check_done
            and len(state.ledger.by_id) >= _SEMANTIC_LLM_MIN_LEDGER
        ):
            state.semantic_check_done = True
            try:
                new_signals = await detect_semantic_contradictions(
                    query=state.query,
                    results=state.merged_results(),
                    existing=state.contradictions,
                )
                for sig in new_signals:
                    if state.contradictions.add(sig):
                        state.pad.think(
                            "⚠ Semantic contradiction (LLM): "
                            f"{sig.render()[:240]}"
                        )
            except Exception as exc:  # noqa: BLE001
                log.debug("contradiction_mw: semantic LLM scan skipped: %s", exc)

    async def gate_finalize(self, state: Any) -> FinalizeGate:
        """Block finalize on high-confidence un-investigated contradiction
        when budget allows.

        Policy details (lifted verbatim from the prior in-loop
        implementation, now properly localised):

        * One forced counter-search per turn maximum
          (``MAX_FORCED_PER_TURN`` in the ledger).
        * Requires at least 2 iterations remaining so the model can
          observe the result before finalizing.
        * Soft signals (confidence < ``FORCE_CONFIDENCE_THRESHOLD``)
          render in the prompt but never auto-force.
        * Last iteration always finalizes — we never blow past the
          iteration cap on a contradiction signal.
        """
        if state.is_last_iteration:
            return FinalizeAllow()
        target = state.contradictions.next_to_force(
            iterations_remaining=state.iterations_remaining(),
        )
        if target is None:
            return FinalizeAllow()

        # Main-topic guard. The user spec is explicit: "RA should
        # prioritize the user's central research topic before chasing
        # side-branch contradictions". A contradiction signal is only
        # WORTH forcing a counter-search when its span shares
        # meaningful vocabulary with the user's actual query — i.e.
        # the contradiction affects the main thesis. If the
        # contradiction is about an adjacent subfield that wouldn't
        # change the recommendation, we let finalize proceed and the
        # contradiction stays in the scratchpad as advisory context.
        if not _contradiction_touches_main_topic(target, query=state.query):
            state.pad.think(
                f"Contradiction surfaced but does not materially touch the "
                f"user's main topic ('{state.query[:80]}…') — leaving in "
                "scratchpad as advisory; not forcing a counter-search that "
                "would pull retrieval off-topic."
            )
            return FinalizeAllow()

        state.pad.think(
            f"Adaptive counter-search: high-confidence un-investigated "
            f"contradiction ({target.render()[:200]}). Issuing one "
            "targeted retrieval before allowing finalize."
        )
        forced_tool = (
            "citation_finder" if get_tool("citation_finder") is not None
            else "deep_search"
        )
        forced_params = (
            {"claim": target.span} if forced_tool == "citation_finder"
            else {"query": target.span}
        )
        state.contradictions.record_forced()
        state.forced_counter_searches += 1
        return FinalizeForceAction(
            action=forced_tool,
            params=forced_params,
            reason="contradiction_counter_search",
            rationale="adaptive counter-search on high-confidence contradiction",
        )
