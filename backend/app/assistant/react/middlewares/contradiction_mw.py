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
