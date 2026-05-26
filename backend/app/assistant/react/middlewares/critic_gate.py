"""Critic-gate middleware — require self-critique before too-early finalize.

The pre-refactor loop had this gate inline. Without it, a hasty model
finalized on iteration 1-2 with zero adversarial pressure on the
evidence base and the synth got no critique score to honour.

The policy:

  * Iteration < ``REACT_MIN_ITERS_BEFORE_FREE_FINALIZE`` (default 3).
  * No critique entry on the scratchpad yet.
  * Not on the final iteration (the iteration cap still wins).
  * Hasn't already forced a critique this turn (default cap is 1, but
    rises to 2 when the prior critique was negative AND the loop still
    has no retrievals of its own — see below).

Shallow-turn deepening:

  When the first forced critique returns a low groundedness or
  completeness score AND the loop has not run a single retrieval of
  its own (``new_results`` empty, ``successful_retrievals == 0``), one
  more critique is forced on the NEXT finalize attempt. The critique
  is a cheap LLM call; the second forcing creates real pressure for
  the model to gather more evidence before shipping rather than
  honouring its initial "plan already executed → finalize" inclination.

The actual critique LLM call lives in
:func:`app.assistant.react_loop._run_self_critique`. This middleware
just enforces the gate; the loop driver runs the critique when
``gate_finalize`` returns :class:`FinalizeForceCritique`.
"""

from __future__ import annotations

from typing import Any

from app.assistant.react.middleware import (
    FinalizeAllow,
    FinalizeForceAction,
    FinalizeForceCritique,
    FinalizeGate,
)
from app.assistant.react.middlewares.base import BaseMiddleware
from app.assistant.tuning import REACT_MIN_ITERS_BEFORE_FREE_FINALIZE

# Hard ceiling on forced critiques per turn so a model that
# repeatedly says "finalize" with no evidence can't be pinned at the
# gate forever. Two rounds covers the realistic case: one to surface
# weak scores, one to insist after the model ignored them.
_MAX_FORCED_CRITIQUES = 2

# Score thresholds for re-forcing the critique. The judge already
# treats anything below 0.6 as "should_repair"; we mirror that here so
# the gate doesn't second-guess the critique's own verdict.
_LOW_GROUNDEDNESS = 0.6
_LOW_COMPLETENESS = 0.5


def _latest_critique_scores(state: Any) -> tuple[float, float]:
    """Walk the scratchpad backwards for the most recent critique
    entry and return ``(groundedness, completeness)`` as floats.

    Returns ``(1.0, 1.0)`` when no critique has landed yet — the
    sentinel says "nothing recorded; treat as passing" so callers
    don't accidentally force another critique on a turn that simply
    hasn't run one yet.
    """
    try:
        for entry in reversed(list(state.pad.entries)):
            if getattr(entry, "kind", "") != "critique":
                continue
            g = float(getattr(entry, "groundedness", 1.0) or 1.0)
            c = float(getattr(entry, "completeness", 1.0) or 1.0)
            return g, c
    except Exception:
        return 1.0, 1.0
    return 1.0, 1.0


class CriticGateMiddleware(BaseMiddleware):
    """Force self-critique before allowing an evidence-poor finalize."""

    name = "critic_gate"

    async def gate_finalize(self, state: Any) -> FinalizeGate:
        if state.is_last_iteration:
            return FinalizeAllow()
        # Free finalize after the configured iteration count, IF the
        # loop has done at least one productive retrieval of its own
        # OR the initial plan was substantive (≥ 2 results). A loop
        # that finalizes on iteration 3 with zero new retrievals AND
        # a one-tool initial plan is the "plan already executed →
        # finalize" pattern the user flagged.
        evidence_substantive = (
            int(getattr(state, "successful_retrievals", 0) or 0) > 0
            or len(getattr(state, "new_results", {}) or {}) > 0
            or len(getattr(state, "prior_results", {}) or {}) >= 2
        )
        if state.iteration_count >= REACT_MIN_ITERS_BEFORE_FREE_FINALIZE and evidence_substantive:
            return FinalizeAllow()
        # Critique-budget exhausted → defer to the rest of the chain
        # (full-paper gate, contradiction gate) and ultimately allow
        # finalize. We never block forever on the model's verdict.
        if state.forced_critiques >= _MAX_FORCED_CRITIQUES:
            return FinalizeAllow()
        critique_on_pad = any(
            getattr(e, "kind", "") == "critique" for e in state.pad.entries
        )
        if critique_on_pad:
            # A critique already landed. Two escalation paths:
            #
            #   1. Critique reported LOW grounding AND we have papers
            #      in scope → force a paper_qa verification on the
            #      top candidate. This is the load-bearing "critique
            #      triggers actual verification" link the user
            #      explicitly asked for. A second critique alone
            #      doesn't change anything — verification does.
            #
            #   2. Critique reported low grounding AND there are NO
            #      papers in scope yet → re-force critique once more
            #      so the model's next decision surfaces the weak
            #      score and pressures it to gather evidence.
            #
            # Both paths are bounded by ``_MAX_FORCED_CRITIQUES``;
            # exhausting the budget falls through to FinalizeAllow.
            g, c = _latest_critique_scores(state)
            scores_low = g < _LOW_GROUNDEDNESS or c < _LOW_COMPLETENESS
            no_loop_retrievals = (
                int(getattr(state, "successful_retrievals", 0) or 0) == 0
                and len(getattr(state, "new_results", {}) or {}) == 0
            )
            # Path 1 — force verification on the highest-priority
            # paper the loop already has. This bridges the gap the
            # user flagged ("critique does not fully lead to actual
            # deeper verification"). We rely on the PaperLedger's
            # insertion order (newest retrieval first), which is
            # already ranked-by-relevance because the retrieval tools
            # add their top hits first.
            ledger = getattr(state, "ledger", None)
            top_paper_id: str = ""
            top_paper_title: str = ""
            try:
                if ledger is not None and getattr(ledger, "by_id", None):
                    for pid, info in ledger.by_id.items():
                        if pid:
                            top_paper_id = str(pid)
                            top_paper_title = str((info or {}).get("title", ""))
                            break
            except Exception:
                top_paper_id = ""
            if (
                scores_low
                and top_paper_id
                and state.forced_critiques < _MAX_FORCED_CRITIQUES
            ):
                state.forced_critiques += 1
                # Build a focused verification question from the user's
                # original query so paper_qa lands on the cited claim.
                question = str(getattr(state, "query", "") or "")[:280]
                return FinalizeForceAction(
                    action="paper_qa",
                    params={
                        "paper_id": top_paper_id,
                        "paper_title": top_paper_title,
                        "question": question,
                    },
                    reason=(
                        f"critique scored low (g={g:.2f}, c={c:.2f}) and the loop has "
                        "a candidate paper in the ledger — forcing one paper_qa "
                        "round before allowing finalize so claims get verified "
                        "against the actual paper body."
                    ),
                    rationale="critique-triggered full-paper verification",
                )
            # Path 2 — no papers in scope yet; re-force critique so
            # the model's next decision surfaces the weak score.
            if (
                state.forced_critiques < _MAX_FORCED_CRITIQUES
                and no_loop_retrievals
                and scores_low
            ):
                state.forced_critiques += 1
                return FinalizeForceCritique(
                    reason=(
                        f"prior critique scored low (g={g:.2f}, c={c:.2f}) and the "
                        "loop has no retrievals of its own yet — re-forcing to "
                        "pressure evidence gathering before finalize"
                    ),
                )
            return FinalizeAllow()
        # No critique on pad. If we already asked once and the model
        # didn't deliver, fall through to allow rather than pinning
        # the loop at the gate — the next attempt would just repeat.
        # The legacy contract was one-shot critique-forcing per turn;
        # we preserve it for the "no critique landed" path.
        if state.forced_critiques >= 1:
            return FinalizeAllow()
        state.forced_critiques += 1
        return FinalizeForceCritique(
            reason=(
                f"loop ran {state.iteration_count} iteration(s) with no critique "
                f"recorded; min before free finalize is "
                f"{REACT_MIN_ITERS_BEFORE_FREE_FINALIZE}"
            ),
        )
