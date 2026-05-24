"""Critic-gate middleware — require self-critique before too-early finalize.

The pre-refactor loop had this gate inline. Without it, a hasty model
finalized on iteration 1-2 with zero adversarial pressure on the
evidence base and the synth got no critique score to honour.

The policy:

  * Iteration < ``REACT_MIN_ITERS_BEFORE_FREE_FINALIZE`` (default 3).
  * No critique entry on the scratchpad yet.
  * Not on the final iteration (the iteration cap still wins).
  * Hasn't already forced a critique this turn (one-shot per turn).

The actual critique LLM call lives in
:func:`app.assistant.react_loop._run_self_critique`. This middleware
just enforces the gate; the loop driver runs the critique when
``gate_finalize`` returns :class:`FinalizeForceCritique`.
"""

from __future__ import annotations

from typing import Any

from app.assistant.react.middleware import (
    FinalizeAllow,
    FinalizeForceCritique,
    FinalizeGate,
)
from app.assistant.react.middlewares.base import BaseMiddleware
from app.assistant.tuning import REACT_MIN_ITERS_BEFORE_FREE_FINALIZE


class CriticGateMiddleware(BaseMiddleware):
    """Force one self-critique before allowing an early finalize."""

    name = "critic_gate"

    async def gate_finalize(self, state: Any) -> FinalizeGate:
        if state.is_last_iteration:
            return FinalizeAllow()
        if state.forced_critiques >= 1:
            # We already injected one critique this turn; the model
            # has seen the result. If it still wants to finalize, let
            # it.
            return FinalizeAllow()
        if state.iteration_count >= REACT_MIN_ITERS_BEFORE_FREE_FINALIZE:
            return FinalizeAllow()
        # Critique already exists on the pad → finalize is allowed.
        if any(getattr(e, "kind", "") == "critique" for e in state.pad.entries):
            return FinalizeAllow()
        state.forced_critiques += 1
        return FinalizeForceCritique(
            reason=(
                f"loop ran {state.iteration_count} iteration(s) with no critique "
                f"recorded; min before free finalize is "
                f"{REACT_MIN_ITERS_BEFORE_FREE_FINALIZE}"
            ),
        )
