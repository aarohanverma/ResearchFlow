"""Diminishing-returns guard middleware.

Two patterns trip it:

  1. **Identical redo** — the model picks a tool the planner (or an
     earlier ReAct iteration) already ran with the same params. We
     skip the dispatch and log the no-op so the loop can move on.

  2. **Empty-set retrieval** — a retrieval tool returned the exact
     same paper IDs as a previous retrieval call. The new call adds
     no information; the loop should finalize on the evidence it
     already has instead of paying latency for a guaranteed-redundant
     second pass.

Both checks reuse the existing helpers in ``react_loop.py``
(``_params_equal``, ``_is_diminishing_returns``) so this middleware
stays a thin policy wrapper over the established logic.
"""

from __future__ import annotations

from typing import Any

from app.assistant.react.middleware import AbortDispatch, CONTINUE, PreDispatchResult
from app.assistant.react.middlewares.base import BaseMiddleware
from app.assistant.tools.base import ToolResult


class DiminishingReturnsMiddleware(BaseMiddleware):
    """Skip redundant tool calls + stop the loop when retrieval saturates."""

    name = "diminishing_returns"

    async def before_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
    ) -> PreDispatchResult:
        from app.assistant.react_loop import _params_equal

        prior = state.prior_results.get(action) or state.new_results.get(action)
        if prior is None:
            return CONTINUE
        if _params_equal(prior, params):
            return AbortDispatch(
                reason="identical_redo",
                observation_summary="Skipped — identical call already executed this turn.",
                error=None,
            )
        return CONTINUE

    async def after_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None:
        """Signal to the loop driver that this retrieval saturated.

        We mark a flag on ``state`` rather than break the loop ourselves —
        the driver checks the flag after running every middleware so a
        later middleware (e.g. observability) still gets to record its
        own observation on this result. The actual ``break`` happens
        in the driver to keep the loop's control flow in one place.
        """
        from app.assistant.react_loop import _is_diminishing_returns

        if _is_diminishing_returns(action, result, state.prior_results, state.new_results):
            state.pad.think(
                f"'{action}' returned no new papers compared to prior retrievals — "
                "diminishing returns. Finalizing."
            )
            # The driver looks at this flag after the after_tool hooks
            # run; setting it via attribute is the lightest-weight way
            # to share the signal without growing the state class for
            # one bool.
            setattr(state, "_diminishing_returns_hit", True)
            state.completed_normally = True
