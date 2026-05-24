"""Tool-ban middleware — short-circuits dispatches to banned tools and
maintains the per-tool failure counters.

A tool that fails repeatedly within one turn gets banned for the rest
of that turn. Without this gate the model can burn every remaining
iteration on the same broken tool (network down, MCP unreachable, etc.).

The ban policy:

  * Threshold lives in :data:`app.assistant.tuning.REACT_SAME_TOOL_FAILURE_CAP`
    (default 2 — two consecutive failures is the inflection where the
    third is unlikely to help).
  * Bans are turn-scoped. A new user turn starts with empty
    ``banned_tools`` and ``tool_fail_counts``.
  * Banned tools are advertised in the decision prompt (see
    :func:`app.assistant.react_loop._decide_next_action`) so the model
    doesn't keep picking them.

The middleware also enforces ``_DISALLOWED_FROM_LOOP`` (memory_write /
memory_delete) — the loop is never allowed to write durable memory
mid-turn because the post-turn auto-memory pass is the single writer
for that state.
"""

from __future__ import annotations

from typing import Any

from app.assistant.react.middleware import AbortDispatch, CONTINUE, PreDispatchResult
from app.assistant.react.middlewares.base import BaseMiddleware
from app.assistant.tuning import REACT_SAME_TOOL_FAILURE_CAP

# Mirror of the symbol historically defined in react_loop.py; kept here
# so the middleware doesn't depend on the loop module's internals.
_DISALLOWED_FROM_LOOP: frozenset[str] = frozenset({
    "memory_write",
    "memory_delete",
})


class ToolBanMiddleware(BaseMiddleware):
    """Block dispatches to banned / disallowed tools and centralise the
    per-tool failure counter that drives the ban."""

    name = "tool_ban"

    async def before_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
    ) -> PreDispatchResult:
        if action in _DISALLOWED_FROM_LOOP:
            return AbortDispatch(
                reason="disallowed_from_loop",
                observation_summary=(
                    f"Tool '{action}' is not callable from the ReAct loop; "
                    "durable memory writes happen on the post-turn pass."
                ),
                error="tool_disallowed",
            )
        if action in state.banned_tools:
            return AbortDispatch(
                reason="banned_after_failures",
                observation_summary=(
                    f"Tool '{action}' has been banned for this turn after "
                    f"{REACT_SAME_TOOL_FAILURE_CAP}+ consecutive failures. "
                    "Pick a different tool or finalize."
                ),
                error="tool_banned",
            )
        return CONTINUE

    async def on_tool_error(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
        exc: BaseException,
    ) -> None:
        # Central accounting point — every dispatch-time error funnels
        # through here so the counter and ban set stay consistent
        # regardless of which middleware first observed the failure.
        state.record_tool_failure(action, ban_cap=REACT_SAME_TOOL_FAILURE_CAP)
