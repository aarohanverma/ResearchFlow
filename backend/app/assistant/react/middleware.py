"""Middleware Protocol + chain composition for the ReAct loop.

The chain has four lifecycle hooks per iteration:

  ``before_iteration``      — called once at iteration start, before the
                              decision LLM is invoked. Use to set up
                              per-iteration counters, refresh prompts.
  ``before_tool``           — called after the model picks an ACTION,
                              before the tool runs. Returns a
                              :class:`PreDispatchResult` to either
                              continue, override (modified params),
                              or abort (skip dispatch).
  ``after_tool``            — called after a tool runs successfully.
                              Use to update ledgers, observability,
                              contradiction detectors.
  ``on_tool_error``         — called when a tool dispatch raises.
                              Use to update fail counts, ban tools,
                              record observations.
  ``gate_finalize``         — called when the model says ``finalize``.
                              Returns a :class:`FinalizeGate` allowing
                              the finalize, forcing a critique, or
                              forcing another action.

Middlewares are sequential by design. The first override / abort /
non-allow gate wins so a high-priority middleware (e.g. tool-ban
policy) can short-circuit the rest. Order matters and is set by the
caller.

All hooks are async because middlewares may need to call LLMs
(semantic contradiction detector) or hit the DB (memory checks).
Hooks that don't need to be async simply return immediately.

Every hook is optional — a middleware that only cares about
``after_tool`` doesn't need to implement the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Union, runtime_checkable

from app.assistant.tools.base import ToolResult


# ── Hook return types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DispatchOverride:
    """``before_tool`` return value that mutates the upcoming dispatch.

    Use when a middleware repairs params (placeholder strip, auto-fill
    from ledger), switches tools (banned-tool redirect), or rewrites
    rationale. Setting ``action`` to a different tool name re-routes
    the dispatch; setting ``params`` overrides the model's arguments.
    """
    action: str | None = None         # None = keep model's action
    params: dict[str, Any] | None = None
    rationale: str | None = None
    note_for_scratchpad: str | None = None


@dataclass(frozen=True)
class AbortDispatch:
    """``before_tool`` return value that *skips* the dispatch entirely.

    Use when a middleware decides this tool call shouldn't run at all
    (e.g. banned tool, redundant call already in this turn). The loop
    logs the abort to the scratchpad and moves on to the next iteration.
    """
    reason: str
    observation_summary: str
    error: str | None = None          # surfaced on the Observation entry


# Sentinel "continue normally" return; lets a middleware express "no
# opinion on this dispatch" without us having to use Optional + None
# everywhere. Stops the type-checker from second-guessing the union.
class _Continue:
    __slots__ = ()
    def __repr__(self) -> str:  # noqa: D401 — debug only
        return "<Continue>"


CONTINUE = _Continue()


PreDispatchResult = Union[_Continue, DispatchOverride, AbortDispatch]


@dataclass(frozen=True)
class FinalizeAllow:
    """``gate_finalize`` return value that lets the model exit cleanly."""


@dataclass(frozen=True)
class FinalizeForceCritique:
    """``gate_finalize`` return value that forces a self-critique pass
    before allowing the finalize. The loop runs the critique, records
    it on the scratchpad, and then re-asks the model on the next
    iteration whether it still wants to finalize."""
    reason: str


@dataclass(frozen=True)
class FinalizeForceAction:
    """``gate_finalize`` return value that injects one more tool call
    before allowing the finalize. The loop dispatches the action and
    re-asks the model. Used by the contradiction-detector middleware
    to force a counter-search on a high-confidence open signal."""
    action: str
    params: dict[str, Any]
    reason: str
    rationale: str = ""


FinalizeGate = Union[FinalizeAllow, FinalizeForceCritique, FinalizeForceAction]


# ── Middleware Protocol ──────────────────────────────────────────────────────


@runtime_checkable
class ReactMiddleware(Protocol):
    """One independently-testable cross-cutting concern in the ReAct loop.

    Every concrete middleware sets ``name`` to a stable identifier
    (used in logs, scratchpad notes, and test fixtures). Hooks that
    the middleware doesn't care about return ``CONTINUE`` /
    ``FinalizeAllow()`` / ``None`` as the no-op.
    """

    name: str

    async def before_iteration(self, state: "_LoopStateT") -> None: ...  # noqa: F821

    async def before_tool(
        self,
        state: "_LoopStateT",                                            # noqa: F821
        action: str,
        params: dict[str, Any],
    ) -> PreDispatchResult: ...

    async def after_tool(
        self,
        state: "_LoopStateT",                                            # noqa: F821
        action: str,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None: ...

    async def on_tool_error(
        self,
        state: "_LoopStateT",                                            # noqa: F821
        action: str,
        params: dict[str, Any],
        exc: BaseException,
    ) -> None: ...

    async def gate_finalize(self, state: "_LoopStateT") -> FinalizeGate: ...  # noqa: F821


# Hint-only — avoids a cyclic import while keeping the Protocol generic.
_LoopStateT = "app.assistant.react.state.LoopState"  # type: ignore[assignment]


# ── Chain ────────────────────────────────────────────────────────────────────


class MiddlewareChain:
    """Sequential composer for a list of :class:`ReactMiddleware`.

    The chain is deliberately simple — no priority queues, no
    dependency graph, no parallel dispatch. Middleware order is the
    caller's contract: register them in the order they should fire.
    The first override / abort / non-allow gate wins, and remaining
    middlewares of the same hook are skipped for that event.

    Failure isolation: a middleware that raises is logged and skipped
    so one buggy middleware doesn't take down the loop. Tests prove
    each middleware's behavior in isolation; production tolerates a
    skipped middleware better than an aborted turn.
    """

    def __init__(self, middlewares: list[ReactMiddleware]) -> None:
        self.middlewares: list[ReactMiddleware] = list(middlewares)

    def names(self) -> list[str]:
        """For tracing / debugging — list the active middleware names
        in dispatch order."""
        return [getattr(m, "name", type(m).__name__) for m in self.middlewares]

    async def before_iteration(self, state: Any) -> None:
        for mw in self.middlewares:
            try:
                await mw.before_iteration(state)
            except Exception as exc:  # noqa: BLE001
                _log_mw_failure(mw, "before_iteration", exc)

    async def before_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
    ) -> PreDispatchResult:
        current_action = action
        current_params = dict(params)
        for mw in self.middlewares:
            try:
                result = await mw.before_tool(state, current_action, current_params)
            except Exception as exc:  # noqa: BLE001
                _log_mw_failure(mw, "before_tool", exc)
                continue
            if isinstance(result, AbortDispatch):
                return result
            if isinstance(result, DispatchOverride):
                if result.action is not None:
                    current_action = result.action
                if result.params is not None:
                    current_params = dict(result.params)
                if result.note_for_scratchpad and getattr(state, "pad", None):
                    state.pad.think(result.note_for_scratchpad)
            # CONTINUE / unknown → fall through
        # If any middleware returned an override, surface the
        # accumulated mutation as a single override.
        if current_action != action or current_params != params:
            return DispatchOverride(action=current_action, params=current_params)
        return CONTINUE

    async def after_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None:
        for mw in self.middlewares:
            try:
                await mw.after_tool(state, action, params, result)
            except Exception as exc:  # noqa: BLE001
                _log_mw_failure(mw, "after_tool", exc)

    async def on_tool_error(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
        exc: BaseException,
    ) -> None:
        for mw in self.middlewares:
            try:
                await mw.on_tool_error(state, action, params, exc)
            except Exception as inner:  # noqa: BLE001
                _log_mw_failure(mw, "on_tool_error", inner)

    async def gate_finalize(self, state: Any) -> FinalizeGate:
        """Walk middlewares in order; first non-allow gate wins.

        The loop is responsible for honouring the returned gate
        (running the critique, dispatching the forced action). After
        the side-effect, the next iteration re-enters the chain — a
        middleware that already fired its one-shot intervention is
        expected to remember (via its own counter on ``state``) and
        return ``FinalizeAllow()`` thereafter."""
        for mw in self.middlewares:
            try:
                gate = await mw.gate_finalize(state)
            except Exception as exc:  # noqa: BLE001
                _log_mw_failure(mw, "gate_finalize", exc)
                continue
            if not isinstance(gate, FinalizeAllow):
                return gate
        return FinalizeAllow()


# ── Internals ────────────────────────────────────────────────────────────────


def _log_mw_failure(mw: Any, hook: str, exc: BaseException) -> None:
    import logging
    log = logging.getLogger(__name__)
    name = getattr(mw, "name", type(mw).__name__)
    log.warning("react_middleware: %s.%s raised: %s", name, hook, exc)
