"""ReAct loop infrastructure — middleware chain + nested subagents.

This package factors the previously-monolithic ``run_react_loop`` into
deepagents-shaped layers without taking the deepagents dependency.
Each cross-cutting concern (param hygiene, tool-ban policy, contradiction
detection, retrieval observability, critic gating, diminishing-returns
guard) lives as an independent :class:`ReactMiddleware` that can be
tested in isolation, replaced per-namespace, or disabled via env flag.

The :class:`MiddlewareChain` composes them in order. The loop itself
(:func:`app.assistant.react_loop.run_react_loop`) is now a thin driver
that walks the chain at each lifecycle point.

Subagents (``app.assistant.react.subagents``) extend the existing
``fanout`` action with proper context-quarantine: each subagent is a
nested ReAct loop with its own scratchpad, tool subset, system prompt,
and structured-output contract. The parent receives a single summary,
not the dozens of intermediate observations the subagent produced.

This package is *internal* to the assistant — external callers should
keep using :func:`app.assistant.react_loop.run_react_loop`.
"""

from app.assistant.react.middleware import (
    AbortDispatch,
    DispatchOverride,
    FinalizeAllow,
    FinalizeForceAction,
    FinalizeForceCritique,
    FinalizeGate,
    MiddlewareChain,
    PreDispatchResult,
    ReactMiddleware,
)
from app.assistant.react.state import LoopState

__all__ = [
    "AbortDispatch",
    "DispatchOverride",
    "FinalizeAllow",
    "FinalizeForceAction",
    "FinalizeForceCritique",
    "FinalizeGate",
    "LoopState",
    "MiddlewareChain",
    "PreDispatchResult",
    "ReactMiddleware",
]
