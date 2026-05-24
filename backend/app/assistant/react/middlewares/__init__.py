"""Concrete middlewares assembled into the default ReAct chain.

Order matters — the default chain is::

    1. ParamPreflight             (placeholder strip + auto-fill from ledger)
    2. ToolBan                    (block banned tools, redirect or abort)
    3. DiminishingReturns         (skip identical-param redo; stop on no-new-IDs)
    4. RetrievalObservability     (record per-call coverage / dispersion / rerank)
    5. ContradictionDetector      (lexical + numeric + LLM semantic; adaptive force)
    6. CriticGate                 (force critique before too-early finalize)
    7. PaperLedger                (accumulate paper IDs from results)

Earlier middlewares get the first say on ``before_tool`` (param fixing
runs before ban checks; ban checks run before redundancy checks).
``gate_finalize`` walks the same order — critique gate fires before
contradiction-forced-counter-search so a sufficient-evidence turn
isn't kept open just to verify a soft contradiction.

Each middleware is independently testable and disable-able (env flag
or per-call list omission). New cross-cutting concerns become one file
here.
"""

from app.assistant.react.middlewares.base import NoopMiddleware
from app.assistant.react.middlewares.contradiction_mw import ContradictionMiddleware
from app.assistant.react.middlewares.critic_gate import CriticGateMiddleware
from app.assistant.react.middlewares.diminishing_returns import DiminishingReturnsMiddleware
from app.assistant.react.middlewares.observability_mw import RetrievalObservabilityMiddleware
from app.assistant.react.middlewares.paper_ledger import PaperLedgerMiddleware
from app.assistant.react.middlewares.param_preflight import ParamPreflightMiddleware
from app.assistant.react.middlewares.tool_ban import ToolBanMiddleware


def default_chain_factory(
    *,
    enable_semantic_contradiction: bool = True,
) -> list:
    """Build the default ordered middleware list for a ReAct loop.

    The factory exists so callers can:
      * tune one knob (``enable_semantic_contradiction``) without
        re-deriving the whole order,
      * swap one middleware for a namespace-specific variant later
        without touching the loop driver,
      * spin up a stripped chain for tests by passing fewer items.
    """
    # Order matters:
    #   * ``before_tool``: preflight before ban, ban before redundancy
    #     check. Each gate may short-circuit the rest.
    #   * ``after_tool``: ledger first (subsequent middlewares read it),
    #     then observability, then contradiction (whose semantic LLM
    #     check needs ledger size ≥ 4), then diminishing returns.
    #   * ``gate_finalize``: critique gate first (force one critique on
    #     too-early finalize), then contradiction (force one counter-
    #     search on a high-confidence open signal). Critique
    #     middleware must appear BEFORE contradiction in this list
    #     because the chain returns the first non-allow gate.
    return [
        ParamPreflightMiddleware(),
        ToolBanMiddleware(),
        DiminishingReturnsMiddleware(),
        PaperLedgerMiddleware(),
        RetrievalObservabilityMiddleware(),
        CriticGateMiddleware(),
        ContradictionMiddleware(enable_semantic_llm=enable_semantic_contradiction),
    ]


__all__ = [
    "ContradictionMiddleware",
    "CriticGateMiddleware",
    "DiminishingReturnsMiddleware",
    "NoopMiddleware",
    "PaperLedgerMiddleware",
    "ParamPreflightMiddleware",
    "RetrievalObservabilityMiddleware",
    "ToolBanMiddleware",
    "default_chain_factory",
]
