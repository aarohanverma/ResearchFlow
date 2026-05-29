"""Concrete middlewares assembled into the default ReAct chain.

Order matters — the default chain (see ``default_chain_factory``) is::

    1. ParamPreflight             (placeholder strip + auto-fill from ledger)
    2. ToolBan                    (block banned tools, redirect or abort)
    3. HitlGate                   (pause for user approval, e.g. genie_synthesize)
    4. DiminishingReturns         (skip identical-param redo; stop on no-new-IDs)
    5. PaperLedger                (accumulate paper IDs from results)
    6. RetrievalObservability     (record per-call coverage / dispersion / rerank)
    7. CriticGate                 (force critique before too-early finalize)
    8. ContradictionDetector      (lexical + numeric + LLM semantic; adaptive force)
    9. FullPaperVerification      (force paper_qa on abstract-only strong claims at finalize)

Earlier middlewares get the first say on ``before_tool`` (param fixing
runs before ban checks; ban checks run before the HITL pause and the
redundancy check). ``gate_finalize`` walks the same order — critique gate
fires before contradiction-forced counter-search (so a sufficient-evidence
turn isn't kept open just to verify a soft contradiction), and
full-paper verification fires last.

Each middleware is independently testable and disable-able (env flag
or per-call list omission). New cross-cutting concerns become one file
here.
"""

from app.assistant.react.middlewares.base import NoopMiddleware
from app.assistant.react.middlewares.contradiction_mw import ContradictionMiddleware
from app.assistant.react.middlewares.critic_gate import CriticGateMiddleware
from app.assistant.react.middlewares.diminishing_returns import DiminishingReturnsMiddleware
from app.assistant.react.middlewares.full_paper_gate import FullPaperVerificationMiddleware
from app.assistant.react.middlewares.hitl_gate import HitlGateMiddleware
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
        # HITL gate sits after param-fixing / tool-ban so it never
        # pauses the user on a dispatch the downstream chain would
        # have aborted anyway. Runs before redundancy/ledger so the
        # ledger view in the gate preview is the same the model saw.
        HitlGateMiddleware(),
        DiminishingReturnsMiddleware(),
        PaperLedgerMiddleware(),
        RetrievalObservabilityMiddleware(),
        CriticGateMiddleware(),
        ContradictionMiddleware(enable_semantic_llm=enable_semantic_contradiction),
        # Full-paper verification fires at finalize: it inspects every
        # strong claim the loop is about to ship and forces a
        # ``paper_qa`` round on any that lack chunk-level evidence.
        # Lives at the end so contradiction-forced counter-searches
        # land first (they may resolve a strong claim outright,
        # avoiding a redundant paper_qa pass).
        FullPaperVerificationMiddleware(),
    ]


__all__ = [
    "ContradictionMiddleware",
    "CriticGateMiddleware",
    "DiminishingReturnsMiddleware",
    "FullPaperVerificationMiddleware",
    "HitlGateMiddleware",
    "NoopMiddleware",
    "PaperLedgerMiddleware",
    "ParamPreflightMiddleware",
    "RetrievalObservabilityMiddleware",
    "ToolBanMiddleware",
    "default_chain_factory",
]
