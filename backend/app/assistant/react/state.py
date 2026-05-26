"""Mutable per-turn state for the ReAct loop.

``LoopState`` is the single source of truth a middleware chain reads
from and mutates in place. Previously the loop carried each of these as
a local variable inside ``run_react_loop``, which made every cross-
cutting concern (contradiction tracking, retrieval observability,
tool-ban policy, paper-ID ledger) leak into the function body. Hoisting
them onto a typed dataclass:

* lets each middleware reach exactly the fields it needs without the
  driver function having to plumb arguments through,
* gives tests a clean fixture to construct + assert against without
  spinning up the whole loop,
* makes the eventual deepagents migration straightforward (the
  ``LoopState`` shape maps cleanly onto a ``DeepAgentState``).

The field semantics are unchanged from the pre-refactor loop — this is
a structural extraction, not a behavioral one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.assistant.claim_ledger import ClaimLedger
from app.assistant.contradiction import ContradictionLedger
from app.assistant.react.investigation_plan import InvestigationPlan
from app.assistant.retrieval_observability import RetrievalObservability
from app.assistant.scratchpad import Scratchpad
from app.assistant.tools.base import ToolContext, ToolResult


@dataclass
class LoopState:
    """Single per-turn state object the middleware chain mutates.

    The fields fall into three groups:

    * **Inputs (read-only)** — ``query``, ``initial_plan_actions``,
      ``prior_results``, ``memory_view``, ``research_brief_text``,
      ``active_context``, ``config``, ``ctx``, ``ctx_factory``,
      ``should_cancel``, ``publish``. Set by the loop driver at
      construction.
    * **Working state (mutated by middleware)** — ``pad``,
      ``new_results``, ``ledger``, ``contradictions``,
      ``retrieval_obs``, ``tool_fail_counts``, ``banned_tools``,
      ``tool_failures``, ``successful_retrievals``,
      ``semantic_check_done``, ``iteration_count``,
      ``forced_critique_count``, ``forced_counter_searches``.
      Each middleware reads/writes a focused subset.
    * **Per-iteration (reset each iteration)** — ``current_action``,
      ``current_params``, ``current_rationale``, ``is_last_iteration``.
      Set by the driver before walking the chain; cleared after the
      dispatch resolves.

    The dataclass is *mutable on purpose*. Middleware composition is
    sequential, and copy-on-write would be wasteful for a per-turn
    object that nothing outside the loop sees.
    """

    # ── Inputs (driver-set) ─────────────────────────────────────────
    query: str
    initial_plan_actions: list[str]
    prior_results: dict[str, ToolResult]
    memory_view: dict[str, Any]
    research_brief_text: str
    active_context: dict[str, Any] | None
    ctx: ToolContext | None
    ctx_factory: Any
    should_cancel: Any
    publish: Any
    config: Any                       # ReactConfig — forward-declared
    deadline: float                   # monotonic timestamp

    # ── Working state ───────────────────────────────────────────────
    pad: Scratchpad = field(default_factory=Scratchpad)
    plan: InvestigationPlan = field(default_factory=InvestigationPlan)
    new_results: dict[str, ToolResult] = field(default_factory=dict)
    ledger: Any = None                # PaperLedger — set in driver
    contradictions: ContradictionLedger = field(default_factory=ContradictionLedger)
    # Strong-claim ledger powering full-paper verification. Populated
    # incrementally by FullPaperVerificationMiddleware.after_tool and
    # consulted at finalize to force paper_qa rounds on any strong
    # claim whose source is the abstract/snippet (not the chunked
    # body). See app.assistant.claim_ledger for the data model.
    claim_ledger: ClaimLedger = field(default_factory=ClaimLedger)
    # One-shot per turn: the full-paper gate forces at most this many
    # paper_qa rounds at finalize before giving up and labelling the
    # remaining strong claims as ``unverifiable`` so the synth can
    # caveat them. Bounded so a torrent of strong claims can't blow
    # the iteration budget at the end of a turn.
    forced_paper_qa: int = 0
    retrieval_obs: RetrievalObservability = field(default_factory=RetrievalObservability)
    tool_fail_counts: dict[str, int] = field(default_factory=dict)
    banned_tools: set[str] = field(default_factory=set)
    tool_failures: int = 0
    successful_retrievals: int = 0
    # Per-tool TOTAL invocation counter (successes + failures) for the
    # per-turn cap. Distinct from ``tool_fail_counts`` (failures only)
    # because a tool that succeeded 5 times this turn is also worth
    # banning further calls of — the planner is plausibly stuck in a
    # loop on it. Mirrors the LangChain ``ToolCallLimitMiddleware``
    # ``run_limit`` semantics: scoped to the current turn, resets on
    # next turn via fresh ``LoopState``.
    tool_invocation_counts: dict[str, int] = field(default_factory=dict)
    semantic_check_done: bool = False
    iteration_count: int = 0
    completed_normally: bool = False
    # Counters used by middlewares to track their own one-shot
    # interventions (avoid forcing the same critique / counter-search
    # again on every iteration).
    forced_critiques: int = 0
    forced_counter_searches: int = 0

    # ── Per-iteration ───────────────────────────────────────────────
    is_last_iteration: bool = False

    # ── Subagent context (only set when this loop IS a subagent) ────
    # Recursion depth — 0 for the top-level RA loop, ≥1 inside nested
    # subagents. We refuse to spawn a subagent when depth > 0 so a
    # subagent cannot itself delegate; that would defeat context
    # quarantine and risk runaway iteration budget. The decision
    # prompt also hides the subagent catalog at depth > 0 so the
    # model isn't even tempted.
    subagent_depth: int = 0
    # When non-None, this loop is running as a named subagent. The
    # role prompt is injected into the decision prompt's system
    # message so the model sees a single coherent role instead of
    # the parent's generic prompt + a role string buried in the
    # query field.
    subagent_role: str | None = None

    # ── Helpers ─────────────────────────────────────────────────────

    def time_remaining(self) -> float:
        """Seconds before the wall-clock deadline trips."""
        return max(0.0, self.deadline - time.monotonic())

    def iterations_remaining(self) -> int:
        """How many ReAct iterations are left in the budget."""
        return max(0, int(self.config.max_iterations) - self.iteration_count)

    def merged_results(self) -> dict[str, ToolResult]:
        """Union of initial-plan results + this loop's accumulated
        results. Middlewares that need to scan the full evidence base
        (contradiction detector, observability) use this view."""
        return {**(self.prior_results or {}), **self.new_results}

    def record_tool_failure(self, action: str, *, ban_cap: int) -> None:
        """Bump the per-tool failure counter and ban the tool when it
        crosses the cap. Centralised so every middleware that observes
        a failure (param-validation, dispatch error, branch failure)
        produces the same accounting."""
        self.tool_failures += 1
        self.tool_fail_counts[action] = self.tool_fail_counts.get(action, 0) + 1
        if self.tool_fail_counts[action] >= ban_cap:
            self.banned_tools.add(action)

    def publish_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Safe publish — silently swallows publish errors so a broken
        SSE bus never aborts the loop."""
        if not self.publish:
            return
        try:
            self.publish(kind, payload)
        except Exception:  # noqa: BLE001 — publish must never abort the loop
            pass
