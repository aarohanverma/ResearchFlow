"""Sub-agent system — proper context quarantine.

The pre-refactor ``fanout`` action ran multiple tool calls concurrently
but every branch shared the parent's prompt, ledger, and scratchpad.
That gave latency parallelism but **no context quarantine**: dozens of
intermediate observations still ended up in the parent's context.

Real subagents fix this. Each :class:`SubAgentSpec` defines:

  * **Role + system prompt** — specialised behaviour ("you are a
    citation auditor", "you are a baseline comparator").
  * **Tool subset** — a restricted catalog (a citation auditor doesn't
    need ``arxiv_import``).
  * **Iteration / deadline cap** — usually tighter than the parent.
  * **Structured response format** — the subagent returns a single
    JSON-shaped summary; the parent gets that summary, not the
    subagent's intermediate observations.

The parent dispatches a subagent via ``action="subagent"`` with a
``subagent_name`` + ``task`` payload. The loop driver:

  1. Looks up the spec in :data:`SUBAGENT_REGISTRY`.
  2. Spawns a nested :class:`LoopState` with a focused query (the
     task), restricted tool catalog, fresh scratchpad, and shared
     ``ctx_factory`` (so DB sessions stay scoped correctly).
  3. Runs the nested loop to completion.
  4. Returns a single :class:`SubAgentResult` to the parent containing
     the structured summary + minimal evidence pointers.
  5. Parent records the summary as a single Observation; the
     intermediate steps stay in the subagent's own scratchpad
     (persisted separately on the message payload for auditability).

This is the deepagents subagent pattern, implemented on our internal
loop without taking the dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.assistant.scratchpad import Scratchpad
from app.assistant.tools.base import ToolResult

log = logging.getLogger(__name__)


# ── Spec ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SubAgentSpec:
    """Static configuration for one named subagent.

    Specs are immutable so a single spec can be shared across many
    concurrent loops. Per-invocation state (scratchpad, results) lives
    on the nested :class:`LoopState`, not on the spec.
    """

    name: str
    description: str          # how the parent decides when to delegate
    role_prompt: str          # injected into the nested loop's system prompt
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    max_iterations: int = 4
    deadline_seconds: float = 45.0
    # Structured response schema the parent expects back. A list of
    # field names + descriptions; the subagent's final "report" must
    # contain these keys.
    response_schema: tuple[tuple[str, str], ...] = ()


# ── Result ───────────────────────────────────────────────────────────────────


@dataclass
class SubAgentResult:
    """Single summary the parent observes; never the raw intermediate steps."""

    subagent_name: str
    summary: str
    structured: dict[str, Any] = field(default_factory=dict)
    paper_ids_surfaced: list[str] = field(default_factory=list)
    iterations: int = 0
    completed_normally: bool = False
    scratchpad: Scratchpad | None = None     # persisted alongside the parent's pad

    def to_tool_result(self) -> ToolResult:
        """Project into a ``ToolResult`` so the parent's after_tool
        pipeline (ledger, observability, contradiction) treats the
        subagent's findings exactly like any other tool result."""
        out: dict[str, Any] = {
            "subagent": self.subagent_name,
            "summary": self.summary,
            "iterations": self.iterations,
            "completed_normally": self.completed_normally,
        }
        if self.structured:
            out["structured"] = self.structured
        if self.paper_ids_surfaced:
            # Mirror the retrieval-tool shape so the paper-ledger
            # middleware picks the IDs up automatically — no special
            # case needed for subagent results.
            out["papers"] = [
                {"paper_id": pid, "title": ""} for pid in self.paper_ids_surfaced
            ]
        return ToolResult(
            output=out,
            summary=self.summary[:600] or f"{self.subagent_name} subagent ran",
        )


# ── Registry — pre-defined research subagents ────────────────────────────────


_RESEARCH_TOOLS = frozenset({
    "deep_search", "literature_survey", "arxiv_search", "arxiv_import",
    "frontier_scan", "citation_finder", "semantic_scholar", "concept_explain",
    "research_trends", "wikipedia",
})

_COMPARISON_TOOLS = frozenset({
    "compare_papers", "paper_qa", "concept_explain", "deep_search",
    "citation_finder",
})

_CRITIQUE_TOOLS = frozenset({
    "citation_finder", "paper_qa", "deep_search",
})

_BASELINE_TOOLS = frozenset({
    "deep_search", "literature_survey", "papers_with_code",
    "github_search", "research_trends",
})

_CONTRADICTION_TOOLS = frozenset({
    "deep_search", "citation_finder", "literature_survey", "paper_qa",
})


SUBAGENT_REGISTRY: dict[str, SubAgentSpec] = {
    "researcher": SubAgentSpec(
        name="researcher",
        description=(
            "Run a focused multi-step literature retrieval on a specific "
            "sub-question. Returns a paper-ID list + a tight summary; the "
            "parent doesn't see the dozens of intermediate observations."
        ),
        role_prompt=(
            "You are a focused literature researcher. Your job is to find "
            "the strongest available evidence on the task you were given — "
            "not to write the final answer. Cite paper IDs. Stop as soon "
            "as the evidence base is sufficient; do NOT keep searching for "
            "marginal returns."
        ),
        allowed_tools=_RESEARCH_TOOLS,
        max_iterations=4,
        deadline_seconds=45.0,
        response_schema=(
            ("summary", "2-4 sentence summary of what you found"),
            ("paper_ids", "List of paper IDs that ground the summary"),
            ("open_questions", "Optional list of follow-up questions"),
        ),
    ),
    "comparator": SubAgentSpec(
        name="comparator",
        description=(
            "Compare a small set of papers / approaches along specific "
            "dimensions (factuality, latency, scalability, etc.). Returns "
            "a structured comparison table; the parent doesn't need to "
            "see each per-paper query."
        ),
        role_prompt=(
            "You are a head-to-head comparator. Given a set of papers or "
            "approaches, produce a tight comparison along the requested "
            "dimensions. Be specific about which side wins where and WHY. "
            "Do NOT invent dimensions the task didn't ask for."
        ),
        allowed_tools=_COMPARISON_TOOLS,
        max_iterations=3,
        deadline_seconds=40.0,
        response_schema=(
            ("summary", "Headline of the comparison"),
            ("rows", "List of {dimension, A_verdict, B_verdict, evidence}"),
        ),
    ),
    "critic": SubAgentSpec(
        name="critic",
        description=(
            "Adversarial reviewer. Take the current evidence base and look "
            "for the strongest reasons NOT to believe the parent's working "
            "thesis: missing baselines, weak controls, unverified citations, "
            "selection bias. Returns a structured objection list."
        ),
        role_prompt=(
            "You are an adversarial reviewer. Look for the strongest reasons "
            "NOT to believe the working thesis: missing baselines, weak "
            "controls, unverified citations, selection bias, replication "
            "concerns. Be specific. Cite the papers you object to. Do NOT "
            "soften your objections to be polite — the parent will weigh "
            "them; honesty here is the value."
        ),
        allowed_tools=_CRITIQUE_TOOLS,
        max_iterations=3,
        deadline_seconds=35.0,
        response_schema=(
            ("summary", "One sentence verdict: defensible / contested / weak"),
            ("objections", "Ordered list of objections with evidence pointers"),
        ),
    ),
    "baseline_finder": SubAgentSpec(
        name="baseline_finder",
        description=(
            "Find the strongest *fair* baseline the parent's proposal must "
            "beat. Returns the baseline name, why it's the right comparison, "
            "and what's currently known about its performance."
        ),
        role_prompt=(
            "You find the strongest fair baseline for a proposal. NOT a "
            "strawman, NOT a toy version — the version a serious reviewer "
            "would expect to see compared against. Name it, justify it, "
            "and surface what's known about its performance."
        ),
        allowed_tools=_BASELINE_TOOLS,
        max_iterations=3,
        deadline_seconds=35.0,
        response_schema=(
            ("summary", "Why this is the right baseline"),
            ("baseline", "Concrete baseline name + the strongest variant"),
            ("known_performance", "What's reported about it"),
        ),
    ),
    "contradiction_hunter": SubAgentSpec(
        name="contradiction_hunter",
        description=(
            "Take a specific claim and actively look for counter-evidence: "
            "papers that contradict it, fail to replicate, report opposite "
            "conclusions, or only support a weaker version."
        ),
        role_prompt=(
            "You hunt for counter-evidence to a specific claim. Find papers "
            "that contradict it, fail to replicate it, report opposite "
            "conclusions, or only support a weaker version. Distinguish "
            "outright contradiction from a narrower scope. Be honest if "
            "you can't find counter-evidence — that's a valid finding."
        ),
        allowed_tools=_CONTRADICTION_TOOLS,
        max_iterations=3,
        deadline_seconds=35.0,
        response_schema=(
            ("summary", "Verdict: contradicted / qualified / unchallenged"),
            ("counter_evidence", "List of papers that challenge the claim"),
        ),
    ),
}


def get_subagent(name: str) -> SubAgentSpec | None:
    """Look up a spec by name. Returns ``None`` for unknown names — the
    loop driver renders a clear observation in that case so the model
    doesn't keep retrying a typo."""
    return SUBAGENT_REGISTRY.get(name)


def describe_subagents_for_prompt() -> str:
    """Render the registry into a compact catalog block for the
    decision prompt. Each line: ``- name: description``."""
    if not SUBAGENT_REGISTRY:
        return "(no subagents registered)"
    return "\n".join(
        f"  - {s.name}: {s.description}"
        for s in SUBAGENT_REGISTRY.values()
    )
