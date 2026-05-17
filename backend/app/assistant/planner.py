"""Heuristic planner — preserves the prior keyword-based plan logic.

Maps user intent to an ordered list of PlannedStep entries the orchestrator
executes. Replaced in M1 by an LLM-driven planner that reads the tool
registry's JSON schemas; until then this keeps behaviour parity with the
previous ``_infer_plan`` while writing structured plans the new orchestrator
can execute uniformly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlannedStep:
    """A single tool invocation the orchestrator should execute."""

    tool: str
    title: str
    params: dict[str, Any]
    rationale: str = ""
    parallel: bool = False  # True → run concurrently with other parallel steps in the same wave


@dataclass
class Plan:
    """Ordered execution plan produced by a planner."""

    rationale: str
    steps: list[PlannedStep]
    actions: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.6


# Positive intent triggers — these enable optional heavy steps.
_FRESH_IMPORT_KEYWORDS = (
    "import", "fresh", "new papers", "latest paper", "recent paper",
    "pull from arxiv", "fetch from arxiv", "discover new",
)
_GENIE_KEYWORDS = (
    "hypothesis", "hypotheses", "genie synthesis", "novel idea",
    "research idea", "research direction", "proposal", "new experiment",
)
# Graph build is heavy and side-effecting — only run on EXPLICIT request.
# "concept map" alone is too weak (LLM often writes it in explanations).
_GRAPH_BUILD_KEYWORDS = (
    "build graph", "build the graph", "rebuild graph", "refresh graph",
    "update graph", "recompute graph", "graph taxonomy", "rebuild taxonomy",
    "build knowledge graph",
)
_FRONTIER_KEYWORDS = (
    "frontier", "what's new", "what is new", "emerging", "cutting edge",
    "cutting-edge", "trending",
)

# Negative directives — when present in the user message we drop matching
# heavy/optional steps even if a positive keyword fired earlier. Order
# matters: we match the longest variants first so "no graph build" wins
# over a bare "graph".
_NEGATIVE_PATTERN = re.compile(
    r"\b(?:skip|don'?t|do\s+not|without|no\s+need\s+to|avoid|exclude|no)\b\s+"
    r"(?:the\s+)?(?:running\s+)?(?:doing\s+)?(?P<target>[a-z_\-\s]+?)\b",
    re.IGNORECASE,
)

# Map matched negative-target phrases → tool name.
_NEGATIVE_TARGET_TO_TOOL: dict[str, str] = {
    "graph": "graph_build",
    "graph build": "graph_build",
    "graph builds": "graph_build",
    "graph building": "graph_build",
    "graph search": "graph_build",          # user's own phrasing in the transcript
    "knowledge graph": "graph_build",
    "graph_build": "graph_build",
    "genie": "genie_synthesize",
    "genie synthesis": "genie_synthesize",
    "synthesis": "genie_synthesize",
    "hypotheses": "genie_synthesize",
    "arxiv": "arxiv_import",
    "arxiv import": "arxiv_import",
    "arxiv search": "arxiv_search",
    "arxiv fetch": "arxiv_import",
    "import": "arxiv_import",
    "imports": "arxiv_import",
    "fetching": "arxiv_import",
    "deep search": "deep_search",
    "search": "deep_search",
    "frontier": "frontier_scan",
}


def parse_negative_directives(query: str) -> set[str]:
    """Return the set of tool names the user explicitly asked NOT to run.

    Catches phrasings like "skip graph build", "no arxiv import",
    "without genie synthesis", "don't build the graph". Best-effort —
    when a match is ambiguous we err on the side of skipping (the LLM
    planner can always override with a positive plan).
    """
    skip: set[str] = set()
    if not query:
        return skip
    for m in _NEGATIVE_PATTERN.finditer(query):
        raw = (m.group("target") or "").strip().lower()
        # Strip trailing filler tokens that aren't part of the target name.
        raw = re.sub(r"\b(?:for|please|this turn|right now|here|tool|step|build|builds|search|searches)\b", " ", raw)
        raw = " ".join(raw.split())
        if not raw:
            continue
        # Try longest-substring match against the lookup so multi-word
        # negatives ("graph build") don't fall through to bare "graph".
        for phrase in sorted(_NEGATIVE_TARGET_TO_TOOL, key=len, reverse=True):
            if phrase in raw:
                skip.add(_NEGATIVE_TARGET_TO_TOOL[phrase])
                break
    return skip


def is_likely_first_turn(history: list[dict] | None) -> bool:
    """Best-effort detection of a brand-new investigation.

    The orchestrator passes recent conversation history (most recent last).
    On a first turn the assistant's only message is the system "workspace
    created" notice; nothing user-tagged exists yet.
    """
    if not history:
        return True
    user_count = sum(1 for m in history if (m.get("role") == "user"))
    # The current turn's user message is already in the history bundle, so
    # "first turn" means strictly one user message exists.
    return user_count <= 1


class HeuristicPlanner:
    """Keyword-driven planner used as the fallback when LLM planning fails.

    Conservative by default: ``deep_search`` always runs, ``arxiv_import``
    only on first turns or explicit fresh-import requests, and
    ``graph_build`` / ``genie_synthesize`` only when the user explicitly
    asks for them. Negative directives ("skip graph", "no arxiv") win
    over positive triggers.
    """

    name = "heuristic"

    def plan(
        self,
        *,
        query: str,
        namespace_key: str,
        namespace_keys: list[str],
        history: list[dict] | None = None,
    ) -> Plan:
        q = (query or "").lower()
        ns_keys = namespace_keys or [namespace_key]
        skip = parse_negative_directives(query)

        wants_fresh = any(k in q for k in _FRESH_IMPORT_KEYWORDS) or is_likely_first_turn(history)
        wants_genie = any(k in q for k in _GENIE_KEYWORDS)
        # graph_build is intentionally NOT auto-planned by the RA. Users
        # build graphs from the dedicated /graph page; the RA only consumes
        # existing graph data via deep_search's graph_retrieve stage.
        wants_frontier = any(k in q for k in _FRONTIER_KEYWORDS)

        steps: list[PlannedStep] = []
        actions: list[str] = []

        # Fresh import — only when needed. Falls back to arxiv_search (no DB
        # write) if the user asked to skip imports. NOTE: we deliberately do
        # NOT pass namespace_keys; the tools default to cross-arXiv search so
        # interdisciplinary queries surface results from other categories
        # (e.g. molecular GNNs in q-bio + cs.LG, not just cs.AI).
        if wants_fresh and "arxiv_import" not in skip:
            steps.append(PlannedStep(
                tool="arxiv_import",
                title="Import fresh arXiv candidates",
                params={
                    "query": query,
                    "namespace_key": namespace_key,
                    "max_results": 6,
                },
                rationale="First turn or user asked for fresh papers — grow corpus before retrieval.",
            ))
            actions.append("arXiv MCP fetch/import")
        elif wants_fresh and "arxiv_search" not in skip:
            # Read-only fallback so the user still sees candidate titles
            # without writing to their feed when imports were declined.
            steps.append(PlannedStep(
                tool="arxiv_search",
                title="Browse arXiv candidates (no import)",
                params={"query": query, "max_results": 8},
                rationale="User wants discovery but asked to skip imports — search only.",
            ))
            actions.append("arXiv MCP search")

        # Frontier scan is cheap — surface emerging work when asked.
        # Cross-namespace by default (frontier work is interdisciplinary).
        if wants_frontier and "frontier_scan" not in skip:
            steps.append(PlannedStep(
                tool="frontier_scan",
                title="Scan the research frontier",
                params={"limit": 8},
                rationale="User asked about frontier / emerging work — cross-namespace scan.",
            ))
            actions.append("Frontier scan")

        # Deep search — the workhorse. Always include unless skipped.
        if "deep_search" not in skip:
            steps.append(PlannedStep(
                tool="deep_search",
                title="Hybrid grounded retrieval",
                params={
                    "query": query,
                    "namespace_keys": ns_keys,
                    "limit": 8,
                    "include_arxiv_mcp": False,
                    "arxiv_max_results": 0,
                },
                rationale="Hybrid keyword+semantic+graph retrieval over the user's corpus.",
            ))
            actions.append("Deep Search")

        # graph_build deliberately not in the heuristic plan — RA leaves
        # graph construction to the dedicated /graph page. Existing graph
        # data is still consumed implicitly by deep_search.

        if wants_genie and "genie_synthesize" not in skip:
            steps.append(PlannedStep(
                tool="genie_synthesize",
                title="Queue Genie synthesis",
                params={
                    "paper_ids": [],
                    "paper_titles": [],
                    "query": query,
                },
                rationale="User asked for hypothesis/ideation — wire Genie with retrieved papers.",
            ))
            actions.append("Genie synthesis")

        # Safety net: every plan needs at least one step. If everything was
        # skipped, run a minimal deep_search so the user gets *some* answer.
        if not steps:
            steps.append(PlannedStep(
                tool="deep_search",
                title="Hybrid grounded retrieval",
                params={
                    "query": query, "namespace_keys": ns_keys, "limit": 8,
                    "include_arxiv_mcp": False, "arxiv_max_results": 0,
                },
                rationale="All optional steps were declined — running minimal deep search.",
            ))
            actions.append("Deep Search")

        rationale = "Keyword plan: deep search always, fresh import on first turns, " \
                    "heavy tools only on explicit request."
        if skip:
            rationale += f" Skipped per user directive: {', '.join(sorted(skip))}."

        return Plan(
            rationale=rationale,
            steps=steps,
            actions=actions,
            trace=[
                {"step": "intent", "summary": "Classified request and selected orchestration primitives"},
                {"step": "scope", "summary": "Using active namespace and selected topics as retrieval boundary"},
                {"step": "skip", "summary": f"Honored skip directives: {sorted(skip) or 'none'}"},
            ],
        )
