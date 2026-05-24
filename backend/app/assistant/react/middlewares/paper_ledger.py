"""Paper-ID ledger middleware — accumulate paper IDs from every result.

The ledger is what gives the model concrete IDs to pass to
``compare_papers`` / ``paper_qa`` / ``genie_synthesize`` in the next
iteration. Without it the model emits placeholders and the params
preflight has to invent values from the user query (less precise).

Two responsibilities:

  * After every tool result, run ``ledger.add_from_result``. Cheap
    no-op for non-retrieval tools.
  * If the result added at least one new paper AND the tool is a
    retrieval-class tool, bump ``successful_retrievals`` so the
    synthesizer's evidence-expansion-failed signal stays accurate.

The ledger itself is constructed by the loop driver (seeded from
``prior_results``) and lives on ``state.ledger``.
"""

from __future__ import annotations

from typing import Any

from app.assistant.react.middlewares.base import BaseMiddleware
from app.assistant.tools.base import ToolResult


# Mirrors the constant in react_loop.py. Centralising here lets us
# evolve the set without coupling to the loop module.
_RETRIEVAL_TOOLS: frozenset[str] = frozenset({
    "deep_search",
    "arxiv_search",
    "arxiv_import",
    "frontier_scan",
    "literature_survey",
    "pubmed",
    "inspire_hep",
    "nasa_ads",
    "semantic_scholar",
    "huggingface_search",
    "github_search",
    "papers_with_code",
    "citation_finder",
})


class PaperLedgerMiddleware(BaseMiddleware):
    """Update the paper-ID ledger after every tool result."""

    name = "paper_ledger"

    async def after_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None:
        added = state.ledger.add_from_result(result)
        if action in _RETRIEVAL_TOOLS and added > 0:
            state.successful_retrievals += 1
