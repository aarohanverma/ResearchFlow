"""Retrieval-observability middleware — measure per-call quality.

After every retrieval-class tool dispatch, snapshot:

  * coverage_ratio    (returned / asked)
  * score_dispersion  (CV across search_score)
  * rerank_disagreement (Spearman footrule between raw_score and rerank)
  * top_score / mean_score

These signals feed both the next decision prompt (so the model sees
"deep_search: thin coverage, broaden the query") and the synthesizer's
``agent_notes`` (so the answer caveats thin or rerank-rescued
evidence). The metric computation lives in
:class:`app.assistant.retrieval_observability.RetrievalObservability`;
this middleware is the dispatch-point wiring.

Also surfaces a scratchpad warning when coverage drops below the
configured ``RETRIEVAL_THIN_COVERAGE`` so the model sees the warning
on the very next iteration without waiting for the synth pass.
"""

from __future__ import annotations

from typing import Any

from app.assistant.react.middlewares.base import BaseMiddleware
from app.assistant.tools.base import ToolResult
from app.assistant.tuning import RETRIEVAL_THIN_COVERAGE


class RetrievalObservabilityMiddleware(BaseMiddleware):
    """Record per-call retrieval quality metrics."""

    name = "retrieval_observability"

    async def after_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None:
        snap = state.retrieval_obs.record(action, params, result)
        if snap is None:
            return
        if snap.coverage_ratio < RETRIEVAL_THIN_COVERAGE:
            state.pad.think(
                f"Retrieval quality warning ({snap.render()}). "
                "Consider broadening the query or switching tools "
                "before finalizing on this evidence base."
            )
