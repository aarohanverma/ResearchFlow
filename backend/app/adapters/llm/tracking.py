"""Tracking wrapper around any :class:`LLMAdapter`.

Records every completion call to the ``token_usage`` table so the Settings
page can show a per-day input/output token breakdown. Reads the current user,
workflow, and node names from :mod:`app.core.tracking` context vars; recording
is fire-and-forget so a tracking failure can never break an LLM call.

Wraps:
    * :meth:`complete` — records exact input/output tokens from CompletionResult
    * :meth:`complete_structured` — same; falls back to text-length estimates
    * :meth:`stream` — yields tokens unchanged and estimates usage at end-of-stream
    * :meth:`complete_with_tools` — records the final CompletionResult

Estimates for streaming use a simple ``len(text) / 4`` heuristic — good enough
to give the user a rough daily total. Non-streaming paths record exact counts
returned by the provider.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.adapters.llm.base import CompletionResult, LLMAdapter
from app.core.tracking import current_node, current_user_id, current_workflow

log = logging.getLogger(__name__)


# Rough USD per 1K tokens — kept here so we don't depend on TokenUsageTracker
# (which would require a DB session that isn't available at LLM-call time).
# These are illustrative; users on the dashboard see token counts as the
# primary signal and cost as a secondary estimate.
_PRICE_PER_1K = {
    # OpenAI
    "gpt-4o":         {"in": 0.0025, "out": 0.010},
    "gpt-4o-mini":    {"in": 0.00015, "out": 0.0006},
    "gpt-5.4":        {"in": 0.0025, "out": 0.010},
    "gpt-5.4-mini":   {"in": 0.00015, "out": 0.0006},
    # Anthropic
    "claude-opus-4-7":   {"in": 0.015, "out": 0.075},
    "claude-sonnet-4-6": {"in": 0.003, "out": 0.015},
    "claude-haiku-4-5":  {"in": 0.001, "out": 0.005},
    # Google
    "gemini-2.0-flash": {"in": 0.000075, "out": 0.0003},
}


def _estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    """Estimate USD cost for a completion. Returns 0 if model unknown."""
    p = _PRICE_PER_1K.get(model)
    if not p:
        # Cheap fallback: average pricing
        return (in_tokens + out_tokens) / 1000 * 0.002
    return in_tokens / 1000 * p["in"] + out_tokens / 1000 * p["out"]


async def _record_usage(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    success: bool = True,
) -> None:
    """Insert one ``TokenUsage`` row. Errors are swallowed and logged."""
    if input_tokens == 0 and output_tokens == 0:
        return  # nothing to record
    from app.db.session import async_session_factory
    from app.repositories.workflow import WorkflowRepository

    try:
        async with async_session_factory() as db:
            repo = WorkflowRepository(db)
            await repo.record_token_usage({
                "user_id":       current_user_id.get(),
                "workflow":      current_workflow.get() or None,
                "node":          current_node.get() or None,
                "provider":      provider,
                "model":         model,
                "call_type":     "llm",
                "input_tokens":  input_tokens,
                "output_tokens": output_tokens,
                "cost_usd":      round(_estimate_cost(model, input_tokens, output_tokens), 6),
                "latency_ms":    latency_ms,
                "success":       success,
            })
            await db.commit()
    except Exception as exc:
        log.debug("token-usage record failed: %s", exc)


class TrackingLLMAdapter:
    """Decorator wrapping any LLMAdapter to record token usage after each call.

    Forwards the adapter's class-level model identifiers and method signatures.
    Recording happens via :func:`asyncio.create_task` so it never blocks the
    caller, and is best-effort — a recording failure is logged at DEBUG and
    swallowed.
    """

    def __init__(self, inner: LLMAdapter) -> None:
        """Wrap an existing ``LLMAdapter`` instance with token-usage tracking."""
        self._inner = inner
        self.provider_id = inner.provider_id
        self.cheap_model = inner.cheap_model
        self.quality_model = inner.quality_model
        self.reasoning_model = inner.reasoning_model

    # ── Pass-through to inner adapter for any other attribute access ──────────
    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        """Delegate any attribute not explicitly defined here to the wrapped adapter."""
        return getattr(self._inner, name)

    async def complete(self, *args: Any, **kwargs: Any) -> CompletionResult:
        """Forward to inner.complete and record usage."""
        result = await self._inner.complete(*args, **kwargs)
        asyncio.create_task(_record_usage(
            provider=result.provider_used,
            model=result.model_used,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        ))
        return result

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        model: str,
        schema: type,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward to inner.complete_structured. No exact token data is returned
        by this path so we estimate from message text length (~4 chars/token)."""
        result = await self._inner.complete_structured(messages, model, schema, **kwargs)
        try:
            in_chars = sum(len(m.get("content", "")) for m in messages)
            out_chars = len(str(result))
            asyncio.create_task(_record_usage(
                provider=self._inner.provider_id,
                model=model,
                input_tokens=in_chars // 4,
                output_tokens=out_chars // 4,
                latency_ms=0,
            ))
        except Exception:
            pass
        return result

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Forward to inner.stream and estimate token usage at end-of-stream.

        We accumulate the streamed text and record an estimate (~4 chars/token)
        when the stream finishes. Provider streaming APIs do not consistently
        expose token counts, so this is an approximation — useful for
        dashboards but not billing-grade.
        """
        in_chars = sum(len(m.get("content", "")) for m in messages)
        out_chars = 0
        try:
            async for token in self._inner.stream(messages, model, **kwargs):
                out_chars += len(token)
                yield token
        finally:
            asyncio.create_task(_record_usage(
                provider=self._inner.provider_id,
                model=model,
                input_tokens=in_chars // 4,
                output_tokens=out_chars // 4,
                latency_ms=0,
            ))

    async def complete_with_tools(self, *args: Any, **kwargs: Any) -> CompletionResult:
        """Forward to inner.complete_with_tools and record final usage."""
        result = await self._inner.complete_with_tools(*args, **kwargs)
        asyncio.create_task(_record_usage(
            provider=result.provider_used,
            model=result.model_used,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
        ))
        return result
