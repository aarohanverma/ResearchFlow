"""TokenUsageTracker — records every LLM/embedding/image call to the DB."""

import time
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.llm.base import CompletionResult
from app.repositories.workflow import WorkflowRepository


class TokenUsageTracker:
    """Records LLM completion and embedding call costs to the ``token_usage`` table.

    Wraps ``WorkflowRepository.record_token_usage`` with typed helpers for
    the two main call types (LLM completions and embedding requests) and
    provides a rough cost estimate for the usage dashboard.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the tracker with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` passed through to the underlying
                ``WorkflowRepository``.
        """
        self._repo = WorkflowRepository(db)

    async def record_completion(
        self,
        result: CompletionResult,
        *,
        user_id: UUID | None = None,
        workflow: str = "",
        node: str = "",
    ) -> None:
        """Persist a ``TokenUsage`` row for a completed LLM call.

        Estimates the USD cost from the model's token counts and the built-in
        price table, then delegates to ``WorkflowRepository.record_token_usage``.

        Args:
            result: The ``CompletionResult`` returned by an LLM adapter call,
                containing provider, model, token counts, and latency.
            user_id: UUID of the user who triggered the call. Pass ``None``
                for background/scheduler jobs. Defaults to ``None``.
            workflow: Name of the workflow that made the call (e.g.
                ``"ingestion"``). Defaults to ``""``.
            node: Name of the workflow node within that workflow (e.g.
                ``"enrich_papers"``). Defaults to ``""``.
        """
        await self._repo.record_token_usage({
            "user_id": user_id,
            "workflow": workflow,
            "node": node,
            "provider": result.provider_used,
            "model": result.model_used,
            "call_type": "llm",
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": self._estimate_cost(result),
            "latency_ms": result.latency_ms,
            "success": True,
        })

    async def record_embedding(
        self,
        provider: str,
        model: str,
        token_count: int,
        latency_ms: int,
        *,
        user_id: UUID | None = None,
        workflow: str = "",
    ) -> None:
        """Persist a ``TokenUsage`` row for a completed embedding call.

        Records the call with ``call_type="embedding"`` and a ``cost_usd`` of
        ``0.0`` (embedding costs are tracked separately if needed).

        Args:
            provider: Embedding provider identifier (e.g. ``"gemini"``).
            model: Model name used for the embedding (e.g.
                ``"text-embedding-004"``).
            token_count: Number of tokens submitted in the embedding request.
            latency_ms: Wall-clock time for the call in milliseconds.
            user_id: UUID of the user who triggered the call. Pass ``None``
                for background/scheduler jobs. Defaults to ``None``.
            workflow: Name of the workflow that made the call. Defaults to
                ``""``.
        """
        await self._repo.record_token_usage({
            "user_id": user_id,
            "workflow": workflow,
            "provider": provider,
            "model": model,
            "call_type": "embedding",
            "input_tokens": token_count,
            "output_tokens": 0,
            "cost_usd": 0.0,  # embedding cost varies — tracked separately if needed
            "latency_ms": latency_ms,
            "success": True,
        })

    def _estimate_cost(self, result: CompletionResult) -> float:
        """Rough cost estimate for the usage dashboard."""
        # Prices per 1M tokens (approximate, USD) — update as pricing changes
        PRICES: dict[str, dict[str, float]] = {
            "gpt-4o-mini": {"in": 0.15, "out": 0.60},
            "gpt-5.4-mini": {"in": 0.40, "out": 1.60},
            "gpt-5.4": {"in": 3.00, "out": 12.00},
            "claude-haiku-4-5": {"in": 0.25, "out": 1.25},
            "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
            "claude-opus-4-6": {"in": 15.00, "out": 75.00},
        }
        p = PRICES.get(result.model_used, {"in": 1.0, "out": 4.0})
        return (result.input_tokens * p["in"] + result.output_tokens * p["out"]) / 1_000_000
