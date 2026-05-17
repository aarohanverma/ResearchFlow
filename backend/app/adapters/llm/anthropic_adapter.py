"""Anthropic LLM adapter — Haiku (cheap), Sonnet (quality), Opus (reasoning)."""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic as anthropic_sdk
import httpx
from anthropic import AsyncAnthropic
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.adapters.llm.base import CompletionResult, LLMAdapter
from app.core.config import settings


def _is_retryable_anthropic(exc: Exception) -> bool:
    """Retry only on transient/server errors. Auth/bad-request never retried."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, anthropic_sdk.RateLimitError):
        return True
    if isinstance(exc, anthropic_sdk.APIConnectionError):
        return True
    if isinstance(exc, anthropic_sdk.APIStatusError) and getattr(exc, "status_code", 0) >= 500:
        return True
    return False


async def _call_with_retry(coro_factory):
    """Run a coroutine factory with 3-attempt jittered exponential backoff."""
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable_anthropic),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=8),
        reraise=True,
    ):
        with attempt:
            return await coro_factory()

# Thinking-capable models where we pass thinking.budget_tokens
_THINKING_MODELS = {
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
}


class AnthropicAdapter(LLMAdapter):
    """LLM adapter for Anthropic's Messages API.

    Uses Claude Haiku (cheap), Sonnet (quality), and Opus (reasoning) model
    tiers. Supports extended thinking via ``reasoning_effort`` for thinking-
    capable models defined in ``_THINKING_MODELS``.
    """

    provider_id = "anthropic"
    cheap_model = "claude-haiku-4-5"
    quality_model = "claude-sonnet-4-6"
    reasoning_model = "claude-opus-4-6"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Anthropic async client.

        Args:
            api_key: Anthropic API key. Falls back to
                ``settings.anthropic_api_key`` if not provided.
        """
        self._client = AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)

    def _build_thinking(self, model: str, reasoning_effort: str | None) -> dict | None:
        """Convert our effort string to Anthropic's thinking budget."""
        if model not in _THINKING_MODELS or reasoning_effort is None:
            return None
        budget_map = {"low": 2000, "medium": 8000, "high": 16000, "xhigh": 32000}
        return {"type": "enabled", "budget_tokens": budget_map.get(reasoning_effort, 8000)}

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> CompletionResult:
        """Send a messages request to Anthropic and return the full response.

        Separates system messages from the conversation and passes them via
        Anthropic's ``system`` parameter. Enables extended thinking for
        models in ``_THINKING_MODELS`` when ``reasoning_effort`` is set.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
                A message with ``role="system"`` is extracted and passed
                separately.
            model: Anthropic model identifier.
            reasoning_effort: Effort level for thinking-capable models
                (``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate. Defaults to 4096.
            response_format: Unused by this adapter; present for interface
                compatibility.

        Returns:
            A ``CompletionResult`` containing the generated text (thinking
            blocks excluded), token counts, model/provider identifiers,
            and latency.
        """
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": user_messages,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        # Build thinking before setting max_tokens so the budget is known.
        # Anthropic requires max_tokens > thinking.budget_tokens; when the
        # caller passes None (no cap) we still need a concrete integer.
        thinking = self._build_thinking(model, reasoning_effort)
        if thinking:
            kwargs["thinking"] = thinking
            # Leave at least 4096 tokens beyond the thinking budget for the
            # actual response text.
            min_needed = thinking["budget_tokens"] + 4096
            kwargs["max_tokens"] = max(max_tokens or 0, min_needed)
        else:
            kwargs["max_tokens"] = max_tokens or 8192

        start = time.monotonic()
        resp = await _call_with_retry(
            lambda: self._client.messages.create(**kwargs)
        )
        latency = int((time.monotonic() - start) * 1000)

        # Extract text (skip thinking blocks)
        text = "".join(
            b.text for b in resp.content
            if hasattr(b, "text")
        )
        return CompletionResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model_used=model,
            provider_used=self.provider_id,
            latency_ms=latency,
        )

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        model: str,
        schema: type,
        *,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Request a completion and parse the response text as JSON.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: Anthropic model identifier.
            schema: Pydantic model or type describing the expected JSON
                structure (used for documentation; validation is left to the
                caller).
            reasoning_effort: Effort level for thinking-capable models.

        Returns:
            Parsed JSON response as a Python dict.
        """
        # Anthropic has no native JSON mode: we must prompt for it.  Inject a
        # system preamble when the word "json" isn't already in the messages so
        # the model knows to return raw JSON without prose or markdown fences.
        has_json_word = any("json" in (m.get("content") or "").lower() for m in messages)
        if not has_json_word:
            messages = [
                {"role": "system", "content": "Return only valid JSON. No markdown fences, no prose."},
                *messages,
            ]
        result = await self.complete(messages, model, reasoning_effort=reasoning_effort)
        text = result.text.strip()
        # Strip markdown fences that some models emit even when asked not to.
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        return json.loads(text)

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream text chunks from an Anthropic messages request.

        Separates system messages and passes them via the ``system`` parameter.
        Enables extended thinking for models in ``_THINKING_MODELS``.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: Anthropic model identifier.
            reasoning_effort: Effort level for thinking-capable models.
            max_tokens: Maximum tokens to generate. Defaults to 8192.

        Yields:
            String chunks of the generated text from the streaming response.
        """
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]

        thinking = self._build_thinking(model, reasoning_effort)

        # Mirror the complete() max_tokens logic: when thinking is enabled,
        # Anthropic requires max_tokens > budget_tokens.  Using max_tokens or 8192
        # would cause a validation error when budget_tokens (e.g. 16000) exceeds
        # the default.
        if thinking:
            min_needed = thinking["budget_tokens"] + 4096
            effective_max = max(max_tokens or 0, min_needed)
        else:
            effective_max = max_tokens or 8192

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": effective_max,
            "messages": user_messages,
        }
        if system:
            kwargs["system"] = system
        if thinking:
            kwargs["thinking"] = thinking

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
