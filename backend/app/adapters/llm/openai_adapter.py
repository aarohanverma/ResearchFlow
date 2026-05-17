"""OpenAI LLM adapter — gpt-4o-mini (cheap), gpt-5.4-mini (quality), gpt-5.4 (reasoning)."""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Callable

import httpx
import openai
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.adapters.llm.base import CompletionResult, LLMAdapter, ToolCall
from app.core.config import settings

log = logging.getLogger(__name__)


def _is_retryable_llm(exc: Exception) -> bool:
    """Retry only on transient/server errors. Auth/bad-request errors are not retried."""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        # insufficient_quota = billing failure, not transient — never retry
        return "insufficient_quota" not in str(exc)
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError) and getattr(exc, "status_code", 0) >= 500:
        return True
    return False


async def _call_with_retry(coro_factory):
    """Run a no-arg async coroutine factory with exp-jitter retries.

    Three attempts with 1s → up to 8s jittered backoff. Re-raises the last
    exception if all attempts fail. Non-retryable errors propagate immediately.
    """
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable_llm),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=8),
        reraise=True,
    ):
        with attempt:
            return await coro_factory()


class OpenAIAdapter(LLMAdapter):
    """LLM adapter for OpenAI's chat completion API.

    Supports gpt-4o-mini (cheap), gpt-5.4-mini (quality), and gpt-5.4
    (reasoning) model tiers. Reasoning effort is forwarded to o-series
    and gpt-5 models via the ``reasoning_effort`` parameter.
    """

    provider_id = "openai"
    cheap_model = "gpt-4o-mini"
    quality_model = "gpt-5.4-mini"
    reasoning_model = "gpt-5.4"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the OpenAI async client.

        Args:
            api_key: OpenAI API key. Falls back to ``settings.openai_api_key``
                if not provided.
        """
        # max_retries=0: tenacity handles all retries; SDK retries would stack
        # on top and produce noisy duplicate tracebacks for the same failure.
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key, max_retries=0)

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
        """Send a chat completion request and return the full response.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: The OpenAI model name to use.
            reasoning_effort: Effort level for o-series/gpt-5 models
                (``"low"``, ``"medium"``, ``"high"``). Ignored for other models.
            temperature: Sampling temperature. Defaults to the model's default
                when not set.
            max_tokens: Maximum tokens to generate in the completion.
            response_format: Dict controlling the response format, e.g.
                ``{"type": "json_object"}`` for JSON mode.

        Returns:
            A ``CompletionResult`` containing the generated text, token counts,
            model/provider identifiers, and latency.
        """
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format
        # Reasoning effort — for o-series / gpt-5 models
        if reasoning_effort and model in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "o3", "o4-mini"):
            kwargs["reasoning_effort"] = reasoning_effort

        start = time.monotonic()
        resp = await _call_with_retry(
            lambda: self._client.chat.completions.create(**kwargs)
        )
        latency = int((time.monotonic() - start) * 1000)

        usage = resp.usage
        return CompletionResult(
            text=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
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
        """Request a JSON-mode completion and parse the response.

        Forces JSON output via ``response_format={"type": "json_object"}``
        and parses the resulting text as a Python dict.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: The OpenAI model name to use.
            schema: Pydantic model or type describing the expected JSON structure
                (currently used for documentation purposes only; validation is
                left to the caller).
            reasoning_effort: Effort level forwarded to o-series/gpt-5 models.

        Returns:
            Parsed JSON response as a Python dict.
        """
        # OpenAI json_object mode requires the word "json" to appear somewhere
        # in the messages — inject a system preamble if it's missing.
        has_json_word = any(
            "json" in (m.get("content") or "").lower() for m in messages
        )
        if not has_json_word:
            messages = [
                {"role": "system", "content": "Return only valid JSON. No markdown, no prose."},
                *messages,
            ]
        result = await self.complete(
            messages,
            model,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
        )
        import json
        return json.loads(result.text)

    async def complete_with_tools(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict],
        tool_executor: Callable,
        *,
        max_tool_rounds: int = 3,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> CompletionResult:
        """Run a tool-use completion loop using the OpenAI function-calling API.

        Repeatedly calls the model with the provided tools defined.  For each
        round where the model requests tool calls, all calls are executed in
        parallel via ``tool_executor`` and their results appended to the message
        history.  The loop terminates when the model produces a plain text
        response or ``max_tool_rounds`` is reached.

        Args:
            messages: Initial conversation messages.
            model: OpenAI model identifier.
            tools: List of tool dicts in OpenAI function-calling format.
            tool_executor: Async callable ``(tool_name: str, arguments: dict) → str``.
            max_tool_rounds: Maximum tool-call/execute iterations. Defaults to 3.
            max_tokens: Maximum tokens for each generation step.
            temperature: Sampling temperature.

        Returns:
            The final ``CompletionResult`` when the model produces a text
            response without requesting further tool calls.
        """
        import asyncio as _aio

        current_messages = list(messages)
        last_result = CompletionResult(
            text="", input_tokens=0, output_tokens=0,
            model_used=model, provider_used=self.provider_id, latency_ms=0,
        )

        for _round in range(max_tool_rounds):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": current_messages,
                "tools": tools,
                "tool_choice": "auto",
            }
            if max_tokens is not None:
                kwargs["max_completion_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature

            start = time.monotonic()
            try:
                resp = await _call_with_retry(
                    lambda: self._client.chat.completions.create(**kwargs)
                )
            except Exception as exc:
                log.warning("complete_with_tools: API call failed (%s) — returning last result", exc)
                return last_result
            latency = int((time.monotonic() - start) * 1000)

            choice = resp.choices[0]
            usage = resp.usage
            last_result = CompletionResult(
                text=choice.message.content or "",
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                model_used=model,
                provider_used=self.provider_id,
                latency_ms=latency,
            )

            # If no tool calls, model produced a final answer — done.
            if not choice.message.tool_calls:
                return last_result

            # Append assistant message (with tool_calls) to history
            current_messages.append({
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ],
            })

            # Execute all tool calls in parallel
            async def _exec(tc):
                """Execute a single tool call and return its result dict."""
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result_str = await tool_executor(tc.function.name, args)
                return {"tool_call_id": tc.id, "name": tc.function.name, "result": result_str}

            tool_results = await _aio.gather(*[_exec(tc) for tc in choice.message.tool_calls])

            # Append each tool result as a tool message
            for tr in tool_results:
                log.debug("tool_call name=%s result_len=%d", tr["name"], len(tr["result"]))
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["result"],
                })

        # Exhausted rounds — return last result
        return last_result

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream token chunks from a chat completion request.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: The OpenAI model name to use.
            reasoning_effort: Effort level forwarded to o-series/gpt-5 models.
            max_tokens: Maximum number of tokens to generate.

        Yields:
            String chunks of the generated completion, one delta at a time.
        """
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        # Retry only the stream initiation (not mid-stream): once tokens are
        # flowing we can't safely restart without confusing the caller.
        stream_cm = await _call_with_retry(
            lambda: self._client.chat.completions.create(**kwargs)
        )
        async with stream_cm as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
