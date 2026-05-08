"""Google Gemini LLM adapter — Flash Lite (cheap), Pro (quality/reasoning)."""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import google.generativeai as genai

from app.adapters.llm.base import CompletionResult, LLMAdapter
from app.core.config import settings


class GoogleAdapter(LLMAdapter):
    """LLM adapter for Google Gemini via the ``google-generativeai`` SDK.

    Uses Gemini Flash Lite (cheap) and Gemini Pro (quality/reasoning) model
    tiers. The Gemini SDK is synchronous and is run inside an executor to
    avoid blocking the async event loop.
    """

    provider_id = "google"
    cheap_model = "gemini-3.1-flash-lite"
    quality_model = "gemini-3.1-pro"
    reasoning_model = "gemini-3.1-pro"  # Deep Think toggled via generation_config

    def __init__(self, api_key: str | None = None) -> None:
        """Configure the Gemini SDK with the provided or configured API key.

        Args:
            api_key: Google AI API key. Falls back to ``settings.google_api_key``
                if not provided.
        """
        genai.configure(api_key=api_key or settings.google_api_key)

    def _to_gemini_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Split OpenAI-style messages into (system_instruction, gemini_history)."""
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        history = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in messages if m["role"] != "system"
        ]
        return system, history

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
        """Send a generation request to Gemini and return the full response.

        Converts OpenAI-style messages to Gemini format, then runs the
        synchronous SDK call in an executor to avoid blocking the event loop.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: Gemini model name (e.g. ``"gemini-3.1-pro"``).
            reasoning_effort: Not used by this adapter; present for interface
                compatibility.
            temperature: Sampling temperature passed to ``generation_config``.
            max_tokens: Maximum output tokens passed to ``generation_config``.
            response_format: If ``{"type": "json_object"}``, sets the response
                MIME type to ``application/json``.

        Returns:
            A ``CompletionResult`` containing the generated text, token counts,
            model/provider identifiers, and latency.
        """
        system, history = self._to_gemini_messages(messages)

        gen_config: dict[str, Any] = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_tokens is not None:
            gen_config["max_output_tokens"] = max_tokens
        if response_format and response_format.get("type") == "json_object":
            gen_config["response_mime_type"] = "application/json"

        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config=gen_config if gen_config else None,
        )

        start = time.monotonic()
        # Gemini SDK is not fully async — run in executor
        import asyncio
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: gemini_model.generate_content(history),
        )
        latency = int((time.monotonic() - start) * 1000)

        usage = resp.usage_metadata
        return CompletionResult(
            text=resp.text,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
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
        """Request a JSON-mode Gemini completion and parse the response.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: Gemini model name.
            schema: Pydantic model or type describing the expected JSON
                structure (used for documentation; validation is left to the
                caller).
            reasoning_effort: Not used by this adapter; present for interface
                compatibility.

        Returns:
            Parsed JSON response as a Python dict.
        """
        result = await self.complete(
            messages, model, response_format={"type": "json_object"}
        )
        return json.loads(result.text)

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream text chunks from a Gemini generation request.

        Runs the synchronous streaming SDK call in an executor, then yields
        the text of each chunk synchronously.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            model: Gemini model name.
            reasoning_effort: Not used by this adapter; present for interface
                compatibility.
            max_tokens: Not used by this adapter; present for interface
                compatibility.

        Yields:
            Non-empty text chunks from each streamed response part.
        """
        system, history = self._to_gemini_messages(messages)
        gemini_model = genai.GenerativeModel(model_name=model, system_instruction=system)

        import asyncio
        responses = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: gemini_model.generate_content(history, stream=True),
        )
        for chunk in responses:
            if chunk.text:
                yield chunk.text
