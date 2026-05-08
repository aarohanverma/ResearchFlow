"""LLMAdapter ABC — all provider implementations must conform to this contract."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    """Represents a single tool-call request returned by the LLM.

    Attributes:
        id: Provider-assigned call ID (used to correlate results).
        name: Name of the tool/function to invoke.
        arguments: Parsed dict of arguments for the tool.
    """

    id: str
    name: str
    arguments: dict


@dataclass
class CompletionResult:
    """Holds the result of a single LLM completion call.

    Attributes:
        text: The generated text content from the model.
        input_tokens: Number of tokens in the prompt/input.
        output_tokens: Number of tokens in the completion/output.
        model_used: Identifier of the model that produced the response.
        provider_used: Identifier of the provider (e.g. "openai", "anthropic").
        latency_ms: Wall-clock latency of the API call in milliseconds.
    """

    text: str
    input_tokens: int
    output_tokens: int
    model_used: str
    provider_used: str
    latency_ms: int


class LLMAdapter(ABC):
    """Abstract base class for all LLM provider adapters.

    Subclasses must set the class-level model identifiers and implement
    the three abstract methods: complete, complete_structured, and stream.

    Attributes:
        provider_id: Short string identifying the provider (e.g. "openai").
        cheap_model: Model name to use for low-cost, fast completions.
        quality_model: Model name for higher-quality completions.
        reasoning_model: Model name for complex reasoning tasks.
    """

    provider_id: str
    cheap_model: str
    quality_model: str
    reasoning_model: str

    @abstractmethod
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
        """Single completion — blocking in the async sense (awaited)."""

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        model: str,
        schema: type,
        *,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """JSON-mode completion validated against schema."""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Token-by-token streaming — yields string chunks."""

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
        """Run a completion with a tool-use loop until the model stops calling tools.

        Calls the LLM, executes any requested tool calls via ``tool_executor``,
        and repeats up to ``max_tool_rounds`` times until the model returns a plain
        text response.

        Subclasses that natively support tool calling (e.g. ``OpenAIAdapter``) should
        override this method to use their provider's function-calling API.
        The default implementation delegates to ``complete`` without invoking tools —
        safe for providers that do not support tool calling.

        Args:
            messages: Initial conversation messages.
            model: Model identifier to use.
            tools: List of tool definition dicts (OpenAI function-calling format).
            tool_executor: Async callable ``(tool_name, arguments) → str``.
                Called for each tool the model requests; return value is the
                tool result string inserted into the message history.
            max_tool_rounds: Maximum tool-call/execute iterations before
                returning the last completion result. Defaults to 3.
            max_tokens: Maximum tokens for each generation step.
            temperature: Sampling temperature.

        Returns:
            The final ``CompletionResult`` after the model stops requesting
            tools (or after ``max_tool_rounds`` is exhausted).
        """
        # Default fallback: call complete without tools (no tool execution)
        import logging as _log
        _log.getLogger(__name__).debug(
            "complete_with_tools: provider %s does not support native tool calling — "
            "falling back to plain complete (tools will not be executed)",
            self.provider_id,
        )
        return await self.complete(
            messages, model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
