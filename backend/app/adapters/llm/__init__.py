"""LLM adapter factory — resolves the correct adapter from user/system settings."""

from app.adapters.llm.anthropic_adapter import AnthropicAdapter
from app.adapters.llm.base import CompletionResult, LLMAdapter
from app.adapters.llm.google_adapter import GoogleAdapter
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.core.config import settings

_ADAPTERS: dict[str, type[LLMAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "google": GoogleAdapter,
}

# Fallback chain per tier (primary → fallback)
FALLBACK_CHAINS: dict[str, list[tuple[str, str]]] = {
    "cheap": [
        ("openai", OpenAIAdapter.cheap_model),
        ("anthropic", AnthropicAdapter.cheap_model),
    ],
    "quality": [
        ("openai", OpenAIAdapter.quality_model),
        ("anthropic", AnthropicAdapter.quality_model),
    ],
    "reasoning": [
        ("openai", OpenAIAdapter.reasoning_model),
        ("anthropic", AnthropicAdapter.reasoning_model),
        ("google", GoogleAdapter.reasoning_model),
    ],
}


def get_llm_adapter(provider: str | None = None) -> LLMAdapter:
    """Instantiate and return the LLM adapter for the given provider.

    Wraps the concrete adapter in :class:`TrackingLLMAdapter` so every
    completion is recorded to the ``token_usage`` table for the Settings
    page dashboard. Tracking is fire-and-forget — failures cannot break
    LLM calls.

    Args:
        provider: Provider identifier (``"openai"``, ``"anthropic"``, or
            ``"google"``). Defaults to ``settings.default_llm_provider``.
            Falls back to ``OpenAIAdapter`` if the identifier is unknown.

    Returns:
        A tracking-wrapped ``LLMAdapter`` for the selected provider.
    """
    from app.adapters.llm.tracking import TrackingLLMAdapter
    provider = provider or settings.default_llm_provider
    cls = _ADAPTERS.get(provider, OpenAIAdapter)
    return TrackingLLMAdapter(cls())  # type: ignore[return-value]


__all__ = ["LLMAdapter", "CompletionResult", "get_llm_adapter", "FALLBACK_CHAINS"]
