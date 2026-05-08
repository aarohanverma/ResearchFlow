"""Embedding adapter factory."""

import logging

from app.adapters.embedding.base import EmbeddingAdapter, MultimodalItem
from app.adapters.embedding.gemini import GeminiEmbeddingAdapter
from app.core.config import settings

log = logging.getLogger(__name__)


def get_embedding_adapter(provider: str | None = None) -> EmbeddingAdapter:
    """Instantiate and return the embedding adapter for the given provider.

    Auto-selects Gemini if ``settings.google_api_key`` is set, then falls
    back to OpenAI. Logs a warning and returns a Gemini adapter (which will
    fail at call time) if no API key is configured.

    Args:
        provider: Provider identifier (``"openai"`` or ``"gemini"``). If
            ``None``, the best available adapter is selected automatically
            based on configured API keys.

    Returns:
        An instantiated ``EmbeddingAdapter`` for the selected provider.
    """
    if provider == "openai":
        from app.adapters.embedding.openai_embed import OpenAIEmbeddingAdapter
        return OpenAIEmbeddingAdapter()
    if provider == "gemini":
        return GeminiEmbeddingAdapter()
    # Auto-select: Gemini first, OpenAI if no Gemini key
    if settings.google_api_key:
        return GeminiEmbeddingAdapter()
    if settings.openai_api_key:
        from app.adapters.embedding.openai_embed import OpenAIEmbeddingAdapter
        log.info("No Gemini API key — using OpenAI embeddings as fallback")
        return OpenAIEmbeddingAdapter()
    log.warning("No embedding API key configured — attempting Gemini (will fail at call time)")
    return GeminiEmbeddingAdapter()


__all__ = ["EmbeddingAdapter", "MultimodalItem", "get_embedding_adapter"]
