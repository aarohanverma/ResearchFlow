"""Embedding adapter factory."""

import logging

from app.adapters.embedding.base import EmbeddingAdapter, MultimodalItem
from app.adapters.embedding.gemini import GeminiEmbeddingAdapter
from app.core.config import settings

log = logging.getLogger(__name__)

# Default models per provider — kept in sync with each adapter's model_id.
_PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "gemini": "gemini-embedding-2-preview",
    "openai": "text-embedding-3-large",
    "voyage": "voyage-3",
}


def resolve_embedding_provider(preferred: str | None = None) -> tuple[str, str]:
    """Return (provider, model) that will actually be used at runtime.

    Respects `preferred` (or ``settings.default_embedding_provider`` when
    None), but falls back to the next available key rather than returning a
    provider whose key is absent and would fail at call time.

    Returns:
        A (provider_id, model_id) tuple reflecting what will actually run.
    """
    p = preferred or settings.default_embedding_provider

    def _has_key(prov: str) -> bool:
        """Return True if a non-empty API key is configured for prov."""
        return bool({
            "gemini": settings.google_api_key,
            "openai": settings.openai_api_key,
            "voyage": settings.voyage_api_key,
        }.get(prov, ""))

    if _has_key(p):
        model = settings.default_embedding_model if p == (preferred or settings.default_embedding_provider) else _PROVIDER_DEFAULT_MODEL.get(p, "")
        return p, model or _PROVIDER_DEFAULT_MODEL.get(p, "")

    # Preferred provider has no key — fall back in priority order.
    for fallback in ("gemini", "openai", "voyage"):
        if fallback != p and _has_key(fallback):
            log.warning(
                "embedding: %s requested but no API key configured — falling back to %s",
                p, fallback,
            )
            return fallback, _PROVIDER_DEFAULT_MODEL[fallback]

    # No key at all — return preferred so the adapter fails with a clear error.
    log.warning("embedding: no API key configured for any provider — using %s (will fail at call time)", p)
    return p, settings.default_embedding_model


def get_embedding_adapter(provider: str | None = None) -> EmbeddingAdapter:
    """Instantiate and return the embedding adapter for the effective provider.

    Respects ``settings.default_embedding_provider`` (or the explicit
    ``provider`` arg) but falls back gracefully when the preferred provider's
    API key is absent, rather than returning an adapter that will fail at
    call time with a cryptic error.

    Args:
        provider: Force a specific provider. ``None`` uses the system default
            with automatic fallback.

    Returns:
        An instantiated ``EmbeddingAdapter`` ready to use.
    """
    effective_provider, _ = resolve_embedding_provider(provider)

    if effective_provider == "openai":
        from app.adapters.embedding.openai_embed import OpenAIEmbeddingAdapter
        return OpenAIEmbeddingAdapter()
    if effective_provider == "gemini":
        return GeminiEmbeddingAdapter()
    if effective_provider == "voyage":
        from app.adapters.embedding.voyage_embed import VoyageEmbeddingAdapter  # type: ignore[import]
        return VoyageEmbeddingAdapter()

    # Unknown — default to Gemini
    return GeminiEmbeddingAdapter()


__all__ = [
    "EmbeddingAdapter",
    "MultimodalItem",
    "get_embedding_adapter",
    "resolve_embedding_provider",
]
