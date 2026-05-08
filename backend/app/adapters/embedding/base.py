"""EmbeddingAdapter ABC — 768-dim Matryoshka (Gemini 2) is the default."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MultimodalItem:
    """Passed to embed_multimodal — exactly one field should be set."""
    text: str | None = None
    image_bytes: bytes | None = None
    image_mime: str = "image/png"


class EmbeddingAdapter(ABC):
    """Abstract base class for all embedding provider adapters.

    Subclasses must set the class-level identifiers and implement
    ``embed_texts`` and ``embed_query``. The default ``embed_multimodal``
    implementation falls back to text-only embedding.

    Attributes:
        provider_id: Short identifier for the provider (e.g. ``"gemini"``).
        model_id: Full model identifier string.
        dimensions: Output vector dimensionality.
        max_batch_size: Maximum number of items per API batch call.
    """

    provider_id: str
    model_id: str
    dimensions: int
    max_batch_size: int

    @abstractmethod
    async def embed_texts(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Batch embed text strings. Returns one vector per input."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Single query embedding (RETRIEVAL_QUERY task type)."""

    async def embed_multimodal(
        self, items: list[MultimodalItem]
    ) -> list[list[float]]:
        """Multimodal embedding — default falls back to text-only."""
        texts = [item.text or "" for item in items]
        return await self.embed_texts(texts)
