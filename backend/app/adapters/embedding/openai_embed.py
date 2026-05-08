"""OpenAI text-embedding-3-large adapter (3072-dim) — swappable alternative."""

from openai import AsyncOpenAI

from app.adapters.embedding.base import EmbeddingAdapter
from app.core.config import settings


class OpenAIEmbeddingAdapter(EmbeddingAdapter):
    """Embedding adapter backed by OpenAI's text-embedding-3-large model.

    Outputs are truncated to 768 dimensions via OpenAI's native Matryoshka
    support so they match the ``vector(768)`` column in the database.
    """

    provider_id = "openai"
    model_id = "text-embedding-3-large"
    dimensions = 768  # truncated via OpenAI's native Matryoshka support to match DB vector(768)
    max_batch_size = 100

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the OpenAI async client.

        Args:
            api_key: OpenAI API key. Falls back to ``settings.openai_api_key``
                if not provided.
        """
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def embed_texts(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Embed a list of text strings using the OpenAI embeddings API.

        Splits the input into batches of at most ``max_batch_size``, sends
        each batch to the API, and returns the vectors sorted by index.

        Args:
            texts: Strings to embed.
            task_type: Ignored by this adapter (OpenAI does not use task types);
                present for interface compatibility.

        Returns:
            A list of 768-dimensional float vectors, one per input string.
        """
        batches = [texts[i : i + self.max_batch_size] for i in range(0, len(texts), self.max_batch_size)]
        all_vecs: list[list[float]] = []
        for batch in batches:
            resp = await self._client.embeddings.create(
                model=self.model_id, input=batch, dimensions=self.dimensions
            )
            all_vecs.extend([d.embedding for d in sorted(resp.data, key=lambda x: x.index)])
        return all_vecs

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: The query string to embed.

        Returns:
            A 768-dimensional float vector for the query.
        """
        vecs = await self.embed_texts([text])
        return vecs[0]
