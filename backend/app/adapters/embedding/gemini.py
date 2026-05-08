"""Gemini Embedding 2 adapter — 768-dim Matryoshka, 8 task types, multimodal."""

import asyncio
from typing import Any

import google.generativeai as genai

from app.adapters.embedding.base import EmbeddingAdapter, MultimodalItem
from app.core.config import settings

# Supported task types per Gemini Embedding 2 docs
TASK_TYPES = {
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "SEMANTIC_SIMILARITY",
    "CLUSTERING",
    "CODE_RETRIEVAL_QUERY",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CLASSIFICATION",
}


class GeminiEmbeddingAdapter(EmbeddingAdapter):
    """Embedding adapter backed by Gemini Embedding 2 (768-dim Matryoshka).

    Supports eight task types and multimodal inputs (image + optional caption).
    The underlying SDK is synchronous and all calls are dispatched to an
    executor to avoid blocking the async event loop.
    """

    provider_id = "gemini"
    model_id = "gemini-embedding-2-preview"
    dimensions = 768           # Matryoshka sweet spot — near-peak quality
    max_batch_size = 100

    def __init__(self, api_key: str | None = None) -> None:
        """Configure the Gemini SDK with the provided or configured API key.

        Args:
            api_key: Google AI API key. Falls back to ``settings.google_api_key``
                if not provided.
        """
        genai.configure(api_key=api_key or settings.google_api_key)

    async def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Embed a single batch (≤ max_batch_size) in an executor (SDK is sync)."""
        def _sync():
            """Run the synchronous Gemini embed_content calls for a single batch."""
            results = []
            for text in texts:
                resp = genai.embed_content(
                    model=f"models/{self.model_id}",
                    content=text,
                    task_type=task_type,
                    output_dimensionality=self.dimensions,
                )
                results.append(resp["embedding"])
            return results

        return await asyncio.get_event_loop().run_in_executor(None, _sync)

    async def embed_texts(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Embed a list of text strings in batches.

        Splits the input into batches of at most ``max_batch_size`` and
        delegates each batch to ``_embed_batch``. Falls back to
        ``"RETRIEVAL_DOCUMENT"`` if an unrecognised task type is supplied.

        Args:
            texts: Strings to embed.
            task_type: Gemini embedding task type. Must be one of the values
                in ``TASK_TYPES``; defaults to ``"RETRIEVAL_DOCUMENT"``.

        Returns:
            A list of 768-dimensional float vectors, one per input string.
        """
        if task_type not in TASK_TYPES:
            task_type = "RETRIEVAL_DOCUMENT"

        # Split into batches of max_batch_size
        batches = [
            texts[i : i + self.max_batch_size]
            for i in range(0, len(texts), self.max_batch_size)
        ]
        results: list[list[float]] = []
        for batch in batches:
            batch_result = await self._embed_batch(batch, task_type)
            results.extend(batch_result)
        return results

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string using the RETRIEVAL_QUERY task type.

        Args:
            text: The query string to embed.

        Returns:
            A 768-dimensional float vector for the query.
        """
        vectors = await self.embed_texts([text], task_type="RETRIEVAL_QUERY")
        return vectors[0]

    async def embed_multimodal(self, items: list[MultimodalItem]) -> list[list[float]]:
        """For figure embeddings — send image bytes via Gemini multimodal API."""
        results: list[list[float]] = []
        for item in items:
            if item.image_bytes:
                def _sync_img(img_bytes=item.image_bytes, caption=item.text):
                    """Embed a single image (with optional caption) via Gemini multimodal API."""
                    import PIL.Image
                    import io
                    img = PIL.Image.open(io.BytesIO(img_bytes))
                    content = [img]
                    if caption:
                        content.append(caption)
                    resp = genai.embed_content(
                        model=f"models/{self.model_id}",
                        content=content,
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=self.dimensions,
                    )
                    return resp["embedding"]
                vec = await asyncio.get_event_loop().run_in_executor(None, _sync_img)
            else:
                vec = (await self.embed_texts([item.text or ""]))[0]
            results.append(vec)
        return results
