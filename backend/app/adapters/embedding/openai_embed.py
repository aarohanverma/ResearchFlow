"""OpenAI text-embedding-3-large adapter (3072-dim) — swappable alternative."""

import httpx
import openai
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.adapters.embedding.base import EmbeddingAdapter
from app.core.config import settings


def _is_retryable_embed(exc: Exception) -> bool:
    """Return ``True`` for transient errors that should be retried.

    Never retries on quota exhaustion (``insufficient_quota``) because
    that is a billing issue, not a transient failure.

    Args:
        exc: The exception raised by the embedding API call.

    Returns:
        ``True`` if the error is transient (network, rate-limit, 5xx).
    """
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return "insufficient_quota" not in str(exc)
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError) and getattr(exc, "status_code", 0) >= 500:
        return True
    return False


async def _embed_with_retry(coro_factory):
    """Run an embedding coroutine factory with 3-attempt jittered exponential backoff.

    Args:
        coro_factory: Zero-argument callable returning an awaitable.

    Returns:
        The result of the coroutine on success.

    Raises:
        The last exception if all 3 attempts fail.
    """
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable_embed),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=8),
        reraise=True,
    ):
        with attempt:
            return await coro_factory()


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
        """Initialise the OpenAI async client for embeddings.

        Args:
            api_key: OpenAI API key. Falls back to ``settings.openai_api_key``
                if not provided.
        """
        # max_retries=0: tenacity handles all retries above; SDK retries would
        # stack on top and produce noisy chained tracebacks for the same error.
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key, max_retries=0)

    async def embed_texts(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Embed a list of texts using ``text-embedding-3-large`` (768-dim).

        Splits large inputs into batches of ``max_batch_size`` (100) and sends
        each batch in a separate API call with exponential-jitter retry.

        Args:
            texts: Strings to embed. Empty strings produce zero vectors.
            task_type: Ignored for OpenAI (present for interface compatibility
                with the Gemini adapter which uses it for task-specific
                embedding optimisation).

        Returns:
            List of 768-dimensional float vectors, one per input text, in the
            same order as ``texts``.
        """
        batches = [texts[i : i + self.max_batch_size] for i in range(0, len(texts), self.max_batch_size)]
        all_vecs: list[list[float]] = []
        for batch in batches:
            resp = await _embed_with_retry(
                lambda b=batch: self._client.embeddings.create(
                    model=self.model_id, input=b, dimensions=self.dimensions
                )
            )
            all_vecs.extend([d.embedding for d in sorted(resp.data, key=lambda x: x.index)])
        return all_vecs

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string and return its 768-dim vector.

        Args:
            text: Query string to embed.

        Returns:
            768-dimensional float vector for the input text.
        """
        vecs = await self.embed_texts([text])
        return vecs[0]
