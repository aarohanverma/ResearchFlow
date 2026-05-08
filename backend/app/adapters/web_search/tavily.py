"""Tavily web search adapter — high-quality results, optimised for LLM agents.

Requires ``TAVILY_API_KEY`` in the environment.  Uses the ``tavily-python``
SDK (``pip install tavily-python``).
"""

import asyncio
import logging

from app.adapters.web_search.base import WebSearchAdapter, WebSearchResult
from app.core.config import settings

log = logging.getLogger(__name__)


class TavilyAdapter(WebSearchAdapter):
    """Web search adapter backed by Tavily AI.

    Tavily returns clean, deduplicated results optimised for LLM consumption.
    Requires a valid ``TAVILY_API_KEY`` environment variable.
    """

    provider_id = "tavily"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the Tavily client.

        Args:
            api_key: Tavily API key. Falls back to ``settings.tavily_api_key``.
        """
        self._api_key = api_key or settings.tavily_api_key

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Search Tavily and return top results for the query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (1–10).

        Returns:
            A list of ``WebSearchResult`` objects.  Returns an empty list on
            any error rather than raising.
        """
        def _sync_search() -> list[WebSearchResult]:
            """Perform a synchronous Tavily search and return results."""
            try:
                from tavily import TavilyClient
                client = TavilyClient(api_key=self._api_key)
                resp = client.search(
                    query,
                    max_results=max(1, min(10, max_results)),
                    search_depth="basic",
                )
                return [
                    WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("content", ""),
                    )
                    for r in resp.get("results", [])
                ]
            except Exception as exc:
                log.warning("TavilyAdapter.search failed: %s", exc)
                return []

        return await asyncio.get_event_loop().run_in_executor(None, _sync_search)
