"""DuckDuckGo web search adapter — free, no API key required.

Uses the ``duckduckgo-search`` library (``pip install duckduckgo-search``).
All network I/O is dispatched to an executor so it doesn't block the async
event loop.
"""

import asyncio
import logging

from app.adapters.web_search.base import WebSearchAdapter, WebSearchResult

log = logging.getLogger(__name__)


class DuckDuckGoAdapter(WebSearchAdapter):
    """Web search adapter backed by DuckDuckGo.

    Does not require an API key.  Suitable for development and light usage.
    Rate-limited by DuckDuckGo server-side; back off if you receive errors.
    """

    provider_id = "duckduckgo"

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Search DuckDuckGo and return top text results.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (1–10).

        Returns:
            A list of ``WebSearchResult`` objects.  Returns an empty list on
            any error rather than raising.
        """
        def _sync_search() -> list[WebSearchResult]:
            """Perform a synchronous DuckDuckGo text search and return results."""
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=max(1, min(10, max_results))))
                return [
                    WebSearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                    )
                    for r in raw
                ]
            except Exception as exc:
                log.warning("DuckDuckGoAdapter.search failed: %s", exc)
                return []

        return await asyncio.get_event_loop().run_in_executor(None, _sync_search)
