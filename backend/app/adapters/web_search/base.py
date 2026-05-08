"""WebSearchAdapter ABC — provider-neutral web search interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WebSearchResult:
    """A single web search result.

    Attributes:
        title: Page or document title.
        url: Full URL of the result.
        snippet: Short excerpt or description of the page content.
    """

    title: str
    url: str
    snippet: str


class WebSearchAdapter(ABC):
    """Abstract base class for web search provider adapters.

    All concrete implementations must be async-safe and return a list of
    ``WebSearchResult`` objects.

    Attributes:
        provider_id: Short identifier for the provider (e.g. ``"duckduckgo"``).
    """

    provider_id: str

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        """Execute a web search and return the top results.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return (1–10).

        Returns:
            A list of ``WebSearchResult`` objects, ordered by provider-reported
            relevance. May be shorter than ``max_results`` if fewer results are
            available.
        """
