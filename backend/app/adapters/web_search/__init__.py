"""Web search adapter factory.

Auto-selects the configured backend (``duckduckgo`` by default; ``tavily``
when ``TAVILY_API_KEY`` is set and ``WEB_SEARCH_PROVIDER=tavily``).

Usage::

    from app.adapters.web_search import get_web_search_adapter

    adapter = get_web_search_adapter()
    results = await adapter.search("attention mechanism transformers", max_results=5)
    # results: list of {"title", "url", "snippet"}
"""

from app.adapters.web_search.base import WebSearchAdapter, WebSearchResult
from app.core.config import settings


def get_web_search_adapter(provider: str | None = None) -> WebSearchAdapter:
    """Return the configured web search adapter.

    Args:
        provider: Override the configured provider (``"duckduckgo"`` or
            ``"tavily"``). Defaults to ``settings.web_search_provider``.

    Returns:
        An instantiated ``WebSearchAdapter``.
    """
    p = provider or settings.web_search_provider
    if p == "tavily" and settings.tavily_api_key:
        from app.adapters.web_search.tavily import TavilyAdapter
        return TavilyAdapter()
    from app.adapters.web_search.duckduckgo import DuckDuckGoAdapter
    return DuckDuckGoAdapter()


__all__ = ["WebSearchAdapter", "WebSearchResult", "get_web_search_adapter"]
