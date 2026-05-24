"""Web search adapter factory.

Provider preference, in order:

  1. ``WEB_SEARCH_PROVIDER`` env var, when set to a provider with a
     valid key (``exa`` / ``tavily`` / ``duckduckgo``).
  2. **Auto-best**: when ``WEB_SEARCH_PROVIDER`` is unset or set to
     ``auto``, pick the best-available LLM-optimised provider whose
     key is present — Exa first, then Tavily, then DuckDuckGo.
     DuckDuckGo is the no-key fallback so the tool always works,
     but its result quality for research queries is markedly lower
     than the LLM-optimised neural / curated alternatives.

The auto-best order is deliberate: Exa's neural search outperforms
keyword retrieval on long, research-flavoured queries; Tavily is the
reliable second choice (cleaner, deduped) when Exa is unavailable;
DuckDuckGo's free HTML-scraping path is the floor.

Usage::

    from app.adapters.web_search import get_web_search_adapter

    adapter = get_web_search_adapter()
    results = await adapter.search("attention mechanism transformers", max_results=5)
    # results: list of {"title", "url", "snippet"}
"""

import logging

from app.adapters.web_search.base import WebSearchAdapter, WebSearchResult
from app.core.config import settings

log = logging.getLogger(__name__)


def _has_exa_key() -> bool:
    return bool(getattr(settings, "exa_api_key", None))


def _has_tavily_key() -> bool:
    return bool(getattr(settings, "tavily_api_key", None))


def get_web_search_adapter(provider: str | None = None) -> WebSearchAdapter:
    """Return the configured web search adapter.

    Args:
        provider: Override the configured provider. Accepts
            ``"exa"``, ``"tavily"``, ``"duckduckgo"``, or ``"auto"``
            (best-available LLM-optimised provider). Defaults to
            ``settings.web_search_provider``.

    Returns:
        An instantiated ``WebSearchAdapter``. Always returns *something*
        — falls back to DuckDuckGo when no API keys are configured so
        the ``web_search`` tool stays functional in any environment.
    """
    p = (provider or settings.web_search_provider or "auto").lower()

    # Explicit provider — honour the request when keys are present;
    # silently fall through to auto-best when the key is missing so we
    # don't ship 0-result calls back to the agent.
    if p == "exa" and _has_exa_key():
        from app.adapters.web_search.exa import ExaAdapter
        return ExaAdapter()
    if p == "tavily" and _has_tavily_key():
        from app.adapters.web_search.tavily import TavilyAdapter
        return TavilyAdapter()
    if p == "duckduckgo":
        from app.adapters.web_search.duckduckgo import DuckDuckGoAdapter
        return DuckDuckGoAdapter()

    # Auto-best path: Exa > Tavily > DuckDuckGo. When the explicit
    # provider missed a key, fall through here so the agent still gets
    # the best engine available rather than a hard fail.
    if _has_exa_key():
        from app.adapters.web_search.exa import ExaAdapter
        log.debug("web_search: auto-selected Exa (EXA_API_KEY present)")
        return ExaAdapter()
    if _has_tavily_key():
        from app.adapters.web_search.tavily import TavilyAdapter
        log.debug("web_search: auto-selected Tavily (TAVILY_API_KEY present)")
        return TavilyAdapter()
    from app.adapters.web_search.duckduckgo import DuckDuckGoAdapter
    log.debug("web_search: falling back to DuckDuckGo (no LLM-optimised keys configured)")
    return DuckDuckGoAdapter()


__all__ = ["WebSearchAdapter", "WebSearchResult", "get_web_search_adapter"]
