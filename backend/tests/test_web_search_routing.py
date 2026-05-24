"""Web-search adapter selection + LLM-optimised provider preference.

The auto-best path must prefer Exa → Tavily → DuckDuckGo, and an
explicit provider request must silently degrade to auto-best when its
key is missing rather than returning a broken adapter.
"""

from __future__ import annotations

import pytest

from app.adapters.web_search import get_web_search_adapter
from app.adapters.web_search.duckduckgo import DuckDuckGoAdapter
from app.adapters.web_search.exa import ExaAdapter
from app.adapters.web_search.tavily import TavilyAdapter


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    """Each test starts from a known clean state — no provider keys
    present, provider preference set to ``auto``."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "web_search_provider", "auto")
    monkeypatch.setattr(settings, "tavily_api_key", "")
    monkeypatch.setattr(settings, "exa_api_key", "")
    yield


def test_auto_falls_back_to_duckduckgo_when_no_keys():
    """No LLM-optimised key present → DuckDuckGo (so the tool always
    works, even in a fresh dev env)."""
    adapter = get_web_search_adapter()
    assert isinstance(adapter, DuckDuckGoAdapter)


def test_auto_prefers_tavily_when_only_tavily_configured(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "tavily_api_key", "tav-key")
    assert isinstance(get_web_search_adapter(), TavilyAdapter)


def test_auto_prefers_exa_when_only_exa_configured(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "exa_api_key", "exa-key")
    assert isinstance(get_web_search_adapter(), ExaAdapter)


def test_auto_prefers_exa_over_tavily_when_both_present(monkeypatch):
    """Exa is the neural-search top choice; auto-best must pick it
    over Tavily when both keys are present."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "exa_api_key", "exa-key")
    monkeypatch.setattr(settings, "tavily_api_key", "tav-key")
    assert isinstance(get_web_search_adapter(), ExaAdapter)


def test_explicit_provider_request_honoured(monkeypatch):
    """When the caller passes an explicit provider AND the key is
    configured, route to that provider regardless of the auto order."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "exa_api_key", "exa-key")
    monkeypatch.setattr(settings, "tavily_api_key", "tav-key")
    # Force Tavily even though Exa is auto-preferred.
    assert isinstance(get_web_search_adapter(provider="tavily"), TavilyAdapter)


def test_explicit_provider_falls_back_when_key_missing(monkeypatch):
    """Asking for ``exa`` without an EXA key must NOT crash — it
    transparently auto-degrades to whatever is available."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "tavily_api_key", "tav-key")
    # We asked for Exa but only Tavily is configured.
    adapter = get_web_search_adapter(provider="exa")
    assert isinstance(adapter, TavilyAdapter)


def test_explicit_duckduckgo_always_returns_ddg():
    """DuckDuckGo needs no key, so an explicit request must always
    return the DDG adapter regardless of other configuration."""
    assert isinstance(get_web_search_adapter(provider="duckduckgo"), DuckDuckGoAdapter)


# ── Exa adapter smoke behaviour (no network) ──────────────────────────────


@pytest.mark.asyncio
async def test_exa_adapter_skips_silently_without_key():
    """An Exa adapter with no key returns an empty list rather than
    raising — callers (the ``web_search`` tool) treat empty results as
    a soft signal and the loop falls through."""
    adapter = ExaAdapter(api_key="")
    results = await adapter.search("transformer scaling laws", max_results=3)
    assert results == []


@pytest.mark.asyncio
async def test_exa_adapter_handles_http_failure(monkeypatch):
    """When the Exa HTTP call raises, the adapter must swallow the
    exception and return an empty list — never propagate to the
    agent."""
    import httpx

    class _BrokenClient:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        async def post(self, *_a, **_kw):
            raise httpx.ConnectError("network down")

    monkeypatch.setattr(httpx, "AsyncClient", _BrokenClient)
    results = await ExaAdapter(api_key="fake-key").search("query", max_results=3)
    assert results == []


@pytest.mark.asyncio
async def test_exa_adapter_parses_well_formed_response(monkeypatch):
    """Round-trip a canned API response shape through the adapter and
    confirm it produces the expected WebSearchResult records."""
    import httpx

    class _OKResponse:
        def __init__(self, payload): self._payload = payload
        def raise_for_status(self): pass
        def json(self): return self._payload

    class _OKClient:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        async def post(self, _url, **_kw):
            return _OKResponse({"results": [
                {"title": "Paper A", "url": "https://example.com/a", "text": "abstract A"},
                {"title": "", "url": "https://example.com/b", "snippet": "snippet B"},
                {"title": "Skip me", "url": ""},  # missing URL — must be dropped
            ]})

    monkeypatch.setattr(httpx, "AsyncClient", _OKClient)
    results = await ExaAdapter(api_key="fake-key").search("query", max_results=5)
    assert len(results) == 2
    assert results[0].title == "Paper A"
    assert results[0].snippet.startswith("abstract A")
    # Empty title falls back to URL so the agent doesn't render a blank link.
    assert results[1].title == "https://example.com/b"
    assert results[1].snippet == "snippet B"
