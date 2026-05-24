"""Exa web search adapter — neural search optimised for LLM agents.

Exa (formerly Metaphor) returns embedding-ranked results with full-text
extracts that are tuned for downstream LLM consumption. It tends to
outperform keyword-only engines for research-flavoured queries because
the index is curated and the ranking is similarity-based rather than
keyword-token-overlap-based.

Requires ``EXA_API_KEY`` in the environment. Uses Exa's HTTP API
directly (no SDK dependency) so we don't pull in another package just
to make one POST request.
"""

from __future__ import annotations

import logging

import httpx

from app.adapters.web_search.base import WebSearchAdapter, WebSearchResult
from app.core.config import settings

log = logging.getLogger(__name__)


_EXA_API = "https://api.exa.ai/search"
_REQUEST_TIMEOUT = 12.0


class ExaAdapter(WebSearchAdapter):
    """Web search adapter backed by Exa's neural-search API.

    Set ``EXA_API_KEY`` and ``WEB_SEARCH_PROVIDER=exa`` to route through
    this adapter. Returns an empty result list on any failure rather
    than raising — callers (the assistant's ``web_search`` tool) treat
    empty results as a soft signal and fall back to other tools.
    """

    provider_id = "exa"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or getattr(settings, "exa_api_key", None)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[WebSearchResult]:
        if not self._api_key:
            log.debug("ExaAdapter.search skipped: no EXA_API_KEY configured")
            return []
        # Exa accepts 1–100; clamp tightly so an upstream bug can't
        # blow the budget. Use ``auto`` search type so Exa picks
        # neural vs keyword per query — works better across the wide
        # range of namespaces RA covers (CS / physics / biology /
        # finance) than locking to neural-only.
        payload = {
            "query": query,
            "numResults": max(1, min(10, int(max_results))),
            "type": "auto",
            "contents": {
                "text": {"maxCharacters": 800, "includeHtmlTags": False},
            },
        }
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                resp = await client.post(
                    _EXA_API,
                    headers={
                        "x-api-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            log.warning("ExaAdapter.search timed out after %.0fs", _REQUEST_TIMEOUT)
            return []
        except Exception as exc:  # noqa: BLE001 — must never raise to the agent
            log.warning("ExaAdapter.search failed: %s", exc)
            return []

        items = data.get("results") or []
        out: list[WebSearchResult] = []
        for r in items:
            if not isinstance(r, dict):
                continue
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            # Prefer the ``text`` excerpt (Exa's extracted body) over the
            # bare ``snippet`` — it's longer and structured for LLM use.
            snippet = (r.get("text") or r.get("snippet") or "").strip()
            if not url:
                continue
            out.append(WebSearchResult(
                title=title or url,
                url=url,
                snippet=snippet[:1200],
            ))
        return out
