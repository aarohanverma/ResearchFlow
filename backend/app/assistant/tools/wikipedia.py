"""Wikipedia tool — fetch authoritative background knowledge.

Uses the Wikipedia REST API to retrieve article summaries and key sections.
Useful for grounding the assistant in established definitions, historical context,
and broad consensus knowledge on a topic before diving into the research literature.
"""

from __future__ import annotations

import logging
import urllib.parse

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult

log = logging.getLogger(__name__)

_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_TIMEOUT = 10.0


class WikipediaInput(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="Concept, term, or topic to look up")
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class WikipediaOutput(BaseModel):
    title: str
    summary: str
    url: str
    found: bool


class WikipediaTool:
    """Retrieve authoritative background knowledge from Wikipedia."""

    name = "wikipedia"
    summary = (
        "Fetch a Wikipedia article summary for a concept, term, person, or event. "
        "Use for: established definitions ('what is backpropagation'), historical "
        "background ('history of neural networks'), consensus knowledge on a topic, "
        "or grounding the answer in well-known reference material. Especially useful "
        "for newcomers who need accurate foundational context before the research literature."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = WikipediaInput
    output_schema = WikipediaOutput

    async def run(self, ctx: ToolContext, params: WikipediaInput) -> ToolResult:
        await ctx.emit_progress(20, f"Looking up Wikipedia: {params.query[:60]}")

        # ── Strategy ────────────────────────────────────────────────────
        # The wikipedia search endpoint is fuzzy but inconsistent — it
        # routinely fails to surface real articles for hyphenated or
        # compound topics that the REST summary endpoint resolves
        # instantly (e.g. "Retrieval-augmented generation"). We therefore
        # try a sequence of strategies before giving up:
        #
        #   1. Direct REST summary lookup (handles redirects automatically).
        #   2. Title-cased variant.
        #   3. Fuzzy search via the action API.
        #   4. The same fuzzy search with hyphens dropped — handles cases
        #      where the user typed "retrieval-augmented" but the index
        #      uses spaces or vice-versa.
        page_title: str | None = None
        article: dict | None = None

        async def _try_summary_direct(client, candidate: str) -> dict | None:
            """Hit the REST summary endpoint directly with the given title."""
            try:
                encoded = urllib.parse.quote(candidate.replace(" ", "_"))
                resp = await client.get(
                    _WIKI_SUMMARY_URL.format(title=encoded),
                    headers={
                        "User-Agent": "ResearchFlow/1.0 (research assistant)",
                        "Accept": "application/json",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # REST summary returns type="standard" for real articles
                    # and type="disambiguation" / "no-extract" otherwise.
                    if data.get("type") not in ("standard", "anchored", None):
                        return None
                    if data.get("extract"):
                        return data
            except Exception:
                pass
            return None

        async def _try_search(client, search_q: str) -> str | None:
            try:
                resp = await client.get(
                    _WIKI_SEARCH_URL,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": search_q,
                        "srlimit": 3,
                        "format": "json",
                        "srprop": "snippet",
                    },
                    headers={"User-Agent": "ResearchFlow/1.0 (research assistant)"},
                )
                resp.raise_for_status()
                data = resp.json()
                hits = (data.get("query") or {}).get("search") or []
                if hits:
                    return hits[0]["title"]
            except Exception as exc:
                log.debug("wikipedia search failed for %r: %s", search_q, exc)
            return None

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                # Step 1: direct REST summary on the user's literal phrase.
                article = await _try_summary_direct(client, params.query.strip())

                # Step 2: try title-cased ("retrieval-augmented generation"
                # → "Retrieval-augmented generation"). Wikipedia titles
                # generally capitalise the first character.
                if article is None and params.query.strip():
                    first = params.query.strip()
                    titled = first[0].upper() + first[1:]
                    if titled != first:
                        article = await _try_summary_direct(client, titled)

                # Step 3: fall back to the action-API search.
                if article is None:
                    page_title = await _try_search(client, params.query)

                # Step 4: relaxed search with hyphens replaced by spaces.
                if article is None and page_title is None and "-" in params.query:
                    page_title = await _try_search(
                        client, params.query.replace("-", " "),
                    )

                # If search found a title, fetch its summary directly.
                if article is None and page_title:
                    article = await _try_summary_direct(client, page_title)

        except Exception as exc:
            log.warning("wikipedia lookup failed: %s", exc)

        if article is None:
            return ToolResult(
                output={"title": "", "summary": "", "url": "", "found": False},
                summary=f"No Wikipedia article found for: {params.query}",
            )

        title = article.get("title", page_title or params.query)
        summary = (article.get("extract") or "")[:3000]
        url = (
            (article.get("content_urls") or {}).get("desktop", {}).get("page")
            or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        )
        await ctx.emit_progress(100, f"Wikipedia: '{title}'")
        return ToolResult(
            output={"title": title, "summary": summary, "url": url, "found": True},
            summary=f"Wikipedia: {title} ({len(summary)} chars)",
        )


wikipedia_tool = WikipediaTool()
