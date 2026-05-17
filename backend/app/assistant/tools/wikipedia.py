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

        # Step 1: search for the best matching page title
        page_title: str | None = None
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                search_resp = await client.get(
                    _WIKI_SEARCH_URL,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": params.query,
                        "srlimit": 3,
                        "format": "json",
                        "srprop": "snippet",
                    },
                    headers={"User-Agent": "ResearchFlow/1.0 (research assistant)"},
                )
                search_resp.raise_for_status()
                search_data = search_resp.json()
                hits = (search_data.get("query") or {}).get("search") or []
                if hits:
                    page_title = hits[0]["title"]
        except Exception as exc:
            log.warning("wikipedia search failed: %s", exc)

        if not page_title:
            return ToolResult(
                output={"title": "", "summary": "", "url": "", "found": False},
                summary=f"No Wikipedia article found for: {params.query}",
            )

        # Step 2: fetch the article summary
        try:
            encoded = urllib.parse.quote(page_title.replace(" ", "_"))
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    _WIKI_SUMMARY_URL.format(title=encoded),
                    headers={
                        "User-Agent": "ResearchFlow/1.0 (research assistant)",
                        "Accept": "application/json",
                    },
                )
                resp.raise_for_status()
                article = resp.json()
        except Exception as exc:
            log.warning("wikipedia summary fetch failed for '%s': %s", page_title, exc)
            return ToolResult(
                output={"title": page_title, "summary": "", "url": "", "found": False},
                summary=f"Wikipedia article found but couldn't fetch content: {exc}",
            )

        title = article.get("title", page_title)
        summary = article.get("extract", "")[:3000]
        url = article.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{encoded}")

        await ctx.emit_progress(100, f"Wikipedia: '{title}'")
        return ToolResult(
            output={"title": title, "summary": summary, "url": url, "found": True},
            summary=f"Wikipedia: {title} ({len(summary)} chars)",
        )


wikipedia_tool = WikipediaTool()
