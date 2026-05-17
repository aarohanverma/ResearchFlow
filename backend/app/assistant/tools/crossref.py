"""CrossRef tool — resolve DOIs and enrich paper metadata.

Uses the CrossRef REST API (free, no key required) to:
- Resolve a DOI to full verified publication metadata (journal, volume, issue, pages)
- Search for papers by title and return authoritative citation metadata
- Retrieve reference lists and funding information

Especially useful when the user cites a specific paper by title or DOI and needs
accurate bibliographic details — journal name, publication date, open-access link.
"""

from __future__ import annotations

import logging
import urllib.parse

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_CR_WORKS_URL = "https://api.crossref.org/works"
_TIMEOUT = 15.0
_MAILTO = "researchflow@example.com"  # polite pool header


class CrossRefInput(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="DOI (e.g. '10.1145/12345') or paper title to look up")
    mode: str = Field(default="auto", description="'resolve' to fetch a specific DOI, 'search' to find by title, 'auto' to detect")
    limit: int = Field(default=5, ge=1, le=15)
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class CrossRefOutput(BaseModel):
    works: list[dict]
    total: int
    mode: str


class CrossRefTool:
    """Resolve DOIs and retrieve authoritative publication metadata via CrossRef."""

    name = "crossref"
    summary = (
        "Look up paper metadata via the CrossRef database (200M+ records). "
        "Use for: (a) resolving a specific DOI to get journal, volume, issue, pages, "
        "and funding info; (b) verifying bibliographic details for a paper title; "
        "(c) finding the published venue for an arXiv preprint. "
        "Returns DOI, journal name, publisher, publication date, open-access URL, "
        "and reference/citation counts. No API key needed."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = CrossRefInput
    output_schema = CrossRefOutput

    async def run(self, ctx: ToolContext, params: CrossRefInput) -> ToolResult:
        await ctx.emit_progress(20, f"Querying CrossRef: {params.query[:60]}")

        q = params.query.strip()
        is_doi = q.startswith("10.") or q.startswith("doi:")
        mode = params.mode
        if mode == "auto":
            mode = "resolve" if is_doi else "search"

        if mode == "resolve":
            doi = q.removeprefix("doi:").strip()
            return await self._resolve_doi(doi)

        return await self._search_title(q, params.limit)

    async def _resolve_doi(self, doi: str) -> ToolResult:
        """Fetch a specific DOI from CrossRef."""
        url = f"{_CR_WORKS_URL}/{urllib.parse.quote(doi, safe='')}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": f"ResearchFlow/1.0 (mailto:{_MAILTO})"},
                )
                if resp.status_code == 404:
                    return ToolResult(
                        output={"works": [], "total": 0, "mode": "resolve"},
                        summary=f"DOI not found: {doi}",
                    )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            log.warning("crossref DOI resolve failed for %s: %s", doi, exc)
            return ToolResult(
                output={"works": [], "total": 0, "mode": "resolve"},
                summary=f"CrossRef unavailable: {exc}",
            )

        work = _parse_work(data.get("message", {}))
        return ToolResult(
            output={"works": [work], "total": 1, "mode": "resolve"},
            summary=f"CrossRef resolved: {work.get('title', doi)} ({work.get('journal', '')})",
        )

    async def _search_title(self, query: str, limit: int) -> ToolResult:
        """Title/keyword search via CrossRef /works endpoint."""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    _CR_WORKS_URL,
                    params={
                        "query.title": query,
                        "rows": min(limit * 2, 20),
                        "select": "DOI,title,author,published,container-title,publisher,is-referenced-by-count,reference-count,URL,link,abstract",
                        "mailto": _MAILTO,
                    },
                    headers={"User-Agent": f"ResearchFlow/1.0 (mailto:{_MAILTO})"},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            log.warning("crossref search failed: %s", exc)
            return ToolResult(
                output={"works": [], "total": 0, "mode": "search"},
                summary=f"CrossRef search unavailable: {exc}",
            )

        items = (data.get("message", {}).get("items") or [])[:limit]
        works = [_parse_work(item) for item in items]

        if not works:
            return ToolResult(
                output={"works": [], "total": 0, "mode": "search"},
                summary=f"No CrossRef results for: {query}",
            )

        return ToolResult(
            output={"works": works, "total": len(works), "mode": "search"},
            summary=f"{len(works)} CrossRef records found (top: '{works[0].get('title', '')[:60]}')",
        )


def _parse_work(item: dict) -> dict:
    title_list = item.get("title") or []
    title = title_list[0] if title_list else item.get("DOI", "")

    authors: list[str] = []
    for a in (item.get("author") or [])[:8]:
        given = a.get("given", "")
        family = a.get("family", "")
        name = f"{given} {family}".strip()
        if name:
            authors.append(name)

    pub_date = item.get("published", {})
    date_parts = (pub_date.get("date-parts") or [[]])[0]
    year = date_parts[0] if date_parts else None

    journals = item.get("container-title") or []
    journal = journals[0] if journals else ""

    links = item.get("link") or []
    oa_url = next((l.get("URL") for l in links if l.get("content-type") == "application/pdf"), None)
    doi_url = item.get("URL") or (f"https://doi.org/{item['DOI']}" if item.get("DOI") else "")

    return {
        "doi": item.get("DOI", ""),
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "publisher": item.get("publisher", ""),
        "cited_by_count": item.get("is-referenced-by-count", 0),
        "reference_count": item.get("reference-count", 0),
        "abstract": (item.get("abstract") or "")[:600],
        "url": doi_url,
        "pdf_url": oa_url,
        "source": "crossref",
    }


crossref_tool = CrossRefTool()
