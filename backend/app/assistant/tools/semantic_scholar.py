"""Semantic Scholar search tool.

Queries the Semantic Scholar Academic Graph API (free, no key required for basic
search) to discover papers with citation counts, influential citations, open-access
PDFs, and rich metadata not always available on arXiv. Complements deep_search
and arxiv_search by surfacing highly-cited, peer-reviewed work.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult

log = logging.getLogger(__name__)

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,year,abstract,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,externalIds,publicationDate"
_TIMEOUT = 15.0


class SemanticScholarInput(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="Research query or paper title to search")
    limit: int = Field(default=8, ge=1, le=20)
    min_citations: int = Field(default=0, ge=0, description="Filter out papers with fewer citations")
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class SemanticScholarOutput(BaseModel):
    papers: list[dict]
    total: int


class SemanticScholarTool:
    """Search Semantic Scholar for peer-reviewed papers with citation data."""

    name = "semantic_scholar"
    summary = (
        "Search Semantic Scholar's academic graph for papers with citation counts, "
        "influential citation metrics, and open-access PDFs. Use when: (a) the user "
        "wants highly-cited or landmark papers in a field, (b) you need peer-reviewed "
        "work beyond arXiv preprints, (c) citation impact is relevant to the query "
        "('most influential papers on X', 'seminal work in Y'). Complements deep_search "
        "and arxiv_search — no API key needed."
    )
    cost_class = "moderate"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = SemanticScholarInput
    output_schema = SemanticScholarOutput

    async def run(self, ctx: ToolContext, params: SemanticScholarInput) -> ToolResult:
        await ctx.emit_progress(20, f"Searching Semantic Scholar: {params.query[:60]}")

        papers: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    _S2_SEARCH_URL,
                    params={
                        "query": params.query,
                        "limit": min(params.limit * 2, 20),
                        "fields": _S2_FIELDS,
                    },
                    headers={"User-Agent": "ResearchFlow/1.0 (research assistant)"},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(2)
                return ToolResult(
                    output={"papers": [], "total": 0},
                    summary="Semantic Scholar rate-limited; try again shortly",
                )
            raise
        except Exception as exc:
            log.warning("semantic_scholar search failed: %s", exc)
            return ToolResult(
                output={"papers": [], "total": 0},
                summary=f"Semantic Scholar unavailable: {exc}",
            )

        raw_papers = data.get("data") or []
        for p in raw_papers:
            citations = p.get("citationCount") or 0
            if citations < params.min_citations:
                continue
            authors = [a.get("name", "") for a in (p.get("authors") or [])]
            ext_ids = p.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv")
            doi = ext_ids.get("DOI")
            oa_pdf = (p.get("openAccessPdf") or {}).get("url")
            papers.append({
                "title": p.get("title", ""),
                "abstract": (p.get("abstract") or "")[:800],
                "authors": authors[:8],
                "year": p.get("year"),
                "citation_count": citations,
                "influential_citations": p.get("influentialCitationCount") or 0,
                "is_open_access": bool(p.get("isOpenAccess")),
                "pdf_url": oa_pdf,
                "arxiv_id": arxiv_id,
                "doi": doi,
                "source": "semantic_scholar",
            })
            if len(papers) >= params.limit:
                break

        await ctx.emit_progress(100, f"Found {len(papers)} papers on Semantic Scholar")

        if not papers:
            return ToolResult(
                output={"papers": [], "total": 0},
                summary="No Semantic Scholar results for this query",
            )

        top = sorted(papers, key=lambda x: x["citation_count"], reverse=True)
        return ToolResult(
            output={"papers": top, "total": len(top)},
            summary=(
                f"{len(top)} Semantic Scholar papers found "
                f"(top: '{top[0]['title'][:60]}', {top[0]['citation_count']} citations)"
            ),
        )


semantic_scholar_tool = SemanticScholarTool()
