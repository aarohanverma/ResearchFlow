"""NASA ADS (Astrophysics Data System) search tool.

https://ui.adsabs.harvard.edu/help/api/

Requires ADS_API_TOKEN in environment (free token from ui.adsabs.harvard.edu/user/settings/token).
Without a token, returns a clear error message.

Essential for astro-ph.* namespaces: astrophysics, cosmology, planetary science,
solar physics, galactic astronomy, space science, instrumentation.
Also useful for physics.* with astronomical overlap.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
_TIMEOUT = 20.0


class NasaAdsInput(BaseModel):
    query: str = Field(
        min_length=2, max_length=400,
        description=(
            "Search query — natural language ('dark matter detection'), author ('author:Planck'), "
            "bibcode ('2019ApJ...884L..33L'), title phrase ('title:gravitational waves'), "
            "or ADS query syntax. Supports full ADS search syntax."
        ),
    )
    limit: int = Field(default=8, ge=1, le=20)
    sort: str = Field(
        default="citation_count desc",
        description="Sort: 'citation_count desc', 'date desc', 'score desc' (relevance).",
    )
    refereed_only: bool = Field(
        default=False,
        description="If true, restrict to peer-reviewed (refereed) publications.",
    )
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class NasaAdsOutput(BaseModel):
    papers: list[dict]
    total_found: int


class NasaAdsTool:
    """Search NASA ADS for astrophysics and space science literature."""

    name = "nasa_ads"
    summary = (
        "Search NASA ADS (Astrophysics Data System) for astronomy, astrophysics, cosmology, "
        "planetary science, solar physics, and space instrumentation literature. "
        "ADS covers 15M+ records including journals, conference proceedings, and preprints. "
        "Use for: any astro-ph.* query, cosmology, dark matter, exoplanets, telescopes, missions, "
        "stellar physics, galactic structure, gravitational waves, CMB, etc. "
        "Supports author, title, abstract, bibcode searches and ADS query syntax. "
        "Returns: bibcode, title, authors, abstract, citation count, journal, year, arXiv ID. "
        "Requires ADS_API_TOKEN env var (free from ui.adsabs.harvard.edu)."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = NasaAdsInput
    output_schema = NasaAdsOutput

    def _get_token(self) -> str:
        try:
            from app.core.config import get_settings
            return getattr(get_settings(), "ads_api_token", "") or ""
        except Exception:
            return ""

    async def run(self, ctx: ToolContext, params: NasaAdsInput) -> ToolResult:
        token = self._get_token()
        if not token:
            return ToolResult(
                output={"papers": [], "total_found": 0},
                summary=(
                    "NASA ADS token not configured. Set ADS_API_TOKEN in environment "
                    "(free token from https://ui.adsabs.harvard.edu/user/settings/token)."
                ),
            )

        await ctx.emit_progress(20, f"Searching NASA ADS: {params.query[:60]}")

        q = params.query.strip()
        if params.refereed_only:
            q = f"({q}) property:refereed"

        valid_sorts = {"citation_count desc", "date desc", "score desc", "citation_count asc", "date asc"}
        sort = params.sort if params.sort in valid_sorts else "citation_count desc"

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    _ADS_SEARCH_URL,
                    params={
                        "q": q,
                        "fl": "bibcode,title,author,abstract,citation_count,pub,year,identifier,doi,arxiv_class",
                        "rows": params.limit,
                        "sort": sort,
                    },
                    headers={
                        "Authorization": f"Bearer {token}",
                        "User-Agent": "ResearchFlow/1.0",
                    },
                )
                if resp.status_code == 401:
                    return ToolResult(
                        output={"papers": [], "total_found": 0},
                        summary="NASA ADS token invalid or expired. Update ADS_API_TOKEN.",
                    )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            log.warning("nasa_ads HTTP %s for query %r", exc.response.status_code, q)
            return ToolResult(
                output={"papers": [], "total_found": 0},
                summary=f"NASA ADS search failed (HTTP {exc.response.status_code})",
            )
        except Exception as exc:
            log.warning("nasa_ads failed: %s", exc)
            return ToolResult(
                output={"papers": [], "total_found": 0},
                summary=f"NASA ADS unavailable: {exc}",
            )

        response = data.get("response", {})
        total = response.get("numFound", 0)
        docs = response.get("docs") or []

        papers: list[dict] = []
        for doc in docs[:params.limit]:
            bibcode = doc.get("bibcode", "")
            title_field = doc.get("title")
            title = (title_field[0] if isinstance(title_field, list) else title_field) or ""
            authors = (doc.get("author") or [])[:6]

            identifiers = doc.get("identifier") or []
            arxiv_id = ""
            doi = ""
            for ident in identifiers:
                if isinstance(ident, str):
                    if ident.startswith("arXiv:"):
                        arxiv_id = ident[6:]
                    elif ident.startswith("10."):
                        doi = ident

            doi_list = doc.get("doi")
            if not doi and doi_list:
                doi = doi_list[0] if isinstance(doi_list, list) else doi_list

            abstract = doc.get("abstract") or ""

            papers.append({
                "bibcode": bibcode,
                "title": title,
                "authors": authors,
                "year": doc.get("year"),
                "journal": doc.get("pub", ""),
                "citation_count": doc.get("citation_count", 0),
                "abstract": abstract[:600],
                "arxiv_id": arxiv_id,
                "doi": doi,
                "url": f"https://ui.adsabs.harvard.edu/abs/{bibcode}",
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "source": "nasa_ads",
            })

        await ctx.emit_progress(100, f"NASA ADS: {len(papers)} papers found (total: {total:,})")

        if not papers:
            return ToolResult(
                output={"papers": [], "total_found": total},
                summary=f"No NASA ADS results for: {q}",
            )

        top = papers[0]
        return ToolResult(
            output={"papers": papers, "total_found": total},
            summary=(
                f"{len(papers)} ADS papers (total: {total:,}) — "
                f"top: '{top['title'][:60]}' ({top.get('year', '?')}, {top.get('citation_count', 0)} citations)"
            ),
        )


nasa_ads_tool = NasaAdsTool()
