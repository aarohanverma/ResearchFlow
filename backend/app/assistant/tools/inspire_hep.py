"""INSPIRE HEP search tool — high-energy physics literature.

https://inspirehep.net/api

Free REST API, no authentication required.
Rate limit: ~30 req/min (be conservative).

Essential for hep-* namespaces: high-energy physics, particle physics, nuclear physics,
field theory. Also covers gr-qc (general relativity), quant-ph (quantum physics),
math-ph (mathematical physics), and nucl-*.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_BASE = "https://inspirehep.net/api"
_TIMEOUT = 20.0


class InspireHepInput(BaseModel):
    query: str = Field(
        min_length=2, max_length=400,
        description=(
            "Search query — natural language, INSPIRE search syntax, or identifiers. "
            "Examples: 'Higgs boson discovery', 't Hooft', 'a Weinberg and t electroweak', "
            "'find eprint 1706.03762', 'texkey:Maldacena:1997re', 'j Phys.Rev.Lett.'. "
            "Supports full INSPIRE search syntax."
        ),
    )
    search_type: str = Field(
        default="literature",
        description="'literature' (papers), 'authors', 'experiments', 'institutions'.",
    )
    limit: int = Field(default=8, ge=1, le=20)
    sort: str = Field(
        default="mostcited",
        description="Sort: 'mostcited', 'mostrecent', 'relevance'.",
    )
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class InspireHepOutput(BaseModel):
    papers: list[dict]
    total_found: int
    search_type: str


class InspireHepTool:
    """Search INSPIRE HEP for high-energy physics literature."""

    name = "inspire_hep"
    summary = (
        "Search INSPIRE HEP for high-energy and particle physics literature (1.4M+ papers). "
        "Use for: particle physics, quantum field theory, string theory, supersymmetry, "
        "collider physics, dark matter theory, neutrino physics, nuclear physics, "
        "general relativity, gravitational waves, quantum gravity, cosmological models. "
        "Preferred over arXiv for hep-ph, hep-th, hep-ex, hep-lat, gr-qc, nucl-th, nucl-ex, math-ph. "
        "Supports INSPIRE query syntax: 'find a author t title j journal'. "
        "Returns: title, authors, citation count, arXiv ID, DOI, journal, inspire ID. "
        "Free API, no key required."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = InspireHepInput
    output_schema = InspireHepOutput

    async def run(self, ctx: ToolContext, params: InspireHepInput) -> ToolResult:
        await ctx.emit_progress(20, f"Searching INSPIRE HEP: {params.query[:60]}")

        q = params.query.strip()
        stype = params.search_type if params.search_type in ("literature", "authors", "experiments", "institutions") else "literature"

        sort_map = {"mostcited": "-citation_count", "mostrecent": "-earliest_date", "relevance": ""}
        sort_param = sort_map.get(params.sort, "-citation_count")

        request_params: dict = {
            "q": q,
            "size": params.limit,
            "fields": "arxiv_eprints,authors,citation_count,dois,inspire_categories,journal_title,preprint_date,publication_info,titles,texkeys",
        }
        if sort_param:
            request_params["sort"] = sort_param

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/{stype}",
                    params=request_params,
                    headers={"User-Agent": "ResearchFlow/1.0"},
                )
                if resp.status_code == 429:
                    return ToolResult(
                        output={"papers": [], "total_found": 0, "search_type": stype},
                        summary="INSPIRE HEP rate limited. Try again shortly.",
                    )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            log.warning("inspire_hep HTTP %s for query %r", exc.response.status_code, q)
            return ToolResult(
                output={"papers": [], "total_found": 0, "search_type": stype},
                summary=f"INSPIRE HEP search failed (HTTP {exc.response.status_code})",
            )
        except Exception as exc:
            log.warning("inspire_hep failed: %s", exc)
            return ToolResult(
                output={"papers": [], "total_found": 0, "search_type": stype},
                summary=f"INSPIRE HEP unavailable: {exc}",
            )

        total = data.get("hits", {}).get("total", 0)
        hits = (data.get("hits", {}).get("hits") or [])[:params.limit]

        papers: list[dict] = []
        for hit in hits:
            meta = hit.get("metadata", {})

            titles = meta.get("titles") or []
            title = (titles[0].get("title", "") if titles else "") or ""

            raw_authors = meta.get("authors") or []
            authors = [
                a.get("full_name", "") for a in raw_authors[:6] if a.get("full_name")
            ]

            arxiv_eprints = meta.get("arxiv_eprints") or []
            arxiv_id = arxiv_eprints[0].get("value", "") if arxiv_eprints else ""

            dois = meta.get("dois") or []
            doi = dois[0].get("value", "") if dois else ""

            pub_info = (meta.get("publication_info") or [{}])[0]
            journal = pub_info.get("journal_title", "")
            year = pub_info.get("year")
            if not year:
                preprint_date = meta.get("preprint_date") or ""
                year = int(preprint_date[:4]) if preprint_date and len(preprint_date) >= 4 else None

            inspire_id = hit.get("id", "")
            categories = [
                c.get("term", "") for c in (meta.get("inspire_categories") or [])[:3]
            ]

            papers.append({
                "inspire_id": inspire_id,
                "title": title,
                "authors": authors,
                "year": year,
                "journal": journal,
                "citation_count": meta.get("citation_count", 0),
                "arxiv_id": arxiv_id,
                "doi": doi,
                "categories": categories,
                "url": f"https://inspirehep.net/literature/{inspire_id}" if stype == "literature" else f"https://inspirehep.net/{stype}/{inspire_id}",
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "source": "inspire_hep",
            })

        await ctx.emit_progress(100, f"INSPIRE HEP: {len(papers)} papers found (total: {total:,})")

        if not papers:
            return ToolResult(
                output={"papers": [], "total_found": total, "search_type": stype},
                summary=f"No INSPIRE HEP results for: {q}",
            )

        top = papers[0]
        return ToolResult(
            output={"papers": papers, "total_found": total, "search_type": stype},
            summary=(
                f"{len(papers)} INSPIRE papers (total: {total:,}) — "
                f"top: '{top['title'][:60]}' ({top.get('year', '?')}, {top.get('citation_count', 0)} citations)"
            ),
        )


inspire_hep_tool = InspireHepTool()
