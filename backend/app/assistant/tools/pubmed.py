"""PubMed tool — biomedical literature search via NCBI E-utilities.

Uses the free NCBI E-utilities API to search PubMed (biomedical literature,
32M+ articles) and PubMed Central (full-text OA biomedical articles).

Essential for any biomedical, life sciences, clinical, or health research.
Without this, ResearchFlow has no biomedical paper coverage at all.

Optional: set NCBI_API_KEY in environment for 10 req/sec instead of 3 req/sec.
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult
from app.core.config import get_settings

log = logging.getLogger(__name__)

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ESUM    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_TIMEOUT = 20.0
_TOOL    = "ResearchFlow"
_EMAIL   = "researchflow@example.com"


class PubMedInput(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="PubMed search query (supports MeSH terms, author, journal, date filters)")
    database: str = Field(default="pubmed", description="'pubmed' (abstracts) or 'pmc' (full-text OA articles)")
    limit: int = Field(default=8, ge=1, le=20)
    sort: str = Field(default="relevance", description="'relevance' or 'pub_date'")
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class PubMedOutput(BaseModel):
    papers: list[dict]
    total_found: int
    database: str


class PubMedTool:
    """Search PubMed/PMC for biomedical literature via NCBI E-utilities."""

    name = "pubmed"
    summary = (
        "Search PubMed (32M+ biomedical articles) or PubMed Central (full-text OA) "
        "for life sciences, medical, clinical, and biology research. "
        "Use for: any query in biology, medicine, biochemistry, genetics, neuroscience, "
        "pharmacology, clinical trials, public health — fields not well-covered by arXiv. "
        "Supports MeSH terms, author filters, date ranges. "
        "Returns PMID, title, authors, abstract, journal, DOI, publication date. "
        "Free NCBI E-utilities API, no key required (optional key for higher rate limits)."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = PubMedInput
    output_schema = PubMedOutput

    def _api_params(self) -> dict:
        p: dict = {"tool": _TOOL, "email": _EMAIL}
        key = getattr(get_settings(), "ncbi_api_key", "") or ""
        if key:
            p["api_key"] = key
        return p

    async def run(self, ctx: ToolContext, params: PubMedInput) -> ToolResult:
        await ctx.emit_progress(15, f"Searching PubMed [{params.database}]: {params.query[:60]}")

        base_params = self._api_params()
        db = "pmc" if params.database == "pmc" else "pubmed"

        # Step 1: esearch — get PMIDs
        pmids: list[str] = []
        total_found = 0
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                search_resp = await client.get(
                    _ESEARCH,
                    params={
                        **base_params,
                        "db": db,
                        "term": params.query,
                        "retmax": params.limit * 2,
                        "sort": "relevance" if params.sort == "relevance" else "pub_date",
                        "retmode": "json",
                        "usehistory": "n",
                    },
                )
                search_resp.raise_for_status()
                sd = search_resp.json()
                esearch = sd.get("esearchresult", {})
                pmids = (esearch.get("idlist") or [])[:params.limit]
                try:
                    total_found = int(esearch.get("count", 0))
                except (ValueError, TypeError):
                    total_found = len(pmids)
        except Exception as exc:
            log.warning("pubmed esearch failed: %s", exc)
            return ToolResult(
                output={"papers": [], "total_found": 0, "database": db},
                summary=f"PubMed search failed: {exc}",
            )

        if not pmids:
            return ToolResult(
                output={"papers": [], "total_found": total_found, "database": db},
                summary=f"No PubMed results for: {params.query}",
            )

        await ctx.emit_progress(50, f"Fetching {len(pmids)} article summaries…")

        # Step 2: esummary — get structured metadata
        papers: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                sum_resp = await client.get(
                    _ESUM,
                    params={
                        **base_params,
                        "db": db,
                        "id": ",".join(pmids),
                        "retmode": "json",
                    },
                )
                sum_resp.raise_for_status()
                sum_data = sum_resp.json()
                uid_map = sum_data.get("result", {})
                for uid in pmids:
                    art = uid_map.get(uid)
                    if not art or not isinstance(art, dict):
                        continue
                    authors = [
                        a.get("name", "") for a in (art.get("authors") or [])[:6]
                    ]
                    doi = ""
                    for id_item in (art.get("articleids") or []):
                        if id_item.get("idtype") == "doi":
                            doi = id_item.get("value", "")
                            break
                    pub_date = art.get("pubdate") or art.get("sortpubdate", "")
                    year: int | None = None
                    try:
                        year = int(pub_date[:4]) if pub_date else None
                    except ValueError:
                        pass
                    papers.append({
                        "pmid": uid,
                        "title": art.get("title", "").rstrip("."),
                        "authors": authors,
                        "journal": art.get("fulljournalname") or art.get("source", ""),
                        "year": year,
                        "doi": doi,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                        "abstract": "",  # esummary doesn't include abstract; would need efetch
                        "source": "pubmed",
                    })
        except Exception as exc:
            log.warning("pubmed esummary failed: %s", exc)

        # Step 3: fetch abstracts for top-3 papers using efetch
        if papers:
            await ctx.emit_progress(75, "Fetching abstracts…")
            top_ids = [p["pmid"] for p in papers[:3]]
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    fetch_resp = await client.get(
                        _EFETCH,
                        params={
                            **base_params,
                            "db": db,
                            "id": ",".join(top_ids),
                            "retmode": "xml",
                            "rettype": "abstract",
                        },
                    )
                    if fetch_resp.status_code == 200:
                        abstracts = _parse_abstracts_xml(fetch_resp.text)
                        for p in papers[:3]:
                            p["abstract"] = abstracts.get(p["pmid"], "")[:600]
            except Exception as exc:
                log.warning("pubmed efetch abstracts failed: %s", exc)

        await ctx.emit_progress(100, f"PubMed: {len(papers)} articles found")

        return ToolResult(
            output={"papers": papers, "total_found": total_found, "database": db},
            summary=(
                f"{len(papers)} PubMed articles found (total: {total_found:,}) "
                + (f"— top: '{papers[0]['title'][:60]}'" if papers else "")
            ),
        )


def _parse_abstracts_xml(xml_text: str) -> dict[str, str]:
    """Extract PMID → abstract text from PubMed efetch XML."""
    abstracts: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
        for article in root.iter("PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""
            abs_texts = [
                t.strip()
                for el in article.findall(".//AbstractText")
                if (t := (el.text or "")) .strip()
            ]
            if pmid and abs_texts:
                abstracts[pmid] = " ".join(abs_texts)
    except Exception as exc:
        log.warning("pubmed XML parse failed: %s", exc)
    return abstracts


pubmed_tool = PubMedTool()
