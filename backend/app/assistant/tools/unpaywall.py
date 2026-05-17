"""Unpaywall tool — find free open-access versions of paywalled papers.

Uses the Unpaywall REST API (free, no key required — just a polite email header)
to find legally free PDF versions of papers given a DOI. Returns the best OA
location: repository preprint, PubMed Central, institutional repository, or
publisher's own OA version.

Essential when the user has a DOI but the paper is behind a paywall.
"""

from __future__ import annotations

import logging
import urllib.parse

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_BASE = "https://api.unpaywall.org/v2"
_MAILTO = "researchflow@example.com"
_TIMEOUT = 12.0


class UnpaywallInput(BaseModel):
    doi: str = Field(min_length=5, max_length=300, description="DOI of the paper (e.g. '10.1038/nature12373' or full URL)")
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class UnpaywallOutput(BaseModel):
    doi: str
    title: str
    is_oa: bool
    oa_status: str      # "gold", "green", "hybrid", "bronze", "closed"
    best_oa_url: str
    best_oa_version: str  # "publishedVersion", "acceptedVersion", "submittedVersion"
    license: str
    journal: str
    year: int | None
    all_oa_locations: list[dict]


class UnpaywallTool:
    """Find free open-access PDF versions of paywalled papers via Unpaywall."""

    name = "unpaywall"
    summary = (
        "Find a free, legal open-access PDF of a paper using its DOI. "
        "Checks publisher OA, PubMed Central, institutional repositories, and preprint servers. "
        "Use when: the user has a DOI and wants to read the paper but may not have access, "
        "'find the free version of DOI:10.xxx', 'is this paper open access?', "
        "'find PDF for [paper DOI]'. Returns the best available OA URL and license info. "
        "Free API, no key needed."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = UnpaywallInput
    output_schema = UnpaywallOutput

    async def run(self, ctx: ToolContext, params: UnpaywallInput) -> ToolResult:
        doi = _clean_doi(params.doi)
        await ctx.emit_progress(20, f"Checking Unpaywall for DOI: {doi}")

        try:
            encoded = urllib.parse.quote(doi, safe="")
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BASE}/{encoded}",
                    params={"email": _MAILTO},
                    headers={"User-Agent": "ResearchFlow/1.0"},
                )
                if resp.status_code == 404:
                    return ToolResult(
                        output=_empty(doi),
                        summary=f"DOI not found in Unpaywall: {doi}",
                    )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            log.warning("unpaywall failed for %s: %s", doi, exc)
            return ToolResult(
                output=_empty(doi),
                summary=f"Unpaywall unavailable: {exc}",
            )

        is_oa = bool(data.get("is_oa"))
        oa_status = data.get("oa_status") or "closed"
        title = data.get("title") or ""
        year = data.get("year")
        journal = data.get("journal_name") or ""

        best_url = ""
        best_version = ""
        best_license = ""
        all_locations: list[dict] = []

        best_loc = data.get("best_oa_location") or {}
        if best_loc:
            best_url = best_loc.get("url_for_pdf") or best_loc.get("url") or ""
            best_version = best_loc.get("version") or ""
            best_license = best_loc.get("license") or ""

        for loc in (data.get("oa_locations") or [])[:6]:
            url = loc.get("url_for_pdf") or loc.get("url") or ""
            if url:
                all_locations.append({
                    "url": url,
                    "version": loc.get("version", ""),
                    "host_type": loc.get("host_type", ""),
                    "license": loc.get("license", ""),
                    "repository_institution": loc.get("repository_institution", ""),
                })

        await ctx.emit_progress(100, f"Unpaywall: {'OA found' if best_url else 'no OA version'}")

        status_msg = f"OA ({oa_status}): {best_url[:80]}" if best_url else f"No OA version (status: {oa_status})"
        return ToolResult(
            output={
                "doi": doi,
                "title": title,
                "is_oa": is_oa,
                "oa_status": oa_status,
                "best_oa_url": best_url,
                "best_oa_version": best_version,
                "license": best_license,
                "journal": journal,
                "year": year,
                "all_oa_locations": all_locations,
            },
            summary=f"Unpaywall: '{title[:60]}' — {status_msg}",
        )


def _clean_doi(raw: str) -> str:
    raw = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "DOI:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    return raw.strip()


def _empty(doi: str) -> dict:
    return {
        "doi": doi, "title": "", "is_oa": False, "oa_status": "closed",
        "best_oa_url": "", "best_oa_version": "", "license": "", "journal": "",
        "year": None, "all_oa_locations": [],
    }


unpaywall_tool = UnpaywallTool()
