"""NVD/CVE search tool — security vulnerability lookup via NIST NVD API v2.

Free REST API (https://nvd.nist.gov/developers/vulnerabilities).
No auth required for basic queries (5 req / 30 s).
Optional NVD_API_KEY env var raises that to 50 req / 30 s.

Use for cs.CR, cs.SE, eess.SY namespaces: CVE lookups, vulnerability
research, security assessments, affected-product queries.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_TIMEOUT = 20.0


class NvdCveInput(BaseModel):
    query: str = Field(
        min_length=2, max_length=300,
        description=(
            "CVE ID ('CVE-2023-1234'), product name, keyword, or vulnerability type. "
            "Supports NVD keyword search syntax."
        ),
    )
    limit: int = Field(default=8, ge=1, le=20)
    severity: str = Field(
        default="",
        description="Optional CVSS severity filter: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'.",
    )
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class NvdCveOutput(BaseModel):
    vulnerabilities: list[dict]
    total_results: int


class NvdCveTool:
    """Search NIST NVD for CVEs and security vulnerabilities."""

    name = "nvd_cve"
    summary = (
        "Search the NIST National Vulnerability Database for CVEs and security vulnerabilities. "
        "Use for: 'CVE-2023-XXXX details', 'vulnerabilities in OpenSSL', 'recent critical CVEs in Apache', "
        "'what is the CVSS score for X'. Returns CVE ID, description, CVSS score, severity, "
        "affected products, published date, and NVD reference links. "
        "Free NIST API, no key required. Preferred for cs.CR, cs.SE security research."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = NvdCveInput
    output_schema = NvdCveOutput

    async def run(self, ctx: ToolContext, params: NvdCveInput) -> ToolResult:
        await ctx.emit_progress(20, f"Searching NVD for: {params.query[:60]}")

        q = params.query.strip()
        is_cve_id = q.upper().startswith("CVE-")

        request_params: dict = {"resultsPerPage": min(params.limit * 2, 20)}
        if is_cve_id:
            request_params["cveId"] = q.upper()
        else:
            request_params["keywordSearch"] = q
        if params.severity and params.severity.upper() in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            request_params["cvssV3Severity"] = params.severity.upper()

        headers: dict = {"User-Agent": "ResearchFlow/1.0 (security research tool)"}
        try:
            from app.core.config import get_settings
            key = getattr(get_settings(), "nvd_api_key", "") or ""
            if key:
                headers["apiKey"] = key
        except Exception:
            pass

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_NVD_URL, params=request_params, headers=headers)
                if resp.status_code == 403:
                    return ToolResult(
                        output={"vulnerabilities": [], "total_results": 0},
                        summary="NVD API rate-limited. Set NVD_API_KEY env var for higher rate limits.",
                    )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            log.warning("nvd_cve HTTP %s for query %r", exc.response.status_code, q)
            return ToolResult(
                output={"vulnerabilities": [], "total_results": 0},
                summary=f"NVD search failed (HTTP {exc.response.status_code})",
            )
        except Exception as exc:
            log.warning("nvd_cve failed: %s", exc)
            return ToolResult(
                output={"vulnerabilities": [], "total_results": 0},
                summary=f"NVD unavailable: {exc}",
            )

        total = data.get("totalResults", 0)
        raw_vulns = (data.get("vulnerabilities") or [])[:params.limit]

        vulns: list[dict] = []
        for item in raw_vulns:
            cve = item.get("cve", {})
            cve_id = cve.get("id", "")
            descs = cve.get("descriptions") or []
            desc = next((d["value"] for d in descs if d.get("lang") == "en"), "")

            metrics = cve.get("metrics", {})
            cvss_list = metrics.get("cvssMetricV31") or metrics.get("cvssMetricV30") or []
            cvss_data = (cvss_list[0].get("cvssData") if cvss_list else {}) or {}
            score = cvss_data.get("baseScore")
            severity = cvss_data.get("baseSeverity", "")

            affected: list[str] = []
            for cfg in (cve.get("configurations") or [])[:2]:
                for node in cfg.get("nodes", []):
                    for cpe in node.get("cpeMatch", [])[:3]:
                        parts = (cpe.get("criteria") or "").split(":")
                        if len(parts) >= 5:
                            affected.append(f"{parts[3]} {parts[4]}")

            refs = [r.get("url", "") for r in (cve.get("references") or [])[:3] if r.get("url")]

            vulns.append({
                "cve_id": cve_id,
                "description": desc[:600],
                "cvss_score": score,
                "severity": severity,
                "affected_products": affected[:6],
                "published": (cve.get("published") or "")[:10],
                "last_modified": (cve.get("lastModified") or "")[:10],
                "references": refs,
                "url": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                "source": "nvd",
            })

        await ctx.emit_progress(100, f"NVD: {len(vulns)} CVEs found (total: {total:,})")
        if not vulns:
            return ToolResult(
                output={"vulnerabilities": [], "total_results": total},
                summary=f"No NVD entries found for: {q}",
            )
        top = vulns[0]
        return ToolResult(
            output={"vulnerabilities": vulns, "total_results": total},
            summary=(
                f"{len(vulns)} CVEs (total: {total:,}) — "
                f"top: {top['cve_id']} CVSS {top.get('cvss_score', '?')} ({top.get('severity', '?')})"
            ),
        )


nvd_cve_tool = NvdCveTool()
