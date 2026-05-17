"""ClinicalTrials.gov search tool — clinical study registry.

Uses the ClinicalTrials.gov v2 API (free, no auth required).
https://clinicaltrials.gov/data-api/api

Essential for q-bio.* namespaces: medicine, clinical research,
pharmacology, public health, drug trials, interventional studies.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_CT_URL = "https://clinicaltrials.gov/api/v2/studies"
_TIMEOUT = 20.0


class ClinicalTrialsInput(BaseModel):
    query: str = Field(
        min_length=2, max_length=400,
        description=(
            "Search query — condition ('breast cancer'), intervention ('pembrolizumab'), "
            "sponsor, or NCT ID ('NCT04280705')."
        ),
    )
    status: str = Field(
        default="",
        description=(
            "Filter by status: 'RECRUITING', 'ACTIVE_NOT_RECRUITING', 'COMPLETED', "
            "'TERMINATED', 'WITHDRAWN'. Empty = all statuses."
        ),
    )
    phase: str = Field(
        default="",
        description="Filter by phase: 'PHASE1', 'PHASE2', 'PHASE3', 'PHASE4'.",
    )
    limit: int = Field(default=8, ge=1, le=20)
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class ClinicalTrialsOutput(BaseModel):
    studies: list[dict]
    total_found: int


class ClinicalTrialsTool:
    """Search ClinicalTrials.gov for registered clinical studies."""

    name = "clinicaltrials"
    summary = (
        "Search ClinicalTrials.gov for registered clinical studies — trials, interventional studies, "
        "observational studies. Use for: 'clinical trials for [condition]', 'phase 3 trials of [drug]', "
        "'recruiting studies for [disease]', 'NCT04280705 details'. "
        "Returns NCT ID, title, status, phase, conditions, interventions, sponsor, start/end dates. "
        "Free API, no key required. Preferred for q-bio.* and medicine queries."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = ClinicalTrialsInput
    output_schema = ClinicalTrialsOutput

    async def run(self, ctx: ToolContext, params: ClinicalTrialsInput) -> ToolResult:
        await ctx.emit_progress(20, f"Searching ClinicalTrials.gov: {params.query[:60]}")

        q = params.query.strip()
        is_nct = q.upper().startswith("NCT")

        request_params: dict = {
            "pageSize": min(params.limit * 2, 20),
            "format": "json",
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,Condition,InterventionName,LeadSponsorName,StartDate,PrimaryCompletionDate,EnrollmentCount,BriefSummary",
        }
        if is_nct:
            request_params["query.id"] = q.upper()
        else:
            request_params["query.term"] = q
        if params.status:
            request_params["filter.overallStatus"] = params.status.upper()
        if params.phase:
            request_params["filter.phase"] = params.phase.upper()

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    _CT_URL, params=request_params,
                    headers={"User-Agent": "ResearchFlow/1.0"},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            log.warning("clinicaltrials search failed: %s", exc)
            return ToolResult(
                output={"studies": [], "total_found": 0},
                summary=f"ClinicalTrials.gov unavailable: {exc}",
            )

        total = data.get("totalCount", 0)
        raw_studies = (data.get("studies") or [])[:params.limit]

        studies: list[dict] = []
        for s in raw_studies:
            ps = s.get("protocolSection", {})
            id_mod = ps.get("identificationModule", {})
            status_mod = ps.get("statusModule", {})
            desc_mod = ps.get("descriptionModule", {})
            design_mod = ps.get("designModule", {})
            cond_mod = ps.get("conditionsModule", {})
            arms_mod = ps.get("armsInterventionsModule", {})
            sponsor_mod = ps.get("sponsorCollaboratorsModule", {})

            nct_id = id_mod.get("nctId", "")
            conditions = (cond_mod.get("conditions") or [])[:5]
            interventions = [
                iv.get("interventionName", "") for iv in (arms_mod.get("interventions") or [])[:4]
                if iv.get("interventionName")
            ]
            phase_list = design_mod.get("phases") or []
            phase = ", ".join(phase_list) if phase_list else ""

            studies.append({
                "nct_id": nct_id,
                "title": id_mod.get("briefTitle", ""),
                "status": status_mod.get("overallStatus", ""),
                "phase": phase,
                "conditions": conditions,
                "interventions": interventions,
                "sponsor": (sponsor_mod.get("leadSponsor") or {}).get("name", ""),
                "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
                "completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
                "enrollment": design_mod.get("enrollmentInfo", {}).get("count"),
                "summary": (desc_mod.get("briefSummary") or "")[:600],
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
                "source": "clinicaltrials",
            })

        await ctx.emit_progress(100, f"ClinicalTrials.gov: {len(studies)} studies found (total: {total:,})")
        if not studies:
            return ToolResult(
                output={"studies": [], "total_found": total},
                summary=f"No clinical trials found for: {q}",
            )
        top = studies[0]
        return ToolResult(
            output={"studies": studies, "total_found": total},
            summary=(
                f"{len(studies)} trials (total: {total:,}) — "
                f"top: '{top['title'][:60]}' ({top.get('status', '')} {top.get('phase', '')})"
            ),
        )


clinicaltrials_tool = ClinicalTrialsTool()
