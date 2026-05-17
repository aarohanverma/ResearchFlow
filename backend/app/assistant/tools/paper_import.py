"""Manual paper import tool — full pipeline for a specific arXiv ID.

Mirrors the manual "Import arXiv paper" button (POST /papers/import-arxiv).
Use when the user already knows the exact paper they want — name + ID, or
a fully-resolved citation — rather than wanting to search-and-pick.

The orchestrator can chain this after a citation_finder / crossref result
that resolved to specific arXiv IDs, or invoke it directly when the user
pastes an arXiv URL into chat.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import feedparser
import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import AssistantTool, ToolContext, ToolResult

log = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")


def _normalise_arxiv_id(raw: str) -> str | None:
    """Strip URL prefixes and version suffixes; return canonical NNNN.NNNNN or None."""
    raw = raw.strip()
    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "arxiv.org/abs/",
        "arxiv:",
        "arXiv:",
    ):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    m = _ARXIV_ID_RE.match(raw)
    return m.group(1) if m else None


async def _fetch_arxiv_entry(arxiv_id: str) -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://export.arxiv.org/api/query",
                params={"id_list": arxiv_id, "max_results": "1"},
            )
            resp.raise_for_status()
    except Exception as exc:
        log.warning("paper_import: arXiv fetch failed id=%s err=%s", arxiv_id, exc)
        return None
    feed = feedparser.parse(resp.text)
    if not feed.entries:
        return None
    entry = feed.entries[0]
    if "missing" in entry.get("title", "").lower() or not entry.get("title"):
        return None
    return entry


class PaperImportInput(BaseModel):
    arxiv_ids: list[str] = Field(
        min_length=1,
        max_length=10,
        description=(
            "One or more arXiv IDs (e.g. '1706.03762'), arXiv URLs, or 'arXiv:...' "
            "citations. Versions and URL prefixes are stripped automatically."
        ),
    )
    namespace_key: str | None = Field(
        default=None,
        description=(
            "Target namespace for the import. Defaults to the session's active "
            "namespace. Each imported paper is marked is_manually_imported=True "
            "so it is always visible in that namespace's feed and search."
        ),
    )


class PaperImportOutput(BaseModel):
    imported: int
    duplicates: int
    failed: int
    paper_ids: list[str]
    items: list[dict]


class PaperImportTool:
    """Import specific arXiv papers (by ID) directly into the user's feed.

    Same pipeline as the manual import button: validate → store → embed →
    knowledge-graph indexing → LLM enrichment → mark as manually imported.
    Idempotent: re-importing the same ID is a no-op (counted as duplicate).
    """

    name = "paper_import"
    summary = (
        "Import specific arXiv papers (by ID or URL) into the user's feed and "
        "knowledge base. Use when the user already knows exactly which paper "
        "they want — pasted URL, mentioned arXiv ID, or a resolved citation. "
        "Persists Paper rows, embeds abstracts, indexes in the knowledge graph, "
        "enriches via LLM, and marks them as manually imported so they always "
        "appear in feed/search regardless of date filters. For exploratory "
        "search-and-import use `arxiv_import` instead."
    )
    cost_class = "moderate"
    side_effects = True
    cancellable = True
    streamable = True
    input_schema = PaperImportInput
    output_schema = PaperImportOutput

    async def run(self, ctx: ToolContext, params: PaperImportInput) -> ToolResult:
        from sqlalchemy import select as _sel, update as _upd

        from app.adapters.sources.base import RawPaper
        from app.models.paper import Paper
        from app.repositories.paper import PaperRepository
        from app.services.arxiv_import import ArxivImportService

        target_ns = params.namespace_key or ctx.namespace_key

        # Normalise + dedupe input IDs
        canonical: list[str] = []
        seen: set[str] = set()
        failed_ids: list[str] = []
        for raw in params.arxiv_ids:
            cid = _normalise_arxiv_id(raw)
            if cid and cid not in seen:
                canonical.append(cid)
                seen.add(cid)
            elif not cid:
                failed_ids.append(raw)

        if not canonical:
            await ctx.emit_progress(100, "No valid arXiv IDs supplied")
            return ToolResult(
                output={
                    "imported": 0,
                    "duplicates": 0,
                    "failed": len(failed_ids),
                    "paper_ids": [],
                    "items": [
                        {"input": r, "status": "invalid_id"} for r in failed_ids
                    ],
                },
                summary="No valid arXiv IDs supplied",
            )

        total = len(canonical)
        imported = 0
        duplicates = 0
        paper_ids: list[str] = []
        items: list[dict] = []
        svc = ArxivImportService(ctx.db)

        for idx, arxiv_id in enumerate(canonical):
            progress = 5 + int((idx / total) * 90)
            await ctx.emit_progress(progress, f"Fetching arXiv:{arxiv_id}")

            if await ctx.should_cancel():
                items.append({"arxiv_id": arxiv_id, "status": "cancelled"})
                continue

            entry = await _fetch_arxiv_entry(arxiv_id)
            if not entry:
                failed_ids.append(arxiv_id)
                items.append({"arxiv_id": arxiv_id, "status": "not_found"})
                continue

            try:
                title = re.sub(r"\s+", " ", entry.get("title", "")).strip()
                abstract = (entry.get("summary") or entry.get("description") or "").strip()
                authors = [a.get("name", "Unknown") for a in entry.get("authors", [])] or ["Unknown"]
                published_at = None
                if entry.get("published"):
                    try:
                        published_at = datetime.fromisoformat(
                            entry["published"].replace("Z", "+00:00")
                        ).astimezone(timezone.utc)
                    except Exception:
                        pass

                raw_paper = RawPaper(
                    external_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    source_url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                    published_at=published_at,
                    namespace_key=target_ns,
                    raw=dict(entry),
                )

                new_papers, _ = await svc.import_raw_papers(
                    [raw_paper],
                    namespace_key=target_ns,
                    create_embeddings=True,
                    update_graph=True,
                )

                # Resolve the paper row (whether newly inserted or pre-existing)
                row = await ctx.db.execute(
                    _sel(Paper).where(
                        Paper.external_id == arxiv_id,
                        Paper.namespace_key == target_ns,
                    )
                )
                paper = row.scalar_one_or_none()
                if not paper:
                    items.append({"arxiv_id": arxiv_id, "status": "store_failed"})
                    failed_ids.append(arxiv_id)
                    continue

                was_new = bool(new_papers)

                # LLM enrichment only for genuinely new papers
                if was_new:
                    try:
                        from app.adapters.llm import get_llm_adapter
                        from app.workflows.ingestion import (
                            _ENRICHMENT_SYSTEM,
                            _coerce_enrichment_item,
                            _parse_enrichment_items,
                        )

                        llm = get_llm_adapter()
                        paper_list = (
                            f"[PAPER 0]\n[START]\n{paper.title}\n\n{paper.abstract}\n[END]"
                        )
                        res = await llm.complete(
                            [
                                {"role": "system", "content": _ENRICHMENT_SYSTEM},
                                {"role": "user", "content": f"Analyze these 1 papers:\n\n{paper_list}"},
                            ],
                            llm.cheap_model,
                            response_format={"type": "json_object"},
                        )
                        parsed = _parse_enrichment_items(res.text)
                        if parsed:
                            enrichment = _coerce_enrichment_item(parsed[0])
                            if not enrichment.get("tldr"):
                                enrichment.pop("tldr", None)
                            await PaperRepository(ctx.db).update_enrichment(paper.id, enrichment)
                            await ctx.db.commit()
                            await ctx.db.refresh(paper)
                    except Exception as exc:
                        log.warning("paper_import: enrichment failed id=%s err=%s", arxiv_id, exc)

                # Mark as manually imported so feed/search always surface it
                try:
                    await ctx.db.execute(
                        _upd(Paper)
                        .where(Paper.id == paper.id)
                        .values(is_manually_imported=True)
                    )
                    await ctx.db.commit()
                except Exception as exc:
                    log.warning("paper_import: flag write failed id=%s err=%s", arxiv_id, exc)

                paper_ids.append(str(paper.id))
                if was_new:
                    imported += 1
                    status = "imported"
                else:
                    duplicates += 1
                    status = "duplicate"
                items.append({
                    "arxiv_id": arxiv_id,
                    "status": status,
                    "paper_id": str(paper.id),
                    "title": paper.title,
                })

            except Exception as exc:
                log.exception("paper_import: pipeline failed id=%s", arxiv_id)
                failed_ids.append(arxiv_id)
                items.append({"arxiv_id": arxiv_id, "status": "failed", "error": str(exc)[:160]})

        await ctx.emit_progress(
            100,
            f"Imported {imported}; {duplicates} already present; {len(failed_ids)} failed",
        )

        summary_bits = []
        if imported:
            summary_bits.append(f"imported {imported}")
        if duplicates:
            summary_bits.append(f"{duplicates} already in feed")
        if failed_ids:
            summary_bits.append(f"{len(failed_ids)} failed")
        summary = "Paper import — " + (", ".join(summary_bits) if summary_bits else "no papers processed")

        return ToolResult(
            output={
                "imported": imported,
                "duplicates": duplicates,
                "failed": len(failed_ids),
                "paper_ids": paper_ids,
                "items": items,
            },
            summary=summary,
            citations=paper_ids,
        )


paper_import_tool = PaperImportTool()
