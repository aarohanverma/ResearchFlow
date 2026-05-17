"""HuggingFace Hub search tool — find models and datasets on HuggingFace Hub.

Searches the HuggingFace Hub for pre-trained models and datasets relevant to a
research topic. Uses the HuggingFace Hub API (free, no key required).

Particularly useful when the user wants to:
- Find pre-trained model weights to experiment with
- Discover datasets for a specific task or domain
- Check if a paper's model/dataset is publicly available on HF Hub
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_TIMEOUT = 15.0
_HF_MODELS_URL = "https://huggingface.co/api/models"
_HF_DATASETS_URL = "https://huggingface.co/api/datasets"
_MAX_RESULTS = 10


class HuggingFaceSearchInput(BaseModel):
    query: str = Field(description="Search query (e.g. 'bert question answering', 'llama instruction tuning')")
    search_type: str = Field(
        default="models",
        description="'models' to find pre-trained models, 'datasets' to find datasets",
        pattern="^(models|datasets)$",
    )
    task: str = Field(
        default="",
        description=(
            "Filter models by task pipeline tag (e.g. 'text-classification', "
            "'token-classification', 'question-answering', 'text-generation', "
            "'image-classification', 'translation', 'summarization')"
        ),
    )
    max_results: int = Field(default=8, ge=1, le=_MAX_RESULTS)
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class HuggingFaceSearchOutput(BaseModel):
    search_type: str
    results: list[dict]
    query: str


class HuggingFaceSearchTool:
    """Search HuggingFace Hub for pre-trained models or datasets."""

    name = "huggingface_search"
    summary = (
        "Search HuggingFace Hub for pre-trained models or datasets related to a research topic. "
        "Use when: user asks about model implementations/weights, wants to find datasets for a task, "
        "or is checking if a paper's artifacts are publicly available on HF Hub. "
        "search_type='models' for model weights; search_type='datasets' for datasets. "
        "Returns model/dataset IDs, download counts, task tags, and HF Hub links. "
        "DO NOT use for finding research papers — use arxiv_search or semantic_scholar for that."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = HuggingFaceSearchInput
    output_schema = HuggingFaceSearchOutput

    async def run(self, ctx: ToolContext, params: HuggingFaceSearchInput) -> ToolResult:
        await ctx.emit_progress(20, f"Searching HuggingFace Hub for '{params.query[:60]}'…")

        base_url = _HF_MODELS_URL if params.search_type == "models" else _HF_DATASETS_URL

        request_params: dict = {
            "search": params.query,
            "limit": params.max_results,
            "full": "true",
            "sort": "downloads",
            "direction": -1,
        }
        if params.search_type == "models" and params.task:
            request_params["pipeline_tag"] = params.task

        results: list[dict] = []

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    base_url,
                    params=request_params,
                    headers={"User-Agent": "ResearchFlow/1.0"},
                )
                resp.raise_for_status()
                items = resp.json()

                for item in items[:params.max_results]:
                    model_id = item.get("modelId") or item.get("id", "")
                    if params.search_type == "models":
                        results.append({
                            "id": model_id,
                            "url": f"https://huggingface.co/{model_id}",
                            "task": item.get("pipeline_tag") or "",
                            "downloads": item.get("downloads", 0),
                            "likes": item.get("likes", 0),
                            "tags": [t for t in (item.get("tags") or []) if not t.startswith("arxiv:")][:6],
                            "arxiv_ids": [
                                t.replace("arxiv:", "")
                                for t in (item.get("tags") or [])
                                if t.startswith("arxiv:")
                            ][:3],
                            "library": item.get("library_name") or "",
                            "last_modified": (item.get("lastModified") or "")[:10],
                        })
                    else:
                        results.append({
                            "id": model_id,
                            "url": f"https://huggingface.co/datasets/{model_id}",
                            "downloads": item.get("downloads", 0),
                            "likes": item.get("likes", 0),
                            "tags": (item.get("tags") or [])[:6],
                            "task_categories": [
                                t.replace("task_categories:", "")
                                for t in (item.get("tags") or [])
                                if t.startswith("task_categories:")
                            ][:3],
                            "last_modified": (item.get("lastModified") or "")[:10],
                        })

        except httpx.HTTPStatusError as exc:
            log.warning("huggingface_search: HTTP %s for query '%s'", exc.response.status_code, params.query)
            return ToolResult(
                output={"search_type": params.search_type, "results": [], "query": params.query},
                summary=f"HuggingFace search failed (HTTP {exc.response.status_code})",
            )
        except Exception as exc:
            log.warning("huggingface_search: error: %s", exc)
            return ToolResult(
                output={"search_type": params.search_type, "results": [], "query": params.query},
                summary=f"HuggingFace search failed: {exc}",
            )

        await ctx.emit_progress(100, f"Found {len(results)} {params.search_type}")

        top_ids = ", ".join(r["id"] for r in results[:3])
        label = "models" if params.search_type == "models" else "datasets"
        return ToolResult(
            output={
                "search_type": params.search_type,
                "results": results,
                "query": params.query,
            },
            summary=(
                f"HuggingFace Hub: {len(results)} {label} for '{params.query[:50]}'. "
                f"Top: {top_ids or 'none found'}"
            ),
        )


huggingface_search_tool = HuggingFaceSearchTool()
