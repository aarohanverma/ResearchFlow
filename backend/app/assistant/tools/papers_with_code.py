"""Papers with Code tool — ML implementations, SOTA leaderboards, and benchmarks.

Uses the free Papers with Code API (no authentication required) to find:
- Paper implementations and official/community repos
- State-of-the-art leaderboards for tasks/benchmarks
- ML methods and architectures by name
- Datasets used in ML research
- Benchmark evaluation results

Essential for ML/AI research where finding the official code, comparing SOTA,
or locating the right dataset is as important as reading the paper itself.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_BASE = "https://paperswithcode.com/api/v1"
_TIMEOUT = 15.0

_PWC_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at",
    "to", "for", "with", "from", "by", "is", "are", "was", "were", "be",
    "have", "has", "had", "do", "does", "did", "this", "that", "those",
    "these", "it", "its", "as", "so", "any", "some", "all", "no", "not",
    "what", "which", "who", "how", "why", "when", "where",
    "find", "show", "give", "want", "need", "please", "make", "build",
    "research", "paper", "papers", "project", "implementation", "code",
})


def _compress_pwc_query(raw: str, *, max_words: int = 6) -> str:
    """Tighten a verbose query into a short keyword phrase for Papers with Code.

    PwC's search treats the query as a substring against titles / method
    names; long sentences match nothing. We strip stop-words and keep
    informative tokens, capped at ``max_words``. Short queries pass
    through unchanged.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        return cleaned
    words = cleaned.split()
    if len(words) <= 5:
        return cleaned
    keep: list[str] = []
    for w in words:
        tok = w.strip(".,;:?!\"'()[]{}")
        if not tok or tok.lower() in _PWC_STOPWORDS:
            continue
        keep.append(tok)
        if len(keep) >= max_words:
            break
    return " ".join(keep) if len(keep) >= 2 else " ".join(words[:max_words])


class PapersWithCodeInput(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=120,
        description=(
            "SHORT canonical name of the ML topic / method / benchmark — "
            "this index is indexed by paper title + method name, NOT by "
            "natural-language questions. Good: 'retrieval augmented "
            "generation', 'mixture of experts', 'ImageNet classification', "
            "'GraphSAGE'. Bad: 'find me research code for RAG with chunk "
            "size and top-k retrieval' (returns 0). Keep to 2–6 words."
        ),
    )
    search_type: str = Field(
        default="papers",
        description=(
            "'papers' — find papers with their code repos and tasks; "
            "'methods' — find ML methods/architectures by name (e.g. 'ResNet', 'attention'); "
            "'datasets' — find benchmark datasets; "
            "'sota' — SOTA results for a specific task (e.g. 'ImageNet classification')"
        ),
    )
    limit: int = Field(default=8, ge=1, le=20)
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class PapersWithCodeOutput(BaseModel):
    results: list[dict]
    total: int
    search_type: str


class PapersWithCodeTool:
    """Search Papers with Code for ML implementations, SOTA tables, and benchmark results."""

    name = "papers_with_code"
    summary = (
        "Search Papers with Code for ML/AI papers with code implementations, SOTA leaderboards, "
        "benchmark datasets, and method architectures. "
        "Use for: 'find the code for X paper', 'what is SOTA on ImageNet?', "
        "'find datasets for object detection', 'how does ResNet compare to ViT on benchmarks?', "
        "'what are the best implementations of Y?'. "
        "Returns repos (GitHub links, star counts), evaluation results, and task metadata. "
        "No API key required. Best for ML/AI/computer vision/NLP research."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = PapersWithCodeInput
    output_schema = PapersWithCodeOutput

    async def run(self, ctx: ToolContext, params: PapersWithCodeInput) -> ToolResult:
        # Papers with Code indexes by title / method name. Long natural-
        # language queries reliably return zero. Compress to keyword form
        # before issuing the request so verbose planner output still hits.
        query = _compress_pwc_query(params.query)
        await ctx.emit_progress(
            15, f"Searching Papers with Code [{params.search_type}]: {query[:60]}",
        )

        results: list[dict] = []
        total = 0

        try:
            # follow_redirects=True: the PwC API may redirect to HuggingFace for
            # some endpoints; following lets us surface whatever the redirect serves.
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                if params.search_type == "papers":
                    results, total = await self._search_papers(client, query, params.limit)
                elif params.search_type == "methods":
                    results, total = await self._search_methods(client, query, params.limit)
                elif params.search_type == "datasets":
                    results, total = await self._search_datasets(client, query, params.limit)
                elif params.search_type == "sota":
                    results, total = await self._search_sota(client, query, params.limit)
                else:
                    results, total = await self._search_papers(client, query, params.limit)
        except Exception as exc:
            log.warning("papers_with_code search failed: %s", exc)
            return ToolResult(
                output={"results": [], "total": 0, "search_type": params.search_type},
                summary=f"Papers with Code unavailable: {exc}",
            )

        if not results:
            return ToolResult(
                output={"results": [], "total": 0, "search_type": params.search_type},
                summary=f"No Papers with Code results for: {params.query}",
            )

        await ctx.emit_progress(100, f"Found {len(results)} results on Papers with Code")
        top = results[0]
        return ToolResult(
            output={"results": results, "total": total, "search_type": params.search_type},
            summary=f"{len(results)} Papers with Code results (top: '{top.get('title') or top.get('name', '')}') ",
        )

    async def _search_papers(self, client: httpx.AsyncClient, query: str, limit: int) -> tuple[list[dict], int]:
        resp = await client.get(
            f"{_BASE}/papers/",
            params={"q": query, "items_per_page": limit},
            headers={"User-Agent": "ResearchFlow/1.0"},
        )
        resp.raise_for_status()
        # Guard against redirect-to-HTML responses (e.g. PwC → HuggingFace redirect chain)
        content_type = resp.headers.get("content-type", "")
        if "json" not in content_type:
            return [], 0
        data = resp.json()
        items = data.get("results") or []
        out = []
        for p in items[:limit]:
            # Fetch repos for this paper
            repos: list[dict] = []
            paper_id = p.get("id") or p.get("paper_id", "")
            if paper_id:
                try:
                    repo_resp = await client.get(
                        f"{_BASE}/papers/{paper_id}/repositories/",
                        headers={"User-Agent": "ResearchFlow/1.0"},
                    )
                    if repo_resp.status_code == 200:
                        rd = repo_resp.json()
                        for r in (rd.get("results") or [])[:3]:
                            repos.append({
                                "url": r.get("url", ""),
                                "stars": r.get("stars", 0),
                                "framework": r.get("framework", ""),
                                "is_official": r.get("is_official", False),
                            })
                except Exception:
                    pass
            out.append({
                "title": p.get("title", ""),
                "abstract": (p.get("abstract") or "")[:400],
                "arxiv_id": p.get("paper_arxiv_id") or p.get("arxiv_id"),
                "url": p.get("url_pdf") or p.get("url_abs"),
                "pwc_url": f"https://paperswithcode.com/paper/{paper_id}" if paper_id else "",
                "repositories": repos,
                "tasks": [t.get("name", "") for t in (p.get("tasks") or [])[:4]],
                "source": "papers_with_code",
            })
        return out, data.get("count", len(out))

    async def _search_methods(self, client: httpx.AsyncClient, query: str, limit: int) -> tuple[list[dict], int]:
        resp = await client.get(
            f"{_BASE}/methods/",
            params={"q": query, "items_per_page": limit},
            headers={"User-Agent": "ResearchFlow/1.0"},
        )
        resp.raise_for_status()
        if "json" not in resp.headers.get("content-type", ""):
            return [], 0
        data = resp.json()
        items = (data.get("results") or [])[:limit]
        out = [
            {
                "name": m.get("name", ""),
                "full_name": m.get("full_name", ""),
                "description": (m.get("description") or "")[:400],
                "paper": m.get("paper"),
                "url": f"https://paperswithcode.com/method/{m.get('id', '')}",
                "source": "papers_with_code_methods",
            }
            for m in items
        ]
        return out, data.get("count", len(out))

    async def _search_datasets(self, client: httpx.AsyncClient, query: str, limit: int) -> tuple[list[dict], int]:
        resp = await client.get(
            f"{_BASE}/datasets/",
            params={"q": query, "items_per_page": limit},
            headers={"User-Agent": "ResearchFlow/1.0"},
        )
        resp.raise_for_status()
        if "json" not in resp.headers.get("content-type", ""):
            return [], 0
        data = resp.json()
        items = (data.get("results") or [])[:limit]
        out = [
            {
                "name": d.get("name", ""),
                "full_name": d.get("full_name", ""),
                "description": (d.get("description") or "")[:400],
                "url": f"https://paperswithcode.com/dataset/{d.get('id', '')}",
                "modalities": d.get("modalities", []),
                "tasks": [t.get("name", "") for t in (d.get("tasks") or [])[:4]],
                "source": "papers_with_code_datasets",
            }
            for d in items
        ]
        return out, data.get("count", len(out))

    async def _search_sota(self, client: httpx.AsyncClient, query: str, limit: int) -> tuple[list[dict], int]:
        # Search tasks first, then get results for the best matching task
        resp = await client.get(
            f"{_BASE}/tasks/",
            params={"q": query, "items_per_page": 5},
            headers={"User-Agent": "ResearchFlow/1.0"},
        )
        resp.raise_for_status()
        if "json" not in resp.headers.get("content-type", ""):
            return [], 0
        tasks_data = resp.json()
        tasks = tasks_data.get("results") or []
        if not tasks:
            return [], 0

        best_task = tasks[0]
        task_id = best_task.get("id", "")
        out: list[dict] = [{
            "task": best_task.get("name", ""),
            "description": (best_task.get("description") or "")[:300],
            "url": f"https://paperswithcode.com/task/{task_id}",
            "benchmarks": [],
            "source": "papers_with_code_sota",
        }]

        # Fetch SOTA results for this task
        if task_id:
            try:
                results_resp = await client.get(
                    f"{_BASE}/tasks/{task_id}/results/",
                    params={"items_per_page": limit},
                    headers={"User-Agent": "ResearchFlow/1.0"},
                )
                if results_resp.status_code == 200:
                    res_data = results_resp.json()
                    for r in (res_data.get("results") or [])[:limit]:
                        out.append({
                            "model": r.get("method_name", ""),
                            "paper": r.get("paper_title", ""),
                            "metrics": r.get("metrics", {}),
                            "dataset": r.get("dataset", ""),
                            "source": "papers_with_code_results",
                        })
            except Exception:
                pass

        return out, len(out)


papers_with_code_tool = PapersWithCodeTool()
