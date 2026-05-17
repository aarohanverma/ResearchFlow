"""LaTeX parse tool — extract structured content from LaTeX source.

Parses LaTeX documents (.tex files, arXiv source URLs, or raw LaTeX text) and
extracts structured content: title, authors, abstract, section outline, key
equations, and bibliography entries. Useful when the user has a LaTeX manuscript
or is reading an arXiv paper with available source.

Handles arXiv source fetching automatically when given an arXiv ID or abstract URL.
"""

from __future__ import annotations

import logging
import re

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_TIMEOUT = 15.0
_ARXIV_SOURCE_URL = "https://arxiv.org/src/{arxiv_id}"
_MAX_CONTENT_CHARS = 80_000   # truncate very large .tex files


class LaTeXParseInput(BaseModel):
    content: str = Field(default="", description="Raw LaTeX content to parse (paste .tex source here)")
    arxiv_id: str = Field(default="", description="arXiv paper ID (e.g. '1706.03762') to fetch source automatically")
    url: str = Field(default="", description="Direct URL to a .tex or .tar.gz file")
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class LaTeXParseOutput(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    sections: list[dict]     # [{title, level, preview}]
    equations: list[str]     # key equation strings
    bibliography: list[str]  # reference strings
    raw_excerpt: str          # first 3000 chars of processed content
    source: str


class LaTeXParseTool:
    """Parse LaTeX source to extract document structure, equations, and bibliography."""

    name = "latex_parse"
    summary = (
        "Parse LaTeX source code to extract structured document content: "
        "title, authors, abstract, section outline, key equations, and bibliography. "
        "Use when: user attaches or pastes LaTeX source (.tex), references a paper by arXiv ID "
        "and wants to read its LaTeX structure, or asks about equations/sections in a specific paper. "
        "Provide raw LaTeX via content= OR an arXiv ID via arxiv_id= for automatic source fetching."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = LaTeXParseInput
    output_schema = LaTeXParseOutput

    async def run(self, ctx: ToolContext, params: LaTeXParseInput) -> ToolResult:
        await ctx.emit_progress(15, "Fetching LaTeX source…")

        latex_source = ""
        source_label = "inline"

        # Priority: content > arxiv_id > url
        if params.content.strip():
            latex_source = params.content[:_MAX_CONTENT_CHARS]
            source_label = "inline"

        elif params.arxiv_id.strip():
            arxiv_id = _clean_arxiv_id(params.arxiv_id.strip())
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                    # arXiv source endpoint returns tar.gz for multi-file papers, raw .tex for single-file
                    resp = await client.get(
                        f"https://arxiv.org/e-print/{arxiv_id}",
                        headers={"User-Agent": "ResearchFlow/1.0"},
                    )
                    if resp.status_code == 200:
                        content_type = resp.headers.get("content-type", "")
                        if "gzip" in content_type or "octet-stream" in content_type:
                            # Try to extract the main .tex from tar.gz
                            latex_source = await _extract_tex_from_targz(resp.content)
                        else:
                            latex_source = resp.text[:_MAX_CONTENT_CHARS]
                    else:
                        # Fallback: try abstract page to at least get metadata
                        abs_resp = await client.get(f"https://arxiv.org/abs/{arxiv_id}")
                        if abs_resp.status_code == 200:
                            latex_source = _extract_abstract_page(abs_resp.text)
            except Exception as exc:
                log.warning("latex_parse: arXiv source fetch failed for %s: %s", arxiv_id, exc)
            source_label = f"arxiv:{arxiv_id}"

        elif params.url.strip():
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                    resp = await client.get(params.url, headers={"User-Agent": "ResearchFlow/1.0"})
                    resp.raise_for_status()
                    latex_source = resp.text[:_MAX_CONTENT_CHARS]
            except Exception as exc:
                log.warning("latex_parse: URL fetch failed: %s", exc)
            source_label = params.url

        if not latex_source:
            return ToolResult(
                output={
                    "title": "", "authors": [], "abstract": "", "sections": [],
                    "equations": [], "bibliography": [], "raw_excerpt": "", "source": source_label,
                },
                summary="No LaTeX source available to parse",
            )

        await ctx.emit_progress(50, "Parsing LaTeX structure…")

        title = _extract_title(latex_source)
        authors = _extract_authors(latex_source)
        abstract = _extract_abstract(latex_source)
        sections = _extract_sections(latex_source)
        equations = _extract_equations(latex_source)
        bibliography = _extract_bibliography(latex_source)
        raw_excerpt = _strip_commands(latex_source)[:3000]

        await ctx.emit_progress(100, f"Parsed: {len(sections)} sections, {len(equations)} equations")

        return ToolResult(
            output={
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "sections": sections[:20],
                "equations": equations[:15],
                "bibliography": bibliography[:20],
                "raw_excerpt": raw_excerpt,
                "source": source_label,
            },
            summary=(
                f"LaTeX parsed: '{title[:60] or 'untitled'}' — "
                f"{len(sections)} sections, {len(equations)} equations, {len(bibliography)} refs"
            ),
        )


# ── Extraction helpers ─────────────────────────────────────────────────────────

def _clean_arxiv_id(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    for prefix in ("https://arxiv.org/abs/", "https://arxiv.org/pdf/", "arxiv:", "arXiv:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    raw = raw.split("v")[0] if re.match(r"\d{4}\.\d{4,5}v\d+", raw) else raw
    return raw


def _extract_title(src: str) -> str:
    m = re.search(r"\\title\s*\{([^}]{1,300})\}", src, re.DOTALL)
    if m:
        return _strip_commands(m.group(1)).strip()
    return ""


def _extract_authors(src: str) -> list[str]:
    m = re.search(r"\\author\s*\{([^}]{1,1000})\}", src, re.DOTALL)
    if not m:
        return []
    raw = m.group(1)
    raw = re.sub(r"\\thanks\{[^}]*\}", "", raw)
    raw = re.sub(r"\\footnote\{[^}]*\}", "", raw)
    parts = re.split(r"\\and\b|,\s*(?=[A-Z])", raw)
    return [_strip_commands(p).strip() for p in parts if _strip_commands(p).strip()][:8]


def _extract_abstract(src: str) -> str:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.DOTALL)
    if m:
        return _strip_commands(m.group(1)).strip()[:2000]
    return ""


def _extract_sections(src: str) -> list[dict]:
    sections = []
    for m in re.finditer(r"\\(section|subsection|subsubsection)\*?\{([^}]{1,200})\}", src):
        level = {"section": 1, "subsection": 2, "subsubsection": 3}[m.group(1)]
        title = _strip_commands(m.group(2)).strip()
        # Grab a short preview of the section body
        start = m.end()
        preview = _strip_commands(src[start:start + 400]).strip()[:200]
        sections.append({"title": title, "level": level, "preview": preview})
    return sections


def _extract_equations(src: str) -> list[str]:
    equations: list[str] = []
    patterns = [
        r"\\begin\{equation\}(.*?)\\end\{equation\}",
        r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
        r"\$\$([^$]{5,200})\$\$",
        r"\\\[(.*?)\\\]",
    ]
    for pat in patterns:
        for m in re.finditer(pat, src, re.DOTALL):
            eq = m.group(1).strip()
            if eq and len(eq) > 4:
                equations.append(eq[:300])
    return equations[:20]


def _extract_bibliography(src: str) -> list[str]:
    refs: list[str] = []
    for m in re.finditer(r"\\bibitem(?:\[[^\]]*\])?\{[^}]*\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", src, re.DOTALL):
        ref = _strip_commands(m.group(1)).strip()[:200]
        if ref:
            refs.append(ref)
    return refs[:30]


def _strip_commands(text: str) -> str:
    """Remove common LaTeX commands, leaving readable text."""
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"\{|\}", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_tex_sync(content: bytes) -> str:
    """Pure-sync tarfile extraction — offloaded to a worker thread so it cannot
    block the event loop on multi-MB LaTeX archives."""
    try:
        import io, tarfile
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            tex_files = [m for m in tar.getmembers() if m.name.endswith(".tex")]
            if not tex_files:
                return ""
            # Prefer the largest .tex file (usually the main manuscript)
            main = max(tex_files, key=lambda m: m.size)
            f = tar.extractfile(main)
            if f:
                return f.read().decode("utf-8", errors="replace")[:_MAX_CONTENT_CHARS]
    except Exception as exc:
        log.warning("latex_parse: tar.gz extraction failed: %s", exc)
    return ""


async def _extract_tex_from_targz(content: bytes) -> str:
    """Extract the main .tex file from an arXiv tar.gz archive (non-blocking).

    tarfile + gzip are CPU-bound for non-trivial archives; running them inline
    on the event loop stalls every other coroutine for the duration. Offload
    to a thread so concurrent SSE streams, scheduler ticks, and other tools
    keep flowing.
    """
    import asyncio as _asyncio
    return await _asyncio.to_thread(_extract_tex_sync, content)


def _extract_abstract_page(html: str) -> str:
    """Extract abstract text from arXiv HTML abstract page as fallback."""
    m = re.search(r'class="abstract mathjax"[^>]*>(.*?)</blockquote>', html, re.DOTALL)
    if m:
        text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        return f"\\begin{{abstract}}\n{text}\n\\end{{abstract}}"
    return ""


latex_parse_tool = LaTeXParseTool()
