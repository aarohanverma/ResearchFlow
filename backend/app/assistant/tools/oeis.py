"""OEIS (On-Line Encyclopedia of Integer Sequences) search tool.

https://oeis.org/wiki/OEIS_Wiki:API

Free REST API, no authentication required.
Rate limit: ~10 req/min (conservative, public service).

Essential for math.* namespaces: combinatorics, number theory, algebra,
discrete mathematics, analysis, graph theory, sequences arising in proofs.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult

log = logging.getLogger(__name__)

_SEARCH_URL = "https://oeis.org/search"
_TIMEOUT = 15.0


class OeisInput(BaseModel):
    query: str = Field(
        min_length=1, max_length=300,
        description=(
            "Search query — integer sequence (comma-separated: '1,1,2,3,5,8'), "
            "OEIS ID ('A000045'), or descriptive keywords ('Fibonacci numbers', "
            "'number of permutations avoiding 132', 'Catalan numbers'). "
            "For sequences, more terms = better disambiguation."
        ),
    )
    limit: int = Field(default=6, ge=1, le=10)
    namespace_key: str = Field(default="")
    namespace_keys: list[str] = Field(default_factory=list)


class OeisOutput(BaseModel):
    sequences: list[dict]
    total_found: int


class OeisTool:
    """Search OEIS for integer sequences and their mathematical properties."""

    name = "oeis"
    summary = (
        "Search the OEIS (On-Line Encyclopedia of Integer Sequences) for integer sequences "
        "and their mathematical properties. "
        "Use for: identifying an integer sequence from terms, finding combinatorial formulas, "
        "number theory sequences, graph-theoretic counts, polynomial sequences, "
        "recurrences, generating functions, special functions, and mathematical constants. "
        "Input: comma-separated sequence terms ('1,4,9,16,25'), OEIS A-number ('A000290'), "
        "or keywords ('triangular numbers', 'Bell numbers'). "
        "Returns: A-number, name, description, sample terms, formulas, references, Mathematica/Python code. "
        "Free API, no key required. Preferred for math.* namespaces."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = True
    streamable = False
    input_schema = OeisInput
    output_schema = OeisOutput

    async def run(self, ctx: ToolContext, params: OeisInput) -> ToolResult:
        await ctx.emit_progress(20, f"Searching OEIS: {params.query[:60]}")

        q = params.query.strip()

        if q.upper().startswith("A") and q[1:].isdigit():
            search_q = f"id:{q.upper()}"
        else:
            search_q = q

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    _SEARCH_URL,
                    params={
                        "q": search_q,
                        "fmt": "json",
                        "start": 0,
                        "n": params.limit,
                    },
                    headers={"User-Agent": "ResearchFlow/1.0"},
                )
                if resp.status_code == 429:
                    return ToolResult(
                        output={"sequences": [], "total_found": 0},
                        summary="OEIS rate limited. Try again shortly.",
                    )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            log.warning("oeis HTTP %s for query %r", exc.response.status_code, q)
            return ToolResult(
                output={"sequences": [], "total_found": 0},
                summary=f"OEIS search failed (HTTP {exc.response.status_code})",
            )
        except Exception as exc:
            log.warning("oeis failed: %s", exc)
            return ToolResult(
                output={"sequences": [], "total_found": 0},
                summary=f"OEIS unavailable: {exc}",
            )

        total = data.get("count", 0)
        raw_results = data.get("results") or []

        sequences: list[dict] = []
        for item in raw_results[:params.limit]:
            a_number = f"A{item.get('number', 0):06d}"

            sample_values = (item.get("data") or "").split(",")[:15]
            sample = ", ".join(v.strip() for v in sample_values if v.strip())

            name = item.get("name", "")
            offset = item.get("offset", "")

            formulas = (item.get("formula") or [])[:3]
            formulas_text = " | ".join(str(f) for f in formulas)[:400]

            references = (item.get("reference") or [])[:2]
            refs_text = " | ".join(str(r) for r in references)[:300]

            links = (item.get("link") or [])[:3]

            code_sections = item.get("program") or []
            python_code = ""
            mathematica_code = ""
            for code_item in code_sections:
                if isinstance(code_item, str):
                    if code_item.startswith("(Python)"):
                        python_code = code_item[8:].strip()[:400]
                    elif code_item.startswith("(Mathematica)"):
                        mathematica_code = code_item[13:].strip()[:300]

            sequences.append({
                "id": a_number,
                "name": name,
                "sample_values": sample,
                "offset": offset,
                "formulas": formulas_text,
                "references": refs_text,
                "links": links,
                "python_code": python_code,
                "mathematica_code": mathematica_code,
                "url": f"https://oeis.org/{a_number}",
                "source": "oeis",
            })

        await ctx.emit_progress(100, f"OEIS: {len(sequences)} sequences found (total: {total:,})")

        if not sequences:
            return ToolResult(
                output={"sequences": [], "total_found": total},
                summary=f"No OEIS sequences found for: {q}",
            )

        top = sequences[0]
        return ToolResult(
            output={"sequences": sequences, "total_found": total},
            summary=(
                f"{len(sequences)} OEIS sequences (total: {total:,}) — "
                f"top: {top['id']} '{top['name'][:60]}' ({top['sample_values'][:40]}…)"
            ),
        )


oeis_tool = OeisTool()
