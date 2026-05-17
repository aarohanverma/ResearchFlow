"""Namespace-aware tool pack definitions for the Research Assistant.

GLOBAL_TOOLS are always visible to the planner regardless of namespace.
NAMESPACE_PACK_TOOLS are additional tools activated for specific namespace prefixes.

The planner receives GLOBAL_TOOLS ∪ relevant NAMESPACE_PACK_TOOLS for the
active session namespace so it can make domain-appropriate choices.
"""

from __future__ import annotations

# ── Global tools: always available ────────────────────────────────────────────
GLOBAL_TOOLS: frozenset[str] = frozenset({
    # Core retrieval
    "deep_search",
    "arxiv_import",
    "arxiv_search",
    "paper_import",
    "frontier_scan",
    "literature_survey",
    # Paper analysis
    "paper_qa",
    "study_paper",
    "compare_papers",
    "concept_explain",
    "draft_section",
    "citation_finder",
    "latex_parse",
    # Web + encyclopedic
    "web_search",
    "wikipedia",
    # Cross-domain search
    "crossref",
    "pubmed",
    "unpaywall",
    "openalex_trends",
    "research_trends",
    "author_network",
    # Internal ResearchFlow features (NEVER remove)
    "genie_read",
    "genie_synthesize",
    "genie_deep_dive",
    "genie_combine",
    "graph_query",
    "graph_build",
    "bookmarks_query",
    "memory_write",
    "memory_recall",
    "memory_delete",
    "parse_context",
    "media_generate",
    # Computation
    "wolfram_alpha",
})

# ── Namespace pack definitions ─────────────────────────────────────────────────
# Maps namespace PREFIX → set of extra tool names available for that namespace.
# Prefixes are matched with str.startswith() so "cs" covers cs.AI, cs.LG, cs.CR, etc.

_NAMESPACE_PACKS: dict[str, frozenset[str]] = {
    # CS / AI / ML — software, systems, security, information retrieval
    "cs": frozenset({
        "github_search",
        "huggingface_search",
        "papers_with_code",
        "nvd_cve",
    }),
    "eess": frozenset({
        "github_search",
        "huggingface_search",
        "papers_with_code",
    }),
    # High-energy physics / nuclear / gravity / quantum
    "hep": frozenset({
        "inspire_hep",
    }),
    "nucl": frozenset({
        "inspire_hep",
    }),
    "gr-qc": frozenset({
        "inspire_hep",
        "nasa_ads",
    }),
    "quant-ph": frozenset({
        "inspire_hep",
    }),
    "math-ph": frozenset({
        "inspire_hep",
        "oeis",
    }),
    # Astrophysics / planetary / space science
    "astro-ph": frozenset({
        "nasa_ads",
    }),
    "physics": frozenset({
        "nasa_ads",
        "inspire_hep",
    }),
    # Mathematics — sequences, combinatorics, number theory
    "math": frozenset({
        "oeis",
    }),
    # Quantitative biology / medicine / health
    "q-bio": frozenset({
        "clinicaltrials",
    }),
    # Economics / quantitative finance
    "econ": frozenset({
        "fred",
    }),
    "q-fin": frozenset({
        "fred",
    }),
    # Statistics — broad coverage, no pack additions beyond global
    "stat": frozenset(),
}


def get_pack_tools(namespace_key: str) -> frozenset[str]:
    """Return the set of extra tools for a given namespace key.

    Matches the namespace_key against pack prefixes (longest match wins).
    Returns an empty frozenset if no pack applies.
    """
    best_prefix = ""
    best_pack: frozenset[str] = frozenset()
    for prefix, pack in _NAMESPACE_PACKS.items():
        if namespace_key.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_pack = pack
    return best_pack


def get_visible_tools(namespace_key: str | None) -> frozenset[str]:
    """Return the complete set of tool names visible for a given namespace."""
    ns = namespace_key or ""
    return GLOBAL_TOOLS | get_pack_tools(ns)


_PACK_DESCRIPTIONS: dict[str, str] = {
    "cs": (
        "CS/AI/ML pack active: github_search (code repos), huggingface_search (models/datasets), "
        "papers_with_code (benchmarks + SoTA tables), nvd_cve (security vulnerabilities). "
        "Use these for implementation lookup, model weights, benchmark comparisons, CVE details."
    ),
    "eess": (
        "EESS pack active: github_search, huggingface_search, papers_with_code. "
        "Use for signal processing implementations, pre-trained models, benchmark results."
    ),
    "hep": (
        "HEP pack active: inspire_hep (INSPIRE HEP — 1.4M+ particle physics papers, "
        "citation counts, experiment records). Preferred over arXiv search for hep-* queries."
    ),
    "nucl": (
        "Nuclear physics pack active: inspire_hep. Preferred for nucl-th, nucl-ex queries."
    ),
    "gr-qc": (
        "GR/QC pack active: inspire_hep + nasa_ads. Use inspire_hep for theory, nasa_ads for observational."
    ),
    "quant-ph": (
        "Quantum physics pack active: inspire_hep. Covers quantum information, quantum optics, "
        "quantum computing theory."
    ),
    "math-ph": (
        "Mathematical physics pack active: inspire_hep + oeis. Use oeis for integer sequences arising in proofs."
    ),
    "astro-ph": (
        "Astrophysics pack active: nasa_ads (15M+ astronomy records, bibcodes, citation counts). "
        "Preferred over arXiv search for astro-ph.* queries. Use for telescopes, missions, observational data."
    ),
    "physics": (
        "Physics pack active: nasa_ads + inspire_hep. Use nasa_ads for astro/instrumentation, "
        "inspire_hep for HEP/nuclear topics."
    ),
    "math": (
        "Mathematics pack active: oeis (On-Line Encyclopedia of Integer Sequences). "
        "Use when query involves integer sequences, combinatorial counts, special functions, recurrences."
    ),
    "q-bio": (
        "Quantitative biology/medicine pack active: clinicaltrials (ClinicalTrials.gov — registered "
        "clinical studies, NCT IDs, phase/status filters). Use for clinical research, drug trials, "
        "interventional studies, public health studies."
    ),
    "econ": (
        "Economics pack active: fred (FRED — St. Louis Fed economic time series: GDP, CPI, "
        "unemployment, interest rates, M2, trade, housing). Requires FRED_API_KEY."
    ),
    "q-fin": (
        "Quantitative finance pack active: fred (FRED macroeconomic data). Useful for backtesting "
        "macro signals and understanding the economic environment for financial models."
    ),
}


def get_pack_description(namespace_key: str | None) -> str:
    """Return a human-readable description of the active namespace pack for the planner."""
    ns = namespace_key or ""
    best_prefix = ""
    for prefix in _PACK_DESCRIPTIONS:
        if ns.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
    if best_prefix:
        return _PACK_DESCRIPTIONS[best_prefix]
    return ""
