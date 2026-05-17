"""Tool registry — registers built-in tools at import time.

Each tool wraps an existing platform capability behind the AssistantTool
contract. Adding a new tool = creating a module + calling register_tool.
"""

from app.assistant.tools import (
    arxiv_import,
    arxiv_search,
    bookmarks_query,
    citation_finder,
    clinicaltrials,
    compare_papers,
    concept_explain,
    crossref,
    deep_search,
    draft_section,
    fred,
    frontier_scan,
    genie_combine,
    genie_deep_dive,
    genie_read,
    genie_synthesize,
    github_search,
    graph_build,    # kept registered so /assistant/tools surfaces it for
                    # operator visibility; the planner is forbidden from
                    # picking it (see planner_llm.py forbidden set).
    graph_query,
    huggingface_search,
    inspire_hep,
    latex_parse,
    literature_survey,
    media_generate,
    memory,
    nasa_ads,
    nvd_cve,
    oeis,
    paper_import,
    paper_qa,
    papers_with_code,
    parse_context,
    pubmed,
    research_trends,
    study_paper,
    unpaywall,
    web_search,
    wikipedia,
    wolfram_alpha,
)
from app.assistant.tools.registry import describe_for_planner, get_tool, list_tools, register_tool

# Built-in tool registrations (idempotent — safe to import multiple times).
# NOTE: semantic_scholar is intentionally NOT registered — it is frequently
# rate-limited and produces unreliable results. OpenAlex (via research_trends)
# covers citation/trend queries without rate-limit issues.
# author_network is also disabled — OpenAlex /authors?search= returns 400
# on compound research-topic queries; the planner kept picking it for
# author-discovery tasks where it always fails.

register_tool(deep_search.DeepSearchTool())
register_tool(arxiv_search.ArxivSearchTool())
register_tool(arxiv_import.ArxivImportTool())
register_tool(paper_import.paper_import_tool)
register_tool(genie_synthesize.GenieSynthesizeTool())
register_tool(frontier_scan.FrontierScanTool())
register_tool(graph_query.GraphQueryTool())
register_tool(graph_build.GraphBuildTool())  # registered but planner-forbidden
register_tool(web_search.WebSearchTool())
register_tool(concept_explain.ConceptExplainTool())
register_tool(compare_papers.ComparePapersTool())
register_tool(bookmarks_query.BookmarksQueryTool())
register_tool(memory.memory_write_tool)
register_tool(memory.memory_recall_tool)
register_tool(memory.memory_delete_tool)
register_tool(parse_context.parse_context_tool)
register_tool(wolfram_alpha.wolfram_alpha_tool)
register_tool(draft_section.DraftSectionTool())
register_tool(genie_read.genie_read_tool)
register_tool(genie_deep_dive.genie_deep_dive_tool)
register_tool(genie_combine.genie_combine_tool)
register_tool(literature_survey.literature_survey_tool)
register_tool(wikipedia.wikipedia_tool)
register_tool(crossref.crossref_tool)
register_tool(research_trends.research_trends_tool)
register_tool(paper_qa.paper_qa_tool)
register_tool(study_paper.study_paper_tool)
register_tool(citation_finder.citation_finder_tool)
register_tool(latex_parse.latex_parse_tool)
register_tool(media_generate.media_generate_tool)

# Previously unregistered tools (now active)
register_tool(pubmed.pubmed_tool)
register_tool(unpaywall.unpaywall_tool)
register_tool(github_search.github_search_tool)
register_tool(huggingface_search.huggingface_search_tool)
register_tool(papers_with_code.papers_with_code_tool)

# Namespace-specific tool packs
register_tool(nvd_cve.nvd_cve_tool)
register_tool(clinicaltrials.clinicaltrials_tool)
register_tool(fred.fred_tool)
register_tool(nasa_ads.nasa_ads_tool)
register_tool(inspire_hep.inspire_hep_tool)
register_tool(oeis.oeis_tool)

__all__ = ["describe_for_planner", "get_tool", "list_tools", "register_tool"]
