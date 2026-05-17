"""Synthesizer — composes the final assistant message from step results.

Pure composition: never invents facts. Reads the orchestrator's step outputs,
picks the relevant slices (papers, arXiv results, graph result, genie session),
and asks the quality LLM to write a grounded answer with inline citations.

Also emits a structured ``blocks`` list that the M2 frontend renders as
heterogeneous UI elements (paper grids, step progress, suggestion chips,
artifact links) instead of a wall of markdown.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)


_EXPERTISE_HINTS = {
    "newcomer": (
        "The user is new to this area. Define every technical term the first "
        "time it appears. Scaffold concepts from familiar ground up. Suggest "
        "one foundational paper to read first if relevant. Avoid implying "
        "they should already know things."
    ),
    "practitioner": (
        "The user is a working researcher / engineer. Balance clarity with "
        "implementation and methodology detail. Skip definitions of standard "
        "terminology. Be specific about tradeoffs and gotchas."
    ),
    "expert": (
        "The user is a domain expert. Be terse and technical; assume "
        "background. Highlight subtle distinctions, contested interpretations, "
        "and frontier debates. Skip preamble entirely."
    ),
}
_ORIENTATION_HINTS = {
    "research": "Emphasize novelty, evidence, gaps, methodological tradeoffs, and openness of the question.",
    "production": "Emphasize implementation maturity, validation, constraints, deployment risks, and reproducibility.",
    "both": "Balance research implications (novelty, gaps) with practical execution (implementation, validation).",
}

# End-to-end research lifecycle scaffolding. Even when the user doesn't ask
# for it, the synthesizer should tee up the *next* stage so beginners get
# pulled forward and seasoned researchers can shortcut to where they need
# to be. The "Next moves" section anchors this — see prompt rule 6.
_LIFECYCLE_GUIDANCE = (
    "RESEARCH LIFECYCLE AWARENESS: The user may be at any stage — "
    "discovery, learning, understanding, exploration, synthesis, ideation, "
    "validation, experimentation, implementation, or publication. Infer the "
    "stage from their question and the conversation history. Always close "
    "with one concrete, stage-appropriate next step (e.g. 'read foundational "
    "paper X', 'compare methods Y vs Z', 'run a small ablation that controls "
    "for W', 'draft a related-work section grounded in [1], [3]'). For "
    "complete beginners, scaffold the path forward; for experts, surface "
    "leverage points and unresolved debates."
)

# Output format guidance — injected into the prompt to break the rigid
# Takeaway/Evidence/Gaps/Next-moves template and let the synthesizer choose
# the best representation for the query.
_FORMAT_GUIDANCE = (
    "OUTPUT FORMAT — ADAPTIVE:\n"
    "Choose the format that best serves this specific query and user stage. "
    "Do NOT force a fixed section template. Examples:\n"
    "• Conceptual question → 1-2 sentence core idea, then layered explanation with cited evidence\n"
    "• Discovery / mapping → short synthesis paragraph + ranked paper list with why-surfaced notes\n"
    "• Comparison request → a concise comparison table or side-by-side prose, then verdict\n"
    "• Research roadmap → numbered learning path from foundations to frontier\n"
    "• Narrow technical question → direct answer first, cited evidence, one follow-up\n"
    "• Hypothesis / ideation → premise → evidence → gap → proposed direction\n"
    "• Teaching / beginner → accessible explanation → example → key paper to read next\n"
    "\n"
    "FORMATTING RULES (the renderer supports full GitHub-flavored markdown):\n"
    "• Headings — use `##` (h2) for top-level sections of a multi-part answer; "
    "use `###` (h3) for sub-sections. Never put a heading at the very top — start "
    "with a single TL;DR paragraph instead.\n"
    "• Paragraphs — separate with one blank line. Never use line-break HTML like `<br>`.\n"
    "• Lists — use `- ` for bullets and `1. ` for ordered steps. ONE blank line "
    "before the first list item, no blank lines between items, ONE blank line after.\n"
    "• Emphasis — **bold** for key terms and definitions; *italic* for paper names "
    "and proper-noun emphasis; `code` (inline backticks) for symbols, identifiers, "
    "numeric thresholds, and file paths.\n"
    "• Math — inline math with `$...$`, display math with `$$...$$`. Use real LaTeX, "
    "not ASCII fallbacks like ``E = m*c^2``.\n"
    "• Code blocks — triple-backticks with a language tag (` ```python ` etc.) "
    "for runnable snippets, pseudocode, or shell commands. Never paste tables of "
    "numbers inside code blocks — use a markdown table instead.\n"
    "• Tables — pipe-separated; use them whenever you are presenting more than three "
    "row-aligned facts (comparison, benchmark numbers, parameter sweeps).\n"
    "• Quotes / callouts — use `> ` blockquotes for direct quotes from a paper.\n"
    "• Mermaid diagrams — when a concept benefits from a flow / state diagram, emit "
    "a ` ```mermaid ` fenced block. Keep node labels short. Validate the syntax in "
    "your head before emitting (no trailing semicolons, no smart quotes in labels).\n"
    "• Citations — every factual sentence must carry at least one citation in the "
    "form `[1]`, `[2]`, or `[A1]` for arXiv candidates.\n"
    "\n"
    "STRUCTURE — ALWAYS:\n"
    "1. Open with a 1–3 sentence answer-first TL;DR (no heading).\n"
    "2. Then layered detail in sections that match the query type from the examples above.\n"
    "3. End with a short *Next steps* or *Open questions* line when the topic warrants it.\n"
    "\n"
    "NEVER produce a wall of dense paragraphs without visual structure. NEVER chain "
    "more than two `#`-headings without intervening prose. Prefer clarity over volume."
)


async def synthesize_answer(
    *,
    query: str,
    papers: list[dict],
    arxiv_results: list[dict],
    imported_count: int,
    graph_result: dict | None,
    genie_session_id: str | None,
    orientation: str,
    expertise: str,
    complexity: str = "medium",
    actions: list[str],
    extra_results: dict | None = None,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """Return a grounded research-workspace answer or a deterministic fallback."""
    context = _build_paper_context(papers, arxiv_results)
    extra_context = _build_extra_context(extra_results or {})
    fallback = _fallback_answer(query, papers, imported_count, graph_result, genie_session_id)

    grounded = bool(papers)
    # Pure-reasoning mode: no tool evidence — answer from general knowledge.
    # Still call the LLM (not the deterministic fallback) so the user gets a
    # real answer for definitional, conversational, or simple knowledge queries.
    pure_reasoning = not context and not extra_context

    try:
        from app.adapters.llm import get_llm_adapter

        llm = get_llm_adapter()
        prompt = _build_prompt(
            query=query,
            context=context,
            extra_context=extra_context,
            actions=actions,
            imported_count=imported_count,
            graph_result=graph_result,
            genie_session_id=genie_session_id,
            orientation=orientation,
            expertise=expertise,
            grounded=grounded,
            pure_reasoning=pure_reasoning,
        )
        # Pure-reasoning: quality model is faster and perfectly adequate.
        # Grounded evidence: use reasoning model for deep synthesis.
        messages = [{"role": "user", "content": prompt}]
        if pure_reasoning:
            if on_delta:
                chunks: list[str] = []
                async for chunk in llm.stream(messages, llm.quality_model):
                    chunks.append(chunk)
                    await on_delta(chunk)
                return "".join(chunks).strip() or fallback
            res = await llm.complete(messages, llm.quality_model, max_tokens=None, temperature=0.3)
        else:
            effort_map = {"simple": "low", "medium": "medium", "complex": "high"}
            r_effort = effort_map.get(complexity, "medium")
            if on_delta:
                chunks = []
                async for chunk in llm.stream(
                    messages, llm.reasoning_model, reasoning_effort=r_effort
                ):
                    chunks.append(chunk)
                    await on_delta(chunk)
                return "".join(chunks).strip() or fallback
            res = await llm.complete(
                messages, llm.reasoning_model, max_tokens=None,
                temperature=None, reasoning_effort=r_effort,
            )
        return res.text.strip() or fallback
    except Exception as exc:
        log.warning("assistant synthesis fallback (reasoning model): %s — retrying with quality model", exc)
        try:
            from app.adapters.llm import get_llm_adapter
            llm2 = get_llm_adapter()
            prompt2 = _build_prompt(
                query=query, context=context, extra_context=extra_context,
                actions=actions, imported_count=imported_count,
                graph_result=graph_result, genie_session_id=genie_session_id,
                orientation=orientation, expertise=expertise, grounded=grounded,
                pure_reasoning=pure_reasoning,
            )
            msgs2 = [{"role": "user", "content": prompt2}]
            if on_delta:
                chunks2: list[str] = []
                async for chunk in llm2.stream(msgs2, llm2.quality_model):
                    chunks2.append(chunk)
                    await on_delta(chunk)
                return "".join(chunks2).strip() or fallback
            res2 = await llm2.complete(msgs2, llm2.quality_model, max_tokens=None, temperature=0.2)
            return res2.text.strip() or fallback
        except Exception as exc2:
            log.warning("assistant synthesis fallback: %s", exc2)
            return fallback


def _build_extra_context(results: dict) -> str:
    """Build a combined context string from all non-paper tool outputs.

    Section labels use angle-bracket XML style so the model treats them as
    structural delimiters, not text to be quoted in the answer.
    """
    parts: list[str] = []

    # Wolfram Alpha — precise computation results
    wa = results.get("wolfram_alpha")
    if wa and wa.output.get("answer"):
        pods = wa.output.get("pods") or []
        pod_text = "\n".join(f"  {p['title']}: {p['text']}" for p in pods[:4]) if pods else ""
        parts.append(
            f"<wolfram_computation>\n"
            f"{wa.output['answer']}"
            + (f"\nPods:\n{pod_text}" if pod_text else "")
            + "\n</wolfram_computation>"
        )

    # Parse context — user-attached files, notes, URLs
    pc = results.get("parse_context")
    if pc and pc.output.get("items"):
        items = pc.output["items"]
        item_texts = []
        for item in items[:5]:
            label = item.get("label") or item.get("kind") or "Attachment"
            text = (item.get("text") or "")[:2000]
            item_texts.append(f"  <item label=\"{label}\">\n  {text}\n  </item>")
        parts.append("<active_context>\n" + "\n\n".join(item_texts) + "\n</active_context>")

    # Concept explain — teaching/explanation output
    ce = results.get("concept_explain")
    if ce and ce.output.get("explanation"):
        parts.append(f"<concept_background>\n{ce.output['explanation'][:3000]}\n</concept_background>")

    # Memory recall — prior session facts (typed: finding/preference/concept/hypothesis/paper_note/context)
    mr = results.get("memory_recall")
    if mr:
        med = mr.output.get("medium") or {}
        lng = mr.output.get("long") or {}
        # Support both new typed format {value, type, ts} and legacy string format
        def _fmt_entry(k: str, v: object) -> str:
            if isinstance(v, dict):
                vtype = v.get("type", "context")
                vval = v.get("value", "")
                return f"  [{vtype}] {k}: {vval}"
            return f"  {k}: {v}"
        if med or lng:
            lines: list[str] = []
            if med:
                lines.append("Session memory:")
                lines.extend(_fmt_entry(k, v) for k, v in list(med.items())[:12])
            if lng:
                lines.append("Namespace memory (persists across sessions):")
                lines.extend(_fmt_entry(k, v) for k, v in list(lng.items())[:8])
            parts.append(f"<session_memory>\n" + "\n".join(lines) + "\n</session_memory>")

    # Frontier scan — recent papers
    fs = results.get("frontier_scan")
    if fs and fs.output.get("papers"):
        fps = fs.output["papers"][:5]
        paper_lines = "\n".join(
            f"  [{i+1}] {p.get('title')} ({p.get('year', '')}) — {p.get('tldr') or p.get('abstract', '')[:200]}"
            for i, p in enumerate(fps)
        )
        parts.append(f"<frontier_papers>\n{paper_lines}\n</frontier_papers>")

    # Bookmarks answer — user's saved papers (arXiv-sourced)
    bq = results.get("bookmarks_query")
    if bq and bq.output.get("answer"):
        parts.append(f"<bookmarks_answer>\n{bq.output['answer'][:1500]}\n</bookmarks_answer>")

    # Web search — external context (use for docs, news, repos; prefer arXiv for academic claims)
    ws = results.get("web_search")
    if ws and ws.output.get("results"):
        web_items = ws.output["results"][:5]
        web_lines = "\n".join(
            f"  {r.get('title')} — {r.get('snippet', '')[:300]} ({r.get('url', '')})"
            for r in web_items
        )
        parts.append(f"<web_search_results>\n{web_lines}\n</web_search_results>")

    # Genie deep dive — comprehensive Opus-generated technical article for a specific idea
    gdd = results.get("genie_deep_dive")
    if gdd:
        dd_status = gdd.output.get("deep_dive_status", "none")
        dd_title = gdd.output.get("title", "")
        dd_href = gdd.output.get("href", "")
        if dd_status == "done" and gdd.output.get("deep_dive_excerpt"):
            hypothesis = gdd.output.get("hypothesis", "")
            excerpt = gdd.output.get("deep_dive_excerpt", "")
            parts.append(
                f"<genie_deep_dive idea=\"{dd_title}\">\n"
                f"IMPORTANT: This is an AI-generated hypothesis and technical exploration, "
                f"NOT an experimentally validated finding. Treat as a creative research direction.\n\n"
                f"Hypothesis: {hypothesis[:400]}\n\n"
                f"Full technical deep dive:\n{excerpt}\n"
                f"</genie_deep_dive>"
            )
        elif dd_status == "generating":
            parts.append(
                f"<genie_deep_dive_status>\n"
                f"Deep dive for '{dd_title}' is generating now (~60-90 seconds). "
                f"The user can view it at: {dd_href}\n"
                f"</genie_deep_dive_status>"
            )

    # Genie ideas (research hypotheses, clearly labelled as unvalidated)
    gi = results.get("genie_read")
    if gi and gi.output.get("ideas"):
        ideas = gi.output["ideas"][:3]
        idea_lines = "\n".join(
            f"  Idea {i+1}: {idea.get('title')} — {(idea.get('hypothesis') or '')[:300]}"
            f"\n    Note: Novelty={idea.get('novelty_score',0):.0%} Feasibility={idea.get('feasibility_score',0):.0%}"
            for i, idea in enumerate(ideas)
        )
        parts.append(
            f"<genie_research_ideas>\n"
            f"IMPORTANT: These are AI-generated hypotheses that have NOT been experimentally validated.\n"
            f"Treat them as creative directions or starting points, not established facts.\n"
            f"{idea_lines}\n"
            f"</genie_research_ideas>"
        )

    # PubMed — biomedical / life sciences literature
    pm = results.get("pubmed")
    if pm and pm.output.get("papers"):
        pm_papers = pm.output["papers"][:5]
        pm_db = pm.output.get("database", "PubMed")
        pm_lines = "\n".join(
            f"  [{i+1}] {p.get('title', '')} ({p.get('year', '')}) — "
            f"PMID: {p.get('pmid', '')} — "
            f"{(p.get('abstract') or '')[:300]}"
            for i, p in enumerate(pm_papers)
        )
        parts.append(f"<pubmed_papers database=\"{pm_db}\">\n{pm_lines}\n</pubmed_papers>")

    # Unpaywall — open-access PDF availability
    uw = results.get("unpaywall")
    if uw and uw.output.get("is_oa") and uw.output.get("best_oa_url"):
        parts.append(
            f"<unpaywall doi=\"{uw.output.get('doi', '')}\">\n"
            f"  Open-access: {uw.output.get('best_oa_version', '')} — {uw.output.get('best_oa_url', '')}\n"
            f"  License: {uw.output.get('license', 'unknown')}\n"
            f"</unpaywall>"
        )

    # NVD CVE — security vulnerability data
    nvd = results.get("nvd_cve")
    if nvd and nvd.output.get("vulnerabilities"):
        vulns = nvd.output["vulnerabilities"][:5]
        vuln_lines = "\n".join(
            f"  [{i+1}] {v.get('id', '')} — CVSS: {v.get('cvss_score', 'N/A')} ({v.get('severity', '')}) — "
            f"{(v.get('description') or '')[:250]}"
            for i, v in enumerate(vulns)
        )
        total = nvd.output.get("total_results", len(vulns))
        parts.append(f"<nvd_cve total=\"{total}\">\n{vuln_lines}\n</nvd_cve>")

    # ClinicalTrials.gov — registered clinical studies
    ct = results.get("clinicaltrials")
    if ct and ct.output.get("studies"):
        studies = ct.output["studies"][:5]
        study_lines = "\n".join(
            f"  [{i+1}] {s.get('title', '')} ({s.get('nct_id', '')}) — "
            f"Phase: {s.get('phase', '?')} | Status: {s.get('status', '?')} — "
            f"Conditions: {', '.join((s.get('conditions') or [])[:3])}"
            for i, s in enumerate(studies)
        )
        total = ct.output.get("total_found", len(studies))
        parts.append(f"<clinical_trials total=\"{total}\">\n{study_lines}\n</clinical_trials>")

    # FRED — macroeconomic / financial time series
    fred = results.get("fred")
    if fred and fred.output.get("series"):
        series_list = fred.output["series"][:4]
        series_lines = []
        for s in series_list:
            obs = s.get("observations") or []
            recent_obs = obs[-3:] if obs else []
            obs_str = ", ".join(f"{o.get('date','')}: {o.get('value','')}" for o in recent_obs)
            series_lines.append(
                f"  {s.get('id','')}: {s.get('title','')} ({s.get('units','')}) — "
                f"Recent: {obs_str or 'no data'}"
            )
        parts.append(f"<fred_data>\n" + "\n".join(series_lines) + "\n</fred_data>")

    # NASA ADS — astronomy / astrophysics papers
    ads = results.get("nasa_ads")
    if ads and ads.output.get("papers"):
        ads_papers = ads.output["papers"][:5]
        ads_lines = "\n".join(
            f"  [{i+1}] {p.get('title', '')} ({p.get('year', '')}) — "
            f"bibcode: {p.get('bibcode', '')} — citations: {p.get('citation_count', 0)} — "
            f"{(p.get('abstract') or '')[:250]}"
            for i, p in enumerate(ads_papers)
        )
        total = ads.output.get("total_found", len(ads_papers))
        parts.append(f"<nasa_ads_papers total=\"{total}\">\n{ads_lines}\n</nasa_ads_papers>")

    # INSPIRE HEP — particle/high-energy physics literature
    ihep = results.get("inspire_hep")
    if ihep and ihep.output.get("papers"):
        ihep_papers = ihep.output["papers"][:5]
        ihep_lines = "\n".join(
            f"  [{i+1}] {p.get('title', '')} ({p.get('year', '')}) — "
            f"citations: {p.get('citation_count', 0)} — "
            f"{(p.get('abstract') or '')[:250]}"
            for i, p in enumerate(ihep_papers)
        )
        total = ihep.output.get("total_found", len(ihep_papers))
        parts.append(f"<inspire_hep_papers total=\"{total}\">\n{ihep_lines}\n</inspire_hep_papers>")

    # OEIS — integer sequences (math namespace)
    oeis = results.get("oeis")
    if oeis and oeis.output.get("sequences"):
        seqs = oeis.output["sequences"][:4]
        seq_lines = "\n".join(
            f"  {s.get('id','')}: {s.get('name','')} — "
            f"values: {', '.join(str(v) for v in (s.get('sample_values') or [])[:8])}"
            + (f"\n    Python: {s.get('python_code','')[:200]}" if s.get('python_code') else "")
            for s in seqs
        )
        parts.append(f"<oeis_sequences>\n{seq_lines}\n</oeis_sequences>")

    # GitHub search — code repositories
    gh = results.get("github_search")
    if gh and gh.output.get("repositories"):
        repos = gh.output["repositories"][:5]
        repo_lines = "\n".join(
            f"  {r.get('full_name', r.get('name',''))} ★{r.get('stars',0)} "
            f"[{r.get('language','?')}] — {r.get('description','')[:200]} — {r.get('url','')}"
            for r in repos
        )
        total = gh.output.get("total_count", len(repos))
        parts.append(f"<github_repos total=\"{total}\">\n{repo_lines}\n</github_repos>")

    # HuggingFace — models and datasets
    hf = results.get("huggingface_search")
    if hf and hf.output.get("results"):
        hf_items = hf.output["results"][:5]
        hf_type = hf.output.get("search_type", "models")
        hf_lines = "\n".join(
            f"  {item.get('id', item.get('name',''))} — "
            f"downloads: {item.get('downloads',0):,} — "
            f"likes: {item.get('likes',0):,} — "
            f"tags: {', '.join((item.get('tags') or [])[:4])}"
            for item in hf_items
        )
        parts.append(f"<huggingface_{hf_type}>\n{hf_lines}\n</huggingface_{hf_type}>")

    # Papers with Code — benchmarks, SoTA results
    pwc = results.get("papers_with_code")
    if pwc and pwc.output.get("results"):
        pwc_items = pwc.output["results"][:5]
        pwc_type = pwc.output.get("search_type", "papers")
        pwc_lines = []
        for item in pwc_items:
            title = item.get("title") or item.get("name") or ""
            benchmarks = item.get("benchmarks") or []
            bm_str = " | ".join(
                f"{b.get('dataset','')}/{b.get('task','')}: {b.get('sota_result','')}"
                for b in benchmarks[:2]
            ) if benchmarks else ""
            pwc_lines.append(f"  {title}" + (f" — SoTA: {bm_str}" if bm_str else ""))
        parts.append(f"<papers_with_code type=\"{pwc_type}\">\n" + "\n".join(pwc_lines) + "\n</papers_with_code>")

    # Wikipedia — authoritative background knowledge
    wp = results.get("wikipedia")
    if wp and wp.output.get("found") and wp.output.get("summary"):
        parts.append(
            f"<wikipedia_background title=\"{wp.output.get('title', '')}\">\n"
            f"{wp.output['summary'][:2000]}\n"
            f"</wikipedia_background>"
        )

    # CrossRef — verified bibliographic metadata
    cr = results.get("crossref")
    if cr and cr.output.get("works"):
        cr_works = cr.output["works"][:4]
        cr_lines = "\n".join(
            f"  [{i+1}] {w.get('title', '')} — {w.get('journal', '')} ({w.get('year', '')}) "
            f"DOI: {w.get('doi', '')} — {(w.get('abstract') or '')[:200]}"
            for i, w in enumerate(cr_works)
        )
        parts.append(f"<crossref_metadata>\n{cr_lines}\n</crossref_metadata>")

    # Research trends — publication growth data
    rt = results.get("research_trends")
    if rt and rt.output.get("yearly_counts"):
        trend = rt.output.get("trend", "unknown")
        peak_year = rt.output.get("peak_year")
        topic = rt.output.get("topic", "")
        yearly = rt.output.get("yearly_counts", [])
        trend_lines = "  " + ", ".join(f"{y['year']}: {y['count']:,}" for y in yearly[-8:])
        venues = rt.output.get("top_venues", [])
        venues_str = ", ".join(v["name"] for v in venues[:4]) if venues else "N/A"
        parts.append(
            f"<research_trends topic=\"{topic}\">\n"
            f"  Trend: {trend} (peak year: {peak_year})\n"
            f"  Publication counts: {trend_lines}\n"
            f"  Top venues (recent): {venues_str}\n"
            f"</research_trends>"
        )

    # Author network — key researchers in the field
    an = results.get("author_network")
    if an and an.output.get("authors"):
        authors = an.output["authors"][:5]
        author_lines = "\n".join(
            f"  {a.get('name', '')} — h-index: {a.get('h_index', '?')}, "
            f"{a.get('citation_count', 0):,} citations, {a.get('paper_count', 0)} papers"
            + (f" [{', '.join(a.get('affiliations', [])[:2])}]" if a.get('affiliations') else "")
            for a in authors
        )
        parts.append(f"<key_researchers>\n{author_lines}\n</key_researchers>")

    # Paper Q&A — targeted answer about a specific paper's content
    pqa = results.get("paper_qa")
    if pqa and pqa.output.get("found") and pqa.output.get("answer"):
        paper_title = pqa.output.get("paper_title", "")
        parts.append(
            f"<paper_qa paper=\"{paper_title}\">\n"
            f"{pqa.output['answer'][:3000]}\n"
            f"</paper_qa>"
        )

    # Citation finder — ranked papers for citing a claim
    cf = results.get("citation_finder")
    if cf and cf.output.get("papers"):
        claim = cf.output.get("claim", "")
        cf_papers = cf.output["papers"][:6]
        cf_lines = "\n".join(
            f"  [{i+1}] {p.get('title', '')} ({p.get('year', '')}) — "
            f"{p.get('relevance_note') or (p.get('abstract') or '')[:200]}"
            for i, p in enumerate(cf_papers)
        )
        parts.append(
            f"<citation_candidates claim=\"{claim[:100]}\">\n"
            f"{cf_lines}\n"
            f"</citation_candidates>"
        )

    # LaTeX parse — document structure and equations
    lp = results.get("latex_parse")
    if lp and (lp.output.get("title") or lp.output.get("sections") or lp.output.get("equations")):
        lp_title = lp.output.get("title", "")
        lp_authors = ", ".join(lp.output.get("authors", [])[:4])
        lp_abstract = lp.output.get("abstract", "")[:600]
        lp_sections = lp.output.get("sections", [])
        sec_lines = "\n".join(
            f"  {'  ' * (s.get('level', 1) - 1)}{'#' * s.get('level', 1)} {s.get('title', '')}"
            for s in lp_sections[:12]
        )
        lp_equations = lp.output.get("equations", [])[:5]
        eq_lines = "\n".join(f"  {eq[:200]}" for eq in lp_equations)
        parts.append(
            f"<latex_document title=\"{lp_title}\">\n"
            f"Authors: {lp_authors}\n"
            f"Abstract: {lp_abstract}\n"
            + (f"Sections:\n{sec_lines}\n" if sec_lines else "")
            + (f"Key equations:\n{eq_lines}\n" if eq_lines else "")
            + f"</latex_document>"
        )

    # Media generation — job queued
    mg = results.get("media_generate")
    if mg and mg.output.get("status") == "queued":
        mg_type = mg.output.get("media_type", "")
        mg_href = mg.output.get("href", "")
        mg_count = mg.output.get("paper_count", 0)
        parts.append(
            f"<media_generation_queued>\n"
            f"{mg_type.capitalize()} generation started for {mg_count} paper(s). "
            f"Track progress at: {mg_href}\n"
            f"</media_generation_queued>"
        )

    return "\n\n".join(parts)


def _build_paper_context(papers: list[dict], arxiv_results: list[dict]) -> str:
    sections: list[str] = []
    if papers:
        sections.append("\n\n".join(
            f"[{i + 1}] {p.get('title')}\nAuthors: {', '.join(p.get('authors') or [])}\n"
            f"Namespace: {p.get('namespace_key', '')}\n"
            f"Abstract/TLDR: {p.get('tldr') or p.get('abstract', '')[:900]}"
            for i, p in enumerate(papers[:8])
        ))
    if arxiv_results:
        # Deduplicate against corpus papers already included above
        corpus_titles = {p.get("title", "").lower() for p in papers}
        deduped = [p for p in arxiv_results if p.get("title", "").lower() not in corpus_titles]
        if deduped:
            sections.append("\n\n".join(
                f"[A{i + 1}] {p.get('title')}\nAuthors: {', '.join(p.get('authors') or [])}\n"
                f"Abstract: {p.get('abstract', '')[:900]}"
                for i, p in enumerate(deduped[:6])
            ))
    return "\n\n".join(sections)


def _build_prompt(
    *,
    query: str,
    context: str,
    extra_context: str,
    actions: list[str],
    imported_count: int,
    graph_result: dict | None,
    genie_session_id: str | None,
    orientation: str,
    expertise: str,
    grounded: bool,
    pure_reasoning: bool = False,
) -> str:
    expertise_hint = _EXPERTISE_HINTS.get(expertise, "")
    orientation_hint = _ORIENTATION_HINTS.get(orientation, "")

    if pure_reasoning:
        # No retrieval evidence — direct knowledge answer.
        return (
            "You are ResearchFlow's AI-native research collaborator — brilliant, warm, precise, and deeply useful.\n\n"
            f"USER QUERY: {query}\n"
            f"USER PROFILE: expertise={expertise}; orientation={orientation}.\n"
            f"{expertise_hint} {orientation_hint}\n\n"
            f"{_FORMAT_GUIDANCE}\n\n"
            "No retrieval tools were needed for this query — answer directly from your knowledge.\n"
            "RESPONSE RULES:\n"
            "• Be accurate, clear, and appropriately detailed for the user's expertise level.\n"
            "• Use proper markdown formatting: bold key terms, headers for multi-section answers, bullet points where appropriate.\n"
            "• Do NOT fabricate citations. Only cite sources if you know them with certainty.\n"
            "• Write a complete answer — never truncate.\n\n"
            "Write your response now:"
        )

    grounding_clause = (
        "Papers marked [1], [2], … are from the user's INDEXED CORPUS — primary evidence."
        if grounded
        else "Papers marked [A1], [A2], … are unindexed arXiv CANDIDATES — treat as lightly-vetted leads, not verified evidence."
    )

    # Build the full evidence section combining papers + all other tool outputs
    evidence_parts: list[str] = []
    if context:
        evidence_parts.append(f"── Corpus papers ──\n{context}")
    if extra_context:
        evidence_parts.append(f"── Additional tool outputs ──\n{extra_context}")
    if graph_result:
        evidence_parts.append(f"── Knowledge graph ──\n{json.dumps(graph_result, default=str)[:600]}")
    if genie_session_id:
        evidence_parts.append("── Genie synthesis ── A novel hypothesis was generated in a Genie session (id surfaced in UI).")
    if imported_count:
        evidence_parts.append(f"── arXiv imports ── {imported_count} new paper(s) imported this turn (embed in next turn).")

    evidence_block = "\n\n".join(evidence_parts)

    return (
        "You are ResearchFlow's AI-native research collaborator — brilliant, warm, precise, and deeply useful.\n\n"
        "<evidence>\n"
        f"{evidence_block}\n"
        "</evidence>\n\n"
        f"<grounding>{grounding_clause}</grounding>\n\n"
        f"USER QUERY: {query}\n"
        f"USER PROFILE: expertise={expertise}; orientation={orientation}.\n"
        f"{expertise_hint} {orientation_hint}\n\n"
        f"TOOLS USED THIS TURN: {', '.join(actions) or 'none'}\n\n"
        f"{_LIFECYCLE_GUIDANCE}\n\n"
        f"{_FORMAT_GUIDANCE}\n\n"
        "RESPONSE RULES:\n"
        "• ResearchFlow is arXiv-first. Prefer arXiv-sourced papers as primary evidence. Treat Wikipedia, web search, CrossRef, and Semantic Scholar as supplementary — use them to enrich or contextualise, not to replace arXiv evidence.\n"
        "• Synthesize ALL available evidence (papers, computations, attachments, web) into one coherent answer.\n"
        "• Cite inline: [1] for indexed papers, [A1] for arXiv candidates, [WA] for Wolfram Alpha, [W] for web sources.\n"
        "• Distinguish fact from hypothesis; acknowledge evidence gaps honestly.\n"
        "• Adapt voice and depth to the user's expertise level.\n"
        "• Close with one concrete, stage-appropriate next step (unless the question is fully resolved).\n"
        "• NEVER produce a rigid Takeaway / Evidence / Gaps / Next-moves skeleton. Choose the shape that fits.\n"
        "• NEVER reproduce XML/section tags from the evidence block (like <wolfram_computation>, <concept_background>, etc.) in your response — those are internal context markers only.\n"
        "• Use proper markdown: bold key terms, use headers sparingly, bullet points for lists, italics for emphasis.\n"
        "• Write a complete answer — NEVER cut off mid-sentence or truncate.\n\n"
        "Write your response now:"
    )


def build_message_blocks(
    *,
    answer: str,
    papers: list[dict],
    arxiv_results: list[dict],
    imported_count: int,
    graph_result: dict | None,
    genie_session_id: str | None,
    suggestions: list[dict],
    actions: list[str],
    web_results: list[dict] | None = None,
    comparison: dict | None = None,
    bookmarks_answer: str | None = None,
    mermaid: tuple[str, str] | None = None,
    domain_papers: list[dict] | None = None,
    nvd_results: list[dict] | None = None,
    fred_series: list[dict] | None = None,
    trials_results: list[dict] | None = None,
    code_results: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the structured ``payload.blocks`` list rendered by the UI.

    Block kinds: ``text``, ``paper_grid``, ``arxiv_grid``, ``source_papers``,
    ``graph_summary``, ``web_results``, ``comparison_table``, ``bookmarks_answer``,
    ``artifact_link``, ``suggestion_chips``, ``actions_taken``,
    ``nvd_results``, ``fred_data``, ``trials_results``, ``code_results``.
    Frontend dispatches per-block so the assistant message is never a wall of markdown.
    """
    blocks: list[dict[str, Any]] = []
    if answer:
        blocks.append({"kind": "text", "content": answer})
    if comparison and comparison.get("rows"):
        blocks.append({
            "kind": "comparison_table",
            "title": "Side-by-side comparison",
            "columns": comparison.get("columns") or [],
            "rows": comparison.get("rows") or [],
            "notes": comparison.get("notes") or "",
        })
    if bookmarks_answer:
        blocks.append({
            "kind": "bookmarks_answer",
            "title": "From your bookmarked papers",
            "content": bookmarks_answer,
        })
    if mermaid:
        title, code = mermaid
        # Only include if the code is non-trivial (real diagram, not empty or error)
        if code and len(code.strip()) > 40 and not code.strip().startswith("error"):
            blocks.append({
                "kind": "mermaid",
                "title": title,
                "code": code,
            })
    if papers:
        blocks.append({
            "kind": "paper_grid",
            "title": "Grounded papers",
            "papers": papers[:10],
        })
    if arxiv_results and not papers:
        blocks.append({
            "kind": "arxiv_grid",
            "title": (
                f"arXiv candidates ({imported_count} imported)"
                if imported_count else "arXiv candidates (browse only)"
            ),
            "papers": arxiv_results[:8],
            "imported_count": imported_count,
        })
    # Domain-specific paper results (pubmed, inspire_hep, nasa_ads, papers_with_code)
    if domain_papers:
        blocks.append({
            "kind": "source_papers",
            "title": "Domain literature",
            "papers": domain_papers[:12],
        })
    # Security vulnerability data (nvd_cve)
    if nvd_results:
        blocks.append({
            "kind": "nvd_results",
            "title": f"Security vulnerabilities ({len(nvd_results)} found)",
            "vulnerabilities": nvd_results[:8],
        })
    # Clinical trials (clinicaltrials.gov)
    if trials_results:
        blocks.append({
            "kind": "trials_results",
            "title": f"Clinical trials ({len(trials_results)} found)",
            "studies": trials_results[:8],
        })
    # FRED macroeconomic time series
    if fred_series:
        blocks.append({
            "kind": "fred_data",
            "title": "FRED economic data",
            "series": fred_series[:4],
        })
    # Code repositories and models (github, huggingface)
    if code_results:
        blocks.append({
            "kind": "code_results",
            "title": "Code & models",
            "items": code_results[:10],
        })
    if web_results:
        blocks.append({
            "kind": "web_results",
            "title": "Web context (secondary)",
            "results": web_results[:6],
        })
    # Only show graph block when the graph has actual renderable content
    if graph_result and _graph_has_content(graph_result):
        blocks.append({
            "kind": "graph_summary",
            "title": "Knowledge graph",
            "summary": graph_result,
            "href": "/graph",
        })
    if genie_session_id:
        blocks.append({
            "kind": "artifact_link",
            "title": "Genie synthesis queued",
            "kind_label": "genie",
            "href": "/genie?tab=discoveries",
            "ref_id": genie_session_id,
        })
    if suggestions:
        blocks.append({
            "kind": "suggestion_chips",
            "title": "Next moves",
            "suggestions": suggestions,
        })
    if actions:
        blocks.append({
            "kind": "actions_taken",
            "actions": actions,
        })
    return blocks


def _graph_has_content(graph_result: dict) -> bool:
    """Return True only when the graph has actual nodes/edges worth rendering."""
    if not graph_result:
        return False
    nodes = graph_result.get("nodes") or graph_result.get("node_count") or []
    edges = graph_result.get("edges") or graph_result.get("edge_count") or []
    n = len(nodes) if isinstance(nodes, (list, dict)) else int(nodes or 0)
    e = len(edges) if isinstance(edges, (list, dict)) else int(edges or 0)
    return n > 0 or e > 0


def _fallback_answer(
    query: str,
    papers: list[dict],
    imported_count: int,
    graph_result: dict | None,
    genie_session_id: str | None,
) -> str:
    if not papers:
        # Honest, actionable empty state — never claim something we don't have.
        # The block renderer surfaces arxiv_results separately so the user
        # still sees candidates without us pretending they're "grounded".
        ns_msg = (
            f"I imported {imported_count} new paper(s) this turn but they "
            "haven't been embedded into the searchable corpus yet — "
            "try the same question again in a moment."
            if imported_count > 0
            else "No indexed papers matched in the current scope."
        )
        return (
            "I don't have grounded evidence to answer this question right now — "
            "I'm flagging that honestly rather than improvising.\n\n"
            f"{ns_msg}\n\n"
            "**What to try next:**\n"
            "- Rephrase the question with broader terminology\n"
            "- Enable arXiv import so fresh candidates land in your feed\n"
            "- Try a follow-up like \"search across all namespaces\""
        )
    lines = [f"I found {len(papers)} relevant paper(s) for your question.\n"]
    for i, p in enumerate(papers[:5], start=1):
        ns = p.get("namespace_key") or ""
        lines.append(f"- **[{i}]** {p.get('title')}{f' ({ns})' if ns else ''}")
    if genie_session_id:
        lines.append(
            "\n*A Genie synthesis workflow has been queued — it will appear "
            "in Genie discoveries when complete.*"
        )
    lines.append(
        "\n*(Synthesis model unavailable — this is a structured fallback without "
        "LLM-composed analysis.)*"
    )
    return "\n".join(lines)
