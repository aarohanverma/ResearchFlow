"""LLM-driven planner.

Uses the platform's LLM adapter with structured JSON output to pick which
tools to call, in which order, with what params. Falls back to the
``HeuristicPlanner`` whenever the LLM is unavailable, returns malformed
JSON, or proposes invalid steps — so the assistant always produces a
valid plan even when the provider is offline.

Tool schemas come from the registry's ``describe_for_planner()`` view; the
planner never sees implementation code.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.assistant.planner import HeuristicPlanner, Plan, PlannedStep
from app.assistant.tools.registry import describe_for_planner, get_tool

log = logging.getLogger(__name__)


_PLANNER_SYSTEM = """You are the planner for ResearchFlow's Research Assistant (RA).

Your job: given the user's query, conversation history, and memory, produce a sharply reasoned, multi-step tool plan. You are a fully agentic research system — always build a real execution plan with at least 1–2 tools. Enrich even simple queries with retrieval, context, and synthesis. Do not shortcut to "pure reasoning" except for pure greetings/acknowledgments.

ResearchFlow has a slight preference for arXiv as a primary source. For CS/AI/ML/Physics/Math queries, arXiv tools are often the best first step. However, you MUST also use domain-specific and enrichment tools when they are clearly better suited (PubMed for biomedical, INSPIRE HEP for particle physics, NASA ADS for astrophysics, FRED for economics, OEIS for math sequences, etc.). ArXiv is a preference, not a hard constraint — always use the right tool for the job.

════════════════════════════════════════
TOOL CATALOGUE
════════════════════════════════════════

The full tool catalogue — name, one-paragraph WHAT/WHEN summary, JSON input
schema, cost class, and side-effect flag — is provided dynamically as
``Available tools`` in the user message. Each tool's ``summary`` is the
authoritative reference for that tool's behaviour. Two policies stay inline
because they apply across tools:

  1. ``genie_synthesize`` and ``media_generate`` are HUMAN-IN-THE-LOOP (see
     policies near the end of this prompt).
  2. ``graph_build`` is forbidden from RA plans entirely — the knowledge
     graph is owned by the dedicated /graph page.

(The verbose per-tool prose that used to live here has been removed — it
duplicated the dynamic catalogue and burned ~1.5k tokens per planner call
for no marginal accuracy gain.)


════════════════════════════════════════
DECISION QUICK-REFERENCE
════════════════════════════════════════

TOOL SELECTION PRINCIPLES:
─────────────────────────────────────────────────────────────────────────────

INTENT → TOOL MAPPING (read this list first; pick the tool that BEST matches the user's intent):

  Intent: "explain a concept / theory / how something works"
    → concept_explain (single concept) or literature_survey (full landscape)

  Intent: "find papers / discover work on X"
    → deep_search (corpus + arXiv MCP, default first step)
    → arxiv_import when corpus is thin or specific landmark papers needed
    → frontier_scan when the user wants what's NEW (last few weeks)

  Intent: "I know which paper(s) I want — ingest these"
    → paper_import (specific IDs / URLs / pasted citations, max 10)

  Intent: "give me a guided walkthrough of THIS paper"
    → study_paper (full Study Mode walkthrough, cached by expertise level)

  Intent: "answer a specific question about ONE paper"
    → paper_qa (RAG over that paper's chunks)

  Intent: "compare these papers / methods / models"
    → compare_papers (structured side-by-side table)

  Intent: "what should I cite for claim X"
    → citation_finder

  Intent: "draft a related-work / intro / methodology section"
    → draft_section

  Intent: "summarise the landscape / state of the art"
    → literature_survey (multi-section structured survey)

  Intent: "explore an existing Genie idea further"
    → genie_deep_dive (full Deep Dive article, generates one if missing)

  Intent: "what ideas have I generated / show my hypotheses"
    → genie_read (lightweight capsule list)

  Intent: "synthesise a NEW idea from PAPERS" — strictly HITL, see policy below
    → genie_synthesize

  Intent: "combine / fuse / merge my saved ideas"
    → genie_combine (2 or 3 capsule ids → new hybrid capsule; runs feasibility judge first)

  Intent: "what's connected to X in the knowledge graph"
    → graph_query (only when a graph has been built — check context)

  Intent: "search the web / news / blog posts"
    → web_search (low-trust, label external in synthesis)
    → wikipedia for encyclopaedic background

  Intent: "compute / solve / evaluate a math expression"
    → wolfram_alpha

  Intent: "produce a podcast / slide deck for paper or capsule"
    → media_generate

  Intent: user referenced an UPLOADED file or URL
    → parse_context (read its content into the synthesizer)

  Intent: domain-specific search
    → pubmed (biomedical) · inspire_hep (HEP/nuclear/quantum) · nasa_ads (astro)
    → fred (economics) · clinicaltrials (medicine) · nvd_cve (security) · oeis (math)
    → github_search / huggingface_search / papers_with_code (code / models / benchmarks)
    → crossref / unpaywall (DOI / open-access PDFs) · research_trends (publication trends)
    → citation_finder (find papers to cite for a claim) · latex_parse (parse a LaTeX source URL)

  Memory tools:
    → memory_recall — only when continuity / prior context is needed
    → memory_write — only when a substantive fact is worth persisting
    → memory_delete — only when the user explicitly asks to forget something

  Pure greeting / acknowledgment / off-topic:
    → empty steps (no tools).

DISCIPLINE:
• PICK MINIMUM SUFFICIENT SET. Two well-chosen tools beat six speculative ones.
• Don't bundle tools "just in case". Every step costs latency + tokens.
• If two intents are both possible, sequence them: heavier retrieval first, then concept_explain / draft / compare on top of the retrieved corpus.
• Domain hint: when the namespace is q-bio/q-bio.*, START with pubmed; for astro-ph, START with nasa_ads; for hep-*, START with inspire_hep; for econ/q-fin, START with fred. arXiv tools remain fallback in those domains.
• Side-effect tools (`arxiv_import`, `paper_import`, `genie_*`, `media_generate`) run when the user clearly asked OR when retrieval has yielded specific targets — never speculatively.
• `study_paper` and `genie_deep_dive` are cache-first: re-asking the same paper at the same expertise level returns instantly. Pick them confidently when the intent matches.
• `genie_combine` requires AT LEAST TWO previously-saved capsules. Do not invoke it from an empty session — chain with `genie_read` first if you need to discover the parent ids.

Always reason about which combination of tools best serves the specific query — do not follow a fixed recipe.

════════════════════════════════════════
MEMORY MANAGEMENT POLICY
════════════════════════════════════════

Always recall memory at the start of a session continuation or when user references prior context.
Write memory proactively when:
  • A key research finding emerges (scope="medium")
  • User states a preference, background, or constraint (scope="medium")
  • A major insight should persist across all sessions in this namespace (scope="long")
Do NOT write trivial facts. One substantive write per turn maximum.

════════════════════════════════════════
GENIE HITL POLICY — ENFORCE STRICTLY
════════════════════════════════════════

genie_synthesize must NOT appear in the plan unless ALL three hold simultaneously:
  1. User EXPLICITLY requested it THIS message (exact phrases: "generate hypothesis", "run Genie", "synthesize ideas", "create a research direction", "new hypothesis", "brainstorm idea", "run synthesis").
  2. Conversation has ≥3 substantive prior assistant responses (not just acknowledgments).
  3. Session has explored a specific, narrow research theme — not broad curiosity.
Missing ANY condition → omit entirely. Never include speculatively or as a "next step" suggestion.

════════════════════════════════════════
MEDIA GENERATION HITL POLICY — ENFORCE STRICTLY
════════════════════════════════════════

media_generate must NOT appear in the plan unless ALL hold simultaneously:
  1. User EXPLICITLY requested podcast or slides generation in this message (phrases: "generate a podcast", "make slides", "create a slide deck", "generate slides", "create a podcast episode", "make a podcast").
  2. User has named or selected specific papers — you have their UUIDs in the context from prior tool results.
  3. Conversation has enough research depth to produce quality media.
  4. You pass only paper_ids the user explicitly named or selected — NEVER infer IDs.
NEVER trigger proactively, speculatively, or without explicit paper selection.
If the user asks for media but hasn't selected papers, ask them to select papers first — do NOT trigger.

════════════════════════════════════════
PARALLEL EXECUTION
════════════════════════════════════════

Set parallel=true on steps that are mutually independent (no data dependency). They run concurrently.
Example: deep_search and web_search on different sub-questions can run in parallel.
Example: literature_survey CANNOT run in parallel with arxiv_import when it depends on the import results.

════════════════════════════════════════
AGENTIC BEHAVIOR — ALWAYS BUILD A PLAN
════════════════════════════════════════

You are a fully agentic research system. ALWAYS produce at least 1–2 tool steps for substantive queries.
Even simple-seeming questions benefit from retrieval + synthesis. "What is attention?" → deep_search + concept_explain.
"Explain backpropagation" → deep_search + concept_explain. "Is RLHF good?" → deep_search.

PURE REASONING (empty steps) is ONLY acceptable for:
  • Pure greetings: "Hi", "Thanks!", "Great job"
  • Pure navigation: "Go back", "Show me tab X"
  • Pure computation already covered by wolfram_alpha (still use wolfram_alpha, not empty steps)

For any substantive research, educational, factual, or analytical query — ALWAYS use tools.
The synthesizer can fall back to reasoning if tool outputs are thin, but the planner must try.

════════════════════════════════════════
CONSTRAINTS
════════════════════════════════════════

• Only use tools from this catalogue. Never invent tool names.
• graph_build is UNAVAILABLE — built from the /graph page only.
• Never use web_search AND deep_search for the same sub-question — pick one.
• Prefer deep_search for follow-ups on previously discussed topics.
• Scope is NOT just the feed: include arxiv_import when the user needs foundational papers not yet ingested.
• Never assume the corpus is sufficient — when uncertain, include arxiv_import.
"""

_PLAN_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string", "description": "One short paragraph explaining the plan."},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "actions": {"type": "array", "items": {"type": "string"}},
        "steps": {
            "type": "array",
            "description": "Ordered tool steps. EMPTY array = pure reasoning mode (no tools needed).",
            "items": {
                "type": "object",
                "properties": {
                    "tool":      {"type": "string"},
                    "title":     {"type": "string"},
                    "params":    {"type": "object"},
                    "rationale": {"type": "string"},
                    "parallel":  {
                        "type": "boolean",
                        "description": (
                            "Set true when this step can run concurrently with "
                            "other parallel=true steps (they don't depend on each "
                            "other's output). Sequential dependencies must be false."
                        ),
                    },
                },
                "required": ["tool", "title", "params"],
            },
        },
    },
    "required": ["rationale", "steps"],
}


_GENIE_EXPLICIT_KEYWORDS = frozenset({
    "genie", "generate hypothesis", "generate hypothes", "run genie",
    "synthesize ideas", "synthesize idea", "create research direction",
    "new hypothesis", "novel idea", "research direction", "brainstorm",
    "creative research", "ideate", "ideation",
})

_COMPLEXITY_DEEP_KEYWORDS = frozenset({
    "survey", "literature review", "comprehensive", "deep dive",
    "compare", "contrast", "explain thoroughly", "analyze", "analyse",
    "mechanism", "trade-off", "tradeoff", "benchmark", "evaluate",
    "relationship between", "how does", "why does", "in-depth",
    "synthesis", "theoretical", "mathematical", "proof", "derive",
})

_COMPLEXITY_SIMPLE_KEYWORDS = frozenset({
    "what is", "define", "definition", "who is", "when was",
    "how many", "list", "name", "give me", "tell me",
    "quick", "brief", "tldr", "summary of",
})


def _assess_query_complexity(query: str, history: list[dict]) -> str:
    """Return 'simple', 'medium', or 'complex' for adaptive compute routing.

    Leans toward 'medium' by default — we'd rather over-invest in a good plan
    than under-invest and produce a shallow answer.
    """
    q = query.lower()
    n_history = sum(1 for m in history if m.get("role") == "assistant")

    # Explicit deep-research markers → complex
    if any(kw in q for kw in _COMPLEXITY_DEEP_KEYWORDS):
        return "complex"

    # Multi-part queries or long queries → complex
    if len(q) > 180 or q.count("?") >= 2 or " vs " in q or " versus " in q:
        return "complex"

    # Conversation with depth → bump to complex (user is in a research flow)
    if n_history >= 5:
        return "complex"

    # Any research/learning query with moderate context → medium
    if n_history >= 1:
        return "medium"

    # Clear one-off simple lookup (short + trivial phrasing) → simple
    # Intentionally narrow: only exact simple-phrasing + very short queries.
    if any(q.startswith(kw) for kw in _COMPLEXITY_SIMPLE_KEYWORDS) and len(q) < 80:
        return "simple"

    # Default: medium — most research queries benefit from 2-3 tools
    return "medium" if len(q) < 160 else "complex"


def _genie_hitl_permitted(query: str, history: list[dict]) -> bool:
    """Return True only when Genie synthesis is appropriate to include in the plan.

    Both conditions must hold simultaneously:
    1. The user explicitly asked for Genie/hypothesis output in the current query.
    2. The conversation has ≥3 substantive prior assistant turns (enough research context).

    This is strict by design — Genie synthesis is expensive and irreversible.
    The planner dropping genie_synthesize simply means the user can ask again
    once more context has accumulated.
    """
    q = query.lower()
    explicit = any(kw in q for kw in _GENIE_EXPLICIT_KEYWORDS)
    if not explicit:
        return False

    # Even with an explicit request, require a minimum research depth.
    # Count only substantive (non-empty) assistant turns.
    n_substantive = sum(
        1 for m in history
        if m.get("role") == "assistant" and len((m.get("content") or "").strip()) > 50
    )
    return n_substantive >= 3


class LLMPlanner:
    """Plan a turn by asking the quality LLM to fill a structured Plan schema."""

    name = "llm"

    def __init__(self, fallback: HeuristicPlanner | None = None) -> None:
        self._fallback = fallback or HeuristicPlanner()

    def plan(
        self,
        *,
        query: str,
        namespace_key: str,
        namespace_keys: list[str],
        history: list[dict] | None = None,
        orientation: str = "both",
        expertise: str = "practitioner",
    ) -> Plan:
        """Return a Plan, falling back to the heuristic planner on any failure.

        Note: synchronous interface — the orchestrator calls this with
        ``await asyncio.to_thread(self.plan, ...)`` to avoid blocking the
        loop on LLM calls. See ``aplan`` for the native async variant.
        """
        # Synchronous wrapper exists only because the heuristic planner is sync;
        # the orchestrator uses ``aplan`` directly so it can await the LLM.
        return self._fallback.plan(
            query=query, namespace_key=namespace_key, namespace_keys=namespace_keys,
        )

    async def aplan(
        self,
        *,
        query: str,
        namespace_key: str,
        namespace_keys: list[str],
        history: list[dict] | None = None,
        orientation: str = "both",
        expertise: str = "practitioner",
        memory: dict | None = None,
        disabled_tools: set[str] | None = None,
    ) -> Plan:
        """Async planner — calls the LLM with adaptive compute based on query complexity."""
        # Pass namespace_key so only tools visible for this namespace are shown.
        catalogue = describe_for_planner(namespace_key=namespace_key)
        if disabled_tools:
            catalogue = [t for t in catalogue if t.get("name") not in disabled_tools]

        # Assess query complexity to choose planning model + depth
        complexity = _assess_query_complexity(query, history or [])

        try:
            from app.adapters.llm import get_llm_adapter

            llm = get_llm_adapter()
            prompt = self._build_prompt(
                query=query,
                namespace_key=namespace_key,
                namespace_keys=namespace_keys,
                history=history or [],
                orientation=orientation,
                expertise=expertise,
                catalogue=catalogue,
                memory=memory or {},
                complexity=complexity,
            )
            messages = [
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            # Always use quality_model for planning — reasoning model adds latency
            # with negligible plan quality gain. The synthesizer uses the reasoning
            # model where it matters: grounded evidence composition.
            plan_model = llm.quality_model
            reasoning_effort = None
            raw = await llm.complete_structured(
                messages,
                plan_model,
                _PLAN_RESPONSE_SCHEMA,  # type: ignore[arg-type]
                reasoning_effort=reasoning_effort,
            )
            plan = self._materialize(raw, query=query, namespace_key=namespace_key,
                                     namespace_keys=namespace_keys,
                                     extra_forbidden=disabled_tools or set(),
                                     history=history or [])
            if plan is None:
                raise ValueError("planner returned no usable steps")
            return plan
        except Exception as exc:
            log.warning("LLM planner fell back to heuristic: %s", exc)
            return self._fallback.plan(
                query=query, namespace_key=namespace_key, namespace_keys=namespace_keys,
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        *,
        query: str,
        namespace_key: str,
        namespace_keys: list[str],
        history: list[dict],
        orientation: str,
        expertise: str,
        catalogue: list[dict],
        memory: dict | None = None,
        complexity: str = "medium",
    ) -> str:
        recent = history[-14:] if history else []
        history_blob = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content') or ''}" for m in recent
        )
        # Strip JSONSchema metadata that adds bytes without changing what the
        # planner needs to know to fill in valid params. We keep ``type``,
        # ``properties``, ``required``, ``items``, ``enum``, ``description``,
        # ``minimum``/``maximum``, ``minLength``/``maxLength``,
        # ``minItems``/``maxItems``, ``pattern``, ``default``. We drop:
        # ``$schema``, ``$defs``, ``additionalProperties``, ``title`` (often
        # a duplicate of the field name), and any other rarely-meaningful
        # housekeeping the Pydantic emitter adds.
        def _strip_schema(node: object) -> object:
            if isinstance(node, dict):
                drop = {"$schema", "$defs", "additionalProperties", "title"}
                return {k: _strip_schema(v) for k, v in node.items() if k not in drop}
            if isinstance(node, list):
                return [_strip_schema(v) for v in node]
            return node

        # Compact JSON (no indent) — parses identically for the LLM but costs
        # ~30% fewer tokens because each leading newline + indentation pair
        # is a separate token.
        catalogue_blob = json.dumps(
            [
                {
                    "name": t["name"],
                    "summary": t["summary"],
                    "cost_class": t["cost_class"],
                    "side_effects": t["side_effects"],
                    "input_schema": _strip_schema(t["input_schema"]),
                }
                for t in catalogue
            ],
            separators=(",", ":"),
        )
        mem = memory or {}
        medium_mem = mem.get("medium") or {}
        long_mem = mem.get("long") or {}

        def _mem_val(v: object) -> str:
            if isinstance(v, dict):
                return str(v.get("value", v))
            return str(v)

        memory_blob = ""
        if medium_mem or long_mem:
            memory_blob = "\nResearch memory (use this to personalize the plan):\n"
            if medium_mem:
                memory_blob += "Session facts:\n" + "\n".join(
                    f"  {k}: {_mem_val(v)}" for k, v in list(medium_mem.items())[:10]
                ) + "\n"
            if long_mem:
                memory_blob += "Namespace insights:\n" + "\n".join(
                    f"  {k}: {_mem_val(v)}" for k, v in list(long_mem.items())[:10]
                ) + "\n"

        # Namespace pack hint — tell the planner about domain-specific tools available
        from app.assistant.tools.namespace_packs import get_pack_description
        pack_hint = get_pack_description(namespace_key)
        pack_blob = f"\nNamespace pack: {pack_hint}\n" if pack_hint else ""

        depth_hint = {
            "simple": (
                "Minimum 2 tools: retrieve from corpus/domain source first, then enrich with "
                "a secondary tool (concept_explain, wikipedia, web_search, crossref). "
                "Even brief factual questions benefit from grounded evidence."
            ),
            "medium": (
                "2-4 tools in a logical sequence. Retrieve broadly (deep_search or domain tool), "
                "enrich with context (concept_explain, research_trends, author_network), "
                "and optionally synthesize or compare. Run parallel steps for unrelated sub-tasks."
            ),
            "complex": (
                "4-6 tools. Deep multi-angle investigation: primary retrieval (deep_search + "
                "domain tools in parallel), secondary enrichment (research_trends, author_network, "
                "crossref), analysis (compare_papers, concept_explain, paper_qa), and optionally "
                "literature_survey. Maximize parallel steps for unrelated lookups. "
                "Synthesize everything into a coherent answer."
            ),
        }.get(complexity, "Medium complexity.")
        return (
            f"User request: {query}\n"
            f"Query complexity: {complexity} — {depth_hint}\n"
            f"Active namespace: {namespace_key}\n"
            f"Topic scope: {', '.join(namespace_keys) or namespace_key}\n"
            f"User profile (soft bias): expertise={expertise}, orientation={orientation}\n"
            f"{pack_blob}"
            f"{memory_blob}"
            f"\nConversation history (most recent last):\n{history_blob or '(no prior turns)'}\n"
            f"\nAvailable tools:\n{catalogue_blob}\n"
            "\nReturn ONLY a JSON object:\n"
            "{\n"
            '  "rationale": "<1-2 sentences explaining the plan>",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "actions": ["<short action label>", ...],\n'
            '  "steps": [{"tool": "<name>", "title": "<short>", '
            '"params": {<matches input_schema>}, "rationale": "<why>", "parallel": false}]\n'
            "}\n"
            "REMINDER: Build a rich, multi-step plan. Retrieve first, then enrich and synthesize. "
            "Use parallel=true for independent lookups. steps=[] ONLY for pure greetings.\n"
        )

    def _materialize(
        self,
        raw: dict[str, Any],
        *,
        query: str,
        namespace_key: str,
        namespace_keys: list[str],
        extra_forbidden: set[str] | None = None,
        history: list[dict] | None = None,
    ) -> Plan | None:
        if not isinstance(raw, dict):
            return None
        steps_raw = raw.get("steps") or []
        if not isinstance(steps_raw, list) or not steps_raw:
            return None
        ns_keys = namespace_keys or [namespace_key]
        steps: list[PlannedStep] = []
        # graph_build is always forbidden from RA plans (owned by /graph page).
        # extra_forbidden carries per-request unavailable tools (e.g. wolfram_alpha
        # when neither env key nor user key is configured).
        forbidden = {"graph_build"} | (extra_forbidden or set())

        # HITL guard: genie_synthesize is off-limits unless the user explicitly
        # asked for it AND the conversation is deep enough to warrant it.
        if not _genie_hitl_permitted(query, history or []):
            forbidden = forbidden | {"genie_synthesize"}

        # HITL guard: media_generate requires explicit request with paper IDs.
        # The planner prompt enforces this, but we double-check at materialization.
        media_keywords = frozenset({
            "generate a podcast", "make a podcast", "create a podcast",
            "generate slides", "make slides", "create slides", "create a slide deck",
            "generate slide deck", "make a slide deck",
        })
        q_lower = query.lower()
        if not any(kw in q_lower for kw in media_keywords):
            forbidden = forbidden | {"media_generate"}
        for s in steps_raw:
            if not isinstance(s, dict):
                continue
            tool_name = str(s.get("tool") or "").strip()
            if tool_name in forbidden:
                log.info("planner: dropped forbidden tool %s from RA plan", tool_name)
                continue
            tool = get_tool(tool_name)
            if not tool:
                # Unknown tool — drop silently; heuristic fallback handles
                # the case where every step is invalid.
                continue
            params = dict(s.get("params") or {})
            # Inject sensible defaults the planner often forgets.
            params.setdefault("namespace_key", namespace_key)
            params.setdefault("namespace_keys", ns_keys)
            params.setdefault("query", query)
            try:
                tool.input_schema(**params)
            except Exception as exc:
                log.warning("planner produced invalid params for tool=%s: %s", tool_name, exc)
                continue
            steps.append(PlannedStep(
                tool=tool_name,
                title=str(s.get("title") or tool_name),
                params=params,
                rationale=str(s.get("rationale") or ""),
                parallel=bool(s.get("parallel", False)),
            ))
        actions = [str(a) for a in (raw.get("actions") or []) if isinstance(a, str)]
        if not steps:
            # Zero-step plan: pure reasoning mode — no tools needed.
            return Plan(
                rationale=str(raw.get("rationale") or "Pure reasoning — no tool retrieval needed."),
                steps=[],
                actions=actions or ["Direct reasoning"],
                trace=[
                    {"step": "planner", "summary": "LLM chose pure reasoning mode (no tools)"},
                ],
                confidence=float(raw.get("confidence") or 0.9),
            )
        if not actions:
            actions = [s.title for s in steps]
        return Plan(
            rationale=str(raw.get("rationale") or "LLM-planned execution."),
            steps=steps,
            actions=actions,
            trace=[
                {"step": "planner", "summary": "LLM produced an ordered tool sequence"},
                {"step": "scope", "summary": f"Namespace={namespace_key} topics={ns_keys}"},
            ],
            confidence=float(raw.get("confidence") or 0.7),
        )
