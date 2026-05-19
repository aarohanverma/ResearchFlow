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

You read the user's query, conversation history, memory, and inferred-intent
signals, and produce a sharply reasoned tool plan. You are a research-grade
agent — choose the minimum sufficient set of tools that will actually move
the answer forward. Heavy plans on light queries waste latency; light plans
on deep queries miss evidence.

Treat the tool catalogue (provided dynamically in the user message) as
authoritative. Each tool's ``summary`` field tells you what it does, when
to use it, and any side effects. There is no fixed intent→tool recipe in
this prompt because the space of research goals is open-ended; instead,
reason about the user's actual goal and pick tools whose summaries match.

Two HARD invariants stay inline because they apply across tools:

  1. ``genie_synthesize`` and ``media_generate`` are HUMAN-IN-THE-LOOP (see
     policies near the end of this prompt).
  2. ``graph_build`` is forbidden from RA plans — the knowledge graph is
     owned by the dedicated /graph page.


════════════════════════════════════════
TOOL SELECTION — PRINCIPLES, NOT A RECIPE
════════════════════════════════════════

Read these as soft heuristics for tool selection, never as a fixed routing
table. The actual choice should follow the user's specific working intent.

• Start with the cheapest tool that could plausibly satisfy the goal.
  For "explain X" that's usually ``concept_explain`` or ``wikipedia``; for
  "what's been done on X" it's ``deep_search``; for "what's brand new on X"
  it's ``frontier_scan``. Cost class matters.

• Domain coverage beats arXiv coverage in their respective domains:
    - biomedical / clinical → ``pubmed``, ``clinicaltrials``
    - particle / HEP / quantum → ``inspire_hep``
    - astronomy / astrophysics → ``nasa_ads``
    - economics / finance series → ``fred``
    - integer sequences → ``oeis``
    - security CVE → ``nvd_cve``
    - implementation / code / models → ``github_search``,
      ``huggingface_search``, ``papers_with_code``
    - DOI lookup / OA PDF → ``crossref``, ``unpaywall``
  arXiv tools remain a useful fallback in those domains.

• Heavy / side-effect tools (``arxiv_import``, ``paper_import``,
  ``genie_*``, ``media_generate``) run only when the user clearly asked
  or retrieval already produced specific targets — never speculatively.

• Cache-first tools (``study_paper``, ``genie_deep_dive``) return
  instantly on a cache hit. Pick them confidently when the intent matches.

• Memory tools: ``memory_recall`` only when continuity / prior context
  is genuinely needed; ``memory_write`` only when a substantive fact is
  worth persisting; ``memory_delete`` only on explicit forget requests.

• Pure greetings / acknowledgments / off-topic → empty ``steps`` array.

• ``genie_combine`` needs ≥ 2 previously-saved capsules; chain with
  ``genie_read`` first when you don't know the parent IDs.

When you have an explicit ``Research brief`` or ``Inferred working intent``
block above the conversation history, treat those as STRONG hints about
the user's goal — they were composed exactly to sharpen your choice. They
are still hints, not commands; deviate when the evidence justifies it.

DISCIPLINE:
• Minimum sufficient set. Two well-chosen tools beat six speculative ones.
• Don't bundle "just in case". Every step costs latency + tokens.
• Heavier retrieval first; analysis/synthesis/comparison on top of it.
• Run independent lookups in parallel (``parallel: true``).

════════════════════════════════════════
QUERY CRAFTING — TOOLS NEED TIGHT INPUTS
════════════════════════════════════════

The user's literal message is rarely the right argument for a tool's
``query`` / ``topic`` field. Each tool indexes differently and choking
it with a verbose sentence reliably returns zero results — that wastes
the call and produces a thin answer. For every step you plan:

1. **Extract a concrete topic phrase.** Tools like ``research_trends``,
   ``literature_survey``, ``papers_with_code``, ``github_search``,
   ``arxiv_search``, ``arxiv_import``, ``wikipedia`` are indexed by
   topic / method / library names — NOT by meta-questions about
   research. A user saying "what AI topics are feasible for a college
   project?" is a META-QUERY; do not forward it verbatim. Either pick a
   concrete topic from the conversation context, or use a discovery
   tool (``deep_search`` / ``frontier_scan``) with a short topical
   keyword phrase, never the meta-question itself.

2. **Compress for keyword-indexed tools.** ``github_search``,
   ``papers_with_code``, ``huggingface_search`` do lexical matching on
   a handful of words. Keep their queries to 3–8 informative terms —
   the method name, the library, the task. "find me research code
   implementations for retrieval-augmented generation with chunk size
   and top-k retrieval" → ``"retrieval augmented generation pytorch"``.

3. **Resolve referential follow-ups.** When the user says "research
   that topic", "proceed with your suggestion", "give me code for it",
   the topic is in the prior assistant turn — extract the specific
   concept (e.g. "retrieval-augmented generation") and use that as the
   tool query, never the literal follow-up phrase.

4. **Wikipedia / concept_explain** want the concept itself, not a
   question form. "Retrieval-augmented generation" not "What is
   retrieval-augmented generation and how does it work?".

5. **Don't fan out to side-channel tools speculatively.**
   ``github_search`` and ``papers_with_code`` should run only when the
   user asks for code / implementations / SOTA — they will not magically
   find papers and a 0-result call is wasted latency.

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
        disabled_features: set[str] | None = None,
        research_brief: str | None = None,
        intent_hint: str | None = None,
    ) -> Plan:
        """Async planner — calls the LLM with adaptive compute based on query complexity."""
        # Pass namespace_key + disabled features so any tool gated by a flag
        # the admin turned off (or that's overridden off for this user) is
        # invisible to the planner. Without this the LLM would happily pick
        # ``graph_query`` and get 404s downstream.
        catalogue = describe_for_planner(
            namespace_key=namespace_key,
            disabled_features=disabled_features,
        )
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
                research_brief=research_brief,
                intent_hint=intent_hint,
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
        research_brief: str | None = None,
        intent_hint: str | None = None,
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
        short_mem = mem.get("short") or {}
        medium_mem = mem.get("medium") or {}
        long_mem = mem.get("long") or {}

        def _mem_val(v: object) -> str:
            if isinstance(v, dict):
                return str(v.get("value", v))
            return str(v)

        def _entry_type(v: object) -> str:
            if isinstance(v, dict):
                return str(v.get("type", "context"))
            return "context"

        # Conditional memory injection. Forced injection on every turn bloats
        # the prompt and risks contaminating answers with stale data, so we
        # only inject when at least ONE of these signals fires:
        #   1. Any stored ``preference`` — universally useful for tone/depth.
        #   2. Continuation cues in the user query ("the same", "as before",
        #      "we discussed", "my", "again", "earlier", pronouns).
        #   3. The conversation has 3+ prior turns (enough accumulated
        #      context that the planner could actually need the memory).
        def _has_preference(d: dict) -> bool:
            return any(_entry_type(v) == "preference" for v in d.values())

        has_pref = _has_preference(short_mem) or _has_preference(medium_mem) or _has_preference(long_mem)
        q_low = (query or "").lower()
        continuation_cues = (
            "the same", "as before", "as we discussed", "you mentioned", "earlier",
            "previously", "last time", "my preference", "my background", "again",
            " it ", " that ", " those ", " these ", " they ",
        )
        has_cue = any(c in q_low for c in continuation_cues)
        prior_turns = sum(1 for m in (history or []) if (m.get("role") == "assistant"))
        inject_memory = bool(has_pref or has_cue or prior_turns >= 3)

        memory_blob = ""
        if inject_memory and (short_mem or medium_mem or long_mem):
            memory_blob = (
                "\nResearch memory (advisory — let it shape the plan only when it "
                "clearly applies; ignore if stale or off-topic):\n"
            )
            # Preferences always first regardless of tier.
            for tier_label, tier_dict in (("chat", short_mem), ("tree", medium_mem), ("namespace", long_mem)):
                prefs = [(k, v) for k, v in tier_dict.items() if _entry_type(v) == "preference"]
                for k, v in prefs[:4]:
                    memory_blob += f"  [pref/{tier_label}] {k}: {_mem_val(v)[:200]}\n"
            for tier_label, tier_dict in (("chat", short_mem), ("tree", medium_mem), ("namespace", long_mem)):
                others = [(k, v) for k, v in tier_dict.items() if _entry_type(v) != "preference"]
                for k, v in others[:5]:
                    memory_blob += f"  [{tier_label}] {k}: {_mem_val(v)[:200]}\n"

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
        brief_blob = ""
        if research_brief:
            brief_blob = (
                "\nResearch brief (pre-planning intent — sharpen tool choice "
                "around this, do not contradict it):\n"
                + research_brief + "\n"
            )
        intent_blob = ""
        if intent_hint:
            intent_blob = (
                "\nInferred working intent (advisory — let it sharpen tool "
                "selection only when it clearly applies):\n"
                + intent_hint + "\n"
            )
        return (
            f"User request: {query}\n"
            f"Query complexity: {complexity} — {depth_hint}\n"
            f"Active namespace: {namespace_key}\n"
            f"Topic scope: {', '.join(namespace_keys) or namespace_key}\n"
            f"User profile (soft bias): expertise={expertise}, orientation={orientation}\n"
            f"{pack_blob}"
            f"{intent_blob}"
            f"{brief_blob}"
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
