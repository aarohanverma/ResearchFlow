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


# Cap on how many procedural memories we splice into the system
# prompt per turn — keeps the static planner prompt small enough
# that the soft-nudge framing still dominates and a runaway count of
# user procedures can't shadow the platform's invariants.
_MAX_PROCEDURAL_INJECT = 6
# Memory types that count as procedural for the hot-path inject.
# Anything else stays in the user-prompt memory block, which is the
# right channel for facts/episodes/context.
_PROCEDURAL_TYPES: frozenset[str] = frozenset({"skill", "procedure"})


def _render_procedural_block(memory: dict[str, Any]) -> str:
    """Return a soft procedural-memory section to append to the
    planner system prompt, or the empty string when no procedural
    entries are stored.

    Pulls from the medium (tree) and long (namespace) tiers — short
    tier is per-chat context and rarely carries procedural intent.
    The block is framed as "soft preferences" so the planner treats
    them as guidance, not as overrides for the platform's hard
    invariants (HITL gates, graph-build ban, etc.).

    Each entry is rendered on its own line as ``- [type] key: value``
    so the model can attribute the procedure to a specific stored
    instruction. Values truncated at 240 chars so a runaway user
    write can't bloat the prompt.
    """
    if not memory:
        return ""

    procedurals: list[tuple[str, str, str, str]] = []  # (tier, key, ptype, value)
    for tier_key in ("medium", "long"):
        tier_view = memory.get(tier_key) or {}
        if not isinstance(tier_view, dict):
            continue
        for key, entry in tier_view.items():
            if not isinstance(entry, dict):
                continue
            etype = str(entry.get("type") or "").lower()
            if etype not in _PROCEDURAL_TYPES:
                continue
            value = str(entry.get("value") or "").strip()
            if not value:
                continue
            procedurals.append((tier_key, str(key), etype, value[:240]))
            if len(procedurals) >= _MAX_PROCEDURAL_INJECT:
                break
        if len(procedurals) >= _MAX_PROCEDURAL_INJECT:
            break

    if not procedurals:
        return ""

    lines = [
        "",
        "",
        "════════════════════════════════════════",
        "USER PROCEDURAL MEMORY (soft preferences)",
        "════════════════════════════════════════",
        "",
        "These are durable behavioural preferences the user has saved. "
        "Honour them when they fit; they are guidance, not commands. The "
        "platform invariants above (HITL gates, tool catalogue, "
        "minimum-sufficient-set discipline) always win when they conflict.",
        "",
    ]
    for tier, key, etype, value in procedurals:
        lines.append(f"- [{tier}/{etype}] {key}: {value}")
    return "\n".join(lines)


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

• Full-paper analysis tools — when you need to GO DEEP on a specific paper:
    - ``paper_qa`` for ONE focused question against the paper body.
    - ``deep_paper_analysis`` for a comprehensive multi-aspect read
      (methods + results + limitations + ablations in one call). Pick
      this when the user asks to "go deeper", "give me a thorough
      read", "what are the limitations", "what ablations did they run",
      OR when you need full-paper grounding instead of abstract-only.
      Requires the paper to be in the corpus — call arxiv_import /
      deep_search first if needed. More expensive than paper_qa
      (4× LLM rounds) but the right tool when the user wants depth.

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
BROAD "TEACH + DIRECT" QUERIES — STAY ON THE MAIN TOPIC
════════════════════════════════════════

When the user asks something broad like "teach me X, then propose
research directions" or "explain Y deeply and suggest where to take
it next", the right plan is FOCUSED, not exhaustive:

  1. Ground the CANONICAL FOUNDATION first — the core mechanism, the
     standard formulation, the most-cited reference. Use
     ``concept_explain`` and ``literature_survey`` on the main
     topic; do NOT chase tangential subfields yet.
  2. Build a concept map: core mechanism → variants → limitations
     → applications → open questions. Each branch grounded in
     1-2 strong papers, not 10 scattered ones.
  3. Verify load-bearing claims with full-paper checks
     (``paper_qa`` / ``deep_paper_analysis``) on the papers the
     final answer relies on. Don't waste verification budget on
     side-citations.
  4. Rank research directions by clarity, impact, feasibility,
     novelty, and falsifiability — proposing one well-scoped
     direction beats listing six speculative ones.
  5. ONLY THEN, if a contradiction or adjacent thread MATERIALLY
     affects the main thesis, follow it. Contradiction-hunting is
     valuable when it changes the answer, not when it ornaments it.

Anti-pattern to avoid: chasing every contradiction signal the loop
surfaces. The contradiction middleware should NOT pull retrieval off
the main topic to investigate a side claim that doesn't change the
recommendation. The user's central topic stays the centre of the
plan; adjacent threads are explored only when they alter the
conclusion.

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
PARAMS HYGIENE — NEVER EMIT PLACEHOLDERS
════════════════════════════════════════

Every tool in the catalogue has a JSON ``input_schema`` with a ``required``
list. Each step's ``params`` block MUST set every required field with a
real concrete value drawn from the user query, the brief, or a prior step's
expected output. Forbidden patterns:

  ✗ ``"query": ""``                — empty strings count as missing
  ✗ ``"query": "__to_fill__"``     — underscore-bounded placeholders
  ✗ ``"paper_id": "<TODO>"``       — angle-bracket placeholders
  ✗ ``"paper_ids": ["{fill}"]``    — brace placeholders inside arrays
  ✗ ``"paper_ids": []`` when the field is required with min_items ≥ 1
  ✗ Cross-step reference templates — ``"$STEP2.paper_ids[0]"``,
    ``"{{step_1.output}}"``, ``"STEP3.paper_id"``. There is NO
    substitution layer downstream; these reach the tool as literal
    strings and fail. Either (a) leave ``paper_id``/``paper_ids``
    empty on the dependent step so the orchestrator's ledger fills
    them from the prior step's actual retrieval results, or (b)
    plan the dependent step in a later turn after the IDs are known.
  ✗ Inventing paper UUIDs / DOIs / IDs — only use IDs that come back
    from a retrieval step you planned earlier in this same plan.

If a step needs a paper id and you don't yet have one, plan the retrieval
step FIRST and accept that the orchestrator will pass the IDs forward; do
NOT emit placeholder paper IDs hoping something fills them in. For
``compare_papers`` / ``paper_qa`` / ``genie_synthesize`` in the same plan
as the retrieval, set ``parallel: false`` so the retrieval lands first.

════════════════════════════════════════
GENIE SYNTHESIS FLOW
════════════════════════════════════════

When the HITL policy below permits genie_synthesize, the correct chain is:

  retrieval (deep_search / arxiv_import / literature_survey)
    → genie_synthesize  (creates a NEW capsule from those papers)
    → (optional) genie_deep_dive on the capsule_id returned by step 2

NEVER plan ``genie_read`` for a synthesis-style request. ``genie_read``
only lists capsules that were created in PRIOR turns; if the user asked
to combine / synthesize / brainstorm using THIS turn's papers, reading
old capsules produces an off-topic answer. Use ``genie_read`` only for
explicit "show me the ideas I generated earlier" requests.

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
        session_id: Any = None,
    ) -> Plan:
        """Async planner — calls the LLM with adaptive compute based on query complexity.

        Memory injection is **query-aware**: before building the prompt,
        ``memory_injector.select_relevant_memory`` collapses each tier
        to its preferences plus the top-K entries semantically closest
        to ``query``. This is the middleware layer described in the
        LangChain memory-pattern guide — the planner prompt receives a
        compact, high-signal slice instead of the full memory dump.

        Behavioural guarantees:
          * If the embedder is unavailable, ``select_relevant_memory``
            falls back to a recency-ordered slice of the same shape;
            the planner never sees a missing-memory failure.
          * If ``session_id`` is omitted, ranking still happens but
            the per-entry embedding cache cannot be persisted, so
            successive turns re-embed the same entries. Acceptable for
            tests; the production orchestrator always passes session_id.
          * Buckets already small enough (≤ per_tier_k non-preference
            entries) are passed through untouched — no semantic call
            is made, no extra latency incurred.
        """
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

        # ── Memory injection middleware ───────────────────────────────
        # Replace the flat per-tier dump with a query-aware focused
        # slice before any prompt building. Preferences are preserved
        # in full; non-preference entries are ranked by semantic
        # similarity to the user query (with graceful recency
        # fallback). This addresses the "automatic, compact, high-
        # signal memory injection" middleware pattern documented in
        # the LangChain agentic harness guide — the explicit
        # memory_recall / memory_write / memory_delete tools remain
        # the agent-controlled path; this is the automatic complement.
        focused_memory: dict | None = memory
        if memory:
            try:
                from app.assistant.memory_injector import select_relevant_memory
                focused_memory = await select_relevant_memory(
                    query=query,
                    memory_view=memory,
                    session_id=session_id,
                )
            except Exception as _mi_exc:  # noqa: BLE001 — never block planning
                log.debug("memory_injector skipped: %s", _mi_exc)
                focused_memory = memory

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
                memory=focused_memory or {},
                complexity=complexity,
                research_brief=research_brief,
                intent_hint=intent_hint,
            )
            # Soft procedural-memory injection. Procedural entries
            # (``skill`` / ``procedure`` types) describe HOW the agent
            # should behave — they're instructions, not facts. The
            # natural channel for those is the system prompt, so the
            # planner is shaped by them throughout the decision rather
            # than reading them as an aside. The block is appended
            # AFTER the static prompt so any user-defined procedure
            # can refine (but never replace) the platform's invariants.
            # Empty-input case is a no-op; the static prompt is used
            # unchanged.
            system_prompt = _PLANNER_SYSTEM + _render_procedural_block(memory or {})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            # Use the REASONING model for planning per user spec:
            # "Use a strong model for planning as well as evaluation."
            # The trade-off (extra latency) is acceptable because the
            # planner is the single most-leveraged LLM call in a turn
            # — every downstream tool depends on its choices. A
            # smarter planner saves more tool round-trips than the
            # extra planning latency costs.
            plan_model = llm.reasoning_model
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
                "clearly applies; ignore if stale or off-topic). "
                "Entries are tagged ``[tier/type]`` so you can weight them: "
                "``preference`` and ``skill``/``procedure`` describe HOW to behave, "
                "``finding``/``concept``/``paper_note`` are facts about the world, "
                "``episode`` is a record of a past interaction.\n"
            )
            # Preferences always first regardless of tier — they're the
            # single most load-bearing kind of memory for plan shaping.
            for tier_label, tier_dict in (("chat", short_mem), ("tree", medium_mem), ("namespace", long_mem)):
                prefs = [(k, v) for k, v in tier_dict.items() if _entry_type(v) == "preference"]
                for k, v in prefs[:4]:
                    memory_blob += f"  [{tier_label}/preference] {k}: {_mem_val(v)[:200]}\n"
            # Then everything else — surface the entry type explicitly
            # so the planner can tell an episode (past event) from a
            # finding (durable fact) and weight them differently.
            for tier_label, tier_dict in (("chat", short_mem), ("tree", medium_mem), ("namespace", long_mem)):
                others = [(k, v) for k, v in tier_dict.items() if _entry_type(v) != "preference"]
                for k, v in others[:5]:
                    etype = _entry_type(v) or "context"
                    memory_blob += f"  [{tier_label}/{etype}] {k}: {_mem_val(v)[:200]}\n"

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
        # Adaptive strategy hint — query-shape classification surfaced
        # so the planner picks the right tool order, retrieval depth,
        # and rerank intensity per-query instead of running the same
        # pipeline regardless of intent. The block is advisory; the
        # planner can deviate when conversation context argues for it.
        try:
            from app.assistant.query_strategy import classify_query
            strategy = classify_query(query, history=history or [])
            strategy_blob = (
                "\nQuery-shape strategy hint (advisory — adjust the plan to "
                "match the shape unless the conversation context overrides it):\n"
                + "  " + strategy.render_for_prompt() + "\n"
            )
        except Exception:
            strategy_blob = ""
        return (
            f"User request: {query}\n"
            f"Query complexity: {complexity} — {depth_hint}\n"
            f"Active namespace: {namespace_key}\n"
            f"Topic scope: {', '.join(namespace_keys) or namespace_key}\n"
            f"User profile (soft bias): expertise={expertise}, orientation={orientation}\n"
            f"{pack_blob}"
            f"{intent_blob}"
            f"{strategy_blob}"
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
            # Preflight repair: strip placeholders + fill any required
            # text-like field that the planner left empty. Mirrors the
            # ReAct loop's hygiene so a plan step that emits
            # ``{"question": "<TODO>"}`` is repaired with the user query
            # instead of dropped silently.
            try:
                from app.assistant.react_loop import (
                    _preflight_and_repair_params,
                    PaperLedger,
                )
                _schema = tool.input_schema.model_json_schema()
                params, _notes = _preflight_and_repair_params(
                    tool_name, params, _schema,
                    query=query, ledger=PaperLedger(),
                )
                if _notes:
                    log.info(
                        "planner: auto-repaired params for tool=%s: %s",
                        tool_name, "; ".join(_notes)[:300],
                    )
            except Exception:
                pass
            try:
                tool.input_schema(**params)
            except Exception as ve:
                # One-shot retry with fully-derived params (no original
                # planner output) — same pattern the loop uses on
                # validation failure.
                try:
                    from app.assistant.react_loop import (
                        _preflight_and_repair_params,
                        PaperLedger,
                    )
                    _schema = tool.input_schema.model_json_schema()
                    fresh_params = {
                        "namespace_key": namespace_key,
                        "namespace_keys": ns_keys,
                    }
                    fresh_params, _ = _preflight_and_repair_params(
                        tool_name, fresh_params, _schema,
                        query=query, ledger=PaperLedger(),
                    )
                    tool.input_schema(**fresh_params)
                    params = fresh_params
                    log.info("planner: recovered tool=%s after validation error", tool_name)
                except Exception:
                    log.warning("planner produced invalid params for tool=%s: %s", tool_name, ve)
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
