"""Research-intent inference — advisory soft signals for downstream routing.

The Research Assistant supports an open set of user goals: learning,
exploration, paper comparison, novelty checks, literature surveys,
synthesis, SOTA tracking, production translation, due-diligence,
publication-oriented end-to-end projects, and many more we cannot
enumerate. Treating that landscape with a fixed intent enum + hard
intent→tool table would force every conversation into one of N rigid
moulds — exactly what we want to avoid.

Instead, this module asks a cheap model to *describe* the user's
working intent in free-form, then returns a small set of advisory
signals that the orchestrator uses as SOFT NUDGES, never as drivers:

  * ``complexity`` — one of ``trivial / single / deep`` (advisory).
  * ``capabilities`` — short list of capability families that would
    likely help (e.g. ``["paper_retrieval", "comparison", "novelty_check"]``).
  * ``user_posture`` — a one-phrase read on who's asking (e.g. "expert
    refining a hypothesis", "novice probing fundamentals").
  * ``response_voice`` / ``depth`` / ``structure`` — advisory shape hints.
  * ``needs_clarification`` + ``clarification_question`` — set only when
    the intent is genuinely ambiguous and asking one short question
    would save several turns of wasted work.

Every field is *advisory*. The orchestrator may ignore any of them.
The classifier itself never fails the turn: on any error, it returns
``None`` and the orchestrator continues with the existing heuristic
behaviour.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ── Soft schema. The labels below are EXAMPLES surfaced to the LLM as
# inspiration, never enforced as an enum. The classifier can return
# something not in the list when the user's intent doesn't fit. ─────────────

# These are reference labels the model can use when one fits cleanly.
# They are deliberately broad and overlap is fine.
_REFERENCE_INTENT_LABELS = [
    "quick_factual_lookup",
    "concept_explanation",
    "learn_fundamentals",
    "literature_survey",
    "state_of_the_art_scan",
    "novelty_or_prior_art_check",
    "paper_comparison",
    "paper_summary",
    "synthesis_or_new_idea",
    "research_gap_identification",
    "hypothesis_design",
    "experiment_design",
    "production_translation",
    "research_to_product_strategy",
    "field_monitoring",
    "due_diligence_or_market_scan",
    "end_to_end_research_program",
    "methodology_coaching",
    "casual_conversation",
    "navigation_or_command",
]

_REFERENCE_CAPABILITIES = [
    "paper_retrieval",
    "semantic_search",
    "corpus_widening",
    "concept_explanation",
    "comparison",
    "novelty_check",
    "trend_analysis",
    "graph_neighbourhood",
    "synthesis",
    "implementation_lookup",
    "data_lookup",
    "code_or_model_lookup",
    "external_news",
    "active_context_use",
    "memory_recall",
    "verification",
]


class ResearchIntent(BaseModel):
    """Advisory inference output for one turn."""

    label: str = Field(
        ...,
        description=(
            "Short snake_case label naming the inferred working intent. "
            "Pick a reference label when one fits cleanly; otherwise invent "
            "a fresh label that fits the user better."
        ),
    )
    description: str = Field(
        ...,
        description="One-sentence description of what the user actually wants this turn.",
    )
    complexity: str = Field(
        ...,
        description=(
            "Routing hint: 'trivial' (fast path, no tools needed or one cheap "
            "tool), 'single' (one focused plan-execute-synth cycle), 'deep' "
            "(full reflection + critique + red-team + gap re-query)."
        ),
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 5 capability families the answer would likely need. Reference "
            "list provided; not exhaustive — invent labels if needed."
        ),
    )
    user_posture: str = Field(
        default="",
        description=(
            "One-phrase read on the user's posture for this turn — e.g. "
            "'expert refining a hypothesis', 'novice probing fundamentals', "
            "'VC scanning commercial implications'. Used as a soft persona hint."
        ),
    )
    response_voice: str = Field(
        default="",
        description="Voice hint: 'terse', 'conversational', 'scholarly', 'didactic'.",
    )
    response_depth: str = Field(
        default="",
        description="Depth hint: 'shallow', 'moderate', 'thorough', 'exhaustive'.",
    )
    response_structure: str = Field(
        default="",
        description=(
            "Structure hint: 'prose', 'bulleted', 'sections', 'comparison_table', "
            "'numbered_steps', 'concept_ladder'. Synthesizer treats as soft "
            "nudge — it can deviate if the content benefits."
        ),
    )
    needs_clarification: bool = Field(
        default=False,
        description=(
            "Set true ONLY when the intent is genuinely ambiguous and ONE "
            "short clarifying question would save the user multiple wasted "
            "turns. Default is false — bias toward proceeding."
        ),
    )
    clarification_question: str | None = Field(
        default=None,
        description="The exact one-line clarifying question to ask. Required iff needs_clarification.",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="How confident the classifier is in its read (0..1).",
    )
    rationale: str = Field(
        default="",
        description="One short sentence on why this intent was chosen.",
    )


_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label":                  {"type": "string", "maxLength": 60},
        "description":            {"type": "string", "maxLength": 280},
        "complexity":             {"type": "string", "enum": ["trivial", "single", "deep"]},
        "capabilities":           {"type": "array",  "items": {"type": "string", "maxLength": 40}, "maxItems": 5},
        "user_posture":           {"type": "string", "maxLength": 160},
        "response_voice":         {"type": "string", "maxLength": 32},
        "response_depth":         {"type": "string", "maxLength": 32},
        "response_structure":     {"type": "string", "maxLength": 64},
        "needs_clarification":    {"type": "boolean"},
        "clarification_question": {"type": ["string", "null"], "maxLength": 240},
        "confidence":             {"type": "number", "minimum": 0, "maximum": 1},
        "rationale":              {"type": "string", "maxLength": 240},
    },
    "required": ["label", "description", "complexity"],
}


def _heuristic_intent(query: str, history: list[dict] | None) -> ResearchIntent:
    """Pure-deterministic fallback. Never raises.

    Used when the LLM call fails or the user is offline. Returns a
    minimum-viable intent so downstream routing still has something to
    work with. Heuristic-only — never appears in normal operation.
    """
    q = (query or "").strip()
    q_lower = q.lower()
    n_prior_assistant = sum(1 for m in (history or []) if m.get("role") == "assistant")
    # Greetings / acknowledgments → trivial.
    if len(q) < 24 and any(
        q_lower.startswith(p) for p in (
            "hi", "hey", "hello", "thanks", "thank", "cool", "ok", "okay", "got it", "great",
        )
    ):
        return ResearchIntent(
            label="casual_conversation",
            description="Short conversational exchange, no research depth needed.",
            complexity="trivial",
            confidence=0.6,
            rationale="Short greeting / acknowledgment pattern.",
        )
    # Very short questions → likely simple lookup or follow-up.
    if len(q) < 60 and "?" in q and n_prior_assistant <= 1:
        return ResearchIntent(
            label="quick_factual_lookup",
            description="Brief factual question; quick targeted answer suffices.",
            complexity="trivial" if len(q) < 40 else "single",
            capabilities=["concept_explanation"] if "what is" in q_lower or "define" in q_lower else [],
            confidence=0.55,
            rationale="Short question without depth markers.",
        )
    # Deep-research markers anywhere → deep.
    deep_markers = (
        "survey", "literature review", "compare", "comparison", "synthesize",
        "synthesis", "novelty", "state of the art", "sota", "comprehensive",
        "deep dive", "end-to-end", "research project", "publish", "publication",
        "research gap", "open problem", "prior art",
    )
    if any(m in q_lower for m in deep_markers):
        return ResearchIntent(
            label="literature_survey",
            description="Substantive research request — broader retrieval + synthesis warranted.",
            complexity="deep",
            capabilities=["paper_retrieval", "comparison", "synthesis", "trend_analysis"],
            confidence=0.55,
            rationale="Heuristic match on deep-research marker phrases.",
        )
    # Default to single-cycle for everything else.
    return ResearchIntent(
        label="research_assistance",
        description="Substantive but bounded research query; one focused cycle should suffice.",
        complexity="single",
        capabilities=["paper_retrieval"],
        confidence=0.5,
        rationale="No strong signal in either direction — defaulting to single-cycle.",
    )


async def infer_intent(
    *,
    query: str,
    history: list[dict] | None,
    memory: dict | None,
    namespace_key: str,
    user_expertise: str = "practitioner",
    user_orientation: str = "both",
) -> ResearchIntent:
    """Return an advisory intent read. Never raises.

    Uses the cheap model with structured output. Falls back to a
    deterministic heuristic on any LLM failure or schema mismatch so the
    orchestrator can rely on always getting an intent — even when the
    LLM provider is offline.
    """
    try:
        from app.adapters.llm import get_llm_adapter

        llm = get_llm_adapter()

        # Compact recent-history blob — bounded but generous so the model
        # has enough trajectory to infer intent shifts mid-conversation.
        recent = (history or [])[-10:]
        history_lines: list[str] = []
        for m in recent:
            role = (m.get("role") or "user").upper()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            history_lines.append(f"{role}: {content[:1200]}")
        history_blob = "\n\n".join(history_lines) or "(no prior turns)"

        # Concise memory snapshot — only labels + values, capped, so the
        # classifier has context without ballooning the prompt.
        mem_lines: list[str] = []
        if isinstance(memory, dict):
            for tier in ("short", "medium", "long"):
                d = memory.get(tier) or {}
                for k, v in list(d.items())[:4]:
                    if isinstance(v, dict):
                        t = v.get("type") or "context"
                        val = str(v.get("value") or "")
                    else:
                        t = "context"
                        val = str(v)
                    if val:
                        mem_lines.append(f"[{tier}/{t}] {k}: {val[:160]}")
        mem_blob = "\n".join(mem_lines) or "(no memory)"

        system = (
            "You are the intent inferer for a research assistant.\n\n"
            "Your job: read the user's latest message in context and return "
            "ADVISORY signals about what the user is really trying to do this "
            "turn. The orchestrator uses these as soft nudges; downstream "
            "components stay LLM-driven and may ignore your output.\n\n"
            "CORE PRINCIPLES:\n"
            "  • Infer the real task — not just the literal query.\n"
            "  • Be generous about complexity: most turns are 'single'. "
            "    Use 'trivial' for greetings / acknowledgments / one-word "
            "    follow-ups, and 'deep' only when the user genuinely needs "
            "    broader retrieval + synthesis + critique (literature "
            "    survey, novelty check, SOTA scan, end-to-end research).\n"
            "  • Capability labels are non-exhaustive — invent when needed.\n"
            "  • Set needs_clarification=true ONLY when one short question "
            "    would save the user multiple wasted turns; default false.\n"
            "  • Response voice/depth/structure are HINTS, not templates.\n"
            "  • If the user posture is unclear, leave it empty rather than "
            "    guessing.\n"
            "  • Return strict JSON matching the schema.\n\n"
            f"Reference intent labels (use one when it fits, otherwise invent):\n"
            f"  {', '.join(_REFERENCE_INTENT_LABELS)}\n\n"
            f"Reference capability labels (non-exhaustive):\n"
            f"  {', '.join(_REFERENCE_CAPABILITIES)}"
        )
        user_msg = (
            f"Namespace: {namespace_key}\n"
            f"User profile (soft bias): expertise={user_expertise}, "
            f"orientation={user_orientation}\n\n"
            f"LATEST USER MESSAGE:\n{query}\n\n"
            f"RECENT CONVERSATION (most recent last):\n{history_blob}\n\n"
            f"STORED MEMORY (advisory):\n{mem_blob}"
        )

        raw = await llm.complete_structured(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            llm.cheap_model,
            _INTENT_SCHEMA,
        )
        if not isinstance(raw, dict):
            raise ValueError("intent classifier returned non-dict")
        intent = ResearchIntent(**raw)
        # Normalise — strip stray whitespace.
        intent.label = re.sub(r"\s+", "_", intent.label.strip().lower())[:60]
        if intent.complexity not in {"trivial", "single", "deep"}:
            intent.complexity = "single"
        if intent.needs_clarification and not (intent.clarification_question or "").strip():
            # Guard against the model setting the flag without a question.
            intent.needs_clarification = False
            intent.clarification_question = None
        log.debug(
            "intent inferred: label=%s complexity=%s conf=%.2f capabilities=%s",
            intent.label, intent.complexity, intent.confidence, intent.capabilities,
        )
        return intent
    except Exception as exc:
        log.debug("intent inference fell back to heuristic: %s", exc)
        return _heuristic_intent(query, history)


def render_intent_hint(intent: ResearchIntent | None) -> str:
    """Render the intent as a short advisory block for prompts.

    Empty string when nothing useful to surface. Always optional — the
    consuming prompt should treat the block as soft guidance only.
    """
    if intent is None:
        return ""
    parts: list[str] = ["Inferred working intent (advisory):"]
    parts.append(f"  - label: {intent.label}")
    if intent.description:
        parts.append(f"  - description: {intent.description}")
    parts.append(f"  - complexity hint: {intent.complexity}")
    if intent.capabilities:
        parts.append(f"  - useful capabilities: {', '.join(intent.capabilities[:5])}")
    if intent.user_posture:
        parts.append(f"  - user posture: {intent.user_posture}")
    if intent.response_voice or intent.response_depth or intent.response_structure:
        bits = []
        if intent.response_voice:     bits.append(f"voice={intent.response_voice}")
        if intent.response_depth:     bits.append(f"depth={intent.response_depth}")
        if intent.response_structure: bits.append(f"structure={intent.response_structure}")
        parts.append(f"  - response shape: {' / '.join(bits)}")
    return "\n".join(parts)
