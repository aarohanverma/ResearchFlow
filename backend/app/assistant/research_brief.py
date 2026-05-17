"""Research Brief Agent — pre-planning intent crystallisation.

A short, structured digest produced before the main planner runs so the
planner sees a precise research target instead of a raw chat message.
Modelled on the "Research Brief Agent" pattern from agentic-research
frameworks: it consolidates

  * the user's current intent,
  * the trajectory of the conversation (history + cached rolling summary),
  * relevant memory across all tiers, and
  * any outstanding open questions

into a 5-field JSON brief the planner consumes verbatim. The brief is
SOFT — it never overrides retrieval evidence — but it does sharpen the
planner's tool selection and parameter choices.

The agent only runs when it's worth the cheap-model call: trivial
greetings and one-shot lookups skip it entirely. For substantive or
ambiguous queries it produces a noticeable lift because the planner no
longer has to re-derive intent from scratch.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


_BRIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "research_question": {
            "type": "string",
            "description": (
                "ONE sentence stating exactly what the user wants answered "
                "right now. Sharper than the literal query — resolves "
                "pronouns and references using context."
            ),
        },
        "research_goal": {
            "type": "string",
            "description": (
                "ONE sentence on the broader goal this turn serves "
                "(why the user is asking)."
            ),
        },
        "must_cover": {
            "type": "array",
            "items": {"type": "string", "maxLength": 160},
            "description": (
                "Up to 5 concrete sub-topics, named methods, or specific "
                "papers the answer should cover for the user to feel "
                "satisfied. Empty array when there is nothing specific."
            ),
        },
        "constraints": {
            "type": "array",
            "items": {"type": "string", "maxLength": 160},
            "description": (
                "Up to 5 hard constraints — scope/exclusion rules, time "
                "windows, paper-source preferences, language/depth "
                "preferences. Empty array when none."
            ),
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 160},
            "description": (
                "Up to 3 unresolved threads from earlier turns that this "
                "answer could productively address even if not explicitly "
                "asked. Empty array when none."
            ),
        },
    },
    "required": ["research_question"],
}


def is_trivial_query(query: str) -> bool:
    """Skip the brief for trivial inputs to save a cheap-model call."""
    q = (query or "").strip()
    if len(q) < 24:
        return True
    lower = q.lower()
    for cue in ("hi", "hello", "thanks", "thank you", "ok", "okay", "got it"):
        if lower.startswith(cue) and len(q) < 60:
            return True
    return False


def format_for_planner(brief: dict[str, Any]) -> str:
    """Render the brief as a short, planner-friendly text block.

    Returns the empty string when there's nothing useful to surface.
    """
    if not brief:
        return ""
    parts: list[str] = []
    question = (brief.get("research_question") or "").strip()
    goal = (brief.get("research_goal") or "").strip()
    must_cover = [str(x).strip() for x in (brief.get("must_cover") or []) if str(x).strip()]
    constraints = [str(x).strip() for x in (brief.get("constraints") or []) if str(x).strip()]
    open_qs = [str(x).strip() for x in (brief.get("open_questions") or []) if str(x).strip()]
    if not (question or goal or must_cover or constraints or open_qs):
        return ""
    parts.append("Research brief (pre-planning):")
    if question:
        parts.append(f"  Question: {question[:400]}")
    if goal:
        parts.append(f"  Goal: {goal[:400]}")
    if must_cover:
        parts.append("  Must cover:")
        for item in must_cover[:5]:
            parts.append(f"    - {item[:240]}")
    if constraints:
        parts.append("  Constraints:")
        for item in constraints[:5]:
            parts.append(f"    - {item[:240]}")
    if open_qs:
        parts.append("  Open threads worth folding in if relevant:")
        for item in open_qs[:3]:
            parts.append(f"    - {item[:240]}")
    return "\n".join(parts)


async def compose_research_brief(
    *,
    user_query: str,
    namespace_key: str,
    history: list[dict],
    memory: dict | None,
    branch_seed_summary: str | None = None,
) -> dict[str, Any] | None:
    """Produce a structured brief from query + history + memory.

    Returns ``None`` on any failure (the caller falls back to the raw
    query). Single cheap-model structured-output call.
    """
    if is_trivial_query(user_query):
        return None
    try:
        from app.adapters.llm import get_llm_adapter
        llm = get_llm_adapter()

        # Recent history compacted into role: content lines — capped at the
        # last 14 messages so the brief prompt stays bounded but still
        # captures the trajectory the planner needs.
        history_lines: list[str] = []
        for m in (history or [])[-14:]:
            role = m.get("role") or "user"
            content = (m.get("content") or "").strip()
            if not content:
                continue
            # Generous per-message cap so multi-paragraph user messages
            # aren't truncated mid-claim. We rely on the cheap model's
            # context window rather than guessing here.
            history_lines.append(f"{role.upper()}: {content[:4000]}")
        history_blob = "\n\n".join(history_lines) or "(no prior turns)"

        mem_lines: list[str] = []
        if memory:
            for tier_label, tier_dict in (
                ("short", memory.get("short") or {}),
                ("medium", memory.get("medium") or {}),
                ("long", memory.get("long") or {}),
            ):
                items = list(tier_dict.items())[:6]
                if not items:
                    continue
                mem_lines.append(f"[{tier_label}]")
                for k, v in items:
                    if isinstance(v, dict):
                        cat = str(v.get("type") or "context")
                        val = str(v.get("value") or "")
                    else:
                        cat = "context"
                        val = str(v)
                    mem_lines.append(f"  - {cat}/{k}: {val[:200]}")
        mem_blob = "\n".join(mem_lines) or "(no memory)"

        seed_blob = f"\nBranch / parent context:\n{branch_seed_summary[:2400]}\n" if branch_seed_summary else ""

        system = (
            "You are a research-brief composer for a research assistant. "
            "Read the user's latest message, the conversation history, and "
            "the agent's stored memory, then crystallise the user's intent "
            "into a precise brief. Resolve pronouns and references using "
            "context. Be concrete: name papers, methods, datasets, time "
            "windows wherever the user implied them.\n\n"
            "Do NOT invent constraints the user did not state. Do NOT pad. "
            "Empty arrays are perfectly fine when the user hasn't expressed "
            "specifics. Return STRICT JSON only."
        )
        user_msg = (
            f"Namespace: {namespace_key}\n\n"
            f"USER LATEST MESSAGE:\n{user_query}\n\n"
            f"CONVERSATION HISTORY (most recent last):\n{history_blob}\n\n"
            f"MEMORY (advisory):\n{mem_blob}"
            f"{seed_blob}"
        )
        result = await llm.complete_structured(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            llm.cheap_model,
            _BRIEF_SCHEMA,
        )
        if not isinstance(result, dict):
            return None
        # Light sanitisation — drop empty strings, cap list lengths.
        result["must_cover"] = [str(x) for x in (result.get("must_cover") or []) if str(x).strip()][:5]
        result["constraints"] = [str(x) for x in (result.get("constraints") or []) if str(x).strip()][:5]
        result["open_questions"] = [str(x) for x in (result.get("open_questions") or []) if str(x).strip()][:3]
        return result
    except Exception as exc:
        log.debug("research brief composition failed: %s", exc)
        return None
