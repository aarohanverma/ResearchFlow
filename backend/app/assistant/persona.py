"""Persona-aware response shaping.

A SOFT advisory layer the synthesizer reads alongside the inferred
intent and the (memory + user profile + history) blob. Returns a
compact natural-language hint about voice, depth, structure, and lens
for this specific turn.

Hard rules we deliberately AVOID:
    * No fixed template per user type.
    * No fixed length cap.
    * No fixed structure (sections / bullets / prose) — the synthesizer
      may always deviate when the content benefits.

The function is best-effort: on any LLM failure it returns a small,
hand-rolled fallback derived from intent + expertise + orientation,
which is exactly the same shape the synthesizer already understood from
``_EXPERTISE_HINTS`` and ``_ORIENTATION_HINTS``.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# Schema kept very small — every field is an advisory string. None of
# these values are validated against a fixed vocabulary because forcing
# enums here would re-introduce the rigid templating we want to avoid.
_SHAPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "voice":              {"type": "string", "maxLength": 32},
        "depth":              {"type": "string", "maxLength": 32},
        "structure":          {"type": "string", "maxLength": 64},
        "lens":               {"type": "string", "maxLength": 64},
        "length_target":      {"type": "string", "maxLength": 32},
        "special_instructions": {"type": "string", "maxLength": 280},
    },
    "required": [],
}


_FALLBACK_DEPTH = {
    "newcomer":     "shallow-to-moderate; ground every term before using it",
    "practitioner": "moderate; assume working vocabulary",
    "expert":       "thorough; lead with non-obvious distinctions",
}

_FALLBACK_LENS = {
    "research":   "research lens — emphasise novelty, evidence quality, methodology, open problems",
    "production": "production lens — emphasise implementation, reliability, validation, deployment risk",
    "both":       "balanced lens — research substance with practical implications",
}


def _fallback_shape(*, user_expertise: str, user_orientation: str) -> dict[str, str]:
    """Conservative shape used when the LLM call is unavailable."""
    return {
        "voice": "conversational; precise; never preachy",
        "depth": _FALLBACK_DEPTH.get(user_expertise, _FALLBACK_DEPTH["practitioner"]),
        "structure": "adaptive — pick what fits this content",
        "lens": _FALLBACK_LENS.get(user_orientation, _FALLBACK_LENS["both"]),
        "length_target": "as long as the content needs, no longer",
        "special_instructions": "",
    }


async def compose_response_shape(
    *,
    query: str,
    intent: Any | None,
    history: list[dict] | None,
    user_expertise: str,
    user_orientation: str,
    memory: dict | None,
) -> dict[str, str]:
    """Produce a small advisory dict the synthesizer can splice into its prompt.

    Always returns a dict with five string fields — never raises.
    The dict is consumed as guidance only; the synthesizer is free to
    deviate when the content benefits.
    """
    fallback = _fallback_shape(
        user_expertise=user_expertise,
        user_orientation=user_orientation,
    )
    try:
        from app.adapters.llm import get_llm_adapter

        llm = get_llm_adapter()

        intent_blob = ""
        if intent is not None:
            try:
                intent_blob = (
                    f"label={getattr(intent, 'label', '')!r} "
                    f"complexity={getattr(intent, 'complexity', '')!r} "
                    f"posture={getattr(intent, 'user_posture', '')!r} "
                    f"voice_hint={getattr(intent, 'response_voice', '')!r} "
                    f"depth_hint={getattr(intent, 'response_depth', '')!r} "
                    f"structure_hint={getattr(intent, 'response_structure', '')!r}"
                )
            except Exception:
                intent_blob = ""

        # Surface only durable preferences from memory — the rest is too
        # noisy for a shaping decision and gets handled by the synthesizer's
        # own memory hint.
        pref_lines: list[str] = []
        if isinstance(memory, dict):
            for tier in ("short", "medium", "long"):
                for k, v in (memory.get(tier) or {}).items():
                    if isinstance(v, dict) and v.get("type") == "preference":
                        val = str(v.get("value") or "")
                        if val:
                            pref_lines.append(f"[{tier}] {k}: {val[:200]}")
        pref_blob = "\n".join(pref_lines[:6]) or "(none)"

        # Compact history — just the last few turns so the model can tell
        # whether the user is mid-thought, mid-implementation, etc.
        last_turns: list[str] = []
        for m in (history or [])[-4:]:
            content = (m.get("content") or "").strip()
            role = (m.get("role") or "user").upper()
            if content:
                last_turns.append(f"{role}: {content[:600]}")
        history_blob = "\n\n".join(last_turns) or "(no prior turns)"

        system = (
            "You shape the response style for ONE turn of a research "
            "assistant. Return SHORT advisory hints — no templates, no "
            "fixed structures. The synthesizer will deviate when the "
            "content benefits, so favour natural prose hints over rules.\n\n"
            "Style fields:\n"
            "  voice            — short adjective phrase\n"
            "  depth            — how deep to go\n"
            "  structure        — soft hint, never a hard template\n"
            "  lens             — research / production / strategic / pedagogical / etc.\n"
            "  length_target    — natural-language ('short and direct',\n"
            "                     'as long as needed', 'one focused paragraph')\n"
            "  special_instructions — one optional line of bespoke guidance\n\n"
            "Be terse. Empty fields are fine. Never invent personas the user "
            "didn't actually project. Return strict JSON matching the schema."
        )
        user_msg = (
            f"USER PROFILE: expertise={user_expertise}, orientation={user_orientation}\n\n"
            f"INTENT SIGNALS: {intent_blob or '(none)'}\n\n"
            f"DURABLE PREFERENCES:\n{pref_blob}\n\n"
            f"RECENT HISTORY:\n{history_blob}\n\n"
            f"CURRENT QUERY:\n{query}"
        )
        raw = await llm.complete_structured(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            llm.cheap_model,
            _SHAPE_SCHEMA,
        )
        if not isinstance(raw, dict):
            return fallback
        out = {k: str(raw.get(k) or fallback.get(k, "")).strip() for k in (
            "voice", "depth", "structure", "lens", "length_target",
            "special_instructions",
        )}
        # Backfill any empty field from the fallback so the synthesizer
        # always has a complete shape to work with.
        for k, default_val in fallback.items():
            if not out.get(k):
                out[k] = default_val
        return out
    except Exception as exc:
        log.debug("response shape composer fell back to defaults: %s", exc)
        return fallback


def render_shape_hint(shape: dict[str, str] | None) -> str:
    """Format the shape dict as a soft advisory block for the synthesizer."""
    if not shape:
        return ""
    parts: list[str] = ["Response shape (advisory — adapt freely when content benefits):"]
    if shape.get("voice"):
        parts.append(f"  - voice: {shape['voice']}")
    if shape.get("depth"):
        parts.append(f"  - depth: {shape['depth']}")
    if shape.get("structure"):
        parts.append(f"  - structure hint: {shape['structure']}")
    if shape.get("lens"):
        parts.append(f"  - lens: {shape['lens']}")
    if shape.get("length_target"):
        parts.append(f"  - length: {shape['length_target']}")
    if shape.get("special_instructions"):
        parts.append(f"  - note: {shape['special_instructions']}")
    return "\n".join(parts) if len(parts) > 1 else ""
