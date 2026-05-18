"""Active clarification gate.

When the intent inferer reports both ``needs_clarification=True`` and a
concrete short question, the orchestrator can short-circuit the heavy
plan-execute-synth pipeline and return the question to the user
immediately. This module owns the *threshold logic* — deciding whether
to actually ask, given:

  * the intent's confidence score,
  * how much prior conversation has accumulated,
  * whether the query is itself ambiguous (very short, pronoun-heavy),
  * whether nearby memory might already disambiguate.

The rule is biased TOWARD proceeding: asking a clarifying question is
useful only when one short question would genuinely save the user
multiple wasted turns. Most turns should answer first.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


_PRONOUN_LOAD = (
    " it ", " that ", " those ", " these ", " they ", " them ", " this ",
)


def _is_pronoun_heavy(query: str) -> bool:
    if not query:
        return False
    q = " " + query.lower() + " "
    hits = sum(1 for p in _PRONOUN_LOAD if p in q)
    return hits >= 2 and len(query) < 80


def should_ask(
    *,
    intent: Any | None,
    query: str,
    history: list[dict] | None,
    memory: dict | None,
) -> bool:
    """Decide whether to surface a clarifying question this turn.

    Returns False unless ALL of:
      * intent says ``needs_clarification`` is true,
      * a non-empty ``clarification_question`` is present,
      * the model's confidence is genuinely low (<= 0.55) OR the query
        is very short and pronoun-heavy with little history,
      * we haven't already asked the same / similar question in the last
        couple of turns (cheap prefix dedup).

    Errs strongly toward proceeding when in doubt.
    """
    if intent is None:
        return False
    needs = bool(getattr(intent, "needs_clarification", False))
    question = (getattr(intent, "clarification_question", None) or "").strip()
    if not (needs and question):
        return False

    conf = float(getattr(intent, "confidence", 0.7) or 0.7)
    pronoun_heavy = _is_pronoun_heavy(query)
    short_query = len(query.strip()) < 80
    prior_assistant_turns = sum(
        1 for m in (history or [])
        if (m.get("role") == "assistant") and (m.get("content") or "").strip()
    )

    # Mid-conversation pronoun-heavy queries almost always benefit from
    # using context rather than asking — skip in that case.
    if prior_assistant_turns >= 2 and pronoun_heavy:
        return False

    # Genuinely-uncertain trigger: low confidence OR short pronoun-heavy
    # opener with no prior turns to lean on.
    triggered = (conf <= 0.55) or (short_query and pronoun_heavy and prior_assistant_turns == 0)
    if not triggered:
        return False

    # Recent-clarification dedup: if the assistant already asked
    # something very similar in the last 4 messages, don't ask again.
    prefix = question.lower()[:48]
    for m in (history or [])[-4:]:
        if m.get("role") != "assistant":
            continue
        content = (m.get("content") or "").lower()
        if prefix and prefix in content:
            return False

    # Memory clearly disambiguates → don't ask.
    if isinstance(memory, dict):
        for tier in ("short", "medium", "long"):
            for v in (memory.get(tier) or {}).values():
                val = (v.get("value") if isinstance(v, dict) else str(v)) or ""
                # Very rough: if any stored preference / context contains a
                # clear marker that overlaps the query subject, assume the
                # downstream pipeline can run without asking.
                q_keywords = [w for w in query.lower().split() if len(w) >= 5]
                if q_keywords and any(w in val.lower() for w in q_keywords):
                    return False
    return True


def render_clarification_message(intent: Any) -> str:
    """Compose the assistant message body for a clarification turn."""
    question = (getattr(intent, "clarification_question", None) or "").strip()
    if not question:
        question = "Could you share a bit more about what you'd like me to focus on?"
    rationale = (getattr(intent, "rationale", None) or "").strip()
    body = [question]
    if rationale and len(rationale) < 220:
        body.append("")
        body.append(f"_Why I'm asking: {rationale}_")
    return "\n".join(body)
