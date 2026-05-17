"""Post-turn auto-memory consolidation.

Memory only becomes useful when it keeps current with the conversation. The
planner can call ``memory_write`` explicitly, but planners frequently
forget — important user preferences and key findings slip through. This
module runs as a fire-and-forget task at the END of every substantive
turn and uses a cheap model to surface genuinely NEW facts worth keeping,
then writes them via the same memory pipeline (so caps, dedup, and the
tier-aware root resolution all still apply).

Design constraints (paraphrased from the user):
    * Memory should only ENRICH, never override grounded evidence.
    * Updates must be smart and intelligent — no force-write of trivia.
    * Tiering must respect: short (this chat), medium (the entire session
      tree), long (namespace-wide).
    * The consolidation cost is bounded — one cheap-model call per turn,
      hard cap of 4 writes per turn, total tokens trivially small.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import UUID

from app.assistant.tools.memory import (
    _MAX_LONG_ENTRIES,
    _MAX_MEDIUM_ENTRIES,
    _MAX_SHORT_ENTRIES,
    _SCOPE_TO_BUCKET,
    _evict_to_cap,
    _normalize_key,
    _resolve_root_session,
)
from app.db.session import async_session_factory
from app.models.assistant import AssistantSession

log = logging.getLogger(__name__)

_VALID_TIERS = {"short", "medium", "long"}
_VALID_TYPES = {
    "finding", "preference", "concept", "hypothesis", "context", "paper_note",
    "episode", "skill", "procedure",
}
_MAX_WRITES_PER_TURN = 4
_MAX_DELETES_PER_TURN = 2


_CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "writes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tier":  {"type": "string", "enum": ["short", "medium", "long"]},
                    "type":  {"type": "string", "enum": list(sorted(_VALID_TYPES))},
                    "key":   {"type": "string", "maxLength": 80},
                    "value": {"type": "string", "maxLength": 800},
                    "reason": {"type": "string", "maxLength": 200},
                },
                "required": ["tier", "type", "key", "value"],
            },
        },
        "deletes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tier": {"type": "string", "enum": ["short", "medium", "long"]},
                    "key":  {"type": "string", "maxLength": 80},
                    "reason": {"type": "string", "maxLength": 200},
                },
                "required": ["tier", "key"],
            },
        },
    },
    "required": ["writes"],
}


async def consolidate_after_turn(
    *,
    session_id: UUID,
    user_id: UUID,
    user_query: str,
    assistant_answer: str,
    namespace_key: str,
) -> None:
    """Decide what to remember from this turn, and apply the changes.

    Never raises — wrapped so the orchestrator can ``asyncio.create_task``
    this and walk away.
    """
    try:
        async with async_session_factory() as db:
            session = await db.get(AssistantSession, session_id)
            if session is None:
                return

            # Load current memory state (across the right scopes) to give
            # the LLM enough context to AVOID duplicating known facts.
            current_state = dict(session.state or {})
            short_mem = current_state.get("chat_memory") or {}
            root = await _resolve_root_session(db, session_id) or session
            root_state = dict(root.state or {})
            tree_mem = root_state.get("tree_memory") or {}
            # Legacy alias for back-compat reads.
            for k, v in (root_state.get("memory") or {}).items():
                tree_mem.setdefault(k, v)
            ns_mem = (current_state.get("ns_memory") or {}) or (root_state.get("ns_memory") or {})

            decision = await _ask_llm_what_to_remember(
                user_query=user_query,
                assistant_answer=assistant_answer,
                namespace_key=namespace_key,
                short_mem=short_mem,
                tree_mem=tree_mem,
                ns_mem=ns_mem,
            )
            if not decision:
                return

            writes = (decision.get("writes") or [])[:_MAX_WRITES_PER_TURN]
            deletes = (decision.get("deletes") or [])[:_MAX_DELETES_PER_TURN]

            await _apply_writes(db, session, root, writes)
            await _apply_deletes(db, session, root, deletes)
            await db.commit()
    except Exception as exc:
        log.warning("auto-memory consolidation failed session=%s: %s", session_id, exc)


# ── Internals ────────────────────────────────────────────────────────────────


async def _ask_llm_what_to_remember(
    *,
    user_query: str,
    assistant_answer: str,
    namespace_key: str,
    short_mem: dict,
    tree_mem: dict,
    ns_mem: dict,
) -> dict[str, Any] | None:
    """Single cheap-model call. Returns ``None`` on any failure."""
    try:
        from app.adapters.llm import get_llm_adapter
        llm = get_llm_adapter()

        # Pass essentially the full answer and query — capping early loses
        # findings mentioned later in long answers, which are exactly the
        # entries we want to remember. The cheap-model context is plenty
        # wide; we just guard against truly runaway inputs.
        answer_excerpt = (assistant_answer or "")[:16000]
        query_excerpt = (user_query or "")[:4000]

        def _summarise(d: dict) -> str:
            if not d:
                return "(empty)"
            lines = []
            for k, v in list(d.items())[:12]:
                val = v.get("value") if isinstance(v, dict) else str(v)
                t = v.get("type") if isinstance(v, dict) else "context"
                lines.append(f"  - [{t}] {k}: {str(val)[:120]}")
            return "\n".join(lines)

        system = (
            "You are the memory librarian for a research assistant. The user "
            "just completed one turn of a conversation. Decide what — if "
            "anything — is worth WRITING into long-lived memory.\n\n"
            "Be conservative. Do NOT save:\n"
            "  - paraphrases of facts already in memory\n"
            "  - generic statements ('research is hard')\n"
            "  - speculation, opinions, or claims the assistant fabricated\n"
            "  - one-off tool outputs (they live in the conversation history)\n\n"
            "DO save. Pick exactly ONE label from one of the two independent groups:\n"
            "\n"
            "  CONTENT TYPES (what the entry says):\n"
            "    - 'finding'    : non-trivial research conclusion\n"
            "    - 'concept'    : definition or term worth keeping\n"
            "    - 'hypothesis' : tracked research hypothesis\n"
            "    - 'paper_note' : specific note about a paper (arXiv ID, claim, …)\n"
            "    - 'preference' : durable user preference (tone, depth, exclusions)\n"
            "    - 'context'    : catch-all fallback (use sparingly)\n"
            "\n"
            "  COGNITIVE-CLASS TYPES (how the agent should treat the entry):\n"
            "    - 'episode'   : a past interaction the user actually lived through\n"
            "    - 'skill'     : a technique the user wants the agent to apply again\n"
            "    - 'procedure' : a routine/workflow worth replaying\n"
            "\n"
            "Do NOT force a content fact (e.g. a hypothesis) into a cognitive class. "
            "If a fact is just content, pick from the first group only.\n\n"
            "Pick the right tier:\n"
            "  short  — only for this conversation (current focus, scratch facts)\n"
            "  medium — entire session tree shares it (the investigation's centre)\n"
            "  long   — every session in this namespace (role, durable prefs, affiliation)\n\n"
            "Cap writes at 4 per turn. Re-use existing keys when refining a known "
            "fact (overwrites instead of accumulating duplicates). Return strict "
            "JSON matching the schema."
        )

        user_msg = (
            f"NAMESPACE: {namespace_key}\n\n"
            f"USER TURN:\n{query_excerpt}\n\n"
            f"ASSISTANT REPLY EXCERPT:\n{answer_excerpt}\n\n"
            "EXISTING MEMORY (do not duplicate):\n"
            f"<short>\n{_summarise(short_mem)}\n</short>\n"
            f"<medium>\n{_summarise(tree_mem)}\n</medium>\n"
            f"<long>\n{_summarise(ns_mem)}\n</long>"
        )

        return await llm.complete_structured(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            llm.cheap_model,
            _CONSOLIDATE_SCHEMA,
        )
    except Exception as exc:
        log.debug("auto-memory LLM call failed: %s", exc)
        return None


async def _apply_writes(db, session, root, writes: list[dict]) -> None:
    from datetime import datetime, timezone
    from sqlalchemy.orm.attributes import flag_modified

    for entry in writes:
        try:
            tier = entry.get("tier")
            if tier not in _VALID_TIERS:
                continue
            mem_type = entry.get("type")
            if mem_type not in _VALID_TYPES:
                mem_type = "context"
            key = _normalize_key(str(entry.get("key") or ""))
            value = str(entry.get("value") or "").strip()
            if not key or not value:
                continue
            # Guard against trivial 1-word writes the model sometimes emits.
            if len(value) < 6:
                continue

            target = root if tier == "medium" else session
            bucket = _SCOPE_TO_BUCKET[tier]
            state = dict(target.state or {})
            mem = dict(state.get(bucket) or {})

            existing = mem.get(key)
            if isinstance(existing, dict) \
                    and existing.get("value") == value \
                    and existing.get("type") == mem_type:
                continue  # idempotent

            new_entry: dict = {
                "value": value,
                "type": mem_type,
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": "auto",
            }
            if tier == "medium" and target.id != session.id:
                new_entry["origin_session"] = str(session.id)
            mem[key] = new_entry

            cap = {
                "short": _MAX_SHORT_ENTRIES,
                "medium": _MAX_MEDIUM_ENTRIES,
                "long": _MAX_LONG_ENTRIES,
            }[tier]
            mem = _evict_to_cap(mem, cap)
            state[bucket] = mem
            target.state = state
            flag_modified(target, "state")
        except Exception as exc:
            log.debug("auto-memory write rejected: %s", exc)
            continue


async def _apply_deletes(db, session, root, deletes: list[dict]) -> None:
    from sqlalchemy.orm.attributes import flag_modified

    for entry in deletes:
        try:
            tier = entry.get("tier")
            if tier not in _VALID_TIERS:
                continue
            key = _normalize_key(str(entry.get("key") or ""))
            target = root if tier == "medium" else session
            bucket = _SCOPE_TO_BUCKET[tier]
            state = dict(target.state or {})
            mem = dict(state.get(bucket) or {})
            if key in mem:
                del mem[key]
                state[bucket] = mem
                target.state = state
                flag_modified(target, "state")
        except Exception as exc:
            log.debug("auto-memory delete rejected: %s", exc)
            continue


# Used by the orchestrator to ask: "is the user query / answer trivial enough
# to skip consolidation entirely?" — short greetings, acknowledgments etc.
_TRIVIAL_PATTERNS = (
    re.compile(r"^(hi|hey|hello|thanks|thank you|ok|okay|cool|great|nice)\b", re.I),
)


def is_trivial_turn(user_query: str) -> bool:
    """Return True when the turn doesn't justify a consolidation pass."""
    q = (user_query or "").strip()
    if len(q) < 12:
        return True
    return any(p.match(q) for p in _TRIVIAL_PATTERNS)
