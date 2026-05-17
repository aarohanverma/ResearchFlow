"""Session title + summary auto-derivation.

Both fields are derived from the actual conversation, not hardcoded:

* ``title`` — a 4-7 word handle for the session list. Generated on the
  first user turn (when the placeholder title is still in place) and
  refreshed if the user keeps the auto-derived title.
* ``summary`` — a 1-2 sentence crisp description of what this session is
  investigating, refreshed after every completed turn so the hover
  tooltip stays current as the conversation evolves.

Both are produced by a single ``cheap_model`` structured-output call so
the per-turn cost is bounded. Failures fall back to deterministic
heuristics — the user always sees something, never an empty title.

Runs as fire-and-forget after the user-facing answer is delivered so
synthesis latency is unaffected.
"""

from __future__ import annotations

import logging
import re
from uuid import UUID

from app.db.session import async_session_factory
from app.repositories.assistant import AssistantRepository

log = logging.getLogger(__name__)


_TITLE_PLACEHOLDERS = ("Research workspace:", "Branch:", "Untitled investigation")
_SUMMARY_MAX_TURNS = 8       # Window for summary generation.
_TITLE_TURN_THRESHOLD = 1    # Refresh title after first user turn.

_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": (
                "Up to 8 words for the session list — concise, descriptive, "
                "and professional. No quotes, no trailing period, no "
                "'Research on'. Reflect the CURRENT topical centre of the "
                "conversation; if the focus has shifted from the original "
                "question, the title shifts with it. 4-6 words is the sweet "
                "spot; 8 is the hard ceiling."
            ),
        },
        "summary": {
            "type": "string",
            "description": (
                "ONE sentence, max 18 words, describing what the user is "
                "investigating in this session RIGHT NOW. Reflect the latest "
                "turn — if direction shifted, the summary shifts. No ellipsis, "
                "no truncation. Concrete nouns over generic ones (e.g. "
                "'attention mechanism efficiency' beats 'a machine learning topic')."
            ),
        },
    },
    "required": ["title", "summary"],
}


async def refresh_session_metadata(session_id: UUID, user_id: UUID) -> None:
    """Best-effort refresh of session title + summary from current messages.

    Title is only updated when it still matches the auto-derived placeholder
    pattern (so user-edited titles are preserved). Summary is always
    refreshed since it's a derived view, not user content. The summary
    prompt is seeded with the session's stored memory so the refresh
    reflects accumulated preferences/findings, not just the latest turn.

    Never raises — all failures are logged and swallowed so this can be
    safely called as ``asyncio.create_task(refresh_session_metadata(...))``.
    """
    try:
        async with async_session_factory() as db:
            repo = AssistantRepository(db)
            session = await repo.get_session(user_id, session_id)
            if not session:
                return

            # Build the conversation snapshot the LLM will summarize.
            recent = (session.messages or [])[-_SUMMARY_MAX_TURNS:]
            convo = "\n".join(_format_msg(m) for m in recent if _is_substantive(m))
            user_count = sum(1 for m in recent if _role(m) == "user")
            if user_count == 0:
                return

            current_title = session.title or ""
            # ``title_user_edited`` is set by the rename endpoint and is the
            # authoritative signal that the user has taken ownership of the
            # title. While that flag is true we never touch the title again.
            # While it's false the title keeps refreshing each turn to reflect
            # the conversation's current focus.
            state = dict(getattr(session, "state", None) or {})
            title_is_user_edited = bool(state.get("title_user_edited"))
            last_auto = (state.get("auto_title") or "").strip()
            should_update_title = (
                user_count >= _TITLE_TURN_THRESHOLD
                and not title_is_user_edited
                and (
                    not last_auto
                    or current_title == last_auto
                    or any(current_title.startswith(p) or current_title == p
                           for p in _TITLE_PLACEHOLDERS or [])
                    or _looks_like_truncated_query(current_title)
                )
            )

            # Memory hint: surface a few session/namespace memory entries so the
            # summary can lean on accumulated context (preferences, key
            # findings) instead of only the freshest turn.
            memory_hint = _memory_hint(session)

            metadata = await _generate_metadata(convo, memory_hint=memory_hint)
            if metadata is None:
                # LLM failed — derive a deterministic summary from messages
                # so the hover tooltip still has something useful to show.
                metadata = {"title": "", "summary": _fallback_summary(recent)}

            new_summary = (metadata.get("summary") or "").strip()
            new_title = (metadata.get("title") or "").strip()

            # Hard cap on length — pydantic accepted whatever the model
            # produced, but we never want a wall-of-text tooltip.
            if new_summary:
                new_summary = _trim_to_one_sentence(new_summary, max_words=24)

            if new_summary and new_summary != session.summary:
                session.summary = new_summary[:240]
            if should_update_title and new_title:
                cleaned = _clean_title(new_title)
                session.title = cleaned
                # Remember the auto-generated title so the next turn's check
                # can tell whether the user has since renamed it.
                from sqlalchemy.orm.attributes import flag_modified
                state["auto_title"] = cleaned
                session.state = state
                flag_modified(session, "state")

            await db.commit()
    except Exception as exc:
        log.warning("session metadata refresh failed session=%s: %s", session_id, exc)


def _memory_hint(session) -> str:
    """Build a short string of stored memory facts for the summary prompt."""
    try:
        state = dict(getattr(session, "state", None) or {})
        med = state.get("memory") or {}
        lng = state.get("ns_memory") or {}
        items: list[str] = []
        for k, v in list(med.items())[:4]:
            val = v.get("value") if isinstance(v, dict) else str(v)
            items.append(f"  - [session] {k}: {str(val)[:140]}")
        for k, v in list(lng.items())[:3]:
            val = v.get("value") if isinstance(v, dict) else str(v)
            items.append(f"  - [namespace] {k}: {str(val)[:140]}")
        if not items:
            return ""
        return "Known context (do not restate verbatim, but let it shape the summary):\n" + "\n".join(items)
    except Exception:
        return ""


def _trim_to_one_sentence(s: str, max_words: int = 24) -> str:
    s = s.strip().strip('"').strip("'")
    # Keep up to the first sentence boundary.
    cut = len(s)
    for stop in (". ", "! ", "? "):
        idx = s.find(stop)
        if 0 < idx < cut:
            cut = idx + 1
    s = s[:cut].rstrip(" .,;:")
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    return s


# ── Helpers ──────────────────────────────────────────────────────────────

def _role(msg) -> str:
    r = getattr(msg, "role", None)
    return r.value if hasattr(r, "value") else str(r)


def _is_substantive(msg) -> bool:
    """Skip empty / system-only messages — they add noise to summaries."""
    if _role(msg) == "system":
        return False
    return bool((getattr(msg, "content", "") or "").strip())


def _format_msg(msg) -> str:
    role = _role(msg)
    content = (getattr(msg, "content", "") or "").strip()
    # Cap to keep prompt size predictable; summaries don't need full text.
    return f"{role}: {content[:600]}"


def _looks_like_truncated_query(title: str) -> bool:
    """The previous heuristic just truncated the user's first query. Detect
    that shape so we can replace it with a properly-LLM-generated title."""
    if not title:
        return True
    # Truncated queries usually end in '…' or are very long with a sentence-y feel.
    return title.endswith("…") or (len(title) >= 50 and " " in title)


def _clean_title(t: str) -> str:
    """Strip wrapping punctuation, prefixes the model sometimes adds, cap length."""
    t = t.strip().strip('"').strip("'").rstrip(".")
    t = re.sub(r"^(?:Research on|Investigating|Exploring|A study of|Title:)\s+", "", t, flags=re.IGNORECASE)
    # Allow up to 8 words to give the auto-titler enough room for descriptive
    # multi-noun topics (e.g. "Comparing RAG vs Long-Context Retrieval Methods"),
    # while still keeping the sidebar readable.
    words = t.split()
    if len(words) > 8:
        t = " ".join(words[:8])
    return t[:80] or "Untitled investigation"


def _fallback_summary(messages) -> str:
    """Deterministic summary when the LLM is unavailable."""
    user_msgs = [m for m in messages if _role(m) == "user"]
    if not user_msgs:
        return "New investigation."
    first = (getattr(user_msgs[0], "content", "") or "").strip()
    if len(user_msgs) == 1:
        return f"Investigating: {first[:160]}"
    return f"{len(user_msgs)} turns starting with: {first[:120]}"


async def _generate_metadata(conversation: str, *, memory_hint: str = "") -> dict | None:
    """Call the cheap LLM to produce {title, summary}; None on failure."""
    if not conversation.strip():
        return None
    try:
        from app.adapters.llm import get_llm_adapter

        llm = get_llm_adapter()
        prompt = (
            "Read the research-assistant conversation below and produce a "
            "compact title and a single-sentence current-state summary.\n\n"
            "Title rules: 4-8 words, no quotes, no period, no 'Research on'. "
            "Describe the investigation topic — not the question form. "
            "Reflect the CURRENT focus: if the conversation has shifted, the "
            "title shifts with it. Aim for 5-6 words; 8 is the hard ceiling.\n"
            "Summary rules: ONE sentence, max 18 words. Reflect the CURRENT "
            "direction (latest turn dominates older context). Be concrete: "
            "name the specific concept/method/paper under investigation. No "
            "ellipsis, no truncation, no filler like 'discusses' or 'explores' — "
            "lead with the actual subject.\n\n"
            + (f"{memory_hint}\n\n" if memory_hint else "")
            + f"Conversation:\n{conversation}"
        )
        return await llm.complete_structured(
            [{"role": "user", "content": prompt}],
            llm.cheap_model,
            _METADATA_SCHEMA,
        )
    except Exception as exc:
        log.info("metadata LLM unavailable, falling back: %s", exc)
        return None
