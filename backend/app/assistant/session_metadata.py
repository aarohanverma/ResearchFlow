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
            "description": "4-7 word title summarizing what this session is about. "
                           "No quotes, no period, no 'Research on'.",
        },
        "summary": {
            "type": "string",
            "description": "1-2 sentences (max ~30 words) describing the investigation. "
                           "Crisp and concrete — what's being studied, what stage.",
        },
    },
    "required": ["title", "summary"],
}


async def refresh_session_metadata(session_id: UUID, user_id: UUID) -> None:
    """Best-effort refresh of session title + summary from current messages.

    Title is only updated when it still matches the auto-derived placeholder
    pattern (so user-edited titles are preserved). Summary is always
    refreshed since it's a derived view, not user content.

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
            should_update_title = (
                user_count >= _TITLE_TURN_THRESHOLD
                and any(current_title.startswith(p) or current_title == p
                        for p in _TITLE_PLACEHOLDERS or [])
                or _looks_like_truncated_query(current_title)
            )

            metadata = await _generate_metadata(convo)
            if metadata is None:
                # LLM failed — derive a deterministic summary from messages
                # so the hover tooltip still has something useful to show.
                metadata = {"title": "", "summary": _fallback_summary(recent)}

            new_summary = (metadata.get("summary") or "").strip()
            new_title = (metadata.get("title") or "").strip()

            if new_summary and new_summary != session.summary:
                session.summary = new_summary[:500]
            if should_update_title and new_title:
                session.title = _clean_title(new_title)

            await db.commit()
    except Exception as exc:
        log.warning("session metadata refresh failed session=%s: %s", session_id, exc)


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
    """Strip wrapping punctuation, prefixes the model sometimes adds."""
    t = t.strip().strip('"').strip("'").rstrip(".")
    t = re.sub(r"^(?:Research on|Investigating|Exploring|A study of|Title:)\s+", "", t, flags=re.IGNORECASE)
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


async def _generate_metadata(conversation: str) -> dict | None:
    """Call the cheap LLM to produce {title, summary}; None on failure."""
    if not conversation.strip():
        return None
    try:
        from app.adapters.llm import get_llm_adapter

        llm = get_llm_adapter()
        prompt = (
            "Read the research-assistant conversation below and produce a "
            "compact title and crisp summary for the session list.\n\n"
            "Title rules: 4-7 words, no quotes, no period, no 'Research on'. "
            "Should describe the investigation topic — not the question form.\n"
            "Summary rules: 1-2 sentences, max ~30 words. Concrete and "
            "specific about what's being studied and the current stage.\n\n"
            f"Conversation:\n{conversation}"
        )
        return await llm.complete_structured(
            [{"role": "user", "content": prompt}],
            llm.cheap_model,
            _METADATA_SCHEMA,
        )
    except Exception as exc:
        log.info("metadata LLM unavailable, falling back: %s", exc)
        return None
