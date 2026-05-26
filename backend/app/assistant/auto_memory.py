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


# ── Safe-memory filters ─────────────────────────────────────────────────────
#
# The librarian prompt is conservative, but cheap-model judgement still
# occasionally drifts into saving speculative / unverified content. These
# regexes act as a deterministic safety net: any write whose value
# matches one of them is dropped before it lands in storage. The user's
# requirement is unambiguous: "only store durable, verified, or
# user-approved information — never raw guesses, uncited summaries,
# transient search results, failed tool outputs, or speculative
# conclusions."

# Hedge / uncertainty markers. A value that leads with hedging is, by
# definition, NOT a durable fact — keep it out of memory so it doesn't
# get re-cited later as if it were established. Matching is via
# word-boundary regex (not substring) — without word boundaries the
# old check rejected legitimate facts like "in may 2026 we shipped X"
# because the month name overlapped the ``"may "`` substring marker.
# Word boundaries also stop us from rejecting words that merely
# *contain* a hedge — e.g. ``unmaybe`` (nonsense, but illustrates
# the principle) or ``unmightiness``. The ``\w+`` suffixes on
# ``speculat`` and ``likel`` cover the ``-ed/-ive/-y/-hood``
# inflections without listing them all.
_HEDGE_PATTERN = re.compile(
    r"\b("
    r"might|could|possibly|perhaps|maybe|"
    r"speculat\w*|uncertain|unverified|unsupported|"
    r"tentative|presumably|supposedly|"
    r"appears\s+to|seems\s+to|potentially|"
    # ``likely``/``likelihood``/``unlikely`` are domain terms in
    # statistics & ML (maximum likelihood, likelihood ratio,
    # unlikely-event sampling). The other modals already cover the
    # hedge intent — dropping these stops the filter from rejecting
    # legitimate technical findings.
    # ``may`` is BOTH a hedge modal AND a month name — matching it
    # bare false-rejects legitimate facts like "in May 2026 ...".
    # Catch it ONLY when followed by a typical hedge verb. Covers
    # "may be / have / not / need / require / prove / signal / imply
    # / indicate / suggest / show / fail / seem". Misses the rare
    # bare-modal usage; that's an acceptable trade for not corrupting
    # memories with month-name dates.
    r"may\s+(?:be|have|has|had|not|need|require|prove|signal|imply|indicate|suggest|show|reveal|fail|seem)|"
    r"i\s+think|we\s+think|i\s+believe|we\s+believe|"
    r"not\s+sure|unclear|unknown|tbd"
    r")\b",
    re.IGNORECASE,
)

# Transient-content markers. These describe RUN-state — search hits we
# just saw, tool failures we just observed, etc. They don't belong in
# durable memory; the conversation history already has them. Substring
# matching here is intentional: phrases like ``"the model "`` (with
# trailing space) and ``"this turn "`` are run-state prefixes; we'd
# rather over-reject the occasional legitimate fact phrased as a
# run-state description than persist a tool-trace into long-lived
# memory.
_TRANSIENT_MARKERS: tuple[str, ...] = (
    "search returned", "search results", "no results",
    "tool failed", "tool error", "synthesis failed",
    "found 0", "found no ", "paper_qa returned",
    "the loop ", "this turn ", "the agent ",
    "the planner ", "the model ", "scratchpad",
)

# Template-placeholder leak guard. If the librarian regurgitated a
# placeholder it saw in a tool input ("{{best_supporting_paper_id}}"),
# the value is by definition not a real fact.
_TEMPLATE_PLACEHOLDER = re.compile(
    r"\{\{\s*[A-Za-z_][\w.\-]*\s*\}\}|\$\{\s*[A-Za-z_][\w.\-]*\s*\}",
)


def _is_unsafe_memory_value(value: str, mem_type: str) -> tuple[bool, str]:
    """Return ``(unsafe, reason)`` when ``value`` looks like content
    the user explicitly told us not to store.

    Conservative on purpose — false positives just mean the librarian's
    decision gets dropped (re-derivable next turn), false negatives let
    speculation slip into long-lived memory where it becomes a
    confidence-inflating citation source for every future turn.

    ``value`` is tolerant of ``None`` (treated as empty) and any
    non-string input (coerced via ``str``). The function never raises;
    a malformed input simply returns ``(True, "empty value")``.
    """
    if value is None:
        return True, "empty value"
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return True, "non-stringable value"
    low = value.lower().strip()
    if not low:
        return True, "empty value"

    if _TEMPLATE_PLACEHOLDER.search(value):
        return True, "value contains unresolved template placeholder"

    # Hedge markers in the FIRST 200 chars only — a long durable fact
    # that mentions "potentially" in passing far from the headline
    # shouldn't be rejected. Hedges at the start strongly indicate the
    # whole entry is uncertain. Word-boundary matching avoids rejecting
    # legitimate text where a hedge word is merely a substring (e.g.
    # ``"may 2026"`` for the month, or surnames like ``Mayer``).
    head = low[:200]
    hedge_match = _HEDGE_PATTERN.search(head)
    if hedge_match:
        return True, f"hedged value (matched {hedge_match.group(0)!r})"

    for marker in _TRANSIENT_MARKERS:
        if marker in low:
            return True, f"transient run-state (matched {marker!r})"

    # Hypothesis-typed entries get an EXTRA bar: they're proposals, so
    # they MUST carry the user's explicit framing. Auto-tagged
    # hypothesis writes are too easy a route for the librarian to
    # smuggle speculation into long-lived memory.
    if mem_type == "hypothesis" and "?" in head:
        return True, "hypothesis value reads as an open question"

    return False, ""


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

    State serialisation:
        The LLM call runs OUTSIDE the per-session lock so multiple
        consolidation passes can compose their decisions in parallel.
        Only the final read-modify-write against ``session.state`` is
        serialised, so we never lose updates between this pass and a
        concurrent branch-summary roll-up / telemetry append. The lock
        is intra-process; cross-process contention is documented as
        last-writer-wins in ``STATE_OWNERSHIP.md``.
    """
    from app.assistant.state_lock import session_state_lock

    try:
        # ── Phase 1: LLM decision (no lock held) ──────────────────────
        async with async_session_factory() as db:
            session = await db.get(AssistantSession, session_id)
            if session is None:
                return
            current_state = dict(session.state or {})
            short_mem = current_state.get("chat_memory") or {}
            root = await _resolve_root_session(db, session_id) or session
            root_state = dict(root.state or {})
            tree_mem = root_state.get("tree_memory") or {}
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
        if not writes and not deletes:
            return

        # ── Phase 2: serialised commit ────────────────────────────────
        # Reload the session under the per-session lock so we apply the
        # decision against the freshest state. Tree memory may live on
        # the root session; for medium-tier writes we additionally lock
        # the root key to avoid sibling branches racing on the same
        # ``tree_memory`` dict.
        async with session_state_lock(session_id):
            async with async_session_factory() as db:
                session = await db.get(AssistantSession, session_id)
                if session is None:
                    return
                root = await _resolve_root_session(db, session_id) or session
                if root.id == session.id:
                    await _apply_writes(
                        db, session, root, writes,
                        namespace_key=namespace_key, user_id=user_id,
                    )
                    await _apply_deletes(
                        db, session, root, deletes,
                        namespace_key=namespace_key, user_id=user_id,
                    )
                    await db.commit()
                    return
                # Critical: re-fetch root AFTER acquiring its lock. The earlier
                # ``_resolve_root_session`` returns a snapshot taken before the
                # lock was held, so a sibling branch's auto-memory pass that
                # committed in the lock-acquisition window would otherwise be
                # silently overwritten when we save ``root.state`` below.
                async with session_state_lock(root.id):
                    root = await db.get(AssistantSession, root.id)
                    if root is None:
                        return
                    await db.refresh(root)
                    await _apply_writes(
                        db, session, root, writes,
                        namespace_key=namespace_key, user_id=user_id,
                    )
                    await _apply_deletes(
                        db, session, root, deletes,
                        namespace_key=namespace_key, user_id=user_id,
                    )
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
            "DO save. Pick exactly ONE label. Long-term memory follows the\n"
            "three-class cognitive taxonomy adopted from CoALA: SEMANTIC (facts),\n"
            "EPISODIC (events / experiences), PROCEDURAL (how-to / instructions).\n"
            "The two groups below let you match an entry's CONTENT shape OR its\n"
            "COGNITIVE class — never both for the same entry.\n"
            "\n"
            "  SEMANTIC content types (facts the user / world):\n"
            "    - 'finding'    : non-trivial research conclusion (semantic)\n"
            "    - 'concept'    : definition or term worth keeping (semantic)\n"
            "    - 'paper_note' : specific note about a paper (semantic)\n"
            "    - 'preference' : durable user preference (semantic preference)\n"
            "    - 'hypothesis' : tracked research hypothesis (no class — proposal)\n"
            "    - 'context'    : catch-all fallback (no class — use sparingly)\n"
            "\n"
            "  EPISODIC type — a specific past interaction:\n"
            "    - 'episode'   : a past interaction the user actually lived through\n"
            "                    (\"User compared GPT-4o vs Claude on this dataset\")\n"
            "\n"
            "  PROCEDURAL types — how-to knowledge:\n"
            "    - 'skill'     : a technique the user wants the agent to apply again\n"
            "                    (\"Always cite Semantic Scholar for biomedical papers\")\n"
            "    - 'procedure' : a routine / workflow worth replaying\n"
            "                    (\"After listing papers, attach a TL;DR matrix\")\n"
            "\n"
            "If unsure between SEMANTIC and EPISODIC: a fact about the world is\n"
            "semantic; a record of what happened in a session is episodic. PROCEDURAL\n"
            "is reserved for actionable instructions, not descriptive facts.\n\n"
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


async def _apply_writes(
    db, session, root, writes: list[dict],
    *,
    namespace_key: str = "",
    user_id: UUID | None = None,
) -> None:
    from datetime import datetime, timezone
    from sqlalchemy.orm.attributes import flag_modified

    from app.assistant.memory_revisions import record_revision

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
            # PII redaction at the memory boundary — strips credit
            # cards (Luhn-validated), emails, API keys, SSNs, and
            # phone numbers from the value BEFORE it lands in
            # ``session.state``. The librarian model otherwise
            # cheerfully "remembers" PII it sees in the transcript;
            # once persisted, that data lives in the embedding cache,
            # tree memory, and any branch-summary roll-up. Helper
            # never raises (on regex failure the original text is
            # kept) so a redactor glitch can never silently break
            # consolidation. Detection events are logged at INFO so
            # operators can audit what classes of PII surfaced.
            try:
                from app.assistant.pii_redactor import redact_pii
                _pii = redact_pii(value)
                if _pii.found:
                    log.info(
                        "auto-memory: redacted %s from %s-tier write key=%r",
                        ",".join(sorted(_pii.found)), tier, key,
                    )
                    value = _pii.text
            except Exception:  # noqa: BLE001 — never abort consolidation
                pass
            # Programmatic safety net on top of the prompt-level guard.
            # The librarian model occasionally tries to save speculative
            # / unverified / transient content; this filter rejects
            # writes whose value reads as a guess, a tool-run trace, or
            # an unresolved template variable. See
            # ``_is_unsafe_memory_value`` for the rule set.
            unsafe, reason = _is_unsafe_memory_value(value, mem_type)
            if unsafe:
                log.info(
                    "auto-memory: refusing %s write (tier=%s, type=%s): %s",
                    key, tier, mem_type, reason,
                )
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

            # Capture the prior value for the audit trail BEFORE we
            # overwrite — ``previous_value`` is what enables compare
            # and restore in the UI.
            prior_value: str | None = None
            if existing is not None:
                if isinstance(existing, dict):
                    prior_value = str(existing.get("value") or "")
                else:
                    prior_value = str(existing)

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

            # ── Automatic supersession ────────────────────────────────
            # Scan the bucket for near-duplicates the new write
            # supersedes. Class-coherent and embedding-thresholded —
            # never deletes the older entry, just flags it. Soft
            # feature: if the embedder is offline or the check
            # raises, supersession is silently skipped and the
            # write proceeds without it. Only runs on persistent
            # tiers (medium / long) because short-tier auto-prunes.
            superseded_keys: list[str] = []
            if tier in ("medium", "long"):
                try:
                    from app.assistant.memory_supersession import (
                        detect_and_mark_supersessions,
                    )
                    superseded_keys = await detect_and_mark_supersessions(
                        bucket=mem,
                        new_key=key,
                        new_value=value,
                        new_type=mem_type,
                        session_id=session.id,
                    )
                except Exception as _sup_exc:  # noqa: BLE001
                    log.debug("supersession scan skipped: %s", _sup_exc)

            state[bucket] = mem
            target.state = state
            flag_modified(target, "state")

            # ── Audit trail ──────────────────────────────────────────
            # Long-tier writes carry the namespace explicitly; medium
            # writes are tree-scoped so namespace is empty in the
            # revision (the inspect UI groups them under "session
            # tree"). Short-tier writes are deliberately NOT recorded
            # — they're per-chat and auto-prune; auditing them would
            # bloat the log without giving the user actionable
            # history.
            if user_id is not None and tier in ("medium", "long"):
                ns_for_row = namespace_key if tier == "long" else ""
                await record_revision(
                    db,
                    user_id=user_id,
                    session_id=session.id,
                    tier=tier,
                    key=key,
                    value=value,
                    action=("update" if prior_value is not None else "create"),
                    namespace_key=ns_for_row,
                    entry_type=mem_type,
                    source="auto",
                    previous_value=prior_value,
                )
                # One ``supersede`` revision per displaced entry so
                # the History modal shows the chain: who superseded
                # whom and by which new key.
                for sup_key in superseded_keys:
                    try:
                        await record_revision(
                            db,
                            user_id=user_id,
                            session_id=session.id,
                            tier=tier,
                            key=sup_key,
                            value="",
                            action="supersede",
                            namespace_key=ns_for_row,
                            entry_type=mem_type,
                            source="auto",
                            previous_value=None,
                            status="superseded",
                            extras={"superseded_by_key": key},
                        )
                    except Exception as _rev_exc:  # noqa: BLE001
                        log.debug(
                            "supersede revision skipped for key=%s: %s",
                            sup_key, _rev_exc,
                        )
        except Exception as exc:
            log.debug("auto-memory write rejected: %s", exc)
            continue


async def _apply_deletes(
    db, session, root, deletes: list[dict],
    *,
    namespace_key: str = "",
    user_id: UUID | None = None,
) -> None:
    from sqlalchemy.orm.attributes import flag_modified

    from app.assistant.memory_revisions import record_revision

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
                removed_entry = mem[key]
                prior_value = ""
                if isinstance(removed_entry, dict):
                    prior_value = str(removed_entry.get("value") or "")
                elif removed_entry is not None:
                    prior_value = str(removed_entry)
                prior_type = "context"
                if isinstance(removed_entry, dict):
                    prior_type = str(removed_entry.get("type") or "context")
                del mem[key]
                state[bucket] = mem
                target.state = state
                flag_modified(target, "state")
                if user_id is not None and tier in ("medium", "long"):
                    ns_for_row = namespace_key if tier == "long" else ""
                    await record_revision(
                        db,
                        user_id=user_id,
                        session_id=session.id,
                        tier=tier,
                        key=key,
                        value="",
                        action="delete",
                        namespace_key=ns_for_row,
                        entry_type=prior_type,
                        source="auto",
                        previous_value=prior_value,
                        status="deleted",
                    )
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
