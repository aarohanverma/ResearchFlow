"""Memory write/recall/delete tools for the Research Assistant.

Three-tier memory system, every entry stored in AssistantSession.state (JSONB):

  short   — ``session.state["chat_memory"]``
            Per-chat facts. Lives on the current session only. Branches and
            siblings do NOT see each other's short memory — it is local
            scratch for the conversation in front of the user.

  medium  — ``root_session.state["tree_memory"]``
            Per session tree (parent + all branches + nested branches). Stored
            at the ROOT session so every node in the tree reads the same
            store and every write — wherever in the tree it originates —
            propagates to the whole tree. This is how a finding discovered
            inside a branch reaches the parent, and how the parent's
            preferences reach every branch.

  long    — ``session.state["ns_memory"]``
            Namespace-wide. Survives across all sessions in this namespace
            and is copied forward to new sessions (see
            ``AssistantRepository.create_session``).

Each entry: ``{value, type, ts}``. Typed categories enable structured recall.
The orchestrator injects ``{short, medium, long}`` into every planner +
synthesizer prompt automatically — the planner only calls these tools to
WRITE new facts or DELETE stale ones; recall is implicit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.assistant.tools.base import ToolContext, ToolResult
from app.models.assistant import AssistantSession

log = logging.getLogger(__name__)

# Memory typing is intentionally TWO INDEPENDENT DIMENSIONS:
#
#   1. ``memory_type`` — what the entry IS about (its content shape).
#      finding, concept, hypothesis, paper_note, preference, context
#
#   2. cognitive category — what cognitive memory class the entry maps to.
#      episode (a past interaction the user lived through), skill /
#      procedure (procedural know-how), or — for the content-shape types
#      above — only a soft mapping when the connection is clean.
#
# We keep both spaces and ONLY produce a soft mapping where it's natural;
# everything else gets ``"-"`` and the planner treats the content-type
# label as authoritative. This avoids the trap of forcing a research
# ``hypothesis`` to pretend it's an "episodic memory" just to fit the
# four-class diagram from agent-memory papers.
_VALID_TYPES = {
    # Content-shape types (what the entry says)
    "finding", "preference", "concept", "hypothesis", "context", "paper_note",
    # Cognitive-class types (how the agent should treat the entry)
    "episode", "skill", "procedure",
}

# Soft mapping ONLY where the connection is unambiguous. Anything not
# in the table maps to "-" — the planner sees the content type as-is and
# does not pretend to know the cognitive class.
_MEMORY_CATEGORY = {
    "episode":    "episodic",
    "skill":      "procedural",
    "procedure":  "procedural",
    "preference": "preference",
    # Soft connections — these are FACTS the user accumulated, which a
    # cognitive-memory model would file under semantic. The planner can
    # use this hint without it implying the content-shape changes.
    "concept":    "semantic",
    "finding":    "semantic",
    "paper_note": "semantic",
    # No soft mapping: hypotheses are tracked research constructs in their
    # own right; context is a fallback bag; neither cleanly belongs in
    # episodic/semantic/procedural.
}


def memory_category(memory_type: str) -> str:
    """Return the cognitive-science category for a memory_type label.

    Returns ``"-"`` when there is no clean mapping — never invents one.
    """
    return _MEMORY_CATEGORY.get(memory_type, "-")

# Scope aliases. We accept legacy ``"medium"`` to mean per-session-tree (the
# new semantics) and ``"session"``/``"short"`` to mean per-chat. New writes
# should use one of {"short", "medium", "long"} explicitly.
_SCOPE_ALIASES = {
    "short": "short",
    "chat": "short",
    "session": "short",
    "medium": "medium",
    "tree": "medium",
    "long": "long",
    "namespace": "long",
    "ns": "long",
}

_SCOPE_TO_BUCKET = {
    "short":  "chat_memory",
    "medium": "tree_memory",
    "long":   "ns_memory",
}

# Eviction caps so a runaway loop can't blow up the JSONB state column.
# Preferences are protected from eviction — they're rare, durable, and
# disproportionately useful — the LRU pass evicts the oldest non-preference
# entry to make room when full.
_MAX_SHORT_ENTRIES = 30
_MAX_MEDIUM_ENTRIES = 80
_MAX_LONG_ENTRIES = 120


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry_value(entry: object) -> str:
    """Normalize stored entry — supports both legacy str and new dict format."""
    if isinstance(entry, dict):
        return entry.get("value", "")
    return str(entry)


def _entry_type(entry: object) -> str:
    if isinstance(entry, dict):
        return entry.get("type", "context")
    return "context"


def _entry_ts(entry: object) -> str:
    if isinstance(entry, dict):
        return entry.get("ts", "")
    return ""


def _normalize_key(raw: str) -> str:
    """Coerce arbitrary planner-emitted keys into stable snake_case slugs.

    The LLM frequently writes ``"User Background"`` one turn and
    ``"user-background"`` the next, accumulating duplicates instead of
    overwriting a single fact. Normalisation collapses both to
    ``user_background``.
    """
    raw = (raw or "").strip().lower()
    # Replace any run of non-alphanumeric chars with a single underscore.
    out = []
    prev_us = False
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    s = "".join(out).strip("_")
    return s[:120] or "memory"


async def _resolve_root_session_id(db, session_id):
    """Return the root session UUID via a single recursive CTE.

    Replaces the per-hop ``db.get(...)`` walk so root resolution stays
    O(1) DB roundtrips regardless of how deeply branches are nested.
    The CTE is bounded to 20 levels as a cycle guard.

    Returns:
        The root ``UUID`` if the chain resolves, else ``session_id`` so
        callers can treat a missing row as "self is root" without an
        extra null check.
    """
    from sqlalchemy import text as _text
    try:
        result = await db.execute(
            _text(
                """
                WITH RECURSIVE chain(id, parent_session_id, depth) AS (
                    SELECT id, parent_session_id, 0
                    FROM assistant_sessions WHERE id = :start
                    UNION ALL
                    SELECT a.id, a.parent_session_id, c.depth + 1
                    FROM assistant_sessions a
                    JOIN chain c ON a.id = c.parent_session_id
                    WHERE c.depth < 20
                )
                SELECT id FROM chain ORDER BY depth DESC LIMIT 1
                """
            ),
            {"start": session_id},
        )
        row = result.first()
        return row[0] if row else session_id
    except Exception:
        # Fall back to the original walk on any error (e.g. stub DB in tests).
        seen: set = set()
        current = await db.get(AssistantSession, session_id)
        if current is None:
            return session_id
        for _ in range(20):
            if current.parent_session_id is None or current.parent_session_id in seen:
                return current.id
            seen.add(current.id)
            nxt = await db.get(AssistantSession, current.parent_session_id)
            if nxt is None:
                return current.id
            current = nxt
        return current.id


async def _resolve_root_session(db, session_id):
    """Return the root ``AssistantSession`` ORM row via :func:`_resolve_root_session_id`.

    Kept as a thin convenience for callers that need the full row (e.g.
    they're about to mutate state). New code should prefer
    :func:`_resolve_root_session_id` when only the UUID is needed — it
    avoids the second SELECT and the message/task selectinloads that
    come with a full row fetch.
    """
    rid = await _resolve_root_session_id(db, session_id)
    if rid is None:
        return None
    return await db.get(AssistantSession, rid)


def _evict_to_cap(mem: dict, cap: int) -> dict:
    """Drop oldest non-preference entries until ``len(mem) <= cap``.

    Preference entries are always kept — even when over cap — because user
    stated preferences (tone, depth, naming conventions) are the highest-value
    long-lived facts a planner can have.
    """
    if len(mem) <= cap:
        return mem
    # Stable order: preferences first (never evictable), then by timestamp
    # ascending (oldest non-preference first).
    items = list(mem.items())
    prefs = [(k, v) for k, v in items if _entry_type(v) == "preference"]
    others = [(k, v) for k, v in items if _entry_type(v) != "preference"]
    others.sort(key=lambda kv: _entry_ts(kv[1]) or "")
    # Drop oldest non-preferences until we fit.
    keep_others = others[max(0, len(others) - (cap - len(prefs))):] if cap > len(prefs) else []
    new = dict(prefs)
    new.update(keep_others)
    return new


# ── Write ─────────────────────────────────────────────────────────────────────


class MemoryWriteInput(BaseModel):
    key: str = Field(
        min_length=1,
        max_length=120,
        description="Short identifier (snake_case, e.g. 'user_background', 'key_finding_attention').",
    )
    value: str = Field(
        min_length=1,
        max_length=2000,
        description="The fact or insight to remember. Plain text, one concept per entry.",
    )
    scope: str = Field(
        default="medium",
        pattern="^(short|chat|session|medium|tree|long|namespace|ns)$",
        description=(
            "Tier:\n"
            "  'short' / 'chat' / 'session' — this chat only (current session).\n"
            "  'medium' / 'tree' — this session tree (parent + all branches share it).\n"
            "  'long' / 'namespace' / 'ns' — every session in this namespace."
        ),
    )
    memory_type: str = Field(
        default="context",
        description=(
            "What this entry IS. Pick the most specific label that applies — "
            "the labels live in two independent groups; do not mix them:\n"
            "\n"
            "  CONTENT TYPES (what the entry says):\n"
            "    - 'finding'    : a non-trivial research conclusion\n"
            "    - 'concept'    : a definition or term worth keeping\n"
            "    - 'hypothesis' : a tracked research hypothesis\n"
            "    - 'paper_note' : a specific note about a paper (arXiv ID, claim, etc.)\n"
            "    - 'preference' : a durable user preference (tone, depth, exclusions)\n"
            "    - 'context'    : catch-all fallback (use sparingly)\n"
            "\n"
            "  COGNITIVE-CLASS TYPES (how the agent should treat the entry):\n"
            "    - 'episode'   : a past interaction or event the user actually lived through\n"
            "    - 'skill'     : a technique the user wants the agent to apply again\n"
            "    - 'procedure' : a routine or workflow worth replaying\n"
            "\n"
            "Use ONE label, not both. If a fact is just content, pick from the first "
            "group. If it's about how to act or a specific past moment, pick from the second."
        ),
    )


class MemoryWriteOutput(BaseModel):
    stored: bool
    scope: str
    key: str
    memory_type: str


class MemoryWriteTool:
    """Store a typed key-value fact into chat / tree / namespace memory."""

    name = "memory_write"
    summary = (
        "Persist a typed factual insight or user preference into research memory. "
        "Tiers: scope='short' for this chat, scope='medium' for the entire session "
        "tree (parent + branches share it), scope='long' for namespace-wide "
        "insights that persist across all sessions. "
        "Typed categories: finding | preference | concept | hypothesis | paper_note | context. "
        "Write only genuinely useful facts — one clear write per turn is better than many trivial ones."
    )
    cost_class = "cheap"
    side_effects = True
    cancellable = False
    streamable = False
    input_schema = MemoryWriteInput
    output_schema = MemoryWriteOutput

    async def run(self, ctx: ToolContext, params: MemoryWriteInput) -> ToolResult:
        from app.assistant.state_lock import session_state_lock

        mem_type = params.memory_type if params.memory_type in _VALID_TYPES else "context"
        norm_key = _normalize_key(params.key)
        tier = _SCOPE_ALIASES.get(params.scope, "medium")
        bucket = _SCOPE_TO_BUCKET[tier]
        await ctx.emit_progress(30, f"Storing {tier}-tier [{mem_type}] memory: {norm_key!r}")
        try:
            # ``short`` and ``long`` write to the current session. ``medium``
            # always writes to the ROOT of the session tree so the whole
            # tree (parent + every branch) sees the same store.
            #
            # We resolve the target BEFORE acquiring the lock so we know
            # which session lock to take (the target's), then re-fetch the
            # row inside the lock and re-read its state. Skipping the
            # in-lock refresh let sibling-branch ``memory_write`` calls
            # silently overwrite each other when both touched the same
            # root's ``tree_memory`` simultaneously.
            target_id = (
                await _resolve_root_session_id(ctx.db, ctx.session_id)
                if tier == "medium"
                else ctx.session_id
            )
            if target_id is None:
                target_id = ctx.session_id

            async with session_state_lock(target_id), ctx.db.begin_nested():
                # Re-fetch target inside the lock.
                target = await ctx.db.get(AssistantSession, target_id)
                if target is None:
                    return ToolResult(
                        output={"stored": False, "scope": tier, "key": norm_key, "memory_type": mem_type},
                        summary="session not found",
                    )
                await ctx.db.refresh(target)
                current = (
                    target if target.id == ctx.session_id
                    else await ctx.db.get(AssistantSession, ctx.session_id)
                ) or target

                state = dict(target.state or {})
                mem = dict(state.get(bucket) or {})

                # Idempotency: skip the write entirely when the existing entry
                # already records the same value + type. Without this, every
                # turn that recalls and re-asserts a known fact bumps the
                # timestamp and burns a flush, polluting recency ordering.
                existing = mem.get(norm_key)
                if isinstance(existing, dict) \
                        and existing.get("value") == params.value \
                        and existing.get("type") == mem_type:
                    return ToolResult(
                        output={"stored": True, "scope": tier, "key": norm_key, "memory_type": mem_type, "noop": True},
                        summary=f"Already stored — no-op: {norm_key!r}",
                    )

                # Trace ``written_from`` for medium writes so we know which
                # branch contributed a fact when surfacing the tree memory.
                entry: dict = {"value": params.value, "type": mem_type, "ts": _now_iso()}
                if tier == "medium" and target.id != current.id:
                    entry["origin_session"] = str(current.id)
                mem[norm_key] = entry
                cap = {
                    "short": _MAX_SHORT_ENTRIES,
                    "medium": _MAX_MEDIUM_ENTRIES,
                    "long": _MAX_LONG_ENTRIES,
                }[tier]
                mem = _evict_to_cap(mem, cap)
                state[bucket] = mem
                target.state = state
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(target, "state")
                await ctx.db.flush()
        except Exception as exc:
            log.warning("memory_write failed: %s", exc)
            return ToolResult(
                output={"stored": False, "scope": tier, "key": norm_key, "memory_type": mem_type},
                summary=f"write failed: {exc}",
            )

        await ctx.emit_progress(100, f"Saved {tier}-tier memory [{mem_type}]")
        return ToolResult(
            output={"stored": True, "scope": tier, "key": norm_key, "memory_type": mem_type},
            summary=f"Stored {tier}-tier [{mem_type}] memory: {norm_key!r}",
        )


# ── Recall ─────────────────────────────────────────────────────────────────────


class MemoryRecallInput(BaseModel):
    namespace_key: str = Field(default="")
    query: str = Field(default="", max_length=500, description="Optional keyword filter on key or value.")
    memory_type: str = Field(
        default="",
        description="Optional type filter: 'finding', 'preference', 'concept', 'hypothesis', 'paper_note', 'context'.",
    )


class MemoryRecallOutput(BaseModel):
    short: dict = {}
    medium: dict
    long: dict
    branches: dict = {}
    total_short: int = 0
    total_medium: int
    total_long: int
    total_branches: int = 0


class MemoryRecallTool:
    """Surface stored memory across all three tiers."""

    name = "memory_recall"
    summary = (
        "Retrieve stored research memory across all tiers: chat (short — this "
        "conversation only), tree (medium — parent + branches share it), and "
        "namespace (long — persists across sessions). "
        "Optionally filter by keyword (query) or type. "
        "Use when: user asks about prior context, their background, what was "
        "discovered in previous sessions, or at the start of a continuation."
    )
    cost_class = "cheap"
    side_effects = False
    cancellable = False
    streamable = False
    input_schema = MemoryRecallInput
    output_schema = MemoryRecallOutput

    async def run(self, ctx: ToolContext, params: MemoryRecallInput) -> ToolResult:
        await ctx.emit_progress(50, "Recalling research memory")
        branches_out: dict = {}
        try:
            row = await ctx.db.get(AssistantSession, ctx.session_id)
            state = dict(row.state or {}) if row else {}
            # short = this chat only
            short = dict(state.get("chat_memory") or {})
            # medium = the entire session tree, stored at the root
            root = await _resolve_root_session(ctx.db, ctx.session_id) if row else None
            root_state = dict(root.state or {}) if root else {}
            medium = dict(root_state.get("tree_memory") or {})

            # ── Branch progress ──────────────────────────────────────────
            # Expose what each branch of this session (parent view) — or each
            # sibling branch (branch view) — has explored. Without this the
            # planner cannot answer questions like "what did we find in the
            # branched chats?": memory_recall would return 0/0/0 even when
            # the parent's state already carries fresh branch summaries.
            self_branches = dict(state.get("branch_summaries") or {})
            parent_branches: dict = {}
            if row and row.parent_session_id:
                parent = await ctx.db.get(AssistantSession, row.parent_session_id)
                if parent is not None:
                    pstate = dict(parent.state or {})
                    parent_branches = {
                        bid: e for bid, e in (pstate.get("branch_summaries") or {}).items()
                        if bid != str(ctx.session_id)  # exclude self
                    }
            # Merge: self branches first (this node's direct children), then
            # sibling branches from the parent. Caller filter by ``query``
            # still applies so the planner can scope by topic.
            merged: dict = {}
            merged.update(self_branches)
            for bid, e in parent_branches.items():
                merged.setdefault(bid, e)

            q_branches = (params.query or "").lower()
            for bid, entry in merged.items():
                summary = (entry.get("summary") or "").strip()
                title = (entry.get("title") or "Branch").strip()
                if not summary:
                    continue
                if q_branches and q_branches not in summary.lower() and q_branches not in title.lower():
                    continue
                branches_out[bid] = {
                    "title": title,
                    "summary": summary,
                    "updated_at": entry.get("updated_at"),
                    "last_message_id": entry.get("last_message_id"),
                }
            # Backwards compat: pick up legacy ``memory`` writes from before
            # the tree-memory migration. Treat them as tree-tier so they
            # surface to the planner; never write back to the legacy key.
            legacy = dict(root_state.get("memory") or {})
            for k, v in legacy.items():
                medium.setdefault(k, v)
            # long = namespace-wide
            long_mem = dict(state.get("ns_memory") or {})
            if not long_mem and root:
                long_mem = dict(root_state.get("ns_memory") or {})

            # Apply filters
            q = (params.query or "").lower()
            t = (params.memory_type or "").lower().strip()

            def _matches(k: str, entry: object) -> bool:
                if t and _entry_type(entry) != t:
                    return False
                if q:
                    return q in k.lower() or q in _entry_value(entry).lower()
                return True

            short = {k: v for k, v in short.items() if _matches(k, v)}
            medium = {k: v for k, v in medium.items() if _matches(k, v)}
            long_mem = {k: v for k, v in long_mem.items() if _matches(k, v)}

            # Sort by recency (newest first) so the planner/synth see the
            # most relevant context first, and preferences float above
            # everything else because they're durably-useful.
            def _sort_key(item: tuple[str, object]) -> tuple[int, str]:
                _, v = item
                pref_first = 0 if _entry_type(v) == "preference" else 1
                return (pref_first, _entry_ts(v) or "")

            short_items = sorted(short.items(), key=_sort_key, reverse=True)[:20]
            medium_items = sorted(medium.items(), key=_sort_key, reverse=True)[:30]
            long_items = sorted(long_mem.items(), key=_sort_key, reverse=True)[:30]

            # Semantic blend: when the caller supplied a meaningful query,
            # re-rank tree + namespace memory by embedding cosine similarity
            # to the query, then fuse with the recency ordering above via
            # reciprocal rank fusion. Falls back transparently to the
            # recency-only result when the embedder is unavailable.
            sem_query = (params.query or "").strip()
            if sem_query and len(sem_query) >= 6:
                try:
                    from app.assistant.semantic_memory import (
                        blend_with_recency,
                        semantically_rank,
                    )
                    sem_medium = await semantically_rank(
                        query=sem_query,
                        entries=medium,
                        session_id=ctx.session_id,
                        top_k=30,
                    )
                    fused_medium = blend_with_recency(
                        sem_medium,
                        medium_items,
                        top_k=30,
                    )
                    if fused_medium:
                        medium_items = [(k, v) for k, v, _ in fused_medium]
                    sem_long = await semantically_rank(
                        query=sem_query,
                        entries=long_mem,
                        session_id=ctx.session_id,
                        top_k=30,
                    )
                    fused_long = blend_with_recency(
                        sem_long,
                        long_items,
                        top_k=30,
                    )
                    if fused_long:
                        long_items = [(k, v) for k, v, _ in fused_long]
                except Exception as exc:
                    log.debug("semantic recall blend skipped: %s", exc)

            def _fmt(item: tuple[str, object]) -> dict:
                k, v = item
                t = _entry_type(v)
                base = {
                    "value": _entry_value(v),
                    "type": t,
                    "category": memory_category(t),
                    "ts": _entry_ts(v),
                }
                if isinstance(v, dict) and v.get("origin_session"):
                    base["origin_session"] = v["origin_session"]
                return base

            short_out = {k: _fmt((k, v)) for k, v in short_items}
            medium_out = {k: _fmt((k, v)) for k, v in medium_items}
            long_out = {k: _fmt((k, v)) for k, v in long_items}
        except Exception as exc:
            log.warning("memory_recall failed: %s", exc)
            short_out = {}
            medium_out, long_out = {}, {}

        await ctx.emit_progress(
            100,
            f"Recalled {len(short_out)} chat + {len(medium_out)} tree + "
            f"{len(long_out)} namespace + {len(branches_out)} branch memories",
        )
        return ToolResult(
            output={
                "short": short_out, "medium": medium_out, "long": long_out,
                "branches": branches_out,
                "total_short": len(short_out),
                "total_medium": len(medium_out),
                "total_long": len(long_out),
                "total_branches": len(branches_out),
            },
            summary=(
                f"{len(short_out)} chat · {len(medium_out)} tree · "
                f"{len(long_out)} namespace · {len(branches_out)} branches"
            ),
        )


# ── Delete ─────────────────────────────────────────────────────────────────────


class MemoryDeleteInput(BaseModel):
    key: str = Field(min_length=1, max_length=120, description="Key of the memory entry to remove.")
    scope: str = Field(
        default="medium",
        pattern="^(short|chat|session|medium|tree|long|namespace|ns)$",
        description="Which memory tier to delete from: short | medium | long.",
    )


class MemoryDeleteOutput(BaseModel):
    deleted: bool
    scope: str
    key: str


class MemoryDeleteTool:
    """Remove a stale or incorrect memory entry."""

    name = "memory_delete"
    summary = (
        "Delete a specific memory entry by key. Use when a stored fact is outdated, "
        "incorrect, or no longer relevant. Provide the exact key and scope used when the entry was written."
    )
    cost_class = "cheap"
    side_effects = True
    cancellable = False
    streamable = False
    input_schema = MemoryDeleteInput
    output_schema = MemoryDeleteOutput

    async def run(self, ctx: ToolContext, params: MemoryDeleteInput) -> ToolResult:
        tier = _SCOPE_ALIASES.get(params.scope, "medium")
        bucket = _SCOPE_TO_BUCKET[tier]
        norm_key = _normalize_key(params.key)
        await ctx.emit_progress(50, f"Deleting {tier}-tier memory: {norm_key!r}")
        deleted = False
        try:
            async with ctx.db.begin_nested():
                current = await ctx.db.get(AssistantSession, ctx.session_id)
                if current is None:
                    return ToolResult(
                        output={"deleted": False, "scope": tier, "key": norm_key},
                        summary="session not found",
                    )
                target = await _resolve_root_session(ctx.db, ctx.session_id) if tier == "medium" else current
                if target is None:
                    target = current
                state = dict(target.state or {})
                mem = dict(state.get(bucket) or {})
                if norm_key in mem:
                    del mem[norm_key]
                    state[bucket] = mem
                    target.state = state
                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(target, "state")
                    await ctx.db.flush()
                    deleted = True
        except Exception as exc:
            log.warning("memory_delete failed: %s", exc)

        await ctx.emit_progress(100, "Memory entry removed" if deleted else "Key not found")
        return ToolResult(
            output={"deleted": deleted, "scope": tier, "key": norm_key},
            summary=f"Deleted {tier}-tier memory: {norm_key!r}" if deleted else f"Key not found: {norm_key!r}",
        )


memory_write_tool = MemoryWriteTool()
memory_recall_tool = MemoryRecallTool()
memory_delete_tool = MemoryDeleteTool()
