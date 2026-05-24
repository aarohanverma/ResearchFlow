"""Background memory consolidation — cluster + merge across all users.

The per-tier eviction caps in :mod:`app.assistant.tools.memory` keep
the stores bounded, but eviction *loses* information. Consolidation
*compresses* it: a cluster of related entries (e.g. five "user said
they prefer terse answers" / "user is a senior researcher" /
"user dislikes long preambles" entries) gets merged into one
higher-level summary that captures the gestalt without burning five
slots.

This module is the production implementation:

  * **Clustering**: pure embedding-similarity cosine over each tier's
    entries per user. We reuse the existing embedding adapter so the
    consolidation pass adds no new dependency.
  * **LLM merge**: one cheap-model structured call per cluster (≥ 3
    entries) that produces a single consolidated entry. Bounded cost.
  * **Provenance**: the new entry carries ``consolidated_from``
    (the original keys) and ``consolidation_ts`` so the recall view
    can show "this fact was consolidated from N earlier entries".
  * **Idempotency**: an entry already produced by consolidation (its
    key starts with ``consolidated_``) is never re-clustered. Stale
    consolidated entries (TTL-flagged) get re-considered on the next
    pass.

The pass is invoked from APScheduler ``memory_consolidation_weekly``
and can also be triggered manually via the ``consolidate_memory_for_user``
helper for tests / admin tools.

What it does NOT do:

  * **Doesn't write durable memory mid-turn** — only the post-turn
    auto-memory + this cron are allowed to mutate ``state[*_memory]``.
    The ReAct loop itself stays a read-only consumer.
  * **Doesn't merge across users / sessions** — consolidation is
    strictly within one (user, session_root) for chat/tree tiers and
    one (user, namespace) for ns_memory. User boundaries are
    inviolable; tree boundaries preserve branch context.
  * **Doesn't auto-delete originals** by default. The originals get
    a ``consolidated_into`` pointer; eviction continues to handle
    actual deletion. This keeps the audit trail intact for one cycle.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.assistant.state_lock import session_state_lock
from app.assistant.tools.memory import _memory_is_stale, _normalize_key
from app.models.assistant import AssistantSession

log = logging.getLogger(__name__)


# ── Tuneables ─────────────────────────────────────────────────────────────────


# Minimum number of entries in a cluster before consolidation fires.
# Smaller clusters don't pay back the LLM call cost; ≥3 entries is the
# inflection where merging actually reduces information per slot.
_MIN_CLUSTER_SIZE = 3

# Cosine-similarity threshold for cluster membership. 0.78 is high
# enough that "user prefers terse answers" + "user dislikes verbose
# preambles" cluster together but "user is a senior researcher" stays
# separate. Tuned conservatively — over-aggressive clustering is the
# failure mode that destroys recall accuracy.
_CLUSTER_COSINE_THRESHOLD = 0.78

# Cap LLM calls per pass. Five clusters per (user, scope) bounds
# both latency and cost — a pathological store with dozens of
# tight clusters will only consolidate the top-5 most-similar ones
# this cycle, the rest next cycle.
_MAX_CLUSTERS_PER_SCOPE = 5

# Entries with origin timestamps newer than this many days are
# skipped — they haven't had time to demonstrate their durability,
# and the consolidation pass is meant for facts that survived a few
# turns / sessions.
_MIN_AGE_DAYS = 1


# ── Public API ───────────────────────────────────────────────────────────────


@dataclass
class ConsolidationReport:
    """One pass's accounting — surfaced in logs + diagnostic endpoints."""

    sessions_scanned: int = 0
    clusters_found: int = 0
    consolidations_written: int = 0
    entries_merged: int = 0
    skipped_too_few: int = 0
    skipped_too_recent: int = 0
    llm_failures: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"sessions_scanned={self.sessions_scanned} "
            f"clusters_found={self.clusters_found} "
            f"consolidations={self.consolidations_written} "
            f"merged={self.entries_merged} "
            f"skipped_too_few={self.skipped_too_few} "
            f"skipped_too_recent={self.skipped_too_recent} "
            f"llm_failures={self.llm_failures} "
            f"errors={len(self.errors)}"
        )


async def consolidate_memory_for_user(
    db: AsyncSession,
    user_id: Any,
    *,
    report: ConsolidationReport | None = None,
) -> ConsolidationReport:
    """Run a consolidation pass for one user across all their session
    trees. Returns the report (also mutated in place when caller passes
    one in)."""
    if report is None:
        report = ConsolidationReport()

    sessions = await _load_user_sessions(db, user_id)
    report.sessions_scanned += len(sessions)
    for session in sessions:
        try:
            await _consolidate_session(db, session, report=report)
        except Exception as exc:  # noqa: BLE001
            log.exception("memory_consolidation: session=%s failed", session.id)
            report.errors.append(f"session {session.id}: {exc!s}")
    return report


async def consolidate_all_users(db: AsyncSession) -> ConsolidationReport:
    """Cron entry point — iterate over every user with at least one
    session, consolidate their memory.

    Bounded by ``_MAX_CLUSTERS_PER_SCOPE`` per (user, scope) so a
    pathological store can't dominate one cycle. Returns the aggregate
    report.
    """
    report = ConsolidationReport()
    user_ids = await _load_distinct_user_ids(db)
    log.info("memory_consolidation: starting pass for %d user(s)", len(user_ids))
    for uid in user_ids:
        try:
            await consolidate_memory_for_user(db, uid, report=report)
        except Exception as exc:  # noqa: BLE001
            log.exception("memory_consolidation: user=%s failed", uid)
            report.errors.append(f"user {uid}: {exc!s}")
    log.info("memory_consolidation: pass complete — %s", report.summary())
    return report


# ── Internals: session traversal ────────────────────────────────────────────


async def _load_distinct_user_ids(db: AsyncSession) -> list[Any]:
    """Distinct user IDs that have at least one assistant session with
    a non-empty memory tier. We filter at the SQL level to keep the
    pass cheap on accounts with thousands of empty sessions."""
    stmt = (
        select(AssistantSession.user_id)
        .where(AssistantSession.state.is_not(None))
        .distinct()
    )
    result = await db.execute(stmt)
    return [row[0] for row in result.all() if row[0] is not None]


async def _load_user_sessions(db: AsyncSession, user_id: Any) -> list[AssistantSession]:
    """All sessions for the user that have any persisted state. We
    only consolidate session-tree roots (sessions with no parent) so
    branch sessions don't get their own tree-tier rewrite — the root
    holds the authoritative tree_memory."""
    stmt = (
        select(AssistantSession)
        .where(AssistantSession.user_id == user_id)
        .where(AssistantSession.parent_session_id.is_(None))
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _consolidate_session(
    db: AsyncSession,
    session: AssistantSession,
    *,
    report: ConsolidationReport,
) -> None:
    """Consolidate every relevant tier on one session row.

    We hold the session lock for the duration so concurrent
    ``memory_write`` calls don't observe a half-written state.
    """
    async with session_state_lock(session.id):
        await db.refresh(session)
        raw_state = dict(session.state or {})
        changed = False

        # Each tier is consolidated independently. Branch summaries
        # and history summaries are NOT consolidated — they're already
        # higher-order summaries.
        for tier_key in ("chat_memory", "tree_memory", "ns_memory"):
            tier = raw_state.get(tier_key)
            if not isinstance(tier, dict) or len(tier) < _MIN_CLUSTER_SIZE:
                report.skipped_too_few += 1
                continue
            updated = await _consolidate_tier(tier, report=report, tier_key=tier_key)
            if updated is not None:
                raw_state[tier_key] = updated
                changed = True

        if changed:
            session.state = raw_state
            flag_modified(session, "state")
            await db.flush()
            await db.commit()


# ── Internals: clustering + merging ─────────────────────────────────────────


async def _consolidate_tier(
    tier: dict[str, Any],
    *,
    report: ConsolidationReport,
    tier_key: str,
) -> dict[str, Any] | None:
    """Cluster + merge one memory tier. Returns the updated tier dict,
    or None when nothing changed.

    We never DELETE the original entries here — the existing eviction
    policy in ``memory_write`` handles that. We just add the
    consolidated entry alongside, with provenance pointing back to
    the originals so recall can show the lineage.
    """
    candidates = _consolidation_candidates(tier)
    if len(candidates) < _MIN_CLUSTER_SIZE:
        report.skipped_too_few += 1
        return None

    try:
        clusters = await _cluster_by_embedding(candidates)
    except Exception as exc:  # noqa: BLE001
        log.warning("memory_consolidation: clustering failed: %s", exc)
        report.errors.append(f"cluster: {exc!s}")
        return None

    clusters = [c for c in clusters if len(c) >= _MIN_CLUSTER_SIZE]
    clusters = clusters[:_MAX_CLUSTERS_PER_SCOPE]
    if not clusters:
        return None
    report.clusters_found += len(clusters)

    updated = dict(tier)
    now = datetime.now(timezone.utc).isoformat()
    for cluster in clusters:
        try:
            merged_value, merged_type = await _llm_merge(cluster)
        except Exception as exc:  # noqa: BLE001
            log.warning("memory_consolidation: LLM merge failed: %s", exc)
            report.llm_failures += 1
            continue
        if not merged_value:
            continue
        source_keys = [k for k, _v in cluster]
        consolidated_key = _consolidated_key(tier_key, source_keys)
        updated[consolidated_key] = {
            "value": merged_value,
            "type": merged_type or "context",
            "ts": now,
            "version": 1,
            "consolidated_from": source_keys,
            "consolidation_ts": now,
        }
        # Tag the originals with a back-reference so the recall view
        # can show "consolidated into X" without breaking existing
        # readers (they ignore unknown keys).
        for source_key in source_keys:
            original = updated.get(source_key)
            if isinstance(original, dict):
                original = dict(original)
                original["consolidated_into"] = consolidated_key
                updated[source_key] = original
        report.consolidations_written += 1
        report.entries_merged += len(source_keys)

    return updated if report.consolidations_written > 0 else None


def _consolidation_candidates(tier: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Entries eligible for clustering: dict-shaped, content-bearing,
    not already a consolidated rollup, and old enough to be durable.

    Stale-TTL'd entries are eligible (they need re-evaluation anyway).
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    out: list[tuple[str, dict[str, Any]]] = []
    for key, entry in tier.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("consolidated_from"):
            continue        # already a rollup
        if entry.get("consolidated_into"):
            continue        # already merged into something else
        if not entry.get("value"):
            continue
        # Skip very recent entries; consolidation is for facts that
        # survived a few turns.
        ts_raw = entry.get("ts")
        if ts_raw:
            try:
                ts = datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
                if age_days < _MIN_AGE_DAYS and not _memory_is_stale(entry, now_iso=now_iso):
                    continue
            except ValueError:
                pass
        out.append((key, entry))
    return out


async def _cluster_by_embedding(
    candidates: list[tuple[str, dict[str, Any]]],
) -> list[list[tuple[str, dict[str, Any]]]]:
    """Greedy single-link clustering on embedding cosine similarity.

    Greedy + non-deterministic-order would be wrong; we sort
    candidates by key for stability so a given input always produces
    the same clusters. The algorithm is O(n²) on the candidate count
    which is fine — tiers are capped at 30-120 entries, so n is small.
    """
    from app.adapters.embedding import get_embedding_adapter

    candidates = sorted(candidates, key=lambda kv: kv[0])
    texts = [f"{k}: {v.get('value','')[:400]}" for k, v in candidates]

    adapter = get_embedding_adapter()
    vectors = await adapter.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    if not vectors or len(vectors) != len(candidates):
        return []

    # Normalise embedding vectors so we can use dot-product as cosine.
    normalised: list[list[float]] = []
    for vec in vectors:
        if not vec:
            normalised.append([])
            continue
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            normalised.append([])
            continue
        normalised.append([x / norm for x in vec])

    # Greedy single-link: walk in order, assign each candidate to the
    # first existing cluster whose centroid hits the threshold;
    # otherwise start a new cluster.
    clusters: list[list[int]] = []
    centroids: list[list[float]] = []

    def _dot(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        return sum(a[i] * b[i] for i in range(n))

    for i, vec in enumerate(normalised):
        if not vec:
            clusters.append([i])
            centroids.append(vec)
            continue
        placed = False
        for cluster_idx, centroid in enumerate(centroids):
            if _dot(vec, centroid) >= _CLUSTER_COSINE_THRESHOLD:
                clusters[cluster_idx].append(i)
                # Update centroid as running average; re-normalise.
                size = len(clusters[cluster_idx])
                centroids[cluster_idx] = _average_normalised(
                    [normalised[j] for j in clusters[cluster_idx]], size,
                )
                placed = True
                break
        if not placed:
            clusters.append([i])
            centroids.append(vec)
    return [[candidates[i] for i in c] for c in clusters]


def _average_normalised(vectors: list[list[float]], size: int) -> list[float]:
    """Compute the L2-normalised mean of a list of vectors."""
    if not vectors:
        return []
    dim = len(vectors[0])
    summed = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            if i < dim:
                summed[i] += x
    mean = [x / max(1, size) for x in summed]
    norm = math.sqrt(sum(x * x for x in mean))
    if norm == 0:
        return mean
    return [x / norm for x in mean]


async def _llm_merge(
    cluster: list[tuple[str, dict[str, Any]]],
) -> tuple[str, str]:
    """Send the cluster to a cheap model and ask for one merged entry.

    Returns ``(merged_value, merged_type)``. Empty string on failure
    or when the model produces unusable output.
    """
    from app.adapters.llm import get_llm_adapter

    schema = {
        "type": "object",
        "properties": {
            "merged_value": {"type": "string", "maxLength": 1500},
            "merged_type": {
                "type": "string",
                "enum": [
                    "finding", "concept", "hypothesis", "paper_note",
                    "preference", "context", "episode", "skill", "procedure",
                ],
            },
        },
        "required": ["merged_value"],
    }
    bullet_items = "\n".join(
        f"  - [{k}] ({v.get('type','context')}) {v.get('value','')[:400]}"
        for k, v in cluster
    )
    sys_msg = (
        "You are a research-assistant memory consolidator. Given a "
        "cluster of related memory entries from a single user, write "
        "ONE merged entry that captures the gestalt without losing "
        "load-bearing specifics. Rules:\n"
        "  - Preserve concrete names, paper IDs, dates, numbers.\n"
        "  - Drop redundancy and weakly-supported parts.\n"
        "  - Keep the merged_value under 400 chars.\n"
        "  - Pick the merged_type that best fits the consolidated content.\n"
        "  - If the cluster is genuinely incoherent (entries don't "
        "actually belong together), return an empty merged_value."
    )
    user_msg = (
        "Consolidate the following related memory entries into a single "
        "entry:\n\n"
        f"{bullet_items}\n\n"
        "Return the merged entry as structured JSON."
    )
    llm = get_llm_adapter()
    raw = await llm.complete_structured(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        llm.cheap_model,
        schema,
    )
    if not isinstance(raw, dict):
        return "", ""
    return (
        str(raw.get("merged_value") or "").strip()[:1500],
        str(raw.get("merged_type") or "context").strip(),
    )


def _consolidated_key(tier_key: str, source_keys: list[str]) -> str:
    """Stable, human-readable key for a consolidated entry. Prefixed
    so eviction policies + idempotency checks can recognise them
    without parsing the value."""
    parts = sorted(_normalize_key(k) for k in source_keys[:4])
    suffix = "_".join(parts)[:80]
    if not suffix:
        suffix = "rollup"
    return f"consolidated__{suffix}"
