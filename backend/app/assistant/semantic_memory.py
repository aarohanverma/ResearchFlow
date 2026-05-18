"""Embedding-ranked memory recall.

The keyword + recency filter on ``memory_recall`` works for fresh facts
with predictable key names, but deeply buried entries become unreachable
once the bucket fills up — the agent forgets effective knowledge even
though it's still stored. This module adds a second, semantic recall
path on top of the existing keyword/recency one:

* For a given query, embed it once.
* Lazily embed each memory entry's value and cache the vector keyed by
  ``(entry_key, content_hash)`` inside ``session.state["memory_embeddings"]``
  so future turns hit the cache instead of paying for a fresh embedding.
* Score each entry by cosine similarity to the query embedding.
* Blend semantic and recency rankings via Reciprocal-Rank-Fusion so the
  result list balances "freshness" with "actually relevant to what the
  user just asked."

The module is strictly additive — when the embedding adapter is
unavailable, every helper returns its keyword/recency fallback so the
recall pipeline still produces output. None of the callers must change
their failure handling.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any
from uuid import UUID

from sqlalchemy.orm.attributes import flag_modified

from app.db.session import async_session_factory
from app.models.assistant import AssistantSession

log = logging.getLogger(__name__)


_CACHE_KEY = "memory_embeddings"
_CACHE_MAX_ENTRIES = 400   # hard cap on stored vectors per session
_RRF_K = 60                # standard RRF damping constant


# ── Helpers ──────────────────────────────────────────────────────────────────


def _hash_content(value: str) -> str:
    """Stable short hash for cache invalidation when an entry's value changes."""
    return hashlib.sha1((value or "").encode("utf-8")).hexdigest()[:16]


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity, safe on empty/mismatched inputs."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _entry_value(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("value") or "")
    return str(entry or "")


def _entry_ts(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("ts") or "")
    return ""


async def _get_embedder():
    """Return the embedding adapter or None if unavailable."""
    try:
        from app.adapters.embedding import get_embedding_adapter
        return get_embedding_adapter()
    except Exception as exc:
        log.debug("semantic memory: embedding adapter unavailable: %s", exc)
        return None


# ── Public API ───────────────────────────────────────────────────────────────


async def semantically_rank(
    *,
    query: str,
    entries: dict[str, dict],
    session_id: UUID | str,
    top_k: int = 12,
) -> list[tuple[str, dict, float]]:
    """Return entries scored by relevance to ``query``.

    ``entries`` is the standard {key: {value,type,ts,...}} dict the memory
    tools surface. The function embeds each entry's value lazily, caches
    the vector on the session, then returns the top-K by cosine sim.

    Falls back to a recency-ordered slice when the embedder is offline,
    so callers can blend or use the output unchanged in either path.
    """
    if not query or not entries:
        return []

    embedder = await _get_embedder()
    if embedder is None:
        # Fallback — return recency-sorted slice, score=0 so callers can
        # detect we did not actually rank by similarity.
        sorted_items = sorted(
            entries.items(),
            key=lambda kv: _entry_ts(kv[1]) or "",
            reverse=True,
        )
        return [(k, v, 0.0) for k, v in sorted_items[:top_k]]

    # Embed the query once.
    try:
        q_vec = await embedder.embed_query(query)
    except Exception as exc:
        log.debug("semantic memory: query embedding failed: %s", exc)
        return [
            (k, v, 0.0)
            for k, v in list(entries.items())[:top_k]
        ]

    # Pull / refresh per-entry embeddings.
    cache = await _load_cache(session_id)
    scored: list[tuple[str, dict, float]] = []
    fresh: dict[str, dict] = {}
    needs_persist = False

    # Collect entries that need embedding so we can issue a single batch.
    pending: list[tuple[str, str, str]] = []  # (key, value_text, content_hash)
    for key, entry in entries.items():
        value = _entry_value(entry)
        if not value.strip():
            continue
        chash = _hash_content(value)
        cached = cache.get(key) or {}
        if cached.get("hash") == chash and cached.get("vec"):
            fresh[key] = cached
        else:
            pending.append((key, value[:1200], chash))

    if pending:
        try:
            texts = [v for _, v, _ in pending]
            vecs = await embedder.embed_texts(texts, task_type="SEMANTIC_SIMILARITY")
            for (k, _val, chash), vec in zip(pending, vecs or []):
                if not vec:
                    continue
                fresh[k] = {"hash": chash, "vec": list(vec)}
                cache[k] = fresh[k]
                needs_persist = True
        except Exception as exc:
            log.debug("semantic memory: batch embed failed: %s", exc)

    for key, entry in entries.items():
        vec_entry = fresh.get(key)
        if not vec_entry:
            continue
        sim = _cosine(q_vec, vec_entry["vec"])
        scored.append((key, entry, sim))

    scored.sort(key=lambda t: t[2], reverse=True)

    if needs_persist:
        # Trim cache to cap before persisting.
        if len(cache) > _CACHE_MAX_ENTRIES:
            cache = dict(sorted(cache.items())[: _CACHE_MAX_ENTRIES])
        await _persist_cache(session_id, cache)

    return scored[:top_k]


def blend_with_recency(
    semantic: list[tuple[str, dict, float]],
    recency: list[tuple[str, dict]],
    *,
    top_k: int = 12,
) -> list[tuple[str, dict, float]]:
    """Fuse semantic + recency rankings via reciprocal rank fusion.

    RRF is robust to score-scale differences: each list contributes its
    own rank-based weight and the merged order surfaces entries that
    rank well on EITHER signal. Falls back gracefully when one list is
    empty.
    """
    if not semantic and not recency:
        return []
    if not semantic:
        return [(k, v, 0.0) for k, v in recency[:top_k]]
    if not recency:
        return semantic[:top_k]

    fused: dict[str, dict[str, Any]] = {}
    for rank, (k, v, sim) in enumerate(semantic):
        slot = fused.setdefault(k, {"entry": v, "rrf": 0.0, "best_sim": 0.0})
        slot["rrf"] += 1.0 / (_RRF_K + rank + 1)
        slot["best_sim"] = max(slot["best_sim"], sim)
    for rank, (k, v) in enumerate(recency):
        slot = fused.setdefault(k, {"entry": v, "rrf": 0.0, "best_sim": 0.0})
        slot["rrf"] += 1.0 / (_RRF_K + rank + 1)

    ordered = sorted(fused.items(), key=lambda kv: kv[1]["rrf"], reverse=True)
    return [(k, v["entry"], v["best_sim"]) for k, v in ordered[:top_k]]


# ── Internal cache helpers ───────────────────────────────────────────────────


async def _load_cache(session_id: UUID | str) -> dict[str, dict]:
    try:
        async with async_session_factory() as db:
            sid = UUID(str(session_id)) if not isinstance(session_id, UUID) else session_id
            row = await db.get(AssistantSession, sid)
            if row is None:
                return {}
            return dict((row.state or {}).get(_CACHE_KEY) or {})
    except Exception as exc:
        log.debug("semantic memory: cache load failed: %s", exc)
        return {}


async def _persist_cache(session_id: UUID | str, cache: dict[str, dict]) -> None:
    try:
        async with async_session_factory() as db:
            sid = UUID(str(session_id)) if not isinstance(session_id, UUID) else session_id
            row = await db.get(AssistantSession, sid)
            if row is None:
                return
            state = dict(row.state or {})
            state[_CACHE_KEY] = cache
            row.state = state
            flag_modified(row, "state")
            await db.commit()
    except Exception as exc:
        log.debug("semantic memory: cache persist failed: %s", exc)
