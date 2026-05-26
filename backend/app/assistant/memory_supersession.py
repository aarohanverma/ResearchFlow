"""Automatic supersession detection for long-term memory.

When a new memory entry lands that's semantically near-identical to
an existing one, the old entry is marked ``superseded`` so the
planner / synthesizer recall pipeline stops surfacing it. The newer
write is kept; the older row stays in storage (and in the audit log)
so the user can still inspect / restore it from Settings → Memory.

Why this matters:

  * The user explicitly listed "automatic supersession" as a gap.
  * Without it, repeated writes of the same fact (e.g. user preference
    paraphrased slightly differently across sessions) pile up and the
    planner sees N copies of the same idea. That bloats the prompt
    and dilutes signal.

Constraints:

  * Soft — uses an embedding similarity threshold; never deletes the
    older entry, just flags it.
  * Bounded — compares only against the most-recent N entries in the
    same tier + namespace. A full scan would be O(N²) per write.
  * Best-effort — if the embedding adapter is offline or the
    comparison fails, the write proceeds without supersession (the
    old entry just stays active alongside the new one, which is the
    pre-feature behaviour).
  * Class-aware — different memory classes (semantic / episodic /
    procedural) never supersede each other. A new procedure does
    NOT supersede a similar finding even when their text overlaps.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.assistant.tools.memory import _entry_type, _entry_value, memory_category

log = logging.getLogger(__name__)


# Infrastructure-safety timeout on the embedding round-trip. NOT a
# quality budget — this prevents a hung embedding provider from
# holding the per-session ``state_lock`` indefinitely. Without the
# guard, other concurrent writes to the same session (auto-memory
# consolidation, telemetry, branch summaries) would block on the
# 30s lock_timeout before bailing. 10s is well above typical
# embedding latency (200ms-2s) so legitimate calls complete; a
# truly stuck provider trips the guard and supersession silently
# skips for this write.
_EMBED_TIMEOUT_S = 10.0


# Cosine similarity at which two entries are considered the SAME
# idea. 0.88 is high enough that a generic "user prefers technical
# answers" doesn't accidentally supersede an unrelated finding, and
# low enough that paraphrases of the same preference catch each other.
_SIMILARITY_THRESHOLD = 0.88

# Maximum number of candidate entries to compare against per write.
# Bounds the cost — sorted by recency, so the freshest entries get
# the comparison and ancient ones are skipped.
_MAX_CANDIDATES = 30

# Entry types that NEVER trigger supersession even when similar:
#
#   * ``hypothesis`` — research hypotheses can be coherent variations;
#     supersession would silently drop one branch of the user's thinking.
#   * ``context`` — catch-all fallback type. The class is too vague to
#     trust similarity scores; let the user decide via History modal.
_SUPERSESSION_EXEMPT_TYPES: frozenset[str] = frozenset({"hypothesis", "context"})


async def detect_and_mark_supersessions(
    *,
    bucket: dict[str, Any],
    new_key: str,
    new_value: str,
    new_type: str,
    session_id: Any,
) -> list[str]:
    """Scan ``bucket`` for entries the new write supersedes.

    Returns a list of keys that were marked ``superseded`` so the
    caller can record the action in the audit log. The bucket is
    mutated in place: each superseded entry gains
    ``superseded_by_key`` + ``superseded_at`` fields so recall paths
    can filter them out.

    Failure modes — all degrade silently to "no supersession":

      * Embedder unavailable
      * Similarity computation raised
      * New entry's type is exempt
      * Bucket has fewer than 2 entries (nothing could be superseded)

    Args:
        bucket: The live memory dict (medium/tree or long/namespace).
            Mutated in place when a supersession is detected.
        new_key: The key the new entry was just written to.
        new_value: The new entry's value text — embedded for the
            similarity check.
        new_type: The new entry's memory type. Used to enforce
            class-coherence (a new procedure doesn't supersede a
            similar finding).
        session_id: Originating session — passed to the embedder so
            per-session caching of vectors works.

    Returns:
        List of keys that were marked superseded. Empty when nothing
        crossed the threshold or when supersession was skipped.
    """
    if (new_type or "").lower() in _SUPERSESSION_EXEMPT_TYPES:
        return []
    if not bucket or len(bucket) < 2:
        return []
    if not new_value or not new_value.strip():
        return []

    # Class-coherence: only compare against entries of the SAME
    # cognitive class. A new ``preference`` doesn't supersede a
    # ``finding`` even when their text overlaps, because they play
    # different roles in the planner.
    new_class = memory_category(new_type or "")
    candidates: list[tuple[str, dict]] = []
    for k, v in bucket.items():
        if k == new_key:
            continue
        if not isinstance(v, dict):
            continue
        if v.get("superseded_by_key"):
            continue  # already superseded — don't double-tag
        v_type = _entry_type(v)
        if (v_type or "").lower() in _SUPERSESSION_EXEMPT_TYPES:
            continue
        # Class match — when new_class is "-" (no clean mapping),
        # require an exact type match to be conservative.
        v_class = memory_category(v_type or "")
        if new_class == "-" or v_class == "-":
            if v_type != new_type:
                continue
        elif v_class != new_class:
            continue
        candidates.append((k, v))

    if not candidates:
        return []

    # Bound the comparison set — newest first.
    candidates.sort(
        key=lambda kv: kv[1].get("ts") or "",
        reverse=True,
    )
    candidates = candidates[:_MAX_CANDIDATES]

    # Lazy import of the embedder + semantic helpers so a build
    # without the embedding adapter (CI, fresh checkout) doesn't
    # explode at import time. Imports the PUBLIC ``cosine_similarity``
    # helper rather than the private ``_cosine`` alias so this
    # cross-module dependency uses a stable, documented surface.
    try:
        from app.assistant.semantic_memory import _get_embedder, cosine_similarity
    except Exception as exc:
        log.debug("supersession: semantic_memory import failed: %s", exc)
        return []

    embedder = await _get_embedder()
    if embedder is None:
        return []

    try:
        # Embed in ONE batch — new value + all candidate values — so
        # we pay a single provider round-trip. Wall-clock guard
        # prevents a hung provider from holding the surrounding
        # ``state_lock`` indefinitely (see _EMBED_TIMEOUT_S docstring).
        texts: list[str] = [new_value[:1200]]
        for _, v in candidates:
            cand_val = _entry_value(v)
            texts.append(cand_val[:1200] if cand_val else "")
        vecs = await asyncio.wait_for(
            embedder.embed_texts(texts, task_type="SEMANTIC_SIMILARITY"),
            timeout=_EMBED_TIMEOUT_S,
        )
        if not vecs or len(vecs) != len(texts):
            return []
    except asyncio.TimeoutError:
        log.warning(
            "supersession: embed timed out after %.1fs — skipping for this write",
            _EMBED_TIMEOUT_S,
        )
        return []
    except Exception as exc:
        log.debug("supersession: batch embed failed: %s", exc)
        return []

    new_vec = vecs[0]
    if not new_vec:
        return []

    superseded: list[str] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for (cand_key, cand_entry), cand_vec in zip(candidates, vecs[1:]):
        if not cand_vec:
            continue
        try:
            sim = cosine_similarity(new_vec, cand_vec)
        except Exception:
            continue
        if sim >= _SIMILARITY_THRESHOLD:
            # Mutate the bucket in place — mark the OLD entry
            # superseded. The user can still see / restore it via
            # Settings → Memory.
            cand_entry["superseded_by_key"] = new_key
            cand_entry["superseded_at"] = now_iso
            cand_entry["superseded_similarity"] = round(float(sim), 4)
            superseded.append(cand_key)
    return superseded


__all__ = ["detect_and_mark_supersessions"]
