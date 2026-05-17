"""Update ``UserInterestProfile.concept_affinity`` from assistant turns.

Every completed turn yields signals about what the user actually cares
about — both the question they asked and the papers the orchestrator
surfaced (especially the ones it cited). This module folds those signals
into the user's interest profile so subsequent frontier_scan / deep_search
runs can soft-bias toward areas they're investigating.

Design choices:

* Soft, exponential decay so a single off-topic question doesn't poison
  the profile and the user's evolving interests are tracked over time.
* Cited papers count more than merely retrieved papers (cited = used).
* Question keywords are noisier than paper concepts — weighted lower.
* Bounded growth: weights clamp at [0, 5] so a runaway loop can't blow
  the JSON column.
* Pure DB writes — no LLM calls — so this can run fire-and-forget after
  every turn without latency or cost overhead.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable
from uuid import UUID

from sqlalchemy import select

from app.db.session import async_session_factory
from app.models.paper import Paper
from app.models.user import UserInterestProfile

log = logging.getLogger(__name__)


# Tuned constants — see _new_weight() for how they combine.
_DECAY = 0.92        # 8% decay per turn — concepts fade if not reinforced.
_CITED_GAIN = 0.40   # Per appearance among the cited papers.
_RETRIEVED_GAIN = 0.10  # Per appearance among non-cited retrieved papers.
_QUERY_GAIN = 0.05   # Per word match in the user's question text.
_MAX_WEIGHT = 5.0
_PRUNE_BELOW = 0.05  # Below this floor, drop the entry to keep the JSON tidy.
_TRACK_LIMIT = 200   # Cap the number of tracked concepts per user.


async def update_from_turn(
    *,
    user_id: UUID,
    user_query: str,
    cited_paper_ids: Iterable[str],
    retrieved_papers: Iterable[dict],
) -> None:
    """Roll forward a user's concept_affinity from one assistant turn.

    Never raises — wrapped in try/except so callers can ``asyncio.create_task``
    this without worrying about background task crashes.
    """
    try:
        cited = {str(pid) for pid in cited_paper_ids if pid}
        retrieved = list(retrieved_papers or [])
        if not cited and not retrieved and not (user_query or "").strip():
            return

        async with async_session_factory() as db:
            # Pull cited paper concepts in one query (when we have IDs).
            cited_concepts: list[str] = []
            if cited:
                from uuid import UUID as _UUID

                cited_uuids: list[_UUID] = []
                for pid in cited:
                    try:
                        cited_uuids.append(_UUID(pid))
                    except ValueError:
                        continue
                if cited_uuids:
                    res = await db.execute(
                        select(Paper.id, Paper.key_concepts).where(Paper.id.in_(cited_uuids))
                    )
                    for _id, concepts in res.fetchall():
                        cited_concepts.extend(concepts or [])

            # Concepts from retrieved-but-not-cited papers carry weaker signal.
            retrieved_concepts: list[str] = []
            for p in retrieved:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("paper_id") or "")
                if pid in cited:
                    # Already counted with stronger weight above.
                    continue
                retrieved_concepts.extend(p.get("key_concepts") or [])

            query_tokens = _query_tokens(user_query)

            # Aggregate per-concept gains for this turn.
            gains: Counter[str] = Counter()
            for c in cited_concepts:
                gains[_normalize(c)] += _CITED_GAIN
            for c in retrieved_concepts:
                gains[_normalize(c)] += _RETRIEVED_GAIN
            for c in query_tokens:
                gains[_normalize(c)] += _QUERY_GAIN
            gains.pop("", None)

            # Load + decay + reinforce + prune.
            profile_row = await db.execute(
                select(UserInterestProfile).where(UserInterestProfile.user_id == user_id)
            )
            profile = profile_row.scalar_one_or_none()
            if profile is None:
                profile = UserInterestProfile(user_id=user_id, concept_affinity={})
                db.add(profile)

            current = dict(profile.concept_affinity or {})
            decayed = {k: float(v) * _DECAY for k, v in current.items()}
            for concept, gain in gains.items():
                decayed[concept] = min(_MAX_WEIGHT, decayed.get(concept, 0.0) + gain)

            # Prune low-weight noise + cap the dict size.
            tidied = {k: round(v, 4) for k, v in decayed.items() if v >= _PRUNE_BELOW}
            if len(tidied) > _TRACK_LIMIT:
                top = sorted(tidied.items(), key=lambda kv: kv[1], reverse=True)[:_TRACK_LIMIT]
                tidied = dict(top)

            profile.concept_affinity = tidied
            await db.commit()
    except Exception as exc:
        log.warning("interest profile update failed user=%s: %s", user_id, exc)


def _normalize(concept: str) -> str:
    """Lowercase + collapse whitespace for stable bucketing."""
    return " ".join((concept or "").lower().split())[:80]


def _query_tokens(text: str) -> list[str]:
    """Naive but useful: lower-case multi-word phrases of length >= 4 chars."""
    if not text:
        return []
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() or ch == "-" else " " for ch in text.lower())
    raw = [w for w in cleaned.split() if len(w) >= 4 and not w.isdigit()]
    # Cap to keep the gain bounded.
    return raw[:30]
