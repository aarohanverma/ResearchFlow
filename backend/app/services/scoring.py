"""ScoringService — computes personalized paper scores in pure SQL (no LLM).

Score formula: ``score = novelty × ow + relevance × (1 - ow) ± concept_affinity``.
Concept affinity: +0.20 if paper.key_concepts ∩ hot_concepts ≠ ∅;
−0.15 if paper.key_concepts ∩ cold_concepts ≠ ∅.  Clamped to [0, 1].
"""

from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Paper, PaperOfDay
from app.models.user import ExpertiseLevel, Orientation, UserInterestProfile
from app.core.config import settings


_ORIENTATION_WEIGHT = {
    Orientation.research: 0.7,    # weight novelty higher
    Orientation.production: 0.3,  # weight relevance higher
    Orientation.both: 0.5,
}


class ScoringService:
    """Computes personalised paper scores in pure SQL — no LLM calls required.

    Scores are derived from ``novelty_score`` and ``relevance_score`` weighted
    by the user's orientation, with optional boosts/penalties based on their
    hot/cold subtopic preferences.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the service with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` used for paper queries.
        """
        self._db = db

    async def score_papers_for_user(
        self,
        user_id: UUID,
        namespace_key: str,
        orientation: Orientation = Orientation.both,
        hot_subtopics: list[str] | None = None,
        cold_subtopics: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Score and rank papers in a namespace for a specific user.

        Applies the orientation-weighted formula to every paper in the
        namespace, adds concept-based affinity boosts and penalties, then
        returns the top-``limit`` results sorted by descending score.

        Formula::

            score = novelty_score × ow + relevance_score × (1 − ow)
                    + 0.20  if paper.key_concepts ∩ hot_subtopics ≠ ∅
                    − 0.15  if paper.key_concepts ∩ cold_subtopics ≠ ∅
            score = clamp(score, 0.0, 1.0)

        The ``hot_subtopics`` / ``cold_subtopics`` lists contain concept strings
        (e.g. ``["transformer", "attention mechanism"]``) derived from liked or
        dismissed papers via ``POST /feed/feedback``.  Concept intersection is
        used (not namespace substring matching) so the signals accurately reflect
        the user's stated research interests.

        Args:
            user_id: UUID of the user requesting the feed.
            namespace_key: The arXiv-style namespace key whose papers to score
                (e.g. ``"cs.AI"``).
            orientation: User's research orientation, controls the weight split
                between ``novelty_score`` and ``relevance_score``. Defaults to
                ``Orientation.both`` (50/50).
            hot_subtopics: Concept strings the user wants to see more of;
                papers sharing any of these concepts receive a ``+0.20`` boost.
                Defaults to ``None`` (no boost).
            cold_subtopics: Concept strings the user wants to see less of;
                papers sharing any of these concepts receive a ``-0.15`` penalty.
                Defaults to ``None`` (no penalty).
            limit: Maximum number of top-scored papers to return. Defaults to
                ``50``.

        Returns:
            A list of dicts, each containing ``paper`` (the ``Paper`` ORM
            object), ``score`` (float clamped to ``[0.0, 1.0]``), and
            ``why_tag`` (a short human-readable label explaining the score).
        """
        ow = _ORIENTATION_WEIGHT[orientation]
        hot = hot_subtopics or []
        cold = cold_subtopics or []

        # Hot/cold boosts perturb the base score by at most ±0.20. Over-fetch
        # by ~3x so the post-boost re-sort can still surface a hot-tagged
        # paper that started a few rungs below the unboosted top-K — but
        # never pull the whole namespace into Python memory just to score+sort.
        fetch_n = max(limit * 3, 60)

        from sqlalchemy import desc as _desc

        base_expr = (Paper.novelty_score * ow) + (Paper.relevance_score * (1 - ow))
        result = await self._db.execute(
            select(Paper)
            .where(Paper.namespace_key == namespace_key)
            .order_by(_desc(base_expr))
            .limit(fetch_n)
        )
        papers = list(result.scalars())

        hot_set = set(hot)
        cold_set = set(cold)

        scored: list[dict] = []
        for p in papers:
            score = p.novelty_score * ow + p.relevance_score * (1 - ow)
            paper_concepts = set(p.key_concepts or [])
            if hot_set and paper_concepts & hot_set:
                score += 0.20
            if cold_set and paper_concepts & cold_set:
                score -= 0.15
            scored.append({
                "paper": p,
                "score": min(1.0, max(0.0, score)),
                "why_tag": self._why_tag(p, hot),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _why_tag(self, paper: Paper, hot_subtopics: list[str]) -> str:
        """Return the human-readable 'why this paper' tag for the feed card."""
        if paper.novelty_score > 0.8:
            return "🔬 High novelty"
        if paper.relevance_score > 0.8:
            return "🔧 Practical relevance"
        if paper.is_breakthrough:
            return "⚡ Breakthrough"
        return "🧠 In your interests"

    async def score_all(self, namespace_key: str) -> tuple[UUID | None, float]:
        """Score all papers in a namespace and return the best paper and its score.

        Uses a simple average of ``novelty_score`` and ``relevance_score``
        (no orientation weighting) to find the single highest-scoring paper.
        Intended for automated Paper of the Day selection.

        Args:
            namespace_key: The arXiv-style namespace key to score papers for.

        Returns:
            A two-tuple ``(paper_id, score)`` where ``paper_id`` is the UUID of
            the top-scoring ``Paper`` and ``score`` is its average score. Returns
            ``(None, 0.0)`` if the namespace contains no papers.
        """
        result = await self._db.execute(
            select(Paper).where(Paper.namespace_key == namespace_key)
        )
        papers = list(result.scalars())
        if not papers:
            return None, 0.0

        best = max(papers, key=lambda p: (p.novelty_score + p.relevance_score) / 2)
        score = (best.novelty_score + best.relevance_score) / 2
        return best.id, score
