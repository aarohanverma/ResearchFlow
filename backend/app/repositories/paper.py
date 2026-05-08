"""PaperRepository — the only layer that touches Paper / PaperChunk / Summary tables."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.paper import Bookmark, FeedFeedback, Paper, PaperChunk, PaperOfDay, Summary


class PaperRepository:
    """Data-access layer for Paper, PaperChunk, Summary, Bookmark, and feedback models.

    All database interactions for paper-related entities are funnelled through
    this class. Methods are async and expect a live ``AsyncSession``.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise the repository with an active async database session.

        Args:
            db: An SQLAlchemy ``AsyncSession`` used for all queries in this
                repository instance.
        """
        self._db = db

    async def get_existing_external_ids(self, namespace_key: str) -> set[str]:
        """Return the set of external IDs already stored for a namespace.

        Used during ingestion to skip papers that have already been persisted.

        Args:
            namespace_key: The arXiv-style namespace key (e.g. ``"cs.AI"``).

        Returns:
            A set of ``external_id`` strings for all papers in the namespace.
        """
        result = await self._db.execute(
            select(Paper.external_id).where(Paper.namespace_key == namespace_key)
        )
        return {row[0] for row in result.fetchall()}

    async def upsert_papers(self, papers: list[dict]) -> list[Paper]:
        """Insert new papers — skip on conflict (external_id + namespace_key)."""
        new_papers: list[Paper] = []
        for data in papers:
            existing = await self._db.execute(
                select(Paper).where(
                    Paper.external_id == data["external_id"],
                    Paper.namespace_key == data["namespace_key"],
                )
            )
            obj = existing.scalar_one_or_none()
            if obj is None:
                obj = Paper(**data)
                self._db.add(obj)
                new_papers.append(obj)
        await self._db.flush()
        return new_papers

    async def get_by_id(self, paper_id: UUID) -> Paper | None:
        """Fetch a single paper by its primary key.

        Args:
            paper_id: The UUID primary key of the paper.

        Returns:
            The matching ``Paper`` ORM object, or ``None`` if not found.
        """
        result = await self._db.execute(select(Paper).where(Paper.id == paper_id))
        return result.scalar_one_or_none()

    async def get_by_namespace(
        self,
        namespace_key: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Paper]:
        """Return papers for a namespace, newest first, with pagination.

        Args:
            namespace_key: The arXiv-style namespace key to filter on.
            limit: Maximum number of papers to return. Defaults to ``50``.
            offset: Number of rows to skip for pagination. Defaults to ``0``.

        Returns:
            A list of ``Paper`` objects ordered by ``ingested_at`` descending.
        """
        result = await self._db.execute(
            select(Paper)
            .where(Paper.namespace_key == namespace_key)
            .order_by(Paper.ingested_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars())

    async def update_enrichment(self, paper_id: UUID, enrichment: dict) -> None:
        """Bulk-update enrichment fields on a paper row.

        Args:
            paper_id: UUID of the paper to update.
            enrichment: Dictionary of column-name → value pairs to apply
                (e.g. ``key_concepts``, ``novelty_score``, ``tldr``).
        """
        await self._db.execute(
            update(Paper).where(Paper.id == paper_id).values(**enrichment)
        )

    async def add_chunk(self, chunk: PaperChunk) -> PaperChunk:
        """Persist a new ``PaperChunk`` and flush to obtain its generated ID.

        Args:
            chunk: The ``PaperChunk`` ORM instance to add.

        Returns:
            The same ``PaperChunk`` instance after the flush (ID is now set).
        """
        self._db.add(chunk)
        await self._db.flush()
        return chunk

    async def get_chunks(self, paper_id: UUID) -> list[PaperChunk]:
        """Return all stored chunks for a paper.

        Args:
            paper_id: UUID of the parent paper.

        Returns:
            A list of ``PaperChunk`` objects (may be empty).
        """
        result = await self._db.execute(
            select(PaperChunk).where(PaperChunk.paper_id == paper_id)
        )
        return list(result.scalars())

    async def get_summary(self, paper_id: UUID, expertise_level: str) -> Summary | None:
        """Look up a cached summary for a specific paper and expertise level.

        Args:
            paper_id: UUID of the paper.
            expertise_level: One of ``"newcomer"``, ``"practitioner"``, or
                ``"expert"``.

        Returns:
            The matching ``Summary`` ORM object, or ``None`` if not cached.
        """
        result = await self._db.execute(
            select(Summary).where(
                Summary.paper_id == paper_id,
                Summary.expertise_level == expertise_level,
            )
        )
        return result.scalar_one_or_none()

    async def upsert_summary(self, data: dict) -> Summary:
        """Insert or update a ``Summary`` row, incrementing the version counter.

        If a summary already exists for the given ``paper_id`` and
        ``expertise_level``, all fields are updated in-place and the
        ``version`` counter is incremented. Otherwise a new row is inserted.

        Args:
            data: Dictionary of column-name → value pairs. Must include
                ``paper_id`` and ``expertise_level`` for the lookup key.

        Returns:
            The persisted ``Summary`` ORM object (inserted or updated).
        """
        existing = await self.get_summary(data["paper_id"], data["expertise_level"])
        if existing:
            for k, v in data.items():
                setattr(existing, k, v)
            existing.version += 1
        else:
            existing = Summary(**data)
            self._db.add(existing)
        await self._db.flush()
        return existing

    async def get_bookmarks(self, user_id: UUID) -> list[Bookmark]:
        """Return all bookmarks for a user, most recently created first.

        Args:
            user_id: UUID of the user whose bookmarks to retrieve.

        Returns:
            A list of ``Bookmark`` ORM objects ordered by ``created_at`` desc.
        """
        result = await self._db.execute(
            select(Bookmark).where(Bookmark.user_id == user_id).order_by(Bookmark.created_at.desc())
        )
        return list(result.scalars())

    async def add_bookmark(self, user_id: UUID, paper_id: UUID, note: str | None = None) -> Bookmark:
        """Create a new bookmark linking a user to a paper.

        Args:
            user_id: UUID of the user creating the bookmark.
            paper_id: UUID of the paper being bookmarked.
            note: Optional user-supplied annotation text. Defaults to ``None``.

        Returns:
            The newly created ``Bookmark`` ORM object with its generated ID.
        """
        bm = Bookmark(user_id=user_id, paper_id=paper_id, note=note)
        self._db.add(bm)
        await self._db.flush()
        return bm

    async def remove_bookmark(self, user_id: UUID, paper_id: UUID) -> None:
        """Delete a bookmark if it exists; silently does nothing if not found.

        Args:
            user_id: UUID of the user who owns the bookmark.
            paper_id: UUID of the paper whose bookmark should be removed.
        """
        result = await self._db.execute(
            select(Bookmark).where(Bookmark.user_id == user_id, Bookmark.paper_id == paper_id)
        )
        bm = result.scalar_one_or_none()
        if bm:
            await self._db.delete(bm)

    async def set_potd(self, namespace_key: str, paper_id: UUID, score: float, is_breakthrough: bool) -> None:
        """Upsert the Paper of the Day record for today in a given namespace.

        If a ``PaperOfDay`` row already exists for today's date and the given
        namespace, it is updated in place. Otherwise a new row is created.

        Args:
            namespace_key: The arXiv-style namespace key (e.g. ``"cs.AI"``).
            paper_id: UUID of the paper selected as paper of the day.
            score: Composite score that determined the selection.
            is_breakthrough: Whether the score exceeded the breakthrough
                threshold.
        """
        today = datetime.now(timezone.utc).date()
        existing = await self._db.execute(
            select(PaperOfDay).where(
                PaperOfDay.namespace_key == namespace_key,
                PaperOfDay.date == today,
            )
        )
        potd = existing.scalar_one_or_none()
        if potd:
            potd.paper_id = paper_id
            potd.score = score
            potd.is_breakthrough = is_breakthrough
        else:
            self._db.add(PaperOfDay(
                namespace_key=namespace_key,
                paper_id=paper_id,
                date=today,
                score=score,
                is_breakthrough=is_breakthrough,
            ))
        await self._db.flush()

    async def add_feedback(self, user_id: UUID, paper_id: UUID, signal: str) -> None:
        """Record a feed-feedback signal (e.g. ``"like"`` or ``"dislike"``) for a paper.

        Args:
            user_id: UUID of the user submitting feedback.
            paper_id: UUID of the paper being rated.
            signal: Feedback type string (e.g. ``"like"``, ``"dislike"``).
        """
        self._db.add(FeedFeedback(user_id=user_id, paper_id=paper_id, signal=signal))
        await self._db.flush()

    async def get_liked_paper_ids(self, user_id: UUID) -> list[str]:
        """Return the IDs of all papers the user has liked.

        Used by the scoring service to boost relevance of papers similar to
        previously liked ones.

        Args:
            user_id: UUID of the user.

        Returns:
            A list of paper UUID strings where the feedback signal is ``"like"``.
        """
        result = await self._db.execute(
            select(FeedFeedback.paper_id).where(
                FeedFeedback.user_id == user_id,
                FeedFeedback.signal == "like",
            )
        )
        return [str(row.paper_id) for row in result.fetchall()]

    async def remove_feedback(self, user_id: UUID, paper_id: UUID, signal: str) -> None:
        """Delete a specific feedback signal for a (user, paper) pair.

        Args:
            user_id: UUID of the user whose feedback to remove.
            paper_id: UUID of the paper.
            signal: The signal type to remove (e.g. ``"like"``).
        """
        from sqlalchemy import delete
        await self._db.execute(
            delete(FeedFeedback).where(
                FeedFeedback.user_id == user_id,
                FeedFeedback.paper_id == paper_id,
                FeedFeedback.signal == signal,
            )
        )
