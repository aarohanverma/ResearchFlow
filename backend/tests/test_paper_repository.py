"""Unit tests for PaperRepository — verifies DB interaction patterns."""

import uuid
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.repositories.paper import PaperRepository


def _make_paper_data(**overrides):
    base = {
        "external_id": "2401.00001",
        "namespace_key": "cs.AI",
        "title": "Test Paper",
        "authors": ["Alice"],
        "abstract": "Test abstract.",
        "source_url": "https://arxiv.org/abs/2401.00001",
        "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
        "published_at": None,
    }
    base.update(overrides)
    return base


def _scalar_none(db):
    """Make db.execute() return a result whose scalar_one_or_none() returns None."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result)


def _scalar_obj(db, obj):
    """Make db.execute() return a result with a specific object."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = obj
    db.execute = AsyncMock(return_value=result)


def _batch_empty(db):
    """Make db.execute() return an empty batch result (no existing pairs).

    Used for the new upsert_papers implementation that issues a single batch
    SELECT instead of per-paper scalar lookups.
    """
    result = MagicMock()
    result.fetchall.return_value = []
    db.execute = AsyncMock(return_value=result)


def _batch_with_pairs(db, pairs: list[tuple[str, str]]):
    """Make db.execute() return batch rows for the given (external_id, namespace_key) pairs."""
    result = MagicMock()
    rows = []
    for ext_id, ns_key in pairs:
        row = MagicMock()
        row.external_id = ext_id
        row.namespace_key = ns_key
        rows.append(row)
    result.fetchall.return_value = rows
    db.execute = AsyncMock(return_value=result)


class TestGetExistingExternalIds:
    @pytest.mark.asyncio
    async def test_returns_set_of_ids(self, mock_db):
        result = MagicMock()
        result.fetchall.return_value = [("2401.00001",), ("2401.00002",)]
        mock_db.execute = AsyncMock(return_value=result)

        repo = PaperRepository(mock_db)
        ids = await repo.get_existing_external_ids("cs.AI")

        assert ids == {"2401.00001", "2401.00002"}

    @pytest.mark.asyncio
    async def test_empty_namespace_returns_empty_set(self, mock_db):
        result = MagicMock()
        result.fetchall.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        repo = PaperRepository(mock_db)
        ids = await repo.get_existing_external_ids("cs.UNKNOWN")
        assert ids == set()


class TestUpsertPapers:
    @pytest.mark.asyncio
    async def test_inserts_new_paper(self, mock_db):
        # Batch query returns no existing pairs — paper should be inserted
        _batch_empty(mock_db)

        repo = PaperRepository(mock_db)
        data = [_make_paper_data()]
        new_papers = await repo.upsert_papers(data)

        assert len(new_papers) == 1
        mock_db.add.assert_called_once()
        mock_db.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_existing_paper(self, mock_db, sample_paper):
        # Batch query returns the existing pair — paper should be skipped
        _batch_with_pairs(mock_db, [("2401.00001", "cs.AI")])

        repo = PaperRepository(mock_db)
        data = [_make_paper_data()]
        new_papers = await repo.upsert_papers(data)

        assert new_papers == []
        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_papers_new_and_existing(self, mock_db, sample_paper):
        # Batch query returns only the second paper as existing
        _batch_with_pairs(mock_db, [("2401.00001", "cs.AI")])

        repo = PaperRepository(mock_db)
        data = [_make_paper_data(external_id="NEW001"), _make_paper_data(external_id="2401.00001")]
        new_papers = await repo.upsert_papers(data)

        assert len(new_papers) == 1
        assert mock_db.add.call_count == 1


class TestGetById:
    @pytest.mark.asyncio
    async def test_returns_paper_when_found(self, mock_db, sample_paper):
        _scalar_obj(mock_db, sample_paper)

        repo = PaperRepository(mock_db)
        result = await repo.get_by_id(sample_paper.id)
        assert result is sample_paper

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, mock_db):
        _scalar_none(mock_db)

        repo = PaperRepository(mock_db)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None


class TestBookmarks:
    @pytest.mark.asyncio
    async def test_add_bookmark(self, mock_db, user_id, paper_id):
        _scalar_none(mock_db)

        repo = PaperRepository(mock_db)
        bm = await repo.add_bookmark(user_id, paper_id, note="Great paper!")

        mock_db.add.assert_called_once()
        mock_db.flush.assert_awaited()
        added = mock_db.add.call_args[0][0]
        assert added.user_id == user_id
        assert added.paper_id == paper_id
        assert added.note == "Great paper!"

    @pytest.mark.asyncio
    async def test_remove_bookmark_when_exists(self, mock_db, user_id, paper_id):
        existing_bm = MagicMock()
        _scalar_obj(mock_db, existing_bm)

        repo = PaperRepository(mock_db)
        await repo.remove_bookmark(user_id, paper_id)

        mock_db.delete.assert_awaited_once_with(existing_bm)

    @pytest.mark.asyncio
    async def test_remove_bookmark_noop_when_missing(self, mock_db, user_id, paper_id):
        _scalar_none(mock_db)

        repo = PaperRepository(mock_db)
        await repo.remove_bookmark(user_id, paper_id)
        mock_db.delete.assert_not_awaited()


class TestFeedFeedback:
    @pytest.mark.asyncio
    async def test_add_feedback_creates_record(self, mock_db, user_id, paper_id):
        repo = PaperRepository(mock_db)
        await repo.add_feedback(user_id, paper_id, "like")

        mock_db.add.assert_called_once()
        added = mock_db.add.call_args[0][0]
        assert added.user_id == user_id
        assert added.paper_id == paper_id
        assert added.signal == "like"
        mock_db.flush.assert_awaited()
