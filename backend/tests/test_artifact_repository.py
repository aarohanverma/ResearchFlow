"""Tests for ArtifactRepository — CRUD for GeneratedArtifact rows.

All DB calls are mocked with AsyncMock so no real database is required.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.artifact import ArtifactStatus, GeneratedArtifact, GenerationType, SourceType
from app.repositories.artifact import ArtifactRepository


def _make_artifact(**kwargs) -> GeneratedArtifact:
    """Build a minimal GeneratedArtifact ORM object for testing."""
    defaults = dict(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        generation_type=GenerationType.podcast,
        source_type=SourceType.paper,
        source_id=uuid.uuid4(),
        status=ArtifactStatus.queued,
        blob_path=None,
        content=None,
        expertise_level="practitioner",
        orientation="both",
        provider=None,
        model_used=None,
        input_tokens=0,
        output_tokens=0,
        generation_duration_ms=0,
        error_message=None,
        artifact_metadata={},
        created_at=datetime.now(timezone.utc),
        completed_at=None,
    )
    defaults.update(kwargs)
    obj = MagicMock(spec=GeneratedArtifact)
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


@pytest.fixture
def db():
    session = AsyncMock()
    return session


@pytest.fixture
def repo(db):
    return ArtifactRepository(db)


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_returns_artifact(self, repo, db):
        """create() should add artifact to session and flush."""
        uid = uuid.uuid4()
        sid = uuid.uuid4()

        # Simulate flush setting the id
        artifact_holder = {}

        async def mock_flush():
            # The artifact was added to db.add(), capture it
            pass

        db.flush = AsyncMock(side_effect=mock_flush)
        db.add = MagicMock()

        result = await repo.create(
            user_id=uid,
            generation_type=GenerationType.slides,
            source_type=SourceType.paper,
            source_id=sid,
            expertise_level="expert",
            orientation="research",
        )

        db.add.assert_called_once()
        db.flush.assert_awaited_once()
        assert result.user_id == uid
        assert result.generation_type == GenerationType.slides
        assert result.source_type == SourceType.paper
        assert result.source_id == sid
        assert result.expertise_level == "expert"
        assert result.status == ArtifactStatus.queued


class TestGetById:
    @pytest.mark.asyncio
    async def test_get_by_id_returns_artifact(self, repo, db):
        art = _make_artifact()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=art)
        db.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_by_id(art.id)

        assert result is art

    @pytest.mark.asyncio
    async def test_get_by_id_missing_returns_none(self, repo, db):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        db.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_by_id(uuid.uuid4())
        assert result is None


class TestGetLatestCompleted:
    @pytest.mark.asyncio
    async def test_returns_completed_artifact(self, repo, db):
        uid = uuid.uuid4()
        sid = uuid.uuid4()
        art = _make_artifact(
            user_id=uid,
            source_id=sid,
            generation_type=GenerationType.podcast,
            status=ArtifactStatus.completed,
        )
        # get_latest_completed uses .scalar_one_or_none() path — mock accordingly
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=art)
        db.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_latest_completed(
            user_id=uid, source_id=sid, generation_type=GenerationType.podcast
        )
        assert result is art

    @pytest.mark.asyncio
    async def test_returns_none_when_no_completed(self, repo, db):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        db.execute = AsyncMock(return_value=mock_result)

        result = await repo.get_latest_completed(
            user_id=uuid.uuid4(),
            source_id=uuid.uuid4(),
            generation_type=GenerationType.slides,
        )
        assert result is None


class TestListForSource:
    @pytest.mark.asyncio
    async def test_returns_list(self, repo, db):
        arts = [_make_artifact(), _make_artifact()]
        mock_scalars = MagicMock()
        mock_scalars.all = MagicMock(return_value=arts)
        mock_result = MagicMock()
        mock_result.scalars = MagicMock(return_value=mock_scalars)
        db.execute = AsyncMock(return_value=mock_result)

        result = await repo.list_for_source(user_id=uuid.uuid4(), source_id=uuid.uuid4())
        assert len(result) == 2


class TestMarkCompleted:
    @pytest.mark.asyncio
    async def test_marks_status_and_sets_fields(self, repo, db):
        art = _make_artifact()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=art)
        db.execute = AsyncMock(return_value=mock_result)

        await repo.mark_completed(
            art.id,
            blob_path="podcasts/abc.mp3",
            provider="openai",
            model_used="tts-1-hd",
            input_tokens=1000,
            output_tokens=500,
            duration_ms=12000,
        )

        assert art.status == ArtifactStatus.completed
        assert art.blob_path == "podcasts/abc.mp3"
        assert art.provider == "openai"
        assert art.model_used == "tts-1-hd"
        assert art.input_tokens == 1000
        assert art.output_tokens == 500
        assert art.completed_at is not None

    @pytest.mark.asyncio
    async def test_mark_completed_noop_on_missing(self, repo, db):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        db.execute = AsyncMock(return_value=mock_result)

        # Should not raise
        await repo.mark_completed(uuid.uuid4(), blob_path="x.mp3")


class TestMarkFailed:
    @pytest.mark.asyncio
    async def test_marks_failed_with_message(self, repo, db):
        art = _make_artifact()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=art)
        db.execute = AsyncMock(return_value=mock_result)

        await repo.mark_failed(art.id, error_message="LLM API rate limit exceeded")

        assert art.status == ArtifactStatus.failed
        assert "rate limit" in art.error_message
        assert art.completed_at is not None

    @pytest.mark.asyncio
    async def test_truncates_long_error(self, repo, db):
        art = _make_artifact()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=art)
        db.execute = AsyncMock(return_value=mock_result)

        long_err = "x" * 3000
        await repo.mark_failed(art.id, error_message=long_err)

        assert len(art.error_message) <= 2000
