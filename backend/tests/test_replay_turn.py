"""Regression test for the Regenerate timestamp-tie bug.

The original implementation looked for "the most recent user message
strictly before the assistant" using ``m.created_at < target.created_at``.
When the user message and its paired assistant reply share a
millisecond-resolution timestamp (very common — they're written in the
same transaction), the ``<`` comparison fails and the user message is
not found, surfacing the confusing "No preceding user message to
regenerate from" error in the UI even though one obviously exists.

The fix walks the messages in deterministic (created_at, id) order and
finds the user message immediately before the target by index, which is
robust to tied timestamps.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _msg(role: str, content: str, ts: datetime, mid: uuid.UUID | None = None):
    """Build a MagicMock(AssistantMessage)-shaped row."""
    m = MagicMock()
    m.id = mid or uuid.uuid4()
    m.role = MagicMock()
    m.role.value = role
    m.content = content
    m.created_at = ts
    return m


@pytest.mark.asyncio
async def test_regenerate_finds_user_message_with_tied_timestamps():
    """User msg and assistant reply with identical created_at — the
    fix must still find the preceding user message via index-based
    walk."""
    from app.services.research_assistant import replay_turn

    user_id = uuid.uuid4()
    sid = uuid.uuid4()
    ts_pair_1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ts_pair_2 = datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc)

    user_msg = _msg("user", "What is BERT?", ts_pair_1)
    assistant_msg = _msg("assistant", "BERT is a transformer.", ts_pair_1)
    # A second pair with later timestamp — the regenerate target is
    # the FIRST assistant reply, so we should walk back to user_msg.
    user_msg_2 = _msg("user", "Tell me more", ts_pair_2)
    assistant_msg_2 = _msg("assistant", "Sure...", ts_pair_2)

    session = MagicMock()
    session.id = sid
    session.user_id = user_id
    session.messages = [user_msg, assistant_msg, user_msg_2, assistant_msg_2]
    session.tasks = []

    # Mock the repository + DB layers — we only care about the
    # message-walk path, not the cancel/delete pipeline below.
    repo_mock = MagicMock()
    repo_mock.get_session = AsyncMock(return_value=session)

    # Patch the dependencies the function actually reaches.
    with patch("app.services.research_assistant.async_session_factory") as factory_patch, \
         patch("app.services.research_assistant.AssistantRepository", MagicMock(return_value=repo_mock)):
        db = AsyncMock()
        db.commit = AsyncMock()
        db.execute = AsyncMock()
        db.flush = AsyncMock()

        class _DBCtx:
            async def __aenter__(self):
                return db
            async def __aexit__(self, *a):
                return False

        factory_patch.return_value = _DBCtx()

        # Run replay_turn — it will raise on some downstream step we
        # haven't bothered to mock (delete cascade, task submit, etc.).
        # That's fine; the property we care about is that the walk
        # past the "No preceding user message" check succeeded. Any
        # OTHER error is acceptable for this test.
        try:
            await replay_turn(
                user_id=user_id,
                session_id=sid,
                message_id=assistant_msg.id,
                new_content=None,
            )
        except Exception as exc:
            assert "No preceding user message" not in str(exc), (
                f"regenerate walk failed on tied timestamps: {exc}"
            )
