"""Integration test for PII redaction at the memory-write boundary.

Tests the wiring (not the redactor itself) — i.e. that the redactor
runs BEFORE the value is persisted into ``session.state``. We don't
need a real DB; we just stub the persistence layer and assert that
the value that reached it was already redacted.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_auto_memory_strips_pii_before_persist(monkeypatch):
    """``_apply_writes`` is the path used by the background auto-memory
    consolidation pass. When the librarian model decides to remember
    something containing PII, the redactor must run BEFORE the value
    is stored in session.state."""
    from app.assistant import auto_memory

    # Fake session + root with empty state dicts the helper will mutate.
    fake_session = MagicMock()
    fake_session.id = "sess-1"
    fake_session.state = {}

    fake_root = MagicMock()
    fake_root.id = "root-1"
    fake_root.state = {}

    fake_db = MagicMock()
    fake_db.commit = AsyncMock()

    # The helper imports flag_modified + record_revision lazily — stub
    # them so we don't need a real DB session.
    monkeypatch.setattr(
        "app.assistant.memory_revisions.record_revision",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "sqlalchemy.orm.attributes.flag_modified",
        lambda *a, **kw: None,
    )

    writes = [{
        "tier": "long",
        "type": "preference",
        "key": "contact",
        "value": "Email me at aarohan@example.com about the result.",
    }]

    await auto_memory._apply_writes(
        fake_db, fake_session, fake_root, writes,
        namespace_key="ai-ml", user_id=None,
    )

    bucket = fake_session.state.get("ns_memory") or {}
    entry = bucket.get("contact")
    assert entry is not None, "write should have landed"
    stored_value = entry.get("value")
    assert "aarohan@example.com" not in stored_value
    assert "[REDACTED_EMAIL]" in stored_value
