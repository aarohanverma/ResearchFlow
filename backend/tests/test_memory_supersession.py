"""Tests for automatic supersession detection.

When a new memory write is semantically near-identical to an existing
entry, the old entry must be marked superseded — preserved in storage
(for audit + restore) but filtered out of recall so the planner stops
seeing duplicates of the same idea.

We mock the embedder so the tests don't depend on a live provider.
The supersession threshold and exempt-type behaviour are exercised
directly; the integration with ``auto_memory`` is covered by the
existing auto-memory tests.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.assistant.memory_supersession import detect_and_mark_supersessions


# Helper to mock the embedder so similarity is deterministic.
def _patch_embedder(vectors_by_text: dict[str, list[float]]):
    """Return a patcher that yields a mock embedder returning the
    pre-baked vector for each text. Any text not in the map gets
    a unit-orthogonal vector that fails the similarity threshold."""
    class _MockEmbedder:
        async def embed_query(self, q):
            return vectors_by_text.get(q, [1.0, 0.0])
        async def embed_texts(self, texts, task_type=None):
            return [vectors_by_text.get(t, [1.0, 0.0]) for t in texts]

    return patch(
        "app.assistant.semantic_memory._get_embedder",
        AsyncMock(return_value=_MockEmbedder()),
    )


@pytest.mark.asyncio
async def test_near_duplicate_marks_old_entry_superseded():
    """When the new value's embedding is near-identical to an
    existing entry's, the OLD entry gets ``superseded_by_key`` set
    and the new one stays untouched."""
    bucket = {
        "old_pref": {
            "value": "User prefers concise technical explanations.",
            "type": "preference",
            "ts": "2026-01-01T00:00:00+00:00",
        },
        "new_pref": {
            "value": "User likes brief technical answers.",
            "type": "preference",
            "ts": "2026-05-01T00:00:00+00:00",
        },
    }
    # Same vector for both → cosine sim = 1.0 → far above 0.88 threshold.
    same_vec = [0.5, 0.5, 0.5]
    patcher = _patch_embedder({
        "User likes brief technical answers.": same_vec,
        "User prefers concise technical explanations.": same_vec,
    })
    with patcher:
        superseded = await detect_and_mark_supersessions(
            bucket=bucket,
            new_key="new_pref",
            new_value="User likes brief technical answers.",
            new_type="preference",
            session_id=uuid.uuid4(),
        )
    assert superseded == ["old_pref"]
    assert bucket["old_pref"]["superseded_by_key"] == "new_pref"
    assert "superseded_at" in bucket["old_pref"]
    assert bucket["old_pref"]["superseded_similarity"] == 1.0
    # The new entry must NOT be marked superseded.
    assert "superseded_by_key" not in bucket["new_pref"]


@pytest.mark.asyncio
async def test_low_similarity_leaves_old_entry_untouched():
    """When the vectors disagree, the old entry stays active."""
    bucket = {
        "unrelated": {
            "value": "BERT-large hits 92% on SQuAD.",
            "type": "finding",
            "ts": "2026-01-01T00:00:00+00:00",
        },
    }
    # Orthogonal vectors → cosine sim ≈ 0 → well under threshold.
    patcher = _patch_embedder({
        "BERT-large hits 92% on SQuAD.": [1.0, 0.0],
        "Transformer self-attention is parallel.": [0.0, 1.0],
    })
    with patcher:
        superseded = await detect_and_mark_supersessions(
            bucket=bucket,
            new_key="new_finding",
            new_value="Transformer self-attention is parallel.",
            new_type="finding",
            session_id=uuid.uuid4(),
        )
    assert superseded == []
    assert "superseded_by_key" not in bucket["unrelated"]


@pytest.mark.asyncio
async def test_hypothesis_type_exempt_from_supersession():
    """``hypothesis`` entries are research proposals — coherent
    variations should NOT supersede each other even when text
    overlaps. The exemption is by design."""
    bucket = {
        "old_hyp": {
            "value": "Attention heads may encode position implicitly.",
            "type": "hypothesis",
            "ts": "2026-01-01T00:00:00+00:00",
        },
    }
    # Identical vectors — would normally supersede, but type=hypothesis
    # is exempt.
    same_vec = [0.5, 0.5]
    patcher = _patch_embedder({
        "Attention heads may encode position implicitly.": same_vec,
        "Position is encoded implicitly by attention heads.": same_vec,
    })
    with patcher:
        superseded = await detect_and_mark_supersessions(
            bucket=bucket,
            new_key="new_hyp",
            new_value="Position is encoded implicitly by attention heads.",
            new_type="hypothesis",
            session_id=uuid.uuid4(),
        )
    assert superseded == []


@pytest.mark.asyncio
async def test_different_class_never_supersedes():
    """A new ``procedure`` does NOT supersede a similar ``finding``
    even when their text overlaps — they play different roles."""
    bucket = {
        "old_finding": {
            "value": "Cite Semantic Scholar for biomedical papers.",
            "type": "finding",
            "ts": "2026-01-01T00:00:00+00:00",
        },
    }
    same_vec = [0.5, 0.5]
    patcher = _patch_embedder({
        "Cite Semantic Scholar for biomedical papers.": same_vec,
        "Always cite Semantic Scholar for biomedical papers.": same_vec,
    })
    with patcher:
        superseded = await detect_and_mark_supersessions(
            bucket=bucket,
            new_key="new_proc",
            new_value="Always cite Semantic Scholar for biomedical papers.",
            new_type="procedure",   # different class from finding (semantic)
            session_id=uuid.uuid4(),
        )
    assert superseded == []


@pytest.mark.asyncio
async def test_already_superseded_entry_not_double_tagged():
    """An entry that's already been superseded by an earlier write
    must not get re-tagged when a third write comes in. Idempotent."""
    bucket = {
        "ancient": {
            "value": "v1: User likes short answers.",
            "type": "preference",
            "ts": "2026-01-01T00:00:00+00:00",
            "superseded_by_key": "middle",
            "superseded_at": "2026-03-01T00:00:00+00:00",
        },
    }
    same_vec = [0.5, 0.5]
    patcher = _patch_embedder({
        "v1: User likes short answers.": same_vec,
        "v3: User wants brief responses.": same_vec,
    })
    with patcher:
        superseded = await detect_and_mark_supersessions(
            bucket=bucket,
            new_key="newest",
            new_value="v3: User wants brief responses.",
            new_type="preference",
            session_id=uuid.uuid4(),
        )
    # ``ancient`` is already superseded — skipped.
    assert superseded == []
    # Original supersession metadata preserved.
    assert bucket["ancient"]["superseded_by_key"] == "middle"


@pytest.mark.asyncio
async def test_no_supersession_when_embedder_offline():
    """If the embedder import / call fails, the write proceeds
    without supersession — the older entry just stays active.
    Best-effort by design."""
    bucket = {
        "old": {"value": "x", "type": "finding", "ts": "2026-01-01"},
    }
    patcher = patch(
        "app.assistant.semantic_memory._get_embedder",
        AsyncMock(return_value=None),
    )
    with patcher:
        superseded = await detect_and_mark_supersessions(
            bucket=bucket,
            new_key="new",
            new_value="something else",
            new_type="finding",
            session_id=uuid.uuid4(),
        )
    assert superseded == []
    assert "superseded_by_key" not in bucket["old"]
