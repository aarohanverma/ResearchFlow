"""Tests for the token-aware rolling history builder.

The orchestrator's rolling-history strategy used to gate verbatim vs.
summarised purely on message COUNT (last 10 verbatim). A single very
long message (user pasting a paper body) blew the context window even
with count < threshold. The token-aware path catches that by walking
backwards from the latest message and stopping when the running token
estimate hits the verbatim budget — old messages spill into the
summary regardless of count.

The current user turn is ALWAYS preserved verbatim even when it alone
exceeds the budget; losing the most recent turn would defeat the
conversation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.assistant.orchestrator import Orchestrator


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ── Token estimator ────────────────────────────────────────────────────────


def test_token_estimate_uses_chars_over_four_heuristic():
    """The estimator uses chars/4 + a small per-message overhead so
    the summed tokens approximate provider-side token counts within a
    factor that still keeps us inside the window."""
    # 400 chars → ~100 tokens + 12 overhead = ~112
    msg = _msg("user", "x" * 400)
    est = Orchestrator._estimate_msg_tokens(msg)
    assert 110 <= est <= 115


def test_token_estimate_handles_multimodal_content():
    """Multimodal messages (text + image parts) count text only —
    images are billed separately by providers and don't consume the
    text window in the same way."""
    msg = {"role": "user", "content": [
        {"type": "text", "text": "x" * 400},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]}
    est = Orchestrator._estimate_msg_tokens(msg)
    assert 110 <= est <= 115


def test_token_estimate_handles_none_content():
    msg = {"role": "user", "content": None}
    # Should not crash; returns just the per-message overhead.
    est = Orchestrator._estimate_msg_tokens(msg)
    assert est >= 1


def test_token_estimate_counts_tool_use_input():
    """Anthropic-style tool_use blocks carry their payload under
    ``input``, not ``text``. A long tool-call argument must contribute
    to the token estimate so a tool-heavy turn doesn't silently blow
    the verbatim budget. Regression for the estimator-only-counts-text
    bug.
    """
    payload = "x" * 8_000  # ~2000 tokens of pure argument
    msg = {"role": "assistant", "content": [
        {"type": "text", "text": "Calling the search tool now."},  # ~7 tokens
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "deep_search",
            "input": {"query": payload, "limit": 10},
        },
    ]}
    est = Orchestrator._estimate_msg_tokens(msg)
    # The tool_use payload alone should drive the estimate well past
    # the bare-text-only value (~20 tokens including overhead).
    assert est >= 1900, f"tool_use input not counted; got {est}"


def test_token_estimate_counts_message_level_tool_calls():
    """OpenAI-style ``tool_calls`` at message level (not inside
    ``content``) must also contribute. Same regression."""
    payload = "y" * 4_000
    msg = {
        "role": "assistant",
        "content": "Calling tool.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": payload},
            },
        ],
    }
    est = Orchestrator._estimate_msg_tokens(msg)
    # Just the content text is ~14 chars → ~3 tokens + overhead.
    # With the tool_calls arguments included, must be >= 900 tokens.
    assert est >= 900, f"tool_calls arguments not counted; got {est}"


def test_token_estimate_nested_content_recursion():
    """Some chat templates nest content under ``content`` keys
    recursively. The estimator must walk the nesting rather than
    silently dropping the inner text."""
    msg = {
        "role": "user",
        "content": [
            {"type": "wrapper", "content": [
                {"type": "text", "text": "z" * 1_000},
            ]},
        ],
    }
    est = Orchestrator._estimate_msg_tokens(msg)
    # ~250 tokens of text + 12 overhead → ~262.
    assert 240 <= est <= 290, f"nested content not summed; got {est}"


# ── Token-aware verbatim cutoff (integration via Orchestrator instance) ────


class _FakeOrchestrator(Orchestrator):
    """Subclass to bypass __init__ side effects so we can call the
    history builder in isolation."""
    def __init__(self):
        # Skip the orchestrator's heavy init; the history builder
        # only touches class attributes + _summarize_turns.
        pass


@pytest.mark.asyncio
async def test_short_session_within_budget_returns_all_messages():
    """A short session that token-fits returns all messages with no
    summary — fast path."""
    orch = _FakeOrchestrator()
    msgs = [_msg("user", "q1"), _msg("assistant", "a1")]
    out = await orch._build_rolling_history(
        all_msgs=msgs, session_state={}, session_id="s1", namespace_key="cs.AI",
    )
    assert out == msgs


@pytest.mark.asyncio
async def test_single_long_message_still_kept_verbatim():
    """A single user paste that BLOWS the verbatim budget on its own
    must still be preserved — losing the current turn would be worse
    than relying on provider-side truncation."""
    orch = _FakeOrchestrator()
    # 1MB user message → ~262k tokens; far past the 51k verbatim budget.
    big = "x" * (1024 * 1024)
    msgs = [_msg("user", big)]
    out = await orch._build_rolling_history(
        all_msgs=msgs, session_state={}, session_id="s1", namespace_key="cs.AI",
    )
    # Only one message; can't be summarised. Must come back verbatim.
    assert out == msgs


@pytest.mark.asyncio
async def test_old_long_message_spills_into_summary(monkeypatch):
    """When an OLDER message is huge, the token-aware cutoff must
    push it into the summary even if the count is below the legacy
    threshold."""
    orch = _FakeOrchestrator()

    async def _fake_summarize(messages, namespace_key):
        return f"SUMMARY-OF-{len(messages)}-TURNS"

    monkeypatch.setattr(orch, "_summarize_turns", _fake_summarize)

    # Old user paste blowing the budget; recent turns small.
    huge = "y" * (1024 * 1024)  # ~262k tokens
    msgs = [
        _msg("user", huge),          # old, blows budget
        _msg("assistant", "ok"),
        _msg("user", "tell me more"),
        _msg("assistant", "sure"),
    ]
    out = await orch._build_rolling_history(
        all_msgs=msgs, session_state={}, session_id="s1", namespace_key="cs.AI",
    )
    # The huge old message must NOT be in the verbatim slice. The
    # summary system message takes its place.
    assert out[0]["role"] == "system"
    assert "SUMMARY-OF" in out[0]["content"]
    # Latest messages survive verbatim.
    recent_contents = [m.get("content") for m in out[1:]]
    assert "tell me more" in recent_contents
    assert "sure" in recent_contents
    # The huge paste does NOT appear verbatim anywhere.
    assert all(huge != (m.get("content") or "") for m in out)


@pytest.mark.asyncio
async def test_count_cutoff_still_applies_for_long_count_sessions(monkeypatch):
    """A session past the count threshold but well under token budget
    still falls into the summarisation path — preserves the legacy
    "keep last 10 verbatim" contract."""
    orch = _FakeOrchestrator()

    async def _fake_summarize(messages, namespace_key):
        return "SUMMARY"

    monkeypatch.setattr(orch, "_summarize_turns", _fake_summarize)

    # 20 small messages; count > threshold but token estimate tiny.
    msgs = [_msg("user" if i % 2 == 0 else "assistant", f"msg {i}") for i in range(20)]
    out = await orch._build_rolling_history(
        all_msgs=msgs, session_state={}, session_id="s1", namespace_key="cs.AI",
    )
    # Summary + last 10 messages = 11 entries.
    assert len(out) == 11
    assert out[0]["role"] == "system"
    assert "SUMMARY" in out[0]["content"]


@pytest.mark.asyncio
async def test_more_aggressive_cutoff_wins_when_tokens_demand_it(monkeypatch):
    """When both token cutoff and count cutoff are computed, the
    cutoff that keeps FEWER recent messages verbatim is chosen — the
    one that better protects the budget."""
    orch = _FakeOrchestrator()

    summarised: list[list[dict]] = []

    async def _fake_summarize(messages, namespace_key):
        summarised.append(list(messages))
        return "S"

    monkeypatch.setattr(orch, "_summarize_turns", _fake_summarize)

    # 15 messages where every one of the last 10 is a hefty paste —
    # the token-aware cutoff should keep fewer than 10 verbatim.
    big = "z" * (40_000)  # ~10k tokens per message
    msgs = [_msg("user", f"q{i}") for i in range(5)]  # 5 short olds
    msgs += [_msg("assistant", big) for _ in range(10)]  # 10 huge recents

    out = await orch._build_rolling_history(
        all_msgs=msgs, session_state={}, session_id="s1", namespace_key="cs.AI",
    )
    # Verbatim slice (everything after the summary system msg) must
    # have fewer than 10 messages — token cutoff dominated.
    verbatim = [m for m in out if m["role"] != "system" or not m["content"].startswith("[Conversation summary")]
    # First out msg is summary; rest are verbatim.
    assert out[0]["role"] == "system"
    assert len(out) - 1 < 10, (
        f"token cutoff did not trim — got {len(out) - 1} verbatim msgs"
    )
    # And the summariser must have been called with the spilled olds.
    assert summarised, "expected summarisation pass to run"
