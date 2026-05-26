"""Tests for soft procedural-memory hot-path injection.

Procedural entries (``skill`` / ``procedure`` types) describe HOW
the agent should behave, not facts about the world. The user
explicitly asked for these to land in the planner system prompt
where they shape the planner's reasoning throughout — not in the
user-prompt memory block where they read as an aside.

The block must:

  * surface only ``skill`` / ``procedure`` entries (semantic / episodic
    types stay in the regular memory block);
  * draw from medium AND long tiers (short is per-chat and rarely
    carries durable procedural intent);
  * frame the entries as SOFT preferences, never as overrides for the
    platform's hard invariants (HITL gates, graph-build ban);
  * be bounded so a runaway count of user procedures can't shadow
    the static prompt.
"""

from __future__ import annotations

import pytest

from app.assistant.planner_llm import _render_procedural_block, _MAX_PROCEDURAL_INJECT


def test_empty_memory_renders_nothing():
    """No memory = no block. The static system prompt is used unchanged."""
    assert _render_procedural_block({}) == ""
    assert _render_procedural_block({"medium": {}, "long": {}}) == ""


def test_non_procedural_types_ignored():
    """Semantic / episodic entries do NOT belong in the procedural
    block — they're facts, not instructions."""
    memory = {
        "long": {
            "user_finding": {"value": "BERT-large hits 92% on SQuAD.", "type": "finding"},
            "user_concept": {"value": "Attention is parallel.", "type": "concept"},
            "user_episode": {"value": "Compared GPT-4 vs Claude.", "type": "episode"},
            "user_pref":    {"value": "Concise answers.", "type": "preference"},
        },
    }
    assert _render_procedural_block(memory) == ""


def test_procedural_entries_render():
    """``skill`` and ``procedure`` entries from any persistent tier
    must surface in the block."""
    memory = {
        "medium": {
            "matrix_after_list": {
                "value": "After listing papers, attach a TL;DR matrix.",
                "type": "procedure",
            },
        },
        "long": {
            "biomed_cites": {
                "value": "Always cite Semantic Scholar for biomedical papers.",
                "type": "skill",
            },
        },
    }
    block = _render_procedural_block(memory)
    assert "USER PROCEDURAL MEMORY" in block
    assert "matrix_after_list" in block
    assert "biomed_cites" in block
    # The framing must read as SOFT — the user spec is explicit that
    # procedures are preferences, not overrides for invariants.
    assert "soft" in block.lower() or "preference" in block.lower()
    # And the tier/type tag must be visible so the model can
    # attribute behaviour to a specific stored procedure.
    assert "[medium/procedure]" in block
    assert "[long/skill]" in block


def test_runaway_procedures_capped():
    """A pathological user with many procedural entries must not
    blow the system prompt. The cap keeps the static prompt
    dominant."""
    memory = {
        "long": {
            f"proc_{i}": {"value": f"Do thing number {i}", "type": "procedure"}
            for i in range(50)
        },
    }
    block = _render_procedural_block(memory)
    rendered = [line for line in block.splitlines() if line.startswith("- [")]
    assert len(rendered) == _MAX_PROCEDURAL_INJECT


def test_malformed_entries_skipped_safely():
    """Non-dict entries or entries missing required fields must NOT
    crash the renderer — best-effort behaviour."""
    memory = {
        "long": {
            "good":      {"value": "Run literature_survey for surveys.", "type": "procedure"},
            "no_value":  {"type": "procedure"},
            "wrong_type": "legacy string entry",   # legacy str shape
            "empty":     {"value": "", "type": "skill"},
        },
    }
    block = _render_procedural_block(memory)
    # Only the well-formed procedural entry should appear.
    assert "good" in block
    assert "no_value" not in block
    # No crash, no malformed lines.
    for line in block.splitlines():
        if line.startswith("- ["):
            # Each rendered line must have shape "- [tier/type] key: value"
            assert ":" in line


def test_value_truncated_to_240_chars():
    """A single huge procedure value must be truncated so a user
    can't fill the planner prompt by writing a 10k-char skill."""
    huge = "x" * 10_000
    memory = {
        "long": {
            "huge_proc": {"value": huge, "type": "procedure"},
        },
    }
    block = _render_procedural_block(memory)
    # Find the rendered line and confirm it's bounded.
    proc_lines = [ln for ln in block.splitlines() if "huge_proc" in ln]
    assert len(proc_lines) == 1
    # Line is "- [long/procedure] huge_proc: <value>" — the value
    # itself is capped at 240. The wrapping prefix adds ~30 chars.
    assert len(proc_lines[0]) <= 280
