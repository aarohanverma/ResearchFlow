"""End-to-end contract harness for the Research Assistant.

This is the *regression net* the user can run on every change to catch
drift before it ships. Each :class:`GoldenCase` in
``golden_cases.CASES`` produces multiple assertions; one pytest test
per case keeps the failure messages focused.

What it covers:
  * Query-shape strategy router lands every canonical query in the
    right shape.
  * Namespace packs expose their domain tools (pubmed for q-bio,
    inspire_hep for physics, fred for econ, …).
  * The strategy router's ``must_prefer`` / ``must_avoid`` lists hold.
  * Per-case extra assertions exercise:
      - the param-preflight + auto-repair
      - lexical + numeric contradiction detection
      - claim-level provenance verdicts
      - repair-drift detection
      - prompt-injection sanitiser

What it deliberately does NOT do:
  * Call real LLMs. Every check is deterministic over fixture inputs.
  * Hit the database. Modules are exercised in isolation.
  * Make any network calls.
"""

from __future__ import annotations

import inspect

import pytest

from tests.eval.golden_cases import CASES, GoldenCase


def _world_for(case: GoldenCase) -> dict:
    """Build the read-only ``world`` dict each case assertion sees."""
    from app.assistant.query_strategy import classify_query
    from app.assistant.tools.namespace_packs import get_visible_tools

    strategy = classify_query(case.query, history=case.history)
    visible = get_visible_tools(case.namespace_key)
    return {
        "case": case,
        "strategy": strategy,
        "visible_tools": set(visible) if visible is not None else None,
    }


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_golden_case(case: GoldenCase) -> None:
    """Run every contract assertion for ``case``."""
    world = _world_for(case)
    strategy = world["strategy"]

    # Shape contract
    if case.expected_shape is not None:
        assert strategy.shape == case.expected_shape, (
            f"[{case.name}] expected shape={case.expected_shape!r}, "
            f"got {strategy.shape!r}"
        )

    # Preferred / avoided tools (advisory but contractual: a strategy
    # hint that flips tools from preferred → avoided across releases
    # is exactly the kind of silent regression the harness is for).
    for t in case.must_prefer:
        assert t in strategy.preferred_tools, (
            f"[{case.name}] expected '{t}' in preferred_tools, "
            f"got {strategy.preferred_tools!r}"
        )
    for t in case.must_avoid:
        assert t in strategy.avoid_tools, (
            f"[{case.name}] expected '{t}' in avoid_tools, "
            f"got {strategy.avoid_tools!r}"
        )

    # Namespace visibility
    visible = world["visible_tools"]
    if visible is not None:
        for t in case.must_be_visible:
            assert t in visible, (
                f"[{case.name}] expected '{t}' to be visible for "
                f"namespace '{case.namespace_key}', got {sorted(visible)!r}"
            )
        for t in case.must_be_hidden:
            assert t not in visible, (
                f"[{case.name}] expected '{t}' to be HIDDEN for "
                f"namespace '{case.namespace_key}', but it is visible."
            )

    # Per-case extras
    for label, fn in case.extra_assertions:
        # The check helpers either take ``world`` or take no args —
        # call with whichever matches their signature so case authors
        # can use the cleaner form.
        try:
            sig = inspect.signature(fn)
            ok = fn(world) if sig.parameters else fn()
        except TypeError:
            ok = fn(world)  # best effort
        assert ok, f"[{case.name}] extra assertion failed: {label}"
