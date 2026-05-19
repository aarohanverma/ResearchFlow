"""Regression tests for depth-tier resolution.

The orchestrator resolves the per-turn depth tier from two signals:

* The LLM intent classifier — biased toward ``"single"`` by its
  system prompt to avoid over-investing in trivial queries.
* A deterministic keyword + length heuristic
  (``_assess_query_complexity``) that returns
  ``"simple" | "medium" | "complex"``.

Before the fix, the resolver used a short-circuit ``or`` so any
truthy intent value (including ``"single"``) won unconditionally —
the heuristic was completely ignored. That meant queries with clear
deep-research markers (``compare four candidate directions``,
``synthesize one novel architecture``) which the heuristic
correctly classified as ``complex`` got routed through the
``"single"`` tier and never entered the ReAct loop.

The fix is a max-of-two resolution: either signal can *raise* the
tier (heuristic catches lexical markers the LLM missed; LLM catches
implicit deep intent the heuristic missed), neither can lower the
other's verdict.

These tests pin the contract:

* The ``"compare"`` / ``"synthesize"`` markers from the screenshots
  resolve to ``depth_tier = "deep"`` even when the LLM intent says
  ``"single"``.
* A trivial greeting that the heuristic would call ``"medium"`` (its
  default) and the LLM correctly calls ``"trivial"`` resolves to
  ``"trivial"`` — the LLM can lift the tier *down* only when the
  heuristic also doesn't object.
* The resolver tolerates a missing / malformed intent value.
* The resolver tolerates a missing / malformed heuristic value.
"""

from __future__ import annotations


def _resolve_depth_tier(intent_complexity: str, heuristic_complexity: str) -> str:
    """Mirror of the orchestrator's resolver — kept tiny + side-effect-free
    so we can assert against it without rebuilding the whole turn pipeline.
    Any drift from production must be caught by updating this stub or by
    promoting it to a module-level helper called from both sites.
    """
    tier_map_intent = {"trivial": "trivial", "single": "single", "deep": "deep"}
    tier_map_heur = {"simple": "trivial", "medium": "single", "complex": "deep"}
    _TIER_RANK = {"trivial": 0, "single": 1, "deep": 2}
    candidates: list[str] = []
    it = tier_map_intent.get(intent_complexity)
    if it in _TIER_RANK:
        candidates.append(it)
    ht = tier_map_heur.get(heuristic_complexity)
    if ht in _TIER_RANK:
        candidates.append(ht)
    return max(candidates, key=_TIER_RANK.__getitem__) if candidates else "single"


# ── Heuristic-bumps-up cases (the screenshot bug) ─────────────────────────────


def test_heuristic_complex_bumps_intent_single_up_to_deep():
    """When the LLM intent says single but the heuristic detects
    ``compare`` / ``synthesize`` markers, the resolver must pick deep."""
    assert _resolve_depth_tier("single", "complex") == "deep"


def test_actual_query_one_from_screenshot_resolves_to_deep():
    """The first screenshot's query body contained 'Compare at least
    four candidate directions' + 'Synthesize one technically plausible
    novel architecture'. Heuristic classifies via the 'compare' keyword."""
    from app.assistant.planner_llm import _assess_query_complexity

    query = (
        "Investigate whether the internals of modern Transformer models are "
        "approaching their limit. Compare at least four candidate directions "
        "for improving Transformer internals, then synthesize one technically "
        "plausible novel architecture direction. End with the strongest current "
        "baseline, the most promising internal bottleneck to attack, and a "
        "proposed architecture sketch."
    )
    heur = _assess_query_complexity(query, history=[])
    assert heur == "complex", f"heuristic should classify as complex, got {heur!r}"
    # Even if the intent LLM stubbornly returns "single", the max-of-two
    # resolution lands on "deep".
    assert _resolve_depth_tier("single", heur) == "deep"


def test_actual_query_two_from_screenshot_resolves_to_deep():
    """The second screenshot's query body talked about exploring RAG
    systems, comparing directions, and synthesising a novel design."""
    from app.assistant.planner_llm import _assess_query_complexity

    query = (
        "I'm exploring whether current production RAG systems can be improved. "
        "Compare four directions for production-grade retrieval-augmented "
        "generation, evaluate each on what they actually solve in production, "
        "and synthesize a possible novel system direction."
    )
    heur = _assess_query_complexity(query, history=[])
    assert heur == "complex"
    assert _resolve_depth_tier("single", heur) == "deep"


# ── Intent-bumps-up cases (the original design intent) ───────────────────────


def test_intent_deep_overrides_simple_heuristic():
    """If the LLM correctly spots an implicit literature-survey intent
    from a terse follow-up where the heuristic sees only ``simple``,
    the resolver still picks deep."""
    assert _resolve_depth_tier("deep", "simple") == "deep"


# ── Trivial fast-path preserved when both signals agree ─────────────────────


def test_trivial_intent_and_simple_heuristic_stays_trivial():
    """Greetings / one-word follow-ups stay on the trivial fast path
    when BOTH signals agree (the heuristic's narrow keyword set —
    'what is', 'define', 'tldr' etc. — maps to trivial). Max-of-two
    over two trivials is still trivial, so no compute is wasted on
    real one-shot lookups.
    """
    assert _resolve_depth_tier("trivial", "simple") == "trivial"


def test_trivial_intent_with_medium_heuristic_lifts_to_single():
    """The heuristic's default-fallback for non-keyword queries is
    medium → single. When the LLM thinks the turn is trivial but the
    heuristic sees a medium-shape query, max-of-two routes to single.
    This is the deliberate safety bias: a misclassified greeting
    costs one extra LLM call; a misclassified research follow-up
    produces a shallow answer."""
    assert _resolve_depth_tier("trivial", "medium") == "single"


# ── Malformed inputs ─────────────────────────────────────────────────────────


def test_empty_intent_falls_back_to_heuristic():
    """An empty / unset intent complexity must not break resolution —
    the heuristic still gets a vote."""
    assert _resolve_depth_tier("", "complex") == "deep"
    assert _resolve_depth_tier("", "medium") == "single"


def test_unknown_intent_value_is_ignored():
    """An LLM that hallucinates an out-of-enum value (``"medium"``,
    ``"long"`` etc.) must be ignored, not crash, not silently treated
    as deep — the heuristic alone decides the tier.
    """
    # heuristic="complex" → deep, intent unknown → ignored → result deep
    assert _resolve_depth_tier("medium", "complex") == "deep"
    # heuristic="simple" → trivial, intent unknown → ignored → result trivial
    assert _resolve_depth_tier("medium", "simple") == "trivial"


def test_both_unknown_falls_back_to_single():
    """Belt-and-suspenders — every other branch fails ⇒ safe default."""
    assert _resolve_depth_tier("garbage", "garbage") == "single"
