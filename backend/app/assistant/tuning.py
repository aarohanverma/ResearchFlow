"""Centralised heuristic-threshold tuning for the Research Assistant.

Every threshold in this file is a **soft prior**, not a derived metric.
The values were picked by judgement against the cases I had visibility
into; they nudge behaviour rather than drive it. Each one is
overridable at runtime via an environment variable, so deployments
that observe overfitting on a particular namespace / corpus can
re-tune without code changes.

Overfitting check-list — for every threshold below:

  * Document **what it controls** (which gate fires harder/softer).
  * Document **why this value** (the inflection point I observed).
  * Provide an **env override** so the deployment can recalibrate
    without a code edit.
  * Prefer soft caveats over hard cuts (``unverified`` instead of
    ``unsupported``, prompt-render instead of force-action) wherever a
    miscalibration would mute the model's own judgement.

Naming convention: ``<MODULE>_<PURPOSE>_<UNIT>``. Env vars use the
same name in upper-snake case, prefixed with ``RA_`` (e.g.
``RA_PROVENANCE_SUPPORT_THRESHOLD``).

When a module needs one of these, import it from here rather than
re-deriving the constant locally. Centralising prevents "the same
threshold in three places drifted apart silently" — a real overfitting
risk when soft priors leak into multiple files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


# ── Provenance verifier ──────────────────────────────────────────────────────
#
# A claim is "supported" when ≥30% of its content tokens (stopwords
# removed) appear in the cited paper's text. The threshold was chosen
# from inspection of supported vs unsupported claim pairs; it sits
# above the "two papers share four common words by chance" floor and
# below the "claim quotes a phrase from the abstract" ceiling.
#
# A claim is "unverified" (kept with a caveat) at ≥12% overlap. Below
# that, we treat it as "unsupported" — the citation is more likely
# wrong than borderline.

PROVENANCE_SUPPORT_THRESHOLD = _env_float("RA_PROVENANCE_SUPPORT_THRESHOLD", 0.30)
PROVENANCE_UNVERIFIED_FLOOR = _env_float("RA_PROVENANCE_UNVERIFIED_FLOOR", 0.12)
PROVENANCE_LLM_BUDGET = _env_int("RA_PROVENANCE_LLM_BUDGET", 8)

# Hard salient-noun veto: minimum count of MISSING salient terms before
# we hard-veto a citation. Single missing terms (often 2-3 letter field
# acronyms like "NLP") are too noisy to veto on; ≥2 missing is the
# inflection where the citation almost certainly belongs to a different
# paper.
PROVENANCE_SALIENT_HARD_VETO_MIN = _env_int("RA_PROVENANCE_SALIENT_HARD_VETO_MIN", 2)


# ── Contradiction detector ───────────────────────────────────────────────────
#
# Adaptive counter-search policy. We only auto-force a counter-search
# when the open contradiction's confidence clears the threshold AND we
# haven't already forced one this turn. Soft signals (below the
# threshold) surface in the prompt and trust the model's judgement.

CONTRADICTION_FORCE_CONFIDENCE = _env_float("RA_CONTRADICTION_FORCE_CONFIDENCE", 0.65)
CONTRADICTION_MAX_FORCED_PER_TURN = _env_int("RA_CONTRADICTION_MAX_FORCED_PER_TURN", 1)
CONTRADICTION_SEMANTIC_MAX_PAIRS = _env_int("RA_CONTRADICTION_SEMANTIC_MAX_PAIRS", 4)
CONTRADICTION_SEMANTIC_TOPIC_FLOOR = _env_float("RA_CONTRADICTION_SEMANTIC_TOPIC_FLOOR", 0.25)


# ── Retrieval observability ─────────────────────────────────────────────────
#
# Coverage / rerank thresholds for "thin" and "rerank-rescued" calls.
# Below these the prompt receives a warning so the loop can broaden
# the next query.

RETRIEVAL_THIN_COVERAGE = _env_float("RA_RETRIEVAL_THIN_COVERAGE", 0.4)
RETRIEVAL_HIGH_RERANK_DISAGREEMENT = _env_float("RA_RETRIEVAL_HIGH_RERANK_DISAGREEMENT", 0.45)


# ── Repair-drift detector ────────────────────────────────────────────────────
#
# Jaccard threshold (stopwords stripped) above which we consider a
# pre/post-repair claim pair to be "same claim, different prose"
# (paraphrase) rather than "new claim" (drift). 0.25 is forgiving on
# purpose — false-negative-on-drift is preferred to false-positive-
# on-drift since ``has_drift`` only fires on genuinely new markers /
# new claims, not on changed_claims.

REPAIR_DRIFT_NEAR_MATCH_JACCARD = _env_float("RA_REPAIR_DRIFT_NEAR_MATCH_JACCARD", 0.25)


# ── ReAct loop ────────────────────────────────────────────────────────────────
#
# These are *defaults* on the ReactConfig — the orchestrator overrides
# them per call when it has a stronger prior (e.g. depth_tier="single"
# pushes the cap down). Env overrides act as deployment-wide ceilings.

REACT_DEFAULT_MAX_ITERATIONS = _env_int("RA_REACT_DEFAULT_MAX_ITERATIONS", 8)
REACT_DEFAULT_DEADLINE_SECONDS = _env_float("RA_REACT_DEFAULT_DEADLINE_SECONDS", 90.0)
REACT_MIN_ITERS_BEFORE_FREE_FINALIZE = _env_int("RA_REACT_MIN_ITERS_BEFORE_FREE_FINALIZE", 3)
REACT_SAME_TOOL_FAILURE_CAP = _env_int("RA_REACT_SAME_TOOL_FAILURE_CAP", 2)
REACT_MAX_FANOUT_BRANCHES = _env_int("RA_REACT_MAX_FANOUT_BRANCHES", 4)
# Per-tool TOTAL invocation cap per turn (successes + failures). The
# existing failure-only cap (``REACT_SAME_TOOL_FAILURE_CAP``) bans a
# repeatedly-broken tool; this cap prevents a planner that is stuck
# in a successful-but-redundant loop (e.g. eight calls to deep_search
# with slightly different queries) from chewing budget. Mirrors the
# LangChain ``ToolCallLimitMiddleware`` ``run_limit``; tunable per
# deployment via env. Set conservatively — a healthy turn rarely
# calls the same tool more than 3–4 times.
REACT_PER_TOOL_INVOCATION_CAP = _env_int("RA_REACT_PER_TOOL_INVOCATION_CAP", 5)


# ── External services ────────────────────────────────────────────────────────

MCP_SEARCH_TIMEOUT_SECONDS = _env_float("RA_MCP_SEARCH_TIMEOUT_SECONDS", 8.0)


@dataclass(frozen=True)
class TuningSnapshot:
    """Read-only snapshot for logging / debugging.

    Useful when investigating a regression: log this once at boot and
    you can tell at a glance whether the deployment is running on
    defaults or has been re-tuned.
    """
    provenance_support: float = PROVENANCE_SUPPORT_THRESHOLD
    provenance_unverified_floor: float = PROVENANCE_UNVERIFIED_FLOOR
    provenance_llm_budget: int = PROVENANCE_LLM_BUDGET
    contradiction_force_confidence: float = CONTRADICTION_FORCE_CONFIDENCE
    contradiction_max_forced_per_turn: int = CONTRADICTION_MAX_FORCED_PER_TURN
    contradiction_semantic_max_pairs: int = CONTRADICTION_SEMANTIC_MAX_PAIRS
    contradiction_semantic_topic_floor: float = CONTRADICTION_SEMANTIC_TOPIC_FLOOR
    retrieval_thin_coverage: float = RETRIEVAL_THIN_COVERAGE
    retrieval_high_rerank_disagreement: float = RETRIEVAL_HIGH_RERANK_DISAGREEMENT
    repair_drift_near_match_jaccard: float = REPAIR_DRIFT_NEAR_MATCH_JACCARD
    react_default_max_iterations: int = REACT_DEFAULT_MAX_ITERATIONS
    react_default_deadline_seconds: float = REACT_DEFAULT_DEADLINE_SECONDS
    react_min_iters_before_free_finalize: int = REACT_MIN_ITERS_BEFORE_FREE_FINALIZE
    react_same_tool_failure_cap: int = REACT_SAME_TOOL_FAILURE_CAP
    react_max_fanout_branches: int = REACT_MAX_FANOUT_BRANCHES
    react_per_tool_invocation_cap: int = REACT_PER_TOOL_INVOCATION_CAP
    mcp_search_timeout_seconds: float = MCP_SEARCH_TIMEOUT_SECONDS
