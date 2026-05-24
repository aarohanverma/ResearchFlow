"""Nested ReAct loop runner for subagent dispatch.

When the parent picks ``action="subagent"`` with a ``subagent_name`` +
``task``, this module is what actually runs. It:

  1. Looks up the spec.
  2. Builds a restricted tool catalog matching the spec's
     ``allowed_tools`` (intersection with what the parent had
     visible — a subagent can never see a tool the parent's user
     couldn't).
  3. Spawns a fresh :class:`LoopState` with a focused query (the
     task), the subagent's role prompt prepended, a *new* scratchpad,
     and a tighter iteration / deadline cap.
  4. Runs the nested loop using the same middleware chain (so the
     subagent benefits from every cross-cutting concern — param
     hygiene, ban policy, observability, etc.).
  5. On completion, distills the subagent's final state into a
     :class:`SubAgentResult` carrying a structured summary and the
     paper IDs surfaced.

The subagent's scratchpad is persisted on the parent's task object
(``task._subagent_scratchpads``) so the UI can render a tree of
"main agent ran" → "researcher subagent ran" / "comparator subagent
ran". The parent's own scratchpad only sees a single Observation
saying "researcher subagent ran" + the summary.

Cancellation propagates through the same ``should_cancel`` callable
the parent uses — cancelling the user turn cancels every active
subagent.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.assistant.react.subagents import (
    SubAgentResult,
    SubAgentSpec,
    get_subagent,
)
from app.assistant.scratchpad import Scratchpad
from app.assistant.tools.registry import describe_for_planner

log = logging.getLogger(__name__)


async def run_subagent(
    *,
    parent_state: Any,        # LoopState — forward ref to avoid cycle
    subagent_name: str,
    task: str,
) -> SubAgentResult:
    """Run a nested ReAct loop scoped to ``subagent_name``.

    Cancellation: inherited from the parent's ``should_cancel``. The
    nested loop respects the same wall-clock cancellation signal so
    cancelling the user turn unwinds every active subagent cleanly.

    Tool catalog: ``parent_visible ∩ spec.allowed_tools``. We never
    expand a subagent beyond the parent's visibility — if a feature
    flag hid a tool from the parent, the subagent doesn't see it.
    """
    spec = get_subagent(subagent_name)
    if spec is None:
        return SubAgentResult(
            subagent_name=subagent_name,
            summary=f"Unknown subagent '{subagent_name}'. Available: "
                    + ", ".join(sorted(_known_names())),
            completed_normally=False,
        )

    # Defer the import to avoid a cycle with the loop driver.
    from app.assistant.react_loop import ReactConfig, run_react_loop

    parent_config = parent_state.config

    # Restrict the catalog visible to the subagent. We do this by
    # constructing a feature-flag-style disabled set: every visible
    # tool that's NOT in spec.allowed_tools.
    parent_visible: set[str] = {
        t["name"] for t in describe_for_planner(
            namespace_key=parent_config.namespace_key,
            disabled_features=parent_config.disabled_features,
        )
    }
    allowed = parent_visible & spec.allowed_tools
    disabled_features_for_sub = _hide_tools(
        parent_config.disabled_features, parent_visible - allowed,
    )

    # Subagents inherit their wall-clock budget from the parent's
    # remaining time, capped at the spec's preferred deadline. When
    # the parent has effectively zero budget left (deadline already
    # tripped), spawning a subagent at all is wasted work — return a
    # quick "skipped" result instead of letting it eat another second.
    parent_remaining = parent_state.time_remaining()
    if parent_remaining < 2.0:
        log.info(
            "subagent '%s' skipped: parent deadline budget too low (%.1fs)",
            spec.name, parent_remaining,
        )
        return SubAgentResult(
            subagent_name=spec.name,
            summary=(
                f"Subagent '{spec.name}' skipped — parent deadline budget too "
                f"low ({parent_remaining:.1f}s remaining). Try again with more "
                "budget or finalize on current evidence."
            ),
            completed_normally=False,
        )

    sub_config = ReactConfig(
        max_iterations=min(spec.max_iterations, max(1, parent_config.max_iterations - 1)),
        deadline_seconds=min(spec.deadline_seconds, parent_remaining),
        namespace_key=parent_config.namespace_key,
        expertise=parent_config.expertise,
        orientation=parent_config.orientation,
        # Feature-flag the tools we want hidden. The factory turns
        # this into a per-call ``disabled_features`` set the catalog
        # respects, mirroring how the parent's flags work.
        disabled_features=disabled_features_for_sub,
    )

    # The role prompt goes in the system message (via subagent_role)
    # so the nested loop sees ONE coherent role instead of the
    # parent's generic prompt + a role string buried in the query.
    # The query is just the task — that's what shows up under "USER
    # QUERY" in the decision prompt.
    focused_query = task

    started = time.monotonic()
    outcome = await run_react_loop(
        query=focused_query,
        initial_plan_actions=[],
        prior_results={},        # context-quarantine: no parent results
        memory_view={},          # subagents don't read durable memory
        research_brief_text="",
        active_context=None,
        ctx_factory=parent_state.ctx_factory,
        ctx=parent_state.ctx,
        should_cancel=parent_state.should_cancel,
        config=sub_config,
        publish=None,            # subagent events stay private
        # Recursion-prevention: bump the depth so the nested loop's
        # decision prompt hides the subagent catalog and the loop
        # refuses any nested subagent dispatch attempt.
        subagent_depth=int(getattr(parent_state, "subagent_depth", 0)) + 1,
        # Role-prompt routing: the spec's role becomes the system
        # message header, replacing the generic "you are RA" prompt.
        subagent_role=spec.role_prompt,
    )
    elapsed = time.monotonic() - started
    log.info(
        "subagent '%s' finished: iterations=%d elapsed=%.1fs successful_retrievals=%d",
        spec.name, outcome.iterations, elapsed, outcome.successful_retrievals,
    )

    return _distill_outcome(spec=spec, outcome=outcome)


# ── Internals ────────────────────────────────────────────────────────────────


def _known_names() -> set[str]:
    from app.assistant.react.subagents import SUBAGENT_REGISTRY
    return set(SUBAGENT_REGISTRY.keys())


# Sentinel used in ``disabled_features`` to indicate "this is a
# tool-name hide, not a feature flag." The tool catalog renderer
# treats tool-name strings prefixed with "tool:" as a per-call
# tool-name blocklist.
_TOOL_HIDE_PREFIX = "tool:"


def _hide_tools(existing_flags: set[str], hide_names: set[str]) -> set[str]:
    """Merge existing disabled feature flags with tool-name hides.

    The catalog respects ``"tool:<name>"`` as a per-call hide so we
    don't have to widen the feature-flag model just to scope a
    subagent's catalog. The catalog renderer (``describe_for_planner``)
    is updated to recognise this prefix.
    """
    flags = set(existing_flags or set())
    for name in hide_names:
        flags.add(f"{_TOOL_HIDE_PREFIX}{name}")
    return flags


def _distill_outcome(*, spec: SubAgentSpec, outcome: Any) -> SubAgentResult:
    """Distill a nested ReAct outcome into a SubAgentResult.

    The summary is a synthesis of:
      * the last critique verdict if one fired,
      * the count of papers the subagent surfaced,
      * the last meaningful Thought entry from the scratchpad
        (typically the model's "I have enough; finalizing" reasoning).

    We deliberately do NOT call the synthesizer for subagent output —
    that would inflate cost and the parent doesn't need a polished
    answer, it needs evidence pointers.
    """
    pad: Scratchpad = outcome.scratchpad
    paper_ids = list(outcome.new_results and _collect_paper_ids(outcome) or [])

    # Walk thoughts in reverse to find the last informative one.
    # We filter out housekeeping thoughts (param repairs, gate fires,
    # finalize handoff) because they don't carry the subagent's actual
    # research finding — the parent doesn't need to know about them.
    _NOISE_PREFIXES = (
        "Auto-repaired", "Forcing a", "Loop finalized",
        "Initial plan already executed", "Decision step failed",
        "Deadline reached", "Cancellation requested",
        "Adaptive counter-search", "Validation-fallback repair",
        "Skipped", "Finalize gated",
    )
    final_thought = ""
    for entry in reversed(pad.entries):
        if getattr(entry, "kind", "") != "thought":
            continue
        txt = (entry.text or "").strip()
        if not txt:
            continue
        if any(txt.startswith(prefix) for prefix in _NOISE_PREFIXES):
            continue
        final_thought = txt
        break

    last_critique = None
    for entry in pad.entries:
        if getattr(entry, "kind", "") == "critique":
            last_critique = entry

    summary_bits: list[str] = []
    if final_thought:
        summary_bits.append(final_thought[:240])
    if paper_ids:
        summary_bits.append(f"Surfaced {len(paper_ids)} paper ID(s).")
    if last_critique:
        verdict = getattr(last_critique, "verdict", "")
        g = float(getattr(last_critique, "groundedness", 0.0) or 0.0)
        summary_bits.append(f"Self-critique: verdict={verdict} g={g:.2f}")
    summary = " ".join(summary_bits) or f"{spec.name} subagent ran ({outcome.iterations} iter)"

    structured: dict[str, Any] = {}
    if spec.response_schema:
        # Best-effort schema population — the response_schema is a
        # *contract*, not a strict validation. We fill what we can
        # from the scratchpad; the parent can read missing keys as
        # "subagent didn't produce that field."
        for key, _desc in spec.response_schema:
            if key == "summary":
                structured[key] = summary
            elif key in ("paper_ids", "papers"):
                structured[key] = paper_ids
            elif key == "objections" and last_critique:
                structured[key] = list(getattr(last_critique, "issues", []) or [])

    return SubAgentResult(
        subagent_name=spec.name,
        summary=summary,
        structured=structured,
        paper_ids_surfaced=paper_ids,
        iterations=outcome.iterations,
        completed_normally=outcome.completed_normally,
        scratchpad=pad,
    )


def _collect_paper_ids(outcome: Any) -> list[str]:
    """Pull paper IDs from every retrieval-shaped result the subagent
    produced. Deduped, order-preserving."""
    seen: set[str] = set()
    out: list[str] = []
    for result in (outcome.new_results or {}).values():
        for key in ("papers", "results", "items", "candidates"):
            col = (result.output or {}).get(key) if isinstance(result.output, dict) else None
            if not isinstance(col, list):
                continue
            for c in col:
                if not isinstance(c, dict):
                    continue
                pid = c.get("paper_id") or c.get("id") or c.get("external_id")
                if not pid:
                    continue
                pid = str(pid)
                if pid in seen:
                    continue
                seen.add(pid)
                out.append(pid)
            break
    return out
