"""ReAct mid-turn agent loop.

Layered on top of the existing plan-then-execute orchestrator. After the
planner's initial plan has executed and we have a first ``results`` dict,
this loop lets the model:

* **THINK** — write free-form reasoning to the scratchpad
* **ACT** — pick another tool from the registry to run (or ``"finalize"``)
* **OBSERVE** — see the structured summary of the tool's output

Loops until the model finalizes, the iteration cap is hit, or the
wall-clock deadline expires. Always bounded — there is no path where the
loop runs forever.

What this is NOT:

* It is not the planner. The planner still composes the initial plan
  and runs first; ReAct only kicks in *after* the initial plan, so simple
  queries don't pay any extra cost.
* It is not the synthesizer. The final answer is still written by the
  existing ``synthesize_answer`` pipeline. The loop just enriches the
  ``results`` dict that synthesis consumes.
* It is not a replacement for the existing critique. The critique
  (``reflection.llm_critique``) becomes one of the tools the model can
  choose during the loop, in addition to running once more post-synth.

What it adds:

* Inspectable working memory (the ``Scratchpad``).
* Adaptive tool use — the model sees the prior observations before
  picking the next ACTION, so it can detect a thin / contradictory /
  empty result and pivot.
* Stopping discipline — every iteration is judged against the cap +
  deadline; the model is also told when its next ACTION will be the
  last so it can prioritize verification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.assistant.scratchpad import Scratchpad
from app.assistant.tools.base import ToolContext, ToolResult
from app.assistant.tools.registry import describe_for_planner, get_tool

log = logging.getLogger(__name__)


# ── Tuneables ────────────────────────────────────────────────────────────────
# Conservative defaults. The orchestrator can override per call when it
# already knows the depth tier (e.g. push max_iterations down for "single").
_DEFAULT_MAX_ITERATIONS = 5
_DEFAULT_DEADLINE_SECONDS = 60.0
_FORCE_FINAL_THRESHOLD = 1   # last iteration auto-finalizes if model doesn't


# Tools the loop refuses to invoke from the ReAct cycle even when offered.
# Synthesis happens via the post-loop pipeline; calling ``memory_write`` as
# an ACTION is also disabled because consolidation is the post-turn job —
# letting the loop write durable memory mid-turn invites premature commits.
_DISALLOWED_FROM_LOOP: frozenset[str] = frozenset({
    "memory_write",
    "memory_delete",
})

# Special pseudo-actions the loop intercepts before tool dispatch. ``critique``
# runs ``reflection.llm_critique`` against the evidence the loop has collected
# so far and records the score on the scratchpad — giving the model a way to
# self-judge mid-flight rather than waiting for the post-synth critique pass.
_PSEUDO_ACTIONS: frozenset[str] = frozenset({"critique"})


# ── Decision schema ──────────────────────────────────────────────────────────
# Strict JSON contract for the per-iteration LLM decision. Anything outside
# this shape is treated as "finalize" so a malformed response never crashes
# the turn — the worst case is one wasted LLM call, not a broken loop.

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "maxLength": 2000},
        "action": {"type": "string", "maxLength": 80},   # tool name OR "finalize"
        "params": {"type": "object"},
        "rationale": {"type": "string", "maxLength": 600},
    },
    "required": ["thought", "action"],
}


@dataclass
class ReactConfig:
    """Knobs the orchestrator passes per call.

    The intent tier already decides whether the loop runs at all; these
    knobs let the same loop run with different ambitions depending on
    how much work the planner already did.
    """
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    deadline_seconds: float = _DEFAULT_DEADLINE_SECONDS
    namespace_key: str = ""
    expertise: str = "practitioner"
    orientation: str = "both"
    disabled_features: set[str] = field(default_factory=set)


@dataclass
class ReactOutcome:
    """Returned to the orchestrator at the end of the loop.

    The orchestrator merges ``new_results`` into its existing ``results``
    dict so the synthesizer sees the full evidence base. ``scratchpad``
    is persisted on the message payload for inspection.
    """
    scratchpad: Scratchpad
    new_results: dict[str, ToolResult]
    completed_normally: bool   # True if model said finalize; False if budget exhausted
    iterations: int


async def run_react_loop(
    *,
    query: str,
    initial_plan_actions: list[str],
    prior_results: dict[str, ToolResult],
    memory_view: dict[str, Any],
    research_brief_text: str,
    active_context: dict[str, Any] | None = None,
    ctx: ToolContext | None = None,
    ctx_factory: Any = None,   # async callable () -> AsyncContextManager[ToolContext]
    should_cancel: Any = None, # optional () -> Awaitable[bool]; checked between iterations
    config: ReactConfig,
    publish: Any = None,   # optional progress publisher (job_id, kind, payload) -> None
) -> ReactOutcome:
    """Run the ReAct loop after the initial plan has executed.

    Args:
        query: The original user query.
        initial_plan_actions: Human-readable summary of what the planner
            already did this turn (e.g. ``["arXiv search", "Deep Search"]``).
            Surfaced to the model so it doesn't redo work.
        prior_results: Results from the initial plan's tool executions —
            keyed by tool name, same shape as the orchestrator's per-turn
            ``results`` dict. NOT mutated here; new tool outputs are
            returned in ``ReactOutcome.new_results``.
        memory_view: Same memory dict the orchestrator passed to the
            planner / synthesizer. Read-only here.
        research_brief_text: The pre-planned research brief (deep tier
            only). Empty string is fine.
        ctx: A ``ToolContext`` the loop will pass through to every tool
            call. Owned by the caller (its lifecycle is the turn).
        config: Iteration + deadline + feature-gate knobs.
        publish: Optional event publisher so the UI can show the loop's
            progress in real time. Signature ``publish(kind: str, payload: dict)``.

    Returns:
        A :class:`ReactOutcome` with the scratchpad, any new tool results
        produced by the loop, whether the model finalized cleanly, and
        the iteration count. The orchestrator merges ``new_results`` into
        its own dict and runs the existing synthesis + final-critique
        pipeline against the union.

    Cancellation:
        ``asyncio.CancelledError`` propagates so the turn-level cancel
        handler can short-circuit cleanly. Anything in-progress is left
        for the cancel handler to mop up.

    Never raises for normal failures — a bad LLM response, a tool that
    errored, or a deadline hit just returns whatever scratchpad / new
    results were accumulated up to that point.
    """
    pad = Scratchpad()
    new_results: dict[str, ToolResult] = {}
    deadline = time.monotonic() + max(5.0, config.deadline_seconds)
    completed_normally = False
    iteration_count = 0

    # Seed the scratchpad with what the initial plan already did so the
    # model can reason about gaps without us re-summarising work.
    if initial_plan_actions:
        pad.think(
            "Initial plan already executed: "
            + ", ".join(initial_plan_actions)
            + ". I will reason about whether more retrieval / verification "
              "is needed, or finalize."
        )

    for i in range(config.max_iterations):
        if time.monotonic() > deadline:
            pad.think("Deadline reached — stopping the loop and finalizing.")
            break
        # Honour the user's Stop button. Without this check the loop
        # only respected the wall-clock deadline, so a 60s deadline
        # meant Stop took up to 60s to react. We surface the cancel
        # as a CancelledError so the orchestrator's existing
        # ``except asyncio.CancelledError`` path runs the standard
        # cleanup; ``Scratchpad.finish()`` still runs in the outer
        # ``finally`` (it doesn't exist yet — the ReactOutcome path
        # is the only one that calls ``finish``), so wrap our break
        # in a structured stop instead of raising mid-loop, which
        # would lose the work we already did.
        cancel_signal = False
        if should_cancel is not None:
            try:
                cancel_signal = bool(await should_cancel())
            except Exception:
                cancel_signal = False
        elif ctx is not None and getattr(ctx, "should_cancel", None) is not None:
            try:
                cancel_signal = bool(await ctx.should_cancel())
            except Exception:
                cancel_signal = False
        if cancel_signal:
            pad.think("Cancellation requested — stopping the loop and finalizing.")
            break

        pad.next_iteration()
        iteration_count = i + 1
        is_last_iteration = (i == config.max_iterations - 1)

        # ── Decide next action ───────────────────────────────────────
        try:
            decision = await _decide_next_action(
                query=query,
                pad=pad,
                prior_results=prior_results,
                new_results=new_results,
                memory_view=memory_view,
                research_brief_text=research_brief_text,
                active_context=active_context,
                config=config,
                is_last_iteration=is_last_iteration,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("react_loop: decision step failed (iter=%d): %s", i, exc)
            pad.think(f"Decision step failed: {exc}. Finalizing.")
            break

        if not decision:
            pad.think("LLM returned no decision — finalizing.")
            break

        thought = (decision.get("thought") or "").strip()
        action = (decision.get("action") or "finalize").strip()
        params = decision.get("params") or {}
        rationale = (decision.get("rationale") or "").strip()

        if thought:
            pad.think(thought)
            if publish:
                try:
                    publish("react_thought", {"iteration": pad.iteration, "text": thought[:400]})
                except Exception:
                    pass

        # ── Stop condition ───────────────────────────────────────────
        if action.lower() in {"finalize", "finish", "done", "stop", ""}:
            completed_normally = True
            pad.think("Loop finalized — handing off to synthesis.")
            break

        # ── Pseudo-action: self-critique ─────────────────────────────
        # The model can ask for a critique of the evidence collected so
        # far. We record the verdict on the scratchpad; a 'revise' verdict
        # is a strong signal to keep iterating, a 'ship' verdict nudges
        # the model to finalize. Critique runs over the prior+new tool
        # outputs, NOT over a draft answer — drafting still happens in
        # synthesis post-loop.
        if action.lower() == "critique":
            await _run_self_critique(
                query=query,
                pad=pad,
                prior_results=prior_results,
                new_results=new_results,
                memory_view=memory_view,
            )
            if publish:
                try:
                    publish("react_critique", {"iteration": pad.iteration})
                except Exception:
                    pass
            continue

        # ── Tool dispatch ────────────────────────────────────────────
        if action in _DISALLOWED_FROM_LOOP:
            pad.think(f"Tool '{action}' is not callable from the ReAct loop; finalizing instead.")
            completed_normally = True
            break

        # Avoid pointless redo of a tool the planner already ran with the
        # same params unless the model deliberately changed params.
        prior = prior_results.get(action) or new_results.get(action)
        if prior is not None and _params_equal(prior, params):
            pad.observe(
                tool=action,
                summary="Skipped — identical call already executed this turn.",
                output_ref=action,
                error=None,
            )
            continue

        tool = get_tool(action)
        if tool is None:
            pad.observe(tool=action, summary="Unknown tool — skipped.", output_ref="", error="tool_not_found")
            continue

        # Run the tool. We accept that this can be slow — the deadline
        # check at the top of the next iteration is the safety net.
        pad.act(tool=action, params=params, rationale=rationale)
        if publish:
            try:
                publish("react_action", {"iteration": pad.iteration, "tool": action, "rationale": rationale[:200]})
            except Exception:
                pass

        try:
            input_schema = tool.input_schema
            validated = input_schema(**params) if isinstance(params, dict) else params  # type: ignore[arg-type]
            # Per-action ToolContext — a single shared session held across
            # the whole loop would (a) be killed by cloud Postgres'
            # idle-in-transaction timeout (typically 60s) for any long
            # loop, and (b) silently lose any tool's flush-without-commit
            # writes when the outer block exits without a commit. The
            # factory hands us a fresh session per call; the orchestrator
            # commits inside the factory's contextmanager on success.
            if ctx_factory is not None:
                async with ctx_factory() as _action_ctx:
                    result: ToolResult = await tool.run(_action_ctx, validated)
            elif ctx is not None:
                result = await tool.run(ctx, validated)
            else:
                raise RuntimeError("react_loop: neither ctx nor ctx_factory provided")
            new_results[action] = result
            pad.observe(
                tool=action,
                summary=(result.summary or "(no summary)"),
                output_ref=action,
                error=None,
            )
            # Diminishing-returns guard. Two retrieval calls that
            # surface the same paper IDs add no information — keep
            # iterating burns latency for zero gain. Stop the loop
            # when we detect such a no-op call and let synthesis
            # work with what we already have.
            if _is_diminishing_returns(action, result, prior_results, new_results):
                pad.think(
                    f"'{action}' returned no new papers compared to prior retrievals — "
                    f"diminishing returns. Finalizing."
                )
                completed_normally = True
                break
            if publish:
                try:
                    publish("react_observation", {
                        "iteration": pad.iteration,
                        "tool": action,
                        "summary": (result.summary or "")[:240],
                    })
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("react_loop: tool '%s' raised: %s", action, exc)
            pad.observe(tool=action, summary=f"Tool error: {exc}", output_ref="", error=str(exc)[:300])

    pad.finish()
    return ReactOutcome(
        scratchpad=pad,
        new_results=new_results,
        completed_normally=completed_normally,
        iterations=iteration_count,
    )


# ── Internals ────────────────────────────────────────────────────────────────


async def _decide_next_action(
    *,
    query: str,
    pad: Scratchpad,
    prior_results: dict[str, ToolResult],
    new_results: dict[str, ToolResult],
    memory_view: dict[str, Any],
    research_brief_text: str,
    active_context: dict[str, Any] | None,
    config: ReactConfig,
    is_last_iteration: bool,
) -> dict[str, Any] | None:
    """One cheap-model call. Returns the parsed decision dict or None."""
    from app.adapters.llm import get_llm_adapter

    # Tool catalog — same view the planner uses, minus tools the loop
    # can't / shouldn't call (see ``_DISALLOWED_FROM_LOOP``).
    catalog = describe_for_planner(
        namespace_key=config.namespace_key,
        disabled_features=config.disabled_features,
    )
    catalog = [t for t in catalog if t.get("name") not in _DISALLOWED_FROM_LOOP]

    # Compact "what we already have" view.
    prior_summary = _summarise_results(prior_results) + _summarise_results(new_results)

    sys_msg = (
        "You are the reasoning engine of a research assistant in the MIDDLE of a turn.\n\n"
        "An initial plan has already executed. Your job each iteration is to either:\n"
        "  (a) call another tool to gather more evidence / verify a claim / fill a gap,\n"
        "  (b) call action 'critique' to self-judge whether the evidence is sufficient\n"
        "      and well-grounded (records a verdict + issues on the scratchpad), OR\n"
        "  (c) call action 'finalize' to hand off to synthesis.\n\n"
        "Use 'critique' when you have enough evidence and want to verify before "
        "finalizing — it costs one cheap LLM call and helps catch unsupported "
        "claims or thin coverage before the answer is drafted.\n\n"
        "Decide based on whether the evidence so far is SUFFICIENT, GROUNDED, and ON-TOPIC. "
        "Prefer 'finalize' when:\n"
        "  - You have enough strong evidence to answer the user's question.\n"
        "  - Additional retrieval is unlikely to materially change the answer.\n"
        "  - You've already tried 2+ retrievals and results are not improving.\n\n"
        "Prefer calling a tool when:\n"
        "  - A specific claim lacks support and a targeted search/verification helps.\n"
        "  - Prior results were thin, empty, or contradictory.\n"
        "  - The user asked for something the existing tool outputs do not cover.\n\n"
        "DO NOT redo work already done. DO NOT call the same tool with the same params twice. "
        "DO NOT call tools that don't help answer the question.\n\n"
        "Return strict JSON: {thought, action, params, rationale}. "
        "'action' is either a tool name from the catalog, or the string 'finalize'."
    )
    if is_last_iteration:
        sys_msg += (
            "\n\nIMPORTANT: this is your LAST iteration. Prefer finalize unless a "
            "single verification call is critically necessary."
        )

    catalog_text = "\n".join(
        f"- {t['name']}: {t['summary'][:200]}" for t in catalog[:30]
    )

    # Active-context block — when the user has uploaded notes / PDFs /
    # URLs / paper-refs into this session, advertise them here so the
    # model can decide to run ``parse_context`` instead of pretending
    # nothing was attached. Empty inventory renders as a single line so
    # the prompt stays short on the common path.
    ac_total = int((active_context or {}).get("total") or 0)
    if ac_total > 0:
        kinds = (active_context or {}).get("kinds") or {}
        labels = (active_context or {}).get("labels") or []
        kinds_str = ", ".join(f"{k}={v}" for k, v in kinds.items()) or "(unknown)"
        labels_preview = "; ".join(labels[:5]) if labels else ""
        active_ctx_block = (
            f"ACTIVE CONTEXT (user attached these — read with parse_context if relevant):\n"
            f"  total={ac_total}; kinds=[{kinds_str}]"
            + (f"; labels=[{labels_preview}]" if labels_preview else "")
        )
    else:
        active_ctx_block = "ACTIVE CONTEXT: (none — user has not attached any documents this session)"

    user_msg = (
        f"USER QUERY:\n{query[:1500]}\n\n"
        f"RESEARCH BRIEF:\n{(research_brief_text or '(none)')[:1500]}\n\n"
        f"{active_ctx_block}\n\n"
        f"TOOL CATALOG (you may call any of these):\n{catalog_text}\n\n"
        f"WHAT THE INITIAL PLAN PRODUCED:\n{prior_summary[:2000]}\n\n"
        f"SCRATCHPAD SO FAR:\n{pad.render_for_prompt()}\n\n"
        "Now decide your next ACTION."
    )

    try:
        llm = get_llm_adapter()
        return await llm.complete_structured(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            llm.cheap_model,
            _DECISION_SCHEMA,
        )
    except Exception as exc:
        log.debug("react_loop: structured LLM call failed: %s", exc)
        return None


def _summarise_results(results: dict[str, ToolResult]) -> str:
    """Compact one-line-per-tool summary for the LLM prompt."""
    if not results:
        return ""
    lines: list[str] = []
    for name, r in results.items():
        try:
            line = f"  - {name}: {(r.summary or '(no summary)')[:240]}"
        except Exception:
            line = f"  - {name}: <unrenderable>"
        lines.append(line)
    return "\n".join(lines) + "\n"


async def _run_self_critique(
    *,
    query: str,
    pad: Scratchpad,
    prior_results: dict[str, ToolResult],
    new_results: dict[str, ToolResult],
    memory_view: dict[str, Any],
) -> None:
    """Run ``reflection.llm_critique`` against the evidence collected so far.

    Records the verdict + scores + issues on the scratchpad as a
    ``Critique`` entry. Never raises — on any failure the scratchpad
    just gets a single ``thought`` noting the critique was skipped.

    The critique is evidence-level, not draft-level: we feed the
    structured tool outputs (papers, comparisons, web results) into the
    judge so it can score groundedness / completeness *before* the
    synthesizer is invoked. Catches "we have 1 paper and the user
    asked for a literature survey" situations early.
    """
    try:
        from app.assistant.reflection import llm_critique

        # Compose a compact evidence excerpt from the merged results.
        merged = {**prior_results, **new_results}
        excerpt_lines: list[str] = []
        for name, r in list(merged.items())[:12]:
            try:
                summary = (r.summary or "")[:600]
                excerpt_lines.append(f"[{name}] {summary}")
            except Exception:
                continue
        evidence_excerpt = "\n".join(excerpt_lines)

        # Memory hint — the critique judge already accepts a memory excerpt;
        # we pass a tier-collapsed snapshot so it can flag answers that
        # contradict durable memory.
        mem_lines: list[str] = []
        for tier_key in ("medium", "long", "branches"):
            tier = memory_view.get(tier_key) or {}
            if not tier:
                continue
            for k, v in list(tier.items())[:6]:
                if isinstance(v, dict):
                    mem_lines.append(f"[{tier_key}] {k}: {(v.get('value') or v.get('summary') or '')[:200]}")
        memory_excerpt = "\n".join(mem_lines)

        # Pre-draft critique — pass an empty answer; the judge scores the
        # evidence base itself so we can decide whether to gather more.
        critique = await llm_critique(
            query=query,
            answer="(pre-draft — evaluate evidence sufficiency only)",
            evidence_excerpt=evidence_excerpt or "(no evidence yet)",
            memory_excerpt=memory_excerpt,
        )
        if not critique:
            pad.think("Self-critique returned no verdict — proceeding.")
            return

        verdict = "revise" if critique.get("should_repair") else "ship"
        pad.critique(
            groundedness=float(critique.get("groundedness") or 0.0),
            completeness=float(critique.get("completeness") or 0.0),
            memory_faithfulness=float(critique.get("memory_faithfulness") or 1.0),
            issues=[str(i) for i in (critique.get("issues") or [])],
            verdict=verdict,  # type: ignore[arg-type]
        )
    except Exception as exc:
        log.debug("react_loop: self-critique failed: %s", exc)
        pad.think(f"Self-critique skipped ({exc}).")


_RETRIEVAL_TOOLS: frozenset[str] = frozenset({
    "deep_search",
    "arxiv_search",
    "arxiv_import",
    "frontier_scan",
    "literature_survey",
    "pubmed",
    "inspire_hep",
    "nasa_ads",
    "semantic_scholar",
    "huggingface_search",
    "github_search",
    "papers_with_code",
    "citation_finder",
})


def _extract_paper_ids(result: ToolResult) -> set[str]:
    """Pull the paper-id set out of a ToolResult's output dict.

    Tolerant of the schema variations across retrieval tools — some
    surface ``papers: [{paper_id: ...}]``, others ``results: [...]``,
    others a flat ``ids`` list. Returns an empty set when the shape
    isn't a recognised retrieval result, which makes the guard a
    no-op for non-retrieval tools.
    """
    try:
        out = result.output or {}
        candidates: list = []
        for key in ("papers", "results", "items", "candidates"):
            v = out.get(key)
            if isinstance(v, list):
                candidates = v
                break
        ids: set[str] = set()
        for c in candidates:
            if not isinstance(c, dict):
                continue
            pid = c.get("paper_id") or c.get("id") or c.get("external_id")
            if pid:
                ids.add(str(pid))
        return ids
    except Exception:
        return set()


def _is_diminishing_returns(
    action: str,
    result: ToolResult,
    prior_results: dict[str, ToolResult],
    new_results: dict[str, ToolResult],
) -> bool:
    """Return True when ``result`` adds no new paper IDs vs. earlier calls.

    Only applies to retrieval-class tools — verification / compare /
    explain tools don't surface paper IDs, so the guard is a no-op
    for them. The new call is considered redundant when the union
    of paper IDs from every PRIOR retrieval already covers every
    paper this call returned.

    ``prior_results`` is the orchestrator's frozen pre-loop snapshot
    (everything the initial plan produced) and is consulted in full —
    a duplicate same-tool retrieval is exactly the case we want to
    catch. ``new_results`` is the loop's own accumulating dict; we
    exclude the entry under the *current* action because the caller
    has already written the freshly-computed ``result`` there.
    """
    if action not in _RETRIEVAL_TOOLS:
        return False
    new_ids = _extract_paper_ids(result)
    if not new_ids:
        # Empty or unparseable result — let the model decide whether
        # to keep trying (e.g. it may want to broaden the query).
        return False
    prior_ids: set[str] = set()
    # All prior retrieval calls (initial plan, any earlier loop iterations).
    for name, r in prior_results.items():
        if name in _RETRIEVAL_TOOLS:
            prior_ids.update(_extract_paper_ids(r))
    # Loop's own results — exclude the slot the caller just wrote.
    for name, r in new_results.items():
        if name == action:
            continue
        if name in _RETRIEVAL_TOOLS:
            prior_ids.update(_extract_paper_ids(r))
    return bool(prior_ids) and new_ids.issubset(prior_ids)


def _params_equal(prior: ToolResult, candidate: dict) -> bool:
    """Return True iff ``candidate`` matches the params recorded on a prior
    ToolResult (best-effort — ToolResult doesn't always carry input params,
    so we only assert equality when both sides have a recorded shape).
    """
    try:
        prior_params = (prior.output or {}).get("__input_params") if isinstance(prior.output, dict) else None
    except Exception:
        prior_params = None
    if prior_params is None:
        return False
    try:
        return json.dumps(prior_params, sort_keys=True) == json.dumps(candidate or {}, sort_keys=True)
    except Exception:
        return False
