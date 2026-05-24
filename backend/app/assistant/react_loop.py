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
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.assistant.contradiction import (
    ContradictionLedger,
    detect_contradictions_in_results,
    detect_semantic_contradictions,
)
from app.assistant.retrieval_observability import RetrievalObservability
from app.assistant.scratchpad import Scratchpad
from app.assistant.tools.base import ToolContext, ToolResult
from app.assistant.tools.registry import describe_for_planner, get_tool

log = logging.getLogger(__name__)


# ── Tuneables ────────────────────────────────────────────────────────────────
# Conservative defaults. The orchestrator can override per call when it
# already knows the depth tier (e.g. push max_iterations down for "single").
_DEFAULT_MAX_ITERATIONS = 8
_DEFAULT_DEADLINE_SECONDS = 90.0
_FORCE_FINAL_THRESHOLD = 1   # last iteration auto-finalizes if model doesn't
# Minimum reasoning depth before the model is allowed to finalize without
# at least one self-critique. A turn that finalized on iteration 1-2 with
# no critique recorded got far too little adversarial pressure; we
# transparently insert a critique step so the synth gets a real
# groundedness / completeness score to honour.
_MIN_ITERS_BEFORE_FREE_FINALIZE = 3


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
        # ``action`` is either a tool name, ``"finalize"``, ``"critique"``,
        # ``"fanout"`` for parallel branches, ``"subagent"`` for context-
        # quarantined delegation, or ``"write_todos"`` to update the
        # investigation plan.
        "action": {"type": "string", "maxLength": 80},
        "params": {"type": "object"},
        # When ``action == "fanout"``, the model emits a ``branches``
        # array. Each branch becomes one parallel tool dispatch.
        "branches": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "maxLength": 80},
                    "params": {"type": "object"},
                    "rationale": {"type": "string", "maxLength": 400},
                },
                "required": ["tool"],
            },
        },
        # When ``action == "write_todos"``, ``todos`` is a list of
        # structured operations applied in order. Each op has
        # ``kind`` ∈ {add, update, complete, cancel, clear} and the
        # relevant fields (``id``, ``text``, ``status``,
        # ``evidence``). See InvestigationPlan.apply_operations for
        # the contract.
        "todos": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "maxLength": 20},
                    "id": {"type": "string", "maxLength": 16},
                    "text": {"type": "string", "maxLength": 280},
                    "status": {"type": "string", "maxLength": 20},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 80},
                        "maxItems": 8,
                    },
                },
                "required": ["kind"],
            },
        },
        "rationale": {"type": "string", "maxLength": 600},
    },
    "required": ["thought", "action"],
}

# Hard cap on parallel branches so a hallucinated fanout doesn't blow
# the iteration budget. Two-to-four is the sweet spot — enough to
# investigate a real comparison query, low enough that the LLM tax on
# the join step stays manageable.
_MAX_FANOUT_BRANCHES = 4


# ── Param hygiene helpers ───────────────────────────────────────────────────
#
# The model frequently emits placeholders like ``__to_fill_from_retrieval__``
# or leaves the dispatching dict empty entirely, which then trips pydantic
# validation downstream and produces an opaque "query field required" error
# instead of a useful retrieval. These helpers (a) render the JSON schema of
# each tool into the catalog the model sees, (b) detect placeholder values,
# (c) auto-fill missing required fields from the user query / paper ledger,
# and (d) get re-applied on validation error so a bad first attempt is
# repaired rather than discarded.

_PLACEHOLDER_PATTERNS = re.compile(
    r"""(?ix)              # case-insensitive, verbose
    ^\s*(?:
        __[a-z0-9_]+__?                |   # __to_fill_*__ / __fill__
        <{1,2}\s*(?:todo|fill|placeholder|tbd|fixme|xxx)[^>]*>{1,2}? |
        \{\s*(?:todo|fill|placeholder|tbd|fixme|xxx)[^}]*\}    |
        \[\s*(?:todo|fill|placeholder|tbd|fixme|xxx)[^\]]*\]  |
        (?:n/?a|tbd|todo|fixme|null|none|undefined|fill_me|fill_in|\?{2,})\s*$
    )
    """,
)


def _looks_like_placeholder(value: Any) -> bool:
    """Return True when ``value`` looks like a model-emitted placeholder.

    Caught:
      - ``"__to_fill_from_retrieval__"`` and ``__like_this__``
      - ``"<TODO>"`` / ``<<fill>>`` / ``{placeholder}`` / ``[tbd]``
      - Bare ``"null"`` / ``"None"`` / ``"undefined"`` strings
      - Empty / whitespace-only strings
      - Lists where every element is itself a placeholder

    Numbers, bools, and well-formed values pass through unchanged.
    """
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        return bool(_PLACEHOLDER_PATTERNS.match(stripped))
    if isinstance(value, (list, tuple)):
        if not value:
            return False  # empty list is legal default for many fields
        return all(_looks_like_placeholder(v) for v in value)
    return False


# Field-name heuristics for auto-fill. These are deliberately scoped to the
# tool input vocabulary we control (every retrieval tool's required text
# field is named one of these). Adding a new retrieval tool with a different
# field name is a one-line addition here.
_QUERY_LIKE_FIELDS: frozenset[str] = frozenset({
    "query", "question", "claim", "topic", "search_query", "text", "prompt",
})
_PAPER_ID_LIST_FIELDS: frozenset[str] = frozenset({"paper_ids", "ids"})
_PAPER_ID_SINGLE_FIELDS: frozenset[str] = frozenset({"paper_id", "id"})


@dataclass
class PaperLedger:
    """Running inventory of paper IDs surfaced by retrieval tools this turn.

    The model needs concrete paper IDs to call ``compare_papers`` /
    ``paper_qa`` / ``genie_synthesize`` etc. Without a ledger it ends up
    emitting placeholders like ``__to_fill_from_retrieval__`` because it
    has no way to refer to "the papers we just retrieved".

    Populated by :meth:`add_from_result` after every tool dispatch (cheap
    no-op for non-retrieval tools) and rendered into the decision prompt
    so the model can copy concrete IDs into the next ACTION's params.
    """

    by_id: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)

    def add_from_result(self, result: ToolResult) -> int:
        """Merge any paper records from ``result.output`` into the ledger.

        Returns the count of NEW IDs added (zero when the tool surfaced no
        papers or only repeats). Order is preserved so the model gets the
        most relevant papers first when we render the top-K view.
        """
        added = 0
        try:
            out = result.output or {}
            candidates: list = []
            for key in ("papers", "results", "items", "candidates"):
                v = out.get(key)
                if isinstance(v, list):
                    candidates = v
                    break
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                pid = c.get("paper_id") or c.get("id") or c.get("external_id")
                if not pid:
                    continue
                pid = str(pid)
                if pid in self.by_id:
                    continue
                self.by_id[pid] = {
                    "title": (c.get("title") or "")[:160],
                    "ns": c.get("namespace_key") or c.get("namespace") or "",
                }
                added += 1
        except Exception:
            return added
        return added

    def ids(self, limit: int | None = None) -> list[str]:
        out = list(self.by_id.keys())
        return out[:limit] if limit is not None else out

    def render_for_prompt(self, limit: int = 12) -> str:
        """Compact view for the decision prompt. ``(none)`` when empty so
        the model knows the ledger is real but has not been populated yet.
        """
        if not self.by_id:
            return "(no papers retrieved yet — call a retrieval tool before compare/paper_qa/synthesize)"
        items = list(self.by_id.items())
        head = items[:limit]
        lines: list[str] = []
        for pid, info in head:
            title = (info.get("title") or "(untitled)")[:120]
            ns = info.get("ns") or ""
            ns_str = f" [{ns}]" if ns else ""
            lines.append(f"  - id={pid}{ns_str} title={title}")
        if len(items) > limit:
            lines.append(f"  ... and {len(items) - limit} more")
        return "\n".join(lines)


def _render_tool_catalog(catalog: list[dict], limit: int = 30) -> str:
    """Render the tool catalog with required + optional params from each
    tool's JSON input schema so the model emits valid ``params`` dicts.

    The screenshots showed the model emitting ``params={}`` to tools
    requiring a ``query`` field — that was because the catalog only
    advertised the tool name + summary. Surfacing the schema fields
    inline removes the guesswork; we keep each tool's block compact
    (one summary line + one params line) so the prompt stays cheap.
    """
    lines: list[str] = []
    for t in catalog[:limit]:
        name = t.get("name", "?")
        summary = (t.get("summary") or "")[:220]
        schema = t.get("input_schema") or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])

        # Required first, then optional, both alphabetised for stability.
        req_parts: list[str] = []
        opt_parts: list[str] = []
        for prop_name in sorted(props.keys()):
            prop = props[prop_name] or {}
            ptype = prop.get("type") or _infer_type(prop)
            desc = (prop.get("description") or "")[:90]
            default = prop.get("default", _MISSING)
            piece = f"{prop_name}({ptype}"
            if prop_name in required:
                piece += ", required"
            elif default is not _MISSING:
                piece += f", default={_compact_default(default)}"
            piece += ")"
            if desc:
                piece += f"—{desc}"
            (req_parts if prop_name in required else opt_parts).append(piece)

        params_line = ""
        if req_parts or opt_parts:
            joined = "; ".join(req_parts + opt_parts)
            if len(joined) > 600:
                joined = joined[:597] + "..."
            params_line = f"\n    params: {joined}"
        lines.append(f"- {name}: {summary}{params_line}")
    return "\n".join(lines)


_MISSING = object()


def _infer_type(prop: dict) -> str:
    """Best-effort type string for JSON schema entries that use ``anyOf`` /
    ``allOf`` instead of a plain ``type`` field."""
    if "anyOf" in prop:
        inner = [p.get("type") for p in prop["anyOf"] if isinstance(p, dict)]
        inner = [t for t in inner if t]
        if inner:
            return "|".join(sorted(set(inner)))
    if "$ref" in prop:
        return prop["$ref"].rsplit("/", 1)[-1]
    return "any"


def _compact_default(value: Any) -> str:
    """One-line stringification of a default value for the catalog view."""
    try:
        s = json.dumps(value, ensure_ascii=False)
    except Exception:
        s = str(value)
    return s if len(s) <= 24 else s[:23] + "…"


def _preflight_and_repair_params(
    action: str,
    raw_params: dict,
    schema: dict,
    *,
    query: str,
    ledger: PaperLedger,
) -> tuple[dict, list[str]]:
    """Strip placeholders and auto-fill missing required fields.

    Returns ``(repaired, notes)``. ``notes`` is a list of human-readable
    repair actions ("auto-filled 'query' from user query") that get folded
    into the action's scratchpad entry so the model can see what got
    rewritten — without that, the model would re-emit the same broken
    params on the next iteration.

    Fill rules:
      * ``query`` / ``question`` / ``claim`` / ``topic`` ← user query
      * ``paper_ids`` ← ledger.ids()[:3] when ≥2 papers retrieved
      * ``paper_id`` ← ledger.ids()[0] when ≥1 paper retrieved
      * Anything else stays None / unset — pydantic will surface the
        missing-field error and the loop will record it as a clear
        observation rather than auto-faking a value we don't have.
    """
    repaired = dict(raw_params or {})
    notes: list[str] = []
    props = schema.get("properties") or {}
    required = list(schema.get("required") or [])

    # 1. Strip placeholders from every supplied key (keeps real values).
    for k in list(repaired.keys()):
        v = repaired[k]
        if _looks_like_placeholder(v):
            del repaired[k]
            preview = (str(v)[:40]) if v is not None else "None"
            notes.append(f"removed placeholder {k}={preview!r}")

    # 2. Auto-fill missing required fields where we have a fact to fill with.
    for k in required:
        if k in repaired and not _looks_like_placeholder(repaired[k]):
            continue
        if k in _QUERY_LIKE_FIELDS and query:
            repaired[k] = query[:480]
            notes.append(f"auto-filled '{k}' from user query")
        elif k in _PAPER_ID_LIST_FIELDS:
            prop = props.get(k) or {}
            min_items = int(prop.get("minItems") or prop.get("min_length") or 1)
            ids = ledger.ids(limit=max(min_items, 3))
            if len(ids) >= min_items:
                repaired[k] = ids[:max(min_items, 2)]
                notes.append(f"auto-filled '{k}' from ledger: {repaired[k]}")
        elif k in _PAPER_ID_SINGLE_FIELDS:
            ids = ledger.ids(limit=1)
            if ids:
                repaired[k] = ids[0]
                notes.append(f"auto-filled '{k}' from ledger: {ids[0]}")

    return repaired, notes


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

    ``tool_failures`` and ``successful_retrievals`` are the signals the
    synthesizer reads to decide whether to downgrade confidence — if
    the loop tried to expand evidence and several tools errored / no
    new papers landed, the synth must say so out loud instead of
    polishing past it.
    """
    scratchpad: Scratchpad
    new_results: dict[str, ToolResult]
    completed_normally: bool   # True if model said finalize; False if budget exhausted
    iterations: int
    tool_failures: int = 0
    successful_retrievals: int = 0
    paper_ledger_size: int = 0
    # Retrieval observability + contradiction state, serialised as
    # plain dicts so the orchestrator can hand them to the synthesizer
    # without leaking dataclass types across module boundaries.
    retrieval_metrics: dict[str, Any] = field(default_factory=dict)
    contradiction_signals: list[dict[str, Any]] = field(default_factory=list)
    # Mid-loop investigation tracker — todos the model declared via
    # ``write_todos`` plus their completion state at finalize. The
    # synthesizer reads ``open`` and ``stuck_in_progress`` to surface
    # unfinished work honestly in the answer.
    investigation_plan: dict[str, Any] = field(default_factory=dict)


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
    # ── Subagent context (only set by the subagent runner) ──────────
    # ``subagent_depth`` ≥ 1 means this is a nested loop run as a
    # subagent. The depth gate stops a subagent from spawning another
    # subagent (which would defeat context quarantine). ``subagent_role``
    # is the spec's role prompt; when set, it replaces the generic
    # system message so the model sees a single coherent role.
    subagent_depth: int = 0,
    subagent_role: str | None = None,
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
    # ── Build the per-turn state object the middleware chain reads ──
    # Hoisting everything onto ``state`` lets each middleware reach
    # exactly the fields it needs without us having to plumb a dozen
    # arguments through every hook call.
    from app.assistant.react import LoopState
    from app.assistant.react.middleware import (
        AbortDispatch,
        DispatchOverride,
        FinalizeAllow,
        FinalizeForceAction,
        FinalizeForceCritique,
        MiddlewareChain,
    )
    from app.assistant.react.middlewares import default_chain_factory
    from app.assistant.react.subagent_runner import run_subagent
    from app.assistant.react.subagents import (
        SUBAGENT_REGISTRY,
        describe_subagents_for_prompt,
    )

    state = LoopState(
        query=query,
        initial_plan_actions=initial_plan_actions,
        prior_results=prior_results or {},
        memory_view=memory_view or {},
        research_brief_text=research_brief_text or "",
        active_context=active_context,
        ctx=ctx,
        ctx_factory=ctx_factory,
        should_cancel=should_cancel,
        publish=publish,
        config=config,
        deadline=time.monotonic() + max(5.0, config.deadline_seconds),
        subagent_depth=int(subagent_depth or 0),
        subagent_role=subagent_role,
    )
    state.ledger = PaperLedger()

    # Pre-populate the ledger + contradiction ledger from anything the
    # initial plan already retrieved. The first ReAct iteration can
    # then issue compare_papers / paper_qa with concrete IDs, and we
    # don't ignore disagreements that landed before the loop started.
    for r in (state.prior_results or {}).values():
        try:
            state.ledger.add_from_result(r)
        except Exception:  # noqa: BLE001 — ledger seed must never abort startup
            pass
    try:
        for _sig in detect_contradictions_in_results(state.prior_results, iteration=0):
            state.contradictions.add(_sig)
    except Exception:  # noqa: BLE001
        pass

    # Compose the middleware chain. Each cross-cutting concern lives
    # in its own file under app/assistant/react/middlewares/ and is
    # independently testable. The factory pins the order.
    chain = MiddlewareChain(default_chain_factory())

    # Seed the scratchpad with what the initial plan already did so the
    # model can reason about gaps without us re-summarising work.
    if state.initial_plan_actions:
        state.pad.think(
            "Initial plan already executed: "
            + ", ".join(state.initial_plan_actions)
            + ". I will reason about whether more retrieval / verification "
              "is needed, or finalize."
        )

    for i in range(config.max_iterations):
        if time.monotonic() > state.deadline:
            state.pad.think("Deadline reached — stopping the loop and finalizing.")
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
        if state.should_cancel is not None:
            try:
                cancel_signal = bool(await state.should_cancel())
            except Exception:  # noqa: BLE001
                cancel_signal = False
        elif state.ctx is not None and getattr(state.ctx, "should_cancel", None) is not None:
            try:
                cancel_signal = bool(await state.ctx.should_cancel())
            except Exception:  # noqa: BLE001
                cancel_signal = False
        if cancel_signal:
            state.pad.think("Cancellation requested — stopping the loop and finalizing.")
            break

        state.pad.next_iteration()
        state.iteration_count = i + 1
        state.is_last_iteration = (i == config.max_iterations - 1)

        # Pre-iteration middleware hook — middlewares can refresh
        # caches, prepare state, run any cheap pre-decision work.
        await chain.before_iteration(state)

        # ── Decide next action ───────────────────────────────────────
        try:
            decision = await _decide_next_action(
                query=state.query,
                pad=state.pad,
                prior_results=state.prior_results,
                new_results=state.new_results,
                memory_view=state.memory_view,
                research_brief_text=state.research_brief_text,
                active_context=state.active_context,
                ledger=state.ledger,
                contradictions=state.contradictions,
                retrieval_obs=state.retrieval_obs,
                banned_tools=state.banned_tools,
                plan=state.plan,
                config=config,
                is_last_iteration=state.is_last_iteration,
                subagent_depth=state.subagent_depth,
                subagent_role=state.subagent_role,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("react_loop: decision step failed (iter=%d): %s", i, exc)
            state.pad.think(f"Decision step failed: {exc}. Finalizing.")
            break

        if not decision:
            state.pad.think("LLM returned no decision — finalizing.")
            break

        thought = (decision.get("thought") or "").strip()
        action = (decision.get("action") or "finalize").strip()
        # Defensive coercion: a model that emits ``params`` as a string,
        # list, or null lands here. We want every downstream consumer
        # (middleware, validator, dispatch) to see a dict — coercing
        # at the loop boundary keeps every middleware free of
        # ``isinstance(params, dict)`` checks.
        raw_params = decision.get("params")
        if isinstance(raw_params, dict):
            params = raw_params
        else:
            params = {}
            if raw_params is not None:
                state.pad.think(
                    f"Decision step returned non-dict params "
                    f"({type(raw_params).__name__}); coerced to empty dict."
                )
        rationale = (decision.get("rationale") or "").strip()

        if thought:
            state.pad.think(thought)
            state.publish_event("react_thought", {
                "iteration": state.pad.iteration, "text": thought[:400],
            })

        # ── Stop condition: walk the finalize gate chain ────────────
        # The chain returns the first non-allow gate. Critique gate
        # fires first (forces a self-critique on too-early finalize);
        # contradiction gate fires after (forces a counter-search on
        # a high-confidence open signal).
        if action.lower() in {"finalize", "finish", "done", "stop", ""}:
            gate = await chain.gate_finalize(state)
            if isinstance(gate, FinalizeAllow):
                state.completed_normally = True
                state.pad.think("Loop finalized — handing off to synthesis.")
                break
            if isinstance(gate, FinalizeForceCritique):
                state.pad.think(
                    "Forcing a self-critique before finalize — "
                    f"reason: {gate.reason}"
                )
                await _run_self_critique(
                    query=state.query, pad=state.pad,
                    prior_results=state.prior_results, new_results=state.new_results,
                    memory_view=state.memory_view,
                )
                continue
            if isinstance(gate, FinalizeForceAction):
                # Inject the forced action and fall through to dispatch.
                action = gate.action
                params = dict(gate.params)
                rationale = gate.rationale or gate.reason
                state.pad.think(
                    f"Finalize gated by middleware: forcing {action} — "
                    f"reason: {gate.reason}"
                )

        # ── Pseudo-action: self-critique ─────────────────────────────
        if action.lower() == "critique":
            await _run_self_critique(
                query=state.query, pad=state.pad,
                prior_results=state.prior_results, new_results=state.new_results,
                memory_view=state.memory_view,
            )
            state.publish_event("react_critique", {"iteration": state.pad.iteration})
            continue

        # ── Pseudo-action: write_todos (investigation tracker) ──────
        # The model maintains a durable mid-loop task list across
        # iterations. Each op is one of ``add`` / ``update`` /
        # ``complete`` / ``cancel`` / ``clear`` — the plan applies
        # them in batch, surfaces any malformed ops as scratchpad
        # notes, and renders the updated plan into the next
        # decision prompt so the model sees its own intentions.
        if action.lower() in {"write_todos", "todos", "plan"}:
            ops = params.get("todos") or decision.get("todos") or []
            if not isinstance(ops, list):
                state.pad.observe(
                    tool="write_todos",
                    summary="write_todos payload missing or non-list — ignored.",
                    output_ref="",
                    error="malformed_todos_payload",
                )
                continue
            applied_notes = state.plan.apply_operations(ops, iteration=state.iteration_count)
            applied_text = "; ".join(applied_notes)[:600] if applied_notes else "(no ops)"
            state.pad.observe(
                tool="write_todos",
                summary=f"Plan updated: {applied_text}",
                output_ref="",
                error=None,
            )
            state.publish_event("react_plan_updated", {
                "iteration": state.pad.iteration,
                "ops_applied": len(applied_notes),
            })
            continue

        # ── Pseudo-action: fanout (parallel multi-branch) ─────────────
        # The model picks ``action="fanout"`` and emits a ``branches``
        # array, each branch a (tool, params) pair. We dispatch all
        # branches concurrently — each through its own ctx_factory
        # session so writes don't collide. Used for genuinely multi-
        # headed questions where each branch is a SINGLE tool call.
        if action.lower() == "fanout":
            branches = decision.get("branches") or []
            if not isinstance(branches, list) or not branches:
                state.pad.think("Fanout requested with no branches — treating as finalize.")
                state.completed_normally = True
                break
            branches = branches[:_MAX_FANOUT_BRANCHES]
            branch_failures = await _run_fanout(
                branches=branches,
                pad=state.pad, query=state.query,
                ledger=state.ledger, contradictions=state.contradictions,
                retrieval_obs=state.retrieval_obs,
                new_results=state.new_results,
                tool_fail_counts=state.tool_fail_counts,
                banned_tools=state.banned_tools,
                ctx_factory=state.ctx_factory, ctx=state.ctx,
                publish=state.publish,
            )
            state.tool_failures += branch_failures
            for _t, _c in list(state.tool_fail_counts.items()):
                if _c >= 2:
                    state.banned_tools.add(_t)
            continue

        # ── Pseudo-action: subagent (context-quarantined delegation) ──
        # The model picks ``action="subagent"`` with ``subagent_name``
        # + ``task``. We spawn a nested ReAct loop with a focused
        # query, restricted tool catalog, fresh scratchpad, and tight
        # iteration cap. The parent's context only sees the subagent's
        # FINAL summary, not its dozens of intermediate observations —
        # this is the real context-quarantine win the fanout action
        # couldn't deliver.
        #
        # Recursion guard: nested subagent dispatch is refused. Allowing
        # a subagent to spawn another subagent would defeat context
        # quarantine (the grandchild's quarantine is irrelevant to the
        # grandparent who already only sees a summary), risk runaway
        # iteration budget, and produce traces that are nearly impossible
        # to debug. The decision prompt also hides the subagent catalog
        # at depth > 0 so the model isn't even tempted.
        if action.lower() == "subagent" and state.subagent_depth > 0:
            state.pad.observe(
                tool="subagent",
                summary=(
                    "Nested subagent dispatch refused — a subagent cannot itself "
                    "delegate. Focus on the assigned task and finalize."
                ),
                output_ref="",
                error="subagent_recursion_blocked",
            )
            continue

        if action.lower() == "subagent":
            subagent_name = str(
                params.get("subagent_name")
                or decision.get("subagent_name")
                or ""
            ).strip()
            task = str(
                params.get("task")
                or decision.get("task")
                or state.query
            ).strip()
            if not subagent_name or subagent_name not in SUBAGENT_REGISTRY:
                state.pad.observe(
                    tool="subagent",
                    summary=(
                        f"Unknown subagent '{subagent_name}'. Available: "
                        + ", ".join(sorted(SUBAGENT_REGISTRY.keys()))
                    ),
                    output_ref="",
                    error="unknown_subagent",
                )
                continue
            state.pad.act(
                tool=f"subagent:{subagent_name}",
                params={"task": task[:500]},
                rationale=rationale,
            )
            state.publish_event("react_subagent_start", {
                "iteration": state.pad.iteration,
                "subagent": subagent_name,
                "task": task[:240],
            })
            try:
                sub_result = await run_subagent(
                    parent_state=state,
                    subagent_name=subagent_name,
                    task=task,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("subagent '%s' raised: %s", subagent_name, exc)
                state.pad.observe(
                    tool=f"subagent:{subagent_name}",
                    summary=f"Subagent error: {exc}",
                    output_ref="",
                    error=str(exc)[:300],
                )
                continue
            # Land the subagent's distilled output as a single result
            # the parent's chain (paper_ledger, observability,
            # contradiction) treats like any other tool result.
            sub_tool_result = sub_result.to_tool_result()
            slot_key = f"subagent:{subagent_name}"
            state.new_results[slot_key] = sub_tool_result
            await chain.after_tool(state, slot_key, params, sub_tool_result)
            state.pad.observe(
                tool=f"subagent:{subagent_name}",
                summary=sub_result.summary,
                output_ref=slot_key,
                error=None,
            )
            state.publish_event("react_subagent_done", {
                "iteration": state.pad.iteration,
                "subagent": subagent_name,
                "iterations": sub_result.iterations,
                "summary": sub_result.summary[:240],
            })
            continue

        # ── Real tool dispatch: walk before_tool chain ───────────────
        pre = await chain.before_tool(state, action, params)
        if isinstance(pre, AbortDispatch):
            state.pad.observe(
                tool=action,
                summary=pre.observation_summary,
                output_ref="",
                error=pre.error,
            )
            continue
        if isinstance(pre, DispatchOverride):
            if pre.action is not None:
                action = pre.action
            if pre.params is not None:
                params = pre.params

        tool = get_tool(action)
        if tool is None:
            state.pad.observe(
                tool=action,
                summary="Unknown tool — skipped.",
                output_ref="",
                error="tool_not_found",
            )
            continue

        state.pad.act(tool=action, params=params, rationale=rationale)
        state.publish_event("react_action", {
            "iteration": state.pad.iteration, "tool": action,
            "rationale": rationale[:200],
        })

        # ── Validate + dispatch ──────────────────────────────────────
        # Preflight already ran via ParamPreflightMiddleware. The
        # validation here is the final pydantic check; on failure we
        # do one more auto-repair from scratch (the middleware's
        # repair only fired if the model emitted partial-but-broken
        # params; this branch handles the case where the model's
        # params validated but the tool's runtime rejected them).
        try:
            input_schema = tool.input_schema
            schema_dict: dict = {}
            try:
                schema_dict = input_schema.model_json_schema()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                schema_dict = {}
            try:
                validated = input_schema(**params) if isinstance(params, dict) else params  # type: ignore[arg-type]
            except Exception as ve:
                if not schema_dict:
                    raise
                fresh_params, fresh_notes = _preflight_and_repair_params(
                    action, {}, schema_dict, query=state.query, ledger=state.ledger,
                )
                try:
                    validated = input_schema(**fresh_params)
                    params = fresh_params
                    if fresh_notes:
                        state.pad.think(
                            f"Validation-fallback repair for {action}: "
                            + "; ".join(fresh_notes)[:400]
                        )
                except Exception as ve2:  # noqa: BLE001
                    state.record_tool_failure(action, ban_cap=2)
                    required = list(schema_dict.get("required") or [])
                    state.pad.observe(
                        tool=action,
                        summary=(
                            f"Invalid params even after auto-repair. "
                            f"Required={required}. "
                            f"Tried={json.dumps(fresh_params, default=str)[:200]}. "
                            f"Error: {str(ve2)[:200]}. "
                            "Pick a different tool or first run a retrieval tool "
                            "(deep_search / arxiv_import / literature_survey) "
                            "to populate the paper ledger."
                        ),
                        output_ref="",
                        error="invalid_params",
                    )
                    continue

            if state.ctx_factory is not None:
                async with state.ctx_factory() as _action_ctx:
                    result: ToolResult = await tool.run(_action_ctx, validated)
            elif state.ctx is not None:
                result = await tool.run(state.ctx, validated)
            else:
                raise RuntimeError("react_loop: neither ctx nor ctx_factory provided")

            state.new_results[action] = result
            state.pad.observe(
                tool=action,
                summary=(result.summary or "(no summary)"),
                output_ref=action,
                error=None,
            )
            # Walk after_tool middleware chain — ledger, observability,
            # contradiction, diminishing-returns. The DiminishingReturns
            # middleware sets ``state._diminishing_returns_hit`` and
            # marks ``completed_normally`` when it fires; we check
            # that flag below to break out of the iteration loop.
            await chain.after_tool(state, action, params, result)
            state.publish_event("react_observation", {
                "iteration": state.pad.iteration,
                "tool": action,
                "summary": (result.summary or "")[:240],
            })
            if getattr(state, "_diminishing_returns_hit", False):
                break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("react_loop: tool '%s' raised: %s", action, exc)
            await chain.on_tool_error(state, action, params, exc)
            state.pad.observe(
                tool=action,
                summary=(
                    f"Tool error: {exc}. "
                    + ("This tool is now banned for the remainder of the turn. "
                       if action in state.banned_tools else "")
                    + "Try a different tool or broaden the query."
                ),
                output_ref="",
                error=str(exc)[:300],
            )

    state.pad.finish()
    return ReactOutcome(
        scratchpad=state.pad,
        new_results=state.new_results,
        completed_normally=state.completed_normally,
        iterations=state.iteration_count,
        tool_failures=state.tool_failures,
        successful_retrievals=state.successful_retrievals,
        paper_ledger_size=len(state.ledger.by_id),
        retrieval_metrics=state.retrieval_obs.to_agent_notes(),
        contradiction_signals=[
            {
                "kind": s.kind, "span": s.span[:300],
                "confidence": s.confidence, "addressed": s.addressed,
                "sources": list(s.sources)[:4],
            }
            for s in state.contradictions.signals[:8]
        ],
        investigation_plan=state.plan.summarize_for_synth(),
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
    ledger: PaperLedger | None = None,
    contradictions: ContradictionLedger | None = None,
    retrieval_obs: RetrievalObservability | None = None,
    banned_tools: set[str] | None = None,
    plan: Any = None,                # InvestigationPlan | None
    config: ReactConfig,
    is_last_iteration: bool,
    subagent_depth: int = 0,
    subagent_role: str | None = None,
) -> dict[str, Any] | None:
    """One cheap-model call. Returns the parsed decision dict or None."""
    from app.adapters.llm import get_llm_adapter

    # Tool catalog — same view the planner uses, minus tools the loop
    # can't / shouldn't call (see ``_DISALLOWED_FROM_LOOP``) and any tool
    # already banned this turn after repeated failures.
    catalog = describe_for_planner(
        namespace_key=config.namespace_key,
        disabled_features=config.disabled_features,
    )
    banned = set(banned_tools or set())
    catalog = [
        t for t in catalog
        if t.get("name") not in _DISALLOWED_FROM_LOOP
        and t.get("name") not in banned
    ]

    # Compact "what we already have" view.
    prior_summary = _summarise_results(prior_results) + _summarise_results(new_results)

    # Role header — replaces the generic prompt when we're running
    # *as* a named subagent. Without this swap the model would see
    # the parent's "you are the reasoning engine of RA" AND the
    # subagent's "you are a focused literature researcher" role
    # buried in the query, and pick whichever shows up later. Putting
    # the role in the system message gives the model one coherent
    # instruction.
    if subagent_role:
        role_header = (
            f"You are a specialised subagent. Role:\n{subagent_role.strip()}\n\n"
            "You were spawned by the parent research assistant for a focused "
            "sub-task. Return a tight summary; your intermediate steps stay "
            "out of the parent's context.\n\n"
        )
    else:
        role_header = (
            "You are the reasoning engine of a research assistant in the "
            "MIDDLE of a turn.\n\n"
        )

    # The subagent action is hidden when this loop is itself a
    # subagent (depth > 0). Allowing nested subagent dispatch would
    # defeat context quarantine and risk runaway iteration budget.
    # Subagents focus on their assigned task; they don't delegate
    # further.
    if subagent_depth <= 0:
        subagent_block = (
            "  (d) call action 'subagent' with 'subagent_name' + 'task' params to\n"
            "      delegate a focused multi-step investigation to a specialised agent\n"
            "      whose intermediate steps stay OUT of your context. The parent only\n"
            "      sees the subagent's final summary. Use this for sub-questions that\n"
            "      would otherwise burn many iterations with retrieval + analysis\n"
            "      chatter you don't need to read in detail. Available subagents:\n"
            f"{_render_subagent_catalog()}\n"
            "      Format: ``{\"action\":\"subagent\",\"params\":{\"subagent_name\":\"researcher\",\"task\":\"...\"}}``,\n"
            "  (e) call action 'finalize' to hand off to synthesis.\n\n"
        )
        delegation_advice = (
            "Prefer 'fanout' when sub-questions are independent and each is one tool "
            "call. Prefer 'subagent' when a sub-question is itself a multi-step "
            "investigation (you want isolation, not just parallelism). Prefer serial "
            "calls when a later step depends on an earlier step's output.\n\n"
        )
    else:
        subagent_block = (
            "  (d) call action 'finalize' to hand off your summary to the parent.\n\n"
        )
        delegation_advice = (
            "You are a subagent — do NOT delegate further. Focus on the task and "
            "finalize as soon as you have a tight summary.\n\n"
        )

    sys_msg = (
        role_header
        + "An initial plan has already executed. Your job each iteration is to either:\n"
        "  (a) call another tool to gather more evidence / verify a claim / fill a gap,\n"
        "  (b) call action 'critique' to self-judge whether the evidence is sufficient\n"
        "      and well-grounded (records a verdict + issues on the scratchpad),\n"
        "  (c) call action 'fanout' with a 'branches' array of 2-4 (tool, params)\n"
        "      pairs that should run in parallel — use when the question is\n"
        "      genuinely multi-headed (e.g. compare four directions, investigate\n"
        "      three sub-claims). Each branch counts as one tool dispatch but the\n"
        "      whole fanout is ONE iteration, so this saves round-trip latency on\n"
        "      complex queries,\n"
        "  (b2) call action 'write_todos' with a 'todos' array of structured ops\n"
        "       (kind ∈ add|update|complete|cancel|clear) to update your durable\n"
        "       investigation plan — entries persist across iterations so you can\n"
        "       declare a multi-part plan early and check items off as you resolve\n"
        "       them. Use this when the user's question has ≥3 distinct\n"
        "       investigations; it's the antidote to mid-loop drift and to\n"
        "       finalizing before every sub-question is answered.\n"
        "       Format: ``{\"action\":\"write_todos\",\"todos\":[{\"kind\":\"add\","
        "\"text\":\"compare RAG vs long-context on real workflows\"},"
        "{\"kind\":\"complete\",\"id\":\"t1\",\"evidence\":[\"paper-id-x\"]}]}``,\n"
        + subagent_block
        + delegation_advice
        +
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
        "ADVERSARIAL RIGOUR (the system trusts honest doubt over confident polish):\n"
        "  - When the user's question has a contested answer, actively look "
        "for COUNTER-EVIDENCE before finalizing. A loop that only retrieves "
        "supporting evidence produces a one-sided answer.\n"
        "  - Watch the observations for explicit disagreement signals "
        "(e.g. 'contradicts', 'fails to replicate', 'weaker than baseline', "
        "'overestimates', 'criticised', 'inconsistent with'). When you see "
        "them, call another retrieval / citation_finder targeted at the "
        "OPPOSING claim.\n"
        "  - When evidence is thin or conflicting, prefer an honest "
        "abstention over a confident-sounding synthesis. Use 'critique' to "
        "score groundedness/completeness; if either is below 0.5, gather "
        "more evidence rather than finalizing.\n\n"
        "PARAMS RULES (critical — broken params burn an iteration):\n"
        "  - Every tool's required + optional params are listed in the catalog. "
        "Send a real value for every 'required' field.\n"
        "  - NEVER emit placeholders like '__to_fill_*__', '<TODO>', '<fill>', "
        "'null', 'tbd'. If you don't have a value, either pick a different "
        "tool or call a retrieval tool first to obtain it.\n"
        "  - For 'paper_ids' / 'paper_id' params, copy concrete IDs verbatim "
        "from the PAPER LEDGER block. Never invent IDs and never use the title "
        "as an ID.\n"
        "  - For 'query' / 'question' / 'claim' params, write a focused text "
        "expression of what you actually want — not a copy of the whole user "
        "turn unless that IS what you want to search.\n\n"
        "GENIE FLOW (when the user asks you to 'synthesize an idea', "
        "'combine these papers', 'propose a novel architecture' or similar):\n"
        "  - Step 1: call 'genie_synthesize' with paper_ids from the LEDGER. "
        "This creates a NEW idea capsule grounded in those papers.\n"
        "  - Step 2: (optional) call 'genie_deep_dive' on the capsule_id "
        "returned by step 1 to expand it.\n"
        "  - DO NOT call 'genie_read' for a synthesis request — that only "
        "lists STALE capsules from prior turns and produces off-topic answers.\n\n"
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

    catalog_text = _render_tool_catalog(catalog, limit=30)
    ledger_text = (
        ledger.render_for_prompt(limit=12)
        if ledger is not None
        else "(ledger unavailable)"
    )
    contradictions_text = (
        contradictions.render_for_prompt(limit=4)
        if contradictions is not None
        else "(detector unavailable)"
    )
    retrieval_text = (
        retrieval_obs.render_for_prompt(limit=6)
        if retrieval_obs is not None
        else "(observability unavailable)"
    )
    try:
        from app.assistant.query_strategy import classify_query
        strategy_text = classify_query(query).render_for_prompt()
    except Exception:
        strategy_text = "(strategy router unavailable)"

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

    banned_note = (
        f"BANNED THIS TURN (failed repeatedly — do not pick): {sorted(banned)}\n\n"
        if banned else ""
    )
    plan_text = (
        plan.render_for_prompt(limit=12)
        if plan is not None
        else "(plan unavailable)"
    )
    user_msg = (
        f"USER QUERY:\n{query[:1500]}\n\n"
        f"RESEARCH BRIEF:\n{(research_brief_text or '(none)')[:1500]}\n\n"
        f"{active_ctx_block}\n\n"
        f"{banned_note}"
        f"TOOL CATALOG (call any of these — read the params carefully):\n{catalog_text}\n\n"
        f"PAPER LEDGER (concrete IDs you may pass to compare_papers / paper_qa / genie_synthesize):\n{ledger_text}\n\n"
        f"CONTRADICTIONS DETECTED (each row shows the contested claim + how confident the detector is + whether the loop has already investigated it):\n{contradictions_text}\n\n"
        f"RETRIEVAL QUALITY (coverage/dispersion/rerank-disagreement per retrieval — thin or rerank-heavy calls are a flag to broaden the next search):\n{retrieval_text}\n\n"
        f"ADAPTIVE STRATEGY HINT (advisory — query-shape router's recommendation):\n  {strategy_text}\n\n"
        f"INVESTIGATION PLAN (your durable mid-loop task list — use 'write_todos' to add/update/complete entries; entries persist across iterations):\n{plan_text}\n\n"
        f"WHAT THE INITIAL PLAN PRODUCED:\n{prior_summary[:2000]}\n\n"
        f"SCRATCHPAD SO FAR:\n{pad.render_for_prompt()}\n\n"
        "Now decide your next ACTION. Remember: every 'required' param needs a "
        "concrete value drawn from the user query, the ledger, or the brief — "
        "never a placeholder. If you can't fill a required param, switch tools "
        "or finalize. Address open contradictions before finalizing — but only "
        "when the contradiction matters for the user's question; soft / "
        "tangential disagreements can be flagged in the final answer instead. "
        "When the user's question is multi-part or you anticipate ≥3 distinct "
        "investigations, use 'write_todos' early to declare your plan so you "
        "stay on track across iterations; mark each todo complete (with "
        "evidence pointers) as you resolve it."
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


def _render_subagent_catalog() -> str:
    """Compact subagent catalog for the decision prompt's action list."""
    try:
        from app.assistant.react.subagents import describe_subagents_for_prompt
        return describe_subagents_for_prompt()
    except Exception:  # noqa: BLE001 — never break the prompt builder on this
        return "        (subagent registry unavailable)"


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


async def _run_fanout(
    *,
    branches: list[dict],
    pad: Scratchpad,
    query: str,
    ledger: PaperLedger,
    contradictions: ContradictionLedger,
    retrieval_obs: RetrievalObservability,
    new_results: dict[str, ToolResult],
    tool_fail_counts: dict[str, int],
    banned_tools: set[str],
    ctx_factory: Any,
    ctx: ToolContext | None,
    publish: Any,
) -> int:
    """Dispatch up to ``_MAX_FANOUT_BRANCHES`` tool calls concurrently.

    Each branch:
      * runs preflight + auto-repair on its params (same hygiene as
        the serial path),
      * dispatches through its own ctx_factory session so writes don't
        collide between concurrent tool calls,
      * lands its result in ``new_results`` keyed by the tool name
        (later branches with the same tool get a ``{tool}#{i}`` key),
      * feeds the ledger + contradiction + retrieval-observability
        scanners exactly like a serial dispatch.

    Branch failures are isolated — one branch erroring does not abort
    the others. Each failure increments per-tool fail counts so the
    same-tool-ban policy still applies on the next iteration.
    """
    failures = 0
    pad.think(
        f"Fanout: dispatching {len(branches)} parallel branch(es) — "
        + ", ".join(str(b.get("tool", "?")) for b in branches)
    )
    if publish:
        try:
            publish("react_fanout", {
                "iteration": pad.iteration,
                "branches": [str(b.get("tool", "?")) for b in branches],
            })
        except Exception:
            pass

    async def _run_one(branch: dict, slot: int) -> None:
        nonlocal failures
        tool_name = str(branch.get("tool") or "").strip()
        branch_params = dict(branch.get("params") or {})
        branch_rationale = str(branch.get("rationale") or "")
        if not tool_name or tool_name in _DISALLOWED_FROM_LOOP or tool_name in banned_tools:
            pad.observe(
                tool=tool_name or "?",
                summary=f"Fanout branch skipped — tool '{tool_name}' is unavailable.",
                output_ref="",
                error="branch_unavailable",
            )
            return
        tool_obj = get_tool(tool_name)
        if tool_obj is None:
            pad.observe(tool=tool_name, summary="Unknown tool — skipped.", output_ref="", error="tool_not_found")
            return
        try:
            input_schema = tool_obj.input_schema
            try:
                schema_dict = input_schema.model_json_schema()
            except Exception:
                schema_dict = {}
            if isinstance(branch_params, dict) and schema_dict:
                branch_params, _notes = _preflight_and_repair_params(
                    tool_name, branch_params, schema_dict,
                    query=query, ledger=ledger,
                )
            try:
                validated = input_schema(**branch_params)
            except Exception:
                # One-shot repair with fully-derived params.
                fresh, _ = _preflight_and_repair_params(
                    tool_name, {}, schema_dict, query=query, ledger=ledger,
                )
                try:
                    validated = input_schema(**fresh)
                    branch_params = fresh
                except Exception as ve2:
                    tool_fail_counts[tool_name] = tool_fail_counts.get(tool_name, 0) + 1
                    failures += 1
                    pad.observe(
                        tool=tool_name,
                        summary=f"Fanout branch invalid params (post-repair): {ve2}",
                        output_ref="",
                        error="invalid_params",
                    )
                    return
            pad.act(tool=tool_name, params=branch_params, rationale=branch_rationale)
            if ctx_factory is not None:
                async with ctx_factory() as _branch_ctx:
                    result = await tool_obj.run(_branch_ctx, validated)
            elif ctx is not None:
                result = await tool_obj.run(ctx, validated)
            else:
                raise RuntimeError("react_loop fanout: no ctx_factory / ctx")
            # Keyed slot — duplicate tool names within one fanout
            # iteration get suffixed so we don't clobber each other.
            slot_key = tool_name if tool_name not in new_results else f"{tool_name}#{slot}"
            new_results[slot_key] = result
            try:
                ledger.add_from_result(result)
            except Exception:
                pass
            try:
                retrieval_obs.record(tool_name, branch_params, result)
            except Exception:
                pass
            try:
                for sig in detect_contradictions_in_results(
                    {tool_name: result}, iteration=pad.iteration,
                ):
                    contradictions.add(sig)
            except Exception:
                pass
            pad.observe(
                tool=tool_name,
                summary=(result.summary or "(no summary)"),
                output_ref=slot_key,
                error=None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            tool_fail_counts[tool_name] = tool_fail_counts.get(tool_name, 0) + 1
            failures += 1
            pad.observe(
                tool=tool_name,
                summary=f"Fanout branch error: {exc}",
                output_ref="",
                error=str(exc)[:300],
            )

    await asyncio.gather(
        *(_run_one(b, i) for i, b in enumerate(branches)),
        return_exceptions=False,
    )
    return failures


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
