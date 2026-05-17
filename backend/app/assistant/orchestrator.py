"""Assistant turn orchestrator.

Executes a Plan over the tool registry, writing one AssistantStep row per
tool call so each step is independently inspectable, cancellable, and
resumable. Composes step outputs through the synthesizer to produce the
final assistant message.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.assistant.events import AssistantEvent, get_event_bus
from app.assistant.planner import HeuristicPlanner, Plan, PlannedStep
from app.assistant.planner_llm import LLMPlanner
from app.assistant.step_cache import get_step_cache
from app.assistant.synthesizer import build_message_blocks, synthesize_answer
from app.assistant.tools.base import ToolContext, ToolResult
from app.assistant.tools.registry import get_tool
from app.db.session import async_session_factory
from app.models.assistant import (
    AssistantMessage,
    AssistantMessageRole,
    AssistantStepStatus,
    AssistantTask,
    AssistantTaskStatus,
)
from app.repositories.assistant import AssistantRepository
from app.repositories.user import UserRepository
from app.services.job_store import get_job_store

log = logging.getLogger(__name__)


# Stage-level progress markers used to update AssistantTask.progress so the
# notification panel and reasoning tree show coherent percentages even though
# individual tools also emit fine-grained per-step progress.
_STAGE_PROGRESS = {
    "planning": 10,
    "executing": 40,
    "synthesizing": 85,
    "completed": 100,
}

# Grounding-paper relevance gate used by `_papers_from_results`. The
# deep_search tool rank-normalises `search_score` to a 1.0 → 0.0 ladder
# (top-ranked == 1.0, last-ranked → 0.0). Anything below 0.4 has typically
# slipped through the LLM rerank tail and adds noise rather than grounding.
# Capping the surviving list at 8 keeps the UI's *Grounded papers* block
# scannable without dropping a useful long tail.
_MIN_GROUNDING_SCORE = 0.40
_MAX_GROUNDING_PAPERS = 8

# Phrases/patterns that reliably indicate an off-topic request — i.e., not
# research, learning, synthesis, exploration, or platform-adjacent tasks.
# Kept conservative: we only reject when confident, never when in doubt.
_OFF_TOPIC_PATTERNS = (
    r"\b(recipe|cooking|bake|baking|cook\b|dish\b|cuisine|ingredient|restaurant)\b",
    r"\b(horoscope|astrology|zodiac|tarot|psychic|fortune\s*tell)\b",
    r"\b(lottery|gambling|casino|bet\b|betting|wager)\b",
    r"\b(sports\s+(score|result|game|match|team|player|league)|nfl|nba|mlb|nhl|fifa|cricket\s+score)\b",
    r"\b(stock\s+(price|ticker|quote)|share\s+price|crypto\s+price|bitcoin\s+price)\b",
    r"\b(movie\s+(review|plot|trailer)|tv\s+(show|series|episode)|celebrity|gossip)\b",
    r"\b(write\s+(me\s+a\s+(poem|song|story|joke|rap)|lyrics)|tell\s+me\s+a\s+joke)\b",
    r"\bpassword\b.{0,30}\brecover|hack\s+(account|password|into|email)\b",
    r"\b(illegal|commit\s+a\s+crime|how\s+to\s+(steal|cheat|defraud))\b",
)

_OFF_TOPIC_REDIRECT = (
    "I'm focused on research, learning, literature exploration, synthesis, "
    "and scientific discovery — this question looks like it falls outside that scope.\n\n"
    "If there's a research angle I can help with — finding papers, explaining "
    "concepts, exploring the scientific literature, or connecting ideas — just let me know."
)

import re as _re

def _is_off_topic(query: str) -> bool:
    """Return True when the query clearly belongs to a non-research domain."""
    q_lower = (query or "").lower()
    return any(_re.search(pat, q_lower) for pat in _OFF_TOPIC_PATTERNS)


async def _rewrite_query(
    *,
    query: str,
    namespace_key: str,
    history: list[dict],
    memory: dict,
) -> str:
    """Return a strengthened internal query for retrieval and planning.

    Only rewrites when the query is short (<60 chars), ambiguous, or uses
    pronouns / vague references that would hurt retrieval quality. Returns
    the original query unchanged on any failure so the turn is never blocked.
    """
    raw = (query or "").strip()
    # Skip rewriting for long, already-specific queries to save latency.
    if len(raw) >= 150:
        return raw
    # Rewrite when query is short, uses pronouns, or has vague references.
    has_pronoun_ref = any(w in raw.lower() for w in (
        "it", "that", "this", "they", "them", "these", "those",
        "the paper", "the method", "the approach", "the model", "the technique",
        "the algorithm", "the result", "the finding", "the study", "the work",
        "previous", "mentioned", "above", "earlier", "last", "same",
    ))
    is_very_short = len(raw) < 50
    if not (has_pronoun_ref or is_very_short):
        return raw

    try:
        from app.adapters.llm import get_llm_adapter
        llm = get_llm_adapter()
        # Use last 8 messages for reference resolution (covers typical "follow-up" depth)
        recent = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content') or ''}"
            for m in (history or [])[-8:]
        )
        mem_hint = ""
        if memory:
            mem_hint = "Research context: " + " | ".join(f"{k}: {v}" for k, v in list(memory.items())[:5])
        system = (
            "You rewrite short or ambiguous research queries into precise retrieval queries. "
            "Resolve pronouns using conversation history. Add domain context from the namespace. "
            "Output ONLY the rewritten query — no explanation, no quotes, no preamble. "
            "If the original query is already clear and specific, output it unchanged."
        )
        user_msg = (
            f"Namespace: {namespace_key}\n"
            f"{mem_hint}\n"
            f"Recent conversation:\n{recent or '(none)'}\n\n"
            f"Original query: {raw}\n\n"
            "Rewritten query:"
        )
        result = await llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            llm.cheap_model,
            max_tokens=100,
            temperature=0.0,
        )
        rewritten = (result.text or "").strip().strip('"').strip("'")
        if rewritten and len(rewritten) >= len(raw) * 0.5:
            log.debug("query rewrite: %r → %r", raw, rewritten)
            return rewritten
    except Exception as exc:
        log.debug("query rewrite failed (using original): %s", exc)
    return raw


class Orchestrator:
    """Plans, runs, and synthesizes one assistant turn end-to-end.

    Split out from research_assistant.submit_turn so the bookkeeping
    (sessions, messages, tasks, JobStore) stays in the service layer and
    the orchestrator owns only execution + checkpointing.
    """

    def __init__(self, planner=None) -> None:
        # LLMPlanner falls back to HeuristicPlanner internally on any failure.
        self._planner = planner or LLMPlanner(fallback=HeuristicPlanner())
        self._bus = get_event_bus()
        self._cache = get_step_cache()
        # Strong references to fire-and-forget post-turn tasks so Python 3.12+
        # doesn't GC them before the metadata/interest-profile work finishes.
        # Tasks self-discard when they complete.
        self._post_turn_tasks: set[asyncio.Task] = set()

    async def run_turn(self, job_id: str) -> None:
        """Execute one assistant turn keyed by ``job_id``."""
        try:
            await self._mark_task(job_id, AssistantTaskStatus.running, "planning",
                                  _STAGE_PROGRESS["planning"], "Planning workflow", started=True)

            # Load context in its own transaction so subsequent steps can
            # open fresh sessions per-step (cleaner cancel semantics).
            async with async_session_factory() as db:
                ctx_bundle = await self._load_context(db, job_id)
                if ctx_bundle is None:
                    return
                task, session, query, namespace_keys, primary_ns, orientation, expertise = ctx_bundle
                # Use branch-aware history computed in _load_context (includes parent msgs).
                history = getattr(session, "_orchestrator_history", None) or [
                    {"role": m.role.value if hasattr(m.role, "value") else str(m.role),
                     "content": m.content}
                    for m in (session.messages or [])[-8:]
                ]
                medium_memory = getattr(session, "_medium_memory", {})
                ns_memory = getattr(session, "_ns_memory", {})

                # Determine which optional tools are unavailable due to missing keys.
                disabled_tools: set[str] = set()
                wolfram_available = await self._check_wolfram_available(db, task.user_id)
                if not wolfram_available:
                    disabled_tools.add("wolfram_alpha")

            # Early off-topic guard — reject clearly irrelevant queries before
            # spending any LLM/retrieval budget on them.
            if _is_off_topic(query):
                await self._finalize_off_topic(job_id, task, query)
                return

            # Set token-usage attribution contextvars: this background coroutine
            # runs detached from the HTTP request, so the ``current_user_id``
            # contextvar set in get_current_user_id() is gone. Re-establishing
            # it ensures every LLM call inside the turn lands in TokenUsage
            # tagged with the right user + workflow ("assistant").
            from app.core.tracking import current_user_id as _ctx_uid, set_workflow_context

            _ctx_uid.set(task.user_id)
            set_workflow_context("assistant", "plan")

            # Optional query rewriting: strengthen/clarify the raw user query
            # for better retrieval and planning. Only applies when the query is
            # short, ambiguous, or missing domain context. Falls through silently.
            rewritten_query = await _rewrite_query(
                query=query,
                namespace_key=primary_ns,
                history=history,
                memory=medium_memory,
            )

            # Compute complexity for adaptive model routing in planner + synthesizer.
            from app.assistant.planner_llm import _assess_query_complexity
            query_complexity = _assess_query_complexity(query, history)

            # LLM planner with heuristic fallback. ``aplan`` never raises —
            # it falls through to the heuristic on any LLM/parse failure.
            if hasattr(self._planner, "aplan"):
                plan = await self._planner.aplan(
                    query=rewritten_query,
                    namespace_key=primary_ns,
                    namespace_keys=namespace_keys,
                    history=history,
                    orientation=orientation,
                    expertise=expertise,
                    memory={"medium": medium_memory, "long": ns_memory},
                    disabled_tools=disabled_tools,
                )
            else:
                plan = self._planner.plan(
                    query=rewritten_query, namespace_key=primary_ns, namespace_keys=namespace_keys,
                )

            pure_reasoning = len(plan.steps) == 0
            self._publish(job_id, "plan_committed", {
                "rationale": plan.rationale,
                "actions": plan.actions,
                "step_count": len(plan.steps),
                "steps": [{"tool": s.tool, "title": s.title} for s in plan.steps],
                "pure_reasoning": pure_reasoning,
            })
            await self._append_progress(
                job_id, plan.actions,
                "Reasoning…" if pure_reasoning else f"Plan ready: {len(plan.steps)} step(s)",
                _STAGE_PROGRESS["executing"],
            )

            # Execute steps. Each step opens its own transaction so a single
            # tool failure doesn't poison the others, and each writes an
            # AssistantStep row for the reasoning tree.
            results = await self._execute_plan(
                plan=plan,
                job_id=job_id,
                task=task,
                primary_ns=primary_ns,
                namespace_keys=namespace_keys,
                orientation=orientation,
                expertise=expertise,
            )

            # Compose the message + persist outcome.
            await self._finalize_turn(
                job_id=job_id,
                task=task,
                plan=plan,
                results=results,
                query=query,
                orientation=orientation,
                expertise=expertise,
                complexity=query_complexity,
            )
        except asyncio.CancelledError:
            await self._handle_cancelled(job_id)
            raise
        except Exception as exc:
            log.exception("assistant turn failed job=%s", job_id)
            await self._handle_failed(job_id, str(exc))

    # ── Tool availability checks ──────────────────────────────────────────

    async def _check_wolfram_available(self, db, user_id) -> bool:
        """Return True if the Wolfram Alpha tool is usable for this user."""
        from app.core.config import get_settings as _gs
        s = _gs()
        if s.wolfram_alpha_app_id or s.wolfram_mcp_command:
            return True
        # Check user-stored key in DB
        try:
            from app.repositories.user import UserRepository
            ps = await UserRepository(db).get_provider_settings(user_id)
            if ps and ps.encrypted_wolfram_key:
                return True
        except Exception:
            pass
        return False

    # ── Phase: load context ───────────────────────────────────────────────

    # Rolling history: keep this many recent messages verbatim; older messages
    # are compressed into a single rich summary injected as a system message.
    _HISTORY_VERBATIM = 10
    _HISTORY_SUMMARIZE_THRESHOLD = 14  # start summarizing when total > this

    # Guardrail limits — prevent runaway agentic loops
    _MAX_STEPS_PER_TURN = 12          # absolute cap on planned steps
    _MAX_STEP_DURATION_S = 180        # per-step wall-clock timeout (seconds) — raised for heavy tools
    _MAX_CONSECUTIVE_EMPTY = 3        # max consecutive empty waves before aborting remaining steps

    async def _load_context(self, db, job_id: str):
        repo = AssistantRepository(db)
        task = await self._get_task(db, job_id)
        if not task:
            log.warning("assistant orchestrator: no task for job=%s", job_id)
            return None
        session = await repo.get_session(task.user_id, task.session_id)
        if not session:
            log.warning("assistant orchestrator: no session for job=%s", job_id)
            return None
        user_msg = await self._latest_user_message(db, task.session_id)
        query = user_msg.content if user_msg else ""
        namespace_keys = list(session.topic_keys or [session.namespace_key])
        primary_ns = session.namespace_key or (namespace_keys[0] if namespace_keys else "cs.AI")
        user_repo = UserRepository(db)
        user = await user_repo.get_by_id(task.user_id)
        orientation = user.orientation.value if user else (session.orientation or "both")
        expertise = user.expertise_level.value if user else (session.expertise_level or "practitioner")

        # History: current session messages + parent session messages when
        # this is a branch. Parent messages provide the context the branch
        # is steering away from, so the model knows what ground has been
        # covered and what the branch query is departing from.
        # IMPORTANT: exclude in-flight assistant messages (empty content, status=running)
        # so a still-running workflow does not pollute the next turn's context.
        def _is_completed_msg(m: AssistantMessage) -> bool:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            if role == "assistant":
                payload = m.payload or {}
                if payload.get("status") == "running":
                    return False
                if not m.content and not (payload.get("blocks") or payload.get("workflow", {}).get("steps")):
                    return False
            return True

        all_completed = [
            {"role": m.role.value if hasattr(m.role, "value") else str(m.role),
             "content": m.content}
            for m in (session.messages or [])
            if _is_completed_msg(m)
        ]

        if session.parent_session_id:
            try:
                parent = await repo.get_session(task.user_id, session.parent_session_id)
                if parent and parent.messages:
                    parent_msgs = [
                        {"role": m.role.value if hasattr(m.role, "value") else str(m.role),
                         "content": m.content}
                        for m in parent.messages
                        if (m.role.value if hasattr(m.role, "value") else str(m.role)) != "system"
                    ][-6:]
                    all_completed = parent_msgs + all_completed
            except Exception:
                log.debug("branch context load failed for parent_session=%s", session.parent_session_id)

        # Rolling history: if the conversation exceeds the threshold, compress
        # older turns into a cached summary stored in session.state so it's
        # only generated once per "overflow batch" and reused across turns.
        session_state = dict(session.state or {})
        own_msgs = await self._build_rolling_history(
            all_msgs=all_completed,
            session_state=session_state,
            session_id=str(task.session_id),
            namespace_key=primary_ns,
        )

        # Collect memory from session state — injected into planner prompt.
        medium_memory = dict(session_state.get("memory") or {})
        ns_memory = dict(session_state.get("ns_memory") or {})
        if session.parent_session_id:
            try:
                parent = await repo.get_session(task.user_id, session.parent_session_id)
                if parent:
                    pstate = dict(parent.state or {})
                    # Merge parent medium memory (child takes precedence on conflicts).
                    medium_memory = {**dict(pstate.get("memory") or {}), **medium_memory}
                    if not ns_memory:
                        ns_memory = dict(pstate.get("ns_memory") or {})
            except Exception:
                pass

        session._orchestrator_history = own_msgs  # type: ignore[attr-defined]
        session._medium_memory = medium_memory  # type: ignore[attr-defined]
        session._ns_memory = ns_memory  # type: ignore[attr-defined]
        return task, session, query, namespace_keys, primary_ns, orientation, expertise

    async def _build_rolling_history(
        self,
        *,
        all_msgs: list[dict],
        session_state: dict,
        session_id: str,
        namespace_key: str,
    ) -> list[dict]:
        """Build the history list using a rolling window with lazy summarization.

        When the conversation is short (≤ threshold), returns all messages.
        When longer, keeps the last ``_HISTORY_VERBATIM`` messages verbatim and
        prepends a rich system summary of the older turns. The summary is stored
        in session.state["history_summary"] so it is generated ONCE and reused
        on every subsequent turn — no redundant LLM calls.

        The cached summary is keyed by the message index of the last message
        included in it, so it is only regenerated when new messages fall out
        of the verbatim window (i.e., when the conversation grows past the
        next summarization checkpoint).
        """
        total = len(all_msgs)
        if total <= self._HISTORY_SUMMARIZE_THRESHOLD:
            # Short session — return all messages, no summary needed.
            return all_msgs

        # Split into "old" (to be summarized) and "recent" (verbatim).
        cutoff = total - self._HISTORY_VERBATIM
        old_msgs = all_msgs[:cutoff]
        recent_msgs = all_msgs[cutoff:]

        # Check cache: summary is still valid if the cutoff index hasn't moved.
        cached = session_state.get("history_summary") or {}
        cached_cutoff = cached.get("cutoff_index", -1)
        summary_text = cached.get("text", "") if cached_cutoff == cutoff else ""

        if not summary_text:
            summary_text = await self._summarize_turns(old_msgs, namespace_key)
            # Persist the new summary back to session state so future turns reuse it.
            # Uses its own transaction so a flush failure never blocks the turn.
            try:
                async with async_session_factory() as _patch_db:
                    _patch_repo = AssistantRepository(_patch_db)
                    await _patch_repo.patch_session_state(
                        session_id,
                        {"history_summary": {"cutoff_index": cutoff, "text": summary_text}},
                    )
                    await _patch_db.commit()
                # Also update local view so memory merging below sees it.
                session_state["history_summary"] = {"cutoff_index": cutoff, "text": summary_text}
            except Exception:
                log.debug("failed to persist history summary for session=%s", session_id)

        summary_msg = {
            "role": "system",
            "content": (
                "[Conversation summary — earlier turns]\n"
                f"{summary_text}\n\n"
                "The messages below are the most recent exchanges in full."
            ),
        }
        return [summary_msg] + recent_msgs

    @staticmethod
    async def _summarize_turns(messages: list[dict], namespace_key: str) -> str:
        """Summarize a list of conversation turns into a rich, lossless digest.

        Designed for zero information loss: preserves paper titles/IDs, method
        names, hypotheses, conclusions, and any entity the user might reference
        later with pronouns or vague pointers ("that paper", "the approach").
        """
        try:
            from app.adapters.llm import get_llm_adapter
            llm = get_llm_adapter()
            conv_lines = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content") or ""
                conv_lines.append(f"{role.upper()}: {content}")
            conv_text = "\n\n".join(conv_lines)

            system = (
                "You are summarizing prior turns of a research conversation for a rolling context window. "
                "The summary MUST be rich and complete — it replaces the full text for future turns, so "
                "nothing important can be omitted. Extract and preserve:\n"
                "1. Research topics, questions, and hypotheses discussed\n"
                "2. Paper titles, authors, arXiv IDs, DOIs, or any bibliographic references mentioned\n"
                "3. Methods, algorithms, datasets, benchmarks, or tools named\n"
                "4. Conclusions reached, findings cited, gaps identified\n"
                "5. User preferences, constraints, or directions stated ('focus on X', 'ignore Y')\n"
                "6. Any specific entities (people, organizations, experiments) named\n"
                "7. Outstanding questions or tasks still pending\n\n"
                "Write in plain, dense prose. No headers, no bullets. Max 600 words. "
                "Prioritize named entities and specific references over general topic labels."
            )
            result = await llm.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Namespace: {namespace_key}\n\nConversation:\n{conv_text}"},
                ],
                llm.cheap_model,
                max_tokens=900,
                temperature=0.0,
            )
            return (result.text or "").strip()
        except Exception as exc:
            log.debug("history summarization failed: %s", exc)
            # Fallback: concatenate full turns rather than losing context entirely.
            lines = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content") or ""
                lines.append(f"{role}: {content}")
            return "\n".join(lines)

    # ── Phase: execute plan ───────────────────────────────────────────────

    async def _execute_plan(
        self,
        *,
        plan: Plan,
        job_id: str,
        task: AssistantTask,
        primary_ns: str,
        namespace_keys: list[str],
        orientation: str,
        expertise: str,
    ) -> dict[str, ToolResult]:
        """Run each PlannedStep, write per-step rows, support replan-after-step.

        Resumability: when ``run_turn`` is invoked a second time for the same
        job_id (orphan reconciliation after a process restart), we look up
        the prior AssistantStep rows and skip any that already completed —
        their ``output`` is hydrated back into ``results`` so downstream
        steps that depend on them keep working without re-execution.

        Replanning hook: tools can hint at follow-ups by returning artifacts;
        the only built-in dynamic behaviour today is feeding deep_search's
        paper ids into a queued genie_synthesize step.
        """
        completed = await self._already_completed_steps(job_id)
        results: dict[str, ToolResult] = {}

        # Hydrate prior tool outputs so step injection (e.g. genie reading
        # deep_search papers) works even when deep_search ran in a prior
        # process and only its row remains.
        for tool_name, prior_output in completed.items():
            results[tool_name] = ToolResult(output=prior_output, summary="resumed from checkpoint")
            self._publish(job_id, "step_completed", {
                "tool": tool_name, "summary": "resumed from prior run", "cache_hit": True,
            })

        # ── Guardrail: cap step count ─────────────────────────────────────
        # Clip the plan to _MAX_STEPS_PER_TURN so a runaway planner (e.g. LLM
        # hallucinating a 30-step plan) can't spin indefinitely. Emit a warning
        # so the operator can tune the limit if legitimate queries are getting
        # clipped.
        if len(plan.steps) > self._MAX_STEPS_PER_TURN:
            log.warning(
                "guardrail: plan has %d steps (max %d) — clipping for job=%s",
                len(plan.steps), self._MAX_STEPS_PER_TURN, job_id,
            )
            plan.steps = plan.steps[: self._MAX_STEPS_PER_TURN]

        # ── Guardrail: deduplicate steps ─────────────────────────────────
        # When the LLM emits duplicate tool calls (same tool + identical params),
        # skip all but the first. This prevents the identical query being sent
        # to an external API multiple times in a single turn.
        seen_step_signatures: set[str] = set()
        deduplicated_steps = []
        for step in plan.steps:
            sig = _step_signature(step)
            if sig in seen_step_signatures:
                log.debug("guardrail: deduplicating repeated step tool=%s job=%s", step.tool, job_id)
                continue
            seen_step_signatures.add(sig)
            deduplicated_steps.append(step)
        plan.steps = deduplicated_steps

        # Group steps into execution waves: parallel steps within a wave run
        # concurrently via asyncio.gather; sequential steps run one at a time.
        # Waves alternate: a batch of parallel steps, then a single sequential
        # step, then another parallel batch, etc.  Order within the plan is
        # preserved — steps are consumed left-to-right.
        planned_steps = [(i, s) for i, s in enumerate(plan.steps) if s.tool not in completed]

        # Track consecutive empty results for the early-exit guard.
        consecutive_empty: int = 0

        idx = 0
        while idx < len(planned_steps):
            step_idx, planned = planned_steps[idx]

            if planned.parallel:
                # Collect this parallel batch (all consecutive parallel steps)
                batch: list[tuple[int, Any]] = []
                while idx < len(planned_steps) and planned_steps[idx][1].parallel:
                    batch.append(planned_steps[idx])
                    idx += 1

                # Run batch concurrently
                async def _run_one(si: int, pl: Any, _results: dict) -> None:
                    t = get_tool(pl.tool)
                    if t is None:
                        log.warning("orchestrator: unknown tool %s — skipping", pl.tool)
                        return
                    params = self._inject_dependencies(pl, _results)
                    await self._run_step(
                        step_idx=si, planned=pl, params=params, tool=t,
                        job_id=job_id, task=task, primary_ns=primary_ns,
                        namespace_keys=namespace_keys, orientation=orientation,
                        expertise=expertise, results=_results,
                    )

                await asyncio.gather(*[_run_one(si, pl, results) for si, pl in batch])
            else:
                # Sequential step
                tool = get_tool(planned.tool)
                if tool is None:
                    log.warning("orchestrator: unknown tool %s — skipping", planned.tool)
                    idx += 1
                    continue

                params = self._inject_dependencies(planned, results)
                await self._run_step(
                    step_idx=step_idx, planned=planned, params=params, tool=tool,
                    job_id=job_id, task=task, primary_ns=primary_ns,
                    namespace_keys=namespace_keys, orientation=orientation,
                    expertise=expertise, results=results,
                )
                idx += 1

            # Cancel-after-wave gate
            if await self._is_cancelled(job_id):
                raise asyncio.CancelledError()

            # ── Guardrail: consecutive-empty circuit breaker ──────────────
            # Check whether any tool in the wave just run returned useful output.
            # If not, increment the counter; once it reaches the limit, abort
            # remaining steps so the synthesizer can work with what was collected.
            wave_tools = (
                [pl.tool for _, pl in batch]
                if planned.parallel
                else [planned.tool]
            )
            if any(_result_is_useful(results.get(t)) for t in wave_tools):
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                if consecutive_empty >= self._MAX_CONSECUTIVE_EMPTY:
                    log.warning(
                        "guardrail: %d consecutive empty steps — aborting remaining steps job=%s",
                        consecutive_empty, job_id,
                    )
                    break

        return results

    async def _already_completed_steps(self, job_id: str) -> dict[str, dict]:
        """Return ``{tool_name: output}`` for steps that completed in a prior run.

        Used by :meth:`_execute_plan` to skip re-execution after a restart.
        Steps in failed/cancelled state are not treated as completed — the
        next run gets a fresh attempt.
        """
        async with async_session_factory() as db:
            repo = AssistantRepository(db)
            steps = await repo.list_steps_for_job(job_id)
        return {
            s.tool_name: dict(s.output or {})
            for s in steps
            if s.status == AssistantStepStatus.completed
        }

    def _inject_dependencies(
        self, planned: PlannedStep, results: dict[str, ToolResult]
    ) -> dict[str, Any]:
        """Wire deep_search outputs into downstream steps that need paper ids."""
        params = dict(planned.params)
        if planned.tool == "genie_synthesize":
            ds = results.get("deep_search")
            if ds and ds.output.get("papers"):
                papers = ds.output["papers"][:5]
                params["paper_ids"] = [str(p.get("paper_id") or "") for p in papers if p.get("paper_id")]
                params["paper_titles"] = [str(p.get("title") or "") for p in papers]
        return params

    async def _run_step(
        self,
        *,
        step_idx: int,
        planned: PlannedStep,
        params: dict[str, Any],
        tool,
        job_id: str,
        task: AssistantTask,
        primary_ns: str,
        namespace_keys: list[str],
        orientation: str,
        expertise: str,
        results: dict[str, ToolResult],
    ) -> None:
        """Execute one planned step with its own transaction + step row."""
        async with async_session_factory() as db:
            repo = AssistantRepository(db)
            step = await repo.create_step(
                session_id=task.session_id,
                parent_message_id=task.assistant_message_id,
                job_id=job_id,
                step_index=step_idx,
                tool_name=planned.tool,
                title=planned.title,
                input_params=params,
            )
            await repo.update_step(step.id, status=AssistantStepStatus.running, started=True)
            await db.commit()
            step_id = step.id

        self._publish(job_id, "step_started", {
            "step_id": str(step_id), "step_index": step_idx,
            "tool": planned.tool, "title": planned.title,
        })

        emit_progress = self._make_progress_emitter(job_id, step_id, planned.tool, step_idx)
        should_cancel = self._make_cancel_checker(job_id)

        try:
            validated = tool.input_schema(**params)
        except Exception as exc:
            await self._mark_step(step_id, status=AssistantStepStatus.failed,
                                  error=f"Invalid params: {exc}", completed=True)
            self._publish(job_id, "step_failed", {
                "step_id": str(step_id), "tool": planned.tool,
                "error": f"Invalid params: {exc}", "retryable": False,
            })
            return

        # Cache lookup for pure tools — short-circuit + still write step row.
        cache_key: str | None = None
        if self._cache.is_cacheable(tool):
            cache_key = self._cache.make_key(
                tool_name=planned.tool,
                params=params,
                user_id=task.user_id,
                namespace_key=primary_ns,
            )
            cached = await self._cache.get(cache_key)
            if cached is not None:
                cached_summary = str(cached.get("__summary") or "cache hit")
                cached_output = {k: v for k, v in cached.items() if k != "__summary"}
                result = ToolResult(output=cached_output, summary=cached_summary)
                results[planned.tool] = result
                await self._mark_step(
                    step_id, status=AssistantStepStatus.completed,
                    output=_json_safe(cached_output),
                    cost={"cache_hit": True}, completed=True,
                    progress={"summary": f"cache hit · {cached_summary}", "percent": 100},
                )
                self._publish(job_id, "step_completed", {
                    "step_id": str(step_id), "tool": planned.tool,
                    "summary": cached_summary, "cache_hit": True,
                })
                return

        async with async_session_factory() as db:
            ctx = ToolContext(
                user_id=task.user_id,
                session_id=task.session_id,
                namespace_key=primary_ns,
                namespace_keys=namespace_keys,
                orientation=orientation,
                expertise_level=expertise,
                job_id=job_id,
                parent_message_id=task.assistant_message_id,
                db=db,
                should_cancel=should_cancel,
                emit_progress=emit_progress,
                metadata={"step_id": str(step_id)},
            )
            # Tag every LLM call inside this tool with its name so the
            # token-usage table shows per-tool spend ("assistant"/"deep_search").
            from app.core.tracking import set_workflow_context

            set_workflow_context("assistant", planned.tool)
            try:
                result = await asyncio.wait_for(
                    tool.run(ctx, validated),
                    timeout=self._MAX_STEP_DURATION_S,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "guardrail: step timeout (%ds) tool=%s step=%s",
                    self._MAX_STEP_DURATION_S, planned.tool, step_id,
                )
                await self._mark_step(step_id, status=AssistantStepStatus.failed,
                                      error="Step timed out", completed=True)
                self._publish(job_id, "step_failed", {
                    "step_id": str(step_id), "tool": planned.tool,
                    "error": f"Step timed out after {self._MAX_STEP_DURATION_S}s", "retryable": False,
                })
                return
            except asyncio.CancelledError:
                await self._mark_step(step_id, status=AssistantStepStatus.cancelled,
                                      completed=True, error="Cancelled by user")
                self._publish(job_id, "step_failed", {
                    "step_id": str(step_id), "tool": planned.tool,
                    "error": "cancelled", "retryable": False,
                })
                raise
            except Exception as exc:
                log.exception("tool %s failed step=%s", planned.tool, step_id)
                await self._mark_step(step_id, status=AssistantStepStatus.failed,
                                      error=str(exc)[:1000], completed=True)
                self._publish(job_id, "step_failed", {
                    "step_id": str(step_id), "tool": planned.tool,
                    "error": str(exc)[:240], "retryable": True,
                })
                return

            results[planned.tool] = result
            # Persist artifacts produced by this step.
            for art in result.artifacts:
                try:
                    repo = AssistantRepository(db)
                    await repo.create_artifact(
                        session_id=task.session_id,
                        user_id=task.user_id,
                        kind=str(art.get("kind") or "unknown"),
                        ref_id=str(art.get("ref_id") or ""),
                        title=str(art.get("title") or ""),
                        href=art.get("href"),
                        preview=art.get("preview") or {},
                        producing_step_id=step_id,
                        producing_message_id=task.assistant_message_id,
                    )
                    await db.commit()
                except Exception:
                    log.exception("failed to persist artifact step=%s", step_id)
                    await db.rollback()

        if cache_key is not None:
            payload = dict(result.output)
            payload["__summary"] = result.summary
            await self._cache.set(cache_key, payload, tool_name=planned.tool)

        await self._mark_step(
            step_id,
            status=AssistantStepStatus.completed,
            output=_json_safe(result.output),
            cost=result.cost or {},
            completed=True,
            progress={"summary": result.summary, "percent": 100},
        )
        self._publish(job_id, "step_completed", {
            "step_id": str(step_id), "tool": planned.tool,
            "summary": result.summary, "cache_hit": False,
        })

    # ── Phase: finalize ───────────────────────────────────────────────────

    async def _finalize_turn(
        self,
        *,
        job_id: str,
        task: AssistantTask,
        plan: Plan,
        results: dict[str, ToolResult],
        query: str,
        orientation: str,
        expertise: str,
        complexity: str = "medium",
    ) -> None:
        # ── Coverage guard ────────────────────────────────────────────────
        # Auto-run arxiv_import when < 2 corpus papers were found, UNLESS:
        #  - arXiv was already imported this turn, OR
        #  - a domain-specific tool (pubmed, inspire_hep, nasa_ads, etc.)
        #    already returned papers/data for this query.
        # Domain tools provide native coverage for their namespace; stacking
        # arXiv on top adds noise without grounding value.
        papers_preview = self._papers_from_results(results)
        already_imported = "arxiv_import" in results
        is_pure_reasoning = len(plan.steps) == 0
        has_domain_coverage = self._has_domain_coverage(results)
        if (
            not is_pure_reasoning
            and len(papers_preview) < 2
            and not already_imported
            and not has_domain_coverage
        ):
            results = await self._coverage_import(
                results=results, query=query, task=task, job_id=job_id,
            )

        await self._append_progress(job_id, plan.actions, "Synthesizing answer",
                                    _STAGE_PROGRESS["synthesizing"])
        papers = self._papers_from_results(results)
        arxiv_results = self._arxiv_results_from_results(results)
        imported_count = self._imported_from_results(results)
        graph_result = self._graph_from_results(results)
        genie_session_id = self._genie_from_results(results)
        web_results = self._web_results_from_results(results)
        comparison = self._comparison_from_results(results)
        bookmarks_answer = self._bookmarks_answer_from_results(results)
        mermaid_pair = self._mermaid_from_results(results)
        domain_papers = self._domain_papers_from_results(results)
        nvd_results = self._nvd_from_results(results)
        fred_series = self._fred_from_results(results)
        trials_results = self._trials_from_results(results)
        code_results = self._code_from_results(results)

        message_id = task.assistant_message_id

        async def _on_delta(chunk: str) -> None:
            self._publish(job_id, "message_delta", {"message_id": message_id, "delta": chunk})

        answer = await synthesize_answer(
            query=query,
            papers=papers,
            arxiv_results=arxiv_results,
            imported_count=imported_count,
            graph_result=graph_result,
            genie_session_id=genie_session_id,
            orientation=orientation,
            expertise=expertise,
            actions=plan.actions,
            extra_results=results,
            complexity=complexity,
            on_delta=_on_delta,
        )

        citations = [str(p["paper_id"]) for p in papers if p.get("paper_id")]
        artifact_refs: list[dict] = []
        if genie_session_id:
            artifact_refs.append({
                "type": "genie_session",
                "id": genie_session_id,
                "href": "/genie?tab=discoveries",
                "label": "Genie synthesis",
            })

        blocks = build_message_blocks(
            answer=answer,
            papers=papers,
            arxiv_results=arxiv_results,
            imported_count=imported_count,
            graph_result=graph_result,
            genie_session_id=genie_session_id,
            suggestions=[],
            actions=plan.actions,
            web_results=web_results,
            comparison=comparison,
            bookmarks_answer=bookmarks_answer,
            mermaid=mermaid_pair,
            domain_papers=domain_papers,
            nvd_results=nvd_results,
            fred_series=fred_series,
            trials_results=trials_results,
            code_results=code_results,
        )

        # ``payload.workflow.actions`` and friends preserved for backward
        # compatibility with the existing JobsPanel/UI rendering. ``blocks``
        # is the M2-style structured render contract.
        payload = {
            "status": "completed",
            "workflow": {"actions": plan.actions, "trace": plan.trace},
            "papers": papers,
            "arxiv_results": arxiv_results[:8],
            "imported_count": imported_count,
            "graph_result": graph_result,
            "genie_session_id": genie_session_id,
            "web_results": web_results[:6],
            "comparison": comparison,
            "bookmarks_answer": bookmarks_answer,
            "domain_papers": domain_papers,
            "nvd_results": nvd_results[:8] if nvd_results else [],
            "fred_series": fred_series[:4] if fred_series else [],
            "trials_results": trials_results[:8] if trials_results else [],
            "code_results": code_results[:8] if code_results else [],
            "suggestions": [],
            "blocks": blocks,
        }

        async with async_session_factory() as db:
            repo = AssistantRepository(db)
            await repo.update_message(
                task.assistant_message_id,
                content=answer,
                citations=citations,
                artifact_refs=artifact_refs,
                payload=payload,
                message_type="research_result",
            )
            await repo.update_task(
                job_id,
                status=AssistantTaskStatus.completed,
                progress={"stage": "completed", "percent": _STAGE_PROGRESS["completed"],
                          "summary": "Assistant task completed", "actions": plan.actions},
                result={"citation_count": len(citations), "imported_count": imported_count},
                completed=True,
            )
            await db.commit()

        await get_job_store().update(job_id, {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": "Assistant task completed",
        })

        def _log_task_exc(t: asyncio.Task, label: str) -> None:
            """Done callback: log any exception from a fire-and-forget task."""
            if not t.cancelled() and (exc := t.exception()):
                log.warning("background task %s failed job=%s: %s", label, job_id, exc)

        # Fire-and-forget metadata refresh — title (when still placeholder)
        # and summary, derived from the conversation. Never blocks the
        # user-facing answer; failures are silent (see session_metadata.py).
        # Tasks rooted in self._post_turn_tasks against Python 3.12+ GC.
        try:
            from app.assistant.session_metadata import refresh_session_metadata

            t = asyncio.create_task(
                refresh_session_metadata(task.session_id, task.user_id),
                name=f"ra:meta:{job_id}",
            )
            self._post_turn_tasks.add(t)
            t.add_done_callback(lambda _t: (_log_task_exc(_t, "session_metadata"), self._post_turn_tasks.discard(_t)))
        except Exception:
            log.exception("failed to schedule session metadata refresh job=%s", job_id)

        # Fire-and-forget interest profile update — folds the concepts the
        # user actually engaged with into UserInterestProfile.concept_affinity
        # so subsequent frontier_scan / deep_search runs bias toward their
        # evolving interests. Pure DB work, no LLM cost.
        try:
            from app.assistant.interest_updater import update_from_turn

            t = asyncio.create_task(
                update_from_turn(
                    user_id=task.user_id,
                    user_query=query,
                    cited_paper_ids=citations,
                    retrieved_papers=papers,
                ),
                name=f"ra:interest:{job_id}",
            )
            self._post_turn_tasks.add(t)
            t.add_done_callback(lambda _t: (_log_task_exc(_t, "interest_updater"), self._post_turn_tasks.discard(_t)))
        except Exception:
            log.exception("failed to schedule interest profile update job=%s", job_id)

        self._publish(job_id, "message_completed", {
            "message_id": str(task.assistant_message_id),
            "citation_count": len(citations),
            "imported_count": imported_count,
            "blocks": payload.get("blocks") or [],
        })
        self._publish(job_id, "task_completed", {
            "summary": "Assistant task completed",
            "citation_count": len(citations),
        })
        # Drop the channel after a short grace window so late SSE subscribers
        # still get the closing events; the bus self-evicts via close().
        try:
            self._bus.close(job_id)
        except Exception:
            pass

    # ── Coverage guard ────────────────────────────────────────────────────

    async def _coverage_import(
        self,
        *,
        results: dict[str, ToolResult],
        query: str,
        task: AssistantTask,
        job_id: str,
    ) -> dict[str, ToolResult]:
        """Auto-import arXiv papers when the corpus returned fewer than 2 results.

        This ensures that for any research query — even when the user's feed is
        empty or lacks the relevant papers — RA surfaces grounded context. It is
        the mechanism that makes 'explain attention mechanism' fetch
        'Attention Is All You Need' when it isn't in the corpus yet.
        """
        try:
            from app.assistant.tools.arxiv_import import ArxivImportInput, ArxivImportTool
            from app.assistant.tools.base import ToolContext
            from app.db.session import async_session_factory

            await self._append_progress(
                job_id, [], "Searching arXiv for relevant papers…", 50
            )

            async def _noop_progress(pct: int, msg: str) -> None:
                pass

            async def _noop_cancel() -> bool:
                return False

            async with async_session_factory() as db:
                ctx = ToolContext(
                    user_id=task.user_id,
                    session_id=task.session_id,
                    namespace_key=task.namespace_key or "",
                    namespace_keys=[task.namespace_key] if task.namespace_key else [],
                    orientation="both",
                    expertise_level="practitioner",
                    job_id=job_id,
                    parent_message_id=task.assistant_message_id or task.session_id,
                    db=db,
                    emit_progress=_noop_progress,
                    should_cancel=_noop_cancel,
                )
                tool = ArxivImportTool()
                tool_params = ArxivImportInput(
                    query=query,
                    namespace_key=task.namespace_key,
                    max_results=8,
                )
                import_result = await tool.run(ctx, tool_params)
                await db.commit()

            n = import_result.output.get("imported", 0)
            log.info(
                "coverage guard imported %d arXiv paper(s) for thin corpus (job=%s)",
                n, job_id,
            )
            results = dict(results)
            results["arxiv_import"] = import_result
        except Exception as exc:
            log.warning("coverage guard arXiv import failed: %s", exc)
        return results

    # ── Result extractors ────────────────────────────────────────────────

    @staticmethod
    def _papers_from_results(results: dict[str, ToolResult]) -> list[dict]:
        """Combine grounded papers from any retrieval tool that produced them.

        Order of preference for the synthesizer's primary corpus:
        ``deep_search`` → ``frontier_scan`` → ``concept_explain.supporting_papers``.
        Falls through to ``arxiv_import`` results when all else is empty
        (coverage guard).

        Relevance gating: deep_search rank-normalises ``search_score`` to
        ``1.0 / 0.x`` where higher == more relevant; we keep entries scoring
        at least :data:`_MIN_GROUNDING_SCORE` and cap the surviving list at
        :data:`_MAX_GROUNDING_PAPERS`. This trims the "weakly-related" tail
        that was previously rendering in the *Grounded papers* block when the
        deep_search recall went wide. When fewer than 2 papers clear the
        threshold we keep the top-N anyway so the UI never collapses to an
        empty grid — a small set of mid-relevance papers is more useful than
        nothing for orientation.
        """

        def _filter(papers: list[dict]) -> list[dict]:
            if not papers:
                return papers
            # Sort by best available signal — search_score for deep_search
            # ranked results, otherwise fall back to relevance_score.
            def _score(p: dict) -> float:
                try:
                    s = p.get("search_score")
                    if s is not None:
                        return float(s)
                    return float(p.get("relevance_score") or 0.0)
                except (TypeError, ValueError):
                    return 0.0
            sorted_papers = sorted(papers, key=_score, reverse=True)
            kept = [p for p in sorted_papers if _score(p) >= _MIN_GROUNDING_SCORE]
            if len(kept) < 2:
                kept = sorted_papers[:_MAX_GROUNDING_PAPERS]
            return kept[:_MAX_GROUNDING_PAPERS]

        for tool_name in ("deep_search", "frontier_scan"):
            r = results.get(tool_name)
            if r and r.output.get("papers"):
                return _filter(list(r.output["papers"]))
        cx = results.get("concept_explain")
        if cx and cx.output.get("supporting_papers"):
            return _filter(list(cx.output["supporting_papers"]))
        # Coverage guard fallback: treat imported arXiv papers as corpus
        ai = results.get("arxiv_import")
        if ai and ai.output.get("arxiv_results"):
            return _filter(list(ai.output["arxiv_results"]))
        return []

    @staticmethod
    def _arxiv_results_from_results(results: dict[str, ToolResult]) -> list[dict]:
        ai = results.get("arxiv_import")
        if ai and ai.output.get("arxiv_results"):
            return list(ai.output["arxiv_results"])
        asch = results.get("arxiv_search")
        return list(asch.output.get("results") or []) if asch else []

    @staticmethod
    def _imported_from_results(results: dict[str, ToolResult]) -> int:
        ai = results.get("arxiv_import")
        return int(ai.output.get("imported") or 0) if ai else 0

    @staticmethod
    def _graph_from_results(results: dict[str, ToolResult]) -> dict | None:
        # graph_build is planner-forbidden but we honour any external entry
        # point that might still write the row. graph_query (read-only) is
        # surfaced as a "graph_summary" block instead of a "build" event.
        gb = results.get("graph_build")
        if gb:
            return gb.output.get("result")
        gq = results.get("graph_query")
        if gq and gq.output.get("has_graph"):
            return gq.output.get("summary")
        return None

    @staticmethod
    def _genie_from_results(results: dict[str, ToolResult]) -> str | None:
        gs = results.get("genie_synthesize")
        return gs.output.get("genie_session_id") if gs else None

    @staticmethod
    def _web_results_from_results(results: dict[str, ToolResult]) -> list[dict]:
        ws = results.get("web_search")
        return list(ws.output.get("results") or []) if ws else []

    @staticmethod
    def _comparison_from_results(results: dict[str, ToolResult]) -> dict | None:
        cp = results.get("compare_papers")
        if cp and cp.output.get("rows"):
            return {
                "columns": cp.output.get("columns") or [],
                "rows": cp.output.get("rows") or [],
                "notes": cp.output.get("notes") or "",
            }
        return None

    @staticmethod
    def _bookmarks_answer_from_results(results: dict[str, ToolResult]) -> str | None:
        bq = results.get("bookmarks_query")
        return (bq.output.get("answer") or None) if bq else None

    @staticmethod
    def _mermaid_from_results(results: dict[str, ToolResult]) -> tuple[str, str] | None:
        """Return (title, mermaid_code) when concept_explain produced a map."""
        cx = results.get("concept_explain")
        if cx and cx.output.get("mermaid"):
            concept = str(cx.output.get("concept") or "Concept map")
            return f"{concept} — concept map", str(cx.output["mermaid"])
        return None

    @staticmethod
    def _has_domain_coverage(results: dict[str, ToolResult]) -> bool:
        """Return True when domain-specific tools already produced paper/data results.

        When True, the coverage guard skips arXiv import — the domain tool
        already provided native coverage appropriate for the namespace.
        """
        domain_tools = (
            "pubmed", "inspire_hep", "nasa_ads", "papers_with_code",
            "nvd_cve", "clinicaltrials", "fred", "oeis",
            "github_search", "huggingface_search",
        )
        for tool in domain_tools:
            r = results.get(tool)
            if not r:
                continue
            out = r.output
            # Any non-empty list in the output counts as domain coverage
            for key in ("papers", "studies", "series", "sequences", "repositories",
                        "results", "vulnerabilities", "trials"):
                val = out.get(key)
                if val and isinstance(val, list) and len(val) > 0:
                    return True
        return False

    @staticmethod
    def _domain_papers_from_results(results: dict[str, ToolResult]) -> list[dict]:
        """Collect papers from domain-specific tools (pubmed, inspire_hep, nasa_ads).

        Returns a flat list with a ``source`` field indicating the tool origin.
        """
        domain_paper_tools = {
            "pubmed": ("papers", "PubMed"),
            "inspire_hep": ("papers", "INSPIRE HEP"),
            "nasa_ads": ("papers", "NASA ADS"),
            "papers_with_code": ("results", "Papers with Code"),
        }
        collected: list[dict] = []
        for tool, (key, source_label) in domain_paper_tools.items():
            r = results.get(tool)
            if not r:
                continue
            papers = r.output.get(key) or []
            for p in papers[:8]:
                entry = dict(p)
                entry.setdefault("source", source_label)
                collected.append(entry)
        return collected[:16]

    @staticmethod
    def _nvd_from_results(results: dict[str, ToolResult]) -> list[dict]:
        r = results.get("nvd_cve")
        return list(r.output.get("vulnerabilities") or []) if r else []

    @staticmethod
    def _fred_from_results(results: dict[str, ToolResult]) -> list[dict]:
        r = results.get("fred")
        return list(r.output.get("series") or []) if r else []

    @staticmethod
    def _trials_from_results(results: dict[str, ToolResult]) -> list[dict]:
        r = results.get("clinicaltrials")
        return list(r.output.get("studies") or []) if r else []

    @staticmethod
    def _code_from_results(results: dict[str, ToolResult]) -> list[dict]:
        """Collect code results from github_search and huggingface_search."""
        collected: list[dict] = []
        gh = results.get("github_search")
        if gh:
            for repo in (gh.output.get("repositories") or [])[:8]:
                collected.append({"kind": "repo", "source": "GitHub", **repo})
        hf = results.get("huggingface_search")
        if hf:
            search_type = hf.output.get("search_type", "models")
            for item in (hf.output.get("results") or [])[:6]:
                collected.append({"kind": search_type, "source": "HuggingFace", **item})
        return collected[:12]

    # ── Off-topic short-circuit ──────────────────────────────────────────

    async def _finalize_off_topic(self, job_id: str, task: AssistantTask, query: str) -> None:
        """Persist a polite redirect message and complete the task without running any tools."""
        async with async_session_factory() as db:
            repo = AssistantRepository(db)
            if task.assistant_message_id:
                await repo.update_message(
                    task.assistant_message_id,
                    content=_OFF_TOPIC_REDIRECT,
                    payload={
                        "status": "completed",
                        "blocks": [{"kind": "text", "content": _OFF_TOPIC_REDIRECT}],
                    },
                    message_type="text",
                )
            await repo.update_task(
                job_id,
                status=AssistantTaskStatus.completed,
                progress={"stage": "completed", "percent": 100, "summary": "Query out of scope"},
                result={"off_topic": True},
                completed=True,
            )
            await db.commit()
        await get_job_store().update(job_id, {"status": "completed", "summary": "Query out of research scope"})
        self._publish(job_id, "message_completed", {
            "message_id": str(task.assistant_message_id) if task.assistant_message_id else "",
            "off_topic": True,
        })
        self._publish(job_id, "task_completed", {"summary": "Query redirected — out of research scope"})
        try:
            self._bus.close(job_id)
        except Exception:
            pass

    # ── Bookkeeping helpers ──────────────────────────────────────────────

    def _make_progress_emitter(self, job_id: str, step_id: UUID, tool_name: str, step_index: int = 0):
        async def emit(percent: int, summary: str) -> None:
            await self._mark_step(
                step_id,
                progress={"percent": percent, "summary": summary, "tool": tool_name},
            )
            self._publish(job_id, "step_progress", {
                "step_id": str(step_id), "step_index": step_index,
                "tool": tool_name, "percent": percent, "summary": summary,
            })
        return emit

    def _publish(self, job_id: str, kind, payload: dict[str, Any]) -> None:
        """Fire an AssistantEvent on the bus. Best-effort — never raises."""
        try:
            self._bus.publish(AssistantEvent(kind=kind, job_id=job_id, payload=payload))
        except Exception:
            log.exception("event bus publish failed kind=%s job=%s", kind, job_id)

    def _make_cancel_checker(self, job_id: str):
        async def check() -> bool:
            return await self._is_cancelled(job_id)
        return check

    async def _is_cancelled(self, job_id: str) -> bool:
        async with async_session_factory() as db:
            task = await self._get_task(db, job_id)
            return bool(task and task.cancel_requested_at)

    async def _mark_step(self, step_id: UUID, **fields) -> None:
        async with async_session_factory() as db:
            repo = AssistantRepository(db)
            await repo.update_step(step_id, **fields)
            await db.commit()

    async def _mark_task(
        self,
        job_id: str,
        status: AssistantTaskStatus,
        stage: str,
        percent: int,
        summary: str,
        *,
        started: bool = False,
    ) -> None:
        async with async_session_factory() as db:
            repo = AssistantRepository(db)
            await repo.update_task(
                job_id,
                status=status,
                progress={"stage": stage, "percent": percent, "summary": summary},
                started=started,
            )
            await db.commit()
        await get_job_store().update(job_id, {"status": status.value, "summary": summary})

    async def _append_progress(self, job_id: str, actions: list[str], summary: str, percent: int) -> None:
        async with async_session_factory() as db:
            repo = AssistantRepository(db)
            await repo.update_task(
                job_id,
                status=AssistantTaskStatus.running,
                progress={"stage": summary, "percent": percent, "summary": summary, "actions": actions},
            )
            await db.commit()
        await get_job_store().update(job_id, {"status": "running", "summary": summary})

    async def _handle_cancelled(self, job_id: str) -> None:
        async with async_session_factory() as db:
            repo = AssistantRepository(db)
            task = await repo.update_task(
                job_id,
                status=AssistantTaskStatus.cancelled,
                progress={"stage": "cancelled", "percent": 100, "summary": "Assistant task cancelled"},
                completed=True,
                cancel_requested=True,
            )
            if task and task.assistant_message_id:
                await repo.update_message(
                    task.assistant_message_id,
                    content="Cancelled. Partial results, if any, were left safely in the workspace.",
                    payload={"status": "cancelled"},
                    message_type="workflow",
                )
            await db.commit()
        await get_job_store().update(job_id, {
            "status": "cancelled",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": "Assistant task cancelled",
        })
        self._publish(job_id, "task_cancelled", {"summary": "Assistant task cancelled"})
        try:
            self._bus.close(job_id)
        except Exception:
            pass

    async def _handle_failed(self, job_id: str, error: str) -> None:
        async with async_session_factory() as db:
            repo = AssistantRepository(db)
            task = await repo.update_task(
                job_id,
                status=AssistantTaskStatus.failed,
                progress={"stage": "failed", "percent": 100, "summary": "Assistant task failed"},
                error=error[:1000],
                completed=True,
            )
            if task and task.assistant_message_id:
                await repo.update_message(
                    task.assistant_message_id,
                    content=f"I hit an orchestration error: {error}",
                    payload={"status": "failed", "error": error[:1000]},
                    message_type="workflow",
                )
            await db.commit()
        await get_job_store().update(job_id, {
            "status": "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": error[:240],
        })
        self._publish(job_id, "task_failed", {"error": error[:1000]})
        try:
            self._bus.close(job_id)
        except Exception:
            pass

    @staticmethod
    async def _get_task(db, job_id: str) -> AssistantTask | None:
        result = await db.execute(select(AssistantTask).where(AssistantTask.job_id == job_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def _latest_user_message(db, session_id: UUID) -> AssistantMessage | None:
        result = await db.execute(
            select(AssistantMessage)
            .where(
                AssistantMessage.session_id == session_id,
                AssistantMessage.role == AssistantMessageRole.user,
            )
            .order_by(AssistantMessage.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


def _suggest_next_steps(plan: Plan, has_genie: bool, has_graph: bool) -> list[dict]:
    """Heuristic next-step suggestions surfaced in the message payload."""
    suggestions = [
        {"label": "Open cited papers", "href": None, "kind": "paper_review"},
        {"label": "Run a narrower Deep Search", "href": "/assistant", "kind": "search"},
    ]
    if has_graph:
        suggestions.append({"label": "Inspect graph map", "href": "/graph", "kind": "graph"})
    if has_genie:
        suggestions.append({"label": "Review Genie synthesis", "href": "/genie?tab=discoveries", "kind": "genie"})
    if any(a.startswith("artifact") for a in plan.actions):
        suggestions.append({"label": "Generate slides or podcast from a selected paper",
                            "href": None, "kind": "artifact"})
    return suggestions


def _json_safe(value: Any) -> Any:
    """Coerce a tool output dict into JSONB-safe primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


import hashlib as _hashlib
import json as _json


def _step_signature(step: Any) -> str:
    """Return a deterministic signature for a planned step.

    Used by the deduplication guardrail to detect when the planner emitted
    the exact same tool + params combination twice in one plan.
    """
    params_repr = _json.dumps(
        {k: v for k, v in sorted(getattr(step, "params", {}).items())},
        sort_keys=True, default=str,
    )
    return _hashlib.md5(f"{step.tool}:{params_repr}".encode()).hexdigest()


def _result_is_useful(result: Any) -> bool:
    """Return True when a ToolResult contains at least one non-empty payload.

    Used by the consecutive-empty guardrail to decide whether to abort
    remaining steps early. A result is "useful" when any list value in its
    output is non-empty, or any string value is non-empty.
    """
    if result is None:
        return False
    out = getattr(result, "output", None) or {}
    for v in out.values():
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, (int, float)) and v > 0:
            return True
        if isinstance(v, dict) and v:
            return True
    return False
