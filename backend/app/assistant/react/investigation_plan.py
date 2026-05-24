"""Mid-loop investigation tracker — the model's structured task list.

The pre-existing scratchpad is great for *post-hoc* inspection — it
records what happened. What it doesn't give the model is a place to
declare what it *intends* to do across iterations. Without that, a
ReAct loop can:

  * Drift away from the original plan because the loop's working
    memory is only the recent Thought/Action/Observation entries;
    earlier intentions decay out of context.
  * Re-investigate things it already investigated because the model
    doesn't remember which sub-questions it's already answered.
  * Miss completing the user's request when the question is
    multi-part — the model handles one part and finalizes, having
    forgotten the others.

This module is the durable mid-loop task list, modelled on
deepagents' ``write_todos`` tool but implemented over our own state.

The model interacts via the pseudo-action ``write_todos`` with a
payload of structured operations (add / update / complete / clear).
The plan renders into the next decision prompt so the model always
sees its own intentions. The plan also lands on ``ReactOutcome.plan``
so the synthesizer and the UI can render the actual investigation
trajectory.

Design points:

* Stable IDs — every todo gets a stable short slug (``t1``, ``t2``)
  so the model can reference one across iterations.
* No deletes — only completion or cancellation, so the audit trail
  stays intact. Stuck todos surface in the synthesizer's agent_notes
  as "remained open at finalize".
* Bounded size — capped at ``_MAX_TODOS`` per turn so a runaway model
  can't fill the prompt with hundreds of items.
* Pure data — no LLM call in this module. Just structured state the
  loop driver manages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


_MAX_TODOS = 20
_MAX_TODO_TEXT = 280


TodoStatus = Literal["pending", "in_progress", "completed", "cancelled"]


@dataclass
class Todo:
    """One investigation step the model wants to remember across iterations.

    Fields:
      * ``id``        — stable short slug the model can reference
                        (e.g. ``"t1"``, ``"t2"``).
      * ``text``      — what the step IS. Short noun-phrase or imperative.
      * ``status``    — lifecycle position.
      * ``iteration`` — the iteration on which this todo was last touched
                        (added / updated / completed). Used for the
                        "stuck-in-progress" surfacing.
      * ``evidence``  — optional list of paper_ids / tool result keys that
                        ground this step's completion. Populated by the
                        loop when the model marks a todo done.
    """

    id: str
    text: str
    status: TodoStatus = "pending"
    iteration: int = 0
    evidence: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Compact one-line render for the decision prompt."""
        symbol = {"pending": "[ ]", "in_progress": "[~]",
                  "completed": "[x]", "cancelled": "[-]"}[self.status]
        evidence = (
            f"  ev=[{', '.join(self.evidence[:3])}{'...' if len(self.evidence) > 3 else ''}]"
            if self.evidence else ""
        )
        return f"  {symbol} {self.id}: {self.text[:240]}{evidence}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigationPlan:
    """The full per-turn todo list.

    Operations come in via :meth:`apply_operations` — a list of
    declarative ``op`` dicts the model emits inside the
    ``write_todos`` pseudo-action payload. Each op has a ``kind``
    (``add`` / ``update`` / ``complete`` / ``cancel``) and the
    relevant fields. We apply them in order, drop malformed entries,
    and never let the plan exceed ``_MAX_TODOS``.

    The plan is *append-mostly* — completed and cancelled todos stay
    in the list so the audit trail is preserved. Eviction only fires
    when the cap is reached, and it evicts completed/cancelled
    entries first (the model's open work is never silently dropped).
    """

    todos: list[Todo] = field(default_factory=list)
    next_id: int = 1
    last_updated_iteration: int = 0

    # ── Query ──────────────────────────────────────────────────────

    def open_todos(self) -> list[Todo]:
        return [t for t in self.todos if t.status in ("pending", "in_progress")]

    def by_id(self, todo_id: str) -> Todo | None:
        for t in self.todos:
            if t.id == todo_id:
                return t
        return None

    def stuck_in_progress(self, *, current_iteration: int, slack: int = 2) -> list[Todo]:
        """Todos that have been ``in_progress`` for more iterations
        than ``slack`` allows. The synthesizer surfaces these as
        "investigated but never marked done" in agent_notes."""
        return [
            t for t in self.todos
            if t.status == "in_progress"
            and (current_iteration - t.iteration) >= slack
        ]

    # ── Rendering ──────────────────────────────────────────────────

    def render_for_prompt(self, *, limit: int = 12) -> str:
        """Compact view for the decision prompt's plan block.

        Open todos first, then recently-completed for context, then
        anything older / cancelled. Capped at ``limit`` so the
        prompt stays bounded even on long investigations.
        """
        if not self.todos:
            return "(no investigation plan yet — use 'write_todos' to draft one)"
        order = {"pending": 0, "in_progress": 0, "completed": 1, "cancelled": 2}
        sorted_todos = sorted(
            self.todos,
            key=lambda t: (order.get(t.status, 3), -t.iteration),
        )
        lines = [t.render() for t in sorted_todos[:limit]]
        if len(self.todos) > limit:
            lines.append(f"  ... and {len(self.todos) - limit} more")
        return "\n".join(lines)

    def summarize_for_synth(self) -> dict[str, Any]:
        """Compact dict for ``ReactOutcome.plan`` + ``agent_notes``.

        Carries the open-vs-completed split + the stuck-in-progress
        list so the synthesizer can flag unfinished work in the
        answer.
        """
        completed = [t.text for t in self.todos if t.status == "completed"]
        open_ = [t.text for t in self.open_todos()]
        cancelled = [t.text for t in self.todos if t.status == "cancelled"]
        return {
            "total": len(self.todos),
            "completed": completed,
            "open": open_,
            "cancelled": cancelled,
            "stuck_in_progress": [
                t.text for t in self.stuck_in_progress(
                    current_iteration=self.last_updated_iteration, slack=2,
                )
            ],
        }

    # ── Mutation ───────────────────────────────────────────────────

    def apply_operations(
        self,
        ops: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> list[str]:
        """Apply a batch of declarative operations.

        Returns a list of human-readable notes describing what
        actually happened (skipped malformed ops, capped-out adds,
        etc.). The loop driver surfaces these on the scratchpad so
        the model can see why its op didn't land.
        """
        notes: list[str] = []
        if not isinstance(ops, list):
            # Bump the iteration cursor even when the batch is malformed
            # so ``stuck_in_progress`` still tracks against the current
            # iteration window. Without this, a session of malformed
            # ``write_todos`` calls would keep ``last_updated_iteration``
            # at its prior value and surface stale "stuck" todos.
            self.last_updated_iteration = iteration
            return ["write_todos payload was not a list — ignored"]
        for op in ops:
            if not isinstance(op, dict):
                notes.append("skipped non-dict op")
                continue
            kind = str(op.get("kind") or op.get("op") or "").strip().lower()
            try:
                if kind == "add":
                    note = self._op_add(op, iteration=iteration)
                elif kind == "update":
                    note = self._op_update(op, iteration=iteration)
                elif kind == "complete":
                    note = self._op_complete(op, iteration=iteration)
                elif kind == "cancel":
                    note = self._op_cancel(op, iteration=iteration)
                elif kind == "clear":
                    note = self._op_clear(iteration=iteration)
                else:
                    note = f"unknown op kind {kind!r}"
            except Exception as exc:  # noqa: BLE001 — never let one bad op kill the batch
                note = f"op {kind!r} failed: {exc!s}"
            notes.append(note)
        self.last_updated_iteration = iteration
        self._enforce_cap()
        return notes

    def _op_add(self, op: dict[str, Any], *, iteration: int) -> str:
        text = str(op.get("text") or "").strip()
        if not text:
            return "skipped add: empty text"
        text = text[:_MAX_TODO_TEXT]
        todo_id = self._next_slug()
        status: TodoStatus = str(op.get("status") or "pending")  # type: ignore[assignment]
        if status not in {"pending", "in_progress", "completed", "cancelled"}:
            status = "pending"
        self.todos.append(Todo(
            id=todo_id, text=text, status=status, iteration=iteration,
            evidence=[str(e) for e in (op.get("evidence") or [])][:8],
        ))
        return f"added {todo_id}: {text[:80]}"

    def _op_update(self, op: dict[str, Any], *, iteration: int) -> str:
        todo_id = str(op.get("id") or "").strip()
        target = self.by_id(todo_id)
        if not target:
            return f"update: unknown id {todo_id!r}"
        if op.get("text"):
            target.text = str(op["text"])[:_MAX_TODO_TEXT]
        new_status = op.get("status")
        if new_status and new_status in {"pending", "in_progress", "completed", "cancelled"}:
            target.status = new_status      # type: ignore[assignment]
        ev = op.get("evidence")
        if isinstance(ev, list):
            # Merge — never overwrite existing evidence pointers.
            merged = list(target.evidence) + [str(e) for e in ev if e]
            seen: set[str] = set()
            target.evidence = [
                x for x in merged
                if not (x in seen or seen.add(x))
            ][:8]
        target.iteration = iteration
        return f"updated {todo_id}"

    def _op_complete(self, op: dict[str, Any], *, iteration: int) -> str:
        todo_id = str(op.get("id") or "").strip()
        target = self.by_id(todo_id)
        if not target:
            return f"complete: unknown id {todo_id!r}"
        target.status = "completed"
        target.iteration = iteration
        ev = op.get("evidence")
        if isinstance(ev, list):
            merged = list(target.evidence) + [str(e) for e in ev if e]
            seen: set[str] = set()
            target.evidence = [
                x for x in merged
                if not (x in seen or seen.add(x))
            ][:8]
        return f"completed {todo_id}"

    def _op_cancel(self, op: dict[str, Any], *, iteration: int) -> str:
        todo_id = str(op.get("id") or "").strip()
        target = self.by_id(todo_id)
        if not target:
            return f"cancel: unknown id {todo_id!r}"
        target.status = "cancelled"
        target.iteration = iteration
        return f"cancelled {todo_id}"

    def _op_clear(self, *, iteration: int) -> str:
        # Hard-cancel every open todo. Completed entries stay for
        # the audit trail.
        cleared = 0
        for t in self.todos:
            if t.status in ("pending", "in_progress"):
                t.status = "cancelled"
                t.iteration = iteration
                cleared += 1
        return f"cleared {cleared} open todo(s)"

    # ── Internals ───────────────────────────────────────────────────

    def _next_slug(self) -> str:
        slug = f"t{self.next_id}"
        self.next_id += 1
        return slug

    def _enforce_cap(self) -> None:
        """When the list overflows the cap, evict completed +
        cancelled entries (oldest first) until we're back under the
        cap. Open work is NEVER evicted — the model's intentions
        stay durable until the model itself completes / cancels them."""
        if len(self.todos) <= _MAX_TODOS:
            return
        # Stable partition: open first, finished after, oldest-finished
        # at the end.
        open_, finished = [], []
        for t in self.todos:
            (open_ if t.status in ("pending", "in_progress") else finished).append(t)
        finished.sort(key=lambda t: t.iteration)
        keep = _MAX_TODOS - len(open_)
        if keep < 0:
            keep = 0
        kept_finished = finished[-keep:] if keep else []
        self.todos = open_ + kept_finished
