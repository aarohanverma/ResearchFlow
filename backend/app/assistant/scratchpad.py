"""Per-turn agent scratchpad — inspectable working memory for the ReAct loop.

This module is the *working* state of a single RA turn. It complements (does
not replace) the multi-dimensional durable memory system in
``app.assistant.tools.memory`` and ``app.assistant.branch_context``:

* **Scratchpad** — lives for the duration of one turn. Typed entries record
  each THOUGHT the model writes, each ACTION it picks, the OBSERVATION the
  tool returned, any CRITIQUE judgments fired, and the PROVENANCE links
  between claims in the final answer and their evidence. Persisted to
  ``AssistantMessage.payload.scratchpad`` for post-hoc inspection.
* **chat / tree / ns memory** — durable across turns and sessions.
  Untouched by this module.
* **branch_summaries / branch_seed / history_summary** — durable rollups
  for cross-branch context. Untouched.

Design points worth knowing:

* Entries are *typed and structured*, not free strings. The model emits
  one free-text ``text`` field per Thought, but everything else is
  schema'd so we can render it in the UI, query it, and feed only the
  relevant slice back into the next LLM call.
* Heavy tool outputs (paper lists, big JSON) are NOT copied into
  observations — only a short summary plus a reference into the per-turn
  ``results: dict[str, ToolResult]``. This keeps the prompt cheap while
  preserving inspectability.
* Provenance entries are claim-level: ``claim_span`` is the chunk of
  prose, ``sources`` are paper IDs / tool result keys that support it.
  Faithful citation auditing depends on this being populated by the
  synthesizer post-processor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Typed entries ────────────────────────────────────────────────────────────


@dataclass
class Thought:
    """Free-text reasoning the model wrote between actions."""
    text: str
    iteration: int
    ts: str = field(default_factory=_now_iso)
    kind: Literal["thought"] = "thought"


@dataclass
class Action:
    """A tool the model chose to call (or ``"finalize"`` to leave the loop).

    ``rationale`` is a short justification the model emitted alongside its
    choice; surfacing it in the UI makes the loop inspectable without
    digging through the whole scratchpad.
    """
    tool: str
    params: dict
    rationale: str
    iteration: int
    ts: str = field(default_factory=_now_iso)
    kind: Literal["action"] = "action"


@dataclass
class Observation:
    """The structured result of executing an Action.

    ``summary`` is a human-readable, prompt-cheap synopsis (a couple of
    lines). ``output_ref`` is a key into the orchestrator's per-turn
    ``results`` dict where the full tool output lives — so future
    THOUGHTs can pull the heavy payload back when they actually need it.
    """
    tool: str
    summary: str
    output_ref: str
    error: str | None
    iteration: int
    ts: str = field(default_factory=_now_iso)
    kind: Literal["observation"] = "observation"


@dataclass
class Critique:
    """A critique judgment recorded after a draft (or mid-turn).

    Mirrors the shape of :func:`app.assistant.reflection.llm_critique` so
    we can reuse the existing scorer wholesale — we just record its output
    on the scratchpad as a first-class entry.
    """
    groundedness: float
    completeness: float
    memory_faithfulness: float
    issues: list[str]
    verdict: Literal["ship", "revise"]
    iteration: int
    ts: str = field(default_factory=_now_iso)
    kind: Literal["critique"] = "critique"


@dataclass
class Provenance:
    """One claim ↔ sources mapping for the final answer.

    Populated by the synthesizer post-processor — for every ``[N]``
    citation marker in the answer, this records the surrounding claim
    span and the paper IDs / tool result keys that support it.
    Auditable downstream by the UI and by the critique step.
    """
    claim_span: str
    sources: list[str]
    marker: str  # e.g. "[3]" — the rendered citation token
    iteration: int
    ts: str = field(default_factory=_now_iso)
    kind: Literal["provenance"] = "provenance"


ScratchpadEntry = Thought | Action | Observation | Critique | Provenance


# ── Container ────────────────────────────────────────────────────────────────


@dataclass
class Scratchpad:
    """Ordered, typed log of one turn's agent state.

    Cheap by construction — entries hold references, not heavy payloads.
    ``to_dict()`` round-trips through ``from_dict()`` so the scratchpad
    can be stored on ``AssistantMessage.payload`` (JSONB) and reloaded
    for inspection or for a follow-up turn that wants to read it.
    """

    entries: list[ScratchpadEntry] = field(default_factory=list)
    started_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    # Optional running counter — gives every entry a stable iteration tag
    # so the UI can group by step.
    iteration: int = 0

    # ── add helpers ──────────────────────────────────────────────────────

    def think(self, text: str) -> None:
        self.entries.append(Thought(text=text.strip(), iteration=self.iteration))

    def act(self, tool: str, params: dict, rationale: str = "") -> None:
        self.entries.append(Action(
            tool=tool,
            params=dict(params or {}),
            rationale=(rationale or "").strip(),
            iteration=self.iteration,
        ))

    def observe(self, tool: str, summary: str, output_ref: str, error: str | None = None) -> None:
        self.entries.append(Observation(
            tool=tool,
            summary=(summary or "").strip()[:1000],
            output_ref=output_ref,
            error=(error or None),
            iteration=self.iteration,
        ))

    def critique(
        self,
        *,
        groundedness: float,
        completeness: float,
        memory_faithfulness: float,
        issues: list[str],
        verdict: Literal["ship", "revise"],
    ) -> None:
        self.entries.append(Critique(
            groundedness=float(groundedness),
            completeness=float(completeness),
            memory_faithfulness=float(memory_faithfulness),
            issues=[str(i) for i in (issues or [])],
            verdict=verdict,
            iteration=self.iteration,
        ))

    def provenance(self, claim_span: str, sources: list[str], marker: str) -> None:
        self.entries.append(Provenance(
            claim_span=(claim_span or "").strip()[:600],
            sources=[str(s) for s in (sources or []) if s],
            marker=marker.strip(),
            iteration=self.iteration,
        ))

    # ── lifecycle ────────────────────────────────────────────────────────

    def next_iteration(self) -> None:
        self.iteration += 1

    def finish(self) -> None:
        self.finished_at = _now_iso()

    # ── views ────────────────────────────────────────────────────────────

    def thoughts(self) -> list[Thought]:
        return [e for e in self.entries if isinstance(e, Thought)]

    def actions(self) -> list[Action]:
        return [e for e in self.entries if isinstance(e, Action)]

    def observations(self) -> list[Observation]:
        return [e for e in self.entries if isinstance(e, Observation)]

    def provenance_entries(self) -> list[Provenance]:
        return [e for e in self.entries if isinstance(e, Provenance)]

    # ── prompt rendering ─────────────────────────────────────────────────

    def render_for_prompt(self, max_entries: int = 24) -> str:
        """Render the most recent ``max_entries`` entries as a compact
        block the next LLM call can read.

        The format is deliberately terse — we want the model to *see*
        its prior reasoning without burning thousands of tokens. Drops
        provenance entries from the prompt view (they only matter for
        UI inspection, not for next-step decisions).
        """
        recent = [e for e in self.entries if not isinstance(e, Provenance)][-max_entries:]
        if not recent:
            return "(scratchpad empty — this is the first iteration)"
        lines: list[str] = []
        for e in recent:
            tag = f"#{e.iteration}"
            if isinstance(e, Thought):
                lines.append(f"{tag} THOUGHT: {e.text[:600]}")
            elif isinstance(e, Action):
                lines.append(f"{tag} ACTION: {e.tool}({_truncate_dict(e.params)})  // {e.rationale[:200]}")
            elif isinstance(e, Observation):
                err = f" [error: {e.error[:200]}]" if e.error else ""
                lines.append(f"{tag} OBSERVATION ({e.tool}): {e.summary[:400]}{err}")
            elif isinstance(e, Critique):
                lines.append(
                    f"{tag} CRITIQUE: verdict={e.verdict} "
                    f"g={e.groundedness:.2f} c={e.completeness:.2f} "
                    f"mf={e.memory_faithfulness:.2f}"
                    + (f"; issues={'; '.join(e.issues)[:300]}" if e.issues else "")
                )
        return "\n".join(lines)

    # ── persistence ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "iteration": self.iteration,
            "entries": [asdict(e) for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scratchpad":
        pad = cls(
            started_at=data.get("started_at") or _now_iso(),
            finished_at=data.get("finished_at"),
            iteration=int(data.get("iteration", 0)),
        )
        for raw in data.get("entries", []):
            kind = raw.get("kind")
            try:
                if kind == "thought":
                    pad.entries.append(Thought(**raw))
                elif kind == "action":
                    pad.entries.append(Action(**raw))
                elif kind == "observation":
                    pad.entries.append(Observation(**raw))
                elif kind == "critique":
                    pad.entries.append(Critique(**raw))
                elif kind == "provenance":
                    pad.entries.append(Provenance(**raw))
            except Exception:
                # Forwards-compat: ignore unknown / malformed entries.
                continue
        return pad


def _truncate_dict(d: dict, max_chars: int = 200) -> str:
    """Compact one-line repr of a params dict, truncated for prompt frugality."""
    try:
        s = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(d)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"
