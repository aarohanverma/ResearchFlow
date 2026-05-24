"""Full-paper verification middleware — force ``paper_qa`` on strong claims.

The user-stated hard requirement: RA must not lean on abstracts when
making strong claims. Abstracts are fine for triage; *final
conclusions* and *synthesis* must be grounded in the paper's body
where possible, and clearly marked as provisional when full-paper
verification is unavailable.

This middleware implements the gate. On every ``after_tool``:

* Scan the result for strong-claim spans
  (:func:`app.assistant.claim_ledger.extract_claims_from_result`).
* Add detected claims to :class:`state.claim_ledger`. ``paper_qa``
  outputs are tagged ``SOURCE_CHUNK`` and land as ``verified`` — they
  came from the paper body already.

On ``gate_finalize``:

* If the ledger has any unverified strong claims sourced from
  abstracts / snippets AND we have iteration budget remaining,
  return :class:`FinalizeForceAction` with a ``paper_qa`` call
  targeting the highest-priority unverified claim. The loop
  dispatches one verification, re-enters, and the loop iterates
  through the claims one at a time.
* When all unverified strong claims have been re-checked OR we hit
  the per-turn forced cap, the gate yields and finalize proceeds.
  Any remaining unverified claims are marked ``unverifiable`` so
  the synthesizer's agent_notes block can label them.

Priority for which claim gets verified first:

1. Causal / SOTA superlatives — most likely to be over-claims.
2. Numeric performance — quantitative, easy to fact-check.
3. Comparative claims — moderate priority.

The detector tags each claim as exactly one of these (via the
patterns in :mod:`app.assistant.claim_ledger`); we sort by the
priority order before picking the next to verify.
"""

from __future__ import annotations

import logging
from typing import Any

from app.assistant.claim_ledger import (
    StrongClaim,
    extract_claims_from_result,
    resolve_paper_qa_verdict,
)
from app.assistant.react.middleware import (
    FinalizeAllow,
    FinalizeForceAction,
    FinalizeGate,
)
from app.assistant.react.middlewares.base import BaseMiddleware
from app.assistant.tools.base import ToolResult
from app.assistant.tools.registry import get_tool

log = logging.getLogger(__name__)


# How many forced paper_qa rounds we allow at finalize before
# yielding. Each round costs one tool dispatch (LLM-backed paper_qa
# synthesis) so we bound this to keep tail latency manageable. Two
# rounds covers the common case: usually only one or two strong
# claims survive un-verified to finalize time, since the model
# already calls paper_qa proactively when it sees abstract-sourced
# claims it wants to lean on.
_MAX_FORCED_PAPER_QA_PER_TURN = 2

# Wait until at least one retrieval has populated the paper ledger
# before scanning for strong claims. Below this threshold the
# detector has nothing to bind claims to (no paper IDs in scope).
_MIN_LEDGER_FOR_SCAN = 1


def _priority(claim: StrongClaim) -> int:
    """Lower = higher priority. Causal / SOTA spans get the most
    aggressive verification because they're the most over-claimed
    in practice; numeric comparisons land in the middle; pure
    comparatives last."""
    span = (claim.span or "").lower()
    if any(kw in span for kw in (
        "state-of-the-art", "sota", "first to ", "causes ", "drives ", "leads to ",
        "outperforms all", "surpasses all",
    )):
        return 0
    if any(kw in span for kw in ("%", "accuracy", "f1", "bleu", "speedup", "latency")):
        return 1
    return 2


class FullPaperVerificationMiddleware(BaseMiddleware):
    """Force ``paper_qa`` verification on strong abstract-only claims."""

    name = "full_paper_gate"

    async def after_tool(
        self,
        state: Any,
        action: str,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None:
        # ── First: resolve any in-flight verification we triggered ──
        # When this middleware forces a paper_qa round at finalize, it
        # marks the target claim as ``in_flight`` and records the
        # target on ``state._fpg_inflight`` so this hook can find it
        # without parsing the question string. Resolving here is the
        # only place the verdict moves off ``in_flight`` — without
        # this the gate would re-flag the same claim on every
        # subsequent finalize until the per-turn budget expired and
        # the answer would always mass-label otherwise-verified claims
        # as ``unverifiable``.
        if action == "paper_qa":
            try:
                self._resolve_inflight_after_paper_qa(state, params, result)
            except Exception as exc:  # noqa: BLE001 — resolver must never abort the loop
                log.debug("full_paper_gate: in-flight resolver failed: %s", exc)

        try:
            new_claims = extract_claims_from_result(
                action=action, result=result, iteration=state.iteration_count,
            )
        except Exception as exc:  # noqa: BLE001 — detector must never abort the loop
            log.debug("full_paper_gate: claim extraction failed: %s", exc)
            return

        if not new_claims:
            return

        added = 0
        for claim in new_claims:
            if state.claim_ledger.add(claim):
                added += 1
        if added:
            state.pad.think(
                f"Full-paper gate: detected {added} new strong claim(s) "
                f"from {action} (total tracked: {len(state.claim_ledger.by_key)})."
            )

    def _resolve_inflight_after_paper_qa(
        self,
        state: Any,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None:
        """Apply the paper_qa verdict to the in-flight target claim.

        Looks up the target claim by ``(paper_id, span_head)`` keyed on
        ``state._fpg_inflight``; that mapping is written by
        :meth:`gate_finalize` immediately before dispatching the forced
        round. When no in-flight target matches the paper_id, this is a
        model-initiated paper_qa call (not gate-forced) and we do
        nothing here — the regular extraction path still mines spans
        from the answer for the ledger.
        """
        inflight: dict[str, str] = getattr(state, "_fpg_inflight", None) or {}
        if not inflight:
            return
        out = (getattr(result, "output", None) or {})
        target_paper_id = str(out.get("paper_id") or params.get("paper_id") or "")
        if not target_paper_id or target_paper_id not in inflight:
            return
        span_head = inflight.pop(target_paper_id)
        # Re-key into the ledger: ``(paper_id, hash(span_head))``
        # mirrors ClaimLedger._key. Use the ledger helper instead of
        # reaching into private state.
        target = state.claim_ledger.find_pending(target_paper_id, span_head)
        if target is None:
            # The claim slot is gone (deduplicated away, or never made
            # it past the add()) — nothing to update.
            return

        answer = (out.get("answer") or "")
        if out.get("found") is False or not answer:
            target.verdict = "unverifiable"
            target.verification_note = (
                "paper_qa could not resolve the paper / chunk index — "
                "claim left unverifiable"
            )
            return

        verdict, note = resolve_paper_qa_verdict(answer)
        target.verdict = verdict
        target.verification_note = note
        target.verified_at_iteration = state.iteration_count
        state.pad.think(
            f"Full-paper gate: paper_qa returned for forced verification — "
            f"claim={target.span[:160]!r} → {verdict.upper()} ({note[:120]})"
        )

    async def gate_finalize(self, state: Any) -> FinalizeGate:
        """At finalize, force one paper_qa round per unverified strong
        claim until we exhaust the per-turn budget or the iteration
        cap.

        Returns :class:`FinalizeAllow` when:
          * no unverified claims remain,
          * the per-turn forced_paper_qa budget is exhausted,
          * no iteration budget remains for a verification round,
          * the ``paper_qa`` tool isn't registered.
        """
        if state.is_last_iteration:
            # Honour the global iteration cap above the gate — the
            # user's "loop must converge" guarantee wins.
            return self._yield_marking_unverifiable(state, reason="iteration_cap")
        if state.forced_paper_qa >= _MAX_FORCED_PAPER_QA_PER_TURN:
            return self._yield_marking_unverifiable(state, reason="per_turn_cap")
        if state.iterations_remaining() < 1:
            return self._yield_marking_unverifiable(state, reason="no_budget")

        # paper_qa is gated by tool registration / namespace pack. If
        # it isn't visible, we can't enforce the requirement — flag
        # all unverified claims so the synth caveats them.
        if get_tool("paper_qa") is None:
            return self._yield_marking_unverifiable(state, reason="paper_qa_missing")

        pending = state.claim_ledger.unverified()
        if not pending:
            return FinalizeAllow()

        # Highest-priority claim first.
        pending.sort(key=_priority)
        target = pending[0]

        state.forced_paper_qa += 1
        state.pad.think(
            f"Full-paper gate: forcing paper_qa on strong claim before finalize. "
            f"claim={target.span[:200]!r} paper={target.paper_id[:12]} "
            f"source={target.source_field} (round {state.forced_paper_qa} "
            f"of {_MAX_FORCED_PAPER_QA_PER_TURN})."
        )

        # Transition the claim to ``in_flight`` so ``needs_verification``
        # excludes it from the unverified() pool — without this the
        # next gate_finalize pass would pick the same claim again and
        # burn the per-turn budget on a single span. The after_tool
        # hook resolves the verdict (verified / contradicted /
        # unverifiable) once paper_qa returns; if the loop ends before
        # that happens (cancel, deadline), the ``in_flight`` verdict
        # is surfaced as still-provisional by ``ClaimLedger.summarize``
        # so the synthesizer hedges on the claim rather than treating
        # it as verified.
        target.verdict = "in_flight"
        target.verification_note = "paper_qa forced at finalize"
        target.verified_at_iteration = state.iteration_count

        # Stash the in-flight mapping so ``after_tool`` can locate the
        # exact target claim once paper_qa returns. Keyed by paper_id
        # because that's the unambiguous join field on the result;
        # value is the span's first 80 chars (the same head the ledger
        # dedups on), letting find_pending re-derive the ledger key.
        # A dict on state survives middleware re-entry; a per-claim
        # attribute would be lost when other middlewares mutate the
        # ledger between iterations.
        inflight: dict[str, str] = getattr(state, "_fpg_inflight", None) or {}
        inflight[str(target.paper_id)] = target.span or ""
        state._fpg_inflight = inflight

        # Build a focused question — the loop will call paper_qa with
        # the claim as the question, so the LLM can answer "does the
        # paper support this exact statement?". Trim aggressively so
        # the synthesizer prompt inside paper_qa doesn't bloat.
        question = (
            "Does the paper's full text actually support this claim? "
            "Quote the relevant passages or state clearly if the claim "
            "is NOT supported by the paper body.\n\nClaim: "
            + target.span[:380]
        )
        return FinalizeForceAction(
            action="paper_qa",
            params={
                "question": question,
                "paper_id": target.paper_id,
            },
            reason="full_paper_verification",
            rationale=(
                "Strong claim sourced from "
                f"{target.source_field}; verifying against full-paper chunks "
                "before allowing the answer to ship."
            ),
        )

    def _yield_marking_unverifiable(self, state: Any, *, reason: str) -> FinalizeGate:
        """Allow finalize, but stamp every remaining unverified claim
        as ``unverifiable`` so the synth can label them as
        provisional / abstract-only.

        The user's spec: "If full-paper verification is unavailable,
        RA should clearly mark the conclusion as provisional." This
        is where that bookkeeping happens.

        Also collapses any ``in_flight`` claim whose paper_qa never
        returned (worker crash, cancel mid-dispatch) to
        ``unverifiable`` so the ledger summary doesn't leave a claim
        stuck in transition — the synthesizer would otherwise hedge on
        the claim with no audit string explaining why.
        """
        flagged = 0
        for claim in state.claim_ledger.unverified():
            claim.verdict = "unverifiable"
            claim.verification_note = (
                f"full-paper verification skipped at finalize ({reason})"
            )
            flagged += 1
        # Sweep stranded in-flight claims — these had paper_qa
        # dispatched but no after_tool resolution arrived (cancel,
        # error mid-dispatch). Treat them the same as never-verified.
        for claim in state.claim_ledger.by_key.values():
            if claim.verdict == "in_flight":
                claim.verdict = "unverifiable"
                claim.verification_note = (
                    f"paper_qa dispatched but did not resolve ({reason})"
                )
                flagged += 1
        if flagged:
            state.pad.think(
                f"Full-paper gate: yielding ({reason}) — {flagged} strong claim(s) "
                "left unverified and flagged as provisional for the synthesizer."
            )
        return FinalizeAllow()
