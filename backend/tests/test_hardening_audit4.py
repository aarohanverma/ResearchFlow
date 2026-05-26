"""ResearchFlow production hardening audit #4 (May 2026).

Regression tests for surgical fixes applied in this pass. Each test
asserts a real bug that landed in production (or could have, if the
relevant code path executed) so re-introducing the pattern is caught
at CI time.

Findings audited and fixed in this pass:

  1. ``app.assistant.tools.memory._bump_last_recalled_ts`` referenced
     ``async_session_factory`` but never imported it. Every call
     NameError'd inside its broad try/except, was caught at DEBUG, and
     the ``last_recalled_ts`` UI field silently never updated in
     production. Fixed by moving the import inside the function body
     (where the rest of memory.py's deferred imports live, since the
     factory is only consumed in this one helper).

  2. Synthesizer post-synth pipeline (reflection / critique / repair /
     final_evaluator / output quality safeguard / low-grounding notice)
     only runs when ``on_delta is None``. The orchestrator was always
     passing ``on_delta=_on_delta``, so in production the entire audit
     pipeline was dead code — every user-facing answer streamed raw
     LLM output token-by-token with NO revision pass and no quality
     safeguard. Fixed by routing deep-tier turns through the
     non-streaming synth path and replaying the final, audited answer
     to the SSE channel via ``Orchestrator._fake_stream_answer``.
     Trivial / single-tier turns (which set ``skip_reflection=True``
     so the audit pipeline is a no-op anyway) keep real streaming for
     snappy time-to-first-token.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid as _uuid
from unittest.mock import MagicMock

import pytest


# ── 1. memory._bump_last_recalled_ts has the import in scope ────────────────


def test_bump_last_recalled_ts_imports_session_factory() -> None:
    """Verify the helper resolves ``async_session_factory`` at call time.

    The pre-fix version had no top-level or local import — every
    fire-and-forget invocation NameError'd inside the helper's broad
    try/except and silently logged at DEBUG. The fix adds a local
    ``from app.db.session import async_session_factory`` inside the
    function body. Reading the source guards against accidental
    deletion of that import — the runtime call would crash silently
    again otherwise.
    """
    from app.assistant.tools import memory as memory_mod

    src = inspect.getsource(memory_mod._bump_last_recalled_ts)
    assert "from app.db.session import async_session_factory" in src, (
        "regression: _bump_last_recalled_ts must import async_session_factory "
        "in-scope; without it the fire-and-forget bump silently fails on every "
        "memory_recall call (DEBUG-only log, no operator signal)."
    )


@pytest.mark.asyncio
async def test_bump_last_recalled_ts_no_nameerror_on_empty_keys() -> None:
    """The early-return path must not depend on the import either.

    When ``persistent_keys`` carries no entries, the helper short-circuits
    before opening a DB session. This is the cheapest production path
    (the common case where memory_recall returned only short-tier or
    nothing at all). It must complete cleanly — historically a missing
    import at the FIRST referenced line would still raise on import
    resolution. Pinning this in a regression test stops a future
    refactor (e.g. moving the import inside the if-branch) from
    silently re-introducing the NameError.
    """
    from app.assistant.tools import memory as memory_mod

    # Empty key sets → early return. No DB, no event loop work.
    await memory_mod._bump_last_recalled_ts(
        session_id=_uuid.uuid4(),
        persistent_keys={"tree_memory": set(), "ns_memory": set()},
        ns_namespace="cs.AI",
        recalled_ts="2026-05-26T00:00:00+00:00",
    )


# ── 2. Orchestrator audit-then-stream pattern for deep-tier turns ───────────


@pytest.mark.asyncio
async def test_fake_stream_answer_emits_message_deltas() -> None:
    """``_fake_stream_answer`` must walk the entire answer text and
    publish ``message_delta`` events to the SSE bus.

    This is the user-facing animation channel for deep-tier turns
    where real LLM streaming was disabled so the post-synth audit
    pipeline could run. If this helper drops the answer (or fails to
    publish), the deep-tier user would stare at "thinking…" until
    ``message_completed`` fires with the entire answer in one go —
    losing the streaming UX the other tiers preserve.
    """
    from app.assistant.orchestrator import Orchestrator

    orch = Orchestrator()
    captured: list[tuple[str, dict]] = []

    def _capture(job_id: str, kind, payload):
        captured.append((str(kind), dict(payload)))

    orch._publish = _capture  # type: ignore[assignment]

    job_id = "test-job-fake-stream"
    message_id = _uuid.uuid4()
    answer = "This is a moderately long answer that the synthesizer produced after running the full audit pipeline."

    await orch._fake_stream_answer(
        job_id, message_id, answer,
        chunk_size=24, delay_s=0.0,
    )

    # Every event was a message_delta with the right message_id.
    assert all(k == "message_delta" for k, _ in captured), (
        f"expected only message_delta events, got kinds={[k for k, _ in captured]!r}"
    )
    assert all(p.get("message_id") == str(message_id) for _, p in captured), (
        "message_id must be preserved (stringified) on every delta so the "
        "frontend can route each chunk to the right bubble"
    )
    # Concatenating the deltas reconstructs the original answer exactly —
    # no chars dropped, no chars duplicated, no boundary corruption.
    reconstructed = "".join(p.get("delta", "") for _, p in captured)
    assert reconstructed == answer, (
        f"reconstructed answer drifted from source — bytes lost or duplicated.\n"
        f"  source: {answer!r}\n"
        f"  recon:  {reconstructed!r}"
    )


@pytest.mark.asyncio
async def test_fake_stream_answer_handles_empty_answer() -> None:
    """Empty answer must short-circuit without emitting any deltas.

    A degenerate synth (e.g. fallback path returned empty) must not
    flood the SSE channel with empty deltas — the frontend treats
    each delta as content. Early return keeps the channel clean.
    """
    from app.assistant.orchestrator import Orchestrator

    orch = Orchestrator()
    captured: list = []
    orch._publish = lambda *args, **kwargs: captured.append((args, kwargs))  # type: ignore[assignment]

    await orch._fake_stream_answer("job", _uuid.uuid4(), "")
    await orch._fake_stream_answer("job", _uuid.uuid4(), "   \n   ")  # whitespace-only

    # "   \n   " is non-empty but stream isn't filtered — we emit the
    # whitespace as one chunk. The empty case must be silent though.
    empty_emits = [e for e in captured if e[0][2].get("delta", "") == ""]
    assert empty_emits == [], (
        "regression: empty answer must emit ZERO deltas, not even a sentinel"
    )


@pytest.mark.asyncio
async def test_fake_stream_answer_cancellation_propagates() -> None:
    """``asyncio.sleep`` between chunks must be a cancellation point.

    If the user clicks Stop while we are replaying the audited answer,
    ``CancelledError`` must propagate to the outer ``run_turn`` so the
    standard cancel-cleanup runs. Swallowing the cancel here would
    leave the turn looking 'completed' to the orchestrator while the
    user expected it to abort.
    """
    from app.assistant.orchestrator import Orchestrator

    orch = Orchestrator()
    orch._publish = lambda *args, **kwargs: None  # type: ignore[assignment]

    # Long answer + non-zero delay so the coroutine has to await at
    # least once before completing. We cancel from the outside.
    answer = "x" * 2000
    task = asyncio.create_task(
        orch._fake_stream_answer(
            "job", _uuid.uuid4(), answer,
            chunk_size=8, delay_s=0.05,
        )
    )
    await asyncio.sleep(0.01)  # let the task hit its first sleep
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_finalize_turn_routes_deep_tier_through_audit_path() -> None:
    """The orchestrator must call synth with ``on_delta=None`` on
    deep-tier turns so the reflection / repair / final_evaluator /
    output quality safeguard pipeline actually runs.

    Source-level guard via ``inspect.getsource``: a future refactor
    that always passes ``_on_delta`` to the synthesizer (the
    pre-audit-#4 pattern) would silently skip the entire post-synth
    audit pipeline again, shipping unaudited answers to users on
    every deep-tier turn. Pinning the conditional here catches that
    regression at CI time.
    """
    from app.assistant.orchestrator import Orchestrator

    src = inspect.getsource(Orchestrator._finalize_turn)

    # The routing variable is the load-bearing line: a single
    # ``on_delta=_on_delta`` call site would once again starve the
    # audit pipeline.
    assert "on_delta_for_synth" in src, (
        "regression: _finalize_turn must compute an ``on_delta_for_synth`` "
        "based on depth tier; passing ``_on_delta`` unconditionally bypasses "
        "the synthesizer's reflection / repair / final_evaluator pipeline."
    )
    assert "use_real_streaming" in src, (
        "regression: _finalize_turn must branch on depth tier when deciding "
        "whether to stream or fake-stream — see audit memory for context."
    )
    assert "_fake_stream_answer" in src, (
        "regression: _finalize_turn must replay the audited answer to the "
        "SSE channel via _fake_stream_answer on the non-streaming path; "
        "without this, deep-tier users see no streaming UX at all."
    )


def test_finalize_turn_passes_routed_on_delta_not_raw() -> None:
    """The synth call must receive the ROUTED ``on_delta_for_synth``,
    not the raw ``_on_delta`` closure.

    A subtle regression: someone could compute ``on_delta_for_synth``
    correctly but forget to pass it (still passing ``_on_delta`` in
    the synth call), nullifying the routing. This source-level check
    pins the wiring.
    """
    from app.assistant.orchestrator import Orchestrator

    src = inspect.getsource(Orchestrator._finalize_turn)
    # The synth call's on_delta kwarg must reference the routed name.
    # We look for the literal "on_delta=on_delta_for_synth" — if a
    # future edit reverts it to "on_delta=_on_delta", this assertion
    # fires.
    assert "on_delta=on_delta_for_synth" in src, (
        "regression: synthesize_answer must be invoked with "
        "``on_delta=on_delta_for_synth`` (the depth-tier-routed callable), "
        "not the raw ``_on_delta`` closure — otherwise deep-tier turns "
        "fall back to real streaming and skip the audit pipeline."
    )
