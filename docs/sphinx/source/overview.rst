Overview
========

ResearchFlow is a self-hosted AI research operating system. It ships:

* Six LangGraph ``StateGraph`` workflows (ingestion, study, RAG, podcast,
  slides, folder consolidation) plus two custom async-generator workflows
  for Genie synthesis and Deep Dive,
* A persistent agentic **Research Assistant** with a 41-tool registry,
  plan-then-execute orchestration, a ReAct mid-turn loop, and a 9-step
  middleware chain for adversarial discipline,
* PostgreSQL 16 + pgvector storage (single database — no separate graph
  store), and
* A Next.js 14 frontend with SSE-streamed UI.

Architecture Layers
-------------------

1. **Frontend** — Next.js 14, App Router, SSE stream consumer. Dev: ``next dev``  /  ``next dev --turbo``. Prod: ``next build && next start`` (pre-compiled, zero on-demand compilation).
2. **API** — FastAPI async routers with Pydantic v2 validation. Routers: ``auth``, ``assistant``, ``feed``, ``search``, ``papers``, ``study``, ``bookmarks``, ``graph``, ``chat``, ``genie``, ``settings``, ``generate``, ``dev``, ``admin`` (+ admin-only ``settings`` sub-router).
3. **Workflows** — LangGraph ``StateGraph`` pipelines (ingestion, study, RAG, podcast, slides, folder consolidation) plus custom async generators for Genie + Deep Dive.
4. **Research Assistant** — Persistent agentic workspace: planner → orchestrator → 41-tool registry → ReAct loop → middleware chain → synthesizer. Per-step checkpointing, SSE streaming, rolling history compression, HITL gate, full-paper claim verification.
5. **Adapters / Repositories** — Provider-agnostic interfaces; only layer touching the DB.
6. **Storage** — PostgreSQL 16 + pgvector. HNSW index on the 768-dim ``paper_chunks.embedding`` column (with idempotent index creation at startup); GIN index on a full-text concatenation of title, tldr, abstract, key_concepts, methods_used.

Key Design Principles
---------------------

* **No vendor lock-in** — every external service has a swappable adapter (LLM, embedding, blob, cache, TTS, slides, web search, PDF parser).
* **Repository pattern** — all SQL lives in ``app/repositories/``; workflows never import SQLAlchemy directly.
* **Prompt injection prevention** — all untrusted paper text is wrapped in ``[START]``/``[END]`` (or ``<<DATA_START>>``/``<<DATA_END>>``) delimiters; the system prompt explicitly instructs the model to treat it as DATA and ignore any embedded instructions.
* **Local → Azure swap** — change four environment variables (``DATABASE_URL``, ``CACHE_BACKEND``, ``BLOB_BACKEND``, ``ENVIRONMENT``), zero code changes.
* **SSE generator invariant** — ``yield`` must precede every potentially-failing ``await`` in async generators so the HTTP connection is always established before any exception can occur.

Workflows
---------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Workflow
     - Trigger
     - What it does
   * - Ingestion
     - ``0 5 * * 2-5`` (05:00 UTC Tue–Fri) / manual API
     - fetch_papers → enrich_papers (LLM) → embed_papers (vector) → update_graph → score_for_potd → mark_complete. Idempotent per (namespace, date) via a unique row in ``workflow_runs``.
   * - Study
     - On-demand SSE
     - check_cache → parse → structure → 12-section parallel generation → diagrams → cache. Keyed per ``(paper, expertise, orientation, prompt_version)``.
   * - RAG
     - On-demand SSE
     - Query rewrite → intent classify → vector + graph retrieve → rerank → self-RAG → synthesize. Orientation + expertise adapted.
   * - Genie
     - Manual / auto-batch / query
     - gather_context → find_bridges → check_viability → hypothesize → critique → elaborate → diagrams → poc_code → save. Three modes (Manual 2–10, Auto full-feed 2–5, Query NL→discover 2–5). Capsules tagged by source mode.
   * - Deep Dive
     - On-demand SSE / background
     - Single-pass: reasoning model generates the 11-section article grounded in source paper text, streams live, persists to ``idea_capsules.deep_dive_content``.
   * - Podcast
     - On-demand / background
     - 5-node StateGraph: load → plan → script → tts → save. Multi-speaker HOST/EXPERT script via OpenAI TTS. Resumable via LangGraph checkpointer.
   * - Slides
     - On-demand / background
     - 4-node StateGraph: load → plan → markdown → render → save. Marp-rendered HTML deck with falls-back-to-Markdown safety. Resumable via LangGraph checkpointer.
   * - Folder consolidation
     - On-demand
     - 3-node StateGraph: load → coherence-score → cross-paper synthesis. Used by the Bookmark Folder "Synthesise" action.

Graph Deep Build
----------------

``build_deep_graph`` in ``services/graph.py`` runs as a background job (``asyncio.create_task``)
capped at **2 concurrent namespace builds** via ``asyncio.Semaphore(2)`` to avoid exhausting
the LLM rate limit.  The build uses a 2-phase LLM taxonomy (Phase 1: canonical bounded structure;
Phase 2: paper assignment) and commits + invalidates the subgraph cache **after each area**
so partial progress appears in the graph incrementally rather than all at once at the end.

Capsule Namespace Scoping
-------------------------

``GET /genie/capsules`` accepts ``namespace_keys`` as a comma-separated query
parameter.  The endpoint batch-resolves seed elements to paper namespace keys and
filters out capsules whose every source paper is from a deselected subject.
Capsules with no resolvable paper namespace are always shown.  The frontend
passes the user's active topic subscriptions as ``namespace_keys`` on every
capsule fetch, so deselecting a subject in Settings immediately hides that
subject's ideas without any manual action.

Idea Q&A Chat Streaming
-----------------------

``POST /genie/capsules/{capsule_id}/chat`` streams Server-Sent Events shaped
as ``{type: "chunk", content: token}`` followed by ``{type: "done"}``.
Failures emit ``{type: "error", message: "..."}``.  The frontend Idea Q&A
panel parses these by ``p.type`` and ``p.content``.

Research Assistant
------------------

The Research Assistant (``app.assistant``) is a persistent, agentic research workspace where
each session is a long-running investigation backed by durable DB rows.

Turn lifecycle (orchestrator):

1. ``POST /assistant/sessions/{session_id}/messages`` creates an ``AssistantMessage`` and an
   ``AssistantTask`` (job_id = UUID), then queues the task via ``scheduler.submit(job_id)``.
2. The ``Orchestrator`` rehydrates session context (messages, history summary, user profile,
   memory tiers), runs the off-topic / clarification gates, and optionally rewrites the
   user's query for retrieval quality.
3. ``LLMPlanner`` selects tools from the registry's JSON-schema view and returns a ``Plan``
   (ordered list of ``PlannedStep``). ``HeuristicPlanner`` handles fallbacks.
4. The orchestrator executes steps, writing one ``AssistantStep`` row per tool call. Completed
   steps are skipped on replay (crash-safe resume).
5. After each parallel wave the ``StepCache`` is consulted; pure tools are cached by
   ``(tool, params_hash, user_id, namespace_key)`` with tool-specific TTLs.
6. **On deep-tier turns**, the ReAct mid-turn loop runs after the initial plan: the model
   reasons in a scratchpad, can dispatch additional tools (with fanout, subagents,
   critique), and a 9-step middleware chain enforces adversarial discipline (see below).
7. The ``Synthesizer`` grounds the final answer in step outputs and emits a ``blocks`` list
   of typed UI elements (paper_grid, comparison_table, mermaid, web_results,
   ``source_papers``, ``nvd_results``, ``fred_data``, ``trials_results``, ``code_results``, etc.).
8. All progress is published as ``AssistantEvent`` objects over the in-process SSE bus;
   the frontend subscribes via ``GET /assistant/tasks/{job_id}/stream``.

Guardrails (per turn):

- **Max 12 plan steps** — runaway planners clipped, warning logged
- **180 s per-step timeout** — slow steps marked failed, orchestrator continues
- **3 consecutive empty waves** → circuit-break and synthesize from available results
- **Duplicate tool+params** combinations are deduplicated before execution
- **Cancel gate** checked after every wave → ``CancelledError`` → graceful partial result

ReAct loop (deep tier only):

- **Max iterations**: 8 (``_DEFAULT_MAX_ITERATIONS``)
- **Wall-clock deadline**: 90 s (``_DEFAULT_DEADLINE_SECONDS``)
- **No free finalize** before iteration 3 — the model is forced through at least one critique
- **Fanout** up to 4 parallel branches per iteration
- **Subagents** with role prompts; recursion depth gated at 1

Middleware chain (executed in order on every iteration):

1. ``ParamPreflight`` — strip placeholder values, auto-fill missing required fields from the user query / paper ledger
2. ``ToolBan`` — block banned tools, redirect or abort
3. ``HitlGate`` — pause for user approval before ``genie_synthesize`` dispatches; 10 s ACK window, then continue with a scratchpad note
4. ``DiminishingReturns`` — skip identical-param redos; abort when retrieval stops returning new IDs
5. ``PaperLedger`` — accumulate paper IDs from every tool result for downstream auto-fill
6. ``RetrievalObservability`` — record per-call coverage, dispersion, and rerank disagreement
7. ``CriticGate`` — force one critique before too-early finalize
8. ``ContradictionDetector`` — lexical + numeric + LLM-semantic; forces at most one counter-search
9. ``FullPaperVerification`` — at finalize, inspect every strong claim in the ledger and force up to 2 ``paper_qa`` rounds on those whose source was only the abstract / snippet; remaining strong claims without chunk-level evidence are labelled ``unverifiable`` so the synth can caveat them

Recovery: ``reconcile_orphans()`` (called in FastAPI lifespan startup) sweeps
``running``/``pending`` tasks, resumes tasks younger than 2 hours, and fails the rest.
Disable per-deploy by setting ``ASSISTANT_AUTO_RESUME=0`` or ``DISABLE_AUTO_RECOVERY=1``.

Rolling history: once a session has more than 14 messages, the oldest turns are compressed
into a ≤600-word summary stored in ``session.state["history_summary"]``. The 10 most recent
messages are always kept verbatim for context. The summary is keyed by ``cutoff_index`` so
it is regenerated only when new messages fall out of the verbatim window.

Namespace packs: domain-specific tool overlays — PubMed + ClinicalTrials for ``q-bio.*``,
FRED for ``econ`` / ``q-fin``, NASA ADS for ``astro-ph``, INSPIRE HEP for ``hep-*``,
NVD CVE for ``cs.CR`` / ``cs.NI`` / ``cs.SY``, GitHub + HuggingFace + Papers with Code for
``cs.*``, OEIS + LaTeX-parse for ``math`` / ``stat``. Each pack overlays its tools on top of
the global set visible to the planner for that namespace.

Background jobs
---------------

The APScheduler instance attached to the FastAPI lifespan registers six recurring jobs:

* ``ingestion_nightly`` — defaults to ``0 5 * * 2-5`` (configurable via ``INGESTION_CRON``); runs ingestion for every configured namespace, gated on the ``arxiv_ingest_enabled`` feature flag.
* ``clustering_weekly`` — placeholder (HDBSCAN clustering is post-MVP); schedule ``CLUSTERING_CRON`` default ``0 5 * * 0``.
* ``cross_namespace_weekly`` — placeholder (cross-namespace bridge edges are post-MVP); schedule ``CROSS_NAMESPACE_CRON`` default ``30 5 * * 0``.
* ``bookmark_index_rebuild_weekly`` — every Sunday at 03:00 UTC, re-embeds any bookmarked papers missing an abstract chunk.
* ``memory_consolidation_weekly`` — every Sunday at 04:30 UTC, clusters + LLM-merges related session-memory entries across every user's chat / tree / namespace tiers.
* ``checkpoint_cleanup_monthly`` — day 1 of each month at 04:00 UTC, deletes ``langgraph_checkpoints`` threads older than 30 days to prevent unbounded growth of the checkpoint tables.

Token Usage Tracking
--------------------

A ``TrackingLLMAdapter`` decorator (``app/adapters/llm/tracking.py``) wraps
every concrete LLM adapter returned by ``get_llm_adapter()``.  After each
``complete``, ``complete_structured``, ``stream``, or ``complete_with_tools``
call, it fires an ``asyncio.create_task`` that records a row to the
``token_usage`` table — provider, model, input tokens, output tokens, latency,
estimated USD cost.  Recording is fire-and-forget and cannot break LLM calls.

Per-request attribution uses three ``ContextVar``s in ``app/core/tracking.py``:

* ``current_user_id`` — set by the auth dependency at request entry.
* ``current_workflow`` / ``current_node`` — set at the top of each workflow
  entry (``run_study``, ``run_genie``, ``run_rag``, ``run_deep_dive``,
  ``run_ingestion``, ``_run_deep_search``) via ``set_workflow_context()``.

``asyncio.create_task`` copies the current context onto the spawned task so
recording sees the right user/workflow even when the LLM call has already
returned.

The ``GET /api/v1/settings/token-usage`` endpoint returns aggregated usage
for the authenticated user — totals, per-day, per-workflow, and per-model
breakdowns.  It accepts ``?from=YYYY-MM-DD&to=YYYY-MM-DD`` and defaults to
today (UTC) when neither bound is supplied.

Streaming token counts are estimated from text length (~4 chars/token) since
provider streaming APIs do not consistently expose final usage.  Non-streaming
``complete()`` paths use the exact counts returned by the provider.
