Overview
========

ResearchFlow is a self-hosted AI research assistant with five LangGraph state-machine
workflows, PostgreSQL + pgvector storage, and a Next.js 14 frontend.

Architecture Layers
-------------------

1. **Frontend** — Next.js 14, App Router, SSE stream consumer. Dev: ``next dev --turbo`` (Turbopack). Prod: ``next build && next start`` (pre-compiled, zero on-demand compilation).
2. **API** — FastAPI async routers with Pydantic v2 validation
3. **Workflows** — LangGraph ``StateGraph`` pipelines (Ingestion, Study, RAG, Genie, Deep Dive)
4. **Research Assistant** — Persistent agentic workspace: LLM planner → orchestrator → 30+ tool registry → synthesizer. Per-step checkpointing, SSE streaming, rolling history compression.
5. **Adapters / Repositories** — Provider-agnostic interfaces; only layer touching the DB
6. **Storage** — PostgreSQL 16 + pgvector (IVFFlat index, 768-dim embeddings)

Key Design Principles
---------------------

* **No vendor lock-in** — every external service has a swappable adapter (LLM, embedding, blob, cache).
* **Repository pattern** — all SQL lives in ``app/repositories/``; workflows never import SQLAlchemy directly.
* **Prompt injection prevention** — all untrusted paper text is wrapped in ``[START]``/``[END]`` delimiters and the system prompt explicitly instructs the model to treat it as DATA and ignore any embedded instructions.
* **Local → Azure swap** — change four environment variables, zero code changes.
* **SSE generator invariant** — ``yield`` must precede every potentially-failing ``await`` in async generators so the HTTP connection is always established before any exception can occur.

Workflows
---------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Workflow
     - Trigger
     - What it does
   * - Ingestion
     - ``0 5 * * 2-5`` (05:00 UTC Tue–Fri) / manual API
     - fetch_papers → store_papers → enrich_papers (LLM) → embed_papers (vector) → update_graph → score_for_potd → mark_complete. Idempotent per day via WorkflowRun. LangGraph checkpointed.
   * - Study
     - On-demand SSE
     - PDF parse → structure extract → explain (3 levels, orientation-aware) → cache per ``(paper, expertise, orientation, prompt_version)``
   * - RAG
     - On-demand SSE
     - Query rewrite → intent → vector+graph retrieve → rerank → self-RAG → synthesize (orientation + expertise adapted)
   * - Genie
     - Manual / auto-batch / query
     - Context gather → bridge discovery → hypothesize → critique → elaborate → save. Three modes: Manual (2–10), Auto (full feed, 2–5), Query (NL → paper discovery, 2–5). Capsules tagged by source mode.
   * - Deep Dive
     - On-demand SSE / background
     - Single-pass: reasoning model generates full 11-section article grounded in source paper text, streams live, persists to ``idea_capsules.deep_dive_content``

Graph Deep Build
----------------

``build_deep_graph`` in ``services/graph.py`` runs as a background job (``asyncio.create_task``)
capped at **2 concurrent namespace builds** via ``asyncio.Semaphore(2)`` to avoid exhausting
the LLM rate limit.  The build uses a 2-phase LLM taxonomy (Phase 1: canonical bounded structure;
Phase 2: paper assignment) and commits + invalidates the subgraph cache **after each area**
so partial progress appears in the graph incrementally rather than all at once at the end.

For full technical detail see :doc:`/api/workflows.ingestion` and siblings.

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
panel parses these by ``p.type`` and ``p.content`` so chunks are no longer
silently dropped.

Concept-Node Deduplication (Frontend)
-------------------------------------

The deep-build LLM can emit inconsistent label casing across runs (e.g.
"Task-Specific Assistants" vs "task-specific assistants").  Because
``get_or_create_node`` matches exactly on ``(label, node_type, namespace_key)``,
each casing variant becomes a separate DB row with its own children — appearing
as floating duplicate cluster nodes in the graph.

The frontend helper ``dedupeConceptNodes`` runs after every graph fetch.  It
groups CONCEPT nodes by ``(lowercase label, namespace_key)``, picks a canonical
node per group (proper-case preferred), redirects all edges from aliases to the
canonical, and drops self-loops and duplicate edges.  This is a purely visual
fix — the database is untouched.  A permanent cleanup requires the user to run
**Clear All** followed by **Build Deep**.

Research Assistant
------------------

The Research Assistant (``app.assistant``) is a persistent, agentic research workspace where
each session is a long-running investigation backed by durable DB rows.

Turn lifecycle:

1. ``POST /assistant/sessions/{session_id}/messages`` creates an ``AssistantMessage`` and an
   ``AssistantTask`` (job_id = UUID), then queues the task via ``scheduler.submit(job_id)``.
2. The ``Orchestrator`` rehydrates session context (messages, history summary, user profile)
   and optionally rewrites the user's query for better retrieval quality.
3. ``LLMPlanner`` selects tools from the registry's JSON-schema view and returns a ``Plan``
   (ordered list of ``PlannedStep``). ``HeuristicPlanner`` handles fallbacks.
4. The orchestrator executes steps, writing one ``AssistantStep`` row per tool call. Completed
   steps are skipped on replay (crash-safe resume).
5. After each parallel wave the ``StepCache`` is consulted; pure tools are cached by
   ``(tool, params_hash, user_id, namespace_key)`` with tool-specific TTLs.
6. The ``Synthesizer`` grounds the final answer in step outputs and emits a ``blocks`` list
   of typed UI elements (paper_grid, comparison_table, mermaid, web_results, etc.).
7. All progress is published as ``AssistantEvent`` objects over the in-process SSE bus;
   the frontend subscribes via ``GET /assistant/tasks/{job_id}/stream``.

Guardrails (per turn):

- Maximum 12 plan steps
- 180 s per-step timeout
- 3 consecutive empty execution waves → circuit-break and synthesize from available results
- Duplicate tool+params combinations are deduplicated before execution

Recovery: ``reconcile_orphans()`` (called in FastAPI lifespan startup) sweeps
``running``/``pending`` tasks, resumes tasks younger than 2 hours, and fails the rest.

Rolling history: once a session has more than 14 messages, the oldest turns are compressed
into a ≤600-word summary stored in ``session.state["history_summary"]``.  The 10 most recent
messages are always kept verbatim for context.

Namespace packs: domain-specific tool overlays — PubMed for ``q-bio``, FRED for ``econ``,
NASA ADS for ``astro-ph``, INSPIRE HEP for ``hep-*``, ClinicalTrials for clinical queries.
Each pack adds its tools to the global set visible to the planner for sessions in that namespace.

Token Usage Tracking
--------------------

A ``TrackingLLMAdapter`` decorator (``app/adapters/llm/tracking.py``) wraps
every concrete LLM adapter returned by ``get_llm_adapter()``.  After each
``complete``, ``complete_structured``, ``stream``, or ``complete_with_tools``
call, it fires an ``asyncio.create_task`` that records a row to the existing
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
today (UTC) when neither bound is supplied.  The Settings → Token Usage tab
in the frontend visualises this data with quick presets and a per-day bar
chart.

Streaming token counts are estimated from text length (~4 chars/token) since
provider streaming APIs do not consistently expose final usage.  Non-streaming
``complete()`` paths use the exact counts returned by the provider.
