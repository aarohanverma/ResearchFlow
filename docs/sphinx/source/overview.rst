Overview
========

ResearchFlow is a self-hosted AI research assistant with five LangGraph state-machine
workflows, PostgreSQL + pgvector storage, and a Next.js 14 frontend.

Architecture Layers
-------------------

1. **Frontend** — Next.js 14, App Router, SSE stream consumer. Dev: ``next dev --turbo`` (Turbopack). Prod: ``next build && next start`` (pre-compiled, zero on-demand compilation).
2. **API** — FastAPI async routers with Pydantic v2 validation
3. **Workflows** — LangGraph ``StateGraph`` pipelines (Ingestion, Study, RAG, Genie, Deep Dive)
4. **Adapters / Repositories** — Provider-agnostic interfaces; only layer touching the DB
5. **Storage** — PostgreSQL 16 + pgvector (IVFFlat index, 768-dim Gemini embeddings)

Key Design Principles
---------------------

* **No vendor lock-in** — every external service has a swappable adapter (LLM, embedding, blob, cache).
* **Repository pattern** — all SQL lives in ``app/repositories/``; workflows never import SQLAlchemy directly.
* **Prompt injection prevention** — all untrusted paper text is wrapped in ``<<DATA_START>>…<<DATA_END>>`` delimiters.
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
     - Nightly cron / manual API
     - Fetch → enrich (LLM) → embed (vector) → graph update → score PoTD
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
     - Two-phase: quality model draft → reasoning model judge → persist

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
