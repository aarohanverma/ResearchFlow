# ResearchFlow State Ownership

Canonical reference for which data lives where, who owns it, and how it
should be cached / scoped. Used by feature work, migrations, and audit
to keep multi-user isolation correct as the system scales.

State is split into four classes. Each row in the system belongs to
exactly one class. Mixing classes inside a single table is permitted
only when documented here.

---

## 1. Private (per user)

User-owned, isolated by ``user_id``, must never leak across users.
All API endpoints that read/write these tables MUST enforce
``user_id == current_user``.

| Table / store | Notes |
| --- | --- |
| ``users`` | The user record itself. |
| ``user_provider_settings`` | API key overrides, encrypted. |
| ``user_interest_profile`` | Concept affinity / hot-cold subtopics. |
| ``assistant_sessions`` | Chat session metadata + ``state`` JSONB. |
| ``assistant_messages`` | Chat messages (FK to session). |
| ``assistant_tasks`` | Background orchestration tasks. |
| ``assistant_steps`` | Per-tool-call audit rows. |
| ``assistant_artifacts`` | RA-owned artifact references. |
| ``assistant_attachments`` | User-uploaded notes / URLs / files attached to a session. |
| ``bookmarks`` + ``bookmark_folders`` + ``bookmark_folder_members`` | Bookmarks and their folders. |
| ``feed_feedback`` | Per-user like / dismiss signals. |
| ``paper_namespace_hide`` | Per-user dismissals from feed. |
| ``query_logs`` | Per-user search history. |
| ``idea_capsules`` | Genie idea capsules (owner cascade). |
| ``genie_sessions`` | Genie synthesis jobs (owner). |
| ``token_usage`` | Per-user LLM token accounting (billing). |
| **Session JSONB state** under ``assistant_sessions.state``: |
| &nbsp;&nbsp;``chat_memory`` | Per-chat memory (short tier). |
| &nbsp;&nbsp;``tree_memory`` (root only) | Per-tree memory (medium tier). |
| &nbsp;&nbsp;``ns_memory`` | Per-namespace memory (long tier). |
| &nbsp;&nbsp;``branch_summaries`` | Rolled-up child-branch context. |
| &nbsp;&nbsp;``branch_seed_summary`` | Parent-context seed for a branch. |
| &nbsp;&nbsp;``history_summary`` | Cached rolling-history digest. |
| &nbsp;&nbsp;``memory_embeddings`` | Lazy-cached embeddings for semantic recall. |
| &nbsp;&nbsp;``turn_telemetry`` | Per-turn outcome ring. |
| &nbsp;&nbsp;``title_user_edited`` / ``auto_title`` | Title-edit ownership. |

**Frontend localStorage state** (browser-scoped, not currently persisted server-side):

- Highlight marks per RA session / Study Mode paper / Idea Dive.
- Sticky notes per RA session.
- RA session-panel width.
- Main sidebar collapsed flag.
- Token Usage view preferences.

These are deliberately browser-local — they are UI ephemera, not
research artifacts. Migration to backend storage is a future option,
gated on the user explicitly wanting cross-device sync.

---

## 2. Shared (global, deterministic, content-keyed)

Data derived from public sources. No ``user_id``. Cached by content
key so two users asking for the same canonical output reuse the same
row. APIs read these without auth scoping; write paths run under the
ingestion / generation pipelines.

| Table / store | Notes |
| --- | --- |
| ``papers`` | Paper metadata. Keyed by ``(external_id, namespace_key)``. |
| ``paper_chunks`` | Embedded chunks per paper. |
| ``paper_citations`` | Citation graph edges. |
| ``paper_of_day`` | Per-namespace daily highlight. |
| ``summaries`` | Cached Study Mode output. Keyed by ``(paper_id, expertise_level)``. |
| ``graph_nodes`` / ``graph_edges`` | Knowledge graph (when not user-scoped). |
| ``source_mappings`` | Namespace → arXiv-category map. |
| ``shared_generation_outputs`` (new) | Deterministic media output dedup. |

The "deterministic" promise: given identical inputs (source_id +
expertise_level + orientation + prompt_hash + model_id + parser_version),
the output is reusable across users. Versioning columns guard against
silent staleness when prompts or models change.

---

## 3. Derived / personalized

User-scoped views over shared data. The shared substrate stays in §2
tables; the personalisation lives in §1 tables that REFERENCE the
shared rows.

| Concept | Substrate (shared) | Personalisation (private) |
| --- | --- | --- |
| Feed scoring | ``papers`` | ``user_interest_profile``, ``feed_feedback``, ``paper_namespace_hide`` |
| Bookmarked papers | ``papers`` | ``bookmarks`` |
| Study session | ``summaries`` | ``assistant_sessions``, ``assistant_attachments`` |
| Generated media | ``shared_generation_outputs`` | ``generated_artifacts`` (owner + reference FK) |
| Search results | ``paper_chunks`` indices | per-request scoring nudge from orientation |

---

## 4. System / global config

Static or admin-managed. Not user-keyed, not content-keyed.

- Namespace tree (in-code constant).
- Tool registry (in-code).
- LLM provider configuration (from environment).
- Source mappings (configurable).
- Background job store (Redis when configured, else in-process).

---

## API auth contract

- Every endpoint operating on §1 data MUST inject ``CurrentUserID`` and
  filter by it. Verified by the auth audit in
  ``backend/tests/test_security.py``.
- Endpoints reading §2 data may be unauthenticated when the data is
  inherently public (paper metadata, namespace catalog) — these are
  documented in this file and audited.
- ``/dev/reset`` and any other destructive admin endpoint MUST require
  an explicit feature flag (``ALLOW_DEV_RESET=1``) and a valid user
  session, regardless of environment.

---

## Concurrency contract for JSONB state

Multiple concurrent turns in the same session tree can race on
``assistant_sessions.state`` mutations. Affected keys:

- ``branch_summaries``: parents updated by sibling branches.
- ``turn_telemetry``: appended per turn.
- ``memory_embeddings``: appended on semantic recall.
- ``tree_memory``: written from any branch in the tree.

The ``app.assistant.state_lock`` helper provides a two-layer lock for
read-modify-write on ``assistant_sessions.state``:

1. **In-process** — an ``asyncio.Lock`` keyed by session_id. Always
   active. Serialises concurrent coroutines inside one Python worker.
2. **Cross-process** — a PostgreSQL session-scoped advisory lock
   (``pg_advisory_lock(bigint)``) keyed by a 64-bit hash of the
   session_id. Engaged automatically when ``ENVIRONMENT != local``,
   or forced via ``STATE_LOCK_PG_ADVISORY=1|0``. Active across every
   worker process touching the same DB.

Together these make the local → cloud transition seamless: single-
worker local dev pays only the in-process cost; multi-worker cloud
deployments automatically get cross-process serialisation without code
changes. The advisory lock has a bounded ``lock_timeout`` (30 s) so a
stuck holder surfaces as a logged warning rather than a deadlock.

Cache-key conventions (intra-process and across the cluster):

- **User-scoped step cache** — keys include ``user_id`` so private-state
  tools (bookmarks_query, memory) never share results.
- **Shared step cache** — listed in
  ``app.assistant.step_cache._SHARED_SOURCE_TOOLS``: arXiv search,
  Wikipedia, Wolfram, Semantic Scholar, etc. Keys drop the user
  segment so all users reuse a single cached fetch.
- **Deep-search query cache** — keyed by ``(ns_hash, q_hash)``: shared.
- **Deep-search job poll cache** (``ds_job:{job_id}``) — stamped with
  ``user_id`` at submit time and enforced at poll: private.
- **Graph build job cache** — operates on the shared knowledge graph;
  authed user may observe/cancel any build (shared infrastructure).
