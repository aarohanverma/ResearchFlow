# Research Assistant — Architecture Reference

> **Status: implemented** (May 2026)  
> This document describes the Research Assistant as it is **currently built and running**. It is a technical reference, not a design plan. For the original design rationale and future milestone ideas, see the git history.

---

## 1. What is the Research Assistant?

The Research Assistant (RA) is a persistent, agentic research workspace — not a chatbot. A **session** is a long-running investigation that the user returns to over days or weeks. Messages, branches, background tool-execution steps, generated artifacts, and session-attached documents all live under one durable database object.

The RA is the product surface that unifies all other ResearchFlow capabilities. It treats Deep Search, Genie synthesis, Study Mode, graph traversal, podcast/slides generation, bookmarks Q&A, and external data sources as a typed **tool registry** — and routes user queries through an LLM planner that picks the right tools per turn.

---

## 2. Source Layout

| Path | What it contains |
|---|---|
| `backend/app/assistant/` | Orchestrator, planner, synthesizer, events bus, scheduler, recovery, step cache, session metadata, interest updater |
| `backend/app/assistant/tools/` | 30+ registered `AssistantTool` implementations + registry + namespace packs |
| `backend/app/services/research_assistant.py` | Session/message/task bookkeeping and JobStore integration |
| `backend/app/models/assistant.py` | All ORM models: session, message, task, step, attachment, artifact |
| `backend/app/repositories/assistant.py` | DB access layer (only layer issuing SQL for RA tables) |
| `backend/app/api/v1/assistant.py` | FastAPI router — sessions, messages, steps, artifacts, attachments, SSE, arXiv import |
| `backend/app/services/arxiv_import.py` | ArxivImportService — search + import pipeline used by the RA |
| `frontend/app/(app)/assistant/page.tsx` | Three-pane workspace UI |

---

## 3. Database Schema

Six tables, all additive (no existing tables altered):

### `assistant_sessions`
One per investigation. Branchable via `parent_session_id`. Carries namespace, topic keys, orientation, expertise level, a rolling JSONB `state` (includes lazy history summary and session memory), and a lifecycle `status` (active/archived).

Branch depth is capped at 3 levels. Branching copies the session's namespace/orientation/expertise to the child and prepends up to 6 parent messages into the child's context window for continuity.

### `assistant_messages`
Ordered conversation. Roles: `user`, `assistant`, `system`. Assistant messages carry:
- `content` — the final synthesized prose
- `payload.blocks` — list of typed render blocks (see §8)
- `payload.workflow.actions` — high-level action labels (backward-compat)
- `citations[]` — paper IDs cited in this message
- `artifact_refs[]` — links to artifacts produced this turn

### `assistant_tasks`
One per submitted turn. Lifecycle: `pending → running → completed | failed | cancelled`. Tracks progress percentage, stage label, and links to the assistant message being populated. Exposed in the notification panel via `/assistant/jobs`.

### `assistant_steps`
One per tool call inside a turn. This is the unit of:
- **Resumability** — completed step rows let a restarted worker skip already-done work
- **Cancellation** — checked between waves; each tool checks `ctx.should_cancel()`
- **Reasoning-tree rendering** — the frontend can display full input params, output, cost, and latency per step

Key columns: `tool_name`, `step_index`, `status`, `input_params`, `output`, `cost`, `error`, `started_at`, `completed_at`.

### `assistant_attachments`
Session-scoped user-supplied context. Kinds: `note`, `url`, `paper_ref`, `pdf`, `image`, `file`. Parsed on upload:
- PDF → platform parser chain (Marker → Docling → Gemini Vision)
- Image → Gemini Vision caption + OCR
- DOCX → python-docx (XML fallback)
- Text/code → UTF-8 decode
- Parse failures never fail the upload — `metadata.parse_error` surfaces the hint

Attachments are **never** mixed into the global paper feed unless explicitly imported via `arxiv_import`.

### `assistant_artifacts`
Polymorphic registry of outputs produced inside a session. Kinds: `study_summary`, `idea_capsule`, `podcast`, `slides`, `mermaid`, `comparison`. Carries a `ref_id` pointing into the canonical table (paper ID, capsule ID, file path), a `preview` JSONB for inline card rendering, and `producing_step_id`.

---

## 4. Turn Lifecycle

Each turn is keyed by a `job_id` and executed by `Orchestrator.run_turn(job_id)` as an `asyncio.create_task` submitted by the scheduler.

```
1. Load context     — session, rolling history (with lazy LLM summarization), memory, 
                      user profile, disabled tools (e.g. wolfram_alpha when no key)
2. Off-topic guard  — regex patterns reject clearly non-research queries immediately
3. Query rewrite    — LLM-rewrites short/pronoun-heavy queries for better retrieval
4. Plan             — LLMPlanner.aplan() → Plan{steps, rationale, actions, trace}
5. Execute steps    — parallel waves + sequential steps; per-step AssistantStep rows
6. Coverage guard   — auto-imports arXiv papers when corpus < 2 results and no domain tool ran
7. Synthesize       — synthesize_answer() streams tokens via SSE event bus
8. Build blocks     — build_message_blocks() produces the typed block list
9. Finalize         — update AssistantMessage.payload.blocks, AssistantTask.status=completed
10. Fire-and-forget — session title/summary refresh + UserInterestProfile update (non-blocking)
```

---

## 5. Tool Registry

Tools are registered via `register_tool(tool)` in `backend/app/assistant/tools/__init__.py`. The registry is loaded at startup when `import app.assistant.tools` is executed in `research_assistant.py`.

### `AssistantTool` protocol (base.py)

```python
class AssistantTool(Protocol):
    name: str
    summary: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    cost_class: Literal["cheap", "moderate", "heavy"]
    side_effects: bool
    cancellable: bool
    streamable: bool

    async def run(self, ctx: ToolContext, params: BaseModel) -> ToolResult: ...
```

`ToolContext` carries session ID, user ID, namespace, topic keys, orientation, expertise level, job ID, DB session factory, `should_cancel()` coroutine, and `emit_progress()` coroutine.

### Registered tools

The registry currently ships **39 tools** (see `backend/app/assistant/tools/__init__.py`).
Two intentionally-omitted tools are documented inline in that file:

- `semantic_scholar` — disabled because the public endpoint is frequently rate-limited
  and produces unreliable results. OpenAlex (via `research_trends`) covers
  citation/trend queries without rate-limit issues.
- `author_network` — disabled because OpenAlex `/authors?search=` returns HTTP 400
  on compound research-topic queries; the planner kept picking it for
  author-discovery tasks where it always failed.

**Core retrieval:** `deep_search`, `arxiv_import`, `arxiv_search`, `paper_import`, `frontier_scan`

`paper_import` is the manual-import button as a tool — it takes a list of
arXiv IDs/URLs (max 10), runs the full ingestion pipeline, and flags each
paper with `is_manually_imported=True`. Use it when a citation tool or the
user supplied specific IDs; use `arxiv_import` for search-and-import.

**Synthesis & writing:** `concept_explain`, `compare_papers`, `genie_synthesize`, `genie_deep_dive`, `genie_read`, `paper_qa`, `bookmarks_query`, `literature_survey`, `draft_section`

**Graph:** `graph_query`, `graph_build` (the planner is forbidden from picking
`graph_build` directly — it remains registered for operator visibility via
`GET /assistant/tools`).

**Web / encyclopedic / bibliographic:** `web_search`, `wikipedia`, `crossref`, `unpaywall`, `citation_finder`, `research_trends`, `papers_with_code`

**Domain-specific (namespace-gated):** `pubmed`, `inspire_hep`, `nasa_ads`, `fred`, `nvd_cve`, `clinicaltrials`, `github_search`, `huggingface_search`, `wolfram_alpha`, `oeis`, `latex_parse`

**Session / utility:** `memory_write`, `memory_recall`, `memory_delete`, `parse_context`, `media_generate`

### Namespace packs

`backend/app/assistant/tools/namespace_packs.py` maps arXiv namespaces to tool sets. Global tools appear in every namespace. Domain packs overlay additional tools per namespace family:

| Namespace family | Extra tools |
|---|---|
| `q-bio`, `q-bio.*` | `pubmed`, `clinicaltrials` |
| `econ`, `q-fin` | `fred` |
| `astro-ph`, `astro-ph.*` | `nasa_ads` |
| `hep-*`, `gr-qc`, `nucl-*` | `inspire_hep` |
| `math`, `math.*`, `stat`, `stat.*` | `oeis`, `latex_parse` |
| `cs.CR`, `cs.NI`, `cs.SY` | `nvd_cve` |
| `cs.*` | `github_search`, `huggingface_search` |

### `wolfram_alpha` availability

The `wolfram_alpha` tool is silently disabled when neither `WOLFRAM_ALPHA_APP_ID` (direct HTTP API) nor `WOLFRAM_MCP_COMMAND` (MCP server subprocess) is configured. The orchestrator checks this at context-load time and adds the tool to `disabled_tools`, which the planner sees and respects.

---

## 6. Planner

### LLMPlanner (`planner_llm.py`)

Primary planner. Calls the quality-tier LLM (adaptive to reasoning-tier for complex queries) with a structured JSON output contract. Inputs to the LLM:

- Namespace key + topic keys
- Conversation history (rolling window, up to 14 messages)
- Session memory (medium + namespace-level)
- Tool catalogue (name + summary + JSON schema only — no implementation code)
- Orientation, expertise level
- Disabled tools set
- Query complexity classification (`simple` / `medium` / `complex`)

Output contract:
```json
{
  "rationale": "...",
  "steps": [{"tool": "deep_search", "title": "...", "params": {...}, "parallel": false}],
  "actions": ["Searching your corpus", "Synthesizing answer"],
  "trace": ["..."]
}
```

### HeuristicPlanner (`planner.py`)

Fallback — deterministic, keyword-driven, zero LLM cost. Activated when the LLM is unavailable, returns malformed JSON, or proposes invalid steps. The orchestrator always gets a valid `Plan`.

### Adaptive model routing

`_assess_query_complexity(query, history)` classifies queries as `simple`, `medium`, or `complex`. Complex queries (multi-part, multi-domain, vague) route to the reasoning-tier LLM for planning. Simple/medium use the quality tier to keep latency low. The synthesizer receives the same label and adjusts output depth and verbosity.

---

## 7. Orchestrator — Execution Engine

### Parallel vs sequential steps

Steps are grouped into "waves". Steps with `parallel=True` run concurrently via `asyncio.gather`. Sequential steps run one at a time. Order in the plan is preserved left-to-right; waves alternate as the planner specifies.

### Guardrails

| Guardrail | Limit | Behaviour when triggered |
|---|---|---|
| Max steps per turn | 12 | Plan clipped; warning logged |
| Per-step timeout | 180 s | Step marked `failed`; orchestrator continues |
| Consecutive empty waves | 3 | Remaining steps skipped; synthesizer runs on what was collected |
| Duplicate step dedup | — | Same `tool + params` signature in one plan → only first executes |
| Cancel gate | — | Checked after every wave → `CancelledError` → graceful partial result |

### Step-level result caching

Pure (no-side-effect) tools are eligible for caching. `StepCache.make_key(tool_name, params, user_id, namespace_key)` is checked before running; hits write a `completed` step row and skip execution. TTLs are tool-declared (e.g., `deep_search` → 1 h, `arxiv_search` → 1 h, `paper_qa` → permanent keyed on prompt version).

### Dependency injection

`_inject_dependencies(planned, results)` wires step outputs into downstream steps. Currently: `deep_search` paper IDs are injected into a queued `genie_synthesize` step so Genie operates on the retrieval results without requiring the planner to hard-code the linkage.

### Coverage guard

After the plan executes, if the combined paper corpus contains fewer than 2 results **and** no domain-specific tool produced native coverage, `_coverage_import()` auto-runs `arxiv_import` with the original query. This ensures even an empty namespace produces grounded context.

---

## 8. Synthesizer and Block Rendering

`synthesize_answer()` in `assistant/synthesizer.py` receives all step results and streams the final answer via `on_delta(chunk)` callbacks. It adapts:
- **Verbosity** to query complexity
- **Vocabulary density** to expertise level
- **Emphasis** to orientation (research vs production)

`build_message_blocks()` produces the typed block list stored in `payload.blocks`. Block kinds:

| Block kind | Rendered as |
|---|---|
| `text` | Markdown prose with inline citation chips |
| `paper_grid` | N-column sortable paper card grid |
| `arxiv_results` | arXiv search results with import buttons |
| `domain_papers` | PubMed / NASA ADS / INSPIRE HEP / etc. results |
| `comparison_table` | Column-per-paper, row-per-dimension structured comparison |
| `mermaid` | Rendered Mermaid diagram |
| `web_results` | External search results (low-trust label) |
| `genie_link` | Card linking to a created Genie idea capsule |
| `graph_summary` | Knowledge graph traversal summary |
| `nvd_results` | CVE vulnerability cards |
| `fred_series` | Macroeconomic data series |
| `trials_results` | ClinicalTrials.gov study cards |
| `code_results` | GitHub repo + HuggingFace model cards |
| `bookmarks_answer` | RAG answer grounded in bookmarked corpus |
| `import_summary` | N papers imported from arXiv |

---

## 9. SSE Streaming — Event Bus

`backend/app/assistant/events.py` implements an in-process pub/sub bus keyed by `job_id`. Subscribers receive buffered events (plan, started steps, progress) plus all subsequent events until the bus closes.

`GET /api/v1/assistant/tasks/{job_id}/stream` relays events as Server-Sent Events with 15-second heartbeats to prevent proxy timeouts.

### Event types

| Kind | When | Key payload |
|---|---|---|
| `plan_committed` | After planner returns | `rationale`, `steps[]`, `actions[]` |
| `step_started` | Before each tool | `step_id`, `tool`, `title` |
| `step_progress` | Mid-tool updates | `step_id`, `percent`, `summary` |
| `step_completed` | After tool finishes | `step_id`, `cache_hit` |
| `step_failed` | After tool errors | `step_id`, `error`, `retryable` |
| `message_delta` | Each synthesizer token | `message_id`, `delta` |
| `message_completed` | After synthesis writes to DB | `message_id`, `citation_count`, `blocks[]` |
| `task_completed` | Turn fully done | `summary` |
| `task_failed` | Unhandled orchestrator error | `error` |
| `task_cancelled` | User cancelled | `summary` |

---

## 10. Recovery — Crash-Safe Restart

`backend/app/assistant/recovery.reconcile_orphans()` is called in the startup lifespan. It scans `AssistantTask` rows with `status ∈ {pending, running}` and:

- **Resumes** recent tasks that have completed step rows — re-submits to the scheduler; orchestrator skips done steps
- **Fails** stale tasks (past recency threshold) — marks `failed` with "process restarted" so UI never shows a permanent spinner
- **Cancels** tasks with `cancel_requested_at` set

Recovery failures are caught silently — a bad DB state never blocks startup.

---

## 11. Rolling History Compression

When a session exceeds 14 messages, the orchestrator compresses older turns into a ≤600-word LLM summary, stored in `session.state["history_summary"]`. The summary is keyed by `cutoff_index` so it is regenerated only when new messages fall out of the verbatim window — no redundant LLM calls per turn.

The verbatim window is the last 10 messages. Branch sessions prepend up to 6 parent messages before the verbatim window to give context on what the branch departed from.

---

## 12. Session Memory

`session.state["memory"]` is a lightweight key-value dict written by the `memory` tool and read back into planner context on every turn. Used for persistent user facts ("focus on transformers", "exclude RLHF papers"). Not a separate DB table — stored as JSONB inside the session's state column.

`session.state["ns_memory"]` is a namespace-level memory layer inherited from parent sessions on branching.

---

## 13. Interest Profile Updates

After each turn, `assistant/interest_updater.update_from_turn()` runs as a fire-and-forget task. It folds the concepts from cited and retrieved papers into `UserInterestProfile.concept_affinity` — a float-valued dict of concept → affinity score. Subsequent `deep_search` and `frontier_scan` runs use this profile to bias retrieval toward the user's evolving interests.

---

## 14. Frontend Workspace (`assistant/page.tsx`)

Three-pane collapsible layout:
- **Left rail** — session list (title, namespace, last activity, branch indicator, running-task indicator). Actions: new session, rename, archive, clear-all, branch-from-message.
- **Center** — conversation + reasoning tree. Each assistant message shows its block-rendered content plus a collapsible step list (step name, tool, status, ETA). Suggestion chips are clickable to submit follow-up turns.
- **Right rail** — active context: namespace, topic keys, active task progress, attachments.

The page subscribes to the SSE stream immediately after turn submission and also polls `GET /sessions/{id}` as a fallback. Partial results from cancelled turns are displayed as-is.

File uploads (drag-and-drop or clip) are sent to `POST /sessions/{id}/attachments/upload` and displayed as attachment chips in the input area.

---

## 15. API Surface

```
# Session lifecycle
GET    /assistant/sessions                       → list (optionally filtered by namespace)
POST   /assistant/sessions                       → create (201)
GET    /assistant/sessions/{id}                  → session + messages + tasks
DELETE /assistant/sessions/{id}                  → soft-archive (204)
PATCH  /assistant/sessions/{id}/title            → rename
POST   /assistant/sessions/clear                 → archive all active sessions
POST   /assistant/sessions/{id}/branch           → create child branch (201)
GET    /assistant/sessions/{id}/export           → markdown or JSON download

# Turn submission and streaming
POST   /assistant/sessions/{id}/messages         → submit turn (202, queues job)
GET    /assistant/tasks/{job_id}/stream          → SSE AssistantEvent stream
POST   /assistant/tasks/{job_id}/cancel          → cancel running turn
GET    /assistant/tasks                          → list user tasks
GET    /assistant/jobs                           → notification-panel view

# Reasoning tree
GET    /assistant/sessions/{id}/steps            → all steps in session
GET    /assistant/messages/{msg_id}/steps        → steps for one assistant message

# Artifacts
GET    /assistant/sessions/{id}/artifacts        → generated outputs

# Attachments
GET    /assistant/sessions/{id}/attachments      → list
POST   /assistant/sessions/{id}/attachments      → create (note/url/paper_ref)
POST   /assistant/sessions/{id}/attachments/upload  → file upload (25 MB cap)
DELETE /assistant/sessions/{id}/attachments/{id} → delete (204)

# arXiv
POST   /assistant/arxiv/search                   → search without importing
POST   /assistant/arxiv/import                   → import selected papers

# Introspection
GET    /assistant/tools                          → registered tool catalogue (schema-only)
GET    /assistant/seeds                          → 4 seed questions for the empty state
```

---

## 16. Known Limitations

- **In-process scheduler only.** The scheduler is `asyncio.create_task` within the FastAPI process. Multi-worker deployments do not share the in-process job queue — use `CACHE_BACKEND=redis` so the JobStore is shared, but each worker independently executes the tasks it submitted. A Redis-backed worker queue (Arq, Celery) would be needed for true multi-worker task routing. The scheduler is now idempotent on double-submit: `submit(job_id)` returns the existing task instead of starting a parallel runner.
- **SSE event bus is in-process.** If the browser connects to a different backend replica than the one running the task, it will not receive step-level SSE events. The polling fallback (`GET /sessions/{id}`) is always available. Channels for turns that finish before any subscriber connects are auto-evicted on `close()` so the bus does not leak memory.
- **Attachment embeddings not yet wired.** Uploaded file text is stored in `assistant_attachments.content` and injected as plain-text context for the synthesizer. A session-scoped vector index for attachment-level semantic retrieval is not yet implemented.
- **No `paper_study` tool.** Study Mode is reachable directly via `/study/{paper_id}` but is not wrapped as an `AssistantTool` — the synthesizer cannot drive a Study Mode walkthrough from inside an RA turn.
- **Branch storage is copy-by-reference at the message level.** Branch semantics are loose: the child session inherits context from the parent's last 6 messages but does not have a hard FK cascade that prevents DELETE-parent-breaks-child scenarios.
- **Voyage embedding adapter is not shipped.** Setting `DEFAULT_EMBEDDING_PROVIDER=voyage` falls back to OpenAI at runtime with a warning. The Literal type still accepts the value for forward compatibility.
