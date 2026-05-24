# Research Assistant — Architecture Reference

> **Status: implemented** (May 2026)
> This document describes the Research Assistant (RA) as it is **currently built and running**. It is a technical reference, not a design plan. For the original design rationale and earlier milestone ideas, see the git history.

---

## 1. What is the Research Assistant?

The Research Assistant (RA) is a persistent, agentic research workspace — not a chatbot. A **session** is a long-running investigation that the user returns to over days or weeks. Messages, branches, background tool-execution steps, generated artifacts, and session-attached documents all live under one durable database object.

The RA is the product surface that unifies all other ResearchFlow capabilities. It treats Deep Search, Genie synthesis, Study Mode, graph traversal, podcast/slides generation, bookmarks Q&A, and external data sources as a typed **tool registry** — and routes user queries through an LLM planner that picks the right tools per turn, with a **ReAct mid-turn loop** on deeper queries.

---

## 2. Source Layout

| Path | What it contains |
|---|---|
| `backend/app/assistant/` | Orchestrator, planner, synthesizer, events bus, scheduler, recovery, step cache, reflection, claim ledger, HITL inbox, intent / clarify / persona / prompt safety, branch context, memory consolidation, query strategy, repair drift, research brief, retrieval observability, scratchpad, semantic memory, session metadata, state lock, telemetry, tuning |
| `backend/app/assistant/react/` | ReAct loop data model — `state.py`, `middleware.py`, `investigation_plan.py`, `subagents.py`, `subagent_runner.py` |
| `backend/app/assistant/react/middlewares/` | Concrete middleware files — `param_preflight`, `tool_ban`, `hitl_gate`, `diminishing_returns`, `paper_ledger`, `observability_mw`, `critic_gate`, `contradiction_mw`, `full_paper_gate` |
| `backend/app/assistant/react_loop.py` | The loop driver itself — `ReactConfig`, `ReactOutcome`, `run_react_loop()` |
| `backend/app/assistant/tools/` | 41 registered `AssistantTool` implementations + registry + namespace packs |
| `backend/app/services/research_assistant.py` | Session/message/task bookkeeping and JobStore integration |
| `backend/app/services/arxiv_import.py` | ArxivImportService — search + import pipeline used by the RA |
| `backend/app/models/assistant.py` | All ORM models: session, message, task, step, attachment, artifact |
| `backend/app/repositories/assistant.py` | DB access layer (only layer issuing SQL for RA tables) |
| `backend/app/api/v1/assistant.py` | FastAPI router — sessions, messages, steps, artifacts, attachments, SSE, HITL ACK, arXiv import |
| `frontend/app/(app)/assistant/page.tsx` | Three-pane workspace UI |

---

## 3. Database Schema

Six tables, all additive (no existing tables altered):

### `assistant_sessions`
One per investigation. Branchable via `parent_session_id`. Carries namespace, topic keys, orientation, expertise level, a rolling JSONB `state` (includes lazy history summary, tiered memory, and various cached scratch values), and a lifecycle `status` (active/archived).

Branch depth is capped at 3 levels. Branching copies the session's namespace/orientation/expertise to the child and prepends up to 6 parent messages plus a compressed parent-context summary into the child's context window for continuity.

### `assistant_messages`
Ordered conversation. Roles: `user`, `assistant`, `system`. Assistant messages carry:
- `content` — the final synthesised prose
- `payload.blocks` — list of typed render blocks (see §8)
- `payload.workflow.actions` — high-level action labels (backward-compat)
- `payload.scratchpad` — the ReAct loop's reasoning trace (when present)
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
- PDF → platform parser chain (`PDF_PARSER` choice → Marker → Gemini Vision → minimal placeholder)
- Image → Gemini Vision caption + OCR
- DOCX → python-docx (XML fallback)
- Text/code → UTF-8 decode
- Parse failures never fail the upload — `metadata.parse_error` surfaces the hint

Attachments are **never** mixed into the global paper feed unless explicitly imported via `arxiv_import` / `paper_import`.

### `assistant_artifacts`
Polymorphic registry of outputs produced inside a session. Kinds: `study_summary`, `idea_capsule`, `podcast`, `slides`, `mermaid`, `comparison`. Carries a `ref_id` pointing into the canonical table (paper ID, capsule ID, file path), a `preview` JSONB for inline card rendering, and `producing_step_id`.

### `idea_capsules.originating_session_id`
Added to support the Genie "From Assistant" view. When `genie_synthesize` runs from inside an RA session it stamps the resulting capsule with the session id; the Genie page renders capsules with this stamp under an "🤖 From Assistant" badge that links back to the originating session.

---

## 4. Turn Lifecycle

Each turn is keyed by a `job_id` and executed by `Orchestrator.run_turn(job_id)` as an `asyncio.create_task` submitted by the scheduler.

```
1. Load context     — session, rolling history (with lazy LLM summarisation), tiered
                      memory, user profile, disabled tools (e.g. wolfram_alpha when no key)
2. Off-topic guard  — regex patterns reject clearly non-research queries immediately
3. Clarify gate     — emits an inline clarification request on highly ambiguous queries
4. Query rewrite    — LLM-rewrites short/pronoun-heavy queries for better retrieval
5. Plan             — LLMPlanner.aplan() → Plan{steps, rationale, actions, trace}
6. Execute steps    — parallel waves + sequential steps; one AssistantStep row per call
7. ReAct loop       — (deep tier only) THINK / ACT / OBSERVE cycle with middleware chain
8. Coverage guard   — auto-imports arXiv papers when corpus < 2 and no domain tool ran
9. Synthesize       — synthesize_answer() streams tokens via SSE event bus
10. Build blocks    — build_message_blocks() produces the typed block list
11. Finalize        — update AssistantMessage.payload.blocks, AssistantTask.status=completed
12. Fire-and-forget — session title/summary refresh, UserInterestProfile update,
                      auto-memory write, repair-drift sweep (all non-blocking)
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

### Registered tools (41 total)

Counted from `app.assistant.tools.__init__` at boot. The registry is the source of truth; the list below mirrors the live state at the time of writing. Two academically obvious tools are intentionally *not* registered:

- `semantic_scholar` — frequently rate-limited and unreliable; OpenAlex (via `research_trends`) covers citation/trend queries without rate-limit issues.
- `author_network` — OpenAlex `/authors?search=` returns HTTP 400 on compound research-topic queries; the planner kept picking it for author-discovery tasks where it always failed.

**Core retrieval** — `deep_search`, `arxiv_search`, `arxiv_import`, `paper_import`, `frontier_scan`

`paper_import` is the manual-import button as a tool: takes a list of arXiv IDs/URLs (max 10), runs the full ingestion pipeline, and flags each paper with `is_manually_imported=True`. Use it when a citation tool or the user supplied specific IDs; use `arxiv_import` for search-and-import.

**Synthesis & writing** — `concept_explain`, `compare_papers`, `genie_synthesize`, `genie_deep_dive`, `genie_read`, `genie_combine`, `paper_qa`, `bookmarks_query`, `literature_survey`, `draft_section`, `study_paper`

`graph_build` is registered but **planner-forbidden** — it remains in `/assistant/tools` for operator visibility; the planner_llm forbidden set keeps it out of generated plans (it's a multi-minute background job that has its own UI trigger).

**Graph** — `graph_query`, `graph_build`

**Web / encyclopedic / bibliographic** — `web_search`, `wikipedia`, `crossref`, `unpaywall`, `citation_finder`, `research_trends`, `papers_with_code`

**Domain-specific (namespace-gated)** — `pubmed`, `inspire_hep`, `nasa_ads`, `fred`, `nvd_cve`, `clinicaltrials`, `github_search`, `huggingface_search`, `wolfram_alpha`, `oeis`, `latex_parse`

**Session / utility** — `memory_write`, `memory_recall`, `memory_delete`, `parse_context`, `media_generate`

### Namespace packs

`backend/app/assistant/tools/namespace_packs.py` maps arXiv namespaces to tool sets. Global tools appear in every namespace. Domain packs overlay additional tools per namespace family:

| Namespace family | Extra tools |
|---|---|
| `q-bio`, `q-bio.*` | `pubmed`, `clinicaltrials` |
| `econ`, `q-fin` | `fred` |
| `astro-ph`, `astro-ph.*`, `physics` | `nasa_ads` (+ `inspire_hep` for `physics`) |
| `hep-*`, `gr-qc`, `nucl-*`, `quant-ph`, `math-ph` | `inspire_hep` (+ `nasa_ads` for `gr-qc`) |
| `math`, `math.*` | `oeis` |
| `cs.*` | `github_search`, `huggingface_search`, `papers_with_code`, `nvd_cve` |

`describe_for_planner(namespace_key)` returns only the tools visible for that namespace (global + matching pack).

### `wolfram_alpha` availability

The `wolfram_alpha` tool is silently disabled when neither `WOLFRAM_ALPHA_APP_ID` (direct HTTP API) nor `WOLFRAM_MCP_COMMAND` (MCP server subprocess) is configured, **and** the requesting user has no per-user `encrypted_wolfram_key` in `user_provider_settings`. The orchestrator checks all three at context-load time and adds the tool to `disabled_tools`, which the planner sees and respects.

---

## 6. Planner

### LLMPlanner (`planner_llm.py`)

Primary planner. Calls the quality-tier LLM (adaptive to reasoning-tier for complex queries) with a structured JSON output contract. Inputs to the LLM:

- Namespace key + topic keys
- Conversation history (rolling window, up to 14 messages)
- Session memory (chat / tree / namespace tiers)
- Tool catalogue (name + summary + JSON schema only — no implementation code)
- Orientation, expertise level
- Disabled tools set
- Query complexity classification (`simple` / `medium` / `complex`)
- Namespace-pack description so the planner knows which domain tools are in scope

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

`_assess_query_complexity(query, history)` classifies queries as `simple`, `medium`, or `complex`. Complex queries (multi-part, multi-domain, vague) route to the reasoning-tier LLM for planning. Simple/medium use the quality tier to keep latency low. The synthesizer receives the same label and adjusts output depth and verbosity. Depth hints prescribe minimum tool counts (2 for simple, 2–4 for medium, 4–6 for complex).

---

## 7. Orchestrator — Execution Engine

### Parallel vs sequential steps

Steps are grouped into "waves". Steps with `parallel=True` run concurrently via `asyncio.gather`. Sequential steps run one at a time. Order in the plan is preserved left-to-right; waves alternate as the planner specifies.

### Guardrails (orchestrator-level)

| Guardrail | Limit | Behaviour when triggered |
|---|---|---|
| Max steps per turn | **12** | Plan clipped; warning logged |
| Per-step timeout | **180 s** | Step marked `failed`; orchestrator continues |
| Consecutive empty waves | **3** | Remaining steps skipped; synthesizer runs on what was collected |
| Duplicate step dedup | — | Same `tool + params` signature in one plan → only first executes |
| Cancel gate | — | Checked after every wave → `CancelledError` → graceful partial result |

### Step-level result caching

Pure (no-side-effect) tools are eligible for caching. `StepCache.make_key(tool_name, params, user_id, namespace_key)` is checked before running; hits write a `completed` step row and skip execution. TTLs are tool-declared — typical values: deterministic tools (`wolfram_alpha`, `oeis`, `unpaywall`) 24 h, stable academic (`pubmed`, `inspire_hep`, `nasa_ads`, `crossref`) 2–4 h, semi-stable (`deep_search`, `arxiv_search`, `research_trends`, `fred`) 1–4 h.

### Dependency injection

`_inject_dependencies(planned, results)` wires step outputs into downstream steps. Currently: `deep_search` paper IDs are injected into a queued `genie_synthesize` step so Genie operates on the retrieval results without requiring the planner to hard-code the linkage.

### Coverage guard

After the plan executes, if the combined paper corpus contains fewer than 2 results **and** no domain-specific tool produced native coverage (`_has_domain_coverage()` checks every domain tool's output list), `_coverage_import()` auto-runs `arxiv_import` with the original query. This ensures even an empty namespace produces grounded context.

### Rolling history

Once a session exceeds 14 messages, the orchestrator compresses older turns into a ≤600-word LLM summary stored in `session.state["history_summary"]` (keyed by `cutoff_index`). The 10 most recent messages are always kept verbatim. The summary is regenerated **only** when new messages fall out of the verbatim window — never per turn. Branch sessions additionally prepend the compressed parent-chain context up to 6 messages before the verbatim window.

---

## 8. ReAct Mid-Turn Loop

On deep-tier turns, after the initial plan has executed, the orchestrator hands control to `run_react_loop()` (`app/assistant/react_loop.py`). The loop lets the model:

- **THINK** — write free-form reasoning to the scratchpad
- **ACT** — pick another tool from the registry, or `"finalize"`, `"critique"`, `"fanout"`, `"subagent"`, `"write_todos"`
- **OBSERVE** — see the structured summary of the tool's output

Loops until the model finalises, the iteration cap is hit, or the wall-clock deadline expires. **Always bounded.**

### ReAct config defaults (`react_loop.ReactConfig`)

| Field | Default | Meaning |
|---|---|---|
| `max_iterations` | 8 | Hard cap on THINK/ACT/OBSERVE cycles |
| `deadline_seconds` | 90.0 | Wall-clock budget |
| `_MIN_ITERS_BEFORE_FREE_FINALIZE` | 3 | A turn cannot finalize before iter 3 without a critique |
| `_MAX_FANOUT_BRANCHES` | 4 | Hard cap on parallel branches per fanout action |

`memory_write` and `memory_delete` are disallowed as ACT targets inside the loop — durable memory writes happen post-turn, not mid-turn, to avoid premature commits.

### Middleware chain

The loop wraps every iteration in a 9-step middleware chain composed by `default_chain_factory()` in `react/middlewares/__init__.py`. Order is load-bearing — earlier middlewares get first say on `before_tool`.

| # | Middleware | What it does |
|---|---|---|
| 1 | `ParamPreflight` | Strips placeholder values (`__to_fill__`, `<TODO>`, empty strings); auto-fills missing required fields from the user query / paper ledger |
| 2 | `ToolBan` | Blocks banned / failing tools, may redirect to a substitute |
| 3 | `HitlGate` | Pauses for user approval before `genie_synthesize` dispatches; emits `react_hitl_pending`, awaits an ACK future with a 10 s timeout (`_ACK_WINDOW_SEC`), then continues with a scratchpad note |
| 4 | `DiminishingReturns` | Skips identical-param redos; signals "no new evidence" when retrieval stops returning new IDs |
| 5 | `PaperLedger` | Accumulates paper IDs from every tool result for downstream auto-fill + ledger-aware prompts |
| 6 | `RetrievalObservability` | Records per-call coverage, dispersion, and rerank disagreement so the synth can downgrade confidence on thin retrievals |
| 7 | `CriticGate` | Forces one critique step before a too-early finalize |
| 8 | `ContradictionDetector` | Lexical + numeric + LLM-semantic; can force at most one counter-search on a high-confidence open signal |
| 9 | `FullPaperVerification` | At finalize, inspects every strong claim in the claim ledger and forces up to **2** `paper_qa` rounds (`_MAX_FORCED_PAPER_QA_PER_TURN = 2`) on those whose source was only the abstract/snippet; remaining strong claims without chunk-level evidence are labelled `unverifiable` so the synth caveats them |

Each middleware returns a `MiddlewareDecision` (`Allow` / `DispatchOverride` / `AbortDispatch` / `FinalizeForceAction` / `FinalizeForceCritique` / `FinalizeAllow`) so the loop driver can compose decisions deterministically.

### Strong-claim ledger (`claim_ledger.py`)

`detect_strong_spans()` matches numeric, SOTA, causal, and comparative regexes. Each detected span becomes a `StrongClaim` carrying its source provenance: `SOURCE_CHUNK` (`paper_qa` results) is verified; `SOURCE_ABSTRACT` / `SOURCE_SNIPPET` (retrieval / arXiv-search results) is provisional until the full-paper gate runs. The synthesiser reads the ledger and renders provisional/unverifiable claims with explicit `(abstract-only)` / `(unverifiable)` labels.

### HITL inbox (`hitl_inbox.py`)

Process-local async inbox:

- `register_pending(request_id, session_id, user_id)` → `(record, future)` — the gate awaits the future
- `resolve(request_id, session_id, user_id, decision)` — the API endpoint's body
- `peek` / `discard` — diagnostics + cleanup

> Multi-worker deployments would need a Redis-backed swap. The SSE listener (which surfaces the `react_hitl_pending` event) and the inbox writer must run in the same process.

The HITL ACK API endpoint is `POST /api/v1/assistant/sessions/{session_id}/hitl/{request_id}` with body `{status: "approve"|"skip"|"modify", params?, note?}`. Owner-validated via the inbox.

### Subagents

`subagent` action delegates a focused sub-task to a context-quarantined inner loop. Recursion depth is gated: a subagent can only run at `subagent_depth == 0` (the top-level loop), so subagents cannot spawn subagents. The decision prompt hides the subagent catalog at depth > 0 to remove the temptation entirely.

---

## 9. Synthesizer and Block Rendering

`synthesize_answer()` in `assistant/synthesizer.py` receives all step results (plus the ReAct loop's `claim_ledger`, `contradictions`, `retrieval_metrics`, and `investigation_plan` when present) and streams the final answer via `on_delta(chunk)` callbacks. Prompt structure:

- **DEPTH DISCIPLINE** block — calibrates verbosity to query complexity
- **EVIDENCE vs INFERENCE LABELLING** — forces explicit `Directly shown by [N]:`, `Reasonable inference from [N], [M]:`, `RA hypothesis (no direct citation):`, `Uncertain:` labels
- **PROVISIONAL CLAIMS** — reads the STRONG-CLAIM LEDGER and labels provisional / unverifiable claims as `(abstract-only)`; refuses to repeat contradicted claims without flag
- **COMPETING EXPLANATIONS** — ranks by evidence strength, plausibility, impact, generality, testability, production relevance — *not* by paper count
- **PRODUCTION-AWARENESS (gated)** — only emitted for applied / deployed-AI questions; pure-research questions get no production checklist

`build_message_blocks()` produces the typed block list stored in `payload.blocks`. Block kinds:

| Block kind | Rendered as |
|---|---|
| `text` | Markdown prose with inline citation chips |
| `paper_grid` | N-column sortable paper card grid |
| `arxiv_results` | arXiv search results with import buttons |
| `domain_papers` (`source_papers`) | PubMed / NASA ADS / INSPIRE HEP / etc. results |
| `comparison_table` | Column-per-paper, row-per-dimension structured comparison |
| `mermaid` | Rendered Mermaid diagram |
| `web_results` | External search results (low-trust label) |
| `genie_link` | Card linking to a created Genie idea capsule |
| `graph_summary` | Knowledge graph traversal summary |
| `nvd_results` | CVE vulnerability cards |
| `fred_data` | Macroeconomic data series |
| `trials_results` | ClinicalTrials.gov study cards |
| `code_results` | GitHub repo + HuggingFace model cards |
| `bookmarks_answer` | RAG answer grounded in bookmarked corpus |
| `import_summary` | N papers imported from arXiv |

---

## 10. SSE Streaming — Event Bus

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
| `react_thought` | Per ReAct THINK | `iteration`, `text` |
| `react_dispatch` | Before a ReAct ACT | `iteration`, `tool`, `params` |
| `react_done` | ReAct loop exits | `iterations`, `finalized`, `new_tools` |
| `react_hitl_pending` | HITL gate fires | `request_id`, `tool`, `params`, `ack_window_sec` |
| `react_hitl_resolved` | HITL gate resolved | `request_id`, `decision` |
| `react_hitl_timeout` | HITL gate timed out | `request_id` |
| `message_delta` | Each synth token | `message_id`, `delta` |
| `message_completed` | Synth writes to DB | `message_id`, `citation_count`, `blocks[]` |
| `task_completed` | Turn fully done | `summary` |
| `task_failed` | Unhandled error | `error` |
| `task_cancelled` | User cancelled | `summary` |

---

## 11. Recovery — Crash-Safe Restart

`backend/app/assistant/recovery.reconcile_orphans()` is called in the startup lifespan. It scans `AssistantTask` rows with `status ∈ {pending, running}` and:

- **Resumes** tasks younger than **2 hours** (`_RESUME_AGE_LIMIT`) — re-submits to the scheduler; the orchestrator skips already-completed steps via `_already_completed_steps(job_id)`
- **Fails** stale tasks (older than 2 hours) — marks `failed` with "Orphaned by process restart (too old to resume)"
- **Cancels** tasks with `cancel_requested_at` set

Operator opt-out: `ASSISTANT_AUTO_RESUME=0` or `DISABLE_AUTO_RECOVERY=1` skips resume entirely and marks every orphan failed. Useful during incident response when you don't want a server restart to chew tokens re-finishing prior turns.

Recovery failures are caught silently — a bad DB state never blocks startup.

---

## 12. Rolling History Compression

See Orchestrator §7.

---

## 13. Tiered Session Memory

- `session.state["chat_memory"]` — this chat only
- `root_session.state["tree_memory"]` — entire session tree (branches share)
- `session.state["ns_memory"]` — namespace-wide
- `session.state["history_summary"]` — compressed older messages (lazy)

Writers: the `memory` tool (`memory_write` / `memory_recall` / `memory_delete`) plus `auto_memory.py`'s fire-and-forget post-turn writer. Readers: every planner / synthesizer call.

`memory_consolidation.py` runs weekly (Sunday 04:30 UTC via APScheduler) to cluster and LLM-merge related entries across every user's tiers. Per-tier eviction caps keep stores bounded; consolidation keeps the *information* by collapsing related entries into a single rollup with provenance pointing back to the originals.

---

## 14. Interest Profile Updates

After each turn, `assistant/interest_updater.update_from_turn()` runs as a fire-and-forget task. It folds the concepts from cited and retrieved papers into `UserInterestProfile.concept_affinity` — a float-valued dict of concept → affinity score. Subsequent `deep_search` and `frontier_scan` runs use this profile to bias retrieval toward the user's evolving interests.

---

## 15. Frontend Workspace (`assistant/page.tsx`)

Three-pane collapsible layout:
- **Left rail** — session list (title, namespace, last activity, branch indicator, running-task indicator). Actions: new session, rename, archive, clear-all, branch-from-message.
- **Center** — conversation + reasoning tree. Each assistant message shows its block-rendered content plus a collapsible step list (step name, tool, status, ETA). The HITL ACK card surfaces approve / skip / modify buttons + a countdown when `react_hitl_pending` fires. Suggestion chips are clickable to submit follow-up turns.
- **Right rail** — active context: namespace, topic keys, active task progress, attachments.

The page subscribes to the SSE stream immediately after turn submission and also polls `GET /sessions/{id}` as a fallback. Partial results from cancelled turns are displayed as-is.

File uploads (drag-and-drop or clip) are sent to `POST /sessions/{id}/attachments/upload` and displayed as attachment chips in the input area. Cap: 25 MB per file.

---

## 16. API Surface

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

# HITL (human-in-the-loop)
POST   /assistant/sessions/{session_id}/hitl/{request_id}
        → body {status: approve|skip|modify, params?, note?} (owner-validated)

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

## 17. Known Limitations

- **In-process scheduler only.** The scheduler is `asyncio.create_task` within the FastAPI process. Multi-worker deployments do not share the in-process job queue — use `CACHE_BACKEND=redis` so the JobStore is shared, but each worker independently executes the tasks it submitted. A Redis-backed worker queue (Arq, Celery) would be needed for true multi-worker task routing. The scheduler is idempotent on double-submit: `submit(job_id)` returns the existing task instead of starting a parallel runner.
- **SSE event bus is in-process.** If the browser connects to a different backend replica than the one running the task, it will not receive step-level SSE events. The polling fallback (`GET /sessions/{id}`) is always available. Channels for turns that finish before any subscriber connects are auto-evicted on `close()` so the bus does not leak memory.
- **HITL inbox is process-local.** Same constraint as the SSE bus — multi-worker would need a Redis-backed swap.
- **Attachment embeddings not yet wired.** Uploaded file text is stored in `assistant_attachments.content` and injected as plain-text context for the synthesizer. A session-scoped vector index for attachment-level semantic retrieval is not yet implemented.
- **`study_paper` tool is shipped but minimal.** It returns a structured summary handle, not the full streaming Study Mode walkthrough — the synthesiser cannot drive a Study Mode page from inside an RA turn.
- **Branch storage is copy-by-reference at the message level.** Branch semantics are loose: the child session inherits context from the parent's last 6 messages plus the compressed parent-chain summary, but does not have a hard FK cascade that prevents DELETE-parent-breaks-child scenarios.
- **Voyage embedding adapter is not shipped.** Setting `DEFAULT_EMBEDDING_PROVIDER=voyage` falls back to OpenAI at runtime with a warning. The Literal type still accepts the value for forward compatibility.
- **`semantic_scholar` and `author_network` tools are not registered.** See §5.
