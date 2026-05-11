# ResearchFlow

**Your personal research operating system.**  
An AI-native platform that ingests arXiv papers nightly, builds a living knowledge graph, generates personalized feeds, explains papers at your expertise level, and synthesizes novel research hypotheses — all locally, all yours.

> **⚠ Proprietary software.** ResearchFlow is **not** open source. The code is published for reference only. Any use — commercial, internal, academic, research, evaluation, or personal — requires a paid license. See [`LICENSE`](LICENSE) and contact **verma.aarohan@gmail.com** for licensing terms.

---

## Why this exists

Existing arXiv tooling solves only half the problem. **arxiv-sanity** ranks papers but doesn't explain them. **Connected Papers** visualizes citation graphs but doesn't read papers. **Elicit** summarizes individual papers but doesn't combine them. **Semantic Scholar** indexes the field but doesn't personalize.

The actual research workflow is *cross-paper synthesis* — taking insights from three disjoint papers and asking *"what novel hypothesis emerges from combining them?"* No tool does that end-to-end, grounded in real source text, with falsifiable predictions.

ResearchFlow is the missing piece:

- **Discovery** — orientation-weighted feed (novelty vs. relevance) over your subscribed namespaces
- **Comprehension** — Study Mode generates a deep walkthrough of any paper at your expertise level (newcomer / practitioner / expert)
- **Retrieval** — hybrid search (PostgreSQL FTS + pgvector cosine, fused with RRF) and a grounded RAG chat with inline citations
- **Synthesis** — Genie takes 2–10 papers/concepts/methods (manual) or auto-discovers groups of 2–5 and produces a testable hypothesis with mechanism, experimental design, predicted outcomes, anti-finding, diagrams, and PoC code
- **Depth** — Deep Dive turns any synthesized hypothesis into an 11-section research synthesis article via single-pass generation with the reasoning model, grounded in the full text of all source papers

It runs entirely on your own infrastructure — local Docker for development, four env-var flips for Azure deployment. No third-party telemetry, no rate-limited SaaS, no opaque algorithms. The full architecture, every workflow, every algorithm is documented in [`docs/architecture.html`](docs/architecture.html).

---

## Table of Contents

- [ResearchFlow](#researchflow)
  - [Why this exists](#why-this-exists)
  - [Table of Contents](#table-of-contents)
  - [What it does](#what-it-does)
  - [Architecture overview](#architecture-overview)
  - [Prerequisites](#prerequisites)
  - [API Keys](#api-keys)
    - [Required](#required)
    - [Optional but recommended](#optional-but-recommended)
    - [Minimum working config](#minimum-working-config)
  - [Local Setup](#local-setup)
    - [Option A — Setup script (recommended)](#option-a--setup-script-recommended)
    - [Option B — Manual step-by-step](#option-b--manual-step-by-step)
  - [Docker Deployment (build/)](#docker-deployment-build)
  - [Test User Credentials](#test-user-credentials)
  - [How to use the app](#how-to-use-the-app)
    - [Feed](#feed)
    - [Hybrid Search / Deep Search](#hybrid-search--deep-search)
    - [Study Mode](#study-mode)
    - [Bookmarks](#bookmarks)
    - [Knowledge Graph](#knowledge-graph)
    - [RAG Chat](#rag-chat)
    - [Genie — Idea Synthesizer](#genie--idea-synthesizer)
      - [Manual mode (Cauldron)](#manual-mode-cauldron)
      - [Auto Discovery mode](#auto-discovery-mode)
      - [Idea Capsules](#idea-capsules)
      - [Deep Dive](#deep-dive)
  - [Manual Feed Refresh](#manual-feed-refresh)
  - [Nightly Ingestion Schedule](#nightly-ingestion-schedule)
  - [Environment Variables Reference](#environment-variables-reference)
  - [Azure Deployment](#azure-deployment)
  - [Project Structure](#project-structure)
  - [Verbose Debug Mode](#verbose-debug-mode)
  - [Running Unit Tests](#running-unit-tests)
  - [Common Issues](#common-issues)
  - [License](#license)

---

## What it does

| Feature | Description |
|---|---|
| **Feed** | Personalized, scored paper timeline from arXiv. Updated nightly. Concept-based hot/cold interest signals adjust ranking. Orientation nudges search relevance order (research → boosts novelty, production → boosts relevance). |
| **Hybrid Search** | Namespace-scoped keyword + semantic search over title, tldr, abstract, key_concepts, and methods_used — fused with RRF. Scoped to the user's selected subjects/topics. Toggle between **Basic** and **Deep Search** modes. |
| **Deep Search** | Natural-language literature query mode. LLM validates and rewrites the query, runs parallel semantic + keyword + graph-concept retrieval with a semantic-heavy fusion (0.70 weight), LLM re-ranks top candidates, and caches results with fuzzy embedding-similarity matching (cosine ≥ 0.92) for instant re-queries. Runs inline or as a background job. |
| **Paper Detail** | Full abstract, concepts, methods, implications — instant, no LLM call. |
| **Study Mode** | Streamed deep walkthrough shaped by **expertise level** (newcomer/practitioner/expert) and **orientation** (research lens emphasises novelty/implications; production lens emphasises deployment/tradeoffs). Cached per `(paper, expertise, orientation)`. |
| **Bookmarks** | Save papers with notes and organize into folders. |
| **Knowledge Graph** | 8-level force-directed graph: **Subject → Topic → Subtopic → Area → Sub-area → Cluster → Papers → Concepts/Methods**. Subject root node (e.g. "Computer Science") groups all domain topics. Scope: Full Feed (namespace-isolated) or Bookmarks (folder filter). Semantically-related papers shown as dotted violet edges. Subgraph cached (4h TTL). LLM uses **2-phase taxonomy** (Phase 1: canonical bounded structure; Phase 2: paper assignment) to prevent area explosion across batches. **Build Deep runs as a background job** for all selected namespaces — tracks progress in the notification panel, re-enables the button only when complete, and auto-refreshes the graph when done (even after navigating away and returning). Concurrent builds are capped at 2 namespaces at a time. Each area is committed incrementally so partial progress appears in the graph as the build runs. Works across all arXiv subjects — Mathematics, Physics, Statistics, Biology, Economics, etc. — using curated labels for 100+ known namespaces and automatic label derivation for any others. A per-topic filter in the toolbar lets you narrow the graph to a single topic when multiple are selected. |
| **RAG Chat** | Chat grounded in your indexed papers. Answer depth/vocabulary adapts to expertise level; emphasis adapts to orientation. Namespace-isolated. |
| **Genie** | Idea synthesizer with three modes: **Manual** (bookmarks, 2–10), **Auto** (full feed 2–5), **Query** (natural language → papers, 2–5). Each capsule tagged Manual/Auto/Query; Query capsules show the input query. Orientation + expertise level shape all modes. Each capsule tagged Manual/Auto/Query; hover over element library items to see TL;DR. Ideas are automatically hidden when their subject is deselected — the capsule list is scoped to the user's current topic subscriptions. |
| **Genie Auto Discovery** | Operates on the **full feed** (all papers in subscribed namespaces, up to 200 per run). 5-signal pair scoring; O(N²) capped at N=200 for sub-second pairing. Namespace-isolated via user subscriptions. |
| **Genie Query Mode** | Natural-language query → LLM validation + rewrite → semantic paper discovery → compatibility scoring → best synthesis group. Best-group papers are auto-selected; users can toggle any paper (bookmarked or feed) in/out. Hover shows TL;DR. Caps at 2–5 papers. |
| **Deep Dive** | Full research synthesis article for any Idea Capsule — multi-phase generation with LLM-as-judge refinement. Runs inline (streaming) or in the background. |
| **Audio Podcast** | One-click podcast generation from any paper or Genie capsule. Multi-speaker HOST/EXPERT teaching conversation, expertise-adapted depth, orientation-shaped emphasis, OpenAI TTS voices. Runs in background; embeds in-page audio player when complete. |
| **Slide Deck** | Marp-based slide deck generation. LLM plans and writes a presentation-grade deck (12–18 slides) with equations, results tables, diagrams, and methodology breakdown. Density guards prevent overflow. Rendered to standalone HTML via marp-cli (falls back to raw Markdown). |
| **Annotations** | Highlight text in any paper and attach personal notes, accessible from the Paper Detail panel. |
| **Token Usage** | Per-call accounting of every LLM completion (input/output tokens, model, cost estimate, latency). Settings → **Token Usage** tab shows totals, daily bar chart, per-workflow and per-model breakdowns. Defaults to today; supports custom date ranges with quick presets (7 days, 30 days, year). |
| **Settings** | Provider config, topic subscriptions, notifications, manual RSS refresh. |

---

## Media Generation (Audio · Slides)

Any paper or Genie idea capsule can be transformed into a podcast or slide
deck. Generation runs in the background, survives page navigation, refresh,
and worker restarts (DB-backed authoritative state + idempotent cache).

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Source: paper | capsule                                                 │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                  ContentLoaderService (deep PDF grounding)
                                 │
                                 ▼
        ┌───────────────────────────────────────────────────────┐
        │  LangGraph workflow per type                           │
        │   • podcast — load → plan → script → TTS → save (5)    │
        │   • slides  — load → plan → markdown → render → save (4)│
        └────────────────┬──────────────────────────────────────┘
                         │
   ┌─────────────────────┼─────────────────────┐
   │                     │                     │
   ▼                     ▼                     ▼
JobStore         GeneratedArtifact         BlobStorage
(Redis or       (Postgres — auth         (Local /blobs or
 in-memory)      state + cache key)       Azure Blob / S3)
```

**Cache key:** `(user, source, type, expertise, orientation, provider, model, parser)`.
A re-trigger hits the cache when *all* fields match; otherwise it generates fresh.

**Deep PDF grounding:** if a paper has not been parsed yet, the loader
parses on demand via the configured parser chain (Marker by default,
Docling if `PDF_PARSER=docling`, Gemini Vision as last-resort fallback)
and persists section chunks. Generations always use the full PDF body —
never just the abstract.

**Adaptive generation:** prompts detect the paper's domain and shape
(theory / systems / empirical / methods) and adapt segment depth,
analogies, and slide structure accordingly. Output completeness is
guarded by a truncation detector that retries the tail when the model
is cut off.

**Background execution:**
- Jobs run via `asyncio.create_task` and update a `GeneratedArtifact` DB row
  through `queued → running → completed | failed`.
- The `JobStore` (in-memory or Redis) provides a fast notification cache.
- On worker restart the lifespan sweeps any orphaned `running` rows to
  `failed` so the UI never shows an indefinite spinner.
- `GET /api/v1/generate/jobs` returns the user's in-flight jobs for the
  notification panel.

### UI placement

- **Study Mode** — sticky-header pill row: Audio · Slides.
  Audio embeds an inline player; Slides open in a fullscreen iframe.
- **Genie Idea Capsule page** — same two pills below the hero scores.
- **Notification bell** — every generation job appears in the JobsPanel
  alongside Study/Genie/Graph jobs. Click a completed job to redirect to
  the source viewer.

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Frontend  — Next.js App Router (TypeScript, React, Framer)      │
├──────────────────────────────────────────────────────────────────┤
│  API Layer — FastAPI async routers  (JWT auth, SSE, DI)          │
│              /feed · /study · /rag · /genie · /generate          │
├──────────────────────────────────────────────────────────────────┤
│  Workflows — LangGraph StateGraph (6) + async generators         │
│    Ingestion · Study · RAG · Podcast · Slides                    │
│    FolderConsolidation (coherence + cross-paper synthesis)       │
│    Genie · Deep Dive (custom async streaming generators)         │
├──────────────────────────────────────────────────────────────────┤
│  Services / Adapters — Scoring · GraphService                    │
│    LLM (OpenAI/Anthropic/Google)                                 │
│    Embedding (Gemini/OpenAI) · TTS (OpenAI)                      │
│    PDF (Docling/Marker/Gemini Vision) · Slides (Marp)            │
│    Blob · Cache · Email                                          │
│    Sources (arXiv RSS / MCP)                                     │
├──────────────────────────────────────────────────────────────────┤
│  Repositories — Paper · Vector · Search · Graph · Workflow       │
│                 Artifact  (only layer that issues SQL)            │
├──────────────────────────────────────────────────────────────────┤
│  Database — PostgreSQL + pgvector  +  generated_artifacts table  │
└──────────────────────────────────────────────────────────────────┘
```

Six layers, clean boundaries. Each layer only calls the one directly below it.  
Local → Azure swap: change four env vars, zero code changes.

### PDF Parser priority chain
```
PDF_PARSER=marker (default)        → Marker → Gemini Vision
PDF_PARSER=docling (opt-in)        → Docling → Marker → Gemini Vision
PDF_PARSER=gemini_vision (opt-in)  → Gemini Vision → Marker
```
Marker is the default because it is memory-friendly and stable on laptops,
WSL, and small VMs. Docling provides richer structured parsing (section
hierarchy, tables, equations) but pulls in PyTorch + EasyOCR models that
can spike RAM by ~2 GB on first run, so it is opt-in.

To enable Docling: `PDF_PARSER=docling` in `.env.local`. To additionally
enable scanned-PDF OCR inside Docling (needs ≥8 GB free RAM): also set
`DOCLING_OCR=1`.

All parsers produce the same `ParsedPaper` schema — zero downstream
changes when switching parsers, and per-tier failures fall through the
chain transparently.

### Generation API
```
POST /api/v1/generate/{paper|capsule|folder}/{id}/{podcast|slides}
GET  /api/v1/generate/artifact/{artifact_id}
GET  /api/v1/generate/{source_type}/{source_id}
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Docker + Docker Compose | v24+ | Runs PostgreSQL + pgvector |
| Python | 3.11+ | Backend |
| Node.js | 20+ | Frontend |
| Git | any | |

You do **not** need PostgreSQL or Redis installed locally — Docker handles both.

---

## API Keys

ResearchFlow needs at least one LLM provider key to function. Embeddings require a Google key (Gemini Embedding 2 is the default). Everything else is optional.

### Required

| Key | Where to get | Used for |
|---|---|---|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | LLM (enrichment, study, RAG, Genie, Deep Dive), image generation |
| `GOOGLE_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | Gemini Embedding 2 (default embeddings) |

### Optional but recommended

| Key | Where to get | Used for |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) | Fallback LLM provider |
| `RESEND_API_KEY` | [resend.com/api-keys](https://resend.com/api-keys) | Email notifications (PoTD, digest, alerts) |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com) | Workflow observability |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) | Web search tool (optional; DuckDuckGo used by default) |

### Minimum working config

```bash
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

With only OpenAI (no Google key): set `DEFAULT_EMBEDDING_PROVIDER=openai` — uses `text-embedding-3-large` instead.

---

## Local Setup

### Option A — Setup script (recommended)

```bash
chmod +x setup.sh
./setup.sh
```

**First run** — full setup (2–5 min):  
Checks prerequisites, handles Python 3.11 via pyenv if needed, collects API keys (skips any already in `.env.local`), writes `.env.local`, starts PostgreSQL, installs all dependencies, seeds the test user, and offers to launch both servers. Saves state to `.setup_state`.

**Every subsequent run** — start only (<5 s):  
Detects `.setup_state`, skips install steps, kills any processes occupying ports 8000/3000, then starts the database and both servers.

```bash
./setup.sh            # auto: setup if fresh, start if done
./setup.sh --run      # force start (skip setup checks)
./setup.sh --setup    # force full setup again
./setup.sh --reset    # wipe .setup_state and redo setup
./setup.sh --help     # show usage
```

---

### Option B — Manual step-by-step

**Step 1 — Clone and configure**

```bash
git clone <your-repo-url>
cd research_flow
cp .env.example .env.local
# Edit .env.local and paste your API keys
```

**Step 2 — Start the database**

```bash
docker compose up db -d
# Wait ~10 seconds for PostgreSQL to be ready
```

**Step 3 — Backend setup**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/seed_db.py
```

**Step 4 — Start the backend**

```bash
uvicorn main:app --reload --port 8000
# Open http://localhost:8000/docs to verify
```

**Step 5 — Frontend setup**

```bash
cd ../frontend
npm install
npm run dev
# Open http://localhost:3000

# For production mode (pre-compiles everything, no on-demand compilation):
npm run prod
```

**Step 6 — Log in and fetch papers**

1. Go to [http://localhost:3000/login](http://localhost:3000/login)
2. Sign in with the test user credentials below
3. Go to **Settings** → **Refresh Feed Manually** → select `cs.AI` → **Refresh Now**
4. Papers appear within 30–60 seconds

---

## Docker Deployment (build/)

The `build/` directory contains production-grade Docker files for running ResearchFlow without manually installing Python, Node.js, or PostgreSQL. Everything runs in containers.

> **Note on image size:** The backend image is ~4–6 GB due to ML dependencies (`marker-pdf` pulls PyTorch, OpenCV). The first build takes 10–20 minutes; subsequent builds are fast (layer cache).

### Quick start

```bash
cd build
cp .env.example .env
# Edit .env — set at minimum: OPENAI_API_KEY (or ANTHROPIC/GOOGLE), JWT_SECRET
nano .env

docker compose up --build
```

Wait for all three services to become healthy (watch `docker compose ps`), then open [http://localhost:3000](http://localhost:3000).

### What's in build/

| File | Purpose |
|---|---|
| `Dockerfile.backend` | Multi-stage Python 3.11 production image — installs all deps, copies source, runs uvicorn with 2 workers |
| `Dockerfile.frontend` | Multi-stage Next.js 20 production image — builds with `npm run build`, runs with `next start` (node_modules kept in image) |
| `docker-compose.yml` | Orchestrates PostgreSQL + pgvector, Redis, backend, and frontend with healthchecks and named volumes |
| `.env.example` | Template for all environment variables — copy to `.env` and fill in |

### Seeding the database

After containers are up, create the default test user:

```bash
docker compose exec backend python scripts/seed_db.py
```

### Useful commands

```bash
# View logs
docker compose logs -f backend
docker compose logs -f frontend

# Restart a single service after config change
docker compose restart backend

# Stop everything
docker compose down

# Wipe data volumes (full reset)
docker compose down -v
```

### Deploying to a server (non-localhost)

If your backend is not on `localhost:8000`, set `NEXT_PUBLIC_API_URL` in `.env` **before building**:

```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

This is baked into the frontend bundle at build time (used for SSE streaming connections). After changing it, rebuild: `docker compose up --build frontend`.

---

## Test User Credentials

| Field | Value |
|---|---|
| **Email** | `test@researchflow.ai` |
| **Password** | `ResearchFlow2024!` |
| **Expertise** | Practitioner |
| **Orientation** | Both |
| **Subscribed namespaces** | `cs.AI`, `cs.ML`, `cs.NLP` |

To reset: `docker compose down -v && docker compose up db -d && python scripts/seed_db.py`

---

## How to use the app

### Feed

The feed shows papers scored by novelty × orientation + relevance × (1 − orientation) with subtopic affinity boosts.

- **Click a card** → opens a slide-in Paper Detail panel
- **Save** → adds to Bookmarks
- **arXiv →** → opens the source paper


![Feed](screenshots/feed.png)

### Hybrid Search / Deep Search

The search bar runs **keyword + semantic** search fused with Reciprocal Rank Fusion:

- **Keyword** — PostgreSQL full-text search (`to_tsvector` + `plainto_tsquery`)
- **Semantic** — pgvector cosine similarity on Gemini/OpenAI embeddings
- **Fusion** — RRF (k=60) boosts papers ranking high in both paths

Each result shows a `hybrid`, `semantic`, or `keyword` match badge.

```bash
GET /api/v1/search?q=attention+mechanism&namespace_key=cs.AI&mode=hybrid&limit=20
```

![Hybrid Search](screenshots/hybrid.png)
![Deep Search](screenshots/deep.png)

### Study Mode

Streams a deep paper walkthrough via SSE:

```
🧩 The Problem → 🏛 Prior Work → 💡 Core Idea → 🔢 The Method
→ 🖼 Diagrams → 📊 Results → 🤔 Open Questions → 💻 Code 
```

Three expertise levels: **Newcomer**, **Practitioner**, **Expert**. Output cached per paper × level.

![Study Mode](screenshots/study.png)

### Bookmarks

Save papers with notes, organize into named color-coded folders. Click a bookmark to open the paper detail panel.

![Bookmarks](screenshots/bookmarks.png)

### Knowledge Graph

Force-directed graph of your research space:

| Color | Node type |
|---|---|
| Indigo | Topics |
| Violet | Subtopics |
| Teal | Concepts |
| Amber | Methods |
| Gray | Papers |

Click "Expand" on any node to load its neighbors. Animated yellow edges = cross-namespace bridges (built weekly).

While a Build Deep job is active, the graph auto-reloads every 20 seconds to pick up intermediate commits. SUBTOPIC nodes with 0 children show an amber **"Building taxonomy…"** hint instead of "0 research areas" while the build is in progress.

- When multiple topics are selected across a subject, use the **topic filter** dropdown in the toolbar to narrow the graph to a single topic at a time.
- Clicking any node correctly collapses its entire subtree (including nested expanded nodes). Semantic dotted edges (related-to) are excluded from the collapse traversal so sibling branches are never accidentally hidden.

![Knowledge Graph](screenshots/graph2.png)
![Knowledge Graph](screenshots/graph1.png)

### RAG Chat

Chat grounded in your indexed papers:
- Select a paper from the feed
- Ask any question — the system searches, reranks, checks sufficiency, then synthesizes a cited answer
- Inline citations `[1]`, `[2]` link back to source papers
- If context is insufficient, the system says so and offers to broaden scope


### Genie — Idea Synthesizer

The flagship feature. Combines research elements to synthesize novel, grounded, testable hypotheses.

#### Manual mode (Cauldron)

1. Open **Genie** from the sidebar
2. Your **Element Library** is on the left — populated automatically as you study papers
3. Click 2–10 elements to add them to the **Cauldron** (manual mode)
4. Select a namespace
5. Click **SYNTHESIZE**
6. Stream: context gathering → bridge discovery → hypotheses → scoring → elaboration → diagrams → method sketch
7. Result saved as an **Idea Capsule** in the **Ideas** tab

#### Auto Discovery mode

Genie scans your bookmarks + top-ranked feed papers, clusters them by **semantic similarity + concept overlap**, and autonomously synthesizes ideas from compatible pairs — no manual element selection needed.

- Click **Run Now** to trigger immediately
- Tunable signals under *Constraints & Thresholds*:
  - **Temperature** — scalar (0–1) controlling exploration vs. safety across five derived variables: semantic threshold, staleness multiplier, freshness bonus, candidate pool size, and semantic dedup threshold. Labels: Safe → Focused → Balanced → Curious → Exploratory
  - **Semantic Similarity** — minimum embedding cosine similarity between papers to be paired (default 0.25)
  - **Concept Overlap** — minimum Jaccard overlap of key concepts (default 0.05)
- Two-layer deduplication prevents regenerating ideas already in your library:
  - Layer 1: Jaccard overlap of paper IDs (structural)
  - Layer 2: Cosine similarity of hypothesis embeddings (semantic)
- Graph structure data contributes a minor 10% enrichment bonus to pair scoring — it does not gate pairings
- Results appear in the **Ideas** tab alongside manually synthesized capsules

#### Idea Capsules

Each synthesized capsule contains:

| Field | Description |
|---|---|
| **Title + TL;DR** | One-sentence summary of the core idea |
| **Hypothesis** | The core scientific claim |
| **Rationale** | Why this is worth pursuing |
| **Mechanism** | How it works technically |
| **Experimental Design** | Concrete protocol to test it |
| **Predicted Outcomes** | Measurable success criteria |
| **Anti-Finding** | What would falsify this idea |
| **Risks & Limitations** | Known failure modes |
| **Open Questions** | What still needs to be resolved |
| **Scores** | Novelty · Feasibility · Impact (0–1) |
| **Method Sketch** | Concise proof-of-concept code or step sketch (when applicable) |
| **Diagrams** | Mermaid architecture/flow diagrams (when applicable) |
| **Source Papers** | Papers the idea was synthesized from, with arXiv links |

Save capsules to keep them; saved capsules become new elements you can recombine.

![Idea Capsule](screenshots/genie.png)

#### Deep Dive

Generate a full research synthesis article from any Idea Capsule.

Click **Generate Deep Dive** — generation queues in the background automatically. A spinner shows while it runs; the article streams in when ready. Navigating away and back is safe — the result is persisted to the database and restored instantly on page load.

**Single-pass generation via the reasoning model:**
Deep Dive uses the reasoning model (`gpt-5.4` / `claude-opus` depending on provider) directly in one pass with a strong system prompt that acts as author and fact-checker simultaneously. The model is instructed to:
- Ground every claim in the provided source paper text
- Strip any hallucinations or unsupported citations
- Write with technical authority and precision
- End with a complete `## References` section with arXiv URLs

The article streams live to the frontend and is persisted to `idea_capsules.deep_dive_content` on completion so it loads instantly on future visits.

**Output structure (11 sections):**

| Section | Content |
|---|---|
| Abstract | ~120 words: what's proposed, why novel, practical impact |
| 1. The Convergence | Shared abstraction or unsolved problem tying the source papers |
| 2. Paper Contributions & Intellectual Lineage | Per paper: exact contribution to THIS idea + the gap it can't fill alone |
| 3. Unified Theoretical Framework | How papers integrate; which element came from which paper (`[N]` citations) |
| 4. Architecture & Mechanism | End-to-end technical description with Mermaid diagram |
| 5. Related Work & Differentiation | 4–6 prior works with comparison table |
| 6. Experimental Design | Concrete reproducible protocol |
| 7. Predicted Outcomes | Quantitative predictions on specific benchmarks |
| 8. Falsification | What specific outcome would disprove this |
| 9. Risks & Mitigations | Concrete failure modes, each with a mitigation |
| 10. Implementation Roadmap | Three phases: PoC → ablation → full eval |
| 11. Scientific Impact | What becomes possible that wasn't before |

All claims are grounded with `[N]` inline citations; a `## References` section with arXiv links is appended automatically.

![Idea Capsule](screenshots/genie_deep.png)

---

## Manual Feed Refresh

```bash
curl -X POST "http://localhost:8000/api/v1/feed/refresh?namespace_key=cs.AI" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Response:
```json
{
  "triggered": true,
  "namespace_key": "cs.AI",
  "message": "Ingestion started for cs.AI. New papers appear in 30–60s."
}
```

Pipeline: arXiv RSS fetch → enrichment → embeddings → graph update → scoring → PoTD selection.

---

## Nightly Ingestion Schedule

| Job | Default schedule | What it does |
|---|---|---|
| Ingestion | `0 5 * * 2-5` — 05:00 UTC, Tue–Fri | Fetches new papers, enriches, embeds, updates graph, scores PoTD. Runs on the four days with fresh arXiv content (Mon–Thu announcements arrive by 01:00 UTC). |
| Clustering | `0 5 * * 0` — Sun 05:00 UTC | Subtopic discovery — job scaffold registered; full HDBSCAN implementation is post-MVP |
| Cross-namespace links | `30 5 * * 0` — Sun 05:30 UTC | Cross-namespace concept bridge edges — job scaffold registered; cosine-similarity pass is post-MVP |
| Bookmark index rebuild | Sun 03:00 UTC (fixed) | Re-embeds any bookmarked papers that are missing an abstract chunk |

Schedules are configurable via `INGESTION_CRON`, `CLUSTERING_CRON`, `CROSS_NAMESPACE_CRON`.

To trigger ingestion manually from Python:
```python
from app.workflows.ingestion import run_ingestion
import asyncio
asyncio.run(run_ingestion("cs.AI"))
```

---

## Environment Variables Reference

| Variable | Default | Required | Description |
|---|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` | ✓ | PostgreSQL connection string |
| `OPENAI_API_KEY` | — | ✓ | OpenAI API key |
| `GOOGLE_API_KEY` | — | ✓* | Google AI key (*required for Gemini embeddings) |
| `ANTHROPIC_API_KEY` | — | ✗ | Anthropic key (fallback LLM) |
| `DEFAULT_LLM_PROVIDER` | `openai` | ✗ | `openai` \| `anthropic` \| `google` |
| `DEFAULT_CHEAP_MODEL` | `gpt-4o-mini` | ✗ | Fast model for query rewrite, intent classify, rerank, enrichment batches |
| `DEFAULT_QUALITY_MODEL` | `gpt-5.4-mini` | ✗ | Mid-tier model for RAG synthesis, Genie hypothesize/critique |
| `DEFAULT_REASONING_MODEL` | `gpt-5.4` | ✗ | Strong reasoning model for Genie elaborate, PoC code, Deep Dive article |
| `DEFAULT_EMBEDDING_PROVIDER` | `gemini` | ✗ | `gemini` \| `openai` \| `voyage` |
| `VOYAGE_API_KEY` | — | ✗ | Required only when `DEFAULT_EMBEDDING_PROVIDER=voyage` |
| `DEFAULT_EMBEDDING_DIM` | `768` | ✗ | Must match the provider's output dimension |
| `INGESTION_MODE` | `rss` | ✗ | `rss` \| `mcp` |
| `CACHE_BACKEND` | `local` | ✗ | `local` \| `redis` |
| `BLOB_BACKEND` | `local` | ✗ | `local` \| `azure` |
| `PDF_PARSER` | `marker` | ✗ | `marker` \| `docling` \| `gemini_vision`. Docling is opt-in (RAM-heavy); Marker is the memory-safe default. |
| `MARKER_API_KEY` | — | ✗ | Marker cloud API key (local `marker-pdf` package works without it) |
| `RESEND_API_KEY` | — | ✗ | Email sending (emails disabled if blank) |
| `LANGSMITH_API_KEY` | — | ✗ | LangSmith observability |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` | ✗ | Web search backend for LLM tool: `duckduckgo` (free) or `tavily` |
| `TAVILY_API_KEY` | — | ✗ | Required when `WEB_SEARCH_PROVIDER=tavily` |
| `BREAKTHROUGH_THRESHOLD` | `0.88` | ✗ | Novelty score cutoff for breakthrough classification |
| `ENVIRONMENT` | `local` | ✗ | `local` \| `azure` |
| `DEBUG` | `false` | ✗ | Enables Swagger UI, SQL echo, `/debug/status`, request logs |
| `LOG_LEVEL` | `INFO` | ✗ | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `JWT_SECRET` | — | ✓ | Change before any non-local deployment |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | ✗ | JSON array of allowed CORS origins |
| `TTS_PROVIDER` | `openai` | ✗ | TTS provider for podcast generation (only `openai` currently) |
| `TTS_MODEL` | `tts-1-hd` | ✗ | OpenAI TTS model: `tts-1` (fast) or `tts-1-hd` (higher quality) |

---

## Azure Deployment

The switch from local to Azure is **env-var only** — zero code changes.

| Local | Azure |
|---|---|
| `DATABASE_URL=postgresql+asyncpg://localhost/...` | `DATABASE_URL=postgresql+asyncpg://<azure-flexible-server>` |
| `CACHE_BACKEND=local` | `CACHE_BACKEND=redis` + `REDIS_URL=rediss://<azure-cache>` |
| `BLOB_BACKEND=local` | `BLOB_BACKEND=azure` + `AZURE_STORAGE_CONNECTION_STRING=...` |
| `JobStore` in-process | `JobStore` shared via Azure Cache for Redis |
| uvicorn locally | Azure Container Apps (multi-replica) |
| `npm run dev` | Azure Static Web Apps |
| APScheduler in-process | Azure Container Apps Jobs |

### Recommended Azure topology

| Component | Azure resource |
|---|---|
| Backend container | Azure Container Apps (HTTP ingress + scale rules) |
| Frontend | Azure Static Web Apps |
| Database | Azure Database for PostgreSQL Flexible Server (with `vector` extension) |
| Cache + JobStore | Azure Cache for Redis |
| Object storage | Azure Blob Storage (private container; `BLOB_BACKEND=azure`) |
| Secrets | Azure Key Vault (mount as env vars in Container Apps) |
| Observability | Azure Monitor + Application Insights (auto-instrumented via OpenTelemetry) |
| Scheduled jobs | Azure Container Apps Jobs running `python -c "from app.workflows.ingestion import run_all_ingestion; ..."` |
| Queue (future) | Azure Service Bus — see `services/job_store.py` ABC for swap point |

### Vector index migration (local IVFFlat → Azure HNSW)

```sql
DROP INDEX paper_chunks_emb_768;
CREATE INDEX paper_chunks_emb_768_hnsw ON paper_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE embedding_dim = 768;
```

### Generation-specific Azure notes

The backend Docker image (`build/Dockerfile.backend`) ships with:
- `marp-cli` (Node 20) — slide rendering
- `ffmpeg` — audio merging
- All Python deps including `marker-pdf`

For Azure Container Apps, expose:
- Port 8000 (HTTP)
- Optional `PDF_PARSER=docling` to switch to IBM Docling (richer structure
  but ~2 GB extra RAM on first parse — requires ≥8 GB free).

---

## AWS Deployment

Same env-var-only swap, AWS edition.

| Local | AWS |
|---|---|
| `DATABASE_URL=postgresql+asyncpg://localhost/...` | `DATABASE_URL=postgresql+asyncpg://<rds-endpoint>` |
| `CACHE_BACKEND=local` | `CACHE_BACKEND=redis` + `REDIS_URL=rediss://<elasticache-endpoint>` |
| `BLOB_BACKEND=local` | S3 — extend `BlobStorageBackend` ABC with an `S3BlobStorage` adapter; the same `blob.upload/download/exists/delete` contract applies. |
| Backend container | ECS Fargate or EKS (with Application Load Balancer ingress) |
| Frontend | CloudFront + S3 static site OR Amplify Hosting |
| Secrets | AWS Secrets Manager → injected as env vars |
| Observability | CloudWatch logs + AWS X-Ray |
| Scheduled jobs | EventBridge Scheduler → ECS Run Task |
| Queue (future) | AWS SQS — implement an `SQSJobStore` against the same `JobStore` ABC |

### Stateless container guarantees (Azure + AWS)

ResearchFlow's containers are stateless by design:
- **Generated artifacts** — persisted in PostgreSQL + BlobStorage (never local disk).
- **Job state** — authoritative in PostgreSQL, optional fast-path in Redis.
- **Auth tokens** — stateless JWTs (no session store).
- **PDF cache** — `BLOB_BACKEND` controls whether parsed PDFs go to local disk
  or Azure Blob/S3.

That means horizontal autoscaling works out of the box on both clouds.

---

## Project Structure

```
research_flow/
├── backend/
│   ├── main.py                  # FastAPI app, lifespan, CORS, startup recovery
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── alembic/                 # DB migrations
│   ├── scripts/
│   │   └── seed_db.py           # Creates test user + SourceMappings
│   └── app/
│       ├── core/
│       │   ├── config.py        # Pydantic settings (all env vars)
│       │   ├── security.py      # JWT, password hashing
│       │   ├── deps.py          # FastAPI DI: DB session, current user
│       │   └── tracking.py      # ContextVars for token-usage attribution
│       ├── db/
│       │   ├── base.py          # SQLAlchemy DeclarativeBase
│       │   ├── session.py       # Async engine + session factory
│       │   └── checkpointer.py  # AsyncPostgresCheckpointer — LangGraph crash-resume for media gen
│       ├── models/              # SQLAlchemy ORM models
│       │   ├── user.py          # User, UserProviderSettings, Annotation
│       │   ├── paper.py         # Paper, PaperChunk, Summary, Bookmark, PoTD, QueryLog, FeedFeedback
│       │   ├── graph.py         # KnowledgeNode, KnowledgeEdge, NamespaceSubscription, SourceMapping
│       │   ├── workflow.py      # WorkflowRun, TokenUsage
│       │   ├── genie.py         # GenieElement, IdeaCapsule, GenieSession
│       │   └── artifact.py      # GeneratedArtifact (podcast/slides)
│       ├── schemas/             # Pydantic v2 request/response schemas
│       ├── adapters/
│       │   ├── llm/             # OpenAI, Anthropic, Google adapters + TrackingLLMAdapter
│       │   ├── embedding/       # Gemini 2, OpenAI embedding adapters
│       │   ├── image_gen/       # Image generation adapter
│       │   ├── pdf/             # Docling, Marker, Gemini Vision — fallback chain
│       │   ├── tts/             # OpenAITTSAdapter (TTS ABC + factory)
│       │   ├── slides/          # MarpSlidesAdapter (Slides ABC + factory)
│       │   ├── cache/           # LocalFile + Redis backends
│       │   ├── blob/            # Local + Azure Blob backends
│       │   ├── email/           # Resend adapter
│       │   └── sources/         # ArXivRssSource, ArXivMcpSource, SourceRegistry
│       ├── resilience/
│       │   └── resilient_call.py # Retry + circuit breaker + fallback
│       ├── repositories/        # DB access layer (only layer that touches DB)
│       │   ├── paper.py
│       │   ├── user.py
│       │   ├── graph.py
│       │   ├── vector.py        # pgvector similarity search
│       │   ├── search.py        # Hybrid search: keyword + semantic + RRF fusion
│       │   ├── workflow.py
│       │   └── artifact.py      # GeneratedArtifact CRUD + cache lookup
│       ├── services/
│       │   ├── scoring.py       # Feed scoring (pure SQL, no LLM)
│       │   ├── graph.py         # GraphService
│       │   ├── namespace.py     # Namespace ↔ arXiv category mapping
│       │   ├── token_usage.py   # Per-call accounting
│       │   ├── email_service.py # PoTD, digest, breakthrough emails
│       │   ├── content_loader.py # Source content loader (paper/capsule/folder, deep PDF grounding)
│       │   └── job_store.py     # JobStore ABC + InMemory/Redis adapters
│       ├── workflows/           # LangGraph agentic workflows
│       │   ├── ingestion.py     # Nightly: fetch→enrich→embed→graph→score
│       │   ├── study.py         # On-demand: parse→structure→explain→stream
│       │   ├── rag.py           # On-demand: rewrite→retrieve→rerank→synthesize
│       │   ├── genie.py         # Synthesis + Auto-batch + Deep Dive (custom async generators)
│       │   ├── _generation_prompts.py # Shared prompts/heuristics for media gen
│       │   ├── _generation_runtime.py # Queue/recovery helpers + orphan re-dispatch
│       │   ├── podcast.py       # 5-node StateGraph + multi-turn segmented script + TTS
│       │   ├── slides.py        # 4-node StateGraph + multi-turn batched Marp generation
│       │   └── folder_consolidation.py # 3-node StateGraph: load → coherence → synthesize
│       ├── api/v1/              # FastAPI routers
│       │   ├── auth.py · feed.py · papers.py · study.py · search.py
│       │   ├── chat.py · bookmarks.py · graph.py · genie.py · settings.py
│       │   └── generate.py      # POST/GET /generate — media generation control plane
│       └── scheduler/
│           └── jobs.py          # APScheduler: nightly + weekly jobs
│
├── frontend/
│   ├── app/
│   │   ├── (auth)/login/        # Login page
│   │   ├── (auth)/signup/       # Signup page
│   │   └── (app)/               # Authenticated app shell (sidebar nav)
│   │       ├── feed/            # Personalized paper feed
│   │       ├── study/[id]/      # Streaming Study Mode
│   │       ├── bookmarks/       # Reading list
│   │       ├── graph/           # Knowledge Graph (React Flow)
│   │       ├── chat/            # RAG Chat
│   │       ├── paper/           # Paper detail (slide-in panel)
│   │       ├── genie/           # Idea Synthesizer + Ideas tab (Cauldron + auto-discovery)
│   │       │   └── idea/[id]/   # Idea Capsule detail + Deep Dive
│   │       └── settings/        # Provider config, subscriptions, refresh
│   ├── components/
│   │   ├── feed/                # PaperCard, SearchBar, FeedFilters
│   │   ├── paper/               # Slide-in paper detail panel
│   │   ├── study/               # Study mode sections and streaming UI
│   │   ├── genie/               # CapsuleCard and Genie-specific components
│   │   ├── graph/               # Knowledge graph canvas and controls
│   │   ├── bookmarks/           # Bookmark cards and folder management
│   │   ├── jobs/                # Background jobs status panel
│   │   ├── layout/              # Sidebar, nav, shell components
│   │   └── ui/                  # Toaster, Skeleton, shared primitives
│   ├── hooks/                   # use-toast and other shared hooks
│   ├── lib/api.ts               # Typed fetch wrapper + SSE helper
│   ├── store/auth.ts            # Zustand auth store (token + user)
│   └── types/index.ts           # TypeScript types mirroring Pydantic schemas
│
├── docs/
│   ├── architecture.html        # Interactive technical reference (architecture, workflows, DB, patterns)
│   └── sphinx/                  # Auto-generated API reference (Sphinx + autodoc + Napoleon)
│       ├── source/              #   RST source files + conf.py
│       └── build/html/          #   Built HTML docs (open index.html)
├── docker-compose.yml           # PostgreSQL + pgvector
├── setup.sh                     # One-shot interactive setup + launch script
├── .env.example                 # Template — copy to .env.local
├── .env.local                   # Your keys (gitignored)
└── README.md                    # This file
```

---

## Verbose Debug Mode

Set `DEBUG=true` (already the default in `.env.local`) to unlock:

| What | How |
|---|---|
| **Swagger UI** | `http://localhost:8000/docs` |
| **SQL query echo** | Every SQL statement printed to backend stdout |
| **Request logs** | `METHOD /path → STATUS  Xms  req_id=abc` for every request |
| **`/debug/status`** | Non-sensitive config snapshot |
| **Response headers** | `X-Request-Id` and `X-Response-Time-Ms` on every response |

```bash
curl http://localhost:8000/debug/status | jq
```

Turn off for production:
```env
DEBUG=false
LOG_LEVEL=WARNING
```

---

## Running Unit Tests

```bash
cd backend
source .venv/bin/activate
pytest
```

Tests run without a real database — all DB calls are mocked with `AsyncMock`.

| Module | What's tested |
|---|---|
| `test_security.py` | Password hashing, JWT creation/decode, expiry |
| `test_scoring.py` | Score formula, orientation weights, clamping, why-tags |
| `test_arxiv_rss.py` | arXiv ID extraction, date parsing, HTTP mock fetch |
| `test_paper_repository.py` | Upsert, bookmark, feedback with mocked DB |
| `test_search_repository.py` | RRF fusion math, keyword/semantic mocks, hybrid calls |
| `test_api_auth.py` | Register, login, /me via TestClient with dependency overrides |
| `test_api_feed.py` | Feed, feedback, refresh, health endpoints |

```bash
pytest -v            # verbose
pytest -k security   # run specific module
pytest --tb=short    # compact tracebacks
```

---

## Token Usage Tracking

Every LLM completion routed through `get_llm_adapter()` is automatically recorded to the `token_usage` table — input/output tokens, model, latency, and a USD cost estimate. The Settings → **Token Usage** tab visualises this:

- **Default scope:** today (UTC)
- **Filters:** date range picker plus quick presets (Today / Last 7 days / Last 30 days / Last year)
- **Breakdowns:** per UTC day (bar chart), per workflow (study, genie, rag, deep_dive, deep_search, ingestion, …), per provider+model
- **Recording is non-blocking** — `asyncio.create_task` schedules the DB insert so an LLM call never waits on it, and a tracking failure cannot break an LLM call.
- **Streaming estimates** — provider streaming APIs do not consistently expose token counts, so streaming paths estimate tokens from text length (~4 chars/token). Non-streaming `complete()` paths record the exact provider counts.

Costs use a built-in price table (`backend/app/adapters/llm/tracking.py`); unknown models fall back to a 0.002 USD / 1K-token average. Edit that table to match your actual contracts if needed.

---

## Common Issues

**"No papers in feed"**  
→ Run a manual refresh: Settings → Refresh Feed Manually → select namespace → Refresh Now.

**"Search returns no results"**  
→ Papers must be ingested first. Run a feed refresh, wait 30–60s, then search.  
→ Keyword search works without an API key. Semantic search requires an embedding key.

**"Study is empty / errors"**  
→ Check `OPENAI_API_KEY` and `GOOGLE_API_KEY` are set correctly in `.env.local`.  
→ Restart the backend after changing env vars.

**"Connection refused on port 5432"**  
→ Run `docker compose up db -d` and wait 10 seconds.

**"Address already in use" on port 8000 or 3000**  
→ The setup script automatically kills any processes on these ports before starting. If running manually: `kill -9 $(lsof -ti:8000)`.

**"marker-pdf import error"**  
→ `marker-pdf` requires `torch`. If unwanted, set `PDF_PARSER=gemini_vision`.

**"Genie element library is empty"**  
→ Elements are populated automatically as you study papers. Study a few papers first.

**"Genie Auto Discovery produces no results"**  
→ You need at least 2 bookmarked papers with sufficient semantic similarity. Lower the *Semantic Similarity* threshold in *Constraints & Thresholds* (try 0.15).

**"Deep Dive output is missing"**  
→ If generated in background mode, the page restores it automatically on next load. If it shows "failed", try *Generate Deep Dive* (inline streaming) instead.

**"Study page shows ERR_EMPTY_RESPONSE on first visit in dev mode"**  
→ First visit to /study/[id] in `next dev` mode triggers on-demand compilation (20–30s). Use `npm run prod` to pre-compile all pages, or restart with `npm run dev` which now uses Turbopack (--turbo) for faster compilation.

**"Graph shows stray/orphaned nodes when switching subjects (e.g. Mathematics)"**  
→ This was caused by incorrect namespace normalization mapping all `math.*` namespaces to `math.OC`. Fixed — each namespace now gets its own correctly-keyed subtopic node. If stray nodes remain from a previous build, click **Clear All** then run **Build Deep** again.

**"Genie shows ideas from a subject I turned off"**  
→ Fixed. The capsule list now accepts `namespace_keys` and hides ideas whose source papers are all from deselected subjects. Re-navigate to Genie or switch tabs to trigger a fresh fetch with the new subscription.

**"Genie or Study page fails with ModuleBuildError / shiki error under Turbopack"**  
→ `transpilePackages: ["shiki", "katex"]` was added to `next.config.mjs`. If you see this error, ensure you have the latest `next.config.mjs` and restart the dev server.

**"Idea Q&A chat panel shows empty response"**  
→ Fixed. The backend SSE format (`{type: 'chunk', content: ...}`) didn't match the frontend parser (which read `p.chunk`). Frontend now reads `p.type` and `p.content` correctly.

**"Graph nodes look 'stray' or disconnected from their parent"**  
→ Two safeguards run automatically: hierarchical pre-positioning (children placed near their parent) plus a stronger link force (0.75). Additionally, a frontend `dedupeConceptNodes` step now merges case-insensitive duplicate concept nodes ("Task-Specific Assistants" and "task-specific assistants") under one canonical node by redirecting edges and dropping aliases — so older deep-build inconsistencies no longer appear as floating duplicates. A rebuild alone does not clean these duplicates from the DB; for a permanent cleanup do **Clear All → Build Deep**.

**LangSmith traces not appearing**  
→ Set `LANGCHAIN_TRACING_V2=true` and provide a valid `LANGSMITH_API_KEY`.

**Tests fail with import errors**  
→ Run pytest from inside the `backend/` directory with the virtualenv active.

---

## License

ResearchFlow is **proprietary, commercial software**. It is **not** open source and is **not** offered under MIT, Apache, BSD, GPL, or any other open-source license.

| What you may do | What you may NOT do without a paid license |
|---|---|
| Read the source code in this repository for reference | Run, deploy, host, or serve the Software |
| Fork it for personal reading | Copy or redistribute it |
| Cite or quote it with attribution | Modify or create derivative works |
|  | Embed it in another product, internal or commercial |
|  | Use it for academic, research, or evaluation purposes |
|  | Train, fine-tune, or evaluate ML models on it |

All Use — including non-profit, educational, individual, and research use — requires a written license agreement and payment of fees and/or royalties. See [`LICENSE`](LICENSE) for the full terms.

**To request a license:** email **verma.aarohan@gmail.com** with your intended scope of use, deployment environment, and expected user count or revenue scale.

Unauthorized use constitutes copyright infringement and may result in civil and criminal penalties.

© 2026 Aarohan Verma. All Rights Reserved.
