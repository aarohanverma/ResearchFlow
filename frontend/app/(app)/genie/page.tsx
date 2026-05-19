"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth";
import { useJobsStore } from "@/store/jobs";
import { useNamespaceStore, NAMESPACE_TREE } from "@/store/namespace";
import { useFeature } from "@/lib/features";
import type { GenieElement, IdeaCapsule, BookmarkFolder, Bookmark } from "@/types";
import { FolderIcon, SlidersHorizontalIcon, RotateCcwIcon, ThermometerIcon } from "lucide-react";
import {
  FlaskConicalIcon,
  ZapIcon,
  Loader2Icon,
  SearchIcon,
  XIcon,
  Trash2Icon,
  CheckIcon,
  SparklesIcon,
  ExternalLinkIcon,
  MessageSquareIcon,
  SendIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  BrainIcon,
  BeakerIcon,
  AlertTriangleIcon,
  LightbulbIcon,
  ArrowRightIcon,
  ClockIcon,
  RefreshCwIcon,
  BookOpenIcon,
  GitMergeIcon,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import MarkdownRenderer from "@/components/ui/MarkdownRenderer";

// ── Types ──────────────────────────────────────────────────────────────────────

type StreamEvent = {
  type:
    | "start" | "status" | "hypothesis" | "scores" | "elaboration"
    | "elaboration_section" | "diagram" | "code" | "done" | "error"
    | "viability" | "not_viable";
  message?: string;
  reason?: string;
  section?: string;
  data?: Record<string, unknown>;
  content?: string;
  novelty?: number;
  feasibility?: number;
  impact?: number;
  capsule_id?: string;
  spec?: string;
  blob_path?: string;
  similarity?: number;
  bridges?: string[];
};

type ElaborationSections = Record<string, string>;

type Mode = "manual" | "auto" | "query";
type Tab = "cauldron" | "discoveries";

interface QueryDiscoverResult {
  papers: Array<{
    paper_id: string;
    title: string;
    namespace_key: string;
    query_relevance: number;
    source_url: string;
    abstract?: string;
    tldr?: string;
  }>;
  best_group: Array<{
    paper_id: string;
    title: string;
    namespace_key: string;
    query_relevance: number;
    source_url: string;
  }>;
  best_group_score: number;
  rewritten_query: string | null;
  session_id: string | null;
  error?: string;
}

interface AutoStatus {
  last_run: string | null;
  last_status: string | null;
  discoveries_count: number;
}

const SECTION_META: Record<string, { label: string; icon: React.ElementType; color: string; bgColor: string; borderColor: string }> = {
  mechanism:           { label: "Mechanism",           icon: BrainIcon,         color: "text-indigo-400",  bgColor: "bg-indigo-950/20",  borderColor: "border-indigo-800/30" },
  methodology_bridge:  { label: "Methodology Bridge",  icon: ArrowRightIcon,    color: "text-violet-400",  bgColor: "bg-violet-950/20",  borderColor: "border-violet-800/30" },
  experimental_design: { label: "Experimental Design", icon: BeakerIcon,        color: "text-teal-400",    bgColor: "bg-teal-950/20",    borderColor: "border-teal-800/30"   },
  expected_outcomes:   { label: "Expected Outcomes",   icon: LightbulbIcon,     color: "text-emerald-400", bgColor: "bg-emerald-950/20", borderColor: "border-emerald-800/30"},
  key_tensions:        { label: "Key Tensions",        icon: AlertTriangleIcon, color: "text-amber-400",   bgColor: "bg-amber-950/20",   borderColor: "border-amber-800/30"  },
  risks_and_limitations:{ label: "Risks & Limitations",icon: AlertTriangleIcon, color: "text-red-400",     bgColor: "bg-red-950/20",     borderColor: "border-red-800/30"    },
  open_questions:      { label: "Open Questions",      icon: SparklesIcon,      color: "text-sky-400",     bgColor: "bg-sky-950/20",     borderColor: "border-sky-800/30"    },
  impact:              { label: "Research Impact",     icon: ZapIcon,           color: "text-orange-400",  bgColor: "bg-orange-950/20",  borderColor: "border-orange-800/30" },
};

const SECTION_ORDER = [
  "mechanism", "methodology_bridge", "experimental_design",
  "expected_outcomes", "key_tensions", "risks_and_limitations",
  "open_questions", "impact",
];

// ── Threshold defaults ─────────────────────────────────────────────────────────

const DEFAULT_THRESHOLDS = {
  semThreshold: 0.25,
  jacThreshold: 0.05,
  temperature: 0.5,
};

const THRESHOLDS_KEY = "genie_thresholds";

/**
 * Build a short, plain-prose TL;DR from an Idea Capsule's hypothesis text.
 *
 * The combined-idea fusion model produces one long paragraph rich with
 * markdown emphasis (``**bold**``, backticks, parens with commas) and
 * almost no early sentence terminators. Splitting on the first ``[.!?]``
 * returns the whole paragraph, which then renders as a wall of raw
 * markdown. This helper strips emphasis markers, lifts the first clause,
 * and caps to a reasonable length so the TL;DR is what it claims to be.
 */
function buildTldr(hypothesis: string, maxLen = 220): string {
  if (!hypothesis) return "";
  // Drop markdown emphasis markers; keep the underlying word.
  let s = hypothesis
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*\n]+)\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
  // Strip a leading "TL;DR — " if the model emitted one verbatim.
  s = s.replace(/^TL[;:]?DR\s*[—\-:]\s*/i, "");
  // Take the first real sentence. Parentheses with commas inside fool a
  // naive split, so we look for ".", "!", or "?" followed by whitespace
  // (or end of string).
  const m = s.match(/^[^.!?]+[.!?](?=\s|$)/);
  let head = (m ? m[0] : s).trim();
  if (head.length > maxLen) {
    head = head.slice(0, maxLen - 1).trimEnd() + "…";
  } else if (!/[.!?…]$/.test(head)) {
    head += ".";
  }
  return head;
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function GeniePage() {
  const router = useRouter();
  const { token } = useAuthStore();
  // Narrow selectors so the page doesn't re-render every time the
  // JobsPanel polls (e.g. while a podcast generates in the background).
  const addGenieJob = useJobsStore((s) => s.addGenieJob);
  const genieJobs = useJobsStore((s) => s.genieJobs);
  const { selectedTopics, activeSubject, getPrimaryNamespaceKey } = useNamespaceStore();
  const searchParams = useSearchParams();
  // Feature gates — hide UI for any Genie sub-feature the admin has
  // turned off. The /genie/* router would already 404 the request, but
  // hiding the button is the cleaner UX so the user never sees a dead
  // control that vanishes when clicked.
  const combineEnabled = useFeature("genie_combine_enabled", true);
  const autoEnabled = useFeature("genie_auto_enabled", true);
  const [mode, setMode] = useState<Mode>("manual");
  // Fall back to manual when the user lands in / has stale auto mode
  // and the admin has since disabled auto-discovery.
  useEffect(() => {
    if (!autoEnabled && mode === "auto") setMode("manual");
  }, [autoEnabled, mode]);
  const [activeTab, setActiveTab] = useState<Tab>(
    searchParams.get("tab") === "discoveries" ? "discoveries" : "cauldron"
  );

  const [elements, setElements] = useState<GenieElement[]>([]);
  const [elemSearch, setElemSearch] = useState("");
  const [cauldron, setCauldron] = useState<GenieElement[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [streamLog, setStreamLog] = useState<StreamEvent[]>([]);
  const [elaborationSections, setElaborationSections] = useState<ElaborationSections>({});
  const [bgJobId, setBgJobId] = useState<string | null>(null);
  const [bgStatus, setBgStatus] = useState<string | null>(null);
  const synthAbortRef = useRef<AbortController | null>(null);
  // Abort the synthesize stream on unmount so navigating away doesn't
  // leave the fetch reader hanging or cause stale state writes.
  useEffect(() => () => synthAbortRef.current?.abort(), []);

  const [capsules, setCapsules] = useState<IdeaCapsule[]>([]);
  // `capsulesLoading` is true on first mount until the initial fetch resolves,
  // and stays false on subsequent refetches (so the grid doesn't flicker into
  // skeletons on every poll). The ghost-card grid is gated on this flag.
  const [capsulesLoading, setCapsulesLoading] = useState(true);
  const [chatCapsule, setChatCapsule] = useState<IdeaCapsule | null>(null);

  // ── Idea-combine multi-select state ────────────────────────────────────────
  // Combine is a deliberate action — the page stays clean by default and only
  // shows per-card checkboxes when the user enters ``combineMode`` via the
  // header toggle. This avoids the persistent-checkbox clutter that made the
  // ideas list feel like a checkout flow.
  const COMBINE_MAX = 3;
  const [combineMode, setCombineMode] = useState(false);
  const [combineSelected, setCombineSelected] = useState<string[]>([]);
  const [combineBusy, setCombineBusy] = useState(false);
  const [combineErr, setCombineErr] = useState<string | null>(null);
  // Hybrid client-side search over the Ideas list — filters on title,
  // hypothesis, open_questions, and source_query as the user types.
  const [ideaSearch, setIdeaSearch] = useState("");
  const toggleCombineSelect = useCallback((capsuleId: string) => {
    setCombineSelected(prev => {
      if (prev.includes(capsuleId)) return prev.filter(id => id !== capsuleId);
      if (prev.length >= COMBINE_MAX) return prev; // hard cap — UI also disables further selection
      return [...prev, capsuleId];
    });
  }, []);
  const clearCombineSelection = useCallback(() => {
    setCombineSelected([]);
    setCombineErr(null);
  }, []);
  const exitCombineMode = useCallback(() => {
    setCombineMode(false);
    setCombineSelected([]);
    setCombineErr(null);
  }, []);

  async function startCombine() {
    if (combineBusy) return;
    if (combineSelected.length < 2 || combineSelected.length > COMBINE_MAX) return;
    setCombineBusy(true);
    setCombineErr(null);
    try {
      const queued = await api.post<{ session_id: string; status: string; parent_ids: string[] }>(
        "/genie/capsules/combine",
        { capsule_ids: combineSelected },
      );
      if (!queued?.session_id) {
        setCombineErr("Combine could not be queued.");
        setCombineBusy(false);
        return;
      }
      // Register the combine run with the global Genie-jobs store — the
      // jobs panel polls and shows the toast/notification automatically,
      // matching the Auto / Query / Manual-Synthesize background UX. No
      // inline polling here; the user can leave the page.
      addGenieJob({
        session_id: queued.session_id,
        status: "running",
        capsule_id: null,
        error: null,
        created_at: new Date().toISOString(),
        completed_at: null,
        label: "Combine ideas",
      });
      setActiveTab("discoveries");
      exitCombineMode();
      setCombineBusy(false);
    } catch (e: unknown) {
      const err = e as { status?: number; detail?: string | { message?: string }; message?: string };
      const detail = typeof err?.detail === "string" ? err.detail : err?.detail?.message;
      setCombineErr(detail || err?.message || "Combine request failed.");
      setCombineBusy(false);
    }
  }

  const [autoStatus, setAutoStatus] = useState<AutoStatus | null>(null);
  const [autoBatchRunning, setAutoBatchRunning] = useState(false);
  const [autoBatchMsg, setAutoBatchMsg] = useState<string | null>(null);

  // Query mode state
  const [queryInput, setQueryInput] = useState("");
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryResult, setQueryResult] = useState<QueryDiscoverResult | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);

  // Threshold controls — shared between manual and auto mode.
  // Always start with DEFAULT_THRESHOLDS on first render so the
  // server-rendered HTML matches the client; we hydrate from
  // localStorage in a useEffect after mount to avoid hydration
  // mismatches (which can crash the page during HMR).
  const [thresholds, setThresholds] = useState<typeof DEFAULT_THRESHOLDS>(DEFAULT_THRESHOLDS);
  const [showConstraints, setShowConstraints] = useState(false);

  useEffect(() => {
    try {
      const saved = typeof window !== "undefined" ? localStorage.getItem(THRESHOLDS_KEY) : null;
      if (saved) setThresholds({ ...DEFAULT_THRESHOLDS, ...JSON.parse(saved) });
    } catch {
      // ignore — corrupt JSON or quota issues fall back to defaults
    }
  }, []);

  function updateThreshold(key: keyof typeof DEFAULT_THRESHOLDS, value: number) {
    setThresholds((prev: typeof DEFAULT_THRESHOLDS) => {
      const next = { ...prev, [key]: value };
      try { localStorage.setItem(THRESHOLDS_KEY, JSON.stringify(next)); } catch {}
      return next;
    });
  }

  function resetThresholds() {
    setThresholds(DEFAULT_THRESHOLDS);
    try { localStorage.removeItem(THRESHOLDS_KEY); } catch {}
  }

  // Folder filtering for manual mode
  const [folders, setFolders] = useState<BookmarkFolder[]>([]);
  const [selectedFolderIds, setSelectedFolderIds] = useState<Set<string>>(new Set());
  // paper_id → folder IDs map (loaded from bookmarks)
  const [paperFolderMap, setPaperFolderMap] = useState<Map<string, Set<string>>>(new Map());
  const [showFolderDropdown, setShowFolderDropdown] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const bgPollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Active subject label for display
  const activeSubjectLabel = NAMESPACE_TREE.find(s => s.key === activeSubject)?.label ?? activeSubject;

  // Helper: build the capsule list URL scoped to the current subscriptions so
  // ideas from deselected subjects are hidden automatically.
  const capsulesUrl = selectedTopics.length > 0
    ? `/genie/capsules?namespace_keys=${encodeURIComponent(selectedTopics.join(","))}`
    : "/genie/capsules";

  useEffect(() => {
    const qs = new URLSearchParams();
    if (selectedTopics.length) qs.set("namespace_keys", selectedTopics.join(","));
    qs.set("bookmarks_only", "true");
    // SWR pattern — paint cached snapshot instantly when a fresh one
    // exists, then refresh in the background. The capsule list is
    // expensive to render with all the score bars; serving it cached
    // makes tab-revisits feel instant.
    import("@/lib/swr").then(({ swrGet }) => {
      swrGet<GenieElement[]>(
        `genie:elements:${qs.toString()}`,
        `/genie/elements?${qs}`,
        (v) => setElements(v),
        45_000,
      ).catch(() => {});

      setCapsulesLoading(true);
      swrGet<IdeaCapsule[]>(
        `genie:capsules:${selectedTopics.join(",")}`,
        capsulesUrl,
        (v, source) => {
          setCapsules(v);
          // Stop showing skeletons as soon as ANY data arrives
          // (cache or network).
          if (source === "cache" || source === "network") setCapsulesLoading(false);
        },
        45_000,
      ).catch(() => setCapsulesLoading(false));
    });
    api.get<AutoStatus>("/genie/auto-status").then(setAutoStatus).catch(() => {});
    api.get<BookmarkFolder[]>("/bookmarks/folders").then(setFolders).catch(() => {});
    api.get<Bookmark[]>("/bookmarks").then(data => {
      const map = new Map<string, Set<string>>();
      for (const b of (Array.isArray(data) ? data : [])) {
        if (b.paper?.id && b.folder_ids?.length) {
          map.set(b.paper.id, new Set(b.folder_ids));
        }
      }
      setPaperFolderMap(map);
    }).catch(() => {});
  }, [selectedTopics]); // eslint-disable-line react-hooks/exhaustive-deps

  // Refresh capsules whenever discoveries tab becomes active
  useEffect(() => {
    if (activeTab === "discoveries") {
      api.get<IdeaCapsule[]>(capsulesUrl).then(setCapsules).catch(() => {});
    }
  }, [activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  // When a genie job transitions to "done", refresh capsules and switch to discoveries
  const prevGenieJobsRef = useRef<typeof genieJobs>([]);
  useEffect(() => {
    const prev = prevGenieJobsRef.current;
    const justCompleted = genieJobs.filter(
      (gj) =>
        gj.status === "done" &&
        prev.find((p) => p.session_id === gj.session_id)?.status !== "done"
    );
    if (justCompleted.length > 0) {
      api.get<IdeaCapsule[]>(capsulesUrl).then(setCapsules).catch(() => {});
      setActiveTab("discoveries");
    }
    prevGenieJobsRef.current = genieJobs;
  }, [genieJobs]); // eslint-disable-line react-hooks/exhaustive-deps

  // Background job polling
  useEffect(() => {
    if (!bgJobId || bgStatus === "done" || bgStatus === "failed") return;
    bgPollRef.current = setTimeout(async () => {
      try {
        const data = await api.get<{ status: string; capsule_id: string | null }>(
          `/genie/sessions/${bgJobId}`
        );
        setBgStatus(data.status);
        if (data.status === "done") {
          api.get<IdeaCapsule[]>(capsulesUrl).then(setCapsules).catch(() => {});
          setActiveTab("discoveries");
        }
      } catch {}
    }, 3000);
    return () => { if (bgPollRef.current) clearTimeout(bgPollRef.current); };
  }, [bgJobId, bgStatus]); // eslint-disable-line react-hooks/exhaustive-deps

  // Max cauldron size: up to 10 for all modes now that backend supports it
  const maxCauldron = 10;

  function addToCauldron(el: GenieElement) {
    if (cauldron.find((c) => c.id === el.id)) return;
    // Dedup by paper_id so the same physical paper (different namespaces) isn't added twice
    if (el.paper_id && cauldron.find((c) => c.paper_id === el.paper_id)) return;
    if (cauldron.length >= maxCauldron) return;
    setCauldron((c) => [...c, el]);
  }

  function removeFromCauldron(id: string) {
    setCauldron((c) => c.filter((el) => el.id !== id));
  }

  async function runAutoBatch() {
    const alreadyRunning = genieJobs.some(
      (gj) => gj.status === "pending" || gj.status === "running"
    );
    if (alreadyRunning) {
      setAutoBatchMsg("A synthesis job is already in progress. Wait for it to complete first.");
      return;
    }
    setAutoBatchRunning(true);
    setAutoBatchMsg(null);
    try {
      const qs = new URLSearchParams({
        sem_threshold: thresholds.semThreshold.toString(),
        jac_threshold: thresholds.jacThreshold.toString(),
        temperature: thresholds.temperature.toString(),
      });
      if (selectedTopics.length > 0) qs.set("namespace_keys", selectedTopics.join(","));
      const res = await api.post<{ queued: number; session_ids?: string[]; message: string }>(
        `/genie/auto-batch?${qs}`, {}
      );
      setAutoBatchMsg(res.message);
      if (res.queued > 0 && res.session_ids?.length) {
        const now = new Date().toISOString();
        res.session_ids.forEach((sid) => {
          addGenieJob({
            session_id: sid,
            status: "running",
            capsule_id: null,
            error: null,
            created_at: now,
            completed_at: null,
            label: "Auto Genie Job",
          });
        });
      }
    } catch {
      setAutoBatchMsg("Failed to trigger auto-synthesis. Try again.");
    }
    setAutoBatchRunning(false);
  }

  async function runQueryDiscover(autoSynthesize = false) {
    if (!queryInput.trim() || queryLoading) return;
    // Enforce single background job limit across all Genie modes
    if (autoSynthesize) {
      const alreadyRunning = genieJobs.some(gj => gj.status === "pending" || gj.status === "running");
      if (alreadyRunning) {
        setQueryError("A synthesis job is already in progress. Wait for it to complete first.");
        return;
      }
    }
    setQueryLoading(true);
    setQueryResult(null);
    setQueryError(null);
    autoSelectedRef.current = new Set(); // reset auto-selection tracking for new query
    try {
      const qs = new URLSearchParams({
        query: queryInput.trim(),
        limit: "15",
        auto_synthesize: autoSynthesize ? "true" : "false",
      });
      if (selectedTopics.length > 0) qs.set("namespace_keys", selectedTopics.join(","));
      const res = await api.post<QueryDiscoverResult>(`/genie/query-discover?${qs}`, {});
      if (res.error) {
        setQueryError(res.error);
      } else {
        setQueryResult(res);
        if (autoSynthesize && res.session_id) {
          const now = new Date().toISOString();
          addGenieJob({
            session_id: res.session_id,
            status: "running",
            capsule_id: null,
            error: null,
            created_at: now,
            completed_at: null,
            label: `Query: ${queryInput.trim().slice(0, 40)}`,
          });
          setActiveTab("discoveries");
        }
      }
    } catch (err) {
      setQueryError(err instanceof Error ? err.message : "Query discover failed.");
    }
    setQueryLoading(false);
  }

  function addQueryPaperToCauldron(paper: { paper_id: string; title: string }) {
    const existing = elements.find(e => e.paper_id === paper.paper_id);
    if (existing) {
      addToCauldron(existing);
    } else {
      setQueryError(`"${paper.title.slice(0, 40)}…" is not bookmarked yet. Bookmark it from the Feed first, then come back.`);
    }
  }

  // Track per-paper loading state while the element is being created on the backend
  const [creatingElementFor, setCreatingElementFor] = useState<Set<string>>(new Set());
  // Track which paper_ids have been auto-selected for the current query result
  const autoSelectedRef = useRef<Set<string>>(new Set());

  async function toggleQueryPaperInCauldron(paper: { paper_id: string; title: string }) {
    // First check the local elements library
    const existing = elements.find(e => e.paper_id === paper.paper_id);

    if (existing) {
      const inCauldron = !!cauldron.find(c => c.id === existing.id);
      if (inCauldron) removeFromCauldron(existing.id);
      else if (cauldron.length < maxCauldron) addToCauldron(existing);
      return;
    }

    // Paper not in library → create element on the fly (query mode searches full feed)
    if (creatingElementFor.has(paper.paper_id)) return; // already in flight
    setCreatingElementFor(prev => new Set([...prev, paper.paper_id]));
    try {
      const el = await api.post<{ id: string; label: string; type: string; paper_id: string }>(
        `/genie/elements/from-paper/${paper.paper_id}`
      );
      const newEl: GenieElement = { id: el.id, label: el.label, type: "paper", paper_id: el.paper_id };
      // Add to local elements list so subsequent interactions are instant
      setElements(prev => [...prev.filter(e => e.id !== newEl.id), newEl]);
      if (cauldron.length < maxCauldron) addToCauldron(newEl);
    } catch (err) {
      setQueryError(err instanceof Error ? err.message : "Failed to add paper.");
    } finally {
      setCreatingElementFor(prev => { const next = new Set(prev); next.delete(paper.paper_id); return next; });
    }
  }

  // Auto-select all best_group papers when query results arrive.
  // Bookmarked papers are added immediately; non-bookmarked ones trigger on-the-fly element creation.
  useEffect(() => {
    if (!queryResult?.best_group?.length) return;
    queryResult.best_group.forEach(paper => {
      if (autoSelectedRef.current.has(paper.paper_id)) return;
      autoSelectedRef.current.add(paper.paper_id);
      const existing = elements.find(e => e.paper_id === paper.paper_id);
      if (existing) {
        if (!cauldron.find(c => c.id === existing.id || (c.paper_id && c.paper_id === existing.paper_id))) {
          addToCauldron(existing);
        }
      } else {
        // Paper not bookmarked — create element on-the-fly so it joins the cauldron
        toggleQueryPaperInCauldron(paper);
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryResult?.best_group]);

  async function synthesize(background = false) {
    if (streaming) return;
    if (cauldron.length < 2) return;

    const geniePayload = {
      seed_element_ids: cauldron.map((e) => e.id),
      namespace_key: getPrimaryNamespaceKey(),
      sem_threshold: thresholds.semThreshold,
    };

    if (background) {
      const alreadyRunning = genieJobs.some(
        (gj) => gj.status === "pending" || gj.status === "running"
      );
      if (alreadyRunning) return;
      try {
        const data = await api.post<{ session_id: string; status: string }>(
          `/genie/synthesize-bg`, geniePayload
        );
        setBgJobId(data.session_id);
        setBgStatus(data.status);
        addGenieJob({
          session_id: data.session_id,
          status: "running",
          capsule_id: null,
          error: null,
          created_at: new Date().toISOString(),
          completed_at: null,
          label: "Custom Genie Job",
        });
        // Match Auto / Query / Combine UX — surface the job-in-progress on
        // the Ideas tab so the user sees the running pill and the freshly
        // synthesized capsule lands there when complete, instead of
        // streaming output into the Cauldron panel.
        setActiveTab("discoveries");
      } catch {}
      return;
    }

    setStreaming(true);
    setStreamLog([]);
    setElaborationSections({});
    setActiveTab("cauldron");

    const apiBase = process.env.NEXT_PUBLIC_API_URL || "";
    const url = `${apiBase}/api/v1/genie/synthesize`;

    synthAbortRef.current?.abort();
    const ctrl = new AbortController();
    synthAbortRef.current = ctrl;

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify(geniePayload),
        signal: ctrl.signal,
      });
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of decoder.decode(value).split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event: StreamEvent = JSON.parse(line.slice(6));
            if (event.type === "elaboration_section" && event.section && event.content) {
              setElaborationSections((s) => ({ ...s, [event.section!]: event.content! }));
            } else {
              setStreamLog((l) => [...l, event]);
            }
            if (event.type === "done") {
              setStreaming(false);
              api.get<IdeaCapsule[]>(capsulesUrl).then(setCapsules).catch(() => {});
            }
          } catch {}
        }
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      }
    } catch (err) {
      setStreamLog((l) => [...l, { type: "error", message: String(err) }]);
    }
    setStreaming(false);
  }

  async function deleteCapsule(id: string) {
    try {
      await api.delete(`/genie/capsules/${id}`);
      setCapsules((cs) => cs.filter((c) => c.id !== id));
      if (chatCapsule?.id === id) setChatCapsule(null);
    } catch (err) {
      console.error("delete capsule failed", err);
      alert("Failed to delete. Please try again.");
    }
  }

  async function saveCapsule(id: string) {
    await api.patch(`/genie/capsules/${id}/status?status=saved`);
    setCapsules((cs) => cs.map((c) => c.id === id ? { ...c, status: "saved" } : c));
  }

  // Deduplicate query result papers by source_url (same arXiv paper can appear in multiple namespaces).
  // When deduping, prefer the entry that belongs to best_group, then the one already in the user's library,
  // then the one with higher query relevance.
  const dedupedQueryPapers = useMemo(() => {
    if (!queryResult?.papers?.length) return [];
    const seen = new Map<string, typeof queryResult.papers[0]>();
    for (const p of queryResult.papers) {
      const key = p.source_url || p.paper_id;
      const existing = seen.get(key);
      if (!existing) {
        seen.set(key, p);
        continue;
      }
      const pInBest = queryResult.best_group?.some(g => g.paper_id === p.paper_id) ?? false;
      const exInBest = queryResult.best_group?.some(g => g.paper_id === existing.paper_id) ?? false;
      const pInLib = elements.some(e => e.paper_id === p.paper_id);
      const exInLib = elements.some(e => e.paper_id === existing.paper_id);
      // Priority: best_group > in-library > higher relevance
      if (
        (pInBest && !exInBest) ||
        (!pInBest && !exInBest && pInLib && !exInLib) ||
        (pInBest === exInBest && pInLib === exInLib && p.query_relevance > existing.query_relevance)
      ) {
        seen.set(key, p);
      }
    }
    return [...seen.values()].sort((a, b) => b.query_relevance - a.query_relevance);
  }, [queryResult?.papers, queryResult?.best_group, elements]);

  // Dedup the best_group display list too (same physical paper across namespaces)
  const dedupedBestGroup = useMemo(() => {
    if (!queryResult?.best_group?.length) return [];
    const seen = new Set<string>();
    return queryResult.best_group.filter(p => {
      const key = p.source_url || p.paper_id;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [queryResult?.best_group]);

  const filteredElements = elements.filter((e) => {
    if (!e.label.toLowerCase().includes(elemSearch.toLowerCase())) return false;
    if (selectedFolderIds.size === 0) return true;
    if (!e.paper_id) return false;
    const fids = paperFolderMap.get(e.paper_id);
    return fids ? [...fids].some(fid => selectedFolderIds.has(fid)) : false;
  });

  return (
    <div className="flex h-full overflow-hidden bg-gray-950">
      {/* Chat overlay */}
      <AnimatePresence>
        {chatCapsule && (
          <CapsuleChatOverlay
            capsule={chatCapsule}
            token={token || ""}
            onClose={() => setChatCapsule(null)}
          />
        )}
      </AnimatePresence>

      {/* ── Left: Source panel ─────────────────────────────────────────── */}
      <aside className="w-64 shrink-0 border-r border-white/5 flex flex-col">
        <div className="p-4 border-b border-white/5">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
              <FlaskConicalIcon size={13} className="text-white" />
            </div>
            <span className="text-sm font-bold text-white">Genie</span>
          </div>

          <div className="flex bg-gray-900 rounded-xl p-0.5 gap-0.5">
            {(([
              { key: "manual", icon: "⚗️", label: "Manual" },
              // Auto tab hides when the admin disables the auto-discovery
              // sub-feature; manual/query keep working in either case.
              ...(autoEnabled ? [{ key: "auto" as Mode, icon: "⚡", label: "Auto" }] : []),
              { key: "query",  icon: "🔍", label: "Query"  },
            ]) as { key: Mode; icon: string; label: string }[]).map(({ key, icon, label }) => (
              <button
                key={key}
                onClick={() => {
                  setMode(key);
                  // Cauldron is only for manual — switch tabs accordingly
                  if (key !== "manual") setActiveTab("discoveries");
                  else setActiveTab("cauldron");
                }}
                className="flex-1 py-1.5 rounded-[10px] text-[11px] font-semibold transition-all"
                style={mode === key ? {
                  background: key === "query"
                    ? "linear-gradient(135deg,#7c3aed,#a855f7)"
                    : "linear-gradient(135deg,#6366f1,#7c3aed)",
                  color: "#fff",
                  boxShadow: "0 2px 8px rgba(99,102,241,0.30)",
                } : { color: "var(--rf-text4)" }}
              >
                {icon} {label}
              </button>
            ))}
          </div>
        </div>

        {mode === "query" ? (
          /* ── Query mode panel ── */
          <div className="flex-1 flex flex-col overflow-hidden p-3 gap-3">
            <div className="rounded-xl p-3" style={{ background: "var(--rf-surface3)", border: "1px solid var(--rf-border2)" }}>
              <div className="flex items-center gap-2 mb-1.5">
                <BrainIcon size={12} style={{ color: "#7c3aed" }} />
                <p className="text-xs font-semibold" style={{ color: "#7c3aed" }}>Query Mode</p>
              </div>
              <p className="text-[10px] leading-relaxed" style={{ color: "var(--rf-text4)" }}>
                Describe a research topic. Genie finds relevant papers, scores compatibility, and surfaces the best synthesis group.
              </p>
            </div>

            {/* Query input */}
            <div
              className="flex flex-col gap-2 rounded-xl border p-2.5 transition-colors"
              style={{
                background: "var(--rf-input)",
                borderColor: queryLoading ? "rgba(139,92,246,0.5)" : "var(--rf-input-border)",
              }}
            >
              <textarea
                value={queryInput}
                onChange={e => setQueryInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runQueryDiscover(); } }}
                placeholder="e.g. methods for efficient LLM fine-tuning on small datasets…"
                rows={3}
                className="w-full bg-transparent text-xs outline-none resize-none leading-relaxed"
                style={{ color: "var(--rf-text2)" }}
              />
              <button
                onClick={() => runQueryDiscover()}
                disabled={queryLoading || queryInput.trim().length < 3}
                className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-[11px] font-semibold transition-all disabled:opacity-40"
                style={{
                  background: "linear-gradient(135deg,#7c3aed,#a855f7)",
                  color: "#fff",
                  boxShadow: "0 2px 6px rgba(124,58,237,0.28)",
                }}
              >
                {queryLoading
                  ? <><Loader2Icon size={11} className="animate-spin" /> Searching…</>
                  : <><SearchIcon size={11} /> Discover Papers</>
                }
              </button>
            </div>

            {/* Wave animation while loading */}
            {queryLoading && (
              <div style={{ height: 2, overflow: "hidden", borderRadius: 1, marginTop: -8 }}>
                <div style={{
                  height: "100%",
                  background: "linear-gradient(90deg, transparent, #8b5cf6, #c084fc, #8b5cf6, transparent)",
                  backgroundSize: "200% 100%",
                  animation: "deepwave 1.5s linear infinite",
                }} />
              </div>
            )}

            {/* Error */}
            {queryError && (
              <p className="text-[10px] text-red-400 px-1 leading-relaxed">{queryError}</p>
            )}

            {/* Results */}
            {queryResult && !queryError && (
              <div className="flex-1 overflow-y-auto space-y-1.5">
                {queryResult.rewritten_query && queryResult.rewritten_query !== queryInput.trim() && (
                  <p className="text-[9px] text-gray-600 px-0.5 italic leading-relaxed">
                    Searched: &ldquo;{queryResult.rewritten_query}&rdquo;
                  </p>
                )}

                {dedupedBestGroup.length >= 2 && (
                  <div className="rounded-xl border border-violet-700/30 bg-violet-950/20 p-2.5 mb-2">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[10px] font-semibold text-violet-300">Best group · {dedupedBestGroup.length} papers</span>
                      <span className="text-[9px] text-gray-600">score {Math.round(queryResult.best_group_score * 100)}%</span>
                    </div>
                    <button
                      onClick={() => synthesize()}
                      disabled={queryLoading || streaming || cauldron.length < 2}
                      className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-[10px] font-semibold transition-all disabled:opacity-40
                        bg-gradient-to-r from-indigo-700/80 to-violet-700/80 hover:from-indigo-600 hover:to-violet-600 text-white"
                    >
                      <ZapIcon size={10} /> Synthesize Group
                    </button>
                  </div>
                )}

                <p className="text-[9px] font-bold text-gray-600 uppercase tracking-wider px-0.5">
                  {dedupedQueryPapers.length} Papers Found{queryResult.papers.length > dedupedQueryPapers.length ? ` (${queryResult.papers.length - dedupedQueryPapers.length} duplicates hidden)` : ""}
                </p>

                {dedupedQueryPapers.map(paper => {
                  const elemInLib = elements.find(e => e.paper_id === paper.paper_id);
                  const inCauldron = !!(elemInLib && cauldron.find(c => c.id === elemInLib.id || (c.paper_id && c.paper_id === elemInLib.paper_id)));
                  const inBestGroup = queryResult.best_group.some(g => g.paper_id === paper.paper_id || g.source_url === paper.source_url);
                  const rel = Math.round((paper.query_relevance ?? 0) * 100);
                  const relColor = rel >= 65 ? "#34d399" : rel >= 40 ? "#fbbf24" : "#6b7280";
                  const isCreating = creatingElementFor.has(paper.paper_id);
                  // Build a tooltip: prefer TL;DR, fall back to first 160 chars of abstract
                  const tooltip = (paper.tldr ?? (paper.abstract ? paper.abstract.slice(0, 160).trim() + "…" : "")) || paper.title;
                  return (
                    <div
                      key={paper.paper_id}
                      title={tooltip}
                      className="rounded-lg border p-2 transition-all cursor-default"
                      style={{
                        borderColor: inCauldron
                          ? "rgba(99,102,241,0.45)"
                          : inBestGroup
                          ? "rgba(139,92,246,0.30)"
                          : "rgba(255,255,255,0.06)",
                        background: inCauldron
                          ? "rgba(99,102,241,0.10)"
                          : inBestGroup
                          ? "rgba(139,92,246,0.08)"
                          : "rgba(17,24,39,0.4)",
                      }}
                    >
                      {/* Header row: badge + title */}
                      <div className="flex items-start gap-1.5 mb-1.5">
                        {/* Best-group indicator / selected checkmark */}
                        {inCauldron ? (
                          <div className="mt-0.5 flex-shrink-0 w-3.5 h-3.5 rounded-full bg-indigo-600 flex items-center justify-center">
                            <CheckIcon size={8} className="text-white" strokeWidth={3} />
                          </div>
                        ) : inBestGroup ? (
                          <div className="mt-0.5 flex-shrink-0 w-3 h-3 rounded-full bg-violet-600/40 border border-violet-500/40 flex items-center justify-center">
                            <ZapIcon size={7} className="text-violet-400" />
                          </div>
                        ) : null}
                        <p className="flex-1 text-[10px] leading-tight" style={{ color: inCauldron ? "#c7d2fe" : "var(--rf-text3)" }}>
                          {paper.title.slice(0, 72)}{paper.title.length > 72 ? "…" : ""}
                        </p>
                      </div>

                      {/* Footer row: namespace + relevance + actions */}
                      <div className="flex items-center justify-between gap-1">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[8px] font-mono" style={{ color: "var(--rf-text5)" }}>{paper.namespace_key}</span>
                          <div className="flex items-center gap-1" title={`Query relevance: ${rel}%`}>
                            <div style={{ width: 24, height: 3, background: "rgba(55,65,81,0.5)", borderRadius: 1.5, overflow: "hidden" }}>
                              <div style={{ width: `${rel}%`, height: "100%", background: relColor, borderRadius: 1.5 }} />
                            </div>
                            <span style={{ fontSize: 8, color: relColor }}>{rel}%</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-1 flex-shrink-0">
                          {/* View paper button */}
                          {paper.source_url && (
                            <a
                              href={paper.source_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              title="View paper on arXiv"
                              className="text-[9px] px-1.5 py-0.5 rounded transition-all"
                              style={{ color: "#6b7280", background: "rgba(55,65,81,0.3)", border: "1px solid rgba(55,65,81,0.4)" }}
                              onClick={e => e.stopPropagation()}
                            >
                              <ExternalLinkIcon size={9} />
                            </a>
                          )}
                          {/* Toggle selection button */}
                          <button
                            onClick={() => toggleQueryPaperInCauldron(paper)}
                            disabled={isCreating || (!inCauldron && cauldron.length >= maxCauldron)}
                            title={
                              isCreating ? "Adding…"
                              : inCauldron ? "Click to deselect"
                              : cauldron.length >= maxCauldron ? `Cauldron full (${maxCauldron} max)`
                              : "Click to select"
                            }
                            className="text-[9px] px-1.5 py-0.5 rounded font-medium transition-all flex items-center gap-1"
                            style={
                              inCauldron
                                ? { color: "#818cf8", background: "rgba(99,102,241,0.18)", border: "1px solid rgba(99,102,241,0.35)" }
                                : { color: "#6b7280", background: "rgba(55,65,81,0.3)", border: "1px solid rgba(55,65,81,0.4)" }
                            }
                          >
                            {isCreating
                              ? <><Loader2Icon size={8} className="animate-spin" /> Adding…</>
                              : inCauldron ? "✓ Selected" : "+ Select"}
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {queryResult && dedupedQueryPapers.length === 0 && !queryError && (
              <p className="text-[11px] text-gray-600 text-center py-6 leading-relaxed">
                No papers found. Try a different query or refresh the feed first.
              </p>
            )}
          </div>
        ) : mode === "manual" ? (
          <div className="flex-1 flex flex-col overflow-hidden p-3">
            <p className="text-[10px] font-bold text-gray-600 uppercase tracking-wider mb-2">
              Your Bookmarks
            </p>

            {/* Folder filter */}
            {folders.length > 0 && (
              <div className="relative mb-2">
                <button
                  onClick={() => setShowFolderDropdown(v => !v)}
                  className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs border transition-all ${
                    selectedFolderIds.size > 0
                      ? "bg-orange-950/30 border-orange-700/30 text-orange-300"
                      : "bg-gray-900/60 border-white/5 text-gray-500 hover:text-gray-300"
                  }`}
                >
                  <FolderIcon size={11} />
                  <span className="flex-1 text-left">
                    {selectedFolderIds.size > 0
                      ? `${selectedFolderIds.size} folder${selectedFolderIds.size > 1 ? "s" : ""} selected`
                      : "All folders"}
                  </span>
                  {selectedFolderIds.size > 0 && (
                    <button onClick={(e) => { e.stopPropagation(); setSelectedFolderIds(new Set()); }}
                      className="text-orange-400/60 hover:text-orange-300 p-0.5">
                      <XIcon size={10} />
                    </button>
                  )}
                </button>
                {showFolderDropdown && (
                  <div className="absolute left-0 right-0 top-full mt-1 z-30 bg-gray-900 border border-white/10 rounded-xl shadow-xl overflow-hidden">
                    {folders.map(f => {
                      const checked = selectedFolderIds.has(f.id);
                      return (
                        <button key={f.id}
                          onClick={() => {
                            setSelectedFolderIds(prev => {
                              const next = new Set(prev);
                              next.has(f.id) ? next.delete(f.id) : next.add(f.id);
                              return next;
                            });
                          }}
                          className="w-full flex items-center gap-2.5 px-3 py-2 hover:bg-white/5 transition-colors"
                        >
                          <div className="w-2 h-2 rounded-sm flex-shrink-0" style={{ backgroundColor: f.color || "#6366f1" }} />
                          <span className="flex-1 text-left text-xs text-gray-300 truncate">{f.name}</span>
                          <div className={`w-3.5 h-3.5 rounded border flex items-center justify-center flex-shrink-0 ${checked ? "bg-orange-500 border-orange-500" : "border-gray-700"}`}>
                            {checked && <CheckIcon size={9} className="text-white" strokeWidth={3} />}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            <div className="flex items-center gap-2 bg-gray-900 border border-white/5 rounded-lg px-2.5 py-1.5 mb-2">
              <SearchIcon size={11} className="text-gray-600 flex-shrink-0" />
              <input
                value={elemSearch}
                onChange={(e) => setElemSearch(e.target.value)}
                placeholder="Filter papers…"
                className="flex-1 bg-transparent text-xs text-gray-300 placeholder-gray-700 outline-none"
              />
              {elemSearch && (
                <button onClick={() => setElemSearch("")}>
                  <XIcon size={10} className="text-gray-600" />
                </button>
              )}
            </div>

            <div className="flex-1 overflow-y-auto space-y-1 pr-0.5">
              {filteredElements.length === 0 ? (
                <p className="text-xs text-gray-700 text-center py-8 leading-relaxed">
                  Bookmark papers on the Feed to see them here
                </p>
              ) : (
                filteredElements.map((el) => {
                  const inCauldron = !!cauldron.find((c) => c.id === el.id);
                  return (
                    <button
                      key={el.id}
                      onClick={() => (inCauldron ? removeFromCauldron(el.id) : addToCauldron(el))}
                      disabled={!inCauldron && cauldron.length >= maxCauldron}
                      title={el.tldr ? `${el.label}\n\n${el.tldr}` : el.label}
                      className={`w-full text-left px-2.5 py-2 rounded-lg text-xs transition-all border ${
                        inCauldron
                          ? "bg-indigo-950/60 border-indigo-500/30 text-indigo-200"
                          : "bg-gray-900/60 border-white/5 text-gray-400 hover:border-white/10 hover:text-gray-200"
                      } disabled:opacity-30`}
                    >
                      <div className="flex items-start gap-2">
                        <div className={`w-3.5 h-3.5 rounded flex-shrink-0 mt-0.5 border flex items-center justify-center ${
                          inCauldron ? "bg-indigo-600 border-indigo-500" : "border-gray-700"
                        }`}>
                          {inCauldron && <CheckIcon size={8} className="text-white" />}
                        </div>
                        <span className="leading-tight">{el.label.slice(0, 58)}{el.label.length > 58 ? "…" : ""}</span>
                      </div>
                    </button>
                  );
                })
              )}
            </div>

            <p className="text-[10px] text-gray-700 text-center mt-2 pt-2 border-t border-white/5">
              {cauldron.length}/{maxCauldron} in cauldron · {filteredElements.length}{selectedFolderIds.size > 0 ? `/${elements.length}` : ""} bookmarks
            </p>
          </div>
        ) : (
          /* ── Auto mode panel ── */
          <div className="flex-1 flex flex-col p-4 gap-4 overflow-y-auto">
            <div className="rounded-xl border border-indigo-800/20 bg-indigo-950/15 p-4">
              <div className="flex items-center gap-2 mb-2">
                <ZapIcon size={13} className="text-indigo-400" />
                <p className="text-xs font-semibold text-indigo-300">Auto Discovery</p>
              </div>
              <p className="text-[11px] text-gray-500 leading-relaxed">
                Genie scans your bookmarks, clusters them by shared concepts and semantic similarity, and synthesizes novel research ideas across compatible papers.
              </p>
            </div>

            <div className="rounded-xl border border-white/5 bg-gray-900/40 p-3 text-center">
              <p className="text-2xl font-bold text-white mb-0.5">
                {capsules.filter(c => c.source_mode === "auto" || c.is_scout_generated).length}
              </p>
              <p className="text-[10px] text-gray-600">Auto Ideas</p>
            </div>

            <button
              onClick={runAutoBatch}
              disabled={autoBatchRunning}
              className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-40 text-white font-semibold rounded-xl px-4 py-2.5 text-sm transition-all shadow-lg shadow-indigo-900/40"
            >
              {autoBatchRunning
                ? <><Loader2Icon size={13} className="animate-spin" /> Running…</>
                : <><RefreshCwIcon size={13} /> Run Now</>
              }
            </button>

            {autoBatchMsg && (
              <motion.p
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-[11px] text-gray-400 text-center leading-relaxed px-1"
              >
                {autoBatchMsg}
              </motion.p>
            )}

            {/* Temperature indicator */}
            <div className="flex items-center gap-2 px-1">
              <ThermometerIcon size={11} className={
                thresholds.temperature >= 0.7 ? "text-orange-400" :
                thresholds.temperature >= 0.4 ? "text-indigo-400" : "text-sky-400"
              } />
              <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    thresholds.temperature >= 0.7 ? "bg-gradient-to-r from-amber-500 to-orange-500" :
                    thresholds.temperature >= 0.4 ? "bg-gradient-to-r from-indigo-500 to-violet-500" :
                    "bg-gradient-to-r from-sky-600 to-blue-500"
                  }`}
                  style={{ width: `${thresholds.temperature * 100}%` }}
                />
              </div>
              <span className="text-[10px] font-mono text-gray-600 w-6 text-right">
                {thresholds.temperature >= 0.7 ? "hot" : thresholds.temperature >= 0.4 ? "mid" : "cool"}
              </span>
            </div>

            <p className="text-[10px] text-gray-700 text-center leading-relaxed">
              Adjust Temperature in Constraints to control how adventurous Genie is when picking paper combinations.
            </p>
          </div>
        )}

        {/* ── Constraints & Thresholds ── */}
        <div className="border-t border-white/5 shrink-0">
          <button
            onClick={() => setShowConstraints((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-2.5 text-xs text-gray-600 hover:text-gray-300 transition-colors"
          >
            <div className="flex items-center gap-2">
              <SlidersHorizontalIcon size={11} />
              <span className="font-semibold">Constraints & Thresholds</span>
            </div>
            {showConstraints ? <ChevronUpIcon size={11} /> : <ChevronDownIcon size={11} />}
          </button>

          <AnimatePresence>
            {showConstraints && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="px-4 pb-4 space-y-3">
                  <TemperatureSlider
                    value={thresholds.temperature}
                    onChange={(v) => updateThreshold("temperature", v)}
                  />
                  <div className="border-t border-white/5 pt-3 space-y-3">
                    <ThresholdSlider
                      label="Semantic Similarity"
                      description="Min. embedding similarity between papers (manual gate + auto pairing)"
                      value={thresholds.semThreshold}
                      min={0.05} max={0.90} step={0.05}
                      onChange={(v) => updateThreshold("semThreshold", v)}
                    />
                    <ThresholdSlider
                      label="Concept Overlap"
                      description="Min. Jaccard overlap of key concepts for auto-batch pairing"
                      value={thresholds.jacThreshold}
                      min={0.00} max={0.50} step={0.05}
                      onChange={(v) => updateThreshold("jacThreshold", v)}
                    />
                  </div>
                  <button
                    onClick={resetThresholds}
                    className="w-full flex items-center justify-center gap-1.5 text-[11px] text-gray-600 hover:text-gray-300 py-1.5 rounded-lg hover:bg-white/5 transition-all"
                  >
                    <RotateCcwIcon size={10} />
                    Reset to defaults
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </aside>

      {/* ── Center ─────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Tabs — Cauldron only visible in Manual mode */}
        <div className="flex border-b border-white/5 px-6 bg-gray-950/90 backdrop-blur-sm shrink-0">
          {([
            // Only show Cauldron tab when in manual mode
            ...(mode === "manual" ? [{ key: "cauldron" as Tab, label: "🧪 Cauldron" }] : []),
            { key: "discoveries" as Tab, label: "💡 Ideas" },
          ]).map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === key
                  ? "border-indigo-500 text-indigo-300"
                  : "border-transparent text-gray-600 hover:text-gray-300"
              }`}
            >
              {label}
            </button>
          ))}

          {/* Combine-mode toggle — surfaces only on the Ideas tab and only
              when there are at least 2 ideas to combine. Hidden entirely
              when the admin has disabled the combine sub-feature. */}
          {activeTab === "discoveries" && capsules.length >= 2 && combineEnabled && (
            <button
              onClick={() => combineMode ? exitCombineMode() : setCombineMode(true)}
              className={
                "ml-auto self-center flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold transition-all " +
                (combineMode
                  ? "bg-fuchsia-950/60 border border-fuchsia-700/50 text-fuchsia-200"
                  : "bg-gray-900/60 border border-gray-800 text-gray-300 hover:text-fuchsia-200 hover:border-fuchsia-700/40")
              }
              title={combineMode ? "Exit combine mode" : "Select 2–3 ideas to fuse"}
            >
              <GitMergeIcon size={12} />
              {combineMode ? "Cancel combine" : "Combine ideas"}
            </button>
          )}

          <AnimatePresence>
            {bgJobId && bgStatus && bgStatus !== "done" && (
              <motion.div
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                className={`${activeTab === "discoveries" && capsules.length >= 2 ? "" : "ml-auto"} self-center flex items-center gap-2 bg-indigo-950/60 border border-indigo-700/40 rounded-full px-3 py-1`}
              >
                <Loader2Icon size={11} className="animate-spin text-indigo-400" />
                <span className="text-[11px] text-indigo-300 font-medium">
                  {bgStatus === "running" ? "Synthesizing in background…" : bgStatus}
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── Cauldron tab ─────────────────────────────────────────────── */}
        {activeTab === "cauldron" && (
          <div className="flex-1 overflow-y-auto p-6 space-y-5">
            <div className="rounded-2xl border border-white/5 bg-gradient-to-b from-gray-900/80 to-gray-900/40 p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-xl bg-gradient-to-br from-indigo-600/30 to-violet-600/30 border border-indigo-500/20 flex items-center justify-center">
                    <FlaskConicalIcon size={14} className="text-indigo-400" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-white leading-none">Idea Cauldron</p>
                    <p className="text-[10px] text-gray-600 mt-0.5">
                      {cauldron.length > 0 ? `${cauldron.length}/${maxCauldron} papers selected` : `Select 2–${maxCauldron} papers to synthesize`}
                    </p>
                  </div>
                </div>
                <span className="bg-gray-800/60 border border-white/10 rounded-lg px-2.5 py-1 text-xs text-gray-500 font-medium">
                  {activeSubjectLabel}
                </span>
              </div>

              <div className="min-h-16 flex flex-wrap gap-2 mb-4 p-3 rounded-xl bg-gray-950/50 border border-white/5">
                {cauldron.length === 0 ? (
                  <p className="text-xs text-gray-700 self-center">
                    Tick bookmarks on the left to add them
                  </p>
                ) : (
                  <AnimatePresence>
                    {cauldron.map((el) => (
                      <motion.div
                        key={el.id}
                        layout
                        initial={{ scale: 0.85, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.85, opacity: 0 }}
                        className="flex items-center gap-1.5 bg-indigo-950/60 border border-indigo-500/25 rounded-lg px-2.5 py-1.5 text-xs text-indigo-200"
                      >
                        <span className="max-w-[180px] truncate">{el.label}</span>
                        <button onClick={() => removeFromCauldron(el.id)} className="text-indigo-600 hover:text-red-400 ml-0.5 transition-colors">
                          <XIcon size={10} />
                        </button>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                )}
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => synthesize(true)}
                  disabled={
                    cauldron.length < 2
                    || streaming
                    || genieJobs.some(gj => gj.status === "pending" || gj.status === "running")
                  }
                  className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-30 text-white font-semibold rounded-xl px-5 py-2 text-sm transition-all shadow-lg shadow-indigo-900/40"
                  title="Synthesize — runs in the background; the resulting idea appears in the Ideas tab when ready."
                >
                  <ZapIcon size={13} />
                  Synthesize
                </button>
              </div>
            </div>

            {/* Inline streaming output was removed — Manual Genie now runs as
                a background job (matching Auto and Query modes). Progress is
                visible via the global Jobs panel; the resulting idea lands on
                the Ideas tab when synthesis completes. */}
          </div>
        )}

        {/* ── Discoveries tab ──────────────────────────────────────────── */}
        {activeTab === "discoveries" && (
          <div className="flex-1 overflow-y-auto p-6">
            {/* Hybrid client-side search across title, hypothesis, open
                questions, and source_query. Matches the Feed search UX —
                instant filter as the user types, no extra round-trip. */}
            {!capsulesLoading && capsules.length > 0 && (
              <div className="mb-5 relative">
                <SearchIcon size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-600 pointer-events-none" />
                <input
                  value={ideaSearch}
                  onChange={(e) => setIdeaSearch(e.target.value)}
                  placeholder="Search ideas by title, hypothesis, question…"
                  className="w-full bg-gray-900/60 border border-white/8 rounded-xl pl-9 pr-9 py-2.5 text-sm text-gray-100 placeholder:text-gray-600 focus:outline-none focus:border-indigo-500/50 transition-colors"
                />
                {ideaSearch && (
                  <button
                    onClick={() => setIdeaSearch("")}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                    title="Clear search"
                  >
                    <XIcon size={13} />
                  </button>
                )}
              </div>
            )}
            {capsulesLoading ? (
              // Skeleton cards while the initial capsule fetch is in flight.
              // Three rows is enough to fill the visible area on most screens
              // without paying for a longer skeleton run than the real fetch.
              <div className="space-y-5">
                {[0, 1, 2].map(i => <CapsuleSkeleton key={i} delay={i * 80} />)}
              </div>
            ) : capsules.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-700 gap-4">
                <div className="w-16 h-16 rounded-2xl bg-gray-900 border border-white/5 flex items-center justify-center">
                  <FlaskConicalIcon size={28} className="opacity-30" />
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500 mb-1">No ideas yet</p>
                  <p className="text-xs text-gray-700">
                    Use the Cauldron to synthesize ideas, or switch to Auto mode and click Run Now.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-5">
                {/* Combine bar — only shown while the user is actively in
                    combine mode. Off-mode keeps the ideas list clean of
                    persistent selection UI. */}
                {combineMode && (
                <div
                  className="sticky top-0 z-10 flex items-center gap-2 px-4 py-2.5 rounded-xl border bg-gray-950/85 backdrop-blur-sm"
                  style={{ borderColor: combineSelected.length >= 2 ? "rgba(217,70,239,0.35)" : "rgba(255,255,255,0.06)" }}
                >
                  <GitMergeIcon size={14} className={combineSelected.length >= 2 ? "text-fuchsia-300" : "text-gray-500"} />
                  <span className="text-[12px] text-gray-300">
                    Combine ideas — select 2 or 3 to fuse into a hybrid hypothesis.
                  </span>
                  <span className="text-[11px] text-gray-500 font-mono">
                    {combineSelected.length} / {COMBINE_MAX} selected
                  </span>
                  <div className="flex-1" />
                  {combineSelected.length > 0 && (
                    <button
                      onClick={clearCombineSelection}
                      disabled={combineBusy}
                      className="text-[11px] text-gray-500 hover:text-gray-300 px-2 py-1 rounded disabled:opacity-40"
                    >
                      Clear
                    </button>
                  )}
                  <button
                    onClick={startCombine}
                    disabled={combineBusy || combineSelected.length < 2 || combineSelected.length > COMBINE_MAX}
                    className="text-[11px] font-semibold px-3 py-1.5 rounded-lg transition-all"
                    style={
                      combineBusy
                        ? { background: "rgba(217,70,239,0.15)", color: "#f0abfc", border: "1px solid rgba(217,70,239,0.3)", cursor: "wait" }
                        : combineSelected.length >= 2 && combineSelected.length <= COMBINE_MAX
                        ? { background: "linear-gradient(135deg,#a21caf,#d946ef)", color: "white", border: "1px solid transparent" }
                        : { background: "rgba(255,255,255,0.04)", color: "var(--rf-text5)", border: "1px solid rgba(255,255,255,0.06)", cursor: "not-allowed" }
                    }
                    title={
                      combineSelected.length < 2
                        ? "Select at least 2 ideas to combine"
                        : combineSelected.length > COMBINE_MAX
                        ? `Maximum ${COMBINE_MAX} ideas`
                        : "Run feasibility check + fusion synthesis"
                    }
                  >
                    {combineBusy ? "Combining…" : "Combine selected"}
                  </button>
                </div>
                )}
                {combineMode && combineErr && (
                  <div className="px-3 py-2 rounded-lg bg-red-950/40 border border-red-800/40 text-red-300 text-[11px]">
                    {combineErr}
                  </div>
                )}

                {(() => {
                  const searchQ = ideaSearch.trim().toLowerCase();
                  const filtered = searchQ
                    ? capsules.filter((c) => {
                        const hay = [
                          c.title,
                          c.hypothesis,
                          c.open_questions,
                          c.source_query,
                          c.rationale,
                        ]
                          .filter(Boolean)
                          .join(" \n ")
                          .toLowerCase();
                        return hay.includes(searchQ);
                      })
                    : capsules;
                  // Build ancestor sets so combined ideas can't be re-fused with
                  // any of their own ancestors (or vice-versa). Walks
                  // `parent_capsule_ids` transitively over the visible capsules.
                  const parentMap = new Map<string, string[]>();
                  for (const c of capsules) {
                    parentMap.set(c.id, c.parent_capsule_ids || []);
                  }
                  const ancestorCache = new Map<string, Set<string>>();
                  const collectAncestors = (id: string): Set<string> => {
                    if (ancestorCache.has(id)) return ancestorCache.get(id)!;
                    const acc = new Set<string>();
                    const stack = [...(parentMap.get(id) || [])];
                    while (stack.length) {
                      const p = stack.pop()!;
                      if (acc.has(p)) continue;
                      acc.add(p);
                      for (const gp of parentMap.get(p) || []) stack.push(gp);
                    }
                    ancestorCache.set(id, acc);
                    return acc;
                  };
                  // A capsule is forbidden if:
                  //   - it is an ancestor of any selected capsule, OR
                  //   - any selected capsule is an ancestor of it.
                  const forbidden = new Set<string>();
                  for (const sel of combineSelected) {
                    for (const a of collectAncestors(sel)) forbidden.add(a);
                    for (const c of capsules) {
                      if (collectAncestors(c.id).has(sel)) forbidden.add(c.id);
                    }
                  }
                  return filtered.map((capsule) => {
                    const isForbidden = combineMode && forbidden.has(capsule.id);
                    return (
                      <CapsuleCard
                        key={capsule.id}
                        capsule={capsule}
                        onDelete={() => deleteCapsule(capsule.id)}
                        onChat={() => setChatCapsule(capsule)}
                        combineSelected={combineMode && combineSelected.includes(capsule.id)}
                        combineDisabled={
                          combineBusy ||
                          isForbidden ||
                          (!combineSelected.includes(capsule.id) && combineSelected.length >= COMBINE_MAX)
                        }
                        onCombineToggle={
                          combineMode && !isForbidden ? () => toggleCombineSelect(capsule.id) : undefined
                        }
                      />
                    );
                  });
                })()}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Temperature slider ─────────────────────────────────────────────────────────

function TemperatureSlider({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const label =
    value <= 0.2 ? "Safe" :
    value <= 0.4 ? "Focused" :
    value <= 0.6 ? "Balanced" :
    value <= 0.8 ? "Curious" : "Exploratory";

  const accent =
    value <= 0.2 ? "text-sky-400" :
    value <= 0.4 ? "text-blue-400" :
    value <= 0.6 ? "text-indigo-400" :
    value <= 0.8 ? "text-violet-400" : "text-orange-400";

  const track =
    value <= 0.2 ? "from-sky-700 to-sky-500" :
    value <= 0.4 ? "from-blue-700 to-blue-500" :
    value <= 0.6 ? "from-indigo-700 to-indigo-500" :
    value <= 0.8 ? "from-violet-700 to-violet-500" :
    "from-amber-600 to-orange-500";

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5">
          <ThermometerIcon size={11} className={accent} />
          <span className="text-[11px] text-gray-400 font-medium">Temperature</span>
        </div>
        <span className={`text-[11px] font-semibold ${accent}`}>{label}</span>
      </div>
      <div className="relative h-4 flex items-center mb-1">
        <div className="absolute inset-x-0 h-1.5 rounded-full bg-gray-800" />
        <div
          className={`absolute left-0 h-1.5 rounded-full bg-gradient-to-r ${track} transition-all pointer-events-none`}
          style={{ width: `${value * 100}%` }}
        />
        <input
          type="range"
          min={0} max={1} step={0.1}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="relative w-full h-1.5 appearance-none bg-transparent cursor-pointer accent-indigo-500 z-10"
        />
      </div>
      <div className="flex justify-between text-[9px] text-gray-700">
        <span>Safe — same top pairs</span>
        <span>Exploratory — new variety</span>
      </div>
      <p className="text-[9px] text-gray-700 mt-1 leading-tight">
        {value <= 0.3
          ? "Focuses on highest-scoring paper combos. Staleness penalties off. Dedup is lenient."
          : value <= 0.6
          ? "Balances quality with novelty. Moderate staleness penalty. Standard dedup."
          : "Actively avoids repeated pairs. Widens the paper net. Strict dedup — no near-duplicate ideas."}
      </p>
    </div>
  );
}

// ── Threshold slider ───────────────────────────────────────────────────────────

function ThresholdSlider({
  label, description, value, min, max, step, onChange,
}: {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] text-gray-400 font-medium">{label}</span>
        <span className="text-[11px] font-mono text-indigo-400 font-bold">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 appearance-none rounded-full bg-gray-800 cursor-pointer accent-indigo-500"
      />
      <p className="text-[9px] text-gray-700 mt-0.5 leading-tight">{description}</p>
    </div>
  );
}

// ── Stream event blocks ────────────────────────────────────────────────────────

function GenieEventBlock({ event }: { event: StreamEvent }) {
  if (event.type === "status") {
    return (
      <div className="flex items-center gap-2 text-xs text-gray-600">
        <div className="w-1 h-1 rounded-full bg-indigo-600 animate-pulse" />
        {event.message}
      </div>
    );
  }
  if (event.type === "error") {
    return (
      <div className="text-sm text-red-400 bg-red-950/20 border border-red-800/30 rounded-xl p-4">
        {event.message}
      </div>
    );
  }
  if (event.type === "not_viable") {
    return (
      <div className="rounded-xl border border-amber-700/30 bg-amber-950/15 p-5">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-xl bg-amber-900/30 flex items-center justify-center flex-shrink-0">
            <AlertTriangleIcon size={15} className="text-amber-400" />
          </div>
          <div>
            <p className="text-sm font-semibold text-amber-300 mb-1.5">Synthesis not possible</p>
            <p className="text-sm text-amber-200/60 leading-relaxed">{event.reason}</p>
            <p className="text-xs text-amber-600/60 mt-2">
              Select papers from the same research area or papers that share methods, datasets, or problem domain.
            </p>
          </div>
        </div>
      </div>
    );
  }
  if (event.type === "viability") {
    const simPct = Math.round((event.similarity ?? 0) * 100);
    const bridges = event.bridges ?? [];
    const color = simPct >= 35 ? "emerald" : simPct >= 18 ? "amber" : "orange";
    return (
      <div className="rounded-xl border border-white/5 bg-gray-900/50 px-4 py-3 flex flex-wrap items-center gap-4 text-xs">
        <div className="flex items-center gap-2">
          <span className="text-gray-600">Semantic overlap</span>
          <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full bg-${color}-500`}
              style={{ width: `${Math.min(simPct * 2, 100)}%` }}
            />
          </div>
          <span className={`font-mono font-bold text-${color}-400`}>{simPct}%</span>
        </div>
        {bridges.length > 0 && (
          <div className="flex items-center gap-1.5">
            <span className="text-gray-600">Bridges</span>
            {bridges.slice(0, 3).map((b) => (
              <span key={b} className="bg-indigo-900/30 text-indigo-300/80 border border-indigo-800/30 px-2 py-0.5 rounded-full">
                {b}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  }
  if (event.type === "hypothesis" && event.data) {
    const hyp = event.data as { title?: string; statement?: string };
    return (
      <div className="bg-gradient-to-b from-indigo-950/40 to-indigo-950/20 border border-indigo-700/25 rounded-xl p-5">
        <p className="text-[10px] text-indigo-400 uppercase tracking-wider mb-2 font-bold">Best Hypothesis</p>
        <h3 className="font-bold text-white mb-2 text-base leading-snug">{hyp.title}</h3>
        <p className="text-sm text-gray-300 leading-relaxed">{hyp.statement}</p>
      </div>
    );
  }
  if (event.type === "scores") {
    return (
      <div className="flex gap-3">
        {(["novelty", "feasibility", "impact"] as const).map((k) => {
          const pct = Math.round((event[k] || 0) * 100);
          const color = pct >= 70 ? "emerald" : pct >= 50 ? "amber" : "gray";
          return (
            <div key={k} className="flex-1 bg-gray-900/60 border border-white/5 rounded-xl p-3 text-center">
              <p className="text-[10px] text-gray-600 capitalize mb-1">{k}</p>
              <p className={`text-2xl font-bold text-${color}-400`}>{pct}%</p>
            </div>
          );
        })}
      </div>
    );
  }
  if (event.type === "done") {
    return (
      <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
        <div className="w-2 h-2 rounded-full bg-emerald-400" />
        Synthesis complete — see Ideas tab
      </div>
    );
  }
  return null;
}

function ElaborationSectionBlock({ section, content }: { section: string; content: string }) {
  const meta = SECTION_META[section];
  if (!meta) return null;
  const Icon = meta.icon;
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl border p-5 ${meta.bgColor} ${meta.borderColor}`}
    >
      <div className="flex items-center gap-2 mb-3">
        <div className={`w-6 h-6 rounded-lg ${meta.bgColor} border ${meta.borderColor} flex items-center justify-center`}>
          <Icon size={12} className={meta.color} />
        </div>
        <p className={`text-[10px] font-bold uppercase tracking-wider ${meta.color}`}>{meta.label}</p>
      </div>
      <div className="text-sm text-gray-300 leading-relaxed">
        <MarkdownRenderer content={content} />
      </div>
    </motion.div>
  );
}

// ── Capsule skeleton (ghost loading card) ───────────────────────────────────────
// Mirrors CapsuleCard's vertical rhythm so the transition to real content
// doesn't cause layout shift. Staggered fade-in via `delay`.

function CapsuleSkeleton({ delay = 0 }: { delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: delay / 1000 }}
      className="rounded-2xl border border-gray-800/60 bg-gradient-to-br from-gray-900/60 to-gray-950/40 px-6 py-5"
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="h-5 w-16 rounded-full bg-gray-800/70 animate-pulse" />
        <div className="h-5 w-14 rounded-full bg-gray-800/50 animate-pulse" />
        <div className="flex-1" />
        <div className="h-5 w-20 rounded-md bg-gray-800/40 animate-pulse" />
      </div>
      <div className="h-6 w-3/4 rounded bg-gray-800/70 animate-pulse mb-2" />
      <div className="h-4 w-2/3 rounded bg-gray-800/40 animate-pulse mb-4" />
      <div className="space-y-2">
        <div className="h-3 w-full rounded bg-gray-800/40 animate-pulse" />
        <div className="h-3 w-5/6 rounded bg-gray-800/40 animate-pulse" />
        <div className="h-3 w-4/6 rounded bg-gray-800/30 animate-pulse" />
      </div>
      <div className="grid grid-cols-3 gap-3 mt-5">
        <div className="h-12 rounded-xl bg-gray-800/30 animate-pulse" />
        <div className="h-12 rounded-xl bg-gray-800/30 animate-pulse" />
        <div className="h-12 rounded-xl bg-gray-800/30 animate-pulse" />
      </div>
    </motion.div>
  );
}


// ── Capsule card ───────────────────────────────────────────────────────────────

function CapsuleCard({
  capsule,
  onDelete,
  onChat,
  combineSelected = false,
  combineDisabled = false,
  onCombineToggle,
}: {
  capsule: IdeaCapsule;
  onDelete: () => void;
  onChat: () => void;
  /** True when this capsule is in the combine multi-selection. */
  combineSelected?: boolean;
  /** True when the user can't add this capsule (cap reached, or busy). */
  combineDisabled?: boolean;
  /** Toggle this capsule in/out of the combine selection. */
  onCombineToggle?: () => void;
}) {
  const router = useRouter();
  const [expanded, setExpanded] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const confirmTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The list endpoint returns slim capsules (no rationale/mechanism/etc.) to
  // keep the Ideas tab snappy. When the user expands, lazily fetch the full
  // record so the long-form sections actually render. ``detail`` falls back
  // to the slim ``capsule`` so already-loaded fields stay visible while the
  // fetch is in flight.
  const [detail, setDetail] = useState<IdeaCapsule | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const viewCapsule: IdeaCapsule = (detail ?? capsule) as IdeaCapsule;
  useEffect(() => {
    if (!expanded) return;
    if (detail !== null) return;
    if (detailLoading) return;
    setDetailLoading(true);
    api.get<IdeaCapsule>(`/genie/capsules/${capsule.id}`)
      .then(setDetail)
      .catch(() => setDetail(capsule))
      .finally(() => setDetailLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expanded]);

  const isSaved = viewCapsule.status === "saved";
  const noveltyPct = Math.round(viewCapsule.novelty_score * 100);
  const feasPct = Math.round(viewCapsule.feasibility_score * 100);
  const impactPct = Math.round(viewCapsule.impact_score * 100);

  const openQuestions = (viewCapsule.open_questions || "")
    .split("\n")
    .map((q: string) => q.replace(/^[•\-\*]\s*/, "").trim())
    .filter(Boolean);

  function handleDelete() {
    if (!confirming) {
      setConfirming(true);
      if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
      confirmTimerRef.current = setTimeout(() => setConfirming(false), 3000);
      return;
    }
    if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
    onDelete();
  }

  const scoreColor = (pct: number) =>
    pct >= 70 ? "text-emerald-400" : pct >= 50 ? "text-amber-400" : "text-gray-500";
  const barColor = (pct: number) =>
    pct >= 70 ? "bg-emerald-500" : pct >= 50 ? "bg-amber-500" : "bg-gray-600";

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.97 }}
      className={`rounded-2xl border transition-all ${
        combineSelected
          ? "border-fuchsia-500/60 bg-gradient-to-b from-fuchsia-950/30 to-gray-900/60 shadow-[0_0_0_1px_rgba(217,70,239,0.25)]"
          : isSaved
          ? "border-emerald-700/25 bg-gradient-to-b from-emerald-950/20 to-gray-900/60"
          : "border-white/5 bg-gradient-to-b from-gray-900/80 to-gray-900/40"
      }`}
    >
      {/* Header */}
      <div className="p-6">
        <div className="flex items-start justify-between gap-3 mb-4">
          {/* Combine multi-select checkbox — only shown when the parent
              page is in combine mode (onCombineToggle is provided). */}
          {onCombineToggle && (
            <button
              onClick={(e) => { e.stopPropagation(); onCombineToggle(); }}
              disabled={combineDisabled}
              title={
                combineDisabled
                  ? "Maximum 3 ideas can be combined"
                  : combineSelected
                  ? "Remove from combine selection"
                  : "Select to combine with another idea"
              }
              className="flex-shrink-0 w-5 h-5 rounded-md border flex items-center justify-center transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              style={
                combineSelected
                  ? { background: "linear-gradient(135deg,#a21caf,#d946ef)", borderColor: "transparent" }
                  : { background: "rgba(15,15,26,0.6)", borderColor: "rgba(255,255,255,0.12)" }
              }
            >
              {combineSelected && <CheckIcon size={12} color="white" strokeWidth={3} />}
            </button>
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              {isSaved && (
                <span className="text-[10px] bg-emerald-900/40 text-emerald-400 border border-emerald-700/30 px-2 py-0.5 rounded-full font-semibold">
                  ✓ Saved
                </span>
              )}
              {/* Mode tag — Manual / Auto / Query / Combined */}
              {capsule.source_mode === "combined" ? (
                <span className="text-[10px] bg-fuchsia-900/40 text-fuchsia-300 border border-fuchsia-700/30 px-2 py-0.5 rounded-full font-semibold">
                  🔗 Combined
                </span>
              ) : capsule.source_mode === "query" ? (
                <span className="text-[10px] bg-violet-900/40 text-violet-300 border border-violet-700/30 px-2 py-0.5 rounded-full font-semibold">
                  🔍 Query
                </span>
              ) : capsule.source_mode === "auto" || capsule.is_scout_generated ? (
                <span className="text-[10px] bg-indigo-900/40 text-indigo-300 border border-indigo-700/30 px-2 py-0.5 rounded-full font-semibold">
                  ⚡ Auto
                </span>
              ) : (
                <span className="text-[10px] bg-gray-800/60 text-gray-500 border border-gray-700/30 px-2 py-0.5 rounded-full font-semibold">
                  ⚗️ Manual
                </span>
              )}
            </div>
            <h3 className="font-bold text-white text-lg mb-1.5 leading-snug">{capsule.title}</h3>
            {/* Show the query that produced this capsule */}
            {capsule.source_mode === "query" && capsule.source_query && (
              <p className="text-[11px] text-violet-400/70 italic mb-1 leading-relaxed">
                &ldquo;{capsule.source_query}&rdquo;
              </p>
            )}
            {/* For combined capsules, surface which two ideas were fused */}
            {capsule.source_mode === "combined" && capsule.source_query && (
              <p className="text-[11px] text-fuchsia-400/70 italic mb-1 leading-relaxed">
                {capsule.source_query}
              </p>
            )}
            <p className="text-sm text-indigo-400/90 font-medium leading-relaxed mb-1">
              TL;DR — {buildTldr(capsule.hypothesis)}
            </p>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <button
              onClick={() => router.push(`/genie/idea/${capsule.id}`)}
              className="p-1.5 rounded-lg text-gray-600 hover:text-violet-400 hover:bg-violet-950/30 transition-colors"
              title="Deep dive study"
            >
              <BookOpenIcon size={14} />
            </button>
            <button onClick={onChat} className="p-1.5 rounded-lg text-gray-600 hover:text-indigo-400 hover:bg-indigo-950/30 transition-colors" title="Chat about this idea">
              <MessageSquareIcon size={14} />
            </button>
            <button
              onClick={handleDelete}
              className={`flex items-center gap-1 px-2 py-1 rounded-lg transition-colors text-[10px] font-semibold ${confirming ? "text-red-400 bg-red-950/30 ring-1 ring-red-500/30" : "text-gray-700 hover:text-red-400 hover:bg-red-950/20"}`}
            >
              <Trash2Icon size={12} />
              {confirming && <span>Confirm?</span>}
            </button>
            <button onClick={() => setExpanded((e) => !e)} className="p-1.5 rounded-lg text-gray-600 hover:text-gray-300 hover:bg-gray-800 transition-colors">
              {expanded ? <ChevronUpIcon size={14} /> : <ChevronDownIcon size={14} />}
            </button>
          </div>
        </div>

        {/* Score bars */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: "Novelty",     val: noveltyPct,   bar: barColor(noveltyPct),   txt: scoreColor(noveltyPct) },
            { label: "Feasibility", val: feasPct,       bar: barColor(feasPct),      txt: scoreColor(feasPct) },
            { label: "Impact",      val: impactPct,  bar: barColor(impactPct), txt: scoreColor(impactPct) },
          ].map(({ label, val, bar, txt }) => (
            <div key={label} className="bg-gray-900/60 border border-white/5 rounded-xl p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] text-gray-600">{label}</span>
                <span className={`text-sm font-bold font-mono ${txt}`}>{val}%</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className={`h-full ${bar} rounded-full transition-all`} style={{ width: `${val}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Expanded deep analysis */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-t border-white/5"
          >
            <div className="p-5 space-y-3">
              {detailLoading && !detail && (
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Loader2Icon size={12} className="animate-spin" />
                  Loading idea summary…
                </div>
              )}
              {/* Concise previews — one line per facet, not the full sections.
                  The full sections live on the dedicated idea page; this card
                  view is just a quick "does this idea interest me?" peek. */}
              {viewCapsule.rationale && (
                <CapsulePreview
                  icon={BrainIcon}
                  label="Why pursue this"
                  color="text-indigo-400"
                  text={firstSentenceOf(viewCapsule.rationale)}
                />
              )}
              {viewCapsule.predicted_outcome && (
                <CapsulePreview
                  icon={LightbulbIcon}
                  label="Predicted outcome"
                  color="text-emerald-400"
                  text={firstSentenceOf(viewCapsule.predicted_outcome)}
                />
              )}
              {viewCapsule.experimental_design && (
                <CapsulePreview
                  icon={BeakerIcon}
                  label="How to test"
                  color="text-teal-400"
                  text={firstSentenceOf(viewCapsule.experimental_design)}
                />
              )}
              {viewCapsule.risks_and_limitations && (
                <CapsulePreview
                  icon={AlertTriangleIcon}
                  label="Key risk"
                  color="text-amber-400"
                  text={firstSentenceOf(viewCapsule.risks_and_limitations)}
                />
              )}
              {openQuestions.length > 0 && (
                <CapsulePreview
                  icon={SparklesIcon}
                  label="Open question"
                  color="text-sky-400"
                  text={openQuestions[0]}
                />
              )}
              {/* Inline CTA — single small footer line, no card chrome. */}
              <div className="pt-1 flex items-center justify-between text-[11px] text-gray-600">
                <span>
                  {[
                    viewCapsule.source_papers && viewCapsule.source_papers.length > 0 ? `${viewCapsule.source_papers.length} source paper${viewCapsule.source_papers.length === 1 ? "" : "s"}` : null,
                    viewCapsule.diagrams && viewCapsule.diagrams.length > 0 ? `${viewCapsule.diagrams.length} diagram${viewCapsule.diagrams.length === 1 ? "" : "s"}` : null,
                    viewCapsule.poc_code ? "PoC code" : null,
                  ].filter(Boolean).join(" · ") || ""}
                </span>
                <button
                  onClick={() => router.push(`/genie/idea/${viewCapsule.id}`)}
                  className="text-indigo-400 hover:text-indigo-300 font-medium"
                >
                  Open full idea →
                </button>
              </div>
              {/* Empty-section safety net — only when nothing exists. */}
              {!detailLoading
                && !viewCapsule.rationale
                && !viewCapsule.predicted_outcome
                && !viewCapsule.experimental_design
                && !viewCapsule.risks_and_limitations
                && openQuestions.length === 0 && (
                <p className="text-xs text-gray-500 italic">
                  No summary fields yet — open the full idea page to generate a Deep Dive.
                </p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function RichSection({
  icon: Icon,
  label,
  color,
  bg,
  border,
  children,
}: {
  icon: React.ElementType;
  label: string;
  color: string;
  bg: string;
  border: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`rounded-xl border p-4 ${bg} ${border}`}>
      <div className="flex items-center gap-2 mb-3">
        <Icon size={13} className={color} />
        <p className={`text-[10px] font-bold uppercase tracking-wider ${color}`}>{label}</p>
      </div>
      <div className="text-sm text-gray-300 leading-relaxed [&_p]:mb-2 [&_ul]:list-disc [&_ul]:pl-4 [&_li]:mb-1 [&_strong]:text-white [&_h3]:text-white [&_h3]:font-semibold [&_h3]:mb-1">
        {children}
      </div>
    </div>
  );
}

/** Compact one-line preview row for an idea facet. */
function CapsulePreview({
  icon: Icon,
  label,
  color,
  text,
}: {
  icon: React.ElementType;
  label: string;
  color: string;
  text: string;
}) {
  if (!text) return null;
  return (
    <div className="flex items-start gap-3">
      <Icon size={13} className={`${color} mt-0.5 flex-shrink-0`} />
      <div className="min-w-0 flex-1">
        <p className={`text-[10px] font-bold uppercase tracking-wider ${color}`}>{label}</p>
        <p className="text-xs text-gray-300 leading-relaxed mt-0.5 line-clamp-2">{text}</p>
      </div>
    </div>
  );
}

/** Extract the first sentence (or first ~28 words) of a markdown blob —
 * used for one-line previews on capsule cards. Strips simple markdown
 * markers so the preview reads naturally.
 */
function firstSentenceOf(blob: string): string {
  if (!blob) return "";
  // Strip markdown emphasis, code fences, and headers cheaply.
  const cleaned = blob
    .replace(/```[\s\S]*?```/g, "")
    .replace(/[`*_>#]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned) return "";
  const m = cleaned.match(/^(.+?[.!?])(\s|$)/);
  if (m && m[1].length >= 24) return m[1];
  const words = cleaned.split(" ");
  if (words.length <= 28) return cleaned;
  return words.slice(0, 28).join(" ") + "…";
}

// ── Capsule chat overlay ───────────────────────────────────────────────────────

function CapsuleChatOverlay({
  capsule,
  token,
  onClose,
}: {
  capsule: IdeaCapsule;
  token: string;
  onClose: () => void;
}) {
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Abort any in-flight stream when the modal unmounts so navigating
  // away doesn't leave the fetch reader hanging.
  useEffect(() => () => abortRef.current?.abort(), []);

  async function sendMessage() {
    if (!input.trim() || streaming) return;
    const userMsg = input.trim();
    setInput("");
    setMessages((m) => [...m, { role: "user", content: userMsg }]);
    setStreaming(true);

    const apiBase = process.env.NEXT_PUBLIC_API_URL || "";
    const url = `${apiBase}/api/v1/genie/capsules/${capsule.id}/chat`;
    setMessages((m) => [...m, { role: "assistant", content: "" }]);

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg, history: messages }),
        signal: ctrl.signal,
      });
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of decoder.decode(value).split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            const ev = JSON.parse(line.slice(6));
            if (ev.type === "chunk") {
              setMessages((m) => {
                const copy = [...m];
                copy[copy.length - 1] = { ...copy[copy.length - 1], content: copy[copy.length - 1].content + ev.content };
                return copy;
              });
              bottomRef.current?.scrollIntoView({ behavior: "smooth" });
            }
          } catch {}
        }
      }
    } catch (err) {
      if ((err as { name?: string })?.name === "AbortError") {
        // expected on unmount — swallow
      }
    }
    setStreaming(false);
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-end justify-end p-6 bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <motion.div
        initial={{ y: 30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 30, opacity: 0 }}
        className="w-[520px] h-[640px] bg-gray-900 border border-white/10 rounded-2xl flex flex-col shadow-2xl shadow-black/60"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/5 shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <div className="w-7 h-7 rounded-xl bg-gradient-to-br from-indigo-600/30 to-violet-600/30 border border-indigo-500/20 flex items-center justify-center shrink-0">
              <MessageSquareIcon size={13} className="text-indigo-400" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-semibold text-white truncate">{capsule.title}</p>
              <p className="text-[10px] text-gray-600">Explore & develop this idea</p>
            </div>
          </div>
          <button onClick={onClose} className="text-gray-600 hover:text-white transition-colors shrink-0 ml-2">
            <XIcon size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center gap-3 text-gray-700">
              <SparklesIcon size={24} className="opacity-30" />
              <p className="text-xs max-w-xs">Ask about methodology, suggest experiments, explore implications, or request deeper analysis of this research idea.</p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[88%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed ${
                msg.role === "user" ? "bg-indigo-600 text-white" : "bg-gray-800/60 text-gray-200"
              }`}>
                {msg.role === "assistant"
                  ? <MarkdownRenderer content={msg.content || (streaming ? "▍" : "")} />
                  : msg.content}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="p-4 border-t border-white/5 shrink-0">
          <div className="flex items-center gap-2 bg-gray-800/60 border border-white/5 rounded-xl px-3.5 py-2.5">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
              placeholder="Ask about this idea…"
              className="flex-1 bg-transparent text-sm text-gray-200 placeholder-gray-600 outline-none"
            />
            <button onClick={sendMessage} disabled={!input.trim() || streaming} className="text-indigo-400 hover:text-indigo-300 disabled:opacity-30 transition-colors">
              <SendIcon size={15} />
            </button>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
