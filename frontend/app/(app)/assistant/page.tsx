"use client";

/**
 * Research Assistant workspace.
 *
 * Three-pane layout: session list (left) ▸ conversation + reasoning (center)
 * ▸ active context + artifacts + attachments (right). Block-rendered messages,
 * SSE turn streaming with polling fallback, click-to-submit suggestion chips,
 * branch-from-message, in-flight cancel, session rename / archive / clear-all,
 * note + URL + paper-ref attachments.
 */

import { Component, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import {
  CheckCircle2Icon, ChevronDownIcon, ChevronLeftIcon, ChevronRightIcon, CircleDashedIcon,
  DownloadIcon, Edit3Icon, ExternalLinkIcon, FileTextIcon, GitBranchIcon, GlobeIcon,
  HighlighterIcon, Loader2Icon, MessageSquareIcon, NetworkIcon, PaperclipIcon, PanelLeftIcon,
  PlusIcon, SendIcon, SparklesIcon, StickyNoteIcon, StopCircleIcon, Trash2Icon,
  XCircleIcon, XIcon, BookmarkIcon, BookOpenIcon,
} from "lucide-react";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth";
import { useNamespaceStore } from "@/store/namespace";
import MarkdownRenderer from "@/components/ui/MarkdownRenderer";
import { PaperPanel } from "@/components/paper/PaperPanel";
import type { Paper as FullPaper } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "/api/v1";

// ─── Types ─────────────────────────────────────────────────────────────────

type SessionSummary = {
  id: string;
  title: string;
  namespace_key: string;
  topic_keys: string[];
  status: string;
  updated_at: string;
  parent_session_id: string | null;
  summary?: string | null;
};

type AssistantMessage = {
  id: string;
  session_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  message_type: string;
  citations: string[];
  artifact_refs: { type: string; id: string; href?: string; label?: string }[];
  payload: Record<string, unknown>;
  created_at: string;
};

type AssistantTask = {
  id: string;
  job_id: string;
  assistant_message_id: string | null;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: { stage?: string; percent?: number; summary?: string; actions?: string[] };
  created_at: string;
  completed_at: string | null;
};

type AssistantSession = SessionSummary & {
  branch_from_message_id: string | null;
  orientation: string;
  expertise_level: string;
  summary: string | null;
  state: Record<string, unknown>;
  created_at: string;
  messages: AssistantMessage[];
  tasks: AssistantTask[];
};

type AssistantStep = {
  id: string;
  parent_message_id: string;
  step_index: number;
  tool_name: string;
  title: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled" | "skipped";
  progress: { summary?: string; percent?: number };
  output: Record<string, unknown>;
  cost: Record<string, unknown>;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
};

type SubmitResponse = {
  session: AssistantSession;
  user_message: AssistantMessage;
  assistant_message: AssistantMessage;
  task: AssistantTask;
};

type Highlight = {
  id: string;
  messageId: string;
  text: string;
  color: string;
};

type StickyNote = {
  id: string;
  messageId: string;
  content: string;
  minimized: boolean;
};

type Attachment = {
  id: string;
  session_id: string;
  kind: "note" | "url" | "paper_ref" | "pdf" | "image";
  label: string;
  content: string | null;
  url: string | null;
  paper_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
};

type SourcePaper = {
  title?: string;
  authors?: string[];
  abstract?: string;
  year?: number | string;
  source?: string;
  url?: string;
  doi?: string;
  pmid?: string;
  bibcode?: string;
  citation_count?: number;
};

type NvdVuln = {
  id: string;
  description?: string;
  cvss_score?: number | string;
  severity?: string;
  published?: string;
  url?: string;
};

type ClinicalStudy = {
  nct_id?: string;
  title?: string;
  status?: string;
  phase?: string;
  conditions?: string[];
  interventions?: string[];
  url?: string;
};

type FredSeriesItem = {
  id?: string;
  title?: string;
  units?: string;
  frequency?: string;
  observations?: { date: string; value: string }[];
};

type CodeItem = {
  kind?: string;
  source?: string;
  name?: string;
  full_name?: string;
  description?: string;
  stars?: number;
  language?: string;
  url?: string;
  id?: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
};

type Block =
  | { kind: "text"; content: string }
  | { kind: "paper_grid"; title?: string; papers: PaperBlock[] }
  | { kind: "arxiv_grid"; title?: string; papers: ArxivCandidate[]; imported_count?: number }
  | { kind: "source_papers"; title?: string; papers: SourcePaper[] }
  | { kind: "graph_summary"; title?: string; summary: Record<string, unknown>; href?: string }
  | { kind: "artifact_link"; title?: string; kind_label: string; href: string; ref_id: string }
  | { kind: "suggestion_chips"; title?: string; suggestions: { label: string; href?: string; kind?: string }[] }
  | { kind: "actions_taken"; actions: string[] }
  | { kind: "web_results"; title?: string; results: { title: string; url: string; snippet: string }[] }
  | { kind: "comparison_table"; title?: string; columns: ComparisonColumn[]; rows: ComparisonRow[]; notes?: string }
  | { kind: "bookmarks_answer"; title?: string; content: string }
  | { kind: "mermaid"; title?: string; code: string }
  | { kind: "nvd_results"; title?: string; vulnerabilities: NvdVuln[] }
  | { kind: "trials_results"; title?: string; studies: ClinicalStudy[] }
  | { kind: "fred_data"; title?: string; series: FredSeriesItem[] }
  | { kind: "code_results"; title?: string; items: CodeItem[] };

type PaperBlock = {
  paper_id: string;
  title: string;
  abstract?: string;
  authors?: string[];
  namespace_key?: string;
  source_url?: string;
  pdf_url?: string;
  tldr?: string;
  novelty_score?: number;
  relevance_score?: number;
  search_score?: number;
  match_type?: string;
  why_surfaced?: { signal: string; label: string; weight?: number }[];
};

type ArxivCandidate = {
  external_id?: string;
  title?: string;
  authors?: string[];
  abstract?: string;
};

type ComparisonColumn = {
  paper_id: string;
  title: string;
  authors?: string[];
  namespace_key?: string;
  source_url?: string;
  tldr?: string;
};

type ComparisonRow = {
  dimension: string;
  cells: Record<string, string>;
};

const STATUS_COLOUR: Record<string, string> = {
  pending: "var(--rf-text5)",
  running: "#6366f1",
  completed: "#22c55e",
  failed: "#ef4444",
  cancelled: "#f59e0b",
  skipped: "var(--rf-text5)",
};

// ─── Error boundary ────────────────────────────────────────────────────────
// Wraps the messages list so reconciler crashes from direct-DOM-mutation
// highlight/search marks don't kill the whole page. On error we auto-remount
// the subtree on the next tick — the highlight effect re-applies marks cleanly.

class ChatErrorBoundary extends Component<
  { children: ReactNode },
  { errorCount: number; hasError: boolean }
> {
  state = { errorCount: 0, hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  componentDidCatch(err: Error) {
    if (typeof window !== "undefined") console.warn("[chat] reconcile recover:", err.message);
  }
  componentDidUpdate() {
    if (this.state.hasError) {
      // Unmount the children (render null) cleared all our stray <mark> elements;
      // bump key and re-render so the message tree mounts fresh.
      setTimeout(() => this.setState(s => ({ errorCount: s.errorCount + 1, hasError: false })), 0);
    }
  }
  render() {
    if (this.state.hasError) return null;
    return <div key={this.state.errorCount} style={{ display: "contents" }}>{this.props.children}</div>;
  }
}

// ─── Page ──────────────────────────────────────────────────────────────────

export default function AssistantPage() {
  const sp = useSearchParams();
  const router = useRouter();
  const initialSession = sp.get("session");

  const { activeSubject, selectedTopics } = useNamespaceStore();
  const namespaceKey = activeSubject || "cs.AI";
  const topicKeys = useMemo(
    () => (selectedTopics.length ? selectedTopics : [namespaceKey]),
    [selectedTopics, namespaceKey],
  );

  const { token } = useAuthStore();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(initialSession);
  const [session, setSession] = useState<AssistantSession | null>(null);
  const [steps, setSteps] = useState<Record<string, AssistantStep[]>>({});
  const [liveJobData, setLiveJobData] = useState<Record<string, {
    rationale?: string;
    plannedSteps?: { tool: string; title: string }[];
    actions?: string[];
  }>>({});
  const [streamingContent, setStreamingContent] = useState<Record<string, string>>({});
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [pendingAttachments, setPendingAttachments] = useState<Attachment[]>([]);
  const [input, setInput] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const sendingRef = useRef(false); // synchronous guard against double-submit races
  const [error, setError] = useState<string | null>(null);
  // Persisted collapse state for the session rail. Defaults to expanded; we
  // hydrate from localStorage on mount so the user's preference survives
  // page reloads. Stored as a separate effect to avoid SSR hydration mismatch.
  const [railCollapsed, setRailCollapsed] = useState(false);
  const [contextRailCollapsed, setContextRailCollapsed] = useState(false);
  useEffect(() => {
    try {
      const saved = localStorage.getItem("rf-assistant-rail-collapsed");
      if (saved === "1") setRailCollapsed(true);
      const savedCtx = localStorage.getItem("rf-assistant-context-rail-collapsed");
      if (savedCtx === "1") setContextRailCollapsed(true);
    } catch { /* localStorage unavailable — default to expanded */ }
  }, []);
  const toggleRail = useCallback(() => {
    setRailCollapsed(prev => {
      const next = !prev;
      try { localStorage.setItem("rf-assistant-rail-collapsed", next ? "1" : "0"); } catch { /* ignore */ }
      return next;
    });
  }, []);
  const toggleContextRail = useCallback(() => {
    setContextRailCollapsed(prev => {
      const next = !prev;
      try { localStorage.setItem("rf-assistant-context-rail-collapsed", next ? "1" : "0"); } catch { /* ignore */ }
      return next;
    });
  }, []);
  const bottomRef = useRef<HTMLDivElement>(null);
  const streamCtrlRef = useRef<Record<string, AbortController>>({});
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // In-chat keyword search
  const [chatSearchOpen, setChatSearchOpen] = useState(false);
  const [chatSearchQuery, setChatSearchQuery] = useState("");
  const [chatMatchCount, setChatMatchCount] = useState(0);
  const [chatMatchIdx, setChatMatchIdx] = useState(0);

  // Highlighter mode + persistent highlights (localStorage per session)
  const [highlightMode, setHighlightMode] = useState(false);
  const [highlights, setHighlights] = useState<Highlight[]>([]);

  // Sticky notes (localStorage per session)
  const [stickyNotes, setStickyNotes] = useState<StickyNote[]>([]);

  // Load → save race guard.
  //
  // Without this, switching activeId fires the load effect and the save effect
  // in the same React tick. setHighlights(loaded) is async; the save effect's
  // closure still sees the OLD highlights value but the NEW activeId, so it
  // writes the previous session's data into the new session's localStorage key
  // before React re-renders. On reload the new session's highlights are gone.
  //
  // The ref is flipped to `true` on every load, and consumed by the next save
  // effect — skipping exactly one save per load. Subsequent user-driven state
  // changes pass through normally.
  const skipNextHighlightSaveRef = useRef(false);
  const skipNextNoteSaveRef = useRef(false);

  // Load highlights + sticky notes from localStorage when session changes
  useEffect(() => {
    if (!activeId) {
      setHighlights([]);
      setStickyNotes([]);
      skipNextHighlightSaveRef.current = true;
      skipNextNoteSaveRef.current = true;
      return;
    }
    try {
      const h = localStorage.getItem(`rf-highlights-${activeId}`);
      setHighlights(h ? JSON.parse(h) : []);
      const n = localStorage.getItem(`rf-notes-${activeId}`);
      setStickyNotes(n ? JSON.parse(n) : []);
    } catch {
      setHighlights([]);
      setStickyNotes([]);
    }
    skipNextHighlightSaveRef.current = true;
    skipNextNoteSaveRef.current = true;
  }, [activeId]);

  // Persist highlights whenever they change — except for the first save after
  // a load, which would otherwise overwrite the loaded value with the previous
  // session's stale closure data (see comment above).
  useEffect(() => {
    if (!activeId) return;
    if (skipNextHighlightSaveRef.current) {
      skipNextHighlightSaveRef.current = false;
      return;
    }
    try { localStorage.setItem(`rf-highlights-${activeId}`, JSON.stringify(highlights)); } catch { /* storage full */ }
  }, [highlights, activeId]);

  // Persist sticky notes whenever they change
  useEffect(() => {
    if (!activeId) return;
    if (skipNextNoteSaveRef.current) {
      skipNextNoteSaveRef.current = false;
      return;
    }
    try { localStorage.setItem(`rf-notes-${activeId}`, JSON.stringify(stickyNotes)); } catch { /* storage full */ }
  }, [stickyNotes, activeId]);

  // Inline PaperPanel — opened when the user clicks a paper card or citation
  // chip from any assistant message. Loads the full Paper row from the
  // backend so the panel has tldr, key_concepts, scores, etc. that the
  // assistant's PaperBlock subset doesn't carry.
  const [openPaper, setOpenPaper] = useState<FullPaper | null>(null);
  const [paperLoading, setPaperLoading] = useState(false);
  const openPaperById = useCallback(async (paperId: string) => {
    if (!paperId) return;
    setPaperLoading(true);
    setContextRailCollapsed(true);
    try {
      const p = await api.get<FullPaper>(`/papers/${encodeURIComponent(paperId)}`);
      setOpenPaper(p);
    } catch (e) {
      setError(`Couldn't load paper: ${(e as Error).message}`);
    } finally {
      setPaperLoading(false);
    }
  }, []);

  // ── Data loaders ────────────────────────────────────────────────────────

  const loadSessions = useCallback(async () => {
    try {
      const params = new URLSearchParams({ limit: "50" });
      if (namespaceKey) params.set("namespace_key", namespaceKey);
      const res = await api.get<SessionSummary[]>(`/assistant/sessions?${params}`);
      setSessions(res);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [namespaceKey]);

  const loadAttachments = useCallback(async (sid: string) => {
    try {
      const res = await api.get<Attachment[]>(`/assistant/sessions/${sid}/attachments?limit=50`);
      setAttachments(res);
    } catch {
      setAttachments([]);
    }
  }, []);

  // Serial counter to discard stale concurrent loadSession responses.
  const loadSessionSeqRef = useRef(0);

  const loadSession = useCallback(async (id: string) => {
    const seq = ++loadSessionSeqRef.current;
    try {
      const res = await api.get<AssistantSession>(`/assistant/sessions/${id}`);
      // Discard if a newer request started while this one was in flight.
      if (seq !== loadSessionSeqRef.current) return;
      setSession(res);
      const stepBundles: Record<string, AssistantStep[]> = {};
      for (const msg of res.messages) {
        if (msg.role !== "assistant") continue;
        try {
          const sx = await api.get<AssistantStep[]>(`/assistant/messages/${msg.id}/steps`);
          if (seq !== loadSessionSeqRef.current) return;
          stepBundles[msg.id] = sx;
        } catch {
          stepBundles[msg.id] = [];
        }
      }
      if (seq !== loadSessionSeqRef.current) return;
      setSteps(stepBundles);
      loadAttachments(id);
    } catch (e) {
      if (seq === loadSessionSeqRef.current) setError((e as Error).message);
    }
  }, [loadAttachments]);

  // ── Lifecycle effects ───────────────────────────────────────────────────

  useEffect(() => { loadSessions(); }, [loadSessions]);
  useEffect(() => { if (activeId) loadSession(activeId); }, [activeId, loadSession]);

  // Poll the active session while a task is in-flight (SSE stream is the
  // primary live channel; this is a belt-and-suspenders catch-up loop in
  // case the SSE drops or never connected).
  useEffect(() => {
    if (!activeId || !session) return;
    const inflight = session.tasks.some(t => t.status === "pending" || t.status === "running");
    if (!inflight) return;
    const t = setInterval(() => loadSession(activeId), 3500);
    return () => clearInterval(t);
  }, [activeId, session, loadSession]);

  useEffect(() => {
    // Don't auto-scroll to bottom while the user is actively using chat search —
    // the navigate() call already scrolls to the current match, and an
    // overlapping auto-scroll yanks the viewport away mid-read.
    if (chatSearchOpen && chatSearchQuery.trim()) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [session?.messages.length, chatSearchOpen, chatSearchQuery]);

  // SSE turn-event consumer.
  const subscribeToJob = useCallback(async (jobId: string) => {
    if (streamCtrlRef.current[jobId]) return;
    const ctrl = new AbortController();
    streamCtrlRef.current[jobId] = ctrl;
    try {
      const resp = await fetch(`${API_BASE}/assistant/tasks/${jobId}/stream`, {
        method: "GET",
        headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        signal: ctrl.signal,
      });
      if (!resp.body) return;
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const frames = buf.split("\n\n");
        buf = frames.pop() ?? "";
        for (const frame of frames) {
          let kind = "message";
          let dataPayload = "";
          for (const line of frame.split("\n")) {
            if (line.startsWith("event:")) kind = line.slice(6).trim();
            else if (line.startsWith("data:")) dataPayload += line.slice(5).trim();
          }
          if (!dataPayload) continue;
          if (kind === "plan_committed") {
            try {
              const d = JSON.parse(dataPayload);
              setLiveJobData(prev => ({
                ...prev,
                [jobId]: { rationale: d.rationale, plannedSteps: d.steps, actions: d.actions },
              }));
            } catch { /* ignore */ }
          }
          if (kind === "message_delta") {
            try {
              const d = JSON.parse(dataPayload);
              if (d.message_id && d.delta) {
                setStreamingContent(prev => ({
                  ...prev,
                  [d.message_id]: (prev[d.message_id] || "") + d.delta,
                }));
              }
            } catch { /* ignore */ }
          }
          if (kind === "step_completed" || kind === "step_started" ||
              kind === "step_progress" || kind === "task_completed") {
            if (activeId) loadSession(activeId);
          }
          if (kind === "task_completed" || kind === "task_failed" || kind === "task_cancelled") {
            ctrl.abort();
            delete streamCtrlRef.current[jobId];
          }
        }
      }
    } catch {
      // Network failure — polling effect picks up the slack.
    } finally {
      delete streamCtrlRef.current[jobId];
    }
  }, [token, activeId, loadSession]);

  useEffect(() => {
    return () => {
      Object.values(streamCtrlRef.current).forEach(c => c.abort());
      streamCtrlRef.current = {};
    };
  }, []);

  // Ctrl/Cmd+F → open in-chat search
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === "f" && chatScrollRef.current?.contains(document.activeElement) === false) {
        e.preventDefault();
        setChatSearchOpen(true);
        setTimeout(() => searchInputRef.current?.focus(), 30);
      }
      if (e.key === "Escape" && chatSearchOpen) {
        setChatSearchOpen(false);
        setChatSearchQuery("");
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [chatSearchOpen]);

  // ── Highlights + search: React-tree decoration approach ────────────────────
  //
  // Earlier iterations of this code manipulated the DOM directly to add and
  // remove <mark> wrappers. That always lost the fight against React's
  // reconciliation — streaming, polling, parent re-renders, and the
  // "current match" style mutation each triggered cleanup loops or wiped
  // the marks entirely.
  //
  // The right architecture: marks are React JSX elements emitted by
  // ``MarkdownRenderer``. We pass ``decorations`` (highlights + search query)
  // down per message. React owns the marks; they survive every re-render
  // automatically. No MutationObserver, no useEffect mutates DOM.

  // Build a per-message highlight map so MarkdownRenderer only gets the
  // highlights it cares about.
  const highlightsByMessage = useMemo(() => {
    const map: Record<string, Highlight[]> = {};
    for (const h of highlights) {
      (map[h.messageId] ||= []).push(h);
    }
    return map;
  }, [highlights]);

  // Stable callback to remove a highlight by id (passed into MarkdownRenderer
  // so the click handler on a <mark> closes over a fresh setHighlights).
  const onRemoveHighlight = useCallback((id: string) => {
    setHighlights(prev => prev.filter(h => h.id !== id));
  }, []);

  // Search match count is derived from the rendered DOM AFTER React commits.
  // useEffect runs post-paint; the DOM at that point contains all React-owned
  // search marks. Counting via querySelectorAll is read-only — no mutation,
  // no feedback loop.
  useEffect(() => {
    const container = chatScrollRef.current;
    if (!container) {
      setChatMatchCount(0);
      return;
    }
    if (!chatSearchQuery.trim()) {
      setChatMatchCount(0);
      return;
    }
    const marks = container.querySelectorAll("mark[data-rf-search]");
    setChatMatchCount(marks.length);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatSearchQuery, session?.messages, streamingContent]);

  // Reset the current-match index ONLY when the user types a different query,
  // never on content updates. This is what makes next/previous reliable.
  useEffect(() => {
    setChatMatchIdx(0);
  }, [chatSearchQuery]);

  // Apply the "current match" style by toggling a class on the Nth search
  // mark. Pure DOM-style change on React-owned elements — no childList
  // mutations, so no feedback loop. Runs after every render so the highlight
  // follows the right element even if the DOM list changed.
  useEffect(() => {
    const container = chatScrollRef.current;
    if (!container) return;
    const marks = Array.from(container.querySelectorAll("mark[data-rf-search]")) as HTMLElement[];
    if (!marks.length) return;
    marks.forEach(m => { m.style.background = "#fbbf24"; });
    const idx = Math.max(0, Math.min(chatMatchIdx, marks.length - 1));
    const cur = marks[idx];
    if (cur) cur.style.background = "#f59e0b";
  }, [chatMatchIdx, chatMatchCount]);

  // Highlight-mode mouseup handler.
  //
  // Behaviour (matches the user's expectation):
  //   * When highlight mode is ON:
  //       - selecting un-highlighted text → adds a new highlight
  //       - selecting an existing highlight's exact text → removes that highlight
  //   * When highlight mode is OFF: no capture (selections are normal).
  //   * Existing highlights stay rendered REGARDLESS of mode (React owns them).
  useEffect(() => {
    if (!highlightMode) return;
    function handleMouseUp() {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) return;
      const selectedText = sel.toString().trim();
      if (selectedText.length < 2) return;
      const container = chatScrollRef.current;
      if (!container) return;

      const range = sel.getRangeAt(0);
      let el: Element | null = range.commonAncestorContainer instanceof Element
        ? range.commonAncestorContainer
        : range.commonAncestorContainer.parentElement;
      let msgId: string | undefined;
      while (el && el !== container) {
        const id = (el as HTMLElement).dataset?.messageId;
        if (id) { msgId = id; break; }
        el = el.parentElement;
      }
      if (!msgId) return;

      // If the user's selection is exactly the text of an existing highlight
      // in this message, treat it as "toggle off" instead of "add".
      setHighlights(prev => {
        const existing = prev.find(h => h.messageId === msgId && h.text === selectedText);
        if (existing) {
          return prev.filter(h => h.id !== existing.id);
        }
        const hl: Highlight = {
          id: crypto.randomUUID(),
          messageId: msgId!,
          text: selectedText,
          color: "#fef08a",
        };
        return [...prev, hl];
      });
      sel.removeAllRanges();
    }
    document.addEventListener("mouseup", handleMouseUp);
    return () => document.removeEventListener("mouseup", handleMouseUp);
  }, [highlightMode]);

  function chatSearchNavigate(dir: 1 | -1) {
    const container = chatScrollRef.current;
    if (!container) return;
    const marks = Array.from(container.querySelectorAll("mark[data-rf-search]")) as HTMLElement[];
    if (!marks.length) return;
    const next = (chatMatchIdx + dir + marks.length) % marks.length;
    setChatMatchIdx(next);
    marks[next].scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // ── Mutations ───────────────────────────────────────────────────────────

  async function ensureSession(title?: string): Promise<string | null> {
    if (activeId) return activeId;
    try {
      const created = await api.post<AssistantSession>("/assistant/sessions", {
        title,
        namespace_key: namespaceKey,
        topic_keys: topicKeys,
      });
      setActiveId(created.id);
      router.replace(`/assistant?session=${created.id}`);
      await loadSessions();
      return created.id;
    } catch (e) {
      setError((e as Error).message);
      return null;
    }
  }

  async function submit(text?: string) {
    const content = (text ?? input).trim();
    if (!content || sendingRef.current) return;
    sendingRef.current = true;
    setSubmitting(true);
    setError(null);
    try {
      // ensureSession is now inside try so finally always resets submitting state,
      // even if session creation or any subsequent await fails or hangs.
      const sid = await ensureSession();
      if (!sid) return;
      if (!text) setInput("");
      const res = await api.post<SubmitResponse>(
        `/assistant/sessions/${sid}/messages`,
        {
          content,
          namespace_key: namespaceKey,
          topic_keys: topicKeys,
          attachments: pendingAttachments.map(a => ({
            id: a.id, kind: a.kind, label: a.label,
          })),
        },
      );
      setSession(prev => {
        if (!prev) return prev;
        const existingIds = new Set(prev.messages.map(m => m.id));
        const newMsgs = [res.user_message, res.assistant_message].filter(m => !existingIds.has(m.id));
        const existingTaskIds = new Set(prev.tasks.map(t => t.id));
        const newTasks = [res.task].filter(t => !existingTaskIds.has(t.id));
        return { ...prev, messages: [...prev.messages, ...newMsgs], tasks: [...prev.tasks, ...newTasks] };
      });
      setPendingAttachments([]);
      subscribeToJob(res.task.job_id);
      loadSession(sid);
      loadSessions();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      sendingRef.current = false;
      setSubmitting(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }

  async function cancelJob(jobId: string) {
    try {
      await api.post(`/assistant/tasks/${jobId}/cancel`);
      if (activeId) loadSession(activeId);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function archiveSession(id: string) {
    if (!confirm("Archive this session? You can re-open it later from the archive view.")) return;
    try {
      await api.delete(`/assistant/sessions/${id}`);
      if (activeId === id) {
        setActiveId(null);
        setSession(null);
        router.replace("/assistant");
      }
      await loadSessions();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function clearAllSessions() {
    if (!confirm("Archive ALL active sessions? This is reversible (sessions stay in the database).")) return;
    try {
      await api.post<{ archived: number }>("/assistant/sessions/clear");
      setActiveId(null);
      setSession(null);
      router.replace("/assistant");
      await loadSessions();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function renameSession(id: string, current: string) {
    const next = prompt("Rename session:", current);
    if (!next || next.trim() === current) return;
    try {
      await api.patch(`/assistant/sessions/${id}/title`, { title: next.trim() });
      await loadSessions();
      if (activeId === id) loadSession(id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function branchFromMessage(messageId: string) {
    if (!activeId) return;
    try {
      const child = await api.post<AssistantSession>(
        `/assistant/sessions/${activeId}/branch`,
        { from_message_id: messageId, title: `Branch from ${session?.title || "session"}` },
      );
      setActiveId(child.id);
      router.replace(`/assistant?session=${child.id}`);
      await loadSessions();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function attachNote(label: string, content: string) {
    if (!content.trim()) return;
    const sid = await ensureSession();
    if (!sid) return;
    try {
      const att = await api.post<Attachment>(
        `/assistant/sessions/${sid}/attachments`,
        { kind: "note", label: label || "Note", content },
      );
      setPendingAttachments(prev => [...prev, att]);
      loadAttachments(sid);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function attachUrl(url: string, label: string) {
    if (!url.trim()) return;
    const sid = await ensureSession();
    if (!sid) return;
    try {
      const att = await api.post<Attachment>(
        `/assistant/sessions/${sid}/attachments`,
        { kind: "url", url: url.trim(), label: label || url.trim() },
      );
      setPendingAttachments(prev => [...prev, att]);
      loadAttachments(sid);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function attachFile(file: File): Promise<void> {
    const sid = await ensureSession();
    if (!sid) return;
    try {
      // Multipart upload — bypass the JSON `api` helper because it forces
      // application/json. Authorization header is added manually.
      const fd = new FormData();
      fd.append("file", file);
      const resp = await fetch(`${API_BASE}/assistant/sessions/${sid}/attachments/upload`, {
        method: "POST",
        headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        body: fd,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `Upload failed (HTTP ${resp.status})`);
      }
      const att: Attachment = await resp.json();
      setPendingAttachments(prev => [...prev, att]);
      loadAttachments(sid);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function deleteAttachment(att: Attachment) {
    try {
      await api.delete(`/assistant/sessions/${att.session_id}/attachments/${att.id}`);
      setPendingAttachments(prev => prev.filter(a => a.id !== att.id));
      loadAttachments(att.session_id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  // ── Highlights ──────────────────────────────────────────────────────────

  function removeHighlight(id: string) {
    setHighlights(prev => prev.filter(h => h.id !== id));
  }

  // ── Sticky notes ────────────────────────────────────────────────────────

  function addStickyNote(messageId: string) {
    const note: StickyNote = { id: crypto.randomUUID(), messageId, content: "", minimized: false };
    setStickyNotes(prev => [...prev, note]);
  }

  function updateStickyNote(id: string, patch: Partial<StickyNote>) {
    setStickyNotes(prev => prev.map(n => n.id === id ? { ...n, ...patch } : n));
  }

  function deleteStickyNote(id: string) {
    setStickyNotes(prev => prev.filter(n => n.id !== id));
  }

  // ── Export session ──────────────────────────────────────────────────────

  async function exportSession() {
    if (!activeId) return;
    try {
      const resp = await fetch(`${API_BASE}/assistant/sessions/${activeId}/export?format=markdown`, {
        headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      });
      if (!resp.ok) throw new Error("Export failed");
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${(session?.title || "research").replace(/\s+/g, "-").slice(0, 40)}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(`Export failed: ${(e as Error).message}`);
    }
  }

  // ── Computed ────────────────────────────────────────────────────────────

  const activeRunningTask = session?.tasks.find(
    t => t.status === "running" || t.status === "pending",
  );

  // Deduplicated, stably-sorted message list.  Dedup is necessary because the
  // optimistic update and a concurrent loadSession can both add the same message,
  // producing duplicates.  Tiebreak by role (user before assistant) so a pair
  // created at the same millisecond always renders in the correct order.
  const orderedMessages = useMemo(() => {
    const seen = new Set<string>();
    return (session?.messages ?? [])
      .filter(m => {
        if (m.role === "system") return false;
        if (seen.has(m.id)) return false;
        seen.add(m.id);
        return true;
      })
      .sort((a, b) => {
        const ta = new Date(a.created_at).getTime();
        const tb = new Date(b.created_at).getTime();
        if (ta !== tb) return ta - tb;
        if (a.role === "user" && b.role !== "user") return -1;
        if (a.role !== "user" && b.role === "user") return 1;
        return a.id.localeCompare(b.id);
      });
  }, [session?.messages]);

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div style={{ display: "flex", height: "100%", background: "var(--rf-bg)" }}>

      {/* Left rail — sessions */}
      <SessionList
        sessions={sessions}
        activeId={activeId}
        collapsed={railCollapsed}
        onToggleCollapsed={toggleRail}
        onSelect={(id) => { setActiveId(id); router.replace(`/assistant?session=${id}`); }}
        onCreate={async () => { setActiveId(null); setSession(null); router.replace("/assistant"); inputRef.current?.focus(); }}
        onRename={renameSession}
        onArchive={archiveSession}
        onClearAll={clearAllSessions}
      />

      {/* Center — conversation */}
      <section style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <header style={{
          padding: "14px 18px", borderBottom: "1px solid var(--rf-border)",
          display: "flex", alignItems: "center", gap: 10,
        }}>
          <SparklesIcon size={16} color="#8b5cf6" />
          <div style={{ flex: 1, minWidth: 0 }}>
            <h1 style={{ fontSize: "13px", fontWeight: 700, color: "var(--rf-text1)", margin: 0,
                         whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {session?.title || "Research Assistant"}
            </h1>
            <p style={{ fontSize: "10px", color: "var(--rf-text5)", margin: 0 }}>
              {namespaceKey} · {topicKeys.length} topic(s)
              {session?.parent_session_id && (
                <> · <span style={{ color: "#a78bfa" }}>branched</span></>
              )}
            </p>
          </div>
          {activeRunningTask && (
            <button
              onClick={() => cancelJob(activeRunningTask.job_id)}
              style={{
                display: "flex", alignItems: "center", gap: 4,
                padding: "5px 10px", borderRadius: 6,
                background: "rgba(239,68,68,0.1)", color: "#ef4444",
                border: "1px solid rgba(239,68,68,0.3)",
                fontSize: "11px", fontWeight: 600, cursor: "pointer",
              }}
            >
              <StopCircleIcon size={12} /> Stop
            </button>
          )}
          {/* Highlighter mode toggle */}
          <button
            onClick={() => setHighlightMode(m => !m)}
            title={highlightMode ? "Highlighter on — select text to highlight (click to switch off)" : "Highlighter off — click to enable"}
            style={{
              background: highlightMode ? "rgba(254,240,138,0.2)" : "none",
              border: highlightMode ? "1px solid rgba(254,240,138,0.6)" : "none",
              borderRadius: 5, cursor: "pointer",
              color: highlightMode ? "#ca8a04" : "var(--rf-text4)",
              padding: "4px 6px", display: "flex", alignItems: "center",
            }}
          >
            <HighlighterIcon size={13} />
          </button>

          {/* Export session */}
          {session && (
            <button
              onClick={exportSession}
              title="Export session as Markdown"
              style={{
                background: "none", border: "none", borderRadius: 5, cursor: "pointer",
                color: "var(--rf-text4)", padding: "4px 6px", display: "flex", alignItems: "center",
              }}
            >
              <DownloadIcon size={13} />
            </button>
          )}

          {/* In-chat keyword search */}
          <button
            onClick={() => { setChatSearchOpen(o => !o); if (!chatSearchOpen) setTimeout(() => searchInputRef.current?.focus(), 30); }}
            title="Search conversation (Ctrl+F)"
            style={{
              background: chatSearchOpen ? "var(--rf-nav-active)" : "none",
              border: chatSearchOpen ? "1px solid var(--rf-nav-border)" : "none",
              borderRadius: 5, cursor: "pointer", color: "var(--rf-text4)",
              padding: "4px 6px", display: "flex", alignItems: "center",
            }}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            </svg>
          </button>
        </header>

        {/* In-chat search bar */}
        {chatSearchOpen && (
          <div style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: "6px 16px", borderBottom: "1px solid var(--rf-border)",
            background: "var(--rf-surface1)",
          }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--rf-text4)" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            </svg>
            <input
              ref={searchInputRef}
              value={chatSearchQuery}
              onChange={e => setChatSearchQuery(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter") chatSearchNavigate(e.shiftKey ? -1 : 1);
                if (e.key === "Escape") { setChatSearchOpen(false); setChatSearchQuery(""); }
              }}
              placeholder="Search conversation…"
              style={{
                flex: 1, background: "transparent", border: "none", outline: "none",
                fontSize: "12px", color: "var(--rf-text1)", fontFamily: "inherit",
              }}
            />
            {chatSearchQuery && (
              <span style={{ fontSize: "10px", color: "var(--rf-text5)", whiteSpace: "nowrap" }}>
                {chatMatchCount === 0 ? "no matches" : `${chatMatchIdx + 1} / ${chatMatchCount}`}
              </span>
            )}
            <button onClick={() => chatSearchNavigate(-1)} disabled={chatMatchCount === 0}
              style={{ background: "none", border: "none", cursor: "pointer", color: "var(--rf-text4)", padding: "2px 4px" }}
              title="Previous (Shift+Enter)">↑</button>
            <button onClick={() => chatSearchNavigate(1)} disabled={chatMatchCount === 0}
              style={{ background: "none", border: "none", cursor: "pointer", color: "var(--rf-text4)", padding: "2px 4px" }}
              title="Next (Enter)">↓</button>
            <button onClick={() => { setChatSearchOpen(false); setChatSearchQuery(""); }}
              style={{ background: "none", border: "none", cursor: "pointer", color: "var(--rf-text4)", padding: "2px 4px" }}>
              <XIcon size={11} />
            </button>
          </div>
        )}

        {error && (
          <div style={{
            padding: "8px 14px", margin: "10px 16px", borderRadius: 6,
            background: "rgba(239,68,68,0.1)", color: "#ef4444", fontSize: "11.5px",
            display: "flex", alignItems: "center", gap: 8,
          }}>
            <span style={{ flex: 1 }}>{error}</span>
            <button onClick={() => setError(null)} style={{
              background: "none", border: "none", color: "#ef4444", cursor: "pointer",
            }}><XIcon size={12} /></button>
          </div>
        )}

        <div ref={chatScrollRef} style={{ flex: 1, overflowY: "auto", padding: "16px 24px" }}>
          {!session && !activeId && (
            <EmptyState
              namespaceKey={namespaceKey}
              onStart={(text) => { setInput(text); inputRef.current?.focus(); }}
            />
          )}
          <ChatErrorBoundary>
            {orderedMessages.map(msg => (
              <MessageBlock
                key={msg.id}
                msg={msg}
                steps={steps[msg.id] || []}
                tasks={session!.tasks}
                onSuggestionClick={(text) => submit(text)}
                onBranch={() => branchFromMessage(msg.id)}
                onCancel={(jobId) => cancelJob(jobId)}
                onOpenPaper={openPaperById}
                liveJobData={liveJobData}
                streamingContent={streamingContent}
                stickyNotes={stickyNotes.filter(n => n.messageId === msg.id)}
                onAddNote={() => addStickyNote(msg.id)}
                onUpdateNote={updateStickyNote}
                onDeleteNote={deleteStickyNote}
                highlightsForMessage={highlightsByMessage[msg.id] || []}
                searchQuery={chatSearchQuery}
                onRemoveHighlight={onRemoveHighlight}
              />
            ))}
          </ChatErrorBoundary>
          <div ref={bottomRef} />
        </div>

        <Composer
          inputRef={inputRef}
          input={input}
          setInput={setInput}
          onSubmit={() => submit()}
          submitting={submitting}
          pendingAttachments={pendingAttachments}
          onRemoveAttachment={deleteAttachment}
          onAttachNote={attachNote}
          onAttachUrl={attachUrl}
          onAttachFile={attachFile}
        />
      </section>

      {/* Right rail — context */}
      <ContextRail
        session={session}
        attachments={attachments}
        onRemoveAttachment={deleteAttachment}
        runningTask={activeRunningTask}
        onCancelTask={cancelJob}
        collapsed={contextRailCollapsed}
        onToggleCollapsed={toggleContextRail}
        liveJobData={activeRunningTask ? liveJobData[activeRunningTask.job_id] : undefined}
      />

      {/* Inline PaperPanel — opens when a paper card or citation chip is
          clicked. Renders on top of the workspace via its own portal-style
          fixed positioning inside the component, so it doesn't disturb the
          three-pane layout. */}
      {paperLoading && !openPaper && (
        <div style={{
          position: "fixed", top: 16, right: 16, padding: "8px 12px",
          background: "var(--rf-surface)", border: "1px solid var(--rf-border)",
          borderRadius: 8, fontSize: "11px", color: "var(--rf-text3)",
          boxShadow: "var(--rf-shadow)",
          display: "flex", alignItems: "center", gap: 6, zIndex: 100,
        }}>
          <Loader2Icon size={12} className="animate-spin" /> Loading paper…
        </div>
      )}
      {openPaper && <PaperPanel paper={openPaper} onClose={() => setOpenPaper(null)} />}
    </div>
  );
}

// (the previous _applyHighlightMark helper has been removed — highlights and
//  search marks are now rendered as React <mark> elements via the
//  ``decorations`` prop on MarkdownRenderer, so DOM mutation isn't needed.)

// ─── Sticky notes layer ────────────────────────────────────────────────────

function StickyNotesLayer({
  notes, onUpdate, onDelete,
}: {
  notes: StickyNote[];
  onUpdate: (id: string, patch: Partial<StickyNote>) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 6 }}>
      {notes.map(note => (
        <StickyNoteCard key={note.id} note={note} onUpdate={onUpdate} onDelete={onDelete} />
      ))}
    </div>
  );
}

const NOTE_MAX_CHARS = 500;
const NOTE_MAX_HEIGHT = 168; // px — ~7 lines at 1.55 line-height + padding

function StickyNoteCard({
  note, onUpdate, onDelete,
}: {
  note: StickyNote;
  onUpdate: (id: string, patch: Partial<StickyNote>) => void;
  onDelete: (id: string) => void;
}) {
  const taRef = useRef<HTMLTextAreaElement>(null);
  const [copied, setCopied] = useState(false);

  // Grow the textarea to fit its content, capped at NOTE_MAX_HEIGHT.
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    const natural = el.scrollHeight;
    el.style.height = `${Math.min(natural, NOTE_MAX_HEIGHT)}px`;
    el.style.overflowY = natural > NOTE_MAX_HEIGHT ? "auto" : "hidden";
  }, [note.content]);

  function copyNote() {
    if (!note.content) return;
    navigator.clipboard.writeText(note.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  if (note.minimized) {
    return (
      <button
        onClick={() => onUpdate(note.id, { minimized: false })}
        title="Expand note"
        style={{
          display: "inline-flex", alignItems: "center", gap: 5,
          padding: "3px 8px", borderRadius: 12,
          background: "rgba(250,204,21,0.15)", border: "1px solid rgba(250,204,21,0.4)",
          color: "#a16207", fontSize: "10px", fontWeight: 600,
          cursor: "pointer", alignSelf: "flex-start",
        }}
      >
        <StickyNoteIcon size={10} color="#ca8a04" />
        {note.content ? note.content.slice(0, 30) + (note.content.length > 30 ? "…" : "") : "Note"}
      </button>
    );
  }

  const atLimit = note.content.length >= NOTE_MAX_CHARS;

  return (
    <div style={{
      borderRadius: 8, overflow: "hidden",
      border: "1px solid rgba(250,204,21,0.4)",
      background: "rgba(254,252,232,0.06)",
      boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
    }}>
      <div style={{
        display: "flex", alignItems: "center", gap: 6,
        padding: "5px 8px",
        background: "rgba(250,204,21,0.12)",
        borderBottom: "1px solid rgba(250,204,21,0.2)",
      }}>
        <StickyNoteIcon size={10} color="#ca8a04" />
        <span style={{ fontSize: "9px", fontWeight: 700, color: "#a16207",
                       textTransform: "uppercase", letterSpacing: "0.06em", flex: 1 }}>
          Note
        </span>
        <button
          onClick={copyNote}
          title="Copy note"
          disabled={!note.content}
          style={{ background: "none", border: "none", cursor: note.content ? "pointer" : "default",
                   color: copied ? "#16a34a" : "#a16207", display: "flex", alignItems: "center",
                   padding: 2, opacity: note.content ? 1 : 0.35, transition: "color 0.2s" }}
        >
          {copied
            ? <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>
            : <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          }
        </button>
        <button
          onClick={() => onUpdate(note.id, { minimized: true })}
          title="Minimize"
          style={{ background: "none", border: "none", cursor: "pointer", color: "#a16207",
                   display: "flex", alignItems: "center", padding: 2 }}
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12h14"/>
          </svg>
        </button>
        <button
          onClick={() => onDelete(note.id)}
          title="Delete note"
          style={{ background: "none", border: "none", cursor: "pointer", color: "#a16207",
                   display: "flex", alignItems: "center", padding: 2 }}
        >
          <XIcon size={10} />
        </button>
      </div>
      <textarea
        ref={taRef}
        value={note.content}
        onChange={e => onUpdate(note.id, { content: e.target.value })}
        placeholder="Write your note…"
        maxLength={NOTE_MAX_CHARS}
        style={{
          width: "100%", padding: "8px 10px", border: "none", outline: "none", resize: "none",
          background: "transparent", color: "var(--rf-text1)", fontSize: "11.5px",
          fontFamily: "inherit", lineHeight: 1.55, boxSizing: "border-box",
          minHeight: "40px", overflowY: "hidden", whiteSpace: "pre-wrap", wordBreak: "break-word",
        }}
      />
      {atLimit && (
        <div style={{
          padding: "2px 10px 5px", fontSize: "9px", color: "#a16207",
          opacity: 0.7, textAlign: "right",
        }}>
          {NOTE_MAX_CHARS}/{NOTE_MAX_CHARS}
        </div>
      )}
    </div>
  );
}

// ─── Session tree ─────────────────────────────────────────────────────────

function SessionTree({
  sessions, activeId, collapsed, onSelect, onRename, onArchive,
}: {
  sessions: SessionSummary[];
  activeId: string | null;
  collapsed: boolean;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onArchive: (id: string) => void;
}) {
  // Build parent → children map
  const childMap = useMemo(() => {
    const map: Record<string, SessionSummary[]> = {};
    for (const s of sessions) {
      const pid = s.parent_session_id || "__root__";
      if (!map[pid]) map[pid] = [];
      map[pid].push(s);
    }
    return map;
  }, [sessions]);

  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const toggleExpanded = useCallback((id: string) => {
    setExpanded(prev => ({ ...prev, [id]: !prev[id] }));
  }, []);

  function renderNode(s: SessionSummary, depth: number): React.ReactNode {
    const children = childMap[s.id] || [];
    const hasChildren = children.length > 0;
    const isExpanded = expanded[s.id] !== false; // default open
    return (
      <div key={s.id}>
        <SessionRow
          session={s}
          active={s.id === activeId}
          collapsed={collapsed}
          depth={depth}
          hasChildren={hasChildren}
          isExpanded={isExpanded}
          onToggleExpand={() => toggleExpanded(s.id)}
          onSelect={() => onSelect(s.id)}
          onRename={() => onRename(s.id, s.title)}
          onArchive={() => onArchive(s.id)}
        />
        {hasChildren && isExpanded && !collapsed && (
          <div style={{ borderLeft: "1px solid rgba(139,92,246,0.3)", marginLeft: 14, paddingLeft: 4 }}>
            {children.map(child => renderNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  }

  const roots = childMap["__root__"] || [];
  return <>{roots.map(s => renderNode(s, 0))}</>;
}

// ─── Session list ──────────────────────────────────────────────────────────

function SessionList({
  sessions, activeId, collapsed, onToggleCollapsed,
  onSelect, onCreate, onRename, onArchive, onClearAll,
}: {
  sessions: SessionSummary[];
  activeId: string | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRename: (id: string, current: string) => void;
  onArchive: (id: string) => void;
  onClearAll: () => void;
}) {
  // Fully collapsed: thin 28px strip with toggle + new-session button only.
  if (collapsed) {
    return (
      <aside style={{
        width: 28, flexShrink: 0, borderRight: "1px solid var(--rf-border)",
        background: "var(--rf-surface1)", display: "flex",
        flexDirection: "column", alignItems: "center", paddingTop: 10, gap: 6,
      }}>
        <button
          onClick={onToggleCollapsed}
          title="Expand session panel"
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "var(--rf-text4)", display: "flex", alignItems: "center",
            justifyContent: "center", padding: 4, borderRadius: 4,
          }}
        >
          <ChevronRightIcon size={14} />
        </button>
        <button
          onClick={onCreate}
          title="New investigation"
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "var(--rf-text4)", display: "flex", alignItems: "center",
            justifyContent: "center", padding: 4, borderRadius: 4,
          }}
        >
          <PlusIcon size={14} />
        </button>
        {sessions.length > 0 && (
          <div style={{
            fontSize: "8px", fontWeight: 700,
            color: "var(--rf-text5)", textAlign: "center",
          }}>{sessions.length}</div>
        )}
      </aside>
    );
  }

  return (
    <aside style={{
      width: 240, flexShrink: 0, borderRight: "1px solid var(--rf-border)",
      display: "flex", flexDirection: "column", background: "var(--rf-surface1)",
    }}>
      <div style={{
        padding: "12px 12px 8px",
        borderBottom: "1px solid var(--rf-border)",
        display: "flex", alignItems: "center", gap: 6,
      }}>
        <button
          onClick={onCreate}
          title="New investigation"
          style={{
            flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
            gap: 6, padding: "8px 10px", borderRadius: 8,
            background: "linear-gradient(135deg,#6366f1,#8b5cf6)", color: "white",
            border: "none", cursor: "pointer", fontSize: "12px", fontWeight: 600,
          }}
        >
          <PlusIcon size={13} />
          New investigation
        </button>
        <button
          onClick={onToggleCollapsed}
          title="Collapse session panel"
          aria-label="Collapse session panel"
          style={{
            ...iconBtnStyle(),
            width: 26, height: 26, color: "var(--rf-text4)",
          }}
        >
          <ChevronLeftIcon size={12} />
        </button>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 6px" }}>
        {sessions.length === 0 && (
          <div style={{ padding: 12, fontSize: "11px", color: "var(--rf-text5)" }}>
            No sessions yet. Start a new investigation.
          </div>
        )}
        <SessionTree
          sessions={sessions}
          activeId={activeId}
          collapsed={collapsed}
          onSelect={onSelect}
          onRename={onRename}
          onArchive={onArchive}
        />
      </div>
      {sessions.length > 0 && (
        <div style={{ padding: "8px 10px", borderTop: "1px solid var(--rf-border)" }}>
          <button
            onClick={onClearAll}
            style={{
              width: "100%", display: "flex", alignItems: "center", justifyContent: "center",
              gap: 6, padding: "6px 10px", borderRadius: 6,
              background: "transparent", color: "var(--rf-text5)",
              border: "1px solid var(--rf-border)",
              fontSize: "10.5px", fontWeight: 500, cursor: "pointer",
            }}
            title="Archive every active session"
          >
            <Trash2Icon size={11} /> Clear all sessions
          </button>
        </div>
      )}
    </aside>
  );
}

function SessionRow({
  session, active, collapsed, depth = 0, hasChildren = false, isExpanded = true,
  onToggleExpand, onSelect, onRename, onArchive,
}: {
  session: SessionSummary;
  active: boolean;
  collapsed: boolean;
  depth?: number;
  hasChildren?: boolean;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
  onSelect: () => void;
  onRename: () => void;
  onArchive: () => void;
}) {
  const [hover, setHover] = useState(false);
  // Backend writes a 1-2 sentence summary post-turn. Falls back to the
  // updated-at + namespace combo so collapsed-mode tooltips are still useful.
  const tooltip = (session.summary && session.summary.trim())
    ? session.summary
    : `${session.namespace_key} · last activity ${new Date(session.updated_at).toLocaleString()}`;

  if (collapsed) {
    // Initial-only icon button — hover reveals title + summary in a fly-out card.
    const initial = (session.title || "?").trim().charAt(0).toUpperCase() || "·";
    return (
      <div
        style={{ position: "relative", marginBottom: 4 }}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
      >
        <button
          onClick={onSelect}
          aria-label={session.title}
          style={{
            width: 36, height: 36, borderRadius: 8,
            background: active
              ? "linear-gradient(135deg,#6366f1,#8b5cf6)"
              : "var(--rf-surface2)",
            border: `1px solid ${active ? "transparent" : "var(--rf-border)"}`,
            color: active ? "white" : "var(--rf-text3)",
            cursor: "pointer", fontSize: "13px", fontWeight: 700,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          {session.parent_session_id
            ? <GitBranchIcon size={14} color={active ? "white" : "#a78bfa"} />
            : initial}
        </button>
        {hover && (
          <SessionTooltip session={session} summary={tooltip} side="right" />
        )}
      </div>
    );
  }

  const paddingLeft = 10 + depth * 8;
  return (
    <div
      onClick={onSelect}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        position: "relative", padding: `7px 30px 7px ${paddingLeft}px`,
        borderRadius: 6, marginBottom: 2,
        background: active ? "var(--rf-nav-active)" : "transparent",
        border: `1px solid ${active ? "var(--rf-nav-border)" : "transparent"}`,
        color: active ? "var(--rf-text1)" : "var(--rf-text3)",
        cursor: "pointer",
      }}
      title={tooltip}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        {hasChildren && onToggleExpand && (
          <button
            onClick={(e) => { e.stopPropagation(); onToggleExpand(); }}
            title={isExpanded ? "Collapse branches" : "Expand branches"}
            style={{ ...iconBtnStyle(), padding: 0, width: 14, height: 14, flexShrink: 0 }}
          >
            {isExpanded
              ? <ChevronDownIcon size={10} />
              : <ChevronRightIcon size={10} />}
          </button>
        )}
        {!hasChildren && session.parent_session_id && (
          <GitBranchIcon size={9} style={{ flexShrink: 0, color: "#a78bfa" }} />
        )}
        <div style={{
          fontSize: "11.5px", fontWeight: 600,
          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", flex: 1,
        }}>
          {session.title}
        </div>
      </div>
      <div style={{ fontSize: "9.5px", color: "var(--rf-text5)", marginTop: 2 }}>
        {session.namespace_key} · {new Date(session.updated_at).toLocaleDateString()}
        {hasChildren && <span style={{ marginLeft: 4, color: "#a78bfa" }}>· {isExpanded ? "↓" : "→"} branches</span>}
      </div>
      <div style={{ position: "absolute", right: 4, top: 6, display: "flex", gap: 2 }}>
        <button
          onClick={(e) => { e.stopPropagation(); onRename(); }}
          title="Rename"
          style={iconBtnStyle()}
        ><Edit3Icon size={11} /></button>
        <button
          onClick={(e) => { e.stopPropagation(); onArchive(); }}
          title="Archive"
          style={iconBtnStyle()}
        ><Trash2Icon size={11} /></button>
      </div>
    </div>
  );
}

function SessionTooltip({
  session, summary,
}: {
  session: SessionSummary;
  summary: string;
  side: "right";
}) {
  return (
    <div style={{
      position: "absolute",
      left: "calc(100% + 8px)",
      top: 0,
      width: 280,
      padding: "10px 12px",
      borderRadius: 8,
      background: "var(--rf-surface1)",
      border: "1px solid var(--rf-border)",
      boxShadow: "0 12px 28px rgba(0,0,0,0.18)",
      pointerEvents: "none",
      zIndex: 50,
      animation: "rfTipFade 0.12s ease-out",
    }}>
      <div style={{ fontSize: "11px", fontWeight: 700, color: "var(--rf-text1)", marginBottom: 4 }}>
        {session.title}
      </div>
      <div style={{ fontSize: "10.5px", color: "var(--rf-text3)", lineHeight: 1.45 }}>
        {summary}
      </div>
      <div style={{
        fontSize: "9.5px", color: "var(--rf-text5)", marginTop: 6,
        display: "flex", gap: 6, flexWrap: "wrap",
      }}>
        <span>{session.namespace_key}</span>
        {session.topic_keys.length > 0 && (
          <span>· {session.topic_keys.length} topic{session.topic_keys.length === 1 ? "" : "s"}</span>
        )}
        {session.parent_session_id && <span>· branched</span>}
        <span style={{ marginLeft: "auto" }}>
          {new Date(session.updated_at).toLocaleDateString()}
        </span>
      </div>
      <style>{`@keyframes rfTipFade { from { opacity: 0; transform: translateX(-4px) } to { opacity: 1; transform: translateX(0) } }`}</style>
    </div>
  );
}

function iconBtnStyle(): React.CSSProperties {
  return {
    background: "var(--rf-surface2)", border: "1px solid var(--rf-border)",
    borderRadius: 4, color: "var(--rf-text4)", cursor: "pointer",
    width: 22, height: 22, display: "flex", alignItems: "center", justifyContent: "center",
    padding: 0,
  };
}

// ─── Right rail — context ──────────────────────────────────────────────────

function ContextRail({
  session, attachments, onRemoveAttachment, runningTask, onCancelTask,
  collapsed, onToggleCollapsed, liveJobData,
}: {
  session: AssistantSession | null;
  attachments: Attachment[];
  onRemoveAttachment: (a: Attachment) => void;
  runningTask: AssistantTask | undefined;
  onCancelTask: (jobId: string) => void;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  liveJobData?: { rationale?: string; plannedSteps?: { tool: string; title: string }[]; actions?: string[] };
}) {
  const [contextOpen, setContextOpen] = useState(false);

  // Collapsed state: render a slim toggle strip only
  if (collapsed) {
    return (
      <aside style={{
        width: 28, flexShrink: 0, borderLeft: "1px solid var(--rf-border)",
        background: "var(--rf-surface1)", display: "flex",
        flexDirection: "column", alignItems: "center", paddingTop: 12,
      }}>
        <button
          onClick={onToggleCollapsed}
          title="Expand context panel"
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "var(--rf-text4)", display: "flex", alignItems: "center",
            justifyContent: "center", padding: 4, borderRadius: 4,
          }}
        >
          <ChevronLeftIcon size={14} />
        </button>
        {runningTask && (
          <div style={{
            marginTop: 8, width: 6, height: 6, borderRadius: "50%",
            background: "#6366f1", animation: "rfThinkPulse 1.4s ease-in-out infinite",
          }} title={runningTask.progress.summary || "Running"} />
        )}
        {attachments.length > 0 && (
          <div style={{
            marginTop: 8, fontSize: "8px", fontWeight: 700,
            color: "var(--rf-text5)", textAlign: "center",
          }}>{attachments.length}</div>
        )}
      </aside>
    );
  }

  return (
    <aside style={{
      width: 280, flexShrink: 0, borderLeft: "1px solid var(--rf-border)",
      background: "var(--rf-surface1)", overflowY: "auto",
      padding: "16px 14px",
    }}>
      {/* Rail header with collapse toggle */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <span style={{ fontSize: "9px", fontWeight: 700, color: "var(--rf-text5)",
                       textTransform: "uppercase", letterSpacing: "0.08em" }}>Context</span>
        <button
          onClick={onToggleCollapsed}
          title="Collapse context panel"
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "var(--rf-text5)", display: "flex", alignItems: "center",
            padding: 2, borderRadius: 4,
          }}
        >
          <ChevronRightIcon size={12} />
        </button>
      </div>

      {/* Collapsible active context section */}
      <button
        onClick={() => setContextOpen(o => !o)}
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          width: "100%", background: "none", border: "none", cursor: "pointer",
          padding: 0, marginBottom: contextOpen ? 8 : 16,
        }}
      >
        <SectionLabel style={{ margin: 0 }}>Active context</SectionLabel>
        <span style={{ fontSize: "9px", color: "var(--rf-text5)" }}>{contextOpen ? "−" : "+"}</span>
      </button>
      {contextOpen && !session && (
        <p style={{ fontSize: "10.5px", color: "var(--rf-text5)", marginBottom: 16 }}>
          Start an investigation to see its context here.
        </p>
      )}
      {contextOpen && session && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 16 }}>
          <RailRow label="Namespace" value={session.namespace_key} />
          <RailRow label="Topics" value={session.topic_keys.join(", ") || "—"} />
          <RailRow label="Profile" value={`${session.expertise_level} · ${session.orientation}`} />
          {session.parent_session_id && <RailRow label="Branched from" value="parent session" />}
        </div>
      )}

      {runningTask && (
        <>
          <SectionLabel>Running</SectionLabel>
          <div style={{
            padding: "10px 12px", borderRadius: 8, marginBottom: 12,
            background: "var(--rf-surface2)", border: "1px solid var(--rf-border)",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Loader2Icon size={12} className="animate-spin" color="#6366f1" />
              <span style={{ fontSize: "11px", color: "var(--rf-text2)", fontWeight: 600 }}>
                {runningTask.progress.summary || runningTask.status}
              </span>
            </div>
            {runningTask.progress.percent != null && (
              <div style={{ height: 3, background: "var(--rf-surface3)", borderRadius: 2, marginTop: 6, overflow: "hidden" }}>
                <div style={{
                  height: "100%", width: `${runningTask.progress.percent}%`,
                  background: "linear-gradient(90deg,#6366f1,#8b5cf6)",
                  transition: "width 0.4s ease",
                }} />
              </div>
            )}
            <button
              onClick={() => onCancelTask(runningTask.job_id)}
              style={{
                marginTop: 8, width: "100%", padding: "5px 8px", borderRadius: 5,
                border: "1px solid rgba(239,68,68,0.3)", color: "#ef4444",
                background: "rgba(239,68,68,0.08)", fontSize: "10.5px",
                fontWeight: 600, cursor: "pointer",
              }}
            >Stop turn</button>
          </div>

          {/* Agent reasoning — rationale + planned steps from plan_committed SSE */}
          {liveJobData && (liveJobData.rationale || liveJobData.plannedSteps?.length) && (
            <div style={{
              padding: "10px 12px", borderRadius: 8, marginBottom: 16,
              background: "rgba(99,102,241,0.06)", border: "1px solid rgba(99,102,241,0.2)",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 6 }}>
                <SparklesIcon size={10} color="#818cf8" />
                <span style={{ fontSize: "9px", fontWeight: 700, color: "#818cf8",
                               textTransform: "uppercase", letterSpacing: "0.07em" }}>
                  Agent reasoning
                </span>
              </div>
              {liveJobData.rationale && (
                <p style={{
                  fontSize: "10.5px", color: "var(--rf-text3)", lineHeight: 1.5,
                  fontStyle: "italic", marginBottom: liveJobData.plannedSteps?.length ? 8 : 0,
                }}>
                  {liveJobData.rationale}
                </p>
              )}
              {liveJobData.plannedSteps && liveJobData.plannedSteps.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  {liveJobData.plannedSteps.map((s, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                      <CircleDashedIcon size={9} color="#6366f1" style={{ flexShrink: 0 }} />
                      <span style={{ fontSize: "10px", color: "var(--rf-text4)" }}>
                        <span style={{ fontWeight: 600, color: "var(--rf-text3)" }}>{s.tool}</span>
                        {" — "}{s.title}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}

      <SectionLabel>
        Attachments {attachments.length > 0 && <span style={{ color: "var(--rf-text5)" }}>· {attachments.length}</span>}
      </SectionLabel>
      {attachments.length === 0 && (
        <p style={{ fontSize: "10.5px", color: "var(--rf-text5)" }}>
          Drop a note or paste a URL via the paperclip in the composer.
        </p>
      )}
      {attachments.map(a => (
        <div key={a.id} style={{
          display: "flex", alignItems: "flex-start", gap: 6,
          padding: "6px 8px", borderRadius: 6, marginBottom: 4,
          background: "var(--rf-surface2)", border: "1px solid var(--rf-border)",
        }}>
          <AttachmentIcon kind={a.kind} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: "10.5px", fontWeight: 600, color: "var(--rf-text2)",
                          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {a.label || a.url || a.kind}
            </div>
            {a.url && (
              <a href={a.url} target="_blank" rel="noopener noreferrer"
                 style={{ fontSize: "9.5px", color: "#818cf8", textDecoration: "none",
                          display: "block", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {a.url}
              </a>
            )}
          </div>
          <button
            onClick={() => onRemoveAttachment(a)}
            style={iconBtnStyle()} title="Remove"
          ><XIcon size={10} /></button>
        </div>
      ))}
    </aside>
  );
}

function AttachmentIcon({ kind }: { kind: Attachment["kind"] }) {
  if (kind === "url") return <GlobeIcon size={12} color="var(--rf-text4)" style={{ marginTop: 2 }} />;
  if (kind === "paper_ref") return <FileTextIcon size={12} color="var(--rf-text4)" style={{ marginTop: 2 }} />;
  if (kind === "pdf") return <BookOpenIcon size={12} color="var(--rf-text4)" style={{ marginTop: 2 }} />;
  return <PaperclipIcon size={12} color="var(--rf-text4)" style={{ marginTop: 2 }} />;
}

function SectionLabel({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <p style={{
      fontSize: "9px", fontWeight: 700, color: "var(--rf-text5)",
      textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8,
      ...style,
    }}>{children}</p>
  );
}

function RailRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
      <span style={{ fontSize: "10.5px", color: "var(--rf-text5)" }}>{label}</span>
      <span style={{ fontSize: "10.5px", color: "var(--rf-text2)", fontWeight: 600,
                     textAlign: "right", maxWidth: "60%",
                     whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{value}</span>
    </div>
  );
}

// ─── Composer ──────────────────────────────────────────────────────────────

function Composer({
  inputRef, input, setInput, onSubmit, submitting,
  pendingAttachments, onRemoveAttachment, onAttachNote, onAttachUrl, onAttachFile,
}: {
  inputRef: React.RefObject<HTMLTextAreaElement>;
  input: string;
  setInput: (s: string) => void;
  onSubmit: () => void;
  submitting: boolean;
  pendingAttachments: Attachment[];
  onRemoveAttachment: (a: Attachment) => void;
  onAttachNote: (label: string, content: string) => void;
  onAttachUrl: (url: string, label: string) => void;
  onAttachFile: (file: File) => Promise<void>;
}) {
  const [showAttach, setShowAttach] = useState(false);

  return (
    <footer style={{ padding: "12px 18px 16px", borderTop: "1px solid var(--rf-border)" }}>
      {pendingAttachments.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 6 }}>
          {pendingAttachments.map(a => (
            <span key={a.id} style={{
              display: "flex", alignItems: "center", gap: 4,
              padding: "3px 8px", borderRadius: 12, fontSize: "10px",
              background: "rgba(99,102,241,0.12)", color: "#818cf8",
              border: "1px solid rgba(99,102,241,0.25)",
            }}>
              <AttachmentIcon kind={a.kind} />
              <span style={{ maxWidth: 160, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {a.label || a.url || a.kind}
              </span>
              <button
                onClick={() => onRemoveAttachment(a)}
                style={{ background: "none", border: "none", color: "#818cf8", cursor: "pointer", padding: 0 }}
              ><XIcon size={10} /></button>
            </span>
          ))}
        </div>
      )}
      <div style={{
        position: "relative",
        display: "flex", gap: 8, alignItems: "flex-end",
        background: "var(--rf-surface2)", borderRadius: 10,
        padding: "8px 10px", border: "1px solid var(--rf-border)",
      }}>
        <button
          onClick={() => setShowAttach(s => !s)}
          title="Attach a note or URL"
          style={{
            padding: 6, borderRadius: 6, background: "none", border: "none",
            color: "var(--rf-text4)", cursor: "pointer", flexShrink: 0,
          }}
        >
          <PaperclipIcon size={14} />
        </button>
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSubmit(); }
          }}
          placeholder="Ask, explore, or describe what you want to investigate…"
          rows={2}
          style={{
            flex: 1, background: "transparent", border: "none", outline: "none",
            fontSize: "12.5px", color: "var(--rf-text1)", resize: "none",
            fontFamily: "inherit", lineHeight: 1.5,
          }}
        />
        <button
          onClick={onSubmit}
          disabled={submitting || !input.trim()}
          style={{
            padding: "8px 14px", borderRadius: 8, border: "none",
            background: submitting || !input.trim()
              ? "var(--rf-surface3)"
              : "linear-gradient(135deg,#6366f1,#8b5cf6)",
            color: "white", fontSize: "12px", fontWeight: 600,
            display: "flex", alignItems: "center", gap: 6,
            cursor: submitting || !input.trim() ? "not-allowed" : "pointer",
          }}
        >
          {submitting ? <Loader2Icon size={13} className="animate-spin" /> : <SendIcon size={13} />}
          Send
        </button>
        {showAttach && (
          <AttachPopover
            onAttachNote={(label, content) => { onAttachNote(label, content); setShowAttach(false); }}
            onAttachUrl={(url, label) => { onAttachUrl(url, label); setShowAttach(false); }}
            onAttachFile={async (file) => { await onAttachFile(file); setShowAttach(false); }}
            onClose={() => setShowAttach(false)}
          />
        )}
      </div>
      <p style={{ fontSize: "9.5px", color: "var(--rf-text5)", margin: "6px 4px 0", textAlign: "center" }}>
        Composes Deep Search · arXiv MCP · Genie · Web · Compare · Bookmarks. Heavy work runs in the background.
      </p>
    </footer>
  );
}

function AttachPopover({
  onAttachNote, onAttachUrl, onAttachFile, onClose,
}: {
  onAttachNote: (label: string, content: string) => void;
  onAttachUrl: (url: string, label: string) => void;
  onAttachFile: (file: File) => Promise<void> | void;
  onClose: () => void;
}) {
  const [tab, setTab] = useState<"note" | "url" | "file">("note");
  const [noteLabel, setNoteLabel] = useState("");
  const [noteContent, setNoteContent] = useState("");
  const [url, setUrl] = useState("");
  const [urlLabel, setUrlLabel] = useState("");
  const [fileBusy, setFileBusy] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFileError(null);
    setFileBusy(true);
    try {
      await onAttachFile(f);
    } catch (err) {
      setFileError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setFileBusy(false);
      // Reset input so re-uploading the same filename re-fires the change event.
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div style={{
      position: "absolute", bottom: "100%", left: 0, marginBottom: 8,
      width: 360, zIndex: 10,
      background: "var(--rf-surface)", border: "1px solid var(--rf-border)",
      borderRadius: 10, padding: 12, boxShadow: "var(--rf-shadow-lg)",
    }}>
      <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
        {(["note", "url", "file"] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: "5px 12px", borderRadius: 6, fontSize: "10.5px", fontWeight: 600,
              background: tab === t ? "var(--rf-nav-active)" : "transparent",
              border: `1px solid ${tab === t ? "var(--rf-nav-border)" : "var(--rf-border)"}`,
              color: tab === t ? "var(--rf-text1)" : "var(--rf-text4)",
              cursor: "pointer", textTransform: "capitalize",
            }}
          >{t === "file" ? "PDF / image" : t}</button>
        ))}
        <button onClick={onClose} style={{ marginLeft: "auto", ...iconBtnStyle() }}>
          <XIcon size={11} />
        </button>
      </div>
      {tab === "note" && (
        <>
          <input
            value={noteLabel} onChange={e => setNoteLabel(e.target.value)}
            placeholder="Title (optional)"
            style={popoverInputStyle()}
          />
          <textarea
            value={noteContent} onChange={e => setNoteContent(e.target.value)}
            placeholder="Paste a note, observation, or fragment of an idea…"
            rows={4}
            style={{ ...popoverInputStyle(), resize: "vertical" }}
          />
          <button
            onClick={() => onAttachNote(noteLabel, noteContent)}
            disabled={!noteContent.trim()}
            style={popoverSubmitStyle(noteContent.trim().length > 0)}
          >Attach note</button>
        </>
      )}
      {tab === "url" && (
        <>
          <input
            value={url} onChange={e => setUrl(e.target.value)}
            placeholder="https://…"
            style={popoverInputStyle()}
          />
          <input
            value={urlLabel} onChange={e => setUrlLabel(e.target.value)}
            placeholder="Label (optional)"
            style={popoverInputStyle()}
          />
          <button
            onClick={() => onAttachUrl(url, urlLabel)}
            disabled={!url.trim()}
            style={popoverSubmitStyle(url.trim().length > 0)}
          >Attach URL</button>
        </>
      )}
      {tab === "file" && (
        <>
          <p style={{
            fontSize: "10.5px", color: "var(--rf-text4)", marginBottom: 8, lineHeight: 1.45,
          }}>
            Drop a PDF (paper, thesis, draft) or an image (screenshot, diagram, whiteboard).
            Text is extracted and indexed for this session only — never added to the public feed.
          </p>
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf,image/*"
            onChange={handleFileChange}
            style={{ display: "none" }}
          />
          <button
            onClick={() => fileRef.current?.click()}
            disabled={fileBusy}
            style={popoverSubmitStyle(!fileBusy)}
          >
            {fileBusy ? "Extracting…" : "Choose file"}
          </button>
          {fileError && (
            <p style={{ fontSize: "10.5px", color: "var(--rf-destructive)", marginTop: 6 }}>
              {fileError}
            </p>
          )}
          <p style={{ fontSize: "9.5px", color: "var(--rf-text5)", marginTop: 8 }}>
            Limit: 25 MB. Parsing runs through Marker (PDF) or vision LLM (image).
          </p>
        </>
      )}
    </div>
  );
}

function popoverInputStyle(): React.CSSProperties {
  return {
    width: "100%", padding: "7px 10px", borderRadius: 6, marginBottom: 6,
    background: "var(--rf-surface2)", border: "1px solid var(--rf-border)",
    color: "var(--rf-text1)", fontSize: "11.5px", outline: "none",
    fontFamily: "inherit",
  };
}

function popoverSubmitStyle(enabled: boolean): React.CSSProperties {
  return {
    width: "100%", padding: "7px 10px", borderRadius: 6, marginTop: 4,
    background: enabled ? "linear-gradient(135deg,#6366f1,#8b5cf6)" : "var(--rf-surface3)",
    color: enabled ? "white" : "var(--rf-text5)",
    border: "none", fontSize: "11px", fontWeight: 600,
    cursor: enabled ? "pointer" : "not-allowed",
  };
}

// ─── Message + reasoning tree ──────────────────────────────────────────────

function MessageBlock({
  msg, steps, tasks, onSuggestionClick, onBranch, onCancel, onOpenPaper,
  liveJobData, streamingContent,
  stickyNotes = [], onAddNote, onUpdateNote, onDeleteNote,
  highlightsForMessage, searchQuery, onRemoveHighlight,
}: {
  msg: AssistantMessage;
  steps: AssistantStep[];
  tasks: AssistantTask[];
  onSuggestionClick: (text: string) => void;
  onBranch: () => void;
  onCancel: (jobId: string) => void;
  onOpenPaper: (paperId: string) => void;
  liveJobData: Record<string, { rationale?: string; plannedSteps?: { tool: string; title: string }[]; actions?: string[] }>;
  streamingContent: Record<string, string>;
  stickyNotes?: StickyNote[];
  onAddNote?: () => void;
  onUpdateNote?: (id: string, patch: Partial<StickyNote>) => void;
  onDeleteNote?: (id: string) => void;
  /** Highlights filtered to just this message — empty array when none. */
  highlightsForMessage?: Highlight[];
  /** Current chat-search query — drives the search ``<mark>`` rendering. */
  searchQuery?: string;
  /** Called when the user clicks a highlight mark to remove it. */
  onRemoveHighlight?: (id: string) => void;
}) {
  const isUser = msg.role === "user";
  const isSystem = msg.role === "system";
  const taskForMsg = tasks.find(t => t.assistant_message_id === msg.id);

  if (isSystem) {
    return (
      <div style={{
        margin: "10px 0", padding: "8px 12px", borderRadius: 6,
        background: "var(--rf-surface2)", color: "var(--rf-text4)", fontSize: "10.5px",
        fontStyle: "italic",
      }}>
        {msg.content}
      </div>
    );
  }

  return (
    <div className="rf-message-enter" style={{ marginBottom: 22 }}>
      <div style={{ display: "flex", gap: 10, flexDirection: isUser ? "row-reverse" : "row" }}>
        <div style={{
          width: 26, height: 26, borderRadius: "50%", flexShrink: 0,
          background: isUser
            ? "linear-gradient(135deg,#6366f1,#8b5cf6)"
            : "linear-gradient(135deg,#0ea5e9,#06b6d4)",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          {isUser
            ? <span style={{ color: "white", fontSize: "10px", fontWeight: 700 }}>U</span>
            : <SparklesIcon size={13} color="white" />}
        </div>
        {/* data-message-id lets the highlight engine locate this element */}
        <div
          data-message-id={msg.id}
          style={{
            flex: 1, maxWidth: "calc(100% - 50px)",
            background: isUser ? "var(--rf-nav-active)" : "var(--rf-surface1)",
            border: "1px solid var(--rf-border)",
            borderRadius: 10, padding: "10px 14px",
            transition: "border-color 0.2s ease",
          }}
        >
          {msg.role === "assistant" && taskForMsg && (
            <ReasoningStrip
              task={taskForMsg}
              steps={steps}
              onCancel={() => onCancel(taskForMsg.job_id)}
              liveData={liveJobData[taskForMsg.job_id]}
            />
          )}
          <MessageBody
            msg={msg}
            isInflight={!!(taskForMsg && (taskForMsg.status === "running" || taskForMsg.status === "pending"))}
            onSuggestionClick={onSuggestionClick}
            onOpenPaper={onOpenPaper}
            streamingText={streamingContent[msg.id]}
            highlightsForMessage={highlightsForMessage}
            searchQuery={searchQuery}
            onRemoveHighlight={onRemoveHighlight}
          />
          <div style={{ marginTop: 8, display: "flex", justifyContent: "flex-end", gap: 4 }}>
            {onAddNote && (
              <button
                onClick={onAddNote}
                title="Add sticky note"
                style={{
                  display: "flex", alignItems: "center", gap: 4,
                  fontSize: "10px", color: "var(--rf-text5)",
                  background: "none", border: "none", cursor: "pointer", padding: "2px 6px",
                }}
              >
                <StickyNoteIcon size={10} /> Note
              </button>
            )}
            {msg.role === "assistant" && (
              <button
                onClick={onBranch}
                style={{
                  display: "flex", alignItems: "center", gap: 4,
                  fontSize: "10px", color: "var(--rf-text5)",
                  background: "none", border: "none", cursor: "pointer", padding: "2px 6px",
                }}
                title="Branch a new investigation from this point"
              >
                <GitBranchIcon size={10} /> Branch from here
              </button>
            )}
          </div>
          {/* Sticky notes anchored to this message */}
          {stickyNotes.length > 0 && onUpdateNote && onDeleteNote && (
            <StickyNotesLayer
              notes={stickyNotes}
              onUpdate={onUpdateNote}
              onDelete={onDeleteNote}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function ReasoningStrip({
  task, steps, onCancel, liveData,
}: {
  task: AssistantTask;
  steps: AssistantStep[];
  onCancel: () => void;
  liveData?: { rationale?: string; plannedSteps?: { tool: string; title: string }[]; actions?: string[] };
}) {
  const isInflight = task.status === "running" || task.status === "pending";
  // Open while a task is in-flight so the user can see progress; collapse
  // automatically when done so completed reasoning trees don't clutter the
  // conversation. Users can still expand them manually.
  const [open, setOpen] = useState(isInflight);
  const orderedSteps = [...steps].sort((a, b) => a.step_index - b.step_index);

  return (
    <div style={{
      marginBottom: 8, padding: "6px 8px", borderRadius: 6,
      background: "var(--rf-surface2)", border: "1px solid var(--rf-border)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <button
          onClick={() => setOpen(o => !o)}
          style={{
            display: "flex", alignItems: "center", gap: 6, flex: 1,
            background: "none", border: "none", cursor: "pointer", padding: 0, textAlign: "left",
            color: "var(--rf-text3)", fontSize: "10.5px", fontWeight: 600,
          }}
        >
          <StatusIcon status={task.status} />
          <span>
            {task.progress.summary || task.status}
            {task.progress.percent != null && ` · ${task.progress.percent}%`}
          </span>
        </button>
        {isInflight && (
          <button
            onClick={onCancel}
            style={{
              fontSize: "9.5px", padding: "2px 7px", borderRadius: 4,
              background: "rgba(239,68,68,0.1)", color: "#ef4444",
              border: "1px solid rgba(239,68,68,0.3)", cursor: "pointer", fontWeight: 600,
            }}
          >Stop</button>
        )}
        <span style={{ color: "var(--rf-text5)" }}>{open ? "−" : "+"}</span>
      </div>
      {open && (
        <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 3 }}>
          {/* Plan rationale from the planner — visible before any steps execute */}
          {liveData?.rationale && (
            <div style={{
              fontSize: "10px", color: "var(--rf-text4)", fontStyle: "italic",
              padding: "3px 6px", borderRadius: 4,
              borderLeft: "2px solid var(--rf-border)", marginBottom: 2,
            }}>
              {liveData.rationale}
            </div>
          )}
          {/* DB steps (ground truth — available once steps start writing to DB) */}
          {orderedSteps.length > 0 && orderedSteps.map(s => (
            <div key={s.id} style={{
              display: "flex", alignItems: "center", gap: 6, padding: "3px 4px",
              borderRadius: 4, fontSize: "10px",
            }}>
              <StatusIcon status={s.status} small />
              <span style={{ color: "var(--rf-text2)", fontWeight: 600 }}>{s.tool_name}</span>
              <span style={{ color: "var(--rf-text4)", flex: 1, overflow: "hidden",
                              textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {s.progress.summary || s.title}
              </span>
              {s.cost && (s.cost as { cache_hit?: boolean }).cache_hit && (
                <span style={{ fontSize: "9px", color: "#22c55e" }} title="Cache hit — no LLM cost">⚡</span>
              )}
              {s.error && (
                <span style={{ color: "#ef4444", fontSize: "9.5px" }} title={s.error}>error</span>
              )}
            </div>
          ))}
          {/* Planned step stubs from SSE plan_committed event — shown before DB steps land */}
          {orderedSteps.length === 0 && liveData?.plannedSteps && liveData.plannedSteps.map((ps, i) => (
            <div key={i} style={{
              display: "flex", alignItems: "center", gap: 6, padding: "3px 4px",
              borderRadius: 4, fontSize: "10px", opacity: 0.6,
            }}>
              <CircleDashedIcon size={11} color="var(--rf-text5)" />
              <span style={{ color: "var(--rf-text3)", fontWeight: 600 }}>{ps.tool}</span>
              <span style={{ color: "var(--rf-text5)", flex: 1, overflow: "hidden",
                              textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {ps.title}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StatusIcon({ status, small = false }: { status: string; small?: boolean }) {
  const size = small ? 11 : 12;
  const colour = STATUS_COLOUR[status] || "var(--rf-text5)";
  if (status === "running" || status === "pending") {
    return <Loader2Icon size={size} color={colour} className="animate-spin" />;
  }
  if (status === "completed") return <CheckCircle2Icon size={size} color={colour} />;
  if (status === "failed" || status === "cancelled") return <XCircleIcon size={size} color={colour} />;
  return <CircleDashedIcon size={size} color={colour} />;
}

// ─── Block renderer ────────────────────────────────────────────────────────

function ThinkingDots() {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
      {[0, 1, 2].map(i => (
        <span
          key={i}
          style={{
            width: 5, height: 5, borderRadius: "50%",
            background: "var(--rf-text4)",
            display: "inline-block",
            animation: "rfThinkPulse 1.4s ease-in-out infinite",
            animationDelay: `${i * 0.22}s`,
          }}
        />
      ))}
      <style>{`@keyframes rfThinkPulse { 0%,80%,100% { opacity: 0.2; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1.1); } }`}</style>
    </span>
  );
}

function MessageBody({
  msg, isInflight = false, onSuggestionClick, onOpenPaper, streamingText,
  highlightsForMessage, searchQuery, onRemoveHighlight,
}: {
  msg: AssistantMessage;
  isInflight?: boolean;
  onSuggestionClick: (text: string) => void;
  onOpenPaper: (paperId: string) => void;
  streamingText?: string;
  highlightsForMessage?: Highlight[];
  searchQuery?: string;
  onRemoveHighlight?: (id: string) => void;
}) {
  // Build the decoration object passed to MarkdownRenderer. The renderer
  // emits <mark> JSX elements for each match, so React owns them and they
  // survive every re-render — no DOM mutation, no observer loop.
  const decorations = useMemo(() => ({
    highlights: (highlightsForMessage || []).map(h => ({ id: h.id, text: h.text, color: h.color })),
    searchQuery: searchQuery || "",
    searchKeyPrefix: msg.id,
    onRemoveHighlight,
  }), [highlightsForMessage, searchQuery, msg.id, onRemoveHighlight]);
  const blocks = (msg.payload?.blocks as Block[] | undefined) || [];

  // Build a 1-based index map for citations: {1: paper_id, A1: paper_id}
  const citationMap = useMemo(() => {
    const map: Record<string, string> = {};
    for (const b of blocks) {
      if (b.kind === "paper_grid" && b.papers) {
        b.papers.forEach((p: PaperBlock, idx: number) => {
          if (p.paper_id) map[String(idx + 1)] = p.paper_id;
        });
      } else if (b.kind === "arxiv_grid" && b.papers) {
        b.papers.forEach((p: ArxivCandidate, idx: number) => {
          if (p.external_id) map[`A${idx + 1}`] = p.external_id;
        });
      }
    }
    return map;
  }, [blocks]);

  const handleCitation = useCallback((num: string, _isArxiv: boolean) => {
    const paperId = citationMap[num];
    if (paperId) onOpenPaper(paperId);
  }, [citationMap, onOpenPaper]);

  // Show live streaming text (synthesis tokens arriving) or Thinking... animation
  if (!blocks.length && !msg.content && isInflight) {
    if (streamingText) {
      return (
        <div style={{ fontSize: "12.5px", color: "var(--rf-text1)", lineHeight: 1.55 }}>
          <MarkdownRenderer content={streamingText} onCitationClick={handleCitation} decorations={decorations} />
          <span style={{ display: "inline-flex", alignItems: "center", gap: 3, marginTop: 4 }}>
            <ThinkingDots />
          </span>
        </div>
      );
    }
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0",
                    fontSize: "12px", color: "var(--rf-text4)" }}>
        <ThinkingDots />
        <span style={{ fontStyle: "italic" }}>Thinking…</span>
      </div>
    );
  }

  if (!blocks.length) {
    // Don't render empty assistant messages that have no content
    if (!msg.content) return null;
    return (
      <div style={{ fontSize: "12.5px", color: "var(--rf-text1)", lineHeight: 1.55 }}>
        <MarkdownRenderer content={msg.content} onCitationClick={handleCitation} decorations={decorations} />
      </div>
    );
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {blocks.map((b, i) => (
        <BlockRenderer key={i} block={b} onSuggestionClick={onSuggestionClick} onOpenPaper={onOpenPaper} onCitationClick={handleCitation} decorations={decorations} />
      ))}
    </div>
  );
}

function BlockRenderer({
  block, onSuggestionClick, onOpenPaper, onCitationClick, decorations,
}: {
  block: Block;
  onSuggestionClick: (text: string) => void;
  onOpenPaper: (paperId: string) => void;
  onCitationClick?: (num: string, isArxiv: boolean) => void;
  decorations?: import("@/components/ui/MarkdownRenderer").MarkdownDecorations;
}) {
  switch (block.kind) {
    case "text":
      return (
        <div style={{ fontSize: "12.5px", color: "var(--rf-text1)", lineHeight: 1.55 }}>
          <MarkdownRenderer content={block.content || ""} onCitationClick={onCitationClick} decorations={decorations} />
        </div>
      );
    case "paper_grid":
      return <PaperGridBlock title={block.title} papers={block.papers} onOpenPaper={onOpenPaper} />;
    case "arxiv_grid":
      return <ArxivGridBlock title={block.title} papers={block.papers} importedCount={block.imported_count} />;
    case "web_results":
      return <WebResultsBlock title={block.title} results={block.results} />;
    case "comparison_table":
      return (
        <ComparisonTableBlock title={block.title} columns={block.columns} rows={block.rows}
                              notes={block.notes} onOpenPaper={onOpenPaper} />
      );
    case "bookmarks_answer":
      return <BookmarksAnswerBlock title={block.title} content={block.content} />;
    case "graph_summary":
      return <GraphSummaryBlock title={block.title} summary={block.summary} href={block.href} />;
    case "artifact_link":
      return <ArtifactLinkBlock title={block.title} href={block.href} kindLabel={block.kind_label} />;
    case "mermaid":
      return <MermaidBlock title={block.title} code={block.code} />;
    case "source_papers":
      return <SourcePapersBlock title={block.title} papers={block.papers} />;
    case "nvd_results":
      return <NvdResultsBlock title={block.title} vulnerabilities={block.vulnerabilities} />;
    case "trials_results":
      return <TrialsResultsBlock title={block.title} studies={block.studies} />;
    case "fred_data":
      return <FredDataBlock title={block.title} series={block.series} />;
    case "code_results":
      return <CodeResultsBlock title={block.title} items={block.items} />;
    case "suggestion_chips":
      return null;
    case "actions_taken":
      return <ActionsTakenBlock actions={block.actions} />;
    default:
      return null;
  }
}

function PaperGridBlock({
  title, papers, onOpenPaper,
}: { title?: string; papers: PaperBlock[]; onOpenPaper: (id: string) => void }) {
  // Collapsed by default — grounded papers are available and easy to inspect
  // but shouldn't dominate the visible answer.
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: "flex", alignItems: "center", gap: 8, width: "100%",
          background: "none", border: "none", cursor: "pointer", padding: 0, marginBottom: open ? 6 : 0,
          textAlign: "left",
        }}
      >
        {title && <BlockTitle style={{ marginBottom: 0, flex: 1 }}>{title}</BlockTitle>}
        <span style={{
          fontSize: "9px", color: "var(--rf-text5)",
          padding: "2px 7px", borderRadius: 8,
          background: "var(--rf-surface3)", border: "1px solid var(--rf-border)",
        }}>
          {papers.length} paper{papers.length === 1 ? "" : "s"} {open ? "−" : "+"}
        </span>
      </button>
      {open && (
        <div style={{ display: "grid", gap: 6 }}>
          {papers.map(p => (
            <PaperCardInline key={p.paper_id || p.title} paper={p} onOpenPaper={onOpenPaper} />
          ))}
        </div>
      )}
    </div>
  );
}

function ArxivGridBlock({
  title, papers, importedCount,
}: { title?: string; papers: ArxivCandidate[]; importedCount?: number }) {
  return (
    <div>
      {title && <BlockTitle>{title}{importedCount ? ` · ${importedCount} new` : ""}</BlockTitle>}
      <div style={{ display: "grid", gap: 6 }}>
        {papers.map((p, i) => (
          <a key={p.external_id || i}
             href={p.external_id ? `https://arxiv.org/abs/${p.external_id}` : "#"}
             target="_blank" rel="noopener noreferrer"
             style={{ ...cardStyle(), textDecoration: "none", display: "block" }}>
            <div style={{ color: "var(--rf-text1)", fontWeight: 600, fontSize: "12px" }}>
              {p.title || "Untitled"}
            </div>
            <div style={{ fontSize: "10.5px", color: "var(--rf-text4)", marginTop: 3 }}>
              {(p.authors || []).slice(0, 3).join(", ")}
              {(p.authors || []).length > 3 ? ", …" : ""}
            </div>
            {p.abstract && (
              <div style={{ fontSize: "11px", color: "var(--rf-text3)", marginTop: 5, lineHeight: 1.45 }}>
                {p.abstract.slice(0, 240)}{p.abstract.length > 240 ? "…" : ""}
              </div>
            )}
          </a>
        ))}
      </div>
    </div>
  );
}

function PaperCardInline({
  paper, onOpenPaper,
}: { paper: PaperBlock; onOpenPaper: (id: string) => void }) {
  const why = paper.why_surfaced || whyBadges(paper);
  // Click anywhere on the card to open the inline PaperPanel; the small
  // external-link icon in the header opens arXiv in a new tab for users
  // who want the source directly.
  const handleOpen = () => paper.paper_id && onOpenPaper(paper.paper_id);
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleOpen}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleOpen(); }
      }}
      style={{
        ...cardStyle(),
        cursor: paper.paper_id ? "pointer" : "default",
        transition: "border-color 150ms ease, box-shadow 150ms ease",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLDivElement).style.borderColor = "var(--rf-accent-border)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLDivElement).style.borderColor = "var(--rf-card-border)";
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
        <FileTextIcon size={13} color="var(--rf-text4)" style={{ flexShrink: 0, marginTop: 2 }} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            display: "flex", alignItems: "flex-start", gap: 8,
            color: "var(--rf-text1)", fontWeight: 600, fontSize: "12.5px", lineHeight: 1.35,
          }}>
            <span style={{ flex: 1 }}>{paper.title || "Untitled"}</span>
            {paper.source_url && (
              <a
                href={paper.source_url}
                target="_blank" rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                title="Open source on arXiv"
                style={{ color: "var(--rf-text4)", display: "inline-flex" }}
              ><ExternalLinkIcon size={11} /></a>
            )}
          </div>
          <div style={{ fontSize: "10.5px", color: "var(--rf-text4)", marginTop: 3 }}>
            {(paper.authors || []).slice(0, 3).join(", ")}
            {(paper.authors || []).length > 3 ? ", …" : ""}
            {paper.namespace_key && ` · ${paper.namespace_key}`}
          </div>
          {(paper.tldr || paper.abstract) && (
            <div style={{ fontSize: "11.5px", color: "var(--rf-text3)", marginTop: 5, lineHeight: 1.5 }}>
              {(paper.tldr || paper.abstract || "").slice(0, 240)}
              {(paper.tldr || paper.abstract || "").length > 240 ? "…" : ""}
            </div>
          )}
          {why.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 6 }}>
              {why.map((w, i) => (
                <span key={`${w.label}-${i}`} style={{
                  fontSize: "9.5px", padding: "1.5px 6px", borderRadius: 8,
                  background: "var(--rf-accent-bg)", color: "var(--rf-accent)",
                  border: "1px solid var(--rf-accent-border)", fontWeight: 600,
                }}>{w.label}</span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function whyBadges(paper: PaperBlock): { label: string; signal: string }[] {
  const out: { label: string; signal: string }[] = [];
  if ((paper.novelty_score ?? 0) >= 0.78) out.push({ signal: "novelty", label: "high novelty" });
  if ((paper.relevance_score ?? 0) >= 0.78) out.push({ signal: "relevance", label: "highly relevant" });
  if (paper.match_type === "arxiv_imported") out.push({ signal: "fresh", label: "fresh import" });
  if (paper.match_type === "frontier") out.push({ signal: "frontier", label: "frontier" });
  return out;
}

function WebResultsBlock({
  title, results,
}: { title?: string; results: { title: string; url: string; snippet: string }[] }) {
  return (
    <div>
      {title && <BlockTitle>{title}</BlockTitle>}
      <div style={{ display: "grid", gap: 6 }}>
        {results.map((r, i) => (
          <a key={i} href={r.url} target="_blank" rel="noopener noreferrer"
             style={{ ...cardStyle(), textDecoration: "none", display: "block" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <GlobeIcon size={11} color="var(--rf-text4)" />
              <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--rf-text1)" }}>{r.title}</span>
            </div>
            <div style={{ fontSize: "10px", color: "#818cf8", marginTop: 3,
                          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {r.url}
            </div>
            {r.snippet && (
              <div style={{ fontSize: "11px", color: "var(--rf-text3)", marginTop: 5, lineHeight: 1.45 }}>
                {r.snippet.slice(0, 220)}…
              </div>
            )}
          </a>
        ))}
      </div>
    </div>
  );
}

function ComparisonTableBlock({
  title, columns, rows, notes, onOpenPaper,
}: {
  title?: string; columns: ComparisonColumn[]; rows: ComparisonRow[];
  notes?: string; onOpenPaper: (id: string) => void;
}) {
  return (
    <div>
      {title && <BlockTitle>{title}</BlockTitle>}
      <div style={{ ...cardStyle(), padding: 0, overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "10.5px", color: "var(--rf-text2)" }}>
          <thead>
            <tr>
              <th style={comparisonCellStyle(true)}>dimension</th>
              {columns.map(c => (
                <th key={c.paper_id} style={comparisonCellStyle(true)}>
                  <button
                    onClick={() => onOpenPaper(c.paper_id)}
                    style={{
                      color: "var(--rf-text1)", background: "none", border: "none",
                      cursor: "pointer", fontWeight: 600, padding: 0, textAlign: "left",
                      fontSize: "9px", textTransform: "uppercase", letterSpacing: "0.05em",
                    }}
                    title="Open paper details"
                  >
                    {c.title.slice(0, 60)}{c.title.length > 60 ? "…" : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.dimension}>
                <td style={comparisonCellStyle(false, true)}>{r.dimension.replace(/_/g, " ")}</td>
                {columns.map(c => (
                  <td key={c.paper_id} style={comparisonCellStyle(false)}>
                    {r.cells[c.paper_id] || r.cells[String(c.paper_id)] || "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {notes && (
        <p style={{ fontSize: "10px", color: "var(--rf-text5)", marginTop: 4 }}>{notes}</p>
      )}
    </div>
  );
}

function comparisonCellStyle(isHeader: boolean, isRowLabel = false): React.CSSProperties {
  return {
    padding: "8px 10px", borderBottom: "1px solid var(--rf-border)",
    textAlign: "left", verticalAlign: "top",
    fontSize: isHeader ? "9px" : "10.5px",
    fontWeight: isHeader || isRowLabel ? 700 : 400,
    textTransform: isHeader ? "uppercase" : "none",
    letterSpacing: isHeader ? "0.05em" : "normal",
    color: isHeader ? "var(--rf-text5)" : "var(--rf-text2)",
    background: isHeader ? "var(--rf-surface3)" : "transparent",
  };
}

function mermaidEdgeCount(code: string): number {
  return (code.match(/-->|---|\-\.\->|-\.-|==>|->>/g) || []).length;
}

function MermaidBlock({ title, code }: { title?: string; code: string }) {
  // Only render diagrams with enough structural complexity — basic 1-3 edge
  // diagrams add noise without insight. Threshold: 4+ connections.
  if (mermaidEdgeCount(code) < 4) return null;

  const fenced = "```mermaid\n" + code.trim() + "\n```";
  return (
    <div>
      {title && <BlockTitle>{title}</BlockTitle>}
      <div style={{ ...cardStyle(), padding: 0, overflowX: "auto" }}>
        <MarkdownRenderer content={fenced} />
      </div>
    </div>
  );
}

function BookmarksAnswerBlock({ title, content }: { title?: string; content: string }) {
  return (
    <div style={{ ...cardStyle(), borderLeft: "3px solid #f59e0b" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
        <BookmarkIcon size={12} color="#f59e0b" />
        <BlockTitle inline>{title || "From your bookmarks"}</BlockTitle>
      </div>
      <div style={{ fontSize: "12px", color: "var(--rf-text2)", lineHeight: 1.55 }}>
        <MarkdownRenderer content={content} />
      </div>
    </div>
  );
}

function GraphSummaryBlock({
  title, summary, href,
}: { title?: string; summary: Record<string, unknown>; href?: string }) {
  const counts = Object.entries(summary || {}).slice(0, 4);
  return (
    <div style={cardStyle()}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <NetworkIcon size={13} color="#22d3ee" />
        <BlockTitle inline>{title || "Knowledge graph"}</BlockTitle>
        {href && (
          <a href={href} style={{ marginLeft: "auto", fontSize: "10px", color: "#818cf8", textDecoration: "none" }}>
            open <ExternalLinkIcon size={10} style={{ verticalAlign: -1 }} />
          </a>
        )}
      </div>
      <div style={{ fontSize: "10.5px", color: "var(--rf-text4)", marginTop: 6 }}>
        {counts.length === 0 && "Updated."}
        {counts.map(([k, v]) => (
          <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "1.5px 0" }}>
            <span>{k}</span><span style={{ color: "var(--rf-text2)", fontWeight: 600 }}>{String(v)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ArtifactLinkBlock({
  title, href, kindLabel,
}: { title?: string; href: string; kindLabel: string }) {
  return (
    <a href={href} style={{ ...cardStyle(), display: "flex", alignItems: "center",
                            gap: 8, textDecoration: "none" }}>
      <SparklesIcon size={13} color="#a78bfa" />
      <span style={{ fontSize: "11.5px", color: "var(--rf-text1)", fontWeight: 600 }}>
        {title || kindLabel}
      </span>
      <ExternalLinkIcon size={11} color="var(--rf-text4)" style={{ marginLeft: "auto" }} />
    </a>
  );
}

function SuggestionChipsBlock({
  title, suggestions, onSubmit,
}: {
  title?: string;
  suggestions: { label: string; href?: string; kind?: string }[];
  onSubmit: (text: string) => void;
}) {
  return (
    <div>
      {title && <BlockTitle>{title}</BlockTitle>}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {suggestions.map((s, i) => {
          const isLink = !!s.href;
          const Tag = isLink ? "a" : "button";
          const handler = isLink ? undefined : () => onSubmit(s.label);
          return (
            <Tag
              key={`${s.label}-${i}`}
              {...(isLink ? { href: s.href } : { onClick: handler, type: "button" })}
              style={{
                fontSize: "10.5px", padding: "5px 10px", borderRadius: 14,
                background: "rgba(99,102,241,0.12)", color: "#818cf8",
                textDecoration: "none", border: "1px solid rgba(99,102,241,0.25)",
                cursor: "pointer", fontWeight: 500,
                display: "inline-flex", alignItems: "center", gap: 4,
              }}
            >
              {!isLink && <ChevronRightIcon size={10} />}
              {s.label}
            </Tag>
          );
        })}
      </div>
    </div>
  );
}

function ActionsTakenBlock({ actions }: { actions: string[] }) {
  return (
    <div style={{
      display: "flex", flexWrap: "wrap", gap: 6, padding: "6px 0",
      borderTop: "1px dashed var(--rf-border)",
    }}>
      <span style={{ fontSize: "9px", color: "var(--rf-text5)", fontWeight: 700,
                     textTransform: "uppercase", letterSpacing: "0.05em" }}>
        Tools used:
      </span>
      {actions.map((a, i) => (
        <span key={`${a}-${i}`} style={{ fontSize: "9.5px", color: "var(--rf-text4)" }}>{a}</span>
      ))}
    </div>
  );
}

function SourcePapersBlock({ title, papers }: { title?: string; papers: SourcePaper[] }) {
  const [open, setOpen] = useState(false);
  if (!papers.length) return null;
  return (
    <div>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: "flex", alignItems: "center", gap: 8, width: "100%",
          background: "none", border: "none", cursor: "pointer", padding: 0, marginBottom: open ? 6 : 0,
          textAlign: "left",
        }}
      >
        {title && <BlockTitle style={{ marginBottom: 0, flex: 1 }}>{title}</BlockTitle>}
        <span style={{
          fontSize: "9px", color: "var(--rf-text5)",
          padding: "2px 7px", borderRadius: 8,
          background: "var(--rf-surface3)", border: "1px solid var(--rf-border)",
        }}>
          {papers.length} {open ? "−" : "+"}
        </span>
      </button>
      {open && (
        <div style={{ display: "grid", gap: 6 }}>
          {papers.map((p, i) => {
            const href = p.url || (p.doi ? `https://doi.org/${p.doi}` : undefined);
            const sourceLabel = p.source || "Source";
            return (
              <div key={i} style={cardStyle()}>
                <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                  <FileTextIcon size={13} color="var(--rf-text4)" style={{ flexShrink: 0, marginTop: 2 }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                      <span style={{ flex: 1, fontWeight: 600, fontSize: "12px", color: "var(--rf-text1)" }}>
                        {p.title || "Untitled"}
                      </span>
                      {href && (
                        <a href={href} target="_blank" rel="noopener noreferrer"
                           style={{ color: "var(--rf-text4)", display: "inline-flex" }}>
                          <ExternalLinkIcon size={11} />
                        </a>
                      )}
                    </div>
                    <div style={{ fontSize: "10px", color: "var(--rf-text5)", marginTop: 2 }}>
                      {(p.authors || []).slice(0, 3).join(", ")}
                      {(p.authors || []).length > 3 ? ", …" : ""}
                      {p.year ? ` · ${p.year}` : ""}
                      {p.citation_count != null ? ` · ${p.citation_count} citations` : ""}
                      {` · `}
                      <span style={{ color: "var(--rf-accent)", fontWeight: 600 }}>{sourceLabel}</span>
                    </div>
                    {p.abstract && (
                      <div style={{ fontSize: "11px", color: "var(--rf-text3)", marginTop: 4, lineHeight: 1.45 }}>
                        {p.abstract.slice(0, 240)}{p.abstract.length > 240 ? "…" : ""}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function NvdResultsBlock({ title, vulnerabilities }: { title?: string; vulnerabilities: NvdVuln[] }) {
  if (!vulnerabilities.length) return null;
  const severityColor = (s?: string) => {
    const sev = (s || "").toLowerCase();
    if (sev === "critical") return "#ef4444";
    if (sev === "high") return "#f97316";
    if (sev === "medium") return "#eab308";
    return "var(--rf-text4)";
  };
  return (
    <div>
      {title && <BlockTitle>{title}</BlockTitle>}
      <div style={{ display: "grid", gap: 6 }}>
        {vulnerabilities.map((v, i) => (
          <div key={i} style={{ ...cardStyle(), borderLeft: `3px solid ${severityColor(v.severity)}` }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <span style={{ fontWeight: 700, fontSize: "11.5px", color: "var(--rf-text1)", fontFamily: "monospace" }}>
                {v.id}
              </span>
              {v.severity && (
                <span style={{
                  fontSize: "9px", padding: "1px 6px", borderRadius: 8, fontWeight: 700,
                  background: `${severityColor(v.severity)}22`,
                  color: severityColor(v.severity),
                  border: `1px solid ${severityColor(v.severity)}44`,
                  textTransform: "uppercase",
                }}>
                  {v.severity}
                </span>
              )}
              {v.cvss_score != null && (
                <span style={{ fontSize: "10px", color: "var(--rf-text4)", marginLeft: "auto" }}>
                  CVSS {v.cvss_score}
                </span>
              )}
              {v.url && (
                <a href={v.url} target="_blank" rel="noopener noreferrer"
                   style={{ color: "var(--rf-text4)" }}>
                  <ExternalLinkIcon size={10} />
                </a>
              )}
            </div>
            {v.description && (
              <div style={{ fontSize: "11px", color: "var(--rf-text3)", lineHeight: 1.45 }}>
                {v.description.slice(0, 280)}{v.description.length > 280 ? "…" : ""}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function TrialsResultsBlock({ title, studies }: { title?: string; studies: ClinicalStudy[] }) {
  if (!studies.length) return null;
  const statusColor = (s?: string) => {
    const st = (s || "").toLowerCase();
    if (st.includes("recruiting")) return "#22c55e";
    if (st.includes("complet")) return "#6366f1";
    if (st.includes("terminat") || st.includes("withdrawn")) return "#ef4444";
    return "var(--rf-text4)";
  };
  return (
    <div>
      {title && <BlockTitle>{title}</BlockTitle>}
      <div style={{ display: "grid", gap: 6 }}>
        {studies.map((s, i) => (
          <a key={i} href={s.url || "#"} target="_blank" rel="noopener noreferrer"
             style={{ ...cardStyle(), display: "block", textDecoration: "none" }}>
            <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, fontSize: "12px", color: "var(--rf-text1)" }}>
                  {s.title || s.nct_id || "Untitled"}
                </div>
                <div style={{ fontSize: "10px", color: "var(--rf-text5)", marginTop: 2 }}>
                  {s.nct_id && <span style={{ fontFamily: "monospace" }}>{s.nct_id}</span>}
                  {s.phase ? ` · Phase ${s.phase}` : ""}
                  {s.status && (
                    <span style={{ marginLeft: 6, color: statusColor(s.status), fontWeight: 600 }}>
                      {s.status}
                    </span>
                  )}
                </div>
                {(s.conditions || []).length > 0 && (
                  <div style={{ fontSize: "10.5px", color: "var(--rf-text3)", marginTop: 4 }}>
                    {(s.conditions || []).slice(0, 3).join(" · ")}
                  </div>
                )}
              </div>
              <ExternalLinkIcon size={11} color="var(--rf-text4)" style={{ flexShrink: 0 }} />
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}

function FredDataBlock({ title, series }: { title?: string; series: FredSeriesItem[] }) {
  if (!series.length) return null;
  return (
    <div>
      {title && <BlockTitle>{title}</BlockTitle>}
      <div style={{ display: "grid", gap: 6 }}>
        {series.map((s, i) => {
          const obs = (s.observations || []).slice(-6);
          return (
            <div key={i} style={cardStyle()}>
              <div style={{ fontWeight: 600, fontSize: "11.5px", color: "var(--rf-text1)" }}>
                {s.title || s.id}
                <span style={{ fontWeight: 400, fontSize: "10px", color: "var(--rf-text4)", marginLeft: 6 }}>
                  {s.units}{s.frequency ? ` · ${s.frequency}` : ""}
                </span>
              </div>
              {obs.length > 0 && (
                <div style={{ display: "flex", gap: 8, marginTop: 6, flexWrap: "wrap" }}>
                  {obs.map((o, oi) => (
                    <div key={oi} style={{ textAlign: "center", minWidth: 60 }}>
                      <div style={{ fontSize: "9px", color: "var(--rf-text5)" }}>{o.date}</div>
                      <div style={{ fontSize: "11.5px", fontWeight: 700, color: "var(--rf-text1)" }}>
                        {o.value === "." ? "N/A" : o.value}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CodeResultsBlock({ title, items }: { title?: string; items: CodeItem[] }) {
  if (!items.length) return null;
  return (
    <div>
      {title && <BlockTitle>{title}</BlockTitle>}
      <div style={{ display: "grid", gap: 6 }}>
        {items.map((item, i) => {
          const name = item.full_name || item.name || item.id || "Untitled";
          const href = item.url || (item.id ? `https://huggingface.co/${item.id}` : undefined);
          const isHF = item.source === "HuggingFace";
          return (
            <a key={i} href={href || "#"} target="_blank" rel="noopener noreferrer"
               style={{ ...cardStyle(), display: "block", textDecoration: "none" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontWeight: 600, fontSize: "12px", color: "var(--rf-text1)", flex: 1 }}>
                  {name}
                </span>
                {item.language && (
                  <span style={{
                    fontSize: "9px", padding: "1px 5px", borderRadius: 6,
                    background: "var(--rf-surface3)", color: "var(--rf-text4)",
                    border: "1px solid var(--rf-border)",
                  }}>{item.language}</span>
                )}
                <span style={{ fontSize: "9px", color: "var(--rf-accent)", fontWeight: 600 }}>
                  {item.source}
                </span>
                <ExternalLinkIcon size={10} color="var(--rf-text4)" />
              </div>
              {item.description && (
                <div style={{ fontSize: "11px", color: "var(--rf-text3)", marginTop: 4, lineHeight: 1.4 }}>
                  {item.description.slice(0, 180)}{item.description.length > 180 ? "…" : ""}
                </div>
              )}
              <div style={{ fontSize: "10px", color: "var(--rf-text5)", marginTop: 4, display: "flex", gap: 12 }}>
                {item.stars != null && <span>★ {item.stars.toLocaleString()}</span>}
                {isHF && item.downloads != null && <span>↓ {item.downloads.toLocaleString()}</span>}
                {isHF && item.likes != null && <span>♥ {item.likes.toLocaleString()}</span>}
                {(item.tags || []).slice(0, 3).map(t => (
                  <span key={t} style={{ color: "var(--rf-text4)" }}>#{t}</span>
                ))}
              </div>
            </a>
          );
        })}
      </div>
    </div>
  );
}

function BlockTitle({
  children, inline = false, style,
}: { children: React.ReactNode; inline?: boolean; style?: React.CSSProperties }) {
  return (
    <div style={{
      fontSize: "10px", fontWeight: 700, color: "var(--rf-text4)",
      textTransform: "uppercase", letterSpacing: "0.05em",
      marginBottom: inline ? 0 : 6,
      ...style,
    }}>{children}</div>
  );
}

function cardStyle(): React.CSSProperties {
  return {
    padding: "10px 12px", borderRadius: 8,
    background: "var(--rf-surface2)", border: "1px solid var(--rf-border)",
  };
}

// ─── Empty state ───────────────────────────────────────────────────────────

// Namespace-aware seed questions so the suggestions feel relevant to the
// user's actual research area rather than being hardcoded to cs.AI topics.
const _NS_SEEDS: Record<string, string[]> = {
  "cs.AI": [
    "What are the frontier directions in mechanistic interpretability right now?",
    "Compare retrieval-augmented generation vs long-context LLMs for knowledge-intensive tasks.",
    "Help me understand the landscape of efficient training for large language models.",
    "I want to start a research project on agentic AI — where do I begin?",
  ],
  "cs.LG": [
    "What are leading approaches for sample-efficient reinforcement learning?",
    "Compare transformer architectures vs state-space models for sequence modelling.",
    "Help me find papers on self-supervised learning for tabular data.",
    "What is the current state of neural scaling laws research?",
  ],
  "cs.CV": [
    "What are frontier directions in 3D scene understanding and reconstruction?",
    "Compare vision-language models: how do CLIP, ALIGN, and SigLIP differ?",
    "Help me explore diffusion models for controllable image generation.",
    "What recent work addresses long-tailed recognition in computer vision?",
  ],
  "cs.CL": [
    "What is the current state of low-resource machine translation?",
    "Compare instruction tuning vs RLHF vs DPO for aligning language models.",
    "Help me find papers on structured prediction and information extraction.",
    "What are the open problems in multilingual NLP?",
  ],
  "cs.RO": [
    "What are leading methods for robot learning from demonstration?",
    "How are foundation models being applied to embodied AI and robotics?",
    "Compare sim-to-real transfer approaches in robotic manipulation.",
    "What is the current state of autonomous vehicle perception research?",
  ],
  "q-bio": [
    "What are state-of-the-art methods for protein structure prediction beyond AlphaFold?",
    "How is deep learning being used for drug discovery and molecular design?",
    "Help me understand graph neural networks for biological interaction networks.",
    "What are open problems in computational genomics?",
  ],
  "math": [
    "What are recent breakthroughs in combinatorics and extremal graph theory?",
    "Help me understand connections between category theory and type theory.",
    "Compare different approaches to automated and interactive theorem proving.",
    "What is the landscape of research on Langlands program and arithmetic geometry?",
  ],
  "physics": [
    "What are the most promising approaches to room-temperature superconductivity?",
    "Help me understand recent experimental results in quantum computing hardware.",
    "What is the current state of dark matter detection experiments?",
    "How is machine learning being applied to high-energy physics data analysis?",
  ],
  "econ": [
    "What are recent developments in algorithmic mechanism design and auction theory?",
    "Help me understand the literature on causal inference in observational studies.",
    "Compare structural estimation vs reduced-form approaches in empirical economics.",
    "What are frontier topics in market microstructure research?",
  ],
  "stat": [
    "What are frontier directions in Bayesian deep learning?",
    "Help me understand modern approaches to causal discovery.",
    "Compare frequentist vs Bayesian approaches to high-dimensional inference.",
    "What is the current state of conformal prediction research?",
  ],
};

const _DEFAULT_SEEDS = [
  "What are the most important recent papers in this research area?",
  "Help me understand the key open problems and frontier directions here.",
  "Compare the leading methodologies and their trade-offs in this field.",
  "I want to start a research project — help me map the landscape.",
];

function _seedsForNamespace(ns: string): string[] {
  if (_NS_SEEDS[ns]) return _NS_SEEDS[ns];
  const prefix = ns.split(".")[0];
  const match = Object.entries(_NS_SEEDS).find(([k]) => k.split(".")[0] === prefix);
  return match ? match[1] : _DEFAULT_SEEDS;
}

function EmptyState({ namespaceKey, onStart }: { namespaceKey: string; onStart: (text: string) => void }) {
  const [seeds, setSeeds] = useState<string[]>(() => _seedsForNamespace(namespaceKey));
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Reset to static seeds immediately for instant feedback, then fetch dynamic seeds
    setSeeds(_seedsForNamespace(namespaceKey));
    if (!namespaceKey) return;
    setLoading(true);
    api.get<{ questions: string[] }>(`/assistant/seeds?namespace_key=${encodeURIComponent(namespaceKey)}`)
      .then(r => { if (r.questions?.length) setSeeds(r.questions); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [namespaceKey]);

  return (
    <div style={{ maxWidth: 640, margin: "60px auto", textAlign: "center" }}>
      <div style={{
        width: 56, height: 56, borderRadius: "50%", margin: "0 auto 14px",
        background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <MessageSquareIcon size={26} color="white" />
      </div>
      <h2 style={{ fontSize: "18px", fontWeight: 700, color: "var(--rf-text1)", margin: 0 }}>
        Research Assistant
      </h2>
      <p style={{ fontSize: "12px", color: "var(--rf-text4)", marginTop: 8 }}>
        A persistent workspace that orchestrates Deep Search, arXiv, Genie, web search, Wolfram Alpha,
        paper comparisons, bookmarks, and your knowledge graph — all in the background.
      </p>
      <div style={{ marginTop: 22, display: "grid", gap: 8 }}>
        {seeds.map(s => (
          <button
            key={s}
            onClick={() => onStart(s)}
            style={{
              padding: "10px 14px", borderRadius: 8, textAlign: "left",
              background: "var(--rf-surface1)", border: "1px solid var(--rf-border)",
              color: loading ? "var(--rf-text5)" : "var(--rf-text2)",
              fontSize: "11.5px", cursor: "pointer", transition: "color 0.2s",
            }}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
