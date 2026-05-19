"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  BookmarkIcon,
  BookOpenIcon,
  ExternalLinkIcon,
  XIcon,
  ClockIcon,
  MessageSquareIcon,
  SendIcon,
  BotIcon,
  UserIcon,
  Loader2Icon,
  ArrowLeftIcon,
} from "lucide-react";
import type { Paper } from "@/types";
import { api } from "@/lib/api";
import { cleanAbstract } from "@/lib/utils";
import { useJobsStore, type StudyJob } from "@/store/jobs";
import { useBookmarksStore } from "@/store/bookmarks";
import { BookmarkFolderPicker } from "@/components/bookmarks/BookmarkFolderPicker";
import { topicLabelFor } from "@/store/namespace";
import MarkdownRenderer from "@/components/ui/MarkdownRenderer";

interface Props {
  paper: Paper;
  onClose: () => void;
}

const EXPERTISE_LEVELS = [
  { key: "newcomer",     label: "Newcomer",     desc: "Jargon-free walkthrough" },
  { key: "practitioner", label: "Practitioner", desc: "Concise, field-aware" },
  { key: "expert",       label: "Expert",       desc: "Novelty-focused" },
] as const;

export function PaperPanel({ paper, onClose }: Props) {
  const router = useRouter();
  const { initialize, isBookmarked, add, remove } = useBookmarksStore();
  const bookmarked = isBookmarked(paper.id);
  const [showFolderPicker, setShowFolderPicker] = useState(false);
  const [folderIds, setFolderIds] = useState<string[]>([]);
  const bmBtnRef = useRef<HTMLButtonElement>(null);
  const [expertise, setExpertise] = useState<"newcomer" | "practitioner" | "expert">("practitioner");
  const [showAllAuthors, setShowAllAuthors] = useState(false);
  const [tldr, setTldr] = useState<string | null>(paper.tldr ?? null);
  const [queuing, setQueuing] = useState(false);
  const [queued, setQueued] = useState(false);
  const [showChat, setShowChat] = useState(false);

  // Resizable panel width — persists across mounts via localStorage. Bounds
  // chosen so the panel never disappears (<320px) or eats the whole screen.
  const [panelWidth, setPanelWidth] = useState<number>(() => {
    if (typeof window === "undefined") return 460;
    try {
      const saved = parseInt(localStorage.getItem("rf_paper_panel_w") || "", 10);
      if (Number.isFinite(saved) && saved >= 320 && saved <= 1100) return saved;
    } catch {}
    return 460;
  });
  const startResize = (e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelWidth;
    const onMove = (ev: MouseEvent) => {
      // Panel is anchored to the right edge → wider = drag left.
      const dx = startX - ev.clientX;
      const w = Math.min(1100, Math.max(320, startW + dx));
      setPanelWidth(w);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      try { localStorage.setItem("rf_paper_panel_w", String(panelWidth)); } catch {}
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  useEffect(() => {
    try { localStorage.setItem("rf_paper_panel_w", String(panelWidth)); } catch {}
  }, [panelWidth]);
  // Narrow selectors so this panel doesn't re-render on every
  // unrelated jobs-store update (e.g. media generation polls).
  const jobs = useJobsStore((s) => s.jobs);
  const fetchJobs = useJobsStore((s) => s.fetchJobs);
  const paperJob = jobs.find(
    (j: StudyJob) => j.paper_id === paper.id && j.expertise_level === expertise
  ) ?? null;

  useEffect(() => { initialize(); }, [initialize]);

  useEffect(() => {
    setShowAllAuthors(false);
    setTldr(paper.tldr ?? null);
    if (!paper.tldr) {
      api.get<{ tldr: string }>(`/papers/${paper.id}/tldr`)
        .then((r) => setTldr(r.tldr))
        .catch(() => {});
    }
  }, [paper.id, paper.tldr]);

  async function queueStudy() {
    setQueuing(true);
    try {
      await api.post(`/study/${paper.id}/queue`, { expertise_level: expertise });
      setQueued(true);
      await fetchJobs();
    } catch {}
    setQueuing(false);
  }

  // "Study now" entry: fire-and-forget the background queue so the JobsPanel
  // shows progress with the paper title, then navigate. The page itself
  // hits cached chunks if the bg parse beat us; otherwise it streams inline.
  // Either way the user can leave the page safely — the bg job continues
  // and the notification bell links them back when done.
  async function startStudyNow() {
    if (paperJob?.status !== "running" && paperJob?.status !== "pending") {
      // best-effort — never block navigation on this
      api.post(`/study/${paper.id}/queue`, { expertise_level: expertise })
        .then(() => fetchJobs())
        .catch(() => {});
    }
    router.push(`/study/${paper.id}?level=${expertise}`);
  }

  function openPicker(e: React.MouseEvent) {
    e.stopPropagation();
    setShowFolderPicker((v) => !v);
  }

  function handleSaved(ids: string[]) {
    setFolderIds(ids);
    add(paper.id);
    setShowFolderPicker(false);
  }

  function handleRemoved() {
    remove(paper.id);
    setFolderIds([]);
    setShowFolderPicker(false);
  }

  return (
    <motion.aside
      key="panel"
      initial={{ x: "100%", opacity: 0.5 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0 }}
      transition={{ type: "spring", damping: 28, stiffness: 320, mass: 0.8 }}
      className="shrink-0 border-l flex flex-col overflow-hidden relative"
      style={{
        borderColor: "var(--rf-border2)",
        background: "var(--rf-surface)",
        width: `${panelWidth}px`,
      }}
    >
      {/* Resize handle — drag the left edge to widen / narrow */}
      <div
        onMouseDown={startResize}
        className="absolute left-0 top-0 bottom-0 w-1 cursor-ew-resize hover:bg-indigo-500/30 transition-colors z-20"
        title="Drag to resize"
      />
      {/* Sticky header */}
      <div className="sticky top-0 z-10 backdrop-blur-sm border-b px-5 py-3.5 flex items-center justify-between" style={{ background: "var(--rf-surface2)", borderColor: "var(--rf-border)" }}>
        <div className="flex items-center gap-2">
          {showChat ? (
            <button
              onClick={() => setShowChat(false)}
              className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-200 transition-colors"
            >
              <ArrowLeftIcon size={13} />
              Back
            </button>
          ) : (
            <>
              {((paper.namespace_keys && paper.namespace_keys.length > 0) ? paper.namespace_keys : [paper.namespace_key]).map((nsKey) => (
                <span
                  key={nsKey}
                  title={nsKey}
                  className="text-[10px] font-semibold text-gray-500 bg-gray-800 px-2 py-0.5 rounded-md border border-gray-700/40"
                >
                  {topicLabelFor(nsKey)}
                </span>
              ))}
              {paper.is_manually_imported && (
                <span
                  title="Manually imported"
                  className="text-[10px] font-semibold px-2 py-0.5 rounded-md border inline-flex items-center gap-1"
                  style={{ background: "rgba(99,102,241,0.10)", color: "#a5b4fc", borderColor: "rgba(99,102,241,0.30)" }}
                >
                  Imported
                </span>
              )}
            </>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowChat((v) => !v)}
            title="Chat about this paper"
            className={`rounded-lg p-1.5 transition-all ${
              showChat
                ? "text-indigo-400 bg-indigo-950/40"
                : "text-gray-600 hover:text-gray-300 hover:bg-gray-800"
            }`}
          >
            <MessageSquareIcon size={15} />
          </button>
          <button
            onClick={onClose}
            className="text-gray-600 hover:text-gray-300 hover:bg-gray-800 rounded-lg p-1.5 transition-all"
          >
            <XIcon size={15} />
          </button>
        </div>
      </div>

      {/* Chat view */}
      {showChat && <PanelChat paperId={paper.id} level={expertise} />}

      {/* Scrollable content */}
      <div className={`flex-1 overflow-y-auto p-5 space-y-5 ${showChat ? "hidden" : ""}`}>
        {/* Title & authors */}
        <div>
          <h2 className="text-base font-bold text-white leading-snug">{paper.title}</h2>
          <div className="flex items-center gap-1.5 mt-1.5 min-w-0">
            <p className={`text-sm text-gray-500 leading-relaxed min-w-0 ${showAllAuthors ? "break-words" : "truncate"}`}>
              {showAllAuthors
                ? paper.authors.join(", ")
                : paper.authors.slice(0, 3).join(", ")}
            </p>
            {!showAllAuthors && paper.authors.length > 3 && (
              <button
                onClick={() => setShowAllAuthors(true)}
                className="flex-shrink-0 text-xs text-indigo-400 hover:text-indigo-300 underline underline-offset-2 whitespace-nowrap"
              >
                +{paper.authors.length - 3} more
              </button>
            )}
          </div>
          {paper.published_at && (
            <p className="text-xs text-gray-700 mt-0.5">
              Published {new Date(paper.published_at).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
            </p>
          )}
        </div>

        {/* Key concepts */}
        {paper.key_concepts.length > 0 && (
          <Section title="Key Concepts">
            <div className="flex flex-wrap gap-1.5">
              {paper.key_concepts.map((c) => (
                <span key={c} className="text-[11px] bg-teal-950/50 text-teal-300/80 border border-teal-900/40 px-2.5 py-1 rounded-full font-medium">
                  {c}
                </span>
              ))}
            </div>
          </Section>
        )}

        {/* Methods */}
        {paper.methods_used.length > 0 && (
          <Section title="Methods">
            <div className="flex flex-wrap gap-1.5">
              {paper.methods_used.map((m) => (
                <span key={m} className="text-[11px] bg-amber-950/40 text-amber-300/80 border border-amber-900/30 px-2.5 py-1 rounded-full font-medium">
                  {m}
                </span>
              ))}
            </div>
          </Section>
        )}

        {/* TLDR */}
        {tldr ? (
          <div className="bg-gray-900/80 border border-gray-700/50 rounded-xl px-4 py-3">
            <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-1">TL;DR</p>
            <p className="text-sm text-gray-200 leading-relaxed">{tldr}</p>
          </div>
        ) : (
          <div className="bg-gray-900/80 border border-gray-700/50 rounded-xl px-4 py-3 animate-pulse">
            <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-wider mb-1">TL;DR</p>
            <div className="h-4 bg-gray-800 rounded w-3/4" />
          </div>
        )}

        {/* Abstract */}
        <Section title="Abstract">
          <p className="text-sm text-gray-300 leading-[1.75] tracking-[0.01em]">
            {cleanAbstract(paper.abstract)}
          </p>
        </Section>

        {/* Why this matters — downstream impact, not a TL;DR rehash */}
        {paper.implications && (
          <div className="bg-indigo-950/20 border border-indigo-900/30 rounded-xl p-4">
            <p className="text-[10px] font-semibold text-indigo-400 uppercase tracking-wider mb-1.5">Why this matters</p>
            <p className="text-sm text-gray-300 leading-relaxed">{paper.implications}</p>
          </div>
        )}

        {/* Study CTA */}
        <div className="space-y-3 pt-1">
          <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Study depth</p>
          <div className="grid grid-cols-3 gap-2">
            {EXPERTISE_LEVELS.map(({ key, label, desc }) => (
              <button
                key={key}
                onClick={() => setExpertise(key)}
                className={`p-2.5 rounded-xl border text-left transition-all duration-150 ${
                  expertise === key
                    ? "border-indigo-500/50 bg-indigo-950/30"
                    : "border-gray-800 bg-gray-900 hover:border-gray-700"
                }`}
              >
                <p className={`text-xs font-semibold ${expertise === key ? "text-indigo-300" : "text-gray-400"}`}>
                  {label}
                </p>
                <p className="text-[10px] text-gray-600 mt-0.5 leading-tight">{desc}</p>
              </button>
            ))}
          </div>

          <div className="flex gap-2">
            <button
              onClick={startStudyNow}
              className="flex-1 py-3 flex items-center justify-center gap-2 text-sm font-semibold rounded-xl transition-all duration-200 active:scale-[0.98]"
              style={{
                background: "linear-gradient(135deg, #6366f1, #7c3aed)",
                color: "#ffffff",
                boxShadow: "0 2px 8px rgba(99,102,241,0.35)",
              }}
            >
              <BookOpenIcon size={15} />
              Study now
            </button>
            <button
              onClick={queueStudy}
              disabled={queuing || queued || paperJob?.status === "pending" || paperJob?.status === "running"}
              title="Generate study in background — check the bell icon when ready"
              className={`py-3 px-3.5 rounded-xl border text-sm font-medium flex items-center gap-1.5 transition-all duration-150 ${
                paperJob?.status === "done"
                  ? "border-emerald-800/40 bg-emerald-950/30 text-emerald-400"
                  : (queued || paperJob?.status === "pending" || paperJob?.status === "running")
                    ? "border-teal-800/40 bg-teal-950/30 text-teal-400"
                    : "btn-outline"
              }`}
            >
              <ClockIcon size={14} />
              {paperJob?.status === "done" ? "Done ✓" : paperJob?.status === "running" ? "Running…" : (queued || paperJob?.status === "pending") ? "Queued" : queuing ? "…" : "Later"}
            </button>
          </div>
        </div>

        {/* Secondary actions */}
        <div className="flex gap-2 pb-2">
          <div className="relative flex-1">
            <button
              ref={bmBtnRef}
              onClick={bookmarked ? undefined : openPicker}
              disabled={bookmarked}
              className={`flex items-center gap-2 w-full justify-center py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ${
                bookmarked
                  ? "bg-amber-950/50 text-amber-400 border border-amber-800/40 cursor-default"
                  : "btn-outline"
              }`}
            >
              <BookmarkIcon size={14} fill={bookmarked ? "currentColor" : "none"} />
              {bookmarked ? "Saved" : "Save"}
            </button>
            <AnimatePresence>
              {showFolderPicker && !bookmarked && (
                <BookmarkFolderPicker
                  paperId={paper.id}
                  isBookmarked={bookmarked}
                  currentFolderIds={folderIds}
                  anchorRef={bmBtnRef}
                  onClose={() => setShowFolderPicker(false)}
                  onSaved={handleSaved}
                  onRemoved={handleRemoved}
                />
              )}
            </AnimatePresence>
          </div>
          <a
            href={paper.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 flex-1 justify-center py-2.5 rounded-xl text-sm font-medium btn-outline"
          >
            <ExternalLinkIcon size={14} />
            arXiv
          </a>
          {paper.pdf_url && (
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 flex-1 justify-center py-2.5 rounded-xl text-sm font-medium btn-outline"
            >
              PDF
            </a>
          )}
        </div>
      </div>
    </motion.aside>
  );
}

// ── Inline chat panel ─────────────────────────────────────────────────────────

interface PanelChatMessage {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

function PanelChat({ paperId, level }: { paperId: string; level: string }) {
  const [messages, setMessages] = useState<PanelChatMessage[]>([
    {
      role: "assistant",
      content: "Ask me anything about this paper — methodology, results, how to apply it, or comparisons to other work.",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  // Abort controller ref so we can cancel an in-flight stream when the panel
  // unmounts or the user sends a new message before the previous one finishes.
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      // Cancel any in-flight stream when the component unmounts so the reader
      // never tries to update state on an unmounted component.
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);

    // Cancel any previous in-flight stream before starting a new one.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const history = messages
      .filter((m) => !m.streaming)
      .slice(-8)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", streaming: true },
    ]);

    try {
      const token = (() => {
        try {
          return JSON.parse(localStorage.getItem("rf_auth") || "{}").state?.token || "";
        } catch { return ""; }
      })();

      const resp = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/study/${paperId}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ message: text, expertise_level: level, history }),
          signal: controller.signal,
        }
      );

      if (!resp.body) throw new Error("no body");
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of decoder.decode(value, { stream: true }).split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            const p = JSON.parse(line.slice(6));
            if (p.chunk) {
              acc += p.chunk;
              setMessages((prev) => [
                ...prev.slice(0, -1),
                { role: "assistant", content: acc, streaming: true },
              ]);
            }
            if (p.done) {
              setMessages((prev) => [
                ...prev.slice(0, -1),
                { role: "assistant", content: acc },
              ]);
            }
          } catch {}
        }
      }
    } catch (err) {
      // AbortError is expected when the stream is cancelled on unmount or by a
      // new send() call — don't show an error message in that case.
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", content: "Something went wrong. Please try again." },
      ]);
    }
    setBusy(false);
    inputRef.current?.focus();
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-2 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
            <div
              className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
                msg.role === "user" ? "bg-indigo-600" : "bg-gray-800 border border-gray-700/50"
              }`}
            >
              {msg.role === "user"
                ? <UserIcon size={11} className="text-white" />
                : <BotIcon size={11} className="text-indigo-400" />
              }
            </div>
            <div
              className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-indigo-600 text-white rounded-tr-sm"
                  : "bg-gray-900 border border-gray-800/60 text-gray-200 rounded-tl-sm"
              }`}
            >
              {msg.role === "assistant" ? (
                <div className="prose-paper-chat">
                  <MarkdownRenderer content={msg.content} />
                </div>
              ) : (
                <div className="whitespace-pre-wrap">{msg.content}</div>
              )}
              {msg.streaming && (
                <span className="inline-block w-1 h-3.5 bg-indigo-400 rounded-sm animate-pulse ml-0.5 align-middle" />
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="p-3 border-t border-gray-800/60">
        <div className="flex gap-2 items-center bg-gray-900 border border-gray-800 rounded-xl px-3 py-2 focus-within:border-indigo-500/50 transition-colors">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
            placeholder="Ask about this paper…"
            className="flex-1 bg-transparent text-sm text-gray-200 placeholder-gray-600 outline-none"
            disabled={busy}
          />
          <button
            onClick={send}
            disabled={!input.trim() || busy}
            className="flex-shrink-0 w-6 h-6 rounded-lg bg-indigo-600 flex items-center justify-center disabled:opacity-40 hover:bg-indigo-500 transition-colors"
          >
            {busy
              ? <Loader2Icon size={11} className="animate-spin text-white" />
              : <SendIcon size={11} className="text-white" />
            }
          </button>
        </div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-wider mb-2">{title}</p>
      {children}
    </div>
  );
}

