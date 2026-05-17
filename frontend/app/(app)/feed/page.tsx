"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { api } from "@/lib/api";
import type { FeedResponse, FeedItem, Paper } from "@/types";
import { PaperCard } from "@/components/feed/PaperCard";
import { PaperPanel } from "@/components/paper/PaperPanel";
import { SearchBar, MatchTypeBadge } from "@/components/feed/SearchBar";
import type { SearchResult, DeepSearchMeta } from "@/components/feed/SearchBar";
import {
  Loader2Icon, ExternalLinkIcon, SparklesIcon, BookmarkIcon,
  RefreshCwIcon, LinkIcon, ZapIcon, ChevronDownIcon, ChevronRightIcon,
  PlusIcon, XIcon, CheckCircleIcon, EyeOffIcon,
} from "lucide-react";
import { cleanAbstract } from "@/lib/utils";
import { useBookmarksStore } from "@/store/bookmarks";
import { useNamespaceStore, NAMESPACE_TREE } from "@/store/namespace";
import { useJobsStore } from "@/store/jobs";

// ─── Types ────────────────────────────────────────────────────────────────────

interface SuggestedResult {
  paper_id: string; title: string; abstract: string | null; tldr: string | null;
  authors: string[]; namespace_key: string; source_url: string;
  novelty_score: number; relevance_score: number;
}
interface SuggestedResponse { suggestions: SuggestedResult[]; based_on: string[] }

function searchResultToPaper(r: SearchResult): Paper {
  return {
    id: r.paper_id, external_id: r.paper_id, namespace_key: r.namespace_key,
    title: r.title, authors: r.authors, abstract: r.abstract, source_url: r.source_url,
    pdf_url: r.pdf_url, published_at: r.published_at,
    key_concepts: r.key_concepts ?? [], methods_used: r.methods_used ?? [],
    implications: r.implications ?? null, novelty_score: r.novelty_score,
    relevance_score: r.relevance_score, is_breakthrough: r.is_breakthrough,
    tldr: r.tldr ?? null, ingested_at: r.ingested_at ?? "",
  };
}

// (namespace selection is in the global sidebar)

// ─── Related Papers Panel ─────────────────────────────────────────────────────

function RelatedPapersPanel({ paper, onSelectPaper }: { paper: Paper; onSelectPaper: (p: Paper) => void }) {
  const [related, setRelated] = useState<SuggestedResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);
  const { add, remove, isBookmarked, initialize } = useBookmarksStore();
  const { selectedTopics } = useNamespaceStore();
  const prevKey = useRef<string>("");

  useEffect(() => { initialize(); }, [initialize]);

  useEffect(() => {
    const key = `${paper.id}::${selectedTopics.join(",")}`;
    if (prevKey.current === key) return;
    prevKey.current = key;
    setLoading(true);
    setRelated([]);
    const nsParam = selectedTopics.length > 0 ? `&namespace_keys=${encodeURIComponent(selectedTopics.join(","))}` : "";
    api.get<SuggestedResult[]>(`/feed/papers/${paper.id}/related?limit=5${nsParam}`)
      .then(setRelated)
      .catch(() => setRelated([]))
      .finally(() => setLoading(false));
  }, [paper.id, selectedTopics]);

  async function toggleBm(e: React.MouseEvent, r: SuggestedResult) {
    e.stopPropagation();
    if (isBookmarked(r.paper_id)) { await api.delete(`/bookmarks/${r.paper_id}`); remove(r.paper_id); }
    else { await api.post("/bookmarks", { paper_id: r.paper_id }); add(r.paper_id); }
  }

  const toPaper = (r: SuggestedResult): Paper => ({
    id: r.paper_id, external_id: r.paper_id, namespace_key: r.namespace_key,
    title: r.title, authors: r.authors, abstract: r.abstract ?? "", source_url: r.source_url,
    pdf_url: null, published_at: null, key_concepts: [], methods_used: [],
    implications: null, novelty_score: r.novelty_score, relevance_score: r.relevance_score,
    is_breakthrough: false, tldr: r.tldr, ingested_at: "",
  });

  return (
    <div style={{
      width: collapsed ? 36 : 276, flexShrink: 0,
      borderLeft: "1px solid rgba(255,255,255,0.05)",
      display: "flex", flexDirection: "column", background: "rgba(6,9,18,0.6)",
      transition: "width 0.2s ease", overflow: "hidden",
    }}>
      <div style={{
        padding: collapsed ? "11px 8px" : "11px 14px",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        display: "flex", alignItems: "center",
        justifyContent: collapsed ? "center" : "space-between",
        flexShrink: 0,
      }}>
        {collapsed ? (
          <button
            onClick={() => setCollapsed(false)}
            title="Show related papers"
            style={{ background: "none", border: "none", cursor: "pointer", color: "#6366f1", display: "flex" }}
          >
            <LinkIcon size={14} />
          </button>
        ) : (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <LinkIcon size={12} color="#6366f1" />
              <span style={{ fontSize: "11px", fontWeight: 700, color: "#e5e7eb" }}>Related Papers</span>
              {loading && <Loader2Icon size={10} color="#6366f1" className="animate-spin" />}
            </div>
            <button
              onClick={() => setCollapsed(true)}
              title="Collapse"
              style={{ background: "none", border: "none", cursor: "pointer", color: "#4b5563", display: "flex" }}
            >
              <ChevronRightIcon size={12} />
            </button>
          </>
        )}
      </div>

      {!collapsed && (
        <div style={{ flex: 1, overflowY: "auto", padding: 8, display: "flex", flexDirection: "column", gap: 6 }}>
          {loading ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {[1, 2, 3].map(i => (
                <div key={i} style={{ borderRadius: 10, border: "1px solid rgba(55,65,81,0.3)", background: "rgba(17,24,39,0.4)", padding: "9px 10px" }}>
                  <div style={{ height: 10, background: "rgba(55,65,81,0.5)", borderRadius: 4, marginBottom: 6, width: "85%", animation: "pulse 1.5s infinite" }} />
                  <div style={{ height: 8, background: "rgba(55,65,81,0.4)", borderRadius: 4, width: "60%", animation: "pulse 1.5s infinite" }} />
                </div>
              ))}
            </div>
          ) : related.length === 0 ? (
            <div style={{ textAlign: "center", padding: "24px 12px", color: "#374151" }}>
              <ZapIcon size={20} style={{ margin: "0 auto 8px" }} />
              <p style={{ fontSize: "11px" }}>No related papers yet — embeddings improve over time.</p>
            </div>
          ) : related.map(r => {
            const bm = isBookmarked(r.paper_id);
            return (
              <div
                key={r.paper_id}
                onClick={() => onSelectPaper(toPaper(r))}
                style={{
                  borderRadius: 10, border: "1px solid rgba(55,65,81,0.5)",
                  background: "rgba(17,24,39,0.6)", padding: "9px 10px",
                  cursor: "pointer", transition: "border-color 0.15s",
                }}
                onMouseEnter={e => (e.currentTarget.style.borderColor = "rgba(99,102,241,0.4)")}
                onMouseLeave={e => (e.currentTarget.style.borderColor = "rgba(55,65,81,0.5)")}
              >
                <p style={{ fontSize: "10.5px", fontWeight: 600, color: "#e5e7eb", lineHeight: 1.35, marginBottom: 4, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" as const, overflow: "hidden" }}>
                  {r.title}
                </p>
                <p style={{ fontSize: "9px", color: "#6b7280", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" as const, overflow: "hidden", marginBottom: 5 }}>
                  {r.tldr ?? cleanAbstract(r.abstract ?? "")}
                </p>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontSize: "8px", color: "#374151", fontFamily: "monospace" }}>{r.namespace_key}</span>
                  <div style={{ display: "flex", alignItems: "center", gap: 4 }} onClick={e => e.stopPropagation()}>
                    <button onClick={e => toggleBm(e, r)} style={{ background: "none", border: "none", cursor: "pointer", color: bm ? "#f59e0b" : "#4b5563", display: "flex" }}>
                      <BookmarkIcon size={10} fill={bm ? "currentColor" : "none"} />
                    </button>
                    <a href={r.source_url} target="_blank" rel="noopener noreferrer" style={{ color: "#4b5563" }}>
                      <ExternalLinkIcon size={9} />
                    </a>
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

// ─── Suggested Panel ──────────────────────────────────────────────────────────

function SuggestedPanel({ onSelectPaper }: { onSelectPaper: (p: Paper) => void }) {
  const [data, setData] = useState<SuggestedResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);
  const { add, remove, isBookmarked, initialize } = useBookmarksStore();
  const { selectedTopics } = useNamespaceStore();

  useEffect(() => { initialize(); }, [initialize]);

  const load = useCallback(async () => {
    setLoading(true);
    const nsParam = selectedTopics.length > 0 ? `?namespace_keys=${encodeURIComponent(selectedTopics.join(","))}` : "";
    try { setData(await api.get<SuggestedResponse>(`/feed/suggested${nsParam}`)); } catch {}
    setLoading(false);
  }, [selectedTopics]);

  useEffect(() => { load(); }, [load]);

  async function toggleBm(e: React.MouseEvent, r: SuggestedResult) {
    e.stopPropagation();
    if (isBookmarked(r.paper_id)) { await api.delete(`/bookmarks/${r.paper_id}`); remove(r.paper_id); }
    else { await api.post("/bookmarks", { paper_id: r.paper_id }); add(r.paper_id); }
  }

  const toPaper = (r: SuggestedResult): Paper => ({
    id: r.paper_id, external_id: r.paper_id, namespace_key: r.namespace_key,
    title: r.title, authors: r.authors, abstract: r.abstract ?? "", source_url: r.source_url,
    pdf_url: null, published_at: null, key_concepts: [], methods_used: [],
    implications: null, novelty_score: r.novelty_score, relevance_score: r.relevance_score,
    is_breakthrough: false, tldr: r.tldr, ingested_at: "",
  });

  return (
    <div style={{
      width: collapsed ? 36 : 276, flexShrink: 0,
      borderLeft: "1px solid rgba(255,255,255,0.05)",
      display: "flex", flexDirection: collapsed ? "column" : "column",
      background: "rgba(6,9,18,0.5)", transition: "width 0.2s ease",
      overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: collapsed ? "11px 8px" : "11px 14px",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        display: "flex", alignItems: "center",
        justifyContent: collapsed ? "center" : "space-between",
        flexShrink: 0,
      }}>
        {collapsed ? (
          <button
            onClick={() => setCollapsed(false)}
            title="Show suggestions"
            style={{ background: "none", border: "none", cursor: "pointer", color: "#6366f1", display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}
          >
            <SparklesIcon size={14} />
          </button>
        ) : (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <SparklesIcon size={12} color="#6366f1" />
              <span style={{ fontSize: "11px", fontWeight: 700, color: "#e5e7eb" }}>Suggested for You</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <button onClick={load} disabled={loading} title="Refresh suggestions" style={{ background: "none", border: "none", cursor: "pointer", color: "#4b5563", display: "flex" }}>
                <RefreshCwIcon size={11} className={loading ? "animate-spin" : ""} />
              </button>
              <button onClick={() => setCollapsed(true)} title="Collapse" style={{ background: "none", border: "none", cursor: "pointer", color: "#4b5563", display: "flex" }}>
                <ChevronRightIcon size={12} />
              </button>
            </div>
          </>
        )}
      </div>

      {!collapsed && (
        <>
          {data?.based_on && data.based_on.length > 0 && (
            <div style={{ padding: "6px 10px", borderBottom: "1px solid rgba(55,65,81,0.2)", display: "flex", flexWrap: "wrap", gap: 3, flexShrink: 0 }}>
              <span style={{ fontSize: "8px", color: "#4b5563", marginRight: 2 }}>based on:</span>
              {data.based_on.slice(0, 4).map(c => (
                <span key={c} style={{ fontSize: "8px", background: "rgba(99,102,241,0.1)", color: "rgba(165,180,252,0.8)", border: "1px solid rgba(99,102,241,0.2)", padding: "1px 5px", borderRadius: 10 }}>{c}</span>
              ))}
            </div>
          )}

          <div style={{ flex: 1, overflowY: "auto", padding: 8, display: "flex", flexDirection: "column", gap: 6 }}>
            {loading ? (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 120, color: "#4b5563", gap: 8 }}>
                <Loader2Icon size={14} className="animate-spin" />
                <span style={{ fontSize: "11px" }}>Finding suggestions…</span>
              </div>
            ) : !data || data.suggestions.length === 0 ? (
              <div style={{ textAlign: "center", padding: "28px 12px", color: "#374151" }}>
                <SparklesIcon size={22} style={{ margin: "0 auto 10px", color: "#374151" }} />
                <p style={{ fontSize: "11px", lineHeight: 1.5 }}>Bookmark papers to get personalized suggestions based on your reading interests.</p>
              </div>
            ) : data.suggestions.map(r => {
              const bm = isBookmarked(r.paper_id);
              return (
                <div
                  key={r.paper_id}
                  onClick={() => onSelectPaper(toPaper(r))}
                  style={{
                    borderRadius: 10, border: "1px solid rgba(55,65,81,0.5)",
                    background: "rgba(17,24,39,0.6)", padding: "9px 10px", cursor: "pointer",
                    transition: "border-color 0.15s",
                  }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = "rgba(99,102,241,0.4)")}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = "rgba(55,65,81,0.5)")}
                >
                  <p style={{ fontSize: "10.5px", fontWeight: 600, color: "#e5e7eb", lineHeight: 1.35, marginBottom: 5, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" as const, overflow: "hidden" }}>
                    {r.title}
                  </p>
                  <p style={{ fontSize: "9px", color: "#6b7280", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" as const, overflow: "hidden", marginBottom: 6 }}>
                    {r.tldr ?? cleanAbstract(r.abstract ?? "")}
                  </p>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span style={{ fontSize: "8px", color: "#374151", fontFamily: "monospace" }}>{r.namespace_key}</span>
                    <div style={{ display: "flex", alignItems: "center", gap: 4 }} onClick={e => e.stopPropagation()}>
                      <button onClick={e => toggleBm(e, r)} style={{ background: "none", border: "none", cursor: "pointer", color: bm ? "#f59e0b" : "#4b5563", display: "flex" }}>
                        <BookmarkIcon size={10} fill={bm ? "currentColor" : "none"} />
                      </button>
                      <a href={r.source_url} target="_blank" rel="noopener noreferrer" style={{ color: "#4b5563" }}>
                        <ExternalLinkIcon size={9} />
                      </a>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

// ─── Ghost loading cards ──────────────────────────────────────────────────────

function FeedGhostCards({ label }: { label?: string }) {
  return (
    <div className="rf-fade-in" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {label && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, color: "#4b5563", fontSize: "12px" }}>
          <Loader2Icon className="animate-spin" size={13} style={{ color: "#818cf8" }} />
          {label}
        </div>
      )}
      {Array.from({ length: 5 }).map((_, i) => (
        <div
          key={i}
          className="rf-ghost-card rf-slide-up"
          style={{ animationDelay: `${i * 60}ms` }}
        >
          {/* Title */}
          <div className="rf-ghost" style={{ height: 14, width: `${72 + (i % 3) * 8}%`, marginBottom: 10 }} />
          {/* Abstract lines */}
          <div className="rf-ghost" style={{ height: 10, width: "95%", marginBottom: 5 }} />
          <div className="rf-ghost" style={{ height: 10, width: "80%", marginBottom: 10 }} />
          {/* Footer row */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div className="rf-ghost" style={{ height: 8, width: 80 }} />
            <div className="rf-ghost" style={{ height: 8, width: 50 }} />
            <div style={{ flex: 1 }} />
            <div className="rf-ghost" style={{ height: 8, width: 40, borderRadius: 8 }} />
            <div className="rf-ghost" style={{ height: 18, width: 18, borderRadius: 4 }} />
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Main Feed Page ───────────────────────────────────────────────────────────

export default function FeedPage() {
  const { activeSubject, selectedTopics, getPrimaryNamespaceKey } = useNamespaceStore();
  const activeNs = getPrimaryNamespaceKey();
  const subject = NAMESPACE_TREE.find(s => s.key === activeSubject);
  const { isBookmarked, initialize: initBm } = useBookmarksStore();
  useEffect(() => { initBm(); }, [initBm]);

  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshMsg, setRefreshMsg] = useState<string | null>(null);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [deepMeta, setDeepMeta] = useState<DeepSearchMeta | null>(null);
  const isSearching = searchResults !== null;
  const isDeepSearch = isSearching && deepMeta !== null;

  // arXiv manual import
  const [importOpen, setImportOpen] = useState(false);
  const [importId, setImportId] = useState("");
  const [importNs, setImportNs] = useState<string[]>([activeNs || "cs.AI"]);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [importMsg, setImportMsg] = useState<string | null>(null);
  // Synchronous ref guard — prevents double-submit from rapid Enter/click
  const importingRef = useRef(false);
  // Track job IDs spawned by this page instance so we refresh the feed exactly once on completion
  const pendingImportJobIds = useRef<Set<string>>(new Set());

  const { addArxivImportJob, arxivImportJobs } = useJobsStore();

  async function handleImport() {
    if (importingRef.current) return;
    const trimmed = importId.trim();
    if (!trimmed) return;
    const namespaces = importNs.length > 0 ? importNs : [activeNs || "cs.AI"];
    importingRef.current = true;
    setImporting(true);
    setImportError(null);
    setImportMsg(null);
    try {
      const res = await api.post<{ jobs: Array<{ job_id: string; namespace_key: string }>; arxiv_id: string; title: string; message: string }>(
        "/papers/import-arxiv",
        { arxiv_id: trimmed, namespace_keys: namespaces },
      );
      setImportMsg(res.message);
      setImportId("");
      const now = new Date().toISOString();
      for (const j of res.jobs) {
        pendingImportJobIds.current.add(j.job_id);
        addArxivImportJob({
          job_id: j.job_id,
          arxiv_id: res.arxiv_id,
          title: res.title,
          namespace_key: j.namespace_key,
          status: "running",
          summary: `Importing into ${j.namespace_key}`,
          created_at: now,
          completed_at: null,
        });
      }
    } catch (err: unknown) {
      const msg = err instanceof Error
        ? err.message
        : (err as { detail?: string })?.detail || "Import failed — check the arXiv ID and try again.";
      setImportError(msg);
    } finally {
      importingRef.current = false;
      setImporting(false);
    }
  }

  // Per-namespace hidden paper IDs
  const [hiddenIds, setHiddenIds] = useState<Set<string>>(new Set());
  const [showHiddenSection, setShowHiddenSection] = useState(false);

  const loadHiddenIds = useCallback(async () => {
    if (!selectedTopics.length) { setHiddenIds(new Set()); return; }
    try {
      const results = await Promise.allSettled(
        selectedTopics.map(ns =>
          api.get<string[]>(`/papers/hidden-ids?namespace_key=${encodeURIComponent(ns)}`)
        )
      );
      const all = new Set<string>();
      for (const r of results) {
        if (r.status === "fulfilled") r.value.forEach(id => all.add(id));
      }
      setHiddenIds(all);
    } catch {
      setHiddenIds(new Set());
    }
  }, [selectedTopics]);

  useEffect(() => { loadHiddenIds(); }, [loadHiddenIds]);

  async function toggleHide(paperId: string, paperNs: string) {
    const isCurrentlyHidden = hiddenIds.has(paperId);
    // Optimistic update
    setHiddenIds(prev => {
      const next = new Set(prev);
      if (isCurrentlyHidden) next.delete(paperId);
      else next.add(paperId);
      return next;
    });
    try {
      const nsParam = `?namespace_key=${encodeURIComponent(paperNs)}`;
      if (isCurrentlyHidden) {
        await api.delete(`/papers/${paperId}/hide${nsParam}`);
      } else {
        await api.post(`/papers/${paperId}/hide${nsParam}`);
      }
    } catch {
      // Revert on failure
      setHiddenIds(prev => {
        const next = new Set(prev);
        if (isCurrentlyHidden) next.add(paperId);
        else next.delete(paperId);
        return next;
      });
    }
  }

  // Sequence counter — discard responses from stale concurrent loadFeed calls
  const loadFeedSeqRef = useRef(0);

  const loadFeed = useCallback(async () => {
    const seq = ++loadFeedSeqRef.current;
    setLoading(true);
    try {
      const results = await Promise.allSettled(
        selectedTopics.map(ns => api.get<FeedResponse>(`/feed?namespace_key=${ns}&limit=20`))
      );
      // Discard if a newer load has already started
      if (seq !== loadFeedSeqRef.current) return;
      const allPapers: FeedItem[] = [];
      const seen = new Set<string>();
      for (const r of results) {
        if (r.status === "fulfilled") {
          for (const item of r.value.papers) {
            if (!seen.has(item.paper.id)) { seen.add(item.paper.id); allPapers.push(item); }
          }
        }
      }
      allPapers.sort((a, b) => b.score - a.score);
      setFeed(allPapers.slice(0, 60));
    } catch (e) {
      if (seq === loadFeedSeqRef.current) console.error("loadFeed error:", e);
    } finally {
      if (seq === loadFeedSeqRef.current) setLoading(false);
    }
  }, [selectedTopics]);

  // When any tracked import job completes, refresh the feed once
  useEffect(() => {
    for (const job of arxivImportJobs) {
      if (pendingImportJobIds.current.has(job.job_id) && (job.status === "completed" || job.status === "failed")) {
        pendingImportJobIds.current.delete(job.job_id);
        if (job.status === "completed") {
          loadFeed().catch(() => {});
        }
      }
    }
  }, [arxivImportJobs, loadFeed]);

  // Ref to track the active refresh poll so it can be cancelled on unmount.
  const refreshPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (refreshPollRef.current) clearInterval(refreshPollRef.current);
    };
  }, []);

  async function handleRefresh() {
    if (refreshing) return;
    setRefreshing(true);
    setRefreshMsg("Fetching from arXiv…");
    let triggered = 0;
    for (const ns of selectedTopics) {
      try {
        await api.post(`/feed/refresh?namespace_key=${encodeURIComponent(ns)}`);
        triggered++;
      } catch { /* namespace may not be mapped yet — skip silently */ }
    }
    if (triggered === 0) {
      setRefreshMsg("Nothing to refresh (unknown namespace)");
      setRefreshing(false);
    } else {
      setRefreshMsg(`Ingesting ${triggered} namespace${triggered > 1 ? "s" : ""}…`);
      // Poll every 10s for up to 2 minutes until papers appear, then generate TLDRs.
      // Ingestion is fully async on the backend and typically takes 30–90s.
      let attempts = 0;
      const maxAttempts = 12; // 2 minutes total
      const poll = setInterval(async () => {
        attempts++;
        await loadFeed();
        // Check if we now have papers by re-reading feed state after load
        const hasPapers = await (async () => {
          try {
            const r = await api.get<FeedResponse>(`/feed?namespace_key=${encodeURIComponent(selectedTopics[0] ?? "cs.AI")}&limit=1`);
            return r.papers.length > 0;
          } catch { return false; }
        })();
        if (hasPapers || attempts >= maxAttempts) {
          clearInterval(poll);
          refreshPollRef.current = null;
          if (hasPapers) {
            setRefreshMsg("Generating summaries…");
            api.post("/papers/generate-tldrs?limit=50")
              .then(() => { loadFeed(); setRefreshMsg(null); })
              .catch(() => { setRefreshMsg(null); });
          } else {
            setRefreshMsg(null);
          }
          setRefreshing(false);
        }
      }, 10000);
      refreshPollRef.current = poll;
    }
  }

  useEffect(() => { if (!isSearching) loadFeed(); }, [loadFeed, isSearching]);
  useEffect(() => { setSelectedPaper(null); setSearchResults(null); }, [activeNs]);

  async function handleFeedback(paperId: string, signal: string) {
    try { await api.post("/feed/feedback", { paper_id: paperId, signal }); } catch {}
  }

  const handleSearchResults = useCallback((
    results: SearchResult[] | null,
    query: string,
    meta?: DeepSearchMeta,
  ) => {
    setSearchResults(results);
    setSearchQuery(query);
    setSelectedPaper(null);
    setDeepMeta(meta ?? null);
  }, []);

  const handleSearchClear = useCallback(() => {
    setSearchResults(null);
    setSearchQuery("");
    setDeepMeta(null);
  }, []);

  const filtered = isSearching ? searchResults! : feed;

  // Split feed into visible and hidden papers (only when not searching)
  const visibleFeed = !isSearching ? (filtered as FeedItem[]).filter(item => !hiddenIds.has(item.paper.id)) : [];
  const hiddenFeed = !isSearching ? (filtered as FeedItem[]).filter(item => hiddenIds.has(item.paper.id)) : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "var(--rf-bg)" }}>
      {/* ── Header ── */}
      <div style={{
        padding: "11px 20px 10px",
        borderBottom: "1px solid var(--rf-border)",
        background: "var(--rf-surface2)", backdropFilter: "blur(12px)", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 9 }}>
          <h1 style={{ fontSize: "18px", fontWeight: 700, color: "var(--rf-text1)" }}>Research Feed</h1>
          <div style={{ display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap", flex: 1 }}>
            {subject && (
              <span style={{ fontSize: "9px", color: subject.color, fontWeight: 700, background: `${subject.color}15`, border: `1px solid ${subject.color}30`, padding: "1px 7px", borderRadius: 10 }}>
                {subject.icon} {subject.label}
              </span>
            )}
            {selectedTopics.map(ns => (
              <span key={ns} style={{ fontSize: "8px", color: "#6b7280", background: "rgba(55,65,81,0.3)", border: "1px solid rgba(55,65,81,0.4)", padding: "1px 6px", borderRadius: 8, fontFamily: "monospace" }}>
                {ns}
              </span>
            ))}
          </div>
          {/* Action buttons */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            {refreshMsg && (
              <span style={{ fontSize: "10px", color: "#6b7280" }}>{refreshMsg}</span>
            )}
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              title="Fetch latest papers from arXiv (idempotent)"
              style={{
                display: "flex", alignItems: "center", gap: 5,
                padding: "5px 10px", borderRadius: 8, border: "1px solid rgba(99,102,241,0.3)",
                background: "rgba(99,102,241,0.08)", color: "#818cf8",
                fontSize: "11px", fontWeight: 500, cursor: refreshing ? "not-allowed" : "pointer",
                opacity: refreshing ? 0.6 : 1, transition: "all 0.15s",
              }}
              onMouseEnter={e => { if (!refreshing) { (e.currentTarget as HTMLButtonElement).style.background = "rgba(99,102,241,0.16)"; } }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(99,102,241,0.08)"; }}
            >
              <RefreshCwIcon size={12} className={refreshing ? "animate-spin" : ""} />
              {refreshing ? "Fetching…" : "Refresh"}
            </button>
            <button
              onClick={() => { setImportOpen(true); setImportError(null); setImportMsg(null); setImportNs(activeNs ? [activeNs] : ["cs.AI"]); }}
              title="Import paper by arXiv ID"
              style={{
                display: "flex", alignItems: "center", gap: 5,
                padding: "5px 10px", borderRadius: 8, border: "1px solid rgba(16,185,129,0.3)",
                background: "rgba(16,185,129,0.08)", color: "#34d399",
                fontSize: "11px", fontWeight: 500, cursor: "pointer", transition: "all 0.15s",
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(16,185,129,0.16)"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(16,185,129,0.08)"; }}
            >
              <PlusIcon size={12} />
              Import
            </button>
          </div>
        </div>
        <SearchBar
          namespace_keys={selectedTopics.length > 0 ? selectedTopics : undefined}
          onResults={handleSearchResults}
          onClear={handleSearchClear}
        />
      </div>

      {/* ── Content ── */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Feed list */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", background: "var(--rf-bg)" }}>
          <div style={{ flex: 1, overflowY: "auto", padding: "14px 18px" }}>
            {isSearching && (
              <div style={{ marginBottom: 12, display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "12px", color: "#6b7280", flexWrap: "wrap" }}>
                  {isDeepSearch && (
                    <span style={{
                      display: "inline-flex", alignItems: "center", gap: 4,
                      fontSize: "9px", fontWeight: 700, color: "#a78bfa",
                      background: "rgba(139,92,246,0.12)", border: "1px solid rgba(139,92,246,0.25)",
                      padding: "1px 7px", borderRadius: 10, textTransform: "uppercase", letterSpacing: "0.05em",
                    }}>
                      ✦ Deep Search
                      {deepMeta?.cached && (
                        <span style={{ color: "#6b7280", fontWeight: 400, fontSize: "8px", marginLeft: 2 }}>· cached</span>
                      )}
                    </span>
                  )}
                  <span>{searchResults!.length} result{searchResults!.length !== 1 ? "s" : ""} for</span>
                  <span style={{ color: "#e5e7eb", fontWeight: 600 }}>&ldquo;{searchQuery}&rdquo;</span>
                </div>
                {isDeepSearch && deepMeta?.rewrittenQuery && deepMeta.rewrittenQuery !== searchQuery && (
                  <div style={{ fontSize: "10px", color: "#6b7280", display: "flex", alignItems: "center", gap: 5 }}>
                    <span>Searched as:</span>
                    <span style={{ color: "#9ca3af", fontStyle: "italic" }}>&ldquo;{deepMeta.rewrittenQuery}&rdquo;</span>
                  </div>
                )}
              </div>
            )}

            {loading && !isSearching ? (
              <FeedGhostCards />
            ) : refreshing && !isSearching && filtered.length === 0 ? (
              <FeedGhostCards label="Fetching from arXiv…" />
            ) : isSearching ? (
              searchResults!.length === 0 ? (
                <div className="rf-fade-in" style={{ textAlign: "center", padding: "60px 20px", color: "#4b5563" }}>
                  <p style={{ fontSize: "15px", marginBottom: 6 }}>No results found</p>
                  <p style={{ fontSize: "12px" }}>Try different keywords or switch namespace.</p>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {searchResults!.filter(r => !hiddenIds.has(r.paper_id)).map((r, i) => (
                    <div key={r.paper_id} className="rf-slide-up" style={{ animationDelay: `${i * 30}ms` }}>
                      <SearchResultCard
                        result={r}
                        isSelected={selectedPaper?.id === r.paper_id}
                        onClick={() => setSelectedPaper(selectedPaper?.id === r.paper_id ? null : searchResultToPaper(r))}
                      />
                    </div>
                  ))}
                </div>
              )
            ) : filtered.length === 0 ? (
              <div className="rf-fade-in" style={{ textAlign: "center", padding: "60px 20px", color: "#4b5563" }}>
                <p style={{ fontSize: "15px", marginBottom: 6 }}>No papers yet</p>
                <p style={{ fontSize: "12px", marginBottom: 16 }}>Hit the Refresh button above to fetch the latest from arXiv.</p>
                <button
                  onClick={handleRefresh}
                  disabled={refreshing}
                  className="rf-btn"
                  style={{
                    padding: "8px 18px", borderRadius: 10, border: "1px solid rgba(99,102,241,0.4)",
                    background: "rgba(99,102,241,0.12)", color: "#818cf8",
                    fontSize: "12px", fontWeight: 600, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: 6,
                    transition: "all 0.15s",
                  }}
                >
                  <RefreshCwIcon size={13} className={refreshing ? "animate-spin" : ""} />
                  Fetch Papers Now
                </button>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {/* Visible papers */}
                {visibleFeed.map(item => (
                  <PaperCard
                    key={item.paper.id}
                    item={item}
                    isSelected={selectedPaper?.id === item.paper.id}
                    onClick={() => setSelectedPaper(selectedPaper?.id === item.paper.id ? null : item.paper)}
                    onFeedback={handleFeedback}
                    isHidden={false}
                    onHide={item.paper.is_manually_imported
                      ? () => toggleHide(item.paper.id, item.paper.namespace_key)
                      : undefined}
                  />
                ))}

                {/* Hidden papers section */}
                {hiddenFeed.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <button
                      onClick={() => setShowHiddenSection(v => !v)}
                      style={{
                        width: "100%", display: "flex", alignItems: "center", gap: 8,
                        padding: "8px 12px", borderRadius: 8,
                        border: "1px solid var(--rf-border)", background: "transparent",
                        color: "var(--rf-text5)", fontSize: "11px", fontWeight: 500,
                        cursor: "pointer", transition: "all 0.15s",
                      }}
                    >
                      <EyeOffIcon size={12} />
                      {showHiddenSection ? "Hide" : "Show"} hidden papers ({hiddenFeed.length})
                      <ChevronDownIcon
                        size={12}
                        style={{
                          marginLeft: "auto",
                          transform: showHiddenSection ? "rotate(180deg)" : "none",
                          transition: "transform 0.2s ease",
                        }}
                      />
                    </button>
                    {showHiddenSection && (
                      <div className="rf-slide-down" style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 10 }}>
                        {hiddenFeed.map(item => (
                          <PaperCard
                            key={item.paper.id}
                            item={item}
                            isSelected={selectedPaper?.id === item.paper.id}
                            onClick={() => setSelectedPaper(selectedPaper?.id === item.paper.id ? null : item.paper)}
                            onFeedback={handleFeedback}
                            isHidden={true}
                            onHide={() => toggleHide(item.paper.id, item.paper.namespace_key)}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right panel */}
        {selectedPaper && (
          <div className="rf-slide-in-right" style={{ display: "flex", height: "100%", overflow: "hidden" }}>
            <PaperPanel paper={selectedPaper} onClose={() => setSelectedPaper(null)} />
            {isBookmarked(selectedPaper.id) && (
              <RelatedPapersPanel paper={selectedPaper} onSelectPaper={p => setSelectedPaper(p)} />
            )}
          </div>
        )}
      </div>

      {/* ── Import arXiv Modal ── */}
      {importOpen && (
        <div
          className="rf-overlay-enter"
          onClick={() => { if (!importing) setImportOpen(false); }}
          style={{
            position: "fixed", inset: 0, zIndex: 100,
            background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          <div
            className="rf-modal-enter"
            onClick={e => e.stopPropagation()}
            style={{
              width: 420, borderRadius: 16,
              border: "1px solid rgba(16,185,129,0.25)",
              background: "rgba(6,12,22,0.97)",
              boxShadow: "0 24px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04)",
              padding: "24px 28px",
            }}
          >
            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ width: 32, height: 32, borderRadius: 8, background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <PlusIcon size={16} color="#34d399" />
                </div>
                <div>
                  <h3 style={{ fontSize: "14px", fontWeight: 700, color: "#f9fafb", margin: 0 }}>Import arXiv Paper</h3>
                  <p style={{ fontSize: "10px", color: "#6b7280", margin: 0 }}>Full ingestion — enriched, embedded, indexed</p>
                </div>
              </div>
              <button
                onClick={() => setImportOpen(false)}
                style={{ background: "none", border: "none", cursor: "pointer", color: "#4b5563", padding: 4, borderRadius: 6, display: "flex" }}
              >
                <XIcon size={16} />
              </button>
            </div>

            {/* arXiv ID input */}
            <div style={{ marginBottom: 14 }}>
              <label style={{ display: "block", fontSize: "11px", fontWeight: 600, color: "#9ca3af", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                arXiv ID
              </label>
              <input
                type="text"
                value={importId}
                onChange={e => { setImportId(e.target.value); setImportError(null); setImportMsg(null); }}
                onKeyDown={e => { if (e.key === "Enter" && !importing) handleImport(); }}
                placeholder="e.g. 1706.03762"
                autoFocus
                style={{
                  width: "100%", padding: "9px 12px", borderRadius: 8,
                  border: "1px solid rgba(55,65,81,0.6)", background: "rgba(17,24,39,0.8)",
                  color: "#f9fafb", fontSize: "13px", fontFamily: "monospace",
                  outline: "none", boxSizing: "border-box",
                }}
                onFocus={e => { (e.target as HTMLInputElement).style.borderColor = "rgba(16,185,129,0.5)"; }}
                onBlur={e => { (e.target as HTMLInputElement).style.borderColor = "rgba(55,65,81,0.6)"; }}
              />
              <p style={{ fontSize: "10px", color: "#4b5563", marginTop: 5 }}>
                Paste the number from arxiv.org/abs/<strong style={{ color: "#6b7280" }}>2305.12345</strong> — version suffix optional
              </p>
            </div>

            {/* Namespace multi-select */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                <label style={{ fontSize: "11px", fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Import into namespaces
                </label>
                <span style={{ fontSize: "10px", color: "#4b5563" }}>
                  {importNs.length === 0 ? "none selected" : `${importNs.length} selected`}
                </span>
              </div>
              <div style={{
                maxHeight: 140, overflowY: "auto", borderRadius: 8,
                border: "1px solid rgba(55,65,81,0.6)", background: "rgba(17,24,39,0.8)",
                padding: "4px 0",
              }}>
                {NAMESPACE_TREE.flatMap(s => s.topics).map(t => {
                  const checked = importNs.includes(t.key);
                  return (
                    <label
                      key={t.key}
                      style={{
                        display: "flex", alignItems: "center", gap: 8,
                        padding: "5px 10px", cursor: "pointer",
                        background: checked ? "rgba(16,185,129,0.08)" : "transparent",
                        transition: "background 0.1s",
                      }}
                      onMouseEnter={e => { if (!checked) (e.currentTarget as HTMLLabelElement).style.background = "rgba(255,255,255,0.03)"; }}
                      onMouseLeave={e => { (e.currentTarget as HTMLLabelElement).style.background = checked ? "rgba(16,185,129,0.08)" : "transparent"; }}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => setImportNs(prev =>
                          checked ? prev.filter(k => k !== t.key) : [...prev, t.key]
                        )}
                        style={{ accentColor: "#34d399", cursor: "pointer", flexShrink: 0 }}
                      />
                      <span style={{ fontSize: "11px", color: checked ? "#34d399" : "#9ca3af", fontFamily: "monospace" }}>{t.key}</span>
                      <span style={{ fontSize: "10px", color: "#4b5563", flex: 1 }}>{t.label}</span>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Status messages */}
            {importError && (
              <div style={{ padding: "8px 12px", borderRadius: 8, background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.25)", marginBottom: 14 }}>
                <p style={{ fontSize: "11px", color: "#f87171", margin: 0 }}>{importError}</p>
              </div>
            )}
            {importMsg && (
              <div style={{ padding: "8px 12px", borderRadius: 8, background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.25)", marginBottom: 14, display: "flex", alignItems: "center", gap: 8 }}>
                <CheckCircleIcon size={14} color="#34d399" />
                <p style={{ fontSize: "11px", color: "#34d399", margin: 0 }}>{importMsg}</p>
              </div>
            )}

            {/* Action buttons */}
            <div style={{ display: "flex", gap: 10 }}>
              <button
                onClick={handleImport}
                disabled={importing || !importId.trim() || importNs.length === 0}
                style={{
                  flex: 1, padding: "10px 0", borderRadius: 9,
                  border: "1px solid rgba(16,185,129,0.35)",
                  background: importing || !importId.trim() || importNs.length === 0 ? "rgba(16,185,129,0.05)" : "rgba(16,185,129,0.12)",
                  color: importing || !importId.trim() || importNs.length === 0 ? "#4b5563" : "#34d399",
                  fontSize: "12px", fontWeight: 600, cursor: importing || !importId.trim() || importNs.length === 0 ? "not-allowed" : "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
                  transition: "all 0.15s",
                }}
              >
                {importing ? (
                  <><Loader2Icon size={13} className="animate-spin" /> Importing…</>
                ) : (
                  <><PlusIcon size={13} /> Import &amp; Enrich</>
                )}
              </button>
              <button
                onClick={() => setImportOpen(false)}
                style={{
                  padding: "10px 18px", borderRadius: 9,
                  border: "1px solid rgba(55,65,81,0.5)", background: "rgba(55,65,81,0.08)",
                  color: "#6b7280", fontSize: "12px", cursor: "pointer",
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Search result card ───────────────────────────────────────────────────────

function RelevanceBar({ score }: { score: number }) {
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  const color = pct >= 65 ? "#34d399" : pct >= 40 ? "#fbbf24" : "#6b7280";
  return (
    <div title={`Relevance to query: ${pct}%`} style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <div style={{ width: 48, height: 4, background: "rgba(55,65,81,0.5)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 2, transition: "width 0.3s" }} />
      </div>
      <span style={{ fontSize: "9px", color, fontWeight: 600, minWidth: 24 }}>{pct}%</span>
    </div>
  );
}

function SearchResultCard({ result, isSelected, onClick }: { result: SearchResult; isSelected: boolean; onClick: () => void }) {
  const isDeepResult = result.match_type === "deep";
  return (
    <div
      onClick={onClick}
      style={{
        padding: "13px 15px", borderRadius: 12, cursor: "pointer",
        border: `1px solid ${isSelected
          ? (isDeepResult ? "rgba(139,92,246,0.5)" : "rgba(99,102,241,0.5)")
          : "var(--rf-card-border)"}`,
        background: isSelected
          ? (isDeepResult ? "rgba(139,92,246,0.08)" : "rgba(99,102,241,0.08)")
          : "var(--rf-card)",
        transition: "border-color 0.15s, background 0.15s",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10, marginBottom: 6 }}>
        <h3 style={{ fontSize: "12.5px", fontWeight: 600, color: "var(--rf-text1)", lineHeight: 1.4, flex: 1 }}>{result.title}</h3>
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
          {isDeepResult && <RelevanceBar score={result.search_score} />}
          {result.is_manually_imported && (
            <span
              title="You imported this paper manually"
              style={{
                display: "inline-flex", alignItems: "center", gap: 3,
                padding: "1px 6px", borderRadius: 4,
                border: "1px solid rgba(34,197,94,0.35)",
                background: "rgba(34,197,94,0.08)",
                color: "#4ade80", fontSize: "9px", fontWeight: 600,
                letterSpacing: "0.04em", textTransform: "uppercase" as const,
              }}
            >
              <PlusIcon size={8} />
              Imported
            </span>
          )}
          <MatchTypeBadge type={result.match_type} />
        </div>
      </div>
      <p style={{ fontSize: "10px", color: "var(--rf-text4)", marginBottom: 10, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" as const, overflow: "hidden" }}>
        {result.tldr ?? cleanAbstract(result.abstract)}
      </p>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: "10px", color: "var(--rf-text4)" }}>
            {result.authors.slice(0, 2).join(", ")}{result.authors.length > 2 ? ` +${result.authors.length - 2}` : ""}
          </span>
          <span style={{ fontSize: "9px", color: "var(--rf-text5)", fontFamily: "monospace" }}>{result.namespace_key}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <a href={result.source_url} target="_blank" rel="noopener noreferrer" onClick={e => e.stopPropagation()} style={{ color: "var(--rf-text5)" }}>
            <ExternalLinkIcon size={12} />
          </a>
        </div>
      </div>
    </div>
  );
}

