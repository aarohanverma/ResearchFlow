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
} from "lucide-react";
import { cleanAbstract } from "@/lib/utils";
import { useBookmarksStore } from "@/store/bookmarks";
import { useNamespaceStore, NAMESPACE_TREE } from "@/store/namespace";

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

  const loadFeed = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch from all selected topics and merge
      const results = await Promise.allSettled(
        selectedTopics.map(ns => api.get<FeedResponse>(`/feed?namespace_key=${ns}&limit=20`))
      );
      const allPapers: FeedItem[] = [];
      const seen = new Set<string>();
      for (const r of results) {
        if (r.status === "fulfilled") {
          for (const item of r.value.papers) {
            if (!seen.has(item.paper.id)) { seen.add(item.paper.id); allPapers.push(item); }
          }
        }
      }
      // Sort by score descending
      allPapers.sort((a, b) => b.score - a.score);
      setFeed(allPapers.slice(0, 60));
    } catch (e) { console.error(e); }
    setLoading(false);
  }, [selectedTopics]);

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
      // First reload after ~8s to show new papers (ingestion is async on backend)
      setTimeout(() => { loadFeed(); setRefreshMsg("Generating summaries…"); }, 8000);
      // Generate TLDRs — backend awaits all completions before responding,
      // so reloading on resolve ensures cards show fresh TLDRs
      api.post("/papers/generate-tldrs?limit=50")
        .then(() => { loadFeed(); setRefreshMsg(null); })
        .catch(() => { setRefreshMsg(null); });
      setRefreshing(false);
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
          {/* Refresh button */}
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
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 200, color: "#4b5563", gap: 8 }}>
                <Loader2Icon className="animate-spin" size={18} /> Loading feed…
              </div>
            ) : isSearching ? (
              searchResults!.length === 0 ? (
                <div style={{ textAlign: "center", padding: "60px 20px", color: "#4b5563" }}>
                  <p style={{ fontSize: "15px", marginBottom: 6 }}>No results found</p>
                  <p style={{ fontSize: "12px" }}>Try different keywords or switch namespace.</p>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {searchResults!.map(r => (
                    <SearchResultCard
                      key={r.paper_id} result={r}
                      isSelected={selectedPaper?.id === r.paper_id}
                      onClick={() => setSelectedPaper(selectedPaper?.id === r.paper_id ? null : searchResultToPaper(r))}
                    />
                  ))}
                </div>
              )
            ) : filtered.length === 0 ? (
              <div style={{ textAlign: "center", padding: "60px 20px", color: "#4b5563" }}>
                <p style={{ fontSize: "15px", marginBottom: 6 }}>No papers yet in {activeNs}</p>
                <p style={{ fontSize: "12px", marginBottom: 16 }}>Hit the Refresh button above to fetch the latest from arXiv.</p>
                <button
                  onClick={handleRefresh}
                  disabled={refreshing}
                  style={{
                    padding: "8px 18px", borderRadius: 10, border: "1px solid rgba(99,102,241,0.4)",
                    background: "rgba(99,102,241,0.12)", color: "#818cf8",
                    fontSize: "12px", fontWeight: 600, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: 6,
                  }}
                >
                  <RefreshCwIcon size={13} className={refreshing ? "animate-spin" : ""} />
                  Fetch Papers Now
                </button>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {(filtered as FeedItem[]).map(item => (
                  <PaperCard
                    key={item.paper.id} item={item}
                    isSelected={selectedPaper?.id === item.paper.id}
                    onClick={() => setSelectedPaper(selectedPaper?.id === item.paper.id ? null : item.paper)}
                    onFeedback={handleFeedback}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right panel */}
        {selectedPaper && (
          <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
            <PaperPanel paper={selectedPaper} onClose={() => setSelectedPaper(null)} />
            {isBookmarked(selectedPaper.id) && (
              <RelatedPapersPanel paper={selectedPaper} onSelectPaper={p => setSelectedPaper(p)} />
            )}
          </div>
        )}
      </div>
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

