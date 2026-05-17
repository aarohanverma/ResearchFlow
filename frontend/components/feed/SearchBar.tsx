"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  SearchIcon, XIcon, Loader2Icon, ZapIcon, SparklesIcon,
} from "lucide-react";
import { api } from "@/lib/api";

// ── Shared result type ────────────────────────────────────────────────────────

export interface SearchResult {
  paper_id: string;
  external_id?: string;
  title: string;
  abstract: string;
  authors: string[];
  namespace_key: string;
  /** All topic memberships visible to the user (deduper output). Optional. */
  namespace_keys?: string[];
  source_url: string;
  pdf_url: string | null;
  novelty_score: number;
  relevance_score: number;
  is_breakthrough: boolean;
  is_manually_imported?: boolean;
  key_concepts: string[] | null;
  methods_used: string[] | null;
  implications: string | null;
  published_at: string | null;
  ingested_at: string | null;
  tldr: string | null;
  search_score: number;
  match_type: "keyword" | "semantic" | "hybrid" | "deep";
}

interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
  mode: string;
}

interface DeepJobResponse {
  job_id: string;
  status: "pending" | "done" | "failed";
  query: string;
  rewritten_query: string | null;
  results: SearchResult[] | null;
  error: string | null;
  cached: boolean;
}

export interface DeepSearchMeta {
  rewrittenQuery: string | null;
  cached: boolean;
  jobId: string | null;
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  /** Namespace keys passed to basic search. When undefined, searches all papers. */
  namespace_keys?: string[];
  onResults: (results: SearchResult[] | null, query: string, deepMeta?: DeepSearchMeta) => void;
  onClear: () => void;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const DEBOUNCE_MS = 350;
const DEEP_POLL_INTERVAL_MS = 2000;
const DEEP_POLL_MAX_ATTEMPTS = 30; // 30 × 2 s = 60 s timeout

export const MATCH_BADGE: Record<string, string> = {
  hybrid:  "border border-indigo-500/40 text-indigo-500",
  semantic:"border border-purple-500/40 text-purple-500",
  keyword: "border border-gray-500/30 text-gray-500",
  deep:    "border border-violet-500/40 text-violet-500",
};

// ── Component ─────────────────────────────────────────────────────────────────

export function SearchBar({ namespace_keys, onResults, onClear }: Props) {
  const [query, setQuery]           = useState("");
  const [deepMode, setDeepMode]     = useState(false);
  // arXiv MCP fetch is opt-in per-search: when on, the deep-search pipeline
  // pulls fresh candidates from arXiv via MCP and imports non-duplicates into
  // the user's feed. Off by default so heavy fetches are explicit.
  const [arxivFetch, setArxivFetch] = useState(false);
  const [loading, setLoading]       = useState(false);
  const [deepStatus, setDeepStatus] = useState<"idle" | "validating" | "searching" | "reranking" | "done" | "failed">("idle");
  const [error, setError]           = useState<string | null>(null);

  const debounceRef    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestQuery    = useRef("");
  const pollRef        = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollAttempts   = useRef(0);

  const nsKey = namespace_keys?.join(",") ?? "";

  // ── Basic search (debounced) ───────────────────────────────────────────────

  const doBasicSearch = useCallback(
    async (q: string) => {
      if (q.trim().length < 2) { onResults(null, ""); return; }

      setLoading(true);
      setError(null);
      latestQuery.current = q;

      try {
        const params = new URLSearchParams({ q, limit: "25", mode: "hybrid" });
        if (nsKey) params.set("namespace_keys", nsKey);

        const data = await api.get<SearchResponse>(`/search?${params}`);
        if (latestQuery.current === q) onResults(data.results, q);
      } catch (err: unknown) {
        if (latestQuery.current === q) {
          setError(err instanceof Error ? err.message : "Search failed");
          onResults([], q);
        }
      } finally {
        if (latestQuery.current === q) setLoading(false);
      }
    },
    [onResults]
  );

  // ── Deep search (submit on Enter / button) ─────────────────────────────────

  const doDeepSearch = useCallback(
    async (q: string) => {
      if (q.trim().length < 3) { setError("Enter at least 3 characters for Deep Search."); return; }

      setLoading(true);
      setError(null);
      setDeepStatus("validating");
      latestQuery.current = q;

      try {
        const body: Record<string, unknown> = {
          query: q,
          limit: 25,
          include_arxiv_mcp: arxivFetch,
          arxiv_max_results: arxivFetch ? 8 : 0,
        };
        if (namespace_keys && namespace_keys.length > 0) {
          body.namespace_keys = namespace_keys;
        }

        // Start background job so the UI doesn't block
        const job = await api.post<DeepJobResponse>("/search/deep-bg", body);

        if (latestQuery.current !== q) { setLoading(false); setDeepStatus("idle"); return; }

        if (job.status === "done") {
          // Cache hit — returned immediately
          setDeepStatus("done");
          setLoading(false);
          onResults(job.results ?? [], q, {
            rewrittenQuery: job.rewritten_query,
            cached: job.cached,
            jobId: job.job_id,
          });
          return;
        }

        if (job.status === "failed") {
          setDeepStatus("failed");
          setLoading(false);
          setError(job.error ?? "Deep search failed.");
          onResults([], q);
          return;
        }

        // Poll for completion
        setDeepStatus("searching");
        pollAttempts.current = 0;

        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = setInterval(async () => {
          pollAttempts.current++;

          if (latestQuery.current !== q) {
            if (pollRef.current) clearInterval(pollRef.current);
            setLoading(false);
            setDeepStatus("idle");
            return;
          }

          if (pollAttempts.current > DEEP_POLL_MAX_ATTEMPTS) {
            if (pollRef.current) clearInterval(pollRef.current);
            setLoading(false);
            setDeepStatus("failed");
            setError("Deep search timed out. Try again.");
            onResults([], q);
            return;
          }

          // Show animated status labels during polling
          if (pollAttempts.current > 8)  setDeepStatus("reranking");
          else if (pollAttempts.current > 3) setDeepStatus("searching");

          try {
            const status = await api.get<DeepJobResponse>(`/search/deep/status/${job.job_id}`);
            if (status.status === "done") {
              if (pollRef.current) clearInterval(pollRef.current);
              setDeepStatus("done");
              setLoading(false);
              onResults(status.results ?? [], q, {
                rewrittenQuery: status.rewritten_query,
                cached: status.cached,
                jobId: status.job_id,
              });
            } else if (status.status === "failed") {
              if (pollRef.current) clearInterval(pollRef.current);
              setDeepStatus("failed");
              setLoading(false);
              setError(status.error ?? "Deep search failed.");
              onResults([], q);
            }
          } catch { /* network blip — keep polling */ }
        }, DEEP_POLL_INTERVAL_MS);

      } catch (err: unknown) {
        setDeepStatus("failed");
        setLoading(false);
        setError(err instanceof Error ? err.message : "Deep search failed");
        onResults([], q);
      }
    },
    [namespace_keys, onResults, arxivFetch]
  );

  // ── Effect: debounce basic search, do nothing for deep mode ───────────────

  useEffect(() => {
    if (deepMode) return; // Deep mode triggers on Enter, not on type

    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!query.trim()) { onClear(); setError(null); setLoading(false); return; }
    debounceRef.current = setTimeout(() => doBasicSearch(query), DEBOUNCE_MS);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [query, deepMode, doBasicSearch, onClear]);

  // ── Effect: re-run basic search when namespace selection changes ───────────
  useEffect(() => {
    if (!deepMode && query.trim().length >= 2) doBasicSearch(query);
  }, [nsKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Handlers ──────────────────────────────────────────────────────────────

  function clear() {
    setQuery("");
    setError(null);
    setLoading(false);
    setDeepStatus("idle");
    latestQuery.current = "";
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    onClear();
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (deepMode && e.key === "Enter" && !loading) {
      e.preventDefault();
      doDeepSearch(query);
    }
  }

  function handleDeepModeToggle() {
    const next = !deepMode;
    setDeepMode(next);
    setError(null);
    // If there's an active query in basic mode, keep showing those results
    // (no re-search on toggle — user must press Enter in deep mode)
    if (next && !query.trim()) {
      // Just switched to deep mode with empty query — clear any old results
    }
  }

  // ── Status label for deep search animation ─────────────────────────────────

  const deepLabel: Record<typeof deepStatus, string> = {
    idle:       "",
    validating: "Validating query…",
    searching:  "Searching papers…",
    reranking:  "Ranking results…",
    done:       "",
    failed:     "",
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  const borderColor = error
    ? "border-red-500"
    : deepMode
    ? "border-violet-500/60 focus-within:border-violet-500"
    : "border-[var(--rf-input-border)] focus-within:border-brand";

  return (
    <div className="relative w-full">
      {/* ── Deep mode indicator bar ── */}
      {deepMode && (
        <div style={{
          display: "flex", alignItems: "center", gap: 6,
          marginBottom: 5, padding: "2px 0",
        }}>
          <SparklesIcon size={10} color="#8b5cf6" />
          <span style={{ fontSize: "9px", color: "#8b5cf6", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase" }}>
            Deep Search — natural language · press Enter to search
          </span>
          {loading && deepLabel[deepStatus] && (
            <>
              <span style={{ fontSize: "9px", color: "#6b7280" }}>·</span>
              <span style={{ fontSize: "9px", color: "#a78bfa" }}>{deepLabel[deepStatus]}</span>
            </>
          )}
          {/* Opt-in: also fetch fresh candidates from arXiv MCP and import
              non-duplicates into the user's feed during this search. */}
          <button
            onClick={() => setArxivFetch(v => !v)}
            disabled={loading}
            title={arxivFetch
              ? "arXiv MCP fetch enabled — non-duplicates will be imported into your feed."
              : "Click to also pull fresh papers from arXiv into your feed during this search."}
            style={{
              marginLeft: "auto",
              display: "flex", alignItems: "center", gap: 4,
              padding: "2px 8px", borderRadius: 6, cursor: loading ? "not-allowed" : "pointer",
              border: `1px solid ${arxivFetch ? "rgba(34,197,94,0.5)" : "rgba(55,65,81,0.6)"}`,
              background: arxivFetch ? "rgba(34,197,94,0.12)" : "rgba(31,41,55,0.4)",
              color: arxivFetch ? "#4ade80" : "#6b7280",
              fontSize: "9px", fontWeight: 700, letterSpacing: "0.04em",
              textTransform: "uppercase" as const,
            }}
            aria-pressed={arxivFetch}
          >
            <ZapIcon size={9} />
            {arxivFetch ? "arXiv: on" : "arXiv: off"}
          </button>
        </div>
      )}

      {/* ── Input row ── */}
      <div
        className={`flex items-center gap-2 border rounded-xl px-3 py-2 transition-colors ${borderColor}`}
        style={{
          background: "var(--rf-input)",
          ...(deepMode ? { boxShadow: "0 0 0 1px rgba(139,92,246,0.15)" } : {}),
        }}
      >
        {/* Left icon — spinner or search */}
        {loading ? (
          <Loader2Icon
            size={16}
            className="flex-shrink-0 animate-spin"
            style={{ color: deepMode ? "#8b5cf6" : "#9ca3af" }}
          />
        ) : deepMode ? (
          <SparklesIcon size={16} className="flex-shrink-0" style={{ color: "#8b5cf6" }} />
        ) : (
          <SearchIcon size={16} className="text-gray-400 flex-shrink-0" />
        )}

        {/* Query input */}
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            deepMode
              ? "Describe what you're looking for in natural language…"
              : "Search papers by topic, method, author…"
          }
          className="flex-1 bg-transparent text-sm outline-none"
          style={{ color: deepMode ? "#e9d5ff" : "var(--rf-text1)" }}
          autoComplete="off"
          spellCheck={false}
        />

        {/* Deep search submit button (only in deep mode, only when query not empty) */}
        {deepMode && query.trim().length >= 3 && !loading && (
          <button
            onClick={() => doDeepSearch(query)}
            title="Run Deep Search"
            style={{
              display: "flex", alignItems: "center", gap: 4,
              background: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.3)",
              borderRadius: 6, padding: "2px 8px", cursor: "pointer",
              color: "#a78bfa", fontSize: "10px", fontWeight: 600,
              flexShrink: 0, whiteSpace: "nowrap",
            }}
          >
            <ZapIcon size={10} />
            Search
          </button>
        )}

        {/* Clear button */}
        {query && (
          <button
            onClick={clear}
            className="text-gray-500 hover:text-gray-300 flex-shrink-0 transition-colors"
            aria-label="Clear search"
          >
            <XIcon size={14} />
          </button>
        )}

        {/* Deep Search toggle */}
        <button
          onClick={handleDeepModeToggle}
          title={deepMode ? "Switch to basic search" : "Switch to Deep Search (natural language)"}
          style={{
            display: "flex", alignItems: "center", gap: 4,
            padding: "3px 8px", borderRadius: 6, cursor: "pointer",
            border: `1px solid ${deepMode ? "rgba(139,92,246,0.5)" : "rgba(55,65,81,0.6)"}`,
            background: deepMode ? "rgba(139,92,246,0.15)" : "rgba(31,41,55,0.4)",
            color: deepMode ? "#a78bfa" : "#6b7280",
            fontSize: "9px", fontWeight: 700,
            letterSpacing: "0.04em", textTransform: "uppercase" as const,
            flexShrink: 0, transition: "all 0.15s",
            userSelect: "none" as const,
          }}
          aria-pressed={deepMode}
        >
          <SparklesIcon size={9} />
          Deep
        </button>
      </div>

      {/* Deep search wave animation — only shown while running */}
      {deepMode && loading && (
        <div style={{ marginTop: 4, height: 2, overflow: "hidden", borderRadius: 1 }}>
          <div style={{
            height: "100%",
            background: "linear-gradient(90deg, transparent, #8b5cf6, #c084fc, #8b5cf6, transparent)",
            backgroundSize: "200% 100%",
            animation: "deepwave 1.5s linear infinite",
          }} />
          <style>{`
            @keyframes deepwave {
              0%   { background-position: 200% 0 }
              100% { background-position: -200% 0 }
            }
          `}</style>
        </div>
      )}

      {/* Error message */}
      {error && (
        <p className="absolute -bottom-5 left-0 text-xs text-red-400">{error}</p>
      )}
    </div>
  );
}

// ── Match type badge ───────────────────────────────────────────────────────────

/** Badge showing how the result was matched. */
export function MatchTypeBadge({ type }: { type: string }) {
  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${
        MATCH_BADGE[type] ?? MATCH_BADGE.keyword
      }`}
    >
      {type === "deep" ? <><SparklesIcon size={8} style={{ marginRight: 2 }} />deep</> : type}
    </span>
  );
}
