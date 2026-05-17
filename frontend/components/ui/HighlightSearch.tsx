"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HighlighterIcon, XIcon } from "lucide-react";
import type { MarkdownDecorations } from "@/components/ui/MarkdownRenderer";

/**
 * Reusable highlight + keyword-search engine, factored out of the Research
 * Assistant chat so Study Mode and the Genie Idea Dive page can share the
 * exact same UX without re-deriving the React-tree decoration architecture.
 *
 * Architecture parity with the RA chat:
 *   * Highlights and search matches are emitted as React ``<mark>`` JSX
 *     elements via MarkdownRenderer's ``decorations`` prop. The renderer
 *     owns the marks, so they survive every parent re-render automatically
 *     — no DOM mutation, no MutationObserver feedback loop.
 *   * Highlights persist per ``scopeKey`` in localStorage (typically a
 *     stable identifier like ``study:<paperId>`` or ``idea:<capsuleId>``).
 *   * Keyword search supports case sensitive/insensitive toggle, prev/next
 *     navigation, and Ctrl/Cmd+F shortcut.
 *   * Highlight mode is opt-in: a single toolbar toggle. While off, plain
 *     selections behave normally; while on, releasing a selection adds a
 *     new highlight (or removes one if the selection exactly matches an
 *     existing highlight's text — re-selecting toggles it off).
 */

export interface Highlight {
  id: string;
  /** Free-form locator for the host page — e.g. section key, block index. */
  scope: string;
  text: string;
  color?: string;
}

export interface UseHighlightSearchResult {
  /** Pass-through decoration object for ``MarkdownRenderer``. */
  decorations: MarkdownDecorations;
  /** Attach to the scrollable container so search nav can scroll matches. */
  scrollRef: React.MutableRefObject<HTMLDivElement | null>;
  /** All toolbar state + setters. Wire into ``<HighlightSearchToolbar>``. */
  toolbar: {
    searchOpen: boolean;
    setSearchOpen: (v: boolean) => void;
    searchQuery: string;
    setSearchQuery: (v: string) => void;
    caseSensitive: boolean;
    setCaseSensitive: (v: boolean) => void;
    wholeWord: boolean;
    setWholeWord: (v: boolean) => void;
    matchCount: number;
    matchIdx: number;
    navigate: (dir: 1 | -1) => void;
    highlightMode: boolean;
    setHighlightMode: (v: boolean) => void;
    hasHighlights: boolean;
    clearHighlights: () => void;
  };
}

/**
 * Build the highlight + search state for a single scrollable region.
 *
 * @param scopeKey  Unique key for localStorage persistence.
 * @param enabled   When false, all decorations are skipped (saves a render
 *                  pass when the host hasn't rendered any content yet).
 */
export function useHighlightSearch(
  scopeKey: string,
  enabled: boolean = true,
): UseHighlightSearchResult {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const [highlightMode, setHighlightMode] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [wholeWord, setWholeWord] = useState(false);
  const [matchCount, setMatchCount] = useState(0);
  const [matchIdx, setMatchIdx] = useState(0);

  const storageKey = `rf-highlights-${scopeKey}`;
  const skipNextSaveRef = useRef(false);

  // Load → save race guard: switching scopeKey resets state and skips the
  // next save closure so the previous scope's data doesn't bleed into the
  // new key. Same pattern as the RA chat highlight effect.
  useEffect(() => {
    if (!enabled) return;
    setSearchOpen(false);
    setSearchQuery("");
    setCaseSensitive(false);
    setWholeWord(false);
    setMatchCount(0);
    setMatchIdx(0);
    try {
      const raw = localStorage.getItem(storageKey);
      setHighlights(raw ? (JSON.parse(raw) as Highlight[]) : []);
    } catch {
      setHighlights([]);
    }
    skipNextSaveRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey, enabled]);

  useEffect(() => {
    if (!enabled) return;
    if (skipNextSaveRef.current) {
      skipNextSaveRef.current = false;
      return;
    }
    try {
      localStorage.setItem(storageKey, JSON.stringify(highlights));
    } catch {}
  }, [highlights, storageKey, enabled]);

  const onRemoveHighlight = useCallback((id: string) => {
    setHighlights(prev => prev.filter(h => h.id !== id));
  }, []);

  // Highlight-mode mouseup handler. Mirrors the RA chat behaviour: selecting
  // un-highlighted text adds a highlight, selecting an existing highlight's
  // exact text removes it. Listens on `document` because text selections
  // can end outside the original target.
  useEffect(() => {
    if (!enabled || !highlightMode) return;
    function handle() {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) return;
      const text = sel.toString().trim();
      if (text.length < 2) return;
      const container = scrollRef.current;
      if (!container) return;
      const range = sel.getRangeAt(0);
      let el: Element | null = range.commonAncestorContainer instanceof Element
        ? range.commonAncestorContainer
        : range.commonAncestorContainer.parentElement;
      // Only accept selections inside our container — the user might select
      // some other unrelated text on the page.
      let inside = false;
      while (el) {
        if (el === container) { inside = true; break; }
        el = el.parentElement;
      }
      if (!inside) return;
      setHighlights(prev => {
        const existing = prev.find(h => h.text === text);
        if (existing) return prev.filter(h => h.id !== existing.id);
        return [...prev, {
          id: (typeof crypto !== "undefined" && crypto.randomUUID)
            ? crypto.randomUUID()
            : `h-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          scope: scopeKey,
          text,
          color: "#fef08a",
        }];
      });
      sel.removeAllRanges();
    }
    document.addEventListener("mouseup", handle);
    return () => document.removeEventListener("mouseup", handle);
  }, [highlightMode, scopeKey, enabled]);

  // Ctrl/Cmd+F → open search (only when our container has focus / is on screen).
  useEffect(() => {
    if (!enabled) return;
    function onKey(e: KeyboardEvent) {
      const container = scrollRef.current;
      if (!container) return;
      const inside = container.contains(document.activeElement) || container.matches(":hover");
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "f" && inside) {
        e.preventDefault();
        setSearchOpen(true);
      }
      if (e.key === "Escape" && searchOpen) {
        setSearchOpen(false);
        setSearchQuery("");
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [searchOpen, enabled]);

  // Recompute match count from the rendered DOM after React commits.
  useEffect(() => {
    if (!enabled) return;
    const container = scrollRef.current;
    if (!container || !searchQuery.trim()) {
      setMatchCount(0);
      return;
    }
    const marks = container.querySelectorAll("mark[data-rf-search]");
    setMatchCount(marks.length);
  }, [searchQuery, caseSensitive, wholeWord, enabled, highlights]);

  // Reset idx when query changes.
  useEffect(() => {
    setMatchIdx(0);
  }, [searchQuery]);

  // Re-tint the current match as the active one. Pure style mutation on
  // React-owned mark elements — no structural DOM change, so React's
  // reconciliation is unaffected.
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const marks = Array.from(container.querySelectorAll("mark[data-rf-search]")) as HTMLElement[];
    if (marks.length === 0) return;
    marks.forEach(m => { m.style.background = "#fbbf24"; });
    const idx = Math.max(0, Math.min(matchIdx, marks.length - 1));
    const cur = marks[idx];
    if (cur) cur.style.background = "#f59e0b";
  }, [matchIdx, matchCount]);

  const navigate = useCallback((dir: 1 | -1) => {
    const container = scrollRef.current;
    if (!container) return;
    const marks = Array.from(container.querySelectorAll("mark[data-rf-search]")) as HTMLElement[];
    if (marks.length === 0) return;
    const next = (matchIdx + dir + marks.length) % marks.length;
    setMatchIdx(next);
    marks[next]?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [matchIdx]);

  const clearHighlights = useCallback(() => setHighlights([]), []);

  const decorations: MarkdownDecorations = useMemo(() => ({
    highlights: highlights.map(h => ({ id: h.id, text: h.text, color: h.color || "#fef08a" })),
    searchQuery,
    searchCaseSensitive: caseSensitive,
    searchWholeWord: wholeWord,
    searchKeyPrefix: scopeKey,
    // Click-to-remove is only enabled while highlighter mode is on so the
    // marks behave like persistent annotations during normal reading, and
    // turn into removable widgets only when the user opts in.
    highlightClickToRemove: highlightMode,
    onRemoveHighlight,
  }), [highlights, searchQuery, caseSensitive, wholeWord, scopeKey, highlightMode, onRemoveHighlight]);

  return {
    decorations,
    scrollRef,
    toolbar: {
      searchOpen, setSearchOpen,
      searchQuery, setSearchQuery,
      caseSensitive, setCaseSensitive,
      wholeWord, setWholeWord,
      matchCount, matchIdx,
      navigate,
      highlightMode, setHighlightMode,
      hasHighlights: highlights.length > 0,
      clearHighlights,
    },
  };
}

/**
 * Toolbar that drives the {@link useHighlightSearch} hook. Pass the
 * ``toolbar`` slice from the hook directly — the component handles all
 * keyboard wiring, navigation, and case-toggle UI.
 */
export function HighlightSearchToolbar({
  toolbar,
  className = "",
}: {
  toolbar: UseHighlightSearchResult["toolbar"];
  className?: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (toolbar.searchOpen) setTimeout(() => inputRef.current?.focus(), 30);
  }, [toolbar.searchOpen]);

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      <div className="flex items-center gap-1.5">
        <button
          onClick={() => toolbar.setHighlightMode(!toolbar.highlightMode)}
          title={toolbar.highlightMode
            ? "Highlighter on — select text to highlight (click to switch off)"
            : "Highlighter off — click to enable"}
          aria-pressed={toolbar.highlightMode}
          className="px-2 py-1 rounded text-[11px] inline-flex items-center gap-1.5 border transition-colors"
          style={{
            background: toolbar.highlightMode ? "rgba(254,240,138,0.2)" : "transparent",
            borderColor: toolbar.highlightMode ? "rgba(254,240,138,0.6)" : "rgba(255,255,255,0.06)",
            color: toolbar.highlightMode ? "#ca8a04" : "var(--rf-text4)",
          }}
        >
          <HighlighterIcon size={12} />
          {toolbar.highlightMode ? "Highlighting" : "Highlight"}
        </button>
        <button
          onClick={() => {
            const next = !toolbar.searchOpen;
            toolbar.setSearchOpen(next);
            if (!next) toolbar.setSearchQuery("");
          }}
          title="Search (Ctrl+F)"
          aria-pressed={toolbar.searchOpen}
          className="px-2 py-1 rounded text-[11px] inline-flex items-center gap-1.5 border transition-colors"
          style={{
            background: toolbar.searchOpen ? "var(--rf-nav-active)" : "transparent",
            borderColor: toolbar.searchOpen ? "var(--rf-nav-border)" : "rgba(255,255,255,0.06)",
            color: "var(--rf-text4)",
          }}
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
          </svg>
          Search
        </button>
        {toolbar.hasHighlights && (
          <button
            onClick={toolbar.clearHighlights}
            title="Remove all highlights"
            className="px-2 py-1 rounded text-[11px] inline-flex items-center gap-1.5 border transition-colors"
            style={{
              background: "transparent",
              borderColor: "rgba(255,255,255,0.06)",
              color: "var(--rf-text5)",
            }}
          >
            <XIcon size={11} />
            Clear marks
          </button>
        )}
      </div>

      {toolbar.searchOpen && (
        <div
          className="flex items-center gap-2 mt-1 px-2 py-1.5 rounded border"
          style={{
            background: "var(--rf-surface1)",
            borderColor: "var(--rf-border)",
          }}
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="var(--rf-text4)" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
          </svg>
          <input
            ref={inputRef}
            value={toolbar.searchQuery}
            onChange={e => toolbar.setSearchQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter") toolbar.navigate(e.shiftKey ? -1 : 1);
              if (e.key === "Escape") {
                toolbar.setSearchOpen(false);
                toolbar.setSearchQuery("");
              }
            }}
            placeholder="Search…"
            className="flex-1 bg-transparent border-0 outline-none text-[12px]"
            style={{ color: "var(--rf-text1)" }}
          />
          {toolbar.searchQuery && (
            <span className="text-[10px] whitespace-nowrap" style={{ color: "var(--rf-text5)" }}>
              {toolbar.matchCount === 0 ? "no matches" : `${toolbar.matchIdx + 1} / ${toolbar.matchCount}`}
            </span>
          )}
          <button
            onClick={() => toolbar.setCaseSensitive(!toolbar.caseSensitive)}
            title={toolbar.caseSensitive
              ? "Case sensitive — click for case-insensitive"
              : "Case insensitive — click for case-sensitive"}
            aria-pressed={toolbar.caseSensitive}
            className="px-1.5 rounded text-[10px] font-mono font-bold border"
            style={{
              background: toolbar.caseSensitive ? "var(--rf-nav-active)" : "transparent",
              borderColor: toolbar.caseSensitive ? "var(--rf-nav-border)" : "transparent",
              color: toolbar.caseSensitive ? "var(--rf-text1)" : "var(--rf-text4)",
            }}
          >Aa</button>
          <button
            onClick={() => toolbar.setWholeWord(!toolbar.wholeWord)}
            title={toolbar.wholeWord
              ? "Exact match — click for partial (substring) matches"
              : "Partial match — click for exact whole-word matches"}
            aria-pressed={toolbar.wholeWord}
            className="px-1.5 rounded text-[10px] font-mono font-bold border"
            style={{
              background: toolbar.wholeWord ? "var(--rf-nav-active)" : "transparent",
              borderColor: toolbar.wholeWord ? "var(--rf-nav-border)" : "transparent",
              color: toolbar.wholeWord ? "var(--rf-text1)" : "var(--rf-text4)",
            }}
          >\b</button>
          <button
            onClick={() => toolbar.navigate(-1)}
            disabled={toolbar.matchCount === 0}
            className="px-1 text-[12px] disabled:opacity-40"
            style={{ color: "var(--rf-text4)", background: "transparent" }}
            title="Previous (Shift+Enter)"
          >↑</button>
          <button
            onClick={() => toolbar.navigate(1)}
            disabled={toolbar.matchCount === 0}
            className="px-1 text-[12px] disabled:opacity-40"
            style={{ color: "var(--rf-text4)", background: "transparent" }}
            title="Next (Enter)"
          >↓</button>
          <button
            onClick={() => { toolbar.setSearchOpen(false); toolbar.setSearchQuery(""); }}
            className="px-1"
            style={{ color: "var(--rf-text4)", background: "transparent" }}
          >
            <XIcon size={11} />
          </button>
        </div>
      )}
    </div>
  );
}
