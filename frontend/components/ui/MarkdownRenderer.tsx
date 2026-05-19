"use client";

import { createContext, Fragment, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { CopyIcon, CheckIcon } from "lucide-react";


// ── Decoration pipeline (highlights + keyword search) ────────────────────────
//
// Highlights and search marks are RENDERED AS REACT ELEMENTS, not inserted
// into the DOM after render. That's the whole point: React owns the mark
// elements, so React's reconciliation never tears them down. Streaming, polling,
// and any other re-render is fine — the marks are part of the virtual DOM and
// survive automatically.
//
// The renderer accepts a ``MarkdownDecorations`` object that describes the
// highlights and the current search query. ``InlineText`` (the leaf that turns
// a plain-text span into React) finds occurrences of each highlight's text and
// the search query within its segment and emits ``<mark>`` JSX nodes. The
// data attributes on those nodes let the parent component locate marks for
// navigation (``data-rf-search-key``) and the click handler ties highlight
// removal to React state.

export interface DecorationHighlight {
  id: string;
  text: string;
  color: string;
}

export interface MarkdownDecorations {
  /** Highlights to render inside this content. */
  highlights?: DecorationHighlight[];
  /** Trimmed search query. Case sensitivity controlled by ``searchCaseSensitive``. */
  searchQuery?: string;
  /** When true, matches must match case exactly. Defaults to false (case-insensitive). */
  searchCaseSensitive?: boolean;
  /**
   * When true, search matches must be WHOLE-WORD only — "race" matches
   * "race" but not "traces". When false (default), substring matches are
   * allowed so "race" also highlights inside "traces".
   */
  searchWholeWord?: boolean;
  /** Optional stable key prefix so the parent can correlate ``data-rf-search-key`` values across messages. */
  searchKeyPrefix?: string;
  /**
   * When true, clicking a highlight mark removes it. False (default) keeps
   * the marks visually present but inert — so existing highlights don't
   * accidentally disappear when the user is just reading or selecting text
   * for copying. Toggle this on alongside the highlighter button.
   */
  highlightClickToRemove?: boolean;
  /** Called when the user clicks a highlight mark to remove it. */
  onRemoveHighlight?: (id: string) => void;
}

// ── Decorations context ──────────────────────────────────────────────────────
//
// Pages with their own custom renderContent / InlineText (study mode, genie
// idea dive) cannot easily plumb a ``decorations`` prop through every recursive
// call site. Instead they wrap their content tree in ``DecorationsProvider``
// and call ``useDecorations()`` inside their local InlineText. The context
// pattern keeps decoration rendering universal without forcing those pages to
// switch to the shared MarkdownRenderer entirely.

const DecorationsContext = createContext<MarkdownDecorations | undefined>(undefined);

export function DecorationsProvider({
  value,
  children,
}: {
  value: MarkdownDecorations | undefined;
  children: ReactNode;
}) {
  return <DecorationsContext.Provider value={value}>{children}</DecorationsContext.Provider>;
}

/** Read the active decoration object (highlights + search query). */
export function useDecorations(): MarkdownDecorations | undefined {
  return useContext(DecorationsContext);
}

/**
 * Public re-export of the decoration emitter so foreign InlineText
 * implementations (study, idea dive) can apply the exact same React
 * ``<mark>`` rendering without duplicating the match-finding logic.
 */
export function decorateString(s: string, dec: MarkdownDecorations | undefined, keyPrefix: string): ReactNode[] {
  return _decorateString(s, dec, keyPrefix);
}

interface DecorationMatch {
  start: number;
  end: number;
  kind: "highlight" | "search";
  id?: string;
  color?: string;
}

/** Return every exact substring match of ``needle`` inside ``hay``. */
function _exactHighlightSpans(hay: string, needle: string): { start: number; end: number }[] {
  const out: { start: number; end: number }[] = [];
  let cursor = 0;
  while (true) {
    const pos = hay.indexOf(needle, cursor);
    if (pos === -1) break;
    out.push({ start: pos, end: pos + needle.length });
    cursor = pos + Math.max(1, needle.length);
  }
  return out;
}

/** Cross-formatting-boundary fallback for highlight matching.
 *
 * Markdown renders bold / italic / code / headings / inline math /
 * citation chips as separate text nodes. A selection that crosses one
 * of those boundaries produces a highlight whose stored text contains
 * the joined contents — but no single rendered text node holds the
 * full string. This helper finds every contiguous slice of ``segment``
 * that also appears verbatim in ``highlight`` and is at least
 * ``minLen`` characters long, so each rendered text node still lights
 * up with the portion of the selection it actually contains.
 *
 * The implementation is intentionally greedy from left to right and
 * never returns overlapping spans — the caller's normal overlap
 * resolver then deduplicates against search matches.
 */
function _fuzzyHighlightSpans(
  segment: string,
  highlight: string,
  minLen: number,
): { start: number; end: number }[] {
  const out: { start: number; end: number }[] = [];
  const segLen = segment.length;
  const hlLen = highlight.length;
  if (segLen === 0 || hlLen === 0 || segLen < minLen) return out;
  let i = 0;
  while (i + minLen <= segLen) {
    // Find the longest substring of ``segment`` starting at ``i`` that
    // also appears inside ``highlight``.  Walk outward from the
    // longest possible end down to ``minLen``; first hit wins.
    let bestEnd = -1;
    for (let end = segLen; end - i >= minLen; end--) {
      const candidate = segment.slice(i, end);
      if (highlight.indexOf(candidate) !== -1) {
        bestEnd = end;
        break;
      }
    }
    if (bestEnd === -1) {
      i += 1;
      continue;
    }
    out.push({ start: i, end: bestEnd });
    i = bestEnd;
  }
  return out;
}

/** Build the sorted, non-overlapping match list for one text segment.
 *
 * Highlight matching used to be a pure ``text.indexOf(hl.text)`` so a
 * selection that spanned ``**bold**``, headings, inline code, LaTeX,
 * or citation chips never produced any highlight at all — the raw
 * stored ``hl.text`` ("part of bold part") never appeared verbatim in
 * a single rendered text node (the bold word lives in a separate node
 * from its surrounding text). The fix here is a two-stage match:
 *
 *   1. Try the exact substring match first — the common case where
 *      the user selected within a homogeneous text run.
 *   2. On miss, fall back to the longest contiguous substring of the
 *      highlight that also appears in this text segment. That covers
 *      cross-boundary selections: each underlying text node only
 *      contains its slice of the original selection, and we mark
 *      whatever slice IS present here. A minimum length of 3 chars
 *      keeps the fallback from highlighting incidental short tokens
 *      like " of " that happen to overlap with the stored text.
 */
function _findMatches(text: string, dec?: MarkdownDecorations): DecorationMatch[] {
  if (!dec) return [];
  const matches: DecorationMatch[] = [];

  // Highlights — case-sensitive (the user selected the exact text).
  for (const hl of dec.highlights || []) {
    if (!hl.text) continue;
    const exactSpans = _exactHighlightSpans(text, hl.text);
    if (exactSpans.length) {
      for (const sp of exactSpans) {
        matches.push({
          start: sp.start,
          end: sp.end,
          kind: "highlight",
          id: hl.id,
          color: hl.color,
        });
      }
      continue;
    }
    // Cross-boundary fallback: locate every chunk of this segment
    // that exists inside the highlight (and is long enough to be
    // meaningful), so e.g. a selection of "part **bold** part" still
    // highlights each of the three rendered text nodes individually.
    const fuzzy = _fuzzyHighlightSpans(text, hl.text, 3);
    for (const sp of fuzzy) {
      matches.push({
        start: sp.start,
        end: sp.end,
        kind: "highlight",
        id: hl.id,
        color: hl.color,
      });
    }
  }

  // Search query — case sensitivity + whole-word are controlled by the caller.
  const rawQ = (dec.searchQuery || "").trim();
  if (rawQ) {
    const caseSensitive = !!dec.searchCaseSensitive;
    const wholeWord = !!dec.searchWholeWord;
    const q = caseSensitive ? rawQ : rawQ.toLowerCase();
    const haystack = caseSensitive ? text : text.toLowerCase();
    const isBoundary = (ch: string | undefined): boolean => {
      if (!ch) return true;
      // Word character set tuned to natural language: letters, digits,
      // underscore. Everything else (spaces, punctuation, symbols) counts
      // as a boundary — so "race." or "(race)" still match in whole-word
      // mode but "traces" does not.
      return !/[A-Za-z0-9_]/.test(ch);
    };
    let cursor = 0;
    while (true) {
      const pos = haystack.indexOf(q, cursor);
      if (pos === -1) break;
      const end = pos + q.length;
      const okBoundary = !wholeWord
        || (isBoundary(text[pos - 1]) && isBoundary(text[end]));
      if (okBoundary) {
        matches.push({ start: pos, end, kind: "search" });
      }
      cursor = pos + Math.max(1, q.length);
    }
  }

  if (matches.length === 0) return matches;

  // Sort by start, then prefer longer matches on tie.
  matches.sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start));

  // Resolve overlaps — keep earlier (longer) matches, drop ones that overlap.
  const filtered: DecorationMatch[] = [];
  let cursor = 0;
  for (const m of matches) {
    if (m.start < cursor) continue;
    filtered.push(m);
    cursor = m.end;
  }
  return filtered;
}

/** Take a plain string and produce React nodes with decoration ``<mark>`` wrappers. */
function _decorateString(
  s: string,
  dec: MarkdownDecorations | undefined,
  keyPrefix: string,
): ReactNode[] {
  const matches = _findMatches(s, dec);
  if (matches.length === 0) return [s];
  const out: ReactNode[] = [];
  let pos = 0;
  matches.forEach((m, i) => {
    if (m.start > pos) out.push(s.slice(pos, m.start));
    const seg = s.slice(m.start, m.end);
    if (m.kind === "highlight") {
      const id = m.id || "";
      const clickToRemove = !!dec?.highlightClickToRemove && !!dec?.onRemoveHighlight;
      out.push(
        <mark
          key={`${keyPrefix}-h-${i}-${id}`}
          data-rf-highlight="1"
          data-rf-highlight-id={id}
          onClick={clickToRemove ? (e) => {
            e.preventDefault();
            e.stopPropagation();
            dec?.onRemoveHighlight?.(id);
          } : undefined}
          style={{
            background: m.color || "#fef08a",
            color: "#1f2937",
            borderRadius: 2,
            padding: "0 1px",
            cursor: clickToRemove ? "pointer" : "inherit",
          }}
          title={clickToRemove ? "Click to remove highlight" : undefined}
        >
          {seg}
        </mark>
      );
    } else {
      // search mark — searchKeyPrefix lets the parent navigate by index
      const searchKey = `${dec?.searchKeyPrefix || ""}-${keyPrefix}-${i}`;
      out.push(
        <mark
          key={`${keyPrefix}-s-${i}`}
          data-rf-search="1"
          data-rf-search-key={searchKey}
          style={{
            background: "#fbbf24",
            color: "#1f2937",
            borderRadius: 2,
            padding: "0 1px",
          }}
        >
          {seg}
        </mark>
      );
    }
    pos = m.end;
  });
  if (pos < s.length) out.push(s.slice(pos));
  return out;
}

// ── Mermaid diagram ───────────────────────────────────────────────────────────

/**
 * LLM-generated mermaid often has small syntax mistakes that mermaid's
 * tokenizer can't recover from. We do a best-effort cleanup pass so a single
 * bad node doesn't kill the whole diagram. All transforms are conservative
 * (no semantic edits) and idempotent.
 */
/**
 * Collapse whitespace runs that appear INSIDE node-label brackets to single
 * spaces. Mermaid expects each node declaration on one line; a hard newline
 * inside ``H[Factual Consistency\nConfidence (c_f)]`` causes the parser to
 * absorb every subsequent statement into that label as raw text — that's the
 * "raw mermaid shown as a single text block" failure mode we saw on Genie
 * deep dives. The walker tracks bracket depth so nested ``[`` / ``(`` inside
 * a label are handled correctly.
 */
function collapseLabelNewlines(spec: string, open: string, close: string): string {
  const out: string[] = [];
  let i = 0;
  while (i < spec.length) {
    const ch = spec[i];
    if (/[A-Za-z0-9_]/.test(ch)) {
      let j = i;
      while (j < spec.length && /[A-Za-z0-9_]/.test(spec[j])) j++;
      if (spec[j] === open) {
        let depth = 1;
        let k = j + 1;
        while (k < spec.length && depth > 0) {
          if (spec[k] === open) depth++;
          else if (spec[k] === close) {
            depth--;
            if (depth === 0) break;
          }
          k++;
        }
        if (k < spec.length && depth === 0) {
          const ident = spec.slice(i, j);
          const content = spec.slice(j + 1, k);
          const cleaned = content.replace(/\s+/g, " ").trim();
          out.push(ident + open + cleaned + close);
          i = k + 1;
          continue;
        }
      }
      out.push(spec.slice(i, j));
      i = j;
      continue;
    }
    out.push(ch);
    i++;
  }
  return out.join("");
}

export function sanitizeMermaidSpec(raw: string): string {
  let s = raw;

  // Strip ```mermaid fences if the LLM included them inside the block.
  s = s.replace(/^\s*```(?:mermaid)?\s*\n?/, "").replace(/\n?```\s*$/, "");

  // Citation markers like [5], [1-3], [1, 2, 3] inside node labels collide
  // with mermaid's label terminator. Convert them to parens — same visual,
  // no parser ambiguity. Standalone `[N]` never has a legal meaning in
  // mermaid source so this is safe everywhere.
  s = s.replace(/\[(\d+(?:\s*[-,]\s*\d+)*)\]/g, "($1)");

  // Collapse internal newlines inside node-label brackets. Without this,
  // a single multi-line label can swallow the rest of the spec into one
  // text block and the diagram fails to render at all.
  s = collapseLabelNewlines(s, "[", "]");
  s = collapseLabelNewlines(s, "(", ")");
  s = collapseLabelNewlines(s, "{", "}");

  // Trailing-semicolon edges sometimes break older parsers; leave as-is —
  // mermaid 10+ tolerates them.

  return s;
}

/**
 * Aggressive fallback: wrap every node label in double quotes so any
 * remaining problematic chars (parens, slashes, ampersands, stray brackets)
 * are treated as literal text. Used only after the first render fails.
 */
export function sanitizeMermaidAggressive(raw: string): string {
  const s = sanitizeMermaidSpec(raw);

  // Match node-label patterns: <id><open-bracket>...<close-bracket>
  // and quote the label content if not already quoted. We bracket-balance
  // the content so nested () or [] inside the label are absorbed correctly.
  function quoteLabels(input: string, open: string, close: string): string {
    const out: string[] = [];
    let i = 0;
    while (i < input.length) {
      const ch = input[i];
      if (/[A-Za-z0-9_]/.test(ch)) {
        let j = i;
        while (j < input.length && /[A-Za-z0-9_]/.test(input[j])) j++;
        if (input[j] === open) {
          let depth = 1;
          let k = j + 1;
          while (k < input.length && depth > 0) {
            if (input[k] === open) depth++;
            else if (input[k] === close) depth--;
            if (depth === 0) break;
            k++;
          }
          if (k < input.length && depth === 0) {
            const id = input.slice(i, j);
            const content = input.slice(j + 1, k);
            const trimmed = content.trim();
            const alreadyQuoted = trimmed.startsWith('"') && trimmed.endsWith('"');
            const safe = alreadyQuoted
              ? content
              : `"${content.replace(/"/g, "&quot;")}"`;
            out.push(id + open + safe + close);
            i = k + 1;
            continue;
          }
        }
        // Identifier but no balanced label — emit the identifier whole and skip past it.
        out.push(input.slice(i, j));
        i = j;
        continue;
      }
      out.push(ch);
      i++;
    }
    return out.join("");
  }

  let result = quoteLabels(s, "[", "]");
  result = quoteLabels(result, "(", ")");
  result = quoteLabels(result, "{", "}");
  return result;
}

function MermaidBlock({ spec }: { spec: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setError(false);
    let cancelled = false;
    (async () => {
      if (!ref.current) return;
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: "dark",
          themeVariables: {
            background: "transparent",
            primaryColor: "#312e81",
            primaryTextColor: "#ddd6fe",
            primaryBorderColor: "#7c3aed",
            lineColor: "#8b5cf6",
            secondaryColor: "#1e1b4b",
            fontSize: "13px",
          },
          flowchart: {
            htmlLabels: true,
            curve: "basis",
            useMaxWidth: true,
            wrappingWidth: 240,
            padding: 20,
            nodeSpacing: 50,
            rankSpacing: 60,
          },
          maxTextSize: 200000,
          securityLevel: "loose",
          // Suppress mermaid's auto-injected "Syntax error in text" bomb SVG.
          // We handle errors ourselves with a graceful fallback.
          suppressErrorRendering: true,
        });
        // Belt-and-suspenders: even with suppressErrorRendering, older paths can
        // still call parseError. Override to no-op so nothing surfaces.
        try { (mermaid as unknown as { parseError?: (...a: unknown[]) => void }).parseError = () => {}; } catch {}

        const tryRender = async (source: string) => {
          // Pre-validate. parse() with suppressErrors returns false on bad input
          // instead of throwing AND injecting the bomb SVG.
          const ok = await mermaid.parse(source, { suppressErrors: true });
          if (ok === false) throw new Error("mermaid parse failed");
          const id = `mermaid-${Math.random().toString(36).slice(2)}`;
          return mermaid.render(id, source);
        };

        let svg: string | null = null;
        try {
          ({ svg } = await tryRender(sanitizeMermaidSpec(spec)));
        } catch {
          // Fallback: quote every label, then retry once.
          try {
            ({ svg } = await tryRender(sanitizeMermaidAggressive(spec)));
          } catch {
            svg = null;
          }
        }
        // Defensive sweep: remove any stray bomb SVGs mermaid may have left in
        // the document body from older or concurrent renders.
        try {
          document.querySelectorAll('svg[aria-roledescription="error"], #mermaid-error-icon').forEach(n => n.remove());
        } catch {}
        if (cancelled) return;
        if (svg && ref.current) {
          ref.current.innerHTML = svg;
        } else {
          setError(true);
          if (ref.current) ref.current.innerHTML = "";
        }
      } catch {
        if (cancelled) return;
        setError(true);
        if (ref.current) ref.current.innerHTML = "";
      }
    })();
    return () => { cancelled = true; };
  }, [spec]);

  if (error) {
    return (
      <pre className="overflow-x-auto text-xs font-mono leading-relaxed p-4 rounded-lg border my-3" style={{ background: "#0d1117", color: "#adbac7", borderColor: "rgba(255,255,255,0.08)" }} data-code-block>
        {spec}
      </pre>
    );
  }
  return <div ref={ref} className="overflow-x-auto w-full my-3 [&_svg]:max-w-full" />;
}

// ── Shiki singleton ───────────────────────────────────────────────────────────

let _highlighter: any = null;
let _highlighterPromise: Promise<any> | null = null;

async function getHighlighter() {
  if (_highlighter) return _highlighter;
  if (!_highlighterPromise) {
    _highlighterPromise = import("shiki")
      .then(({ createHighlighter }) =>
        createHighlighter({
          themes: ["github-dark-dimmed"],
          langs: [
            "python", "javascript", "typescript", "tsx", "jsx",
            "bash", "sh", "rust", "go", "java", "cpp", "c",
            "sql", "json", "yaml", "toml", "markdown",
          ],
        })
      )
      .then((h) => { _highlighter = h; return h; });
  }
  return _highlighterPromise;
}

// ── Code block ────────────────────────────────────────────────────────────────

export function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [html, setHtml] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getHighlighter().then((h) => {
      if (cancelled) return;
      const safe = lang || "bash";
      try {
        setHtml(h.codeToHtml(code, { lang: safe, theme: "github-dark-dimmed" }));
      } catch {
        try {
          setHtml(h.codeToHtml(code, { lang: "bash", theme: "github-dark-dimmed" }));
        } catch {
          setHtml(null);
        }
      }
    });
    return () => { cancelled = true; };
  }, [code, lang]);

  function copy() {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const displayLang = lang
    ? lang === "tsx" || lang === "jsx"
      ? lang.toUpperCase()
      : lang.charAt(0).toUpperCase() + lang.slice(1)
    : "Code";

  return (
    <div className="rounded-xl overflow-hidden border border-gray-700/50 bg-[#0d1117] shadow-lg my-3" data-code-block>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-700/50 bg-[#161b22]">
        <div className="flex items-center gap-2.5">
          {/* Traffic lights */}
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/50" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/50" />
            <div className="w-3 h-3 rounded-full bg-green-500/50" />
          </div>
          <span className="text-xs font-semibold text-gray-400 font-mono">{displayLang}</span>
        </div>
        <button
          onClick={copy}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs text-gray-400 hover:text-white hover:bg-gray-700 transition-all"
          title="Copy code"
        >
          {copied ? (
            <>
              <CheckIcon size={11} className="text-emerald-400" />
              <span className="text-emerald-400">Copied!</span>
            </>
          ) : (
            <>
              <CopyIcon size={11} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code */}
      {html ? (
        <div
          className="overflow-x-auto text-[13px] leading-relaxed [&_pre]:px-5 [&_pre]:py-4 [&_pre]:m-0 [&_pre]:min-w-full [&_pre]:bg-transparent [&_code]:font-mono"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre className="overflow-x-auto px-5 py-4 text-[13px] leading-relaxed font-mono text-gray-300">
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}

// ── Inline text ───────────────────────────────────────────────────────────────

export function InlineText({
  text,
  onCitationClick,
  decorations,
}: {
  text: string;
  onCitationClick?: (num: string, isArxiv: boolean) => void;
  decorations?: MarkdownDecorations;
}) {
  const normalised = text.replace(/\\\(([\s\S]+?)\\\)/g, (_m, e) => `$${e}$`);
  // citation markers [1], [A1] included so they render as interactive chips
  const parts = normalised.split(
    /(\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`|\$[^$\n]{1,80}\$|\[A?\d+\])/g
  );
  // Helper to wrap any inner string in highlight/search decoration so
  // bold/italic/code/heading text is searchable & highlightable the same
  // way regular paragraph text is. Without this, selecting "bold word"
  // inside **bold word** silently dropped the highlight — that was the
  // "doesn't work on all text" bug.
  const dec = (s: string, k: string): ReactNode =>
    decorations ? <>{_decorateString(s, decorations, k)}</> : s;
  return (
    <>
      {parts.map((part, i) => {
        if (/^\*\*[^*]+\*\*$/.test(part))
          return <strong key={i} className="font-semibold text-white">{dec(part.slice(2, -2), `b${i}`)}</strong>;
        if (/^\*[^*\n]+\*$/.test(part) && !part.startsWith("**"))
          return <em key={i} className="italic text-gray-300">{dec(part.slice(1, -1), `i${i}`)}</em>;
        if (/^`[^`]+`$/.test(part))
          return (
            <code key={i} className="px-1.5 py-0.5 rounded-md bg-gray-800 border border-gray-700/60 text-[12px] font-mono text-indigo-300 mx-0.5">
              {dec(part.slice(1, -1), `c${i}`)}
            </code>
          );
        if (/^\$[^$]+\$$/.test(part)) {
          return <InlineMath key={i} expr={part.slice(1, -1)} />;
        }
        // Inline citation links: [1] → indexed corpus paper, [A1] → arXiv candidate
        const citeMatch = part.match(/^\[(A?)(\d+)\]$/);
        if (citeMatch) {
          const isArxiv = citeMatch[1] === "A";
          const num = citeMatch[2];
          const label = isArxiv ? `A${num}` : num;
          if (onCitationClick) {
            const isExt = isArxiv;
            return (
              <span
                key={i}
                role="link"
                tabIndex={0}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onCitationClick(label, isArxiv);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onCitationClick(label, isArxiv);
                  }
                }}
                title={isExt
                  ? `Open arXiv reference ${label} in a new tab`
                  : `View grounded paper ${label}`}
                style={{
                  color: isExt ? "#fbbf24" : "#818cf8",
                  cursor: "pointer",
                  textDecoration: "underline",
                  textDecorationStyle: "dotted",
                  textUnderlineOffset: 2,
                  fontSize: "0.85em",
                  verticalAlign: "super",
                  fontWeight: 600,
                  // A subtle background tint so the citation chip
                  // reads as an actionable link, not a typographic
                  // ornament. The corpus and arXiv colours stay
                  // distinct so users can tell at a glance whether
                  // clicking will open the Paper Panel or arXiv.
                  background: isExt
                    ? "rgba(251,191,36,0.10)"
                    : "rgba(129,140,248,0.10)",
                  border: isExt
                    ? "1px solid rgba(251,191,36,0.28)"
                    : "1px solid rgba(129,140,248,0.28)",
                  borderRadius: 4,
                  padding: "0 3px",
                  margin: "0 1px",
                }}
              >
                [{label}]
              </span>
            );
          }
          return (
            <span key={i} style={{
              color: "#818cf8", fontSize: "0.85em",
              verticalAlign: "super", fontWeight: 500,
            }}>
              [{label}]
            </span>
          );
        }
        // Plain text segment — apply highlight / search decorations as JSX
        // (not via DOM mutation, so React owns the marks and they survive
        // re-renders automatically).
        if (typeof part === "string" && part && decorations) {
          return <Fragment key={i}>{_decorateString(part, decorations, String(i))}</Fragment>;
        }
        return part;
      })}
    </>
  );
}

// Lightweight LaTeX → Unicode rewrite for the most common typographic
// commands. Used as the InlineMath fallback so a brief KaTeX-loading delay
// (or a silent KaTeX failure) shows readable text instead of raw
// "$\rightarrow$" syntax. Extend cautiously — anything that needs real
// math typesetting (fractions, integrals, subscripts) must still go
// through KaTeX.
const _LATEX_TO_UNICODE: Array<[RegExp, string]> = [
  [/\\rightarrow\b/g, "→"],
  [/\\leftarrow\b/g, "←"],
  [/\\Rightarrow\b/g, "⇒"],
  [/\\Leftarrow\b/g, "⇐"],
  [/\\leftrightarrow\b/g, "↔"],
  [/\\Leftrightarrow\b/g, "⇔"],
  [/\\to\b/g, "→"],
  [/\\implies\b/g, "⇒"],
  [/\\iff\b/g, "⇔"],
  [/\\times\b/g, "×"],
  [/\\cdot\b/g, "·"],
  [/\\pm\b/g, "±"],
  [/\\leq\b/g, "≤"],
  [/\\geq\b/g, "≥"],
  [/\\neq\b/g, "≠"],
  [/\\approx\b/g, "≈"],
  [/\\sim\b/g, "∼"],
  [/\\infty\b/g, "∞"],
  [/\\alpha\b/g, "α"],
  [/\\beta\b/g, "β"],
  [/\\gamma\b/g, "γ"],
  [/\\delta\b/g, "δ"],
  [/\\theta\b/g, "θ"],
  [/\\lambda\b/g, "λ"],
  [/\\mu\b/g, "μ"],
  [/\\pi\b/g, "π"],
  [/\\sigma\b/g, "σ"],
  [/\\Sigma\b/g, "Σ"],
  [/\\Omega\b/g, "Ω"],
  [/\\sum\b/g, "∑"],
  [/\\prod\b/g, "∏"],
  // \text{X} → X (the surrounding braces get stripped too)
  [/\\text\{([^}]*)\}/g, "$1"],
  [/\\mathrm\{([^}]*)\}/g, "$1"],
  [/\\mathbf\{([^}]*)\}/g, "$1"],
  [/\\,/g, " "],
];

function latexLikeToUnicode(expr: string): string {
  let out = expr;
  for (const [pat, rep] of _LATEX_TO_UNICODE) out = out.replace(pat, rep);
  return out;
}

function InlineMath({ expr }: { expr: string }) {
  const [html, setHtml] = useState("");
  useEffect(() => {
    import("katex").then((k) => {
      try {
        setHtml(k.default.renderToString(expr, { throwOnError: false }));
      } catch {}
    });
  }, [expr]);
  if (!html) {
    // Fallback: convert the most common LaTeX commands to Unicode so the
    // user sees "→" instead of "$\rightarrow$" while KaTeX is loading or
    // if KaTeX silently fails on this expression.
    return <span className="mx-0.5">{latexLikeToUnicode(expr)}</span>;
  }
  return <span dangerouslySetInnerHTML={{ __html: html }} className="mx-0.5" />;
}

// ── Inline-list normalizer ────────────────────────────────────────────────────

/**
 * LLMs often compress lists onto a single line:
 *   "1. Foo 2. Bar 3. Baz"  →  each item on its own line
 *   "- Foo - Bar - Baz"      →  each item on its own line
 *   "Intro: - Foo - Bar"     →  intro line, then each item
 * This runs before paragraph splitting so the bullet/number renderers
 * see proper line-separated items.
 */
function normalizeInlineLists(raw: string): string {
  return raw.split('\n').map(line => {
    const t = line.trim();
    if (!t) return line;

    // 1. Inline numbered list: starts with "N." and has further "M." items inlined
    //    "1. Good retrieval 2. Incomplete retrieval 3. Conflicting retrieval"
    if (/^\d+\.\s/.test(t)) {
      // Replace "[non-whitespace][spaces][digit+period+space]" with newline boundary
      const split = t.replace(/(\S)\s+(\d+\.\s)/g, '$1\n$2');
      if (split !== t) return split;
    }

    // 2. Inline bullet list: starts with "- " and has more "- " items inlined
    //    "- Answer accuracy: ... - Context compliance: ..."
    if (/^-\s/.test(t) && / - /.test(t)) {
      return t.replace(/\s+-\s+/g, '\n- ');
    }

    // 3. "Intro text: - item1 - item2 - item3" — prose intro followed by inline list
    //    "Measure: - accuracy - compliance - efficiency"
    if (/:\s*-\s/.test(t) && (t.match(/ - /g) || []).length >= 1) {
      const m = t.match(/^(.*?:)\s*-\s/);
      if (m) {
        const intro = m[1];
        const rest = '- ' + t.slice(m[0].length);
        return intro + '\n' + rest.replace(/\s+-\s+/g, '\n- ');
      }
    }

    return line;
  }).join('\n');
}

// ── Full markdown renderer ────────────────────────────────────────────────────

export function renderMarkdown(
  rawContent: string,
  onCitationClick?: (num: string, isArxiv: boolean) => void,
  decorations?: MarkdownDecorations,
): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const IT = ({ text }: { text: string }) => (
    <InlineText text={text} onCitationClick={onCitationClick} decorations={decorations} />
  );

  // Normalize inline lists before any other processing
  const normalised = normalizeInlineLists(rawContent);

  // Normalise LaTeX delimiters → $ syntax
  const content = normalised
    .replace(/\\\[([\s\S]*?)\\\]/g, (_m, e) => `$$${e}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_m, e) => `$${e}$`);

  // 1. Extract code fences with exec loop (robust against varied whitespace/newlines)
  type TextOrCode = { type: "text"; text: string } | { type: "code"; lang: string; code: string };
  const segs: TextOrCode[] = [];
  const fenceRe = /```([\w]*)[ \t]*([\s\S]*?)```/g;
  let last = 0;
  let fm: RegExpExecArray | null;
  while ((fm = fenceRe.exec(content)) !== null) {
    if (fm.index > last) segs.push({ type: "text", text: content.slice(last, fm.index) });
    segs.push({ type: "code", lang: fm[1] || "bash", code: fm[2].trim() });
    last = fm.index + fm[0].length;
  }
  if (last < content.length) segs.push({ type: "text", text: content.slice(last) });

  segs.forEach((seg, ci) => {
    if (seg.type === "code") {
      if (seg.lang === "mermaid") {
        nodes.push(<MermaidBlock key={`mb-${ci}`} spec={seg.code} />);
      } else {
        nodes.push(<CodeBlock key={`cb-${ci}`} lang={seg.lang} code={seg.code} />);
      }
      return;
    }
    const part = seg.text;
    if (!part.trim()) return;

    // 2. Split on display math $$...$$
    const mathParts = part.split(/(\$\$[\s\S]*?\$\$)/g);
    mathParts.forEach((mp, mi) => {
      const mathMatch = mp.match(/^\$\$([\s\S]*?)\$\$$/);
      if (mathMatch) {
        nodes.push(<DisplayMath key={`dm-${ci}-${mi}`} expr={mathMatch[1].trim()} />);
        return;
      }

      if (!mp.trim()) return;

      // 3. Split on paragraphs
      mp.split(/\n\n+/).forEach((para, pi) => {
        if (!para.trim()) return;
        const trimmed = para.trim();
        const key = `p-${ci}-${mi}-${pi}`;

        // Heading — extract first line; render trailing content as a paragraph
        const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)/);
        if (headingMatch) {
          const level = headingMatch[1].length;
          const firstLine = headingMatch[0].split("\n")[0];
          const headText = headingMatch[2].split("\n")[0].trim();
          const afterHead = trimmed.slice(firstLine.length).trim();
          if (level === 1) {
            nodes.push(
              <h1 key={key} className="text-2xl font-bold text-white mt-8 mb-3 tracking-tight">
                <IT text={headText} />
              </h1>
            );
          } else if (level === 2) {
            nodes.push(
              <div key={key} className="mt-10 mb-4 pb-2 border-b border-gray-700/60">
                <h2 className="text-xl font-bold text-white leading-snug tracking-tight"><InlineText text={headText} decorations={decorations} /></h2>
              </div>
            );
          } else if (level === 3) {
            // h3 is a real sub-section heading — was previously styled as a
            // tiny uppercase label which was easy to miss inside a wall of
            // prose. Render it like an actual heading: medium-sized, sentence
            // case, with a thin colour accent so it visually separates groups
            // of paragraphs without competing with h2.
            nodes.push(
              <h3 key={key} className="text-base font-semibold text-gray-100 mt-7 mb-2.5 leading-snug">
                <IT text={headText} />
              </h3>
            );
          } else {
            // h4+ → bold inline-style sub-headings; still distinguishable
            // from regular prose but quieter than h3.
            nodes.push(
              <p key={key} className="text-sm font-semibold text-gray-200 mt-4 mb-1.5">
                <IT text={headText} />
              </p>
            );
          }
          if (afterHead) {
            nodes.push(
              <p key={`${key}-body`} className="text-sm text-gray-300 leading-[1.85] break-words">
                <IT text={afterHead} />
              </p>
            );
          }
          return;
        }

        // Numbered list
        if (/^\d+\.\s/.test(trimmed)) {
          const items = trimmed.split(/\n(?=\d+\.\s)/);
          nodes.push(
            <ol key={key} className="space-y-2 my-2">
              {items.map((item, k) => {
                const m = item.match(/^(\d+)\.\s([\s\S]*)$/);
                const num = m ? m[1] : String(k + 1);
                const text = m ? m[2] : item;
                return (
                  <li key={k} className="flex gap-2.5 items-start">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-gray-800 border border-gray-700/50 text-[10px] font-bold text-gray-400 flex items-center justify-center mt-0.5">
                      {num}
                    </span>
                    <span className="text-sm text-gray-300 leading-relaxed flex-1 min-w-0 break-words">
                      <IT text={text} />
                    </span>
                  </li>
                );
              })}
            </ol>
          );
          return;
        }

        // Bullet list
        if (/^[-•*]\s/.test(trimmed)) {
          const items = trimmed.split(/\n(?=[-•*]\s)/);
          nodes.push(
            <ul key={key} className="space-y-2 my-2">
              {items.map((item, k) => {
                const text = item.replace(/^[-•*]\s/, "");
                return (
                  <li key={k} className="flex gap-2.5 items-start text-sm text-gray-300 leading-relaxed">
                    <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-indigo-400/60 mt-2.5" />
                    <span className="flex-1 min-w-0 break-words"><IT text={text} /></span>
                  </li>
                );
              })}
            </ul>
          );
          return;
        }

        // Blockquote
        if (/^>\s/.test(trimmed)) {
          const bq = trimmed.split("\n").map((l) => l.replace(/^>\s?/, "")).join(" ").trim();
          nodes.push(
            <div key={key} className="border-l-2 border-indigo-500/50 bg-indigo-950/15 rounded-r-lg px-4 py-2.5 my-1">
              <p className="text-sm text-indigo-200/80 leading-relaxed italic">
                <IT text={bq} />
              </p>
            </div>
          );
          return;
        }

        // Table
        if (trimmed.includes("|") && trimmed.split("\n").length >= 2) {
          const rows = trimmed.split("\n").filter((r) => r.trim() && !/^[\s|:-]+$/.test(r));
          if (rows.length >= 2) {
            const parseRow = (r: string) => r.split("|").map((c) => c.trim()).filter(Boolean);
            const [head, ...body] = rows;
            const headers = parseRow(head);
            if (headers.length >= 2) {
              nodes.push(
                <div key={key} className="overflow-x-auto rounded-xl border border-gray-700/50 my-3">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-700/50 bg-gray-800/40">
                        {headers.map((h, hi) => (
                          <th key={hi} className="px-4 py-2.5 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider align-top">
                            <IT text={h} />
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800/60">
                      {body.map((row, ri) => (
                        <tr key={ri} className="hover:bg-gray-800/20 transition-colors">
                          {parseRow(row).map((cell, ci2) => (
                            <td key={ci2} className={`px-4 py-2.5 text-gray-300 align-top break-words ${ci2 === 0 ? "font-medium text-white" : ""}`}>
                              <IT text={cell} />
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
              return;
            }
          }
        }

        // Regular paragraph — generous line-height for readability across long
        // multi-paragraph outputs (Study Mode, Deep Dive). `break-words` keeps
        // long identifiers and URLs from forcing horizontal scroll.
        nodes.push(
          <p key={key} className="text-sm text-gray-300 leading-[1.8] break-words">
            <IT text={trimmed} />
          </p>
        );
      });
    });
  });

  return nodes;
}

function DisplayMath({ expr }: { expr: string }) {
  const [html, setHtml] = useState("");
  useEffect(() => {
    import("katex").then((k) => {
      try {
        setHtml(k.default.renderToString(expr, { displayMode: true, throwOnError: false }));
      } catch {}
    });
  }, [expr]);
  if (!html) return (
    <pre className="font-mono text-sm my-3 px-4 py-2 rounded-lg border overflow-x-auto" style={{ background: "#161b22", color: "#e3b341", borderColor: "rgba(255,255,255,0.08)" }} data-code-block>
      {`$$${expr}$$`}
    </pre>
  );
  return (
    <div
      dangerouslySetInnerHTML={{ __html: html }}
      className="my-4 px-4 py-3 rounded-xl bg-gray-900/60 border border-gray-700/40 overflow-x-auto text-center"
    />
  );
}

// ── Default export — drop-in renderer ────────────────────────────────────────

export default function MarkdownRenderer({
  content,
  className = "",
  onCitationClick,
  decorations,
}: {
  content: string;
  className?: string;
  onCitationClick?: (num: string, isArxiv: boolean) => void;
  /**
   * Optional decoration pipeline — supply ``highlights`` and/or ``searchQuery``
   * to have matches rendered as ``<mark>`` React elements that survive
   * arbitrary re-renders (streaming, polling, parent state updates).
   */
  decorations?: MarkdownDecorations;
}) {
  const nodes = renderMarkdown(content, onCitationClick, decorations);
  return (
    <DecorationsProvider value={decorations}>
      <div className={`space-y-3 min-w-0 ${className}`}>
        {nodes}
      </div>
    </DecorationsProvider>
  );
}
