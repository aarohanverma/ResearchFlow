"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import katex from "katex";
import {
  ArrowLeftIcon,
  BotIcon,
  UserIcon,
  SendIcon,
  XIcon,
  Loader2Icon,
  MessageSquareIcon,
  CopyIcon,
  CheckIcon,
  ZapIcon,
  FlaskConicalIcon,
  SearchIcon,
  LightbulbIcon,
  AlertTriangleIcon,
  HelpCircleIcon,
  TargetIcon,
  LinkIcon,
  CodeIcon,
  SparklesIcon,
  FileTextIcon,
  GitMergeIcon,
} from "lucide-react";
import type { IdeaCapsule, DiagramSpec, GeneratedArtifact, GenerationType } from "@/types";
import MarkdownRenderer, {
  sanitizeMermaidSpec,
  sanitizeMermaidAggressive,
  DecorationsProvider,
  decorateString,
  useDecorations,
} from "@/components/ui/MarkdownRenderer";
import { useHighlightSearch, HighlightSearchToolbar } from "@/components/ui/HighlightSearch";
import { PaperPanel } from "@/components/paper/PaperPanel";
import type { Paper } from "@/types";
import { useFeature } from "@/lib/features";
import { SectionNavPanel } from "@/components/ui/SectionNavPanel";
import { useAuthStore } from "@/store/auth";
import { api } from "@/lib/api";
import { useJobsStore } from "@/store/jobs";
import { RefreshCwIcon } from "lucide-react";

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
          langs: ["python", "javascript", "typescript", "bash", "rust", "go", "java", "cpp", "sql", "json"],
        })
      )
      .then((h) => { _highlighter = h; return h; });
  }
  return _highlighterPromise;
}

// ── Code block ────────────────────────────────────────────────────────────────

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [html, setHtml] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getHighlighter().then((h) => {
      if (cancelled) return;
      try {
        setHtml(h.codeToHtml(code, { lang: lang || "bash", theme: "github-dark-dimmed" }));
      } catch {
        try { setHtml(h.codeToHtml(code, { lang: "bash", theme: "github-dark-dimmed" })); } catch { setHtml(null); }
      }
    });
    return () => { cancelled = true; };
  }, [code, lang]);

  function copy() {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div data-code-block className="rounded-2xl overflow-hidden border border-gray-800/60 bg-[#0d0f12] shadow-xl shadow-black/50 my-1">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800/50">
        <div className="flex items-center gap-2.5">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/50" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/50" />
            <div className="w-3 h-3 rounded-full bg-green-500/50" />
          </div>
          <span className="text-sm font-semibold text-gray-300 font-mono">
            {lang ? lang.charAt(0).toUpperCase() + lang.slice(1) : "Code"}
          </span>
        </div>
        <button
          onClick={copy}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gray-800/80 border border-gray-700/50 text-xs font-semibold text-gray-300 hover:text-white hover:bg-gray-700 transition-all"
        >
          {copied ? (
            <><CheckIcon size={12} className="text-emerald-400" /><span className="text-emerald-400">Copied!</span></>
          ) : (
            <><CopyIcon size={12} />Copy</>
          )}
        </button>
      </div>
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

// ── Mermaid diagram ───────────────────────────────────────────────────────────

function MermaidDiagram({ spec }: { spec: string }) {
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
            primaryColor: "#2e1065",
            primaryTextColor: "#ddd6fe",
            primaryBorderColor: "#7c3aed",
            lineColor: "#8b5cf6",
            secondaryColor: "#1e1b4b",
            tertiaryColor: "#1a1635",
            edgeLabelBackground: "#2e1065",
            nodeTextColor: "#ede9fe",
            fontSize: "13px",
            fontFamily: "ui-monospace, SFMono-Regular, monospace",
          },
          flowchart: {
            htmlLabels: true,
            curve: "basis",
            padding: 20,
            useMaxWidth: true,
            wrappingWidth: 240,
            nodeSpacing: 50,
            rankSpacing: 60,
          },
          maxTextSize: 200000,
          securityLevel: "loose",
          suppressErrorRendering: true,
        });
        try { (mermaid as unknown as { parseError?: (...a: unknown[]) => void }).parseError = () => {}; } catch {}
        const tryRender = async (source: string) => {
          const ok = await mermaid.parse(source, { suppressErrors: true });
          if (ok === false) throw new Error("mermaid parse failed");
          const id = `mermaid-${Math.random().toString(36).slice(2)}`;
          return mermaid.render(id, source);
        };
        let svg: string | null = null;
        try {
          ({ svg } = await tryRender(sanitizeMermaidSpec(spec)));
        } catch {
          try {
            ({ svg } = await tryRender(sanitizeMermaidAggressive(spec)));
          } catch {
            svg = null;
          }
        }
        try {
          document.querySelectorAll('svg[aria-roledescription="error"], #mermaid-error-icon').forEach(n => n.remove());
        } catch {}
        if (cancelled) return;
        if (!svg) {
          setError(true);
          if (ref.current) ref.current.innerHTML = "";
          return;
        }
        if (!ref.current) return;
        ref.current.innerHTML = svg;
        const svgEl = ref.current.querySelector("svg");
        if (!svgEl) return;
        // Let SVG scale to its natural viewBox aspect ratio: capping height
        // squishes nodes and truncates labels. ``useMaxWidth: true`` already
        // makes mermaid emit a fully responsive SVG; we only need to ensure
        // it actually fills the container width and keeps its own aspect.
        svgEl.removeAttribute("height");
        svgEl.setAttribute("width", "100%");
        svgEl.setAttribute("preserveAspectRatio", "xMidYMid meet");
        svgEl.style.cssText = "width:100%;height:auto;max-width:100%;display:block;";
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
      <pre className="overflow-x-auto text-xs text-gray-500 font-mono leading-relaxed p-4 bg-gray-950/60 rounded-lg border border-gray-800/50">
        {spec}
      </pre>
    );
  }
  return <div ref={ref} className="overflow-x-auto w-full" />;
}

// ── Inline text renderer ──────────────────────────────────────────────────────

// Citation handler context — wired from the IdeaDetail page so any [N]
// reference inside the deep-dive markdown opens the same paper preview
// overlay used by the Source Papers section.
const CitationCtx = React.createContext<((n: string) => void) | null>(null);

function renderInlineWithCitations(text: string, onClick: ((n: string) => void) | null, keyPrefix: string): React.ReactNode {
  if (!onClick) return text;
  const out: React.ReactNode[] = [];
  const re = /\[(\d+(?:\s*[-,]\s*\d+)*)\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const inner = m[1];
    const nums = inner.split(/\s*,\s*/).flatMap((part) => {
      const r = part.match(/^(\d+)\s*-\s*(\d+)$/);
      if (r) {
        const a = parseInt(r[1], 10), b = parseInt(r[2], 10);
        if (a <= b && b - a < 50) return Array.from({ length: b - a + 1 }, (_, k) => String(a + k));
      }
      return [part.trim()];
    });
    out.push(
      <span key={`${keyPrefix}-cit-${i++}`} className="inline-flex gap-0.5 align-baseline mx-0.5">
        {nums.map((n, j) => (
          <button
            key={j}
            type="button"
            onClick={() => onClick(n)}
            className="text-[11px] leading-none px-1 py-0.5 rounded bg-indigo-900/40 border border-indigo-700/40 text-indigo-300 hover:bg-indigo-800/60 hover:text-indigo-200 transition-colors"
            title="Open source paper"
          >
            [{n}]
          </button>
        ))}
      </span>,
    );
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function InlineText({ text }: { text: string }) {
  // Decorations come from React context so the page-wide highlight + search
  // hook reaches every span — bold, italic, code, headings, list items,
  // callouts, table cells — without prop-drilling.
  const dec = useDecorations();
  const onCite = React.useContext(CitationCtx);
  const wrap = (s: string, k: string): React.ReactNode => {
    const cited = onCite ? renderInlineWithCitations(s, onCite, k) : s;
    if (!dec) return cited;
    // ``decorateString`` operates on strings, not React nodes, so only
    // apply it when there are no citation buttons to preserve.
    if (typeof cited === "string") {
      return <>{decorateString(cited, dec, k)}</>;
    }
    return cited;
  };
  const normalised = text.replace(/\\\(([\s\S]+?)\\\)/g, (_m, e) => `$${e}$`);
  // Allow up to 600 chars and across newlines so long equations render correctly
  const parts = normalised.split(
    /(\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`|\$[^$]{1,600}\$)/g
  );
  return (
    <>
      {parts.map((part, i) => {
        if (/^\*\*[^*]+\*\*$/.test(part))
          return <strong key={i} className="font-semibold text-white">{wrap(part.slice(2, -2), `b${i}`)}</strong>;
        if (/^\*[^*\n]+\*$/.test(part) && !part.startsWith("**"))
          return <em key={i} className="italic text-gray-200">{wrap(part.slice(1, -1), `i${i}`)}</em>;
        if (/^`[^`]+`$/.test(part))
          return <code key={i} className="px-1.5 py-0.5 rounded-md bg-gray-800 border border-gray-700/60 text-[12px] font-mono text-violet-300 mx-0.5">{wrap(part.slice(1, -1), `c${i}`)}</code>;
        if (/^\$[^$]+\$$/.test(part)) {
          try {
            return <span key={i} dangerouslySetInnerHTML={{ __html: katex.renderToString(part.slice(1, -1), { throwOnError: false }) }} className="mx-0.5" />;
          } catch { return <span key={i}>{part}</span>; }
        }
        return <React.Fragment key={i}>{wrap(part, `t${i}`)}</React.Fragment>;
      })}
    </>
  );
}

// ── Callout styles ────────────────────────────────────────────────────────────

const CALLOUT_STYLES: Record<string, { bg: string; border: string; iconBg: string; labelColor: string; label: string }> = {
  "💡": { bg: "bg-amber-950/25",   border: "border-amber-700/40",   iconBg: "bg-amber-900/60",   labelColor: "text-amber-400",   label: "Key Insight" },
  "💬": { bg: "bg-violet-950/25",  border: "border-violet-700/40",  iconBg: "bg-violet-900/60",  labelColor: "text-violet-400",  label: "Analogy" },
  "🔧": { bg: "bg-orange-950/25",  border: "border-orange-700/40",  iconBg: "bg-orange-900/60",  labelColor: "text-orange-400",  label: "In Practice" },
  "📊": { bg: "bg-emerald-950/25", border: "border-emerald-700/40", iconBg: "bg-emerald-900/60", labelColor: "text-emerald-400", label: "By the Numbers" },
  "⚠️": { bg: "bg-red-950/25",     border: "border-red-700/40",     iconBg: "bg-red-900/60",     labelColor: "text-red-400",     label: "Heads Up" },
  "🎯": { bg: "bg-indigo-950/25",  border: "border-indigo-700/40",  iconBg: "bg-indigo-900/60",  labelColor: "text-indigo-400",  label: "Why It Matters" },
  "🔬": { bg: "bg-teal-950/25",    border: "border-teal-700/40",    iconBg: "bg-teal-900/60",    labelColor: "text-teal-400",    label: "Fun Fact" },
  "🤔": { bg: "bg-purple-950/25",  border: "border-purple-700/40",  iconBg: "bg-purple-900/60",  labelColor: "text-purple-400",  label: "Think About This" },
  "✨": { bg: "bg-pink-950/25",    border: "border-pink-700/40",    iconBg: "bg-pink-900/60",    labelColor: "text-pink-400",    label: "Highlight" },
  "🚀": { bg: "bg-sky-950/25",     border: "border-sky-700/40",     iconBg: "bg-sky-900/60",     labelColor: "text-sky-400",     label: "Impact" },
  "📝": { bg: "bg-gray-800/40",    border: "border-gray-700/40",    iconBg: "bg-gray-700/60",    labelColor: "text-gray-400",    label: "Note" },
  "⭐": { bg: "bg-yellow-950/25",  border: "border-yellow-700/40",  iconBg: "bg-yellow-900/60",  labelColor: "text-yellow-400",  label: "Standout" },
  "🧠": { bg: "bg-cyan-950/25",    border: "border-cyan-700/40",    iconBg: "bg-cyan-900/60",    labelColor: "text-cyan-400",    label: "Mental Model" },
  "🎪": { bg: "bg-rose-950/25",    border: "border-rose-700/40",    iconBg: "bg-rose-900/60",    labelColor: "text-rose-400",    label: "The Clever Part" },
};

// ── Content segmenter ─────────────────────────────────────────────────────────

type Segment = { type: "code"; lang: string; code: string } | { type: "text"; text: string };

function segmentContent(raw: string): Segment[] {
  const s = raw
    .replace(/\\\[([\s\S]*?)\\\]/g, (_m, e) => `$$${e}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_m, e) => `$${e}$`);
  const segments: Segment[] = [];
  const fenceRe = /```([\w]*)[ \t]*([\s\S]*?)```/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = fenceRe.exec(s)) !== null) {
    if (m.index > last) segments.push({ type: "text", text: s.slice(last, m.index) });
    segments.push({ type: "code", lang: m[1] || "bash", code: m[2].trim() });
    last = m.index + m[0].length;
  }
  if (last < s.length) segments.push({ type: "text", text: s.slice(last) });
  return segments;
}

// ── Rich content renderer ─────────────────────────────────────────────────────

function renderContent(rawContent: string) {
  const nodes: React.ReactNode[] = [];
  const segments = segmentContent(rawContent);

  segments.forEach((seg, ci) => {
    if (seg.type === "code") {
      if (seg.lang === "mermaid") {
        nodes.push(<MermaidDiagram key={`mermaid-${ci}`} spec={seg.code} />);
      } else {
        nodes.push(<CodeBlock key={`code-${ci}`} lang={seg.lang} code={seg.code} />);
      }
      return;
    }
    const part = seg.text;
    if (!part.trim()) return;

    const mathParts = part.split(/(\$\$[\s\S]*?\$\$)/g);
    mathParts.forEach((mp, mi) => {
      const displayMathMatch = mp.match(/^\$\$([\s\S]*?)\$\$$/);
      if (displayMathMatch) {
        const expr = displayMathMatch[1].trim();
        try {
          nodes.push(
            <div key={`math-${ci}-${mi}`}
              className="my-4 px-4 py-3 rounded-xl bg-gray-900/60 border border-gray-700/40 overflow-x-auto text-center"
              dangerouslySetInnerHTML={{ __html: katex.renderToString(expr, { displayMode: true, throwOnError: false }) }}
            />
          );
        } catch {
          nodes.push(<p key={`math-${ci}-${mi}`} className="text-gray-400 font-mono text-sm">{mp}</p>);
        }
        return;
      }
      if (!mp.trim()) return;
      mp.split(/\n\n+/).forEach((para, pi) => {
        if (!para.trim()) return;
        const trimmed = para.trim();

        // Headings (handle up to 6 levels; 4+ treated as h3)
        const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)/);
        if (headingMatch) {
          const level = Math.min(headingMatch[1].length, 3);
          const firstLine = headingMatch[0].split("\n")[0];
          const headText = headingMatch[2].split("\n")[0].trim();
          const afterHead = trimmed.slice(firstLine.length).trim();
          if (level === 1) {
            nodes.push(
              <h1 key={`h-${ci}-${mi}-${pi}`} className="text-2xl font-bold text-white mt-8 mb-3">
                <InlineText text={headText} />
              </h1>
            );
          } else if (level === 2) {
            nodes.push(
              <div key={`h-${ci}-${mi}-${pi}`} className="mt-10 mb-4 pb-2 border-b border-gray-700/60">
                <h2 className="text-xl font-bold text-white leading-snug"><InlineText text={headText} /></h2>
              </div>
            );
          } else {
            nodes.push(
              <h3 key={`h-${ci}-${mi}-${pi}`} className="text-sm font-semibold text-gray-300 uppercase tracking-widest mt-6 mb-2">
                <InlineText text={headText} />
              </h3>
            );
          }
          if (afterHead) nodes.push(<p key={`h-body-${ci}-${mi}-${pi}`} className="text-gray-300 text-sm leading-[1.9]"><InlineText text={afterHead} /></p>);
          return;
        }

        // Numbered list
        if (/^\d+\.\s/.test(trimmed)) {
          const items = trimmed.split(/\n(?=\d+\.\s)/);
          nodes.push(
            <ol key={`ol-${ci}-${mi}-${pi}`} className="space-y-2">
              {items.map((item, k) => {
                const numMatch = item.match(/^(\d+)\.\s([\s\S]*)$/);
                const num = numMatch ? numMatch[1] : String(k + 1);
                const text = numMatch ? numMatch[2] : item;
                return (
                  <li key={k} className="flex gap-3 items-start">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-gray-800 border border-gray-700/50 text-[11px] font-bold text-gray-400 flex items-center justify-center mt-0.5">{num}</span>
                    <span className="text-sm text-gray-300 leading-relaxed flex-1"><InlineText text={text} /></span>
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
            <ul key={`ul-${ci}-${mi}-${pi}`} className="space-y-2">
              {items.map((item, k) => (
                <li key={k} className="flex gap-3 items-start text-sm text-gray-300 leading-relaxed">
                  <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-violet-500/60 mt-2.5" />
                  <span className="flex-1"><InlineText text={item.replace(/^[-•*]\s/, "")} /></span>
                </li>
              ))}
            </ul>
          );
          return;
        }

        // Blockquote / callout — matches both `> 💡 text` and bare `💡 text`
        const CALLOUT_EMOJI_RE = /^(💡|💬|🔧|📊|⚠️|🎯|🔬|🤔|✨|🚀|📝|⭐|🧠|🎪)/;
        const isBlockquote = /^>\s/.test(trimmed);
        const bareCallout = !isBlockquote && CALLOUT_EMOJI_RE.test(trimmed);
        if (isBlockquote || bareCallout) {
          const bqLines = isBlockquote
            ? trimmed.split("\n").map(l => l.replace(/^>\s?/, "")).join(" ").trim()
            : trimmed.split("\n").join(" ").trim();
          const calloutMatch = bqLines.match(/^(💡|💬|🔧|📊|⚠️|🎯|🔬|🤔|✨|🚀|📝|⭐|🧠|🎪)\s*([\s\S]*)/);
          if (calloutMatch) {
            const emoji = calloutMatch[1];
            const body = calloutMatch[2].trim();
            const s = CALLOUT_STYLES[emoji] || CALLOUT_STYLES["💡"];
            nodes.push(
              <div key={`callout-${ci}-${mi}-${pi}`} className={`flex gap-3.5 rounded-2xl border ${s.border} ${s.bg} p-4`}>
                <div className={`flex-shrink-0 w-9 h-9 rounded-xl ${s.iconBg} flex items-center justify-center text-lg`}>{emoji}</div>
                <div className="flex-1 min-w-0">
                  <p className={`text-[10px] font-bold uppercase tracking-widest ${s.labelColor} mb-1.5`}>{s.label}</p>
                  <p className="text-sm text-gray-200 leading-relaxed"><InlineText text={body} /></p>
                </div>
              </div>
            );
          } else if (isBlockquote) {
            nodes.push(
              <div key={`bq-${ci}-${mi}-${pi}`} className="border-l-2 border-violet-500/40 bg-violet-950/15 rounded-r-xl px-4 py-3">
                <p className="text-sm text-violet-200/80 leading-relaxed italic"><InlineText text={bqLines} /></p>
              </div>
            );
          } else {
            nodes.push(<p key={`p-${ci}-${mi}-${pi}`} className="text-gray-300 text-sm leading-[1.9]"><InlineText text={trimmed} /></p>);
          }
          return;
        }

        // Table
        if (trimmed.includes("|") && trimmed.split("\n").length >= 2) {
          const rows = trimmed.split("\n").filter(r => r.trim() && !/^[\s|:-]+$/.test(r));
          if (rows.length >= 2) {
            const parseRow = (r: string) => r.split("|").map(c => c.trim()).filter(Boolean);
            const [head, ...body] = rows;
            const headers = parseRow(head);
            if (headers.length >= 2) {
              nodes.push(
                <div key={`tbl-${ci}-${mi}-${pi}`} className="overflow-x-auto rounded-xl border border-gray-700/50 my-1">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-700/50 bg-gray-800/40">
                        {headers.map((h, hi) => (
                          <th key={hi} className="px-4 py-2.5 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap"><InlineText text={h} /></th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800/60">
                      {body.map((row, ri) => (
                        <tr key={ri} className="hover:bg-gray-800/20 transition-colors">
                          {parseRow(row).map((cell, ci2) => (
                            <td key={ci2} className={`px-4 py-2.5 text-gray-300 ${ci2 === 0 ? "font-medium text-white" : ""}`}><InlineText text={cell} /></td>
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

        // Regular paragraph
        nodes.push(
          <p key={`p-${ci}-${mi}-${pi}`} className="text-gray-300 text-sm leading-[1.9]">
            <InlineText text={trimmed} />
          </p>
        );
      });
    });
  });

  return nodes;
}

// ── Diagram section ───────────────────────────────────────────────────────────

function stripMermaidFences(raw: string): string {
  return raw
    .replace(/^```(?:mermaid)?\s*\n?/i, "")
    .replace(/\n?```\s*$/i, "")
    .trim();
}

function DiagramSection({ diagram, index }: { diagram: DiagramSpec; index: number }) {
  const cleanSpec = diagram.spec ? stripMermaidFences(diagram.spec) : undefined;
  const typeLabel = diagram.type === "hero_image" ? "AI Visualization" : diagram.type.replace("_", " ");
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      className="relative rounded-2xl overflow-hidden shadow-xl shadow-violet-500/10"
    >
      <div className="absolute inset-0 rounded-2xl border border-violet-500/30 z-10 pointer-events-none" />
      <div className="absolute -inset-0.5 rounded-2xl bg-gradient-to-br from-violet-500/20 via-purple-500/10 to-indigo-500/20 blur-sm -z-10" />
      <div className="absolute inset-0 bg-[#0a0612]" />
      <div className="absolute inset-0 opacity-20" style={{ backgroundImage: `radial-gradient(circle at 1px 1px, rgba(139,92,246,0.2) 1px, transparent 0)`, backgroundSize: "28px 28px" }} />
      <div className="relative flex items-center gap-3 px-5 py-3 border-b border-violet-800/30 bg-violet-950/40 backdrop-blur-sm z-10">
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
          <div className="w-2.5 h-2.5 rounded-full bg-amber-500/60" />
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/60" />
        </div>
        <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
        <span className="text-xs font-semibold uppercase tracking-widest text-violet-300">{typeLabel}</span>
      </div>
      <div className="relative p-6 z-10">
        {cleanSpec && <MermaidDiagram spec={cleanSpec} />}
        {diagram.blob_path && (
          <img
            src={`/blobs/${diagram.blob_path}`}
            alt="AI-generated research visualization"
            className="w-full rounded-xl border border-violet-800/20 shadow-lg"
          />
        )}
      </div>
    </motion.div>
  );
}

// ── Section definitions ───────────────────────────────────────────────────────

interface SectionDef {
  key: keyof IdeaCapsule;
  label: string;
  subtitle: string;
  emoji: string;
  icon: React.ReactNode;
  colorClasses: { text: string; border: string; bg: string; iconBg: string; accent: string };
}

const SECTIONS: SectionDef[] = [
  {
    key: "hypothesis",
    label: "Hypothesis",
    subtitle: "The core scientific claim",
    emoji: "💡",
    icon: <LightbulbIcon size={22} />,
    colorClasses: { text: "text-violet-300", border: "border-violet-800/50", bg: "bg-violet-950/20", iconBg: "bg-violet-950/60 border-violet-800/60", accent: "from-violet-500 to-purple-600" },
  },
  {
    key: "rationale",
    label: "Rationale",
    subtitle: "Why this idea is worth pursuing — the scientific motivation",
    emoji: "🎯",
    icon: <TargetIcon size={22} />,
    colorClasses: { text: "text-sky-300", border: "border-sky-800/50", bg: "bg-sky-950/20", iconBg: "bg-sky-950/60 border-sky-800/60", accent: "from-sky-500 to-blue-600" },
  },
  {
    key: "mechanism",
    label: "Mechanism",
    subtitle: "How it works — the technical approach and causal chain",
    emoji: "⚡",
    icon: <ZapIcon size={22} />,
    colorClasses: { text: "text-teal-300", border: "border-teal-800/50", bg: "bg-teal-950/20", iconBg: "bg-teal-950/60 border-teal-800/60", accent: "from-teal-400 to-cyan-500" },
  },
  {
    key: "experimental_design",
    label: "Experimental Design",
    subtitle: "Concrete protocol to test the hypothesis",
    emoji: "🧪",
    icon: <FlaskConicalIcon size={22} />,
    colorClasses: { text: "text-amber-300", border: "border-amber-800/50", bg: "bg-amber-950/20", iconBg: "bg-amber-950/60 border-amber-800/60", accent: "from-amber-400 to-orange-500" },
  },
  {
    key: "predicted_outcome",
    label: "Predicted Outcomes",
    subtitle: "What success looks like — numeric targets and signals",
    emoji: "📊",
    icon: <TargetIcon size={22} />,
    colorClasses: { text: "text-emerald-300", border: "border-emerald-800/50", bg: "bg-emerald-950/20", iconBg: "bg-emerald-950/60 border-emerald-800/60", accent: "from-emerald-400 to-teal-500" },
  },
  {
    key: "anti_finding",
    label: "Anti-Finding",
    subtitle: "What would falsify or kill this idea",
    emoji: "⚠️",
    icon: <AlertTriangleIcon size={22} />,
    colorClasses: { text: "text-rose-300", border: "border-rose-800/50", bg: "bg-rose-950/20", iconBg: "bg-rose-950/60 border-rose-800/60", accent: "from-rose-500 to-pink-600" },
  },
  {
    key: "risks_and_limitations",
    label: "Risks & Limitations",
    subtitle: "Known failure modes, constraints, and boundary conditions",
    emoji: "🔶",
    icon: <AlertTriangleIcon size={22} />,
    colorClasses: { text: "text-orange-300", border: "border-orange-800/50", bg: "bg-orange-950/20", iconBg: "bg-orange-950/60 border-orange-800/60", accent: "from-orange-500 to-amber-600" },
  },
  {
    key: "open_questions",
    label: "Open Questions",
    subtitle: "What still needs to be figured out to ship this",
    emoji: "❓",
    icon: <HelpCircleIcon size={22} />,
    colorClasses: { text: "text-purple-300", border: "border-purple-800/50", bg: "bg-purple-950/20", iconBg: "bg-purple-950/60 border-purple-800/60", accent: "from-purple-500 to-violet-600" },
  },
];

// ── Capsule hero ──────────────────────────────────────────────────────────────

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(value * 100);
  const barColors: Record<string, string> = { violet: "bg-violet-500", emerald: "bg-emerald-500", sky: "bg-sky-500" };
  const textColors: Record<string, string> = { violet: "text-violet-400", emerald: "text-emerald-400", sky: "text-sky-400" };
  return (
    <div className="flex items-center gap-2.5">
      <span className="text-[11px] text-gray-500 w-20 flex-shrink-0">{label}</span>
      <div className="h-1.5 flex-1 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
          className={`h-full rounded-full ${barColors[color]}`}
        />
      </div>
      <span className={`text-[11px] font-mono font-semibold w-7 text-right ${textColors[color]}`}>{pct}</span>
    </div>
  );
}

function CapsuleHero({ capsule }: { capsule: IdeaCapsule }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="mb-10 pb-8 border-b border-gray-800/50"
    >
      <div className="flex items-center gap-2 mb-5 flex-wrap">
        {/* Origin mode tag */}
        {capsule.source_mode === "combined" ? (
          <span className="px-3 py-1 rounded-full bg-fuchsia-950/50 border border-fuchsia-700/30 text-fuchsia-400 text-[11px] font-semibold flex items-center gap-1.5">
            <GitMergeIcon size={10} />Combined
          </span>
        ) : capsule.source_mode === "query" ? (
          <span className="px-3 py-1 rounded-full bg-violet-950/50 border border-violet-700/30 text-violet-400 text-[11px] font-semibold flex items-center gap-1.5">
            <SearchIcon size={10} />Query
          </span>
        ) : capsule.source_mode === "auto" || capsule.is_scout_generated ? (
          <span className="px-3 py-1 rounded-full bg-indigo-950/50 border border-indigo-700/30 text-indigo-400 text-[11px] font-semibold flex items-center gap-1.5">
            <ZapIcon size={10} />Auto
          </span>
        ) : (
          <span className="px-3 py-1 rounded-full bg-gray-800/50 border border-gray-700/40 text-gray-500 text-[11px] font-semibold flex items-center gap-1.5">
            <FlaskConicalIcon size={10} />Manual
          </span>
        )}
        <span className={`px-3 py-1 rounded-full text-[11px] font-semibold ${capsule.status === "saved" ? "bg-emerald-950/50 border border-emerald-700/30 text-emerald-400" : "bg-gray-800/50 border border-gray-700/40 text-gray-500"}`}>
          {capsule.status}
        </span>
        <span className="px-3 py-1 rounded-full bg-gray-800/50 border border-gray-700/40 text-gray-500 text-[11px]">
          {new Date(capsule.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
        </span>
      </div>

      <h1 className="text-3xl font-bold text-white leading-tight tracking-tight mb-3">{capsule.title}</h1>
      {capsule.source_mode === "query" && capsule.source_query && (
        <p className="text-sm text-violet-400/70 italic mb-6 flex items-center gap-1.5">
          <SearchIcon size={12} className="flex-shrink-0" />
          &ldquo;{capsule.source_query}&rdquo;
        </p>
      )}
      {capsule.source_mode === "combined" && capsule.source_query && (
        <p className="text-sm text-fuchsia-400/80 mb-6 flex items-center gap-1.5">
          <GitMergeIcon size={12} className="flex-shrink-0" />
          <span className="italic">{capsule.source_query}</span>
        </p>
      )}

      <div className="rounded-2xl bg-gray-900/40 border border-gray-800/50 px-5 py-4 space-y-3">
        <p className="text-[10px] font-bold uppercase tracking-widest text-gray-600 mb-3.5">Quality Scores</p>
        <ScoreBar label="Novelty" value={capsule.novelty_score} color="violet" />
        <ScoreBar label="Feasibility" value={capsule.feasibility_score} color="emerald" />
        <ScoreBar label="Impact" value={capsule.impact_score} color="sky" />
      </div>
    </motion.div>
  );
}

// ── Section block with rich rendering ────────────────────────────────────────

function SectionBlock({ def, content, index }: { def: SectionDef; content: string; index: number }) {
  const { label, subtitle, icon, colorClasses } = def;
  return (
    <motion.div
      id={`genie-section-${def.key}`}
      initial={{ opacity: 0, y: 28 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1], delay: index * 0.05 }}
      className="scroll-mt-6"
    >
      <div className="flex items-center gap-4 mb-5">
        <div className={`relative flex items-center justify-center w-14 h-14 rounded-2xl border ${colorClasses.iconBg} flex-shrink-0 shadow-xl overflow-hidden`}>
          <div className={`absolute inset-0 bg-gradient-to-br ${colorClasses.accent} opacity-15`} />
          <span className={`relative z-10 ${colorClasses.text}`}>{icon}</span>
        </div>
        <div>
          <h2 className={`text-2xl font-bold ${colorClasses.text} tracking-tight leading-tight`}>{label}</h2>
          <p className="text-xs text-gray-600 mt-0.5">{subtitle}</p>
          <div className={`h-0.5 mt-2 w-16 rounded-full bg-gradient-to-r ${colorClasses.accent}`} />
        </div>
      </div>

      <div className={`relative rounded-2xl border ${colorClasses.border} ${colorClasses.bg} px-7 py-6 shadow-lg`}>
        <div className={`absolute left-0 top-8 bottom-8 w-0.5 rounded-full bg-gradient-to-b ${colorClasses.accent} opacity-50`} />
        <div className="pl-4 space-y-4">
          {renderContent(content)}
        </div>
      </div>
    </motion.div>
  );
}

// ── Chat panel ────────────────────────────────────────────────────────────────

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

function IdeaChatPanel({
  capsuleId,
  onClose,
  onCitationClick,
}: {
  capsuleId: string;
  onClose: () => void;
  onCitationClick?: (num: string, isArxiv: boolean) => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: "I have this research idea fully loaded. Ask me anything — how to test it, what could go wrong, how to extend it, related work, or how to pitch it." },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  useEffect(() => () => abortRef.current?.abort(), []);

  async function sendMessage() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    const history = messages.filter(m => !m.streaming).slice(-10).map(m => ({ role: m.role, content: m.content }));
    setMessages(prev => [...prev, { role: "user", content: text }, { role: "assistant", content: "", streaming: true }]);
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const token = (() => { try { return JSON.parse(localStorage.getItem("rf_auth") || "{}").state?.token || ""; } catch { return ""; } })();
      const resp = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/genie/capsules/${capsuleId}/chat`,
        { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` }, body: JSON.stringify({ message: text, history }), signal: ctrl.signal }
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
            // Backend emits {type: 'chunk', content: token}, {type: 'done'}, {type: 'error', message}
            if (p.type === "chunk" && p.content) {
              acc += p.content;
              setMessages(prev => [...prev.slice(0, -1), { role: "assistant", content: acc, streaming: true }]);
            } else if (p.type === "done") {
              setMessages(prev => [...prev.slice(0, -1), { role: "assistant", content: acc || "(empty response)" }]);
            } else if (p.type === "error") {
              setMessages(prev => [...prev.slice(0, -1), { role: "assistant", content: `Error: ${p.message || "unknown"}` }]);
            }
          } catch {}
        }
      }
    } catch (err) {
      if ((err as { name?: string })?.name !== "AbortError") {
        setMessages(prev => [...prev.slice(0, -1), { role: "assistant", content: "Something went wrong. Try again." }]);
      }
    }
    setBusy(false);
    inputRef.current?.focus();
  }

  // Resizable side-panel width — persists via localStorage so the user's
  // preferred width survives navigation / reload.
  const [panelW, setPanelW] = useState<number>(() => {
    if (typeof window === "undefined") return 420;
    try {
      const saved = parseInt(localStorage.getItem("rf_idea_qa_w") || "", 10);
      if (Number.isFinite(saved) && saved >= 320 && saved <= 1100) return saved;
    } catch {}
    return 420;
  });
  const startResize = (e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelW;
    const onMove = (ev: MouseEvent) => {
      const dx = startX - ev.clientX;
      const w = Math.min(1100, Math.max(320, startW + dx));
      setPanelW(w);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      try { localStorage.setItem("rf_idea_qa_w", String(panelW)); } catch {}
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };
  useEffect(() => {
    try { localStorage.setItem("rf_idea_qa_w", String(panelW)); } catch {}
  }, [panelW]);

  return (
    <motion.div
      initial={{ x: "100%", opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0 }}
      transition={{ type: "spring", damping: 30, stiffness: 340 }}
      className="fixed right-0 top-0 bottom-0 bg-gray-950 border-l border-gray-800/70 flex flex-col z-40 shadow-2xl shadow-black/60"
      style={{ width: `${panelW}px` }}
    >
      <div
        onMouseDown={startResize}
        className="absolute left-0 top-0 bottom-0 w-1 cursor-ew-resize hover:bg-violet-500/30 transition-colors z-20"
        title="Drag to resize"
      />
      <div className="flex items-center justify-between px-4 py-3.5 border-b border-gray-800/60 bg-gray-950/95 backdrop-blur-sm">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-violet-600/20 border border-violet-500/30 flex items-center justify-center">
            <BotIcon size={13} className="text-violet-400" />
          </div>
          <div>
            <p className="text-xs font-semibold text-white">Idea Q&A</p>
            <p className="text-[10px] text-gray-600">Grounded in this research capsule</p>
          </div>
        </div>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-300 p-1.5 rounded-lg hover:bg-gray-800 transition-colors"><XIcon size={14} /></button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
            <div className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs ${msg.role === "user" ? "bg-violet-600" : "bg-gray-800 border border-gray-700/50"}`}>
              {msg.role === "user" ? <UserIcon size={12} className="text-white" /> : <BotIcon size={12} className="text-violet-400" />}
            </div>
            <div className={`max-w-[87%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${msg.role === "user" ? "bg-violet-600 text-white rounded-tr-sm" : "bg-gray-900 border border-gray-800/60 text-gray-200 rounded-tl-sm"}`}>
              {msg.role === "user"
                ? <span className="whitespace-pre-wrap">{msg.content}</span>
                : <MarkdownRenderer content={msg.content} onCitationClick={onCitationClick} />}
              {msg.streaming && <span className="inline-block w-1.5 h-4 bg-violet-400 rounded-sm animate-pulse ml-0.5 align-middle" />}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="p-3 border-t border-gray-800/60">
        <div className="flex gap-2 items-center bg-gray-900 border border-gray-800 rounded-xl px-3 py-2.5 focus-within:border-violet-500/50 transition-colors">
          <input
            ref={inputRef} value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
            placeholder="Ask anything about this idea…"
            className="flex-1 bg-transparent text-sm text-gray-200 placeholder-gray-600 outline-none"
            disabled={busy}
          />
          <button onClick={sendMessage} disabled={!input.trim() || busy}
            className="flex-shrink-0 w-7 h-7 rounded-lg bg-violet-600 flex items-center justify-center disabled:opacity-40 hover:bg-violet-500 transition-colors">
            {busy ? <Loader2Icon size={12} className="animate-spin text-white" /> : <SendIcon size={12} className="text-white" />}
          </button>
        </div>
      </div>
    </motion.div>
  );
}

// ── Idea Generation Bar (same pattern as Study page) ─────────────────────────

const IDEA_GEN_BUTTONS: { type: GenerationType; emoji: string; label: string }[] = [
  { type: "podcast", emoji: "🎙", label: "Audio"  },
  { type: "slides",  emoji: "📊", label: "Slides" },
];

function IdeaGenerationBar({ capsuleId, ddStatus }: { capsuleId: string; ddStatus: DeepDiveStatus }) {
  const [artifacts, setArtifacts] = useState<Record<GenerationType, GeneratedArtifact | null>>({
    podcast: null, slides: null,
  });
  const [loading, setLoading] = useState<Record<GenerationType, boolean>>({
    podcast: false, slides: false,
  });
  const [viewer, setViewer] = useState<GenerationType | null>(null);
  const [viewerArtifact, setViewerArtifact] = useState<GeneratedArtifact | null>(null);
  const addGenerationJob = useJobsStore((s) => s.addGenerationJob);

  useEffect(() => {
    if (!capsuleId) return;
    api.get<GeneratedArtifact[]>(`/generate/capsule/${capsuleId}`)
      .then((arts) => {
        // Backend returns newest-first. Take the first artifact seen per type
        // so a newer running/queued artifact is never overwritten by an older
        // completed one.
        const map: Record<GenerationType, GeneratedArtifact | null> = {
          podcast: null, slides: null,
        };
        for (const a of arts) {
          const t = a.generation_type as GenerationType;
          if (!map[t]) map[t] = a;
        }
        setArtifacts(map);
      })
      .catch(() => {});
  }, [capsuleId]);

  // The JobsPanel already polls /generate/jobs every 4s — read in-flight
  // artifact status from the store rather than running a duplicate poll.
  const fetchedCompletedRef = useRef<Set<string>>(new Set());
  const storeGenJobs = useJobsStore((s) => s.generationJobs);
  useEffect(() => {
    const toFetch: { artifactId: string; genType: GenerationType }[] = [];
    for (const gj of storeGenJobs) {
      if (gj.source_id !== capsuleId) continue;
      if (
        gj.status === "completed" &&
        !gj.blob_path &&
        gj.artifact_id &&
        !fetchedCompletedRef.current.has(gj.artifact_id)
      ) {
        fetchedCompletedRef.current.add(gj.artifact_id);
        toFetch.push({ artifactId: gj.artifact_id, genType: gj.generation_type as GenerationType });
      }
    }

    setArtifacts((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const gj of storeGenJobs) {
        if (gj.source_id !== capsuleId) continue;
        const t = gj.generation_type as GenerationType;
        const existing = prev[t];
        if (!existing) continue;
        if (existing.id !== gj.artifact_id) continue;
        if (existing.status === gj.status && existing.blob_path === gj.blob_path) continue;
        next[t] = {
          ...existing,
          status: gj.status as GeneratedArtifact["status"],
          // Never overwrite with null from JobStore (same pattern as study page)
          blob_path: gj.blob_path ?? existing.blob_path ?? null,
          content: gj.content ?? existing.content ?? null,
          error_message: gj.error_message,
          completed_at: gj.completed_at,
        };
        changed = true;
      }
      return changed ? next : prev;
    });

    for (const { artifactId, genType } of toFetch) {
      api.get<GeneratedArtifact>(`/generate/artifact/${artifactId}`)
        .then((art) => setArtifacts((prev) => ({ ...prev, [genType]: art })))
        .catch(() => {});
    }
  }, [storeGenJobs, capsuleId]);

  async function trigger(genType: GenerationType, forceRegenerate = false) {
    setLoading((prev) => ({ ...prev, [genType]: true }));
    const qs = forceRegenerate ? "?force_regenerate=true" : "";
    try {
      const resp = await api.post<{ artifact_id: string; job_id: string; status: string; source_title?: string }>(
        `/generate/capsule/${capsuleId}/${genType}${qs}`
      );
      if (resp.status === "completed") {
        const art = await api.get<GeneratedArtifact>(`/generate/artifact/${resp.artifact_id}`);
        setArtifacts((prev) => ({ ...prev, [genType]: art }));
      } else {
        const sourceTitle = resp.source_title || `${genType} for idea`;
        const optimistic: GeneratedArtifact = {
          id: resp.artifact_id,
          generation_type: genType,
          source_type: "capsule",
          source_id: capsuleId,
          source_title: sourceTitle,
          status: "queued",
          blob_path: null, content: null, expertise_level: null, orientation: null,
          provider: null, model_used: null, input_tokens: 0, output_tokens: 0,
          generation_duration_ms: 0, error_message: null,
          created_at: new Date().toISOString(), completed_at: null,
        };
        setArtifacts((prev) => ({ ...prev, [genType]: optimistic }));
        addGenerationJob({
          artifact_id: resp.artifact_id, job_id: resp.job_id,
          source_type: "capsule", source_id: capsuleId,
          generation_type: genType, title: sourceTitle,
          status: "queued", error_message: null, blob_path: null, content: null,
          created_at: new Date().toISOString(), completed_at: null,
        });
      }
    } catch (err) {
      console.error("generation trigger failed:", err);
    } finally {
      setLoading((prev) => ({ ...prev, [genType]: false }));
    }
  }

  return (
    <>
      <div className="flex items-center gap-2 flex-wrap my-4 px-1">
        <span className="text-[10px] font-semibold text-gray-600 uppercase tracking-wider">Generate:</span>
        {ddStatus !== "done" ? (
          IDEA_GEN_BUTTONS.map(({ type, emoji, label }) => (
            <button
              key={type}
              disabled
              title="Generate the Deep Dive first to unlock media generation"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold border border-gray-800/50 text-gray-600 cursor-not-allowed opacity-50"
            >
              <span>{emoji}</span>
              {label}
              <span className="text-[9px] text-gray-600 ml-0.5">·&nbsp;needs deep dive</span>
            </button>
          ))
        ) : (
          IDEA_GEN_BUTTONS.map(({ type, emoji, label }) => {
            const art = artifacts[type];
            const isLoading = loading[type];
            const isDone = art?.status === "completed";
            const isRunning = art?.status === "queued" || art?.status === "running";

            return (
              <button
                key={type}
                onClick={() => {
                  if (isDone) { setViewerArtifact(art); setViewer(type); return; }
                  if (!isRunning && !isLoading) trigger(type);
                }}
                disabled={isLoading || isRunning}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold border transition-all disabled:opacity-60 ${
                  isDone
                    ? "bg-violet-600/15 border-violet-500/50 text-violet-200"
                    : "border-gray-700/50 text-gray-400 hover:border-violet-600/40 hover:text-violet-300 hover:bg-violet-900/20"
                }`}
              >
                {(isLoading || isRunning) ? (
                  <Loader2Icon size={10} className="animate-spin" />
                ) : (
                  <span>{emoji}</span>
                )}
                {label}
                {isDone && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 ml-0.5" />}
              </button>
            );
          })
        )}
      </div>

      {/* Minimal inline viewer for capsule artifacts */}
      <AnimatePresence>
        {viewer && viewerArtifact && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
            onClick={() => setViewer(null)}
          >
            <motion.div
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.95 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-gray-950 border border-gray-800 rounded-2xl w-full max-w-2xl max-h-[80vh] overflow-y-auto shadow-2xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-bold text-white capitalize">
                  {IDEA_GEN_BUTTONS.find(b => b.type === viewer)?.emoji} {viewer}
                </h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => { if (viewer) { trigger(viewer, true); setViewer(null); } }}
                    title="Regenerate (background job)"
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl text-[10px] font-semibold border border-gray-700/50 text-gray-400 hover:text-indigo-300 hover:border-indigo-500/50 hover:bg-indigo-950/20 transition-all"
                  >
                    <RefreshCwIcon size={10} /> Regenerate
                  </button>
                  <button onClick={() => setViewer(null)} className="text-gray-500 hover:text-white">
                    <XIcon size={16} />
                  </button>
                </div>
              </div>
              {(() => {
                const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                // Prefer the live artifact from the store (updated by per-artifact GET)
                // over the snapshot taken at modal-open time, so blob_path is always
                // current even if it arrived after the user clicked "Audio"/"Slides".
                const displayArt = (viewer && artifacts[viewer as GenerationType]) ?? viewerArtifact;
                const blobUrl = displayArt.blob_path ? `${API}/blobs/${displayArt.blob_path}` : null;
                const script = displayArt.content?.script as string | undefined;
                const markdown = displayArt.content?.marp_markdown as string | undefined;

                if (viewer === "podcast") {
                  return (
                    <div className="space-y-3">
                      {blobUrl ? (
                        <audio controls className="w-full rounded-xl" src={blobUrl} />
                      ) : (
                        <p className="text-xs text-gray-400">Audio not yet available.</p>
                      )}
                      {script && (
                        <details>
                          <summary className="text-xs font-semibold text-gray-400 cursor-pointer hover:text-gray-200">View Script</summary>
                          <pre className="mt-2 text-[11px] text-gray-400 whitespace-pre-wrap max-h-60 overflow-y-auto bg-gray-900/60 rounded-xl p-3">{script}</pre>
                        </details>
                      )}
                    </div>
                  );
                }

                if (viewer === "slides") {
                  if (blobUrl) {
                    return (
                      <div className="space-y-2">
                        <iframe
                          src={blobUrl}
                          className="w-full rounded-xl border border-gray-800"
                          style={{ height: "420px" }}
                          title="Slide Deck"
                        />
                        <a href={blobUrl} download className="inline-flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300">
                          Download HTML
                        </a>
                      </div>
                    );
                  }
                  if (markdown) {
                    return (
                      <pre className="text-[11px] text-gray-400 whitespace-pre-wrap max-h-80 overflow-y-auto bg-gray-900/60 rounded-xl p-3">
                        {markdown}
                      </pre>
                    );
                  }
                }

                return null;
              })()}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

type DeepDiveStatus = "idle" | "streaming" | "done" | "error";

export default function IdeaDeepDivePage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { token } = useAuthStore();
  const addDeepDiveJob = useJobsStore((s) => s.addDeepDiveJob);
  const [capsule, setCapsule] = useState<IdeaCapsule | null>(null);
  const [status, setStatus] = useState<"loading" | "done" | "error">("loading");
  const [showChat, setShowChat] = useState(false);
  const [readPct, setReadPct] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const deepDiveRef = useRef<HTMLDivElement>(null);

  // Highlight + keyword search scoped to this idea. Mirrors the RA chat
  // experience: select text to highlight, Ctrl+F to search, persists in
  // localStorage so the marks survive reload and idea-switch.
  const highlightSearch = useHighlightSearch(`idea:${id ?? "_"}`, !!id);

  // Source-paper preview panel — clicking a source paper opens the same
  // PaperPanel overlay used on the Feed so the user can read the abstract,
  // bookmark, or jump into Study Mode without leaving this page.
  const [previewPaper, setPreviewPaper] = useState<Paper | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  async function openSourcePaper(paperId: string) {
    if (!paperId || previewLoading) return;
    setPreviewLoading(true);
    try {
      const paper = await api.get<Paper>(`/papers/${paperId}`);
      setPreviewPaper(paper);
    } catch (e) {
      console.error("openSourcePaper failed:", e);
    } finally {
      setPreviewLoading(false);
    }
  }

  // Deep dive state
  const [ddStatus, setDdStatus] = useState<DeepDiveStatus>("idle");
  const [ddText, setDdText] = useState("");
  const [ddStatusMsg, setDdStatusMsg] = useState("");

  // Combine-with-another-capsule modal state.
  // ``combineOpen`` controls visibility; ``combineList`` is lazy-loaded the
  // first time the picker opens so the page doesn't pay for it upfront.
  // ``combineSelected`` is the set of OTHER capsule ids the user picked —
  // the current idea (``id``) is always counted in, so we cap this at 2
  // (current + 2 others = max 3 capsules fused).
  const [combineOpen, setCombineOpen] = useState(false);
  // Combine sub-feature gate — hides the "Combine with…" CTA when the
  // admin (or per-user override) has turned the feature off.
  const combineEnabled = useFeature("genie_combine_enabled", true);
  const [combineList, setCombineList] = useState<IdeaCapsule[] | null>(null);
  const [combineLoading, setCombineLoading] = useState(false);
  const [combineSubmitting, setCombineSubmitting] = useState(false);
  const [combineError, setCombineError] = useState<string | null>(null);
  const [combineQuery, setCombineQuery] = useState("");
  const [combineSelected, setCombineSelected] = useState<string[]>([]);

  // Reset deep-dive state whenever the idea id changes so stale content from a
  // previous idea never flashes while the new one is loading.
  useEffect(() => {
    setDdText("");
    setDdStatus("idle");
    setDdStatusMsg("");
  }, [id]);

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    setReadPct(Math.min(100, (el.scrollTop / (el.scrollHeight - el.clientHeight)) * 100));
  }

  useEffect(() => {
    if (!id || !token) return;
    const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${API}/api/v1/genie/capsules/${id}`, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((data: IdeaCapsule) => {
        setCapsule(data);
        setStatus("done");
        // Restore stored deep dive if already generated
        if (data.deep_dive_status === "done" && data.deep_dive_content) {
          setDdText(data.deep_dive_content);
          setDdStatus("done");
        } else if (data.deep_dive_status === "generating") {
          setDdStatus("streaming");
          setDdStatusMsg("Deep Dive is generating in the background…");
        }
      })
      .catch(() => setStatus("error"));
  }, [id, token]);

  // Poll capsule when a background generation is running
  useEffect(() => {
    if (ddStatus !== "streaming" || ddText) return; // only poll if no live stream
    if (!id || !token) return;
    const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    // Hard deadline: surface an error if the backend never finishes within 15 min
    // so the user is never stuck on an infinite "Background generation in progress…".
    const deadline = setTimeout(() => {
      clearInterval(interval);
      setDdStatus("error");
      setDdStatusMsg("Generation timed out. Please try again.");
    }, 15 * 60 * 1000);

    const interval = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/v1/genie/capsules/${id}`, { headers: { Authorization: `Bearer ${token}` } });
        if (!r.ok) return;
        const data: IdeaCapsule = await r.json();
        if (data.deep_dive_status === "done" && data.deep_dive_content) {
          clearTimeout(deadline);
          setDdText(data.deep_dive_content);
          setDdStatus("done");
          setDdStatusMsg("");
          clearInterval(interval);
        } else if (data.deep_dive_status === "failed") {
          clearTimeout(deadline);
          setDdStatus("error");
          setDdStatusMsg("");
          clearInterval(interval);
        }
      } catch {}
    }, 5000);
    return () => {
      clearTimeout(deadline);
      clearInterval(interval);
    };
  }, [ddStatus, ddText, id, token]);

  // Lazily load every other capsule the user owns so the picker can show them.
  async function openCombine() {
    setCombineOpen(true);
    setCombineError(null);
    setCombineSelected([]);
    if (combineList !== null) return;
    setCombineLoading(true);
    try {
      const list = await api.get<IdeaCapsule[]>("/genie/capsules");
      setCombineList(list);
    } catch (e) {
      setCombineError(`Failed to load capsule list: ${String(e).slice(0, 120)}`);
      setCombineList([]);
    } finally {
      setCombineLoading(false);
    }
  }

  function toggleCombineSelected(otherId: string) {
    setCombineError(null);
    setCombineSelected((prev) => {
      if (prev.includes(otherId)) return prev.filter((x) => x !== otherId);
      if (prev.length >= 2) return prev; // cap at 2 others (max 3 fused)
      return [...prev, otherId];
    });
  }

  // Submit the combine request. The backend queues the combine and returns a
  // GenieSession id immediately (202). We then poll the session row every
  // ~2.5 s until it lands on a terminal status:
  //   * status="done" + result_capsule_id  → navigate to the new hybrid capsule
  //   * status="failed" or "done_empty"    → surface the error to the user
  //
  // Polling caps at 6 minutes (~145 attempts) — combine can be heavy when both
  // parents' deep dives need to be generated first, but anything longer than
  // 6 minutes is almost certainly a stuck task and we want the user to know.
  async function submitCombine(otherIds: string[]) {
    if (!id || otherIds.length === 0 || combineSubmitting) return;
    setCombineSubmitting(true);
    setCombineError(null);
    try {
      const queued = await api.post<{ session_id: string; status: string; parent_ids: string[] }>(
        "/genie/capsules/combine",
        { capsule_ids: [id, ...otherIds] },
      );
      if (!queued?.session_id) {
        setCombineError("Combine could not be queued.");
        return;
      }
      // Register with the global jobs store — the panel shows progress,
      // a toast fires on completion, and the user can navigate freely.
      useJobsStore.getState().addGenieJob({
        session_id: queued.session_id,
        status: "running",
        capsule_id: null,
        error: null,
        created_at: new Date().toISOString(),
        completed_at: null,
        label: "Combine ideas",
      });
      setCombineOpen(false);
    } catch (e: unknown) {
      // 400 / 404 / 5xx surface here. The backend uses 422 historically for
      // infeasible pairs, but the new endpoint moves that verdict into the
      // session row (status="done_empty" + error).
      const err = e as { status?: number; detail?: string | { message?: string }; message?: string };
      const detail = typeof err?.detail === "string" ? err.detail : err?.detail?.message;
      const reason = detail || err?.message || "Combine request failed. Please try again.";
      setCombineError(reason);
    } finally {
      setCombineSubmitting(false);
    }
  }

  async function startDeepDiveBg() {
    if (!id || !token || ddStatus === "streaming") return;
    const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    try {
      const resp = await fetch(`${API}/api/v1/genie/capsules/${id}/deep-dive-bg`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: "{}",
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setDdStatus("streaming");
      setDdText("");
      setDdStatusMsg("Generating in the background — you can keep working; the Deep Dive will appear here when it's ready.");
      addDeepDiveJob({
        capsule_id: id,
        capsule_title: capsule?.title ?? "",
        status: "generating",
        created_at: new Date().toISOString(),
        completed_at: null,
        error: null,
      });
      setTimeout(() => deepDiveRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (e) {
      setDdStatusMsg(`Couldn't start the Deep Dive — ${String(e).slice(0, 80)}`);
    }
  }

  const diagrams = capsule?.diagrams ?? [];
  const hasDiagrams = diagrams.some(d => d.spec || d.blob_path);


  const navItems = useMemo(() => {
    if (!capsule || status !== "done") return [];
    const items: { id: string; label: string; icon: string }[] = [];
    for (const def of SECTIONS) {
      const content = capsule[def.key] as string | null;
      if (!content) continue;
      items.push({ id: `genie-section-${def.key}`, label: def.label, icon: def.emoji });
    }
    if (capsule.poc_code) {
      const isCode = /^```(\w*)\n([\s\S]*?)```\s*$/.test(capsule.poc_code);
      items.push({ id: "genie-section-poc", label: isCode ? "Proof of Concept" : "Method Sketch", icon: "💻" });
    }
    if (capsule.source_papers?.length) items.push({ id: "genie-section-sources", label: "Source Papers", icon: "📄" });
    if (hasDiagrams) items.push({ id: "genie-section-diagrams", label: "Diagrams", icon: "🖼" });
    if (ddStatus !== "idle") items.push({ id: "genie-section-deepdive", label: "Full Deep Dive", icon: "🔬" });
    return items;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capsule, status, hasDiagrams, ddStatus]);

  return (
    <div className="flex h-full" style={{ background: "var(--rf-bg)" }}>
      {/* Section navigation panel */}
      {status === "done" && navItems.length > 0 && (
        <SectionNavPanel items={navItems} scrollRef={scrollRef} accent="violet" />
      )}

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className={`flex-1 overflow-y-auto transition-all duration-300 ${showChat ? "mr-[420px]" : ""}`}
        style={{ background: "var(--rf-bg)" }}
      >
        {/* Reading progress */}
        <div className="sticky top-0 left-0 right-0 h-0.5 bg-gray-800/40 z-20">
          <motion.div
            className="h-full bg-gradient-to-r from-violet-500 via-purple-500 to-fuchsia-500"
            animate={{ width: `${readPct}%` }}
            transition={{ duration: 0.1 }}
          />
        </div>

        <div className="max-w-4xl mx-auto px-8 py-8">
          {/* Header */}
          <div className="sticky top-0.5 z-10 mb-8">
            <div className="flex items-center gap-3 bg-gray-900/80 backdrop-blur-md border border-gray-800/60 rounded-2xl px-4 py-2.5 shadow-lg shadow-black/20">
              <button onClick={() => router.push("/genie?tab=discoveries")} className="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-200 transition-colors font-medium">
                <ArrowLeftIcon size={13} />Ideas
              </button>
              <div className="flex-1" />
              {status === "done" && (
                <>
                  <button
                    onClick={startDeepDiveBg}
                    disabled={ddStatus === "streaming"}
                    className="flex items-center gap-2 px-4 py-1.5 rounded-xl text-xs font-semibold transition-all"
                    style={
                      ddStatus === "streaming"
                        ? { background: "rgba(124,58,237,0.15)", color: "#a78bfa", border: "1px solid rgba(124,58,237,0.3)", cursor: "wait" }
                        : ddStatus === "done"
                        ? { background: "rgba(16,185,129,0.12)", color: "#34d399", border: "1px solid rgba(16,185,129,0.25)" }
                        : { background: "linear-gradient(135deg,#7c3aed,#a855f7)", color: "#fff", boxShadow: "0 2px 10px rgba(124,58,237,0.35)" }
                    }
                  >
                    {ddStatus === "streaming" ? (
                      <><Loader2Icon size={12} className="animate-spin" />Generating…</>
                    ) : ddStatus === "done" ? (
                      <><FileTextIcon size={12} />Regenerate Deep Dive</>
                    ) : (
                      <><SparklesIcon size={12} />Generate Deep Dive</>
                    )}
                  </button>
                  {combineEnabled && (
                    <button
                      onClick={openCombine}
                      className="flex items-center gap-2 px-4 py-1.5 rounded-xl text-xs font-semibold transition-all bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700/50"
                      title="Fuse this idea with another to produce a new hybrid hypothesis"
                    >
                      <GitMergeIcon size={12} />
                      Combine with…
                    </button>
                  )}
                  {ddStatus === "done" && (
                    <button
                      onClick={() => setShowChat(v => !v)}
                      className={`flex items-center gap-2 px-4 py-1.5 rounded-xl text-xs font-semibold transition-all ${showChat ? "bg-violet-600 text-white" : "bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700/50"}`}
                    >
                      <MessageSquareIcon size={12} />
                      {showChat ? "Close Chat" : "Ask Questions"}
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Loading */}
          {status === "loading" && (
            <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
              <div className="relative w-16 h-16">
                <div className="absolute inset-0 rounded-full bg-violet-500/10 blur-xl animate-pulse" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Loader2Icon size={28} className="text-violet-400 animate-spin" />
                </div>
              </div>
              <p className="text-sm text-gray-500">Loading research idea…</p>
            </div>
          )}

          {/* Error */}
          {status === "error" && (
            <div className="mt-8 bg-red-950/50 border border-red-800/60 rounded-2xl p-5 text-red-300 text-sm">
              Could not load this idea. It may have been dismissed or deleted.
            </div>
          )}

          {/* Content */}
          {status === "done" && capsule && (
            <DecorationsProvider value={highlightSearch.decorations}>
            <CitationCtx.Provider value={(num: string) => {
              const idx = parseInt(num, 10) - 1;
              const sp = capsule?.source_papers?.[idx];
              if (sp?.id) openSourcePaper(sp.id);
            }}>
            <div ref={highlightSearch.scrollRef}>
              <CapsuleHero capsule={capsule} />

              {/* Highlight + keyword-search toolbar — anchored just under the
                  hero so it scrolls into view above the actual content. The
                  toolbar's empty state collapses to a small two-button strip
                  so it never crowds the dense idea-detail layout. */}
              <div className="mb-3">
                <HighlightSearchToolbar toolbar={highlightSearch.toolbar} />
              </div>

              {/* Media generation bar for capsule */}
              <IdeaGenerationBar capsuleId={capsule.id} ddStatus={ddStatus} />
              <div className="space-y-10">
                {SECTIONS.map((def, i) => {
                  const content = capsule[def.key] as string | null;
                  if (!content) return null;
                  return <SectionBlock key={def.key} def={def} content={content} index={i} />;
                })}

                {/* POC Code / Method Sketch */}
                {capsule.poc_code && (() => {
                  const fenceMatch = capsule.poc_code.match(/^```(\w*)\n([\s\S]*?)```\s*$/);
                  const isCode = !!fenceMatch;
                  const codeContent = fenceMatch ? fenceMatch[2].trim() : capsule.poc_code;
                  const lang = fenceMatch ? (fenceMatch[1] || "python") : "python";
                  return (
                    <motion.div id="genie-section-poc" initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} className="scroll-mt-6">
                      <div className="flex items-center gap-4 mb-5">
                        <div className="relative flex items-center justify-center w-14 h-14 rounded-2xl border bg-gray-800/80 border-gray-700/60 flex-shrink-0 shadow-xl overflow-hidden">
                          <div className="absolute inset-0 bg-gradient-to-br from-gray-500 to-gray-600 opacity-15" />
                          <CodeIcon size={22} className="relative z-10 text-gray-300" />
                        </div>
                        <div>
                          <h2 className="text-2xl font-bold text-gray-300 tracking-tight leading-tight">
                            {isCode ? "Proof of Concept" : "Method Sketch"}
                          </h2>
                          <p className="text-xs text-gray-600 mt-0.5">
                            {isCode ? "Core implementation of the proposed mechanism" : "Concise step-by-step sketch of the proposed approach"}
                          </p>
                          <div className="h-0.5 mt-2 w-16 rounded-full bg-gradient-to-r from-gray-500 to-gray-600" />
                        </div>
                      </div>
                      {isCode ? (
                        <CodeBlock code={codeContent} lang={lang} />
                      ) : (
                        <div className="rounded-2xl border border-gray-700/50 bg-gray-900/50 px-5 py-4 text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                          {codeContent}
                        </div>
                      )}
                    </motion.div>
                  );
                })()}

                {/* Source Papers */}
                {capsule.source_papers && capsule.source_papers.length > 0 && (
                  <motion.div id="genie-section-sources" initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} className="scroll-mt-6">
                    <div className="flex items-center gap-4 mb-5">
                      <div className="relative flex items-center justify-center w-14 h-14 rounded-2xl border bg-sky-950/60 border-sky-800/60 flex-shrink-0 shadow-xl overflow-hidden">
                        <div className="absolute inset-0 bg-gradient-to-br from-sky-500 to-blue-600 opacity-15" />
                        <FileTextIcon size={22} className="relative z-10 text-sky-300" />
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold text-sky-300 tracking-tight leading-tight">Source Papers</h2>
                        <p className="text-xs text-gray-600 mt-0.5">Papers this idea was synthesized from</p>
                        <div className="h-0.5 mt-2 w-16 rounded-full bg-gradient-to-r from-sky-500 to-blue-600" />
                      </div>
                    </div>
                    <div className="rounded-2xl border border-sky-800/30 bg-sky-950/10 divide-y divide-sky-900/30 overflow-hidden">
                      {capsule.source_papers.map((p, idx) => (
                        <button
                          key={p.id}
                          onClick={() => openSourcePaper(p.id)}
                          disabled={previewLoading}
                          className="w-full text-left flex items-start gap-4 px-5 py-4 hover:bg-sky-900/20 transition-colors group disabled:opacity-60"
                          title="Open paper preview"
                        >
                          <span className="mt-0.5 flex-shrink-0 w-6 h-6 rounded-full bg-sky-900/60 border border-sky-700/50 flex items-center justify-center text-xs font-bold text-sky-400">
                            {idx + 1}
                          </span>
                          <div className="min-w-0 flex-1">
                            <p className="text-sm font-semibold text-sky-200 group-hover:text-sky-100 transition-colors leading-snug">
                              {p.title}
                            </p>
                            <p className="text-xs text-gray-500 mt-0.5">
                              {p.authors.slice(0, 3).join(", ")}{p.authors.length > 3 ? " et al." : ""}
                              {p.year ? ` · ${p.year}` : ""}
                            </p>
                          </div>
                          {previewLoading ? (
                            <Loader2Icon size={14} className="flex-shrink-0 mt-1 text-sky-600 animate-spin" />
                          ) : (
                            <LinkIcon size={14} className="flex-shrink-0 mt-1 text-sky-600 group-hover:text-sky-400 transition-colors" />
                          )}
                        </button>
                      ))}
                    </div>
                  </motion.div>
                )}

                {/* Diagrams */}
                {hasDiagrams && (
                  <motion.div id="genie-section-diagrams" initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} className="scroll-mt-6">
                    <div className="flex items-center gap-4 mb-5">
                      <div className="relative flex items-center justify-center w-14 h-14 rounded-2xl border bg-violet-950/60 border-violet-800/60 flex-shrink-0 shadow-xl overflow-hidden">
                        <div className="absolute inset-0 bg-gradient-to-br from-violet-500 to-purple-600 opacity-15" />
                        <LinkIcon size={22} className="relative z-10 text-violet-300" />
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold text-violet-300 tracking-tight leading-tight">Diagrams</h2>
                        <p className="text-xs text-gray-600 mt-0.5">Visual representations of the idea</p>
                        <div className="h-0.5 mt-2 w-16 rounded-full bg-gradient-to-r from-violet-500 to-purple-600" />
                      </div>
                    </div>
                    <div className="space-y-6">
                      {diagrams.filter(d => d.spec || d.blob_path).map((d, i) => <DiagramSection key={i} diagram={d} index={i} />)}
                    </div>
                  </motion.div>
                )}

                {/* ── Deep Dive Article ── */}
                {(ddStatus !== "idle") && (
                  <motion.div
                    id="genie-section-deepdive"
                    ref={deepDiveRef}
                    initial={{ opacity: 0, y: 32 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                    className="mt-4 scroll-mt-6"
                  >
                    {/* Section header */}
                    <div className="flex items-center gap-4 mb-6">
                      <div className="relative flex items-center justify-center w-14 h-14 rounded-2xl border bg-fuchsia-950/60 border-fuchsia-800/60 flex-shrink-0 shadow-xl overflow-hidden">
                        <div className="absolute inset-0 bg-gradient-to-br from-fuchsia-500 to-purple-600 opacity-20" />
                        <SparklesIcon size={22} className="relative z-10 text-fuchsia-300" />
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold text-fuchsia-300 tracking-tight leading-tight">Full Deep Dive</h2>
                        <p className="text-xs text-gray-600 mt-0.5">
                          {ddStatusMsg || (ddStatus === "streaming" ? "Generating deep research article…" : "Deep reasoning model · Research-grade article")}
                        </p>
                        <div className="h-0.5 mt-2 w-16 rounded-full bg-gradient-to-r from-fuchsia-500 to-purple-600" />
                      </div>
                      {ddStatus === "streaming" && (
                        <div className="ml-auto flex items-center gap-2 text-xs text-fuchsia-400/70">
                          <Loader2Icon size={13} className="animate-spin" />
                          <span>{ddText ? "Writing…" : "Thinking…"}</span>
                        </div>
                      )}
                    </div>

                    {/* Article body */}
                    <div className="relative rounded-2xl border border-fuchsia-800/30 bg-gradient-to-b from-fuchsia-950/10 to-purple-950/5 px-8 py-7 shadow-xl">
                      <div className="absolute left-0 top-8 bottom-8 w-0.5 rounded-full bg-gradient-to-b from-fuchsia-500 to-purple-600 opacity-40" />
                      <div className="pl-5 space-y-4">
                        {ddStatus === "streaming" && !ddText ? (
                          <div className="flex flex-col gap-3 py-4">
                            <p className="text-sm text-fuchsia-400/70 animate-pulse">
                              {ddStatusMsg || "Composing synthesis draft…"}
                            </p>
                            <div className="space-y-2 opacity-30">
                              {[80, 65, 90, 55, 75].map((w, i) => (
                                <div key={i} className={`h-2 rounded-full bg-fuchsia-500/40 animate-pulse`} style={{ width: `${w}%`, animationDelay: `${i * 0.15}s` }} />
                              ))}
                            </div>
                          </div>
                        ) : (
                          <>
                            {renderContent(ddText)}
                            {ddStatus === "streaming" && (
                              <span className="inline-block w-2 h-5 bg-fuchsia-400 rounded-sm animate-pulse align-middle ml-0.5" />
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* CTA when deep dive not yet started */}
                {ddStatus === "idle" && (
                  <motion.div
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.3 }}
                    className="mt-4 rounded-2xl border border-dashed border-fuchsia-800/40 bg-fuchsia-950/10 p-8 flex flex-col items-center gap-4 text-center"
                  >
                    <div className="w-14 h-14 rounded-2xl bg-fuchsia-950/60 border border-fuchsia-800/50 flex items-center justify-center">
                      <SparklesIcon size={24} className="text-fuchsia-400" />
                    </div>
                    <div>
                      <p className="text-base font-semibold text-white mb-1">Generate Full Deep Dive</p>
                      <p className="text-sm text-gray-500 max-w-md">
                        A comprehensive 4000+ word research article — theoretical foundations,
                        experimental protocol, predicted outcomes, risks, and scientific impact.
                        Uses the source paper content as context.
                      </p>
                    </div>
                    <button
                      onClick={startDeepDiveBg}
                      className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-fuchsia-600 to-purple-600 text-white text-sm font-semibold hover:from-fuchsia-500 hover:to-purple-500 transition-all shadow-lg shadow-fuchsia-500/20"
                    >
                      <SparklesIcon size={14} />
                      Generate Deep Dive
                    </button>
                  </motion.div>
                )}
              </div>
              <div className="h-16" />
            </div>
            </CitationCtx.Provider>
            </DecorationsProvider>
          )}
        </div>
      </div>

      <AnimatePresence>
        {showChat && id && (
          <IdeaChatPanel
            capsuleId={id}
            onClose={() => setShowChat(false)}
            onCitationClick={(num) => {
              // Citations rendered by the chat are 1-indexed against the
              // capsule's ``source_papers`` array — open the same paper
              // preview overlay used by the Source Papers section.
              const idx = parseInt(num, 10) - 1;
              const sp = capsule?.source_papers?.[idx];
              if (sp?.id) openSourcePaper(sp.id);
            }}
          />
        )}
      </AnimatePresence>

      {/* Source-paper preview overlay — same PaperPanel used on the Feed,
          so behaviour stays consistent across the app. */}
      <AnimatePresence>
        {previewPaper && (
          <PaperPanel
            key={previewPaper.id}
            paper={previewPaper}
            onClose={() => setPreviewPaper(null)}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {combineOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm px-4"
            onClick={() => !combineSubmitting && setCombineOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.96, y: 12, opacity: 0 }}
              animate={{ scale: 1, y: 0, opacity: 1 }}
              exit={{ scale: 0.96, y: 12, opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={(e) => e.stopPropagation()}
              className="w-full max-w-2xl rounded-2xl bg-gray-950 border border-gray-800 shadow-2xl overflow-hidden flex flex-col max-h-[80vh]"
            >
              <div className="px-6 pt-5 pb-4 border-b border-gray-800/60 flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-violet-950/40 border border-violet-800/30 flex items-center justify-center text-violet-300">
                  <GitMergeIcon size={18} />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-base font-semibold text-white">Combine with another idea</h3>
                  <p className="text-xs text-gray-500 mt-0.5">
                    A reasoning-tier model fuses both deep dives into a new hybrid hypothesis. A feasibility judge will decline pairs that are too similar or too disjoint.
                  </p>
                </div>
                <button
                  onClick={() => !combineSubmitting && setCombineOpen(false)}
                  disabled={combineSubmitting}
                  className="text-gray-500 hover:text-gray-200 transition-colors p-1 disabled:opacity-40 disabled:cursor-not-allowed"
                  aria-label="Close"
                >
                  <XIcon size={16} />
                </button>
              </div>

              <div className="px-6 py-3 border-b border-gray-800/40">
                <input
                  type="text"
                  value={combineQuery}
                  onChange={(e) => setCombineQuery(e.target.value)}
                  placeholder="Search your ideas by title…"
                  className="w-full bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-violet-600/50"
                  autoFocus
                />
              </div>

              {combineError && (
                <div className="mx-6 mt-3 px-3 py-2 rounded-lg bg-red-950/40 border border-red-800/40 text-red-300 text-xs">
                  {combineError}
                </div>
              )}

              <div className="flex-1 overflow-y-auto px-3 py-2">
                {combineLoading ? (
                  <div className="flex items-center justify-center py-12 text-sm text-gray-500 gap-2">
                    <Loader2Icon size={14} className="animate-spin" />
                    Loading your ideas…
                  </div>
                ) : (combineList ?? []).filter(c => c.id !== id && (
                  !combineQuery.trim() || (c.title || "").toLowerCase().includes(combineQuery.toLowerCase().trim())
                )).length === 0 ? (
                  <div className="text-center py-12 text-sm text-gray-500">
                    {(combineList?.length ?? 0) === 0
                      ? "You don't have any other saved ideas yet."
                      : "No matches. Try a different search."}
                  </div>
                ) : (
                  <ul className="space-y-1">
                    {(combineList ?? [])
                      .filter(c => c.id !== id && (
                        !combineQuery.trim() || (c.title || "").toLowerCase().includes(combineQuery.toLowerCase().trim())
                      ))
                      .map((c) => {
                        const checked = combineSelected.includes(c.id);
                        const disabled = combineSubmitting || (!checked && combineSelected.length >= 2);
                        return (
                          <li key={c.id}>
                            <button
                              type="button"
                              onClick={() => toggleCombineSelected(c.id)}
                              disabled={disabled}
                              aria-pressed={checked}
                              className={
                                "w-full text-left px-3 py-2.5 rounded-lg transition-colors flex items-start gap-3 " +
                                (checked
                                  ? "bg-violet-950/40 border border-violet-700/50"
                                  : "border border-transparent hover:bg-gray-900") +
                                (disabled ? " opacity-40 cursor-not-allowed" : "")
                              }
                            >
                              <span
                                aria-hidden
                                className={
                                  "mt-0.5 w-4 h-4 rounded border flex items-center justify-center text-[10px] flex-shrink-0 " +
                                  (checked
                                    ? "bg-violet-500 border-violet-400 text-white"
                                    : "border-gray-700 bg-gray-950")
                                }
                              >
                                {checked ? "✓" : ""}
                              </span>
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-gray-200 truncate">{c.title || "Untitled"}</p>
                                <p className="text-xs text-gray-500 line-clamp-2 mt-0.5">{c.hypothesis || ""}</p>
                                <div className="flex items-center gap-2 mt-1.5 text-[10px] text-gray-600">
                                  <span className="px-1.5 py-0.5 rounded bg-gray-900 border border-gray-800">
                                    {c.source_mode || "manual"}
                                  </span>
                                  <span>novelty {(c.novelty_score ?? 0).toFixed(2)}</span>
                                  <span>·</span>
                                  <span>feasibility {(c.feasibility_score ?? 0).toFixed(2)}</span>
                                </div>
                              </div>
                              <SparklesIcon size={12} className="text-violet-400/60 mt-1 flex-shrink-0" />
                            </button>
                          </li>
                        );
                      })}
                  </ul>
                )}
              </div>

              <div className="px-6 py-3 border-t border-gray-800/60 flex items-center justify-between gap-3">
                <p className="text-[11px] text-gray-600">
                  {combineSubmitting
                    ? "Running feasibility check + fusion synthesis…"
                    : combineSelected.length === 0
                      ? "Select 1 or 2 ideas (max 3 total fused)."
                      : `${combineSelected.length + 1} idea(s) selected (incl. this one).`}
                </p>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => !combineSubmitting && setCombineOpen(false)}
                    disabled={combineSubmitting}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium text-gray-400 hover:text-gray-200 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => submitCombine(combineSelected)}
                    disabled={combineSubmitting || combineSelected.length === 0}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium bg-violet-600 text-white hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {combineSubmitting ? "Combining…" : "Combine selected"}
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
