"use client";

import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import MarkdownRenderer from "@/components/ui/MarkdownRenderer";
import { SectionNavPanel } from "@/components/ui/SectionNavPanel";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Loader2Icon,
  MessageSquareIcon,
  SendIcon,
  XIcon,
  BotIcon,
  UserIcon,
  CopyIcon,
  CheckIcon,
  ExternalLinkIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  MicIcon,
  PresentationIcon,
  DownloadIcon,
  RefreshCwIcon,
} from "lucide-react";
import katex from "katex";
import type { StudySection, Paper, GeneratedArtifact, GenerationType, GenerationSourceType } from "@/types";
import { useAuthStore } from "@/store/auth";
import { api, openSSE } from "@/lib/api";
import { useJobsStore } from "@/store/jobs";

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
            "python",
            "javascript",
            "typescript",
            "bash",
            "rust",
            "go",
            "java",
            "cpp",
            "sql",
            "json",
          ],
        })
      )
      .then((h) => {
        _highlighter = h;
        return h;
      });
  }
  return _highlighterPromise;
}

// ── Code block with Shiki ─────────────────────────────────────────────────────

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

  const displayLang = lang ? lang.charAt(0).toUpperCase() + lang.slice(1) : "Code";

  return (
    <div className="rounded-2xl overflow-hidden border border-gray-800/60 bg-[#0d0f12] shadow-xl shadow-black/50 my-1" data-code-block>
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800/50">
        <div className="flex items-center gap-2.5">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/50" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/50" />
            <div className="w-3 h-3 rounded-full bg-green-500/50" />
          </div>
          <span className="text-sm font-semibold text-gray-300 font-mono">{displayLang}</span>
        </div>
        <button
          onClick={copy}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gray-800/80 border border-gray-700/50 text-xs font-semibold text-gray-300 hover:text-white hover:bg-gray-700 transition-all"
          title="Copy code"
        >
          {copied ? (
            <>
              <CheckIcon size={12} className="text-emerald-400" />
              <span className="text-emerald-400">Copied!</span>
            </>
          ) : (
            <>
              <CopyIcon size={12} />
              Copy
            </>
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

// ── Inline text with bold, italic, code, and inline math ──────────────────────

function InlineText({ text }: { text: string }) {
  // Normalise \(...\) → $...$  before splitting
  const normalised = text.replace(/\\\(([\s\S]+?)\\\)/g, (_m, e) => `$${e}$`);
  const parts = normalised.split(
    /(\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`|\$[^$\n]{1,80}\$)/g
  );
  return (
    <>
      {parts.map((part, i) => {
        if (/^\*\*[^*]+\*\*$/.test(part))
          return (
            <strong key={i} className="font-semibold text-white">
              {part.slice(2, -2)}
            </strong>
          );
        if (/^\*[^*\n]+\*$/.test(part) && !part.startsWith("**"))
          return (
            <em key={i} className="italic text-gray-200">
              {part.slice(1, -1)}
            </em>
          );
        if (/^`[^`]+`$/.test(part))
          return (
            <code
              key={i}
              className="px-1.5 py-0.5 rounded-md bg-gray-800 border border-gray-700/60 text-[12px] font-mono text-indigo-300 mx-0.5"
            >
              {part.slice(1, -1)}
            </code>
          );
        if (/^\$[^$]+\$$/.test(part)) {
          try {
            return (
              <span
                key={i}
                dangerouslySetInnerHTML={{
                  __html: katex.renderToString(part.slice(1, -1), {
                    throwOnError: false,
                  }),
                }}
                className="mx-0.5"
              />
            );
          } catch {
            return <span key={i}>{part}</span>;
          }
        }
        return part;
      })}
    </>
  );
}

// ── Callout styles (must be before renderContent) ─────────────────────────────

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

// ── Segment content by code fences first, then process text ──────────────────

type Segment = { type: "code"; lang: string; code: string } | { type: "text"; text: string };

function segmentContent(raw: string): Segment[] {
  // Normalise LaTeX delimiters
  const s = raw
    .replace(/\\\[([\s\S]*?)\\\]/g, (_m, e) => `$$${e}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_m, e) => `$${e}$`);

  const segments: Segment[] = [];
  // Permissive fence regex: handles content starting on same line as language tag
  // (some LLMs emit ```python"""code""" without a newline separator)
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

// ── Content renderer ──────────────────────────────────────────────────────────

function renderContent(rawContent: string) {
  const nodes: React.ReactNode[] = [];
  const segments = segmentContent(rawContent);

  segments.forEach((seg, ci) => {
    if (seg.type === "code") {
      nodes.push(<CodeBlock key={`code-${ci}`} lang={seg.lang} code={seg.code} />);
      return;
    }

    const part = seg.text;
    if (!part.trim()) return;

    // 2. For non-code parts, split on $$ display math
    const mathParts = part.split(/(\$\$[\s\S]*?\$\$)/g);

    mathParts.forEach((mp, mi) => {
      const displayMathMatch = mp.match(/^\$\$([\s\S]*?)\$\$$/);
      if (displayMathMatch) {
        const expr = displayMathMatch[1].trim();
        try {
          const mathHtml = katex.renderToString(expr, {
            displayMode: true,
            throwOnError: false,
          });
          nodes.push(
            <div
              key={`math-${ci}-${mi}`}
              className="my-4 px-4 py-3 rounded-xl bg-gray-900/60 border border-gray-700/40 overflow-x-auto text-center"
              dangerouslySetInnerHTML={{ __html: mathHtml }}
            />
          );
        } catch {
          nodes.push(
            <p key={`math-${ci}-${mi}`} className="text-gray-400 font-mono text-sm">
              {mp}
            </p>
          );
        }
        return;
      }

      // 3. Split on paragraphs
      if (!mp.trim()) return;
      mp.split(/\n\n+/).forEach((para, pi) => {
        if (!para.trim()) return;
        const trimmed = para.trim();

        // Markdown headings — extract first line; render trailing content separately
        const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)/);
        if (headingMatch) {
          const level = headingMatch[1].length;
          const firstLine = headingMatch[0].split("\n")[0];
          const headText = headingMatch[2].split("\n")[0].trim();
          const afterHead = trimmed.slice(firstLine.length).trim();
          const headClass = level === 1
            ? "text-xl font-bold text-white mt-4 mb-1"
            : level === 2
            ? "text-base font-bold text-gray-100 mt-3 mb-1"
            : level === 3
            ? "text-sm font-semibold text-gray-200 mt-2 mb-0.5"
            : "text-sm font-medium text-gray-400 mt-1.5 mb-0.5";
          nodes.push(
            <p key={`h-${ci}-${mi}-${pi}`} className={headClass}>
              <InlineText text={headText} />
            </p>
          );
          if (afterHead) {
            nodes.push(
              <p key={`h-body-${ci}-${mi}-${pi}`} className="text-gray-300 text-sm leading-[1.9]">
                <InlineText text={afterHead} />
              </p>
            );
          }
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
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-gray-800 border border-gray-700/50 text-[11px] font-bold text-gray-400 flex items-center justify-center mt-0.5">
                      {num}
                    </span>
                    <span className="text-sm text-gray-300 leading-relaxed flex-1">
                      <InlineText text={text} />
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
            <ul key={`ul-${ci}-${mi}-${pi}`} className="space-y-2">
              {items.map((item, k) => {
                const text = item.replace(/^[-•*]\s/, "");
                return (
                  <li key={k} className="flex gap-3 items-start text-sm text-gray-300 leading-relaxed">
                    <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-gray-500 mt-2.5" />
                    <span className="flex-1">
                      <InlineText text={text} />
                    </span>
                  </li>
                );
              })}
            </ul>
          );
          return;
        }

        // Blockquote / Callout box
        if (/^>\s/.test(trimmed)) {
          // Handle multi-line blockquotes (lines starting with ">")
          const bqLines = trimmed.split("\n").map(l => l.replace(/^>\s?/, "")).join(" ").trim();
          const calloutMatch = bqLines.match(/^(💡|💬|🔧|📊|⚠️|🎯|🔬|🤔|✨|🚀|📝|⭐|🧠|🎪)\s*([\s\S]*)/);
          if (calloutMatch) {
            const emoji = calloutMatch[1];
            const body = calloutMatch[2].trim();
            const { bg, border, iconBg, labelColor, label } = CALLOUT_STYLES[emoji] || CALLOUT_STYLES["💡"];
            nodes.push(
              <div key={`callout-${ci}-${mi}-${pi}`}
                className={`flex gap-3.5 rounded-2xl border ${border} ${bg} p-4`}>
                <div className={`flex-shrink-0 w-9 h-9 rounded-xl ${iconBg} flex items-center justify-center text-lg`}>
                  {emoji}
                </div>
                <div className="flex-1 min-w-0">
                  <p className={`text-[10px] font-bold uppercase tracking-widest ${labelColor} mb-1.5`}>{label}</p>
                  <p className="text-sm text-gray-200 leading-relaxed"><InlineText text={body} /></p>
                </div>
              </div>
            );
          } else {
            nodes.push(
              <div key={`bq-${ci}-${mi}-${pi}`}
                className="border-l-2 border-indigo-500/40 bg-indigo-950/15 rounded-r-xl px-4 py-3">
                <p className="text-sm text-indigo-200/80 leading-relaxed italic">
                  <InlineText text={bqLines} />
                </p>
              </div>
            );
          }
          return;
        }

        // Markdown table (lines with | )
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
                          <th key={hi} className="px-4 py-2.5 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider whitespace-nowrap">
                            <InlineText text={h} />
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800/60">
                      {body.map((row, ri) => (
                        <tr key={ri} className="hover:bg-gray-800/20 transition-colors">
                          {parseRow(row).map((cell, ci2) => (
                            <td key={ci2} className={`px-4 py-2.5 text-gray-300 ${ci2 === 0 ? "font-medium text-white" : ""}`}>
                              <InlineText text={cell} />
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

// ── Section color system ──────────────────────────────────────────────────────

const COLOR_MAP: Record<
  string,
  { text: string; border: string; bg: string; iconBg: string; accent: string; ring: string }
> = {
  rose:    { text: "text-rose-300",    border: "border-rose-800/50",    bg: "bg-rose-950/20",    iconBg: "bg-rose-950/60 border-rose-800/60",    accent: "from-rose-500 to-pink-600",    ring: "ring-rose-500/20" },
  sky:     { text: "text-sky-300",     border: "border-sky-800/50",     bg: "bg-sky-950/20",     iconBg: "bg-sky-950/60 border-sky-800/60",     accent: "from-sky-500 to-blue-600",     ring: "ring-sky-500/20" },
  amber:   { text: "text-amber-300",   border: "border-amber-800/50",   bg: "bg-amber-950/20",   iconBg: "bg-amber-950/60 border-amber-800/60",   accent: "from-amber-400 to-orange-500",  ring: "ring-amber-500/20" },
  violet:  { text: "text-violet-300",  border: "border-violet-800/50",  bg: "bg-violet-950/20",  iconBg: "bg-violet-950/60 border-violet-800/60",  accent: "from-violet-500 to-purple-600", ring: "ring-violet-500/20" },
  indigo:  { text: "text-indigo-300",  border: "border-indigo-800/50",  bg: "bg-indigo-950/20",  iconBg: "bg-indigo-950/60 border-indigo-800/60",  accent: "from-indigo-500 to-blue-600",   ring: "ring-indigo-500/20" },
  cyan:    { text: "text-cyan-300",    border: "border-cyan-800/50",    bg: "bg-cyan-950/20",    iconBg: "bg-cyan-950/60 border-cyan-800/60",    accent: "from-cyan-400 to-teal-500",    ring: "ring-cyan-500/20" },
  orange:  { text: "text-orange-300",  border: "border-orange-800/50",  bg: "bg-orange-950/20",  iconBg: "bg-orange-950/60 border-orange-800/60",  accent: "from-orange-500 to-amber-600",  ring: "ring-orange-500/20" },
  emerald: { text: "text-emerald-300", border: "border-emerald-800/50", bg: "bg-emerald-950/20", iconBg: "bg-emerald-950/60 border-emerald-800/60", accent: "from-emerald-400 to-teal-500",  ring: "ring-emerald-500/20" },
  red:     { text: "text-red-300",     border: "border-red-800/50",     bg: "bg-red-950/20",     iconBg: "bg-red-950/60 border-red-800/60",     accent: "from-red-500 to-rose-600",     ring: "ring-red-500/20" },
  purple:  { text: "text-purple-300",  border: "border-purple-800/50",  bg: "bg-purple-950/20",  iconBg: "bg-purple-950/60 border-purple-800/60",  accent: "from-purple-500 to-violet-600", ring: "ring-purple-500/20" },
  lime:    { text: "text-lime-300",    border: "border-lime-800/50",    bg: "bg-lime-950/20",    iconBg: "bg-lime-950/60 border-lime-800/60",    accent: "from-lime-400 to-green-500",   ring: "ring-lime-500/20" },
  teal:    { text: "text-teal-300",    border: "border-teal-800/50",    bg: "bg-teal-950/20",    iconBg: "bg-teal-950/60 border-teal-800/60",    accent: "from-teal-400 to-cyan-500",    ring: "ring-teal-500/20" },
  gray:    { text: "text-gray-300",    border: "border-gray-700/50",    bg: "bg-gray-900/40",    iconBg: "bg-gray-800/80 border-gray-700/60",    accent: "from-gray-500 to-gray-600",    ring: "ring-gray-500/10" },
};

const SECTION_LIST = [
  { key: "🎓 Background & Context",    icon: "🎓", colorKey: "teal",    label: "Background & Context",     subtitle: "Prerequisites and foundational concepts" },
  { key: "🧩 The Problem",             icon: "🧩", colorKey: "rose",    label: "The Problem",              subtitle: "Why this research was needed" },
  { key: "🏛 Prior Work",              icon: "🏛", colorKey: "sky",     label: "Prior Art",                subtitle: "What others tried before" },
  { key: "💡 Core Idea",               icon: "💡", colorKey: "amber",   label: "The Big Idea",             subtitle: "The central insight" },
  { key: "✨ Key Innovations",          icon: "✨", colorKey: "violet",  label: "What's New",               subtitle: "Concrete breakthroughs" },
  { key: "🔢 The Method",              icon: "🔢", colorKey: "indigo",  label: "How It Works",             subtitle: "Technical approach, step by step" },
  { key: "∑ Mathematical Formulation", icon: "∑",  colorKey: "cyan",    label: "The Math",                 subtitle: "Equations & what they mean" },
  { key: "⚙️ Implementation Details",  icon: "⚙️", colorKey: "orange",  label: "Building It",              subtitle: "Datasets, hyperparams, setup" },
  { key: "📊 Results & Benchmarks",    icon: "📊", colorKey: "emerald", label: "Does It Work?",            subtitle: "Numbers, comparisons, ablations" },
  { key: "🔬 Critical Analysis",       icon: "🔬", colorKey: "red",     label: "Honest Critique",          subtitle: "What works, what doesn't" },
  { key: "🤔 Open Questions",          icon: "🤔", colorKey: "purple",  label: "What's Next?",             subtitle: "Open problems & future directions" },
  { key: "🎯 Practical Takeaways",     icon: "🎯", colorKey: "lime",    label: "Use This Now",             subtitle: "Actionable notes for implementers" },
  { key: "💻 Code",                    icon: "💻", colorKey: "gray",    label: "Show Me the Code",         subtitle: "Core implementation" },
] as const;

function getSectionMeta(label: string) {
  const found = SECTION_LIST.find((s) => s.key === label);
  if (!found) return { icon: "📄", color: COLOR_MAP.gray, label, subtitle: "" };
  return { icon: found.icon, color: COLOR_MAP[found.colorKey], label: found.label, subtitle: found.subtitle };
}

// ── Paper hero ────────────────────────────────────────────────────────────────

function PaperHero({ paper }: { paper: Paper }) {
  const [expanded, setExpanded] = useState(false);
  const ABSTRACT_LIMIT = 280;
  const isLong = paper.abstract.length > ABSTRACT_LIMIT;
  const displayAbstract =
    expanded || !isLong
      ? paper.abstract
      : paper.abstract.slice(0, ABSTRACT_LIMIT).trimEnd() + "…";

  const shownAuthors = paper.authors.slice(0, 4);
  const extraAuthors = paper.authors.length - shownAuthors.length;

  const publishedYear = paper.published_at
    ? new Date(paper.published_at).getFullYear()
    : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="mb-10 pb-8 border-b border-gray-800/50"
    >
      {/* Metadata chips */}
      <div className="flex items-center gap-2 mb-5 flex-wrap">
        <span className="px-3 py-1 rounded-full bg-indigo-950/50 border border-indigo-700/30 text-indigo-400 text-[11px] font-semibold">
          Novelty {(paper.novelty_score * 10).toFixed(1)}/10
        </span>
        {publishedYear && (
          <span className="px-3 py-1 rounded-full bg-gray-800/50 border border-gray-700/40 text-gray-500 text-[11px]">
            {publishedYear}
          </span>
        )}
        {paper.namespace_key && (
          <span className="px-3 py-1 rounded-full bg-gray-800/50 border border-gray-700/40 text-gray-500 text-[11px] capitalize">
            {paper.namespace_key.replace(/_/g, " ")}
          </span>
        )}
      </div>

      {/* Title */}
      <h1 className="text-3xl font-bold text-white leading-tight tracking-tight mb-3">
        {paper.title}
      </h1>

      {/* Authors */}
      <p className="text-sm text-gray-500 mb-6">
        {shownAuthors.join(" · ")}
        {extraAuthors > 0 && (
          <span className="text-gray-600"> · +{extraAuthors} more</span>
        )}
      </p>

      {/* TL;DR */}
      {paper.tldr && (
        <div className="flex gap-3.5 p-4 rounded-2xl bg-amber-950/15 border border-amber-700/25 mb-4">
          <span className="text-xl flex-shrink-0 mt-0.5">💡</span>
          <div className="min-w-0">
            <p className="text-[10px] font-bold uppercase tracking-widest text-amber-500 mb-1.5">TL;DR</p>
            <p className="text-sm text-amber-100/75 leading-relaxed">{paper.tldr}</p>
          </div>
        </div>
      )}

      {/* Abstract */}
      <div className="rounded-2xl bg-gray-900/40 border border-gray-800/50 px-5 py-4 mb-5">
        <p className="text-[10px] font-bold uppercase tracking-widest text-gray-600 mb-2.5">Abstract</p>
        <p className="text-sm text-gray-400 leading-[1.85]">{displayAbstract}</p>
        {isLong && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="mt-2.5 flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 transition-colors font-medium"
          >
            {expanded ? (
              <>
                <ChevronUpIcon size={12} /> Show less
              </>
            ) : (
              <>
                <ChevronDownIcon size={12} /> Read more
              </>
            )}
          </button>
        )}
      </div>

      {/* Key concepts + external link */}
      <div className="flex items-center gap-2 flex-wrap">
        {paper.key_concepts.slice(0, 6).map((c) => (
          <span
            key={c}
            className="px-2.5 py-1 rounded-lg bg-gray-800/50 border border-gray-700/40 text-gray-500 text-[11px]"
          >
            {c}
          </span>
        ))}
        {paper.source_url && (
          <a
            href={paper.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-auto flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-gray-800/60 border border-gray-700/50 text-gray-300 hover:text-white hover:border-indigo-600/50 hover:bg-indigo-950/30 transition-all text-xs font-semibold"
          >
            <ExternalLinkIcon size={11} />
            View Paper
          </a>
        )}
        {paper.pdf_url && (
          <a
            href={paper.pdf_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-red-950/30 border border-red-800/40 text-red-400 hover:text-red-300 hover:border-red-700/60 transition-all text-xs font-semibold"
          >
            <ExternalLinkIcon size={11} />
            PDF
          </a>
        )}
      </div>
    </motion.div>
  );
}

// ── Loading animation ─────────────────────────────────────────────────────────

const STUDY_STEPS = [
  { label: "Fetching paper content",  icon: "📄", desc: "Downloading and parsing the PDF" },
  { label: "Extracting structure",    icon: "🔍", desc: "Identifying sections and key components" },
  { label: "Analyzing methodology",  icon: "🧪", desc: "Understanding algorithms and methods" },
  { label: "Generating study guide",  icon: "✍️",  desc: "Writing detailed explanations" },
  { label: "Rendering diagrams",      icon: "🖼",  desc: "Creating architecture visualizations" },
  { label: "Finding related papers",  icon: "🔗", desc: "Searching the knowledge graph" },
];

function StudyLoadingAnimation() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const iv = setInterval(
      () => setStep((s) => Math.min(s + 1, STUDY_STEPS.length - 1)),
      5000
    );
    return () => clearInterval(iv);
  }, []);
  const current = STUDY_STEPS[step];

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-10 py-16">
      {/* Animated orb */}
      <div className="relative w-28 h-28">
        {/* Outer glow */}
        <div className="absolute inset-0 rounded-full bg-indigo-500/10 blur-2xl scale-150 animate-pulse" />
        {/* Spinning arc */}
        <svg
          className="absolute inset-0 w-full h-full animate-spin"
          style={{ animationDuration: "3s" }}
          viewBox="0 0 112 112"
        >
          <circle
            cx="56"
            cy="56"
            r="52"
            fill="none"
            stroke="url(#grad)"
            strokeWidth="2"
            strokeLinecap="round"
            strokeDasharray="220 107"
          />
          <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#6366f1" stopOpacity="0" />
              <stop offset="50%" stopColor="#6366f1" />
              <stop offset="100%" stopColor="#a855f7" />
            </linearGradient>
          </defs>
        </svg>
        {/* Inner circle */}
        <div className="absolute inset-3 rounded-full bg-gray-900 border border-gray-800 flex items-center justify-center shadow-inner">
          <AnimatePresence mode="wait">
            <motion.span
              key={step}
              initial={{ opacity: 0, scale: 0.4, rotate: -20 }}
              animate={{ opacity: 1, scale: 1, rotate: 0 }}
              exit={{ opacity: 0, scale: 0.4, rotate: 20 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="text-4xl"
            >
              {current.icon}
            </motion.span>
          </AnimatePresence>
        </div>
      </div>

      {/* Label */}
      <AnimatePresence mode="wait">
        <motion.div
          key={step}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.3 }}
          className="text-center space-y-1"
        >
          <p className="text-base font-semibold text-white">{current.label}</p>
          <p className="text-sm text-gray-600">{current.desc}</p>
        </motion.div>
      </AnimatePresence>

      {/* Step progress */}
      <div className="flex items-center gap-1.5">
        {STUDY_STEPS.map((_, i) => (
          <motion.div
            key={i}
            animate={{
              width: i === step ? 36 : i < step ? 20 : 8,
              backgroundColor:
                i < step
                  ? "rgb(52 211 153)"
                  : i === step
                  ? "rgb(99 102 241)"
                  : "rgb(55 65 81)",
              opacity: i > step ? 0.5 : 1,
            }}
            transition={{ duration: 0.4, ease: "easeInOut" }}
            className="h-2 rounded-full"
          />
        ))}
      </div>
      <p className="text-xs text-gray-700 font-mono">
        Step {step + 1} of {STUDY_STEPS.length}
      </p>
    </div>
  );
}

// ── Mermaid diagram ───────────────────────────────────────────────────────────

const MIN_DIAGRAM_H = 240;
const MAX_DIAGRAM_H = 540;

// Mermaid is a singleton — initialize once per app session, not per render.
// Calling initialize() on every mount causes state corruption when multiple
// diagrams render simultaneously.
let _mermaidInstance: typeof import("mermaid").default | null = null;
let _mermaidReady = false;
let _mermaidCounter = 0;

async function getMermaid() {
  if (!_mermaidInstance) {
    _mermaidInstance = (await import("mermaid")).default;
  }
  if (!_mermaidReady) {
    _mermaidInstance.initialize({
      startOnLoad: false,
      theme: "dark",
      themeVariables: {
        background: "transparent",
        primaryColor: "#1e1b4b",
        primaryTextColor: "#c7d2fe",
        primaryBorderColor: "#4f46e5",
        lineColor: "#6366f1",
        secondaryColor: "#0f0c29",
        tertiaryColor: "#1a1635",
        edgeLabelBackground: "#1e1b4b",
        nodeTextColor: "#e0e7ff",
        fontSize: "13px",
        fontFamily: "ui-monospace, SFMono-Regular, monospace",
      },
      flowchart: { htmlLabels: true, curve: "basis", padding: 20 },
      securityLevel: "loose",
      // Suppress mermaid's auto-injected "Syntax error in text" bomb SVG.
      // We handle errors via our own fallback UI.
      suppressErrorRendering: true,
    });
    try { (_mermaidInstance as unknown as { parseError?: (...a: unknown[]) => void }).parseError = () => {}; } catch {}
    _mermaidReady = true;
  }
  return _mermaidInstance;
}

/** Pre-validate a spec without throwing or injecting the bomb SVG. */
async function safeMermaidParse(m: typeof import("mermaid").default, source: string): Promise<boolean> {
  try {
    const ok = await m.parse(source, { suppressErrors: true });
    return ok !== false;
  } catch { return false; }
}

function stripMermaidFences(raw: string): string {
  return raw
    .replace(/^```(?:mermaid)?\s*\n?/i, "")
    .replace(/\n?```\s*$/i, "")
    .trim();
}

/** Auto-fix common LLM-generated Mermaid syntax issues before passing to the renderer. */
function sanitizeMermaidSpec(raw: string): string {
  let s = stripMermaidFences(raw).trim();
  // "graph TD/LR" is deprecated in Mermaid 10 — rewrite to "flowchart"
  s = s.replace(/^graph\s+(TD|TB|LR|RL|BT)\b/im, "flowchart $1");
  // Remove stray HTML comment blocks that break the parser
  s = s.replace(/<!--[\s\S]*?-->/g, "");
  // Citation markers [5] / [1-3] / [1, 2] inside node labels collide with mermaid's
  // label terminator. Convert to parens — same visual, no parser ambiguity.
  s = s.replace(/\[(\d+(?:\s*[-,]\s*\d+)*)\]/g, "($1)");
  // Collapse multiple blank lines
  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
}

/**
 * Aggressive fallback: wrap every node label in double quotes so any remaining
 * problematic chars (parens, slashes, ampersands, stray brackets) become literal.
 * Used only after the normal sanitizer + retry both fail.
 */
function quoteAllLabels(raw: string): string {
  function pass(input: string, open: string, close: string): string {
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
            const safe = alreadyQuoted ? content : `"${content.replace(/"/g, "&quot;")}"`;
            out.push(id + open + safe + close);
            i = k + 1;
            continue;
          }
        }
        out.push(input.slice(i, j));
        i = j;
        continue;
      }
      out.push(ch);
      i++;
    }
    return out.join("");
  }
  let s = pass(raw, "[", "]");
  s = pass(s, "(", ")");
  s = pass(s, "{", "}");
  return s;
}

function MermaidDiagram({ spec, maxHeight }: { spec: string; maxHeight?: number }) {
  const ref       = useRef<HTMLDivElement>(null);
  const cancelRef = useRef(false);
  const [error, setError] = useState(false);
  const [height, setHeight] = useState(360);

  useEffect(() => {
    cancelRef.current = false;
    setError(false);

    (async () => {
      if (!ref.current) return;

      const clean = sanitizeMermaidSpec(spec);
      if (!clean) return;

      // Unique, stable ID for this render (avoids collisions with parallel renders)
      const id = `mermaid-${++_mermaidCounter}`;

      let svg = "";
      const mermaid = await getMermaid();
      const tryRender = async (source: string, suffix: string): Promise<string | null> => {
        if (!(await safeMermaidParse(mermaid, source))) return null;
        try {
          const r = await mermaid.render(`${id}${suffix}`, source);
          return r.svg;
        } catch { return null; }
      };

      svg = (await tryRender(clean, "")) ?? "";

      // Retry 1: strip leading prose before the first diagram keyword.
      if (!svg) {
        const match = clean.match(/(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|gitGraph|mindmap|timeline)[\s\S]*/i);
        if (match) svg = (await tryRender(match[0].trim(), "r")) ?? "";
      }

      // Retry 2: quote every node label so stray brackets/parens become literals.
      if (!svg) {
        svg = (await tryRender(quoteAllLabels(clean), "q")) ?? "";
      }

      if (!svg) {
        if (!cancelRef.current) setError(true);
        // Defensive sweep for any bomb SVGs other code paths might have left behind.
        try {
          document.querySelectorAll('svg[aria-roledescription="error"], #mermaid-error-icon').forEach(n => n.remove());
        } catch {}
        return;
      }

      // Guard: component may have unmounted during the await
      if (cancelRef.current || !ref.current) return;

      ref.current.innerHTML = svg;
      const svgEl = ref.current.querySelector("svg");
      if (!svgEl) return;

      // Read natural dimensions from viewBox or explicit attrs
      let nw = 0, nh = 0;
      const vb = svgEl.getAttribute("viewBox");
      if (vb) {
        const parts = vb.trim().split(/[\s,]+/);
        nw = parseFloat(parts[2] || "0");
        nh = parseFloat(parts[3] || "0");
      }
      if (!nw) nw = parseFloat(svgEl.getAttribute("width") || "0");
      if (!nh) nh = parseFloat(svgEl.getAttribute("height") || "0");

      // Scale to container width, clamp height
      const containerW = ref.current.clientWidth || 800;
      const clampMax   = maxHeight ?? MAX_DIAGRAM_H;
      let displayH     = 360;
      if (nw > 0 && nh > 0) {
        displayH = Math.round((containerW / nw) * nh);
        displayH = Math.max(MIN_DIAGRAM_H, Math.min(displayH, clampMax));
      }

      svgEl.setAttribute("width",  "100%");
      svgEl.setAttribute("height", String(displayH));
      svgEl.style.cssText = `width:100%;height:${displayH}px;display:block;`;
      setHeight(displayH);
    })();

    return () => { cancelRef.current = true; };
  }, [spec, maxHeight]);

  if (error) {
    return (
      <pre className="overflow-x-auto text-xs text-gray-500 font-mono leading-relaxed p-4 bg-gray-950/60 rounded-lg border border-gray-800/50">
        {spec}
      </pre>
    );
  }
  return (
    <div
      ref={ref}
      className="overflow-x-auto w-full"
      style={{ height: `${height}px` }}
    />
  );
}

// ── Diagram zoom modal ────────────────────────────────────────────────────────

function DiagramZoomModal({
  spec,
  blobPath,
  caption,
  onClose,
}: {
  spec?: string;
  blobPath?: string;
  caption: string;
  onClose: () => void;
}) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === "Escape") onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-[100] bg-black/92 backdrop-blur-sm flex items-center justify-center p-6"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-5xl max-h-[90vh] overflow-auto rounded-2xl bg-[#080612] border border-indigo-500/30 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-indigo-800/30">
          <span className="text-xs font-semibold text-indigo-300 uppercase tracking-widest">{caption}</span>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-lg leading-none w-7 h-7 flex items-center justify-center rounded-lg hover:bg-gray-800 transition-colors"
          >
            ✕
          </button>
        </div>
        <div className="p-6">
          {spec && <MermaidDiagram spec={stripMermaidFences(spec)} maxHeight={9999} />}
          {blobPath && (
            <img src={`/blobs/${blobPath}`} alt={caption} className="w-full rounded-xl" />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Diagram card ──────────────────────────────────────────────────────────────

function DiagramCard({ section }: { section: StudySection }) {
  const [zoomed, setZoomed] = useState(false);
  const caption = section.caption || "Diagram";
  const kind = section.diagram_kind || "overview";

  const kindMeta: Record<string, { label: string; color: string; dot: string }> = {
    architecture: { label: "Architecture Overview",     color: "text-indigo-300", dot: "bg-indigo-400" },
    overview:     { label: "System Overview",           color: "text-violet-300", dot: "bg-violet-400" },
    algorithm:    { label: "Algorithm Flow",            color: "text-cyan-300",   dot: "bg-cyan-400"   },
    pipeline:     { label: "Data & Training Pipeline",  color: "text-teal-300",   dot: "bg-teal-400"   },
    image:        { label: "Architecture Diagram",      color: "text-amber-300",  dot: "bg-amber-400"  },
    mermaid:      { label: "Architecture Overview",     color: "text-indigo-300", dot: "bg-indigo-400" },
    mermaid_algo: { label: "Algorithm Flow",            color: "text-cyan-300",   dot: "bg-cyan-400"   },
  };
  const meta = kindMeta[kind] || kindMeta.overview;

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="relative rounded-2xl overflow-hidden shadow-2xl shadow-indigo-500/10"
    >
      {/* Glowing border effect */}
      <div className="absolute inset-0 rounded-2xl border border-indigo-500/30 z-10 pointer-events-none" />
      <div className="absolute -inset-0.5 rounded-2xl bg-gradient-to-br from-indigo-500/20 via-violet-500/10 to-purple-500/20 blur-sm -z-10" />

      {/* Dark background with grid */}
      <div className="absolute inset-0 bg-[#080612]" />
      <div
        className="absolute inset-0 opacity-30"
        style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, rgba(99,102,241,0.15) 1px, transparent 0)`,
          backgroundSize: "28px 28px",
        }}
      />
      {/* Subtle center glow */}
      <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/20 via-transparent to-violet-900/10" />

      {/* Header chrome */}
      <div className="relative flex items-center justify-between px-5 py-3 border-b border-indigo-800/30 bg-indigo-950/40 backdrop-blur-sm z-10">
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
            <div className="w-2.5 h-2.5 rounded-full bg-amber-500/60" />
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/60" />
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-1.5 h-1.5 rounded-full ${meta.dot} animate-pulse`} />
            <span className={`text-xs font-semibold uppercase tracking-widest ${meta.color}`}>
              {caption || meta.label}
            </span>
          </div>
        </div>
        <button
          onClick={() => setZoomed(true)}
          title="Expand diagram"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-mono text-indigo-400 hover:text-white hover:bg-indigo-900/50 border border-indigo-700/30 hover:border-indigo-500/50 transition-all"
        >
          <ExternalLinkIcon size={10} />
          {section.blob_path ? "png" : "mermaid"}
        </button>
      </div>

      {/* Content — click anywhere to zoom */}
      <div className="relative p-8 z-10 cursor-zoom-in" onClick={() => setZoomed(true)}>
        {section.spec && <MermaidDiagram spec={stripMermaidFences(section.spec)} />}
        {section.blob_path && (
          <img
            src={`/blobs/${section.blob_path}`}
            alt={caption}
            className="w-full rounded-xl border border-indigo-800/20"
          />
        )}
      </div>

      {/* Zoom modal */}
      {zoomed && (
        <DiagramZoomModal
          spec={section.spec ? stripMermaidFences(section.spec) : undefined}
          blobPath={section.blob_path}
          caption={caption}
          onClose={() => setZoomed(false)}
        />
      )}
    </motion.div>
  );
}

// ── Section block ─────────────────────────────────────────────────────────────

function StudySectionBlock({
  section,
  index,
}: {
  section: StudySection;
  index: number;
}) {
  if (section.type === "diagram") {
    return <DiagramCard section={section} />;
  }

  if (section.type !== "section" || !section.label) return null;

  const { icon, color, label, subtitle } = getSectionMeta(section.label);

  return (
    <motion.div
      id={`section-${index}`}
      initial={{ opacity: 0, y: 28 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="scroll-mt-6"
    >
      {/* Section header */}
      <div className="flex items-center gap-4 mb-5">
        <div className={`relative flex items-center justify-center w-14 h-14 rounded-2xl border ${color.iconBg} flex-shrink-0 shadow-xl overflow-hidden`}>
          <div className={`absolute inset-0 bg-gradient-to-br ${color.accent} opacity-15`} />
          <span className="relative text-2xl z-10">{icon}</span>
        </div>
        <div>
          <h2 className={`text-2xl font-bold ${color.text} tracking-tight leading-tight`}>{label}</h2>
          <p className="text-xs text-gray-600 mt-0.5">{subtitle}</p>
          <div className={`h-0.5 mt-2 w-16 rounded-full bg-gradient-to-r ${color.accent}`} />
        </div>
      </div>

      {/* Content card */}
      <div className={`group relative rounded-2xl border ${color.border} ${color.bg} px-7 py-6 shadow-lg transition-all duration-300 hover:shadow-xl`}>
        <div className={`absolute left-0 top-8 bottom-8 w-0.5 rounded-full bg-gradient-to-b ${color.accent} opacity-50`} />
        <div className="pl-4 space-y-4">
          {renderContent(section.content || "")}
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

function StudyChatPanel({
  paperId,
  level,
  onClose,
}: {
  paperId: string;
  level: string;
  onClose: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "I've loaded the paper and full study guide. Ask me anything — methodology, math, implementation, comparisons, or how to build on this work.",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Abort any in-flight chat stream on unmount so navigating away doesn't
  // leave the fetch hanging or cause state updates on an unmounted tree.
  useEffect(() => () => abortRef.current?.abort(), []);

  async function sendMessage() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);

    const history = messages
      .filter((m) => !m.streaming)
      .slice(-10)
      .map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", streaming: true },
    ]);

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const token = (() => {
        try {
          return (
            JSON.parse(localStorage.getItem("rf_auth") || "{}").state?.token ||
            ""
          );
        } catch {
          return "";
        }
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
          signal: ctrl.signal,
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
      if ((err as { name?: string })?.name !== "AbortError") {
        setMessages((prev) => [
          ...prev.slice(0, -1),
          { role: "assistant", content: "Something went wrong. Try again." },
        ]);
      }
    }
    setBusy(false);
    inputRef.current?.focus();
  }

  return (
    <motion.div
      initial={{ x: "100%", opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0 }}
      transition={{ type: "spring", damping: 30, stiffness: 340 }}
      className="fixed right-0 top-0 bottom-0 w-[420px] bg-gray-950 border-l border-gray-800/70 flex flex-col z-40 shadow-2xl shadow-black/60"
    >
      <div className="flex items-center justify-between px-4 py-3.5 border-b border-gray-800/60 bg-gray-950/95 backdrop-blur-sm">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center">
            <BotIcon size={13} className="text-indigo-400" />
          </div>
          <div>
            <p className="text-xs font-semibold text-white">Paper Q&A</p>
            <p className="text-[10px] text-gray-600">
              Grounded in PDF + study guide
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 p-1.5 rounded-lg hover:bg-gray-800 transition-colors"
        >
          <XIcon size={14} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
          >
            <div
              className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs ${
                msg.role === "user"
                  ? "bg-indigo-600"
                  : "bg-gray-800 border border-gray-700/50"
              }`}
            >
              {msg.role === "user" ? (
                <UserIcon size={12} className="text-white" />
              ) : (
                <BotIcon size={12} className="text-indigo-400" />
              )}
            </div>
            <div
              className={`max-w-[87%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-indigo-600 text-white rounded-tr-sm"
                  : "bg-gray-900 border border-gray-800/60 text-gray-200 rounded-tl-sm"
              }`}
            >
              {msg.role === "user" ? (
                <span className="whitespace-pre-wrap">{msg.content}</span>
              ) : (
                <MarkdownRenderer content={msg.content} />
              )}
              {msg.streaming && (
                <span className="inline-block w-1.5 h-4 bg-indigo-400 rounded-sm animate-pulse ml-0.5 align-middle" />
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="p-3 border-t border-gray-800/60">
        <div className="flex gap-2 items-center bg-gray-900 border border-gray-800 rounded-xl px-3 py-2.5 focus-within:border-indigo-500/50 transition-colors">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Ask anything about this paper…"
            className="flex-1 bg-transparent text-sm text-gray-200 placeholder-gray-600 outline-none"
            disabled={busy}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || busy}
            className="flex-shrink-0 w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center disabled:opacity-40 hover:bg-indigo-500 transition-colors"
          >
            {busy ? (
              <Loader2Icon size={12} className="animate-spin text-white" />
            ) : (
              <SendIcon size={12} className="text-white" />
            )}
          </button>
        </div>
      </div>
    </motion.div>
  );
}

// ── Media Generation Bar ──────────────────────────────────────────────────────

const GEN_BUTTONS: { type: GenerationType; icon: React.ElementType; label: string; color: string }[] = [
  { type: "podcast", icon: MicIcon,          label: "Audio",  color: "violet" },
  { type: "slides",  icon: PresentationIcon, label: "Slides", color: "blue" },
];

const GEN_COLOR_MAP: Record<string, { btn: string; active: string; badge: string }> = {
  violet: {
    btn:    "border-violet-700/40 hover:border-violet-500/60 hover:bg-violet-900/30 text-violet-300",
    active: "bg-violet-600/20 border-violet-500/60 text-violet-200",
    badge:  "bg-violet-600 text-white",
  },
  blue: {
    btn:    "border-blue-700/40 hover:border-blue-500/60 hover:bg-blue-900/30 text-blue-300",
    active: "bg-blue-600/20 border-blue-500/60 text-blue-200",
    badge:  "bg-blue-600 text-white",
  },
};

function MediaGenerationBar({ sourceId, sourceTitle, sourceType }: { sourceId: string; sourceTitle: string; sourceType: GenerationSourceType }) {
  // Media generation is level-independent — grounded in abstract + PDF only.
  // User's orientation/expertise from Settings drives the generation style.
  const [allArtifacts, setAllArtifacts] = React.useState<GeneratedArtifact[]>([]);
  const [loading, setLoading] = React.useState<Record<GenerationType, boolean>>({
    podcast: false, slides: false,
  });
  const [viewer, setViewer] = React.useState<GenerationType | null>(null);
  const addGenerationJob = useJobsStore((s) => s.addGenerationJob);

  // Show the most recent artifact per type regardless of study depth level.
  // allArtifacts is newest-first from the API so the first entry per type wins.
  const artifacts = React.useMemo<Record<GenerationType, GeneratedArtifact | null>>(() => {
    const map: Record<GenerationType, GeneratedArtifact | null> = { podcast: null, slides: null };
    for (const a of allArtifacts) {
      const t = a.generation_type as GenerationType;
      if (!map[t]) map[t] = a;
    }
    return map;
  }, [allArtifacts]);

  // On mount, fetch all existing artifacts for this source (all expertise levels at once).
  React.useEffect(() => {
    if (!sourceId) return;
    api.get<GeneratedArtifact[]>(`/generate/${sourceType}/${sourceId}`)
      .then((arts) => setAllArtifacts(arts))
      .catch(() => {});
  }, [sourceId, sourceType]);

  // Mirror in-flight jobs from the global store into allArtifacts (status updates only).
  // The JobsPanel polls /generate/jobs every 4s — we piggyback on that instead of polling ourselves.
  const fetchedCompletedRef = React.useRef<Set<string>>(new Set());
  const storeGenJobs = useJobsStore((s) => s.generationJobs);
  React.useEffect(() => {
    const relevant = storeGenJobs.filter((gj) => gj.source_id === sourceId);
    if (!relevant.length) return;

    // One-shot DB fetch for completed jobs: blob_path lives only in the DB, not in the JobStore.
    const toFetch: string[] = [];
    for (const gj of relevant) {
      if (
        gj.status === "completed" &&
        !gj.blob_path &&
        gj.artifact_id &&
        !fetchedCompletedRef.current.has(gj.artifact_id)
      ) {
        fetchedCompletedRef.current.add(gj.artifact_id);
        toFetch.push(gj.artifact_id);
      }
    }

    // Update status of matching artifacts without clobbering expertise_level or blob_path.
    setAllArtifacts((prev) => {
      let changed = false;
      const next = prev.map((a) => {
        const gj = relevant.find((j) => j.artifact_id === a.id);
        if (!gj || a.status === gj.status) return a;
        changed = true;
        return {
          ...a,
          status: gj.status as GeneratedArtifact["status"],
          blob_path: gj.blob_path ?? a.blob_path,
          content: gj.content ?? a.content,
          error_message: gj.error_message,
          completed_at: gj.completed_at,
        };
      });
      return changed ? next : prev;
    });

    for (const artifactId of toFetch) {
      api.get<GeneratedArtifact>(`/generate/artifact/${artifactId}`)
        .then((art) => setAllArtifacts((prev) => {
          const idx = prev.findIndex((a) => a.id === art.id);
          if (idx === -1) return [...prev, art];
          const next = [...prev]; next[idx] = art; return next;
        }))
        .catch(() => {});
    }
  }, [storeGenJobs, sourceId]);

  async function trigger(genType: GenerationType, forceRegenerate = false) {
    setLoading((prev) => ({ ...prev, [genType]: true }));
    const params = new URLSearchParams();
    if (forceRegenerate) params.set("force_regenerate", "true");
    // No expertise_level param — backend reads it from the user's profile settings.
    // Media generation is independent of study depth.
    try {
      const resp = await api.post<{ artifact_id: string; job_id: string; status: string; source_title: string; message: string }>(
        `/generate/${sourceType}/${sourceId}/${genType}?${params.toString()}`
      );
      const resolvedTitle = resp.source_title || sourceTitle;

      if (resp.status === "completed") {
        const art = await api.get<GeneratedArtifact>(`/generate/artifact/${resp.artifact_id}`);
        setAllArtifacts((prev) => {
          const idx = prev.findIndex((a) => a.id === art.id);
          if (idx === -1) return [...prev, art];
          const next = [...prev]; next[idx] = art; return next;
        });
      } else {
        const optimistic: GeneratedArtifact = {
          id: resp.artifact_id,
          generation_type: genType,
          source_type: sourceType,
          source_id: sourceId,
          source_title: resolvedTitle,
          status: "queued",
          blob_path: null, content: null,
          expertise_level: null,  // not level-specific
          orientation: null,
          provider: null, model_used: null, input_tokens: 0, output_tokens: 0,
          generation_duration_ms: 0, error_message: null,
          created_at: new Date().toISOString(), completed_at: null,
        };
        setAllArtifacts((prev) => {
          const idx = prev.findIndex((a) => a.id === resp.artifact_id);
          if (idx === -1) return [...prev, optimistic];
          const next = [...prev]; next[idx] = optimistic; return next;
        });
        addGenerationJob({
          artifact_id: resp.artifact_id, job_id: resp.job_id,
          source_type: sourceType, source_id: sourceId,
          generation_type: genType,
          title: resolvedTitle,
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
      <div className="flex items-center gap-2 flex-wrap mt-2 mb-1">
        {GEN_BUTTONS.map(({ type, icon: Icon, label, color }) => {
          const art = artifacts[type];
          const isLoading = loading[type];
          const status = art?.status;
          const colors = GEN_COLOR_MAP[color];
          const isDone = status === "completed";
          const isRunning = status === "queued" || status === "running";
          const isFailed = status === "failed";

          return (
            <div key={type} className="relative">
              <button
                onClick={() => {
                  if (isRunning || isLoading) return;
                  if (isDone) { setViewer(type); return; }
                  trigger(type);
                }}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-semibold border transition-all ${
                  isDone ? colors.active : colors.btn
                }`}
                title={
                  isDone ? `View ${label} (regenerate from inside the viewer)` :
                  isRunning ? `Generating ${label}…` :
                  isFailed ? `${label} previously failed — click to retry` :
                  `Generate ${label}`
                }
              >
                {(isLoading || isRunning) ? (
                  <Loader2Icon size={10} className="animate-spin" />
                ) : (
                  <Icon size={10} />
                )}
                {label}
                {isDone && <span className="ml-0.5 w-1.5 h-1.5 rounded-full bg-emerald-400" />}
                {isFailed && <span className="ml-0.5 w-1.5 h-1.5 rounded-full bg-red-400" />}
              </button>
            </div>
          );
        })}
      </div>

      {/* Viewer modals */}
      <AnimatePresence>
        {viewer && artifacts[viewer]?.status === "completed" && (
          <GenerationViewer
            type={viewer}
            artifact={artifacts[viewer]!}
            onClose={() => setViewer(null)}
            onRegenerate={() => trigger(viewer!, true)}
          />
        )}
      </AnimatePresence>
    </>
  );
}

// ── Generation Viewer Modal ────────────────────────────────────────────────────

function GenerationViewer({
  type,
  artifact,
  onClose,
  onRegenerate,
}: {
  type: GenerationType;
  artifact: GeneratedArtifact;
  onClose: () => void;
  onRegenerate: () => void;
}) {
  const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const blobUrl = artifact.blob_path ? `${API}/blobs/${artifact.blob_path}` : null;
  const content = artifact.content;
  const [regenerating, setRegenerating] = React.useState(false);

  const LABELS: Record<GenerationType, string> = {
    podcast: "🎙 Audio Podcast",
    slides: "📊 Slide Deck",
  };

  async function handleRegenerate() {
    setRegenerating(true);
    try {
      await onRegenerate();
    } finally {
      setRegenerating(false);
      onClose();
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="bg-gray-950 border border-gray-800 rounded-2xl w-full shadow-2xl overflow-hidden flex flex-col max-w-3xl max-h-[85vh]"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800/60">
          <div>
            <h2 className="text-sm font-bold text-white">{LABELS[type]}</h2>
            <p className="text-[11px] text-gray-500 mt-0.5">
              {artifact.model_used} · {artifact.expertise_level} · {artifact.orientation}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* Regenerate button — queues a new background job and closes modal */}
            <button
              onClick={handleRegenerate}
              disabled={regenerating}
              title="Regenerate (background job — overwrites current)"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[11px] font-semibold border border-gray-700/50 text-gray-400 hover:text-indigo-300 hover:border-indigo-500/50 hover:bg-indigo-950/20 transition-all disabled:opacity-40"
            >
              {regenerating ? (
                <Loader2Icon size={11} className="animate-spin" />
              ) : (
                <RefreshCwIcon size={11} />
              )}
              Regenerate
            </button>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-white p-1.5 rounded-lg hover:bg-gray-800 transition-colors"
            >
              <XIcon size={16} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {type === "podcast" && (
            <PodcastViewer blobUrl={blobUrl} content={content} />
          )}
          {type === "slides" && (
            <SlidesViewer blobUrl={blobUrl} content={content} />
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

// ── Individual viewers ─────────────────────────────────────────────────────────

function PodcastViewer({ blobUrl, content }: { blobUrl: string | null; content: Record<string, unknown> | null }) {
  const title = content?.episode_title as string | undefined;
  const tagline = content?.tagline as string | undefined;
  const script = content?.script as string | undefined;

  return (
    <div className="space-y-4">
      {title && <h3 className="text-lg font-bold text-white">{title}</h3>}
      {tagline && <p className="text-sm text-gray-400 italic">{tagline}</p>}

      {blobUrl ? (
        <div>
          <audio controls className="w-full rounded-xl" src={blobUrl}>
            Your browser does not support the audio element.
          </audio>
          <a
            href={blobUrl}
            download
            className="mt-2 inline-flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300"
          >
            <DownloadIcon size={11} /> Download MP3
          </a>
        </div>
      ) : (
        <div className="bg-gray-900 rounded-xl p-3 text-xs text-gray-400">
          Audio file not available. Script preview below.
        </div>
      )}

      {script && (
        <details className="mt-3">
          <summary className="text-xs font-semibold text-gray-400 cursor-pointer hover:text-gray-200">
            View Script
          </summary>
          <pre className="mt-2 text-[11px] text-gray-400 whitespace-pre-wrap leading-relaxed max-h-80 overflow-y-auto bg-gray-900/60 rounded-xl p-3">
            {script}
          </pre>
        </details>
      )}
    </div>
  );
}

function SlidesViewer({ blobUrl, content }: { blobUrl: string | null; content: Record<string, unknown> | null }) {
  const title = content?.deck_title as string | undefined;
  const markdown = content?.marp_markdown as string | undefined;
  // rendered_format may be absent when content is null (store doesn't cache
  // the full content blob). Infer HTML from blob_path extension as a fallback.
  const isHtmlBlob = blobUrl
    ? blobUrl.endsWith(".html") || (content?.rendered_format as string | undefined) === "html"
    : false;

  return (
    <div className="space-y-4">
      {title && <h3 className="text-lg font-bold text-white">{title}</h3>}

      {blobUrl ? (
        <div>
          <iframe
            src={blobUrl}
            className="w-full rounded-xl border border-gray-800"
            style={{ height: "500px" }}
            title="Slide Deck"
          />
          <a
            href={blobUrl}
            download
            className="mt-2 inline-flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300"
          >
            <DownloadIcon size={11} /> {isHtmlBlob ? "Download HTML" : "Download"}
          </a>
        </div>
      ) : markdown ? (
        <div>
          <p className="text-xs text-gray-400 mb-2">
            Marp slides (marp-cli not installed — showing source):
          </p>
          <pre className="text-[11px] text-gray-300 whitespace-pre-wrap leading-relaxed max-h-[500px] overflow-y-auto bg-gray-900/60 rounded-xl p-4">
            {markdown}
          </pre>
        </div>
      ) : (
        <p className="text-sm text-gray-400">Slides content unavailable.</p>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

function StudyContent() {
  const { id } = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();
  const { token, user } = useAuthStore();
  // Default to the user's configured expertise level; fall back to "practitioner"
  const defaultLevel = (user?.expertise_level as "newcomer" | "practitioner" | "expert") || "practitioner";
  const level = (searchParams.get("level") as "newcomer" | "practitioner" | "expert") || defaultLevel;

  // Per-level section cache. Switching levels uses the cached sections
  // immediately (no SSE restart) if we already loaded that level before.
  const [levelCache, setLevelCache] = useState<
    Partial<Record<string, { sections: StudySection[]; status: "done" | "error" }>>
  >({});
  const [paper, setPaper] = useState<Paper | null>(null);
  const [sections, setSections] = useState<StudySection[]>([]);
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [showChat, setShowChat] = useState(false);
  const [readPct, setReadPct] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    setReadPct(
      Math.min(100, (el.scrollTop / (el.scrollHeight - el.clientHeight)) * 100)
    );
  }

  // Fetch paper metadata upfront
  useEffect(() => {
    if (!id || !token) return;
    const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${API}/api/v1/papers/${id}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => { if (data) setPaper(data); })
      .catch(() => {});
  }, [id, token]);

  useEffect(() => {
    if (!id || !token) return;

    // If we already loaded this level in this session, restore from cache
    // immediately without re-opening an SSE connection.
    const cached = levelCache[level];
    if (cached) {
      setSections(cached.sections);
      setStatus(cached.status);
      return;
    }

    setSections([]);
    setStatus("loading");
    setShowChat(false);

    const incoming: StudySection[] = [];
    const es = openSSE(`/study/${id}?expertise_level=${level}`);

    // Hard timeout: if the backend hangs and never sends "done", close the
    // connection after 8 minutes and surface an error so the user isn't left
    // with a permanent spinner.
    const timeout = setTimeout(() => {
      es.close();
      setStatus("error");
    }, 8 * 60 * 1000);

    es.onmessage = (e) => {
      try {
        const data: StudySection = JSON.parse(e.data);
        if (data.type === "done") {
          clearTimeout(timeout);
          setLevelCache((prev) => ({
            ...prev,
            [level]: { sections: incoming.slice(), status: "done" },
          }));
          setStatus("done");
          es.close();
        } else if (data.type === "error") {
          clearTimeout(timeout);
          setStatus("error");
          es.close();
        } else if (data.type !== "start" && data.type !== "related") {
          incoming.push(data);
          setSections((prev) => [...prev, data]);
          bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        }
      } catch {}
    };
    es.onerror = () => {
      clearTimeout(timeout);
      setStatus("error");
      es.close();
    };
    return () => {
      clearTimeout(timeout);
      es.close();
    };
  }, [id, level, token]);

  const sectionCount = sections.filter((s) => s.type === "section").length;

  const navItems = useMemo(() => {
    return sections
      .map((s, i) => ({ s, i }))
      .filter(({ s }) => s.type === "section" && s.label)
      .map(({ s, i }) => {
        const meta = getSectionMeta(s.label!);
        return { id: `section-${i}`, label: meta.label, icon: meta.icon };
      });
  }, [sections]);

  return (
    <div className="flex h-full" style={{ background: "var(--rf-bg)" }}>
      {/* Section navigation panel — shown once generation is complete */}
      {status === "done" && navItems.length > 0 && (
        <SectionNavPanel items={navItems} scrollRef={scrollRef} accent="indigo" />
      )}

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className={`flex-1 overflow-y-auto transition-all duration-300 ${
          showChat ? "mr-[420px]" : ""
        }`}
        style={{ background: "var(--rf-bg)" }}
      >
        {/* Reading progress bar */}
        <div className="sticky top-0 left-0 right-0 h-0.5 bg-gray-800/40 z-20">
          <motion.div
            className="h-full bg-gradient-to-r from-indigo-500 via-violet-500 to-fuchsia-500"
            animate={{ width: `${readPct}%` }}
            transition={{ duration: 0.1 }}
          />
        </div>

        <div className="max-w-5xl mx-auto px-8 py-8">
          {/* Paper hero */}
          {paper && <PaperHero paper={paper} />}

          {/* Header bar */}
          <div className="sticky top-0.5 z-10 mb-8">
            <div className="flex items-center gap-3 bg-gray-900/80 backdrop-blur-md border border-gray-800/60 rounded-2xl px-4 py-2.5 shadow-lg shadow-black/20">
              {/* Level tabs — use router.replace so the URL updates without a
                  full page reload. The MediaGenerationBar stays mounted; each
                  level has independent cached artifacts so switching is instant. */}
              <div className="flex bg-gray-800/60 rounded-xl p-0.5 gap-0.5">
                {(["newcomer", "practitioner", "expert"] as const).map((l) => (
                  <button
                    key={l}
                    onClick={() => {
                      if (l === level) return;
                      router.replace(`/study/${id}?level=${l}`, { scroll: false });
                    }}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      level === l
                        ? "bg-indigo-600 text-white shadow-sm shadow-indigo-500/30"
                        : "text-gray-500 hover:text-gray-200"
                    }`}
                  >
                    {l.charAt(0).toUpperCase() + l.slice(1)}
                  </button>
                ))}
              </div>

              {/* Status */}
              {status === "loading" && (
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Loader2Icon size={11} className="animate-spin" />
                  <span>
                    {sectionCount > 0
                      ? `${sectionCount} sections`
                      : "Generating…"}
                  </span>
                </div>
              )}
              {status === "done" && (
                <div className="flex items-center gap-1.5 text-xs text-emerald-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  Complete
                </div>
              )}

              {/* Generate media buttons */}
              {status === "done" && id && (
                <div className="flex items-center">
                  <MediaGenerationBar sourceId={id} sourceTitle={paper?.title ?? ""} sourceType="paper" />
                </div>
              )}

              {/* Chat button */}
              {status === "done" && (
                <button
                  onClick={() => setShowChat((v) => !v)}
                  className={`ml-auto flex items-center gap-2 px-4 py-1.5 rounded-xl text-xs font-semibold transition-all ${
                    showChat
                      ? "bg-indigo-600 text-white"
                      : "bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700/50"
                  }`}
                >
                  <MessageSquareIcon size={12} />
                  {showChat ? "Close Chat" : "Ask Questions"}
                </button>
              )}
            </div>
          </div>

          {/* Loading animation (shown only when no sections yet) */}
          {status === "loading" && sections.length === 0 && (
            <StudyLoadingAnimation />
          )}

          {/* Sections */}
          <div className="space-y-8">
            <AnimatePresence initial={false}>
              {sections.map((section, i) => (
                <StudySectionBlock key={i} section={section} index={i} />
              ))}
            </AnimatePresence>
          </div>

          {/* Error state */}
          {status === "error" && (
            <div className="mt-8 bg-red-950/50 border border-red-800/60 rounded-2xl p-5 text-red-300 text-sm">
              Couldn&apos;t load this Study. Try again or switch expertise level.
            </div>
          )}

          <div ref={bottomRef} className="h-16" />
        </div>
      </div>

      <AnimatePresence>
        {showChat && id && (
          <StudyChatPanel
            paperId={id}
            level={level}
            onClose={() => setShowChat(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default function StudyPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-full text-gray-500">
          <Loader2Icon size={20} className="animate-spin mr-2" />
          Loading…
        </div>
      }
    >
      <StudyContent />
    </Suspense>
  );
}
