"use client";

import { useEffect, useRef, useState } from "react";
import { CopyIcon, CheckIcon } from "lucide-react";

// ── Mermaid diagram ───────────────────────────────────────────────────────────

/**
 * LLM-generated mermaid often has small syntax mistakes that mermaid's
 * tokenizer can't recover from. We do a best-effort cleanup pass so a single
 * bad node doesn't kill the whole diagram. All transforms are conservative
 * (no semantic edits) and idempotent.
 */
export function sanitizeMermaidSpec(raw: string): string {
  let s = raw;

  // Strip ```mermaid fences if the LLM included them inside the block.
  s = s.replace(/^\s*```(?:mermaid)?\s*\n?/, "").replace(/\n?```\s*$/, "");

  // Citation markers like [5], [1-3], [1, 2, 3] inside node labels collide
  // with mermaid's label terminator. Convert them to parens — same visual,
  // no parser ambiguity. Standalone `[N]` never has a legal meaning in
  // mermaid source so this is safe everywhere.
  s = s.replace(/\[(\d+(?:\s*[-,]\s*\d+)*)\]/g, "($1)");

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
          flowchart: { htmlLabels: true, curve: "basis" },
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

export function InlineText({ text }: { text: string }) {
  const normalised = text.replace(/\\\(([\s\S]+?)\\\)/g, (_m, e) => `$${e}$`);
  const parts = normalised.split(
    /(\*\*[^*\n]+\*\*|\*[^*\n]+\*|`[^`\n]+`|\$[^$\n]{1,80}\$)/g
  );
  return (
    <>
      {parts.map((part, i) => {
        if (/^\*\*[^*]+\*\*$/.test(part))
          return <strong key={i} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
        if (/^\*[^*\n]+\*$/.test(part) && !part.startsWith("**"))
          return <em key={i} className="italic text-gray-300">{part.slice(1, -1)}</em>;
        if (/^`[^`]+`$/.test(part))
          return (
            <code key={i} className="px-1.5 py-0.5 rounded-md bg-gray-800 border border-gray-700/60 text-[12px] font-mono text-indigo-300 mx-0.5">
              {part.slice(1, -1)}
            </code>
          );
        if (/^\$[^$]+\$$/.test(part)) {
          // Inline KaTeX — lazy import to avoid SSR issues
          return <InlineMath key={i} expr={part.slice(1, -1)} />;
        }
        return part;
      })}
    </>
  );
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
  if (!html) return <span className="font-mono text-amber-300/80">${expr}$</span>;
  return <span dangerouslySetInnerHTML={{ __html: html }} className="mx-0.5" />;
}

// ── Full markdown renderer ────────────────────────────────────────────────────

export function renderMarkdown(rawContent: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];

  // Normalise LaTeX delimiters → $ syntax
  const content = rawContent
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
              <h1 key={key} className="text-2xl font-bold text-white mt-8 mb-3">
                <InlineText text={headText} />
              </h1>
            );
          } else if (level === 2) {
            nodes.push(
              <div key={key} className="mt-10 mb-4 pb-2 border-b border-gray-700/60">
                <h2 className="text-xl font-bold text-white leading-snug"><InlineText text={headText} /></h2>
              </div>
            );
          } else if (level === 3) {
            nodes.push(
              <h3 key={key} className="text-sm font-semibold text-gray-300 uppercase tracking-widest mt-6 mb-2">
                <InlineText text={headText} />
              </h3>
            );
          } else {
            nodes.push(
              <p key={key} className="text-sm font-medium text-gray-400 mt-3 mb-1">
                <InlineText text={headText} />
              </p>
            );
          }
          if (afterHead) {
            nodes.push(
              <p key={`${key}-body`} className="text-sm text-gray-300 leading-[1.85]">
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
            <ol key={key} className="space-y-1.5 my-1">
              {items.map((item, k) => {
                const m = item.match(/^(\d+)\.\s([\s\S]*)$/);
                const num = m ? m[1] : String(k + 1);
                const text = m ? m[2] : item;
                return (
                  <li key={k} className="flex gap-2.5 items-start">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-gray-800 border border-gray-700/50 text-[10px] font-bold text-gray-400 flex items-center justify-center mt-0.5">
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
            <ul key={key} className="space-y-1.5 my-1">
              {items.map((item, k) => {
                const text = item.replace(/^[-•*]\s/, "");
                return (
                  <li key={k} className="flex gap-2.5 items-start text-sm text-gray-300 leading-relaxed">
                    <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-indigo-400/60 mt-2.5" />
                    <span className="flex-1"><InlineText text={text} /></span>
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
                <InlineText text={bq} />
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
                <div key={key} className="overflow-x-auto rounded-xl border border-gray-700/50 my-2">
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
          <p key={key} className="text-sm text-gray-300 leading-[1.85]">
            <InlineText text={trimmed} />
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
}: {
  content: string;
  className?: string;
}) {
  const nodes = renderMarkdown(content);
  return (
    <div className={`space-y-3 min-w-0 ${className}`}>
      {nodes}
    </div>
  );
}
