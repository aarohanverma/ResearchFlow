"use client";

/**
 * Floating "Ask on this" popover that surfaces next to a text
 * selection inside any element marked ``data-rf-quotable="1"``.
 * Clicking the popover hands the selection back to the parent via
 * the ``onAsk`` callback — typically the parent populates a
 * "quoted context" state that its own composer renders as a quote
 * chip above the input.
 *
 * Used in three chat surfaces:
 *   1. Assistant chat (main RA chat)
 *   2. Study Mode chat (PaperPanel's inline RAG chat about a paper)
 *   3. Genie Idea Dive chat (CapsuleChatOverlay)
 *
 * Each chat marks its message bodies with ``data-rf-quotable="1"``
 * so the popover only fires inside the chat — not on sidebars,
 * composers, or other unrelated UI.
 *
 * Multiple instances mounted simultaneously are SAFE — each
 * instance owns its own onAsk callback. The shared selectionchange
 * listener will fire on every instance, but only one popover will
 * actually be visible at a time (the one whose data-rf-quotable
 * ancestor matched the selection). For the rare case where multiple
 * chats are visible (a Study chat overlay on top of the assistant),
 * the topmost (highest z-index) wins visually.
 */

import { useEffect, useState } from "react";
import { SparklesIcon } from "lucide-react";

const MIN_LEN = 8;
const QUOTABLE_ATTR = "data-rf-quotable";

/** Optional label override. The default "Ask on this" reads naturally
 *  inside the assistant; Study Mode + Idea Dive can opt for a
 *  context-specific label (e.g. "Ask about this paper").
 *
 *  ``scope`` filters which quotable subtrees this instance reacts to.
 *  When multiple chats are open simultaneously (e.g. PaperPanel
 *  layered on the assistant), without a scope filter each popover
 *  would fire for selections in EITHER subtree and the user might
 *  click the wrong popover, routing the quote to the wrong
 *  composer. With scopes, each popover only matches selections
 *  whose innermost ``data-rf-quotable`` element matches its scope.
 *
 *  Backward compat: when ``scope`` is omitted, the popover matches
 *  any ``data-rf-quotable`` value (including the legacy ``"1"``).
 *  Existing code that hasn't been updated keeps working.
 */
export function AskOnSelectionPopover({
  onAsk, label = "Ask on this", scope,
}: {
  onAsk: (text: string) => void;
  label?: string;
  scope?: string;
}) {
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const [text, setText] = useState<string>("");

  useEffect(() => {
    function isInsideQuotable(node: Node | null): boolean {
      let el: Node | null = node;
      while (el && el.nodeType !== Node.ELEMENT_NODE) el = el.parentNode;
      let cur = el as HTMLElement | null;
      // Walk UP from the selection. The FIRST quotable ancestor we
      // find is the "owning" subtree. We only fire when that
      // innermost owner matches our scope — without this check,
      // multiple stacked popovers would all fire and the user
      // could route a quote to the wrong composer.
      while (cur) {
        if (cur.getAttribute) {
          const attr = cur.getAttribute(QUOTABLE_ATTR);
          if (attr) {
            if (!scope) return true;            // no scope → match any
            return attr === scope;              // strict match
          }
        }
        cur = cur.parentElement;
      }
      return false;
    }

    function update() {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) { setPos(null); return; }
      const raw = sel.toString().trim();
      if (raw.length < MIN_LEN) { setPos(null); return; }
      const anchor = sel.anchorNode;
      const focus = sel.focusNode;
      if (!isInsideQuotable(anchor) && !isInsideQuotable(focus)) {
        setPos(null);
        return;
      }
      const range = sel.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) { setPos(null); return; }
      // The popover root uses ``position: fixed`` so coordinates must
      // be VIEWPORT-relative — getBoundingClientRect already returns
      // viewport-relative numbers. Adding window.scrollY / scrollX
      // (as the earlier version did, copy-pasted from an
      // ``position: absolute`` snippet) shifted the popover off-
      // screen by exactly the scroll offset. That bug is what made
      // the feature appear "broken" inside Study Mode and the Genie
      // Idea Dive chat panels — both surfaces scroll the page and/or
      // the chat container, so any non-zero scroll sent the popover
      // out of view.
      const top = Math.max(8, rect.top - 36);
      const left = Math.min(
        window.innerWidth - 160,
        Math.max(8, rect.left + rect.width / 2 - 70),
      );
      setText(raw);
      setPos({ top, left });
    }

    function onScrollOrResize() { setPos(null); }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setPos(null);
    }
    // Debounce ``selectionchange`` to ~80ms so the popover only
    // resolves once dragging stops. Without it, the ancestor walk +
    // bounding-rect read fires on every mouse move during a drag.
    let timer: ReturnType<typeof setTimeout> | null = null;
    function onSelectionChange() {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => { timer = null; update(); }, 80);
    }
    function onMouseDown(e: MouseEvent) {
      const target = e.target as HTMLElement | null;
      if (target && target.closest && target.closest("[data-rf-ask-popover='1']")) return;
      setTimeout(() => {
        const sel = window.getSelection();
        if (!sel || sel.isCollapsed) setPos(null);
      }, 0);
    }

    document.addEventListener("selectionchange", onSelectionChange);
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKeyDown);
    window.addEventListener("scroll", onScrollOrResize, true);
    window.addEventListener("resize", onScrollOrResize);
    return () => {
      if (timer) { clearTimeout(timer); timer = null; }
      document.removeEventListener("selectionchange", onSelectionChange);
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("scroll", onScrollOrResize, true);
      window.removeEventListener("resize", onScrollOrResize);
    };
  }, []);

  if (!pos || !text) return null;
  return (
    <div
      data-rf-ask-popover="1"
      onMouseDown={(e) => e.preventDefault()}
      style={{
        position: "fixed",
        top: pos.top, left: pos.left,
        zIndex: 1000,
        background: "var(--rf-surface3, rgba(20,20,30,0.96))",
        border: "1px solid var(--rf-border, rgba(99,102,241,0.4))",
        borderRadius: 8,
        boxShadow: "0 8px 24px rgba(0,0,0,0.4)",
        padding: "5px 9px",
        display: "flex", alignItems: "center", gap: 6,
        color: "var(--rf-text1, #f3f4f6)",
        fontSize: "11px",
        cursor: "pointer",
        userSelect: "none",
      }}
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (text) onAsk(text);
        setPos(null);
        const sel = window.getSelection();
        if (sel) sel.removeAllRanges();
      }}
      title="Quote this selection in your next message"
    >
      <SparklesIcon size={11} color="#a3a3ff" />
      <span style={{ fontWeight: 600 }}>{label}</span>
    </div>
  );
}
