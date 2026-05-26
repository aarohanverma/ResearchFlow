"use client";

/**
 * Persisted, drag-resizable width hook.
 *
 * Returns the current width plus a mousedown handler that starts a
 * drag. Width is persisted to ``localStorage`` under the supplied
 * ``storageKey`` so it survives reloads — without the persistence
 * users would have to re-establish their layout every visit.
 *
 * Bounds (``minWidth`` / ``maxWidth``) are enforced during drag so a
 * pathological mouse movement can't collapse the panel to zero or
 * eat the entire viewport. Double-click handler returns the panel
 * to its default width — escape hatch for layouts the user
 * accidentally squished.
 *
 * The handler swaps the body cursor to ``ew-resize`` and disables
 * text selection during the drag so the UI feels right; both are
 * restored on mouseup or unmount.
 *
 * Two anchors are supported because some panels are anchored to the
 * RIGHT edge (drag-left widens) and others to the LEFT (drag-right
 * widens). ``anchor: "left"`` is the default (most sidebars).
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface UseResizableWidthOptions {
  /** localStorage key — pick something stable per panel. */
  storageKey: string;
  /** Default width applied on first mount or after reset. */
  defaultWidth: number;
  /** Minimum width during drag — typically 180-280. */
  minWidth: number;
  /** Maximum width during drag — typically 480-1100 depending on layout. */
  maxWidth: number;
  /**
   * Which edge the panel is anchored to. ``"left"`` (default) means
   * the panel sits on the left side and a rightward drag widens it;
   * ``"right"`` means the panel sits on the right side and a leftward
   * drag widens it.
   */
  anchor?: "left" | "right";
}

export interface UseResizableWidthResult {
  width: number;
  onResizeStart: (e: React.MouseEvent) => void;
  reset: () => void;
}

export function useResizableWidth({
  storageKey,
  defaultWidth,
  minWidth,
  maxWidth,
  anchor = "left",
}: UseResizableWidthOptions): UseResizableWidthResult {
  const [width, setWidth] = useState<number>(() => {
    if (typeof window === "undefined") return defaultWidth;
    try {
      const raw = parseInt(localStorage.getItem(storageKey) || "", 10);
      if (Number.isFinite(raw) && raw >= minWidth && raw <= maxWidth) return raw;
    } catch {}
    return defaultWidth;
  });

  // Ref keeps the latest width visible to listeners that captured an
  // older closure — otherwise mouseup persists the stale starting
  // width to localStorage.
  const widthRef = useRef<number>(width);
  useEffect(() => {
    widthRef.current = width;
  }, [width]);

  const onResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = widthRef.current;
    const dir = anchor === "right" ? -1 : 1;
    const onMove = (ev: MouseEvent) => {
      const dx = (ev.clientX - startX) * dir;
      const next = Math.min(maxWidth, Math.max(minWidth, startW + dx));
      setWidth(next);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      try { localStorage.setItem(storageKey, String(widthRef.current)); } catch {}
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "ew-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [anchor, maxWidth, minWidth, storageKey]);

  const reset = useCallback(() => {
    setWidth(defaultWidth);
    try { localStorage.setItem(storageKey, String(defaultWidth)); } catch {}
  }, [defaultWidth, storageKey]);

  return { width, onResizeStart, reset };
}

/**
 * Shared inline style for the thin invisible strip that triggers a
 * resize drag. Pin it absolutely to whichever edge of the panel
 * faces the resize direction. Width 6 px (3 inside, 3 outside) keeps
 * the hit area generous without showing chrome until hover.
 */
export const RESIZE_HANDLE_STYLE: React.CSSProperties = {
  position: "absolute",
  top: 0,
  bottom: 0,
  width: 6,
  cursor: "ew-resize",
  zIndex: 20,
};
