"use client";

/**
 * Next.js App Router error boundary for the (auth) and (app) route groups.
 *
 * Any uncaught render error inside a page lands here. The user sees a
 * friendly recovery card instead of a blank white tab; the dev console
 * still gets the full stack trace via `console.error`.
 *
 * This is the officially-supported Next.js pattern — it plays nicely
 * with Fast Refresh / HMR (a class-based ErrorBoundary does not).
 */

import { useEffect } from "react";

export default function GlobalRouteError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.error("[App Error]", error);
  }, [error]);

  function hardReset() {
    try {
      // Clear most-likely-stale persisted stores (auth preserved so
      // the user doesn't have to log back in).
      localStorage.removeItem("research-flow-jobs");
      localStorage.removeItem("rf_bookmarks");
    } catch {
      // ignore — quota / privacy modes
    }
    if (typeof window !== "undefined") window.location.reload();
  }

  return (
    <div
      style={{
        minHeight: "60vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 32,
        background: "var(--rf-bg, #0d1117)",
        color: "var(--rf-text1, #e8e8f0)",
      }}
    >
      <div
        style={{
          maxWidth: 460,
          background: "var(--rf-surface3, #161b22)",
          border: "1px solid var(--rf-border, #30363d)",
          borderRadius: 16,
          padding: 28,
        }}
      >
        <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>
          Something went wrong
        </h2>
        <p
          style={{
            fontSize: 13,
            color: "var(--rf-text3, #9ca3af)",
            marginBottom: 18,
          }}
        >
          The page hit an unexpected error. Your data on the server is safe.
          Retry the page, or do a clean reload if the problem repeats.
        </p>
        <pre
          style={{
            fontSize: 11,
            background: "rgba(0,0,0,0.4)",
            padding: 10,
            borderRadius: 8,
            maxHeight: 140,
            overflow: "auto",
            marginBottom: 18,
            whiteSpace: "pre-wrap",
            color: "#f87171",
          }}
        >
          {error.message || String(error)}
        </pre>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => reset()}
            style={{
              flex: 1,
              padding: "10px 14px",
              borderRadius: 10,
              background: "linear-gradient(135deg,#6366f1,#7c3aed)",
              color: "white",
              fontWeight: 600,
              fontSize: 13,
              border: "none",
              cursor: "pointer",
            }}
          >
            Retry
          </button>
          <button
            onClick={hardReset}
            style={{
              flex: 1,
              padding: "10px 14px",
              borderRadius: 10,
              background: "transparent",
              border: "1px solid var(--rf-border, #30363d)",
              color: "var(--rf-text2, #d1d5db)",
              fontWeight: 600,
              fontSize: 13,
              cursor: "pointer",
            }}
          >
            Clean reload
          </button>
        </div>
      </div>
    </div>
  );
}
