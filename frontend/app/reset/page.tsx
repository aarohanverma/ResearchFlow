"use client";

/**
 * /reset — clear ALL persisted client-side state so the user gets a
 * truly fresh start. Wipes localStorage, sessionStorage, IndexedDB,
 * caches, and service-worker registrations, then redirects to /login.
 *
 * Useful when stale persisted state from an earlier app version is
 * causing render crashes that survive a normal page reload.
 */

import { useEffect, useState } from "react";

export default function ResetPage() {
  const [stage, setStage] = useState("Wiping local data…");
  const [done, setDone] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        // 1) localStorage — every Zustand persist key + any of our own
        try { localStorage.clear(); } catch { /* ignore quota */ }

        // 2) sessionStorage — anything page-scoped
        try { sessionStorage.clear(); } catch { /* ignore */ }

        // 3) IndexedDB — Next.js app router and any libraries
        try {
          if (typeof indexedDB !== "undefined" && "databases" in indexedDB) {
            const dbs: { name?: string }[] = await (
              indexedDB as unknown as { databases: () => Promise<{ name?: string }[]> }
            ).databases();
            await Promise.all(
              dbs
                .map((d) => d.name)
                .filter((n): n is string => !!n)
                .map(
                  (name) =>
                    new Promise<void>((resolve) => {
                      const req = indexedDB.deleteDatabase(name);
                      req.onsuccess = req.onerror = req.onblocked = () => resolve();
                    })
                )
            );
          }
        } catch { /* ignore — older browsers don't expose .databases() */ }

        // 4) Cache Storage — service worker / Next.js asset caches
        try {
          if (typeof caches !== "undefined") {
            const keys = await caches.keys();
            await Promise.all(keys.map((k) => caches.delete(k)));
          }
        } catch { /* ignore */ }

        // 5) Service worker registrations
        try {
          if (typeof navigator !== "undefined" && "serviceWorker" in navigator) {
            const regs = await navigator.serviceWorker.getRegistrations();
            await Promise.all(regs.map((r) => r.unregister()));
          }
        } catch { /* ignore */ }

        // 6) Cookies for this origin (best-effort — only non-HttpOnly)
        try {
          for (const c of document.cookie.split(";")) {
            const eq = c.indexOf("=");
            const name = (eq > -1 ? c.slice(0, eq) : c).trim();
            document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
          }
        } catch { /* ignore */ }

        setStage("Cleared. Redirecting…");
        setDone(true);
        // Hard redirect with cache-bust so we don't pick up any in-memory
        // module-level state that survived the wipe.
        setTimeout(() => {
          window.location.replace("/login?fresh=1");
        }, 600);
      } catch (err) {
        setStage(`Reset failed: ${(err as Error)?.message || "unknown error"}`);
      }
    })();
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#0d1117",
        color: "#e8e8f0",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <div
        style={{
          maxWidth: 380,
          padding: 28,
          background: "#161b22",
          border: "1px solid #30363d",
          borderRadius: 16,
          textAlign: "center",
        }}
      >
        <div
          style={{
            width: 44,
            height: 44,
            borderRadius: 10,
            margin: "0 auto 16px",
            background: "linear-gradient(135deg,#6366f1,#7c3aed)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 20,
          }}
        >
          {done ? "✓" : "⟳"}
        </div>
        <h1 style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>
          ResearchFlow — Fresh Start
        </h1>
        <p style={{ fontSize: 13, color: "#9ca3af" }}>{stage}</p>
        <p style={{ fontSize: 11, color: "#6b7280", marginTop: 16 }}>
          You will be sent to the login page.
        </p>
      </div>
    </div>
  );
}
