"use client";

import { AnimatePresence, motion } from "framer-motion";
import { CheckCircleIcon, XCircleIcon, XIcon } from "lucide-react";
import { useToastState } from "@/hooks/use-toast";

/** Toaster — theme-aware (uses CSS variables) so toasts render correctly in
 *  both light and dark mode. Auto-fades after 3.5s, dismissable via X. */
export function Toaster() {
  const { toasts, dismiss } = useToastState();

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      <AnimatePresence>
        {toasts.map((t) => {
          const isError = t.variant === "error";
          const isSuccess = t.variant === "success";
          // Variant-specific accent colors layered over theme-aware background.
          // CSS vars (--rf-surface1, --rf-text1, --rf-text4, --rf-border) are
          // defined in globals.css per [data-theme="light"|"dark"].
          const bg = isError
            ? "color-mix(in srgb, #b91c1c 16%, var(--rf-surface1))"
            : isSuccess
            ? "color-mix(in srgb, #047857 16%, var(--rf-surface1))"
            : "var(--rf-surface1)";
          const borderColor = isError
            ? "rgba(239,68,68,0.45)"
            : isSuccess
            ? "rgba(16,185,129,0.45)"
            : "var(--rf-border)";
          return (
            <motion.div
              key={t.id}
              role="status"
              initial={{ opacity: 0, y: 16, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.96 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
              style={{
                background: bg,
                border: `1px solid ${borderColor}`,
                color: "var(--rf-text1)",
                boxShadow: "0 12px 32px rgba(0,0,0,0.18)",
              }}
              className="pointer-events-auto flex items-start gap-3 min-w-[280px] max-w-sm px-4 py-3.5 rounded-xl"
            >
              {isSuccess && <CheckCircleIcon size={16} style={{ color: "#10b981", marginTop: 2, flexShrink: 0 }} />}
              {isError && <XCircleIcon size={16} style={{ color: "#ef4444", marginTop: 2, flexShrink: 0 }} />}
              <div className="flex-1 min-w-0">
                <p style={{ fontSize: "13px", fontWeight: 600, color: "var(--rf-text1)" }}>{t.title}</p>
                {t.description && (
                  <p style={{ fontSize: "11.5px", color: "var(--rf-text4)", marginTop: 2 }}>{t.description}</p>
                )}
              </div>
              <button
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss notification"
                style={{ color: "var(--rf-text5)", background: "none", border: "none", cursor: "pointer", flexShrink: 0 }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.color = "var(--rf-text2)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.color = "var(--rf-text5)"; }}
              >
                <XIcon size={14} />
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
