"use client";

import { AnimatePresence, motion } from "framer-motion";
import { CheckCircleIcon, XCircleIcon, XIcon } from "lucide-react";
import { useToastState } from "@/hooks/use-toast";

export function Toaster() {
  const { toasts, dismiss } = useToastState();

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      <AnimatePresence>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, y: 16, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.96 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className={`pointer-events-auto flex items-start gap-3 min-w-[280px] max-w-sm px-4 py-3.5 rounded-xl border shadow-xl shadow-black/50 ${
              t.variant === "error"
                ? "bg-red-950/90 border-red-800/60"
                : t.variant === "success"
                ? "bg-emerald-950/90 border-emerald-800/60"
                : "bg-gray-900 border-gray-700/60"
            }`}
          >
            {t.variant === "success" && <CheckCircleIcon size={16} className="text-emerald-400 mt-0.5 flex-shrink-0" />}
            {t.variant === "error" && <XCircleIcon size={16} className="text-red-400 mt-0.5 flex-shrink-0" />}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white">{t.title}</p>
              {t.description && (
                <p className="text-xs text-gray-400 mt-0.5">{t.description}</p>
              )}
            </div>
            <button
              onClick={() => dismiss(t.id)}
              className="text-gray-600 hover:text-gray-400 flex-shrink-0 transition-colors"
            >
              <XIcon size={14} />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
