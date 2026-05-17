"use client";

import { useState, useCallback, useEffect } from "react";

export interface Toast {
  id: string;
  title: string;
  description?: string;
  variant?: "default" | "success" | "error" | "info";
  duration?: number;
}

// Pre-mount toast buffer: toasts fired before the Toaster mounts are queued
// here and flushed the moment _addToast is registered.
const _buffer: Omit<Toast, "id">[] = [];
let _addToast: ((t: Omit<Toast, "id">) => void) | null = null;

export function toast(t: Omit<Toast, "id">) {
  if (_addToast) {
    _addToast(t);
  } else {
    // Buffer until Toaster mounts
    _buffer.push(t);
  }
}

export function useToastState() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const add = useCallback((t: Omit<Toast, "id">) => {
    const id = Math.random().toString(36).slice(2);
    const duration = t.duration ?? (t.variant === "error" ? 5000 : 3500);
    setToasts((prev) => {
      // De-duplicate: don't show same title+variant twice within the duration window
      const isDupe = prev.some(x => x.title === t.title && x.variant === t.variant);
      if (isDupe) return prev;
      return [...prev, { ...t, id }];
    });
    setTimeout(() => setToasts((prev) => prev.filter((x) => x.id !== id)), duration);
  }, []);

  // Register on mount, flush buffer, deregister on unmount
  useEffect(() => {
    _addToast = add;
    // Flush buffered pre-mount toasts
    const buffered = _buffer.splice(0);
    buffered.forEach(add);
    return () => {
      _addToast = null;
    };
  }, [add]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((x) => x.id !== id));
  }, []);

  return { toasts, dismiss };
}
