"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react";

export interface NavItem {
  id: string;
  label: string;
  icon: string;
}

interface SectionNavPanelProps {
  items: NavItem[];
  scrollRef: React.RefObject<HTMLDivElement | null>;
  accent?: "indigo" | "violet" | "fuchsia";
}

const ACCENT = {
  indigo:  { bg: "bg-indigo-500/10",  text: "text-indigo-400",  bar: "bg-indigo-500",  header: "text-indigo-400/60"  },
  violet:  { bg: "bg-violet-500/10",  text: "text-violet-400",  bar: "bg-violet-500",  header: "text-violet-400/60"  },
  fuchsia: { bg: "bg-fuchsia-500/10", text: "text-fuchsia-400", bar: "bg-fuchsia-500", header: "text-fuchsia-400/60" },
};

export function SectionNavPanel({
  items,
  scrollRef,
  accent = "indigo",
}: SectionNavPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const colors = ACCENT[accent];

  // Stable refs so the observer callback never goes stale
  const observerRef  = useRef<IntersectionObserver | null>(null);
  const visibleRef   = useRef<Set<string>>(new Set());
  const observedRef  = useRef<Set<string>>(new Set());
  const itemsRef     = useRef<NavItem[]>(items);

  // Keep itemsRef current on every render (no effect needed)
  itemsRef.current = items;

  // Pick the topmost visible section (by items order)
  const pickActive = useCallback(() => {
    const visible = visibleRef.current;
    const first = itemsRef.current.find((item) => visible.has(item.id));
    setActiveId(first?.id ?? null);
  }, []);

  // Create the observer once, keyed to the scroll container
  useEffect(() => {
    const root = scrollRef.current;
    if (!root) return;

    // Tear down previous observer if scroll container changed
    observerRef.current?.disconnect();
    visibleRef.current.clear();
    observedRef.current.clear();

    observerRef.current = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) visibleRef.current.add(entry.target.id);
          else                       visibleRef.current.delete(entry.target.id);
        }
        pickActive();
      },
      { root, rootMargin: "0px 0px -45% 0px", threshold: 0 }
    );

    // Observe all items already available at mount time
    for (const { id } of itemsRef.current) {
      const el = document.getElementById(id);
      if (el) { observerRef.current.observe(el); observedRef.current.add(id); }
    }

    return () => {
      observerRef.current?.disconnect();
      observerRef.current = null;
      visibleRef.current.clear();
      observedRef.current.clear();
    };
  // Only recreate when the scroll container itself changes
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scrollRef]);

  // Incrementally observe NEW items as they arrive (e.g. during streaming)
  useEffect(() => {
    const observer = observerRef.current;
    if (!observer) return;
    for (const { id } of items) {
      if (!observedRef.current.has(id)) {
        const el = document.getElementById(id);
        if (el) { observer.observe(el); observedRef.current.add(id); }
      }
    }
  }, [items]);

  const scrollTo = useCallback((id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveId(id);
  }, []);

  if (items.length === 0) return null;

  return (
    <motion.div
      initial={false}
      animate={{ width: collapsed ? 44 : 180 }}
      transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
      className="flex-shrink-0 flex flex-col h-full border-r border-gray-800/40 overflow-hidden select-none"
      style={{ background: "var(--rf-bg)" }}
    >
      {/* Header */}
      <div
        className="flex items-center h-10 px-2 border-b border-gray-800/30 flex-shrink-0 gap-1.5"
        style={{ background: "var(--rf-bg)" }}
      >
        <AnimatePresence initial={false}>
          {!collapsed && (
            <motion.span
              key="label"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className={`flex-1 text-[9px] font-bold uppercase tracking-widest whitespace-nowrap overflow-hidden ${colors.header}`}
            >
              On this page
            </motion.span>
          )}
        </AnimatePresence>
        <button
          onClick={() => setCollapsed((v) => !v)}
          className="flex items-center justify-center w-7 h-7 rounded-lg text-gray-600 hover:text-gray-300 hover:bg-gray-800/60 transition-all flex-shrink-0 ml-auto"
          title={collapsed ? "Expand navigation" : "Collapse navigation"}
        >
          {collapsed ? <ChevronRightIcon size={12} /> : <ChevronLeftIcon size={12} />}
        </button>
      </div>

      {/* Items */}
      <div className="overflow-y-auto flex-1 py-1.5">
        <AnimatePresence initial={false}>
          {items.map(({ id, label, icon }) => {
            const isActive = activeId === id;
            return (
              <motion.button
                key={id}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                onClick={() => scrollTo(id)}
                title={label}
                className={`w-full flex items-center gap-2.5 text-left transition-colors relative group ${
                  collapsed ? "px-2.5 py-2 justify-center" : "px-2.5 py-2"
                } ${
                  isActive
                    ? `${colors.bg} ${colors.text}`
                    : "text-gray-500 hover:text-gray-200 hover:bg-gray-800/30"
                }`}
              >
                {/* Active indicator */}
                {isActive && (
                  <span className={`absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-full ${colors.bar}`} />
                )}

                <span className="flex-shrink-0 text-[13px] leading-none w-5 text-center">
                  {icon}
                </span>

                <AnimatePresence initial={false}>
                  {!collapsed && (
                    <motion.span
                      key="text"
                      initial={{ opacity: 0, width: 0 }}
                      animate={{ opacity: 1, width: "auto" }}
                      exit={{ opacity: 0, width: 0 }}
                      transition={{ duration: 0.18 }}
                      className="text-[10.5px] font-medium truncate leading-snug overflow-hidden whitespace-nowrap"
                    >
                      {label}
                    </motion.span>
                  )}
                </AnimatePresence>

                {/* Collapsed tooltip */}
                {collapsed && (
                  <span className="pointer-events-none absolute left-full top-1/2 -translate-y-1/2 ml-2 px-2 py-1 rounded-md bg-gray-900 border border-gray-700/60 text-[10px] text-gray-200 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity z-50 shadow-lg">
                    {label}
                  </span>
                )}
              </motion.button>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Section count when collapsed */}
      {collapsed && items.length > 0 && (
        <div className="px-2 py-2 border-t border-gray-800/30 flex justify-center">
          <span className={`text-[9px] font-bold tabular-nums ${colors.header}`}>
            {items.length}
          </span>
        </div>
      )}
    </motion.div>
  );
}
