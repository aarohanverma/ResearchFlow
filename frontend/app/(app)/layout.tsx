"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuthStore } from "@/store/auth";
import { useNamespaceStore, NAMESPACE_TREE } from "@/store/namespace";
import { useThemeStore } from "@/store/theme";
import { logout } from "@/lib/api";
import {
  BookmarkIcon, FlaskConicalIcon, HomeIcon, LogOutIcon,
  NetworkIcon, SettingsIcon, ZapIcon, ChevronDownIcon, ChevronRightIcon,
  SunIcon, MoonIcon,
} from "lucide-react";
import { JobsNotification } from "@/components/jobs/JobsPanel";

const NAV = [
  { href: "/feed",      label: "Feed",     icon: HomeIcon,          desc: "Paper feed" },
  { href: "/bookmarks", label: "Saved",    icon: BookmarkIcon,      desc: "Bookmarks" },
  { href: "/graph",     label: "Graph",    icon: NetworkIcon,       desc: "Knowledge graph" },
  { href: "/genie",     label: "Genie",    icon: FlaskConicalIcon,  desc: "Idea synthesizer" },
  { href: "/settings",  label: "Settings", icon: SettingsIcon,      desc: "Preferences" },
];

// ─── Hierarchical namespace sidebar ──────────────────────────────────────────

function NamespaceSidebar() {
  const {
    subscribedSubjects, activeSubject, selectedTopics, collapsedSubjects,
    setSubject, toggleTopic, selectAllTopics, toggleSubjectCollapse,
  } = useNamespaceStore();

  const visibleSubjects = NAMESPACE_TREE.filter(s => subscribedSubjects.includes(s.key));

  return (
    <div style={{
      padding: "0 8px 8px",
      borderBottom: "1px solid var(--rf-border)",
      marginBottom: 8,
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 4px", marginBottom: 6 }}>
        <p style={{
          fontSize: "8.5px", fontWeight: 700, color: "var(--rf-text5)",
          textTransform: "uppercase", letterSpacing: "0.1em",
        }}>Namespace</p>
        <Link href="/settings" style={{ textDecoration: "none" }}>
          <span style={{ fontSize: "8px", color: "var(--rf-text4)", cursor: "pointer" }}>+ manage</span>
        </Link>
      </div>

      {visibleSubjects.length === 0 && (
        <div style={{ padding: "8px 4px" }}>
          <p style={{ fontSize: "9px", color: "var(--rf-text4)", lineHeight: 1.4 }}>
            No subjects subscribed.{" "}
            <Link href="/settings" style={{ color: "#6366f1", textDecoration: "none" }}>Add in Settings →</Link>
          </p>
        </div>
      )}

      {visibleSubjects.map(subject => {
        const isActive     = activeSubject === subject.key;
        const isCollapsed  = collapsedSubjects.includes(subject.key);
        const topicKeys    = subject.topics.map(t => t.key);
        const selectedHere = topicKeys.filter(k => selectedTopics.includes(k));
        const allSelected  = selectedHere.length === topicKeys.length;

        return (
          <div key={subject.key} style={{ marginBottom: 2 }}>
            {/* Subject row */}
            <div
              style={{
                display: "flex", alignItems: "center", gap: 6,
                padding: "5px 6px", borderRadius: 8, cursor: "pointer",
                background: isActive ? `${subject.color}14` : "transparent",
                border: `1px solid ${isActive ? `${subject.color}30` : "transparent"}`,
                transition: "all 0.15s",
              }}
            >
              {/* Collapse toggle */}
              <button
                onClick={() => toggleSubjectCollapse(subject.key)}
                style={{ background: "none", border: "none", cursor: "pointer", color: "var(--rf-text5)", display: "flex", padding: 0, flexShrink: 0 }}
              >
                {isCollapsed
                  ? <ChevronRightIcon size={10} />
                  : <ChevronDownIcon size={10} />}
              </button>

              {/* Subject label — click to set as active subject */}
              <button
                onClick={() => { setSubject(subject.key); }}
                style={{
                  flex: 1, background: "none", border: "none", cursor: "pointer",
                  textAlign: "left", display: "flex", alignItems: "center", gap: 5,
                }}
              >
                <span style={{ fontSize: "11px" }}>{subject.icon}</span>
                <span style={{ fontSize: "10.5px", fontWeight: isActive ? 700 : 500, color: isActive ? subject.color : "var(--rf-text3)" }}>
                  {subject.label}
                </span>
              </button>

              {/* Count badge */}
              {selectedHere.length > 0 && (
                <span style={{
                  fontSize: "8px", fontWeight: 700, color: subject.color,
                  background: `${subject.color}20`, borderRadius: 10,
                  padding: "1px 5px", flexShrink: 0,
                }}>
                  {selectedHere.length}
                </span>
              )}
            </div>

            {/* Topics */}
            <AnimatePresence initial={false}>
              {!isCollapsed && isActive && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.18 }}
                  style={{ overflow: "hidden", paddingLeft: 16 }}
                >
                  {/* Select all row */}
                  <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 4px", marginBottom: 1 }}>
                    <input
                      type="checkbox"
                      checked={allSelected}
                      onChange={() => {
                        if (allSelected) {
                          useNamespaceStore.setState({ selectedTopics: [topicKeys[0]] });
                        } else {
                          selectAllTopics(subject.key);
                        }
                      }}
                      style={{ accentColor: subject.color, width: 10, height: 10, cursor: "pointer" }}
                    />
                    <button
                      onClick={() => allSelected
                        ? useNamespaceStore.setState({ selectedTopics: [topicKeys[0]] })
                        : selectAllTopics(subject.key)
                      }
                      style={{ background: "none", border: "none", cursor: "pointer", fontSize: "9.5px", color: "var(--rf-text4)", fontWeight: 600 }}
                    >
                      All {subject.label}
                    </button>
                  </div>

                  {subject.topics.map(topic => {
                    const checked = selectedTopics.includes(topic.key);
                    return (
                      <label
                        key={topic.key}
                        style={{
                          display: "flex", alignItems: "center", gap: 6,
                          padding: "3px 4px", borderRadius: 5, cursor: "pointer",
                          background: checked ? `${subject.color}10` : "transparent",
                          transition: "background 0.12s",
                          marginBottom: 1,
                        }}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleTopic(topic.key)}
                          style={{ accentColor: subject.color, width: 10, height: 10, cursor: "pointer", flexShrink: 0 }}
                        />
                        <span style={{
                          fontSize: "9.5px", fontWeight: checked ? 600 : 400,
                          color: checked ? subject.color : "#6b7280",
                          userSelect: "none",
                        }}>
                          {topic.label}
                        </span>
                        <span style={{ fontSize: "7.5px", color: "var(--rf-text5)", fontFamily: "monospace", marginLeft: "auto" }}>
                          {topic.key}
                        </span>
                      </label>
                    );
                  })}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        );
      })}
    </div>
  );
}

// ─── Layout ───────────────────────────────────────────────────────────────────

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router   = useRouter();
  const { token, user } = useAuthStore();
  const { theme, toggle: toggleTheme } = useThemeStore();
  const [mounted, setMounted] = useState(false);

  // Apply persisted theme on mount (also done inline in root layout to avoid FOUC)
  useEffect(() => {
    setMounted(true);
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", theme);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (mounted && !token) router.replace("/login"); }, [mounted, token, router]);

  if (!mounted) return <div className="h-screen w-screen" style={{ background: "var(--rf-bg)" }} />;
  if (!token) return null;

  const isLight = theme === "light";

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--rf-bg)" }}>
      {/* Sidebar */}
      <nav
        style={{
          width: 230, flexShrink: 0,
          borderRight: "1px solid var(--rf-border)",
          background: "var(--rf-sidebar)",
          display: "flex", flexDirection: "column",
          overflowY: "auto",
        }}
      >
        {/* Logo + theme toggle + bell */}
        <div style={{ padding: "16px 12px 10px", flexShrink: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{
              width: 28, height: 28, borderRadius: 8,
              background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
              display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 0 14px rgba(99,102,241,0.4)",
            }}>
              <ZapIcon size={14} color="white" />
            </div>
            <span style={{ fontSize: "13px", fontWeight: 700, color: "var(--rf-text1)", flex: 1 }}>ResearchFlow</span>
            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              title={isLight ? "Switch to dark mode" : "Switch to light mode"}
              style={{
                width: 24, height: 24, borderRadius: 6,
                background: "var(--rf-surface3)", border: "1px solid var(--rf-border2)",
                display: "flex", alignItems: "center", justifyContent: "center",
                cursor: "pointer", flexShrink: 0, color: "var(--rf-text4)",
                transition: "all 0.15s",
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = "var(--rf-text2)"; }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = "var(--rf-text4)"; }}
            >
              {isLight ? <MoonIcon size={11} /> : <SunIcon size={11} />}
            </button>
            <JobsNotification />
          </div>
        </div>

        {/* Nav */}
        <div style={{ padding: "0 8px", marginBottom: 12, flexShrink: 0 }}>
          {NAV.map(({ href, label, icon: Icon }) => {
            const active = pathname.startsWith(href);
            return (
              <Link key={href} href={href} style={{ textDecoration: "none" }}>
                <div style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: active ? "8px 10px 8px 7px" : "8px 10px",
                  borderRadius: 9, marginBottom: 2,
                  background: active ? "var(--rf-nav-active)" : "transparent",
                  borderTop: `1px solid ${active ? "var(--rf-nav-border)" : "transparent"}`,
                  borderRight: `1px solid ${active ? "var(--rf-nav-border)" : "transparent"}`,
                  borderBottom: `1px solid ${active ? "var(--rf-nav-border)" : "transparent"}`,
                  borderLeft: active ? "3px solid #6366f1" : "3px solid transparent",
                  cursor: "pointer", transition: "all 0.12s",
                }}>
                  <Icon size={14} color={active ? "#818cf8" : "var(--rf-text4)"} />
                  <span style={{ fontSize: "12px", fontWeight: active ? 600 : 500, color: active ? (isLight ? "#4338ca" : "#e0e7ff") : "var(--rf-text3)" }}>
                    {label}
                  </span>
                </div>
              </Link>
            );
          })}
        </div>

        {/* Hierarchical namespace selector */}
        <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
          <NamespaceSidebar />
        </div>

        {/* User + logout */}
        <div style={{ padding: "10px 10px", borderTop: "1px solid var(--rf-border)", flexShrink: 0 }}>
          {user && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 8px", marginBottom: 4 }}>
              <div style={{
                width: 28, height: 28, borderRadius: "50%",
                background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
                display: "flex", alignItems: "center", justifyContent: "center",
                flexShrink: 0, fontSize: "11px", fontWeight: 700, color: "white",
              }}>
                {user.display_name?.[0]?.toUpperCase() ?? "U"}
              </div>
              <div style={{ minWidth: 0 }}>
                <p style={{ fontSize: "11px", fontWeight: 600, color: "var(--rf-text2)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {user.display_name}
                </p>
                <p style={{ fontSize: "9px", color: "var(--rf-text4)" }}>{user.expertise_level} · {user.orientation}</p>
              </div>
            </div>
          )}
          <button
            onClick={logout}
            style={{
              width: "100%", display: "flex", alignItems: "center", gap: 8,
              padding: "7px 10px", borderRadius: 8, background: "none", border: "none",
              color: "var(--rf-text4)", fontSize: "11px", fontWeight: 500, cursor: "pointer",
              transition: "color 0.15s, background 0.15s",
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = "#f87171"; (e.currentTarget as HTMLButtonElement).style.background = "rgba(239,68,68,0.1)"; }}
            onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = "var(--rf-text4)"; (e.currentTarget as HTMLButtonElement).style.background = "none"; }}
          >
            <LogOutIcon size={13} />
            Sign out
          </button>
        </div>
      </nav>

      {/* Main */}
      <main style={{ flex: 1, overflow: "hidden" }}>
        <motion.div
          key={pathname}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.12, ease: "easeOut" }}
          style={{ height: "100%" }}
        >
          {children}
        </motion.div>
      </main>
    </div>
  );
}
