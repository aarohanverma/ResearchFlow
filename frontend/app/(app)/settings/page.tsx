"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth";
import type { User } from "@/types";
import {
  Loader2Icon, CheckIcon, BellIcon,
  CpuIcon, TagIcon, UserIcon,
  ChevronDownIcon, ChevronRightIcon,
  KeyIcon, EyeIcon, EyeOffIcon, XCircleIcon, XIcon,
  BarChart3Icon,
} from "lucide-react";
import { useNamespaceStore, NAMESPACE_TREE } from "@/store/namespace";

const TABS = [
  { key: "profile",       label: "Profile",      icon: UserIcon },
  { key: "usage",         label: "Token Usage",   icon: BarChart3Icon },
  { key: "topics",        label: "Topics",        icon: TagIcon },
  { key: "provider",      label: "AI Provider",   icon: CpuIcon },
  { key: "api-keys",      label: "API Keys",      icon: KeyIcon },
  { key: "notifications", label: "Notifications", icon: BellIcon },
] as const;

type Tab = typeof TABS[number]["key"];

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<Tab>("profile");

  return (
    <div className="h-full overflow-y-auto px-8 py-8 max-w-4xl mx-auto w-full">
      <div className="mb-8">
        <h1 className="text-xl font-bold text-white tracking-tight">Settings</h1>
        <p className="text-sm text-gray-500 mt-1">Configure your research environment.</p>
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1 bg-gray-900/60 border border-gray-800/60 rounded-xl p-1 mb-7 overflow-x-auto">
        {TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`relative flex-shrink-0 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-xs font-medium transition-all duration-200 ${
              activeTab === key ? "text-white" : "text-gray-500 hover:text-gray-300"
            }`}
          >
            {activeTab === key && (
              <motion.div
                layoutId="settings-tab"
                className="absolute inset-0 bg-gray-800 rounded-lg"
                transition={{ type: "spring", damping: 30, stiffness: 400 }}
              />
            )}
            <Icon size={12} className="relative z-10" />
            <span className="relative z-10">{label}</span>
          </button>
        ))}
      </div>

      {/* Panels */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.18 }}
        >
          {activeTab === "profile"       && <ProfilePanel />}
          {activeTab === "usage"         && <UsagePanel />}
          {activeTab === "topics"        && <TopicsPanel />}
          {activeTab === "provider"      && <ProviderPanel />}
          {activeTab === "api-keys"      && <ApiKeysPanel />}
          {activeTab === "notifications" && <NotificationsPanel />}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

/* ── Profile Panel ─────────────────────────────────────────────────────────── */
function ProfilePanel() {
  const { user, setUser } = useAuthStore();
  const [form, setForm] = useState({
    display_name: user?.display_name ?? "",
    expertise_level: user?.expertise_level ?? "practitioner",
    orientation: user?.orientation ?? "both",
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.get<{ display_name: string; expertise_level: string; orientation: string }>(
      "/settings/profile"
    ).then((d) => setForm({
      display_name: d.display_name,
      expertise_level: (d.expertise_level as "newcomer" | "practitioner" | "expert") || "practitioner",
      orientation: (d.orientation as "research" | "production" | "both") || "both",
    })).catch(() => {});
  }, []);

  async function save() {
    setSaving(true);
    try {
      await api.patch("/settings/profile", form);
      // refresh user in store
      const me = await api.get<User>("/auth/me");
      setUser(me);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch {}
    setSaving(false);
  }

  return (
    <Card
      icon={<UserIcon size={16} className="text-indigo-400" />}
      title="Your Profile"
      description="Name and research preferences that personalise your feed and explanations."
    >
      <div className="space-y-4">
        <div className="space-y-1">
          <label className="text-[11px] font-semibold text-gray-600 uppercase tracking-wider">Display Name</label>
          <input
            value={form.display_name}
            onChange={(e) => setForm((f) => ({ ...f, display_name: e.target.value }))}
            className="input-base"
          />
        </div>

        <div className="space-y-1">
          <label className="text-[11px] font-semibold text-gray-600 uppercase tracking-wider">Expertise Level</label>
          <div className="flex gap-2">
            {(["newcomer", "practitioner", "expert"] as const).map((lvl) => (
              <button
                key={lvl}
                onClick={() => setForm((f) => ({ ...f, expertise_level: lvl }))}
                className={`flex-1 py-2.5 rounded-xl text-sm font-medium transition-all border ${
                  form.expertise_level === lvl
                    ? "bg-indigo-600/20 border-indigo-500 text-indigo-300"
                    : "border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300"
                }`}
              >
                {lvl.charAt(0).toUpperCase() + lvl.slice(1)}
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-600">
            {form.expertise_level === "newcomer" && "Plain-language explanations, more background context."}
            {form.expertise_level === "practitioner" && "Technical depth, skip basics, focus on methods."}
            {form.expertise_level === "expert" && "Dense notation, full rigor, minimal scaffolding."}
          </p>
        </div>

        <div className="space-y-1">
          <label className="text-[11px] font-semibold text-gray-600 uppercase tracking-wider">Orientation</label>
          <div className="flex gap-2">
            {([
              { key: "research", label: "Research", desc: "Theory, benchmarks, SOTA" },
              { key: "production", label: "Production", desc: "Deployment, efficiency, engineering" },
              { key: "both", label: "Both", desc: "Balanced coverage" },
            ] as const).map(({ key, label, desc }) => (
              <button
                key={key}
                onClick={() => setForm((f) => ({ ...f, orientation: key }))}
                className={`flex-1 py-2.5 px-3 rounded-xl text-xs font-medium transition-all border text-left ${
                  form.orientation === key
                    ? "bg-teal-600/20 border-teal-500 text-teal-300"
                    : "border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300"
                }`}
              >
                <div className="font-semibold">{label}</div>
                <div className="text-[10px] opacity-70 mt-0.5">{desc}</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      <SaveButton saving={saving} saved={saved} onClick={save} />
    </Card>
  );
}

/* ── Topics Panel ─────────────────────────────────────────────────────────── */
function TopicsPanel() {
  const {
    subscribedSubjects, selectedTopics,
    subscribeSubject, unsubscribeSubject,
    toggleTopic, selectAllTopics, setSubject, activeSubject,
  } = useNamespaceStore();

  const [expandedSubject, setExpandedSubject] = useState<string | null>(null);

  function toggleSubscribe(subjectKey: string) {
    if (subscribedSubjects.includes(subjectKey)) {
      unsubscribeSubject(subjectKey);
    } else {
      subscribeSubject(subjectKey);
    }
  }

  return (
    <Card
      icon={<TagIcon size={16} className="text-teal-400" />}
      title="Research Namespace Subscriptions"
      description="Subscribe to subjects to show them in the sidebar. Then select specific topics inside each subject to scope feed, bookmarks, graph, and genie. Cross-subject access is available only for bookmark imports."
    >
      <div className="space-y-2">
        {NAMESPACE_TREE.map(subject => {
          const isSubscribed = subscribedSubjects.includes(subject.key);
          const isExpanded   = expandedSubject === subject.key;
          const topicKeys    = subject.topics.map(t => t.key);
          const selectedHere = isSubscribed && activeSubject === subject.key
            ? topicKeys.filter(k => selectedTopics.includes(k))
            : [];
          const allSelected  = selectedHere.length === topicKeys.length;

          return (
            <div
              key={subject.key}
              className="rounded-xl border transition-all duration-200"
              style={{
                borderColor: isSubscribed ? `${subject.color}40` : "rgba(255,255,255,0.06)",
                background: isSubscribed ? `${subject.color}08` : "rgba(255,255,255,0.02)",
              }}
            >
              {/* Subject header row */}
              <div className="flex items-center gap-3 px-4 py-3">
                <span className="text-base flex-shrink-0">{subject.icon}</span>

                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-gray-200">{subject.label}</p>
                  <p className="text-[10px] text-gray-500">
                    {subject.topics.length} topics
                    {isSubscribed && activeSubject === subject.key && selectedHere.length > 0 &&
                      ` · ${selectedHere.length} active`}
                  </p>
                </div>

                {/* Subscribed status + expand toggle */}
                {isSubscribed && (
                  <button
                    onClick={() => setExpandedSubject(isExpanded ? null : subject.key)}
                    className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 transition-colors px-2 py-1"
                  >
                    {isExpanded ? <ChevronDownIcon size={12} /> : <ChevronRightIcon size={12} />}
                    <span>Topics</span>
                  </button>
                )}

                {/* Toggle switch */}
                <button
                  onClick={() => toggleSubscribe(subject.key)}
                  disabled={isSubscribed && subscribedSubjects.length === 1}
                  title={isSubscribed && subscribedSubjects.length === 1 ? "At least one subject must be subscribed" : undefined}
                  className="relative flex-shrink-0 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <div
                    className="w-10 h-5 rounded-full transition-colors duration-200"
                    style={{ background: isSubscribed ? subject.color : "#374151" }}
                  />
                  <div
                    className="absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform duration-200"
                    style={{ left: isSubscribed ? "calc(100% - 18px)" : "2px" }}
                  />
                </button>
              </div>

              {/* Topic checkboxes — only when subscribed and expanded */}
              <AnimatePresence initial={false}>
                {isSubscribed && isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.18 }}
                    style={{ overflow: "hidden" }}
                  >
                    <div className="px-4 pb-3 border-t border-white/5">
                      <p className="text-[10px] text-gray-500 mt-2.5 mb-2">
                        {activeSubject !== subject.key
                          ? <button
                              onClick={() => setSubject(subject.key)}
                              className="text-indigo-400 hover:text-indigo-300 underline transition-colors"
                            >
                              Switch to {subject.label} to select topics
                            </button>
                          : "Select which topics to show in feed, bookmarks, graph & genie:"
                        }
                      </p>
                      {activeSubject === subject.key && (
                        <>
                          {/* Select all */}
                          <label className="flex items-center gap-2 mb-2 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={allSelected}
                              onChange={() => allSelected
                                ? useNamespaceStore.setState({ selectedTopics: [topicKeys[0]] })
                                : selectAllTopics(subject.key)
                              }
                              className="w-3 h-3 rounded cursor-pointer"
                              style={{ accentColor: subject.color }}
                            />
                            <span className="text-[11px] font-semibold text-gray-400">All topics</span>
                          </label>

                          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                            {subject.topics.map(topic => {
                              const checked = selectedTopics.includes(topic.key);
                              return (
                                <label
                                  key={topic.key}
                                  className="flex items-center gap-2 cursor-pointer py-0.5"
                                >
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={() => toggleTopic(topic.key)}
                                    className="w-3 h-3 rounded cursor-pointer flex-shrink-0"
                                    style={{ accentColor: subject.color }}
                                  />
                                  <div className="min-w-0">
                                    <span className="text-[11px] text-gray-300 block truncate">{topic.label}</span>
                                    <span className="text-[9px] text-gray-600 font-mono">{topic.key}</span>
                                  </div>
                                </label>
                              );
                            })}
                          </div>
                        </>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      <div className="pt-1 border-t border-gray-800/40">
        <p className="text-[11px] text-gray-600 leading-relaxed">
          <strong className="text-gray-500">Subject scope</strong> — feed, bookmarks, graph & genie show all topics under the active subject.<br />
          <strong className="text-gray-500">Topic scope</strong> — tick specific topics to narrow down within a subject.<br />
          <strong className="text-gray-500">Cross-subject</strong> — only allowed for bookmark imports; all other views are single-subject scoped.
        </p>
      </div>
    </Card>
  );
}

/* ── Provider Panel ───────────────────────────────────────────────────────── */
function ProviderPanel() {
  // Initial values match config.py defaults; overwritten by the GET response
  // which always returns the effective backend configuration.
  const [cfg, setCfg] = useState({
    llm_provider: "openai",
    cheap_model: "gpt-4o-mini",
    quality_model: "gpt-5.4-mini",
    reasoning_model: "gpt-5.4",
    embedding_provider: "gemini",
    embedding_model: "gemini-embedding-2-preview",
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    // Backend always returns effective settings (DB row or system defaults),
    // so we can overwrite the local state unconditionally.
    api.get<typeof cfg>("/settings/provider")
      .then((d) => setCfg((c) => ({ ...c, ...d })))
      .catch(() => {});
  }, []);

  async function save() {
    setSaving(true);
    try {
      await api.patch("/settings/provider", cfg);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch {}
    setSaving(false);
  }

  const fields = [
    { key: "llm_provider",       label: "LLM Provider",       options: ["openai", "anthropic", "google"] },
    { key: "cheap_model",        label: "Fast / Cheap Model",  options: ["gpt-4o-mini", "claude-haiku-4-5", "gemini-2.0-flash"] },
    { key: "quality_model",      label: "Quality Model",       options: ["gpt-5.4-mini", "gpt-4o", "claude-sonnet-4-6", "gemini-2.5-pro"] },
    { key: "reasoning_model",    label: "Reasoning Model",     options: ["gpt-5.4", "o3", "o3-mini", "claude-opus-4-7", "claude-opus-4-6"] },
    { key: "embedding_provider", label: "Embedding Provider",  options: ["gemini", "openai", "voyage"] },
    { key: "embedding_model",    label: "Embedding Model",     options: ["gemini-embedding-2-preview", "text-embedding-3-large", "voyage-3"] },
  ] as const;

  return (
    <Card
      icon={<CpuIcon size={16} className="text-purple-400" />}
      title="AI Provider Configuration"
      description="Changes take effect on the next request. Requires valid API keys in .env.local."
    >
      <div className="space-y-3">
        {fields.map(({ key, label, options }) => (
          <div key={key} className="space-y-1">
            <label className="text-[11px] font-semibold text-gray-600 uppercase tracking-wider">{label}</label>
            <select
              value={(cfg as Record<string, string>)[key]}
              onChange={(e) => setCfg((c) => ({ ...c, [key]: e.target.value }))}
              className="w-full bg-gray-800 border border-gray-700/60 rounded-xl px-3.5 py-2.5 text-sm text-gray-300 outline-none focus:border-indigo-500 transition-colors"
            >
              {options.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
        ))}
      </div>
      <SaveButton saving={saving} saved={saved} onClick={save} />
    </Card>
  );
}

/* ── API Keys Panel ───────────────────────────────────────────────────────── */
interface KeyStatus { is_set: boolean; from_env: boolean; is_overridden: boolean; masked: string }
interface ApiKeyState { openai: KeyStatus; anthropic: KeyStatus; google: KeyStatus }
type Provider = "openai" | "anthropic" | "google";

function ApiKeysPanel() {
  const [status, setStatus] = useState<ApiKeyState | null>(null);
  // null = not editing; "" = editing with empty value; "sk-..." = editing with typed value
  const [editing, setEditing] = useState<Record<Provider, string | null>>({ openai: null, anthropic: null, google: null });
  const [visible, setVisible] = useState<Record<Provider, boolean>>({ openai: false, anthropic: false, google: false });
  const [saving, setSaving] = useState<Provider | null>(null);

  useEffect(() => {
    api.get<ApiKeyState>("/settings/api-keys").then(setStatus).catch(() => {});
  }, []);

  async function saveKey(provider: Provider) {
    const val = editing[provider];
    if (val === null) return;
    setSaving(provider);
    try {
      await api.patch("/settings/api-keys", { [`${provider}_key`]: val || null });
      const fresh = await api.get<ApiKeyState>("/settings/api-keys");
      setStatus(fresh);
      setEditing(v => ({ ...v, [provider]: null }));
    } catch {}
    setSaving(null);
  }

  async function clearKey(provider: Provider) {
    try {
      await api.patch("/settings/api-keys", { [`${provider}_key`]: null });
      const fresh = await api.get<ApiKeyState>("/settings/api-keys");
      setStatus(fresh);
    } catch {}
  }

  const KEYS: { id: Provider; label: string; placeholder: string; hint: string }[] = [
    { id: "openai",    label: "OpenAI",    placeholder: "sk-…",     hint: "Used for GPT models and text-embedding-3" },
    { id: "anthropic", label: "Anthropic", placeholder: "sk-ant-…", hint: "Used for Claude models" },
    { id: "google",    label: "Google",    placeholder: "AIza…",    hint: "Used for Gemini models and embeddings" },
  ];

  return (
    <Card
      icon={<KeyIcon size={16} className="text-amber-400" />}
      title="API Keys"
      description="Keys are read from environment variables by default. Override here to use your own. Values are always hidden."
    >
      <div className="space-y-5">
        {KEYS.map(({ id, label, placeholder, hint }) => {
          const st = status?.[id];
          const isEditing = editing[id] !== null;
          const isVis = visible[id];
          const isSaving = saving === id;

          return (
            <div key={id} className="space-y-1.5">
              {/* Label + badge row */}
              <div className="flex items-center gap-2">
                <label className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider flex-1">{label}</label>
                {st && (
                  <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full border ${
                    st.is_overridden
                      ? "bg-indigo-950/40 border-indigo-700/40 text-indigo-300"
                      : st.from_env
                        ? "bg-teal-950/40 border-teal-700/40 text-teal-300"
                        : "bg-gray-800 border-gray-700 text-gray-500"
                  }`}>
                    {st.is_overridden ? "override" : st.from_env ? "from env" : "not set"}
                  </span>
                )}
              </div>

              {/* In-place field: masked display → clicks open editable input */}
              {isEditing ? (
                <div className="relative">
                  <input
                    type={isVis ? "text" : "password"}
                    value={editing[id]!}
                    onChange={(e) => setEditing(v => ({ ...v, [id]: e.target.value }))}
                    placeholder={placeholder}
                    autoFocus
                    className="w-full bg-gray-800 border border-indigo-500/50 rounded-xl px-3.5 py-2.5 pr-24 text-sm text-gray-300 placeholder-gray-600 outline-none font-mono"
                    autoComplete="off"
                    spellCheck={false}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") saveKey(id);
                      if (e.key === "Escape") setEditing(v => ({ ...v, [id]: null }));
                    }}
                  />
                  <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
                    <button type="button" onClick={() => setVisible(v => ({ ...v, [id]: !v[id] }))} className="p-1.5 text-gray-600 hover:text-gray-400 transition-colors">
                      {isVis ? <EyeOffIcon size={12} /> : <EyeIcon size={12} />}
                    </button>
                    <button type="button" onClick={() => saveKey(id)} disabled={isSaving} className="p-1.5 text-indigo-400 hover:text-indigo-300 disabled:opacity-40 transition-colors">
                      {isSaving ? <Loader2Icon size={12} className="animate-spin" /> : <CheckIcon size={12} />}
                    </button>
                    <button type="button" onClick={() => setEditing(v => ({ ...v, [id]: null }))} className="p-1.5 text-gray-600 hover:text-gray-400 transition-colors">
                      <XIcon size={12} />
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setEditing(v => ({ ...v, [id]: "" }))}
                  className="w-full flex items-center gap-2 bg-gray-800/40 border border-gray-700/40 hover:border-indigo-500/30 rounded-xl px-3 py-2.5 transition-colors group text-left"
                  title="Click to override"
                >
                  <span className="flex-1 text-xs font-mono tracking-wider">
                    {st?.is_set
                      ? <span className="text-gray-400">{st.masked || "••••••••••••••••••••••••"}</span>
                      : <span className="text-gray-600 italic">{placeholder}</span>}
                  </span>
                  {st?.is_overridden ? (
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); clearKey(id); }}
                      title="Remove override, revert to env"
                      className="text-gray-600 hover:text-red-400 transition-colors flex-shrink-0"
                    >
                      <XCircleIcon size={13} />
                    </button>
                  ) : (
                    <span className="text-[10px] text-gray-700 group-hover:text-indigo-400 transition-colors flex-shrink-0">
                      click to {st?.is_set ? "override" : "set"}
                    </span>
                  )}
                </button>
              )}
              <p className="text-[10px] text-gray-600">{hint}</p>
            </div>
          );
        })}
      </div>

      <div className="pt-2 border-t border-gray-800/40">
        <p className="text-[11px] text-gray-600 leading-relaxed">
          Overrides apply immediately. Press Enter or ✓ to save. Clear an override to revert to the server environment variable.
        </p>
      </div>
    </Card>
  );
}

/* ── Notifications Panel ──────────────────────────────────────────────────── */
function NotificationsPanel() {
  const [prefs, setPrefs] = useState({
    notify_potd: true,
    notify_digest: true,
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.get<typeof prefs>("/settings/notifications")
      .then((d) => { if (Object.keys(d).length > 0) setPrefs(d); })
      .catch(() => {});
  }, []);

  const NOTIF_ITEMS = [
    { key: "notify_potd",    label: "Paper of the Day", desc: "One hand-picked paper every morning", icon: "🌅" },
    { key: "notify_digest",  label: "Weekly Digest",    desc: "Top papers from your research week",  icon: "📚" },
  ] as const;

  async function save() {
    setSaving(true);
    try {
      await api.patch("/settings/notifications", prefs);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch {}
    setSaving(false);
  }

  return (
    <Card
      icon={<BellIcon size={16} className="text-amber-400" />}
      title="Email Notifications"
      description="Requires a valid Resend API key. Configure EMAIL_FROM in .env.local."
    >
      <div className="space-y-2">
        {NOTIF_ITEMS.map(({ key, label, desc, icon }) => (
          <label
            key={key}
            className="flex items-center justify-between p-4 bg-gray-800/50 hover:bg-gray-800 rounded-xl cursor-pointer transition-colors border border-gray-700/30"
          >
            <div className="flex items-center gap-3">
              <span className="text-lg">{icon}</span>
              <div>
                <p className="text-sm font-medium text-gray-200">{label}</p>
                <p className="text-xs text-gray-500 mt-0.5">{desc}</p>
              </div>
            </div>
            <div className="relative flex-shrink-0">
              <input
                type="checkbox"
                checked={(prefs as Record<string, boolean>)[key]}
                onChange={(e) => setPrefs((p) => ({ ...p, [key]: e.target.checked }))}
                className="sr-only peer"
              />
              <div className="w-10 h-5 bg-gray-700 peer-checked:bg-indigo-600 rounded-full transition-colors duration-200" />
              <div className="absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform duration-200 peer-checked:translate-x-5 shadow" />
            </div>
          </label>
        ))}
      </div>
      <SaveButton saving={saving} saved={saved} onClick={save} label="Save Preferences" />
    </Card>
  );
}

/* ── Shared helpers ───────────────────────────────────────────────────────── */
function SaveButton({
  saving, saved, onClick, label = "Save Changes",
}: {
  saving: boolean; saved: boolean; onClick: () => void; label?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={saving}
      className="btn-primary flex items-center gap-2 px-5 py-2.5 text-sm"
    >
      {saving ? <Loader2Icon size={14} className="animate-spin" /> : saved ? <CheckIcon size={14} /> : null}
      {saved ? "Saved!" : saving ? "Saving…" : label}
    </button>
  );
}

function Card({
  icon, title, description, children,
}: {
  icon: React.ReactNode; title: string; description: string; children: React.ReactNode;
}) {
  return (
    <div className="bg-gray-900/60 border border-gray-800/60 rounded-2xl p-6 space-y-5">
      <div className="flex items-center gap-2.5">
        {icon}
        <div>
          <h2 className="text-sm font-semibold text-white">{title}</h2>
          <p className="text-xs text-gray-500 mt-0.5">{description}</p>
        </div>
      </div>
      {children}
    </div>
  );
}

/* ── Token Usage Panel ─────────────────────────────────────────────────────── */

interface UsageRow { input_tokens: number; output_tokens: number; total_tokens: number; cost_usd?: number; calls?: number }
interface UsageTotals extends UsageRow { calls: number }
interface UsageDayRow extends UsageRow { date: string }
interface UsageWfRow  extends UsageRow { workflow: string; calls: number }
interface UsageMdlRow extends UsageRow { provider: string; model: string; calls: number }
interface UsageResponse {
  range: { from: string; to: string };
  totals: UsageTotals;
  by_day: UsageDayRow[];
  by_workflow: UsageWfRow[];
  by_model: UsageMdlRow[];
}

function todayISO(): string {
  return new Date().toISOString().slice(0, 10);
}

function fmtNum(n: number): string {
  return n.toLocaleString("en-US");
}

function UsagePanel() {
  // Default: today only
  const [from, setFrom] = useState<string>(todayISO());
  const [to,   setTo]   = useState<string>(todayISO());
  const [data, setData] = useState<UsageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancel = false;
    setLoading(true);
    setErr(null);
    api.get<UsageResponse>(`/settings/token-usage?from=${from}&to=${to}`)
      .then(d => { if (!cancel) setData(d); })
      .catch((e: Error) => { if (!cancel) setErr(e.message || "Failed to load"); })
      .finally(() => { if (!cancel) setLoading(false); });
    return () => { cancel = true; };
  }, [from, to]);

  const setQuick = (kind: "today" | "7d" | "30d" | "all") => {
    const now = new Date();
    const t = now.toISOString().slice(0, 10);
    if (kind === "today") { setFrom(t); setTo(t); return; }
    const days = kind === "7d" ? 6 : kind === "30d" ? 29 : 365;
    const d = new Date(now);
    d.setDate(d.getDate() - days);
    setFrom(d.toISOString().slice(0, 10));
    setTo(t);
  };

  const totals = data?.totals;
  const maxDayTotal = Math.max(1, ...(data?.by_day || []).map(d => d.total_tokens));

  return (
    <div className="space-y-5">
      {/* Date range controls */}
      <div className="bg-gray-900/40 border border-gray-800/60 rounded-xl p-4 mb-5">
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="block text-[10px] uppercase font-semibold text-gray-500 tracking-wider mb-1">From</label>
            <input type="date" value={from} onChange={e => setFrom(e.target.value)}
              className="bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-indigo-600/60" />
          </div>
          <div>
            <label className="block text-[10px] uppercase font-semibold text-gray-500 tracking-wider mb-1">To</label>
            <input type="date" value={to} onChange={e => setTo(e.target.value)}
              className="bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-indigo-600/60" />
          </div>
          <div className="flex items-center gap-1.5 ml-auto">
            {(["today", "7d", "30d", "all"] as const).map(k => (
              <button key={k} onClick={() => setQuick(k)}
                className="text-[11px] px-2.5 py-1.5 rounded-lg border border-gray-800 bg-gray-900/60 text-gray-400 hover:text-gray-200 hover:border-gray-700 transition-colors">
                {k === "today" ? "Today" : k === "7d" ? "Last 7 days" : k === "30d" ? "Last 30 days" : "Last year"}
              </button>
            ))}
          </div>
        </div>
        {data && (
          <p className="text-[10px] text-gray-600 mt-2.5">
            Showing usage from <span className="text-gray-400">{data.range.from}</span> to <span className="text-gray-400">{data.range.to}</span> (UTC).
          </p>
        )}
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-xs text-gray-500 py-6 justify-center">
          <Loader2Icon size={13} className="animate-spin" /> Loading usage…
        </div>
      )}
      {err && !loading && (
        <div className="bg-red-950/30 border border-red-800/40 text-red-300 text-xs rounded-xl p-3">{err}</div>
      )}

      {!loading && totals && (
        <>
          {/* Totals row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
            <StatCard label="Input tokens"  value={fmtNum(totals.input_tokens)}  tone="indigo" />
            <StatCard label="Output tokens" value={fmtNum(totals.output_tokens)} tone="violet" />
            <StatCard label="Total tokens"  value={fmtNum(totals.total_tokens)}  tone="emerald" emphasis />
            <StatCard label="LLM calls"     value={fmtNum(totals.calls)}         tone="amber" sub={`${(totals.cost_usd || 0).toFixed(4)} USD est.`} />
          </div>

          {/* Per-day mini bar chart (only meaningful for multi-day ranges) */}
          {(data?.by_day.length || 0) > 1 && (
            <div className="bg-gray-900/40 border border-gray-800/60 rounded-xl p-4 mb-5">
              <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-3">Daily total</p>
              <div className="flex items-end gap-1.5 h-24">
                {data!.by_day.map(d => {
                  const h = Math.max(2, (d.total_tokens / maxDayTotal) * 100);
                  return (
                    <div key={d.date} className="flex-1 flex flex-col items-center gap-1" title={`${d.date}: ${fmtNum(d.total_tokens)} tokens`}>
                      <div className="w-full bg-indigo-600/40 hover:bg-indigo-500/70 rounded-t transition-colors" style={{ height: `${h}%` }} />
                      <span className="text-[8px] text-gray-600">{d.date.slice(5)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* By workflow */}
          {(data?.by_workflow.length || 0) > 0 && (
            <div className="bg-gray-900/40 border border-gray-800/60 rounded-xl p-4 mb-5">
              <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-3">By workflow</p>
              <div className="space-y-1.5">
                {data!.by_workflow.map((w, i) => (
                  <div key={i} className="flex items-center justify-between text-xs px-2.5 py-1.5 rounded-lg bg-gray-950/60 border border-gray-800/40">
                    <span className="text-gray-300 capitalize">{w.workflow}</span>
                    <div className="flex items-center gap-3 text-gray-500 font-mono text-[11px]">
                      <span title="input">{fmtNum(w.input_tokens)} in</span>
                      <span title="output">{fmtNum(w.output_tokens)} out</span>
                      <span className="text-gray-200">{fmtNum(w.total_tokens)} total</span>
                      <span className="text-gray-700">{w.calls} call{w.calls === 1 ? "" : "s"}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* By model */}
          {(data?.by_model.length || 0) > 0 && (
            <div className="bg-gray-900/40 border border-gray-800/60 rounded-xl p-4">
              <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-3">By model</p>
              <div className="space-y-1.5">
                {data!.by_model.map((m, i) => (
                  <div key={i} className="flex items-center justify-between text-xs px-2.5 py-1.5 rounded-lg bg-gray-950/60 border border-gray-800/40">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] uppercase text-gray-600 tracking-wider">{m.provider}</span>
                      <span className="text-gray-300 font-mono">{m.model}</span>
                    </div>
                    <div className="flex items-center gap-3 text-gray-500 font-mono text-[11px]">
                      <span>{fmtNum(m.input_tokens)} in</span>
                      <span>{fmtNum(m.output_tokens)} out</span>
                      <span className="text-gray-200">{fmtNum(m.total_tokens)} total</span>
                      <span className="text-gray-700">{m.calls} call{m.calls === 1 ? "" : "s"}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {totals.calls === 0 && (
            <p className="text-xs text-gray-600 text-center py-6">No LLM calls recorded for this range.</p>
          )}
        </>
      )}
    </div>
  );
}

function StatCard({ label, value, sub, tone, emphasis }: {
  label: string; value: string; sub?: string;
  tone: "indigo" | "violet" | "emerald" | "amber"; emphasis?: boolean;
}) {
  const accent: Record<string, string> = {
    indigo:  "border-indigo-800/40  text-indigo-300",
    violet:  "border-violet-800/40  text-violet-300",
    emerald: "border-emerald-800/40 text-emerald-300",
    amber:   "border-amber-800/40   text-amber-300",
  };
  return (
    <div className={`rounded-xl border ${accent[tone]} bg-gray-900/40 px-4 py-3.5 ${emphasis ? "ring-1 ring-emerald-800/40" : ""}`}>
      <p className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold mb-1">{label}</p>
      <p className={`text-lg font-bold tabular-nums ${accent[tone]}`}>{value}</p>
      {sub && <p className="text-[10px] text-gray-600 mt-0.5">{sub}</p>}
    </div>
  );
}
