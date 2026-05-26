"use client";

import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth";
import type { User } from "@/types";
import {
  Loader2Icon, CheckIcon, BellIcon,
  CpuIcon, TagIcon, UserIcon,
  ChevronDownIcon, ChevronRightIcon,
  KeyIcon, EyeIcon, EyeOffIcon, XCircleIcon, XIcon,
  BarChart3Icon, BrainIcon, Trash2Icon, AlertTriangleIcon, SearchIcon,
} from "lucide-react";
import { useNamespaceStore, NAMESPACE_TREE } from "@/store/namespace";

const TABS = [
  { key: "profile",       label: "Profile",      icon: UserIcon },
  { key: "memory",        label: "Memory",        icon: BrainIcon },
  { key: "usage",         label: "Token Usage",   icon: BarChart3Icon },
  { key: "topics",        label: "Topics",        icon: TagIcon },
  { key: "provider",      label: "AI Provider",   icon: CpuIcon },
  { key: "api-keys",      label: "API Keys",      icon: KeyIcon },
  { key: "notifications", label: "Notifications", icon: BellIcon },
] as const;

type Tab = typeof TABS[number]["key"];

export default function SettingsPage() {
  const searchParams = useSearchParams();
  const initialTab: Tab = (() => {
    const raw = (searchParams.get("tab") || "").toLowerCase();
    return (TABS.some(t => t.key === raw) ? (raw as Tab) : "profile");
  })();
  const [activeTab, setActiveTab] = useState<Tab>(initialTab);
  // Keep tab in sync with URL when user navigates back/forward.
  useEffect(() => {
    const raw = (searchParams.get("tab") || "").toLowerCase();
    if (TABS.some(t => t.key === raw)) setActiveTab(raw as Tab);
  }, [searchParams]);

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
          {activeTab === "memory"        && <MemoryPanel />}
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

// Model catalogues — one block per provider so the UI never offers a model
// the chosen provider cannot serve. Update these when providers ship a new
// model; the dropdown order is "newest / strongest first within each tier".
type LlmProvider = "openai" | "anthropic" | "google";
type EmbeddingProvider = "openai" | "gemini" | "voyage";

const LLM_MODELS: Record<LlmProvider, { cheap: string[]; quality: string[]; reasoning: string[] }> = {
  openai: {
    cheap:     ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-3.5-turbo"],
    quality:   ["gpt-5.4-mini", "gpt-5-mini", "gpt-4.1", "gpt-4o"],
    reasoning: ["gpt-5.4", "gpt-5", "o4-mini", "o3", "o3-mini", "o1"],
  },
  anthropic: {
    cheap:     ["claude-haiku-4-5", "claude-haiku-3-5"],
    quality:   ["claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-3-7"],
    reasoning: ["claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5"],
  },
  google: {
    cheap:     ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"],
    quality:   ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"],
    reasoning: ["gemini-2.5-pro", "gemini-1.5-pro"],
  },
};

const EMBEDDING_MODELS: Record<EmbeddingProvider, string[]> = {
  openai: ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"],
  gemini: ["gemini-embedding-2-preview", "text-embedding-004"],
  // Voyage adapter is reserved — the runtime falls back to OpenAI with a warning.
  voyage: ["voyage-3", "voyage-3-large", "voyage-code-3"],
};

interface ProviderConfig {
  llm_provider: LlmProvider;
  cheap_model: string;
  quality_model: string;
  reasoning_model: string;
  embedding_provider: EmbeddingProvider;
  embedding_model: string;
}

function ProviderPanel() {
  // Initial values match config.py defaults; overwritten by the GET response
  // which always returns the effective backend configuration.
  const [cfg, setCfg] = useState<ProviderConfig>({
    llm_provider: "openai",
    cheap_model: "gpt-4o-mini",
    quality_model: "gpt-5.4-mini",
    reasoning_model: "gpt-5.4",
    embedding_provider: "openai",
    embedding_model: "text-embedding-3-large",
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    // Backend always returns effective settings (DB row or system defaults),
    // so we can overwrite the local state unconditionally.
    api.get<ProviderConfig>("/settings/provider")
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

  // When the LLM provider changes, snap each tier's model to the provider's
  // first available option only if the current selection isn't valid for the
  // new provider — that way switching providers doesn't silently keep the
  // wrong model name selected and silently fall back at request time.
  function setLlmProvider(next: LlmProvider) {
    setCfg(c => {
      const cat = LLM_MODELS[next];
      return {
        ...c,
        llm_provider: next,
        cheap_model:     cat.cheap.includes(c.cheap_model)     ? c.cheap_model     : cat.cheap[0],
        quality_model:   cat.quality.includes(c.quality_model) ? c.quality_model   : cat.quality[0],
        reasoning_model: cat.reasoning.includes(c.reasoning_model) ? c.reasoning_model : cat.reasoning[0],
      };
    });
  }

  function setEmbeddingProvider(next: EmbeddingProvider) {
    setCfg(c => {
      const opts = EMBEDDING_MODELS[next];
      return {
        ...c,
        embedding_provider: next,
        embedding_model: opts.includes(c.embedding_model) ? c.embedding_model : opts[0],
      };
    });
  }

  const cat = LLM_MODELS[cfg.llm_provider];
  const embOpts = EMBEDDING_MODELS[cfg.embedding_provider];

  // Tiers carry an explanatory hint so users know which model gets called when.
  const tiers: { key: "cheap_model" | "quality_model" | "reasoning_model"; label: string; hint: string; options: string[] }[] = [
    { key: "cheap_model",     label: "Fast / Cheap Model", hint: "Used for query rewrite, intent classification, rerank, enrichment batches.", options: cat.cheap },
    { key: "quality_model",   label: "Quality Model",       hint: "Used for RAG synthesis, Genie hypothesize / critique, default Research Assistant turn.", options: cat.quality },
    { key: "reasoning_model", label: "Reasoning Model",     hint: "Used for Genie elaborate, PoC code, Deep Dive article, idea-combine fusion.", options: cat.reasoning },
  ];

  return (
    <Card
      icon={<CpuIcon size={16} className="text-purple-400" />}
      title="AI Provider Configuration"
      description="Changes take effect on the next request. Requires valid API keys in .env.local."
    >
      <div className="space-y-4">
        {/* LLM provider */}
        <div className="space-y-1">
          <label className="text-[11px] font-semibold text-gray-600 uppercase tracking-wider">LLM Provider</label>
          <select
            value={cfg.llm_provider}
            onChange={(e) => setLlmProvider(e.target.value as LlmProvider)}
            className="w-full bg-gray-800 border border-gray-700/60 rounded-xl px-3.5 py-2.5 text-sm text-gray-300 outline-none focus:border-indigo-500 transition-colors"
          >
            <option value="openai">openai</option>
            <option value="anthropic">anthropic</option>
            <option value="google">google</option>
          </select>
        </div>

        {/* Model tiers — gated to the chosen LLM provider */}
        {tiers.map(({ key, label, hint, options }) => (
          <div key={key} className="space-y-1">
            <label className="text-[11px] font-semibold text-gray-600 uppercase tracking-wider">{label}</label>
            <select
              value={cfg[key]}
              onChange={(e) => setCfg((c) => ({ ...c, [key]: e.target.value }))}
              className="w-full bg-gray-800 border border-gray-700/60 rounded-xl px-3.5 py-2.5 text-sm text-gray-300 outline-none focus:border-indigo-500 transition-colors"
            >
              {options.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
            <p className="text-[10px] text-gray-600 mt-1">{hint}</p>
          </div>
        ))}

        {/* Embedding provider + model (gated) */}
        <div className="pt-2 border-t border-gray-800/60 space-y-4">
          <div className="space-y-1">
            <label className="text-[11px] font-semibold text-gray-600 uppercase tracking-wider">Embedding Provider</label>
            <select
              value={cfg.embedding_provider}
              onChange={(e) => setEmbeddingProvider(e.target.value as EmbeddingProvider)}
              className="w-full bg-gray-800 border border-gray-700/60 rounded-xl px-3.5 py-2.5 text-sm text-gray-300 outline-none focus:border-indigo-500 transition-colors"
            >
              <option value="openai">openai</option>
              <option value="gemini">gemini</option>
              <option value="voyage">voyage (reserved — falls back to openai at runtime)</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-[11px] font-semibold text-gray-600 uppercase tracking-wider">Embedding Model</label>
            <select
              value={cfg.embedding_model}
              onChange={(e) => setCfg((c) => ({ ...c, embedding_model: e.target.value }))}
              className="w-full bg-gray-800 border border-gray-700/60 rounded-xl px-3.5 py-2.5 text-sm text-gray-300 outline-none focus:border-indigo-500 transition-colors"
            >
              {embOpts.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
            <p className="text-[10px] text-gray-600 mt-1">
              All embedding outputs are stored as 768-dim vectors (Matryoshka truncation when the model is larger).
            </p>
          </div>
        </div>
      </div>
      <SaveButton saving={saving} saved={saved} onClick={save} />
    </Card>
  );
}

/* ── API Keys Panel ───────────────────────────────────────────────────────── */
interface KeyStatus { is_set: boolean; from_env: boolean; is_overridden: boolean; masked: string }
interface ApiKeyState { openai: KeyStatus; anthropic: KeyStatus; google: KeyStatus; wolfram: KeyStatus }
type Provider = "openai" | "anthropic" | "google" | "wolfram";

function ApiKeysPanel() {
  const [status, setStatus] = useState<ApiKeyState | null>(null);
  // null = not editing; "" = editing with empty value; "sk-..." = editing with typed value
  const [editing, setEditing] = useState<Record<Provider, string | null>>({ openai: null, anthropic: null, google: null, wolfram: null });
  const [visible, setVisible] = useState<Record<Provider, boolean>>({ openai: false, anthropic: false, google: false, wolfram: false });
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
    { id: "openai",    label: "OpenAI",         placeholder: "sk-…",     hint: "Used for GPT models and text-embedding-3" },
    { id: "anthropic", label: "Anthropic",      placeholder: "sk-ant-…", hint: "Used for Claude models" },
    { id: "google",    label: "Google",         placeholder: "AIza…",    hint: "Used for Gemini models and embeddings" },
    { id: "wolfram",   label: "Wolfram Alpha",  placeholder: "XXXX-…",   hint: "Enables the Wolfram Alpha computation tool in the Research Assistant (free at developer.wolframalpha.com)" },
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
  // Zero-fill the selected range so the chart adapts to the FROM/TO inputs
  // and shows every day in the window — including ones with no LLM activity.
  // Without this, days with zero usage were silently skipped, making the
  // chart look like it didn't respond to range changes.
  const filledByDay = useMemo(() => {
    if (!data?.range?.from || !data?.range?.to) return data?.by_day || [];
    const start = new Date(data.range.from + "T00:00:00Z");
    const end = new Date(data.range.to + "T00:00:00Z");
    if (isNaN(start.getTime()) || isNaN(end.getTime()) || end < start) return data.by_day || [];
    const map: Record<string, UsageDayRow> = {};
    for (const d of data.by_day || []) map[d.date] = d;
    const out: UsageDayRow[] = [];
    const cur = new Date(start);
    // Hard cap at 400 days so a pathological range doesn't blow the page.
    let safety = 400;
    while (cur <= end && safety > 0) {
      const iso = cur.toISOString().slice(0, 10);
      out.push(map[iso] ?? { date: iso, input_tokens: 0, output_tokens: 0, total_tokens: 0, cost_usd: 0 });
      cur.setUTCDate(cur.getUTCDate() + 1);
      safety -= 1;
    }
    return out;
  }, [data]);
  // Adaptive binning: the per-day bar chart works for short ranges, but
  // when the user selects a full year the chart degenerates into 365
  // hair-thin bars that show no structure. Bin the rows into months,
  // weeks, or days depending on the window length so the graph stays
  // readable at every zoom level.
  type BinUnit = "day" | "week" | "month";
  const bins = useMemo(() => {
    const rows = filledByDay;
    if (rows.length === 0) return { unit: "day" as BinUnit, items: [] as { key: string; label: string; total_tokens: number; subtitle: string }[] };
    let unit: BinUnit = "day";
    if (rows.length > 90) unit = "month";
    else if (rows.length > 14) unit = "week";
    if (unit === "day") {
      return {
        unit,
        items: rows.map(d => ({
          key: d.date,
          label: d.date.slice(5),         // MM-DD
          total_tokens: d.total_tokens,
          subtitle: d.date,
        })),
      };
    }
    if (unit === "week") {
      // Anchor each week to the ISO Monday so labels are stable across
      // re-renders and the same date always falls in the same bucket.
      const bucket: Record<string, { total: number; first: string; last: string }> = {};
      for (const d of rows) {
        const dt = new Date(d.date + "T00:00:00Z");
        const dow = (dt.getUTCDay() + 6) % 7;   // Mon=0 … Sun=6
        const monday = new Date(dt);
        monday.setUTCDate(monday.getUTCDate() - dow);
        const key = monday.toISOString().slice(0, 10);
        if (!bucket[key]) bucket[key] = { total: 0, first: d.date, last: d.date };
        bucket[key].total += d.total_tokens;
        if (d.date < bucket[key].first) bucket[key].first = d.date;
        if (d.date > bucket[key].last) bucket[key].last = d.date;
      }
      return {
        unit,
        items: Object.entries(bucket)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([k, v]) => ({
            key: k,
            label: k.slice(5),           // MM-DD of the Monday
            total_tokens: v.total,
            subtitle: `Week of ${k} (${v.first} → ${v.last})`,
          })),
      };
    }
    // unit === "month"
    const bucket: Record<string, { total: number }> = {};
    for (const d of rows) {
      const key = d.date.slice(0, 7);    // YYYY-MM
      if (!bucket[key]) bucket[key] = { total: 0 };
      bucket[key].total += d.total_tokens;
    }
    return {
      unit,
      items: Object.entries(bucket)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([k, v]) => ({
          key: k,
          label: k.slice(5),             // MM
          total_tokens: v.total,
          subtitle: k,                   // YYYY-MM
        })),
    };
  }, [filledByDay]);
  const maxBinTotal = Math.max(1, ...bins.items.map(b => b.total_tokens));

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
            <StatCard label="LLM calls"     value={fmtNum(totals.calls)}         tone="amber" />
          </div>

          {/* Adaptive mini bar chart — uses zero-filled days for short
              ranges, weekly bins for month-scale windows, and monthly
              bins for year-scale windows. The bar width scales with the
              bin count so a year of usage shows ~12 month bars instead
              of 365 hair-thin daily slivers. */}
          {bins.items.length > 1 && (
            <div className="bg-gray-900/40 border border-gray-800/60 rounded-xl p-4 mb-5">
              <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-3">
                {bins.unit === "month" ? "Monthly total" : bins.unit === "week" ? "Weekly total" : "Daily total"}
              </p>
              <div className="flex items-end gap-1.5 h-24">
                {bins.items.map(b => {
                  const h = b.total_tokens === 0
                    ? 0
                    : Math.max(2, (b.total_tokens / maxBinTotal) * 100);
                  return (
                    <div
                      key={b.key}
                      className="flex-1 h-full flex items-end"
                      title={`${b.subtitle}: ${fmtNum(b.total_tokens)} tokens`}
                    >
                      <div
                        className={`w-full ${b.total_tokens === 0 ? "bg-gray-800/30" : "bg-indigo-600/50 hover:bg-indigo-500/80"} rounded-t transition-colors`}
                        style={{ height: `${h}%`, minHeight: b.total_tokens === 0 ? 0 : 2 }}
                      />
                    </div>
                  );
                })}
              </div>
              <div className="flex gap-1.5 mt-1.5">
                {bins.items.map(b => (
                  <span
                    key={b.key}
                    className={`flex-1 text-center text-[8px] font-mono ${b.total_tokens === 0 ? "text-gray-700" : "text-gray-600"}`}
                    title={`${b.subtitle}: ${fmtNum(b.total_tokens)} tokens`}
                  >
                    {b.label}
                  </span>
                ))}
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

/* ── Memory Panel ─────────────────────────────────────────────────────────── */
//
// Inspect, selectively delete, and clear long-term assistant memory.
// The endpoint at ``/settings/memory*`` returns medium-tier (session
// tree) and long-tier (per-namespace) entries — the two tiers that
// persist across turns. Short-term (chat) memory is intentionally
// NOT exposed here; it auto-prunes on session end and clearing
// long-term memory must not touch it (the user's explicit spec).

interface MemoryEntry {
  tier: "medium" | "long";
  namespace_key: string;
  subject: string;
  topic: string;
  key: string;
  value: string;
  type: string;
  memory_class: string;       // "semantic" | "episodic" | "procedural" | "preference" | "-"
  ts: string;
  source: string;
  ttl_days: number | null;
  origin_session: string | null;
  status: "active" | "stale" | "superseded";
  version: number;
  last_recalled_ts: string | null;
  // Supersession metadata. Set when a newer write was detected as
  // semantically near-identical and this entry was flagged as
  // superseded. The UI displays the chain and disables recall but
  // keeps the entry visible so the user can inspect / restore it.
  superseded_by_key?: string | null;
  superseded_at?: string | null;
  superseded_similarity?: number | null;
  root_session_id: string;
}

interface MemoryListResponse {
  entries: MemoryEntry[];
  counts: { medium: number; long: number };
  namespaces: string[];
  subjects: string[];
  topics: string[];
  class_counts: Record<string, number>;
  tiers: string[];
  memory_classes: string[];
  injection_enabled: boolean;
  /** Per-namespace overrides on top of the global toggle. A namespace
   *  in this map uses ``overrides[ns]`` as its effective state instead
   *  of ``injection_enabled``. */
  injection_overrides: Record<string, boolean>;
}

interface MemoryRevision {
  id: string;
  action: "create" | "update" | "delete" | "restore" | "supersede";
  status: string;
  value: string;
  previous_value: string | null;
  entry_type: string;
  source: string;
  ttl_days: number | null;
  confidence: number | null;
  created_at: string;
  subject: string;
  topic: string;
  origin_session_id: string | null;
  root_session_id: string;
  extras: Record<string, unknown>;
}

type MemoryClassFilter = "all" | "semantic" | "episodic" | "procedural" | "preference" | "-";

function MemoryPanel() {
  const [entries, setEntries] = useState<MemoryEntry[]>([]);
  const [namespaces, setNamespaces] = useState<string[]>([]);
  const [subjects, setSubjects] = useState<string[]>([]);
  const [topics, setTopics] = useState<string[]>([]);
  const [classCounts, setClassCounts] = useState<Record<string, number>>({});
  const [counts, setCounts] = useState<{ medium: number; long: number }>({ medium: 0, long: 0 });
  const [injectionEnabled, setInjectionEnabled] = useState<boolean>(true);
  const [injectionOverrides, setInjectionOverrides] = useState<Record<string, boolean>>({});
  const [adding, setAdding] = useState(false);
  // Root sessions list — needed so the Add modal can target a
  // concrete root bucket even when ``entries`` is empty (first-time
  // user with sessions but no stored memory yet). Loaded once on
  // mount.
  const [roots, setRoots] = useState<{
    id: string; title: string; namespace_key: string;
  }[]>([]);
  useEffect(() => {
    api.get<{ roots: typeof roots }>("/settings/memory/roots")
      .then(d => setRoots(d.roots || []))
      .catch(() => {});
  }, []);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [tierFilter, setTierFilter] = useState<"all" | "medium" | "long">("all");
  const [classFilter, setClassFilter] = useState<MemoryClassFilter>("all");
  const [nsFilter, setNsFilter] = useState<string>("");
  const [subjectFilter, setSubjectFilter] = useState<string>("");
  const [topicFilter, setTopicFilter] = useState<string>("");
  const [query, setQuery] = useState<string>("");
  const [confirming, setConfirming] = useState<"all" | string | null>(null);
  const [busy, setBusy] = useState(false);
  // History modal — populated when the user clicks "History" on a row.
  const [historyFor, setHistoryFor] = useState<MemoryEntry | null>(null);
  const [editingFor, setEditingFor] = useState<MemoryEntry | null>(null);

  const reload = async () => {
    setLoading(true);
    setErr(null);
    try {
      const params = new URLSearchParams();
      if (tierFilter !== "all") params.set("tier", tierFilter);
      if (nsFilter) params.set("namespace_key", nsFilter);
      if (subjectFilter) params.set("subject", subjectFilter);
      if (topicFilter) params.set("topic", topicFilter);
      if (classFilter !== "all") params.set("memory_class", classFilter);
      const data = await api.get<MemoryListResponse>(
        `/settings/memory${params.toString() ? `?${params}` : ""}`,
      );
      setEntries(data.entries || []);
      setNamespaces(data.namespaces || []);
      setSubjects(data.subjects || []);
      setTopics(data.topics || []);
      setClassCounts(data.class_counts || {});
      setCounts(data.counts || { medium: 0, long: 0 });
      setInjectionEnabled(data.injection_enabled);
      setInjectionOverrides(data.injection_overrides || {});
    } catch (e: unknown) {
      setErr((e as Error).message || "Failed to load memory");
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => { reload(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [tierFilter, nsFilter, classFilter, subjectFilter, topicFilter]);

  async function toggleInjection(opts?: { namespace?: string }) {
    setBusy(true);
    try {
      const ns = opts?.namespace || "";
      // Resolve effective current state for the scope being toggled
      // (global vs per-namespace override) so the flip computes
      // against the value actually in effect.
      const effective = ns
        ? (ns in injectionOverrides ? injectionOverrides[ns] : injectionEnabled)
        : injectionEnabled;
      const next = !effective;
      const body: Record<string, unknown> = { enabled: next };
      if (ns) body.namespace_key = ns;
      await api.post("/settings/memory/injection", body);
      if (ns) {
        setInjectionOverrides(prev => ({ ...prev, [ns]: next }));
      } else {
        setInjectionEnabled(next);
      }
    } catch (e: unknown) {
      setErr((e as Error).message || "Failed to toggle memory injection");
    } finally {
      setBusy(false);
    }
  }

  async function clearNamespaceOverride(ns: string) {
    setBusy(true);
    try {
      await api.delete(`/settings/memory/injection?namespace_key=${encodeURIComponent(ns)}`);
      setInjectionOverrides(prev => {
        const out = { ...prev };
        delete out[ns];
        return out;
      });
    } catch (e: unknown) {
      setErr((e as Error).message || "Failed to clear override");
    } finally {
      setBusy(false);
    }
  }

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter(e =>
      e.key.toLowerCase().includes(q) ||
      e.value.toLowerCase().includes(q) ||
      e.namespace_key.toLowerCase().includes(q) ||
      e.type.toLowerCase().includes(q)
    );
  }, [entries, query]);

  async function deleteOne(entry: MemoryEntry) {
    setBusy(true);
    try {
      await api.delete("/settings/memory", {
        tier: entry.tier,
        key: entry.key,
        namespace_key: entry.namespace_key,
        root_session_id: entry.root_session_id,
      });
      await reload();
    } catch (e: unknown) {
      setErr((e as Error).message || "Delete failed");
    } finally {
      setBusy(false);
    }
  }

  async function clearScope(scope: "all" | { ns: string }) {
    setBusy(true);
    try {
      const body = scope === "all"
        ? { tier: "long" as const, namespace_key: null }
        : { tier: "long" as const, namespace_key: scope.ns };
      await api.post("/settings/memory/clear", body);
      setConfirming(null);
      await reload();
    } catch (e: unknown) {
      setErr((e as Error).message || "Clear failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      icon={<BrainIcon size={16} className="text-fuchsia-400" />}
      title="Long-term Memory"
      description="Inspect what the Research Assistant remembers across turns. Pause injection without deleting, filter by class / scope, view history, restore prior versions, or delete entries individually."
    >
      {/* Pause toggle — the "stop using my memory without deleting it" knob */}
      <div className="bg-gray-900/40 border border-gray-800/60 rounded-xl px-3.5 py-2.5 space-y-2">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-gray-200">Memory injection · global</p>
            <p className="text-[10px] text-gray-500 mt-0.5">
              {injectionEnabled
                ? "RA is using stored memory to enrich planning and answers."
                : "Paused — RA will plan and answer WITHOUT injecting long-term memory. Entries stay safely stored."}
            </p>
          </div>
          <button
            onClick={() => toggleInjection()}
            disabled={busy}
            className="relative flex-shrink-0 disabled:opacity-40"
            aria-label={injectionEnabled ? "Pause memory injection" : "Resume memory injection"}
            title={injectionEnabled ? "Pause memory injection" : "Resume memory injection"}
          >
            <div
              className="w-10 h-5 rounded-full transition-colors"
              style={{ background: injectionEnabled ? "#a855f7" : "#374151" }}
            />
            <div
              className="absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform"
              style={{ left: injectionEnabled ? "calc(100% - 18px)" : "2px" }}
            />
          </button>
        </div>
        {/* Per-namespace overrides — shown only when at least one
            exists OR a namespace is currently selected, so the
            common case stays uncluttered. */}
        {(Object.keys(injectionOverrides).length > 0 || nsFilter) && (
          <div className="border-t border-gray-800/40 pt-2 space-y-1">
            <p className="text-[10px] uppercase tracking-wider text-gray-600 font-semibold">Per-namespace overrides</p>
            {Object.entries(injectionOverrides).map(([ns, enabled]) => (
              <div key={ns} className="flex items-center justify-between gap-2 text-[11px]">
                <span className="font-mono text-gray-400">{ns}</span>
                <div className="flex items-center gap-2">
                  <span className={enabled ? "text-emerald-300" : "text-amber-300"}>
                    {enabled ? "injecting" : "paused"}
                  </span>
                  <button
                    onClick={() => toggleInjection({ namespace: ns })}
                    disabled={busy}
                    className="text-[10px] text-indigo-300/80 hover:text-indigo-200 px-1.5 py-0.5 rounded"
                  >
                    flip
                  </button>
                  <button
                    onClick={() => clearNamespaceOverride(ns)}
                    disabled={busy}
                    className="text-[10px] text-gray-500 hover:text-gray-300 px-1.5 py-0.5 rounded"
                    title="Remove override — namespace falls back to global default"
                  >
                    reset
                  </button>
                </div>
              </div>
            ))}
            {nsFilter && !(nsFilter in injectionOverrides) && (
              <button
                onClick={() => toggleInjection({ namespace: nsFilter })}
                disabled={busy}
                className="text-[10px] text-indigo-300/80 hover:text-indigo-200"
                title={`Add an override for ${nsFilter}`}
              >
                + Add override for &quot;{nsFilter}&quot;
              </button>
            )}
          </div>
        )}
      </div>

      {/* Class filter — semantic / episodic / procedural cognitive taxonomy */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {([
          ["all",         "All classes"],
          ["semantic",    "Semantic (facts)"],
          ["episodic",    "Episodic (events)"],
          ["procedural",  "Procedural (how-to)"],
          ["preference",  "Preferences"],
          ["-",           "Other"],
        ] as [MemoryClassFilter, string][]).map(([k, label]) => {
          const count = k === "all"
            ? Object.values(classCounts).reduce((a, b) => a + b, 0)
            : (classCounts[k] || 0);
          const active = classFilter === k;
          if (k !== "all" && count === 0) return null;
          return (
            <button
              key={k}
              onClick={() => setClassFilter(k)}
              className={`text-[11px] px-2.5 py-1 rounded-full border transition-colors ${
                active
                  ? "bg-fuchsia-600/30 text-fuchsia-100 border-fuchsia-500/60"
                  : "bg-gray-900/40 text-gray-400 border-white/8 hover:text-gray-200 hover:border-white/20"
              }`}
              title={
                k === "semantic" ? "Facts and concepts (preferences, definitions, findings)"
                : k === "episodic" ? "Specific past experiences and events"
                : k === "procedural" ? "How-to knowledge (skills, procedures)"
                : k === "preference" ? "User-stated preferences"
                : k === "-" ? "Entries without a clean class mapping (hypothesis / context)"
                : "All classes"
              }
            >
              {label}
              <span className={"ml-1 font-mono " + (active ? "text-fuchsia-300/80" : "text-gray-600")}>
                {count}
              </span>
            </button>
          );
        })}
      </div>

      {/* Tier / namespace / search row */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1.5 bg-gray-900/60 border border-gray-800 rounded-lg p-1">
          {(["all", "medium", "long"] as const).map(k => (
            <button
              key={k}
              onClick={() => setTierFilter(k)}
              className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${
                tierFilter === k ? "bg-gray-800 text-white" : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {k === "all" ? `All (${counts.medium + counts.long})`
               : k === "medium" ? `Session tree (${counts.medium})`
               : `Namespace (${counts.long})`}
            </button>
          ))}
        </div>
        <select
          value={nsFilter}
          onChange={e => setNsFilter(e.target.value)}
          className="bg-gray-900 border border-gray-800 rounded-lg px-2 py-1 text-[11px] text-gray-300 focus:outline-none focus:border-indigo-600/50"
        >
          <option value="">All namespaces</option>
          {namespaces.map(ns => <option key={ns} value={ns}>{ns}</option>)}
        </select>
        {subjects.length > 1 && (
          <select
            value={subjectFilter}
            onChange={e => setSubjectFilter(e.target.value)}
            className="bg-gray-900 border border-gray-800 rounded-lg px-2 py-1 text-[11px] text-gray-300 focus:outline-none focus:border-indigo-600/50"
          >
            <option value="">All subjects</option>
            {subjects.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        )}
        {topics.length > 1 && (
          <select
            value={topicFilter}
            onChange={e => setTopicFilter(e.target.value)}
            className="bg-gray-900 border border-gray-800 rounded-lg px-2 py-1 text-[11px] text-gray-300 focus:outline-none focus:border-indigo-600/50"
          >
            <option value="">All topics</option>
            {topics.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        )}
        <div className="flex-1 min-w-[160px] relative">
          <SearchIcon size={11} className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-600" />
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search key / value / type"
            className="w-full bg-gray-900 border border-gray-800 rounded-lg pl-7 pr-2 py-1 text-[11px] text-gray-300 placeholder-gray-600 focus:outline-none focus:border-indigo-600/50"
          />
        </div>
        <button
          onClick={() => setAdding(true)}
          // The button is enabled whenever the user has at least one
          // root session OR at least one stored entry. Earlier we
          // required ``entries.length > 0`` which blocked users who
          // had RA sessions but no auto-memory writes yet — that
          // was the bug the user flagged.
          disabled={busy || (roots.length === 0 && entries.length === 0)}
          className="flex items-center gap-1.5 text-[11px] px-2.5 py-1 rounded-md border border-indigo-700/40 text-indigo-200 hover:bg-indigo-950/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          title={roots.length === 0 && entries.length === 0
            ? "Start a session first — manual writes need a root session to attach to"
            : "Add a new long-term memory entry"}
        >
          <BrainIcon size={11} /> Add memory
        </button>
        <button
          onClick={() => setConfirming("all")}
          disabled={busy || (counts.long === 0)}
          className="flex items-center gap-1.5 text-[11px] px-2.5 py-1 rounded-md border border-red-700/40 text-red-300 hover:bg-red-950/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <Trash2Icon size={11} /> Clear long-term
        </button>
      </div>

      {/* Confirm overlay */}
      <AnimatePresence>
        {confirming && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="bg-red-950/30 border border-red-700/40 rounded-xl p-3.5 text-xs space-y-2.5"
          >
            <div className="flex items-start gap-2">
              <AlertTriangleIcon size={14} className="text-red-300 flex-shrink-0 mt-0.5" />
              <div className="text-red-200">
                {confirming === "all"
                  ? "Clear ALL long-term namespace memory? This removes every stored fact, preference, and finding from every namespace and cannot be undone."
                  : `Clear all long-term memory inside namespace "${confirming}"? Other namespaces are untouched.`}
              </div>
            </div>
            <div className="flex items-center gap-2 justify-end">
              <button
                onClick={() => setConfirming(null)}
                disabled={busy}
                className="text-[11px] px-3 py-1 rounded-md text-gray-400 hover:text-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={() => clearScope(confirming === "all" ? "all" : { ns: confirming })}
                disabled={busy}
                className="text-[11px] px-3 py-1 rounded-md bg-red-700/40 border border-red-600/60 text-red-100 hover:bg-red-700/60 disabled:opacity-40"
              >
                {busy ? <Loader2Icon size={11} className="animate-spin inline" /> : "Confirm clear"}
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Status */}
      {loading && (
        <div className="flex items-center gap-2 text-xs text-gray-500 py-6 justify-center">
          <Loader2Icon size={13} className="animate-spin" /> Loading memory…
        </div>
      )}
      {err && !loading && (
        <div className="bg-red-950/30 border border-red-800/40 text-red-300 text-xs rounded-xl p-3">{err}</div>
      )}

      {!loading && filtered.length === 0 && (
        <p className="text-xs text-gray-600 text-center py-6">
          {entries.length === 0
            ? "No long-term memory stored yet. Memory grows as you converse with the Research Assistant."
            : "No entries match your filters."}
        </p>
      )}

      {/* Entries */}
      {!loading && filtered.length > 0 && (
        <div className="space-y-1.5">
          {filtered.map((e, i) => (
            <MemoryRow
              key={`${e.tier}-${e.namespace_key}-${e.key}-${i}`}
              entry={e}
              onDelete={() => deleteOne(e)}
              onClearNs={() => setConfirming(e.namespace_key)}
              onViewHistory={() => setHistoryFor(e)}
              onEdit={() => setEditingFor(e)}
              disabled={busy}
            />
          ))}
        </div>
      )}

      <div className="pt-2 border-t border-gray-800/40">
        <p className="text-[11px] text-gray-600 leading-relaxed">
          <strong className="text-gray-500">Memory classes</strong> — Semantic (facts), Episodic (past events), Procedural (how-to).<br />
          <strong className="text-gray-500">Session-tree memory</strong> — facts shared across a session and all its branches.<br />
          <strong className="text-gray-500">Namespace memory</strong> — facts that persist across every session in a subject/topic.<br />
          <strong className="text-gray-500">Short-term chat memory</strong> is NOT shown here and is never cleared by these controls — it lives on each chat and prunes itself.
        </p>
      </div>

      {/* History / restore modal */}
      {historyFor && (
        <MemoryHistoryModal
          entry={historyFor}
          onClose={() => setHistoryFor(null)}
          onRestored={async () => { setHistoryFor(null); await reload(); }}
        />
      )}
      {/* Add new / edit existing memory entry */}
      {adding && (
        <MemoryEditModal
          mode="add"
          namespaces={
            // Merge namespaces from existing entries AND root sessions
            // — the empty-memory case has zero entries but the user
            // may still have root sessions whose namespace_key we
            // can offer as a default.
            Array.from(new Set([
              ...namespaces,
              ...roots.map(r => r.namespace_key).filter(Boolean),
            ])).sort()
          }
          // Use the first entry's root if we have one, otherwise fall
          // back to the user's first root session (loaded once on
          // mount). Without this fallback the modal couldn't save
          // for first-time users who had RA sessions but no auto-
          // memory writes yet — that was the bug surfaced in the UI.
          defaultRootSessionId={entries[0]?.root_session_id || roots[0]?.id || ""}
          onClose={() => setAdding(false)}
          onSaved={async () => { setAdding(false); await reload(); }}
        />
      )}
      {editingFor && (
        <MemoryEditModal
          mode="edit"
          namespaces={namespaces}
          defaultRootSessionId={editingFor.root_session_id}
          initial={editingFor}
          onClose={() => setEditingFor(null)}
          onSaved={async () => { setEditingFor(null); await reload(); }}
        />
      )}
    </Card>
  );
}

/** Add-or-edit modal for long-term memory entries. */
function MemoryEditModal({
  mode, namespaces, defaultRootSessionId, initial, onClose, onSaved,
}: {
  mode: "add" | "edit";
  namespaces: string[];
  defaultRootSessionId: string;
  initial?: Partial<MemoryEntry>;
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const [tier, setTier] = useState<"medium" | "long">(
    (initial?.tier as "medium" | "long") || "long",
  );
  const [namespace, setNamespace] = useState<string>(
    initial?.namespace_key || namespaces[0] || "",
  );
  const [key, setKey] = useState<string>(initial?.key || "");
  const [value, setValue] = useState<string>(initial?.value || "");
  const [entryType, setEntryType] = useState<string>(initial?.type || "context");
  const [ttlDays, setTtlDays] = useState<string>(
    initial?.ttl_days != null ? String(initial.ttl_days) : "",
  );
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function save() {
    setBusy(true);
    setErr(null);
    try {
      await api.post("/settings/memory", {
        tier,
        key: key.trim(),
        value: value.trim(),
        entry_type: entryType,
        namespace_key: tier === "long" ? namespace : "",
        root_session_id: initial?.root_session_id || defaultRootSessionId,
        ttl_days: ttlDays ? Math.max(1, Math.min(365, parseInt(ttlDays, 10) || 0)) : null,
      });
      await onSaved();
    } catch (e: unknown) {
      setErr((e as Error).message || "Failed to save");
    } finally {
      setBusy(false);
    }
  }

  const canSave = key.trim().length > 0 && value.trim().length > 0
    && (tier === "medium" || namespace.length > 0);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-gray-950 border border-gray-800 rounded-2xl p-5 space-y-3"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3">
          <h3 className="text-sm font-semibold text-white flex-1">
            {mode === "add" ? "Add a long-term memory" : `Edit memory · ${initial?.key}`}
          </h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300 p-1">
            <XIcon size={14} />
          </button>
        </div>
        <p className="text-[11px] text-gray-600">
          Manual writes use the same store the assistant writes to. Every save is
          recorded in History so you can restore later.
        </p>

        <div className="grid grid-cols-2 gap-2.5">
          <div className="space-y-1">
            <label className="text-[10px] uppercase font-semibold text-gray-500 tracking-wider">Tier</label>
            <select
              value={tier}
              onChange={e => setTier(e.target.value as "medium" | "long")}
              disabled={mode === "edit"}
              className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2 py-1.5 text-[11px] text-gray-300"
            >
              <option value="long">Namespace (long)</option>
              <option value="medium">Session tree (medium)</option>
            </select>
          </div>
          {tier === "long" && (
            <div className="space-y-1">
              <label className="text-[10px] uppercase font-semibold text-gray-500 tracking-wider">Namespace</label>
              {/* Dropdown sourced from the user's own namespaces +
                  the platform's NAMESPACE_TREE topic keys. Falls
                  back to free-text only when nothing's available
                  (degenerate first-time state). The user explicitly
                  asked for this — namespaces should mirror the
                  Settings → Topics surface so manual writes can't
                  land in a typo'd bucket. */}
              {(() => {
                // Build the union: namespaces already in use +
                // every topic key from the global tree. Sorted,
                // deduped, and capped only for sanity.
                const treeKeys: string[] = [];
                for (const subject of NAMESPACE_TREE) {
                  for (const topic of subject.topics) {
                    if (topic.key) treeKeys.push(topic.key);
                  }
                }
                const all = Array.from(new Set([
                  ...namespaces,
                  ...treeKeys,
                  ...(namespace ? [namespace] : []),  // preserve current value
                ])).sort();
                if (all.length === 0) {
                  return (
                    <input
                      value={namespace}
                      onChange={e => setNamespace(e.target.value)}
                      placeholder="cs.AI"
                      disabled={mode === "edit"}
                      className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2 py-1.5 text-[11px] text-gray-300 placeholder-gray-700"
                    />
                  );
                }
                return (
                  <select
                    value={namespace}
                    onChange={e => setNamespace(e.target.value)}
                    disabled={mode === "edit"}
                    className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2 py-1.5 text-[11px] text-gray-300"
                  >
                    {/* If no namespace is selected yet, surface a
                        placeholder option so the dropdown's resting
                        state is honest. */}
                    {!namespace && <option value="">— pick a namespace —</option>}
                    {all.map(ns => (
                      <option key={ns} value={ns}>{ns}</option>
                    ))}
                  </select>
                );
              })()}
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2.5">
          <div className="space-y-1">
            <label className="text-[10px] uppercase font-semibold text-gray-500 tracking-wider">Key</label>
            <input
              value={key}
              onChange={e => setKey(e.target.value)}
              placeholder="user_pref_depth"
              disabled={mode === "edit"}
              className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2 py-1.5 text-[11px] text-gray-300 placeholder-gray-700 font-mono"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[10px] uppercase font-semibold text-gray-500 tracking-wider">Type</label>
            <select
              value={entryType}
              onChange={e => setEntryType(e.target.value)}
              className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2 py-1.5 text-[11px] text-gray-300"
            >
              <optgroup label="Semantic">
                <option value="finding">finding</option>
                <option value="concept">concept</option>
                <option value="paper_note">paper_note</option>
                <option value="preference">preference</option>
              </optgroup>
              <optgroup label="Episodic">
                <option value="episode">episode</option>
              </optgroup>
              <optgroup label="Procedural">
                <option value="skill">skill</option>
                <option value="procedure">procedure</option>
              </optgroup>
              <optgroup label="Other">
                <option value="hypothesis">hypothesis</option>
                <option value="context">context</option>
              </optgroup>
            </select>
          </div>
        </div>

        <div className="space-y-1">
          <label className="text-[10px] uppercase font-semibold text-gray-500 tracking-wider">Value</label>
          <textarea
            value={value}
            onChange={e => setValue(e.target.value)}
            rows={5}
            placeholder="The fact, preference, or procedure to remember."
            className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2 py-1.5 text-[11px] text-gray-300 placeholder-gray-700 resize-y leading-relaxed"
          />
        </div>

        <div className="space-y-1">
          <label className="text-[10px] uppercase font-semibold text-gray-500 tracking-wider">TTL (days)</label>
          <input
            value={ttlDays}
            onChange={e => setTtlDays(e.target.value.replace(/[^0-9]/g, ""))}
            placeholder="leave empty for evergreen"
            className="w-full bg-gray-900 border border-gray-800 rounded-lg px-2 py-1.5 text-[11px] text-gray-300 placeholder-gray-700"
          />
        </div>

        {err && (
          <div className="bg-red-950/30 border border-red-800/40 text-red-300 text-[11px] rounded-lg p-2">{err}</div>
        )}

        <div className="flex items-center justify-end gap-2 pt-1">
          <button
            onClick={onClose}
            disabled={busy}
            className="text-[11px] px-3 py-1.5 rounded-md text-gray-400 hover:text-gray-200"
          >
            Cancel
          </button>
          <button
            onClick={save}
            disabled={busy || !canSave}
            className="text-[11px] px-3 py-1.5 rounded-md bg-indigo-600/40 border border-indigo-500/60 text-indigo-100 hover:bg-indigo-600/60 disabled:opacity-40"
          >
            {busy ? <Loader2Icon size={11} className="animate-spin inline" /> : (mode === "add" ? "Save memory" : "Update")}
          </button>
        </div>
      </div>
    </div>
  );
}

function MemoryHistoryModal({
  entry, onClose, onRestored,
}: {
  entry: MemoryEntry;
  onClose: () => void;
  onRestored: () => Promise<void>;
}) {
  const [revisions, setRevisions] = useState<MemoryRevision[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [restoring, setRestoring] = useState<string | null>(null);

  useEffect(() => {
    let cancel = false;
    setLoading(true);
    setErr(null);
    const params = new URLSearchParams({
      tier: entry.tier,
      key: entry.key,
      namespace_key: entry.namespace_key || "",
    });
    api.get<{ revisions: MemoryRevision[]; count: number }>(`/settings/memory/revisions?${params}`)
      .then(d => { if (!cancel) setRevisions(d.revisions || []); })
      .catch((e: Error) => { if (!cancel) setErr(e.message || "Failed to load history"); })
      .finally(() => { if (!cancel) setLoading(false); });
    return () => { cancel = true; };
  }, [entry.tier, entry.key, entry.namespace_key]);

  async function restore(rev: MemoryRevision) {
    setRestoring(rev.id);
    try {
      await api.post("/settings/memory/restore", { revision_id: rev.id });
      await onRestored();
    } catch (e: unknown) {
      setErr((e as Error).message || "Restore failed");
    } finally {
      setRestoring(null);
    }
  }

  const actionColour = (a: MemoryRevision["action"]) =>
    a === "create"    ? "text-emerald-300 border-emerald-700/40 bg-emerald-950/30"
    : a === "update"    ? "text-indigo-300  border-indigo-700/40  bg-indigo-950/30"
    : a === "delete"    ? "text-red-300     border-red-700/40     bg-red-950/30"
    : a === "restore"   ? "text-fuchsia-300 border-fuchsia-700/40 bg-fuchsia-950/30"
    : "text-amber-300   border-amber-700/40   bg-amber-950/30";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl max-h-[80vh] overflow-y-auto bg-gray-950 border border-gray-800 rounded-2xl p-6 space-y-3"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3">
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-white">History · <span className="font-mono text-fuchsia-300">{entry.key}</span></h3>
            <p className="text-[11px] text-gray-500 mt-1">
              {entry.tier === "long" ? `namespace ${entry.namespace_key || "—"}` : "session tree"}
              {" · "}{entry.memory_class || "-"} · {revisions.length} revision{revisions.length === 1 ? "" : "s"}
            </p>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300 p-1">
            <XIcon size={14} />
          </button>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-xs text-gray-500 py-6 justify-center">
            <Loader2Icon size={13} className="animate-spin" /> Loading history…
          </div>
        )}
        {err && !loading && (
          <div className="bg-red-950/30 border border-red-800/40 text-red-300 text-xs rounded-xl p-3">{err}</div>
        )}
        {!loading && revisions.length === 0 && (
          <p className="text-xs text-gray-600 text-center py-6">
            No revision history recorded for this entry yet. (Revisions are
            tracked from now onward; entries created before this release
            won&apos;t show prior history.)
          </p>
        )}
        {!loading && revisions.length > 0 && (
          <div className="space-y-2">
            {revisions.map((r) => {
              const dt = r.created_at ? new Date(r.created_at) : null;
              const showSideBySide = !!(r.previous_value && r.value);
              return (
                <div key={r.id} className="bg-gray-900/60 border border-gray-800/60 rounded-xl p-3 space-y-2">
                  <div className="flex items-center gap-2 text-[10px]">
                    <span className={`uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded border ${actionColour(r.action)}`}>
                      {r.action}
                    </span>
                    <span className="text-gray-500">{r.entry_type}</span>
                    <span className="text-gray-600">·</span>
                    <span className="text-gray-500">source: {r.source}</span>
                    <div className="flex-1" />
                    {dt && !isNaN(dt.getTime()) && (
                      <span className="text-gray-600 font-mono">{dt.toLocaleString()}</span>
                    )}
                  </div>
                  {showSideBySide ? (
                    // Two-column line-level diff: previous on the
                    // left in red, current on the right in green.
                    // The DiffPanel component below handles word-level
                    // intra-line highlighting so changed tokens are
                    // visible at a glance, not just changed lines.
                    <DiffPanel prev={r.previous_value || ""} next={r.value || ""} />
                  ) : (
                    <>
                      {r.previous_value && (
                        <div className="text-[11px]">
                          <span className="text-[9px] uppercase text-gray-600 tracking-wider mr-1">prev</span>
                          <span className="text-red-200/80 line-through whitespace-pre-wrap">{r.previous_value}</span>
                        </div>
                      )}
                      {r.value && (
                        <div className="text-[11px]">
                          <span className="text-[9px] uppercase text-gray-600 tracking-wider mr-1">value</span>
                          <span className="text-emerald-200/90 whitespace-pre-wrap">{r.value}</span>
                        </div>
                      )}
                    </>
                  )}
                  {r.action !== "restore" && (
                    <div className="flex items-center justify-end">
                      <button
                        onClick={() => restore(r)}
                        disabled={!!restoring}
                        className="text-[10px] px-2.5 py-1 rounded-md border border-fuchsia-700/40 text-fuchsia-200 hover:bg-fuchsia-950/30 disabled:opacity-40 transition-colors"
                      >
                        {restoring === r.id ? <Loader2Icon size={10} className="animate-spin inline" /> : "Restore this version"}
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Two-column side-by-side diff for a memory revision.
 *
 * Renders the previous value on the LEFT (red tint, line-through on
 * removed words) and the current value on the RIGHT (green tint,
 * highlighted on inserted words). Word-level diff runs over each
 * paired line so the user sees exactly which tokens changed instead
 * of just "lines differ". For multi-line entries with no shared
 * lines we fall back to a whole-blob diff per side.
 *
 * Deliberately implemented locally (no react-diff-viewer dependency)
 * to keep the bundle small and avoid adding a build-time peer. The
 * algorithm is the textbook LCS-based diff on whitespace-tokenised
 * words — fast enough for memory entries (capped at 4000 chars) and
 * good enough to surface real intent.
 */
function DiffPanel({ prev, next }: { prev: string; next: string }) {
  const tokens = useMemo(() => diffWords(prev, next), [prev, next]);
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 8,
      }}
    >
      <div
        className="bg-red-950/20 border border-red-800/30 rounded-lg p-2 text-[11px] whitespace-pre-wrap leading-relaxed"
        title="Previous value"
      >
        <div className="text-[9px] uppercase tracking-wider text-red-300/70 font-semibold mb-1">previous</div>
        {tokens.map((t, i) =>
          t.type === "removed" ? (
            <span key={i} className="bg-red-900/40 text-red-100 rounded px-0.5">
              {t.text}
            </span>
          ) : t.type === "equal" ? (
            <span key={i} className="text-red-200/70">{t.text}</span>
          ) : null,
        )}
      </div>
      <div
        className="bg-emerald-950/20 border border-emerald-800/30 rounded-lg p-2 text-[11px] whitespace-pre-wrap leading-relaxed"
        title="Current / new value"
      >
        <div className="text-[9px] uppercase tracking-wider text-emerald-300/70 font-semibold mb-1">current</div>
        {tokens.map((t, i) =>
          t.type === "added" ? (
            <span key={i} className="bg-emerald-900/40 text-emerald-100 rounded px-0.5">
              {t.text}
            </span>
          ) : t.type === "equal" ? (
            <span key={i} className="text-emerald-200/70">{t.text}</span>
          ) : null,
        )}
      </div>
    </div>
  );
}

/** Diff token kind. ``equal`` appears on both sides; ``removed`` only
 *  in ``previous``; ``added`` only in ``current``. */
type DiffToken = { type: "equal" | "added" | "removed"; text: string };

/**
 * Word-level diff via LCS (longest common subsequence). Tokenises on
 * whitespace but preserves the whitespace in the output so the
 * rendered text stays readable.
 *
 * Complexity: O(n*m) time and memory in the token counts. Fine for
 * normal memory entries (under ~1500 tokens combined → ~2M ops,
 * runs in <50ms). For pathological inputs (e.g. a user manually
 * pasting a 10k-token paper body into a memory entry) we'd block
 * the main thread for seconds. The ``MAX_TOKENS`` cap below falls
 * back to a coarse "whole-side" diff in that case — the rendered
 * view is less granular but the browser stays responsive.
 */
const _DIFF_TOKEN_BUDGET = 1500;   // n + m ceiling for the LCS path

function diffWords(a: string, b: string): DiffToken[] {
  // Tokenise with a regex that captures whitespace runs as their
  // own tokens. That way ``"hello world"`` becomes
  // ["hello", " ", "world"] and we can render the spaces exactly as
  // they appeared.
  const tokA = a.split(/(\s+)/).filter(s => s.length > 0);
  const tokB = b.split(/(\s+)/).filter(s => s.length > 0);
  const n = tokA.length;
  const m = tokB.length;

  // Pathological-size guard: when the combined token count exceeds
  // the LCS budget, fall back to a coarse "removed-then-added" view
  // so the main thread doesn't hang on a quadratic walk. The user
  // still sees both sides; they just don't get the per-token
  // alignment highlight.
  if (n + m > _DIFF_TOKEN_BUDGET) {
    return [
      { type: "removed", text: a },
      { type: "added", text: b },
    ];
  }

  // LCS DP table. Each cell stores the length of the longest common
  // subsequence of tokA[0..i] and tokB[0..j].
  const dp: number[][] = Array(n + 1).fill(null).map(() => Array(m + 1).fill(0));
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (tokA[i - 1] === tokB[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }
  // Walk backwards from (n, m) emitting tokens in REVERSE order, then
  // reverse at the end. Standard LCS reconstruction.
  const out: DiffToken[] = [];
  let i = n;
  let j = m;
  while (i > 0 && j > 0) {
    if (tokA[i - 1] === tokB[j - 1]) {
      out.push({ type: "equal", text: tokA[i - 1] });
      i--; j--;
    } else if (dp[i - 1][j] >= dp[i][j - 1]) {
      out.push({ type: "removed", text: tokA[i - 1] });
      i--;
    } else {
      out.push({ type: "added", text: tokB[j - 1] });
      j--;
    }
  }
  while (i > 0) { out.push({ type: "removed", text: tokA[i - 1] }); i--; }
  while (j > 0) { out.push({ type: "added", text: tokB[j - 1] }); j--; }
  return out.reverse();
}

function MemoryRow({ entry, onDelete, onClearNs, onViewHistory, onEdit, disabled }: {
  entry: MemoryEntry;
  onDelete: () => void;
  onClearNs: () => void;
  onViewHistory: () => void;
  onEdit: () => void;
  disabled: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const ts = entry.ts ? new Date(entry.ts) : null;
  const tsStr = ts && !isNaN(ts.getTime()) ? ts.toLocaleString() : "—";
  const tierLabel = entry.tier === "long" ? "namespace" : "session tree";
  const tierColour = entry.tier === "long"
    ? "text-fuchsia-300 border-fuchsia-700/40 bg-fuchsia-950/30"
    : "text-teal-300 border-teal-700/40 bg-teal-950/30";
  const classColour =
    entry.memory_class === "semantic"   ? "text-emerald-300 border-emerald-700/40 bg-emerald-950/20"
    : entry.memory_class === "episodic" ? "text-sky-300 border-sky-700/40 bg-sky-950/20"
    : entry.memory_class === "procedural" ? "text-amber-300 border-amber-700/40 bg-amber-950/20"
    : entry.memory_class === "preference" ? "text-violet-300 border-violet-700/40 bg-violet-950/20"
    : "text-gray-500 border-gray-700/40 bg-gray-900/30";
  const statusColour =
    entry.status === "stale"
      ? "text-amber-300 border-amber-700/40 bg-amber-950/30"
    : entry.status === "superseded"
      ? "text-violet-300 border-violet-700/40 bg-violet-950/30"
      : "text-emerald-300 border-emerald-700/40 bg-emerald-950/30";
  return (
    <div className="bg-gray-900/40 border border-gray-800/60 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded(v => !v)}
        className="w-full px-3 py-2 flex items-center gap-2 text-left hover:bg-gray-900/80 transition-colors"
      >
        {expanded ? <ChevronDownIcon size={11} className="text-gray-600" /> : <ChevronRightIcon size={11} className="text-gray-600" />}
        <span className={`text-[9px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded border ${tierColour}`}>
          {tierLabel}
        </span>
        {entry.memory_class && entry.memory_class !== "-" && (
          <span className={`text-[9px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded border ${classColour}`}
                title={`Cognitive class: ${entry.memory_class}`}>
            {entry.memory_class}
          </span>
        )}
        <span
          className={`text-[9px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded border ${statusColour}`}
          title={
            entry.status === "stale"
              ? "Past its TTL — value may be outdated"
            : entry.status === "superseded"
              ? `Superseded by '${entry.superseded_by_key || "newer entry"}'` +
                (entry.superseded_similarity
                  ? ` (similarity ${(entry.superseded_similarity * 100).toFixed(0)}%)`
                  : "")
              + " — preserved for audit, not injected into recall"
              : "Active and within TTL"
          }
        >
          {entry.status}
        </span>
        {entry.namespace_key && (
          <span className="text-[10px] font-mono text-gray-500">{entry.namespace_key}</span>
        )}
        <span className="text-[10px] font-mono text-gray-300 flex-1 truncate">{entry.key}</span>
        <span className="text-[10px] text-gray-600 hidden md:inline">{entry.type}</span>
        {entry.version > 1 && (
          <span className="text-[9px] text-gray-600 font-mono" title={`Version ${entry.version}`}>v{entry.version}</span>
        )}
      </button>
      {expanded && (
        <div className="px-3 pb-3 pt-1 border-t border-gray-800/40 text-[11px] text-gray-300 space-y-2">
          <div className="whitespace-pre-wrap leading-relaxed">{entry.value}</div>
          <div className="flex items-center gap-3 text-[10px] text-gray-600 flex-wrap">
            <span>updated: {tsStr}</span>
            <span>source: {entry.source}</span>
            {entry.last_recalled_ts && (
              <span title="Last time RA surfaced this entry to a turn">
                last used: {(() => {
                  const d = new Date(entry.last_recalled_ts);
                  return !isNaN(d.getTime()) ? d.toLocaleString() : entry.last_recalled_ts;
                })()}
              </span>
            )}
            {entry.ttl_days && <span>ttl: {entry.ttl_days}d</span>}
            {entry.subject && <span>subject: {entry.subject}</span>}
            {entry.topic && <span>topic: {entry.topic}</span>}
            <div className="flex-1" />
            <button
              onClick={onEdit}
              disabled={disabled}
              className="text-[10px] text-emerald-400/80 hover:text-emerald-300 transition-colors"
              title="Edit this memory's value or type"
            >
              Edit
            </button>
            <button
              onClick={onViewHistory}
              disabled={disabled}
              className="text-[10px] text-indigo-400/80 hover:text-indigo-300 transition-colors"
              title="View version history and restore prior values"
            >
              History
            </button>
            {entry.tier === "long" && (
              <button
                onClick={onClearNs}
                disabled={disabled}
                className="text-[10px] text-amber-400/80 hover:text-amber-300 transition-colors"
              >
                Clear all in &quot;{entry.namespace_key}&quot;
              </button>
            )}
            <button
              onClick={onDelete}
              disabled={disabled}
              className="flex items-center gap-1 text-[10px] text-red-400/80 hover:text-red-300 transition-colors"
            >
              <Trash2Icon size={10} /> Delete
            </button>
          </div>
        </div>
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
