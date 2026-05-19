"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth";

// ─── Types ────────────────────────────────────────────────────────────────────

type AdminUserItem = {
  id: string;
  email: string;
  display_name: string;
  is_active: boolean;
  is_admin: boolean;
  onboarding_complete: boolean;
  created_at: string;
};

type Analytics = {
  users: { total: number; active: number; admins: number; new_last_7_days: number };
  content: { papers: number; bookmarks: number; ideas: number; assistant_sessions: number };
  activity: { assistant_messages_last_7_days: number; assistant_messages_last_30_days: number };
  generated_at: string;
};

type FeatureCatalog = Record<string, { default: boolean; label: string; description: string }>;
type FeatureMap = Record<string, boolean>;
type UserFeaturesPayload = {
  overrides: Record<string, boolean>;
  effective: FeatureMap;
  defaults: FeatureMap;
};

// ─── Page ────────────────────────────────────────────────────────────────────

export default function AdminPage() {
  const router = useRouter();
  const { user } = useAuthStore();
  const [authorised, setAuthorised] = useState<"checking" | "yes" | "no">("checking");

  // Data
  const [users, setUsers] = useState<AdminUserItem[]>([]);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [catalog, setCatalog] = useState<FeatureCatalog>({});
  const [globalFeatures, setGlobalFeatures] = useState<FeatureMap>({});

  // UI
  const [selectedUser, setSelectedUser] = useState<AdminUserItem | null>(null);
  const [userFeatures, setUserFeatures] = useState<UserFeaturesPayload | null>(null);
  const [tab, setTab] = useState<"overview" | "features" | "users">("overview");
  const [filter, setFilter] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  // Create-user form state — collapsed by default to keep the Users tab clean.
  const [showCreateUser, setShowCreateUser] = useState(false);
  const [newUserForm, setNewUserForm] = useState<{ email: string; password: string; display_name: string; is_admin: boolean }>(
    { email: "", password: "", display_name: "", is_admin: false }
  );

  // ── Access guard ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (user && !user.is_admin) {
      router.replace("/feed");
      return;
    }
    if (user?.is_admin) setAuthorised("yes");
  }, [user, router]);

  // ── Initial load ──────────────────────────────────────────────────────────
  const reload = useCallback(async () => {
    setErr(null);
    try {
      const [u, a, gf, cat] = await Promise.all([
        api.get<AdminUserItem[]>("/admin/users"),
        api.get<Analytics>("/admin/analytics"),
        api.get<FeatureMap>("/admin/features"),
        api.get<FeatureCatalog>("/settings/features/catalog"),
      ]);
      setUsers(u);
      setAnalytics(a);
      setGlobalFeatures(gf);
      setCatalog(cat);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to load admin data";
      setErr(message);
      if (/403/.test(message)) setAuthorised("no");
    }
  }, []);

  useEffect(() => {
    if (authorised === "yes") void reload();
  }, [authorised, reload]);

  // ── Per-user features ─────────────────────────────────────────────────────
  async function openUser(u: AdminUserItem) {
    setSelectedUser(u);
    setUserFeatures(null);
    try {
      const data = await api.get<UserFeaturesPayload>(`/admin/users/${u.id}/features`);
      setUserFeatures(data);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to load user features");
    }
  }

  async function patchUserOverride(featureKey: string, value: boolean | null) {
    if (!selectedUser) return;
    setBusy(true);
    try {
      const res = await api.patch<{ effective: FeatureMap }>(
        `/admin/users/${selectedUser.id}/features`,
        { [featureKey]: value },
      );
      setUserFeatures((prev) => {
        if (!prev) return prev;
        const overrides = { ...prev.overrides };
        if (value === null) delete overrides[featureKey];
        else overrides[featureKey] = value;
        return { ...prev, overrides, effective: res.effective };
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to update override");
    }
    setBusy(false);
  }

  async function patchGlobalFeature(featureKey: string, value: boolean) {
    setBusy(true);
    try {
      const res = await api.patch<FeatureMap>("/admin/features", { [featureKey]: value });
      // ``/admin/features`` PATCH returns the full merged settings dict.
      setGlobalFeatures((prev) => ({ ...prev, ...res }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to update global feature");
    }
    setBusy(false);
  }

  async function patchUser(id: string, patch: { is_active?: boolean; is_admin?: boolean }) {
    setBusy(true);
    try {
      const updated = await api.patch<AdminUserItem>(`/admin/users/${id}`, patch);
      setUsers((prev) => prev.map((u) => (u.id === id ? updated : u)));
      if (selectedUser?.id === id) setSelectedUser(updated);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to update user");
    }
    setBusy(false);
  }

  async function createUser() {
    if (busy) return;
    const { email, password, display_name, is_admin } = newUserForm;
    if (!email.trim() || password.length < 8) {
      setErr("Email and password (≥ 8 chars) are required.");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const created = await api.post<AdminUserItem>("/admin/users", {
        email: email.trim(),
        password,
        display_name: display_name.trim() || undefined,
        is_admin,
      });
      setUsers((prev) => [created, ...prev]);
      setNewUserForm({ email: "", password: "", display_name: "", is_admin: false });
      setShowCreateUser(false);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to create user");
    }
    setBusy(false);
  }

  async function deleteUser(u: AdminUserItem) {
    if (busy || u.id === user?.id) return;
    if (typeof window !== "undefined" && !window.confirm(`Delete account ${u.email}? This cannot be undone.`)) return;
    setBusy(true);
    try {
      await api.delete(`/admin/users/${u.id}`);
      setUsers((prev) => prev.filter((x) => x.id !== u.id));
      if (selectedUser?.id === u.id) {
        setSelectedUser(null);
        setUserFeatures(null);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to delete user");
    }
    setBusy(false);
  }

  async function resetUserPassword(u: AdminUserItem) {
    if (busy) return;
    const next = typeof window !== "undefined"
      ? window.prompt(`Set a new password for ${u.email} (min 8 chars):`)
      : null;
    if (!next || next.length < 8) return;
    setBusy(true);
    try {
      await api.post(`/admin/users/${u.id}/password`, { new_password: next });
      if (typeof window !== "undefined") window.alert("Password updated.");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to reset password");
    }
    setBusy(false);
  }

  if (authorised === "checking") {
    return <div style={{ padding: 32, color: "var(--rf-text4)" }}>Checking access…</div>;
  }
  if (authorised === "no") {
    return <div style={{ padding: 32, color: "var(--rf-text4)" }}>Admin privileges required.</div>;
  }

  const filteredUsers = filter.trim()
    ? users.filter((u) =>
        u.email.toLowerCase().includes(filter.toLowerCase())
        || u.display_name.toLowerCase().includes(filter.toLowerCase()))
    : users;

  return (
    <div style={{ padding: 28, color: "var(--rf-text1)", overflowY: "auto", height: "100vh" }}>
      <header style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 18 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700 }}>Admin Panel</h1>
        <p style={{ fontSize: 11, color: "var(--rf-text4)" }}>
          Logged in as <b>{user?.email}</b>
        </p>
      </header>

      {err && (
        <div style={{
          padding: 10, borderRadius: 6, marginBottom: 16,
          background: "rgba(239,68,68,0.1)", color: "#ef4444",
          border: "1px solid rgba(239,68,68,0.3)", fontSize: 12,
        }}>{err}</div>
      )}

      <nav style={{ display: "flex", gap: 4, marginBottom: 16, borderBottom: "1px solid var(--rf-border)" }}>
        {(["overview", "features", "users"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: "8px 14px", fontSize: 12, fontWeight: 600, textTransform: "capitalize",
              border: "none", borderBottom: "2px solid " + (tab === t ? "#818cf8" : "transparent"),
              color: tab === t ? "var(--rf-text1)" : "var(--rf-text4)",
              background: "none", cursor: "pointer",
            }}
          >
            {t}
          </button>
        ))}
      </nav>

      {/* ── Overview tab ─────────────────────────────────────────── */}
      {tab === "overview" && analytics && (
        <section>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(180px,1fr))", gap: 10 }}>
            <Stat label="Users (total)" value={analytics.users.total} />
            <Stat label="Active users" value={analytics.users.active} />
            <Stat label="Admins" value={analytics.users.admins} />
            <Stat label="New users · 7d" value={analytics.users.new_last_7_days} />
            <Stat label="Papers" value={analytics.content.papers} />
            <Stat label="Bookmarks" value={analytics.content.bookmarks} />
            <Stat label="Ideas (capsules)" value={analytics.content.ideas} />
            <Stat label="Assistant sessions" value={analytics.content.assistant_sessions} />
            <Stat label="RA messages · 7d" value={analytics.activity.assistant_messages_last_7_days} />
            <Stat label="RA messages · 30d" value={analytics.activity.assistant_messages_last_30_days} />
          </div>
          <p style={{ fontSize: 10, color: "var(--rf-text5)", marginTop: 8 }}>
            Generated {new Date(analytics.generated_at).toLocaleString()}
          </p>
        </section>
      )}

      {/* ── Global features tab ──────────────────────────────────── */}
      {tab === "features" && (
        <section>
          <p style={{ fontSize: 11, color: "var(--rf-text4)", marginBottom: 12 }}>
            Flags here apply to <b>all users</b>. Per-user overrides on the Users tab take precedence.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {Object.entries(catalog).map(([key, meta]) => {
              const on = !!globalFeatures[key];
              return (
                <div
                  key={key}
                  style={{
                    display: "flex", alignItems: "center", gap: 12, padding: 12,
                    borderRadius: 8, background: "var(--rf-surface1)",
                    border: "1px solid var(--rf-border)",
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>{meta.label}</div>
                    <div style={{ fontSize: 11, color: "var(--rf-text4)", marginTop: 2 }}>
                      {meta.description}
                    </div>
                    <code style={{ fontSize: 10, color: "var(--rf-text5)" }}>{key}</code>
                  </div>
                  <button
                    onClick={() => patchGlobalFeature(key, !on)}
                    disabled={busy}
                    style={togglePill(on)}
                  >
                    {on ? "Enabled" : "Disabled"}
                  </button>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* ── Users tab ────────────────────────────────────────────── */}
      {tab === "users" && (
        <section style={{ display: "grid", gridTemplateColumns: selectedUser ? "1fr 1fr" : "1fr", gap: 16 }}>
          <div>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              <input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Filter users by name or email…"
                style={{
                  flex: 1, padding: "7px 10px", borderRadius: 6,
                  background: "var(--rf-surface2)", color: "var(--rf-text1)",
                  border: "1px solid var(--rf-border)", fontSize: 12,
                }}
              />
              <button
                onClick={() => setShowCreateUser((s) => !s)}
                style={{
                  padding: "7px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600,
                  background: showCreateUser ? "var(--rf-surface3)" : "linear-gradient(135deg,#6366f1,#8b5cf6)",
                  color: showCreateUser ? "var(--rf-text3)" : "#fff",
                  border: showCreateUser ? "1px solid var(--rf-border)" : "none",
                  cursor: "pointer",
                }}
              >
                {showCreateUser ? "Cancel" : "+ New user"}
              </button>
            </div>
            {showCreateUser && (
              <div style={{
                padding: 12, borderRadius: 8, marginBottom: 12,
                background: "var(--rf-surface1)", border: "1px solid var(--rf-border)",
                display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8,
              }}>
                <input
                  value={newUserForm.email}
                  onChange={(e) => setNewUserForm((f) => ({ ...f, email: e.target.value }))}
                  placeholder="Email"
                  style={createInputStyle}
                />
                <input
                  value={newUserForm.display_name}
                  onChange={(e) => setNewUserForm((f) => ({ ...f, display_name: e.target.value }))}
                  placeholder="Display name (optional)"
                  style={createInputStyle}
                />
                <input
                  type="password"
                  value={newUserForm.password}
                  onChange={(e) => setNewUserForm((f) => ({ ...f, password: e.target.value }))}
                  placeholder="Password (min 8 chars)"
                  style={createInputStyle}
                />
                <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--rf-text3)" }}>
                  <input
                    type="checkbox"
                    checked={newUserForm.is_admin}
                    onChange={(e) => setNewUserForm((f) => ({ ...f, is_admin: e.target.checked }))}
                  />
                  Admin (full panel access)
                </label>
                <div style={{ gridColumn: "1 / -1", display: "flex", justifyContent: "flex-end" }}>
                  <button
                    onClick={createUser}
                    disabled={busy || !newUserForm.email.trim() || newUserForm.password.length < 8}
                    style={{
                      padding: "7px 16px", borderRadius: 6, fontSize: 12, fontWeight: 600,
                      background: "linear-gradient(135deg,#6366f1,#8b5cf6)", color: "#fff",
                      border: "none", cursor: busy ? "wait" : "pointer",
                      opacity: !newUserForm.email.trim() || newUserForm.password.length < 8 ? 0.5 : 1,
                    }}
                  >
                    Create
                  </button>
                </div>
              </div>
            )}
            <div style={{ border: "1px solid var(--rf-border)", borderRadius: 8, overflow: "hidden" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ background: "var(--rf-surface2)" }}>
                    <th style={th}>Email</th>
                    <th style={th}>Name</th>
                    <th style={th}>Active</th>
                    <th style={th}>Admin</th>
                    <th style={th}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredUsers.map((u) => (
                    <tr
                      key={u.id}
                      style={{
                        borderTop: "1px solid var(--rf-border)",
                        background: selectedUser?.id === u.id ? "var(--rf-surface2)" : "transparent",
                        cursor: "pointer",
                      }}
                      onClick={() => openUser(u)}
                    >
                      <td style={td}>{u.email}</td>
                      <td style={td}>{u.display_name}</td>
                      <td style={td}>
                        <button
                          onClick={(e) => { e.stopPropagation(); patchUser(u.id, { is_active: !u.is_active }); }}
                          disabled={busy || u.id === user?.id}
                          title={u.id === user?.id ? "Cannot deactivate yourself" : undefined}
                          style={pillStyle(u.is_active, u.id === user?.id)}
                        >
                          {u.is_active ? "Active" : "Inactive"}
                        </button>
                      </td>
                      <td style={td}>
                        <button
                          onClick={(e) => { e.stopPropagation(); patchUser(u.id, { is_admin: !u.is_admin }); }}
                          disabled={busy || u.id === user?.id}
                          title={u.id === user?.id ? "Cannot remove your own admin" : undefined}
                          style={pillStyle(u.is_admin, u.id === user?.id)}
                        >
                          {u.is_admin ? "Admin" : "Member"}
                        </button>
                      </td>
                      <td style={td}>
                        <div style={{ display: "flex", gap: 4 }} onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => resetUserPassword(u)}
                            disabled={busy}
                            title="Reset password"
                            style={iconBtn}
                          >
                            Reset PW
                          </button>
                          <button
                            onClick={() => deleteUser(u)}
                            disabled={busy || u.id === user?.id}
                            title={u.id === user?.id ? "Cannot delete your own account" : "Delete user"}
                            style={{ ...iconBtn, borderColor: "rgba(239,68,68,0.35)", color: "#ef4444" }}
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {selectedUser && (
            <aside style={{
              borderRadius: 10, padding: 14,
              background: "var(--rf-surface1)", border: "1px solid var(--rf-border)",
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600 }}>{selectedUser.display_name}</div>
                  <div style={{ fontSize: 11, color: "var(--rf-text4)" }}>{selectedUser.email}</div>
                </div>
                <button
                  onClick={() => { setSelectedUser(null); setUserFeatures(null); }}
                  style={{ background: "none", border: "none", color: "var(--rf-text4)", cursor: "pointer", fontSize: 12 }}
                >
                  Close ×
                </button>
              </div>

              <h3 style={{
                marginTop: 14, marginBottom: 8, fontSize: 11, letterSpacing: 1,
                textTransform: "uppercase", color: "var(--rf-text4)",
              }}>
                Per-user feature overrides
              </h3>
              <p style={{ fontSize: 11, color: "var(--rf-text4)", marginBottom: 10 }}>
                Override = always-on or always-off for this user, regardless of the global flag.
                Click <b>Clear</b> to inherit the global default.
              </p>

              {!userFeatures && <div style={{ fontSize: 11, color: "var(--rf-text5)" }}>Loading…</div>}
              {userFeatures && (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {Object.entries(catalog).map(([key, meta]) => {
                    const override = userFeatures.overrides[key];
                    const hasOverride = override !== undefined;
                    const effective = !!userFeatures.effective[key];
                    return (
                      <div
                        key={key}
                        style={{
                          display: "flex", alignItems: "center", gap: 10,
                          padding: 10, borderRadius: 6, background: "var(--rf-surface2)",
                          border: "1px solid var(--rf-border)",
                        }}
                      >
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 12, fontWeight: 600 }}>{meta.label}</div>
                          <div style={{ fontSize: 10, color: "var(--rf-text5)" }}>
                            Effective: <b style={{ color: effective ? "#22c55e" : "#ef4444" }}>{effective ? "on" : "off"}</b>
                            {hasOverride
                              ? " · per-user override applied"
                              : " · inherits global"}
                          </div>
                        </div>
                        <div style={{ display: "flex", gap: 4 }}>
                          <button
                            onClick={() => patchUserOverride(key, true)}
                            disabled={busy}
                            style={overrideBtn(override === true)}
                          >
                            Force on
                          </button>
                          <button
                            onClick={() => patchUserOverride(key, false)}
                            disabled={busy}
                            style={overrideBtn(override === false)}
                          >
                            Force off
                          </button>
                          <button
                            onClick={() => patchUserOverride(key, null)}
                            disabled={busy || !hasOverride}
                            style={{
                              ...overrideBtn(false),
                              opacity: hasOverride ? 1 : 0.4,
                              borderStyle: "dashed",
                            }}
                          >
                            Clear
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </aside>
          )}
        </section>
      )}
    </div>
  );
}

// ─── Styling helpers ─────────────────────────────────────────────────────────

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div style={{
      padding: 12, borderRadius: 8,
      background: "var(--rf-surface1)", border: "1px solid var(--rf-border)",
    }}>
      <div style={{ fontSize: 10, color: "var(--rf-text5)", textTransform: "uppercase", letterSpacing: 0.7 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, marginTop: 2 }}>{value.toLocaleString()}</div>
    </div>
  );
}

const th: React.CSSProperties = {
  textAlign: "left", padding: "8px 12px",
  fontSize: 11, textTransform: "uppercase", letterSpacing: 0.6,
  color: "var(--rf-text4)", fontWeight: 600,
};
const td: React.CSSProperties = { padding: "8px 12px" };

const createInputStyle: React.CSSProperties = {
  padding: "7px 10px", borderRadius: 6,
  background: "var(--rf-surface2)", color: "var(--rf-text1)",
  border: "1px solid var(--rf-border)", fontSize: 12,
};

const iconBtn: React.CSSProperties = {
  padding: "3px 8px", borderRadius: 5, fontSize: 10, fontWeight: 600,
  background: "var(--rf-surface3)", color: "var(--rf-text3)",
  border: "1px solid var(--rf-border)", cursor: "pointer",
};

function pillStyle(on: boolean, disabled: boolean): React.CSSProperties {
  return {
    padding: "3px 9px", borderRadius: 999, fontSize: 11, fontWeight: 600,
    border: "1px solid " + (on ? "rgba(99,102,241,0.45)" : "var(--rf-border2)"),
    background: on ? "rgba(99,102,241,0.15)" : "var(--rf-surface2)",
    color: on ? "#a5b4fc" : "var(--rf-text4)",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.6 : 1,
  };
}

function togglePill(on: boolean): React.CSSProperties {
  return {
    padding: "6px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600,
    border: "1px solid " + (on ? "rgba(34,197,94,0.45)" : "var(--rf-border2)"),
    background: on ? "rgba(34,197,94,0.12)" : "var(--rf-surface2)",
    color: on ? "#22c55e" : "var(--rf-text3)",
    cursor: "pointer",
  };
}

function overrideBtn(active: boolean): React.CSSProperties {
  return {
    padding: "4px 8px", borderRadius: 5, fontSize: 11, fontWeight: 600,
    border: "1px solid " + (active ? "#818cf8" : "var(--rf-border2)"),
    background: active ? "rgba(99,102,241,0.15)" : "var(--rf-surface3)",
    color: active ? "#a5b4fc" : "var(--rf-text3)",
    cursor: "pointer",
  };
}
