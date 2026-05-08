"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth";
import type { User } from "@/types";
import { ZapIcon, EyeIcon, EyeOffIcon, Loader2Icon, ArrowRightIcon } from "lucide-react";

const FIELDS = [
  { key: "display_name", label: "Your name", type: "text", placeholder: "Ada Lovelace" },
  { key: "email",        label: "Email",      type: "email", placeholder: "you@example.com" },
  { key: "password",     label: "Password",   type: "password", placeholder: "8+ characters" },
] as const;

export default function SignupPage() {
  const router = useRouter();
  const { setToken, setUser } = useAuthStore();
  const [form, setForm] = useState({ email: "", password: "", display_name: "" });
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (form.password.length < 8) { setError("Password must be at least 8 characters"); return; }
    setLoading(true);
    setError("");
    try {
      const data = await api.post<{ access_token: string }>("/auth/register", form);
      setToken(data.access_token);
      const user = await api.get<User>("/auth/me");
      setUser(user);
      router.push("/settings/onboarding");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 relative overflow-hidden">
      <div className="absolute top-1/3 right-1/4 w-[500px] h-[500px] bg-violet-600/6 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/3 left-1/4 w-72 h-72 bg-indigo-600/6 rounded-full blur-3xl pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-[400px] px-4"
      >
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-[52px] h-[52px] rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 mb-4 shadow-lg shadow-indigo-500/20">
            <ZapIcon size={22} className="text-white" />
          </div>
          <h1 className="text-[22px] font-bold text-white tracking-tight">ResearchFlow</h1>
          <p className="text-sm text-gray-500 mt-1">Start your research journey</p>
        </div>

        <div className="bg-gray-900/90 border border-gray-800/80 rounded-2xl p-7 shadow-2xl shadow-black/50">
          <h2 className="text-base font-semibold text-white mb-5">Create your account</h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-red-950/60 border border-red-800/60 rounded-xl px-4 py-3 text-red-300 text-sm"
              >
                {error}
              </motion.div>
            )}

            {FIELDS.map(({ key, label, type, placeholder }, idx) => (
              <div key={key} className="space-y-1.5">
                <label className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">
                  {label}
                </label>
                <div className="relative">
                  <input
                    type={key === "password" && showPw ? "text" : type}
                    value={form[key]}
                    onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
                    placeholder={placeholder}
                    required
                    autoFocus={idx === 0}
                    className={`input-base ${key === "password" ? "pr-11" : ""}`}
                  />
                  {key === "password" && (
                    <button
                      type="button"
                      onClick={() => setShowPw((s) => !s)}
                      className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-600 hover:text-gray-400 transition-colors"
                      tabIndex={-1}
                    >
                      {showPw ? <EyeOffIcon size={15} /> : <EyeIcon size={15} />}
                    </button>
                  )}
                </div>
              </div>
            ))}

            <button
              type="submit"
              disabled={loading || !form.email || !form.password || !form.display_name}
              className="btn-primary w-full py-3 flex items-center justify-center gap-2 mt-1 text-sm"
            >
              {loading ? (
                <><Loader2Icon size={15} className="animate-spin" /> Creating account…</>
              ) : (
                <>Get started <ArrowRightIcon size={14} /></>
              )}
            </button>
          </form>
        </div>

        <p className="text-center text-sm text-gray-600 mt-5">
          Already have an account?{" "}
          <Link href="/login" className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors">
            Sign in
          </Link>
        </p>
      </motion.div>
    </div>
  );
}
