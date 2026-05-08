"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth";
import type { User } from "@/types";
import { ZapIcon, EyeIcon, EyeOffIcon, Loader2Icon, ArrowRightIcon } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const { setToken, setUser } = useAuthStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [guestLoading, setGuestLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const data = await api.post<{ access_token: string }>("/auth/login", { email, password });
      setToken(data.access_token);
      const user = await api.get<User>("/auth/me");
      setUser(user);
      router.push(user.onboarding_complete ? "/feed" : "/settings/onboarding");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Invalid credentials");
    } finally {
      setLoading(false);
    }
  }

  async function handleGuestLogin() {
    setGuestLoading(true);
    setError("");
    try {
      const data = await api.post<{ access_token: string }>("/auth/login", {
        email: "test@researchflow.ai",
        password: "ResearchFlow2024!",
      });
      setToken(data.access_token);
      const user = await api.get<User>("/auth/me");
      await api.patch("/settings/profile", { display_name: "Guest Researcher" });
      setUser({ ...user, display_name: "Guest Researcher" });
      router.push(user.onboarding_complete ? "/feed" : "/settings/onboarding");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Guest login failed");
    } finally {
      setGuestLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-950 relative overflow-hidden">
      {/* Ambient background */}
      <div className="absolute top-1/4 left-1/3 w-[500px] h-[500px] bg-indigo-600/6 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/3 w-80 h-80 bg-violet-600/6 rounded-full blur-3xl pointer-events-none" />

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-[400px] px-4"
      >
        {/* Brand mark */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.1, duration: 0.4 }}
            className="inline-flex items-center justify-center w-13 h-13 w-[52px] h-[52px] rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 mb-4 shadow-lg shadow-indigo-500/20"
          >
            <ZapIcon size={22} className="text-white" />
          </motion.div>
          <h1 className="text-[22px] font-bold text-white tracking-tight">ResearchFlow</h1>
          <p className="text-sm text-gray-500 mt-1">Your personal research operating system</p>
        </div>

        {/* Card */}
        <div className="bg-gray-900/90 border border-gray-800/80 rounded-2xl p-7 shadow-2xl shadow-black/50">
          <h2 className="text-base font-semibold text-white mb-5">Sign in to continue</h2>

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

            <div className="space-y-1.5">
              <label className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
                autoFocus
                className="input-base"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Password</label>
              <div className="relative">
                <input
                  type={showPw ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  className="input-base pr-11"
                />
                <button
                  type="button"
                  onClick={() => setShowPw((s) => !s)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-600 hover:text-gray-400 transition-colors"
                  tabIndex={-1}
                >
                  {showPw ? <EyeOffIcon size={15} /> : <EyeIcon size={15} />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !email || !password}
              className="btn-primary w-full py-3 flex items-center justify-center gap-2 mt-1 text-sm"
            >
              {loading ? (
                <><Loader2Icon size={15} className="animate-spin" /> Signing in…</>
              ) : (
                <>Sign in <ArrowRightIcon size={14} /></>
              )}
            </button>
          </form>

          <div className="mt-5 flex items-center gap-3">
            <div className="flex-1 h-px bg-gray-800" />
            <span className="text-[11px] text-gray-600 uppercase tracking-wider">or</span>
            <div className="flex-1 h-px bg-gray-800" />
          </div>

          <button
            type="button"
            onClick={handleGuestLogin}
            disabled={guestLoading || loading}
            className="mt-4 w-full py-2.5 rounded-xl border border-gray-700/60 bg-gray-800/40 hover:bg-gray-800/70 text-sm text-gray-300 hover:text-white transition-all flex items-center justify-center gap-2"
          >
            {guestLoading ? (
              <><Loader2Icon size={14} className="animate-spin" /> Signing in as guest…</>
            ) : (
              "Continue as Guest"
            )}
          </button>
        </div>

        <p className="text-center text-sm text-gray-600 mt-5">
          No account?{" "}
          <Link href="/signup" className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors">
            Create one
          </Link>
        </p>
      </motion.div>
    </div>
  );
}
