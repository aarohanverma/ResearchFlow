/**
 * Auth store — Zustand. Persists token in localStorage.
 * Used by all API calls and route guards.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "@/types";
import { useBookmarksStore } from "./bookmarks";
import { useLikesStore } from "./likes";

interface AuthState {
  token: string | null;
  user: User | null;
  setToken: (token: string) => void;
  setUser: (user: User) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      user: null,
      setToken: (token) => set({ token }),
      setUser: (user) => set({ user }),
      logout: () => {
        set({ token: null, user: null });
        useBookmarksStore.getState().reset();
        useLikesStore.getState().reset();
        if (typeof window !== "undefined") {
          window.location.href = "/login";
        }
      },
    }),
    { name: "rf_auth" }
  )
);
