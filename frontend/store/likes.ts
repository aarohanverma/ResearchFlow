/**
 * Likes store — tracks which papers the current user has liked.
 * Loads from GET /feed/liked-ids once per session; updates optimistically on like.
 */

import { create } from "zustand";
import { api } from "@/lib/api";

interface LikesState {
  ids: Set<string>;
  loaded: boolean;
  loading: boolean;
  initialize: () => Promise<void>;
  isLiked: (paperId: string) => boolean;
  add: (paperId: string) => void;
  remove: (paperId: string) => void;
  reset: () => void;
}

export const useLikesStore = create<LikesState>((set, get) => ({
  ids: new Set<string>(),
  loaded: false,
  loading: false,

  async initialize() {
    if (get().loaded || get().loading) return;
    set({ loading: true });
    try {
      const data = await api.get<string[]>("/feed/liked-ids");
      set({ ids: new Set(data), loaded: true, loading: false });
    } catch {
      set({ loaded: true, loading: false });
    }
  },

  isLiked(paperId: string) {
    return get().ids.has(paperId);
  },

  add(paperId: string) {
    set((s) => ({ ids: new Set([...s.ids, paperId]) }));
  },

  remove(paperId: string) {
    set((s) => {
      const next = new Set(s.ids);
      next.delete(paperId);
      return { ids: next };
    });
  },

  reset() {
    set({ ids: new Set(), loaded: false, loading: false });
  },
}));
