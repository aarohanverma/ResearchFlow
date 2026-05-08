/**
 * Bookmarks store — tracks which papers are bookmarked by the current user.
 * Loads once from /bookmarks on first use; all PaperCards share this singleton
 * so the saved state is consistent across the feed without per-card API calls.
 */

import { create } from "zustand";
import { api } from "@/lib/api";

interface BookmarkEntry {
  id: string;
  paper_id: string;
}

interface BookmarksState {
  ids: Set<string>;
  loaded: boolean;
  loading: boolean;
  initialize: () => Promise<void>;
  isBookmarked: (paperId: string) => boolean;
  add: (paperId: string) => void;
  remove: (paperId: string) => void;
  reset: () => void;
}

export const useBookmarksStore = create<BookmarksState>((set, get) => ({
  ids: new Set<string>(),
  loaded: false,
  loading: false,

  async initialize() {
    if (get().loaded || get().loading) return;
    set({ loading: true });
    try {
      const data = await api.get<BookmarkEntry[]>("/bookmarks");
      set({
        ids: new Set(data.map((b) => b.paper_id)),
        loaded: true,
        loading: false,
      });
    } catch {
      set({ loaded: true, loading: false });
    }
  },

  isBookmarked(paperId: string) {
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
