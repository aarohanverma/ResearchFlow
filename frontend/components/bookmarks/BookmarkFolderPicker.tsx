"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/lib/api";
import type { BookmarkFolder } from "@/types";
import {
  BookmarkIcon, CheckIcon, FolderPlusIcon,
  Loader2Icon, Trash2Icon, XIcon, GripHorizontalIcon,
} from "lucide-react";

const FOLDER_COLORS = [
  "#6366f1", "#8b5cf6", "#ec4899", "#f59e0b",
  "#10b981", "#3b82f6", "#ef4444", "#14b8a6",
];

const PANEL_WIDTH  = 256;
const PANEL_HEIGHT = 380; // conservative max height estimate

interface Props {
  paperId: string;
  isBookmarked: boolean;
  currentFolderIds: string[];
  anchorRef: React.RefObject<HTMLElement | null>;
  onClose: () => void;
  onSaved: (folderIds: string[]) => void;
  onRemoved: () => void;
}

export function BookmarkFolderPicker({
  paperId, isBookmarked, currentFolderIds, anchorRef, onClose, onSaved, onRemoved,
}: Props) {
  const [folders, setFolders] = useState<BookmarkFolder[]>([]);
  const [loadingFolders, setLoadingFolders] = useState(true);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set(currentFolderIds));
  const [saving, setSaving] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState(FOLDER_COLORS[0]);
  const [creatingBusy, setCreatingBusy] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);

  // Viewport-aware fixed position relative to anchor button
  const [pos, setPos] = useState({ top: 0, left: 0 });

  useLayoutEffect(() => {
    if (!anchorRef.current) return;
    const rect = anchorRef.current.getBoundingClientRect();
    const vh = window.innerHeight;
    const vw = window.innerWidth;

    // Prefer below; flip above if not enough room
    const spaceBelow = vh - rect.bottom;
    const topBelow   = rect.bottom + 6;
    const topAbove   = rect.top - PANEL_HEIGHT - 6;

    let top = spaceBelow >= PANEL_HEIGHT || spaceBelow > topAbove
      ? topBelow
      : Math.max(8, topAbove);

    // Align right edge with button's right edge; clamp to viewport
    let left = rect.right - PANEL_WIDTH;
    left = Math.max(8, Math.min(left, vw - PANEL_WIDTH - 8));

    setPos({ top: Math.max(8, top), left });
  }, [anchorRef]);

  useEffect(() => {
    api.get<BookmarkFolder[]>("/bookmarks/folders")
      .then((data) => { setFolders(data); setLoadingFolders(false); })
      .catch(() => setLoadingFolders(false));
  }, []);

  // Close on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (
        panelRef.current && !panelRef.current.contains(e.target as Node) &&
        anchorRef.current && !anchorRef.current.contains(e.target as Node)
      ) {
        onClose();
      }
    }
    document.addEventListener("mousedown", handler, true);
    return () => document.removeEventListener("mousedown", handler, true);
  }, [anchorRef, onClose]);

  // Close on Escape
  useEffect(() => {
    function handler(e: KeyboardEvent) { if (e.key === "Escape") onClose(); }
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  function toggle(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  async function createFolder() {
    if (!newName.trim() || creatingBusy) return;
    setCreatingBusy(true);
    try {
      const f = await api.post<BookmarkFolder>("/bookmarks/folders", {
        name: newName.trim(), color: newColor,
      });
      setFolders((prev) => [...prev, f]);
      setSelectedIds((prev) => new Set([...prev, f.id]));
      setNewName("");
      setCreating(false);
    } catch {}
    setCreatingBusy(false);
  }

  async function save() {
    setSaving(true);
    try {
      const ids = [...selectedIds];
      if (!isBookmarked) {
        await api.post("/bookmarks", { paper_id: paperId, folder_ids: ids });
      } else {
        await api.put(`/bookmarks/${paperId}/folders`, { folder_ids: ids });
      }
      onSaved(ids);
    } catch {}
    setSaving(false);
    onClose();
  }

  async function removeBookmark() {
    try { await api.delete(`/bookmarks/${paperId}`); } catch {}
    onRemoved();
    onClose();
  }

  const content = (
    <motion.div
      ref={panelRef}
      drag
      dragMomentum={false}
      dragElastic={0}
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.15 }}
      style={{
        position: "fixed",
        top: pos.top,
        left: pos.left,
        zIndex: 9999,
        width: PANEL_WIDTH,
      }}
      className="bg-gray-900 border border-white/10 rounded-2xl shadow-2xl shadow-black/70 overflow-hidden select-none"
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header — doubles as drag handle */}
      <div
        className="flex items-center justify-between px-4 py-3 border-b border-white/5"
        style={{ cursor: "grab" }}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2">
          <BookmarkIcon size={13} className="text-amber-400" />
          <span className="text-xs font-semibold text-white">
            {isBookmarked ? "Manage Folders" : "Save to Folders"}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <GripHorizontalIcon size={13} className="text-gray-600" />
          <button
            onClick={onClose}
            onPointerDown={(e) => e.stopPropagation()}
            className="text-gray-600 hover:text-gray-400 p-0.5 transition-colors"
          >
            <XIcon size={13} />
          </button>
        </div>
      </div>

      {/* Folder list */}
      <div className="max-h-52 overflow-y-auto" style={{ cursor: "default" }}>
        {loadingFolders ? (
          <div className="flex items-center justify-center py-6">
            <Loader2Icon size={14} className="animate-spin text-gray-600" />
          </div>
        ) : folders.length === 0 && !creating ? (
          <p className="text-xs text-gray-600 text-center py-5 px-4">
            No folders yet — create one below.
          </p>
        ) : (
          <div className="py-1">
            {folders.map((f) => {
              const checked = selectedIds.has(f.id);
              return (
                <button
                  key={f.id}
                  onClick={() => toggle(f.id)}
                  onPointerDown={(e) => e.stopPropagation()}
                  className="w-full flex items-center gap-3 px-4 py-2 hover:bg-white/5 transition-colors"
                >
                  <div className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: f.color || "#6366f1" }} />
                  <span className="flex-1 text-xs text-left text-gray-300 truncate">{f.name}</span>
                  <span className="text-[10px] text-gray-700 mr-1">{f.bookmark_count}</span>
                  <div className={`w-4 h-4 rounded-md border flex items-center justify-center flex-shrink-0 transition-all ${checked ? "bg-amber-500 border-amber-500" : "border-gray-700 bg-transparent"}`}>
                    {checked && <CheckIcon size={10} className="text-white" strokeWidth={3} />}
                  </div>
                </button>
              );
            })}
          </div>
        )}

        {/* Inline folder creation */}
        <AnimatePresence>
          {creating && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="px-3 py-2.5 border-t border-white/5"
            >
              <input
                autoFocus
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") createFolder();
                  if (e.key === "Escape") setCreating(false);
                }}
                onPointerDown={(e) => e.stopPropagation()}
                placeholder="Folder name…"
                className="w-full bg-gray-800 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-white placeholder-gray-600 outline-none focus:border-amber-500/50 mb-2"
              />
              <div className="flex gap-1 mb-2 flex-wrap">
                {FOLDER_COLORS.map((c) => (
                  <button
                    key={c}
                    onClick={() => setNewColor(c)}
                    onPointerDown={(e) => e.stopPropagation()}
                    className={`w-4 h-4 rounded-sm transition-all flex-shrink-0 ${newColor === c ? "ring-2 ring-white ring-offset-1 ring-offset-gray-900" : ""}`}
                    style={{ backgroundColor: c }}
                  />
                ))}
              </div>
              <div className="flex gap-1.5">
                <button
                  onClick={createFolder}
                  onPointerDown={(e) => e.stopPropagation()}
                  disabled={!newName.trim() || creatingBusy}
                  className="flex-1 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white text-[11px] font-semibold rounded-lg py-1.5 transition-colors"
                >
                  {creatingBusy ? <Loader2Icon size={10} className="animate-spin mx-auto" /> : "Create"}
                </button>
                <button
                  onClick={() => setCreating(false)}
                  onPointerDown={(e) => e.stopPropagation()}
                  className="flex-1 bg-gray-800 hover:bg-gray-700 text-gray-400 text-[11px] rounded-lg py-1.5 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <div className="px-3 py-2.5 border-t border-white/5 space-y-1.5" style={{ cursor: "default" }}>
        {!creating && (
          <button
            onClick={() => setCreating(true)}
            onPointerDown={(e) => e.stopPropagation()}
            className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-gray-500 hover:text-gray-300 hover:bg-white/5 transition-all"
          >
            <FolderPlusIcon size={12} />
            New folder
          </button>
        )}
        <div className="flex gap-1.5">
          {isBookmarked && (
            <button
              onClick={removeBookmark}
              onPointerDown={(e) => e.stopPropagation()}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] text-red-400/80 hover:text-red-400 hover:bg-red-950/20 transition-all"
            >
              <Trash2Icon size={11} />
              Remove
            </button>
          )}
          <button
            onClick={save}
            onPointerDown={(e) => e.stopPropagation()}
            disabled={saving}
            className="flex-1 flex items-center justify-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-semibold bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white transition-all"
          >
            {saving
              ? <Loader2Icon size={11} className="animate-spin" />
              : <BookmarkIcon size={11} />
            }
            {isBookmarked ? "Update" : "Save"}
          </button>
        </div>
      </div>
    </motion.div>
  );

  if (typeof document === "undefined") return null;
  return createPortal(content, document.body);
}
