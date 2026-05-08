"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/lib/api";
import type { Bookmark, BookmarkFolder, Paper } from "@/types";
import { PaperPanel } from "@/components/paper/PaperPanel";
import {
  BookmarkIcon,
  Trash2Icon,
  ExternalLinkIcon,
  MessageSquareIcon,
  XIcon,
  SendIcon,
  BotIcon,
  UserIcon,
  Loader2Icon,
  LibraryIcon,
  SearchIcon,
  ClockIcon,
  FolderIcon,
  FolderOpenIcon,
  FolderPlusIcon,
  PencilIcon,
  CheckIcon,
  RefreshCwIcon,
  ChevronRightIcon,
} from "lucide-react";
import { cleanAbstract } from "@/lib/utils";
import { useAuthStore } from "@/store/auth";
import { useNamespaceStore } from "@/store/namespace";
import { useBookmarksStore } from "@/store/bookmarks";
import MarkdownRenderer from "@/components/ui/MarkdownRenderer";

interface IndexingStatus { total: number; indexed: number; ready: boolean; }
interface ChatMsg { role: "user" | "assistant"; content: string; streaming?: boolean; }

// ── Folder colors palette ──────────────────────────────────────────────────────
const FOLDER_COLORS = [
  "#6366f1", "#8b5cf6", "#ec4899", "#f59e0b",
  "#10b981", "#3b82f6", "#ef4444", "#14b8a6",
];

// ── KB Chat (folder or namespace scoped) ──────────────────────────────────────
function BookmarksKBChat({
  count,
  namespaceKeys,
  folderId,
  folderName,
  indexingReady,
  onClose,
}: {
  count: number;
  namespaceKeys: string[];
  folderId: string | null;
  folderName: string | null;
  indexingReady: boolean;
  onClose: () => void;
}) {
  const { token } = useAuthStore();
  const scopeLabel = folderName ?? (namespaceKeys.length ? namespaceKeys.join(", ") : "all namespaces");
  const [messages, setMessages] = useState<ChatMsg[]>([{
    role: "assistant",
    content: `Library loaded: ${count} paper${count !== 1 ? "s" : ""}${folderName ? ` from "${folderName}"` : ""}. Every answer cites the source paper. What would you like to explore?`,
  }]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || busy || !indexingReady) return;
    setInput("");
    setBusy(true);
    const history = messages.filter((m) => !m.streaming).slice(-8)
      .map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", streaming: true },
    ]);
    try {
      const resp = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/bookmarks/chat`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify({
            message: text,
            expertise_level: "practitioner",
            history,
            namespace_keys: namespaceKeys,
            folder_id: folderId,
          }),
        }
      );
      if (!resp.body) throw new Error("no body");
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        for (const line of decoder.decode(value, { stream: true }).split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            const p = JSON.parse(line.slice(6));
            if (p.chunk) {
              acc += p.chunk;
              setMessages((prev) => [...prev.slice(0, -1), { role: "assistant", content: acc, streaming: true }]);
            }
            if (p.done) setMessages((prev) => [...prev.slice(0, -1), { role: "assistant", content: acc }]);
          } catch {}
        }
      }
    } catch {
      setMessages((prev) => [...prev.slice(0, -1), { role: "assistant", content: "Something went wrong. Try again." }]);
    }
    setBusy(false);
    inputRef.current?.focus();
  }

  return (
    <motion.div
      initial={{ x: "100%", opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0 }}
      transition={{ type: "spring", damping: 30, stiffness: 340 }}
      className="w-[420px] shrink-0 border-l border-gray-800/70 bg-gray-950 flex flex-col"
    >
      <div className="flex items-center justify-between px-4 py-3.5 border-b border-gray-800/60 bg-gray-950/95 backdrop-blur-sm">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-8 h-8 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center flex-shrink-0">
            <LibraryIcon size={14} className="text-indigo-400" />
          </div>
          <div className="min-w-0">
            <p className="text-xs font-semibold text-white">Library RAG</p>
            <p className="text-[10px] text-gray-500 truncate">{count} papers · {scopeLabel}</p>
          </div>
        </div>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-300 p-1.5 rounded-lg hover:bg-gray-800 transition-colors">
          <XIcon size={14} />
        </button>
      </div>

      {!indexingReady && (
        <div className="px-4 py-2.5 bg-amber-950/30 border-b border-amber-900/30 flex items-center gap-2">
          <ClockIcon size={12} className="text-amber-400 animate-pulse flex-shrink-0" />
          <p className="text-[11px] text-amber-300/80">Papers are still being indexed. Chat will unlock when ready.</p>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
            <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${msg.role === "user" ? "bg-indigo-600" : "bg-gray-800 border border-gray-700/50"}`}>
              {msg.role === "user" ? <UserIcon size={11} className="text-white" /> : <BotIcon size={11} className="text-indigo-400" />}
            </div>
            <div className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${msg.role === "user" ? "bg-indigo-600 text-white rounded-tr-sm" : "bg-gray-900 border border-gray-800/60 text-gray-200 rounded-tl-sm"}`}>
              {msg.role === "user" ? <span className="whitespace-pre-wrap">{msg.content}</span> : <MarkdownRenderer content={msg.content} />}
              {msg.streaming && msg.content === "" && <span className="inline-flex items-center gap-1 text-gray-500 text-xs"><Loader2Icon size={10} className="animate-spin" />Thinking…</span>}
              {msg.streaming && msg.content !== "" && <span className="inline-block w-1.5 h-4 bg-indigo-400 rounded-sm animate-pulse ml-0.5 align-middle" />}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="p-3 border-t border-gray-800/60">
        {!indexingReady ? (
          <div className="flex items-center gap-2 bg-gray-900/60 border border-gray-800 rounded-xl px-4 py-3 opacity-60 cursor-not-allowed select-none">
            <ClockIcon size={13} className="text-amber-400/60 animate-pulse" />
            <span className="text-sm text-gray-500">Waiting for indexing…</span>
          </div>
        ) : (
          <div className="flex gap-2 items-center bg-gray-900 border border-gray-800 rounded-xl px-3 py-2.5 focus-within:border-indigo-500/50 transition-colors">
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder={`Ask about ${scopeLabel}…`}
              className="flex-1 bg-transparent text-sm text-gray-200 placeholder-gray-600 outline-none"
              disabled={busy}
              autoFocus
            />
            <button onClick={send} disabled={!input.trim() || busy} className="flex-shrink-0 w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center disabled:opacity-40 hover:bg-indigo-500 transition-colors">
              {busy ? <Loader2Icon size={12} className="animate-spin text-white" /> : <SendIcon size={12} className="text-white" />}
            </button>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ── Folder sidebar ────────────────────────────────────────────────────────────
function FolderSidebar({
  folders,
  selectedFolderId,
  onSelect,
  onCreateFolder,
  onRenameFolder,
  onDeleteFolder,
  allCount,
}: {
  folders: BookmarkFolder[];
  selectedFolderId: string | null;
  onSelect: (id: string | null) => void;
  onCreateFolder: (name: string, color: string) => Promise<void>;
  onRenameFolder: (id: string, name: string) => Promise<void>;
  onDeleteFolder: (id: string) => Promise<void>;
  allCount: number;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState(FOLDER_COLORS[0]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");

  async function submitCreate() {
    if (!newName.trim()) return;
    await onCreateFolder(newName.trim(), newColor);
    setNewName("");
    setCreating(false);
  }

  async function submitRename(id: string) {
    if (!editName.trim()) return;
    await onRenameFolder(id, editName.trim());
    setEditingId(null);
  }

  if (collapsed) {
    return (
      <div className="w-10 shrink-0 border-r border-white/5 flex flex-col items-center py-3 gap-3 bg-gray-950/50 transition-all">
        <button
          onClick={() => setCollapsed(false)}
          title="Expand folders"
          className="text-gray-600 hover:text-gray-300 transition-colors p-1"
        >
          <ChevronRightIcon size={14} />
        </button>
        {/* Folder color dots */}
        <button
          onClick={() => { setCollapsed(false); onSelect(null); }}
          title="All Bookmarks"
          className={`w-5 h-5 rounded-md flex items-center justify-center transition-colors ${selectedFolderId === null ? "bg-amber-500/30" : "hover:bg-white/5"}`}
        >
          <BookmarkIcon size={10} className={selectedFolderId === null ? "text-amber-400" : "text-gray-600"} />
        </button>
        {folders.map((f) => (
          <button
            key={f.id}
            onClick={() => { setCollapsed(false); onSelect(f.id); }}
            title={f.name}
            className="w-4 h-4 rounded-sm flex-shrink-0 transition-opacity hover:opacity-80"
            style={{ backgroundColor: f.color || "#6366f1", opacity: selectedFolderId === f.id ? 1 : 0.5 }}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="w-52 shrink-0 border-r border-white/5 flex flex-col bg-gray-950/50 transition-all">
      <div className="p-3 border-b border-white/5">
        <div className="flex items-center justify-between mb-2">
          <p className="text-[10px] font-bold text-gray-600 uppercase tracking-wider">Folders</p>
          <button
            onClick={() => setCollapsed(true)}
            title="Collapse folders"
            className="text-gray-700 hover:text-gray-400 transition-colors p-0.5"
          >
            <ChevronRightIcon size={12} className="rotate-180" />
          </button>
        </div>

        {/* All Bookmarks */}
        <button
          onClick={() => onSelect(null)}
          className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs transition-all ${
            selectedFolderId === null
              ? "bg-amber-950/40 text-amber-300 border border-amber-700/30"
              : "text-gray-400 hover:text-gray-200 hover:bg-white/5"
          }`}
        >
          <BookmarkIcon size={12} className={selectedFolderId === null ? "text-amber-400" : "text-gray-600"} />
          <span className="flex-1 text-left font-medium">All Bookmarks</span>
          <span className="text-[10px] text-gray-600">{allCount}</span>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {folders.map((folder) => {
          const isSelected = selectedFolderId === folder.id;
          const isEditing = editingId === folder.id;
          return (
            <div key={folder.id} className={`group flex items-center gap-1.5 px-2 py-1.5 rounded-lg transition-all cursor-pointer ${isSelected ? "bg-gray-800/70 text-white" : "text-gray-500 hover:text-gray-300 hover:bg-white/5"}`}
              onClick={() => { if (!isEditing) onSelect(folder.id); }}>
              <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: folder.color || "#6366f1" }} />
              {isEditing ? (
                <input
                  autoFocus
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") submitRename(folder.id); if (e.key === "Escape") setEditingId(null); }}
                  onBlur={() => submitRename(folder.id)}
                  className="flex-1 bg-transparent text-xs text-white outline-none"
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <span className="flex-1 text-xs truncate">{folder.name}</span>
              )}
              <span className="text-[10px] text-gray-700">{folder.bookmark_count}</span>
              <div className="hidden group-hover:flex items-center gap-0.5 ml-0.5">
                <button onClick={(e) => { e.stopPropagation(); setEditingId(folder.id); setEditName(folder.name); }}
                  className="p-0.5 text-gray-600 hover:text-gray-300 transition-colors">
                  <PencilIcon size={10} />
                </button>
                <button onClick={(e) => { e.stopPropagation(); onDeleteFolder(folder.id); }}
                  className="p-0.5 text-gray-600 hover:text-red-400 transition-colors">
                  <XIcon size={10} />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      <div className="p-2 border-t border-white/5">
        {creating ? (
          <div className="space-y-2">
            <input
              autoFocus
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") submitCreate(); if (e.key === "Escape") setCreating(false); }}
              placeholder="Folder name…"
              className="w-full bg-gray-900 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-white placeholder-gray-600 outline-none focus:border-indigo-500/50"
            />
            <div className="flex gap-1 flex-wrap">
              {FOLDER_COLORS.map((c) => (
                <button key={c} onClick={() => setNewColor(c)}
                  className={`w-4 h-4 rounded-sm transition-all ${newColor === c ? "ring-2 ring-white ring-offset-1 ring-offset-gray-950" : ""}`}
                  style={{ backgroundColor: c }} />
              ))}
            </div>
            <div className="flex gap-1.5">
              <button onClick={submitCreate} className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white text-[11px] font-semibold rounded-lg py-1 transition-colors">Create</button>
              <button onClick={() => setCreating(false)} className="flex-1 bg-gray-800 hover:bg-gray-700 text-gray-400 text-[11px] rounded-lg py-1 transition-colors">Cancel</button>
            </div>
          </div>
        ) : (
          <button onClick={() => setCreating(true)}
            className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-gray-600 hover:text-gray-300 hover:bg-white/5 transition-all">
            <FolderPlusIcon size={12} />
            New Folder
          </button>
        )}
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function BookmarksPage() {
  const { selectedTopics } = useNamespaceStore();
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [folders, setFolders] = useState<BookmarkFolder[]>([]);
  const [selectedFolderId, setSelectedFolderId] = useState<string | null>(null);
  const [selected, setSelected] = useState<Paper | null>(null);
  const [showChat, setShowChat] = useState(false);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [indexing, setIndexing] = useState<IndexingStatus>({ total: 0, indexed: 0, ready: true });
  const [reindexing, setReindexing] = useState(false);
  const [movingBookmarkId, setMovingBookmarkId] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchIndexingStatus = useCallback(async (namespaces?: string[]) => {
    try {
      const qs = new URLSearchParams();
      const ns = namespaces ?? selectedTopics;
      if (ns.length) qs.set("namespace_keys", ns.join(","));
      const status = await api.get<IndexingStatus>(`/bookmarks/indexing-status?${qs}`);
      setIndexing(status);
      if (!status.ready && status.total > 0)
        pollRef.current = setTimeout(() => fetchIndexingStatus(ns), 6000);
    } catch {}
  }, [selectedTopics]);

  const loadFolders = useCallback(async () => {
    try { setFolders(await api.get<BookmarkFolder[]>("/bookmarks/folders")); } catch {}
  }, []);

  const loadBookmarks = useCallback(async () => {
    setLoading(true);
    try {
      const qs = new URLSearchParams();
      if (selectedTopics.length) qs.set("namespace_keys", selectedTopics.join(","));
      const data = await api.get<Bookmark[]>(`/bookmarks?${qs}`);
      setBookmarks(Array.isArray(data) ? data : []);
    } catch {}
    setLoading(false);
  }, [selectedTopics]);

  useEffect(() => {
    loadBookmarks();
    loadFolders();
    fetchIndexingStatus();
    return () => { if (pollRef.current) clearTimeout(pollRef.current); };
  }, [loadBookmarks, loadFolders, fetchIndexingStatus]);

  // Derived: bookmarks for selected folder (or all)
  const visibleBookmarks = selectedFolderId
    ? bookmarks.filter((b) => b.folder_ids.includes(selectedFolderId!))
    : bookmarks;

  const filteredBookmarks = searchQuery.trim()
    ? visibleBookmarks.filter((bm) => {
        if (!bm.paper) return false;
        const q = searchQuery.toLowerCase();
        return bm.paper.title.toLowerCase().includes(q) ||
          bm.paper.authors.some((a) => a.toLowerCase().includes(q)) ||
          bm.paper.key_concepts.some((c) => c.toLowerCase().includes(q));
      })
    : visibleBookmarks;

  const selectedFolder = folders.find((f) => f.id === selectedFolderId) ?? null;

  async function createFolder(name: string, color: string) {
    try {
      const f = await api.post<BookmarkFolder>("/bookmarks/folders", { name, color });
      setFolders((fs) => [...fs, f]);
    } catch {}
  }

  async function renameFolder(id: string, name: string) {
    try {
      await api.patch(`/bookmarks/folders/${id}`, { name });
      setFolders((fs) => fs.map((f) => f.id === id ? { ...f, name } : f));
    } catch {}
  }

  async function deleteFolder(id: string) {
    try {
      await api.delete(`/bookmarks/folders/${id}`);
      setFolders((fs) => fs.filter((f) => f.id !== id));
      if (selectedFolderId === id) setSelectedFolderId(null);
      // Remove this folder from all bookmarks' folder_ids lists
      setBookmarks((bs) =>
        bs.map((b) => ({ ...b, folder_ids: b.folder_ids.filter((f) => f !== id) }))
      );
    } catch {}
  }

  async function setBookmarkFolders(paperId: string, folderIds: string[]) {
    try {
      await api.put(`/bookmarks/${paperId}/folders`, { folder_ids: folderIds });
      setBookmarks((bs) =>
        bs.map((b) => b.paper_id === paperId ? { ...b, folder_ids: folderIds } : b)
      );
      loadFolders();
      setMovingBookmarkId(null);
    } catch {}
  }

  async function remove(paperId: string) {
    await api.delete(`/bookmarks/${paperId}`);
    setBookmarks((bs) => bs.filter((b) => b.paper_id !== paperId));
    if (selected?.id === paperId) setSelected(null);
    useBookmarksStore.getState().remove(paperId);
    loadFolders();
    fetchIndexingStatus();
  }

  async function handleReindex() {
    setReindexing(true);
    try {
      await api.post("/bookmarks/reindex", {});
      setTimeout(fetchIndexingStatus, 3000);
    } catch {}
    setReindexing(false);
  }

  const indexingPct = indexing.total > 0 ? Math.round((indexing.indexed / indexing.total) * 100) : 100;

  return (
    <div className="flex h-full overflow-hidden">
      {/* Folder sidebar */}
      <FolderSidebar
        folders={folders}
        selectedFolderId={selectedFolderId}
        onSelect={(id) => { setSelectedFolderId(id); setShowChat(false); setSelected(null); }}
        onCreateFolder={createFolder}
        onRenameFolder={renameFolder}
        onDeleteFolder={deleteFolder}
        allCount={bookmarks.length}
      />

      {/* Main list */}
      <div className="flex-1 overflow-y-auto px-6 py-6 min-w-0">
        {/* Header */}
        <div className="flex items-center gap-3 mb-4 flex-wrap">
          {selectedFolder ? (
            <>
              <div className="w-5 h-5 rounded-md" style={{ backgroundColor: selectedFolder.color || "#6366f1" }} />
              <h1 className="text-xl font-bold text-white">{selectedFolder.name}</h1>
              <span className="text-sm text-gray-500">({visibleBookmarks.length})</span>
            </>
          ) : (
            <>
              <BookmarkIcon size={20} className="text-amber-400" />
              <h1 className="text-xl font-bold text-white">All Bookmarks</h1>
              <span className="text-sm text-gray-500">({bookmarks.length})</span>
            </>
          )}

          <div className="ml-auto flex items-center gap-2">
            {/* Indexing status */}
            {!indexing.ready && indexing.total > 0 && (
              <div className="flex items-center gap-2 bg-amber-950/30 border border-amber-900/40 rounded-xl px-3 py-1.5">
                <ClockIcon size={12} className="text-amber-400 animate-pulse" />
                <span className="text-xs text-amber-300/80">Indexing {indexing.indexed}/{indexing.total}</span>
                <div className="w-16 h-1 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-amber-500 rounded-full transition-all duration-500" style={{ width: `${indexingPct}%` }} />
                </div>
                <button onClick={handleReindex} disabled={reindexing} className="text-amber-400 hover:text-amber-300 transition-colors" title="Re-trigger indexing">
                  <RefreshCwIcon size={11} className={reindexing ? "animate-spin" : ""} />
                </button>
              </div>
            )}

            {visibleBookmarks.filter((b) => b.paper).length > 0 && (
              <button
                onClick={() => { setSelected(null); setShowChat((v) => !v); }}
                disabled={!indexing.ready}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-sm font-medium transition-all border ${
                  !indexing.ready
                    ? "opacity-40 cursor-not-allowed text-gray-500 border-gray-800"
                    : showChat
                      ? "bg-indigo-600/20 text-indigo-300 border-indigo-600/40"
                      : "text-gray-400 border-gray-700/60 hover:border-indigo-500/50 hover:text-indigo-300 hover:bg-indigo-950/20"
                }`}
              >
                <MessageSquareIcon size={14} />
                {showChat ? "Close Chat" : selectedFolder ? `Chat with "${selectedFolder.name}"` : "Chat with Library"}
              </button>
            )}
          </div>
        </div>

        {/* Search */}
        {!loading && visibleBookmarks.length > 0 && (
          <div className="flex items-center gap-3 mb-5">
            <div className="flex-1 flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-xl px-3.5 py-2.5 focus-within:border-indigo-500/50 transition-colors">
              <SearchIcon size={14} className="text-gray-600 flex-shrink-0" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter by title, author, or concept…"
                className="flex-1 bg-transparent text-sm text-gray-200 placeholder-gray-600 outline-none"
              />
              {searchQuery && (
                <button onClick={() => setSearchQuery("")} className="text-gray-600 hover:text-gray-400 transition-colors">
                  <XIcon size={13} />
                </button>
              )}
            </div>
            {searchQuery && <span className="text-xs text-gray-500 whitespace-nowrap">{filteredBookmarks.length} of {visibleBookmarks.length}</span>}
          </div>
        )}

        {/* Content */}
        {loading ? (
          <div className="text-gray-500 py-16 text-center">Loading…</div>
        ) : bookmarks.length === 0 ? (
          <div className="text-center py-24 text-gray-500">
            <BookmarkIcon size={32} className="mx-auto mb-4 opacity-30" />
            <p>No bookmarks yet. Save papers from the Feed.</p>
          </div>
        ) : filteredBookmarks.length === 0 ? (
          <div className="text-center py-16 text-gray-500">
            {selectedFolder
              ? <p className="text-sm">No bookmarks in &ldquo;{selectedFolder.name}&rdquo;. Use the folder icon on each card to add papers.</p>
              : <p className="text-sm">No bookmarks match your filter.</p>}
          </div>
        ) : (
          <div className="space-y-3">
            {filteredBookmarks.map((bm) =>
              bm.paper ? (
                <BookmarkCard
                  key={bm.id}
                  bm={bm}
                  paper={bm.paper}
                  folders={folders}
                  isSelected={selected?.id === bm.paper.id}
                  showingMoveMenu={movingBookmarkId === bm.paper_id}
                  onSelect={() => { setShowChat(false); setSelected((p) => p?.id === bm.paper!.id ? null : bm.paper!); }}
                  onRemove={() => remove(bm.paper_id)}
                  onToggleMoveMenu={() => setMovingBookmarkId((id) => id === bm.paper_id ? null : bm.paper_id)}
                  onSetFolders={(ids) => setBookmarkFolders(bm.paper_id, ids)}
                />
              ) : null
            )}
          </div>
        )}
      </div>

      {/* Right panel: detail or chat */}
      <AnimatePresence mode="wait">
        {selected && (
          <PaperPanel key="detail" paper={selected} onClose={() => setSelected(null)} />
        )}
        {showChat && !selected && (
          <BookmarksKBChat
            key={`kb-${selectedFolderId ?? "all"}-${selectedTopics.join(",")}`}
            count={visibleBookmarks.filter((b) => b.paper).length}
            namespaceKeys={selectedTopics}
            folderId={selectedFolderId}
            folderName={selectedFolder?.name ?? null}
            indexingReady={indexing.ready}
            onClose={() => setShowChat(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Bookmark card ─────────────────────────────────────────────────────────────
function BookmarkCard({
  bm, paper, folders, isSelected, showingMoveMenu,
  onSelect, onRemove, onToggleMoveMenu, onSetFolders,
}: {
  bm: Bookmark; paper: Paper; folders: BookmarkFolder[];
  isSelected: boolean; showingMoveMenu: boolean;
  onSelect: () => void; onRemove: () => void;
  onToggleMoveMenu: () => void; onSetFolders: (ids: string[]) => void;
}) {
  const [showAllAuthors, setShowAllAuthors] = useState(false);
  const [localFolderIds, setLocalFolderIds] = useState<Set<string>>(new Set(bm.folder_ids));
  const paperFolders = folders.filter((f) => bm.folder_ids.includes(f.id));

  function toggleFolder(fid: string) {
    setLocalFolderIds((prev) => {
      const next = new Set(prev);
      next.has(fid) ? next.delete(fid) : next.add(fid);
      return next;
    });
  }

  function applyFolders() {
    onSetFolders([...localFolderIds]);
  }

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.97 }}
      transition={{ duration: 0.18 }}
      onClick={onSelect}
      className={`group cursor-pointer rounded-2xl border p-4 transition-all duration-200 relative ${
        isSelected
          ? "border-amber-600/40 bg-amber-950/10 shadow-lg shadow-amber-900/10"
          : "border-gray-800/80 bg-gray-900/60 hover:border-gray-700 hover:bg-gray-900"
      }`}
    >
      {/* Top row */}
      <div className="flex items-start justify-between gap-3 mb-2.5">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[10px] font-mono font-semibold bg-gray-800 text-gray-400 px-2 py-0.5 rounded-md border border-gray-700/50">
            {paper.namespace_key}
          </span>
          {paperFolders.map((f) => (
            <div key={f.id} className="flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-md" style={{ backgroundColor: `${f.color || "#6366f1"}20`, color: f.color || "#6366f1" }}>
              <div className="w-1.5 h-1.5 rounded-sm" style={{ backgroundColor: f.color || "#6366f1" }} />
              {f.name}
            </div>
          ))}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {/* Folder membership picker */}
          <div className="relative">
            <button
              onClick={(e) => { e.stopPropagation(); setLocalFolderIds(new Set(bm.folder_ids)); onToggleMoveMenu(); }}
              className={`transition-colors p-1 ${paperFolders.length > 0 ? "text-indigo-400" : "text-gray-600 hover:text-indigo-400"}`}
              title="Manage folders"
            >
              <FolderIcon size={13} />
            </button>
            {showingMoveMenu && (
              <motion.div
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="absolute right-0 top-7 z-20 bg-gray-900 border border-white/10 rounded-xl shadow-2xl shadow-black/50 overflow-hidden min-w-[180px]"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="px-3 py-2 border-b border-white/5">
                  <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Folders</p>
                </div>
                <div className="py-1 max-h-48 overflow-y-auto">
                  {folders.length === 0 ? (
                    <p className="text-xs text-gray-600 px-3 py-2">No folders yet.</p>
                  ) : folders.map((f) => {
                    const checked = localFolderIds.has(f.id);
                    return (
                      <button key={f.id} onClick={() => toggleFolder(f.id)}
                        className="w-full flex items-center gap-2.5 px-3 py-1.5 hover:bg-white/5 transition-colors">
                        <div className="w-2 h-2 rounded-sm flex-shrink-0" style={{ backgroundColor: f.color || "#6366f1" }} />
                        <span className="flex-1 text-xs text-left text-gray-300 truncate">{f.name}</span>
                        <div className={`w-3.5 h-3.5 rounded border flex items-center justify-center flex-shrink-0 transition-all ${checked ? "bg-amber-500 border-amber-500" : "border-gray-700"}`}>
                          {checked && <CheckIcon size={9} className="text-white" strokeWidth={3} />}
                        </div>
                      </button>
                    );
                  })}
                </div>
                <div className="px-2 py-2 border-t border-white/5">
                  <button onClick={applyFolders}
                    className="w-full text-[11px] font-semibold bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg py-1.5 transition-colors">
                    Apply
                  </button>
                </div>
              </motion.div>
            )}
          </div>
          <button onClick={(e) => { e.stopPropagation(); onRemove(); }} className="text-gray-600 hover:text-red-400 transition-colors p-1">
            <Trash2Icon size={13} />
          </button>
        </div>
      </div>

      <h3 className={`font-semibold text-sm leading-snug mb-1.5 transition-colors ${isSelected ? "text-amber-200" : "text-white group-hover:text-amber-100"}`}>
        {paper.title}
      </h3>

      <div className="flex items-center gap-1.5 mb-2.5 min-w-0">
        <p className={`text-xs text-gray-500 leading-relaxed min-w-0 ${showAllAuthors ? "break-words" : "truncate"}`}>
          {showAllAuthors ? paper.authors.join(", ") : paper.authors.slice(0, 2).join(", ")}
        </p>
        {!showAllAuthors && paper.authors.length > 2 && (
          <button onClick={(e) => { e.stopPropagation(); setShowAllAuthors(true); }}
            className="flex-shrink-0 text-xs text-amber-400/70 hover:text-amber-300 underline underline-offset-2 whitespace-nowrap">
            +{paper.authors.length - 2} more
          </button>
        )}
        {paper.published_at && <span className="flex-shrink-0 text-xs text-gray-700 whitespace-nowrap">· {new Date(paper.published_at).getFullYear()}</span>}
      </div>

      <p className="text-xs text-gray-400 line-clamp-2 leading-relaxed mb-3">{paper.tldr ?? cleanAbstract(paper.abstract)}</p>

      {bm.note && (
        <p className="text-[11px] text-gray-600 italic truncate max-w-full mb-3">&ldquo;{bm.note}&rdquo;</p>
      )}

      {paper.key_concepts.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {paper.key_concepts.slice(0, 3).map((c) => (
            <span key={c} className="text-[10px] bg-teal-950/40 text-teal-400/80 border border-teal-900/30 px-2 py-0.5 rounded-full">{c}</span>
          ))}
          {paper.key_concepts.length > 3 && <span className="text-[10px] text-gray-600">+{paper.key_concepts.length - 3}</span>}
        </div>
      )}

      <div className="flex items-center gap-2 pt-2.5 border-t border-gray-800/60" onClick={(e) => e.stopPropagation()}>
        <BookmarkIcon size={10} className="text-amber-400" fill="currentColor" />
        <span className="text-[11px] text-amber-500/70">Saved</span>
        <a href={paper.source_url} target="_blank" rel="noopener noreferrer"
          className="ml-auto flex items-center gap-1 text-[11px] text-indigo-400 hover:text-indigo-300 transition-colors font-medium">
          arXiv <ExternalLinkIcon size={10} />
        </a>
      </div>
    </motion.div>
  );
}

