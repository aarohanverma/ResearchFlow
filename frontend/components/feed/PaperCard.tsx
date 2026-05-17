"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BookmarkIcon, ExternalLinkIcon, EyeOffIcon, EyeIcon } from "lucide-react";
import type { FeedItem } from "@/types";
import { cleanAbstract } from "@/lib/utils";
import { useBookmarksStore } from "@/store/bookmarks";
import { BookmarkFolderPicker } from "@/components/bookmarks/BookmarkFolderPicker";

interface Props {
  item: FeedItem;
  isSelected: boolean;
  onClick: () => void;
  onFeedback: (paperId: string, signal: string) => void;
  isHidden?: boolean;
  onHide?: () => void;
}

export function PaperCard({ item, isSelected, onClick, onFeedback, isHidden, onHide }: Props) {
  const { paper } = item;
  const { initialize, isBookmarked, add, remove } = useBookmarksStore();
  const bookmarked = isBookmarked(paper.id);
  const [showAllAuthors, setShowAllAuthors] = useState(false);
  const [showFolderPicker, setShowFolderPicker] = useState(false);
  const [folderIds, setFolderIds] = useState<string[]>([]);
  const bmBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => { initialize(); }, [initialize]);

  function openPicker(e: React.MouseEvent) {
    e.stopPropagation();
    setShowFolderPicker((v) => !v);
  }

  function handleSaved(ids: string[]) {
    setFolderIds(ids);
    add(paper.id);
    setShowFolderPicker(false);
    onFeedback(paper.id, "save");
  }

  function handleRemoved() {
    remove(paper.id);
    setFolderIds([]);
    setShowFolderPicker(false);
  }

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: isHidden ? 0.45 : 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.2 }}
      onClick={onClick}
      className={`group cursor-pointer rounded-2xl border p-5 transition-all duration-200 ${
        isSelected ? "border-indigo-500/40" : ""
      }`}
      style={{
        background: isSelected ? "rgba(99,102,241,0.06)" : "var(--rf-card)",
        borderColor: isSelected ? undefined : isHidden ? "var(--rf-border)" : "var(--rf-card-border)",
        boxShadow: isSelected ? "0 4px 16px rgba(99,102,241,0.08)" : "0 1px 3px rgba(0,0,0,0.04)",
        transform: "translateZ(0)",
        filter: isHidden ? "saturate(0.5)" : undefined,
      }}
    >
      {/* Top row */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-md border"
            style={{ background: "var(--rf-surface3)", color: "var(--rf-text4)", borderColor: "var(--rf-border2)" }}>
            {paper.namespace_key}
          </span>
        </div>
      </div>

      {/* Title */}
      <h3
        className="font-semibold leading-snug mb-1.5 transition-colors duration-150"
        style={{ color: isSelected ? "#6366f1" : "var(--rf-text1)" }}
      >
        {paper.title}
      </h3>

      {/* Authors */}
      <div className="flex items-center gap-1.5 mb-2.5 min-w-0">
        <p className={`text-xs text-gray-500 leading-relaxed min-w-0 ${showAllAuthors ? "break-words" : "truncate"}`}>
          {showAllAuthors
            ? paper.authors.join(", ")
            : paper.authors.slice(0, 2).join(", ")}
        </p>
        {!showAllAuthors && paper.authors.length > 2 && (
          <button
            onClick={(e) => { e.stopPropagation(); setShowAllAuthors(true); }}
            className="flex-shrink-0 text-xs text-indigo-400/70 hover:text-indigo-300 underline underline-offset-2 whitespace-nowrap"
          >
            +{paper.authors.length - 2} more
          </button>
        )}
        {paper.published_at && (
          <span className="flex-shrink-0 text-xs text-gray-700 whitespace-nowrap">
            · {new Date(paper.published_at).getFullYear()}
          </span>
        )}
      </div>

      {/* TLDR / abstract preview */}
      <p className="text-sm line-clamp-2 leading-relaxed mb-3.5" style={{ color: "var(--rf-text4)" }}>
        {paper.tldr ?? cleanAbstract(paper.abstract)}
      </p>

      {/* Concepts */}
      {paper.key_concepts.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3.5">
          {paper.key_concepts.slice(0, 4).map((c) => (
            <span key={c} className="text-[10px] bg-teal-950/40 text-teal-400/80 border border-teal-900/30 px-2 py-0.5 rounded-full">
              {c}
            </span>
          ))}
          {paper.key_concepts.length > 4 && (
            <span className="text-[10px] text-gray-600">+{paper.key_concepts.length - 4}</span>
          )}
        </div>
      )}

      {/* Actions */}
      <div
        className="flex items-center gap-1 pt-3 border-t relative"
        style={{ borderColor: "var(--rf-border)" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Bookmark button + folder picker */}
        <div className="relative">
          <button
            ref={bmBtnRef}
            onClick={bookmarked ? undefined : openPicker}
            disabled={bookmarked}
            className={`flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1.5 rounded-lg transition-all duration-150 ${
              bookmarked
                ? "text-amber-400 bg-amber-950/40 cursor-default"
                : "text-gray-600 hover:text-gray-300 hover:bg-gray-800"
            }`}
          >
            <BookmarkIcon size={12} fill={bookmarked ? "currentColor" : "none"} />
            {bookmarked ? "Saved" : "Save"}
          </button>

          <AnimatePresence>
            {showFolderPicker && !bookmarked && (
              <BookmarkFolderPicker
                paperId={paper.id}
                isBookmarked={bookmarked}
                currentFolderIds={folderIds}
                anchorRef={bmBtnRef}
                onClose={() => setShowFolderPicker(false)}
                onSaved={handleSaved}
                onRemoved={handleRemoved}
              />
            )}
          </AnimatePresence>
        </div>

        {onHide && (
          <button
            onClick={(e) => { e.stopPropagation(); onHide(); }}
            title={isHidden ? "Unhide from this namespace" : "Hide from this namespace"}
            className={`flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1.5 rounded-lg transition-all duration-150 ${
              isHidden
                ? "text-indigo-400 bg-indigo-950/40 hover:bg-indigo-900/40"
                : "text-gray-600 hover:text-gray-300 hover:bg-gray-800"
            }`}
          >
            {isHidden ? <EyeIcon size={12} /> : <EyeOffIcon size={12} />}
            {isHidden ? "Unhide" : "Hide"}
          </button>
        )}

        <a
          href={paper.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto flex items-center gap-1 text-[11px] text-indigo-400 hover:text-indigo-300 transition-colors font-medium"
        >
          arXiv <ExternalLinkIcon size={10} />
        </a>
      </div>
    </motion.div>
  );
}

