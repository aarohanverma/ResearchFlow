"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  BellIcon,
  CheckCircleIcon,
  Loader2Icon,
  XCircleIcon,
  ClockIcon,
  FlaskConicalIcon,
  BookOpenIcon,
  BotIcon,
  XIcon,
  SquareIcon,
  NetworkIcon,
  SparklesIcon,
  PinIcon,
  Trash2Icon,
  DownloadIcon,
} from "lucide-react";
import { useJobsStore, type StudyJob, type GenieJob, type GraphBuildJob, type GenerationJob, type DeepDiveJob, type AssistantJob, type ArxivImportJob } from "@/store/jobs";

export function JobsNotification() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const bellRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const {
    jobs, genieJobs, graphBuildJobs, generationJobs, deepDiveJobs, assistantJobs, arxivImportJobs,
    unreadCount, lastSeenAt, pinnedJobKeys,
    fetchJobs, markRead, dismissGenieJob, dismissJob, cancelGenieJob,
    dismissGraphBuildJob, cancelGraphBuildJob, dismissGenerationJob, cancelGenerationJob, dismissDeepDiveJob,
    cancelAssistantJob, dismissAssistantJob, dismissArxivImportJob,
    pinJob, unpinJob,
  } = useJobsStore();

  const pinned = new Set(pinnedJobKeys);

  function clearAllCompleted() {
    // Pinned jobs are immune — dismiss only unpinned terminal-state jobs
    jobs.filter(j => !pinned.has(j.job_id) && (j.status === "done" || j.status === "error")).forEach(j => dismissJob(j.job_id));
    genieJobs.filter(g => !pinned.has(g.session_id) && (g.status === "done" || g.status === "done_empty" || g.status === "failed" || g.status === "cancelled")).forEach(g => dismissGenieJob(g.session_id));
    graphBuildJobs.filter(g => !pinned.has(g.job_id) && (g.status === "done" || g.status === "failed" || g.status === "cancelled")).forEach(g => dismissGraphBuildJob(g.job_id));
    generationJobs.filter(g => !pinned.has(g.artifact_id) && (g.status === "completed" || g.status === "failed")).forEach(g => dismissGenerationJob(g.artifact_id));
    deepDiveJobs.filter(d => !pinned.has(d.capsule_id) && (d.status === "done" || d.status === "failed")).forEach(d => dismissDeepDiveJob(d.capsule_id));
    assistantJobs.filter(a => !pinned.has(a.job_id) && (a.status === "completed" || a.status === "failed" || a.status === "cancelled")).forEach(a => dismissAssistantJob(a.job_id));
    arxivImportJobs.filter(a => !pinned.has(a.job_id) && (a.status === "completed" || a.status === "failed")).forEach(a => dismissArxivImportJob(a.job_id));
  }

  const hasAnyCompleted =
    jobs.some(j => !pinned.has(j.job_id) && (j.status === "done" || j.status === "error")) ||
    genieJobs.some(g => !pinned.has(g.session_id) && (g.status === "done" || g.status === "done_empty" || g.status === "failed" || g.status === "cancelled")) ||
    graphBuildJobs.some(g => !pinned.has(g.job_id) && (g.status === "done" || g.status === "failed" || g.status === "cancelled")) ||
    generationJobs.some(g => !pinned.has(g.artifact_id) && (g.status === "completed" || g.status === "failed")) ||
    deepDiveJobs.some(d => !pinned.has(d.capsule_id) && (d.status === "done" || d.status === "failed")) ||
    assistantJobs.some(a => !pinned.has(a.job_id) && (a.status === "completed" || a.status === "failed" || a.status === "cancelled")) ||
    arxivImportJobs.some(a => !pinned.has(a.job_id) && (a.status === "completed" || a.status === "failed"));

  // Adaptive poll cadence refs — updated without recreating the interval so
  // every status change doesn't tear down and rebuild the timer (interval thrash).
  const cadenceRef = useRef(15000);

  useEffect(() => {
    const hasShortPending =
      jobs.some((j) => j.status === "pending" || j.status === "running") ||
      genieJobs.some((g) => g.status === "pending" || g.status === "running") ||
      graphBuildJobs.some((g) => g.status === "running") ||
      arxivImportJobs.some((a) => a.status === "running");
    const hasMediaPending =
      generationJobs.some((g) => g.status === "queued" || g.status === "running") ||
      deepDiveJobs.some((d) => d.status === "generating") ||
      assistantJobs.some((a) => a.status === "pending" || a.status === "running");
    cadenceRef.current = hasShortPending ? 4000 : hasMediaPending ? 12000 : 15000;
  }, [jobs, genieJobs, graphBuildJobs, generationJobs, deepDiveJobs, assistantJobs, arxivImportJobs]);

  useEffect(() => {
    fetchJobs();
    // Single stable interval that reads cadenceRef on each tick so that
    // cadence changes take effect on the next poll without tearing down the
    // interval. This avoids the previous pattern where any status change
    // (including the ones produced by fetchJobs itself) caused the interval
    // to be cleared and recreated, producing interval thrash under active jobs.
    let timeoutId: ReturnType<typeof setTimeout>;
    let cancelled = false;

    function schedulePoll() {
      if (cancelled) return;
      timeoutId = setTimeout(() => {
        if (cancelled) return;
        fetchJobs();
        schedulePoll();
      }, cadenceRef.current);
    }

    schedulePoll();
    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      const target = e.target as Node;
      const inBell = bellRef.current?.contains(target);
      const inDropdown = dropdownRef.current?.contains(target);
      if (!inBell && !inDropdown) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  function toggle() {
    setOpen((o) => {
      if (!o) markRead();
      return !o;
    });
  }

  const totalJobs = jobs.length + genieJobs.length + graphBuildJobs.length + generationJobs.length + deepDiveJobs.length + assistantJobs.length + arxivImportJobs.length;
  const hasActive =
    jobs.some((j) => j.status === "running" || j.status === "pending") ||
    genieJobs.some((g) => g.status === "running" || g.status === "pending") ||
    graphBuildJobs.some((g) => g.status === "running") ||
    generationJobs.some((g) => g.status === "queued" || g.status === "running") ||
    deepDiveJobs.some((d) => d.status === "generating") ||
    assistantJobs.some((a) => a.status === "pending" || a.status === "running") ||
    arxivImportJobs.some((a) => a.status === "running");

  return (
    <div ref={bellRef} className="relative">
      <button
        onClick={toggle}
        className="relative flex items-center justify-center w-8 h-8 rounded-lg text-gray-500 hover:text-gray-300 hover:bg-gray-800 transition-all"
        title="Background jobs"
      >
        {/* Ping ring — fires when new completions arrive and panel is closed */}
        {unreadCount > 0 && !open && !hasActive && (
          <span className="absolute inset-0 rounded-lg animate-ping bg-emerald-500/25 pointer-events-none" />
        )}
        {hasActive ? (
          <Loader2Icon size={15} className="animate-spin text-indigo-400" />
        ) : (
          <BellIcon size={15} className={unreadCount > 0 ? "text-emerald-400" : ""} />
        )}
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-emerald-500 rounded-full text-[9px] font-bold text-white flex items-center justify-center">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div
          ref={dropdownRef}
          className="fixed left-[228px] top-4 w-80 bg-gray-900 border border-gray-700/60 rounded-xl shadow-2xl shadow-black/60 z-50 overflow-hidden"
        >
          <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between gap-2">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider flex-1">
              Background Jobs
            </p>
            <div className="flex items-center gap-2">
              {hasAnyCompleted && (
                <button
                  onClick={clearAllCompleted}
                  title="Clear all completed (pinned items are preserved)"
                  className="flex items-center gap-1 px-2 py-1 rounded text-[10px] text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                >
                  <Trash2Icon size={10} />
                  Clear
                </button>
              )}
              <span className="text-[10px] text-gray-700">{totalJobs}</span>
            </div>
          </div>

          <div className="max-h-80 overflow-y-auto">
            {totalJobs === 0 && (
              <p className="text-sm text-gray-600 text-center py-6">No jobs yet</p>
            )}

            {/* Feed Import jobs */}
            {arxivImportJobs.length > 0 && (
              <>
                <div className="px-4 py-2 border-b border-gray-800/60">
                  <p className="text-[10px] text-gray-700 font-semibold uppercase tracking-wider flex items-center gap-1.5">
                    <DownloadIcon size={10} className="text-emerald-600" />
                    Feed Import
                  </p>
                </div>
                <div className="divide-y divide-gray-800/40">
                  {arxivImportJobs.map((job) => (
                    <ArxivImportJobRow
                      key={job.job_id}
                      job={job}
                      isPinned={pinned.has(job.job_id)}
                      onDismiss={() => dismissArxivImportJob(job.job_id)}
                      onPin={() => pinJob(job.job_id)}
                      onUnpin={() => unpinJob(job.job_id)}
                    />
                  ))}
                </div>
              </>
            )}

            {/* Research Assistant jobs */}
            {assistantJobs.length > 0 && (
              <>
                <div className="px-4 py-2 border-b border-gray-800/60">
                  <p className="text-[10px] text-gray-700 font-semibold uppercase tracking-wider flex items-center gap-1.5">
                    <BotIcon size={10} className="text-sky-600" />
                    Research Assistant
                  </p>
                </div>
                <div className="divide-y divide-gray-800/40">
                  {assistantJobs.map((aj) => (
                    <AssistantJobRow
                      key={aj.job_id}
                      job={aj}
                      isPinned={pinned.has(aj.job_id)}
                      onClick={() => { router.push(aj.href || `/assistant?session=${aj.session_id}`); setOpen(false); }}
                      onCancel={() => cancelAssistantJob(aj.job_id)}
                      onDismiss={() => dismissAssistantJob(aj.job_id)}
                      onPin={() => pinJob(aj.job_id)}
                      onUnpin={() => unpinJob(aj.job_id)}
                    />
                  ))}
                </div>
              </>
            )}

            {/* Deep Dive generation jobs */}
            {deepDiveJobs.length > 0 && (
              <>
                <div className="px-4 py-2 border-b border-gray-800/60">
                  <p className="text-[10px] text-gray-700 font-semibold uppercase tracking-wider flex items-center gap-1.5">
                    <SparklesIcon size={10} className="text-fuchsia-600" />
                    Deep Dive
                  </p>
                </div>
                <div className="divide-y divide-gray-800/40">
                  {deepDiveJobs.map((dj) => (
                    <DeepDiveJobRow
                      key={dj.capsule_id}
                      job={dj}
                      isNew={isRecentlyCompleted(dj.completed_at, lastSeenAt)}
                      isPinned={pinned.has(dj.capsule_id)}
                      onClick={() => { if (dj.status === "done") { router.push(`/genie/idea/${dj.capsule_id}`); setOpen(false); } }}
                      onDismiss={() => dismissDeepDiveJob(dj.capsule_id)}
                      onPin={() => pinJob(dj.capsule_id)}
                      onUnpin={() => unpinJob(dj.capsule_id)}
                    />
                  ))}
                </div>
              </>
            )}

            {/* Graph Build jobs — grouped by group_id so one click = one entry */}
            {graphBuildJobs.length > 0 && (() => {
              // Collapse jobs from the same click into a single "group" entry
              const groups = new Map<string, typeof graphBuildJobs>();
              graphBuildJobs.forEach(gb => {
                const gid = gb.group_id || gb.job_id; // legacy jobs without group_id shown individually
                if (!groups.has(gid)) groups.set(gid, []);
                groups.get(gid)!.push(gb);
              });
              return (
                <>
                  <div className="px-4 py-2 border-b border-gray-800/60">
                    <p className="text-[10px] text-gray-700 font-semibold uppercase tracking-wider flex items-center gap-1.5">
                      <NetworkIcon size={10} className="text-violet-600" />
                      Graph Deep Build
                    </p>
                  </div>
                  <div className="divide-y divide-gray-800/40">
                    {[...groups.entries()].map(([gid, jobs]) => (
                      <GraphBuildGroupRow
                        key={gid}
                        jobs={jobs}
                        isPinned={pinned.has(gid)}
                        onClick={() => { router.push("/graph"); setOpen(false); }}
                        onDismiss={() => jobs.forEach(j => dismissGraphBuildJob(j.job_id))}
                        onCancel={() => jobs.forEach(j => cancelGraphBuildJob(j.job_id))}
                        onPin={() => pinJob(gid)}
                        onUnpin={() => unpinJob(gid)}
                      />
                    ))}
                  </div>
                </>
              );
            })()}

            {/* Media Generation jobs (podcast / slides) */}
            {generationJobs.length > 0 && (
              <>
                <div className="px-4 py-2 border-b border-gray-800/60">
                  <p className="text-[10px] text-gray-700 font-semibold uppercase tracking-wider flex items-center gap-1.5">
                    <SparklesIcon size={10} className="text-amber-600" />
                    Media Generation
                  </p>
                </div>
                <div className="divide-y divide-gray-800/40">
                  {generationJobs.map((gj) => (
                    <GenerationJobRow
                      key={gj.artifact_id}
                      job={gj}
                      isNew={isRecentlyCompleted(gj.completed_at, lastSeenAt)}
                      isPinned={pinned.has(gj.artifact_id)}
                      onClick={() => {
                        if (gj.status !== "completed") return;
                        if (gj.source_type === "paper") router.push(`/study/${gj.source_id}`);
                        else if (gj.source_type === "capsule") router.push(`/genie/idea/${gj.source_id}`);
                        setOpen(false);
                      }}
                      onCancel={() => cancelGenerationJob(gj.artifact_id)}
                      onDismiss={() => dismissGenerationJob(gj.artifact_id)}
                      onPin={() => pinJob(gj.artifact_id)}
                      onUnpin={() => unpinJob(gj.artifact_id)}
                    />
                  ))}
                </div>
              </>
            )}

            {/* Genie jobs */}
            {genieJobs.length > 0 && (
              <>
                <div className="px-4 py-2 border-b border-gray-800/60">
                  <p className="text-[10px] text-gray-700 font-semibold uppercase tracking-wider flex items-center gap-1.5">
                    <FlaskConicalIcon size={10} className="text-indigo-600" />
                    Genie Synthesis
                  </p>
                </div>
                <div className="divide-y divide-gray-800/40">
                  {genieJobs.map((gj) => (
                    <GenieJobRow
                      key={gj.session_id}
                      job={gj}
                      isPinned={pinned.has(gj.session_id)}
                      onClick={() => { if (gj.status === "done") { router.push("/genie?tab=discoveries"); setOpen(false); } }}
                      onDismiss={() => dismissGenieJob(gj.session_id)}
                      onCancel={() => cancelGenieJob(gj.session_id)}
                      onPin={() => pinJob(gj.session_id)}
                      onUnpin={() => unpinJob(gj.session_id)}
                    />
                  ))}
                </div>
              </>
            )}

            {/* Study jobs */}
            {jobs.length > 0 && (
              <>
                <div className="px-4 py-2 border-b border-gray-800/60">
                  <p className="text-[10px] text-gray-700 font-semibold uppercase tracking-wider flex items-center gap-1.5">
                    <BookOpenIcon size={10} className="text-teal-600" />
                    Study Generation
                  </p>
                </div>
                <div className="divide-y divide-gray-800/40">
                  {jobs.map((job) => (
                    <StudyJobRow
                      key={job.job_id}
                      job={job}
                      isPinned={pinned.has(job.job_id)}
                      onClick={() => { if (job.status === "done") { router.push(`/study/${job.paper_id}?level=${job.expertise_level}`); setOpen(false); } }}
                      onDismiss={() => dismissJob(job.job_id)}
                      onPin={() => pinJob(job.job_id)}
                      onUnpin={() => unpinJob(job.job_id)}
                    />
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (status === "done" || status === "completed") return <CheckCircleIcon size={13} className="text-emerald-400" />;
  if (status === "done_empty") return <CheckCircleIcon size={13} className="text-gray-600" />;
  if (status === "error" || status === "failed") return <XCircleIcon size={13} className="text-red-400" />;
  if (status === "cancelled") return <XCircleIcon size={13} className="text-gray-500" />;
  if (status === "running") return <Loader2Icon size={13} className="text-indigo-400 animate-spin" />;
  return <ClockIcon size={13} className="text-gray-600" />;
}

function StatusLabel({ status }: { status: string }) {
  if (status === "done" || status === "completed") return <span className="text-emerald-400">Done — click to view</span>;
  if (status === "done_empty") return <span className="text-gray-500">Complete — no synthesis produced</span>;
  if (status === "error" || status === "failed") return <span className="text-red-400">Failed</span>;
  if (status === "cancelled") return <span className="text-gray-500">Cancelled</span>;
  if (status === "running") return <span className="text-indigo-400 animate-pulse">Running…</span>;
  return <span className="text-gray-600">Queued</span>;
}

function isRecentlyCompleted(completedAt: string | null, lastSeenAt: string | null): boolean {
  if (!completedAt) return false;
  if (!lastSeenAt) return true;
  return completedAt > lastSeenAt;
}

function GenieJobRow({ job, isPinned, onClick, onDismiss, onCancel, onPin, onUnpin }: { job: GenieJob; isPinned?: boolean; onClick: () => void; onDismiss: () => void; onCancel: () => void; onPin: () => void; onUnpin: () => void }) {
  const isClickable = job.status === "done";
  const isActive = job.status === "pending" || job.status === "running";
  return (
    <div
      onClick={isClickable ? onClick : undefined}
      className={`px-4 py-3 flex items-start gap-3 transition-colors group ${isClickable ? "cursor-pointer hover:bg-gray-800/50" : "cursor-default"}`}
    >
      <div className="mt-0.5 shrink-0">
        <StatusIcon status={job.status} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-gray-200 truncate">
          {job.label || "Genie Synthesis"}
        </p>
        <p className="text-[10px] text-gray-600 mt-0.5">
          <StatusLabel status={job.status} />
        </p>
        {job.error && (
          <p className="text-[10px] text-red-500 mt-0.5 truncate">{job.error}</p>
        )}
      </div>
      <div className="mt-0.5 shrink-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
        <button onClick={(e) => { e.stopPropagation(); isPinned ? onUnpin() : onPin(); }}
          className={`transition-colors ${isPinned ? "opacity-100 text-indigo-400 hover:text-indigo-300" : "text-gray-600 hover:text-indigo-400"}`}
          title={isPinned ? "Unpin" : "Pin"}>
          <PinIcon size={11} className={isPinned ? "rotate-45" : ""} />
        </button>
        {isActive && (
          <button onClick={(e) => { e.stopPropagation(); onCancel(); }}
            className="text-red-500/70 hover:text-red-400 transition-colors" title="Stop job">
            <SquareIcon size={11} />
          </button>
        )}
        {!isActive && !isPinned && (
          <button onClick={(e) => { e.stopPropagation(); onDismiss(); }}
            className="text-gray-600 hover:text-gray-400 transition-colors" title="Dismiss">
            <XIcon size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

function StudyJobRow({ job, isPinned, onClick, onDismiss, onPin, onUnpin }: { job: StudyJob; isPinned?: boolean; onClick: () => void; onDismiss: () => void; onPin: () => void; onUnpin: () => void }) {
  const isClickable = job.status === "done";
  return (
    <div
      onClick={isClickable ? onClick : undefined}
      className={`px-4 py-3 flex items-start gap-3 transition-colors group ${
        isClickable ? "cursor-pointer hover:bg-gray-800/50" : "cursor-default"
      }`}
    >
      <div className="mt-0.5 shrink-0">
        <StatusIcon status={job.status} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-gray-200 truncate">
          {job.paper_title || job.paper_id}
        </p>
        <p className="text-[10px] text-gray-600 mt-0.5">
          {job.expertise_level} · <StatusLabel status={job.status} />
        </p>
      </div>
      <div className="mt-0.5 shrink-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
        <button onClick={(e) => { e.stopPropagation(); isPinned ? onUnpin() : onPin(); }}
          className={`transition-colors ${isPinned ? "opacity-100 text-indigo-400 hover:text-indigo-300" : "text-gray-600 hover:text-indigo-400"}`}
          title={isPinned ? "Unpin" : "Pin"}>
          <PinIcon size={11} className={isPinned ? "rotate-45" : ""} />
        </button>
        {!isPinned && (
          <button onClick={(e) => { e.stopPropagation(); onDismiss(); }}
            className="text-gray-600 hover:text-gray-400 transition-colors" title="Dismiss">
            <XIcon size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

/** Single row for a media-generation job. Click → redirect when completed. */
function GenerationJobRow({
  job, isNew, isPinned, onClick, onCancel, onDismiss, onPin, onUnpin,
}: {
  job: GenerationJob;
  isNew?: boolean;
  isPinned?: boolean;
  onClick: () => void;
  onCancel: () => void;
  onDismiss: () => void;
  onPin: () => void;
  onUnpin: () => void;
}) {
  const isClickable = job.status === "completed";
  const isActive = job.status === "queued" || job.status === "running";
  const TYPE_EMOJI: Record<string, string> = {
    podcast: "🎙",
    slides: "📊",
  };
  const SOURCE_LABEL: Record<string, string> = {
    paper: "Paper",
    capsule: "Idea",
    folder: "Folder",
  };

  // Map backend status to the StatusIcon/StatusLabel vocabulary
  const uiStatus =
    job.status === "completed" ? "done" :
    job.status === "failed"    ? "failed" :
    job.status === "running"   ? "running" :
                                 "pending";

  // Defensive: a stale persisted row could have undefined fields. Never
  // assume strings are non-empty before calling .charAt / .slice.
  const genType = String(job.generation_type ?? "");
  const sourceType = String(job.source_type ?? "");
  const titleLabel =
    (typeof job.title === "string" && job.title.trim()) ||
    SOURCE_LABEL[sourceType] ||
    sourceType ||
    "Job";
  const niceTypeName = genType
    ? genType.charAt(0).toUpperCase() + genType.slice(1)
    : "Generation";

  return (
    <div
      onClick={isClickable ? onClick : undefined}
      className={`px-4 py-3 flex items-start gap-3 transition-colors group relative ${
        isClickable ? "cursor-pointer hover:bg-gray-800/50" : "cursor-default"
      } ${isNew && isClickable ? "bg-emerald-500/5" : ""}`}
    >
      {isNew && isClickable && (
        <span className="absolute left-0 top-0 bottom-0 w-0.5 bg-emerald-500/70 rounded-r" />
      )}
      <div className="mt-0.5 shrink-0">
        <StatusIcon status={uiStatus} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-gray-200 truncate">
          {TYPE_EMOJI[genType] || ""}{" "}
          {niceTypeName}
        </p>
        <p className="text-[10px] text-gray-400 mt-0.5 truncate font-medium">
          {titleLabel}
        </p>
        <p className="text-[10px] text-gray-600 mt-0.5">
          <StatusLabel status={uiStatus} />
        </p>
        {job.error_message && (
          <p className="text-[10px] text-red-500 mt-0.5 truncate">{job.error_message}</p>
        )}
      </div>
      <div className="mt-0.5 shrink-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
        {/* Pin button — always visible on hover; pinned state persists across Clear All */}
        <button
          onClick={(e) => { e.stopPropagation(); isPinned ? onUnpin() : onPin(); }}
          className={`transition-colors ${isPinned ? "opacity-100 text-indigo-400 hover:text-indigo-300" : "text-gray-600 hover:text-indigo-400"}`}
          title={isPinned ? "Unpin" : "Pin (survives Clear All)"}
        >
          <PinIcon size={11} className={isPinned ? "rotate-45" : ""} />
        </button>
        {isActive && (
          <button
            onClick={(e) => { e.stopPropagation(); onCancel(); }}
            className="text-red-500/70 hover:text-red-400 transition-colors"
            title="Stop generation"
          >
            <SquareIcon size={11} />
          </button>
        )}
        {!isActive && !isPinned && (
          <button
            onClick={(e) => { e.stopPropagation(); onDismiss(); }}
            className="text-gray-600 hover:text-gray-400 transition-colors"
            title="Dismiss"
          >
            <XIcon size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

function DeepDiveJobRow({ job, isNew, isPinned, onClick, onDismiss, onPin, onUnpin }: { job: DeepDiveJob; isNew?: boolean; isPinned?: boolean; onClick: () => void; onDismiss: () => void; onPin: () => void; onUnpin: () => void }) {
  const isClickable = job.status === "done";
  const isActive = job.status === "generating";
  const uiStatus = job.status === "done" ? "done" : job.status === "failed" ? "failed" : "running";
  return (
    <div
      onClick={isClickable ? onClick : undefined}
      className={`px-4 py-3 flex items-start gap-3 transition-colors group relative ${isClickable ? "cursor-pointer hover:bg-gray-800/50" : "cursor-default"} ${isNew && isClickable ? "bg-emerald-500/5" : ""}`}
    >
      {isNew && isClickable && (
        <span className="absolute left-0 top-0 bottom-0 w-0.5 bg-emerald-500/70 rounded-r" />
      )}
      <div className="mt-0.5 shrink-0">
        <StatusIcon status={uiStatus} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-gray-200 truncate">
          {job.capsule_title || "Deep Dive"}
        </p>
        <p className="text-[10px] text-gray-600 mt-0.5">
          {isClickable
            ? <span className="text-emerald-400">Done — click to view</span>
            : uiStatus === "failed"
            ? <span className="text-red-400">Failed</span>
            : <span className="text-fuchsia-400 animate-pulse">Generating…</span>}
        </p>
        {job.error && <p className="text-[10px] text-red-500 mt-0.5 truncate">{job.error}</p>}
      </div>
      <div className="mt-0.5 shrink-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
        <button
          onClick={(e) => { e.stopPropagation(); isPinned ? onUnpin() : onPin(); }}
          className={`transition-colors ${isPinned ? "opacity-100 text-indigo-400 hover:text-indigo-300" : "text-gray-600 hover:text-indigo-400"}`}
          title={isPinned ? "Unpin" : "Pin (survives Clear All)"}
        >
          <PinIcon size={11} className={isPinned ? "rotate-45" : ""} />
        </button>
        {!isActive && !isPinned && (
          <button
            onClick={(e) => { e.stopPropagation(); onDismiss(); }}
            className="text-gray-600 hover:text-gray-400 transition-colors"
            title="Dismiss"
          >
            <XIcon size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

function AssistantJobRow({
  job, isPinned, onClick, onCancel, onDismiss, onPin, onUnpin,
}: {
  job: AssistantJob;
  isPinned?: boolean;
  onClick: () => void;
  onCancel: () => void;
  onDismiss: () => void;
  onPin: () => void;
  onUnpin: () => void;
}) {
  const isActive = job.status === "pending" || job.status === "running";
  const isClickable = !isActive;
  return (
    <div
      onClick={isClickable ? onClick : undefined}
      className={`px-4 py-3 flex items-start gap-3 transition-colors group ${isClickable ? "cursor-pointer hover:bg-gray-800/50" : "cursor-default"}`}
    >
      <div className="mt-0.5 shrink-0">
        <StatusIcon status={job.status} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-gray-200 truncate">{job.title || "Research Assistant"}</p>
        <p className="text-[10px] text-gray-500 mt-0.5 truncate">
          {job.namespace_key ? `${job.namespace_key} · ` : ""}
          <StatusLabel status={job.status} />
        </p>
        {job.summary && <p className="text-[10px] text-gray-600 mt-0.5 truncate">{job.summary}</p>}
      </div>
      <div className="mt-0.5 shrink-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
        <button
          onClick={(e) => { e.stopPropagation(); isPinned ? onUnpin() : onPin(); }}
          className={`transition-colors ${isPinned ? "opacity-100 text-indigo-400 hover:text-indigo-300" : "text-gray-600 hover:text-indigo-400"}`}
          title={isPinned ? "Unpin" : "Pin"}
        >
          <PinIcon size={11} className={isPinned ? "rotate-45" : ""} />
        </button>
        {isActive && (
          <button onClick={(e) => { e.stopPropagation(); onCancel(); }}
            className="text-red-500/70 hover:text-red-400 transition-colors" title="Stop assistant task">
            <SquareIcon size={11} />
          </button>
        )}
        {!isActive && !isPinned && (
          <button onClick={(e) => { e.stopPropagation(); onDismiss(); }}
            className="text-gray-600 hover:text-gray-400 transition-colors" title="Dismiss">
            <XIcon size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

function ArxivImportJobRow({ job, isPinned, onDismiss, onPin, onUnpin }: {
  job: ArxivImportJob;
  isPinned?: boolean;
  onDismiss: () => void;
  onPin: () => void;
  onUnpin: () => void;
}) {
  const isActive = job.status === "running";
  const uiStatus = job.status === "completed" ? "done" : job.status === "failed" ? "failed" : "running";
  return (
    <div className="px-4 py-3 flex items-start gap-3 transition-colors group cursor-default">
      <div className="mt-0.5 shrink-0">
        <StatusIcon status={uiStatus} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-gray-200 truncate">{job.title || job.arxiv_id}</p>
        <p className="text-[10px] text-gray-500 mt-0.5 truncate font-mono">{job.namespace_key}</p>
        <p className="text-[10px] text-gray-600 mt-0.5">
          {uiStatus === "done"
            ? <span className="text-emerald-400">Imported — check your feed</span>
            : uiStatus === "failed"
            ? <span className="text-red-400">{job.summary ?? "Failed"}</span>
            : <span className="text-emerald-400 animate-pulse">Importing…</span>}
        </p>
      </div>
      <div className="mt-0.5 shrink-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
        <button onClick={(e) => { e.stopPropagation(); isPinned ? onUnpin() : onPin(); }}
          className={`transition-colors ${isPinned ? "opacity-100 text-indigo-400 hover:text-indigo-300" : "text-gray-600 hover:text-indigo-400"}`}
          title={isPinned ? "Unpin" : "Pin"}>
          <PinIcon size={11} className={isPinned ? "rotate-45" : ""} />
        </button>
        {!isActive && !isPinned && (
          <button onClick={(e) => { e.stopPropagation(); onDismiss(); }}
            className="text-gray-600 hover:text-gray-400 transition-colors" title="Dismiss">
            <XIcon size={12} />
          </button>
        )}
      </div>
    </div>
  );
}

/** Renders a single row for a *group* of namespace build jobs from the same click. */
function GraphBuildGroupRow({ jobs, isPinned, onClick, onCancel, onDismiss, onPin, onUnpin }: { jobs: GraphBuildJob[]; isPinned?: boolean; onClick: () => void; onCancel: () => void; onDismiss: () => void; onPin: () => void; onUnpin: () => void }) {
  // Derive aggregate status: running > failed > done
  const anyRunning = jobs.some(j => j.status === "running");
  const anyFailed  = jobs.some(j => j.status === "failed");
  const anyCancelled = jobs.some(j => j.status === "cancelled");
  const groupStatus = anyRunning ? "running" : anyFailed ? "failed" : anyCancelled ? "cancelled" : "done";
  const isClickable = groupStatus === "done";
  const n = jobs.length;
  const doneCount = jobs.filter(j => j.status === "done").length;

  return (
    <div
      onClick={isClickable ? onClick : undefined}
      className={`px-4 py-3 flex items-start gap-3 transition-colors group ${isClickable ? "cursor-pointer hover:bg-gray-800/50" : "cursor-default"}`}
    >
      <div className="mt-0.5 shrink-0">
        <StatusIcon status={groupStatus} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-gray-200 truncate">
          Graph Deep Build
        </p>
        <p className="text-[10px] text-gray-600 mt-0.5">
          {groupStatus === "done"
            ? <span className="text-emerald-400">Done — click to view graph</span>
            : groupStatus === "failed"
            ? <span className="text-red-400">Failed</span>
            : <span className="text-violet-400 animate-pulse">Building…</span>}
        </p>
      </div>
      <div className="mt-0.5 shrink-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-all">
        <button onClick={(e) => { e.stopPropagation(); isPinned ? onUnpin() : onPin(); }}
          className={`transition-colors ${isPinned ? "opacity-100 text-indigo-400 hover:text-indigo-300" : "text-gray-600 hover:text-indigo-400"}`}
          title={isPinned ? "Unpin" : "Pin"}>
          <PinIcon size={11} className={isPinned ? "rotate-45" : ""} />
        </button>
        {groupStatus === "running" && (
          <button onClick={(e) => { e.stopPropagation(); onCancel(); }}
            className="text-red-500/70 hover:text-red-400 transition-colors" title="Stop graph build">
            <SquareIcon size={11} />
          </button>
        )}
        {groupStatus !== "running" && !isPinned && (
          <button onClick={(e) => { e.stopPropagation(); onDismiss(); }}
            className="text-gray-600 hover:text-gray-400 transition-colors" title="Dismiss">
            <XIcon size={12} />
          </button>
        )}
      </div>
    </div>
  );
}
