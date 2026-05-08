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
  XIcon,
  SquareIcon,
  NetworkIcon,
} from "lucide-react";
import { useJobsStore, type StudyJob, type GenieJob, type GraphBuildJob } from "@/store/jobs";

export function JobsNotification() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const bellRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const { jobs, genieJobs, graphBuildJobs, unreadCount, fetchJobs, markRead, dismissGenieJob, dismissJob, cancelGenieJob, dismissGraphBuildJob } = useJobsStore();

  const statuses = [
    ...jobs.map((j) => j.status),
    ...genieJobs.map((g) => g.status),
    ...graphBuildJobs.map((g) => g.status),
  ].join(",");

  useEffect(() => {
    fetchJobs();
    const hasPending =
      jobs.some((j) => j.status === "pending" || j.status === "running") ||
      genieJobs.some((g) => g.status === "pending" || g.status === "running") ||
      graphBuildJobs.some((g) => g.status === "running");
    const interval = setInterval(fetchJobs, hasPending ? 4000 : 15000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobs.length, genieJobs.length, graphBuildJobs.length, statuses]);

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
    setOpen((o) => !o);
    if (!open) markRead();
  }

  const totalJobs = jobs.length + genieJobs.length + graphBuildJobs.length;
  const hasActive =
    jobs.some((j) => j.status === "running" || j.status === "pending") ||
    genieJobs.some((g) => g.status === "running" || g.status === "pending") ||
    graphBuildJobs.some((g) => g.status === "running");

  return (
    <div ref={bellRef} className="relative">
      <button
        onClick={toggle}
        className="relative flex items-center justify-center w-8 h-8 rounded-lg text-gray-500 hover:text-gray-300 hover:bg-gray-800 transition-all"
        title="Background jobs"
      >
        {hasActive ? (
          <Loader2Icon size={15} className="animate-spin text-indigo-400" />
        ) : (
          <BellIcon size={15} />
        )}
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-indigo-500 rounded-full text-[9px] font-bold text-white flex items-center justify-center">
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div
          ref={dropdownRef}
          className="fixed left-[228px] top-4 w-80 bg-gray-900 border border-gray-700/60 rounded-xl shadow-2xl shadow-black/60 z-50 overflow-hidden"
        >
          <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Background Jobs
            </p>
            <span className="text-[10px] text-gray-600">{totalJobs} total</span>
          </div>

          <div className="max-h-80 overflow-y-auto">
            {totalJobs === 0 && (
              <p className="text-sm text-gray-600 text-center py-6">No jobs yet</p>
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
                        onClick={() => { router.push("/graph"); setOpen(false); }}
                        onDismiss={() => jobs.forEach(j => dismissGraphBuildJob(j.job_id))}
                      />
                    ))}
                  </div>
                </>
              );
            })()}

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
                      onClick={() => {
                        if (gj.status === "done") {
                          router.push("/genie?tab=discoveries");
                          setOpen(false);
                        }
                      }}
                      onDismiss={() => dismissGenieJob(gj.session_id)}
                      onCancel={() => cancelGenieJob(gj.session_id)}
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
                      onClick={() => {
                        if (job.status === "done") {
                          router.push(`/study/${job.paper_id}?level=${job.expertise_level}`);
                          setOpen(false);
                        }
                      }}
                      onDismiss={() => dismissJob(job.job_id)}
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
  if (status === "done") return <CheckCircleIcon size={13} className="text-emerald-400" />;
  if (status === "done_empty") return <CheckCircleIcon size={13} className="text-gray-600" />;
  if (status === "error" || status === "failed") return <XCircleIcon size={13} className="text-red-400" />;
  if (status === "cancelled") return <XCircleIcon size={13} className="text-gray-500" />;
  if (status === "running") return <Loader2Icon size={13} className="text-indigo-400 animate-spin" />;
  return <ClockIcon size={13} className="text-gray-600" />;
}

function StatusLabel({ status }: { status: string }) {
  if (status === "done") return <span className="text-emerald-400">Done — click to view</span>;
  if (status === "done_empty") return <span className="text-gray-500">Complete — no synthesis produced</span>;
  if (status === "error" || status === "failed") return <span className="text-red-400">Failed</span>;
  if (status === "cancelled") return <span className="text-gray-500">Cancelled</span>;
  if (status === "running") return <span className="text-indigo-400 animate-pulse">Running…</span>;
  return <span className="text-gray-600">Queued</span>;
}

function GenieJobRow({ job, onClick, onDismiss, onCancel }: { job: GenieJob; onClick: () => void; onDismiss: () => void; onCancel: () => void }) {
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
        {isActive && (
          <button
            onClick={(e) => { e.stopPropagation(); onCancel(); }}
            className="text-red-500/70 hover:text-red-400 transition-colors"
            title="Stop job"
          >
            <SquareIcon size={11} />
          </button>
        )}
        {!isActive && (
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

function StudyJobRow({ job, onClick, onDismiss }: { job: StudyJob; onClick: () => void; onDismiss: () => void }) {
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
      <button
        onClick={(e) => { e.stopPropagation(); onDismiss(); }}
        className="mt-0.5 shrink-0 opacity-0 group-hover:opacity-100 text-gray-600 hover:text-gray-400 transition-all"
        title="Dismiss"
      >
        <XIcon size={12} />
      </button>
    </div>
  );
}

/** Renders a single row for a *group* of namespace build jobs from the same click. */
function GraphBuildGroupRow({ jobs, onClick, onDismiss }: { jobs: GraphBuildJob[]; onClick: () => void; onDismiss: () => void }) {
  // Derive aggregate status: running > failed > done
  const anyRunning = jobs.some(j => j.status === "running");
  const anyFailed  = jobs.some(j => j.status === "failed");
  const groupStatus = anyRunning ? "running" : anyFailed ? "failed" : "done";
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
      {groupStatus !== "running" && (
        <button
          onClick={(e) => { e.stopPropagation(); onDismiss(); }}
          className="mt-0.5 shrink-0 opacity-0 group-hover:opacity-100 text-gray-600 hover:text-gray-400 transition-all"
          title="Dismiss"
        >
          <XIcon size={12} />
        </button>
      )}
    </div>
  );
}
