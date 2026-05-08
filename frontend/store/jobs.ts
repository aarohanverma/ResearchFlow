import { create } from "zustand";
import { persist } from "zustand/middleware";
import { api } from "@/lib/api";

export interface StudyJob {
  job_id: string;
  paper_id: string;
  paper_title: string;
  expertise_level: string;
  status: "pending" | "running" | "done" | "error";
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface GenieJob {
  session_id: string;
  status: "pending" | "running" | "done" | "done_empty" | "failed" | "cancelled";
  capsule_id: string | null;
  error: string | null;
  created_at: string;
  completed_at: string | null;
  label?: string;
}

export interface GraphBuildJob {
  job_id: string;
  namespace_key: string | null;
  status: "running" | "done" | "failed";
  message: string | null;
  created_at: string;
  completed_at: string | null;
  group_id: string;          // all jobs from one Build Deep click share the same group_id
  namespace_count: number;   // total namespaces in this build group (informational)
}

export type AnyJob =
  | ({ kind: "study" } & StudyJob)
  | ({ kind: "genie" } & GenieJob)
  | ({ kind: "graph" } & GraphBuildJob);

interface JobsStore {
  jobs: StudyJob[];
  genieJobs: GenieJob[];
  graphBuildJobs: GraphBuildJob[];
  unreadCount: number;
  lastSeenAt: string | null;
  fetchJobs: () => Promise<void>;
  fetchGenieJob: (session_id: string, label?: string) => Promise<void>;
  addGenieJob: (job: GenieJob & { label?: string }) => void;
  cancelGenieJob: (session_id: string) => Promise<void>;
  dismissGenieJob: (session_id: string) => void;
  dismissJob: (job_id: string) => void;
  addGraphBuildJob: (job: GraphBuildJob) => void;
  dismissGraphBuildJob: (job_id: string) => void;
  markRead: () => void;
}

export const useJobsStore = create<JobsStore>()(
  persist(
    (set, get) => ({
      jobs: [],
      genieJobs: [],
      graphBuildJobs: [],
      unreadCount: 0,
      lastSeenAt: null,

      fetchJobs: async () => {
        try {
          const jobs = await api.get<StudyJob[]>("/study/jobs");
          const { lastSeenAt, genieJobs, graphBuildJobs } = get();

          const updatedGenie = await Promise.all(
            genieJobs.map(async (gj) => {
              if (gj.status === "done" || gj.status === "failed") return gj;
              try {
                const fresh = await api.get<GenieJob>(`/genie/sessions/${gj.session_id}`);
                return { ...gj, ...fresh };
              } catch {
                return gj;
              }
            })
          );

          // Poll graph build jobs that are still running
          const updatedGraphBuilds = await Promise.all(
            graphBuildJobs.map(async (gb) => {
              if (gb.status === "done" || gb.status === "failed") return gb;
              try {
                const fresh = await api.get<{ status: string; message?: string }>(
                  `/graph/build-deep/status/${gb.job_id}`
                );
                // "not_found" means the 2-hour TTL expired — treat as done so the
                // job doesn't stay stuck in "running" indefinitely.
                const effectiveStatus =
                  fresh.status === "not_found" ? "done" : fresh.status;
                const completed = effectiveStatus === "done" || effectiveStatus === "failed";
                return {
                  ...gb,
                  status: effectiveStatus as GraphBuildJob["status"],
                  message: fresh.message ?? null,
                  // Only record completion time on the first transition — never overwrite.
                  completed_at: completed && !gb.completed_at
                    ? new Date().toISOString()
                    : gb.completed_at,
                };
              } catch {
                return gb;
              }
            })
          );

          const newStudyDone = jobs.filter(
            (j) =>
              j.status === "done" &&
              j.finished_at &&
              (!lastSeenAt || j.finished_at > lastSeenAt)
          ).length;

          const newGenieDone = updatedGenie.filter(
            (gj) =>
              gj.status === "done" &&
              gj.capsule_id !== null &&
              gj.completed_at &&
              (!lastSeenAt || gj.completed_at > lastSeenAt)
          ).length;

          const newGraphDone = updatedGraphBuilds.filter(
            (gb) =>
              gb.status === "done" &&
              gb.completed_at &&
              (!lastSeenAt || gb.completed_at > lastSeenAt)
          ).length;

          set({
            jobs,
            genieJobs: updatedGenie,
            graphBuildJobs: updatedGraphBuilds,
            unreadCount: newStudyDone + newGenieDone + newGraphDone,
          });
        } catch {}
      },

      fetchGenieJob: async (session_id: string, label?: string) => {
        try {
          const data = await api.get<GenieJob>(`/genie/sessions/${session_id}`);
          const existing = get().genieJobs.find((gj) => gj.session_id === session_id);
          if (existing) {
            set((s) => ({
              genieJobs: s.genieJobs.map((gj) =>
                gj.session_id === session_id ? { ...gj, ...data } : gj
              ),
            }));
          } else {
            set((s) => ({ genieJobs: [...s.genieJobs, { ...data, label }] }));
          }
        } catch {}
      },

      addGenieJob: (job) => {
        set((s) => ({
          genieJobs: [
            ...s.genieJobs.filter((gj) => gj.session_id !== job.session_id),
            job,
          ],
        }));
      },

      cancelGenieJob: async (session_id) => {
        try {
          await api.post(`/genie/sessions/${session_id}/cancel`);
        } catch {}
        set((s) => ({ genieJobs: s.genieJobs.filter((gj) => gj.session_id !== session_id) }));
      },

      dismissGenieJob: (session_id) => {
        set((s) => ({ genieJobs: s.genieJobs.filter((gj) => gj.session_id !== session_id) }));
      },

      dismissJob: (job_id) => {
        set((s) => ({ jobs: s.jobs.filter((j) => j.job_id !== job_id) }));
      },

      addGraphBuildJob: (job) => {
        set((s) => ({
          graphBuildJobs: [
            ...s.graphBuildJobs.filter((g) => g.job_id !== job.job_id),
            { ...job, group_id: job.group_id || job.job_id, namespace_count: job.namespace_count || 1 },
          ],
        }));
      },

      dismissGraphBuildJob: (job_id) => {
        set((s) => ({ graphBuildJobs: s.graphBuildJobs.filter((g) => g.job_id !== job_id) }));
      },

      markRead: () => {
        set({ unreadCount: 0, lastSeenAt: new Date().toISOString() });
      },
    }),
    {
      name: "research-flow-jobs",
      partialize: (state) => ({
        genieJobs: state.genieJobs,
        graphBuildJobs: state.graphBuildJobs,
      }),
    }
  )
);
