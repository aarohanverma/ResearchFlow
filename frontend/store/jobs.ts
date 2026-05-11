import { create } from "zustand";
import { persist } from "zustand/middleware";
import { api } from "@/lib/api";
import type { GenerationSourceType } from "@/types";

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

export interface GenerationJob {
  artifact_id: string;
  job_id: string;
  source_type: GenerationSourceType;
  source_id: string;
  generation_type: "podcast" | "slides";
  title: string;           // human-readable source entity name (paper title / folder name / idea title)
  status: "queued" | "running" | "completed" | "failed";
  error_message: string | null;
  blob_path: string | null;
  content: Record<string, unknown> | null;
  created_at: string;
  completed_at: string | null;
}

export interface DeepDiveJob {
  capsule_id: string;
  capsule_title: string;
  status: "generating" | "done" | "failed";
  created_at: string;
  completed_at: string | null;
  error: string | null;
}

export type AnyJob =
  | ({ kind: "study" } & StudyJob)
  | ({ kind: "genie" } & GenieJob)
  | ({ kind: "graph" } & GraphBuildJob)
  | ({ kind: "generation" } & GenerationJob)
  | ({ kind: "deepdive" } & DeepDiveJob);

interface JobsStore {
  jobs: StudyJob[];
  genieJobs: GenieJob[];
  graphBuildJobs: GraphBuildJob[];
  generationJobs: GenerationJob[];
  deepDiveJobs: DeepDiveJob[];
  // Artifact IDs dismissed this session. fetchJobs filters these out so
  // polls never resurrect a row the user already dismissed.
  // Plain string[] so Zustand can serialize it (Set is not JSON-safe).
  dismissedArtifactIds: string[];
  // Same idea for study jobs — /study/jobs returns the full list every poll,
  // so dismissing locally without tracking the ID lets the next poll bring it back.
  dismissedStudyJobIds: string[];
  // Per-job pin keys. Pinned jobs survive Clear All and cannot be dismissed.
  pinnedJobKeys: string[];
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
  addGenerationJob: (job: GenerationJob) => void;
  updateGenerationJob: (artifact_id: string, updates: Partial<GenerationJob>) => void;
  cancelGenerationJob: (artifact_id: string) => Promise<void>;
  dismissGenerationJob: (artifact_id: string) => void;
  addDeepDiveJob: (job: DeepDiveJob) => void;
  dismissDeepDiveJob: (capsule_id: string) => void;
  pinJob: (key: string) => void;
  unpinJob: (key: string) => void;
  markRead: () => void;
}

export const useJobsStore = create<JobsStore>()(
  persist(
    (set, get) => ({
      jobs: [],
      genieJobs: [],
      graphBuildJobs: [],
      generationJobs: [],
      deepDiveJobs: [],
      dismissedArtifactIds: [],
      dismissedStudyJobIds: [],
      pinnedJobKeys: [],
      unreadCount: 0,
      lastSeenAt: null,

      fetchJobs: async () => {
        // Guard against re-entrant polling — when the backend is busy
        // with a long-running generation the per-artifact GET can lag
        // beyond the 4s interval. Without this, multiple fetchJobs
        // pile up in parallel and stomp on each other's state writes.
        const w = window as unknown as { __rf_fetchJobs_inflight?: boolean };
        if (w.__rf_fetchJobs_inflight) return;
        w.__rf_fetchJobs_inflight = true;
        try {
          const rawJobs = await api.get<StudyJob[]>("/study/jobs");
          const { lastSeenAt, genieJobs, graphBuildJobs, deepDiveJobs, dismissedStudyJobIds } = get();
          const dismissedStudySet = new Set(dismissedStudyJobIds);
          const jobs = rawJobs.filter((j) => !dismissedStudySet.has(j.job_id));

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

          // Poll in-progress generation jobs from the backend JobStore.
          // The backend job store survives worker restart (when Redis is enabled)
          // and is the authoritative source — local store is only an optimistic mirror.
          let updatedGeneration = get().generationJobs;
          try {
            const remote = await api.get<{ jobs: GenerationJob[]; total: number }>(`/generate/jobs`);
            const dismissedSet = new Set(get().dismissedArtifactIds);
            const byId = new Map<string, GenerationJob>();
            // Backend wins for jobs it knows about — skip dismissed rows so
            // a poll never re-surfaces a job the user already dismissed.
            for (const j of remote.jobs || []) {
              if (dismissedSet.has(j.artifact_id)) continue;
              byId.set(j.artifact_id, {
                ...j,
                blob_path: j.blob_path ?? null,
                content: (j.content as Record<string, unknown> | undefined) ?? null,
                error_message: j.error_message ?? null,
                completed_at: j.completed_at ?? null,
              });
            }
            // Keep optimistic local entries that aren't yet in backend store
            for (const j of get().generationJobs) {
              if (!dismissedSet.has(j.artifact_id) && !byId.has(j.artifact_id))
                byId.set(j.artifact_id, j);
            }
            updatedGeneration = Array.from(byId.values());

            // We deliberately DO NOT fan out per-artifact GETs here.
            // /generate/jobs above is the authoritative backend snapshot
            // and already includes status/blob_path/content for every
            // in-flight job. Issuing N more parallel requests on every
            // 4-second tick saturates the worker queue when the user
            // has multiple generations running and was a frequent source
            // of UI freezes during media generation.
          } catch { /* /generate/jobs is best-effort; local store keeps working */ }

          // Poll in-progress deep dive jobs
          const updatedDeepDives = await Promise.all(
            deepDiveJobs.map(async (dj) => {
              if (dj.status === "done" || dj.status === "failed") return dj;
              try {
                const fresh = await api.get<{ deep_dive_status: string }>(`/genie/capsules/${dj.capsule_id}`);
                const newStatus =
                  fresh.deep_dive_status === "done" ? "done" :
                  fresh.deep_dive_status === "failed" ? "failed" :
                  "generating";
                const terminal = newStatus === "done" || newStatus === "failed";
                return {
                  ...dj,
                  status: newStatus as DeepDiveJob["status"],
                  completed_at: terminal && !dj.completed_at ? new Date().toISOString() : dj.completed_at,
                };
              } catch {
                return dj;
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

          const newGenerationDone = updatedGeneration.filter(
            (gj) =>
              gj.status === "completed" &&
              gj.completed_at &&
              (!lastSeenAt || gj.completed_at > lastSeenAt)
          ).length;

          const newDeepDiveDone = updatedDeepDives.filter(
            (dj) =>
              dj.status === "done" &&
              dj.completed_at &&
              (!lastSeenAt || dj.completed_at > lastSeenAt)
          ).length;

          // Functional set so we re-filter against the LATEST dismissed list.
          // Without this, a dismiss action that fires between the poll's
          // network awaits and this set() call would be silently
          // overwritten by the stale generationJobs computed earlier.
          set((s) => {
            const dropSet = new Set(s.dismissedArtifactIds);
            const finalGen = updatedGeneration.filter(
              (g) => !dropSet.has(g.artifact_id)
            );
            const dropStudySet = new Set(s.dismissedStudyJobIds);
            const finalJobs = jobs.filter((j) => !dropStudySet.has(j.job_id));
            return {
              jobs: finalJobs,
              genieJobs: updatedGenie,
              graphBuildJobs: updatedGraphBuilds,
              generationJobs: finalGen,
              deepDiveJobs: updatedDeepDives,
              unreadCount: newStudyDone + newGenieDone + newGraphDone + newGenerationDone + newDeepDiveDone,
            };
          });
        } catch {}
        finally {
          w.__rf_fetchJobs_inflight = false;
        }
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
        set((s) => {
          if (s.pinnedJobKeys.includes(session_id)) return s;
          return { genieJobs: s.genieJobs.filter((gj) => gj.session_id !== session_id) };
        });
      },

      dismissJob: (job_id) => {
        set((s) => {
          if (s.pinnedJobKeys.includes(job_id)) return s;
          return {
            jobs: s.jobs.filter((j) => j.job_id !== job_id),
            dismissedStudyJobIds: s.dismissedStudyJobIds.includes(job_id)
              ? s.dismissedStudyJobIds
              : [...s.dismissedStudyJobIds, job_id],
          };
        });
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
        set((s) => {
          if (s.pinnedJobKeys.includes(job_id)) return s;
          return { graphBuildJobs: s.graphBuildJobs.filter((g) => g.job_id !== job_id) };
        });
      },

      addGenerationJob: (job) => {
        set((s) => ({
          generationJobs: [
            ...s.generationJobs.filter((g) => g.artifact_id !== job.artifact_id),
            job,
          ],
        }));
      },

      updateGenerationJob: (artifact_id, updates) => {
        set((s) => ({
          generationJobs: s.generationJobs.map((g) =>
            g.artifact_id === artifact_id ? { ...g, ...updates } : g
          ),
        }));
      },

      cancelGenerationJob: async (artifact_id) => {
        // Optimistically flip to cancelled. The backend cancel endpoint now also
        // updates the JobStore, so the next poll returns "failed" instead of
        // "running" — no need to add to dismissedArtifactIds (which would make
        // the job vanish silently rather than showing a cancellable "Failed" row).
        set((s) => ({
          generationJobs: s.generationJobs.map((g) =>
            g.artifact_id === artifact_id
              ? { ...g, status: "failed" as const, error_message: "cancelled" }
              : g
          ),
        }));
        try {
          await api.post(`/generate/artifact/${artifact_id}/cancel`);
        } catch {
          /* best-effort — local state already shows cancelled */
        }
      },

      addDeepDiveJob: (job) => {
        set((s) => ({
          deepDiveJobs: [
            ...s.deepDiveJobs.filter((d) => d.capsule_id !== job.capsule_id),
            job,
          ],
        }));
      },

      dismissDeepDiveJob: (capsule_id) => {
        set((s) => {
          if (s.pinnedJobKeys.includes(capsule_id)) return s;
          return { deepDiveJobs: s.deepDiveJobs.filter((d) => d.capsule_id !== capsule_id) };
        });
      },

      pinJob: (key) => {
        set((s) => ({
          pinnedJobKeys: s.pinnedJobKeys.includes(key) ? s.pinnedJobKeys : [...s.pinnedJobKeys, key],
        }));
      },

      unpinJob: (key) => {
        set((s) => ({ pinnedJobKeys: s.pinnedJobKeys.filter((k) => k !== key) }));
      },

      dismissGenerationJob: (artifact_id) => {
        set((s) => {
          if (s.pinnedJobKeys.includes(artifact_id)) return s;
          return {
            generationJobs: s.generationJobs.filter((g) => g.artifact_id !== artifact_id),
            dismissedArtifactIds: s.dismissedArtifactIds.includes(artifact_id)
              ? s.dismissedArtifactIds
              : [...s.dismissedArtifactIds, artifact_id],
          };
        });
      },

      markRead: () => {
        set({ unreadCount: 0, lastSeenAt: new Date().toISOString() });
      },
    }),
    {
      name: "research-flow-jobs",
      // Bump version when the persisted shape changes. We strip any rows
      // whose generation_type is no longer supported (e.g. legacy "video"
      // / "interactive" rows from earlier installs) so the JobsPanel
      // never tries to render a deprecated card and crash the app.
      version: 2,
      migrate: (persisted: unknown, _version: number) => {
        const state = (persisted ?? {}) as Partial<JobsStore>;
        const allowed = new Set<GenerationJob["generation_type"]>(["podcast", "slides"]);
        const cleanGen = Array.isArray(state.generationJobs)
          ? state.generationJobs.filter(
              (g): g is GenerationJob =>
                !!g && typeof g === "object" && allowed.has((g as GenerationJob).generation_type)
            )
          : [];
        return {
          ...(state as object),
          generationJobs: cleanGen,
        } as JobsStore;
      },
      partialize: (state) => ({
        // Only persist genie/graph slices which are small + bounded.
        // generationJobs is intentionally NOT persisted — its `content`
        // field can be 30-50KB per row (podcast script / slide markdown),
        // and the JobsPanel polls /generate/jobs every 12s, which would
        // otherwise trigger a 50KB synchronous localStorage write on
        // every poll while a podcast generates and stutter the UI.
        // The backend /generate/jobs endpoint is the source of truth on
        // refresh anyway.
        genieJobs: state.genieJobs,
        graphBuildJobs: state.graphBuildJobs,
        pinnedJobKeys: state.pinnedJobKeys,
        // Persist dismissals — without these the backend re-serves dismissed
        // rows on hard reload and the notification returns.
        dismissedStudyJobIds: state.dismissedStudyJobIds,
        dismissedArtifactIds: state.dismissedArtifactIds,
      }),
    }
  )
);
