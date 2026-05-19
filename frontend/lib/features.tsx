"use client";

/**
 * Feature flag context — fetches /settings/public once per session and
 * exposes a hook that any descendant component can use to gate UI.
 *
 * Resolution is server-side: the backend already merges
 *   defaults → admin global → per-user override
 * and returns the effective map. The frontend trusts whatever it gets
 * and falls back to "feature off" if the fetch fails, so misconfigured
 * deploys never accidentally surface a gated feature.
 *
 * Add a new flag in three places to make it fully wired end-to-end:
 *   1. ``backend/app/services/feature_flags.py``  — register the key.
 *   2. Backend call sites — guard the route and any cross-feature
 *      callsite with ``Depends(require_feature("…"))``.
 *   3. Frontend — call ``useFeature("…")`` wherever the button / nav /
 *      panel lives and conditionally render.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { api } from "@/lib/api";

export type FeatureMap = Record<string, boolean>;

interface FeatureContextValue {
  features: FeatureMap;
  loaded: boolean;
  refresh: () => Promise<void>;
}

const FeatureContext = createContext<FeatureContextValue>({
  features: {},
  loaded: false,
  refresh: async () => {},
});

export function FeatureProvider({
  enabled,
  children,
}: {
  /** Set false when the user isn't authenticated yet — skips the fetch. */
  enabled: boolean;
  children: ReactNode;
}) {
  const [features, setFeatures] = useState<FeatureMap>({});
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    if (!enabled) {
      setLoaded(true);
      return;
    }
    try {
      const s = await api.get<{ graph_enabled: boolean; features: FeatureMap }>("/settings/public");
      // Backend always returns the full effective map; ``graph_enabled``
      // is still there for backward compatibility.
      setFeatures({ ...(s.features || {}), graph_enabled: !!s.graph_enabled });
    } catch {
      // Network error / 401 / etc. — leave features empty so callers
      // see ``useFeature(...) === false`` and conditionally render off.
      setFeatures({});
    } finally {
      setLoaded(true);
    }
  }, [enabled]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const value = useMemo(() => ({ features, loaded, refresh }), [features, loaded, refresh]);

  return <FeatureContext.Provider value={value}>{children}</FeatureContext.Provider>;
}

/**
 * Returns whether ``name`` is enabled for the current user.
 *
 * Defaults to ``defaultValue`` (false unless caller overrides) when the
 * map hasn't loaded yet — so UI never flickers a gated control on first
 * paint before the fetch completes.
 */
export function useFeature(name: string, defaultValue = false): boolean {
  const { features, loaded } = useContext(FeatureContext);
  if (!loaded) return defaultValue;
  const v = features[name];
  return typeof v === "boolean" ? v : defaultValue;
}

/**
 * Returns the whole feature map and a refresh callback. Useful for
 * admin tooling that needs to react to live changes.
 */
export function useFeatures(): FeatureContextValue {
  return useContext(FeatureContext);
}
