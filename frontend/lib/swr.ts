/**
 * Tiny stale-while-revalidate helper for GET requests.
 *
 * Speeds up perceived load on Feed / Genie / Bookmarks pages by:
 *
 *   1. First paint reads from sessionStorage if a fresh entry exists,
 *      so a tab-revisit shows data instantly instead of a skeleton.
 *   2. Always fires a network refresh in the background so the next
 *      paint reflects the source of truth.
 *
 * UI-only — never apply to mutation responses or to anything where
 * consistency matters more than latency. Per-key TTL keeps stale
 * snapshots from lingering past the point where they're misleading.
 */

import { api } from "@/lib/api";

interface CacheEntry<T> {
  v: T;
  t: number;  // epoch ms when written
}

const NS = "rf:swr:";

function read<T>(key: string, ttlMs: number): T | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(NS + key);
    if (!raw) return null;
    const entry = JSON.parse(raw) as CacheEntry<T>;
    if (!entry || typeof entry.t !== "number") return null;
    if (Date.now() - entry.t > ttlMs) return null;
    return entry.v;
  } catch {
    return null;
  }
}

function write<T>(key: string, value: T): void {
  if (typeof window === "undefined") return;
  try {
    sessionStorage.setItem(NS + key, JSON.stringify({ v: value, t: Date.now() }));
  } catch {
    // Quota exceeded — drop silently. The next read returns null and
    // the caller falls through to a normal fetch.
  }
}

/**
 * GET a path, fanning out cached value + fresh fetch.
 *
 * @param key      Unique cache key (typically the path + querystring).
 * @param path     API path to GET.
 * @param onValue  Receives every value — first call may be the cached
 *                 stale value, second the fresh one. Caller dedupes if
 *                 they care about reference equality.
 * @param ttlMs    Cache TTL — keep small for changing data.
 */
export async function swrGet<T>(
  key: string,
  path: string,
  onValue: (value: T, source: "cache" | "network") => void,
  ttlMs = 60_000,
): Promise<void> {
  const cached = read<T>(key, ttlMs);
  if (cached !== null) onValue(cached, "cache");
  try {
    const fresh = await api.get<T>(path);
    write(key, fresh);
    onValue(fresh, "network");
  } catch {
    // Surface nothing — caller already has cache (if any). A real
    // failure handler belongs in the caller's catch around their own
    // fetch when the cache miss matters.
    if (cached === null) throw new Error(`swrGet failed for ${path}`);
  }
}

/** Invalidate a specific key (call after a mutation). */
export function swrInvalidate(keyPrefix: string): void {
  if (typeof window === "undefined") return;
  try {
    const remove: string[] = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      const k = sessionStorage.key(i);
      if (k && k.startsWith(NS + keyPrefix)) remove.push(k);
    }
    for (const k of remove) sessionStorage.removeItem(k);
  } catch { /* ignore */ }
}
