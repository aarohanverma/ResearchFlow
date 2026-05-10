/**
 * API client — thin wrapper around fetch that handles auth, base URL,
 * and consistent error surfacing. Every call returns typed data or throws.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "/api/v1";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem("rf_auth");
    if (!raw) return null;
    return (JSON.parse(raw) as { state?: { token?: string } })?.state?.token ?? null;
  } catch {
    return null;
  }
}

// Default request timeout. Long enough for cold backend handlers, short
// enough that a stuck request never freezes a page navigation. SSE/stream
// endpoints don't go through this client so they're not affected.
const DEFAULT_TIMEOUT_MS = 30_000;

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  opts: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...((opts.headers as Record<string, string>) || {}),
  };

  // Wire up an AbortController so the request can never hang forever.
  // If the caller already supplied a signal, we honour theirs and add
  // ours via abort-on-either.
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), DEFAULT_TIMEOUT_MS);
  const externalSignal = opts.signal;
  if (externalSignal) {
    if (externalSignal.aborted) ctrl.abort();
    else externalSignal.addEventListener("abort", () => ctrl.abort(), { once: true });
  }

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      headers,
      body: body != null ? JSON.stringify(body) : undefined,
      ...opts,
      signal: ctrl.signal,
    });
  } catch (err) {
    if ((err as { name?: string })?.name === "AbortError") {
      throw new Error("Request timed out. Please try again.");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    if (res.status === 401) {
      // Session expired or user deleted — clear auth and redirect to login.
      if (typeof window !== "undefined") {
        const { useAuthStore } = await import("@/store/auth");
        useAuthStore.getState().logout();
      }
      throw new Error("Session expired. Please log in again.");
    }
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body?: unknown) => request<T>("POST", path, body),
  put: <T>(path: string, body?: unknown) => request<T>("PUT", path, body),
  patch: <T>(path: string, body?: unknown) => request<T>("PATCH", path, body),
  delete: <T>(path: string) => request<T>("DELETE", path),
};

export function logout(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem("rf_auth");
  window.location.href = "/login";
}

/** SSE helper — returns an EventSource for streaming endpoints.
 *
 * Uses URLSearchParams to safely encode the token so special characters
 * (padding `=`, `+`, `/` in base64url JWTs) are always percent-encoded
 * and cannot break the query string.
 */
export function openSSE(path: string): EventSource {
  const token = getToken();
  let url = `${BASE}${path}`;
  if (token) {
    const sep = path.includes("?") ? "&" : "?";
    url += `${sep}token=${encodeURIComponent(token)}`;
  }
  return new EventSource(url);
}
