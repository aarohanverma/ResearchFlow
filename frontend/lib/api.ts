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

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body != null ? JSON.stringify(body) : undefined,
    ...opts,
  });

  if (!res.ok) {
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
  localStorage.removeItem("rf_auth");
  window.location.href = "/login";
}

/** SSE helper — returns an EventSource for streaming endpoints */
export function openSSE(path: string): EventSource {
  const token = getToken();
  const sep = path.includes("?") ? "&" : "?";
  const url = `${BASE}${path}${token ? `${sep}token=${token}` : ""}`;
  return new EventSource(url);
}
