const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  code: string | null;

  constructor(message: string, status: number, code: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function apiGet<T>(path: string, sessionId?: string): Promise<T> {
  return request<T>(path, { method: "GET" }, sessionId);
}

export async function apiPost<T>(path: string, body: unknown, sessionId?: string): Promise<T> {
  return request<T>(
    path,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    },
    sessionId
  );
}

export async function apiDownload(path: string, sessionId: string): Promise<Blob> {
  const response = await fetch(apiUrl(path), {
    headers: { "X-UI-Session-ID": sessionId }
  });
  if (!response.ok) {
    throw await parseError(response);
  }
  return response.blob();
}

async function request<T>(path: string, init: RequestInit, sessionId?: string): Promise<T> {
  const headers = new Headers(init.headers);
  if (sessionId) {
    headers.set("X-UI-Session-ID", sessionId);
  }
  const response = await fetch(apiUrl(path), { ...init, headers });
  if (!response.ok) {
    throw await parseError(response);
  }
  return response.json() as Promise<T>;
}

async function parseError(response: Response): Promise<ApiError> {
  try {
    const payload = await response.json();
    const error = payload?.error;
    const message = error?.message ?? payload?.detail ?? `Falha HTTP ${response.status}`;
    return new ApiError(String(message), response.status, error?.code ?? null);
  } catch {
    return new ApiError(`Falha HTTP ${response.status}`, response.status);
  }
}
