const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

export async function apiFetchJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const res = await fetch(url, init);
  const contentType = res.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const body = isJson ? await res.json().catch(() => null) : await res.text().catch(() => null);
  if (!res.ok) {
    const detail =
      typeof body === "object" && body && "detail" in (body as any) ? String((body as any).detail) : null;
    throw new ApiError(detail || `Request failed (${res.status})`, res.status, body);
  }
  return body as T;
}

export function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") return reject(new Error("Invalid file result"));
      const commaIdx = result.indexOf(",");
      resolve(commaIdx >= 0 ? result.slice(commaIdx + 1) : result);
    };
    reader.readAsDataURL(file);
  });
}
