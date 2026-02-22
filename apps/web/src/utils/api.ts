const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || "http://localhost:8000";
const DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024;
const DEFAULT_MAX_DIMENSION = 1600;

type Base64Options = {
  maxBytes?: number;
  maxDimension?: number;
  quality?: number;
  onProgress?: (percent: number) => void;
};

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

function readAsDataUrl(file: Blob, onProgress?: (percent: number) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onprogress = (event) => {
      if (!onProgress) return;
      if (!event.lengthComputable || event.total <= 0) return;
      onProgress(Math.max(1, Math.min(100, Math.round((event.loaded / event.total) * 100))));
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") return reject(new Error("Invalid file result"));
      resolve(result);
    };
    reader.readAsDataURL(file);
  });
}

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to decode image"));
    img.src = dataUrl;
  });
}

async function resizeImage(file: File, maxDimension: number, quality: number): Promise<Blob> {
  const srcDataUrl = await readAsDataUrl(file);
  const img = await loadImage(srcDataUrl);
  const ratio = Math.min(1, maxDimension / Math.max(img.width, img.height));
  const targetW = Math.max(1, Math.round(img.width * ratio));
  const targetH = Math.max(1, Math.round(img.height * ratio));

  const canvas = document.createElement("canvas");
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Failed to initialize image canvas");
  ctx.drawImage(img, 0, 0, targetW, targetH);

  return await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) return reject(new Error("Failed to encode resized image"));
        resolve(blob);
      },
      "image/jpeg",
      quality,
    );
  });
}

export async function toBase64(file: File, options: Base64Options = {}): Promise<string> {
  const maxBytes = options.maxBytes ?? DEFAULT_MAX_UPLOAD_BYTES;
  const maxDimension = options.maxDimension ?? DEFAULT_MAX_DIMENSION;
  const quality = options.quality ?? 0.82;
  const onProgress = options.onProgress;

  let working: Blob = file;
  if (file.size > maxBytes) {
    working = await resizeImage(file, maxDimension, quality);
  }

  if (working.size > maxBytes) {
    throw new Error(`이미지 용량이 너무 큽니다. ${Math.ceil(maxBytes / (1024 * 1024))}MB 이하로 업로드해 주세요.`);
  }

  const dataUrl = await readAsDataUrl(working, onProgress);
  const commaIdx = dataUrl.indexOf(",");
  return commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : dataUrl;
}
