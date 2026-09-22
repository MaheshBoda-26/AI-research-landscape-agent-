/**
 * API client.
 *
 * The browser talks to the FastAPI service directly rather than through the
 * Next rewrite, because a long-lived streaming POST is exactly the shape that
 * intermediaries buffer. The API's CORS config already allows the dev origin.
 * Set `NEXT_PUBLIC_API_ORIGIN` to point somewhere else.
 */

import type {
  HealthOut,
  LandscapeDetail,
  LandscapeSummary,
  StageEvent,
  StreamDone,
  StreamError,
} from "./types";

export const API_ORIGIN =
  process.env.NEXT_PUBLIC_API_ORIGIN ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
      return String(body.detail[0].msg);
    }
    return JSON.stringify(body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_ORIGIN}${path}`, {
    signal,
    cache: "no-store",
  });
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  return (await response.json()) as T;
}

export const getHealth = (signal?: AbortSignal) =>
  getJson<HealthOut>("/health", signal);

export const listLandscapes = (signal?: AbortSignal) =>
  getJson<LandscapeSummary[]>("/v1/landscapes", signal);

export const getLandscape = (id: number, signal?: AbortSignal) =>
  getJson<LandscapeDetail>(`/v1/landscapes/${id}`, signal);

export async function deleteLandscape(id: number): Promise<void> {
  const response = await fetch(`${API_ORIGIN}/v1/landscapes/${id}`, {
    method: "DELETE",
  });
  if (!response.ok && response.status !== 204) {
    throw new ApiError(await readError(response), response.status);
  }
}

// --------------------------------------------------------------------------- //
// Streaming
// --------------------------------------------------------------------------- //

export type StreamHandlers = {
  onStage: (event: StageEvent) => void;
  onError: (error: StreamError) => void;
  onDone: (done: StreamDone) => void;
};

/**
 * Run the pipeline, invoking `handlers` as each frame arrives.
 *
 * `EventSource` cannot POST a body, which is why this reads a streamed response
 * body instead. The cost is losing the browser's automatic reconnection, which
 * is why the server also persists every stage event for replay.
 */
async function stream(
  path: string,
  init: RequestInit,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_ORIGIN}${path}`, {
    ...init,
    signal,
    headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
  });

  if (!response.ok || !response.body) {
    throw new ApiError(await readError(response), response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line. Keep the trailing partial frame in
    // the buffer: a chunk boundary can land mid-JSON.
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      dispatch(frame, handlers);
    }
  }

  if (buffer.trim()) dispatch(buffer, handlers);
}

function dispatch(frame: string, handlers: StreamHandlers): void {
  let event = "message";
  let data = "";
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;

  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    // A malformed frame is not worth tearing the stream down for.
    return;
  }

  if (event === "stage") handlers.onStage(parsed as StageEvent);
  else if (event === "error") handlers.onError(parsed as StreamError);
  else if (event === "done" || event === "expanded") {
    handlers.onDone(parsed as StreamDone);
  }
}

export function streamLandscape(
  topic: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  return stream(
    "/v1/landscapes/stream",
    { method: "POST", body: JSON.stringify({ topic }) },
    handlers,
    signal,
  );
}

export function expandLandscape(
  id: number,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  return stream(
    `/v1/landscapes/${id}/expand`,
    { method: "POST", body: JSON.stringify({}) },
    handlers,
    signal,
  );
}
