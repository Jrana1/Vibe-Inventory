import type { AgentEvent } from "./ai-types";
import { readAgentStream } from "./agent-stream";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

export async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!response.ok) {
    const body = await response.json();
    throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? "Request failed"));
  }
  return response.json();
}

export async function streamAgent(
  message: string, signal: AbortSignal, onEvent: (event: AgentEvent) => void,
  parentRunId: string | null = null,
  compareRouting = false,
) {
  const response = await fetch(`${API}/ai/chat/stream`, {
    method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ message, parent_run_id: parentRunId, compare_routing: compareRouting }), signal,
  });
  if (!response.ok) {
    const body = await response.json();
    throw new Error(typeof body.detail === "string" ? body.detail : "Unable to start the assistant");
  }
  await readAgentStream(response, onEvent);
}
