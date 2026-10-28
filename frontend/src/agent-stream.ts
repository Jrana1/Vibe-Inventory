import type { AgentEvent } from "./ai-types";

export async function readAgentStream(response: Response, onEvent: (event: AgentEvent) => void) {
  if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new Error("The server did not return an answer stream");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer = (buffer + decoder.decode(value, { stream: !done })).replace(/\r\n/g, "\n");
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const eventName = frame.split("\n").find(line => line.startsWith("event:"))?.slice(6).trim();
        const data = frame.split("\n").filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
        if (!eventName || !data) continue;
        if (!["progress", "warning", "error", "delta", "validated", "done"].includes(eventName)) {
          throw new Error("Unknown assistant stream event");
        }
        const event = { event: eventName, data: JSON.parse(data) } as AgentEvent;
        if (event.event === "error") throw new Error(event.data.message);
        onEvent(event);
        if (event.event === "done") return;
      }
      if (done) throw new Error("The response stream ended early. Check traceability before retrying.");
    }
  } finally {
    try { await reader.cancel(); }
    finally { reader.releaseLock(); }
  }
}
