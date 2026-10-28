import assert from "node:assert/strict";
import test from "node:test";
import { readAgentStream } from "../src/agent-stream.ts";

const encoder = new TextEncoder();
const frame = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
const response = body => new Response(body, { headers: { "Content-Type": "text/event-stream" } });

test("decodes split UTF-8, split CRLF, and keepalives without losing Markdown", async () => {
  const wire = ": keepalive\n\n" + frame("delta", { text: "## Résultat\n\n**5** units" }) + frame("done", { id: "run" });
  const bytes = encoder.encode(wire.replaceAll("\n", "\r\n"));
  const stream = new ReadableStream({
    start(controller) {
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      controller.close();
    },
  });
  const events = [];
  await readAgentStream(response(stream), event => events.push(event));
  assert.deepEqual(events.map(event => event.event), ["delta", "done"]);
  assert.equal(events[0].data.text, "## Résultat\n\n**5** units");
});

test("delivers deltas before the response finishes and cancels after terminal done", async () => {
  let producer;
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) { producer = controller; },
    cancel() { cancelled = true; },
  });
  const events = [];
  let firstDelta;
  const arrived = new Promise(resolve => { firstDelta = resolve; });
  const reading = readAgentStream(response(stream), event => {
    events.push(event);
    if (event.event === "delta") firstDelta();
  });
  producer.enqueue(encoder.encode(frame("delta", { text: "Partial answer" })));
  await arrived;
  assert.equal(events.length, 1);
  producer.enqueue(encoder.encode(frame("done", { id: "run" })));
  await reading;
  assert.equal(events.length, 2);
  assert.equal(cancelled, true);
});

test("truncated connections fail rather than claiming completion", async () => {
  await assert.rejects(readAgentStream(response(frame("delta", { text: "Partial" })), () => {}), /ended early/);
});

test("server errors are surfaced explicitly", async () => {
  await assert.rejects(readAgentStream(response(frame("error", { message: "Provider unavailable" })), () => {}), /Provider unavailable/);
});

test("rejects invalid event types and incorrect response media types", async () => {
  await assert.rejects(readAgentStream(response(frame("invalid", {})), () => {}), /Unknown assistant stream event/);
  await assert.rejects(readAgentStream(new Response("{}"), () => {}), /did not return an answer stream/);
});

test("aborted streams propagate failure and release the reader lock", async () => {
  const stream = new ReadableStream({ start(controller) { controller.error(new DOMException("Stopped", "AbortError")); } });
  await assert.rejects(readAgentStream(response(stream), () => {}), { name: "AbortError" });
  assert.equal(stream.locked, false);
});
