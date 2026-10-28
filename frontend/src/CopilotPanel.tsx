import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, Bot, ChevronRight, ExternalLink, RotateCcw, Send, Square, X } from "lucide-react";
import { request, streamAgent } from "./api";
import type { AgentRun, AIStatus, Proposal } from "./ai-types";
import { snapshotValue } from "./ai-types";
import { chooseStarterQuestions } from "./starter-questions";
import { Markdown } from "./Markdown";
import "./copilot.css";

export function CopilotPanel({ offline, onInventoryChanged }: Readonly<{
  offline: boolean; onInventoryChanged: () => Promise<void>;
}>) {
  const [open, setOpen] = useState(false);
  const [width, setWidth] = useState(() => {
    const saved = Number(localStorage.getItem("inventory-agent-width"));
    return Number.isFinite(saved) && saved >= 340 ? saved : 460;
  });
  const [viewport, setViewport] = useState(window.innerWidth);
  const [status, setStatus] = useState<AIStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [message, setMessage] = useState("");
  const [run, setRun] = useState<AgentRun | null>(null);
  const [chatStarted, setChatStarted] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [suggestions] = useState(chooseStarterQuestions);
  const startersUsed = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const parentRunId = useRef<string | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const launcher = useRef<HTMLButtonElement>(null);
  const transcript = useRef<HTMLDivElement>(null);
  const maxWidth = Math.max(300, viewport - 24);
  const minWidth = Math.min(340, maxWidth);
  const actualWidth = Math.max(minWidth, Math.min(width, maxWidth));

  async function refreshStatus() {
    try { setStatus(await request<AIStatus>("/ai/status")); setError(""); }
    catch (err) { setError(err instanceof Error ? err.message : "Unable to load AI status"); }
  }

  useEffect(() => {
    void refreshStatus();
    const resize = () => setViewport(window.innerWidth);
    window.addEventListener("resize", resize);
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => {
      window.removeEventListener("resize", resize); window.clearInterval(timer);
      controller.current?.abort();
    };
  }, []);
  useEffect(() => { if (open) input.current?.focus(); }, [open]);
  useEffect(() => {
    const element = transcript.current;
    if (element && element.scrollHeight - element.scrollTop - element.clientHeight < 200) {
      element.scrollTop = element.scrollHeight;
    }
  }, [run?.explanation, progress]);

  function close() { setOpen(false); requestAnimationFrame(() => launcher.current?.focus()); }
  function newChat() {
    if (busy) return;
    parentRunId.current = null;
    setRun(null); setMessage(""); setError(""); setProgress("");
    setChatStarted(false);
    input.current?.focus();
  }
  function resize(next: number) {
    const size = Math.round(Math.max(minWidth, Math.min(next, maxWidth)));
    setWidth(size); localStorage.setItem("inventory-agent-width", String(size));
  }
  async function send() {
    if (!message.trim() || busy) return;
    const submittedMessage = message.trim();
    setMessage("");
    setChatStarted(true);
    startersUsed.current = true;
    const abort = new AbortController();
    controller.current = abort;
    setOpen(true); setBusy(true); setError(""); setRun(null);
    setProgress("Connecting to the inventory agent...");
    try {
      await streamAgent(submittedMessage, abort.signal, event => {
        if (event.event === "progress") setProgress(event.data.message);
        else if (event.event === "warning") setError(event.data.message);
        else if (event.event === "validated" || event.event === "done") {
          parentRunId.current = event.data.id;
          setRun(event.data);
        }
        else if (event.event === "delta") setRun(current => current && ({
          ...current, explanation: (current.explanation || "") + event.data.text,
          explanation_status: "streaming",
        }));
      }, parentRunId.current);
    } catch (err) {
      setError(abort.signal.aborted
        ? "Streaming stopped. Validated results and pending previews remain saved; no inventory was changed."
        : err instanceof Error ? err.message : "Unable to complete your request");
    } finally {
      setBusy(false); setProgress(""); controller.current = null;
    }
  }
  async function decide(id: string, decision: "approve" | "reject") {
    setBusy(true); setError(""); setProgress(decision === "approve" ? "Applying approved changes..." : "Rejecting preview...");
    try {
      const proposal = await request<Proposal>(`/ai/proposals/${id}/${decision}`, { method: "POST" });
      setRun(current => current && ({
        ...current, proposals: current.proposals.map(p => p.id === id ? proposal : p),
      }));
      if (decision === "approve") await onInventoryChanged();
    } catch (err) { setError(err instanceof Error ? err.message : "Unable to save your decision"); }
    finally { setBusy(false); setProgress(""); }
  }

  const unavailable = offline || !status?.enabled || status.providers.some(p => !p.configured);
  const answer = run?.explanation || run?.answer || "";
  const starterQuestions = <>
    {suggestions.map(example =>
      <button key={example.question} className="agent-suggestion" onClick={() => { setMessage(example.question); input.current?.focus(); }}>
        {example.question}<ChevronRight size={16} />
      </button>)}
  </>;
  return <>
    <div className={`agent-launcher ${open ? "is-hidden" : ""}`}>
      <button ref={launcher} className="agent-launch-title" aria-expanded={open} aria-controls="inventory-agent-drawer"
        onClick={() => setOpen(true)}><span className="agent-orb"><Bot size={19} /></span>
        <span>Ask Inventory Agent<small>Your warehouse copilot</small></span><ArrowUpRight size={18} />
      </button>
      <input aria-label="Quick inventory question" placeholder="Type a question..." maxLength={2000}
        value={message} onChange={event => setMessage(event.target.value)} onFocus={() => setOpen(true)}
        onKeyDown={event => { if (event.key === "Enter" && !unavailable && !busy) void send(); }} />
    </div>
    {open && <aside id="inventory-agent-drawer" className="agent-drawer" aria-label="Inventory AI assistant"
      style={{ width: actualWidth }} onKeyDown={event => { if (event.key === "Escape") close(); }}>
      <div className="agent-resizer" role="separator" tabIndex={0} aria-label="Resize inventory assistant"
        aria-orientation="vertical" aria-valuemin={minWidth} aria-valuemax={maxWidth} aria-valuenow={actualWidth}
        onPointerDown={event => { event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); }}
        onPointerMove={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) resize(viewport - event.clientX - (viewport <= 600 ? 6 : 12)); }}
        onPointerUp={event => event.currentTarget.releasePointerCapture(event.pointerId)}
        onKeyDown={event => {
          if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
            event.preventDefault(); resize(actualWidth + (event.key === "ArrowLeft" ? 32 : -32));
          } else if (event.key === "Home") resize(minWidth);
          else if (event.key === "End") resize(maxWidth);
        }}><span /></div>
      <header className="agent-header">
        <span className="agent-orb"><Bot size={21} /></span>
        <div><h2>Inventory Agent</h2></div>
        <button className="agent-icon agent-reset" aria-label="New chat" title="New chat: clear conversation memory"
          disabled={busy || !chatStarted} onClick={newChat}><RotateCcw size={17} /></button>
        <button className="agent-icon" aria-label="Minimize inventory assistant" onClick={close}><X size={19} /></button>
      </header>
      <div className="agent-transcript" ref={transcript}>
        {!chatStarted && !busy && <div className="agent-welcome">
          <span className="agent-welcome-icon"><Bot size={30} /></span>
          {!startersUsed.current && starterQuestions}
          {status?.knowledge && !status.knowledge.configured &&
            <p className="ai-warning">PDF knowledge needs Qdrant configuration and indexing. Live inventory, quantity and price features are still available.</p>}
        </div>}
        {run && <article className="agent-answer" aria-label="Agent response">
          <Markdown>{answer}</Markdown>
          {run.sources && run.sources.length > 0 && <div className="agent-sources" aria-label="Document citations">
            <span>PDF sources</span>
            {run.sources.map(source => <small key={source.source_id}>[{source.source_id}] {source.document} / page {source.page}</small>)}
          </div>}
          {(run.explanation_status === "failed" || run.explanation_status === "interrupted") &&
            <p className="ai-warning">Explanation incomplete. {run.explanation_error} The validated answer: {run.answer}</p>}
          {run.query?.truncated && <p className="ai-warning">Results are limited to {run.query.max_rows} rows. Narrow your question for a complete result.</p>}
          {run.proposals.map(proposal => <div className="ai-proposal" key={proposal.id}>
            <h3>{proposal.action.kind === "price_batch" ? "Price" : "Quantity"} preview / {proposal.status}</h3>
            {proposal.status === "pending" && <p>Nothing has changed yet. Review every item before approving.</p>}
            {proposal.status === "approved" && <p>These changes were applied successfully.</p>}
            {proposal.status === "rejected" && <p>This preview was rejected. No inventory was changed.</p>}
            <div className="ai-table-scroll"><table aria-label={proposal.action.kind === "price_batch" ? "Price change preview" : "Quantity change preview"}>
              <thead><tr><th>Item</th><th>Current</th><th>New</th></tr></thead>
              <tbody>{proposal.before.map((before, index) => <tr key={before.item_id}>
                <td>#{before.item_id} {before.item_name}</td><td>{snapshotValue(before)}</td>
                <td>{snapshotValue(proposal.after[index])}</td>
              </tr>)}</tbody>
            </table></div>
            {proposal.status === "pending" && <>
              <p>{proposal.expires_at * 1000 <= now ? "Preview expired. Submit a fresh request." : `Expires ${new Date(proposal.expires_at * 1000).toLocaleTimeString()}`}</p>
              <div className="ai-actions"><button className="agent-send" disabled={busy || offline || proposal.expires_at * 1000 <= now}
                onClick={() => void decide(proposal.id, "approve")}>Approve all changes</button>
                <button className="agent-secondary" disabled={busy || offline} onClick={() => void decide(proposal.id, "reject")}>Reject</button>
              </div>
            </>}
          </div>)}
        </article>}
        {busy && <p className="agent-progress" role="status"><span className="agent-pulse" />{progress}</p>}
        {error && <p className="ai-error" role="alert">{error}</p>}
        {offline && <p className="ai-warning">Offline. Requests and approvals are unavailable.</p>}
        {status && !status.enabled && <p className="ai-warning">AI is disabled. Enable AI in your local environment and restart the backend.</p>}
        {status?.enabled && status.providers.some(p => !p.configured) && <p className="ai-warning">A provider key is missing. Check your local configuration.</p>}
        {!status && <button className="agent-secondary" onClick={() => void refreshStatus()}>Retry connection</button>}
      </div>
      <footer className="agent-composer">
        <form className={busy ? "is-processing" : undefined} aria-busy={busy}
          onSubmit={event => { event.preventDefault(); if (!unavailable) void send(); }}>
          <label className="sr-only" htmlFor="ai-message">Your question or inventory-update request</label>
          <textarea ref={input} id="ai-message" maxLength={2000} value={message}
            onChange={event => setMessage(event.target.value)} placeholder="Ask about your inventory..."
            disabled={busy || unavailable} rows={2} onKeyDown={event => {
              if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); if (!unavailable) void send(); }
            }} />
          {busy && controller.current
            ? <button type="button" className="agent-icon" aria-label="Stop streaming" onClick={() => controller.current?.abort()}><Square size={16} /></button>
            : <button className="agent-send" aria-label="Send request" disabled={busy || unavailable || !message.trim()}><Send size={18} /></button>}
        </form>
        <div className="agent-footer-links"><small>AI can make mistakes. Verify before approving.</small>
          <a href={`/ai/traces${run ? `?run=${encodeURIComponent(run.id)}` : ""}`} target="_blank" rel="noopener noreferrer">
            Traceability <ExternalLink size={12} /></a>
        </div>
      </footer>
    </aside>}
  </>;
}
