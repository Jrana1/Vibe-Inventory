import { useEffect, useState } from "react";
import { Activity, ArrowLeft, ExternalLink, RefreshCw } from "lucide-react";
import { request } from "./api";
import { agentNames, snapshotValue } from "./ai-types";
import type { AgentRun, AIStatus, HistoryItem, Stage } from "./ai-types";
import { Markdown } from "./Markdown";
import { RouterComparison } from "./RouterComparison";
import "./copilot.css";
import "./traces.css";

const duration = (ms: number) => ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
const tokens = (runs: HistoryItem[]) => runs.flatMap(run => run.usage);

function Bars({ title, items, unit = "" }: Readonly<{
  title: string; items: { label: string; value: number }[]; unit?: string;
}>) {
  const maximum = Math.max(1, ...items.map(item => item.value));
  return <figure className="trace-chart" aria-label={title}><figcaption>{title}</figcaption>
    {items.map(item => <div className="trace-bar" key={item.label}>
      <span>{item.label}</span><div className="trace-track"><div style={{ width: `${item.value / maximum * 100}%` }} /></div>
      <strong>{Math.round(item.value).toLocaleString()}{unit}</strong>
    </div>)}
  </figure>;
}

function Timeline({ stages, total }: Readonly<{ stages: Stage[]; total: number }>) {
  const maximum = Math.max(1, total, ...stages.map(stage => stage.offset_ms + stage.duration_ms));
  return <div className="trace-timeline" role="img" aria-label="Agent execution timeline">
    {stages.filter(stage => stage.name !== "inventory_assistant").map((stage, index) =>
      <div className="trace-stage" key={index}>
        <span>{stage.name.replace(/_/g, " ")}</span>
        <div className="trace-track"><div className={stage.status === "ERROR" ? "failed" : ""}
          style={{ marginLeft: `${stage.offset_ms / maximum * 100}%`, width: `${Math.max(.5, stage.duration_ms / maximum * 100)}%` }} /></div>
        <strong>{duration(stage.duration_ms)}</strong>
      </div>)}
  </div>;
}

export function TracePage() {
  const [runs, setRuns] = useState<HistoryItem[]>([]);
  const [selected, setSelected] = useState(new URLSearchParams(location.search).get("run") || "");
  const [detail, setDetail] = useState<AgentRun | null>(null);
  const [status, setStatus] = useState<AIStatus | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    const abort = new AbortController();
    setLoading(true); setError("");
    void Promise.all([
      request<HistoryItem[]>("/ai/runs", { signal: abort.signal }),
      request<AIStatus>("/ai/status", { signal: abort.signal }),
    ]).then(([nextRuns, nextStatus]) => {
      setRuns(nextRuns); setStatus(nextStatus);
      setSelected(current => current || nextRuns[0]?.id || "");
    }).catch(err => {
      if (!abort.signal.aborted) setError(err instanceof Error ? err.message : "Unable to load trace history");
    }).finally(() => { if (!abort.signal.aborted) setLoading(false); });
    return () => abort.abort();
  }, [revision]);

  useEffect(() => {
    if (!selected) return;
    const abort = new AbortController();
    setDetail(null); setDetailLoading(true); setError("");
    history.replaceState(null, "", `/ai/traces?run=${encodeURIComponent(selected)}`);
    void request<AgentRun>(`/ai/runs/${encodeURIComponent(selected)}`, { signal: abort.signal })
      .then(setDetail)
      .catch(err => { if (!abort.signal.aborted) setError(err instanceof Error ? err.message : "Unable to load this trace"); })
      .finally(() => { if (!abort.signal.aborted) setDetailLoading(false); });
    return () => abort.abort();
  }, [selected, revision]);

  const completed = runs.filter(run => run.status === "completed");
  const measured = tokens(runs);
  const reported = measured.filter(usage => usage.input_tokens !== null && usage.output_tokens !== null);
  const totalTokens = reported.reduce((sum, usage) => sum + (usage.input_tokens ?? 0) + (usage.output_tokens ?? 0), 0);
  const average = runs.length ? runs.reduce((sum, run) => sum + run.latency_ms, 0) / runs.length : 0;
  const recent = [...runs].reverse();
  const maxLatency = Math.max(1, ...recent.map(run => run.latency_ms));
  return <div className="trace-page">
    <header className="trace-topbar"><a href="/"><ArrowLeft size={16} /> Back to inventory</a>
      <span><Activity size={17} /> Vibe Inventory / AI observability</span>
      <button className="agent-secondary" disabled={loading} onClick={() => setRevision(value => value + 1)}>
        <RefreshCw size={14} /> Refresh traces</button>
    </header>
    <main className="trace-content">
      <div className="trace-heading"><div><p className="eyebrow">FOLLOW THE DECISION</p><h1>Agent traceability</h1></div>
        <a className="agent-secondary" href="https://smith.langchain.com/" target="_blank" rel="noopener noreferrer">
          Open LangSmith <ExternalLink size={13} /></a>
      </div>
      {error && <p role="alert" className="ai-error">{error}</p>}
      {loading && <p role="status">Loading traces...</p>}
      <section className="trace-kpis" aria-label="AI performance summary">
        {[["Saved requests", runs.length], ["Agent completed", `${completed.length}/${runs.length}`],
          ["Average total time", duration(average)], ["Reported tokens", totalTokens.toLocaleString()]].map(([label, value]) =>
          <div key={label}><span>{label}</span><strong>{value}</strong></div>)}
      </section>
      <p className="trace-note">Completion means agent validation succeeded; explanation failures are shown separately.
        Tokens cover {reported.length} of {measured.length} provider calls with complete usage reports. Unreported usage is unknown, not zero.</p>
      <section className="trace-charts" aria-label="Trace charts">
        <RouterComparison revision={revision} />
        <Bars title="Requests by agent" items={Object.entries(agentNames).map(([agent, label]) => ({
          label, value: runs.filter(run => run.agent === agent).length,
        })).concat([{ label: "Clarification / unassigned", value: runs.filter(run => !run.agent).length }])} />
        <figure className="trace-chart"><figcaption>Total latency by request</figcaption>
          {recent.length ? <><svg viewBox="0 0 520 140" role="img" aria-label="Recent request latency chart">
            <line x1="10" y1="120" x2="510" y2="120" stroke="#dce6dc" />
            <polyline fill="none" stroke="#2f7b5c" strokeWidth="2.5" points={recent.map((run, index) =>
              `${10 + index / Math.max(1, recent.length - 1) * 500},${120 - run.latency_ms / maxLatency * 105}`).join(" ")} />
            {recent.map((run, index) => <circle key={run.id} r="3" fill={run.status === "failed" ? "#df6e52" : "#2f7b5c"}
              cx={10 + index / Math.max(1, recent.length - 1) * 500} cy={120 - run.latency_ms / maxLatency * 105}>
              <title>{run.id}: {duration(run.latency_ms)}</title></circle>)}
          </svg><div className="trace-chart-axis"><span>Older requests</span><span>Peak {duration(maxLatency)}</span><span>Newest</span></div></>
            : <p>No requests yet.</p>}
        </figure>
      </section>
      <div className="trace-workspace">
        <nav className="trace-history" aria-label="Saved AI requests"><h2>Request history</h2>
          {runs.length === 0 && !loading && <p>No saved requests yet. Ask the Inventory Agent to create a trace.</p>}
          {runs.map(run => <button key={run.id} className={selected === run.id ? "selected" : ""}
            aria-current={selected === run.id ? "true" : undefined} onClick={() => setSelected(run.id)}>
            <strong>{run.agent ? agentNames[run.agent] : "Clarification"}</strong>
            <span>{new Date(run.created_at * 1000).toLocaleString()}</span>
            <span>{run.status} / {duration(run.latency_ms)}</span>
            {run.explanation_status && ["failed", "interrupted"].includes(run.explanation_status) &&
              <span className="trace-failed">Explanation {run.explanation_status}</span>}
          </button>)}
        </nav>
        <section className="trace-detail" aria-label="Selected request trace">
          {detailLoading && <p role="status">Loading request trace...</p>}
          {!detail && !detailLoading && <p>Select a request to inspect its trace.</p>}
          {detail && <>
            <div className="trace-detail-heading"><h2>{detail.agent ? agentNames[detail.agent] : "Clarification"}</h2>
              <span className="trace-status">{detail.status}</span></div>
            <p className="trace-question">{detail.request}</p>
            <dl className="trace-identifiers">
              <dt>Audit request</dt><dd>{detail.id}</dd>
              <dt>MLflow trace</dt><dd>{detail.trace_id}</dd>
              <dt>LangSmith run</dt><dd>{detail.observability?.langsmith_run_id || (
                status?.observability.langsmith_enabled ? "Not exported; check configuration and warnings" : "Disabled"
              )}</dd>
              <dt>Explanation</dt><dd>{detail.explanation_status || "Not requested (earlier request)"}</dd>
              <dt>Previous turn</dt><dd>{detail.parent_run_id || "New conversation"}</dd>
              <dt>Memory turns</dt><dd>{detail.memory?.run_ids.length ?? 0}</dd>
              <dt>Prior verified items</dt><dd>{detail.memory?.item_ids.join(", ") || "None / unresolved"}</dd>
            </dl>
            {detail.observability?.warnings.map(warning => <p className="ai-warning" key={warning}>{warning}</p>)}
            <section><h3>Execution timeline</h3>
              {detail.stages?.length ? <Timeline stages={detail.stages} total={detail.latency_ms} />
                : <p>Stage timings were not recorded for this earlier request.</p>}
            </section>
            {detail.routing && <Bars title={`Jev routing / ${Math.round(detail.routing.confidence * 100)}% selected confidence`}
              items={Object.entries(detail.routing.probabilities).map(([label, value]) => ({ label, value: value * 100 }))} unit="%" />}
            <section><h3>Provider calls and reported usage</h3>
              <div className="ai-table-scroll"><table aria-label="Provider call measurements">
                <thead><tr><th>Provider / model</th><th>Time</th><th>Attempts</th><th>Input tokens</th><th>Output tokens</th></tr></thead>
                <tbody>{detail.usage.map((usage, index) => <tr key={index}>
                  <td>{usage.provider}<small>{usage.model}</small></td><td>{duration(usage.latency_ms)}</td><td>{usage.attempts}</td>
                  <td>{usage.input_tokens ?? "Unknown"}</td><td>{usage.output_tokens ?? "Unknown"}</td>
                </tr>)}</tbody>
              </table></div>
            </section>
            <section><h3>Validated answer</h3><Markdown>{detail.answer}</Markdown>
              {detail.explanation && <><h3>Streamed explanation</h3><Markdown>{detail.explanation}</Markdown></>}
              {detail.explanation_error && <p className="ai-warning">{detail.explanation_error}</p>}
            </section>
            {detail.sql && <section><h3>Validated read-only SQL</h3><pre>{detail.sql}</pre></section>}
            {detail.query && <section><h3>Actual query evidence / {detail.query.row_count} rows</h3>
              {detail.query.truncated && <p className="ai-warning">Truncated at {detail.query.max_rows} rows.</p>}
              <div className="ai-table-scroll"><table aria-label="SQL query evidence">
                <thead><tr>{detail.query.columns.map((column, index) => <th key={index}>{column}</th>)}</tr></thead>
                <tbody>{detail.query.rows.map((row, index) => <tr key={index}>
                  {row.map((cell, column) => <td key={column}>{cell === null ? "NULL" : String(cell)}</td>)}
                </tr>)}</tbody>
              </table></div>
            </section>}
            {detail.faqs.length > 0 && <section><h3>Trusted demo FAQ sources</h3>{detail.faqs.map(faq =>
              <div key={faq.id}><h4>{faq.id} / {faq.question}</h4><p>{faq.answer}</p></div>)}</section>}
            {detail.retrieval && <section><h3>PDF retrieval</h3>
              <dl className="trace-identifiers">
                <dt>Embedding model</dt><dd>{detail.retrieval.embedding_model}</dd>
                <dt>Corpus version</dt><dd>{detail.retrieval.corpus_version}</dd>
                <dt>Retrieved chunks</dt><dd>{detail.retrieval.retrieved_chunks}</dd>
                <dt>Score cutoff</dt><dd>{detail.retrieval.score_threshold} cosine similarity (not confidence)</dd>
              </dl>
              {detail.sources?.map(source => <div key={source.source_id}>
                <h4>[{source.source_id}] {source.document} / page {source.page}</h4>
                {source.quotes.map(quote => <blockquote key={quote}>{quote}</blockquote>)}
              </div>)}
              <details><summary>Retrieved PDF passages and similarity scores</summary>
                {detail.retrieved_sources?.map(source => <div key={source.id}>
                  <h4>[{source.source_id}] {source.document} / page {source.page} / score {source.score.toFixed(3)}</h4>
                  <p>{source.text}</p>
                </div>)}
              </details>
            </section>}
            {detail.proposals.map(proposal => <section key={proposal.id}><h3>{proposal.action.kind === "price_batch" ? "Price" : "Quantity"} decision / {proposal.status}</h3>
              <p>Proposal {proposal.id}</p><p>{proposal.decided_at ? `Decided ${new Date(proposal.decided_at * 1000).toLocaleString()}` : `Expires ${new Date(proposal.expires_at * 1000).toLocaleString()}`}</p>
              <div className="ai-table-scroll"><table aria-label={proposal.action.kind === "price_batch" ? "Price decision evidence" : "Quantity decision evidence"}>
                <thead><tr><th>Item</th><th>Before</th><th>Proposed</th></tr></thead>
                <tbody>{proposal.before.map((row, index) => <tr key={row.item_id}>
                  <td>#{row.item_id} {row.item_name}</td><td>{snapshotValue(row)}</td><td>{snapshotValue(proposal.after[index])}</td>
                </tr>)}</tbody>
              </table></div>
              {proposal.trace_id && <p>Decision MLflow trace: {proposal.trace_id}</p>}
            </section>)}
          </>}
        </section>
      </div>
    </main>
  </div>;
}
