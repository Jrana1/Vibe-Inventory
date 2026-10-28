import { useEffect, useState } from "react";
import { request } from "./api";

type Outcome = {
  provider: string; model: string; status: "ok" | "error"; error: string | null;
  latency_ms: number; agent: string | null; confidence: number | null;
  input_tokens: number | null; output_tokens: number | null; attempts: number;
};
type Pair = {
  index: number; question?: string; expected: string | null; jev: Outcome; mistral: Outcome;
  run_id?: string | null;
};
type ProviderMetrics = {
  attempted: number; errors: number; paired_samples: number;
  mean_ms: number | null; median_ms: number | null; p95_ms: number | null;
  token_samples: number; reported_token_samples: number; mean_input_tokens: number | null; mean_output_tokens: number | null;
};
type Metrics = {
  providers: Record<"jev" | "mistral", ProviderMetrics>;
  paired_samples: number; agreements: number; agreement_rate: number | null;
};
type Comparison = {
  id: string; created_at: number; mode: string; status: string; completed: number; total: number;
  pairs: Pair[]; metrics: Metrics; error: string | null; models: Record<string, string>;
  run_id: string | null; trace_id: string | null;
};
type Dashboard = { live: Metrics; benchmarks: Comparison[]; live_results: Comparison[] };
const time = (ms: number | null) => ms === null ? "No data" : `${ms.toFixed(0)} ms`;
const number = (value: number | null) => value === null ? "Unknown" : value.toFixed(1);
const providers = ["jev", "mistral"] as const;

export function RouterComparison({ revision }: Readonly<{ revision: number }>) {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [selected, setSelected] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [metric, setMetric] = useState<"mean_ms" | "median_ms" | "p95_ms">("mean_ms");

  useEffect(() => {
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function load() {
      setLoading(true);
      try {
        const next = await request<Dashboard>("/ai/comparisons", { signal: abort.signal });
        if (abort.signal.aborted) return;
        setDashboard(next); setError("");
        setSelected(current => current || next.benchmarks[0]?.id || "live");
        if (next.benchmarks.some(item => item.status === "running")) {
          timer = setTimeout(() => { void load(); }, 2000);
        }
      } catch (err) {
        if (!abort.signal.aborted) setError(err instanceof Error ? err.message : "Unable to load router comparison");
      } finally {
        if (!abort.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => { abort.abort(); if (timer !== undefined) clearTimeout(timer); };
  }, [revision, refresh]);

  async function start() {
    if (starting || dashboard?.benchmarks.some(item => item.status === "running")) return;
    setStarting(true); setError("");
    try {
      const created = await request<Comparison>("/ai/comparisons/benchmark", { method: "POST" });
      setSelected(created.id);
      setDashboard(current => current && ({ ...current, benchmarks: [created, ...current.benchmarks] }));
      setRefresh(value => value + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start router benchmark");
    } finally { setStarting(false); }
  }

  const benchmark = dashboard?.benchmarks.find(item => item.id === selected);
  const metrics = selected === "live" ? dashboard?.live : benchmark?.metrics;
  const pairs = selected === "live"
    ? dashboard?.live_results.flatMap(item => item.pairs.map(row => ({ ...row, run_id: item.run_id }))) ?? []
    : benchmark?.pairs ?? [];
  const maximum = Math.max(1, ...providers.map(provider => metrics?.providers[provider][metric] ?? 0));
  const running = dashboard?.benchmarks.some(item => item.status === "running");
  return <section className="trace-chart trace-agent-speed" aria-label="Jev versus Mistral routing comparison">
    <div className="router-comparison-heading"><h2>Jev vs Mistral / same routing task</h2>
      <button className="agent-secondary" disabled={starting || running || !dashboard}
        onClick={() => void start()}>{starting ? "Starting..." : running ? "Benchmark running..." : "Run 20-case benchmark (40 paid calls)"}</button>
    </div>
    <div className="router-comparison-controls">
      <label>Results <select value={selected} onChange={event => setSelected(event.target.value)}>
        <option value="live">Live shadow comparisons</option>
        {dashboard?.benchmarks.map(item => <option key={item.id} value={item.id}>
          {new Date(item.created_at * 1000).toLocaleString()} / {item.status} / {item.completed}/{item.total}
        </option>)}
      </select></label>
      <label>Bar metric <select value={metric} onChange={event => setMetric(event.target.value === "p95_ms" ? "p95_ms" : event.target.value === "median_ms" ? "median_ms" : "mean_ms")}>
        <option value="mean_ms">Mean latency</option><option value="median_ms">Median latency</option><option value="p95_ms">P95 latency</option>
      </select></label>
    </div>
    {error && <p className="ai-error" role="alert">{error} <button className="agent-secondary"
      onClick={() => setRefresh(value => value + 1)}>Retry</button></p>}
    {benchmark?.error && <p className="ai-warning">{benchmark.error}</p>}
    {benchmark?.status === "running" && <p role="status">Completed {benchmark.completed} of {benchmark.total} paired evaluations. You can leave this page; do not restart the backend.</p>}
    {loading && !dashboard && <p role="status">Loading comparisons...</p>}
    {metrics && <>
      {providers.map(provider => <div className="trace-bar" key={provider}>
        <span>{provider === "jev" ? "Jev" : "Mistral"}<small>{metrics.providers[provider].paired_samples} matched samples</small></span>
        <div className="trace-track"><div style={{ width: `${(metrics.providers[provider][metric] ?? 0) / maximum * 100}%` }} /></div>
        <strong>{time(metrics.providers[provider][metric])}</strong>
      </div>)}
      <div className="ai-table-scroll router-comparison-table"><table aria-label="Router performance metrics">
        <thead><tr><th>Router</th><th>Mean</th><th>Median</th><th>P95</th><th>Mean input tokens</th><th>Mean output tokens</th><th>Token reports</th><th>Errors / calls</th></tr></thead>
        <tbody>{providers.map(provider => {
          const value = metrics.providers[provider];
          return <tr key={provider}><td>{provider === "jev" ? "Jev" : "Mistral"}
            <small>{pairs[0]?.[provider].model ?? benchmark?.models[provider] ?? "No model data"}</small></td>
            <td>{time(value.mean_ms)}</td><td>{time(value.median_ms)}</td><td>{time(value.p95_ms)}</td>
            <td>{number(value.mean_input_tokens)}</td><td>{number(value.mean_output_tokens)}</td>
            <td>{value.reported_token_samples}/{value.paired_samples}</td><td>{value.errors}/{value.attempted}</td></tr>;
        })}</tbody>
      </table></div>
      {metrics.paired_samples === 0 && <p>No matched results yet. Run the benchmark or select saved results.</p>}
    </>}
    {pairs.length > 0 && <details><summary>Individual routing results and errors ({pairs.length} pairs)</summary>
      <div className="ai-table-scroll"><table aria-label="Paired router results">
        <thead><tr><th>Input / expected</th><th>Jev</th><th>Mistral</th><th>Agreement</th></tr></thead>
        <tbody>{pairs.map((row, index) => <tr key={index}><td>{row.question || `Live request ${index + 1}`}
          {row.run_id && <small>Run: {row.run_id}</small>}
          {row.expected && <small>Expected: {row.expected}</small>}</td>
          {providers.map(provider => <td key={provider}><strong>{row[provider].agent || "Error"}</strong>
            <small>{row[provider].model} / {time(row[provider].latency_ms)}</small>
            <small>Input/output: {row[provider].input_tokens ?? "Unknown"} / {row[provider].output_tokens ?? "Unknown"}</small>
            {row[provider].error && <p className="ai-error">{row[provider].error}</p>}
          </td>)}
          <td>{row.jev.status !== "ok" || row.mistral.status !== "ok" ? "Not comparable" : row.jev.agent === row.mistral.agent ? "Agree" : "Disagree"}</td>
        </tr>)}</tbody>
      </table></div>
    </details>}
  </section>;
}
