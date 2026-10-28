export type Provider = { name: string; model: string; configured: boolean };
export type AIStatus = {
  enabled: boolean; confidence_threshold: number; providers: Provider[];
  observability: { mlflow: boolean; langsmith_enabled: boolean; langsmith_configured: boolean };
  knowledge: { configured: boolean; embedding_model: string; documents: string[] };
};
export type DocumentSource = {
  source_id: string; document: string; page: number; score: number; quotes: string[];
};
export type SuggestedQuestion = { agent: AgentName; question: string };
export type Snapshot = { item_id: number; item_name: string } & (
  { stock_quantity: number } | { price: number | null }
);
export function snapshotValue(snapshot: Snapshot): string {
  return "price" in snapshot
    ? snapshot.price === null ? "Unpriced" : `$${snapshot.price.toFixed(2)}`
    : String(snapshot.stock_quantity);
}
export type Proposal = {
  action: { kind: "quantity_batch" | "price_batch" };
  id: string; status: string; expires_at: number; before: Snapshot[]; after: Snapshot[];
  decided_at?: number | null; trace_id?: string | null;
};
export type Usage = {
  provider: string; model: string; latency_ms: number; attempts: number;
  input_tokens: number | null; output_tokens: number | null;
};
export type AgentName = "sql" | "faq" | "quantity" | "price";
export type Stage = {
  name: string; offset_ms: number; duration_ms: number; status: string;
  attributes: Record<string, string | number | boolean>;
};
export type AgentRun = {
  id: string; trace_id: string; status: string; answer: string; request: string;
  agent: AgentName | null;
  parent_run_id?: string | null;
  comparison_id?: string | null;
  memory?: { run_ids: string[]; item_ids: number[] };
  routing: { agent: string; confidence: number; probabilities: Record<string, number> } | null;
  sql: string | null;
  query: {
    sql: string; columns: string[]; rows: (string | number | null)[][];
    row_count: number; truncated: boolean; max_rows: number;
  } | null;
  faqs: { id: string; question: string; answer: string }[];
  sources?: DocumentSource[];
  retrieved_sources?: {
    source_id: string; document: string; page: number; text: string; score: number; id: string;
  }[];
  retrieval?: {
    corpus_version: string; embedding_model: string; retrieved_chunks: number;
    score_threshold: number; latency_ms: number;
  } | null;
  proposals: Proposal[]; usage: Usage[]; latency_ms: number;
  explanation?: string; explanation_status?: string; explanation_error?: string | null;
  stages?: Stage[];
  observability?: {
    mlflow_trace_id: string; langsmith_run_id: string | null;
    langsmith_enabled: boolean; warnings: string[];
  };
};
export type HistoryItem = {
  id: string; status: string; created_at: number; agent: AgentName | null;
  latency_ms: number; usage: Usage[]; explanation_status: string | null;
  stages?: Pick<Stage, "name" | "duration_ms" | "status">[];
};
export type AgentEvent =
  | { event: "progress" | "warning" | "error"; data: { message: string } }
  | { event: "delta"; data: { text: string } }
  | { event: "validated" | "done"; data: AgentRun };
export const agentNames = { sql: "Text-to-SQL", faq: "PDF knowledge", quantity: "Quantity update", price: "Price setting" };
