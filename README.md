# Vibe Inventory

Project identifiers use `vibe-inventory`, the inventory database is
`vibe_inventory.db`, and the UI brand mark is **VI**. Historical audit/trace records
and archived submission contents are retained as originally recorded; the renamed
submission ZIP is an earlier snapshot, not a newly generated release.

A full-stack inventory management application built for the Vibe Inventory interview challenge.

## Requirements

- Python 3.10+
- Node.js 20.19+
- npm
- Git Bash on Windows, or Bash on macOS/Linux/WSL
- The supplied `vibe_inventory.db` file in the repository root

## Run Locally

Use two terminals from the repository root.

### 1. Backend

Run the setup script once:

```bash
bash backend/setup.sh
```

On Windows Git Bash, activate the virtual environment and start the backend:

```bash
source .venv/Scripts/activate
python -m uvicorn backend.main:app --reload --port 8000
```

On macOS/Linux/WSL:

```bash
./.venv/bin/python -m uvicorn backend.main:app --reload --port 8000
```

Backend URLs:

- API: http://localhost:8000
- Health check: http://localhost:8000/api/health
- Swagger documentation: http://localhost:8000/docs

### 2. Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

Vite requires port 5173 (`strictPort: true`); it will not fall back to 5174.
If the port is occupied, stop the existing frontend before starting another.
The backend permits browser origins on port 5173 only, so other frontend ports
will fail CORS preflight checks.

The frontend expects the backend at `http://localhost:8000/api`.

## Query the Database in VS Code

Install the **SQLite** extension by Alex Covizzi, then use **SQLite: Open Database**
from the Command Palette to select `vibe_inventory.db` in the project root (not in `backend`).
Open [inventory_queries.sql](inventory_queries.sql), select a query, and run
**SQLite: Run Selected Query**. The supplied queries only read data.

## Configuration

### Optional Inventory AI Lab

The AI assistant is a **local-only showcase**, not a publicly deployable multi-user
service. It has one Jev router and four Mistral agents: read-only Text-to-SQL,
PDF-grounded RAG answers, single/bulk quantity updates, and absolute price setting.
MLflow traces routing and
execution. The existing inventory dashboard and manual editing are unchanged.
The optional AI feature requires **Python 3.11+** for SQLite resource limits.

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements-ai.txt
if (-not (Test-Path .\.env)) { Copy-Item .\.env.example .\.env }
```

Edit `.env` locally: set fresh `TYPESAFE_API_KEY` and
`MISTRAL_API_KEY` values, then set `AI_ENABLED=true`. Never paste keys into chat,
commit them, or put them in frontend `VITE_*` variables. Revoke any exposed keys.
Environment variables take precedence over `.env`; restart the backend after changes.
The status panel reports whether keys are configured, not whether they are valid.

Start the backend bound to loopback:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Start the frontend normally and use **Ask Inventory Agent** at the bottom-right.
Click the launcher or its small input to open the right-hand assistant drawer. Drag
its left edge to resize in either direction; keyboard users can focus the resize
separator and use Left/Right (32px steps), Home (minimum), or End (maximum).
The width is remembered locally. Escape or the close button minimizes the drawer
without cancelling an active answer. The Stop button cancels response streaming.
Use `localhost:5173` or `127.0.0.1:5173`. AI endpoints reject non-loopback clients,
unexpected Host headers, and untrusted browser origins. Do not expose the backend
through a reverse proxy or public tunnel: this is not authentication, and the
existing non-AI inventory endpoints remain unauthenticated.

#### Routing and safe actions

1. TypeSafe **Jev** chooses `sql`, `faq`, `quantity`, `price`, or `unsupported`. Its typed
   Choice response supplies probabilities and confidence.
2. Below `JEV_CONFIDENCE_THRESHOLD`, or for unsupported requests, the application
   asks for clarification without calling a generation model.
3. Exactly one **Mistral Large 4** (`mistral-large-4-0`) agent produces a typed
   response. There is no tool-calling loop, automatic SQL repair, or provider fallback.
4. The SQL agent generates and executes one validated read-only query. The FAQ/RAG
   agent retrieves PDF passages from Qdrant and generates citation-checked claims.
   Quantity and price agents prepare separate typed approval previews; neither can
   directly execute SQL. Mixed quantity/price requests require separate turns.
5. Quantity and price previews expire after ten minutes. Approval checks every item under a
   SQLite write lock and applies the entire batch atomically. Missing items, stale
   quantities/prices, invalid values, or execution errors prevent partial updates.
   Approvals are idempotent and rejections never change inventory.
6. The drawer streams a separate Markdown explanation after validation. This is an
   additional paid Mistral call, not an unvalidated token-by-token SQL/action plan.
   Unsupported requests, failed validation, and unknown FAQs skip this extra call.
   Approvals always use the validated server-side preview, never the explanation.

Example requests:

- `What is the stock quantity of item 12?`
- `Summarize inventory value by supplier.`
- `Show all unique suppliers in the inventory table.`
- `Show the product with the maximum unit price.`
- `How does the operations manual handle negative stock quantities?`
- `What supplier-data controls does the procurement policy require?`
- `What issues were identified in the monthly inventory audit report?`
- `Set item 12 stock quantity to 80.`
- `Set item 12 to 80 units and item 7 to 25 units.`
- `Set the price of product ID 379 from NULL to 120.`
- `Set item 12 price to 19.99 and item 7 price to 25.`

Requests support **page-lifetime short-term memory**. The browser keeps only the last
run ID in React memory (not local/session storage). Each request explicitly links to
its parent; the backend loads up to six recent turns, capped at 16,000 serialized
history characters, from the local audit store. Separate tabs and fresh pages start
independent chains. Minimize/reopen preserves context; **New chat** or page refresh
clears the link. Audit records and pending previews are retained, not deleted or
automatically rejected. This is a local demo, not authenticated multi-user isolation.

Recent requests, bounded canonical answers/results, and current proposal decision
statuses go to Jev/Mistral as untrusted context, not new tasks or current stock facts.
Streamed explanations and old PDF quotations are not used as memory evidence.
Contextual FAQ turns add a Mistral call to rewrite a standalone retrieval question;
answers still require fresh Qdrant retrieval and verified PDF citations. Context
uses no extra memory database, embeddings, or long-term user profile.

For example, ask `Show item 12`, then `Increase this product's quantity to 10`.
Quantity and price agents can use **verified items** returned by the previous product
query or update preview. A singular reference (`this product`, `it`) requires one
item. `Both` requires exactly two items; `them`, `these products`, or `those items`
refers to the entire verified group (2-25 items). For example, after looking up two
products, `Increase quantity for both these products to 3` previews a final quantity
of 3 for each, not an increment. A singular reference to several items, a mismatched
`both`, truncated results, aggregate rows, missing IDs, or unresolved product names
requires explicit IDs. Broad requests like `all products` do not implicitly target
the last group. Product-row
queries are instructed to include `item_id`; computed/aggregate aliases are not
accepted as product identity. A new explicit ID overrides prior context. The server
rejects update plans whose IDs differ from the current explicit IDs or the full
referenced session selection; partial or expanded plans never create a preview.

Final non-negative integer quantities are still required, up to 25 items and
1,000,000 units per item. **Increase to 10** means a final quantity of 10;
**increase by 10** is relative and remains unsupported. Old quantities are not
instructions for a new update. Previews read fresh inventory, and approval rechecks
every snapshot. Always verify the preview: an LLM can still misunderstand a request.
API clients opt in using `parent_run_id` on both
chat endpoints; omitting it starts a new conversation.

#### Approval-gated absolute price setting

The price agent can initialize a missing (`NULL`) unit price or replace an existing
price for 1-25 products. It requires an explicit final numeric price in the app's
existing USD unit, from 0 to 1,000,000 with at most two decimal places. Negative,
non-finite, string/boolean, excessive-precision and NULL new prices are rejected;
values are never silently rounded. Percentage/relative changes and currency
conversion are not supported by this agent. Existing manual bulk adjustments are
unchanged and still reject unpriced products.

Previews show **Unpriced** separately from **$0.00**, and approval is always
required. The server rechecks each item's identity/name and current price under
the SQLite write lock before updating any price; a changed price (including NULL
becoming priced), missing item, expired preview or failed write prevents the
entire batch. Price snapshots contain prices rather than quantities, so unrelated
stock edits do not invalidate price approvals. Price decisions, streaming failures
and traces use the existing audit/approval workflow. Restart recovery preserves
valid `price_batch` previews but still rejects removed legacy adjustment actions.

#### Read-only Text-to-SQL

The agent sees the inventory schema, not the database contents. The backend parses
its SQL with SQLGlot and accepts only one inventory-only `SELECT`. Aggregates,
distinct values, grouping, sorting, filters, and inventory subqueries are supported.
CTEs, joins, set operations, schema/audit-table reads, and all write statements are
rejected. SQLite independently enforces read-only file access, `query_only`, a
table/function authorizer, execution limits, and output-size limits.

SQL and the actual result table are available on the separate **Traceability** page.
The drawer shows the AI explanation and any actionable approval preview, without
technical trace clutter. At most 100 result rows are saved/displayed, with an explicit
truncation warning. Query execution has a one-second progress budget and a
one-million-instruction cap; oversized or expensive queries fail explicitly.
Maximum-price requests are instructed to include tied maxima. Missing prices remain
null; the assistant cannot answer sales/history/demand questions from nonexistent data.
SQL validation prevents unsupported operations, but cannot prove that a query
correctly interprets the user's question. Inspect the SQL and results in Traceability.

For the streamed explanation, up to 20 result rows (at most 16,000 serialized
characters of row data) are sent to Mistral, along with column labels and the question.
Quantity explanations receive the pending before/after preview. These are model
inputs, not observability exports. Use only non-sensitive demo inventory. Samples
are marked incomplete; a generated explanation can still make factual mistakes.
The complete validated answer/results remain available in the local trace page.

#### PDF RAG knowledge agent

The hard-coded FAQ has been replaced by retrieval-augmented generation using the
three text-based PDFs in `backend/faq_data`:

- `Inventory_Operations_Manual.pdf`
- `Supplier_and_Procurement_Policy.pdf`
- `Monthly_Inventory_Audit_Report.pdf`

These are fictional demonstration documents, not real business commitments. Only
index public/non-sensitive data: **extracted PDF text is sent to Mistral for embeddings
and stored in Qdrant Cloud**, and retrieved passages go to Mistral for answers.
This is separate from metadata-only MLflow/LangSmith observability.

Configure the project-root `.env` (uppercase variable names; never `VITE_*`):

```dotenv
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=<fresh-key-kept-only-in-local-env>
QDRANT_COLLECTION_ALIAS=vibe-inventory-faq
MISTRAL_EMBEDDING_MODEL=mistral-embed
```

The supplied cluster URL is included in `.env.example`. Rotate keys exposed in chat.
The indexer requires a Qdrant key allowed to create collections, upsert points and
manage aliases. From the project root, explicitly index the documents:

```powershell
.\.venv\Scripts\python.exe -m backend.services.ai_rag index
```

Restart the backend after environment changes. Startup and chat never automatically
upload or reindex documents, so simply opening the application does not create an
embedding bill. After changing, adding, or removing PDFs, rerun the same command.

Indexing uses pypdf, normalizes extracted whitespace, and creates page-local
1,400-character chunks with 200-character overlap. Each chunk retains document name,
1-based PDF page, text, offset, deterministic ID and corpus fingerprint. Embeddings use
Mistral `mistral-embed` (1,024 dimensions); batches contain at most 32 chunks. Vector
dimensions, finite values and provider ordering are validated before upsert.

Each build writes to a new versioned collection, confirms the exact point count,
then atomically switches the application-owned alias. An unsuccessful build never
replaces the active alias. Reindexing an unchanged corpus is a no-op without paid
embeddings. Fingerprints include file bytes/names, model and chunking version; changed
documents refuse retrieval until reindexed, rather than silently serving stale text.
Previous/failed-build collections are not automatically deleted. Review storage and
remove only known obsolete collections in Qdrant when they are no longer in use.
Do not run parallel ingestion commands; alias ownership belongs to this application.

For each document question, embed the question, retrieve up to six passages above
cosine score `0.65`, and verify the returned payloads against the current local PDFs.
The cutoff is a conservative demo heuristic, **not calibrated answer confidence**.
Tune it using a held-out question set before production. Empty retrieval returns an
explicit no-evidence answer, with no generation call or fallback to the removed FAQ.

Mistral generates atomic claims referencing retrieved source IDs and server-numbered
excerpt IDs. The server resolves those references to exact PDF text; the model never
has to reproduce a quote character-for-character. Unknown source/excerpt IDs are
rejected, and displayed quotations come directly from retrieved passages.
This proves provenance, not semantic entailment: the model can still
misinterpret a quote. The subsequent streamed explanation is instructed to preserve
the claims and citation labels; a missing/unknown citation label in its final output
invalidates that explanation and restores the canonical answer. The chat always shows server-verified PDF/page
references; Traceability exposes canonical claims, supporting quotes, retrieved
passages, similarity scores, corpus version and embedding usage. Historical audit
figures are document evidence, never assumed to be today's live inventory.

Only local top-level `.pdf` files are ingested. Limits are 20 MB/200 pages per PDF and
2,000 chunks per corpus. Encrypted and textless PDFs fail explicitly; OCR is not
included. Textless pages are logged/skipped. All three current PDFs are text-based.
Missing configuration, authentication, unavailable services and stale indexes are
explicit errors; SQL and quantity agents do not depend on Qdrant.

#### Instant starter questions

The frontend contains 30 curated prompts, each at most ten words: ten live SQL
questions, ten PDF knowledge questions, and ten quantity-preview requests.
It randomly selects one per original starter scope (SQL, PDF, quantity) once per page load. The three starters appear
immediately, without network requests, loading states, LLM costs, or trace calls.
They remain unchanged on minimize/reopen and disappear after the first submission.
New chat does not restore them; a page reload selects a new random set (repeats
across reloads are possible). There is no refresh button or follow-up list.

Selecting a prompt only fills the composer; the user must send it. IDs are examples,
not guaranteed current inventory records, and quantity changes still require approval.
The paid `/api/ai/suggestions` endpoint and generation logic have been removed.
The composer clears immediately on Send, preserving the submitted request in its
saved audit run. The compact launcher has an animated gradient border. The composer
uses the same animation while requests or approvals are processing. Reduced-motion
preferences keep these gradients static.

#### Provider configuration

Only TypeSafe/Jev and Mistral are used for AI; no browser receives API keys. Requests
and the relevant schema or retrieved PDF passages go to the providers; bounded inventory facts
also go to Mistral for the explanation, as described above. Mistral uses structured
JSON for plans and SSE for explanation text, with temperature zero, `top_p=1`, and
`reasoning_effort=none`. Thinking blocks are discarded. Each provider attempt has a
20-second timeout (45 seconds for context-rich RAG answer generation); Mistral retries HTTP 503 at most twice, waiting two then four
seconds. Jev does not retry. Model errors are explicit, not silently routed elsewhere.
Changing a model or key requires a backend restart. Paid calls still incur provider
charges. Optional routing comparisons are described below; no monetary cost is estimated.

#### Streaming and trace dashboard

`POST /api/ai/chat/stream` returns server-sent events: `progress`, `validated`,
incremental `delta` text, optional `warning`, and terminal `done` (or fatal `error`).
The browser consumes the response as it arrives, renders GitHub-flavored Markdown
(tables, lists, code, emphasis), and disallows raw HTML/images. There is no fake
typewriter animation. The non-streaming `/chat` endpoint retains its existing
validated-answer behavior without an extra explanation call.

If explanation generation fails or is stopped, the partial response is marked
incomplete and the already-validated evidence remains saved. Quantity previews
remain pending; stopping never approves them. Cancelling during routing prevents
dispatch at the next phase boundary. An in-flight provider call may take up to its
timeout to release the single-operation gate. The connection sends keepalives while
waiting; explanation streaming has a 90-second soft budget checked between lines.
Mistral 503 retries occur only before a successful response stream, never by replaying
partially emitted text.
After a backend restart, explanations that were in progress are marked interrupted;
the validated result and any pending preview remain available.

The **Traceability** button opens `/ai/traces` in a new tab, preselecting the current
request. It includes local request history (latest 50), routing probabilities,
execution timeline, request counts by agent, latency chart, reported tokens,
per-provider call measurements, PDF citations/retrieval evidence, SQL/results, and quantity/price
decision evidence. Unknown token usage is labeled unknown and excluded from the
reported-token total. An agent-completed count is not a claim that its explanation
also completed. Refresh the page to see new requests or approvals.

The former cross-agent average-time chart is replaced **on this same page** by a
Jev-versus-Mistral router comparison: bar chart, metric table, benchmark controls,
progress, saved results, and individual decisions/errors. It has two separate modes:

- **Live shadow:** available through the chat API with `compare_routing: true`;
  the frontend chat does not expose this option and leaves it disabled. Jev is called exactly once and remains authoritative. Mistral
  classifies the identical memory-augmented input, using the same five route labels,
  but never dispatches an agent or creates/approves a mutation. This sequential extra
  paid call adds latency. A Mistral error is surfaced without invalidating Jev's task;
  a Jev error never falls back to Mistral. The dashboard aggregates the latest 50 live pairs.
- **Labeled benchmark:** click **Run 20-case benchmark (40 paid calls)**. Twenty fixed
  synthetic prompts, four per label, are classified once by each router, without
  executing any inventory, document, or approval action. Provider order alternates
  per example; calls are sequential and neither classifier retries. This uses the
  same single-operation gate as chat. Each pair is saved; progress is polled on the
  existing trace page, and the latest ten benchmark runs are selectable there.
  Leaving the page does not cancel the backend job. Restarting marks it interrupted,
  retains partial results, and never automatically spends money retrying it.

Mean, median, and nearest-rank P95 latency use only pairs where **both** classifiers
succeeded, measured over request plus response validation, excluding telemetry and
downstream execution. Input/output token means use the further matched subset where
both providers reported both token counts. Coverage is displayed; missing usage is
unknown, never zero. Individual reported usage remains available even if excluded
from matched means. Errors count all attempted pairs. Route agreement is **not
accuracy**; labeled expected routes are shown for reference, not a quality score.
The protocol is `router-comparison-v1`, dataset `inventory-routing-20-v1`.
Small samples, different provider APIs/output formats, and uncalibrated confidence
scores limit these demo measurements; they are not a general model leaderboard.

#### MLflow, LangSmith, and audit persistence

With AI enabled, startup creates `agent_runs`, `agent_proposals`, and `router_comparisons` tables alongside
inventory in `vibe_inventory.db`; no inventory rows are seeded or replaced. Back up this
database before enabling the feature. Audit records include the original request, SQL/results and
proposal decisions with timestamps. They contain demo inventory data and are local,
not encrypted or immutable. There is no authenticated approver identity.

MLflow defaults to the ignored local `mlflow.db` SQLite store. Trace spans cover
routing, agents, SQL execution, and approval decisions; they contain IDs, model names,
routing confidence, latency, reported usage, retry counts, and operation metadata, **not API keys, prompts,
or inventory content**. Full run evidence is available from the local application
audit history. Returned trace IDs correlate application runs with MLflow.
Only local `sqlite:` and `file:` tracking URIs are allowed in this showcase.

Optional LangSmith sends explicit **metadata-only** spans to the hosted US LangSmith
API alongside MLflow. To enable it, rotate any key exposed in chat, then add these
settings to your ignored local `.env` and restart the backend:

```dotenv
LANGSMITH_ENABLED=true
LANGSMITH_API_KEY=<fresh-key-stored-only-locally>
LANGSMITH_PROJECT=vibe-inventory
```

The server uses the LangSmith SDK directly with empty inputs/outputs, hidden payloads,
no automatic tracing, and no runtime-environment capture. Exports contain only
stage/run IDs, selected agent, confidence, timings, model names, reported token usage,
attempt counts, row counts, and error types. Prompts, SQL, inventory values, and
credentials are never included. Parent/child span relationships are preserved.
LangSmith export errors are logged by exception type and shown in local trace
warnings; they do not roll back inventory operations or silently claim export success.
Its two-second request timeout and zero automatic retries bound telemetry overhead.
No LangSmith key is required when disabled. The trace page links to LangSmith; search
the configured project using the displayed root run ID.

Open MLflow in another terminal from the project root:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Visit `http://127.0.0.1:5000` and select `vibe-inventory-agent`.
Keep MLflow local; it is not an authenticated public dashboard.

AI endpoints: `GET /api/ai/status`, `POST /api/ai/chat`,
`POST /api/ai/chat/stream`,
`GET /api/ai/runs`, `GET /api/ai/runs/{id}`,
`GET /api/ai/comparisons`, `GET /api/ai/comparisons/{id}`,
`POST /api/ai/comparisons/benchmark` (202 accepted),
and `POST /api/ai/proposals/{id}/approve` or `/reject`.
Provider errors are explicit in saved failed runs. There is no silent fallback to
another paid provider. One chat can run at a time per backend process;
concurrent operations receive HTTP 429. Run a single backend worker for this demo.
Startup marks interrupted running requests as failed. Proposals and successful run
results commit together before explanation generation; streaming interruption cannot
apply a pending action. The pending preview remains visible in the saved run.
The old multi-model agent benchmark and tool-based flow are removed; the new opt-in
benchmark compares only classification. Existing historical audit
rows are retained, but old-format runs are not shown in the simplified UI. Pending
old-format quantity/price proposals are rejected on startup and cannot be approved.
No inventory rows or historical audit records are deleted during this transition.

Deterministic tests use mocked providers, temporary inventory, and a temporary local
MLflow database. They require no real keys or paid calls:

```powershell
.\.venv\Scripts\python.exe -m unittest backend.tests.test_ai backend.tests.test_rag -v
```

The streaming decoder also has Node-native tests (Node 22.6+ for TypeScript stripping;
the installed Node 24 works). From `frontend`, run `npm run test:stream`. These cover
incremental delivery, fragmented UTF-8/CRLF, keepalives, errors, cancellation, and
truncated streams. `npm run build` checks the complete frontend TypeScript and bundle.

Before public production deployment, add authenticated users, per-action authorization,
CSRF/session protections appropriate to that authentication, durable job workers,
shared rate/spend limits, schema migrations, protected audit storage/retention,
broader held-out evaluation, monitoring, and a deployment-specific database strategy.

The frontend defaults to the local backend URL. To use another backend host or port:

```bash
cp frontend/.env.example frontend/.env.local
```

Then update:

```text
VITE_API_URL=http://localhost:8000/api
```

## Features

- Inventory dashboard with value, units, low-stock, data-quality, and unpriced-SKU KPIs
- Category value and supplier exposure graphs
- Search by item, category, or supplier
- Category, supplier, and stock-status filters
- 12 inventory items per page with Previous/Next navigation
- Explicit quantity editing with direct entry, `+`, `-`, and Save controls
- Bulk percentage or fixed-amount price updates for selected items
- Price updates block duplicate submissions and display cents; inventory and dashboard refresh independently, with explicit refresh errors
- Clear handling for missing item names, missing prices, and negative quantities
- Last successful API responses cached in the browser for unreliable connectivity
- Floating, resizable AI assistant: Jev routing, streamed Markdown answers, approved single/bulk quantity and absolute price changes, a charted trace dashboard, MLflow, and optional metadata-only LangSmith

## Backend Structure

- `backend/core/database.py` - SQLAlchemy engine, session factory, and dependency
- `backend/models.py` - SQLAlchemy ORM model mapped to the supplied database
- `backend/schemas/` - Pydantic request and response models
- `backend/services/` - inventory and dashboard business/query logic
- `backend/routers/` - FastAPI route modules

## Validation

Backend compile and API smoke check:

```bash
./.venv/Scripts/python.exe -m compileall backend
./.venv/Scripts/python.exe -c "from fastapi.testclient import TestClient; from backend.main import app; client=TestClient(app); assert client.get('/api/health').status_code == 200; assert client.get('/api/inventory?page_size=1').status_code == 200; assert client.get('/api/dashboard').status_code == 200; print('Backend smoke test passed')"
```

Use `./.venv/bin/python` instead on macOS/Linux/WSL.

Frontend production build:

```bash
cd frontend
npm run build
```

## Troubleshooting

### Git Bash reports `sed: command not found` or cannot find `.venv/bin/activate`

Close the current Git Bash window and open a fresh one. On Windows, do not use `.venv/bin/activate`; use the direct Windows venv path:

```bash
bash backend/setup.sh
source .venv/Scripts/activate
python -m uvicorn backend.main:app --reload --port 8000
```

Activation is optional.

### Port 8000 is already in use

Start the backend on another port and set the matching frontend URL in `frontend/.env.local`:

```bash
source .venv/Scripts/activate
python -m uvicorn backend.main:app --reload --port 8001
```

```text
VITE_API_URL=http://localhost:8001/api
```

### Frontend dependencies are missing

From the frontend directory:

```bash
npm install
```

### The frontend shows cached or offline data

Confirm the backend is running, then refresh the browser. Editing is disabled while the API is unavailable.

## Submission Documentation

See [DECISIONS.md](DECISIONS.md) for design decisions, limitations, and possible future improvements.
