# VibeWorx Inventory

A full-stack inventory management application built for the VibeWorx interview challenge.

## Requirements

- Python 3.10+
- Node.js 20.19+
- npm
- Git Bash on Windows, or Bash on macOS/Linux/WSL
- The supplied `vibeworx.db` file in the repository root

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

The frontend expects the backend at `http://localhost:8000/api`.

## Query the Database in VS Code

Install the **SQLite** extension by Alex Covizzi, then use **SQLite: Open Database**
from the Command Palette to select `vibeworx.db` in the project root (not in `backend`).
Open [inventory_queries.sql](inventory_queries.sql), select a query, and run
**SQLite: Run Selected Query**. The supplied queries only read data.

## Configuration

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
- Clear handling for missing item names, missing prices, and negative quantities
- Last successful API responses cached in the browser for unreliable connectivity

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
