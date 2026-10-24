#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="$(command -v python3 || command -v python)"
"$PYTHON_BIN" -m venv .venv

if [[ -f .venv/Scripts/activate ]]; then
	VENV_PYTHON=".venv/Scripts/python.exe"
	ACTIVATION_COMMAND="source .venv/Scripts/activate"
else
	VENV_PYTHON=".venv/bin/python"
	ACTIVATION_COMMAND="source .venv/bin/activate"
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r backend/requirements.txt

printf '\nBackend environment ready. Activate it with:\n  %s\nRun the API with:\n  python -m uvicorn backend.main:app --reload --port 8000\n' "$ACTIVATION_COMMAND"
