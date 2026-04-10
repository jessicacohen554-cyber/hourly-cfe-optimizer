#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Clean Energy Profile Modeler — Linux/Mac Launcher
# ─────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

# Check Python
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python 3.9+ required. Install from https://www.python.org"
    exit 1
fi

echo "[OK] Found Python"

# Install deps
echo "Installing dependencies..."
"$PYTHON" -m pip install -r requirements.txt --quiet 2>/dev/null || true
echo "[OK] Dependencies installed"

# Open browser
(sleep 2 && (xdg-open http://127.0.0.1:8050 2>/dev/null || open http://127.0.0.1:8050 2>/dev/null || true)) &

echo ""
echo "================================================================"
echo "  Clean Energy Profile Modeler"
echo "  Running at http://127.0.0.1:8050"
echo "  Press Ctrl+C to stop"
echo "================================================================"
echo ""

PYTHONPATH="$(pwd):$(pwd)/../scripts" "$PYTHON" -m uvicorn backend.server:app --host 127.0.0.1 --port 8050
