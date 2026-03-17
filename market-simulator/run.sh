#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Market Simulator Launcher (Mac / Linux)
# Run: bash run.sh   or   chmod +x run.sh && ./run.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8000

# ── Find Python ──────────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$cmd"
            echo "[OK] Found $cmd ($version)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python 3.9+ is required but not found."
    echo "  Install Python from https://www.python.org/downloads/"
    exit 1
fi

# ── Install dependencies ────────────────────────────────────────────────────
echo "Installing dependencies..."
$PYTHON -m pip install -r app-startup/requirements.txt --quiet 2>&1 | tail -1
echo "[OK] Dependencies installed"

# ── Generate synthetic data if needed ────────────────────────────────────────
if [ ! -f "data/profiles/eia_demand_profiles.json" ]; then
    echo "Generating synthetic data profiles..."
    PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/backend:$SCRIPT_DIR/scripts" $PYTHON scripts/generate_synthetic_profiles.py 2>/dev/null || true
    echo "[OK] Data profiles ready"
else
    echo "[OK] Data profiles found"
fi

# ── Open browser (after short delay for server startup) ──────────────────────
(sleep 2 && {
    URL="http://127.0.0.1:$PORT"
    if command -v open &>/dev/null; then
        open "$URL"                          # macOS
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$URL"                      # Linux
    else
        echo "Open your browser to: $URL"
    fi
}) &

# ── Start server ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Market Simulator running at http://127.0.0.1:$PORT"
echo "  Press Ctrl+C to stop"
echo "================================================================"
echo ""

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/backend:$SCRIPT_DIR/scripts"
trap 'echo ""; echo "Server stopped."; exit 0' INT TERM

$PYTHON -m uvicorn backend.main:app --host 127.0.0.1 --port $PORT
