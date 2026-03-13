#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Market Simulator Launcher (Mac / Linux)
# Double-click or run: bash run.sh
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
            echo "✓ Found $cmd ($version)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "✗ Python 3.9+ is required but not found."
    echo "  Install Python from https://www.python.org/downloads/"
    exit 1
fi

# ── Install dependencies ────────────────────────────────────────────────────
echo "Installing dependencies..."
$PYTHON -m pip install -r requirements.txt --quiet 2>&1 | tail -1
echo "✓ Dependencies installed"

# ── Generate synthetic data if needed ────────────────────────────────────────
if [ ! -f "data/demand_profiles_2025.json" ]; then
    echo "Generating synthetic data profiles..."
    PYTHONPATH="$SCRIPT_DIR" $PYTHON scripts/generate_synthetic_profiles.py 2>/dev/null || true
    echo "✓ Data profiles ready"
else
    echo "✓ Data profiles found"
fi

# ── Download JS libraries if missing ─────────────────────────────────────────
if [ ! -f "frontend/js/plotly.min.js" ]; then
    echo "Downloading Plotly.js for offline use..."
    curl -sL "https://cdn.plot.ly/plotly-2.27.0.min.js" -o frontend/js/plotly.min.js 2>/dev/null || \
        echo "  (CDN unavailable — will use online fallback)"
fi

if [ ! -f "frontend/js/html2canvas.min.js" ]; then
    echo "Downloading html2canvas for chart export..."
    curl -sL "https://html2canvas.hertzen.com/dist/html2canvas.min.js" -o frontend/js/html2canvas.min.js 2>/dev/null || \
        echo "  (CDN unavailable — will use online fallback)"
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
echo "═══════════════════════════════════════════════════════════════"
echo "  Market Simulator running at http://127.0.0.1:$PORT"
echo "  Press Ctrl+C to stop"
echo "═══════════════════════════════════════════════════════════════"
echo ""

export PYTHONPATH="$SCRIPT_DIR"
trap 'echo ""; echo "Server stopped."; exit 0' INT TERM

$PYTHON -m uvicorn backend.main:app --host 127.0.0.1 --port $PORT
