#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Market Simulator Launcher (Mac / Linux)
# Run: bash run.sh   or   chmod +x run.sh && ./run.sh
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/backend:$SCRIPT_DIR/scripts"

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
if $PYTHON -m pip install -r app-startup/requirements.txt --quiet 2>&1 | tail -1; then
    echo "[OK] Dependencies installed"
else
    echo "  WARNING: Dependency installation may have failed. Some features may not work."
fi

# ── Generate synthetic data if needed ────────────────────────────────────────
if [ ! -f "data/profiles/eia_demand_profiles.json" ]; then
    echo "Generating synthetic data profiles..."
    $PYTHON scripts/generate_synthetic_profiles.py || echo "  WARNING: Synthetic profile generation failed"
fi
echo "✓ Data profiles ready"

# ── Generate plant heat rates if needed ──────────────────────────────────────
if [ ! -f "data/plant_heat_rates.json" ]; then
    echo "Generating plant-specific heat rates..."
    $PYTHON scripts/generate_plant_heat_rates.py || echo "  WARNING: Heat rate generation failed"
fi
echo "✓ Plant heat rates ready"

# ── Generate interchange profiles if needed ──────────────────────────────────
if [ ! -f "data/profiles/eia_interchange_profiles.json" ]; then
    echo "Generating inter-regional interchange profiles..."
    # generate_synthetic_profiles.py also produces interchange profiles
    $PYTHON scripts/generate_synthetic_profiles.py || echo "  WARNING: Interchange generation failed"
fi
echo "✓ Interchange profiles ready"

# ── Run 1,215-scenario parametric sweep if cached results missing ────────────
if [ ! -f "results/sweep_1215/sweep_1215_flat.parquet" ]; then
    echo ""
    echo "================================================================"
    echo "  Sweep cache not found. Running 1,215-scenario parametric sweep."
    echo "  This is a one-time operation and may take a while..."
    echo "  (3 demand x 5 price x 3 PPA x 3 gas x 3 queue x 3 fossil cost)"
    echo "  x 7 ISOs x 6 years = 51,030 simulations"
    echo "================================================================"
    echo ""
    $PYTHON scripts/run_sweep_1215.py --output-dir results/sweep_1215 || \
        echo "  WARNING: Sweep generation failed. Sweep-dependent features will be unavailable."
fi
echo "✓ Sweep cache ready"

# ── Build fleet scenario data if missing ─────────────────────────────────────
if [ ! -f "frontend/data/fleet_scenario_results_sample.json" ]; then
    echo "Building fleet scenario data..."
    $PYTHON scripts/build_fleet_scenario_data.py || echo "  WARNING: Fleet scenario build failed"
fi
echo "✓ Fleet scenario data ready"

# ── Generate constellation scenarios if missing ──────────────────────────────
if [ ! -f "frontend/data/constellation_scenarios.json" ]; then
    echo "Generating constellation scenarios..."
    $PYTHON scripts/generate_constellation_scenarios.py || echo "  WARNING: Constellation scenarios failed"
fi
echo "✓ Constellation scenarios ready"

# ── Extract ISO sweep data for dashboard if missing ──────────────────────────
if [ ! -f "frontend/data/sweep_dispatch_data.json" ]; then
    echo "Extracting ISO sweep data for dashboard..."
    $PYTHON scripts/extract_iso_sweep_data.py || echo "  WARNING: ISO sweep extraction failed"
fi
echo "✓ ISO sweep data ready"

# ── Download JS libraries for offline use (if missing) ───────────────────────
FRONTEND_LIB="frontend/js/lib"
mkdir -p "$FRONTEND_LIB"

if [ ! -f "$FRONTEND_LIB/plotly-latest.min.js" ]; then
    echo "Downloading Plotly.js for offline use..."
    if command -v curl &>/dev/null; then
        curl -sL "https://cdn.plot.ly/plotly-latest.min.js" -o "$FRONTEND_LIB/plotly-latest.min.js" || \
            echo "  WARNING: Plotly download failed. Charts will use CDN fallback."
    elif command -v wget &>/dev/null; then
        wget -q "https://cdn.plot.ly/plotly-latest.min.js" -O "$FRONTEND_LIB/plotly-latest.min.js" || \
            echo "  WARNING: Plotly download failed. Charts will use CDN fallback."
    fi
fi

if [ ! -f "$FRONTEND_LIB/html2canvas.min.js" ]; then
    echo "Downloading html2canvas for offline use..."
    if command -v curl &>/dev/null; then
        curl -sL "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js" -o "$FRONTEND_LIB/html2canvas.min.js" || \
            echo "  WARNING: html2canvas download failed."
    elif command -v wget &>/dev/null; then
        wget -q "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js" -O "$FRONTEND_LIB/html2canvas.min.js" || \
            echo "  WARNING: html2canvas download failed."
    fi
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

trap 'echo ""; echo "Server stopped."; exit 0' INT TERM

$PYTHON -m uvicorn backend.main:app --host 127.0.0.1 --port $PORT
