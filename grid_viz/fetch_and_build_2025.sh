#!/bin/bash
# Fetch 2025 ERCOT data and rebuild grid animation
#
# Usage:
#   export EIA_API_KEY=your_key_here
#   bash grid_viz/fetch_and_build_2025.sh
#
# This will:
# 1. Fetch raw EIA-930 hourly generation + demand for the battery_2025 scenario
# 2. Run the frame prep to build animation parquets (including battery plants)
# 3. Export to dashboard JS
#
# Prerequisites: pip install pandas pyarrow requests

set -e

if [ -z "$EIA_API_KEY" ]; then
    echo "ERROR: Set EIA_API_KEY first"
    echo "  export EIA_API_KEY=your_key_here"
    echo "  Get a free key at: https://www.eia.gov/opendata/register.php"
    exit 1
fi

echo "=== Step 1: Fetch 2025 EIA-930 data ==="
python3 grid_viz/fetch/fetch_eia930_raw.py --scenario battery_2025

echo ""
echo "=== Step 2: Build animation frames ==="
python3 grid_viz/prep/prep_animation_frames.py

echo ""
echo "=== Step 3: Export to dashboard JS ==="
python3 grid_viz/prep/export_animation_json.py

echo ""
echo "=== Done! ==="
echo "Open dashboard/grid_animation.html and select '2025 Solar + Battery Peak'"
