#!/bin/bash
# Post-Optimizer Pipeline
# Run after optimize_overprocure.py completes
# Steps: 1) Recompute CO₂  2) Run QA analysis  3) Report

set -e
cd "$(dirname "$0")"

echo "========================================"
echo "  POST-OPTIMIZER PIPELINE"
echo "========================================"
echo "  Started: $(date)"
echo ""

# Step 1: Verify results exist and are fresh
RESULTS="dashboard/overprocure_results.json"
if [ ! -f "$RESULTS" ]; then
    echo "ERROR: Results file not found: $RESULTS"
    exit 1
fi

FSIZE=$(stat -c %s "$RESULTS")
echo "  Results file: $FSIZE bytes"

# Step 2: Recompute CO₂ with hourly emission rates
echo ""
echo "  Step 1: Recomputing CO₂ with hourly fossil-fuel emission rates..."
python3 recompute_co2.py

# Step 3: Run QA/QC analysis
echo ""
echo "  Step 2: Running QA/QC analysis..."
python3 analyze_results.py

echo ""
echo "========================================"
echo "  PIPELINE COMPLETE: $(date)"
echo "========================================"
