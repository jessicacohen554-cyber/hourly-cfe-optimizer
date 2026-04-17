#!/bin/bash
# Self-healing sequential driver for run_pathway_sweep.py across 7 ISOs.
# MANIFEST skip-logic makes restarts idempotent: each relaunch picks up
# at the first uncompleted (iso, pathway, endpoint) combo. The outer
# while-loop automatically restarts the whole sweep if any ISO exits
# non-zero OR the bash process is re-spawned after sandbox kill.
# Exits 0 cleanly only when MANIFEST shows all 420 expected runs.
set -u
cd "$(dirname "$0")/.."
MANIFEST=analysis/reliability-tax/data/MANIFEST.json
TARGET=420
while :; do
    COUNT=$(python3 -c "import json; print(len(json.load(open('$MANIFEST')).get('runs', {})))" 2>/dev/null || echo 0)
    if [ "$COUNT" -ge "$TARGET" ]; then
        echo "=== $(date -u) SWEEP COMPLETE: $COUNT / $TARGET runs"
        exit 0
    fi
    echo "=== $(date -u) OUTER LOOP start: MANIFEST=$COUNT / $TARGET ==="
    for iso in CAISO ERCOT PJM NYISO NEISO MISO SPP; do
        echo "=== $(date -u) ISO=$iso ==="
        python3 scripts/run_pathway_sweep.py --iso "$iso" || {
            echo "!!! $(date -u) ISO=$iso exited non-zero; continuing"
        }
    done
    # If outer while-loop survives the whole inner for-loop, re-check
    # MANIFEST; if we're not at TARGET, loop again (picks up any combos
    # that failed mid-run).
    sleep 2
done
