#!/bin/bash
# Wrapper that drives scripts/run_pathway_sweep.py across all 7 ISOs for the
# post-Step 3 v2 accounting-fix rerun. 5 distinct pathways × 10 endpoints × 7
# ISOs = 350 runs. Sequential execution (avoid OOM from parallel EF loads).
# Emit one status line per ISO for the Monitor to stream.
set -u
cd /home/user/hourly-cfe-optimizer

ISOS=(CAISO ERCOT PJM NYISO NEISO MISO SPP)
ENDPOINTS="0.60,0.70,0.75,0.80,0.85,0.90,0.95,0.975,0.99,0.999"
PATHWAYS="1,1a,2a,2b,3"
OUT=analysis/reliability-tax/data
LOG=logs/rt_step3fix_sweep.log

mkdir -p logs
{
  echo "=== 350-run sweep starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "endpoints=$ENDPOINTS pathways=$PATHWAYS output=$OUT"
} > "$LOG"

for iso in "${ISOS[@]}"; do
    echo "ISO_START $iso $(date -u +%H:%M:%S)" | tee -a "$LOG"
    t0=$(date +%s)
    python3 scripts/run_pathway_sweep.py \
        --iso "$iso" \
        --endpoints "$ENDPOINTS" \
        --pathways "$PATHWAYS" \
        --growth Medium \
        --output-root "$OUT" \
        >> "$LOG" 2>&1
    rc=$?
    t1=$(date +%s)
    dt=$((t1 - t0))
    if [ $rc -eq 0 ]; then
        echo "ISO_DONE $iso elapsed=${dt}s" | tee -a "$LOG"
    else
        echo "ISO_FAIL $iso rc=$rc elapsed=${dt}s" | tee -a "$LOG"
    fi
done

echo "SWEEP_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
