#!/bin/bash
# Parallel 7-ISO fanout of scripts/_sweep_pathways_inproc.py for the post-Step
# 3 v2 accounting-fix rerun. Each ISO runs in its own long-lived Python
# process (in-process pathway sweep — keeps numba JIT + dispatch cache +
# EF parquet hot) so the 50 (pathway × endpoint) combos per ISO amortize
# their startup cost. Expected wall clock after cache-memo fix: ~30–60 s
# for all 350 runs on a single machine.
#
# Launch pattern (per CLAUDE.md Optimizer Run Discipline):
#   nohup setsid bash scripts/_sweep_all_isos_step3fix.sh \
#       > logs/rt_step3fix_sweep_wrapper.log 2>&1 &
#
set -u
cd /home/user/hourly-cfe-optimizer

ISOS=(CAISO ERCOT PJM NYISO NEISO MISO SPP)
ENDPOINTS="0.60,0.70,0.75,0.80,0.85,0.90,0.95,0.975,0.99,0.999"
PATHWAYS="1,1a,2a,2b,3"
OUT=analysis/reliability-tax/data
LOG_DIR=logs
SUMMARY=$LOG_DIR/rt_step3fix_sweep.log

mkdir -p "$LOG_DIR"
{
  echo "=== 350-run sweep starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "endpoints=$ENDPOINTS pathways=$PATHWAYS output=$OUT"
  echo "strategy: in-process per-ISO workers, parallel fan-out"
} > "$SUMMARY"

T0=$(date +%s)
pids=()
for iso in "${ISOS[@]}"; do
    iso_log=$LOG_DIR/rt_step3fix_${iso}.log
    {
        echo "ISO_START $iso $(date -u +%H:%M:%S)"
        python3 scripts/_sweep_pathways_inproc.py \
            --iso "$iso" \
            --output-root "$OUT" \
            --endpoints "$ENDPOINTS" \
            --pathways "$PATHWAYS"
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "ISO_DONE $iso rc=$rc"
        else
            echo "ISO_FAIL $iso rc=$rc"
        fi
    } > "$iso_log" 2>&1 &
    pids+=($!)
done

echo "Workers launched: ${pids[*]}" | tee -a "$SUMMARY"

# Wait for all workers; collect per-ISO logs into the summary.
fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=$((fail + 1))
    fi
done

for iso in "${ISOS[@]}"; do
    iso_log=$LOG_DIR/rt_step3fix_${iso}.log
    {
        echo "----- $iso -----"
        tail -5 "$iso_log"
    } >> "$SUMMARY"
done

T1=$(date +%s)
DT=$((T1 - T0))
echo "SWEEP_DONE workers_failed=$fail elapsed=${DT}s $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SUMMARY"

exit $fail
