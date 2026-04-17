#!/bin/bash
# Sequential driver for run_pathway_sweep.py across all 7 ISOs.
# Resumption: MANIFEST.json is empty on first launch (old ELCC MANIFEST was
# renamed to MANIFEST.stale-elcc-backup.json). Each completed pathway run
# writes its JSON + updates MANIFEST; if the sandbox kills mid-sweep, the
# next launch picks up at the first MANIFEST gap.
set -u
cd "$(dirname "$0")/.."
for iso in CAISO ERCOT PJM NYISO NEISO MISO SPP; do
    echo "=== $(date -u) launching ISO=$iso ==="
    python3 scripts/run_pathway_sweep.py --iso "$iso" || {
        echo "!!! $(date -u) ISO=$iso exited non-zero; continuing to next ISO"
    }
done
echo "=== $(date -u) all ISOs done ==="
