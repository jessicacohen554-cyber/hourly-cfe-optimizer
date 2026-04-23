#!/bin/bash
# Parallel pathway sweep across 7 ISOs.
#
# Problem: append_to_manifest in step_2_3 is read-modify-write with no
# locking, so concurrent writers clobber each other's MANIFEST entries.
# Problem: subprocess-per-run loses the 2.5 s/run dispatch-cache +
# gen-profile loads (cProfile Apr 17 2026).
#
# Fix: one in-process driver per ISO, each writing to a per-ISO
# output-root. After all workers finish, merge the per-ISO JSONs and
# MANIFESTs into the canonical data/step2.3-pathway/.
set -u
cd "$(dirname "$0")/.."

STAGE=data/step2.3-pathway/_workers
FINAL=data/step2.3-pathway
mkdir -p "$STAGE"
mkdir -p logs/rt_workers

echo "=== $(date -u) launching 7 parallel in-process ISO workers ==="
PIDS=()
for iso in CAISO ERCOT PJM NYISO NEISO MISO SPP; do
    mkdir -p "$STAGE/$iso"
    python3 scripts/_sweep_pathways_inproc.py \
        --iso "$iso" \
        --output-root "$STAGE/$iso" \
        > "logs/rt_workers/$iso.log" 2>&1 &
    PIDS+=($!)
done

echo "=== $(date -u) waiting on ${#PIDS[@]} workers: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "!!! worker pid=$pid exited non-zero"
done

echo "=== $(date -u) merging per-worker outputs into $FINAL ==="
python3 - <<'PY'
import json, os, shutil
from pathlib import Path
stage = Path('data/step2.3-pathway/_workers')
final = Path('data/step2.3-pathway')
final.mkdir(parents=True, exist_ok=True)

merged_runs = {}
stale_backup = final / 'MANIFEST.stale-elcc-backup.json'

for iso_dir in sorted(stage.iterdir()):
    if not iso_dir.is_dir():
        continue
    iso = iso_dir.name
    dest = final / iso
    dest.mkdir(exist_ok=True)
    # Copy per-run JSONs (overwrite canonical location).
    n_json = 0
    for p in sorted(iso_dir.glob('*/pathway*.json')):
        shutil.copy2(p, dest / p.name)
        n_json += 1
    # Also check direct children in case the layout is flat.
    for p in sorted(iso_dir.glob('pathway*.json')):
        shutil.copy2(p, dest / p.name)
        n_json += 1
    # Merge MANIFEST.
    mf = iso_dir / 'MANIFEST.json'
    if mf.exists():
        m = json.load(open(mf))
        runs = m.get('runs', {})
        merged_runs.update(runs)
        print(f"  {iso}: +{n_json} JSONs, +{len(runs)} MANIFEST entries")
    else:
        print(f"  {iso}: +{n_json} JSONs, NO MANIFEST")

canonical = final / 'MANIFEST.json'
out = {
    'schema_version': 2,
    'description': (
        "Reliability Tax pathway optimizer manifest "
        "(v2: Card U new-build gas + Card F' CF-stranding + "
        "Card S 10-endpoint sweep \u2014 7 ISOs \u00d7 6 pathways \u00d7 10 endpoints = 420 runs). "
        "Regenerated under SPEC \u00a724.5 worst-hour gas sizing "
        "(Apr 17 2026) via parallel per-ISO workers then merged."
    ),
    'runs': merged_runs,
}
with open(canonical, 'w') as fh:
    json.dump(out, fh, indent=2, default=str)
print(f"\nmerged MANIFEST: {len(merged_runs)} runs -> {canonical}")
PY

# Clean up staging — keep the logs, delete the per-worker dirs
rm -rf "$STAGE"
echo "=== $(date -u) DONE ==="
