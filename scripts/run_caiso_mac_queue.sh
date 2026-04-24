#!/bin/bash
# Run CAISO MAC queue one pathway at a time to avoid OOM.
# Each pathway runs in its own process (~85s, ~4GB peak).
# Results are written to temp dir then merged.

set -e

OUTDIR="data/step3-dispatch/mac_queue"
TMPDIR="data/step3-dispatch/mac_queue/caiso_tmp"
mkdir -p "$TMPDIR"

SENSITIVITIES="all_low all_med all_high high_vre_low_firm high_firm_low_vre"
GROWTHS="Low Medium High"

for sens in $SENSITIVITIES; do
  for growth in $GROWTHS; do
    echo "=== Running $sens / $growth ==="
    python3 scripts/step3b_mac_queue.py \
      --iso CAISO \
      --sensitivity "$sens" \
      --growth "$growth" \
      --output-dir "$TMPDIR" 2>&1 | tail -5

    # Rename output to include pathway info
    if [ -f "$TMPDIR/mac_queue_CAISO.parquet" ]; then
      mv "$TMPDIR/mac_queue_CAISO.parquet" "$TMPDIR/mac_queue_CAISO_${sens}_${growth}.parquet"
      echo "  Saved: mac_queue_CAISO_${sens}_${growth}.parquet"
    fi
    echo ""
  done
done

echo "=== Merging all pathways ==="
python3 -c "
import pandas as pd
import os, glob, json, time

tmpdir = '$TMPDIR'
outdir = '$OUTDIR'

# Merge all pathway parquets
parts = sorted(glob.glob(os.path.join(tmpdir, 'mac_queue_CAISO_*.parquet')))
print(f'Found {len(parts)} pathway files')

dfs = [pd.read_parquet(p) for p in parts]
merged = pd.concat(dfs, ignore_index=True)
print(f'Total rows: {len(merged)}')

# Save merged CAISO parquet
merged.to_parquet(os.path.join(outdir, 'mac_queue_CAISO.parquet'), index=False)
print('Saved mac_queue_CAISO.parquet')

# Now rebuild summary JSON with all ISOs
all_dfs = []
for iso_file in sorted(glob.glob(os.path.join(outdir, 'mac_queue_*.parquet'))):
    if 'CAISO_' in os.path.basename(iso_file):
        continue  # skip tmp pathway files
    all_dfs.append(pd.read_parquet(iso_file))

all_df = pd.concat(all_dfs, ignore_index=True)
results = all_df.to_dict('records')

summary = {
    'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
    'isos': sorted(all_df['iso'].unique().tolist()),
    'total_rows': len(results),
    'results': results,
}
with open(os.path.join(outdir, 'mac_queue_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f'Saved summary: {len(results)} rows across {len(summary[\"isos\"])} ISOs')
"

# Cleanup tmp
rm -rf "$TMPDIR"
echo "=== Done ==="
