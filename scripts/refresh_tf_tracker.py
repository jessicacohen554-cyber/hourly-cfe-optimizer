#!/usr/bin/env python3
"""
Scan data/step2.3-de/thermal-fleet/ for REAL thermal fleet parquets
and emit a JSON blob keyed as iso|pw|scen|demand|gas|pop|iter → cost.

Validation: a file is a genuine TF result IFF its parquet schema contains
the 'coal_mw' column. Gas-only solver files committed to this folder with
_g1_tf.parquet naming lack that column and are excluded.

When multiple variants exist for the same key (e.g. _tf_tf and _tf2_tf),
tf2 wins.

Usage:
    python scripts/refresh_tf_tracker.py [--out path/to/data.json]

The output JSON can be pasted into the HTML tracker's RAW constant
or loaded by a downstream script.
"""

import argparse
import glob
import json
import re
import sys

import pyarrow.parquet as pq

TF_DIR = "data/step2.3-de/thermal-fleet"

# Matches: {ISO}_pathway{PW}_{scenario}_{demand}_p{pop}_i{iter}_s{seed}_g{gas}_{variant}.parquet
FNAME_RE = re.compile(
    r"([A-Z]+)_pathway([A-D])_(.+?)_(High|Medium|Low)_p(\d+)_i(\d+)_s\d+_g([123])_(.+?)\.parquet$"
)


def is_real_tf(filepath: str) -> bool:
    """Check parquet schema for coal_mw — the thermal fleet discriminator."""
    try:
        schema = pq.read_schema(filepath)
        return "coal_mw" in schema.names
    except Exception:
        return False


def extract_cost(filepath: str) -> float | None:
    """Read final-year cumulative_B from a TF parquet."""
    try:
        df = pq.read_table(filepath).to_pandas()
        return round(float(df.iloc[-1]["cumulative_B"]), 1)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Refresh TF tracker data")
    parser.add_argument(
        "--out", default="data/step2.3-de/thermal-fleet/tracker_data.json",
        help="Output JSON path (default: tracker_data.json in thermal-fleet dir)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{TF_DIR}/*_g[123]_*tf.parquet"))
    results = {}
    skipped_schema = 0
    skipped_parse = 0
    skipped_cost = 0

    for f in files:
        fname = f.split("/")[-1]
        m = FNAME_RE.match(fname)
        if not m:
            skipped_parse += 1
            continue

        # Schema gate: must have coal_mw
        if not is_real_tf(f):
            skipped_schema += 1
            continue

        iso, pw, scen, demand, pop, itr, gas, variant = m.groups()
        pop = int(pop)
        itr = int(itr)
        is_tf2 = "tf2" in variant

        key = f"{iso}|{pw}|{scen}|{demand}|{gas}|{pop}|{itr}"

        cost = extract_cost(f)
        if cost is None:
            skipped_cost += 1
            continue

        # tf2 overrides tf for the same key
        if key not in results or is_tf2:
            results[key] = cost

    # Summary
    scenarios = sorted(set(k.split("|")[2] for k in results))
    demands = sorted(set(k.split("|")[3] for k in results))
    gases = sorted(set(k.split("|")[4] for k in results))
    pops = sorted(set(int(k.split("|")[5]) for k in results))
    iters = sorted(set(int(k.split("|")[6]) for k in results))

    print(f"Real TF cells:    {len(results)}")
    print(f"Skipped (schema): {skipped_schema}")
    print(f"Skipped (parse):  {skipped_parse}")
    print(f"Skipped (cost):   {skipped_cost}")
    print(f"Scenarios:        {scenarios}")
    print(f"Demands:          {demands}")
    print(f"Gas modes:        {gases}")
    print(f"Pops:             {pops}")
    print(f"Iters:            {iters}")

    if args.verbose:
        from collections import Counter
        combo = Counter()
        for k in results:
            p = k.split("|")
            combo[(p[2], p[3], p[4])] += 1
        print("\nCells by scenario × demand × gas:")
        for (s, d, g), n in sorted(combo.items()):
            print(f"  {s} / {d} / g{g}: {n}")

    with open(args.out, "w") as f:
        json.dump(results, f, separators=(",", ":"))

    print(f"\nWrote {args.out} ({len(results)} entries)")


if __name__ == "__main__":
    main()
