#!/usr/bin/env python3
"""
extract_tf_gas_compare.py

Reads *_g1_*_tf.parquet and *_g3_*_tf.parquet from the thermal-fleet folder.
For each (ISO, pathway, gas_mode, cost_scenario, demand_growth) combo, picks
the min-cost run across solver configs (popsize, maxiter, seed). Computes
median/p10/p90 gas_mw trajectories and median cost across 9 scenario combos
per ISO, then writes the result as JSON.

The companion HTML (site/2_3_gas_compare_tf.html) fetches this JSON at runtime.

File filter:
  INCLUDE: *_g1_*_tf.parquet, *_g3_*_tf.parquet
  EXCLUDE: *_tf2_*.parquet (second-pass reruns)
  EXCLUDE: files without gas_mode tag or with _g2_ (not used in A_g3 vs B_g1)

Selection: min cumulative_B per (iso, pathway, gas_mode, scenario, demand).
           Different p/i/s configs are convergence attempts, not distinct scenarios.

Cost stat: median across 9 combos (not mean — solver failures wreck the mean).

Usage:
    python scripts/extract_tf_gas_compare.py

Outputs:
    site/data/tf_gas_compare.json
"""

import pyarrow.parquet as pq
import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FOLDER = Path("data/step2.3-de/thermal-fleet")
OUT_JSON = Path("site/data/tf_gas_compare.json")

ISOS = ["CAISO", "ERCOT", "MISO", "NEISO", "NYISO", "PJM", "SPP"]
SCENARIOS = ["base", "firm_high_vre_low", "firm_low_vre_high"]
DEMANDS = ["Low", "Medium", "High"]

# Baseline fossil fleet capacity per ISO (GW)
BASELINE_FOSSIL_GW = {
    "CAISO": 0.7, "ERCOT": 23.9, "MISO": 41.2,
    "NEISO": 10.1, "NYISO": 7.8, "PJM": 62.9, "SPP": 20.3,
}

# Annual demand per ISO (TWh) — used for $/MWh stat card
DEMAND_TWH = {
    "CAISO": 224.0, "ERCOT": 488.0, "MISO": 663.8,
    "NEISO": 115.3, "NYISO": 151.6, "PJM": 843.3, "SPP": 299.8,
}

# Which (pathway, gas_mode) pairs to compare
TARGETS = [("A_g3", "A", 3), ("B_g1", "B", 1)]


# ---------------------------------------------------------------------------
# 1. File filter + min-cost selection per combo
# ---------------------------------------------------------------------------
def load_best_runs(folder):
    """
    Return dict: (iso, pathway, scenario, demand, gas_mode) -> DataFrame

    Only reads *_tf.parquet files with explicit _g1_ or _g3_ tag.
    Excludes _tf2_ files. Picks min cumulative cost when multiple
    solver configs (popsize, maxiter, seed) exist for the same combo.
    """
    files = [
        f for f in os.listdir(folder)
        if f.endswith("_tf.parquet")
        and "_tf2_" not in f
        and ("_g1_" in f or "_g3_" in f)
    ]
    print(f"  Files passing filter: {len(files)}")

    combos = defaultdict(list)
    for f in files:
        df = pq.read_table(str(folder / f)).to_pandas()
        r = df.iloc[0]
        gm = int(r["gas_mode"])
        key = (r["iso"], r["pathway"], r["scenario"], r["demand_growth"], gm)
        final_cost = float(df.iloc[-1]["cumulative_B"])
        combos[key].append((final_cost, df))

    # Min cost per combo
    best = {}
    for k, runs in combos.items():
        runs.sort(key=lambda x: x[0])
        best[k] = runs[0][1]

    return best


# ---------------------------------------------------------------------------
# 2. Aggregate: median/p10/p90 across 9 scenario combos
# ---------------------------------------------------------------------------
def build_chart_data(best):
    data = {}
    for iso in ISOS:
        bl = BASELINE_FOSSIL_GW[iso]
        result = {"demand_twh": DEMAND_TWH[iso], "baseline_fossil_gw": bl}

        for label, pw, gm in TARGETS:
            series = []
            total_costs = []

            for scen in SCENARIOS:
                for dem in DEMANDS:
                    key = (iso, pw, scen, dem, gm)
                    if key not in best:
                        continue
                    df = best[key]
                    gas_above = (df["gas_mw"].values / 1000.0) - bl
                    series.append(gas_above)
                    total_costs.append(float(df.iloc[-1]["cumulative_B"]))

            if not series:
                continue

            arr = np.array(series)
            years = best[(iso, pw, SCENARIOS[0], DEMANDS[1], gm)]["year"].values

            med = np.median(arr, axis=0)
            lo = np.percentile(arr, 10, axis=0)
            hi = np.percentile(arr, 90, axis=0)

            # Median cost — not mean. Solver failures in firm_high/low scenarios
            # produce $10K–$150K B outliers that blow out the mean.
            med_cost = float(np.median(total_costs))
            dollar_mwh = med_cost * 1e9 / (DEMAND_TWH[iso] * 1e6 * 25)

            def pts(ys):
                return [{"x": int(y), "y": round(float(v), 1)} for y, v in zip(years, ys)]

            result[label] = {
                "med": pts(med),
                "lo": pts(lo),
                "hi": pts(hi),
                "n": len(series),
                "total_cost_B": round(med_cost, 1),
                "dollar_mwh": round(dollar_mwh, 1),
            }

        data[iso] = result

    return data


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    print(f"Reading parquets from {FOLDER} ...")
    best = load_best_runs(FOLDER)

    # Report combo counts
    for label, pw, gm in TARGETS:
        counts = {
            iso: sum(1 for k in best if k[0] == iso and k[1] == pw and k[4] == gm)
            for iso in ISOS
        }
        print(f"  {label}: {counts}")

    print("Building chart data (median cost, median/p10/p90 gas) ...")
    data = build_chart_data(best)

    # Verification
    for iso in ISOS:
        for label, _, _ in TARGETS:
            if label in data[iso]:
                d = data[iso][label]
                print(
                    f"  {iso:6s} {label}: "
                    f"${d['total_cost_B']:>10,.1f}B  "
                    f"${d['dollar_mwh']:>8.1f}/MWh  "
                    f"({d['n']} combos)"
                )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {OUT_JSON} ({OUT_JSON.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
