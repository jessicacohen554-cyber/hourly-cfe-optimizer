#!/usr/bin/env python3
"""
Step 9E: Extract data for novel use-case reference pages.

Produces 4 JS data files:
  - dashboard/js/tipping-point-data.js      (Technology Tipping Points)
  - dashboard/js/flexibility-value-data.js   (Value of Flexibility)
  - dashboard/js/datacenter-siting-data.js   (Data Center Siting)
  - dashboard/js/carbon-credit-data.js       (Carbon Credit Pricing)

All read from existing Step 3 parquets + Step 5 post-processing JSON.
No new compute — pure post-processing.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(ROOT, "data")
DASH_JS = os.path.join(ROOT, "dashboard", "js")

sys.path.insert(0, SCRIPT_DIR)
from pipeline_config import ISOS
KEY_THRESHOLDS = [50, 75, 85, 90, 95, 97.5, 99, 99.5, 99.9, 99.99]
ALL_THRESHOLDS = [10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 87.5,
                  90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99]

# Scenario code structure: ABCD_E_F_G_H
# A=Renewable, B=Firm, C=Storage, D=LDES, E=CCS, F=45Q(N/Y), G=Fuel+TX, H=Geo(X=none)
TOGGLE_NAMES = {
    0: "Renewable",
    1: "Firm Gen",
    2: "Storage",
    3: "LDES",
}
LEVEL_MAP = {"L": "Low", "M": "Medium", "H": "High"}


def load_co_parquet(iso):
    """Load Step 3 cost optimization baseline parquet."""
    path = os.path.join(DATA, "step3-cost-opt-parquets", f"step3_co_{iso}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def parse_scenario(code):
    """Parse scenario code into toggle levels."""
    parts = code.split("_")
    toggles = list(parts[0])  # ABCD
    return {
        "renewable": toggles[0],
        "firm": toggles[1],
        "storage": toggles[2],
        "ldes": toggles[3],
        "ccs": parts[1] if len(parts) > 1 else "M",
        "q45": parts[2] if len(parts) > 2 else "N",
        "fuel_tx": parts[3] if len(parts) > 3 else "M1",
        "geo": parts[4] if len(parts) > 4 else "X",
    }


# ─────────────────────────────────────────────
# USE CASE #10: Technology Tipping Points
# ─────────────────────────────────────────────

def extract_tipping_points():
    """Identify cost scenarios where the optimal resource mix undergoes phase transitions."""
    print("  Extracting tipping points...")
    results = {}

    for iso in ISOS:
        df = load_co_parquet(iso)
        if df is None:
            continue

        iso_results = {}
        for t in ALL_THRESHOLDS:
            sub = df[df["threshold"] == t].copy()
            if len(sub) == 0:
                continue

            # Find the winning (cheapest) mix per scenario
            mix_cols = ["mix_clean_firm", "mix_solar", "mix_wind"]
            storage_cols = [c for c in ["battery_dispatch_pct", "ldes_dispatch_pct",
                                        "h2_dispatch_pct"] if c in sub.columns]

            # Group by each toggle dimension and find where dominant resource changes
            transitions = []

            # For each pair of toggle levels, find if the winning mix changes
            for toggle_name in ["renewable", "firm", "storage", "ldes", "ccs"]:
                # Parse scenarios
                sub["_parsed"] = sub["scenario"].apply(lambda s: parse_scenario(s).get(toggle_name, "M"))

                for level_from, level_to in [("L", "M"), ("M", "H"), ("L", "H")]:
                    from_rows = sub[sub["_parsed"] == level_from]
                    to_rows = sub[sub["_parsed"] == level_to]

                    if len(from_rows) == 0 or len(to_rows) == 0:
                        continue

                    # Find the cheapest mix in each subset
                    best_from = from_rows.loc[from_rows["cost_total_cost"].idxmin()]
                    best_to = to_rows.loc[to_rows["cost_total_cost"].idxmin()]

                    # Check if dominant resource changed
                    from_mix = {c: float(best_from[c]) for c in mix_cols}
                    to_mix = {c: float(best_to[c]) for c in mix_cols}

                    from_dominant = max(from_mix, key=from_mix.get)
                    to_dominant = max(to_mix, key=to_mix.get)

                    if from_dominant != to_dominant:
                        transitions.append({
                            "toggle": toggle_name,
                            "from_level": level_from,
                            "to_level": level_to,
                            "from_dominant": from_dominant.replace("mix_", ""),
                            "to_dominant": to_dominant.replace("mix_", ""),
                            "from_cost": round(float(best_from["cost_total_cost"]), 1),
                            "to_cost": round(float(best_to["cost_total_cost"]), 1),
                            "from_mix": {k.replace("mix_", ""): round(v, 1) for k, v in from_mix.items()},
                            "to_mix": {k.replace("mix_", ""): round(v, 1) for k, v in to_mix.items()},
                        })

                sub.drop(columns=["_parsed"], inplace=True)

            # Also compute the full mix diversity at this threshold
            unique_mixes = sub.drop_duplicates(subset=mix_cols)
            mix_summary = []
            for _, row in unique_mixes.iterrows():
                mix_summary.append({
                    "clean_firm": round(float(row["mix_clean_firm"]), 1),
                    "solar": round(float(row["mix_solar"]), 1),
                    "wind": round(float(row["mix_wind"]), 1),
                    "min_cost": round(float(sub[
                        (sub["mix_clean_firm"] == row["mix_clean_firm"]) &
                        (sub["mix_solar"] == row["mix_solar"]) &
                        (sub["mix_wind"] == row["mix_wind"])
                    ]["cost_total_cost"].min()), 1),
                    "max_cost": round(float(sub[
                        (sub["mix_clean_firm"] == row["mix_clean_firm"]) &
                        (sub["mix_solar"] == row["mix_solar"]) &
                        (sub["mix_wind"] == row["mix_wind"])
                    ]["cost_total_cost"].max()), 1),
                })

            iso_results[str(t)] = {
                "transitions": transitions,
                "n_unique_mixes": len(unique_mixes),
                "mixes": sorted(mix_summary, key=lambda x: x["min_cost"]),
            }

        results[iso] = iso_results

    return results


# ─────────────────────────────────────────────
# USE CASE #11: Value of Flexibility
# ─────────────────────────────────────────────

def extract_flexibility_value():
    """Compare min cost WITH vs WITHOUT each storage type."""
    print("  Extracting flexibility value...")
    results = {}

    for iso in ISOS:
        df = load_co_parquet(iso)
        if df is None:
            continue

        iso_results = {}
        storage_types = {
            "battery_4hr": "battery_dispatch_pct",
            "battery_8hr": "battery8_dispatch_pct",
            "ldes": "ldes_dispatch_pct",
            "green_h2": "h2_dispatch_pct",
        }

        for t in ALL_THRESHOLDS:
            sub = df[df["threshold"] == t]
            if len(sub) == 0:
                continue

            # Medium-cost scenarios only (filter for M in renewable toggle)
            med = sub[sub["scenario"].str.startswith("MMM")]
            if len(med) == 0:
                med = sub  # fallback

            baseline_cost = float(med["cost_total_cost"].min())
            threshold_data = {"baseline_cost": round(baseline_cost, 2), "storage_value": {}}

            for stype, col in storage_types.items():
                if col not in med.columns:
                    continue
                # Cost WITHOUT this storage type
                without = med[med[col] <= 0.0001]
                if len(without) == 0:
                    # All mixes have this storage — value is high
                    threshold_data["storage_value"][stype] = {
                        "cost_with": round(baseline_cost, 2),
                        "cost_without": None,
                        "value_per_mwh": None,
                        "note": "All optimal mixes include this storage"
                    }
                    continue

                cost_without = float(without["cost_total_cost"].min())
                value = cost_without - baseline_cost

                threshold_data["storage_value"][stype] = {
                    "cost_with": round(baseline_cost, 2),
                    "cost_without": round(cost_without, 2),
                    "value_per_mwh": round(value, 2),
                }

            # Also compute across ALL scenarios (L/M/H fan) — vectorized
            fan = {"p10": {}, "p50": {}, "p90": {}}
            for stype, col in storage_types.items():
                if col not in sub.columns:
                    continue
                # Min cost per scenario WITH storage (all mixes)
                all_min = sub.groupby("scenario")["cost_total_cost"].min()
                # Min cost per scenario WITHOUT this storage
                without = sub[sub[col] <= 0.0001]
                if len(without) == 0:
                    continue
                without_min = without.groupby("scenario")["cost_total_cost"].min()
                # Align and compute delta
                common = all_min.index.intersection(without_min.index)
                if len(common) == 0:
                    continue
                deltas = without_min[common] - all_min[common]
                values = deltas[deltas >= 0].values
                if len(values) > 0:
                    fan["p10"][stype] = round(float(np.percentile(values, 10)), 2)
                    fan["p50"][stype] = round(float(np.percentile(values, 50)), 2)
                    fan["p90"][stype] = round(float(np.percentile(values, 90)), 2)

            threshold_data["fan"] = fan
            iso_results[str(t)] = threshold_data

        results[iso] = iso_results

    return results


# ─────────────────────────────────────────────
# USE CASE #1: Data Center Siting
# ─────────────────────────────────────────────

def extract_datacenter_siting():
    """Cross-ISO cost comparison for data center buyers at key thresholds."""
    print("  Extracting data center siting data...")
    results = {"iso_comparison": {}, "cost_curves": {}}

    for iso in ISOS:
        df = load_co_parquet(iso)
        if df is None:
            continue

        iso_data = {}
        cost_curve = []

        for t in ALL_THRESHOLDS:
            sub = df[df["threshold"] == t]
            if len(sub) == 0:
                continue

            # Medium costs
            med = sub[sub["scenario"].str.startswith("MMM")]
            if len(med) == 0:
                med = sub

            best = med.loc[med["cost_total_cost"].idxmin()]

            entry = {
                "threshold": t,
                "cost_total": round(float(best["cost_total_cost"]), 2),
                "cost_incremental": round(float(best["cost_incremental"]), 2) if "cost_incremental" in best.index else None,
                "mix": {
                    "clean_firm": round(float(best["mix_clean_firm"]), 1),
                    "solar": round(float(best["mix_solar"]), 1),
                    "wind": round(float(best["mix_wind"]), 1),
                },
                "storage": {
                    "battery": round(float(best.get("battery_dispatch_pct", 0)), 4),
                    "ldes": round(float(best.get("ldes_dispatch_pct", 0)), 4),
                },
            }

            if t in [90, 95, 99, 99.99]:
                iso_data[str(t)] = entry

            cost_curve.append({"t": t, "cost": entry["cost_total"]})

            # Cost fan (P10/P50/P90 across all scenarios)
            costs = sub["cost_total_cost"]
            entry["cost_p10"] = round(float(costs.quantile(0.10)), 2)
            entry["cost_p50"] = round(float(costs.quantile(0.50)), 2)
            entry["cost_p90"] = round(float(costs.quantile(0.90)), 2)

        results["iso_comparison"][iso] = iso_data
        results["cost_curves"][iso] = cost_curve

    # Rank ISOs at key thresholds
    rankings = {}
    for t_str in ["90", "95", "99"]:
        ranked = []
        for iso in ISOS:
            if iso in results["iso_comparison"] and t_str in results["iso_comparison"][iso]:
                ranked.append({
                    "iso": iso,
                    "cost": results["iso_comparison"][iso][t_str]["cost_total"],
                    "mix": results["iso_comparison"][iso][t_str]["mix"],
                })
        ranked.sort(key=lambda x: x["cost"])
        rankings[t_str] = ranked

    results["rankings"] = rankings

    return results


# ─────────────────────────────────────────────
# USE CASE #6: Carbon Credit Pricing
# ─────────────────────────────────────────────

def extract_carbon_credit_pricing():
    """Reframe MAC curves as carbon credit fair-value pricing."""
    print("  Extracting carbon credit pricing data...")

    # Load existing MAC stats
    mac_path = os.path.join(DATA, "step5-post-processing", "mac_stats.json")
    if not os.path.exists(mac_path):
        print("    WARNING: mac_stats.json not found")
        return {}

    with open(mac_path) as f:
        mac = json.load(f)

    # Load optimal targets for crossover data
    targets_path = os.path.join(DATA, "step5-post-processing", "optimal_targets.json")
    targets = {}
    if os.path.exists(targets_path):
        with open(targets_path) as f:
            targets = json.load(f)

    # Carbon market benchmarks ($/tCO2)
    benchmarks = {
        "voluntary_verra": {"price": 12, "label": "Voluntary (Verra avg)", "type": "voluntary"},
        "voluntary_premium": {"price": 35, "label": "Premium Voluntary", "type": "voluntary"},
        "rggi": {"price": 15, "label": "RGGI (NE US)", "type": "compliance"},
        "eu_ets": {"price": 88, "label": "EU ETS", "type": "compliance"},
        "ca_cap_trade": {"price": 38, "label": "CA Cap-and-Trade", "type": "compliance"},
        "scc_epa": {"price": 190, "label": "SCC (EPA 2024)", "type": "reference"},
        "scc_rennert": {"price": 185, "label": "SCC (Rennert et al.)", "type": "reference"},
        "dac_current": {"price": 600, "label": "DAC (current)", "type": "technology"},
        "dac_target": {"price": 100, "label": "DAC (2050 target)", "type": "technology"},
    }

    results = {
        "mac_fan": mac.get("fan_chart", {}),
        "crossovers": mac.get("crossovers", {}),
        "benchmarks": benchmarks,
        "thresholds": ALL_THRESHOLDS,
        "optimal_targets": {},
    }

    # Extract optimal target per ISO
    for iso in ISOS:
        if iso in targets:
            iso_targets = targets[iso]
            results["optimal_targets"][iso] = {
                "crossover_range": iso_targets.get("crossover_range"),
                "no_regrets": iso_targets.get("no_regrets"),
            }

    # Compute quality premium: hourly-matched vs annual credit value
    # At each threshold, the MAC tells us $/tCO2 for hourly-matched clean energy
    # Annual matching would be cheaper but less rigorous
    quality_premiums = {}
    for iso in ISOS:
        if iso not in results["mac_fan"]:
            continue
        fan = results["mac_fan"][iso]
        if "p50" not in fan:
            continue

        premiums = []
        p50 = fan["p50"]
        for i, t in enumerate(ALL_THRESHOLDS):
            if i < len(p50) and p50[i] is not None:
                # Rough proxy: annual matching MAC ≈ 60% of hourly MAC
                # (annual matching is less strict, so cheaper per tCO2)
                annual_mac = p50[i] * 0.6
                premium = p50[i] - annual_mac
                premiums.append({
                    "threshold": t,
                    "hourly_mac": round(p50[i], 1),
                    "annual_mac_est": round(annual_mac, 1),
                    "quality_premium": round(premium, 1),
                })
        quality_premiums[iso] = premiums

    results["quality_premiums"] = quality_premiums

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def write_js(filename, varname, data):
    """Write data as a JS variable."""
    path = os.path.join(DASH_JS, filename)
    with open(path, "w") as f:
        f.write(f"// Auto-generated by step9e_extract_use_case_data.py\n")
        f.write(f"const {varname} = ")
        json.dump(data, f, indent=2, default=str)
        f.write(";\n")
    size_kb = os.path.getsize(path) / 1024
    print(f"  Wrote {filename} ({size_kb:.1f} KB)")


def main():
    print("Step 9E: Extracting use-case reference page data")
    print("=" * 50)

    tp = extract_tipping_points()
    write_js("tipping-point-data.js", "TIPPING_POINT_DATA", tp)

    fv = extract_flexibility_value()
    write_js("flexibility-value-data.js", "FLEXIBILITY_VALUE_DATA", fv)

    dc = extract_datacenter_siting()
    write_js("datacenter-siting-data.js", "DATACENTER_SITING_DATA", dc)

    cc = extract_carbon_credit_pricing()
    write_js("carbon-credit-data.js", "CARBON_CREDIT_DATA", cc)

    print("\nDone! All 4 JS data files written to dashboard/js/")


if __name__ == "__main__":
    main()
