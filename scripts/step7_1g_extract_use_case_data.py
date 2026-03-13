#!/usr/bin/env python3
"""
Step 7.1G: Extract data for novel use-case reference pages.

Produces 4 JS data files:
  - dashboard/js/tipping-point-data.js      (Technology Tipping Points)
  - dashboard/js/flexibility-value-data.js   (Value of Flexibility)
  - dashboard/js/datacenter-siting-data.js   (Data Center Siting)
  - dashboard/js/carbon-credit-data.js       (Carbon Credit Pricing)

All read from existing Step 2.2 parquets + Step 4 analysis JSON.
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
    path = os.path.join(DATA, "step2.2-cost", f"step_2_2a_CO_{iso}.parquet")
    if not os.path.exists(path):
        path = os.path.join(DATA, "step2.2-cost", f"step3_co_{iso}.parquet")
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
    mac_path = os.path.join(DATA, "step4-analysis", "mac_stats.json")
    if not os.path.exists(mac_path):
        print("    WARNING: mac_stats.json not found")
        return {}

    with open(mac_path) as f:
        mac = json.load(f)

    # Load optimal targets for crossover data
    targets_path = os.path.join(DATA, "step4-analysis", "optimal_targets.json")
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
# USE CASE #3: Grid Resilience Stress Testing
# ─────────────────────────────────────────────

def _apply_stress(demand_arr, solar_arr, wind_arr, scenario, H=8760):
    """Apply a parametric stress scenario to hourly profiles.

    Returns modified copies — originals are not mutated.
    """
    d = demand_arr.copy()
    s = solar_arr.copy()
    w = wind_arr.copy()

    if scenario == "summer_heat":
        # +15% demand for 2 weeks in Jul-Aug (hours 4344..4680), -10% wind
        start, end = 4344, 4680
        d[start:end] *= 1.15
        w[start:end] *= 0.90

    elif scenario == "winter_freeze":
        # +20% demand for 1 week in Jan (hours 168..336), -25% solar, -15% wind
        start, end = 168, 336
        d[start:end] *= 1.20
        s[start:end] *= 0.75
        w[start:end] *= 0.85

    elif scenario == "low_wind":
        # -30% wind for 30 days in spring (hours 2160..2880)
        start, end = 2160, 2880
        w[start:end] *= 0.70

    elif scenario == "low_solar":
        # -30% solar for 30 days in winter (hours 0..720)
        start, end = 0, 720
        s[start:end] *= 0.70

    elif scenario == "compound":
        # +15% demand + -20% solar + -20% wind for 1 week in Aug (hours 5088..5256)
        start, end = 5088, 5256
        d[start:end] *= 1.15
        s[start:end] *= 0.80
        w[start:end] *= 0.80

    return d, s, w


def extract_grid_resilience():
    """Score optimized portfolios against synthetic worst-case weather stress scenarios."""
    print("  Extracting grid resilience stress test data...")

    # Import dispatch utilities
    sys.path.insert(0, SCRIPT_DIR)
    from dispatch_utils import load_common_data, get_demand_profile, get_supply_profiles
    from dispatch_utils import build_supply_matrix, reconstruct_hourly_dispatch

    demand_data, gen_profiles, _, _ = load_common_data()

    SCENARIOS = ["summer_heat", "winter_freeze", "low_wind", "low_solar", "compound"]
    SCENARIO_LABELS = {
        "summer_heat": "Summer Heat Wave",
        "winter_freeze": "Winter Freeze",
        "low_wind": "Low Wind Month",
        "low_solar": "Low Solar Month",
        "compound": "Compound Event",
    }
    results = {"scenarios": SCENARIO_LABELS}

    for iso in ISOS:
        df = load_co_parquet(iso)
        if df is None:
            continue

        demand_norm, total_mwh = get_demand_profile(iso, demand_data)
        supply_profiles = get_supply_profiles(iso, gen_profiles)
        supply_matrix = build_supply_matrix(supply_profiles)

        # Extract solar and wind profiles as arrays
        solar_arr = np.array(supply_profiles.get("solar", [0.0] * 8760)[:8760], dtype=np.float64)
        wind_arr = np.array(supply_profiles.get("wind", [0.0] * 8760)[:8760], dtype=np.float64)

        iso_results = {}

        for t in KEY_THRESHOLDS:
            sub = df[df["threshold"] == t]
            if len(sub) == 0:
                continue

            # Find optimal mix at medium cost
            med = sub[sub["scenario"].str.startswith("MMM")]
            if len(med) == 0:
                med = sub
            best = med.loc[med["cost_total_cost"].idxmin()]

            resource_pcts = {
                "clean_firm": float(best["mix_clean_firm"]),
                "solar": float(best["mix_solar"]),
                "wind": float(best["mix_wind"]),
                "offshore_wind": float(best.get("mix_offshore_wind", 0)),
                "ccs_ccgt": float(best.get("mix_ccs_ccgt", 0)),
                "hydro": float(best.get("mix_hydro", 0)),
            }
            battery_pct = float(best.get("battery_dispatch_pct", 0))
            battery8_pct = float(best.get("battery8_dispatch_pct", 0))
            ldes_pct = float(best.get("ldes_dispatch_pct", 0))
            h2_pct = float(best.get("h2_dispatch_pct", 0))

            # Baseline dispatch
            baseline = reconstruct_hourly_dispatch(
                demand_norm, supply_profiles, resource_pcts,
                battery_dispatch_pct=battery_pct, battery8_dispatch_pct=battery8_pct,
                ldes_dispatch_pct=ldes_pct, h2_dispatch_pct=h2_pct,
                supply_matrix=supply_matrix
            )
            baseline_match = demand_norm.copy()
            baseline_match[demand_norm > 0] = np.minimum(
                baseline["total_clean"][demand_norm > 0] / demand_norm[demand_norm > 0], 1.0
            )
            baseline_cfe = float(np.mean(baseline_match) * 100)

            scenario_results = {}
            for scenario in SCENARIOS:
                stressed_d, stressed_s, stressed_w = _apply_stress(
                    demand_norm, solar_arr, wind_arr, scenario
                )
                # Rebuild supply matrix with stressed profiles
                stressed_profiles = dict(supply_profiles)
                stressed_profiles["solar"] = stressed_s.tolist()
                stressed_profiles["wind"] = stressed_w.tolist()
                stressed_matrix = build_supply_matrix(stressed_profiles)

                result = reconstruct_hourly_dispatch(
                    stressed_d, stressed_profiles, resource_pcts,
                    battery_dispatch_pct=battery_pct, battery8_dispatch_pct=battery8_pct,
                    ldes_dispatch_pct=ldes_pct, h2_dispatch_pct=h2_pct,
                    supply_matrix=stressed_matrix
                )

                # Compute metrics
                hourly_match = np.zeros(8760)
                nonzero = stressed_d > 0
                hourly_match[nonzero] = np.minimum(
                    result["total_clean"][nonzero] / stressed_d[nonzero], 1.0
                )
                stressed_cfe = float(np.mean(hourly_match) * 100)

                # Max gap: longest consecutive run of hours with match < 50%
                below_half = hourly_match < 0.5
                max_gap = 0
                current_gap = 0
                for b in below_half:
                    if b:
                        current_gap += 1
                        max_gap = max(max_gap, current_gap)
                    else:
                        current_gap = 0

                # Worst week: min rolling 168-hour average
                if len(hourly_match) >= 168:
                    cumsum = np.concatenate([[0], np.cumsum(hourly_match)])
                    week_avgs = (cumsum[168:] - cumsum[:-168]) / 168.0
                    worst_week = float(np.min(week_avgs) * 100)
                else:
                    worst_week = stressed_cfe

                # Hours below 50% match
                hours_below = int(np.sum(below_half))

                scenario_results[scenario] = {
                    "stressed_cfe": round(stressed_cfe, 1),
                    "resilience_ratio": round(stressed_cfe / max(baseline_cfe, 0.01), 3),
                    "max_gap_hours": max_gap,
                    "worst_week_cfe": round(worst_week, 1),
                    "hours_below_50pct": hours_below,
                }

            iso_results[str(t)] = {
                "baseline_cfe": round(baseline_cfe, 1),
                "mix": {k: round(v, 1) for k, v in resource_pcts.items() if v > 0},
                "scenarios": scenario_results,
            }

        results[iso] = iso_results

    return results


# ─────────────────────────────────────────────
# USE CASE #4: Curtailment Economics
# ─────────────────────────────────────────────

def extract_curtailment_economics():
    """Quantify surplus clean energy and its economic value under different buyer profiles."""
    print("  Extracting curtailment economics data...")

    sys.path.insert(0, SCRIPT_DIR)
    from dispatch_utils import load_common_data, get_demand_profile, get_supply_profiles
    from dispatch_utils import build_supply_matrix, reconstruct_hourly_dispatch
    from pipeline_config import REGIONAL_DEMAND_TWH

    demand_data, gen_profiles, _, _ = load_common_data()

    # Buyer profiles: (label, low_price, high_price) in $/MWh
    BUYER_PROFILES = {
        "green_h2": {"label": "Green Hydrogen", "price_low": 30, "price_high": 60},
        "desal": {"label": "Desalination", "price_low": 15, "price_high": 25},
        "industrial": {"label": "Industrial Heat", "price_low": 20, "price_high": 45},
        "ev_charging": {"label": "EV / Flexible Load", "price_low": 40, "price_high": 80},
    }

    results = {"buyer_profiles": {k: v["label"] for k, v in BUYER_PROFILES.items()}}

    for iso in ISOS:
        df = load_co_parquet(iso)
        if df is None:
            continue

        demand_norm, total_mwh = get_demand_profile(iso, demand_data)
        supply_profiles = get_supply_profiles(iso, gen_profiles)
        supply_matrix = build_supply_matrix(supply_profiles)
        demand_twh = REGIONAL_DEMAND_TWH.get(iso, total_mwh / 1e6)

        iso_results = {}

        for t in ALL_THRESHOLDS:
            sub = df[df["threshold"] == t]
            if len(sub) == 0:
                continue

            med = sub[sub["scenario"].str.startswith("MMM")]
            if len(med) == 0:
                med = sub
            best = med.loc[med["cost_total_cost"].idxmin()]

            resource_pcts = {
                "clean_firm": float(best["mix_clean_firm"]),
                "solar": float(best["mix_solar"]),
                "wind": float(best["mix_wind"]),
                "offshore_wind": float(best.get("mix_offshore_wind", 0)),
                "ccs_ccgt": float(best.get("mix_ccs_ccgt", 0)),
                "hydro": float(best.get("mix_hydro", 0)),
            }

            result = reconstruct_hourly_dispatch(
                demand_norm, supply_profiles, resource_pcts,
                battery_dispatch_pct=float(best.get("battery_dispatch_pct", 0)),
                battery8_dispatch_pct=float(best.get("battery8_dispatch_pct", 0)),
                ldes_dispatch_pct=float(best.get("ldes_dispatch_pct", 0)),
                h2_dispatch_pct=float(best.get("h2_dispatch_pct", 0)),
                supply_matrix=supply_matrix, detailed=True
            )

            # Curtailed energy per hour (normalized)
            curtailed = result["curtailed"]  # (8760,) normalized
            curtailed_mwh = curtailed * total_mwh  # scale to MWh
            total_curtailment_twh = float(np.sum(curtailed_mwh) / 1e6)

            total_clean_mwh = np.sum(result["total_clean"]) * total_mwh
            curtailment_pct = float(np.sum(curtailed_mwh) / max(total_clean_mwh, 1) * 100)
            peak_mw = float(np.max(curtailed_mwh))

            # 24-hour average shape
            hourly_shape = []
            for h in range(24):
                hour_mask = np.arange(8760) % 24 == h
                hourly_shape.append(round(float(np.mean(curtailed_mwh[hour_mask])), 1))

            # Monthly pattern
            month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
            monthly_twh = []
            idx = 0
            for mh in month_hours:
                monthly_twh.append(round(float(np.sum(curtailed_mwh[idx:idx + mh]) / 1e6), 4))
                idx += mh

            # Per-resource surplus breakdown
            resource_surplus = {}
            total_surplus_norm = 0
            for rtype in ["clean_firm", "ccs_ccgt", "solar", "wind", "offshore_wind", "hydro"]:
                key = f"surplus_{rtype}"
                if key in result:
                    surplus_sum = float(np.sum(result[key]))
                    resource_surplus[rtype] = surplus_sum
                    total_surplus_norm += surplus_sum

            by_resource = {}
            if total_surplus_norm > 0:
                for rtype, val in resource_surplus.items():
                    share = val / total_surplus_norm * 100
                    if share > 0.5:
                        by_resource[rtype] = round(share, 1)

            # Value scenarios
            value_scenarios = {}
            for buyer_key, buyer in BUYER_PROFILES.items():
                revenue_low = total_curtailment_twh * 1e6 * buyer["price_low"] / 1e6  # $M
                revenue_high = total_curtailment_twh * 1e6 * buyer["price_high"] / 1e6
                value_scenarios[buyer_key] = {
                    "revenue_low_m": round(revenue_low, 1),
                    "revenue_high_m": round(revenue_high, 1),
                    "price_range": [buyer["price_low"], buyer["price_high"]],
                }

            iso_results[str(t)] = {
                "curtailment_twh": round(total_curtailment_twh, 3),
                "curtailment_pct": round(curtailment_pct, 1),
                "peak_mw": round(peak_mw, 0),
                "hourly_shape": hourly_shape,
                "monthly_twh": monthly_twh,
                "by_resource": by_resource,
                "value_scenarios": value_scenarios,
            }

        results[iso] = iso_results

    return results


# ─────────────────────────────────────────────
# USE CASE #5: Inter-ISO Energy Trading
# ─────────────────────────────────────────────

def extract_inter_iso_trading():
    """Quantify the value of cross-regional clean energy economic netting."""
    print("  Extracting inter-ISO trading data...")

    sys.path.insert(0, SCRIPT_DIR)
    from pipeline_config import REGIONAL_DEMAND_TWH

    # Inter-ISO transmission cost assumptions ($/MWh) — distance-based
    # Adjacent ISOs have lower costs; distant pairs have higher
    TX_COST_MATRIX = {
        ("CAISO", "SPP"): 12, ("CAISO", "ERCOT"): 15, ("CAISO", "MISO"): 18,
        ("CAISO", "PJM"): 22, ("CAISO", "NYISO"): 24, ("CAISO", "NEISO"): 25,
        ("ERCOT", "SPP"): 6, ("ERCOT", "MISO"): 10, ("ERCOT", "PJM"): 16,
        ("ERCOT", "NYISO"): 18, ("ERCOT", "NEISO"): 19,
        ("SPP", "MISO"): 5, ("SPP", "PJM"): 14, ("SPP", "NYISO"): 16,
        ("SPP", "NEISO"): 17,
        ("MISO", "PJM"): 8, ("MISO", "NYISO"): 10, ("MISO", "NEISO"): 12,
        ("PJM", "NYISO"): 4, ("PJM", "NEISO"): 6,
        ("NYISO", "NEISO"): 3,
    }

    def get_tx_cost(iso_a, iso_b):
        if iso_a == iso_b:
            return 0
        key = (min(iso_a, iso_b), max(iso_a, iso_b))
        # Fix ordering — use sorted tuple
        for k, v in TX_COST_MATRIX.items():
            if set(k) == set([iso_a, iso_b]):
                return v
        return 20  # default for unlisted pairs

    # Load min cost at medium scenario per (ISO, threshold)
    island_costs = {}  # {iso: {threshold_str: cost}}
    for iso in ISOS:
        df = load_co_parquet(iso)
        if df is None:
            continue
        iso_costs = {}
        for t in ALL_THRESHOLDS:
            sub = df[df["threshold"] == t]
            if len(sub) == 0:
                continue
            med = sub[sub["scenario"].str.startswith("MMM")]
            if len(med) == 0:
                med = sub
            iso_costs[str(t)] = round(float(med["cost_total_cost"].min()), 2)
        island_costs[iso] = iso_costs

    # Analysis 1: Island vs Pool cost at each threshold
    island_vs_pool = {}
    for t in ALL_THRESHOLDS:
        t_str = str(t)
        costs_at_t = {}
        for iso in ISOS:
            if iso in island_costs and t_str in island_costs[iso]:
                costs_at_t[iso] = island_costs[iso][t_str]

        if len(costs_at_t) < 2:
            continue

        # Pool cost: weighted average of cheapest ISOs + TX
        # Simple model: each ISO can buy from the cheapest available ISO (incl TX)
        pool_costs = {}
        for buyer in costs_at_t:
            best_source_cost = costs_at_t[buyer]  # default: self-supply
            best_source = buyer
            for seller in costs_at_t:
                if seller == buyer:
                    continue
                delivered = costs_at_t[seller] + get_tx_cost(buyer, seller)
                if delivered < best_source_cost:
                    best_source_cost = delivered
                    best_source = seller
            pool_costs[buyer] = {
                "cost": round(best_source_cost, 2),
                "source": best_source,
            }

        # Demand-weighted average
        total_demand = sum(REGIONAL_DEMAND_TWH.get(iso, 0) for iso in costs_at_t)
        if total_demand == 0:
            continue
        island_avg = sum(costs_at_t[iso] * REGIONAL_DEMAND_TWH.get(iso, 0)
                         for iso in costs_at_t) / total_demand
        pool_avg = sum(pool_costs[iso]["cost"] * REGIONAL_DEMAND_TWH.get(iso, 0)
                       for iso in pool_costs) / total_demand
        savings_pct = (island_avg - pool_avg) / max(island_avg, 0.01) * 100

        island_vs_pool[t_str] = {
            "island_costs": costs_at_t,
            "pool_costs": {iso: v["cost"] for iso, v in pool_costs.items()},
            "pool_sources": {iso: v["source"] for iso, v in pool_costs.items()},
            "island_avg": round(island_avg, 2),
            "pool_avg": round(pool_avg, 2),
            "savings_pct": round(savings_pct, 1),
        }

    # Analysis 2: Net exporter/importer position
    net_position = {}
    for iso in ISOS:
        iso_pos = {}
        for t_str, data in island_vs_pool.items():
            if iso not in data.get("pool_sources", {}):
                continue
            # If other ISOs source from this ISO, it's an exporter
            export_count = sum(1 for v in data["pool_sources"].values() if v == iso)
            import_from = data["pool_sources"].get(iso, iso)
            if export_count > 1:
                iso_pos[t_str] = "exporter"
            elif import_from != iso:
                iso_pos[t_str] = "importer"
            else:
                iso_pos[t_str] = "neutral"
        net_position[iso] = iso_pos

    # Analysis 3: Trading pair values
    trading_pairs = {}
    for i, iso_a in enumerate(ISOS):
        for iso_b in ISOS[i + 1:]:
            pair_key = f"{iso_a}_{iso_b}"
            tx = get_tx_cost(iso_a, iso_b)
            pair_values = []
            for t_str in island_vs_pool:
                ic = island_vs_pool[t_str]["island_costs"]
                if iso_a in ic and iso_b in ic:
                    cost_diff = abs(ic[iso_a] - ic[iso_b])
                    net_value = max(0, cost_diff - tx)
                    pair_values.append(net_value)
            if pair_values:
                trading_pairs[pair_key] = {
                    "avg_value": round(float(np.mean(pair_values)), 2),
                    "max_value": round(float(np.max(pair_values)), 2),
                    "tx_cost": tx,
                    "cheaper_iso": min(
                        [iso_a, iso_b],
                        key=lambda x: island_costs.get(x, {}).get("90", 999)
                    ),
                }

    # Analysis 4: TX sensitivity
    tx_sensitivity = {}
    for tx_mult_name, tx_mult in [("low", 0.5), ("medium", 1.0), ("high", 2.0)]:
        savings = []
        for t_str in island_vs_pool:
            ic = island_vs_pool[t_str]["island_costs"]
            total_demand = sum(REGIONAL_DEMAND_TWH.get(iso, 0) for iso in ic)
            if total_demand == 0:
                continue
            island_avg = sum(ic[iso] * REGIONAL_DEMAND_TWH.get(iso, 0)
                             for iso in ic) / total_demand
            # Recompute pool with scaled TX
            pool_sum = 0
            for buyer in ic:
                best = ic[buyer]
                for seller in ic:
                    if seller == buyer:
                        continue
                    delivered = ic[seller] + get_tx_cost(buyer, seller) * tx_mult
                    if delivered < best:
                        best = delivered
                pool_sum += best * REGIONAL_DEMAND_TWH.get(buyer, 0)
            pool_avg = pool_sum / total_demand
            s = (island_avg - pool_avg) / max(island_avg, 0.01) * 100
            savings.append(s)
        tx_sensitivity[tx_mult_name] = {
            "avg_savings_pct": round(float(np.mean(savings)) if savings else 0, 1),
            "tx_multiplier": tx_mult,
        }

    return {
        "island_vs_pool": island_vs_pool,
        "net_position": net_position,
        "trading_pairs": trading_pairs,
        "tx_sensitivity": tx_sensitivity,
    }


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
    print("Step 7.1G: Extracting use-case reference page data")
    print("=" * 50)

    tp = extract_tipping_points()
    write_js("tipping-point-data.js", "TIPPING_POINT_DATA", tp)

    fv = extract_flexibility_value()
    write_js("flexibility-value-data.js", "FLEXIBILITY_VALUE_DATA", fv)

    dc = extract_datacenter_siting()
    write_js("datacenter-siting-data.js", "DATACENTER_SITING_DATA", dc)

    cc = extract_carbon_credit_pricing()
    write_js("carbon-credit-data.js", "CARBON_CREDIT_DATA", cc)

    gr = extract_grid_resilience()
    write_js("grid-resilience-data.js", "GRID_RESILIENCE_DATA", gr)

    ce = extract_curtailment_economics()
    write_js("curtailment-economics-data.js", "CURTAILMENT_ECONOMICS_DATA", ce)

    it = extract_inter_iso_trading()
    write_js("inter-iso-trading-data.js", "INTER_ISO_TRADING_DATA", it)

    print("\nDone! All 7 JS data files written to dashboard/js/")


if __name__ == "__main__":
    main()
