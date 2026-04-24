#!/usr/bin/env python3
"""Generate ceiling-explorer-data.js from step2.4-renewables-ceiling parquets.

Reads ceiling_summary.parquet and cost_curves_*.parquet, then writes a JS file
that attaches `window.CeilingExplorerData` for use by ceiling_explorer.html.

Usage:
    python scripts/step2_4c_generate_ceiling_explorer_js.py

Outputs:
    dashboard/js/ceiling-explorer-data.js
"""

import json
import math
import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (one level up from scripts/)
DATA_DIR = os.path.join(BASE_DIR, "data", "step2.4-renewables-ceiling")
OUT_PATH = os.path.join(BASE_DIR, "dashboard", "js", "ceiling-explorer-data.js")

ISOS = ["CAISO", "ERCOT", "PJM", "NYISO", "NEISO", "MISO", "SPP"]
OFFSHORE_ISOS = ["CAISO", "NEISO", "NYISO", "PJM"]
YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
LEVELS = ["Low", "Medium", "High"]


def _safe(v):
    """Convert NaN/inf to None for JSON serialization."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(v, 2)


def _safe_int(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return int(round(v))


# ---------------------------------------------------------------------------
# 1. Ceiling summary
# ---------------------------------------------------------------------------
def build_ceiling_summary():
    df = pd.read_parquet(os.path.join(DATA_DIR, "ceiling_summary.parquet"))

    # Scene A: baseline vs ceiling bar chart (2025, Medium, onshore_only)
    med_25 = df[
        (df["year"] == 2025)
        & (df["demand_level"] == "Medium")
        & (df["view"] == "onshore_only")
    ].set_index("iso")
    med_25 = med_25.reindex(ISOS)

    scene_headline = {
        "isos": ISOS,
        "baseline_score_pct": [_safe(med_25.loc[i, "baseline_score_pct"]) for i in ISOS],
        "ceiling_score_pct": [_safe(med_25.loc[i, "ceiling_score_pct"]) for i in ISOS],
        "score_lift_pp": [_safe(med_25.loc[i, "score_lift_pp"]) for i in ISOS],
    }

    # Scene B: capacity + cost bar chart (2025, Medium, onshore_only)
    scene_capacity = {
        "isos": ISOS,
        "added_wind_gw": [_safe_int(med_25.loc[i, "ceiling_added_wind_gw"]) for i in ISOS],
        "added_solar_gw": [_safe_int(med_25.loc[i, "ceiling_added_solar_gw"]) for i in ISOS],
        "annual_cost_usd_med": [_safe(med_25.loc[i, "ceiling_annual_cost_usd_med"]) for i in ISOS],
        "demand_twh": [_safe(med_25.loc[i, "demand_twh"]) for i in ISOS],
    }

    # Scene C: baseline erosion over time (Medium, onshore_only)
    on_med = df[(df["demand_level"] == "Medium") & (df["view"] == "onshore_only")]
    erosion = {}
    for iso in ISOS:
        iso_df = on_med[on_med["iso"] == iso].sort_values("year")
        erosion[iso] = {
            "years": [int(y) for y in iso_df["year"]],
            "baseline_pct": [_safe(v) for v in iso_df["baseline_score_pct"]],
            "ceiling_pct": [_safe(v) for v in iso_df["ceiling_score_pct"]],
            "demand_twh": [_safe(v) for v in iso_df["demand_twh"]],
            "cost_B": [_safe(v / 1e9) for v in iso_df["ceiling_annual_cost_usd_med"]],
        }

    # Scene D: demand trajectory small multiples (all levels)
    on_all = df[df["view"] == "onshore_only"]
    trajectory = {"years": YEARS, "byISO": {}}
    for iso in ISOS:
        iso_data = {}
        for lvl in LEVELS:
            lvl_df = on_all[(on_all["iso"] == iso) & (on_all["demand_level"] == lvl)].sort_values("year")
            iso_data[lvl] = {
                "baseline_pct": [_safe(v) for v in lvl_df["baseline_score_pct"]],
                "ceiling_pct": [_safe(v) for v in lvl_df["ceiling_score_pct"]],
            }
        trajectory["byISO"][iso] = iso_data

    return scene_headline, scene_capacity, erosion, trajectory


# ---------------------------------------------------------------------------
# 2. Cost curves
# ---------------------------------------------------------------------------
def build_cost_curves():
    """Extract cost-frontier data for all ISOs, views, and years."""
    cost_data = {}

    for iso in ISOS:
        fpath = os.path.join(DATA_DIR, f"cost_curves_{iso}.parquet")
        if not os.path.exists(fpath):
            continue
        cc = pd.read_parquet(fpath)

        iso_curves = {}
        views = cc["view"].unique()
        for view in views:
            for year in YEARS:
                for lvl in LEVELS:
                    sub = cc[
                        (cc["frontier_type"] == "joint_optimal")
                        & (cc["view"] == view)
                        & (cc["year"] == year)
                        & (cc["demand_level"] == lvl)
                    ].sort_values("target_score_pct")
                    if sub.empty:
                        continue

                    # Deduplicate: keep only rows where achieved score steps up
                    prev_a = None
                    points = []
                    for _, r in sub.iterrows():
                        a = r["achieved_score_pct"]
                        if prev_a is not None and a <= prev_a:
                            continue
                        points.append({
                            "score": _safe(a),
                            "cost_B": _safe(r["min_annual_cost_med"] / 1e9),
                            "lcoe": _safe(r["effective_lcoe_med"]),
                            "curt_pct": _safe(r["curtailment_rate"] * 100) if r["curtailment_rate"] == r["curtailment_rate"] else 0,
                            "wind_gw": _safe_int(r["optimal_wind_gw"]),
                            "solar_gw": _safe_int(r["optimal_solar_gw"]),
                            "offshore_gw": _safe_int(r.get("optimal_offshore_gw", 0)),
                        })
                        prev_a = a

                    key = f"{view}_{year}_{lvl}"
                    iso_curves[key] = points

        cost_data[iso] = iso_curves

    return cost_data


# ---------------------------------------------------------------------------
# 3. Offshore comparison
# ---------------------------------------------------------------------------
def build_offshore_comparison():
    """For coastal ISOs, compare onshore-only vs onshore+offshore cost frontiers."""
    offshore = {}
    for iso in OFFSHORE_ISOS:
        fpath = os.path.join(DATA_DIR, f"cost_curves_{iso}.parquet")
        if not os.path.exists(fpath):
            continue
        cc = pd.read_parquet(fpath)

        iso_off = {}
        for view in ["onshore_only", "onshore_plus_offshore"]:
            sub = cc[
                (cc["frontier_type"] == "joint_optimal")
                & (cc["view"] == view)
                & (cc["year"] == 2025)
                & (cc["demand_level"] == "Medium")
            ].sort_values("target_score_pct")
            if sub.empty:
                continue

            prev_a = None
            points = []
            for _, r in sub.iterrows():
                a = r["achieved_score_pct"]
                if prev_a is not None and a <= prev_a:
                    continue
                points.append([_safe(a), _safe(r["min_annual_cost_med"] / 1e9)])
                prev_a = a
            iso_off[view] = points

        offshore[iso] = iso_off

    return offshore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Reading parquets from", DATA_DIR)

    headline, capacity, erosion, trajectory = build_ceiling_summary()
    cost_curves = build_cost_curves()
    offshore = build_offshore_comparison()

    output = {
        "meta": {
            "isos": ISOS,
            "offshore_isos": OFFSHORE_ISOS,
            "years": YEARS,
            "levels": LEVELS,
            "source": "scripts/step2_4c_generate_ceiling_explorer_js.py",
            "generated_from": "data/step2.4-renewables-ceiling/",
            "notes": [
                "Existing clean generation frozen at absolute 2025 TWh.",
                "No storage, no clean-firm build — pure wind/solar ceiling.",
                "Cost curves show joint-optimal frontier (cheapest wind/solar mix per CFE target).",
                "Costs are undiscounted 2025 USD, annualized.",
            ],
        },
        "headline": headline,
        "capacity": capacity,
        "erosion": erosion,
        "trajectory": trajectory,
        "cost_curves": cost_curves,
        "offshore": offshore,
    }

    js_str = "// Auto-generated by scripts/step2_4c_generate_ceiling_explorer_js.py\n"
    js_str += "// Do not edit manually. Rerun the generator against parquets.\n\n"
    js_str += "window.CeilingExplorerData = "
    js_str += json.dumps(output, indent=2)
    js_str += ";\n"

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(js_str)

    size_kb = len(js_str) / 1024
    n_cost_keys = sum(len(v) for v in cost_curves.values())
    print(f"Wrote {OUT_PATH} ({size_kb:.0f} KB)")
    print(f"  {len(ISOS)} ISOs, {n_cost_keys} cost-curve series, {len(offshore)} offshore comparisons")


if __name__ == "__main__":
    main()
