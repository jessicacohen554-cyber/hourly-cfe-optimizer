#!/usr/bin/env python3
"""
Sensitivity Analysis of Optimizer Results
==========================================
Answers:
1. Which sensitivity toggles most influence final outcomes?
2. What are "no regrets" resources across all sensitivities?
3. Under low demand growth, what's the floor for new wind?
"""

import json
import numpy as np
from collections import defaultdict

# Load results
with open("dashboard/overprocure_results.json") as f:
    DATA = json.load(f)

REGIONS = ["CAISO", "ERCOT", "PJM", "NYISO", "NEISO"]
RESOURCES = ["clean_firm", "solar", "wind", "ccs_ccgt", "hydro"]
# Key thresholds to analyze
KEY_THRESHOLDS = ["75", "80", "85", "90", "95", "97.5", "99", "100"]

# Scenario code: RRR_F_T
# R[0] = Renewable (L/M/H), R[1] = Firm (L/M/H), R[2] = Storage (L/M/H)
# F = Fossil (L/M/H), T = Transmission (N/L/M/H)
TOGGLE_NAMES = ["Renewable Gen", "Firm Gen", "Storage", "Fossil Fuel", "Transmission"]

def parse_scenario_code(code):
    """Parse 'LMH_M_N' -> dict of toggle -> level"""
    parts = code.split("_")
    return {
        "Renewable Gen": parts[0][0],   # L/M/H
        "Firm Gen": parts[0][1],        # L/M/H
        "Storage": parts[0][2],         # L/M/H
        "Fossil Fuel": parts[1],        # L/M/H
        "Transmission": parts[2],       # N/L/M/H
    }

def get_scenarios(region, threshold):
    """Get all 324 scenarios for a region/threshold."""
    try:
        return DATA["results"][region]["thresholds"][threshold]["scenarios"]
    except KeyError:
        return {}


# =============================================================================
# ANALYSIS 1: Toggle sensitivity — which toggles swing outcomes most?
# =============================================================================
print("=" * 80)
print("ANALYSIS 1: SENSITIVITY TOGGLE INFLUENCE ON COST & MIX")
print("=" * 80)

for metric_name, metric_key in [("Effective Cost ($/MWh)", "effective_cost"),
                                  ("Total Cost ($/MWh)", "total_cost")]:
    print(f"\n--- {metric_name} ---")
    print(f"{'Toggle':<18} {'Region':<8} {'Threshold':<10} {'Low→High Swing':<18} {'% of Range':<12}")
    print("-" * 70)

    # For each region & threshold, compute the range attributable to each toggle
    toggle_importance = defaultdict(list)  # toggle -> list of (swing, pct_of_range)

    for region in REGIONS:
        for thresh in ["90", "95", "100"]:  # Focus on key thresholds
            scenarios = get_scenarios(region, thresh)
            if not scenarios:
                continue

            # Get all costs
            all_costs = []
            for code, sc in scenarios.items():
                c = sc.get("costs", {}).get(metric_key)
                if c is not None:
                    all_costs.append(c)

            if not all_costs:
                continue
            total_range = max(all_costs) - min(all_costs)
            if total_range == 0:
                continue

            # For each toggle, compute avg cost at each level
            for toggle in TOGGLE_NAMES:
                level_costs = defaultdict(list)
                for code, sc in scenarios.items():
                    parsed = parse_scenario_code(code)
                    level = parsed[toggle]
                    c = sc.get("costs", {}).get(metric_key)
                    if c is not None:
                        level_costs[level].append(c)

                # Compute mean at each level
                level_means = {lvl: np.mean(vals) for lvl, vals in level_costs.items()}
                if len(level_means) < 2:
                    continue

                swing = max(level_means.values()) - min(level_means.values())
                pct = swing / total_range * 100
                toggle_importance[toggle].append((region, thresh, swing, pct))

    # Summarize: average influence per toggle
    print(f"\n{'Toggle':<18} {'Avg Swing $/MWh':<18} {'Avg % of Range':<16} {'Max Swing':<12}")
    print("-" * 65)
    toggle_avg = {}
    for toggle in TOGGLE_NAMES:
        entries = toggle_importance[toggle]
        if entries:
            avg_swing = np.mean([e[2] for e in entries])
            avg_pct = np.mean([e[3] for e in entries])
            max_swing = max(e[2] for e in entries)
            toggle_avg[toggle] = avg_pct
            print(f"{toggle:<18} ${avg_swing:<17.2f} {avg_pct:<15.1f}% ${max_swing:.2f}")

    # Rank
    ranked = sorted(toggle_avg.items(), key=lambda x: -x[1])
    print(f"\nRanking (most → least influential on {metric_name}):")
    for i, (t, pct) in enumerate(ranked, 1):
        print(f"  {i}. {t} ({pct:.1f}% avg influence)")


# =============================================================================
# ANALYSIS 2: Toggle influence on RESOURCE MIX
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 2: SENSITIVITY TOGGLE INFLUENCE ON RESOURCE MIX")
print("=" * 80)

for resource in RESOURCES:
    toggle_mix_swing = defaultdict(list)

    for region in REGIONS:
        for thresh in ["90", "95", "100"]:
            scenarios = get_scenarios(region, thresh)
            if not scenarios:
                continue

            all_vals = []
            for code, sc in scenarios.items():
                v = sc.get("resource_mix", {}).get(resource)
                if v is not None:
                    all_vals.append(v)

            if not all_vals:
                continue
            total_range = max(all_vals) - min(all_vals)
            if total_range == 0:
                continue

            for toggle in TOGGLE_NAMES:
                level_vals = defaultdict(list)
                for code, sc in scenarios.items():
                    parsed = parse_scenario_code(code)
                    level = parsed[toggle]
                    v = sc.get("resource_mix", {}).get(resource)
                    if v is not None:
                        level_vals[level].append(v)

                level_means = {lvl: np.mean(vals) for lvl, vals in level_vals.items()}
                if len(level_means) < 2:
                    continue
                swing = max(level_means.values()) - min(level_means.values())
                pct = swing / total_range * 100
                toggle_mix_swing[toggle].append((region, thresh, resource, swing, pct))

    print(f"\n--- {resource.upper()} (% of demand) ---")
    print(f"{'Toggle':<18} {'Avg Swing (pp)':<16} {'Avg % of Range':<16}")
    print("-" * 50)
    for toggle in TOGGLE_NAMES:
        entries = [e for e in toggle_mix_swing[toggle] if e[2] == resource]
        if entries:
            avg_swing = np.mean([e[3] for e in entries])
            avg_pct = np.mean([e[4] for e in entries])
            print(f"{toggle:<18} {avg_swing:<15.1f}pp {avg_pct:<15.1f}%")


# =============================================================================
# ANALYSIS 3: "NO REGRETS" RESOURCES — minimum across ALL sensitivities
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 3: 'NO REGRETS' RESOURCES — MINIMUM MIX ACROSS ALL 324 SCENARIOS")
print("=" * 80)
print("(Floor = minimum % of demand for this resource across ALL sensitivity combos)")
print("(If floor > 0, this resource appears in EVERY optimal mix regardless of assumptions)\n")

for region in REGIONS:
    print(f"\n{'='*50}")
    print(f"  {region}")
    print(f"{'='*50}")
    print(f"{'Threshold':<12}", end="")
    for r in RESOURCES:
        print(f"{'Min '+r:<14}", end="")
    print(f"{'Max '+r:<14}" if False else "")
    print(f"{'Threshold':<12} {'clean_firm':<14} {'solar':<14} {'wind':<14} {'ccs_ccgt':<14} {'hydro':<14}  |  {'Median Cost':<14}")
    print("-" * 100)

    for thresh in KEY_THRESHOLDS:
        scenarios = get_scenarios(region, thresh)
        if not scenarios:
            continue

        mins = {}
        maxs = {}
        medians = {}
        costs = []
        for r in RESOURCES:
            vals = [sc["resource_mix"].get(r, 0) for sc in scenarios.values()]
            mins[r] = min(vals) if vals else 0
            maxs[r] = max(vals) if vals else 0
            medians[r] = np.median(vals) if vals else 0

        for sc in scenarios.values():
            c = sc.get("costs", {}).get("effective_cost")
            if c is not None:
                costs.append(c)

        med_cost = np.median(costs) if costs else 0

        parts = []
        for r in RESOURCES:
            if mins[r] > 0:
                parts.append(f"{mins[r]:>3}–{maxs[r]:<3}%    ")
            else:
                parts.append(f"  0–{maxs[r]:<3}%    ")

        print(f"{thresh+'%':<12} {'  '.join(parts)}  |  ${med_cost:.1f}/MWh")


# =============================================================================
# ANALYSIS 4: WIND FLOOR UNDER LOW DEMAND GROWTH
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 4: WIND FLOOR UNDER LOW DEMAND GROWTH (FOSSIL = 'L')")
print("=" * 80)
print("Filtering to scenarios where Fossil Fuel toggle = Low (implies low demand growth)")
print("Shows the MINIMUM wind % across all remaining sensitivities\n")

for region in REGIONS:
    demand_mwh = DATA["results"][region].get("annual_demand_mwh", 0)
    demand_twh = demand_mwh / 1e6

    print(f"\n--- {region} (Demand: {demand_twh:.1f} TWh) ---")
    print(f"{'Threshold':<12} {'Min Wind %':<12} {'Min Wind TWh':<14} {'Max Wind %':<12} {'Max Wind TWh':<14} {'Median Wind %':<14}")
    print("-" * 80)

    for thresh in KEY_THRESHOLDS:
        scenarios = get_scenarios(region, thresh)
        if not scenarios:
            continue

        # Filter: Fossil = Low (low demand growth, low gas prices)
        low_growth_wind = []
        for code, sc in scenarios.items():
            parsed = parse_scenario_code(code)
            if parsed["Fossil Fuel"] == "L":
                w = sc["resource_mix"].get("wind", 0)
                low_growth_wind.append(w)

        if not low_growth_wind:
            continue

        mn = min(low_growth_wind)
        mx = max(low_growth_wind)
        md = np.median(low_growth_wind)

        print(f"{thresh+'%':<12} {mn:<12}% {mn/100*demand_twh:<13.2f} {mx:<12}% {mx/100*demand_twh:<13.2f} {md:<14}%")


# =============================================================================
# ANALYSIS 5: RESOURCE FLOORS BY TOGGLE COMBINATION (detailed wind analysis)
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 5: WIND — TRULY 'NO REGRETS' NEW BUILD (FLOOR ACROSS ALL SCENARIOS)")
print("=" * 80)
print("For each region/threshold: min wind across ALL 324 scenarios = safe new-build floor")
print("'New wind' = wind% minus existing wind share from grid mix\n")

grid_mix = DATA.get("config", {}).get("grid_mix_shares", {})

for region in REGIONS:
    demand_mwh = DATA["results"][region].get("annual_demand_mwh", 0)
    demand_twh = demand_mwh / 1e6
    existing_wind_pct = grid_mix.get(region, {}).get("wind", 0)
    existing_wind_twh = existing_wind_pct / 100 * demand_twh

    print(f"\n--- {region} (Demand: {demand_twh:.1f} TWh, Existing Wind: {existing_wind_pct}% = {existing_wind_twh:.2f} TWh) ---")
    print(f"{'Threshold':<12} {'Min Total Wind':<16} {'Min NEW Wind':<16} {'New Wind TWh':<14} {'Stranding Risk?':<16}")
    print("-" * 76)

    for thresh in KEY_THRESHOLDS:
        scenarios = get_scenarios(region, thresh)
        if not scenarios:
            continue

        wind_vals = [sc["resource_mix"].get("wind", 0) for sc in scenarios.values()]
        min_wind = min(wind_vals)
        min_new_wind = max(0, min_wind - existing_wind_pct)
        new_wind_twh = min_new_wind / 100 * demand_twh

        # Under low demand growth specifically
        low_growth_wind = []
        for code, sc in scenarios.items():
            parsed = parse_scenario_code(code)
            if parsed["Fossil Fuel"] == "L":
                low_growth_wind.append(sc["resource_mix"].get("wind", 0))

        low_min = min(low_growth_wind) if low_growth_wind else 0
        low_new = max(0, low_min - existing_wind_pct)

        risk = "SAFE" if min_new_wind > 0 else "VARIES"
        print(f"{thresh+'%':<12} {min_wind:<15}% {min_new_wind:<15}% {new_wind_twh:<13.2f} {risk:<16} (Low-growth floor: {low_new}% new)")


# =============================================================================
# ANALYSIS 6: SOLAR FLOOR (same analysis)
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 6: SOLAR — 'NO REGRETS' NEW BUILD FLOOR")
print("=" * 80)

for region in REGIONS:
    demand_mwh = DATA["results"][region].get("annual_demand_mwh", 0)
    demand_twh = demand_mwh / 1e6
    existing_solar_pct = grid_mix.get(region, {}).get("solar", 0)
    existing_solar_twh = existing_solar_pct / 100 * demand_twh

    print(f"\n--- {region} (Demand: {demand_twh:.1f} TWh, Existing Solar: {existing_solar_pct}% = {existing_solar_twh:.2f} TWh) ---")
    print(f"{'Threshold':<12} {'Min Total Solar':<16} {'Min NEW Solar':<16} {'New Solar TWh':<14}")
    print("-" * 60)

    for thresh in KEY_THRESHOLDS:
        scenarios = get_scenarios(region, thresh)
        if not scenarios:
            continue

        solar_vals = [sc["resource_mix"].get("solar", 0) for sc in scenarios.values()]
        min_solar = min(solar_vals)
        min_new_solar = max(0, min_solar - existing_solar_pct)
        new_solar_twh = min_new_solar / 100 * demand_twh

        print(f"{thresh+'%':<12} {min_solar:<15}% {min_new_solar:<15}% {new_solar_twh:<13.2f}")


# =============================================================================
# ANALYSIS 7: COST RANGE BY THRESHOLD (to show the "uncertainty funnel")
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS 7: COST UNCERTAINTY RANGE BY THRESHOLD")
print("=" * 80)

for region in REGIONS:
    print(f"\n--- {region} ---")
    print(f"{'Threshold':<12} {'Min Cost':<12} {'P25':<12} {'Median':<12} {'P75':<12} {'Max Cost':<12} {'Range':<12}")
    print("-" * 76)

    for thresh in KEY_THRESHOLDS:
        scenarios = get_scenarios(region, thresh)
        if not scenarios:
            continue

        costs = [sc["costs"]["effective_cost"] for sc in scenarios.values()
                 if sc.get("costs", {}).get("effective_cost") is not None]

        if not costs:
            continue

        arr = np.array(costs)
        print(f"{thresh+'%':<12} ${np.min(arr):<11.1f} ${np.percentile(arr,25):<11.1f} ${np.median(arr):<11.1f} ${np.percentile(arr,75):<11.1f} ${np.max(arr):<11.1f} ${np.max(arr)-np.min(arr):<11.1f}")

print("\n\nDone.")
