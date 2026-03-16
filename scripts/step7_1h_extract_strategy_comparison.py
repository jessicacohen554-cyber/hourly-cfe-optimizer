#!/usr/bin/env python3
"""
Step 7.1H: Extract strategy comparison data for the hourly matching strategy
comparison reference page.

Reads strategy2_hourly.json and wrights_law_curves.json, extracts compact
data for dashboard/js/strategy-comparison-data.js.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from pipeline_config import ISOS

DATA_DIR = "data/step5-scenarios"
WRIGHTS_DIR = "data/step5-wrights"
OUTPUT = "dashboard/js/strategy-comparison-data.js"
STRATEGIES = ["strategy2A", "strategy2B", "strategy2C", "strategy2C_rolloff"]
STRAT_LABELS = {"strategy2A": "2A", "strategy2B": "2B", "strategy2C": "2C", "strategy2C_rolloff": "2C_rolloff"}

# Key thresholds to extract (keep data compact)
KEY_THRESHOLDS = [50, 60, 70, 80, 85, 90, 95, 97.5, 99, 99.9]

# SSS TWh by ISO (from procurement_utils.py)
SSS_TWH = {
    "PJM": 95, "NYISO": 42, "CAISO": 35, "MISO": 30,
    "NEISO": 20, "SPP": 5, "ERCOT": 0
}

# ISO context descriptions
ISO_CONTEXT = {
    "PJM": "Largest ISO. 95 TWh SSS from IL CMC + NJ/PA ZEC nuclear + state hydro. Nuclear-heavy grid where SSS dynamics dominate.",
    "NYISO": "42 TWh SSS from NY ZEC Tier 3 nuclear + NYPA hydro. Strong policy support but limited new-build pipeline.",
    "CAISO": "35 TWh SSS (Diablo Canyon + state hydro). Solar-rich, diverse mix with geothermal + floating wind potential.",
    "MISO": "30 TWh SSS from IL CMC MISO-zone nuclear. Wind-rich region with growing solar, coal retirement driving change.",
    "NEISO": "20 TWh SSS from Millstone CT CCEF nuclear + state hydro. Winter gas constraints add $13/MWh CCS adder.",
    "SPP": "5 TWh SSS from Wolf Creek shared nuclear. Wind-dominant region, minimal existing clean firm.",
    "ERCOT": "0 TWh SSS \u2014 fully deregulated, no state clean mandates. Pure market test where all strategies converge."
}


def load_data():
    with open(os.path.join(DATA_DIR, "strategy2_hourly.json")) as f:
        strategy_data = json.load(f)

    wrights_path = os.path.join(WRIGHTS_DIR, "wrights_law_curves.json")
    wrights_data = None
    if os.path.exists(wrights_path):
        with open(wrights_path) as f:
            wrights_data = json.load(f)

    return strategy_data, wrights_data


def extract_snapshot(entries, target_threshold):
    """Find the entry closest to target_threshold at the latest year with data."""
    best = None
    best_diff = float("inf")
    for e in entries:
        if not e.get("resource_mix"):
            continue
        diff = abs(e["threshold"] - target_threshold)
        if diff < best_diff or (diff == best_diff and e["year"] > (best["year"] if best else 0)):
            best = e
            best_diff = diff
    return best


def categorize_resources(resource_mix):
    """Categorize resource mix into VRE, firm, storage, existing."""
    vre_twh = 0
    firm_twh = 0
    storage_twh = 0
    existing_twh = 0
    sss_twh = 0

    resources = {}
    for k, v in resource_mix.items():
        twh = v.get("twh", 0)
        cat = v.get("category", "")
        price = v.get("price", 0)
        cost_m = v.get("cost_m", 0)

        # Normalize resource names
        rname = k.lower().replace("-", "_")

        if rname in ("solar", "wind", "offshore_wind"):
            vre_twh += twh
        elif rname in ("clean_firm", "ccs", "nuclear_uprate", "geothermal"):
            firm_twh += twh
        elif rname in ("battery", "ldes", "green_h2"):
            storage_twh += twh
        elif rname in ("grid_mix_allocation",):
            existing_twh += twh
        elif rname in ("existing_merchant",):
            existing_twh += twh
        elif rname in ("sss_allocation",):
            sss_twh += twh

        resources[rname] = {"twh": round(twh, 2), "price": round(price, 1), "cost_m": round(cost_m, 1), "cat": cat}

    return {
        "resources": resources,
        "vre_twh": round(vre_twh, 2),
        "firm_twh": round(firm_twh, 2),
        "storage_twh": round(storage_twh, 2),
        "existing_twh": round(existing_twh, 2),
        "sss_twh": round(sss_twh, 2),
    }


def extract_strategy_data(strategy_data):
    """Extract compact comparison data across strategies, ISOs, thresholds."""
    result = {}

    for strat_key in STRATEGIES:
        strat_label = STRAT_LABELS[strat_key]
        strat_data = strategy_data.get(strat_key, {})
        result[strat_label] = {}

        for iso in ISOS:
            iso_data = strat_data.get(iso, {}).get("Medium", {})
            if not iso_data:
                continue

            result[strat_label][iso] = {"by_threshold": {}, "by_participation": {}}

            # Use 10pct participation as the default view (reasonable mid-point)
            default_part = "10pct"
            if default_part not in iso_data:
                # Fallback to first available
                default_part = list(iso_data.keys())[0] if iso_data else None
            if not default_part:
                continue

            entries = iso_data[default_part]

            # Extract by threshold
            for t in KEY_THRESHOLDS:
                snap = extract_snapshot(entries, t)
                if snap:
                    cats = categorize_resources(snap["resource_mix"])
                    result[strat_label][iso]["by_threshold"][str(t)] = {
                        "cost": round(snap["cost_per_mwh"], 1),
                        "total_cost_m": round(snap["total_cost_m"], 1),
                        "co2": round(snap["co2_abated_mmt"], 2),
                        "mac": round(snap["mac_per_ton"], 1) if snap["mac_per_ton"] else None,
                        "year": snap["year"],
                        "threshold": snap["threshold"],
                        **cats,
                    }

            # Extract by participation level (at ~90% threshold)
            for part_key in iso_data:
                part_entries = iso_data[part_key]
                snap = extract_snapshot(part_entries, 90)
                if snap:
                    cats = categorize_resources(snap["resource_mix"])
                    pct_val = float(part_key.replace("pct", "")) / 100
                    result[strat_label][iso]["by_participation"][part_key] = {
                        "pct": pct_val,
                        "cost": round(snap["cost_per_mwh"], 1),
                        "co2": round(snap["co2_abated_mmt"], 2),
                        "mac": round(snap["mac_per_ton"], 1) if snap["mac_per_ton"] else None,
                        **cats,
                    }

    return result


def extract_wrights_data(wrights_data):
    """Extract Wright's Law learning curve data."""
    if not wrights_data:
        return None

    result = {}

    # Figure 8: blended cost at 90% threshold by participation
    fig8 = wrights_data.get("figure8", {})
    if fig8:
        result["figure8"] = {
            "participation_pcts": fig8.get("participation_pcts", []),
            "blended_2c": fig8.get("blended_2c_at_90", []),
            "strat_2a": fig8.get("strat_2a_at_90", []) if "strat_2a_at_90" in fig8 else None,
            "strat_2b": fig8.get("strat_2b_at_90", []) if "strat_2b_at_90" in fig8 else None,
        }

    # Strategy comparison at 90%
    sc90 = wrights_data.get("strategy_comparison_90", {})
    if sc90:
        result["strategy_cost_90"] = {}
        for k, v in sc90.items():
            result["strategy_cost_90"][k] = [round(x, 1) if x else None for x in v] if isinstance(v, list) else v

    # LCOE trajectories
    lcoe = wrights_data.get("lcoe_trajectories", {})
    if lcoe:
        result["lcoe"] = {k: round(v, 1) if isinstance(v, (int, float)) else v for k, v in lcoe.items()}

    # ISO spend at 90%
    iso_spend = wrights_data.get("iso_spend_90", {})
    if iso_spend:
        result["iso_spend_90"] = iso_spend

    return result


def build_national_aggregate(strat_data):
    """Compute demand-weighted national aggregate across ISOs."""
    aggregates = {}

    for strat_label in ["2A", "2B", "2C", "2C_rolloff"]:
        if strat_label not in strat_data:
            continue
        agg = {}
        for t_str in [str(t) for t in KEY_THRESHOLDS]:
            total_weight = 0
            weighted = {"cost": 0, "co2": 0, "vre_twh": 0, "firm_twh": 0,
                        "storage_twh": 0, "existing_twh": 0, "sss_twh": 0}

            for iso in ISOS:
                iso_t = strat_data[strat_label].get(iso, {}).get("by_threshold", {}).get(t_str)
                if not iso_t:
                    continue
                # Use co2 as weight proxy (demand-proportional)
                w = max(abs(iso_t.get("co2", 0)), 0.01)
                total_weight += w
                for k in weighted:
                    weighted[k] += iso_t.get(k, 0) * w

            if total_weight > 0:
                for k in weighted:
                    weighted[k] = round(weighted[k] / total_weight, 2)
            agg[t_str] = weighted

        aggregates[strat_label] = agg

    return aggregates


def main():
    print("Loading data...")
    strategy_data, wrights_data = load_data()

    print("Extracting strategy comparison data...")
    strat_data = extract_strategy_data(strategy_data)

    print("Building national aggregates...")
    national = build_national_aggregate(strat_data)

    print("Extracting Wright's Law data...")
    wrights = extract_wrights_data(wrights_data)

    output = {
        "strategies": strat_data,
        "national": national,
        "wrights": wrights,
        "sss_twh": SSS_TWH,
        "iso_context": ISO_CONTEXT,
        "thresholds": KEY_THRESHOLDS,
    }

    # Write JS file
    js_content = f"// Auto-generated by step7_1h_extract_strategy_comparison.py\n"
    js_content += f"// Generated: {__import__('datetime').datetime.now().isoformat()}\n"
    js_content += f"const STRATEGY_CMP = {json.dumps(output, separators=(',', ':'))};\n"

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write(js_content)

    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"Wrote {OUTPUT} ({size_kb:.1f} KB)")
    print(f"  Strategies: {list(strat_data.keys())}")
    print(f"  ISOs: {ISOS}")
    print(f"  Thresholds: {KEY_THRESHOLDS}")


if __name__ == "__main__":
    main()
