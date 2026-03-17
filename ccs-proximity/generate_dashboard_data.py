#!/usr/bin/env python3
"""
Generate CCS proximity dashboard data for the frontend.

Reads the viability ranking CSV and GeoJSON spatial files, pre-computes CCS
reduction math, and outputs a single JS file for the interactive map dashboard.

Output: dashboard/js/ccs-proximity-data.js
"""

import json
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_JS = os.path.join(REPO_DIR, "dashboard", "js", "ccs-proximity-data.js")

# CCS assumptions
HEAT_RATE_PENALTY = 0.15   # 15% more fuel to run CCS
CAPTURE_RATE = 0.95        # 95% carbon capture
CAPACITY_FACTOR = 0.80     # assumed going-forward CF

# Short tons → million metric tons conversion
ST_TO_MMT = 1 / 1_000_000 * 0.907185  # 1 short ton = 0.907185 metric tons


def load_geojson(path):
    """Load a GeoJSON file and return the features list."""
    with open(path) as f:
        data = json.load(f)
    return data.get("features", [])


def extract_coords_from_feature(feature):
    """Extract coordinate arrays from a GeoJSON feature geometry."""
    geom = feature.get("geometry", {})
    return geom.get("coordinates", [])


def main():
    print("Generating CCS proximity dashboard data...")

    # ── Load plant ranking ──
    ranking_path = os.path.join(SCRIPT_DIR, "output", "ccus_viability_ranking.csv")
    df = pd.read_csv(ranking_path)
    print(f"  Loaded {len(df)} plants from viability ranking")

    # ── Pre-compute CCS math per plant ──
    # baseline_co2: actual 2023 equity-adjusted emissions (short tons)
    # post_ccs_co2: residual after 15% heat rate penalty + 95% capture
    # co2_reduction: baseline - residual
    df["baseline_co2_st"] = df["PLCO2AN_EQUITY"]
    df["post_ccs_co2_st"] = df["baseline_co2_st"] * (1 + HEAT_RATE_PENALTY) * (1 - CAPTURE_RATE)
    df["co2_reduction_st"] = df["baseline_co2_st"] - df["post_ccs_co2_st"]

    # Convert to million metric tons for display
    df["baseline_mmt"] = (df["baseline_co2_st"] * ST_TO_MMT).round(4)
    df["post_ccs_mmt"] = (df["post_ccs_co2_st"] * ST_TO_MMT).round(4)
    df["reduction_mmt"] = (df["co2_reduction_st"] * ST_TO_MMT).round(4)

    total_baseline_mmt = df["baseline_mmt"].sum()
    total_post_ccs_mmt = df["post_ccs_mmt"].sum()
    print(f"  Fleet baseline: {total_baseline_mmt:.1f} MMT CO2")
    print(f"  Fleet post-CCS: {total_post_ccs_mmt:.1f} MMT CO2 (all 39 plants)")
    print(f"  Fleet reduction: {total_baseline_mmt - total_post_ccs_mmt:.1f} MMT CO2")

    # ── Build plant records ──
    plants = []
    for _, row in df.iterrows():
        plants.append({
            "orispl": int(row["ORISPL"]),
            "name": str(row["PNAME"]),
            "fleetName": str(row.get("FLEET_NAME", "")),
            "state": str(row["PSTATABB"]),
            "iso": str(row["ISO"]),
            "lat": round(float(row["LAT"]), 5),
            "lon": round(float(row["LON"]), 5),
            "namepcap_mw": round(float(row["NAMEPCAP"]), 1),
            "ownedCapacity_mw": round(float(row.get("OWNED_CAPACITY_MW", row["NAMEPCAP"])), 1),
            "equityShare": round(float(row.get("EQUITY_SHARE", 1.0)), 3),
            "baselineCO2_mmt": round(float(row["baseline_mmt"]), 4),
            "postCcsCO2_mmt": round(float(row["post_ccs_mmt"]), 4),
            "reductionCO2_mmt": round(float(row["reduction_mmt"]), 4),
            "viabilityScore": round(float(row["viability_score"]), 1),
            "scoreEmissions": round(float(row["score_emissions"]), 1),
            "scorePipeline": round(float(row["score_pipeline"]), 1),
            "scoreWell": round(float(row["score_well"]), 1),
            "scoreFormation": round(float(row["score_formation"]), 1),
            "distPipeline_km": round(float(row["dist_pipeline_km"]), 1),
            "distWell_km": round(float(row["dist_well_km"]), 1),
            "withinFormation": bool(row["within_formation"]),
            "rank": int(row["rank"]),
        })

    # ── Load GeoJSON layers ──
    # Pipelines
    pipeline_features = load_geojson(
        os.path.join(SCRIPT_DIR, "data", "co2_pipelines.geojson")
    )
    pipelines = []
    for f in pipeline_features:
        props = f.get("properties", {})
        pipelines.append({
            "name": props.get("name", "Unknown"),
            "operator": props.get("operator", ""),
            "miles": props.get("miles", 0),
            "coords": extract_coords_from_feature(f),
        })
    print(f"  Loaded {len(pipelines)} pipeline segments")

    # Wells
    well_features = load_geojson(
        os.path.join(SCRIPT_DIR, "data", "class_vi_wells.geojson")
    )
    wells = []
    for f in well_features:
        props = f.get("properties", {})
        coords = extract_coords_from_feature(f)
        wells.append({
            "name": props.get("name", "Unknown"),
            "state": props.get("state", ""),
            "status": props.get("status", "Pending"),
            "operator": props.get("operator", ""),
            "formation": props.get("formation", ""),
            "lat": coords[1] if len(coords) >= 2 else 0,
            "lon": coords[0] if len(coords) >= 2 else 0,
        })
    print(f"  Loaded {len(wells)} Class VI wells")

    # Formations
    formation_features = load_geojson(
        os.path.join(SCRIPT_DIR, "data", "saline_formations.geojson")
    )
    formations = []
    for f in formation_features:
        props = f.get("properties", {})
        formations.append({
            "name": props.get("name", "Unknown"),
            "type": props.get("type", ""),
            "storageCapacity_gt": props.get("storage_gt_co2", 0),
            "coords": extract_coords_from_feature(f),
        })
    print(f"  Loaded {len(formations)} saline formations")

    # ── Assemble output ──
    data = {
        "metadata": {
            "generatedAt": pd.Timestamp.now().isoformat(),
            "plantCount": len(plants),
            "totalBaselineMmt": round(total_baseline_mmt, 2),
            "totalPostCcsMmt": round(total_post_ccs_mmt, 2),
            "totalReductionMmt": round(total_baseline_mmt - total_post_ccs_mmt, 2),
            "totalCapacityMw": round(df["NAMEPCAP"].sum(), 0),
            "topScore": round(df["viability_score"].max(), 1),
            "assumptions": {
                "heatRatePenalty": HEAT_RATE_PENALTY,
                "captureRate": CAPTURE_RATE,
                "capacityFactor": CAPACITY_FACTOR,
                "note": "Baseline emissions are 2023 actuals (equity-adjusted). "
                        "CCS adds 15% heat rate penalty, captures 95% of gross emissions. "
                        "80% capacity factor assumed for forward projections."
            }
        },
        "plants": plants,
        "pipelines": pipelines,
        "wells": wells,
        "formations": formations,
    }

    # ── Write JS ──
    os.makedirs(os.path.dirname(OUTPUT_JS), exist_ok=True)
    js_content = "// Auto-generated by generate_dashboard_data.py — do not edit\n"
    js_content += "const CCS_DATA = " + json.dumps(data, indent=2) + ";\n"

    with open(OUTPUT_JS, "w") as f:
        f.write(js_content)

    size_kb = os.path.getsize(OUTPUT_JS) / 1024
    print(f"\n  Output: {OUTPUT_JS} ({size_kb:.1f} KB)")
    print("  Done.")


if __name__ == "__main__":
    main()
