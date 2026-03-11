#!/usr/bin/env python3
"""
Export animation parquets to compact JSON for static dashboard page.

Structure:
- plants[]: static plant info (id, name, lat, lon, fuel, color, capacity)
- scenarios.{id}: per-scenario data
  - hours[]: datetime strings
  - output[][]: plants × hours MW values (2D, rounded)
  - stack.{fuel}[]: aggregated MW per hour
  - demand[]: system demand MW per hour
  - generation[]: total generation per hour
  - renewable[]: renewable MW per hour
  - fossil[]: fossil MW per hour
"""

import os
import json
import numpy as np
import pandas as pd

FRAMES_DIR = "grid_viz/data/frames"
STORIES_DIR = "grid_viz/stories"
OUTPUT_PATH = "dashboard/js/grid-animation-data.js"

SCENARIOS = ["uri_2021", "summer_2023_heat", "normal_spring"]

SCENARIO_META = {
    "uri_2021": {
        "name": "Winter Storm Uri",
        "dateRange": "Feb 10–20, 2021",
        "description": "Record cold snap caused cascading failures. Over 30 GW offline. 4.5M customers lost power.",
        "badgeColor": "danger",
    },
    "summer_2023_heat": {
        "name": "August 2023 Heat Dome",
        "dateRange": "Aug 20–25, 2023",
        "description": "Extreme heat drove demand near 85 GW. Low wind during peak afternoon hours stressed reserves.",
        "badgeColor": "warning",
    },
    "normal_spring": {
        "name": "Normal Spring Day",
        "dateRange": "Apr 15–16, 2023",
        "description": "Typical spring day with moderate demand, strong wind/solar, comfortable reserves.",
        "badgeColor": "success",
    },
}

STACK_ORDER = ["NUCLEAR", "COAL", "GAS", "OIL", "HYDRO", "WIND", "SOLAR", "BIOMASS", "OTHF", "OFSL"]


def load_annotations(scenario_id):
    path = os.path.join(STORIES_DIR, f"{scenario_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def main():
    # --- Build plant roster from first available scenario ---
    first_frames = None
    for sid in SCENARIOS:
        fp = os.path.join(FRAMES_DIR, f"{sid}_frames.parquet")
        if os.path.exists(fp):
            first_frames = pd.read_parquet(fp)
            break

    if first_frames is None:
        print("ERROR: No frame parquets found")
        return

    # Get unique plants from ALL hours (not just hour 0 — solar plants have zero
    # output at night and are missing from hour 0 entirely)
    plant_roster = first_frames.drop_duplicates(subset=["plant_id"], keep="first")[
        ["plant_id", "plant_name", "lat", "lon", "fuel", "fuel_label",
         "capacity_mw", "color_r", "color_g", "color_b"]
    ].copy()
    plant_roster = plant_roster.sort_values("plant_id").reset_index(drop=True)

    # Build plant ID to index mapping
    pid_to_idx = {pid: i for i, pid in enumerate(plant_roster["plant_id"])}
    n_plants = len(plant_roster)

    plants_list = []
    for _, p in plant_roster.iterrows():
        plants_list.append({
            "id": int(p["plant_id"]),
            "name": str(p["plant_name"]),
            "lat": round(float(p["lat"]), 4),
            "lon": round(float(p["lon"]), 4),
            "fuel": str(p["fuel"]),
            "label": str(p["fuel_label"]),
            "cap": round(float(p["capacity_mw"]), 1),
            "r": int(p["color_r"]),
            "g": int(p["color_g"]),
            "b": int(p["color_b"]),
        })

    # --- Process each scenario ---
    scenarios_data = {}

    for sid in SCENARIOS:
        frames_path = os.path.join(FRAMES_DIR, f"{sid}_frames.parquet")
        stack_path = os.path.join(FRAMES_DIR, f"{sid}_stack.parquet")
        demand_path = os.path.join(FRAMES_DIR, f"{sid}_demand.parquet")

        if not all(os.path.exists(p) for p in [frames_path, stack_path, demand_path]):
            print(f"Skipping {sid} — missing files")
            continue

        frames = pd.read_parquet(frames_path)
        stack = pd.read_parquet(stack_path)
        demand = pd.read_parquet(demand_path)

        n_hours = int(frames["hour_index"].max()) + 1
        hours = []

        # Get datetime strings per hour
        for h in range(n_hours):
            hf = frames[frames["hour_index"] == h]
            if len(hf) > 0:
                hours.append(str(hf["datetime"].iloc[0]))
            else:
                hours.append("")

        # Build 2D output array: plants × hours
        output_arr = np.zeros((n_plants, n_hours), dtype=np.float32)
        co2_arr = np.zeros((n_plants, n_hours), dtype=np.float32)

        for _, row in frames.iterrows():
            pid = row["plant_id"]
            h = int(row["hour_index"])
            if pid in pid_to_idx and h < n_hours:
                idx = pid_to_idx[pid]
                mw = row["output_mw"]
                output_arr[idx, h] = mw if not np.isnan(mw) else 0.0
                co2 = row["co2_tons_hr"]
                co2_arr[idx, h] = co2 if not np.isnan(co2) else 0.0

        # Round to 1 decimal for compactness
        output_list = [[round(float(v), 1) for v in row] for row in output_arr]
        # Skip CO2 per-plant for size — compute from stack instead

        # Stack data: fuel → [mw_per_hour]
        stack_dict = {}
        stack_co2_dict = {}
        stack_pivot = stack.pivot_table(
            index="hour_index", columns="fuel", values="total_mw", fill_value=0
        ).reindex(range(n_hours), fill_value=0)

        stack_co2_pivot = stack.pivot_table(
            index="hour_index", columns="fuel", values="total_co2_tons", fill_value=0
        ).reindex(range(n_hours), fill_value=0)

        for fuel in STACK_ORDER:
            if fuel in stack_pivot.columns:
                vals = stack_pivot[fuel].values
                if vals.sum() > 0:
                    stack_dict[fuel] = [round(float(v), 1) for v in vals]
                    co2_vals = stack_co2_pivot[fuel].values if fuel in stack_co2_pivot.columns else np.zeros(n_hours)
                    stack_co2_dict[fuel] = [round(float(v), 1) for v in co2_vals]

        # Demand data
        demand_sorted = demand.sort_values("hour_index")
        demand_list = [round(float(v), 1) for v in demand_sorted["demand_mw"].values]
        gen_list = [round(float(v), 1) for v in demand_sorted["total_generation_mw"].values]
        ren_list = [round(float(v), 1) for v in demand_sorted["renewable_mw"].values]
        fos_list = [round(float(v), 1) for v in demand_sorted["fossil_mw"].values]

        # Annotations
        annotations = load_annotations(sid)

        scenarios_data[sid] = {
            "meta": SCENARIO_META.get(sid, {}),
            "hours": hours,
            "output": output_list,
            "stack": stack_dict,
            "stackCo2": stack_co2_dict,
            "demand": demand_list,
            "generation": gen_list,
            "renewable": ren_list,
            "fossil": fos_list,
            "annotations": annotations,
        }

        print(f"{sid}: {n_hours} hours, {n_plants} plants, output={len(output_list)}×{len(output_list[0])}")

    # --- Write JS file ---
    data = {
        "plants": plants_list,
        "scenarios": scenarios_data,
    }

    js_content = "// Auto-generated by export_animation_json.py — do not edit\n"
    js_content += f"const GRID_ANIMATION_DATA = {json.dumps(data, separators=(',', ':'))};\n"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(js_content)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\nWrote {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
