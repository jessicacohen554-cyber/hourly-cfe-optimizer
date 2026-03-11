#!/usr/bin/env python3
"""
Prepare animation frames for ERCOT grid visualization.

Data sources (in priority order):
1. Raw EIA-930 hourly MW by fuel type (grid_viz/data/ercot/raw_generation_*.csv)
   - Actual historical data, fetched via fetch_eia930_raw.py or GitHub workflow
   - Critical for event accuracy (Uri collapse, summer peaks, etc.)
2. Normalized EIA-930 profiles (data/eia-930/eia_generation_profiles.parquet)
   - Fallback: averaged profile shapes × annual generation = hourly MW estimate
   - Smooths out actual events — only use if raw data not available

Also merges:
- eGRID plant locations (lat/lon, fuel type, capacity)
- EIA-930 hourly demand profiles (total ERCOT load in MW)
- EIA-930 fossil fuel mix shares (coal/gas/oil hourly split)

For each scenario day range, produces precomputed DataFrames for animation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SCENARIO_DAYS, FUEL_COLORS_RGB, FRAMES_DIR, ERCOT_DIR

# Emission rates (lbs CO2/MWh) from egrid_emission_rates.json
EMISSION_RATES = {
    "COAL": 2324.8,
    "GAS": 867.4,
    "OIL": 2894.4,
    "BIOMASS": 0.0,
    "NUCLEAR": 0.0,
    "WIND": 0.0,
    "SOLAR": 0.0,
    "HYDRO": 0.0,
    "OTHF": 1180.3,
    "OFSL": 1180.3,
}

# EIA-930 fuel name → eGRID PLFUELCT mapping
EIA_TO_EGRID_FUEL = {
    "wind": "WIND",
    "solar": "SOLAR",
    "nuclear": "NUCLEAR",
    "hydro": "HYDRO",
}


def try_load_raw_generation(scenario_id: str) -> pd.DataFrame | None:
    """Try to load raw (actual) EIA-930 hourly generation data for a scenario.

    Returns DataFrame with columns: datetime_local, date_local, hour_local, fuel, mw
    or None if raw data doesn't exist (fall back to normalized profiles).
    """
    path = os.path.join(ERCOT_DIR, f"raw_generation_{scenario_id}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✓ Using RAW hourly generation data from {path} ({len(df)} records)")
        return df
    return None


def try_load_raw_demand(scenario_id: str) -> pd.DataFrame | None:
    """Try to load raw (actual) EIA-930 hourly demand data for a scenario."""
    path = os.path.join(ERCOT_DIR, f"raw_demand_{scenario_id}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✓ Using RAW hourly demand data from {path} ({len(df)} records)")
        return df
    return None


def load_all_data():
    """Load all required datasets."""
    plants = pd.read_csv("grid_viz/data/egrid/ercot_plants.csv")
    print(f"Plants: {len(plants)}")

    gen_profiles = pd.read_parquet("data/eia-930/eia_generation_profiles.parquet")
    gen_profiles = gen_profiles[gen_profiles["iso"] == "ERCOT"]
    print(f"Gen profiles: {len(gen_profiles)} records")

    demand = pd.read_parquet("data/eia-930/eia_demand_profiles.parquet")
    demand = demand[demand["iso"] == "ERCOT"]
    print(f"Demand: {len(demand)} records")

    fossil_mix = pd.read_parquet("data/eia-930/eia_fossil_mix.parquet")
    fossil_mix = fossil_mix[fossil_mix["iso"] == "ERCOT"]
    print(f"Fossil mix: {len(fossil_mix)} records")

    demand_meta = pd.read_parquet("data/eia-930/eia_demand_meta.parquet")
    demand_meta = demand_meta[demand_meta["iso"] == "ERCOT"]
    print(f"Demand meta: {len(demand_meta)} years")

    return plants, gen_profiles, demand, fossil_mix, demand_meta


def get_annual_gen_by_fuel(plants: pd.DataFrame, year: int, demand_meta: pd.DataFrame) -> dict:
    """
    Get annual generation by fuel type, scaled to the target year.

    eGRID data is 2023. Scale to target year proportionally to demand change.
    """
    # eGRID 2023 annual generation
    egrid_gen = plants.groupby("PLFUELCT")["PLNGENAN"].sum().to_dict()

    # Scale factor: target year demand / 2023 demand
    meta_2023 = demand_meta[demand_meta["year"] == 2023]
    meta_target = demand_meta[demand_meta["year"] == year]

    if len(meta_2023) > 0 and len(meta_target) > 0:
        scale = meta_target["total_annual_mwh"].values[0] / meta_2023["total_annual_mwh"].values[0]
    else:
        scale = 1.0

    return {fuel: gen * scale for fuel, gen in egrid_gen.items()}, scale


def compute_scenario_frames(
    scenario_id: str,
    scenario_info: dict,
    plants: pd.DataFrame,
    gen_profiles: pd.DataFrame,
    demand: pd.DataFrame,
    fossil_mix: pd.DataFrame,
    demand_meta: pd.DataFrame,
) -> tuple:
    """Compute animation frames for one scenario."""

    begin_str, end_str = scenario_info["date_range"]
    begin_date = datetime.strptime(begin_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")
    year = begin_date.year

    year_start = datetime(year, 1, 1)
    start_hour = int((begin_date - year_start).total_seconds() / 3600)
    n_days = (end_date - begin_date).days + 1
    n_hours = n_days * 24
    end_hour = start_hour + n_hours

    print(f"\n=== {scenario_id}: {begin_str} to {end_str} ({n_days} days, {n_hours} hours) ===")

    # Try raw data first (actual historical MW, not averaged profiles)
    raw_gen = try_load_raw_generation(scenario_id)
    raw_demand = try_load_raw_demand(scenario_id)
    use_raw = raw_gen is not None
    if use_raw:
        print(f"  Using RAW data — actual historical generation for {scenario_id}")
    else:
        print(f"  Using NORMALIZED profiles (fallback) — run fetch_eia930_raw.py for actual data")

    # Get annual generation by fuel for this year
    annual_gen, year_scale = get_annual_gen_by_fuel(plants, year, demand_meta)
    print(f"  Year scale factor (vs 2023): {year_scale:.3f}")

    # Build lookup arrays for the scenario hour range
    gen_year = gen_profiles[gen_profiles["year"] == year]
    demand_year = demand[demand["year"] == year]
    fossil_year = fossil_mix[fossil_mix["year"] == year]

    # Pre-extract hourly data into dicts for fast lookup
    # Gen profiles: {fuel: array of 8760 values}
    gen_by_fuel = {}
    for fuel in gen_year["fuel"].unique():
        fuel_data = gen_year[gen_year["fuel"] == fuel].sort_values("hour")
        gen_by_fuel[fuel] = fuel_data["value"].values

    # Demand: array of 8760 MW values
    demand_arr = demand_year.sort_values("hour")["raw_mw"].values

    # Fossil mix: arrays of 8760 share values
    fossil_sorted = fossil_year.sort_values("hour")
    coal_shares = fossil_sorted["coal_share"].values
    gas_shares = fossil_sorted["gas_share"].values
    oil_shares = fossil_sorted["oil_share"].values

    # Plant capacity by fuel type
    fuel_capacity = plants.groupby("PLFUELCT")["NAMEPCAP"].sum().to_dict()

    # Precompute plant arrays by fuel for vectorized allocation
    plant_groups = {}
    for fuel_type in plants["PLFUELCT"].unique():
        fuel_plants = plants[plants["PLFUELCT"] == fuel_type].copy()
        fuel_plants = fuel_plants.reset_index(drop=True)
        total_cap = fuel_plants["NAMEPCAP"].sum()
        plant_groups[fuel_type] = {
            "df": fuel_plants,
            "capacities": fuel_plants["NAMEPCAP"].values,
            "total_cap": total_cap,
            "n": len(fuel_plants),
        }

    # Build all frames
    all_frames = []
    stack_records = []
    demand_records = []

    for h_offset in range(n_hours):
        hour_of_year = start_hour + h_offset
        hour_of_day = hour_of_year % 24
        day_offset = h_offset // 24
        current_date = begin_date + timedelta(days=day_offset)
        dt_str = current_date.strftime("%Y-%m-%d") + f" {hour_of_day:02d}:00"

        # Clamp to array bounds
        h_idx = min(hour_of_year, len(demand_arr) - 1)

        # Demand: prefer raw data, fall back to normalized profiles
        if raw_demand is not None:
            rd = raw_demand[(raw_demand["date_local"] == current_date.strftime("%Y-%m-%d")) &
                           (raw_demand["hour_local"] == hour_of_day)]
            total_demand_mw = rd["demand_mw"].values[0] if len(rd) > 0 else demand_arr[h_idx]
        else:
            total_demand_mw = demand_arr[h_idx]

        # Generation: prefer raw data (actual MW per fuel), fall back to normalized
        if use_raw:
            # RAW PATH: actual historical MW by fuel type for this hour
            hour_gen = raw_gen[(raw_gen["date_local"] == current_date.strftime("%Y-%m-%d")) &
                              (raw_gen["hour_local"] == hour_of_day)]
            all_gen = {}
            for _, row in hour_gen.iterrows():
                fuel = row["fuel"]
                mw = row["mw"] if not pd.isna(row["mw"]) else 0
                all_gen[fuel] = all_gen.get(fuel, 0) + mw
        else:
            # FALLBACK: normalized profiles × annual generation
            renewable_gen = {}
            for eia_fuel, egrid_fuel in EIA_TO_EGRID_FUEL.items():
                if eia_fuel in gen_by_fuel and h_idx < len(gen_by_fuel[eia_fuel]):
                    norm_val = gen_by_fuel[eia_fuel][h_idx]
                    annual_mwh = annual_gen.get(egrid_fuel, 0)
                    hourly_mw = norm_val * annual_mwh
                    renewable_gen[egrid_fuel] = hourly_mw
                else:
                    renewable_gen[egrid_fuel] = 0

            total_renewable = sum(renewable_gen.values())
            fossil_residual = max(0, total_demand_mw - total_renewable)

            if h_idx < len(coal_shares):
                cs, gs, os_ = coal_shares[h_idx], gas_shares[h_idx], oil_shares[h_idx]
            else:
                cs, gs, os_ = 0.27, 0.73, 0.0

            fossil_gen = {
                "COAL": fossil_residual * cs,
                "GAS": fossil_residual * gs,
                "OIL": fossil_residual * os_,
            }
            all_gen = {**renewable_gen, **fossil_gen}

        # Stack summary for this hour
        for fuel, mw in all_gen.items():
            if mw > 0:
                stack_records.append({
                    "hour_index": h_offset,
                    "datetime": dt_str,
                    "fuel": fuel,
                    "total_mw": round(mw, 1),
                    "total_co2_tons": round(mw * EMISSION_RATES.get(fuel, 0) / 2000, 2),
                })

        demand_records.append({
            "hour_index": h_offset,
            "datetime": dt_str,
            "demand_mw": round(total_demand_mw, 1),
            "total_generation_mw": round(sum(all_gen.values()), 1),
            "renewable_mw": round(total_renewable, 1),
            "fossil_mw": round(fossil_residual, 1),
        })

        # Allocate to individual plants
        for fuel_type, total_mw in all_gen.items():
            if fuel_type not in plant_groups or total_mw <= 0:
                continue

            pg = plant_groups[fuel_type]
            # Proportional to capacity, capped at nameplate
            if pg["total_cap"] > 0:
                allocations = pg["capacities"] * (total_mw / pg["total_cap"])
                allocations = np.minimum(allocations, pg["capacities"])
            else:
                continue

            color = FUEL_COLORS_RGB.get(fuel_type, [128, 128, 128])

            for i in range(pg["n"]):
                p = pg["df"].iloc[i]
                output_mw = allocations[i]
                if output_mw < 0.01:
                    continue  # Skip near-zero output

                all_frames.append({
                    "plant_id": int(p["ORISPL"]),
                    "plant_name": p["PNAME"],
                    "lat": p["LAT"],
                    "lon": p["LON"],
                    "fuel": fuel_type,
                    "fuel_label": p.get("FUEL_LABEL", fuel_type),
                    "capacity_mw": p["NAMEPCAP"],
                    "output_mw": round(output_mw, 1),
                    "capacity_factor": round(output_mw / p["NAMEPCAP"], 3) if p["NAMEPCAP"] > 0 else 0,
                    "co2_tons_hr": round(output_mw * EMISSION_RATES.get(fuel_type, 0) / 2000, 2),
                    "datetime": dt_str,
                    "hour_of_day": hour_of_day,
                    "hour_index": h_offset,
                    "color_r": color[0],
                    "color_g": color[1],
                    "color_b": color[2],
                })

        # Add zero-output entries for offline plants (smaller fuels)
        for fuel_type in ["OTHF", "BIOMASS", "OFSL"]:
            if fuel_type not in all_gen and fuel_type in plant_groups:
                pg = plant_groups[fuel_type]
                color = FUEL_COLORS_RGB.get(fuel_type, [128, 128, 128])
                for i in range(pg["n"]):
                    p = pg["df"].iloc[i]
                    all_frames.append({
                        "plant_id": int(p["ORISPL"]),
                        "plant_name": p["PNAME"],
                        "lat": p["LAT"],
                        "lon": p["LON"],
                        "fuel": fuel_type,
                        "fuel_label": p.get("FUEL_LABEL", fuel_type),
                        "capacity_mw": p["NAMEPCAP"],
                        "output_mw": 0.0,
                        "capacity_factor": 0.0,
                        "co2_tons_hr": 0.0,
                        "datetime": dt_str,
                        "hour_of_day": hour_of_day,
                        "hour_index": h_offset,
                        "color_r": color[0],
                        "color_g": color[1],
                        "color_b": color[2],
                    })

    df_frames = pd.DataFrame(all_frames)
    df_stack = pd.DataFrame(stack_records)
    df_demand = pd.DataFrame(demand_records)

    # Summary
    print(f"  Generated {len(df_frames)} plant-hour records")
    print(f"  Active plants: {df_frames[df_frames['output_mw'] > 0]['plant_id'].nunique()}")
    print(f"  Demand range: {df_demand['demand_mw'].min():.0f} - {df_demand['demand_mw'].max():.0f} MW")

    # Per-fuel summary
    for fuel in sorted(df_stack["fuel"].unique()):
        fuel_data = df_stack[df_stack["fuel"] == fuel]
        avg_mw = fuel_data["total_mw"].mean()
        max_mw = fuel_data["total_mw"].max()
        print(f"    {fuel:10s}: avg {avg_mw:8,.0f} MW, peak {max_mw:8,.0f} MW")

    return df_frames, df_stack, df_demand


def save_frames(scenario_id, frames, stack, demand):
    """Save precomputed frames to parquet."""
    os.makedirs(FRAMES_DIR, exist_ok=True)
    frames.to_parquet(os.path.join(FRAMES_DIR, f"{scenario_id}_frames.parquet"), index=False)
    stack.to_parquet(os.path.join(FRAMES_DIR, f"{scenario_id}_stack.parquet"), index=False)
    demand.to_parquet(os.path.join(FRAMES_DIR, f"{scenario_id}_demand.parquet"), index=False)
    print(f"  Saved to {FRAMES_DIR}/{scenario_id}_*.parquet")


def main():
    plants, gen_profiles, demand, fossil_mix, demand_meta = load_all_data()

    for scenario_id, info in SCENARIO_DAYS.items():
        frames, stack, demand_df = compute_scenario_frames(
            scenario_id, info, plants, gen_profiles, demand, fossil_mix, demand_meta,
        )
        save_frames(scenario_id, frames, stack, demand_df)


if __name__ == "__main__":
    main()
