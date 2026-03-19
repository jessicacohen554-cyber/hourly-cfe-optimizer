#!/usr/bin/env python3
"""
Fleet Dispatch Mapper — Phase 2.2 of the Fleet Scenario Playbook.

Maps grid-level 405-scenario sweep results to company-specific fleet emissions
using plant-level heat rates and equity shares. Fully vectorized — no Python
for-loops over scenarios.

Usage:
    python fleet_dispatch.py \
        --fleet market-simulator/fleet_scenarios/sample_fleet.json \
        --sweep market-simulator/results/sweep_405/sweep_405_flat.parquet \
        --output market-simulator/results/fleet_results.json

Input:
    - Fleet JSON (Phase 2.1 schema): per-plant orispl, capacity, heat rate, etc.
    - Sweep parquet (Phase 1): 17,010 rows (405 scenarios × 7 ISOs × 6 years)
      with ge_{fuel}_cf, ge_{fuel}_margin_mwh columns.

Output:
    - fleet_results.json: P10/P50/P90 emissions envelopes + plant-level detail.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas required. Install with: pip install pandas pyarrow", file=sys.stderr)
    sys.exit(1)

# Reference heat rates from lmp_engine.py — used to compute efficiency adjustment
REFERENCE_HEAT_RATES = {
    "gas_ccgt": 7.0,
    "gas_ct": 10.5,
    "coal_steam": 10.0,
    "oil_ct": 10.5,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fleet(fleet_path: str) -> list[dict]:
    """Load fleet JSON and return the plants list with defaults applied."""
    with open(fleet_path) as f:
        data = json.load(f)

    plants = data["plants"]
    for p in plants:
        # Apply CCS defaults for retrofit plants
        if p.get("status") == "ccs_retrofit":
            p.setdefault("ccs_capture_rate", 0.95)
            p.setdefault("ccs_heat_rate_penalty", 1.14)
        else:
            p["ccs_capture_rate"] = 0.0
            p["ccs_heat_rate_penalty"] = 1.0
    return plants


def load_sweep(sweep_path: str) -> pd.DataFrame:
    """Load the 405-sweep flat parquet."""
    df = pd.read_parquet(sweep_path)
    required = {"iso", "year", "scenario"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sweep parquet missing columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Vectorized fleet dispatch
# ---------------------------------------------------------------------------

def compute_fleet_emissions(plants: list[dict], sweep_df: pd.DataFrame) -> dict:
    """
    Vectorized fleet dispatch computation.

    For each plant, we:
      1. Filter sweep rows to the plant's ISO
      2. Look up fleet-average CF and margin for the plant's fuel type
      3. Adjust CF by heat-rate efficiency ratio (better plant → higher CF)
      4. Zero out generation if margin < 0 (economic retirement)
      5. Apply CCS capture rate and equity share
      6. Sum emissions across plants → fleet total per (scenario, year)
      7. Compute P10/P50/P90 across 405 scenarios for each year

    Returns a dict with envelopes and plant-level detail.
    """
    # Group plants by ISO to minimize DataFrame filtering
    iso_plants: dict[str, list[dict]] = {}
    for p in plants:
        if p["status"] == "retired":
            continue  # Skip retired plants entirely
        iso_plants.setdefault(p["iso"], []).append(p)

    # Unique (scenario, year) pairs — these become our index
    years = sorted(sweep_df["year"].unique())
    scenarios = sorted(sweep_df["scenario"].unique())
    n_scenarios = len(scenarios)
    n_years = len(years)

    # Build scenario-to-index mapping for fast lookup
    scenario_idx = {s: i for i, s in enumerate(scenarios)}
    year_idx = {y: i for i, y in enumerate(years)}

    # Result array: (n_scenarios, n_years) — fleet emissions in Mt CO2
    fleet_emissions = np.zeros((n_scenarios, n_years), dtype=np.float64)

    # Plant-level detail: keyed by year → list of plant dicts (at P50 scenario)
    # We'll fill this after computing envelopes
    plant_arrays = []  # list of (plant_dict, gen_array, emis_array) for detail

    for iso, iso_plant_list in iso_plants.items():
        # Filter sweep to this ISO
        iso_df = sweep_df[sweep_df["iso"] == iso].copy()
        if iso_df.empty:
            continue

        # Build (n_scenarios, n_years) arrays for each fuel type's CF and margin
        # Index: scenario_idx[s] × year_idx[y]
        fuel_cf_arrays: dict[str, np.ndarray] = {}
        fuel_margin_arrays: dict[str, np.ndarray] = {}

        fuel_types_needed = {p["fuel_type"] for p in iso_plant_list}

        for fuel in fuel_types_needed:
            cf_col = f"ge_{fuel}_cf"
            margin_col = f"ge_{fuel}_margin_mwh"

            cf_arr = np.full((n_scenarios, n_years), np.nan, dtype=np.float64)
            margin_arr = np.full((n_scenarios, n_years), np.nan, dtype=np.float64)

            for _, row in iso_df.iterrows():
                si = scenario_idx.get(row["scenario"])
                yi = year_idx.get(row["year"])
                if si is not None and yi is not None:
                    cf_val = row.get(cf_col)
                    margin_val = row.get(margin_col)
                    if pd.notna(cf_val):
                        cf_arr[si, yi] = cf_val
                    if pd.notna(margin_val):
                        margin_arr[si, yi] = margin_val

            fuel_cf_arrays[fuel] = cf_arr
            fuel_margin_arrays[fuel] = margin_arr

        # Now compute per-plant emissions vectorized over (scenarios, years)
        for p in iso_plant_list:
            fuel = p["fuel_type"]
            ref_hr = REFERENCE_HEAT_RATES.get(fuel, 10.0)
            plant_hr = p["heat_rate_mmbtu_mwh"]

            # Efficiency ratio: lower heat rate = more efficient = dispatched more
            # A plant with HR 6.0 vs fleet avg 7.0 gets CF * (7.0/6.0) = +17%
            # Capped at 1.5× to prevent unrealistic CFs
            efficiency_ratio = min(ref_hr / plant_hr, 1.5) if plant_hr > 0 else 1.0

            # Get fleet-avg CF and margin arrays for this fuel
            base_cf = fuel_cf_arrays[fuel]       # (n_scenarios, n_years)
            margin = fuel_margin_arrays[fuel]     # (n_scenarios, n_years)

            # Adjust CF by efficiency ratio, cap at 0.95
            adjusted_cf = np.minimum(base_cf * efficiency_ratio, 0.95)

            # Economic retirement: zero out if margin < 0
            # NaN margins → treat as no data → zero generation (conservative)
            retired_mask = np.isnan(margin) | (margin < 0)
            adjusted_cf = np.where(retired_mask, 0.0, adjusted_cf)

            # Also zero NaN CFs
            adjusted_cf = np.where(np.isnan(adjusted_cf), 0.0, adjusted_cf)

            # Capacity (equity-adjusted)
            cap_mw = p["capacity_mw"] * p["equity_share"]

            # Generation (MWh) = capacity × CF × 8760
            gen_mwh = cap_mw * adjusted_cf * 8760.0  # (n_scenarios, n_years)

            # CO2 rate adjustment for CCS
            base_co2 = p["co2_rate_t_mwh"]
            if p["status"] == "ccs_retrofit":
                # CCS increases heat rate (parasitic load) but captures most CO2
                hr_penalty = p["ccs_heat_rate_penalty"]
                capture = p["ccs_capture_rate"]
                # Effective CO2 = base_rate × hr_penalty × (1 - capture)
                effective_co2 = base_co2 * hr_penalty * (1.0 - capture)
                # Also adjust generation for parasitic load (lower net output)
                gen_mwh = gen_mwh / hr_penalty  # net generation after CCS parasitic
            else:
                effective_co2 = base_co2

            # Emissions (Mt CO2) = generation × CO2 rate / 1e6
            emis_mt = gen_mwh * effective_co2 / 1e6  # (n_scenarios, n_years)

            # Accumulate into fleet total
            fleet_emissions += emis_mt

            # Store for plant-level detail
            plant_arrays.append((p, gen_mwh, emis_mt, adjusted_cf))

    # -----------------------------------------------------------------------
    # Compute P10/P50/P90 envelopes across scenarios
    # -----------------------------------------------------------------------
    envelope = {}
    for yi, y in enumerate(years):
        year_emissions = fleet_emissions[:, yi]  # (n_scenarios,)
        valid = year_emissions[~np.isnan(year_emissions)]
        if len(valid) == 0:
            envelope[int(y)] = {"p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0}
        else:
            envelope[int(y)] = {
                "p10": round(float(np.percentile(valid, 10)), 4),
                "p50": round(float(np.percentile(valid, 50)), 4),
                "p90": round(float(np.percentile(valid, 90)), 4),
                "mean": round(float(np.mean(valid)), 4),
            }

    # -----------------------------------------------------------------------
    # Plant-level detail at P50 scenario
    # -----------------------------------------------------------------------
    # Find the P50 scenario for each year (the one closest to median emissions)
    plant_detail = {}
    for yi, y in enumerate(years):
        year_emissions = fleet_emissions[:, yi]
        valid_mask = ~np.isnan(year_emissions)
        if not valid_mask.any():
            continue
        median_val = np.median(year_emissions[valid_mask])
        # Find scenario index closest to median
        diffs = np.abs(year_emissions - median_val)
        diffs[~valid_mask] = np.inf
        p50_si = int(np.argmin(diffs))

        year_plants = []
        for p, gen_arr, emis_arr, cf_arr in plant_arrays:
            year_plants.append({
                "orispl": p["orispl"],
                "name": p["name"],
                "iso": p["iso"],
                "fuel_type": p["fuel_type"],
                "capacity_mw": round(p["capacity_mw"] * p["equity_share"], 1),
                "status": p["status"],
                "capacity_factor": round(float(cf_arr[p50_si, yi]), 4),
                "generation_gwh": round(float(gen_arr[p50_si, yi]) / 1e3, 2),
                "emissions_kt": round(float(emis_arr[p50_si, yi]) * 1e3, 2),
            })
        plant_detail[int(y)] = year_plants

    # -----------------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------------
    total_capacity_mw = sum(
        p["capacity_mw"] * p["equity_share"]
        for p in plants if p["status"] != "retired"
    )

    return {
        "fleet_summary": {
            "total_plants": len([p for p in plants if p["status"] != "retired"]),
            "total_capacity_mw": round(total_capacity_mw, 1),
            "isos": sorted(iso_plants.keys()),
            "n_scenarios": n_scenarios,
            "years": [int(y) for y in years],
        },
        "envelope": envelope,
        "plant_detail": plant_detail,
    }


# ---------------------------------------------------------------------------
# Optimized sweep loading — vectorize the ISO loop fill
# ---------------------------------------------------------------------------

def _build_fuel_arrays_fast(
    iso_df: pd.DataFrame,
    fuel_types: set[str],
    scenario_idx: dict,
    year_idx: dict,
    n_scenarios: int,
    n_years: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Build (n_scenarios, n_years) arrays for CF and margin using vectorized
    pandas operations instead of iterrows().
    """
    fuel_cf = {}
    fuel_margin = {}

    # Map scenario and year to integer indices
    si_arr = iso_df["scenario"].map(scenario_idx).values
    yi_arr = iso_df["year"].map(year_idx).values
    valid = ~(np.isnan(si_arr) | np.isnan(yi_arr))  # type: ignore[arg-type]
    si_arr = si_arr[valid].astype(np.intp)
    yi_arr = yi_arr[valid].astype(np.intp)

    for fuel in fuel_types:
        cf_col = f"ge_{fuel}_cf"
        margin_col = f"ge_{fuel}_margin_mwh"

        cf_arr = np.full((n_scenarios, n_years), np.nan, dtype=np.float64)
        margin_arr = np.full((n_scenarios, n_years), np.nan, dtype=np.float64)

        if cf_col in iso_df.columns:
            vals = iso_df[cf_col].values[valid]
            cf_arr[si_arr, yi_arr] = vals

        if margin_col in iso_df.columns:
            vals = iso_df[margin_col].values[valid]
            margin_arr[si_arr, yi_arr] = vals

        fuel_cf[fuel] = cf_arr
        fuel_margin[fuel] = margin_arr

    return fuel_cf, fuel_margin


def compute_fleet_emissions_fast(plants: list[dict], sweep_df: pd.DataFrame) -> dict:
    """
    Fully vectorized fleet dispatch — replaces iterrows() with numpy indexing.
    """
    iso_plants: dict[str, list[dict]] = {}
    for p in plants:
        if p["status"] == "retired":
            continue
        iso_plants.setdefault(p["iso"], []).append(p)

    years = sorted(sweep_df["year"].unique())
    scenarios = sorted(sweep_df["scenario"].unique())
    n_scenarios = len(scenarios)
    n_years = len(years)

    scenario_idx = {s: i for i, s in enumerate(scenarios)}
    year_idx = {y: i for i, y in enumerate(years)}

    fleet_emissions = np.zeros((n_scenarios, n_years), dtype=np.float64)
    plant_arrays = []

    for iso, iso_plant_list in iso_plants.items():
        iso_df = sweep_df[sweep_df["iso"] == iso]
        if iso_df.empty:
            continue

        fuel_types_needed = {p["fuel_type"] for p in iso_plant_list}
        fuel_cf, fuel_margin = _build_fuel_arrays_fast(
            iso_df, fuel_types_needed, scenario_idx, year_idx, n_scenarios, n_years
        )

        for p in iso_plant_list:
            fuel = p["fuel_type"]
            ref_hr = REFERENCE_HEAT_RATES.get(fuel, 10.0)
            plant_hr = p["heat_rate_mmbtu_mwh"]
            efficiency_ratio = min(ref_hr / plant_hr, 1.5) if plant_hr > 0 else 1.0

            base_cf = fuel_cf[fuel]
            margin = fuel_margin[fuel]

            adjusted_cf = np.minimum(base_cf * efficiency_ratio, 0.95)
            retired_mask = np.isnan(margin) | (margin < 0)
            adjusted_cf = np.where(retired_mask, 0.0, adjusted_cf)
            adjusted_cf = np.where(np.isnan(adjusted_cf), 0.0, adjusted_cf)

            cap_mw = p["capacity_mw"] * p["equity_share"]
            gen_mwh = cap_mw * adjusted_cf * 8760.0

            base_co2 = p["co2_rate_t_mwh"]
            if p["status"] == "ccs_retrofit":
                hr_penalty = p["ccs_heat_rate_penalty"]
                capture = p["ccs_capture_rate"]
                effective_co2 = base_co2 * hr_penalty * (1.0 - capture)
                gen_mwh = gen_mwh / hr_penalty
            else:
                effective_co2 = base_co2

            emis_mt = gen_mwh * effective_co2 / 1e6
            fleet_emissions += emis_mt
            plant_arrays.append((p, gen_mwh, emis_mt, adjusted_cf))

    # Envelopes
    envelope = {}
    for yi, y in enumerate(years):
        col = fleet_emissions[:, yi]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            envelope[int(y)] = {"p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0}
        else:
            envelope[int(y)] = {
                "p10": round(float(np.percentile(valid, 10)), 4),
                "p50": round(float(np.percentile(valid, 50)), 4),
                "p90": round(float(np.percentile(valid, 90)), 4),
                "mean": round(float(np.mean(valid)), 4),
            }

    # Plant detail at P50 scenario
    plant_detail = {}
    for yi, y in enumerate(years):
        col = fleet_emissions[:, yi]
        valid_mask = ~np.isnan(col)
        if not valid_mask.any():
            continue
        median_val = np.median(col[valid_mask])
        diffs = np.abs(col - median_val)
        diffs[~valid_mask] = np.inf
        p50_si = int(np.argmin(diffs))

        year_plants = []
        for p, gen_arr, emis_arr, cf_arr in plant_arrays:
            year_plants.append({
                "orispl": p["orispl"],
                "name": p["name"],
                "iso": p["iso"],
                "fuel_type": p["fuel_type"],
                "capacity_mw": round(p["capacity_mw"] * p["equity_share"], 1),
                "status": p["status"],
                "capacity_factor": round(float(cf_arr[p50_si, yi]), 4),
                "generation_gwh": round(float(gen_arr[p50_si, yi]) / 1e3, 2),
                "emissions_kt": round(float(emis_arr[p50_si, yi]) * 1e3, 2),
            })
        plant_detail[int(y)] = year_plants

    total_capacity_mw = sum(
        p["capacity_mw"] * p["equity_share"]
        for p in plants if p["status"] != "retired"
    )

    return {
        "fleet_summary": {
            "total_plants": len([p for p in plants if p["status"] != "retired"]),
            "total_capacity_mw": round(total_capacity_mw, 1),
            "isos": sorted(iso_plants.keys()),
            "n_scenarios": n_scenarios,
            "years": [int(y) for y in years],
        },
        "envelope": envelope,
        "plant_detail": plant_detail,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fleet Dispatch Mapper — map 405-sweep results to company fleet emissions."
    )
    parser.add_argument(
        "--fleet", required=True,
        help="Path to fleet JSON file (Phase 2.1 schema)."
    )
    parser.add_argument(
        "--sweep", required=True,
        help="Path to sweep_405_flat.parquet."
    )
    parser.add_argument(
        "--output", default="market-simulator/results/fleet_results.json",
        help="Output path for fleet results JSON."
    )
    args = parser.parse_args()

    print(f"Loading fleet: {args.fleet}")
    plants = load_fleet(args.fleet)
    print(f"  → {len(plants)} plants loaded")

    print(f"Loading sweep: {args.sweep}")
    sweep_df = load_sweep(args.sweep)
    print(f"  → {len(sweep_df)} rows ({sweep_df['scenario'].nunique()} scenarios × "
          f"{sweep_df['year'].nunique()} years × {sweep_df['iso'].nunique()} ISOs)")

    print("Computing fleet dispatch (vectorized)...")
    results = compute_fleet_emissions_fast(plants, sweep_df)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")

    # Print summary
    print("\n=== Fleet Emissions Envelope (Mt CO2) ===")
    print(f"{'Year':>6}  {'P10':>8}  {'P50':>8}  {'P90':>8}  {'Mean':>8}")
    print("-" * 44)
    for y in results["fleet_summary"]["years"]:
        e = results["envelope"][str(y)] if str(y) in results["envelope"] else results["envelope"].get(y, {})
        print(f"{y:>6}  {e.get('p10', 0):>8.4f}  {e.get('p50', 0):>8.4f}  "
              f"{e.get('p90', 0):>8.4f}  {e.get('mean', 0):>8.4f}")


if __name__ == "__main__":
    main()
