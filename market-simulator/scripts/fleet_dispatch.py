#!/usr/bin/env python3
"""
Fleet Dispatch Mapper — Phases 2.2 + 3.2 of the Fleet Scenario Playbook.

Maps grid-level 1,215-scenario sweep results to company-specific fleet emissions
using plant-level heat rates and equity shares. Supports multi-scenario fleet
configurations with year-sensitive CCS ramp, retirements, and new plant additions.

Fully vectorized — no Python for-loops over the 1,215 scenarios.

Usage (single fleet, Phase 2.1 schema):
    python fleet_dispatch.py \
        --fleet fleet_scenarios/sample_fleet.json \
        --sweep results/sweep_1215/sweep_1215_flat.parquet \
        --output results/fleet_results.json

Usage (multi-scenario, Phase 3.1 schema):
    python fleet_dispatch.py \
        --scenarios fleet_scenarios/constellation_scenarios.json \
        --sweep results/sweep_1215/sweep_1215_flat.parquet \
        --output results/fleet_scenario_results.json
"""

import argparse
import copy
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

# Default CO2 rates for deriving co2_rate on new plants
DEFAULT_CO2_RATES = {
    "gas_ccgt": 0.37,
    "gas_ct": 0.55,
    "coal_steam": 0.95,
    "oil_ct": 0.65,
}

# Emission factors (t CO2 / MMBtu) for deriving co2_rate from heat rate
EMISSION_FACTORS = {
    "gas_ccgt": 0.05306,
    "gas_ct": 0.05306,
    "coal_steam": 0.09552,
    "oil_ct": 0.07396,
}


# ---------------------------------------------------------------------------
# CCS ramp schedule
# ---------------------------------------------------------------------------

def ccs_ramp_fraction(year: int, year_online: int, target_rate: float = 0.95) -> float:
    """
    CCS capture ramp: 0% at year_online, 30% by +2yr, 70% by +5yr, 100% by +8yr.
    Returns the effective capture rate (fraction of target_rate) for a given year.

    The ramp is linearly interpolated between milestones:
      year_online + 0yr →   0% of target
      year_online + 2yr →  30% of target
      year_online + 5yr →  70% of target
      year_online + 8yr → 100% of target
    """
    if year < year_online:
        return 0.0

    elapsed = year - year_online
    # Piecewise linear interpolation
    milestones = [(0, 0.0), (2, 0.30), (5, 0.70), (8, 1.00)]
    if elapsed >= milestones[-1][0]:
        return target_rate

    for i in range(len(milestones) - 1):
        t0, f0 = milestones[i]
        t1, f1 = milestones[i + 1]
        if t0 <= elapsed < t1:
            frac = f0 + (f1 - f0) * (elapsed - t0) / (t1 - t0)
            return frac * target_rate

    return target_rate


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fleet(fleet_path: str) -> list[dict]:
    """Load Phase 2.1 fleet JSON and return the plants list with defaults."""
    with open(fleet_path) as f:
        data = json.load(f)

    plants = data.get("plants", data.get("base_fleet", []))
    _apply_plant_defaults(plants)
    return plants


def load_scenario_file(path: str) -> dict:
    """Load Phase 3.1 scenario JSON. Returns the full structure."""
    with open(path) as f:
        data = json.load(f)

    if "base_fleet" not in data or "scenarios" not in data:
        raise ValueError("Scenario file must have 'base_fleet' and 'scenarios' keys.")

    _apply_plant_defaults(data["base_fleet"])
    return data


def _apply_plant_defaults(plants: list[dict]):
    """Apply CCS defaults to a plant list in-place."""
    for p in plants:
        if p.get("status") == "ccs_retrofit":
            p.setdefault("ccs_capture_rate", 0.95)
            p.setdefault("ccs_heat_rate_penalty", 1.14)
        else:
            p.setdefault("ccs_capture_rate", 0.0)
            p.setdefault("ccs_heat_rate_penalty", 1.0)
        p.setdefault("equity_share", 1.0)


def load_sweep(sweep_path: str) -> pd.DataFrame:
    """Load the 1,215-sweep flat parquet."""
    df = pd.read_parquet(sweep_path)
    required = {"iso", "year", "scenario"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sweep parquet missing columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Scenario application (Phase 3.2)
# ---------------------------------------------------------------------------

def apply_scenario(base_fleet: list[dict], modifications: list[dict],
                   years: list[int]) -> list[dict]:
    """
    Apply scenario modifications to a deep copy of base_fleet.

    Each plant gets additional year-aware fields:
      _year_online: year the modification takes effect (None if unmodified)
      _action: "ccs_retrofit" | "retire" | None
      _ccs_target_rate: target capture rate for ramp calculation

    New plants from add_plant are appended with status="pending" and _year_online set.
    """
    fleet = copy.deepcopy(base_fleet)

    # Index by orispl for targeted lookups
    by_orispl: dict[int, list[dict]] = {}
    for p in fleet:
        by_orispl.setdefault(p["orispl"], []).append(p)

    for mod in modifications:
        action = mod["action"]

        if action == "add_plant":
            # Create a new plant entry
            fuel = mod["fuel_type"]
            hr = mod.get("heat_rate_mmbtu_mwh", REFERENCE_HEAT_RATES.get(fuel, 10.0))
            co2 = mod.get("co2_rate_t_mwh")
            if co2 is None:
                ef = EMISSION_FACTORS.get(fuel, 0.05306)
                co2 = round(hr * ef, 5)

            new_plant = {
                "orispl": 0,
                "name": mod["name"],
                "iso": mod["iso"],
                "capacity_mw": mod["capacity_mw"],
                "fuel_type": fuel,
                "heat_rate_mmbtu_mwh": hr,
                "co2_rate_t_mwh": co2,
                "equity_share": mod.get("equity_share", 1.0),
                "status": "operating",
                "ccs_capture_rate": 0.0,
                "ccs_heat_rate_penalty": 1.0,
                "_action": "add_plant",
                "_year_online": mod["year_online"],
                "_ccs_target_rate": 0.0,
            }
            fleet.append(new_plant)
            continue

        # Determine target plants
        targets = []
        if "orispl" in mod:
            targets = by_orispl.get(mod["orispl"], [])
        elif "fuel_type" in mod:
            targets = [p for p in fleet if p["fuel_type"] == mod["fuel_type"]]

        year_online = mod["year_online"]

        for p in targets:
            if action == "retire":
                p["_action"] = "retire"
                p["_year_online"] = year_online
            elif action == "ccs_retrofit":
                p["_action"] = "ccs_retrofit"
                p["_year_online"] = year_online
                p["_ccs_target_rate"] = mod.get("ccs_capture_rate", 0.95)
                p["ccs_heat_rate_penalty"] = mod.get("ccs_heat_rate_penalty", 1.14)

    # Set defaults for unmodified plants
    for p in fleet:
        p.setdefault("_action", None)
        p.setdefault("_year_online", None)
        p.setdefault("_ccs_target_rate", 0.0)

    return fleet


# ---------------------------------------------------------------------------
# Vectorized fleet dispatch — year-aware CCS ramp + retirements
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
    pandas operations.
    """
    fuel_cf = {}
    fuel_margin = {}

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
            cf_arr[si_arr, yi_arr] = iso_df[cf_col].values[valid]
        if margin_col in iso_df.columns:
            margin_arr[si_arr, yi_arr] = iso_df[margin_col].values[valid]

        fuel_cf[fuel] = cf_arr
        fuel_margin[fuel] = margin_arr

    return fuel_cf, fuel_margin


def compute_fleet_emissions_fast(plants: list[dict], sweep_df: pd.DataFrame) -> dict:
    """
    Fully vectorized fleet dispatch with year-aware CCS ramp and retirements.

    Plants with _action="retire" produce zero gen/emissions from _year_online onward.
    Plants with _action="ccs_retrofit" ramp capture rate per ccs_ramp_fraction().
    Plants with _action="add_plant" produce zero gen/emissions before _year_online.

    All operations are vectorized over (n_scenarios, n_years).
    """
    # Separate operating plants from fully static-retired ones
    active_plants = []
    for p in plants:
        if p.get("status") == "retired" and p.get("_action") is None:
            continue  # Permanently retired, skip entirely
        active_plants.append(p)

    # Group by ISO
    iso_plants: dict[str, list[dict]] = {}
    for p in active_plants:
        iso_plants.setdefault(p["iso"], []).append(p)

    years = sorted(sweep_df["year"].unique())
    scenarios = sorted(sweep_df["scenario"].unique())
    n_scenarios = len(scenarios)
    n_years = len(years)
    years_arr = np.array([int(y) for y in years])

    scenario_idx = {s: i for i, s in enumerate(scenarios)}
    year_idx = {y: i for i, y in enumerate(years)}

    fleet_emissions = np.zeros((n_scenarios, n_years), dtype=np.float64)
    plant_arrays = []  # (plant_dict, gen_mwh, emis_mt, cf, status_per_year)

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

            base_cf = fuel_cf[fuel]       # (n_scenarios, n_years)
            margin = fuel_margin[fuel]    # (n_scenarios, n_years)

            # Base CF adjustment
            adjusted_cf = np.minimum(base_cf * efficiency_ratio, 0.95)
            econ_retire = np.isnan(margin) | (margin < 0)
            adjusted_cf = np.where(econ_retire, 0.0, adjusted_cf)
            adjusted_cf = np.where(np.isnan(adjusted_cf), 0.0, adjusted_cf)

            cap_mw = p["capacity_mw"] * p["equity_share"]
            action = p.get("_action")
            year_online = p.get("_year_online")

            # ------ Year-aware masks (broadcast across scenarios) ------
            # Shape: (n_years,) → broadcast to (n_scenarios, n_years)

            if action == "add_plant" and year_online is not None:
                # New plant: zero before year_online
                online_mask = years_arr >= year_online  # (n_years,)
                adjusted_cf = adjusted_cf * online_mask[np.newaxis, :]

            if action == "retire" and year_online is not None:
                # Retired: zero from year_online onward
                active_mask = years_arr < year_online  # (n_years,)
                adjusted_cf = adjusted_cf * active_mask[np.newaxis, :]

            # Generation (MWh)
            gen_mwh = cap_mw * adjusted_cf * 8760.0  # (n_scenarios, n_years)

            # ------ CCS capture (year-varying) ------
            base_co2 = p["co2_rate_t_mwh"]

            if action == "ccs_retrofit" and year_online is not None:
                target_rate = p.get("_ccs_target_rate", 0.95)
                hr_penalty = p.get("ccs_heat_rate_penalty", 1.14)

                # Build per-year capture rate array: (n_years,)
                capture_arr = np.array([
                    ccs_ramp_fraction(int(y), year_online, target_rate)
                    for y in years
                ], dtype=np.float64)

                # Effective CO2 per year: base × hr_penalty × (1 - capture)
                # Shape: (n_years,) → broadcast to (n_scenarios, n_years)
                effective_co2 = base_co2 * hr_penalty * (1.0 - capture_arr)
                effective_co2 = effective_co2[np.newaxis, :]  # (1, n_years)

                # Net generation after parasitic load (also year-varying:
                # only apply penalty in years where CCS is active)
                ccs_active = capture_arr > 0  # (n_years,)
                penalty_arr = np.where(ccs_active, hr_penalty, 1.0)
                gen_mwh = gen_mwh / penalty_arr[np.newaxis, :]

            elif p.get("status") == "ccs_retrofit":
                # Static CCS (from Phase 2.1 schema, no ramp)
                hr_penalty = p.get("ccs_heat_rate_penalty", 1.14)
                capture = p.get("ccs_capture_rate", 0.95)
                effective_co2 = base_co2 * hr_penalty * (1.0 - capture)
                gen_mwh = gen_mwh / hr_penalty
            else:
                effective_co2 = base_co2

            # Emissions (Mt CO2)
            emis_mt = gen_mwh * effective_co2 / 1e6
            fleet_emissions += emis_mt

            # Build per-year status string for plant detail
            status_per_year = []
            for yi, y in enumerate(years):
                y_int = int(y)
                if action == "retire" and year_online and y_int >= year_online:
                    status_per_year.append("retired")
                elif action == "ccs_retrofit" and year_online and y_int >= year_online:
                    status_per_year.append("ccs_retrofit")
                elif action == "add_plant" and year_online and y_int < year_online:
                    status_per_year.append("not_yet_built")
                else:
                    status_per_year.append(p.get("status", "operating"))

            plant_arrays.append((p, gen_mwh, emis_mt, adjusted_cf, status_per_year))

    # -----------------------------------------------------------------------
    # Envelopes
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Plant detail at P50 scenario
    # -----------------------------------------------------------------------
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
        for p, gen_arr, emis_arr, cf_arr, status_arr in plant_arrays:
            year_plants.append({
                "orispl": p["orispl"],
                "name": p["name"],
                "iso": p["iso"],
                "fuel_type": p["fuel_type"],
                "capacity_mw": round(p["capacity_mw"] * p["equity_share"], 1),
                "status": status_arr[yi],
                "capacity_factor": round(float(cf_arr[p50_si, yi]), 4),
                "generation_gwh": round(float(gen_arr[p50_si, yi]) / 1e3, 2),
                "emissions_kt": round(float(emis_arr[p50_si, yi]) * 1e3, 2),
            })
        plant_detail[int(y)] = year_plants

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_capacity_mw = sum(
        p["capacity_mw"] * p["equity_share"]
        for p in active_plants
    )

    return {
        "fleet_summary": {
            "total_plants": len(active_plants),
            "total_capacity_mw": round(total_capacity_mw, 1),
            "isos": sorted(iso_plants.keys()),
            "n_scenarios": n_scenarios,
            "years": [int(y) for y in years],
        },
        "envelope": envelope,
        "plant_detail": plant_detail,
        "_fleet_emissions": fleet_emissions,  # (n_scenarios, n_years) — stripped before JSON output
    }


# ---------------------------------------------------------------------------
# Target trajectories (Phase 3.3)
# ---------------------------------------------------------------------------

def build_target_trajectories(targets_config: dict, years: list[int],
                              baseline_mt: float) -> dict[str, np.ndarray]:
    """
    Build target emissions trajectories from the scenario file's 'targets' block.

    Supported target types:
      - sbti_15c: -4.2% annual linear reduction from baseline_mt starting 2023
      - at_power_nz: company-specific {year: emissions_mt} milestones, linearly
        interpolated between points
      - custom_*: arbitrary {year: emissions_mt} pairs, same interpolation

    Returns {target_name: np.array of emissions_mt for each year in years}.
    """
    trajectories = {}

    for name, spec in targets_config.items():
        ttype = spec.get("type", "custom")

        if ttype == "sbti_15c":
            # -4.2% per year linear reduction from baseline
            rate = spec.get("annual_reduction_pct", 4.2) / 100.0
            base_year = spec.get("base_year", 2023)
            base_val = spec.get("baseline_mt", baseline_mt)
            traj = np.array([
                max(0.0, base_val * (1.0 - rate * (y - base_year)))
                for y in years
            ])
            trajectories[name] = traj

        elif ttype in ("at_power_nz", "custom"):
            # Interpolate from milestone points {year: emissions_mt}
            milestones = spec.get("milestones", {})
            if not milestones:
                continue
            # Sort milestone years
            ms_years = sorted(int(k) for k in milestones.keys())
            ms_vals = [float(milestones[str(y)]) for y in ms_years]

            traj = np.interp(
                years,
                ms_years,
                ms_vals,
                left=ms_vals[0],     # extrapolate flat before first milestone
                right=ms_vals[-1],   # extrapolate flat after last milestone
            )
            trajectories[name] = traj

    return trajectories


def compute_target_analysis(fleet_emissions: np.ndarray, years: list[int],
                            target_traj: np.ndarray) -> dict:
    """
    Compute target gap, achievement year, and probability of meeting target.

    Args:
        fleet_emissions: (n_scenarios, n_years) array of fleet emissions in Mt CO2
        years: list of years matching the column axis
        target_traj: (n_years,) array of target emissions in Mt CO2

    Returns dict with:
        gap_to_target_mt: {year: P50 emissions - target} (positive = above target)
        year_of_target_achievement: first year P50 <= target (null if never)
        probability_of_meeting_target: {year: % of scenarios below target}
    """
    n_scenarios, n_years = fleet_emissions.shape

    # P50 per year
    p50_arr = np.nanmedian(fleet_emissions, axis=0)  # (n_years,)

    # Gap: P50 - target (positive = above, negative = below)
    gap = p50_arr - target_traj  # (n_years,)

    gap_dict = {}
    for yi, y in enumerate(years):
        gap_dict[y] = round(float(gap[yi]), 4)

    # Year of target achievement: first year where P50 <= target
    achievement_year = None
    for yi, y in enumerate(years):
        if p50_arr[yi] <= target_traj[yi] and p50_arr[yi] > 0:
            # Only count if there's actual emissions data (skip 2023 = 0 problem)
            achievement_year = y
            break

    # Probability of meeting target: % of scenarios with emissions <= target
    prob_dict = {}
    for yi, y in enumerate(years):
        col = fleet_emissions[:, yi]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            prob_dict[y] = 0.0
        else:
            pct = float(np.sum(valid <= target_traj[yi]) / len(valid) * 100)
            prob_dict[y] = round(pct, 1)

    return {
        "gap_to_target_mt": gap_dict,
        "year_of_target_achievement": achievement_year,
        "probability_of_meeting_target": prob_dict,
    }


# ---------------------------------------------------------------------------
# Multi-scenario runner (Phase 3.2 + 3.3)
# ---------------------------------------------------------------------------

def run_scenarios(scenario_file: str, sweep_df: pd.DataFrame) -> dict:
    """
    Process all named scenarios from a Phase 3.1 scenario file.
    If 'targets' block is present, computes target analysis per scenario.

    Returns {
      "scenarios": {name: {description, envelope, plant_detail, fleet_summary, targets}},
      "target_trajectories": {name: {year: emissions_mt}},
      "baseline_mt": float
    }
    """
    data = load_scenario_file(scenario_file)
    base_fleet = data["base_fleet"]
    scenarios = data["scenarios"]
    targets_config = data.get("targets", {})
    years = sorted(int(y) for y in sweep_df["year"].unique())

    # Compute baseline_mt: sum of base fleet emissions at a reference year
    # Use the scenario file's configured value, or estimate from fleet
    baseline_mt = data.get("baseline_mt", 0.0)

    # Build target trajectories
    target_trajs = build_target_trajectories(targets_config, years, baseline_mt)

    # Serialize target trajectories for output (numpy → dict)
    target_traj_output = {}
    for tname, traj in target_trajs.items():
        target_traj_output[tname] = {
            "description": targets_config[tname].get("description", ""),
            "values": {y: round(float(traj[yi]), 4) for yi, y in enumerate(years)},
        }

    scenario_results = {}
    for name, spec in scenarios.items():
        description = spec.get("description", "")
        modifications = spec.get("modifications", [])

        fleet = apply_scenario(base_fleet, modifications, years)
        dispatch = compute_fleet_emissions_fast(fleet, sweep_df)
        dispatch["description"] = description

        # Extract raw emissions array for target analysis, then remove from output
        fleet_emissions = dispatch.pop("_fleet_emissions")

        # Compute target analysis for each target trajectory
        if target_trajs:
            targets_analysis = {}
            for tname, traj in target_trajs.items():
                targets_analysis[tname] = compute_target_analysis(
                    fleet_emissions, years, traj
                )
            dispatch["targets"] = targets_analysis

        scenario_results[name] = dispatch

    return {
        "scenarios": scenario_results,
        "target_trajectories": target_traj_output,
        "baseline_mt": baseline_mt,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fleet Dispatch Mapper — map sweep results to company fleet emissions."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fleet",
        help="Path to fleet JSON (Phase 2.1 single-fleet schema)."
    )
    group.add_argument(
        "--scenarios",
        help="Path to scenario JSON (Phase 3.1 multi-scenario schema)."
    )
    parser.add_argument(
        "--sweep", required=True,
        help="Path to sweep_1215_flat.parquet."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for results JSON."
    )
    args = parser.parse_args()

    print(f"Loading sweep: {args.sweep}")
    sweep_df = load_sweep(args.sweep)
    print(f"  {len(sweep_df)} rows ({sweep_df['scenario'].nunique()} scenarios × "
          f"{sweep_df['year'].nunique()} years × {sweep_df['iso'].nunique()} ISOs)")

    if args.fleet:
        # Single fleet mode (Phase 2.2)
        out_path = args.output or "market-simulator/results/fleet_results.json"
        print(f"\nLoading fleet: {args.fleet}")
        plants = load_fleet(args.fleet)
        print(f"  {len(plants)} plants")

        print("Computing fleet dispatch...")
        results = compute_fleet_emissions_fast(plants, sweep_df)

        # Strip internal numpy array before JSON serialization
        results.pop("_fleet_emissions", None)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        _print_envelope(results)
        print(f"\nResults → {out_path}")

    else:
        # Multi-scenario mode (Phase 3.2 + 3.3)
        out_path = args.output or "market-simulator/results/fleet_scenario_results.json"
        print(f"\nLoading scenarios: {args.scenarios}")
        data = load_scenario_file(args.scenarios)
        print(f"  {len(data['base_fleet'])} base plants, "
              f"{len(data['scenarios'])} scenarios")

        results = run_scenarios(args.scenarios, sweep_df)
        scenario_results = results["scenarios"]

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        # Print comparison table
        first_scenario = scenario_results[next(iter(scenario_results))]
        years = first_scenario["fleet_summary"]["years"]
        print(f"\n{'Scenario':<30}  ", end="")
        for y in years:
            print(f"  {y}", end="")
        print("  (P50 Mt CO2)")
        print("-" * (32 + 8 * len(years)))

        for name, res in scenario_results.items():
            print(f"{name:<30}  ", end="")
            for y in years:
                val = res["envelope"].get(y, res["envelope"].get(str(y), {})).get("p50", 0)
                print(f"  {val:>5.1f}", end="")
            print()

        # Print target analysis if present
        if results.get("target_trajectories"):
            print(f"\n--- Target Trajectories ---")
            print(f"Baseline: {results['baseline_mt']:.2f} Mt CO2")
            for tname, tdata in results["target_trajectories"].items():
                print(f"\n  {tname}: {tdata.get('description', '')}")
                print(f"    Values: ", end="")
                for y, v in tdata["values"].items():
                    print(f"  {y}: {v:.2f}", end="")
                print()

            print(f"\n--- Target Achievement (P50 gap in Mt CO2) ---")
            print(f"{'Scenario':<30}  {'Target':<20}  {'Achievement':>12}  ", end="")
            for y in years:
                print(f"  {y:>6}", end="")
            print()

            for sname, sres in scenario_results.items():
                for tname in results["target_trajectories"]:
                    tanalysis = sres.get("targets", {}).get(tname, {})
                    gap = tanalysis.get("gap_to_target_mt", {})
                    ach = tanalysis.get("year_of_target_achievement")
                    ach_str = str(ach) if ach else "never"
                    print(f"{sname:<30}  {tname:<20}  {ach_str:>12}  ", end="")
                    for y in years:
                        g = gap.get(y, gap.get(str(y), 0))
                        print(f"  {g:>6.2f}", end="")
                    print()

        print(f"\nResults → {out_path}")


def _print_envelope(results: dict):
    """Print a single fleet's envelope table."""
    print(f"\n{'Year':>6}  {'P10':>8}  {'P50':>8}  {'P90':>8}  {'Mean':>8}")
    print("-" * 44)
    for y in results["fleet_summary"]["years"]:
        e = results["envelope"].get(y, {})
        print(f"{y:>6}  {e.get('p10', 0):>8.2f}  {e.get('p50', 0):>8.2f}  "
              f"{e.get('p90', 0):>8.2f}  {e.get('mean', 0):>8.2f}")


if __name__ == "__main__":
    main()
