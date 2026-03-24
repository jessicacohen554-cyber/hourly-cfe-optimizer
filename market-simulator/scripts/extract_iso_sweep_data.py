#!/usr/bin/env python3
"""
Extract ISO market sweep data for the CEG ISO Market Sweep dashboard page.

Reads sweep_1215_flat.parquet and constellation_scenarios.json, computes:
1. P10/P25/P50/P75/P90 envelopes per ISO/year for ~25 metrics
2. Sensitivity decomposition (variance contribution of each sweep dimension)
3. CEG asset archetype bundles (how Constellation assets perform under scenario clusters)
4. Fleet reference data

Outputs: dashboard/js/iso-sweep-data.js
"""
import json
import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "results" / "sweep_1215" / "sweep_1215_flat.parquet"
FLEET_JSON = ROOT / "frontend" / "data" / "constellation_scenarios.json"
OUTPUT = ROOT.parent / "dashboard" / "js" / "iso-sweep-data.js"

# Metrics to compute envelopes for
ENVELOPE_METRICS = [
    "avg_lmp", "lmp_p90", "clean_pct", "demand_twh",
    "emissions_mt", "emission_rate_tco2_mwh",
    "cost_per_mwh", "revenue_per_mwh",
    "energy_rev_mwh", "capacity_rev_mwh", "rec_rev_mwh",
    "total_new_fossil_mw", "total_economic_retirement_mw",
    "ordc_scarcity_hours",
    # Resource mix
    "mix_solar_twh", "mix_wind_twh", "mix_offshore_wind_twh",
    "mix_ccs_ccgt_twh", "mix_hydro_twh", "mix_clean_firm_twh",
    # Generator economics — fleet aggregate
    "ge_gas_ccgt_cf", "ge_gas_ccgt_margin_mwh", "ge_gas_ccgt_capacity_mw",
    "ge_gas_ct_cf", "ge_gas_ct_margin_mwh",
    "ge_coal_steam_cf", "ge_coal_steam_margin_mwh", "ge_coal_steam_capacity_mw",
    # Generator economics — per heat-rate tier (5 tiers: very_low → very_high)
    "ge_gas_ccgt_very_low_cf", "ge_gas_ccgt_very_low_margin_mwh", "ge_gas_ccgt_very_low_capacity_mw",
    "ge_gas_ccgt_low_cf", "ge_gas_ccgt_low_margin_mwh", "ge_gas_ccgt_low_capacity_mw",
    "ge_gas_ccgt_medium_cf", "ge_gas_ccgt_medium_margin_mwh", "ge_gas_ccgt_medium_capacity_mw",
    "ge_gas_ccgt_high_cf", "ge_gas_ccgt_high_margin_mwh", "ge_gas_ccgt_high_capacity_mw",
    "ge_gas_ccgt_very_high_cf", "ge_gas_ccgt_very_high_margin_mwh", "ge_gas_ccgt_very_high_capacity_mw",
    "ge_coal_steam_very_low_cf", "ge_coal_steam_low_cf", "ge_coal_steam_medium_cf",
    "ge_coal_steam_high_cf", "ge_coal_steam_very_high_cf",
    # Nuclear
    "nuc_energy_rev_mwh", "nuc_capacity_rev_mwh", "nuc_ptc_mwh", "nuc_total_mwh",
    # CCS breakeven
    "ccs_be_ccs_vs_existing_carbon_price", "ccs_be_new_gas_vs_old_carbon_price",
    "ccs_be_ccs_lcoe",
    # Retirements by fuel
    "ret_gas_ccgt_mw", "ret_gas_ct_mw", "ret_oil_ct_mw", "ret_coal_steam_mw",
    # New builds
    "nb_gas_ccgt_mw", "nb_gas_ct_mw",
    # Emissions by fuel
    "emis_gas_ccgt_mt", "emis_coal_mt", "emis_oil_mt",
]

# CEG asset metrics for archetype analysis
CEG_METRICS = [
    "nuc_total_mwh", "nuc_energy_rev_mwh", "nuc_capacity_rev_mwh",
    "ge_gas_ccgt_cf", "ge_gas_ccgt_margin_mwh", "ge_gas_ccgt_capacity_mw",
    # Per heat-rate tier CCGT CFs (Constellation fleet dispatch fidelity)
    "ge_gas_ccgt_very_low_cf", "ge_gas_ccgt_low_cf", "ge_gas_ccgt_medium_cf",
    "ge_gas_ccgt_high_cf", "ge_gas_ccgt_very_high_cf",
    "ge_gas_ct_cf", "ge_gas_ct_margin_mwh",
    "emissions_mt", "clean_pct", "avg_lmp",
    "ret_gas_ccgt_mw", "ret_gas_ct_mw",
    "total_economic_retirement_mw",
]

SENSITIVITY_METRICS = [
    "avg_lmp", "clean_pct", "emissions_mt",
    "ge_gas_ccgt_cf", "nuc_total_mwh", "total_economic_retirement_mw",
]

PERCENTILES = [10, 25, 50, 75, 90]
PRICE_BUNDLES = ["all_low", "all_med", "all_high", "high_vre_low_firm", "high_firm_low_vre"]

# Key years for archetype deep-dives (reduce data size)
KEY_YEARS = [2023, 2025, 2030, 2035, 2040, 2045, 2050]


def parse_scenario_id(sid: str) -> dict:
    """Parse scenario_id like MKT_L_all_low_L_L_L_L into dimensions."""
    # Format: MKT_{demand}_{price_sensitivity}_{ppa}_{gas_friction}_{queue}_{new_fossil}
    parts = sid.split("_")
    # demand is parts[1]
    demand = parts[1]
    # new_fossil is parts[-1], queue is parts[-2], gas_friction is parts[-3], ppa is parts[-4]
    new_fossil = parts[-1]
    queue = parts[-2]
    gas_friction = parts[-3]
    ppa = parts[-4]
    # price_sensitivity is everything between parts[2] and parts[-4]
    price_parts = parts[2:-4]
    price_sensitivity = "_".join(price_parts)
    return {
        "demand_growth": demand,
        "price_sensitivity": price_sensitivity,
        "ppa_level": ppa,
        "gas_friction": gas_friction,
        "queue_capacity": queue,
        "new_fossil_cost": new_fossil,
    }


def compute_envelopes(df: pd.DataFrame) -> dict:
    """Compute P10/P25/P50/P75/P90 envelopes per ISO/year."""
    print("  Computing envelopes...")
    result = {}
    available_metrics = [m for m in ENVELOPE_METRICS if m in df.columns]

    for iso, iso_df in df.groupby("iso"):
        result[iso] = {}
        for year, ydf in iso_df.groupby("year"):
            year_data = {}
            for metric in available_metrics:
                vals = ydf[metric].dropna()
                if len(vals) == 0:
                    continue
                year_data[metric] = {
                    f"p{p}": round(float(np.percentile(vals, p)), 4)
                    for p in PERCENTILES
                }
                year_data[metric]["mean"] = round(float(vals.mean()), 4)
            result[iso][str(year)] = year_data

    return result


def compute_sensitivity(df: pd.DataFrame, dims_df: pd.DataFrame) -> dict:
    """Compute variance decomposition: how much each sweep dimension explains."""
    print("  Computing sensitivity decomposition...")
    merged = df.join(dims_df)
    dim_cols = list(dims_df.columns)
    result = {}

    available_metrics = [m for m in SENSITIVITY_METRICS if m in df.columns]

    for iso, iso_df in merged.groupby("iso"):
        result[iso] = {}
        for year in KEY_YEARS:
            ydf = iso_df[iso_df["year"] == year]
            if len(ydf) < 10:
                continue
            year_data = {}
            for metric in available_metrics:
                vals = ydf[metric].dropna()
                total_var = vals.var()
                if total_var == 0 or np.isnan(total_var):
                    continue
                dim_contributions = {}
                for dim in dim_cols:
                    # Between-group variance / total variance
                    group_means = ydf.groupby(dim)[metric].mean()
                    grand_mean = vals.mean()
                    n_per_group = ydf.groupby(dim)[metric].count()
                    between_var = float(
                        ((group_means - grand_mean) ** 2 * n_per_group).sum()
                        / len(vals)
                    )
                    dim_contributions[dim] = round(between_var / total_var, 4)
                year_data[metric] = dim_contributions
            result[iso][str(year)] = year_data

    return result


def compute_ceg_archetypes(df: pd.DataFrame, dims_df: pd.DataFrame) -> dict:
    """Group scenarios by price bundle and demand level, compute CEG asset stats."""
    print("  Computing CEG asset archetypes...")
    merged = df.join(dims_df)
    available_metrics = [m for m in CEG_METRICS if m in df.columns]
    result = {}

    for iso, iso_df in merged.groupby("iso"):
        iso_result = {"price_bundles": {}, "demand_bundles": {}}

        for year in KEY_YEARS:
            ydf = iso_df[iso_df["year"] == year]
            if len(ydf) < 10:
                continue

            # By price sensitivity bundle
            for bundle, bdf in ydf.groupby("price_sensitivity"):
                if bundle not in iso_result["price_bundles"]:
                    iso_result["price_bundles"][bundle] = {}
                bundle_year = {}
                for metric in available_metrics:
                    vals = bdf[metric].dropna()
                    if len(vals) == 0:
                        continue
                    bundle_year[metric] = {
                        "p10": round(float(np.percentile(vals, 10)), 4),
                        "p50": round(float(np.percentile(vals, 50)), 4),
                        "p90": round(float(np.percentile(vals, 90)), 4),
                        "mean": round(float(vals.mean()), 4),
                    }
                iso_result["price_bundles"][bundle][str(year)] = bundle_year

            # By demand growth level
            for dlevel, ddf in ydf.groupby("demand_growth"):
                label = {"L": "low", "M": "medium", "H": "high"}.get(dlevel, dlevel)
                if label not in iso_result["demand_bundles"]:
                    iso_result["demand_bundles"][label] = {}
                demand_year = {}
                for metric in available_metrics:
                    vals = ddf[metric].dropna()
                    if len(vals) == 0:
                        continue
                    demand_year[metric] = {
                        "p10": round(float(np.percentile(vals, 10)), 4),
                        "p50": round(float(np.percentile(vals, 50)), 4),
                        "p90": round(float(np.percentile(vals, 90)), 4),
                        "mean": round(float(vals.mean()), 4),
                    }
                iso_result["demand_bundles"][label][str(year)] = demand_year

        result[iso] = iso_result

    return result


def compute_supply_stacks(df: pd.DataFrame, dims_df: pd.DataFrame) -> dict:
    """Compute median supply stack (generation TWh by resource) per price bundle per year."""
    print("  Computing supply stacks...")
    merged = df.join(dims_df)
    mix_cols = [c for c in df.columns if c.startswith("mix_") and c.endswith("_twh")]
    result = {}

    for iso, iso_df in merged.groupby("iso"):
        iso_result = {}
        for bundle, bdf in iso_df.groupby("price_sensitivity"):
            bundle_data = {}
            for year in KEY_YEARS:
                ydf = bdf[bdf["year"] == year]
                if len(ydf) < 5:
                    continue
                stack = {}
                for col in mix_cols:
                    resource = col.replace("mix_", "").replace("_twh", "")
                    vals = ydf[col].dropna()
                    if len(vals) > 0:
                        stack[resource] = {
                            "p50": round(float(np.percentile(vals, 50)), 2),
                            "p10": round(float(np.percentile(vals, 10)), 2),
                            "p90": round(float(np.percentile(vals, 90)), 2),
                        }
                bundle_data[str(year)] = stack
            iso_result[bundle] = bundle_data
        result[iso] = iso_result

    return result


def compute_outcome_drivers(df: pd.DataFrame, dims_df: pd.DataFrame) -> dict:
    """Identify which scenario combos drive extreme outcomes for CEG assets."""
    print("  Computing outcome drivers...")
    merged = df.join(dims_df)
    dim_cols = list(dims_df.columns)
    result = {}

    outcomes = {
        "high_nuc_revenue": ("nuc_total_mwh", "top"),
        "low_gas_cf": ("ge_gas_ccgt_cf", "bottom"),
        "low_emissions": ("emissions_mt", "bottom"),
        "high_clean_pct": ("clean_pct", "top"),
        "high_retirements": ("total_economic_retirement_mw", "top"),
    }

    for iso, iso_df in merged.groupby("iso"):
        iso_result = {}
        for year in [2030, 2035, 2040, 2050]:
            ydf = iso_df[iso_df["year"] == year]
            if len(ydf) < 50:
                continue
            year_result = {}
            for outcome_name, (metric, direction) in outcomes.items():
                if metric not in ydf.columns:
                    continue
                vals = ydf[metric].dropna()
                if len(vals) < 50:
                    continue
                # Get top/bottom 20% of scenarios
                threshold = np.percentile(vals, 20 if direction == "bottom" else 80)
                if direction == "bottom":
                    extreme = ydf[ydf[metric] <= threshold]
                else:
                    extreme = ydf[ydf[metric] >= threshold]

                # What dimensions are overrepresented in extreme scenarios?
                driver_profile = {}
                for dim in dim_cols:
                    dim_dist = extreme[dim].value_counts(normalize=True).to_dict()
                    baseline_dist = ydf[dim].value_counts(normalize=True).to_dict()
                    # Enrichment ratio: how much more frequent is each value in extreme vs baseline
                    enrichment = {}
                    for val in dim_dist:
                        base = baseline_dist.get(val, 0.01)
                        enrichment[val] = round(dim_dist[val] / base, 3) if base > 0 else 0
                    driver_profile[dim] = enrichment

                year_result[outcome_name] = {
                    "threshold": round(float(threshold), 4),
                    "n_scenarios": int(len(extreme)),
                    "drivers": driver_profile,
                }
            iso_result[str(year)] = year_result
        result[iso] = iso_result

    return result


def compute_fleet_plant_mapping(df: pd.DataFrame, dims_df: pd.DataFrame) -> dict:
    """Map CEG fleet plants to market sweep results by fuel+ISO."""
    print("  Computing fleet plant mapping...")
    if not FLEET_JSON.exists():
        return {}
    with open(FLEET_JSON) as f:
        fleet_data = json.load(f)

    merged = df.join(dims_df)
    plants = fleet_data.get("base_fleet", [])

    # Fuel type to sweep column mapping
    FUEL_TO_COLS = {
        "nuclear": {"cf": None, "margin": None, "rev": "nuc_total_mwh", "energy_rev": "nuc_energy_rev_mwh", "cap_rev": "nuc_capacity_rev_mwh"},
        "gas_ccgt": {"cf": "ge_gas_ccgt_cf", "margin": "ge_gas_ccgt_margin_mwh", "rev": None, "capacity": "ge_gas_ccgt_capacity_mw"},
        "gas_ct": {"cf": "ge_gas_ct_cf", "margin": "ge_gas_ct_margin_mwh", "rev": None, "capacity": "ge_gas_ct_capacity_mw"},
        "oil_ct": {"cf": "ge_oil_ct_cf", "margin": "ge_oil_ct_margin_mwh", "rev": None},
        "coal_steam": {"cf": "ge_coal_steam_cf", "margin": "ge_coal_steam_margin_mwh", "rev": None},
    }

    # Group plants by ISO+fuel
    plant_groups = {}
    for p in plants:
        iso = p.get("iso", "NA")
        fuel = p.get("fuel_type", "unknown")
        key = f"{iso}_{fuel}"
        if key not in plant_groups:
            plant_groups[key] = {
                "iso": iso, "fuel": fuel,
                "plants": [], "total_mw": 0,
            }
        plant_groups[key]["plants"].append({
            "name": p.get("name", ""),
            "capacity_mw": p.get("capacity_mw", 0),
            "state": p.get("state", ""),
        })
        plant_groups[key]["total_mw"] += p.get("capacity_mw", 0)

    # Compute sweep-derived metrics for each plant group
    result = {}
    for key, group in plant_groups.items():
        iso = group["iso"]
        fuel = group["fuel"]
        if iso not in merged.iso.values:
            continue
        cols_map = FUEL_TO_COLS.get(fuel, {})
        if not cols_map:
            continue

        iso_df = merged[merged.iso == iso]
        group_result = {
            "fuel": fuel,
            "iso": iso,
            "total_mw": round(group["total_mw"], 1),
            "plant_count": len(group["plants"]),
            "plants": group["plants"][:5],  # Top 5 by name
            "sweep_metrics": {},
        }

        for year in KEY_YEARS:
            ydf = iso_df[iso_df.year == year]
            if len(ydf) < 10:
                continue
            year_metrics = {}
            for metric_name, col in cols_map.items():
                if col and col in ydf.columns:
                    vals = ydf[col].dropna()
                    if len(vals) > 0:
                        year_metrics[metric_name] = {
                            "p10": round(float(np.percentile(vals, 10)), 4),
                            "p50": round(float(np.percentile(vals, 50)), 4),
                            "p90": round(float(np.percentile(vals, 90)), 4),
                        }
            group_result["sweep_metrics"][str(year)] = year_metrics
        result[key] = group_result

    return result


def load_fleet_data() -> dict:
    """Load Constellation fleet reference data."""
    print("  Loading fleet data...")
    if not FLEET_JSON.exists():
        return {}
    with open(FLEET_JSON) as f:
        fleet = json.load(f)

    # Summarize by ISO and fuel type
    by_iso = {}
    for plant in fleet.get("base_fleet", []):
        iso = plant.get("iso", "NA")
        fuel = plant.get("fuel_type", "unknown")
        cap = plant.get("capacity_mw", 0)
        if iso not in by_iso:
            by_iso[iso] = {}
        if fuel not in by_iso[iso]:
            by_iso[iso][fuel] = {"count": 0, "capacity_mw": 0}
        by_iso[iso][fuel]["count"] += 1
        by_iso[iso][fuel]["capacity_mw"] += cap

    # Round capacities
    for iso in by_iso:
        for fuel in by_iso[iso]:
            by_iso[iso][fuel]["capacity_mw"] = round(by_iso[iso][fuel]["capacity_mw"], 1)

    targets = fleet.get("targets", {})
    gen_estimates = fleet.get("metadata", {}).get("generation_estimates_twh", {})

    return {
        "by_iso": by_iso,
        "targets": targets,
        "generation_twh": gen_estimates,
        "total_plants": len(fleet.get("base_fleet", [])),
    }


def main():
    print("=" * 60)
    print("CEG ISO Market Sweep Data Extraction")
    print("=" * 60)

    if not PARQUET.exists():
        print(f"ERROR: Parquet not found at {PARQUET}")
        sys.exit(1)

    print(f"\nLoading {PARQUET}...")
    df = pd.read_parquet(PARQUET)
    print(f"  Shape: {df.shape}")
    print(f"  ISOs: {sorted(df.iso.unique())}")
    print(f"  Years: {sorted(df.year.unique())}")
    print(f"  Scenarios per ISO/year: {len(df[(df.iso == 'CAISO') & (df.year == 2030)])}")

    # Parse scenario dimensions
    print("\nParsing scenario dimensions...")
    dims = df["scenario_id"].apply(parse_scenario_id).apply(pd.Series)
    dims.index = df.index

    # 1. Envelopes
    print("\n[1/6] Envelopes")
    envelopes = compute_envelopes(df)

    # 2. Sensitivity
    print("\n[2/6] Sensitivity")
    sensitivity = compute_sensitivity(df, dims)

    # 3. CEG archetypes
    print("\n[3/6] CEG Archetypes")
    ceg_archetypes = compute_ceg_archetypes(df, dims)

    # 4. Supply stacks
    print("\n[4/6] Supply Stacks")
    supply_stacks = compute_supply_stacks(df, dims)

    # 5. Outcome drivers
    print("\n[5/6] Outcome Drivers")
    outcome_drivers = compute_outcome_drivers(df, dims)

    # 6. Fleet plant mapping
    print("\n[6/7] Fleet Plant Mapping")
    fleet_mapping = compute_fleet_plant_mapping(df, dims)

    # 7. Fleet data
    print("\n[7/7] Fleet Data")
    fleet = load_fleet_data()

    # Assemble output
    output = {
        "meta": {
            "scenarios": int(df.scenario_id.nunique()),
            "isos": sorted(df.iso.unique().tolist()),
            "years": sorted(df.year.unique().tolist()),
            "key_years": KEY_YEARS,
            "sweep_dims": [
                "demand_growth", "price_sensitivity", "ppa_level",
                "gas_friction", "queue_capacity", "new_fossil_cost",
            ],
            "price_bundles": PRICE_BUNDLES,
            "percentiles": PERCENTILES,
            "generated": pd.Timestamp.now().isoformat(),
        },
        "envelopes": envelopes,
        "sensitivity": sensitivity,
        "ceg_archetypes": ceg_archetypes,
        "supply_stacks": supply_stacks,
        "outcome_drivers": outcome_drivers,
        "fleet_mapping": fleet_mapping,
        "fleet": fleet,
    }

    # Write JS file
    os.makedirs(OUTPUT.parent, exist_ok=True)
    js_content = "// Auto-generated by extract_iso_sweep_data.py\n"
    js_content += "// Do not edit manually\n"
    js_content += f"const ISO_SWEEP_DATA = {json.dumps(output, separators=(',', ':'))};\n"

    with open(OUTPUT, "w") as f:
        f.write(js_content)

    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"\n{'=' * 60}")
    print(f"Output: {OUTPUT}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
