#!/usr/bin/env python3
"""
EAC Scarcity Analysis — Compute SSS baselines, corporate demand scenarios,
scarcity classifications, and clean premiums per ISO.

Saves results to dashboard/eac_scarcity_results.json with incremental
checkpointing after each ISO×year combo to avoid losing work.

Based on SPEC.md SSS framework (Feb 15, 2026).
"""

import json
import os
import sys
import time
import math

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_PATH = "dashboard/overprocure_results.json"
OUTPUT_PATH = "dashboard/eac_scarcity_results.json"
CHECKPOINT_PATH = "data/checkpoints/eac_scarcity_checkpoint.json"

# ── SSS Data from SPEC.md (2025 baseline) ──────────────────────────────────
# SSS split into two components:
#   1. Fixed-fleet SSS (nuclear ZEC/CMC, public hydro, rate-base assets)
#      — absolute TWh, does NOT scale with demand, only changes on policy expiry
#   2. RPS SSS (state renewable/clean mandates)
#      — scales proportionally with demand (RPS is % of retail sales)
#
# non_sss = merchant generation available for voluntary corporate procurement
SSS_2025 = {
    "CAISO": {
        "total_clean_twh": 172,
        "sss_fixed_twh": 55,     # Diablo Canyon (~17), public hydro (~25), rate-base geothermal (~13) — permanent
        "sss_rps_twh": 92.5,     # SB 100 RPS-driven renewables
        "non_sss_twh": 24.5,     # midpoint of 17-32 (merchant wind/solar)
        "sss_pct": 0.85,
        "notes": "Diablo Canyon is fixed SSS (state-supported indefinitely). SB 100 RPS drives renewable growth."
    },
    "ERCOT": {
        "total_clean_twh": 205,
        "sss_fixed_twh": 20,     # STP nuclear (~20 TWh)
        "sss_rps_twh": 2.5,      # Minimal RPS requirement
        "non_sss_twh": 185,      # midpoint of 180-190 (merchant wind/solar)
        "sss_pct": 0.12,
        "notes": "Minimal RPS, mostly merchant renewables, deregulated market"
    },
    "PJM": {
        "total_clean_twh": 280,
        "sss_fixed_twh": 95,     # Nuclear ZEC/CMC programs (~94 IL + NJ expired), rate-base hydro (~1)
        "sss_rps_twh": 70,       # State RPS-driven renewables (NJ/MD/VA/IL/PA)
        "non_sss_twh": 115,      # midpoint of 100-130 (merchant nuclear + renewables)
        "sss_pct": 0.57,
        "notes": "Mixed: state RPSs + merchant nuclear (post-NJ ZEC expiry)"
    },
    "NYISO": {
        "total_clean_twh": 60,
        "sss_fixed_twh": 35,     # NY ZEC nuclear (~30), NYPA hydro (~5)
        "sss_rps_twh": 17,       # CLCPA/CES-driven renewables
        "non_sss_twh": 8,        # midpoint of 5-11 (limited merchant)
        "sss_pct": 0.85,
        "notes": "NY ZEC extended through 2049, CLCPA 70% by 2030"
    },
    "NEISO": {
        "total_clean_twh": 50,
        "sss_fixed_twh": 15,     # Millstone ZEC (~14), public hydro (~1)
        "sss_rps_twh": 12.5,     # MA/CT/RI/VT RPS mandates
        "non_sss_twh": 17.5,     # midpoint of 15-20 (merchant wind/solar/hydro)
        "sss_pct": 0.55,
        "notes": "CT Millstone PPA, MA/CT CES, some merchant hydro/wind"
    }
}

# ── Demand Growth Rates (from dashboard.html) ─────────────────────────────
DEMAND_GROWTH_RATES = {
    "CAISO":  {"Low": 0.014, "Medium": 0.019, "High": 0.025},
    "ERCOT":  {"Low": 0.020, "Medium": 0.035, "High": 0.055},
    "PJM":    {"Low": 0.015, "Medium": 0.024, "High": 0.036},
    "NYISO":  {"Low": 0.012, "Medium": 0.020, "High": 0.032},
    "NEISO":  {"Low": 0.008, "Medium": 0.014, "High": 0.022},
}

# ── RPS / Clean Energy Target Trajectories (% of demand) ─────────────────
# Clean supply modeled as target % of projected demand, per state RPS/CES
# mandates. As demand grows, absolute TWh of required clean generation grows
# proportionally. Sources: DSIRE, state legislation, LBNL deployment data.
#
# Low = legislative mandate only (minimum compliance)
# Medium = mandate + market economics + IRA incentives
# High = accelerated deployment (favorable interconnection, policy expansion)
RPS_TARGET_TRAJECTORIES = {
    "CAISO": {
        # SB 100: 60% RPS by 2030, 100% clean by 2045
        # 2025 actual: ~61% clean (172 TWh / 280 TWh equiv load basis)
        "Low":    {"2025": 0.61, "2030": 0.60, "2035": 0.68, "2040": 0.80, "2045": 0.95, "2050": 1.00},
        "Medium": {"2025": 0.61, "2030": 0.65, "2035": 0.75, "2040": 0.88, "2045": 1.00, "2050": 1.00},
        "High":   {"2025": 0.61, "2030": 0.72, "2035": 0.85, "2040": 0.95, "2045": 1.00, "2050": 1.00},
    },
    "ERCOT": {
        # No binding RPS. Growth driven by economics (cheapest wind/solar in US)
        # 2025 actual: ~42% clean (205 TWh / 488 TWh)
        # LBNL data: 31 GW solar added nationally in 2024; ERCOT 30.6 GW solar operational
        "Low":    {"2025": 0.42, "2030": 0.48, "2035": 0.52, "2040": 0.55, "2045": 0.58, "2050": 0.60},
        "Medium": {"2025": 0.42, "2030": 0.55, "2035": 0.62, "2040": 0.68, "2045": 0.72, "2050": 0.75},
        "High":   {"2025": 0.42, "2030": 0.62, "2035": 0.72, "2040": 0.80, "2045": 0.85, "2050": 0.88},
    },
    "PJM": {
        # Blended state RPSs: NJ 50% by 2030, VA VCEA 100% by 2050,
        # MD 50% by 2030, IL 50% by 2040, PA ~8% AEPS
        # 2025 actual: ~33% clean (280 TWh / 843 TWh)
        # LBNL: PJM hit 11.7 GW solar peak Apr 2025 (73% YoY increase)
        "Low":    {"2025": 0.33, "2030": 0.38, "2035": 0.43, "2040": 0.48, "2045": 0.55, "2050": 0.62},
        "Medium": {"2025": 0.33, "2030": 0.42, "2035": 0.50, "2040": 0.58, "2045": 0.66, "2050": 0.75},
        "High":   {"2025": 0.33, "2030": 0.48, "2035": 0.58, "2040": 0.68, "2045": 0.78, "2050": 0.85},
    },
    "NYISO": {
        # CLCPA: 70% renewable by 2030, 100% zero-emission by 2040
        # 2025 actual: ~40% clean (60 TWh / 152 TWh)
        # Aggressive offshore wind pipeline but permitting delays
        "Low":    {"2025": 0.40, "2030": 0.55, "2035": 0.65, "2040": 0.80, "2045": 0.90, "2050": 0.95},
        "Medium": {"2025": 0.40, "2030": 0.62, "2035": 0.78, "2040": 0.92, "2045": 1.00, "2050": 1.00},
        "High":   {"2025": 0.40, "2030": 0.70, "2035": 0.88, "2040": 1.00, "2045": 1.00, "2050": 1.00},
    },
    "NEISO": {
        # MA CES 80% by 2040, CT 100% by 2040, RI 100% by 2033
        # 2025 actual: ~43% clean (50 TWh / 115 TWh)
        # Constrained by transmission + offshore wind delays
        "Low":    {"2025": 0.43, "2030": 0.48, "2035": 0.55, "2040": 0.65, "2045": 0.72, "2050": 0.78},
        "Medium": {"2025": 0.43, "2030": 0.55, "2035": 0.65, "2040": 0.78, "2045": 0.85, "2050": 0.90},
        "High":   {"2025": 0.43, "2030": 0.62, "2035": 0.75, "2040": 0.88, "2045": 0.95, "2050": 1.00},
    },
}

# ── SSS Policy Evolution (fraction of NEW supply that becomes SSS) ────────
# As new clean builds come online, what fraction is SSS vs merchant?
# Declining over time as RPS mandates saturate and merchant share grows
SSS_NEW_BUILD_FRACTION = {
    "CAISO":  {"2025": 0.80, "2030": 0.70, "2035": 0.60, "2040": 0.50, "2045": 0.45, "2050": 0.40},
    "ERCOT":  {"2025": 0.10, "2030": 0.10, "2035": 0.10, "2040": 0.10, "2045": 0.10, "2050": 0.10},
    "PJM":    {"2025": 0.55, "2030": 0.50, "2035": 0.45, "2040": 0.40, "2045": 0.35, "2050": 0.30},
    "NYISO":  {"2025": 0.80, "2030": 0.75, "2035": 0.65, "2040": 0.55, "2045": 0.50, "2050": 0.45},
    "NEISO":  {"2025": 0.55, "2030": 0.50, "2035": 0.45, "2040": 0.40, "2045": 0.35, "2050": 0.30},
}

# ── Policy Expirations (SSS TWh that shifts to non-SSS) ──────────────────
# From SPEC.md: IL ZEC/CMC expires mid-2027, NJ ZEC expired June 2025
# "component" field: "fixed" = nuclear/hydro fixed fleet, "rps" = RPS-eligible
SSS_EXPIRATIONS = {
    "PJM": [
        {"year": 2027, "twh_shift": 94, "component": "fixed",
         "note": "IL ZEC/CMC expiry — Dresden, Braidwood, Byron, LaSalle, Clinton, Quad Cities"},
    ],
    # NJ ZEC already expired (reflected in 2025 baseline)
    # NY ZEC extended through 2049
    # Diablo Canyon: fixed SSS indefinitely (state-supported)
}

# ── Corporate PPA Consumption (TWh already contracted, reducing non-SSS) ──
# Major data center PPAs already consuming non-SSS supply
EXISTING_CORPORATE_PPAS = {
    "PJM":   {"twh": 25, "note": "Amazon-Susquehanna, Meta-Vistra, Microsoft-Crane + others"},
    "ERCOT": {"twh": 15, "note": "Large corporate solar/wind PPAs in TX"},
    "CAISO": {"twh": 5,  "note": "Tech company PPAs in CA"},
    "NYISO": {"twh": 2,  "note": "Limited corporate PPAs"},
    "NEISO": {"twh": 3,  "note": "Limited corporate PPAs"},
}

# ── Scenario Parameters ───────────────────────────────────────────────────
PARTICIPATION_RATES = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
MATCH_TARGETS = [75, 80, 85, 90, 95, 99, 100]
TIME_HORIZONS = [2025, 2030, 2035, 2040, 2045, 2050]
GROWTH_LEVELS = ["Low", "Medium", "High"]
ISOS = ["CAISO", "ERCOT", "PJM", "NYISO", "NEISO"]

# ── Optimizer-Derived Data ─────────────────────────────────────────────────
# Loaded at runtime from overprocure_results.json
# Provides per-ISO procurement_pct and incremental_above_baseline at each threshold
OPTIMIZER_DATA = {}  # populated by load_optimizer_data()


def load_optimizer_data():
    """Load procurement_pct and incremental_above_baseline per ISO×threshold
    from the optimizer results JSON.  Populates the global OPTIMIZER_DATA dict.

    Structure: OPTIMIZER_DATA[iso][threshold_int] = {
        "procurement_pct": float,   # e.g. 143 means 143% of target
        "incremental_cost": float,  # $/MWh above baseline wholesale
    }
    """
    global OPTIMIZER_DATA
    with open(RESULTS_PATH) as f:
        raw = json.load(f)

    for iso in ISOS:
        OPTIMIZER_DATA[iso] = {}
        thresholds = raw["results"][iso]["thresholds"]
        for thr_key, thr_data in thresholds.items():
            t = int(float(thr_key))
            OPTIMIZER_DATA[iso][t] = {
                "procurement_pct": thr_data["procurement_pct"],
                "incremental_cost": thr_data["costs"]["incremental_above_baseline"],
            }


def interp_optimizer(iso, match_target, field):
    """Interpolate an optimizer field for any match_target 0-100.

    field: "procurement_pct" or "incremental_cost"
    For targets below the lowest optimizer threshold, use the lowest value.
    For targets above the highest, use the highest value.
    Between thresholds, linear interpolation.
    """
    iso_data = OPTIMIZER_DATA.get(iso, {})
    if not iso_data:
        # Fallback if optimizer data missing for this ISO
        return 100.0 if field == "procurement_pct" else 0.0

    thresholds = sorted(iso_data.keys())

    # Exact match
    if match_target in iso_data:
        return iso_data[match_target][field]

    # Below lowest
    if match_target <= thresholds[0]:
        return iso_data[thresholds[0]][field]

    # Above highest
    if match_target >= thresholds[-1]:
        return iso_data[thresholds[-1]][field]

    # Interpolate between bracketing thresholds
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if lo <= match_target <= hi:
            t = (match_target - lo) / (hi - lo)
            v0 = iso_data[lo][field]
            v1 = iso_data[hi][field]
            return v0 + t * (v1 - v0)

    return iso_data[thresholds[-1]][field]


# ── Scarcity Classification Thresholds ────────────────────────────────────
SCARCITY_BANDS = {
    "abundant":   {"max_ratio": 0.3,  "label": "Abundant",   "color": "#22c55e"},
    "adequate":   {"max_ratio": 0.6,  "label": "Adequate",   "color": "#84cc16"},
    "tightening": {"max_ratio": 0.8,  "label": "Tightening", "color": "#eab308"},
    "scarce":     {"max_ratio": 0.95, "label": "Scarce",     "color": "#f97316"},
    "critical":   {"max_ratio": 999,  "label": "Critical",   "color": "#ef4444"},
}


def classify_scarcity(demand_ratio):
    """Classify scarcity based on demand/supply ratio."""
    for band_key, band in SCARCITY_BANDS.items():
        if demand_ratio <= band["max_ratio"]:
            return band_key, band["label"], band["color"]
    return "critical", "Critical", "#ef4444"


def get_sss_new_fraction(iso, year):
    """Interpolate SSS new-build fraction for any year."""
    fractions = SSS_NEW_BUILD_FRACTION[iso]
    years = sorted(int(y) for y in fractions.keys())
    if year <= years[0]:
        return fractions[str(years[0])]
    if year >= years[-1]:
        return fractions[str(years[-1])]
    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            t = (year - years[i]) / (years[i + 1] - years[i])
            v0 = fractions[str(years[i])]
            v1 = fractions[str(years[i + 1])]
            return v0 + t * (v1 - v0)
    return fractions[str(years[-1])]


def compute_clean_premium(iso, match_target, scarcity_ratio):
    """Compute clean premium ($/MWh) from optimizer incremental cost × scarcity.

    Base cost: optimizer's incremental_above_baseline for this ISO × threshold.
    Scarcity multiplier: as demand/supply ratio rises, premiums escalate
    above the optimizer's 2025 base cost.
    """
    # Base premium from optimizer results (2025 cost curve)
    base = interp_optimizer(iso, match_target, "incremental_cost")

    # Scarcity multiplier: as supply tightens over time, premiums escalate
    if scarcity_ratio < 0.3:
        multiplier = 1.0
    elif scarcity_ratio < 0.6:
        multiplier = 1.0 + (scarcity_ratio - 0.3) * 1.0  # up to 1.3×
    elif scarcity_ratio < 0.8:
        multiplier = 1.3 + (scarcity_ratio - 0.6) * 2.5  # up to 1.8×
    elif scarcity_ratio < 0.95:
        multiplier = 1.8 + (scarcity_ratio - 0.8) * 8.0  # up to 3.0×
    else:
        multiplier = 3.0 + (scarcity_ratio - 0.95) * 40.0  # exponential

    return round(base * multiplier, 2)


def interpolate_rps_target(iso, year, growth_level):
    """Interpolate RPS clean energy target % for a given ISO/year/growth."""
    targets = RPS_TARGET_TRAJECTORIES[iso][growth_level]
    years = sorted(int(y) for y in targets.keys())
    if year <= years[0]:
        return targets[str(years[0])]
    if year >= years[-1]:
        return targets[str(years[-1])]
    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            t = (year - years[i]) / (years[i + 1] - years[i])
            v0 = targets[str(years[i])]
            v1 = targets[str(years[i + 1])]
            return v0 + t * (v1 - v0)
    return targets[str(years[-1])]


def evolve_supply(iso, year, growth_level, demand_twh_at_year):
    """Project clean supply (total, SSS, non-SSS) at a future year.

    Two-component SSS model:
    1. Fixed-fleet SSS (nuclear, public hydro, rate-base) — constant TWh,
       only changes on policy expiry. Does NOT scale with demand.
    2. RPS SSS — scales proportionally with demand (RPS = % of retail sales).

    Total clean supply = RPS target % × demand (demand-proportional).
    New supply above 2025 baseline split into SSS vs merchant per
    SSS_NEW_BUILD_FRACTION.
    """
    base = SSS_2025[iso]

    if year == 2025:
        total = base["total_clean_twh"]
        sss_fixed = base["sss_fixed_twh"]
        sss_rps = base["sss_rps_twh"]
        non_sss = base["non_sss_twh"]
    else:
        # ── Step 1: Fixed-fleet SSS (public hydro, rate-base assets) ──
        # Constant TWh — does NOT scale with demand.
        # Only changes on policy expiry of fixed-fleet components.
        sss_fixed = base["sss_fixed_twh"]
        rps_expiry_reduction = 0
        if iso in SSS_EXPIRATIONS:
            for exp in SSS_EXPIRATIONS[iso]:
                if year >= exp["year"]:
                    if exp.get("component", "fixed") == "fixed":
                        sss_fixed -= exp["twh_shift"]
                    else:
                        # RPS-eligible expiry (e.g. Diablo Canyon in CAISO CES)
                        rps_expiry_reduction += exp["twh_shift"]
        sss_fixed = max(0, sss_fixed)

        # ── Step 2: Total clean supply = RPS target % × demand ──
        rps_target = interpolate_rps_target(iso, year, growth_level)
        total = demand_twh_at_year * rps_target
        total = max(total, base["total_clean_twh"])  # never below 2025

        # ── Step 3: RPS demand (what state mandates require) ──
        # RPS mandates require a % of retail sales from renewables/clean.
        # This demand is satisfied FIRST by existing clean supply (both
        # existing SSS renewables AND existing merchant renewables),
        # then new build covers the shortfall.
        #
        # RPS demand grows with load. Non-nuclear clean available for RPS:
        non_nuclear_clean_2025 = base["sss_rps_twh"] + base["non_sss_twh"]

        # Total RPS-eligible supply needed at this year
        # (total clean minus what fixed fleet provides)
        rps_eligible_needed = total - sss_fixed

        if rps_eligible_needed <= non_nuclear_clean_2025:
            # Existing supply covers RPS. Split between SSS and merchant
            # using same ratio as 2025 baseline.
            base_rps_share = base["sss_rps_twh"] / non_nuclear_clean_2025 if non_nuclear_clean_2025 > 0 else 0.5
            sss_rps = rps_eligible_needed * base_rps_share
            non_sss = rps_eligible_needed * (1 - base_rps_share)
        else:
            # Need new build. Existing supply fully consumed.
            # New build beyond existing is split per SSS_NEW_BUILD_FRACTION.
            new_build = rps_eligible_needed - non_nuclear_clean_2025
            avg_sss_frac = get_sss_new_fraction(iso, year)

            # RPS SSS = existing RPS SSS + new SSS portion of new build
            sss_rps = base["sss_rps_twh"] + new_build * avg_sss_frac
            # Non-SSS = existing merchant + new merchant portion
            non_sss = base["non_sss_twh"] + new_build * (1 - avg_sss_frac)

            # As RPS grows, it can claim existing merchant renewables.
            # RPS demand that exceeds (existing SSS RPS + new SSS build)
            # must be met by pulling from merchant pool.
            total_rps_mandate = rps_eligible_needed * avg_sss_frac + base["sss_rps_twh"]
            # This is already captured above since new build fraction
            # determines the SSS/merchant split of incremental supply.

    # Apply RPS-component expirations (e.g. Diablo Canyon leaving CES pool)
    if year > 2025 and rps_expiry_reduction > 0:
        sss_rps = max(0, sss_rps - rps_expiry_reduction)
        non_sss += rps_expiry_reduction  # shifts to merchant pool

    sss = sss_fixed + sss_rps

    # Subtract existing corporate PPAs from available non-SSS
    existing_ppas = EXISTING_CORPORATE_PPAS.get(iso, {}).get("twh", 0)
    available_non_sss = max(0, non_sss - existing_ppas)

    return {
        "total_clean_twh": round(total, 1),
        "sss_twh": round(sss, 1),
        "sss_fixed_twh": round(max(0, sss_fixed), 1),
        "sss_rps_twh": round(sss_rps, 1),
        "non_sss_twh": round(non_sss, 1),
        "available_non_sss_twh": round(available_non_sss, 1),
        "existing_ppas_twh": existing_ppas,
        # SSS pro-rata share: fraction of load covered by total SSS allocation
        # Used to derate corporate EAC demand (they don't start from zero)
        "sss_share_of_total": round(sss / total, 4) if total > 0 else 0,
    }


def load_demand_data():
    """Load ISO demand from optimizer results."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    demands = {}
    for iso, iso_data in data["results"].items():
        demands[iso] = iso_data["annual_demand_mwh"] / 1e6  # TWh
    return demands


def load_checkpoint():
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


def save_checkpoint(completed, results):
    """Save checkpoint with completed combos and partial results."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    cp = {
        "completed": completed,
        "results": results,
        "timestamp": time.time(),
    }
    # Atomic write
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cp, f)
    os.replace(tmp, CHECKPOINT_PATH)


def save_final_results(results):
    """Save final results JSON."""
    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, OUTPUT_PATH)
    print(f"Saved final results to {OUTPUT_PATH}")


def compute_inflection_points(iso, year, growth_level, supply_data, demand_twh):
    """Find the participation rate at which each match target hits scarcity."""
    inflections = {}
    available = supply_data["available_non_sss_twh"]
    sss_share = supply_data["sss_share_of_total"]

    for match_target in MATCH_TARGETS:
        # Procurement multiplier from optimizer (procurement_pct / 100)
        procurement_mult = interp_optimizer(iso, match_target, "procurement_pct") / 100.0

        # SSS pro-rata derate: incremental need = (target% - sss%) × load
        incremental_need_frac = max(0, match_target / 100 - sss_share)

        # Find participation rate where demand exceeds available supply
        inflection_pct = None
        for pct in range(1, 101):
            corp_demand = demand_twh * (pct / 100) * incremental_need_frac * procurement_mult
            if corp_demand > available:
                inflection_pct = pct
                break

        inflections[str(match_target)] = {
            "inflection_participation_pct": inflection_pct,
            "procurement_multiplier": round(procurement_mult, 3),
            "max_supportable_twh": round(available / procurement_mult, 1) if procurement_mult > 0 else 0,
        }

    return inflections


def main():
    print("=" * 60)
    print("EAC Scarcity Analysis — Computing...")
    print("=" * 60)

    # Load optimizer data (procurement multipliers + incremental costs per ISO×threshold)
    load_optimizer_data()
    print(f"Loaded optimizer data for {len(OPTIMIZER_DATA)} ISOs: "
          f"{sorted(next(iter(OPTIMIZER_DATA.values())).keys())} thresholds")

    # Load demand data
    demands = load_demand_data()
    print(f"Loaded demand data for {len(demands)} ISOs")

    # Check for checkpoint
    checkpoint = load_checkpoint()
    completed_keys = set()
    results = {
        "metadata": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "EAC scarcity analysis: SSS baselines, corporate demand scenarios, scarcity classifications, clean premiums",
            "framework": "SSS = mandatory/non-bypassable procurement (RPS/CES, public ownership, rate-base, state nuclear programs)",
            "parameters": {
                "participation_rates_pct": PARTICIPATION_RATES,
                "match_targets_pct": MATCH_TARGETS,
                "time_horizons": TIME_HORIZONS,
                "growth_levels": GROWTH_LEVELS,
                "isos": ISOS,
            },
        },
        "sss_2025_baseline": SSS_2025,
        "demand_growth_rates": DEMAND_GROWTH_RATES,
        "scarcity_bands": SCARCITY_BANDS,
        "clean_premium_benchmarks": {
            "social_cost_carbon_epa": 51,
            "social_cost_carbon_rennert": 185,
            "eu_ets_range": [60, 100],
        },
        "supply_projections": {},
        "scenarios": {},
        "inflection_points": {},
        "national_summary": {},
    }

    if checkpoint:
        completed_keys = set(checkpoint.get("completed", []))
        # Restore partial results
        prev = checkpoint.get("results", {})
        for section in ["supply_projections", "scenarios", "inflection_points", "national_summary"]:
            if section in prev:
                results[section] = prev[section]
        print(f"Resumed from checkpoint: {len(completed_keys)} combos already done")

    # ── Phase 1: Supply projections ───────────────────────────────────────
    total_combos = len(ISOS) * len(TIME_HORIZONS) * len(GROWTH_LEVELS)
    done = 0

    for iso in ISOS:
        demand_twh = demands[iso]
        if iso not in results["supply_projections"]:
            results["supply_projections"][iso] = {}

        for year in TIME_HORIZONS:
            year_key = str(year)
            if year_key not in results["supply_projections"][iso]:
                results["supply_projections"][iso][year_key] = {}

            for growth in GROWTH_LEVELS:
                combo_key = f"supply_{iso}_{year}_{growth}"
                if combo_key in completed_keys:
                    done += 1
                    continue

                # Project demand at this year/growth
                growth_rate = DEMAND_GROWTH_RATES[iso][growth]
                projected_demand = demand_twh * (1 + growth_rate) ** (year - 2025)

                supply = evolve_supply(iso, year, growth, projected_demand)
                results["supply_projections"][iso][year_key][growth] = supply
                completed_keys.add(combo_key)
                done += 1

            # Checkpoint after each ISO×year
            save_checkpoint(list(completed_keys), results)

    print(f"Phase 1 complete: {done} supply projections")

    # ── Phase 2: Scenario matrix ──────────────────────────────────────────
    scenario_count = 0
    total_scenarios = len(ISOS) * len(TIME_HORIZONS) * len(GROWTH_LEVELS) * len(PARTICIPATION_RATES) * len(MATCH_TARGETS)
    print(f"Phase 2: Computing {total_scenarios} scenarios...")

    for iso in ISOS:
        demand_twh = demands[iso]
        if iso not in results["scenarios"]:
            results["scenarios"][iso] = {}

        for year in TIME_HORIZONS:
            year_key = str(year)
            if year_key not in results["scenarios"][iso]:
                results["scenarios"][iso][year_key] = {}

            # Grow demand
            for growth in GROWTH_LEVELS:
                growth_rate = DEMAND_GROWTH_RATES[iso][growth]
                projected_demand = demand_twh * (1 + growth_rate) ** (year - 2025)

                if growth not in results["scenarios"][iso][year_key]:
                    results["scenarios"][iso][year_key][growth] = []

                combo_key = f"scenario_{iso}_{year}_{growth}"
                if combo_key in completed_keys:
                    scenario_count += len(PARTICIPATION_RATES) * len(MATCH_TARGETS)
                    continue

                supply = results["supply_projections"][iso][year_key][growth]
                available = supply["available_non_sss_twh"]
                sss_share = supply["sss_share_of_total"]

                scenarios_for_combo = []

                for participation in PARTICIPATION_RATES:
                    for match_target in MATCH_TARGETS:
                        # Procurement multiplier from optimizer results
                        mult = interp_optimizer(iso, match_target, "procurement_pct") / 100.0

                        # Corporate load participating
                        corp_load = projected_demand * (participation / 100)
                        # SSS pro-rata derate: corporations already receive
                        # a pro-rata share of SSS clean energy. Their
                        # incremental EAC need = (target% - sss%) × load
                        incremental_need_frac = max(0, match_target / 100 - sss_share)
                        corp_eac_demand = corp_load * incremental_need_frac * mult
                        demand_ratio = corp_eac_demand / available if available > 0 else 999

                        band_key, label, color = classify_scarcity(demand_ratio)
                        premium = compute_clean_premium(iso, match_target, min(demand_ratio, 2.0))

                        scenarios_for_combo.append({
                            "participation_pct": participation,
                            "match_target_pct": match_target,
                            "projected_demand_twh": round(projected_demand, 1),
                            "corp_load_twh": round(corp_load, 1),
                            "sss_pro_rata_pct": round(sss_share * 100, 1),
                            "incremental_need_pct": round(incremental_need_frac * 100, 1),
                            "corp_eac_demand_twh": round(corp_eac_demand, 1),
                            "available_supply_twh": round(available, 1),
                            "demand_supply_ratio": round(demand_ratio, 3),
                            "scarcity_class": band_key,
                            "scarcity_label": label,
                            "clean_premium_per_mwh": premium,
                        })
                        scenario_count += 1

                results["scenarios"][iso][year_key][growth] = scenarios_for_combo
                completed_keys.add(combo_key)

            # Checkpoint after each ISO×year
            save_checkpoint(list(completed_keys), results)
            print(f"  {iso} {year}: {scenario_count}/{total_scenarios} scenarios", flush=True)

    print(f"Phase 2 complete: {scenario_count} scenarios")

    # ── Phase 3: Inflection points ────────────────────────────────────────
    print("Phase 3: Computing inflection points...")
    for iso in ISOS:
        if iso not in results["inflection_points"]:
            results["inflection_points"][iso] = {}

        for year in TIME_HORIZONS:
            year_key = str(year)
            if year_key not in results["inflection_points"][iso]:
                results["inflection_points"][iso][year_key] = {}

            for growth in GROWTH_LEVELS:
                combo_key = f"inflection_{iso}_{year}_{growth}"
                if combo_key in completed_keys:
                    continue

                supply = results["supply_projections"][iso][year_key][growth]
                growth_rate = DEMAND_GROWTH_RATES[iso][growth]
                projected_demand = demands[iso] * (1 + growth_rate) ** (year - 2025)

                inflections = compute_inflection_points(iso, year, growth, supply, projected_demand)
                results["inflection_points"][iso][year_key][growth] = inflections
                completed_keys.add(combo_key)

        save_checkpoint(list(completed_keys), results)

    print("Phase 3 complete")

    # ── Phase 4: National summary ─────────────────────────────────────────
    print("Phase 4: National summary...")
    for year in TIME_HORIZONS:
        year_key = str(year)
        results["national_summary"][year_key] = {}

        for growth in GROWTH_LEVELS:
            total_clean = sum(
                results["supply_projections"][iso][year_key][growth]["total_clean_twh"]
                for iso in ISOS
            )
            total_sss = sum(
                results["supply_projections"][iso][year_key][growth]["sss_twh"]
                for iso in ISOS
            )
            total_non_sss = sum(
                results["supply_projections"][iso][year_key][growth]["non_sss_twh"]
                for iso in ISOS
            )
            total_available = sum(
                results["supply_projections"][iso][year_key][growth]["available_non_sss_twh"]
                for iso in ISOS
            )
            total_demand = sum(
                demands[iso] * (1 + DEMAND_GROWTH_RATES[iso][growth]) ** (year - 2025)
                for iso in ISOS
            )

            results["national_summary"][year_key][growth] = {
                "total_clean_twh": round(total_clean, 1),
                "total_sss_twh": round(total_sss, 1),
                "total_non_sss_twh": round(total_non_sss, 1),
                "total_available_twh": round(total_available, 1),
                "total_demand_twh": round(total_demand, 1),
                "national_sss_pct": round(total_sss / total_clean * 100, 1) if total_clean > 0 else 0,
            }

    # ── Save final results ────────────────────────────────────────────────
    save_final_results(results)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint cleaned up")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for iso in ISOS:
        s2025 = results["supply_projections"][iso]["2025"]["Medium"]
        s2035 = results["supply_projections"][iso]["2035"]["Medium"]
        print(f"\n{iso}:")
        print(f"  2025: {s2025['available_non_sss_twh']} TWh available (of {s2025['total_clean_twh']} total clean)")
        print(f"  2035: {s2035['available_non_sss_twh']} TWh available (of {s2035['total_clean_twh']} total clean)")
        inf = results["inflection_points"][iso]["2025"]["Medium"]
        for mt in ["90", "95", "100"]:
            ip = inf.get(mt, {}).get("inflection_participation_pct")
            print(f"  2025 @{mt}% match: scarcity at {ip}% participation" if ip else f"  2025 @{mt}% match: no scarcity at 100% participation")

    nat = results["national_summary"]["2025"]["Medium"]
    print(f"\nNational 2025 (Medium growth):")
    print(f"  Total clean: {nat['total_clean_twh']} TWh")
    print(f"  SSS: {nat['total_sss_twh']} TWh ({nat['national_sss_pct']}%)")
    print(f"  Available for corporate: {nat['total_available_twh']} TWh")
    print(f"  Total demand: {nat['total_demand_twh']} TWh")

    print(f"\nTotal scenarios computed: {scenario_count}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
