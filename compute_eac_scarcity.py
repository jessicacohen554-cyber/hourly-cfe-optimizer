#!/usr/bin/env python3
"""
EAC Scarcity Analysis — Combined RPS + Voluntary Supply Stack Model (v2).

Two-track demand model:
  1. RPS-mandated build is forced regardless of economics
  2. Voluntary corporate demand rides on top
Both compete for the same finite buildable capacity on the supply stack.
Marginal cost set by combined demand (RPS + voluntary).

Validated against Xu et al. (2024 Joule / Princeton ZERO Lab),
Gillenwater (2008), Denholm et al. (NREL 2021).

Saves results to dashboard/eac_scarcity_results.json with incremental
checkpointing after each ISO×year combo to avoid losing work.
"""

import json
import os
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
    # Aligned with optimizer dashboard (dashboard.html) researched values
    "CAISO":  {"Low": 0.014, "Medium": 0.019, "High": 0.025},
    "ERCOT":  {"Low": 0.020, "Medium": 0.035, "High": 0.055},
    "PJM":    {"Low": 0.015, "Medium": 0.024, "High": 0.036},
    "NYISO":  {"Low": 0.013, "Medium": 0.020, "High": 0.044},
    "NEISO":  {"Low": 0.009, "Medium": 0.018, "High": 0.029},
}

# ── RPS / Clean Energy Target Trajectories (% of demand) ─────────────────
# Drives MANDATORY new build — RPS compliance build happens regardless of economics.
RPS_TARGET_TRAJECTORIES = {
    "CAISO": {
        "Low":    {"2025": 0.61, "2030": 0.60, "2035": 0.68, "2040": 0.80, "2045": 0.95, "2050": 1.00},
        "Medium": {"2025": 0.61, "2030": 0.65, "2035": 0.75, "2040": 0.88, "2045": 1.00, "2050": 1.00},
        "High":   {"2025": 0.61, "2030": 0.72, "2035": 0.85, "2040": 0.95, "2045": 1.00, "2050": 1.00},
    },
    "ERCOT": {
        "Low":    {"2025": 0.42, "2030": 0.48, "2035": 0.52, "2040": 0.55, "2045": 0.58, "2050": 0.60},
        "Medium": {"2025": 0.42, "2030": 0.55, "2035": 0.62, "2040": 0.68, "2045": 0.72, "2050": 0.75},
        "High":   {"2025": 0.42, "2030": 0.62, "2035": 0.72, "2040": 0.80, "2045": 0.85, "2050": 0.88},
    },
    "PJM": {
        "Low":    {"2025": 0.33, "2030": 0.38, "2035": 0.43, "2040": 0.48, "2045": 0.55, "2050": 0.62},
        "Medium": {"2025": 0.33, "2030": 0.42, "2035": 0.50, "2040": 0.58, "2045": 0.66, "2050": 0.75},
        "High":   {"2025": 0.33, "2030": 0.48, "2035": 0.58, "2040": 0.68, "2045": 0.78, "2050": 0.85},
    },
    "NYISO": {
        "Low":    {"2025": 0.40, "2030": 0.55, "2035": 0.65, "2040": 0.80, "2045": 0.90, "2050": 0.95},
        "Medium": {"2025": 0.40, "2030": 0.62, "2035": 0.78, "2040": 0.92, "2045": 1.00, "2050": 1.00},
        "High":   {"2025": 0.40, "2030": 0.70, "2035": 0.88, "2040": 1.00, "2045": 1.00, "2050": 1.00},
    },
    "NEISO": {
        "Low":    {"2025": 0.43, "2030": 0.48, "2035": 0.55, "2040": 0.65, "2045": 0.72, "2050": 0.78},
        "Medium": {"2025": 0.43, "2030": 0.55, "2035": 0.65, "2040": 0.78, "2045": 0.85, "2050": 0.90},
        "High":   {"2025": 0.43, "2030": 0.62, "2035": 0.75, "2040": 0.88, "2045": 0.95, "2050": 1.00},
    },
}

# ── SSS Policy Evolution (fraction of NEW supply that becomes SSS) ────────
SSS_NEW_BUILD_FRACTION = {
    "CAISO":  {"2025": 0.80, "2030": 0.70, "2035": 0.60, "2040": 0.50, "2045": 0.45, "2050": 0.40},
    "ERCOT":  {"2025": 0.10, "2030": 0.10, "2035": 0.10, "2040": 0.10, "2045": 0.10, "2050": 0.10},
    "PJM":    {"2025": 0.55, "2030": 0.50, "2035": 0.45, "2040": 0.40, "2045": 0.35, "2050": 0.30},
    "NYISO":  {"2025": 0.80, "2030": 0.75, "2035": 0.65, "2040": 0.55, "2045": 0.50, "2050": 0.45},
    "NEISO":  {"2025": 0.55, "2030": 0.50, "2035": 0.45, "2040": 0.40, "2045": 0.35, "2050": 0.30},
}

# ── Policy Expirations (SSS TWh that shifts to non-SSS) ──────────────────
# BUG FIX (v2): These TWh TRANSFER to non-SSS — they don't vanish.
# Plants still generate; they just lost their subsidy and become merchant.
SSS_EXPIRATIONS = {
    "PJM": [
        {"year": 2027, "twh_shift": 94, "component": "fixed",
         "note": "IL ZEC/CMC expiry — Dresden, Braidwood, Byron, LaSalle, Clinton, Quad Cities"},
    ],
}

# ── Corporate PPA Consumption (TWh already contracted, reducing non-SSS) ──
EXISTING_CORPORATE_PPAS = {
    "PJM":   {"twh": 25, "note": "Amazon-Susquehanna, Meta-Vistra, Microsoft-Crane + others"},
    "ERCOT": {"twh": 15, "note": "Large corporate solar/wind PPAs in TX"},
    "CAISO": {"twh": 5,  "note": "Tech company PPAs in CA"},
    "NYISO": {"twh": 2,  "note": "Limited corporate PPAs"},
    "NEISO": {"twh": 3,  "note": "Limited corporate PPAs"},
}

# ── Committed Hyperscaler Clean PPA Pipeline ─────────────────────────────
COMMITTED_CLEAN_PIPELINE = {
    "PJM": {
        "phasing_gw": {
            "2025": 1.0,   # Susquehanna campus + early deals
            "2027": 2.0,   # Additional contracts online
            "2028": 3.0,   # TMI Unit 1 restart
            "2030": 4.0,   # Full pipeline
            "2050": 4.0,   # Held constant post-2030
        },
        "capacity_factor": 0.90,
        "note": "Amazon-Talen Susquehanna, Microsoft-Constellation TMI, others",
    },
}

# ── Scenario Parameters ───────────────────────────────────────────────────
PARTICIPATION_RATES = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
MATCH_TARGETS = [75, 80, 85, 90, 95, 99, 100]
TIME_HORIZONS = list(range(2025, 2051))  # Annual: 2025–2050 (26 years)
GROWTH_LEVELS = ["Low", "Medium", "High"]
ISOS = ["CAISO", "ERCOT", "PJM", "NYISO", "NEISO"]

# ── C&I Share of Total Demand ─────────────────────────────────────────────
CI_SHARE = 0.62

# ── Wholesale Prices ($/MWh) from optimizer config ────────────────────────
WHOLESALE_PRICES = {
    "CAISO": 30, "ERCOT": 27, "PJM": 34, "NYISO": 42, "NEISO": 41
}

# ── Clean Supply Cost Stack per ISO ───────────────────────────────────────
# Ordered by LCOE (cheapest first). LCOEs from optimizer config (NREL ATB-derived).
# annual_add_twh: max annual buildable capacity based on interconnection queue
#   data (LBNL "Queued Up") and resource potential.
# max_cumulative_twh: ceiling on total buildable through 2050.
CLEAN_SUPPLY_STACK = {
    "ERCOT": [
        {"resource": "wind",       "lcoe": 40,  "annual_add_twh": 20, "max_cumulative_twh": 300},
        {"resource": "solar",      "lcoe": 54,  "annual_add_twh": 25, "max_cumulative_twh": 400},
        {"resource": "clean_firm", "lcoe": 90,  "annual_add_twh": 3,  "max_cumulative_twh": 50},
        {"resource": "storage",    "lcoe": 100, "annual_add_twh": 2,  "max_cumulative_twh": 30},
    ],
    "CAISO": [
        {"resource": "solar",      "lcoe": 60,  "annual_add_twh": 10, "max_cumulative_twh": 150},
        {"resource": "wind",       "lcoe": 73,  "annual_add_twh": 3,  "max_cumulative_twh": 50},
        {"resource": "clean_firm", "lcoe": 90,  "annual_add_twh": 2,  "max_cumulative_twh": 30},
        {"resource": "storage",    "lcoe": 100, "annual_add_twh": 2,  "max_cumulative_twh": 25},
    ],
    "PJM": [
        {"resource": "wind",       "lcoe": 62,  "annual_add_twh": 8,  "max_cumulative_twh": 120},
        {"resource": "solar",      "lcoe": 65,  "annual_add_twh": 12, "max_cumulative_twh": 200},
        {"resource": "clean_firm", "lcoe": 90,  "annual_add_twh": 3,  "max_cumulative_twh": 50},
        {"resource": "storage",    "lcoe": 100, "annual_add_twh": 2,  "max_cumulative_twh": 30},
    ],
    "NYISO": [
        {"resource": "wind",       "lcoe": 81,  "annual_add_twh": 3,  "max_cumulative_twh": 40},
        {"resource": "clean_firm", "lcoe": 90,  "annual_add_twh": 1,  "max_cumulative_twh": 15},
        {"resource": "solar",      "lcoe": 92,  "annual_add_twh": 2,  "max_cumulative_twh": 25},
        {"resource": "storage",    "lcoe": 100, "annual_add_twh": 1,  "max_cumulative_twh": 15},
    ],
    "NEISO": [
        {"resource": "wind",       "lcoe": 73,  "annual_add_twh": 3,  "max_cumulative_twh": 40},
        {"resource": "solar",      "lcoe": 82,  "annual_add_twh": 2,  "max_cumulative_twh": 30},
        {"resource": "clean_firm", "lcoe": 90,  "annual_add_twh": 1,  "max_cumulative_twh": 15},
        {"resource": "storage",    "lcoe": 100, "annual_add_twh": 1,  "max_cumulative_twh": 15},
    ],
}

# ── Procurement Ratio (MWh clean per MWh matched demand) ──────────────────
PROCUREMENT_RATIO = {
    75: 0.80, 80: 0.85, 85: 0.95,
    90: 1.05, 95: 1.15, 99: 1.30, 100: 1.45
}

# ── Scarcity Classification Thresholds ────────────────────────────────────
SCARCITY_BANDS = {
    "abundant":   {"max_ratio": 0.3,  "label": "Abundant",   "color": "#22c55e"},
    "adequate":   {"max_ratio": 0.6,  "label": "Adequate",   "color": "#84cc16"},
    "tightening": {"max_ratio": 0.8,  "label": "Tightening", "color": "#eab308"},
    "scarce":     {"max_ratio": 0.95, "label": "Scarce",     "color": "#f97316"},
    "critical":   {"max_ratio": 999,  "label": "Critical",   "color": "#ef4444"},
}


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def classify_scarcity(demand_ratio):
    """Classify scarcity based on demand/supply ratio."""
    for band_key, band in SCARCITY_BANDS.items():
        if demand_ratio <= band["max_ratio"]:
            return band_key, band["label"], band["color"]
    return "critical", "Critical", "#ef4444"


def interp_dict(data, year):
    """Linearly interpolate a {str_year: value} dict at any year."""
    years = sorted(int(y) for y in data.keys())
    if year <= years[0]:
        return data[str(years[0])]
    if year >= years[-1]:
        return data[str(years[-1])]
    for i in range(len(years) - 1):
        if years[i] <= year <= years[i + 1]:
            t = (year - years[i]) / (years[i + 1] - years[i])
            return data[str(years[i])] + t * (data[str(years[i + 1])] - data[str(years[i])])
    return data[str(years[-1])]


def interpolate_rps_target(iso, year, growth_level):
    """Interpolate RPS clean energy target % for a given ISO/year/growth."""
    return interp_dict(RPS_TARGET_TRAJECTORIES[iso][growth_level], year)


def get_sss_new_fraction(iso, year):
    """Interpolate SSS new-build fraction for any year."""
    return interp_dict(SSS_NEW_BUILD_FRACTION[iso], year)


def get_committed_pipeline_twh(iso, year):
    """Compute TWh locked up by committed hyperscaler clean PPAs at a given year."""
    if iso not in COMMITTED_CLEAN_PIPELINE:
        return 0
    pipeline = COMMITTED_CLEAN_PIPELINE[iso]
    gw = interp_dict(pipeline["phasing_gw"], year)
    return gw * 8.760 * pipeline["capacity_factor"]


def get_procurement_ratio(match_target):
    """Interpolate procurement ratio for any match target."""
    thresholds = sorted(PROCUREMENT_RATIO.keys())
    if match_target <= thresholds[0]:
        return PROCUREMENT_RATIO[thresholds[0]]
    if match_target >= thresholds[-1]:
        return PROCUREMENT_RATIO[thresholds[-1]]
    if match_target in PROCUREMENT_RATIO:
        return PROCUREMENT_RATIO[match_target]
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if lo <= match_target <= hi:
            t = (match_target - lo) / (hi - lo)
            return PROCUREMENT_RATIO[lo] + t * (PROCUREMENT_RATIO[hi] - PROCUREMENT_RATIO[lo])
    return PROCUREMENT_RATIO[thresholds[-1]]


def get_total_buildable(iso, year):
    """Total new-build capacity available on the supply stack for a given ISO/year."""
    years_elapsed = max(0, year - 2025)
    total = 0
    for tier in CLEAN_SUPPLY_STACK[iso]:
        total += min(tier["annual_add_twh"] * years_elapsed, tier["max_cumulative_twh"])
    return total


# ═══════════════════════════════════════════════════════════════════════════
# Supply-side: RPS-mandated build (forced) + SSS transfer
# ═══════════════════════════════════════════════════════════════════════════

def evolve_supply(iso, year, growth_level, demand_twh_at_year):
    """Project clean supply at a future year.

    Two-track model:
    1. Existing supply with SSS policy expirations (transfer, not vanish)
    2. RPS-mandated new build (forced regardless of economics)
    New build splits between SSS and merchant per SSS_NEW_BUILD_FRACTION.
    """
    base = SSS_2025[iso]

    if year == 2025:
        total = base["total_clean_twh"]
        sss_fixed = base["sss_fixed_twh"]
        sss_rps = base["sss_rps_twh"]
        non_sss = base["non_sss_twh"]
        rps_mandated_new = 0
    else:
        # ── Fixed-fleet SSS: policy expiry TRANSFERS to non-SSS ──
        sss_fixed = base["sss_fixed_twh"]
        sss_to_merchant = 0
        if iso in SSS_EXPIRATIONS:
            for exp in SSS_EXPIRATIONS[iso]:
                if year >= exp["year"] and exp.get("component", "fixed") == "fixed":
                    sss_fixed -= exp["twh_shift"]
                    sss_to_merchant += exp["twh_shift"]  # BUG FIX: transfer, don't vanish
        sss_fixed = max(0, sss_fixed)

        # ── RPS-mandated new build (forced regardless of economics) ──
        rps_target = interpolate_rps_target(iso, year, growth_level)
        existing_total_clean = base["total_clean_twh"]
        rps_demand = rps_target * demand_twh_at_year
        rps_mandated_new = max(0, rps_demand - existing_total_clean)

        # Cap new build at what the supply stack can physically deliver
        total_buildable = get_total_buildable(iso, year)
        rps_mandated_new = min(rps_mandated_new, total_buildable)

        # Split new supply into SSS (RPS compliance) and merchant
        avg_sss_frac = get_sss_new_fraction(iso, year)
        new_sss = rps_mandated_new * avg_sss_frac
        new_merchant = rps_mandated_new * (1 - avg_sss_frac)

        sss_rps = base["sss_rps_twh"] + new_sss
        non_sss = base["non_sss_twh"] + sss_to_merchant + new_merchant
        total = sss_fixed + sss_rps + non_sss

    sss = sss_fixed + sss_rps

    # Subtract existing corporate PPAs + committed hyperscaler pipeline
    existing_ppas = EXISTING_CORPORATE_PPAS.get(iso, {}).get("twh", 0)
    committed_pipeline = get_committed_pipeline_twh(iso, year)
    total_claimed = existing_ppas + committed_pipeline
    available_non_sss = max(0, non_sss - total_claimed)

    total = sss_fixed + sss_rps + non_sss

    return {
        "total_clean_twh": round(total, 1),
        "sss_twh": round(sss, 1),
        "sss_fixed_twh": round(max(0, sss_fixed), 1),
        "sss_rps_twh": round(sss_rps, 1),
        "non_sss_twh": round(non_sss, 1),
        "available_non_sss_twh": round(available_non_sss, 1),
        "existing_ppas_twh": existing_ppas,
        "committed_pipeline_twh": round(committed_pipeline, 1),
        "sss_share_of_total": round(sss / total, 4) if total > 0 else 0,
        "rps_mandated_new_twh": round(rps_mandated_new, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Clean premium: COMBINED RPS + voluntary demand on supply stack
# ═══════════════════════════════════════════════════════════════════════════

def walk_supply_stack(iso, year, total_new_demand_twh):
    """Walk up the supply stack with combined demand, return marginal LCOE.

    Both RPS-mandated build and voluntary corporate demand compete for the
    same finite buildable capacity. The marginal cost is set by where
    combined demand lands on the stack.
    """
    wholesale = WHOLESALE_PRICES[iso]
    years_elapsed = max(1, year - 2025)

    if total_new_demand_twh <= 0:
        return wholesale  # No new demand — marginal cost = wholesale

    remaining = total_new_demand_twh
    marginal_lcoe = wholesale
    stack = CLEAN_SUPPLY_STACK[iso]

    for tier in stack:
        tier_capacity = min(tier["annual_add_twh"] * years_elapsed, tier["max_cumulative_twh"])
        if tier_capacity <= 0:
            continue
        if remaining <= tier_capacity:
            marginal_lcoe = tier["lcoe"]
            remaining = 0
            break
        remaining -= tier_capacity
        marginal_lcoe = tier["lcoe"]

    # If demand exceeds all tiers — scarcity pricing
    if remaining > 0:
        scarcity_surcharge = remaining * 3  # $3/MWh per TWh of shortage
        marginal_lcoe = marginal_lcoe + scarcity_surcharge

    return marginal_lcoe


def compute_clean_premium(iso, year, rps_new_demand_twh, corp_demand_twh, available_existing_twh):
    """Compute clean premium from combined demand on the supply stack.

    1. Existing non-SSS supply serves corp demand at ~$0 premium (already built).
    2. Remaining corp demand + ALL RPS new demand compete on the new-build stack.
    3. Marginal premium = LCOE of the tier where COMBINED demand lands - wholesale.
    """
    wholesale = WHOLESALE_PRICES[iso]

    if corp_demand_twh <= 0:
        return 0.0

    # Phase 1: Serve corporate demand from existing merchant supply (near-zero premium)
    corp_remaining = max(0, corp_demand_twh - available_existing_twh)

    if corp_remaining <= 0:
        # All corp demand met by existing merchant — minimal premium
        utilization = corp_demand_twh / available_existing_twh if available_existing_twh > 0 else 0
        return round(utilization * 5, 2)  # 0-5 $/MWh based on utilization

    # Phase 2: Combined demand on the new-build stack
    # RPS mandates consume the cheap tiers; corp demand rides on top
    total_new_demand = rps_new_demand_twh + corp_remaining
    marginal_lcoe = walk_supply_stack(iso, year, total_new_demand)

    premium = max(0, marginal_lcoe - wholesale)
    return round(premium, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Inflection points
# ═══════════════════════════════════════════════════════════════════════════

def compute_inflection_points(iso, year, growth_level, supply_data, demand_twh):
    """Find the participation rate at which each match target hits scarcity."""
    inflections = {}
    available = supply_data["available_non_sss_twh"]
    sss_share = supply_data["sss_share_of_total"]
    rps_mandated = supply_data["rps_mandated_new_twh"]

    total_buildable = get_total_buildable(iso, year)

    for match_target in MATCH_TARGETS:
        procurement_mult = get_procurement_ratio(match_target)
        incremental_need_frac = max(0, match_target / 100 - sss_share)

        inflection_pct = None
        for pct in range(1, 101):
            corp_load = demand_twh * CI_SHARE * (pct / 100)
            corp_eac_demand = corp_load * incremental_need_frac * procurement_mult
            corp_remaining = max(0, corp_eac_demand - available)

            # Combined demand on new-build stack
            combined_new_demand = rps_mandated + corp_remaining
            if combined_new_demand > total_buildable:
                inflection_pct = pct
                break

        inflections[str(match_target)] = {
            "inflection_participation_pct": inflection_pct,
            "procurement_multiplier": round(procurement_mult, 3),
            "total_buildable_twh": round(total_buildable, 1),
            "rps_mandated_twh": round(rps_mandated, 1),
            "remaining_for_voluntary_twh": round(max(0, total_buildable - rps_mandated), 1),
        }

    return inflections


# ═══════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════

def load_demand_data():
    """Load ISO demand from optimizer results."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    demands = {}
    for iso, iso_data in data["results"].items():
        demands[iso] = iso_data["annual_demand_mwh"] / 1e6  # TWh
    return demands


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return None


def save_checkpoint(completed, results):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    cp = {"completed": completed, "results": results, "timestamp": time.time()}
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cp, f)
    os.replace(tmp, CHECKPOINT_PATH)


def save_final_results(results):
    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, OUTPUT_PATH)
    print(f"Saved final results to {OUTPUT_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("EAC Scarcity Analysis — Combined RPS + Voluntary Model (v2)")
    print("=" * 60)

    demands = load_demand_data()
    print(f"Loaded demand data for {len(demands)} ISOs")
    print(f"Time horizons: {TIME_HORIZONS[0]}-{TIME_HORIZONS[-1]} ({len(TIME_HORIZONS)} years)")

    # Print stack summary
    for iso in ISOS:
        ws = WHOLESALE_PRICES[iso]
        print(f"  {iso}: wholesale=${ws}/MWh, 2025 clean={SSS_2025[iso]['total_clean_twh']} TWh")

    checkpoint = load_checkpoint()
    completed_keys = set()
    results = {
        "metadata": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": "v2 — combined RPS + voluntary supply stack",
            "description": "EAC scarcity: RPS mandated build + voluntary corporate demand compete on same supply stack",
            "framework": "Two-track: RPS forced build + voluntary on top. Marginal cost = combined demand on stack.",
            "literature": [
                "Xu et al. (2024) Joule — GenX combined RPS + voluntary on regional supply curves",
                "Gillenwater (2008) Energy Policy — REC supply-demand equilibrium",
                "Denholm et al. (2021) NREL — last few percent cost escalation",
            ],
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
        "wholesale_prices": WHOLESALE_PRICES,
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
        prev = checkpoint.get("results", {})
        for section in ["supply_projections", "scenarios", "inflection_points", "national_summary"]:
            if section in prev:
                results[section] = prev[section]
        print(f"Resumed from checkpoint: {len(completed_keys)} combos already done")

    # ── Phase 1: Supply projections ───────────────────────────────────────
    done = 0
    total_supply = len(ISOS) * len(TIME_HORIZONS) * len(GROWTH_LEVELS)
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

                growth_rate = DEMAND_GROWTH_RATES[iso][growth]
                projected_demand = demand_twh * (1 + growth_rate) ** (year - 2025)

                supply = evolve_supply(iso, year, growth, projected_demand)
                results["supply_projections"][iso][year_key][growth] = supply
                completed_keys.add(combo_key)
                done += 1

            save_checkpoint(list(completed_keys), results)

    print(f"Phase 1 complete: {done}/{total_supply} supply projections")

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
                rps_mandated = supply["rps_mandated_new_twh"]
                total_buildable = get_total_buildable(iso, year)

                scenarios_for_combo = []

                for participation in PARTICIPATION_RATES:
                    for match_target in MATCH_TARGETS:
                        mult = get_procurement_ratio(match_target)

                        # Corporate load (C&I only)
                        corp_load = projected_demand * CI_SHARE * (participation / 100)
                        # SSS pro-rata derate
                        incremental_need_frac = max(0, match_target / 100 - sss_share)
                        corp_eac_demand = corp_load * incremental_need_frac * mult

                        # Combined demand: RPS mandated + corp demand beyond existing
                        corp_beyond_existing = max(0, corp_eac_demand - available)
                        combined_new_demand = rps_mandated + corp_beyond_existing

                        # Scarcity ratio = combined demand / total buildable
                        if total_buildable > 0:
                            demand_ratio = combined_new_demand / total_buildable
                        elif combined_new_demand > 0:
                            demand_ratio = 999
                        else:
                            demand_ratio = 0

                        band_key, label, color = classify_scarcity(demand_ratio)

                        # Marginal premium from combined demand on supply stack
                        premium = compute_clean_premium(
                            iso, year, rps_mandated, corp_eac_demand, available)

                        scenarios_for_combo.append({
                            "participation_pct": participation,
                            "match_target_pct": match_target,
                            "projected_demand_twh": round(projected_demand, 1),
                            "ci_share_pct": round(CI_SHARE * 100, 1),
                            "corp_load_twh": round(corp_load, 1),
                            "sss_pro_rata_pct": round(sss_share * 100, 1),
                            "incremental_need_pct": round(incremental_need_frac * 100, 1),
                            "procurement_ratio": round(mult, 3),
                            "corp_eac_demand_twh": round(corp_eac_demand, 1),
                            "available_supply_twh": round(available, 1),
                            "rps_mandated_new_twh": round(rps_mandated, 1),
                            "combined_new_demand_twh": round(combined_new_demand, 1),
                            "total_buildable_twh": round(total_buildable, 1),
                            "demand_supply_ratio": round(demand_ratio, 3),
                            "scarcity_class": band_key,
                            "scarcity_label": label,
                            "clean_premium_per_mwh": premium,
                        })
                        scenario_count += 1

                results["scenarios"][iso][year_key][growth] = scenarios_for_combo
                completed_keys.add(combo_key)

            save_checkpoint(list(completed_keys), results)
            if year % 5 == 0:
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
                for iso in ISOS)
            total_sss = sum(
                results["supply_projections"][iso][year_key][growth]["sss_twh"]
                for iso in ISOS)
            total_non_sss = sum(
                results["supply_projections"][iso][year_key][growth]["non_sss_twh"]
                for iso in ISOS)
            total_available = sum(
                results["supply_projections"][iso][year_key][growth]["available_non_sss_twh"]
                for iso in ISOS)
            total_demand = sum(
                demands[iso] * (1 + DEMAND_GROWTH_RATES[iso][growth]) ** (year - 2025)
                for iso in ISOS)
            total_rps_new = sum(
                results["supply_projections"][iso][year_key][growth]["rps_mandated_new_twh"]
                for iso in ISOS)
            total_buildable = sum(get_total_buildable(iso, year) for iso in ISOS)

            results["national_summary"][year_key][growth] = {
                "total_clean_twh": round(total_clean, 1),
                "total_sss_twh": round(total_sss, 1),
                "total_non_sss_twh": round(total_non_sss, 1),
                "total_available_twh": round(total_available, 1),
                "total_demand_twh": round(total_demand, 1),
                "total_rps_mandated_new_twh": round(total_rps_new, 1),
                "total_buildable_twh": round(total_buildable, 1),
                "national_sss_pct": round(total_sss / total_clean * 100, 1) if total_clean > 0 else 0,
            }

    # ── Save ──────────────────────────────────────────────────────────────
    save_final_results(results)

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for iso in ISOS:
        s2025 = results["supply_projections"][iso]["2025"]["Medium"]
        s2030 = results["supply_projections"][iso]["2030"]["Medium"]
        s2035 = results["supply_projections"][iso]["2035"]["Medium"]
        pipeline_2025 = s2025.get("committed_pipeline_twh", 0)
        pipeline_note = f" (pipeline: {pipeline_2025} TWh)" if pipeline_2025 else ""
        print(f"\n{iso}:")
        print(f"  2025: {s2025['available_non_sss_twh']} TWh avail{pipeline_note} | SSS {s2025['sss_twh']} TWh ({round(s2025['sss_share_of_total']*100,1)}%)")
        print(f"  2030: {s2030['available_non_sss_twh']} TWh avail | RPS new build: {s2030['rps_mandated_new_twh']} TWh | SSS {s2030['sss_twh']} TWh ({round(s2030['sss_share_of_total']*100,1)}%)")
        print(f"  2035: {s2035['available_non_sss_twh']} TWh avail | RPS new build: {s2035['rps_mandated_new_twh']} TWh | SSS {s2035['sss_twh']} TWh ({round(s2035['sss_share_of_total']*100,1)}%)")
        inf = results["inflection_points"][iso]["2030"]["Medium"]
        for mt in ["90", "95", "100"]:
            ip = inf.get(mt, {}).get("inflection_participation_pct")
            rps = inf.get(mt, {}).get("rps_mandated_twh", 0)
            remain = inf.get(mt, {}).get("remaining_for_voluntary_twh", 0)
            print(f"  2030 @{mt}% match: scarcity at {ip}% participation (RPS consumes {rps} TWh, {remain} TWh left for vol)" if ip else f"  2030 @{mt}% match: no scarcity at 100% (RPS: {rps} TWh, remaining: {remain} TWh)")

    nat = results["national_summary"]["2025"]["Medium"]
    print(f"\nNational 2025 (Medium growth):")
    print(f"  Total clean: {nat['total_clean_twh']} TWh")
    print(f"  SSS: {nat['total_sss_twh']} TWh ({nat['national_sss_pct']}%)")
    print(f"  Available for corporate: {nat['total_available_twh']} TWh")
    print(f"  Total demand: {nat['total_demand_twh']} TWh")

    nat30 = results["national_summary"]["2030"]["Medium"]
    print(f"\nNational 2030 (Medium growth):")
    print(f"  Total clean: {nat30['total_clean_twh']} TWh (RPS new: {nat30['total_rps_mandated_new_twh']} TWh)")
    print(f"  SSS: {nat30['total_sss_twh']} TWh ({nat30['national_sss_pct']}%)")
    print(f"  Available for corporate: {nat30['total_available_twh']} TWh")
    print(f"  Total buildable: {nat30['total_buildable_twh']} TWh")
    print(f"\nTotal scenarios: {scenario_count}")


if __name__ == "__main__":
    main()
