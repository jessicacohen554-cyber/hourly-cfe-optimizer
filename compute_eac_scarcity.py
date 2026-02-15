#!/usr/bin/env python3
"""
EAC Scarcity Analysis — Supply-stack / marginal-cost model.

New clean capacity enters only when economically viable (LCOE < wholesale + RPS adder).
Clean premium = marginal cost of the next MWh on the supply stack to serve corporate demand.

Saves results to dashboard/eac_scarcity_results.json with incremental
checkpointing after each ISO×year combo to avoid losing work.

Based on SPEC.md Supply Stack methodology (Feb 15, 2026).
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
# Used to drive the RPS demand adder — NOT to directly set supply volume.
# As RPS targets rise, the compliance premium (adder) increases, which
# determines how much of the supply stack is economic for compliance.
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
# Large tech companies have committed to specific nuclear PPAs in PJM that
# lock up clean generation. This is modeled as a supply reduction (not demand
# growth) because the demand is disproportionately clean-energy-targeted.
# Phased in GW → TWh at 90% capacity factor (1 GW ≈ 7.884 TWh/yr).
COMMITTED_CLEAN_PIPELINE = {
    "PJM": {
        # {year: cumulative_gw} — interpolated for intermediate years
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
TIME_HORIZONS = [2025, 2030, 2035, 2040, 2045, 2050]
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

# ── RPS Base Adder ($/MWh) — observed 2025 REC prices ─────────────────────
# The price signal RPS compliance creates above wholesale.
# New clean is built when LCOE < wholesale + rps_adder.
RPS_BASE_ADDER = {
    "CAISO": 10,   # CA SB100 REC ~$5-15/MWh
    "ERCOT": 2,    # No binding RPS, minimal voluntary REC value
    "PJM":   8,    # Blended state RPS REC prices ~$5-15/MWh
    "NYISO": 22,   # NY Tier 1 RECs ~$20-25/MWh
    "NEISO": 30,   # MA/CT Class I RECs ~$25-35/MWh
}

# ── Procurement Ratio (MWh clean per MWh matched demand) ──────────────────
# Accounts for temporal mismatch — higher targets need more over-procurement.
# Derived from VRE curtailment physics (not optimizer results).
# National average curve; ISO-specific refinement possible in future.
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
    # GW → TWh: GW × 8760 hours × capacity factor / 1000
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


def compute_rps_adder(iso, year, growth_level):
    """Compute RPS demand adder at a given year.

    Adder rises proportionally with RPS target. As mandates increase,
    compliance demand rises, pushing REC prices higher.
    """
    rps_now = interpolate_rps_target(iso, year, growth_level)
    rps_2025 = interpolate_rps_target(iso, 2025, growth_level)
    if rps_2025 <= 0:
        return RPS_BASE_ADDER[iso]
    # Convex scaling: adder rises faster-than-linearly as targets approach 100%
    # ratio^1.3 gives gentle convexity
    ratio = rps_now / rps_2025
    return RPS_BASE_ADDER[iso] * (ratio ** 1.3)


# ═══════════════════════════════════════════════════════════════════════════
# Supply-side: economics-driven growth
# ═══════════════════════════════════════════════════════════════════════════

def compute_new_economic_supply(iso, year, growth_level):
    """Compute new clean supply that has been built because it's economic.

    New clean enters when LCOE < wholesale + RPS adder.
    Returns total new TWh added since 2025, and the supply stack with
    cumulative capacity at each tier.
    """
    if year <= 2025:
        return 0, []

    wholesale = WHOLESALE_PRICES[iso]
    rps_adder = compute_rps_adder(iso, year, growth_level)
    price_signal = wholesale + rps_adder
    years_elapsed = year - 2025

    stack = CLEAN_SUPPLY_STACK[iso]
    new_total = 0
    tier_details = []

    for tier in stack:
        if tier["lcoe"] <= price_signal:
            # This tier is economic — capacity has been building
            added = min(tier["annual_add_twh"] * years_elapsed, tier["max_cumulative_twh"])
            new_total += added
            tier_details.append({
                "resource": tier["resource"],
                "lcoe": tier["lcoe"],
                "added_twh": round(added, 1),
                "economic": True,
            })
        else:
            # Not yet economic — no new build in this tier
            tier_details.append({
                "resource": tier["resource"],
                "lcoe": tier["lcoe"],
                "added_twh": 0,
                "economic": False,
            })

    return new_total, tier_details


def evolve_supply(iso, year, growth_level, demand_twh_at_year):
    """Project clean supply at a future year using economics-driven model.

    Existing clean is fixed. New clean only enters when LCOE < wholesale + RPS adder.
    New supply split between SSS (RPS compliance) and merchant per SSS_NEW_BUILD_FRACTION.
    """
    base = SSS_2025[iso]

    if year == 2025:
        total = base["total_clean_twh"]
        sss_fixed = base["sss_fixed_twh"]
        sss_rps = base["sss_rps_twh"]
        non_sss = base["non_sss_twh"]
        rps_adder = RPS_BASE_ADDER[iso]
        new_economic = 0
    else:
        # ── Fixed-fleet SSS (constant, policy expiry only) ──
        sss_fixed = base["sss_fixed_twh"]
        if iso in SSS_EXPIRATIONS:
            for exp in SSS_EXPIRATIONS[iso]:
                if year >= exp["year"] and exp.get("component", "fixed") == "fixed":
                    sss_fixed -= exp["twh_shift"]
        sss_fixed = max(0, sss_fixed)

        # ── New economic supply ──
        new_economic, _ = compute_new_economic_supply(iso, year, growth_level)
        rps_adder = compute_rps_adder(iso, year, growth_level)

        # Split new supply into SSS (RPS compliance) and merchant
        avg_sss_frac = get_sss_new_fraction(iso, year)
        new_sss = new_economic * avg_sss_frac
        new_merchant = new_economic * (1 - avg_sss_frac)

        sss_rps = base["sss_rps_twh"] + new_sss
        non_sss = base["non_sss_twh"] + new_merchant
        total = sss_fixed + sss_rps + non_sss

    sss = sss_fixed + sss_rps

    # Subtract existing corporate PPAs + committed hyperscaler pipeline
    existing_ppas = EXISTING_CORPORATE_PPAS.get(iso, {}).get("twh", 0)
    committed_pipeline = get_committed_pipeline_twh(iso, year)
    total_claimed = existing_ppas + committed_pipeline
    available_non_sss = max(0, non_sss - total_claimed)

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
        "rps_adder": round(rps_adder, 1),
        "new_economic_twh": round(new_economic, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Clean premium: marginal cost from supply stack
# ═══════════════════════════════════════════════════════════════════════════

def compute_clean_premium(iso, year, growth_level, corp_demand_twh, available_existing_twh):
    """Walk the supply stack to compute marginal clean premium.

    1. Existing non-SSS supply is available at ~$0 premium (already built).
    2. If corp demand exceeds existing, walk up the new-build cost stack.
    3. Marginal premium = LCOE of the tier being consumed - wholesale.
    4. If demand exceeds entire stack, scarcity pricing kicks in.
    """
    wholesale = WHOLESALE_PRICES[iso]

    if corp_demand_twh <= 0:
        return 0.0

    # Phase 1: Serve from existing supply (near-zero premium)
    remaining = corp_demand_twh - available_existing_twh
    if remaining <= 0:
        # All demand met by existing merchant clean — minimal premium
        # Still a small premium for the match-target difficulty (temporal value)
        utilization = corp_demand_twh / available_existing_twh if available_existing_twh > 0 else 0
        return round(utilization * 5, 2)  # 0-5 $/MWh based on utilization

    # Phase 2: Walk up the new-build supply stack
    years_elapsed = max(1, year - 2025)
    stack = CLEAN_SUPPLY_STACK[iso]
    marginal_lcoe = wholesale  # default if no tiers consumed

    for tier in stack:
        tier_capacity = min(tier["annual_add_twh"] * years_elapsed, tier["max_cumulative_twh"])
        if remaining <= tier_capacity:
            marginal_lcoe = tier["lcoe"]
            remaining = 0
            break
        remaining -= tier_capacity
        marginal_lcoe = tier["lcoe"]

    # Phase 3: If demand exceeds all tiers — scarcity pricing
    if remaining > 0:
        # Beyond the buildable stack — exponential scarcity
        scarcity_surcharge = remaining * 3  # $3/MWh per TWh of shortage
        marginal_lcoe = marginal_lcoe + scarcity_surcharge

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

    for match_target in MATCH_TARGETS:
        procurement_mult = get_procurement_ratio(match_target)
        incremental_need_frac = max(0, match_target / 100 - sss_share)

        # Find participation rate where demand exceeds available + buildable supply
        total_buildable = available
        years_elapsed = max(1, year - 2025)
        for tier in CLEAN_SUPPLY_STACK[iso]:
            total_buildable += min(tier["annual_add_twh"] * years_elapsed, tier["max_cumulative_twh"])

        inflection_pct = None
        for pct in range(1, 101):
            corp_demand = demand_twh * CI_SHARE * (pct / 100) * incremental_need_frac * procurement_mult
            if corp_demand > total_buildable:
                inflection_pct = pct
                break

        inflections[str(match_target)] = {
            "inflection_participation_pct": inflection_pct,
            "procurement_multiplier": round(procurement_mult, 3),
            "max_supportable_twh": round(total_buildable / procurement_mult, 1) if procurement_mult > 0 else 0,
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
    print("EAC Scarcity Analysis — Supply Stack Model")
    print("=" * 60)

    demands = load_demand_data()
    print(f"Loaded demand data for {len(demands)} ISOs")

    # Print stack summary
    for iso in ISOS:
        ws = WHOLESALE_PRICES[iso]
        adder = RPS_BASE_ADDER[iso]
        print(f"  {iso}: wholesale=${ws}, RPS adder=${adder}, threshold=${ws+adder}/MWh")

    checkpoint = load_checkpoint()
    completed_keys = set()
    results = {
        "metadata": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "EAC scarcity analysis: supply-stack / marginal-cost model",
            "framework": "New clean enters when LCOE < wholesale + RPS adder. Premium = marginal cost on stack.",
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
                        mult = get_procurement_ratio(match_target)

                        # Corporate load (C&I only)
                        corp_load = projected_demand * CI_SHARE * (participation / 100)
                        # SSS pro-rata derate
                        incremental_need_frac = max(0, match_target / 100 - sss_share)
                        corp_eac_demand = corp_load * incremental_need_frac * mult

                        # Demand ratio against existing available supply
                        demand_ratio = corp_eac_demand / available if available > 0 else 999
                        band_key, label, color = classify_scarcity(demand_ratio)

                        # Marginal premium from supply stack
                        premium = compute_clean_premium(
                            iso, year, growth, corp_eac_demand, available)

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
                            "demand_supply_ratio": round(demand_ratio, 3),
                            "scarcity_class": band_key,
                            "scarcity_label": label,
                            "clean_premium_per_mwh": premium,
                        })
                        scenario_count += 1

                results["scenarios"][iso][year_key][growth] = scenarios_for_combo
                completed_keys.add(combo_key)

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

            results["national_summary"][year_key][growth] = {
                "total_clean_twh": round(total_clean, 1),
                "total_sss_twh": round(total_sss, 1),
                "total_non_sss_twh": round(total_non_sss, 1),
                "total_available_twh": round(total_available, 1),
                "total_demand_twh": round(total_demand, 1),
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
        s2035 = results["supply_projections"][iso]["2035"]["Medium"]
        adder_2035 = s2035.get("rps_adder", "?")
        pipeline_2025 = s2025.get("committed_pipeline_twh", 0)
        pipeline_2035 = s2035.get("committed_pipeline_twh", 0)
        pipeline_note = f" (pipeline: {pipeline_2025} TWh)" if pipeline_2025 else ""
        print(f"\n{iso}:")
        print(f"  2025: {s2025['available_non_sss_twh']} TWh available{pipeline_note} | RPS adder ${s2025.get('rps_adder', '?')}/MWh")
        pipeline_note_35 = f" | pipeline: {pipeline_2035} TWh" if pipeline_2035 else ""
        print(f"  2035: {s2035['available_non_sss_twh']} TWh available ({s2035.get('new_economic_twh', 0)} new){pipeline_note_35} | RPS adder ${adder_2035}/MWh")
        inf = results["inflection_points"][iso]["2030"]["Medium"]
        for mt in ["90", "95", "100"]:
            ip = inf.get(mt, {}).get("inflection_participation_pct")
            print(f"  2030 @{mt}% match: scarcity at {ip}% participation" if ip else f"  2030 @{mt}% match: no scarcity at 100% participation")

    nat = results["national_summary"]["2025"]["Medium"]
    print(f"\nNational 2025 (Medium growth):")
    print(f"  Total clean: {nat['total_clean_twh']} TWh")
    print(f"  SSS: {nat['total_sss_twh']} TWh ({nat['national_sss_pct']}%)")
    print(f"  Available for corporate: {nat['total_available_twh']} TWh")
    print(f"  Total demand: {nat['total_demand_twh']} TWh")
    print(f"\nTotal scenarios: {scenario_count}")


if __name__ == "__main__":
    main()
