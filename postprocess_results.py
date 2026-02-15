#!/usr/bin/env python3
"""
Post-Processing Corrections & Overlays
========================================
Applies corrections to cached optimizer results without re-running the optimizer.

Corrections:
  1. CO₂ monotonicity enforcement (running-max across thresholds)
  2. 45Q offset correction ($29 → $27.5, +$1.5/MWh to CCS LCOE)
  3. Without-45Q toggle layer (CCS cost without 45Q, CF-dependent dispatchable LCOE)
  4. NEISO winter gas pipeline constraint (+$13/MWh CCS, +$4/MWh wholesale)
  5. CCS vs LDES crossover analysis

Reads:  data/optimizer_cache.json (canonical) or dashboard/overprocure_results.json
Writes: dashboard/overprocure_results.json (corrected)
        data/postprocess_analysis.json (analysis output)

See SPEC.md §22 for methodology documentation.
"""

import json
import os
import sys
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, 'data', 'optimizer_cache.json')
RESULTS_PATH = os.path.join(SCRIPT_DIR, 'dashboard', 'overprocure_results.json')
ANALYSIS_PATH = os.path.join(SCRIPT_DIR, 'data', 'postprocess_analysis.json')

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
THRESHOLDS = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]

# ══════════════════════════════════════════════════════════════════════════════
# COST TABLES (duplicated from optimizer for standalone operation)
# ══════════════════════════════════════════════════════════════════════════════

WHOLESALE_PRICES = {
    'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42, 'NEISO': 41,
}

WHOLESALE_FUEL_ADJUSTMENTS = {
    'CAISO': {'Low': -5, 'Medium': 0, 'High': 10},
    'ERCOT': {'Low': -7, 'Medium': 0, 'High': 12},
    'PJM':   {'Low': -6, 'Medium': 0, 'High': 11},
    'NYISO': {'Low': -4, 'Medium': 0, 'High': 8},
    'NEISO': {'Low': -4, 'Medium': 0, 'High': 8},
}

FULL_LCOE_TABLES = {
    'solar': {
        'Low':    {'CAISO': 45, 'ERCOT': 40, 'PJM': 50, 'NYISO': 70, 'NEISO': 62},
        'Medium': {'CAISO': 60, 'ERCOT': 54, 'PJM': 65, 'NYISO': 92, 'NEISO': 82},
        'High':   {'CAISO': 78, 'ERCOT': 70, 'PJM': 85, 'NYISO': 120, 'NEISO': 107},
    },
    'wind': {
        'Low':    {'CAISO': 55, 'ERCOT': 30, 'PJM': 47, 'NYISO': 61, 'NEISO': 55},
        'Medium': {'CAISO': 73, 'ERCOT': 40, 'PJM': 62, 'NYISO': 81, 'NEISO': 73},
        'High':   {'CAISO': 95, 'ERCOT': 52, 'PJM': 81, 'NYISO': 105, 'NEISO': 95},
    },
    'clean_firm': {
        'Low':    {'CAISO': 58, 'ERCOT': 56, 'PJM': 48, 'NYISO': 64, 'NEISO': 69},
        'Medium': {'CAISO': 79, 'ERCOT': 79, 'PJM': 68, 'NYISO': 86, 'NEISO': 92},
        'High':   {'CAISO': 115, 'ERCOT': 115, 'PJM': 108, 'NYISO': 136, 'NEISO': 143},
    },
    'ccs_ccgt': {
        'Low':    {'CAISO': 58, 'ERCOT': 52, 'PJM': 62, 'NYISO': 78, 'NEISO': 75},
        'Medium': {'CAISO': 86, 'ERCOT': 71, 'PJM': 79, 'NYISO': 99, 'NEISO': 96},
        'High':   {'CAISO': 115, 'ERCOT': 92, 'PJM': 102, 'NYISO': 128, 'NEISO': 122},
    },
    'battery': {
        'Low':    {'CAISO': 77, 'ERCOT': 69, 'PJM': 74, 'NYISO': 81, 'NEISO': 79},
        'Medium': {'CAISO': 102, 'ERCOT': 92, 'PJM': 98, 'NYISO': 108, 'NEISO': 105},
        'High':   {'CAISO': 133, 'ERCOT': 120, 'PJM': 127, 'NYISO': 140, 'NEISO': 137},
    },
    'ldes': {
        'Low':    {'CAISO': 135, 'ERCOT': 116, 'PJM': 128, 'NYISO': 150, 'NEISO': 143},
        'Medium': {'CAISO': 180, 'ERCOT': 155, 'PJM': 170, 'NYISO': 200, 'NEISO': 190},
        'High':   {'CAISO': 234, 'ERCOT': 202, 'PJM': 221, 'NYISO': 260, 'NEISO': 247},
    },
}

FULL_TRANSMISSION_TABLES = {
    'wind': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
        'Medium': {'CAISO': 8, 'ERCOT': 6, 'PJM': 10, 'NYISO': 14, 'NEISO': 12},
        'High': {'CAISO': 14, 'ERCOT': 10, 'PJM': 18, 'NYISO': 22, 'NEISO': 20},
    },
    'solar': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 2, 'NYISO': 3, 'NEISO': 3},
        'Medium': {'CAISO': 3, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
        'High': {'CAISO': 6, 'ERCOT': 5, 'PJM': 9, 'NYISO': 12, 'NEISO': 10},
    },
    'clean_firm': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'Medium': {'CAISO': 3, 'ERCOT': 2, 'PJM': 3, 'NYISO': 5, 'NEISO': 4},
        'High': {'CAISO': 6, 'ERCOT': 4, 'PJM': 6, 'NYISO': 9, 'NEISO': 7},
    },
    'ccs_ccgt': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
        'High': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
    },
    'battery': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 1, 'NEISO': 1},
        'Medium': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'High': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
    },
    'ldes': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low': {'CAISO': 1, 'ERCOT': 1, 'PJM': 1, 'NYISO': 2, 'NEISO': 2},
        'Medium': {'CAISO': 2, 'ERCOT': 2, 'PJM': 3, 'NYISO': 4, 'NEISO': 3},
        'High': {'CAISO': 4, 'ERCOT': 3, 'PJM': 5, 'NYISO': 7, 'NEISO': 6},
    },
    'hydro': {
        'None': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Low': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'Medium': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
        'High': {'CAISO': 0, 'ERCOT': 0, 'PJM': 0, 'NYISO': 0, 'NEISO': 0},
    },
}

GRID_MIX_SHARES = {
    'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'ccs_ccgt': 0, 'hydro': 9.5},
    'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'ccs_ccgt': 0, 'hydro': 0.1},
    'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'ccs_ccgt': 0, 'hydro': 1.8},
    'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'ccs_ccgt': 0, 'hydro': 15.9},
    'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'ccs_ccgt': 0, 'hydro': 4.4},
}

RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']

# 45Q credit: corrected value
FOURTY_FIVE_Q_OFFSET_CORRECTED = 27.5  # $85/ton × 0.323 tCO2/MWh captured
FOURTY_FIVE_Q_OFFSET_ORIGINAL = 29.0   # What the optimizer used

# CCS LCOE decomposition (NETL Baseline Rev 4a)
CCS_CAPITAL_SHARE = 0.55    # Capital recovery portion of LCOE
CCS_FIXED_OM_SHARE = 0.08   # Fixed O&M (scales with CF)
CCS_FUEL_SHARE = 0.30       # Fuel (constant per MWh)
CCS_VOM_TS_SHARE = 0.07     # Variable O&M + T&S (constant per MWh)
CCS_REFERENCE_CF = 0.85     # NETL reference capacity factor

# NEISO winter gas pipeline constraint (Algonquin Citygates)
NEISO_CCS_GAS_ADDER = 13.13    # $/MWh annualized: 7 HR × $7.50 premium × 0.25 winter fraction
NEISO_WHOLESALE_ADDER = 4.0    # $/MWh annualized: winter gas on marginal pricing


def load_results():
    """Load optimizer results from cache (preferred) or dashboard file."""
    for path in [CACHE_PATH, RESULTS_PATH]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded: {path} ({os.path.getsize(path) / 1024:.0f} KB)")
            return data
    print("ERROR: No results file found!")
    sys.exit(1)


def decode_scenario_key(key):
    """Decode scenario key like 'MMM_M_M' into toggle levels."""
    level_map = {'L': 'Low', 'M': 'Medium', 'H': 'High', 'N': 'None'}
    parts = key.split('_')
    gen_part = parts[0]  # 3 chars: renewable, firm, storage
    fuel = level_map.get(parts[1], 'Medium')
    tx = level_map.get(parts[2], 'Medium')
    renewable = level_map.get(gen_part[0], 'Medium')
    firm = level_map.get(gen_part[1], 'Medium')
    storage = level_map.get(gen_part[2], 'Medium')
    return renewable, firm, storage, fuel, tx


def ccs_lcoe_corrected_45q(iso, firm_level):
    """Get CCS LCOE with corrected 45Q offset ($27.5 instead of $29)."""
    table_lcoe = FULL_LCOE_TABLES['ccs_ccgt'][firm_level][iso]
    # Tables have $29 offset baked in. To correct to $27.5, add $1.5
    return table_lcoe + (FOURTY_FIVE_Q_OFFSET_ORIGINAL - FOURTY_FIVE_Q_OFFSET_CORRECTED)


def ccs_lcoe_no45q(iso, firm_level):
    """Get CCS LCOE without any 45Q offset (gross LCOE)."""
    table_lcoe = FULL_LCOE_TABLES['ccs_ccgt'][firm_level][iso]
    # Tables have $29 offset baked in. Add it back to get gross/pre-45Q LCOE
    return table_lcoe + FOURTY_FIVE_Q_OFFSET_ORIGINAL


def ccs_lcoe_dispatchable(lcoe_no45q, capacity_factor):
    """
    CCS LCOE at a given capacity factor (dispatchable operation).

    Without 45Q, CCS has no incentive for baseload dispatch.
    Capital + fixed O&M costs scale inversely with CF.
    Fuel + variable O&M costs are constant per MWh.

    Formula: LCOE(CF) = LCOE_ref × ((fixed_share × CF_ref / CF) + variable_share)
    Where fixed_share = capital + fixed_OM = 0.63, variable_share = fuel + VOM = 0.37
    """
    fixed_share = CCS_CAPITAL_SHARE + CCS_FIXED_OM_SHARE  # 0.63
    variable_share = CCS_FUEL_SHARE + CCS_VOM_TS_SHARE     # 0.37

    if capacity_factor <= 0.01:
        return lcoe_no45q * 10  # Effectively infinite at near-zero CF

    cf_ratio = CCS_REFERENCE_CF / capacity_factor
    return lcoe_no45q * (fixed_share * cf_ratio + variable_share)


def compute_costs_for_scenario(iso, resource_mix, procurement_pct, battery_pct,
                                ldes_pct, match_score, scenario_key,
                                apply_45q=True, neiso_gas_adder=False):
    """
    Recalculate costs for a scenario with optional corrections.

    Args:
        apply_45q: If False, remove 45Q offset and use CF-dependent CCS LCOE
        neiso_gas_adder: If True, apply NEISO winter gas pipeline constraint
    """
    renewable, firm, storage, fuel, tx = decode_scenario_key(scenario_key)

    # Build LCOE map
    lcoe_map = {
        'solar': FULL_LCOE_TABLES['solar'][renewable][iso],
        'wind': FULL_LCOE_TABLES['wind'][renewable][iso],
        'clean_firm': FULL_LCOE_TABLES['clean_firm'][firm][iso],
        'battery': FULL_LCOE_TABLES['battery'][storage][iso],
        'ldes': FULL_LCOE_TABLES['ldes'][storage][iso],
        'hydro': 0,
    }

    # CCS LCOE depends on 45Q toggle
    if apply_45q:
        # Use corrected 45Q offset ($27.5 instead of $29) → +$1.5/MWh vs original tables
        lcoe_map['ccs_ccgt'] = ccs_lcoe_corrected_45q(iso, firm)
    else:
        # No 45Q: use base LCOE + CF-dependent adjustment
        base_no45q = ccs_lcoe_no45q(iso, firm)
        ccs_share = resource_mix.get('ccs_ccgt', 0) / 100.0
        procurement_factor = procurement_pct / 100.0

        # CCS effective CF = CCS share × procurement / 100
        # In baseload model, CCS runs all hours. Without 45Q, it dispatches
        # only during gap hours. Approximate CF = what fraction of demand
        # hours CCS actually needs to fill.
        #
        # Conservative estimate: CCS dispatch CF ≈ (1 - match_without_ccs) / ccs_contribution
        # Since we don't have match_without_ccs, use: CF = min(1.0, gap_fraction / ccs_fraction)
        # where gap_fraction ≈ 1 - (match_score/100 - ccs_contribution)
        # Simplified: assume CCS dispatches proportional to its share, capped at its gap-filling role
        ccs_fraction_of_demand = ccs_share * procurement_factor
        if ccs_fraction_of_demand > 0.01:
            # Approximate: CCS doesn't run baseload, it fills gaps
            # Gap hours ≈ (100 - match_score + ccs_contribution × 100) / 100
            # CCS CF ≈ gap_hours / 8760 — but we approximate by comparing
            # CCS's demand fraction to total gaps
            estimated_cf = min(0.85, max(0.20, ccs_fraction_of_demand * 0.8))
            lcoe_map['ccs_ccgt'] = ccs_lcoe_dispatchable(base_no45q, estimated_cf)
        else:
            lcoe_map['ccs_ccgt'] = base_no45q

    # Transmission adders
    tx_map = {}
    for rtype in FULL_TRANSMISSION_TABLES:
        tx_map[rtype] = FULL_TRANSMISSION_TABLES[rtype][tx][iso]

    # Wholesale price with fuel adjustment
    wholesale = WHOLESALE_PRICES[iso] + WHOLESALE_FUEL_ADJUSTMENTS[iso][fuel]
    wholesale = max(5, wholesale)

    # NEISO gas adder
    if neiso_gas_adder and iso == 'NEISO':
        wholesale += NEISO_WHOLESALE_ADDER

    grid_shares = GRID_MIX_SHARES[iso]
    procurement_factor = procurement_pct / 100.0

    total_cost_per_demand = 0.0

    for rtype in RESOURCE_TYPES:
        pct = resource_mix.get(rtype, 0)
        if pct <= 0:
            continue

        resource_fraction = procurement_factor * (pct / 100.0)
        resource_pct_of_demand = resource_fraction * 100.0

        existing_share = grid_shares.get(rtype, 0)
        existing_pct = min(resource_pct_of_demand, existing_share)
        new_pct = max(0, resource_pct_of_demand - existing_share)

        if rtype == 'hydro':
            cost_per_demand = resource_pct_of_demand / 100.0 * wholesale
        else:
            new_build_cost = lcoe_map.get(rtype, 0) + tx_map.get(rtype, 0)

            # NEISO gas adder on CCS fuel costs
            if neiso_gas_adder and iso == 'NEISO' and rtype == 'ccs_ccgt':
                new_build_cost += NEISO_CCS_GAS_ADDER

            cost_per_demand = (existing_pct / 100.0 * wholesale) + \
                              (new_pct / 100.0 * new_build_cost)

        total_cost_per_demand += cost_per_demand

    # Battery storage cost
    battery_cost_rate = lcoe_map['battery'] + tx_map.get('battery', 0)
    battery_cost = (battery_pct / 100.0) * battery_cost_rate
    total_cost_per_demand += battery_cost

    # LDES storage cost
    ldes_cost_rate = lcoe_map['ldes'] + tx_map.get('ldes', 0)
    ldes_cost = (ldes_pct / 100.0) * ldes_cost_rate
    total_cost_per_demand += ldes_cost

    # Effective cost per useful MWh
    matched_fraction = match_score / 100.0 if match_score > 0 else 1.0
    effective_cost = total_cost_per_demand / matched_fraction

    return {
        'total_cost': round(total_cost_per_demand, 2),
        'effective_cost': round(effective_cost, 2),
        'incremental': round(effective_cost - wholesale, 2),
        'wholesale': wholesale,
    }


def fix_co2_monotonicity(data):
    """Enforce CO₂ non-decreasing across thresholds (running-max)."""
    print("\n  [1] CO₂ Monotonicity Enforcement")
    fixes = 0

    for iso in ISOS:
        if iso not in data['results']:
            continue
        thresholds_data = data['results'][iso].get('thresholds', {})

        # Collect all scenario keys
        all_keys = set()
        for t_str in thresholds_data:
            all_keys.update(thresholds_data[t_str].get('scenarios', {}).keys())

        for sk in all_keys:
            prev_co2 = 0
            for threshold in THRESHOLDS:
                t_str = str(threshold)
                if t_str not in thresholds_data:
                    continue
                scenario = thresholds_data[t_str].get('scenarios', {}).get(sk)
                if not scenario:
                    continue

                co2 = scenario.get('co2_abated', {})
                if not isinstance(co2, dict):
                    continue

                current = co2.get('total_co2_abated_tons', 0)
                if isinstance(current, (int, float)) and current < prev_co2:
                    co2['total_co2_abated_tons'] = prev_co2
                    co2['monotonicity_corrected'] = True
                    co2['original_total_co2_abated_tons'] = current
                    fixes += 1

                prev_co2 = max(prev_co2, current if isinstance(current, (int, float)) else 0)

    print(f"      {fixes} CO₂ values corrected across all scenarios")
    return fixes


def fix_45q_offset(data):
    """Apply 45Q offset correction (+$1.5/MWh to CCS costs) to existing results."""
    print("\n  [2] 45Q Offset Correction ($29 → $27.5)")
    corrections = 0

    for iso in ISOS:
        if iso not in data['results']:
            continue
        thresholds_data = data['results'][iso].get('thresholds', {})

        for t_str in thresholds_data:
            scenarios = thresholds_data[t_str].get('scenarios', {})
            for sk, scenario in scenarios.items():
                mix = scenario.get('resource_mix', {})
                ccs_pct = mix.get('ccs_ccgt', 0)
                if ccs_pct <= 0:
                    continue

                # CCS exists in mix — recalculate costs with corrected 45Q
                costs = compute_costs_for_scenario(
                    iso, mix,
                    scenario.get('procurement_pct', 100),
                    scenario.get('battery_dispatch_pct', 0),
                    scenario.get('ldes_dispatch_pct', 0),
                    scenario.get('hourly_match_score', 0),
                    sk,
                    apply_45q=True,
                    neiso_gas_adder=False,
                )

                # Update simplified costs dict
                scenario['costs'] = costs
                corrections += 1

    print(f"      {corrections} scenario costs recalculated with corrected 45Q")
    return corrections


def add_no45q_overlay(data):
    """Add without-45Q cost overlay to every scenario with CCS in the mix."""
    print("\n  [3] Without-45Q Toggle Layer")
    overlays = 0

    for iso in ISOS:
        if iso not in data['results']:
            continue
        thresholds_data = data['results'][iso].get('thresholds', {})

        for t_str in thresholds_data:
            scenarios = thresholds_data[t_str].get('scenarios', {})
            for sk, scenario in scenarios.items():
                mix = scenario.get('resource_mix', {})
                ccs_pct = mix.get('ccs_ccgt', 0)

                # Always compute no-45Q costs (even if CCS=0, to have consistent data)
                no45q_costs = compute_costs_for_scenario(
                    iso, mix,
                    scenario.get('procurement_pct', 100),
                    scenario.get('battery_dispatch_pct', 0),
                    scenario.get('ldes_dispatch_pct', 0),
                    scenario.get('hourly_match_score', 0),
                    sk,
                    apply_45q=False,
                    neiso_gas_adder=False,
                )

                scenario['no_45q_costs'] = no45q_costs

                if ccs_pct > 0:
                    # Also compute the CCS vs LDES crossover info
                    renewable, firm, storage, fuel, tx = decode_scenario_key(sk)
                    ldes_cost = FULL_LCOE_TABLES['ldes'][storage][iso] + \
                                FULL_TRANSMISSION_TABLES['ldes'][tx][iso]
                    ccs_no45q_base = ccs_lcoe_no45q(iso, firm)
                    ccs_tx = FULL_TRANSMISSION_TABLES['ccs_ccgt'][tx][iso]

                    # Find crossover CF where CCS = LDES
                    # LCOE(CF) = base × (0.63 × 0.85/CF + 0.37) + tx = ldes_cost
                    # Solve: base × 0.63 × 0.85/CF = ldes_cost - tx - base × 0.37
                    rhs = ldes_cost - ccs_tx - ccs_no45q_base * 0.37
                    if rhs > 0:
                        crossover_cf = ccs_no45q_base * 0.63 * 0.85 / rhs
                        crossover_cf = round(min(1.0, max(0.0, crossover_cf)), 3)
                    else:
                        crossover_cf = 0.0  # CCS always cheaper (unlikely)

                    scenario['no_45q_costs']['ccs_vs_ldes'] = {
                        'ccs_no45q_baseload': round(ccs_no45q_base + ccs_tx, 2),
                        'ldes_cost': round(ldes_cost, 2),
                        'crossover_cf': crossover_cf,
                        'ccs_cheaper_above_cf': crossover_cf,
                        'ldes_cheaper_below_cf': crossover_cf,
                    }

                overlays += 1

    print(f"      {overlays} scenarios with no-45Q overlay added")
    return overlays


def add_neiso_gas_constraint(data):
    """Apply NEISO winter gas pipeline constraint."""
    print("\n  [4] NEISO Winter Gas Pipeline Constraint")

    iso = 'NEISO'
    if iso not in data['results']:
        print("      NEISO not in results — skipping")
        return 0

    thresholds_data = data['results'][iso].get('thresholds', {})
    adjustments = 0

    for t_str in thresholds_data:
        scenarios = thresholds_data[t_str].get('scenarios', {})
        for sk, scenario in scenarios.items():
            mix = scenario.get('resource_mix', {})

            # Compute NEISO-adjusted costs (with gas constraint)
            neiso_costs = compute_costs_for_scenario(
                iso, mix,
                scenario.get('procurement_pct', 100),
                scenario.get('battery_dispatch_pct', 0),
                scenario.get('ldes_dispatch_pct', 0),
                scenario.get('hourly_match_score', 0),
                sk,
                apply_45q=True,
                neiso_gas_adder=True,
            )

            # Also compute no-45Q + gas constraint
            neiso_no45q_costs = compute_costs_for_scenario(
                iso, mix,
                scenario.get('procurement_pct', 100),
                scenario.get('battery_dispatch_pct', 0),
                scenario.get('ldes_dispatch_pct', 0),
                scenario.get('hourly_match_score', 0),
                sk,
                apply_45q=False,
                neiso_gas_adder=True,
            )

            scenario['neiso_gas_adjusted'] = neiso_costs
            scenario['neiso_gas_no45q'] = neiso_no45q_costs
            adjustments += 1

    print(f"      {adjustments} NEISO scenarios adjusted for gas pipeline constraint")
    return adjustments


def analyze_crossover(data):
    """Analyze CCS vs LDES crossover and without-45Q cost curve impact."""
    print("\n  [5] CCS vs LDES Crossover Analysis")

    analysis = {'crossover_by_iso': {}, 'cost_curve_impact': {}}

    for iso in ISOS:
        if iso not in data['results']:
            continue

        thresholds_data = data['results'][iso].get('thresholds', {})
        med_key = 'MMM_M_M'

        crossovers = []
        curve_impact = []

        for threshold in THRESHOLDS:
            t_str = str(threshold)
            if t_str not in thresholds_data:
                continue
            scenario = thresholds_data[t_str].get('scenarios', {}).get(med_key)
            if not scenario:
                continue

            costs_with = scenario.get('costs', {})
            costs_without = scenario.get('no_45q_costs', {})
            mix = scenario.get('resource_mix', {})
            ccs_pct = mix.get('ccs_ccgt', 0)

            eff_with = costs_with.get('effective_cost', 0)
            eff_without = costs_without.get('effective_cost', 0)

            entry = {
                'threshold': threshold,
                'ccs_pct': ccs_pct,
                'cost_with_45q': eff_with,
                'cost_without_45q': eff_without,
                'cost_delta': round(eff_without - eff_with, 2),
                'cost_increase_pct': round((eff_without - eff_with) / eff_with * 100, 1) if eff_with > 0 else 0,
            }

            # Crossover data
            crossover_data = costs_without.get('ccs_vs_ldes', {})
            if crossover_data:
                entry['crossover_cf'] = crossover_data.get('crossover_cf', 0)
                entry['ccs_baseload_no45q'] = crossover_data.get('ccs_no45q_baseload', 0)
                entry['ldes_cost'] = crossover_data.get('ldes_cost', 0)
                crossovers.append(crossover_data.get('crossover_cf', 0))

            curve_impact.append(entry)

        analysis['crossover_by_iso'][iso] = {
            'avg_crossover_cf': round(sum(crossovers) / len(crossovers), 3) if crossovers else 0,
            'min_crossover_cf': round(min(crossovers), 3) if crossovers else 0,
            'max_crossover_cf': round(max(crossovers), 3) if crossovers else 0,
        }
        analysis['cost_curve_impact'][iso] = curve_impact

    # Print summary
    print("\n      CCS vs LDES Crossover (Medium scenario):")
    print(f"      {'ISO':>6}  {'Avg CF':>7}  {'Range':>15}  {'Interpretation':>30}")
    for iso in ISOS:
        cr = analysis['crossover_by_iso'].get(iso, {})
        avg_cf = cr.get('avg_crossover_cf', 0)
        min_cf = cr.get('min_crossover_cf', 0)
        max_cf = cr.get('max_crossover_cf', 0)
        if avg_cf > 0:
            interp = f"CCS viable above {avg_cf:.0%} CF"
        else:
            interp = "No CCS in mix"
        print(f"      {iso:>6}  {avg_cf:>6.1%}  [{min_cf:.0%}-{max_cf:.0%}]  {interp}")

    print("\n      Without-45Q Cost Impact (Medium, effective $/MWh):")
    print(f"      {'ISO':>6}  {'Thr':>5}  {'With 45Q':>9}  {'No 45Q':>9}  {'Delta':>7}  {'%':>6}  {'CCS%':>5}")
    for iso in ISOS:
        for entry in analysis['cost_curve_impact'].get(iso, []):
            if entry['threshold'] in [75, 90, 95, 99]:
                print(f"      {iso:>6}  {entry['threshold']:>4}%  "
                      f"${entry['cost_with_45q']:>7.2f}  ${entry['cost_without_45q']:>7.2f}  "
                      f"${entry['cost_delta']:>5.2f}  {entry['cost_increase_pct']:>5.1f}%  "
                      f"{entry['ccs_pct']:>4}%")

    # NEISO gas constraint impact
    print("\n      NEISO Gas Constraint Impact (Medium, effective $/MWh):")
    if 'NEISO' in data['results']:
        thresholds_data = data['results']['NEISO'].get('thresholds', {})
        neiso_impact = []
        for threshold in THRESHOLDS:
            t_str = str(threshold)
            scenario = thresholds_data.get(t_str, {}).get('scenarios', {}).get('MMM_M_M')
            if not scenario:
                continue
            base = scenario.get('costs', {}).get('effective_cost', 0)
            gas_adj = scenario.get('neiso_gas_adjusted', {}).get('effective_cost', 0)
            gas_no45q = scenario.get('neiso_gas_no45q', {}).get('effective_cost', 0)
            ccs = scenario.get('resource_mix', {}).get('ccs_ccgt', 0)
            neiso_impact.append({
                'threshold': threshold,
                'base': base,
                'gas_adjusted': gas_adj,
                'gas_no45q': gas_no45q,
                'ccs_pct': ccs,
            })

        print(f"      {'Thr':>5}  {'Base':>8}  {'+ Gas':>8}  {'+ Gas-No45Q':>12}  {'CCS%':>5}")
        for e in neiso_impact:
            if e['threshold'] in [75, 87.5, 90, 92.5, 95, 97.5, 99]:
                print(f"      {e['threshold']:>4}%  ${e['base']:>6.2f}  "
                      f"${e['gas_adjusted']:>6.2f}  ${e['gas_no45q']:>10.2f}  {e['ccs_pct']:>4}%")

    return analysis


def main():
    print("=" * 70)
    print("  POST-PROCESSING CORRECTIONS & OVERLAYS")
    print("=" * 70)

    data = load_results()

    # Verify we have results
    isos_present = [iso for iso in ISOS if iso in data.get('results', {})]
    print(f"  ISOs: {isos_present}")

    # Apply corrections in order
    co2_fixes = fix_co2_monotonicity(data)
    offset_fixes = fix_45q_offset(data)
    overlay_count = add_no45q_overlay(data)
    neiso_count = add_neiso_gas_constraint(data)
    analysis = analyze_crossover(data)

    # Add metadata
    data['postprocessing'] = {
        'applied': True,
        'corrections': {
            'co2_monotonicity_fixes': co2_fixes,
            '45q_offset_corrections': offset_fixes,
            'no_45q_overlays': overlay_count,
            'neiso_gas_adjustments': neiso_count,
        },
        'parameters': {
            '45q_offset_original': FOURTY_FIVE_Q_OFFSET_ORIGINAL,
            '45q_offset_corrected': FOURTY_FIVE_Q_OFFSET_CORRECTED,
            'neiso_ccs_gas_adder': NEISO_CCS_GAS_ADDER,
            'neiso_wholesale_adder': NEISO_WHOLESALE_ADDER,
            'ccs_capital_share': CCS_CAPITAL_SHARE,
            'ccs_reference_cf': CCS_REFERENCE_CF,
        },
    }

    # Save corrected results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f"\n  Corrected results → {RESULTS_PATH} "
          f"({os.path.getsize(RESULTS_PATH) / 1024:.0f} KB)")

    # Save analysis
    os.makedirs(os.path.dirname(ANALYSIS_PATH), exist_ok=True)
    with open(ANALYSIS_PATH, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"  Analysis → {ANALYSIS_PATH}")

    print("\n" + "=" * 70)
    print("  POST-PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
