#!/usr/bin/env python3
"""
Statistical Post-Processing + Path-Constrained MAC (PCHIP + Isotonic)
=====================================================================
Reads existing optimizer results and computes:

1. Monotonic envelope MAC (isotonic regression on average MAC per ISO)
2. MAC uncertainty fan (P10/P25/P50/P75/P90, isotonic-smoothed)
3. Stepwise marginal MAC (PCHIP spline derivative + isotonic regression)
4. ANOVA sensitivity decomposition (which toggles drive MAC variance)
5. Path-constrained reference MAC (monotonic resource deployment)

Outputs:
  - dashboard/js/mac-stats-data.js              (JavaScript constants for dashboard)
  - data/step5-post-processing/mac_stats.json    (full JSON for programmatic use)

Performance: Uses vectorized pandas/numpy operations throughout.
Data is loaded as DataFrames and never converted to nested dicts.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import isotonic_regression as _scipy_isotonic
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))
from parquet_io import (find_input_dir, find_parquet, find_dg_parquets,
                        DEFAULT_INPUT_DIRS, RESOURCE_TYPES, REGIONAL_DEMAND_MWH)
from step3_cost_optimization import OUTPUT_THRESHOLDS as THRESHOLDS

RESULTS_PATH = os.path.join(BASE_DIR, 'dashboard', 'overprocure_results.json')
JS_OUTPUT_PATH = os.path.join(BASE_DIR, 'dashboard', 'js', 'mac-stats-data.js')
STEP5_DIR = os.path.join(BASE_DIR, 'data', 'step5-post-processing')
JSON_OUTPUT_PATH = os.path.join(STEP5_DIR, 'mac_stats.json')

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
THRESHOLD_STRS = [str(t) if t != int(t) else str(int(t)) for t in THRESHOLDS]

# Toggle factor names for ANOVA (6 paired dimensions — Battery/LDES split from Storage)
TOGGLE_NAMES = ['Renewable Gen', 'Nuclear', 'Battery Cost', 'LDES Cost', 'Fossil Fuel', 'Transmission']
TOGGLE_COLS = ['toggle_ren', 'toggle_nuc', 'toggle_batt', 'toggle_ldes', 'toggle_fuel', 'toggle_tx']

# Medium scenario keys (any of these is acceptable)
MEDIUM_KEYS_SET = frozenset({
    'MMMM_M_M_M1_X', 'MMMM_M_M_M1_M',
    'MMM_M_M_M1_M', 'MMM_M_M_M1_X', 'MMM_M_M',
})

# (Wholesale prices removed — no longer used in MAC calculation.
#  MAC uses pure LCOE of new-build resources / CO₂ displaced.)

# ── Dispatch-model-based CO₂ baseline ──
# Import canonical grid mix shares and fossil caps from dispatch_utils
try:
    from dispatch_utils import (
        GRID_MIX_SHARES, BASE_DEMAND_TWH, COAL_CAP_TWH, OIL_CAP_TWH,
        COAL_OIL_RETIREMENT_THRESHOLD,
    )
    HAS_DISPATCH_UTILS = True
except ImportError:
    HAS_DISPATCH_UTILS = False
    # Fallback constants
    GRID_MIX_SHARES = {
        'CAISO': {'clean_firm': 7.9, 'solar': 22.3, 'wind': 8.8, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 9.5},
        'ERCOT': {'clean_firm': 8.6, 'solar': 13.8, 'wind': 23.6, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 0.1},
        'PJM':   {'clean_firm': 32.1, 'solar': 2.9, 'wind': 3.8, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 1.8},
        'NYISO': {'clean_firm': 18.4, 'solar': 0.0, 'wind': 4.7, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 15.9},
        'NEISO': {'clean_firm': 23.8, 'solar': 1.4, 'wind': 3.9, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 4.4},
        'MISO':  {'clean_firm': 13.1, 'solar': 2.1, 'wind': 14.5, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 1.6},
        'SPP':   {'clean_firm': 5.2, 'solar': 0.4, 'wind': 37.1, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 4.3},
    }
    BASE_DEMAND_TWH = {
        'CAISO': 224.039, 'ERCOT': 488.02, 'PJM': 843.331, 'NYISO': 151.599,
        'NEISO': 115.336, 'MISO': 660.0, 'SPP': 296.0,
    }
    COAL_CAP_TWH = {
        'CAISO': 0.00, 'ERCOT': 67.58, 'PJM': 139.09, 'NYISO': 0.00,
        'NEISO': 0.31, 'MISO': 125.0, 'SPP': 42.0,
    }
    OIL_CAP_TWH = {
        'CAISO': 0.60, 'ERCOT': 0.00, 'PJM': 4.59, 'NYISO': 0.15,
        'NEISO': 1.29, 'MISO': 0.50, 'SPP': 0.20,
    }
    COAL_OIL_RETIREMENT_THRESHOLD = 70.0

# Load emission rates for CO₂ computation
_LB_PER_TON = 2204.623
_EGRID_PATH = os.path.join(BASE_DIR, 'data', 'egrid_emission_rates.json')
with open(_EGRID_PATH) as _f:
    _EGRID = json.load(_f)

# Per-ISO fuel emission rates (tCO₂/MWh)
FUEL_RATES = {}
for iso in ISOS:
    rates = _EGRID.get(iso, {})
    FUEL_RATES[iso] = {
        'coal': rates.get('coal_co2_lb_per_mwh', 0.0) / _LB_PER_TON,
        'oil': rates.get('oil_co2_lb_per_mwh', 0.0) / _LB_PER_TON,
        'gas': rates.get('gas_co2_lb_per_mwh', 0.0) / _LB_PER_TON,
    }

# Existing clean percentage per ISO (2025 baseline)
EXISTING_CLEAN_PCT = {iso: sum(GRID_MIX_SHARES.get(iso, {}).values()) for iso in ISOS}


def compute_total_fossil_emissions_mt(iso, clean_pct, demand_twh=None):
    """Compute total fossil CO₂ emissions (Mt) at a given clean energy level.

    Uses merit-order dispatch: coal remains first, then oil, then gas.
    As clean_pct increases, fossil shrinks and coal/oil retire at 70% threshold.

    Returns emissions in Mt (= TWh × tCO₂/MWh, since 1 TWh = 1e6 MWh).
    """
    if demand_twh is None:
        demand_twh = BASE_DEMAND_TWH.get(iso, 0)

    fossil_pct = max(0, (100.0 - clean_pct)) / 100.0
    fossil_twh = demand_twh * fossil_pct
    if fossil_twh <= 0.01:
        return 0.0

    coal_cap = COAL_CAP_TWH.get(iso, 0)
    oil_cap = OIL_CAP_TWH.get(iso, 0)
    rates = FUEL_RATES.get(iso, {'coal': 0, 'oil': 0, 'gas': 0})

    if clean_pct >= COAL_OIL_RETIREMENT_THRESHOLD:
        # All coal and oil retired — only gas remains
        return fossil_twh * rates['gas']
    else:
        # Fossil fleet: coal fills first, then oil, then gas
        coal_twh = min(coal_cap, fossil_twh)
        remaining = fossil_twh - coal_twh
        oil_twh = min(oil_cap, remaining)
        gas_twh = max(0, remaining - oil_twh)
        return (coal_twh * rates['coal'] + oil_twh * rates['oil']
                + gas_twh * rates['gas'])


# Precompute baseline emissions (existing clean only at 2025 demand) per ISO
BASELINE_EMISSIONS_MT = {}
for _iso in ISOS:
    BASELINE_EMISSIONS_MT[_iso] = compute_total_fossil_emissions_mt(
        _iso, EXISTING_CLEAN_PCT[_iso])

# Precompute scenario emissions for all (ISO, threshold) at 2025 demand
SCENARIO_EMISSIONS_MT = {}
for _iso in ISOS:
    for _t in THRESHOLDS:
        SCENARIO_EMISSIONS_MT[(_iso, _t)] = compute_total_fossil_emissions_mt(
            _iso, _t)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — Direct DataFrame (no dict conversion)
# ══════════════════════════════════════════════════════════════════════════════

def load_combined_df(input_dir, isos):
    """Load all ISO parquets into a single DataFrame.

    Skips the dict conversion entirely — all computation works on the DataFrame.
    """
    frames = []
    for iso in isos:
        parquet_path = find_parquet(input_dir, iso)
        if parquet_path is None:
            continue
        df = pd.read_parquet(parquet_path)
        if 'iso' not in df.columns:
            df['iso'] = iso
        sz = os.path.getsize(parquet_path) / 1024
        print(f"  Loaded {iso}: {len(df):,} rows ({sz:.0f} KB)")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    df = pd.concat(frames, ignore_index=True)

    # Ensure annual_demand_mwh exists (fallback to regional constants)
    if 'annual_demand_mwh' not in df.columns:
        df['annual_demand_mwh'] = df['iso'].map(REGIONAL_DEMAND_MWH).fillna(0)
    else:
        # Fill NaN values with regional constants
        mask = df['annual_demand_mwh'].isna() | (df['annual_demand_mwh'] == 0)
        df.loc[mask, 'annual_demand_mwh'] = df.loc[mask, 'iso'].map(REGIONAL_DEMAND_MWH)

    # Ensure CO2 column exists
    if 'co2_total_co2_abated_tons' not in df.columns:
        df['co2_total_co2_abated_tons'] = 0.0

    return df


def add_mac_column(df):
    """Add MAC column: cost_of_new_resources / co2_reduced_by_new_resources.

    Cost numerator: LCOE of NEW clean resources only. Step 3 already prices
    existing resources at $0 (sunk fleet), so cost_total_cost contains only
    new-build LCOE + transmission. Gas backup (resource adequacy) is subtracted
    because it's system reliability, not abatement investment.

    NO wholesale offset. Wholesale electricity prices, fuel costs, and system
    costs play no role in MAC. MAC = pure deployment cost / CO₂ displaced.

    CO₂ denominator: Baseline emissions (existing clean only at 2025 TWh) minus
    scenario emissions (at threshold level). Uses dispatch-stack merit-order
    retirement model (coal → oil → gas).

    MAC = new_resource_cost_$M / co2_reduced_Mt = $/tCO₂
    """
    # ── 1. Cost of NEW resources only (per MWh of demand) ──
    # Step 3 already excludes existing resources (they're at $0 LCOE).
    # cost_total_cost = LCOE of new-build clean + storage + transmission.
    cost_total = (df['cost_total_cost'] if 'cost_total_cost' in df.columns
                  else df['cost_effective_cost'])
    new_cost = cost_total.copy()

    # Subtract gas backup cost (system reliability, not abatement)
    for gas_col in ['ra_gas_backup_cost_per_mwh', 'gas_gas_cost_per_mwh']:
        if gas_col in df.columns:
            new_cost = new_cost - df[gas_col].fillna(0)
            break
    new_cost = new_cost.clip(lower=0)

    # Store new-resource-only cost for use by other functions (stepwise, envelope)
    df['mac_new_cost'] = new_cost

    # ── 3. CO₂ reduced by new resources (tons) ──
    # Baseline: fossil emissions when only existing clean runs (2025 TWh, at
    # demand level from row). Scenario: emissions at the row's threshold.
    # CO₂ reduced = baseline - scenario.
    baseline_mt = df['iso'].map(BASELINE_EMISSIONS_MT).fillna(0)

    # Vectorized scenario emissions lookup (precomputed for all ISO × threshold)
    scenario_mt = pd.Series(
        [SCENARIO_EMISSIONS_MT.get((iso, t), 0)
         for iso, t in zip(df['iso'], df['threshold'])],
        index=df.index, dtype=float,
    )
    co2_reduced_mt = (baseline_mt - scenario_mt).clip(lower=0)
    co2_reduced_tons = co2_reduced_mt * 1e6  # Mt → tons

    # ── 4. MAC = new_resource_cost × demand_mwh / co2_reduced_tons ──
    valid = (co2_reduced_tons > 0) & new_cost.notna() & (new_cost > 0)
    df['mac'] = np.where(
        valid,
        (new_cost * df['annual_demand_mwh']) / co2_reduced_tons,
        np.nan,
    )
    return df


def parse_toggle_levels(df):
    """Parse scenario keys into 6 toggle-level columns (vectorized).

    Handles 9-dim (RFBL_F_TX_...) and 8-dim/5-dim (RFS_F_TX_...) formats.
    L=0, M=1, H=2, N=0.
    """
    level_map = {'L': 0, 'M': 1, 'H': 2, 'N': 0}

    parts = df['scenario'].str.split('_')
    gen = parts.str[0]   # e.g., "MMMM" or "MMM"
    fuel = parts.str[1]  # e.g., "M"
    tx = parts.str[2]    # e.g., "M"

    has_4 = gen.str.len() >= 4

    df['toggle_ren'] = gen.str[0].map(level_map).fillna(1).astype(np.int8)
    df['toggle_nuc'] = gen.str[1].map(level_map).fillna(1).astype(np.int8)

    # gen[2] is battery for 4-char, or storage for 3-char
    char2 = gen.str[2].map(level_map).fillna(1)
    # gen[3] only exists for 4-char keys (NaN for 3-char)
    char3 = gen.str[3].map(level_map).fillna(1)

    df['toggle_batt'] = char2.astype(np.int8)
    # For 4-char: LDES = gen[3]; for 3-char: LDES = same as battery (storage)
    df['toggle_ldes'] = np.where(has_4, char3, char2).astype(np.int8)
    df['toggle_fuel'] = fuel.map(level_map).fillna(1).astype(np.int8)
    df['toggle_tx'] = tx.map(level_map).fillna(1).astype(np.int8)

    return df


def fmt_threshold(t):
    """50.0 → '50', 87.5 → '87.5'"""
    return str(t) if t != int(t) else str(int(t))


# Pre-computed threshold string lookup (avoids repeated fmt_threshold calls in loops)
THRESHOLD_FMT = {t: fmt_threshold(t) for t in THRESHOLDS}


def medium_key(iso):
    """Return the all-Medium 9-dim scenario key for an ISO."""
    geo = 'M' if iso == 'CAISO' else 'X'
    return f'MMMM_M_M_M1_{geo}'


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE FUNCTIONS — All vectorized on DataFrames
# ══════════════════════════════════════════════════════════════════════════════

def compute_fan_and_anova(df):
    """Compute fan chart percentiles AND ANOVA decomposition in a single pass.

    Consolidates what were previously two separate groupby iterations over
    (iso, threshold) into one loop, halving the groupby overhead.

    Fan chart: P10/P25/P50/P75/P90 MAC percentiles per ISO/threshold.
    ANOVA: Eta-squared decomposition of MAC variance by toggle group.
    """
    fan_data = {}
    anova_results = {}
    pcts = [10, 25, 50, 75, 90]
    pct_names = ['p10', 'p25', 'p50', 'p75', 'p90']
    valid = df[df['mac'].notna()]

    # Initialize output structures
    for iso in ISOS:
        fan_data[iso] = {p: [] for p in pct_names}
        anova_results[iso] = {}

    # Track per-iso toggle contributions across thresholds for ANOVA averaging
    toggle_contributions = {iso: {name: [] for name in TOGGLE_NAMES} for iso in ISOS}

    # Build a lookup of threshold index for ordered insertion into fan lists
    threshold_index = {t: i for i, t in enumerate(THRESHOLDS)}

    # Pre-fill fan_data with None lists so we can assign by index
    for iso in ISOS:
        for p in pct_names:
            fan_data[iso][p] = [None] * len(THRESHOLDS)

    # Single pass: group by (iso, threshold), compute both fan + ANOVA per group
    for (iso, t), grp in valid.groupby(['iso', 'threshold']):
        if iso not in fan_data or t not in threshold_index:
            continue
        t_idx = threshold_index[t]
        t_macs = grp['mac'].values

        # ── Fan chart percentiles ──
        if len(t_macs) > 0:
            vals = np.percentile(t_macs, pcts)
            for name, val in zip(pct_names, vals):
                fan_data[iso][name][t_idx] = round(float(val), 1)

        # ── ANOVA: eta-squared per toggle ──
        if len(t_macs) >= 10:
            total_var = np.var(t_macs)
            if total_var >= 1e-6:
                ss_total = total_var * len(t_macs)
                grand_mean = np.mean(t_macs)

                for toggle_name, toggle_col in zip(TOGGLE_NAMES, TOGGLE_COLS):
                    group_stats = grp.groupby(toggle_col)['mac'].agg(['mean', 'count'])
                    ss_between = (group_stats['count'] * (group_stats['mean'] - grand_mean) ** 2).sum()
                    eta_squared = ss_between / ss_total
                    toggle_contributions[iso][toggle_name].append(eta_squared)

    # NOTE: No isotonic regression on average MAC fan chart percentiles.
    # Average MAC is naturally U-shaped (high at low thresholds, drops to
    # minimum around 85-95%, rises slightly at last-mile). Isotonic regression
    # (enforcing non-decreasing) collapses U-shaped data to a flat constant.
    # Raw percentile values are the correct representation of average MAC.

    # Average ANOVA contributions across thresholds
    for iso in ISOS:
        for toggle_name in TOGGLE_NAMES:
            vals = toggle_contributions[iso][toggle_name]
            anova_results[iso][toggle_name] = round(float(np.mean(vals)), 3) if vals else 0.0

    return fan_data, anova_results


def compute_stepwise_fan(df):
    """Compute P10/P50/P90 of stepwise marginal MAC between adjacent thresholds.

    Step MAC = delta_new_cost / delta_co2_reduced per scenario.
    Uses new-resource-only cost (mac_new_cost column) and dispatch-model CO₂
    (delta between baseline-scenario emissions at adjacent thresholds).
    """
    fan_data = {}

    for iso in ISOS:
        iso_df = df[df['iso'] == iso]
        fan_data[iso] = {'p10': [None], 'p50': [None], 'p90': [None]}

        # Precompute CO₂ reduced at each threshold for this ISO
        baseline_mt = BASELINE_EMISSIONS_MT.get(iso, 0)

        for i in range(1, len(THRESHOLDS)):
            t_prev, t_curr = THRESHOLDS[i - 1], THRESHOLDS[i]

            # Use mac_new_cost (new-resource-only) for stepwise MAC
            prev = iso_df.loc[iso_df['threshold'] == t_prev,
                              ['scenario', 'mac_new_cost',
                               'annual_demand_mwh']].set_index('scenario')
            curr = iso_df.loc[iso_df['threshold'] == t_curr,
                              ['scenario', 'mac_new_cost']].set_index('scenario')

            merged = prev.join(curr, rsuffix='_next', how='inner')
            if merged.empty:
                for p in ['p10', 'p50', 'p90']:
                    fan_data[iso][p].append(None)
                continue

            delta_cost = ((merged['mac_new_cost_next'] - merged['mac_new_cost'])
                          * merged['annual_demand_mwh'])

            # CO₂ delta from dispatch model (same for all scenarios at same threshold)
            co2_prev = (baseline_mt - SCENARIO_EMISSIONS_MT.get((iso, t_prev), 0)) * 1e6
            co2_curr = (baseline_mt - SCENARIO_EMISSIONS_MT.get((iso, t_curr), 0)) * 1e6
            delta_co2 = co2_curr - co2_prev  # scalar, same for all rows

            if delta_co2 > 0:
                valid = delta_cost >= 0
                step_macs = (delta_cost[valid] / delta_co2).values
            else:
                step_macs = np.array([])

            if len(step_macs) > 0:
                p10, p50, p90 = np.percentile(step_macs, [10, 50, 90])
                fan_data[iso]['p10'].append(round(float(p10), 1))
                fan_data[iso]['p50'].append(round(float(p50), 1))
                fan_data[iso]['p90'].append(round(float(p90), 1))
            else:
                for p in ['p10', 'p50', 'p90']:
                    fan_data[iso][p].append(None)

    # Apply isotonic regression to stepwise fan percentile curves
    if HAS_SCIPY:
        for iso in ISOS:
            for p in ['p10', 'p50', 'p90']:
                vals = fan_data[iso][p]
                valid = [(i, v) for i, v in enumerate(vals) if v is not None]
                if len(valid) >= 3:
                    idx, arr = zip(*valid)
                    result = _scipy_isotonic(np.array(arr, dtype=np.float64))
                    smoothed = result.x if hasattr(result, 'x') else result
                    for j, i in enumerate(idx):
                        fan_data[iso][p][i] = round(float(smoothed[j]), 1)

    return fan_data


def compute_envelope_and_path(df):
    """Compute average MAC, marginal MAC (PCHIP), AND path-constrained MAC.

    Consolidates compute_monotonic_envelope() and compute_path_constrained_mac()
    which both filter to medium scenario and iterate over the same ISO/threshold
    pairs. One medium-scenario filter, one ISO loop, one threshold loop.

    Envelope: raw average MAC (no isotonic — U-shape is scientifically correct).
    Stepwise envelope: PCHIP derivative of cost vs CO₂ curve + isotonic regression
        for smooth monotonic marginal MAC ($/ton CO₂).
    Path-constrained: enforces non-decreasing absolute resource deployment.
    """
    envelope = {}
    path_mac = {}
    med_df = df[df['scenario'].isin(MEDIUM_KEYS_SET)]

    for iso in ISOS:
        iso_med = med_df[med_df['iso'] == iso].sort_values('threshold')
        demand_mwh = iso_med['annual_demand_mwh'].iloc[0] if len(iso_med) > 0 else 1

        # Build threshold lookup once for both computations (vectorized, no iterrows)
        results_by_t = {row['threshold']: row for row in iso_med.to_dict('records')}
        # Convert records back to Series-like access via dict .get()
        results_by_t = {t: iso_med[iso_med['threshold'] == t].iloc[0]
                        for t in iso_med['threshold'].unique()}

        # Baseline emissions (Mt) — fossil emissions with only existing clean
        baseline_mt = BASELINE_EMISSIONS_MT.get(iso, 0)

        # ── Envelope + path state ──
        raw_macs = []
        costs_at_t = []
        co2_at_t = []

        prev_abs = {r: 0 for r in RESOURCE_TYPES}
        prev_batt = 0
        prev_ldes = 0
        prev_h2 = 0
        prev_cost = 0
        prev_co2 = 0

        path_macs = []
        path_mixes = []
        path_costs = []

        has_any_row = len(iso_med) > 0

        # Single pass over thresholds: compute both envelope inputs and path-constrained
        for t in THRESHOLDS:
            row = results_by_t.get(t)
            if row is None:
                raw_macs.append(None)
                costs_at_t.append(None)
                co2_at_t.append(None)
                path_macs.append(None)
                path_mixes.append(None)
                path_costs.append(None)
                continue

            # ── New-resource-only cost ──
            new_cost = float(row['mac_new_cost']) if 'mac_new_cost' in row.index else 0.0

            # CO₂ reduced by new resources (dispatch-model-based)
            scenario_mt = SCENARIO_EMISSIONS_MT.get((iso, t), 0)
            co2_reduced_mt = max(0, baseline_mt - scenario_mt)
            co2_reduced_tons = co2_reduced_mt * 1e6

            # ── Envelope data collection ──
            costs_at_t.append(new_cost)
            co2_at_t.append(co2_reduced_mt)

            # MAC = new_cost × demand / co2_reduced_tons
            if co2_reduced_tons > 0 and new_cost > 0:
                raw_macs.append(round((new_cost * demand_mwh) / co2_reduced_tons, 1))
            else:
                raw_macs.append(None)

            # ── Path-constrained computation ──
            if has_any_row:
                # Build resource mix — handle missing columns gracefully
                # (e.g., mix_offshore_wind may not exist in older step3 parquets)
                mix = {}
                for r in RESOURCE_TYPES:
                    col = f'mix_{r}'
                    if col in row.index and not pd.isna(row[col]):
                        mix[r] = int(row[col])
                    else:
                        mix[r] = 0
                # v5.0: procurement_pct is always 100 (baked into resource percentages)
                batt = round(float(row['battery_dispatch_pct']), 4)
                ldes = round(float(row['ldes_dispatch_pct']), 4)
                h2 = round(float(row.get('h2_dispatch_pct', 0)), 4) if 'h2_dispatch_pct' in row.index else 0.0

                # Compute absolute deployment for this threshold's optimal mix
                # procurement is always 100% in v5.0
                curr_abs = {r: mix[r] for r in RESOURCE_TYPES}

                # Enforce monotonicity: each resource's absolute deployment >= previous
                constrained_abs = {r: max(curr_abs[r], prev_abs[r]) for r in RESOURCE_TYPES}
                constrained_batt = max(batt, prev_batt)
                constrained_ldes = max(ldes, prev_ldes)
                constrained_h2 = max(h2, prev_h2)

                # Reconstruct mix percentages from constrained absolute values
                total_abs = sum(constrained_abs.values())
                if total_abs > 0:
                    constrained_mix = {r: round(constrained_abs[r] / total_abs * 100, 1)
                                       for r in constrained_abs}
                else:
                    constrained_mix = mix

                # Cost is at least as high as previous constrained cost
                constrained_total = max(new_cost, prev_cost)

                # Average MAC (CO₂ reduced by new resources)
                if co2_reduced_tons > 0 and constrained_total > 0:
                    avg_mac = round((constrained_total * demand_mwh) / co2_reduced_tons, 1)
                else:
                    avg_mac = None

                path_macs.append(avg_mac)
                path_mixes.append(constrained_mix)
                path_costs.append(round(constrained_total, 2))

                # Update state for next threshold
                prev_abs = constrained_abs
                prev_batt = constrained_batt
                prev_ldes = constrained_ldes
                prev_h2 = constrained_h2
                prev_cost = constrained_total
                prev_co2 = co2_reduced_mt
            else:
                path_macs.append(None)
                path_mixes.append(None)
                path_costs.append(None)

        # ── Post-loop: build envelope from collected data ──
        # PCHIP spline + isotonic regression for smooth monotonic MAC
        # (matches methodology in step5_compute_optimal_targets.py)
        env_macs = [None] * len(THRESHOLDS)
        step_env = [None] * len(THRESHOLDS)

        if HAS_SCIPY:
            # --- Average MAC: PCHIP smoothing (NO isotonic — U-shape is correct) ---
            # Average MAC (total_cost / total_CO₂_reduced) naturally decreases
            # with scale then rises slightly at high thresholds. Isotonic
            # regression on a U-shaped curve collapses it to a flat constant.
            # Use raw values directly — they are the scientifically correct
            # average MAC at each threshold.
            valid_avg = [(i, v) for i, v in enumerate(raw_macs) if v is not None]
            if len(valid_avg) >= 3:
                idx_avg, vals_avg = zip(*valid_avg)
                # Use raw values directly — no isotonic
                for j, i in enumerate(idx_avg):
                    env_macs[i] = round(float(vals_avg[j]), 1)
            else:
                env_macs = list(raw_macs)

            # --- Stepwise marginal MAC: PCHIP derivative + isotonic ---
            # FIX: Convert CO₂ from Mt to tons so PCHIP derivative is $/ton
            # (previously was $/Mt = 1e6× too large, all values hit 9999 cap)
            valid_pts = [(i, co2_at_t[i], costs_at_t[i])
                         for i in range(len(THRESHOLDS))
                         if costs_at_t[i] is not None and co2_at_t[i] and co2_at_t[i] > 0]
            if len(valid_pts) >= 3:
                v_idx, v_co2, v_cost = zip(*valid_pts)
                co2_arr = np.array(v_co2, dtype=np.float64) * 1e6  # Mt → tons
                cost_total = np.array(v_cost, dtype=np.float64) * demand_mwh

                # Ensure CO2 strictly increasing for PCHIP
                mask = np.ones(len(co2_arr), dtype=bool)
                for j in range(1, len(co2_arr)):
                    if co2_arr[j] <= co2_arr[j - 1]:
                        mask[j] = False
                co2_mono = co2_arr[mask]
                cost_mono = cost_total[mask]
                idx_mono = [v_idx[j] for j in range(len(v_idx)) if mask[j]]

                if len(co2_mono) >= 3:
                    pchip = PchipInterpolator(co2_mono, cost_mono)
                    raw_deriv = pchip.derivative()(co2_mono)  # now $/ton
                    raw_deriv = np.maximum(raw_deriv, 0.01)
                    iso_step = _scipy_isotonic(raw_deriv)
                    smoothed_step = iso_step.x if hasattr(iso_step, 'x') else iso_step
                    for j, i in enumerate(idx_mono):
                        step_env[i] = round(min(float(smoothed_step[j]), 9999), 1)
        else:
            # Fallback: running max (no scipy) — use raw for envelope
            for i, mac in enumerate(raw_macs):
                if mac is not None:
                    env_macs[i] = round(mac, 1)
            step_running_max = 0
            for i in range(1, len(THRESHOLDS)):
                if costs_at_t[i] is not None and costs_at_t[i - 1] is not None:
                    delta_cost = (costs_at_t[i] - costs_at_t[i - 1]) * demand_mwh
                    # Convert CO2 delta from Mt to tons
                    delta_co2 = ((co2_at_t[i] or 0) - (co2_at_t[i - 1] or 0)) * 1e6
                    if delta_co2 > 0 and delta_cost >= 0:
                        step_mac = delta_cost / delta_co2
                        step_running_max = max(step_running_max, step_mac)
                        step_env[i] = round(step_running_max, 1)
                    else:
                        step_env[i] = round(step_running_max, 1)

        envelope[iso] = {
            'raw': raw_macs,
            'envelope': env_macs,
            'stepwise_envelope': step_env,
        }

        if not has_any_row:
            path_mac[iso] = {
                'mac': [None] * len(THRESHOLDS),
                'mixes': [None] * len(THRESHOLDS),
                'costs': [None] * len(THRESHOLDS),
            }
        else:
            path_mac[iso] = {
                'mac': path_macs,
                'mixes': path_mixes,
                'costs': path_costs,
            }
    return envelope, path_mac


def compute_monotonic_envelope(df):
    """Compute monotonic envelope MAC — running max to enforce non-decreasing MAC.

    Delegates to compute_envelope_and_path() which consolidates both envelope
    and path-constrained computation in a single pass over medium-scenario data.
    """
    envelope, _ = compute_envelope_and_path(df)
    return envelope

def compute_path_constrained_mac(df):
    """Compute path-constrained reference MAC from medium scenario.

    Delegates to compute_envelope_and_path() which consolidates both envelope
    and path-constrained computation in a single pass over medium-scenario data.
    """
    _, path_mac = compute_envelope_and_path(df)
    return path_mac


def compute_anova(df):
    """ANOVA-style sensitivity decomposition (standalone fallback).

    Prefer compute_fan_and_anova() which consolidates this with fan chart
    computation in a single groupby pass.
    """
    _, anova_results = compute_fan_and_anova(df)
    return anova_results


def compute_crossover_analysis(fan_data, envelope_data):
    """Compute threshold where MAC crosses key benchmarks."""
    benchmarks = {
        'scc_epa_190': 190,
        'scc_rennert_185': 185,
        'dac_low_400': 400,
        'dac_mid_600': 600,
        'carbon_credits_15': 15,
        'eu_ets_88': 88,
    }

    crossovers = {}
    for iso in ISOS:
        crossovers[iso] = {}
        env = envelope_data.get(iso, {}).get('envelope', [])
        p50 = fan_data.get(iso, {}).get('p50', [])

        for bm_name, bm_cost in benchmarks.items():
            env_cross = '>99'
            for i, mac in enumerate(env):
                if mac is not None and mac > bm_cost:
                    env_cross = THRESHOLDS[i]
                    break

            p50_cross = '>99'
            for i, mac in enumerate(p50):
                if mac is not None and mac > bm_cost:
                    p50_cross = THRESHOLDS[i]
                    break

            crossovers[iso][bm_name] = {
                'envelope': env_cross,
                'median': p50_cross,
            }

    return crossovers


def compute_dg_mac(input_dir, isos):
    """Compute demand growth MAC directly from parquet DataFrames.

    Uses new-resource-only cost and dispatch-model-based CO₂:
    - Cost: cost_total_cost - gas_backup (per MWh of demand). No wholesale offset.
    - CO₂: baseline emissions (existing clean at 2025 TWh, diluted by demand
      growth) minus scenario emissions (at threshold level, grown demand).
    """
    dg_mac = {}

    for iso in isos:
        files = find_dg_parquets(input_dir, iso)
        if not files:
            for alt_dir in DEFAULT_INPUT_DIRS:
                files = find_dg_parquets(alt_dir, iso, 'step3_dg_')
                if files:
                    break
        if not files:
            continue

        frames = [pd.read_parquet(f) for f in files]
        dg_df = pd.concat(frames, ignore_index=True)

        demand_col = 'annual_demand_mwh'
        if demand_col not in dg_df.columns or 'year' not in dg_df.columns:
            continue
        if 'growth_level' not in dg_df.columns:
            continue

        # ── Compute new-resource-only cost ──
        # Step 3 already prices existing resources at $0 (sunk fleet).
        # cost_total_cost = LCOE of new-build clean only. No wholesale offset.
        cost_total = (dg_df['cost_total_cost'] if 'cost_total_cost' in dg_df.columns
                      else dg_df['cost_effective_cost'])
        new_cost = cost_total.copy()
        for gas_col in ['ra_gas_backup_cost_per_mwh', 'gas_gas_cost_per_mwh']:
            if gas_col in dg_df.columns:
                new_cost = new_cost - dg_df[gas_col].fillna(0)
                break
        new_cost = new_cost.clip(lower=0)

        # ── CO₂ reduced: baseline (existing clean diluted by growth) vs scenario ──
        existing_clean_pct = EXISTING_CLEAN_PCT.get(iso, 0)
        base_demand = BASE_DEMAND_TWH.get(iso, 0)

        # Growth factor per row: demand_mwh / (base_demand * 1e6)
        gf_arr = dg_df[demand_col].astype(float) / (base_demand * 1e6)
        gf_arr = gf_arr.clip(lower=1.0)

        # Existing clean as % of grown demand (TWh stays at 2025 level)
        existing_pct_diluted = existing_clean_pct / gf_arr

        # Vectorize: baseline and scenario emissions per row
        demand_twh_arr = dg_df[demand_col].astype(float) / 1e6
        thresholds = dg_df['threshold'].astype(float).values

        baseline_emissions = np.array([
            compute_total_fossil_emissions_mt(iso, ep, dt)
            for ep, dt in zip(existing_pct_diluted, demand_twh_arr)
        ])
        scenario_emissions = np.array([
            compute_total_fossil_emissions_mt(iso, t, dt)
            for t, dt in zip(thresholds, demand_twh_arr)
        ])
        co2_reduced_mt = np.maximum(0, baseline_emissions - scenario_emissions)
        co2_reduced_tons = co2_reduced_mt * 1e6

        # MAC
        valid = (co2_reduced_tons > 0) & new_cost.notna() & (new_cost > 0) & (dg_df[demand_col] > 0)
        dg_valid = dg_df[valid].copy()
        dg_valid['mac'] = (new_cost[valid].values * dg_valid[demand_col].values) / co2_reduced_tons[valid]

        if dg_valid.empty:
            continue

        iso_mac = {}
        for (threshold, year, g_level), grp in dg_valid.groupby(
                ['threshold', 'year', 'growth_level']):
            t_str = THRESHOLD_FMT.get(threshold, fmt_threshold(threshold))
            arr = grp['mac'].values
            demand_sample = float(grp[demand_col].iloc[0])
            gf_sample = float(grp['growth_factor'].iloc[0]) if 'growth_factor' in grp.columns else 1.0

            if t_str not in iso_mac:
                iso_mac[t_str] = {}

            p10, p50, p90 = np.percentile(arr, [10, 50, 90])
            iso_mac[t_str][g_level] = {
                'mac_p10': round(float(p10), 1),
                'mac_p50': round(float(p50), 1),
                'mac_p90': round(float(p90), 1),
                'year': int(year),
                'growth_factor': round(float(gf_sample), 4),
                'demand_mwh': round(demand_sample, 0),
                'n_scenarios': len(arr),
            }

        if iso_mac:
            dg_mac[iso] = iso_mac
            n_combos = sum(len(v) for v in iso_mac.values())
            print(f"  {iso}: DG MAC computed for {len(iso_mac)} thresholds, {n_combos} (thr,growth) combos")

    return dg_mac


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def format_js_output(fan_data, stepwise_fan, envelope_data, path_mac, anova, crossovers):
    """Format all computed data as JavaScript constants for dashboard use."""
    lines = [
        '// ============================================================================',
        '// MAC STATISTICS — Option B: Statistical Post-Processing + Path-Constrained MAC',
        '// ============================================================================',
        '// Auto-generated by step5_compute_mac_stats.py — do not edit manually',
        f'// Generated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '// Source: 16,200 optimizer scenarios (10 thresholds × 324 cost combos × 5 ISOs)',
        '//',
        '// Methodology:',
        '//   - Fan chart: P10/P25/P50/P75/P90 of average MAC, isotonic-smoothed',
        '//   - Envelope: Isotonic regression on Medium MAC (smooth monotonic)',
        '//   - Stepwise: PCHIP spline derivative + isotonic regression (marginal MAC)',
        '//   - Path-constrained: Enforces non-decreasing resource deployment',
        '//   - ANOVA: Eta-squared decomposition of MAC variance by toggle group',
        '// ============================================================================',
        '',
    ]

    # Fan chart data
    lines.append('// --- MAC Fan Chart (P10/P25/P50/P75/P90 average MAC across 324 scenarios) ---')
    lines.append(f'const MAC_FAN_DATA = {json.dumps(fan_data, indent=4)};')
    lines.append('')

    # Stepwise fan
    lines.append('// --- Stepwise Marginal MAC Fan (P10/P50/P90 of step MAC between thresholds) ---')
    lines.append(f'const MAC_STEPWISE_FAN = {json.dumps(stepwise_fan, indent=4)};')
    lines.append('')

    # Monotonic envelope
    env_js = {}
    for iso in ISOS:
        env_js[iso] = envelope_data[iso]
    lines.append('// --- Monotonic Envelope MAC (running max of Medium, smooths rebalancing) ---')
    lines.append(f'const MAC_ENVELOPE_DATA = {json.dumps(env_js, indent=4)};')
    lines.append('')

    # Path-constrained MAC
    path_js = {}
    for iso in ISOS:
        path_js[iso] = {
            'mac': path_mac[iso]['mac'],
            'costs': path_mac[iso]['costs'],
        }
    lines.append('// --- Path-Constrained Reference MAC (monotonic resource deployment) ---')
    lines.append(f'const MAC_PATH_CONSTRAINED = {json.dumps(path_js, indent=4)};')
    lines.append('')

    # ANOVA
    lines.append('// --- ANOVA: Fraction of MAC variance explained by each toggle group ---')
    lines.append('// Values are eta-squared (0-1): the proportion of total MAC variance')
    lines.append('// attributable to each sensitivity toggle. Higher = more influential.')
    lines.append(f'const MAC_ANOVA = {json.dumps(anova, indent=4)};')
    lines.append('')

    # Crossover analysis
    lines.append('// --- Crossover Analysis: threshold where MAC exceeds benchmarks ---')
    lines.append(f'const MAC_CROSSOVERS = {json.dumps(crossovers, indent=4)};')
    lines.append('')

    return '\n'.join(lines) + '\n'


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # Print baseline emissions summary
    print("Baseline fossil emissions (existing clean only, 2025 demand):")
    for iso in ISOS:
        pct = EXISTING_CLEAN_PCT.get(iso, 0)
        emit = BASELINE_EMISSIONS_MT.get(iso, 0)
        print(f"  {iso}: existing clean={pct:.1f}%  baseline emissions={emit:.1f} Mt")

    print("\nLoading optimizer results from parquets...")
    # Prefer co2-enriched parquets (step5 co2_results), fall back to step4/step3
    co2_dir = os.path.join(STEP5_DIR, 'co2_results')
    input_dir = co2_dir if os.path.isdir(co2_dir) and any(
        f.endswith('.parquet') for f in os.listdir(co2_dir)
    ) else find_input_dir(ISOS)
    if not input_dir:
        raise FileNotFoundError("No parquet input directory found")

    df = load_combined_df(input_dir, ISOS)
    n_isos = df['iso'].nunique()
    print(f"  Total: {len(df):,} rows across {n_isos} ISOs")

    # Precompute MAC column and toggle levels (once for all computations)
    t1 = time.time()
    add_mac_column(df)
    parse_toggle_levels(df)
    mac_valid = df['mac'].notna().sum()
    print(f"  Precomputed MAC ({mac_valid:,} valid) + toggle levels in {time.time()-t1:.1f}s")

    t1 = time.time()
    print("\nComputing MAC fan chart + ANOVA (consolidated single pass)...")
    fan_data, anova = compute_fan_and_anova(df)
    print(f"  Done in {time.time()-t1:.1f}s")

    t1 = time.time()
    print("Computing stepwise marginal MAC fan...")
    stepwise_fan = compute_stepwise_fan(df)
    print(f"  Done in {time.time()-t1:.1f}s")

    t1 = time.time()
    print("Computing monotonic envelope + path-constrained MAC (consolidated pass)...")
    envelope_data, path_mac = compute_envelope_and_path(df)
    print(f"  Done in {time.time()-t1:.1f}s")

    print("Computing crossover analysis...")
    crossovers = compute_crossover_analysis(fan_data, envelope_data)

    # Compute DG MAC (threshold-year paired demand growth scenarios)
    t1 = time.time()
    print("\nComputing demand growth MAC (threshold-year paired)...")
    dg_mac = compute_dg_mac(input_dir, ISOS)
    print(f"  Done in {time.time()-t1:.1f}s")

    # Write JavaScript output
    js_content = format_js_output(fan_data, stepwise_fan, envelope_data, path_mac, anova, crossovers)
    if dg_mac:
        js_content += f'\n// --- Demand Growth MAC (threshold-year paired, P10/P50/P90) ---\n'
        js_content += f'const MAC_DEMAND_GROWTH = {json.dumps(dg_mac, indent=4)};\n'
    with open(JS_OUTPUT_PATH, 'w') as f:
        f.write(js_content)
    print(f"Wrote {JS_OUTPUT_PATH}")

    # Write JSON output to step5 results directory
    os.makedirs(STEP5_DIR, exist_ok=True)
    json_output = {
        'fan_chart': fan_data,
        'stepwise_fan': stepwise_fan,
        'envelope': envelope_data,
        'path_constrained': {iso: path_mac[iso] for iso in ISOS},
        'anova': anova,
        'crossovers': crossovers,
        'demand_growth_mac': dg_mac,
        'metadata': {
            'thresholds': THRESHOLDS,
            'isos': ISOS,
            'scenario_count': 324,
            'methodology': 'Option B: Statistical post-processing + path-constrained reference',
            'demand_growth': 'Threshold-year paired: each threshold at its SBTi target year × L/M/H',
        }
    }
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Wrote {JSON_OUTPUT_PATH}")

    # Print summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"MAC STATISTICS COMPLETE — {elapsed:.1f}s total")
    print("=" * 70)

    for iso in ISOS:
        p50 = fan_data[iso]['p50']
        p10 = fan_data[iso]['p10']
        p90 = fan_data[iso]['p90']
        env = envelope_data[iso]['envelope']
        pc = path_mac[iso]['mac']

        print(f"\n--- {iso} ---")
        print(f"  Fan P50: {p50}")
        print(f"  Fan P10: {p10}")
        print(f"  Fan P90: {p90}")
        print(f"  Envelope: {env}")
        print(f"  Path-constrained: {pc}")
        print(f"  ANOVA: {anova[iso]}")

    print(f"\n{'=' * 70}")
    print("CROSSOVER THRESHOLDS (where MAC exceeds benchmark)")
    print("=" * 70)
    for iso in ISOS:
        print(f"\n  {iso}:")
        for bm, vals in crossovers[iso].items():
            print(f"    {bm}: envelope={vals['envelope']}, median={vals['median']}")


if __name__ == '__main__':
    main()
