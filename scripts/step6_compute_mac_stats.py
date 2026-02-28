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

# Wholesale prices (duplicated from optimizer for standalone use)
WHOLESALE_PRICES = {'CAISO': 30, 'ERCOT': 27, 'PJM': 34, 'NYISO': 42, 'NEISO': 41, 'MISO': 30, 'SPP': 25}

# ── EIA/eGRID baseline: CO₂ already avoided by existing clean generation ──
# existing_clean_avoidance = (fossil_rate − actual_grid_rate) × demand / 2204.623
# New investment only gets credit for CO₂ above this pre-existing avoidance.
_LB_PER_TON = 2204.623
_EGRID_PATH = os.path.join(BASE_DIR, 'data', 'egrid_emission_rates.json')
with open(_EGRID_PATH) as _f:
    _EGRID = json.load(_f)

EXISTING_CLEAN_AVOIDANCE_TONS = {
    iso: (_EGRID[iso]['fossil_co2_lb_per_mwh'] - _EGRID[iso]['total_co2_lb_per_mwh'])
         * REGIONAL_DEMAND_MWH.get(iso, 0) / _LB_PER_TON
    for iso in ISOS
}


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
    """Add MAC column: (cost_total_cost × annual_demand_mwh) / co2_abated_by_new_capital.

    Cost numerator: full system LCOE (cost_total_cost × annual_demand) — captures
    the actual capital investment, not just the premium above wholesale.

    CO₂ denominator: co2_total_co2_abated_tons minus the CO₂ already avoided by
    existing clean generation (from EIA/eGRID actuals). New investment only gets
    credit for CO₂ displacement above the pre-existing baseline.
    """
    # Per-ISO existing clean avoidance (CO₂ not credited to new investment)
    existing_avoidance = df['iso'].map(EXISTING_CLEAN_AVOIDANCE_TONS).fillna(0)
    co2_net = (df['co2_total_co2_abated_tons'] - existing_avoidance).clip(lower=0)

    cost_col = df['cost_total_cost'] if 'cost_total_cost' in df.columns else df['cost_effective_cost']
    valid = (co2_net > 0) & cost_col.notna()
    df['mac'] = np.where(
        valid,
        (cost_col * df['annual_demand_mwh']) / co2_net,
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
        if iso not in threshold_index or t not in threshold_index:
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

    # Apply isotonic regression to fan percentile curves for smooth monotonic shape
    if HAS_SCIPY:
        for iso in ISOS:
            for p in pct_names:
                vals = fan_data[iso][p]
                valid = [(i, v) for i, v in enumerate(vals) if v is not None]
                if len(valid) >= 3:
                    idx, arr = zip(*valid)
                    result = _scipy_isotonic(np.array(arr, dtype=np.float64))
                    smoothed = result.x if hasattr(result, 'x') else result
                    for j, i in enumerate(idx):
                        fan_data[iso][p][i] = round(float(smoothed[j]), 1)

    # Average ANOVA contributions across thresholds
    for iso in ISOS:
        for toggle_name in TOGGLE_NAMES:
            vals = toggle_contributions[iso][toggle_name]
            anova_results[iso][toggle_name] = round(float(np.mean(vals)), 3) if vals else 0.0

    return fan_data, anova_results


def compute_stepwise_fan(df):
    """Compute P10/P50/P90 of stepwise marginal MAC between adjacent thresholds.

    Step MAC = delta_cost / delta_co2 per scenario, computed via DataFrame merge.
    """
    fan_data = {}

    for iso in ISOS:
        iso_df = df[df['iso'] == iso]
        fan_data[iso] = {'p10': [None], 'p50': [None], 'p90': [None]}

        for i in range(1, len(THRESHOLDS)):
            t_prev, t_curr = THRESHOLDS[i - 1], THRESHOLDS[i]

            # Use total_cost (full LCOE) for MAC — delta between adjacent thresholds
            # gives incremental capital required for each step. Baseline offset cancels
            # in the CO₂ delta so no adjustment needed on the denominator side.
            cost_field = 'cost_total_cost' if 'cost_total_cost' in iso_df.columns else 'cost_effective_cost'
            prev = iso_df.loc[iso_df['threshold'] == t_prev,
                              ['scenario', cost_field, 'co2_total_co2_abated_tons',
                               'annual_demand_mwh']].set_index('scenario')
            curr = iso_df.loc[iso_df['threshold'] == t_curr,
                              ['scenario', cost_field,
                               'co2_total_co2_abated_tons']].set_index('scenario')

            merged = prev.join(curr, rsuffix='_next', how='inner')
            if merged.empty:
                for p in ['p10', 'p50', 'p90']:
                    fan_data[iso][p].append(None)
                continue

            delta_cost = ((merged[f'{cost_field}_next'] - merged[cost_field])
                          * merged['annual_demand_mwh'])
            delta_co2 = (merged['co2_total_co2_abated_tons_next']
                         - merged['co2_total_co2_abated_tons'])

            valid = (delta_co2 > 0) & (delta_cost >= 0)
            step_macs = (delta_cost[valid] / delta_co2[valid]).values

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
    """Compute monotonic envelope MAC AND path-constrained MAC in a single pass.

    Consolidates compute_monotonic_envelope() and compute_path_constrained_mac()
    which both filter to medium scenario and iterate over the same ISO/threshold
    pairs. One medium-scenario filter, one ISO loop, one threshold loop.

    Envelope: running max to enforce non-decreasing MAC.
    Path-constrained: enforces non-decreasing absolute resource deployment.
    """
    envelope = {}
    path_mac = {}
    med_df = df[df['scenario'].isin(MEDIUM_KEYS_SET)]

    for iso in ISOS:
        iso_med = med_df[med_df['iso'] == iso].sort_values('threshold')
        demand_mwh = iso_med['annual_demand_mwh'].iloc[0] if len(iso_med) > 0 else 1

        # Build threshold lookup once for both computations
        results_by_t = {}
        for _, row in iso_med.iterrows():
            results_by_t[row['threshold']] = row

        # CO₂ already avoided by existing clean — not credited to new investment
        existing_avoidance = EXISTING_CLEAN_AVOIDANCE_TONS.get(iso, 0)

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

            eff_cost = float(row['cost_effective_cost'])
            co2_tons = float(row['co2_total_co2_abated_tons'])
            total_cost = float(row['cost_total_cost']) if 'cost_total_cost' in row.index else float(row['cost_effective_cost'])

            # CO₂ abated by new capital only (subtract existing clean baseline)
            co2_net = max(0.0, (co2_tons if pd.notna(co2_tons) else 0.0) - existing_avoidance)

            # ── Envelope data collection ──
            costs_at_t.append(total_cost)
            co2_at_t.append(co2_net)

            # Use total_cost (full LCOE) for proper new-capital MAC
            if co2_net > 0:
                raw_macs.append(round((total_cost * demand_mwh) / co2_net, 1))
            else:
                raw_macs.append(None)

            # ── Path-constrained computation ──
            if has_any_row:
                mix = {r: int(row[f'mix_{r}']) for r in RESOURCE_TYPES}
                # v5.0: procurement_pct is always 100 (baked into resource percentages)
                batt = int(row['battery_dispatch_pct'])
                ldes = int(row['ldes_dispatch_pct'])
                h2 = int(row.get('h2_dispatch_pct', 0)) if 'h2_dispatch_pct' in row.index else 0

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
                constrained_total = max(total_cost, prev_cost)

                # Average MAC (CO₂ net of existing baseline)
                if co2_net > 0:
                    avg_mac = round((constrained_total * demand_mwh) / co2_net, 1)
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
                prev_co2 = co2_net
            else:
                path_macs.append(None)
                path_mixes.append(None)
                path_costs.append(None)

        # ── Post-loop: build envelope from collected data ──
        # PCHIP spline + isotonic regression for smooth monotonic MAC
        # (matches methodology in step6_compute_optimal_targets.py)
        env_macs = [None] * len(THRESHOLDS)
        step_env = [None] * len(THRESHOLDS)

        if HAS_SCIPY:
            # --- Average MAC envelope: isotonic regression ---
            valid_avg = [(i, v) for i, v in enumerate(raw_macs) if v is not None]
            if len(valid_avg) >= 3:
                idx_avg, vals_avg = zip(*valid_avg)
                iso_result = _scipy_isotonic(np.array(vals_avg, dtype=np.float64))
                smoothed_avg = iso_result.x if hasattr(iso_result, 'x') else iso_result
                for j, i in enumerate(idx_avg):
                    env_macs[i] = round(float(smoothed_avg[j]), 1)
            else:
                env_macs = list(raw_macs)

            # --- Stepwise marginal MAC: PCHIP derivative + isotonic ---
            valid_pts = [(i, co2_at_t[i], costs_at_t[i])
                         for i in range(len(THRESHOLDS))
                         if costs_at_t[i] is not None and co2_at_t[i] and co2_at_t[i] > 0]
            if len(valid_pts) >= 3:
                v_idx, v_co2, v_cost = zip(*valid_pts)
                co2_arr = np.array(v_co2, dtype=np.float64)
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
                    raw_deriv = pchip.derivative()(co2_mono)
                    raw_deriv = np.maximum(raw_deriv, 0.01)
                    iso_step = _scipy_isotonic(raw_deriv)
                    smoothed_step = iso_step.x if hasattr(iso_step, 'x') else iso_step
                    for j, i in enumerate(idx_mono):
                        step_env[i] = round(min(float(smoothed_step[j]), 9999), 1)
        else:
            # Fallback: running max (no scipy)
            running_max = 0
            for i, mac in enumerate(raw_macs):
                if mac is not None:
                    running_max = max(running_max, mac)
                    env_macs[i] = round(running_max, 1)
            step_running_max = 0
            for i in range(1, len(THRESHOLDS)):
                if costs_at_t[i] is not None and costs_at_t[i - 1] is not None:
                    delta_cost = (costs_at_t[i] - costs_at_t[i - 1]) * demand_mwh
                    delta_co2 = (co2_at_t[i] or 0) - (co2_at_t[i - 1] or 0)
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

    Loads DG parquets as DataFrames and uses vectorized MAC + groupby percentiles.
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

        # Load all DG files for this ISO into one DataFrame
        frames = [pd.read_parquet(f) for f in files]
        dg_df = pd.concat(frames, ignore_index=True)

        # Check required columns
        co2_col = 'co2_total_co2_abated_tons'
        demand_col = 'annual_demand_mwh'
        if co2_col not in dg_df.columns or demand_col not in dg_df.columns:
            continue
        if 'year' not in dg_df.columns or 'growth_level' not in dg_df.columns:
            continue

        # Vectorized MAC computation
        valid = ((dg_df[co2_col] > 0) & dg_df['cost_effective_cost'].notna()
                 & (dg_df[demand_col] > 0))
        dg_valid = dg_df[valid].copy()
        if dg_valid.empty:
            continue

        dg_valid['mac'] = ((dg_valid['cost_effective_cost'] * dg_valid[demand_col])
                           / dg_valid[co2_col])

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
        '// Auto-generated by step6_compute_mac_stats.py — do not edit manually',
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

    print("Loading optimizer results from parquets...")
    input_dir = find_input_dir(ISOS)
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
