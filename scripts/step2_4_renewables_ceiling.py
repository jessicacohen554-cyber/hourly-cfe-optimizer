#!/usr/bin/env python3
"""
Step 2.4: Renewables-Only CFE Ceiling Analysis — with Curtailment Economics
===========================================================================

Question answered: Holding existing clean generation frozen at its 2025
absolute-TWh baseline, what is the **highest achievable hourly CFE matching
score** reachable by adding only wind and solar? What mix achieves it, what
does it cost in undiscounted 2025 dollars, and at what capacity do diminishing
returns kick in?

**Curtailment economics (3 metric families):**

  1. Curtailment tracking:
     - total_gen_twh   : total supply produced (baseline + added), before clipping
     - useful_gen_twh  : hourly-matched supply = score × demand_twh
     - curtailed_twh   : total_gen_twh − useful_gen_twh
     - curtailment_rate: curtailed_twh / total_gen_twh
     - added_gen_twh   : nameplate generation from added renewables only

  2. Effective LCOE:
     - incremental_useful_twh : useful_gen_twh − baseline_useful_twh
       (the useful energy gained by adding renewables)
     - effective_lcoe_med/low/high : annual_cost / incremental_useful_twh
       ($/MWh of *actually matched* clean energy — this is what blows up)

  3. Marginal cost of CFE:
     - Along the cost frontier (cheapest path to each CFE target),
       d(cost)/d(CFE_pp) = how many $/yr to buy the next percentage point.
     - Along single-axis slices for comparison.

Performance notes (vs. prior version):
  - Total generation computed algebraically: O(N×K) instead of O(N×8760).
  - Fast-path restored for small grids (N ≤ chunk_size skips Python loop).
  - Internal computation uses lightweight dict-of-arrays; DataFrames built
    only for final output / surface writes.
  - Cost frontier computed once per scenario, shared between summary and
    cost-curve outputs.

Two output tables:
  ceiling_summary.parquet          — one row per scenario (enriched with curtailment)
  cost_curves.parquet              — frontier cost curve per scenario
                                     (joint + single-axis, all years × levels)

Per-scenario sweep surfaces (opt-in via --write-surfaces) include
curtailment + effective LCOE columns at every grid point.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Project imports ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_config import (  # noqa: E402
    ISOS,
    OFFSHORE_ISOS,
    REGIONAL_DEMAND_TWH,
    GRID_MIX_SHARES,
    EXTERNAL_CLEAN_IMPORTS_TWH,
    DEMAND_GROWTH_RATES,
    LCOE_TABLES,
    TX_TABLES,
    RESOURCE_CAPACITY_FACTORS,
    BASE_YEAR,
    H,
)
from dispatch_utils import get_supply_profiles  # noqa: E402
from eia_data_io import load_demand_profiles, load_generation_profiles  # noqa: E402

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
LEVELS = ['Low', 'Medium', 'High']

WIND_CAP_PCT = 400
SOLAR_CAP_PCT = 300
OFFSHORE_CAP_PCT = 100

COARSE_WIND_STEP = 5
COARSE_SOLAR_STEP = 5
COARSE_OFFSHORE_STEP = 5

FINE_WINDOW_PCT = 15
FINE_STEP = 1
FINE_OFFSHORE_STEP = 1

ELBOW_THRESHOLD_FIRST_KNEE = 0.10
ELBOW_THRESHOLD_DEEP = 0.02

FRONTIER_SCORE_STEP_PP = 1.0

RESOURCES = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind']

DATA_OUT = REPO_DIR / 'data' / 'step2.4-renewables-ceiling'


# -------------------------------------------------------------------------
# Utility: mix_pct ↔ GW conversion
# -------------------------------------------------------------------------
def pct_to_gw(pct, iso, resource, demand_twh):
    cf = RESOURCE_CAPACITY_FACTORS[resource][iso]
    return (pct / 100.0) * demand_twh * 1000.0 / (cf * 8760.0)


def pct_to_twh(pct, demand_twh):
    return (pct / 100.0) * demand_twh


def cost_per_pct(resource, iso, demand_twh, lcoe_level='Medium', tx_level='Medium'):
    lcoe = LCOE_TABLES[resource][lcoe_level][iso]
    tx_entry = TX_TABLES[resource][tx_level]
    tx = tx_entry[iso] if isinstance(tx_entry, dict) else tx_entry
    return 0.01 * demand_twh * 1e6 * (lcoe + tx)


# -------------------------------------------------------------------------
# Per-scenario ceiling kernel
# -------------------------------------------------------------------------
def build_supply_matrix(iso, gen_profiles):
    profs = get_supply_profiles(iso, gen_profiles)
    return np.stack([
        np.asarray(profs[r][:H], dtype=np.float64) for r in RESOURCES
    ])


def build_demand_arr(iso, demand_data):
    yd = demand_data.get(iso, {}).get('2025') or demand_data.get(iso, {}).get('2024')
    if yd is None:
        raise RuntimeError(f"No demand profile for {iso}")
    arr = np.asarray(yd['normalized'][:H], dtype=np.float64)
    s = arr.sum()
    if s > 0 and abs(s - 1.0) > 1e-9:
        arr = arr * (1.0 / s)
    return arr


_SOLAR_IDX = RESOURCES.index('solar')
_WIND_IDX = RESOURCES.index('wind')
_OFFSHORE_IDX = RESOURCES.index('offshore_wind')


def _fold_baseline_hourly(baseline, supply_matrix):
    pcts = np.array([baseline[r] for r in RESOURCES], dtype=np.float64)
    return (pcts / 100.0) @ supply_matrix   # (8760,)


def _compute_baseline_useful_frac(demand_arr, baseline_hourly):
    return float(np.minimum(demand_arr, baseline_hourly).sum())


# ── OPT 1: Algebraic total generation ────────────────────────────────────
#
# total_supply_frac[i] = baseline_sum + (added_mix[i] / 100) · swept_colsums
#
# swept_colsums is a (K,) vector (sum of each swept row's 8760 values),
# precomputed once.  This replaces an O(N×8760) sum with an O(N×K) dot,
# saving ~2.4B FLOPs on a 300K-point grid.

def _useful_scores(demand_arr, baseline_hourly, swept_rows, added_mix,
                   chunk_size=10000):
    """Vectorized useful fraction: sum_h min(demand_h, supply_h).

    OPT 3: Restores fast-path for small N (skips Python for-loop).

    Returns: (N,) useful fraction of demand.
    """
    N = added_mix.shape[0]
    if N <= chunk_size:
        supply = (added_mix / 100.0) @ swept_rows
        supply += baseline_hourly
        np.minimum(demand_arr, supply, out=supply)
        return supply.sum(axis=1)

    out = np.empty(N, dtype=np.float64)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        supply = (added_mix[start:end] / 100.0) @ swept_rows
        supply += baseline_hourly
        np.minimum(demand_arr, supply, out=supply)
        out[start:end] = supply.sum(axis=1)
    return out


def _total_gen_frac_algebraic(baseline_sum, swept_colsums, added_mix):
    """Total generation fraction via O(N×K) dot — no 8760-wide matrix.

    baseline_sum:    scalar — sum of baseline_hourly
    swept_colsums:   (K,)   — column-sums of the swept resource profiles
    added_mix:       (N, K) — added pct per resource per grid point

    Returns: (N,) total generation as fraction of demand (before clipping).
    """
    return baseline_sum + (added_mix / 100.0) @ swept_colsums


# ── OPT 4: Lightweight dict-of-arrays instead of DataFrame ──────────────
#
# _score_grid_arrays returns a plain dict of numpy arrays.  DataFrame
# construction is deferred to the write-surfaces path or final output,
# avoiding repeated allocation of 30-column × N-row DataFrames with
# broadcast string scalars for every internal grid call.

def _score_grid_arrays(
    iso,
    view,
    year,
    level,
    baseline,
    demand_twh,
    gf,
    demand_arr,
    supply_matrix,
    wind_axis,
    solar_axis,
    offshore_axis,
    phase_tag,
    baseline_hourly,
    baseline_sum,
    baseline_useful_frac,
):
    """Vectorized scorer returning a lightweight dict of arrays.

    All curtailment economics are computed here.  No DataFrame overhead.
    """
    is_offshore_view = offshore_axis is not None

    if is_offshore_view:
        WW, SS, OO = np.meshgrid(wind_axis, solar_axis, offshore_axis, indexing='ij')
        added_wind = WW.ravel()
        added_solar = SS.ravel()
        added_offshore = OO.ravel()
    else:
        WW, SS = np.meshgrid(wind_axis, solar_axis, indexing='ij')
        added_wind = WW.ravel()
        added_solar = SS.ravel()
        added_offshore = np.zeros(WW.size, dtype=np.float64)

    N = added_wind.size

    # Build swept matrix
    if is_offshore_view:
        swept_rows = supply_matrix[[_SOLAR_IDX, _WIND_IDX, _OFFSHORE_IDX], :]
        added_mix = np.stack([added_solar, added_wind, added_offshore], axis=1)
    else:
        swept_rows = supply_matrix[[_SOLAR_IDX, _WIND_IDX], :]
        added_mix = np.stack([added_solar, added_wind], axis=1)

    # OPT 1: precompute column sums (K,) — done once per call, reused for total gen
    swept_colsums = swept_rows.sum(axis=1)  # (K,)

    # Useful fraction via matmul + clip (the expensive path — must touch 8760)
    useful_frac = _useful_scores(demand_arr, baseline_hourly, swept_rows, added_mix)

    # OPT 1: Total generation fraction via algebraic O(N×K) dot
    total_gen_frac = _total_gen_frac_algebraic(baseline_sum, swept_colsums, added_mix)

    scores_pct = useful_frac * 100.0

    # Curtailment economics (all array ops, no DataFrames)
    useful_gen_twh = useful_frac * demand_twh
    total_gen_twh = total_gen_frac * demand_twh
    curtailed_twh = total_gen_twh - useful_gen_twh

    with np.errstate(divide='ignore', invalid='ignore'):
        curtailment_rate = np.where(total_gen_twh > 0, curtailed_twh / total_gen_twh, 0.0)

    added_gen_twh = (pct_to_twh(added_wind, demand_twh)
                     + pct_to_twh(added_solar, demand_twh)
                     + pct_to_twh(added_offshore, demand_twh))

    baseline_useful_twh = baseline_useful_frac * demand_twh
    incremental_useful_twh = useful_gen_twh - baseline_useful_twh

    # GW conversions
    wind_gw = pct_to_gw(added_wind, iso, 'wind', demand_twh)
    solar_gw = pct_to_gw(added_solar, iso, 'solar', demand_twh)
    offshore_gw = (
        pct_to_gw(added_offshore, iso, 'offshore_wind', demand_twh)
        if is_offshore_view else np.zeros(N)
    )

    # Cost
    def _cost(level_key):
        c = (cost_per_pct('solar', iso, demand_twh, level_key, 'Medium') * added_solar
             + cost_per_pct('wind', iso, demand_twh, level_key, 'Medium') * added_wind)
        if is_offshore_view:
            c = c + cost_per_pct('offshore_wind', iso, demand_twh, level_key, 'Medium') * added_offshore
        return c

    cost_med = _cost('Medium')
    cost_low = _cost('Low')
    cost_high = _cost('High')

    # Effective LCOE
    with np.errstate(divide='ignore', invalid='ignore'):
        eff_lcoe_med = np.where(incremental_useful_twh > 1e-9,
                                cost_med / (incremental_useful_twh * 1e6), np.nan)
        eff_lcoe_low = np.where(incremental_useful_twh > 1e-9,
                                cost_low / (incremental_useful_twh * 1e6), np.nan)
        eff_lcoe_high = np.where(incremental_useful_twh > 1e-9,
                                 cost_high / (incremental_useful_twh * 1e6), np.nan)

    return {
        'phase': phase_tag,
        'added_wind_pct': added_wind,
        'added_solar_pct': added_solar,
        'added_offshore_pct': added_offshore,
        'added_wind_gw': wind_gw,
        'added_solar_gw': solar_gw,
        'added_offshore_gw': offshore_gw,
        'added_wind_twh': pct_to_twh(added_wind, demand_twh),
        'added_solar_twh': pct_to_twh(added_solar, demand_twh),
        'added_offshore_twh': pct_to_twh(added_offshore, demand_twh),
        'score_pct': scores_pct,
        'total_gen_twh': total_gen_twh,
        'useful_gen_twh': useful_gen_twh,
        'curtailed_twh': curtailed_twh,
        'curtailment_rate': curtailment_rate,
        'added_gen_twh': added_gen_twh,
        'incremental_useful_twh': incremental_useful_twh,
        'effective_lcoe_med': eff_lcoe_med,
        'effective_lcoe_low': eff_lcoe_low,
        'effective_lcoe_high': eff_lcoe_high,
        'annual_cost_usd_med': cost_med,
        'annual_cost_usd_low': cost_low,
        'annual_cost_usd_high': cost_high,
    }


def _arrays_to_dataframe(arrays, iso, view, year, level, demand_twh, gf, baseline):
    """Promote a dict-of-arrays into a full DataFrame (for surface writes)."""
    N = len(arrays['score_pct'])
    df = pd.DataFrame(arrays)
    df['iso'] = iso
    df['view'] = view
    df['year'] = year
    df['demand_level'] = level
    df['demand_twh'] = demand_twh
    df['demand_growth_factor'] = gf
    df['baseline_clean_firm_pct'] = baseline['clean_firm']
    df['baseline_solar_pct'] = baseline['solar']
    df['baseline_wind_pct'] = baseline['wind']
    df['baseline_hydro_pct'] = baseline['hydro']
    df['baseline_offshore_pct'] = baseline['offshore_wind']
    return df


def _concat_array_dicts(*dicts):
    """Concatenate multiple array-dicts along axis 0."""
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts]
        if isinstance(vals[0], np.ndarray):
            out[k] = np.concatenate(vals)
        else:
            # scalar (phase_tag string) — repeat per chunk
            parts = []
            for d in dicts:
                v = d[k]
                n = len(d['score_pct'])
                if isinstance(v, np.ndarray):
                    parts.append(v)
                else:
                    parts.append(np.full(n, v, dtype=object))
            out[k] = np.concatenate(parts)
    return out


def sweep_scenario(
    iso,
    view,
    year,
    level,
    demand_arr,
    supply_matrix,
):
    """Two-phase renewables-only ceiling surface for one (iso, view, year, level).

    Returns a dict-of-arrays (lightweight).  Caller converts to DataFrame
    only if writing surfaces.
    """
    gf = (1.0 + DEMAND_GROWTH_RATES[iso][level]) ** max(0, year - BASE_YEAR)
    demand_twh = REGIONAL_DEMAND_TWH[iso] * gf

    gm = GRID_MIX_SHARES[iso]
    baseline = {r: gm.get(r, 0.0) / gf for r in RESOURCES}
    imports_twh = EXTERNAL_CLEAN_IMPORTS_TWH.get(iso, 0.0)
    if imports_twh > 0:
        baseline['clean_firm'] += imports_twh / demand_twh * 100.0

    is_offshore_view = (view == 'onshore_plus_offshore') and (iso in OFFSHORE_ISOS)

    # Precompute once per scenario
    baseline_hourly = _fold_baseline_hourly(baseline, supply_matrix)
    baseline_sum = float(baseline_hourly.sum())  # OPT 1
    baseline_useful_frac = _compute_baseline_useful_frac(demand_arr, baseline_hourly)

    common = dict(
        baseline=baseline, demand_twh=demand_twh, gf=gf,
        demand_arr=demand_arr, supply_matrix=supply_matrix,
        baseline_hourly=baseline_hourly,
        baseline_sum=baseline_sum,
        baseline_useful_frac=baseline_useful_frac,
    )

    # ── Phase 1: coarse ──────────────────────────────────────────────────
    wind_c = np.arange(0, WIND_CAP_PCT + 1, COARSE_WIND_STEP, dtype=np.float64)
    solar_c = np.arange(0, SOLAR_CAP_PCT + 1, COARSE_SOLAR_STEP, dtype=np.float64)
    offshore_c = (
        np.arange(0, OFFSHORE_CAP_PCT + 1, COARSE_OFFSHORE_STEP, dtype=np.float64)
        if is_offshore_view else None
    )
    coarse = _score_grid_arrays(iso, view, year, level,
                                wind_axis=wind_c, solar_axis=solar_c,
                                offshore_axis=offshore_c, phase_tag='coarse',
                                **common)

    imax = int(np.argmax(coarse['score_pct']))
    w_star = float(coarse['added_wind_pct'][imax])
    s_star = float(coarse['added_solar_pct'][imax])
    o_star = float(coarse['added_offshore_pct'][imax])

    # ── Phase 2: fine ────────────────────────────────────────────────────
    def _window(center, cap, step, min_val=0):
        lo = max(min_val, center - FINE_WINDOW_PCT)
        hi = min(cap, center + FINE_WINDOW_PCT)
        return np.arange(lo, hi + step / 2, step, dtype=np.float64)

    wind_f = _window(w_star, WIND_CAP_PCT, FINE_STEP)
    solar_f = _window(s_star, SOLAR_CAP_PCT, FINE_STEP)
    offshore_f = (
        _window(o_star, OFFSHORE_CAP_PCT, FINE_OFFSHORE_STEP)
        if is_offshore_view else None
    )
    fine = _score_grid_arrays(iso, view, year, level,
                              wind_axis=wind_f, solar_axis=solar_f,
                              offshore_axis=offshore_f, phase_tag='fine',
                              **common)

    # Single-axis stripes
    wind_stripe = _score_grid_arrays(
        iso, view, year, level,
        wind_axis=np.arange(0, WIND_CAP_PCT + 1, 1, dtype=np.float64),
        solar_axis=np.array([0.0]),
        offshore_axis=np.array([0.0]) if is_offshore_view else None,
        phase_tag='axis_wind', **common)

    solar_stripe = _score_grid_arrays(
        iso, view, year, level,
        wind_axis=np.array([0.0]),
        solar_axis=np.arange(0, SOLAR_CAP_PCT + 1, 1, dtype=np.float64),
        offshore_axis=np.array([0.0]) if is_offshore_view else None,
        phase_tag='axis_solar', **common)

    parts = [coarse, fine, wind_stripe, solar_stripe]
    if is_offshore_view:
        offshore_stripe = _score_grid_arrays(
            iso, view, year, level,
            wind_axis=np.array([0.0]), solar_axis=np.array([0.0]),
            offshore_axis=np.arange(0, OFFSHORE_CAP_PCT + 1, 1, dtype=np.float64),
            phase_tag='axis_offshore', **common)
        parts.append(offshore_stripe)

    merged = _concat_array_dicts(*parts)

    # Attach scenario metadata for downstream consumers
    merged['_meta'] = {
        'iso': iso, 'view': view, 'year': year, 'level': level,
        'demand_twh': demand_twh, 'gf': gf, 'baseline': baseline,
    }
    return merged


# -------------------------------------------------------------------------
# Cost frontier extraction
# -------------------------------------------------------------------------
def extract_cost_frontier(arrays, phase_mask=None):
    """Extract the cheapest path to each achievable CFE score level.

    Operates on dict-of-arrays (no DataFrame needed).

    If phase_mask is provided (boolean array), restricts to those indices.
    """
    if phase_mask is not None:
        sub = {k: v[phase_mask] if isinstance(v, np.ndarray) else v
               for k, v in arrays.items()}
    else:
        sub = arrays

    scores = sub['score_pct']
    if len(scores) == 0:
        return pd.DataFrame()

    baseline_score = float(scores.min())
    ceiling_score = float(scores.max())

    targets = np.arange(
        np.floor(baseline_score),
        np.ceil(ceiling_score) + FRONTIER_SCORE_STEP_PP,
        FRONTIER_SCORE_STEP_PP
    )

    # Vectorized frontier: for each target, find cheapest eligible point.
    cost = sub['annual_cost_usd_med']
    rows = []
    for target in targets:
        mask = scores >= target - 0.5
        if not np.any(mask):
            continue
        eligible_cost = np.where(mask, cost, np.inf)
        imin = int(np.argmin(eligible_cost))
        rows.append({
            'target_score_pct': target,
            'achieved_score_pct': float(scores[imin]),
            'min_annual_cost_med': float(sub['annual_cost_usd_med'][imin]),
            'min_annual_cost_low': float(sub['annual_cost_usd_low'][imin]),
            'min_annual_cost_high': float(sub['annual_cost_usd_high'][imin]),
            'optimal_wind_gw': float(sub['added_wind_gw'][imin]),
            'optimal_solar_gw': float(sub['added_solar_gw'][imin]),
            'optimal_offshore_gw': float(sub['added_offshore_gw'][imin]),
            'optimal_wind_pct': float(sub['added_wind_pct'][imin]),
            'optimal_solar_pct': float(sub['added_solar_pct'][imin]),
            'optimal_offshore_pct': float(sub['added_offshore_pct'][imin]),
            'total_gen_twh': float(sub['total_gen_twh'][imin]),
            'useful_gen_twh': float(sub['useful_gen_twh'][imin]),
            'curtailed_twh': float(sub['curtailed_twh'][imin]),
            'curtailment_rate': float(sub['curtailment_rate'][imin]),
            'added_gen_twh': float(sub['added_gen_twh'][imin]),
            'incremental_useful_twh': float(sub['incremental_useful_twh'][imin]),
            'effective_lcoe_med': float(sub['effective_lcoe_med'][imin]),
        })

    if not rows:
        return pd.DataFrame()

    frontier = pd.DataFrame(rows).sort_values('target_score_pct').reset_index(drop=True)

    # Marginal cost per pp of CFE
    dcost = np.diff(frontier['min_annual_cost_med'].values)
    dscore = np.diff(frontier['achieved_score_pct'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        marginal = np.where(dscore > 1e-6, dcost / dscore, np.nan)
    frontier['marginal_cost_per_pp'] = np.concatenate([[np.nan], marginal])

    # Marginal effective LCOE
    d_useful = np.diff(frontier['incremental_useful_twh'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        marginal_eff = np.where(d_useful > 1e-9, dcost / (d_useful * 1e6), np.nan)
    frontier['marginal_effective_lcoe_med'] = np.concatenate([[np.nan], marginal_eff])

    return frontier


def _phase_mask(arrays, tags):
    """Boolean mask for rows whose phase is in `tags`."""
    phase = arrays['phase']
    if isinstance(phase, np.ndarray):
        return np.isin(phase, tags)
    # scalar — all rows have the same phase
    return np.full(len(arrays['score_pct']), phase in tags)


def extract_all_cost_curves(arrays):
    """Extract cost frontiers for joint surface + each single-axis slice.

    OPT 2: Called once per scenario; result is reused by extract_summary_row.
    """
    meta = arrays['_meta']
    curves = []

    # Joint frontier (coarse + fine)
    joint_mask = _phase_mask(arrays, ['coarse', 'fine'])
    joint = extract_cost_frontier(arrays, phase_mask=joint_mask)
    if len(joint):
        joint['frontier_type'] = 'joint_optimal'
        for k in ('iso', 'view', 'year', 'level'):
            joint[k] = meta[k] if k != 'level' else meta['level']
        joint['demand_twh'] = meta['demand_twh']
        joint.rename(columns={'level': 'demand_level'}, inplace=True, errors='ignore')
        curves.append(joint)

    # Single-axis frontiers
    for axis_tag, label in [('axis_wind', 'wind_only'),
                            ('axis_solar', 'solar_only'),
                            ('axis_offshore', 'offshore_only')]:
        mask = _phase_mask(arrays, [axis_tag])
        if not np.any(mask):
            continue
        ax = extract_cost_frontier(arrays, phase_mask=mask)
        if len(ax):
            ax['frontier_type'] = label
            for k in ('iso', 'view', 'year'):
                ax[k] = meta[k]
            ax['demand_level'] = meta['level']
            ax['demand_twh'] = meta['demand_twh']
            curves.append(ax)

    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


# -------------------------------------------------------------------------
# Saturation elbow extraction
# -------------------------------------------------------------------------
def saturation_elbow_single_axis(gw_arr, score_arr, threshold_pp_per_gw):
    gw_arr = np.asarray(gw_arr, dtype=np.float64)
    score_arr = np.asarray(score_arr, dtype=np.float64)
    order = np.argsort(gw_arr)
    gw_arr = gw_arr[order]
    score_arr = score_arr[order]
    if len(gw_arr) < 4:
        return float('nan')

    baseline_score = score_arr[0]
    start_idx = int(np.searchsorted(score_arr, baseline_score + 1.0, side='left'))
    start_idx = min(start_idx, len(score_arr) - 4)

    dscore = np.diff(score_arr)
    dgw = np.diff(gw_arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        marginal = np.where(dgw > 0, dscore / dgw, np.inf)
    below = marginal < threshold_pp_per_gw

    for i in range(start_idx, len(below) - 2):
        if below[i] and below[i + 1] and below[i + 2]:
            return float(gw_arr[i])
    return float(gw_arr[-1])


def joint_efficient_elbow(arrays, phase_tags, threshold_pp_per_gw):
    mask = _phase_mask(arrays, phase_tags)
    total_gw = (arrays['added_wind_gw'][mask]
                + arrays['added_solar_gw'][mask]
                + arrays['added_offshore_gw'][mask])
    scores = arrays['score_pct'][mask]
    envelope = (
        pd.DataFrame({'gw': total_gw, 'score': scores})
        .groupby(pd.cut(total_gw, bins=100, duplicates='drop'), observed=True)
        .agg(gw=('gw', 'mean'), score=('score', 'max'))
        .dropna()
        .sort_values('gw')
    )
    return saturation_elbow_single_axis(
        envelope['gw'].values, envelope['score'].values, threshold_pp_per_gw
    )


# -------------------------------------------------------------------------
# Summary row extraction
# -------------------------------------------------------------------------
def extract_summary_row(arrays, cost_curves_df):
    """Summarize one scenario's surface into a single row.

    OPT 2: Receives pre-computed cost_curves_df instead of recomputing.
    """
    meta = arrays['_meta']
    iso = meta['iso']; view = meta['view']
    year = int(meta['year']); level = meta['level']
    demand_twh = meta['demand_twh']; gf = meta['gf']
    baseline = meta['baseline']

    # Baseline score: where all added pcts are zero
    zero_mask = ((arrays['added_wind_pct'] == 0)
                 & (arrays['added_solar_pct'] == 0)
                 & (arrays['added_offshore_pct'] == 0))
    zero_scores = arrays['score_pct'][zero_mask]
    baseline_score = float(zero_scores[0]) if len(zero_scores) else float('nan')

    # Ceiling point
    imax = int(np.argmax(arrays['score_pct']))

    def _top(key):
        return float(arrays[key][imax])

    # Elbows from single-axis stripes
    wind_mask = _phase_mask(arrays, ['axis_wind'])
    wind_gw_arr = arrays['added_wind_gw'][wind_mask]
    wind_score_arr = arrays['score_pct'][wind_mask]
    order = np.argsort(wind_gw_arr)
    wind_gw_arr = wind_gw_arr[order]
    wind_score_arr = wind_score_arr[order]
    sat_wind_first = saturation_elbow_single_axis(wind_gw_arr, wind_score_arr, ELBOW_THRESHOLD_FIRST_KNEE)
    sat_wind_deep = saturation_elbow_single_axis(wind_gw_arr, wind_score_arr, ELBOW_THRESHOLD_DEEP)
    wind_only_max = float(wind_score_arr.max()) if len(wind_score_arr) else float('nan')

    solar_mask = _phase_mask(arrays, ['axis_solar'])
    solar_gw_arr = arrays['added_solar_gw'][solar_mask]
    solar_score_arr = arrays['score_pct'][solar_mask]
    order = np.argsort(solar_gw_arr)
    solar_gw_arr = solar_gw_arr[order]
    solar_score_arr = solar_score_arr[order]
    sat_solar_first = saturation_elbow_single_axis(solar_gw_arr, solar_score_arr, ELBOW_THRESHOLD_FIRST_KNEE)
    sat_solar_deep = saturation_elbow_single_axis(solar_gw_arr, solar_score_arr, ELBOW_THRESHOLD_DEEP)
    solar_only_max = float(solar_score_arr.max()) if len(solar_score_arr) else float('nan')

    if view == 'onshore_plus_offshore':
        off_mask = _phase_mask(arrays, ['axis_offshore'])
        off_gw = arrays['added_offshore_gw'][off_mask]
        off_sc = arrays['score_pct'][off_mask]
        order = np.argsort(off_gw)
        sat_offshore_first = saturation_elbow_single_axis(off_gw[order], off_sc[order], ELBOW_THRESHOLD_FIRST_KNEE)
        sat_offshore_deep = saturation_elbow_single_axis(off_gw[order], off_sc[order], ELBOW_THRESHOLD_DEEP)
    else:
        sat_offshore_first = float('nan')
        sat_offshore_deep = float('nan')

    sat_joint_first = joint_efficient_elbow(arrays, ['fine'], ELBOW_THRESHOLD_FIRST_KNEE)
    sat_joint_deep = joint_efficient_elbow(arrays, ['fine'], ELBOW_THRESHOLD_DEEP)

    added_wind_twh = _top('added_wind_twh')
    added_solar_twh = _top('added_solar_twh')
    added_offshore_twh = _top('added_offshore_twh')
    total_added_twh = added_wind_twh + added_solar_twh + added_offshore_twh
    wind_share = added_wind_twh / total_added_twh if total_added_twh > 0 else float('nan')

    # OPT 2: Use pre-computed frontier for milestone LCOEs
    joint_frontier = cost_curves_df[cost_curves_df.frontier_type == 'joint_optimal'] \
        if len(cost_curves_df) else pd.DataFrame()
    milestone_lcoes = {}
    for target in [60, 70, 80, 85, 90, 95]:
        hit = joint_frontier[joint_frontier.target_score_pct >= target] if len(joint_frontier) else pd.DataFrame()
        if len(hit):
            row = hit.iloc[0]
            milestone_lcoes[f'eff_lcoe_at_{target}pct'] = float(row['effective_lcoe_med'])
            milestone_lcoes[f'curtailment_at_{target}pct'] = float(row['curtailment_rate'])
            val = row.get('marginal_cost_per_pp', np.nan)
            milestone_lcoes[f'marginal_cost_pp_at_{target}pct'] = float(val) if not np.isnan(val) else np.nan
        else:
            milestone_lcoes[f'eff_lcoe_at_{target}pct'] = float('nan')
            milestone_lcoes[f'curtailment_at_{target}pct'] = float('nan')
            milestone_lcoes[f'marginal_cost_pp_at_{target}pct'] = float('nan')

    result = {
        'iso': iso,
        'view': view,
        'year': year,
        'demand_level': level,
        'demand_twh': demand_twh,
        'demand_growth_factor': gf,
        'baseline_score_pct': baseline_score,
        'ceiling_score_pct': _top('score_pct'),
        'score_lift_pp': _top('score_pct') - baseline_score,
        'ceiling_added_wind_gw': _top('added_wind_gw'),
        'ceiling_added_solar_gw': _top('added_solar_gw'),
        'ceiling_added_offshore_gw': _top('added_offshore_gw'),
        'ceiling_added_wind_twh': added_wind_twh,
        'ceiling_added_solar_twh': added_solar_twh,
        'ceiling_added_offshore_twh': added_offshore_twh,
        'ceiling_wind_share_of_added_twh': wind_share,
        'ceiling_annual_cost_usd_med': _top('annual_cost_usd_med'),
        'ceiling_annual_cost_usd_low': _top('annual_cost_usd_low'),
        'ceiling_annual_cost_usd_high': _top('annual_cost_usd_high'),
        'ceiling_total_gen_twh': _top('total_gen_twh'),
        'ceiling_useful_gen_twh': _top('useful_gen_twh'),
        'ceiling_curtailed_twh': _top('curtailed_twh'),
        'ceiling_curtailment_rate': _top('curtailment_rate'),
        'ceiling_effective_lcoe_med': _top('effective_lcoe_med')
            if not np.isnan(arrays['effective_lcoe_med'][imax]) else np.nan,
        'wind_only_max_score_pct': wind_only_max,
        'solar_only_max_score_pct': solar_only_max,
        'sat_wind_gw_first_knee': sat_wind_first,
        'sat_wind_gw_deep': sat_wind_deep,
        'sat_solar_gw_first_knee': sat_solar_first,
        'sat_solar_gw_deep': sat_solar_deep,
        'sat_offshore_gw_first_knee': sat_offshore_first,
        'sat_offshore_gw_deep': sat_offshore_deep,
        'sat_joint_total_gw_first_knee': sat_joint_first,
        'sat_joint_total_gw_deep': sat_joint_deep,
    }
    result.update(milestone_lcoes)
    return result


# -------------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------------
def run_verification(summary_df):
    msgs = []

    bad = summary_df[summary_df.ceiling_score_pct < summary_df.baseline_score_pct - 1e-6]
    msgs.append(f"Monotonicity {'FAIL: ' + str(len(bad)) + ' rows' if len(bad) else 'OK'}")

    for iso in OFFSHORE_ISOS:
        for yr in YEARS:
            for lvl in LEVELS:
                a = summary_df[(summary_df.iso == iso) & (summary_df.year == yr)
                               & (summary_df.demand_level == lvl) & (summary_df.view == 'onshore_only')]
                b = summary_df[(summary_df.iso == iso) & (summary_df.year == yr)
                               & (summary_df.demand_level == lvl) & (summary_df.view == 'onshore_plus_offshore')]
                if len(a) and len(b):
                    if b['ceiling_score_pct'].iloc[0] < a['ceiling_score_pct'].iloc[0] - 1e-6:
                        msgs.append(f"Offshore-superset FAIL: {iso} {yr} {lvl}")
    msgs.append("Offshore-superset check complete")

    for iso in ISOS:
        for lvl in LEVELS:
            sub = summary_df[(summary_df.iso == iso) & (summary_df.demand_level == lvl)
                             & (summary_df.view == 'onshore_only')].sort_values('year')
            if len(sub) >= 2:
                s = sub['baseline_score_pct'].values
                if np.any(np.diff(s) > 1e-3):
                    msgs.append(f"Baseline-monotone FAIL: {iso} {lvl}")
    msgs.append("Baseline-monotone check complete")

    ref = summary_df[(summary_df.iso == 'ERCOT') & (summary_df.year == 2030)
                     & (summary_df.demand_level == 'Medium') & (summary_df.view == 'onshore_only')]
    if len(ref):
        r = ref.iloc[0]
        exp = (r.ceiling_added_wind_twh * 1e6 * (LCOE_TABLES['wind']['Medium']['ERCOT']
                                                 + TX_TABLES['wind']['Medium']['ERCOT'])
               + r.ceiling_added_solar_twh * 1e6 * (LCOE_TABLES['solar']['Medium']['ERCOT']
                                                    + TX_TABLES['solar']['Medium']['ERCOT']))
        diff = abs(r.ceiling_annual_cost_usd_med - exp) / max(exp, 1.0)
        msgs.append(f"Cost spot-check {'FAIL' if diff > 1e-6 else 'OK'} (ERCOT 2030 Med)")

    # Curtailment checks
    if 'ceiling_curtailed_twh' in summary_df.columns:
        neg = summary_df[summary_df.ceiling_curtailed_twh < -1e-6]
        msgs.append(f"Curtailment non-negativity {'FAIL: ' + str(len(neg)) if len(neg) else 'OK'}")

    base_2025 = summary_df[(summary_df.year == 2025) & (summary_df.demand_level == 'Medium')
                           & (summary_df.view == 'onshore_only')]
    has_curt = (base_2025['ceiling_curtailment_rate'] > 0.01).sum()
    msgs.append(f"Curtailment at ceiling > 1%: {has_curt}/{len(base_2025)} ISOs (2025 Med)")

    high_curt = summary_df[summary_df.ceiling_curtailment_rate > 0.30]
    if len(high_curt):
        msgs.append(f"High-curtailment rows (>30%): {len(high_curt)}, "
                    f"median eff LCOE: ${high_curt.ceiling_effective_lcoe_med.median():.1f}/MWh")

    print("\n=== Verification ===")
    for m in msgs:
        print(f"  {m}")


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--isos', nargs='+', default=list(ISOS))
    ap.add_argument('--years', nargs='+', type=int, default=YEARS)
    ap.add_argument('--levels', nargs='+', default=LEVELS)
    ap.add_argument('--write-surfaces', action='store_true',
                    help='Write per-scenario surface parquets with curtailment columns')
    ap.add_argument('--skip-offshore-view', action='store_true')
    args = ap.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    print("Loading EIA profiles …", flush=True)
    demand_data = load_demand_profiles()
    gen_profiles = load_generation_profiles()

    iso_cache = {}
    for iso in args.isos:
        iso_cache[iso] = {
            'demand_arr': build_demand_arr(iso, demand_data),
            'supply_matrix': build_supply_matrix(iso, gen_profiles),
        }
    print(f"Profiles loaded in {time.monotonic()-t0:.1f}s", flush=True)

    summary_rows = []
    cost_curve_frames = []
    total_scenarios = 0
    t1 = time.monotonic()

    for iso in args.isos:
        da = iso_cache[iso]['demand_arr']
        sm = iso_cache[iso]['supply_matrix']
        for view in ['onshore_only', 'onshore_plus_offshore']:
            if view == 'onshore_plus_offshore':
                if args.skip_offshore_view:
                    continue
                if iso not in OFFSHORE_ISOS:
                    continue
            for year in args.years:
                for level in args.levels:
                    t_scn = time.monotonic()

                    # OPT 4: sweep returns dict-of-arrays, not DataFrame
                    arrays = sweep_scenario(iso, view, year, level, da, sm)

                    # OPT 2: compute frontier once, pass to summary
                    curves = extract_all_cost_curves(arrays)
                    if len(curves):
                        cost_curve_frames.append(curves)

                    summary_rows.append(extract_summary_row(arrays, curves))
                    total_scenarios += 1

                    # OPT 4: DataFrame only built when writing surfaces
                    if args.write_surfaces:
                        meta = arrays['_meta']
                        # Remove _meta before DataFrame conversion
                        arr_copy = {k: v for k, v in arrays.items() if k != '_meta'}
                        df = _arrays_to_dataframe(arr_copy, meta['iso'], meta['view'],
                                                  meta['year'], meta['level'],
                                                  meta['demand_twh'], meta['gf'],
                                                  meta['baseline'])
                        fn = DATA_OUT / f"sweep_surface_{iso}_{view}_{year}_{level}.parquet"
                        df.to_parquet(fn, compression='zstd', index=False)

                    # Progress line
                    imax = int(np.argmax(arrays['score_pct']))
                    zero_mask = ((arrays['added_wind_pct'] == 0)
                                 & (arrays['added_solar_pct'] == 0)
                                 & (arrays['added_offshore_pct'] == 0))
                    base_s = float(arrays['score_pct'][zero_mask][0]) if np.any(zero_mask) else float('nan')
                    dt = time.monotonic() - t_scn
                    print(f"  {iso:6} {view:22} {year} {level:6} "
                          f"pts={len(arrays['score_pct']):>6} "
                          f"base={base_s:.1f}% "
                          f"ceil={arrays['score_pct'][imax]:.1f}% "
                          f"curt={arrays['curtailment_rate'][imax]:.0%} "
                          f"eff_lcoe=${arrays['effective_lcoe_med'][imax]:.0f}/MWh "
                          f"t={dt:.2f}s",
                          flush=True)

    print(f"\n{total_scenarios} scenarios in {time.monotonic()-t1:.1f}s", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_out = DATA_OUT / 'ceiling_summary.parquet'
    summary_df.to_parquet(summary_out, compression='zstd', index=False)
    print(f"Wrote {summary_out} ({len(summary_df)} rows)")

    if cost_curve_frames:
        cost_curves = pd.concat(cost_curve_frames, ignore_index=True)
        curves_out = DATA_OUT / 'cost_curves.parquet'
        cost_curves.to_parquet(curves_out, compression='zstd', index=False)
        print(f"Wrote {curves_out} ({len(cost_curves)} rows)")

    run_verification(summary_df)
    print(f"\nTotal wall time: {time.monotonic()-t0:.1f}s")


if __name__ == '__main__':
    main()
