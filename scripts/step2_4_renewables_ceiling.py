#!/usr/bin/env python3
"""
Step 2.4: Renewables-Only CFE Ceiling Analysis — with Curtailment Economics
===========================================================================

Question answered: Holding existing clean generation frozen at its 2025
absolute-TWh baseline, what is the **highest achievable hourly CFE matching
score** reachable by adding only wind and solar? What mix achieves it, what
does it cost in undiscounted 2025 dollars, and at what capacity do diminishing
returns kick in?

**NEW — Curtailment economics (3 metric families):**

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

Two output tables:
  ceiling_summary.parquet          — one row per scenario (enriched with curtailment)
  cost_curve_{iso}_{view}.parquet  — frontier cost curve per ISO/view
                                     (all years × levels stacked)

Per-scenario sweep surfaces (opt-in via --write-surfaces) now include
curtailment + effective LCOE columns at every grid point.

Two views per ISO:
  - onshore_only   : add onshore wind + solar (primary, all 7 ISOs)
  - onshore_plus_offshore : add onshore wind + solar + offshore wind
                     (NEISO, NYISO, PJM, CAISO)

Demand trajectory sweep:
  years    : 2025, 2030, 2035, 2040, 2045, 2050
  levels   : Low, Medium, High (CAGRs from pipeline_config.DEMAND_GROWTH_RATES)
  baseline : existing clean frozen at 2025 TWh — as demand grows, its share
             of demand shrinks (1 / demand_growth_factor).
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

# Two-phase sweep: coarse locates the ceiling, fine nails the elbow.
# All values in pipeline-native "mix_pct" = nameplate share of demand.
WIND_CAP_PCT = 400
SOLAR_CAP_PCT = 300
OFFSHORE_CAP_PCT = 100

# Phase 1 — coarse sweep over the full grid (cheap: ~1,300 pts / scenario).
COARSE_WIND_STEP = 5
COARSE_SOLAR_STEP = 5
COARSE_OFFSHORE_STEP = 5

# Phase 2 — fine sweep in a ±window around the coarse maximum (precise: elbow hunt).
FINE_WINDOW_PCT = 15          # ±15 pp window
FINE_STEP = 1                 # 1 pp step inside the window
FINE_OFFSHORE_STEP = 1

# Saturation elbow thresholds (marginal score-lift per additional GW).
ELBOW_THRESHOLD_FIRST_KNEE = 0.10   # pp CFE per GW — first sign of bending
ELBOW_THRESHOLD_DEEP = 0.02          # pp CFE per GW — fully diminishing returns

# Cost frontier: CFE score bins for the "cheapest path" curve.
FRONTIER_SCORE_STEP_PP = 1.0         # 1 pp bins for frontier extraction

# Resource columns we actually evaluate. Ordered to match the supply matrix.
RESOURCES = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind']

DATA_OUT = REPO_DIR / 'data' / 'step2.4-renewables-ceiling'


# -------------------------------------------------------------------------
# Utility: mix_pct ↔ GW conversion
# -------------------------------------------------------------------------
def pct_to_gw(pct, iso, resource, demand_twh):
    """Convert mix_pct (% of demand, pipeline-native) → nameplate GW."""
    cf = RESOURCE_CAPACITY_FACTORS[resource][iso]
    return (pct / 100.0) * demand_twh * 1000.0 / (cf * 8760.0)


def pct_to_twh(pct, demand_twh):
    return (pct / 100.0) * demand_twh


def cost_per_pct(resource, iso, demand_twh, lcoe_level='Medium', tx_level='Medium'):
    """Return $/yr per 1 pp of demand added for this resource at this LCOE tier."""
    lcoe = LCOE_TABLES[resource][lcoe_level][iso]
    tx_entry = TX_TABLES[resource][tx_level]
    tx = tx_entry[iso] if isinstance(tx_entry, dict) else tx_entry
    return 0.01 * demand_twh * 1e6 * (lcoe + tx)


# -------------------------------------------------------------------------
# Per-scenario ceiling kernel
# -------------------------------------------------------------------------
def build_supply_matrix(iso, gen_profiles):
    """(5, 8760) supply matrix limited to non-storage, non-hybrid resources."""
    profs = get_supply_profiles(iso, gen_profiles)
    mat = np.stack([
        np.asarray(profs[r][:H], dtype=np.float64) for r in RESOURCES
    ])
    return mat


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
    """Collapse frozen-resource contribution into a single (8760,) vector."""
    pcts = np.array([baseline[r] for r in RESOURCES], dtype=np.float64)
    return (pcts / 100.0) @ supply_matrix   # (8760,)


def _scores_and_generation_folded(demand_arr, baseline_hourly, swept_rows,
                                  added_mix, chunk_size=10000):
    """Vectorized scorer that returns BOTH useful and total generation.

    Returns:
        useful:       (N,) — sum_h min(demand_h, supply_h)   (fraction of demand)
        total_supply: (N,) — sum_h supply_h                  (fraction of demand)

    The difference (total_supply − useful) is the curtailed fraction.
    Multiply either by demand_twh to get TWh.
    """
    N = added_mix.shape[0]
    useful = np.empty(N, dtype=np.float64)
    total_supply = np.empty(N, dtype=np.float64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        supply = (added_mix[start:end] / 100.0) @ swept_rows   # (chunk, 8760)
        supply += baseline_hourly                                # broadcast add
        total_supply[start:end] = supply.sum(axis=1)
        np.minimum(demand_arr, supply, out=supply)
        useful[start:end] = supply.sum(axis=1)

    return useful, total_supply


def _score_grid(
    iso,
    view,
    year,
    level,
    baseline,             # dict of frozen mix_pcts per resource
    demand_twh,           # absolute demand TWh for this scenario
    gf,                   # demand growth factor
    demand_arr,           # (8760,) normalized
    supply_matrix,        # (5, 8760)
    wind_axis,
    solar_axis,
    offshore_axis,        # None → onshore-only view, 2-D grid
    phase_tag,            # 'coarse' | 'fine' | 'axis_*'
    baseline_hourly=None, # (8760,) — precomputed fold; built lazily if None
    baseline_useful_frac=None,  # scalar — useful fraction at zero added renewables
):
    """Vectorized scorer with full curtailment economics.

    New columns vs. original:
      total_gen_twh, useful_gen_twh, curtailed_twh, curtailment_rate,
      added_gen_twh, incremental_useful_twh,
      effective_lcoe_med, effective_lcoe_low, effective_lcoe_high
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

    if baseline_hourly is None:
        baseline_hourly = _fold_baseline_hourly(baseline, supply_matrix)

    # Build swept matrix
    if is_offshore_view:
        swept_rows = supply_matrix[[_SOLAR_IDX, _WIND_IDX, _OFFSHORE_IDX], :]
        added_mix = np.stack([added_solar, added_wind, added_offshore], axis=1)
    else:
        swept_rows = supply_matrix[[_SOLAR_IDX, _WIND_IDX], :]
        added_mix = np.stack([added_solar, added_wind], axis=1)

    # ── KEY CHANGE: get both useful AND total generation ─────────────────
    useful_frac, total_supply_frac = _scores_and_generation_folded(
        demand_arr, baseline_hourly, swept_rows, added_mix
    )

    scores_pct = useful_frac * 100.0

    # Convert fractions → TWh
    useful_gen_twh = useful_frac * demand_twh
    total_gen_twh = total_supply_frac * demand_twh
    curtailed_twh = total_gen_twh - useful_gen_twh

    # Curtailment rate: fraction of total generation that is wasted
    with np.errstate(divide='ignore', invalid='ignore'):
        curtailment_rate = np.where(
            total_gen_twh > 0,
            curtailed_twh / total_gen_twh,
            0.0
        )

    # Added generation TWh (nameplate, from new renewables only — ignores baseline)
    added_gen_twh = (pct_to_twh(added_wind, demand_twh)
                     + pct_to_twh(added_solar, demand_twh)
                     + pct_to_twh(added_offshore, demand_twh))

    # Incremental useful TWh = useful now − useful at baseline (zero added)
    if baseline_useful_frac is None:
        baseline_useful_frac = _compute_baseline_useful_frac(
            demand_arr, baseline_hourly)
    baseline_useful_twh = baseline_useful_frac * demand_twh
    incremental_useful_twh = useful_gen_twh - baseline_useful_twh

    # ── GW conversions ───────────────────────────────────────────────────
    wind_gw = pct_to_gw(added_wind, iso, 'wind', demand_twh)
    solar_gw = pct_to_gw(added_solar, iso, 'solar', demand_twh)
    offshore_gw = (
        pct_to_gw(added_offshore, iso, 'offshore_wind', demand_twh)
        if is_offshore_view else np.zeros(N)
    )

    # ── Cost model (unchanged) ───────────────────────────────────────────
    def _cost(level_key):
        c = (cost_per_pct('solar', iso, demand_twh, level_key, 'Medium') * added_solar
             + cost_per_pct('wind', iso, demand_twh, level_key, 'Medium') * added_wind)
        if is_offshore_view:
            c += cost_per_pct('offshore_wind', iso, demand_twh, level_key, 'Medium') * added_offshore
        return c

    cost_med = _cost('Medium')
    cost_low = _cost('Low')
    cost_high = _cost('High')

    # ── Effective LCOE: $/MWh of *useful* energy gained ──────────────────
    # This is the metric that explodes at high penetration.
    # effective_lcoe = annual_cost / (incremental useful TWh × 1e6 MWh/TWh)
    with np.errstate(divide='ignore', invalid='ignore'):
        eff_lcoe_med = np.where(
            incremental_useful_twh > 1e-9,
            cost_med / (incremental_useful_twh * 1e6),
            np.nan
        )
        eff_lcoe_low = np.where(
            incremental_useful_twh > 1e-9,
            cost_low / (incremental_useful_twh * 1e6),
            np.nan
        )
        eff_lcoe_high = np.where(
            incremental_useful_twh > 1e-9,
            cost_high / (incremental_useful_twh * 1e6),
            np.nan
        )

    return pd.DataFrame({
        'iso': iso,
        'view': view,
        'year': year,
        'demand_level': level,
        'demand_twh': demand_twh,
        'demand_growth_factor': gf,
        'phase': phase_tag,
        'baseline_clean_firm_pct': baseline['clean_firm'],
        'baseline_solar_pct': baseline['solar'],
        'baseline_wind_pct': baseline['wind'],
        'baseline_hydro_pct': baseline['hydro'],
        'baseline_offshore_pct': baseline['offshore_wind'],
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
        # ── NEW: curtailment economics ───────────────────────────────────
        'total_gen_twh': total_gen_twh,
        'useful_gen_twh': useful_gen_twh,
        'curtailed_twh': curtailed_twh,
        'curtailment_rate': curtailment_rate,
        'added_gen_twh': added_gen_twh,
        'incremental_useful_twh': incremental_useful_twh,
        # ── NEW: effective LCOE ($/MWh of matched energy gained) ─────────
        'effective_lcoe_med': eff_lcoe_med,
        'effective_lcoe_low': eff_lcoe_low,
        'effective_lcoe_high': eff_lcoe_high,
        # ── Existing: nameplate cost ─────────────────────────────────────
        'annual_cost_usd_med': cost_med,
        'annual_cost_usd_low': cost_low,
        'annual_cost_usd_high': cost_high,
    })


def _compute_baseline_useful_frac(demand_arr, baseline_hourly):
    """Useful fraction at zero added renewables — scalar."""
    return float(np.minimum(demand_arr, baseline_hourly).sum())


def sweep_scenario(
    iso,
    view,                 # 'onshore_only' or 'onshore_plus_offshore'
    year,
    level,
    demand_arr,           # (8760,) normalized to sum=1
    supply_matrix,        # (5, 8760)
):
    """Two-phase renewables-only ceiling surface for one (iso, view, year, level).

    Returns the full sweep DataFrame with curtailment economics at every
    grid point.
    """
    # Demand growth factor & absolute demand TWh
    gf = (1.0 + DEMAND_GROWTH_RATES[iso][level]) ** max(0, year - BASE_YEAR)
    demand_twh = REGIONAL_DEMAND_TWH[iso] * gf

    # Baseline clean mix_pcts — 2025 pcts divided by gf to preserve ABSOLUTE TWh.
    gm = GRID_MIX_SHARES[iso]
    baseline = {r: gm.get(r, 0.0) / gf for r in RESOURCES}
    imports_twh = EXTERNAL_CLEAN_IMPORTS_TWH.get(iso, 0.0)
    if imports_twh > 0:
        baseline['clean_firm'] += imports_twh / demand_twh * 100.0

    is_offshore_view = (view == 'onshore_plus_offshore') and (iso in OFFSHORE_ISOS)

    # Precompute folded baseline + baseline useful fraction (once per scenario)
    baseline_hourly = _fold_baseline_hourly(baseline, supply_matrix)
    baseline_useful_frac = _compute_baseline_useful_frac(demand_arr, baseline_hourly)

    # Common kwargs for all _score_grid calls in this scenario
    common = dict(
        baseline_hourly=baseline_hourly,
        baseline_useful_frac=baseline_useful_frac,
    )

    # ── Phase 1: coarse sweep ────────────────────────────────────────────
    wind_c = np.arange(0, WIND_CAP_PCT + 1, COARSE_WIND_STEP, dtype=np.float64)
    solar_c = np.arange(0, SOLAR_CAP_PCT + 1, COARSE_SOLAR_STEP, dtype=np.float64)
    offshore_c = (
        np.arange(0, OFFSHORE_CAP_PCT + 1, COARSE_OFFSHORE_STEP, dtype=np.float64)
        if is_offshore_view else None
    )
    coarse = _score_grid(iso, view, year, level, baseline, demand_twh, gf,
                         demand_arr, supply_matrix, wind_c, solar_c, offshore_c,
                         'coarse', **common)

    imax = coarse['score_pct'].idxmax()
    w_star = float(coarse.loc[imax, 'added_wind_pct'])
    s_star = float(coarse.loc[imax, 'added_solar_pct'])
    o_star = float(coarse.loc[imax, 'added_offshore_pct'])

    # ── Phase 2: fine sweep in ±window around coarse max ─────────────────
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
    fine = _score_grid(iso, view, year, level, baseline, demand_twh, gf,
                       demand_arr, supply_matrix, wind_f, solar_f, offshore_f,
                       'fine', **common)

    # Single-axis stripes at 1 pp resolution
    wind_stripe = _score_grid(
        iso, view, year, level, baseline, demand_twh, gf,
        demand_arr, supply_matrix,
        np.arange(0, WIND_CAP_PCT + 1, 1, dtype=np.float64),
        np.array([0.0]),
        np.array([0.0]) if is_offshore_view else None,
        'axis_wind', **common)
    solar_stripe = _score_grid(
        iso, view, year, level, baseline, demand_twh, gf,
        demand_arr, supply_matrix,
        np.array([0.0]),
        np.arange(0, SOLAR_CAP_PCT + 1, 1, dtype=np.float64),
        np.array([0.0]) if is_offshore_view else None,
        'axis_solar', **common)
    stripes = [wind_stripe, solar_stripe]
    if is_offshore_view:
        offshore_stripe = _score_grid(
            iso, view, year, level, baseline, demand_twh, gf,
            demand_arr, supply_matrix,
            np.array([0.0]), np.array([0.0]),
            np.arange(0, OFFSHORE_CAP_PCT + 1, 1, dtype=np.float64),
            'axis_offshore', **common)
        stripes.append(offshore_stripe)

    return pd.concat([coarse, fine] + stripes, ignore_index=True)


# -------------------------------------------------------------------------
# Cost frontier extraction
# -------------------------------------------------------------------------
def extract_cost_frontier(df, axis_tag=None):
    """Extract the cheapest path to each achievable CFE score level.

    For each 1-pp CFE bin, find the grid point with the MINIMUM cost that
    achieves at least that score.  This traces the efficient frontier —
    the curve a rational buyer would follow, adding the cheapest next
    increment of wind/solar at each step.

    If axis_tag is provided (e.g. 'axis_wind'), restricts to that single-axis
    slice.  Otherwise uses all grid points (the joint 2-D/3-D surface).

    Returns a DataFrame with one row per CFE target level, including:
      target_score_pct, min_annual_cost_med, optimal wind/solar/offshore GW,
      curtailment_rate, effective_lcoe_med,
      marginal_cost_per_pp ($/yr to gain the next 1 pp of CFE).
    """
    if axis_tag is not None:
        sub = df[df.phase == axis_tag].copy()
    else:
        sub = df.copy()

    if len(sub) == 0:
        return pd.DataFrame()

    baseline_score = float(sub['score_pct'].min())
    ceiling_score = float(sub['score_pct'].max())

    # Build target bins from baseline to ceiling
    targets = np.arange(
        np.floor(baseline_score),
        np.ceil(ceiling_score) + FRONTIER_SCORE_STEP_PP,
        FRONTIER_SCORE_STEP_PP
    )

    rows = []
    for target in targets:
        eligible = sub[sub.score_pct >= target - 0.5]  # ±0.5 pp tolerance
        if len(eligible) == 0:
            continue
        # Cheapest point that meets the target
        imin = eligible['annual_cost_usd_med'].idxmin()
        pt = eligible.loc[imin]
        rows.append({
            'target_score_pct': target,
            'achieved_score_pct': float(pt['score_pct']),
            'min_annual_cost_med': float(pt['annual_cost_usd_med']),
            'min_annual_cost_low': float(pt['annual_cost_usd_low']),
            'min_annual_cost_high': float(pt['annual_cost_usd_high']),
            'optimal_wind_gw': float(pt['added_wind_gw']),
            'optimal_solar_gw': float(pt['added_solar_gw']),
            'optimal_offshore_gw': float(pt['added_offshore_gw']),
            'optimal_wind_pct': float(pt['added_wind_pct']),
            'optimal_solar_pct': float(pt['added_solar_pct']),
            'optimal_offshore_pct': float(pt['added_offshore_pct']),
            'total_gen_twh': float(pt['total_gen_twh']),
            'useful_gen_twh': float(pt['useful_gen_twh']),
            'curtailed_twh': float(pt['curtailed_twh']),
            'curtailment_rate': float(pt['curtailment_rate']),
            'added_gen_twh': float(pt['added_gen_twh']),
            'incremental_useful_twh': float(pt['incremental_useful_twh']),
            'effective_lcoe_med': float(pt['effective_lcoe_med'])
                if not np.isnan(pt['effective_lcoe_med']) else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    frontier = pd.DataFrame(rows).sort_values('target_score_pct').reset_index(drop=True)

    # Marginal cost of CFE: d(cost)/d(score) between adjacent frontier points.
    # Units: $/yr per 1 pp of CFE score gained.
    dcost = np.diff(frontier['min_annual_cost_med'].values)
    dscore = np.diff(frontier['achieved_score_pct'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        marginal = np.where(dscore > 1e-6, dcost / dscore, np.nan)
    # Assign marginal cost to the UPPER point (cost of going from prev → this).
    frontier['marginal_cost_per_pp'] = np.concatenate([[np.nan], marginal])

    # Marginal effective LCOE: marginal_cost / (marginal useful TWh gained × 1e6)
    d_useful = np.diff(frontier['incremental_useful_twh'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        marginal_eff = np.where(
            d_useful > 1e-9,
            dcost / (d_useful * 1e6),
            np.nan
        )
    frontier['marginal_effective_lcoe_med'] = np.concatenate([[np.nan], marginal_eff])

    return frontier


def extract_all_cost_curves(df):
    """Extract cost frontiers for joint surface + each single-axis slice."""
    r = df.iloc[0]
    meta = {
        'iso': r['iso'], 'view': r['view'],
        'year': int(r['year']), 'demand_level': r['demand_level'],
        'demand_twh': float(r['demand_twh']),
    }

    curves = []

    # Joint frontier (uses coarse + fine — the full 2-D/3-D surface)
    joint = extract_cost_frontier(df[df.phase.isin(['coarse', 'fine'])])
    if len(joint):
        joint['frontier_type'] = 'joint_optimal'
        for k, v in meta.items():
            joint[k] = v
        curves.append(joint)

    # Single-axis frontiers
    for axis_tag, label in [('axis_wind', 'wind_only'),
                            ('axis_solar', 'solar_only'),
                            ('axis_offshore', 'offshore_only')]:
        ax = extract_cost_frontier(df, axis_tag=axis_tag)
        if len(ax):
            ax['frontier_type'] = label
            for k, v in meta.items():
                ax[k] = v
            curves.append(ax)

    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


# -------------------------------------------------------------------------
# Saturation elbow extraction  (unchanged from original)
# -------------------------------------------------------------------------
def saturation_elbow_single_axis(gw_arr, score_arr, threshold_pp_per_gw):
    """Return the GW at which marginal d(score)/d(gw) drops below the
    threshold for 3 consecutive steps."""
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


def joint_efficient_elbow(df, threshold_pp_per_gw):
    """Upper envelope of score vs total added GW, then elbow rule."""
    total_gw = df['added_wind_gw'] + df['added_solar_gw'] + df['added_offshore_gw']
    envelope = (
        pd.DataFrame({'gw': total_gw, 'score': df['score_pct']})
        .groupby(pd.cut(total_gw, bins=100, duplicates='drop'), observed=True)
        .agg(gw=('gw', 'mean'), score=('score', 'max'))
        .dropna()
        .sort_values('gw')
    )
    return saturation_elbow_single_axis(
        envelope['gw'].values, envelope['score'].values, threshold_pp_per_gw
    )


# -------------------------------------------------------------------------
# Summary row extraction (enriched with curtailment economics)
# -------------------------------------------------------------------------
def extract_summary_row(df):
    """Summarize one scenario's surface into a single row."""
    r = df.iloc[0]
    iso = r['iso']; view = r['view']; year = int(r['year']); level = r['demand_level']

    baseline_row = df[(df.added_wind_pct == 0) & (df.added_solar_pct == 0)
                      & (df.added_offshore_pct == 0)]
    baseline_score = float(baseline_row['score_pct'].iloc[0]) if len(baseline_row) else float('nan')

    imax = df['score_pct'].idxmax()
    top = df.loc[imax]

    # Elbows (unchanged)
    wind_only = df[df.phase == 'axis_wind'].sort_values('added_wind_gw')
    wind_gw_arr = wind_only['added_wind_gw'].values
    wind_score_arr = wind_only['score_pct'].values
    sat_wind_first = saturation_elbow_single_axis(wind_gw_arr, wind_score_arr, ELBOW_THRESHOLD_FIRST_KNEE)
    sat_wind_deep = saturation_elbow_single_axis(wind_gw_arr, wind_score_arr, ELBOW_THRESHOLD_DEEP)
    wind_only_max_score = float(wind_score_arr.max()) if len(wind_score_arr) else float('nan')

    solar_only = df[df.phase == 'axis_solar'].sort_values('added_solar_gw')
    solar_gw_arr = solar_only['added_solar_gw'].values
    solar_score_arr = solar_only['score_pct'].values
    sat_solar_first = saturation_elbow_single_axis(solar_gw_arr, solar_score_arr, ELBOW_THRESHOLD_FIRST_KNEE)
    sat_solar_deep = saturation_elbow_single_axis(solar_gw_arr, solar_score_arr, ELBOW_THRESHOLD_DEEP)
    solar_only_max_score = float(solar_score_arr.max()) if len(solar_score_arr) else float('nan')

    if view == 'onshore_plus_offshore':
        off_only = df[df.phase == 'axis_offshore'].sort_values('added_offshore_gw')
        off_gw_arr = off_only['added_offshore_gw'].values
        off_score_arr = off_only['score_pct'].values
        sat_offshore_first = saturation_elbow_single_axis(off_gw_arr, off_score_arr, ELBOW_THRESHOLD_FIRST_KNEE)
        sat_offshore_deep = saturation_elbow_single_axis(off_gw_arr, off_score_arr, ELBOW_THRESHOLD_DEEP)
    else:
        sat_offshore_first = float('nan')
        sat_offshore_deep = float('nan')

    fine_only = df[df.phase == 'fine']
    sat_joint_first = joint_efficient_elbow(fine_only, ELBOW_THRESHOLD_FIRST_KNEE)
    sat_joint_deep = joint_efficient_elbow(fine_only, ELBOW_THRESHOLD_DEEP)

    added_wind_twh = float(top['added_wind_twh'])
    added_solar_twh = float(top['added_solar_twh'])
    added_offshore_twh = float(top['added_offshore_twh'])
    total_added_twh = added_wind_twh + added_solar_twh + added_offshore_twh
    wind_share_of_added = added_wind_twh / total_added_twh if total_added_twh > 0 else float('nan')

    # ── NEW: extract cost frontier for milestone effective LCOEs ──────────
    # Report effective LCOE at key CFE milestones along the optimal frontier.
    frontier = extract_cost_frontier(df[df.phase.isin(['coarse', 'fine'])])
    milestone_lcoes = {}
    for target in [60, 70, 80, 85, 90, 95]:
        hit = frontier[frontier.target_score_pct >= target]
        if len(hit):
            row = hit.iloc[0]
            milestone_lcoes[f'eff_lcoe_at_{target}pct'] = float(row['effective_lcoe_med'])
            milestone_lcoes[f'curtailment_at_{target}pct'] = float(row['curtailment_rate'])
            milestone_lcoes[f'marginal_cost_pp_at_{target}pct'] = (
                float(row['marginal_cost_per_pp'])
                if not np.isnan(row.get('marginal_cost_per_pp', np.nan)) else np.nan
            )
        else:
            milestone_lcoes[f'eff_lcoe_at_{target}pct'] = float('nan')
            milestone_lcoes[f'curtailment_at_{target}pct'] = float('nan')
            milestone_lcoes[f'marginal_cost_pp_at_{target}pct'] = float('nan')

    result = {
        'iso': iso,
        'view': view,
        'year': year,
        'demand_level': level,
        'demand_twh': float(top['demand_twh']),
        'demand_growth_factor': float(top['demand_growth_factor']),
        'baseline_score_pct': baseline_score,
        'ceiling_score_pct': float(top['score_pct']),
        'score_lift_pp': float(top['score_pct']) - baseline_score,
        'ceiling_added_wind_gw': float(top['added_wind_gw']),
        'ceiling_added_solar_gw': float(top['added_solar_gw']),
        'ceiling_added_offshore_gw': float(top['added_offshore_gw']),
        'ceiling_added_wind_twh': added_wind_twh,
        'ceiling_added_solar_twh': added_solar_twh,
        'ceiling_added_offshore_twh': added_offshore_twh,
        'ceiling_wind_share_of_added_twh': wind_share_of_added,
        'ceiling_annual_cost_usd_med': float(top['annual_cost_usd_med']),
        'ceiling_annual_cost_usd_low': float(top['annual_cost_usd_low']),
        'ceiling_annual_cost_usd_high': float(top['annual_cost_usd_high']),
        # ── NEW: curtailment at ceiling ──────────────────────────────────
        'ceiling_total_gen_twh': float(top['total_gen_twh']),
        'ceiling_useful_gen_twh': float(top['useful_gen_twh']),
        'ceiling_curtailed_twh': float(top['curtailed_twh']),
        'ceiling_curtailment_rate': float(top['curtailment_rate']),
        'ceiling_effective_lcoe_med': float(top['effective_lcoe_med'])
            if not np.isnan(top['effective_lcoe_med']) else np.nan,
        # ── Existing elbows ──────────────────────────────────────────────
        'wind_only_max_score_pct': wind_only_max_score,
        'solar_only_max_score_pct': solar_only_max_score,
        'sat_wind_gw_first_knee': sat_wind_first,
        'sat_wind_gw_deep': sat_wind_deep,
        'sat_solar_gw_first_knee': sat_solar_first,
        'sat_solar_gw_deep': sat_solar_deep,
        'sat_offshore_gw_first_knee': sat_offshore_first,
        'sat_offshore_gw_deep': sat_offshore_deep,
        'sat_joint_total_gw_first_knee': sat_joint_first,
        'sat_joint_total_gw_deep': sat_joint_deep,
    }
    # Merge milestone LCOEs into summary row
    result.update(milestone_lcoes)
    return result


# -------------------------------------------------------------------------
# Verification (enriched)
# -------------------------------------------------------------------------
def run_verification(summary_df):
    """Run sanity checks including curtailment economics."""
    msgs = []

    # 1. Monotonicity: ceiling >= baseline
    bad = summary_df[summary_df.ceiling_score_pct < summary_df.baseline_score_pct - 1e-6]
    if len(bad) > 0:
        msgs.append(f"Monotonicity FAIL: {len(bad)} rows have ceiling < baseline")
    else:
        msgs.append("Monotonicity OK: ceiling >= baseline for all rows")

    # 2. Offshore superset
    for iso in OFFSHORE_ISOS:
        for yr in YEARS:
            for lvl in LEVELS:
                a = summary_df[(summary_df.iso == iso) & (summary_df.year == yr)
                               & (summary_df.demand_level == lvl)
                               & (summary_df.view == 'onshore_only')]
                b = summary_df[(summary_df.iso == iso) & (summary_df.year == yr)
                               & (summary_df.demand_level == lvl)
                               & (summary_df.view == 'onshore_plus_offshore')]
                if len(a) and len(b):
                    if b['ceiling_score_pct'].iloc[0] < a['ceiling_score_pct'].iloc[0] - 1e-6:
                        msgs.append(
                            f"Offshore-superset FAIL: {iso} {yr} {lvl}")
    msgs.append("Offshore-superset check complete")

    # 3. Baseline score weakly decreasing with year
    for iso in ISOS:
        for lvl in LEVELS:
            sub = summary_df[(summary_df.iso == iso) & (summary_df.demand_level == lvl)
                             & (summary_df.view == 'onshore_only')].sort_values('year')
            if len(sub) >= 2:
                s = sub['baseline_score_pct'].values
                if np.any(np.diff(s) > 1e-3):
                    msgs.append(f"Baseline-monotone FAIL: {iso} {lvl}")
    msgs.append("Baseline-monotone check complete")

    # 4. Cost spot-check (ERCOT 2030 Med onshore)
    ref = summary_df[(summary_df.iso == 'ERCOT') & (summary_df.year == 2030)
                     & (summary_df.demand_level == 'Medium')
                     & (summary_df.view == 'onshore_only')]
    if len(ref):
        r = ref.iloc[0]
        exp = (r.ceiling_added_wind_twh * 1e6 * (LCOE_TABLES['wind']['Medium']['ERCOT']
                                                 + TX_TABLES['wind']['Medium']['ERCOT'])
               + r.ceiling_added_solar_twh * 1e6 * (LCOE_TABLES['solar']['Medium']['ERCOT']
                                                    + TX_TABLES['solar']['Medium']['ERCOT']))
        diff = abs(r.ceiling_annual_cost_usd_med - exp) / max(exp, 1.0)
        if diff > 1e-6:
            msgs.append(f"Cost spot-check FAIL: ERCOT 2030 Med")
        else:
            msgs.append(f"Cost spot-check OK (ERCOT 2030 Med)")

    # ── NEW: curtailment sanity checks ───────────────────────────────────

    # 5. Curtailment at baseline should be near zero (clean generation ≤ demand
    #    in most hours for most ISOs at 2025)
    base_2025 = summary_df[(summary_df.year == 2025) & (summary_df.demand_level == 'Medium')
                           & (summary_df.view == 'onshore_only')]
    # Just check that curtailment at ceiling > 0 (it must be if renewables are over-built)
    has_curtailment = base_2025['ceiling_curtailment_rate'] > 0.01
    msgs.append(f"Curtailment at ceiling > 1%: {has_curtailment.sum()}/{len(base_2025)} ISOs (2025 Med)")

    # 6. Effective LCOE at ceiling should be >> nameplate LCOE
    # (if curtailment is high, effective LCOE must be much higher than input LCOE)
    high_curt = summary_df[summary_df.ceiling_curtailment_rate > 0.30]
    if len(high_curt):
        # For points with >30% curtailment, effective LCOE should be >1.4× nameplate
        msgs.append(f"High-curtailment rows (>30%): {len(high_curt)}, "
                    f"median effective LCOE: ${high_curt.ceiling_effective_lcoe_med.median():.1f}/MWh")

    # 7. Curtailment should be non-negative everywhere
    if 'ceiling_curtailed_twh' in summary_df.columns:
        neg = summary_df[summary_df.ceiling_curtailed_twh < -1e-6]
        if len(neg):
            msgs.append(f"Negative curtailment FAIL: {len(neg)} rows")
        else:
            msgs.append("Curtailment non-negativity OK")

    print("\n=== Verification ===")
    for m in msgs:
        print(f"  {m}")


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--isos', nargs='+', default=list(ISOS),
                    help='ISOs to run (default: all 7)')
    ap.add_argument('--years', nargs='+', type=int, default=YEARS)
    ap.add_argument('--levels', nargs='+', default=LEVELS)
    ap.add_argument('--write-surfaces', action='store_true',
                    help='Write per-scenario surface parquets with curtailment columns')
    ap.add_argument('--skip-offshore-view', action='store_true',
                    help='Skip the 3-D offshore-inclusive view entirely')
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
                    df = sweep_scenario(iso, view, year, level, da, sm)
                    summary_rows.append(extract_summary_row(df))
                    total_scenarios += 1

                    # Extract cost curves (frontier + single-axis)
                    curves = extract_all_cost_curves(df)
                    if len(curves):
                        cost_curve_frames.append(curves)

                    if args.write_surfaces:
                        fn = DATA_OUT / f'sweep_surface_{iso}_{view}_{year}_{level}.parquet'
                        df.to_parquet(fn, compression='zstd', index=False)

                    dt = time.monotonic() - t_scn
                    base_row = df[(df.added_wind_pct == 0) & (df.added_solar_pct == 0)
                                  & (df.added_offshore_pct == 0)]
                    base_s = base_row['score_pct'].iloc[0] if len(base_row) else float('nan')
                    ceil_idx = df['score_pct'].idxmax()
                    ceil_s = df.loc[ceil_idx, 'score_pct']
                    ceil_curt = df.loc[ceil_idx, 'curtailment_rate']
                    ceil_eff = df.loc[ceil_idx, 'effective_lcoe_med']
                    print(f"  {iso:6} {view:22} {year} {level:6} "
                          f"pts={len(df):>6} "
                          f"base={base_s:.1f}% "
                          f"ceil={ceil_s:.1f}% "
                          f"curt={ceil_curt:.0%} "
                          f"eff_lcoe=${ceil_eff:.0f}/MWh "
                          f"t={dt:.2f}s",
                          flush=True)

    print(f"\n{total_scenarios} scenarios in {time.monotonic()-t1:.1f}s", flush=True)

    # Write enriched summary
    summary_df = pd.DataFrame(summary_rows)
    summary_out = DATA_OUT / 'ceiling_summary.parquet'
    summary_df.to_parquet(summary_out, compression='zstd', index=False)
    print(f"Wrote {summary_out} ({len(summary_df)} rows)")

    # Write cost curves
    if cost_curve_frames:
        cost_curves = pd.concat(cost_curve_frames, ignore_index=True)
        curves_out = DATA_OUT / 'cost_curves.parquet'
        cost_curves.to_parquet(curves_out, compression='zstd', index=False)
        print(f"Wrote {curves_out} ({len(cost_curves)} rows)")

    run_verification(summary_df)

    print(f"\nTotal wall time: {time.monotonic()-t0:.1f}s")


if __name__ == '__main__':
    main()
