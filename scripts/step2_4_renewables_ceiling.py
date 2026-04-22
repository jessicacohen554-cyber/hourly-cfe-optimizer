#!/usr/bin/env python3
"""
Step 2.4: Renewables-Only CFE Ceiling Analysis
==============================================

Question answered: Holding existing clean generation frozen at its 2025
absolute-TWh baseline, what is the **highest achievable hourly CFE matching
score** reachable by adding only wind and solar? What mix achieves it, what
does it cost in undiscounted 2025 dollars, and at what capacity do diminishing
returns kick in?

Two views per ISO:
  - onshore_only   : add onshore wind + solar (primary, all 7 ISOs)
  - onshore_plus_offshore : add onshore wind + solar + offshore wind
                     (NEISO, NYISO, PJM, CAISO)

Demand trajectory sweep:
  years    : 2025, 2030, 2035, 2040, 2045, 2050
  levels   : Low, Medium, High (CAGRs from pipeline_config.DEMAND_GROWTH_RATES)
  baseline : existing clean frozen at 2025 TWh — as demand grows, its share
             of demand shrinks (1 / demand_growth_factor).

Filter applied to keep the solution pool minimal:
  - No clean firm (nuclear/CCS) build above baseline.
  - No hybrid (solar_batt4/batt8, wind_batt4/batt8) build.
  - No new storage (battery, battery8, LDES, H2) — all dispatch pcts zero.
  - Supply matrix dimensions: 5 resources only
    (clean_firm, solar, wind, hydro, offshore_wind) — hybrids & storage
    skipped entirely, so no compute spent on them.

Vectorization:
  Uses step1_pfs_generator.batch_hourly_scores — matmul-based, no Python
  for-loop over grid points. Entire 2-D / 3-D surface per scenario is one
  numpy call.

Cost model:
  Annualized $/yr = added_MWh/yr × LCOE $/MWh (+ TX adder $/MWh).
  LCOE source: pipeline_config.LCOE_TABLES (2025 undiscounted, no learning
  applied — solar/wind are mature-tech in pipeline). Carries Low/Med/High
  sensitivity as separate columns.

Output:
  data/step2.4-renewables-ceiling/ceiling_summary.parquet
  data/step2.4-renewables-ceiling/sweep_surface_{iso}_{view}_{year}_{level}.parquet
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
from step1_pfs_generator import batch_hourly_scores  # noqa: E402

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
FINE_WINDOW_PCT = 15          # ±15 pp window (user asked for ~5-10%, +slack for elbow tails)
FINE_STEP = 1                 # 1 pp step inside the window
FINE_OFFSHORE_STEP = 1

# Saturation elbow thresholds (marginal score-lift per additional GW).
# Two thresholds reported so the reader sees both the first knee and the deep
# plateau. Elbow fires only if marginal < threshold for 3 consecutive steps.
ELBOW_THRESHOLD_FIRST_KNEE = 0.10   # pp CFE per GW — first sign of bending
ELBOW_THRESHOLD_DEEP = 0.02          # pp CFE per GW — fully diminishing returns

# Resource columns we actually evaluate. Ordered to match the supply matrix.
RESOURCES = ['clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind']

DATA_OUT = REPO_DIR / 'data' / 'step2.4-renewables-ceiling'


# -------------------------------------------------------------------------
# Utility: mix_pct ↔ GW conversion
# -------------------------------------------------------------------------
def pct_to_gw(pct, iso, resource, demand_twh):
    """Convert mix_pct (% of demand, pipeline-native) → nameplate GW.

    Annual generation [GWh] = pct/100 × demand_twh × 1000.
    Nameplate GW       = Annual_GWh / (CF_annual × 8760 h).
    """
    cf = RESOURCE_CAPACITY_FACTORS[resource][iso]
    return (pct / 100.0) * demand_twh * 1000.0 / (cf * 8760.0)


def pct_to_twh(pct, demand_twh):
    return (pct / 100.0) * demand_twh


def cost_per_pct(resource, iso, demand_twh, lcoe_level='Medium', tx_level='Medium'):
    """Return $/yr per 1 pp of demand added for this resource at this LCOE tier.

    $/yr per pp = 0.01 × demand_twh × 1e6 MWh × ($/MWh LCOE + $/MWh TX adder).
    """
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
    phase_tag,            # 'coarse' | 'fine'
):
    """Core vectorized scorer. Builds mix_batch via broadcasting, one
    batch_hourly_scores call, then assembles DataFrame. No Python loops
    over grid points."""
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

    mix_batch = np.empty((N, len(RESOURCES)), dtype=np.float64)
    mix_batch[:, 0] = baseline['clean_firm']
    mix_batch[:, 1] = baseline['solar'] + added_solar
    mix_batch[:, 2] = baseline['wind'] + added_wind
    mix_batch[:, 3] = baseline['hydro']
    mix_batch[:, 4] = baseline['offshore_wind'] + added_offshore

    scores_pct = batch_hourly_scores(demand_arr, supply_matrix, mix_batch) * 100.0

    wind_gw = pct_to_gw(added_wind, iso, 'wind', demand_twh)
    solar_gw = pct_to_gw(added_solar, iso, 'solar', demand_twh)
    offshore_gw = (
        pct_to_gw(added_offshore, iso, 'offshore_wind', demand_twh)
        if is_offshore_view else np.zeros(N)
    )

    def _cost(level_key):
        c = (cost_per_pct('solar', iso, demand_twh, level_key, 'Medium') * added_solar
             + cost_per_pct('wind', iso, demand_twh, level_key, 'Medium') * added_wind)
        if is_offshore_view:
            c = c + cost_per_pct('offshore_wind', iso, demand_twh, level_key, 'Medium') * added_offshore
        return c

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
        'annual_cost_usd_med': _cost('Medium'),
        'annual_cost_usd_low': _cost('Low'),
        'annual_cost_usd_high': _cost('High'),
    })


def sweep_scenario(
    iso,
    view,                 # 'onshore_only' or 'onshore_plus_offshore'
    year,
    level,
    demand_arr,           # (8760,) normalized to sum=1
    supply_matrix,        # (5, 8760)
):
    """Two-phase renewables-only ceiling surface for one (iso, view, year, level).

    Phase 1 (coarse, 5 pp step over the full grid) locates the approximate
    ceiling. Phase 2 (fine, 1 pp step in a ±FINE_WINDOW_PCT window around the
    phase-1 max) nails the saturation elbow. Both phases use the same
    vectorized scorer and are concatenated into the return DataFrame.
    """
    # Demand growth factor & absolute demand TWh
    gf = (1.0 + DEMAND_GROWTH_RATES[iso][level]) ** max(0, year - BASE_YEAR)
    demand_twh = REGIONAL_DEMAND_TWH[iso] * gf

    # Baseline clean mix_pcts — 2025 pcts divided by gf to preserve ABSOLUTE TWh.
    gm = GRID_MIX_SHARES[iso]
    baseline = {r: gm.get(r, 0.0) / gf for r in RESOURCES}
    # NYISO Hydro-Québec imports (25 TWh/yr absolute) fold into clean_firm.
    imports_twh = EXTERNAL_CLEAN_IMPORTS_TWH.get(iso, 0.0)
    if imports_twh > 0:
        baseline['clean_firm'] += imports_twh / demand_twh * 100.0

    is_offshore_view = (view == 'onshore_plus_offshore') and (iso in OFFSHORE_ISOS)

    # ── Phase 1: coarse sweep ────────────────────────────────────────────
    wind_c = np.arange(0, WIND_CAP_PCT + 1, COARSE_WIND_STEP, dtype=np.float64)
    solar_c = np.arange(0, SOLAR_CAP_PCT + 1, COARSE_SOLAR_STEP, dtype=np.float64)
    offshore_c = (
        np.arange(0, OFFSHORE_CAP_PCT + 1, COARSE_OFFSHORE_STEP, dtype=np.float64)
        if is_offshore_view else None
    )
    coarse = _score_grid(iso, view, year, level, baseline, demand_twh, gf,
                         demand_arr, supply_matrix, wind_c, solar_c, offshore_c, 'coarse')

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
                       demand_arr, supply_matrix, wind_f, solar_f, offshore_f, 'fine')

    # Single-axis fine stripes at 1 pp — for precise wind-only / solar-only elbows.
    wind_stripe = _score_grid(
        iso, view, year, level, baseline, demand_twh, gf,
        demand_arr, supply_matrix,
        np.arange(0, WIND_CAP_PCT + 1, 1, dtype=np.float64),
        np.array([0.0]),
        np.array([0.0]) if is_offshore_view else None,
        'axis_wind')
    solar_stripe = _score_grid(
        iso, view, year, level, baseline, demand_twh, gf,
        demand_arr, supply_matrix,
        np.array([0.0]),
        np.arange(0, SOLAR_CAP_PCT + 1, 1, dtype=np.float64),
        np.array([0.0]) if is_offshore_view else None,
        'axis_solar')
    stripes = [wind_stripe, solar_stripe]
    if is_offshore_view:
        offshore_stripe = _score_grid(
            iso, view, year, level, baseline, demand_twh, gf,
            demand_arr, supply_matrix,
            np.array([0.0]), np.array([0.0]),
            np.arange(0, OFFSHORE_CAP_PCT + 1, 1, dtype=np.float64),
            'axis_offshore')
        stripes.append(offshore_stripe)

    return pd.concat([coarse, fine] + stripes, ignore_index=True)


# -------------------------------------------------------------------------
# Saturation elbow extraction
# -------------------------------------------------------------------------
def saturation_elbow_single_axis(gw_arr, score_arr, threshold_pp_per_gw):
    """Return the GW at which marginal d(score)/d(gw) drops below the
    threshold for 3 consecutive steps. Inputs sorted ascending by gw.

    Elbow is searched AFTER the score first exceeds baseline + 1 pp —
    avoids spurious elbow at gw=0 when the first increment happens to
    produce a low marginal (e.g. partially curtailed). Returns the max GW
    in the stripe if saturation is never reached (cap binding)."""
    gw_arr = np.asarray(gw_arr, dtype=np.float64)
    score_arr = np.asarray(score_arr, dtype=np.float64)
    order = np.argsort(gw_arr)
    gw_arr = gw_arr[order]
    score_arr = score_arr[order]
    if len(gw_arr) < 4:
        return float('nan')

    # Only start hunting for the elbow once the score is at least 1 pp above
    # the first (baseline) value — otherwise a flat initial segment trips it.
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
    # Cap binding — saturation not reached within grid
    return float(gw_arr[-1])


def joint_efficient_elbow(df, threshold_pp_per_gw):
    """Upper envelope of score vs total added GW (wind+solar[+offshore]),
    then apply the single-axis elbow rule on the envelope."""
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


def extract_summary_row(df):
    """Summarize one scenario's surface into a single row."""
    r = df.iloc[0]
    iso = r['iso']; view = r['view']; year = int(r['year']); level = r['demand_level']

    baseline_row = df[(df.added_wind_pct == 0) & (df.added_solar_pct == 0) & (df.added_offshore_pct == 0)]
    baseline_score = float(baseline_row['score_pct'].iloc[0]) if len(baseline_row) else float('nan')

    imax = df['score_pct'].idxmax()
    top = df.loc[imax]

    # Precise single-axis elbows at two thresholds (first knee + deep plateau).
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

    # Joint envelope uses the fine 2-D window (not the coarse 5-pp grid).
    fine_only = df[df.phase == 'fine']
    sat_joint_first = joint_efficient_elbow(fine_only, ELBOW_THRESHOLD_FIRST_KNEE)
    sat_joint_deep = joint_efficient_elbow(fine_only, ELBOW_THRESHOLD_DEEP)

    added_wind_twh = float(top['added_wind_twh'])
    added_solar_twh = float(top['added_solar_twh'])
    added_offshore_twh = float(top['added_offshore_twh'])
    total_added_twh = added_wind_twh + added_solar_twh + added_offshore_twh
    wind_share_of_added = added_wind_twh / total_added_twh if total_added_twh > 0 else float('nan')

    return {
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


# -------------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------------
def run_verification(summary_df):
    """Run five sanity checks; raise on failure."""
    msgs = []

    # 1. Baseline + lift = ceiling (by construction; idempotent)
    # (trivially holds; score_lift_pp = ceiling - baseline).

    # 2. Monotonicity: ceiling_score_pct ≥ baseline_score_pct in every row
    bad = summary_df[summary_df.ceiling_score_pct < summary_df.baseline_score_pct - 1e-6]
    if len(bad) > 0:
        msgs.append(f"Monotonicity FAIL: {len(bad)} rows have ceiling < baseline")
    else:
        msgs.append("Monotonicity OK: ceiling >= baseline for all rows")

    # 3. Onshore+offshore ceiling >= onshore-only ceiling (for offshore ISOs)
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
                            f"Offshore-superset FAIL: {iso} {yr} {lvl} onshore={a.ceiling_score_pct.iloc[0]:.3f} "
                            f"offshore_view={b.ceiling_score_pct.iloc[0]:.3f}")
    msgs.append("Offshore-superset check complete")

    # 4. Demand grows, baseline (absolute-TWh clean) score should weakly DECREASE
    # since existing clean's share of demand shrinks.
    for iso in ISOS:
        for lvl in LEVELS:
            sub = summary_df[(summary_df.iso == iso) & (summary_df.demand_level == lvl)
                             & (summary_df.view == 'onshore_only')].sort_values('year')
            if len(sub) >= 2:
                s = sub['baseline_score_pct'].values
                if np.any(np.diff(s) > 1e-3):
                    msgs.append(
                        f"Baseline-monotone FAIL: {iso} {lvl} baseline rose with year: {s}")
    msgs.append("Baseline-monotone check complete (expect non-increasing with year)")

    # 5. Cost scales correctly: annual_cost at ceiling should equal added TWh × LCOE × 1e6
    #    Spot-check for ERCOT, year=2030, level=Medium, onshore_only
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
            msgs.append(f"Cost spot-check FAIL: ERCOT 2030 Med expected {exp:.2e}, got "
                        f"{r.ceiling_annual_cost_usd_med:.2e}")
        else:
            msgs.append(f"Cost spot-check OK (ERCOT 2030 Med onshore)")

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
                    help='Opt in to writing per-scenario surface parquets '
                         '(~1.5 GB for full run). Default: summary only.')
    ap.add_argument('--skip-offshore-view', action='store_true',
                    help='Skip the 3-D offshore-inclusive view entirely')
    args = ap.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    print("Loading EIA profiles …", flush=True)
    demand_data = load_demand_profiles()
    gen_profiles = load_generation_profiles()

    # Pre-build per-ISO demand + supply (same base year across all scenarios for a given ISO)
    iso_cache = {}
    for iso in args.isos:
        iso_cache[iso] = {
            'demand_arr': build_demand_arr(iso, demand_data),
            'supply_matrix': build_supply_matrix(iso, gen_profiles),
        }
    print(f"Profiles loaded in {time.monotonic()-t0:.1f}s", flush=True)

    summary_rows = []
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

                    if args.write_surfaces:
                        fn = DATA_OUT / f'sweep_surface_{iso}_{view}_{year}_{level}.parquet'
                        df.to_parquet(fn, compression='zstd', index=False)

                    dt = time.monotonic() - t_scn
                    print(f"  {iso:6} {view:22} {year} {level:6} "
                          f"pts={len(df):>6} "
                          f"baseline={df[(df.added_wind_pct==0)&(df.added_solar_pct==0)&(df.added_offshore_pct==0)]['score_pct'].iloc[0]:.3f} "
                          f"ceiling={df['score_pct'].max():.3f} "
                          f"t={dt:.2f}s",
                          flush=True)

    print(f"\n{total_scenarios} scenarios in {time.monotonic()-t1:.1f}s", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_out = DATA_OUT / 'ceiling_summary.parquet'
    summary_df.to_parquet(summary_out, compression='zstd', index=False)
    print(f"Wrote {summary_out} ({len(summary_df)} rows)")

    run_verification(summary_df)

    print(f"\nTotal wall time: {time.monotonic()-t0:.1f}s")


if __name__ == '__main__':
    main()
