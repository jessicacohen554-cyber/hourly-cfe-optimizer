#!/usr/bin/env python3
"""Step 2.4b — Generate dashboard JS payload for the VRE-ceiling scrollytell.

Reads data/step2.4-renewables-ceiling/ceiling_summary.parquet (198 rows, emitted
by scripts/step2_4_renewables_ceiling.py) and pre-slices it into the five scene
payloads consumed by dashboard/vre_ceiling_dashboard.html. Writes the result to
dashboard/js/vre-ceiling-data.js as `window.VRECeilingData = {...};` so the page
can pick it up without any client-side filtering.

Scene payloads
--------------
scene1_headline            : baseline vs ceiling score per ISO (2025 Medium, onshore_only)
scene2_capacity_cost       : added wind GW, added solar GW, annual cost USD by ISO (same filter)
scene3_ercot_elbows        : ERCOT onshore_only 2025 Medium — saturation elbows + single-axis maxes
scene4_demand_trajectory   : per-ISO × year × demand-level ceiling and baseline trajectory
scene5_offshore_kicker     : NEISO/NYISO/PJM/CAISO onshore_only vs onshore_plus_offshore
meta                       : ISO order, offshore ISOs, year list, level list, source citation
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent

IN_PATH = REPO_DIR / 'data' / 'step2.4-renewables-ceiling' / 'ceiling_summary.parquet'
OUT_PATH = REPO_DIR / 'dashboard' / 'js' / 'vre-ceiling-data.js'

ISO_ORDER = ['ERCOT', 'PJM', 'CAISO', 'NYISO', 'NEISO', 'MISO', 'SPP']
OFFSHORE_ISOS = ['NEISO', 'NYISO', 'PJM', 'CAISO']
YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
LEVELS = ['Low', 'Medium', 'High']


def _round(x, n=3):
    if pd.isna(x):
        return None
    return round(float(x), n)


def build_scene1(df: pd.DataFrame) -> dict:
    """Baseline vs ceiling score per ISO — 2025 Medium, onshore-only view."""
    sub = df[(df.year == 2025) & (df.demand_level == 'Medium') & (df.view == 'onshore_only')]
    rows = {r.iso: r for r in sub.itertuples(index=False)}
    return {
        'isos': ISO_ORDER,
        'baseline_score_pct': [_round(rows[i].baseline_score_pct, 2) for i in ISO_ORDER],
        'ceiling_score_pct':  [_round(rows[i].ceiling_score_pct, 2) for i in ISO_ORDER],
        'score_lift_pp':      [_round(rows[i].score_lift_pp, 2) for i in ISO_ORDER],
    }


def build_scene2(df: pd.DataFrame) -> dict:
    """Added wind/solar GW + annual cost USD by ISO (2025 Medium, onshore-only)."""
    sub = df[(df.year == 2025) & (df.demand_level == 'Medium') & (df.view == 'onshore_only')]
    rows = {r.iso: r for r in sub.itertuples(index=False)}
    return {
        'isos': ISO_ORDER,
        'added_wind_gw':          [_round(rows[i].ceiling_added_wind_gw, 0) for i in ISO_ORDER],
        'added_solar_gw':         [_round(rows[i].ceiling_added_solar_gw, 0) for i in ISO_ORDER],
        'added_wind_twh':         [_round(rows[i].ceiling_added_wind_twh, 0) for i in ISO_ORDER],
        'added_solar_twh':        [_round(rows[i].ceiling_added_solar_twh, 0) for i in ISO_ORDER],
        'demand_twh':             [_round(rows[i].demand_twh, 0) for i in ISO_ORDER],
        'annual_cost_usd_low':    [_round(rows[i].ceiling_annual_cost_usd_low, 0) for i in ISO_ORDER],
        'annual_cost_usd_med':    [_round(rows[i].ceiling_annual_cost_usd_med, 0) for i in ISO_ORDER],
        'annual_cost_usd_high':   [_round(rows[i].ceiling_annual_cost_usd_high, 0) for i in ISO_ORDER],
    }


def build_scene3(df: pd.DataFrame) -> dict:
    """ERCOT onshore-only 2025 Medium — saturation envelope data for diminishing-returns chart."""
    row = df[(df.iso == 'ERCOT') & (df.year == 2025) & (df.demand_level == 'Medium')
             & (df.view == 'onshore_only')].iloc[0]
    return {
        'iso': 'ERCOT',
        'baseline_score_pct':   _round(row.baseline_score_pct, 2),
        'ceiling_score_pct':    _round(row.ceiling_score_pct, 2),
        'wind_only_max_pct':    _round(row.wind_only_max_score_pct, 2),
        'solar_only_max_pct':   _round(row.solar_only_max_score_pct, 2),
        'sat_wind_first_knee_gw':     _round(row.sat_wind_gw_first_knee, 0),
        'sat_wind_deep_gw':           _round(row.sat_wind_gw_deep, 0),
        'sat_solar_first_knee_gw':    _round(row.sat_solar_gw_first_knee, 0),
        'sat_solar_deep_gw':          _round(row.sat_solar_gw_deep, 0),
        'sat_joint_first_knee_gw':    _round(row.sat_joint_total_gw_first_knee, 0),
        'sat_joint_deep_gw':          _round(row.sat_joint_total_gw_deep, 0),
        'ceiling_added_wind_gw':      _round(row.ceiling_added_wind_gw, 0),
        'ceiling_added_solar_gw':     _round(row.ceiling_added_solar_gw, 0),
        'ceiling_joint_total_gw':     _round(row.ceiling_added_wind_gw + row.ceiling_added_solar_gw, 0),
    }


def build_scene4(df: pd.DataFrame) -> dict:
    """Demand stress: ceiling + baseline by year × level, per ISO (onshore_only)."""
    payload = {'years': YEARS, 'levels': LEVELS, 'byISO': {}}
    for iso in ISO_ORDER:
        sub = df[(df.iso == iso) & (df.view == 'onshore_only')]
        by_level = {}
        for lvl in LEVELS:
            s = sub[sub.demand_level == lvl].sort_values('year')
            by_level[lvl] = {
                'ceiling_pct':  [_round(v, 2) for v in s.ceiling_score_pct.tolist()],
                'baseline_pct': [_round(v, 2) for v in s.baseline_score_pct.tolist()],
                'demand_twh':   [_round(v, 0) for v in s.demand_twh.tolist()],
                'added_wind_gw':  [_round(v, 0) for v in s.ceiling_added_wind_gw.tolist()],
                'added_solar_gw': [_round(v, 0) for v in s.ceiling_added_solar_gw.tolist()],
            }
        payload['byISO'][iso] = by_level
    return payload


def build_scene5(df: pd.DataFrame) -> dict:
    """Offshore kicker: onshore-only vs onshore+offshore at 2025 Medium for offshore-eligible ISOs."""
    payload = {'isos': OFFSHORE_ISOS}
    for iso in OFFSHORE_ISOS:
        onsh = df[(df.iso == iso) & (df.year == 2025) & (df.demand_level == 'Medium')
                  & (df.view == 'onshore_only')].iloc[0]
        both = df[(df.iso == iso) & (df.year == 2025) & (df.demand_level == 'Medium')
                  & (df.view == 'onshore_plus_offshore')].iloc[0]
        payload[iso] = {
            'onshore_only': {
                'ceiling_score_pct':   _round(onsh.ceiling_score_pct, 2),
                'added_wind_gw':       _round(onsh.ceiling_added_wind_gw, 0),
                'added_solar_gw':      _round(onsh.ceiling_added_solar_gw, 0),
                'added_offshore_gw':   0,
                'annual_cost_usd_med': _round(onsh.ceiling_annual_cost_usd_med, 0),
            },
            'onshore_plus_offshore': {
                'ceiling_score_pct':   _round(both.ceiling_score_pct, 2),
                'added_wind_gw':       _round(both.ceiling_added_wind_gw, 0),
                'added_solar_gw':      _round(both.ceiling_added_solar_gw, 0),
                'added_offshore_gw':   _round(both.ceiling_added_offshore_gw, 0),
                'annual_cost_usd_med': _round(both.ceiling_annual_cost_usd_med, 0),
            },
            'score_lift_pp':    _round(both.ceiling_score_pct - onsh.ceiling_score_pct, 2),
            'cost_delta_usd':   _round(both.ceiling_annual_cost_usd_med - onsh.ceiling_annual_cost_usd_med, 0),
        }
    return payload


def main() -> int:
    if not IN_PATH.exists():
        print(f'ERROR: ceiling_summary.parquet not found at {IN_PATH}', file=sys.stderr)
        return 1
    df = pd.read_parquet(IN_PATH)
    print(f'Loaded {len(df)} rows from {IN_PATH.relative_to(REPO_DIR)}')

    payload = {
        'meta': {
            'isos': ISO_ORDER,
            'offshore_isos': OFFSHORE_ISOS,
            'years': YEARS,
            'levels': LEVELS,
            'source': 'scripts/step2_4_renewables_ceiling.py',
            'source_commit': '47d0b22',
            'generated_from': 'data/step2.4-renewables-ceiling/ceiling_summary.parquet',
            'notes': [
                'Existing clean generation frozen at absolute 2025 TWh (share of demand shrinks as demand grows).',
                'No storage, no clean-firm build, no hybrids — pure wind/solar ceiling.',
                'Costs are undiscounted 2025 USD, annualized as added MWh/yr × LCOE plus transmission adder.',
                'LCOE Low/Med/High columns carry sensitivity; no learning curves applied.',
            ],
        },
        'scene1_headline':         build_scene1(df),
        'scene2_capacity_cost':    build_scene2(df),
        'scene3_ercot_elbows':     build_scene3(df),
        'scene4_demand_trajectory': build_scene4(df),
        'scene5_offshore_kicker':  build_scene5(df),
    }

    os.makedirs(OUT_PATH.parent, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        f.write('// Auto-generated by scripts/step2_4b_generate_vre_ceiling_js.py\n')
        f.write('// Do not edit manually. Rerun the generator against ceiling_summary.parquet.\n\n')
        f.write('window.VRECeilingData = ')
        json.dump(payload, f, indent=2)
        f.write(';\n')

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f'Wrote {OUT_PATH.relative_to(REPO_DIR)} ({size_kb:.1f} KB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
