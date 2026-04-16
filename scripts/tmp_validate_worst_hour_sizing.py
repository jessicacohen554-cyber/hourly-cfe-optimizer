#!/usr/bin/env python3
"""Validation — worst-hour vs ELCC gas sizing (SPEC.md §24.5 verification gate).

Picks 10 mixes across 3 ISOs (ERCOT, NYISO, CAISO), computes gas_needed under
BOTH the legacy ELCC formula (diagnostic only) and the new worst-hour
99.97th-percentile formula, and prints a diff table. Also deliberately
includes ≥2 mixes whose archetype keys are not yet in the dispatch cache,
exercising the on-demand archetype expansion path.

Expected pattern per SPEC.md §24.5:
  • VRE-heavy mixes   → worst_hour_mw  >  elcc_mw   (worst-hour sizes MORE gas)
  • Clean-firm-heavy  → worst_hour_mw  ≈  elcc_mw
  • Any mix where worst_hour_mw < elcc_mw is a HARD STOP: flagged as a bug.

Prints to stdout. Not a test-suite entry — a one-shot diagnostic to show the
user before authorizing the step-2.2 rerun.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step2_2a_cost_optimization import (  # noqa: E402
    WORST_HOUR_PERCENTILE,
    _elcc_gas_mw,
    _gas_raw_2025_worst_hour,
    _worst_hour_residual_norm,
    EXISTING_GAS_CAPACITY_MW,
    GAS_AVAILABILITY_FACTOR,
    GAS_RAW_2025_ELCC,
    _WH_STATE,
)
from pipeline_config import (  # noqa: E402
    REGIONAL_DEMAND_TWH, RESOURCE_ADEQUACY_MARGIN,
)


def _arrays_from_mix(mix, storage):
    """Build a 1-row arrays dict matching step2_2a's pricing-path schema."""
    def v(key, default=0.0):
        return np.array([float(mix.get(key, default))], dtype=np.float64)

    arrays = {
        'clean_firm':     v('clean_firm'),
        'solar':          v('solar'),
        'wind':           v('wind'),
        'hydro':          v('hydro'),
        'offshore_wind':  v('offshore_wind'),
        'geothermal':     v('geothermal'),
        'solar_batt4':    v('solar_batt4'),
        'solar_batt8':    v('solar_batt8'),
        'wind_batt4':     v('wind_batt4'),
        'wind_batt8':     v('wind_batt8'),
        'battery_dispatch_pct':  np.array(
            [float(storage.get('battery_dispatch_pct', 0.0))], dtype=np.float64),
        'battery8_dispatch_pct': np.array(
            [float(storage.get('battery8_dispatch_pct', 0.0))], dtype=np.float64),
        'ldes_dispatch_pct':     np.array(
            [float(storage.get('ldes_dispatch_pct', 0.0))], dtype=np.float64),
        'h2_dispatch_pct':       np.array(
            [float(storage.get('h2_dispatch_pct', 0.0))], dtype=np.float64),
    }
    return arrays


def _worst_hour_gas_needed(iso, mix, storage, gf=1.0):
    arrays = _arrays_from_mix(mix, storage)
    resid_norm = _worst_hour_residual_norm(iso, arrays)[0]
    demand_mwh = float(REGIONAL_DEMAND_TWH[iso]) * 1.0e6 * gf
    gap_mw = resid_norm * demand_mwh
    gaf = GAS_AVAILABILITY_FACTOR[iso]
    gas_raw = max(0.0, gap_mw * (1.0 + RESOURCE_ADEQUACY_MARGIN)) / gaf
    gas_raw_2025 = _gas_raw_2025_worst_hour(iso)
    return max(0.0, EXISTING_GAS_CAPACITY_MW[iso] + (gas_raw - gas_raw_2025))


def _elcc_gas_needed(iso, mix, storage, gf=1.0):
    arrays = _arrays_from_mix(mix, storage)
    return float(_elcc_gas_mw(iso, arrays, float(REGIONAL_DEMAND_TWH[iso]), float(gf))[0])


# Test mix library — chosen for spread across archetypes and to hit
# characterization categories: VRE-heavy, storage-heavy, clean-firm-heavy.
# ERCOT gets summer-peak + wind-heavy; NYISO gets winter-peak + offshore;
# CAISO gets summer-peak + geothermal-aware.
TEST_MIXES = [
    # ---- ERCOT ----------------------------------------------------------
    ('ERCOT', 'VRE-heavy (solar+wind+Li)',
     {'clean_firm': 0, 'solar': 45, 'wind': 45, 'ccs_ccgt': 0, 'hydro': 0},
     {'battery_dispatch_pct': 5, 'battery8_dispatch_pct': 0,
      'ldes_dispatch_pct': 0, 'h2_dispatch_pct': 0}),
    ('ERCOT', 'VRE-only no storage (STRESS)',
     {'clean_firm': 0, 'solar': 40, 'wind': 55, 'ccs_ccgt': 0, 'hydro': 0},
     {'battery_dispatch_pct': 0, 'battery8_dispatch_pct': 0,
      'ldes_dispatch_pct': 0, 'h2_dispatch_pct': 0}),
    ('ERCOT', 'Clean-firm heavy',
     {'clean_firm': 80, 'solar': 10, 'wind': 5, 'ccs_ccgt': 0, 'hydro': 0},
     {'battery_dispatch_pct': 0, 'battery8_dispatch_pct': 0,
      'ldes_dispatch_pct': 0, 'h2_dispatch_pct': 0}),
    # ---- NYISO (winter peak) --------------------------------------------
    ('NYISO', 'Solar-heavy winter (winter peak STRESS)',
     {'clean_firm': 5, 'solar': 60, 'wind': 15, 'offshore_wind': 0,
      'ccs_ccgt': 0, 'hydro': 15},
     {'battery_dispatch_pct': 3, 'battery8_dispatch_pct': 0,
      'ldes_dispatch_pct': 0, 'h2_dispatch_pct': 0}),
    ('NYISO', 'Offshore+onshore VRE (winter peak)',
     {'clean_firm': 0, 'solar': 10, 'wind': 25, 'offshore_wind': 40,
      'ccs_ccgt': 0, 'hydro': 15},
     {'battery_dispatch_pct': 5, 'battery8_dispatch_pct': 3,
      'ldes_dispatch_pct': 2, 'h2_dispatch_pct': 0}),
    ('NYISO', 'Clean-firm heavy (nuclear-ish)',
     {'clean_firm': 60, 'solar': 10, 'wind': 10, 'offshore_wind': 5,
      'ccs_ccgt': 0, 'hydro': 15},
     {'battery_dispatch_pct': 0, 'battery8_dispatch_pct': 0,
      'ldes_dispatch_pct': 0, 'h2_dispatch_pct': 0}),
    # ---- CAISO (summer peak, geothermal) --------------------------------
    ('CAISO', 'Solar + battery (summer peak)',
     {'clean_firm': 9, 'solar': 55, 'wind': 10, 'offshore_wind': 0,
      'geothermal': 6, 'ccs_ccgt': 0, 'hydro': 13},
     {'battery_dispatch_pct': 5, 'battery8_dispatch_pct': 0,
      'ldes_dispatch_pct': 0, 'h2_dispatch_pct': 0}),
    ('CAISO', 'Solar+Batt8 hybrid + LDES (high CFE)',
     {'clean_firm': 9, 'solar': 30, 'wind': 10, 'geothermal': 6,
      'ccs_ccgt': 0, 'hydro': 13, 'solar_batt8': 20},
     {'battery_dispatch_pct': 0, 'battery8_dispatch_pct': 0,
      'ldes_dispatch_pct': 5, 'h2_dispatch_pct': 0}),
    ('CAISO', 'Clean-firm dominant',
     {'clean_firm': 60, 'solar': 10, 'wind': 5, 'geothermal': 6,
      'ccs_ccgt': 0, 'hydro': 13},
     {'battery_dispatch_pct': 0, 'battery8_dispatch_pct': 0,
      'ldes_dispatch_pct': 0, 'h2_dispatch_pct': 0}),
    # ---- Deliberate cache-miss (exotic mix not in current cache) --------
    ('ERCOT', 'Exotic (cache-miss stress)',
     {'clean_firm': 3, 'solar': 37, 'wind': 41, 'ccs_ccgt': 2, 'hydro': 0,
      'wind_batt4': 8},
     {'battery_dispatch_pct': 4, 'battery8_dispatch_pct': 2,
      'ldes_dispatch_pct': 1, 'h2_dispatch_pct': 0}),
]


def _cache_rowcount(iso):
    state = _WH_STATE.get(iso)
    if state is None:
        return 0
    return len(state['cache'])


def main():
    print('=' * 92)
    print(f'  Worst-hour (p{WORST_HOUR_PERCENTILE}) vs ELCC gas sizing — 10 mixes × 3 ISOs')
    print('=' * 92)
    print(f'  {"ISO":<6} {"label":<40} '
          f'{"ELCC MW":>10} {"WH MW":>10} {"Δ MW":>10} {"Δ %":>8}  note')
    print('-' * 92)

    # Snapshot cache sizes before to observe expansion.
    import dispatch_utils as _du
    pre_rows = {iso: 0 for iso in ('ERCOT', 'NYISO', 'CAISO')}
    for iso in pre_rows:
        cache = _du.load_dispatch_cache(iso, require_version=_du.CACHE_VERSION) or {}
        pre_rows[iso] = len(cache)

    violations = []
    for iso, label, mix, storage in TEST_MIXES:
        elcc = _elcc_gas_needed(iso, mix, storage)
        worst = _worst_hour_gas_needed(iso, mix, storage)
        delta = worst - elcc
        pct = (delta / elcc * 100.0) if elcc > 0 else float('nan')
        note = ''
        if worst < elcc - 1.0:  # 1 MW tolerance for numerical noise
            note = 'BUG: worst-hour < ELCC'
            violations.append((iso, label, elcc, worst))
        print(f'  {iso:<6} {label:<40} '
              f'{elcc:>10.0f} {worst:>10.0f} {delta:>10.0f} '
              f'{pct:>7.1f}%  {note}')

    # Show cache growth on disk.
    print('-' * 92)
    for iso in pre_rows:
        cache_now = _du.load_dispatch_cache(iso, require_version=_du.CACHE_VERSION) or {}
        n_now = len(cache_now)
        n_pre = pre_rows[iso]
        growth = n_now - n_pre
        flag = '← expanded' if growth > 0 else ''
        print(f'  {iso:<6} dispatch cache: {n_pre} → {n_now} archetypes ({growth:+d}) {flag}')

    print('-' * 92)
    if violations:
        print(f'  STATUS: ❌ {len(violations)} violation(s) — worst-hour < ELCC. '
              'STOP. Debug archetype lookup or dispatch profile.')
        for iso, label, elcc, worst in violations:
            print(f'    - {iso} | {label} | ELCC={elcc:.0f} worst={worst:.0f}')
        return 1
    print('  STATUS: ✅ All rows pass invariant worst_hour ≥ ELCC.')
    print('=' * 92)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
