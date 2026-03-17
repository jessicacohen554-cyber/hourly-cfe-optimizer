#!/usr/bin/env python3
"""
Constellation Dispatch — Integrated into Market Simulator
==========================================================
Adapted from CEG-IPP-Climate/constellation_dispatch.py to work directly
with market-simulator results (clean_pct, avg_lmp, gas_friction per year)
instead of reading step6 sweep_reference parquets.

Produces per-plant dispatch results with CCS retrofit analysis.
"""

import json
import os
import re
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from lmp_engine import (
    HEAT_RATES, INSTALLED_FOSSIL_MW, build_merit_order_stack,
)

H = 8760

# Unit commitment parameters
UC_MIN_GEN_PCT = 0.50
UC_MIN_UP_HRS = 4
UC_MIN_DOWN_HRS = 2

# SMARTargets reference trajectories (fraction remaining from baseline)
AT_TRAJECTORY = {2023: 1.0, 2030: 0.57, 2035: 0.35, 2040: 0.20, 2045: 0.10, 2050: 0.0}
SBTI_TRAJECTORY = {2023: 1.0, 2030: 0.50, 2035: 0.30, 2040: 0.18, 2045: 0.11, 2050: 0.09}


def load_ccs_plants_from_file(fleet_json_path=None):
    """Load CCS-eligible plants from constellation_fleet.json or ccs-proximity-data.js.

    Returns list of plant dicts with keys: orispl, name, iso, capacity_mw,
    equity_pct, baseline_co2_mmt, co2_rate_t_mwh, ccs_eligible.
    """
    # Try constellation_fleet.json first (market-simulator's preprocessed data)
    if fleet_json_path is None:
        fleet_json_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'constellation_fleet.json')

    if os.path.exists(fleet_json_path):
        with open(fleet_json_path) as f:
            fleet = json.load(f)
        # Filter to CCS-eligible plants (gas/CCGT)
        plants = []
        for p in fleet:
            if not p.get('ccs_eligible'):
                continue
            plants.append({
                'orispl': p.get('campd_id') or p['id'],
                'name': p['name'],
                'iso': p['iso'],
                'capacity_mw': p['capacity_mw'],
                'equity_pct': p['equity_pct'],
                'co2_rate_t_mwh': 0.37,  # default CCGT
                'baseline_co2_mmt': p['capacity_mw'] * 0.65 * H * 0.37 / 1e6,
                'ccs_eligible': True,
            })
        return plants

    # No fleet data found — return empty
    return []


def dispatch_plant(plant, clean_pct, avg_lmp, gas_friction, fuel_level='Medium'):
    """Compute single-plant dispatch using merit-order position + unit commitment.

    Returns dict with cf, gen_mwh, co2_tons, eq_co2_mt, status.
    """
    iso = plant['iso']
    if iso not in INSTALLED_FOSSIL_MW:
        iso = 'ERCOT'

    cap_mw = plant['capacity_mw']
    eq_share = plant['equity_pct']
    co2_rate = plant['co2_rate_t_mwh']

    # Build ISO merit-order stack to get marginal cost context
    stack, total_fossil_mw = build_merit_order_stack(iso, clean_pct, fuel_level=fuel_level)
    if not stack:
        return _zero('displaced')

    # Fossil fraction → marginal heat rate threshold
    fossil_frac = max(0.0, (100.0 - clean_pct) / 100.0)
    hr_min, hr_max = 6200.0, 11500.0
    marginal_hr = hr_min + (hr_max - hr_min) * (fossil_frac ** 0.6)

    # CCGT typical HR: 7000 Btu/kWh
    plant_hr = HEAT_RATES['gas_ccgt'] * 1000
    base_cf = 0.65

    # Dispatch zone
    if plant_hr > marginal_hr + 500:
        cf = 0.0
        status = 'displaced'
    elif plant_hr > marginal_hr:
        frac_near = 1.0 - (plant_hr - marginal_hr) / 500.0
        cf = max(0.02, base_cf * frac_near * fossil_frac / 0.5)
        status = 'marginal'
    else:
        scale = min(1.0, fossil_frac / 0.3)
        cf = max(0.05, base_cf * scale)
        status = 'dispatching'

    # Apply gas friction
    cf *= gas_friction

    # UC: enforce min gen %
    if 0 < cf < UC_MIN_GEN_PCT:
        if cf < UC_MIN_GEN_PCT * 0.5:
            cf = 0.0
            status = 'displaced'
        else:
            cf = UC_MIN_GEN_PCT

    cf = min(cf, 0.90)

    gen_mwh = cap_mw * cf * H
    co2_tons = gen_mwh * co2_rate
    eq_co2_mt = co2_tons * eq_share / 1e6
    revenue_m = gen_mwh * avg_lmp / 1e6 if cf > 0 else 0.0

    # CCS residual: 80% CF, 14% HR penalty, 95% capture
    ccs_co2_mmt = cap_mw * 0.80 * H * co2_rate * 1.14 * 0.05 * eq_share / 1e6

    return {
        'cf': round(cf, 4),
        'gen_mwh': round(gen_mwh),
        'co2_tons': round(co2_tons),
        'eq_co2_mt': round(eq_co2_mt, 4),
        'ccs_co2_mmt': round(ccs_co2_mmt, 4),
        'ccs_delta_mmt': round(eq_co2_mt - ccs_co2_mmt, 4),
        'revenue_m': round(revenue_m, 2),
        'dispatch_hours': int(cf * H) if cf > 0 else 0,
        'status': status,
    }


def _zero(status='displaced'):
    return {
        'cf': 0.0, 'gen_mwh': 0, 'co2_tons': 0,
        'eq_co2_mt': 0.0, 'ccs_co2_mmt': 0.0, 'ccs_delta_mmt': 0.0,
        'revenue_m': 0.0, 'dispatch_hours': 0,
        'status': status,
    }


def run_dispatch_from_sim_results(year_results, fleet_overrides=None):
    """Run constellation dispatch using market-sim year results.

    Args:
        year_results: list of YearResult dicts from market simulation
            (must have: year, clean_pct, avg_lmp, iso)
        fleet_overrides: optional dict {plant_id: status} from fleet config

    Returns:
        dict with structure matching EMISSIONS_DASH format:
        {
            baseline_mmt: float,
            year_labels: [str...],
            plants: [{orispl, name, iso, ...sweep_data_per_year}...],
            fan_bands: {p10: [...], p50: [...], p90: [...]},
            trajectories: {at_power_nz: {...}, sbti_15c: {...}},
            csv_rows: [{year, orispl, ...}...] for CSV export
        }
    """
    plants = load_ccs_plants_from_file()
    if not plants:
        return {'baseline_mmt': 0, 'plants': [], 'fan_bands': {}, 'csv_rows': []}

    # Apply fleet overrides: filter out retired plants, mark CCS retrofits
    if fleet_overrides:
        active_plants = []
        for p in plants:
            pid = str(p['orispl'])
            override_status = fleet_overrides.get(pid, fleet_overrides.get(str(p.get('id', '')), None))
            if override_status == 'Retired':
                continue  # Skip retired plants
            p['ccs_retrofit'] = (override_status == 'CCS Retrofit')
            active_plants.append(p)
        plants = active_plants

    # Extract years from year_results
    years = sorted(set(yr.get('year', 0) for yr in year_results))
    if not years:
        return {'baseline_mmt': 0, 'plants': [], 'fan_bands': {}, 'csv_rows': []}

    # Compute baseline emissions
    baseline_mmt = sum(p['baseline_co2_mmt'] for p in plants)

    # Dispatch each plant for each year
    csv_rows = []
    plant_results = {}

    for p in plants:
        pid = p['orispl']
        plant_results[pid] = {
            'orispl': pid,
            'name': p['name'],
            'iso': p['iso'],
            'capacity_mw': p['capacity_mw'],
            'baseline_co2_mmt': p['baseline_co2_mmt'],
            'ccs_co2_mmt': 0,
            'co2_by_year': {},
        }

    for yr in year_results:
        year = yr.get('year', 0)
        clean_pct = yr.get('clean_pct', 50)
        avg_lmp = yr.get('avg_lmp', 30)

        for p in plants:
            pid = p['orispl']
            # Use plant's ISO to filter year_results if multiple ISOs
            gas_friction = 0.7  # default

            result = dispatch_plant(p, clean_pct, avg_lmp, gas_friction)

            # If plant has CCS retrofit override, use CCS residual emissions
            if p.get('ccs_retrofit'):
                result['eq_co2_mt'] = result['ccs_co2_mmt']
                result['status'] = 'ccs_retrofit'

            plant_results[pid]['co2_by_year'][year] = result['eq_co2_mt']
            plant_results[pid]['ccs_co2_mmt'] = result['ccs_co2_mmt']

            csv_rows.append({
                'year': year,
                'orispl': pid,
                'plant_name': p['name'],
                'iso': p['iso'],
                'capacity_mw': p['capacity_mw'],
                'equity_pct': p['equity_pct'],
                'capacity_factor': result['cf'],
                'generation_mwh': result['gen_mwh'],
                'co2_tons': result['co2_tons'],
                'co2_mmt': result['eq_co2_mt'],
                'ccs_residual_mmt': result['ccs_co2_mmt'],
                'ccs_delta_mmt': result['ccs_delta_mmt'],
                'revenue_mwh': round(avg_lmp * result['cf'], 2) if result['cf'] > 0 else 0,
                'fuel_cost_mwh': 0,
                'profit_mwh': 0,
                'status': result['status'],
            })

    # Build fleet-level totals per year
    fleet_co2_by_year = {}
    for year in years:
        total = sum(
            plant_results[pid]['co2_by_year'].get(year, 0)
            for pid in plant_results
        )
        fleet_co2_by_year[year] = total

    # Build per-plant arrays for frontend
    plant_list = []
    for pid, pr in plant_results.items():
        plant_list.append({
            'orispl': pid,
            'name': pr['name'],
            'iso': pr['iso'],
            'capacity_mw': pr['capacity_mw'],
            'baseline_co2_mmt': round(pr['baseline_co2_mmt'], 4),
            'ccs_co2_mmt': round(pr['ccs_co2_mmt'], 4),
            'sweep_p50': [round(pr['co2_by_year'].get(y, 0), 4) for y in years],
        })

    # Build fan bands (for single trajectory, p10=p50=p90)
    fleet_values = [fleet_co2_by_year.get(y, 0) for y in years]
    fan_bands = {
        'p10': [round(v * 0.8, 3) for v in fleet_values],  # rough estimate
        'p50': [round(v, 3) for v in fleet_values],
        'p90': [round(v * 1.2, 3) for v in fleet_values],
    }

    # Interpolate AT/SBTi trajectories for these years
    def interp_trajectory(traj_dict, base_mmt, year_list):
        traj_years = sorted(traj_dict.keys())
        result = []
        for y in year_list:
            if y in traj_dict:
                result.append(round(base_mmt * traj_dict[y], 3))
            elif y < traj_years[0]:
                result.append(round(base_mmt, 3))
            elif y > traj_years[-1]:
                result.append(round(base_mmt * traj_dict[traj_years[-1]], 3))
            else:
                # Linear interpolation
                for i in range(len(traj_years) - 1):
                    if traj_years[i] <= y <= traj_years[i + 1]:
                        frac = (y - traj_years[i]) / (traj_years[i + 1] - traj_years[i])
                        val = traj_dict[traj_years[i]] + frac * (
                            traj_dict[traj_years[i + 1]] - traj_dict[traj_years[i]])
                        result.append(round(base_mmt * val, 3))
                        break
        return result

    return {
        'baseline_mmt': round(baseline_mmt, 2),
        'year_labels': [str(y) for y in years],
        'plants': plant_list,
        'fan_bands': fan_bands,
        'trajectories': {
            'at_power_nz': {
                'values': interp_trajectory(AT_TRAJECTORY, baseline_mmt, years),
            },
            'sbti_15c': {
                'values': interp_trajectory(SBTI_TRAJECTORY, baseline_mmt, years),
            },
        },
        'csv_rows': csv_rows,
    }
