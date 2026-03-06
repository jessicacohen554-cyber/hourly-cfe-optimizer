#!/usr/bin/env python3
"""Re-export Scenario A from cached MAC queue parquets using compute_mix_cost.

Reads mac_queue_{ISO}.parquet (already computed), filters to high_firm_low_vre/Medium,
then prices each threshold winner through compute_mix_cost() — the same cost model
Scenario B uses. This avoids re-running the full MAC queue optimization.

Usage:
  python reexport_scenario_a.py              # All ISOs
  python reexport_scenario_a.py --iso PJM    # Single ISO
"""

import json
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scenario_common import (
    ISOS, SCENARIO_A, THRESHOLDS as SCENARIO_THRESHOLDS,
    BASE_DEMAND_TWH, HYDRO_CAP_TWH, SBTI_YEAR_MAP,
    GRID_MIX_SHARES, WHOLESALE_PRICES, FUEL_ADJUSTMENTS,
    compute_mix_cost, save_scenario_results,
    get_demand_growth_factor,
    _build_existing_only_entry, _existing_match_pct,
    learning_fraction as scenario_learning_fraction,
    _build_learning_overrides,
)
from pipeline_config import (
    NUCLEAR_NEWBUILD_LCOE, compute_clean_firm_tranches,
    GEOTHERMAL_LCOE, GEO_CAP_TWH,
)
from dispatch_utils import (
    load_common_data, get_supply_profiles, get_demand_profile,
    build_supply_matrix,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          'data', 'step5-post-processing', 'mac_queue')
PARQUET_DIR = OUTPUT_DIR


def reexport(isos, sensitivity='high_firm_low_vre', growth='Medium'):
    """Re-export scenario A from cached parquets using compute_mix_cost."""
    print("Loading common data (demand + generation profiles)...")
    demand_data, gen_profiles, _, _ = load_common_data()

    results_by_iso = {}

    for iso in isos:
        pq_path = os.path.join(PARQUET_DIR, f'mac_queue_{iso}.parquet')
        if not os.path.exists(pq_path):
            print(f"  {iso}: No parquet found — skipping")
            continue

        df = pd.read_parquet(pq_path)
        canonical = df[(df['price_sensitivity'] == sensitivity) &
                       (df['demand_growth'] == growth)].sort_values('threshold')

        if canonical.empty:
            print(f"  {iso}: No {sensitivity}/{growth} results — skipping")
            continue

        # Load dispatch data
        demand_norm, _ = get_demand_profile(iso, demand_data)
        supply_profiles = get_supply_profiles(iso, gen_profiles)
        supply_matrix = build_supply_matrix(supply_profiles)

        iso_data = {}
        sens_toggles = SCENARIO_A['toggles']
        existing_cf_pct = GRID_MIX_SHARES[iso].get('clean_firm', 0)
        first_cf_deployment_year = None

        # Pre-50% existing-only entries
        exist_match = _existing_match_pct(iso)
        for t_pre in [t for t in SCENARIO_THRESHOLDS if t < 50]:
            if t_pre <= exist_match:
                gf_pre = get_demand_growth_factor(iso, t_pre)
                demand_pre = BASE_DEMAND_TWH[iso] * gf_pre
                existing_twh = {
                    res: GRID_MIX_SHARES[iso].get(res, 0) / 100.0 * BASE_DEMAND_TWH[iso]
                    for res in ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']
                }
                existing_twh.update({'battery': 0, 'ldes': 0})
                iso_data[t_pre] = _build_existing_only_entry(
                    iso, t_pre, demand_pre, gf_pre, existing_twh, sens_toggles)

        # Process each threshold winner through compute_mix_cost
        for _, r in canonical.iterrows():
            t = r['threshold']
            demand_twh = r['demand_twh']
            demand_mwh = demand_twh * 1e6
            gf = demand_twh / BASE_DEMAND_TWH[iso] if BASE_DEMAND_TWH[iso] > 0 else 1.0

            cf_pct = r.get('winner_clean_firm_pct', 0)
            sol_pct = r.get('winner_solar_pct', 0)
            wnd_pct = r.get('winner_wind_pct', 0)
            osw_pct = r.get('winner_offshore_wind_pct', 0)
            hyd_pct = r.get('winner_hydro_pct', 0)
            geo_pct = r.get('winner_geothermal_pct', 0)
            match_score = r.get('winner_match_score', 0)

            bat4_pct = r.get('winner_battery_dispatch_pct', 0)
            bat8_pct = r.get('winner_battery8_dispatch_pct', 0)
            ldes_pct = r.get('winner_ldes_dispatch_pct', 0)
            h2_pct = r.get('winner_h2_dispatch_pct', 0)

            hydro_cap = HYDRO_CAP_TWH.get(iso, hyd_pct / 100.0 * demand_twh)
            hydro_twh = min(hyd_pct / 100.0 * demand_twh, hydro_cap)

            # CCS tranche split
            existing_cf_twh_base = GRID_MIX_SHARES[iso].get('clean_firm', 0) / 100.0 * BASE_DEMAND_TWH[iso]
            total_cf_twh = cf_pct / 100.0 * demand_twh
            new_cf_twh = max(0, total_cf_twh - existing_cf_twh_base)
            geo_physics_twh = geo_pct / 100.0 * demand_twh if iso == 'CAISO' else 0

            if new_cf_twh > 0:
                tranches = compute_clean_firm_tranches(
                    new_cf_twh=new_cf_twh, iso=iso,
                    firm_lev='H', ccs_lev='H', q45='1', tx_name='Medium',
                    geo_lev='H' if iso == 'CAISO' else None,
                    geo_physics_new_twh=geo_physics_twh,
                )
                ccs_twh = tranches.get('ccs_tranche_twh', 0)
            else:
                ccs_twh = 0

            resource_twh = {
                'clean_firm': total_cf_twh,
                'solar': sol_pct / 100.0 * demand_twh,
                'wind': wnd_pct / 100.0 * demand_twh,
                'offshore_wind': osw_pct / 100.0 * demand_twh,
                'ccs_ccgt': ccs_twh,
                'hydro': hydro_twh,
            }

            # Deployment-gated learning
            if first_cf_deployment_year is None and cf_pct > existing_cf_pct + 0.1:
                first_cf_deployment_year = SBTI_YEAR_MAP.get(t, 2050)

            learning_frac = scenario_learning_fraction(
                t, scenario='A', first_deployment_year=first_cf_deployment_year)

            overrides = _build_learning_overrides(iso, learning_frac) if learning_frac > 0 else {}

            # System cost via compute_mix_cost (same model as Scenario B)
            mix_array = [cf_pct, sol_pct, wnd_pct, osw_pct, 0.0, hyd_pct,
                         match_score, bat4_pct, bat8_pct, ldes_pct, h2_pct]

            cost_result = compute_mix_cost(
                mix_array, sens_toggles, iso, demand_twh,
                overrides=overrides, growth_factor=gf,
                demand_norm=demand_norm,
                supply_profiles=supply_profiles,
                supply_matrix=supply_matrix,
            )

            result = {
                # System cost from compute_mix_cost (unified with Scenario B)
                'total_cost': cost_result['total_cost'],
                'effective_cost': cost_result['effective_cost'],
                'incremental': cost_result['incremental'],
                'wholesale': cost_result['wholesale'],
                'gas_cost': cost_result['gas_cost'],
                'resource_twh': resource_twh,
                'resource_pct': cost_result['resource_pct'],
                'match_score': match_score,
                'battery_twh': cost_result['battery_twh'],
                'battery8_twh': cost_result['battery8_twh'],
                'ldes_twh': cost_result['ldes_twh'],
                'h2_twh': cost_result['h2_twh'],
                'gas_backup_mw': cost_result['gas_backup_mw'],
                'new_gas_mw': cost_result['new_gas_mw'],
                'existing_gas_used_mw': cost_result['existing_gas_used_mw'],
                'clean_peak_mw': cost_result['clean_peak_mw'],
                'net_peak_residual_mw': cost_result['clean_peak_mw'],
                'learning_fraction': round(learning_frac, 3),
                'first_cf_deployment_year': first_cf_deployment_year,
                'new_build_cost_total': cost_result['new_build_cost_total'],
                'new_gen_twh': cost_result['new_gen_twh'],
                'blended_new_lcoe': cost_result['blended_new_lcoe'],
                'demand_twh': demand_twh,
                'demand_mwh': demand_mwh,
                'mix_raw': mix_array,
                'mix_source': 'MAC-queue',
                # Step5d MAC/CO2 (authoritative, pass-through)
                'mac_this_step': float(r.get('mac_this_step', 0)),
                'co2_avoided_mt': float(r.get('co2_avoided_mt', 0)),
                'co2_after_mt': float(r.get('co2_after_mt', 0)),
                'baseline_co2_mt': float(r.get('baseline_co2_mt', 0)),
                'cumulative_new_build_cost': float(r.get('cumulative_new_build_cost', 0)),
                'cumulative_co2_avoided_mt': float(r.get('cumulative_co2_avoided_mt', 0)),
                'cumulative_mac': float(r.get('cumulative_mac', 0)),
                'target_year': int(r.get('target_year', 2050)),
                # Dispatch percentages
                'battery_dispatch_pct': bat4_pct,
                'battery8_dispatch_pct': bat8_pct,
                'ldes_dispatch_pct': ldes_pct,
                'h2_dispatch_pct': h2_pct,
            }

            iso_data[t] = result

        results_by_iso[iso] = iso_data
        t_min = min(iso_data.keys())
        t_max = max(iso_data.keys())
        print(f"  {iso}: {len(iso_data)} thresholds, "
              f"${iso_data[t_min]['effective_cost']:.0f}/MWh @ {t_min}% -> "
              f"${iso_data[t_max]['effective_cost']:.0f}/MWh @ {t_max}%")

    if results_by_iso:
        save_scenario_results(results_by_iso, 'A', list(results_by_iso.keys()),
                              out_dir=OUTPUT_DIR)
        print(f"\n  Scenario A re-export complete: {len(results_by_iso)} ISOs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Re-export Scenario A from cached parquets')
    parser.add_argument('--iso', nargs='*', default=None)
    args = parser.parse_args()

    isos = args.iso if args.iso else ISOS
    reexport(isos)
