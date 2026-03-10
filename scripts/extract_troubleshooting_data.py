#!/usr/bin/env python3
"""Extract Step 3 CO vs Step 5D MAC Queue comparison data for troubleshooting dashboard.

Pulls directly from:
  - Step 3 CO parquets (cost optimization results)
  - Step 4 annual manifests (dispatch percentages → TWh)
  - Step 5A CO2 parquets (emission data)
  - Step 5D MAC queue parquets (sequential deployment results)

Outputs: dashboard/js/troubleshooting-data.js
"""

import json
import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from pipeline_config import ISOS

# Map step5d price_sensitivity names → step3 scenario keys
# Format: {Ren}{Firm}{Batt}{LDES}_{Fuel}_{Tx}_{CCS}{45Q}_{Geo}
SENSITIVITY_TO_SCENARIO = {
    'all_low':            {'non_caiso': 'LLLL_L_L_L1_X', 'caiso': 'LLLL_L_L_L1_L'},
    'all_med':            {'non_caiso': 'MMMM_M_M_M1_X', 'caiso': 'MMMM_M_M_M1_M'},
    'all_high':           {'non_caiso': 'HHHH_H_H_H1_X', 'caiso': 'HHHH_H_H_H1_H'},
    'high_vre_low_firm':  {'non_caiso': 'HLHH_M_M_L1_X', 'caiso': 'HLHH_M_M_L1_L'},
    'high_firm_low_vre':  {'non_caiso': 'LHLL_M_M_H1_X', 'caiso': 'LHLL_M_M_H1_H'},
}

SENSITIVITY_LABELS = {
    'all_low': 'All Low',
    'all_med': 'All Medium',
    'all_high': 'All High',
    'high_vre_low_firm': 'High VRE / Low Firm',
    'high_firm_low_vre': 'High Firm / Low VRE',
}

# 2025 baseline CO2 emissions by ISO (Mt CO2) from pipeline_config
BASELINE_CO2_MT = {}


def load_baseline_co2():
    """Load baseline CO2 from step5a co2 results."""
    global BASELINE_CO2_MT
    for iso in ISOS:
        co2_path = os.path.join(REPO, 'data/step5-post-processing/co2_results', f'co2_{iso}.parquet')
        if os.path.exists(co2_path):
            df = pd.read_parquet(co2_path)
            # Use emission rate × demand for the median scenario at lowest threshold
            row = df[df['threshold'] == df['threshold'].min()].iloc[0]
            demand_mwh = row['annual_demand_mwh']
            rate = row['co2_emission_rate_tco2_mwh']
            BASELINE_CO2_MT[iso] = (demand_mwh * rate) / 1e6


def load_scenario_a_json(iso):
    """Load scenario_a JSON for enriched gas/cost data."""
    path = os.path.join(REPO, 'data/step5-post-processing/mac_queue', f'scenario_a_{iso}.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def get_scenario_key(iso, sensitivity):
    """Get step3 scenario key for a given ISO and sensitivity."""
    mapping = SENSITIVITY_TO_SCENARIO[sensitivity]
    if iso == 'CAISO':
        return mapping['caiso']
    return mapping['non_caiso']


def extract_step3_data(iso, scenario_key):
    """Extract Step 3 CO data for one ISO × scenario."""
    co_path = os.path.join(REPO, 'data/step3-cost-opt-parquets', f'step3_co_{iso}.parquet')
    df = pd.read_parquet(co_path)
    df = df[df['scenario'] == scenario_key].copy()
    df = df.sort_values('threshold')
    return df


def extract_step4_manifest(iso):
    """Load Step 4 annual manifest for mix → dispatch mapping."""
    path = os.path.join(REPO, 'data/step4-dispatch-cache', f'{iso}_annual_manifest.parquet')
    return pd.read_parquet(path)


def extract_co2_data(iso, scenario_key):
    """Extract Step 5A CO2 results for one ISO × scenario."""
    path = os.path.join(REPO, 'data/step5-post-processing/co2_results', f'co2_{iso}.parquet')
    df = pd.read_parquet(path)
    df = df[df['scenario'] == scenario_key].copy()
    df = df.sort_values('threshold')
    return df


def extract_step5d_data(iso, sensitivity):
    """Extract Step 5D MAC queue results for one ISO × sensitivity (Medium growth only)."""
    path = os.path.join(REPO, 'data/step5-post-processing/mac_queue', f'mac_queue_{iso}.parquet')
    df = pd.read_parquet(path)
    df = df[(df['price_sensitivity'] == sensitivity) & (df['demand_growth'] == 'Medium')].copy()
    df = df.sort_values('threshold')
    return df


def match_manifest_row(manifest_df, mix_cf, mix_solar, mix_wind, mix_ow, mix_ccs, mix_hydro,
                        batt_pct, batt8_pct, ldes_pct, h2_pct):
    """Find matching row in step4 manifest by mix composition + storage config."""
    for _, row in manifest_df.iterrows():
        if (row['mix_clean_firm'] == mix_cf and
            row['mix_solar'] == mix_solar and
            row['mix_wind'] == mix_wind and
            row['mix_offshore_wind'] == mix_ow and
            row['mix_ccs_ccgt'] == mix_ccs and
            row['mix_hydro'] == mix_hydro and
            abs(row['battery_capacity_pct'] - batt_pct) < 0.01 and
            abs(row['battery8_capacity_pct'] - batt8_pct) < 0.01 and
            abs(row['ldes_capacity_pct'] - ldes_pct) < 0.01 and
            abs(row['h2_capacity_pct'] - h2_pct) < 0.01):
            return row
    # Fallback: match without storage (use first match with same gen mix)
    for _, row in manifest_df.iterrows():
        if (row['mix_clean_firm'] == mix_cf and
            row['mix_solar'] == mix_solar and
            row['mix_wind'] == mix_wind and
            row['mix_offshore_wind'] == mix_ow and
            row['mix_ccs_ccgt'] == mix_ccs and
            row['mix_hydro'] == mix_hydro):
            return row
    return None


def build_step3_row(s3_row, co2_row, manifest_row, demand_twh):
    """Build comparison data for one Step 3 threshold."""
    result = {
        'threshold': float(s3_row['threshold']),
        'demand_twh': demand_twh,
        'match_score': float(s3_row['hourly_match_score']),
    }

    # Resource mix in TWh — from step3 percentages × demand
    # Step 3 mix values are % of demand
    mix_cf_pct = float(s3_row['mix_clean_firm'])
    mix_solar_pct = float(s3_row['mix_solar'])
    mix_wind_pct = float(s3_row['mix_wind'])
    mix_ow_pct = float(s3_row['mix_offshore_wind'])
    mix_ccs_pct = float(s3_row['mix_ccs_ccgt'])
    mix_hydro_pct = float(s3_row['mix_hydro'])
    mix_geo_pct = float(s3_row.get('mix_geothermal', 0))

    # Total generation TWh (procurement × demand)
    gen_factor = demand_twh / 100.0

    # Clean firm tranche breakdown (TWh) — from step3 directly
    existing_cf_twh = float(s3_row['tranche_cf_existing_twh'])
    uprate_twh = float(s3_row['tranche_uprate_twh'])
    geo_twh = float(s3_row['tranche_geo_twh'])
    nuclear_nb_twh = float(s3_row['tranche_nuclear_newbuild_twh'])
    ccs_tranche_twh = float(s3_row['tranche_ccs_tranche_twh'])
    new_cf_twh = float(s3_row['tranche_new_cf_twh'])

    # Use manifest dispatch percentages if available for actual TWh
    if manifest_row is not None:
        dispatch = {
            'nuclear_existing': existing_cf_twh,
            'nuclear_new': nuclear_nb_twh + uprate_twh,
            'geothermal': geo_twh,
            'ccs': ccs_tranche_twh,
            'solar_dispatch': float(manifest_row['solar_dispatch_pct']) * gen_factor,
            'solar_surplus': float(manifest_row['solar_surplus_pct']) * gen_factor,
            'wind_dispatch': float(manifest_row['wind_dispatch_pct']) * gen_factor,
            'wind_surplus': float(manifest_row['wind_surplus_pct']) * gen_factor,
            'offshore_wind_dispatch': float(manifest_row['offshore_wind_dispatch_pct']) * gen_factor,
            'offshore_wind_surplus': float(manifest_row['offshore_wind_surplus_pct']) * gen_factor,
            'hydro_dispatch': float(manifest_row['hydro_dispatch_pct']) * gen_factor,
            'hydro_surplus': float(manifest_row['hydro_surplus_pct']) * gen_factor,
            'clean_firm_dispatch': float(manifest_row['clean_firm_dispatch_pct']) * gen_factor,
            'clean_firm_surplus': float(manifest_row['clean_firm_surplus_pct']) * gen_factor,
            'ccs_dispatch': float(manifest_row['ccs_ccgt_dispatch_pct']) * gen_factor,
            'ccs_surplus': float(manifest_row['ccs_ccgt_surplus_pct']) * gen_factor,
            'battery_dispatch': float(manifest_row['battery_dispatch_pct']) * gen_factor,
            'battery_charge': float(manifest_row['battery_charge_pct']) * gen_factor,
            'battery8_dispatch': float(manifest_row['battery8_dispatch_pct']) * gen_factor,
            'battery8_charge': float(manifest_row['battery8_charge_pct']) * gen_factor,
            'ldes_dispatch': float(manifest_row['ldes_dispatch_pct']) * gen_factor,
            'ldes_charge': float(manifest_row['ldes_charge_pct']) * gen_factor,
            'h2_dispatch': float(manifest_row.get('h2_dispatch_pct', 0)) * gen_factor,
            'h2_charge': float(manifest_row.get('h2_charge_pct', 0)) * gen_factor,
            'total_curtailment': float(manifest_row['total_curtailment_pct']) * gen_factor,
            'gap_pct': float(manifest_row['gap_pct']),
        }
    else:
        # Fallback: use step3 mix percentages (generation capacity, not dispatch)
        dispatch = {
            'nuclear_existing': existing_cf_twh,
            'nuclear_new': nuclear_nb_twh + uprate_twh,
            'geothermal': geo_twh,
            'ccs': ccs_tranche_twh,
            'solar_dispatch': mix_solar_pct * gen_factor,
            'solar_surplus': 0,
            'wind_dispatch': mix_wind_pct * gen_factor,
            'wind_surplus': 0,
            'offshore_wind_dispatch': mix_ow_pct * gen_factor,
            'offshore_wind_surplus': 0,
            'hydro_dispatch': mix_hydro_pct * gen_factor,
            'hydro_surplus': 0,
            'clean_firm_dispatch': mix_cf_pct * gen_factor,
            'clean_firm_surplus': 0,
            'ccs_dispatch': mix_ccs_pct * gen_factor,
            'ccs_surplus': 0,
            'battery_dispatch': 0, 'battery_charge': 0,
            'battery8_dispatch': 0, 'battery8_charge': 0,
            'ldes_dispatch': 0, 'ldes_charge': 0,
            'h2_dispatch': 0, 'h2_charge': 0,
            'total_curtailment': 0,
            'gap_pct': 100 - float(s3_row['hourly_match_score']),
        }

    result['dispatch'] = dispatch

    # Mix percentages (raw from step3)
    result['mix_pct'] = {
        'clean_firm': mix_cf_pct,
        'solar': mix_solar_pct,
        'wind': mix_wind_pct,
        'offshore_wind': mix_ow_pct,
        'ccs_ccgt': mix_ccs_pct,
        'hydro': mix_hydro_pct,
        'geothermal': mix_geo_pct,
    }

    # Gas data
    result['gas'] = {
        'backup_needed_mw': float(s3_row['gas_gas_backup_needed_mw']),
        'existing_used_mw': float(s3_row['gas_existing_gas_used_mw']),
        'new_build_mw': float(s3_row['gas_new_gas_build_mw']),
        'cost_per_mwh': float(s3_row['gas_gas_cost_per_mwh']),
    }

    # Cost data
    result['cost'] = {
        'total_cost': float(s3_row['cost_total_cost']),
        'effective_cost': float(s3_row['cost_effective_cost']),
        'incremental': float(s3_row['cost_incremental']),
        'wholesale': float(s3_row['cost_wholesale']),
        'new_build_cf_twh': new_cf_twh,
    }

    # CO2 data from step5a
    if co2_row is not None:
        co2_abated = float(co2_row['co2_total_co2_abated_tons'])
        co2_rate = float(co2_row['co2_emission_rate_tco2_mwh'])
        demand_mwh = float(s3_row['annual_demand_mwh'])
        co2_total_system = co2_rate * demand_mwh  # total system emissions (tons)
        co2_after = co2_total_system - co2_abated  # remaining after clean build
        result['co2'] = {
            'abated_mt': co2_abated / 1e6,
            'system_mt': co2_total_system / 1e6,
            'after_mt': co2_after / 1e6,
            'rate_tco2_mwh': co2_rate,
        }
    else:
        result['co2'] = None

    # Storage dispatch
    result['storage'] = {
        'battery_dispatch_pct': float(s3_row['battery_dispatch_pct']),
        'battery8_dispatch_pct': float(s3_row['battery8_dispatch_pct']),
        'ldes_dispatch_pct': float(s3_row['ldes_dispatch_pct']),
        'h2_dispatch_pct': float(s3_row['h2_dispatch_pct']),
    }

    return result


def build_step5d_row(s5d_row, demand_twh, scenario_a_entry=None):
    """Build comparison data for one Step 5D threshold."""
    result = {
        'threshold': float(s5d_row['threshold']),
        'demand_twh': demand_twh,
        'match_score': float(s5d_row['winner_match_score']),
    }

    # Dispatch data — step5d stores winner_*_twh directly
    cf_twh = float(s5d_row['winner_clean_firm_twh'])
    solar_twh = float(s5d_row['winner_solar_twh'])
    wind_twh = float(s5d_row['winner_wind_twh'])
    ow_twh = float(s5d_row['winner_offshore_wind_twh'])
    hydro_twh = float(s5d_row['winner_hydro_twh'])
    geo_twh = float(s5d_row['winner_geothermal_twh'])

    # New build vs floor (existing)
    nb_cf = float(s5d_row['new_build_clean_firm_twh'])
    nb_solar = float(s5d_row['new_build_solar_twh'])
    nb_wind = float(s5d_row['new_build_wind_twh'])
    nb_ow = float(s5d_row['new_build_offshore_wind_twh'])
    nb_hydro = float(s5d_row['new_build_hydro_twh'])
    nb_geo = float(s5d_row['new_build_geothermal_twh'])

    floor_cf = float(s5d_row['floor_clean_firm_twh'])
    floor_solar = float(s5d_row['floor_solar_twh'])
    floor_wind = float(s5d_row['floor_wind_twh'])
    floor_ow = float(s5d_row['floor_offshore_wind_twh'])
    floor_hydro = float(s5d_row['floor_hydro_twh'])
    floor_geo = float(s5d_row['floor_geothermal_twh'])

    # Storage dispatch pcts from winner
    batt_pct = float(s5d_row['winner_battery_dispatch_pct'])
    batt8_pct = float(s5d_row['winner_battery8_dispatch_pct'])
    ldes_pct = float(s5d_row['winner_ldes_dispatch_pct'])
    h2_pct = float(s5d_row['winner_h2_dispatch_pct'])

    # Convert storage dispatch_pct to TWh
    gen_factor = demand_twh / 100.0

    # For step5d, we have total TWh directly for generation resources
    # Curtailment: total generation - matched demand
    total_gen = cf_twh + solar_twh + wind_twh + ow_twh + hydro_twh + geo_twh
    matched_twh = demand_twh * float(s5d_row['winner_match_score']) / 100.0

    dispatch = {
        'nuclear_existing': floor_cf,
        'nuclear_new': nb_cf,
        'geothermal': geo_twh,
        'ccs': 0,  # step5d doesn't separate CCS from clean_firm in same way
        'solar_dispatch': solar_twh,  # These are total TWh, not split dispatch/surplus
        'solar_surplus': 0,
        'wind_dispatch': wind_twh,
        'wind_surplus': 0,
        'offshore_wind_dispatch': ow_twh,
        'offshore_wind_surplus': 0,
        'hydro_dispatch': hydro_twh,
        'hydro_surplus': 0,
        'clean_firm_dispatch': cf_twh,
        'clean_firm_surplus': 0,
        'ccs_dispatch': 0,
        'ccs_surplus': 0,
        'battery_dispatch': batt_pct * gen_factor,
        'battery_charge': 0,
        'battery8_dispatch': batt8_pct * gen_factor,
        'battery8_charge': 0,
        'ldes_dispatch': ldes_pct * gen_factor,
        'ldes_charge': 0,
        'h2_dispatch': h2_pct * gen_factor,
        'h2_charge': 0,
        'total_curtailment': max(0, total_gen - matched_twh),
        'gap_pct': 100 - float(s5d_row['winner_match_score']),
    }

    result['dispatch'] = dispatch

    # Mix percentages
    result['mix_pct'] = {
        'clean_firm': float(s5d_row['winner_clean_firm_pct']),
        'solar': float(s5d_row['winner_solar_pct']),
        'wind': float(s5d_row['winner_wind_pct']),
        'offshore_wind': float(s5d_row['winner_offshore_wind_pct']),
        'ccs_ccgt': 0,
        'hydro': float(s5d_row['winner_hydro_pct']),
        'geothermal': float(s5d_row['winner_geothermal_pct']),
    }

    # Gas data — enrich from scenario_a JSON if available
    if scenario_a_entry:
        result['gas'] = {
            'backup_needed_mw': float(scenario_a_entry.get('gas_backup_mw', 0)),
            'existing_used_mw': float(scenario_a_entry.get('existing_gas_used_mw', 0)),
            'new_build_mw': float(scenario_a_entry.get('new_gas_mw', 0)),
            'cost_per_mwh': float(scenario_a_entry.get('gas_cost', 0)),
        }
    else:
        result['gas'] = {
            'backup_needed_mw': 0,
            'existing_used_mw': 0,
            'new_build_mw': 0,
            'cost_per_mwh': 0,
        }

    # Cost data — enrich from scenario_a JSON
    nb_cost = float(s5d_row['new_build_cost_total'])
    nb_twh = float(s5d_row['new_build_twh_total'])
    result['cost'] = {
        'total_cost': float(scenario_a_entry.get('total_cost', 0)) if scenario_a_entry else 0,
        'effective_cost': float(scenario_a_entry.get('effective_cost', 0)) if scenario_a_entry else 0,
        'incremental': float(scenario_a_entry.get('incremental', 0)) if scenario_a_entry else 0,
        'wholesale': float(scenario_a_entry.get('wholesale', 0)) if scenario_a_entry else 0,
        'new_build_cost_total': nb_cost,
        'new_build_twh': nb_twh,
        'new_build_lcoe': float(s5d_row['new_build_cost_per_mwh_nb']) if nb_twh > 0 else 0,
        'cumulative_cost': float(s5d_row['cumulative_new_build_cost']),
        'new_build_cf_twh': nb_cf,
        'blended_new_lcoe': float(scenario_a_entry.get('blended_new_lcoe', 0)) if scenario_a_entry else 0,
    }

    # CO2 data from step5d directly
    result['co2'] = {
        'abated_mt_step': float(s5d_row['co2_avoided_mt']),  # this step only
        'abated_mt': float(s5d_row['cumulative_co2_avoided_mt']),  # cumulative
        'system_mt': float(s5d_row.get('baseline_co2_mt', 0)),  # baseline at this step
        'after_mt': float(s5d_row['co2_after_mt']),
        'rate_tco2_mwh': 0,
    }

    # MAC data
    result['mac'] = {
        'stepwise': float(s5d_row['mac_this_step']),
        'cumulative': float(s5d_row['cumulative_mac']),
    }

    # Storage
    result['storage'] = {
        'battery_dispatch_pct': batt_pct,
        'battery8_dispatch_pct': batt8_pct,
        'ldes_dispatch_pct': ldes_pct,
        'h2_dispatch_pct': h2_pct,
    }

    # New build breakdown
    result['new_build'] = {
        'clean_firm_twh': nb_cf,
        'solar_twh': nb_solar,
        'wind_twh': nb_wind,
        'offshore_wind_twh': nb_ow,
        'hydro_twh': nb_hydro,
        'geothermal_twh': nb_geo,
    }

    return result


def extract_all():
    """Extract comparison data for all ISOs × sensitivities."""
    load_baseline_co2()

    output = {
        'metadata': {
            'sensitivities': list(SENSITIVITY_LABELS.keys()),
            'sensitivity_labels': SENSITIVITY_LABELS,
            'isos': ISOS,
            'baseline_co2_mt': BASELINE_CO2_MT,
        },
        'data': {},
    }

    for iso in ISOS:
        print(f"Processing {iso}...")
        output['data'][iso] = {}

        # Load step4 manifest once per ISO
        manifest = extract_step4_manifest(iso)
        scenario_a = load_scenario_a_json(iso)
        demand_twh = None

        for sens_key in SENSITIVITY_TO_SCENARIO:
            scenario_key = get_scenario_key(iso, sens_key)
            print(f"  {sens_key} → {scenario_key}")

            # Step 3 data
            s3 = extract_step3_data(iso, scenario_key)
            # Step 5A CO2 data
            co2 = extract_co2_data(iso, scenario_key)
            # Step 5D data
            s5d = extract_step5d_data(iso, sens_key)

            # Get demand TWh from step3
            if len(s3) > 0:
                demand_twh = float(s3.iloc[0]['annual_demand_mwh']) / 1e6  # MWh → TWh

            combo = {
                'scenario_key': scenario_key,
                'demand_twh': demand_twh,
                'step3': [],
                'step5d': [],
            }

            # Process each step3 threshold
            for _, row in s3.iterrows():
                thr = float(row['threshold'])
                # Find matching CO2 row
                co2_match = co2[co2['threshold'] == thr]
                co2_row = co2_match.iloc[0] if len(co2_match) > 0 else None

                # Find matching manifest row
                manifest_match = match_manifest_row(
                    manifest,
                    row['mix_clean_firm'], row['mix_solar'], row['mix_wind'],
                    row['mix_offshore_wind'], row['mix_ccs_ccgt'], row['mix_hydro'],
                    row['battery_dispatch_pct'], row['battery8_dispatch_pct'],
                    row['ldes_dispatch_pct'], row['h2_dispatch_pct']
                )

                s3_result = build_step3_row(row, co2_row, manifest_match, demand_twh)
                s3_result['manifest_matched'] = manifest_match is not None
                combo['step3'].append(s3_result)

            # Process each step5d threshold
            for _, row in s5d.iterrows():
                s5d_demand = float(row['demand_twh'])
                thr_key = str(int(row['threshold'])) if row['threshold'] == int(row['threshold']) else str(row['threshold'])
                # Get matching scenario_a entry (only available for all_med sensitivity)
                sa_entry = None
                if scenario_a and sens_key == 'all_med':
                    sa_entry = scenario_a.get('results', {}).get(thr_key)
                s5d_result = build_step5d_row(row, s5d_demand, sa_entry)
                combo['step5d'].append(s5d_result)

            output['data'][iso][sens_key] = combo

    return output


def main():
    data = extract_all()

    # Write as JS module
    out_path = os.path.join(REPO, 'dashboard', 'js', 'troubleshooting-data.js')
    with open(out_path, 'w') as f:
        f.write('// Auto-generated by scripts/extract_troubleshooting_data.py\n')
        f.write('// Step 3 CO vs Step 5D MAC Queue comparison data\n')
        f.write('const TROUBLESHOOTING_DATA = ')
        json.dump(data, f, indent=1, default=str)
        f.write(';\n')

    print(f"\nWrote {out_path}")
    # Print size
    size_kb = os.path.getsize(out_path) / 1024
    print(f"File size: {size_kb:.1f} KB")


if __name__ == '__main__':
    main()
