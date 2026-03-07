#!/usr/bin/env python3
"""Step 12: Nuclear Retirement & Stranding Analysis

Recomputes nuclear plant economics from scratch using Step 10 sweep parquets
and nuclear policy data. Models state programs and 45U as CfD-style floors
(not additive subsidies) — payments fill the gap between market revenue and
the guaranteed floor, they do NOT stack on top of already-profitable operations.

Revenue model per plant per scenario per year:
  1. Base revenue = Energy (LMP × gen) + Capacity (cap × ELCC × degraded price)
  2. PPA revenue (if contracted) = PPA price × gen (replaces energy revenue)
  3. Applicable CfD floor = max(state_policy_floor, 45U_floor) when active
  4. CfD payment = max(0, floor − base_rev_per_mwh) × gen_mwh
  5. Total revenue = base_revenue + CfD_payment
  6. Profit = total_revenue − fixed_O&M

Output:
  - dashboard/js/nuclear-retirement-data.js (pre-computed dashboard data)
  - data/step12-nuclear-retirement/nuclear_stranding_results.parquet
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Project paths ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
SWEEP_DIR = DATA / 'step10-smartargets'
POLICY_FILE = DATA / 'nuclear_policy_data.json'
FLEET_FILE = DATA / 'ipp_fleet_config.json'
OUT_DIR = DATA / 'step12-nuclear-retirement'
JS_OUT = ROOT / 'dashboard' / 'js' / 'nuclear-retirement-data.js'

# ─── Constants (from pipeline_config / step11) ─────────────────────────
NUCLEAR_FOM_KW_YR = 120      # $/kW-yr fixed O&M
NUCLEAR_ELCC = 0.95           # Effective load carrying capacity
NUCLEAR_CF = 0.90             # Baseline capacity factor

CAPACITY_MARKET_PRICES = {
    'CAISO': 75, 'ERCOT': 0, 'PJM': 120, 'NYISO': 85,
    'NEISO': 55, 'MISO': 25, 'SPP': 0,
}

CAP_DEGRADE_ALPHA = {
    'CAISO': 0.40, 'ERCOT': 0.0, 'PJM': 0.35, 'NYISO': 0.40,
    'NEISO': 0.35, 'MISO': 0.0, 'SPP': 0.0,
}

FEDERAL_45U_FLOOR_MWH = 15.0
FEDERAL_45U_EXPIRY = 2032

SIMULATION_YEARS = [2023, 2030, 2035, 2040, 2045, 2050]
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']


def load_nuclear_plants():
    """Load nuclear plants from policy data + fleet config, return list of dicts."""
    with open(POLICY_FILE) as f:
        policy_data = json.load(f)

    plants = []
    for p in policy_data['plants']:
        plants.append({
            'name': p['name'],
            'owner': p['owner'],
            'iso': p['iso'],
            'state': p['state'],
            'cap_mw': p['cap_mw'],
            'nrc_ol_expiry': p.get('nrc_ol_expiry', 2060),
            'slr_status': p.get('slr_status', 'not_applied'),
            'slr_expiry': p.get('slr_expiry'),
            'state_policy': p.get('state_policy'),
            'policy_expiry': p.get('policy_expiry'),
            'policy_floor_mwh': p.get('policy_floor_mwh', 0),
            'hyperscaler_ppa': p.get('hyperscaler_ppa'),
            'ppa_price_mwh': p.get('ppa_price_mwh'),
            'ppa_expiry': p.get('ppa_expiry'),
        })
    return plants


def load_sweep_data():
    """Load Step 10 reference sweep parquets for all ISOs."""
    frames = []
    for iso in ISOS:
        fpath = SWEEP_DIR / f'sweep_reference_{iso}.parquet'
        if fpath.exists():
            df = pd.read_parquet(fpath)
            frames.append(df)
    if not frames:
        print('ERROR: No sweep parquets found in', SWEEP_DIR)
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def compute_nuclear_economics(plants, sweep_df):
    """Vectorized computation of nuclear plant economics across all scenarios.

    Returns DataFrame with one row per (plant × scenario × year).
    """
    N = len(plants)
    scenarios = sweep_df[['scenario', 'conditions', 'demand_growth',
                          'price_sens', 'ppa_level', 'gas_friction']].drop_duplicates()
    n_scenarios = len(scenarios)
    n_years = len(SIMULATION_YEARS)

    # ─── Plant arrays (N,) ──────────────────────────────────────────
    cap_mw = np.array([p['cap_mw'] for p in plants], dtype=np.float64)
    nrc_expiry = np.array([
        p['slr_expiry'] if p['slr_expiry'] else p['nrc_ol_expiry']
        for p in plants
    ], dtype=np.float64)
    policy_expiry = np.array([
        p['policy_expiry'] if p['policy_expiry'] else 0
        for p in plants], dtype=np.float64)
    policy_floor = np.array([p['policy_floor_mwh'] for p in plants], dtype=np.float64)
    has_ppa = np.array([p['ppa_price_mwh'] is not None for p in plants], dtype=bool)
    ppa_price = np.array([p['ppa_price_mwh'] or 0 for p in plants], dtype=np.float64)
    ppa_expiry = np.array([p['ppa_expiry'] or 0 for p in plants], dtype=np.float64)
    iso_list = [p['iso'] for p in plants]
    cap_price_base = np.array([CAPACITY_MARKET_PRICES.get(iso, 0) for iso in iso_list], dtype=np.float64)
    cap_degrade = np.array([CAP_DEGRADE_ALPHA.get(iso, 0) for iso in iso_list], dtype=np.float64)
    fom_m = cap_mw * NUCLEAR_FOM_KW_YR / 1e3  # Fixed O&M in $M/yr

    # Generation at 90% CF
    gen_mwh = cap_mw * NUCLEAR_CF * 8760  # MWh/yr per plant

    # ─── Build ISO→sweep lookup for avg_lmp and clean_pct ───────────
    sweep_lookup = {}
    for _, row in sweep_df.iterrows():
        key = (row['iso'], row['scenario'], int(row['year']))
        sweep_lookup[key] = {
            'avg_lmp': row['avg_lmp'],
            'clean_pct': row['clean_pct'],
        }

    # ─── Iterate scenarios × years (vectorized across plants) ───────
    results = []
    scenario_list = scenarios.to_dict('records')

    for s_idx, scen in enumerate(scenario_list):
        scen_id = scen['scenario']
        for year in SIMULATION_YEARS:
            # Get per-plant LMP and clean_pct from their ISO
            avg_lmp = np.zeros(N, dtype=np.float64)
            clean_pct = np.zeros(N, dtype=np.float64)
            for i, iso in enumerate(iso_list):
                key = (iso, scen_id, year)
                if key in sweep_lookup:
                    avg_lmp[i] = sweep_lookup[key]['avg_lmp']
                    clean_pct[i] = sweep_lookup[key]['clean_pct']

            # ─── License check: plant retired if year > NRC expiry ──
            licensed = year <= nrc_expiry

            # ─── Base energy revenue ────────────────────────────────
            # If plant has hyperscaler PPA and PPA is active, use PPA price
            ppa_active = has_ppa & (year <= ppa_expiry)
            energy_price = np.where(ppa_active, ppa_price, avg_lmp)
            energy_rev_m = gen_mwh * energy_price / 1e6

            # ─── Capacity revenue ───────────────────────────────────
            cap_price_eff = cap_price_base * np.maximum(
                0.0, 1.0 - cap_degrade * clean_pct / 100.0
            )
            capacity_rev_m = cap_mw * NUCLEAR_ELCC * cap_price_eff / 1e3

            # ─── Base revenue per MWh (before CfD) ──────────────────
            base_rev_m = energy_rev_m + capacity_rev_m
            base_rev_per_mwh = np.where(gen_mwh > 0, base_rev_m * 1e6 / gen_mwh, 0)

            # ─── CfD floor computation ──────────────────────────────
            # Applicable floor = max(state_policy_floor if active, 45U if active)
            # State policy floor: active if year <= policy_expiry
            state_floor_active = (policy_expiry > 0) & (year <= policy_expiry)
            effective_state_floor = np.where(state_floor_active, policy_floor, 0)

            # 45U floor: active if year <= 2032
            federal_floor = FEDERAL_45U_FLOOR_MWH if year <= FEDERAL_45U_EXPIRY else 0.0

            # Take the higher floor (they don't stack)
            applicable_floor = np.maximum(effective_state_floor, federal_floor)

            # CfD payment = max(0, floor - base_rev_per_mwh) × gen_mwh
            cfd_payment_per_mwh = np.maximum(0, applicable_floor - base_rev_per_mwh)
            cfd_payment_m = cfd_payment_per_mwh * gen_mwh / 1e6

            # ─── Total revenue and profit ───────────────────────────
            total_rev_m = base_rev_m + cfd_payment_m
            profit_m = total_rev_m - fom_m

            # ─── Apply license mask ─────────────────────────────────
            profit_m = np.where(licensed, profit_m, np.nan)
            total_rev_m = np.where(licensed, total_rev_m, np.nan)

            # ─── Status classification ──────────────────────────────
            # stranded < -5M, marginal [-5M, 0), operating >= 0
            status = np.where(
                ~licensed, 'retired',
                np.where(profit_m < -5, 'stranded',
                np.where(profit_m < 0, 'marginal', 'operating'))
            )

            # ─── Counterfactual: what would profit be without CfD? ──
            merchant_profit_m = base_rev_m - fom_m
            merchant_profit_m = np.where(licensed, merchant_profit_m, np.nan)

            # ─── Driver tagging ─────────────────────────────────────
            # For stranded scenarios, identify primary driver
            is_stranded_merchant = licensed & (merchant_profit_m < -5)
            saved_by_cfd = licensed & (merchant_profit_m < 0) & (profit_m >= 0)

            for i in range(N):
                results.append({
                    'plant': plants[i]['name'],
                    'owner': plants[i]['owner'],
                    'iso': plants[i]['iso'],
                    'state': plants[i]['state'],
                    'cap_mw': cap_mw[i],
                    'scenario': scen_id,
                    'conditions': scen['conditions'],
                    'demand_growth': scen['demand_growth'],
                    'price_sens': scen['price_sens'],
                    'ppa_level': scen['ppa_level'],
                    'gas_friction': scen['gas_friction'],
                    'year': year,
                    'avg_lmp': avg_lmp[i],
                    'clean_pct': clean_pct[i],
                    'energy_rev_m': energy_rev_m[i] if licensed[i] else np.nan,
                    'capacity_rev_m': capacity_rev_m[i] if licensed[i] else np.nan,
                    'base_rev_per_mwh': base_rev_per_mwh[i] if licensed[i] else np.nan,
                    'applicable_floor_mwh': applicable_floor[i],
                    'cfd_payment_m': cfd_payment_m[i] if licensed[i] else np.nan,
                    'total_rev_m': total_rev_m[i] if licensed[i] else np.nan,
                    'fixed_om_m': fom_m[i],
                    'profit_m': profit_m[i] if licensed[i] else np.nan,
                    'merchant_profit_m': merchant_profit_m[i] if licensed[i] else np.nan,
                    'status': status[i],
                    'ppa_active': bool(ppa_active[i]),
                    'cfd_active': bool(applicable_floor[i] > 0),
                    'saved_by_cfd': bool(saved_by_cfd[i]),
                    'stranded_without_cfd': bool(is_stranded_merchant[i]),
                })

        if (s_idx + 1) % 50 == 0:
            print(f'  Processed {s_idx + 1}/{n_scenarios} scenarios...')

    return pd.DataFrame(results)


def compute_stranding_probabilities(df):
    """Compute per-plant per-year stranding probability across 270 scenarios."""
    # For licensed plants only
    licensed = df[df['status'] != 'retired'].copy()

    # With CfD support
    strand_with_cfd = licensed.groupby(['plant', 'year']).apply(
        lambda g: (g['status'] == 'stranded').mean()
    ).reset_index(name='strand_prob_with_cfd')

    # Without CfD (pure merchant)
    strand_merchant = licensed.groupby(['plant', 'year']).apply(
        lambda g: (g['merchant_profit_m'] < -5).mean()
    ).reset_index(name='strand_prob_merchant')

    # Saved by CfD probability
    saved = licensed.groupby(['plant', 'year']).apply(
        lambda g: g['saved_by_cfd'].mean()
    ).reset_index(name='saved_by_cfd_prob')

    probs = strand_with_cfd.merge(strand_merchant, on=['plant', 'year'])
    probs = probs.merge(saved, on=['plant', 'year'])
    return probs


def compute_iso_aggregates(df):
    """Compute ISO-level nuclear MW at risk by year."""
    licensed = df[df['status'] != 'retired'].copy()

    iso_agg = licensed.groupby(['iso', 'year', 'scenario']).agg({
        'cap_mw': 'sum',
        'profit_m': 'sum',
    }).reset_index()

    # Stranded MW per ISO per year
    stranded = df[df['status'] == 'stranded'].groupby(
        ['iso', 'year', 'scenario']
    )['cap_mw'].sum().reset_index(name='stranded_mw')

    iso_agg = iso_agg.merge(stranded, on=['iso', 'year', 'scenario'], how='left')
    iso_agg['stranded_mw'] = iso_agg['stranded_mw'].fillna(0)

    # P10/P50/P90 bands
    bands = iso_agg.groupby(['iso', 'year']).agg(
        total_mw=('cap_mw', 'first'),
        stranded_p10=('stranded_mw', lambda x: np.percentile(x, 10)),
        stranded_p50=('stranded_mw', lambda x: np.percentile(x, 50)),
        stranded_p90=('stranded_mw', lambda x: np.percentile(x, 90)),
        profit_p10=('profit_m', lambda x: np.percentile(x, 10)),
        profit_p50=('profit_m', lambda x: np.percentile(x, 50)),
        profit_p90=('profit_m', lambda x: np.percentile(x, 90)),
    ).reset_index()

    return bands


def compute_driver_decomposition(df):
    """For stranded-without-CfD scenarios, tag primary driver."""
    stranded = df[(df['stranded_without_cfd'] == True) & (df['status'] != 'retired')].copy()
    if stranded.empty:
        return pd.DataFrame()

    # Driver tagging logic
    drivers = []
    for _, row in stranded.iterrows():
        yr = row['year']
        d = {
            'plant': row['plant'],
            'year': yr,
            'scenario': row['scenario'],
        }
        # 45U expiry: post-2032 and plant would have been profitable with 45U
        if yr > FEDERAL_45U_EXPIRY:
            hypothetical_with_45u = row['merchant_profit_m'] + (row['cap_mw'] * NUCLEAR_CF * 8760 * FEDERAL_45U_FLOOR_MWH / 1e6)
            d['driver_45u_expiry'] = hypothetical_with_45u >= 0
        else:
            d['driver_45u_expiry'] = False

        # State policy expiry
        d['driver_state_policy_expiry'] = bool(row.get('cfd_active', False) == False and row.get('applicable_floor_mwh', 0) == 0)

        # Low LMP
        d['driver_low_lmp'] = row['avg_lmp'] < 30  # Below breakeven

        # Capacity degradation
        d['driver_cap_degradation'] = row['clean_pct'] > 60

        drivers.append(d)

    return pd.DataFrame(drivers)


def build_revenue_waterfall(df, plants):
    """Build revenue waterfall data for each plant at key years."""
    waterfalls = {}
    for p in plants:
        pname = p['name']
        pdf = df[(df['plant'] == pname) & (df['status'] != 'retired')]
        if pdf.empty:
            continue

        wf = {}
        for year in [2030, 2035, 2040, 2050]:
            ydf = pdf[pdf['year'] == year]
            if ydf.empty:
                continue
            wf[str(year)] = {
                'energy_rev_p50': round(float(ydf['energy_rev_m'].median()), 1),
                'capacity_rev_p50': round(float(ydf['capacity_rev_m'].median()), 1),
                'cfd_payment_p50': round(float(ydf['cfd_payment_m'].median()), 1),
                'fixed_om': round(float(ydf['fixed_om_m'].iloc[0]), 1),
                'profit_p50': round(float(ydf['profit_m'].median()), 1),
                'profit_p10': round(float(ydf['profit_m'].quantile(0.10)), 1),
                'profit_p90': round(float(ydf['profit_m'].quantile(0.90)), 1),
            }
        waterfalls[pname] = wf
    return waterfalls


def generate_js_output(plants, probs, iso_bands, waterfalls, driver_df, full_df):
    """Generate pre-computed JS data for dashboard page."""
    # Plant metadata
    plant_meta = []
    for p in plants:
        plant_meta.append({
            'name': p['name'],
            'owner': p['owner'],
            'iso': p['iso'],
            'state': p['state'],
            'cap_mw': p['cap_mw'],
            'slr_status': p['slr_status'],
            'slr_expiry': p.get('slr_expiry'),
            'nrc_expiry': p.get('slr_expiry') or p['nrc_ol_expiry'],
            'state_policy': p.get('state_policy'),
            'policy_expiry': p.get('policy_expiry'),
            'policy_floor_mwh': p.get('policy_floor_mwh', 0),
            'hyperscaler_ppa': p.get('hyperscaler_ppa'),
            'ppa_price_mwh': p.get('ppa_price_mwh'),
            'ppa_expiry': p.get('ppa_expiry'),
        })

    # Stranding probabilities
    prob_data = {}
    for _, row in probs.iterrows():
        pname = row['plant']
        if pname not in prob_data:
            prob_data[pname] = {}
        prob_data[pname][str(int(row['year']))] = {
            'with_cfd': round(float(row['strand_prob_with_cfd']), 3),
            'merchant': round(float(row['strand_prob_merchant']), 3),
            'saved_by_cfd': round(float(row['saved_by_cfd_prob']), 3),
        }

    # ISO bands
    iso_band_data = {}
    for _, row in iso_bands.iterrows():
        iso = row['iso']
        if iso not in iso_band_data:
            iso_band_data[iso] = {
                'years': [],
                'total_mw': int(row['total_mw']),
                'stranded': {'p10': [], 'p50': [], 'p90': []},
                'profit': {'p10': [], 'p50': [], 'p90': []},
            }
        iso_band_data[iso]['years'].append(int(row['year']))
        for metric in ['stranded', 'profit']:
            iso_band_data[iso][metric]['p10'].append(round(float(row[f'{metric}_p10']), 1))
            iso_band_data[iso][metric]['p50'].append(round(float(row[f'{metric}_p50']), 1))
            iso_band_data[iso][metric]['p90'].append(round(float(row[f'{metric}_p90']), 1))

    # Driver decomposition: fraction of stranded-merchant scenarios by driver by year
    driver_summary = {}
    if not driver_df.empty:
        for year in SIMULATION_YEARS:
            ydf = driver_df[driver_df['year'] == year]
            if ydf.empty:
                continue
            n = len(ydf)
            driver_summary[str(year)] = {
                '45u_expiry': round(float(ydf['driver_45u_expiry'].sum() / n), 3) if n > 0 else 0,
                'low_lmp': round(float(ydf['driver_low_lmp'].sum() / n), 3) if n > 0 else 0,
                'cap_degradation': round(float(ydf['driver_cap_degradation'].sum() / n), 3) if n > 0 else 0,
            }

    # Dimension sensitivity: how stranding probability varies by scenario dimension
    dim_sensitivity = {}
    licensed = full_df[full_df['status'] != 'retired']
    for dim in ['conditions', 'demand_growth', 'price_sens', 'gas_friction']:
        dim_data = {}
        for val in licensed[dim].unique():
            subset = licensed[licensed[dim] == val]
            by_year = {}
            for year in SIMULATION_YEARS:
                ydf = subset[subset['year'] == year]
                if ydf.empty:
                    continue
                by_year[str(year)] = {
                    'strand_rate': round(float((ydf['status'] == 'stranded').mean()), 3),
                    'merchant_strand_rate': round(float((ydf['merchant_profit_m'] < -5).mean()), 3),
                    'avg_profit_m': round(float(ydf['profit_m'].mean()), 1),
                }
            dim_data[str(val)] = by_year
        dim_sensitivity[dim] = dim_data

    # Policy timeline data
    policy_timeline = {
        'IL_CEJA_CMC': {'start': 2022, 'end': 2027, 'floor': 33.38, 'plants_mw': 11782},
        'NY_ZEC': {'start': 2017, 'end': 2029, 'floor': 17.48, 'plants_mw': 3325},
        'NJ_ZEC': {'start': 2019, 'end': 2025, 'floor': 10.00, 'plants_mw': 3467},
        '45U_PTC': {'start': 2024, 'end': 2032, 'floor': 15.00, 'plants_mw': 'all'},
    }

    # Headline stats
    total_nuc_mw = sum(p['cap_mw'] for p in plants)
    # Merchant stranding by 2035 and 2040
    prob_2035 = probs[probs['year'] == 2035]
    prob_2040 = probs[probs['year'] == 2040]
    mw_at_risk_2035 = 0
    mw_at_risk_2040 = 0
    for _, row in prob_2035.iterrows():
        plant = next((p for p in plants if p['name'] == row['plant']), None)
        if plant and row['strand_prob_merchant'] > 0.5:
            mw_at_risk_2035 += plant['cap_mw']
    for _, row in prob_2040.iterrows():
        plant = next((p for p in plants if p['name'] == row['plant']), None)
        if plant and row['strand_prob_merchant'] > 0.5:
            mw_at_risk_2040 += plant['cap_mw']

    headline = {
        'total_nuclear_mw': total_nuc_mw,
        'total_nuclear_gw': round(total_nuc_mw / 1000, 1),
        'n_plants': len(plants),
        'mw_at_risk_2035_merchant': mw_at_risk_2035,
        'mw_at_risk_2040_merchant': mw_at_risk_2040,
        'pct_at_risk_2035': round(mw_at_risk_2035 / total_nuc_mw * 100, 1) if total_nuc_mw > 0 else 0,
        'pct_at_risk_2040': round(mw_at_risk_2040 / total_nuc_mw * 100, 1) if total_nuc_mw > 0 else 0,
        '45u_annual_value_b': round(total_nuc_mw * NUCLEAR_CF * 8760 * FEDERAL_45U_FLOOR_MWH / 1e9, 2),
    }

    js_data = {
        'headline': headline,
        'plants': plant_meta,
        'stranding_probs': prob_data,
        'iso_bands': iso_band_data,
        'waterfalls': waterfalls,
        'driver_summary': driver_summary,
        'dim_sensitivity': dim_sensitivity,
        'policy_timeline': policy_timeline,
        'years': SIMULATION_YEARS,
    }

    os.makedirs(JS_OUT.parent, exist_ok=True)
    with open(JS_OUT, 'w') as f:
        f.write('// Auto-generated by step12_nuclear_retirement.py\n')
        f.write('// Nuclear fleet stranding analysis with CfD-style policy modeling\n')
        f.write(f'const NUCLEAR_RETIREMENT_DATA = {json.dumps(js_data, indent=2)};\n')

    print(f'  Wrote {JS_OUT}')
    return js_data


def main():
    print('=' * 60)
    print('Step 12: Nuclear Retirement & Stranding Analysis')
    print('=' * 60)

    # ─── Load data ──────────────────────────────────────────────────
    print('\n1. Loading nuclear plant data...')
    plants = load_nuclear_plants()
    print(f'   {len(plants)} nuclear plants loaded')
    total_mw = sum(p['cap_mw'] for p in plants)
    print(f'   Total capacity: {total_mw:,.0f} MW ({total_mw/1000:.1f} GW)')

    print('\n2. Loading Step 10 sweep data...')
    sweep_df = load_sweep_data()
    n_scenarios = sweep_df['scenario'].nunique()
    print(f'   {len(sweep_df)} rows, {n_scenarios} scenarios, {len(sweep_df["iso"].unique())} ISOs')

    # ─── Compute economics ──────────────────────────────────────────
    print('\n3. Computing nuclear plant economics (CfD model)...')
    results_df = compute_nuclear_economics(plants, sweep_df)
    print(f'   {len(results_df)} plant-scenario-year records')

    # ─── Stranding probabilities ────────────────────────────────────
    print('\n4. Computing stranding probabilities...')
    probs = compute_stranding_probabilities(results_df)
    print(f'   {len(probs)} plant-year probability records')

    # ─── ISO aggregates ─────────────────────────────────────────────
    print('\n5. Computing ISO-level aggregates...')
    iso_bands = compute_iso_aggregates(results_df)
    print(f'   {len(iso_bands)} ISO-year band records')

    # ─── Revenue waterfalls ─────────────────────────────────────────
    print('\n6. Building revenue waterfalls...')
    waterfalls = build_revenue_waterfall(results_df, plants)
    print(f'   {len(waterfalls)} plant waterfalls')

    # ─── Driver decomposition ───────────────────────────────────────
    print('\n7. Computing driver decomposition...')
    driver_df = compute_driver_decomposition(results_df)
    print(f'   {len(driver_df)} stranded-scenario driver records')

    # ─── Save outputs ───────────────────────────────────────────────
    print('\n8. Saving outputs...')
    os.makedirs(OUT_DIR, exist_ok=True)
    results_df.to_parquet(OUT_DIR / 'nuclear_stranding_results.parquet', index=False)
    print(f'   Wrote {OUT_DIR / "nuclear_stranding_results.parquet"}')

    probs.to_parquet(OUT_DIR / 'nuclear_stranding_probabilities.parquet', index=False)
    print(f'   Wrote {OUT_DIR / "nuclear_stranding_probabilities.parquet"}')

    # ─── Generate JS ────────────────────────────────────────────────
    print('\n9. Generating dashboard JS data...')
    js_data = generate_js_output(plants, probs, iso_bands, waterfalls, driver_df, results_df)

    # ─── Summary ────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    h = js_data['headline']
    print(f"  Fleet: {h['n_plants']} plants, {h['total_nuclear_gw']} GW")
    print(f"  45U annual value: ${h['45u_annual_value_b']}B/yr")
    print(f"  MW at risk (merchant, P>50%) by 2035: {h['mw_at_risk_2035_merchant']:,} MW ({h['pct_at_risk_2035']}%)")
    print(f"  MW at risk (merchant, P>50%) by 2040: {h['mw_at_risk_2040_merchant']:,} MW ({h['pct_at_risk_2040']}%)")

    # Per-plant stranding highlights
    print('\n  Per-plant merchant stranding probability (2035 / 2040):')
    for p in plants:
        probs_p = js_data['stranding_probs'].get(p['name'], {})
        p35 = probs_p.get('2035', {}).get('merchant', 0)
        p40 = probs_p.get('2040', {}).get('merchant', 0)
        policy_note = f" [{p.get('state_policy', 'merchant')}→{p.get('policy_expiry', 'n/a')}]" if p.get('state_policy') else ''
        ppa_note = f" [PPA:{p.get('hyperscaler_ppa')}→{p.get('ppa_expiry')}]" if p.get('hyperscaler_ppa') else ''
        print(f"    {p['name']:40s}  {p35*100:5.1f}% / {p40*100:5.1f}%{policy_note}{ppa_note}")

    print('\nDone.')


if __name__ == '__main__':
    main()
