#!/usr/bin/env python3
"""Step 12: Nuclear Retirement & Stranding Analysis — Multi-Model Stress Test

Assesses merchant nuclear stranding risk under three independent modeling approaches:

  Model A: Step 10 Reference Sweep (270 parametric market scenarios)
    - 2 conditions × 3 demand × 5 price × 3 PPA × 3 gas friction = 270
    - Provides LMP and clean_pct trajectories through 2050

  Model B: Step 10 Net-Zero Pathways (10 structured transition scenarios)
    - AT1/AT2: Power Sector NZ (Facilitating/Challenging)
    - AT3/AT4: Economy-Wide NZ (Facilitating/Challenging)
    - QT1-QT4: Queue-driven trajectories
    - R1/R2: Reference pathways
    - Each provides ISO-level LMP + clean_pct trajectory

  Model C: Step 3A Track 1 Baseline + Capacity Adders
    - 5,832 cost scenarios per ISO per threshold
    - Wholesale price distribution → energy revenue
    - Capacity market signals from gas backup needs
    - Cross-scenario LMP/capacity price envelope

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
  - data/step12-nuclear-retirement/nuclear_stranding_probabilities.parquet
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
STEP3_DIR = DATA / 'step3-cost-opt-parquets'
POLICY_FILE = DATA / 'nuclear_policy_data.json'
FLEET_FILE = DATA / 'ipp_fleet_config.json'
OUT_DIR = DATA / 'step12-nuclear-retirement'
JS_OUT = ROOT / 'dashboard' / 'js' / 'nuclear-retirement-data.js'

# ─── Constants ─────────────────────────────────────────────────────────
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

# NZ scenario definitions
NZ_SCENARIOS = {
    'AT1': 'Power Sector NZ: Facilitating',
    'AT2': 'Power Sector NZ: Challenging',
    'AT3': 'Economy-Wide NZ: Facilitating',
    'AT4': 'Economy-Wide NZ: Challenging',
    'QT1': 'Queue Medium: Facilitating',
    'QT2': 'Queue Medium: Challenging',
    'QT3': 'Queue High: Facilitating',
    'QT4': 'Queue High: Challenging',
    'R1': 'Reference: Facilitating',
    'R2': 'Reference: Challenging',
}

# Step 3A capacity adder thresholds to sample
STEP3_THRESHOLDS = [50.0, 75.0, 90.0, 95.0, 99.0]


def load_nuclear_plants():
    """Load nuclear plants from policy data, return list of dicts."""
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


# ─── Shared economics engine (vectorized across plants) ───────────────

def compute_plant_economics_vectorized(plants, scenario_rows, model_tag):
    """Compute economics for all plants × scenario_rows.

    scenario_rows: list of dicts, each with keys:
        scenario, year, iso→{avg_lmp, clean_pct}, plus optional metadata keys

    Returns list of result dicts.
    """
    N = len(plants)
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
    fom_m = cap_mw * NUCLEAR_FOM_KW_YR / 1e3
    gen_mwh = cap_mw * NUCLEAR_CF * 8760

    results = []
    for sr in scenario_rows:
        year = sr['year']
        scen_id = sr['scenario']
        iso_data = sr['iso_data']  # dict: iso → {avg_lmp, clean_pct}

        avg_lmp = np.zeros(N, dtype=np.float64)
        clean_pct = np.zeros(N, dtype=np.float64)
        for i, iso in enumerate(iso_list):
            if iso in iso_data:
                avg_lmp[i] = iso_data[iso].get('avg_lmp', 30.0)
                clean_pct[i] = iso_data[iso].get('clean_pct', 40.0)

        licensed = year <= nrc_expiry
        ppa_active = has_ppa & (year <= ppa_expiry)
        energy_price = np.where(ppa_active, ppa_price, avg_lmp)
        energy_rev_m = gen_mwh * energy_price / 1e6

        cap_price_eff = cap_price_base * np.maximum(
            0.0, 1.0 - cap_degrade * clean_pct / 100.0
        )
        # Apply capacity adder if provided
        if 'capacity_adder' in sr:
            cap_price_eff = cap_price_eff + sr['capacity_adder']
        capacity_rev_m = cap_mw * NUCLEAR_ELCC * cap_price_eff / 1e3

        base_rev_m = energy_rev_m + capacity_rev_m
        base_rev_per_mwh = np.where(gen_mwh > 0, base_rev_m * 1e6 / gen_mwh, 0)

        state_floor_active = (policy_expiry > 0) & (year <= policy_expiry)
        effective_state_floor = np.where(state_floor_active, policy_floor, 0)
        federal_floor = FEDERAL_45U_FLOOR_MWH if year <= FEDERAL_45U_EXPIRY else 0.0
        applicable_floor = np.maximum(effective_state_floor, federal_floor)

        cfd_payment_per_mwh = np.maximum(0, applicable_floor - base_rev_per_mwh)
        cfd_payment_m = cfd_payment_per_mwh * gen_mwh / 1e6

        total_rev_m = base_rev_m + cfd_payment_m
        profit_m = total_rev_m - fom_m

        profit_m = np.where(licensed, profit_m, np.nan)
        total_rev_m = np.where(licensed, total_rev_m, np.nan)

        status = np.where(
            ~licensed, 'retired',
            np.where(profit_m < -5, 'stranded',
            np.where(profit_m < 0, 'marginal', 'operating'))
        )

        merchant_profit_m = base_rev_m - fom_m
        merchant_profit_m = np.where(licensed, merchant_profit_m, np.nan)

        is_stranded_merchant = licensed & (merchant_profit_m < -5)
        saved_by_cfd = licensed & (merchant_profit_m < 0) & (profit_m >= 0)

        for i in range(N):
            row = {
                'plant': plants[i]['name'],
                'owner': plants[i]['owner'],
                'iso': plants[i]['iso'],
                'state': plants[i]['state'],
                'cap_mw': cap_mw[i],
                'model': model_tag,
                'scenario': scen_id,
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
            }
            # Copy metadata
            for k in sr.get('meta', {}):
                row[k] = sr['meta'][k]
            results.append(row)

    return results


# ─── Model A: Step 10 Reference Sweep ─────────────────────────────────

def load_model_a_scenarios():
    """Load Step 10 reference sweep parquets → scenario_rows for economics."""
    frames = []
    for iso in ISOS:
        fpath = SWEEP_DIR / f'sweep_reference_{iso}.parquet'
        if fpath.exists():
            frames.append(pd.read_parquet(fpath))
    if not frames:
        print('  WARNING: No sweep parquets found')
        return []
    sweep_df = pd.concat(frames, ignore_index=True)

    # Build lookup: (scenario, year) → {iso → {avg_lmp, clean_pct}}
    lookup = {}
    for _, row in sweep_df.iterrows():
        key = (row['scenario'], int(row['year']))
        if key not in lookup:
            lookup[key] = {'iso_data': {}, 'meta': {
                'conditions': row['conditions'],
                'demand_growth': row['demand_growth'],
                'price_sens': row['price_sens'],
                'ppa_level': row['ppa_level'],
                'gas_friction': row['gas_friction'],
            }}
        lookup[key]['iso_data'][row['iso']] = {
            'avg_lmp': row['avg_lmp'],
            'clean_pct': row['clean_pct'],
        }

    scenario_rows = []
    for (scen_id, year), data in lookup.items():
        scenario_rows.append({
            'scenario': f'A_{scen_id}',
            'year': year,
            'iso_data': data['iso_data'],
            'meta': data['meta'],
        })

    return scenario_rows


# ─── Model B: Step 10 Net-Zero Pathways ───────────────────────────────

def load_model_b_scenarios():
    """Load Step 10 NZ scenario summaries → scenario_rows."""
    scenario_rows = []
    for tag, name in NZ_SCENARIOS.items():
        fpath = SWEEP_DIR / f'smartargets_{tag}_summary.json'
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)

        # Each scenario has results per ISO, each with a trajectory
        for year in SIMULATION_YEARS:
            iso_data = {}
            for iso in ISOS:
                if iso not in data.get('results', {}):
                    continue
                traj = data['results'][iso].get('trajectory', [])
                # Find this year or interpolate
                match = next((t for t in traj if t['year'] == year), None)
                if match:
                    iso_data[iso] = {
                        'avg_lmp': match.get('avg_lmp', 30.0),
                        'clean_pct': match.get('clean_pct', 40.0),
                    }

            if iso_data:
                # Determine NZ category
                if tag.startswith('AT') and int(tag[2]) <= 2:
                    nz_type = 'power_sector_nz'
                elif tag.startswith('AT') and int(tag[2]) >= 3:
                    nz_type = 'economy_wide_nz'
                elif tag.startswith('R'):
                    nz_type = 'reference'
                else:
                    nz_type = 'queue_driven'

                scenario_rows.append({
                    'scenario': f'B_{tag}',
                    'year': year,
                    'iso_data': iso_data,
                    'meta': {
                        'nz_scenario': tag,
                        'nz_name': name,
                        'nz_type': nz_type,
                    },
                })

    return scenario_rows


# ─── Model C: Step 3A Track 1 Baseline + Capacity Adders ──────────────

def load_model_c_scenarios():
    """Load Step 3A cost optimization → derive LMP/capacity signals for nuclear.

    Uses wholesale price as LMP proxy, and gas backup needs to derive
    capacity price pressure. Samples across thresholds to capture the
    range of clean penetration scenarios.
    """
    scenario_rows = []

    # Load step3 parquets for each ISO
    for iso in ISOS:
        fpath = STEP3_DIR / f'step3_co_{iso}.parquet'
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)

        # Sample representative scenarios at key thresholds
        for threshold in STEP3_THRESHOLDS:
            tdf = df[df['threshold'] == threshold]
            if tdf.empty:
                continue

            # Get P10/P50/P90 wholesale prices across all cost scenarios
            wholesale_p10 = tdf['cost_wholesale'].quantile(0.10)
            wholesale_p50 = tdf['cost_wholesale'].quantile(0.50)
            wholesale_p90 = tdf['cost_wholesale'].quantile(0.90)

            # Gas backup as capacity adder signal
            # More gas backup needed = tighter capacity market = higher prices
            gas_backup_p50 = tdf['gas_gas_backup_needed_mw'].median()
            # Capacity adder: if gas backup exceeds current fleet, new CCGT needed
            # → capacity prices rise. Rough model: +$10-30/kW-yr per GW of new gas
            existing_gas = tdf['gas_existing_gas_used_mw'].median()
            new_gas_needed = max(0, gas_backup_p50 - existing_gas)
            # Capacity adder: scarcity premium scales with new gas build
            cap_adder = min(40, new_gas_needed / 1000 * 5)  # $/kW-yr per GW

            for price_case, lmp_val in [('low', wholesale_p10), ('mid', wholesale_p50), ('high', wholesale_p90)]:
                # Map threshold to approximate clean_pct
                clean_pct_approx = threshold

                iso_data = {iso: {'avg_lmp': lmp_val, 'clean_pct': clean_pct_approx}}

                # Project across years — Step 3A is a snapshot model (2025)
                # but we project forward with clean energy growth assumptions
                for year in SIMULATION_YEARS:
                    # Clean pct grows over time toward threshold
                    if year <= 2025:
                        yr_clean = min(threshold, 40)  # Current ~40% average
                    elif year <= 2030:
                        yr_clean = min(threshold, 40 + (threshold - 40) * 0.3)
                    elif year <= 2035:
                        yr_clean = min(threshold, 40 + (threshold - 40) * 0.5)
                    elif year <= 2040:
                        yr_clean = min(threshold, 40 + (threshold - 40) * 0.7)
                    else:
                        yr_clean = threshold

                    # LMP adjusts with clean penetration (price suppression)
                    lmp_adj = lmp_val * (1.0 - 0.003 * max(0, yr_clean - 40))

                    yr_iso_data = {iso: {'avg_lmp': lmp_adj, 'clean_pct': yr_clean}}

                    scenario_rows.append({
                        'scenario': f'C_{iso}_{threshold:.0f}pct_{price_case}',
                        'year': year,
                        'iso_data': yr_iso_data,
                        'capacity_adder': cap_adder,
                        'meta': {
                            'step3_threshold': threshold,
                            'step3_price_case': price_case,
                            'step3_wholesale': float(lmp_val),
                            'step3_cap_adder': float(cap_adder),
                            'step3_new_gas_gw': float(new_gas_needed / 1000),
                        },
                    })

    return scenario_rows


# ─── Aggregation & Analysis ───────────────────────────────────────────

def compute_stranding_probabilities(df, model_filter=None):
    """Compute per-plant per-year stranding probability across scenarios."""
    if model_filter:
        df = df[df['model'] == model_filter]
    licensed = df[df['status'] != 'retired'].copy()

    strand_with_cfd = licensed.groupby(['plant', 'year']).apply(
        lambda g: (g['status'] == 'stranded').mean()
    ).reset_index(name='strand_prob_with_cfd')

    strand_merchant = licensed.groupby(['plant', 'year']).apply(
        lambda g: (g['merchant_profit_m'] < -5).mean()
    ).reset_index(name='strand_prob_merchant')

    saved = licensed.groupby(['plant', 'year']).apply(
        lambda g: g['saved_by_cfd'].mean()
    ).reset_index(name='saved_by_cfd_prob')

    probs = strand_with_cfd.merge(strand_merchant, on=['plant', 'year'])
    probs = probs.merge(saved, on=['plant', 'year'])
    return probs


def compute_model_comparison(df, plants):
    """Compute stranding stats per model for cross-model comparison."""
    models = df['model'].unique()
    comparison = {}
    for model in models:
        mdf = df[(df['model'] == model) & (df['status'] != 'retired')]
        by_year = {}
        for year in SIMULATION_YEARS:
            ydf = mdf[mdf['year'] == year]
            if ydf.empty:
                continue
            n_scenarios = ydf['scenario'].nunique()
            strand_rate = (ydf['status'] == 'stranded').mean()
            merchant_strand = (ydf['merchant_profit_m'] < -5).mean()
            avg_profit = ydf['profit_m'].mean()
            by_year[str(year)] = {
                'n_scenarios': int(n_scenarios),
                'strand_rate': round(float(strand_rate), 3),
                'merchant_strand_rate': round(float(merchant_strand), 3),
                'avg_profit_m': round(float(avg_profit), 1),
                'mw_stranded_p50': round(float(
                    ydf[ydf['status'] == 'stranded'].groupby('scenario')['cap_mw'].sum().median()
                ), 0) if (ydf['status'] == 'stranded').any() else 0,
            }
        comparison[model] = by_year
    return comparison


def compute_robust_findings(df, plants):
    """Identify findings that hold across ALL three models."""
    models = ['model_a', 'model_b', 'model_c']
    available_models = [m for m in models if m in df['model'].unique()]

    findings = {
        'robust_safe': [],      # Plants safe across all models
        'robust_at_risk': [],   # Plants at risk across all models
        'model_dependent': [],  # Plants where risk depends on model
        'agreement_by_year': {},
    }

    for year in [2035, 2040, 2050]:
        plant_risk = {}
        for plant in [p['name'] for p in plants]:
            risk_by_model = {}
            for model in available_models:
                mdf = df[(df['model'] == model) & (df['plant'] == plant) &
                        (df['year'] == year) & (df['status'] != 'retired')]
                if mdf.empty:
                    continue
                merchant_strand = (mdf['merchant_profit_m'] < -5).mean()
                risk_by_model[model] = merchant_strand

            if not risk_by_model:
                continue
            plant_risk[plant] = risk_by_model

        # Classify
        for plant, risks in plant_risk.items():
            all_safe = all(r < 0.1 for r in risks.values())
            all_risky = all(r > 0.3 for r in risks.values())
            cap = next((p['cap_mw'] for p in plants if p['name'] == plant), 0)

            if all_safe and year == 2040:
                findings['robust_safe'].append({'plant': plant, 'cap_mw': cap})
            elif all_risky and year == 2040:
                findings['robust_at_risk'].append({'plant': plant, 'cap_mw': cap,
                    'max_risk': round(max(risks.values()), 2)})
            elif year == 2040:
                findings['model_dependent'].append({
                    'plant': plant, 'cap_mw': cap,
                    'risk_range': [round(min(risks.values()), 2), round(max(risks.values()), 2)],
                })

        # Model agreement rate
        agree_count = 0
        total = 0
        for plant, risks in plant_risk.items():
            total += 1
            vals = list(risks.values())
            if all(v < 0.2 for v in vals) or all(v > 0.2 for v in vals):
                agree_count += 1
        findings['agreement_by_year'][str(year)] = {
            'agreement_rate': round(agree_count / total, 2) if total > 0 else 0,
            'n_plants': total,
        }

    return findings


def compute_nz_pathway_detail(df):
    """Extract NZ pathway-specific insights for accordion sections."""
    nz_df = df[df['model'] == 'model_b']
    if nz_df.empty:
        return {}

    pathways = {}
    for _, row in nz_df.iterrows():
        meta = {}
        for k in ['nz_scenario', 'nz_name', 'nz_type']:
            if k in row:
                meta[k] = row[k]
        tag = meta.get('nz_scenario', row['scenario'].replace('B_', ''))
        if tag not in pathways:
            pathways[tag] = {
                'name': meta.get('nz_name', tag),
                'type': meta.get('nz_type', 'unknown'),
                'years': {},
            }

    # Aggregate by pathway × year
    for tag in pathways:
        path_df = nz_df[nz_df['scenario'] == f'B_{tag}']
        for year in SIMULATION_YEARS:
            ydf = path_df[(path_df['year'] == year) & (path_df['status'] != 'retired')]
            if ydf.empty:
                continue
            pathways[tag]['years'][str(year)] = {
                'n_operating': int((ydf['status'] == 'operating').sum()),
                'n_stranded': int((ydf['status'] == 'stranded').sum()),
                'n_marginal': int((ydf['status'] == 'marginal').sum()),
                'total_profit_m': round(float(ydf['profit_m'].sum()), 1),
                'avg_lmp': round(float(ydf['avg_lmp'].mean()), 1),
                'avg_clean_pct': round(float(ydf['clean_pct'].mean()), 1),
            }

    return pathways


def compute_step3_capacity_impact(df):
    """Analyze how capacity adders from Step 3A affect nuclear economics."""
    c_df = df[df['model'] == 'model_c']
    if c_df.empty:
        return {}

    impact = {}
    for year in [2035, 2040, 2050]:
        ydf = c_df[(c_df['year'] == year) & (c_df['status'] != 'retired')]
        if ydf.empty:
            continue

        # Group by threshold
        by_threshold = {}
        for t in STEP3_THRESHOLDS:
            tdf = ydf[ydf['scenario'].str.contains(f'_{t:.0f}pct_')]
            if tdf.empty:
                continue
            by_threshold[str(int(t))] = {
                'strand_rate': round(float((tdf['status'] == 'stranded').mean()), 3),
                'avg_profit_m': round(float(tdf['profit_m'].mean()), 1),
                'avg_lmp': round(float(tdf['avg_lmp'].mean()), 1),
            }

        # Group by price case
        by_price = {}
        for case in ['low', 'mid', 'high']:
            pdf = ydf[ydf['scenario'].str.contains(f'_{case}$')]
            if pdf.empty:
                continue
            by_price[case] = {
                'strand_rate': round(float((pdf['status'] == 'stranded').mean()), 3),
                'avg_profit_m': round(float(pdf['profit_m'].mean()), 1),
            }

        impact[str(year)] = {
            'by_threshold': by_threshold,
            'by_price': by_price,
        }

    return impact


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


def build_per_model_probs(df, plants):
    """Per-model stranding probabilities for each plant × year."""
    models = df['model'].unique()
    per_model = {}
    for model in models:
        mdf = df[(df['model'] == model) & (df['status'] != 'retired')]
        model_data = {}
        for p in plants:
            pdata = {}
            for year in SIMULATION_YEARS:
                ydf = mdf[(mdf['plant'] == p['name']) & (mdf['year'] == year)]
                if ydf.empty:
                    continue
                pdata[str(year)] = {
                    'merchant': round(float((ydf['merchant_profit_m'] < -5).mean()), 3),
                    'with_cfd': round(float((ydf['status'] == 'stranded').mean()), 3),
                    'avg_profit': round(float(ydf['profit_m'].mean()), 1),
                }
            if pdata:
                model_data[p['name']] = pdata
        per_model[model] = model_data
    return per_model


def generate_js_output(plants, full_df, probs_all, model_comparison,
                       robust_findings, nz_pathways, step3_impact,
                       waterfalls, per_model_probs):
    """Generate pre-computed JS data for the dashboard."""
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

    # Combined stranding probabilities
    prob_data = {}
    for _, row in probs_all.iterrows():
        pname = row['plant']
        if pname not in prob_data:
            prob_data[pname] = {}
        prob_data[pname][str(int(row['year']))] = {
            'with_cfd': round(float(row['strand_prob_with_cfd']), 3),
            'merchant': round(float(row['strand_prob_merchant']), 3),
            'saved_by_cfd': round(float(row['saved_by_cfd_prob']), 3),
        }

    # Dimension sensitivity from Model A
    dim_sensitivity = {}
    model_a_df = full_df[full_df['model'] == 'model_a']
    licensed = model_a_df[model_a_df['status'] != 'retired']
    for dim in ['conditions', 'demand_growth', 'price_sens', 'gas_friction']:
        if dim not in licensed.columns:
            continue
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

    # ISO bands from combined data
    iso_band_data = {}
    for iso in ISOS:
        iso_df = full_df[(full_df['iso'] == iso) & (full_df['status'] != 'retired')]
        if iso_df.empty:
            continue
        iso_agg = iso_df.groupby(['year', 'scenario']).agg(
            cap_mw=('cap_mw', 'sum'),
            profit_m=('profit_m', 'sum'),
        ).reset_index()

        stranded = full_df[(full_df['iso'] == iso) & (full_df['status'] == 'stranded')].groupby(
            ['year', 'scenario']
        )['cap_mw'].sum().reset_index(name='stranded_mw')
        iso_agg = iso_agg.merge(stranded, on=['year', 'scenario'], how='left')
        iso_agg['stranded_mw'] = iso_agg['stranded_mw'].fillna(0)

        bands = iso_agg.groupby('year').agg(
            total_mw=('cap_mw', 'first'),
            stranded_p10=('stranded_mw', lambda x: np.percentile(x, 10)),
            stranded_p50=('stranded_mw', lambda x: np.percentile(x, 50)),
            stranded_p90=('stranded_mw', lambda x: np.percentile(x, 90)),
            profit_p10=('profit_m', lambda x: np.percentile(x, 10)),
            profit_p50=('profit_m', lambda x: np.percentile(x, 50)),
            profit_p90=('profit_m', lambda x: np.percentile(x, 90)),
        ).reset_index()

        iso_band_data[iso] = {
            'years': bands['year'].tolist(),
            'total_mw': int(bands['total_mw'].iloc[0]) if len(bands) else 0,
            'stranded': {
                'p10': [round(float(v), 1) for v in bands['stranded_p10']],
                'p50': [round(float(v), 1) for v in bands['stranded_p50']],
                'p90': [round(float(v), 1) for v in bands['stranded_p90']],
            },
            'profit': {
                'p10': [round(float(v), 1) for v in bands['profit_p10']],
                'p50': [round(float(v), 1) for v in bands['profit_p50']],
                'p90': [round(float(v), 1) for v in bands['profit_p90']],
            },
        }

    # Policy timeline
    policy_timeline = {
        'IL_CEJA_CMC': {'start': 2022, 'end': 2027, 'floor': 33.38, 'plants_mw': 11782},
        'NY_ZEC': {'start': 2017, 'end': 2029, 'floor': 17.48, 'plants_mw': 3325},
        'NJ_ZEC': {'start': 2019, 'end': 2025, 'floor': 10.00, 'plants_mw': 3467},
        '45U_PTC': {'start': 2024, 'end': 2032, 'floor': 15.00, 'plants_mw': 'all'},
    }

    # Headline stats
    total_nuc_mw = sum(p['cap_mw'] for p in plants)
    prob_2035 = probs_all[probs_all['year'] == 2035]
    prob_2040 = probs_all[probs_all['year'] == 2040]
    mw_at_risk_2035 = sum(
        p['cap_mw'] for p in plants
        if any((prob_2035['plant'] == p['name']) & (prob_2035['strand_prob_merchant'] > 0.5))
    )
    mw_at_risk_2040 = sum(
        p['cap_mw'] for p in plants
        if any((prob_2040['plant'] == p['name']) & (prob_2040['strand_prob_merchant'] > 0.5))
    )

    # Robust findings headline
    robust_safe_mw = sum(p['cap_mw'] for p in robust_findings.get('robust_safe', []))
    robust_risk_mw = sum(p['cap_mw'] for p in robust_findings.get('robust_at_risk', []))
    model_dep_mw = sum(p['cap_mw'] for p in robust_findings.get('model_dependent', []))

    headline = {
        'total_nuclear_mw': total_nuc_mw,
        'total_nuclear_gw': round(total_nuc_mw / 1000, 1),
        'n_plants': len(plants),
        'n_models': len(full_df['model'].unique()),
        'total_scenarios': full_df['scenario'].nunique(),
        'mw_at_risk_2035_merchant': mw_at_risk_2035,
        'mw_at_risk_2040_merchant': mw_at_risk_2040,
        'pct_at_risk_2035': round(mw_at_risk_2035 / total_nuc_mw * 100, 1) if total_nuc_mw > 0 else 0,
        'pct_at_risk_2040': round(mw_at_risk_2040 / total_nuc_mw * 100, 1) if total_nuc_mw > 0 else 0,
        '45u_annual_value_b': round(total_nuc_mw * NUCLEAR_CF * 8760 * FEDERAL_45U_FLOOR_MWH / 1e9, 2),
        'robust_safe_gw': round(robust_safe_mw / 1000, 1),
        'robust_risk_gw': round(robust_risk_mw / 1000, 1),
        'model_dependent_gw': round(model_dep_mw / 1000, 1),
        'agreement_2040': robust_findings.get('agreement_by_year', {}).get('2040', {}).get('agreement_rate', 0),
    }

    js_data = {
        'headline': headline,
        'plants': plant_meta,
        'stranding_probs': prob_data,
        'per_model_probs': per_model_probs,
        'model_comparison': model_comparison,
        'robust_findings': robust_findings,
        'nz_pathways': nz_pathways,
        'step3_impact': step3_impact,
        'iso_bands': iso_band_data,
        'waterfalls': waterfalls,
        'dim_sensitivity': dim_sensitivity,
        'policy_timeline': policy_timeline,
        'years': SIMULATION_YEARS,
        'model_names': {
            'model_a': 'Parametric Market Sweep (270 scenarios)',
            'model_b': 'Net-Zero Transition Pathways (10 scenarios)',
            'model_c': 'Cost-Optimized Baseline + Capacity (90 scenarios)',
        },
    }

    os.makedirs(JS_OUT.parent, exist_ok=True)
    with open(JS_OUT, 'w') as f:
        f.write('// Auto-generated by step12_nuclear_retirement.py\n')
        f.write('// Nuclear fleet stranding analysis — multi-model stress test\n')
        f.write(f'const NUCLEAR_RETIREMENT_DATA = {json.dumps(js_data, indent=2)};\n')

    print(f'  Wrote {JS_OUT}')
    return js_data


def main():
    print('=' * 60)
    print('Step 12: Nuclear Retirement & Stranding Analysis')
    print('          Multi-Model Stress Test')
    print('=' * 60)

    # ─── Load plant data ─────────────────────────────────────────────
    print('\n1. Loading nuclear plant data...')
    plants = load_nuclear_plants()
    print(f'   {len(plants)} plants, {sum(p["cap_mw"] for p in plants)/1000:.1f} GW')

    # ─── Load all three models ───────────────────────────────────────
    print('\n2. Loading Model A: Step 10 Reference Sweep...')
    model_a_rows = load_model_a_scenarios()
    print(f'   {len(model_a_rows)} scenario-year combinations')

    print('\n3. Loading Model B: Net-Zero Pathways...')
    model_b_rows = load_model_b_scenarios()
    print(f'   {len(model_b_rows)} NZ pathway-year combinations')

    print('\n4. Loading Model C: Step 3A Track 1 Baseline + Capacity Adders...')
    model_c_rows = load_model_c_scenarios()
    print(f'   {len(model_c_rows)} cost-scenario-year combinations')

    # ─── Compute economics ───────────────────────────────────────────
    print('\n5. Computing plant economics — Model A...')
    results_a = compute_plant_economics_vectorized(plants, model_a_rows, 'model_a')
    print(f'   {len(results_a)} records')

    print('   Computing plant economics — Model B...')
    results_b = compute_plant_economics_vectorized(plants, model_b_rows, 'model_b')
    print(f'   {len(results_b)} records')

    print('   Computing plant economics — Model C...')
    results_c = compute_plant_economics_vectorized(plants, model_c_rows, 'model_c')
    print(f'   {len(results_c)} records')

    # Combine
    all_results = results_a + results_b + results_c
    full_df = pd.DataFrame(all_results)
    print(f'\n   Total: {len(full_df)} plant-scenario-year records across {full_df["model"].nunique()} models')

    # ─── Stranding probabilities (combined) ──────────────────────────
    print('\n6. Computing stranding probabilities...')
    probs_all = compute_stranding_probabilities(full_df)
    print(f'   {len(probs_all)} plant-year probability records')

    # ─── Model comparison ────────────────────────────────────────────
    print('\n7. Cross-model comparison...')
    model_comparison = compute_model_comparison(full_df, plants)
    for model, data in model_comparison.items():
        yr40 = data.get('2040', {})
        print(f'   {model}: 2040 merchant strand rate = {yr40.get("merchant_strand_rate", 0)*100:.1f}%')

    # ─── Robust findings ─────────────────────────────────────────────
    print('\n8. Identifying robust cross-model findings...')
    robust_findings = compute_robust_findings(full_df, plants)
    n_safe = len(robust_findings['robust_safe'])
    n_risk = len(robust_findings['robust_at_risk'])
    n_dep = len(robust_findings['model_dependent'])
    print(f'   Robust safe: {n_safe} plants, At risk: {n_risk}, Model-dependent: {n_dep}')

    # ─── NZ pathway detail ───────────────────────────────────────────
    print('\n9. Extracting NZ pathway details...')
    nz_pathways = compute_nz_pathway_detail(full_df)
    print(f'   {len(nz_pathways)} NZ pathways analyzed')

    # ─── Step 3A capacity impact ─────────────────────────────────────
    print('\n10. Analyzing Step 3A capacity market impact...')
    step3_impact = compute_step3_capacity_impact(full_df)

    # ─── Revenue waterfalls ──────────────────────────────────────────
    print('\n11. Building revenue waterfalls...')
    waterfalls = build_revenue_waterfall(full_df, plants)
    print(f'   {len(waterfalls)} plant waterfalls')

    # ─── Per-model probabilities ─────────────────────────────────────
    print('\n12. Computing per-model probabilities...')
    per_model_probs = build_per_model_probs(full_df, plants)

    # ─── Save outputs ────────────────────────────────────────────────
    print('\n13. Saving outputs...')
    os.makedirs(OUT_DIR, exist_ok=True)
    full_df.to_parquet(OUT_DIR / 'nuclear_stranding_results.parquet', index=False)
    probs_all.to_parquet(OUT_DIR / 'nuclear_stranding_probabilities.parquet', index=False)

    # ─── Generate JS ─────────────────────────────────────────────────
    print('\n14. Generating dashboard JS data...')
    js_data = generate_js_output(
        plants, full_df, probs_all, model_comparison,
        robust_findings, nz_pathways, step3_impact,
        waterfalls, per_model_probs
    )

    # ─── Summary ─────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    h = js_data['headline']
    print(f"  Fleet: {h['n_plants']} plants, {h['total_nuclear_gw']} GW")
    print(f"  Models: {h['n_models']}, Total scenarios: {h['total_scenarios']}")
    print(f"  45U annual value: ${h['45u_annual_value_b']}B/yr")
    print(f"  MW at risk (merchant, P>50%) by 2035: {h['mw_at_risk_2035_merchant']:,.0f} MW ({h['pct_at_risk_2035']}%)")
    print(f"  MW at risk (merchant, P>50%) by 2040: {h['mw_at_risk_2040_merchant']:,.0f} MW ({h['pct_at_risk_2040']}%)")
    print(f"  Robust safe (2040): {h['robust_safe_gw']} GW")
    print(f"  Robust at risk (2040): {h['robust_risk_gw']} GW")
    print(f"  Model-dependent (2040): {h['model_dependent_gw']} GW")
    print(f"  Cross-model agreement (2040): {h['agreement_2040']*100:.0f}%")

    print('\n  Per-plant merchant stranding probability (2040, by model):')
    for p in plants:
        parts = []
        for model in ['model_a', 'model_b', 'model_c']:
            mp = per_model_probs.get(model, {}).get(p['name'], {}).get('2040', {})
            parts.append(f"{mp.get('merchant', 0)*100:5.1f}%")
        print(f"    {p['name']:40s}  A:{parts[0]}  B:{parts[1]}  C:{parts[2]}")

    print('\nDone.')


if __name__ == '__main__':
    main()
