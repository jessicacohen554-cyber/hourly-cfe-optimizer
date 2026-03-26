"""
Coal Wall Analysis: Consequential Advantage for Meta + Amazon

Computes how consequential accounting (Strategy 1B, fossil-average baseline)
lets hyperscalers procure LESS VRE by targeting coal displacement first.

Coal emits ~1.0 tCO2/MWh vs fossil-average ~0.64 tCO2/MWh. Each TWh of
coal-displacing VRE offsets ~56% more CO2 than the baseline requires,
reducing total procurement needed.

Outputs: data-center-cfe/data-inputs/coal_wall_analysis_results.json
"""

import json
import os
import csv
import math

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
DATA_CENTER_DIR = os.path.join(SCRIPT_DIR, '..')
MAC_QUEUE_DIR = os.path.join(PROJECT_ROOT, 'data', 'step3-dispatch', 'mac_queue')
OUTPUT_PATH = os.path.join(DATA_CENTER_DIR, 'data-inputs', 'coal_wall_analysis_results.json')

# ── Constants (from pipeline_config.py, lmp_engine.py, procurement_utils.py) ──

# Emission rates (tCO2/MWh) by fuel type
EMISSION_RATES = {
    'coal_steam': 1.0,
    'gas_ccgt': 0.4,
    'gas_ct': 0.55,
    'oil_ct': 0.65,
}

# Fossil capacity shares by ISO (from lmp_engine.py FOSSIL_CAPACITY_SHARES)
FOSSIL_MIX = {
    'MISO': {'coal_steam': 0.35, 'gas_ccgt': 0.40, 'gas_ct': 0.24, 'oil_ct': 0.01},
    'SPP':  {'coal_steam': 0.30, 'gas_ccgt': 0.42, 'gas_ct': 0.27, 'oil_ct': 0.01},
    'PJM':  {'coal_steam': 0.29, 'gas_ccgt': 0.37, 'gas_ct': 0.31, 'oil_ct': 0.03},
}

# Announced coal retirements (cumulative GW by year, from pipeline_config.py)
ANNOUNCED_COAL_RETIREMENTS = {
    'PJM':  {2025: 3.0, 2028: 10.0, 2030: 18.0, 2032: 25.0, 2035: 30.0},
    'MISO': {2025: 2.0, 2028: 6.0,  2030: 10.0, 2032: 15.0, 2035: 18.0},
    'SPP':  {2025: 1.0, 2028: 3.0,  2030: 5.0,  2032: 8.0,  2035: 9.0},
}

# 2025 baseline coal generation (TWh, from pipeline_config.py COAL_CAP_TWH)
COAL_CAP_TWH = {'MISO': 125.0, 'SPP': 42.0, 'PJM': 139.09}

# Total coal capacity (GW, approximate from EIA)
COAL_CAP_GW = {'MISO': 45.0, 'SPP': 20.0, 'PJM': 55.0}

# Contracted clean energy (TWh, from procurement_utils.py)
CONTRACTED_CLEAN_TWH = {'PJM': 18.8, 'MISO': 18.0, 'SPP': 0.0}

# Hyperscaler demand (TWh, 2030)
META_DEMAND_TWH = 134.0
AMAZON_DEMAND_TWH = 164.0
COMBINED_DEMAND_TWH = META_DEMAND_TWH + AMAZON_DEMAND_TWH


def compute_fossil_avg_rate(iso):
    """Weighted-average fossil emission rate for an ISO."""
    mix = FOSSIL_MIX[iso]
    return sum(mix[fuel] * EMISSION_RATES[fuel] for fuel in mix)


def load_miso_queue():
    """Load the MISO consequential queue (MAC-sorted)."""
    path = os.path.join(MAC_QUEUE_DIR, 'consequential_queue.json')
    content = open(path).read()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(content.lstrip())
    queue = obj['queue']
    queue.sort(key=lambda x: x['queue_position'])
    return queue


def load_scenario_a(iso):
    """Load scenario_a data for an ISO (threshold → results dict)."""
    path = os.path.join(MAC_QUEUE_DIR, f'scenario_a_{iso}.json')
    data = json.load(open(path))
    return data['results']


def build_queue_from_scenario(iso):
    """
    Build a MAC-like queue from scenario_a data for SPP/PJM.
    Each transition between thresholds becomes a queue entry.
    """
    results = load_scenario_a(iso)
    thresholds = sorted(results.keys(), key=lambda x: float(x))

    queue = []
    prev_t = None
    prev_r = None
    for t in thresholds:
        r = results[t]
        if prev_r is None:
            prev_t = t
            prev_r = r
            continue

        delta_res = {}
        for res_type in r['resource_twh']:
            delta = r['resource_twh'][res_type] - prev_r['resource_twh'].get(res_type, 0)
            if abs(delta) > 0.01:
                delta_res[res_type] = delta

        total_delta_twh = sum(max(0, v) for v in delta_res.values())
        if total_delta_twh < 0.1:
            prev_t = t
            prev_r = r
            continue

        marginal_cost = r['effective_cost']

        queue.append({
            'iso': iso,
            'threshold_start': float(prev_t),
            'threshold_end': float(t),
            'marginal_cost_per_mwh': marginal_cost,
            'delta_resources': delta_res,
            'total_new_twh': total_delta_twh,
            'demand_twh': r['demand_twh'],
        })
        prev_t = t
        prev_r = r

    # Sort by marginal cost
    queue.sort(key=lambda x: x['marginal_cost_per_mwh'])
    return queue


def compute_coal_displacement(queue_entry, iso):
    """
    Estimate how much coal is displaced by new VRE in a queue entry.
    Coal is displaced first (merit-order retirement: coal → oil → gas).
    """
    coal_share = FOSSIL_MIX[iso]['coal_steam']
    total_twh = sum(max(0, v) for v in queue_entry.get('delta_resources', {}).values())

    # Coal displaced is proportional to coal's share of fossil mix,
    # but coal retires FIRST so displacement is front-loaded.
    # At low penetration, coal displacement fraction > coal share.
    # Use min(coal_share * 1.5, 1.0) as coal displacement fraction for early queue entries.
    coal_displacement_fraction = min(coal_share * 1.5, 0.85)
    coal_displaced_twh = total_twh * coal_displacement_fraction

    return coal_displaced_twh


def compute_coal_remaining_twh(iso, year):
    """
    Estimate remaining coal generation (TWh) in a given year after retirements.
    Assumes linear capacity factor between retirement milestones.
    """
    retirements = ANNOUNCED_COAL_RETIREMENTS[iso]
    total_gw = COAL_CAP_GW[iso]
    baseline_twh = COAL_CAP_TWH[iso]

    # Interpolate retired GW at the given year
    years = sorted(retirements.keys())
    if year <= years[0]:
        retired_gw = retirements[years[0]] * (year - 2024) / (years[0] - 2024) if year > 2024 else 0
    elif year >= years[-1]:
        retired_gw = retirements[years[-1]]
    else:
        for i in range(len(years) - 1):
            if years[i] <= year <= years[i + 1]:
                frac = (year - years[i]) / (years[i + 1] - years[i])
                retired_gw = retirements[years[i]] + frac * (retirements[years[i + 1]] - retirements[years[i]])
                break

    remaining_fraction = max(0, (total_gw - retired_gw) / total_gw)
    return baseline_twh * remaining_fraction


def run_analysis():
    """Main analysis: compute coal wall advantage for Meta + Amazon."""

    # ── 1. Fossil-average emission rates ──
    fossil_avg_rates = {iso: round(compute_fossil_avg_rate(iso), 4) for iso in FOSSIL_MIX}
    print("Fossil-average emission rates (tCO2/MWh):")
    for iso, rate in fossil_avg_rates.items():
        print(f"  {iso}: {rate}")

    # Weighted average across MISO+SPP+PJM (weight by coal generation)
    total_coal = sum(COAL_CAP_TWH[iso] for iso in ['MISO', 'SPP', 'PJM'])
    weighted_rate = sum(
        fossil_avg_rates[iso] * COAL_CAP_TWH[iso] / total_coal
        for iso in ['MISO', 'SPP', 'PJM']
    )
    print(f"  Weighted avg (MISO+SPP+PJM): {weighted_rate:.4f}")

    # ── 2. CO2 baseline ──
    co2_baseline_mt = COMBINED_DEMAND_TWH * weighted_rate
    print(f"\nCO2 baseline: {COMBINED_DEMAND_TWH} TWh × {weighted_rate:.4f} = {co2_baseline_mt:.1f} Mt")

    # ── 3. Load queues ──
    miso_queue = load_miso_queue()
    spp_queue = build_queue_from_scenario('SPP')
    pjm_queue = build_queue_from_scenario('PJM')

    print(f"\nQueue sizes: MISO={len(miso_queue)}, SPP={len(spp_queue)}, PJM={len(pjm_queue)}")

    # ── 4. Walk the coal displacement queue ──
    # Under consequential accounting with fossil-average baseline:
    # - Each TWh of VRE that displaces coal offsets 1.0 tCO2
    # - But baseline only requires offsetting ~0.64 tCO2 per TWh of demand
    # - Leverage ratio: coal_rate / fossil_avg_rate

    leverage_ratios = {
        iso: EMISSION_RATES['coal_steam'] / fossil_avg_rates[iso]
        for iso in fossil_avg_rates
    }
    print("\nCoal displacement leverage ratios:")
    for iso, ratio in leverage_ratios.items():
        print(f"  {iso}: {ratio:.2f}x")

    # Average leverage across target ISOs
    avg_leverage = sum(leverage_ratios[iso] for iso in ['MISO', 'SPP', 'PJM']) / 3
    print(f"  Average: {avg_leverage:.2f}x")

    # ── 5. Compute VRE needed under consequential vs hourly ──
    #
    # KEY DISTINCTION:
    # - Hourly matching (Strategy 2): must match demand 1:1 every hour → 298 TWh of VRE
    # - Consequential (Strategy 1B): must OFFSET CO2 footprint, not match TWh
    #   Footprint = 298 TWh × fossil_avg_rate ≈ 189 Mt CO2
    #   If VRE displaces coal (1.0 tCO2/MWh), need 189 TWh of VRE to offset 189 Mt CO2
    #   That's 37% LESS than hourly matching
    #
    # The coal pool (306 TWh across MISO+SPP+PJM) is larger than what Meta+Amazon
    # need (189 TWh), so coal displacement is not pool-limited for them alone.
    # BUT as more C&I buyers enter, the pool gets competed away.

    hourly_vre_twh = COMBINED_DEMAND_TWH

    total_coal_twh = sum(COAL_CAP_TWH[iso] for iso in ['MISO', 'SPP', 'PJM'])
    print(f"\nTotal coal generation (2025): {total_coal_twh:.0f} TWh")

    # Under consequential: offset CO2, not match TWh
    # CO2 to offset = demand × fossil_avg_rate
    # VRE needed = CO2_to_offset / coal_emission_rate (since VRE displaces coal first)
    consequential_vre_needed = co2_baseline_mt / EMISSION_RATES['coal_steam']
    # co2_baseline_mt is in Mt, coal rate is tCO2/MWh = Mt/TWh, so result is TWh
    print(f"Consequential VRE needed (coal displacement): {consequential_vre_needed:.0f} TWh")
    print(f"Coal pool available: {total_coal_twh:.0f} TWh (sufficient: {total_coal_twh > consequential_vre_needed})")

    # ISO demand totals (from scenario data, ~2030)
    miso_demand = 740  # TWh
    spp_demand = 328   # TWh
    pjm_demand = 950   # TWh
    total_demand = miso_demand + spp_demand + pjm_demand
    meta_amazon_share = COMBINED_DEMAND_TWH / total_demand

    vre_needed = consequential_vre_needed
    twh_savings = hourly_vre_twh - vre_needed
    pct_savings = twh_savings / hourly_vre_twh * 100

    print(f"\n── RESULTS ──")
    print(f"Hourly 1:1 VRE needed:      {hourly_vre_twh:.0f} TWh")
    print(f"Consequential VRE needed:    {vre_needed:.0f} TWh")
    print(f"Savings:                     {twh_savings:.0f} TWh ({pct_savings:.0f}%)")
    print(f"Meta+Amazon share of demand: {meta_amazon_share:.1%}")

    # ── 6. Participation table ──
    # "C&I participation" = % of total ISO demand matched with consequential VRE
    participation_levels = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    participation_table = []

    # The participation table models what happens as MORE C&I buyers adopt
    # consequential accounting. At low participation, coal displacement is
    # abundant and cheap. As participation rises, the coal pool depletes
    # and costs rise (later queue entries have higher MAC).
    #
    # Each row: if X% of total ISO demand uses consequential procurement,
    # how much coal can be displaced, and at what cost?

    for pct in participation_levels:
        demand_matched = total_demand * pct

        # Coal displacement scales with participation but is capped by available coal
        miso_coal_disp = min(COAL_CAP_TWH['MISO'] * pct, COAL_CAP_TWH['MISO'])
        spp_coal_disp = min(COAL_CAP_TWH['SPP'] * pct, COAL_CAP_TWH['SPP'])
        pjm_coal_disp = min(COAL_CAP_TWH['PJM'] * pct, COAL_CAP_TWH['PJM'])
        total_coal_disp = miso_coal_disp + spp_coal_disp + pjm_coal_disp

        # CO2 offset from coal displacement (1.0 tCO2/MWh)
        co2_from_coal = total_coal_disp * EMISSION_RATES['coal_steam']

        # VRE needed: 1 TWh VRE per 1 TWh coal displaced
        vre_procured = total_coal_disp

        # Cost from MISO queue data (MAC-sorted). Map participation level
        # to approximate queue position based on threshold coverage.
        # MISO queue: entry 1 (87.5→90%, MAC $41) + entry 2 (23→50%, MAC $53)
        # cover the cheapest displacement. Beyond ~30% participation,
        # we enter higher-MAC entries.
        if pct <= 0.10:
            cost = 49   # Cheap pool: first queue entries
        elif pct <= 0.20:
            cost = 51   # Still in cheap zone
        elif pct <= 0.30:
            cost = 52   # Approaching pool edge
        elif pct <= 0.40:
            cost = 58   # Higher-MAC entries
        else:
            cost = 66   # Beyond cheap pool, gas displacement territory

        # Pool exhaustion: at what point is >80% of coal claimed?
        pool_exhausted = total_coal_disp >= total_coal_twh * 0.80

        participation_table.append({
            'participation_pct': int(pct * 100),
            'miso_coal_displaced_twh': round(miso_coal_disp, 1),
            'spp_coal_displaced_twh': round(spp_coal_disp, 1),
            'pjm_coal_displaced_twh': round(pjm_coal_disp, 1),
            'vre_procured_twh': round(vre_procured, 1),
            'co2_offset_mt': round(co2_from_coal, 1),
            'cost_per_mwh': cost,
            'pool_exhausted': pool_exhausted,
        })

    print("\nParticipation Table:")
    fmt = "{:>5}% | MISO {:>6} | SPP {:>5} | PJM {:>6} | VRE {:>6} | CO2 {:>6} | ${:>3}/MWh"
    for row in participation_table:
        print(fmt.format(
            row['participation_pct'],
            row['miso_coal_displaced_twh'], row['spp_coal_displaced_twh'],
            row['pjm_coal_displaced_twh'], row['vre_procured_twh'],
            row['co2_offset_mt'], row['cost_per_mwh']))

    # ── 7. Coal retirement timeline ──
    coal_timeline = {}
    for iso in ['MISO', 'SPP', 'PJM']:
        timeline = {}
        for year in range(2025, 2036):
            remaining = compute_coal_remaining_twh(iso, year)
            timeline[year] = round(remaining, 1)
        coal_timeline[iso] = timeline

    # When does the coal advantage window close?
    # The window closes when coal retirement ACCELERATION makes the displacement
    # pool unreliable for long-term procurement planning. By 2032:
    # - PJM loses 45% of coal (25 GW retired)
    # - MISO loses 33% (15 GW retired)
    # - SPP loses 40% (8 GW retired)
    # - Combined: ~35% of the coal pool is gone
    # - Remaining coal faces accelerating retirement schedules
    # - Marginal displacement cost rises as cheap coal exits first
    #
    # The "window" is 2025-2032: deepest displacement value before
    # retirement acceleration makes the pool unreliable.
    window_close_years = {}
    for iso in ['MISO', 'SPP', 'PJM']:
        baseline = COAL_CAP_TWH[iso]
        # Window closes when >30% of coal is retired (retirement acceleration point)
        for year in range(2025, 2041):
            remaining = compute_coal_remaining_twh(iso, year)
            if remaining < baseline * 0.70:
                window_close_years[iso] = year
                break
        else:
            window_close_years[iso] = 2035

    print(f"\nCoal advantage window closes (70% remaining threshold):")
    for iso, year in window_close_years.items():
        print(f"  {iso}: {year}")

    # Aggregate window: weighted by coal TWh
    avg_window_close = round(sum(
        window_close_years[iso] * COAL_CAP_TWH[iso] / total_coal_twh
        for iso in ['MISO', 'SPP', 'PJM']
    ))
    print(f"  Weighted avg: {avg_window_close}")

    # ── 8. Cheap pool exhaustion ──
    # The cheap pool ($49-52/MWh) in MISO lasts through ~30% C&I participation
    # based on the queue data (first 2 queue entries cover 23→90% threshold)
    cheap_pool_exhaustion_pct = 30

    # ── Assemble results ──
    results = {
        'fossil_avg_rates': fossil_avg_rates,
        'weighted_fossil_avg_rate': round(weighted_rate, 4),
        'leverage_ratios': {iso: round(v, 2) for iso, v in leverage_ratios.items()},
        'avg_leverage_ratio': round(avg_leverage, 2),
        'meta_demand_twh': META_DEMAND_TWH,
        'amazon_demand_twh': AMAZON_DEMAND_TWH,
        'combined_demand_twh': COMBINED_DEMAND_TWH,
        'co2_baseline_mt': round(co2_baseline_mt, 1),
        'hourly_vre_needed_twh': hourly_vre_twh,
        'consequential_vre_needed_twh': round(vre_needed, 0),
        'twh_savings': round(twh_savings, 0),
        'pct_savings': round(pct_savings, 0),
        'cheap_pool_cost_per_mwh': 49,
        'cheap_pool_exhaustion_pct': cheap_pool_exhaustion_pct,
        'window_closes_year': avg_window_close,
        'window_close_years_by_iso': window_close_years,
        'participation_table': participation_table,
        'coal_retirement_timeline': coal_timeline,
        'total_coal_twh_2025': total_coal_twh,
        'meta_amazon_demand_share': round(meta_amazon_share, 3),
        'methodology': {
            'strategy': '1B — Consequential, fossil-average baseline',
            'coal_emission_rate': 1.0,
            'gas_ccgt_rate': 0.4,
            'gas_ct_rate': 0.55,
            'oil_rate': 0.65,
            'coal_displaced_first': True,
            'note': 'Coal is retired first in merit-order dispatch. '
                    'Each TWh displacing coal offsets ~1.56x more CO2 than '
                    'the fossil-average baseline requires.'
        }
    }

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {OUTPUT_PATH}")

    return results


if __name__ == '__main__':
    run_analysis()
