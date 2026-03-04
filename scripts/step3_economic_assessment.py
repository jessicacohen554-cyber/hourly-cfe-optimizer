#!/usr/bin/env python3
"""Step 3 Economic Assessment: Full lifecycle economics for storage.

Augments Step 3 cost optimization results with storage revenue streams:
  1. Capacity market payments ($/kW-yr × capacity × credit)
  2. Energy arbitrage (peak-offpeak spread × daily cycles × RTE)
  3. Ancillary services (regulation + spinning reserve)
  4. Degradation costs (cycle-based battery replacement)

Applied to ALL tracks (1, 2, 3) — this is an enhanced economic lens on
existing results, not a separate track.

Revenue model uses summary LMP stats (peak/offpeak/avg) from
step5_compute_lmp_prices.py rather than 8,760-hour profiles. This is
accurate for V1 (daily-cycle dispatch) and can be upgraded to hourly
dispatch in V2 when the enhanced multi-service model is built.

Output: data/step3-economic-parquets/{ISO}_economic.parquet

Usage:
  python scripts/step3_economic_assessment.py                   # All ISOs
  python scripts/step3_economic_assessment.py --iso PJM         # Single ISO
  python scripts/step3_economic_assessment.py --iso ALL --v2    # Use 1D.2 EF source
"""

import argparse
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pipeline_config import (
    CAPACITY_MARKET_PRICES,
    ANCILLARY_SERVICE_RATES,
    ANCILLARY_HOURS,
    BATTERY_DEGRADATION,
    PEAK_CAPACITY_CREDITS,
    BATTERY_EFFICIENCY,
    BATTERY8_EFFICIENCY,
    LDES_EFFICIENCY,
    H2_EFFICIENCY,
    BATTERY_DURATION_HOURS,
    BATTERY8_DURATION_HOURS,
    LDES_DURATION_HOURS,
    H2_DURATION_HOURS,
)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. pip install pyarrow")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
STEP3_DIR = os.path.join(DATA_DIR, 'step3-cost-opt-parquets')
LMP_DIR = os.path.join(DATA_DIR, 'step5-post-processing', 'lmp')
OUTPUT_DIR = os.path.join(DATA_DIR, 'step3-economic-parquets')

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Hours per year
H = 8760


# ══════════════════════════════════════════════════════════════════════════════
# LMP DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_lmp_spreads(iso):
    """Load peak-offpeak LMP spreads per (threshold, fuel_level).

    Returns dict: {(threshold, fuel_level): {avg, peak, offpeak, spread}}.
    Falls back to default spreads if LMP data unavailable.
    """
    lmp_path = os.path.join(LMP_DIR, f'{iso}_lmp.parquet')
    if not os.path.exists(lmp_path):
        print(f"  WARNING: No LMP data for {iso}, using default spreads")
        return None

    t = pq.read_table(lmp_path,
                       columns=['threshold', 'fuel_level',
                                'avg_lmp', 'peak_avg_lmp', 'offpeak_avg_lmp'])

    thresholds = t.column('threshold').to_numpy()
    fuel_levels = t.column('fuel_level').to_pylist()
    avg = t.column('avg_lmp').to_numpy()
    peak = t.column('peak_avg_lmp').to_numpy()
    offpeak = t.column('offpeak_avg_lmp').to_numpy()

    spreads = {}
    for i in range(len(thresholds)):
        key = (float(thresholds[i]), fuel_levels[i])
        spreads[key] = {
            'avg': float(avg[i]),
            'peak': float(peak[i]),
            'offpeak': float(offpeak[i]),
            'spread': float(peak[i] - offpeak[i]),
        }

    return spreads


def get_spread_for_row(spreads, threshold, fuel_level):
    """Get LMP spread for a given threshold and fuel level.

    Falls back to nearest threshold, then to default spread.
    """
    if spreads is None:
        return 20.0  # Default peak-offpeak spread ($/MWh)

    key = (threshold, fuel_level)
    if key in spreads:
        return max(spreads[key]['spread'], 0.0)

    # Fallback: nearest threshold at same fuel level
    same_fuel = {k: v for k, v in spreads.items() if k[1] == fuel_level}
    if same_fuel:
        nearest = min(same_fuel.keys(), key=lambda k: abs(k[0] - threshold))
        return max(same_fuel[nearest]['spread'], 0.0)

    return 20.0  # Default


# ══════════════════════════════════════════════════════════════════════════════
# REVENUE COMPUTATION (VECTORIZED)
# ══════════════════════════════════════════════════════════════════════════════

def compute_storage_economics(iso, step3_table, lmp_spreads):
    """Compute storage revenue streams for all rows in a Step 3 table.

    All computations are vectorized over the N rows of the table.
    Revenue is normalized to $/MWh of total annual demand for direct
    comparison with cost_effective_cost.

    Returns dict of new column arrays (each length N).
    """
    n = step3_table.num_rows

    # Extract arrays
    demand_mwh = step3_table.column('annual_demand_mwh').to_numpy()
    bat4_pct = step3_table.column('battery_dispatch_pct').to_numpy()
    bat8_pct = step3_table.column('battery8_dispatch_pct').to_numpy()
    ldes_pct = step3_table.column('ldes_dispatch_pct').to_numpy()
    h2_pct = step3_table.column('h2_dispatch_pct').to_numpy()
    total_cost = step3_table.column('cost_total_cost').to_numpy()
    effective_cost = step3_table.column('cost_effective_cost').to_numpy()

    thresholds = step3_table.column('threshold').to_numpy()
    scenarios = step3_table.column('scenario').to_pylist()

    # ── 1. CAPACITY MARKET REVENUE ──
    # Revenue = storage_MW × capacity_credit × capacity_price
    # storage_MW = dispatch_pct/100 × demand_mwh / H / duration_hours
    # Normalized: revenue_per_mwh = revenue / demand_mwh

    cap_price = CAPACITY_MARKET_PRICES.get(iso, 0)  # $/kW-yr

    # Convert dispatch_pct to energy dispatched (MWh/yr)
    bat4_energy = bat4_pct / 100.0 * demand_mwh
    bat8_energy = bat8_pct / 100.0 * demand_mwh
    ldes_energy = ldes_pct / 100.0 * demand_mwh
    h2_energy = h2_pct / 100.0 * demand_mwh

    # Power capacity (MW) = energy / (cycles × duration × RTE)
    # For daily cycle: cycles = 365, energy/cycle = total_energy/365
    bat4_mw = bat4_energy / (365 * BATTERY_DURATION_HOURS * BATTERY_EFFICIENCY)
    bat8_mw = bat8_energy / (365 * BATTERY8_DURATION_HOURS * BATTERY8_EFFICIENCY)
    # LDES: 52 cycles/yr (weekly)
    ldes_mw = ldes_energy / (52 * LDES_DURATION_HOURS * LDES_EFFICIENCY)
    # H2: 12 cycles/yr (monthly)
    h2_cycles = max(BATTERY_DEGRADATION['h2']['cycles_per_year'], 1)
    h2_mw = np.where(h2_energy > 0,
                     h2_energy / (h2_cycles * H2_DURATION_HOURS * H2_EFFICIENCY),
                     0.0)

    # Capacity revenue ($/yr)
    cap_rev = (
        bat4_mw * PEAK_CAPACITY_CREDITS.get('battery', 0.95) * cap_price * 1000 +
        bat8_mw * PEAK_CAPACITY_CREDITS.get('battery8', 0.95) * cap_price * 1000 +
        ldes_mw * PEAK_CAPACITY_CREDITS.get('ldes', 0.90) * cap_price * 1000 +
        h2_mw * PEAK_CAPACITY_CREDITS.get('h2', 0.85) * cap_price * 1000
    )
    # Normalize to $/MWh of demand
    capacity_rev_per_mwh = np.where(demand_mwh > 0, cap_rev / demand_mwh, 0.0)

    # ── 2. ENERGY ARBITRAGE REVENUE ──
    # Revenue = cycles × duration × RTE × spread × MW
    # For battery: buy at offpeak, sell at peak, earn spread × RTE
    # Using LMP peak-offpeak spread as proxy

    # Build spread array via INDEX MAPPING — O(unique_combos) lookups, not O(n).
    # There are at most ~63 unique (threshold, fuel_level) combos even with
    # 122k+ rows, so the expensive dict lookups happen <100 times.
    if lmp_spreads is not None:
        # Extract fuel level index from scenario key.
        # Scenario format: "LMLH_M_M_H1_X" — fuel char at position after 4th '_'
        # Map: L=0, M=1, H=2 → index into fuel_names
        fuel_names = ['Low', 'Medium', 'High']
        fuel_char_to_idx = {'L': 0, 'M': 1, 'H': 2}

        # Parse fuel index per unique scenario (not per row)
        unique_scenarios = list(set(scenarios))
        scen_to_fuel_idx = {}
        for s in unique_scenarios:
            parts = s.split('_') if isinstance(s, str) else []
            scen_to_fuel_idx[s] = fuel_char_to_idx.get(parts[4], 1) if len(parts) > 4 else 1

        # Build fuel_idx array via dict lookup on scenarios list
        fuel_idx = np.array([scen_to_fuel_idx.get(s, 1) for s in scenarios], dtype=np.int8)

        # Get unique thresholds
        unique_thresholds = np.unique(thresholds)

        # Build spread table: shape (n_thresholds, 3) indexed by [thr_idx, fuel_idx]
        spread_table = np.full((len(unique_thresholds), 3), 20.0, dtype=np.float64)
        thr_to_idx = {float(t): i for i, t in enumerate(unique_thresholds)}
        for ti, thr in enumerate(unique_thresholds):
            for fi, fname in enumerate(fuel_names):
                spread_table[ti, fi] = get_spread_for_row(lmp_spreads, float(thr), fname)

        # Map threshold values to indices (vectorized via searchsorted)
        thr_idx = np.searchsorted(unique_thresholds, thresholds)
        thr_idx = np.clip(thr_idx, 0, len(unique_thresholds) - 1)

        # Index into spread table — fully vectorized
        spread_arr = spread_table[thr_idx, fuel_idx]
    else:
        spread_arr = np.full(n, 20.0, dtype=np.float64)

    # Battery arbitrage: daily cycle, earn spread × RTE on each cycle
    bat4_arb = 365 * BATTERY_DURATION_HOURS * BATTERY_EFFICIENCY * spread_arr * bat4_mw
    bat8_arb = 365 * BATTERY8_DURATION_HOURS * BATTERY8_EFFICIENCY * spread_arr * bat8_mw
    # LDES: weekly cycle, spread is less predictable — use 50% of spread
    ldes_arb = 52 * LDES_DURATION_HOURS * LDES_EFFICIENCY * spread_arr * 0.5 * ldes_mw
    # H2: Seasonal, minimal arbitrage value
    h2_arb = h2_cycles * H2_DURATION_HOURS * H2_EFFICIENCY * spread_arr * 0.3 * h2_mw

    arb_rev = bat4_arb + bat8_arb + ldes_arb + h2_arb
    arbitrage_rev_per_mwh = np.where(demand_mwh > 0, arb_rev / demand_mwh, 0.0)

    # ── 3. ANCILLARY SERVICES REVENUE ──
    # Battery: regulation + spinning. LDES: spinning only.
    reg_rate = ANCILLARY_SERVICE_RATES['regulation'].get(iso, 10)
    spin_rate = ANCILLARY_SERVICE_RATES['spinning'].get(iso, 5)
    reg_hours = ANCILLARY_HOURS['regulation']
    spin_hours = ANCILLARY_HOURS['spinning']

    # Revenue = MW × rate × hours available
    ancillary_rev = (
        (bat4_mw + bat8_mw) * reg_rate * reg_hours +     # Battery regulation
        (bat4_mw + bat8_mw) * spin_rate * spin_hours +    # Battery spinning
        ldes_mw * spin_rate * spin_hours +                 # LDES spinning only
        h2_mw * spin_rate * spin_hours * 0.5               # H2 limited availability
    )
    ancillary_rev_per_mwh = np.where(demand_mwh > 0, ancillary_rev / demand_mwh, 0.0)

    # ── 4. DEGRADATION COST ──
    # Cost = (cycles_per_year / cycle_life) × replacement_fraction × CAPEX/yr
    # Since CAPEX is already in the LCOE cost, degradation is an ADDITIONAL cost
    # for augmentation/replacement beyond the initial 20-year asset life.
    # Normalized as: (annual_degradation_fraction) × storage_cost_component

    # We approximate degradation as a fraction of the storage portion of total_cost.
    # Storage cost fraction ≈ storage_energy / total_demand × LCOE_storage / effective_cost
    # For simplicity, use a flat $/MWh-discharged degradation adder.

    bat4_deg_params = BATTERY_DEGRADATION['battery']
    bat8_deg_params = BATTERY_DEGRADATION['battery8']
    ldes_deg_params = BATTERY_DEGRADATION['ldes']
    h2_deg_params = BATTERY_DEGRADATION['h2']

    # Degradation rate = cycles/year / cycle_life × replacement_fraction
    # This gives annual capacity loss cost as fraction of original CAPEX
    bat4_deg_rate = bat4_deg_params['cycles_per_year'] / bat4_deg_params['cycle_life_80pct'] * bat4_deg_params['replacement_fraction']
    bat8_deg_rate = bat8_deg_params['cycles_per_year'] / bat8_deg_params['cycle_life_80pct'] * bat8_deg_params['replacement_fraction']
    ldes_deg_rate = ldes_deg_params['cycles_per_year'] / ldes_deg_params['cycle_life_80pct'] * ldes_deg_params['replacement_fraction']
    h2_deg_rate = h2_deg_params['cycles_per_year'] / h2_deg_params['cycle_life_80pct'] * h2_deg_params['replacement_fraction']

    # Import LCOE tables for storage cost reference
    from step3_cost_optimization import LCOE_TABLES

    # Get Medium LCOE for storage ($/MWh-capacity)
    bat4_lcoe = LCOE_TABLES.get('battery', {}).get('Medium', {}).get(iso, 5.0)
    bat8_lcoe = LCOE_TABLES.get('battery8', {}).get('Medium', {}).get(iso, 4.0)
    ldes_lcoe = LCOE_TABLES.get('ldes', {}).get('Medium', {}).get(iso, 0.6)
    h2_lcoe = LCOE_TABLES.get('h2', {}).get('Medium', {}).get(iso, 2.8)

    # Annual degradation cost = deg_rate × LCOE × energy_dispatched
    deg_cost = (
        bat4_deg_rate * bat4_lcoe * bat4_energy +
        bat8_deg_rate * bat8_lcoe * bat8_energy +
        ldes_deg_rate * ldes_lcoe * ldes_energy +
        h2_deg_rate * h2_lcoe * h2_energy
    )
    degradation_cost_per_mwh = np.where(demand_mwh > 0, deg_cost / demand_mwh, 0.0)

    # ── NET ECONOMICS ──
    net_revenue = capacity_rev_per_mwh + arbitrage_rev_per_mwh + ancillary_rev_per_mwh
    net_cost_adjustment = net_revenue - degradation_cost_per_mwh

    economic_effective_cost = effective_cost - net_cost_adjustment
    economic_total_cost = total_cost - net_cost_adjustment

    return {
        'storage_capacity_rev_per_mwh': np.round(capacity_rev_per_mwh, 4).astype(np.float32),
        'storage_arbitrage_rev_per_mwh': np.round(arbitrage_rev_per_mwh, 4).astype(np.float32),
        'storage_ancillary_rev_per_mwh': np.round(ancillary_rev_per_mwh, 4).astype(np.float32),
        'storage_degradation_cost_per_mwh': np.round(degradation_cost_per_mwh, 4).astype(np.float32),
        'storage_net_revenue_per_mwh': np.round(net_cost_adjustment, 4).astype(np.float32),
        'economic_effective_cost': np.round(economic_effective_cost, 2).astype(np.float32),
        'economic_total_cost': np.round(economic_total_cost, 2).astype(np.float32),
        # Storage capacity in MW (for dashboard display)
        'storage_battery4_mw': np.round(bat4_mw, 1).astype(np.float32),
        'storage_battery8_mw': np.round(bat8_mw, 1).astype(np.float32),
        'storage_ldes_mw': np.round(ldes_mw, 1).astype(np.float32),
        'storage_h2_mw': np.round(h2_mw, 1).astype(np.float32),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_iso(iso):
    """Run economic assessment for one ISO."""
    t0 = time.time()

    # Load Step 3 cost data
    step3_path = os.path.join(STEP3_DIR, f'step3_co_{iso}.parquet')
    if not os.path.exists(step3_path):
        print(f"  {iso}: No Step 3 data found at {step3_path}")
        return None

    step3_table = pq.read_table(step3_path)
    n = step3_table.num_rows
    print(f"  {iso}: {n:,} Step 3 rows loaded")

    # Load LMP spreads
    lmp_spreads = load_lmp_spreads(iso)
    if lmp_spreads:
        print(f"  {iso}: {len(lmp_spreads):,} LMP spread entries loaded")
    else:
        print(f"  {iso}: Using default LMP spreads")

    # Compute storage economics (vectorized)
    econ = compute_storage_economics(iso, step3_table, lmp_spreads)

    # Augment the Step 3 table with economic columns
    new_cols = {}
    for name in step3_table.column_names:
        new_cols[name] = step3_table.column(name)
    for name, arr in econ.items():
        new_cols[name] = pa.array(arr)

    result_table = pa.table(new_cols)

    # Write output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{iso}_economic.parquet')
    pq.write_table(result_table, out_path, compression='zstd')
    file_kb = os.path.getsize(out_path) / 1024

    elapsed = time.time() - t0

    # Print summary stats
    eff = econ['economic_effective_cost']
    orig = step3_table.column('cost_effective_cost').to_numpy()
    net_rev = econ['storage_net_revenue_per_mwh']
    nonzero = net_rev[net_rev > 0.001]

    print(f"  {iso}: Written {n:,} rows ({file_kb:.0f} KB) in {elapsed:.1f}s")
    if len(nonzero) > 0:
        print(f"    Net storage revenue: "
              f"mean=${np.mean(nonzero):.3f}/MWh, "
              f"max=${np.max(nonzero):.3f}/MWh "
              f"({len(nonzero):,}/{n:,} rows with storage)")
        print(f"    Effective cost shift: "
              f"mean=${np.mean(orig - eff):.3f}/MWh, "
              f"max=${np.max(orig - eff):.3f}/MWh")
    else:
        print(f"    No storage revenue (all dispatch_pct = 0)")

    return result_table


def main():
    parser = argparse.ArgumentParser(
        description="Step 3 Economic Assessment: storage revenue stacking")
    parser.add_argument("--iso", default="ALL",
                        help="ISO name or 'ALL' (default: ALL)")
    parser.add_argument("--v2", action="store_true",
                        help="Use Step 1D.2 enhanced storage source "
                             "(reads from step2-ef-v2-parquets)")
    args = parser.parse_args()

    isos = ISOS if args.iso.upper() == 'ALL' else [args.iso.upper()]
    for iso in isos:
        if iso not in ISOS:
            print(f"ERROR: Unknown ISO '{iso}'")
            sys.exit(1)

    print("=" * 70)
    print("  STEP 3 ECONOMIC ASSESSMENT — Storage Revenue Stacking")
    print(f"  ISOs: {', '.join(isos)}")
    if args.v2:
        print(f"  Mode: Step 1D.2 enhanced storage (2050 caps)")
    print("=" * 70)

    total_start = time.time()
    for iso in isos:
        process_iso(iso)

    elapsed = time.time() - total_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)
    print("  ECONOMIC ASSESSMENT COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
