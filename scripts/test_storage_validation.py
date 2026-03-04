#!/usr/bin/env python3
"""Quick validation: does storage at corrected scale provide incremental value?

CAISO 95% threshold, ~1000 archetypes, wide storage sweep.
Evaluates cost under Low VRE / Low TX / Low Battery / High Clean Firm.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pyarrow.parquet as pq
import step1_pfs_generator as s1

# ── Load CAISO profiles ──
print("Loading CAISO profiles...")
demand_data, gen_profiles, _, _ = s1.load_data()
demand_norm = demand_data['CAISO']['normalized']
supply_profiles = s1.get_supply_profiles('CAISO', gen_profiles)
demand_arr, supply_matrix = s1.prepare_numpy_profiles('CAISO', demand_norm, supply_profiles)

# ── Load coarse cache (wider score range for high-curtailment mixes) ──
print("Loading coarse cache...")
resource_types = s1.get_resource_types('CAISO')
cc_df = pq.read_table('data/step1-pfs-parquets/CAISO_coarse_cache.parquet').to_pandas()
nm_scores = cc_df['score'].values.astype(np.float64)
nm_combos = cc_df[resource_types].values.astype(np.float64)
n_mixes = len(nm_scores)
print(f"  Total mixes: {n_mixes:,}, score range: {nm_scores.min():.4f}-{nm_scores.max():.4f}")

# ── Sample ~1000 diverse archetypes spanning 75-95% base score ──
# Key insight: high-curtailment lean mixes (75-85%) might be cheap + storage bridges gap
np.random.seed(42)
# Test OUTSIDE the near-miss window (currently 20pp → 75-95% for target=95%)
# Key question: can mixes below 75% reach 95% with storage? If so, widen window.
band_55_65 = np.where((nm_scores >= 0.55) & (nm_scores < 0.65))[0]
band_65_75 = np.where((nm_scores >= 0.65) & (nm_scores < 0.75))[0]
band_75_80 = np.where((nm_scores >= 0.75) & (nm_scores < 0.80))[0]
band_80_85 = np.where((nm_scores >= 0.80) & (nm_scores < 0.85))[0]
band_85_90 = np.where((nm_scores >= 0.85) & (nm_scores < 0.90))[0]
band_90_95 = np.where((nm_scores >= 0.90) & (nm_scores < 0.95))[0]

print(f"  55-65%: {len(band_55_65):,}")
print(f"  65-75%: {len(band_65_75):,}")
print(f"  75-80%: {len(band_75_80):,}")
print(f"  80-85%: {len(band_80_85):,}")
print(f"  85-90%: {len(band_85_90):,}")
print(f"  90-95%: {len(band_90_95):,}")

# Sample 200 from each band — focus on OUTSIDE the current window
bands = [band_55_65, band_65_75, band_75_80, band_80_85, band_85_90, band_90_95]
per_band = 200
test_indices = []
for band in bands:
    n_take = min(per_band, len(band))
    if n_take > 0:
        test_indices.append(np.random.choice(band, n_take, replace=False))
test_indices = np.concatenate(test_indices)
np.random.shuffle(test_indices)

# Count by band
t_scores = nm_scores[test_indices]
print(f"  Test archetypes: {len(test_indices)}")
print(f"    55-65%: {((t_scores >= 0.55) & (t_scores < 0.65)).sum()}")
print(f"    65-75%: {((t_scores >= 0.65) & (t_scores < 0.75)).sum()}")
print(f"    75-80%: {((t_scores >= 0.75) & (t_scores < 0.80)).sum()}")
print(f"    80-85%: {((t_scores >= 0.80) & (t_scores < 0.85)).sum()}")
print(f"    85-90%: {((t_scores >= 0.85) & (t_scores < 0.90)).sum()}")
print(f"    90-95%: {((t_scores >= 0.90) & (t_scores < 0.95)).sum()}")

# ── Storage grids (new corrected values — % of annual demand) ──
bat4_grid = np.array([0, 0.005, 0.01, 0.02, 0.04, 0.06], dtype=np.float64)
bat8_grid = np.array([0, 0.01, 0.02, 0.04, 0.06, 0.08], dtype=np.float64)
ldes_grid = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5], dtype=np.float64)
h2_grid = np.array([0, 0.3, 0.5, 1.0, 2.0], dtype=np.float64)

n_b4, n_b8, n_l, n_h2 = len(bat4_grid), len(bat8_grid), len(ldes_grid), len(h2_grid)
n_combos = n_b4 * n_b8 * n_l * n_h2
print(f"  Storage combos: {n_combos} ({n_b4}×{n_b8}×{n_l}×{n_h2})")

# ── Storage constants ──
batt_eff = 0.85
batt8_eff = 0.85
ldes_eff = 0.50
h2_eff = 0.35
batt4_dur = 4.0
batt8_dur = 8.0
ldes_dur = 100.0
h2_dur = 1000.0
ldes_window = 7 * 24
h2_window = 30 * 24
batt8_window = 48

# ── JIT warmup ──
print("Warming up Numba JIT...")
dummy_supply = np.ones((1, 8760), dtype=np.float64)
dummy_b = np.array([0.0, 0.01], dtype=np.float64)
dummy_l = np.array([0.0], dtype=np.float64)
dummy_h = np.array([0.0], dtype=np.float64)
s1._batch_mixes_storage_screen(
    demand_arr, dummy_supply, 1.0, 1,
    dummy_b, dummy_b, dummy_l,
    2, 2, 1,
    batt_eff, batt8_eff, ldes_eff,
    batt4_dur, batt8_dur, ldes_dur,
    ldes_window, batt8_window,
    dummy_h, 1, h2_eff, h2_dur, float(h2_window))
print("JIT ready.")

# ── Run storage sweep ──
print(f"\nRunning storage sweep on {len(test_indices)} archetypes...")
t0 = time.time()

# Process in batches of 100
batch_size = 100
all_results = []  # (mix_idx, base_score, best_score, best_b4, best_b8, best_ldes, best_h2, no_storage_score)

for batch_start in range(0, len(test_indices), batch_size):
    batch_end = min(batch_start + batch_size, len(test_indices))
    b_idx = test_indices[batch_start:batch_end]
    n_batch = len(b_idx)

    # Build supply profiles for this batch
    batch_fracs = nm_combos[b_idx] / 100.0
    batch_supply = batch_fracs @ supply_matrix  # (n_batch, 8760)

    # Run PFS kernel with storage grid
    batch_scores = s1._batch_mixes_storage_screen(
        demand_arr, batch_supply, 1.0, n_batch,
        bat4_grid, bat8_grid, ldes_grid,
        n_b4, n_b8, n_l,
        batt_eff, batt8_eff, ldes_eff,
        batt4_dur, batt8_dur, ldes_dur,
        ldes_window, batt8_window,
        h2_grid, n_h2, h2_eff, h2_dur, float(h2_window))

    # batch_scores shape: (n_batch, n_combos)
    for i in range(n_batch):
        mix_idx = b_idx[i]
        base_score = nm_scores[mix_idx]
        scores = batch_scores[i]
        no_storage_score = scores[0]  # combo 0 = all zeros
        best_combo_idx = np.argmax(scores)
        best_score = scores[best_combo_idx]

        # Decode combo index to storage levels
        h2i = best_combo_idx % n_h2
        rem = best_combo_idx // n_h2
        li = rem % n_l
        rem = rem // n_l
        b8i = rem % n_b8
        b4i = rem // n_b8

        all_results.append((
            mix_idx, base_score, best_score,
            bat4_grid[b4i], bat8_grid[b8i], ldes_grid[li], h2_grid[h2i],
            no_storage_score
        ))

    if (batch_start + batch_size) % 500 == 0 or batch_end == len(test_indices):
        print(f"  Processed {batch_end}/{len(test_indices)} mixes ({time.time()-t0:.1f}s)")

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s\n")

# ── Analyze results ──
results = np.array([(r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in all_results])
base_scores = results[:, 0]
best_scores = results[:, 1]
best_b4 = results[:, 2]
best_b8 = results[:, 3]
best_ldes = results[:, 4]
best_h2 = results[:, 5]
no_storage = results[:, 6]

# 1. Saturation check — use base_score from cache (not -1 sentinel from kernel)
print("=" * 70)
print("1. SATURATION CHECK (does score increase with more storage?)")
print("=" * 70)
lift = best_scores - base_scores
print(f"  Score lift from storage: min={lift.min():.4f}, median={np.median(lift):.4f}, max={lift.max():.4f}")
print(f"  Mixes with >0 lift: {(lift > 0).sum()} / {len(lift)}")
print(f"  Mixes reaching 95%+ with storage: {(best_scores >= 0.95).sum()} / {len(best_scores)}")
print(f"  Mixes below 95% base that reach 95%+: {((base_scores < 0.95) & (best_scores >= 0.95)).sum()} / {(base_scores < 0.95).sum()}")
print(f"  By band:")
for lo, hi in [(0.55, 0.65), (0.65, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95)]:
    mask = (base_scores >= lo) & (base_scores < hi)
    n_band = mask.sum()
    if n_band > 0:
        reached = (mask & (best_scores >= 0.95)).sum()
        avg_lift = lift[mask].mean()
        print(f"    {lo*100:.0f}-{hi*100:.0f}%: {reached}/{n_band} reach 95% (avg lift: {avg_lift:.4f})")

# 2. Monotonic improvement check - pick a high-curtailment mix (~80% base)
print("\n" + "=" * 70)
print("2. MONOTONIC IMPROVEMENT CHECK (one example mix)")
print("=" * 70)
# Find a mix with ~70% base score to test wide storage bridging (outside window)
candidates = [(i, all_results[i]) for i in range(len(all_results))
              if 0.68 < all_results[i][1] < 0.72]
if not candidates:
    candidates = [(i, all_results[i]) for i in range(len(all_results))
                  if 0.75 < all_results[i][1] < 0.80]
if candidates:
    idx, result = candidates[0]
    mix_idx = all_results[idx][0]
    print(f"  Mix {mix_idx}: base_score={result[1]:.4f}")

    # Rescore this one mix at bat4-only levels (all other storage = 0)
    single_fracs = nm_combos[mix_idx:mix_idx+1] / 100.0
    single_supply = single_fracs @ supply_matrix

    for b4_val in bat4_grid:
        b4_test = np.array([b4_val], dtype=np.float64)
        b8_test = np.array([0.0], dtype=np.float64)
        l_test = np.array([0.0], dtype=np.float64)
        h2_test = np.array([0.0], dtype=np.float64)
        test_score = s1._batch_mixes_storage_screen(
            demand_arr, single_supply, 1.0, 1,
            b4_test, b8_test, l_test,
            1, 1, 1,
            batt_eff, batt8_eff, ldes_eff,
            batt4_dur, batt8_dur, ldes_dur,
            ldes_window, batt8_window,
            h2_test, 1, h2_eff, h2_dur, float(h2_window))
        energy_gwh = b4_val / 100 * 224000  # CAISO annual demand in GWh
        power_gw = energy_gwh / 4
        print(f"    bat4={b4_val:.3f}% ({energy_gwh:.0f} GWh / {power_gw:.1f} GW) → score={test_score[0,0]:.6f}")
else:
    print("  No candidates with base_score < 94% found")

# 3. Distribution of optimal storage levels
print("\n" + "=" * 70)
print("3. OPTIMAL STORAGE DISTRIBUTION (best combo per mix)")
print("=" * 70)
feasible = best_scores >= 0.95
print(f"  Feasible mixes (score ≥ 95%): {feasible.sum()}")
if feasible.sum() > 0:
    print(f"  bat4:  mean={best_b4[feasible].mean():.4f}%, median={np.median(best_b4[feasible]):.4f}%")
    print(f"  bat8:  mean={best_b8[feasible].mean():.4f}%, median={np.median(best_b8[feasible]):.4f}%")
    print(f"  LDES:  mean={best_ldes[feasible].mean():.4f}%, median={np.median(best_ldes[feasible]):.4f}%")
    print(f"  H2:    mean={best_h2[feasible].mean():.4f}%, median={np.median(best_h2[feasible]):.4f}%")

    # Histogram of bat4 levels
    print(f"\n  bat4 histogram (feasible mixes):")
    for level in bat4_grid:
        count = (best_b4[feasible] == level).sum()
        pct = count / feasible.sum() * 100
        bar = '#' * int(pct / 2)
        print(f"    {level:.3f}%: {count:4d} ({pct:5.1f}%) {bar}")

# 4. Cost comparison — find CHEAPEST feasible combo (not max-score)
print("\n" + "=" * 70)
print("4. COST COMPARISON (Low VRE / Low TX / Low Battery / High Firm)")
print("=" * 70)

# Import updated LCOE tables
from step3_cost_optimization import LCOE_TABLES, WHOLESALE_PRICES, FUEL_ADJUSTMENTS

iso = 'CAISO'
# Low VRE, Low TX, Low Battery, High Clean Firm
sol_price = LCOE_TABLES['solar']['Low'][iso] + 1  # Low TX solar
wnd_price = LCOE_TABLES['wind']['Low'][iso] + 4   # Low TX wind
osw_price = LCOE_TABLES['offshore_wind']['Low'][iso] + 10  # Low TX OSW
bat4_price = LCOE_TABLES['battery']['Low'][iso]
bat8_price = LCOE_TABLES['battery8']['Low'][iso]
ldes_price = LCOE_TABLES['ldes']['Low'][iso]
h2_price = 17344.80  # Low H2

# High clean firm = expensive nuclear/CCS
uprate_price = 40  # High uprate
nuclear_price = 140 + 6  # High nuclear + High TX
ccs_price = 116.5 + 4  # High CCS + High TX
remaining_price = min(nuclear_price, ccs_price)
geo_price = 110 + 6  # High geo + High TX

wholesale = max(5, WHOLESALE_PRICES[iso] + FUEL_ADJUSTMENTS[iso]['Low'])

# Existing resource shares for CAISO (approximate)
existing = {'clean_firm': 23, 'solar': 24, 'wind': 8, 'hydro': 18,
            'ccs_ccgt': 0, 'offshore_wind': 0, 'geothermal': 0}

print(f"\n  Prices: solar=${sol_price}, wind=${wnd_price}")
print(f"  bat4=${bat4_price:.0f}/%-ann, bat8=${bat8_price:.0f}/%-ann")
print(f"  LDES=${ldes_price:.0f}/%-ann, H2=${h2_price:.0f}/%-ann")
print(f"  nuclear=${nuclear_price}, ccs=${ccs_price}, geo=${geo_price}")

demand_twh = 224.0

def compute_cost(combo, b4, b8, ld, h2v):
    """Compute $/MWh cost for a mix + storage combo."""
    cf = combo[resource_types.index('clean_firm')]
    sol = combo[resource_types.index('solar')]
    wnd = combo[resource_types.index('wind')]
    hyd = combo[resource_types.index('hydro')]
    osw = combo[resource_types.index('offshore_wind')]
    geo = combo[resource_types.index('geothermal')]

    sol_new = max(0, sol - existing['solar']) / 100
    wnd_new = max(0, wnd - existing['wind']) / 100
    osw_new = max(0, osw - existing['offshore_wind']) / 100
    cf_new = max(0, cf - existing['clean_firm']) / 100

    new_cf_twh = cf_new * demand_twh
    uprate_twh = min(new_cf_twh, 1.45)
    rem = max(0, new_cf_twh - uprate_twh)
    geo_twh = min(rem, 39.0)
    rem = max(0, rem - geo_twh)

    exist_frac = (min(existing['clean_firm'], cf) / 100 +
                  min(existing['solar'], sol) / 100 +
                  min(existing['wind'], wnd) / 100 +
                  hyd / 100)

    return (exist_frac * wholesale +
            sol_new * sol_price + wnd_new * wnd_price + osw_new * osw_price +
            uprate_twh / demand_twh * uprate_price +
            geo_twh / demand_twh * geo_price +
            rem / demand_twh * remaining_price +
            b4 / 100 * bat4_price + b8 / 100 * bat8_price +
            ld / 100 * ldes_price + h2v / 100 * h2_price)

# Find cheapest mix+storage that reaches 95% by re-scanning all storage combos
print("\n  Searching for cheapest feasible storage combo per mix...")
best_cost = np.inf
best_mix_detail = None

for bi in range(0, len(test_indices), batch_size):
    be = min(bi + batch_size, len(test_indices))
    b_idx = test_indices[bi:be]
    n_batch = len(b_idx)

    batch_fracs = nm_combos[b_idx] / 100.0
    batch_supply = batch_fracs @ supply_matrix
    batch_scores = s1._batch_mixes_storage_screen(
        demand_arr, batch_supply, 1.0, n_batch,
        bat4_grid, bat8_grid, ldes_grid,
        n_b4, n_b8, n_l,
        batt_eff, batt8_eff, ldes_eff,
        batt4_dur, batt8_dur, ldes_dur,
        ldes_window, batt8_window,
        h2_grid, n_h2, h2_eff, h2_dur, float(h2_window))

    for i in range(n_batch):
        mix_idx = b_idx[i]
        combo = nm_combos[mix_idx]
        scores = batch_scores[i]

        # Find cheapest combo that scores >= 0.95
        for j in range(n_combos):
            if scores[j] < 0.95:
                continue
            # Decode storage levels
            h2i = j % n_h2
            rem = j // n_h2
            li = rem % n_l
            rem = rem // n_l
            b8i = rem % n_b8
            b4i = rem // n_b8

            b4, b8, ld, h2v = bat4_grid[b4i], bat8_grid[b8i], ldes_grid[li], h2_grid[h2i]
            cost = compute_cost(combo, b4, b8, ld, h2v)

            if cost < best_cost:
                best_cost = cost
                cf = combo[resource_types.index('clean_firm')]
                sol = combo[resource_types.index('solar')]
                wnd = combo[resource_types.index('wind')]
                hyd = combo[resource_types.index('hydro')]
                osw = combo[resource_types.index('offshore_wind')]
                geo = combo[resource_types.index('geothermal')]
                best_mix_detail = {
                    'mix_idx': mix_idx,
                    'cf': cf, 'sol': sol, 'wnd': wnd, 'hyd': hyd,
                    'osw': osw, 'geo': geo,
                    'bat4': b4, 'bat8': b8, 'ldes': ld, 'h2': h2v,
                    'score': scores[j],
                    'base_score': nm_scores[mix_idx],
                    'cost': cost,
                    'total_proc': cf + sol + wnd + hyd + osw + geo
                }

# Load current CAISO 95% winner from step3
print("\n  Loading current CAISO 95% winner from step3...")
import pandas as pd
step3_df = pd.read_parquet('data/step3-cost-opt-parquets/step3_co_CAISO.parquet')
t95 = step3_df[step3_df['threshold'] == 95]

# Find the Low VRE / Low TX / Low Battery / High Firm scenario
# Scenario naming: LLHL (vre_L, tx_L, battery_H?, firm_H) — need to check
# The scenario format is complex; let's just find the cheapest 95% mix
t95_sorted = t95.sort_values('cost_total_cost')
current_winner = t95_sorted.iloc[0]
print(f"  Current step3 cheapest 95% mix:")
print(f"    Scenario: {current_winner['scenario']}")
print(f"    Cost: ${current_winner['cost_total_cost']:.2f}/MWh")
print(f"    Mix: CF={current_winner['mix_clean_firm']}% Sol={current_winner['mix_solar']}%"
      f" Wnd={current_winner['mix_wind']}% Hyd={current_winner['mix_hydro']}%")
print(f"    Storage: bat4={current_winner['battery_dispatch_pct']:.4f}%"
      f" bat8={current_winner['battery8_dispatch_pct']:.4f}%"
      f" LDES={current_winner['ldes_dispatch_pct']:.4f}%")
print(f"    Score: {current_winner['hourly_match_score']:.2f}%")

# Also find the specific scenario: Low VRE, Low TX, Low Battery, High Firm
# Scenario encoding: first 4 chars = VRE+TX+Batt+Firm (L/M/H)
high_firm_scenarios = t95[t95['scenario'].str.contains('H', na=False)]
if len(high_firm_scenarios) > 0:
    # Find ones with 'L' for VRE and battery
    for _, row in t95_sorted.head(20).iterrows():
        sc = row['scenario']
        if sc.startswith('L') and 'H' in sc:
            print(f"\n  Relevant scenario match: {sc}")
            print(f"    Cost: ${row['cost_total_cost']:.2f}/MWh")
            print(f"    Mix: CF={row['mix_clean_firm']}% Sol={row['mix_solar']}%"
                  f" Wnd={row['mix_wind']}% Hyd={row['mix_hydro']}%")
            break

print(f"\n  BEST WITH STORAGE (from test):")
if best_mix_detail:
    m = best_mix_detail
    print(f"    Score: {m['score']:.4f} (base: {m['base_score']:.4f}, lift: {m['score']-m['base_score']:.4f})")
    print(f"    Cost:  ${m['cost']:.2f}/MWh")
    print(f"    Mix:   CF={m['cf']}% Sol={m['sol']}% Wnd={m['wnd']}% Hyd={m['hyd']}% OSW={m['osw']}% Geo={m['geo']}%")
    print(f"    Storage: bat4={m['bat4']:.3f}% bat8={m['bat8']:.3f}% LDES={m['ldes']:.3f}% H2={m['h2']:.3f}%")
    print(f"    Total procurement: {m['total_proc']:.0f}%")
    bat4_gwh = m['bat4'] / 100 * 224000
    bat4_gw = bat4_gwh / 4
    print(f"    bat4 physical: {bat4_gwh:.0f} GWh / {bat4_gw:.1f} GW")

    if m['base_score'] < 0.75:
        print(f"\n  ** THIS MIX IS OUTSIDE THE CURRENT NEAR-MISS WINDOW (base < 75%) **")
        print(f"  ** Near-miss window needs widening to capture this! **")
    elif m['base_score'] < 0.80:
        print(f"\n  ** This mix is at the edge of the near-miss window (75-80%) **")
else:
    print("    No feasible storage-enhanced mix found!")

print("\nDone.")
