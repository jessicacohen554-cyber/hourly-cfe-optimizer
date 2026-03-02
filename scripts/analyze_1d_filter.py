#!/usr/bin/env python3
"""Analyze curtailment-based filter for step 1D vs current dominance filter.

User's proposal:
  - Include mix if curtailment_pct_of_demand >= (target - score) + buffer
  - No dominance filter
  - "Buffer zone of up to 20% extra curtailment"

Runs on PJM 65% threshold as a case study, then shows multi-threshold comparison.
"""
import os
import sys
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import step1_pfs_generator as s1
from step1d_storage_refinement import (
    load_coarse_cache, get_near_miss_width, STORAGE_SWEEP_FLOOR,
    NM_CHUNK,
)
from step1c_zone_search import dominance_filter_arrays

ISO = 'PJM'
THRESHOLD = 65
TARGET = THRESHOLD / 100.0

print(f"Loading EIA data for {ISO}...")
demand_data, gen_profiles, _, _ = s1.load_data()
demand_norm = demand_data[ISO]["normalized"]
supply_profiles = s1.get_supply_profiles(ISO, gen_profiles)
demand_arr, supply_matrix = s1.prepare_numpy_profiles(
    ISO, demand_norm, supply_profiles)
demand_total = demand_arr.sum()

print(f"Loading coarse cache for {ISO}...")
combos, scores = load_coarse_cache(ISO)
print(f"  Total mixes: {len(combos):,}")
rtypes = s1.get_resource_types(ISO)
best_eff = max(s1.BATTERY_EFFICIENCY, s1.BATTERY8_EFFICIENCY, s1.LDES_EFFICIENCY)

# ── Compute surplus for ALL mixes once ──
print(f"\nComputing surplus for ALL {len(combos):,} mixes...")
all_surplus = np.empty(len(combos), dtype=np.float64)
for cs in range(0, len(combos), NM_CHUNK):
    ce = min(cs + NM_CHUNK, len(combos))
    chunk_fracs = combos[cs:ce].astype(np.float64) / 100.0
    chunk_supply = chunk_fracs @ supply_matrix
    chunk_surplus = np.maximum(chunk_supply - demand_arr[np.newaxis, :], 0.0)
    all_surplus[cs:ce] = chunk_surplus.sum(axis=1)

# Curtailment as % of demand (0-1 scale, so 0.20 = 20%)
all_curtailment_pct = all_surplus / demand_total
has_surplus = all_surplus > 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CURRENT APPROACH: near-miss window + dominance filter + bridge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
nm_width = get_near_miss_width(THRESHOLD)
near_miss_lower = max(TARGET - nm_width, STORAGE_SWEEP_FLOOR)
nm_mask = (scores >= near_miss_lower) & (scores < TARGET)
nm_idx = np.where(nm_mask)[0]

print(f"\n{'='*70}")
print(f"CURRENT APPROACH — PJM {THRESHOLD}%")
print(f"{'='*70}")
print(f"  Step 1: Near-miss window [{near_miss_lower*100:.0f}%, {TARGET*100:.0f}%): {len(nm_idx):,} mixes")

# Dominance filter
nm_combos_sub = combos[nm_idx]
nm_scores_sub = scores[nm_idx]
dom_mask = dominance_filter_arrays(nm_combos_sub, nm_scores_sub)
n_after_dom = dom_mask.sum()
print(f"  Step 2: After dominance filter: {n_after_dom:,} ({len(nm_idx) - n_after_dom:,} removed = {(len(nm_idx) - n_after_dom)/len(nm_idx)*100:.0f}%)")

# Physics bridge (on post-dominance set)
dom_idx = nm_idx[dom_mask]
dom_gap = TARGET - scores[dom_idx]
dom_max_lift = all_surplus[dom_idx] * best_eff / demand_total
dom_has_curt = all_surplus[dom_idx] > 0
dom_can_bridge = dom_max_lift >= dom_gap
dom_viable = dom_has_curt & dom_can_bridge
print(f"  Step 3: After bridge check: {dom_viable.sum():,} viable for storage sweep")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROPOSED APPROACH: curtailment % >= gap (user's formula)
# No dominance, no efficiency discount in the filter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*70}")
print(f"PROPOSED APPROACH — PJM {THRESHOLD}%")
print(f"curtailment_pct >= gap_to_target (+ buffer)")
print(f"{'='*70}")

below_target = scores < TARGET

for buffer_pct in [0, 5, 10, 15, 20, 25, 30]:
    buffer = buffer_pct / 100.0
    gap = TARGET - scores  # per-mix gap

    # User's filter: curtailment % of demand >= gap to target
    # With buffer: curtailment_pct >= gap - buffer
    # (buffer LOOSENS the filter — allows mixes with less curtailment than gap)
    # Wait, re-reading user: "buffer zone of up to 20% extra curtailment"
    # This means: curtailment_pct <= gap + buffer
    # NO — the user's examples show they want MORE curtailment included:
    # "mix B has 50% curtailment, scores 45%, target 75% → gap=30% → should be assessed"
    # 50% > 30% gap ✓, and 50% <= 30% + 20% buffer = 50% ✓
    # So buffer is: include mixes where curtailment >= gap AND curtailment <= gap + buffer
    # Wait that doesn't make sense either. Let me re-read...
    #
    # Actually: the buffer is about including mixes where curtailment
    # EXCEEDS the gap by up to 20%. Without buffer: curtailment must == gap.
    # With buffer: curtailment can be up to 20pp ABOVE the gap.
    #
    # NO. Re-reading again: "the precise window of mixes where CURTAILMENT
    # as a percentage of total demand is equal to or exceeds the delta"
    # "give it a buffer zone of up to like 20% extra curtailment"
    #
    # So the base rule is: curtailment_pct >= gap
    # The buffer means: also include mixes with curtailment UP TO gap + 20%
    # But that's already included in curtailment >= gap!
    #
    # I think the buffer means we ALSO want to cap the upper end:
    # curtailment_pct <= gap + buffer
    # This prevents over-procured mixes (way too much curtailment) from flooding in.
    #
    # Let me test both interpretations:
    # A) curtailment >= gap (lower bound only — no cap)
    # B) gap <= curtailment <= gap + buffer (windowed)

    # Interpretation A: curtailment >= gap (include all mixes with enough curtailment)
    mask_a = below_target & has_surplus & (all_curtailment_pct >= gap)
    n_a = mask_a.sum()

    # Interpretation B: gap <= curtailment <= gap + buffer (windowed)
    mask_b = below_target & has_surplus & (all_curtailment_pct >= gap) & (all_curtailment_pct <= gap + buffer)
    n_b = mask_b.sum()

    # Interpretation C: curtailment >= gap - buffer (loosened lower bound)
    # This would include mixes with LESS curtailment than the gap
    mask_c = below_target & has_surplus & (all_curtailment_pct >= (gap - buffer))
    n_c = mask_c.sum()

    if buffer_pct == 0:
        print(f"\n  Base (no buffer): curtailment >= gap")
        print(f"    {n_a:,} mixes")
        if n_a > 0:
            s = scores[mask_a]
            c = all_curtailment_pct[mask_a]
            print(f"    Score range: [{s.min()*100:.1f}%, {s.max()*100:.1f}%]")
            print(f"    Curtailment: [{c.min()*100:.1f}%, {c.max()*100:.1f}%]")
    else:
        print(f"\n  Buffer = {buffer_pct}%:")
        print(f"    A) curtailment >= gap (no upper cap):           {n_a:,} mixes")
        print(f"    B) gap <= curtailment <= gap+{buffer_pct}% (windowed): {n_b:,} mixes")
        print(f"    C) curtailment >= gap-{buffer_pct}% (loosened):        {n_c:,} mixes")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MULTI-THRESHOLD COMPARISON (current vs proposed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*70}")
print(f"MULTI-THRESHOLD COMPARISON — PJM")
print(f"Current (dominance) vs Proposed (curtailment >= gap, windowed ±20%)")
print(f"{'='*70}")
print(f"{'Thresh':>7} {'Current':>9} {'Proposed':>9} {'Ratio':>7} {'Score range':>18} {'Curt range':>18}")
print(f"{'-'*7:>7} {'-'*9:>9} {'-'*9:>9} {'-'*7:>7} {'-'*18:>18} {'-'*18:>18}")

buffer = 0.20
for threshold in [55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95]:
    target = threshold / 100.0
    gap = target - scores
    bt = scores < target

    # Current: near-miss + dominance + bridge
    nm_w = get_near_miss_width(threshold)
    nm_lo = max(target - nm_w, STORAGE_SWEEP_FLOOR)
    nm_m = (scores >= nm_lo) & (scores < target)
    nm_i = np.where(nm_m)[0]
    if len(nm_i) > 0:
        nm_c = combos[nm_i]
        nm_s = scores[nm_i]
        d_mask = dominance_filter_arrays(nm_c, nm_s)
        d_idx = nm_i[d_mask]
        d_gap = target - scores[d_idx]
        d_lift = all_surplus[d_idx] * best_eff / demand_total
        d_viable = (all_surplus[d_idx] > 0) & (d_lift >= d_gap)
        n_current = d_viable.sum()
    else:
        n_current = 0

    # Proposed: curtailment-based windowed filter
    # curtailment >= gap AND curtailment <= gap + buffer
    # Also keep the physics bridge check (curtailment * eff >= gap) as a safety net
    curt_sufficient = all_curtailment_pct >= gap
    bridge_possible = (all_surplus * best_eff / demand_total) >= gap
    upper_cap = all_curtailment_pct <= (gap + buffer)
    proposed_mask = bt & has_surplus & curt_sufficient & upper_cap & bridge_possible
    n_proposed = proposed_mask.sum()

    ratio = n_proposed / n_current if n_current > 0 else float('inf')

    # Stats on proposed
    if n_proposed > 0:
        ps = scores[proposed_mask]
        pc = all_curtailment_pct[proposed_mask]
        score_str = f"[{ps.min()*100:.0f}%-{ps.max()*100:.0f}%]"
        curt_str = f"[{pc.min()*100:.0f}%-{pc.max()*100:.0f}%]"
    else:
        score_str = "—"
        curt_str = "—"

    print(f"{threshold:>6}% {n_current:>9,} {n_proposed:>9,} {ratio:>6.1f}x {score_str:>18} {curt_str:>18}")

print(f"\nDone.")
