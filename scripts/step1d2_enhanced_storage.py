#!/usr/bin/env python3
"""Step 1D.2: Enhanced storage sweep with 2050-oriented capacity caps.

Thin wrapper around step1d_fine_storage.py that overrides storage capacity
limits to research-informed 2050 projections while reusing the same three-pass
adaptive sweep architecture and Numba dispatch kernels.

Key differences from Step 1D:
  - Battery 4hr:  0.5% → 5.0%  of demand (NREL Storage Futures: 200 GW ref)
  - Battery 8hr:  1.0% → 10.0% of demand (DOE: 8hr key for deep decarb)
  - LDES:         5.0% → 25.0% of demand (DOE LDES Liftoff: 225-460 GW)
  - H2:          25.0% → 25.0% of demand (unchanged)

Output: data/step1d2-storage-parquets/{ISO}_t{XX}_storage.parquet

Uses the same near-miss cache from Step 1C (100k mixes/ISO). No 1A-1C
rerun needed — just expanded storage search space.

Usage:
  python scripts/step1d2_enhanced_storage.py --iso CAISO
  python scripts/step1d2_enhanced_storage.py --iso PJM --auto-commit
  python scripts/step1d2_enhanced_storage.py --iso ALL
  python scripts/step1d2_enhanced_storage.py --iso ERCOT --thresholds "90,95,99"
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import the base step1d module — we'll override its constants
import step1d_fine_storage as s1d

# ══════════════════════════════════════════════════════════════════════════════
# OVERRIDE CONSTANTS: 2050-oriented storage caps
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
V2_OUTPUT_DIR = os.path.join(DATA_DIR, 'step1d2-storage-parquets')

# Pass 0: Maximum storage levels (ceiling screen)
# 10× current for battery, 5× for LDES
V2_MAX_BAT4 = np.array([5.0], dtype=np.float64)
V2_MAX_BAT8 = np.array([10.0], dtype=np.float64)
V2_MAX_LDES = np.array([25.0], dtype=np.float64)
V2_MAX_H2 = np.array([25.0], dtype=np.float64)

# Pass 1: Extended coarse grids covering the wider range
# Log-spaced with denser coverage in the cost-meaningful low range
V2_FULL_BAT4 = np.array([0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0],
                         dtype=np.float64)
V2_FULL_BAT8 = np.array([0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                         dtype=np.float64)
V2_FULL_LDES = np.array([0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0],
                         dtype=np.float64)
V2_FULL_H2 = np.array([0, 5, 10, 15, 20, 25], dtype=np.float64)

# Wider gap buckets: with higher storage caps, larger gaps are reachable
V2_GAP_BUCKET_PP = [10, 25, 50, 100]


def apply_v2_overrides():
    """Monkey-patch step1d module constants with V2 values."""
    s1d.STEP1D_OUTPUT_DIR = V2_OUTPUT_DIR
    s1d.MAX_BAT4 = V2_MAX_BAT4
    s1d.MAX_BAT8 = V2_MAX_BAT8
    s1d.MAX_LDES = V2_MAX_LDES
    s1d.MAX_H2 = V2_MAX_H2
    s1d.FULL_BAT4 = V2_FULL_BAT4
    s1d.FULL_BAT8 = V2_FULL_BAT8
    s1d.FULL_LDES = V2_FULL_LDES
    s1d.FULL_H2 = V2_FULL_H2
    s1d.GAP_BUCKET_PP = V2_GAP_BUCKET_PP

    # Also update _build_bucket_grid's inline floor values
    # (the function references FULL_BAT4 etc. by module-level name,
    #  so the override above is sufficient — it reads s1d.FULL_BAT4)

    os.makedirs(V2_OUTPUT_DIR, exist_ok=True)
    print(f"  Step 1D.2 — Enhanced storage caps (2050-oriented)")
    print(f"  Battery 4hr max: {V2_MAX_BAT4[0]}%  (was 0.5%)")
    print(f"  Battery 8hr max: {V2_MAX_BAT8[0]}%  (was 1.0%)")
    print(f"  LDES max:        {V2_MAX_LDES[0]}%  (was 5.0%)")
    print(f"  H2 max:          {V2_MAX_H2[0]}%  (unchanged)")
    print(f"  Output: {V2_OUTPUT_DIR}")
    print(f"  Coarse grid: bat4={len(V2_FULL_BAT4)} levels, "
          f"bat8={len(V2_FULL_BAT8)}, ldes={len(V2_FULL_LDES)}, "
          f"h2={len(V2_FULL_H2)}")
    print(f"  Total coarse combos (excl H2): "
          f"{len(V2_FULL_BAT4) * len(V2_FULL_BAT8) * len(V2_FULL_LDES)}")
    print()


if __name__ == "__main__":
    apply_v2_overrides()
    s1d.main()
