#!/usr/bin/env python3
"""Pool-gap diagnostic & EF augmentation for Step 2.3.

Two modes:
  --diagnose   Scan every ISO × endpoint × year and report band windows
               where the candidate pool is sparse (< MIN_CANDIDATES rows).
               Outputs JSON report + console summary.

  --augment    For each sparse window found by the diagnostic, interpolate
               between the two neighbouring EF bands to synthesise new
               candidates and write them as *_interp_part*.parquet sidecars.

  --all        Run diagnose then augment.

The solver's band window is  [cfe_tgt_yi − 0.5,  cfe_ceil)  where cfe_ceil
is constant = _next_band_ceiling(endpoint_pct).  A window is "sparse" when
fewer than MIN_CANDIDATES rows from the full EF pool fall inside it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── Import from the optimizer so we use the exact same constants ────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Lightweight imports — only what we need for the diagnostic
_ROOT = _HERE.parent if (_HERE.parent / 'data').exists() else _HERE.parent
sys.path.insert(0, str(_ROOT / 'scripts'))
import pipeline_config as pc

ISOS = list(pc.ISOS)
_ISO_FILTER = os.environ.get('ISO_FILTER', '').strip().upper()
if _ISO_FILTER:
    ISOS = [_ISO_FILTER]

BASE_YEAR, END_YEAR = 2025, 2050
YEARS = list(range(BASE_YEAR, END_YEAR + 1))
N_YEARS = len(YEARS)

ENDPOINT_TO_THRESHOLD = {0.90: 90, 0.95: 95, 0.99: 99, 0.999: 99.9}
EF_DIR = _ROOT / 'data' / 'step2.1-ef'
OUTPUT_DIR = _ROOT / 'data' / 'step2.3-pathway'

_EF_BAND_THRESHOLDS = (
    10, 20, 30, 40, 50, 55, 60, 65, 70, 75,
    80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9,
)
_SORTED_BANDS = sorted(_EF_BAND_THRESHOLDS)

_DEFAULT_CFE_WAYPOINTS = ((2030, 50), (2035, 70), (2040, 90), (2045, 95))

# Minimum candidates in a band window before we flag it as sparse
MIN_CANDIDATES = 50
# How many interpolated rows to generate per sparse window
INTERP_ROWS_PER_GAP = 500

_RESOURCE_COLS = (
    'clean_firm', 'solar', 'wind', 'hydro', 'offshore_wind', 'geothermal',
    'ccs_ccgt', 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8',
)
_STORAGE_COLS = (
    'battery_dispatch_pct', 'battery8_dispatch_pct',
    'ldes_dispatch_pct', 'h2_dispatch_pct',
)
_ALL_MIX_COLS = _RESOURCE_COLS + _STORAGE_COLS + ('hourly_match_score',)


# ── Shared helpers (mirror optimizer logic exactly) ─────────────────────────

def _next_band_ceiling(target_pct: float) -> float:
    for b in _SORTED_BANDS:
        if b > target_pct + 0.01:
            return float(b)
    return 100.0


def _endpoint_year(pct: float) -> int:
    return int(pc.THRESHOLD_TARGET_YEARS[pct])


def _cfe_target(year: int, ep_pct: float,
                waypoints=_DEFAULT_CFE_WAYPOINTS,
                base_pct: float = 0.0) -> float:
    ey = _endpoint_year(ep_pct)
    pts = [(y, t) for y, t in [(BASE_YEAR, base_pct)] + list(waypoints)
           if y < ey] + [(ey, ep_pct)]
    if year <= pts[0][0]:
        return base_pct
    if year >= ey:
        return ep_pct
    for (y0, t0), (y1, t1) in zip(pts, pts[1:]):
        if y0 <= year <= y1:
            return min(ep_pct, t0 + (t1 - t0) * (year - y0) / (y1 - y0))
    return ep_pct


def _threshold_tag(pct) -> str:
    return str(int(pct)) if float(pct).is_integer() else str(pct)


# ── P1 pathway mask (mirrors optimizer exactly) ──────────────────────────

def _demand_twh_vec(iso: str, level: str = 'Medium') -> np.ndarray:
    """Reproduce demand_twh_vec from the optimizer."""
    bd = float(pc.REGIONAL_DEMAND_TWH[iso])
    g = pc.DEMAND_GROWTH_RATES[iso][level]
    return bd * (1.0 + g) ** np.arange(N_YEARS)


def _p1_mask_params(iso: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (pf_y, uc_pct_y) exactly as _build_ctx computes them.

    pf_y[yi]   = bl_twh / demand_twh[yi] * 100 + 0.01   (declining %)
    uc_pct_y   = UPRATE_CAP_TWH[iso] / demand_twh_vec[yi] * 100

    Existing nuclear is a fixed TWh output (base-year share × base-year
    demand).  Its percentage share declines as demand grows.
    """
    bp = float(pc.GRID_MIX_SHARES[iso].get('clean_firm', 0.0))
    bd = float(pc.REGIONAL_DEMAND_TWH[iso])
    bl_twh = bp / 100.0 * bd           # fixed TWh output
    uc = float(pc.UPRATE_CAP_TWH[iso])
    dv = _demand_twh_vec(iso)
    pf_y = bl_twh / np.maximum(dv, 1e-6) * 100.0 + 0.01
    uc_pct_y = uc / np.maximum(dv, 1e-6) * 100.0
    return pf_y, uc_pct_y


def _p1_cf_cap(pf: float, uc_pct_yi: float) -> float:
    """The clean_firm ceiling for a given year: pf + uc_pct_yi + 0.5."""
    return pf + uc_pct_yi + 0.5


def _apply_p1_mask(cf_arr: np.ndarray, ccs_arr: np.ndarray,
                   pf: float, uc_pct_yi: float) -> np.ndarray:
    """Boolean mask matching the optimizer's pathway mask for P1/P1a.

    clean_firm <= pf + uc_pct_yi + 0.5  AND  ccs_ccgt <= 0.5
    """
    cap = _p1_cf_cap(pf, uc_pct_yi)
    return (cf_arr <= cap) & (ccs_arr <= 0.5)


# ── EF pool loader (scores only, lightweight) ──────────────────────────────

def _load_scores(iso: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load hourly_match_score, clean_firm, and ccs_ccgt from all EF bands.

    Returns (scores, clean_firm, ccs_ccgt) — all float64 arrays of length N.
    """
    s_arrays, cf_arrays, ccs_arrays = [], [], []
    cols = ['hourly_match_score', 'clean_firm', 'ccs_ccgt']
    for t in _EF_BAND_THRESHOLDS:
        name = _threshold_tag(t)
        parts = sorted(EF_DIR.glob(f'step_2_1_EF_{iso}_{name}_part*.parquet'))
        if not parts:
            main = EF_DIR / f'step_2_1_EF_{iso}_{name}.parquet'
            if main.exists():
                parts = [main]
        for p in parts:
            tbl = pq.read_table(p, columns=cols)
            s_arrays.append(tbl.column('hourly_match_score').to_numpy())
            cf_arrays.append(tbl.column('clean_firm').to_numpy())
            ccs_arrays.append(tbl.column('ccs_ccgt').to_numpy()
                              if 'ccs_ccgt' in tbl.column_names
                              else np.zeros(tbl.num_rows))
    if not s_arrays:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    return (np.concatenate(s_arrays).astype(np.float64),
            np.concatenate(cf_arrays).astype(np.float64),
            np.concatenate(ccs_arrays).astype(np.float64))


def _load_full_pool(iso: str) -> pa.Table:
    """Load full EF pool as a single arrow table (for augmentation source)."""
    tables = []
    for t in _EF_BAND_THRESHOLDS:
        name = _threshold_tag(t)
        parts = sorted(EF_DIR.glob(f'step_2_1_EF_{iso}_{name}_part*.parquet'))
        if not parts:
            main = EF_DIR / f'step_2_1_EF_{iso}_{name}.parquet'
            if main.exists():
                parts = [main]
        for p in parts:
            tables.append(pq.read_table(p))
    if not tables:
        raise FileNotFoundError(f"No EF bands for {iso}")
    return pa.concat_tables(tables, promote_options='permissive')


# ── Band-level row counts (for per-band reporting) ─────────────────────────

def _band_row_counts(iso: str) -> dict[float, int]:
    """Count rows per original EF band threshold."""
    counts = {}
    for t in _EF_BAND_THRESHOLDS:
        name = _threshold_tag(t)
        parts = sorted(EF_DIR.glob(f'step_2_1_EF_{iso}_{name}_part*.parquet'))
        if not parts:
            main = EF_DIR / f'step_2_1_EF_{iso}_{name}.parquet'
            if main.exists():
                parts = [main]
        n = 0
        for p in parts:
            n += pq.ParquetFile(p).metadata.num_rows
        counts[t] = n
    return counts


# ── Diagnostic ──────────────────────────────────────────────────────────────

def diagnose(iso: str) -> dict:
    """For one ISO, scan every endpoint × year and report sparse windows.

    Reports both candidates_total (all rows in window) and candidates_p1
    (rows that also pass the P1 pathway mask).  A window is flagged sparse
    when candidates_p1 < MIN_CANDIDATES — because P1 is the binding constraint.
    """
    scores, cf_arr, ccs_arr = _load_scores(iso)
    if len(scores) == 0:
        return {'iso': iso, 'pool_size': 0, 'band_counts': {},
                'endpoints': {}, 'sparse_windows': []}

    pool_size = len(scores)
    band_counts = _band_row_counts(iso)
    base_clean_pct = sum(pc.GRID_MIX_SHARES.get(iso, {}).values())

    # P1 mask parameters (same as optimizer's _build_ctx)
    pf_y, uc_pct_y = _p1_mask_params(iso)

    endpoints_report = {}
    sparse_windows = []

    for ep_frac, ep_pct in ENDPOINT_TO_THRESHOLD.items():
        ceil = _next_band_ceiling(ep_pct)
        ey = _endpoint_year(ep_pct)
        eyi = YEARS.index(ey) if ey in YEARS else N_YEARS - 1

        year_details = []
        for yi in range(eyi + 1):
            year = YEARS[yi]
            tgt = _cfe_target(year, ep_pct, base_pct=base_clean_pct)
            lo = tgt - 0.5
            hi = ceil

            in_window = (scores >= lo) & (scores < hi)
            candidates_total = int(np.sum(in_window))

            # Apply P1 mask to the in-window rows
            pf_yi = float(pf_y[yi])
            uc_pct_yi = float(uc_pct_y[yi])
            p1_ok = _apply_p1_mask(cf_arr, ccs_arr, pf_yi, uc_pct_yi)
            candidates_p1 = int(np.sum(in_window & p1_ok))

            # P1 is the binding constraint — flag on P1 count
            is_sparse = candidates_p1 < MIN_CANDIDATES

            detail = {
                'year': year,
                'cfe_target': round(tgt, 2),
                'window_lo': round(lo, 2),
                'window_hi': round(hi, 2),
                'candidates_total': candidates_total,
                'candidates_p1': candidates_p1,
                'cf_cap': round(_p1_cf_cap(pf_yi, uc_pct_yi), 2),
                'sparse': is_sparse,
            }
            year_details.append(detail)

            if is_sparse:
                sparse_windows.append({
                    'iso': iso,
                    'endpoint_pct': ep_pct,
                    'year': year,
                    'window_lo': round(lo, 2),
                    'window_hi': round(hi, 2),
                    'candidates_total': candidates_total,
                    'candidates_p1': candidates_p1,
                    'cf_cap': round(_p1_cf_cap(pf_yi, uc_pct_yi), 2),
                })

        endpoints_report[str(ep_pct)] = {
            'ceiling': ceil,
            'endpoint_year': ey,
            'years': year_details,
        }

    return {
        'iso': iso,
        'pool_size': pool_size,
        'band_counts': {str(k): v for k, v in band_counts.items()},
        'endpoints': endpoints_report,
        'sparse_windows': sparse_windows,
    }


def run_diagnose(output_path: Path | None = None):
    """Run diagnostic across all ISOs, print summary, optionally save JSON."""
    all_results = {}
    total_sparse = 0

    for iso in ISOS:
        print(f"[diag] {iso}: loading pool...", end=' ', flush=True)
        result = diagnose(iso)
        all_results[iso] = result
        n_sparse = len(result['sparse_windows'])
        total_sparse += n_sparse
        print(f"pool={result['pool_size']:,}  sparse_windows={n_sparse}", flush=True)

        # Per-endpoint summary line
        for ep_key, ep_data in result['endpoints'].items():
            years = ep_data['years']
            sp = [y for y in years if y['sparse']]
            if sp:
                min_p1 = min(y['candidates_p1'] for y in sp)
                min_total = min(y['candidates_total'] for y in sp)
                year_range = f"{sp[0]['year']}-{sp[-1]['year']}"
                print(f"       ep{ep_key}: {len(sp)} sparse years "
                      f"({year_range}), min_p1={min_p1} "
                      f"(total={min_total})", flush=True)

    # Console summary table
    print(f"\n{'='*88}")
    print(f"POOL GAP DIAGNOSTIC SUMMARY  (sparse = P1-eligible < {MIN_CANDIDATES})")
    print(f"{'='*88}")
    print(f"{'ISO':<8} {'Pool':>10} {'Sparse':>8} "
          f"{'Worst P1':>10} {'Worst Total':>13} {'Worst Band':>14}")
    print(f"{'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*13} {'-'*14}")
    for iso in ISOS:
        r = all_results[iso]
        sw = r['sparse_windows']
        if sw:
            worst = min(sw, key=lambda x: x['candidates_p1'])
            worst_p1 = worst['candidates_p1']
            worst_total = worst['candidates_total']
            worst_band = f"{worst['window_lo']:.1f}-{worst['window_hi']:.1f}"
        else:
            worst_p1 = '-'
            worst_total = '-'
            worst_band = '-'
        print(f"{iso:<8} {r['pool_size']:>10,} {len(sw):>8} "
              f"{worst_p1!s:>10} {worst_total!s:>13} {worst_band:>14}")
    print(f"{'='*88}")
    print(f"Total sparse windows across all ISOs: {total_sparse}")

    # Band density table per ISO
    for iso in ISOS:
        r = all_results[iso]
        bc = r['band_counts']
        print(f"\n[{iso}] Band density:")
        print(f"  {'Band':>8} {'Rows':>10}")
        for b in _SORTED_BANDS:
            cnt = bc.get(str(int(b)) if float(b).is_integer() else str(b), 0)
            marker = '  ◄ EMPTY' if cnt == 0 else ('  ◄ thin' if cnt < 100 else '')
            print(f"  {b:>8} {cnt:>10,}{marker}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull report → {output_path}")

    return all_results


# ── Augmentation ────────────────────────────────────────────────────────────

def _find_neighbor_bands(target_lo: float, target_hi: float):
    """Find the two EF band thresholds that bracket the sparse window.
    Returns (lower_band, upper_band) or None if at the edges."""
    lower, upper = None, None
    for b in _SORTED_BANDS:
        if b <= target_lo:
            lower = b
        if b >= target_hi and upper is None:
            upper = b
    return lower, upper


def _interpolate_rows(pool: pa.Table, lo_score: float, hi_score: float,
                      n_rows: int, rng: np.random.Generator,
                      cf_cap_pct: float | None = None) -> pa.Table:
    """Generate n_rows interpolated EF candidates with hourly_match_score
    uniformly distributed in [lo_score, hi_score).

    If cf_cap_pct is provided (P1 mode):
      - Source rows are pre-filtered to P1-eligible (clean_firm <= cap, ccs_ccgt <= 0.5)
      - Output rows have ccs_ccgt forced to 0
      - Output clean_firm is clamped to cf_cap_pct

    Strategy: pick pairs of rows from the pool — one from just below the
    window and one from just above — and linearly interpolate their mix
    columns.  The interpolation weight is chosen so the resulting
    hourly_match_score lands inside the target window.
    """
    scores = pool.column('hourly_match_score').to_numpy().astype(np.float64)

    # ── P1 pre-filter: restrict source rows to P1-eligible ──
    if cf_cap_pct is not None:
        cf_raw = pool.column('clean_firm').to_numpy().astype(np.float64)
        ccs_raw = (pool.column('ccs_ccgt').to_numpy().astype(np.float64)
                   if 'ccs_ccgt' in pool.column_names
                   else np.zeros(len(scores)))
        p1_ok = (cf_raw <= cf_cap_pct) & (ccs_raw <= 0.5)
    else:
        p1_ok = np.ones(len(scores), dtype=bool)

    # Rows in the band just below the window (within 5 pp), P1-eligible
    below_mask = (scores >= lo_score - 5.0) & (scores < lo_score) & p1_ok
    # Rows in the band just above the window (within 5 pp), P1-eligible
    above_mask = (scores >= hi_score) & (scores < hi_score + 5.0) & p1_ok

    below_idx = np.where(below_mask)[0]
    above_idx = np.where(above_mask)[0]

    if len(below_idx) == 0 or len(above_idx) == 0:
        # Fallback: widen the search to ±10 pp (still P1-filtered)
        below_mask = (scores >= lo_score - 10.0) & (scores < lo_score) & p1_ok
        above_mask = (scores >= hi_score) & (scores < hi_score + 10.0) & p1_ok
        below_idx = np.where(below_mask)[0]
        above_idx = np.where(above_mask)[0]

    if len(below_idx) == 0 or len(above_idx) == 0:
        tag = " (P1-filtered)" if cf_cap_pct is not None else ""
        print(f"  [augment] WARNING: no neighbor rows{tag} for "
              f"[{lo_score:.1f}, {hi_score:.1f}), skipping", flush=True)
        return None

    # Collect numeric columns we can interpolate
    mix_cols = [c for c in pool.column_names
                if c in _ALL_MIX_COLS or c.startswith('mix_')]
    numeric_cols = []
    for c in mix_cols:
        try:
            pool.column(c).to_numpy().astype(np.float64)
            numeric_cols.append(c)
        except (ValueError, TypeError):
            continue

    # Build source matrices
    below_mat = np.column_stack(
        [pool.column(c).to_numpy().astype(np.float64)[below_idx]
         for c in numeric_cols])
    above_mat = np.column_stack(
        [pool.column(c).to_numpy().astype(np.float64)[above_idx]
         for c in numeric_cols])

    # Score columns for computing the interpolation weight
    score_col_idx = numeric_cols.index('hourly_match_score')

    # Sample pairs and interpolate
    bi = rng.integers(0, len(below_idx), size=n_rows)
    ai = rng.integers(0, len(above_idx), size=n_rows)
    target_scores = rng.uniform(lo_score, hi_score, size=n_rows)

    b_rows = below_mat[bi]  # (n_rows, D)
    a_rows = above_mat[ai]  # (n_rows, D)
    b_scores = b_rows[:, score_col_idx]
    a_scores = a_rows[:, score_col_idx]

    # Weight so that interpolated score ≈ target_score
    denom = a_scores - b_scores
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    alpha = np.clip((target_scores - b_scores) / denom, 0.0, 1.0)

    interp = b_rows * (1.0 - alpha[:, None]) + a_rows * alpha[:, None]

    # Force the score column to be exactly the target
    interp[:, score_col_idx] = target_scores

    # Round percentage columns to integers (like original EF)
    pct_cols_idx = [i for i, c in enumerate(numeric_cols)
                    if c in _RESOURCE_COLS or c in _STORAGE_COLS]
    for ci in pct_cols_idx:
        interp[:, ci] = np.round(interp[:, ci]).astype(np.float64)

    # ── P1 post-enforcement: clamp clean_firm, zero ccs_ccgt ──
    if cf_cap_pct is not None:
        if 'ccs_ccgt' in numeric_cols:
            ccs_ci = numeric_cols.index('ccs_ccgt')
            interp[:, ccs_ci] = 0.0
        if 'clean_firm' in numeric_cols:
            cf_ci = numeric_cols.index('clean_firm')
            interp[:, cf_ci] = np.minimum(interp[:, cf_ci],
                                          np.floor(cf_cap_pct))

    # Rebuild the hourly_match_score from rounded resource shares
    # (sum of resource cols, clamped to [0, 100])
    resource_idx = [i for i, c in enumerate(numeric_cols) if c in _RESOURCE_COLS]
    if resource_idx:
        recomputed_score = np.clip(interp[:, resource_idx].sum(axis=1), 0.0, 100.0)
        # Only use recomputed if it's in-window; otherwise keep the target
        in_window = (recomputed_score >= lo_score) & (recomputed_score < hi_score)
        interp[:, score_col_idx] = np.where(in_window, recomputed_score, target_scores)

    # Build arrow table
    out_cols = {}
    for i, c in enumerate(numeric_cols):
        if c in _RESOURCE_COLS or c in _STORAGE_COLS:
            out_cols[c] = pa.array(interp[:, i].astype(np.float32))
        else:
            out_cols[c] = pa.array(interp[:, i].astype(np.float32))

    # Copy non-numeric columns from a random source row (for schema compat)
    for c in pool.column_names:
        if c not in out_cols:
            # Repeat a single value from the below-band rows
            src_val = pool.column(c)[int(below_idx[0])].as_py()
            out_cols[c] = pa.array([src_val] * n_rows)

    return pa.table(out_cols)


def _interpolate_p1_rows(pool: pa.Table, lo_score: float, hi_score: float,
                         n_rows: int, rng: np.random.Generator,
                         pf: float, uc_pct_yi: float) -> pa.Table | None:
    """Generate P1-eligible interpolated rows.

    Differences from _interpolate_rows:
      1. Source rows are pre-filtered to P1-eligible only
      2. Output ccs_ccgt is forced to 0
      3. Output clean_firm is clamped to the P1 cap (pf + uc_pct_yi + 0.5)

    The cf_cap_pct passed via uc_pct_yi should be the most permissive
    (earliest year → highest uc_pct) for the window, so the generated
    rows are valid for the widest range of years.
    """
    cf_cap = pf + uc_pct_yi + 0.5

    # ── Step 1: filter pool to P1-eligible rows only ──
    cf_arr = pool.column('clean_firm').to_numpy().astype(np.float64)
    ccs_arr = pool.column('ccs_ccgt').to_numpy().astype(np.float64)
    p1_ok = _apply_p1_mask(cf_arr, ccs_arr, pf, uc_pct_yi)
    p1_idx = np.where(p1_ok)[0]
    if len(p1_idx) == 0:
        print(f"  [p1-augment] WARNING: zero P1-eligible rows in entire pool, "
              f"cannot interpolate", flush=True)
        return None
    pool_p1 = pool.take(p1_idx)

    # ── Step 2: standard interpolation on P1-filtered pool ──
    scores = pool_p1.column('hourly_match_score').to_numpy().astype(np.float64)

    below_mask = (scores >= lo_score - 5.0) & (scores < lo_score)
    above_mask = (scores >= hi_score) & (scores < hi_score + 5.0)
    below_idx = np.where(below_mask)[0]
    above_idx = np.where(above_mask)[0]

    if len(below_idx) == 0 or len(above_idx) == 0:
        below_mask = (scores >= lo_score - 10.0) & (scores < lo_score)
        above_mask = (scores >= hi_score) & (scores < hi_score + 10.0)
        below_idx = np.where(below_mask)[0]
        above_idx = np.where(above_mask)[0]

    if len(below_idx) == 0 or len(above_idx) == 0:
        print(f"  [p1-augment] WARNING: no P1-eligible neighbor rows for "
              f"[{lo_score:.1f}, {hi_score:.1f}), skipping", flush=True)
        return None

    mix_cols = [c for c in pool_p1.column_names
                if c in _ALL_MIX_COLS or c.startswith('mix_')]
    numeric_cols = []
    for c in mix_cols:
        try:
            pool_p1.column(c).to_numpy().astype(np.float64)
            numeric_cols.append(c)
        except (ValueError, TypeError):
            continue

    below_mat = np.column_stack(
        [pool_p1.column(c).to_numpy().astype(np.float64)[below_idx]
         for c in numeric_cols])
    above_mat = np.column_stack(
        [pool_p1.column(c).to_numpy().astype(np.float64)[above_idx]
         for c in numeric_cols])

    score_col_idx = numeric_cols.index('hourly_match_score')

    bi = rng.integers(0, len(below_idx), size=n_rows)
    ai = rng.integers(0, len(above_idx), size=n_rows)
    target_scores = rng.uniform(lo_score, hi_score, size=n_rows)

    b_rows = below_mat[bi]
    a_rows = above_mat[ai]
    b_scores = b_rows[:, score_col_idx]
    a_scores = a_rows[:, score_col_idx]

    denom = a_scores - b_scores
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    alpha = np.clip((target_scores - b_scores) / denom, 0.0, 1.0)

    interp = b_rows * (1.0 - alpha[:, None]) + a_rows * alpha[:, None]
    interp[:, score_col_idx] = target_scores

    # ── Step 3: enforce P1 constraints on output ──
    # Force ccs_ccgt = 0
    if 'ccs_ccgt' in numeric_cols:
        ccs_idx = numeric_cols.index('ccs_ccgt')
        interp[:, ccs_idx] = 0.0

    # Clamp clean_firm to the P1 cap
    if 'clean_firm' in numeric_cols:
        cf_idx = numeric_cols.index('clean_firm')
        interp[:, cf_idx] = np.minimum(interp[:, cf_idx], cf_cap)

    # Round percentage columns
    pct_cols_idx = [i for i, c in enumerate(numeric_cols)
                    if c in _RESOURCE_COLS or c in _STORAGE_COLS]
    for ci in pct_cols_idx:
        interp[:, ci] = np.round(interp[:, ci]).astype(np.float64)

    # Recompute score from rounded shares
    resource_idx = [i for i, c in enumerate(numeric_cols) if c in _RESOURCE_COLS]
    if resource_idx:
        recomputed_score = np.clip(interp[:, resource_idx].sum(axis=1), 0.0, 100.0)
        in_window = (recomputed_score >= lo_score) & (recomputed_score < hi_score)
        interp[:, score_col_idx] = np.where(in_window, recomputed_score, target_scores)

    # ── Step 4: final P1 validation — drop any row that drifted past cap ──
    if 'clean_firm' in numeric_cols and 'ccs_ccgt' in numeric_cols:
        cf_idx = numeric_cols.index('clean_firm')
        ccs_idx = numeric_cols.index('ccs_ccgt')
        valid = (interp[:, cf_idx] <= cf_cap + 0.01) & (interp[:, ccs_idx] <= 0.5)
        interp = interp[valid]
        if len(interp) == 0:
            print(f"  [p1-augment] WARNING: all interpolated rows failed P1 "
                  f"validation for [{lo_score:.1f}, {hi_score:.1f})", flush=True)
            return None

    # Build arrow table
    out_cols = {}
    for i, c in enumerate(numeric_cols):
        out_cols[c] = pa.array(interp[:, i].astype(np.float32))

    for c in pool_p1.column_names:
        if c not in out_cols:
            src_val = pool_p1.column(c)[int(below_idx[0])].as_py()
            out_cols[c] = pa.array([src_val] * len(interp))

    return pa.table(out_cols)


def run_augment(diag_results: dict, dry_run: bool = False):
    """Generate interpolated EF rows for each sparse window.

    For windows that are sparse only due to the P1 mask (candidates_total
    is fine but candidates_p1 < MIN_CANDIDATES), generates P1-eligible
    rows using _interpolate_p1_rows which:
      - Uses only P1-eligible source rows
      - Forces ccs_ccgt = 0
      - Clamps clean_firm to the P1 cap
    """
    rng = np.random.default_rng(42)

    # Deduplicate windows: group by (iso, window_lo, window_hi).
    # Track whether each window needs general or P1-specific augmentation.
    from dataclasses import dataclass, field

    @dataclass
    class _Window:
        lo: float; hi: float
        needs_general: bool = False   # total candidates < MIN
        needs_p1: bool = False        # P1 candidates < MIN
        p1_cf_cap: float = 999.0      # most permissive (largest) cap across years
        year_indices: list = field(default_factory=list)

    windows_by_iso: dict[str, dict[tuple[float, float], _Window]] = defaultdict(dict)

    for iso, result in diag_results.items():
        for sw in result['sparse_windows']:
            key = (sw['window_lo'], sw['window_hi'])
            if key not in windows_by_iso[iso]:
                windows_by_iso[iso][key] = _Window(lo=sw['window_lo'],
                                                    hi=sw['window_hi'])
            w = windows_by_iso[iso][key]
            if sw['candidates_total'] < MIN_CANDIDATES:
                w.needs_general = True
            if sw['candidates_p1'] < MIN_CANDIDATES:
                w.needs_p1 = True
            # Keep the most permissive (largest) P1 cap — earliest year has
            # highest uc_pct because demand is smallest.
            w.p1_cf_cap = max(w.p1_cf_cap, sw.get('cf_cap', 999.0))
            # Track year index for selecting uc_pct_y
            yi = YEARS.index(sw['year']) if sw['year'] in YEARS else 0
            w.year_indices.append(yi)

    total_written = 0
    for iso, wdict in windows_by_iso.items():
        windows = list(wdict.values())
        if not windows:
            continue

        n_general = sum(1 for w in windows if w.needs_general)
        n_p1 = sum(1 for w in windows if w.needs_p1 and not w.needs_general)
        print(f"\n[augment] {iso}: {len(windows)} unique sparse windows "
              f"({n_general} general, {n_p1} P1-only)", flush=True)

        if dry_run:
            for w in sorted(windows, key=lambda x: x.lo):
                kind = 'general+P1' if w.needs_general and w.needs_p1 else (
                    'P1-only' if w.needs_p1 else 'general')
                print(f"  [dry-run] would augment [{w.lo:.1f}, {w.hi:.1f}) "
                      f"[{kind}]  cf_cap={w.p1_cf_cap:.1f}", flush=True)
            continue

        # Load full pool once per ISO
        print(f"  loading full pool...", end=' ', flush=True)
        pool = _load_full_pool(iso)
        print(f"{pool.num_rows:,} rows", flush=True)

        pf_y, uc_pct_y = _p1_mask_params(iso)

        for w in sorted(windows, key=lambda x: x.lo):
            lower_band, upper_band = _find_neighbor_bands(w.lo, w.hi)
            if lower_band is None or upper_band is None:
                print(f"  [{w.lo:.1f}, {w.hi:.1f}): no bracketing bands, skip",
                      flush=True)
                continue

            band_tag = _threshold_tag(lower_band)

            # General augmentation (if total pool is sparse)
            if w.needs_general:
                interp_tbl = _interpolate_rows(pool, w.lo, w.hi,
                                               INTERP_ROWS_PER_GAP, rng)
                if interp_tbl is not None:
                    out_name = (f'step_2_1_EF_{iso}_{band_tag}'
                                f'_interp_{w.lo:.0f}_{w.hi:.0f}.parquet')
                    out_path = EF_DIR / out_name
                    pq.write_table(interp_tbl, out_path,
                                   compression='zstd', compression_level=3)
                    total_written += interp_tbl.num_rows
                    print(f"  [{w.lo:.1f}, {w.hi:.1f}): wrote "
                          f"{interp_tbl.num_rows} general rows → {out_name}",
                          flush=True)

            # P1 augmentation (if P1 pool is sparse)
            if w.needs_p1:
                # Use the most permissive year's params (earliest year in
                # window → highest pf and uc_pct because demand is smallest)
                earliest_yi = min(w.year_indices) if w.year_indices else 0
                pf_yi_max = float(pf_y[earliest_yi])
                uc_pct_yi_max = float(uc_pct_y[earliest_yi])

                p1_tbl = _interpolate_p1_rows(pool, w.lo, w.hi,
                                              INTERP_ROWS_PER_GAP, rng,
                                              pf_yi_max, uc_pct_yi_max)
                if p1_tbl is not None:
                    out_name = (f'step_2_1_EF_{iso}_{band_tag}'
                                f'_p1interp_{w.lo:.0f}_{w.hi:.0f}.parquet')
                    out_path = EF_DIR / out_name
                    pq.write_table(p1_tbl, out_path,
                                   compression='zstd', compression_level=3)
                    total_written += p1_tbl.num_rows
                    print(f"  [{w.lo:.1f}, {w.hi:.1f}): wrote "
                          f"{p1_tbl.num_rows} P1 rows → {out_name} "
                          f"(cf_cap={pf_yi_max + uc_pct_yi_max + 0.5:.1f})",
                          flush=True)

    print(f"\n[augment] Total rows written: {total_written:,}")
    return total_written


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='Diagnose EF pool gaps and optionally augment sparse bands')
    ap.add_argument('--diagnose', action='store_true',
                    help='Scan all ISOs and report sparse band windows')
    ap.add_argument('--augment', action='store_true',
                    help='Generate interpolated rows for sparse windows')
    ap.add_argument('--all', action='store_true',
                    help='Run diagnose then augment')
    ap.add_argument('--dry-run', action='store_true',
                    help='Show what augment would do without writing files')
    ap.add_argument('--min-candidates', type=int, default=MIN_CANDIDATES,
                    help=f'Sparsity threshold (default: {MIN_CANDIDATES})')
    ap.add_argument('--interp-rows', type=int, default=INTERP_ROWS_PER_GAP,
                    help=f'Rows to generate per gap (default: {INTERP_ROWS_PER_GAP})')
    ap.add_argument('--report', type=str, default=None,
                    help='Path to write JSON diagnostic report')
    args = ap.parse_args(argv)

    global MIN_CANDIDATES, INTERP_ROWS_PER_GAP
    MIN_CANDIDATES = args.min_candidates
    INTERP_ROWS_PER_GAP = args.interp_rows

    if args.all:
        args.diagnose = args.augment = True

    if not (args.diagnose or args.augment):
        print('Specify --diagnose, --augment, or --all', file=sys.stderr)
        return 2

    diag_results = None
    if args.diagnose or args.augment:
        report_path = Path(args.report) if args.report else (
            OUTPUT_DIR / 'pool_gap_diagnostic.json')
        diag_results = run_diagnose(report_path)

    if args.augment and diag_results:
        run_augment(diag_results, dry_run=args.dry_run)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
