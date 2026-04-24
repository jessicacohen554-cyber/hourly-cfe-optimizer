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


# ── EF pool loader (scores only, lightweight) ──────────────────────────────

def _load_scores(iso: str) -> np.ndarray:
    """Load only hourly_match_score from all EF bands — minimal memory."""
    arrays = []
    for t in _EF_BAND_THRESHOLDS:
        name = _threshold_tag(t)
        parts = sorted(EF_DIR.glob(f'step_2_1_EF_{iso}_{name}_part*.parquet'))
        if not parts:
            main = EF_DIR / f'step_2_1_EF_{iso}_{name}.parquet'
            if main.exists():
                parts = [main]
        for p in parts:
            tbl = pq.read_table(p, columns=['hourly_match_score'])
            arrays.append(tbl.column('hourly_match_score').to_numpy())
    if not arrays:
        return np.array([], dtype=np.float64)
    return np.concatenate(arrays).astype(np.float64)


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
    """For one ISO, scan every endpoint × year and report sparse windows."""
    scores = _load_scores(iso)
    if len(scores) == 0:
        return {'iso': iso, 'pool_size': 0, 'band_counts': {},
                'endpoints': {}, 'sparse_windows': []}

    pool_size = len(scores)
    band_counts = _band_row_counts(iso)
    base_clean_pct = sum(pc.GRID_MIX_SHARES.get(iso, {}).values())

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

            in_window = int(np.sum((scores >= lo) & (scores < hi)))
            is_sparse = in_window < MIN_CANDIDATES

            detail = {
                'year': year,
                'cfe_target': round(tgt, 2),
                'window_lo': round(lo, 2),
                'window_hi': round(hi, 2),
                'candidates_in_window': in_window,
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
                    'candidates': in_window,
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
                min_cand = min(y['candidates_in_window'] for y in sp)
                year_range = f"{sp[0]['year']}-{sp[-1]['year']}"
                print(f"       ep{ep_key}: {len(sp)} sparse years "
                      f"({year_range}), min_candidates={min_cand}", flush=True)

    # Console summary table
    print(f"\n{'='*72}")
    print(f"POOL GAP DIAGNOSTIC SUMMARY")
    print(f"{'='*72}")
    print(f"{'ISO':<8} {'Pool':>10} {'Sparse Windows':>16} "
          f"{'Worst Gap':>12} {'Worst Band':>14}")
    print(f"{'-'*8} {'-'*10} {'-'*16} {'-'*12} {'-'*14}")
    for iso in ISOS:
        r = all_results[iso]
        sw = r['sparse_windows']
        if sw:
            worst = min(sw, key=lambda x: x['candidates'])
            worst_gap = worst['candidates']
            worst_band = f"{worst['window_lo']:.1f}-{worst['window_hi']:.1f}"
        else:
            worst_gap = '-'
            worst_band = '-'
        print(f"{iso:<8} {r['pool_size']:>10,} {len(sw):>16} "
              f"{worst_gap!s:>12} {worst_band:>14}")
    print(f"{'='*72}")
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
                      n_rows: int, rng: np.random.Generator) -> pa.Table:
    """Generate n_rows interpolated EF candidates with hourly_match_score
    uniformly distributed in [lo_score, hi_score).

    Strategy: pick pairs of rows from the pool — one from just below the
    window and one from just above — and linearly interpolate their mix
    columns.  The interpolation weight is chosen so the resulting
    hourly_match_score lands inside the target window.
    """
    scores = pool.column('hourly_match_score').to_numpy().astype(np.float64)

    # Rows in the band just below the window (within 5 pp)
    below_mask = (scores >= lo_score - 5.0) & (scores < lo_score)
    # Rows in the band just above the window (within 5 pp)
    above_mask = (scores >= hi_score) & (scores < hi_score + 5.0)

    below_idx = np.where(below_mask)[0]
    above_idx = np.where(above_mask)[0]

    if len(below_idx) == 0 or len(above_idx) == 0:
        # Fallback: widen the search to ±10 pp
        below_mask = (scores >= lo_score - 10.0) & (scores < lo_score)
        above_mask = (scores >= hi_score) & (scores < hi_score + 10.0)
        below_idx = np.where(below_mask)[0]
        above_idx = np.where(above_mask)[0]

    if len(below_idx) == 0 or len(above_idx) == 0:
        # Still nothing — can't interpolate, skip
        print(f"  [augment] WARNING: no neighbor rows for "
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


def run_augment(diag_results: dict, dry_run: bool = False):
    """Generate interpolated EF rows for each sparse window."""
    rng = np.random.default_rng(42)

    # Deduplicate windows: group by (iso, window_lo, window_hi)
    # since many endpoints share the same window in early years.
    windows_by_iso: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for iso, result in diag_results.items():
        seen = set()
        for sw in result['sparse_windows']:
            key = (sw['window_lo'], sw['window_hi'])
            if key not in seen:
                seen.add(key)
                windows_by_iso[iso].append(key)

    total_written = 0
    for iso, windows in windows_by_iso.items():
        if not windows:
            continue

        print(f"\n[augment] {iso}: {len(windows)} unique sparse windows", flush=True)

        if dry_run:
            for lo, hi in sorted(windows):
                print(f"  [dry-run] would augment [{lo:.1f}, {hi:.1f})", flush=True)
            continue

        # Load full pool once per ISO
        print(f"  loading full pool...", end=' ', flush=True)
        pool = _load_full_pool(iso)
        print(f"{pool.num_rows:,} rows", flush=True)

        for lo, hi in sorted(windows):
            lower_band, upper_band = _find_neighbor_bands(lo, hi)
            if lower_band is None or upper_band is None:
                print(f"  [{lo:.1f}, {hi:.1f}): no bracketing bands, skip", flush=True)
                continue

            interp_tbl = _interpolate_rows(pool, lo, hi, INTERP_ROWS_PER_GAP, rng)
            if interp_tbl is None:
                continue

            # Determine which band this augments (use lower threshold name)
            band_tag = _threshold_tag(lower_band)
            out_name = f'step_2_1_EF_{iso}_{band_tag}_interp_{lo:.0f}_{hi:.0f}.parquet'
            out_path = EF_DIR / out_name

            pq.write_table(interp_tbl, out_path,
                           compression='zstd', compression_level=3)
            total_written += interp_tbl.num_rows
            print(f"  [{lo:.1f}, {hi:.1f}): wrote {interp_tbl.num_rows} rows → {out_name}",
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
