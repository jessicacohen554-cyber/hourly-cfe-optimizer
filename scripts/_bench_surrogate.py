#!/usr/bin/env python3
"""Benchmark harness for the surrogate model in step2_2a_cost_optimization.py.

Usage:
    python scripts/_bench_surrogate.py --iso ERCOT --threshold 80
    python scripts/_bench_surrogate.py --iso CAISO --threshold 90

Acceptance gate: MAE < 0.5 AND top-100 overlap > 80.
"""
import argparse
import os
import sys
import time

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iso", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    args = parser.parse_args()

    iso = args.iso
    threshold = args.threshold

    print(f"\n=== _bench_surrogate  iso={iso}  threshold={threshold} ===")

    # ------------------------------------------------------------------
    # 1. Load EF mixes
    # ------------------------------------------------------------------
    print("Loading EF mixes ...", flush=True)
    from step_2_3_pathway_optimizer import load_ef_mixes

    t0 = time.perf_counter()
    arrays = load_ef_mixes(iso, threshold)
    print(f"  loaded N={len(arrays['clean_firm'])} rows in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. Fit surrogate (populates _SURROGATE_CACHE)
    # ------------------------------------------------------------------
    print("Fitting surrogate ...", flush=True)
    from step2_2a_cost_optimization import _SURROGATE_CACHE, _fit_surrogate

    t0 = time.perf_counter()
    _fit_surrogate(iso, threshold, arrays)
    t_fit = time.perf_counter() - t0

    cache = _SURROGATE_CACHE[(iso, threshold)]
    K = cache["k_trained"]
    print(
        f"  fit complete: K={K} training rows, "
        f"n_unique={cache['n_unique_at_fit']}  ({t_fit:.1f}s)"
    )

    # ------------------------------------------------------------------
    # 3. Hold-out split — last 20% of training rows (no shuffle)
    # ------------------------------------------------------------------
    split = int(0.8 * K)
    X_all = cache["features_raw"]   # (K, 13) float64
    y_all = cache["targets"]        # (K,)    float64
    scaler = cache["scaler"]
    model = cache["model"]

    X_test = X_all[split:]          # (K_test, 13)
    y_exact = y_all[split:]         # (K_test,)
    K_test = X_test.shape[0]
    print(f"  test split: {K_test} rows (indices {split}:{K})")

    # ------------------------------------------------------------------
    # 4. Predict on held-out rows
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    y_pred = model.predict(scaler.transform(X_test)).astype(np.float64)
    t_pred = time.perf_counter() - t0
    print(f"  predict: {t_pred * 1000:.1f} ms")

    # ------------------------------------------------------------------
    # 5. MAE
    # ------------------------------------------------------------------
    mae = float(np.mean(np.abs(y_pred - y_exact)))

    # ------------------------------------------------------------------
    # 6. Top-100 overlap
    # ------------------------------------------------------------------
    TOP_N = 100
    if K_test < TOP_N:
        overlap: int | None = None
        overlap_str = f"N/A (K_test={K_test} < {TOP_N})"
        overlap_ok = False
    else:
        top_exact = set(np.argpartition(y_exact, -TOP_N)[-TOP_N:].tolist())
        top_pred = set(np.argpartition(y_pred, -TOP_N)[-TOP_N:].tolist())
        overlap = len(top_exact & top_pred)
        overlap_str = str(overlap)
        overlap_ok = overlap > 80

    # ------------------------------------------------------------------
    # 7. Gate check and report
    # ------------------------------------------------------------------
    mae_ok = mae < 0.5
    passed = mae_ok and overlap_ok

    print()
    print("-" * 52)
    print(f"  iso={iso}  threshold={threshold}")
    print(f"  MAE        = {mae:.4f}   {'PASS' if mae_ok else 'FAIL'}  (gate < 0.5)")
    print(f"  top-100 ∩  = {overlap_str:>6}   {'PASS' if overlap_ok else 'FAIL'}  (gate > 80)")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    print("-" * 52)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
