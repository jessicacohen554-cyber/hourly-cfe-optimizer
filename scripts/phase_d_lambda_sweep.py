#!/usr/bin/env python3
"""Phase D λ-calibration sweep — memo §7 Phase D.

Run matrix per ISO:
  1 endpoint (2049, ep99) × 5 λ values {0.0, 0.05, 0.15, 0.5, 1.5}
  = 5 foresight runs, plus 1 paired myopic run.
Total: 7 ISOs × 6 in-process runs = 42 solver invocations producing
35 foresight rows + 7 myopic baselines.

Per-λ row emitted to data/phase_d/sweep_{ISO}.json:
  chosen_lambda, foresight_penalty_vs_myopic, cascade_activations,
  winners_diff_from_myopic_count, time_to_90pct, cum_undisc_cost_usd.

Per-ISO selection rule (memo §7 Phase D):
  • pick λ that maximizes foresight_penalty_vs_myopic
  • subject to: cascade_activations ≤ myopic_cascade + 1
  • flag band_edge when chosen λ ∈ {0.0, 1.5}
  • flag all_negative when every positive-λ penalty < 0 → Phase D Task 4
    replacement-cost-argmin fallback is the remediation (not implemented
    here; this script surfaces the trigger)

In-process design (per CLAUDE.md vectorization + Phase C amortization
notes): one Python process owns one ISO. Module-scope loads (dispatch
cache, fossil mix, Wright's curves, EGRID) happen once at import; the
step 2.2A target-shares lookup is cached internally by
opt._compute_endpoint_target_shares so 5 foresight runs share one
parquet read. The only per-run cost is load_ef_mixes + the vectorized
year loop.

Usage:
  # Hot-process form (recommended — one ISO per invocation):
  ISO_FILTER=ERCOT python3 scripts/phase_d_lambda_sweep.py --iso ERCOT
  # Single-process-all form (slower; module-scope loads carry all 7):
  python3 scripts/phase_d_lambda_sweep.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import step_2_3_pathway_optimizer as opt
import pipeline_config as pc


ALL_ISOS = ('CAISO', 'ERCOT', 'MISO', 'NEISO', 'NYISO', 'PJM', 'SPP')
PATHWAY = '3'
ENDPOINT = 0.99       # RunConfig.endpoint (fraction)
ENDPOINT_PCT = 99.0   # RunConfig.endpoint_pct (percent)
LAMBDAS = (0.0, 0.05, 0.15, 0.5, 1.5)

OUT_DIR = _HERE.parent / 'data' / 'phase_d'


def _cum_undisc_cost(result: opt.PathwayRunResult, endpoint_yi: int) -> float:
    """Cumulative undiscounted cost over [BASE_YEAR, endpoint_year] inclusive.

    Both solvers freeze-forward past endpoint_yi so annual_cost_rows length
    is always N_YEARS; we truncate to the endpoint prefix. Matches memo §6.5
    ("solver simulates [start_year, endpoint_year] inclusive and stops").
    """
    rows = result.annual_cost_rows[: endpoint_yi + 1]
    return float(sum(r['net_annual_cost_usd'] for r in rows))


def _cascade_count(result: opt.PathwayRunResult, endpoint_yi: int) -> int:
    """Cascade activations within prefix (tier > 0). Both solvers share the
    same tier encoding (0=Tier1, 1=Tier2, 2=Tier3, 3=Tier4, -1=post-endpoint).
    """
    tier_y = result.cascade_tier_y
    if tier_y is None:
        return 0
    return int((tier_y[: endpoint_yi + 1] > 0).sum())


def _time_to_90(result: opt.PathwayRunResult, endpoint_yi: int) -> Optional[int]:
    pfx = result.achieved_cfe_pct[: endpoint_yi + 1]
    hits = np.where(pfx >= 90.0 - 1e-9)[0]
    return int(opt.YEARS[int(hits[0])]) if hits.size else None


def _winners_diff(myopic_r: opt.PathwayRunResult,
                  foresight_r: opt.PathwayRunResult,
                  endpoint_yi: int) -> int:
    m = myopic_r.winners[: endpoint_yi + 1]
    f = foresight_r.winners[: endpoint_yi + 1]
    return int((m != f).sum())


def _run_myopic(iso: str) -> opt.PathwayRunResult:
    cfg = opt.RunConfig(
        iso=iso, pathway=PATHWAY,
        endpoint=ENDPOINT, endpoint_pct=ENDPOINT_PCT,
    )
    return opt.solve_pathway(cfg)


def _run_foresight(iso: str, lam: float,
                   target_shares: tuple) -> opt.PathwayRunResult:
    cfg = opt.RunConfig(
        iso=iso, pathway=PATHWAY,
        endpoint=ENDPOINT, endpoint_pct=ENDPOINT_PCT,
        solver_mode='foresight',
        foresight_lambda=lam,
        endpoint_year=opt._foresight_endpoint_year(ENDPOINT_PCT),
        endpoint_target_shares=target_shares,
    )
    return opt.solve_pathway_with_foresight(cfg)


def _select_lambda(per_lambda_rows: list[dict],
                   myopic_cascade: int) -> tuple[float, list[str]]:
    """Apply memo §7 Phase D selection rule. Returns (chosen_λ, flags).

    Gate (memo §7 Phase D exit gate):
      chosen λ must satisfy cascade_activations ≤ myopic_cascade + 1 AND
      foresight_penalty_vs_myopic > 0. If no candidate clears both, the
      'needs_replacement_cost_fallback' flag fires and Phase D Task 4's
      replacement-cost-argmin target takes over (re-run that ISO).
    """
    flags: list[str] = []
    cascade_limit = myopic_cascade + 1
    feasible = [r for r in per_lambda_rows
                if r['cascade_activations'] <= cascade_limit]
    if not feasible:
        # Every λ violated the cascade gate — no valid choice in the band.
        flags.append('all_cascade_violations')
        flags.append('needs_replacement_cost_fallback')
        return 0.0, flags

    # Pick max foresight_penalty_vs_myopic among the cascade-feasible set.
    feasible.sort(key=lambda r: r['foresight_penalty_vs_myopic'], reverse=True)
    chosen = feasible[0]
    chosen_lam = float(chosen['chosen_lambda'])
    chosen_penalty = float(chosen['foresight_penalty_vs_myopic'])

    # Fallback trigger: no positive-λ candidate strictly beat myopic. λ=0
    # is the myopic-equivalent baseline (penalty ≈ 0 — Phase C task 6)
    # and doesn't count as "foresight winning." Memo §7 Phase D exit gate
    # requires foresight_penalty_vs_myopic > 0 at the chosen λ.
    positive_rows = [r for r in per_lambda_rows if r['chosen_lambda'] > 0]
    if not any(r['foresight_penalty_vs_myopic'] > 0 for r in positive_rows):
        flags.append('needs_replacement_cost_fallback')
        if positive_rows and all(r['foresight_penalty_vs_myopic'] < 0
                                 for r in positive_rows):
            flags.append('all_negative_penalty')
        else:
            flags.append('no_positive_penalty')

    # Band-edge flags: chosen λ at top or bottom of the memo §7 band.
    # band_edge_low also fires when foresight degenerated to λ=0 because no
    # higher λ produced a positive penalty.
    if chosen_lam == 0.0:
        flags.append('band_edge_low')
    elif chosen_lam == max(LAMBDAS):
        flags.append('band_edge_high')

    # Diagnostic: how much margin does the chosen λ have over myopic?
    # (Already captured as chosen_penalty — included in flags only when
    #  the gate is tight enough to matter for Phase E.)
    if 0.0 < chosen_penalty < 0.001:
        flags.append('marginal_positive_penalty')

    return chosen_lam, flags


def run_iso_sweep(iso: str) -> dict:
    """Run the 1 myopic + 5 foresight matrix for one ISO. Returns the
    sweep_<ISO>.json payload."""
    t0 = time.perf_counter()
    print(f'[{iso}] myopic baseline ...', flush=True)
    myopic_result = _run_myopic(iso)
    endpoint_year = opt._foresight_endpoint_year(ENDPOINT_PCT)
    endpoint_yi = opt.YEARS.index(endpoint_year)
    myopic_cost = _cum_undisc_cost(myopic_result, endpoint_yi)
    myopic_cascade = _cascade_count(myopic_result, endpoint_yi)
    myopic_t90 = _time_to_90(myopic_result, endpoint_yi)
    t_myopic = time.perf_counter() - t0
    print(f'[{iso}] myopic: cum_undisc=${myopic_cost/1e9:.2f}B  '
          f'cascade={myopic_cascade}  t90={myopic_t90}  '
          f'({t_myopic:.1f}s)', flush=True)

    # One step 2.2A target-shares lookup shared across all 5 λ values
    # (opt._compute_endpoint_target_shares caches internally by scenario key).
    target_arr = opt._compute_endpoint_target_shares(
        iso=iso, endpoint_pct=ENDPOINT_PCT,
        demand_growth_level='Medium',
        geo_cost_level=None,
        firm_cost_level='M', ccs_cost_level='M',
        tx_level='Medium',
    )
    target_shares = tuple(float(x) for x in target_arr)

    per_lambda: list[dict] = []
    for lam in LAMBDAS:
        t1 = time.perf_counter()
        foresight_result = _run_foresight(iso, lam, target_shares)
        f_cost = _cum_undisc_cost(foresight_result, endpoint_yi)
        # memo §6.3: positive = myopic more expensive than foresight.
        penalty = ((myopic_cost - f_cost) / f_cost) if f_cost > 0 else 0.0
        row = {
            'chosen_lambda': float(lam),
            'foresight_penalty_vs_myopic': float(penalty),
            'cum_undisc_cost_usd': round(f_cost, 2),
            'cascade_activations': _cascade_count(foresight_result, endpoint_yi),
            'winners_diff_from_myopic_count': _winners_diff(
                myopic_result, foresight_result, endpoint_yi),
            'time_to_90pct': _time_to_90(foresight_result, endpoint_yi),
            'wall_seconds': round(time.perf_counter() - t1, 2),
        }
        per_lambda.append(row)
        print(f'[{iso}] λ={lam:<5g}  penalty={penalty*100:+6.3f}%  '
              f'cascade={row["cascade_activations"]}  '
              f'Δwinners={row["winners_diff_from_myopic_count"]:3d}  '
              f't90={row["time_to_90pct"]}  '
              f'({row["wall_seconds"]}s)', flush=True)

    chosen_lambda, flags = _select_lambda(per_lambda, myopic_cascade)

    payload = {
        'iso': iso,
        'pathway': PATHWAY,
        'endpoint_pct': ENDPOINT_PCT,
        'endpoint_year': int(endpoint_year),
        'endpoint_yi': int(endpoint_yi),
        'lambdas_swept': list(LAMBDAS),
        'myopic_baseline': {
            'cum_undisc_cost_usd': round(myopic_cost, 2),
            'cascade_activations': int(myopic_cascade),
            'time_to_90pct': myopic_t90,
            'wall_seconds': round(t_myopic, 2),
        },
        'target_shares': {
            k: float(v) for k, v in zip(opt._FORESIGHT_SHARE_KEYS, target_arr)
        },
        'target_shares_method': 'cheapest_cumulative_path_step_2_2a',
        'per_lambda': per_lambda,
        'chosen_lambda': float(chosen_lambda),
        'chosen_flags': flags,
        'cascade_gate_limit': int(myopic_cascade + 1),
        'total_wall_seconds': round(time.perf_counter() - t0, 2),
    }
    return payload


def write_sweep(payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'sweep_{payload["iso"]}.json'
    tmp = out.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    tmp.replace(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Phase D λ-calibration sweep')
    ap.add_argument('--iso', choices=ALL_ISOS,
                    help='Run one ISO (recommended: one Python process '
                         'per ISO with ISO_FILTER matching).')
    ap.add_argument('--all', action='store_true',
                    help='Run all 7 ISOs in a single process (slower; '
                         'module-scope loads carry all 7 dispatch caches).')
    args = ap.parse_args(argv)

    if not args.iso and not args.all:
        print('Provide --iso <ISO> OR --all', file=sys.stderr)
        return 2

    isos = [args.iso] if args.iso else list(ALL_ISOS)
    for iso in isos:
        payload = run_iso_sweep(iso)
        out = write_sweep(payload)
        print(f'[{iso}] wrote {out}  chosen_λ={payload["chosen_lambda"]}  '
              f'flags={payload["chosen_flags"]}  '
              f'({payload["total_wall_seconds"]}s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
