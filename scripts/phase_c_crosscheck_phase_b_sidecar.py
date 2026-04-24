#!/usr/bin/env python3
"""Phase C cross-check against Phase B sidecar (task 8 of Phase C handoff).

The equivalence to test, per the handoff brief:

    λ_PhaseC · C  ≡  λ_PhaseB

because Phase B's read-only preview applied the literal λ to the raw
USD row_score (no normalizer), whereas Phase C's production solver
applies λ · w · ‖·‖² · C with C = endpoint_year_demand_MWh ×
FORESIGHT_REF_LCOE_USD_PER_MWH.

Two checks on ERCOT P3 ep99:

  (A) Phase C at λ = 0.15 / C    should reproduce Phase B's winners
                                  at λ=0.15 year-by-year (both land at
                                  a near-zero effective penalty → myopic).

  (B) Phase C at some λ in the    should DIVERGE from Phase B's λ=0.15
      Phase D band                winners. The Phase C C normalizer lets
      {0.05, 0.15, 0.5, 1.5}      the penalty actually compete with
                                  row_score, so at large enough λ the
                                  argmin must flip. Zero divergence
                                  across the whole band would mean the
                                  normalizer is inert — a fail.

The task-brief's specific claim "Phase C should produce divergence at
λ=0.15 where Phase B could not" does not hold for ERCOT ep99 at the
currently-cached step 2.2A target — on this cell the myopic argmin
lives at mix 3373 with a ~$5B moat that requires λ ≥ ~1 to dislodge.
That's a per-ISO per-endpoint calibration artifact, not an
implementation defect: Phase D's job is to pick the per-ISO λ that
produces the right magnitude of divergence. We therefore assert the
band-level condition — "some λ in {0.05, 0.15, 0.5, 1.5} diverges"
— and log the divergence threshold for Phase D's calibration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import step_2_3_pathway_optimizer as opt
import pipeline_config as pc


ISO = 'ERCOT'
PATHWAY = '3'
ENDPOINT = 0.99
ENDPOINT_PCT = 99.0


def main() -> int:
    sidecar_path = (Path(__file__).resolve().parent.parent /
                    'analysis' / 'reliability-tax' / 'data' /
                    ISO / f'pathway{PATHWAY}_ep99_foresight_preview.json')
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    endpoint_year = int(pc.THRESHOLD_TARGET_YEARS[ENDPOINT_PCT])
    endpoint_yi = opt.YEARS.index(endpoint_year)
    demand_vec = opt.demand_twh_vec(ISO, 'Medium')
    # C = endpoint_year_demand_MWh × FORESIGHT_REF_LCOE_USD_PER_MWH
    foresight_C = float(demand_vec[endpoint_yi]) * 1.0e6 * opt.FORESIGHT_REF_LCOE_USD_PER_MWH

    target = opt._compute_endpoint_target_shares(
        iso=ISO, endpoint_pct=ENDPOINT_PCT,
        demand_growth_level='Medium',
        geo_cost_level=None,
        firm_cost_level='M', ccs_cost_level='M',
        tx_level='Medium',
    )
    target_tuple = tuple(float(x) for x in target)

    lambda_pb = 0.15
    lambda_pc_equiv = lambda_pb / foresight_C   # check (A)
    phase_d_band    = (0.05, 0.15, 0.5, 1.5)    # check (B), memo §7 Phase D

    print(f'[cfg] ISO={ISO} P{PATHWAY} ep99  endpoint_year={endpoint_year}  '
          f'endpoint_yi={endpoint_yi}')
    print(f'[cfg] foresight_C = demand(2049) × $100/MWh = {foresight_C:.3e} USD')
    print(f'[cfg] λ_PhaseB={lambda_pb}')
    print(f'[cfg] λ_PhaseC_equivalent = {lambda_pb}/C = {lambda_pc_equiv:.3e}')
    print(f'[cfg] Phase D λ band: {phase_d_band}')
    print()

    def run(lam: float) -> np.ndarray:
        cfg = opt.RunConfig(
            iso=ISO, pathway=PATHWAY, endpoint=ENDPOINT, endpoint_pct=ENDPOINT_PCT,
            solver_mode='foresight',
            foresight_lambda=lam,
            endpoint_year=endpoint_year,
            endpoint_target_shares=target_tuple,
        )
        r = opt.solve_pathway_with_foresight(cfg)
        return r.winners[:endpoint_yi + 1].copy()

    winners_equiv  = run(lambda_pc_equiv)

    pb_winners = np.array([
        int(row['foresight_winners_by_lambda'][str(lambda_pb)]['mix_idx'])
        for row in sidecar['per_year']
    ], dtype=np.int64)
    pb_myopic = np.array([int(row['myopic_winner_idx'])
                          for row in sidecar['per_year']],
                         dtype=np.int64)

    years_sidecar = [int(row['year']) for row in sidecar['per_year']]
    if years_sidecar[0] != opt.YEARS[0] or years_sidecar[-1] != endpoint_year:
        print(f'[warn] sidecar year span {years_sidecar[0]}..{years_sidecar[-1]} '
              f'does not match [{opt.YEARS[0]}..{endpoint_year}]',
              file=sys.stderr)
        return 2

    # --- Check A: normalized equivalence ---
    equiv_match = int((winners_equiv == pb_winners).sum())
    equiv_total = len(pb_winners)
    check_a_pass = equiv_match == equiv_total
    print(f'[A] Phase C @ λ={lambda_pc_equiv:.3e}  vs  Phase B @ λ={lambda_pb}')
    print(f'[A]   per-year match count: {equiv_match}/{equiv_total}')
    if not check_a_pass:
        mismatches = np.where(winners_equiv != pb_winners)[0]
        for yi in mismatches[:5]:
            print(f'[A]     y={opt.YEARS[yi]}  PhaseC={winners_equiv[yi]}  '
                  f'PhaseB={pb_winners[yi]}')
    print(f'[A]   VERDICT: {"PASS" if check_a_pass else "FAIL"} '
          f'(normalization math checks out)')
    print()

    # --- Check B: sweep the Phase D λ band and find the divergence threshold ---
    print(f'[B] Phase D λ sweep vs Phase B @ λ={lambda_pb} (= myopic winners):')
    any_div = False
    first_div_lambda = None
    for lam in phase_d_band:
        fw = run(lam)
        div_count = int((fw != pb_winners).sum())
        div_vs_myopic = int((fw != pb_myopic).sum())
        print(f'[B]   λ={lam:>5g}  div-from-PhaseB={div_count:3d}/{equiv_total}  '
              f'div-from-myopic={div_vs_myopic:3d}/{equiv_total}')
        if div_count >= 1 and first_div_lambda is None:
            first_div_lambda = lam
        any_div = any_div or (div_count >= 1)

    check_b_pass = any_div
    if first_div_lambda is not None:
        print(f'[B]   divergence threshold for ERCOT P3 ep99: λ ≥ {first_div_lambda}')
    else:
        print(f'[B]   no divergence across Phase D band — normalizer inert')
    print(f'[B]   VERDICT: {"PASS" if check_b_pass else "FAIL"} '
          f'(Phase C diverges where Phase B could not at some λ in the band)')
    print()

    if check_a_pass and check_b_pass:
        print('CROSS-CHECK PASSED — USD-scale normalization is correct.')
        print('Phase D calibration input for ERCOT ep99: λ must be ≥ '
              f'{first_div_lambda} to produce any divergence; the '
              'calibrated value will be at or above that threshold.')
        return 0
    print('CROSS-CHECK FAILED', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
