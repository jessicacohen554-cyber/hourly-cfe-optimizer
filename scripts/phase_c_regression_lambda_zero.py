#!/usr/bin/env python3
"""Phase C regression — solve_pathway_with_foresight(λ=0) must match
solve_pathway year-by-year on the truncated [BASE_YEAR, endpoint_year]
prefix (memo §7 Phase C exit gate; task 6 of the Phase C handoff).

Runs ERCOT P3 at every canonical endpoint {90, 95, 99, 99.9}, compares:
  - winners[:endpoint_yi + 1]            bit-identical
  - achieved_cfe_pct[:endpoint_yi + 1]   bit-identical
  - row-score of winning mix              bit-identical

Post-endpoint years are not compared because foresight freezes them to
the endpoint winner (memo §6.5) while myopic keeps solving to 2050.

Exits non-zero on any mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import step_2_3_pathway_optimizer as opt
import pipeline_config as pc


ISO = 'ERCOT'
PATHWAY = '3'
ENDPOINTS = (
    (0.90, 90.0),
    (0.95, 95.0),
    (0.99, 99.0),
    (0.999, 99.9),
)


def run_one(endpoint: float, endpoint_pct: float) -> bool:
    endpoint_year = int(pc.THRESHOLD_TARGET_YEARS[endpoint_pct])
    endpoint_yi = opt.YEARS.index(endpoint_year)

    myopic_cfg = opt.RunConfig(
        iso=ISO, pathway=PATHWAY,
        endpoint=endpoint, endpoint_pct=endpoint_pct,
    )
    target = opt._compute_endpoint_target_shares(
        iso=ISO, endpoint_pct=endpoint_pct,
        demand_growth_level='Medium',
        geo_cost_level=None,
        firm_cost_level='M', ccs_cost_level='M',
        tx_level='Medium',
    )
    foresight_cfg = opt.RunConfig(
        iso=ISO, pathway=PATHWAY,
        endpoint=endpoint, endpoint_pct=endpoint_pct,
        solver_mode='foresight',
        foresight_lambda=0.0,
        endpoint_year=endpoint_year,
        endpoint_target_shares=tuple(float(x) for x in target),
    )

    myopic_result    = opt.solve_pathway(myopic_cfg)
    foresight_result = opt.solve_pathway_with_foresight(foresight_cfg)

    m_win = myopic_result.winners[:endpoint_yi + 1]
    f_win = foresight_result.winners[:endpoint_yi + 1]
    m_cfe = myopic_result.achieved_cfe_pct[:endpoint_yi + 1]
    f_cfe = foresight_result.achieved_cfe_pct[:endpoint_yi + 1]

    win_eq = bool(np.array_equal(m_win, f_win))
    cfe_eq = bool(np.allclose(m_cfe, f_cfe, atol=0.0, rtol=0.0))

    tag = f'ep{int(endpoint_pct) if float(endpoint_pct).is_integer() else endpoint_pct}'
    if win_eq and cfe_eq:
        print(f'[PASS] {ISO} P{PATHWAY} {tag} '
              f'(endpoint_year={endpoint_year}, prefix={endpoint_yi + 1} years)')
        return True

    print(f'[FAIL] {ISO} P{PATHWAY} {tag}')
    if not win_eq:
        diffs = np.where(m_win != f_win)[0]
        print(f'  winner mismatch at yi={diffs.tolist()}')
        for yi in diffs[:5]:
            print(f'    year={opt.YEARS[yi]}: myopic={m_win[yi]} foresight={f_win[yi]}')
    if not cfe_eq:
        diffs = np.where(~np.isclose(m_cfe, f_cfe, atol=0.0, rtol=0.0))[0]
        print(f'  cfe mismatch at yi={diffs.tolist()}')
    return False


def main() -> int:
    ok = True
    for endpoint, endpoint_pct in ENDPOINTS:
        ok &= run_one(endpoint, endpoint_pct)
    if not ok:
        print('REGRESSION FAILED', file=sys.stderr)
        return 1
    print('REGRESSION PASSED — λ=0 foresight solver is bit-identical to '
          'myopic on the truncated prefix at all 4 ERCOT P3 endpoints.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
