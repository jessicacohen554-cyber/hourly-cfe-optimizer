#!/usr/bin/env python3
"""SPEC §24.5 surrogate A/B metrics — compute per-combo deltas.

Reads analysis/reliability-tax/_tmp_surrogate_ab/ab_summary.json (emitted by
`scripts/_surrogate_ab_driver.py`), loads the paired truth and surrogate
JSONs for each combo, computes the six decision metrics, and writes
analysis/reliability-tax/_tmp_surrogate_ab/ab_metrics.json. Also prints a
readable table to stdout.

Metrics per combo (pathway, endpoint):
  total_cost_delta_pct         = 100 * (surr - tru) / tru on headline.undiscounted_cost_usd
  npv7_delta_pct               = 100 * (surr - tru) / tru on headline.npv_at_7pct
  achieved_cfe_delta           = surr - tru on headline.achieved_cfe_pct
  endpoint_mix_drift           = max over union(keys) of |surr[res] - tru[res]|,
                                 missing-as-zero, in pct-of-demand
  gas_2050_cumulative_mw_delta = |surr - tru| on
                                 tables.annual_buildout[-1].gas_sizing.new_gas_required_cumulative_mw
  speedup                      = wallclock_truth_s / wallclock_surrogate_s

Decision rule (applied by the caller, not here):
  PASS if ALL combos: |total_cost_delta_pct| < 1.0 AND mix_drift < 3.0 AND
                      speedup >= 2.0
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TMP_DIR = _REPO_ROOT / 'analysis' / 'reliability-tax' / '_tmp_surrogate_ab'
_SUMMARY_PATH = _TMP_DIR / 'ab_summary.json'
_METRICS_PATH = _TMP_DIR / 'ab_metrics.json'

# Decision thresholds (SPEC §24.5 / task brief).
_COST_DELTA_PCT_MAX = 1.0
_MIX_DRIFT_MAX = 3.0
_SPEEDUP_MIN = 2.0


def _load(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _pct(num: float, denom: float) -> float:
    return 100.0 * (num - denom) / denom if denom else float('nan')


def _mix_drift(tru_mix: dict, surr_mix: dict) -> float:
    keys = set(tru_mix) | set(surr_mix)
    if not keys:
        return 0.0
    return max(
        abs(float(surr_mix.get(k, 0.0)) - float(tru_mix.get(k, 0.0)))
        for k in keys
    )


def _terminal_gas_mw(payload: dict) -> float:
    buildout = payload['tables']['annual_buildout']
    # Schema: last row is 2050; field name verified against
    # step_2_3_pathway_optimizer.py:2664.
    return float(buildout[-1]['gas_sizing']['new_gas_required_cumulative_mw'])


def _combo_metrics(combo: dict) -> dict:
    tru = _load(combo['truth_path'])
    surr = _load(combo['surrogate_path'])

    tru_cost = float(tru['headline']['undiscounted_cost_usd'])
    surr_cost = float(surr['headline']['undiscounted_cost_usd'])
    tru_npv7 = float(tru['headline']['npv_at_7pct'])
    surr_npv7 = float(surr['headline']['npv_at_7pct'])
    tru_cfe = float(tru['headline']['achieved_cfe_pct'])
    surr_cfe = float(surr['headline']['achieved_cfe_pct'])

    wc_tru = float(combo['wallclock_truth_s'])
    wc_surr = float(combo['wallclock_surrogate_s'])

    return {
        'pathway': combo['pathway'],
        'endpoint': combo['endpoint'],
        'total_cost_delta_pct': _pct(surr_cost, tru_cost),
        'npv7_delta_pct': _pct(surr_npv7, tru_npv7),
        'achieved_cfe_delta': surr_cfe - tru_cfe,
        'endpoint_mix_drift': _mix_drift(
            tru.get('endpoint_mix_pct', {}) or {},
            surr.get('endpoint_mix_pct', {}) or {},
        ),
        'gas_2050_cumulative_mw_delta': abs(
            _terminal_gas_mw(surr) - _terminal_gas_mw(tru)
        ),
        'speedup': wc_tru / wc_surr if wc_surr > 0 else float('inf'),
        'wallclock_truth_s': wc_tru,
        'wallclock_surrogate_s': wc_surr,
    }


def _passes(m: dict) -> bool:
    return (
        abs(m['total_cost_delta_pct']) < _COST_DELTA_PCT_MAX
        and m['endpoint_mix_drift'] < _MIX_DRIFT_MAX
        and m['speedup'] >= _SPEEDUP_MIN
    )


def _print_table(combos: list[dict]) -> None:
    header = (
        f"{'pathway':>8} {'endpoint':>9} "
        f"{'cost Δ%':>10} {'npv7 Δ%':>10} {'cfe Δ':>8} "
        f"{'mix drift':>10} {'gas2050 Δ MW':>14} {'speedup':>9} {'pass':>6}"
    )
    print(header)
    print('-' * len(header))
    for m in combos:
        print(
            f"{m['pathway']:>8} {m['endpoint']:>9.2f} "
            f"{m['total_cost_delta_pct']:>10.4f} "
            f"{m['npv7_delta_pct']:>10.4f} "
            f"{m['achieved_cfe_delta']:>8.4f} "
            f"{m['endpoint_mix_drift']:>10.4f} "
            f"{m['gas_2050_cumulative_mw_delta']:>14.1f} "
            f"{m['speedup']:>9.2f} "
            f"{('Y' if _passes(m) else 'N'):>6}"
        )


def main() -> int:
    summary = _load(_SUMMARY_PATH)
    per_combo = [_combo_metrics(c) for c in summary['combos']]

    summary_block = {
        'max_total_cost_delta_pct': max(abs(m['total_cost_delta_pct']) for m in per_combo),
        'max_mix_drift': max(m['endpoint_mix_drift'] for m in per_combo),
        'min_speedup': min(m['speedup'] for m in per_combo),
        'combos_passing': sum(1 for m in per_combo if _passes(m)),
        'combos_total': len(per_combo),
        'thresholds': {
            'cost_delta_pct_max': _COST_DELTA_PCT_MAX,
            'mix_drift_max': _MIX_DRIFT_MAX,
            'speedup_min': _SPEEDUP_MIN,
        },
    }
    summary_block['overall_pass'] = (
        summary_block['combos_passing'] == summary_block['combos_total']
    )

    out = {
        'iso': summary.get('iso', 'ERCOT'),
        'combos': per_combo,
        'summary': summary_block,
    }
    _METRICS_PATH.write_text(json.dumps(out, indent=2))

    _print_table(per_combo)
    print()
    print(
        f"summary: max|cost Δ%|={summary_block['max_total_cost_delta_pct']:.4f}  "
        f"max mix drift={summary_block['max_mix_drift']:.4f}  "
        f"min speedup={summary_block['min_speedup']:.2f}×  "
        f"{summary_block['combos_passing']}/{summary_block['combos_total']} "
        f"combos pass  → overall {'PASS' if summary_block['overall_pass'] else 'FAIL'}"
    )
    print(f"wrote {_METRICS_PATH}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
