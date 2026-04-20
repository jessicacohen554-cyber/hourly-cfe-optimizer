# AUDIT — §24.5 surrogate residual A/B (ERCOT)

**Date**: 2026-04-20 · **Master SHA at session start**: `c1699a2` · **Branch**: `claude/surrogate-ab-finish-IBTwN`

**One-sentence summary**: Attempted the ERCOT four-combo A/B validation for `USE_SURROGATE_RESIDUAL_IN_ARGMIN` (truth 99.97-percentile dispatch vs. HistGBM surrogate) — the driver was killed by the sandbox mid-run on the first TRU combo, so **no decision was reached**. Flag remains `False`.

## Methodology (as designed)

Driver: `scripts/_surrogate_ab_driver.py` (committed at `36304ab`, wip pre-session). Runs four combos — (pathway, endpoint) ∈ {(1, 0.80), (1, 0.95), (3, 0.80), (3, 0.95)} — twice each under a single hot Python process, prewarming dispatch + EF caches once at the top:

| Run | Flag | Output path |
|---|---|---|
| truth | `USE_SURROGATE_RESIDUAL_IN_ARGMIN=False` | `analysis/reliability-tax/data/ERCOT/pathway{P}_ep{EE}.json` |
| surrogate | `USE_SURROGATE_RESIDUAL_IN_ARGMIN=True`  | `analysis/reliability-tax/data/ERCOT/pathway{P}_ep{EE}.surrogate.json` |

Per-run wallclock is measured via `time.perf_counter()` around `run_pathway`. Summary lands at `analysis/reliability-tax/_tmp_surrogate_ab/ab_summary.json` with one row per combo: `{pathway, endpoint, wallclock_truth_s, wallclock_surrogate_s, truth_path, surrogate_path}`.

### Metric definitions (`scripts/_surrogate_ab_metrics.py`)

Source fields are named verbatim from `scripts/step_2_3_pathway_optimizer.py` (`_serialize_run_result`, line 2978+):

| Metric | Formula | Source field |
|---|---|---|
| `total_cost_delta_pct` | 100·(surr − tru)/tru | `headline.undiscounted_cost_usd` |
| `npv7_delta_pct` | 100·(surr − tru)/tru | `headline.npv_at_7pct` |
| `achieved_cfe_delta` | surr − tru (expected ≈ 0 — endpoint gates both runs) | `headline.achieved_cfe_pct` |
| `endpoint_mix_drift` | max over res ∈ union(keys) of \|surr[res] − tru[res]\|, missing-as-zero, in pct-of-demand | `endpoint_mix_pct` (top level) |
| `gas_2050_cumulative_mw_delta` | \|surr − tru\| | `tables.annual_buildout[-1].gas_sizing.new_gas_required_cumulative_mw` (line 2664) |
| `speedup` | `wallclock_truth_s / wallclock_surrogate_s` | driver summary |

### Decision rule

PASS if **all four** combos satisfy: `|total_cost_delta_pct| < 1.0` AND `endpoint_mix_drift < 3.0` AND `speedup ≥ 2.0`.

## Per-combo results

| Pathway | Endpoint | total_cost_delta_pct | npv7_delta_pct | mix_drift | gas_2050_delta_mw | speedup |
|---|---|---|---|---|---|---|
| 3 | 0.80 | N/A — SIGKILL mid-run | N/A | N/A | N/A | N/A |
| 1 | 0.80 | not attempted          | N/A | N/A | N/A | N/A |
| 3 | 0.95 | not attempted          | N/A | N/A | N/A | N/A |
| 1 | 0.95 | not attempted          | N/A | N/A | N/A | N/A |

**Summary row**: not computable — zero combos produced paired output.

## Execution record

1. **Preflight** — `scripts/_surrogate_ab_driver.py` and its imports failed at module load on a clean environment: `ModuleNotFoundError: No module named 'sklearn'` (via `step2_2a_cost_optimization.py:107`, the HistGBM surrogate host) and then `ModuleNotFoundError: No module named 'pandas'`. Resolved by `pip install scikit-learn pandas` and pinning `scikit-learn>=1.3.0` in `requirements.txt` — `pandas>=2.0.0` was already declared but uninstalled. Commit: `6d20386`.

2. **First successful launch** — driver reached `[ab] prewarming caches for ERCOT (4 combos)` and then `[ab] TRU  pathway=3 ep=0.8 ...`. It then entered archetype-expansion inside that TRU run (up to 2,861 worst-hour archetypes reconstructed into the dispatch cache).

3. **Termination** — the process was SIGKILL'd silently. Signature: the captured stdout ends mid-print at `[expand] ERCOT: archetype 2761/2861 key=f84dc5da56` — the trailing `reconstructed in 0.00s` line for archetype 2761 was never written. No Python traceback, no `Error`, no `Killed` text, no dmesg OOM entry (gVisor sandbox OOMs are not kernel-logged). Memory sampled post-kill showed 20 GiB free, consistent with the process having been reaped and its RSS returned. The `[ab] TRU pathway=3 ep=0.8` completion line never printed, and the driver's `finally:` block never ran — no `ab_summary.json` was emitted.

## Conclusion — **INCOMPLETE**, flag unchanged

Zero of four combos completed a paired truth+surrogate run. The decision rule cannot be evaluated; `scripts/pipeline_config.py:1454 USE_SURROGATE_RESIDUAL_IN_ARGMIN` remains at its committed default `False`.

**Failing combos** (against their thresholds): not applicable — the run never produced measurements. The failure is environmental (silent SIGKILL inside archetype expansion), not a methodology failure of the surrogate or the driver.

### Candidate unblockers (not applied here — out of scope for this session)

1. Hoist the archetype-expansion `persist=True` write out of the year loop (already flagged in `LESSONS.md` item 11; the deferred-write path in the driver at `iso_state['deferred'] = True` is in that direction but appears insufficient on this sandbox).
2. Run one combo per Python invocation instead of a single hot process holding all four combos' state — trades cold-start overhead for lower peak RSS.
3. Re-attempt on a host with more container memory; the computed speedup would also be more trustworthy from a less-constrained environment.
