# Session D — Full Sweep Progress Log

Branch: `claude/pathway-optimizer-full-sweep-ZdtSf`
Prereq: commit `0eed484` (Prompt 3b cross-ISO sanity) — confirmed ancestor.

## Phase 1 — Pre-run cleanup (complete)

1. Fixture copy: **SKIPPED**. `analysis/reliability-tax/data/ERCOT/pathway1_ep90.json`
   is already `schema_version==2` (Prompt 3/3b overwrote the v1 payload). The
   historical-receipt fixture at `reliability_tax/fixtures/v1_sample_ERCOT_p1_ep90.json`
   would have been a v2 copy pretending to be v1, so the copy was skipped per
   the brief's fallback clause. The `reliability_tax/fixtures/` directory has
   been created empty.
2. Deleted 0 non-v2 JSONs. All 58 existing `pathway*_ep*.json` files under
   `analysis/reliability-tax/data/{ERCOT,CAISO}/` carry `schema_version==2`
   and were preserved.
3. `analysis/reliability-tax/data/MANIFEST.json` was not present — no-op.

GATE 1 result: only `schema_version==2` payloads remain on disk; the fixture
skip is logged here.

## Phase 2 — ERCOT-only smoke sweep (complete)

Ran `_run_iso_all('ERCOT')` in a single Python process (one import → caches
loaded once → 50 combos solved sequentially). The `--iso-only` flag in the
brief does not exist in the CLI, so the single-process route was used
instead of a shell loop — it avoids paying the dispatch-cache load cost per
subprocess and preserves the warm Stage-1 sidecar.

Wall time: **221 s** (~3.7 min). All 10 ERCOT peakclean sidecars were already
warm on disk; no Stage-1 rebuild was triggered. Warm cold-cache ratio:
effectively 1:1 because no cold runs were needed.

**ERCOT count note.** The brief said 40 combos, but the optimizer CLI's
endpoint grid has 10 entries (`0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
0.975, 0.99, 0.999`), yielding 5 pathways × 10 endpoints = **50 ERCOT
JSONs**. The full 7-ISO sweep therefore totals 7 × 50 = **350 payloads**,
not 1,120 as the brief states. Proceeding with the actual grid; flagging
the discrepancy here and in the Phase 3 report.

### GATE 2 spot-check: `ERCOT/pathway1_ep90.json`

| check | expected | actual | pass |
|---|---|---|---|
| top-level keys | 13 | 13 | ✅ |
| schema_version | 2 | 2 | ✅ |
| new_gas_fleet len | 1 | 1 | ✅ |
| headline.achieved_cfe_pct | ∈ [85, 92] | 90.07 | ✅ |
| reliability_tax.usd_per_mwh | ∈ [0, 200] | 17.06 | ✅ |
| stranding_metadata.fleet_size_mw | ∈ [100, 150] GW | 139.98 GW | ✅ |
| v1-reference ratio (111 GW → 140 GW) | ∈ [0.7, 1.5] × | 1.26× | ✅ |

### Known anomaly — logged, not blocking

All 5 pathways at ep90 (and at every other endpoint) produce **identical**
fleet size, achieved_cfe_pct, reliability_tax, endpoint_mix, and
stranded_capex values. This matches what was seen in the 12-combo cross-ISO
sanity (commit `0eed484`), so it's a known v2 behavior — pathway
differentiation is apparently not expressed in any of the headline metrics
the MANIFEST will aggregate. Proceeding under the brief's scope (run the
sweep, flag out-of-band runs); leaving the pathway-differentiation
investigation to a separate task.

## Phase 3a — CAISO + PJM sweeps (Session 3a)

Branch: `claude/caiso-pjm-full-sweep-n98d1`
Starting HEAD: `e47753e`

### PJM size-check decision

Ran parquet row-count probe across all PJM threshold files:

| threshold | rows |
|---|---|
| 60 | 158,714 |
| 70 | 275,270 |
| 75 | 318,211 |
| 80 | 218,039 |
| 85 | 40,136 |
| 87.5 | 65,422 |
| 90 | 4,814 |
| 92.5 | 54,436 |
| 95 | 249,818 |
| 97.5 | 296,464 |
| 99 | 257,282 |
| 99.5 | 400,177 |
| 99.9 | 628,172 |

**Decision: PJM is FAT.** PJM_99.9 has 628k rows, exceeding the 500k
threshold. Running PJM via per-combo subprocess (same loop shape as CAISO)
so memory fully resets between combos.
