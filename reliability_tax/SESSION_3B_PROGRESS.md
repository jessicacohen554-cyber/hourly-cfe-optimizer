# Session 3B — Stage-1 peak-hour fix — resume prompt

Branch: `claude/fix-stage1-peak-hour-cav8i`
Last commit: `3d43a1c` (Phase 1 partial — dispatch-cache lookup wired, MW
scale correct, 100% archetype match on ERCOT ep90, but fleet_size_mw
landed at 196 GW instead of the ~111 GW v1 target).

## Decision locked this session

- Methodology = **99.97th-percentile residual** with RA-margin baked into
  demand (v1 method — `WORST_HOUR_PERCENTILE=99.97` in
  `scripts/step2_2a_cost_optimization.py:173`). Not argmax(demand−clean) as
  the brief originally said.
- Approach = **Option B**: refactor Stage-2 so Stage-1 returns
  `resid_norm_p9997` (fraction of annual demand) and `gas_sizing_matrix`
  uses v1's residual-based formula. Option A (keep Stage-2, derive a
  compatible `clean_arr`) was ruled out — the two formulas scale
  differently across 26 years of demand growth, so no single per-mix
  scalar satisfies both.

## What's already done (committed)

- `scripts/step_2_3_pathway_optimizer.py`:
  - Module-scope `_DEMAND_PROFILE_BY_ISO` (reads
    `data/eia-930/eia_demand_profiles.parquet` directly via pyarrow;
    pandas is NOT installed in this env — don't import pandas).
  - Module-scope `_STAGE1_MISSES` tracker.
  - `precompute_clean_peak_hour_mw` rewritten to do vectorized archetype
    NN lookup against `_DISPATCH_CACHE_BY_ISO`, scale `total_clean` by
    `REGIONAL_DEMAND_TWH * 1e6`, then pick argmax hour. **This is the
    step to replace** with 99.97th percentile.
  - `_synthetic_clean_peak_fallback` helper for archetype-miss rows.
  - Sidecar `stage1_version='2'` metadata + staleness check in
    `_load_or_build_peakclean`.

## What's left for Phase 1

1. **Replace argmax with 99.97th percentile in Stage-1**.
   - Keep `TC` as fractions (remove the `* annual_mwh` scaling at
     `scripts/step_2_3_pathway_optimizer.py:404`).
   - Compute per-archetype residual percentile ONCE (outside the batch
     loop, since it's archetype-level not EF-row-level):
     ```python
     demand_frac = demand_hr / annual_mwh  # sums to 1.0
     demand_margin_frac = demand_frac * (1.0 + pc.RESOURCE_ADEQUACY_MARGIN)
     margin_resid = np.maximum(0.0, demand_margin_frac[None, :] - TC)  # (A, 8760)
     resid_p9997_per_arch = np.percentile(margin_resid, 99.97, axis=1)  # (A,)
     ```
   - Per EF row: `resid_p9997[row] = resid_p9997_per_arch[cache_idx]`.
   - Sidecar schema change: **new column** `resid_norm_p9997` (fraction,
     float32). Keep `clean_peak_hour_mw` as a diagnostic (derived from
     residual — see below) OR replace downstream consumption.

2. **Refactor `gas_sizing_matrix`** (line 533) to consume
   `resid_norm_p9997` instead of `clean_peak_hour_mw`:
   ```python
   def gas_sizing_matrix(annual_mwh_vec, existing_gas_vec, resid_arr,
                         gas_raw_2025, existing_base, gaf):
       # annual_mwh_vec shape (N_YEARS,) = REGIONAL_DEMAND_TWH * 1e6 * (1+g)^y
       # resid_arr shape (n_mixes,) — fraction of annual demand
       gap_mwh = resid_arr[None, :] * annual_mwh_vec[:, None]  # (N_YEARS, n_mixes)
       gas_raw = gap_mwh / gaf
       gas_needed = np.maximum(0.0, existing_base + gas_raw - gas_raw_2025)
       new_gas = np.maximum(0.0, gas_needed - existing_gas_vec[:, None])
       return new_gas
   ```
   Fully vectorized — preserves the N_YEARS=26 invariant.

3. **Precompute `_GAS_RAW_2025_BY_ISO`** at module scope:
   - For each ISO, build `mix_pct = GRID_MIX_SHARES[iso]` (resources,
     no storage, no hybrids).
   - Find closest archetype via `_archetype_for_endpoint_mix` against
     the manifest (or directly against the cache key built from
     `dispatch_utils._archetype_key` — the manifest NN is simpler).
   - Get its `total_clean` 8760 fraction profile.
   - `resid_2025_norm = np.percentile(max(0, demand_margin_frac - tc_frac), 99.97)`
   - `gas_raw_2025 = resid_2025_norm * annual_mwh_base / GAS_AVAILABILITY_FACTOR[iso]`

4. **Update call sites** at lines 713, 742:
   ```python
   annual_mwh_vec = demand_vec * 1.0e6  # TWh → MWh; demand_vec already grown
   gas_required = gas_sizing_matrix(
       annual_mwh_vec, existing_gas_vec, resid_arr,
       _GAS_RAW_2025_BY_ISO[cfg.iso],
       float(pc.EXISTING_GAS_CAPACITY_MW[cfg.iso]),
       pc.GAS_AVAILABILITY_FACTOR[cfg.iso])
   ```

5. **Keep `clean_peak_hour_mw` as a display-only diagnostic** at line 1029.
   Simplest: `clean_peak_hour_mw_diag = (1 - resid_norm_p9997) * PEAK_DEMAND_MW[iso]`
   — rough but readable. Or drop the field from the display dict.

6. **Gate 1 re-run**: `python scripts/step_2_3_pathway_optimizer.py
   --iso ERCOT --pathway 1 --endpoint 0.90`. Expect fleet ≈ 111 GW.

7. **Phase 2 sweep**: 12 combos `{ERCOT, CAISO} × {P1, P3} × {ep90, ep95,
   ep99}`. See plan file `/root/.claude/plans/fix-the-stage-1-precompute-clean-peak-ho-magical-muffin.md`.

## Hard constraints (don't violate)

- Vectorized numpy only. No numba, threads, locks, timeouts.
- No import from `scripts/step2_2a_cost_optimization.py` or the OLD
  `scripts/step_2_3_pathway_optimizer.py`.
- N_YEARS=26 per-year loop invariant preserved (Stage-2 refactor must
  stay vectorized across years × mixes).
- 13-key JSON schema preserved (no new top-level keys, no removed keys).
- pandas is NOT installed — use pyarrow (`pq.read_table`, `.to_numpy()`,
  `.as_py()`).

## Key file refs

- Stage-1 function: `scripts/step_2_3_pathway_optimizer.py:367`
- Stage-2 gas_sizing_matrix: `scripts/step_2_3_pathway_optimizer.py:533`
- Call sites to update: lines 713, 742
- Display use of clean_peak_hour_mw: line 1029
- v1 methodology reference: `scripts/step2_2a_cost_optimization.py:435`
  (`_gas_raw_2025_worst_hour`) and `:485` (`_worst_hour_residual_norm`)
- v1 result (111 GW ERCOT P1 ep90): `SPEC_LOG.md:5254`
- Approved plan: `/root/.claude/plans/fix-the-stage-1-precompute-clean-peak-ho-magical-muffin.md`

## Resume command

```
git checkout claude/fix-stage1-peak-hour-cav8i
# Read SESSION_3B_PROGRESS.md (this file) + the plan file
# Start at Phase 1 step "Replace argmax with 99.97th percentile"
```
