##SONNET 4.6

Critical (OOM — must fix)

1. solve_pathway_with_foresight: remove all dense (N_YEARS, n_mixes) pre-allocations.
   Lines 1980–2031 allocate ~10–15 dense (26 × N) float64/bool arrays before the year
   loop: pmask, cfe_mask, gas_required, ratchet_fleet_y, new_gas_capex_y, new_gas_fom_y,
   gas_fuel, gas_fixed_y, shares_nr, the five tg tranche arrays, and storage_cost_yn.
   For CAISO (21M mixes) that's ~50 GB. The myopic solver eliminated these in
   "Phase 1a Commit B" — the foresight solver never got the same refactor, even though
   line 1963 says it mirrors that rewrite.

2. solve_pathway_with_foresight: move to chunked streaming year loop.
   The year loop (lines 2080–2173) currently passes the full N-mix arrays into
   _score_year_sunk_cost and _project_shares_to_endpoint each year. Apply the same
   _POOL_CHUNK_SIZE = 500_000 chunking as the myopic solver: maintain running_max_gas_n
   (N,) for gas ratchet tracking, inline tranche decomposition per-chunk, and call the
   already-available _project_shares_to_endpoint_chunk (line 1337) instead of the
   full-pool version.

3. Post-loop gas_sizing_matrix call in foresight (lines 2187–2190).
   Returns (26, N) float64 just to index out 26 winner values — 4.4 GB for CAISO.
   Replace with the winner-indexed formula the myopic solver uses (lines 1899–1903):
   resid_arr[winners] * annual_mwh_vec, etc. — same (26,) result.

────────────────────────────────────────────────────────────────

Important (wall clock)

4. Safety-net re-loop in solve_pathway (lines 1762–1826) is a verbatim copy of the
   main chunk loop. Extract a _chunk_pass(...) helper so the ~60-line kernel lives
   once. No semantic change; reduces future drift risk.

5. _finalize_run vintage scan (line 2427).
   [v for v in ledger.entries if v.cod_year <= year] runs for every one of 26 years:
   O(entries × years). Sort ledger.entries once by cod_year and slice with bisect for
   O(entries + years log entries).

6. decompose_clean_firm_tranches per-year Python loop (lines 1057–1070).
   Nuke-vs-CCS merit-order branch can be vectorized with a precomputed
   nuke_wins_y = (nuke_eff_lcoe <= ccs_eff_lcoe) bool vector, turning 26 Python
   iterations into three numpy ops.

────────────────────────────────────────────────────────────────

Minor (cleanliness / latent slowness)

7. _compute_gas_raw_2025_by_iso linear list search (line 226).
   cache_keys.index(arch_key) is O(n). The key_to_cache_row dict already exists in
   precompute_clean_peak_hour_mw; build it once at import from the dispatch cache
   and reuse here.

8. existing_gas_available_vec called twice per foresight solve (seed pass line 1983,
   realized pass line 2186). The second call is unavoidable (needs realized winners),
   but the seed result could be passed in rather than recomputed when the same cfg
   is used across multiple λ sweeps.


##HAIKU 4.5

STEP 2.3 REFACTORING RECOMMENDATIONS: WALL-CLOCK & OOM OPTIMIZATION
====================================================================

CRITICAL — Memory Killers (Enable CAISO, Fix OOM)
-------------------------------------------------

1. Foresight: Chunk the year loop (lines 2080–2173)
   - Current: Pre-allocates storage_cost_yn as dense (26, N_mixes) = 1.7 GB ERCOT / 4.2 GB CAISO
   - Fix: Apply myopic's chunk-streaming pattern → compute in (26, chunk_size) transient chunks
   - Impact: Frees 2.5 GB, enables CAISO runs, ~10% faster
   - Effort: ~50 lines

2. Foresight: Inline tranche decomposition (lines 1989–1991, 2001–2030)
   - Current: Pre-compute uprate_twh_yn, geo_twh_yn, nuke_new_twh_yn, ccs_twh_yn as (Y, N)
   - Fix: Move compute_clean_firm_tranches() call inside chunk loop
   - Impact: Frees another 1–2 GB, mathematically identical
   - Effort: ~20 lines

3. Foresight: Eliminate dense storage_cost_yn (line 2024)
   - Current: Allocate full (Y, N) for storage costs, aggregate column-by-column
   - Fix: Compute storage contribution inline within chunk scoring
   - Impact: ~500 MB freed
   - Effort: ~10 lines


HIGH — Runtime Killers (Wall-Clock Savings)
--------------------------------------------

4. Myopic: Memoize safety-net tier success (lines 1755–1832)
   - Current: Always re-scan full pool after tier 0–2 search
   - Fix: Record best_tier_found; only re-scan if best_tier == -1
   - Impact: ~2–3 sec per pathological year
   - Effort: ~5 lines

5. Foresight: Compute gas-sizing only for winners (line 2187)
   - Current: Allocate full gas_sizing_matrix (Y, N), then index winner rows
   - Fix: Compute winners_gas_y as (Y,) scalar vector directly at winner-selection time
   - Impact: ~1 sec per foresight combo
   - Effort: ~5 lines

6. Foresight: Avoid redundant _project_shares_to_endpoint calls (lines 2112–2118)
   - Current: Call per year for full pool to compute penalty
   - Fix: Memoize by endpoint; one call per endpoint, reuse for all years
   - Impact: ~2–3 sec per foresight combo
   - Effort: ~15 lines


MEDIUM — Code Quality (No Wall-Clock, Minor Memory)
---------------------------------------------------

7. Clear dispatch cache after each combo (lines 2821–2823)
   - Current: gc.collect() runs but cache lives forever
   - Fix: Clear cache via _DISPATCH_CACHE_BY_ISO[iso] = None after serialize_run_result()
   - Impact: ~100 MB session-drift prevention
   - Effort: 1 line

8. Use chunk-size constant uniformly
   - Current: _POOL_CHUNK_SIZE = 500000 defined, but hardcoded 500k elsewhere
   - Fix: Use _POOL_CHUNK_SIZE everywhere
   - Impact: Config-in-one-place maintainability
   - Effort: 3 grep/replace lines


IMPLEMENTATION ORDER
--------------------

Phase A (Memory): #1 → #2 → #3  (enables CAISO, mandatory)
Phase B (Runtime): #4 → #5 → #6  (10–15 sec savings)
Phase C (Polish): #7 → #8        (optional cleanup)

All changes are deterministic rewrites—mathematically identical outputs (float64 ≈ 1e-14 epsilon only).


##OPUS 4.7

STEP 2.3 REFACTOR — RECOMMENDATIONS (math-preserving)
=====================================================

WALL-CLOCK WINS
---------------
1. Hoist the per-year parquet write out of the 26-year loop.
   LESSONS.md #11: save_dispatch_cache() fires up to 26x per run_pathway
   call (0.1-0.3s each + 47->110 MB file growth). Thread persist=False
   through _expand_missing_archetypes / expand_cache_for_mixes; accumulate
   novel archetypes in memory; call save_dispatch_cache() once at end of
   run_pathway. Same on-disk result. ~25x fewer writes in full sweep
   (9,100 -> 350).

2. Verify the active orchestrator is _sweep_pathways_inproc.py, not
   run_pathway_sweep.py. LESSONS.md #9: subprocess-per-run pays ~5.7s
   overhead for dispatch-cache + gen-profile reload. Delete or gate the
   subprocess driver so it can't be selected by accident.

3. Cache the Stage-1 sidecar output to parquet keyed by
   (iso, threshold, ef_hash, manifest_hash). Currently the 28
   _prebuild_sidecar tasks (7 ISOs x 4 thresholds) run every invocation.
   Sidecar is deterministic in its inputs -> skip on hash hit. Saves the
   entire Stage-P0 phase on iterative re-runs.

4. Short-circuit gen-profile and hybrid-profile loads for resources an
   ISO doesn't carry. _load_gen_profiles loops 6 fuels x 7 ISOs
   unconditionally; offshore wind is empty for interior ISOs, geothermal
   only for CAISO. Use a per-ISO resource whitelist from pipeline_config.

5. Collapse the tier-argmin's four np.where(tmask, score, np.inf)
   allocations into one pass. At 500k x 4 tiers x float64 that's ~16 MB
   transient + 4x the write bandwidth per chunk. A single (k,) tier-index
   array + one grouped argmin over (tier_id * big + score) gives the same
   winner per tier. Pure reordering — bit-exact.

OOM WINS
--------
6. Port the myopic solver's 500k-row chunking pattern into the foresight
   solver. Foresight currently materializes dense (26, n) tensors:
   gas_required, gas_fuel, ratchet_fleet_y, storage_cost_yn. At CAISO
   n=4.5M that's ~900 MB per tensor, ~3.6 GB total, on top of ~1 GB
   baseline. SPEC.md's "500k-row chunks, running argmin, same total ops"
   prescription is the intended fix. BIGGEST OOM LEVER.

7. Pack the foresight pmask / cfe_mask bools as bitfields or compute them
   per-year chunk. Two (26, n) bool arrays = 225 MB at ERCOT, used only
   to gate the argmin. Either np.packbits (8x shrink) or lazily
   materialize one year's mask at a time.

8. Make module-scope ISO loads lazy and free them as each ISO finishes.
   Less urgent than originally scoped — the CI workflow already runs
   per-ISO via ISO_FILTER. Still useful for local dev and the
   _surrogate_ab_driver (which OOMed on 4-combo runs per LESSONS.md #13).

9. Memory-map the dispatch-cache parquets (pq.ParquetFile with
   memory_map=True) instead of pq.read_table. Shares the OS page cache
   across processes and keeps unused columns on disk. Requires auditing
   Stage-1 column access to keep it zero-copy.

10. Downcast the per-chunk transients to float32. The (k, 10)
    shares_nr_chunk and the dozen (k,) gas/storage buffers total ~170 MB
    per chunk in float64. Score math stays float64 (keeps achieved_cfe_pct
    and npv_at_7pct bit-exact per Phase C regression) but intermediate
    broadcast arrays can safely be float32. Verify under
    _surrogate_ab_metrics tolerance bands.

11. Release the PyArrow Table after extracting numpy views.
    _DISPATCH_CACHE_BY_ISO[iso] is kept as a full pa.Table for the
    lifetime of the ISO run, but Stage-2 only reads a handful of columns
    via .column(c).to_numpy(). Convert once, drop the Table, save
    ~50-100 MB/ISO.

12. Adopt one-combo-per-process for the surrogate A/B driver per
    LESSONS.md #13. Plumb checkpoint files so a driver can resume after
    SIGKILL. Orthogonal to step-2.3 refactor but closes the observed OOM
    regression on gVisor.

STRUCTURAL / CORRECTNESS GUARDRAILS
-----------------------------------
13. Keep the Phase C lambda=0 regression
    (phase_c_regression_lambda_zero.py) as the CI gate for the refactor.
    Enforces np.array_equal(winners) + np.allclose(cfe, rtol=0, atol=0) —
    bit-exact proof that the refactored solver reproduces the original.

14. Add a second, looser gate via _surrogate_ab_metrics tolerance bands
    (|cost_delta_pct| < 1.0, mix_drift < 3.0) to catch float32
    conversions that don't affect the argmin but drift NPV.

15. When downcasting (rec #10), accumulate sums in float64.
    achieved_cfe_pct, net_annual_cost_usd, npv_at_7pct are contract
    fields downstream readers compare bit-exactly; final reductions must
    stay float64 even if per-chunk multiplies go float32.

ORDERING
--------
Quick wins (days of work):       #1, #2, #4, #5
Biggest OOM lever:               #6 (foresight chunking)
Best RSS-per-line-changed:       #11 (drop pa.Table), #7 (pack bools)
Defer until harness ready:       #3, #10

Benchmark target: CAISO, not ERCOT. 4.5M mixes + 9D geothermal makes it
the binding case for both wall-clock and peak RSS.


