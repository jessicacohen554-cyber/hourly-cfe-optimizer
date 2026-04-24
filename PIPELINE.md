# Pipeline Architecture

Reference doc for the 8-step optimization pipeline. Extracted from CLAUDE.md (Apr 2026) to keep CLAUDE.md lean. For project decisions see `SPEC.md`. For dashboard CSS/HTML standards see `DESIGN_SYSTEM.md`.

## Overview

**8-Step Pipeline (0–7)** — Step 1 expensive (hours), Steps 2–7 cheap (seconds to minutes). Only re-run what changed.

**Naming convention**: `.1/.2/.3` = sequential sub-steps (must run in order). `A/B/C` = parallel scripts (can run simultaneously).

**Key principle**: Steps 2–7 are cheap (seconds to minutes). Step 1 is expensive (hours). Default to Step 2.2 + post-processing unless physics assumptions change.

**Data contract**: Step 2.2 must NOT change existing columns in `shared-data.js` or `overprocure_results.json` — only ADD new columns/fields.

---

## Step 0: Data Acquisition

`step0_*.py`, 9 scripts, annual cadence:
- `step0_fetch_all_data.py`, `step0_fetch_egrid.py`, `step0_extract_egrid_baselines.py`, `step0_fetch_eia_multiyear.py`, `step0_fetch_lmp_2025.py`, `step0_fetch_offshore_wind.py`, `step0_fix_offshore_wind.py`, `step0_fix_utc_profiles.py`, `step0_consolidate_miso_spp.py`

## Step 1: Physics Feasible Space (sequential sub-steps)

- **Monolithic**: `step1_pfs_generator.py` — runs full PFS generation in one process.
- **Modular (CI/CD)**:
  - **1.1** `step1_1a_generate_mixes.py` + `step1_1b_score_mixes.py` — Generate + score coarse mixes.
  - **1.2** `step1_2_zone_search.py` — Zone search around promising regions.
  - **1.3** `step1_3_floor_aware_pfs.py` — Floor-aware PFS augmentation (50-80%).
  - **1.4** `step1_4_fine_grid_pfs.py` — Fine-grid PFS (40-75%).
  - **1.5** `step1_5_storage_refinement.py` — Storage dispatch refinement (battery/LDES/H2 grid).
- Storage grids are the union of V1 (near-term) and V2 (2050-oriented) caps. Floor/fine mix augmentation always on.
- Utilities: `step1_prior_windows.py` (search window computation from prior EF results).
- 8–10D adaptive grid search: base resources (clean_firm, solar, wind, hydro) + 4 hybrid dimensions (solar_batt4, solar_batt8, wind_batt4, wind_batt8) + procurement sweep + battery dispatch (4hr 85% RTE, 8hr 85% RTE) + LDES dispatch (100hr 50% RTE). CAISO uses 9D (adds geothermal). With offshore wind where applicable, up to 10D. Hybrid profiles are pre-computed 8760 shapes loaded via `dispatch_utils.get_supply_profiles()`.
- Output: `data/step1-pfs/`. **Only re-run if dispatch logic, generation profiles, or demand curves change.**

## Step 2: Optimization (sequential: 2.1 → 2.2)

- **2.1** `step2_1_efficient_frontier.py` — Extracts non-dominated mixes from PFS. Filters existing gen utilization, procurement minimization, strict dominance removal. Output: `data/step2.1-ef/`.
- **2.2A** `step2_2a_cost_optimization.py` — Track 1 baseline: vectorized cross-eval of EF mixes under 5,832 combos (17,496 CAISO). Merit-order tranche pricing for clean firm (uprate → geothermal → cheapest of nuclear/CCS). ─┐ parallel
- **2.2B** `step2_2b_track_nb_ctr.py` — Track 2 (newbuild) + Track 3 (cost-to-replace): greenfield cost analysis. Demand growth sweep (25 years × 3 growth rates) with FOAK→NOAK learning curves (Wright's Law). ─┘
- Includes NEISO winter gas pipeline constraint (+$13.13/MWh CCS adder), 45Q correction ($27.5/MWh).
- Output: `data/step2.2-cost/`. **Run when cost assumptions change. No physics re-run needed.**

- **2.3** `step_2_3_pathway_optimizer.py` — Pathway optimizer: 7 ISOs × 5 pathways × 4 endpoints, 2025–2050. Two solver modes: `solve_pathway` (myopic, P1/P1a/P2a/P2b/P3) and `solve_pathway_with_foresight` (P3 foresight, λ-calibrated, P2a/P2b/P3 only). Output: `data/step2.3-pathway/<ISO>/` (20 myopic + 12 foresight = 32 JSONs per ISO). Driver: `scripts/run_pathway_sweep.py`. ← needs Steps 2.1 + 3A.

  **Band-loading contract — read `step2_1_efficient_frontier.py` before touching `_read_ef_table` or `_load_or_build_peakclean`:** Step 2.1 writes mixes to non-overlapping bands — a mix with score=85 lives in `band=80`, NOT `band=90`. Consumers must load ALL bands ≥ the target threshold: `glob('step_2_1_EF_{iso}_*.parquet')` excluding `*_peakclean.*`, then `pa.concat_tables(tbls, promote_options='default')`, zero-fill missing resource columns per band (some bands lack `ccs_ccgt`). Loading only the target band returns an endpoint-compliant-only pool where every candidate already passes every ratchet rung — the ladder becomes a no-op (root cause of Phase D pool defect, all 7 ISOs).

  **Pool split by solver mode (NOT interchangeable):** `solve_pathway` → full concatenated pool, streamed in 500k-row chunks. `solve_pathway_with_foresight` → resource-share-capped pool: max share per resource across step 2.2A's 4 endpoint targets + 5pp slack on zero-target resources; verify post-filter n≥99 is non-trivial before running (CAISO/ERCOT/NYISO/PJM needed 5pp vs 0.5pp slack to stay above 75 mixes).

## Step 3: Caches (parallel)

- **3A** `step3a_build_dispatch_cache.py` — Pre-computes 8,760-hour dispatch for all unique mixes. Versioned NPZ cache (v2) with per-resource matched/surplus + charge profiles. ← needs Step 2. Output: `data/step3-dispatch/`.
- **3B** `step3b_mac_queue.py` — Path-dependent MAC queue for consequential deployment. Reads raw PFS (Step 1) + shared utilities, NOT Step 2 output. ← needs Step 1 only. Output: `data/step3-dispatch/`.

## Step 4: Analysis (two parallel tiers: 4.1 → 4.2)

**Tier 4.1** (parallel, mixed deps on Steps 2/3):
- **4.1A** `step4_1a_fossil_dispatch.py` — CO₂ + LMP: fossil dispatch-stack model, merit-order retirement (coal → oil → gas), synthetic hourly LMP. ← needs 3A. Output: `data/step4-analysis/co2_results/`, `data/step4-analysis/lmp/`.
- **4.1A** `step4_1a_augment_capacity_rev.py` — Augments capacity revenue data onto LMP results.
- **4.1B** `step4_1b_compress_day_profiles.py` — 24-hour representative day profiles from dispatch cache. ← needs 3A.
- **4.1C** `step4_1c_compute_mac_stats.py` — 6 MAC metrics: average fan (P10/P50/P90), stepwise marginal, monotonic envelope, path-constrained. ANOVA decomposition. Crossover vs DAC/SCC/ETS. ← needs Step 2.
- **4.1D** `step4_1d_compute_optimal_targets.py` — Optimal CFE target per ISO via marginal MAC × DAC crossover (PCHIP spline). 3×3 grid-cost × DAC-scenario matrix. No-regrets resource analysis. ← needs Step 2. Output: `optimal_targets.json` + `dashboard/js/optimal-target-data.js`.
- **4.1E** `step4_1e_export_tracks.py` — Exports track parquets (NB + CTR) to `track_results.json`. ← needs Step 2.

**Tier 4.2** (parallel, needs specific 4.1 outputs):
- **4.2A** `step4_2a_extract_resource_density.py` — Resource density analysis across thresholds. ← needs 4.1D.
- **4.2B** `step4_2b_analyze_storage.py` — Battery/LDES utilization, dispatch patterns, capacity factor analysis. ← needs Step 2 + 4.1B.
- **4.2C** `step4_2c_analyze_tracks.py` — Track cost envelopes (P10/P50/P90), resource mix differentials. ← needs 4.1E.

## Step 5: Scenarios & Procurement (parallel)

- **5.2B** `step5_2b_strategy_consequential.py` — Strategy 1 (A/B/C): cross-regional consequential netting under 3 emission baselines. ← needs 3B. ─┐
- **5.2C** `step5_2c_strategy_hourly.py` — Strategy 2 (A/B/C): hourly matching same-ISO with existing clean credit variants. │ parallel
- **5.2D** `step5_2d_strategy_annual.py` — Strategy 3 (A/B/C/D): annual matching 2×2 matrix. ← needs Step 2. │
- **5.2E** `step5_2e_wrights_law_curves.py` — Wright's Law learning curves & critical mass threshold. ← independent. ─┘
- Shared utilities: `procurement_utils.py` (SSS allocation, EAC pricing, LMP feedback, PPA premiums, learning curves, 25-year timeline).

## Step 6: Policy Analysis (sequential then parallel)

- **6.1** `step6_1_smartargets.py` — Regional SMARTargets modeling. ← needs Step 2 + 4.1A (LMP engine).
- **6.1B** `step6_1b_dashboard_data.py` — Converts SMARTargets parquets to dashboard JS.
- **6.2A** `step6_2a_ipp_smartargets.py` — IPP fleet modeling applied to SMARTargets results. ← needs 6.1. ─┐ parallel
- **6.2B** `step6_2b_nuclear_retirement.py` — Nuclear stranding risk analysis. ← needs 6.1 + Step 2. ─┘

## Step 7: Dashboard Aggregation (parallel data gen, then sequential)

- **7.1A** `step7_1a_generate_shared_data.py` — Extracts all results into `dashboard/js/shared-data.js`. SBTi milestone mapping, DAC trajectory projections, LCOE/transmission tables for client-side repricing. Aggregates all upstream outputs. ─┐
- **7.1B** `step7_1b_extract_deployment_data.py` — Deployment queue data for dashboard. │
- **7.1C** `step7_1c_generate_foak_noak.py` — FOAK/NOAK learning curve data. │ parallel
- **7.1E** `step7_1e_dispatch_deployment.py` — Dispatch deployment visualization data. │
- **7.1F** `step7_1f_extract_hourly_comparison.py` — Hourly comparison data. │
- **7.1G** `step7_1g_extract_use_case_data.py` — Use case analysis data. ─┘
- **7.2** `step7_2_extract_no_regrets.py` — Optimal targets and no-regrets resource investments from crossover analysis. ← needs 7.1A.

## Utility Modules (no step prefix)

- `pipeline_config.py` — **Single source of truth** for all shared constants (LCOE tables, fuel adjustments, CCS caps, storage parameters, wholesale prices). All step scripts import from here.
- `dispatch_utils.py` — Dispatch reconstruction, supply profiles, fossil retirement, cache I/O. Imports constants from `pipeline_config`.
- `scenario_common.py` — Shared Scenario A/B logic: cost tables, demand growth, learning curves, EF/PFS loading. Imports constants from `pipeline_config`.
- `eia_data_io.py` — Standardized EIA multi-year profile loading.
- `lmp_engine.py` — Synthetic hourly LMP from merit-order fossil dispatch. Used by `step4_1a`, `step6_1`, `calibrate_lmp_model`.
- `calibrate_lmp_model.py` — LMP model validation against actual ISO data.
- Other: `anthropic_image_utils.py`, `extract_shared_data.py`, `analyze_pjm_lmp.py`, `analyze_results.py`, `sensitivity_analysis.py`.

## GitHub Actions (~21 workflows, all `workflow_dispatch`)

- Core pipeline: `step1-1-scored-database.yml` → `step1-2-3-zone-floor.yml` → `step1-4-5-fine-storage.yml` → `step2-1-efficient-frontier.yml` → `step2-2-cost-optimization.yml`
- Caches: `step3a-dispatch-cache.yml`, `step3b-mac-queue.yml` (parallel)
- Analysis: `step4-1b-day-profiles.yml`, `step4-tracks.yml`, `step4-2b-storage-analysis.yml`, `step4-derived-analytics.yml`
- Scenarios & Procurement: `step5-procurement.yml`, `step5-2e-wrights-law.yml`
- Policy: `step6-1-smartargets-reference.yml`, `step6-1-smartargets-power-nz.yml`, `step6-1-smartargets-economy-nz.yml`, `step6-1-smartargets-quick-transition.yml`
- Dashboard: `step7-dashboard-data.yml`
- Data: `step0-fetch-lmp-data.yml`, `step0-fetch-offshore-wind.yml`
- See `.github/workflows/README.md` for full docs and common patterns.

## Data Directories

- `data/step1-pfs/` — Step 1: PFS + storage parquets
- `data/step2.1-ef/` — Step 2.1: efficient frontier parquets
- `data/step2.2-cost/` — Step 2.2: cost optimization parquets
- `data/step3-dispatch/` — Step 3: dispatch cache (NPZ) + MAC queue
- `data/step4-analysis/` — Step 4: CO₂, LMP, MAC stats, optimal targets, tracks, building blocks, resource density
- `data/step5-scenarios/` — Step 5: scenario comparison, procurement strategies
- `data/step5-wrights/` — Step 5.2E: Wright's Law learning curves
- `data/step6-smartargets/` — Step 6: SMARTargets, IPP, nuclear retirement
