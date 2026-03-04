# GitHub Actions Workflows

All workflows are triggered manually via `workflow_dispatch` from the GitHub Actions UI.
Every workflow creates a fresh branch, commits results, and opens a PR.

## Naming Convention

- **`stepN-description.yml`** — Core pipeline steps (run in order)
- **`stepNx-description.yml`** — Sub-step workflows (e.g., step5a, step5b)
- **`lmp-calibration.yml`** — Standalone utility (not a pipeline step)

## Pipeline Order

Run these in sequence. Each step depends on the output of the previous step.

| # | Workflow | Display Name | What It Does | Inputs |
|---|----------|-------------|--------------|--------|
| 0 | `step0-fetch-lmp-data.yml` | **Step 0: Fetch Actual LMP Data** | Fetches actual DA hourly LMP from ISO APIs via gridstatus. | ISO, year, force |
| 0 | `step0-fetch-offshore-wind.yml` | **Step 0: Fetch Offshore Wind** | Fetches offshore wind profiles from NREL Wind Toolkit API. | — |
| 1a | `step1a-scored-database.yml` | **Step 1A: Build Scored Mix Database** | Generates all resource fraction combos and scores them → `coarse_cache.parquet`. | ISO, **script** (generate-mixes / score-mixes), chunk_size, force flags |
| 1b | `step1b-zone-search.yml` | **Step 1B: Zone-Based Fine Search** | Zone-based fine search around PFS boundary. | ISO, thresholds |
| 1c | `step1c-storage-refinement.yml` | **Step 1C: Storage Refinement** | Fine-grid storage refinement for step1b gaps. | ISO, thresholds |
| 2 | `step2-efficient-frontier.yml` | **Step 2: Efficient Frontier** | EF filter. | ISO, target_mixes, dry_run |
| 3 | `step3-cost-optimization.yml` | **Step 3: Cost Optimization** | Vectorized cost optimization across 5,832 sensitivity combos. 3 tracks. | ISO, **track**, sensitivity_mode |
| 4 | `step4-dispatch-cache.yml` | **Step 4: Build Dispatch Cache** | Pre-computes 8760-hour dispatch for all unique mixes per ISO. | ISO |

## Analysis Workflows (Steps 5–9)

Post-processing analysis pipelines. Steps 5A–5F can run in parallel after Step 4.
Steps 6–8 depend on Step 5 outputs. Step 9 runs last.

| # | Workflow | Display Name | What It Does | Scripts |
|---|----------|-------------|--------------|---------|
| 5A | `step5a-compute-co2.yml` | **Step 5A: Compute CO₂** | Dispatch-stack CO₂ emissions. **Run before Step 6.** | `step5a_compute_co2.py` |
| 5B | `step5b-compute-lmp.yml` | **Step 5B: Compute LMP** | Synthetic 8760-hour LMP per ISO × threshold × fuel level. | `step5b_compute_lmp_prices.py` |
| 5C | `step5c-dashboard-update.yml` | **Step 5C: Dashboard Day Profiles** | 24-hour representative day profiles per unique mix. | `step5c_compress_day_profiles.py` |
| 5CD | `step5cd-supplemental.yml` | **Step 5C+D: Supplemental Analytics** | Compressed day + consequential queue (parallel). | `step5c_compress_day_profiles.py`, `step5d_deployment_queue.py` |
| 5E | `step5e-track-analysis.yml` | **Step 5E: Track Analysis** | Export tracks + track cost envelopes (P10/P50/P90). | `step5e_export_tracks.py`, `step6c_analyze_tracks.py` |
| 5F | `step5f-storage-analysis.yml` | **Step 5F: Storage Analysis** | Battery/LDES utilization, dispatch patterns, capacity factors. | `step5f_analyze_storage.py` |
| 6 | `step6-derived-analytics.yml` | **Step 6: Derived Analytics** | MAC stats + Optimal targets. | `step6a_compute_mac_stats.py`, `step6b_compute_optimal_targets.py` |
| 7 | `step7-scenario-comparison.yml` | **Step 7: Scenario Comparison** | Scenario A → B → Compare. | `step7a_scenario_consequential.py`, `step7b_scenario_hourly.py`, `step7c_scenario_comparison.py` |
| 8 | `step8-procurement-strategies.yml` | **Step 8: Procurement Strategies** | 10 strategy variants → combined dashboard JS. | `step8a_strategy_consequential.py`, `step8b_strategy_hourly.py`, `step8c_strategy_annual.py` |
| 8D | `step8d-wrights-law.yml` | **Step 8D: Wright's Law Curves** | FOAK→NOAK learning curve projections. | `step8d_wrights_law_curves.py` |
| 9 | `step9-generate-shared-data.yml` | **Step 9: Generate Shared Data** | Consolidates all results into `shared-data.js`. **Run this last.** | `step9a_generate_shared_data.py` |

## Utilities

| Workflow | Display Name | What It Does |
|----------|-------------|--------------|
| `lmp-calibration.yml` | **LMP Calibration: Validate Model** | Validates synthetic LMP model against calibration targets. |
| `unified-cleanup.yml` | **Unified Data Directory Cleanup** | Selectively clears pipeline step data directories. Boolean toggle per directory, ISO selector, dry-run preview. |
| `squash-pipeline-branches.yml` | **Squash Pipeline Branches** | Consolidates all unmerged `auto/*` branches into a single squash commit. |
| `cleanup-large-files.yml` | **Cleanup Large File Blobs from History** | Rewrites git history to strip dead/overwritten data blobs. Destructive — all collaborators must re-clone. |
| `run-tests.yml` | **Run Tests** | Runs test suite. |
| `deploy-pages.yml` | **Deploy Pages** | Deploys dashboard to GitHub Pages. |

## Common Inputs

All workflows accept these standard inputs:

- **`iso`** — ISO region selector (ALL, CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP)
- **`branch`** — Branch to checkout (default: `master`)
- **`script`** — Which specific script to run (ALL runs the full sequence)

Additional inputs on specific workflows:
- **`thresholds`** — Comma-separated threshold list or "all" (Step 1B)
- **`track`** — Track selector: baseline, newbuild, cost-to-replace (Step 3)
- **`strategy`** — Strategy family selector (Step 8)
- **`fuel_level`** — Fuel price level: L/M/H (Step 5B)

## Typical Usage Patterns

### "I just want to update the Optimizer Dashboard"
```
Step 5C → Step 9
```

### "I just want to update the Abatement page"
```
Step 5A (if CO₂ needs refresh) → Step 6 → Step 9
```

### "I just want to update the LMP page"
```
Step 5B → Step 9
```

### "I just want to regenerate shared-data.js and deploy"
```
Step 9
```

### "I changed cost assumptions and need to refresh everything"
```
Step 3 → Step 4 → Step 5A → Steps 5B + 5C + 6 + 7 (parallel) → Step 9
```

### "Full pipeline from scratch"
```
Step 0 (optional) → Step 1A → Step 1B → Step 1C → Step 2 → Step 3 → Step 4 → Steps 5A–5F (parallel) → Steps 6–8 → Step 9
```

### "I want to clear data directories before re-running a pipeline step"
```
Unified Data Directory Cleanup (dry_run=true first to preview, then dry_run=false)
```
- Toggle each directory independently (checkboxes in the GitHub UI)
- Filter by ISO or clear ALL

### "I ran a bunch of pipeline steps and want to merge all the branches at once"
```
Squash Pipeline Branches (dry_run=true first to preview, then dry_run=false)
```
- **`branch_pattern`** — Filter branches: `auto/` (all), `auto/.*step1d` (step 1d only)
- **`merge_direct`** — `true` to merge straight to master, `false` (default) to open a PR
- **`delete_branches`** — Cleans up merged remote branches (default: true)
- **`close_prs`** — Auto-closes superseded PRs (default: true)
