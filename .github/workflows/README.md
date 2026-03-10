# GitHub Actions Workflows

All workflows are triggered manually via `workflow_dispatch` from the GitHub Actions UI.
Every workflow creates a fresh branch, commits results, and opens a PR.

## Naming Convention

- **`stepN-description.yml`** — Pipeline steps (N = major step number)
- **`stepN-Na-description.yml`** — Sub-step workflows (e.g., step4-1b)
- **`lmp-calibration.yml`** — Standalone utility (not a pipeline step)

**Step numbering**: `.1/.2/.3` = sequential sub-steps. `A/B/C` = parallel scripts.

## Pipeline Order

Run these in sequence. Steps at the same level with A/B suffixes can run in parallel.

| Step | Workflow | Display Name | What It Does | Inputs |
|------|----------|-------------|--------------|--------|
| 0 | `step0-fetch-lmp-data.yml` | **Step 0: Fetch Actual LMP Data** | Fetches actual DA hourly LMP from ISO APIs via gridstatus. | ISO, year, force |
| 0 | `step0-fetch-offshore-wind.yml` | **Step 0: Fetch Offshore Wind** | Fetches offshore wind profiles from NREL Wind Toolkit API. | — |
| 1.1 | `step1-1-scored-database.yml` | **Step 1.1: Build Scored Mix Database** | Generates all resource fraction combos and scores them. | ISO, script, chunk_size, force |
| 1.2–1.3 | `step1-2-3-zone-floor.yml` | **Step 1.2-1.3: Zone Search + Floor PFS** | Zone-based fine search + floor-aware PFS. | ISO, thresholds |
| 1.4–1.5 | `step1-4-5-fine-storage.yml` | **Step 1.4-1.5: Fine Grid + Storage Refinement** | Fine-grid PFS + storage gap filling. | ISO, thresholds |
| 2.1 | `step2-1-efficient-frontier.yml` | **Step 2.1: Efficient Frontier** | EF filter. | ISO, target_mixes, dry_run |
| 2.2 | `step2-2-cost-optimization.yml` | **Step 2.2: Cost Optimization** | Vectorized cost optimization (5,832 combos × 3 tracks). | ISO, track, sensitivity_mode |
| 3A | `step3a-dispatch-cache.yml` | **Step 3A: Build Dispatch Cache** | Pre-computes 8760-hour dispatch for all unique mixes. | ISO |
| 3B | `step3b-mac-queue.yml` | **Step 3B: MAC Queue** | MAC-optimized consequential queue + Scenario A export. | ISO |

## Analysis Workflows (Steps 4–7)

Post-processing analysis pipelines. Step 4.1 scripts can run in parallel after Steps 2/3.
Steps 5–6 depend on Step 4 outputs. Step 7 runs last.

| Step | Workflow | Display Name | What It Does | Scripts |
|------|----------|-------------|--------------|---------|
| 4 | `step4-derived-analytics.yml` | **Step 4: Derived Analytics** | MAC stats + optimal targets + building blocks + resource density. | `step4_1c_*.py`, `step4_1d_*.py`, `step4_1f_*.py`, `step4_2a_*.py` |
| 4.1B | `step4-1b-day-profiles.yml` | **Step 4.1B: Compress Day Profiles** | 24-hour representative day profiles. | `step4_1b_compress_day_profiles.py` |
| 4 | `step4-tracks.yml` | **Step 4: Track Analysis** | Export tracks + track cost envelopes. | `step4_1e_export_tracks.py`, `step4_2c_analyze_tracks.py` |
| 4.2B | `step4-2b-storage-analysis.yml` | **Step 4.2B: Storage Analysis** | Battery/LDES utilization, dispatch patterns. | `step4_2b_analyze_storage.py` |
| 5 | `step5-scenarios.yml` | **Step 5: Scenarios** | Scenario B + Compare (Scenario A from MAC queue). | `step5_1_scenario_hourly.py`, `step5_2a_scenario_comparison.py` |
| 5 | `step5-procurement.yml` | **Step 5: Procurement Strategies** | 10 strategy variants → combined dashboard JS. | `step5_2b_*.py`, `step5_2c_*.py`, `step5_2d_*.py` |
| 5.2E | `step5-2e-wrights-law.yml` | **Step 5.2E: Wright's Law Learning Curves** | FOAK→NOAK learning curve projections. | `step5_2e_wrights_law_curves.py` |
| 6.1 | `step6-1-smartargets-reference.yml` | **Step 6.1: SMARTargets** | Regional SMARTargets — Reference scenario. | `step6_1_smartargets.py` |
| 6.1 | `step6-1-smartargets-power-nz.yml` | **Step 6.1: SMARTargets — Power Sector NZ** | Regional SMARTargets — Power Sector NZ. | `step6_1_smartargets.py` |
| 6.1 | `step6-1-smartargets-economy-nz.yml` | **Step 6.1: SMARTargets — Economy-Wide NZ** | Regional SMARTargets — Economy-Wide NZ. | `step6_1_smartargets.py` |
| 6.1 | `step6-1-smartargets-quick-transition.yml` | **Step 6.1: SMARTargets — Quick Transition** | Regional SMARTargets — Quick Transition. | `step6_1_smartargets.py` |
| 7 | `step7-dashboard-data.yml` | **Step 7: Dashboard Data** | Consolidates all results into `shared-data.js`. **Run this last.** | `step7_1a_generate_shared_data.py` |

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
- **`thresholds`** — Comma-separated threshold list or "all" (Step 1.2–1.5)
- **`track`** — Track selector: baseline, newbuild, cost-to-replace (Step 2.2)
- **`strategy`** — Strategy family selector (Step 5)
- **`fuel_level`** — Fuel price level: L/M/H (Step 4.1A)

## Typical Usage Patterns

### "I just want to update the Optimizer Dashboard"
```
Step 4.1B → Step 7
```

### "I just want to update the Abatement page"
```
Step 4.1A (if CO₂ needs refresh) → Step 4 (derived analytics) → Step 7
```

### "I just want to regenerate shared-data.js and deploy"
```
Step 7
```

### "I changed cost assumptions and need to refresh everything"
```
Step 2.2 → Step 3 → Step 4 (parallel) → Step 5+6 → Step 7
```

### "Full pipeline from scratch"
```
Step 0 → Step 1.1 → 1.2–1.3 → 1.4–1.5 → Step 2.1 → 2.2 → Step 3A ∥ 3B → Step 4 → Step 5+6 → Step 7
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
- **`branch_pattern`** — Filter branches: `auto/` (all), `auto/.*step1` (step 1 only)
- **`merge_direct`** — `true` to merge straight to master, `false` (default) to open a PR
- **`delete_branches`** — Cleans up merged remote branches (default: true)
- **`close_prs`** — Auto-closes superseded PRs (default: true)
