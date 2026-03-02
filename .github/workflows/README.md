# GitHub Actions Workflows

All workflows are triggered manually via `workflow_dispatch` from the GitHub Actions UI.
Every workflow creates a fresh branch, commits results, and opens a PR.

## Naming Convention

- **`stepN-description.yml`** — Core pipeline steps (run in order)
- **`stepN.X-description.yml`** — Page-oriented or sub-step workflows
- **`lmp-calibration.yml`** — Standalone utility (not a pipeline step)

## Pipeline Order

Run these in sequence. Each step depends on the output of the previous step.

| # | Workflow | Display Name | What It Does | Inputs |
|---|----------|-------------|--------------|--------|
| 0 | `step0-fetch-lmp-data.yml` | **Step 0: Fetch Actual LMP Data** | Fetches actual DA hourly LMP from ISO APIs via gridstatus. | ISO, year, force |
| 1a | `step1a-scored-database.yml` | **Step 1a: Build Scored Mix Database** | Generates all resource fraction combos and scores them → `coarse_cache.parquet`. | ISO, **script** (generate-mixes / score-mixes), chunk_size, force flags |
| 1b | `step1b-build-pfs.yml` | **Step 1b: Build PFS from Scored Database** | Mines Physics Feasible Space from scored DB per threshold. | ISO, **thresholds** |
| 2 | `step2-efficient-frontier.yml` | **Step 2: Efficient Frontier** | EF filter + optional EF expansion for Scenario A floors (merged from step 2.5). | ISO, **script** (ef-only / expand-ef / ef-then-expand), target_mixes, dry_run |
| 3 | `step3-cost-optimization.yml` | **Step 3: Cost Optimization** | Vectorized cost optimization across 5,832 sensitivity combos. 3 tracks. | ISO, **track**, sensitivity_mode |
| 4 | `step4-gas-ccs.yml` | **Step 4: Gas CCS Post-Processing** | NEISO winter pipeline constraint, 45Q correction, resource adequacy margin. | ISO |
| 5 | `step5-dispatch-cache.yml` | **Step 5: Dispatch Cache & Track Analysis** | Pre-computes 8760-hour dispatch cache, exports/analyzes track results. | ISO, **script** (dispatch-cache / export-tracks / analyze-tracks) |

## Page-Oriented Workflows (Step 6+)

Each workflow is organized by **which dashboard page it updates**. Every multi-script workflow has a **script selector dropdown** so you can run individual scripts or ALL sequentially.

| # | Workflow | Display Name | What It Does | Scripts (sequential) |
|---|----------|-------------|--------------|---------------------|
| 6.0 | `step6.0-compute-co2.yml` | **Step 6.0: Compute CO₂ (Shared)** | Shared utility — dispatch-stack CO₂ emissions. **Run before 6.1.** | `step6_recompute_co2.py` |
| 6.1 | `step6.1-update-mac-page.yml` | **Step 6.1: Update Abatement Page** | MAC stats → Optimal targets → Shared data. | **script**: mac-stats / optimal-targets / shared-data |
| 6.2 | `step6.2-update-lmp-page.yml` | **Step 6.2: Update LMP Page** | Compute synthetic LMP → Extract dashboard JS. | **script**: compute-lmp / extract-dashboard |
| 6.3 | `step6.3-update-scenarios-page.yml` | **Step 6.3: Update Scenarios Page** | Scenario A → B → Compare. | **script**: scenario-a / scenario-b / scenario-compare |
| 6.4 | `step6.4-procurement-strategies.yml` | **Step 6.4: Update Procurement Page** | 10 strategy variants → combined dashboard JS. | **strategy**: strategy1 / strategy2 / strategy3 |
| 6.5 | `step6.5-supplemental-analytics.yml` | **Step 6.5: Supplemental Analytics** | Compressed day + consequential queue (parallel). | **script**: compressed-day / consequential-queue |
| 6.6 | `step6.6-update-optimizer-dashboard.yml` | **Step 6.6: Update Optimizer Dashboard** | Compressed day profiles + shared-data.js for dashboard.html. | **script**: compressed-day / shared-data |
| 7 | `step7-generate-shared-data.yml` | **Step 7: Update Home Page / Shared Data** | Consolidates all results into `shared-data.js`. | (single script) |

## Utilities

| Workflow | Display Name | What It Does |
|----------|-------------|--------------|
| `lmp-calibration.yml` | **LMP Calibration: Validate Model** | Validates synthetic LMP model against calibration targets (weather-normalized or QA against actual data). |
| `squash-pipeline-branches.yml` | **Squash Pipeline Branches** | Consolidates all unmerged `auto/*` branches into a single squash commit. Skips conflicts, closes old PRs, deletes merged branches. |

## Common Inputs

All workflows accept these standard inputs:

- **`iso`** — ISO region selector (ALL, CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP)
- **`branch`** — Branch to checkout (default: `master`)
- **`script`** — Which specific script to run (ALL runs the full sequence)

Additional inputs on specific workflows:
- **`thresholds`** — Comma-separated threshold list or "all" (Step 1b)
- **`track`** — Track selector: baseline, newbuild, cost-to-replace (Step 3)
- **`strategy`** — Strategy family selector (Step 6.4)
- **`fuel_level`** — Fuel price level: L/M/H (Step 6.2)

## Typical Usage Patterns

### "I just want to update the Optimizer Dashboard"
```
Step 6.6
```

### "I just want to update the Abatement page"
```
Step 6.0 (if CO₂ needs refresh) → Step 6.1
```

### "I just want to update the LMP page"
```
Step 6.2
```

### "I just want to update the Home Page"
```
Step 7
```

### "I changed cost assumptions and need to refresh everything"
```
Step 3 → Step 4 → Step 5 → Step 6.0 → Steps 6.1 + 6.2 + 6.3 + 6.6 (parallel) → Step 7
```

### "Full pipeline from scratch"
```
Step 0 (optional) → Step 1a → Step 1b → Step 2 → Step 3 → Step 4 → Step 5 → Step 6.0 → Steps 6.1-6.6 → Step 7
```

### "I ran a bunch of pipeline steps and want to merge all the branches at once"
```
Squash Pipeline Branches (dry_run=true first to preview, then dry_run=false)
```
- **`branch_pattern`** — Filter branches: `auto/` (all), `auto/.*step1d` (step 1d only), `auto/.*step3` (step 3 only)
- **`merge_direct`** — `true` to merge straight to master, `false` (default) to open a PR for review
- **`delete_branches`** — Cleans up merged remote branches (default: true)
- **`close_prs`** — Auto-closes superseded PRs (default: true)
