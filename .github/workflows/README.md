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
| 1a | `step1a-scored-database.yml` | **Step 1a+1b: Build Scored Mix Database** | Generates all resource fraction combos and scores them → `coarse_cache.parquet`. | ISO, **script** (generate-mixes / score-mixes), chunk_size, force flags |
| 1c | `step1c-zone-search.yml` | **Step 1c: Zone-Based Fine Search** | Zone-based fine search around PFS boundary. | ISO, thresholds |
| 1d | `step1d-fine-storage-v2.yml` | **Step 1d: Two-Pass Fine Storage** | Fine-grid storage refinement for step1c gaps. | ISO, thresholds |
| 2 | `step2-efficient-frontier.yml` | **Step 2: Efficient Frontier** | EF filter + optional EF expansion for Scenario A floors. | ISO, **script** (ef-only / expand-ef / ef-then-expand), target_mixes, dry_run |
| 2.5 | `step2.5-expand-ef.yml` | **Step 2.5: Expand EF for Scenario A** | Standalone EF expansion for per-resource floor constraints. | ISO, thresholds |
| 3 | `step3-cost-optimization.yml` | **Step 3: Cost Optimization** | Vectorized cost optimization across 5,832 sensitivity combos. 3 tracks. | ISO, **track**, sensitivity_mode |
| 4 | `step4-dispatch-cache.yml` | **Step 4: Build Dispatch Cache** | Pre-computes 8760-hour dispatch for all unique mixes per ISO. | ISO |
| 4.2 | `step4.2-track-analysis.yml` | **Step 4.2: Export & Analyze Tracks** | Exports track parquets to JSON, track cost envelopes (P10/P50/P90). | **script** (export-tracks-only / analyze-tracks-only) |

## Page-Oriented Workflows (Step 5+)

Each workflow is organized by **which dashboard page it updates**. Every multi-script workflow has a **script selector dropdown** so you can run individual scripts or ALL sequentially.

| # | Workflow | Display Name | What It Does | Scripts (sequential) |
|---|----------|-------------|--------------|---------------------|
| 5.0 | `step5.0-compute-co2.yml` | **Step 5.0: Compute CO₂ (Shared)** | Shared utility — dispatch-stack CO₂ emissions. **Run before 5.1.** | `step5_compute_co2.py` |
| 5.1 | `step5.1-update-mac-page.yml` | **Step 5.1: Update Abatement Page** | MAC stats → Optimal targets → Shared data. | **script**: mac-stats / optimal-targets / shared-data |
| 5.2 | `step5.2-update-lmp-page.yml` | **Step 5.2: Update LMP Page** | Compute synthetic LMP → Extract dashboard JS. | **script**: compute-lmp / extract-dashboard |
| 5.3 | `step5.3-update-scenarios-page.yml` | **Step 5.3: Update Scenarios Page** | Scenario A → B → Compare. | **script**: scenario-a / scenario-b / scenario-compare |
| 5.4 | `step5.4-procurement-strategies.yml` | **Step 5.4: Update Procurement Page** | 10 strategy variants → combined dashboard JS. | **strategy**: strategy1 / strategy2 / strategy3 |
| 5.5 | `step5.5-supplemental-analytics.yml` | **Step 5.5: Supplemental Analytics** | Compressed day + consequential queue (parallel). | **script**: compressed-day / consequential-queue |
| 5.6 | `step5.6-update-optimizer-dashboard.yml` | **Step 5.6: Update Optimizer Dashboard** | Compressed day profiles + shared-data.js for dashboard.html. | **script**: compressed-day / shared-data |
| 6 | `step6-generate-shared-data.yml` | **Step 6: Update Home Page / Shared Data** | Consolidates all results into `shared-data.js`. | (single script) |

## Utilities

| Workflow | Display Name | What It Does |
|----------|-------------|--------------|
| `lmp-calibration.yml` | **LMP Calibration: Validate Model** | Validates synthetic LMP model against calibration targets (weather-normalized or QA against actual data). |
| `unified-cleanup.yml` | **Unified Data Directory Cleanup** | Selectively clears pipeline step data directories. Boolean toggle per directory, ISO selector, dry-run preview. Replaces individual cleanup workflows. |
| `squash-pipeline-branches.yml` | **Squash Pipeline Branches** | Consolidates all unmerged `auto/*` branches into a single squash commit. Skips conflicts, closes old PRs, deletes merged branches. |
| `cleanup-large-files.yml` | **Cleanup Large File Blobs from History** | Rewrites git history to strip dead/overwritten data blobs. Protects all files live in HEAD. Destructive — all collaborators must re-clone. |

## Common Inputs

All workflows accept these standard inputs:

- **`iso`** — ISO region selector (ALL, CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP)
- **`branch`** — Branch to checkout (default: `master`)
- **`script`** — Which specific script to run (ALL runs the full sequence)

Additional inputs on specific workflows:
- **`thresholds`** — Comma-separated threshold list or "all" (Step 1b)
- **`track`** — Track selector: baseline, newbuild, cost-to-replace (Step 3)
- **`strategy`** — Strategy family selector (Step 5.4)
- **`fuel_level`** — Fuel price level: L/M/H (Step 5.2)

## Typical Usage Patterns

### "I just want to update the Optimizer Dashboard"
```
Step 5.6
```

### "I just want to update the Abatement page"
```
Step 5.0 (if CO₂ needs refresh) → Step 5.1
```

### "I just want to update the LMP page"
```
Step 5.2
```

### "I just want to update the Home Page"
```
Step 6
```

### "I changed cost assumptions and need to refresh everything"
```
Step 3 → Step 4 → Step 5.0 → Steps 5.1 + 5.2 + 5.3 + 5.6 (parallel) → Step 6
```

### "Full pipeline from scratch"
```
Step 0 (optional) → Step 1a → Step 1c → Step 1d → Step 2 → Step 3 → Step 4 → Step 5.0 → Steps 5.1-5.6 → Step 6
```

### "I want to clear data directories before re-running a pipeline step"
```
Unified Data Directory Cleanup (dry_run=true first to preview, then dry_run=false)
```
- Toggle each directory independently (checkboxes in the GitHub UI)
- Filter by ISO or clear ALL
- Directories are preserved — only files are removed

### "I ran a bunch of pipeline steps and want to merge all the branches at once"
```
Squash Pipeline Branches (dry_run=true first to preview, then dry_run=false)
```
- **`branch_pattern`** — Filter branches: `auto/` (all), `auto/.*step1d` (step 1d only), `auto/.*step3` (step 3 only)
- **`merge_direct`** — `true` to merge straight to master, `false` (default) to open a PR for review
- **`delete_branches`** — Cleans up merged remote branches (default: true)
- **`close_prs`** — Auto-closes superseded PRs (default: true)
