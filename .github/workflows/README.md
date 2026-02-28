# GitHub Actions Workflows

All workflows are triggered manually via `workflow_dispatch` from the GitHub Actions UI.
Every workflow creates a fresh branch, commits results, and opens a PR.

## Naming Convention

- **`stepN-description.yml`** — Core pipeline steps (run in order)
- **`stepN.X-description.yml`** — Sub-steps within a major step
- **`lmp-calibration.yml`** — Standalone utility (not a pipeline step)

## Pipeline Order

Run these in sequence. Each step depends on the output of the previous step.

| # | Workflow | Display Name | What It Does | Inputs |
|---|----------|-------------|--------------|--------|
| 0 | `step0-fetch-lmp-data.yml` | **Step 0: Fetch Actual LMP Data** | Fetches actual DA hourly LMP from ISO APIs via gridstatus. Calibration data for synthetic LMP model. | ISO, year, force |
| 1a | `step1a-scored-database.yml` | **Step 1a: Build Scored Mix Database** | Generates all resource fraction combos and scores them → `coarse_cache.parquet`. Foundation that Step 1b mines. | ISO, chunk_size, force flags |
| 1b | `step1b-build-pfs.yml` | **Step 1b: Build PFS from Scored Database** | Mines Physics Feasible Space from scored DB per threshold. Auto-commits each threshold as it completes. | ISO, **thresholds** |
| 2 | `step2-efficient-frontier.yml` | **Step 2: Efficient Frontier Filter** | Extracts Efficient Frontier from PFS. Filters ~21M → ~1.8M rows via dominance removal. | ISO |
| 2.5 | `step2.5-expand-ef.yml` | **Step 2.5: Expand EF for Scenario A** | Augments step3 feasible parquets with floor-compatible PFS mixes for Scenario A. | ISO, target_mixes, dry_run |
| 3 | `step3-cost-optimization.yml` | **Step 3: Cost Optimization** | Vectorized cost optimization across 5,832 sensitivity combos. 3 tracks: baseline, new-build, cost-to-replace. | ISO, **track**, sensitivity_mode |
| 4 | `step4-gas-ccs.yml` | **Step 4: Gas CCS Post-Processing** | NEISO winter pipeline constraint, 45Q correction, resource adequacy margin, CCS vs LDES crossover. | ISO |
| 5 | `step5-dispatch-cache.yml` | **Step 5: Dispatch Cache & Track Analysis** | Pre-computes 8760-hour dispatch cache, exports track results, analyzes cost envelopes. | ISO, script selector |

## Page-Oriented Workflows (Step 6+)

These are organized by **which dashboard page they update**. Each runs the right scripts in the right sequence to fully populate that page's data.

| # | Workflow | Display Name | What It Does | Scripts (sequential) |
|---|----------|-------------|--------------|---------------------|
| 6.0 | `step6.0-compute-co2.yml` | **Step 6.0: Compute CO₂ (Shared)** | Shared utility — computes dispatch-stack CO₂ emissions. **Run before 6.1.** | `step6_recompute_co2.py` |
| 6.1 | `step6.1-update-mac-page.yml` | **Step 6.1: Update Abatement Page** | MAC stats → Optimal targets → Generate shared data. Everything needed for the Abatement page. | `step6_compute_mac_stats.py` → `step6_compute_optimal_targets.py` → `step7_generate_shared_data.py` |
| 6.2 | `step6.2-update-lmp-page.yml` | **Step 6.2: Update LMP Page** | Compute synthetic LMP → Extract dashboard JS. Everything needed for the Wholesale Price page. | `step6_compute_lmp_prices.py` → `extract_lmp_dashboard_data.py` |
| 6.3 | `step6.3-update-scenarios-page.yml` | **Step 6.3: Update Scenarios Page** | Scenario A → B → Compare. Everything needed for the Scenarios comparison page. | `step6_scenario_a.py` → `step6_scenario_b.py` → `step6_scenario_compare.py` |
| 6.4 | `step6.4-procurement-strategies.yml` | **Step 6.4: Update Procurement Page** | 10 strategy variants across 3 families → combined dashboard JS. | `step6_5_strategy{1,2,3}_*.py` → combine JS |
| 6.5 | `step6.5-supplemental-analytics.yml` | **Step 6.5: Supplemental Analytics** | Compressed day profiles + consequential queue (parallel). | `step6_compressed_day.py`, `step6_consequential_queue.py` |
| 7 | `step7-generate-shared-data.yml` | **Step 7: Update Home Page / Shared Data** | Consolidates all results into `shared-data.js`. Updates Home Page and Dashboard. | `step7_generate_shared_data.py` |

## Utilities

| Workflow | Display Name | What It Does |
|----------|-------------|--------------|
| `lmp-calibration.yml` | **LMP Calibration: Validate Model** | Validates synthetic LMP model against calibration targets (weather-normalized or QA against actual data). |

## Common Inputs

All workflows accept these standard inputs:

- **`iso`** — ISO region selector (ALL, CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP). Present on every workflow.
- **`branch`** — Branch to checkout scripts from (default: `master`). Present on every workflow.
- **`thresholds`** — Comma-separated threshold list or "all". Present on Step 1b (big compute).
- **`track`** — Track selector (baseline, newbuild, cost-to-replace). Present on Step 3.
- **`scenario`** — Scenario selector (ALL, scenario-a-only, scenario-b-only). Present on Step 6.3.
- **`strategy`** — Strategy family selector. Present on Step 6.4.

## Typical Usage Patterns

### "I changed cost assumptions and need to refresh everything"
```
Step 3 → Step 4 → Step 5 → Step 6.0 → Step 6.1 + 6.2 + 6.3 (parallel) → Step 7
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

### "Full pipeline from scratch"
```
Step 0 (optional) → Step 1a → Step 1b → Step 2 → Step 3 → Step 4 → Step 5 → Step 6.0 → Steps 6.1-6.5 → Step 7
```

## Legacy Workflows (Removed)

The following workflows were removed in this reorganization:

| Old File | Reason |
|----------|--------|
| `run-step1-full-pfs.yml` | Replaced by Step 1a + 1b split (scored database → PFS mining) |
| `run-step3-tracks.yml` | Duplicate — track functionality consolidated into `step3-cost-optimization.yml` |
| `run-step6-post-processors.yml` | Monolith with 10 script options — replaced by focused page-oriented 6.x workflows |
| `run-step6-generate-shared-data.yml` | Misnamed (said "step6" but ran step7) — absorbed into 6.1 and standalone Step 7 |
| `run-lmp-all-isos.yml` | Standalone LMP compute — absorbed into `step6.2-update-lmp-page.yml` |
| `run-step2-efficient-frontier.yml` | Renamed to `step2-efficient-frontier.yml` |
| `run-step2-5-expand-ef.yml` | Renamed to `step2.5-expand-ef.yml` |
| `run-step3-cost-optimization.yml` | Renamed to `step3-cost-optimization.yml` |
| `run-step4-gas-ccs.yml` | Renamed to `step4-gas-ccs.yml` |
| `run-step5-post-processors.yml` | Renamed to `step5-dispatch-cache.yml` |
| `run-procurement-strategies.yml` | Renamed to `step6.4-procurement-strategies.yml` |
| `fetch-actual-lmp.yml` | Renamed to `step0-fetch-lmp-data.yml` |
| `run-lmp-calibration.yml` | Renamed to `lmp-calibration.yml` |
| `step1-build-pfs.yml` | Renamed to `step1b-build-pfs.yml` |
| `step1-build-scored-database.yml` | Renamed to `step1a-scored-database.yml` |
