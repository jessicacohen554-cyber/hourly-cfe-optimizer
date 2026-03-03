# Claude Code — Session Instructions

## If Resuming This Project

1. **Read SPEC.md first** — it contains every design decision, cost table, and implementation detail
2. **Check the todo list** or review git log to see what's been completed
3. **Repo**: `jessicacohen554-cyber/hourly-cfe-optimizer`

## Workflow Preferences (Apply to EVERY Session)

### Documentation-First Development
- **CRITICAL — Decisions go to SPEC.md IMMEDIATELY**: The *very first action* after any design, methodology, or architectural decision is confirmed by the user is to write it to SPEC.md. Do NOT continue with implementation, code changes, or further discussion until the decision is captured. This is the highest-priority workflow rule — sessions can be disrupted or hit token limits at any time, and SPEC.md is the single source of truth that enables seamless continuity. The lag between a decision being made and it being recorded in SPEC.md must be zero.
- **Always maintain this CLAUDE.md** — update it when new preferences, design decisions, or architecture changes are established
- Before ending any session, ensure both files reflect all decisions made during the session
- If multiple decisions are made in rapid succession (e.g., user approves a batch), pause implementation and write ALL of them to SPEC.md before proceeding with any code

### Parallel Execution
- **Deploy as many agents as possible in parallel** for non-dependent tasks to maximize efficiency
- Run searches, builds, file edits, and validations concurrently whenever they don't depend on each other

### Vectorization-First Code Design (Critical — No Sequential Inner Loops)
- **NEVER write sequential Python `for` loops over large data arrays** (>1K rows). Always use numpy vectorized operations, Numba `@njit` kernels, or pandas vectorized methods instead. A Python for-loop over 7M mixes takes 20+ minutes; the same operation vectorized takes <1 second.
- **Pattern**: Convert data to numpy arrays early, precompute scalar parameters into flat arrays, apply vectorized operations, then unpack only the winner(s) back to Python dicts.
- **Numba preference**: For complex per-element arithmetic (cost functions, dispatch models), use `@njit(cache=True)` with a numpy fallback for environments without Numba. The kernel should take `(N, K)` array + flat params → `(N,)` result.
- **Filter operations**: Use numpy boolean masking (`arr[mask]`) instead of Python list comprehension with conditionals.
- **Argmin/argmax**: Use `np.argmin(costs)` to find the best element, then call the scalar version only on the winner for the full result dict.
- **This rule exists because**: Multiple sessions have produced scripts with `for mix in mixes: compute_cost(mix)` patterns that work fine on test data but time out on production-scale data (7–75M mixes). The fix is always the same: vectorize. Do it right the first time.

### Compute Execution (Critical — Preserve Token Budget)
- **NEVER run full pipeline scripts (Step 1–6) end-to-end locally in the session** unless explicitly directed by the user. Full pipeline runs burn significant token budget on compute that should happen via GitHub Actions.
- **Iterative testing IS allowed**: Running scripts on a **subset of data** (e.g., 1–2 ISOs, single threshold, limited rows) for debugging, validation, and iterative development is permitted and encouraged. Quick targeted tests that complete in under 60 seconds are fine. This includes:
  - Running a script on a single ISO to verify correctness after code changes
  - Loading a subset of data to validate vectorization, filtering, or cost logic
  - Benchmarking performance on a representative slice
  - Syntax checks, import validation, unit tests
- **Full dataset / all-ISO runs**: Create or update GitHub Actions workflows so the user can trigger full execution independently. Do not run all 7 ISOs × 21 thresholds locally unless the user explicitly asks.
- **Session compute should be limited to**: syntax checks, quick verification reads (parquet schema, constants), targeted tests on subsets, and lightweight validation.
- **This rule exists because**: Previous sessions burned significant token budget running multi-minute optimizer scripts locally when the same compute could have been done for free via GitHub Actions. The user's token budget is finite and expensive — but short iterative tests are essential for efficient debugging and should not require a full CI roundtrip.

### Git & Commits
- **3-minute commit cadence** — commit work every 3 minutes during active development to avoid losing work. Don't wait for a feature to be "done" to commit — frequent incremental commits protect against session interruptions and token limits. Squash into a clean commit before pushing.
- **CRITICAL — Commit before session expiration**: Always commit all work-in-progress before the session ends or runs out of context. Uncommitted changes are lost forever when a session expires. If the session is approaching limits, immediately commit and push whatever is done — partial progress is infinitely better than lost progress. This is the #1 cause of wasted work across sessions.
- **Squash-style commits** — one descriptive commit per feature/task, not granular per-file commits
- **Descriptive paragraph messages** — commit messages should explain *what* and *why*, not just list files changed
- **Detailed PR descriptions** — include summary, what changed, why, and any decisions made
- **Push only when a feature is complete** — don't push partial/broken work; finish the task, QA it, then push

### Decision-Making (Structured Approval)
- **Present decisions as structured option trees** — user selects the path forward before implementation
- Format: numbered items (1, 2, 3), lettered options (A, B, C, D), roman numeral sub-options (i, ii, iii) if needed
- Each option must include **pros and cons** so the user can make an informed choice
- User responds with shorthand like `1-A-i` to select their preferred path
- **Never contradict directions** already given in this file or SPEC.md
- Trivial/obvious decisions (formatting, variable names, minor refactors consistent with existing patterns) don't need approval — just do them

### File Boundaries
- **Never modify raw data files** in `data/` — these are preserved from source (EIA, eGRID, etc.)
- If data needs transformation, create a copy or derived file — never edit the original
- Freely edit: optimizer code, dashboard HTML/JS, methodology, research paper, build scripts, config files

### Priority Ordering (When Tradeoffs Arise)
1. **Data accuracy** — always highest priority by default
2. **Mobile compatibility / Visual polish** — equal priority, both matter
3. **Performance** — optimize only after correctness and presentation are solid
- **Override signal**: If user says "representative viz" or "create a representative [chart/visualization]", that means storytelling and visual impact take priority over perfect data accuracy for that specific element
- User can always override this ordering for specific tasks

### Writing Voice (Research Paper & Narrative Content)
- **Match the user's voice** — direct, confident, analytical. Not overly formal or academic-stiff. Think "senior analyst briefing stakeholders" rather than "PhD thesis."
- **Brevity over verbosity** — every sentence must earn its place. Cut filler, hedging, and over-qualification. If it can be said in 10 words, don't use 30.
- **Maintain adequate detail for peer review** — brevity ≠ vagueness. Be precise about methodology, assumptions, and data sources. Just don't be wordy about it.
- **Professional but human** — contractions OK, passive voice only when necessary, active voice preferred. No "it should be noted that" or "it is important to consider."
- **Lead with the insight** — state the finding first, then the supporting evidence. Don't build up to the conclusion.

### Communication Style
- **Don't narrate — just do.** Skip "Let me read the file...", "Now I'll edit...", "Let me check..." filler. Execute the work, report the outcome.
- **Use the TodoWrite checklist on a frequent cadence** — the todo list IS the status communication. Update it in real-time so the user always sees current progress without needing to ask.
- **Don't echo back the user's decisions** — when they confirm something, acknowledge briefly and act. Don't restate what they said.
- **Be verbose when it matters** — emphasize important decisions, tradeoffs, and anything the user needs to know
- **Be concise otherwise** — don't pad responses with filler or restate the obvious
- **Explain reasoning concisely** — a sentence or two on *why*, not a paragraph
- **Prefer bullets with clear headers and numbered lists** — avoid walls of prose
- **Only surface errors when you can't resolve them** — try to self-recover first; if stuck after reasonable attempts, explain what failed and what you tried
- **Skip QA narration unless something fails** — don't describe each passing check. Just "QA passed" or report failures.

### Completion Verification (Critical — Never Claim False Completions)
- **NEVER mark a task as [x] completed in SPEC.md or todo list without verification.** Verification means:
  1. For HTML pages: Grep the file for actual JavaScript initialization (e.g., `new Chart(`) — canvas elements without JS are NOT "done"
  2. For charts: Confirm `new Chart(` calls exist for every `<canvas id=...>` element
  3. For controls/toggles: Confirm `addEventListener` or equivalent wiring exists
  4. For narrative content: Confirm actual text exists, not empty containers or placeholder divs
  5. For data-dependent features: Note explicitly that they need optimizer results — don't mark complete
- **If a page is a wireframe/skeleton**: Say so. "Structure created, awaiting implementation" is honest. "[x] Created page with 4 charts" when the charts are blank canvases is a false claim.
- **Run a verification audit before every commit** that touches SPEC.md status: For each [x] item, the evidence must be in the file (grep for JS, check line counts, verify content exists).
- **Stub pages get their own status**: `[ ] page.html — wireframe only (structure + CSS, no JS/content)` is the correct way to track a page that exists but doesn't work.
- **This rule exists because**: Previous sessions marked pages as "complete" when they were empty shells with canvas placeholders. This wasted user tokens on false confidence and delayed real progress.

### Token Efficiency (Critical — Protect User's Weekly Budget)
- **Targeted file reads only** — always use `offset`/`limit` on large files. Never re-read a file already read in the same session unless it's been modified since the last read.
- **Exploration agents: return summaries, not raw content** — exploration agents should return structured summaries (architecture, key functions, line numbers). Never paste full file contents back. A 2K-line file dump in an agent response is a token waste.
- **Prefer Grep/Glob over Explore agents for directed lookups** — if searching for a specific function, pattern, or file name, use Grep/Glob directly. Explore agents are 10× more expensive and should only be used for broad, open-ended codebase understanding.
- **Batch all related edits into one response** — don't make 6 sequential edits with narration between each. Plan them, execute them all in parallel where possible, report once.
- **Don't repeat large code blocks back to the user** — if the user can see the file, don't paste it into the response. Reference by file:line instead.

### Pipeline Architecture (Critical — Know What You're Changing)

**6-Step Pipeline** — Step 1 expensive (hours), Steps 2–6 cheap (seconds to minutes). Only re-run what changed.

**Step 0: Data Fetch/Prep** (`step0_*.py`, 8 scripts):
- `step0_fetch_eia_master.py`, `step0_fetch_all_data.py`, `step0_fetch_egrid.py`, `step0_fetch_eia_multiyear.py`, `step0_fetch_lmp_2025.py`, `step0_fix_dst_profiles.py`, `step0_fix_utc_profiles.py`, `step0_consolidate_miso_spp.py`

**Step 1: PFS Generator** — Two execution paths:
- **Monolithic**: `scripts/step1_pfs_generator.py` — runs the full PFS generation in one process.
- **Modular (CI/CD)**: `step1a_generate_mixes.py` → `step1b_score_mixes.py` → `step1c_build_pfs.py` → `step1d_storage_refinement.py`. Step 1d fills storage gaps from 1c's coarse grid.
- 4D adaptive grid search (clean_firm, solar, wind, hydro) + procurement sweep + battery dispatch (4hr 85% RTE, 8hr 85% RTE) + LDES dispatch (100hr 50% RTE) + Green H2 (1000hr 35% RTE, ≥95% only). CAISO uses 5D (adds geothermal).
- Output: `data/step1-pfs-parquets/` + `data/step1d-storage-parquets/`. **Only re-run if dispatch logic, generation profiles, or demand curves change.**

**Step 2: Efficient Frontier** (`scripts/step2_efficient_frontier.py` + `step2_5_expand_ef_for_floors.py`):
- Extracts non-dominated mixes from PFS. Reads both step1 and step1d parquets.
- Validates step1d coverage for all active thresholds (50%+) before processing.
- Filters existing gen utilization, procurement minimization, strict dominance removal.
- Optional EF expansion for Scenario A per-resource floor constraints.
- Output: `data/step2-ef-parquets/`. **Only re-run if PFS or filtering criteria change.**

**Step 3: Cost Optimization** (`scripts/step3_cost_optimization.py` + `step3_track_nb_ctr.py`):
- Track 1 baseline: vectorized cross-eval of EF mixes under 5,832 combos (17,496 CAISO). Merit-order tranche pricing for clean firm (uprate → geothermal → cheapest of nuclear/CCS).
- Includes NEISO winter gas pipeline constraint (+$13.13/MWh CCS adder), 45Q correction ($27.5/MWh).
- Track 2 (newbuild) + Track 3 (cost-to-replace): greenfield cost analysis.
- Demand growth sweep (25 years × 3 growth rates) with FOAK→NOAK learning curves (Wright's Law).
- Output: `data/step3-cost-opt-parquets/`. **Run when cost assumptions change. No physics re-run needed.**

**Step 4: Dispatch Cache + Independent Analysis** (run after Step 3, output to `data/step5-post-processing/`):
- `step4_build_dispatch_cache.py` — **Run first.** Pre-computes 8,760-hour dispatch for all unique mixes. Versioned NPZ cache (v2) with per-resource matched/surplus + charge profiles.
- `step4_export_track_results.py` — Exports track parquets (NB + CTR) to `track_results.json`. No cache dependency.
- `step4_analyze_tracks.py` — Track cost envelopes (P10/P50/P90), resource mix differentials. No cache dependency.

**Step 5: Dispatch-Cache-Dependent Analysis** (10 scripts, output to `data/step5-post-processing/`):
- `step5_compute_co2.py` — CO₂ dispatch-stack model. Merit-order retirement (coal → oil → gas). Coal/oil capped at 2025 TWh. **Run before MAC stats.**
- `step5_compute_mac_stats.py` — 6 MAC metrics: average fan (P10/P50/P90), stepwise marginal, monotonic envelope, path-constrained. ANOVA decomposition. Crossover vs DAC/SCC/ETS.
- `step5_compute_lmp_prices.py` — 8,760-hour dispatch; synthetic hourly LMP from merit-order fossil stack. All 7 ISOs. Output: `data/step5-post-processing/lmp/`.
- `step5_compute_optimal_targets.py` — Optimal CFE target per ISO via marginal MAC × DAC crossover (PCHIP spline). 3×3 grid-cost × DAC-scenario matrix. No-regrets resource analysis. Output: `optimal_targets.json` + `dashboard/js/optimal-target-data.js`.
- `step5_compress_day_profiles.py` — 24-hour representative day profiles. Reads from dispatch cache; falls back to live compute on miss.
- `step5_consequential_deployment_queue.py` — Cross-regional deployment path under consequential accounting. Hourly emission accounting via dispatch cache.
- `step5_scenario_a_consequential.py` — Forward-stepping consequential procurement with per-resource floor ratchets. PFS fallback on filter exhaustion.
- `step5_scenario_b_hourly.py` — Hourly matching procurement strategy.
- `step5_scenario_comparison.py` — Consequential vs. hourly matching comparison.
- `step5_analyze_storage.py` — Battery/LDES utilization, dispatch patterns, capacity factor analysis.

**Step 5.5: Corporate Procurement Strategy Simulation** (output to `data/step5-post-processing/`):
- `step5_5_procurement_utils.py` — Shared utilities (SSS allocation, EAC pricing, LMP feedback, PPA premiums, learning curves, 25-year timeline).
- `step5_5_strategy1_consequential.py` — Strategy 1 (A/B/C): cross-regional consequential netting under 3 emission baselines.
- `step5_5_strategy2_hourly.py` — Strategy 2 (A/B/C): hourly matching same-ISO with existing clean credit variants.
- `step5_5_strategy3_annual.py` — Strategy 3 (A/B/C/D): annual matching 2×2 matrix.

**Step 6: Dashboard Data Generation:**
- `step6_generate_shared_data.py` — Extracts all results into `dashboard/js/shared-data.js`. SBTi milestone mapping, DAC trajectory projections, LCOE/transmission tables for client-side repricing. Aggregates Step 4/5 outputs. Runs last.
- `step6_extract_no_regrets.py` — Optimal targets and no-regrets resource investments from crossover analysis.

**Utility modules** (no step prefix):
- `pipeline_config.py` — **Single source of truth** for all shared constants (LCOE tables, fuel adjustments, CCS caps, storage parameters, wholesale prices). All step scripts import from here.
- `dispatch_utils.py` — Dispatch reconstruction, supply profiles, fossil retirement, cache I/O. Imports constants from pipeline_config.
- `scenario_common.py` — Shared Scenario A/B logic: cost tables, demand growth, learning curves, EF/PFS loading. Imports constants from pipeline_config.
- `eia_data_io.py` — Standardized EIA multi-year profile loading.
- `calibrate_lmp_model.py` — LMP model validation against actual ISO data.
- Other: `anthropic_image_utils.py`, `extract_shared_data.py`, `analyze_pjm_lmp.py`, `analyze_results.py`, `sensitivity_analysis.py`

**GitHub Actions** (~20 workflows, all `workflow_dispatch`):
- Core pipeline: `step1a-scored-database.yml` → `step1b-build-pfs.yml` → `step1d-storage-refinement.yml` → `step2-efficient-frontier.yml` → `step3-cost-optimization.yml` → `step4-dispatch-cache.yml`
- Page-oriented: `step5.0-compute-co2.yml`, `step5.1-update-mac-page.yml`, `step5.2-update-lmp-page.yml`, `step5.3-update-scenarios-page.yml`, `step5.4-procurement-strategies.yml`, `step5.5-supplemental-analytics.yml`, `step5.6-update-optimizer-dashboard.yml`
- Final: `step6-generate-shared-data.yml`
- See `.github/workflows/README.md` for full docs and common patterns.

**Data contract**: Step 3 must NOT change existing columns in shared-data.js or overprocure_results.json — only ADD new columns/fields.

**Key principle**: Steps 2–6 are cheap (seconds to minutes). Step 1 is expensive (hours). Default to Steps 3 + post-processing unless physics assumptions change.

### Incremental Results (Critical — Never Rerun What's Already Computed)
- **Default to temp functions for new analysis tracks** — when adding a new analysis dimension (e.g., new-build track, LMP module, CO2 dispatch), write a standalone temp script that computes ONLY the missing results and appends them to the existing output files. Never rerun the full pipeline when only a subset of results is needed.
- **Pattern**: (1) Write temp function to compute delta results, (2) Append/merge into existing output JSON/parquet, (3) Update primary scripts for future iterations (but don't rerun them)
- **Step 3 cost optimization is semi-expensive with large EFs** — 27M mixes × 5,832 scenarios × 7 ISOs × numpy takes hours without Numba. Always preserve existing baseline results and only compute new tracks/dimensions incrementally.
- **CO2 dispatch model**: Only run on mixes NOT already in results. Read existing results, identify gaps, compute only the gap, merge back.
- **This rule exists because**: A full step3 rerun on 27M mixes took 5+ hours when only the new track results (~30% of compute) were actually needed. The existing baseline results were perfectly valid and didn't need recomputation.

### Optimizer Run Discipline (Critical — Token Budget Protection)
- **Step 1 (physics) runs are expensive** — they cost compute time AND user tokens. A stale run that gets thrown away wastes both. Treat every Step 1 run as a high-value operation that must succeed. Steps 2–6 are cheap and can be re-run freely.
- **NEVER start a Step 1 run while decisions are still being discussed.** The optimizer must reflect ALL decisions made up to the point of launch.
- **Pre-run gate**: Before launching Step 1, explicitly verify:
  1. All decisions from the current conversation have been implemented in the optimizer code
  2. All decisions have been captured in SPEC.md (per Documentation-First rule above)
  3. No open questions remain that could change optimizer logic, cost tables, or methodology
  4. The code passes a syntax check (`python -c "import py_compile; py_compile.compile(...)"`)
  5. **Full QA/QC and debug sweep** — verify ALL key assumptions (hydro caps, cost tables, resource constraints, dispatch logic, procurement bounds, storage parameters) match SPEC.md and real-world data. Run a dry-run test: imports, constants, data loading, checkpoint save/load round-trip. Confirm no hardcoded values contradict prior decisions. Present the user with a summary of verified assumptions before starting. **This gate exists because**: a previous run wasted 3+ hours of compute due to incorrect hydro caps that weren't caught before launch.
  6. **Checkpoint system verified** — confirm checkpoint save/load/resume works correctly and interval is set appropriately
- **Once running, the optimizer is the top priority.** Do NOT let it get interrupted, stopped, or deprioritized. It runs in the background — other non-optimizer work can happen concurrently, but nothing should kill the process. If the session is approaching token limits, warn the user that the optimizer is still running and needs to complete.
- **If new decisions are made while the optimizer is running in the background**: Immediately flag to the user that the running optimizer does NOT reflect the new decision, and confirm whether to (a) let it finish anyway (if the decision doesn't affect current run), or (b) stop it and re-run after implementing the change. Never silently let a stale run continue as if it's current.
- **If the user asks to run the optimizer**: Treat it as a trigger to do a final audit — scan the recent conversation for any unimplemented decisions before starting the run. If anything is missing, implement it first, THEN run.
- **Background optimizer + other edits is fine** — but only for edits that don't touch optimizer logic (e.g., HTML, CSS, documentation, dashboard JS). If an edit changes anything the optimizer consumes (cost tables, algorithms, thresholds, resource types, dispatch logic), the optimizer must be re-run after the current run completes.
- **If the optimizer crashes or exits without writing results**: Automatically troubleshoot, debug, and retry. Don't wait idle — check logs/stderr, identify the failure mode (OOM, timeout, runtime error, etc.), apply a fix or workaround, and re-launch. Only escalate to the user if the root cause is ambiguous or requires a design decision.

### Change Propagation (Critical)
- **"Fix something" = fix it everywhere** — any request to fix, update, or change something applies to ALL regions and ALL pages by default, not just the one being discussed
- Pages to update: `dashboard.html`, `abatement_dashboard.html`, `optimizer_methodology.html`, `research_paper.html`, `scenario_comparison.html`, `storage_analysis.html`, `lmp_trends.html`
- **Always update the research paper** (`research_paper.html`) when optimizer results, methodology, or findings change
- **Proactively update narrative text** after new results are generated — don't wait to be asked
- Only scope a fix to a single page if the user explicitly says so (e.g., "just on CAISO")

### Session Start Checklist
1. **Read SPEC.md first** — it contains every design decision, cost table, and implementation detail
2. **Read this file (CLAUDE.md)** — it contains all user preferences and project context
3. **Install dependencies**: `pip install numba` — Numba JIT is required for fast vectorized cost evaluation. Always install at the start of every session (environments don't persist). **NEVER run any optimizer script (step1-3, temp scripts, post-processing) without first verifying Numba is installed** (`python3 -c "from numba import njit; print('OK')"`). Running without Numba falls back to numpy and is 10-50× slower.
4. **Check the todo list** or review git log to see what's been completed
5. **Confirm which branch you're on** — develop on the designated branch for your task

### Session End / Mid-Task Handoff
**Goal: seamless pickup by the next session — zero lost context.**

1. Update SPEC.md with any new decisions or changes made during the session
2. Update CLAUDE.md if new preferences or architectural context was established
3. Commit and push all work to the designated branch
4. Write a `## Current Status` section at the top of SPEC.md with:
   - **What was accomplished** this session
   - **What's in progress** (partially done work, current state, what's left)
   - **Next steps** — detailed enough that a fresh session can pick up immediately without re-reading the full conversation
   - **Open questions** — anything unresolved that needs user input
   - **Checkbox TODO list** for user awareness (e.g., `- [x] Built cost tables`, `- [ ] Wire up LDES dispatch`)
5. The TODO list is for user readability; the prose context around it is what enables the next session to continue seamlessly

## Project

- **Repo**: `jessicacohen554-cyber/hourly-cfe-optimizer`

## Architecture Overview

- **Pipeline**: 6-step optimization pipeline (see Pipeline Architecture above) — 40+ Python scripts
- **Dashboard**: 20+ interactive HTML pages in `dashboard/`
- **Homepage**: `dashboard/index.html` — scrollytelling landing page with key findings
- **Cost Optimizer**: `dashboard/dashboard.html` — interactive optimizer with all sensitivity toggles
- **Abatement**: `dashboard/abatement_dashboard.html` — CO₂ Abatement Analysis (scrollytell + static cost envelopes)
- **Scenarios**: `dashboard/scenario_comparison.html` — Consequential vs hourly matching
- **LMP Analysis**: `dashboard/lmp_trends.html` — Synthetic LMP trend analysis
- **Storage**: `dashboard/storage_analysis.html` — Battery/LDES dispatch analysis
- **Procurement**: `dashboard/procurement_strategies.html` — Corporate procurement strategy comparison
- **Methodology**: `dashboard/optimizer_methodology.html` — technical specs only
- **Research Paper**: `dashboard/research_paper.html` — full standalone paper with regional deep-dives
- **Data**: `data/` — EIA hourly profiles, eGRID emission rates, fossil mix data, pipeline outputs
- **Results**: `dashboard/js/shared-data.js` — pre-computed dashboard data (Step 6 output)

## Site Architecture Intent

- **Research paper** (`research_paper.html`): Standalone academic artifact. Intentionally duplicates analysis from other pages — designed to be read independently as a complete paper. Do NOT cut content to avoid duplication with the interactive site.
- **Interactive site** (all other pages): Scrollytelling/interactive mode of the same research. Pages reference each other and build a narrative journey. Cut duplicate content BETWEEN these pages (but not between them and the paper).
- **Homepage** (`index.html`): Entry point with scrollytell narrative and key conclusions. Emphasis on "what you need depends on what you have and where you're going" — this framing is intentional and should appear on both homepage and dashboard.
- **Dashboard** (`dashboard.html`): Interactive optimizer with all parametric toggles. The "what you need depends on what you have" framing is reinforced here too.
- **Regional content**: Lives in research paper and homepage scrollytell (standalone regional page deleted Feb 2026).

## Key Design Principles

- 2025 snapshot model (no forward projections)
- **7 ISOs**: CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP
- All sensitivity toggles use Low/Medium/High naming (never "Base" or "Baseline")
- All new features layered on top of existing — never remove existing visuals or controls
- **COST DRIVES RESOURCE MIX** — cost and resource mix are co-optimized for every scenario. Different cost assumptions produce different optimal resource mixes. This is the core scientific contribution of the project. Never decouple cost from mix optimization or treat cost as a secondary overlay.
- **8 toggle groups**: 5 paired (Renewable Gen, Firm Gen, Storage, Fossil Fuel, Transmission) + CCS (L/M/H) + 45Q (On/Off) + Geothermal (CAISO-only, L/M/H)
- **21 thresholds** (10, 20, 30, 40 [coarse only], 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, ≥99.99) — 10% steps in coarse low range, 5% in mid range, 2.5% in inflection zone, 0.5%/0.1% in the last mile. Top threshold is ≥99.99% (not 100%) — true 100% hourly matching is physically unreachable. Thresholds 10–40 are coarse-grid only (no fine zone search, no step1d storage). Thresholds 50–≥99.99 are the 17 active thresholds with full pipeline coverage.
- **5,832 cost scenarios per region/threshold** (3×3×3×3×2×3×4 = non-CAISO; 17,496 for CAISO with geothermal toggle)
- Resource mix optimization at Medium costs; sensitivity toggles recalculate costs on cached physics
- Hydro is always existing-only, wholesale-priced, $0 transmission
- CCS-CCGT includes 45Q offset in LCOE, modeled as flat baseload
- LDES = 100hr iron-air, 50% RT efficiency, 7-day rolling window dispatch
- Battery = 4hr/8hr Li-ion, 85% RT efficiency, daily-cycle dispatch
- Green H2 = 1000hr, 35% RTE, 30-day rolling window, ≥95% thresholds only
- Geothermal = CAISO only, 5th physics dimension, flat year-round, 39 TWh cap

## Critical: Scientific Rigor vs. Compute

**NEVER sacrifice scientific integrity to save compute.** Use as much compute as necessary to achieve academically rigorous results. The user expects this project to withstand academic scrutiny.

When facing compute vs. rigor tradeoffs:
1. **Always discuss the tradeoff with the user first** — don't unilaterally choose minimal compute
2. **Find the best middle ground** that balances rigor with feasibility
3. **Pairing variables** (e.g., 5 paired toggles vs. 10 individual) is an acceptable rigor-compute tradeoff because it reflects real-world cost correlations
4. **21 thresholds** preserves inflection points while covering the full range — 10/20/30/40 coarse low range, 5% steps 50–80, 2.5% steps 85–97.5, last-mile 99/99.5/99.9/≥99.99
5. **Never decouple cost from optimization** — the co-optimization of cost + resource mix is the whole point
6. **Never re-rank cached results as a shortcut** when full optimization is needed — if costs change the cost function, the optimization must use that cost function

## User Preferences (Do Not Re-Ask These)

### Naming & Terminology
- ALL toggles: Low / Medium / High (NEVER "Base", "Baseline", or "Mid")
- Transmission toggle also has "None" option: None / Low / Medium / High
- Resources: Clean Firm, Solar, Wind, CCS-CCGT, Hydro, Battery, LDES

### Visual & UX
- Banner goes ABOVE intro text on main page (not below)
- ALL pages share same header banner styling — only title and tagline vary per page
- Top navigation bar on ALL pages: Home | Cost Optimizer | Analysis (dropdown) | Research (dropdown)
- Current page highlighted in nav; mobile gets hamburger/collapsible nav
- Scrollytelling format for abatement analysis page, matching main dashboard style
- **Homepage (index.html)** is the landing page with scrollytell narrative and key conclusions; dashboard.html is the interactive optimizer
- **Static pages default to Medium cost sensitivities** — homepage and abatement page use all-Medium toggle data unless a figure is explicitly designed to show L/M/H comparison ranges
- Layer in explanations for model elements — assume reader has minimal energy domain knowledge
- Clean, crisp visual identity — no clutter
- Both mobile and desktop compatible (44px min tap targets, responsive charts, no horizontal overflow)

### Dashboard CSS/HTML Standards (Critical — No Off-Book Styles)

**All dashboard pages MUST use the centralized design system.** Never write new inline CSS for any component that already has a shared class. This section is the law.

#### Architecture
- **`dashboard/styles/shared.css`** — Single source of truth for ALL visual styles (variables, components, layout, responsive rules). Every page links to this file.
- **`dashboard/js/nav.js`** — Shared navigation bar (auto-injected). Include on every page.
- **`dashboard/js/shared-header.js`** — Injects SVG waveform/heartbeat overlay into `.header` elements. Include on every page.
- **`dashboard/js/chart-colors.js`** — Canonical color constants for Chart.js (`RESOURCE_COLORS`, `ISO_COLORS`, `SEMANTIC_COLORS`). Include on every page with charts.

#### Required `<head>` Includes (Every Page)
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Rajdhani:wght@400;500;600;700&family=Barlow+Semi+Condensed:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="styles/shared.css">
<script src="js/nav.js"></script>
<script src="js/chart-colors.js"></script>
<script src="js/shared-header.js"></script>
```

#### Standard Page Header (Every Page)
```html
<div class="header" id="pageHeader">
    <h1>Page Title Here</h1>
    <div class="subtitle">One-line page description</div>
    <div class="header-accent"></div>
</div>
```
The SVG waveform overlay (energy curves + heartbeat/EKG lines) is auto-injected by `shared-header.js`. Do NOT create custom header gradients or banner styles.

#### Canonical Resource Colors (NEVER Hardcode — Use These)
| Resource | CSS Variable | Hex | Chart.js Constant |
|----------|-------------|-----|-------------------|
| Solar | `--solar` | `#F59E0B` | `RESOURCE_COLORS.solar` |
| Wind | `--wind` | `#22C55E` | `RESOURCE_COLORS.wind` |
| Hydro | `--hydro` | `#0EA5E9` | `RESOURCE_COLORS.hydro` |
| Nuclear | `--nuclear` | `#7C3AED` | `RESOURCE_COLORS.nuclear` |
| CCS-CCGT | `--ccs` | `#0891B2` | `RESOURCE_COLORS.ccs` |
| Clean Firm | `--clean-firm` | `#1E3A5F` | `RESOURCE_COLORS.cleanFirm` |
| Battery | `--battery` | `#8B5CF6` | `RESOURCE_COLORS.battery` |
| LDES | `--ldes` | `#E91E63` | `RESOURCE_COLORS.ldes` |
| Green H₂ | `--green-h2` | `#10B981` | `RESOURCE_COLORS.greenH2` |
| Geothermal | `--geothermal` | `#10B981` | `RESOURCE_COLORS.geothermal` |
| Fossil Gas | `--fossil-gas` | `#6B7280` | `RESOURCE_COLORS.fossilGas` |
| Fossil Coal | `--fossil-coal` | `#374151` | `RESOURCE_COLORS.fossilCoal` |

#### Canonical ISO Colors
| ISO | CSS Variable | Hex | Chart.js Constant |
|-----|-------------|-----|-------------------|
| CAISO | `--iso-caiso` | `#F59E0B` | `ISO_COLORS.CAISO` |
| ERCOT | `--iso-ercot` | `#22C55E` | `ISO_COLORS.ERCOT` |
| PJM | `--iso-pjm` | `#0EA5E9` | `ISO_COLORS.PJM` |
| NYISO | `--iso-nyiso` | `#E91E63` | `ISO_COLORS.NYISO` |
| NEISO | `--iso-neiso` | `#9C27B0` | `ISO_COLORS.NEISO` |
| MISO | `--iso-miso` | `#06B6D4` | `ISO_COLORS.MISO` |
| SPP | `--iso-spp` | `#A855F7` | `ISO_COLORS.SPP` |

Each color has transparent variants: CSS `--iso-caiso-t` (12% opacity) / JS `ISO_COLORS.CAISO_T`.

#### Standard Component Classes (Use Instead of Custom CSS)
| Component | Class | Notes |
|-----------|-------|-------|
| White card | `.card` | White bg, light border, subtle shadow |
| Chart panel | `.chart-panel` | Glass effect, blur backdrop |
| Stat card | `.stat-card` + `.stat-value` + `.stat-label` | Metric display |
| Insight callout | `.insight-box` | Blue left border; variants: `.insight-warn`, `.insight-danger`, `.insight-success` |
| Section container | `.content-section` | 1320px max-width, padded |
| Narrow container | `.content-section-narrow` | 900px max-width |
| Section heading | `.section-title` | Navy, heading font |
| Section subtitle | `.section-subtitle` | Muted, body font |
| Toggle group | `.toggle-btn-group` + `button.active` | L/M/H toggles |
| ISO selector | `.iso-selector` + `.iso-btn.active` | ISO pill buttons |
| Chart container | `.chart-container` | 320px min-height |
| Chart small | `.chart-container-sm` | 240px min-height |
| Chart large | `.chart-container-lg` | 400px min-height |
| 2-column grid | `.grid-2col` | Responsive, collapses to 1col on mobile |
| 3-column grid | `.grid-3col` | Responsive |
| Auto-fit grid | `.grid-auto` | `minmax(280px, 1fr)` |
| Stats grid | `.grid-stats` | `minmax(120px, 1fr)` |
| Data table | `.data-table` | Compact, hover rows |
| Legend | `.legend` + `.legend-item` + `.legend-dot` | Chart legend |
| Headline card | `.headline-card` + `.val` + `.lbl` | Hero stats |
| Story section | `.story-section` | Scrollytell with fade-in |
| Badge | `.story-badge` | Pill tag; variants: `.story-badge-red`, `.story-badge-green` |
| Footer | `.page-footer` | Dark navy footer |
| Bottom accent | `.bottom-banner` | 4px gradient bar |

#### Rules for New Pages or Features
1. **NEVER write inline `<style>` blocks for components that exist in shared.css.** Page-specific styles are ONLY for layouts/elements unique to that page.
2. **NEVER hardcode font-family** — use `var(--font-heading)`, `var(--font-body)`, `var(--font-data)`, `var(--font-mono)`.
3. **NEVER hardcode hex colors for resources or ISOs** — use CSS variables in styles, `RESOURCE_COLORS.*` / `ISO_COLORS.*` in Chart.js.
4. **NEVER create custom header/banner gradients** — use `.header` class and `shared-header.js` for the SVG overlay.
5. **NEVER duplicate footer styles** — use `.page-footer`, `.footer-links`, `.bottom-banner`.
6. **Use spacing variables** — `var(--space-xs)` through `var(--space-3xl)` and `var(--pad-page)`.
7. **Use shadow variables** — `var(--shadow-sm)` through `var(--shadow-xl)`.
8. **Use radius variables** — `var(--radius-sm)` through `var(--radius-pill)`.
9. **Body background** — use `var(--bg-page)` (light gray default) or `var(--bg-page-white)`. Never hardcode.
10. **If a shared component is close but not quite right**, extend it with a modifier class rather than creating a new component. Add the modifier to shared.css if it will be reused.

### Content & Audience
- Dashboard audience: Business professionals, minimal energy sector knowledge
- Tooltips/info icons on controls explaining what each toggle does and why it matters
- Chart titles should tell the story, not just label axes
- Abatement page: Build understanding progressively, lead with "so what" before "how"
- Research paper: Academic rigor, withstand scrutiny, but still accessible to new readers
- Methodology HTML: Technical specs only (detailed narrative lives in PDF paper)

### QA/QC Requirements (Before Any Push)
- Validate optimizer results against published research (NREL ATB, Lazard, LBNL)
- Check HTML formatting, visual consistency, all controls functional across ALL pages
- Mobile compatibility at 320px, 375px, 768px viewports
- All text readable in all figures at all sizes
- No console errors, no broken layouts
- **Always do a full QA/QC sweep** on functionality, visuals, and narrative before pushing
- **Proactively update narrative and explanatory text** after new results are generated — don't leave stale numbers or descriptions
- Verify research paper reflects current results and methodology

### Figure & Chart Standards (QA/QC Sweep Checklist)
- **Adequate height/spacing on mobile**: Charts must not be compressed or unreadable on small screens. Set min-height for chart containers (e.g., 300px mobile, 400px desktop)
- **Threshold label spacing**: Don't label every threshold point. Space labels to avoid crowding — show 75, 90, 95, ≥99.99 (skip intermediate values) on scrollytell figures. Dashboard charts can use tooltips for unlabeled points
- **Data-driven but clean**: Scrollytell figures pull from actual optimizer results but should be illustrative — clean axes, clear legends, readable font sizes (min 12px on mobile)
- **Dashboard tooltips**: Interactive dashboard charts should have hover tooltips showing exact values at all threshold points, so labeled points can be sparse without losing precision
- **Consistent color palette**: Use `RESOURCE_COLORS.*` from `chart-colors.js` and CSS variables from `shared.css`. See "Dashboard CSS/HTML Standards" section above for the canonical color table. NEVER hardcode hex values in Chart.js datasets.
- **Chart.js responsive options**: Always set `responsive: true`, `maintainAspectRatio: false`, and use `padding` options for readable labels
- **Mobile tap targets**: Touch targets on charts/controls must be 44px minimum

### Animations & Interactivity
- Abatement analysis page should be illustrative and dynamic with animations
- Abatement comparison page: creative animations, animated number counters, scroll-triggered transitions
- Use CSS animations, scroll-based triggers, Chart.js animation options
- Keep it professional (Bloomberg/McKinsey quality) but engaging

### CO2 & Abatement Modeling (Decided)
- **CO2 emission rate**: Dynamic — shifts with fossil fuel price toggle using regional fuel-switching elasticity
- **Abatement benchmarks**: Static L/M/H bands (DAC, SAF, BECCS, etc.) as fixed horizontal bands on charts
- **Social cost of carbon references**: EPA $51/ton + Rennert et al. $185/ton + EU ETS $60-100/ton range — all three shown on charts

### Data Persistence (Critical — Never Lose Compute Results)
- **NEVER gitignore compute-intensive outputs** — `data/step1-pfs-parquets/`, `data/step1d-storage-parquets/`, `data/step2-ef-parquets/`, and downstream parquets must be committed to git. Previous loss of 21M PFS solutions was caused by gitignoring cache files.
- **Commit parquet caches immediately after optimizer runs** — the moment Step 1 completes, commit and push before doing anything else. This is higher priority than any code changes.
- **After any Step 1 run**: `git add data/step1-pfs-parquets/ data/step1d-storage-parquets/ && git commit -m "Bank PFS cache" && git push`
- **Checkpoint directories (`data/checkpoints/`, `data/checkpoints_v4/`)** are gitignored and removed from the repo — they're crash-recovery artifacts not used downstream. The main parquet outputs are sacred.

### Build Process
- Deploy agents in parallel for non-dependent tasks (see Workflow Preferences above)
- Push only after feature is complete and QA'd (see Git & Commits above)
- **After every optimizer run**: Always save a final cached results data file (`data/optimizer_cache.json`) that can be read into future projects as input. Include full co-optimized results for all thresholds × scenarios × ISOs with resource mixes, costs, scores, and metadata.

### Research & Exploration
- **Start with a quick survey** — broad scan first, then dive deeper only where the user asks
- Don't over-research upfront; present a summary and let the user direct where to go deep

### Rollback & Data/Analytical Issues
- **Visual/UX issues**: iterate on what's there — revert only if the approach is fundamentally wrong
- **Data or analytical accuracy issues**: check SPEC.md first for prior decisions, then **ask the user** before changing anything if the issue hasn't been discussed before
- SPEC.md is the record of the user's analytical decisions — always consult it before making judgment calls on data/methodology
- Capture all data and analytical decisions in SPEC.md so they persist across sessions

### Working Style
- Use TodoWrite tool actively to track all tasks and give visibility into progress
- Break complex tasks into small, trackable steps
- Ask clarifying questions early rather than making wrong assumptions
- When making changes, read the relevant code first — never propose blind edits
