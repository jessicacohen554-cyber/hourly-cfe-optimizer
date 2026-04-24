# OPS.md — Operational Runbook

> Operational runbook for heavy compute and data ops.
> Read before running the optimizer, launching multi-hour sweeps, or touching pipeline-output caches.
> Session-level workflow rules live in `CLAUDE.md`; analytical decisions live in `SPEC.md`.

## Compute Execution

- **Local script execution is preferred** — GitHub Actions minutes are limited.
- **Heavy compute** (Step 1 full runs, all-ISO sweeps, multi-hour jobs): ask before starting.
- **Everything else** (single-ISO runs, Steps 2–7, post-processing, validation, benchmarking): run freely.
- **Syntax checks + import validation** are the cheapest first step before any run.

## Optimizer Run Discipline (Critical — Token Budget Protection)

- **Step 1 (physics) runs are expensive** — compute time AND user tokens. A stale run thrown away wastes both. Treat every Step 1 run as high-value. Steps 2–7 are cheap and can be re-run freely.
- **NEVER start a Step 1 run while decisions are still being discussed.** The optimizer must reflect ALL decisions made up to the point of launch.
- **Pre-run gate.** Before launching Step 1, explicitly verify:
  1. All decisions from the current conversation are implemented in the optimizer code.
  2. All decisions are captured in SPEC.md.
  3. No open questions remain that could change optimizer logic, cost tables, or methodology.
  4. Code passes a syntax check (`python -c "import py_compile; py_compile.compile(...)"`).
  5. **Full QA/QC and debug sweep** — verify all key assumptions (hydro caps, cost tables, resource constraints, dispatch logic, procurement bounds, storage parameters) match SPEC.md and real-world data. Dry-run test: imports, constants, data loading, checkpoint save/load round-trip. Confirm no hardcoded values contradict prior decisions. Present the user with a summary of verified assumptions before starting. **This gate exists because** a previous run wasted 3+ hours due to incorrect hydro caps caught too late.
  6. Checkpoint system verified — save/load/resume correctness, interval set appropriately.
- **Once running, the optimizer is the top priority.** Do NOT let it get interrupted, stopped, or deprioritized. Runs in background — other non-optimizer work can happen concurrently, but nothing kills the process. If the session approaches token limits, warn the user that the optimizer is still running.
- **If new decisions land while the optimizer is running**: immediately flag that the running optimizer does NOT reflect the new decision. Confirm whether to (a) let it finish anyway (if decision doesn't affect current run) or (b) stop + re-run after implementing. Never silently let a stale run continue.
- **If the user asks to run the optimizer**: treat as a final-audit trigger. Scan recent conversation for unimplemented decisions. If anything is missing, implement first, THEN run.
- **Background optimizer + other edits is fine** — but only for edits that don't touch optimizer logic (HTML, CSS, docs, dashboard JS). Edits to cost tables, algorithms, thresholds, resource types, or dispatch logic require a re-run after the current one completes.
- **If the optimizer crashes or exits without writing results**: auto-troubleshoot and retry. Check logs/stderr, identify failure mode (OOM, timeout, runtime error), apply fix or workaround, re-launch. Only escalate if the root cause is ambiguous or requires a design decision.

## Incremental Results (Never Rerun What's Already Computed)

- **Default to temp functions for new analysis tracks.** When adding a new analysis dimension (new-build track, LMP module, CO2 dispatch), write a standalone temp script that computes ONLY the missing results and appends/merges into existing outputs. Never rerun the full pipeline when only a subset is needed.
- **Pattern**: (1) temp function computes delta results → (2) append/merge into existing output JSON/parquet → (3) update primary scripts for future iterations (but don't rerun them).
- **Step 2.2 is semi-expensive with large EFs** (27M mixes × 5,832 scenarios × 7 ISOs × numpy = hours without Numba). Preserve baseline results, compute only new tracks/dimensions incrementally.
- **CO2 dispatch model**: only run on mixes NOT already in results. Read existing, identify gaps, compute the gap, merge back.

## Completion Verification (Never Claim False Completions)

- **NEVER mark a task `[x]` complete in SPEC.md or the todo list without verification.** Verification means:
  1. HTML pages: grep for actual JavaScript initialization (e.g., `new Chart(`). Canvas elements without JS are NOT done.
  2. Charts: confirm `new Chart(` calls exist for every `<canvas id=...>` element.
  3. Controls/toggles: confirm `addEventListener` or equivalent wiring exists.
  4. Narrative content: confirm actual text exists, not empty containers or placeholder divs.
  5. Data-dependent features: note explicitly that they need optimizer results — don't mark complete.
- **Wireframes/skeletons get their own status.** `[ ] page.html — wireframe only (structure + CSS, no JS/content)` is the correct way to track a page that exists but doesn't work.
- **Run a verification audit before every commit** that touches SPEC.md status.

## Data Persistence (Never Lose Compute Results)

- **NEVER gitignore compute-intensive outputs.** `data/step1-pfs/`, `data/step2.1-ef/`, and downstream parquets must be committed. Previous loss of 21M PFS solutions was caused by gitignoring cache files.
- **Commit parquet caches immediately after optimizer runs.** The moment Step 1 completes: `git add data/step1-pfs/ && git commit -m "Bank PFS cache" && git push`. Higher priority than any code changes.
- **Checkpoint directories** (`data/checkpoints/`, `data/checkpoints_v4/`) are gitignored — crash-recovery artifacts not used downstream. Main parquet outputs are sacred.

## Build Process

**After every optimizer run**: save a final cached results data file (`data/optimizer_cache.json`) that future projects can consume as input. Include full co-optimized results for all thresholds × scenarios × ISOs with resource mixes, costs, scores, and metadata.
