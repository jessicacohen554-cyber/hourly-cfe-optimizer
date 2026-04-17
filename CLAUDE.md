# Claude Code — Session Instructions

> Source of truth for project decisions: **SPEC.md**.
> Reference docs: **PIPELINE.md** (8-step pipeline), **DESIGN_SYSTEM.md** (dashboard CSS/HTML standards).
> Repo: `jessicacohen554-cyber/hourly-cfe-optimizer`.

## On Session Start

1. Read `SPEC.md` first — every design decision, cost table, methodology.
2. Read this file.
3. Install Numba: `pip install numba`. Never run optimizer scripts without verifying `python3 -c "from numba import njit; print('OK')"` — numpy fallback is 10–50× slower.
4. Confirm branch matches your assigned task.
5. Review `SPEC.md` `## Current Status` + git log to pick up where the last session left off.

## Reasoning & Thinking

- **Always use max thinking budget.** The harness enforces `MAX_THINKING_TOKENS=64000` via `.claude/settings.json`. Do not downshift reasoning based on perceived task simplicity.
- **No adaptive reasoning.** Use full depth for every non-trivial task. Only trivial lookups (single Read/Grep) may use light thinking.

## Proactive Session Management

**Goal: seamless pickup by the next session, zero lost work.** Trigger when context usage climbs past ~80%, a long-running tool nears timeout, or the user signals session wrap-up.

When triggered:
1. Stop starting new work.
2. Commit + push all WIP to the current branch. Partial progress beats lost progress.
3. Write/update `## Current Status` at the top of `SPEC.md` with: what was accomplished, in-progress state (what's half-done, where to resume), next steps detailed enough to continue without re-reading the conversation, open questions needing user input.
4. Output a **resume prompt** the user can paste into the next session. Focus on task context (current task, key files touched, exact next step, pending user decisions). Do NOT include branch name — master has everything after PRs merge.

**Hook-backed.** The `Stop` hook in `.claude/settings.json` blocks exit and surfaces uncommitted work. If it fires with a dirty tree, commit before proceeding.

## Decision-Making

- **Present multi-way choices as clickable decision cards via `AskUserQuestion`.** Not plain-text option trees. Each card: short question, 2–4 options, 1-sentence descriptions with tradeoffs. For triage, use Keep / Discard / Update as the option set.
- **Trivial/obvious decisions** (formatting, variable names, minor refactors consistent with existing patterns) don't need a card — just do them.
- **Never contradict** directions already in this file or SPEC.md.

## Documentation-First Development

Decisions go to `SPEC.md` **immediately** after user confirmation — before any implementation, code changes, or further discussion. Zero lag. If a batch of decisions lands in rapid succession, write ALL of them to SPEC.md before writing any code. SPEC.md is the single source of truth that survives session interruptions and token limits.

## Execution Efficiency

- **Parallelize independent work.** Multiple agents, Grep/Glob, Read, Edit in one response when there are no dependencies between calls.
- **Targeted reads only.** Use `offset`/`limit` on large files. Never re-read a file unchanged since the last read.
- **Agents return summaries, not raw content.** Structured summaries (architecture, key functions, line numbers) — never paste 2K-line file dumps back.
- **Prefer Grep/Glob over Explore agents** for directed lookups (function/pattern/file name). Explore is 10× more expensive; reserve it for open-ended codebase understanding.
- **Batch related edits** into one response. Don't make 6 sequential edits with narration between each.
- **Don't paste large code blocks** back to the user — reference by `file_path:line_number`.

## Communication Style

- **Don't narrate — just do.** Skip "Let me read the file...", "Now I'll edit..." filler. Execute, report outcome.
- **TodoWrite on a frequent cadence** — the todo list IS the status communication.
- **Don't echo decisions back.** Acknowledge briefly and act. Don't restate what the user said.
- **Be verbose when it matters** — important decisions, tradeoffs, things the user needs to know.
- **Be concise otherwise** — no filler, no restating the obvious. A sentence or two on *why*, not a paragraph.
- **Prefer bullets + numbered lists** over walls of prose.
- **Only surface errors when self-recovery failed.** Skip QA narration unless something failed.

## Writing Voice (Research Paper & Narrative Content)

- **Match the user's voice**: direct, confident, analytical. Senior analyst briefing stakeholders, not PhD thesis. Active voice, contractions OK.
- **Brevity over verbosity.** Every sentence earns its place. If it can be said in 10 words, don't use 30.
- **Detail sufficient for peer review** — brevity ≠ vagueness. Precise about methodology, assumptions, data sources.
- **Lead with the insight**, then the evidence.
- **Audience: business professionals** with minimal energy-sector background. Add tooltips on controls. Chart titles tell the story, not just label axes. Abatement/narrative pages build progressively — "so what" before "how". Research paper stays rigorous but accessible. Methodology HTML = technical specs only; detailed narrative lives in the paper.

## Git & Commits

- **Frequent incremental commits** during active work — protects against session interruptions. Squash into a clean commit before pushing.
- **Squash-style commits** — one descriptive commit per feature/task, not granular per-file.
- **Descriptive paragraph messages** — explain *what* and *why*, not just file lists.
- **Detailed PR descriptions** — summary, what changed, why, decisions made.
- **Push only when complete and QA'd.** Don't push partial/broken work.

## Vectorization-First Code Design

**Never write sequential Python `for` loops over data arrays >1K rows.** A loop over 7M mixes takes 20+ minutes; vectorized takes <1 second.

- Convert data to numpy arrays early, precompute scalar params into flat arrays, apply vectorized ops, unpack only the winner(s) back to dicts.
- Use Numba `@njit(cache=True)` for complex per-element arithmetic (cost functions, dispatch models) with a numpy fallback. Kernel signature: `(N, K)` array + flat params → `(N,)` result.
- Boolean masking (`arr[mask]`) instead of list comprehension.
- `np.argmin(costs)` to find the winner, then scalar version only on the winner for the full result dict.

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

## File Boundaries

- **Never modify raw data files** in `data/` — preserved from source (EIA, eGRID, etc.).
- Transformations create a copy or derived file; never edit the original.
- Freely edit: optimizer code, dashboard HTML/JS, methodology, research paper, build scripts, config files.

## Change Propagation

- **"Fix something" = fix it everywhere.** Any request to fix/update/change applies to ALL regions and ALL pages by default, not just the one being discussed.
- Pages to update on a cross-cutting fix: `dashboard.html`, `abatement_dashboard.html`, `optimizer_methodology.html`, `research_paper.html`, `storage_analysis.html`, `lmp_trends.html`.
- **Always update the research paper** when optimizer results, methodology, or findings change.
- **Proactively update narrative text** after new results are generated — don't wait to be asked.
- Only scope a fix to a single page if the user explicitly says so (e.g., "just on CAISO").

## Priority Ordering (When Tradeoffs Arise)

1. **Data accuracy** — highest priority by default.
2. **Mobile compatibility / visual polish** — equal priority, both matter.
3. **Performance** — optimize only after correctness and presentation are solid.

**Override signal**: if the user says "representative viz" or "create a representative [chart/visualization]", storytelling and visual impact take priority over perfect data accuracy for that specific element.

## Research & Exploration

- **Start with a quick survey** — broad scan first, then dive deeper only where the user asks.
- Don't over-research upfront; present a summary and let the user direct where to go deep.

## Rollback & Data/Analytical Issues

- **Visual/UX issues**: iterate on what's there — revert only if the approach is fundamentally wrong.
- **Data or analytical accuracy issues**: check SPEC.md first for prior decisions. If the issue hasn't been discussed before, **ask the user** before changing anything.
- SPEC.md is the record of analytical decisions — always consult it before making judgment calls on data/methodology.

## Build Process

**After every optimizer run**: save a final cached results data file (`data/optimizer_cache.json`) that future projects can consume as input. Include full co-optimized results for all thresholds × scenarios × ISOs with resource mixes, costs, scores, and metadata.

---

# Project Context

## Architecture Overview

- **Pipeline**: 8-step optimization pipeline (Steps 0–7) — 40+ Python scripts. Full details in `PIPELINE.md`.
- **Dashboard**: 20+ interactive HTML pages in `dashboard/`.
- **Homepage**: `dashboard/index.html` — scrollytelling landing with key findings.
- **Cost Optimizer**: `dashboard/dashboard.html` — interactive optimizer with all sensitivity toggles.
- **Abatement**: `dashboard/abatement_dashboard.html` — CO₂ abatement analysis (scrollytell + static cost envelopes).
- **LMP Analysis**: `dashboard/lmp_trends.html` — synthetic LMP trend analysis.
- **Storage**: `dashboard/storage_analysis.html` — battery/LDES dispatch analysis.
- **Procurement**: `dashboard/procurement_strategies.html` — corporate procurement strategy comparison.
- **Methodology**: `dashboard/optimizer_methodology.html` — technical specs only.
- **Research Paper**: `dashboard/research_paper.html` — full standalone paper with regional deep-dives.
- **Data**: `data/` — EIA hourly profiles, eGRID emission rates, fossil mix data, pipeline outputs.
- **Results**: `dashboard/js/shared-data.js` — pre-computed dashboard data (Step 7 output).

## Site Architecture Intent

- **Research paper** (`research_paper.html`): standalone academic artifact. Intentionally duplicates analysis from other pages — designed to be read independently. Do NOT cut content to avoid duplication with the interactive site.
- **Interactive site** (all other pages): scrollytelling/interactive mode of the same research. Pages reference each other and build a narrative journey. Cut duplicate content BETWEEN these pages (but not between them and the paper).
- **Homepage** (`index.html`): entry point with scrollytell narrative. "What you need depends on what you have and where you're going" framing is intentional and appears on both homepage and dashboard.
- **Dashboard** (`dashboard.html`): interactive optimizer with all parametric toggles. Same framing reinforced.
- **Regional content**: lives in research paper and homepage scrollytell (standalone regional page deleted Feb 2026).

## Key Design Principles

- 2025 snapshot model (no forward projections).
- **7 ISOs**: CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP.
- All sensitivity toggles use Low/Medium/High (never "Base" or "Baseline").
- All new features layered on top of existing — never remove existing visuals or controls.
- **COST DRIVES RESOURCE MIX** — cost and resource mix are co-optimized for every scenario. Different cost assumptions produce different optimal mixes. This is the core scientific contribution. Never decouple cost from mix optimization or treat cost as a secondary overlay.
- **8 toggle groups**: 5 paired (Renewable Gen, Firm Gen, Storage, Fossil Fuel, Transmission) + CCS (L/M/H) + 45Q (On/Off) + Geothermal (CAISO-only, L/M/H).
- **20 thresholds** (10, 20, 30, 40 [coarse only], 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9). Top threshold is ≥99.9% — labeled "effectively 100%" (8.76 unmatched hours/year). Thresholds 10–40 are coarse-grid only (no fine zone search, no storage refinement). 50–99.9 get full pipeline coverage.
- **5,832 cost scenarios per region/threshold** (3×3×3×3×2×3×4 non-CAISO; 17,496 CAISO with geothermal).
- Resource mix optimization at Medium costs; sensitivity toggles recalculate costs on cached physics.
- Hydro is always existing-only, wholesale-priced, $0 transmission.
- CCS-CCGT includes 45Q offset in LCOE, modeled as flat baseload.
- LDES = 100hr iron-air, 50% RT efficiency, 7-day rolling window dispatch.
- Battery = 4hr/8hr Li-ion, 85% RT efficiency, daily-cycle dispatch.
- Green H2 = 1000hr, 35% RTE, 30-day rolling window, ≥95% thresholds only.
- Geothermal = CAISO only, 5th physics dimension, flat year-round, 39 TWh cap.

## Scientific Rigor vs. Compute

**NEVER sacrifice scientific integrity to save compute.** Use as much compute as necessary for academic rigor. The project must withstand peer scrutiny.

When facing rigor-vs-compute tradeoffs:
1. Discuss the tradeoff with the user first — don't unilaterally choose minimal compute.
2. Find the middle ground that balances rigor and feasibility.
3. Pairing variables (e.g., 5 paired toggles vs. 10 individual) is acceptable — reflects real-world cost correlations.
4. 20 thresholds preserves inflection points while covering the full range.
5. Never decouple cost from optimization.
6. Never re-rank cached results as a shortcut when full optimization is needed.

## Naming & Terminology

- ALL toggles: Low / Medium / High (NEVER "Base", "Baseline", or "Mid").
- Transmission toggle also has a "None" option: None / Low / Medium / High.
- Resources: Clean Firm, Solar, Wind, CCS-CCGT, Hydro, Battery, LDES, Solar+Batt4, Solar+Batt8, Wind+Batt4, Wind+Batt8.

## Visual & UX

- Banner goes ABOVE intro text on main page (not below).
- ALL pages share the same header banner styling — only title and tagline vary per page.
- Top navigation bar on ALL pages: Home | Cost Optimizer | Analysis (dropdown) | Research (dropdown). Current page highlighted; mobile gets hamburger.
- Scrollytelling format for the abatement analysis page, matching main dashboard style.
- `index.html` is the landing page; `dashboard.html` is the interactive optimizer.
- **Static pages default to Medium cost sensitivities** — homepage and abatement page use all-Medium toggle data unless a figure is explicitly designed to show L/M/H comparison ranges.
- Layer in explanations for model elements — assume reader has minimal energy-domain knowledge.
- Clean, crisp visual identity — no clutter.
- Both mobile and desktop compatible (44px min tap targets, responsive charts, no horizontal overflow).
- For full CSS/HTML standards see `DESIGN_SYSTEM.md`.

## QA/QC Requirements (Before Any Push)

- Validate optimizer results against published research (NREL ATB, Lazard, LBNL).
- Check HTML formatting, visual consistency, all controls functional across ALL pages.
- Mobile compatibility at 320px, 375px, 768px viewports.
- All text readable in all figures at all sizes.
- No console errors, no broken layouts.
- **Always do a full QA/QC sweep** on functionality, visuals, and narrative before pushing.
- **Proactively update narrative and explanatory text** after new results — don't leave stale numbers or descriptions.
- Verify research paper reflects current results and methodology.

## Figure & Chart Standards

- **Adequate height/spacing on mobile**: min-height 300px mobile, 400px desktop. No compressed/unreadable charts.
- **Threshold label spacing**: don't label every point. Show 75, 90, 95, ≥99.9 on scrollytell figures. Dashboard charts use tooltips for unlabeled points.
- **Data-driven but clean**: scrollytell figures pull real optimizer results but stay illustrative — clean axes, clear legends, min 12px fonts on mobile.
- **Dashboard tooltips**: interactive charts show exact values at all threshold points on hover.
- **Consistent color palette**: use `RESOURCE_COLORS.*` / `ISO_COLORS.*` from `chart-colors.js` and CSS variables from `shared.css`. NEVER hardcode hex values in Chart.js datasets. See `DESIGN_SYSTEM.md` for the canonical tables.
- **Chart.js responsive options**: always `responsive: true`, `maintainAspectRatio: false`, `padding` for readable labels.
- **Mobile tap targets**: 44px minimum.

## Scrollytell Layout & Interaction Patterns

- **Linear layout for all acts**: intro text above → visual (chart/viz + toggle) → post-visual narrative below. No side-by-side grids. Top-to-bottom on all screens.
- **Illusion/Reality toggle on every act**: reuse `.mode-toggle-wrap` switch component. Illusion = simplified/flat data (the common misconception); Reality = actual data with animated transition. Auto-transitions to Reality on first scroll; user can toggle back and forth.
- **Legend markers**: thin line swatches (16×3px), not circle dots. Demand = dashed swatch.

## Animations & Interactivity

- Abatement analysis page: illustrative and dynamic with animations.
- Abatement comparison page: creative animations, animated number counters, scroll-triggered transitions.
- Use CSS animations, scroll-based triggers, Chart.js animation options.
- Keep it professional (Bloomberg/McKinsey quality) but engaging.
