# Claude Code — Session Instructions

> Draft replacement for `CLAUDE.md` — review this file, then move it into place.
> Source of truth for project decisions: **SPEC.md**.

## On every turn — non-negotiables

1. **Use full thinking depth** on every non-trivial edit. Don't downshift because a task "seems mechanical." The harness sets `MAX_THINKING_TOKENS=64000` — use it.
2. **Anchor every new page to a reference filename** ("build it like `abatement_dashboard.html`"). If no reference was named in the task brief, ask before starting.
3. **Anchor every code task to reference files.** Before writing or editing any script, identify the task category from the **code-reference table below** and read the listed reference files in full. Pattern-matching from filename alone produces wrong code.
4. **Plan before writing** any data-touching code or any user-facing prose >100 words. Output the plan, wait for OK.
5. **Vectorize before looping.** Never write a Python `for` loop over data arrays >1k rows. If you're about to, stop and write the vectorized kernel signature first, show it, wait for OK. Reference exemplar: `market-simulator/scripts/fleet_dispatch.py` ("Fully vectorized — no Python for-loops over the 1,215 scenarios").
6. **Run `/fix-prose` before committing** any HTML/MD page you wrote or substantially edited. The `jargon-fixer` and `voice-fixer` agents catch self-referential jargon and AI-tell language you missed.

## Source docs — what to read when

| File | Read when |
|---|---|
| `SPEC.md` | Session start, before picking up work — current status, locked-in spec, open questions |
| `LESSONS.md` | Session start, if non-empty — durable learnings that don't fit as CLAUDE.md rules |
| `OPS.md` | Before running optimizer / heavy compute / data ops |
| `DESIGN_SYSTEM.md` | Before any HTML/CSS work — color tokens, typography, chart conventions |
| `PIPELINE.md` | Reference for the 8-step pipeline only |
| `SPEC_LOG.md` | Historical decision archive — reference, don't restate |

## Reference-page table — anchor every new page

| Building a... | Use as reference |
|---|---|
| Scrollytell analysis page | `dashboard/abatement_dashboard.html` |
| Single-company report | `dashboard/ipp_vistra.html` |
| Multi-company comparison report | `dashboard/ipp-transition-report.html` |
| Market overview | `dashboard/gen_market_overview.html` |
| Interactive optimizer page | `dashboard/dashboard.html` |
| Methodology / spec page | `dashboard/optimizer_methodology.html` |
| Research-paper-style page | `dashboard/research_paper.html` |

## Code-reference table — anchor every code task

When the task involves writing or editing code, identify the category and read the listed files in full **before writing anything**.

| Working on... | Read first |
|---|---|
| Reliability-tax chart payload generators | Generator layer cleared Apr 22, 2026 — pending rewrite in a dedicated presentation session. Read `reliability_tax/charts/data_loader.py` (shared loader, kept), `reliability_tax/methodogy.md`, `reliability_tax/PATHWAY_OUTPUT_SCHEMA.md`, and the raw solver outputs under `data/step2.3-pathway/<ISO>/` before drafting new generators. |
| Reliability-tax pipeline / data ingestion | `reliability_tax/charts/data_loader.py`, `analysis/reliability-tax/README.md`, `data/step2.3-pathway/` (sample run output) |
| Step 2.3 pathway optimizer (`step_2_3_pathway_optimizer.py`) | `scripts/step2_1_efficient_frontier.py` lines 1–100 (band-loading contract — read before touching `_read_ef_table` or `_load_or_build_peakclean`), `PIPELINE.md` §Step 2.3, `scripts/pipeline_config.py` |
| Optimizer pipeline (Steps 0–7, 7-ISO sweeps) | `PIPELINE.md`, `market-simulator/scripts/pipeline_config.py`, `market-simulator/scripts/sweep_params_io.py`, `market-simulator/scripts/run_sweep_1215.py`, `data/step1-pfs/` … `data/step7*` for cached outputs |
| Fleet dispatch / 1,215-scenario sweep | `market-simulator/scripts/fleet_dispatch.py` (canonical vectorized exemplar), `market-simulator/scripts/build_fleet_scenario_data.py`, `market-simulator/scripts/fleet_model.py`, `market-simulator/scripts/scenario_common.py` |
| LMP modeling / wholesale price simulation | `market-simulator/scripts/lmp_engine.py`, `market-simulator/scripts/zonal_lmp.py`, `market-simulator/scripts/dispatch_utils.py`, `market-simulator/scripts/market_simulation.py` |
| EIA / fuel-price / interchange ingestion | `market-simulator/scripts/eia_data_io.py`, `market-simulator/scripts/step0_fetch_interchange.py`, `market-simulator/scripts/step0_parse_aeo_fuel_prices.py`, `market-simulator/scripts/fuel_price_projections.py`, `data/eia-860/`, `data/eia-923/`, `data/eia-930/` |
| Plant-level data (heat rates, retirement, eGRID) | `market-simulator/scripts/generate_plant_heat_rates.py`, `market-simulator/scripts/validate_plant_retirement.py`, `data/egrid_emission_rates.json`, `data/eia-860/` |
| Synthetic profile generation | `market-simulator/scripts/generate_synthetic_profiles.py`, `data/eia_demand_profiles_multiyear.json`, `data/eia_generation_profiles_multiyear.json` |
| Profile modeling / dispatch optimizer (separate codebase) | `profile-modeling/dispatch.py`, `profile-modeling/optimizer.py`, `profile-modeling/supply.py`, `profile-modeling/load_profiles.py`, `profile-modeling/run_analysis.py` |
| Hybrid VRE+storage profiles | `scripts/step7_1h_extract_hybrid_data.py`, `data/hybrid_profiles/*.npz`, `dashboard/js/hybrid-analysis-data.js` |
| Datacenter-load modeling | `scripts/step0_generate_datacenter_load.py`, `data-center-cfe/analysis/coal_wall_analysis.py`, `data-center-cfe/analysis/gap_analysis.py`, `data/datacenter_load_metadata.json`, `data/datacenter_load_profile.csv` |
| Constellation / multi-fleet scenarios | `market-simulator/scripts/generate_constellation_scenarios.py`, `market-simulator/fleet_scenarios/`, `market-simulator/scripts/scenario_common.py` |
| Sensitivity analysis | `market-simulator/scripts/sensitivity_analysis.py`, `market-simulator/scripts/sweep_params_io.py` |
| Smoke / validation tests | `run_ercot_smoke_test.py`, `qa_check.sh`, `market-simulator/scripts/tests/`, `market-simulator/scripts/validate_plant_retirement.py`, `pytest.ini` |
| Dashboard JS / Chart.js wiring | `dashboard/js/chart-colors.js`, `dashboard/js/scroll-observer.js`, `dashboard/js/shared-header.js`, `dashboard/js/shared-footer.js`, plus the matching reference dashboard page from the page table above |
| Dashboard CSS / styling | `dashboard/styles/shared.css`, `dashboard/styles/article.css`, `dashboard/styles/scrollytell.css`, `DESIGN_SYSTEM.md` |
| Building a new chart-payload `gen_*.py` | `reliability_tax/charts/data_loader.py` (canonical loader), the closest sibling `gen_*.py` to your new chart, the JSON schema of an existing payload, and the dashboard page that consumes it |
| Market-simulator desktop app changes | `market-simulator/desktop_app.py`, `market-simulator/backend/main.py`, `market-simulator/backend/models.py`, `market-simulator/USER_MANUAL.md` |

## Code-task rules

In addition to the non-negotiables above, apply these when the task category matches.

- **Grep before designing.** Before scoping N subprocess invocations, writing a new helper, or treating "build X or reuse Y?" as an open question: grep `scripts/`, `market-simulator/scripts/`, `reliability_tax/`, and subproject READMEs for existing utilities and cached pipeline outputs. If a pipeline output already answers the question, reuse it — don't surface it as a design decision.
- **Profile one run before launching N.** A docstring is a claim, not a measurement. Any multi-run sweep gets a one-run `cProfile` gate first. If cold-start vs. hot-loop shape doesn't match the docstring's claim, fix the hot spot before scaling up.
- **Audit `persist=True` sites for loop-containment.** Disk-write-per-iteration inside a solve loop silently wrecks wall-clock and produces no stdout (runs look hung, aren't). Thread `persist=False` through the loop; persist once at the end.
- **Parity probes on flag-gated changes cover both branches.** Flag=OFF reproduces baseline bit-for-bit AND flag=ON completes end-to-end at ≥1 config before the commit lands. A passing OFF-only probe does not validate the ON code path.
- **When sweep results are uniformly wrong-signed across orders of magnitude, audit what the solver consumes before tuning parameters.** Read the upstream module's header for the architectural contract (e.g. "each mix in exactly one band; consumers load bands ≥ target"). A uniformly-wrong answer is a data-flow bug, not a calibration problem.
- **At resume-prompt start, verify named files and imports at current HEAD.** `head -50` any driver script the prompt names; confirm every import resolves before relying on its workflow. File:line anchors on the target file also get a quick verification. Branches drift between sessions.
- **Before rebasing, read master's version of the conflict area first.** If a parallel session landed a coherent pattern with passing tests on the same axis as your change, conform to it — don't reintroduce your own convention on top of theirs. Parallel conventions on the same axis are a schema bifurcation nobody wants to untangle later.
- **For dashboard tasks, verify the anchor page's actual DOM before building.** The brief's description of an anchor's layout pattern may not match what the file does. Read the anchor's HTML/CSS directly.

## Voice — what gets shipped to readers

Direct, confident, analytical. Senior analyst briefing stakeholders, not PhD thesis. Active voice, contractions OK. Brevity over verbosity. Lead with the insight, then the evidence. Detail sufficient for peer review — brevity ≠ vagueness.

**Three voice rules, all enforced by the prose-fixer agents:**

- **No self-referential project shorthand in user copy.** That means no `SPEC §X.Y`, no `Card [A-Z]'`, no bare `§24.X`, no internal endpoint codes (`ep90`, `ep95`, `ep99p9`), no bare pathway codes (`P1`, `P1a`, `P2a`, `P2b`, `P3`), no `NOAK-2035`/`NOAK-2040`/`NOAK-2045` as codenames. Run `/fix-jargon`.
- **Industry acronyms must be defined on first use per page.** ELCC, NOAK, FOAK, LCOE, CCS, LDES, 45Q, 45U, ITC, PTC, ATB, LOLE, CFE, VRE, BESS, CCGT, IPP, ISO, AEO, NREL, LBNL, EIA, NERC. The `jargon-fixer` agent expands them on first occurrence and leaves subsequent uses alone.
- **No AI-tell language.** No "It's worth noting," "Importantly," "Moreover," "Furthermore," "leverages," "unlocks," "delves into," "underscores," "robust framework," "holistic approach," "paradigm." Run `/fix-voice`.

## Decision-making

- Multi-way decisions with real tradeoffs → `AskUserQuestion`. Each card: short question, 2–4 options, 1-sentence descriptions.
- Trivial decisions (formatting, naming consistent with existing patterns) → just do them.
- Decisions land in `SPEC.md` immediately after user confirmation, before any implementation.

## Communication style

- Don't narrate — just do. Skip "Let me read the file..." filler.
- TodoWrite on a frequent cadence — the todo list IS the status communication.
- Don't echo decisions back. Acknowledge briefly and act.
- Be concise. Bullets and numbered lists over prose walls.
- Only surface errors when self-recovery failed.

## Git

- Frequent incremental commits during active work.
- Squash into a clean commit before pushing.
- Descriptive paragraph commit messages — explain *what* and *why*.
- Push only when complete and QA'd.
- Never push to a branch other than the one assigned in the task brief.

## Change propagation

"Fix something" = fix it everywhere. Cross-cutting fixes touch every relevant page. Only scope to a single page if the user explicitly says so.

## Proactive session management

When context climbs past ~80% or the user signals wrap-up:
1. Stop starting new work.
2. Commit + push WIP to the current branch.
3. Update `## Current Status` at the top of `SPEC.md`.
4. If a load-bearing lesson surfaced that doesn't fit as a CLAUDE.md rule, write it to `LESSONS.md`. Otherwise skip — no per-session summaries, no postmortems.
5. Output a resume prompt focused on task context.

The `Stop` hook in `.claude/settings.json` enforces this on exit.

## SPEC.md / SPEC_LOG.md maintenance

When writing a new `## Current Status` block to `SPEC.md`, in the same commit move the oldest Current Status block out of `SPEC.md` and append it to `SPEC_LOG.md`. `SPEC.md` is capped at ~500 lines; `SPEC_LOG.md` grows indefinitely.

## File boundaries

- Never modify raw data in `data/` — preserved from source.
- Never run the prose-fixer agents against `SPEC.md`, `CLAUDE.md`, `PIPELINE.md`, `DESIGN_SYSTEM.md`, `OPS.md`, or `.claude/` — those are internal docs and *supposed* to use project shorthand.

## Heavy compute

See `OPS.md` for the full pre-run gate, checkpoint discipline, and rerun policy. Short version: never start a Step 1 run while decisions are still being discussed; always pre-flight (syntax check, assumption verification, save/load round-trip) before launch.
