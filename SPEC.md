# Advanced Sensitivity Model — Complete Specification

> **Authoritative reference for all design decisions.** If a future session needs context, read this file first.
> Last updated: 2026-04-18.

## Current Status (Apr 18, 2026)

### Coding-Session Sub-Agent — LANDED (Apr 18, 2026, late evening)

**What landed.** `.claude/agents/coding-session.md` — sub-agent for analysis / code tasks in this project. Enforces the three-phase workflow from `CLAUDE.md`: (1) silent orient (read `CLAUDE.md`, `SPEC.md` current status, `LESSONS.md`, `OPS.md` if heavy compute, plus every file in the code-reference table for the task category, plus the reference page from the reference-page table); (2) structured plan with mandatory sections — restated task, insertion point in the 8-step pipeline (upstream producers / downstream consumers), reference anchors, data flow including existing-cache reuse, methodology decisions flagged for approval, performance plan (vectorized kernel signature, `numba @njit` decision, expected iteration count), reuse & drift (existing utilities to import, duplicates flagged as promotion candidates), validation — then wait for explicit OK; (3) implement with TodoWrite and run the promised validation. Hard rules baked in: never loop over data arrays >1k rows (vectorize first, canonical exemplar `fleet_dispatch.py`), use `numba @njit` only when numpy can't express the kernel, load caches once and slice, grep before writing helpers, never fork a utility to add a flag, honor step boundaries in `PIPELINE.md`. Load-bearing methodology choices (capacity metric, cost basis, time-binning, dispatch ordering, retirement rule, counterfactuals, aggregation level, weather year, fuel-price trajectory) require explicit approval before code is written. Invoke with `subagent_type: "coding-session"`.

**Resume prompt for next session:** *"Coding-session sub-agent landed in `.claude/agents/coding-session.md`. Invoke it for any new analysis / pipeline code via `subagent_type: \"coding-session\"`. The agent enforces a plan-then-approve-then-implement workflow, reads the code-reference table in `CLAUDE.md` to locate exemplars, and blocks on methodology approval before writing code. Next natural work: either (a) return to the reliability-tax project-infra workstream below (CLAUDE.draft migration, OPS.md creation, `/fix-prose` test-drive), or (b) take the coding-session agent for a real spin on a pipeline task."*

### Reliability Tax Page Redesign + Project Infra Refresh — IN PROGRESS (Apr 18, 2026, late evening)

**Task context.** User rejected the prior reliability-tax page as too jargon-heavy and self-referential. Two parallel workstreams now in flight: (1) finish redesigning `dashboard/reliability-tax.html` in plain-language voice, with pathway comparisons on single plots (no toggles); (2) build durable project infrastructure that prevents the same regressions on future pages.

**Workstream 1 — page redesign (LANDED across 3 commits on `claude/redesign-reliability-tax-page-LlqKc`):**
- Hero rewritten in plain language with a 4-tile key-findings grid.
- All 8 sections renamed away from "The Setup / The Hump / The Abandonment / The Tax / The Cost of Waiting" to descriptive titles using the numbered `section-header` + `section-number` pattern from `clean_firm_case.html` and `lmp_trends.html`.
- Every `SPEC §`, `Card [A-Z]'`, `NOAK-2035`, `ELCC`, `LOLE`, `NERC`, internal endpoint code (`ep90`, `ep95`), and bare pathway code (`P1`, `P1a`, `P2a`, `P2b`, `P3`) removed from user-facing copy.
- Pathway labels swapped to readable names ("Wind + solar + storage", "Onshore only", "Reactive pivot (90% wall)", "Reactive pivot (economics)", "Proactive clean firm") in buttons, table cells, tooltips, captions.
- §2 hump chart: pathway toggle removed; all three pathways now overlay as colored lines on the same axes per ISO. Peak-build markers in pathway colors, filtered out of the legend.
- §6 stranding chart: pathway toggle removed; converted from a vintage-year bar chart of one pathway to a horizontal bar chart of all five pathways, sorted descending. Caption identifies worst and least-bad pathway.

**Workstream 2 — project infra (IN PROGRESS, this session):**
- `.claude/agents/jargon-fixer.md` — sub-agent that ships in-place edits to remove self-referential project shorthand (`SPEC §X.Y`, `Card [A-Z]'`, `§24.X`, internal endpoint codes, bare pathway codes, `NOAK-YYYY` codenames) AND defines industry acronyms (ELCC, NOAK, FOAK, LCOE, CCS, LDES, 45Q, 45U, ITC, PTC, ATB, LOLE, CFE, VRE, BESS, CCGT, IPP, ISO, AEO, NREL, LBNL, EIA, NERC) parenthetically on first use per page. **LANDED.**
- `.claude/agents/voice-fixer.md` — sub-agent that ships in-place edits to remove AI-tell language: hedge phrases ("It's worth noting"), filler transitions ("Moreover"), LLM-tell verbs ("leverages," "unlocks," "delves into," "underscores"), business-school abstractions ("robust framework," "holistic approach," "paradigm"). Flags sentence-rhythm tells (uniform length, em-dash overuse, triadic structure overload) for human review. **LANDED.**
- `.claude/commands/fix-jargon.md`, `.claude/commands/fix-voice.md`, `.claude/commands/fix-prose.md` — three slash commands. `/fix-jargon` and `/fix-voice` invoke the matching agent; `/fix-prose` runs both sequentially against the same target. All take a file path as arg, fall back to dirty working tree. **LANDED.**
- `CLAUDE.draft.md` — proposed lean replacement for `CLAUDE.md` (92 lines vs 279). Restructured around six top-of-file non-negotiables, a source-docs table, a reference-page table, a voice-rules section that points at the prose-fixer agents. Methodology / ops / design-system content cut and routed to `SPEC.md`, `OPS.md` (to be created), and `DESIGN_SYSTEM.md`. **DRAFTED, awaiting user approval before replacing.**

**What's still to do (gated on user approval of `CLAUDE.draft.md`):**
1. Approve or revise `CLAUDE.draft.md`.
2. `mv CLAUDE.draft.md CLAUDE.md` once approved.
3. Create `OPS.md` and migrate the optimizer-run-discipline / compute-execution / incremental-results / completion-verification / data-persistence / build-process content out of the current `CLAUDE.md`.
4. Create `LESSONS.md` with this session's key learnings: (a) section names should describe content not metaphors, (b) never cite `SPEC §` in user-facing copy, (c) pathway comparison charts must show all pathways on one plot — toggles are not comparisons, (d) industry acronyms get defined on first use per page rather than banned, (e) prose-fixer agents replace the originally proposed pre-commit jargon hook.
5. Create `SPEC_LOG.md` as the historical-decision archive; cap `SPEC.md` at ~500 lines by moving everything older than the prior status update into the log.
6. Test-drive `/fix-prose` against the current `dashboard/reliability-tax.html` to validate the agents and surface any rule gaps.

**Open questions for the user:**
- Approve the lean `CLAUDE.draft.md`, request revisions, or punt to a later session?
- Should `LESSONS.md` accumulate forever, or rotate (e.g., last 50 lessons in-file, older archived)?
- For the bake-off: user is running Sonnet (mobile constraint, no Opus 4.6 access) vs Opus 4.7 in parallel on the same redesign task to compare output quality. Both prompts already drafted in conversation; user is executing them in separate sessions.

**Resume prompt for next session:** *"Pick up the project-infra workstream on the reliability-tax redesign branch. Three sub-agents and three slash commands are landed (`jargon-fixer`, `voice-fixer`, plus `/fix-jargon`, `/fix-voice`, `/fix-prose`). `CLAUDE.draft.md` is on disk awaiting user approval to replace `CLAUDE.md`. Once approved: (1) move draft into place; (2) create `OPS.md` with optimizer-run / compute / data-persistence content extracted from the current `CLAUDE.md`; (3) seed `LESSONS.md` with this session's five learnings; (4) create `SPEC_LOG.md` archive and cap `SPEC.md`; (5) test-drive `/fix-prose` against `dashboard/reliability-tax.html` to validate. Do NOT replace `CLAUDE.md` until the user explicitly approves the draft."*

### Reliability Tax Page Redesign — IN PROGRESS (Apr 18, 2026, late)

**Task context.** User rejected current `dashboard/reliability-tax.html` as too weird, jargon-heavy, and self-referential. Specific feedback: section names like "The Setup" / "The Hump" / "The Abandonment" / "The Tax" / "The Cost of Waiting" are pretentious; in-body `SPEC §24.5`, `SPEC §24.6`, `SPEC §24.7`, `SPEC §24.8`, `Card F'`, `Card J`, `Card R` references are idiotic for public readers; charts/layout weak. Goal: reframe streamlined, plain-language, in the voice of `clean_firm_case.html` and `lmp_trends.html` (numbered `section-header` + `section-number` badges, descriptive titles, direct analytical prose, no methodology-doc self-reference).

**What's done (committed this session):**
- **Hero (§0)** — rewrote lead paragraph in plain language; added 4-tile key-findings grid ($4–18/MWh tax, 30–40% stranded, 50–70% cheaper with P3, ~1% ERCOT/SPP convergence); replaced SPEC-speak "reliability tax formula" insight with "what this bill is made of" plain-English version.
- **Section 1 (was "The Setup")** — renamed to **"Why a clean-only grid still builds gas"** using `section-header` + `section-number` pattern. Duck-panel titles switched `70% CFE → 70% clean`, `95% CFE → 95% clean` with plain-language subs. Body prose fully rewritten: dropped SPEC §24.5 citation, ELCC-credit jargon, "99.97th percentile of hourly margin-on-demand residual", NERC / PJM 1-day-in-10-years LOLE benchmark. Kept the physical rule (2.6 tail hours/year) in accessible terms.
- **Section 2 (was "The Hump")** — renamed to **"How much new gas gets built"** with `section-header` wrapper + descriptive subtitle.

**What's still to do (unchanged in the file):**
1. **Section 2 insight box** — still has "The hump." lead, §24.6 citation, Card F′ reference, SPP/MISO/PJM/ERCOT run-on methodology sentence. Rewrite as plain-language "where new-gas build peaks and why".
2. **Section 3 ("The Abandonment")** — rename to e.g. **"How much of it gets stranded"**. Strip Card F′ references from subtitle and legend. Update `#abandonmentInsight` JS-rendered copy.
3. **Section 4 ("The Tax")** — rename to e.g. **"The bill on ratepayers"**. Strip §24.7 reference from subtitle. The delta-insight JS is mostly fine (already plain-language) but drop the `§24.8 NOAK-2035` clause in the verdict strings.
4. **Section 5 ("Five pathways, five taxes")** — keep the title, rewrite the green-bordered insight box so it doesn't cite "§24.8 findings to watch for"; list the findings plainly. Strip §24.8 from the `ERCOT P1 ≡ P3` and `PJM P3 saves 66.7%` lines. Pathway-card `PATHWAYS` array descriptions also need cleanup: P1a mentions "(Card R)", P3 mentions "§24.8 NOAK-2035 Wright's-Law curve", etc.
5. **Section 6 (Stranding Sankey)** — rewrite subtitle ("§24.6 peak-year snapshot" → plain), drop Card F′ from the chart-panel meta line and bottom insight box.
6. **Section 7 ("The Cost of Waiting")** — keep the title (it's fine), rewrite closing insight box to drop "Card F′", "§24.8 NOAK-2035 window", and P1a shorthand.
7. **Section 8 ("All 175 runs")** — keep title. Bottom insight box drops Card J and italic-cost ceiling citation.
8. **Footer** — `data-footer-note` has `v2 methodology (SPEC §24.4)`. Strip the SPEC citation.
9. **Chart titles/axes** — pass through every Chart.js `options.plugins.title` / `scales.*.title.text` and simplify: e.g. "Cumulative new gas (MW, 2050)" is fine, but `"Stranded capex ($B, 2025–2050)"` etc. are OK. Mostly fine already; verify after the copy rewrites land.
10. **Chart-title elements** — existing page has no `<div class="chart-title">` labels above each canvas (unlike `clean_firm_case.html`/`lmp_trends.html`). Add one to each `.chart-panel` so users know what the chart shows at a glance without reading the narrative first.

**Style-reference pages used.** `dashboard/clean_firm_case.html` (section-header numbered pattern, narrative-card style, foak-hero layout) and `dashboard/lmp_trends.html` (key-findings-panel grid, Era 1/2/3 title convention, section subtitles).

**Resume prompt:** *"Continue the Reliability Tax page redesign in `dashboard/reliability-tax.html`. Hero (§0), Section 1, and Section 2 opener have been rewritten in plain language (no SPEC §, no Card F′). Still to do: (a) Section 2 insight box, (b) Section 3 'Abandonment' rename + prose + JS insight, (c) Section 4 'The Tax' rename + prose + JS delta-insight verdict strings, (d) Section 5 insight box and PATHWAYS array descriptions, (e) Section 6 Sankey subtitle + bottom insight, (f) Section 7 closing insight, (g) Section 8 bottom insight + footer data-footer-note, (h) add per-chart `<div class="chart-title">` headers above each `<canvas>` matching the clean_firm_case pattern. Reference pages: `clean_firm_case.html` and `lmp_trends.html`. Do NOT reference SPEC.md or any `Card X'` shorthand anywhere in the user-facing copy."*

> Older status blocks moved to `SPEC_LOG.md` (Apr 18, 2026 archive cut). See that file for the historical decision log.


## §15: Corporate Procurement Strategy Simulation

### §15.1 Overview

Extension of the optimizer to model how different GHG accounting policies and procurement strategies affect clean energy deployment, costs, and emissions at varying levels of corporate participation. Builds on existing hourly matching (Track 2 NB), cost-to-replace (Track 3 CTR), and consequential accounting (Scenario A/B) frameworks.

### §15.2 Strategy Taxonomy

**Strategy 1 — Consequential Cross-Regional Netting**
Buyers purchase cheapest $/tCO₂ clean energy anywhere in the US to "net" against location-based carbon emissions. Requires new build or nuclear uprates. No ISO boundary constraint.

| Variant | Emission Baseline | Description |
|---------|------------------|-------------|
| **1A** | Grid-average | Buyer's ISO grid-average emission rate (includes clean in denominator). Lowest bar. |
| **1B** | Fossil-average | Buyer's ISO fossil-only fleet average. Higher bar. |
| **1C** | Marginal emissions | Short-run marginal emission rate. Highest bar in coal-heavy ISOs (MISO +17%, SPP +22% vs fossil avg). Negligible difference in gas-dominated ISOs. |

**Strategy 2 — Hourly Matching (Same-ISO)**
Buyer matches load hour-by-hour within their own ISO. No cross-regional procurement. Highest verifiability, highest cost.

| Variant | Existing Clean Credit | Description |
|---------|----------------------|-------------|
| **2A** | None | 100% new build. Maximum additionality. Equivalent to existing Track 2 NB analysis. |
| **2B** | Grid baseline | Buyer takes credit for existing clean grid mix as hourly baseline, procures new build on top. Reduces cost in clean-grid ISOs. |
| **2C** | Pro-rata allocation + premium | Pro-rata share of RPS/nuclear/public utility clean allocated. Premium for existing clean to keep it online (cost-to-replace). New build on top. |

**Strategy 3 — Annual Matching**
Volumetric annual matching without hourly temporal constraint. 2×2 matrix: {Same-ISO, Cross-Regional} × {Additionality Required, No Additionality}.

| Variant | Boundary | Additionality | Description |
|---------|----------|---------------|-------------|
| **3A** | Same-ISO | New build required | Annual matching within buyer's ISO. Only new-build clean energy counts. Comparable to Strategy 2A but annual. |
| **3B** | Cross-regional | New build required | Annual matching from any US ISO. Only new-build clean counts. Comparable to Strategy 1 but annual volumetric rather than consequential netting. |
| **3C** | Same-ISO | No additionality | Annual matching within buyer's ISO. Existing clean counts (includes unbundled RECs from existing generators). |
| **3D** | Cross-regional | No additionality | Annual matching from any US ISO. Existing clean counts. Cheapest option — unbundled RECs from anywhere. This is the "status quo" for most corporate procurement today. |

**Cross-cutting toggle: FOAK-to-NOAK Learning Curves (On/Off)**
Toggle (not suffix) applicable to all strategies simultaneously. When On, each strategy's cost curve shifts based on cumulative clean firm deployment along its mapped trajectory (see §15.10). When Off, static Medium costs from existing optimizer.

### §15.10 Learning Curve Integration (Decided Feb 27)

**Toggle:** "Learning curve: On/Off" in interactive section. Applies to all strategies simultaneously.

**Strategy → Trajectory Mapping:**

| Strategy | Trajectory | Rationale |
|----------|------------|-----------|
| **Strategy 1** (Consequential) | **Scenario A** (delayed) | Chases cheap $/tCO₂ with VRE → no clean firm investment → FOAK when firm is finally needed. Learning period 2035-2047, never fully reaches NOAK. |
| **Strategy 2** (Hourly) | **Scenario B** (accelerated) | Hourly matching *forces* early clean firm + storage investment → accelerates Wright's Law learning → NOAK by 2040. Learning period 2030-2040. |
| **Strategy 3** (Annual) | **Scenario A** (delayed) | Annual flexibility lets buyers avoid firm clean (VRE + unbundled RECs satisfy annual targets) → same delayed investment dynamic as consequential. |

**SBTi Milestone Mapping:** (existing constants from `step6_generate_shared_data.py`)
- 2025: Today (0%) | 2030: SBTi 50% | 2035: SBTi ~70% | 2040: SBTi 90% | 2045: SBTi ~95% | 2050: Net-Zero (≥99.9%)

**Core argument:** Hourly matching incentivizes earlier corporate investment in clean firm, accelerating the learning curve, making the entire system cheaper on a net-zero trajectory. It is significantly more expensive to reach net zero by 2050 if you delay investment in firm clean. The three compounding adverse effects of delay are documented in §15.11.

**Implementation:** Uses existing `learning_fraction()` from `step6_scenario_comparison.py` (Scenario A: FOAK until 2035, learning 2035-2047; Scenario B: FOAK until 2030, learning 2030-2040, NOAK by 2040). Cost at each SBTi milestone = FOAK × (1 - learning_fraction) + NOAK × learning_fraction for clean firm resources.

### §15.11 Adverse Effects of Delayed Hourly Matching (Decided Feb 27)

Three compounding effects when strategies don't require hourly deliverable matching:

**1. Learning Curve Delay (§15.10)**
Consequential/annual strategies defer firm clean investment. When 90%+ targets require firm clean (2040 SBTi milestone), buyers using Strategy 1/3 face near-FOAK prices. Strategy 2 buyers have already driven costs to NOAK via early deployment.

**2. Stranded VRE Overbuild**
Cheap $/tCO₂ logic under annual/consequential accounting drives massive solar/wind procurement at low thresholds (50-70%). But at high thresholds (90%+), additional VRE has sharply diminishing returns — surplus solar during peak hours is already being curtailed. The VRE built at 60% to satisfy annual accounting doesn't deliver physical electrons during nighttime/low-wind hours when the grid actually needs them. This capacity may not be useful by the time you need a deeply decarbonized grid. Hourly matching forces buyers to confront the residual gap early → invests in resources that actually close it.

**3. Gas Lock-in from Missing Storage Signal**
Without hourly matching, there is no price signal to invest in storage (battery + LDES) for nighttime/low-wind hours. Gas fills that gap by default. Once gas capacity is built or retained, it creates political and economic inertia (stranded asset risk, workforce dependencies, pipeline contracts) to keep running it. Hourly matching creates direct demand for storage to cover every hour → displaces gas earlier → shorter gas plant lifetimes → less stranded fossil infrastructure. The longer gas is held, the more expensive the eventual retirement (stranded asset write-downs, decommissioning, workforce transition).

**Compounding on SBTi Timeline:**

| SBTi Milestone | Strategy 1/3 (Annual/Consequential) | Strategy 2 (Hourly) |
|---|---|---|
| 2030 (50%) | Cheap — lots of VRE, looks great on paper | Slightly more expensive — investing in firm + storage |
| 2035 (70%) | Still cheap — more VRE, gas fills gaps | Firm clean hitting learning curve, storage displacing gas |
| 2040 (90%) | **Wall** — VRE saturated, firm at FOAK, gas locked in | Firm at NOAK, storage mature, gas already retiring |
| 2050 (≥99.9%) | Scramble — paying FOAK for firm, retiring gas at huge cost, stranded VRE | Smooth glide — infrastructure already in place |

These effects should be modeled explicitly in the dashboard and presented as a key finding in the scrollytell narrative and research paper.

### §15.11b CO2 Reduction Framing (Decided Mar 8, 2026)

**Critical reframing:** The procurement dashboard primary axis is **buyer CO2 reduction %** (not CFE target %). Each strategy defines "reduction" via different emission accounting:

**Strategy 1A (Consequential, grid-average baseline):**
- Baseline emissions = participating corporate load × grid-average emission rate
- Reduction = displacing fossil CO2 with cheapest $/mtCO2 clean energy purchases from anywhere
- Queue must follow physical buildout order: within each ISO, steps are monotonic (can't access 75→80% in MISO until 31→75% is done). Sort globally by cheapest $/mtCO2, then by step. Exclude missing data intervals.

**Strategy 1B (Consequential, fossil-average baseline):**
- Baseline emissions = participating corporate load × fossil-average emission rate
- Same queue-based reduction as 1A but higher baseline (fossil fleet only, no clean in denominator)

**Strategy 2A/2B/2C (Hourly matching):**
- Emissions = Σ(hourly MWh − hourly match) × fossil-average emission rate
- Reduction comes from MWh matched hourly. Unmatched hours contribute to residual emissions.

**Strategy 3A/3B (Annual matching):**
- Emissions = (total MWh load − total MWh procured) × grid-average emission rate
- Reduction comes from MWh matched annually.

**Key insight:** The same "50% CO2 reduction" means different things under different strategies because they use different baselines and temporal granularity. This is the core comparative insight of the procurement analysis.

### §15.12 Supply Constraint Handling (Decided Feb 27)

**Approach:** Show constraints explicitly. When a strategy hits a physical supply ceiling in an ISO, display it as "infeasible above X% participation" with a red/hatched band on the chart.

**Constraint sources:**
- EAC scarcity (§15.7): NEISO has only ~3-8 TWh available for voluntary procurement. Same-ISO strategies (2, 3A, 3C) hit hard walls.
- Cross-regional strategies (1, 3B, 3D) can route around ISO-level constraints by procuring from surplus ISOs (ERCOT, SPP).
- Resource adequacy: High participation rates under hourly matching may exceed buildable capacity in constrained ISOs.

**Key finding (to highlight):** The existence of supply constraints is itself a major result — it demonstrates *why* cross-regional accounting matters and where same-ISO hourly matching faces physical limits. This should be a prominent element of the scrollytell narrative.

### §15.13 Participation Slider Defaults (Decided Feb 27)

- **Hyperscaler participation:** Default 5-6% of C&I load (current market share). Range 0-15%.
- **Other corporate participation:** Default 7-8% of C&I load (current mid-market). Range 0-40%.
- **Data center electricity:** ~130-150 TWh (2024), growing ~15-20%/yr. Data center share of C&I: ~5-6% today, projected ~8-10% by 2028.
- **Total current corporate procurement:** ~315 TWh (~13% of C&I), with 84% from tech/hyperscaler buyers.

### §15.14 Step 6.5 Procurement Strategy Compute Architecture (Decided Feb 28)

**Purpose:** Compute pipeline to model all 10 procurement strategy variants at varying participation levels, producing data for the procurement comparison dashboard page (§15.5).

#### §15.14.1 Script Architecture (Card 1 — Selected: 1A)

One script per strategy family + shared utility module:
- `scripts/step5_5_strategy1_consequential.py` — Strategies 1A, 1B, 1C
- `scripts/step5_5_strategy2_hourly.py` — Strategies 2A, 2B, 2C
- `scripts/step5_5_strategy3_annual.py` — Strategies 3A, 3B, 3C, 3D
- `scripts/step5_5_procurement_utils.py` — Shared utilities (SSS allocation, EAC pricing, LMP feedback, participation scaling)

Each script is independently runnable. Shared utils handle cross-cutting logic.

#### §15.14.2 Existing Clean Pricing — Dual Toggle (Card 2 — Decided)

Two independent premium mechanisms, each with its own dashboard toggle:

**Toggle (i): 45U-Based Clean Premium**
- Applies to **existing nuclear** in ISOs with 45U PTC eligibility
- Price = 45U credit value ($15/MWh, inflation-adjusted) + small margin (5%)
- Rationale: 45U provides a known revenue floor for existing nuclear. The premium represents what a buyer pays to "claim" the clean attribute of existing nuclear generation beyond what 45U already covers.
- L/M/H sensitivity on the margin above 45U

**Toggle (ii): CTR (Cost-to-Replace) NOAK-Based Premium**
- Uses existing Track 3 CTR values directly from `track_results.json`
- Premium = delta between CTR effective cost and ECF effective cost at each threshold
- Represents what it would cost to replace existing dispatchable clean if it retired
- NOAK-adjusted: applies learning curve discount to replacement cost (reflects that under Strategy 2C with sufficient participation, replacement would be at NOAK, not FOAK)
- L/M/H maps to FOAK/Mid-Learning/NOAK replacement cost

Both toggles can be on simultaneously (additive). Default: Toggle (i) On, Toggle (ii) On at Medium.

#### §15.14.3 Existing Clean Baseline Allocation (Card 3 — Decided Feb 28)

**Strategy 2B — Grid Baseline (Simple):**
- Buyer takes credit for **existing clean grid at 8760 hourly shape** as baseline
- Like Track 1 baseline — existing clean generation follows its actual hourly profile (nuclear flat 24/7, solar daytime, wind stochastic, hydro seasonal)
- Buyer procures **above** this baseline to reach target CFE threshold
- No premium paid — free-rides on existing clean being online
- Cheapest hourly variant but creates stranding risk (§15.5.2)

**Strategy 2C — SSS Allocation + Premium Tranches:**
- **Layer 1: SSS allocation** — Buyer receives pro-rata share of SSS (state-sponsored/contracted) clean within their ISO, following **8760 shape of SSS generation** (nuclear baseload, solar daytime, hydro seasonal — shaped by actual SSS resource mix per ISO)
- **Layer 2: Existing clean beyond SSS** — Remaining existing clean (total existing minus SSS) follows its **8760 hourly shape**. Buyer can procure from this pool at premium prices (see §15.14.4 below)
- **Layer 3: New-build** — Any remaining gap above existing is filled with new-build procurement at LCOE (or LCOE + PPA premium — see §15.14.4)
- Buyer procures **above SSS** allocation to reach target, drawing from Layer 2 (existing non-SSS at premium) then Layer 3 (new-build)

**Key distinction:** 2B gets existing clean for free (no payment signal). 2C pays for existing clean via explicit premiums, creating a revenue signal to prevent retirement.

#### §15.14.4 Procurement Cost Tranches for Strategy 2C (Decided Feb 28)

Tranche merit-order for 2C procurement above SSS:

| Tranche | Source | Price | Status |
|---------|--------|-------|--------|
| 1 | Existing nuclear (non-SSS) | 45U + 5% margin, or CTR delta | **Decided** (§15.14.2) |
| 2 | Nuclear uprates | Uprate LCOE × (1 + premium_pct) | **Decided** (see PPA pricing below) |
| 3 | Existing hydro/solar/wind (non-SSS) | EAC market proxy ($3-5/MWh) | **Decided** (from §Decision 5e) |
| 4 | New-build VRE (solar, wind) | LCOE × (1 + VRE_PPA_premium) | **Decided** (see PPA pricing below) |
| 5 | New-build clean firm (nuclear, CCS, geothermal) | LCOE × (1 + Firm_PPA_premium) | **Decided** (see PPA pricing below) |

**PPA Premium Model (Decided Feb 28):**

PPA prices are set by developer financial models (capital recovery + equity return + risk). The PPA-to-LCOE gap reflects the difference between LCOE's assumed WACC (6-8% real) and actual project financing costs (10-12% nominal equity + 5-7% debt), plus transaction costs. Empirically (LBNL PPA tracking), wind/solar PPAs run 10-25% above NREL ATB LCOE. The percentage model is used because developer returns scale with capital deployed.

`PPA_price = LCOE × (1 + premium_pct)`

| Resource Category | L Premium | M Premium | H Premium | Rationale |
|-------------------|----------|-----------|-----------|-----------|
| **VRE (solar, wind)** | +5% | +12% | +22% | Commodity market, many competing developers, lower risk |
| **Clean firm (nuclear, CCS, geothermal)** | +12% | +22% | +38% | Fewer projects, higher development risk, longer timelines |
| **Nuclear uprates** | +10% | +20% | +35% | Limited supply (~4.4 GW nationally), bilateral negotiation |

**L/M/H mapping**: Low = competitive market, ample supply, multiple bidders. Medium = balanced market, typical bilateral dynamics. High = constrained supply, limited developers, high demand for EACs.

**Example at Medium costs, PJM:**
- Solar LCOE $32 → PPA $36/MWh (+$4)
- Wind LCOE $38 → PPA $43/MWh (+$5)
- Uprate LCOE $25 → PPA $30/MWh (+$5)
- Nuclear new-build LCOE $105 → PPA $128/MWh (+$23)
- CCS LCOE $79 (45Q on) → PPA $96/MWh (+$17)

#### §15.14.5 Participation-to-CFE Target Mapping (Card 4 — Selected: 4B)

Independent annual demand-share model + hourly translation + scarcity feedback:

1. **Annual demand share:** At X% participation, buyer's annual demand = X% × ISO C&I load (TWh)
2. **Hourly translation:** Apply 8760-hour load shape to get hourly demand profile. Each hour, buyer needs (hourly_demand × CFE_target%) matched by clean generation.
3. **EAC scarcity model:** As cumulative demand for EACs increases (more participation), available supply tightens → price escalation. Uses existing `step5_compute_eac_scarcity.py` supply curves.

No "market clearing" between strategies — each strategy computed independently at each participation level.

#### §15.14.6 LMP Wholesale Price Feedback (Card 5 — Selected: 5C)

Full 8760-hour LMP model for **all 7 ISOs**. All 7 ISOs now have calibrated price model classes.

**Implementation status (Feb 28):**
1. PJM — fully calibrated (v10, PJMPriceModel), target $34.7/MWh ✓
2. ERCOT — ERCOTPriceModel, ORDC exponential knee (VOLL=$5K, knee=3GW, cap=$500), target $26/MWh ✓
3. CAISO — CAISOPriceModel, RA + duck curve -$60 floor, target $38/MWh ✓
4. NYISO — NYISOPriceModel, ICAP + tight geography, target $42/MWh ✓
5. NEISO — NEISOPriceModel, FCM + winter gas $13.13/MWh, target $39.5/MWh ✓
6. MISO — MISOPriceModel, PRA, $3,500 VOLL, coal 35%, target $31/MWh ✓
7. SPP — SPPPriceModel, limited capacity market, wind 37%, target $26/MWh ✓

**Calibration data:** `calibrate_lmp_model.py` has `ISO_CALIBRATION_TARGETS` with 2024 SOM data for all ISOs. Sources: PJM IMM, Potomac Economics (MISO/NYISO), SPP MMU, ISO-NE EMM, CAISO DMM, ERCOT/Modo Energy.

**Actual LMP fetching:** `step0_fetch_lmp_2025.py` extended to support `--year 2024` and all 7 ISOs (MISO + SPP added via gridstatus). GitHub Actions workflow: `fetch-actual-lmp.yml`.

**Wholesale price degradation analysis (Decided Feb 28):**
- Run LMP model for all 7 ISOs at all 20 thresholds to produce price degradation curves
- Key output: avg LMP vs clean energy threshold (50%→99.9%) showing merit-order price depression
- Correlation with clean penetration demonstrates cannibalization effect
- Data feeds into both the wholesale price dashboard page AND the procurement strategy comparison
- Price degradation directly affects procurement cost: as more buyers adopt hourly matching, wholesale prices fall, making EACs relatively more expensive (the "stranding" effect documented in §15.11)

Each ISO requires calibration against actual LMP data. ISO-specific price formation rules from §Decision 6 (LMP Module) apply.

#### §15.14.7 SBTi Timeline + 25-Year Demand Growth (Card 6 — Selected: 6D, Extended Feb 28)

Default to SBTi milestone mapping (2030→50%, 2035→70%, 2040→90%, 2050→≥99.9%) with manual override slider for custom targets.

Uses existing constants from `step6_generate_shared_data.py` SBTI_MILESTONES.

**25-Year Demand Growth Dimension (Decided Feb 28):**
The procurement strategy page is built on the **25-year demand growth trajectory / SBTi timeline** already computed by the optimizer:
- Existing demand growth sweep: 25 years × 3 growth rates (L/M/H) per ISO (from Step 3)
- Each year maps to an SBTi milestone CFE target → determines how much clean procurement is needed
- Procurement cost trajectory: for each strategy, compute annual cost over 25 years as the CFE target ratchets up along the SBTi curve
- Strategy comparison becomes: "what is the total cost of getting from today's procurement to 99.9% by 2050, under each strategy?"
- Demand growth interacts with EAC scarcity (higher demand = more competition for same EAC supply = higher prices)
- Learning curve effects compound over the timeline (early adoption at FOAK → late adoption at NOAK)
- Key visualization: cumulative cost envelope (25 years × 10 strategies × L/M/H demand growth) showing when each strategy becomes optimal

#### §15.14.8 Output Format (Card 7 — Selected: 7B)

Standalone JS data file: `dashboard/js/procurement-strategy-data.js`
- Loaded only by `procurement_comparison.html`
- Contains all strategy comparison data (costs, resource mixes, CO₂, MAC, participation curves)
- Generated by a new step in the pipeline (after Step 6, before Step 7)
- Does NOT bloat shared-data.js

### §15.15 Scenario B Redesign: Four-Pool GHG Protocol Hourly Matching Model (Decided Mar 7)

**Problem**: Original Scenario B modeled a single corporate buyer's procurement optimization (north-star endpoint + S-curve pacing). Didn't capture the systemic incentive structure created by GHG Protocol hourly Scope 2 accounting.

**New approach**: Model what the grid buildout looks like when hourly matching + hourly Scope 2 accounting with Standard Supply Service (SSS) is the dominant incentive framework, building toward >95% clean.

**Four supply pools** (SSS ≠ grid existing — SSS is specifically publicly owned, rate-based, or policy-supported):

| Pool | Description | Pricing | Examples |
|------|-------------|---------|----------|
| **1: SSS** | Policy-supported clean | $0 (embedded in rates) | Nuclear ZECs (Dresden, Byron, Nine Mile Point, Millstone), public hydro (NYPA, BPA), RPS-mandated new-build (40% of RPS additions) |
| **2: Contracted** | Locked via corporate PPAs | N/A (unavailable) | Susquehanna → Amazon (PJM, 18.8 TWh), Clinton + Duane Arnold → Meta (MISO, 18 TWh) |
| **3: Existing Merchant** | Available for voluntary EACs | $3-5/MWh EAC premium | Merchant nuclear (LaSalle, Peach Bottom, South Texas), existing solar/wind on grid (massive in ERCOT: ~183 TWh) |
| **4: New-Build** | Investment signal from hourly gap | LCOE + PPA premium | Solar, wind, nuclear, CCS, battery, LDES — learning curves 2030-2040 |

**Key nuances** (per user feedback):
- Not all nuclear is SSS. LaSalle is merchant. Dresden is SSS (IL ZEC). Distinction matters.
- Contracted nuclear (Pool 2) must be subtracted from both SSS and merchant pools.
- ERCOT hourly matching is NOT all incremental — massive existing merchant solar/wind covers daytime hours.
- Meta contracted Duane Arnold (MISO) + Ohio nuclear in addition to Clinton.

**Algorithm**: For each ISO × threshold:
1. Compute SSS (Pool 1) TWh + 8760 hourly shape at SBTi year
2. Subtract contracted (Pool 2) — unavailable
3. Compute merchant clean (Pool 3) = total existing - SSS - contracted
4. Compute hourly coverage from Pools 1+3
5. Compute residual hourly gap = target% × demand - available (per hour)
6. Evaluate EF mixes: feasibility from cached physics, pool-aware cost metric
7. Apply learning curves (2030-2040) to Pool 4 costs
8. Select cheapest by pool-aware incremental cost, floor ratchet

**Design decisions (confirmed)**:
- Mix scoring: **Hybrid** — cached physics for feasibility, pool-aware cost for economics
- Demand scope: **Full grid demand** — systemic endpoint, not participation-weighted
- Learning curve: **2030-2040** — unchanged from original Scenario B

**Zone-based comparison framework** (A vs B vs Annual):
- **Early (50-65%)**: Strategies cluster — cheap VRE available everywhere
- **Inflection (80-90%)**: Diverge sharply — hourly matching forces firm/storage investment
- **Last mile (95-≥99.9%)**: Maximum divergence — learning curve payoff vs FOAK cliff

**Output enrichment**: Per-threshold results include pool1/2/3/4 TWh breakdown, hourly gap characterization (firm-demand hours, storage opportunity), pool-aware cost metrics, zone classification.

**Files**: `step7b_scenario_hourly.py` (major rewrite), `procurement_utils.py` (CONTRACTED_CLEAN_TWH, get_merchant_clean_twh), `scenario_common.py` (SCENARIO_B metadata), `step7c_scenario_comparison.py` (zone comparison + pool metadata).

---

### §15.3 Participation Model (Decided Feb 27)

**Two national sliders:**
1. **Hyperscaler participation** (% of C&I load from top ~30 hyperscaler/tech buyers)
2. **All other corporate participation** (% of remaining C&I load from mid-market, Fortune 500 non-tech, etc.)

Both are national-level sliders (not per-ISO). Total corporate participation = hyperscaler_share × hyperscaler_pct + other_share × other_pct, applied uniformly across ISOs.

**Rationale:** Market is increasingly bifurcated (BNEF 2025: tech = 84% of deal activity, only 33 unique buyers). Modeling the two cohorts separately captures the structural difference between hyperscaler procurement capacity and mid-market adoption.

### §15.4 Target/Outcome Model (Decided Feb 27)

**Dual-mode:**

1. **Per-buyer emission reduction target:** Each participating buyer targets X% clean energy (using the CFE threshold slider, paralleling existing dashboard). The *calculation* of what constitutes "X% clean" differs by strategy:
   - Strategy 1: Cross-regional netting against emission baseline (grid-avg/fossil-avg/marginal)
   - Strategy 2: Hourly matching within ISO (with or without existing clean credit)
   - Strategy 3: Annual MWh matching (same-ISO or cross-regional)

2. **System-wide CO₂ reduction panel:** Separate interactive panel. User sets a system-wide CO₂ reduction target (e.g., "reduce US power sector emissions by 30%"), and the model backs into what corporate participation rate each strategy would need to achieve it. Shows required participation as a function of strategy choice.

### §15.5 Dashboard Page Design (Updated Feb 27)

**Hybrid scrollytell + interactive** with integrated tradeoff matrix, failure mode demonstration, and strategy horse race.

**Core thesis:** Every procurement strategy has failure modes at scale. The design details within each strategy family matter as much as the family choice itself. The page makes the reader uncomfortable about *all* the options, then shows which design choices minimize systemic risk.

**Key framing:** This is not a polemic against consequential — it's a rigorous demonstration that strategy choice is nuanced. The GHG Protocol debate is largely framed as "should we allow consequential?" when the real question is "which *version* of any matching approach actually works at scale?"

**Structure:**
1. **Scrollytell intro:** Explains the three strategy families (1: Consequential, 2: Hourly, 3: Annual), builds intuition about tradeoffs. Key message: "every strategy looks fine at 10% participation — the question is what happens at scale."
2. **Tradeoff matrix:** Summary table showing all 10 strategies × key metrics (cost, CO₂, build required, $/tCO₂) — embedded in scrollytell flow
3. **Failure mode demonstration charts (5 interactive):** Participation slider as unifying x-axis across all charts. All 10 strategies shown. As participation increases, watch each strategy's failure modes activate. Charts link to deep-dive research pages for full analysis.
4. **Strategy horse race:** Fixed outcome comparison + fixed budget comparison — embedded in scrollytell
5. **Interactive explorer:** Strategy selector, participation sliders, CFE threshold slider, ISO selector. Full exploratory mode.
6. **System-wide panel:** CO₂ reduction target → required participation by strategy

#### §15.5.1 Failure Mode Demonstration Charts (Decided Feb 27)

Five interactive charts, all sharing a **participation rate slider** (x-axis: 0-80% of C&I load). Each chart shows all 10 strategy variants as lines/areas. The participation slider is the unifying interaction — as you drag it from 5% to 80%, you watch every failure mode activate in sequence.

**Chart 1: Cost Trajectory Divergence**
- Y-axis: Effective $/MWh at the selected CFE threshold
- Shows all 10 strategies diverging as participation scales
- At low participation, strategies cluster. At high participation, massive spread.
- Key inflection: Where Strategy 2A (all new-build hourly) starts triggering wholesale erosion, converging cost-wise toward Strategy 1 failures despite different mechanism
- Data source: Scenario comparison trajectories + Step 3 cost optimization repriced per strategy

**Chart 2: Capital Allocation by ISO**
- Y-axis: % of total clean energy investment going to each ISO
- Shows geographic clustering: Strategy 1/3B/3D concentrate capital in coal-heavy ISOs (SPP, MISO). Strategy 2/3A/3C forced same-ISO.
- "Fair share" reference lines (proportional to ISO demand)
- Key inflection: 20-30% participation where gas grids drop below fair share under cross-regional strategies
- Data source: Deployment queue from consequential_queue.json, indexed by cumulative TWh as % of C&I load

**Chart 3: Wholesale Price Erosion & Existing Clean Stranding**
- Y-axis: Estimated wholesale LMP ($/MWh) + Section 45U strike price reference
- Shows merit-order effect: as clean penetration rises under each strategy, wholesale prices drop
- **Key insight for Strategy 2 debate:** 2A (all new-build) accelerates wholesale erosion locally — floods market with zero-marginal-cost gen. 2C (premium + new) mitigates via revenue floor for existing generators. 2B falls between.
- Annotation: "When LMP < $44 (45U strike) - operating costs → nuclear stranding begins"
- Links to: [Cost to Replace →] and [New Build Analysis →] for full regional breakdown
- Data source: LMP reconstruction from step5_compute_lmp_prices.py + Track 3 CTR data

**Chart 4: MAC Escalation (Marginal Abatement Cost)**
- Y-axis: $/tCO₂ for the marginal ton abated under each strategy
- Consequential (Strategy 1) starts cheap then hits a wall when coal exhausted
- Hourly (Strategy 2) starts higher but stays flatter — no saturation cliff
- Annual (Strategy 3) variants fall between, depending on boundary + additionality
- Key inflection: Coal exhaustion point (varies by ISO) where Strategy 1 MAC jumps to gas-displacement levels
- Data source: MAC stats from step5_compute_mac_stats.py + deployment queue MAC ordering

**Chart 5: Resource Mix Divergence**
- Y-axis: Stacked resource mix (clean firm, solar, wind, CCS, battery, LDES)
- Side-by-side or toggled comparison at selected threshold showing what gets built under each strategy
- Strategy 1/3 = VRE-heavy, no firm. Strategy 2A = firm + storage + VRE. Strategy 2C = existing + firm + VRE.
- Key insight: Strategy 2A and 2C build different mixes — 2A builds more new firm (drives learning), 2C preserves existing (prevents stranding). Different tradeoff.
- Data source: Scenario comparison resource trajectories + Step 3 resource mix data

#### §15.5.2 Strategy 2 Internal Debate (Decided Feb 27)

The page must demonstrate that the debate within hourly matching (2A vs 2B vs 2C) is as consequential as the debate between strategy families:

| Dimension | 2A (All New) | 2B (Grid Baseline) | 2C (Premium + New) |
|---|---|---|---|
| Wholesale erosion | **Accelerates** — new zero-marginal gen floods market | **Accelerates** — takes credit for existing without supporting it, still adds new | **Mitigates** — premium provides revenue floor for existing |
| Existing clean stranding | **Ignores** — no revenue signal, doesn't acknowledge existing clean exists | **Strands** — takes credit for existing generation without paying for it. Worst of both: claims the benefit while starving generators of revenue signal to stay online | **Addressed** — explicit clean premium keeps plants viable by paying them for being clean |
| Learning curves | **Maximum** — most new firm built earliest | **Moderate** — less new build needed (credited baseline reduces requirement) | **Moderate** — premium $ supports existing, new build on top |
| Cost trajectory | Highest near-term → lowest long-term (FOAK→NOAK) | Cheapest near-term (free-rides on existing), but vulnerable to replacement spike when unpaid existing retires | Higher near-term, avoids replacement spike — pays now to prevent paying more later |
| Firm investment signal | **Strong** — hourly constraint forces it | **Diluted** — baseline credit reduces urgency to build | **Strong** for both existing (premium) + new (hourly constraint) |
| Additionality | **Maximum** — 100% new build | **Problematic** — claims credit for existing without driving new investment or sustaining existing | **Transparent** — explicitly values existing clean (premium) and requires new build on top |

**Key question the charts must demonstrate:** At high participation, does 2A converge toward the same wholesale erosion failure mode as Strategy 1 — just locally instead of cross-regionally? Both flood the market with zero-marginal-cost generation without a mechanism to preserve existing clean. The geography is different but the wholesale destruction is the same.

**2C as the reference strategy — and its one failure mode:** Strategy 2C's failure mode is the inverse of every other strategy. Every other approach degrades as participation *increases* — signal degradation, saturation, wholesale erosion, stranding. 2C's risk is at *insufficient* scale: if participation stays too low, all you've done is pay a premium to keep existing nuclear and hydro running (valuable but not transformative). You never build enough new firm clean to push the learning curve from FOAK to NOAK. The premium keeps existing assets alive but doesn't drive the additionality needed for the NOAK ROI.

Above a critical mass threshold — the participation level where cumulative new-build firm clean investment under 2C is sufficient to trigger Wright's Law learning — 2C is the only strategy that both preserves existing clean AND drives NOAK economics. Below that threshold, it's just an expensive maintenance program for existing generation.

**The chart moment:** Every other strategy line degrades as the participation slider moves right. 2C's risk zone is on the *left* — shaded band below the critical mass threshold. Once past that threshold, 2C is structurally sound at any participation level. The page should make this visually obvious: 2C starts in a "needs more adoption" zone, crosses into "works at any scale," while every other strategy starts in "looks fine" and crosses into various failure modes.

The critical mass threshold is quantifiable via two key numbers:

**1. Critical mass threshold (% participation):** The participation level where *aggregate* new-build firm clean volume across all 7 ISOs under 2C exceeds the deployment needed for Wright's Law cost reductions (first doubling of installed capacity per technology). Critically, learning is **global, not regional** — a nuclear plant built in PJM drives NOAK for nuclear everywhere. LDES deployed in ERCOT brings down iron-air costs in NEISO. So the threshold is lower than a per-ISO calculation would suggest because all 7 ISOs contribute to the same global learning pool.

**2. Investment pool composition:** At each participation level, total 2C spend splits into:
- **Existing clean premiums** (keeping nuclear/hydro online — maintenance spend)
- **New-build capital** (firm clean + storage — learning curve spend)

The split varies dramatically by ISO. Nuclear-heavy ISOs (PJM 32%, NEISO 24%) allocate more to premiums; renewable-heavy ISOs (ERCOT, SPP) have less existing clean to maintain → higher share flows to new-build → they are the **learning curve workhorses** even though they're not where the premium payments concentrate.

**Step 8 computation target:** For each participation level (0-80% of C&I), compute:
- Total new-build firm TWh across all ISOs (the number that matters for Wright's Law)
- Premium spend vs new-build spend by ISO (shows where learning investment concentrates)
- Participation threshold where aggregate new-build hits first doubling → NOAK pricing activates
- Post-NOAK cost trajectory showing the strategy pays for itself

Data sources: Track 2 NB (new-build costs), Track 3 CTR (existing premium costs), learning curve parameters from `step6_scenario_comparison.py`, resource mix data from shared-data.js.

**Step 8 implementation: `scripts/step8_wrights_law_curves.py`** — COMPLETE. Vectorized numpy, no sequential loops. Output:
- `data/step5-wrights/wrights_law_curves.parquet` (4 KB, snappy compressed — 240 rows: 12 participation levels × 20 thresholds)
- `data/step5-wrights/wrights_law_curves.json` (8.4 KB — dashboard-ready figure data)

Key results at 90% CFE:
- **Critical mass threshold: 25% C&I participation** — where cumulative CCS-CCGT deployment exceeds 8 GW globally
- At 95% CFE: 10% participation; at 99.9%: 5% (higher thresholds drive more new-build per participant)
- PJM dominates new-build spend (largest demand, most CCS needed); SPP is 100% premium (all wind, zero new firm)
- Wright's Law gating: learning fraction = 0 below critical mass (maintenance mode), ramps via exponent 0.6 above it
- First-doubling thresholds: nuclear 5 GW, CCS 8 GW, LDES 3 GW, geothermal 2 GW (DOE Liftoff / INL SOAR calibrated)

#### §15.5.3 Cross-References to Deep Dive Pages

The procurement comparison page is a **hub**. At each failure mode inflection point, surface the relevant link:
- **Wholesale erosion / stranding → [Nuclear Revenue Crossover]** (`lmp_trends.html#nuclear-crossover`) — full regional replacement premium analysis (merged from cost_to_replace.html)
- **Wholesale erosion / stranding → [New Build Analysis]** (`new_build_analysis.html`) — supply ceiling, LMP feedback loop, 45U stranding threshold
- **Geographic clustering / saturation → [Consequential Vacuum]** (`consequential_vacuum.html`) — 5 failure modes deep dive with dispatch-based evidence
- **Learning curves / cost trajectory → [Scenario Comparison]** (`scenario_comparison.html`) — FOAK→NOAK dynamics, Scenario A vs B full trajectories
- **MAC escalation → [Abatement Dashboard]** (`abatement_dashboard.html`) — MAC fan charts, DAC crossover, optimal target analysis

Charts on this page show *that* something is happening at a participation threshold. Links take the reader to the page that explains *why* in depth.

#### §15.5.4 Failure Mode × Strategy Matrix (Reference)

| Failure Mode | 1A | 1B | 1C | 2A | 2B | 2C | 3A | 3B | 3C | 3D |
|---|---|---|---|---|---|---|---|---|---|---|
| Signal degradation | Worst | Bad | Less bad | Immune | Immune | Immune | Immune | Immune | Vulnerable | Vulnerable |
| Saturation (coal wall) | Yes | Yes | Yes | N/A | N/A | N/A | N/A | Yes | N/A | Yes |
| Fossil lock-in | Severe | Severe | Severe | Low | Low | Low | Partial | Partial | Severe | Severe |
| Geographic clustering | Core | Core | Core | Impossible | Impossible | Impossible | Impossible | Replicates | Impossible | Replicates |
| Wholesale erosion | Accelerates | Accelerates | Accelerates | **Accelerates** | **Accelerates** | **Mitigates** | Neutral | Accelerates | Accelerates | Accelerates |
| Existing clean stranding | No signal | No signal | No signal | **Ignores** | **Strands** (worst — claims credit without paying) | **Addressed** | Neutral | No signal | Free-rides | No signal |

**Critical nuance:** 2B is arguably *worse* than 2A for existing clean stranding. 2A ignores existing clean — doesn't help, doesn't harm. 2B actively takes credit for existing clean generation (reducing the buyer's procurement cost and requirement) without directing any revenue to those generators. It free-rides on existing clean being online while starving it of the payment signal needed to stay online. When the unpaid existing generation retires, 2B buyers face the same replacement cost spike as everyone else — but they've also reduced the market signal that could have prevented the retirement. Strategy 2C is the only hourly variant that explicitly pays existing generators for being clean, creating the revenue floor needed to prevent premature retirement.

### §15.5.5 Page Narrative Map & Figure Descriptions (Feb 27)

**File:** `dashboard/procurement_comparison.html` (new page — `procurement_research.html` is the content plan, kept separately)

---

#### ACT 1: "EVERY STRATEGY LOOKS FINE AT LOW ADOPTION" (Scrollytell)

**Purpose:** Build the reader's mental model of the three strategy families, then plant the seed that things break at scale.

**Section 1.1 — Opening Hook**
- **Text:** "Today, ~13% of US commercial & industrial electricity is covered by voluntary clean energy procurement. At that level, every strategy works. The question isn't which approach looks best at 13% — it's which ones survive at 40%, 60%, 80%."
- **Visual:** Animated counter showing current market: 315 TWh / 2,400 TWh C&I load = 13%. Simple, cinematic.
- No chart. Just the number landing with weight.

**Section 1.2 — The Three Families**
- **Text:** Brief intro to each strategy family (3 cards, scroll-triggered reveal):
  - **Consequential (Strategy 1):** "Buy the cheapest clean energy anywhere in the US. Net it against your emissions. Maximum flexibility, minimum cost."
  - **Hourly (Strategy 2):** "Match your load hour-by-hour within your own grid region. Most rigorous. Most expensive."
  - **Annual (Strategy 3):** "Match your annual consumption with clean energy certificates. The status quo for most buyers today."
- **Visual:** Three strategy family cards with icons. No chart yet. Clean, simple.

**Section 1.3 — The Variant Tree**
- **Text:** "But within each family, design choices matter enormously. There are 10 distinct strategy variants — and the differences within a family can be larger than the differences between families."
- **FIGURE 1: Strategy Taxonomy Tree**
  - **Type:** Interactive tree/org-chart diagram
  - **Description:** Visual hierarchy: 3 families → 10 variants. Each node shows: variant code (1A, 2C, etc.), one-line description, key distinguishing feature. Color-coded by family. Clicking a variant highlights it across all subsequent charts.
  - **Data source:** Static (strategy definitions from §15.2). No compute needed.
  - **Key design:** This becomes the "legend" for the rest of the page. Reader builds familiarity with the codes here so they can track them through subsequent figures.

---

#### ACT 2: "WHAT BREAKS, AND WHEN" (Scrollytell → Interactive)

**Purpose:** The core analytical payload. Five failure mode demonstrations, each building on the last. Shared participation slider ties them together.

**Transition text:** "Now drag the participation slider from 13% to 80% and watch what happens to each strategy."

**Global control: Participation Slider** — Sticky/floating, visible across all Act 2 charts. Range: 5–80% of C&I load. Default position: 13% (current market). Dragging it right is the primary interaction.

**Section 2.1 — Cost Trajectory Divergence**
- **Lead text:** "At low participation, all strategies cost roughly the same. At high participation, the spread is enormous."
- **FIGURE 2: Cost Divergence Fan**
  - **Type:** Multi-line chart (10 lines, one per strategy variant)
  - **X-axis:** Participation rate (5–80% of C&I load)
  - **Y-axis:** Effective cost ($/MWh) at 90% CFE threshold
  - **Behavior:** At 13%, lines cluster in a $30–50/MWh band. As slider moves right, lines diverge. Strategy 3D (status quo RECs) stays flat and cheap. Strategy 2A (all new-build hourly) rises steeply then curves. Strategy 1A (consequential, grid-avg) stays cheap until ~40% then jumps (coal exhaustion).
  - **Key annotation:** Vertical dashed line at current market (13%). Shaded "comfort zone" where strategies look similar.
  - **Data source:** For strategies 2A and 2C — **existing data**: `track_results.json` (newbuild = 2A, cost_to_replace = 2C) repriced at Medium costs gives $/MWh at each threshold. Map thresholds to participation via SBTi timeline. For 3D — near-zero (unbundled REC price). For 1A/1B/1C — derive from `consequential_queue.json` (deployment queue MAC × emission rate gives effective cost). **Strategies 2B, 3A, 3B, 3C need Step 8 compute** — show as dashed/estimated lines initially.
  - **Callout box:** "The cheap strategies aren't actually cheap — they're deferring costs to the future."

**Section 2.2 — Where the Money Goes (Geographic Clustering)**
- **Lead text:** "Cross-regional strategies chase the cheapest abatement. That concentrates investment in a few regions and starves others."
- **FIGURE 3: Capital Allocation Heatmap**
  - **Type:** Stacked bar chart or heatmap (7 ISOs × selected strategies)
  - **At each participation level:** Shows what % of total clean energy investment goes to each ISO
  - **Strategy 1 (consequential):** Capital clusters in SPP and MISO (coal-heavy, cheapest MAC). CAISO, NEISO, NYISO get almost nothing.
  - **Strategy 2 (hourly):** Capital distributed proportionally — each ISO serves its own load.
  - **"Fair share" reference:** Dashed lines showing demand-proportional allocation.
  - **Data source:** `consequential_queue.json` → `deployment_queue` entries have `iso` field. Sum `delta_cost_total_bn` by ISO at each cumulative step. For hourly: `track_results.json` — inherently same-ISO so allocation = demand share. **Available now for Strategy 1 and 2. Strategy 3 variants need Step 8.**
  - **Callout box:** "When SPP and MISO receive 60%+ of clean investment while serving 25% of demand, the other five regions are subsidizing their transition while getting none of their own."

**Section 2.3 — Wholesale Destruction & Nuclear Stranding**
- **Lead text:** "Every MWh of zero-marginal-cost generation added to a grid pushes wholesale prices down. That's great for consumers — until it kills the existing clean generation you're counting on."
- **FIGURE 4: Wholesale Price Erosion**
  - **Type:** Dual-axis line chart. Primary: LMP ($/MWh). Secondary: Nuclear operating cost reference line.
  - **X-axis:** Participation rate (5–80%)
  - **Lines:** LMP trajectory under Strategy 1 (cross-regional), Strategy 2A (all new, same-ISO), Strategy 2C (premium + new, same-ISO)
  - **Key feature:** Horizontal band at ~$44/MWh (45U strike price) with annotation: "Below this line, existing nuclear can't cover operating costs." When Strategy 2A's line crosses below this band, highlight it.
  - **Strategy 2C difference:** Its line stays higher because the premium mechanism acts as a revenue floor — you're paying existing clean to stay online rather than flooding the market with competing new zero-marginal gen.
  - **Data source:** `data/step4-analysis/lmp/lmp_summary.json` has PJM LMP data. `scenario_comparison.json` has stranding analysis (`stranding_a`, `stranding_b`). **LMP currently computed for PJM only — show PJM as representative, note other ISOs forthcoming.** Track 3 CTR effective costs from `track_results.json` provide the "premium" price signal.
  - **Callout box:** "Strategy 2A and Strategy 1 both destroy wholesale prices — they just do it in different geographies. Strategy 2C is the only variant with a built-in mechanism to prevent it."

**Section 2.4 — The MAC Wall**
- **Lead text:** "Consequential strategies look cheap because they pick off the lowest-hanging fruit first. But that fruit runs out."
- **FIGURE 5: Marginal Abatement Cost Escalation**
  - **Type:** Line chart with shaded uncertainty bands
  - **X-axis:** Cumulative CO₂ abated (Mt) — maps to participation via deployment queue
  - **Y-axis:** Marginal $/tCO₂ for the next ton abated
  - **Strategy 1 (consequential):** Starts at ~$80-100/tCO₂ (coal displacement in SPP/MISO). Stays flat through ~200 Mt. Then **wall** when coal is exhausted → jumps to $300-500+/tCO₂ (gas displacement).
  - **Strategy 2 (hourly):** Starts higher (~$150-200/tCO₂) but stays **flatter** — no saturation cliff because you're always building the full stack (firm + storage + VRE) rather than cherry-picking.
  - **Horizontal reference bands:** DAC ($400-600), EU ETS ($60-100), EPA SCC ($190), Rennert SCC ($185).
  - **Data source:** `mac_stats.json` — `stepwise_fan` has P10/P50/P90 MAC by threshold for all 7 ISOs. `consequential_queue.json` → `deployment_queue` has `marginal_mac` per zone. `scenario_comparison.json` → `queue_a` and `queue_b` have MAC trajectories for both scenarios. **Fully available now.**
  - **Callout box:** "The coal wall is a cliff, not a hill. Once you've displaced all the coal, the next ton costs 3-5× more."

**Section 2.5 — What Gets Built**
- **Lead text:** "Different strategies build different grids. That matters more than the cost difference."
- **FIGURE 6: Resource Mix Comparison**
  - **Type:** Stacked area or grouped bar chart
  - **Comparison:** At a selected threshold (default 90%), show the resource mix under 3-4 key strategies side by side
  - **Strategy 1 (consequential):** Heavy VRE (solar + wind), minimal firm clean, no storage. Gas fills the gaps.
  - **Strategy 2A (all new hourly):** VRE + firm clean + battery + LDES. Balanced portfolio forced by hourly constraint.
  - **Strategy 2C (premium + new hourly):** Existing nuclear/hydro preserved + new firm + VRE + storage. Most diversified.
  - **Strategy 3D (status quo annual):** Unbundled RECs from existing — no new build at all. Cheapest but builds nothing.
  - **Color coding:** Solar=amber, Wind=blue, Clean Firm=green, CCS=teal, Hydro=cyan, Battery=purple, LDES=pink (per project standard)
  - **Data source:** `track_results.json` → `resource_mix` for newbuild (2A) and cost_to_replace (2C) at each threshold. `scenario_comparison.json` → `trajectories` → `pure_consequential` has `resource_twh` per threshold for Strategy 1. `shared-data.js` → `RESOURCE_MIX_DATA` has Medium-cost mix by ISO. **Available now for 1, 2A, 2C. 3D is trivially zero new-build.**
  - **Callout box:** "Hourly matching is the only approach that forces investment in the resources you actually need for a deeply decarbonized grid — firm clean generation and long-duration storage."

---

#### ACT 3: "THE DEBATE WITHIN HOURLY" (Scrollytell)

**Purpose:** Shift from family-level comparison to the 2A vs 2B vs 2C internal debate. This is the most nuanced section — it argues that the debate *within* hourly matching is as important as the debate *between* strategies.

**Transition text:** "If hourly matching is the most robust family, the next question is: which version? The differences are larger than you'd think."

**Section 3.1 — The Stranding Paradox**
- **Text:** "Strategy 2A (all new-build) sounds maximally additional. But it ignores the 380 TWh of existing nuclear and hydro already running on US grids. Strategy 2B claims credit for that existing clean without paying for it — the worst of both worlds. Strategy 2C pays existing generators a premium to stay online, then builds new on top."
- **FIGURE 7: Strategy 2 Internal Comparison Table**
  - **Type:** Animated comparison table (not a chart — a styled, scroll-triggered table)
  - **Rows:** 6 dimensions: Wholesale erosion, Existing clean stranding, Learning curves, Cost trajectory, Firm investment signal, Additionality
  - **Columns:** 2A, 2B, 2C — color-coded cells (green=good, amber=mixed, red=bad)
  - **Scroll animation:** Rows reveal one at a time as user scrolls. Each row highlights the "winner" and "loser."
  - **Data source:** Static (the §15.5.2 comparison table). No compute needed.
  - **Key moment:** When the "Existing clean stranding" row reveals, 2B's cell turns red with bold text: "Worst — claims credit without paying." This is the insight most readers won't expect.

**Section 3.2 — The 2C Critical Mass Question**
- **Text:** "Every other strategy degrades as adoption increases. Strategy 2C has the opposite problem — it needs *enough* adoption to work. Below a critical mass threshold, you're just paying premiums to keep existing plants alive. Above it, you've funded the learning curve that makes new clean firm affordable everywhere."
- **FIGURE 8: The 2C Threshold Diagram**
  - **Type:** Single-line chart with shaded zones
  - **X-axis:** Participation rate (5–80%)
  - **Y-axis:** Effective $/MWh for Strategy 2C (blended existing premium + new build)
  - **Key feature:** Two shaded zones:
    - **Left zone (red/amber):** "Maintenance mode" — below critical mass. Premium spend dominates. Not enough new-build volume to trigger learning.
    - **Right zone (green):** "Learning activated" — past critical mass. Aggregate new-build firm across all ISOs hits first Wright's Law doubling. NOAK pricing begins. Cost curve bends down.
  - **Vertical line:** Critical mass threshold (computed from aggregate new-build TWh needed for first doubling)
  - **Contrast overlay:** Faded lines for other strategies showing their degradation at high participation — 2C is the only one that *improves* past its threshold.
  - **Data source:** `track_results.json` cost_to_replace effective costs for 2C base. `scenario_comparison.json` trajectories for learning curve application. Critical mass threshold = **Step 8 compute needed** for exact number, but can estimate from existing resource mix data (how much new firm TWh at each participation level × learning curve parameters from `step6_scenario_comparison.py`).
  - **Callout box:** "The critical mass threshold is lower than you'd think — because learning is global. A nuclear plant built in PJM drives NOAK pricing for nuclear in NEISO. All 7 ISOs contribute to the same global learning pool."

**Section 3.3 — The Regional Role Map**
- **Text:** "Under Strategy 2C, different regions play different roles. Nuclear-heavy ISOs (PJM, NEISO) are the premium-payers — their spend keeps existing clean alive. Renewable-rich ISOs (ERCOT, SPP) are the learning-curve drivers — their spend deploys the new technologies. Both are essential."
- **FIGURE 9: Investment Pool Composition by ISO**
  - **Type:** Stacked bar chart (7 ISOs)
  - **Each bar split into:** Existing clean premium (gray/blue) vs. New-build capital (green/amber)
  - **At selected participation level** (linked to global slider)
  - **Key insight:** PJM's bar is 60-70% premium (keeping 32% nuclear fleet online). ERCOT's bar is 80%+ new-build (little existing clean to maintain). Both contribute to the same NOAK outcome.
  - **Data source:** `track_results.json` — newbuild (2A) gives new-build costs; cost_to_replace (2C) gives total including premium. Delta = premium portion. **Available for all 7 ISOs** (CAISO, ERCOT, NEISO, NYISO, PJM, MISO, SPP).
  - **Callout box:** "PJM free-rides on the learning curve that ERCOT is paying for — and that's the system working as designed."

---

#### ACT 4: "THE TIMELINE TRAP" (Scrollytell)

**Purpose:** Introduce the temporal dimension — SBTi milestones create a timeline that makes strategy choice path-dependent. What you choose at 50% determines what you face at 90%.

**Section 4.1 — The SBTi Ratchet**
- **Text:** "Corporate decarbonization isn't a static optimization — it's a 25-year ratchet. Science-based targets lock in progressively tighter commitments: 50% by 2030, 70% by 2035, 90% by 2040, 100% by 2050. What you build at 50% determines what you face at 90%."
- **FIGURE 10: The FOAK→NOAK Timeline**
  - **Type:** Dual-trajectory line chart with SBTi milestone markers
  - **X-axis:** Year (2025–2050), with SBTi milestone markers
  - **Y-axis:** Clean firm LCOE ($/MWh) — blended nuclear + CCS + LDES
  - **Line A (Strategy 1/3 — Consequential/Annual):** Stays at FOAK through 2035 (no investment). Learning starts 2035, compressed. Still near-FOAK at 2040 (the 90% milestone). Reaches NOAK only by ~2047.
  - **Line B (Strategy 2 — Hourly):** FOAK from 2025-2030. Learning 2030-2040. NOAK by 2040 — right when you need it for the 90% target.
  - **Shaded band:** Cost difference area between the two lines = "the learning curve premium" — what hourly matching costs upfront vs. what it saves long-term.
  - **SBTi markers:** Vertical dashed lines at 2030 (50%), 2035 (70%), 2040 (90%), 2050 (100%)
  - **Data source:** `step6_scenario_comparison.py` → `learning_fraction()` gives the curve shape. LCOE_TABLES from `shared-data.js` give FOAK (High) and NOAK (Low) costs. `scenario_comparison.json` → `trajectories` have `blended_new_lcoe` at each threshold/year. **Fully available now.**
  - **Callout box:** "By 2040, hourly matching has driven clean firm costs to NOAK. Consequential strategies are still paying near-FOAK — for the same technology, at the same time, because they delayed investment."

**Section 4.2 — Three Compounding Failures**
- **Text:** Brief narrative on the three adverse effects of delayed hourly matching (from §15.11):
  1. Learning curve delay → FOAK at 90%
  2. Stranded VRE overbuild → curtailed solar doesn't help at night
  3. Gas lock-in → no storage signal, gas fills by default
- **FIGURE 11: The Compounding Timeline**
  - **Type:** SBTi milestone comparison table (styled, scroll-animated)
  - **Rows:** 4 SBTi milestones (2030/50%, 2035/70%, 2040/90%, 2050/100%)
  - **Columns:** "Strategy 1/3 (Annual/Consequential)" vs "Strategy 2 (Hourly)"
  - **Cell content:** Status descriptor + cost indicator. E.g., 2040 row: Strategy 1/3 = "WALL — VRE saturated, firm at FOAK, gas locked in" (red). Strategy 2 = "Firm at NOAK, storage mature, gas retiring" (green).
  - **Data source:** Static text from §15.11 table, enriched with cost numbers from trajectory data. **Available now.**

---

#### ACT 5: "THE HORSE RACE" (Interactive)

**Purpose:** Direct comparison mode. Same outcome, which strategy gets there cheapest? Same budget, which strategy achieves the most?

**Section 5.1 — Fixed Outcome: "Get to 90% CFE. What does it cost?"**
- **FIGURE 12: Cost to Reach 90% by Strategy**
  - **Type:** Horizontal bar chart (10 strategies ranked by cost)
  - **Y-axis:** Strategy variants (labeled)
  - **X-axis:** Effective $/MWh to achieve 90% hourly CFE (or equivalent)
  - **Key insight:** Strategy 3D is "cheapest" but achieves nothing physical. Strategy 2A is most expensive but builds the most. Strategy 2C is moderate and sustainable.
  - **Annotations:** Each bar annotated with what it actually built (resource mix icons) and what's at risk (failure mode flag)
  - **Data source:** For 2A and 2C: `track_results.json` at threshold 90, Medium scenario. For Strategy 1: `scenario_comparison.json` trajectory at 90% threshold. **Available now for 1, 2A, 2C. Others need Step 8.**
  - **Toggle:** ISO selector (default: all-ISO weighted average)

**Section 5.2 — Fixed Budget: "Spend $60/MWh. What do you get?"**
- **FIGURE 13: Achievement at Fixed Budget**
  - **Type:** Horizontal bar chart (10 strategies ranked by CFE% achieved)
  - **X-axis:** CFE threshold achieved with $60/MWh budget
  - **Annotations:** Each bar annotated with CO₂ abated and resource mix
  - **Data source:** Interpolate from cost curves at each threshold. **Derivable from existing data for strategies with cost trajectories.**

---

#### ACT 6: "THE SYSTEM VIEW" (Interactive Panel)

**Purpose:** Flip the question. Instead of "what does it cost per buyer?" ask "what participation level does each strategy need to hit a system-wide CO₂ target?"

**Section 6.1 — Required Participation by Strategy**
- **FIGURE 14: Participation Required for 30% US Power Sector CO₂ Reduction**
  - **Type:** Horizontal bar chart or gauge visualization
  - **Y-axis:** Strategy variants
  - **X-axis:** Required C&I participation (% of load)
  - **Key insight:** Some strategies can't get there at any participation level (supply constraints, saturation). Others need implausibly high participation. A few are feasible at realistic levels.
  - **Infeasibility markers:** Strategies that hit physical supply ceilings before reaching the target show as "infeasible" with hatched bars.
  - **Data source:** Requires mapping strategy → CO₂ displaced at each participation level. `consequential_queue.json` has `co2_displaced_mt` per step for consequential. `co2_results.json` for dispatch-based emission reductions (currently PJM only). **Partial data — full computation is Step 8.**
  - **User control:** CO₂ reduction target slider (10–50% of US power sector emissions)

**Section 6.2 — Interactive Explorer (Full Controls)**
- **Text:** "Explore the full parameter space."
- **Controls:**
  - Strategy selector (checkbox — select multiple to compare)
  - Participation slider (5–80%)
  - CFE threshold slider (50–100%)
  - ISO selector (7 ISOs or all)
  - Learning curves toggle (On/Off)
- **FIGURE 15: Explorer Output Panel**
  - Multi-panel: Cost ($/MWh), Resource mix (stacked bar), CO₂ abated (bar), MAC (line), Gas capacity (bar)
  - Updates in real-time as controls change
  - **Data source:** All existing data files, combined. This is the "power user" interface. **Largely available for strategies with computed data.**

---

#### PAGE FOOTER

Cross-reference links to deep-dive pages:
- Nuclear Revenue Crossover → `lmp_trends.html#nuclear-crossover` (merged from cost_to_replace.html)
- New Build Analysis → `new_build_analysis.html`
- Consequential Vacuum → `consequential_vacuum.html`
- Scenario Comparison → `scenario_comparison.html`
- Abatement Dashboard → `abatement_dashboard.html`

---

#### DATA AVAILABILITY SUMMARY

| Figure | Strategies with Data Now | Strategies Needing Step 8 |
|--------|--------------------------|---------------------------|
| Fig 1 (Taxonomy tree) | All 10 (static) | — |
| Fig 2 (Cost divergence) | 1A-C (from queue), 2A, 2C, 3D (≈$0) | 2B, 3A, 3B, 3C |
| Fig 3 (Capital allocation) | 1 (queue), 2 (inherently same-ISO) | 3 variants |
| Fig 4 (Wholesale erosion) | 2A, 2C (PJM LMP data) | 1, 2B, 3 |
| Fig 5 (MAC escalation) | 1 (queue MACs), 2 (mac_stats) | 3 variants |
| Fig 6 (Resource mix) | 1 (trajectory), 2A, 2C, 3D (=nothing) | 2B, 3A, 3B, 3C |
| Fig 7 (2ABC table) | All (static) | — |
| Fig 8 (2C threshold) | 2C (track data + learning curve) | Critical mass exact point |
| Fig 9 (Regional roles) | 2C (track data, 7 ISOs) | — |
| Fig 10 (FOAK→NOAK) | Both (learning_fraction + LCOEs) | — |
| Fig 11 (Compounding table) | All (static + trajectory data) | — |
| Fig 12 (Horse race: cost) | 1, 2A, 2C | 2B, 3A-D |
| Fig 13 (Horse race: budget) | 1, 2A, 2C | 2B, 3A-D |
| Fig 14 (System CO₂) | Partial (consequential queue CO₂) | Most strategies |
| Fig 15 (Explorer) | 1, 2A, 2C | 2B, 3A-D |

**Bottom line:** Acts 1–4 (the scrollytell narrative) are ~80% buildable with existing data. Strategies 2A and 2C have the richest data. Strategy 1 has deployment queue data. The remaining strategies (2B, 3A-3D) need Step 8 compute for precise numbers but can be shown as estimated/dashed lines derived from the strategies we do have.

### §15.6 Emission Rate Data (Research, Feb 27)

| ISO | Grid Avg (tCO₂/MWh) | Fossil Avg | Marginal | Marginal vs Fossil |
|-----|---------------------|------------|----------|-------------------|
| CAISO | 0.168 | 0.392 | 0.397 | ≈ same |
| ERCOT | 0.333 | 0.535 | 0.526 | ≈ same |
| PJM | 0.325 | 0.539 | 0.573 | +6% |
| MISO | 0.354 | 0.567 | 0.663 | **+17%** |
| NYISO | 0.217 | 0.415 | 0.437 | +5% |
| NEISO | 0.246 | 0.387 | 0.425 | +10% |
| SPP | 0.340 | 0.544 | 0.665 | **+22%** |

Sources: EPA eGRID2023 (grid-avg, fossil-avg), VERACI-T/WattTime (marginal), Holland et al. 2022 PNAS.

### §15.7 EAC Scarcity by ISO (Research, Feb 27)

| ISO | Total Clean (TWh) | Committed (TWh) | Available for Voluntary (TWh) | REC Price Signal |
|-----|-------------------|-----------------|------------------------------|-----------------|
| CAISO | ~172 | ~158-164 | 10-20 | Moderate |
| ERCOT | ~200-205 | ~30-40 | **130-160** | Very Low ($1-5) |
| PJM | ~310-330 | ~230-290 | 50-80 | High ($35+) |
| MISO | ~200-215 | ~170-230 | 30-50 | Low-Moderate |
| NYISO | ~62-66 | ~47-63 | 5-15 | High ($20-35) |
| NEISO | ~43-45 | ~35-60 | **3-8** | Critical (~$40) |
| SPP | ~136 | ~65-90 | 40-60 | Very Low ($1-5) |

**Key finding:** 20x scarcity variation across ISOs. ERCOT has ~130-160 TWh unclaimed; NEISO has ~3-8 TWh. This directly drives the economics of cross-regional (Strategy 1/3B) vs same-ISO (Strategy 2/3A) strategies.

### §15.8 C&I Load Share (Research, Feb 27)

National C&I = ~62% of total US load (~2,400 of ~3,860 TWh). Range by ISO: 52-57% (NEISO) to 63-67% (ERCOT). Voluntary procurement currently covers ~13% of C&I load (~315 TWh/yr, NREL 2024).

### §15.9 Corporate Procurement Market (Research, Feb 27)

- Voluntary market: ~315 TWh (2024), 7.7% of total US demand
- Corporate PPAs: 28 GW signed in 2024, 29.5 GW in 2025 (BNEF)
- Concentration: Tech/data = 84% of deal activity; Big 4 hyperscalers = 49% of global activity
- Unique US buyers: Fell 51% YoY to 33 companies in 2025
- 41% of all US clean energy added since 2014 was corporate-procured (CEBA)
- Long-term contracts (PPAs + utility) overtook unbundled RECs in 2023 (~46% of volume)

---

## 1. Model Framework

- **2025 snapshot model** — all data, profiles, costs, grid mix shares reflect fixed 2025 actuals
- **No demand growth projections** — point-in-time scenario analysis only
- **Grid mix baseline** = actual 2025 regional shares, priced at wholesale, selectable as reference scenario (fixed, not adjustable by user)
- **Regions**: CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP (7 ISOs)
- **Repo**: `jessicacohen554-cyber/hourly-cfe-optimizer`

---

## 2. Resources (6 total — v4.0 rebuild: CCS merged into Clean Firm)

| # | Resource | Profile Type | New-Build? | Cost Toggle? | Transmission Adder? |
|---|---|---|---|---|---|
| 1 | **Clean Firm** (nuclear/geothermal/CCS-CCGT) | Blended: seasonal-derated baseload (nuclear/geo) + flat baseload (CCS) | Yes | Low/Med/High (regional) | Yes (regional) |
| 2 | **Solar** | EIA 2025 hourly regional | Yes | Low/Med/High (regional) | Yes (regional) |
| 3 | **Wind** | EIA 2025 hourly regional | Yes | Low/Med/High (regional) | Yes (regional) |
| 4 | **Hydro** | EIA 2025 hourly regional | **No** — capped at existing | **No** — wholesale only | **No** — always $0 |
| 5 | **Battery** (4hr Li-ion) | Daily cycle dispatch | Yes | Low/Med/High (regional) | Yes (regional) |
| 6 | **LDES** (100hr iron-air) | Multi-day/seasonal dispatch | Yes | Low/Med/High (regional) | Yes (regional) |

### v4.0 Change: CCS-CCGT merged into Clean Firm (Decision 6D)
- **Rationale**: Reduces resource mix search space from 5D to 4D, dramatically cutting grid search combinatorics (~40-60% fewer combos). Both nuclear and CCS-CCGT are modeled as baseload (CCS runs flat due to 45Q incentives), making them functionally similar for dispatch purposes.
- **Implementation**: The optimizer allocates a single `clean_firm` percentage. Within that allocation, the sub-split between nuclear/geothermal and CCS-CCGT is determined by cost optimization — the cost model evaluates different sub-allocations and picks the cheapest blend. CCS retains its distinct cost profile (LCOE, 45Q offset, fuel linkage) and emission characteristics (95% capture, residual 0.0185 tCO2/MWh).
- **Dispatch profile**: Weighted blend of nuclear seasonal-derated profile and CCS flat profile, based on sub-allocation ratio.
- **Dashboard impact**: Results still report the nuclear/CCS sub-split for transparency.

### Key resource decisions:
- **Green H2 seasonal storage** (added Feb 2026):
  - **Parameters**: 35% round-trip efficiency (electrolysis 70% × storage 95% × turbine 55%), 1000hr duration (~42 days at full power), 30-day rolling dispatch window
  - **Physics**: Dispatches as Phase 4 after battery4 → battery8 → LDES on post-LDES residual surplus/gap. Same window-based charge/discharge as LDES but with longer window and lower RTE.
  - **Sweep levels**: Only evaluated at ≥95% thresholds (too expensive for lower). Levels: [0, 0.3, 1.0] % of demand (reduced March 2026 from 9 levels — H2 never won on cost, and the 9-level grid added a 9× multiplier to Step 1C combo counts causing high-threshold cells to stall).
  - **Cost**: LCOS-based, shares `ldes_lvl` sensitivity toggle. L=$185-230, M=$260-330, H=$365-460 $/MWh by ISO. Transmission adders: L=$2-3, M=$3-6, H=$5-10.
  - **Peak capacity credit**: 0.85 (dispatchable but slower ramp than gas/battery)
  - **Merit order rationale**: Battery → LDES → H2 is economically robust because (1) higher RTE storage should fill short gaps first to minimize surplus waste, (2) battery $/kW is lower than LDES for 4hr needs, (3) H2's only advantage is very cheap $/kWh (salt caverns) at multi-week timescales where LDES is prohibitively expensive.
- **CAISO geothermal as 5th physics dimension** (added Feb 2026):
  - CAISO uses 5D grid search: [clean_firm (nuclear/CCS only), solar, wind, hydro, geothermal] — each as independent % of demand (no sum constraint).
  - **Geothermal profile**: Flat year-round (1/8760 per hour). No seasonal derate — geothermal has no refueling outages.
  - **CAISO clean_firm profile**: Now purely nuclear with full seasonal derate (NUCLEAR_SHARE_OF_CLEAN_FIRM = 1.0 for CAISO). The 70/30 nuclear/geo blend is removed; geothermal physics are captured by the separate dimension.
  - **Geothermal cap**: (existing_geo_TWh + GEO_CAP_TWH) / CAISO_demand_TWh = (5.31 + 39.0) / 224.039 = 19.8% → capped at 20% in grid search.
  - **Non-CAISO ISOs**: Stay 4D. No geothermal resource.
  - **Rationale**: Geothermal has fundamentally different physics than nuclear/CCS (no seasonal derate, no outages). Lumping into clean_firm understated CAISO's winter/spring firm capacity.
- **Clean Firm nuclear derate**: Seasonal spring/fall derate applied to nuclear/CCS portion. Reflects staggered refueling outages for nuclear and scheduled maintenance for CCS-CCGT in shoulder months. Summer/winter: ~100% CF. Spring/fall: reduced CF based on observed EIA 2021-2025 patterns. CCS-CCGT aggregate fleet maintenance in shoulder months produces a similar derate pattern.
- **Hydro**: Existing only, capped at regional capacity, wholesale priced, no new-build tier, $0 transmission
- **CCS-CCGT** (within Clean Firm): 95% capture rate, residual ~0.0185 tCO2/MWh, 45Q ($85/ton = ~$27.5/MWh offset) baked into LCOE, fuel cost linked to gas price toggle. **Modeled as flat baseload (not dispatchable) by design** — while CCS-CCGT is physically dispatchable, the 45Q tax credit ($85/ton for geologic storage) incentivizes running at maximum capacity factor to maximize capture credits. This is an economics-driven decision, not a physical constraint.
- **LDES**: 100-hour iron-air, 50% round-trip efficiency, capacity-constrained dispatch with dynamic capacity sizing. LCOS reflects actual utilization of built capacity. (Decision 7A — kept current.)
- **Battery**: 4-hour Li-ion, 85% round-trip efficiency, capacity-constrained daily-cycle dispatch. LCOS reflects actual utilization — oversized capacity that sits idle drives cost up. (Decision 7A — kept current.)

---

## 3. Thresholds (20 total — v4.2: added 10/20/30/40 coarse low range + 99.5/99.9 last-mile; v4.3: dropped 99.99)

```
10, 20, 30, 40 [coarse only], 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9
```

- **10%, 20%, 30%, 40%** (v4.2): Coarse-grid only — no fine zone search, no step1d storage refinement. Captures early adoption / RPS-range where mixes are easy to achieve and cost curves are flat.
- **50%, 55%, 60%, 65%, 70%**: Captures the easy-to-achieve baseline region where most mixes succeed. 5% granularity anchors the cost curve left side.
- 5% intervals from 75–85 (captures broad trend)
- 2.5% intervals from 87.5–97.5 (captures steep cost inflection zone)
- **99%, 99.5%, 99.9%** (v4.2 added 99.5/99.9): Last-mile granularity at the near-perfect end. ≥99.9% is the ceiling, labeled "effectively 100%" (8.76 unmatched hours/year). True 100% is physically unreachable.
- Key inflection behavior (CCS/LDES entering mix, storage costs spiking) captured at 90–97.5
- Dashboard interpolates smoothly between anchor points for abatement curves

---

## 4. Dashboard Controls (7 total — paired toggles)

### Preserved (2):
1. **Region/ISO select** (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP — 7 ISOs)
2. **Threshold select** (20 values: 10, 20, 30, 40 [coarse only], 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, 99.9)

### Sensitivity toggles (7 toggles + 1 binary switch):

Cost sensitivities are organized into 7 graduated toggles (L/M/H) plus one binary policy switch (45Q). CCS and Geothermal are separated from Firm Gen to allow independent sensitivity analysis of these distinct technologies.

| # | Toggle | Options | Controls | Affects |
|---|---|---|---|---|
| 3 | **Renewable Generation Cost** | Low / Medium / High | Solar LCOE + Wind LCOE | Both solar and wind generation costs (regional) |
| 4 | **Firm Generation Cost** | Low / Medium / High | Clean Firm (nuclear) LCOE — uprate + new-build | Nuclear uprate and new-build costs (regional) |
| 5 | **Storage Cost** | Low / Medium / High | Battery LCOS + LDES LCOS | Both storage technology costs (regional) |
| 6 | **CCS Cost** | Low / Medium / High | CCS-CCGT underlying cost (capex, transport, storage) | CCS technology maturity — L=mature/low capex, H=immature/high capex |
| 7 | **45Q Credit** | On / Off | $29/MWh 45Q tax credit offset on CCS LCOE | Binary policy switch — On=full 45Q offset, Off=no offset |
| 8 | **Fossil Fuel Price** | Low / Medium / High | Gas + Coal + Oil prices | Wholesale electricity price + CCS fuel cost + emission rates |
| 9 | **Transmission Cost** | None / Low / Medium / High | All resource transmission adders | Transmission adders on all new-build resources (regional) |
| 10 | **Geothermal Cost** | Low / Medium / High | Geothermal LCOE (CAISO only) | **CAISO only** — no geothermal resource in other ISOs |

**Toggle separation rationale**:
- **CCS separated from Firm Gen**: CCS has a distinct cost structure (capture + transport + storage + fuel) and policy dependency (45Q) that makes it independently variable from nuclear. Pairing them hides the 45Q sensitivity.
- **Geothermal separated and CAISO-only**: Geothermal is a regionally constrained resource — only CAISO has meaningful hydrothermal potential (5 GW cap from USGS identified resources). Other ISOs have zero geothermal potential for power generation. Toggle is hidden/disabled for non-CAISO regions.
- **45Q as binary switch**: The 45Q credit is a policy decision (exists or doesn't), not a cost spectrum. Keeping it binary allows clean analysis of "what if 45Q expires/isn't renewed."

**L/M/H maturity mapping for CCS**:
- **Low**: Mature CCS deployment — nth-of-a-kind plants, established Class VI wells, optimized CO₂ transport networks, low capex
- **Medium**: Mid-range — some learning curve benefits, moderate infrastructure availability
- **High**: Immature/early deployment — first-of-a-kind plants, new well permitting, long transport distances, high capex

**Scenario count**:
- Non-CAISO: 3×3×3×3×2×3×4 = **5,832 cost scenarios** per region per threshold
- CAISO: 5,832 × 3 = **17,496 cost scenarios** per threshold (includes geothermal toggle)
- Total: 17,496 + 5,832×4 = **40,824 scenarios** per threshold set
- All Step 3 (arithmetic on cached physics) — runs in minutes, not hours

**Sensitivity key format**:
- Non-CAISO: `RFSC_QFF_TX` (e.g., `MMMM_1M_M` = all Medium, 45Q on)
- CAISO: `RFSC_QFF_TX_G` (e.g., `MMMM_1M_M_M` = all Medium, 45Q on, Medium geo)
- Q = `1` (45Q on) or `0` (45Q off)

**NOTE**: All graduated toggles use **Low / Medium / High** naming consistently (never "Base" or "Baseline").

**Optimizer approach**: Resource mix co-optimized with costs for EVERY scenario. Different cost assumptions produce different optimal resource mixes — this is the core scientific contribution. Physics cached from Step 1; Step 3 cross-evaluates all EF mixes under each sensitivity combo to find the cheapest valid mix.

### 4.1 Warm-Start Optimization (Trifold Seed Strategy)

**Problem**: Full 3-phase co-optimization (Phase 1 coarse grid → Phase 2 medium refinement → Phase 3 fine-tune) takes 5-10 minutes per scenario. With 44 representative scenarios per threshold × 20 thresholds × 7 ISOs, full Phase 1 for every scenario is prohibitively slow.

**Solution**: Trifold warm-start seeding — run full 3-phase for 3 categories of scenarios, then warm-start the remainder with the discovered mix archetypes:

1. **Medium scenario** (`MMM_M_M`): Full 3-phase. The central cost case — most scenarios converge to similar mixes since physics dominates.
2. **Extreme archetype scenarios** (7 scenarios): Full 3-phase. These represent opposite corners of the cost space where the optimal mix is most likely to diverge from Medium:
   - `HLL_L_N` — High renewables, low firm/storage/fuel, no transmission (VRE-dominant)
   - `LHL_L_M` — High firm, low renewables (firm-dominant)
   - `LLH_H_M` — High storage, high fuel (storage-dominant)
   - `HHH_H_H` — All high (maximum cost pressure)
   - `LLL_L_L` — All low (minimum cost environment)
   - `HLL_L_H` — High renewables + high transmission (VRE with tx penalty)
   - `LHL_H_N` — High firm + high fuel, no transmission (fuel-stressed firm)
3. **All remaining scenarios**: Warm-started from the diverse seed pool discovered in steps 1-2. Skip Phase 1 coarse grid; start with seed mixes + their 5% neighborhoods + edge-case seeds, then run full Phase 2/3 refinement.

**Dynamic archetype discovery**: As warm-started scenarios find new mix archetypes (mixes that differ by >5% in any resource dimension from known archetypes), these are automatically added to the seed pool for subsequent scenarios. This ensures rare-but-valid mixes discovered mid-run are propagated forward.

**Fallback**: If warm-start fails to find any feasible solution for a scenario, it automatically falls back to full Phase 1 coarse grid search.

**Scientific validity**: This approach is equivalent to providing a smarter initial guess, not a shortcut:
- Hourly matching scores are physics-based and cost-independent. The score cache is shared across all scenarios (same mix + procurement → same physics).
- Phase 2 (5% neighborhood) and Phase 3 (1% neighborhood) refinement run identically regardless of whether warm-start or full Phase 1 was used.
- The only difference is the set of candidates entering Phase 2. Warm-start uses the discovered archetype pool + edge seeds instead of the full 270-combo coarse grid. Since most of those 270 combos evaluate to the same few optimal regions anyway (especially at lower thresholds), the archetype pool covers the same solution space more efficiently.
- Cross-pollination after all scenarios still evaluates every discovered mix against every cost scenario, catching any missed optimizations.
- Monotonicity re-sweep uses full Phase 1 (resweep=True disables warm-start), providing an additional safety net.

**Expected speedup**: ~3-5× per threshold (estimated reduction from ~5-10 min/scenario to ~1-3 min/scenario for warm-started cases).

**Risks and limitations**:
1. **Missed global optima at extreme cost corners**: If an extreme cost combination produces an optimal mix radically different from any archetype, warm-start's neighborhood search might not find it. **Mitigation**: The 7 extreme archetype scenarios cover the most divergent cost corners; dynamic archetype discovery catches emergent patterns; cross-pollination provides a second chance; monotonicity re-sweep with full Phase 1 provides a final safety net.
2. **Phase 2 neighborhood radius**: The 5% step with radius 2 covers ±10% in each resource dimension from the warm-start mix. Optimal mixes more than 10% away in any dimension from all seed archetypes would be missed. **Mitigation**: Edge-case seeds (100% solar, 100% wind, etc.) are always included regardless of warm-start. At observed convergence rates, ≤14 unique mixes typically serve 5,832 scenarios at lower thresholds, well within the archetype pool's coverage.
3. **Threshold-dependent risk**: Higher thresholds (95-100%) have more diverse optimal mixes across cost scenarios. **Mitigation**: The archetype pool grows dynamically; extreme scenarios are more likely to diverge at high thresholds, populating the pool with the right seeds.
4. **Not used during re-sweep**: Monotonicity re-sweep always uses full Phase 1 (warm_start_result is not passed when resweep=True). This is intentional — re-sweep needs the broadest possible search to resolve violations.

### 4.2 Scenario Pruning & Adaptive Resampling Pipeline

**Problem**: 5,832 cost scenarios × 20 thresholds × 7 ISOs = 816,480 co-optimizations (16 active thresholds for full cost optimization, 4 coarse thresholds for coarse pass only). Even with warm-start, running all 5,832 per threshold is slow. Empirically, physics dominates at lower thresholds — only ~14 unique mixes serve all 5,832 scenarios.

**Solution**: 5-stage pipeline runs 44 representative scenarios, then fills the remaining ~5,788 via cross-pollination, with adaptive resampling as a safety net.

#### Stage 1: Medium Seed (1 scenario)
- Run `MMM_M_M` with full 3-phase optimization (no warm-start)
- Becomes the primary warm-start seed for all subsequent scenarios

#### Stage 2: Extreme Archetypes (7 scenarios)
- Run 7 corner scenarios with full Phase 1 (no warm-start): `HLL_L_N`, `LHL_L_M`, `LLH_H_M`, `HHH_H_H`, `LLL_L_L`, `HLL_L_H`, `LHL_H_N`
- These explore the most divergent regions of cost space to discover distinct mix archetypes

#### Stage 3: Remaining Representatives (~36 scenarios, totaling ~44)
- `_build_representative_scenarios()` generates a set of ~54 keys covering cost space corners, axis sweeps, and diagonals. After dedup, ~44 unique scenarios.
- The ~36 scenarios not already run as Medium/archetypes are warm-started from Medium + all diverse seed mixes discovered in Stages 1-2
- New archetypes discovered during this stage are dynamically added to the seed pool

#### Stage 4: Adaptive Resampling (if needed)
- After Stage 3, count unique resource mix archetypes found across the ~44 scenarios
- **Uniqueness threshold**: 50% — if unique mixes > 50% of scenarios run (i.e., >22 unique mixes from 44 scenarios), the representative set didn't adequately capture the diversity
- **Action**: Add midpoint scenarios from the unrun 280, spread evenly across cost space
- Target: enough additional scenarios to bring the ratio below 50%
- Up to 5 resampling rounds, each adding scenarios until convergence
- **If unique mixes ≤ 22**: Proceed directly — the 44 representatives captured the full archetype space

#### Stage 5: Cross-Pollination (fills remaining to 5,832)
- Collect all unique mixes discovered across Stages 1-4
- For ALL 5,832 scenarios (including the ~5,788 not directly optimized): evaluate every discovered mix under that scenario's cost function
- If a mix found optimal for scenario A is cheaper for scenario B than B's current best, assign it
- Result: all 5,832 scenarios have cost-optimal assignments, even the ~5,788 that were never directly optimized

**Why this works**: At lower thresholds, physics strongly constrains the feasible solution space — the same ~10-14 resource mixes are optimal across all 5,832 cost scenarios, just at different costs. Cross-pollination guarantees every scenario gets the cheapest-for-it mix from the full discovered set. Adaptive resampling is the safety net: if we're seeing more diversity than expected (>22 unique from 44), we add more direct optimizations to make sure we're not missing archetypes.

**Applies to all thresholds**: `PRUNING_THRESHOLD_CUTOFF = 100` — empirically, even at 95-100%, the archetype pool from 44 reps + resampling + cross-pollination captures the full solution space.

---

## 5. Complete Cost Tables

### 5.1 Solar LCOE ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $45 | $40 | $50 | $70 | $62 |
| Medium | $60 | $54 | $65 | $92 | $82 |
| High | $78 | $70 | $85 | $120 | $107 |

### 5.2 Wind LCOE ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $55 | $30 | $47 | $61 | $55 |
| Medium | $73 | $40 | $62 | $81 | $73 |
| High | $95 | $52 | $81 | $105 | $95 |

### 5.3 Clean Firm LCOE ($/MWh) — Merit-Order Tranche Model (Step 3)

Clean firm cost uses a **merit-order supply curve** with two tranches, filled cheapest-first. The effective LCOE depends on how much clean firm a scenario requires — small amounts are cheap (all uprates), large amounts are expensive (hitting new-build tranche). This is a Step 3 cost calculation applied to the Step 2 efficient frontier.

#### Tranche 1: Nuclear Uprates (Cheapest, Capped)

**Uprate LCOE** (incremental cost of adding capacity to existing plants):
| Level | LCOE ($/MWh) | Basis |
|---|---|---|
| Low | $15 | MUR-dominated (measurement recapture, minimal capital) |
| Medium | $25 | Typical MUR + stretch blend |
| High | $40 | Stretch/small EPU with equipment replacement |

*Sources: INL LWRS Program, NRC uprate database, NEI fleet data, Thunder Said Energy capex analysis, IRA §45Y PTC*

**Uprate cap** — 8% of existing nuclear capacity (includes MUR + stretch + good EPU opportunities):

| Region | Existing Nuclear (GW) | Uprate Cap (GW) | Uprate Cap (TWh/yr @ 90% CF) |
|---|---|---|---|
| **CAISO** | 2.3 (Diablo Canyon) | 0.18 | 1.5 |
| **ERCOT** | 2.7 (South Texas Project) | 0.22 | 1.7 |
| **PJM** | 32.0 (largest US fleet) | 2.56 | 20.2 |
| **NYISO** | 3.4 (Nine Mile, FitzPatrick, Ginna) | 0.27 | 2.1 |
| **NEISO** | 3.5 (Millstone, Seabrook) | 0.28 | 2.2 |
| **Total** | **43.9** | **3.51** | **27.7** |

*8% chosen: NRC has approved ~8% fleet-wide historically (MUR + stretch + EPU). Good EPU opportunities remain across ~27 of 94 reactors, particularly BWR plants. DOE executive order targets ~3-5 GW; INL LWRS estimates 3-8% remaining. 8% reflects full remaining potential including EPU deployment at $15-40/MWh — the cheapest new dispatchable clean capacity available.*

#### Tranche 2: Geothermal (CAISO Only, Capped at 5 GW)

**CAISO only.** Geothermal fills before nuclear new-build, capped at 5 GW (~39 TWh/yr at 90% CF). Based on USGS identified hydrothermal resources (Salton Sea, Imperial Valley, The Geysers). Non-CAISO ISOs have zero geothermal potential for power generation (temperature gradients too low — see §5.4.3).

Geothermal LCOE controlled by **Geothermal Cost** toggle (CAISO only):

| Level | CAISO | Basis |
|---|---|---|
| Low | $63 | Mature hydrothermal flash (Lazard low-end, NREL ATB) |
| Medium | $88 | Blended hydrothermal flash + binary (NREL 2025 Market Report) |
| High | $110 | Binary plants + early EGS (NREL ATB conservative) |

*Sources: NREL ATB 2024, NREL 2025 US Geothermal Market Report, Lazard LCOE+ v18, USGS 2008 Assessment (FS 2008-3082), USGS 2025 Great Basin EGS Assessment.*

**Geothermal cap**: 5 GW = ~39 TWh/yr at 90% CF. Conservative bound using USGS identified hydrothermal only (excludes undiscovered and EGS). After geothermal cap is filled, remaining CAISO clean firm demand falls to Tranche 3 (nuclear new-build) or CCS, whichever is cheaper.

**Non-CAISO geothermal**: Zero. ERCOT has nascent EGS demos (Sage Geosystems) but no operating capacity. PJM/NYISO/NEISO have temperature gradients of 20-25°C/km — far below power generation thresholds. Toggle hidden/disabled for non-CAISO regions.

#### Tranche 3: Nuclear New-Build (Uncapped)

Nuclear new-build LCOE reflects advanced SMR/Gen IV technology. Controlled by **Firm Generation Cost** toggle. For CAISO, this tranche fills after geothermal cap is exhausted. For all other ISOs, this is the first new-build tranche after uprates.

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $70 | $68 | $72 | $75 | $73 |
| Medium | $95 | $90 | $105 | $110 | $108 |
| High | $140 | $135 | $160 | $170 | $165 |

*Low = nth-of-a-kind SMR deployment target ($70/MWh). Regional variation at Low is minimal (mature deployment compresses cost differences). Medium/High retain larger regional spreads reflecting siting, permitting, and labor differentials. ERCOT lowest (favorable siting/permitting). NYISO highest (siting constraints, labor costs).*

#### Merit-Order Cost Calculation (Step 3 Pipeline)

For each cached scenario's new clean firm demand (above existing grid share), the merit order fills cheapest-first. **CAISO has 4 tranches; other ISOs have 3.**

**Non-CAISO merit order:**
```
new_cf_twh = max(0, total_cf_pct - existing_cf_pct) / 100 × demand_twh
uprate_twh = min(new_cf_twh, uprate_cap_twh)
remaining = max(0, new_cf_twh - uprate_twh)
# Remaining filled by cheapest of: nuclear new-build vs CCS (toggle-dependent)
nuclear_price = NEWBUILD_LCOE[firm_level][iso] + tx_adder
ccs_price = CCS_LCOE[ccs_level][45q_state][iso] + tx_adder
# Each MWh goes to whichever is cheaper
```

**CAISO merit order (includes geothermal tranche):**
```
new_cf_twh = max(0, total_cf_pct - existing_cf_pct) / 100 × demand_twh
uprate_twh = min(new_cf_twh, uprate_cap_twh)
remaining_after_uprate = max(0, new_cf_twh - uprate_twh)
geo_twh = min(remaining_after_uprate, GEO_CAP_TWH)  # 39 TWh cap
remaining_after_geo = max(0, remaining_after_uprate - geo_twh)
# Remaining filled by cheapest of: nuclear new-build vs CCS (toggle-dependent)
```

At low clean firm demand → effective LCOE approaches uprate price ($25/MWh Medium).
At high clean firm demand → effective LCOE approaches new-build price ($88-110/MWh Medium).
The transition point (where uprate cap is exhausted) varies by region — PJM has the most uprate headroom.

**Replaces**: The previous fixed-blend model (§5.3 legacy: `uprate_share × uprate + (1-uprate_share) × new_build`) which applied the same effective LCOE regardless of quantity demanded. The tranche model makes clean firm cost quantity-dependent, which shifts optimal resource mixes at high thresholds.

#### Legacy Blended Values (Preserved for Reference)

Previous blended LCOE (still used in Step 1 physics optimization cache):
| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $58 | $56 | $48 | $64 | $69 |
| Medium | $79 | $79 | $68 | $86 | $92 |
| High | $115 | $115 | $108 | $136 | $143 |

*These are what the Step 1 optimizer used. Step 3 reprices using the tranche model above.*

### 5.4 CCS-CCGT LCOE ($/MWh) — Separate Toggle with 45Q Switch

CCS cost is controlled by two independent toggles: **CCS Cost** (L/M/H maturity) and **45Q Credit** (On/Off).

#### 5.4.1 CCS LCOE with 45Q ON ($/MWh)

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $58 | $52 | $62 | $78 | $75 |
| Medium | $86 | $71 | $79 | $99 | $96 |
| High | $115 | $92 | $102 | $128 | $122 |

#### 5.4.2 CCS LCOE with 45Q OFF ($/MWh)

45Q OFF = add back $29/MWh offset. Same underlying capex/transport/storage assumptions.

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $87 | $81 | $91 | $107 | $104 |
| Medium | $115 | $100 | $108 | $128 | $125 |
| High | $144 | $121 | $131 | $157 | $151 |

*ERCOT lowest (Gulf Coast Class VI wells, abundant geology, cheap gas, shortest CO2 transport). NYISO highest (no suitable sequestration geology, longest transport, highest permitting burden).*

**L/M/H maturity mapping**:
- **Low**: Mature nth-of-a-kind CCS, established CO₂ infrastructure, low capex
- **Medium**: Mid-range deployment maturity
- **High**: Immature/early deployment, first-of-a-kind, high capex

**CCS-CCGT cost buildup**:
- Capture cost: ~$30-40/MWh (technology-dependent, relatively uniform)
- CO2 transport: $2-20/MWh (regional — distance to Class VI well)
- CO2 storage: $5-15/MWh (regional — geology, well costs)
- Fuel cost: Heat rate × gas price (responds to gas toggle)
- 45Q offset (when ON): -$29/MWh ($85/ton × 0.34 tCO2/MWh × 95% capture)
- Capture rate: 95%
- Residual emissions: ~0.0185 tCO2/MWh (= 0.37 × 0.05)

**45Q behavioral note**: With 45Q ON, CCS modeled as flat baseload (45Q incentivizes max CF to maximize capture credits). With 45Q OFF, CCS dispatch assumption unchanged in Step 3 (same cached physics), but the cost premium reflects the absence of the policy subsidy.

#### 5.4.3 Regional CCS Capacity Caps (TWh/yr)

CCS-CCGT allocation is capped per ISO based on geologic CO₂ storage availability, infrastructure, and regulatory feasibility — identical pattern to the geothermal cap (`GEO_CAP_TWH = 39.0`) for CAISO. The cap is enforced in Step 3 cost optimization (merit-order tranche logic) and will be propagated to Step 1 mix filtering in the next physics run.

```python
CCS_CAP_TWH = {
    'CAISO': 25.0,    # 11% of 224 TWh demand
    'ERCOT': 200.0,   # 41% of 488 TWh demand
    'PJM':   125.0,   # 15% of 843 TWh demand
    'NYISO': 0.0,     # Hard zero — no geologic storage
    'NEISO': 0.0,     # Hard zero — no geologic storage
    'MISO':  200.0,   # 30% of 660 TWh demand
    'SPP':   50.0,    # 17% of 296 TWh demand
}
```

**Regional justification:**

- **NYISO (0 TWh)**: No suitable onshore CO₂ storage geology. Newark Rift Basin assessed as "low potential" by USGS/NETL. Offshore Atlantic (Baltimore Canyon Trough) is decades from permitting. Zero Class VI well applications filed, state not pursuing primacy.
- **NEISO (0 TWh)**: Crystalline and metamorphic bedrock — zero identified CO₂ storage units in the USGS National Carbon Sequestration Database. No saline formations or depleted reservoirs anywhere in New England. Additionally constrained by winter gas pipeline bottleneck (Step 4 adder).
- **CAISO (25 TWh / 11%)**: Excellent geology (San Joaquin Basin 14–56 Gt, Sacramento Basin ~3 Gt) but SB 905 imposes strictest CCS regulatory framework in US. Zero operating CCS projects, zero CO₂ pipeline infrastructure in-state.
- **SPP (50 TWh / 17%)**: Good geology (Anadarko Basin, Arbuckle Group — 780 Mt P50 in KS) but Oklahoma induced seismicity from underground injection creates regulatory/social resistance. State pursuing but has not received Class VI primacy.
- **PJM (125 TWh / 15%)**: Stark east-west split. Western PJM (WV/OH/western PA) sits on Appalachian Basin (450–500 Gt theoretical); WV received Class VI primacy Jan 2025. Eastern PJM (DC/MD/VA/NJ/DE — majority of demand) has unsuitable Piedmont/Coastal Plain geology. No CO₂ transport infrastructure connecting east to west.
- **ERCOT (200 TWh / 41%)**: Best CCS region in US. Gulf Coast has 20+ Gt depleted offshore fields, 100s Gt offshore saline formations. TX received Class VI primacy Dec 2025 (64 apps from EPA). Denbury CO₂ pipeline network (900+ mi) is densest in US. Multiple storage hubs under development.
- **MISO (200 TWh / 30%)**: Mt. Simon Sandstone (12–172 Gt) is the most characterized formation in US with 2+ Mt successfully injected at ADM Decatur. ND has had primacy since 2018 with 3 active projects. Broadwing 400 MW CCS-CCGT (Google-backed, FID Q2 2026) would be first in US.

**Implementation (Step 3, March 2026)**: CCS cap enforced in Tranche 3 within `price_mix_batch`:
1. **Implicit CCS residual** (`ccs_pct = 100 - sum(cf, sol, wnd, hyd)`): **NOT priced** — tracked for output only. The residual represents unmatched demand served by the existing grid, not a real CCS build decision. Previously this was priced at CCS LCOE which made low-threshold mixes artificially expensive.
2. **Tranche 3 CCS** (clean_firm overflow after uprate + geothermal): CCS headroom = full `CCS_CAP_TWH[iso]` (not reduced by residual since residual isn't built). If CCS is cheaper than nuclear but headroom exhausted, overflow goes to nuclear new-build.
For NYISO/NEISO (cap=0), all tranche 3 CCS → nuclear automatically.

*Sources: USGS National Carbon Sequestration Database (NATCARB), NETL Carbon Storage Atlas V (2015), DOE CarbonSAFE program status (2024–2025), EPA Class VI well permit tracker, California SB 905 (2022), Princeton Net-Zero America (2021), Global CCS Institute Status Report (2024), IEEFA CCS deployment analysis (2024).*

### 5.5 Battery Costs — NREL Component Model + Wright's Law Decline

**Updated March 2026.** Battery costs re-anchored to NREL ATB 2024 component model with Wright's Law learning curves for future cost decline.

**CAPEX derivation** — NREL ATB 2024 separates battery costs into energy ($/kWh) and power ($/kW) components. Total installed cost per kWh = Energy + Power/Duration. This gives the correct 4hr→8hr ratio (~14% cheaper for 8hr, because power electronics spread over 2× the energy capacity).

| Level | Energy ($/kWh) | Power ($/kW) | 4hr Total | 8hr Total | 8hr/4hr |
|---|---|---|---|---|---|
| Low | $170 | $280 | $240/kWh | $205/kWh | 85.4% |
| Medium | $210 | $340 | $295/kWh | $253/kWh | 85.6% |
| High | $270 | $420 | $375/kWh | $323/kWh | 86.0% |

*Low = aggressive LFP procurement + competitive BOS. Medium = typical US utility project (~$295/kWh vs NREL $334 benchmark — reflecting 2025 market reality below NREL's conservative bottom-up model). High = tariff-exposed, constrained interconnection.*

**Financial parameters**: WACC=8%, 20yr life, FOM=2.5% of CAPEX($/kW) per NREL (includes augmentation). Annualized = CAPEX × (CRF + 0.025) / 8760 × 1000 × regional_mult. Regional multipliers: ERCOT=1.00 (cheapest), CAISO=1.11, NYISO=1.18 (highest).

**Annualized capacity costs** ($/MWh-cap, 2025 starting values):

| Level | Type | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|---|---|---|---|---|---|---|---|---|
| Low | 4hr | 3.86 | 3.48 | 3.70 | 4.08 | 3.97 | 3.62 | 3.52 |
| Medium | 4hr | 4.75 | 4.27 | 4.55 | 5.02 | 4.87 | 4.46 | 4.33 |
| High | 4hr | 6.03 | 5.43 | 5.78 | 6.38 | 6.20 | 5.66 | 5.50 |
| Low | 8hr | 3.30 | 2.97 | 3.16 | 3.49 | 3.39 | 3.10 | 3.01 |
| Medium | 8hr | 4.06 | 3.66 | 3.89 | 4.30 | 4.17 | 3.81 | 3.70 |
| High | 8hr | 5.19 | 4.67 | 4.97 | 5.49 | 5.33 | 4.87 | 4.73 |

**LCOS cross-check** (4hr Medium ERCOT, 365 cycles, 85% RTE): **$121/MWh**. Consistent with Lazard 2024 ($115-220/MWh range).

**Wright's Law learning curves** — Battery costs decline from 2025 starting values toward terminal NOAK floor. This is the reverse direction from other technologies (which start at FOAK and decline to NOAK): batteries are already at manufacturing scale, so 2025 IS the starting point. Curves calibrated to NREL 2050 cost projections.

Terminal NOAK ($/kWh): Low=50%, Medium=56%, High=80% of 2025 starting cost.
- Low 4hr: $120/kWh by 2042 | Med 4hr: $165/kWh by 2048 | High 4hr: $300/kWh by 2050
- Low 8hr: $102/kWh by 2040 | Med 8hr: $141/kWh by 2046 | High 8hr: $258/kWh by 2050

Learning curve exponent: 0.6 (concave ramp — steeper initially, asymptotic approach). 8hr reaches NOAK ~2yr faster than 4hr because cell costs (which decline faster) are a larger share of 8hr total cost.

**Trajectory (4hr Medium ERCOT):**
| Year | Wright's fraction | CAPEX | Annualized | LCOS (365 cyc) |
|---|---|---|---|---|
| 2025 | 0.00 | $295/kWh | $4.27/MWh-cap | $121/MWh |
| 2030 | 0.40 | $243/kWh | $3.52/MWh-cap | $99/MWh |
| 2035 | 0.61 | $216/kWh | $3.13/MWh-cap | $88/MWh |
| 2040 | 0.77 | $194/kWh | $2.82/MWh-cap | $79/MWh |
| 2048+ | 1.00 | $165/kWh | $2.39/MWh-cap | $67/MWh |

*Sources: [NREL ATB 2024](https://atb.nrel.gov/electricity/2024/utility-scale_battery_storage), [NREL Cost Projections 2025 Update](https://docs.nrel.gov/docs/fy25osti/93281.pdf), [Ember Battery Storage Costs](https://ember-energy.org/latest-insights/how-cheap-is-battery-storage/), Wright's Law learning rate literature.*

### 5.6 LDES LCOS ($/MWh, 100hr iron-air) — Regionalized

| Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Low | $135 | $116 | $128 | $150 | $143 |
| Medium | $180 | $155 | $170 | $200 | $190 |
| High | $234 | $202 | $221 | $260 | $247 |

*ERCOT lowest (Gulf Coast geology for compressed air variants, low labor). NYISO highest (expensive labor, constrained siting, limited geology).*

### 5.7 Transmission Adders ($/MWh, new-build only)

| Resource | Level | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|---|
| **Wind** | Low | $4 | $3 | $5 | $7 | $6 |
| | Medium | $8 | $6 | $10 | $14 | $12 |
| | High | $14 | $10 | $18 | $22 | $20 |
| **Solar** | Low | $1 | $1 | $2 | $3 | $3 |
| | Medium | $3 | $3 | $5 | $7 | $6 |
| | High | $6 | $5 | $9 | $12 | $10 |
| **Clean Firm** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $3 | $2 | $3 | $5 | $4 |
| | High | $6 | $4 | $6 | $9 | $7 |
| **CCS-CCGT** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $2 | $2 | $3 | $4 | $3 |
| | High | $4 | $3 | $5 | $7 | $6 |
| **Battery** | Low | $0 | $0 | $0 | $1 | $1 |
| | Medium | $1 | $1 | $1 | $2 | $2 |
| | High | $2 | $2 | $3 | $4 | $3 |
| **LDES** | Low | $1 | $1 | $1 | $2 | $2 |
| | Medium | $2 | $2 | $3 | $4 | $3 |
| | High | $4 | $3 | $5 | $7 | $6 |
| **Hydro** | All | $0 | $0 | $0 | $0 | $0 |

*ERCOT lowest (CREZ buildout, less congestion). NYISO highest (constrained corridors, siting opposition). Sources: LBNL "Queued Up", MISO/SPP interconnection data.*

### 5.8 Fuel Prices

| Fuel | Low | Medium | High |
|---|---|---|---|
| Natural Gas | $2.00/MMBtu | $3.50/MMBtu | $6.00/MMBtu |
| Coal | $1.80/MMBtu | $2.50/MMBtu | $4.00/MMBtu |
| Oil | $55/bbl | $75/bbl | $110/bbl |

### 5.9 Fuel Price → Wholesale + Emission Rate Impact

**Wholesale**: Shifts based on regional 2025 fossil fuel mix composition. Uses **hourly wholesale price profiles** from EIA 2025 data (not flat averages).

**Wholesale fuel price adjustments** ($/MWh adder to base wholesale, by fossil fuel toggle level):

| Region | Low | Medium | High | Rationale |
|--------|-----|--------|------|-----------|
| CAISO  | -5  |   0    | +10  | ~40% gas generation |
| ERCOT  | -7  |   0    | +12  | ~50% gas, most sensitive to fuel prices |
| PJM    | -6  |   0    | +11  | ~40% gas + coal mix |
| NYISO  | -4  |   0    |  +8  | ~35% gas, more nuclear insulates from fuel |
| NEISO  | -4  |   0    |  +8  | ~35% gas, more nuclear insulates from fuel |

**Emission rate — Regional fuel-switching elasticity**:

| Region | Coal Fleet Status | Switching Elasticity | Rationale |
|---|---|---|---|
| ERCOT | Largely retired (~10GW remaining) | **Low** | Limited coal to switch to; gas price barely shifts emission rate |
| PJM | Substantial remaining (~45GW) | **High** | Gas price ↑ drives meaningful coal resurgence, emission rate jumps |
| CAISO | Near zero | **Very low** | Almost no coal option |
| NYISO | Minimal | **Low** | Small effect |
| NEISO | Minimal (retiring) | **Low** | Small effect |

---

## 6. Storage Algorithms

### 6.1 Battery (4hr Li-ion) — CAPACITY-CONSTRAINED dispatch

**Key principle**: Cost comes from capacity built. LCOS must reflect actual utilization — can't have huge redundant capacity that's barely used. The optimizer co-optimizes capacity size and dispatch.

1. `battery_dispatch_pct` maps to a **capacity** (MWh) and **power rating** (MW = capacity / 4hr)
2. Each day: charge from surplus hours up to min(available surplus, capacity), discharge to gap hours up to min(stored energy × 85% RTE, capacity)
3. Days with insufficient surplus → partial cycle → less dispatch that day
4. Annual MWh dispatched = sum of actual daily dispatches (variable, not uniform)
5. **Utilization factor** = actual annual cycles / 365 theoretical max cycles
6. **LCOS** = annualized capital cost of built capacity / actual MWh dispatched — underutilized capacity drives LCOS up, creating a natural cost penalty for oversizing
7. Optimizer finds the sweet spot: enough capacity to be useful at the target threshold, not so much that idle capacity inflates cost

### 6.2 LDES (100hr iron-air) — CAPACITY-CONSTRAINED dispatch with dynamic sizing

**Same capacity-constrained principle as battery.**

1. `ldes_dispatch_pct` maps to a **capacity** (MWh) that scales dynamically (not fixed at 1 day of demand) and **power rating** (MW = capacity / 100hr)
2. **Rolling 7-day window**: identify sustained multi-day surplus periods (spring wind, long sunny stretches) and deficit periods (winter evening doldrums, cloudy windless stretches)
3. Charge during surplus periods up to min(available surplus, power rating), respecting energy capacity
4. Discharge during deficit periods up to min(stored energy × 50% RTE, power rating)
5. State of charge carries over between windows
6. **Utilization factor** = actual annual energy throughput / (capacity × theoretical max cycles)
7. **LCOS** = annualized capital cost of built capacity / actual MWh dispatched — same utilization penalty as battery
8. Seasonal shifting: captures week-to-week and seasonal patterns batteries cannot

### 6.3 Storage Grid Refinement — Sub-Percent Granularity (Decision: Feb 21, 2026)

**Problem identified**: The original storage sweep grid `[0, 2, 5, 8, 10, 15, 20]` (% of annual demand) had a blind spot. Battery4 and Battery8 max SOC never exceeds ~1.0% of annual demand even under peak-stress conditions (high RE, low CF, max procurement, >90% targets). The jump from 0% → 2% skipped the entire range where batteries actually saturate, meaning the cost optimizer never tested right-sized battery configurations. This systematically overpriced storage (paying for 4-20× idle capacity) and biased the optimizer toward avoiding batteries when properly-sized batteries could be cost-competitive.

**Empirical saturation thresholds** (max SOC as % of annual demand, unconstrained capacity, high-RE stress mixes at 97.5-99% targets):

| ISO | Bat4 (4hr) 90% Sat | Bat8 (8hr) 90% Sat | LDES (100hr) |
|-----|---------------------|---------------------|--------------|
| CAISO | 0.577% | 0.577% | >50% (always saturated) |
| ERCOT | 0.663% | 0.663% | >50% |
| PJM | **1.155%** | **1.155%** | >50% |
| NYISO | 0.922% | 0.922% | >50% |
| NEISO | 0.975% | 0.975% | >50% |

**Root cause**: Battery daily surplus/gap is small relative to annual demand (~0.5% of annual demand on peak days). The 4hr/8hr durations provide sufficient power headroom that power rating never binds — only energy capacity matters. PJM is the binding case due to high-wind mixes at 99% creating larger daily swings.

**LDES is fundamentally different**: Multi-day accumulation over 7-day windows means LDES fills to capacity even at 20% of annual demand. LDES is capacity-hungry through the entire tested range. Fine granularity for LDES is about optimizing the marginal cost/benefit tradeoff, not finding saturation.

**Refined storage grids** — 0.1% intervals below max saturation, then coarser above:

```python
# Bat4: 0.1% intervals to 1.5% (covers PJM binding case + margin), then coarser
batt_levels  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 5, 10, 15, 20]  # 20 levels

# Bat8: identical physics (same surplus, duration only affects power limit which never binds)
batt8_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 5, 10, 15, 20]  # 20 levels

# LDES: 0.5% intervals to 2.5% (marginal value optimization), then coarser
ldes_levels  = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10, 15, 20]  # 11 levels
```

**Combo count**: 20 × 20 × 11 = **4,400** storage combos per mix (vs. old 7³ = 343 → **12.8×** increase).

**Implementation notes**:
- Storage levels change from integer to float → parquet schema uses `pa.float64()` for `battery_dispatch_pct`, `battery8_dispatch_pct`, `ldes_dispatch_pct`
- All downstream code (Step 2 EF, Step 3 cost, Step 4 postprocess, dashboard) must handle float storage values
- Existing cache with integer storage values must be preserved and converted on merge

### 6.4 Storage Sweep Optimizations — Batched Dispatch + Parallel Mixes (Feb 21, 2026)

The 12.8× increase in nominal storage combos is offset by a batched + parallel architecture:

#### 1. Batched Storage Dispatch (`_batch_storage_scores`)

Instead of calling `_score_with_all_storage` per combo (each recomputing bat4/bat8 dispatch from scratch), a single `_batch_storage_scores` call evaluates ALL bat4×bat8×LDES combos for a given (mix, procurement):

- **Bat4 dispatch residual reuse**: Computed once per bat4 level, reused across all bat8 levels
- **Bat8 dispatch residual reuse**: Computed once per (bat4, bat8) pair, reused across all LDES levels
- **LDES dispatch**: Only the innermost loop; runs on post-bat4+bat8 residual

This eliminates the redundant base dispatch recomputation that dominated the old triple-nested loop.

#### 2. Parallel Mix Screening (`_batch_mixes_storage_screen`)

Near-miss mixes are processed in batches of `MAX_MIX_BATCH = 100`. Each batch:
1. Pre-computes curtailment-MW caps for all mixes (fast: one 8760-hour pass per mix)
2. Gathers supply rows into a (N_batch, 8760) array
3. Calls `_batch_mixes_storage_screen` which uses **Numba `prange`** to distribute mixes across CPU cores
4. Each core runs `_batch_storage_scores` for its assigned mixes in parallel

This prevents large ISOs (NYISO with hydro_cap=15.9%, PJM) from stalling by parallelizing across mixes.

#### 3. Energy-Based Storage Cap (Per-Mix Physics Ceiling)

For each (mix, max_procurement), `_compute_storage_caps` computes the maximum surplus energy that could charge each storage type over its operational window:

- `bat4_cap = max_daily_surplus` — max energy surplus in any single day (4hr daily-cycle battery can't charge more than available daily curtailment)
- `bat8_cap = max_2day_surplus` — max energy surplus in any 2-day window (8hr battery uses 48hr dispatch window)
- `ldes_cap = max_7day_surplus` — max energy surplus in any 7-day window (100hr iron-air)

This is an **energy-based** ceiling, not power-based. A 4hr battery at 200% pure solar saturates at ~0.3% of annual demand capacity because the discharge-side gap (nighttime hours) limits useful capacity, not the charge-side surplus. The 0.3% capacity cycles 365×/year, delivering ~57% of annual demand throughput at 61% utilization.

Levels above the per-mix cap are auto-skipped in the storage sweep.

#### 4. Curtailment Frequency Filter

Daily-cycle batteries (bat4/bat8) need **≥ 150 surplus days** to justify capacity. Mixes with fewer surplus days skip battery combos entirely; only LDES is evaluated (which accumulates across multi-day windows).

### 6.5 Step 1D Storage Refinement Module (Decision: Mar 1, 2026)

**Problem**: Step 1C's coarse storage levels at <95% thresholds ([0,1,3] for bat4, [0,2,4] for bat8, [0,5,10] for LDES as % of annual demand) are too wide for the physical storage caps. Typical caps are bat4=0.2–0.5%, bat8=0.5–1.0%, LDES=1.0–3.0%. The first non-zero coarse level already exceeds the cap for most mixes, so the cap filtering skips ALL non-zero levels — effectively never exploring storage at <95% thresholds.

**Solution**: `step1d_storage_refinement.py` — a standalone module that reads the Step 1C coarse cache to identify candidate mixes and evaluates intermediate storage levels. No rerun of Steps 1A–1C.

**Storage levels for 65–92.5% thresholds** (full intermediate sweep):
```python
bat4:  [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # 14 levels
bat8:  [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]                 # 12 levels
LDES:  [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0, 10.0]          # 13 levels
H2:    [0]                                                                                # 1 level
# Total: 14 × 12 × 13 × 1 = 2,184 storage combos per mix
```

**Storage levels for ≥95% thresholds** (LDES intermediates only):
```python
bat4:  [0, 1, 3, 5]           # same as 1C (caps are larger at high procurement)
bat8:  [0, 2, 4, 6]           # same as 1C
LDES:  [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]  # 13 levels (fills 0→5 gap)
H2:    [0, 5, 10, 20]         # same as 1C
# Total: 4 × 4 × 13 × 4 = 832 storage combos per mix
```

**Algorithm**:
1. Load coarse cache (no-storage scores) from Step 1B
2. For each threshold ≥65%: identify near-miss mixes (score >= max(target − 0.40, 0.50) AND score < target)
3. Compute physical storage caps per mix via `_batch_compute_storage_caps` (Numba parallel)
4. Batch-evaluate all storage combos via `_batch_mixes_storage_screen` (100 mixes/batch, Numba prange)
5. Cap filtering: skip levels exceeding per-mix physical cap (same as 1C §6.4.3)
6. Curtailment filter: batteries need ≥150 surplus days (same as 1C §6.4.4)
7. Output feasible solutions (score ≥ target AND at least one storage > 0) to new parquets

**Output**: `data/step1-pfs/{ISO}_t{XX}_storage_refined.parquet` — same schema as Step 1C PFS parquets, `pareto_type = 'storage_refined'`.

**Step 2 integration**: `step2_efficient_frontier.py` scans `data/step1-pfs/`. Deduplication handles any overlap between 1C and 1D results (keeps max score per unique resource + storage key).

**Test results** (ERCOT t75): 3,303,625 new storage-enabled solutions found in 78s (vs 0 from Step 1C). Cap ranges confirmed: bat4=[0.00%, 0.34%, 0.80%], LDES=[0.00%, 2.10%, 5.15%].

#### 5. Bat8 Two-Day Dispatch Window

Battery 8hr dispatch uses 48hr (2-day) windows instead of daily 24hr windows, reflecting ~200 cycles/year operational pattern. This allows accumulating surplus across 2 days before discharging.

```python
batt8_window = 48  # 2-day window → ~183 windows/year → ~200 actual cycles
```

#### 6. Dominance Pruning (LDES Early Stop)

Within each (bat4, bat8) pair's LDES sweep, if increasing LDES doesn't reduce minimum feasible procurement, stop — higher LDES adds cost without reducing procurement.

#### 7. Infeasible Screening

The batch call at max procurement screens ALL combos in one shot. Combos infeasible at max procurement are skipped without binary search (biggest win — most combos are infeasible).

**Net architecture**: `_batch_mixes_storage_screen` (Numba prange across mixes) → `_batch_storage_scores` (residual reuse across storage combos) → post-filter by per-mix caps → binary search only feasible combos.

---

## 7. CO2 & Abatement

### 7.1 CO2 Emissions Abated — Dispatch-Stack Retirement Model

**Core assumption (Decision: Feb 19, 2026)**: As clean energy percentage grows, fossil fuels retire in merit order — coal first (dirtiest, most expensive), then oil, then gas. **Above 70% clean, all coal and oil have retired; only gas CCGT + clean remains.** This replaces the previous uniform hourly fossil mix model where coal/gas/oil shares were constant regardless of clean energy percentage.

**Validated by regional data**: Coal exhausts well before 70% clean in every ISO:
- CAISO: 0.0% coal (already gone at any threshold)
- ERCOT: 13.9% coal → exhausted at ~60% clean
- PJM: 16.5% coal → exhausted at ~57% clean
- NYISO: 0.0% coal (already gone)
- NEISO: 0.3% coal → exhausted at ~34% clean

**Merit-order retirement stack** (per ISO):
1. **Coal retires first** — highest emitter (~1.0-1.05 tCO₂/MWh). As clean % grows from baseline, each additional MWh of clean displaces coal until the regional coal fleet is fully retired.
2. **Oil retires second** — mid emitter (~0.82-1.31 tCO₂/MWh). After coal is gone, clean MWh displace oil. Oil shares are tiny (<1.1% of total gen in all ISOs), so this band is narrow.
3. **Gas retires last** — lowest fossil emitter (~0.38-0.41 tCO₂/MWh). Once coal and oil are gone (at or before 70% clean), all remaining fossil is gas CCGT. Every additional MWh of clean energy above this point displaces gas only.

**Calculation for a given clean energy threshold T%**:
```
baseline_clean = sum of existing clean shares (GRID_MIX_SHARES)
fossil_total = 100% - baseline_clean
coal_total = coal_share_of_fossil × fossil_total  (from EIA fossil mix data)
oil_total = oil_share_of_fossil × fossil_total
gas_total = gas_share_of_fossil × fossil_total

additional_clean = T% - baseline_clean  (new clean energy added)

# Merit-order displacement:
coal_displaced = min(additional_clean, coal_total)
remaining = additional_clean - coal_displaced
oil_displaced = min(remaining, oil_total)
remaining = remaining - oil_displaced
gas_displaced = min(remaining, gas_total)

# Emission rate of remaining fossil fleet at threshold T:
coal_remaining = coal_total - coal_displaced
oil_remaining = oil_total - oil_displaced
gas_remaining = gas_total - gas_displaced
fossil_remaining = coal_remaining + oil_remaining + gas_remaining

if fossil_remaining > 0:
    emission_rate = (coal_remaining × coal_rate + oil_remaining × oil_rate + gas_remaining × gas_rate) / fossil_remaining
else:
    emission_rate = 0  (100% clean)
```

**Above 70% clean**: Forced to gas-only emission rate regardless of stack calculation (simplifying assumption). `emission_rate = gas_rate` (~0.39 tCO₂/MWh). Fuel-switching elasticity (Section 5.9) is zeroed out above 70% — no coal to switch with.

**Per-fuel emission rates** (from eGRID 2023, static per region):
- `coal_rate[iso]` = eGRID coal CO₂ lb/MWh (e.g., ERCOT: 2325, PJM: 2216)
- `gas_rate[iso]` = eGRID gas CO₂ lb/MWh (e.g., ERCOT: 867, PJM: 867)
- `oil_rate[iso]` = eGRID oil CO₂ lb/MWh (e.g., ERCOT: 2894, PJM: 1919)

**CO₂ abated** (hourly resolution):
- For each hour h: `fossil_displaced[h] = clean_supply[h] − max(0, clean_supply[h] − demand[h])`
- `CO₂_abated = Σ_h fossil_displaced[h] × emission_rate_at_threshold`
- The emission rate is threshold-dependent (not hourly-variable anymore): at a given clean %, the fossil fleet composition is fixed by the retirement stack
- CCS-CCGT gets **partial credit**: 90% capture → residual ~0.037 tCO₂/MWh (vs ~0.39 unabated CCGT)

**Storage CO₂ attribution** (hourly dispatch tracking):
- Track exact hours each storage type (battery/LDES) dispatches into → use threshold-level emission rate for abatement credit
- Storage charging from surplus clean energy → charge emissions = 0
- Storage charging during hours when fossil is still marginal → charge has real emissions that reduce net abatement

**Impact vs previous model**:
- **Low thresholds (50-70%)**: Higher CO₂ abatement — first MWh of clean displaces coal (~1.0 tCO₂/MWh), not a blended average (~0.5 tCO₂/MWh)
- **High thresholds (>70%)**: Lower marginal CO₂ abatement — displacing gas only (~0.39 tCO₂/MWh), not a blended average
- **MAC at high thresholds increases** — same cost but less CO₂ per MWh displaced
- Fuel-switching elasticity irrelevant above 70% (no coal/oil to switch)

**Why this matters**: The previous uniform model assumed the fossil fleet composition stays constant as clean energy grows. In reality, coal plants are the first to retire (most expensive, most regulated, dirtiest). The dispatch-stack model correctly captures decreasing marginal emission reductions as the grid gets cleaner — the "easy" high-emission tons are abated first, and the last tons (displacing efficient gas) are the hardest.

**Absolute coal/oil caps — no new fossil build (Decision: Feb 19, 2026)**:

No new coal or oil capacity is built. Coal and oil generation are capped at their 2025 absolute TWh levels. As demand grows, only gas CCGT fills the gap — so coal/oil's share of total generation naturally declines, and the average fossil emission rate trends toward gas-only.

2025 caps (from EIA hourly data):

| ISO | Coal TWh | Oil TWh | Gas TWh | Coal Peak MW | Oil Peak MW |
|-----|----------|---------|---------|-------------|-------------|
| CAISO | 0.00 | 0.60 | 114.8 | 15 | 470 |
| ERCOT | 67.58 | 0.00 | 195.5 | 14,379 | 0 |
| PJM | 139.09 | 4.59 | 357.3 | 29,861 | 5,608 |
| NYISO | 0.00 | 0.15 | 92.3 | 0 | 1,948 |
| NEISO | 0.31 | 1.29 | 75.1 | 653 | 6,554 |

Effect: At 2025 base demand, caps equal actual generation (no change). Under demand growth scenarios, fossil fleet composition shifts:
```
grown_demand_twh = base_demand_twh × (1 + annual_rate)^(target_year − 2025)
grown_fossil_twh = grown_demand_twh × (1 − clean_pct/100)
coal_twh = min(COAL_CAP_TWH[iso], coal_cap)  # capped at 2025 level
oil_twh = min(OIL_CAP_TWH[iso], oil_cap)    # capped at 2025 level
gas_twh = grown_fossil_twh − coal_twh − oil_twh  # gas absorbs all growth
```
This means the merit-order retirement stack uses absolute TWh internally, not fixed percentages. PJM's 139 TWh of coal stays at 139 TWh even if demand doubles — its share of fossil drops from 28% to ~16%, pulling the average fossil rate toward gas.

**Data sources**:
- `data/egrid_emission_rates.json` — 2023 eGRID per-fuel CO₂ rates (lb/MWh) by region
- `data/eia_fossil_mix.json` — EIA hourly fossil fuel mix shares (coal/gas/oil) by ISO

**Implementation note**: CO₂ calculation is post-hoc (doesn't affect cost/matching optimization). The optimizer's resource mix and cost results are unaffected. CO₂ values can be recomputed on cached results.

**Bug fix (2026-02-16)**: The optimizer was applying marginal fossil emission rates to ALL storage charging hours, including hours with clean surplus (curtailment). Since storage in this model only charges from surplus clean energy, this incorrectly inflated charge emissions to ~21M tons (ERCOT 92.5%), making storage appear CO₂-neutral or negative. Fix: `charge_emission_rate = np.where(surplus > 0, 0.0, hourly_rates)` — zero rate when curtailment is occurring, marginal fossil rate otherwise. Post-processed `overprocure_results.json` and updated `MAC_DATA` in `shared-data.js`. CAISO MAC at 90% dropped from $122 to $98/ton; other regions with storage deployment similarly affected.

### 7.2 Demand Growth Counterfactual — New Gas at 350 kg/MWh (Decision: Feb 19, 2026)

**Problem**: Current CO₂ abatement only counts displaced existing grid emissions. But demand growth MWh that aren't served by clean energy would be met by new gas-fired generation. The counterfactual is that without clean procurement, those MWh produce emissions at **350 kg/MWh (0.35 tCO₂/MWh)** — the emission rate of a new CCGT.

**Formula**:
```
growth_mwh = base_demand × ((1 + annual_growth_rate)^(target_year − 2025) − 1) × 1,000,000
counterfactual_growth_emissions = growth_mwh × 0.35
total_co2_abated = existing_grid_displacement + counterfactual_growth_emissions
```

**Implementation**: Add growth counterfactual to `step5_compute_co2.py`. Growth rates from `step3_cost_optimization.py` DEMAND_GROWTH_RATES (CAISO 1.4–2.5%, ERCOT 2.0–5.5%, PJM 1.5–3.6%, NYISO 1.3–4.4%, NEISO 0.9–2.9%). New gas rate is 350 kg/MWh (representative CCGT heat rate ~6,400 BTU/kWh, pipeline gas). This is a post-hoc calculation — doesn't change resource mix or cost optimization.

### 7.3 SBTi Timeline-Indexed DAC Learning Curve (Decision: Feb 19, 2026)

**Approach**: Piecewise linear DAC cost projections from literature anchor points, overlaid on abatement charts where x-axis maps both clean energy threshold AND SBTi target year.

**SBTi Threshold-to-Year Mapping**:
| Year | SBTi Requirement | Optimizer Threshold |
|------|------------------|--------------------|
| 2025 | (today)          | Baseline           |
| 2030 | 50% hourly       | 50%                |
| 2035 | ~70% (interpolated) | 70%             |
| 2040 | 90% hourly       | 90%                |
| 2045 | ~95% (interpolated) | 95%             |
| 2050 | 100% (net-zero)  | 100%               |

**DAC Cost Trajectories ($/ton CO₂, net DACCS)** — *Revised Feb 27, 2026*:

| Year | Optimistic | Central | Conservative |
|------|-----------|---------|-------------|
| 2025 | $600      | $800    | $1,100      |
| 2030 | $350      | $500    | $750        |
| 2035 | $230      | $375    | $550        |
| 2040 | $175      | $300    | $450        |
| 2045 | $130      | $250    | $375        |
| 2050 | $100      | $200    | $300        |

**Revision rationale (Feb 27, 2026)**: Previous trajectories were too aggressive, particularly the optimistic scenario ($400 in 2025, $200 by 2030). Actual 2025 DAC costs are $600–$1,500/tCO₂ (Climeworks ~$1,000/ton, market average ~$600–$1,500 with subsidies). No credible source projects sub-$300 by 2030. Revised trajectories are anchored to:
- **2025 actuals**: Climeworks operational costs, CDR marketplace prices
- **2030–2035**: IEAGHG NOAK estimates ($194–$230 at 1 MtCO₂/yr, achievable "by as early as 2035"), Belfer Center projections ($400–$1,000 by 2030)
- **2040–2050**: Sievert et al. (Joule 2024) learning curves ($341/tCO₂ central at Gt scale), Climeworks roadmap (well below $500 by ~2040, $200–$250 towards 2050)

**Sources**: Climeworks (2024/2025 operational data), Sievert et al. (Joule 2024), IEAGHG (2021/2024), Belfer Center/Harvard (2023), DOE Liftoff (2023), IEA DAC (2022/2024), Fasihi et al. (J. Cleaner Prod. 2019), DOE Carbon Negative Shot, NAS (2019), Young et al. (One Earth 2023), Keith et al. (Joule 2018).

**Key assumptions by trajectory**:
- **Optimistic**: IEAGHG NOAK costs by 2035, strong learning rates, low-cost renewable energy, GtCO₂/yr scale by 2050
- **Central**: Belfer Center mid-range, Climeworks roadmap trajectory, moderate policy support, 100–500 MtCO₂/yr by 2050
- **Conservative**: Slow scale-up, limited policy, high energy costs, <100 MtCO₂/yr by 2050

**Visualization**: Abatement charts get dual x-axis (threshold % bottom, SBTi year top). DAC trajectory shown as 3 declining curves with shaded band. MAC curve intersections with DAC curves show the crossover points where grid decarbonization becomes more/less expensive than DAC at each milestone year.

All values are 2024 USD, net tons CO₂ removed (accounting for 5–12% lifecycle emissions). Full DACCS (capture + transport + storage + MRV).

### 7.2b Abatement Cost Curves (2 new charts)
- **Average Cost of Abatement**: Total incremental cost / Total CO2 abated = **$/ton CO2**
- **Marginal Cost of Abatement**: (Cost_{X+1%} − Cost_{X%}) / (CO2_{X+1%} − CO2_{X%}) = **$/ton CO2**
- **X-axis**: 75% to 100%, **linear numeric scale** (proportional spacing — distance from 85→90 equals 75→80)
- Both curves respond dynamically to **all 10 sensitivity toggles**
- 1% intervals from 85% provide smooth curve in the inflection zone
- Marginal curve shows hockey-stick shape: cheap early tons, expensive last tons

### 7.4 Optimal CFE Target per ISO — MAC × DAC Crossover (Decision: Feb 26, 2026)

**Goal**: For each ISO, identify the CFE threshold range where marginal grid decarbonization cost exceeds DAC — the "optimal target" beyond which buying offsets is cheaper than building more clean energy.

**Why stepwise MAC failed**: The existing stepwise MAC (Δcost/ΔCO₂ between adjacent thresholds) is wildly non-monotonic because:
1. Each threshold is independently optimized — the portfolio at 90% isn't built incrementally from the 87.5% portfolio
2. Coal retirement cliff at 70% causes a regime change in the CO₂ denominator
3. Fine threshold spacing (2.5% steps) amplifies small-denominator noise

**Solution — Option B: Smooth Marginal MAC from Cost Frontier**:
1. At each threshold, take the independently-optimized cheapest system (from Step 3)
2. Total cost premium ($M/yr = (system_cost - wholesale) × demand) and total CO₂ abated (Mt) form curves vs. threshold
3. Apply isotonic regression to enforce monotonicity (cost and CO₂ must be non-decreasing with threshold)
4. Fit monotone cubic splines (PCHIP) to the corrected curves
5. Marginal MAC = d(TotalCost)/d(CO₂) — the derivative of cost w.r.t. CO₂ along the spline
6. Cross with DAC cost trajectories to find crossover thresholds

**Crossover Range**: 3 grid cost tiers (L/M/H) × 3 DAC scenarios (optimistic/central/conservative) = 9 crossover points. The range = [min crossover, max crossover] across all 9 combinations. This captures: "between X% if DAC costs are low and clean energy costs are high, and Y% if DAC costs are high and clean energy costs are low."

**Option A: Target-Specific Analysis Within the Range**:
For each discrete threshold inside the crossover range (±1 step for context):
- Resource mix composition, system cost, total investment
- Comparison to DAC at the corresponding SBTi year
- Shows WHAT changes in the system as you stretch toward higher targets

**Demand Growth (L/M/H)**:
- Annual growth rates per ISO: CAISO 1.4/1.9/2.5%, ERCOT 2.0/3.5/5.5%, PJM 1.5/2.4/3.6%, NYISO 1.3/2.0/4.4%, NEISO 0.9/1.8/2.9%, MISO/SPP 2.0% (uniform)
- **Key finding**: Marginal MAC ($/tCO₂) is scale-invariant w.r.t. demand growth — both d(cost) and d(CO₂) scale by the same growth factor, so the ratio is unchanged. The crossover threshold % is the same regardless of demand growth.
- Demand growth DOES affect: total investment $M, total CO₂ abated, absolute resource quantities (TWh/GW). These are critical for the no-regrets analysis.

**No-Regrets Resource Investment Analysis**:
Within the crossover range, some resource investments are needed regardless of where the optimal target lands:
- **Floor**: minimum % share of each resource across all thresholds in the range — the absolute minimum you'd build regardless
- **Consensus**: resources that are non-zero at every threshold in the range — they show up across the board
- **Average**: expected investment level across the range
- All three scaled by L/M/H demand growth for absolute TWh quantities

**Implementation**: `scripts/step5_compute_optimal_targets.py` (Step 6 post-processor, runs in parallel with MAC/LMP/etc.)
- Outputs: `data/step4-analysis/optimal_targets.json`, `dashboard/js/optimal-target-data.js`
- Consumed by: `step6_generate_shared_data.py` → OPTIMAL_TARGETS in shared-data.js
- Depends on: CLEAN_COST (L/M/H effective_cost, no gas backup), RESOURCE_MIX_DATA, emission rates, DAC trajectories
- No dispatch cache dependency — uses pre-computed Step 3 cost data

#### 7.4.1 Scenario Comparison MAC — PCHIP Spline Smoothing (Decision: Feb 28, 2026)

**Problem**: The scenario comparison page (`step6_scenario_comparison.py`) computes per-threshold stepwise MAC as `Δnew_build_cost / ΔCO₂_abated` between adjacent thresholds. Because each threshold is independently optimized, the marginal cost bounces wildly (e.g., CAISO: null, 278, 278, 278, 12560, 9999, 232, 1421, 140, 9999...). This produces an unreadable, non-monotonic MAC curve instead of the expected hockey-stick shape.

**Solution — PCHIP Spline + Isotonic Regression** (two-pass approach):
1. **Pass 1**: Collect cumulative `(CO₂_abated, new_build_cost)` data points at each threshold per ISO from dispatch-cache CO₂ and `new_build_cost_total`.
2. **Pass 2**: Fit a PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) monotone spline to the cumulative supply curve `cost = f(CO₂)`.
3. Take the PCHIP derivative at each data point → raw marginal MAC = d(cost)/d(CO₂).
4. Apply `scipy.optimize.isotonic_regression` to enforce non-decreasing marginal cost (isotonic constraint).
5. Replace raw stepwise MAC values with smoothed values in the trajectory output.

**Why PCHIP**: PCHIP preserves monotonicity of the interpolant and avoids the Runge phenomenon (wild oscillations) that plague polynomial/cubic spline fits. It produces smooth curves through the data while respecting the natural convexity of the abatement supply curve.

**Why isotonic regression**: Even after PCHIP smoothing, numerical derivatives can produce minor non-monotonicity at certain data points. Isotonic regression is the minimum-perturbation projection onto the non-decreasing constraint set — it enforces the hockey-stick shape without artificially inflating values.

**Result**: Smooth, monotonically non-decreasing marginal MAC curve that starts low (~$50-300/t) for easy decarbonization and rises steeply at high thresholds (>95%) — the expected hockey-stick shape.

**Files changed**: `scripts/step6_scenario_comparison.py` — `_build_trajectory()` function rewritten with two-pass PCHIP approach. Added `scipy.interpolate.PchipInterpolator` and `scipy.optimize.isotonic_regression` imports. No changes to downstream consumers — `stepwise_mac` field in trajectory output dicts remains the same interface.

**Regeneration required**: Run Step 6 workflow with `step6_scenario_comparison` to regenerate `dashboard/js/scenario-comparison-data.js`.

---

#### 7.4.2 Gas Cost Separation in MAC (Decision: Feb 26, 2026)

**Decision**: MAC calculations use `effective_cost` (clean procurement only). Gas backup capacity cost is excluded from MAC because it is a system reliability cost, not an abatement cost.

**Rationale**:
- The MAC answers: "how much does it cost to abate one more ton of CO₂ via clean energy procurement?"
- Gas backup capacity is needed for grid reliability regardless of CFE target — it keeps the lights on
- Including gas backup in the MAC conflates the abatement cost with the reliability cost, distorting the crossover with DAC
- `step5_compute_mac_stats.py` already correctly uses `cost_incremental` (= `effective_cost - wholesale`) for MAC
- `step6_scenario_comparison.py` already correctly subtracts gas_cost before computing MAC: `new_build_per_mwh = total_cost - gas_cost`

**What changed**:
- `step5_compute_optimal_targets.py`: `SYSTEM_COST` (total_system_cost incl. gas) → `CLEAN_COST` (effective_cost only)
  - Medium: exact `effective_cost` from EFFECTIVE_COST_DATA
  - Low/High: approximation = `SYSTEM_COST(P10/P90) - gas_backup_cost(medium scenario)`
- `step5_consequential_deployment_queue.py`: MAC formula stripped `+ start/end['gas_cost']`; gas cost tracked separately as `delta_gas_cost_per_mwh`
- `step6_generate_shared_data.py`: added `CLEAN_COST_DATA` extraction (P10/P50/P90 of `effective_cost` across scenarios)

**Gas capacity as educational warning**:
- Gas backup cost is NOT part of the MAC but IS a critical educational point
- Consequential scenario dashboard must prominently warn: "Chasing cheap carbon without understanding system needs means retaining or building new gas capacity — an unavoidable system cost"
- `GAS_BACKUP_COST` per threshold per ISO tracked in optimal targets output for dashboard overlay
- `gas_cost_per_mwh_end` and `delta_gas_cost_per_mwh` added to consequential queue output

---

## 8. Dashboard Layout

### 8.1 Existing visuals — ALL PRESERVED
- Donut chart (resource mix) → updated to show 7 resources
- Compressed day profile (stacked area) → updated with Battery/LDES/CCS-CCGT split
- Peak capacity panel → updated with new resources
- Cost breakdown panel → updated with all resources
- Metric tiles: match score, procurement level, blended cost, curtailment → preserved
- Key finding box → preserved
- Scrollytelling narrative sections → preserved

### 8.2 New additions (layered on top)
- **CO2 abated metric tile** — tons of CO2 displaced for selected scenario
- **Average abatement cost curve chart** — $/ton across 75-100% thresholds
- **Marginal abatement cost curve chart** — incremental $/ton at each threshold step
- **"What You Need Depends on What You Have" panel**:
  - Starting point: grid mix baseline hourly match score
  - Target: selected threshold
  - Gap: target − baseline
  - Incremental resources needed to close gap
  - Incremental $/MWh above wholesale
  - CO2 impact of closing gap
  - Dynamic regional insight text (e.g., "NYISO's strong nuclear fleet means 40% less incremental clean firm needed vs. ERCOT")
- **Sensitivity toggle panel** (10 new toggles in control area)

### 8.3 Chart axis rules
- Abatement curves: **Linear numeric x-axis** (not categorical). 1 percentage point = same pixel distance everywhere.
- Data points at 75, 80, 85, 86, 87, ..., 100. Longer line segments between sparse points (75→80→85) are honest about lower granularity there.

---

## 9. Existing Grid Mix (2025 Actuals)

### Grid Mix Shares (% of generation):
| Resource | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Clean Firm | 7.9% | 8.6% | 32.1% | 18.4% | 23.8% |
| Solar | 22.3% | 13.8% | 2.9% | 0% | 1.4% |
| Wind | 8.8% | 23.6% | 3.8% | 4.7% | 3.9% |
| Hydro | 9.5% | 0.1% | 1.8% | 15.9% | 4.4% |

### Hydro Caps (2025 actual share of demand, from EIA):
| Region | CAISO | ERCOT | PJM | NYISO | NEISO |
|---|---|---|---|---|---|
| Cap (%) | 9.5 | 0.1 | 1.8 | 15.9 | 4.4 |
| 5yr range (%) | 5.2–11.2 | 0.07–0.12 | 1.9–2.1 | 15.9–18.3 | 4.5–7.8 |

**Notes**: Using 2025 actuals (not 5-year average) to match our 2025 snapshot model. CAISO hydro varies enormously by water year (2025 was above average). NYISO imports significant hydro from Quebec/Ontario.

### Wholesale Market Prices (2025 hourly profiles from EIA, not flat averages):
- Average reference points: CAISO ~$30, ERCOT ~$27, PJM ~$34, NYISO ~$42, NEISO ~$41
- Actual hourly data used for storage arbitrage, deficit-hour costing, curtailment economics

---

## 10. Two-Tier Pricing Model (Preserved)

For each resource:
- **Existing share** (up to grid mix %) → priced at wholesale
- **New-build share** (above grid mix %) → priced at LCOE + transmission adder
- **Hydro**: Always wholesale (existing only, no new-build tier)
- **CCS-CCGT** (within Clean Firm): No existing share (new resource) → all new-build priced

---

## 11. Performance Optimizations (v4.0 Rebuild)

### v4.0 Architecture (replaces v3.x sequential architecture)

- **Parallel ISO processing (A+F)**: All 7 ISOs run in parallel on 16 cores (~2 cores/ISO). Shared memory for cross-ISO data coordination. Replaces sequential processing.
- **Vectorized storage dispatch (B)**: Battery and LDES scoring use NumPy reshape/vectorized ops instead of Python day-loops. `surplus.reshape(365, 24)` for battery, vectorized rolling windows for LDES.
- **Batch mix evaluation (C)**: Grid search evaluates all combos in a single matrix multiply: `(N, 4) @ (4, 8760) = (N, 8760)`. Eliminates Python loop over individual mixes.
- **Numba JIT with fallback (D)**: Storage scoring functions compiled to machine code via Numba. If Numba unavailable, falls back to B+C (vectorized NumPy).
- **Checkpointing**: Saves after each threshold (20 per ISO); resumes from checkpoint on restart
- **Score caching**: Matching scores cached across 5,832 cost scenarios per threshold (physics reuse — cost-independent)
- **Cross-pollination**: After representative scenarios run per threshold, every unique mix re-evaluated against all scenarios
- **20 thresholds × 7 regions × 5,832 scenarios** — incremental saves essential for reliability

### 11.1 Direct Resource Fractions (v5.0 — replaces procurement multiplier)

**Decision (Feb 2026)**: Procurement multiplier removed. Resource fractions are now expressed directly as % of annual demand. No sum-to-100% constraint. Total generation (sum of all fractions) is what "procurement" used to be, but implicit.

**Old approach (removed)**: Mix shape (clean_firm + solar + wind + hydro = 100%) × procurement multiplier (50–500%). This created redundant evaluations — e.g., `(50% solar, 50% wind) @ 200%` and `(100% solar, 0% wind)` at different procurement levels could produce similar supply profiles. The procurement dimension was an unnecessary indirection.

**New approach**: Each resource varies independently as % of demand:
| Resource | Range | Step (coarse) | Step (fine) | Cap logic |
|----------|-------|---------------|-------------|-----------|
| Clean Firm (nuclear/CCS) | 0–120% | 5% | 1% | Nuclear/CCS with seasonal derate; 120% allows surplus for storage |
| Solar | 0–250% | 5% | 1% | High values capture solar+storage strategies |
| Wind | 0–250% | 5% | 1% | High values capture wind+storage strategies |
| Hydro | 0–(cap+10%) | 5% | 1% | Regional cap + 10% adder for run-of-river innovation potential. Extra hydro beyond existing cap is physics-only — NOT priced in Step 3. |
| Geothermal (CAISO only) | 0–20% | 5% | 1% | (existing + potential) / demand |

**Two-phase architecture**:

**Phase 1a — One-time coarse sweep per ISO** (cached to `data/step1-pfs/{ISO}_coarse_cache.parquet`):
- Generate all resource fraction combos at 5% step
- Score each combo once: `supply[h] = sum(frac[r] * profile[r][h])`, `score = sum(min(demand[h], supply[h]))`
- Cache `(resource fractions, score)` — reusable across ALL thresholds
- Run once per ISO; subsequent threshold work reads from cache

**Per-threshold work** (reads from coarse cache):
- Filter: `score >= target` → feasible combos (no-storage)
- Filter: `score >= target - 0.40` → storage zone → storage sweep
- Fine refinement at 1% step around frontier combos
- Save per-threshold results to `{ISO}_t{XX}_raw_pfs.parquet`

**What this removes**:
- `procurement_pct` column from all parquets (step1 → step2 → step3 → dashboard)
- `PROCUREMENT_BOUNDS` dict
- `vectorized_procurement_sweep()` function
- Binary search on procurement in storage sweep
- Cross-threshold pruning logic (unnecessary — all scores known from single sweep)

**Cost formula change**: Step 3 simplifies from `resource_frac/100 × procurement/100 × LCOE` to `resource_frac/100 × LCOE × demand_TWh`. Resource fractions directly represent generation volume.

**Persistent solution cache**: Results accumulated in `data/step1-pfs/` as per-ISO/threshold parquet files. Deduplication by (resource fractions, storage levels) key — no procurement dimension.

### 11.2 Edge Case Seed Mixes

Forced seed combos injected into coarse sweep to guarantee extreme-but-potentially-optimal strategies are evaluated. Now expressed as direct % of demand (not mix fractions):

- **High solar + storage**: solar=200%, wind=0%, CF=0%. Relies entirely on solar surplus + storage.
- **High wind + storage**: solar=0%, wind=200%, CF=0%. Same for wind-dominant regions.
- **Balanced high renewable**: solar=125%, wind=125%, CF=0%. Diversified variable generation.
- **Clean firm dominant**: CF=100%, solar=0%, wind=0%. Pure baseload.
- **CF + moderate solar**: CF=60%, solar=80%, wind=20%. Firm backbone + solar.
- **CF + moderate wind**: CF=60%, solar=20%, wind=80%. Firm backbone + wind.
- **Minimal firm**: CF=10%, solar=120%, wind=120%. Almost pure renewables.

Seeds filtered at runtime by regional hydro cap and geothermal cap (CAISO). Negligible compute cost, significant coverage improvement.

### 11.3 Monotonicity Re-Sweep Mechanism

**Problem**: The 3-phase heuristic search (coarse → medium → fine) can miss the global optimum at a lower threshold, producing a non-monotonic cost curve where cost(T_lower) > cost(T_higher). This is a diagnostic signal, not a valid result — achieving a lower CFE target should never cost more than achieving a higher one.

**Solution**: Post-hoc re-sweep with broader parameters. After all thresholds are initially optimized:

1. **Detection**: For each cost scenario, check that cost is non-decreasing across thresholds. Tolerance: $0.01/MWh (allows floating-point rounding).
2. **Collection**: Group all violations by threshold — `{threshold: {scenario_key: better_threshold}}`.
3. **Re-sweep** (up to 2 rounds): For each violated threshold:
   - **Seed injection**: Collect winning mixes from the thresholds that achieved better cost. These become Phase 1 seeds, guaranteeing the re-sweep explores the region of solution space that worked at the higher target.
   - **Broader Phase 1**: Use 5% step instead of 10% (~7-14× more combos), exploring the space more densely.
   - **Expanded procurement bounds**: Default bounds widened by -20% (min) and +30% (max) to search outside the assumed-optimal range.
   - **More Phase 2 candidates**: Top 30 instead of 20, with 2.0× cost filter (vs. 1.5×).
   - **More Phase 3 finalists**: Top 15 instead of 8, with 1.2× cost filter (vs. 1.1×).
   - Cross-pollination within re-swept threshold after re-optimization.
4. **Verification**: Re-check monotonicity after each round. If all violations resolved, stop early.
5. **Acceptance**: After 2 rounds, accept remaining violations with a warning (search space exhausted).

**Design rationale**: This approach finds the *true* optimum rather than masking the problem by pulling down from a higher threshold's result. The re-sweep is targeted (only violated scenarios) and seeded (with known-good mixes from higher thresholds), so it's both rigorous and compute-efficient.

**Compute overhead**: Typically affects 5-15% of scenarios at 1-3 thresholds per ISO. Phase 1 at 5% step generates ~2,000-5,000 combos (vs. ~280 at 10% step), but scoring is cached. Net overhead: ~10-20% of total runtime.

---

## 12. Methodology Documentation Checklist

The methodology page must include:
- [ ] All generation cost tables (Sections 5.1-5.4) with sources (NREL ATB 2024, Lazard, LBNL)
- [ ] All storage cost tables (Sections 5.5-5.6) with sources
- [ ] Complete transmission adder table (Section 5.7) with sources (LBNL "Queued Up", MISO/SPP)
- [ ] Fuel price scenario table (Section 5.8) with sources (EIA AEO, Henry Hub)
- [ ] Fuel → wholesale + emission rate linkage methodology (Section 5.9)
- [ ] Regional fuel-switching elasticity rationale
- [ ] CCS-CCGT complete cost buildup (capture + transport + storage + fuel − 45Q)
- [ ] Class VI well availability rationale by region
- [ ] 45Q tax credit mechanics and levelized impact
- [ ] LDES vs battery technology specs, efficiency, duration
- [ ] LDES dispatch algorithm description
- [ ] Battery dispatch algorithm description (preserved)
- [ ] CO2 emission factor methodology (eGRID marginal rates)
- [ ] CCS-CCGT partial credit methodology (90% capture)
- [ ] Abatement curve formulas (average and marginal)
- [ ] Hydro treatment rationale (existing-only, wholesale, regional caps)
- [ ] Two-tier pricing explanation
- [ ] Hourly wholesale price profile methodology
- [ ] Grid mix baseline methodology
- [ ] All cited sources: LBNL, NREL ATB 2024, Lazard, EIA AEO, eGRID, MISO/SPP, FERC/ISO reports

---

## 13. Regional Deep-Dive Pages (1 combined page)

All 7 regions covered in a single scrollytelling page with region selector.

### Structure
- **Status**: DELETED (Feb 19, 2026). Regional deep-dive content consolidated into research paper and homepage scrollytell.

### Default Cost Scenario for Static Pages
- **Homepage (index.html)** and **Regional Deep-Dive pages**: All figures and narrative use **Medium cost sensitivities** (all 8 toggle groups at Medium — Renewable Gen, Firm Gen, Storage, CCS, 45Q=On, Fossil Fuel, Transmission, Geothermal) unless a figure is explicitly designed to show Low/Medium/High ranges for comparison purposes.
- **Dashboard (dashboard.html)**: Interactive — user controls sensitivities via toggles.
- This ensures consistency across static narrative pages and reserves L/M/H range displays for intentional comparison figures (e.g., cost sensitivity deep-dive section #4 below).

### Per-Region Content Sections
1. **Region Overview** — grid composition, geography, market structure, current clean energy share
2. **Baseline Analysis** — existing grid hourly match score, seasonal patterns, strengths/weaknesses
3. **Decarbonization Pathway** — how optimal resource mix evolves from 75% → 100% threshold
4. **Cost Sensitivity Deep-Dive** — how L/M/H scenarios on key resources (regional priority resources) change the cost picture (this section explicitly shows ranges)
5. **Storage Role** — battery vs LDES contributions at different thresholds, dispatch patterns
6. **CO2 Abatement Profile** — regional abatement curves with commentary on inflection points
7. **Key Regional Insights** — unique factors (e.g., ERCOT's wind dominance, NYISO's nuclear fleet, CAISO's solar+storage, PJM's coal fleet switching dynamics)
8. **Comparison to National Context** — how this region compares to others

### Regional Priority Resources
- **CAISO**: Solar + battery, geothermal clean firm
- **ERCOT**: Wind + CCS-CCGT (Gulf Coast geology), low-cost solar
- **PJM**: Nuclear clean firm, coal→gas switching dynamics, wind
- **NYISO**: Nuclear fleet, hydro, limited renewables siting
- **NEISO**: Offshore wind potential, nuclear, limited solar

---

## 14. Research Paper PDF

### Format
- Generated as PDF (via HTML→PDF or direct HTML print stylesheet)
- Academic paper structure with executive summary
- Includes all regional deep-dive content as paper sections
- Includes full methodology detail (more than the methodology HTML page)

### Paper Structure
1. **Executive Summary** — key findings across all regions
2. **Introduction** — hourly CFE matching problem, why annual matching isn't enough
3. **Methodology** — full model description, all cost tables, algorithms, data sources
4. **National Results** — overview across all 7 regions, comparison charts
5. **Regional Deep-Dives** (5 sections, one per region — content from deep-dive pages)
6. **Sensitivity Analysis** — how key assumptions drive results
7. **Policy Implications** — what this means for procurement strategy
8. **Appendix** — full data tables, source citations, technical specifications

### Audience
- **Primary**: Business professionals with minimal energy domain knowledge
- **Secondary**: Academic/policy reviewers (must withstand scrutiny)
- Accessible first, rigorous underneath

---

## 15. Abatement Cost Comparison Page (NEW)

### Concept
A "Liebreich ladder for grid decarbonization" — analyzing when/where/under what conditions pushing grid decarbonization % is no longer cost-effective compared to alternative mitigation and carbon removal options. Linked from dashboard navigation.

**Status**: DELETED (Feb 19, 2026). Consolidated into `abatement_dashboard.html` (now "CO₂ Abatement Analysis").

**Core Question**: "Should we focus the next marginal dollar on the last 5% of PJM grid decarbonization, sustainable aviation fuel, or direct air capture?"

### Analysis Framework

**Y-axis**: Cost of carbon abatement ($/ton CO2)
**X-axis**: Cumulative abatement potential or grid % target

**Grid Decarbonization Curves** (from our model):
- Regional marginal abatement cost curves (75-100%) for each ISO under L/M/H sensitivities
- Show hockey-stick inflection where costs spike (typically 95-100%)
- Each region's curve under different sensitivity scenarios

**Comparison Benchmarks** (horizontal lines/bands on same chart):
| Mitigation Option | Low $/ton | Medium $/ton | High $/ton | Source |
|---|---|---|---|---|
| Energy efficiency (buildings) | $0 | $20 | $50 | IEA, McKinsey |
| Industrial electrification | $30 | $75 | $150 | IEA |
| Sustainable Aviation Fuel (SAF) | $150 | $250 | $400 | ICCT, BloombergNEF |
| Green hydrogen (industrial) | $100 | $200 | $350 | Liebreich, BNEF |
| BECCS | $100 | $175 | $300 | IPCC AR6 |
| Direct Air Capture (DAC) | $250 | $400 | $600+ | Carbon Engineering, Climeworks |
| Enhanced weathering | $50 | $125 | $200 | IPCC |
| Carbon credits (voluntary market) | $10 | $50 | $150 | Ecosystem Marketplace |
| EU ETS carbon price (2024-2025) | $60 | $80 | $100 | EMBER |

### Key Analytical Sections

1. **The Grid Decarbonization Curve** — Our model's regional marginal abatement curves plotted together. Where does each region's curve cross the DAC line? The SAF line?

2. **The Inflection Point Analysis** — For each region × sensitivity scenario, identify the % threshold where grid decarbonization costs exceed:
   - The social cost of carbon ($190/ton 2024 EPA central (3% SDR), $185/ton Rennert et al.)
   - DAC costs ($300-1,100/ton, trajectory-dependent)
   - SAF costs ($150-400/ton)
   - Voluntary carbon market prices ($10-150/ton)

3. **The Liebreich-Style Ladder** — Rank all mitigation options by cost-effectiveness at each grid % level. At 85% grid target, what's cheaper? At 95%? At 99%?

4. **Regional Divergence** — Some regions (ERCOT with cheap wind) stay cost-competitive deep into high %'s. Others (NYISO) become expensive earlier. Map the crossover points.

5. **The Net-Zero Pathway** — Given that residual emissions exist at any grid %, what's the optimal split between:
   - Pushing grid % higher (expensive past inflection)
   - Investing in DAC for residual emissions
   - Investing in other sectors (SAF, industrial) for cross-sector abatement

6. **Sensitivity Scenarios** — How do different cost assumptions shift the inflection points? Under low DAC cost assumptions, the crossover happens earlier. Under high renewable cost assumptions, same.

### Path-Dependency & Retroactive Cost Modeling (Under Development)

**Problem**: The optimizer independently optimizes each threshold. The 85% mix and 95% mix may differ fundamentally — heavy solar at 85%, heavy clean firm at 95%. Building the 85%-optimal mix then upgrading to 95% would strand solar assets and cost more than building toward 95% from the start.

**Proposed Approach**: Model backwards from the inflection point where the optimal grid solution's LCOE crosses a benchmark price:
- **Primary benchmark**: DAC cost projected to the target year (see §7.3 revised trajectories)
  - 2025: $600-1,100/ton → grid dominates through ~97%+
  - 2035: $230-550/ton → grid dominates through ~93-95%
  - 2045: $130-375/ton → grid dominates through ~90-93%
- At the crossover threshold, the optimal mix is fixed. Then model the build-up path from lower thresholds using the cheapest-first resource ordering that converges to the crossover mix.
- DAC learning curve: ~15-20% cost reduction per doubling of deployment (ETH Zurich/Climeworks data)
- This creates a **declining optimal grid target over time** as removal costs fall — counterintuitive but logical.

**Status**: Waiting for optimizer results to analyze mix divergence between thresholds. If divergence is small (resources are additive), the current independent optimization is sufficient. If large (mix pivots between thresholds), path-constrained modeling is needed.

### DAC-VRE Co-Optimization Insight (Under Development)

**Core insight**: DAC is a flexible load that can absorb curtailed renewable energy. At high grid targets, significant curtailment occurs — this energy is nearly free ($0-5/MWh). DAC facilities co-located with sequestration geology (Class VI wells) can use curtailed power to remove CO₂ at dramatically reduced costs.

**Regional specialization model**:
- **ERCOT/CAISO**: Push grid to 95-97% (cheap wind/solar), operate DAC on curtailed surplus. Gulf Coast & Imperial Valley have Class VI well capacity.
- **PJM**: Push grid to 93-95%, buy ERCOT/regional DAC credits for residual emissions.
- **NYISO/NEISO**: Push grid to 90-92% (expensive beyond), heavy DAC credit procurement from regions with cheaper removal.

**DAC cost with curtailed power**: If energy is the #1 DAC cost driver (~60% of total), curtailed power at $0-5/MWh could reduce DAC from $600-1,100/ton to $250-450/ton — making it competitive with grid decarbonization costs above 93-95% in most regions.

**Analysis needed**:
1. From optimizer results: quantify curtailed MWh at each threshold × region
2. Estimate DAC capacity supportable by curtailed energy (assume 2 MWh/ton)
3. Derive DAC marginal cost curve as a function of curtailment availability
4. Compare DAC-on-curtailment cost to grid MAC at each threshold
5. Find the optimal regional grid target + DAC allocation

**Why this matters**: This reframes the "100% clean grid" question. If DAC-on-curtailment is cheaper than the last 5-10% of grid matching, the rational strategy is to overbuild VRE (creating more curtailment) and co-locate DAC — achieving net-zero at lower total cost than pure grid matching.

**DAC operating model: curtailment-only (20-30% CF)**:
- DAC as a **grid balancing tool** — runs only on otherwise-curtailed renewable energy
- This avoids adding demand that competes with other loads, which would drive up wholesale prices and congestion
- From a social good standpoint: DAC should not increase grid costs; it should absorb energy that would be wasted
- At 20-30% CF: CapEx/ton is 3-6x higher than full utilization, but energy cost is near-zero ($0-5/MWh)
- 2025 total: ~$835/ton (too expensive); **2040 projected: ~$360-410/ton** (competitive above 96-97% grid MAC)
- **2045-2050 projected: ~$280-320/ton** (competitive above 93-95% grid MAC in most regions)
- Standard DAC MAC curves assume ~90% CF — our model adjusts for curtailment-only operation
- Additional value not captured in per-ton cost: DAC provides grid stabilization by absorbing excess generation

**Abatement page section**: Dedicated section with narrative walkthrough + findings in the Key Insights panel at top.

**Implementation scope**: Supplementary analysis for the **abatement page only** — not the main dashboard.
- Run supplementary optimizer scenarios for ERCOT and CAISO (both have Class VI well capacity — Gulf Coast and Central Valley/Salton Sea respectively)
- Model allows monetizing curtailed energy via DAC-VRE co-location
- These regions push to 100%+ procurement with excess curtailment → DAC
- DAC credits offset residual emissions in NYISO/NEISO/PJM at high targets where grid costs are steep
- Produces a "cross-regional portfolio" where cheap-DAC regions export removal credits to expensive-grid regions

### Visual Design
- Large interactive chart: Regional MAC curves overlaid with benchmark bands
- Horizontal benchmark lines clearly labeled with color-coded bands
- Inflection point callouts where curves cross benchmarks
- Toggle: Region selector, sensitivity scenario
- Scrollytelling narrative explaining the analysis

### Audience
- Same business professional audience
- Build the case: "Here's when clean energy procurement stops being the cheapest path and alternatives become more efficient"
- Frame as strategic portfolio optimization, not just grid optimization

---

## 15b. Methodology & Research Paper (Merged)

- **research_paper.html** is now the single source of truth for methodology + research content
- **Appendix B** added with: B.1 Two-Tier Pricing Model, B.2 Generation Cost Tables, B.3 Storage Cost Tables, B.4 Transmission Adders, B.5 Sensitivity Toggle Pairing, B.6 CO₂ Emission Factor Methodology, B.7 Hydro Treatment
- **optimizer_methodology.html** preserved but removed from nav — all content consolidated
- Nav link: "Methodology & Paper" → research_paper.html
- Clickable table of contents at top of page

---

## 16. Header Banner & Navigation

### Banner Placement
- **Main dashboard**: Banner appears ABOVE intro text (not below)
- **All pages** (dashboard, regional deep-dives, methodology): Same header banner styling
- Banner includes page-specific name + tagline

### Per-Page Banner Content
| Page | Title | Tagline |
|---|---|---|
| Homepage (index.html) | The 8,760 Problem | Most climate solutions depend on a clean grid. But how clean is clean enough? |
| Cost Optimizer (dashboard.html) | Hourly CFE Optimizer | Advanced Sensitivity Model |
| CO₂ Abatement Analysis (abatement_dashboard.html) | CO₂ Abatement Analysis | Comparing Grid Decarbonization to Alternative Pathways |
| Methodology & Paper (research_paper.html) | Technical Methodology & Research Paper | Full Paper with Appendix B Cost Tables |

### Navigation (Updated Feb 14)
- Top navigation bar on ALL pages
- Links: Home | Cost Optimizer | Abatement Dashboard | Regional Deep Dives | CO₂ Abatement Summary | Methodology & Paper
- Current page highlighted in nav (nav-active class)
- Mobile: collapsible/hamburger nav
- "Back to Home" button at top of all non-home pages
- Methodology page (optimizer_methodology.html) still exists but removed from primary nav — content consolidated into research_paper.html Appendix B

---

## 17. Audience & UX Guidelines

### Dashboard (business professional audience)
- Layer in explanations for model elements, figures, toggles
- Tooltips or info icons (ⓘ) on each control explaining what it does and why it matters
- Chart titles that tell the story, not just label the axis
- Key finding boxes that translate numbers into business implications
- Assume reader does NOT know what LCOE, LCOS, capacity factor, or hourly matching mean

### Regional Deep-Dive Pages
- Written for a reader encountering the topic for the first time
- Build understanding progressively (scrollytelling)
- Lead with "so what" before diving into "how"
- Use analogies and real-world comparisons where helpful

### Research Paper / Methodology
- More technical depth acceptable
- Must still be accessible to first-time readers
- Withstand academic scrutiny: cite sources, show methodology, acknowledge limitations
- Full cost table transparency

---

## 18. QA/QC Requirements

### Optimizer Results QA (after first region completes)
- Validate hourly match scores against expected ranges from existing research
- Check that resource mixes make directional sense (e.g., wind-heavy in ERCOT, nuclear-heavy in PJM)
- Verify cost figures fall within published LCOE/LCOS ranges (NREL ATB, Lazard)
- Confirm CO2 abatement numbers are physically reasonable (tons displaced per MWh)
- Check that higher thresholds always cost more than lower ones (monotonicity)
- Verify storage dispatch increases with threshold (more storage needed at higher targets)

### Dashboard HTML QA
- Visual consistency: fonts, colors, spacing, alignment across all sections
- All toggles functional and responsive
- Chart rendering correct with proper labels, legends, axes
- Metric tiles update correctly when controls change
- No broken layouts at any control combination
- Clean/crisp visual identity — no cluttered elements

### Mobile Compatibility
- All figures render with readable text on mobile screens (320px-768px)
- Touch-friendly toggle controls (minimum 44px tap targets)
- Charts scale properly (responsive Canvas/Chart.js)
- Scrollytelling sections work on touch scroll
- No horizontal overflow or text truncation
- Navigation accessible on mobile (hamburger or stacked)
- Test at: 320px (small phone), 375px (iPhone), 768px (tablet)

### Pre-Push Checklist
- [ ] Optimizer results QA passed for all 7 regions
- [ ] All dashboard controls functional
- [ ] All charts render correctly
- [ ] Mobile compatibility verified
- [ ] Regional deep-dive pages complete and linked
- [ ] Research paper PDF generated
- [ ] Methodology page trimmed
- [ ] No console errors in browser
- [ ] Standalone HTML builds successfully
- [ ] All text readable at all viewport sizes

---

## 18. Summary Counts (v4.0)

| Item | Count |
|---|---|
| Resources (optimization dimensions) | 4 (clean_firm, solar, wind, hydro) — CCS merged into clean_firm |
| Resources (total modeled) | 6 (clean_firm incl. CCS, solar, wind, hydro, battery, LDES) |
| Thresholds | 13 (expanded from 10: added 50%, 60%, 70%) |
| Regions | 5 |
| Dashboard controls | 12 (2 existing + 7 graduated toggles + 1 binary + 2 region-conditional) |
| Sensitivity toggles | 7 graduated (L/M/H) + 1 binary (45Q On/Off) + 1 CAISO-only (Geothermal L/M/H) |
| Step 1 physics scenarios per region/threshold | 324 (3×3×3×3×4) — each independently co-optimized |
| Step 3 cost scenarios (non-CAISO) | 5,832 (3×3×3×3×2×3×4) per region/threshold |
| Step 3 cost scenarios (CAISO) | 17,496 (5,832 × 3 geothermal) per threshold |
| Total Step 3 evaluations | ~40,824 sensitivity combos × unique mixes per (region, threshold) |
| Pareto points per scenario | 3-5 (procurement/storage tradeoff frontier) |
| Regional deep-dive pages | 1 (combined, with region selector) |
| Research paper sections | 8 (including 5 regional deep-dives) |
| QA checkpoints | 3 (optimizer, HTML, mobile) |
| Output formats | 2 (JSON + Parquet) |

---

## 19. Model Limitations & Simplifying Assumptions

This section documents known simplifying assumptions for transparency and academic rigor. These should be acknowledged in the research paper and methodology page.

### 19.1 Static LDES LCOS (Utilization-Independent)

**Assumption**: LDES (100hr iron-air) uses a static LCOS ($/MWh) from published cost tables at assumed cycling frequency, regardless of the scenario's realized dispatch utilization.

**Why this matters**: LDES is extremely capital-intensive (~$5,000-10,000/kW installed at 100hr duration). The LCOS is dominated by capital recovery, so it is highly sensitive to utilization. A scenario where LDES cycles 50 times/year has a dramatically lower effective LCOS than one where it cycles 5 times/year — yet both use the same $/MWh in the model.

**Impact**: In scenarios with low LDES utilization (e.g., solar-dominant mixes with limited multi-day surplus), the model may understate the true cost of LDES. In scenarios with high utilization (wind-dominant mixes with abundant multi-day surplus to time-shift), the model may overstate LDES costs.

**Justification**: This approach is consistent with standard practice in published energy models (NREL ATB, Lazard LCOS). These sources quote LCOS at assumed utilization rates, and most capacity expansion models use static cost inputs without feedback from dispatch results. Implementing utilization-dependent LCOS would create a cost ↔ dispatch feedback loop (cost depends on dispatch, which depends on mix, which depends on cost) that, while convergent, adds significant methodological complexity. The same limitation applies to CCS-CCGT capacity factor effects on LCOE, though to a lesser degree given CCS's lower capital intensity per kW.

**Mitigation**: The optimizer's resource mix co-optimization partially self-corrects for this — it won't allocate large LDES shares in mixes that don't produce sufficient multi-day surplus to fill it, because the matching score won't benefit enough to justify the cost. The limitation is most relevant at the margin, where small LDES allocations face the highest effective cost per useful MWh.

### 19.2 CCS-CCGT at Assumed Baseload Capacity Factor

**Assumption**: CCS-CCGT LCOE reflects assumed high-capacity-factor baseload operation. In practice, CCS plants in a high-renewable grid might operate at lower capacity factors, increasing their effective LCOE.

**Impact**: Similar to LDES, the model may understate CCS-CCGT costs in scenarios where it operates at low utilization. However, since the optimizer models CCS as flat baseload (1/8760 profile), allocated CCS capacity runs at 100% CF by construction. The limitation applies to whether that assumption reflects real-world operations in a grid with significant renewable penetration.

**Mitigation**: The firm generation cost toggle (Low/Medium/High) provides sensitivity analysis around the LCOE assumption. High firm generation costs can be interpreted as a proxy for reduced capacity factor economics.

### 19.3 DST-Aware Solar Nighttime Correction (Fixed)

**Implementation**: Solar nighttime zeroing now accounts for Daylight Saving Time. The optimizer applies a 6am–7pm local prevailing time daylight window, converting to UTC using DST-adjusted offsets:
- **Standard time** (Nov–Mar): CAISO UTC+8, ERCOT UTC+6, PJM/NYISO/NEISO UTC+5
- **DST** (Mar–Nov, ~day 69–307): Offsets decrease by 1 (CAISO UTC+7, ERCOT UTC+5, PJM/NYISO/NEISO UTC+4)
- DST boundaries use representative dates across 2021–2025 (2nd Sunday of March ≈ day 69, 1st Sunday of November ≈ day 307)

**Optimizer matching**: Unaffected — `demand[h]` and `supply[h]` refer to the same physical UTC hour. DST adjustment only affects which hours get nighttime solar zeroing.

**If re-importing EIA data**: The `fetch_all_data.py` script stores all profiles in sequential UTC order. The DST correction is applied at profile loading time in the optimizer, NOT during data import. This means raw data files are always UTC and the DST logic lives only in `get_supply_profiles()`.

### 19.4 Multi-Year Data Usage (Implemented)

**Data split — what comes from where:**
- **2021-2025 average**: Hourly profile *shapes* for both generation (solar, wind, hydro, nuclear) and demand. Element-wise average across 5 years smooths single-year weather anomalies.
- **2025 actuals**: Total annual MWh (demand and generation), existing grid mix shares, hydro caps, peak demand. These anchor the model to current-year reality.
- **Solar nighttime correction**: Solar generation zeroed during nighttime hours using DST-aware local time windows (see §19.3).

**Leap year handling**: 2024 (8784 hours) is included by removing Feb 29 hours (indices 1416–1439) before averaging, preserving seasonal alignment with 8760-hour non-leap years.

**Data quality validation**: Demand profiles are validated at load time for statistical outliers using hour-of-day median comparison. Years where any hour-of-day has a maximum value exceeding 100× the median for that hour are excluded from the average. This catches EIA data entry errors (e.g., unit conversion errors that inflate individual hours by orders of magnitude). **Known exclusion**: PJM 2021 is excluded — October 19, 2021 hours 03:00-05:00 UTC contain demand values ~20,000× normal (0.31–0.44 of annual normalized demand concentrated in 3 hours), likely an EIA reporting error. PJM demand shape is averaged over 2022-2025 (4 years). All other ISO-year combinations pass validation. Raw data is preserved unmodified in `eia_demand_profiles.json` for auditability.

**Implementation in `load_data()`**:
- `_remove_leap_day(profile)`: Excises Feb 29 from 8784→8760
- `_validate_demand_profile(iso, year, profile)`: Detects corrupt years via hour-of-day outlier check
- `_average_profiles(yearly_profiles)`: Element-wise mean across years
- Generation profiles: `gen_profiles[iso][resource_type]` → direct access (no year key)
- Demand profiles: `demand_data[iso]['normalized']` uses averaged shape; `total_annual_mwh` and `peak_mw` from 2025

**Key constraint**: Profile shapes are weather-averaged; absolute quantities are 2025 actuals. This means the optimizer uses realistic hourly patterns (no single-year weather bias) scaled to actual 2025 generation levels.

**If re-importing EIA data**: The `fetch_all_data.py` script stores raw per-year profiles in `eia_generation_profiles.json` and `eia_demand_profiles.json`, each year-keyed. The 5-year averaging happens at optimizer load time, NOT during import. Raw data files preserve full per-year resolution for auditability.

### 19.5 NYISO Solar Proxy

**Status**: Working correctly. NYISO uses NEISO solar generation profile as proxy since NYISO lacks meaningful solar generation data in EIA 930. The proxy is stored in `eia_generation_profiles.json` as `solar_proxy` under NYISO and matches NEISO solar values exactly. The optimizer code (line 298-302) checks for `solar_proxy` first, falls back to NEISO solar.

### 19.6 Geothermal Scope: Conventional Hydrothermal Only (CAISO)

**Assumption**: Geothermal resource is modeled only for CAISO, using conventional hydrothermal potential (identified by USGS). Enhanced Geothermal Systems (EGS) and other advanced geothermal technologies (closed-loop, superhot rock) are excluded from all ISOs.

**Why this matters**: EGS could theoretically unlock GW-scale firm clean power in regions with no conventional hydrothermal resource (PJM, NYISO, MISO, etc.). DOE's Enhanced Geothermal Shot initiative targets 2035 for cost-competitive EGS, and projects like Fervo Energy's Utah pilot (2026) and DOE's FORGE site are advancing the technology. If EGS reaches commercial scale, the firm clean power landscape changes substantially for non-CAISO regions.

**Justification**: This is a 2025 snapshot model. Conventional hydrothermal is the only geothermal technology commercially deployed at scale today, and CAISO is the only modeled ISO with meaningful resource (5.3 TWh existing + ~39 TWh USGS identified = ~44 TWh potential, capped at 5 GW). Non-CAISO ISOs sit on geology unsuitable for conventional hydrothermal — Appalachian basement rock (PJM/NYISO/NEISO) with 20–25°C/km thermal gradients, deep sedimentary basins (MISO/SPP), or early-stage pilots (ERCOT — Sage Geosystems). None have commercial-scale geothermal in the 2025 timeframe.

**EGS exclusion rationale**: EGS commercial deployment timelines (DOE targets 2035, industry consensus >2030 for first utility-scale projects) fall outside this model's 2025 snapshot scope. Including speculative EGS capacity would require forward-looking assumptions about cost learning curves, drilling success rates, and induced seismicity risk that are inconsistent with the model's empirical, current-year methodology. EGS is noted as a potential model enhancement for post-2030 analysis (see §21).

**Impact**: The model may overstate the long-term cost of firm clean power for non-CAISO regions if EGS achieves cost targets. For the 2025 snapshot, this is immaterial — no EGS capacity exists to procure today.

**Geothermal cap citation**: 39 TWh based on USGS 2008 Assessment of Moderate- and High-Temperature Geothermal Resources of the United States (Fact Sheet 2008-3082), updated with California Energy Commission 2021 identified resource estimates for Salton Sea/Imperial Valley. Conventional hydrothermal sites account for ~80% of near-term identified US geothermal potential.

### 19.7 Single-Zone ISO Assumption (Added Mar 2026)

**Assumption**: Each ISO is modeled as a single zone with perfect internal transmission. No intra-ISO congestion or locational price variation.

**Why this matters**: Real ISOs have significant internal congestion. PJM Western Hub vs. Eastern Hub can differ $5-15/MWh. CAISO North/South congestion is material, particularly for solar (south) serving load (north). ERCOT has West-to-Houston congestion for wind-heavy portfolios.

**Impact**: The model may understate transmission costs for resources located far from load centers within an ISO, and may not capture locational basis risk that affects PPA pricing.

**Scope limitation**: This model is suitable for corporate procurement portfolio analysis at ISO-level granularity. It is not designed for transmission system planning, nodal pricing analysis, or locational resource adequacy assessment. Intra-ISO congestion effects would require sub-zonal modeling with transmission flow constraints.

### 19.8 Weather-Year Sensitivity (Added Mar 2026)

**Assumption**: Generation and demand profiles use 5-year element-wise averages (2021-2025). No weather-year sensitivity analysis (P10/P50/P90 years).

**Why this matters**: A low-wind or low-hydro year could significantly shift optimal resource mixes and costs, particularly at high thresholds (≥95%) where resource adequacy during adverse weather is the binding constraint. Solar and wind capacity factors vary ±15-25% between weather years.

**Mitigation**: The 5-year average smooths single-year anomalies. Low/Medium/High renewable cost toggles provide some proxy sensitivity (low-cost implies higher CF assumptions). Weather-year sensitivity with per-year profiles and Step 1 re-runs is noted as future work (§21).

### 19.9 Storage Dispatch Priority — Greedy Sequential (Added Mar 2026)

**Assumption**: Storage dispatches in fixed priority order (4hr battery → 8hr battery → LDES → Green H2), using window-based greedy algorithms with no price signal or foresight.

**Why this matters**: A global LP optimization with perfect foresight would optimally allocate surplus across storage types, potentially increasing total storage utilization by 10-20%. The greedy approach may under-dispatch LDES in scenarios where saving surplus for multi-day shifting would be more valuable than daily battery cycling.

**Justification**: Greedy sequential dispatch represents an operational lower bound on storage utilization. Perfect foresight LP would be unrealistic for actual dispatch operations. The window-based approach (24hr for 4hr battery, 48hr for 8hr, 7-day for LDES, 30-day for H2) captures the intended operational pattern of each technology.

### 19.10 No Demand Response or Demand-Side Management (Added Mar 2026)

**Assumption**: Demand is a fixed hourly profile with no flexibility. No demand response, EV managed charging, thermal storage, or other demand-side resources are modeled.

**Why this matters**: Modern utility and corporate procurement strategies increasingly include demand-side resources. DR/EV flexibility could reduce procurement costs 5-15% at high thresholds by shifting demand to match clean supply profiles.

**Scope limitation**: This is a supply-side procurement analysis. Demand-side resources are not modeled; adding them is noted as a future extension (§21).

### 19.11 SSS and Merchant Cost Simplification (Added Mar 2026)

**Assumption**: State-Supported Structure (SSS) pool savings use wholesale price × capacity as the benefit. Merchant LCOE is static at $35/MWh with no ISO variation. No validation against actual PPA or EAC market data.

**Why this matters**: Actual pool-adjusted costs vary significantly by ISO, PPA vintage, and market conditions. The $35/MWh merchant LCOE is an approximation of recent wind/solar PPA prices but does not capture ISO-specific premiums or discounts.

**Justification**: Pool-adjusted costs are approximate by design — the model's primary contribution is the physics-to-cost co-optimization, not PPA pricing precision. Sensitivity to pool cost assumptions can be explored via the renewable cost toggles (L/M/H).

### 19.12 Hourly Emission Rates — Scalar Per Threshold (Added Mar 2026)

**Assumption**: CO₂ emission rate is a single scalar applied uniformly across all 8,760 hours for a given clean energy threshold. The rate changes across thresholds (coal → oil → gas merit-order retirement) but not within a year.

**Why this matters**: Real marginal emission rates vary 2-3× across day/night and season. Coal dominates baseload (night/winter), gas dominates peaks (afternoon/summer). A constant rate overestimates abatement from displacing daytime fossil (which is cleaner gas) and underestimates abatement from displacing nighttime fossil (which is dirtier coal).

**Current implementation**: `compute_fossil_retirement()` in `dispatch_utils.py` computes a scalar rate based on merit-order displacement. The hourly `fossil_displaced[h]` array captures temporal dispatch patterns, but all hours use the same emission factor.

**Planned improvement**: Implement hourly marginal emission rate approximation using the dispatch stack model. See audit item H7.

---

## 20. Model Alignment and Differentiation vs. Existing Energy Models

This section documents how our model compares to established capacity expansion and procurement models, where we align with standard methodology, and where we deliberately diverge with justification.

### 20.1 Alignment with Standard Methodology

| Feature | Our Model | Industry Standard (GenX, ReEDS, SWITCH) | Alignment |
|---|---|---|---|
| **Hourly temporal resolution** | 8760 hours | 8760 hours (GenX), representative weeks (ReEDS), 12-288 time slices (SWITCH) | ✓ Matches GenX; exceeds ReEDS/SWITCH |
| **LCOS at reference utilization** | Static LCOS from NREL ATB/Lazard | Same — static cost inputs without dispatch feedback | ✓ Full alignment |
| **Solar/wind hourly profiles** | EIA 930 actual generation data, 5-year average | NREL ATB capacity factors, or NSRDB/WIND Toolkit | ✓ Comparable rigor; actual generation vs. modeled resource |
| **Two-tier pricing** | Existing capacity at wholesale; new-build at LCOE + transmission | Standard in procurement models (LevelTen, 3Degrees) | ✓ Full alignment |
| **Co-optimization of cost + mix** | Cost drives resource mix selection at every threshold | Standard in all capacity expansion models | ✓ Full alignment |
| **Regional granularity** | 7 ISOs (CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP) | GenX: zonal; ReEDS: 134 BAs; SWITCH: load zones | ✓ Comparable scope for procurement analysis |

### 20.2 Deliberate Differentiations (with justification)

| Feature | Our Model | Standard Models | Why We Diverge |
|---|---|---|---|
| **CCS-CCGT as flat baseload** | Always-on, 100% CF | Dispatchable (ramps with system needs) | **45Q tax credit incentive**: $85/ton for captured CO₂ creates a strong economic incentive to maximize capacity factor regardless of grid need. The policy distortion means CCS would run baseload in practice, not dispatch. Standard models don't account for 45Q's perverse incentive structure. |
| **Nuclear seasonal derate** | Monthly flat derate from 5-year EIA data (spring/fall refueling) | Flat 90-93% annual CF (NREL ATB) or explicit outage scheduling (PLEXOS) | **Seasonal accuracy matters for hourly matching**: A flat annual CF misses the spring/fall refueling pattern where clean firm availability drops 15-20%. For hourly CFE procurement, this seasonal gap is exactly when storage or CCS must compensate. Our approach uses observed EIA data rather than assumed CF, and preserves high summer/winter availability when clean firm is most valuable. |
| **Storage capacity-constrained dispatch** | Capacity built = physical limit on daily/weekly dispatch | Varies: some use exogenous capacity, some co-optimize | **Prevents unrealistic dispatch**: The optimizer can't claim more storage dispatch than the built capacity allows. Days with insufficient surplus get partial cycles. This is more conservative than models that assume perfect foresight dispatch or exogenous capacity sizing. |
| **CO₂ hourly attribution with charge netting** | Track exact dispatch hours + net charge-side emissions | Flat marginal emission rate or annual average | **Prevents CO₂ overcounting**: Storage charging from fossil-marginal hours carries real emissions. Our approach credits storage abatement only for the net emission reduction, not the gross displacement. This is consistent with the GHG Protocol Scope 2 hourly matching framework. |
| **Hydro as existing-only** | Capped at 5-year average share, wholesale-priced, no new-build | Varies: some allow new-build hydro/pumped storage | **Reflects procurement reality**: New conventional hydro is effectively unavailable in the US (permitting, environmental constraints). Treating it as existing-only matches what a corporate buyer can actually procure. |
| **Procurement-focused objective** | Minimize $/MWh to achieve target CFE % | Minimize total system cost or maximize welfare | **Different question**: We're asking "what should a buyer procure?" not "what should the system build?" This means we don't model transmission expansion, retirement decisions, or inter-regional trade — we take the grid as-given and optimize the buyer's clean energy portfolio within it. |

### 20.3 Key Assumptions Where We Use Standard Values

- **Battery**: 4hr Li-ion, 85% RTE, daily-cycle dispatch → NREL ATB 2024 reference
- **LDES**: 100hr iron-air, 50% RTE → Form Energy published specs, NREL ATB storage module
- **CCS capture rate**: 95% → DOE/NETL reference for next-gen CCGT+CCS (conservative vs. 90% in older literature)
- **45Q offset**: $85/ton × 95% capture × ~0.37 tCO₂/MWh ≈ $29/MWh LCOE reduction → IRC §45Q(a)(3)(A)
- **Discount rate**: Implicit in LCOE tables (NREL ATB uses WACC by technology)
- **Transmission adders**: Regional, based on published interconnection queue data and MISO/PJM/CAISO tariff filings

### 20.4 What Our Model Does NOT Include (Scope Boundaries)

- **Transmission expansion or congestion** — we use existing interconnection costs
- **Retirement/entry decisions** — we take the existing grid as a given
- **Inter-regional trade / import-export** — each ISO is modeled as self-contained. Unmatched demand hours are assumed met by fossil generation priced at regional fossil cost sensitivities (coal/gas/oil). We do not consider interconnection or power flows across grid boundaries. This is a meaningful simplification for ISOs that rely on imports (e.g., CAISO imports from Pacific NW hydro, NYISO imports from Quebec/Ontario hydro). The effect is that our model may slightly overstate the difficulty of meeting high CFE thresholds in import-dependent regions.
- **Demand response or demand flexibility** — demand is fixed hourly profile
- **Hydrogen storage** — explicitly excluded (immature for grid-scale energy storage)
- **Multi-year capacity planning** — single 2025 snapshot, not a trajectory
- **Reliability/adequacy constraints (ELCC)** — under consideration (Section 21.1)
- **NEISO gas delivery constraints** — New England has well-documented natural gas pipeline constraints (Algonquin pipeline congestion during winter peaks). This creates winter gas price spikes that aren't captured by our flat L/M/H gas price sensitivity. Future iteration should model seasonal gas price multipliers or a NEISO-specific winter gas adder. See §21.3.
- **BECCS (Bioenergy with CCS)** — Not modeled in current version. Relevant for NEISO where high CCS shares (50%+ at 92.5%) suggest a natural use case. BECCS could offer negative emissions AND firm dispatchable generation. Future post-processing: derate CCS scenarios with a BECCS cost overlay to avoid full re-optimization. See §21.3.

---

## 21. Planned Enhancements

### 21.1 Capacity Reserve Margin / ELCC (Under Consideration)

**Concept**: Layer in a capacity reserve margin constraint using Effective Load Carrying Capability (ELCC) to ensure resource mixes maintain grid reliability.

**What ELCC does**: ELCC measures the firm capacity contribution of each resource type — how much peak demand it can reliably serve. Variable resources (solar, wind) have lower ELCC than their nameplate capacity because they may not generate during peak demand hours.

**Typical ELCC values** (from NREL/regional ISO studies):
| Resource | ELCC Range | Notes |
|---|---|---|
| Clean Firm (nuclear) | 90-95% | Near-firm, planned outages reduce |
| Solar | 30-70% | Varies by region, declines with penetration |
| Wind | 10-30% | Highly region-dependent |
| CCS-CCGT | 85-95% | Dispatchable, similar to CCGT |
| Battery (4hr) | 60-95% | Duration-limited; declines as peak broadens |
| LDES (100hr) | 85-95% | Long duration → high capacity value |

**Implementation approach**: Add a constraint to the optimizer that the ELCC-weighted capacity of the resource mix must meet a minimum reserve margin (e.g., 15% above peak demand). This would:
- Prevent resource mixes that meet hourly matching targets but lack capacity adequacy
- Penalize solar-heavy mixes at high thresholds (solar ELCC drops with penetration)
- Favor firm resources and storage at the margin
- Better reflect real planning constraints

**Complexity**: Moderate. The ELCC calculation is a post-hoc check on each candidate mix during optimization. The main challenge is ELCC values that decline with penetration (saturation effects), which creates non-linear constraints. A simplified version could use fixed ELCC percentages per resource type.

**Decision**: Under consideration — user to confirm whether to implement for next optimizer run.

### 21.2 Post-Processing from Cached Results (Decided — Future Iterations)

**Principle**: After monotonicity sweeps are complete, use cached results (resource mixes + score caches + optimizer_cache.json) for future post-processing instead of full re-optimization runs. This enables:
- Rapid sensitivity analysis (re-price existing mixes under new cost assumptions)
- BECCS overlays (derate CCS scenarios with BECCS costs)
- Gas constraint scenarios (apply winter price multipliers to existing mixes)
- Carbon price sensitivity (overlay SCC/ETS prices on existing results)

**How**: `optimizer_cache.json` stores the full co-optimized results for all 16,200 scenarios. `compute_costs_parameterized()` can re-price any cached mix in milliseconds. Only changes that fundamentally alter the optimization landscape (new resource types, new dispatch algorithms, new constraint structures) require full re-runs.

**Two-file architecture** (decided):
1. **`data/optimizer_cache.json`** — Raw, untouched optimizer output. Never modified after a run. This is the canonical record of what the optimizer produced. Includes all resource mixes, costs, scores, metadata.
2. **`dashboard/overprocure_results.json`** — Post-processed copy that feeds the live dashboard. Derived from the cache + any post-processing overlays (BECCS derating, gas adjustments, etc.). Can always be regenerated from the cache.

Any post-processing (cost overlays, BECCS, gas constraints, carbon pricing) operates on a copy of the cache data and writes to the dashboard results file. The raw cache is always preserved as ground truth.

### 21.3 NEISO Gas Pipeline Constraints + BECCS (Future Iteration)

**NEISO Gas Delivery Constraints**: New England has severe natural gas pipeline constraints, particularly on the Algonquin City Gate pipeline during winter peaks. Key literature:
- **ISO-NE Operational Fuel Security Analysis (2018)**: Documented reliability risk from winter gas constraints; gas generators unable to secure fuel during cold snaps
- **Algonquin basis differentials**: Winter spot gas prices in New England can spike to $20-30/MMBtu (vs. $3.50 Henry Hub medium), reflecting pipeline congestion
- **Grid-scale impact**: During 2017-2018 "bomb cyclone", New England gas prices exceeded $30/MMBtu, oil generation surged to 30%+ of total
- **Current model limitation**: Our flat L/M/H gas price sensitivity ($2/$3.50/$6 MMBtu) doesn't capture this seasonal volatility. The "High" gas scenario ($6) still understates winter peaks by 3-5x.

**Potential fix for future iteration**: Apply a NEISO-specific seasonal gas price multiplier (e.g., 3-5x during Dec-Feb) or model a winter gas constraint that caps gas-fired generation availability. This would increase the value of non-gas firm resources (nuclear, BECCS) and storage in NEISO.

**BECCS for NEISO**: Current optimizer shows NEISO needs 50%+ CCS at 92.5% matching. This creates a natural use case for BECCS (Bioenergy with CCS):
- BECCS provides firm dispatchable generation (like CCS-CCGT) PLUS negative emissions
- NEISO has significant forestry biomass resource (wood pellets, forestry residues)
- Cost estimate: ~$120-180/MWh LCOE (NREL ATB) — higher than CCS-CCGT but with carbon-negative value
- **Post-processing approach**: For scenarios with high CCS share (>25%), run a cost overlay replacing a fraction of CCS with BECCS pricing. Include negative emissions credit at SCC values ($185–190/tCO2 — Rennert et al. / 2024 EPA central). This avoids full re-optimization — just re-price cached mixes.

**Decision**: Implemented in post-processing (Feb 15, 2026). See §22.

### 21.4 Offshore Wind — New Resource Dimension (Planned, Mar 2, 2026)

**Rationale**: Onshore wind is 22–25% CF with strong diurnal swing. Offshore wind at ~47% CF with a flat diurnal profile fundamentally changes the cost curve for 80%+ thresholds — less storage needed to cover overnight gaps, less VRE overbuild required. This is material for Atlantic ISOs.

#### 21.4.1 Capacity Caps (TWh)

| ISO | Cap (TWh) | Rationale |
|-----|-----------|-----------|
| NYISO | 37 | 9 GW pipeline (Empire Wind, Sunrise Wind, etc.) |
| NEISO | 37 | 9 GW capacity (Vineyard Wind, Revolution Wind, etc.) |
| PJM | 30 | NJ 7.5 GW mandated + DE/MD/VA pipeline |
| CAISO | 20 | ~5 GW (Morro Bay + Humboldt WEAs) |
| ERCOT | 0 | No meaningful offshore resource |
| MISO | 0 | No meaningful offshore resource |
| SPP | 0 | No meaningful offshore resource |

#### 21.4.2 Profile Shape (8760)

**Data source**: NREL NOW-23 (National Offshore Wind dataset) — 2 km grid, hourly wind speeds, 2000–2020 (21 years).

**Reference turbine**: IEA 15 MW (150 m hub height, 240 m rotor diameter). Power curve CSV from NREL/IEA Wind Task 37 GitHub.

**Build process**:
1. Extract wind speeds at 140 m + 160 m from NOW-23 at lease area coordinates
2. Interpolate to 150 m hub height (linear between heights)
3. Apply IEA 15 MW power curve (wind speed → capacity factor)
4. Apply loss stack: wake (~10%) × electrical (~2.5%) × availability (~5%) = **net ~83.4% of gross**
5. Average across 5 years (2016–2020) — same methodology as EIA onshore profiles
6. Normalize profile to sum = 1.0 (consistent with all other generation profiles)

**Representative coordinates per ISO**:

| ISO | Location |
|-----|----------|
| NYISO | NY Bight (Empire Wind lease area) |
| NEISO | Vineyard Wind area (south of Martha's Vineyard) |
| PJM | NJ lease areas (Atlantic Shores / Ocean Wind) |
| CAISO | Morro Bay WEA |

#### 21.4.3 Variability Calibration

The 5-year average preserves:
- **Seasonal swing** (winter peak in NE, summer peak in CA)
- **Multi-day weather patterns** (storm cycles, lull periods)
- **Flat diurnal envelope** (no day/night swing like onshore — key differentiator)
- **Realistic zero-generation hours** (~20% of hours for offshore)

It smooths:
- **Interannual anomalies** (e.g., whether 2018 was unusually windy)

**Validation targets**: Compare resulting annual CF against South Fork Wind actual (46.4%) and NREL ATB projections (~49%). This is the **exact same treatment** as onshore wind and solar — the offshore profile naturally shows higher CF and flatter diurnal, but still has weather-driven gaps. The optimizer handles it as VRE, not firm.

#### 21.4.4 Integration in the Pipeline

**Step 0** — New `step0_fetch_offshore_wind.py`:
- Fetches NOW-23 data via NREL API (requires API key from developer.nrel.gov)
- Applies IEA 15 MW power curve + loss stack
- Generates normalized 8760 profiles per ISO
- Output: `data/offshore_wind_profiles/` (one parquet per ISO)

**Step 1** — `offshore_wind` as new resource dimension:
- NYISO, NEISO, PJM, CAISO: 5D grid search (clean_firm, solar, wind, **offshore_wind**, hydro) — analogous to CAISO's 5D with geothermal
- CAISO becomes 6D (clean_firm, solar, wind, offshore_wind, hydro, geothermal)
- ERCOT, MISO, SPP: remain 4D (no offshore resource)
- Offshore wind grid levels: [0, 5, 10, 15, 20, 25, 30]% of demand (capped by ISO TWh limits above)

**Step 3** — Cost tables:
- New `OFFSHORE_WIND_LCOE` table (L/M/H by ISO). Range: ~$80–150/MWh (higher than onshore $30–95 reflecting fixed-bottom/floating costs)
- New `OFFSHORE_WIND_TX` transmission adder table (submarine cable + grid interconnection)
- Shares `Renewable Gen` sensitivity toggle pairing with solar + onshore wind

**Dashboard** — Toggle and display:
- Offshore wind appears as a distinct resource in mix charts (new color in `chart-colors.js`)
- Capacity caps shown in methodology page

#### 21.4.5 Cost Tables (Finalized, Mar 2, 2026)

**LCOE tables ($/MWh)** — shares `Renewable Gen` sensitivity toggle with solar + onshore wind:

| Level | NYISO | NEISO | PJM | CAISO (floating) | Sources |
|-------|-------|-------|-----|---------|---------|
| Low | 72 | 68 | 65 | 110 | Lazard v17 low, NREL ATB Advanced |
| Medium | 95 | 90 | 85 | 150 | BNEF 2025, NREL ATB Moderate |
| High | 125 | 118 | 112 | 200 | NREL FORCE model, supply-chain stress |

Regional hierarchy: PJM cheapest (shallowest, closest to shore, NJ 7.5 GW pipeline), NEISO mid (Vineyard Wind precedent, 51% CF), NYISO most expensive East Coast (NY Bight permitting + Jones Act), CAISO dramatically higher (floating, no US commercial experience).

**Transmission tables ($/MWh)** — submarine cable + offshore substation:

| Level | NYISO | NEISO | PJM | CAISO |
|-------|-------|-------|-----|-------|
| None | 0 | 0 | 0 | 0 |
| Low | 8 | 7 | 6 | 10 |
| Medium | 15 | 13 | 11 | 20 |
| High | 25 | 22 | 18 | 35 |

**Wright's Law learning curves** — two separate curves for fixed-bottom vs floating:

Fixed-bottom (NYISO, NEISO, PJM):
- FOAK: 1.15× High (Vineyard Wind era, supply chain stress). NYISO $144, NEISO $136, PJM $129.
- NOAK: Low values (post-learning equilibrium). NYISO $52-88, NEISO $50-85, PJM $50-82.
- Learning rate: ~8.8% per capacity doubling (NREL ATB 2024 Moderate). Global base: 83 GW.
- Window: L=(2026,2034), M=(2028,2038), H=(2032,2045)

Floating (CAISO):
- FOAK: 1.25× High (pre-commercial, no US floating experience). $250/MWh.
- NOAK: DOE Wind Shot aligned. $55-100/MWh.
- Learning rate: ~11.5% per capacity doubling (NREL ATB Moderate). Global base: 0.3 GW (nascent). Multiple doublings ahead.
- Window: L=(2029,2037), M=(2031,2042), H=(2035,2050)

**Capacity factor**: NYISO 0.49, NEISO 0.51, PJM 0.48, CAISO 0.43 (from NOW-23 profiles).
**Peak capacity credit**: 0.25 (higher than onshore wind 0.10 — flatter profile, less correlated with system peak).

#### 21.4.6 NOW-23 API Details (Research Complete, Mar 2, 2026)

**Regional endpoints** — NOW-23 uses separate API endpoints per region:

| Region | API Path | ISOs Served |
|--------|----------|-------------|
| North Atlantic | `offshore-north-atlantic-download` | NEISO |
| Mid Atlantic | `offshore-mid-atlantic-download` | NYISO, PJM |
| California | `offshore-ca-download` | CAISO |

**Base URL**: `https://developer.nrel.gov/api/wind-toolkit/v2/wind/{endpoint}.{format}`

**Key parameters**:
- `api_key` — free from developer.nrel.gov/signup
- `wkt` — WKT geometry, e.g., `POINT(-74.5 39.5)` (longitude first!)
- `attributes` — e.g., `windspeed_140m,windspeed_160m`
- `names` — year, e.g., `2020`
- `interval` — `60` for hourly
- `utc` — `true`
- `email` — required for async requests

**Available heights**: 10m, 20m, 40m, 60m, 80m, 100m, 120m, **140m**, **160m**, 180m, 200m, 220m, 240m, 260m, 280m, 300m, 400m, 500m. Both 140m and 160m confirmed available — linear interpolation to 150m hub height.

**Coverage**: 2000–2020 (21 years) for Atlantic regions; 2000–2019 (20 years) for California (API returns 400 for 2020+).

**Rate limits**: CSV format = 10,000 requests/day, 1/second. Each request = 1 point × 1 year. 5 years × 4 ISOs = 20 requests total — well within limits.

**Bulk alternative**: AWS S3 at `s3://nrel-pds-wtk/` (no account needed). Also accessible via NREL's HSDS service with `h5pyd` or `NREL-rex` packages.

#### 21.4.7 IEA 15 MW Power Curve (Research Complete, Mar 2, 2026)

**Source**: `turbine-models` Python package (`pip install turbine-models`), file `Offshore/IEA_Reference_15MW_240.csv`. Also on GitHub: `github.com/NREL/turbine-models`.

**Key turbine specs**:
| Parameter | Value |
|-----------|-------|
| Rated Power | 15 MW |
| Rotor Diameter | 240 m |
| Hub Height | 150 m |
| Cut-in Wind Speed | 3 m/s |
| Rated Wind Speed | 10.59 m/s |
| Cut-out Wind Speed | 25 m/s |
| IEC Class | IB |
| Design Cp | 0.489 |

**Power curve summary** (59 data points from 3–25 m/s):
- 3 m/s: 70 kW (cut-in)
- 7 m/s: 4,339 kW
- 10 m/s: 12,661 kW
- 10.59 m/s: ~14,995 kW (rated)
- 10.6–25 m/s: 14,995 kW (constant, pitch-controlled)
- 25 m/s: 14,998 kW → cut-out

Full CSV with 59 wind speed × power × Cp × thrust data points available in the installed package.

#### 21.4.8 Steps 4–7 + Dashboard Integration (Decided, Mar 3, 2026)

**Resource display order** (user-confirmed): Nuclear → Geothermal → Hydro → CCS → Offshore Wind → Onshore Wind → Solar → Battery 4 → Battery 8 → LDES → H2. Internal `RESOURCE_TYPES` in dispatch_utils.py keeps processing order; display order applied at presentation layer (Step 7 + dashboard JS).

**Color palette updates** (user-confirmed):
| Resource | Old Color | New Color | Hex |
|----------|-----------|-----------|-----|
| Nuclear (clean_firm) | Dark Navy `#1E3A5F` | Indigo 500 | `#6366F1` |
| CCS-CCGT | Cyan `#0891B2` | Slate | `#64748B` |
| Offshore Wind | (new) | Material Teal | `#009688` |
| Geothermal | Green `#10B981` | Ochre | `#B45309` |

**FEASIBLE_MIXES positional array**: 12 elements in display order:
`[clean_firm, geothermal, hydro, ccs_ccgt, offshore_wind, wind, solar, score, bat4, bat8, ldes, h2]`

**Resource cap integration scope** (user-confirmed): Geothermal, CCS, and offshore wind TWh caps propagated into:
- Scenario A (step5_scenario_a_consequential.py) — floor ratchet respects caps as upper bounds
- Scenario B (step5_scenario_b_hourly.py) — hourly matching cap enforcement
- Scenario comparison (step5_scenario_comparison.py)
- Procurement Strategies 1–3 (step6_5_strategy1/2/3)
- Track 2 New-Build (step3_track_nb_ctr.py)
- Track 3 Cost-to-Replace (step3_track_nb_ctr.py)

**CCS cap table** (geological sequestration storage, from USGS/NETL):
| ISO | Cap (TWh) | Rationale |
|-----|-----------|-----------|
| ERCOT | 85 | Gulf Coast saline formations + depleted O&G reservoirs |
| PJM | 120 | Appalachian Basin + Midcontinent saline formations |
| MISO | 95 | Illinois Basin + Gulf Coast formations |
| SPP | 110 | Anadarko Basin + Permian Basin saline formations |
| NYISO | 15 | Limited NY/NJ offshore saline capacity |
| NEISO | 10 | Very limited NE geological storage |
| CAISO | 0 | No significant identified storage (seismic risk) |

**Dashboard cap export**: `RESOURCE_CAPS` JS constant in shared-data.js containing all three cap dicts (offshore_wind, ccs_ccgt, geothermal).

**dispatch_utils.py changes**:
- `RESOURCE_TYPES` expanded from 5 → 6 (add `offshore_wind`)
- `OFFSHORE_ISOS`, `OFFSHORE_WIND_CAP_TWH`, `CCS_CAP_TWH`, `GEOTHERMAL_CAP_TWH` constants
- `get_supply_profiles()` loads offshore wind profile (zeros for non-offshore ISOs)
- `build_supply_matrix()` builds (6, H) matrix
- `reconstruct_hourly_dispatch()` adds matched/surplus offshore_wind arrays
- `CACHE_VERSION` → v3 (old v2 caches rebuilt)

**Backward compat**: All parquet loading defaults `mix_offshore_wind` to 0 when column is missing.

#### 21.4.8 Implementation Approach (Decided)

**Recommended approach** (simplest, no HSDS complexity):
1. Use developer.nrel.gov CSV endpoint — one request per point per year
2. Request `windspeed_140m` + `windspeed_160m` at `interval=60`
3. Linear interpolation to 150m hub height
4. Apply IEA 15MW power curve via `numpy.interp()` on the 59-point CSV
5. Apply loss stack (wake 10% × electrical 2.5% × availability 5% = net 83.4%)
6. Average 5 years (2016–2020), normalize to sum = 1.0

**Python packages needed**: `requests`, `numpy`, `pandas` (all already in the project). The `turbine-models` package provides the power curve CSV but isn't needed at runtime — we'll embed the 59-point curve directly in the script.

#### 21.4.9 Blockers

1. **NREL API key** required for NOW-23 data access — sign up at developer.nrel.gov/signup
2. **Cost table finalization** — need NREL ATB 2024 offshore wind LCOE by region + Lazard cross-check
3. **Step 1 compute impact** — adding a 5th/6th dimension increases grid search combinatorics significantly. May need aggressive pruning or adaptive grid for offshore ISOs.

#### 21.4.10 Why This Matters

Offshore wind at 47% CF with flat diurnal is a **qualitatively different resource** from onshore wind:
- Onshore: low CF, strong diurnal → needs massive overbuild + storage for 80%+ matching
- Offshore: high CF, flat diurnal → approaches dispatchable VRE characteristics
- At 90%+ thresholds, offshore wind could displace significant clean firm / storage need in Atlantic ISOs
- The cost question is whether the $80–150/MWh LCOE premium over onshore ($30–95) is offset by reduced storage and overbuild needs — this is exactly what the optimizer will answer

### 21.5 Weather-Year Sensitivity Analysis (Added Mar 2026)

**Concept**: Run Step 1 with individual per-year profiles (2021, 2022, 2023, 2024, 2025) instead of the 5-year average. Compare optimal resource mixes and costs across weather years to produce P10/P50/P90 confidence intervals.

**Value**: Quantifies the sensitivity of results to weather variability. Particularly important for high thresholds (≥95%) where extreme weather drives the marginal resource requirement. Would also reveal which ISOs are most weather-sensitive.

**Cost**: 5× Step 1 compute (one per weather year). Could be done incrementally: run worst-case year first (highest variability), add others as time permits.

### 21.6 Demand Response and Demand-Side Resources (Added Mar 2026)

**Concept**: Add demand flexibility as a virtual resource that can shift load to better match clean supply profiles. Model DR as a dispatchable resource with capacity limits, shift duration constraints, and associated costs.

**Value**: DR/EV flexibility could reduce procurement costs 5-15% at high thresholds by shifting demand to match clean supply profiles. Particularly relevant for corporate buyers with operational flexibility (data centers, industrial loads).

**Complexity**: Moderate — requires demand elasticity parameters and shift window constraints. Could start with a simplified model (e.g., 10% demand shiftable within 4-hour window at $X/MWh).

### 21.7 Hourly Marginal Emission Rate Model (Added Mar 2026)

**Concept**: Replace scalar per-threshold emission rates with hourly marginal emission rate approximation. Use the dispatch stack model to determine which fossil fuel is at the margin in each hour, then apply the appropriate emission factor.

**Value**: More accurate CO₂ abatement accounting. Captures the fact that clean energy displacing nighttime coal is more valuable (per tCO₂) than displacing daytime gas.

**Implementation**: Compute hourly fossil residual from dispatch cache, apply merit-order stack to determine marginal fuel, assign fuel-specific emission factor per hour.

---

## 22. Post-Processing Corrections & Overlays (Feb 15, 2026)

Applied to Step 3 cost optimization results via `step4_postprocess.py`. Corrected results written to `dashboard/overprocure_results.json`.

### 22.1 CO₂ Monotonicity Enforcement

**Problem**: CO₂ abatement is non-monotonic across thresholds in most ISOs. Higher hourly match targets can result in LESS CO₂ abated (up to -15.3M tons in ERCOT 90%→92.5%). Root cause: the optimizer minimizes cost, not CO₂. A cheaper mix at a higher threshold may procure less total clean energy (substituting storage for overprocurement), reducing total fossil displacement even as temporal matching improves.

**Fix**: Running-max constraint — `co2_corrected[t] = max(co2[t], co2[t-1])` across thresholds. Ensures abatement narrative never shows "paying more for less CO₂."

### 22.2 45Q Offset Correction

**Problem**: Model calculates 45Q credit as $85/ton × 0.34 tCO₂/MWh = $29/MWh. Correct calculation: $85 × 0.34 × 0.95 (captured only) = $27.5/MWh. Overstated by ~$1.5/MWh.

**Fix**: Adjust CCS LCOE by +$1.5/MWh across all scenarios. Negligible impact on results.

### 22.3 Without-45Q Toggle Layer

**Design**: Dashboard toggle "45Q Credit: On / Off" showing cost impact of removing the 45Q incentive from CCS-CCGT.

**Without-45Q CCS cost model**:
- Remove $27.5/MWh 45Q offset from CCS LCOE
- Model CCS as **dispatchable** (not baseload) — without 45Q, there's no perverse incentive to maximize captured CO₂ by running 24/7
- CCS LCOE becomes **capacity-factor-dependent**: at lower CF, capital recovery per MWh increases

**CCS LCOE decomposition** (from NETL Baseline Rev 4a):
- Capital recovery: 55% of LCOE (scales inversely with CF)
- Fixed O&M: 8% of LCOE (scales inversely with CF)
- Fuel: 30% of LCOE (constant per MWh)
- Variable O&M + T&S: 7% of LCOE (constant per MWh)
- Reference CF: 85% (NETL standard)

**CF-dependent formula**: `LCOE(CF) = LCOE_no45q × ((0.63 × 0.85 / CF_actual) + 0.37)`

**CCS vs LDES crossover**: At each region's Medium costs, the CF at which CCS-without-45Q equals LDES cost. Below this CF, LDES is cheaper. This determines whether CCS would ever be built without 45Q.

**Implementation**: For each cached scenario, recalculate costs assuming no 45Q. CCS mix share implies an effective CF that determines the dispatchable LCOE. Compare to what the cost would be if CCS share were replaced by LDES or additional clean firm.

### 22.4 NEISO Winter Gas Pipeline Constraint

**Problem**: NEISO has structural winter gas price spikes due to Algonquin Citygates pipeline congestion. Winter spot prices historically $15-30/MMBtu vs. ~$5-6/MMBtu annual average. The model's flat L/M/H gas sensitivity ($2/$3.50/$6 MMBtu) understates NEISO winter costs by 3-5×.

**Post-processing approach**:
- Winter months (Dec-Feb, ~25% of year): +$7.50/MMBtu above annual average (midpoint of $5-10 range)
- CCS fuel impact: 7 MMBtu/MWh heat rate × $7.50 × 0.25 = **+$13.13/MWh annualized CCS adder** for NEISO
- Wholesale impact: gas-on-margin × winter premium → **+$4/MWh annualized wholesale adder** for NEISO
- Applied to NEISO results only; all other ISOs unaffected

**Sources**: ISO-NE Operational Fuel Security Analysis (2018), Algonquin Citygates historical basis differentials, 2017-2018 bomb cyclone gas pricing data.

### 22.5 ERCOT Battery LCOS Low ($69/MWh) — Retained

**Finding**: The $69/MWh ERCOT Low battery LCOS lacks a peer-reviewed citation. It was set based on regional qualitative factors (low labor costs, fast permitting, flat terrain, minimal unionization, extensive solar co-location potential).

**Decision**: Retain $69/MWh. ERCOT is genuinely the lowest-cost US market for battery deployment. Lazard's national unsubsidized range ($115-$254) reflects high-cost assumptions (80% equity at 12% return) and diverse geographies. ERCOT-specific conditions (non-ERCOT interconnection queue, streamlined permitting, LFP oversupply benefiting Texas ports) justify costs below national averages. The Low case explicitly represents an optimistic-but-plausible scenario.

**Mitigation**: Document in research paper that regional battery cost differentiation is based on qualitative assessment of market conditions, not published regional cost studies. Note that all Low-case costs represent aggressive forward trajectories.

### 22.6 Post-Processing Peer Review Fixes (Feb 15, 2026)

**Findings from third-party code review:**

1. **`costs_detail` sync** — `fix_45q_offset()` was updating `scenario['costs']` but not `scenario['costs_detail']` for Medium scenarios (MMM_M_M), causing a data inconsistency where the dashboard's detail views showed stale pre-correction numbers. **Fixed**: Now syncs `effective_cost_per_useful_mwh`, `total_cost_per_demand_mwh`, `incremental_above_baseline`, and `baseline_wholesale_cost` between both dicts.

2. **Crossover edge-case comment** — When `rhs ≤ 0` (LDES variable cost alone exceeds LDES cost), the comment incorrectly stated "CCS always cheaper." **Fixed**: Corrected to "LDES always cheaper."

3. **Dead import** — `import copy` was unused. **Removed.**

### 22.7 Gas Availability Factor (GAF) — Resource Adequacy Deration (Feb 20, 2026)

**Problem**: The model assumed 100% gas availability at peak — if gas backup = 10,000 MW needed, exactly 10,000 MW was built. This contradicts all ISO practice and empirical evidence. Gas plants experience both independent forced outages (EFORd ~5-7%) and correlated failures during extreme weather events (Winter Storm Uri: 49% outage; Elliott: 24% outage). PJM's 2024 ELCC methodology rates gas CCGT at ~80% effective capacity.

**Fix**: Divide raw gas backup requirement by an ISO-specific Gas Availability Factor (GAF):
```
gas_needed_mw = max(0, ra_peak - clean_peak) / GAF
```

**ISO-specific GAF values** (applied in both Step 3 cost optimization and Step 4 post-processing):

| ISO | GAF | Deration | Rationale |
|-----|-----|----------|-----------|
| CAISO | 0.88 | 12% | Summer ambient derate + mechanical outages |
| ERCOT | 0.83 | 17% | Extreme weather both seasons, gas supply correlation |
| PJM | 0.82 | 18% | PJM ELCC data, Winter Storm Elliott evidence |
| NYISO | 0.82 | 18% | Pipeline constraints, winter gas competition |
| NEISO | 0.85 | 15% | Mechanical + weather only (pipeline handled separately) |

**NEISO note**: GAF captures only mechanical/weather unavailability. The pipeline capacity constraint is structurally different — an absolute MW ceiling, not a proportional derate — and is modeled separately (see §22.8).

**Sources**: PJM ELCC Class Ratings (2024/25), NERC GADS EFORd class averages, FERC Final Reports on Winter Storm Uri (2021) and Elliott (2022), Brattle Group VRR Curve Review (2025), UCS gas reliability analyses, ERCOT Aurora RA Assessment (2025).

**Impact on optimization**: GAF increases gas backup MW requirements by 12-18% across ISOs, which increases gas backup costs. This tilts cost-optimal mixes toward resources with higher peak capacity credits (clean firm, CCS, battery) and away from resources with low capacity credits (solar, wind) at high matching thresholds. The effect is modest at low thresholds (gas backup is small) and material at 95%+ (where gas backup costs are a significant fraction of total cost).

### 22.9 Delta RA Approach — Calibrated Gas Backup (Feb 25, 2026)

**Problem**: The previous RA formula `gas_needed = max(0, ra_peak - clean_peak) / GAF` computed clean_peak from energy allocations (`proc * pct / 100 * avg_demand_mw * CC`), which conflates average generation MW with installed capacity. This systematically underestimated clean peak contribution (by 2-4x for solar/wind) because it didn't convert energy → installed MW using capacity factors. Result: ERCOT 2025 showed ~94 GW gas needed when only 55 GW exists.

**Fix**: Delta RA approach calibrated to 2025 reality:
1. At base year (2025): `gas = EXISTING_GAS_CAPACITY_MW` (calibrated to actual installed fleet)
2. Compute `EXISTING_CLEAN_PEAK_MW` from 2025 fleet using `avg_mw / capacity_factor * capacity_credit`
3. Compute `GAS_RAW_2025 = max(0, RA_peak - EXISTING_CLEAN_PEAK_MW) / GAF` as theoretical baseline
4. For any scenario: `gas_raw = max(0, ra_peak_grown - total_clean_peak) / GAF`
5. `gas_delta = gas_raw - GAS_RAW_2025`
6. `total_gas = max(0, EXISTING_GAS + gas_delta)`

**New-build peak uses capacity factor conversion**: `installed_mw = avg_generation_mw / CF[resource][iso]` then `peak_mw = installed_mw * CC[resource]`. This properly accounts for wind's low CF (high installed per MWh) and solar's low CF.

**Resource capacity factors (EIA Form 923, eGRID 2022-2024)**:
| Resource | CAISO | ERCOT | PJM | NYISO | NEISO |
|----------|-------|-------|-----|-------|-------|
| Nuclear  | 0.90  | 0.93  | 0.93| 0.90  | 0.90  |
| Solar    | 0.28  | 0.24  | 0.17| 0.15  | 0.15  |
| Wind     | 0.25  | 0.38  | 0.30| 0.28  | 0.30  |
| CCS-CCGT | 0.85  | 0.85  | 0.85| 0.85  | 0.85  |
| Hydro    | 0.40  | 0.30  | 0.35| 0.40  | 0.40  |

**Peak demand growth**: Peak scales with demand growth factor (`peak_grown = PEAK_2025 * gf`).

**Result**: ERCOT 50% at 2030 Medium growth now shows ~71 GW gas (down from >100 GW), consistent with real-world expectations of 65-70 GW for modest clean energy expansion from a 46% clean baseline with ~53 GW existing gas.

### 22.8 NEISO Pipeline Capacity Constraint — Informational Metric (Feb 20, 2026)

**Problem**: NEISO's gas constraint is an absolute physical ceiling (~4.5 BCF/day total pipeline capacity; ~1.5 BCF/day available for power generation after heating demand during winter peak), not a proportional deration. Building more gas plants doesn't help if the pipeline can't feed them. As demand grows, the constraint worsens (same pipeline, more load).

**Approach**: Compute as a downstream informational metric, NOT integrated into the optimization. For each NEISO scenario:
1. Compare gas backup MW (post-GAF) against pipeline-deliverable gas MW ceiling (8,300 MW)
2. If gas exceeds pipeline capacity: compute shortfall MW and annualized pipeline expansion cost
3. Output as `pipeline_constraint` sub-dict in gas_backup results

**Constants**:
- Pipeline-deliverable gas at peak: **8,300 MW** (1.5 BCF/day ÷ 7.5 MMBtu/MWh heat rate)
- Pipeline expansion cost: **$2,400/MW-yr** annualized ($150M/BCF-day, 30yr at 8% WACC)

**Source**: ISO-NE Gas Availability Study (2025), FERC pipeline project filings.

**Rationale for informational-only**: The pipeline constraint is structural and binary — scenarios either exceed the ceiling or don't. Baking it into the optimization would distort mix selection by treating a New England infrastructure policy question as an engineering parameter. Instead, it's presented as: "this scenario requires X MW of gas backup, but the pipeline can only deliver 8,300 MW — here's what closing that gap would cost."

4. **CCS CF estimation floor** (documented limitation) — The 0.20 minimum CF floor in `ccs_lcoe_dispatchable()` may understate no-45Q costs for small CCS shares (where actual CF might be 0.08-0.15). Without hourly dispatch data in the results JSON, we can't improve this in post-processing. Documented as a conservative (cost-understating) assumption.

5. **No-45Q mix bias** (documented limitation) — The no-45Q overlay reprices the same resource mix that was co-optimized WITH 45Q. This mix over-represents CCS, making the no-45Q cost a conservative upper bound. A true no-45Q re-optimization would substitute LDES/renewables for CCS, yielding lower costs.

### 22.10 ≥99.9% Hourly Match Asymptote — Literature Review & Procurement Bounds

**Decision (Feb 2026):** Top threshold lowered from 100% to ≥99.9%. True 100% hourly matching is physically unreachable due to float precision and dispatch constraints. ≥99.9% is labeled "effectively 100%" (8.76 unmatched hours/year). This makes the threshold honest — we label what we can actually achieve.

**Key literature findings:**
- NREL (Cole et al., 2021, Joule): Marginal abatement cost 99%→100% = **$930/ton** — 15× the average cost of the full 100% target. Nonlinear in all 22 sensitivities tested.
- Riepin & Brown (2024, Energy Strategy Reviews): 98% CFE = 54% premium over annual matching. 100% doubles costs again. With clean firm + LDES, 100% premium drops to just 15%.
- Peninsula Clean Energy MATCH Model (2023): 99%→100% requires **34% more supply**, +10% portfolio cost. 0%→99% costs only +2%.
- Budischak et al. (2013, J. Power Sources): Cost-optimal 99.9% requires ~280% nameplate capacity. "Least cost solutions yield seemingly-excessive generation capacity."
- WattTime: 100% hourly matching may require PPAs for **up to 400%** of annual consumption.

**Granularity consensus:** The 90–≥99.9% zone needs 2.5% resolution minimum. Our threshold set (90, 92.5, 95, 97.5, 99, 99.5, 99.9) is well-aligned with literature practice.

**Procurement bound assessment:**
- Current bound: 200% of demand
- Actual usage at 99%: max 135% (CAISO), 130% (NYISO), 125% (NEISO), 123% (PJM), 118% (ERCOT)
- ≥99.99% threshold (now dropped): 0 feasible scenarios found (all ISOs) at 200% bound — this was a key reason for dropping 99.99% in v4.3.
- Max hourly match achieved: 99.6% (PJM at 123% procurement)
- **Decision**: The 200% bound is sufficient for ≤99.9% targets. The old 99.99% threshold would have required 250%+ bounds (Budischak 280%, WattTime 400%).

**Archetype diversity in cache:**
- 46–70 unique resource mix archetypes per ISO across all thresholds
- Only 4–14 unique mixes per threshold (massive redundancy across 5,832 scenarios)
- Cache comprehensively covers the feasible solution space — new constraint runs can seed from existing archetypes rather than cold-start

### 22.11 Step 1 PFS Improvement Opportunities (Feb 21, 2026)

**Constraint: No changes may sacrifice the ability to find the full PFS.** All improvements below are backward-compatible — they improve speed and/or coverage without changing the feasible space definition or dispatch physics.

**Post-process script**: `postprocess_storage_resweep.py` — standalone Numba parallel re-sweep that runs between Step 1 and Step 2. Demonstrates patterns 1, 4, and 8 below. Uses `@njit(parallel=True)` with `prange` to batch-evaluate near-miss mixes across CPU cores. Checkpoints to `data/resweep_checkpoints/resweep_progress.parquet` after each ISO×threshold.

#### 1. Numba Parallel Storage Sweep (High Impact — 4-8× speedup)
**Current**: Step 1 Phase 1b evaluates storage combos sequentially per mix — one mix at a time through 342 storage configs × N procurement levels.
**Improvement**: Wrap `_score_with_all_storage` in `@njit(parallel=True)` with `prange` across mixes. All near-miss mixes at a single (procurement, storage) config are evaluated simultaneously across CPU cores.
**Pattern**: `batch_score_storage()` in `postprocess_storage_resweep.py` — takes `(demand, supply_rows[N, 8760], procurement, N, storage_params...)`, returns `scores[N]` via `prange`.
**Impact**: Phase 1b is 60-80% of Step 1 runtime. Multi-core parallel cuts this proportional to available cores (typically 4-8× on modern machines).

#### 2. Consistent Scoring Metric (Correctness)
**Current**: Two different scoring metrics used within Step 1:
- Phase 1a (no storage): `np.sum(np.minimum(supply / demand, 1.0)) / H` — hourly average match fraction (weights all hours equally)
- Phase 1b (with storage): `sum(min(demand[h], supply[h]))` — total energy match fraction (weights by demand magnitude)
**Problem**: These produce different scores for the same mix. A mix could pass the no-storage check but fail the storage check (or vice versa) at the same threshold. The PFS mixes `hourly_match_score` column contains a mix of both metrics.
**Fix**: Unify to the energy metric (`sum(min(demand, supply))`) everywhere. The energy metric is more physically meaningful — it answers "what fraction of total demand energy is met?" rather than "what fraction of hours have some matching?"
**Risk**: None to PFS completeness. May change which mixes are classified as near-miss in Phase 1a, but the `batch_score_no_storage()` kernel in the post-process script shows how to do this efficiently.

#### 3. Wider Near-Miss Window (Coverage)
**Current**: Step 1 uses 15% near-miss window (`target - 0.15`).
**Improvement**: Expand to 25%. The post-process re-sweep with 25% window found 471K+ new solutions at CAISO 50% alone — mixes that scored 25-50% without storage but reached 50%+ with storage.
**Trade-off**: More near-miss mixes → more storage evaluations → longer runtime. With parallel kernels (improvement 1), the marginal cost is acceptable.

#### 4. Procurement Binary Search (2-5× faster per mix)
**Current**: Linear sweep of procurement levels — typically 30-50 evaluations per (mix, storage) combo with early stopping.
**Improvement**: Binary search for minimum feasible procurement: O(log₂ N) ≈ 5-6 evaluations instead of O(N) ≈ 30.
**Prerequisite**: Score is monotonically increasing with procurement (true by construction — more supply always helps).
**Integration**: Use batch evaluation at max procurement first to identify feasible mixes, then binary search per mix for the minimum.

#### 5. Full Storage Grid on Phase 2 Refinement (Coverage)
**Current**: Phase 2 (1% resolution refinement) only tries `[2, 5, 10]` for battery4 and battery8 levels.
**Improvement**: Use the full `[0, 2, 5, 8, 10, 15, 20]` grid, matching Phase 1b. Catches refinement mixes that need higher storage levels (15-20%) to become feasible.
**Cost**: Modest — refinement mixes are few (neighborhoods of Phase 1 archetypes), so the extra storage combos add seconds, not minutes.

#### 6. Two-Phase Adaptive Storage Sweep (Feb 21, 2026) — IMPLEMENTED

**Replaces** the previously rejected "adaptive storage tiers" approach. The two-phase approach does NOT skip storage levels — it sweeps ALL levels in both phases, differing only in granularity.

**Phase 1 — Coarse sweep (0.25% steps):** Identifies saturation range per ISO.
- bat4: `[0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5]`
- bat8: `[0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]`
- LDES: `[0, 0.5, 1.0, 1.5, 2.0, 2.5, 5, 8, 10]`
- After Phase 1, analyzes max used levels across all thresholds to find saturation point.

**Phase 2 — Fine sweep (0.05% steps):** Fills in granularity within the saturation range.
- bat4: `[0, 0.05, 0.10, ..., max_bat4 + 0.25%]` — e.g., ERCOT: 21 levels (0-1.00%)
- bat8: `[0, 0.05, 0.10, ..., max_bat8 + 0.25%]` — e.g., ERCOT: 36 levels (0-1.75%)
- LDES: Same coarse levels (less sensitive to granularity)
- Phase 1 coarse solutions merged with Phase 2 fine solutions (deduped by full key).

**Results per ISO:**
- ERCOT: 2,033,961 solutions (Phase 1: 226K coarse → Phase 2: 1.8M fine), 11 min
- CAISO: ~1.8M solutions (saturation: bat4=1.00%, bat8=2.00%), ~25 min
- Output: Per-ISO/threshold parquet files (`data/step1-pfs/{ISO}_t{XX}_raw_pfs.parquet`)

**Scientific rigor preserved:** Every level from 0 to saturation+margin is swept at 0.05% resolution. No levels are skipped or short-circuited. The coarse phase just identifies WHERE the fine sweep should focus.

#### 7. Cross-Threshold Solution Injection (Coverage)
**Current**: Step 1 uses "cross-threshold pollination" to track proven-feasible mixes and skip their storage sweep at higher thresholds. But doesn't inject these solutions — just avoids redundant work.
**Improvement**: Inject known-feasible (mix, storage) configs from lower thresholds as seeds for higher thresholds' procurement sweep. A mix feasible at 85% with battery=10 is likely feasible at 87.5% with battery=15 or higher procurement — the seed gives a starting point for the procurement search.

#### 8. Vectorized Phase 1a Procurement (Minor)
**Current**: Phase 1a loops over procurement levels sequentially, computing vectorized scores at each level.
**Improvement**: Batch multiple procurement levels into a single evaluation: `supply_batch[N, P, H] = supply_rows[:, None, :] * proc_array[None, :, None]`. Score all (mix × procurement) combinations in one vectorized operation.
**Caveat**: Memory-intensive — N_mixes × N_procurement × 8760 × 8 bytes. For 20K mixes × 50 proc levels = 70 GB. Only viable with chunking or for small grids.

#### Priority Ranking
| # | Improvement | Impact | Effort | Risk to PFS |
|---|---|---|---|---|
| 1 | Numba parallel storage | 4-8× speedup | Low (pattern exists) | None |
| 2 | Consistent scoring metric | Correctness | Low | None |
| 3 | Wider near-miss window | More solutions | Low | None |
| 4 | Binary search procurement | 2-5× per-mix speedup | Medium | None |
| 5 | Full refinement storage grid | More solutions | Low | None |
| 6 | Two-phase adaptive sweep | 0.05% granularity | Medium | None — **IMPLEMENTED** |
| 7 | Cross-threshold injection | More solutions | Medium | None |
| 8 | Vectorized Phase 1a | Minor speedup | Low | None (memory) |

**Implementation path**: Improvements 1-3 can be applied to `step1_pfs_generator.py` directly by copying kernels from `postprocess_storage_resweep.py`. Improvements 4-7 require refactoring the `optimize_threshold()` function. None require re-running from scratch — all are refinements to existing logic.

#### Implemented (Feb 21, 2026)

| # | Improvement | Status | Notes |
|---|---|---|---|
| 1 | Numba parallel storage | **Done** | `_batch_score_storage()` and `_batch_score_no_storage()` added with `prange`. JIT warmup includes batch kernels. |
| 2 | Consistent scoring metric | **Done** | Phase 1a and Phase 2 now use `sum(min(demand, supply))` (total matched energy), consistent with `_score_with_all_storage` base_matched. Previously used `sum(min(supply/demand, 1.0)) / H` (hourly average fraction) — different metric. |
| 5 | Full refinement storage grid | **Done** | Phase 2 now uses full `batt_levels × batt8_levels × ldes_levels` grid (matching Phase 1b) instead of limited `[2, 5, 10]` for battery4/battery8 only. LDES now modeled in Phase 2. |
| 3 | Wider near-miss window | **Done** | Phase 1a: 15% → 25%. Phase 2: 10% → 15%. More near-miss mixes enter storage testing → more feasible solutions found. |
| 4 | Binary search procurement | **Done** | All phases (1a, 1b, Phase 2) now use binary search O(log₂ N) instead of linear sweep O(N). Phase 1b also checks max procurement first and skips infeasible (mix, storage) combos entirely. |
| 8 | Vectorized Phase 1a procurement | **Done** | `batch_hourly_scores()` classifies all mixes at max procurement in one matrix multiply. Only feasible mixes enter per-mix binary search. Eliminates per-mix × per-proc scoring loop. |

**Code cleanup (Feb 21, 2026)**:
- **Removed 3 redundant scoring functions**: `_score_hourly`, `_score_with_battery`, `_score_with_both_storage` — all subsets of `_score_with_all_storage` (passing 0 for unused storage types skips those phases via capacity guards)
- **Removed redundant Phase 1b battery4-only loop**: Was a separate sweep before the full triple storage loop. Now the full triple loop covers all non-zero storage combos including battery4-only (no `b8p == 0 and lp == 0` skip)
- **Vectorized `_average_profiles()`**: Replaced nested Python loop (O(N×8760)) with `np.mean(np.array(profiles), axis=0)`
- **Vectorized `get_supply_profiles()` clean_firm**: Replaced hour-by-hour Python loop with `np.repeat(month_cfs, month_hours)`
- **Vectorized `get_supply_profiles()` solar DST correction**: Replaced 365×24 nested Python loop with numpy boolean mask across all 8760 hours
- **Vectorized `get_supply_profiles()` post-processing**: Replaced list comprehension `[max(0, v) for v in p]` with `np.maximum(arr, 0.0, out=arr)`
- **Vectorized `generate_4d_combos()`**: Replaced triple-nested Python loop with `np.meshgrid` + vectorized filter

| 7 | Cross-threshold solution injection | **Done** | Solutions from lower thresholds that score >= current threshold are injected directly into higher threshold candidate lists. Combined with existing `cross_skip` (Phase 1b skips re-testing known-feasible mixes), this avoids redundant computation while ensuring all qualifying solutions propagate upward. Done-threshold parquets load full solutions for seeding. |

**Rejected**:
- ~~Item 6: Adaptive storage tiers~~ — **Deleted from consideration**. Would skip higher-tier storage configs for mixes already feasible with simpler configs, undermining scientific rigor by missing storage diversity needed for Step 3 cost optimization.

### Clean Firm FOAK→NOAK Learning Curves in Step 3 Demand Growth (Mar 1, 2026)

**Decision**: Integrate Wright's Law FOAK→NOAK learning curves directly into Step 3 cost optimization (Phase 2 — demand growth sweep). Each threshold's SBTi target year determines the learning-curve-adjusted cost for new-build clean firm technologies. This replaces the static-cost model where all demand growth years used the same 2025 LCOE snapshot.

**Problem**: Step 3 currently uses identical LCOEs for all years (2025-2050). A buyer in 2030 faces FOAK pricing for nuclear/CCS/LDES, but the optimizer prices it at NOAK. This systematically underprices new-build clean firm in early years, making storage non-competitive at any threshold — inconsistent with real-world storage deployment.

**Scope**: Phase 2 (demand growth sweep) only. Phase 1 (base year 2025) remains at static L/M/H costs.

**Design choices**:
1. **Step 3 integration** (not post-hoc repricing) — learning curves must be inside the optimization to change which mixes are selected, not just reprice fixed mixes.
2. **Paired adoption speed + NOAK optimism** (3 combos, not 6) — each technology's L/M/H toggle controls both NOAK endpoint and adoption speed. Correlated in reality: fast deployment → more learning → lower NOAK. Avoids scenario count explosion.
3. **Technologies with learning curves**: Nuclear new-build, CCS-CCGT, LDES (100hr iron-air), Green H2, Geothermal (CAISO only), Battery 4hr, Battery 8hr (shallow curves, see Storage Cost Fix section below). Solar/wind already mature — static costs.
4. **Uprates unchanged** — $15/25/40 (L/M/H), no learning curve. Existing fleet, sunk cost.

**FOAK Cost Tables** (first-of-a-kind, pre-learning, single value per ISO — same for all L/M/H):

Nuclear new-build FOAK ($/MWh) — ~1.25× High (Vogtle-era pricing):
| ISO | FOAK |
|-----|------|
| CAISO | 175 |
| ERCOT | 169 |
| PJM | 200 |
| NYISO | 212 |
| NEISO | 206 |
| MISO | 194 |
| SPP | 175 |

CCS-CCGT FOAK 45Q ON ($/MWh) — ~1.20× High:
| ISO | FOAK |
|-----|------|
| CAISO | 138 |
| ERCOT | 110 |
| PJM | 122 |
| NYISO | 154 |
| NEISO | 146 |
| MISO | 115 |
| SPP | 106 |

CCS-CCGT FOAK 45Q OFF ($/MWh) — ~1.20× High:
| ISO | FOAK |
|-----|------|
| CAISO | 173 |
| ERCOT | 145 |
| PJM | 157 |
| NYISO | 188 |
| NEISO | 181 |
| MISO | 150 |
| SPP | 140 |

Geothermal FOAK (CAISO only): $150/MWh (~1.35× High)

LDES FOAK ($/MWh-cap, annualized capacity cost) — ~1.40× High:
| ISO | FOAK |
|-----|------|
| CAISO | 1.40 |
| ERCOT | 1.20 |
| PJM | 1.32 |
| NYISO | 1.55 |
| NEISO | 1.48 |
| MISO | 1.26 |
| SPP | 1.23 |

Green H2 FOAK ($/MWh-cap, annualized capacity cost) — ~1.30× High:
| ISO | FOAK |
|-----|------|
| CAISO | 5.32 |
| ERCOT | 4.69 |
| PJM | 5.04 |
| NYISO | 5.85 |
| NEISO | 5.58 |
| MISO | 4.82 |
| SPP | 4.60 |

**Learning Curve Parameters** (per toggle level):

| Level | Adoption | FOAK Start | NOAK Year | Duration | NOAK Endpoint | Wright's Law Exponent |
|-------|----------|------------|-----------|----------|---------------|----------------------|
| L (Optimistic/Fast) | Fast | 2028 | 2036 | 8 years | Low cost table | 0.6 |
| M (Central) | Central | 2030 | 2040 | 10 years | Medium cost table | 0.6 |
| H (Pessimistic/Slow) | Slow | 2036 | 2048 | 12 years | High cost table | 0.6 |

**Unified timelines for all clean firm technologies** (simplified from prior per-technology overrides — now all clean firm techs share the same L/M/H learning schedule):

| Technology | Toggle | L FOAK→NOAK | M FOAK→NOAK | H FOAK→NOAK |
|------------|--------|-------------|-------------|-------------|
| Nuclear new-build | Firm | 2028→2036 | 2030→2040 | 2036→2048 |
| CCS-CCGT | CCS | 2028→2036 | 2030→2040 | 2036→2048 |
| Geothermal | Geo | 2028→2036 | 2030→2040 | 2036→2048 |
| LDES | LDES | 2028→2036 | 2030→2040 | 2036→2048 |
| Green H2 | LDES | 2028→2036 | 2030→2040 | 2036→2048 |
| Battery 4hr | Batt | 2025→2030 | 2026→2032 | 2027→2035 |
| Battery 8hr | Batt | 2025→2030 | 2026→2032 | 2027→2035 |

**Year-adjusted cost formula**: `cost(year) = FOAK × (1 - frac) + NOAK × frac` where `frac = learning_fraction(year, foak_start, noak_year)`.

**`learning_fraction(year, foak_start, noak_year)`**:
- Before `foak_start`: 0.0 (pure FOAK)
- After `noak_year`: 1.0 (full NOAK)
- During learning: `((year - foak_start) / (noak_year - foak_start)) ** 0.6`
- Exponent 0.6 produces Wright's Law concave ramp: steep initial drop (first 40% of cost reduction in first 30% of timeline), then asymptotic approach to NOAK.

**Example impact (PJM, 70% threshold = year 2035, Firm=M, CCS=M)**:
- Nuclear FOAK=$200, NOAK_M=$105, frac=`((2035-2030)/(2040-2030))^0.6 = 0.50^0.6 ≈ 0.66` → year cost = $200×0.34 + $105×0.66 = **$137/MWh**
- CCS FOAK 45Q ON=$122, NOAK_M=$79, frac=`((2035-2030)/(2040-2030))^0.6 = 0.50^0.6 ≈ 0.66` → year cost = $122×0.34 + $79×0.66 = **$93/MWh**
- Battery 4hr (static): **$98/MWh**
- **Battery now competitive with nuclear new-build at 70% threshold**

**Implementation**: Modify `precompute_all_prices()` in `step3_cost_optimization.py` to accept optional `target_year` parameter. In Phase 2, compute a year-specific price matrix for each unique DG year (14 matrices × ~0.01s each = negligible overhead). No new scenario dimensions — learning curves are embedded in existing L/M/H toggles.

**Supersedes**: Line 1483 of this file ("Scope: PP3 scenario comparison only. Step 3 cost optimization is NOT modified"). Step 3 DG sweep now uses learning curves. Step 6 scenario comparison curves remain unchanged (their own timeline parameters are for the A/B strategy comparison, not the core optimization).

**No compute cost increase**: Same 5,832/17,496 sensitivity combos per threshold. Only the price lookup changes for DG years. Phase 1 (base year) unchanged. Step 1/2 not affected.

### Storage Cost Fix: LCOS → Annualized Capacity Cost (Mar 1, 2026)

**Bug**: Step 3 priced storage as `bat_pct / 100.0 × LCOS`, where `bat_pct/100` is a normalized energy capacity parameter (energy capacity as fraction of avg hourly demand MWh). LCOS is $/MWh of *discharged* energy. The product gives the wrong units — it treats the capacity sizing parameter as an annual dispatch fraction, overpricing storage by 10-50× depending on utilization assumptions. Example: bat_pct=3, LCOS=$102 → $3.06/MWh, but actual annual cost of a 900 MW/3.6 GWh battery for CAISO is ~$0.28/MWh of demand.

**Fix**: Replace LCOS tables with annualized capacity cost per % of annual demand, matching the coefficient model:
```
price = CAPEX_kWh × (CRF + FOM_rate) × 1000 × regional_mult
```
Now `bat_pct / 100.0 × price` directly gives the annual fixed cost of that storage capacity as a fraction of total demand cost ($/MWh of demand). Storage uses the same unit (% of annual demand) as all other resources. Eliminates cycling assumptions — prices pure capacity, not utilization.

**Financial parameters**:
- WACC: 8%
- Battery lifetime: 20 years → CRF = 0.10185
- LDES lifetime: 25 years → CRF = 0.09368
- H2 lifetime: 20 years → CRF = 0.10185

**CAPEX per kWh** (NREL ATB 2024):
| Technology | Low | Medium | High | Duration | FOM ($/kW-yr) |
|-----------|-----|--------|------|----------|---------------|
| Battery 4hr | $115 | $140 | $165 | 4 hr | $25 |
| Battery 8hr | $95 | $120 | $145 | 8 hr | $25 |
| LDES (iron-air) | $30 | $50 | $80 | 100 hr | $5 |
| Green H2 | $150 | $220 | $310 | 168 hr | $8 |

**Regional multipliers**: Derived from existing LCOS ratio (normalize to ERCOT=1.0), baked into capacity prices. TX adders set to $0 for all storage types.

**LCOS cross-check** (validates capacity prices against known LCOS benchmarks):
- Battery 4hr Med @ 365 cycles/yr = $121/MWh (Lazard 2024 range: $115-220)
- Battery 8hr Med @ 300 cycles/yr = $107/MWh
- LDES Med @ 50 cycles/yr = $95/MWh

**Battery learning curves** (updated March 2026 — Wright's Law with NREL-calibrated trajectories):
- Direction: LCOE_TABLES (2025 starting) → NOAK_BATTERY (terminal floor). Reverse of other techs.
- NOAK fractions: Low=50%, Med=56%, High=80% of starting cost. Calibrated to NREL 2050 projections.
- Timelines: bat4 L=(2025,2042), M=(2025,2048), H=(2025,2050). bat8 2yr faster.
- Exponent: 0.6 (concave ramp). Net effect: meaningful decline over 20+ year horizon, not the old shallow 5yr curve.
- See §5.5 for full trajectory table.

**Storage FOAK tables** ($/MWh-cap):
- Battery 4hr/8hr: FOAK = High (no premium — batteries at scale). Not used by learning curves (batteries use LCOE→NOAK direction).
- LDES: 1.40× High capacity cost per ISO
- H2: 1.30× High capacity cost per ISO

**Propagated to**: `step3_cost_optimization.py`, `scenario_common.py` (has independent LCOE_TABLES copy — updated Mar 4, 2026). All storage LCOE/FOAK/NOAK tables now use % of annual demand unit across all files.

**Supersedes**: Prior LCOS values in LCOE_TABLES for battery, battery8, ldes, h2. Line 3764 of this file updated — batteries now have learning curves too (previously listed as "already mature — static costs").

---

## §16: Step 1D.2 Enhanced Storage Model + Full Economic Assessment

### §16.1 Overview

The current pipeline has two gaps:
1. **Storage caps are conservative**: Battery 4hr=0.5%, Battery 8hr=1.0%, LDES=5.0% of demand — reflecting near-term (2025-2030) deployment. For a model analyzing paths to 2050, these caps artificially constrain storage-dominant solutions.
2. **No revenue offsets**: Tracks 1-3 price storage as pure annualized CAPEX. Real-world storage economics include capacity market payments, energy arbitrage revenue, and ancillary services — which can offset 30-60% of gross cost.

Step 1D.2 addresses gap (1) with research-informed 2050 capacity caps. The Economic Assessment layer addresses gap (2) by adding revenue streams to the cost evaluation of Tracks 1-3.

**Key framing**: This is NOT a separate Track 4. It's a full economic & revenue source assessment applied to the same Tracks 1-3. The output is "here's what Track 1/2/3 looks like under pure LCOE vs. full lifecycle economics."

### §16.2 Storage Capacity Caps (Research-Informed, 2050 Horizon)

**NOTE**: All storage values are in **% of annual demand** (energy capacity as fraction of annual demand). Same unit as all other resources.

**Current (Step 1D):**
| Storage Type | Max (% of annual demand) | CAISO equivalent | Rationale |
|---|---|---|---|
| Battery 4hr | 0.06% | 134 GWh / 33.6 GW | Near-term deployment |
| Battery 8hr | 0.08% | 179 GWh / 22.4 GW | Near-term deployment |
| LDES | 0.5% | 1,120 GWh / 11.2 GW | Early iron-air |
| H2 | 2.0% | 4,480 GWh / 4.5 GW | Seasonal storage |

**Step 1D.2 (2050-oriented):**
| Storage Type | Max (% of annual demand) | CAISO equivalent | Rationale |
|---|---|---|---|
| Battery 4hr | 0.10% | 224 GWh / 56 GW | NREL: 200 GW reference case |
| Battery 8hr | 0.15% | 336 GWh / 42 GW | DOE: 8hr key for deep decarb overnight coverage |
| LDES | 1.0% | 2,240 GWh / 22.4 GW | DOE: 225-460 GW LDES by 2050 |
| H2 | 3.0% | 6,720 GWh / 6.7 GW | Seasonal storage |

**Research basis:**
- NREL Storage Futures Study: 125-680 GW total storage by 2050, 200 GW / 1,200 GWh reference case
- DOE LDES Liftoff: 225-460 GW LDES needed for net-zero; $10-20B annual savings vs. gas capacity
- Princeton Net Zero America: 1,300 GWh grid storage by 2050
- NREL ATB 2024: Battery costs decline 47-68% by 2050 (mid/low scenarios)

### §16.3 Step 1D.2 Script Architecture

**Script**: `scripts/step1d2_enhanced_storage.py`
- Fork of `step1d_fine_storage.py` with higher caps from `STORAGE_MAX_V2`
- Same three-pass architecture (Pass 0 ceiling screen, Pass 1 adaptive coarse, Pass 2 fine refinement)
- Same Numba dispatch kernels from `dispatch_utils.py`
- Uses existing near-miss cache (already 100k mixes/ISO from Step 1C)
- Output: `data/step1d2-storage-parquets/{ISO}_t{THRESHOLD}_storage.parquet`

**Key differences from Step 1D:**
1. Reads `STORAGE_MAX_V2` instead of `STORAGE_MAX`
2. Coarser initial grid (wider range → need 0.5% coarse step instead of 0.25%)
3. Same fine resolution (0.05%) on frontier boundary mixes
4. Output directory: `data/step1-pfs/` (consolidated storage output)

### §16.4 Pipeline Integration

**Step 2 (Efficient Frontier)**:
- Add `--storage-source` flag: `step1d` (default) or `step1d2`
- When `--storage-source step1d2`: reads from `data/step1d2-storage-parquets/`, outputs to `data/step2-ef-v2-parquets/`
- Both EF sets maintained in parallel — downstream scripts choose which to consume

**Step 3 (Cost Optimization)**:
- Add `--ef-source` flag: `step2` (default) or `step2-v2`
- Economic assessment applied as post-processing on Step 3 output (see §16.5)

**Dispatch Cache**:
- Separate cache for 1D.2 mixes (higher storage penetration = different dispatch profiles)

### §16.5 Storage Revenue Credit (Implemented)

**Approach**: Simple revenue credit subtracted from gross storage LCOE in all cost evaluation paths. Not a separate script — directly embedded in `step3_cost_optimization.py` and `scenario_common.py` cost functions.

**Decision**: Chose Option 1 (simple credit) over Option 2 (LMP feedback loop) for directional correctness with minimal complexity. Revenue credit captures the key economic reality that storage earns capacity, arbitrage, and ancillary revenue that offsets gross LCOE.

**Formula**:
```
net_storage_cost = max(0, gross_LCOE - revenue_credit)

revenue_credit = capacity_payment + (arbitrage + ancillary) × stacking_factor
credit_in_lcoe_units = 1000 × revenue_$/kW-yr ÷ duration_hr
```

**Parameters**:
- **Revenue stacking factor**: 0.70 — battery can't simultaneously do arbitrage and ancillary in the same hour; capacity is always earned (availability-based)
- **Net LCOE floor**: `max(0, ...)` prevents negative storage costs
- **Ancillary product eligibility**: Battery → regulation (fast response); LDES/H2 → spinning reserve
- **Ancillary hours**: Regulation 2,000 hrs/yr (~23%); Spinning 4,000 hrs/yr (~46%)

**Revenue sources ($/kW-yr) by ISO**:

| Source | CAISO | ERCOT | PJM | NYISO | NEISO | MISO | SPP |
|--------|-------|-------|-----|-------|-------|------|-----|
| Capacity market | 75 | 0 | 120 | 85 | 55 | 25 | 0 |
| Battery 4hr arbitrage | 50 | 50 | 45 | 55 | 35 | 30 | 25 |
| Battery 8hr arbitrage | 43 | 43 | 38 | 47 | 30 | 26 | 21 |
| LDES arbitrage | 15 | 20 | 12 | 14 | 10 | 8 | 7 |
| H2 arbitrage | 5 | 6 | 4 | 5 | 3 | 3 | 2 |
| Regulation rate ($/MW-hr) | 12 | 15 | 18 | 14 | 10 | 8 | 6 |
| Spinning rate ($/MW-hr) | 5 | 8 | 6 | 5 | 4 | 3 | 3 |

**Sources**: CAISO/ERCOT arbitrage compressed vs. 2022-23 peaks — 2024-25 data shows $30-55/MWh avg spread (CAISO duck curve moderated by 10GW+ installed storage). PJM capacity from RPM 2025-2028 BRA clearing ($98-269/MW-day). Ancillary rates from ISO tariff schedules.

**Pre-computed credits** (LCOE-equivalent units, same as LCOE_TABLES storage):
- `STORAGE_REVENUE_CREDITS` dict in `pipeline_config.py`, computed by `compute_storage_revenue_credit()`
- Pre-computed at import time for all 4 storage types × 7 ISOs

**Impact on cost evaluation** (example — Medium battery, CAISO):
- Gross LCOE: 41,610 → Net LCOE: ~9,910 (76% reduction)
- PJM net battery cost goes to $0 (high capacity payments exceed gross LCOE at Low costs)
- LDES: 8-18% reduction. H2: <1% reduction (seasonal cycling, minimal revenue streams)

**Files modified**:
- `pipeline_config.py`: Added `STORAGE_ARBITRAGE_REVENUE`, `REVENUE_STACKING_FACTOR`, `STORAGE_ANCILLARY_PRODUCT`, `compute_storage_revenue_credit()`, `STORAGE_REVENUE_CREDITS`
- `step3_cost_optimization.py`: Revenue credit subtracted in 3 code paths — `compute_costs_vectorized`, `get_scenario_prices`, and demand growth sweep price matrix
- `scenario_common.py`: Revenue credit subtracted in 4 code paths — helper function, `compute_mix_cost`, `_get_excess_lcoe`, and `_precompute_cost_params`

**Known limitations (V1)**:
- Revenue streams are static per ISO (no spread compression with storage deployment)
- No hour-by-hour co-optimization of arbitrage vs. ancillary
- Capacity market prices don't decline with ELCC saturation at high penetration
- Stacking factor is a flat 70% (real co-optimization efficiency varies by ISO and penetration)

### §16.6 New Constants (pipeline_config.py)

```python
# Step 1D.2 storage caps (2050-oriented)
STORAGE_MAX_V2 = {
    'battery': 5.0,     # 5.0% of demand (10× current)
    'battery8': 10.0,   # 10.0% of demand (10× current)
    'ldes': 25.0,       # 25.0% of demand (5× current)
    'h2': 25.0,         # 25.0% of demand (unchanged)
}

# Capacity market prices ($/kW-yr) — from 2024-2025 auction results
CAPACITY_MARKET_PRICES = {
    'CAISO': 75,    # RA program, system-wide avg
    'ERCOT': 0,     # No capacity market (energy-only)
    'PJM': 120,     # RPM 2025/2026-2027/2028 BRA clearing ($98-269/MW-day)
    'NYISO': 85,    # ICAP monthly spot, annualized
    'NEISO': 55,    # FCM FCA-19 clearing price
    'MISO': 25,     # PRA Zone 1-7 average
    'SPP': 0,       # No capacity market
}

# Ancillary service rates ($/MW-hr)
ANCILLARY_SERVICE_RATES = {
    'regulation': {  # Frequency regulation (battery only)
        'CAISO': 12, 'ERCOT': 15, 'PJM': 18, 'NYISO': 14,
        'NEISO': 10, 'MISO': 8, 'SPP': 6,
    },
    'spinning': {  # Spinning reserve (battery + LDES)
        'CAISO': 5, 'ERCOT': 8, 'PJM': 6, 'NYISO': 5,
        'NEISO': 4, 'MISO': 3, 'SPP': 3,
    },
}

# Ancillary service eligibility (hours/year)
ANCILLARY_HOURS = {
    'regulation': 2000,  # Battery available ~23% of year for reg
    'spinning': 4000,    # Battery/LDES available ~46% of year for spin
}

# Battery degradation parameters
BATTERY_DEGRADATION = {
    'battery': {
        'cycles_per_year': 365,     # Daily cycling
        'cycle_life_80pct': 5000,   # Cycles to 80% capacity
        'replacement_fraction': 0.4, # Cost of augmentation/replacement
    },
    'battery8': {
        'cycles_per_year': 365,
        'cycle_life_80pct': 4000,   # Deeper discharge = faster degradation
        'replacement_fraction': 0.45,
    },
    'ldes': {
        'cycles_per_year': 52,      # Weekly cycling
        'cycle_life_80pct': 20000,  # Iron-air minimal degradation
        'replacement_fraction': 0.15,
    },
    'h2': {
        'cycles_per_year': 12,      # Monthly cycling
        'cycle_life_80pct': 50000,  # Electrolysis replacement is the main cost
        'replacement_fraction': 0.25,
    },
}
```

### §16.7 Phased Implementation

**Phase 1 (Current)**: Step 1D.2 V1 — higher caps, same dispatch
- `step1d2_enhanced_storage.py` — fork with `STORAGE_MAX_V2`
- `pipeline_config.py` — add `STORAGE_MAX_V2`, economic constants
- GitHub Actions workflow: `step1d2-enhanced-storage.yml`

**Phase 2 (Current)**: Economic assessment
- `step3_economic_assessment.py` — revenue stacking post-processor
- Step 2 integration (`--storage-source step1d2`)

**Phase 3 (Later)**: Enhanced multi-service dispatch
- Price-responsive charge/discharge (LMP-aware scheduling)
- Co-optimized battery + LDES dispatch
- New Numba kernels
- Replaces V1 dispatch in 1D.2

**Phase 4 (Later)**: Dashboard integration
- Revenue waterfall charts on storage_analysis.html
- Toggle: LCOE vs Full Economic on dashboard.html
- Enhanced storage section with economic metrics

### §16.8 GitHub Actions

**New workflow**: `.github/workflows/step1d2-enhanced-storage.yml`
- Same structure as `step1d-fine-storage-v2.yml`
- Uses `STORAGE_MAX_V2` caps
- Per-threshold execution with auto-commits
- Output: `data/step1d2-storage-parquets/`

**New workflow**: `.github/workflows/step3-economic-assessment.yml`
- Depends on: Step 3 cost output + dispatch cache + LMP data
- Runs economic assessment post-processing
- Output: `data/step3-economic-parquets/`

### §16.9 Level 2: Endogenous Price Feedback (Future)

**Goal**: Close the feedback loop between storage deployment and wholesale prices.
Currently (V1), each mix sees the same base-case LMPs regardless of how much
storage/clean energy it deploys. In reality, high clean energy penetration
suppresses prices and high storage deployment compresses peak-offpeak spreads.

**Architecture**: `step3b_endogenous_pricing.py` — single forward-pass post-processor.

**Pipeline position**: Runs after Step 4 dispatch cache + Step 5 LMP model.
```
Step 1D.2 (physics) → Step 3 (cost) → Step 4 (dispatch cache)
                                         ↓
Step 5 LMP (base case) → Step 3b (endogenous pricing)
```

**Algorithm per mix**:
1. Load 8,760-hour dispatch profile from Step 4 cache (matched, surplus, gap per resource)
2. Compute residual fossil demand = `total_demand − clean_supply − storage_discharge`, hourly
3. Run residual through merit-order fossil stack to get **mix-specific hourly LMPs**
   - Same stack as `step5_compute_lmp_prices.py` but with this mix's residual, not base-case
   - High clean energy → less fossil on the margin → lower LMPs in those hours
   - High storage → peaks shaved → less scarcity pricing → compressed spreads
4. Compute arbitrage revenue on **endogenous prices** (not base-case)
   - Specifically: `profit = Σ_h(discharge_h × LMP_h) − Σ_h(charge_h × LMP_h)`
   - This is true co-optimized hourly arbitrage value
5. Adjust capacity value: if residual peak demand is significantly reduced,
   haircut capacity price proportionally (ELCC saturation proxy)

**Feedback loops captured**:
- **Spread compression**: More storage → flatter price profile → less arbitrage
- **Renewable price suppression**: More solar/wind → more zero-marginal-cost hours
- **Fossil retirement pricing**: As clean energy displaces fossil, marginal generator changes
- **Duck curve dynamics**: Solar midday trough deepens/fills based on mix composition

**Feedback NOT captured** (would require Level 3 equilibrium model):
- Capacity market clearing price response to aggregate storage deployment
- Cross-regional price effects
- Investor response dynamics (build/no-build decisions over time)
- Iterative equilibrium (storage changes prices which changes optimal storage)

**Compute**: Reads from existing caches. No Step 1 rerun. Per-mix merit-order dispatch
is O(8760) per mix — fast enough for all 7 ISOs × all thresholds as post-processing.

**Key design principle**: One forward pass, not iterative. The endogenous prices
reflect "what would prices be IF this mix were deployed" — not "what is the
equilibrium deployment." This is the sweet spot between static overlay (V1) and
full equilibrium (Level 3) in terms of rigor vs. complexity.

### §16.10 Coarse Grid Density (Step 1D.2)

The V2 expanded storage range requires denser intermediate grid points at the
high end to avoid coarse-sweep gaps that Pass 2 fine refinement (±0.5pp) cannot
bridge.

**V2 coarse grids (11 × 11 × 12 = 1,452 combos, excluding H2)**:
- bat4:  `[0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]` — 11 levels
- bat8:  `[0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]` — 11 levels
- LDES:  `[0, 0.25, 0.5, 1.0, 2.0, 5.0, 8.0, 12.0, 15.0, 17.0, 21.0, 25.0]` — 12 levels
- H2:    `[0, 5, 10, 15, 20, 25]` — 6 levels (unchanged)

Maximum coarse gap: bat4=1.0pp, bat8=2.0pp, LDES=4.0pp. Fine sweep at ±0.5pp
ensures all gaps are bridgeable. 1.6× the original 900 combos — minimal
compute increase for significantly better coverage at the high end.

**Empirical validation (March 2026)**: Grid sensitivity tests across CAISO (solar-heavy,
6D), NEISO (winter-constrained, 5D), and ERCOT (wind-heavy, 4D) confirm:
- **Storage grid precision does NOT change which resource mix wins** — at every threshold
  (95%, 97.5%, 99%, 99.9%), the same mix is cost-optimal regardless of LDES grid density.
- **Coarse grid only overshoots LDES level** by ~2-2.5pp at 99.9% threshold — same mix
  wins but at a higher (costlier) LDES than necessary. Pass 2 fine sweep corrects this.
- **Score response to LDES is smooth and monotonic** — no step-function behavior, no
  discontinuities. Each 0.25pp LDES adds a consistent ~0.02pp score (92% base) to
  ~0.002pp (99% base).
- **ERCOT plateaus at 1% LDES** (wind diversity fills gaps), **CAISO gains steadily to 25%**
  (solar intermittency needs long-duration shifting).
- **Winner flips**: 0/51 at thresholds ≤99%, 2-3/51 at 99.9% — all same-mix, different LDES.
- **Conclusion**: V2 grid density is adequate for storage-vs-storage comparison.

**Cost tradeoff: storage substitution vs resource overbuild (March 2026)**:
The critical question is not whether storage grid precision flips one storage mix vs
another — it's whether storage-augmented mixes outcompete pure resource overbuild.

CAISO empirical test at 99.9% threshold (Medium LCOE, Medium TX):
- **Baseline winner** (no storage search): 106% CF / 7% solar / 13% wind — 148% total
  overbuild at $75.57/MWh. Already scores 99.9995% from sheer overbuild.
- **With LDES substitution**: Scaling down to 88% of this mix + 20.5% LDES achieves
  99.9% at $65.89/MWh — a **$9.69/MWh (13%) savings**. LDES at ~$0.63/%-demand is
  far cheaper than the marginal new resources it replaces ($63-98/MWh).
- **Floor at ~86% scale**: Below this, 25% LDES cannot compensate — the resource mix
  must provide sufficient base hourly diversity. Storage shifts surplus to gaps but
  cannot create energy.
- **Near-miss cache limitation**: Step 1C's 100k near-miss mixes were optimized WITHOUT
  storage. They over-represent fat overbuild mixes and under-represent lean+storage
  combos. Step 1D.2 specifically searches the storage dimension to find these cheaper
  alternatives.

---

## 23. Hybrid Co-Located Resources (Mar 2026)

### 23.1 Overview

Four co-located hybrid resource types extend the optimizer's resource set. Each pairs a generation technology with on-site battery storage behind a single point of interconnection (POI). The battery charges exclusively from co-located generation (no grid charging), qualifying for ITC benefits under IRA rules.

| Type | Generation | Battery | Primary Value |
|------|-----------|---------|---------------|
| `solar_batt4` | Solar | 4hr Li-ion | Core clipping recovery — captures DC overgeneration |
| `solar_batt8` | Solar | 8hr Li-ion | Extended clipping + deeper temporal shift |
| `wind_batt4` | Wind | 4hr Li-ion | Short shifting, dual-cycle, narrow arbitrage blocks |
| `wind_batt8` | Wind | 8hr Li-ion | Deep overnight-to-peak shifting |

**Offshore wind hybrids skipped** — co-located storage doesn't reshape the offshore gen profile; it's a transmission/grid-level problem better handled by standalone grid storage.

**Hybrids are additive, not substitutional** — a mix can include standalone solar AND solar_batt4 simultaneously. They're different assets with different cost and output profiles. No forced substitution constraint.

### 23.2 Physical Models

#### 23.2.1 Solar Hybrids — DC:AC Clipping Model

Solar panels (DC) are oversized relative to grid interconnection (AC). During peak sun, excess generation ("clipped" energy) charges the co-located battery. Discharge occurs during net-peak hours.

**DC:AC Ratios (validated via `scripts/validate_dcac_ratios.py`):**

| Type | CAISO | All Other ISOs | Rationale |
|------|-------|----------------|-----------|
| `solar_batt4` | 1.35 | 1.50 | Current industry practice; 4hr captures all clipping |
| `solar_batt8` | 1.70 | 2.00 | Aggressive overbuild so 8hr differentiates from 4hr |

At solar_batt4 ratios, the 4hr battery already captures all clipped energy — making 8hr redundant (identical profiles). Higher ratios create enough clipping to fill the 8hr battery: +434 to +666 more active hours, flatter output shapes.

**Dispatch model:**
```
solar_dc = dc_capacity × solar_profile[h]
ac_cap = dc_capacity / dc_ac_ratio
clipped = max(0, solar_dc - ac_cap)
charge = min(clipped, battery_headroom) × √RTE
to_grid = min(solar_dc, ac_cap)
# Discharge during top-N net-peak hours/day (N = battery duration)
if is_net_peak[h] and soc > 0:
    discharge = min(power_rating, soc)
    to_grid += discharge × √RTE
    to_grid = min(to_grid, ac_cap)  # POI cap
```

#### 23.2.2 Wind Hybrids — Temporal Shifting Model

No DC:AC overbuild — wind turbines are AC machines with no clipping dynamic. Battery charges from off-peak wind surplus and discharges during net-peak hours.

**Battery:Wind MW Ratio:** 25–40% of wind nameplate (ISO-specific, derived from wind profile analysis). Sized to POI ceiling: `battery_MW = POI_limit − avg_wind_during_peak_hours`.

**Dispatch model:**
```
wind_gen = capacity × wind_profile[h]
if is_off_peak[h] and wind_gen > demand_share and soc < max_soc:
    to_grid = demand_share
    charge = min(wind_gen - demand_share, battery_headroom) × √RTE
else:
    to_grid = wind_gen
# Discharge during top-N net-peak hours/day
if is_net_peak[h] and soc > 0:
    discharge = min(power_rating, soc)
    to_grid += discharge × √RTE
    to_grid = min(to_grid, interconnect_cap)  # POI cap
```

#### 23.2.3 Common Parameters

| Parameter | 4hr variants | 8hr variants |
|-----------|-------------|-------------|
| Duration | 4 hours | 8 hours |
| Round-trip efficiency | 85% | 85% |
| Grid charging | None | None |
| Discharge trigger | Top 4 net-peak hours/day | Top 8 net-peak hours/day |

**Net peak** = hours where (demand − total renewable generation) is highest each day.

**Profile integration**: Pre-computed 8760 hybrid profiles per ISO, normalized to sum ~1.0. The optimizer sees hybrid output as a single resource — dispatch is pre-resolved into the 8760 shape (not decomposed into solar+battery at runtime).

### 23.3 Cost Model — Component-Additive LCOE

**Formula:**
```
hybrid_LCOE = gen_LCOE + storage_LCOE_duration_weighted − ITC_benefit + TX_adder_AC
```

Where:
- `gen_LCOE` = parent generation LCOE (solar or wind L/M/H from existing toggle)
- `storage_LCOE_duration_weighted` = battery LCOE for the appropriate duration (4hr or 8hr) scaled by battery:gen capacity ratio
- `ITC_benefit` = 30% ITC applied to total hybrid project cost (both solar and wind hybrids qualify under IRA §48)
- `TX_adder_AC` = single transmission adder sized to AC interconnection rating (not DC nameplate)

**Worked Example — solar_batt4, PJM, Medium costs:**
```
Solar LCOE (Med):        $31.80/MWh
Battery 4hr LCOE (Med):  $11.20/MWh (duration-weighted at 1.0 battery:POI ratio)
ITC 30% benefit:         −$12.90/MWh (30% × combined project cost)
TX adder (Med, AC-rated): $5.50/MWh (single adder, not doubled)
─────────────────────────
Hybrid LCOE:             $35.60/MWh
Standalone equivalent:   $31.80 + $11.20 + $5.50 + $5.50 = $54.00/MWh
Savings:                 $18.40/MWh (34% cheaper — shared interconnection + ITC)
```

**Worked Example — wind_batt8, ERCOT, Medium costs:**
```
Wind LCOE (Med):         $24.50/MWh
Battery 8hr LCOE (Med):  $16.80/MWh (duration-weighted at 0.35 battery:wind ratio)
ITC 30% benefit:         −$12.39/MWh
TX adder (Med, AC-rated): $4.20/MWh
─────────────────────────
Hybrid LCOE:             $33.11/MWh
Standalone equivalent:   $24.50 + $16.80 + $4.20 + $4.20 = $49.70/MWh
Savings:                 $16.59/MWh (33% cheaper)
```

#### 23.3.1 ITC Treatment

- **30% ITC for both solar and wind hybrids** under IRA §48/§48E
- Solar+storage: Well-established; battery must charge ≥80% from co-located solar (our model is 100% — no grid charging)
- Wind+storage: Qualifies under IRA expansion of energy storage ITC to all qualified clean energy facilities
- ITC applied to combined project capex (generation + storage), reducing effective LCOE
- No separate hybrid FOAK/NOAK learning curves — component learning curves (solar/wind + battery) apply independently, weighted by cost share

#### 23.3.2 AC-Rating-Adjusted Transmission

Hybrid projects pay **one** transmission adder sized to the AC interconnection rating:
- `solar_batt`: TX sized to `DC_nameplate / DC:AC_ratio` (the AC rating is smaller than DC nameplate)
- `wind_batt`: TX sized to wind nameplate (already AC-rated)
- Standalone equivalent would pay two adders (one per resource) — hybrid saves 1× TX adder per project

### 23.4 Toggle Pairing — No New Toggles

Hybrids inherit existing toggle sensitivities:
- **Generation cost**: Follows parent resource toggle (Renewable Gen L/M/H for solar hybrids, same for wind)
- **Storage cost**: Follows Storage toggle (L/M/H)
- **Transmission**: Follows Transmission toggle (None/L/M/H)
- **ITC**: Always 30% (not toggled independently)

No new dashboard controls needed. The 5,832 existing scenario combinations (17,496 for CAISO) automatically cover hybrid cost variation through the existing toggle pairs.

### 23.5 Family Caps

Resource caps constrain the grid search to prevent combinatorial explosion:

| Cap | Formula | Purpose |
|-----|---------|---------|
| `SOLAR_FAMILY_CAP` | `solar + solar_batt4 + solar_batt8 ≤ cap` | Prevents unrealistic total solar deployment |
| `WIND_FAMILY_CAP` | `wind + wind_batt4 + wind_batt8 ≤ cap` | Prevents unrealistic total wind deployment |
| `HYBRID_MAX_PER_TYPE` | Per-type cap (e.g., solar_batt4 ≤ 115%) | Individual hybrid resource ceiling |

Per-ISO caps from empirical analysis + buffer:

| ISO | Sol | SB4 | SB8 | Wind | WB4 | WB8 | Total |
|-----|-----|-----|-----|------|-----|-----|-------|
| CAISO | 95 | 115 | 115 | 145 | 165 | 165 | 225 |
| ERCOT | 85 | 105 | 105 | 205 | 225 | 225 | 240 |
| PJM | 85 | 105 | 105 | 160 | 180 | 180 | 225 |
| NYISO | 95 | 115 | 115 | 90 | 110 | 110 | 185 |
| NEISO | 95 | 115 | 115 | 95 | 115 | 115 | 220 |
| MISO | 60 | 80 | 80 | 215 | 235 | 235 | 255 |
| SPP | 60 | 80 | 80 | 195 | 215 | 215 | 225 |

Hybrid caps = parent resource max + 30pp buffer (no prior data, need exploration room). After first hybrid-inclusive run, re-extract with `extract_empirical_caps.py` and tighten.

### 23.6 Pipeline Integration Status

| Step | Status | Notes |
|------|--------|-------|
| **Step 0** | No change | Existing EIA profiles used; hybrid profiles derived in Step 1 |
| **Step 1.1** | Modified | 8–10D grid search (4D base + 4 hybrid dims; 5D CAISO + 4 hybrid). Memory-safe chunked generation. |
| **Step 1.1b** | Auto-adapts | Reads columns from parquet schema — no changes needed |
| **Step 1.2–1.5** | Modified | Zone/floor/fine/storage search extended to hybrid dimensions |
| **Step 2.1** | Modified | Efficient frontier extraction includes hybrid resource columns |
| **Step 2.2** | Modified | Component-additive LCOE with ITC and AC-adjusted TX in cost function |
| **Step 3A** | Modified | Dispatch cache includes hybrid resource profiles |
| **Step 3B** | Modified | MAC queue includes hybrid archetypes |
| **Step 4** | Modified | All analysis scripts handle hybrid columns |
| **Step 5** | Inherits | Procurement strategies auto-include hybrid resources via EF data |
| **Step 6** | Inherits | SMARTargets consumes Step 2 output — hybrids flow through |
| **Step 7** | Modified | Dashboard data extraction includes hybrid resources |
| **Dashboard** | Modified | New resource colors/labels in `chart-colors.js` and `shared.css` |

**Dimensionality**: Non-CAISO ISOs go from 4D → 8D (clean_firm, solar, wind, hydro + solar_batt4, solar_batt8, wind_batt4, wind_batt8). CAISO goes from 5D → 9D (adds geothermal). With offshore wind where applicable, up to 10D.

### 23.7 Compute Trimming (Offsetting Hybrid Complexity)

Three changes keep Step 1 tractable with 4 new hybrid dimensions:

1. **Drop H2 storage** — never selected as cost-optimal in any of 161 Step 2.2 parquets. Step 1.5 storage grid: 990 combos (3× reduction from 2,970).
2. **Drop 99.99% threshold** — ≥99.9% becomes the ceiling, labeled "effectively 100%" (8.76 unmatched hours/year). 20 thresholds instead of 21.
3. **Empirical resource caps** — max observed winning % + 10pp buffer per resource per ISO, constraining the grid search to proven-useful ranges. Source: `data/step2.2-cost/empirical_resource_caps.json`.
- **Implication for Step 1D.2**: The storage search IS the cost optimization mechanism
  for the last-mile thresholds (≥99%). The value is not in storage grid precision
  (LDES cost per pp is negligible) but in **unlocking cheaper resource mixes** that
  wouldn't qualify without storage. This means adequate coverage of the near-miss
  mix space is more important than fine LDES granularity.

---

## 24. Reliability Tax (Sub-Project)

A sub-project quantifying the cost gradient of pushing CFE matching from ~90% to 99.9% over 2025–2050 across the 7 ISOs, comparing four deployment pathways. **Authoritative documentation lives in `reliability_tax/README.md`** — this SPEC section captures only the locked invariants so they survive session handoff. Methodology, pathway implementation, stranding math, and results are TBD in Prompt 1B+ and will be documented here as decisions are made.

### 24.1 Locked Invariants (Prompt 1A)

These are load-bearing for every script, figure, and writeup in the sub-project. Any change requires an explicit user decision and an update both here and in `reliability_tax/README.md`.

1. **Endpoint targets**: CFE ≥ 90%, 95%, 97.5%, 99%, 99.9% by 2050.
2. **Planning horizon**: 2025–2050 (25 years; 26 year-indices including the 2025 baseline).
3. **Pathways**:
   - (1) VRE + batteries only
   - (2a) Behavioral pivot at the 90% CFE plateau
   - (2b) Economic pivot when marginal `$/CFE%` > clean firm LCOE
   - (3) Clean firm proactive from year 1
4. **Clean firm bucket**: Nuclear + CCGT+CCS + geothermal, subject to existing regional constraints (`CCS_CAP_TWH`, `GEOTHERMAL_ISOS = ['CAISO']`). **Offshore wind is NOT clean firm** — it is VRE, available to all pathways.
5. **Cost basis**: Real 2025 USD. No inflation adjustment.
6. **Cost reporting**: Undiscounted cumulative 2025–2050 plus NPV@5%, 7%, 9% real. **Objective = NPV@7%.**
7. **ISO scope**: Fully ISO-parametric. Smoke-test on a single ISO, then run all 7.
8. **Stranding scope and thresholds**:
   - Stranded fossil = **new-build gas only** with capacity factor <20% in 2050. Existing fleet is out of scope.
   - VRE stranding = curtailment >30% in 2050.
   - Transmission stranding = underutilized new-build transmission (definition to be refined in 1B).
9. **Demand growth sensitivity**: Section 2 only, reusing the existing L/M/H values from `pipeline_config.DEMAND_GROWTH_RATES`. No new growth rates are invented.

### 24.2 Reusable infrastructure

The sub-project layers on top of the existing 8-step pipeline and reuses (without modifying) `scripts/pipeline_config.py`, `scripts/procurement_utils.py::build_25yr_trajectory`, `data/step2.1-ef/`, `data/step2.2-cost/`, `data/step3-dispatch/`, and `data/step5-wrights/`. See `reliability_tax/README.md` for the full map with file paths and per-asset usage notes.

### 24.3 Status

Prompt 1A (discovery + README) complete. Prompts 1B+ will lock methodology and begin implementation. No scripts, no results, no dashboard page exist yet for this sub-project.

### 24.4 Methodology Rewrite (Apr 16, 2026)

Mid-project audit revealed the original methodology (Cards M, F, K) was landing on the wrong story. The existing optimizer (`scripts/step_2_3_pathway_optimizer.py`) never builds new gas in any pathway — it only adds clean resources and lets existing gas retire endogenously. Result: at high CFE targets, P1a (VRE-only) produces absurd VRE+storage overbuild (e.g., MISO P1a ep99.9 = 5,280 GW new-build for 237 GW peak demand = 22×) rather than the intended "stranded peaker" story. The published JSON payloads additionally carry bugs (zeroed costs, `achieved_cfe_pct=0`, mislabeled capacity) from generator scripts that read the wrong fields. Raw optimizer run JSONs are structurally valid (`analysis/reliability-tax/data/<ISO>/pathway*_ep*.json` — verified ERCOT P1a ep95 has achieved_cfe=95.25% and real cost/buildout data).

The rewrite reframes "reliability tax" as **absolute ratepayer cost under each pathway**, computed gross (not netted against capacity payments), with the comparative story told through the delta between P1a and P3.

**Card M' — Capacity market treatment (SUPERSEDES Card M).** Remove capacity-market-revenue netting entirely. Reliability tax is a gross ratepayer-cost metric. Capacity payments are a transfer within the ratepayer–developer system, not a cost reduction. ERCOT and SPP (energy-only ISOs) are no longer awkward exceptions. Any existing `capacity_rev_netted_usd` fields stay in the raw run JSON but are NOT consumed by the reliability-tax metric or the dashboard payloads.

**Card U (NEW) — New-build gas in the build menu.** The pathway optimizer is extended to allow new-build gas (CCGT only, for simplicity) as a reliability resource. Clean resources build first subject to a VRE floor ratchet (capacity year-over-year cannot decrease); any residual reliability gap after clean dispatch is filled by new-build CCGT sized to match peak-net-of-clean. Per-vintage tracking: year built, initial MW, annual CF, retirement year, cumulative recovered revenue. Cost inputs reused from `pipeline_config.py`: `NEW_CCGT_COST_KW_YR` (annualized capex+FOM by ISO, range $88–$114/kW-yr) and `EXISTING_GAS_FOM_KW_YR` ($13–$17/kW-yr). Fuel cost comes from the existing fossil-dispatch logic. Step 2.2A (`scripts/step2_2a_cost_optimization.py`) already embeds the sizing formula (residual-demand → new-gas MW) and cost application — the pathway optimizer must port this pattern, NOT duplicate constants.

**Card F' — Stranding value calc (SUPERSEDES Card F).** For every new-build gas vintage, accumulate dispatch revenue + (existing-fleet) FOM recovery year over year. If annual CF falls below 15% for two consecutive years, the vintage is marked stranded and the unrecovered portion of its book value is written off. Stranded capex for a vintage = `overnight_capex × (1 − cumulative_recovered_revenue / (capex + required_return_over_life))`. Primary reporting metric is undiscounted cumulative stranded capex 2025–2050; NPV@7% is the secondary metric. The 15% CF threshold is the developer-breakeven rule of thumb and is the single defensible knob; it must be documented on the page and sensitivity-tested against 10% and 20%.

**Card K' — Stranding definition (SUPERSEDES Card K).** Absolute, not comparative. A new-gas vintage is stranded per the Card F' CF-based test applied to its own pathway, independent of other pathways. The P1a-vs-P3 comparison survives as the **headline narrative device** (delta in reliability tax $/MWh) but not as the definition of stranding itself. Every pathway has some reliability tax; P3 has a small one (it still builds some peakers to handle tails), P1a has a large one. Existing clean firm in P2a/P2b/P3 is never stranded — it serves real load across the horizon.

**Reliability Tax formula (locked):**
```
reliability_tax_$_per_MWh[pathway, ISO]
  = [Σ_years annualized_new_gas_capex
     + Σ_years new_gas_FOM
     + Σ_years existing_gas_FOM_carried_forward
     + Σ_years priced_VRE_curtailment
     + Σ_years annualized_VRE_storage_overbuild_capex]
    ÷ Σ_years demand_MWh
```
Each component is reported separately in stacked-bar visualizations. "Overbuild" for VRE+storage = capacity whose energetic contribution (dispatched MWh) is below some CF threshold (same 15% default) — otherwise it is serving real load. Curtailment is priced at the weighted-average VRE LCOE.

**Card S (NEW) — Extended endpoint coverage.** Add 5 lower endpoint targets (60%, 70%, 75%, 80%, 85%) to the run set so the "hump" of new-gas builds at medium thresholds is visible. Full set: 60/70/75/80/85/90/95/97.5/99/99.9 = 10 endpoints × 5 pathways × 7 ISOs = 350 runs. Per-run wall-clock has been reported by the user as "~90 seconds" — budget ~8–9 hours for the full re-run. P1a ep99.9 is the single most informative run (climbs through every intermediate CFE target on the way up); if compute is tight, its year-by-year trajectory is a cheap proxy for the full endpoint sweep.

**Downstream implications:**
- `scripts/step_2_3_pathway_optimizer.py` is rewritten to implement Card U (new-gas build logic ported from step2_2a). All 5 pathways inherit the capability, but P1a/P1b still forbid new clean firm per the original Card R.
- `scripts/run_pathway_sweep.py` is extended to include the 5 new endpoints (Card S).
- `analysis/reliability-tax/data/` is regenerated end-to-end. Old MANIFEST is preserved in `analysis/reliability-tax/data-archive-2026-04-16/` for reference.
- `reliability_tax/charts/data_loader.py` is updated with the new endpoint list; `gen_section2_overbuild.py` and `gen_section3_reliability_tax.py` are rewritten against the new schema (new-gas vintages + stranding ledger); `gen_sankey.py` and `gen_four_journeys.py` are updated for Card K' framing.
- `dashboard/reliability-tax.html` narrative, charts, and section structure are rewritten against Cards M', F', K', U, S. Capacity-market references are deleted. ERCOT-specific callout acknowledges energy-only scarcity-price recovery.
- `research_paper.html`, `optimizer_methodology.html`, and any homepage mention of reliability tax / stranding are brought in line.

**Verification gate (before declaring done):**
- ERCOT P1a ep95: new-gas build nonzero in at least one year between 2030–2045, and at least some portion is CF<15% by 2050 (i.e., the hump+strand pattern appears).
- No JSON payload contains `capacity_rev_netted_usd`, `net_fom_million_usd`, or any "net of capacity" language.
- `total_new_build_gw / peak_demand_gw` for any pathway does not exceed 5× (sanity ceiling; current MISO 22× indicates model failure).
- Headline tax for P1a > headline tax for P3 in every ISO (directionally, the delta is the whole story).

**Status (Apr 17, 2026).**
- `dashboard/reliability-tax.html` rewritten against v2 methodology (Cards M'/F'/K'/U/S). 9 sections: hero, setup, THE HUMP, abandonment, tax decomposition, journey cards, stranding Sankey, cost of waiting, 175-row summary table.
- Capacity-market netting, developer-ROI framing, and comparative-to-P3 stranding language removed from all dashboard copy.
- ERCOT/SPP callout added: ratepayers pay reliability tax via scarcity-priced energy rather than capacity payments.
- `optimizer_methodology.html`, `index.html`, `abatement_dashboard.html` propagated to match the v2 framing.
- Remaining work: `research_paper.html` does not exist in the repo; if/when authored, its reliability-tax sections should follow the v2 cards.

### 24.5 Worst-Hour Gas Sizing (Apr 16, 2026)

**Decision (locked).** Replace the fixed-ELCC `total_clean_peak` formula in `scripts/step2_2a_cost_optimization.py` (scalar path ~385–431 and duplicate batch path ~707–746) and in `scripts/step_2_3_pathway_optimizer.py` (`size_required_gas_mw`, `_clean_peak_mw_from_mix`, `_existing_clean_peak_mw`) with a dispatch-driven sizing signal computed from the 8760 `total_clean` profile in `data/step3-dispatch/{ISO}_dispatch_cache.parquet`.

**Percentile (locked — option B-i).** Gas is sized to the **99.97th percentile** of the margin-inclusive residual-gap distribution:
```
residual[h]   = max(0, demand[h] × (1 + RESOURCE_ADEQUACY_MARGIN) − total_clean[h])
gap_mw        = np.percentile(residual, 99.97)            # LOLE ≤ 2.6 h/yr
gas_raw       = gap_mw / GAS_AVAILABILITY_FACTOR[iso]
```
RA-margin convention (locked Apr 16, 2026 — amended from initial "margin on gap" formulation after validation exposed a false-positive invariant violation in clean-firm-heavy mixes). The 15% reserve margin is applied to **demand** per-hour before subtracting clean supply — the NERC/PJM/MISO RA-planning tradition. Rationale: a reserve margin is a safety factor on the demand forecast, not on the post-subtraction residual; under this convention the 2025-baseline and every scenario use the same hourly arithmetic and the worst-hour sizing is a strict physical upgrade over the legacy ELCC formula (`WH_gas_needed ≥ ELCC_gas_needed` for every mix — no exceptions). The 99.97th percentile leaves ≈2.6 h/yr of residual tail above the sized gap, consistent with the NERC / PJM "1-day-in-10-years" LOLE ≈ 2.4 h/yr benchmark. This is a single deterministic knob — sensitivity against A (1-hour maximum) and B-ii (99.94th ≈ 5 h/yr) can be run without structural code changes by flipping the `p` parameter.

**Why not ELCC.** The fixed `PEAK_CAPACITY_CREDITS` (solar 0.30, wind 0.10, battery 0.95, etc.) are annual-average approximations. In VRE-heavy mixes the Σ-of-credits balloons past total peak demand (ERCOT ep99.9: `total_clean_peak` = 907 GW vs `ra_peak` = 227 GW, implying zero gas need) while the actual 8760 dispatch shows real residual gap. In winter-peaking ISOs, solar's average 0.30 credit overstates its peak-hour contribution (which is ≈0). Worst-hour dispatch is the truth; ELCC is a simplification that breaks at the extremes we care about.

**ELCC retained — diagnostic only.** `_elcc_gas_mw(...)` stays in the codebase as a bug-detection helper invoked by `scripts/tmp_validate_worst_hour_sizing.py`. It is **not** used in any production sizing path. Invariant under the amended margin-on-demand convention: for any mix, `worst_hour_gas_mw ≥ elcc_gas_mw`. VRE-heavy mixes should show large positive deltas (ELCC under-sizes because it over-credits variable resources); clean-firm-heavy mixes should show near-zero deltas (ELCC ≈ WH when both apply the RA margin on demand and clean supply is nearly flat). A strict violation signals a dispatch-profile or archetype-lookup bug and is treated as a hard stop.

**On-demand archetype expansion.** If a mix's archetype key is not present in the dispatch cache, the sizing helper calls into `scripts/step3a_build_dispatch_cache.py` to compute and persist the missing archetype's 8760 profile, then re-reads the cache. Missing archetypes are batched per ISO (single Step 3a call per batch, not per-mix) and the cache grows monotonically — subsequent hits on the same key are direct reads. No fallback to ELCC for missing archetypes. Cache expansions must be committed alongside any rerun so downstream sessions inherit the expanded cache.

**Architecture (locked Apr 17, 2026 after scalability audit).** Option A — **in-loop ELCC, post-process worst-hour on winners only.** The initial implementation tried to compute worst-hour sizing per candidate inside the step-2.2A pricing path. Audit surfaced that the EF candidate pool is 99.97% unique archetypes (CAISO ep95: 31,173 unique / 31,181 rows; all-thresholds: ~21 M unique / ISO), so the dispatch-cache amortization is broken at that layer — an O(N_unique × dispatch_time) work profile that balloons to >175 hours per ISO. Architecture change:
- `scripts/step2_2a_cost_optimization.py` pricing paths (`price_mix_batch` and `precompute_base_year_coefficients`) retain the **vectorized ELCC formula** for the in-loop `gas_needed_mw` column (unchanged from the pre-worst-hour baseline — fast, per-candidate, scales to 21M mixes trivially).
- Immediately after all of an ISO's parquets are written (baseline, tracks, DG per threshold, feasible), `_rewrite_gas_columns_worst_hour` overwrites the `gas_*` columns **row-by-row using the SPEC §24.5 worst-hour formula**, computed via the dispatch cache. Because only unique winners are in the output (hundreds to low-thousands of archetypes per ISO), on-demand archetype expansion and percentile lookup run in their intended regime (seconds per ISO). gas is not in the cost function, so optimizer cost-rankings are identical to the pre-worst-hour baseline.
- `scripts/step_2_3_pathway_optimizer.py::size_required_gas_mw` continues to call the scalar worst-hour helper directly — N=1 per call, a few thousand unique mixes per full 350-run sweep, within cache regime. No post-process layer needed.
- The dispatch-cache expansion policy is unchanged: missing archetypes are computed via step3a's `expand_cache_for_mixes` and persisted to disk, monotonic growth, commit alongside results.

**Blast radius (documented for downstream sessions).**
- `data/step2.2-cost/*.parquet` requires a full 27M-mix rerun (hours of compute; do NOT launch without explicit approval).
- `analysis/reliability-tax/data/` 350-run sweep must be regenerated after step 2.2 lands.
- `dashboard/js/shared-data.js` (Step 7) rebuilds propagate to `dashboard.html`, `abatement_dashboard.html`, `research_paper.html`, `lmp_trends.html`, `optimizer_methodology.html`, and `reliability-tax.html`.
- Dispatch cache parquets grow; commit alongside results.

**Verification gate (before any rerun).**
- Temp script `scripts/tmp_validate_worst_hour_sizing.py` prints a 10-mix × 3-ISO (ERCOT, NYISO, CAISO) diff table of `{ELCC_mw, worst_hour_mw, delta_pct}`. Expected pattern: VRE-heavy mixes → worst-hour > ELCC; clean-firm-heavy mixes → approximately equal. Any `worst_hour < ELCC` is a hard stop.
- Validation must exercise the on-demand archetype expansion path with ≥2 cache-miss mixes; the second read must hit the newly-written rows.
- User-approved validation table is a prerequisite for step-2.2 rerun.

### 24.6 Peak-Year Gas Fleet Snapshot (Apr 17, 2026)

**Context.** Audit of `analysis/reliability-tax/data/ERCOT/pathway1_ep90.json` revealed the reported new-build gas fleet (457 GW) was ≈4× any physically defensible stock — ERCOT existing gas is 55 GW and 2050 peak demand under Medium growth is ≈197 GW. Three compounding bugs, two fixed in this section.

**Bug 1 — Cross-endpoint fleet seeding (FIXED).** `scripts/run_pathway_sweep.py` chained endpoint runs per pathway via `--seed-run`, passing each terminal `new_gas_fleet` into the next endpoint's `initial_fleet`. Result: ep99.9's reported fleet was a UNION of every vintage ever built across ep60 → … → ep99.9. A standalone (pathway, endpoint, growth) scenario is a 2025–2050 trajectory — each runs independently. **Fix.** `PlannedRun.seed_run_path` removed, `run_pathway_sweep.sweep` no longer tracks `last_output`, `solve_pathway/_solve_and_annotate/run_pathway` accept `initial_fleet=None` for signature compatibility but ignore it (explicit `del initial_fleet`). The `--seed-run` CLI flag is retained as a no-op with a deprecation warning so old invocations don't crash.

**Bug 2 — Cumulative build every year as demand grows (FIXED).** Old intra-run loop: if `sizing['new_gas_required_cumulative_mw'] > already_built_new_mw`, add a new vintage that year. Demand growth (3.5 %/yr in ERCOT, compounding 2.36× over 26 years) makes the requirement monotonically grow whenever CFE plateaus — producing a new vintage every single year. The fleet becomes the SUM of these annual increments instead of the maximum of the requirement trajectory. **Fix.** `solve_pathway` now runs in two phases:
```
Phase A (main loop, years 2025–2050):
  compute target, sizing, CFE, new_gas_required_cumulative_mw[y]
  (intra-run clean-resource ratchet via ledger is UNCHANGED)
Phase B (post-loop aggregation):
  fleet_size_mw   = max(new_gas_required_cumulative_mw[y])
  peak_year       = argmax year
  active_mw[y]    = running max through year y (monotonic ratchet up to peak, then flat)
  new_gas_need_2050_mw = new_gas_required_cumulative_mw[2050]
  stranded_mw_at_2050  = fleet_size_mw − new_gas_need_2050_mw
  years_in_service_2050 = 2050 − peak_year
  years_remaining_2050  = max(0, NEW_GAS_ASSET_LIFE_YEARS − years_in_service_2050)
  stranded_capex_usd    = stranded_mw × 1000 × CCGT_OVERNIGHT_CAPEX_USD_KW × (years_remaining_2050 / NEW_GAS_ASSET_LIFE_YEARS)
```
A single consolidated vintage is booked at `peak_year` with `initial_cap_mw = fleet_size_mw`. Card F' CF-streak trigger is retained as a diagnostic (annual CF list) but is NO LONGER the stranding trigger — peak vs. 2050 need is the deterministic rule.

**Bug 3 — Double-applied demand growth in the size_required_gas_mw call (FIXED).** `worst_hour_gas_sizing(iso, mix, storage, demand_twh, gf)` expects `demand_twh` to be BASE-year demand and applies `gf` internally (`demand_mwh_grown = demand_twh * 1e6 * gf`). The pathway optimizer was passing `demand_for_year(iso, year, growth)` — which is already grown — producing a `demand^2 × gf` scaling that inflated gas need by the growth factor again (≈2.36× for ERCOT 2050 Medium). **Fix.** Pathway optimizer now passes `pc.REGIONAL_DEMAND_TWH[iso]` (base) into `size_required_gas_mw`.

**Output schema changes.** The per-run JSON `stranding_metadata` dict adds:
- `methodology: "peak_year_snapshot_v2"`
- `fleet_size_mw: <MW>`
- `peak_year: <YYYY>`
- `new_gas_need_2050_mw: <MW>`
- `stranded_mw_at_2050: <MW>`
Legacy keys (`cf_threshold_default`, `cf_threshold_sensitivity`, etc.) stay for back-compat with downstream readers. `tables.new_gas_fleet` is now a single-vintage list (year_built = peak year, stranded_capex_usd = peak-to-2050 write-off). Annual buildout `gas_sizing.new_gas_built_this_year_mw` is non-zero only at `peak_year`; `active_new_gas_fleet_mw[y]` is the running-max ratchet trajectory.

**Empirical validation (ERCOT P1, Medium growth, medium costs).** New fleet sizes are now physically defensible:
- ep60: 53 GW new (was 250 GW) — peak = 2050, stranded = 0.
- ep90: 111 GW new (was 457 GW) — peak = 2050, stranded = 0.
- ep99.9: 73 GW new (was 457 GW) — peak = 2039, stranded = 10 GW, stranded capex = $6.7 B.
At high CFE endpoints, gas CF collapses to <1 % by 2050 (73 GW fleet carrying for reliability only), which is the "hump + strand" story the methodology is meant to show.

**Downstream implications.**
- `analysis/reliability-tax/data/` requires a full 350-run regeneration. Old `MANIFEST.json` preserved as `MANIFEST.stale-*.json`; the stale-ELCC backup from §24.5 is untouched.
- `reliability_tax/charts/*.py` readers consume fields that still exist (`tables.new_gas_fleet[]`, `reliability_tax.components_usd`, `gas_sizing.active_new_gas_fleet_mw`, `gas_sizing.gas_fleet_cf`). No chart regenerator changes are required for Step 1; they will be re-run after all three steps land.
- Each (pathway, endpoint) run is now idempotent — re-running ep90 alone does NOT depend on ep60 being fresh.

**Future-work knobs left in place.**
- `NEW_CCGT_COST_KW_YR` (per-ISO annualized capex+FOM) is still the capex recovery knob — single vintage pays `fleet_size × NEW_CCGT_COST_KW_YR × 1000` every year it is active.
- `EXISTING_GAS_FOM_KW_YR` carries existing-gas FOM on the full `EXISTING_GAS_CAPACITY_MW` nameplate every year (unchanged — that is the "we keep the plant on the grid for reliability" story).
- `CCGT_OVERNIGHT_CAPEX_USD_KW` and `NEW_GAS_ASSET_LIFE_YEARS` are the stranding write-off inputs.

**Scope boundary.** Step 1 (this section) fixes gas-fleet sizing only. VRE stranding (priced curtailed MWh × vintage LCOE) is Step 2 (see §24.7). Pathway-specific Wright's Law NOAK years + Pathway 3 proactive clean-firm floor are Step 3 (pending). Both tracked in this session's Current Status.

### 24.7 Priced VRE Curtailment (Apr 17, 2026 — Step 2)

**Context.** Step 1 fixed gas-side reliability tax; the VRE side of the same ledger was still dark. `tax_components_cumulative['priced_vre_curtailment_usd']` was hardcoded to zero, so high-CFE VRE-only pathways appeared to have no stranded energetic cost even though those mixes curtail 40–60 % of their gross VRE generation. Step 2 prices that curtailment year by year against each resource's locked vintage LCOE, capturing the "build more VRE than the grid can absorb and pay the capex anyway" dynamic that is the VRE analog of the Card F' new-gas stranding story.

**Locked formula (Step 2).**
```
priced_vre_curtailment_usd[y]
  = Σ_r  surplus_frac[r, y] × demand_mwh[y] × vintage_lcoe[r, y]

surplus_frac[r, y]    = sum of `surplus_<r>` 8760 profile from the dispatch
                        cache entry keyed on (target_mix_pct, storage_pct)
                        for year y's selected target.
                        Cache is demand-normalized → sum is fraction of annual demand.
vintage_lcoe[r, y]    = TWh-weighted average locked_lcoe across all VintageLedger
                        entries with resource = r and cod_year ≤ y <
                        retire_year. Weights are v.twh_per_year.

r iterates over       : solar, wind, offshore_wind,
                        solar_batt4, solar_batt8, wind_batt4, wind_batt8.
                        Hybrid surplus (solar_batt4, etc.) is priced at the
                        underlying VRE ledger key (solar or wind), NOT a
                        hybrid-specific key — the battery did not produce the
                        curtailed MWh, so battery capex is not part of the
                        stranded energetic cost.

priced_vre_curtailment_usd (cumulative)
  = Σ_y priced_vre_curtailment_usd[y]   for y in 2025..2050.
```

**Why vintage-weighted LCOE (and not current-year marginal LCOE).** The $/MWh a developer "paid" to generate curtailed energy is locked at the year of commercial operation per Card N. A solar vintage built in 2029 carries 2029's LCOE for life; pricing its 2045 curtailment at 2045's Wright's-Law-discounted LCOE would understate the stranded capex by the full learning-curve delta. The ledger already records `locked_lcoe` per vintage, so the correct cost signal is available without additional state.

**Implementation.** `scripts/step_2_3_pathway_optimizer.py` adds four helpers (`_vintage_weighted_lcoe`, `_dispatch_cache_entry`, `_archetype_key_for_target`, `_per_year_vre_curtailment_usd`) above `compute_vre_curtailment_at_endpoint`. The dispatch cache is populated on-demand by `size_required_gas_mw` during the first pass of `solve_pathway`; the second pass reads from the step2_2a in-memory cache (falling back to disk if absent). The per-year priced curtailment is accumulated into `tax_components_cumulative['priced_vre_curtailment_usd']` and mirrored per-row as `priced_vre_curtailment_usd_this_year` in `annual_cost` for diagnostics.

**Edge-case behavior (locked).**
- Archetype missing from cache after `size_required_gas_mw` expansion → year's priced curtailment = 0. Emits no warning; the cache expansion path is an invariant of Step 1. If this ever fires, it signals a Step 1 regression, not a Step 2 bug.
- No active vintages for resource r in year y → vintage LCOE = 0 → priced curtailment for that slice = 0. Consistent with "no book cost to strand" — existing fleet that predates the ledger is intentionally not charged a stranding tax by Step 2 (it was treated as pre-paid sunk cost in every prior Card).
- Target mix has resource r at 0 % → skipped. Avoids spurious lookups of surplus profiles that are structurally zero.
- `surplus_<r>` absent from cache entry (older cache version) → resource skipped. Cache v2+ always includes these columns; older caches trigger the skip path silently.

**Output schema change.** `annual_cost[]` rows now carry `priced_vre_curtailment_usd_this_year`. The `reliability_tax.components_usd['priced_vre_curtailment_usd']` field is unchanged in shape but now flows a non-zero value for VRE-heavy pathways. No legacy keys removed.

**Empirical validation (ERCOT P1, Medium growth, medium costs — test suite).**
- ep60: $0.00 B priced VRE curtailment (low-CFE VRE mix carries no surplus).
- ep99.9: $193.58 B priced VRE curtailment, ramping year-over-year from $0 through 2027 → $25.2 B in 2050 as the CFE target climbs and the VRE overbuild factor grows. This is ~5× the cumulative Step 1 gas reliability-tax components ($94.97 B new-gas capex + $18.59 B existing-gas FOM), correctly placing the VRE-only pathway's dominant cost pressure on stranded VRE capex rather than on the small residual peaker fleet.
- $/MWh reliability tax at ep99.9: $15.23, vs $2.51 at ep60 — the "hump + strand" story now spans both sides of the generation stack.

**Scope boundary.** Storage overbuild capex (`vre_storage_overbuild_capex_usd`) remains at 0 in the solver and is still planned for a later step. Step 3 lands the pathway-specific Wright's-Law NOAK years and Pathway 3 clean-firm floor (§24.8).

### 24.8 Exogenous Per-Pathway NOAK + P3 Clean-Firm Floor (Apr 17, 2026 — Step 3)

**Context.** Steps 1 (§24.6) and 2 (§24.7) fixed the gas-fleet and VRE-stranding cost signals. The third and final reliability-tax fix addresses the *pathway differentiation* problem audited at the top of this session: Pathways 1 and 3 were producing nearly identical results at many endpoints because (a) the clean-firm learning curve did not reflect pathway-specific deployment timing, and (b) `_filter_pathway_3` was a no-op — P3 ran against the full EF and routinely picked the same cheapest-VRE-only rows P1 picked. Step 3 replaces both of those endogenous/identity-filtered behaviors with exogenous deterministic policy signals.

**Decision 1 — Per-pathway NOAK year override (locked).** Each pathway commits to clean-firm scale-up at a different moment in the horizon, and that moment dictates when the Wright's Law curve lands its NOAK floor. We encode this as an exogenous per-pathway NOAK year, replacing the default tech-specific NOAK date from `LEARNING_PARAMS`:

| Pathway | Interpretation | NOAK year | Rationale |
|---------|----------------|-----------|-----------|
| P3 | Proactive clean firm from year 1 | **2035** | Early commitment + sustained deployment drives the learning curve to floor fast. |
| P2b | Economic pivot trigger (~plateau) | **2040** | Approximately matches the default M-level NOAK; pivot coincides with the natural learning window. |
| P2a | Behavioral SBTi-90% plateau pivot | **2045** | Late pivot postpones deployment; slower cumulative installs push NOAK further out. |
| P1 / P1a / P1b | VRE-only (no clean-firm build) | *default* | Override is irrelevant — no clean-firm vintages are priced via the learning curve. |

**Scope (locked).** The pathway-specific NOAK override applies to clean-firm techs only — `PATHWAY_NOAK_TECHS = {'nuclear', 'ccs', 'geo'}`. Battery, VRE, LDES, and H2 learning curves are driven by global markets rather than US-pathway deployment choices and keep their default `LEARNING_PARAMS` windows.

**Implementation.** `pipeline_config.py` gains `NOAK_YEAR_BY_PATHWAY`, `PATHWAY_NOAK_TECHS`, and `get_pathway_noak_window(tech, level_short, pathway)` which returns `(foak_start, noak_year)` with the override applied when (a) pathway is in the override dict and (b) tech is in the scope set. `scripts/step_2_3_pathway_optimizer.py::_learning_window` accepts an optional `pathway` arg and delegates to the new pipeline_config helper. The three clean-firm LCOE helpers (`nuclear_newbuild_lcoe_at_year`, `ccs_lcoe_at_year`, `geothermal_lcoe_at_year`) accept and forward `pathway`; `marginal_lcoe`, `cheapest_clean_firm_lcoe`, `compute_clean_firm_tranches_for_year`, and `_clean_firm_total_cost_batch` pull it from `config.pathway` and thread it through. `foak_start` is never moved; only the NOAK terminal year shifts, so FOAK-era cost is identical across pathways by construction.

**Decision 2 — Pathway 3 clean-firm floor (SUPERSEDED).** An initial Step 3 v1 commit (`acf5bd7`) shipped with a clean-firm floor `clean_firm ≥ k × threshold_pct` on P3 at `k = 0.30`. User rejected it immediately on methodology grounds: *"a floor hard-wires the finding (P3 ≠ P1 by construction) and smothers the regional heterogeneity that IS the scientific content — ERCOT (VRE-rich) legitimately converging toward VRE+storage even under NOAK-2035 is a finding, not a bug."* The floor was removed in Step 3 v2 (`8f87a56`); `_filter_pathway_3` is now a no-op and P2a/P2b/P3 all see the full EF from year 1.

**Decision 2 v2 (LOCKED) — P2a/P2b/P3 are exogenous-NOAK-only.** P2a/P2b/P3 are differentiated *solely* by `pc.NOAK_YEAR_BY_PATHWAY` (Decision 1 above). All three see the full EF from year 1; none carry pre-pivot filters, clean-firm floors, or endogenous pivot triggers. The `PivotState` dataclass is retained for JSON-schema compat but is never `.trigger()`'d — `should_pivot_2a` / `should_pivot_2b` live in the module as dead code for reference. The cost-optimal mix surfaces organically in response to the accelerated Wright's Law curve each pathway sees; the sharpness of the P3-vs-P1 divergence in a given ISO is itself the finding.

**Decision 3 (LOCKED — commit c576cec) — Pre-seeded VintageLedger + base-year clean-firm target.** Two interlocking changes close the "identical mix, different cost" accounting gap surfaced in Step 3 v2:

1. **Pre-seed existing fleet.** `solve_pathway` calls `_seed_existing_fleet_vintages(iso)` before any new-build booking. The helper produces one zero-LCOE `Vintage` per resource in `GRID_MIX_SHARES[iso]` with TWh = `BASE_DEMAND_TWH × share / 100`, `cod_year = BASE_YEAR − 1 = 2024` (so `ledger.active(y)` sees it from year 1 onward), `locked_lcoe = 0.0` (sunk cost), `tx_adder = 0.0`, `retire_year = None`. A new ledger key `clean_firm_existing` distinguishes the sunk-cost fleet from new-build tranche keys (`uprate`, `geothermal`, `nuclear_newbuild`, `ccs_ccgt`) and is included in the `existing_cf` subtraction sum inside `_derive_delta_vintages`. This makes the `VintageLedger` a single source of truth for "physical assets this pathway operates" across existing + new vintages.

2. **Clean-firm target uses BASE demand.** The EF row encodes `clean_firm` as a share of base-year demand. The previous implementation multiplied by year-t grown demand, creating a ratchet (ERCOT 2050 Medium growth: 9 % × 1153 = 104 TWh vs 9 % × 488 = 44 TWh) that booked new clean firm every year purely to maintain share. Post-fix `cf_target_twh = _twh_from_pct(cf_pct, pc.REGIONAL_DEMAND_TWH[iso])` is invariant across years. Combined with the pre-seed, `cf_pct ≤ existing-fleet share → zero new clean firm booked` across all pathways, and any `cf_pct > existing share` is booked once in year 1 as the absolute delta (no demand-growth ratchet). The `nuclear_cap_twh` knob is retained for P1/1a/1b as a soft cap (`min()` preserves the existing-fleet ceiling), and the `_is_existing_fleet_only` branch is kept so P1 continues to skip the clean-firm tranche block entirely.

**Why base-demand semantics for clean-firm.** Cross-pathway identical-mix identity (the ERCOT VRE-rich finding) requires P3 to not book new clean firm whenever P1 is structurally unable to. P1's `_is_existing_fleet_only` shortcut means P1 never adds clean firm regardless of year-t demand; matching that behavior in P2a/P2b/P3 without re-introducing a pathway-specific filter requires `cf_target_twh` to be frozen at base-year absolute TWh rather than scaled with grown demand. VRE, storage, and hybrid targets *do* continue to scale with demand (energy_resources loop uses `demand_twh`) — that's consistent with the "maintain share" interpretation for resources that are assumed to be incrementally built out over the horizon. Clean firm is the methodology-driven exception because it is the pathway-differentiating resource.

**Ripple effects (documented).**
- **Priced VRE curtailment** (§24.7) shifts modestly because `_vintage_weighted_lcoe` now includes the zero-LCOE existing VRE vintages in the denominator, diluting the TWh-weighted LCOE. Empirical: ERCOT P1 ep99.9 curtailment $193.58 B → $182.50 B (−5.7 %). This is consistent with §24.7's "existing fleet predating the ledger is not charged a stranding tax" principle; the new-build VRE capex recovery cost is still fully captured.
- **Stranding ledger** (`_book_value_stranded`) walks vintages newest → oldest. The pre-seeded existing vintages (cod_year 2024) are oldest; any stranded TWh attributed to them contributes zero book value. Net-zero effect on the comparative stranding story.
- **No schema changes** to `annual_cost`, `reliability_tax`, `stranding_metadata`, or `endpoint_mix_pct`. The accounting fix is internal to `_derive_delta_vintages` + `solve_pathway` + one new helper.

**Empirical validation (post-fix, ERCOT + PJM, Medium growth, medium costs, commit c576cec):**

| ISO | endpoint | P1 undisc. | P3 undisc. | Δ | P3 mix (cf / sol / wind) | Notes |
|---|---|---:|---:|---:|---|---|
| ERCOT | ep60 | $525.1 B | $526.3 B | **+0.23 %** | 9 / 14 / 42 (≡ P1 every year) | Passes the <1 % gate. Residual is a 1.9 TWh year-1 delta from the 9 % EF row vs the 8.6 % existing share. |
| ERCOT | ep90 | $1,514.5 B | $1,460.1 B | −3.59 % | 0 / 0 / 90 (endpoint ≡ P1) | Intermediate years 2032–2039 diverge: P3 picks uprate + CCS under NOAK-2035 while P1 stays pure-VRE. Real methodology divergence, not accounting. |
| ERCOT | ep99.9 | $3,146.2 B | $3,137.0 B | −0.29 % | 10 / 0 / 41 (≈ P1 at cf=9) | Endpoint differs by 1 pp clean firm under NOAK-2035. |
| PJM | ep90 | $7,167.8 B | $2,390.3 B | **−66.65 %** | 79 / 0 / 10 (strong divergence) | Central PJM finding sharpens: pre-fix −53.5 % → post-fix −66.65 % because P1 no longer benefits from phantom existing-fleet scaling. VRE-curtailment tax for P1 at ep90 = $275 B; P3 = $0. |

- **ERCOT result survives** — VRE-rich regions stay VRE+storage even at NOAK-2035. P1 and P3 pick identical every-year mixes at ep60, and identical endpoint mixes at ep90 + ep99.9 with only NOAK-driven intermediate divergences.
- **PJM result survives and is crisper** — VRE-constrained ISOs with weak winter solar see cheap-clean-firm beat VRE+storage under the NOAK-2035 curve; the accounting fix removes the phantom existing-fleet premium that had been understating the divergence.

**Downstream implications.**
- `analysis/reliability-tax/data/` 350-run sweep is now unblocked. Launch via `python3 scripts/run_pathway_sweep.py`. Expected wall clock ~8–9 hours end-to-end across 7 ISOs × 10 endpoints × 5 pathways.
- No Step 1 / Step 2 cache invalidation — the accounting fix affects only in-loop `solve_pathway` state (ledger seeding + target computation), which is recomputed per run.
- No schema changes to `annual_cost`, `reliability_tax`, or `stranding_metadata` beyond Step 2's `priced_vre_curtailment_usd_this_year` field.
- Dispatch cache (`data/step3-dispatch/*.parquet`) grows monotonically as the sweep expands archetype coverage; commit alongside the 350 per-run JSONs.

**Scope boundary.** Step 3 v2 (`8f87a56`) + accounting fix (`c576cec`) is the final methodology for pathway differentiation. The accounting fix is a bug repair, not a methodology change — it restores the property that identical mixes produce identical costs across pathways while preserving the exogenous-NOAK divergence that is the scientific content of the sub-project.

### 24.9 Dashboard regeneration against v2 sweep (Apr 18, 2026)

**Context.** Step 3 v2 accounting fix (`c576cec`) + 350-run sweep (`c7868df`) left the dashboard chart payloads and `reliability-tax.html` narrative stale. This session regenerated every payload from the banked sweep and walked the dashboard + cross-page references to reflect the real §24.6 / §24.7 / §24.8 numbers.

**Scope (landed this session on branch `claude/regen-reliability-tax-v2-d0U5j`).**

**Phase A — chart regeneration.**
- `reliability_tax/charts/data_loader.py`: `PATHWAYS` corrected to the actual sweep set `["1", "1a", "2a", "2b", "3"]` (was `["1a", "1b", "2a", "2b", "3"]`; pathway 1b retired before v2). Docstring updated to flag the v2 schema changes — non-zero `priced_vre_curtailment_usd` for VRE-heavy pathways, non-None `stranding_metadata.peak_year`, pre-seeded `terminal_ledger` at `cod_year = 2024` per §24.8.
- `gen_section3_reliability_tax.py`: priced-VRE-curtailment and VRE+storage-overbuild components now read directly from the solver's `reliability_tax.components_usd` dict instead of recomputing a local mix-excess proxy. This resolves the "Action item (open)" from the prior Current Status block. P1 now shows the real §24.7 vintage-weighted curtailment stack ($9.19/MWh on PJM P1 ep90, dominant at high CFE).
- `gen_closing_summary_table.py`: per-ISO aggregation switched from ep95 P1a-vs-P3 to ep90 P1-vs-P3 (matches §24.8 empirical table); column description updated; pathway-listing text refreshed.
- `gen_section1_narratives.py`, `gen_section1_worst_hours.py`, `gen_section2_gas_hump.py`: headline VRE pathway swapped from `"1a"` → `"1"` in all data lookups, labels, and print statements. `gen_section6_cost_of_waiting.py` keeps `"1a"` as the strict-onshore "never commit" anchor and documents that choice inline.
- `gen_sankey.py`: added §24.8 seed-vintage note to `meta.note` (consumer reads `tables.new_gas_fleet[]` with `cod_year >= 2025` by construction; seed vintages are not double-counted).
- `gen_act1_trilemma.py`: `reliability_tax_definition` rewritten to the gross Card M' framing (dropped "net of capacity market revenues" language).
- All 12 generators re-run without error; stdout/stderr captured to `logs/rt_charts_regen.log`. Output parquets for `reliability_tax/charts/*.json` and mirrored `dashboard/js/reliability-tax/*.json` all parse.

**Phase B — HTML narrative update (`dashboard/reliability-tax.html`).**
- Pathway wiring across Sections 2/3/4/5/6/8 switched from `[1a, 1b, 2a, 2b, 3]` → `[1, 1a, 2a, 2b, 3]`. P1b button removed everywhere; P1 added as the headline VRE pathway; P1a retained as the strict-onshore Card R baseline.
- Section 4 delta-insight JS rewritten to distinguish VRE-rich convergence (ERCOT, SPP — "P1 ≡ P3 within ~$1/MWh") from VRE-constrained divergence (PJM, NYISO, NEISO — "P3 cuts the reliability tax by 50–70%"). Verdict copy fires off the sign of the P1 − P3 delta and the ISO identity.
- Section 5 journey subtitle + new green-accent insight box surfaces the two §24.8 findings verbatim (ERCOT P1 ≡ P3 at every endpoint within ~1% on cost; PJM P3 saves 66.7% of undiscounted cost at ep90, wipes the $275B priced-curtailment bar). Journey cards now render at ep90 (the §24.8 central-finding endpoint) rather than ep95.
- Section 2 "hump" insight updated to §24.6 peak-year-snapshot semantics (ERCOT P1 peak = 134 GW @ ep80 not 111 GW @ ep90 as in the pre-fix data; PJM P1 peak = 106 GW @ ep80; MISO peak = 70 GW @ ep90).
- Section 6 Sankey subtitle reframed for the §24.6 single-consolidated-vintage-at-peak schema (drops the old vintage-year histogram framing).
- Section 7 cost-of-waiting insight cites the §24.8 NOAK-2035 window explicitly; retains P1a as the "never commit" anchor.
- ERCOT/SPP callout (capacity-market-vs-scarcity-pricing language) preserved verbatim — methodology-neutral and unchanged by v2.

**Phase C — cross-page propagation.**
- `dashboard/optimizer_methodology.html`: Card K' description (P1a → P1 headline), Card S pathway list corrected, new §24.8 card appended (per-pathway exogenous NOAK + VRE-rich convergence finding), priced-curtailment formula block rewritten to the §24.7 formal form.
- `dashboard/index.html` Act 4: scanned; no hardcoded reliability-tax $/MWh or $B/GW numbers. Narrative uses paper-Strategy-1B/2C framing from the separate main-optimizer shared-data.js pipeline, not the reliability-tax sweep. Methodology-neutral language compatible with v2. No edits required.
- `dashboard/abatement_dashboard.html`: no reliability-tax references in page body. No edits required.
- `research_paper.html`: not in the repo. Skipped per §24.4 Status note.

**Headline numeric shifts (pre-regen → post-regen, selected).**
```
hook_reliability_tax.json (cross-ISO P1 vs P3 @ ep95, feasible ISOs sum):
  pathway1_total_usd                $18.93T  ->  $17.94T
  pathway3_total_usd                $16.09T  ->  $13.84T
  delta_pct                           17.70% ->   29.60%

section3_reliability_tax.json (rtax $/MWh):
  ERCOT P1 ep90                       N/A    ->  $11.36/MWh   (new P1 track)
  ERCOT P3 ep90                      $19.23  ->  $11.34/MWh
  PJM P1 ep90                         N/A    ->  $12.94/MWh
  PJM P3 ep90                        $11.09  ->   $2.47/MWh   (-66.7% cost vs P1)

closing_summary_table.json by_iso ($/MWh, swap ep95 P1a→ep90 P1):
  CAISO  +2.59 ->  +12.10    (VRE-tight, P3 still wins)
  ERCOT  +0.00 ->   +0.02    (P1 ≡ P3 convergence gate)
  MISO   +0.60 ->   +1.07
  NEISO  +0.87 ->   -2.56    (P1 cheaper — VRE + offshore suffices)
  NYISO  +0.89 ->   +0.00
  PJM    -0.04 ->  +10.47    (P3 saves 66.7% cost; $10.47/MWh rtax)
  SPP    +0.00 ->   -3.55    (VRE-rich — P1 actually cheaper)

section5_stranding_sankey.json (Card K' absolute @ ep95):
  total_stranded_capex            $5,639B  ->   $14.6B
```
The $5.6T → $15B stranding collapse is a direct consequence of dropping the old Card K comparative framing: VRE book-value deltas were being attributed as "stranding" pre-v2, while Card K' absolute counts only new-gas vintages that trip the CF<15%-for-2-years Card F' test.

**Data-loader schema changes (locked).**
- `data_loader.PATHWAYS = ["1", "1a", "2a", "2b", "3"]`.
- Module docstring enumerates v2 schema invariants that every downstream consumer must honor: (a) read `priced_vre_curtailment_usd` from solver, not recompute; (b) `stranding_metadata.peak_year` can be 2050 (meaning "no stranding") but is never None; (c) `terminal_ledger` top-level list now includes pre-seeded `cod_year=2024, locked_lcoe=0` entries that must be either filtered to `cod_year >= 2025` for new-build attribution or bucketed as "existing fleet" via `resource == 'clean_firm_existing'`.
- No generator currently iterates `terminal_ledger` directly, so (c) is a forward-looking invariant for any future generator that touches vintage ledgers.

**Scope boundary.** Dashboard regeneration is complete. No further optimizer runs required. The next open item is the optional 7-ISO × 3-endpoint sensitivity sweep at the L/H cost corners (Medium only is banked) — gated on user authorization per the Pre-Run Gate rule.
