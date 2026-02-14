# Claude Code — Session Continuity Instructions

## If Resuming This Project

1. **Read SPEC.md first** — it contains every design decision, cost table, and implementation detail
2. **Check the todo list** or review git log to see what's been completed
3. **Branch**: `claude/enhance-optimizer-pairing-k0h9h`
4. **Repo**: `jessicacohen554-cyber/hourly-cfe-optimizer` (all advanced model work on designated branch)
# Claude Code — Session Instructions

## Workflow Preferences (Apply to EVERY Session)

### Documentation-First Development
- **CRITICAL — Decisions go to SPEC.md IMMEDIATELY**: The *very first action* after any design, methodology, or architectural decision is confirmed by the user is to write it to SPEC.md. Do NOT continue with implementation, code changes, or further discussion until the decision is captured. This is the highest-priority workflow rule — sessions can be disrupted or hit token limits at any time, and SPEC.md is the single source of truth that enables seamless continuity. The lag between a decision being made and it being recorded in SPEC.md must be zero.
- **Always maintain this CLAUDE.md** — update it when new preferences, design decisions, or architecture changes are established
- Before ending any session, ensure both files reflect all decisions made during the session
- If multiple decisions are made in rapid succession (e.g., user approves a batch), pause implementation and write ALL of them to SPEC.md before proceeding with any code

### Parallel Execution
- **Deploy as many agents as possible in parallel** for non-dependent tasks to maximize efficiency
- Run searches, builds, file edits, and validations concurrently whenever they don't depend on each other

### Git & Commits
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

### Optimizer Run Discipline (Critical — Token Budget Protection)
- **Optimizer runs are expensive** — they cost compute time AND user tokens. A stale run that gets thrown away wastes both. Treat every optimizer run as a high-value operation that must succeed.
- **NEVER start an optimizer run while decisions are still being discussed.** The optimizer must reflect ALL decisions made up to the point of launch.
- **Pre-run gate**: Before launching `optimize_overprocure.py`, explicitly verify:
  1. All decisions from the current conversation have been implemented in the optimizer code
  2. All decisions have been captured in SPEC.md (per Documentation-First rule above)
  3. No open questions remain that could change optimizer logic, cost tables, or methodology
  4. The code passes a syntax check (`python -c "import py_compile; py_compile.compile(...)"`)
  5. **Full QA/QC and debug sweep** — verify ALL key assumptions (hydro caps, cost tables, resource constraints, dispatch logic, procurement bounds, storage parameters) match SPEC.md and real-world data. Run a dry-run test: imports, constants, data loading, checkpoint save/load round-trip. Confirm no hardcoded values contradict prior decisions. Present the user with a summary of verified assumptions before starting. **This gate exists because**: a previous run wasted 3+ hours of compute due to incorrect hydro caps that weren't caught before launch.
  6. **Checkpoint system verified** — confirm checkpoint save/load/resume works correctly and interval is set appropriately (currently 5 scenarios, ~27s max loss)
- **Once running, the optimizer is the top priority.** Do NOT let it get interrupted, stopped, or deprioritized. It runs in the background — other non-optimizer work can happen concurrently, but nothing should kill the process. If the session is approaching token limits, warn the user that the optimizer is still running and needs to complete.
- **If new decisions are made while the optimizer is running in the background**: Immediately flag to the user that the running optimizer does NOT reflect the new decision, and confirm whether to (a) let it finish anyway (if the decision doesn't affect current run), or (b) stop it and re-run after implementing the change. Never silently let a stale run continue as if it's current.
- **If the user asks to run the optimizer**: Treat it as a trigger to do a final audit — scan the recent conversation for any unimplemented decisions before starting the run. If anything is missing, implement it first, THEN run.
- **Background optimizer + other edits is fine** — but only for edits that don't touch optimizer logic (e.g., HTML, CSS, documentation, dashboard JS). If an edit changes anything the optimizer consumes (cost tables, algorithms, thresholds, resource types, dispatch logic), the optimizer must be re-run after the current run completes.
- **If the optimizer crashes or exits without writing results**: Automatically troubleshoot, debug, and retry. Don't wait idle — check logs/stderr, identify the failure mode (OOM, timeout, runtime error, etc.), apply a fix or workaround, and re-launch. Only escalate to the user if the root cause is ambiguous or requires a design decision.

### Change Propagation (Critical)
- **"Fix something" = fix it everywhere** — any request to fix, update, or change something applies to ALL regions and ALL pages by default, not just the one being discussed
- Pages to update: `dashboard.html`, `dashboard_standalone.html`, all `region_*.html` pages, `optimizer_methodology.html`, `research_paper.html`
- **Always update the research paper** (`research_paper.html`) when optimizer results, methodology, or findings change
- **Proactively update narrative text** after new results are generated — don't wait to be asked
- Only scope a fix to a single page if the user explicitly says so (e.g., "just on CAISO")

### Session Start Checklist
1. **Read SPEC.md first** — it contains every design decision, cost table, and implementation detail
2. **Read this file (CLAUDE.md)** — it contains all user preferences and project context
3. **Check the todo list** or review git log to see what's been completed
4. **Confirm which branch you're on** — develop on the designated branch for your task

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

- **Optimizer**: `optimize_overprocure.py` — Python, numpy-accelerated 3-phase sweep
- **Dashboard**: `dashboard/dashboard.html` — pure HTML5 + Chart.js, no framework
- **Regional Pages**: `dashboard/region_caiso.html`, `region_ercot.html`, etc. — 5 deep-dive pages
- **Standalone**: `dashboard/dashboard_standalone.html` — self-contained with inlined assets
- **Methodology**: `dashboard/optimizer_methodology.html` — technical specs only (trimmed)
- **Research Paper**: `dashboard/research_paper.html` — full paper with regional deep-dives
- **Data**: `data/` — EIA hourly profiles, eGRID emission rates, fossil mix data
- **Results**: `dashboard/overprocure_results.json` — pre-computed optimization output

## Key Design Principles

- 2025 snapshot model (no forward projections)
- All sensitivity toggles use Low/Medium/High naming (never "Base" or "Baseline")
- All new features layered on top of existing — never remove existing visuals or controls
- **COST DRIVES RESOURCE MIX** — cost and resource mix are co-optimized for every scenario. Different cost assumptions produce different optimal resource mixes. This is the core scientific contribution of the project. Never decouple cost from mix optimization or treat cost as a secondary overlay.
- **5 paired toggle groups** replace 10 individual toggles (Renewable Gen, Firm Gen, Storage, Fossil Fuel, Transmission)
- **10 thresholds** (75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100) — reduced from 18, with 2.5% granularity in inflection zone
- **16,200 total scenarios** (10 thresholds × 5 regions × 324 paired toggle combos) — each with its own co-optimized resource mix and cost
- Resource mix optimization at Medium costs; sensitivity toggles recalculate costs on cached mixes
- Hydro is always existing-only, wholesale-priced, $0 transmission
- H2 storage explicitly excluded
- CCS-CCGT includes 45Q offset in LCOE
- LDES = 100hr iron-air, 50% RT efficiency, new multi-day dispatch algorithm
- Battery = 4hr Li-ion, 85% RT efficiency, existing daily-cycle dispatch preserved

## Critical: Scientific Rigor vs. Compute

**NEVER sacrifice scientific integrity to save compute.** Use as much compute as necessary to achieve academically rigorous results. The user expects this project to withstand academic scrutiny.

When facing compute vs. rigor tradeoffs:
1. **Always discuss the tradeoff with the user first** — don't unilaterally choose minimal compute
2. **Find the best middle ground** that balances rigor with feasibility
3. **Pairing variables** (e.g., 5 paired toggles vs. 10 individual) is an acceptable rigor-compute tradeoff because it reflects real-world cost correlations
4. **Reducing thresholds** from 18 to 7 is acceptable because it preserves key inflection points
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
- Top navigation bar on ALL pages: Home | Dashboard | Regional Deep Dives | CO2 Abatement | Methodology | Paper
- Current page highlighted in nav; mobile gets hamburger/collapsible nav
- Scrollytelling format for regional deep-dive pages, matching main dashboard style
- **Homepage (index.html)** is the landing page with scrollytell narrative and key conclusions; dashboard.html is the interactive optimizer
- **Static pages default to Medium cost sensitivities** — homepage and regional deep dives use all-Medium toggle data unless a figure is explicitly designed to show L/M/H comparison ranges
- Layer in explanations for model elements — assume reader has minimal energy domain knowledge
- Clean, crisp visual identity — no clutter
- Both mobile and desktop compatible (44px min tap targets, responsive charts, no horizontal overflow)

### Content & Audience
- Dashboard audience: Business professionals, minimal energy sector knowledge
- Tooltips/info icons on controls explaining what each toggle does and why it matters
- Chart titles should tell the story, not just label axes
- Regional pages: Build understanding progressively, lead with "so what" before "how"
- Research paper: Academic rigor, withstand scrutiny, but still accessible to new readers
- Methodology HTML: Technical specs only (detailed narrative lives in PDF paper)

### QA/QC Requirements (Before Any Push)
- Validate optimizer results against published research (NREL ATB, Lazard, LBNL)
- Check HTML formatting, visual consistency, all controls functional
- Mobile compatibility at 320px, 375px, 768px viewports
- All text readable in all figures at all sizes
- No console errors, no broken layouts
- **Always do a full QA/QC sweep** on functionality, visuals, and narrative before pushing
- Validate optimizer results against published research (NREL ATB, Lazard, LBNL)
- Check HTML formatting, visual consistency, all controls functional across ALL pages
- Mobile compatibility at 320px, 375px, 768px viewports
- All text readable in all figures at all sizes
- No console errors, no broken layouts
- **Proactively update narrative and explanatory text** after new results are generated — don't leave stale numbers or descriptions
- Verify research paper reflects current results and methodology

### Animations & Interactivity
- Regional deep-dive pages should be illustrative and dynamic with animations
- Abatement comparison page: creative animations, animated number counters, scroll-triggered transitions
- Use CSS animations, scroll-based triggers, Chart.js animation options
- Keep it professional (Bloomberg/McKinsey quality) but engaging

### CO2 & Abatement Modeling (Decided)
- **CO2 emission rate**: Dynamic — shifts with fossil fuel price toggle using regional fuel-switching elasticity
- **Abatement benchmarks**: Static L/M/H bands (DAC, SAF, BECCS, etc.) as fixed horizontal bands on charts
- **Social cost of carbon references**: EPA $51/ton + Rennert et al. $185/ton + EU ETS $60-100/ton range — all three shown on charts

### Build Process
- Deploy agents in parallel for non-dependent tasks
- Push early so user can iterate/review while optimizer builds
- Standalone HTML must be rebuilt after all changes
- **After every optimizer run**: Always save a final cached results data file (`data/optimizer_cache.json`) that can be read into future projects as input. Include full co-optimized results for all thresholds × scenarios × ISOs with resource mixes, costs, scores, and metadata.
### Build Process
- Deploy agents in parallel for non-dependent tasks (see Workflow Preferences above)
- Standalone HTML must be rebuilt after all changes (`python dashboard/build_standalone.py`)
- Push only after feature is complete and QA'd (see Git & Commits above)

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
