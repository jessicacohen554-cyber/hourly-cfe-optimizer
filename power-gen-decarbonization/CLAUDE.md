# Claude Code — Session Instructions

## If Resuming This Project

1. **Read SPEC.md first** — it contains every design decision, data source, and analysis detail
2. **Check the todo list** or review git log to see what's been completed
3. **Repo**: `jessicacohen554-cyber/power-gen-decarbonization`

## Workflow Preferences (Apply to EVERY Session)

### Documentation-First Development
- **CRITICAL — Decisions go to SPEC.md IMMEDIATELY**: The *very first action* after any design, methodology, or analytical decision is confirmed by the user is to write it to SPEC.md. Do NOT continue with implementation, code changes, or further discussion until the decision is captured. This is the highest-priority workflow rule — sessions can be disrupted or hit token limits at any time, and SPEC.md is the single source of truth that enables seamless continuity.
- **Always maintain this CLAUDE.md** — update it when new preferences, design decisions, or architecture changes are established
- Before ending any session, ensure both files reflect all decisions made during the session
- If multiple decisions are made in rapid succession, pause implementation and write ALL of them to SPEC.md before proceeding with any code

### Parallel Execution
- **Deploy as many agents as possible in parallel** for non-dependent tasks to maximize efficiency
- Run searches, builds, file edits, and validations concurrently whenever they don't depend on each other

### Git & Commits
- **Squash-style commits** — one descriptive commit per feature/task, not granular per-file commits
- **Descriptive paragraph messages** — commit messages should explain *what* and *why*, not just list files changed
- **Detailed PR descriptions** — include summary, what changed, why, and any decisions made
- **Push only when a feature is complete** — don't push partial/broken work; finish the task, QA it, then push
- **Commit checkpoints** — save progress at meaningful milestones, not just at the end

### Decision-Making (Structured Approval)
- **Present decisions as structured option trees** — user selects the path forward before implementation
- Format: numbered items (1, 2, 3), lettered options (A, B, C, D), roman numeral sub-options (i, ii, iii) if needed
- Each option must include **pros and cons** so the user can make an informed choice
- User responds with shorthand like `1-A-i` to select their preferred path
- **Never contradict directions** already given in this file or SPEC.md
- Trivial/obvious decisions (formatting, variable names, minor refactors consistent with existing patterns) don't need approval — just do them

### File Boundaries
- **Never modify raw data files** in `data/raw/` — these are preserved from source (EPA eGRID, EIA, S&P, etc.)
- If data needs transformation, create a derived file in `data/processed/` — never edit the original
- Freely edit: analysis scripts, site HTML/JS/CSS, methodology docs, config files

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
  5. For data-dependent features: Note explicitly that they need data pipeline results — don't mark complete
- **If a page is a wireframe/skeleton**: Say so. "Structure created, awaiting implementation" is honest.
- **Run a verification audit before every commit** that touches SPEC.md status.
- **Stub pages get their own status**: `[ ] page.html — wireframe only (structure + CSS, no JS/content)`

### Token Efficiency (Critical — Protect User's Weekly Budget)
- **Targeted file reads only** — always use `offset`/`limit` on large files. Never re-read a file already read in the same session unless it's been modified since the last read.
- **Exploration agents: return summaries, not raw content** — never paste full file contents back.
- **Prefer Grep/Glob over Explore agents for directed lookups** — Explore agents are 10x more expensive.
- **Batch all related edits into one response** — don't make sequential edits with narration between each.
- **Don't repeat large code blocks back to the user** — reference by file:line instead.

### Change Propagation (Critical)
- **"Fix something" = fix it everywhere** — any request to fix, update, or change something applies to ALL pages by default
- Only scope a fix to a single page if the user explicitly says so

### Session Start Checklist
1. **Read SPEC.md first** — it contains every design decision and analysis detail
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
   - **Next steps** — detailed enough that a fresh session can pick up immediately
   - **Open questions** — anything unresolved that needs user input
   - **Checkbox TODO list** for user awareness
5. The TODO list is for user readability; the prose context around it is what enables the next session to continue seamlessly

## Project

- **Repo**: `jessicacohen554-cyber/power-gen-decarbonization`
- **Sister project**: `jessicacohen554-cyber/hourly-cfe-optimizer` — shares visual ecosystem, may be cross-linked

## Architecture Overview

- **Site**: `site/` — pure HTML5 + Chart.js scrollytelling, matching hourly-cfe-optimizer visual identity
- **Analysis**: `analysis/` — Python scripts for data processing, scenario modeling
- **Data (raw)**: `data/raw/` — eGRID, EIA, S&P, CDP source data (never modify)
- **Data (processed)**: `data/processed/` — derived datasets, company profiles, emissions mappings
- **Company profiles**: `data/processed/company_profiles.json` — structured data for all 15 generators

## Visual Identity (Shared with hourly-cfe-optimizer)

- Same typography, color palette, nav structure, scrollytelling approach
- Sites should feel part of the same research ecosystem
- New site may be nested/linked from the CFE optimizer site

## Content & Audience
- Audience: Business professionals, investors, policymakers — minimal energy domain knowledge assumed
- Layer in explanations for technical concepts
- Chart titles should tell the story, not just label axes
- Lead with "so what" before "how"
- Academic rigor: analysis should withstand scrutiny but remain accessible
- Bloomberg/McKinsey quality presentation — professional but engaging

## Key Design Principles

- Top 15 US power generators by TWh (post-Constellation/Calpine merger)
- eGRID 2023 + S&P data for plant-level ownership and emissions
- Fleet characteristics drive decarbonization pathway feasibility
- Insights from hourly CFE optimizer inform scenario analysis
- Assess against: IPCC, IEA NZE, SBTi Power Sector v2, US govt targets
- Identify enabling conditions (tax credits, carbon price) and barriers

## Data Sources
- **EPA eGRID 2023** — plant-level emissions, generation, ownership
- **EIA-923 / EIA-860** — generation data, plant characteristics
- **S&P Global** — parent company mapping, corporate structure
- **MJ Bradley / ERM** — Benchmarking Air Emissions report
- **CDP** — corporate climate disclosures
- **Company sustainability reports** — targets, strategies, progress
- **IPCC AR6, IEA NZE, SBTi v2** — climate targets and pathways

## QA/QC Requirements (Before Any Push)
- Validate emissions data against eGRID published totals
- Check HTML formatting, visual consistency, all controls functional
- Mobile compatibility at 320px, 375px, 768px viewports
- All text readable in all figures at all sizes
- No console errors, no broken layouts
- Cross-reference company data against their public filings

## Animations & Interactivity
- Scrollytelling format with scroll-triggered transitions
- Animated number counters, dynamic charts
- CSS animations, scroll-based triggers, Chart.js animation options
- Professional but engaging — Bloomberg/McKinsey quality

## Research & Exploration
- **Start with a quick survey** — broad scan first, then dive deeper only where the user asks
- Don't over-research upfront; present a summary and let the user direct where to go deep

## Working Style
- Use TodoWrite tool actively to track all tasks and give visibility into progress
- Break complex tasks into small, trackable steps
- Ask clarifying questions early rather than making wrong assumptions
- When making changes, read the relevant code first — never propose blind edits
- Commit checkpoints at meaningful milestones
