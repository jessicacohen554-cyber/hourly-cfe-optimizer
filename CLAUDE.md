# Claude Code — Session Continuity Instructions

## If Resuming This Project

1. **Read SPEC.md first** — it contains every design decision, cost table, and implementation detail
2. **Check the todo list** or review git log to see what's been completed
3. **Branch**: `claude/enhance-optimizer-model-IqSpe`
4. **Repo**: `jessicacohen554-cyber/hourly-cfe-optimizer` (all advanced model work on designated branch)

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
- Top navigation bar on ALL pages: Dashboard | CAISO | ERCOT | PJM | NYISO | NEISO | Methodology | Paper
- Current page highlighted in nav; mobile gets hamburger/collapsible nav
- Scrollytelling format for regional deep-dive pages, matching main dashboard style
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
