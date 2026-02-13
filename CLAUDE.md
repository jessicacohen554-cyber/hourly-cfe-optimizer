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
- Resource mix optimization at Medium costs; sensitivity toggles recalculate costs on cached mixes
- Hydro is always existing-only, wholesale-priced, $0 transmission
- H2 storage explicitly excluded
- CCS-CCGT includes 45Q offset in LCOE
- LDES = 100hr iron-air, 50% RT efficiency, new multi-day dispatch algorithm
- Battery = 4hr Li-ion, 85% RT efficiency, existing daily-cycle dispatch preserved

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

### Build Process
- Deploy agents in parallel for non-dependent tasks
- Push early so user can iterate/review while optimizer builds
- Standalone HTML must be rebuilt after all changes
