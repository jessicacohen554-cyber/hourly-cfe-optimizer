# Codebase Refactoring Plan

> **STATUS: PLAN ONLY — Do not implement until user approves.**
> Created: 2026-02-15

## Goals

1. **Eliminate duplication** — ~3,500-4,000 lines of repeated CSS, JS, data, and HTML across 15+ files
2. **Centralize data** — single source of truth for MAC data, benchmarks, colors, thresholds (no more hardcoding)
3. **Nested navigation** — group related pages under dropdown menus; integrate generator analysis project
4. **QA/QC checkpoints** — narrow debug windows with validation at every refactor step
5. **Optimizer cleanup** — streamline 3,630-line optimizer for maintainability
6. **Reference architecture** — make future updates cheap (change one file, propagate everywhere)
7. **Systematic approach** — no more patching with redundant lines that conflict with old code

## Current State (Problems)

### Duplication Inventory

| Category | Lines Duplicated | Files | Severity |
|----------|-----------------|-------|----------|
| Navigation bar (HTML + CSS) | ~1,035 | 9 main + 6 gen-analysis | CRITICAL |
| CSS root variables / colors | ~200-240 | 8+ | HIGH |
| CSS reset / typography / layout | ~2,000-2,500 | 10 | HIGH |
| MAC_DATA arrays | 60 | 2 | MEDIUM |
| BENCHMARKS (static/dynamic/extra) | 95 | 2 | MEDIUM |
| REGION_COLORS definitions | 50+ | 6 | MEDIUM |
| Chart.js boilerplate patterns | ~200-400 | 6 | MEDIUM |
| Utility functions (findCrossover, etc.) | 40 | 2 | LOW |
| **TOTAL** | **~3,500-4,000** | — | — |

### Architecture Problems

- **Zero shared files** — 100% of CSS/JS is inline per HTML file
- **Hardcoded data arrays** — MAC_DATA, benchmarks, colors defined separately in each file that needs them
- **No module system** — every file reinvents fetch patterns, chart config, state management
- **Generator analysis is a separate project** — has its own CSS file but isn't integrated into main nav
- **Flat navigation** — 8+ nav links on one level, no grouping of related pages
- **No automated QA** — no linting, no tests, no pre-commit hooks, no CI
- **Optimizer monolith** — 3,630 lines in one file, no separation of concerns

---

## Refactoring Plan — 7 Phases

### Phase 1: Shared CSS Foundation (saves ~2,500 lines)

**What:** Extract all common CSS into `dashboard/css/shared.css`

**Contents of shared.css:**
- CSS reset & box-sizing (currently duplicated in every file)
- `:root` variables — all colors, fonts, spacing, shadows, radii
- Navigation bar styles (`.top-nav`, `.top-nav-inner`, `.nav-hamburger`, responsive rules)
- Typography rules (body, h1-h4, p, lists)
- Common layout classes (`.section-narrative`, `.chart-container`, `.callout-box`, `.counter-row`)
- Footer styles
- Responsive breakpoints (320px, 375px, 768px, 1024px, 1440px)
- Chart container defaults (`.chart-card`, min-heights, aspect ratios)
- Button/toggle control styles (`.ctrl-toggle`, `.scenario-btn`, `.info-icon`)

**What stays inline:** Page-specific overrides only (unique layouts, one-off animations)

**QA Checkpoint 1A:**
- [ ] Open each page in browser — visual diff against screenshots taken before refactoring
- [ ] Check mobile viewport (375px) — no broken layouts
- [ ] Verify all CSS variable references resolve (no undefined vars)
- [ ] Grep for any remaining inline `:root` blocks — should be zero

**QA Checkpoint 1B:**
- [ ] Run `grep -c 'top-nav' dashboard/*.html` — nav CSS should only appear in shared.css
- [ ] Verify nav hamburger works on mobile in every page

---

### Phase 2: Shared Navigation Component (saves ~1,035 lines)

**What:** Create `dashboard/components/nav.html` template and `dashboard/js/nav.js` loader

**Approach:** Since these are static HTML files (no server-side includes), use a JS-based component:

```
dashboard/js/nav.js — injects nav HTML into <div id="site-nav"></div>
```

**nav.js functionality:**
- Injects full nav bar HTML with nested dropdowns
- Auto-highlights current page based on `window.location`
- Handles hamburger toggle for mobile
- Handles dropdown open/close

**New Navigation Structure (nested menus):**

```
Home | Cost Optimizer | Analysis ▾ | Research ▾ | Generator Analysis ▾

Analysis ▾
├── CO₂ Abatement Summary
├── Abatement Dashboard
├── EAC Scarcity
└── Regional Deep Dives

Research ▾
├── Research Paper
├── Methodology
├── Policy Context
└── About

Generator Analysis ▾
├── Overview
├── Fleet Analysis
├── Dashboard
├── Targets & Standards
├── Policy Scenarios
└── Methodology
```

**Generator analysis integration:**
- Move `power-gen-decarbonization/site/*.html` into `dashboard/gen-analysis/`
- Update all internal links to use relative paths within the unified site
- Generator pages use the same shared.css + nav.js (adapting their existing CSS file as an override)
- Add gen-analysis CSS variables to shared.css

**Each HTML file changes from:**
```html
<nav class="top-nav">
  <span class="nav-brand">The 8,760 Problem</span>
  <button class="nav-hamburger" ...>☰</button>
  <div class="top-nav-inner" id="navLinks">
    <a href="index.html">Home</a>
    ... (9 links)
  </div>
</nav>
```

**To:**
```html
<div id="site-nav"></div>
<script src="js/nav.js"></script>
```

**QA Checkpoint 2A:**
- [ ] Every page renders correct nav with current page highlighted
- [ ] Dropdown menus open/close on hover (desktop) and tap (mobile)
- [ ] All links work — test every nav link from every page
- [ ] Mobile hamburger collapses all dropdowns properly
- [ ] Generator analysis pages accessible from main nav
- [ ] "Back to Home" breadcrumb still works on all sub-pages

**QA Checkpoint 2B:**
- [ ] Grep for `<nav class="top-nav"` — should appear in ZERO HTML files (only in nav.js)
- [ ] Verify no duplicate nav HTML remaining in any file

---

### Phase 3: Shared Data Module (saves ~250 lines, eliminates hardcoding)

**What:** Create `dashboard/js/shared-data.js` — single source of truth for all data constants

**Contents:**
```javascript
// dashboard/js/shared-data.js
const SITE_DATA = {
    thresholds: [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99],

    mac_data: {
        medium: { CAISO: [...], ERCOT: [...], ... },
        low: { ... },
        high: { ... }
    },

    region_colors: {
        CAISO: '#F59E0B', ERCOT: '#22C55E', PJM: '#4a90d9',
        NYISO: '#E91E63', NEISO: '#9C27B0'
    },

    resource_colors: {
        clean_firm: '#1E3A5F', ccs_ccgt: '#0D9488', solar: '#F59E0B',
        wind: '#22C55E', hydro: '#0EA5E9', battery: '#8B5CF6', ldes: '#EC4899'
    },

    resource_labels: {
        clean_firm: 'Clean Firm', ccs_ccgt: 'CCS-CCGT', solar: 'Solar',
        wind: 'Wind', hydro: 'Hydro', battery: 'Battery (4h)', ldes: 'LDES (100h)'
    },

    benchmarks_static: [ ... ],
    benchmarks_dynamic: { dac: { ... }, industrial: { ... }, removal: { ... } },
    benchmarks_extra: [ ... ],

    cost_ranges: { ... }  // From literature_review.json
};

// Utility functions
function findCrossover(regionData, costLevel, thresholds) { ... }
function cellClass(val) { ... }
function getAllBenchmarks(state) { ... }
```

**Rule: No data array is ever defined in an HTML file.** If a page needs data, it imports from shared-data.js or fetches from overprocure_results.json. Period.

**Updating data in the future:**
1. Change the value in `shared-data.js`
2. Every page that uses it automatically picks up the change
3. No grep-and-replace across 6 files

**QA Checkpoint 3:**
- [ ] Grep for `const MAC_DATA` — should appear ONLY in shared-data.js
- [ ] Grep for `const BENCHMARKS_STATIC` — should appear ONLY in shared-data.js
- [ ] Grep for `REGION_COLORS` definition — should appear ONLY in shared-data.js
- [ ] All charts still render correctly in abatement_comparison.html and abatement_dashboard.html
- [ ] Inflection table still populates correctly
- [ ] Toggle controls still update charts

---

### Phase 4: Chart.js Shared Configuration (saves ~200-400 lines)

**What:** Create `dashboard/js/chart-helpers.js` — shared chart config patterns

**Contents:**
```javascript
// Default Chart.js options all charts inherit
const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        datalabels: { display: false },
        legend: { labels: { usePointStyle: true, pointStyle: 'circle', padding: 14 } }
    }
};

// Axis presets
const AXIS_CLEAN = { grid: { display: false }, border: { display: true, color: '#D4D8E0' } };

// Builder functions
function buildLineChart(canvasId, datasets, xConfig, yConfig, annotations, extraOpts) { ... }
function buildBarChart(canvasId, labels, barData, colors, extraOpts) { ... }
function buildAnnotationLine(yValue, label, color) { ... }
function buildAnnotationBand(yMin, yMax, label, color) { ... }

// Tooltip formatters
const TOOLTIP_DOLLAR = { callbacks: { label: ctx => '$' + ctx.raw.toLocaleString() } };
const TOOLTIP_PERCENT = { callbacks: { label: ctx => ctx.raw + '%' } };
```

**Rule: No chart config boilerplate in HTML files.** Each chart call becomes:
```javascript
buildLineChart('macCurveChart', datasets,
    { type: 'linear', min: 74, max: 101, label: 'Hourly Clean Energy Target (%)' },
    { type: 'linear', min: 0, max: 250, label: 'Average Abatement Cost ($/ton CO₂)' },
    annotations);
```

Instead of 50+ lines of Chart.js config per chart.

**QA Checkpoint 4:**
- [ ] Every chart renders identically before/after (visual comparison)
- [ ] Tooltips work on every chart
- [ ] Legend filtering (hiding band datasets) still works
- [ ] Mobile chart sizing preserved (min-heights, aspect ratios)
- [ ] Annotation lines/bands render at correct positions

---

### Phase 5: Optimizer Code Cleanup (no line-count target — maintainability focus)

**What:** Refactor `optimize_overprocure.py` (3,630 lines) into logical modules

**Current structure (monolith):**
```
optimize_overprocure.py
├── Constants & config (~300 lines)
├── Data loading & validation (~400 lines)
├── Profile processing (~500 lines)
├── Dispatch algorithms (~800 lines)
├── Cost calculation (~600 lines)
├── CO2 calculation (~300 lines)
├── Checkpoint system (~200 lines)
├── Main optimization loop (~500 lines)
└── Results output (~200 lines)
```

**Proposed structure:**
```
optimizer/
├── __init__.py
├── config.py          — Constants, cost tables, resource definitions
├── data_loader.py     — EIA data loading, profile averaging, validation
├── dispatch.py        — Battery, LDES, hydro, CCS dispatch algorithms
├── cost_engine.py     — LCOE calculations, transmission, wholesale adjustments
├── co2_engine.py      — Hourly emission rates, abatement calculations
├── checkpoint.py      — Save/load/resume logic
├── optimizer.py       — Main optimization loop (calls dispatch + cost + co2)
├── results.py         — Output formatting, JSON export
└── run.py             — Entry point (replaces optimize_overprocure.py)
```

**Key principle:** `config.py` becomes THE single source of truth for all optimizer constants. Cost tables, hydro caps, resource definitions, dispatch parameters — all in one file. When a decision changes, you edit `config.py` and the optimizer picks it up.

**Backward compatibility:** Keep `optimize_overprocure.py` as a thin wrapper that imports from the new modules. No external scripts need to change.

**QA Checkpoint 5A (after each module extraction):**
- [ ] `python -c "from optimizer import config"` — imports work
- [ ] Run optimizer on a single ISO/threshold — output matches pre-refactor output byte-for-byte
- [ ] Checkpoint save/load round-trip works
- [ ] Full regression: run all 5 ISOs at threshold 75 — compare results to saved reference

**QA Checkpoint 5B (full regression):**
- [ ] Run full optimizer suite — compare `overprocure_results.json` against pre-refactor version
- [ ] Verify all 5 ISOs × 9 thresholds × 324 scenarios match
- [ ] Cost values match to 4 decimal places
- [ ] CO2 values match to 4 decimal places

---

### Phase 6: Generator Analysis Integration

**What:** Move generator analysis into the unified site structure

**File moves:**
```
power-gen-decarbonization/site/index.html       → dashboard/gen-analysis/index.html
power-gen-decarbonization/site/fleet-analysis.html → dashboard/gen-analysis/fleet-analysis.html
power-gen-decarbonization/site/dashboard.html    → dashboard/gen-analysis/dashboard.html
power-gen-decarbonization/site/targets.html      → dashboard/gen-analysis/targets.html
power-gen-decarbonization/site/policy.html       → dashboard/gen-analysis/policy.html
power-gen-decarbonization/site/methodology.html  → dashboard/gen-analysis/methodology.html
power-gen-decarbonization/site/css/style.css     → dashboard/css/gen-analysis.css
```

**Integration steps:**
1. Update all internal links in gen-analysis pages to use `../` paths for main site navigation
2. Replace gen-analysis nav bars with the shared `nav.js` component (Phase 2)
3. Add `<link rel="stylesheet" href="../css/shared.css">` + `<link rel="stylesheet" href="../css/gen-analysis.css">` to each page
4. Remove duplicated CSS from gen-analysis pages that's now in shared.css
5. Update all cross-references between main site and gen-analysis (currently using `../power-gen-decarbonization/site/`)

**QA Checkpoint 6:**
- [ ] All gen-analysis pages render correctly with shared nav
- [ ] Main nav dropdown links to gen-analysis pages work from every page
- [ ] Gen-analysis pages link back to main site correctly
- [ ] Gen-analysis charts still work (their JS is mostly inline and independent)
- [ ] CSS override layer (gen-analysis.css) applies correctly on top of shared.css
- [ ] Old `power-gen-decarbonization/site/` links redirect or are updated everywhere

---

### Phase 7: QA Infrastructure & Reference System

**What:** Build the automated validation layer and reference architecture

#### 7A: Automated QA Script (`qa_check.sh`)

```bash
#!/bin/bash
# Run after every refactor step

echo "=== HTML Validation ==="
# Check all canvas elements have Chart initialization
for f in dashboard/*.html dashboard/gen-analysis/*.html; do
    canvases=$(grep -c '<canvas' "$f" 2>/dev/null || echo 0)
    charts=$(grep -c 'new Chart' "$f" 2>/dev/null || echo 0)
    if [ "$canvases" -gt "$charts" ]; then
        echo "WARN: $f has $canvases canvases but only $charts Chart() calls"
    fi
done

echo "=== Duplication Check ==="
# Ensure shared data is centralized
grep -rl 'const MAC_DATA' dashboard/ | grep -v 'shared-data.js'
grep -rl 'const BENCHMARKS_STATIC' dashboard/ | grep -v 'shared-data.js'
grep -rl 'const REGION_COLORS' dashboard/ | grep -v 'shared-data.js'

echo "=== Link Validation ==="
# Check for broken internal links
for f in dashboard/*.html; do
    grep -oP 'href="[^"#]+"' "$f" | while read link; do
        target=$(echo "$link" | sed 's/href="//;s/"//')
        if [[ ! -f "dashboard/$target" && ! -f "$target" ]]; then
            echo "BROKEN: $f → $target"
        fi
    done
done

echo "=== Nav Consistency ==="
# Verify all pages use shared nav
grep -rL 'id="site-nav"' dashboard/*.html | grep -v 'nav.js'
```

#### 7B: Python Dependencies

Create `requirements.txt`:
```
numpy>=1.24
```

#### 7C: Reference Data Manifest (`data/manifest.json`)

```json
{
    "optimizer_results": {
        "file": "dashboard/overprocure_results.json",
        "generated": "2026-02-15",
        "isos": ["CAISO", "ERCOT", "PJM", "NYISO", "NEISO"],
        "thresholds": [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99],
        "scenarios_per_threshold": 324,
        "checksum": "<sha256>"
    },
    "literature_review": {
        "file": "data/literature_review.json",
        "version": "1.0",
        "references": 15,
        "purpose": "Prevent redundant web sweeps"
    },
    "shared_data": {
        "file": "dashboard/js/shared-data.js",
        "contains": ["MAC_DATA", "BENCHMARKS", "REGION_COLORS", "RESOURCE_COLORS", "THRESHOLDS"],
        "source": "Computed from optimizer_results via MAC formula"
    }
}
```

**Purpose:** Any future session reads `manifest.json` to understand what data exists, where it came from, and whether it's current. Eliminates the "is this data stale?" problem.

#### 7D: CLAUDE.md Update — Reference Architecture Section

Add to CLAUDE.md:
```markdown
## Reference Architecture (Post-Refactor)

### Shared Files (change here, propagates everywhere)
- `dashboard/css/shared.css` — all common styles, variables, nav CSS
- `dashboard/js/nav.js` — navigation component (nested dropdowns)
- `dashboard/js/shared-data.js` — MAC_DATA, benchmarks, colors, thresholds, utilities
- `dashboard/js/chart-helpers.js` — Chart.js builder functions and defaults
- `data/manifest.json` — data inventory and freshness tracking

### Data Flow
- Optimizer → `overprocure_results.json` → computed MAC → `shared-data.js`
- `shared-data.js` → consumed by all dashboard/abatement pages
- `literature_review.json` → reference for future research sessions

### When Updating Data
1. If optimizer re-runs: update `shared-data.js` MAC_DATA arrays
2. If benchmarks change: update `shared-data.js` BENCHMARKS
3. If colors change: update `shared-data.js` + `shared.css` `:root` variables
4. Run `qa_check.sh` after every change

### When Adding a New Page
1. Copy minimal template (just content divs)
2. Add `<link rel="stylesheet" href="css/shared.css">`
3. Add `<div id="site-nav"></div><script src="js/nav.js"></script>`
4. Add page to nav.js page list
5. Page-specific CSS goes in `<style>` block (override only)
6. Page-specific data comes from `shared-data.js` or fetch

### Commit Discipline (Post-Refactor)
- **Never define data constants in HTML files** — always in shared-data.js
- **Never add CSS that's already in shared.css** — use existing classes or extend
- **Never hardcode nav HTML** — always use nav.js
- **Run qa_check.sh before every commit**
```

**QA Checkpoint 7:**
- [ ] `qa_check.sh` runs cleanly with zero warnings
- [ ] `manifest.json` is accurate and complete
- [ ] CLAUDE.md reference architecture section is current
- [ ] All links across entire site validate

---

## Implementation Order & Commit Strategy

Each phase is a single squash commit. QA checkpoints must pass before committing.

| Phase | Commit | Dependencies | Est. Files Changed |
|-------|--------|-------------|-------------------|
| 1: Shared CSS | `refactor: extract shared.css, remove inline duplicate CSS` | None | 10+ HTML files + 1 new CSS |
| 2: Shared Nav | `refactor: shared nav component with nested menus` | Phase 1 | 15+ HTML files + 2 new JS |
| 3: Shared Data | `refactor: centralize data constants in shared-data.js` | None (parallel with 1-2) | 6 HTML files + 1 new JS |
| 4: Chart Helpers | `refactor: shared chart config and builder functions` | Phase 3 | 6 HTML files + 1 new JS |
| 5: Optimizer | `refactor: split optimizer monolith into modules` | None (parallel with 1-4) | 1 Python file → 10 files |
| 6: Gen-Analysis | `refactor: integrate generator analysis into main site` | Phase 1 + 2 | 6 HTML files moved + updated |
| 7: QA Infra | `refactor: add qa_check.sh, manifest.json, requirements.txt` | Phase 1-6 | New files only |

**Total commits: 7 (one per phase)**

**Safe parallelism:** Phases 1+3+5 can run simultaneously (CSS, JS data, Python are independent). Phase 2 depends on Phase 1. Phase 4 depends on Phase 3. Phase 6 depends on Phases 1+2. Phase 7 is last.

---

## Risk Mitigation

### Narrow Debug Windows
- **Commit after every phase** — if something breaks, the search space is one phase
- **Visual regression screenshots** — take before/after screenshots of every page before and after each phase
- **Git tag before starting** — `git tag pre-refactor` so we can always diff against baseline
- **One file at a time within phases** — don't batch 10 files; do one, verify, continue

### Rollback Strategy
- If any QA checkpoint fails: `git diff` to see what changed, fix in the same commit
- If a phase is fundamentally broken: `git checkout -- .` and restart the phase
- Pre-refactor tag means we can always see the working baseline

### Preserving Functionality
- **Zero visual changes** — every page must render identically before and after
- **Zero functionality changes** — all toggles, charts, animations, links work exactly as before
- **Zero data changes** — all numbers, text, citations identical

---

## Token Budget Estimate

| Phase | Estimated Tokens | Why |
|-------|-----------------|-----|
| 1: Shared CSS | ~8,000-12,000 | Reading 10 files, extracting common CSS, creating shared.css, updating all files |
| 2: Shared Nav | ~6,000-10,000 | Creating nav.js with dropdowns, updating all 15 pages |
| 3: Shared Data | ~4,000-6,000 | Creating shared-data.js, updating 6 pages |
| 4: Chart Helpers | ~6,000-8,000 | Creating helpers, refactoring 6 chart files |
| 5: Optimizer | ~10,000-15,000 | Reading 3,630 lines, splitting into modules, regression testing |
| 6: Gen-Analysis | ~5,000-8,000 | Moving files, updating links, integrating nav |
| 7: QA Infra | ~2,000-3,000 | Creating scripts, manifest, updating CLAUDE.md |
| **TOTAL** | **~41,000-62,000** | Spread across 2-3 sessions |

**Future savings per session:** ~2,000-4,000 tokens (reduced file noise, no re-reading duplicate code, faster targeted edits). Pays back within 10-20 sessions of active editing.

---

## Post-Refactor File Tree

```
hourly-cfe-optimizer/
├── CLAUDE.md                    (updated with reference architecture)
├── SPEC.md                      (existing — design decisions)
├── REFACTOR_PLAN.md             (this file — archive after completion)
├── requirements.txt             (NEW — Python dependencies)
├── qa_check.sh                  (NEW — automated QA script)
├── data/
│   ├── manifest.json            (NEW — data inventory)
│   ├── literature_review.json   (existing)
│   ├── optimizer_cache.json     (existing)
│   └── ... (EIA, eGRID data)
├── optimizer/                   (NEW — refactored Python modules)
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── dispatch.py
│   ├── cost_engine.py
│   ├── co2_engine.py
│   ├── checkpoint.py
│   ├── optimizer.py
│   └── run.py
├── optimize_overprocure.py      (KEPT — thin wrapper for backward compat)
├── dashboard/
│   ├── css/
│   │   ├── shared.css           (NEW — all common styles)
│   │   └── gen-analysis.css     (MOVED from power-gen-decarbonization)
│   ├── js/
│   │   ├── nav.js               (NEW — shared navigation component)
│   │   ├── shared-data.js       (NEW — centralized data constants)
│   │   └── chart-helpers.js     (NEW — Chart.js builder utilities)
│   ├── gen-analysis/            (NEW — integrated generator analysis)
│   │   ├── index.html
│   │   ├── fleet-analysis.html
│   │   ├── dashboard.html
│   │   ├── targets.html
│   │   ├── policy.html
│   │   └── methodology.html
│   ├── index.html               (TRIMMED — uses shared CSS/JS/nav)
│   ├── dashboard.html           (TRIMMED)
│   ├── abatement_comparison.html (TRIMMED)
│   ├── abatement_dashboard.html  (TRIMMED)
│   ├── region_deepdive.html     (TRIMMED)
│   ├── research_paper.html      (TRIMMED)
│   ├── eac_scarcity.html        (TRIMMED)
│   ├── about.html               (TRIMMED)
│   ├── policy_context.html      (TRIMMED)
│   └── overprocure_results.json (existing)
└── power-gen-decarbonization/   (ARCHIVED — symlink or redirect to dashboard/gen-analysis/)
```
