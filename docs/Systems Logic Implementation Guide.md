# Systems Logic Implementation Guide: Noesis to Dianoia

This guide facilitates the transition from **Noesis** (holistic synthesis) to **Dianoia** (linear proof). The homepage becomes a cinematic scrollytell spine that walks users through a 6-act narrative funnel — *here's the problem → here's why naive solutions fail → here's what actually works → here's how to get there* — with each act backed by real optimizer data and linking out to the deep-dive page where the full analysis lives.

---

## 1. Global Visual DNA (System Constraints)

**Typography:** Plus Jakarta Sans (Headings), DM Sans (Body) — inherited from `styles/shared.css`.

**Linework:** 2px solid outlines using CSS variables (e.g., `--solar`, `--fossil-gas`, `--battery`).

**Fills:** 12% opacity (`rgba(..., 0.12)`) matching the saturated outline color.

**Animation:** `stroke-dashoffset` for flow vectors, CSS `pulse` keyframes for FOAK signals, `scroll-observer.js` for scroll-triggered transitions.

**Stack:** Vanilla JS + Chart.js + scroll-observer.js (no React). All data from existing `dashboard/js/*.js` modules.

---

## 2. Homepage Architecture

```
┌─────────────────────────────────────────┐
│  Hero: "What if 100% clean isn't?"      │  ← provocative hook
│  Animated energy waveform (shared-header)│
├─────────────────────────────────────────┤
│  Act 1: The Grid is Not a Bank          │  ← ISO carbon heatstrips
├─────────────────────────────────────────┤
│  Act 2: The 100% Clean Illusion         │  ← annual vs hourly gap chart
├─────────────────────────────────────────┤
│  Act 3: The VRE-Gas Trap                │  ← LMP collapse animation
├─────────────────────────────────────────┤
│  Act 4: The Stranded Asset Cliff        │  ← MAC fan chart with cliff
├─────────────────────────────────────────┤
│  Act 5: The Integrated Portfolio        │  ← VRE-only vs optimized comparison
├─────────────────────────────────────────┤
│  Act 6: The Proof                       │  ← flush-fit dispatch stack
├─────────────────────────────────────────┤
│  CTA: Explore the Data Yourself         │  ← links to dashboard, paper
│  "7 ISOs · 5,832 scenarios · Your grid" │
└─────────────────────────────────────────┘
```

Each act is a `.story-section` with scroll-triggered fade-in, a single cinematic chart, narrative text, and an "Explore deeper →" link to the relevant deep-dive page.

---

## 3. Act-by-Act Implementation Prompts

### Act 1: "The Grid is Not a Bank"

**Narrative:** "The grid is a live, interconnected machine where carbon intensity varies wildly by the hour and location. Most clean energy policies ignore this physics, treating all MWh as equal."

**Data source:** `compressed_day_profiles.json` (24-hour dispatch profiles per ISO), regional emission rates from `shared-data.js`.

**Visual:** 24-hour carbon intensity heatstrips for 3 contrasting ISOs (CAISO, PJM, ERCOT) showing real hourly variation. CAISO's midday solar glut vs. PJM's coal-heavy baseload vs. ERCOT's wind-at-night.

**Links to:** Dashboard (regional selector), LMP Trends page.

**Session prompt:**
```
Build a scrollytell section for dashboard/index.html — Act 1: "The Grid is Not a Bank."

Context: This is a `.story-section` on the homepage. Use vanilla JS + Chart.js (no React).
Inherit all styles from `styles/shared.css` and colors from `js/chart-colors.js` (ISO_COLORS, RESOURCE_COLORS).
Use `scroll-observer.js` for scroll-triggered animation.

Data: Load 24-hour carbon intensity profiles from `compressed_day_profiles.json` for CAISO, PJM, and ERCOT.
If that file doesn't have emission intensity by hour, derive it from the dispatch profiles
(fossil share × regional emission rate from shared-data.js).

Visual: Three horizontal heatstrip bars (one per ISO), each 24 cells wide (hours 0-23).
Color scale: green (low carbon) → amber → red (high carbon). 2px solid outlines, 12% alpha fills.
Animate in on scroll — strips appear sequentially with a 200ms stagger.

Interaction: Toggle between "Illusion" (flat average — uniform color per strip) and "Reality"
(actual hourly variation). Default to "Illusion", auto-transition to "Reality" on scroll
with a CSS scanline sweep animation.

Narrative panel (left on desktop, above on mobile): Include the "Grid is Not a Bank" narrative text.
End with "Explore regional patterns →" link to lmp_trends.html.

Aesthetic: 2px saturated outlines, 12% fills. Use --font-data for tooltip values.
Mobile: Stack strips vertically, 44px min tap targets. Responsive via shared.css grid classes.
```

---

### Act 2: "The 100% Clean Illusion"

**Narrative:** "Annual accounting matches abundant daytime solar with nighttime demand. The math works on paper, but physically, the company relies entirely on the dirty grid at night."

**Data source:** Dispatch profiles from `scenario_a_*.json` or `compressed_day_profiles.json` — solar generation curve vs. flat load line for a representative ISO (CAISO is most dramatic). Procurement strategy data from `procurement-strategy-data.js` (annual vs. hourly match scores).

**Visual:** Animated area chart — solar generation stacks up during the day then collapses at night, revealing the fossil gap shaded in red. A counter ticks: "Annual match: 100%. Actual clean hours: 47%."

**Links to:** Procurement Strategies page, Abatement Dashboard.

**Session prompt:**
```
Build a scrollytell section for dashboard/index.html — Act 2: "The 100% Clean Illusion."

Context: `.story-section` on homepage. Vanilla JS + Chart.js. Styles from shared.css, colors from chart-colors.js.

Data: Load a 24-hour generation profile for CAISO from compressed_day_profiles.json.
Need: solar generation curve, total VRE curve, and flat demand line.
If exact data isn't in that shape, derive from the dispatch cache:
solar supply by hour vs. demand by hour. Calculate annual match % (total gen / total demand)
and hourly match % (hours where clean gen >= demand / 8760).

Visual: Chart.js area chart. X-axis: hours 0-23. Layers:
- Flat demand line (2px dashed, --font-body color)
- Solar generation area (filled --solar at 12% alpha, 2px solid --solar outline)
- Wind generation area stacked (filled --wind at 12% alpha)
- Gap between total clean and demand: filled --fossil-gas at 12% alpha during night hours,
  --solar at 12% alpha where surplus exists during day

Scroll animation: On scroll entry, first show a flat "Annual Average" line that matches demand
(the "illusion" — everything looks balanced). Then animate to the real hourly shape,
revealing the daytime surplus and nighttime deficit. Use Chart.js update() with transition.

Counter overlay: Two animated stat-values in `.headline-card` style:
  "Annual Match: 100%" (in green) and "Actual Clean Hours: XX%" (in red, counting up from 0).

Narrative panel: "The 100% Clean Illusion" text. End with "Compare procurement strategies →"
link to procurement_strategies.html.

Aesthetic: 2px outlines, 12% fills. "100% Annually Matched" badge overlay (`.story-badge`)
that cracks/fades when reality is revealed.
```

---

### Act 3: "The VRE-Gas Trap"

**Narrative:** "By only deploying VRE, we crash LMPs during easy hours, forcing the grid to keep redundant gas online for the dark hours. This capital is now 'locked in.'"

**Data source:** `lmp-data.js` — zero-price hour counts by clean %, LMP envelope (P10/P50/P90), peak-offpeak spreads. `fleet-scenarios.js` for gas fleet utilization data.

**Visual:** LMP envelope chart that animates as the user "increases" the clean target via scroll. Zero-price hours spike, revenue envelope collapses. Annotated: "Gas plants lose money 6,000 hours/year but can't retire."

**Links to:** LMP Trends page, Fleet Dispatch analysis.

**Session prompt:**
```
Build a scrollytell section for dashboard/index.html — Act 3: "The VRE-Gas Trap."

Context: `.story-section` on homepage. Vanilla JS + Chart.js. Styles from shared.css.

Data: Load from lmp-data.js:
- LMP envelope by clean % threshold (P10/P25/P50/P75/P90 bands)
- Zero-price hour counts by threshold
- Peak vs. off-peak spread by threshold
If lmp-data.js doesn't export directly, check lmp-capacity-data.js or fleet-scenarios.js.

Visual: Chart.js line chart with filled P10-P90 band. X-axis: clean energy % (50→99.99).
Y-axis: wholesale LMP ($/MWh).
- P10-P90 band filled at 12% alpha in --fossil-gas color
- P50 median line: 2px solid
- A secondary y-axis or inset showing zero-price hours count (bar chart overlay)

Scroll animation: The chart starts showing only the 50% threshold data point.
As user scrolls, a vertical "sweep line" moves rightward across clean %,
progressively revealing the LMP collapse. At ~80%, trigger an annotation:
"Zero-price hours: 4,200/yr" with a pulse animation. At 95%+:
"Gas plants lose money 6,000 hrs/yr but can't retire" in an .insight-box.

Stat cards below chart (`.stat-card` style):
- "Zero-Price Hours" (animated counter)
- "Gas Fleet Revenue Loss" (%)
- "Structural Lock-in" (qualitative badge)

Narrative: "The VRE-Gas Trap" text. End with "See the full market analysis →" link to lmp_trends.html.

Aesthetic: 2px outlines, 12% fills. Use ISO_COLORS for multi-region variant if showing more than one ISO.
```

---

### Act 4: "The Stranded Asset Cliff"

**Narrative:** "In regions like PJM, you might successfully deplete the coal wall with cheap wind. But you are left with a massive gas fleet and redundant VRE that eventually become stranded."

**Data source:** `mac-stats-data.js` — P10/P25/P50/P75/P90 fan chart data across 16,200 scenarios. `shared-data.js` — MAC_DATA with DAC/SCC crossover points. `optimal-target-data.js` — DAC cost trajectories.

**Visual:** MAC fan chart that animates in on scroll. A vertical "cliff line" sweeps from left to right, cost bands visibly explode upward. DAC/SCC reference lines appear as cost passes them.

**Links to:** Abatement Dashboard (regional deep-dives).

**Session prompt:**
```
Build a scrollytell section for dashboard/index.html — Act 4: "The Stranded Asset Cliff."

Context: `.story-section` on homepage. Vanilla JS + Chart.js. Styles from shared.css.

Data: Load from mac-stats-data.js:
- MAC fan chart data: P10/P25/P50/P75/P90 marginal abatement cost by threshold
- Use a representative ISO (PJM — most dramatic cliff) or all-region average
Also load from optimal-target-data.js:
- DAC cost scenarios (optimistic/central/conservative)
- SCC references: EPA $51/ton, Rennert $185/ton, EU ETS $60-100/ton

Visual: Chart.js line chart with stacked fill bands.
X-axis: CFE target % (50→99.99). Y-axis: $/tCO₂ (log scale or linear, whichever reads better).
- P10-P90 band: lightest fill (12% alpha)
- P25-P75 band: medium fill (25% alpha)
- P50 line: 2px solid
- Horizontal reference lines: DAC central (dashed --storage color), SCC EPA (dotted),
  SCC Rennert (dotted), EU ETS band (light fill)

Scroll animation: Chart builds progressively left to right as user scrolls.
At the inflection point (~85-90%), trigger a "cliff" annotation with a vertical
dashed red line and label: "The Wall." Cost numbers animate (counter style) as
the sweep passes key thresholds. When MAC crosses DAC: flash "Grid costs now exceed
direct air capture" in `.insight-box.insight-danger`.

Stat cards:
- "Cost at 80%" → moderate number
- "Cost at 95%" → large number (red)
- "DAC Crossover" → threshold % where grid MAC > DAC

Narrative: "The Stranded Asset Cliff" text. End with "Explore regional abatement curves →"
link to abatement_dashboard.html.
```

---

### Act 5: "The Integrated Portfolio"

**Narrative:** "The best path is incentive structures grounded in spatial and temporal realities. We must pay a necessary clean premium for FOAK Firm Clean tech today."

**Data source:** `deployment-data.js` — resource mixes at each threshold for VRE-only vs. integrated portfolio. `no-regrets-data.js` — resources that appear in optimal mix across all cost scenarios. `shared-data.js` — cost comparison data.

**Visual:** Before/after comparison. Left: VRE-only portfolio (high cost, residual emissions). Right: Integrated portfolio from optimizer (lower cost, near-zero emissions). Firm clean resources animate in to fill gaps. No-regrets resources highlighted with pulse animation.

**Links to:** Main Dashboard (interactive optimizer), Storage Analysis.

**Session prompt:**
```
Build a scrollytell section for dashboard/index.html — Act 5: "The Integrated Portfolio."

Context: `.story-section` on homepage. Vanilla JS + Chart.js. Styles from shared.css, colors from chart-colors.js.

Data: Load from deployment-data.js or shared-data.js:
- Resource mix at a high threshold (95%) for a representative ISO
- Need two scenarios: (1) VRE-only constrained mix (solar + wind + battery only),
  (2) Full integrated mix (solar + wind + nuclear/CCS + battery + LDES)
- Cost per MWh for each scenario
- Residual fossil % for each scenario
Also load from no-regrets-data.js:
- Resources that appear in optimal mix across all 5,832 cost scenarios

Visual: Two side-by-side donut charts (or stacked horizontal bars) in a `.grid-2col` layout.
Left: "VRE-Only Path" — solar, wind, battery, large gap segment (fossil/unmatched).
Right: "Integrated Path" — solar, wind, clean firm, battery, LDES, tiny/no gap.
Use RESOURCE_COLORS for all segments. 2px outlines, 12% fills.

Scroll animation: Left chart appears first (the "naive" approach). On further scroll,
right chart animates in — firm clean segments grow from zero, gap segment shrinks.
No-regrets resources get a 2px pulsing border (CSS `pulse` keyframe).

Stat comparison (`.headline-card` row):
- "VRE-Only Cost: $XX/MWh" vs. "Integrated Cost: $XX/MWh"
- "VRE-Only Clean Hours: XX%" vs. "Integrated Clean Hours: XX%"
- Delta badge: "XX% cheaper" or "XX% more clean hours"

Narrative: "The Integrated Portfolio" text. Highlight no-regrets resources by name.
End with "Build your own portfolio →" link to dashboard.html.
```

---

### Act 6: "The True Endpoint"

**Narrative:** "A high-fidelity grid where corporate procurement signals have solved the temporal and spatial dependency. We minimize waste and achieve physical decarbonization."

**Data source:** `compressed_day_profiles.json` — 24-hour dispatch stack for a high-threshold optimized mix. `scenario_a_*.json` for full 8,760-hour data if needed. Best ISO example: whichever achieves the tightest flush fit at 99%+ (likely CAISO with geothermal or ERCOT with wind).

**Visual:** Full 24-hour stacked area chart — resources fill in one by one as user scrolls: solar first (daytime), wind (fills gaps), storage (shifts energy), firm clean (seals remaining). Load line stays fixed. When complete: "99.9% hourly matched" badge.

**Links to:** Dashboard (full interactive), Research Paper.

**Session prompt:**
```
Build a scrollytell section for dashboard/index.html — Act 6: "The True Endpoint."

Context: `.story-section` on homepage. Vanilla JS + Chart.js. Styles from shared.css.

Data: Load from compressed_day_profiles.json:
- 24-hour dispatch profile for a high-threshold mix (≥99% CFE target)
- Use the best-performing ISO (check which has tightest load-match — likely CAISO or ERCOT)
- Need per-resource hourly generation: solar, wind, hydro, clean_firm, battery_discharge, ldes_discharge
- Need demand curve (flat or shaped)

Visual: Chart.js stacked area chart. X-axis: hours 0-23. Y-axis: MW or normalized to 1.0.
Stack order (bottom to top): Hydro, Nuclear/Clean Firm, Wind, Solar, Battery, LDES.
Flat demand line overlay (2px dashed black).
Each resource uses its RESOURCE_COLORS with 12% alpha fill and 2px solid outline.

Scroll animation — the climactic build:
1. First scroll position: Only demand line visible (the challenge)
2. Next: Solar fills in (daytime hump) — gap still huge at night
3. Next: Wind fills in (helps but doesn't solve night)
4. Next: Storage layers fill — battery shifts solar to evening, LDES covers deeper gaps
5. Next: Clean firm seals remaining gaps — stack meets demand line
6. Final: Badge appears — "99.9% Hourly Matched" in `.stat-value` style with success green border.
   "System Integral Solved" subtitle.

Each step triggered by scroll position. Use Chart.js dataset visibility toggling with
smooth transitions. Counter overlays update: "Matched: 47% → 68% → 82% → 94% → 99.9%"

Stat cards (final state):
- "Hourly Match: 99.9%"
- "Cost: $XX/MWh"
- "Resources: X types"

Narrative: "The True Endpoint" text. End with "Read the full research paper →" link to
research_paper.html and "Explore all 7 regions →" link to dashboard.html.
```

---

## 4. Data Integration Strategy

When prompting Claude for these components, always specify:

**Shared CSS:** "Inherit all variables and component classes from `styles/shared.css`. Use `styles/scrollytell.css` for story sections. Never write inline styles for existing components."

**Color constants:** "Use `RESOURCE_COLORS.*` and `ISO_COLORS.*` from `js/chart-colors.js`. Never hardcode hex values."

**Data sources:** "Load from existing `dashboard/js/*.js` data modules. Primary sources: `shared-data.js`, `deployment-data.js`, `mac-stats-data.js`, `lmp-data.js`, `compressed_day_profiles.json`, `procurement-strategy-data.js`."

**Scroll behavior:** "Use `js/scroll-observer.js` for intersection-based triggers. Each `.story-section` fades in on scroll entry. Chart animations trigger on section visibility."

**Responsive:** "Use shared.css grid classes (`.grid-2col`, `.grid-auto`). Charts: `responsive: true`, `maintainAspectRatio: false`. Min-height 300px mobile, 400px desktop. 44px touch targets."

**Navigation:** "Include `js/nav.js` and `js/shared-header.js`. Each act ends with an 'Explore deeper →' link to the relevant deep-dive page."

---

## 5. Implementation Order

Recommended build sequence for session work:

1. **Act 2 first** (The 100% Clean Illusion) — most visceral, uses simplest data (solar curve vs. demand), proves the scrollytell pattern works
2. **Act 6 next** (The True Endpoint) — the payoff, uses dispatch stack data, bookends the narrative
3. **Act 4** (The Stranded Asset Cliff) — MAC fan chart is the analytical centerpiece
4. **Act 1** (The Grid is Not a Bank) — heatstrip is a novel component, good opener
5. **Act 3** (The VRE-Gas Trap) — LMP envelope, ties into existing LMP page
6. **Act 5** (The Integrated Portfolio) — comparison view, depends on Acts 2+6 establishing the contrast
7. **Hero + CTA + wiring** — final pass to connect all acts with transitions and navigation

Each act is independent and can be built in a single session. Test each act standalone before integrating into the full homepage scroll sequence.

---

## 6. Adaptations from Original Doc

| Original concept | Adapted to | Rationale |
|---|---|---|
| React + useState | Vanilla JS + scroll-observer.js | Matches existing stack, no build step |
| Generic placeholder data | Real ISO optimizer results | Real data is more compelling than mockups |
| "5 nodes" heatmap | 7 actual ISOs (show 3 for clarity) | Real regional diversity from optimizer |
| Circuit diagram (Act 3) | LMP envelope animation | LMP data tells the same story with real numbers |
| FOAK scatter plot (Act 5) | Resource mix before/after comparison | Optimizer output is a stronger proof |
| "regional_pathways.json" | shared-data.js + deployment-data.js | Actual data sources in the repo |
| Standalone React components | `.story-section` blocks in index.html | Integrated into existing homepage |
