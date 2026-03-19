# Fleet Scenario Comparison — Visualization Design Spec

> **Status**: Design only — no HTML/JS yet. Implements Prompt 4.1 from `FLEET-SCENARIO-PLAYBOOK.md`.

> **Page**: `market-simulator/frontend/fleet-scenarios.html`

> **Data source**: `fleet_scenario_results.json` (Phase 3 output)

---

## Page Structure

Uses the Constellation design system (`CEG-style.css`) with the standard page template from `CONSTELLATION_STYLE_GUIDE.md`.

```
┌─────────────────────────────────────────────────────┐
│  Nav (fixed, frosted glass, 60px)                   │
│  Progress bar (3px gradient)                        │
├─────────────────────────────────────────────────────┤
│  Hero Section                                       │
│  Eyebrow: "FLEET ANALYSIS"                          │
│  H1: Fleet Scenario <strong>Comparison</strong>     │
│  Desc: P10/P50/P90 emissions envelopes across       │
│        fleet decarbonization scenarios               │
├─────────────────────────────────────────────────────┤
│  ┌─ Controls Strip (.glass-card) ──────────────┐    │
│  │ Scenario Checkboxes │ Target Selector │ Year │    │
│  │                     │ (SBTi/AT/Custom)│Slider│    │
│  └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│  ┌─ Main Chart: Emissions Fan Chart ───────────┐    │
│  │  Full-width .glass-card                      │    │
│  │  Chart.js canvas, min-height 480px           │    │
│  │  X: 2023–2050  Y: Fleet Emissions (Mt CO₂)  │    │
│  │  P10/P50/P90 bands per scenario              │    │
│  │  Target overlay lines                        │    │
│  │  Custom legend (buildLegend utility)         │    │
│  └──────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│  ┌─ Secondary Charts (2-col grid) ─────────────┐    │
│  │ ┌──────────────────┐ ┌────────────────────┐  │    │
│  │ │ Plant Waterfall   │ │ Generation Mix     │  │    │
│  │ │ (bar, horizontal) │ │ (grouped bar)      │  │    │
│  │ │ min-height 400px  │ │ min-height 360px   │  │    │
│  │ └──────────────────┘ └────────────────────┘  │    │
│  │ ┌──────────────────────────────────────────┐  │    │
│  │ │ Emissions by Fuel (stacked area)         │  │    │
│  │ │ Full-width, min-height 360px             │  │    │
│  │ └──────────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│  Footer + bottom-banner (3px gradient)              │
└─────────────────────────────────────────────────────┘
```

---

## Controls

### Scenario Checkboxes

- Positioned in a horizontal `.glass-card` strip below the hero
- Each scenario gets a checkbox + colored swatch + label
- Toggling a checkbox shows/hides that scenario's fan band + P50 line on the main chart
- All scenarios visible by default on page load
- Uses standard form styling from `simulator.css`

| Scenario | Color (hex) | Swatch |
|----------|-------------|--------|
| Baseline | `#7E8083` (--ceg-gray) | Gray band |
| CCS-Only | `#2372B9` (--ceg-blue) | Blue band |
| CCS + New Gas | `#F47B27` (--ceg-orange) | Orange band |
| Retire + CCS | `#6BA543` (--ceg-green) | Green band |

These are scenario-specific colors, not resource colors. They don't conflict with `RESOURCE_COLORS` or `ISO_COLORS` from `chart-colors.js` — they use the brand palette from `CEG-style.css`.

### Target Selector

- Dropdown or radio-button group (3 options):
  - **SBTi 1.5°C** — dashed line, `#DC2626` (red), -4.2% annual from 2023 baseline
  - **AT Power NZ** — solid line, `#6366F1` (purple), company-committed target
  - **Custom** — dotted line, `#6B7280` (gray), user-defined {year: Mt} pairs
- Changing target updates the overlay line on the main chart and recalculates gap-to-target in tooltips
- Only one target visible at a time (radio behavior), or allow multi-select with distinct line styles

### Year Slider

- HTML range input, 2023–2050, step = 1
- Current year displayed as a label next to the slider
- Controls which year the **waterfall chart** displays
- Also draws a vertical reference line on the main fan chart at the selected year
- Styled per `simulator.css` range input pattern

---

## Chart 1: Emissions Fan Chart (Main)

**Type**: Chart.js line chart with filled areas

**Container**: Full-width `.glass-card`, min-height 480px desktop / 360px mobile

### Data Mapping

For each visible scenario, render 3 datasets:
1. **P10/P90 band** — two `line` datasets with `fill: '+1'` (fill between). Border: scenario color at 0.3 opacity. Fill: scenario color at 0.15 opacity.
2. **P50 line** — `line` dataset, solid 2.5px border in scenario color, `fill: false`.

### Target Overlays

Rendered as additional `line` datasets on top:
- **SBTi 1.5°C**: `borderDash: [8, 4]`, color `#DC2626`, 2px width
- **AT Power NZ**: solid line, color `#6366F1`, 2px width
- **Custom**: `borderDash: [4, 4]`, color `#6B7280`, 2px width

### Axes

- **X-axis**: Years 2023–2050, labeled at 2023, 2025, 2030, 2035, 2040, 2045, 2050 (skip intermediate to avoid crowding)
- **Y-axis**: Fleet emissions (Mt CO₂), auto-scaled with 10% padding above max P90
- Grid lines: light gray (`#E0E6EF`), no vertical grid lines

### Year Slider Reference Line

- Vertical dashed line at the year selected by the slider
- Color: `#1A232F` at 0.3 opacity, `borderDash: [4, 4]`
- Drawn via Chart.js annotation plugin or a custom vertical line dataset

### Tooltip

On hover at any year point, show a structured tooltip:

```
─── 2035 ───────────────────
Baseline        P10: 42.1  P50: 48.3  P90: 55.7
CCS-Only        P10: 28.4  P50: 33.1  P90: 39.2
CCS + New Gas   P10: 30.1  P50: 35.8  P90: 42.0
Retire + CCS    P10: 18.2  P50: 22.5  P90: 28.1

Gap to SBTi:    Baseline +16.8 Mt | CCS-Only +1.6 Mt | Retire -9.0 Mt ✓
```

- Use Chart.js `external` tooltip callback for custom HTML tooltip
- Color-code each scenario row with its swatch color
- Gap-to-target: positive (red text, above target), negative (green text, below target + checkmark)
- Tooltip positioned to avoid chart edge clipping

### Legend

- Use `buildLegend()` from `chart-colors.js` placed below the chart
- Items:
  - Each scenario: `type: 'band'` with scenario color
  - SBTi target: `type: 'dashed'` with red
  - AT Power NZ: `type: 'line'` with purple
  - Custom target: `type: 'dashed'` with gray
- Only show legend items for currently visible scenarios/targets

### Interactions

- **Checkbox toggle**: Animate scenario bands in/out with Chart.js `update('active')` transition (300ms ease)
- **Hover**: Cross-hair cursor, tooltip tracks mouse X position snapping to nearest year
- **Click on P50 line**: Snap the year slider to that year (updates waterfall chart)

---

## Chart 2: Plant-Level Waterfall

**Type**: Chart.js horizontal bar chart

**Container**: Left column of 2-col grid, `.glass-card`, min-height 400px

### Purpose

Shows which plants contribute how much to the emissions delta between **baseline** and **the selected scenario** at the **year slider's current year**.

### Data Mapping

- Compute per-plant: `delta_mt = baseline_emissions - scenario_emissions` at the selected year
- Sort plants by absolute delta descending (biggest contributors first)
- Top 15 plants shown; remainder aggregated into "Other" bar
- Positive delta (green bar) = emissions reduced by this plant
- Negative delta (red bar) = emissions increased (e.g., new gas plant)

### Visual

- Horizontal bars, plant name as Y-axis label
- Green bars (`#6BA543`) for reductions, red bars (`#DC2626`) for increases
- Running total line overlay showing cumulative delta
- Bar labels: `±X.XX Mt` on the bar face (inside if bar is wide enough, outside if narrow)

### Header

- `.section-eyebrow`: "PLANT IMPACT"
- Title: "Emissions Delta vs. Baseline — {Year}" (updates with slider)
- Subtitle showing which scenario is selected

### Controls

- Scenario selector (if multiple non-baseline scenarios are toggled, pick the first checked one, or add a small dropdown above the chart)
- Year comes from the main slider — no separate control needed

---

## Chart 3: Generation Mix Bar Chart

**Type**: Chart.js stacked bar chart (vertical)

**Container**: Right column of 2-col grid, `.glass-card`, min-height 360px

### Purpose

Fleet generation (TWh) by fuel type for each scenario at the selected year.

### Data Mapping

- One bar group per visible scenario (X-axis = scenario names)
- Stacked segments per fuel type: Coal, Gas CCGT, Gas CT, Oil, Nuclear, CCS-CCGT
- Use `RESOURCE_COLORS` from `chart-colors.js`:
  - Coal: `RESOURCE_COLORS.fossilCoal` (`#2C3E50`)
  - Gas CCGT: `RESOURCE_COLORS.fossilGas` (`#2372B9`)
  - Gas CT: `RESOURCE_COLORS.fossilGasCT` (`#007FA4`)
  - Oil: `RESOURCE_COLORS.fossilOil` (`#9B6B3A`)
  - Nuclear: `RESOURCE_COLORS.nuclear` (`#6366F1`)
  - CCS-CCGT: `RESOURCE_COLORS.ccs` (`#64748B`)

### Header

- `.section-eyebrow`: "GENERATION MIX"
- Title: "Fleet Generation by Fuel — {Year}"

### Tooltip

Hover on segment → show fuel type, TWh, % of total generation for that scenario.

---

## Chart 4: Emissions by Fuel (Stacked Area)

**Type**: Chart.js stacked area chart (line with `fill`)

**Container**: Full-width below the 2-col grid, `.glass-card`, min-height 360px

### Purpose

Shows how coal/gas/oil emissions evolve over time for the **currently selected scenario**.

### Data Mapping

- X-axis: years 2023–2050 (same as main chart)
- Y-axis: Emissions (Mt CO₂)
- Stacked areas (bottom to top): Coal, Gas CCGT, Gas CT, Oil
- Same fuel colors as Chart 3
- Uses P50 values for the selected scenario

### Header

- `.section-eyebrow`: "EMISSIONS BREAKDOWN"
- Title: "Emissions by Fuel Type — {Scenario Name}"

### Controls

- Responds to scenario checkbox selection. If multiple checked, show a small dropdown to pick which one to display (can't stack multiple scenarios in a stacked area).

### Tooltip

Hover at year → show each fuel's Mt CO₂ and percentage of total.

---

## Responsive Behavior

| Breakpoint | Layout |
|------------|--------|
| > 900px | Controls strip horizontal. Main chart full-width. Secondary charts 2-col grid (waterfall + gen mix side by side), emissions-by-fuel full-width below. |
| ≤ 900px | Controls stack vertically. All charts single column. |
| ≤ 768px | Reduced padding. Chart min-heights drop (360px main, 300px secondary). Tooltip repositions to bottom of chart on mobile. Year slider full-width. |
| ≤ 480px | Scenario checkboxes wrap to 2×2 grid. Chart font sizes scale down (min 11px). |

All charts: `responsive: true`, `maintainAspectRatio: false`.

Touch targets: all checkboxes, buttons, slider thumb ≥ 44px.

---

## Data Contract

The page loads `fleet_scenario_results.json` via `fetch()`. Expected schema:

```json
{
  "scenarios": {
    "baseline": {
      "description": "Status quo — no CCS, no retirements",
      "color": "#7E8083",
      "envelope": {
        "2023": {"p10": 52.1, "p50": 55.3, "p90": 58.7},
        "2030": {"p10": 45.2, "p50": 48.3, "p90": 52.1},
        ...
      },
      "plant_detail": {
        "2030": [
          {"name": "Brandon Shores", "orispl": 1553, "fuel_type": "gas_ccgt",
           "gen_twh": 4.2, "emissions_mt": 1.8, "status": "operating"}
        ]
      },
      "generation_by_fuel": {
        "2030": {"coal_steam": 12.5, "gas_ccgt": 45.3, "gas_ct": 8.1,
                 "oil_ct": 0.2, "nuclear": 0.0, "ccs_ccgt": 0.0}
      },
      "emissions_by_fuel": {
        "2030": {"coal_steam": 12.1, "gas_ccgt": 18.2, "gas_ct": 5.4, "oil_ct": 0.2}
      }
    },
    "ccs_only": { ... },
    "ccs_plus_new_gas": { ... },
    "retire_coal_ccs_gas": { ... }
  },
  "targets": {
    "sbti_15": {
      "label": "SBTi 1.5°C",
      "trajectory": {"2023": 55.3, "2030": 41.2, "2035": 33.8, ...}
    },
    "at_power_nz": {
      "label": "AT Power NZ",
      "trajectory": {"2023": 55.3, "2030": 38.0, "2040": 20.0, "2050": 0.0}
    }
  },
  "gap_analysis": {
    "baseline": {
      "sbti_15": {
        "gap_mt": {"2030": 7.1, "2040": 18.3, ...},
        "year_achieved": null,
        "prob_meeting": {"2030": 0.12, "2040": 0.03}
      }
    }
  },
  "metadata": {
    "fleet_name": "Constellation Energy",
    "sweep_count": 405,
    "generated_at": "2026-03-19T12:00:00Z"
  }
}
```

---

## Includes (HTML `<head>`)

```html
<link rel="stylesheet" href="styles/CEG-style.css">
<link rel="stylesheet" href="styles/simulator.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
<script src="js/chart-colors.js"></script>
```

---

## CSS Notes

- **No custom inline styles** for components that exist in `CEG-style.css` (cards, hero, nav, footer, section headers, buttons)
- Page-specific styles limited to:
  - `.scenario-checkbox-strip` — horizontal layout for the controls bar
  - `.year-slider-container` — wrapper for range input + year label
  - Fan chart tooltip (`.fan-tooltip`) — positioned absolutely, white card with shadow
- All spacing via CSS variables (`--space-xs` through `--space-3xl`)
- All radii via `--radius-sm` / `--radius-md` / `--radius-lg`
- All shadows via `--shadow-sm` / `--shadow-md`

---

## Nav Integration

Add to the existing nav in `<nav>` → `<ul class="nav-links">`:

```html
<li><a href="/fleet-scenarios">Fleet Scenarios</a></li>
```

Position: after "CCS Emissions", before "IPP Report".

---

## Animation & Transitions

- **Scenario toggle**: Chart.js `update('active')` with 300ms easing. Bands fade in/out.
- **Year slider**: Waterfall and gen-mix charts update on `input` event (live drag), not just `change`. Use `requestAnimationFrame` debounce if needed for smooth performance.
- **Page load**: Charts animate in via Chart.js default `animation.duration: 800`.
- **No scroll-triggered animations** — this is a tool page, not a storytelling page.
