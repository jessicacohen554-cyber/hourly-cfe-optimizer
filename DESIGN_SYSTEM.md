# Dashboard Design System

Reference doc for dashboard CSS/HTML standards. Extracted from CLAUDE.md (Apr 2026) to keep CLAUDE.md lean. For pipeline architecture see `PIPELINE.md`. For project decisions see `SPEC.md`.

**All dashboard pages MUST use the centralized design system.** Never write new inline CSS for any component that already has a shared class. This section is the law.

## Architecture

- **`dashboard/styles/shared.css`** — Single source of truth for ALL visual styles (variables, components, layout, responsive rules). Every page links to this file.
- **`dashboard/js/nav.js`** — Shared navigation bar (auto-injected). Include on every page.
- **`dashboard/js/shared-header.js`** — Injects SVG waveform/heartbeat overlay into `.header` elements. Include on every page.
- **`dashboard/js/chart-colors.js`** — Canonical color constants for Chart.js (`RESOURCE_COLORS`, `ISO_COLORS`, `SEMANTIC_COLORS`). Include on every page with charts.

## Required `<head>` Includes (Every Page)

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="styles/shared.css">
<link rel="stylesheet" href="styles/{page-type}.css"> <!-- scrollytell / dashboard-ui / article / reference -->
<script src="js/nav.js"></script>
<script src="js/chart-colors.js"></script>
<script src="js/shared-header.js"></script>
<script src="js/scroll-observer.js"></script>
```

## Standard Page Header (Every Page)

```html
<div class="header" id="pageHeader">
    <h1>Page Title Here</h1>
    <div class="subtitle">One-line page description</div>
    <div class="header-accent"></div>
</div>
```

The SVG waveform overlay (energy curves + heartbeat/EKG lines) is auto-injected by `shared-header.js`. Do NOT create custom header gradients or banner styles.

## Canonical Resource Colors (NEVER Hardcode — Use These)

| Resource | CSS Variable | Hex | Chart.js Constant |
|----------|-------------|-----|-------------------|
| Solar | `--solar` | `#F59E0B` | `RESOURCE_COLORS.solar` |
| Wind (Onshore) | `--wind` | `#22C55E` | `RESOURCE_COLORS.wind` |
| Offshore Wind | `--offshore-wind` | `#009688` | `RESOURCE_COLORS.offshoreWind` |
| Hydro | `--hydro` | `#0EA5E9` | `RESOURCE_COLORS.hydro` |
| Nuclear | `--nuclear` | `#6366F1` | `RESOURCE_COLORS.nuclear` |
| CCS-CCGT | `--ccs` | `#64748B` | `RESOURCE_COLORS.ccs` |
| Clean Firm | `--clean-firm` | `#6366F1` | `RESOURCE_COLORS.cleanFirm` |
| Battery 4hr | `--battery` / `--battery4` | `#06B6D4` | `RESOURCE_COLORS.battery` / `.battery4` |
| Battery 8hr | `--battery8` | `#0891B2` | `RESOURCE_COLORS.battery8` |
| LDES | `--ldes` | `#E91E63` | `RESOURCE_COLORS.ldes` |
| Green H₂ | `--green-h2` | `#10B981` | `RESOURCE_COLORS.greenH2` |
| Geothermal | `--geothermal` | `#D97706` | `RESOURCE_COLORS.geothermal` |
| Storage | `--storage` | `#EF4444` | `RESOURCE_COLORS.storage` |
| Gap | `--gap` | `#D1D5DB` | `RESOURCE_COLORS.gap` |
| Fossil Gas | `--fossil-gas` | `#6B7280` | `RESOURCE_COLORS.fossilGas` |
| Fossil Coal | `--fossil-coal` | `#374151` | `RESOURCE_COLORS.fossilCoal` |
| Fossil Oil | `--fossil-oil` | `#92400E` | `RESOURCE_COLORS.fossilOil` |
| Solar+Batt 4hr | `--solar-batt4` | `#E6890B` | `RESOURCE_COLORS.solarBatt4` |
| Solar+Batt 8hr | `--solar-batt8` | `#CC7A0A` | `RESOURCE_COLORS.solarBatt8` |
| Wind+Batt 4hr | `--wind-batt4` | `#1AA34E` | `RESOURCE_COLORS.windBatt4` |
| Wind+Batt 8hr | `--wind-batt8` | `#158F42` | `RESOURCE_COLORS.windBatt8` |

## Canonical ISO Colors

| ISO | CSS Variable | Hex | Chart.js Constant |
|-----|-------------|-----|-------------------|
| CAISO | `--iso-caiso` | `#F59E0B` | `ISO_COLORS.CAISO` |
| ERCOT | `--iso-ercot` | `#22C55E` | `ISO_COLORS.ERCOT` |
| PJM | `--iso-pjm` | `#0EA5E9` | `ISO_COLORS.PJM` |
| NYISO | `--iso-nyiso` | `#E91E63` | `ISO_COLORS.NYISO` |
| NEISO | `--iso-neiso` | `#9C27B0` | `ISO_COLORS.NEISO` |
| MISO | `--iso-miso` | `#F97316` | `ISO_COLORS.MISO` |
| SPP | `--iso-spp` | `#14B8A6` | `ISO_COLORS.SPP` |

Each color has transparent variants: CSS `--iso-caiso-t` (12% opacity) / JS `ISO_COLORS.CAISO_T`.

## Standard Component Classes (Use Instead of Custom CSS)

| Component | Class | Notes |
|-----------|-------|-------|
| White card | `.card` | White bg, light border, subtle shadow |
| Chart panel | `.chart-panel` | Glass effect, blur backdrop |
| Stat card | `.stat-card` + `.stat-value` + `.stat-label` | Metric display |
| Insight callout | `.insight-box` | Blue left border; variants: `.insight-warn`, `.insight-danger`, `.insight-success` |
| Section container | `.content-section` | 1320px max-width, padded |
| Narrow container | `.content-section-narrow` | 900px max-width |
| Section heading | `.section-title` | Navy, heading font |
| Section subtitle | `.section-subtitle` | Muted, body font |
| Toggle group | `.toggle-btn-group` + `button.active` | L/M/H toggles |
| ISO selector | `.iso-selector` + `.iso-btn.active` | ISO pill buttons |
| Chart container | `.chart-container` | 320px min-height |
| Chart small | `.chart-container-sm` | 240px min-height |
| Chart large | `.chart-container-lg` | 400px min-height |
| 2-column grid | `.grid-2col` | Responsive, collapses to 1col on mobile |
| 3-column grid | `.grid-3col` | Responsive |
| Auto-fit grid | `.grid-auto` | `minmax(280px, 1fr)` |
| Stats grid | `.grid-stats` | `minmax(120px, 1fr)` |
| Data table | `.data-table` | Compact, hover rows |
| Legend | `.legend` + `.legend-item` + `.legend-dot` | Chart legend |
| Headline card | `.headline-card` + `.val` + `.lbl` | Hero stats |
| Story section | `.story-section` | Scrollytell with fade-in |
| Badge | `.story-badge` | Pill tag; variants: `.story-badge-red`, `.story-badge-green` |
| Footer | `.page-footer` | Dark navy footer |
| Bottom accent | `.bottom-banner` | 4px gradient bar |

## Rules for New Pages or Features

1. **NEVER write inline `<style>` blocks for components that exist in shared.css.** Page-specific styles are ONLY for layouts/elements unique to that page.
2. **NEVER hardcode font-family** — use `var(--font-heading)`, `var(--font-body)`, `var(--font-data)`, `var(--font-mono)`.
3. **NEVER hardcode hex colors for resources or ISOs** — use CSS variables in styles, `RESOURCE_COLORS.*` / `ISO_COLORS.*` in Chart.js.
4. **NEVER create custom header/banner gradients** — use `.header` class and `shared-header.js` for the SVG overlay.
5. **NEVER duplicate footer styles** — use `.page-footer`, `.footer-links`, `.bottom-banner`.
6. **Use spacing variables** — `var(--space-xs)` through `var(--space-3xl)` and `var(--pad-page)`.
7. **Use shadow variables** — `var(--shadow-sm)` through `var(--shadow-xl)`.
8. **Use radius variables** — `var(--radius-sm)` through `var(--radius-pill)`.
9. **Body background** — use `var(--bg-page)` (light gray default) or `var(--bg-page-white)`. Never hardcode.
10. **If a shared component is close but not quite right**, extend it with a modifier class rather than creating a new component. Add the modifier to shared.css if it will be reused.
