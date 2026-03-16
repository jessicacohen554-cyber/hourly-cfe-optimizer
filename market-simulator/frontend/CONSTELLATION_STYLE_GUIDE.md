# Constellation Energy — Design System & Style Guide

Copy-pasteable design system for all Constellation Energy internal tools.
Based on `fossil_fleet_supply_stack.html` reference implementation.

---

## 1. Font Stack

```css
--font: 'Benton Sans Light', 'Franklin Gothic Book', 'Source Sans Pro', Calibri, Arial, sans-serif;
```

**Google Fonts fallback** (add to `<head>`):
```html
<link rel="stylesheet" href="styles/CEG-style.css">
```

The `@import` for Source Sans Pro is built into CEG-style.css. No separate font link needed.

**Weight usage:**
| Weight | Use |
|--------|-----|
| 300 (Light) | Body text, h1, h2, descriptions |
| 400 (Regular) | Secondary text, labels |
| 600 (SemiBold) | Nav links, buttons, table headers, labels |
| 700 (Bold) | Eyebrow labels, KPI values, strong emphasis |

---

## 2. Color Palette

### Brand Colors
| Variable | Hex | Use |
|----------|-----|-----|
| `--ceg-blue` | `#2372B9` | Primary brand — buttons, active states, links, KPI values |
| `--ceg-orange` | `#F47B27` | Accent — eyebrow labels, CTAs, warnings |
| `--ceg-green` | `#6BA543` | Success — positive values, growth indicators |
| `--ceg-teal` | `#007FA4` | Secondary blue — gradients, Gas CT color |
| `--ceg-yellow` | `#FBB254` | Warning — highlights, caution states |
| `--ceg-lime` | `#CADB2E` | Tertiary accent — gradient endpoints |
| `--ceg-gray` | `#7E8083` | Muted text, secondary labels |
| `--ceg-slate` | `#7F8F97` | Subtle text, hints |

### Backgrounds
| Variable | Hex | Use |
|----------|-----|-----|
| `--bg-dark` | `#FFFFFF` | Page background (white) |
| `--bg-mid` | `#F7FAFC` | Section backgrounds, alternating rows, footer |
| `--bg-card` | `#FFFFFF` | Card backgrounds |
| `--glass-bg` | `#FFFFFF` | Glass-effect card background |
| `--glass-border` | `#E0E6EF` | Card borders, dividers |

### Text
| Variable | Hex | Use |
|----------|-----|-----|
| `--text-bright` | `#1A232F` | Primary text — headings, body, labels |
| `--text-dim` | `#7E8083` | Secondary text — descriptions, captions |

### Fuel/Resource Colors
| Resource | Hex | Variable |
|----------|-----|----------|
| Coal | `#2C3E50` | `--coal-bar` |
| Gas CCGT | `#2372B9` | `--ng-cc` |
| Gas CT | `#007FA4` | `--ng-ct` |
| Oil CT | `#9B6B3A` | `--oil-ct` |

---

## 3. Typography Scale

```css
h1 { font-size: clamp(2.2rem, 4vw, 3.6rem); font-weight: 300; line-height: 1.15; }
h1 strong { font-weight: 700; color: var(--ceg-blue); }

h2 { font-size: clamp(1.6rem, 3vw, 2.4rem); font-weight: 300; }
h2 strong { font-weight: 700; }

h3 { font-size: clamp(1.2rem, 2vw, 1.6rem); font-weight: 600; }

body/p { font-size: 1.05rem; font-weight: 300; line-height: 1.7; }

/* Eyebrow labels */
.section-eyebrow {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 2px;
  text-transform: uppercase; color: var(--ceg-blue);
}

/* Hero eyebrow (orange, with dash) */
.hero-eyebrow {
  font-size: 0.75rem; font-weight: 700; letter-spacing: 2px;
  text-transform: uppercase; color: var(--ceg-orange);
}
.hero-eyebrow::before { content: ''; width: 24px; height: 2px; background: var(--ceg-orange); }
```

---

## 4. Navigation

Fixed frosted-glass nav bar, 60px height:

```html
<nav>
  <div class="nav-logo">
    <div class="nav-logo-mark">
      <!-- SVG logo or <img src="logo.png"> -->
    </div>
    <div>
      <div class="nav-title">Constellation Energy</div>
      <div class="nav-subtitle">Tool Name Here</div>
    </div>
  </div>
  <ul class="nav-links">
    <li><a href="#" class="active">Page 1</a></li>
    <li><a href="#">Page 2</a></li>
  </ul>
</nav>
```

```css
nav {
  position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
  background: #FFFFFFCC;
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--glass-border);
  height: 60px;
}
```

---

## 5. Hero Section

White background with subtle radial gradients — NOT a colored banner:

```html
<div class="hero-section">
  <div class="hero-inner">
    <div class="hero-eyebrow">Tool Category</div>
    <h1>Page <strong>Title</strong></h1>
    <p class="hero-desc">One-line description of what this page does.</p>
  </div>
</div>
```

```css
.hero-section {
  padding-top: 60px;
  background:
    radial-gradient(ellipse 80% 60% at 60% 40%, rgba(35,114,185,0.06) 0%, transparent 70%),
    radial-gradient(ellipse 40% 40% at 20% 70%, rgba(0,127,164,0.04) 0%, transparent 60%),
    white;
}
```

---

## 6. Cards & Panels

### Glass Card (default)
```css
.glass-card {
  background: white;
  border: 1px solid #E0E6EF;
  border-radius: 14px;
  padding: 1.5rem;
  backdrop-filter: blur(8px);
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
```

### KPI Card
```html
<div class="kpi-card">
  <div class="kpi-value">65</div>
  <div class="kpi-unit">plants</div>
  <div class="kpi-label">CEG Fleet</div>
</div>
```

### Insight Chip (horizontal scroll strip)
```html
<div class="insight-chip orange">
  <div class="insight-number">$42.50</div>
  <div class="insight-text">Average fleet marginal cost</div>
</div>
```

### Story Callout
```html
<div class="story-callout">Blue-bordered callout for key insights.</div>
<div class="story-callout orange">Orange for warnings/action items.</div>
<div class="story-callout green">Green for positive outcomes.</div>
```

---

## 7. ISO Tab Switcher

```html
<div class="iso-tabs">
  <button class="iso-tab active">ERCOT<span class="iso-tab-sub">Texas</span></button>
  <button class="iso-tab">PJM<span class="iso-tab-sub">Mid-Atlantic</span></button>
  <button class="iso-tab">CAISO<span class="iso-tab-sub">California</span></button>
</div>
```

Active tab gets `box-shadow: inset 0 -3px 0 var(--ceg-blue)` underline.

---

## 8. Buttons

```html
<a class="btn-primary">Primary Action</a>
<a class="btn-outline">Secondary Action</a>
```

Primary: `--ceg-blue` fill, white text, 8px radius, uppercase, 700 weight.
Outline: transparent, `--glass-border` border, hover → blue border + blue text.

---

## 9. Section Structure

```html
<section>
  <div class="section-header">
    <div class="section-eyebrow">Category</div>
    <h2>Section <strong>Title</strong></h2>
    <p class="section-desc">Brief description of this section.</p>
  </div>
  <!-- content -->
</section>
```

Default section padding: `5rem 4rem` (desktop), `3rem 1.5rem` (mobile).
Max-width: `1400px`, centered.

---

## 10. Footer

Light gray footer — NOT dark navy:

```html
<footer>
  <div class="footer-brand"><strong>Constellation Energy</strong> · Tool Name</div>
  <div class="footer-note">Legal/descriptive note here</div>
</footer>
<div class="bottom-banner"></div>
```

```css
footer {
  background: #F7FAFC;
  border-top: 1px solid #E0E6EF;
  padding: 3rem 4rem;
}

.bottom-banner {
  height: 3px;
  background: linear-gradient(90deg, #2372B9, #007FA4, #6BA543);
}
```

---

## 11. Progress Bar

Thin 3px gradient bar below the nav:

```html
<div class="progress-bar"><div class="progress-bar-fill"></div></div>
```

```css
.progress-bar {
  position: fixed; top: 60px; left: 0; right: 0; height: 3px;
  background: rgba(224,230,239,0.3); z-index: 999;
}
.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #2372B9, #007FA4);
}
```

---

## 12. Spacing & Radius Variables

```css
--space-xs: 0.25rem;  --space-sm: 0.5rem;  --space-md: 1rem;
--space-lg: 1.5rem;   --space-xl: 2rem;    --space-2xl: 3rem;
--space-3xl: 4rem;

--radius-sm: 6px;     --radius-md: 10px;   --radius-lg: 14px;
--radius-pill: 9999px;

--shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
--shadow-md: 0 4px 12px rgba(0,0,0,0.08);
--shadow-lg: 0 8px 24px rgba(0,0,0,0.10);
```

---

## 13. Scrollbar

Thin 4px scrollbar with brand blue thumb:

```css
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: white; }
::-webkit-scrollbar-thumb { background: #2372B9; border-radius: 2px; }
```

---

## 14. Responsive Breakpoints

| Breakpoint | Behavior |
|------------|----------|
| `> 900px` | Full layout — 2/3 column grids, side-by-side nav |
| `≤ 900px` | Single column, nav links hidden |
| `≤ 768px` | Reduced padding, smaller type, compact tables |
| `≤ 480px` | 2-col stats grid, stacked ISO buttons |

---

## 15. Quick Start

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tool Name — Constellation Energy</title>
  <link rel="stylesheet" href="styles/CEG-style.css">
</head>
<body>
  <div class="progress-bar"><div class="progress-bar-fill"></div></div>
  <nav>
    <div class="nav-logo">
      <div class="nav-logo-mark"><!-- logo --></div>
      <div>
        <div class="nav-title">Constellation Energy</div>
        <div class="nav-subtitle">Tool Name</div>
      </div>
    </div>
    <ul class="nav-links">
      <li><a href="#" class="active">Page</a></li>
    </ul>
  </nav>

  <div class="hero-section">
    <div class="hero-inner">
      <div class="hero-eyebrow">Category</div>
      <h1>Page <strong>Title</strong></h1>
      <p class="hero-desc">Description here.</p>
    </div>
  </div>

  <div class="content-section">
    <!-- Your content here -->
  </div>

  <footer>
    <div class="footer-brand"><strong>Constellation Energy</strong> · Tool Name</div>
    <div class="footer-note">Note</div>
  </footer>
  <div class="bottom-banner"></div>
</body>
</html>
```

---

## File Locations

| File | Path | Purpose |
|------|------|---------|
| `CEG-style.css` | `market-simulator/frontend/styles/` | Full design system |
| `CEG-style.css` | `data-center-cfe/` | Copy for data center CFE project |
| `shared.css` | `market-simulator/frontend/styles/` | Identical to CEG-style.css |
| `simulator.css` | `market-simulator/frontend/styles/` | Form controls (extends CEG-style) |
| `results.css` | `market-simulator/frontend/styles/` | Results page (extends CEG-style) |
