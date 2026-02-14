# Hourly CFE Optimizer — Visual Style Guide

Copy-pasteable design system for replicating this project's look and feel in other repos.

---

## 1. Google Fonts (add to `<head>`)

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700&family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Rajdhani:wght@400;500;600;700&family=Barlow+Semi+Condensed:wght@400;500;600;700;800&display=swap" rel="stylesheet">
```

### Franklin Gothic Demi (system font fallback)

```css
@font-face {
    font-family: 'Franklin Gothic Demi';
    src: local('Franklin Gothic Demi'), local('Franklin Gothic Medium'),
         local('ITC Franklin Gothic Demi'), local('Franklin Gothic Demi Cond');
    font-weight: 600;
    font-style: normal;
    font-display: swap;
}
```

### Font Stack Usage

| Role | Stack |
|------|-------|
| **Body text** | `'Plus Jakarta Sans', 'Helvetica Neue', Arial, sans-serif` |
| **Headings** | `'Barlow Semi Condensed', 'Arial Narrow', 'Helvetica Neue', Arial, sans-serif` |
| **Banner title** | `'Franklin Gothic Demi', 'Lexend', 'Rajdhani', 'Arial Narrow', 'Helvetica Neue', Arial, sans-serif` |
| **Banner subtitle** | `'Rajdhani', 'Arial Narrow', 'Helvetica Neue', Arial, sans-serif` |
| **Monospace** | `'Consolas', 'Courier New', monospace` |

---

## 2. CSS Custom Properties (paste into `:root`)

```css
:root {
    /* ---- Core palette ---- */
    --navy: #1A2744;
    --navy-light: #2D3A52;
    --accent: #EF4444;
    --accent-dark: #DC2626;

    /* ---- Surfaces ---- */
    --bg: #F3F4F8;
    --bg-alt: #F8F9FC;
    --bg-card: #FFFFFF;
    --bg-card-hover: #F7F8FA;
    --border: #D4D8E0;
    --border-light: #E5E7EB;

    /* ---- Text ---- */
    --text-primary: #000000;
    --text-secondary: #374151;
    --text-muted: #6B7280;
    --link: #1D4ED8;
    --link-hover: #1E40AF;

    /* ---- Energy resource colors ---- */
    --clean-firm: #1E3A5F;
    --solar: #F59E0B;
    --wind: #22C55E;
    --hydro: #0EA5E9;
    --storage: #EF4444;
    --gap: #D1D5DB;
    --outline: #374151;

    /* ---- Grays ---- */
    --primary-grey: #6B7280;
    --secondary-grey: #9CA3AF;
    --red: #EF4444;
    --red-light: rgba(239, 68, 68, 0.08);
    --indigo: #1A2744;

    /* ---- 55% opacity resource variants ---- */
    --clean-firm-70: rgba(30, 58, 95, 0.55);
    --solar-70: rgba(245, 158, 11, 0.55);
    --wind-70: rgba(34, 197, 94, 0.55);
    --hydro-70: rgba(14, 165, 233, 0.55);
    --storage-70: rgba(239, 68, 68, 0.55);
    --gap-70: rgba(209, 213, 219, 0.55);

    /* ---- Fonts ---- */
    --font-body: 'Plus Jakarta Sans', 'Helvetica Neue', Arial, sans-serif;
    --font-heading: 'Barlow Semi Condensed', 'Arial Narrow', 'Helvetica Neue', Arial, sans-serif;
    --font-heading-banner: 'Rajdhani', 'Arial Narrow', 'Helvetica Neue', Arial, sans-serif;
    --font-mono: 'Consolas', 'Courier New', monospace;
}
```

---

## 3. Reset & Base Styles

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html {
    font-size: 17px;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

body {
    font-family: var(--font-body);
    background: #0F1A2E;
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(14,165,233,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(245,158,11,0.04) 0%, transparent 50%);
    background-attachment: fixed;
    color: var(--text-primary);
    line-height: 1.55;
    min-height: 100vh;
}
```

---

## 4. Banner / Header (complete copy-paste block)

### HTML structure

```html
<div class="header">
    <div class="header-accent"></div>
    <h1>Your Project Title</h1>
    <div class="subtitle">Your tagline or description goes here</div>
</div>
```

### CSS

```css
/* ========== HEADER ========== */
.header {
    background: linear-gradient(135deg, #0F1A2E 0%, #122952 30%, #1565C0 70%, #0D47A1 100%);
    color: #ffffff;
    padding: 90px 24px 80px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

/* Radial energy glows + dispatch stack curves (bottom) */
.header::before {
    content: '';
    position: absolute;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background:
        radial-gradient(ellipse 55% 70% at 12% 75%, rgba(14,165,233,0.4) 0%, transparent 65%),
        radial-gradient(ellipse 45% 60% at 38% 85%, rgba(34,197,94,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 50% 65% at 62% 80%, rgba(245,158,11,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 40% 55% at 88% 70%, rgba(239,68,68,0.3) 0%, transparent 55%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 240' preserveAspectRatio='none'%3E%3Cdefs%3E%3ClinearGradient id='a1' x1='0' y1='0' x2='0' y2='1'%3E%3Cstop offset='0' stop-color='rgba(255,255,255,0.15)'/%3E%3Cstop offset='1' stop-color='rgba(255,255,255,0.01)'/%3E%3C/linearGradient%3E%3ClinearGradient id='a2' x1='0' y1='0' x2='0' y2='1'%3E%3Cstop offset='0' stop-color='rgba(34,197,94,0.35)'/%3E%3Cstop offset='1' stop-color='rgba(34,197,94,0.03)'/%3E%3C/linearGradient%3E%3ClinearGradient id='a3' x1='0' y1='0' x2='0' y2='1'%3E%3Cstop offset='0' stop-color='rgba(245,158,11,0.35)'/%3E%3Cstop offset='1' stop-color='rgba(245,158,11,0.03)'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cpath d='M0,240 L0,130 C200,100 400,70 600,80 C800,90 1000,120 1200,100 L1200,240 Z' fill='url(%23a1)'/%3E%3Cpath d='M0,240 L0,160 C200,140 400,115 600,125 C800,135 1000,155 1200,140 L1200,240 Z' fill='url(%23a2)'/%3E%3Cpath d='M0,240 L0,190 C200,180 400,165 600,170 C800,175 1000,188 1200,180 L1200,240 Z' fill='url(%23a3)'/%3E%3Cpath d='M0,130 C200,100 400,70 600,80 C800,90 1000,120 1200,100' fill='none' stroke='rgba(255,255,255,0.45)' stroke-width='2'/%3E%3Cpath d='M0,160 C200,140 400,115 600,125 C800,135 1000,155 1200,140' fill='none' stroke='%2322C55E' stroke-width='2' stroke-opacity='0.6'/%3E%3Cpath d='M0,190 C200,180 400,165 600,170 C800,175 1000,188 1200,180' fill='none' stroke='%23F59E0B' stroke-width='2' stroke-opacity='0.6'/%3E%3C/svg%3E") no-repeat bottom center;
    background-size: 100% 100%, 100% 100%, 100% 100%, 100% 100%, 100% 100%;
}

/* Vibrant pulse lines (top) + dark text backing overlay */
.header::after {
    content: '';
    position: absolute;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background:
        radial-gradient(ellipse 70% 45% at 50% 45%, rgba(5,15,35,0.55) 0%, transparent 100%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 80' preserveAspectRatio='none'%3E%3Cpath d='M0,50 L250,50 L270,18 L288,68 L306,22 L324,60 L342,50 L1200,50' fill='none' stroke='rgba(14,165,233,0.5)' stroke-width='2'/%3E%3Cpath d='M0,35 L600,35 L618,10 L636,60 L654,14 L672,55 L690,35 L1200,35' fill='none' stroke='rgba(245,158,11,0.4)' stroke-width='2'/%3E%3C/svg%3E") no-repeat top center;
    background-size: 100% 100%, 100% 80px;
}

/* Energy spectrum accent bar at bottom of header */
.header-accent {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #1A2744 0%, #0EA5E9 25%, #22C55E 50%, #F59E0B 75%, #EF4444 100%);
    z-index: 2;
}

.header h1 {
    font-family: 'Franklin Gothic Demi', 'Lexend', 'Rajdhani', 'Arial Narrow', 'Helvetica Neue', Arial, sans-serif;
    font-weight: 700;
    font-size: 3.2rem;
    letter-spacing: 0.5px;
    text-transform: none;
    margin-bottom: 14px;
    position: relative;
    z-index: 3;
    text-shadow: 0 2px 20px rgba(0,0,0,0.5), 0 1px 3px rgba(0,0,0,0.3);
    line-height: 1.15;
}

.header .subtitle {
    font-family: 'Plus Jakarta Sans', 'Helvetica Neue', Arial, sans-serif;
    font-size: 1.15rem;
    font-weight: 400;
    opacity: 0.92;
    max-width: 660px;
    margin: 0 auto;
    letter-spacing: 0.2px;
    position: relative;
    z-index: 3;
    text-shadow: 0 2px 12px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.2);
    line-height: 1.55;
}
```

---

## 5. Bottom Accent Banner (fixed to viewport bottom)

```html
<div class="bottom-banner"></div>
```

```css
.bottom-banner {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 6px;
    background: linear-gradient(90deg, #1A2744 0%, #0EA5E9 25%, #22C55E 50%, #F59E0B 75%, #EF4444 100%);
    z-index: 1000;
    overflow: hidden;
}

.bottom-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 6' preserveAspectRatio='none'%3E%3Cpath d='M0,3 L120,3 L135,0.5 L150,5.5 L165,1 L180,5 L195,3 L400,3 L415,0.5 L430,5.5 L445,1 L460,5 L475,3 L700,3 L715,0.5 L730,5.5 L745,1 L760,5 L775,3 L1000,3 L1015,0.5 L1030,5.5 L1045,1 L1060,5 L1075,3 L1200,3' fill='none' stroke='rgba(255,255,255,0.45)' stroke-width='1.2'/%3E%3C/svg%3E") no-repeat center center;
    background-size: 100% 100%;
    pointer-events: none;
}
```

---

## 6. Main Content Container

```css
.main-content {
    max-width: 1440px;
    margin: 0 auto;
    padding: 32px 44px 64px;
    background: #F3F4F8;
    border-radius: 16px;
}
```

---

## 7. Typography Scale

### Desktop (17px base)

| Element | Size | Weight | Font | Extra |
|---------|------|--------|------|-------|
| h1 (banner) | 3.2rem | 700 | Franklin Gothic Demi | letter-spacing: 0.5px |
| h2 (section) | 1.55-1.6rem | 700 | Barlow Semi Condensed | letter-spacing: 0.02em; color: `--accent` |
| h3 (panel) | 1.15-1.3rem | 600 | Barlow Semi Condensed / Franklin Gothic Demi | letter-spacing: 0.04em |
| Body | 0.92-0.95rem | 400 | Plus Jakarta Sans | line-height: 1.55-1.75 |
| Labels | 0.76-0.88rem | 600-700 | Franklin Gothic Demi | uppercase, letter-spacing: 0.6-1px |
| Subtitle | 1.05-1.15rem | 400 | Plus Jakarta Sans | line-height: 1.55 |

### Responsive scaling

```css
@media (max-width: 768px) {
    html { font-size: 15px; }
    .header h1 { font-size: 1.5rem; }
}

@media (max-width: 480px) {
    html { font-size: 14px; }
    .header h1 { font-size: 1.3rem; }
}
```

---

## 8. Card Styles

```css
/* Standard white card */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 28px;
    box-shadow: 0 4px 6px -4px rgba(0,0,0,0.04);
}

/* Elevated card on hover */
.card:hover {
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
}

/* Dark metric tile */
.metric-tile {
    background: linear-gradient(145deg, #1F3158 0%, #1A2744 100%);
    border-radius: 14px;
    padding: 18px;
    color: #ffffff;
    transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.3s;
}

.metric-tile:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(26,39,68,0.3);
}

/* Frosted glass card */
.glass-card {
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 24px;
}

/* Note/callout box */
.note-box {
    background: linear-gradient(135deg, rgba(245,158,11,0.06) 0%, rgba(239,68,68,0.04) 100%);
    border-left: 4px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 18px 24px;
}

/* Formula / dark code box */
.formula-box {
    background: linear-gradient(145deg, #1F3158 0%, #1A2744 100%);
    border-radius: 10px;
    padding: 18px 24px;
    color: #ffffff;
}
```

---

## 9. Buttons & Controls

```css
/* Select dropdown */
select {
    font-family: 'Franklin Gothic Demi', var(--font-body);
    font-size: 1rem;
    border: 1.5px solid var(--border);
    border-radius: 8px;
    padding: 11px 38px 11px 16px;
    background: #fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%231A2744' stroke-width='2' fill='none'/%3E%3C/svg%3E") no-repeat right 14px center;
    background-size: 12px 8px;
    appearance: none;
    -webkit-appearance: none;
    cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s;
}

select:hover { border-color: var(--navy); }
select:focus {
    outline: none;
    border-color: #1565C0;
    box-shadow: 0 0 0 3px rgba(26, 39, 68, 0.12);
}

/* Toggle pill button */
.toggle-pill {
    font-family: 'Franklin Gothic Demi', var(--font-body);
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    border: 2px solid var(--border);
    border-radius: 9999px;
    padding: 5px 18px;
    background: rgba(255,255,255,0.7);
    cursor: pointer;
    transition: all 0.15s ease;
}

.toggle-pill:hover {
    background: rgba(21, 101, 192, 0.08);
    border-color: #1565C0;
}

.toggle-pill.active {
    background: var(--navy);
    color: #ffffff;
    border-color: var(--navy);
}

/* Action button */
.btn-primary {
    font-family: 'Franklin Gothic Demi', var(--font-body);
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: var(--navy);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-primary:hover {
    background: var(--accent);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
}
```

---

## 10. Navigation / Toolbar

```css
.nav-toolbar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--bg-alt);
    border-bottom: 1px solid var(--border-light);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.nav-link {
    font-family: 'Franklin Gothic Demi', var(--font-body);
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--navy);
    text-decoration: none;
    border-bottom: 1px solid rgba(239,68,68,0.3);
    transition: all 0.15s;
}

.nav-link:hover {
    color: var(--accent);
    border-bottom-color: var(--accent-dark);
}
```

---

## 11. Grid & Layout Patterns

```css
/* 4-column metric row */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
}

@media (max-width: 768px) {
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 480px) {
    .metrics-grid { grid-template-columns: 1fr; }
}

/* Sidebar + main chart layout */
.chart-layout {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 24px;
}

@media (max-width: 900px) {
    .chart-layout { grid-template-columns: 1fr; }
}

/* Auto-fit stat cards */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 18px;
}
```

---

## 12. Key Gradients (quick reference)

| Name | CSS |
|------|-----|
| **Header** | `linear-gradient(135deg, #0F1A2E 0%, #122952 30%, #1565C0 70%, #0D47A1 100%)` |
| **Energy spectrum** | `linear-gradient(90deg, #1A2744 0%, #0EA5E9 25%, #22C55E 50%, #F59E0B 75%, #EF4444 100%)` |
| **Dark tile** | `linear-gradient(145deg, #1F3158 0%, #1A2744 100%)` |
| **Note box** | `linear-gradient(135deg, rgba(245,158,11,0.06) 0%, rgba(239,68,68,0.04) 100%)` |
| **Page background** | `#0F1A2E` with radial glow overlays (see body styles above) |

---

## 13. Box Shadows (elevation system)

```css
/* Level 0 — flat */
box-shadow: none;

/* Level 1 — subtle (cards, containers) */
box-shadow: 0 4px 6px -4px rgba(0,0,0,0.04);

/* Level 2 — medium (main panels) */
box-shadow: 0 4px 24px rgba(0,0,0,0.12);

/* Level 3 — elevated (hover states) */
box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);

/* Level 4 — dramatic (interactive focus) */
box-shadow: 0 8px 24px rgba(26,39,68,0.3);
```

---

## 14. Animations (keyframes)

```css
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}

@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.9); }
    to   { opacity: 1; transform: scale(1); }
}

@keyframes countUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes bounceDown {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(10px); }
}
```

### Common animation applications

```css
/* Staggered card entrance */
.card:nth-child(1) { animation: fadeInUp 0.5s ease both 0.05s; }
.card:nth-child(2) { animation: fadeInUp 0.5s ease both 0.10s; }
.card:nth-child(3) { animation: fadeInUp 0.5s ease both 0.15s; }
.card:nth-child(4) { animation: fadeInUp 0.5s ease both 0.20s; }

/* Smooth scroll reveal */
.scroll-section {
    opacity: 0;
    transition: opacity 0.6s cubic-bezier(0.4, 0, 0.2, 1),
                transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.scroll-section.visible {
    opacity: 1;
    transform: translateY(0);
}
```

---

## 15. Responsive Breakpoints Summary

| Breakpoint | Base font | Layout behavior |
|------------|-----------|-----------------|
| **1440px+** | 17px | Full 4-col metrics, side-by-side charts |
| **1024px** | 17px | Scrollytelling stacks vertically |
| **900px** | 17px | Chart layout goes single-column |
| **768px** | 15px | Metrics go 2-column, padding reduced |
| **480px** | 14px | Everything single-column, compact spacing |

Touch targets: minimum **44px** height/width on `@media (hover: none) and (pointer: coarse)`.

---

## 16. Print Styles

```css
@media print {
    body { background: white; font-size: 11pt; line-height: 1.6; }
    .header {
        background: var(--navy) !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
    .nav-toolbar, .header-accent, .bottom-banner { display: none; }
    .card { break-inside: avoid; }
    table { break-inside: avoid; }
    a { color: var(--text-primary) !important; text-decoration: none !important; }
}
```

---

## 17. Chart.js Color Map (for JS)

```javascript
const CHART_COLORS = {
    cleanFirm:  '#1E3A5F',
    solar:      '#F59E0B',
    wind:       '#22C55E',
    hydro:      '#0EA5E9',
    storage:    '#EF4444',
    gap:        '#D1D5DB',
    outline:    '#374151',
    navy:       '#1A2744',
    accent:     '#EF4444',
};

// 55% opacity variants for overlays/fills
const CHART_COLORS_ALPHA = {
    cleanFirm:  'rgba(30, 58, 95, 0.55)',
    solar:      'rgba(245, 158, 11, 0.55)',
    wind:       'rgba(34, 197, 94, 0.55)',
    hydro:      'rgba(14, 165, 233, 0.55)',
    storage:    'rgba(239, 68, 68, 0.55)',
    gap:        'rgba(209, 213, 219, 0.55)',
};
```

---

## 18. Quick-Start Template

Minimal HTML skeleton using the full style system:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Your Project</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Rajdhani:wght@400;500;600;700&family=Barlow+Semi+Condensed:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Paste the @font-face block from Section 1 */
    /* Paste :root variables from Section 2 */
    /* Paste reset & base from Section 3 */
    /* Paste header styles from Section 4 */
    /* Paste bottom-banner from Section 5 */
    /* Paste main-content from Section 6 */
    /* Add card, button, grid styles as needed from Sections 8-11 */
</style>
</head>
<body>

<div class="header">
    <div class="header-accent"></div>
    <h1>Your Project Title</h1>
    <div class="subtitle">Description or tagline</div>
</div>

<div class="main-content">
    <!-- Your content here -->
</div>

<div class="bottom-banner"></div>

</body>
</html>
```
