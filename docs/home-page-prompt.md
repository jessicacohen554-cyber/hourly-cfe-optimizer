# Executive Summary Page: “What We Choose to Count”

## Claude Code Prompt Sequence — Production-Grade Visual Specs

Four prompts, run in order. Each produces working code for new sections.
The page lives at `dashboard/executive_summary.html`.

**CRITICAL ASSUMPTIONS — do not recreate any of these:**

- The site uses a **PNG circuit board background image**. Reference it the same way existing dashboard pages do — check any page’s CSS for the `background-image` path and replicate that pattern.
- The site has an **existing nav bar / header structure**. Copy it from an existing dashboard page (e.g., `cfe_strategy_assessment.html`) and change only the title/subtitle text.
- The site has an **existing CSS design system** in its dashboard pages — glass-morphism panels, typography (IBM Plex Sans/Mono, DM Serif Display), section numbering, color tokens. Reuse all of it. Do not redeclare what exists.
- **Strategy colors for 1B and 2C:** Open `dashboard/cfe_strategy_assessment.html` and extract the exact CSS colors used for Strategy 1B and Strategy 2C (check CSS variables, class definitions, or inline styles for the strategy cards, chart series, labels, and accent elements). Use those exact same colors throughout this new page. Do NOT invent new colors for 1B and 2C — they must match the existing strategy assessment page precisely.
- All dashboard pages are **single HTML files with inline CSS/JS**. Follow the same pattern.

**Before writing any code in each prompt:**

1. Read `/mnt/skills/public/frontend-design/SKILL.md`
1. Read the full HTML source of `dashboard/cfe_strategy_assessment.html` to extract: the page scaffold, nav/header pattern, CSS variables and design tokens, **exact 1B and 2C strategy colors**, glass-panel styles, section numbering approach, font imports, background treatment, and any scroll animation patterns
1. Also check `dashboard/procurement_deployment.html` for additional visual patterns (jar viz, strategy color assignments)
1. Build the new page reusing those patterns. Only add new CSS/JS for the new visual elements described below.

-----

-----

# PROMPT 1 OF 4

## Page scaffold (from existing patterns) + Hero + “The Fork in the Road” particle animation + “The Signal You Send” panels

-----

### Task

Create `dashboard/executive_summary.html` — the executive summary for The 8,760 Problem project. A scroll-driven visual essay arguing that corporate Scope 2 accounting choices shape IPP investment behavior and determine whether grids end up locked into gas or on a path to firm clean power.

**Scaffold the page by copying the structure from `cfe_strategy_assessment.html`.** Use the same `<head>` setup, font imports, CSS variables, nav/header pattern, circuit board PNG background, glass-panel styles, and section numbering. Change only the page title and subtitle.

**Page title:** “What We Choose to Count”
**Page subtitle:** “How corporate carbon accounting reshapes the grid — for better or worse”

**Color rule for entire page:** Wherever Strategy 1B is referenced visually (borders, glows, labels, particle colors, chart lines, icon strokes), use the **exact 1B color** from `cfe_strategy_assessment.html`. Wherever Strategy 2C is referenced, use the **exact 2C color**. Extract these from the existing page’s CSS. These two colors are the visual identity of the entire argument and must be consistent across both pages.

-----

### Section 1: Hero

Below the nav/header, add a hero content area:

**Lede paragraph** (centered, `max-width: 720px`, same body text style as other pages):

> The GHG Protocol is revising Scope 2 for the first time since 2015. The outcome will redirect hundreds of billions in clean energy capital. One path — consequential netting — optimizes for the cheapest ton of CO₂ displaced anywhere on the map. The other — hourly matching — requires clean energy to meet load every hour in the buyer’s grid. Both reduce emissions. They build fundamentally different grids. This analysis, built on an 8,760-hour optimization model across 7 US ISOs, 540+ market scenarios, and 7 IPP fleet assessments, shows why the distinction matters — and what’s at stake if we get it wrong.

**Hero stats row** — 4 glass-morphism stat cards (same glass-panel style as existing pages). Desktop: 4 columns. Below 768px: 2 columns.

Each card: large number in IBM Plex Mono (site’s teal accent) with label below (uppercase, small, muted). Numbers **count up from 0** on scroll using `requestAnimationFrame` with ease-out over 1.8s. Cards stagger left-to-right, 120ms delay.

Stats:

1. `8,760` — “Hours per year”
1. `7` — “US ISO/RTOs modeled”
1. `540+` — “Market scenarios per IPP” (count to 540, append “+” after)
1. `10` — “Strategies assessed”

-----

### Section 2: “The Fork in the Road” — Animated Particle Divergence

**This is the signature visual of the entire page. Maximum creative effort here.**

Use the site’s existing section numbering style for the section label.
**Heading:** “The Fork in the Road”
**Subhead:** “The same corporate procurement dollar. Two paths. Two very different grids.”

#### Canvas Particle System

Full-width canvas (max 900px), ~700px tall desktop (500px mobile), centered. **Canvas background MUST be transparent** so the circuit board PNG shows through.

**Particle system:**

1. **Source emitter** at top-center. Subtle glowing circle (8px, white, `shadowBlur: 20`). Label above: “CORPORATE PROCUREMENT” in IBM Plex Mono ~12px, muted.
1. **Trunk** — source to fork point (~30% of canvas height). Particles flow straight down, ±4px horizontal jitter. Color: white at 60%.
1. **Fork point** at ~30% height. Subtle horizontal line (60px, 1px, white 15%). Label: “ACCOUNTING FRAMEWORK” in IBM Plex Mono ~11px, muted.
1. **Left branch (1B):** Particles curve left via quadratic bezier to approximately (25% of width, 60% of height). As they enter the branch, **color transitions from white to the 1B strategy color** (extracted from `cfe_strategy_assessment.html`) over ~15 frames. 2.5px radius, `shadowBlur: 6` with the 1B color at 40% as shadowColor.
1. **Right branch (2C):** Mirror, curving right. Color transitions to the **2C strategy color**. Same glow specs in 2C color.
1. **Behavior:** One particle per 120ms. Alternate left/right at fork. Each lives ~3s, fades over last 0.3s. Trail: store last 8 positions, draw with decreasing opacity (0.4→0.02) and radius (2.5→1px). `globalCompositeOperation: 'lighter'` for additive blending at fork overlap. Cap at 40-50 active particles. Pause via IntersectionObserver when off-viewport.
1. **Canvas clear:** Each frame, clear with `fillStyle` using the site’s `--bg-primary` color at ~12% opacity for motion blur. If this obscures the circuit board background, switch to full `clearRect` and rely on the trail array. Circuit board visibility is non-negotiable.

**Outcome zone overlays** (HTML panels positioned over the bottom of the canvas area):

**Left panel (1B):** Glass-panel, border tinted with the **1B color** at ~15%.

- `position: absolute; bottom: 40px; left: 5%; width: 42%`
- **Icon row** (flex, ~28px SVG line-art in **1B color** stroke): Solar panel, Wind turbine, Gas plant. The gas plant icon should **pulse** — `animation` cycling its glow between the 1B color and a rose/red accent every 2s. Make it ~32px (slightly larger) to draw attention.
- **Label:** “1B — Consequential” in IBM Plex Mono, small, uppercase, **1B color**
- **Headline:** “Cheap today. More gas tomorrow.” — 18px, weight 600
- **Text:** “Deploys VRE cross-regionally at lowest cost. IPPs build gas to backstop intermittency. Gas becomes an asset its owners will defend.” — 15px, secondary text
- **Stat:** “Gas remaining:” + `{{GAS_1B_50_95}}` GW in **1B color**, IBM Plex Mono ~28px

**Right panel (2C):** Glass-panel, border tinted with **2C color**.

- Mirrored position: `right: 5%; width: 42%`
- **Icon row** (**2C color** stroke): Solar, Wind, Nuclear (circle + atom dot), Battery, LDES — 5 icons showing portfolio diversity
- **Label:** “2C — Hourly Matching” in **2C color**
- **Headline:** “Costs more now. Builds the grid we need.”
- **Text:** “Matches load every hour in the buyer’s grid. Forces firm clean + storage. Displaces gas directly. Learning curves bend cost down over time.”
- **Stat:** `{{GAS_2C_50_95}}` GW in **2C color**

**Mobile (below 768px):** Panels stack below canvas at full width, 20px gap.

-----

### Section 3: “The Signal You Send” — Three Panels

**Heading:** “The Signal You Send”
**Subhead:** “Corporate procurement isn’t just buying electrons. It’s an investment signal that shapes what IPPs build — and what they’ll fight to keep.”

3-column grid (site’s existing grid pattern), stacks below 1024px. Panels fade up on scroll with 180ms stagger.

**Panel 1 — top accent stripe in 1B color:**

- **SVG icon** (64×64, **1B color** stroke): Solar + wind side-by-side. Hover: `drop-shadow` glow in 1B color.
- **Heading:** “You buy cheap VRE cross-regionally”
- **Text:** “Strategy 1B directs capital to the cheapest MWh available — solar and onshore wind deployed anywhere on the map. IPPs respond rationally: they build VRE where it’s cheapest and gas where it’s needed for reliability. The VRE gets built. The gas gets built too. Both become infrastructure their owners will defend.”

**Panel 2 — top accent stripe in 2C color:**

- **Animated SVG icon** (64×64): Clock face — circle in **2C color** with 24 tick marks. On scroll-trigger (0.5s after visible), ticks animate sequentially clockwise, filling from muted to **2C color** at ~50ms per tick (~1.2s total). CSS `animation-delay` on each tick.
- **Heading:** “You match load every hour in your grid”
- **Text:** “Strategy 2C requires clean generation every hour — including nights, winter evenings, and multi-day low-wind events. IPPs can’t fill these hours with solar and wind alone. The demand signal pulls forward investment in nuclear, CCS, geothermal, and long-duration storage — technologies the grid needs for deep decarbonization but that wholesale markets alone won’t finance.”

**Panel 3 — top accent stripe gradient (1B color → rose/red):**

- **SVG icon** (64×64): Factory silhouette + shield overlay. Pulsing glow cycling rose at 3s. Rose stroke.
- **Heading:** “What gets built gets defended”
- **Text:** “IPPs that build gas will fight to keep it profitable — opposing carbon pricing, clean energy mandates, and retirement schedules that strand their assets. IPPs that build firm clean have aligned incentives: decarbonization policy protects their investment. Procurement strategy doesn’t just shape today’s grid. It shapes the political coalition for or against deeper decarbonization.”

**Link below panels:** “Explore the full IPP climate transition analysis across 540 scenarios and 7 generators →” to `ipp_climate_transition.html`. Teal text link with arrow, hover shifts right 4px.

### Scroll Animation System

Reuse whatever scroll animation approach exists in the site’s other pages. If none exists, implement a single IntersectionObserver (`threshold: 0.2, rootMargin: '0px 0px -60px 0px'`) adding `.visible` to `[data-animate]` elements. Default: `opacity 0→1, translateY(30px)→0, 0.7s cubic-bezier(0.16, 1, 0.3, 1)`. Stagger via `[data-animate-stagger]` with nth-child delays at 150ms.

### Deliverable

Working HTML file with site scaffold + Sections 1-3. Particle fork should be smooth, luminous, unmistakable. Test at 1440, 1024, 768, 375px widths.

-----

-----

# PROMPT 2 OF 4

## “Where You Start” regional bars + “The Coal Wall” animated SVG + “The Last Mile” DAC crossover

-----

### Task

Append Sections 4, 5, and 6 to `dashboard/executive_summary.html`. Do not modify existing sections.

Read `/mnt/skills/public/frontend-design/SKILL.md` before writing code. Reference `cfe_strategy_assessment.html` for 1B/2C colors if needed.

-----

### Section 4: “Where You Start”

**Heading:** “Where You Start”
**Subhead:** “Each grid begins from a different place. The gap between where market forces stop and where net-zero requires — that’s what procurement strategy must close.”

#### “Seven Starting Lines” — Animated stacked bars

Seven horizontal bars (one per ISO), stacked vertically, 16px gap. Each shows generation mix.

Per bar: ISO label (64px, IBM Plex Mono 13px, secondary) | bar track (remaining width, 36px tall, `border-radius: 6px`, flex, background white 2%) | gap indicator (IBM Plex Mono 13px, rose/red accent, e.g. “45% gap”).

**Segment colors** (use teal/amber/white/violet from the site’s existing palette — match the resource color conventions used in `procurement_deployment.html` if applicable):

- Clean: teal accent at ~40%, subtle gradient
- Gas: 1B color at ~35%, gradient
- Coal: white at ~12%
- Other: violet at ~20%

**Animation:** On scroll, segments grow from 0% to target width. CSS transition: `width 1s cubic-bezier(0.16, 1, 0.3, 1)`. Bars stagger 80ms top-to-bottom.

**Data:**

|ISO  |Clean|Gas|Coal|Other|Gap|
|-----|-----|---|----|-----|---|
|CAISO|55%  |40%|0%  |5%   |45%|
|ERCOT|40%  |45%|12% |3%   |60%|
|PJM  |40%  |35%|20% |5%   |60%|
|NYISO|35%  |55%|0%  |10%  |65%|
|NEISO|30%  |55%|0%  |15%  |70%|
|MISO |25%  |35%|35% |5%   |75%|
|SPP  |45%  |25%|25% |5%   |55%|

**Insight block** (glass-panel, full-width, `border-left: 3px solid` 2C color):

> “The market will not deliver net-zero on its own. Across 540 market scenarios per region, clean energy deployment is profitable up to a point — then it stalls. The gap between where economics alone takes each grid and what net-zero requires is what corporate procurement and policy must close. The question: does your procurement strategy help close *your grid’s specific gap* — or does it optimize globally while your grid stays stuck?”

**Three ISO annotations** (compact text cards, colored left borders, stagger 200ms):

1. **ERCOT** (1B-color border): “Already ~40% clean from massive merchant wind and solar. But fully deregulated — zero policy-supported supply (SSS). Hourly matching draws heavily from existing merchant clean, with new-build needed mainly for night and low-wind gaps. Closing the remaining 60% requires firm clean in a market historically hostile to mandates.”
1. **PJM** (2C-color border): “Large SSS from Exelon’s ZEC-supported nuclear fleet (~95 TWh). But 18.8 TWh is locked to Amazon via the Susquehanna PPA. The Virginia data center corridor is the epicenter of corporate clean energy demand — and the grid behind it is still heavily gas-dependent. Strategy choice here directly shapes whether that demand pulls firm clean or more gas.”
1. **NEISO** (rose border): “Supply-constrained, pipeline-constrained, and expensive. Limited natural gas pipeline capacity during winter forces generators to burn oil or LNG at premium prices, adding ~$13/MWh to CCS costs. The hardest grid to decarbonize — and where strategy choice matters most, because consequential sends capital elsewhere while this grid stays stuck.”

**Link:** “Explore regional pathways →” to `gen_regional_pathways.html`

-----

### Section 5: “The Coal Wall”

**Heading:** “The Coal Wall”
**Subhead:** “Consequential’s cost advantage has an expiration date. It’s the moment the last coal plant retires from the dispatch stack.”

#### Animated vertical cross-section (inline SVG)

`max-width: 600px`, centered, ~450px tall. `viewBox="0 0 600 450"`.

**Stack (bottom-up):**

- **Coal layer:** rect y=300→450 (150px). Fill: white 8%. Labels: “COAL” left, “~2,200 lb CO₂/MWh” right. IBM Plex Mono 11px, muted.
- **Gas layer:** rect y=100→300 (200px). Fill: 1B-color at 8%. Labels: “GAS”, “~850 lb CO₂/MWh”.
- **Coal wall boundary** at y=300: dashed line (`stroke-dasharray: 6 4`, rose 50%, `stroke-width: 1.5`). Label: “THE COAL WALL” in rose, IBM Plex Mono 11px.

**Sweep line:** Horizontal line rising from bottom to top. 2C-color, `stroke-width: 2`, `filter: drop-shadow(0 0 10px [2C color at 50%])`. Above the line, fossil layers go near-transparent (displaced). Implementation: animated `<clipPath>` or `<mask>` rectangle with transitioning `y`.

**Three-phase animation (scroll-triggered):**

- **Phase 1 (0→1.5s):** Line rises y=450→300 (coal). Counter near line: “Displacement: ~2,200 lb/MWh” in 2C color.
- **Phase 2 (1.5→2.3s):** Line **pauses** at y=300. Coal wall label pulses (scale 1→1.08→1, 0.4s). Annotation fades in: “All economically displaceable coal is gone. In most ISOs, this happens around ~70% clean. Beyond this point, every clean MWh only displaces gas — cutting the carbon benefit nearly in half.”
- **Phase 3 (2.3→4s):** Line continues through gas, slower (`ease-in`). Counter shifts to “~850 lb/MWh”, color blends toward 1B color.

**Key finding card** (glass-panel, `max-width: 720px`):

> “Both strategies face escalating costs past the coal wall — but for different reasons. Consequential (1B) runs out of cheap coal to displace; every additional MWh yields half the carbon benefit against gas. Hourly matching (2C) costs more because firm clean power and storage for the hardest hours is inherently expensive. The critical difference: 2C’s cost escalation drives deployment of technologies the grid actually needs — nuclear, CCS, LDES — and those costs come down via Wright’s Law. 1B’s escalation buys diminishing marginal displacement of the same gas fleet.”

**Link:** “See consequential accounting analysis →” to `consequential_accounting.html`

-----

### Section 6: “The Last Mile”

**Heading:** “The Last Mile”
**Subhead:** “At what point does the last percent of grid decarbonization cost more than just removing carbon directly?”

#### Two-curve crossover chart (inline SVG)

`max-width: 700px`, centered, ~350px tall.

- **X-axis:** CFE target 80%→100%. Ticks at 80, 85, 90, 95, 97.5, 99, 99.9. IBM Plex Mono 11px.
- **Y-axis:** Marginal cost ($/tCO₂). IBM Plex Mono 11px.
- **Grid decarb curve:** Starts low at 80%, moderate rise to 95%, hockey stick 97.5→100%. `<path>`, `stroke-width: 2.5`, no fill. Color: 2C color for 80-95% range, transitioning toward rose at 95%+ (use `<linearGradient>` along path, or overlapping paths with clip).
- **DAC line:** Roughly horizontal, dashed, violet, `stroke-dasharray: 8 4`, `stroke-width: 1.5`. Intersects grid curve ~95-99%. Label: “Direct Air Capture reference”.
- **Crossover marker:** Circle (6px, rose fill, white stroke) where curves meet. Vertical dashed line to x-axis. Label: “~96-98% CFE” (placeholder).
- **Overshoot zone:** Below grid curve, above DAC line, post-crossover: rose fill at 8%.

**Animation:** Curves draw via `stroke-dashoffset`, 2s, ease-out. Crossover marker fades in 0.5s after.

**Text below:**

> “The optimal CFE target isn’t necessarily 100%. It’s the point where marginal grid decarbonization exceeds marginal carbon removal. This crossover shifts by region — earlier in supply-constrained grids (NEISO: ~95%), later in resource-rich ones (ERCOT: ~98%).”

> “What matters for 1B vs. 2C: the costs 2C incurs *below* the crossover — building firm clean and storage in the 85–97% range — push the crossover point higher for everyone by driving learning curves down. 1B avoids those costs but contributes nothing to moving the frontier.”

**Link:** “Regional pathways and DAC backstop →” to `gen_regional_pathways.html`

### Deliverable

Sections 4-6 appended. Coal wall animation is the dramatic centerpiece — the pause at the boundary is the beat.

-----

-----

# PROMPT 3 OF 4

## IPP demand response + Learning curves + Nuclear swing variable

-----

### Task

Append Sections 7, 8, and 9 to `dashboard/executive_summary.html`. Don’t modify existing sections.

Read `/mnt/skills/public/frontend-design/SKILL.md` before writing code. Use 1B/2C colors from `cfe_strategy_assessment.html`.

-----

### Section 7: “The IPP Response”

**Heading:** “The IPP Response”
**Subhead:** “540 scenarios. 7 generators. One consistent finding: demand signals shape what gets built.”

**Opening text** (`max-width: 720px`):

> “Independent power producers are not passive recipients of policy. They are strategic actors who invest based on revenue signals, defend assets they’ve built, and lobby for market rules that protect their portfolio. The question isn’t just ‘what gets built’ — it’s what incentive structure each procurement framework creates for the companies that actually build and operate the grid at scale.”

#### “Two Futures for the Same Fleet” — Constellation case study

Two glass-panels side-by-side (2-col grid, `gap: 32px`, stacks on mobile). `min-height: 400px` each.

**Fleet icon grid** in each panel: 16 small circles (20px, 8px gap) — initial state both panels: 12 in **2C color** (nuclear), 3 in **1B color** (gas), 1 in blue (renewable).

**Left — “Low PPA Depth” (1B-aligned):**

- 1B-color top accent stripe (3px)
- Label: “LOW PPA DEPTH” — IBM Plex Mono, 1B color, uppercase
- Sublabel: “Weak demand for hourly clean → revenue uncertainty”
- **Scroll animation (2s, starts 0.5s after visible):**
  - 3 nuclear circles fade: background → white 10%, opacity → 0.4
  - 1 gas circle grows `scale(1.15)` with pulsing 1B-color glow
- **Counter:** `34.3 Mt CO₂` in **1B color**, IBM Plex Mono, large. Label: “Annual fleet emissions”
- **Caption:** “Without strong PPA demand for 24/7 clean, merchant nuclear faces revenue uncertainty as state support programs expire (NJ ZEC 2026, IL CMC 2028, NY ZEC 2030). Plants stay online under 45U PTC but without fleet reinvestment certainty. New gas fills reliability gaps.”

**Right — “High PPA Depth” (2C-aligned):**

- 2C-color top accent stripe
- Label: “HIGH PPA DEPTH” — 2C color
- Sublabel: “Strong demand for hourly clean → revenue certainty”
- **Animation:**
  - All 12 nuclear circles brighten with `box-shadow: 0 0 8px [2C color at 40%]`, `scale(1.05)`
  - 2 gas circles: `scale(0.7), opacity: 0.3`
  - 1 new icon appears (2C color, hexagonal to distinguish from existing nuclear): scales 0→1
- **Counter:** `9.8 Mt CO₂` in **2C color**, large
- **Caption:** “Corporate demand for hourly-matched clean creates long-term PPA revenue ($10–40/MWh premium for dispatchable clean over VRE). This underwrites nuclear fleet maintenance, enables restarts like Crane/Three Mile Island, and finances new firm clean. Microsoft’s 20-year Constellation PPA alone: ~14 TWh/year of 24/7 carbon-free generation.”

**3.5× stat callout** centered below panels. Glass-panel, `max-width: 560px, text-align: center, padding: 40px`. Should be one of the most visually striking elements on the page — give it a subtle rose-tinted border glow (`box-shadow: 0 0 40px -10px rgba(239,68,68,0.15)`):

- “3.5×” — rose accent, IBM Plex Mono, `font-size: clamp(48px, 6vw, 64px)`
- “emissions differential from the same fleet” — 18px, primary text
- “driven entirely by demand signal” — 15px, secondary text

**Political economy text** (`max-width: 720px`):

> “IPPs that have built gas have a rational incentive to oppose policies that strand those assets — carbon pricing, accelerated retirement, stricter emissions standards. IPPs whose revenue depends on clean generation have the opposite incentive: decarbonization policy protects their investment. Strategy 1B’s demand signals are more likely to create conditions for new gas development — it doesn’t penalize temporal mismatch, so gas backup is an acceptable complement to VRE-heavy portfolios. Strategy 2C’s hourly constraint makes gas backup a cost, not a complement — driving IPPs toward firm clean resources that can actually match load patterns.”

**Links:** “IPP climate transition →” `ipp_climate_transition.html` | “Constellation deep dive →” `ipp_constellation.html`

-----

### Section 8: “The Learning Curve Dividend”

**Heading:** “The Learning Curve Dividend”
**Subhead:** “Clean firm technologies are still on steep learning curves. Which strategy reaches critical mass fastest?”

**Opening text:**

> “Wright’s Law: costs decline predictably with cumulative deployment. Solar and wind are far down their curves — mature, cheap. Nuclear, CCS, and long-duration storage are still in the steep part where each doubling drives 15–25% cost reduction. The strategy that reaches critical mass first pulls costs down for the entire grid.”

#### Two diverging curves (inline SVG)

`max-width: 700px`, centered, ~320px tall.

- X: “Cumulative firm clean deployment” (abstract). Y: “Cost premium over fossil ($/MWh)”.
- **2C curve (2C color):** Starts high, drops steeply, crosses zero. `stroke-width: 2.5`. At zero crossing: **pulsing circle** (scale 1→1.3→1 every 2s with glow). Label: “Critical mass: learning curves outpace premiums”
- **1B curve (1B color):** Same start, much shallower. May not cross zero within chart range. `stroke-width: 2.5`.
- **Gap shading** between curves: 2C color at 6%. Label: “Learning curve dividend” italic 13px.
- **Zero line:** Horizontal, white 10%. “Cost parity with fossil”.

**Animation:** Both draw via `stroke-dashoffset`, 2.5s. Critical mass marker fades in 0.5s after teal curve arrives.

**Three cards below** (3-col grid, stacks mobile):

1. (violet border) **“IRA Accelerates the Timeline”** — “Federal credits cut clean LCOE 30–40%. Section 45Y PTC (~$28/MWh), 48E ITC (up to 50%), 45Q ($85/tonne CCS) compress the gap between VRE-only and diversified portfolios.”
1. (2C-color border) **“Hyperscale Demand Is Already 2C-Aligned”** — “Microsoft’s PPA restarted Crane nuclear (1.8 GW). Google contracts for Fervo geothermal and Kairos nuclear. Amazon invests in SMRs. These buyers are pulling the 2C learning curve down now.”
1. (blue border) **“Policy Is Converging on Hourly”** — “GHG Protocol Scope 2 revision (~2027) proposes hourly matching. SBTi Power Sector v2 moves toward hourly. 14 states have 100% clean laws. EU rules require temporal correlation. 2C’s premium is early-mover positioning.”

**Links:** “Policy landscape →” `gen_policy_conditions.html` | “Strategy deep dive →” `strategy_deep_dive.html`

-----

### Section 9: “The Swing Variable” — Nuclear

**Heading:** “The Swing Variable”
**Subhead:** “Existing nuclear provides roughly half of US clean electricity. Its fate depends on procurement framework.”

#### “The Fulcrum” (inline SVG)

`max-width: 500px`, centered, ~220px tall.

- Fulcrum triangle bottom-center. Beam: horizontal, `stroke-width: 3`, white 30%.
- Left end: nuclear plant silhouette (2C-color stroke). Label: “Nuclear fleet”.
- Right end: stacked revenue rectangles — “ZEC/CMC” (violet dim), “45U PTC” (blue dim), “Capacity” (white 6%), **“Corporate PPAs”** (2C color, taller/brighter, glowing border — the variable that swings it).

**Animation:** On scroll, beam tilts `rotate(-3deg)`, `1.5s ease-out`.

**Three stat cards:**

1. `{{NUCLEAR_COST_INCREASE}}` — “Cost increase if nuclear retires” (rose)
1. `{{NUCLEAR_CO2_LOST}}` — “CO₂ displacement lost” (1B color)
1. `{{NUCLEAR_GAS_INCREASE}}` — “Additional gas on grid” (1B color)

**Roll-off timeline** — horizontal, three rose dots:

- 2026: “NJ ZEC expires (−15 TWh PJM)”
- 2028: “IL CMC expires (−50 TWh PJM, −15 TWh MISO)”
- 2030: “NY ZEC Tier 3 expires (−42 TWh NYISO)”

**Text:** “These plants don’t retire — they move from SSS to merchant under 45U PTC. But without corporate PPA demand for long-term revenue certainty, fleet reinvestment tilts toward deferral. The procurement framework determines whether existing nuclear is an asset or a liability.”

### Deliverable

Sections 7-9 appended. The 3.5× callout and Two Futures animation are emotional peaks. Nuclear section: concise, supporting argument.

-----

-----

# PROMPT 4 OF 4

## Synthesis essay + Navigation grid + Polish pass

-----

### Task

Append final Sections 10-11 to `dashboard/executive_summary.html`. Full polish pass.

Read `/mnt/skills/public/frontend-design/SKILL.md` before writing code.

-----

### Section 10: “Rowing in the Same Direction” — Synthesis

**Special treatment.** Pure prose, no charts. The argument lands through typography.

**Background:** Subtle gradient overlay on this section only: `linear-gradient(180deg, transparent, [2C color at 2%] 30%, [2C color at 2%] 70%, transparent)`.

**Heading:** “Rowing in the Same Direction” — DM Serif Display at `clamp(32px, 4.5vw, 48px)`. This + page title are the ONLY display font uses.

**Body:** `max-width: 760px`, `font-size: 18px`, `line-height: 1.78`, `paragraph margin-bottom: 24px`. Should read like a typeset editorial.

**Text:**

> Single-buyer PPAs for VRE or unbundled RECs were the right tools for the first phase of the energy transition: displacing coal, building the renewable supply chain, proving that clean energy could compete on cost. That phase succeeded. Most of the cheap coal displacement is done or underway.

> The next phase is qualitatively different. Getting grids from 60–70% clean to net-zero means solving temporal dependencies — nights, winter evenings, multi-day low-wind events — and spatial dependencies — each grid needs its own portfolio of firm clean power and storage tuned to its fossil fleet, weather patterns, and demand profile. These are not problems that cross-regional VRE deployment can solve. They are not problems that annual accounting can even see.

> Consequential accounting is individually rational. Each buyer directing their marginal dollar to the cheapest ton of CO₂ displacement is defensible in isolation. But in aggregate, consequential concentrates capital in grids with the cheapest abatement — coal-heavy MISO and SPP — while gas-dependent grids like NEISO, NYISO, and CAISO receive less. It creates no structural demand signal for firm clean power. It leaves gas on the grid. And the owners of that gas — the IPPs who built it because the demand signal said “VRE plus backup” — will rationally fight to keep it running.

> Hourly matching is more expensive today. That is a real cost, not a footnote. But it builds coherence by construction. When procurement must match load hour-by-hour within the buyer’s grid, investment flows to the resources that specific grid actually needs. It creates demand signals for firm clean power and storage. It pulls forward the learning curves the entire grid needs. It aligns individual procurement with collective grid transformation. And it positions early adopters on the right side of a policy trajectory — GHG Protocol, SBTi, state mandates, EU rules — that is converging on hourly matching as the standard.

> The question isn’t which strategy is cheaper today. It’s which strategy builds the grid we need by 2040.

> Across this analysis — 8,760 hours at full resolution, 7 ISO/RTOs, 10 procurement strategies, 540+ scenarios per IPP, 7 fleet assessments — the finding is directionally consistent. Hourly matching drives deeper gas displacement, reaches learning-curve critical mass faster, creates revenue certainty for firm clean investment, and aligns with the regulatory trajectory. Consequential netting is cheaper per MWh but leaves more gas on the grid, concentrates investment geographically, and does not create structural demand signals for the technologies the next phase requires.

> Deep grid decarbonization requires rowing in the same direction — a shared understanding of where each grid can and should end up, and procurement signals coherent with that destination. The era of each buyer picking off the cheapest ton from anywhere on the map was sufficient when the challenge was displacing coal. It is not sufficient for the era of firm clean power, long-duration storage, and hourly temporal matching. That era is already here. The GHG Protocol revision will determine whether corporate accounting catches up to the physics — or stays a decade behind it.

-----

### Section 11: “Explore the Analysis” — Navigation Grid

**Heading:** “Explore the Analysis”
**Subhead:** “Each page explores a different dimension of the question.”

4-column card grid (site’s existing pattern). 2-col below 1024px. 1-col below 600px. Cards: glass-panels, hover lift `translateY(-4px)`, entire card is `<a>`. Stagger 100ms.

Each card: 8px accent dot (colored circle), title (16px, weight 600), description (13px, secondary), arrow (teal “→”).

1. 2C-color dot. “The Full Comparison” → `cfe_strategy_assessment.html` — “1B vs. 2C across every metric and participation level.”
1. Violet dot. “10 Strategies” → `consequential_accounting.html` — “Three accounting families and the tradeoffs.”
1. 1B-color dot. “Where Capital Flows” → `procurement_deployment.html` — “How each strategy reshapes each ISO’s mix.”
1. Blue dot. “Strategy Deep Dive” → `strategy_deep_dive.html` — “Gas displacement, learning curves, decarb matrices.”
1. 2C-color dot. “Regional Pathways” → `gen_regional_pathways.html` — “Where the market stops and the gap to net-zero.”
1. Rose dot. “Policy Landscape” → `gen_policy_conditions.html` — “IRA, state mandates, corporate demand pull.”
1. 1B-color dot. “IPP Transition” → `ipp_climate_transition.html` — “540 scenarios across 7 generators.”
1. Violet dot. “The Model” → `optimizer_methodology.html` — “8,760-hour resolution. Full methodology.”

-----

### Footer

Use the same footer pattern as other pages on the site.

### Meta Tags

```html
<title>What We Choose to Count — The 8,760 Problem</title>
<meta name="description" content="How corporate carbon accounting reshapes the grid. Hourly vs. consequential clean energy procurement across 7 US ISOs, 540+ market scenarios, and 7 IPP fleet assessments.">
<meta property="og:title" content="What We Choose to Count — The 8,760 Problem">
<meta property="og:description" content="Corporate Scope 2 accounting choices shape IPP investment and determine whether grids lock into gas or transition to firm clean power.">
<meta property="og:type" content="article">
```

### Polish Pass

1. **1B/2C color consistency:** Verify every visual element uses the exact colors from `cfe_strategy_assessment.html` — no drifted approximations.
1. **Scroll timing:** All observers at `threshold: 0.2`, `rootMargin: '0px 0px -60px 0px'`. No animation >1s except coal wall sweep and particle fork.
1. **Typography:** DM Serif Display ONLY on page title + Section 10 heading. Everything else IBM Plex Sans/Mono.
1. **Responsive:** Test 1440, 1024, 768, 375px. Multi-col layouts stack. Fork panels go below canvas on mobile.
1. **Performance:** Canvas particles capped 50, paused off-viewport. CSS animations use `transform`/`opacity` only. SVGs use `stroke-dashoffset`.
1. **Canvas transparency:** Circuit board PNG must show through. Test the clear approach.
1. **Links:** All relative paths from `dashboard/`. Verify filenames match real pages.

### Deliverable

Complete page. Single HTML file. Polished editorial data essay — not a dashboard, not a blog post. Something you’d send to a GHG Protocol working group member.