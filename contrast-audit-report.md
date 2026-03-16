# Contrast Audit Report

Generated: 2026-03-16T08:40:10.417Z

## Summary

| Metric | Count |
|--------|-------|
| Total elements | 8950 |
| Passing | 8892 |
| **Failing** | **58** |
| Critical | 43 |
| Major | 3 |
| Minor | 12 |

## dashboard/GHG-accounting-overview.html

**5 failures**

### [CRITICAL] section.hero-overview.section-light > div.hero-inner > div.comparison-visual > div.compare-box.hourly > p

- **Text**: "Proposed Scope 2 Revision✓ Hourly granularity✓ Deliverability regions✓ Drives fi"
- **Color**: #56C0F0 on #FFFFFF
- **Ratio**: 2.06:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #107AAA or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] section.hero-overview.section-light > div.hero-inner > div.comparison-visual > div.compare-box.conseq > p

- **Text**: "Impact-Based Framework✓ Cheapest $/tCO₂✓ Cross-regional optimization✗ No tempora"
- **Color**: #64D68E on #FFFFFF
- **Ratio**: 1.81:1 (need 4.5:1)
- **Font**: 12px, weight 400
- **Fix**: Change text to #14863E or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] body > section.concept-section.section-light > div.concept-grid > div.concept-card > h3

- **Text**: "1. Annual Attribution Current"
- **Color**: #B91C1C on #2C3344
- **Ratio**: 1.96:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #E14444 or adjust for 3:1 on bg #2C3344

### [CRITICAL] body > section.concept-section.section-light > div.concept-grid > div.concept-card > h3

- **Text**: "2. Hourly Attribution Proposed"
- **Color**: #0369A1 on #2C3344
- **Ratio**: 2.13:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #2187BF or adjust for 3:1 on bg #2C3344

### [CRITICAL] body > section.concept-section.section-light > div.concept-grid > div.concept-card > h3

- **Text**: "3. Consequential / Impact Impact-Based"
- **Color**: #166534 on #2C3344
- **Ratio**: 1.77:1 (need 3:1)
- **Font**: 15px, weight 700
- **Fix**: Change text to #3E8D5C or adjust for 3:1 on bg #2C3344

## dashboard/about.html

**6 failures**

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "01"
- **Color**: #94A3B8 on #FFFFFF
- **Ratio**: 2.56:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #8594A9 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "02"
- **Color**: #94A3B8 on #FFFFFF
- **Ratio**: 2.56:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #8594A9 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "03"
- **Color**: #94A3B8 on #FFFFFF
- **Ratio**: 2.56:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #8594A9 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "04"
- **Color**: #94A3B8 on #FFFFFF
- **Ratio**: 2.56:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #8594A9 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "05"
- **Color**: #94A3B8 on #FFFFFF
- **Ratio**: 2.56:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #8594A9 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] section#sec-layers > div.story-content > div.layer-stack > div.layer-card > span.layer-num

- **Text**: "06"
- **Color**: #94A3B8 on #FFFFFF
- **Ratio**: 2.56:1 (need 3:1)
- **Font**: 40px, weight 800
- **Fix**: Change text to #8594A9 or adjust for 3:1 on bg #FFFFFF

## dashboard/archive/ipp_smartargets.html

**3 failures**

### [CRITICAL] html > body > section.hero > h1

- **Text**: "What Do Regional Targets Mean for Your Fleet?"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 3:1)
- **Font**: 35px, weight 800
- **Fix**: Change text to #919191 or adjust for 3:1 on bg #FFFFFF

### [CRITICAL] html > body > section.hero > p

- **Text**: "540 parametric scenarios reveal the range of outcomes for each IPP fleet. P10/P5"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 17px, weight 400
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

### [CRITICAL] html > body > div.global-controls > button#passiveBtn

- **Text**: "Passive"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

## dashboard/archive/pipeline_map.html

**2 failures**

### [CRITICAL] html > body > section.pages-section.section-dark > div.pages-inner.section-dark-inner > h2.section-heading

- **Text**: "Dashboard Pages & Data Dependencies"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 26px, weight 800
- **Fix**: Change text to #646464 or adjust for 3:1 on bg #0B1120

### [CRITICAL] html > body > section.pages-section.section-dark > div.pages-inner.section-dark-inner > p.section-subheading

- **Text**: "Each page consumes one or more JS data files generated by the pipeline. Hover fo"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #7D7D7D or adjust for 4.5:1 on bg #0B1120

## dashboard/archive/procurement_comparison.html

**2 failures**

### [CRITICAL] html > body > section.hero-opening > div.hero-counter > span#counterPct

- **Text**: "13"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 56px, weight 900
- **Fix**: Change text to #646464 or adjust for 3:1 on bg #0B1120

### [CRITICAL] html > body > section#findingsHero > div.findings-inner > div.findings-label

- **Text**: "Key Findings"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #7D7D7D or adjust for 4.5:1 on bg #0B1120

## dashboard/archive/scenario_comparison.html

**25 failures**

### [CRITICAL] html > body > section.content-section.story-section > div.card > span.story-badge

- **Text**: "75–90% • The Inflection Point"
- **Color**: #D97706 on #FEF5E7
- **Ratio**: 2.95:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #B65400 or adjust for 4.5:1 on bg #FEF5E7

### [CRITICAL] html > body > section.content-section.section-light > h2.section-title

- **Text**: "What Gets Built: Strategy 1B vs Strategy 2C"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #646464 or adjust for 3:1 on bg #0B1120

### [CRITICAL] html > body > section.content-section.section-light > p.section-subtitle

- **Text**: "Resource mix in TWh at each threshold.
Dashed line = CFE target — solid bars = u"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #7D7D7D or adjust for 4.5:1 on bg #0B1120

### [CRITICAL] html > body > section.content-section.section-light > p.section-subtitle > strong

- **Text**: "Dashed line = CFE target"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #646464 or adjust for 3:1 on bg #0B1120

### [CRITICAL] body > section.content-section.section-light > div.chart-pair > div.chart-box > span.scenario-tag.tag-a

- **Text**: "Strategy 1B"
- **Color**: #FFFFFF on #B2AFE7
- **Ratio**: 2.07:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #464646 or adjust for 4.5:1 on bg #B2AFE7

### [CRITICAL] body > section.content-section.section-light > div.chart-pair > div.chart-box > span.scenario-tag.tag-b

- **Text**: "Strategy 2C"
- **Color**: #FFFFFF on #99C9D6
- **Ratio**: 1.8:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #4B4B4B or adjust for 4.5:1 on bg #99C9D6

### [CRITICAL] html > body > section.content-section.section-light > h2.section-title

- **Text**: "System Cost & Marginal Abatement Cost"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #646464 or adjust for 3:1 on bg #0B1120

### [CRITICAL] html > body > section.content-section.section-light > p.section-subtitle

- **Text**: "Effective $/MWh system cost (left) and stepwise $/tCO₂ (right) at each SBTi mile"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #7D7D7D or adjust for 4.5:1 on bg #0B1120

### [CRITICAL] html > body > section.content-section.section-light > h2.section-title

- **Text**: "Clean Firm & Gas Capacity"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #646464 or adjust for 3:1 on bg #0B1120

### [CRITICAL] html > body > section.content-section.section-light > p.section-subtitle

- **Text**: "Clean firm deployment (left) and gas backup capacity (right) at each milestone."
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #7D7D7D or adjust for 4.5:1 on bg #0B1120

### [CRITICAL] html > body > section.content-section.section-light > h2.section-title

- **Text**: "Sequential Deployment Queue"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 3:1)
- **Font**: 16px, weight 700
- **Fix**: Change text to #646464 or adjust for 3:1 on bg #0B1120

### [CRITICAL] html > body > section.content-section.section-light > p.section-subtitle

- **Text**: "Cross-regional deployment order from Step 5D MAC queue, sorted by cheapest margi"
- **Color**: #000000 on #0B1120
- **Ratio**: 1.12:1 (need 4.5:1)
- **Font**: 16px, weight 400
- **Fix**: Change text to #7D7D7D or adjust for 4.5:1 on bg #0B1120

### [CRITICAL] html > body > section.content-section.section-light > div#macQueueAccordion > span.scenario-tag.tag-a

- **Text**: "Strategy 1B — Sequential Deployment Queue"
- **Color**: #FFFFFF on #B2AFE7
- **Ratio**: 2.07:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #464646 or adjust for 4.5:1 on bg #B2AFE7

### [CRITICAL] div > table > thead > tr > th

- **Text**: "Toggle"
- **Color**: #060912 on #0B1120
- **Ratio**: 1.06:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #797C85 or adjust for 4.5:1 on bg #0B1120

### [CRITICAL] div > table > thead > tr > th

- **Text**: "Strategy 1B"
- **Color**: #4F46E5 on #0B1120
- **Ratio**: 2.99:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #7269FF or adjust for 4.5:1 on bg #0B1120

### [MAJOR] body > section.hero > div.hero-chart-wrap > p.chart-note > span

- **Text**: "Strategy 2C"
- **Color**: #0891B2 on #ECECEE
- **Ratio**: 3.13:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #007394 or adjust for 4.5:1 on bg #ECECEE

### [MAJOR] div.section-dark-inner > div.scenario-cards > div.scenario-card > p > a

- **Text**: "See Strategy 2C deep dive →"
- **Color**: #158FC7 on #303541
- **Ratio**: 3.38:1 (need 4.5:1)
- **Font**: 14px, weight 400
- **Fix**: Change text to #2EA8E0 or adjust for 4.5:1 on bg #303541

### [MINOR] div > table > tbody > tr > td

- **Text**: "Low (NOAK)"
- **Color**: #15803D on #0B1120
- **Ratio**: 3.75:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #248F4C or adjust for 4.5:1 on bg #0B1120

### [MINOR] div > table > tbody > tr > td

- **Text**: "High (FOAK)"
- **Color**: #DC2626 on #0B1120
- **Ratio**: 3.9:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #EB3535 or adjust for 4.5:1 on bg #0B1120

### [MINOR] div > table > tbody > tr > td

- **Text**: "Low (NOAK)"
- **Color**: #15803D on #0B1120
- **Ratio**: 3.75:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #248F4C or adjust for 4.5:1 on bg #0B1120

### [MINOR] div > table > tbody > tr > td

- **Text**: "High (FOAK)"
- **Color**: #DC2626 on #0B1120
- **Ratio**: 3.9:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #EB3535 or adjust for 4.5:1 on bg #0B1120

### [MINOR] div > table > tbody > tr > td

- **Text**: "Low (NOAK)"
- **Color**: #15803D on #0B1120
- **Ratio**: 3.75:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #248F4C or adjust for 4.5:1 on bg #0B1120

### [MINOR] div > table > tbody > tr > td

- **Text**: "High (FOAK)"
- **Color**: #DC2626 on #0B1120
- **Ratio**: 3.9:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #EB3535 or adjust for 4.5:1 on bg #0B1120

### [MINOR] div > table > tbody > tr > td

- **Text**: "Low (NOAK)"
- **Color**: #15803D on #0B1120
- **Ratio**: 3.75:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #248F4C or adjust for 4.5:1 on bg #0B1120

### [MINOR] div > table > tbody > tr > td

- **Text**: "High (FOAK)"
- **Color**: #DC2626 on #0B1120
- **Ratio**: 3.9:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #EB3535 or adjust for 4.5:1 on bg #0B1120

## dashboard/clean_firm_case.html

**1 failures**

### [CRITICAL] section.concept-section.section-light > div.concept-inner > div.concept-chart-col > div.chart-panel > div.chart-title

- **Text**: "Gas Backup Cost vs Nuclear LCOE at 90% Clean — All ISOs"
- **Color**: #FFFFFF on #FFFFFF
- **Ratio**: 1:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #737373 or adjust for 4.5:1 on bg #FFFFFF

## dashboard/gen_target_setting.html

**1 failures**

### [CRITICAL] div.main-content > section.story-section.fade-in > div.story-content > div.chart-panel > p

- **Text**: "SBTi pathway from Sectoral Decarbonization Approach. SMARTargets AT milestones f"
- **Color**: #566370 on #2C3344
- **Ratio**: 2.06:1 (need 4.5:1)
- **Font**: 13px, weight 400
- **Fix**: Change text to #929FAC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_aes.html

**1 failures**

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_climate_transition.html

**1 failures**

### [CRITICAL] html > body > div.content-section.panel-navy > h2.section-title

- **Text**: "Cross-Company Comparative Analysis"
- **Color**: #1A2744 on #0F172A
- **Ratio**: 1.21:1 (need 3:1)
- **Font**: 24px, weight 700
- **Fix**: Change text to #5B6885 or adjust for 3:1 on bg #0F172A

## dashboard/ipp_constellation.html

**1 failures**

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_nextera.html

**1 failures**

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_nrg.html

**1 failures**

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_pseg.html

**1 failures**

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_talen.html

**1 failures**

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/ipp_vistra.html

**1 failures**

### [CRITICAL] main > section#strategic > div.grid-2col > div.card > h3

- **Text**: "Robustness Scorecard"
- **Color**: #1A2744 on #2C3344
- **Ratio**: 1.17:1 (need 4.5:1)
- **Font**: 15px, weight 600
- **Fix**: Change text to #929FBC or adjust for 4.5:1 on bg #2C3344

## dashboard/abatement_dashboard.html

**2 failures**

### [MAJOR] section.deepdive-section.section-light > div.deepdive-content > div.deepdive-narrative-col > div#deepdiveNarrative > span#ddNarrativeTag

- **Text**: "ERCOT"
- **Color**: #15803D on #11282E
- **Ratio**: 3.07:1 (need 4.5:1)
- **Font**: 12px, weight 700
- **Fix**: Change text to #339E5B or adjust for 4.5:1 on bg #11282E

### [MINOR] body > section.hero-overview > div.hero-findings-row > div.hero-narrative-block > span.finding-highlight

- **Text**: "Grid wins to 69–98% (region-dependent) · SPP & MISO: deepest grid advantage · NE"
- **Color**: #15803D on #E7F5EF
- **Ratio**: 4.46:1 (need 4.5:1)
- **Font**: 14px, weight 700
- **Fix**: Change text to #107B38 or adjust for 4.5:1 on bg #E7F5EF

## dashboard/archive/grid_animation.html

**1 failures**

### [MINOR] div.grid-viz-wrapper > div.scenario-intro > div.scenario-cards-row > div.scenario-pick.selected > span.badge.badge-danger

- **Text**: "Feb 10–20, 2021"
- **Color**: #DC2626 on #FDECEC
- **Ratio**: 4.22:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #D21C1C or adjust for 4.5:1 on bg #FDECEC

## dashboard/ref-nuclear-retirement.html

**1 failures**

### [MINOR] body > div.article-wrapper > div#heroPanel > div#heroInsight > div.insight-label

- **Text**: "Cross-Model Synthesis"
- **Color**: #93C5FD on #48518C
- **Ratio**: 4.14:1 (need 4.5:1)
- **Font**: 11px, weight 600
- **Fix**: Change text to #9DCFFF or adjust for 4.5:1 on bg #48518C

## dashboard/typography-mockup.html

**1 failures**

### [MINOR] body > div.mockup-grid > div.mockup-section > div.mockup-label > span.current-label

- **Text**: "Current"
- **Color**: #EF4444 on #322131
- **Ratio**: 3.98:1 (need 4.5:1)
- **Font**: 12px, weight 600
- **Fix**: Change text to #FE5353 or adjust for 4.5:1 on bg #322131

