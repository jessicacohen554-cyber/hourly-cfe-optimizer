# The Reliability Tax

**What does the last mile of clean energy actually cost — and does the path you take to get there change the answer?**

## Overview

The Reliability Tax quantifies the cumulative cost of pursuing progressively higher hourly carbon-free energy (CFE) targets across 7 US ISOs. It compares four distinct procurement pathways from 2025 to 2040, measuring not just the direct cost of clean energy procurement but also the stranded assets, curtailed generation, and path-dependent cost penalties each pathway creates.

This is a **deterministic cost minimization under constraint** — not a market forecast. Given fixed cost assumptions and a target CFE endpoint, what is the cheapest build-order, and what does each pathway leave behind?

## The Four Pathways

| # | Pathway | Strategy | Clean Firm Entry |
|---|---------|----------|-----------------|
| 1 | **VRE + Storage Only** | Solar, wind, offshore wind, batteries, LDES only. No clean firm. | Never |
| 2a | **Behavioral Pivot** | VRE-only until hitting a 90% CFE plateau, then add clean firm. | At 90% CFE wall |
| 2b | **Economic Pivot** | VRE-only until marginal $/CFE% exceeds clean firm LCOE, then switch. | At cost crossover |
| 3 | **Proactive Clean Firm** | Clean firm (nuclear, CCS-CCGT, geothermal) deployed from year 1 alongside VRE. | Year 1 |

**Pathway 1** may be infeasible at high endpoints (99%+) for some ISOs. When infeasibility is detected, the ceiling is recorded and the run is flagged — no forced solution is manufactured.

## Endpoint Targets

Five CFE thresholds, each representing hourly matching across all 8,760 hours:

| Target | Interpretation |
|--------|---------------|
| **90%** | Strong clean portfolio; ~876 unmatched hours/year |
| **95%** | Near-complete; ~438 unmatched hours |
| **97.5%** | Last-mile inflection zone; ~219 unmatched hours |
| **99%** | Near-total; ~88 unmatched hours |
| **99.9%** | Effectively 100%; ~9 unmatched hours |

## Run Matrix

**Core runs**: 4 pathways x 5 endpoints x 7 ISOs = **140 runs**

**Demand growth sensitivity** (Section 2 only): L/M/H demand growth applied to a subset of runs using existing `DEMAND_GROWTH_RATES` from the parent pipeline.

**Cost reporting per run**:
- Undiscounted cumulative 2025-2040
- NPV at 5%, 7%, 9% real discount rates
- **Objective function: NPV @ 7%**

All costs in real 2025 dollars, no inflation adjustment.

## ISO Scope

Fully ISO-parametric. All 7 US ISOs:

| ISO | Key characteristic |
|-----|-------------------|
| CAISO | Solar-rich, geothermal available, floating offshore |
| ERCOT | Wind-heavy, high demand growth, energy-only market |
| PJM | Largest grid, nuclear-heavy, fixed-bottom offshore |
| NYISO | Hydro-Quebec imports, state policy, fixed-bottom offshore |
| NEISO | Winter gas constraints, fixed-bottom offshore |
| MISO | Wind corridor, moderate demand growth |
| SPP | Wind-dominated, smallest demand |

## Output Artifacts

Each run produces:

1. **Cost trajectory** — Annual and cumulative $/MWh by resource, 2025-2040
2. **Build-order manifest** — Year-by-year capacity additions (MW) and generation (TWh) by resource
3. **Stranding ledger** — Assets meeting stranding criteria at 2040:
   - New-build gas with <20% capacity factor
   - VRE with >30% fleet-wide curtailment
   - Associated transmission
4. **Pathway comparison** — Cross-pathway delta costs, pivot timing, infeasibility flags

Aggregate outputs:
- **Reliability tax curves** — Marginal and cumulative cost vs. CFE% by pathway
- **Pathway divergence maps** — Where and why pathways diverge per ISO
- **Stranding heatmap** — ISO x pathway x endpoint stranding exposure

## Relationship to Parent Pipeline

This analysis inherits from the existing hourly-cfe-optimizer pipeline:

| Source | What's inherited | What's changed |
|--------|-----------------|----------------|
| **Step 2.1** (Efficient Frontier) | EF parquet data as input | — |
| **Step 2.2A** (Cost Optimization) | Vectorized cost engine, coefficient matrices, clean firm tranching | Restructured for annual build-order sequencing |
| **Step 2.2B** (Tracks) | Demand growth coefficients, Pareto pruning | Reduced to L/M/H sensitivity only |
| **Step 6** (SMARTargets) | Wright's Law, merit-order dispatch, fossil retirement | All policy/market/stochastic mechanisms stripped |
| **pipeline_config** | All constants (LCOE, FOAK, NOAK, learning curves, demand growth) | Unchanged |

See `INHERITANCE_PLAN.md` for the complete function-level mapping.

## File Layout

```
analysis/reliability-tax/
  README.md                  # This file
  INHERITANCE_PLAN.md        # Function-level inheritance mapping
  DECISION_CARDS.md          # Methodology decisions requiring approval
  METHODOLOGY.md             # Running methodology log (updated as decisions are made)
  # Future (Prompt 2+):
  # reliability_tax.py       # Main analysis script
  # config.py                # Analysis-specific configuration
  # results/                 # Output parquets and JSON
```

## Status

**Phase**: Discovery (Prompt 1 of N)
**Next**: User reviews DECISION_CARDS.md, approves/overrides each card, then Prompt 2 writes the analysis pipeline.
