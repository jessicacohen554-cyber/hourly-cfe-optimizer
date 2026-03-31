# New Analytical Opportunities for VRE Investment Thesis

Proposed analyses leveraging the hourly-cfe-optimizer pipeline infrastructure + VRE research data to strengthen the standalone VRE investment case.

---

### Analysis 1: VRE-Only Portfolio Value Quantification

**Question it answers:** What is a standalone VRE portfolio (solar + wind only) worth across ISO regions WITHOUT storage or firm clean backstop — and how much value does bundling add?

**Relevance to VRE standalone thesis:** This is THE core question of the deck. The thesis argues VRE is worth it *as part of a portfolio* with firm + storage. But investors need to see the standalone baseline to understand the "portfolio premium." If standalone VRE delivers $35/MWh but VRE+storage delivers $94/MWh (as Slide 9 claims), this analysis proves the delta with real optimizer physics rather than back-of-envelope estimates.

**Data inputs needed:**
- Step 2.1 efficient frontier parquets (`data/step2.1-ef/step_2_1_EF_{ISO}_{T}.parquet`) — filter to mixes where `clean_firm == 0` and all storage dispatch = 0
- Step 2.2A cost optimization results (`data/step2.2-cost/step_2_2a_CO_{ISO}.parquet`) — baseline full-portfolio costs for comparison
- LCOE tables from `pipeline_config.py` (solar: $40-120/MWh, wind: $28-105/MWh by ISO)
- Wholesale prices from `pipeline_config.py` ($25-42/MWh by ISO)
- Capacity factors from `pipeline_config.py` (solar 0.15-0.28, wind 0.25-0.42 by ISO)

**Script approach:**
1. Load EF parquets for each ISO × threshold
2. Filter to VRE-only mixes: `clean_firm_pct == 0`, `battery_dispatch_pct == 0`, `battery8_dispatch_pct == 0`, `ldes_dispatch_pct == 0`, `h2_dispatch_pct == 0`
3. For each VRE-only mix, compute cost using the Step 2.2A cost function at Medium sensitivities
4. Find cheapest VRE-only mix per ISO × threshold → this is the "standalone VRE frontier"
5. Compare against the full-portfolio optimal (from Step 2.2A results) at each threshold
6. Compute the "portfolio premium" = (VRE-only cost - full portfolio cost) / full portfolio cost
7. Identify the maximum achievable CFE threshold for VRE-only (where does it hit a physics wall?)

**Expected output:** `data-center-cfe/data-inputs/vre_standalone_value.json`
```json
{
  "PJM": {
    "thresholds": [50, 55, 60, ...],
    "vre_only_cost_per_mwh": [42.3, 45.1, ...],
    "full_portfolio_cost_per_mwh": [38.5, 40.2, ...],
    "portfolio_premium_pct": [9.9, 12.2, ...],
    "vre_only_max_threshold": 75.0,
    "vre_only_solar_pct": [45, 42, ...],
    "vre_only_wind_pct": [55, 58, ...]
  }
}
```

**Which slide it enhances or creates:**
- **Enhances Slide 9** (Hybrid Economics) — replaces the illustrative $35→$94 revenue stack with physics-backed numbers from the optimizer
- **Enhances Slide 12** (Investment Case) — quantifies the "standalone discount" vs portfolio premium with real data
- **Could create new slide:** "The Standalone VRE Ceiling" showing max achievable CFE% without storage/firm by ISO

**Estimated effort:** Quick (< 1 hour) — uses existing Step 2.1 EF data, just filters and re-evaluates. No Step 1 re-run needed.

---

### Analysis 2: Capture Rate Impact on VRE IRR

**Question it answers:** How does solar/wind capture rate erosion (solar dropping to <30% in CAISO, ~70% in PJM by 2030) affect project-level IRR under different PPA structures?

**Relevance to VRE standalone thesis:** The deck's Slide 5 shows capture rate erosion as a risk but doesn't quantify the *financial* impact on project returns. Investors need to see IRR sensitivity curves — "if capture rates drop to X%, my project IRR drops from Y% to Z%." This converts a physics risk into a financial risk metric that drives investment decisions. It also validates the deck's regional recommendations (PJM > ERCOT > CAISO) with IRR math.

**Data inputs needed:**
- LCOE tables from `pipeline_config.py` (solar/wind by ISO and L/M/H)
- Wholesale prices from `pipeline_config.py` ($25-42/MWh by ISO)
- Capacity factors from `pipeline_config.py` (solar 0.15-0.28, wind 0.25-0.42)
- Capture rate data from `vreresearch.md`: CAISO solar <30% by 2030, PJM solar ~70%, ERCOT solar ~50%
- PPA price assumptions: fixed-price ($35-50/MWh), merchant (wholesale × capture rate), indexed (wholesale + fixed premium)

**Script approach:**
1. Define 3 PPA structures: (a) fixed-price PPA at contract price, (b) merchant exposure at wholesale × capture rate, (c) indexed PPA at wholesale + $10-15/MWh premium
2. For each ISO × resource (solar, wind):
   - Compute annual revenue per MWh under each PPA structure across capture rate range (30%-100%)
   - Compute project-level costs: LCOE + transmission (Medium) + O&M
   - Compute unlevered IRR = (revenue - opex) / capex, annualized over 25-year PPA term
   - Standard finance: CAPEX = LCOE / CRF (reverse-engineer from LCOE tables), Revenue = capacity_factor × 8760 × price × capture_rate
3. Generate IRR sensitivity curves: IRR vs. capture rate for each ISO × PPA structure
4. Identify "breakeven capture rate" where IRR drops below cost of capital (8% WACC)

**Expected output:** `data-center-cfe/data-inputs/capture_rate_irr.json`
```json
{
  "PJM": {
    "solar": {
      "capture_rates": [0.30, 0.40, ..., 1.00],
      "irr_fixed_ppa": [12.1, 12.1, ...],
      "irr_merchant": [-2.3, 1.5, ..., 14.2],
      "irr_indexed": [4.5, 6.2, ..., 15.1],
      "breakeven_capture_merchant": 0.62,
      "breakeven_capture_indexed": 0.41
    },
    "wind": { ... }
  }
}
```

**Which slide it enhances or creates:**
- **Enhances Slide 5** (LMP Cannibalization) — adds IRR sensitivity chart alongside capture rate erosion, converting physics risk to financial risk
- **Enhances Slide 12** (Investment Case) — "10-12% IRR" claim gets backed by capture-rate-adjusted math
- **Could create new slide:** "PPA Structure Matters: IRR Under Capture Rate Stress" showing that fixed-price PPAs insulate returns while merchant exposure is lethal in CAISO

**Estimated effort:** Quick (< 1 hour) — pure financial math using existing LCOE/wholesale/CF constants from pipeline_config. No pipeline data needed beyond the config tables.

---

### Analysis 3: GHG Protocol Scenario Impact on VRE Value

**Question it answers:** How much more is a PJM-sited VRE REC worth than a MISO/SPP-sited VRE REC under the new GHG Protocol deliverability + hourly matching rules — and what's the total revenue uplift for a 1 GW VRE portfolio?

**Relevance to VRE standalone thesis:** The deck's core argument (Slides 13-14) is that GHG Protocol changes create a "localized REC premium." But it never quantifies the premium in dollar terms. The vreresearch.md cites a 3-7x REC price multiplier when deliverability restrictions apply. This analysis converts that multiplier into $/MWh revenue uplift by ISO, proving PJM VRE is worth dramatically more than SPP VRE under the new rules. This is the #1 differentiator for the deck's regional recommendations.

**Data inputs needed:**
- Existing Step 5 procurement strategy results (`data/step5-scenarios/strategy2_hourly.json`) — hourly matching costs by ISO
- Step 5 annual matching results (`data/step5-scenarios/strategy3_annual.json`) — annual matching costs for comparison
- Wholesale prices from `pipeline_config.py` by ISO
- REC price assumptions: current unbundled national REC ($5-15/MWh), current in-region REC ($15-30/MWh), post-GHG-Protocol localized hourly REC (3-7x multiplier from vreresearch.md)
- Grid emission intensity by ISO (from `pipeline_config.py` fossil mix and emission rates)

**Script approach:**
1. Define 3 REC accounting regimes:
   - **Status quo (annual, any-region):** REC price = national unbundled ($5-15/MWh depending on market)
   - **Transitional (annual, deliverability-restricted):** REC price = in-region unbundled ($15-30/MWh)
   - **New GHG Protocol (hourly, deliverability-restricted):** REC price = localized hourly (3-7x base)
2. For each ISO, compute the REC premium multiplier based on:
   - Grid carbon intensity (higher = more valuable REC — PJM's 0.64 tCO2/MWh > CAISO's 0.35)
   - VRE scarcity (lower existing VRE penetration = higher premium — PJM 6.7% vs SPP 37.5%)
   - Demand concentration (data center load density in region)
3. Compute total revenue per MWh for a 1 GW solar/wind asset under each regime × ISO
4. Output the "GHG Protocol premium" = new regime revenue - status quo revenue

**Expected output:** `data-center-cfe/data-inputs/ghg_protocol_premium.json`
```json
{
  "PJM": {
    "current_rec_value": 12.0,
    "transitional_rec_value": 28.0,
    "hourly_local_rec_value": 65.0,
    "ghg_protocol_premium_per_mwh": 53.0,
    "annual_uplift_1gw_solar_millions": 23.4,
    "grid_carbon_intensity": 0.64,
    "existing_vre_penetration_pct": 6.7
  },
  "ERCOT": { ... },
  "CAISO": { ... }
}
```

**Which slide it enhances or creates:**
- **Enhances Slide 13** (Procurement Accounting) — adds dollar-value premium per MWh by ISO under each accounting regime
- **Enhances Slide 14** (Cross-Regional Procurement) — quantifies why PJM-sited VRE commands 4-5x the REC value of SPP-sited VRE
- **Enhances Slide 11** (Regional Matrix) — adds a "GHG Protocol REC Premium" row to the heatmap

**Estimated effort:** Quick (< 1 hour) — uses Step 5 outputs + pipeline_config constants + vreresearch multipliers. Straightforward financial modeling, no physics re-runs.

---

### Analysis 4: Tier 2 Colocation Demand Sizing & Addressable Market

**Question it answers:** How large is the addressable market (in TWh and $B/yr) for an IPP selling VRE + firm clean 24/7 CFE products to the 6 major colocation providers, and what does their clean energy gap look like under hourly matching?

**Relevance to VRE standalone thesis:** Slide 7 lists colocation providers and their climate commitments but doesn't size the *market opportunity* in revenue terms. The deck argues Tier 2 colos are "desperate" for third-party PPAs — this analysis quantifies exactly how desperate by modeling their clean energy gaps under hourly matching (much larger than annual matching gaps) and pricing the procurement needed to fill those gaps. This turns a qualitative claim into a TAM (total addressable market) number.

**Data inputs needed:**
- Colocation consumption data from `vreresearch.md`: Digital Realty 11.1 TWh, Equinix 8.56 TWh, CyrusOne 4.2 TWh, QTS 2.6 TWh, Vantage 2.6 TWh, Iron Mountain ~2 TWh (est.)
- Clean energy coverage percentages from vreresearch.md: Equinix 96%, Digital Realty 75%, CyrusOne ~99% (EU), QTS ~100% (via RECs), Vantage <50%, Iron Mountain 100% (annual)
- Over-procurement ratios from `step5_2c_strategy_hourly.py` (1.05x at 50% → 2.8x at 99.9%)
- PPA pricing from Step 5 results or pipeline_config LCOE tables
- Geographic distribution assumptions: PJM ~40% of US colo load (Northern Virginia), ERCOT ~20%, CAISO ~15%, other ~25%

**Script approach:**
1. Build a colo provider model with: name, total TWh, current clean %, target clean %, target year, geographic split by ISO
2. For each provider, compute:
   - Annual matching gap = total_twh × (target% - current%) — what they need under today's rules
   - Hourly matching gap = annual_gap × over_procurement_ratio(target_threshold) — what they need under new GHG Protocol
   - The "hourly penalty" = hourly_gap - annual_gap (the extra procurement forced by hourly matching)
3. Price the gap at PPA rates by ISO (from Step 5 hourly strategy results)
4. Aggregate across all 6 providers to get total addressable market in TWh/yr and $B/yr
5. Segment by ISO to show where the demand concentrates

**Expected output:** `data-center-cfe/data-inputs/colocation_market_sizing.json`
```json
{
  "providers": {
    "Equinix": {
      "total_twh": 8.56,
      "current_clean_pct": 96,
      "target_clean_pct": 100,
      "annual_gap_twh": 0.34,
      "hourly_gap_twh": 0.75,
      "hourly_penalty_twh": 0.41,
      "cost_to_fill_gap_millions": 52.3,
      "iso_split": {"PJM": 0.45, "ERCOT": 0.20, "CAISO": 0.15, "other": 0.20}
    }
  },
  "totals": {
    "aggregate_twh": 31.06,
    "aggregate_annual_gap_twh": 8.2,
    "aggregate_hourly_gap_twh": 18.5,
    "total_addressable_market_billions": 2.1,
    "by_iso": {"PJM": 0.95, "ERCOT": 0.45, "CAISO": 0.30, "other": 0.40}
  }
}
```

**Which slide it enhances or creates:**
- **Enhances Slide 7** (Colocation Developers) — adds TAM sizing, converts the qualitative table into a quantified market opportunity
- **Enhances Slide 12** (Investment Case) — "addressable market = $2.1B/yr from Tier 2 alone" is a compelling data point
- **Could create new slide:** "The $2B+ Tier 2 Opportunity" with stacked bar chart showing hourly vs annual gap by provider

**Estimated effort:** Quick (< 1 hour) — mostly data assembly from vreresearch.md + applying over-procurement ratios from step5_2c. No pipeline re-runs.

---

### Analysis 5: Hyperscaler Vertical Integration Risk (Bear Case Stress Test)

**Question it answers:** If the Big 4 hyperscalers increasingly self-develop VRE (like Google's $4.75B Intersect Power acquisition), what happens to third-party PPA demand — and is the remaining market still large enough to justify a standalone VRE pipeline?

**Relevance to VRE standalone thesis:** This is the #1 bear case against the thesis. Google just acquired 10.8 GW of solar+storage. If Amazon, Microsoft, and Meta follow suit and vertically integrate their VRE procurement, the PPA market shrinks dramatically. The deck must address this head-on with scenario modeling, not hand-waving. Showing the market is still profitable even at 75% self-supply by the Big 4 makes the thesis robust.

**Data inputs needed:**
- Hyperscaler demand from vreresearch.md: Amazon 164 TWh, Google 112 TWh, Microsoft 186 TWh, Meta 134 TWh (2030 projections)
- Current contracted clean energy by hyperscaler (from deck Slide 4 gap analysis)
- Colocation demand from Analysis 4 above (~31 TWh from 6 providers)
- Regional demand splits (PJM ~40% of data center load, ERCOT ~25%, etc.)
- PPA pricing from pipeline_config LCOE tables
- Total clean energy gap from deck Slide 4: ~399 TWh in 2030

**Script approach:**
1. Define 4 vertical integration scenarios:
   - **Base case (status quo):** Big 4 self-supply 10% of VRE needs, 90% via third-party PPAs
   - **Moderate integration (25%):** Post-Intersect trend — each hyperscaler acquires 1-2 dev platforms
   - **Aggressive integration (50%):** Big 4 build internal dev teams, acquire multiple platforms
   - **Extreme integration (75%):** Big 4 become de facto utilities (Google-like model at scale)
2. For each scenario:
   - Compute remaining third-party PPA demand = total_gap × (1 - self_supply_pct) + tier2_demand (Tier 2 never self-develops)
   - Price the remaining market at PPA rates by ISO
   - Compute market concentration: fewer buyers = more pricing power for IPPs (or less?)
3. Key insight: Tier 2 colocation demand is IMMUNE to vertical integration — they can't self-develop. So even at 75% Big 4 self-supply, the Tier 2 market alone is $2B+/yr.
4. Additional analysis: vertical integration creates *more* demand for PPAs in the short term because self-development takes 3-5 years, during which hyperscalers still need bridge PPAs.

**Expected output:** `data-center-cfe/data-inputs/vertical_integration_scenarios.json`
```json
{
  "scenarios": {
    "base_10pct": {
      "big4_self_supply_pct": 10,
      "remaining_big4_ppa_twh": 537.6,
      "tier2_ppa_twh": 18.5,
      "total_third_party_market_twh": 556.1,
      "total_market_value_billions": 28.4
    },
    "moderate_25pct": { ... },
    "aggressive_50pct": { ... },
    "extreme_75pct": { ... }
  },
  "key_insight": "Even at 75% Big 4 self-supply, remaining market = 168 TWh Big 4 + 18.5 TWh Tier 2 = 186 TWh ($9.5B/yr). Tier 2 is the floor.",
  "bridge_demand_twh": 120.0
}
```

**Which slide it enhances or creates:**
- **Creates new slide** (after Slide 7 or as part of Slide 12): "Stress-Testing the Bear Case" — waterfall chart showing market size under each scenario, with Tier 2 as the resilient floor
- **Enhances Slide 12** (Investment Case) — addresses the #1 investor concern directly with scenario math
- **Enhances Slide 6** (Hyperscaler Deals) — contextualizes Intersect Power acquisition as one data point in a spectrum of outcomes

**Estimated effort:** Quick (< 1 hour) — scenario math using demand figures from vreresearch.md and gap analysis. No pipeline data needed.

---

### Analysis 6: Storage Pairing Value Quantification

**Question it answers:** What is the incremental $/MWh value of adding 4hr or 8hr battery storage to a standalone VRE asset, broken down by ISO and CFE threshold — and is that value from arbitrage or from enabling higher CFE matching?

**Relevance to VRE standalone thesis:** Slide 9 makes the qualitative case that "storage changes everything" with a $35→$94/MWh revenue uplift. This analysis decomposes that uplift into its two components: (a) **arbitrage value** — shifting cheap midday solar to expensive evening peak, and (b) **CFE enablement value** — the premium from being able to offer hourly-matched products that standalone VRE cannot. This distinction matters because arbitrage is a commodity play (erodes with more storage on the grid), while CFE enablement is a structural advantage tied to GHG Protocol rules.

**Data inputs needed:**
- Step 3A dispatch cache (`data/step3-dispatch/`) — 8,760-hour dispatch profiles for VRE-only vs VRE+storage mixes
- Step 2.2A cost results — cost difference between VRE-only optimal and VRE+storage optimal at each threshold
- Storage LCOE from `pipeline_config.py`: battery4 $30K-53K/% (annualized), battery8 $26K-45K/%
- Storage arbitrage revenue from `pipeline_config.py`: battery $25-55/kW-yr by ISO
- Storage revenue credits from `pipeline_config.py` (pre-computed: `STORAGE_REVENUE_CREDITS`)
- Capacity market prices from `pipeline_config.py`: $0 (ERCOT/SPP) to $120/kW-yr (PJM)

**Script approach:**
1. For each ISO × threshold, pull two mixes from Step 2.2A results:
   - **VRE-only optimal:** cheapest mix with solar+wind only (from Analysis 1 if available, or filter here)
   - **VRE+storage optimal:** cheapest mix with solar+wind+battery (no clean firm)
2. Compute the cost delta: `storage_value = vre_only_cost - vre_storage_cost`
3. Decompose the value:
   - **Arbitrage component:** storage revenue credits from `STORAGE_REVENUE_CREDITS[battery][iso]` — this is the value from energy price spreads
   - **CFE enablement component:** residual value = total_value - arbitrage_value — this is the premium from reaching a higher CFE threshold (or reaching the same threshold at lower total procurement)
4. For thresholds where VRE-only *can't reach* (e.g., >75% in most ISOs), the entire value is "enablement" — storage makes otherwise impossible products feasible
5. Compute the "storage multiplier": for each $1 invested in storage LCOE, how many $/MWh of portfolio value does it create?

**Expected output:** `data-center-cfe/data-inputs/storage_pairing_value.json`
```json
{
  "PJM": {
    "thresholds": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "vre_only_achievable": true,  // up to threshold X
    "vre_only_max_threshold": 72.0,
    "battery4_value": {
      "total_uplift_per_mwh": [3.2, 4.1, 5.8, 8.2, 12.1, 18.5, null, null, null, null],
      "arbitrage_component": [3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2],
      "cfe_enablement_component": [0.0, 0.9, 2.6, 5.0, 8.9, 15.3, "enables", "enables", "enables", "enables"],
      "storage_multiplier": [1.8, 2.1, 2.5, 3.1, 4.2, 5.8, "infinite", "infinite", "infinite", "infinite"]
    },
    "battery8_value": { ... }
  }
}
```

**Which slide it enhances or creates:**
- **Enhances Slide 9** (Hybrid Economics) — replaces illustrative Sankey with real optimizer data decomposed into arbitrage vs CFE enablement
- **Enhances Slide 11** (Regional Matrix) — adds "Storage Pairing Multiplier" metric per ISO
- **Could create new slide:** "Storage: Arbitrage vs. Enablement" — stacked bar chart showing the two value components across thresholds, with a clear visual breakpoint where VRE-only hits its ceiling and storage becomes *essential* (not just profitable)

**Estimated effort:** Medium (1-3 hours) — requires reading Step 3 dispatch cache NPZ files and cross-referencing Step 2.2A results. Some coding to extract VRE-only vs VRE+storage subsets from EF parquets. No Step 1 re-run, but more data wrangling than the other analyses.

---

## Priority Ranking

| # | Analysis | Effort | Impact on Deck | Priority |
|---|----------|--------|----------------|----------|
| 1 | VRE-Only Portfolio Value | Quick | High — core thesis proof | **P0** |
| 3 | GHG Protocol REC Premium | Quick | High — #1 differentiator | **P0** |
| 4 | Tier 2 Colocation Market Sizing | Quick | High — TAM quantification | **P1** |
| 2 | Capture Rate IRR Sensitivity | Quick | Medium — converts physics→finance | **P1** |
| 5 | Vertical Integration Bear Case | Quick | Medium — addresses #1 risk | **P1** |
| 6 | Storage Pairing Decomposition | Medium | High — deepens core argument | **P2** |

**Recommended execution order:** 1 → 3 → 4 → 2 → 5 → 6

Analyses 1-5 are all Quick (<1hr) and use existing pipeline outputs or config constants. Analysis 6 is the only one requiring dispatch cache reads, so it's deferred but highest analytical value.

**No analysis requires a Step 1 re-run.** All use existing Step 2-5 outputs or pipeline_config constants.
