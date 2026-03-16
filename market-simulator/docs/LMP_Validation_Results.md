# Synthetic LMP Validation Against Actual ISO Clearing Prices

**Technical Note — Market Simulator LMP Engine**
**Version**: 11.3 (March 2026)

---

## 1. Validation Approach

The synthetic LMP model is validated against published ISO market statistics from State of the Market (SOM) reports and EIA wholesale market data. Two validation modes are used:

1. **Statistical calibration**: Compare model output distribution statistics (P10, P50, P90, mean, std dev, negative hours, scarcity hours) against published ISO annual statistics.
2. **QA against actuals**: Where raw hourly data is available (EIA-930), compare model LMP computed from actual hourly demand and generation against published clearing prices.

### Validation Script

```bash
# Run all-ISO calibration
python calibrate_lmp_model.py --iso ALL

# QA against EIA actuals (PJM)
python calibrate_lmp_model.py --iso PJM --qa-actual --year 2024
```

---

## 2. Calibration Targets by ISO

All targets sourced from 2024 SOM reports and EIA wholesale market data.

### 2.1 PJM — Monitoring Analytics SOM 2024

| Metric | Target | Source |
|--------|--------|--------|
| Avg LMP | $34.70/MWh | RT load-weighted avg (MA SOM 2024 Table 3-1) |
| Peak avg | $42.00/MWh | On-peak hours (MA SOM 2024, Sec. 3) |
| Off-peak avg | $28.00/MWh | Off-peak hours (MA SOM 2024, Sec. 3) |
| P10 | $18.00/MWh | Estimated from distribution (MA SOM Figure 3-5) |
| P25 | $23.00/MWh | Gas CCGT marginal cost floor |
| P50 | $30.00/MWh | Median — gas CCGT marginal |
| P75 | $42.00/MWh | Gas CT marginal |
| P90 | $55.00/MWh | Gas CT + congestion |
| Volatility | $25.00 | Standard deviation |
| Negative hours | ~200 | DA negative price intervals (est.) |
| Scarcity hours | ~100 | Hours > $200/MWh |

**PJM-specific calibration notes**:
- PJM has the most detailed publicly available calibration data of any ISO
- 10% cost-based offer adder applied (PJM IMM rule — generators bid cost + 10% markup)
- Congestion revenues $3.01/MWh system-wide (2024 SOM)
- Ancillary redispatch cost $1.33/MWh
- DA risk premium +$2/MWh vs RT

### 2.2 ERCOT — Modo Energy / ERCOT Commercial Markets

| Metric | Target | Source |
|--------|--------|--------|
| Avg LMP | $26.00/MWh | RT avg (Modo Energy 2024 year-end report) |
| Peak avg | $36.00/MWh | On-peak |
| Off-peak avg | $18.00/MWh | Off-peak |
| P10 | $5.00/MWh | Deep overnight wind |
| P50 | $22.00/MWh | Median |
| P90 | $50.00/MWh | Summer afternoon peak |
| Volatility | $35.00 | Higher than capacity-market ISOs |
| Negative hours | ~400 | Wind surplus hours |
| Scarcity hours | ~80 | ORDC-priced hours |

**ERCOT-specific calibration notes**:
- Energy-only market — no capacity payments to dampen volatility
- ORDC (Operating Reserve Demand Curve) modeled as separate exponential mechanism
- No cost-based offer adder (energy-only market, competitive bidding)
- 2024 was unusually mild — solar+storage entry reduced scarcity pricing dramatically vs 2023 ($26 vs $63 avg)
- VOLL = $5,000/MWh (post-2023 PUCT reform)

### 2.3 CAISO — DMM 2024 Q4 + Annual Report

| Metric | Target | Source |
|--------|--------|--------|
| Avg LMP | $38.00/MWh | WEIM avg (CAISO DMM 2024) |
| Peak avg | $55.00/MWh | Evening ramp (5-9 PM) |
| Off-peak avg | $22.00/MWh | Midday solar surplus |
| P10 | -$5.00/MWh | Duck curve midday |
| P25 | $12.00/MWh | Solar surplus shoulder |
| P50 | $32.00/MWh | Median |
| P75 | $50.00/MWh | Evening ramp |
| P90 | $72.00/MWh | Summer peak + congestion |
| Volatility | $38.00 | Highest among ISOs |
| Negative hours | ~600 | Solar surplus midday hours |
| Scarcity hours | ~60 | Evening ramp + summer |

**CAISO-specific calibration notes**:
- Largest peak/off-peak spread of any US ISO (duck curve)
- GHG allowance cost adds $14-19/MWh to carbon-emitting generators
- Spring prices higher than summer in 2024 (unusual — hydro conditions)
- P10 target is negative ($-5) — unique among ISOs
- Deep negative pricing floor (-$60) from mandatory solar curtailment

### 2.4 NYISO — Potomac Economics SOM 2024

| Metric | Target | Source |
|--------|--------|--------|
| Avg LMP | $42.00/MWh | System avg (Potomac Economics) |
| Peak avg | $52.00/MWh | On-peak |
| Off-peak avg | $32.00/MWh | Off-peak |
| P10 | $18.00/MWh | Overnight baseload |
| P50 | $36.00/MWh | Median |
| P90 | $68.00/MWh | NYC congestion premium |
| Volatility | $30.00 | Moderate |
| Negative hours | ~150 | Fewest among eastern ISOs |
| Scarcity hours | ~70 | NYC load pocket scarcity |

**NYISO-specific calibration notes**:
- ICAP capacity market dampens energy price volatility
- Tight geography: Long Island/NYC congestion adds $10-30/MWh to zonal prices
- Model produces system-average LMP; actual zonal variation much higher
- Dual-fuel capability in NYC reduces scarcity pricing

### 2.5 ISO-NE (NEISO) — EMM Report 2024

| Metric | Target | Source |
|--------|--------|--------|
| Avg LMP | $39.50/MWh | Published annual avg (ISO-NE) |
| Peak avg | $50.00/MWh | On-peak |
| Off-peak avg | $30.00/MWh | Off-peak |
| P10 | $18.00/MWh | Overnight baseload |
| P50 | $34.00/MWh | Median |
| P90 | $65.00/MWh | Winter peak |
| Volatility | $28.00 | Moderate |
| Negative hours | ~180 | Nuclear baseload surplus |
| Scarcity hours | ~60 | Winter gas constraint |

**NEISO-specific calibration notes**:
- FCM capacity market + 4 GW import capability (HQ, NB Power) limits scarcity
- Winter gas pipeline constraint (Algonquin bottleneck) modeled separately: +$13.13/MWh CCS adder on winter peak hours (Dec-Feb, above P60 demand)
- Scarcity cap set low ($400) reflecting import availability
- High transmission rates ($24/MWh)

### 2.6 MISO — Potomac Economics SOM 2024

| Metric | Target | Source |
|--------|--------|--------|
| Avg LMP | $31.00/MWh | RT avg (14% decrease from 2023) |
| Peak avg | $38.00/MWh | On-peak |
| Off-peak avg | $25.00/MWh | Off-peak |
| P10 | $14.00/MWh | Overnight wind + coal baseload |
| P50 | $27.00/MWh | Median |
| P90 | $50.00/MWh | Summer peak |
| Volatility | $22.00 | Moderate |
| Negative hours | ~300 | Wind congestion |
| Scarcity hours | ~50 | PRA cushion |

**MISO-specific calibration notes**:
- Coal-heavy fleet (35% of fossil) — highest with PJM at 29%
- Wind drives ~40% of real-time congestion
- PRA capacity market provides adequate reserve cushion
- VOLL: $3,500/MWh (2024), approved increase to $10,000 (Sept 2025)
- Zone 5 capacity price outlier at $719.81/MW-day

### 2.7 SPP — MMU Annual SOM 2024

| Metric | Target | Source |
|--------|--------|--------|
| Avg LMP | $26.00/MWh | RT avg — cheapest US market |
| Peak avg | $33.00/MWh | On-peak |
| Off-peak avg | $20.00/MWh | Off-peak |
| P10 | $8.00/MWh | Overnight wind surplus |
| P50 | $23.00/MWh | Median |
| P90 | $45.00/MWh | Summer peak |
| Volatility | $24.00 | Moderate |
| Negative hours | ~500 | Wind 37.1% of generation |
| Scarcity hours | ~30 | Rare |

**SPP-specific calibration notes**:
- Wind at 37.1% of generation — highest in US
- Wind markups average -$37.99/MWh (heavily negative bidding)
- Very low gas prices in SPP footprint ($1.65/MMBtu summer 2024)
- Flat geography → low transmission congestion
- Limited capacity market → closer to energy-only than capacity-market ISOs
- Widest negative-price band (P22 threshold) reflecting massive overnight wind

---

## 3. Model Configuration

### 3.1 Merit-Order Stack

Fossil generators are ordered by marginal cost:

```
MC = heat_rate × fuel_price × (1 + ISO_adder) + VOM + CO₂_rate × carbon_price
```

| Unit Type | Heat Rate (MMBtu/MWh) | VOM ($/MWh) | CO₂ Rate (tons/MWh) |
|-----------|----------------------|-------------|---------------------|
| Gas CCGT | 7.0 | 3.50 | 0.411 |
| Gas CT | 10.5 | 5.50 | 0.571 |
| Coal Steam | 10.0 | 5.00 | 0.920 |
| Oil CT | 12.5 | 8.00 | 0.840 |

**ISO-specific cost-based offer adder**:
- PJM, NYISO, NEISO, MISO: 10% (capacity-market ISOs — markup above cost-based offers, per IMM observations)
- ERCOT: 0% (energy-only market — competitive bidding without cost-based regulation)
- CAISO: 5% (RA market — partial cost-based, lower markup than eastern ISOs)
- SPP: 0% (limited capacity market, competitive bidding)

### 3.2 Capacity Degradation

Fossil capacity retires via S-curve model (logistic function):

```
remaining = 1 / (1 + exp(-k × (clean_pct - midpoint)))
```

Parameters calibrated to PJM RPM auction data (2020-2024) and ERCOT bilateral retirement announcements.

### 3.3 Load-Dependent Heat Rate Ramp

Within each merit-order band, marginal cost increases 0-15% as utilization rises:

```
heat_rate_ramp = 1.0 + 0.15 × position_in_band^1.5
```

Based on gas CCGT heat rate variation: ~6.7-7.5 MMBtu/MWh across load range (~12%). Gas CT: ~9.5-11.5 (~20%). Coal: ~9.5-11.0 (~16%).

---

## 4. Validation Results Summary

### 4.1 Accuracy by Metric

| Metric | Avg Absolute Error | Notes |
|--------|-------------------|-------|
| Average LMP | ±$2-4/MWh (±8%) | Consistently within tolerance |
| Peak/off-peak spread | ±$3-5/MWh (±12%) | Slight overestimate of peak premium |
| P10 | ±$2-5/MWh | Tail sensitive to negative pricing params |
| P50 | ±$2-3/MWh (±8%) | Best-calibrated metric |
| P90 | ±$5-8/MWh (±15%) | Scarcity tail inherently more variable |
| Volatility | ±$3-5 (±15%) | Within tolerance for screening |
| Negative hours | ±30% | Order-of-magnitude accuracy |
| Scarcity hours | ±40% | Rare events inherently harder to match |

### 4.2 Confidence Ratings by ISO

| ISO | Calibration Quality | Data Availability | Confidence |
|-----|-------------------|-------------------|------------|
| PJM | Excellent | Best (MA SOM detailed tables) | High |
| ERCOT | Good | Good (Modo Energy, ERCOT CMU) | High |
| CAISO | Good | Good (DMM quarterly reports) | Medium-High |
| MISO | Good | Good (Potomac Economics SOM) | Medium-High |
| SPP | Good | Good (SPP MMU annual) | Medium-High |
| NYISO | Fair | Fair (Potomac SOM, limited detail) | Medium |
| NEISO | Fair | Fair (ISO-NE EMM, limited detail) | Medium |

### 4.3 Known Biases

| Direction | Magnitude | Cause | Mitigation |
|-----------|-----------|-------|------------|
| +$2-4/MWh average for ERCOT/SPP | Systematic positive | Was: uniform 10% adder. Now: ISO-specific (0% for energy-only) | **Fixed** in v11.3 |
| -$1-3/MWh for NEISO winter | Systematic negative | Gas pipeline constraint adder only on CCS, not on gas dispatch | Documented as known limitation |
| +$5-15/MWh P90 for all ISOs | Positive tail bias | Demand-quantile scarcity tail is linear; real scarcity is clustered | Linear ramp intentional; demand elasticity dampening (v11.3) moderates extreme prices by 3-7% |
| ±30% negative hours | High variance | Negative pricing driven by rare surplus events | Order-of-magnitude accuracy acceptable for screening |

---

## 5. Limitations and Appropriate Use

### Fit-for-Purpose Assessment

The synthetic LMP model is designed as a **screening tool**, not a replacement for production-grade nodal market models (GenX, IPM, ReEDS, PLEXOS). It reliably answers:

- **Directional questions** (>95% confidence): "Will higher renewable penetration reduce average LMP?" Yes.
- **Ordinal questions** (>90% confidence): "Which ISO has the highest scarcity risk?" Correct rank ordering.
- **Magnitude questions** (>80% confidence): "What is the average LMP at 60% clean?" Within ±$5/MWh.

It should **not** be used for:
- Facility-siting decisions requiring nodal price resolution
- Generator-specific revenue projections (zonal variation too large)
- Regulatory compliance filings requiring auditable dispatch models
- Hour-ahead or day-ahead price forecasting

### When to Escalate

Use GenX/IPM/ReEDS when:
- Transmission expansion is a decision variable
- Unit commitment constraints materially affect outcomes
- Results must withstand regulatory/legal scrutiny
- Facility-specific revenue calculations needed

---

## 6. Data Sources

| Source | Use | URL |
|--------|-----|-----|
| PJM IMM SOM 2024 | PJM calibration targets | monitoringanalytics.com/reports |
| Potomac Economics | MISO, NYISO SOM data | potomaceconomics.com |
| CAISO DMM | CAISO calibration targets | caiso.com/market-monitoring |
| ISO-NE EMM | NEISO calibration targets | iso-ne.com/markets |
| SPP MMU | SPP calibration targets | spp.org/markets-operations |
| Modo Energy | ERCOT market data | modoenergy.com |
| EIA Wholesale Markets | Cross-validation | eia.gov/electricity/wholesalemarkets |
| EIA-930 Hourly Grid Monitor | QA actual profiles | eia.gov/electricity/gridmonitor |
