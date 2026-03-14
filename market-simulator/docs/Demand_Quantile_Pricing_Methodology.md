# Demand-Quantile Pricing Methodology

**Technical Note — Market Simulator LMP Engine**
**Version**: 11.3 (March 2026)

---

## 1. Overview

The demand-quantile pricing layer is a post-merit-order adjustment that captures real-world price formation effects missing from a single-bus dispatch model: transmission congestion, gas supply tightness, bid markup above marginal cost, and must-run surplus economics.

Standard merit-order models produce a step function — the marginal unit's cost at each hour. Real ISO clearing prices deviate substantially from pure marginal cost due to:
- **Congestion**: Locational price separation when transmission constraints bind
- **Gas tightness**: Seasonal pipeline constraints (Algonquin in NEISO, winter peaks across eastern ISOs)
- **Bid markup**: Generators bid above marginal cost in tight markets (10-30% markup observed in PJM SOM 2024)
- **Must-run surplus**: Nuclear and wind operate at negative variable cost, driving prices negative during low-demand hours

Rather than modeling each effect mechanistically (which requires nodal transmission topology, gas pipeline flow models, and generator-specific bidding behavior), we use a **reduced-form approach**: demand percentile rank as a single predictor of price deviation from merit-order marginal cost.

This approach is **descriptive, not causal**. It reproduces observed price distribution statistics but does not model the underlying mechanisms. Reliability degrades beyond calibration conditions (2024 grid composition) — particularly at VRE penetrations above 50% where structural changes in price formation occur. The VRE-scaled negative pricing adjustment (Section 6) partially addresses this limitation.

---

## 2. Architecture

The pricing computation applies five layers sequentially to each of 8,760 hourly prices:

```
Layer 1: Merit-Order Dispatch (base layer)
  └─ Walk fossil stack → marginal unit → marginal cost
  └─ Load-dependent heat rate ramp: +0-15% based on position within band

Layer 2: Reserve Scarcity Check
  └─ If reserve_ratio < scarcity_threshold → exponential scarcity adder

Layer 3: Surplus Pricing (clean energy > demand)
  └─ Exponential decay toward floor_price based on surplus ratio

Layer 4: Demand-Quantile Adjustments (this methodology)
  ├─ 4a: High-demand congestion adder (top 20-28% of hours)
  ├─ 4b: Scarcity tail (top 1-5% of hours)
  ├─ 4c: Low-demand negative pricing (bottom 7-22% of hours)
  ├─ 4d: Mid-low price compression (between low and midlow percentiles)
  └─ 4e: Clean surplus merit-order effect (> 3% surplus ratio)

Layer 5: Demand Elasticity Dampening (v11.3)
  └─ If LMP > threshold ($200-300/MWh) → logarithmic curtailment dampening
  └─ Reduces extreme prices by 3-7% (ISO-specific max curtailment 8-15%)
```

### Implementation Location

- **File**: `lmp_engine.py`, lines 947-1050
- **Entry point**: `compute_hourly_lmp_vectorized()`, applied after merit-order dispatch

---

## 3. Parameter Definitions

### 3.1 High-Demand Congestion Adder

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `dq_high_percentile` | P_high | Demand percentile threshold (hours above get adder) |
| `dq_high_max_adder` | A_max | Maximum $/MWh adder at peak demand hour |
| `dq_high_exponent` | γ | Curvature — higher = sharper peak premium |

**Formula**:
```
position = (demand_rank - P_high/100) / (1 - P_high/100)
adder = A_max × position^γ
```

**Calibration basis**: PJM congestion revenues averaged $3.01/MWh system-wide in 2024 (IMM SOM), but peak-hour congestion reaches $20-80/MWh at constrained interfaces (AP South, COMED, PSEG).

### 3.2 Scarcity Tail

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `dq_scarcity_percentile` | P_scar | Percentile for scarcity tail onset |
| `dq_scarcity_max` | S_max | Maximum scarcity-like adder ($/MWh) |

**Formula**:
```
position = (demand_rank - P_scar/100) / (1 - P_scar/100)
scarcity_adder = S_max × position   (linear ramp)
```

**Calibration basis**: PJM penalty factor regime ($1,000/MWh cap, rarely triggered), ERCOT ORDC ($5,000 VOLL post-2023 reform), CAISO scarcity pricing frequency (~60 hours in 2024).

### 3.3 Low-Demand Negative Pricing

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `dq_low_percentile` | P_low | Percentile below which prices depress |
| `dq_low_floor` | F_low | $/MWh floor for negative pricing |
| `dq_low_exponent` | α | Curvature for negative pricing |

**Formula**:
```
position = 1 - (demand_rank / P_low/100)    # 1 at lowest, 0 at threshold
depression = position^α
price = current × (1 - depression) + F_low × depression
```

**Calibration basis**: CAISO DMM reports ~600 negative-price hours in 2024 (duck curve midday), SPP ~500 (overnight wind surplus), PJM ~200 (nuclear baseload overnight).

### 3.4 Mid-Low Price Compression

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `dq_midlow_percentile` | P_midlow | Upper bound of mid-low band |
| `dq_midlow_discount` | δ | Maximum discount factor (0-1) |

**Formula**:
```
position = 1 - (demand_rank - P_low/100) / (P_midlow/100 - P_low/100)
discount = δ × position
price = price × (1 - discount)
```

**Purpose**: Off-peak hours (overnight, shoulders) consistently price below merit-order marginal cost due to gas CCGT minimum load constraints, cheap coal baseload, and wind energy. This band creates the realistic P25 → P50 price gradient.

### 3.5 Clean Surplus Merit-Order Effect

Applied when clean energy surplus exceeds 3% of demand:

```
surplus_ratio = surplus_mw / (demand_mw + 1)
surplus_factor = clip((surplus_ratio - 0.03) × 8, 0, 1)
depressed = current × (1 - surplus_factor × 0.6) + floor × surplus_factor × 0.6
price = min(current, depressed)    # Only depress, never increase
```

**Purpose**: Captures the direct merit-order displacement effect of zero-marginal-cost renewables, independent of demand percentile rank. Critical for CAISO midday solar surplus and SPP/ERCOT overnight wind surplus.

---

## 4. ISO-Specific Calibration Parameters

### Parameter Table (v11.3)

| Parameter | PJM | ERCOT | CAISO | NYISO | NEISO | MISO | SPP |
|-----------|-----|-------|-------|-------|-------|------|-----|
| **High-demand** | | | | | | | |
| dq_high_percentile | 75 | 80 | 72 | 76 | 80 | 76 | 76 |
| dq_high_max_adder | 23 | 28 | 80 | 50 | 25 | 38 | 38 |
| dq_high_exponent | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| **Scarcity tail** | | | | | | | |
| dq_scarcity_percentile | 95.5 | 98.5 | 99 | 98.5 | 99.5 | 97.5 | 98 |
| dq_scarcity_max | 155 | 220 | 160 | 160 | 50 | 140 | 130 |
| **Low-demand (negative)** | | | | | | | |
| dq_low_percentile | 9 | 12 | 15 | 7 | 8 | 12 | 22 |
| dq_low_floor | -25 | -50 | -50 | -20 | -25 | -30 | -30 |
| dq_low_exponent | 1.8 | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 | 1.8 |
| **Mid-low compression** | | | | | | | |
| dq_midlow_percentile | 70 | 72 | 65 | 68 | 68 | 72 | 75 |
| dq_midlow_discount | 0.55 | 0.70 | 0.52 | 0.45 | 0.55 | 0.70 | 0.75 |
| **Structural** | | | | | | | |
| floor_price | -30 | -50 | -60 | -20 | -25 | -30 | -40 |
| scarcity_cap | 2000 | 5000 | 2000 | 2000 | 400 | 3500 | 3500 |
| scarcity_threshold | 0.03 | 0.02 | 0.03 | 0.03 | 0.02 | 0.02 | 0.02 |

### Calibration Rationale by ISO

**PJM**: Calibrated against Monitoring Analytics 2024 SOM. 10% cost-based offer adder (now ISO-specific) + congestion + ancillary redispatch. Tightest calibration of any ISO — PJM SOM publishes hourly distribution statistics.

**ERCOT**: Energy-only market with ORDC. Modo Energy 2024 reports avg RT ~$26/MWh (down from $63 in 2023) due to massive solar+storage entry. Deep negative pricing (-$50 floor) from wind surplus. ORDC scarcity implemented as separate exponential mechanism (not demand-quantile).

**CAISO**: Largest peak/off-peak spread (duck curve). CAISO DMM reports ~600 negative-price hours, aggressive midday solar compression. High congestion adder ($80 max) reflects evening ramp premium.

**NYISO**: ICAP capacity market dampens volatility relative to PJM. Tight geography → high congestion but moderate negative pricing (~150 hours). Potomac Economics 2024 SOM avg $42/MWh.

**NEISO**: FCM capacity market + 4 GW import capability from HQ/NB Power. Winter gas pipeline constraint modeled separately (Algonquin premium: +$13.13/MWh on winter peak hours). Low scarcity cap ($400) reflects import availability.

**MISO**: Coal-heavy fleet (35%), wind congestion drives 40% of real-time congestion. PRA capacity market. Potomac Economics 2024 SOM avg $31/MWh (14% decrease from 2023).

**SPP**: Cheapest US wholesale market ($26.18/MWh avg). Wind at 37.1% of generation, markups avg -$38/MWh. Widest low-demand band (P22) reflects massive overnight wind surplus. Very low gas prices ($1.65/MMBtu summer 2024).

---

## 5. Calibration Process

### 5.1 Iterative SOM Calibration (v11 series)

The calibration used an iterative process against published ISO market monitor statistics:

1. **v11.0**: Initial parametrization from PJM SOM 2024. All ISOs start with PJM parameters.
2. **v11.1**: ISO-specific tuning using 2024 SOM data from each market monitor (Monitoring Analytics, Potomac Economics, CAISO DMM, ISO-NE EMM, SPP MMU). Adjusted scarcity_threshold, surplus_decay, and demand-quantile parameters.
3. **v11.2**: Focus on CAISO and NYISO — adjusted high_percentile, scarcity_percentile, and midlow compression to match duck curve and congestion patterns.
4. **v11.3**: Final pass — reduced MISO/SPP volatility, refined ERCOT negative hours (517→~400), tuned SPP off-peak compression.

### 5.2 Calibration Targets

For each ISO, the model is calibrated against:

| Metric | Source | Tolerance |
|--------|--------|-----------|
| Average LMP | ISO SOM annual report | ±10% (GOOD), ±25% (FAIR) |
| Peak/off-peak averages | ISO SOM | ±15% |
| P10, P25, P50, P75, P90 | ISO SOM / EIA wholesale | ±25% for tails |
| Price volatility (std dev) | ISO SOM | ±30% |
| Negative price hours | ISO SOM / OASIS | ±50% (order of magnitude) |
| Scarcity hours (>$200) | ISO SOM | ±50% |

### 5.3 Calibration Script

`calibrate_lmp_model.py` provides automated calibration with two modes:

```bash
# Weather-normalized calibration (all 7 ISOs)
python calibrate_lmp_model.py --iso ALL

# QA against actual EIA hourly data
python calibrate_lmp_model.py --iso PJM --qa-actual --year 2024

# Compare against raw hourly LMP CSV
python calibrate_lmp_model.py --iso PJM --hourly path/to/pjm_lmp_2024.csv
```

---

## 6. VRE-Scaled Negative Pricing Extension

For trajectory mode simulations where VRE penetration increases beyond 2024 levels, negative pricing parameters scale dynamically:

```python
# Scale factors (applied when vre_penetration > 0.25)
vre_excess = vre_penetration - 0.25
floor_scale = 1.0 + 1.5 × vre_excess     # Deepens negative floor
pct_scale = 1.0 + 1.0 × vre_excess       # Widens negative-price band
# Both capped at 2.0×
```

| VRE % | Floor Scale | Band Scale | Effect |
|-------|-----------|----------|--------|
| 25% (baseline) | 1.0× | 1.0× | No adjustment |
| 35% | 1.15× | 1.10× | Mild deepening |
| 50% | 1.38× | 1.25× | Moderate (CAISO 2024 conditions) |
| 75% | 1.75× | 1.50× | Strong (CAISO projected 2035) |
| 100% | 2.0× (cap) | 2.0× (cap) | Maximum |

**Calibration source**: CAISO DMM quarterly reports 2019-2024 show negative price hours scaling approximately linearly with solar penetration share.

---

## 7. Known Limitations

1. **Descriptive, not causal**: The demand-quantile approach reproduces statistics but doesn't model congestion, gas pipeline flow, or generator bidding. Reliability degrades beyond calibration conditions.

2. **No cross-hour coupling**: Each hour priced independently. Real markets have start-up costs, minimum run times, and ramp constraints that create temporal price linkages.

3. **No locational variation**: Single-bus model produces system-average LMP. Real ISOs have substantial zonal/nodal variation (PJM: $10-20/MWh spread, NYISO: $30+ between upstate and NYC).

4. **Static calibration baseline**: Parameters calibrated to 2024 market structure. Structural changes (coal retirements, gas plant additions, storage deployment) may shift price formation beyond what VRE scaling captures.

5. **Simplified demand elasticity** (**Addressed v11.3**): Post-pricing dampening applied when LMP exceeds ISO-specific threshold ($200-300/MWh). Logarithmic curtailment ramp (5-15% max, ISO-specific) moderates extreme prices. This is a reduced-form approximation — real demand response involves complex contract structures, notification delays, and minimum curtailment durations not captured here.

---

## 8. References

- Monitoring Analytics (2024). *State of the Market Report for PJM 2024*.
- Potomac Economics (2024). *MISO State of the Market Report 2024*.
- Potomac Economics (2024). *NYISO State of the Market Report 2024*.
- CAISO Department of Market Monitoring (2024). *Q4 Report on Market Issues and Performance*.
- ISO New England (2024). *External Market Monitor Report 2024*.
- SPP Market Monitoring Unit (2024). *Annual State of the Market Report 2024*.
- Modo Energy (2024). *ERCOT 2024 Commercial Markets Update*.
- ERCOT (2024). *ORDC Methodology — Value of Lost Load Calculation*.
- EIA (2024). *Wholesale Electricity Market Data*. https://www.eia.gov/electricity/wholesalemarkets/
