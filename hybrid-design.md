# Hybrid Solar+Storage & Wind+Storage Design Document

## Status: Design Phase

---

## 1. Overview

Adding co-located hybrid resources as new asset types in the optimizer:
- **Solar+Battery (4hr)**: DC:AC overbuild captures clipped solar energy
- **Wind+Battery (8hr)**: Temporal shifting of off-peak wind to net-peak demand

These are distinct from standalone solar + standalone battery because:
- Single interconnection (shared queue position, shared transmission upgrades)
- Battery charges only from co-located generation (no grid charging → ITC-qualifying)
- Combined output capped at interconnection (AC) rating

---

## 2. Solar+Battery Hybrid

### 2.1 DC:AC Ratio (Region-Specific)

Solar panels (DC) are oversized relative to the grid interconnection (AC). During peak sun, generation exceeds interconnection capacity — the excess ("clipped" energy) charges the co-located battery instead of being wasted.

| ISO | Solar CF (approx) | DC:AC Ratio | Rationale |
|--------|----------|-------------|-----------|
| CAISO | ~29% | 1.40 | Strong resource, significant midday clipping |
| ERCOT | ~26% | 1.35 | Good resource, moderate clipping |
| SPP | ~24% | 1.30 | Decent resource |
| MISO | ~22% | 1.30 | Moderate resource |
| PJM | ~21% | 1.25 | Weaker resource, less clipping |
| NYISO | ~19% | 1.20 | Weak resource, minimal clipping |
| NEISO | ~18% | 1.20 | Weakest, minimal overbuild justified |

**Validation**: Initial test sweep against existing EIA-930 solar profiles to confirm clipping hours and recovered energy at each ratio. See Section 5.

### 2.2 Battery Specification

- **Duration**: 4 hours
- **Round-trip efficiency**: 85% (consistent with existing standalone battery model)
- **Grid charging**: None (pure hybrid — charges only from clipped solar)
- **Discharge rule**: Net-peak hours (demand minus renewable generation is highest)

### 2.3 Dispatch Model

Hourly dispatch logic (per hour h):

```
solar_dc[h] = installed_dc_capacity × solar_profile[h]
interconnect_cap = installed_dc_capacity / dc_ac_ratio

if solar_dc[h] > interconnect_cap:
    to_grid[h] = interconnect_cap
    clipped[h] = solar_dc[h] - interconnect_cap
    charge[h] = min(clipped[h], battery_remaining_capacity) × rte_charge
else:
    to_grid[h] = solar_dc[h]
    clipped[h] = 0
    charge[h] = 0

# Discharge during net-peak hours
if is_net_peak[h] and battery_soc > 0:
    discharge[h] = min(battery_power_rating, battery_soc)
    to_grid[h] += discharge[h]
    # Combined output still capped at interconnection
    to_grid[h] = min(to_grid[h], interconnect_cap)

hybrid_output[h] = to_grid[h]
```

### 2.4 Net-Peak Definition

Net peak = hours where (demand - total renewable generation) is highest in each day. For a 4hr battery, the top 4 net-peak hours per day trigger discharge. This avoids 8760 LMP optimization while producing nearly identical results (net peak and high LMP are highly correlated).

---

## 3. Wind+Battery Hybrid

### 3.1 Key Differences from Solar Hybrid

Wind does not have the DC:AC clipping dynamic. The value proposition is:
- **Shared interconnection** — one queue slot, one set of transmission upgrades
- **Temporal shifting** — overnight/off-peak wind generation shifted to daytime demand peak
- No overbuild ratio; wind output goes directly to grid up to interconnection capacity

### 3.2 Battery Specification

- **Duration**: 8 hours (wind generates over longer continuous periods; 4hr would leave significant generation unshifted)
- **Round-trip efficiency**: 85%
- **Grid charging**: None (pure hybrid)
- **Discharge rule**: Net-peak hours (top 8 hours per day)

### 3.3 Dispatch Model

```
wind_gen[h] = installed_capacity × wind_profile[h]

# Wind feeds grid directly; excess charges battery
if wind_gen[h] > demand_signal[h] and battery_soc < max_soc:
    to_grid[h] = demand_signal[h]
    excess[h] = wind_gen[h] - demand_signal[h]
    charge[h] = min(excess[h], battery_remaining_capacity) × rte_charge
else:
    to_grid[h] = wind_gen[h]

# Discharge during net-peak hours
if is_net_peak[h] and battery_soc > 0:
    discharge[h] = min(battery_power_rating, battery_soc)
    to_grid[h] += discharge[h]
    to_grid[h] = min(to_grid[h], interconnect_cap)

hybrid_output[h] = to_grid[h]
```

### 3.4 Charging Trigger

Unlike solar hybrids (where clipping is the charging trigger), wind hybrids charge when:
- Wind generation exceeds the co-located demand signal (off-peak surplus)
- Prioritize overnight hours when wind is typically strongest and demand is lowest

---

## 4. Cost Model

### 4.1 Solar+Battery Hybrid

```
hybrid_lcoe = weighted_blend(solar_lcoe, battery_lcoe)
             - shared_interconnection_savings
             + ITC_benefit (battery qualifies via co-location)
```

Components:
- **Solar LCOE**: Same as standalone, from LCOE_TABLES (L/M/H by ISO)
- **Battery LCOE**: 4hr standalone cost, reduced by ITC qualification
- **Interconnection savings**: One queue position instead of two; reduced transmission adder
- **ITC**: Co-located storage qualifies for solar ITC (30%+ depending on bonus criteria)

### 4.2 Wind+Battery Hybrid

```
hybrid_lcoe = weighted_blend(wind_lcoe, battery_8hr_lcoe)
             - shared_interconnection_savings
```

Components:
- **Wind LCOE**: Onshore wind from LCOE_TABLES
- **Battery LCOE**: 8hr standalone cost
- **Interconnection savings**: Same shared-queue benefit
- **ITC**: Wind uses PTC, not ITC — storage co-location ITC benefit is smaller/different (needs research for exact treatment under IRA rules)

### 4.3 Transmission Adder

Hybrid resources use a **single transmission adder** (not two). The adder is sized to the AC interconnection rating, not the total DC+storage capacity:

```
hybrid_tx = tx_adder[iso][level]  # One connection, not two
standalone_tx = tx_adder[iso][level] × 2  # Solar + battery each need one
savings = standalone_tx - hybrid_tx
```

---

## 5. Validation Plan

### 5.1 DC:AC Ratio Validation (Pre-Implementation)

Quick discovery script per ISO:
1. Load existing EIA-930 solar profiles (8760 hours)
2. Normalize to capacity factor
3. For DC:AC ratios [1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5]:
   - Count hours with clipping
   - Total clipped energy (TWh)
   - Energy recovered by 4hr battery (TWh)
   - Net capacity factor improvement
4. Confirm proposed regional ratios are in the sweet spot (diminishing returns above)

### 5.2 Wind Shifting Value (Pre-Implementation)

Quick analysis per ISO:
1. Load wind profiles
2. Identify overnight surplus hours (generation > demand share)
3. Quantify shiftable energy with 8hr battery
4. Compare temporal match improvement vs standalone wind

---

## 6. Pipeline Integration

### Phase 1: Step 1 (Physics)
- Add `solar_batt` and `wind_batt` as new resource types
- New dispatch columns in PFS output
- DC:AC ratio applied during solar_batt dispatch
- Hybrid output profiles stored alongside existing resource profiles

### Phase 2: Steps 2-7 (Downstream)
- Add hybrid cost model to pipeline_config.py
- Include in Step 2.2 cost optimization candidate pool
- Step 3B MAC queue: hybrids compete as archetypes
- Dashboard: new resource colors and labels
- Deferred — implement after Step 1 integration is validated

---

## 7. Decisions Log

| # | Decision | Date | Notes |
|---|----------|------|-------|
| 1 | Solar hybrid: 4hr battery | 2026-03-28 | Matches clipping window duration |
| 2 | Wind hybrid: 8hr battery | 2026-03-28 | Longer temporal shift needed |
| 3 | Pure hybrid (no grid charging) | 2026-03-28 | ITC-qualifying, cleaner model |
| 4 | Net-peak discharge rule | 2026-03-28 | Avoids 8760 LMP complexity |
| 5 | DC:AC fixed per-ISO | 2026-03-28 | Based on solar CF, validated by sweep |
| 6 | Wind: no DC:AC overbuild | 2026-03-28 | No clipping dynamic for wind |
| 7 | Start with Step 1, propagate later | 2026-03-28 | Phase 1 = physics only |

---

## 8. Open Questions

- [ ] Exact ITC treatment for wind+storage co-location under IRA rules
- [ ] Offshore wind+storage hybrids — include or defer?
- [ ] LCOE source data for hybrid-specific costs (NREL ATB 2024 has hybrid benchmarks)
- [ ] DC:AC validation results (pending sweep script)
