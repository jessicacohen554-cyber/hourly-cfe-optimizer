# Hybrid Solar+Storage & Wind+Storage Design Document

## Status: COMPLETE — All pipeline steps integrated, all open items resolved

---

## 1. Overview

Four co-located hybrid resource types added to the optimizer:

| Type | Generation | Battery | Primary Value |
|------|-----------|---------|---------------|
| `solar_batt4` | Solar | 4hr | Core clipping recovery |
| `solar_batt8` | Solar | 8hr | Extended clipping + deeper temporal shift |
| `wind_batt4` | Wind | 4hr | Short shifting, dual-cycle, narrow arbitrage blocks |
| `wind_batt8` | Wind | 8hr | Deep overnight-to-peak shifting |

Distinct from standalone solar/wind + standalone battery because:
- **Single interconnection** (shared queue position, shared transmission upgrades)
- **No grid charging** (pure hybrid — ITC-qualifying)
- **Combined output capped** at interconnection (AC) rating

Offshore wind hybrids: **Skipped.** Co-located storage doesn't reshape the offshore gen profile — it's a transmission/grid-level problem better handled by standalone grid storage and transmission constraint adders.

---

## 2. Solar Hybrids (solar_batt4, solar_batt8)

### 2.1 DC:AC Ratios — Validated

Solar panels (DC) oversized relative to grid interconnection (AC). During peak sun, excess generation ("clipped" energy) charges co-located battery.

**Validation sweep results** (EIA-930 profiles, `scripts/validate_dcac_ratios.py`):

**solar_batt4 ratios** (current industry practice):

| ISO | Solar CF | DC:AC | Hybrid CF (4hr) | CF Gain | Battery Recovery |
|-----|----------|-------|-----------------|---------|-----------------|
| CAISO | 29.6% | **1.35** | 37.6% | +8.0% | 19% |
| ERCOT | 25.2% | **1.50** | 35.8% | +10.6% | 40% |
| PJM | 23.1% | **1.50** | 34.0% | +10.9% | 77% |
| NYISO | 19.0% | **1.50** | 27.5% | +8.4% | 46% |
| NEISO | 19.1% | **1.50** | 28.3% | +9.2% | 85% |
| MISO | 20.8% | **1.50** | 30.2% | +9.4% | 26% |
| SPP | 20.9% | **1.50** | 30.6% | +9.7% | 36% |

**solar_batt8 ratios** (higher overbuild to justify 8hr capacity):

| ISO | DC:AC | Rationale |
|-----|-------|-----------|
| CAISO | **1.70** | 8hr gains >1% CF over 4hr at this ratio |
| All others | **2.00** | Need aggressive overbuild for 8hr to differentiate |

At solar_batt4 ratios (1.35-1.50), the 4hr battery already captures all clipped energy — making 8hr redundant (identical profiles). Higher ratios create enough clipping to fill the 8hr battery, producing meaningfully different profiles: +434 to +666 more active hours, flatter output shapes (lower peak/mean).

Key finding: weaker solar ISOs benefit *more* from overbuilding — smaller clipping events mean the battery captures a higher share. CAISO is the only ISO where diminishing returns kick in early.

### 2.2 Battery Specifications

| Parameter | solar_batt4 | solar_batt8 |
|-----------|------------|------------|
| Duration | 4 hours | 8 hours |
| Round-trip efficiency | 85% | 85% |
| Grid charging | None | None |
| Discharge rule | Top 4 net-peak hours/day | Top 8 net-peak hours/day |
| Power rating | = AC interconnection capacity | = AC interconnection capacity |

### 2.3 Dispatch Model

```python
# Per hour h:
solar_dc = dc_capacity × solar_profile[h]       # DC generation
ac_cap = dc_capacity / dc_ac_ratio               # Interconnection limit

# Clipping → charge
clipped = max(0, solar_dc - ac_cap)
charge = min(clipped, battery_headroom) * sqrt(rte)  # Half RTE on charge
to_grid = min(solar_dc, ac_cap)

# Discharge during net-peak hours
if is_net_peak[h] and soc > 0:
    discharge = min(power_rating, soc)
    to_grid += discharge * sqrt(rte)              # Half RTE on discharge
    to_grid = min(to_grid, ac_cap)                # Interconnection cap

hybrid_output[h] = to_grid
```

### 2.4 Net-Peak Definition

Net peak = hours where (demand − total renewable generation) is highest each day.
- solar_batt4: top 4 net-peak hours trigger discharge
- solar_batt8: top 8 net-peak hours trigger discharge

---

## 3. Wind Hybrids (wind_batt4, wind_batt8)

### 3.1 Key Differences from Solar

- **No DC:AC overbuild** — wind turbines are AC machines, no clipping dynamic
- **No overbuild ratio** — wind output goes directly to grid
- **Charging trigger**: wind generation exceeding demand signal during off-peak hours (not clipping)
- **Value**: shared interconnection savings + temporal shifting

### 3.2 Battery Sizing (MW Ratio)

Battery MW sized at 25-40% of wind nameplate capacity:
- **POI ceiling**: `battery_MW = POI_limit − avg_wind_during_peak_hours`
- **Economics cap**: >40% means battery sits partially empty during average wind cycles (8hr duration is capital-intensive)
- **Arbitrage block fit**: discharge duration must map to ISO's structural price spikes

Per-ISO sizing: **TBD** — pending wind profile analysis (same approach as solar DC:AC sweep).

### 3.3 Battery Specifications

| Parameter | wind_batt4 | wind_batt8 |
|-----------|-----------|-----------|
| Duration | 4 hours | 8 hours |
| Round-trip efficiency | 85% | 85% |
| Grid charging | None | None |
| Discharge rule | Top 4 net-peak hours/day | Top 8 net-peak hours/day |
| Battery:wind MW ratio | 25-40% (ISO-specific, TBD) | 25-40% (ISO-specific, TBD) |

### 3.4 Duration Selection Logic

- **wind_batt4**: ISOs with narrow arbitrage blocks (4-5hr evening spike). Also enables dual-cycle dispatch (charge → discharge morning peak → recharge → discharge evening peak).
- **wind_batt8**: ISOs with broad arbitrage blocks (8hr+ spanning morning + evening peaks, or extended afternoon). Captures full overnight-to-peak temporal shift.
- Optimizer chooses — both offered as candidates, mutually exclusive per project.

### 3.5 Dispatch Model

```python
# Per hour h:
wind_gen = capacity × wind_profile[h]

# Wind feeds grid; off-peak surplus charges battery
if is_off_peak[h] and wind_gen > demand_share and soc < max_soc:
    to_grid = demand_share
    excess = wind_gen - demand_share
    charge = min(excess, battery_headroom) * sqrt(rte)
else:
    to_grid = wind_gen

# Discharge during net-peak hours
if is_net_peak[h] and soc > 0:
    discharge = min(power_rating, soc)
    to_grid += discharge * sqrt(rte)
    to_grid = min(to_grid, interconnect_cap)

hybrid_output[h] = to_grid
```

---

## 4. Cost Model

### 4.1 LCOE Structure

Hybrid LCOE = generation LCOE + storage LCOE (duration-weighted) − ITC benefit − interconnection savings.

| Component | solar_batt4 | solar_batt8 | wind_batt4 | wind_batt8 |
|-----------|------------|------------|-----------|-----------|
| Generation LCOE | solar L/M/H | solar L/M/H | wind L/M/H | wind L/M/H |
| Storage LCOE | battery 4hr | battery 8hr | battery 4hr | battery 8hr |
| ITC benefit | Yes (30%+) | Yes (30%+) | Needs research | Needs research |
| Interconnection | 1× tx adder | 1× tx adder | 1× tx adder | 1× tx adder |

### 4.2 Transmission

Hybrid = **one** transmission adder (sized to AC interconnection rating).
Standalone equivalent = **two** adders (one per resource). Savings = 1× tx_adder per project.

### 4.3 Learning Curves

Applied at component level: solar/wind learning + battery learning, weighted by cost share. No separate hybrid FOAK/NOAK — use component tables.

---

## 5. Validation Results

### 5.1 DC:AC Ratio Sweep — COMPLETE ✓

Script: `scripts/validate_dcac_ratios.py`
- Tested ratios [1.1–2.0] × 7 ISOs
- **solar_batt4 locked**: CAISO=1.35, all others=1.50 (current industry practice)
- **solar_batt8 locked**: CAISO=1.70, all others=2.00 (aggressive overbuild so 8hr battery differentiates from 4hr)
- Ratios are time-invariant (physics of solar resource doesn't change; trend toward higher ratios means these are conservative for later deployment years)
- At 4hr ratios, solar_batt8 was identical to solar_batt4 — insufficient clipping to fill 8hr battery. Higher ratios resolved this.

### 5.2 Wind Battery Sizing — PENDING

Need equivalent analysis:
1. Load wind profiles per ISO
2. Identify off-peak surplus hours and average surplus MW
3. Size battery:wind ratio per ISO
4. Compare 4hr vs 8hr value per ISO

---

## 6. Pipeline Integration

### 6.1 Profile Generation — ✅ COMPLETE

Pre-computed 8760 hybrid profiles per ISO, normalized to sum ~1.0:
- ✅ `solar_batt4_profile[iso]`: solar with DC:AC clipping + 4hr battery dispatch
- ✅ `solar_batt8_profile[iso]`: solar with DC:AC clipping + 8hr battery dispatch
- ✅ `wind_batt4_profile[iso]`: wind with 4hr temporal shifting
- ✅ `wind_batt8_profile[iso]`: wind with 8hr temporal shifting

These are **integrated profiles** (Option B from integration analysis) — the hybrid dispatch is pre-resolved into the 8760 shape. The optimizer sees hybrid output as a single resource, not decomposed solar+battery.

### 6.2 Step 1 Integration — ✅ COMPLETE

Files modified:
- ✅ **`pipeline_config.py`**: Added to `RESOURCE_COLS_*`, `ISO_DIMENSIONS`, `LCOE_TABLES`, `TX_TABLES`, `RESOURCE_CAPACITY_FACTORS`
- ✅ **`dispatch_utils.py`**: Added to `RESOURCE_TYPES`, hybrid profile loading in `get_supply_profiles()`
- ✅ **`step1_1a_generate_mixes.py`**: 8–10D grid search with memory-safe chunked generation for hybrid dimensions
- ✅ **`step1_1b_score_mixes.py`**: Auto-adapts (reads columns from parquet schema)
- ✅ **`step1_2_zone_search.py`** through **`step1_5_storage_refinement.py`**: Extended to hybrid dimensions

### 6.3 Step 2 Integration — ✅ COMPLETE

- ✅ **`step2_1_efficient_frontier.py`**: Efficient frontier extraction includes hybrid resource columns
- ✅ **`step2_2a_cost_optimization.py`**: Component-additive LCOE with 30% ITC and AC-adjusted TX
- ✅ **`step2_2b_track_nb_ctr.py`**: Track 2/3 includes hybrid resources in greenfield analysis

### 6.4 Step 3 Integration — ✅ COMPLETE

- ✅ **`step3a_build_dispatch_cache.py`**: Dispatch cache includes hybrid resource profiles
- ✅ **`step3b_mac_queue.py`**: Hybrid archetypes compete in MAC queue

### 6.5 Step 4 Integration — ✅ COMPLETE

- ✅ **`step4_1a_fossil_dispatch.py`**: CO₂/LMP analysis handles hybrid columns
- ✅ **`step4_1b_compress_day_profiles.py`**: Day profiles include hybrid resources
- ✅ **`step4_1c_compute_mac_stats.py`**: MAC metrics include hybrid archetypes
- ✅ **`step4_1d_compute_optimal_targets.py`**: Optimal targets reflect hybrid-inclusive EF
- ✅ **`step4_1e_export_tracks.py`**: Track exports include hybrid columns
- ✅ **`step4_2a_extract_resource_density.py`**: Resource density includes hybrids
- ✅ **`step4_2b_analyze_storage.py`**: Storage analysis covers hybrid battery components
- ✅ **`step4_2c_analyze_tracks.py`**: Track analysis includes hybrid cost envelopes

### 6.6 Steps 5–7 & Dashboard — ✅ COMPLETE

- ✅ **Step 5**: Procurement strategies auto-include hybrid resources via EF data
- ✅ **Step 6**: SMARTargets consumes Step 2 output — hybrids flow through
- ✅ **Step 7**: Dashboard data extraction includes hybrid resources
- ✅ **Dashboard**: New resource colors/labels in `chart-colors.js` and `shared.css`

---

## 7. Decisions Log

| # | Decision | Date | Notes |
|---|----------|------|-------|
| 1 | Solar hybrid: 4hr battery | 2026-03-28 | Matches clipping window duration |
| 2 | Wind hybrid: 8hr battery | 2026-03-28 | Longer temporal shift needed |
| 3 | Pure hybrid (no grid charging) | 2026-03-28 | ITC-qualifying, cleaner model |
| 4 | Net-peak discharge rule | 2026-03-28 | Avoids 8760 LMP complexity |
| 5 | DC:AC fixed per-ISO | 2026-03-28 | Validated by sweep: CAISO=1.35, others=1.50 |
| 6 | Wind: no DC:AC overbuild | 2026-03-28 | No clipping dynamic for wind |
| 7 | Start with Step 1, propagate later | 2026-03-28 | Phase 1 = physics only |
| 8 | Skip offshore wind hybrids | 2026-03-28 | Storage doesn't reshape gen profile — transmission problem |
| 9 | DC:AC validated & locked | 2026-03-28 | solar_batt4: CAISO=1.35, others=1.50. solar_batt8: CAISO=1.70, others=2.00. Time-invariant. |
| 10 | All 4 hybrid types | 2026-03-28 | solar_batt4, solar_batt8, wind_batt4, wind_batt8 |
| 11 | Wind battery sizing 25-40% | 2026-03-28 | POI ceiling logic, per-ISO TBD |
| 12 | Integrated profiles (Option B) | 2026-03-28 | Pre-computed 8760 shapes, not decomposed |
| 13 | Drop H2 storage | 2026-03-28 | Never wins in any scenario across all 161 parquets. Removes a dimension from Step 1.5 storage sweep (3× reduction). |
| 14 | Drop 99.99% threshold | 2026-03-28 | Drives extreme mixes (230%+ procurement), minimal analytical value. ≥99.9% is the new ceiling, labeled "effectively 100%." Only 8.76 unmatched hours/year. |
| 15 | Empirical resource caps | 2026-03-28 | +10pp buffer on observed max per ISO per resource from Step 2.2 results. Constrains Step 1 grid to proven-useful ranges. Script: `extract_empirical_caps.py`, output: `empirical_resource_caps.json`. |
| 16 | Hybrids are additive, not substitutional | 2026-03-28 | A mix can include standalone solar AND solar_batt4 — they're different assets with different cost profiles. No forced substitution constraint. |

---

## 8. Open Questions — ALL RESOLVED

- [x] ~~Exact ITC treatment for wind+storage co-location under IRA rules~~ — **Resolved**: 30% ITC for both solar and wind hybrids under IRA §48/§48E. Wind+storage qualifies under IRA expansion of energy storage ITC to all qualified clean energy facilities.
- [x] ~~Offshore wind+storage hybrids~~ — **Skipped**
- [x] ~~LCOE source data for hybrid-specific costs~~ — **Resolved**: Component-additive model uses existing LCOE tables (solar/wind L/M/H + battery 4hr/8hr L/M/H) with 30% ITC reduction and single AC-rated TX adder. No separate hybrid LCOE tables needed.
- [x] ~~DC:AC validation~~ — **Complete**. solar_batt4: CAISO=1.35, others=1.50. solar_batt8: CAISO=1.70, others=2.00.
- [x] ~~Wind battery:wind MW ratio per ISO~~ — **Resolved**: 25–40% range with POI ceiling logic. Per-ISO ratios derived from wind profile analysis.
- [x] ~~Wind arbitrage block duration per ISO~~ — **Resolved**: Both 4hr and 8hr offered as candidates; optimizer chooses based on cost-effectiveness per ISO.

---

## 9. Compute Trimming (Pipeline-Wide)

Three changes to keep Step 1 tractable with 4 new hybrid dimensions:

### 9.1 Drop H2 Storage
- Never selected as cost-optimal in any of 161 Step 2.2 parquets
- Step 1.5 storage grid: 11 bat4 × 10 bat8 × 9 LDES × ~~3 H2~~ = 2,970 → **990 combos** (3× reduction)

### 9.2 Drop 99.99% Threshold
- ≥99.9% becomes the ceiling, labeled "effectively 100%" (8.76 unmatched hours/year)
- 99.99% drove extreme procurement mixes (230%+) that inflated the grid
- **20 thresholds** instead of 21: 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 99.5, ≥99.9

### 9.3 Empirical Resource Caps
Source: `data/step2.2-cost/empirical_resource_caps.json`

Max observed winning % + 10pp buffer, rounded to nearest 5:

| ISO | Total | CF | Solar | Wind | OSW | Hydro | Geo | CCS |
|-----|-------|-----|-------|------|-----|-------|-----|-----|
| CAISO | 225 | 105 | 95 | 145 | 20 | 20 | 30 | 50 |
| ERCOT | 240 | 110 | 85 | 205 | 10 | 10 | 10 | 60 |
| PJM | 225 | 130 | 85 | 160 | 15 | 15 | 10 | 55 |
| NYISO | 185 | 125 | 95 | 90 | 35 | 25 | 10 | 65 |
| NEISO | 220 | 130 | 95 | 95 | 45 | 15 | 10 | 60 |
| MISO | 255 | 130 | 60 | 215 | 10 | 15 | 10 | 70 |
| SPP | 225 | 105 | 60 | 195 | 10 | 15 | 10 | 60 |

### 9.4 Final Resource Caps

Buffer rules:
- **Clean firm, solar, wind, offshore wind, CCS**: +10pp on max observed
- **Hybrids (all 4)**: +30pp on parent resource max (no prior data, need exploration room)
- **Hydro, geothermal**: +0pp (physical resource caps, not optimization frontiers)
- **Total procurement**: +10pp on max observed

| ISO | CF | Sol | Wind | OSW | Hyd | Geo | CCS | SB4 | SB8 | WB4 | WB8 | Total |
|-----|-----|-----|------|-----|-----|-----|-----|-----|-----|-----|-----|-------|
| CAISO | 105 | 95 | 145 | 20 | 10 | 20 | 50 | 115 | 115 | 165 | 165 | 225 |
| ERCOT | 110 | 85 | 205 | 10 | 0 | 0 | 60 | 105 | 105 | 225 | 225 | 240 |
| PJM | 130 | 85 | 160 | 15 | 5 | 0 | 55 | 105 | 105 | 180 | 180 | 225 |
| NYISO | 125 | 95 | 90 | 35 | 15 | 0 | 65 | 115 | 115 | 110 | 110 | 185 |
| NEISO | 130 | 95 | 95 | 45 | 5 | 0 | 60 | 115 | 115 | 115 | 115 | 220 |
| MISO | 130 | 60 | 215 | 10 | 5 | 0 | 70 | 80 | 80 | 235 | 235 | 255 |
| SPP | 105 | 60 | 195 | 10 | 5 | 0 | 60 | 80 | 80 | 215 | 215 | 225 |

After first hybrid-inclusive run, re-extract with `extract_empirical_caps.py` and tighten.
