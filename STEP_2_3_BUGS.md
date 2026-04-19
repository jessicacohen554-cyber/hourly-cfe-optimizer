# Step 2.3 Bug Analysis — Performance & Crash Issues

## Critical Bug: Hybrid VRE Double-Counting (Lines 2092-2109)

**Location:** `scripts/step_2_3_pathway_optimizer.py:2092-2109` in `_derive_delta_vintages()`

### The Problem
The hybrid VRE side (solar+battery, wind+battery) is missing the floor ratchet logic and adds the full `target_twh` every year without subtracting what was already built.

```python
# Line 2098: Gets existing capacity
existing_vre = ledger.capacity_twh(vre_res, year)

# Line 2102: But IGNORES it — sets inc_vre = target_twh directly
inc_vre = target_twh  # treat hybrid VRE as additive

# Line 2104-2109: Adds full target_twh as a new vintage EVERY YEAR
new_vintages.append(Vintage(
    resource=vre_res, cod_year=year,
    twh_per_year=max(0.0, inc_vre),  # ← Just target_twh, not a delta!
    ...
))
```

### Why It Crashes
Over 26 years (2025–2050), this creates:
- **26 redundant vintage entries** per hybrid resource per iso/pathway/endpoint run
- For 4 hybrid types (solar_batt4, solar_batt8, wind_batt4, wind_batt8) × 3 ISOs × 6 pathways × 10 endpoints = thousands of duplicate vintages
- Memory balloons quadratically; lists keep growing
- The ledger's capacity tracking becomes nonsense (cumulative reads see inflated values)

### The Fix
Replace lines 2102–2109 with floor-ratchet logic matching the energy_resources block:

```python
# Line 2098: existing_vre is the cumulative VRE capacity built so far
existing_vre = ledger.capacity_twh(vre_res, year)
inc_vre = max(0.0, target_twh - existing_vre)  # ← Floor ratchet: only new build
if inc_vre > 1e-6:  # ← Guard: skip if no new build needed
    new_vintages.append(Vintage(
        resource=vre_res, cod_year=year,
        twh_per_year=inc_vre,
        locked_lcoe=_vintage_lcoe_for_resource(vre_res, iso, year, config),
        tx_adder=transmission_adder(vre_res, iso, config.tx_level),
    ))
```

---

## Pattern Comparison: How It Should Work

### ✅ Energy Resources (Lines 2062–2083) — CORRECT
```python
existing_twh = ledger.capacity_twh(ledger_key, year)
new_twh = target_twh - existing_twh  # ← Subtracts existing
if new_twh <= 1e-6:
    continue
new_vintages.append(Vintage(..., twh_per_year=new_twh, ...))
```

### ❌ Hybrid VRE (Lines 2092–2109) — BUG
```python
existing_vre = ledger.capacity_twh(vre_res, year)  # ← Loaded but unused!
inc_vre = target_twh  # ← Missing subtraction: should be target_twh - existing_vre
new_vintages.append(Vintage(..., twh_per_year=max(0.0, inc_vre), ...))
```

### ✅ Hybrid Battery (Lines 2112–2120) — CORRECT
```python
existing_batt = ledger.capacity_twh(batt_res, year)
inc_batt = max(0.0, target_twh * 0.5 - existing_batt)  # ← Subtracts existing
if inc_batt > 1e-6:
    new_vintages.append(...)
```

### ✅ Standalone Storage (Lines 2129–2142) — CORRECT
```python
existing = ledger.capacity_twh(res, year)
new_twh = target_twh - existing  # ← Subtracts existing
if new_twh <= 1e-6:
    continue
new_vintages.append(...)
```

---

## Impact

- **Each run** (iso, pathway, endpoint) appends 26+ redundant hybrid VRE vintages
- **Ledger memory grows** from ~100 entries to 1,000+ entries per run
- **Cascading cost inflation**: The ledger's `operating_cost()` method sums all vintages; duplicates inflate annual costs
- **Runtime explosion**: Looping over inflated vintage lists in `annual_cost` aggregation (line 2562+)

---

## Testing the Fix
After applying: Run a single (iso, pathway, endpoint) and check that:
1. `len(ledger.vintages)` ≈ 75–100 (not 300+)
2. Hybrid resources (solar_batt4, etc.) have ≤1 new vintage per year with decreasing deltas
3. Annual cost values stabilize and match prior (correct) runs
