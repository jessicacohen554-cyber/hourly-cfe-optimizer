# Hybrid Resource Support: 1215 Market Sweep Audit & Fix Prompts

## Background

Prompts 1-5 added hybrid resource support (solar_batt4, solar_batt8, wind_batt4, wind_batt8) across the core pipeline: `pipeline_config.py`, `dispatch_utils.py`, and `market_simulation.py`. The deployment loop, profile loading, REC eligibility, and queue caps all correctly handle hybrids. However, five gaps remained in the LCOE evaluation, transmission cost, curtailment feedback, LCOE snapshot, and post-processing paths.

**Impact**: Without these fixes, hybrids were priced at a hardcoded fallback of $50/MWh (instead of their actual $90-180/MWh blended LCOE), received zero transmission cost, and were excluded from curtailment feedback and envelope metrics. This silently distorted the sweep's merit-order ranking in favor of hybrids.

**No new sweep parameters needed**: The existing 5 price sensitivity bundles already bundle `ren` and `batt` levels together. Hybrids read the existing `ren` level from `_price_sens` for both renewable and battery LCOE components. The 1,215 scenario count is unchanged.

---

## Issues & Fixes

### ISSUE 1 (Critical): `get_resource_lcoe()` returns wrong LCOE for hybrids

**File**: `market-simulator/scripts/market_simulation.py`
**Function**: `get_resource_lcoe()` (around line 2845)

**Problem**: Hybrids fall into the `else` branch which does `LCOE_TABLES.get(res, {})`. Since LCOE_TABLES has no hybrid keys, this returns `{}` and falls back to 50 $/MWh.

**Fix**: Add a new `elif res in HYBRID_TYPES:` branch before `else` that calls `get_hybrid_lcoe()` from pipeline_config with the correct VRE level from `_price_sens`.

<details>
<summary>Session Prompt</summary>

```
In market-simulator/scripts/market_simulation.py, find the get_resource_lcoe() function.
Before the final `else:` branch (which handles storage/generic resources via
LCOE_TABLES.get(res, {})), insert a new `elif` for hybrid resources:

1. Add these imports to the `from pipeline_config import (...)` block:
   - get_hybrid_lcoe
   - get_hybrid_tx
   - HYBRID_DC_AC_RATIOS
   - _HYBRID_PARENT_REN

2. Add this branch before the `else:` in get_resource_lcoe():

    elif res in HYBRID_TYPES:
        # Hybrid LCOE = (renewable LCOE / DC:AC) + battery LCOS * (1 - ITC)
        # ren and batt levels are bundled in sweep price sensitivities
        ps = (conditions or {}).get('_price_sens', {})
        vre_level = ps.get('ren', lcoe_level)  # fallback for non-sweep callers
        base = get_hybrid_lcoe(res, vre_level, vre_level, iso)
        # PTC delta adjustment for renewable component
        if conditions:
            parent = _HYBRID_PARENT_REN[res]
            user_ptc = conditions.get(f'ptc_{parent}', BASELINE_PTC_IN_LCOE.get(parent, 26.0))
            delta = user_ptc - BASELINE_PTC_IN_LCOE.get(parent, 26.0)
            base -= delta
        return max(0, base)

Verify: call get_resource_lcoe('solar_batt4', 'CAISO', 'Medium', {}, 'Medium', 2025)
and confirm the result is ~$120-160/MWh, not $50.
```

</details>

---

### ISSUE 2 (High): Transmission cost omits hybrids

**File**: `market-simulator/scripts/market_simulation.py`
**Location**: Merit-order candidate builder (around line 1903)

**Problem**: TX cost guard only includes `('solar', 'wind', 'clean_firm', 'offshore_wind', 'ccs_ccgt', 'geothermal')`. Hybrids get zero TX, understating their total cost.

**Fix**: Add `elif res in HYBRID_TYPES:` that calls `get_hybrid_tx()`.

<details>
<summary>Session Prompt</summary>

```
In market-simulator/scripts/market_simulation.py, find the transmission cost block
in the merit-order candidate builder (the block starting with:
    if res in ('solar', 'wind', 'clean_firm', 'offshore_wind', 'ccs_ccgt', 'geothermal'):
        ...
        lcoe += tx

After the closing `lcoe += tx` line, add:

        elif res in HYBRID_TYPES:
            tx = get_hybrid_tx(res, tx_level, iso)
            lcoe += tx

get_hybrid_tx is already imported from pipeline_config. It computes TX adjusted
for DC:AC ratio: parent_tx / DC_AC_RATIO.
```

</details>

---

### ISSUE 3 (Medium): Curtailment feedback excludes hybrids

**File**: `market-simulator/scripts/market_simulation.py`
**Location**: R10 curtailment feedback block (around line 1933)

**Problem**: Only `('solar', 'wind', 'offshore_wind')` get curtailment CF adjustment. Hybrids have a VRE component subject to curtailment, but their co-located battery partially mitigates it.

**Fix**: Add `elif res in HYBRID_TYPES:` with DC:AC-based mitigation factor.

<details>
<summary>Session Prompt</summary>

```
In market-simulator/scripts/market_simulation.py, find the R10 curtailment feedback
block:
    if res in ('solar', 'wind', 'offshore_wind') and curtailment_rate > 0:
        ...
        cf = effective_cf

After this block, add:

        elif res in HYBRID_TYPES and curtailment_rate > 0:
            # Hybrids partially mitigate curtailment via co-located storage
            dc_ac = HYBRID_DC_AC_RATIOS.get(res, {}).get(iso, 1.3)
            mitigation = 1.0 / dc_ac  # battery absorbs proportional to oversize
            effective_curt = curtailment_rate * mitigation
            effective_cf = cf * (1.0 - effective_curt)
            if effective_cf > 0:
                effective_lcoe = lcoe / (1.0 - effective_curt)
            else:
                effective_lcoe = float('inf')
            cf = effective_cf

The mitigation factor uses the inverse of DC:AC ratio — a DC:AC of 1.5 means
the battery can absorb ~33% of curtailed energy.
```

</details>

---

### ISSUE 4 (Low): `compute_lcoe_snapshot()` overwrites base tech with hybrid LCOE

**File**: `market-simulator/scripts/market_simulation.py`
**Function**: `compute_lcoe_snapshot()` (around line 2926)

**Problem**: `RESOURCE_TO_TECH` maps `solar_batt4→'solar'`, so `snapshot['solar']` gets overwritten by hybrid LCOE. The last hybrid processed wins, corrupting the base tech's reported LCOE.

**Fix**: Use the resource name directly as key for hybrids.

<details>
<summary>Session Prompt</summary>

```
In market-simulator/scripts/market_simulation.py, find compute_lcoe_snapshot().
Replace the line:
        tech = RESOURCE_TO_TECH.get(res, res)
        snapshot[tech] = round(lcoe, 2)

With:
        if res in HYBRID_TYPES:
            snapshot[res] = round(lcoe, 2)  # Keep hybrid as separate key
        else:
            tech = RESOURCE_TO_TECH.get(res, res)
            snapshot[tech] = round(lcoe, 2)

This ensures snapshot has separate entries like:
  {'solar': 60, 'solar_batt4': 135, 'solar_batt8': 128, ...}
instead of solar being overwritten to the last hybrid's value.
```

</details>

---

### ISSUE 5 (Medium): `extract_iso_sweep_data.py` ENVELOPE_METRICS missing hybrid columns

**File**: `market-simulator/scripts/extract_iso_sweep_data.py`
**Location**: `ENVELOPE_METRICS` list (around line 35)

**Problem**: Hardcoded resource mix list only includes 6 columns. Hybrid mix columns are produced by the upstream `flatten_year_result()` but excluded from envelope percentile calculations.

**Fix**: Add 4 hybrid entries.

<details>
<summary>Session Prompt</summary>

```
In market-simulator/scripts/extract_iso_sweep_data.py, find the ENVELOPE_METRICS
list. After the line:
    "mix_ccs_ccgt_twh", "mix_hydro_twh", "mix_clean_firm_twh",

Add:
    "mix_solar_batt4_twh", "mix_solar_batt8_twh",
    "mix_wind_batt4_twh", "mix_wind_batt8_twh",
```

</details>

---

## Files Modified

| File | Changes |
|------|---------|
| `market-simulator/scripts/market_simulation.py` | Added imports (get_hybrid_lcoe, get_hybrid_tx, HYBRID_DC_AC_RATIOS, _HYBRID_PARENT_REN); added hybrid LCOE branch in get_resource_lcoe(); added hybrid TX cost; added hybrid curtailment feedback; fixed compute_lcoe_snapshot() overwrite |
| `market-simulator/scripts/extract_iso_sweep_data.py` | Added 4 hybrid mix columns to ENVELOPE_METRICS |

## Files Confirmed OK (No Changes Needed)

| File | Reason |
|------|--------|
| `run_sweep_1215.py` | Dynamic resource extraction via `for res, twh in rmix.items()` — hybrids flow through automatically |
| `sensitivity_analysis.py` | Only analyzes aggregate metrics (clean_pct, cost_per_mwh, etc.), no resource-specific logic |
| `pipeline_config.py` | Already has all hybrid utilities (get_hybrid_lcoe, get_hybrid_tx, HYBRID_TYPES, capacity factors, capacity credits) |
| `dispatch_utils.py` | Already has HYBRID_TYPES, RESOURCE_TYPES_HYBRID, profile loading with include_hybrids flag |

## Verification

```bash
cd market-simulator/scripts

# Syntax check
python -c "import py_compile; py_compile.compile('market_simulation.py')"
python -c "import py_compile; py_compile.compile('extract_iso_sweep_data.py')"

# Import test
python -c "
from market_simulation import get_resource_lcoe, DEPLOYABLE_RESOURCES
from pipeline_config import get_hybrid_lcoe, get_hybrid_tx, HYBRID_TYPES
print('DEPLOYABLE_RESOURCES:', DEPLOYABLE_RESOURCES)
print('HYBRID_TYPES:', HYBRID_TYPES)
"

# LCOE spot-check (should be ~$120-160, not $50)
python -c "
from market_simulation import get_resource_lcoe
for h in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
    lcoe = get_resource_lcoe(h, 'CAISO', 'Medium', {}, 'Medium', 2025)
    print(f'{h}: {lcoe:.1f} $/MWh')
"

# TX spot-check (should be non-zero)
python -c "
from pipeline_config import get_hybrid_tx
for h in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
    tx = get_hybrid_tx(h, 'Medium', 'CAISO')
    print(f'{h} TX: {tx:.1f} $/MWh')
"
```
