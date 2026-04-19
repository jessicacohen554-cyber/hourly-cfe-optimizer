# Hybrid Integration — QA/QC, Steps 5–7, Workflows & Documentation Prompts

These prompts continue from `hybrid-session-prompts.md` (Sessions A–F) and `hybrid-session-prompts-steps3-4.md` (Sessions G–K). They cover bug fixes in existing code, Steps 5–7 hybrid propagation, GitHub Actions updates, and methodology documentation.

## Dependencies
```
Prompt 1 (Bug-Fix QA)  →  Prompts 2, 3, 4 (parallel: Steps 5, 6, 7)
                           →  Prompt 5 (GitHub Actions) — after 2-4
                           →  Prompt 6 (Methodology Docs) — after 2-4
```

---

## Prompt 1: Bug-Fix & QA/QC for Existing Hybrid Code (Steps 1–4)

### Task
Fix bugs and inconsistencies found in the existing hybrid integration across Steps 1–4. Add missing hybrid resource entries to dashboard assets (CSS variables, Chart.js colors, labels). Verify end-to-end data flow from pipeline_config through step2_2a.

### Branch
Develop on branch `claude/hybrid-optimizer-integration-Uwnsp`

### Background

The hybrid integration across Sessions A–K added four co-located resource types (`solar_batt4`, `solar_batt8`, `wind_batt4`, `wind_batt8`) to the pipeline. A code review has identified several issues that need fixing before Steps 5–7 can consume hybrid data correctly.

### Bug 1: `RESOURCE_CAPACITY_FACTORS` Missing Hybrid Entries

**File**: `scripts/pipeline_config.py`
**Line**: ~605 (`RESOURCE_CAPACITY_FACTORS` dict)

**Problem**: The `RESOURCE_CAPACITY_FACTORS` dict has entries for all base resources (clean_firm, solar, wind, offshore_wind, ccs_ccgt, hydro, geothermal) but **no entries for hybrid types**. Multiple scripts look up CFs from this dict:

- `step2_2a_cost_optimization.py:410` — uses `RESOURCE_CAPACITY_FACTORS[_parent][iso]` for hybrids (correct workaround)
- `step4_1a_fossil_dispatch.py:107` — uses `.get(res, {}).get(iso, 0.30)` which returns **0.30 default** for hybrids
- `step6_1_smartargets.py:726` — same `.get()` pattern, returns **0.30 default**
- `step6_1_smartargets.py:1122` — returns **0.25 default**
- `procurement_utils.py:210` — maps resource key to CF key, would miss hybrids

**Fix**: Add hybrid entries to `RESOURCE_CAPACITY_FACTORS`. Hybrid CFs are higher than standalone because the co-located battery shifts generation to fill gaps:

```python
# After existing entries in RESOURCE_CAPACITY_FACTORS:
# Hybrid co-located — CF includes battery temporal shift (higher than standalone parent)
# Values derived from 8760 hybrid profile analysis (generate_hybrid_profiles.py)
'solar_batt4': {'CAISO': 0.376, 'ERCOT': 0.358, 'PJM': 0.340, 'NYISO': 0.275, 'NEISO': 0.283, 'MISO': 0.302, 'SPP': 0.306},
'solar_batt8': {'CAISO': 0.400, 'ERCOT': 0.385, 'PJM': 0.370, 'NYISO': 0.300, 'NEISO': 0.310, 'MISO': 0.330, 'SPP': 0.335},
'wind_batt4':  {'CAISO': 0.310, 'ERCOT': 0.380, 'PJM': 0.330, 'NYISO': 0.310, 'NEISO': 0.320, 'MISO': 0.370, 'SPP': 0.400},
'wind_batt8':  {'CAISO': 0.330, 'ERCOT': 0.400, 'PJM': 0.350, 'NYISO': 0.330, 'NEISO': 0.340, 'MISO': 0.390, 'SPP': 0.420},
```

**IMPORTANT**: These CF values should be **verified** from the actual hybrid profile NPZ files. Run this check:
```python
import numpy as np
for iso in ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']:
    data = np.load(f'data/hybrid_profiles/{iso}_hybrid_profiles.npz')
    for htype in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
        if htype in data:
            arr = data[htype]
            cf = arr.mean() / arr.max() if arr.max() > 0 else 0
            print(f"  {iso} {htype}: CF = {cf:.3f}")
```

If the NPZ files don't exist yet, use the values from `hybrid-design.md` Section 2.1 (validated sweep results):
- solar_batt4: CAISO=37.6%, ERCOT=35.8%, PJM=34.0%, NYISO=27.5%, NEISO=28.3%, MISO=30.2%, SPP=30.6%
- For solar_batt8 and wind hybrids, derive from the profile generation script output.

### Bug 2: Dashboard CSS Missing Hybrid Resource Variables

**File**: `dashboard/styles/shared.css`

**Problem**: The CSS file has variables for all base resources (`--solar`, `--wind`, `--nuclear`, etc.) but **no variables for hybrid types**. The CLAUDE.md canonical color table also doesn't list hybrid colors.

**Fix**: Add these CSS variables in the `:root` section (after the existing resource color variables):

```css
/* Hybrid co-located resources */
--solar-batt4: #E6890B;    /* Slightly darker solar gold */
--solar-batt4-t: rgba(230,137,11,0.12);
--solar-batt8: #CC7A0A;    /* Even darker solar gold */
--solar-batt8-t: rgba(204,122,10,0.12);
--wind-batt4: #1AA34E;     /* Slightly darker wind green */
--wind-batt4-t: rgba(26,163,78,0.12);
--wind-batt8: #158F42;     /* Even darker wind green */
--wind-batt8-t: rgba(21,143,66,0.12);
```

**Rationale**: Hybrid colors are darker shades of their parent renewable to create visual family grouping (solar family = gold spectrum, wind family = green spectrum).

### Bug 3: Chart.js Colors Missing Hybrid Entries

**File**: `dashboard/js/chart-colors.js`

**Problem**: `RESOURCE_COLORS` object has entries for all base resources and storage types but **no entries for hybrid types**.

**Fix**: Add to the `RESOURCE_COLORS` object:

```javascript
// Hybrid co-located resources
solarBatt4:  '#E6890B',
solarBatt8:  '#CC7A0A',
windBatt4:   '#1AA34E',
windBatt8:   '#158F42',
```

Also add transparent variants if the file follows that pattern (check if other resources have `_T` variants):
```javascript
solarBatt4T: 'rgba(230,137,11,0.12)',
solarBatt8T: 'rgba(204,122,10,0.12)',
windBatt4T:  'rgba(26,163,78,0.12)',
windBatt8T:  'rgba(21,143,66,0.12)',
```

### Bug 4: Verify `step2_2a` Hybrid Price Vector Completeness

**File**: `scripts/step2_2a_cost_optimization.py`

**Concern**: The price vector must have exactly `_N_COEFFS = 17` elements. Verify ALL places where the price vector/matrix is constructed:

1. **Line ~984**: `price_matrix = np.empty((B, _N_COEFFS), dtype=np.float64)` — Check that columns 13-16 (SB4, SB8, WB4, WB8) are populated in the inner loop.
2. **Search for all `price_matrix[` assignments** — verify indices 13-16 are set for every scenario.
3. **Search for `np.zeros(_N_COEFFS)` or `np.empty(..._N_COEFFS)`** — verify these produce length-17 arrays.

Run this verification:
```python
python -c "
import sys; sys.path.insert(0, 'scripts')
from step2_2a_cost_optimization import _N_COEFFS, _COL_SB4, _COL_SB8, _COL_WB4, _COL_WB8
assert _N_COEFFS == 17, f'Expected 17, got {_N_COEFFS}'
assert _COL_SB4 == 13
assert _COL_SB8 == 14
assert _COL_WB4 == 15
assert _COL_WB8 == 16
print('All coefficient indices verified OK')
"
```

### Bug 5: Verify Hybrid Profile NPZ Files Exist

**Location**: `data/hybrid_profiles/`

Before any pipeline run, verify all 7 ISOs have hybrid profile files:
```bash
for iso in CAISO ERCOT PJM NYISO NEISO MISO SPP; do
    f="data/hybrid_profiles/${iso}_hybrid_profiles.npz"
    if [ -f "$f" ]; then
        python -c "import numpy as np; d=np.load('$f'); print(f'  $iso: {list(d.keys())}, shapes: {[d[k].shape for k in d.keys()]}')"
    else
        echo "  MISSING: $f"
    fi
done
```

If any are missing, generate them:
```bash
python scripts/generate_hybrid_profiles.py
```

### Verification Checklist

1. `python -c "from pipeline_config import RESOURCE_CAPACITY_FACTORS; print({k: v for k, v in RESOURCE_CAPACITY_FACTORS.items() if 'batt' in k})"` — should show 4 hybrid entries with per-ISO CFs
2. `grep -c 'solar-batt4' dashboard/styles/shared.css` — should return ≥1
3. `grep -c 'solarBatt4' dashboard/js/chart-colors.js` — should return ≥1
4. `python -c "import sys; sys.path.insert(0,'scripts'); from step2_2a_cost_optimization import _N_COEFFS; assert _N_COEFFS == 17"` — should pass
5. `ls data/hybrid_profiles/*.npz | wc -l` — should be 7

### Commit
Commit with message: "Fix hybrid QA issues: add RESOURCE_CAPACITY_FACTORS entries, dashboard CSS/JS colors, verify price vector completeness"

---

## Prompt 2: Step 5 — Procurement Strategies Hybrid Integration

### Task
Update the procurement strategy scripts (Steps 5.2B–5.2E) and shared `procurement_utils.py` to propagate hybrid resource columns through EF data loading, resource cost computation, PPA pricing, and strategy output. After this session, all three procurement strategies (consequential, hourly, annual) will correctly account for hybrid co-located resources.

### Branch
Develop on branch `claude/hybrid-optimizer-integration-Uwnsp`

### Prerequisites
- Prompt 1 (Bug-Fix) must be complete
- Steps 1–4 hybrid integration must be complete (Sessions A–K)

### Background

**Step 5 computes procurement strategies** — how a corporate buyer should source clean energy to achieve CFE targets. Three strategies are modeled:
- **5.2B Consequential**: Cross-regional netting using MAC queue deployment order
- **5.2C Hourly**: Same-ISO hourly matching with EAC credits
- **5.2D Annual**: Annual matching (simplest, cheapest, least effective)

All three strategies consume **Step 2 EF data** (resource mixes at each threshold) and use `procurement_utils.py` for shared cost computation. Currently, every resource iteration loop and EF data extraction is hardcoded to 6–7 base resources — hybrids are invisible.

**Step 5.2E** (Wright's Law) is semi-independent — it models learning curves for technology deployment. Hybrid resources need entries in the technology category definitions.

**Key design point**: Hybrid resources in procurement strategies are treated like any other generation resource. A corporate buyer can procure hybrid PPAs (solar+storage or wind+storage) just like standalone solar/wind PPAs. The PPA price for a hybrid includes the component-additive LCOE (generation + co-located battery with ITC). The hourly delivery profile uses the pre-computed hybrid 8760 shape.

### Files to Modify

#### 1. `scripts/procurement_utils.py` (1601 lines) — Shared Foundation

##### a) Update `load_ef_resource_mix()` (line 1038–1084)

**Current code (line 1066-1069):**
```python
for col, key in [('mix_solar', 'solar'), ('mix_wind', 'wind'),
                 ('mix_clean_firm', 'clean_firm'), ('mix_ccs_ccgt', 'ccs'),
                 ('mix_hydro', 'hydro'), ('mix_offshore_wind', 'offshore_wind'),
                 ('mix_geothermal', 'geothermal')]:
```

**Fix**: Add hybrid columns:
```python
for col, key in [('mix_solar', 'solar'), ('mix_wind', 'wind'),
                 ('mix_clean_firm', 'clean_firm'), ('mix_ccs_ccgt', 'ccs'),
                 ('mix_hydro', 'hydro'), ('mix_offshore_wind', 'offshore_wind'),
                 ('mix_geothermal', 'geothermal'),
                 ('mix_solar_batt4', 'solar_batt4'), ('mix_solar_batt8', 'solar_batt8'),
                 ('mix_wind_batt4', 'wind_batt4'), ('mix_wind_batt8', 'wind_batt8')]:
    val = float(row.get(col, 0) or 0)
    if val > 0:
        mix[key] = val
```

**Note**: The column may be `solar_batt4` (no `mix_` prefix) depending on the parquet schema. Use the pattern:
```python
val = float(row.get(col, row.get(key, 0)) or 0)
```
to handle both naming conventions.

##### b) Update `_load_ef_storage_cache()` (line 1260–1290)

**Current code (line 1281-1286):** Extracts 6 hardcoded `mix_*` columns into the cache dict.

**Fix**: Add hybrid resource extraction:
```python
# After existing mix_hydro line:
# Hybrid resources (may or may not be present in older parquets)
for ht in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
    col = f'mix_{ht}'
    ef_storage[key][col] = float(row.get(col, row.get(ht, 0)) or 0)
```

##### c) Update existing clean resource sums (lines 498, 513, 685, 1412-1416)

These lines compute `existing_clean_pct` by summing `['clean_firm', 'solar', 'wind', 'offshore_wind', 'hydro']`. Hybrid resources are **NOT existing** (they're all new-build in the 2025 snapshot), so these sums are **CORRECT as-is**. Do NOT add hybrids to existing clean sums.

**BUT** — verify this assumption holds: search for any place where "total procurement" or "total clean supply" is computed. If a function sums `resource_mix` values to get total procurement, hybrids must be included. Specifically:

- Line 180: `compute_endogenous_cost()` — iterates `resource_mix.items()` which is dynamic, so hybrids will be included if they're in the dict. **OK as-is.**
- Line 1358: `rm = entry.get('resource_mix', {})` — dynamic, OK.
- Lines 1373-1416: Builds EF overlay — hardcoded to `['clean_firm', 'solar', 'wind', 'hydro']` — this needs hybrid entries added for the new-build portion.

**Fix for line 1373**:
```python
# Build EF overlay dict
overlay = {
    'clean_firm': ef['mix_clean_firm'],
    'solar': ef['mix_solar'],
    'wind': ef['mix_wind'],
    'offshore_wind': ef.get('mix_offshore_wind', 0),
    'ccs_ccgt': ef.get('mix_ccs_ccgt', 0),
    'hydro': ef['mix_hydro'],
}
# Add hybrid resources if present
for ht in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
    val = ef.get(f'mix_{ht}', ef.get(ht, 0))
    if val and val > 0:
        overlay[ht] = val
```

##### d) Add `get_resource_ppa_price()` support for hybrids (line 1087+)

Read this function to understand how PPA prices are computed per resource. Add hybrid PPA price computation:

```python
# For hybrid resources, PPA price = hybrid LCOE + TX
# Import from pipeline_config:
from pipeline_config import get_hybrid_lcoe, get_hybrid_tx

# In the function, add a hybrid branch:
if resource in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
    base_lcoe = get_hybrid_lcoe(resource, level, ppa_level, iso)
    tx = get_hybrid_tx(resource, level, iso)
    return base_lcoe + tx + ppa_premium
```

**Note**: Read `get_resource_ppa_price()` to understand the full signature and how `level` maps to sensitivity names. The hybrid LCOE function takes `(hybrid_type, ren_level, batt_level, iso)` — both `ren_level` and `batt_level` may map to the same `level` parameter or different ones. Check the existing code to see how standalone battery pricing uses the battery sensitivity level.

#### 2. `scripts/step5_2b_strategy_consequential.py` (569 lines)

##### a) Update `_RESOURCE_TO_PPA` (line 62-71)

Add hybrid PPA mappings:
```python
_RESOURCE_TO_PPA = {
    'solar': 'solar',
    'wind': 'wind',
    'clean_firm': 'nuclear_newbuild',
    'ccs_ccgt': 'ccs_45q_on',
    'offshore_wind': 'wind',
    'hydro': None,
    'battery': None,
    'ldes': None,
    # Hybrid co-located — PPA covers bundled generation+storage
    'solar_batt4': 'solar_batt4',
    'solar_batt8': 'solar_batt8',
    'wind_batt4': 'wind_batt4',
    'wind_batt8': 'wind_batt8',
}
```

##### b) Update MAC queue resource iteration (lines 170-195, 270-295)

The script reads the consequential MAC queue JSON (`consequential_queue.json`) and iterates over `delta_resources` in each queue entry. Hybrid resources will appear in `delta_resources` automatically if Step 3B (Session I) added them to the MAC queue output.

**Verify**: Read lines 170-195 to see how `delta_resources` is iterated. The code does:
```python
for resource, twh in entry['delta_resources'].items():
    ppa_resource = _RESOURCE_TO_PPA.get(resource)
```

If `resource` is `'solar_batt4'` and `_RESOURCE_TO_PPA` has the mapping, the PPA pricing will work. **But**: the PPA price lookup (`get_resource_ppa_price()`) must support hybrid types (see procurement_utils fix d above).

**No code change needed here** if `_RESOURCE_TO_PPA` and `get_resource_ppa_price()` are updated. The iteration is dynamic.

#### 3. `scripts/step5_2c_strategy_hourly.py` (797 lines)

##### a) Update `HOURLY_MIX_TEMPLATE` (line 107-120)

Add hybrid resource fractions. At lower thresholds, hybrids are optional; at higher thresholds they become more valuable:

```python
HOURLY_MIX_TEMPLATE = {
    50:    {'solar': 0.35, 'wind': 0.30, 'firm': 0.10, 'battery': 0.08, 'ldes': 0.02, 'uprate': 0.05, 'solar_batt4': 0.05, 'wind_batt4': 0.05},
    70:    {'solar': 0.28, 'wind': 0.25, 'firm': 0.15, 'battery': 0.08, 'ldes': 0.04, 'uprate': 0.05, 'solar_batt4': 0.08, 'wind_batt4': 0.07},
    85:    {'solar': 0.20, 'wind': 0.20, 'firm': 0.22, 'battery': 0.08, 'ldes': 0.06, 'uprate': 0.05, 'solar_batt4': 0.10, 'wind_batt4': 0.09},
    90:    {'solar': 0.18, 'wind': 0.18, 'firm': 0.25, 'battery': 0.08, 'ldes': 0.08, 'uprate': 0.05, 'solar_batt4': 0.10, 'wind_batt4': 0.08},
    95:    {'solar': 0.14, 'wind': 0.14, 'firm': 0.30, 'battery': 0.09, 'ldes': 0.10, 'uprate': 0.05, 'solar_batt4': 0.10, 'wind_batt4': 0.08},
    99:    {'solar': 0.12, 'wind': 0.12, 'firm': 0.33, 'battery': 0.09, 'ldes': 0.12, 'uprate': 0.05, 'solar_batt4': 0.09, 'wind_batt4': 0.08},
    99.5:  {'solar': 0.11, 'wind': 0.11, 'firm': 0.34, 'battery': 0.09, 'ldes': 0.13, 'uprate': 0.05, 'solar_batt4': 0.09, 'wind_batt4': 0.08},
    99.9:  {'solar': 0.10, 'wind': 0.10, 'firm': 0.35, 'battery': 0.09, 'ldes': 0.14, 'uprate': 0.05, 'solar_batt4': 0.09, 'wind_batt4': 0.08},
}
```

**IMPORTANT**: These are **fallback templates** only used when EF data is unavailable. The preferred path is `load_ef_resource_mix()` which will include hybrids from the EF parquets directly (if procurement_utils fix a is applied). The templates must still sum to ~1.0 across all resources.

**Alternative approach**: Instead of hardcoding hybrid fractions in the template, make `get_resource_mix_fractions()` pull from EF data first and only fall back to the template. This avoids maintaining two sets of numbers. Check if the function already does this (line 172-179). If so, the template is truly a last resort and the hybrid fractions can be zero or omitted (since real EF data will have them).

##### b) Update resource breakdown construction

Search for `resource_breakdown` dict construction in the three strategy variants (2A, 2B, 2C). These build per-resource output dicts with `twh`, `price`, `cost` fields. Hybrid resources must be included when the EF mix has them.

**Pattern to follow**: Read how existing resources are added to `resource_breakdown`, then add the same pattern for hybrid types. Use `get_resource_ppa_price()` for hybrid pricing (after procurement_utils fix d).

##### c) Update hourly profile blending

If the script blends hourly generation profiles (e.g., to compute hourly matching), the blended profile must include hybrid profiles. Search for calls to `get_supply_profiles()` or manual profile loading. If the script loads EIA-930 profiles and blends them by resource share, add hybrid profile loading:

```python
from dispatch_utils import _load_hybrid_profiles

# When building the hourly supply profile:
if 'solar_batt4' in ef_mix and ef_mix['solar_batt4'] > 0:
    hybrid_profiles = _load_hybrid_profiles(iso)
    for ht in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
        if ht in ef_mix and ef_mix[ht] > 0 and ht in hybrid_profiles:
            hourly_supply += (ef_mix[ht] / 100.0) * hybrid_profiles[ht] * total_demand
```

#### 4. `scripts/step5_2d_strategy_annual.py` (816 lines)

**Integration pattern**: Same as step5_2c — update `resource_breakdown` dicts. Annual strategies are simpler because they don't do hourly matching — they just sum total procurement. The key changes:

1. **Resource cost computation**: When computing cost per resource, include hybrid costs using `get_hybrid_lcoe()` + `get_hybrid_tx()`.
2. **Resource breakdown output**: Add hybrid entries to the output dict.
3. **EF mix loading**: Uses `load_ef_resource_mix()` — already fixed in procurement_utils.

Read the file to identify the exact locations of resource iteration and cost computation, then extend them.

#### 5. `scripts/step5_2e_wrights_law_curves.py` (957 lines)

##### a) Find `TECH_CATEGORIES` or equivalent technology definitions

Search for technology category definitions (solar, wind, battery, etc.). Add hybrid categories:

```python
# Hybrid technologies follow parent renewable + battery learning curves
# Battery component has its own Wright's Law curve; renewable component is at scale
'solar_batt4': {'learning_rate': 0.18, 'category': 'solar_storage'},
'solar_batt8': {'learning_rate': 0.18, 'category': 'solar_storage'},
'wind_batt4':  {'learning_rate': 0.15, 'category': 'wind_storage'},
'wind_batt8':  {'learning_rate': 0.15, 'category': 'wind_storage'},
```

**Note**: Read the file first to understand how technology categories are structured. The learning rate applies to the battery component — solar/wind are already at scale with minimal further cost reduction. The hybrid learning curve is dominated by the storage component.

##### b) Update deployment tracking

If the script tracks cumulative deployment by technology, add hybrid resource tracking from EF/MAC queue data. Hybrid deployments should be counted separately from standalone solar/wind.

### Critical Pitfalls

1. **Existing clean sum**: Do NOT add hybrids to `existing_clean_pct` sums — there is no existing hybrid capacity in 2025.
2. **PPA pricing**: Hybrid PPA price must use `get_hybrid_lcoe()` (component-additive with ITC), not standalone solar/wind LCOE.
3. **Double-counting**: A mix with both standalone solar and solar_batt4 is valid — they're different assets. Don't treat them as substitutes in procurement accounting.
4. **Template fractions**: If updating `HOURLY_MIX_TEMPLATE`, all fractions must sum to 1.0.
5. **Backward compatibility**: Scripts must work with pre-hybrid EF parquets (hybrid columns absent). Always use `.get()` with zero defaults.

### Verification

1. `python -c "import sys; sys.path.insert(0,'scripts'); from procurement_utils import load_ef_resource_mix; m = load_ef_resource_mix('CAISO', 90); print([k for k in m if 'batt' in k])"` — should show hybrid keys if EF parquets have them
2. Run step5_2b on one ISO: `python scripts/step5_2b_strategy_consequential.py --iso SPP`
3. Run step5_2c on one ISO: `python scripts/step5_2c_strategy_hourly.py --iso SPP`
4. Run step5_2d on one ISO: `python scripts/step5_2d_strategy_annual.py --iso SPP`
5. Check output JSONs for hybrid resource entries in `resource_mix` dicts
6. Verify non-hybrid ISOs produce identical output

### Commit
Commit with message: "Add hybrid resource support to Step 5 procurement strategies — PPA pricing, EF loading, mix templates, Wright's Law"

---

## Prompt 3: Step 6 — SMARTargets & Policy Hybrid Integration

### Task
Update Step 6 scripts (SMARTargets modeling, dashboard data export, IPP fleet modeling, nuclear retirement) to propagate hybrid resource columns through EF data loading, resource iteration, capacity deployment, and output parquets/JS. After this session, SMARTargets modeling will correctly account for hybrid co-located resources in pathway planning and policy analysis.

### Branch
Develop on branch `claude/hybrid-optimizer-integration-Uwnsp`

### Prerequisites
- Prompt 1 (Bug-Fix) must be complete
- Steps 1–4 hybrid integration must be complete

### Background

**Step 6.1** (`step6_1_smartargets.py`, 2312 lines) is the largest and most complex script. It runs parametric sweeps of SBTi-aligned decarbonization pathways — for each ISO × scenario, it selects the cost-optimal resource mix at each threshold over a 25-year timeline. The script has several hardcoded resource lists that need hybrid entries.

**Step 6.1B** (`step6_1b_dashboard_data.py`, 262 lines) converts SMARTargets parquet output to dashboard JavaScript. Hardcoded `mix_cols` list at line 52-53.

**Step 6.2A** (`step6_2a_ipp_smartargets.py`, 1000 lines) models IPP fleet behavior under SMARTargets scenarios. Has hardcoded technology categories (`CLEAN_FUELS`, `LEARNING_RATE`, `CLEAN_CF`, etc.).

**Step 6.2B** (`step6_2b_nuclear_retirement.py`, 1003 lines) analyzes nuclear stranding risk. This is nuclear-specific — it consumes LMP/capacity signals but does NOT iterate over resource types. **Minimal changes needed** — hybrid resources only affect this script indirectly through LMP prices (already handled by Step 4.1A).

### Files to Modify

#### 1. `scripts/step6_1_smartargets.py` (2312 lines) — Main SMARTargets Engine

##### a) Update `RESOURCE_TO_TECH` (line 268-273)

Add hybrid technology mappings:
```python
RESOURCE_TO_TECH = {
    'clean_firm': 'nuclear', 'solar': 'solar', 'wind': 'wind',
    'offshore_wind': 'offshore_wind', 'ccs_ccgt': 'ccs', 'hydro': 'hydro',
    'battery': 'battery', 'battery8': 'battery8', 'ldes': 'ldes', 'h2': 'h2',
    'geothermal': 'geothermal',
    # Hybrid co-located — parent tech for Wright's Law + storage learning curve
    'solar_batt4': 'solar_batt4', 'solar_batt8': 'solar_batt8',
    'wind_batt4': 'wind_batt4', 'wind_batt8': 'wind_batt8',
}
```

##### b) Update `resource_pcts` dict construction (line 1076-1083)

**Current code:**
```python
resource_pcts = {
    'clean_firm': float(row.get('mix_clean_firm', 0)),
    'solar': float(row.get('mix_solar', 0)),
    'wind': float(row.get('mix_wind', 0)),
    'offshore_wind': float(row.get('mix_offshore_wind', 0)),
    'ccs_ccgt': float(row.get('mix_ccs_ccgt', 0)),
    'hydro': float(row.get('mix_hydro', 0)),
}
```

**Fix**: Add hybrid resource entries:
```python
resource_pcts = {
    'clean_firm': float(row.get('mix_clean_firm', 0)),
    'solar': float(row.get('mix_solar', 0)),
    'wind': float(row.get('mix_wind', 0)),
    'offshore_wind': float(row.get('mix_offshore_wind', 0)),
    'ccs_ccgt': float(row.get('mix_ccs_ccgt', 0)),
    'hydro': float(row.get('mix_hydro', 0)),
}
# Add hybrid resources if present in EF data
for ht in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
    val = float(row.get(f'mix_{ht}', row.get(ht, 0)) or 0)
    if val > 0:
        resource_pcts[ht] = val
```

##### c) Update output parquet columns (lines 2020-2022)

**Current code:**
```python
for res in ['clean_firm', 'solar', 'wind', 'offshore_wind',
            'ccs_ccgt', 'hydro', 'battery', 'ldes']:
    row[f'mix_{res}_twh'] = yr['resource_mix_twh'].get(res, 0)
```

**Fix**: Add hybrid resources:
```python
for res in ['clean_firm', 'solar', 'wind', 'offshore_wind',
            'ccs_ccgt', 'hydro', 'battery', 'ldes',
            'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
    row[f'mix_{res}_twh'] = yr['resource_mix_twh'].get(res, 0)
```

**Also update the duplicate at line 2286-2288** — same pattern, same fix.

##### d) Update `twh_from_resource_pcts()` or equivalent function

Search for where `resource_pcts` dict is converted to TWh values. This function multiplies percentages by demand_twh. Hybrid resources in the dict will be included automatically IF the function iterates over `resource_pcts.items()` dynamically. If it has a hardcoded resource list, extend it.

##### e) Update capacity deployment tracking

Search for where cumulative capacity (GW) is computed per resource. Lines ~720-730 use `RESOURCE_CAPACITY_FACTORS` to convert TWh → GW:
```python
cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.30)
```

After Prompt 1 fixes `RESOURCE_CAPACITY_FACTORS`, this will return correct CFs for hybrids. But verify that the calling loop includes hybrid resources:
```python
# If the loop is:
for res in resource_pcts:
    cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.30)
    gw = twh / (cf * 8.76)
```
This is dynamic and will include hybrids. But if the loop iterates a hardcoded list, extend it.

##### f) Update cost computation for hybrid resources

Search for where per-resource costs are computed in the pathway planner. This likely uses LCOE tables from `pipeline_config`. Add hybrid cost blocks:

```python
from pipeline_config import get_hybrid_lcoe, get_hybrid_tx

# In the cost computation section:
for ht in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
    if ht in resource_pcts and resource_pcts[ht] > 0:
        ht_lcoe = get_hybrid_lcoe(ht, ren_level, batt_level, iso)
        ht_tx = get_hybrid_tx(ht, tx_level, iso)
        ht_cost = (resource_pcts[ht] / 100.0) * demand_twh * (ht_lcoe + ht_tx)
        total_cost += ht_cost
```

**Note**: Read the actual cost computation code to understand the exact structure. SMARTargets may use its own cost model or import from pipeline_config. Match the existing pattern.

#### 2. `scripts/step6_1b_dashboard_data.py` (262 lines)

##### a) Update `mix_cols` (line 52-53)

**Current code:**
```python
mix_cols = ['mix_clean_firm_twh', 'mix_solar_twh', 'mix_wind_twh', 'mix_offshore_wind_twh',
            'mix_ccs_ccgt_twh', 'mix_hydro_twh', 'mix_battery_twh', 'mix_ldes_twh']
```

**Fix**: Add hybrid columns:
```python
mix_cols = ['mix_clean_firm_twh', 'mix_solar_twh', 'mix_wind_twh', 'mix_offshore_wind_twh',
            'mix_ccs_ccgt_twh', 'mix_hydro_twh', 'mix_battery_twh', 'mix_ldes_twh',
            'mix_solar_batt4_twh', 'mix_solar_batt8_twh', 'mix_wind_batt4_twh', 'mix_wind_batt8_twh']
```

**Also update line 132** — there's a duplicate `mix_cols` definition. Apply the same fix.

##### b) Handle missing columns gracefully

The existing code (line 54-56) already handles missing columns:
```python
for col in mix_cols:
    if col in iso_df.columns:
        data[col] = iso_df[col].round(2).tolist()
```

This will gracefully skip hybrid columns if the input parquet doesn't have them. **No additional backward compatibility code needed.**

#### 3. `scripts/step6_2a_ipp_smartargets.py` (1000 lines) — IPP Fleet Modeling

##### a) Update `CLEAN_FUELS` (line 144)

```python
CLEAN_FUELS = {'nuclear', 'solar', 'wind', 'battery', 'hydro', 'geothermal',
               'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8'}
```

##### b) Update `CLEAN_CF` (lines 103-106)

Add hybrid capacity factors. These should match `RESOURCE_CAPACITY_FACTORS` from `pipeline_config.py`:
```python
'solar_batt4': {'CAISO': 0.376, 'ERCOT': 0.358, 'PJM': 0.340, 'NYISO': 0.275,
                'NEISO': 0.283, 'MISO': 0.302, 'SPP': 0.306},
'solar_batt8': {'CAISO': 0.400, 'ERCOT': 0.385, 'PJM': 0.370, 'NYISO': 0.300,
                'NEISO': 0.310, 'MISO': 0.330, 'SPP': 0.335},
'wind_batt4':  {'CAISO': 0.310, 'ERCOT': 0.380, 'PJM': 0.330, 'NYISO': 0.310,
                'NEISO': 0.320, 'MISO': 0.370, 'SPP': 0.400},
'wind_batt8':  {'CAISO': 0.330, 'ERCOT': 0.400, 'PJM': 0.350, 'NYISO': 0.330,
                'NEISO': 0.340, 'MISO': 0.390, 'SPP': 0.420},
```

**Better approach**: Import `RESOURCE_CAPACITY_FACTORS` from `pipeline_config` instead of duplicating values:
```python
from pipeline_config import RESOURCE_CAPACITY_FACTORS
# Then use RESOURCE_CAPACITY_FACTORS.get(tech, {}).get(iso, 0.25) everywhere
```

##### c) Update `LEARNING_RATE`, `NEW_LCOE_2025`, `CUMULATIVE_GW_2025`, `NATIONAL_DEPLOY_GW_YR` (lines 131-140)

Add hybrid entries:
```python
NEW_LCOE_2025 = {'solar': 60, 'wind': 50, 'battery': 10,
                 'solar_batt4': 85, 'solar_batt8': 90, 'wind_batt4': 75, 'wind_batt8': 80}
# Hybrid learning rates dominated by battery component
LEARNING_RATE = {'solar': 0.24, 'wind': 0.15, 'battery': 0.18,
                 'solar_batt4': 0.18, 'solar_batt8': 0.18, 'wind_batt4': 0.15, 'wind_batt8': 0.15}
CUMULATIVE_GW_2025 = {'solar': 180, 'wind': 160, 'battery': 35,
                      'solar_batt4': 15, 'solar_batt8': 3, 'wind_batt4': 8, 'wind_batt8': 2}
NATIONAL_DEPLOY_GW_YR = {'solar': 40, 'wind': 15, 'battery': 12,
                         'solar_batt4': 10, 'solar_batt8': 3, 'wind_batt4': 5, 'wind_batt8': 2}
```

**IMPORTANT**: These deployment numbers are rough estimates. Verify against NREL ATB 2024 or LBNL Q2/Q3 2024 interconnection queue data for hybrid project counts.

##### d) Update technology iteration loops (lines 464, 482-492)

```python
# Line 464 — add hybrid to VRE category:
category = 'VRE' if tech in ('solar', 'wind', 'battery',
                              'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8') else 'Firm'

# Lines 482-492 — add hybrid CF lookup:
for tech in ['solar', 'wind', 'battery', 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
    # ... existing pattern for CF lookup ...
    if tech.startswith('solar_batt'):
        cfs = [CLEAN_CF.get(tech, {}).get(iso, 0.35) for iso in isos]
    elif tech.startswith('wind_batt'):
        cfs = [CLEAN_CF.get(tech, {}).get(iso, 0.35) for iso in isos]
```

#### 4. `scripts/step6_2b_nuclear_retirement.py` (1003 lines) — Minimal Changes

This script is nuclear-specific. Hybrid resources only affect it through:
1. LMP prices (Step 4.1A output — already handles hybrids)
2. Capacity market signals (hybrid deployments may crowd out nuclear)

**Only change needed**: If the script reads resource mixes from Step 2.2 parquets to compute clean supply percentages, ensure it includes hybrid columns in the sum. Search for any `sum(...)` over resource percentages.

If no resource iteration is found, **no changes needed** — the script operates on LMP/capacity signals that already incorporate hybrid effects from upstream steps.

### Critical Pitfalls

1. **RESOURCE_CAPACITY_FACTORS dependency**: Step 6.1 uses `.get(res, {}).get(iso, 0.30)` — after Prompt 1 fix, this returns correct CFs. But step6_2a has its own `CLEAN_CF` dict that must also be updated.
2. **Cost model consistency**: Step 6.1 may use different cost computation than pipeline_config. Verify the cost model matches `get_hybrid_lcoe()` + `get_hybrid_tx()`.
3. **Duplicate `mix_cols`**: Step 6.1b has TWO separate `mix_cols` definitions (lines 52 and 132). Both must be updated.
4. **Output schema change**: Adding hybrid columns to SMARTargets parquets changes the schema consumed by step6_1b (dashboard JS). Both must be updated together.
5. **Learning curve data**: The hybrid learning rates and deployment numbers in step6_2a are estimates. Flag to user for review against published data.

### Verification

1. `python -c "import sys; sys.path.insert(0,'scripts'); from step6_1_smartargets import RESOURCE_TO_TECH; print({k: v for k, v in RESOURCE_TO_TECH.items() if 'batt' in k})"` — 4 entries
2. Run step6_1 on one scenario: `python scripts/step6_1_smartargets.py --scenarios R1 --isos SPP`
3. Check output: `python -c "import pandas as pd; df = pd.read_parquet('data/step6-smartargets/smartargets_R1.parquet'); print([c for c in df.columns if 'batt' in c])"`
4. Run step6_1b: `python scripts/step6_1b_dashboard_data.py`
5. Run step6_2a: `python scripts/step6_2a_ipp_smartargets.py`

### Commit
Commit with message: "Add hybrid resource support to Step 6 — SMARTargets pathways, dashboard export, IPP fleet modeling"

---

## Prompt 4: Step 7 — Dashboard Data Aggregation Hybrid Integration

### Task
Update all Step 7 scripts to include hybrid resource columns in the dashboard JavaScript data files. This is the final pipeline step — it aggregates all upstream results into `dashboard/js/shared-data.js` and other JS files consumed by the interactive dashboard. After this session, the dashboard will have hybrid resource data available for visualization.

### Branch
Develop on branch `claude/hybrid-optimizer-integration-Uwnsp`

### Prerequisites
- Prompt 1 (Bug-Fix) complete — dashboard CSS/JS has hybrid colors
- Steps 1–6 hybrid integration complete — upstream parquets contain hybrid columns

### Background

Step 7 is the data extraction layer between the Python pipeline and the JavaScript dashboard. It reads parquets from Steps 2–6 and writes pre-computed JavaScript data files. **Every hardcoded resource list in Step 7 must be extended** or the dashboard will silently drop hybrid data.

**Critical output files:**
- `dashboard/js/shared-data.js` — Primary data file consumed by dashboard.html, index.html, and other pages
- `dashboard/js/no-regrets-data.js` — No-regrets resource investment analysis
- `dashboard/js/optimal-target-data.js` — Optimal CFE target per ISO
- Various deployment/dispatch/comparison JS files

### Files to Modify

#### 1. `scripts/step7_1a_generate_shared_data.py` (1898 lines) — PRIMARY DATA EXTRACTION

This is the **most critical file**. It generates `shared-data.js` with all the data the dashboard needs.

##### a) Update `RESOURCES` constant (line 42)

**Current:**
```python
RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
```

**Fix:**
```python
RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro',
             'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']
```

##### b) Update `MATCHED_RESOURCES` constant (line 43)

**Current:**
```python
MATCHED_RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro', 'battery', 'battery8', 'ldes', 'h2']
```

**Fix:**
```python
MATCHED_RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro',
                     'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8',
                     'battery', 'battery8', 'ldes', 'h2']
```

##### c) Update `WYN_RESOURCES` (line 669)

**Current:**
```python
WYN_RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro', 'battery', 'battery8', 'ldes', 'h2']
```

**Fix:** Add 4 hybrid types after hydro (before standalone storage):
```python
WYN_RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro',
                 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8',
                 'battery', 'battery8', 'ldes', 'h2']
```

##### d) Update JS `MIX_RESOURCES` constant (line 831)

**Current:**
```python
lines.append("const MIX_RESOURCES = ['clean_firm', 'geothermal', 'hydro', 'ccs_ccgt', 'offshore_wind', 'wind', 'solar', 'battery', 'battery8', 'ldes', 'h2'];")
```

**Fix:**
```python
lines.append("const MIX_RESOURCES = ['clean_firm', 'geothermal', 'hydro', 'ccs_ccgt', 'offshore_wind', 'wind', 'solar', 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8', 'battery', 'battery8', 'ldes', 'h2'];")
```

##### e) Update JS `MIX_LABELS_MAP` (lines 833-845)

Add hybrid labels after the solar entry:
```python
lines.append("    solar_batt4:   'Solar+Batt (4hr)',")
lines.append("    solar_batt8:   'Solar+Batt (8hr)',")
lines.append("    wind_batt4:    'Wind+Batt (4hr)',")
lines.append("    wind_batt8:    'Wind+Batt (8hr)',")
```

##### f) Update JS `MIX_COLORS` (lines 847-859)

Add hybrid colors (matching CSS variables from Prompt 1):
```python
lines.append("    solar_batt4:   { fill: 'rgba(230,137,11,0.50)',  border: '#E6890B' },")
lines.append("    solar_batt8:   { fill: 'rgba(204,122,10,0.50)',  border: '#CC7A0A' },")
lines.append("    wind_batt4:    { fill: 'rgba(26,163,78,0.50)',   border: '#1AA34E' },")
lines.append("    wind_batt8:    { fill: 'rgba(21,143,66,0.50)',   border: '#158F42' },")
```

##### g) Update `RESOURCE_CAPS` (lines 862-868)

Hybrid resources don't have physical resource caps (they're deployment-limited, not resource-limited). No new entries needed — the existing caps apply to the solar/wind land area. But if the dashboard uses `RESOURCE_CAPS` to validate resource percentages, it should know about hybrid types.

**Add a comment** in the JS output:
```python
lines.append("    // Hybrid resources: no separate physical cap — limited by parent renewable resource")
```

##### h) Update parquet column extraction (lines 380-530)

**RESOURCE_MIX_DATA extraction** (lines 380-530): This section reads Step 2.2 parquets and extracts resource percentages per ISO × threshold × scenario.

Search for `iso_data = {r: [] for r in RESOURCES}` (line 421). Since `RESOURCES` is now extended with hybrids, this will automatically create hybrid entries. **But verify** that the parquet columns are named correctly:

```python
# Line ~501: Extracting from parquet
for res in RESOURCES:
    col = f'mix_{res}'  # or just 'res' — check naming
    iso_data[res].append(float(row.get(col, 0)))
```

If the column name is `mix_solar_batt4`, the extraction works. If it's `solar_batt4`, you need to check which convention the Step 2.2 parquets use. Read one:
```python
python -c "import pyarrow.parquet as pq; s = pq.read_schema('data/step2.2-cost/step_2_2a_CAISO.parquet'); print([c for c in s.names if 'batt' in c])"
```

##### i) Update dispatch data extraction (lines 575-660)

**matched/surplus extraction**: This iterates over `MATCHED_RESOURCES` (line 618) to get per-resource dispatch data from the dispatch cache. Since `MATCHED_RESOURCES` is extended, hybrid matched/surplus data will be included.

**Verify**: The dispatch cache entry keys are `matched_solar_batt4`, `surplus_solar_batt4`, etc. (from Step 3A). The extraction code does:
```python
for res in MATCHED_RESOURCES:
    matched_key = f'matched_{res}'
```

This will work for hybrids. **No code change needed** beyond updating the constant.

##### j) Update WYN_RESOURCE_COSTS extraction (lines 665-680)

This extracts per-resource cost data. Since `WYN_RESOURCES` is extended, hybrid cost data will be included automatically IF the source parquets have cost columns for hybrids (they should, from Step 2.2 — verify with the column check above).

##### k) Update RESOURCE_MIX_DATA JS output (lines 1051-1120)

The JS output writes `const RESOURCE_MIX_DATA = {...}` with nested resource arrays. The loop at line 1063:
```python
all_keys = RESOURCES + ['battery', 'battery8', 'ldes', 'h2', 'battery_cap', 'battery8_cap', 'ldes_cap', 'h2_cap']
```

**Fix**: Add hybrid keys:
```python
all_keys = RESOURCES + ['battery', 'battery8', 'ldes', 'h2', 'battery_cap', 'battery8_cap', 'ldes_cap', 'h2_cap']
# RESOURCES already includes hybrids after fix (a), so all_keys automatically includes them
```

**Verify this is the case** — if `RESOURCES` is used at line 42 and `all_keys` at line 1063 extends `RESOURCES`, the fix is automatic.

#### 2. `scripts/step7_1b_extract_deployment_data.py` (474 lines)

##### a) Update `RESOURCE_ALIASES` (line 32)

**Current**: Maps resource names to display names. Read the current dict and add hybrid entries:
```python
'solar_batt4': 'Solar+Batt4',
'solar_batt8': 'Solar+Batt8',
'wind_batt4': 'Wind+Batt4',
'wind_batt8': 'Wind+Batt8',
```

##### b) Update deployment data extraction

Search for where deployment data is extracted from MAC queue or EF parquets. Add hybrid resource extraction. The script likely iterates `delta_resources` from the MAC queue JSON — hybrids will appear there if Step 3B added them.

#### 3. `scripts/step7_1c_generate_foak_noak.py` (190 lines)

Add hybrid technology entries to FOAK/NOAK learning curve data. Search for technology definitions:

```python
# Hybrid FOAK/NOAK — battery component drives learning
'solar_batt4': {'foak': 95, 'noak': 70, 'lr': 0.18},  # $/MWh (gen+storage combined)
'solar_batt8': {'foak': 105, 'noak': 75, 'lr': 0.18},
'wind_batt4': {'foak': 85, 'noak': 60, 'lr': 0.15},
'wind_batt8': {'foak': 95, 'noak': 65, 'lr': 0.15},
```

**Note**: Read the existing technology definitions to match the data format.

#### 4. `scripts/step7_1e_dispatch_deployment.py` (490 lines)

Search for supply matrix construction — if the script calls `get_supply_profiles()` or `build_supply_matrix()`, pass `include_hybrids=True`:

```python
supply_profiles = get_supply_profiles(iso, gen_profiles, include_hybrids=True)
rtypes = RESOURCE_TYPES_HYBRID
supply_matrix = build_supply_matrix(supply_profiles, resource_types=rtypes)
```

Also update any resource iteration loops to include hybrid types.

#### 5. `scripts/step7_1f_extract_hourly_comparison.py` (123 lines)

##### a) Update `RESOURCE_KEYS` (line 32)

**Current:**
```python
RESOURCE_KEYS = ['solar', 'wind', 'clean_firm', 'nuclear_uprate', 'battery', 'ldes',
                 'ccs_ccgt', 'hydro', 'offshore_wind', 'geothermal']
```

**Fix:**
```python
RESOURCE_KEYS = ['solar', 'wind', 'clean_firm', 'nuclear_uprate', 'battery', 'ldes',
                 'ccs_ccgt', 'hydro', 'offshore_wind', 'geothermal',
                 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']
```

#### 6. `scripts/step7_1g_extract_use_case_data.py` (963 lines)

Search for hardcoded resource lists and extend them with hybrid types. This script extracts use-case-specific data slices.

#### 7. `scripts/step7_2_extract_no_regrets.py` (389 lines)

##### a) Update resource constants (lines 54-56)

**Current:**
```python
MIX_RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
STORAGE_RESOURCES = ['battery', 'battery8', 'ldes']
ALL_RESOURCES = MIX_RESOURCES + STORAGE_RESOURCES
```

**Fix:**
```python
MIX_RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro',
                 'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']
STORAGE_RESOURCES = ['battery', 'battery8', 'ldes']
ALL_RESOURCES = MIX_RESOURCES + STORAGE_RESOURCES
```

##### b) Update no-regrets analysis (lines 252-290)

The script computes floor/consensus/average resource allocations across thresholds. The iteration at line 252-257:
```python
for res in MIX_RESOURCES:
    col = f'mix_{res}'
    ...
```

Since `MIX_RESOURCES` is extended, hybrid columns will be included automatically. **But verify** the column naming convention matches the Step 2.2 parquets.

### Critical Pitfalls

1. **Column naming**: Step 7 scripts use `mix_{resource}` column names. Verify this matches Step 2.2 output: is it `mix_solar_batt4` or `solar_batt4`? Run: `python -c "import pyarrow.parquet as pq; s = pq.read_schema('data/step2.2-cost/step_2_2a_CAISO.parquet'); print([c for c in s.names if 'batt' in c])"` before modifying code.

2. **JS constant ordering**: `MIX_RESOURCES` JS array ordering matters for stacking charts. Place hybrid types between their parent renewable and standalone storage:
   `clean_firm, geothermal, hydro, ccs_ccgt, offshore_wind, wind, wind_batt4, wind_batt8, solar, solar_batt4, solar_batt8, battery, battery8, ldes, h2`
   This groups the solar family and wind family visually in stacked charts.

3. **Color consistency**: The colors in `MIX_COLORS` JS must match `RESOURCE_COLORS` in `chart-colors.js` (Prompt 1) and CSS variables in `shared.css` (Prompt 1).

4. **Backward compatibility**: If upstream parquets don't have hybrid columns yet (pre-hybrid pipeline run), all scripts must handle missing columns gracefully with zero defaults.

5. **JSON serialization**: When writing JS data files, hybrid resource values may be 0 for many ISOs at low thresholds. Consider omitting zero entries to keep file size down, matching the existing pattern.

### Verification

1. Run step7_1a: `python scripts/step7_1a_generate_shared_data.py`
2. Check shared-data.js: `grep -c 'solar_batt4' dashboard/js/shared-data.js` — should be ≥5
3. Check MIX_RESOURCES JS: `grep 'MIX_RESOURCES' dashboard/js/shared-data.js` — should include hybrid types
4. Check MIX_LABELS_MAP: `grep 'solar_batt4' dashboard/js/shared-data.js` — should have label
5. Check MIX_COLORS: `grep 'E6890B' dashboard/js/shared-data.js` — should have hybrid color
6. Run step7_2: `python scripts/step7_2_extract_no_regrets.py`
7. Check no-regrets JS: `grep 'solar_batt4' dashboard/js/no-regrets-data.js`
8. Verify non-hybrid output is unchanged for ISOs without hybrid data

### Commit
Commit with message: "Add hybrid resource support to Step 7 dashboard data — shared-data.js, no-regrets, deployment, hourly comparison"

---

## Prompt 5: GitHub Actions Workflow Updates & Cleanup

### Task
Update GitHub Actions workflows to ensure hybrid resources flow correctly through the CI/CD pipeline. Verify auto-detection works for Steps 1.2–2.2, add `--hybrid` flag where needed, clean up threshold references, and update the workflow README.

### Branch
Develop on branch `claude/hybrid-optimizer-integration-Uwnsp`

### Prerequisites
- Prompts 1–4 must be complete (all pipeline scripts updated)

### Background

**Current state**: The `--hybrid` flag is only explicitly passed in `step1-1-scored-database.yml` (lines 105, 124). Steps 1.2–1.5 use auto-detection from input parquet schema (`'solar_batt4' in schema.names`). Steps 2.1+ don't have explicit hybrid flags — they read columns dynamically from parquets.

**Auto-detection verified**: step1_2_zone_search.py (line 780), step1_5_storage_refinement.py (line 412/1096), step2_1_efficient_frontier.py, step3a_build_dispatch_cache.py, step3b_mac_queue.py all detect hybrids from input parquet schema. **No `--hybrid` flag is needed** for these workflows — auto-detection is sufficient.

**Key concern**: If a Step 1.1 run produces hybrid parquets, ALL downstream steps must handle them. Since auto-detection is in place, the main risk is that a **non-hybrid** Step 1.1 run (without `--hybrid`) feeds into a pipeline that expects hybrids. The workflow should always pass `--hybrid` to Step 1.1 to ensure hybrid columns are always present.

### Files to Modify

#### 1. `.github/workflows/step1-1-scored-database.yml`

**Current state**: Already passes `--hybrid` flag (lines 105, 124). **No changes needed.**

Verify the flag is set unconditionally (not behind a conditional or input toggle):
```yaml
# Line 105 should be:
FLAGS="--hybrid"
# NOT:
if [ "${{ inputs.hybrid }}" == "true" ]; then FLAGS="--hybrid"; fi
```

If it's behind a conditional, make it unconditional — hybrid is now a permanent feature.

#### 2. `.github/workflows/step1-2-3-zone-floor.yml`

**Current invocations** (no `--hybrid` flag):
```yaml
python scripts/step1_2_zone_search.py $ARGS    # line 195
python scripts/step1_3_floor_aware_pfs.py --iso ${{ matrix.iso }}    # line 214
python scripts/step1_4_fine_grid_pfs.py --iso ${{ matrix.iso }}      # line 233
```

**No changes needed** — all three scripts auto-detect hybrids from input parquet schema. Adding `--hybrid` would be redundant but harmless. **Decision**: Leave as-is (auto-detection is more robust than flags since it adapts to the actual data).

**Optional enhancement**: Add a log line to confirm hybrid detection:
```yaml
- name: Run step1_2
  run: |
    echo "Checking for hybrid columns in coarse cache..."
    python -c "import pyarrow.parquet as pq; s = pq.read_schema('data/step1-pfs/${{ matrix.iso }}_coarse_cache.parquet'); hyb = 'solar_batt4' in s.names; print(f'Hybrid mode: {hyb}')"
    python scripts/step1_2_zone_search.py $ARGS
```

#### 3. `.github/workflows/step1-5-storage-refinement.yml`

**No changes needed** — auto-detection via `_detect_hybrids()` at line 412.

#### 4. `.github/workflows/step2-1-efficient-frontier.yml` through `.github/workflows/step7-dashboard-data.yml`

**No changes needed for hybrid** — all downstream steps read columns dynamically from input parquets.

#### 5. `.github/workflows/README.md`

Update the workflow documentation to mention hybrid resource support:

1. In the pipeline overview section, add:
```markdown
### Hybrid Resource Support
All pipeline steps support hybrid co-located resources (solar_batt4, solar_batt8, wind_batt4, wind_batt8).
- Step 1.1: Generates hybrid mixes via `--hybrid` flag (always enabled)
- Steps 1.2–2.2: Auto-detect hybrids from input parquet schema (`solar_batt4` column presence)
- Steps 3–7: Dynamically read all columns from input parquets — hybrids flow through automatically
```

2. Update any threshold lists that reference 21 thresholds to mention the current count (20 if 99.99 is dropped, 21 if not).

3. Add a note about the `--hybrid` flag permanence:
```markdown
**Note**: The `--hybrid` flag in Step 1.1 is now a permanent feature. Do not remove it.
All downstream steps auto-detect hybrid columns from parquet schema.
```

#### 6. Stale Workflow Cleanup

Based on the code review, **no stale workflows were found**. All 32 workflows reference scripts that exist and use current patterns. Minor items to verify:

1. **`step5-scenarios.yml`** is mentioned in README but doesn't exist in workflows directory. If it was renamed to `step5-procurement.yml`, update the README reference.

2. **Threshold references**: Search all workflow files for `99.99`:
```bash
grep -r '99.99' .github/workflows/
```
If the 99.99% threshold has been dropped (hybrid-design.md Decision #14), update any workflow that hardcodes threshold lists. **BUT** — only do this if the user confirms the threshold change has been implemented. It may be a future task.

3. **H2 storage references**: Search for `h2_dispatch_pct` or `H2` in workflow files. If H2 has been dropped (Decision #13), remove H2-specific processing steps. Again, confirm with user first.

### Verification

1. Verify `--hybrid` is permanent in step1-1: `grep -n 'hybrid' .github/workflows/step1-1-scored-database.yml`
2. Verify auto-detection scripts work: `python -c "import pyarrow.parquet as pq; s = pq.read_schema('data/step1-pfs/SPP_coarse_cache.parquet'); print('solar_batt4' in s.names)"`
3. Check README is updated: `grep -c 'Hybrid' .github/workflows/README.md`
4. Ensure no stale references: `for f in .github/workflows/*.yml; do echo "=== $f ==="; grep -n '99.99' "$f" 2>/dev/null; done`

### Commit
Commit with message: "Update GitHub Actions workflow docs for hybrid resources, verify auto-detection, clean up stale references"

---

## Prompt 6: Methodology Documentation & Dashboard Asset Updates

### Task
Update the optimizer methodology HTML page to document hybrid co-located resources. Update dashboard CSS and JS assets with hybrid resource definitions. Update CLAUDE.md canonical color tables. After this session, the methodology page will explain the hybrid resource model, and all dashboard pages will have access to hybrid colors, labels, and CSS variables.

### Branch
Develop on branch `claude/hybrid-optimizer-integration-Uwnsp`

### Prerequisites
- Prompt 1 (Bug-Fix) should be complete (basic CSS/JS colors added)
- Prompts 2–4 (Steps 5–7) should be complete or in progress

### Background

The optimizer methodology page (`dashboard/optimizer_methodology.html`, 2765 lines) is the technical specifications document for the optimizer. It currently has **zero mention of hybrid co-located resources** — "hybrid" only appears in the context of the coal retirement model (unrelated).

Section 4 "Resources Modeled" (line 647-710) lists 9 resource types with colored dots. It needs 4 new hybrid entries. The cost model section needs a new subsection explaining the component-additive LCOE formula. The pipeline section needs a note about hybrid dimensions.

### Files to Modify

#### 1. `dashboard/optimizer_methodology.html` (2765 lines)

##### a) Add hybrid resources to Section 4 "Resources Modeled" (after line 690, before the closing `</div>`)

Insert 4 new resource items using the CSS variables from Prompt 1:

```html
<div class="resource-item">
    <div class="resource-dot" style="background: var(--solar-batt4);"></div>
    <div><strong>Solar + Battery (4hr)</strong> &mdash; Co-located solar with 4-hour battery; clipping recovery + peak shifting</div>
</div>
<div class="resource-item">
    <div class="resource-dot" style="background: var(--solar-batt8);"></div>
    <div><strong>Solar + Battery (8hr)</strong> &mdash; Co-located solar with 8-hour battery; extended temporal shifting</div>
</div>
<div class="resource-item">
    <div class="resource-dot" style="background: var(--wind-batt4);"></div>
    <div><strong>Wind + Battery (4hr)</strong> &mdash; Co-located wind with 4-hour battery; short temporal shifting</div>
</div>
<div class="resource-item">
    <div class="resource-dot" style="background: var(--wind-batt8);"></div>
    <div><strong>Wind + Battery (8hr)</strong> &mdash; Co-located wind with 8-hour battery; deep overnight-to-peak shifting</div>
</div>
```

##### b) Add hybrid description paragraph after the resource grid (after line 698)

Insert after the existing resource description paragraph:

```html
<h3 id="hybrid-resources">Hybrid Co-Located Resources</h3>

<p>
    The optimizer includes four <strong>hybrid co-located</strong> resource types that combine
    renewable generation with on-site battery storage at a single point of interconnection.
    These differ fundamentally from standalone renewables + grid-connected storage:
</p>

<ul>
    <li><strong>Single interconnection</strong> &mdash; shared grid connection (IQ position, TX adder)</li>
    <li><strong>No grid charging</strong> &mdash; battery only charges from co-located generation (ITC-qualifying)</li>
    <li><strong>Combined output capped</strong> at the AC interconnection rating</li>
    <li><strong>Pre-computed 8760-hour profiles</strong> &mdash; co-located battery dispatch is resolved into the generation shape before optimization</li>
</ul>

<h4>Solar Hybrids (solar_batt4, solar_batt8)</h4>

<p>
    Solar panels are installed at a DC:AC ratio exceeding 1.0 (1.35&ndash;2.0 depending on ISO
    and battery duration). During peak sun, generation exceeding the AC interconnection limit is
    "clipped" and charges the co-located battery. The battery discharges during net-peak hours,
    shifting solar energy to when it's most valuable.
</p>

<table class="data-table" style="max-width: 600px;">
    <thead>
        <tr>
            <th>Parameter</th><th>solar_batt4</th><th>solar_batt8</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>Battery duration</td><td>4 hours</td><td>8 hours</td></tr>
        <tr><td>Round-trip efficiency</td><td>85%</td><td>85%</td></tr>
        <tr><td>DC:AC ratio (CAISO)</td><td>1.35</td><td>1.70</td></tr>
        <tr><td>DC:AC ratio (others)</td><td>1.50</td><td>2.00</td></tr>
        <tr><td>Grid charging</td><td>None</td><td>None</td></tr>
        <tr><td>Discharge trigger</td><td>Top 4 net-peak hours/day</td><td>Top 8 net-peak hours/day</td></tr>
    </tbody>
</table>

<h4>Wind Hybrids (wind_batt4, wind_batt8)</h4>

<p>
    Wind turbines produce AC power with no clipping dynamic. The co-located battery charges
    from off-peak wind surplus (when generation exceeds demand share) and discharges during
    net-peak hours. The value proposition is temporal shifting &mdash; moving wind generation
    from low-value overnight hours to high-value peak hours.
</p>

<table class="data-table" style="max-width: 600px;">
    <thead>
        <tr>
            <th>Parameter</th><th>wind_batt4</th><th>wind_batt8</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>Battery duration</td><td>4 hours</td><td>8 hours</td></tr>
        <tr><td>Round-trip efficiency</td><td>85%</td><td>85%</td></tr>
        <tr><td>Battery:wind MW ratio</td><td>25&ndash;40%</td><td>25&ndash;40%</td></tr>
        <tr><td>Grid charging</td><td>None</td><td>None</td></tr>
        <tr><td>Discharge trigger</td><td>Top 4 net-peak hours/day</td><td>Top 8 net-peak hours/day</td></tr>
    </tbody>
</table>

<h4>Hybrid LCOE Model</h4>

<p>
    Hybrid LCOE uses a <strong>component-additive</strong> formula:
</p>

<pre style="background: #f1f5f9; padding: 16px; border-radius: 8px; font-family: var(--font-mono); font-size: 0.85rem; line-height: 1.6; overflow-x: auto;">
hybrid_lcoe = adjusted_renewable_lcoe + battery_lcos × (1 − ITC_rate) + AC_adjusted_tx

Where:
  adjusted_renewable_lcoe = parent LCOE / DC:AC ratio  (solar hybrids)
                          = parent LCOE                 (wind hybrids)
  battery_lcos = co-located battery $/MWh discharged (NREL ATB 2024)
  ITC_rate = 0.30 (IRA §48/§48E, both solar and wind hybrids qualify)
  AC_adjusted_tx = parent TX adder / DC:AC ratio  (solar hybrids)
                 = parent TX adder                 (wind hybrids)
</pre>

<p>
    The 30% ITC applies to both solar+storage and wind+storage co-located facilities under
    the IRA expansion of energy storage ITC to all qualified clean energy facilities. The
    DC:AC ratio adjustment for solar reflects that overbuilding DC capacity proportionally
    increases capacity factor, lowering the effective $/MWh. Wind turbines are AC machines
    with no overbuild dynamic, so no DC:AC adjustment applies.
</p>
```

##### c) Update Section 3 "Model Scope & Scale" (around line 629)

In the "Sensitivity Analysis" subsection, add a note:
```html
<p>
    The model evaluates <strong>8&ndash;10 resource dimensions</strong> per ISO: 4&ndash;6 base
    resources (solar, wind, clean firm, hydro, CCS, offshore wind/geothermal where applicable)
    plus 4 hybrid co-located types (solar+battery 4hr/8hr, wind+battery 4hr/8hr). Storage
    dispatch (standalone battery, LDES) adds additional sweep dimensions.
</p>
```

##### d) Update Section 5 "Analytical Pipeline" Phase 1 (around line 718)

Add a note about hybrid dimensions in the adaptive physics search:
```html
<p>
    With hybrid resources, the grid search operates in 8&ndash;10 dimensions: base resources
    plus four hybrid co-located types. Empirical resource caps constrain each dimension to
    proven-useful ranges (existing max + buffer), preventing combinatorial explosion while
    ensuring the optimizer explores hybrid-inclusive frontiers.
</p>
```

#### 2. `dashboard/styles/shared.css` — Verify & Complete

Prompt 1 should have added the CSS variables. **Verify** they exist:
```bash
grep 'solar-batt4' dashboard/styles/shared.css
```

If missing, add to the `:root` section:
```css
--solar-batt4: #E6890B;
--solar-batt4-t: rgba(230,137,11,0.12);
--solar-batt8: #CC7A0A;
--solar-batt8-t: rgba(204,122,10,0.12);
--wind-batt4: #1AA34E;
--wind-batt4-t: rgba(26,163,78,0.12);
--wind-batt8: #158F42;
--wind-batt8-t: rgba(21,143,66,0.12);
```

#### 3. `dashboard/js/chart-colors.js` — Verify & Complete

Prompt 1 should have added the Chart.js constants. **Verify** they exist:
```bash
grep 'solarBatt4' dashboard/js/chart-colors.js
```

If missing, add to the `RESOURCE_COLORS` object:
```javascript
solarBatt4:  '#E6890B',
solarBatt8:  '#CC7A0A',
windBatt4:   '#1AA34E',
windBatt8:   '#158F42',
solarBatt4T: 'rgba(230,137,11,0.12)',
solarBatt8T: 'rgba(204,122,10,0.12)',
windBatt4T:  'rgba(26,163,78,0.12)',
windBatt8T:  'rgba(21,143,66,0.12)',
```

#### 4. `CLAUDE.md` — Update Canonical Color Tables

Update the "Canonical Resource Colors" table in the "Dashboard CSS/HTML Standards" section. Add 4 new rows:

```markdown
| Solar+Batt 4hr | `--solar-batt4` | `#E6890B` | `RESOURCE_COLORS.solarBatt4` |
| Solar+Batt 8hr | `--solar-batt8` | `#CC7A0A` | `RESOURCE_COLORS.solarBatt8` |
| Wind+Batt 4hr | `--wind-batt4` | `#1AA34E` | `RESOURCE_COLORS.windBatt4` |
| Wind+Batt 8hr | `--wind-batt8` | `#158F42` | `RESOURCE_COLORS.windBatt8` |
```

Also add hybrid types to the "Key Design Principles" section where resources are listed.

### Critical Pitfalls

1. **CSS variable naming**: Use kebab-case (`--solar-batt4`) in CSS, camelCase (`solarBatt4`) in JS, and snake_case (`solar_batt4`) in Python. These are three different naming conventions for the same resource.

2. **HTML escaping**: In the methodology HTML, use `&ndash;` for ranges (e.g., "4&ndash;6"), `&mdash;` for em-dashes, and `&#8322;` for subscript 2 (CO₂).

3. **Existing page style**: The methodology page uses shared.css classes. Use `class="data-table"` for tables, `class="resource-item"` + `class="resource-dot"` for the resource grid. Do NOT create custom styles.

4. **Don't duplicate content**: The research paper (`research_paper.html`) may also need updating, but that's a separate task. Only update `optimizer_methodology.html` in this prompt.

5. **Verify CSS variables exist** before referencing them in HTML. If Prompt 1 hasn't been executed yet, add the CSS variables first.

### Verification

1. Open `optimizer_methodology.html` in a browser — verify:
   - 4 new resource dots appear in Section 4
   - Hybrid description section renders correctly
   - Tables are styled properly
   - No broken layout
2. `grep -c 'solar-batt4' dashboard/styles/shared.css` — ≥2
3. `grep -c 'solarBatt4' dashboard/js/chart-colors.js` — ≥2
4. `grep -c 'Solar+Batt' CLAUDE.md` — ≥4
5. No console errors when loading methodology page

### Commit
Commit with message: "Document hybrid resources in methodology page, update dashboard CSS/JS colors, update CLAUDE.md color tables"

---

## End-to-End Verification (After All Prompts)

After executing all 6 prompts, perform this end-to-end verification:

### 1. Syntax Check — All Modified Scripts
```bash
cd scripts
for f in pipeline_config.py dispatch_utils.py \
         step5_2b_strategy_consequential.py step5_2c_strategy_hourly.py \
         step5_2d_strategy_annual.py step5_2e_wrights_law_curves.py \
         procurement_utils.py step6_1_smartargets.py step6_1b_dashboard_data.py \
         step6_2a_ipp_smartargets.py step7_1a_generate_shared_data.py \
         step7_1b_extract_deployment_data.py step7_1f_extract_hourly_comparison.py \
         step7_2_extract_no_regrets.py; do
    python -c "import py_compile; py_compile.compile('$f')" && echo "OK: $f" || echo "FAIL: $f"
done
```

### 2. Import Chain Verification
```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
from pipeline_config import HYBRID_TYPES, get_hybrid_lcoe, get_hybrid_tx, RESOURCE_CAPACITY_FACTORS
from dispatch_utils import RESOURCE_TYPES_HYBRID, get_supply_profiles
print('HYBRID_TYPES:', HYBRID_TYPES)
print('Hybrid CFs present:', all(ht in RESOURCE_CAPACITY_FACTORS for ht in HYBRID_TYPES))
print('get_hybrid_lcoe test:', get_hybrid_lcoe('solar_batt4', 'Medium', 'Medium', 'CAISO'))
print('get_hybrid_tx test:', get_hybrid_tx('solar_batt4', 'Medium', 'CAISO'))
"
```

### 3. Dashboard Asset Verification
```bash
echo "=== CSS Variables ==="
grep -c 'solar-batt4' dashboard/styles/shared.css
echo "=== Chart.js Colors ==="
grep -c 'solarBatt4' dashboard/js/chart-colors.js
echo "=== shared-data.js MIX_RESOURCES ==="
grep 'MIX_RESOURCES' dashboard/js/shared-data.js | head -1
echo "=== Methodology Page ==="
grep -c 'Solar + Battery' dashboard/optimizer_methodology.html
```

### 4. E2E Pipeline Run (One ISO)
```bash
# Run Steps 5-7 on SPP (cheapest ISO) to verify hybrid data flows end-to-end
python scripts/step5_2c_strategy_hourly.py --iso SPP
python scripts/step7_1a_generate_shared_data.py
python scripts/step7_2_extract_no_regrets.py

# Verify hybrid data in outputs
python -c "
import json
# Check shared-data.js has hybrid entries
with open('dashboard/js/shared-data.js') as f:
    content = f.read()
    assert 'solar_batt4' in content, 'Missing solar_batt4 in shared-data.js'
    print('shared-data.js: OK')

# Check no-regrets has hybrid entries
with open('dashboard/js/no-regrets-data.js') as f:
    content = f.read()
    has_hybrid = 'solar_batt4' in content
    print(f'no-regrets-data.js: {\"OK\" if has_hybrid else \"No hybrid data (expected if no hybrid EF data)\"} ')
"
```
