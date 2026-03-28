# Hybrid Integration — Session Prompts

These prompts are designed for execution in separate Claude Code sessions. Run Session A first, then B/C/D in parallel, then E last.

## Dependencies
```
Session A (foundation) → Sessions B, C, D (parallel) → Session E (cost optimization)
```

---

## Session A: Foundation — `pipeline_config.py` + `dispatch_utils.py`

### Task
Add hybrid resource constants, LCOE tables, TX tables, and profile support to the shared configuration and dispatch utilities. This is the foundation that all downstream hybrid integration depends on.

### Branch
Develop on branch `claude/integrate-hybrid-resources-2pk5J`

### Background Context

We have 4 hybrid co-located resource types being integrated into the pipeline:
- `solar_batt4`: Solar + 4hr battery (DC:AC clipping recovery)
- `solar_batt8`: Solar + 8hr battery (extended clipping + deeper temporal shift)
- `wind_batt4`: Wind + 4hr battery (short temporal shifting, no DC:AC overbuild)
- `wind_batt8`: Wind + 8hr battery (deep overnight-to-peak shifting, no DC:AC overbuild)

**Confirmed design decisions:**
1. **LCOE**: Component-additive — Hybrid LCOE = Adjusted Renewable LCOE + Co-located Battery LCOE (with 30% ITC)
2. **Toggles**: Paired with existing — renewable component follows Renewable toggle, battery component follows Battery toggle. No new toggles, no scenario key change.
3. **Transmission**: AC-rating-adjusted — solar hybrid TX = parent solar TX / DC:AC ratio. Wind hybrids have no DC:AC overbuild, so wind hybrid TX = wind TX (no adjustment).
4. **ITC**: Both solar+storage and wind+storage get 30% ITC on battery component.

### Key Design Details

**Solar hybrid DC:AC ratios** (validated, from `scripts/generate_hybrid_profiles.py`):
- `solar_batt4`: CAISO=1.35, all others=1.50
- `solar_batt8`: CAISO=1.70, all others=2.00

**Wind hybrids**: No DC:AC overbuild (wind turbines are AC machines). Battery MW ratio = 25-40% of wind nameplate. Already defined in `WIND_BATT_MW_RATIO` in `scripts/generate_hybrid_profiles.py`. Wind hybrid profile = wind generation + temporal shifting via battery. The battery charges from off-peak wind surplus (not from clipping).

**Hybrid LCOE formula (component-additive):**

For solar hybrids:
```
solar_batt4_lcoe(ren_level, batt_level, iso) =
    LCOE_TABLES['solar'][ren_level][iso] / DC_AC_RATIO_4HR[iso]   # CF-adjusted solar $/MWh
  + HYBRID_BATT_LCOE_4HR[batt_level][iso] * (1 - ITC_RATE)        # ITC-discounted battery component
```
The solar LCOE is divided by DC:AC ratio because overbuilding increases CF proportionally, lowering effective $/MWh. The battery component uses a separate table (`HYBRID_BATT_LCOE`) because co-located battery costs differ from standalone grid storage (which is in annualized $/yr per % of demand). Hybrid battery LCOE is in $/MWh since it's an adder to generation cost.

For wind hybrids:
```
wind_batt4_lcoe(ren_level, batt_level, iso) =
    LCOE_TABLES['wind'][ren_level][iso]                             # Standard wind $/MWh (no DC:AC adjustment)
  + HYBRID_BATT_LCOE_4HR[batt_level][iso] * (1 - ITC_RATE)         # ITC-discounted battery component
```

**Hybrid battery component LCOE derivation:**
The co-located battery $/MWh adder needs to be derived from NREL ATB 2024 battery component costs:
- 4hr CAPEX components: Energy=$170-270/kWh + Power=$280-420/kW → Total 4hr: L=$240, M=$295, H=$375 per kWh
- 8hr CAPEX components: Energy=$170-270/kWh + Power=$280-420/kW → Total 8hr: L=$205, M=$253, H=$323 per kWh
- Financial: WACC=8%, 20yr life → CRF=0.1019, FOM=2.5% of CAPEX($/kW)
- ITC: 30% reduction applied at evaluation time, not baked into table

To convert from CAPEX/kWh to $/MWh adder:
```
hybrid_batt_lcoe_mwh = CAPEX_kWh × (CRF + FOM_rate) × 1000 / (CF_hours × 365)
```
where CF_hours = battery utilization hours per day (depends on hybrid dispatch pattern — use the 8760 profile stats from `generate_hybrid_profiles.py`).

**BUT** — the standalone battery tables in `LCOE_TABLES` are already in annualized $/yr per % of demand (not $/MWh). The hybrid battery component needs to be in $/MWh to add to the generation LCOE. This requires a different derivation. The simplest approach: compute hybrid battery LCOS ($/MWh discharged) from the same CAPEX components, using the empirical cycling frequency from the 8760 hybrid profiles.

**Alternative simpler approach**: Since `step2_2a` already handles standalone batteries differently (they're in the coefficient matrix as dispatch_pct columns), we can handle hybrid battery costs inline in the coefficient computation. The hybrid LCOE for the coefficient matrix would be:
```
hybrid_coeff = (renewable_lcoe_adjusted + hybrid_battery_lcos) × (new_hybrid_pct / 100.0)
```
This means we need `HYBRID_BATTERY_LCOS` tables ($/MWh discharged, L/M/H × ISO) added to `pipeline_config.py`.

### Files to Modify

**`scripts/pipeline_config.py`** (~1625 lines):

1. **Add hybrid column lists** (after line 56):
   ```python
   HYBRID_TYPES = ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']
   RESOURCE_COLS_HYBRID = HYBRID_TYPES  # columns added when hybrids present

   # Full resource column lists with hybrids
   RESOURCE_COLS_BASE_HYBRID = RESOURCE_COLS_BASE + HYBRID_TYPES
   RESOURCE_COLS_OFFSHORE_HYBRID = RESOURCE_COLS_OFFSHORE + HYBRID_TYPES
   RESOURCE_COLS_CAISO_HYBRID = RESOURCE_COLS_CAISO + HYBRID_TYPES
   ```

2. **Add `get_resource_cols(iso, include_hybrids=False)`** — update existing function to accept optional `include_hybrids` parameter, returning extended column list when True.

3. **Add DC:AC ratio constants** (new section):
   ```python
   HYBRID_DC_AC_RATIOS = {
       'solar_batt4': {'CAISO': 1.35, 'ERCOT': 1.50, 'PJM': 1.50, 'NYISO': 1.50, 'NEISO': 1.50, 'MISO': 1.50, 'SPP': 1.50},
       'solar_batt8': {'CAISO': 1.70, 'ERCOT': 2.00, 'PJM': 2.00, 'NYISO': 2.00, 'NEISO': 2.00, 'MISO': 2.00, 'SPP': 2.00},
   }
   ITC_STORAGE_RATE = 0.30  # 30% ITC for co-located battery (IRA Section 48)
   ```

4. **Add hybrid battery LCOS tables** ($/MWh discharged, L/M/H × ISO):
   Derive from existing CAPEX components. For 4hr battery at Medium:
   - CAPEX = $295/kWh, CRF+FOM = 0.1019 + 0.025 = 0.127
   - Annualized = $295 × 0.127 = $37.47/kWh-yr
   - LCOS = $37.47 / (365 × avg_daily_cycles × 0.85 RTE) per kWh
   - For ~1 cycle/day: LCOS ≈ $37.47 / (365 × 0.85) = $120.7/MWh
   - With regional multipliers matching existing battery tables

   These values need to be computed carefully — read the existing battery LCOE derivation comments in pipeline_config.py (lines 966-984) and apply the same methodology but output $/MWh instead of annualized $/yr per % of demand.

5. **Add hybrid entries to `PEAK_CAPACITY_CREDITS`** (line 571):
   ```python
   'solar_batt4': 0.70, 'solar_batt8': 0.85,  # Higher than standalone solar (0.30) — battery shifts to peak
   'wind_batt4': 0.50, 'wind_batt8': 0.65,    # Higher than standalone wind (0.10) — same logic
   ```

6. **Add hybrid entries to `TX_TABLES`** (after line 1035):
   Solar hybrids: AC-rating-adjusted = solar TX / DC:AC ratio (compute inline or pre-compute tables)
   Wind hybrids: Same as wind TX (no DC:AC overbuild)

   ```python
   # Hybrid TX = parent TX / DC:AC ratio (solar) or parent TX (wind)
   # Computed by get_hybrid_tx() function instead of static tables
   ```

7. **Add `get_hybrid_tx(hybrid_type, tx_level, iso)` function** that computes:
   - For `solar_batt4/8`: `get_tx('solar', tx_level, iso) / HYBRID_DC_AC_RATIOS[hybrid_type][iso]`
   - For `wind_batt4/8`: `get_tx('wind', tx_level, iso)` (no adjustment)

**`scripts/dispatch_utils.py`**:

1. Read this file first to understand its current structure.
2. If it has a `RESOURCE_TYPES` constant, update it to optionally include hybrid types.
3. If it has `get_supply_profiles()` or `build_supply_matrix()`, ensure they can accept hybrid profiles.
4. The hybrid profile loading already exists in `step1_pfs_generator.py` as `load_hybrid_profiles(iso)` — dispatch_utils may need to import or mirror this.

### Verification
1. `python -c "from pipeline_config import HYBRID_TYPES, HYBRID_DC_AC_RATIOS, ITC_STORAGE_RATE, get_resource_cols; print(get_resource_cols('CAISO', include_hybrids=True))"`
2. `python -c "from pipeline_config import get_hybrid_tx; print(get_hybrid_tx('solar_batt4', 'Medium', 'CAISO'))"`
3. Verify all imports work: `python -c "import pipeline_config; import dispatch_utils"`

### Commit
Commit with message: "Add hybrid resource constants, LCOE tables, and TX functions to pipeline_config and dispatch_utils"

---

## Session B: Steps 1.2–1.4 — Zone Search, Floor-Aware, Fine Grid

### Task
Add `--hybrid` flag support to Steps 1.2, 1.3, and 1.4. These scripts generate increasingly refined resource mixes. Hybrid support means they use 8-10D resource types instead of 4-6D, include hybrid profiles in the supply matrix, and write hybrid columns to output parquets.

### Branch
Develop on branch `claude/integrate-hybrid-resources-2pk5J`

### Prerequisites
Session A must be complete (pipeline_config.py and dispatch_utils.py updated with hybrid constants).

### Background

**How hybrid detection works**: Downstream steps detect hybrids by checking if `solar_batt4` column exists in the input `{ISO}_coarse_cache.parquet`. If present, the step runs in hybrid mode automatically. The `--hybrid` CLI flag is an override to force hybrid mode even when auto-detection fails.

**Key pattern**: All three scripts follow the same integration pattern:
1. Detect hybrid columns in input parquet (or `--hybrid` flag)
2. Call `get_resource_types(iso, include_hybrids=True)` for extended resource list
3. Load hybrid profiles via `load_hybrid_profiles(iso)` from `step1_pfs_generator`
4. Build supply matrix via `prepare_numpy_profiles(include_hybrids=True, hybrid_profiles=...)`
5. Use extended resource list for grid generation bounds
6. Score via existing `batch_hourly_scores()` (already dimension-agnostic)
7. Write output parquet with hybrid columns included

### Files to Modify

**`scripts/step1_2_zone_search.py`** (~1045 lines):

This script is already mostly dimension-agnostic. Key changes:

1. **Add `--hybrid` CLI argument** (in `argparse` section, near bottom of file):
   ```python
   parser.add_argument('--hybrid', action='store_true', help='Enable hybrid resource types')
   ```

2. **Auto-detect hybrid mode** from input coarse cache:
   ```python
   # In main() or load_data(), after loading coarse cache parquet:
   coarse_schema = pq.read_schema(coarse_path)
   has_hybrids = 'solar_batt4' in coarse_schema.names
   include_hybrids = args.hybrid or has_hybrids
   ```

3. **Update resource type retrieval** (line ~752):
   ```python
   rtypes = s1.get_resource_types(iso, include_hybrids=include_hybrids)
   ```

4. **Load hybrid profiles** when in hybrid mode:
   ```python
   if include_hybrids:
       hybrid_profiles = s1.load_hybrid_profiles(iso)
   else:
       hybrid_profiles = None
   ```

5. **Update supply matrix construction** to use `prepare_numpy_profiles(include_hybrids=..., hybrid_profiles=...)`.

6. **Update mix hash bases** — the `_MIX_HASH_BASES` array uses 7 elements (base-301 hashing). With hybrids, this needs 11 elements (7 base + 4 hybrid). Search for `_MIX_HASH_BASES` and extend it:
   ```python
   n_dims = len(rtypes) + len(STORAGE_COLS)  # e.g., 10 resource + 4 storage = 14
   _MIX_HASH_BASES = np.array([301**i for i in range(n_dims)], dtype=np.int64)
   ```

7. **Update `compute_zone_resource_bounds()`** — currently reads resource bounds from coarse cache. With hybrid columns present, bounds for hybrid resources will be extracted automatically IF the function iterates over `rtypes`. Verify this is the case.

8. **Update `generate_fine_grid()`** — currently builds Cartesian product over resource dimensions. Must include hybrid dimensions in the product. Check if it uses `rtypes` dynamically or has hardcoded resource lists.

9. **Update parquet output** — `save_threshold_pfs()` (lines 428-452) creates columns from `rtypes`. Verify it iterates over `rtypes` dynamically (not hardcoded column names). If hardcoded, update to use `rtypes`.

10. **Update near-miss output** — same pattern, ensure hybrid columns included.

**`scripts/step1_3_floor_aware_pfs.py`** (~469 lines):

This script has more hardcoding than step1_2. Key changes:

1. **Replace `RESOURCE_TYPES` import** with dynamic resource type lookup:
   ```python
   # Instead of: from dispatch_utils import RESOURCE_TYPES
   # Use: rtypes = s1.get_resource_types(iso, include_hybrids=include_hybrids)
   ```

2. **Add `--hybrid` CLI argument** and auto-detection (same pattern as step1_2).

3. **Update `generate_floor_aware_grid()`** — this function has hardcoded resource ranges (solar 0-80%, wind 0-80%, CF 0-40%, etc.). For hybrid mode, add hybrid resource ranges:
   - `solar_batt4`: 0-40% (capped at `HYBRID_MAX_PER_TYPE`)
   - `solar_batt8`: 0-40%
   - `wind_batt4`: 0-40%
   - `wind_batt8`: 0-40%
   - Must respect `SOLAR_FAMILY_CAP` and `WIND_FAMILY_CAP` constraints
   - Grid step: 5% for hybrids (same as offshore/geo)

4. **Update supply matrix construction** — load hybrid profiles and pass to `prepare_numpy_profiles()`.

5. **Update CCS residual calculation** — CCS is computed as `100 - sum(explicit resources)`. With hybrids, the sum includes hybrid columns.

6. **Update parquet output columns** — ensure hybrid columns written to output.

**`scripts/step1_4_fine_grid_pfs.py`** (~462 lines):

Same pattern as step1_3 — very similar structure. Key changes:

1. Same `--hybrid` flag + auto-detection pattern.
2. **Update `generate_fine_grid()`** — add hybrid dimensions at 2% step (finer than floor-aware):
   - `solar_batt4`: 0-30%
   - `solar_batt8`: 0-30%
   - `wind_batt4`: 0-30%
   - `wind_batt8`: 0-30%
3. Same supply matrix, CCS residual, and parquet output updates.

### Critical Pitfalls

1. **Combinatorial explosion**: Adding 4 hybrid dimensions at even 5% step = 9^4 = 6,561× more mixes per base combination. Use the existing OOM protection fallbacks (step 1.4 already has them at line 156-167). Consider:
   - Coarser steps for hybrids (10% in floor-aware, 5% in fine grid)
   - Family budget constraint (solar + solar_batt4 + solar_batt8 ≤ SOLAR_FAMILY_CAP) to prune
   - Use `step1_1a_generate_mixes.py` as a reference — it already handles this with family-budget compositions

2. **Hash collisions**: The base-301 hash needs enough bases for all dimensions. With 10 resource types + 4 storage = 14 total, ensure `_MIX_HASH_BASES` has 14 elements.

3. **Family caps**: The family budget constraints from `step1_1a` must be enforced in zone search and grid generation too. Import `SOLAR_FAMILY_CAP`, `WIND_FAMILY_CAP` from `step1_pfs_generator`.

### Verification
1. Run step1_2 on one small ISO (SPP) with `--hybrid`: `python scripts/step1_2_zone_search.py --iso SPP --hybrid`
2. Check output: `python -c "import pyarrow.parquet as pq; t = pq.read_table('data/step1-pfs/SPP_t75_raw_pfs.parquet'); print(t.schema)"`
3. Verify hybrid columns exist and have non-zero values
4. Repeat quick check for step1_3 and step1_4

### Commit
Commit with message: "Add hybrid resource support to Steps 1.2, 1.3, and 1.4 (zone search, floor-aware, fine grid)"

---

## Session C: Step 1.5 — Storage Refinement

### Task
Update Step 1.5 to handle hybrid columns in input/output while ensuring grid-level storage dispatch does NOT double-count co-located hybrid storage.

### Branch
Develop on branch `claude/integrate-hybrid-resources-2pk5J`

### Prerequisites
Session A must be complete.

### Background

**Critical design point**: Grid-level storage (standalone batteries from Step 1.5) is SEPARATE from hybrid co-located storage. The hybrid battery is already dispatched in the 8760 profile — Step 1.5 only sweeps standalone grid storage ON TOP of the hybrid mix. The hybrid mix columns flow through as-is; Step 1.5 does NOT modify them.

This means the integration is simpler than Steps 1.2-1.4 — Step 1.5 doesn't generate new hybrid mixes, it just needs to:
1. Read input parquets that now have 8-10 resource columns (including hybrid)
2. Pass all resource columns to the scoring function (supply matrix must include hybrid profiles)
3. Preserve hybrid columns in output parquets
4. NOT add hybrid-specific storage sweeps

### Files to Modify

**`scripts/step1_5_storage_refinement.py`** (~1624 lines):

1. **Update resource type retrieval** (line ~1067, 1111):
   ```python
   # Auto-detect hybrids from input parquet schema
   schema = pq.read_schema(near_miss_path)
   has_hybrids = 'solar_batt4' in schema.names
   rtypes = s1.get_resource_types(iso, include_hybrids=has_hybrids)
   ```

2. **Load hybrid profiles** when detected:
   ```python
   if has_hybrids:
       hybrid_profiles = s1.load_hybrid_profiles(iso)
   ```

3. **Update supply matrix** — `prepare_numpy_profiles(include_hybrids=True, hybrid_profiles=...)` so the scoring function accounts for hybrid generation in the hourly matching.

4. **Update mix hash bases** — same as step1_2, extend `_MIX_HASH_BASES` to handle 10+ dimensions:
   ```python
   n_dims = len(rtypes) + len(storage_cols)
   _MIX_HASH_BASES = np.array([301**i for i in range(n_dims)], dtype=np.int64)
   ```

5. **Update column handling in `load_near_miss()`** (lines 399-421):
   - Currently does `np.column_stack([table.column(rt).to_numpy() for rt in rtypes])`
   - With hybrid columns, `rtypes` is longer → column stack produces wider array
   - Verify that downstream indexing (e.g., `nm_combos[result_indices, :]`) handles the wider array

6. **Update `save_storage_results()`** (lines 955-993):
   - Currently writes columns by iterating over `rtypes`
   - Verify it uses `rtypes` dynamically, not hardcoded column names
   - Add hybrid columns to output schema

7. **Update `_load_floor_fine_mixes()`** (lines 256-294):
   - Loads floor/fine PFS files for augmentation
   - Must handle hybrid columns in these files too

8. **Do NOT add hybrid storage sweep logic** — the battery component of hybrids is already in the 8760 profile. Step 1.5 storage parameters (FULL_BAT4, FULL_BAT8, FULL_LDES, FULL_H2) remain unchanged.

### Critical Pitfall

The scoring functions `_batch_score_storage()` and `_batch_mixes_storage_screen()` in `step1_pfs_generator.py` take `supply_rows` as input. The supply matrix must now include hybrid profiles so that the hourly matching correctly accounts for hybrid generation. Without this, the scorer would undercount the supply from hybrids and incorrectly assess how much standalone storage is needed.

### Verification
1. Load a hybrid near-miss parquet and verify hybrid columns are present
2. Run step1_5 on one ISO: `python scripts/step1_5_storage_refinement.py --iso SPP`
3. Check output parquet has hybrid columns preserved with original values
4. Verify storage dispatch values are reasonable (not inflated by missing hybrid supply)

### Commit
Commit with message: "Add hybrid column passthrough to Step 1.5 storage refinement"

---

## Session D: Step 2.1 — Efficient Frontier

### Task
Update Step 2.1 to handle hybrid resource columns in efficient frontier extraction. The dominance/dedup logic operates on total procurement and score — it should work as-is. The main work is ensuring hybrid columns flow through parquet I/O and dedup key computation.

### Branch
Develop on branch `claude/integrate-hybrid-resources-2pk5J`

### Prerequisites
Session A must be complete.

### Background

Step 2.1 does three things:
1. **Threshold gate**: Keeps only rows matching target thresholds
2. **Resource cap filter**: Removes mixes exceeding caps (solar > 100%, total > 350%)
3. **Dedup**: Same resource allocation + same storage dispatch → keep row with max score

There's NO dominance removal — that's delegated to Step 2.2 (cost-based selection).

### Files to Modify

**`scripts/step2_1_efficient_frontier.py`** (~943 lines):

1. **Update `get_resource_cols()` usage** — currently calls `pipeline_config.get_resource_cols(iso)`. Update to detect hybrids from input parquet schema:
   ```python
   schema = pq.read_schema(pfs_path)
   has_hybrids = 'solar_batt4' in schema.names
   resource_cols = get_resource_cols(iso, include_hybrids=has_hybrids)
   ```

2. **Update `resource_cap_filter()`** — currently checks:
   - `solar > 100%` (SOLAR_CAP)
   - `sum(resource_cols) > 350%` (TOTAL_PROCUREMENT_CAP)
   - `hydro > floor(GRID_MIX_SHARES[iso]['hydro'])`

   With hybrids, need to add family cap constraints:
   - `solar + solar_batt4 + solar_batt8 ≤ SOLAR_FAMILY_CAP[iso]`
   - `wind + wind_batt4 + wind_batt8 ≤ WIND_FAMILY_CAP[iso]`
   - `solar_batt4 ≤ HYBRID_MAX_PER_TYPE` (40%)
   - `solar_batt8 ≤ HYBRID_MAX_PER_TYPE` (40%)
   - `wind_batt4 ≤ HYBRID_MAX_PER_TYPE` (40%)
   - `wind_batt8 ≤ HYBRID_MAX_PER_TYPE` (40%)
   - Total procurement cap increases with hybrids (sum includes hybrid columns)

   Import caps from `step1_pfs_generator`: `SOLAR_FAMILY_CAP`, `WIND_FAMILY_CAP`, `HYBRID_MAX_PER_TYPE`.

3. **Update `deduplicate_mixes()`** — uses `resource_cols + STORAGE_COLS` as groupby key. If `resource_cols` is extended with hybrid columns, this works automatically. Verify the dedup key hash handles the wider column set.

4. **Update parquet I/O** — input reads from `data/step1-pfs/{ISO}_t*_storage.parquet`. Output writes to `data/step2.1-ef/{ISO}_ef.parquet`. Both must preserve hybrid columns. Check that column iteration uses `resource_cols` dynamically.

5. **Update `partition_by_threshold_band()`** — operates on score only, should be unaffected. Verify no column assumptions.

### Critical Pitfall

The CCS residual column: some scripts compute `ccs_pct = 100 - sum(explicit resources)`. With hybrid columns, the sum changes. Verify how CCS residual is handled in step2_1 — it may need to exclude hybrid columns from the "explicit resources" sum if CCS residual is meant to represent non-procured generation.

Actually, re-read the model: CCS is NOT a separate optimization variable. It's the residual: whatever the mix doesn't cover with clean firm + solar + wind + hydro + offshore + geo, the CCS fills. With hybrids, the procurement sum is higher (hybrids contribute supply), so CCS residual should be LOWER. Check if step2_1 recomputes CCS residual or just passes it through.

### Verification
1. Create a test hybrid EF parquet: run step2_1 on one ISO after Steps 1.2-1.5 have hybrid outputs
2. Check output schema: `python -c "import pyarrow.parquet as pq; t = pq.read_table('data/step2.1-ef/SPP_ef.parquet'); print(t.schema); print(t.num_rows)"`
3. Verify hybrid columns present and non-zero
4. Verify family cap constraints are enforced (no row violates solar family cap)

### Commit
Commit with message: "Add hybrid column support to Step 2.1 efficient frontier extraction"

---

## Session E: Step 2.2 — Cost Optimization

### Task
This is the largest integration point. Update the cost function to price hybrid resources using component-additive LCOEs with ITC discount and AC-rating-adjusted transmission.

### Branch
Develop on branch `claude/integrate-hybrid-resources-2pk5J`

### Prerequisites
Sessions A and D must be complete (pipeline_config constants + EF parquets with hybrid columns).

### Background

**Current coefficient matrix structure**: 13 columns (see `scripts/step2_2a_cost_optimization.py` lines 524-538):
```
_COL_WHOLESALE=0, _COL_SOL_NEW=1, _COL_WND_NEW=2, _COL_OSW_NEW=3,
_COL_CCS_NEW=4, _COL_UPRATE=5, _COL_GEO=6, _COL_REMAINING=7,
_COL_REMAINING_NUC=8, _COL_BAT4=9, _COL_BAT8=10, _COL_LDES=11, _COL_H2=12
```

With hybrids, we need 4 new coefficient columns:
```
_COL_SB4=13, _COL_SB8=14, _COL_WB4=15, _COL_WB8=16
_N_COEFFS = 17
```

Each hybrid coefficient = `hybrid_pct / 100.0` (fraction of demand from hybrid).
Each hybrid price = `adjusted_renewable_lcoe + itc_discounted_battery_lcos + ac_adjusted_tx`.

### Design Details

**Hybrid pricing formula** (computed per scenario):

For `solar_batt4` at scenario (ren_level, batt_level, tx_level):
```python
solar_lcoe = LCOE_TABLES['solar'][ren_level][iso]
solar_lcoe_adjusted = solar_lcoe / HYBRID_DC_AC_RATIOS['solar_batt4'][iso]  # CF adjustment
batt_lcos = HYBRID_BATTERY_LCOS['4hr'][batt_level][iso]  # $/MWh discharged
batt_lcos_itc = batt_lcos * (1 - ITC_STORAGE_RATE)  # 30% ITC
tx = get_hybrid_tx('solar_batt4', tx_level, iso)  # AC-rating-adjusted
price_sb4 = solar_lcoe_adjusted + batt_lcos_itc + tx
```

For `wind_batt4`:
```python
wind_lcoe = LCOE_TABLES['wind'][ren_level][iso]  # No DC:AC adjustment for wind
batt_lcos = HYBRID_BATTERY_LCOS['4hr'][batt_level][iso]
batt_lcos_itc = batt_lcos * (1 - ITC_STORAGE_RATE)
tx = get_hybrid_tx('wind_batt4', tx_level, iso)  # = wind TX (no adjustment)
price_wb4 = wind_lcoe + batt_lcos_itc + tx
```

**Key insight**: Hybrid prices depend on 3 toggles (ren, batt, tx) but the coefficient is scenario-invariant. This fits perfectly into the existing precompute architecture — coefficients are static, prices are computed per scenario.

### Files to Modify

**`scripts/step2_2a_cost_optimization.py`** (~2967 lines):

1. **Extend coefficient columns** (lines 524-538):
   ```python
   _COL_SB4 = 13  # solar_batt4 hybrid
   _COL_SB8 = 14  # solar_batt8 hybrid
   _COL_WB4 = 15  # wind_batt4 hybrid
   _COL_WB8 = 16  # wind_batt8 hybrid
   _N_COEFFS = 17
   ```

2. **Update `precompute_base_year_coefficients()`** (line 541+):
   - Add hybrid arrays extraction from input:
     ```python
     sb4_pct = arrays.get('solar_batt4', np.zeros(N, dtype=np.float64))
     sb8_pct = arrays.get('solar_batt8', np.zeros(N, dtype=np.float64))
     wb4_pct = arrays.get('wind_batt4', np.zeros(N, dtype=np.float64))
     wb8_pct = arrays.get('wind_batt8', np.zeros(N, dtype=np.float64))
     ```
   - Extend coefficient matrix to 17 columns:
     ```python
     coeff_matrix = np.empty((N, _N_COEFFS), dtype=np.float64)
     # ... existing 13 columns unchanged ...
     coeff_matrix[:, _COL_SB4] = sb4_pct / 100.0
     coeff_matrix[:, _COL_SB8] = sb8_pct / 100.0
     coeff_matrix[:, _COL_WB4] = wb4_pct / 100.0
     coeff_matrix[:, _COL_WB8] = wb8_pct / 100.0
     ```
   - **Update CCS residual calculation**: Currently `ccs_pct = max(0, 100 - (cf + sol + wnd + hyd + osw + geo))`. With hybrids: `ccs_pct = max(0, 100 - (cf + sol + wnd + hyd + osw + geo + sb4 + sb8 + wb4 + wb8))`. This is critical — hybrids displace CCS in the residual.

3. **Update `_build_price_vector()` or equivalent** — the function that maps scenario toggles to a price array:
   ```python
   # Existing: prices = np.array([wholesale, sol_price, wnd_price, osw_price, ccs_price,
   #                               uprate_price, geo_price, remaining_price, remaining_nuc_price,
   #                               bat4_price, bat8_price, ldes_price, h2_price])
   # Extended:
   prices = np.array([..., sb4_price, sb8_price, wb4_price, wb8_price])  # 17 elements
   ```

   Where hybrid prices are computed per scenario:
   ```python
   from pipeline_config import HYBRID_DC_AC_RATIOS, ITC_STORAGE_RATE, HYBRID_BATTERY_LCOS, get_hybrid_tx

   def compute_hybrid_prices(iso, ren_level, batt_level, tx_level):
       sol_lcoe = LCOE_TABLES['solar'][ren_level][iso]
       wnd_lcoe = LCOE_TABLES['wind'][ren_level][iso]
       b4_lcos = HYBRID_BATTERY_LCOS['4hr'][batt_level][iso] * (1 - ITC_STORAGE_RATE)
       b8_lcos = HYBRID_BATTERY_LCOS['8hr'][batt_level][iso] * (1 - ITC_STORAGE_RATE)

       sb4_price = sol_lcoe / HYBRID_DC_AC_RATIOS['solar_batt4'][iso] + b4_lcos + get_hybrid_tx('solar_batt4', tx_level, iso)
       sb8_price = sol_lcoe / HYBRID_DC_AC_RATIOS['solar_batt8'][iso] + b8_lcos + get_hybrid_tx('solar_batt8', tx_level, iso)
       wb4_price = wnd_lcoe + b4_lcos + get_hybrid_tx('wind_batt4', tx_level, iso)
       wb8_price = wnd_lcoe + b8_lcos + get_hybrid_tx('wind_batt8', tx_level, iso)
       return sb4_price, sb8_price, wb4_price, wb8_price
   ```

4. **Update `_table_to_arrays()`** — the function that converts parquet table to numpy arrays dict. Must extract hybrid columns:
   ```python
   for col in ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']:
       if col in table.schema.names:
           arrays[col] = table.column(col).to_numpy().astype(np.float64)
   ```

5. **Update `price_mix_batch()`** — if it constructs prices differently from the coefficient path, ensure hybrid pricing is included there too.

6. **Update batch eval functions** — `eval_cost_fast()`, `_eval_cost_numba()`, `_batch_eval_and_argmin()` all operate on `(N, K)` coefficient matrices. Since K is determined at runtime from `coeff_matrix.shape[1]`, these should work automatically with K=17. **Verify** that no hardcoded `13` exists in these functions.

7. **Update winner detail extraction** — when the cost-optimal mix is found, the code extracts a detailed breakdown. Add hybrid resource costs to this breakdown:
   ```python
   # In the winner detail dict:
   detail['solar_batt4_pct'] = sb4_pct[winner_idx]
   detail['solar_batt4_cost'] = sb4_pct[winner_idx] / 100.0 * demand_twh * sb4_price
   # ... etc for other hybrids
   ```

8. **Update output parquet schema** — ensure hybrid columns are written to output files in `data/step2.2-cost/`.

9. **Scenario key format** — UNCHANGED. Hybrids use existing toggles (ren, batt, tx), so the scenario key `{r}{f}{batt}{ldes}_{ff}_{tx}_{ccs}{q45}_{geo}` stays the same. The hybrid prices are derived from these same toggle values.

### Existing vs. New-Build Split for Hybrids

**All hybrid procurement is new-build** — there is no "existing hybrid" capacity in the 2025 snapshot (co-located hybrids are an emerging deployment model). This simplifies pricing:
```python
# Hybrid coefficient = total hybrid pct / 100.0 (no existing/new split needed)
coeff_matrix[:, _COL_SB4] = sb4_pct / 100.0  # ALL new-build
```

### Critical Pitfalls

1. **CCS residual displacement**: Hybrids reduce CCS residual. If a mix has 20% solar_batt4, that's 20% less CCS needed. The `ccs_eligible_twh` calculation must subtract hybrid TWh.

2. **Price vector length**: The Numba kernel `_eval_cost_numba` uses `K = coeff_matrix.shape[1]` dynamically. But the price vector must also be length 17. Search for ALL places where the price vector is constructed and ensure they produce 17 elements.

3. **Batch price matrix**: `batch_eval_and_argmin_all()` takes a `price_matrix` of shape `(S, K)` where S = number of scenarios and K = number of coefficients. The `_build_price_matrix()` function (or equivalent) must generate K=17 columns per scenario row.

4. **Numba cache invalidation**: The `@njit(cache=True)` kernels will need cache refresh since the array shapes change. Delete `__pycache__` and `.nbi` files before testing.

5. **Track 2 (newbuild) and Track 3 (cost-to-replace)**: `step2_2b_track_nb_ctr.py` also uses coefficients. It will need the same hybrid updates in a future session, but is NOT in scope for this session. Just ensure the changes to step2_2a don't break step2_2b's imports.

### Verification
1. Syntax check: `python -c "import py_compile; py_compile.compile('scripts/step2_2a_cost_optimization.py')"`
2. Import check: `python -c "from step2_2a_cost_optimization import precompute_base_year_coefficients, eval_cost_fast"`
3. Run on one ISO with small EF: `python scripts/step2_2a_cost_optimization.py --iso SPP --dry-run` (if dry-run flag exists)
4. Verify output parquet has hybrid columns and non-zero hybrid costs
5. Spot-check: for a mix with 10% solar_batt4, verify the cost = 0.10 × demand_twh × (adjusted_solar_lcoe + itc_batt_lcos + ac_tx)

### Commit
Commit with message: "Integrate hybrid resource pricing into Step 2.2 cost optimization — component-additive LCOE with ITC and AC-adjusted TX"

---

## Session F: Documentation & SPEC.md Updates

### Task
Record all hybrid integration design decisions in SPEC.md and hybrid-design.md. Update CLAUDE.md pipeline architecture section.

### Branch
Develop on branch `claude/integrate-hybrid-resources-2pk5J`

### What to Document

1. **SPEC.md** — Add a new section "§X Hybrid Co-Located Resources" covering:
   - 4 hybrid types and their physical models
   - Component-additive LCOE formula with worked examples
   - DC:AC ratios (solar) and battery MW ratios (wind)
   - ITC treatment (30% for both solar and wind hybrids)
   - AC-rating-adjusted transmission formula
   - Toggle pairing (no new toggles)
   - Family caps (SOLAR_FAMILY_CAP, WIND_FAMILY_CAP, HYBRID_MAX_PER_TYPE)
   - Pipeline integration status (which steps support hybrids)

2. **hybrid-design.md** — Update "Pipeline Integration Status" section with completion checkmarks. Mark all open items as resolved.

3. **CLAUDE.md** — Update pipeline architecture to mention hybrid dimensions (8-10D) and hybrid profile loading.

### Commit
Commit with message: "Document hybrid integration decisions in SPEC.md and hybrid-design.md"
