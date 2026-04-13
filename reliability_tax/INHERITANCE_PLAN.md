# Reliability Tax — Inheritance Plan

Locks which functions, classes, and constants the Reliability Tax pipeline step inherits from the existing 8-step pipeline. The new step is a **deterministic cost minimization under constraint** (pathway simulation 2025–2050, NPV@7% objective) — NOT a market prediction.

This file is built up in three prompts:

- **1B-i (this prompt)**: Step 2.1 — Efficient Frontier extraction / loading.
- **1B-ii**: Step 2.2 — Cost optimization (to be added).
- **1B-iii**: Step 6 — Market/policy model + cross-cutting + DECISION_CARDS refs (to be added).

For locked invariants (endpoint targets, pathways, clean firm bucket, cost basis, objective, ISO scope, stranding definitions, demand growth) see `reliability_tax/README.md § Locked invariants`. They are not duplicated here.

---

## Top-level structure

- [Step 2.1 — Efficient Frontier](#step-21--efficient-frontier)
  - Imported unchanged
  - Copied and modified
  - Written fresh (near Step 2.1's domain)
- [Step 2.2 — Cost Optimization](#step-22--cost-optimization) *(Prompt 1B-ii)*
- [Step 6 — Market / Policy Model](#step-6--market--policy-model) *(Prompt 1B-iii)*
- [Cross-cutting / written fresh](#cross-cutting--written-fresh) *(Prompt 1B-iii)*
- [Questions for DECISION_CARDS](#questions-for-decision_cards)

---

## Step 2.1 — Efficient Frontier

Source files read in full for this section:

- `scripts/step2_1_efficient_frontier.py` (1,076 lines) — build-time EF extraction from PFS.
- `scripts/step2_1b_augment_thin_ef.py` (513 lines) — post-EF augmentation of thin bands.

**What the Reliability Tax step needs from this domain:** a consumer-side loader for the EF parquets (per ISO × threshold band), pathway-aware resource-cap filtering, and helpers to navigate the threshold ladder. The Reliability Tax step does NOT rebuild the EF — it reads `data/step2.1-ef/step_2_1_EF_{ISO}_{T}.parquet` files that Step 2.1 has already produced.

### Imported unchanged

Imported directly from the source files — no fork, no modification. If upstream changes, the Reliability Tax step picks up the change automatically.

| Symbol | Source file | Lines | Why we use it unchanged |
|---|---|---|---|
| `load_ef_parquet(iso, threshold)` | `step2_1b_augment_thin_ef.py` | 80–97 | Handles the single-file AND multi-part (`_partN`) layouts created by `split_large_parquets`. Returns a pandas DataFrame per (ISO, threshold). This is the exact loader the Reliability Tax pathway iterator needs to pull candidate mixes for a given year's CFE target. |
| `get_resource_cols(df)` | `step2_1b_augment_thin_ef.py` | 100–103 | Extracts resource column names (excludes `iso`, `hourly_match_score`, `pareto_type`, and the four `STORAGE_COLS`). ISO-aware via the df schema (handles CAISO geothermal and NEISO/NYISO offshore wind automatically). |
| `deduplicate(df, resource_cols)` | `step2_1b_augment_thin_ef.py` | 106–122 | Pandas-side dedup using Step 2.1's canonical key (int32-cast resources + 20×-scaled int32 storage dispatch, keep max `hourly_match_score`). Reliability Tax needs this when merging EF bands across adjacent thresholds (see "Written fresh" below). |
| `get_threshold_bounds(threshold)` | `step2_1b_augment_thin_ef.py` | 62–68 | Returns `[lower, upper)` score bounds for a threshold band, respecting `ACTIVE_THRESHOLDS`. Used to verify that loaded mixes actually fall in the band the pathway is asking for. |
| `get_adjacent_thresholds(threshold)` | `step2_1b_augment_thin_ef.py` | 71–77 | Returns `(below, above)` threshold neighbors. Needed for pathway year-to-year transitions where a mix may slip between bands after dedup or augmentation. |
| `normalize_table(t, iso, include_hybrids=False)` | `step2_1_efficient_frontier.py` | 150–203 | Canonical schema cast (int16 resources, float64 storage + score, string ISO/pareto_type). Reliability Tax uses this when reading pyarrow tables directly (some pathway code will prefer arrow to pandas for memory). |
| `_detect_hybrids_in_schema(schema_names)` | `step2_1_efficient_frontier.py` | 145–147 | One-liner check for hybrid columns (`solar_batt4`, `solar_batt8`, `wind_batt4`, `wind_batt8`) in a parquet schema. Needed so pathway code can decide whether to request hybrid-aware resource column lists. |
| `_extract_threshold_from_filename(fname)` | `step2_1_efficient_frontier.py` | 206–212 | Parses `{ISO}_t{XX}_...parquet` filenames into a float threshold. Reliability Tax needs this when scanning `data/step2.1-ef/` to discover which `(ISO, threshold)` bands actually exist on disk before the pathway asks for them. |
| `format_threshold(thr)` | `step2_1_efficient_frontier.py` | 612–614 | Canonical filename formatter (`87.5 → '87.5'`, `10.0 → '10'`). Reliability Tax must produce the exact same string when constructing EF filenames, or `load_ef_parquet` won't find them. |
| `TARGET_THRESHOLDS` (constant) | `step2_1_efficient_frontier.py` | 71 | The full 20-entry threshold ladder `[10, 20, 30, 40, 50, ..., 99, 99.5, 99.9]`. Reliability Tax's SBTi-year mapping must resolve each `THRESHOLD_TARGET_YEARS` target to the closest ladder entry — importing this constant guarantees no drift. |
| `STORAGE_COLS` (constant) | `step2_1_efficient_frontier.py` | 75 | `['battery_dispatch_pct', 'battery8_dispatch_pct', 'ldes_dispatch_pct', 'h2_dispatch_pct']`. Used by every dedup + normalize path. |
| `STEP2_EF_OUTPUT_DIR` (constant) | `step2_1_efficient_frontier.py` | 67 | Absolute path to `data/step2.1-ef/`. Single source of truth for where the EF parquets live. |

### Copied and modified

Forked into `reliability_tax/scripts/` and modified. Each entry explains *what changed* and *why* — the rationale is always traceable to a locked invariant or a consumer-side vs. build-side difference in contract.

#### `pathway_cap_filter(table, iso, pathway_id, year, include_hybrids=False)`

- **Source**: `resource_cap_filter(table, iso, include_hybrids=False)` in `step2_1_efficient_frontier.py` lines 500–558.
- **Modification**: Accepts a `pathway_id` ∈ {`vre_only`, `behavioral_pivot`, `economic_pivot`, `clean_firm_proactive`} and a `year` ∈ [2025, 2050]. On top of the inherited `SOLAR_CAP` / `TOTAL_PROCUREMENT_CAP` / hydro-ceiling / hybrid-family-cap checks, it enforces pathway-specific bucket constraints per the locked invariants in `README.md § Locked invariants`:
  - `vre_only` → `clean_firm == 0` for all years; CAISO `geothermal == 0` (geothermal is in the clean-firm bucket per invariant 4).
  - `behavioral_pivot` → `clean_firm == 0` until the year the ISO hits the 90% CFE plateau; unconstrained (subject to CCS caps) afterward. Geothermal treated as clean firm per invariant 4.
  - `economic_pivot` → identical to `behavioral_pivot` but the trigger year is resolved by the pathway iterator from marginal `$/CFE%` vs. clean-firm LCOE, not by a fixed 90% plateau.
  - `clean_firm_proactive` → no added lower bound on clean firm; inherited upper bounds (CCS cap via `CCS_CAP_TWH`, geothermal CAISO-only) still apply.
- **Rationale**: Step 2.1's `resource_cap_filter` enforces *physics-feasibility* caps that are pathway-agnostic. The Reliability Tax step layers *pathway-feasibility* caps on top — these cannot live in the upstream function because the main pipeline does not know about pathways. Forking (rather than wrapping) keeps the filter hot-path vectorized and avoids adding pathway awareness to a script that's shared across every ISO × threshold × cost-scenario combination in the main pipeline.
- **Must track upstream**: Whenever `SOLAR_CAP`, `TOTAL_PROCUREMENT_CAP`, `SOLAR_FAMILY_CAP`, `WIND_FAMILY_CAP`, `HYBRID_MAX_PER_TYPE`, or the hydro-ceiling logic change in `step2_1_efficient_frontier.py`, the fork must be updated in the same commit. A smoke-test assertion that calls both functions with `pathway_id='clean_firm_proactive'` on a random sample and checks row-for-row equality is the cheapest way to catch drift.

#### `load_ef_band_union(iso, target_threshold, pathway_id, year)`

- **Source**: `load_ef_parquet(iso, threshold)` in `step2_1b_augment_thin_ef.py` lines 80–97.
- **Modification**: Instead of returning a single threshold band, loads **all bands at or above `target_threshold`** and concatenates them. Step 2.1 partitions mixes into non-overlapping bands where each mix lives in the highest band satisfying `T ≤ score` (see `partition_by_threshold_band` lines 617–642). To recover the full set of mixes qualifying for a target `T*`, a consumer must read bands `T*, next_T_above, …, 99.9`. The function then applies `pathway_cap_filter(..., pathway_id, year, ...)` before returning, so the pathway iterator never sees mixes that violate pathway constraints. Uses `TARGET_THRESHOLDS` (imported unchanged) to enumerate the bands to load.
- **Rationale**: The existing `load_ef_parquet` is single-band by design — Step 2.1b only ever augments one band at a time. The Reliability Tax pathway iterator asks "give me every mix that meets CFE ≥ 95% in 2040 under the behavioral-pivot rules", which requires a union-of-bands read plus pathway filtering in one call. Building this as a modified clone of `load_ef_parquet` (rather than a wrapper) means the pathway code can request the same return shape — a single concatenated, deduplicated pandas DataFrame — regardless of how many bands were read under the hood.
- **Must track upstream**: Any change to `step_2_1_EF_{ISO}_{T}.parquet` filename conventions, the `_partN` split strategy (`split_large_parquets` lines 88–142), or the band-partitioning scheme (`partition_by_threshold_band` lines 617–642) must be mirrored here. A filename-format regression test is cheap insurance.
