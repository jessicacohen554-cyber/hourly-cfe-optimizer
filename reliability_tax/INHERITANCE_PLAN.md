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

### Written fresh (near Step 2.1's domain)

New functions with no upstream analog, but logically part of the EF-consumer layer. They live in `reliability_tax/scripts/ef_loader.py` (planned in Prompt 1C) so the pathway iterator has one place to import from.

#### `resolve_target_threshold(year, iso)`

- **Purpose**: Maps a calendar year ∈ [2025, 2050] to the canonical `TARGET_THRESHOLDS` ladder entry that the pathway must meet that year.
- **Logic**: Reads `pipeline_config.THRESHOLD_TARGET_YEARS` (SBTi milestone ladder: 2030→50%, 2035→70%, 2040→90%, 2045→95%, 2050→99.9%), linearly interpolates between milestone years, then snaps to the nearest `TARGET_THRESHOLDS` entry. The snap rule is *"floor to the highest ladder entry ≤ interpolated target"* so the pathway never claims credit for a band it has not actually reached.
- **Why it belongs here**: The SBTi ladder and the EF threshold ladder are two different discretizations of "CFE%". Every call to `load_ef_band_union` needs this resolution, and it's a pure function of the two constants — it should not be reinvented inside the pathway iterator.

#### `check_hydro_floor(df, iso)`

- **Purpose**: Returns a boolean mask keeping only mixes where `hydro == ceil(GRID_MIX_SHARES[iso]['hydro'])`. The main pipeline's `resource_cap_filter` only enforces a ceiling (hydro ≤ existing share) because the physics search explored +10pp hydro-adder mixes. Reliability Tax additionally enforces a *floor* — pathways never retire existing hydro, so hydro stays pinned at its existing share.
- **Why it belongs here**: The main pipeline is agnostic about whether existing hydro can be retired (retirement is a Step 4 fossil-only concern). Reliability Tax takes a firm stance per the locked invariants: hydro is existing-only and never contracts. This must be a consumer-side filter because the EF parquets do contain mixes with `hydro < existing_share` (Step 1 explores the full grid, including sub-existing hydro mixes).
- **Question surfaced** — see DECISION_CARDS Q2.

#### `tag_pathway_eligible(df, pathway_id, iso, year)`

- **Purpose**: Adds a boolean column `pathway_eligible` to a loaded EF DataFrame without dropping rows, so the pathway iterator can score ineligible mixes at infinity-cost rather than silently dropping them. This preserves the candidate-pool cardinality across pathways and makes debugging easier (you can see *why* a mix was excluded).
- **Logic**: Wraps `pathway_cap_filter` but returns a mask rather than a filtered table, plus a string `pathway_exclusion_reason` column (`'clean_firm>0 in vre_only'`, `'clean_firm>0 before pivot year'`, etc.).
- **Why it belongs here**: Sits between the loader and the pathway iterator. Ensures the iterator can emit audit reports on why mixes were filtered — a reviewer-facing concern that upstream Step 2.1 doesn't need.

#### `load_ef_manifest()`

- **Purpose**: Scans `data/step2.1-ef/` once at pathway-iterator startup and returns a dict `{(iso, threshold): [list_of_file_paths]}` covering every `.parquet` (including `_partN` splits, excluding `_batch_` transients). Used to fail fast if any `(iso, threshold)` the pathway will need is missing on disk, before spending compute on a 25-year iteration.
- **Logic**: Glob `step_2_1_EF_*.parquet`, parse filenames with `_extract_threshold_from_filename`, skip `_batch_` files, group by ISO + threshold. Cross-check against `ISOS × TARGET_THRESHOLDS` from `pipeline_config` and raise `FileNotFoundError` listing every missing combination.
- **Why it belongs here**: The equivalent build-side scanner `scan_iso_files` (lines 215–295 of the main script) scans the **Step 1 PFS** directory, not the **Step 2.1 EF** directory, and its output contract is "raw/fine/floor/storage PFS filenames grouped by ISO" — wrong shape for a consumer. Building fresh is cleaner than contorting the upstream function.

---

## Step 2.2 — Cost Optimization

*To be populated in Prompt 1B-ii. Do not edit in this prompt.*

---

## Step 6 — Market / Policy Model

*To be populated in Prompt 1B-iii. Do not edit in this prompt.*

---

## Cross-cutting / written fresh

*To be populated in Prompt 1B-iii. Covers: pathway iterator scaffolding, stranding bookkeeping, NPV discounting, Wright's Law integration into the 25-year loop, reliability-tax-specific output schemas.*

---

## Questions for DECISION_CARDS

Accumulated across Prompts 1B-i / ii / iii. Each question must be resolved before DECISION_CARDS.md is written (in Prompt 1B-iii) and before any Reliability Tax scripts are written.

### From Step 2.1 (Prompt 1B-i)

1. **Dedup provenance loss.** Step 2.1's `deduplicate_mixes` (lines 561–609) keeps max `hourly_match_score` per unique physical configuration and drops the threshold column. After dedup, Reliability Tax cannot recover *which Step 1 threshold run* produced a given mix. Is the final score alone sufficient for pathway ranking, or does the pathway iterator need the originating threshold (e.g., for audit reports, or to discount mixes that came from the floor-aware augmentation pass)?

2. **Hydro floor vs. ceiling.** The main pipeline's `resource_cap_filter` (lines 500–558) enforces `hydro ≤ ceil(existing share)` but allows `hydro < existing share`. Locked invariant 8 states existing fleet is out of scope for stranding — this implies existing hydro cannot be retired in a pathway. Should Reliability Tax enforce `hydro == existing share` (floor + ceiling) via `check_hydro_floor`, or allow pathways to under-procure hydro when cheaper resources dominate? Affects every pathway's VRE build-out ratio.

3. **Offshore wind in the VRE bucket.** Locked invariant 4 says offshore wind is VRE, not clean firm. NEISO and NYISO are the only `OFFSHORE_ISOS` with the `offshore_wind` resource column in their EF parquets. Does `pathway_cap_filter` treat `offshore_wind` identically to `wind` (subject to `WIND_FAMILY_CAP`), or as a distinct bucket with its own cap? If distinct, where does the cap come from — a new constant in `pipeline_config` or a Reliability-Tax-specific override?

4. **CAISO geothermal under VRE-only pathway.** Locked invariant 4 says geothermal is in the clean firm bucket, CAISO-only. Under Pathway 1 (VRE + batteries only), geothermal must be zeroed — but CAISO's EF parquets already contain mixes with `geothermal > 0`. Confirm that `pathway_cap_filter(iso='CAISO', pathway_id='vre_only')` drops all such mixes (reducing the CAISO candidate pool substantially). This is the expected behavior per the invariant, but it should be an explicit decision since it materially affects CAISO's Pathway 1 feasibility.

5. **Band-union vs. single-band reads.** Step 2.1 stores each mix in exactly one band (highest `T ≤ score`). Recovering "all mixes qualifying for CFE ≥ T*" requires loading bands `[T*, next_T, …, 99.9]` and concatenating. For target thresholds near the tails (10–40% or 99.5–99.9%) this is cheap; near the densest point (~75–90%) it may pull 10M+ rows into memory per ISO. Does the pathway iterator need a chunked reader, or is in-memory union acceptable (pathway iteration is per-ISO, and the Reliability Tax target set is ≤90% so the worst case is bands [90, 92.5, 95, 97.5, 99, 99.5, 99.9])?

6. **Pareto pre-filtering under fixed objective.** Step 2.1 intentionally skips cross-mix dominance removal because different LCOE assumptions reshuffle the cost ranking. Reliability Tax uses a *fixed* objective (NPV@7% real, 2025 USD, no inflation — invariants 5–6). Under a fixed objective, much of the EF is dominated and could be dropped upfront to shrink the candidate pool. Should Reliability Tax run a pathway-specific dominance filter at load time (post `pathway_cap_filter`), or carry the full pool through and let downstream cost minimization handle it?

7. **`pareto_type='augmented'` mixes.** Step 2.1b fills thin bands with perturbed/interpolated candidates and tags them `pareto_type='augmented'`. They are physically valid (scored through the same Numba kernel as native mixes) but have no Step 1 provenance. Should Reliability Tax treat augmented mixes as first-class candidates, filter them out (physics-only provenance), or down-weight them (e.g., only use them as tie-breakers)? Affects the size of the thin-band candidate pools, which are exactly the ones pathways are most likely to query.

8. **Storage dispatch granularity.** Storage dispatch columns are stored at 0.05% resolution (20×-scaled int32 dedup keys, lines 586–597). Reliability Tax pathway iteration may want to evaluate finer storage sweeps across years as Wright's Law brings storage cost down. Is 0.05% sufficient, or does the new step need its own higher-resolution storage sweep (and if so, does it live in this inheritance plan or a fresh Step 1.5 clone)?

*Questions from Step 2.2 will be added in Prompt 1B-ii. Questions from Step 6 and cross-cutting will be added in Prompt 1B-iii.*
