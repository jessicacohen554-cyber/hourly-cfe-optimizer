# `scripts/step_2_3_pathway_optimizer.py` — v2 rewrite plan (≤500 lines)

## Context

The existing `scripts/step_2_3_pathway_optimizer.py` is 3,909 lines. Its hot
loop (`score_ef_batch_with_gas` → `size_required_gas_mw` →
`worst_hour_gas_sizing`) calls the dispatch-cache percentile helper **per
candidate per year**. With ~thousands of EF rows per threshold × 26 years ×
350 runs in the sweep, this is O(n_mixes × n_years × dispatch_cost) and is
the central reason full-sweep wall-clock is measured in hours. It has also
accreted three years of methodology drift: v1 Cards M/F/K have been
superseded by M'/F'/K' (SPEC §24.4), cross-endpoint gas-fleet seeding was a
bug (§24.6 Bug 1), gas sizing double-counted demand growth (§24.6 Bug 3),
priced VRE curtailment was hardcoded zero until §24.7, and per-pathway NOAK
plus the clean-firm floor were reworked twice (§24.8).

v2 methodology is locked. The frozen contract lives at
`reliability_tax/PATHWAY_OUTPUT_SCHEMA.md` — 13 top-level keys, 4-key
`tables`, 7-key `vre_curtailment_at_endpoint`, 11-key `endpoint_mix_pct`,
15-key `stranding_metadata` (9 live + 6 inert), `new_gas_fleet` is a
list[1] with a single consolidated vintage at `peak_year`, and
`reliability_tax.components_usd.vre_storage_overbuild_capex_usd` is
hardcoded `0.0`.

The rewrite targets ≤500 lines, single file, single CLI, and implements the
two-stage architecture below that eliminates the per-mix-per-year
dispatch-cache bottleneck.

## THE KEY ARCHITECTURAL DECISION

`worst_hour_gas_sizing` is factored into two pieces:

1. **One-time precompute per ISO** — writes a sidecar parquet at
   `data/step2.1-ef/step_2_1_EF_{ISO}_peakclean.parquet` mapping each
   deduplicated mix (keyed on the resource+storage share tuple) to a
   scalar `clean_peak_hour_mw`. Computed as the value of the sum of
   clean-resource 8,760-hour profiles at `argmin(clean_profile_sum)` —
   i.e., the MW of clean contribution at the hour when clean is weakest.
   Pure numpy over the dispatch cache. Runs once per ISO; reruns only when
   the dispatch cache version bumps or the EF row union grows.

2. **Per-year vectorized scalar** — `gas_sizing_year(year, clean_arr) →
   np.ndarray[n_mixes]`:

       np.maximum(0,
                  peak_demand_mw(year) * (1 + RA_margin)
                  - clean_arr
                  - existing_gas_available_mw(year))

   One numpy op producing the full (n_mixes,) gas-need vector for that
   year. No dispatch-utils call inside this path. Stacked across 26 years
   gives the (26, n_mixes) `gas_required_matrix` that feeds the argmin
   cost matrix.

The old function's per-candidate dispatch-cache call is eliminated
entirely.

## Port decisions (as specified in the brief)

- **`worst_hour_gas_sizing`** — NOT imported. The RA-margin +
  existing-gas-availability arithmetic is reimplemented clean as
  `gas_sizing_year` (per-year scalar, vectorized over mixes). The old
  function embedded dispatch-cache calls inside the residual-percentile
  step; eliminating that embedding is the whole point of the rewrite.

- **`compute_fossil_retirement`** — body copied verbatim from
  `scripts/dispatch_utils.py` lines 950–1051 into the new file, along
  with its immediate helper `coal_fraction_at_clean_pct` (dispatch_utils
  lines 925–943) and the `_FOSSIL_CAPACITY_FACTORS` dict and `_twh_to_gw`
  helper used by the caller in the old step 2.3 (lines 2858–2886). Zero
  imports from `dispatch_utils`.

## Permitted imports

- `scripts.pipeline_config` (as `pc`) — constants only (`PEAK_DEMAND_MW`,
  `EXISTING_GAS_CAPACITY_MW`, `GAS_AVAILABILITY_FACTOR`,
  `RESOURCE_ADEQUACY_MARGIN`, `REGIONAL_DEMAND_TWH`, `DEMAND_GROWTH_RATES`,
  `NEW_CCGT_COST_KW_YR`, `EXISTING_GAS_FOM_KW_YR`,
  `CCGT_OVERNIGHT_CAPEX_USD_KW`, `NEW_GAS_ASSET_LIFE_YEARS`,
  `GRID_MIX_SHARES`, `COAL_CAP_TWH`, `OIL_CAP_TWH`,
  `COAL_PHASE_OUT_START`, `COAL_PHASE_OUT_END`,
  `get_announced_coal_retired_gw`, `NOAK_YEAR_BY_PATHWAY`,
  `PATHWAY_NOAK_TECHS`, `TX_ADDER_USD_PER_MWH`,
  `compute_clean_firm_tranches`).
- `pyarrow`, `pyarrow.parquet` (as `pq`), `numpy`, `json`, `argparse`,
  `pathlib.Path`, `dataclasses`.
- **NEVER** import from `scripts/step_2_3_pathway_optimizer.py` (the old
  file being replaced), `scripts/step2_2a_cost_optimization.py`, or
  `scripts/dispatch_utils.py`.

## Module-scope loads (once, at import time)

```python
_EGRID_EMISSION_RATES = json.load(open('data/egrid_emission_rates.json'))
_WRIGHTS              = json.load(open('data/step5-wrights/wrights_law_curves.json'))
_FOSSIL_MIX_BY_ISO    = {iso: pq.read_table('data/eia-930/eia_fossil_mix.parquet',
                                            filters=[('iso','=',iso)])
                         for iso in ISOS}
_DISPATCH_CACHE_BY_ISO = {iso: pq.read_table(f'data/step3-dispatch/{iso}_dispatch_cache.parquet')
                          for iso in ISOS}
_ANNUAL_MANIFEST_BY_ISO = {iso: pq.read_table(f'data/step3-dispatch/{iso}_annual_manifest.parquet')
                           for iso in ISOS}
_PEAKCLEAN_BY_ISO = _load_or_build_peakclean_sidecar()   # the Stage-1 precompute
```

The peakclean sidecar loader is: if
`data/step2.1-ef/step_2_1_EF_{ISO}_peakclean.parquet` exists and its
`source_cache_version` matches the current dispatch cache version, read it;
otherwise call `precompute_clean_peak_hour_mw(iso)` and write it. No locks,
no threads — a single pass.

## File outline (≤500 lines total)

```
  1-  20   module docstring + imports
 20-  55   module-scope constants (ISOS, PATHWAYS, YEARS, discount rates,
           paths) + the 5 module-scope loads above
 55- 110   port: compute_fossil_retirement + coal_fraction_at_clean_pct +
           _FOSSIL_CAPACITY_FACTORS + _twh_to_gw (copied from dispatch_utils)
110- 150   dataclasses: Vintage, VintageLedger, RunConfig
150- 200   Stage-1 precompute: precompute_clean_peak_hour_mw(iso) +
           _load_or_build_peakclean_sidecar
200- 230   EF loader: load_ef_mixes(iso, threshold) + peakclean join
230- 270   vectorized year helpers: peak_demand_vec, existing_gas_vec,
           gas_sizing_matrix, operating_cost_matrix
270- 340   solve_pathway: builds (26, n_mixes) cost matrix, argmins,
           accumulates winners/ledger; plus Phase-B peak-year aggregation
340- 370   retirement_timeline wrapper (calls copied compute_fossil_retirement)
370- 400   priced_vre_curtailment + vre_curtailment_at_endpoint
400- 470   serialize_run_result: the 13-key JSON dict
470- 500   CLI: build_argparser, main
```

## Function signatures (outline only — no bodies)

### Stage-1 precompute + EF loading

```python
def precompute_clean_peak_hour_mw(iso: str) -> pa.Table:
    """Return a table of (mix_hash, clean_peak_hour_mw) for every unique EF
    row across all 10 thresholds for this ISO."""
```
Reads the dispatch cache for `iso`, unions the EF rows from all thresholds
under `data/step2.1-ef/step_2_1_EF_{iso}_*.parquet`, deduplicates on the
archetype hash, and for each unique mix emits
`clean_profile_sum[argmin(clean_profile_sum)]` as a single float. Writes
the result to `data/step2.1-ef/step_2_1_EF_{iso}_peakclean.parquet` with a
`source_cache_version` metadata field so reruns are idempotent.

```python
def _load_or_build_peakclean_sidecar(iso: str) -> pa.Table:
    """Load the peakclean sidecar or build it if missing/stale."""
```
Checks the sidecar path; if present and cache-version matches, reads it;
otherwise calls `precompute_clean_peak_hour_mw`. Runs once per ISO at
module import, cached in `_PEAKCLEAN_BY_ISO`.

```python
def load_ef_mixes(iso: str, threshold: int) -> pa.Table:
    """Load EF rows for (iso, threshold) and left-join the peakclean table."""
```
Reads `data/step2.1-ef/step_2_1_EF_{iso}_{threshold}.parquet`, joins
against the in-memory peakclean table on the archetype hash, and zero-pads
any optional resource columns absent for this ISO (`geothermal`,
`offshore_wind`, `ccs_ccgt`).

### Ported fossil-retirement helpers (from `dispatch_utils.py`)

```python
_FOSSIL_CAPACITY_FACTORS = {'coal': 0.55, 'oil': 0.20, 'gas': 0.45}

def _twh_to_gw(twh_per_year: float, capacity_factor: float) -> float:
    """Convert annual TWh to nameplate GW at the given CF."""

def coal_fraction_at_clean_pct(clean_pct: float) -> float:
    """Piecewise-linear coal ramp-down between COAL_PHASE_OUT_START and _END."""

def compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix,
                              demand_growth_factor=1.0, year=None
                              ) -> tuple[float, dict]:
    """Return (displaced_tCO2_per_MWh, retirement_info_dict) for year y."""
```
All four copied verbatim from `dispatch_utils.py` lines 925–1051. Pure
compute — no file I/O inside these functions.

### Vintage tracking + pathway config

```python
@dataclass(frozen=True)
class Vintage:
    resource: str
    cod_year: int
    twh_per_year: float
    locked_lcoe: float
    tx_adder: float
    retire_year: int | None

class VintageLedger:
    """Append-only log of Vintage entries; methods for year-filtered views."""
    def add(self, v: Vintage) -> None: ...
    def active(self, year: int) -> list[Vintage]: ...
    def serialize(self) -> list[dict]: ...       # feeds terminal_ledger JSON key

@dataclass(frozen=True)
class RunConfig:
    iso: str; pathway: str; endpoint: float; endpoint_pct: float
    demand_growth_level: str = 'Medium'
    firm_cost_level: str = 'M'; ccs_cost_level: str = 'M'
    tx_level: str = 'M'; geo_cost_level: str | None = None
    @property
    def output_path(self) -> Path: ...
```

### Vectorized year helpers (the new architecture)

```python
def peak_demand_vec(iso: str, growth_level: str) -> np.ndarray:
    """Return shape (26,) of peak MW per year 2025..2050."""
```
`pc.PEAK_DEMAND_MW[iso] × (1 + pc.DEMAND_GROWTH_RATES[growth_level])**arange(26)`
— pure numpy, no loop.

```python
def existing_gas_available_vec(iso: str, clean_pct_vec: np.ndarray,
                                growth_level: str) -> np.ndarray:
    """Return shape (26,) of existing-gas MW still on-system each year,
    after endogenous retirement."""
```
For each of 26 years, calls the ported `compute_fossil_retirement(iso,
clean_pct[y], ...)` and subtracts the running-max `gas_displaced_twh` from
`pc.EXISTING_GAS_CAPACITY_MW[iso]`. One 26-length loop, but each iteration
is a scalar call — not vectorizable across years because the retirement
rule depends on `clean_pct[y]` which is set by the winning mix at each
year (see §24.8).

```python
def gas_sizing_matrix(peak_vec: np.ndarray,      # shape (26,)
                      existing_gas_vec: np.ndarray,  # shape (26,)
                      clean_arr: np.ndarray,     # shape (n_mixes,)
                      ra_margin: float
                      ) -> np.ndarray:           # shape (26, n_mixes)
    """Per-year-per-mix new gas requirement (MW), fully vectorized."""
```
```
return np.maximum(0,
    peak_vec[:, None] * (1 + ra_margin)
    - clean_arr[None, :]
    - existing_gas_vec[:, None])
```
One numpy op producing the full (26, n_mixes) matrix. This is the
non-negotiable architectural change — no dispatch-utils anywhere in this
path.

```python
def operating_cost_matrix(ef: pa.Table, demand_vec: np.ndarray,
                          iso: str, pathway: str, cfg: RunConfig
                          ) -> np.ndarray:       # shape (26, n_mixes)
    """Per-year-per-mix clean-resource operating cost (USD)."""
```
Pulls each resource column from `ef` as a (n_mixes,) array, looks up
per-year per-resource LCOE from the Wright's-curve table under
`pc.NOAK_YEAR_BY_PATHWAY[pathway]`, and multiplies
`demand[y] × share[m,r] × (lcoe[y,r] + tx_adder[r])` summed over r.
Vectorized as an einsum over (26, n_mixes, n_resources).

```python
def gas_fuel_cost_matrix(ef: pa.Table, demand_vec: np.ndarray,
                         fuel_price_vec: np.ndarray   # shape (26,)
                         ) -> np.ndarray:             # shape (26, n_mixes)
    """Per-year-per-mix gas-fuel USD cost."""
```
Because `gas_mwh = (1 − hourly_match_score/100) × demand`, gas_mw cancels
out of `gas_mwh × fuel_price`: result is
`(1 − score[m]/100) × demand[y] × fuel_price[y]`. Pure broadcast.

### Solver + Phase-B aggregation

```python
def solve_pathway(cfg: RunConfig) -> PathwayRunResult:
    """Run the 26-year optimization for one (iso, pathway, endpoint) triple."""
```
Builds the full (26, n_mixes) cost matrix (operating + gas_capex + gas_fuel
+ existing_fom + pathway-mask penalty), then does per-year argmin in a
26-iteration Python loop whose only non-vectorizable work is (i) reading
`winners[y] = cost_matrix[y, mask_y].argmin()` where `mask_y` reflects
the ratchet floor set by prior winners, and (ii) appending Vintage entries
to the ledger. Phase B (post-loop) computes
`fleet_size_mw = max(new_gas_required_cumulative_mw)`,
`peak_year = argmax`, `active_new_gas_fleet_mw = np.maximum.accumulate`,
`stranded_mw_at_2050`, and `stranded_capex_usd` per SPEC §24.6.

```python
@dataclass
class PathwayRunResult:
    config: RunConfig
    ledger: VintageLedger
    winners: np.ndarray                # shape (26,) int
    new_gas_required_cumulative_mw: np.ndarray   # shape (26,)
    active_new_gas_fleet_mw: np.ndarray          # shape (26,)
    gas_fleet_cf: np.ndarray                     # shape (26,)
    achieved_cfe_pct: np.ndarray                 # shape (26,)
    demand_vec: np.ndarray                       # shape (26,)
    annual_cost_rows: list[dict]                 # len 26
    endpoint_hourly_dispatch: np.ndarray         # shape (8760,)
    feasibility: dict                            # {physical: bool, notes: list[str]}
    stranding_metadata: dict                     # 15 keys
    retirement_timeline: list[dict]              # len 26
    vre_curtailment_at_endpoint: dict            # 7 keys
    endpoint_mix_pct: dict                       # 11 keys
    endpoint_storage_pct: dict                   # 4 keys
    reliability_tax: dict                        # total_demand_mwh, components_usd, total, usd_per_mwh
    headline: dict                               # 7 keys
```

### Output-side helpers

```python
def retirement_timeline(iso, clean_pct_vec, growth_level) -> list[dict]:
    """Return 26 cumulative-retirement rows using ported compute_fossil_retirement."""

def priced_vre_curtailment_per_year(iso, winners, ef, demand_vec, ledger
                                    ) -> np.ndarray:   # shape (26,)
    """Per-year priced curtailment per SPEC §24.7, vintage-LCOE weighted."""

def vre_curtailment_at_endpoint(iso, winner_endpoint) -> dict:
    """Return the 7 fixed keys: solar, wind, offshore_wind, solar_batt{4,8},
    wind_batt{4,8} curtailment fractions at 2050."""

def serialize_run_result(result: PathwayRunResult) -> dict:
    """Build the 13-key JSON dict exactly per PATHWAY_OUTPUT_SCHEMA.md."""

def build_argparser() -> argparse.ArgumentParser: ...
def main(argv: list[str] | None = None) -> int: ...
```

## Numpy arrays flowing through the year loop (8 shapes)

| # | Name | Shape | dtype | Source | Role |
|---|---|---|---|---|---|
| 1 | `clean_peak_hour_mw_arr` | `(n_mixes,)` | float64 | Stage-1 sidecar joined into `ef` | Per-mix scalar: MW of clean at its weakest hour; vectorized argument to `gas_sizing_matrix` |
| 2 | `peak_demand_vec` | `(26,)` | float64 | `pc.PEAK_DEMAND_MW × (1+gf)**y` | Year-indexed peak demand MW |
| 3 | `clean_pct_vec` | `(26,)` | float64 | Filled in-loop from winning EF row's `hourly_match_score` | Feeds `existing_gas_available_vec` (retirement rule) and `gas_fuel_cost_matrix` |
| 4 | `existing_gas_vec` | `(26,)` | float64 | `existing_gas_available_vec(iso, clean_pct_vec, growth)` via ported `compute_fossil_retirement` | Subtracted from residual in `gas_sizing_matrix`; also seeds `retirement_timeline` output |
| 5 | `gas_required_matrix` | `(26, n_mixes)` | float64 | `gas_sizing_matrix(peak_vec, existing_gas_vec, clean_peak_hour_mw_arr, RA)` | Per-year-per-mix new-gas MW; one-shot broadcast |
| 6 | `cost_matrix` | `(26, n_mixes)` | float64 | `operating_cost_matrix + gas_capex_matrix + gas_fuel_cost_matrix + existing_fom_vec[:, None] + penalty_mask` | The full year-mix cost surface the argmin reads |
| 7 | `winners` | `(26,)` | int64 | `cost_matrix[y, mask_y].argmin()` per year | The one scalar-per-year Python loop iteration — everything downstream reads off this index |
| 8 | `new_gas_required_cumulative_mw` | `(26,)` | float64 | `gas_required_matrix[np.arange(26), winners]` | Fed to Phase-B: `fleet_size_mw = max`, `peak_year = argmax`, `active_new_gas_fleet_mw = np.maximum.accumulate`. The §24.6 peak-year-snapshot rule consumes only this vector |

Secondary arrays (not in the hot path but flow through the loop):
- `achieved_cfe_pct: (26,)` — `ef['hourly_match_score'][winners]`.
- `gas_fleet_cf: (26,)` — energy-balance `(1 − cfe/100) × demand / (active_new_gas_fleet_mw × 8760)` with zero-guard.
- `priced_vre_curtailment_vec: (26,)` — from `priced_vre_curtailment_per_year`.
- `endpoint_hourly_dispatch: (8760,)` — read once from the dispatch cache for the winning 2050 archetype.

## Peakclean sidecar shape and flow

Sidecar parquet at `data/step2.1-ef/step_2_1_EF_{ISO}_peakclean.parquet`:

| column | dtype | notes |
|---|---|---|
| `mix_hash` | uint64 | archetype hash over the share columns (join key to EF) |
| `clean_peak_hour_mw` | float32 | MW of clean contribution at `argmin(clean_profile_sum)` |
| `clean_pct_mw_per_twh_demand` | float32 | same value normalized per TWh of annual demand (for cross-demand scenarios) |
| `source_cache_version` | string (metadata) | must match `data/step3-dispatch/{ISO}_dispatch_cache.parquet` |

Row count: union of distinct mix hashes across all 10 thresholds for the
ISO, O(thousands).

Flow into year loop:

```
(module import)
  ├─ _PEAKCLEAN_BY_ISO[iso] = _load_or_build_peakclean_sidecar(iso)   # once per ISO

(per run)
  ├─ ef = load_ef_mixes(iso, threshold)                               # joins peakclean
  ├─ clean_peak_hour_mw_arr = ef['clean_peak_hour_mw'].to_numpy()     # (n_mixes,)
  ├─ peak_vec       = peak_demand_vec(iso, growth)                    # (26,)
  ├─ existing_gas   = existing_gas_available_vec(iso, clean_pct, gf)  # (26,)
  └─ gas_required   = gas_sizing_matrix(peak_vec, existing_gas,
                                        clean_peak_hour_mw_arr, RA)   # (26, n_mixes)
```

No dispatch-utils call anywhere in the gas-sizing path inside the optimizer
run; the only dispatch-cache read at run time is the single 8,760 endpoint
lookup for `endpoint_hourly_dispatch` and the 7-key curtailment table.

## 13-key JSON write map

`serialize_run_result(result)` returns a dict with exactly these keys in
this order. Left column is the schema key; right column is the
`PathwayRunResult` attribute or array slice that feeds it.

| # | JSON key | Source |
|---|---|---|
| 1 | `schema_version` | literal `2` |
| 2 | `run_key` | `f"{cfg.iso}__pathway{cfg.pathway}__ep{cfg.endpoint_pct}"` |
| 3 | `config` | 9-key echo of `cfg` fields (no `q45` per schema v2) |
| 4 | `feasibility` | `result.feasibility` — `{physical: bool, notes: list[str]}` (no `economic` field) |
| 5 | `headline` | built from `result.annual_cost_rows[*].net_annual_cost_usd` (undisc sum + per-rate NPV), `result.achieved_cfe_pct[-1]`, `cfg.endpoint`, inert `pivot={pivoted:False,pivot_year:None,pivot_reason:None}` |
| 6 | `tables` | 4-key dict: |
| 6a | `tables.annual_buildout` | list[26] built from `_derive_delta_vintages(ledger, winners[y])` + `gas_sizing` dict assembled per year from `peak_demand_vec[y]`, `ra_peak_mw = peak × (1+RA)`, `clean_peak_hour_mw_arr[winners[y]]` as `total_clean_peak_mw`, `existing_gas_vec[y]` as `existing_gas_used_mw`, `new_gas_required_cumulative_mw[y]`, `new_gas_built_this_year_mw` (non-zero only at `peak_year`), `active_new_gas_fleet_mw[y]`, `gas_fleet_cf[y]` |
| 6b | `tables.annual_cost` | `result.annual_cost_rows` — list[26] with `demand_twh`, `gross_operating_usd`, `new_gas_annualized_capex_fom_usd = active_new_gas_fleet_mw × pc.NEW_CCGT_COST_KW_YR[iso] × 1000`, `existing_gas_fom_carried_usd = pc.EXISTING_GAS_CAPACITY_MW × pc.EXISTING_GAS_FOM_KW_YR × 1000`, `gas_fuel_usd`, `capacity_rev_netted_usd = 0.0` (Card M' locked), `net_annual_cost_usd`, `achieved_cfe_pct[y]`, `gas_fleet_cf[y]`, `priced_vre_curtailment_usd_this_year` |
| 6c | `tables.endpoint_hourly_dispatch` | `result.endpoint_hourly_dispatch.tolist()` — (8760,) float list |
| 6d | `tables.new_gas_fleet` | list[1] consolidated vintage `{year_built: peak_year, initial_cap_mw: fleet_size_mw, retirement_year: None, stranded_flag: stranded_mw>0, stranded_year: 2050 if stranded else None, stranded_capex_usd: result.stranding_metadata['total_stranded_capex_usd']}`. No `annual_cf` or `recovered_revenue_per_kw` fields |
| 7 | `reliability_tax` | 4-key dict: `total_demand_mwh_2025_2050 = sum(demand_vec × 1e6)`; `components_usd = {new_gas_capex_annualized_usd: Σ new_gas_annualized_capex_fom_usd, new_gas_fom_usd: 0 (bundled in annualized per pc convention), existing_gas_fom_carried_usd: Σ existing_gas_fom_carried_usd, priced_vre_curtailment_usd: Σ priced_vre_curtailment_vec, vre_storage_overbuild_capex_usd: 0.0 (hardcoded per schema + brief)}`; `total_usd = sum(components)`; `usd_per_mwh = total_usd / total_demand_mwh` |
| 8 | `stranding_metadata` | `result.stranding_metadata` — 15 keys. **Live (9):** `methodology='peak_year_snapshot_v2'`, `asset_life_years=pc.NEW_GAS_ASSET_LIFE_YEARS`, `overnight_capex_usd_kw=pc.CCGT_OVERNIGHT_CAPEX_USD_KW`, `fleet_size_mw`, `peak_year`, `new_gas_need_2050_mw=new_gas_required_cumulative_mw[-1]`, `stranded_mw_at_2050=fleet_size_mw − new_gas_need_2050_mw`, `total_new_gas_built_mw=fleet_size_mw`, `total_stranded_capex_usd`. **Inert (6, for back-compat):** `stranded_vintage_count=1 if stranded else 0`, `cf_threshold_default=0.15`, `cf_threshold_sensitivity=[0.10,0.15,0.20]`, `consecutive_years=2`, `required_real_return=0.07`, `reference_cf=0.15` |
| 9 | `retirement_timeline` | `result.retirement_timeline` — list[26] from `retirement_timeline(iso, clean_pct_vec, growth)`; each row has `year`, `clean_pct`, `coal_retired_gw`, `oil_retired_gw`, `gas_retired_gw`, `coal_retired_twh`, `oil_retired_twh`, `gas_retired_twh`. Cumulative via running-max on displaced TWh |
| 10 | `vre_curtailment_at_endpoint` | `result.vre_curtailment_at_endpoint` — 7 fixed keys: `{solar, wind, offshore_wind, solar_batt4, solar_batt8, wind_batt4, wind_batt8}`, each `Σ surplus_<r> / Σ(matched_<r>+surplus_<r>)` at 2050 from dispatch cache; `0.0` for unbuilt |
| 11 | `endpoint_mix_pct` | `result.endpoint_mix_pct` — 11 fixed keys: `{clean_firm, solar, wind, hydro, offshore_wind, geothermal, ccs_ccgt, solar_batt4, solar_batt8, wind_batt4, wind_batt8}`, echoed from `ef` row at `winners[-1]`. Zero for ISO-ineligible resources |
| 12 | `endpoint_storage_pct` | `result.endpoint_storage_pct` — 4 keys: `battery_dispatch_pct, battery8_dispatch_pct, ldes_dispatch_pct, h2_dispatch_pct` from the same endpoint EF row |
| 13 | `terminal_ledger` | `result.ledger.serialize()` — existing-fleet entries at `cod_year=2025, locked_lcoe=0, tx_adder=0` (schema-resolved; fixes the §24.8 `cod_year=2024` bug) plus new-build vintages with their Wright's-curve-locked LCOEs |

Explicitly **NOT** written (schema v2 drops): `terminal_new_gas_fleet`,
`tables.stranding_ledger`, `config.q45`, `feasibility.economic`,
`tables.new_gas_fleet[0].annual_cf`,
`tables.new_gas_fleet[0].recovered_revenue_per_kw`.

## CLI

```
python scripts/step_2_3_pathway_optimizer.py \
    --iso {CAISO|ERCOT|PJM|NYISO|NEISO|MISO|SPP} \
    --pathway {1|1a|2a|2b|3} \
    --endpoint {0.60|0.70|0.75|0.80|0.85|0.90|0.95|0.975|0.99|0.999} \
    [--all]
```

- `--iso`, `--pathway`, `--endpoint` are required unless `--all` is given.
- `--all` runs the full sweep: 7 ISOs × 5 pathways × 10 endpoints = 350
  runs in a single-process for-loop. No threading, no multiprocessing.
  Each run writes its output JSON to
  `analysis/reliability-tax/data/{ISO}/pathway{p}_ep{endpoint_pct}.json`
  atomically (tmp file + rename).
- No `--growth`, `--firm`, `--ccs`, `--tx`, `--geo`, `--q45`, `--seed-run`
  flags. The old optimizer had them; v2 pins growth=Medium, firm=M, ccs=M,
  tx=M, geo=None (CAISO default), q45=on-always (schema drop). Any
  demand-sensitivity or cost-sensitivity run is a separate invocation
  that temporarily edits the RunConfig defaults — no CLI surface.

## Critical files to be modified / created

- **Create:** `scripts/step_2_3_pathway_optimizer.py` (new, ≤500 lines — the
  file specified in this plan).
- **Create on first run:** `data/step2.1-ef/step_2_1_EF_{ISO}_peakclean.parquet`
  (one per ISO; Stage-1 precompute sidecar).
- **Do not touch:** `scripts/step2_2a_cost_optimization.py`,
  `scripts/dispatch_utils.py`, `scripts/pipeline_config.py`,
  `scripts/run_pathway_sweep.py`. The sweep orchestrator's CLI is
  backward-compatible with this new optimizer because its subprocess call
  already uses `--iso --pathway --endpoint` as positional-equivalent flags.
- **Replace (post-verification):** the old
  `scripts/step_2_3_pathway_optimizer.py` (3,909 lines) gets overwritten
  by this plan's output in the same Git commit.

## Verification plan

1. **Smoke test one ISO end-to-end.** Run
   `python scripts/step_2_3_pathway_optimizer.py --iso ERCOT --pathway 1
   --endpoint 0.90` and confirm:
   - Stage-1 sidecar created at
     `data/step2.1-ef/step_2_1_EF_ERCOT_peakclean.parquet` with row count
     > 0 and a `source_cache_version` matching
     `data/step3-dispatch/ERCOT_dispatch_cache.parquet`.
   - Output JSON at
     `analysis/reliability-tax/data/ERCOT/pathway1_ep90.json` parses, has
     exactly 13 top-level keys, `tables` has exactly 4 keys,
     `vre_curtailment_at_endpoint` has exactly 7, `endpoint_mix_pct` has
     exactly 11, `stranding_metadata` has exactly 15 keys,
     `tables.new_gas_fleet` is a list of length 1, and
     `reliability_tax.components_usd.vre_storage_overbuild_capex_usd ==
     0.0`.
   - §24.6 empirical check: `stranding_metadata.fleet_size_mw` near 111 GW
     at ep90 (ERCOT P1 Medium growth reference from SPEC_LOG.md line
     5254).
   - `headline.achieved_cfe_pct >= 90`.

2. **Byte-for-byte schema conformance.** Diff the key set of the new
   output against `PATHWAY_OUTPUT_SCHEMA.md` via a Python one-liner that
   loads the JSON, walks the documented key structure, and asserts
   equality. Any extra key (e.g., a leftover `terminal_new_gas_fleet`) or
   missing key fails the gate.

3. **Timing gate.** Time the ERCOT single-run end-to-end:
   target ≤90 s per run (matches SPEC §24.4 "~90 seconds"). If the new
   optimizer is >10 % slower than the old one on a cache-warm run, the
   architectural change hasn't paid off and the plan is regressed. If it
   is >2× faster (expected given the elimination of per-mix
   dispatch-cache calls), log the delta in `LESSONS.md`.

4. **Sweep parity spot-check.** Run the 5 ERCOT pathways × 3 endpoints
   (0.60, 0.90, 0.999) = 15-run subset. Compare
   `headline.undiscounted_cost_usd` against the last committed 350-run
   sweep's values for the same tuples. Expected agreement: within ±2 % on
   any run (small numerical drift acceptable from the reworked gas-sizing
   path); >10 % deltas require explanation. Discrepancies > ±2 % are
   escalated, not silently accepted.

5. **Downstream chart smoke.** Run
   `python reliability_tax/charts/gen_section3_reliability_tax.py` against
   the new JSONs and confirm it loads without key-missing errors. This is
   the sharpest test that the 13-key contract is honoured — the generator
   reads every top-level key plus `tables.annual_cost`,
   `reliability_tax.components_usd`, and `stranding_metadata`.

6. **Line budget.** `wc -l scripts/step_2_3_pathway_optimizer.py` must
   return ≤500. If it's over, the plan is renegotiated (drop an
   output-side helper, move port helpers to a sibling module) rather
   than shipping a bigger file.


