# Session E — gen_*.py v1→v2 Drift Inventory

**Scope:** `reliability_tax/charts/gen_*.py` (12 files).
**Reference contract:** `reliability_tax/PATHWAY_OUTPUT_SCHEMA.md` (frozen 2026-04-20).
**Method:** literal string match for the six dropped v1 keys + targeted review of
four shape assumptions (`vre_curtailment_at_endpoint`, `endpoint_mix_pct`,
`new_gas_fleet` iteration, `schema_version` check).

**Counts**

- Total hits: **9**
  - Live-code hits (break on v2 payload): **4**
  - Narrative / comment / framing drift (code survives v2, intent rots): **5**
- NEEDS DECISION: **1** (comparative VRE/storage stranding surface in the Sankey)

---

## NEEDS DECISION (must resolve before coding-session picks this up)

| file | line(s) | ambiguity |
|---|---|---|
| `gen_sankey.py` | 88–101, 137–143, 219–222, 244 | `_collect_vre_storage_stranding` pulls per-resource VRE + storage "comparative stranded book value" from `tables.stranding_ledger` (Card K framing: pathway_twh − pathway3_twh). v2 drops `stranding_ledger` entirely. v2's `stranding_metadata` only carries **new-gas** absolute stranding; there is no per-resource VRE/storage stranding surface in the v2 contract. The Sankey currently keeps this as a "secondary data block" alongside the primary new-gas flow. **Decision needed:** drop the secondary block outright (chart becomes new-gas-only), or add a new per-resource VRE/storage stranding computation into the v2 writer (requires new schema field and card coverage). PATHWAY_OUTPUT_SCHEMA.md offers no direct replacement. |

---

## Live-code hits (P1 = section3 / section6 / four_journeys / section2_gas_hump, P2 = other)

| file | line | v1 key or shape assumption | v2 replacement per PATHWAY_OUTPUT_SCHEMA.md "Resolved decision" | priority | notes |
|---|---|---|---|---|---|
| `gen_closing_summary_table.py` | 126 | `feas.get("economic", False)` — reads dropped `feasibility.economic` | Drop the read. Remove `feasible_economic` column (line 53) and `economic_ceiling_note` column (line 99), plus the `econ` / `econ_note` variables and the feasibility-note parsing branch at line 197. v2 row 4: "v2 reports true cost of hitting targets, doesn't opine on affordability." | P2 | Field currently defaults to `False`; v2 payloads won't carry it at all. Will silently return wrong value (always False) rather than KeyError because of the default. |
| `gen_sankey.py` | 64 | `v.get("recovered_revenue_per_kw", 0.0)` on each `new_gas_fleet` vintage — reads dropped sub-field | Drop the read and the `recovered_revenue_per_kw` key in the emitted `stranded_vintages` dict. v2 row 6d: "Drop `recovered_revenue_per_kw` (Card F' revenue-accumulator dead under Card K')." | P2 | Silent zero under v2 because of the default; downstream chart definition string at line 201 ("locks recovered_revenue_per_kw") also becomes obsolete — separate narrative-drift row below. |
| `gen_sankey.py` | 90 (via `DL_BASE.get_pathway_stranding`) | Reads `tables.stranding_ledger` through `data_loader` helper | See NEEDS DECISION row. | P2 | `data_loader.get_pathway_stranding` at line 303 reads `run.get("tables", {}).get("stranding_ledger", []) or []` — under v2 this returns `[]`, so `_collect_vre_storage_stranding` returns `{"by_resource_usd": {}, "total_usd": 0.0}`. No crash; the secondary Sankey block just empties out. |
| `gen_sankey.py` | 57, 107, 159–163, 167–176 | `new_gas_fleet` iterated as list-of-many; builds per-vintage source nodes keyed by `(pathway, year_built)` | v2 row 6d: `new_gas_fleet` is `list[1]` — single consolidated vintage at `peak_year` with `initial_cap_mw = max(new_gas_required_cumulative_mw)`. Per-vintage Sankey collapses to per-(pathway, peak_year) aggregation across ISOs. Rewrite framing: "one source node per (pathway, peak_year)" and update the docstring at lines 18–24. | P2 | Iteration code survives because `list[1]` is a valid input to `for v in ngf`. Narrative rot: header "one node per (pathway, vintage_year)" and SPEC §24.6 callout still imply a multi-vintage ledger. |

---

## Narrative / comment / framing drift (code survives v2, intent rots)

| file | line(s) | v1 assumption in prose / comment | v2 replacement per PATHWAY_OUTPUT_SCHEMA.md | priority | notes |
|---|---|---|---|---|---|
| `gen_section1_worst_hours.py` | 15, 17, 22, 104, 350–352 | Docstring + `meta.note` claim `endpoint_hourly_dispatch is None in all schema_version=1 runs` and the 100 profiles are **derived from** `stranding_ledger.pathway_twh`. Actual code already migrated to `annual_buildout` (line 313), but the prose still says stranding_ledger. | v2 row 6c: `tables.endpoint_hourly_dispatch` is now populated (`list[8760]` of per-hour demand-normalized matched-clean fraction). `tables.stranding_ledger` is gone. Either (a) keep the derived/synthetic methodology but strip all `stranding_ledger` references from prose and change the `schema_version=1` caveat, or (b) rewrite the whole file to extract 100 worst-demand hours from the real `endpoint_hourly_dispatch` series. | P2 | Methodology decision beyond a pure comment fix — but the minimum drift fix is prose-only. |
| `gen_sankey.py` | 7, 10, 13, 89, 201, 243–244, 253–257 | Primary/secondary source callouts in docstring + `meta.note` still describe Card F' "locks recovered_revenue_per_kw" mechanics and `stranding_ledger` as a "retained secondary source". References `cod_year=2024` seed vintages. | v2 row 6d / terminal_ledger row 13: drop recovered_revenue_per_kw, drop stranding_ledger framing; existing-fleet rows are at `cod_year=2025` (sample's 2024 was a bug). Rewrite to "primary_source: tables.new_gas_fleet[0] where stranded_flag=True" and drop the secondary_source line. | P2 | Ships to dashboard meta block — will be visible to readers of the payload. |
| `gen_closing_summary_table.py` | 181–187 | `vre_curtailment_at_endpoint` is read for only `solar` / `wind` with a fallback to `solar_batt4` / `wind_batt4`. | v2 row 10 fixes the schema at **7 keys**: `{solar, wind, offshore_wind, solar_batt4, solar_batt8, wind_batt4, wind_batt8}`. The current two-column surface (`vre_curtailment_solar_pct`, `vre_curtailment_wind_pct`) can stand — the 7-key schema is a superset of what's read — but consider whether to expose the full 7 so hybrid-heavy ISOs don't hide their curtailment inside `solar_batt4`/`wind_batt4` buckets. | P2 | Not breaking; flagged as a shape-assumption follow-up. If left as-is, rename columns to make the "bare-VRE-only, falls back to 4h hybrid" semantics explicit. |
| `gen_closing_summary_table.py` | 53, 99, 197, 207, 253 | Column definitions + output fields `feasible_economic` and `economic_ceiling_note` surface the dropped `feasibility.economic` trigger to the dashboard table. | Drop both columns from `COLUMNS` and from the returned row. See live-code row above for the paired read. | P2 | Same fix as the live-code hit; listing separately because it also touches the column schema / JSON contract consumed by the dashboard. |
| `gen_section2_gas_hump.py` | 22, 118–121 | Docstring + `meta.stranded_method` phrased as "sum initial_cap_mw for vintages with stranded_flag=True" — implies multiple stranded vintages. | Still correct under v2 (the sum degenerates to at most one term), but the phrasing should acknowledge v2 row 6d's `list[1]`. | P1 | Code works; prose should be tightened to "the single v2 vintage if `stranded_flag=True`, else 0." |

---

## Clean files (no hits)

`gen_hook.py`, `gen_hockey_stick.py`, `gen_four_journeys.py`, `gen_section1_narratives.py`, `gen_act1_trilemma.py`, `gen_section2_overbuild.py`, `gen_section3_reliability_tax.py`, `gen_section6_cost_of_waiting.py`.

All consume only v2-surviving keys (`feasibility.physical`, `feasibility.notes`, `tables.annual_buildout`, `tables.annual_cost`, `reliability_tax.*`, `retirement_timeline`, `undiscounted_cost_usd`, `npv_at_*pct`, `achieved_cfe_pct`). `gen_section2_overbuild.py` iterates `new_gas_fleet` via sum/filter (degenerate on `list[1]`, no framing rot). None of the eight read `stranding_ledger`, `recovered_revenue_per_kw`, `annual_cf`, `terminal_new_gas_fleet`, `config.q45`, or `feasibility.economic`. None check `schema_version` — uniformly absent, treated as benign (no file asserts `==1`).

---

## Literal-key grep coverage (for audit)

| key | occurrences across `reliability_tax/charts/gen_*.py` |
|---|---|
| `terminal_new_gas_fleet` | 0 |
| `stranding_ledger` | 7 (all in `gen_sankey.py` live-code path via data_loader + doc strings; `gen_section1_worst_hours.py` comments only) |
| `q45` (case-insensitive) | 0 |
| `recovered_revenue_per_kw` | 3 (all in `gen_sankey.py`) |
| `annual_cf` | 0 (verified no false positives against `annual_cost`) |
| `feasibility.economic` (via `.get("economic"`) | 1 (`gen_closing_summary_table.py:126`) |

**Shared helper drift (out of gen_*.py scope, worth flagging):** `reliability_tax/charts/data_loader.py` lines 183, 248, 273, 303, 362, 395–403, 454, 577 reference `feasibility.economic` / `stranding_ledger`. Any gen_*.py that calls `get_pathway_stranding` or `is_feasible(..., "economic")` inherits that drift. Fixing `data_loader.py` is a single choke point but lives outside this inventory.
