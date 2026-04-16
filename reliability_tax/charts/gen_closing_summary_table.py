"""
gen_closing_summary_table.py — Generate closing_summary_table.json

Data lineage:
  - analysis/reliability-tax/data/MANIFEST.json + per-run JSONs
  - Per MANIFEST.json required fields (methodology.md):
    ISO, pathway, endpoint,
    feasibility (physical, economic), achieved_cfe_pct,
    undiscounted_cost_usd, npv_at_5pct, npv_at_7pct, npv_at_9pct,
    retirement_timeline (cumulative GW retired by resource, 2050),
    comparative_stranding (GW and $ beyond Pathway 3),
    vre_curtailment (per-resource rates at endpoint)
  - Per Card J: physical_feasibility_ceiling and economic_feasibility_ceiling columns
  - Per Card L revised: existing_fleet_retirement_cumulative_gw_2050 column
  - Per Card K revised: comparative_stranding columns

Full table = 140 rows (7 ISO × 4 pathway × 5 endpoint).

Payload structure:
  columns: list[column_definition]  — name, description, unit
  rows: list[140 × dict]            — one row per run, all columns
  aggregations: dict                — cross-cuts and summary stats
  meta: ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import (  # noqa: E402
    DL_BASE,
    get_run, get_pathway_stranding, get_pivot_info,
    ISOS, PATHWAYS, ENDPOINTS, _EP_LABEL,
)

OUT_PATH = Path(__file__).parent / "closing_summary_table.json"

EXISTING_GAS_GW = {
    "CAISO": 37.0, "ERCOT": 55.0, "PJM": 75.0,
    "NYISO": 18.0, "NEISO": 14.0, "MISO": 68.0, "SPP": 32.0,
}

# Card S: new-build gas stranding parameters
CCS_CCGT_CF = 0.85
CCS_CCGT_CAPITAL_PER_GW = 2.2e9   # $2,200/kW = $2.2B/GW (NREL ATB 2024 moderate)
CCS_CCGT_USEFUL_LIFE = 25          # years
MODEL_ENDPOINT_YEAR = 2050


def _new_gas_stranding(run: dict) -> tuple[float, float]:
    """
    Return (stranded_gw, stranded_usd) for new CCS-CCGT built and not fully amortized by 2050.

    For each vintage built in year Y: stranded_fraction = max(0, (Y + 25 - 2050) / 25).
    Plants built 2025 have 0 stranding; plants built 2032 have 7/25 = 28% stranding.
    Capital basis: $2,200/kW overnight (NREL ATB 2024 moderate).
    """
    bo = run.get("tables", {}).get("annual_buildout", [])
    total_stranded_gw = 0.0
    total_stranded_usd = 0.0
    for yr_row in bo:
        yr = yr_row.get("year", MODEL_ENDPOINT_YEAR)
        for v in yr_row.get("new_vintages", []):
            if v.get("resource") != "ccs_ccgt":
                continue
            twh = v.get("twh_per_year", 0.0)
            if twh <= 0:
                continue
            gw = twh / (8.76 * CCS_CCGT_CF)
            stranded_frac = max(0.0, (yr + CCS_CCGT_USEFUL_LIFE - MODEL_ENDPOINT_YEAR) / CCS_CCGT_USEFUL_LIFE)
            total_stranded_gw += gw * stranded_frac
            total_stranded_usd += gw * CCS_CCGT_CAPITAL_PER_GW * stranded_frac
    return round(total_stranded_gw, 2), round(total_stranded_usd, 0)

COLUMNS = [
    {"name": "run_key",            "description": "Unique run identifier", "unit": "string"},
    {"name": "iso",                "description": "ISO/RTO region", "unit": "string"},
    {"name": "pathway",            "description": "Decarbonization pathway ID", "unit": "string"},
    {"name": "endpoint",           "description": "CFE target endpoint (fraction)", "unit": "fraction"},
    {"name": "endpoint_pct",       "description": "CFE target (% label)", "unit": "%"},
    {"name": "feasible_physical",  "description": "Physically achievable (Card J)", "unit": "bool"},
    {"name": "feasible_economic",  "description": "Below $10k/CFE% ceiling (Card J)", "unit": "bool"},
    {"name": "achieved_cfe_pct",   "description": "Achieved hourly CFE at endpoint year (2050)", "unit": "%"},
    # Primary cost metric: absolute actual cost (Card S)
    {"name": "undiscounted_cost_trillion", "description": "PRIMARY: Cumulative system operating cost 2025–2050 (undiscounted, real 2025$)", "unit": "$T"},
    {"name": "new_gas_stranded_gw",       "description": "New CCS-CCGT capacity with undepreciated book value at 2050 (Card S)", "unit": "GW"},
    {"name": "new_gas_stranded_billion",  "description": "Stranded book value of new-build CCS-CCGT at 2050 (Card S, $2,200/kW capital)", "unit": "$B"},
    {"name": "total_actual_cost_trillion","description": "PRIMARY TOTAL: undiscounted_cost + new_gas_stranded. Translates to customer rates (Card S)", "unit": "$T"},
    # Secondary cost metrics: NPV (investment comparison)
    {"name": "npv_5pct_trillion",  "description": "SECONDARY: NPV at 5% real discount rate", "unit": "$T"},
    {"name": "npv_7pct_trillion",  "description": "SECONDARY: NPV at 7% real discount rate (optimizer objective)", "unit": "$T"},
    {"name": "npv_9pct_trillion",  "description": "SECONDARY: NPV at 9% real discount rate", "unit": "$T"},
    {"name": "pivoted",            "description": "Did clean firm unlock during planning period?", "unit": "bool"},
    {"name": "pivot_year",         "description": "Year clean firm unlocked (null if not pivoted)", "unit": "year"},
    {"name": "pivot_reason",       "description": "Pivot trigger: 90pct_plateau | econ_trigger | null", "unit": "string"},
    {"name": "demand_2050_twh",    "description": "Annual demand at 2050 endpoint year", "unit": "TWh/yr"},
    # Retirement (Card L revised)
    {"name": "coal_retired_cumulative_gw", "description": "Cumulative coal capacity economically displaced by 2050", "unit": "GW"},
    {"name": "oil_retired_cumulative_gw",  "description": "Cumulative oil capacity economically displaced by 2050", "unit": "GW"},
    {"name": "gas_retired_cumulative_gw",  "description": "Cumulative gas capacity economically displaced by 2050", "unit": "GW"},
    {"name": "gas_remaining_gw",   "description": "Existing gas fleet still active at 2050", "unit": "GW"},
    # Stranding (Card K revised)
    {"name": "total_stranded_twh",  "description": "Total stranded capacity (comparative-to-P3, TWh/yr generation equivalent)", "unit": "TWh/yr"},
    {"name": "total_stranded_billion_usd", "description": "Book value of stranded capital at 2050", "unit": "$B"},
    {"name": "vre_curtailment_solar_pct", "description": "Solar curtailment rate at endpoint", "unit": "%"},
    {"name": "vre_curtailment_wind_pct",  "description": "Wind curtailment rate at endpoint", "unit": "%"},
    # Feasibility ceilings (Card J)
    {"name": "physical_ceiling_note", "description": "Physical feasibility ceiling note", "unit": "string"},
    {"name": "economic_ceiling_note", "description": "Economic ceiling note ($10k/CFE% trigger)", "unit": "string"},
]


def _build_row(iso: str, pathway: str, ep: float) -> dict:
    """Build one summary table row."""
    ep_label = _EP_LABEL[ep]
    run_key = f"{iso}__pathway{pathway}__{ep_label}"

    try:
        run = DL_BASE.get_run(iso, pathway, ep)
    except (KeyError, FileNotFoundError):
        return {
            "run_key": run_key, "iso": iso, "pathway": pathway,
            "endpoint": ep, "endpoint_pct": f"{ep * 100:.1f}%",
            "data_missing": True,
        }

    feas = run.get("feasibility", {})
    phys = bool(feas.get("physical", False))
    econ = bool(feas.get("economic", False))
    feas_notes = feas.get("notes", [])

    # Costs
    undiscounted = run.get("undiscounted_cost_usd", 0.0) / 1e12
    npv5 = run.get("npv_at_5pct", 0.0) / 1e12
    npv7 = run.get("npv_at_7pct", 0.0) / 1e12
    npv9 = run.get("npv_at_9pct", 0.0) / 1e12

    # New-build gas stranding (Card S)
    ng_stranded_gw, ng_stranded_usd = _new_gas_stranding(run)
    total_actual_cost = undiscounted + ng_stranded_usd / 1e12

    # Pivot info
    pivot = get_pivot_info(run)

    # Demand 2050
    cost_rows = run.get("tables", {}).get("annual_cost", [])
    demand_2050 = cost_rows[-1]["demand_twh"] if cost_rows else None

    # Retirements (Card L revised)
    rt = run.get("retirement_timeline", [])
    if rt:
        rt_2050 = rt[-1]
        coal_retired = rt_2050.get("coal_retired_gw", 0.0)
        oil_retired = rt_2050.get("oil_retired_gw", 0.0)
        gas_retired = rt_2050.get("gas_retired_gw", 0.0)
    else:
        coal_retired = oil_retired = gas_retired = None

    gas_remaining = (
        round(max(0.0, EXISTING_GAS_GW.get(iso, 0.0) - (gas_retired or 0.0)), 1)
        if gas_retired is not None else None
    )

    # Stranding (Card K revised)
    strand = get_pathway_stranding(iso, pathway, ep)
    total_stranded_usd = strand["total_stranded_usd"]
    total_stranded_twh = strand["total_stranded_twh"]
    curtailment = strand.get("vre_curtailment", {})
    solar_curt = curtailment.get("solar", curtailment.get("solar_batt4", None))
    wind_curt = curtailment.get("wind", curtailment.get("wind_batt4", None))

    # Physical and economic ceiling notes (Card J)
    phys_note = None
    econ_note = None
    for note in feas_notes:
        if isinstance(note, str):
            if "physical" in note.lower():
                phys_note = note
            elif "economic" in note.lower() or "10000" in note or "10k" in note.lower():
                econ_note = note

    return {
        "run_key": run_key,
        "iso": iso,
        "pathway": pathway,
        "endpoint": ep,
        "endpoint_pct": f"{ep * 100:.1f}%",
        "feasible_physical": phys,
        "feasible_economic": econ,
        "achieved_cfe_pct": round(run.get("achieved_cfe_pct", 0.0), 2),
        "undiscounted_cost_trillion": round(undiscounted, 4),
        "new_gas_stranded_gw": ng_stranded_gw,
        "new_gas_stranded_billion": round(ng_stranded_usd / 1e9, 2),
        "total_actual_cost_trillion": round(total_actual_cost, 4),
        "npv_5pct_trillion": round(npv5, 4),
        "npv_7pct_trillion": round(npv7, 4),
        "npv_9pct_trillion": round(npv9, 4),
        "pivoted": pivot.get("pivoted", False),
        "pivot_year": pivot.get("pivot_year"),
        "pivot_reason": pivot.get("pivot_reason"),
        "demand_2050_twh": round(demand_2050, 1) if demand_2050 is not None else None,
        # Retirement (Card L)
        "coal_retired_cumulative_gw": round(coal_retired, 2) if coal_retired is not None else None,
        "oil_retired_cumulative_gw": round(oil_retired, 2) if oil_retired is not None else None,
        "gas_retired_cumulative_gw": round(gas_retired, 2) if gas_retired is not None else None,
        "gas_remaining_gw": gas_remaining,
        # Stranding (Card K)
        "total_stranded_twh": round(total_stranded_twh, 1),
        "total_stranded_billion_usd": round(total_stranded_usd / 1e9, 2),
        "vre_curtailment_solar_pct": round(solar_curt * 100, 1) if solar_curt is not None else None,
        "vre_curtailment_wind_pct": round(wind_curt * 100, 1) if wind_curt is not None else None,
        # Feasibility ceilings (Card J)
        "physical_ceiling_note": phys_note,
        "economic_ceiling_note": econ_note,
    }


def _compute_aggregations(rows: list[dict]) -> dict:
    """Compute cross-cut summary statistics."""
    feasible_rows = [r for r in rows if r.get("feasible_physical")]

    # By ISO: best (P3) and worst (P1) NPV@7% at ep99
    iso_summary: dict[str, dict] = {}
    for iso in ISOS:
        iso_rows = [r for r in feasible_rows if r["iso"] == iso and r["endpoint"] == 0.99]
        if not iso_rows:
            continue
        p3_rows = [r for r in iso_rows if r["pathway"] == "3"]
        p1_rows = [r for r in iso_rows if r["pathway"] == "1a"]
        if p3_rows and p1_rows:
            p3 = p3_rows[0]
            p1 = p1_rows[0]
            iso_summary[iso] = {
                "p3_npv7_trillion": p3["npv_7pct_trillion"],
                "p1_npv7_trillion": p1["npv_7pct_trillion"],
                "delta_p1_vs_p3_trillion": round(p1["npv_7pct_trillion"] - p3["npv_7pct_trillion"], 4),
                "p1_total_stranded_billion": p1["total_stranded_billion_usd"],
                "max_gas_retired_gw": max(
                    (r["gas_retired_cumulative_gw"] or 0)
                    for r in iso_rows
                    if r.get("gas_retired_cumulative_gw") is not None
                ),
            }

    # By pathway: average cost delta vs P3 across all ISOs at ep99
    pathway_summary: dict[str, dict] = {}
    for path in PATHWAYS:
        path_rows = [r for r in feasible_rows if r["pathway"] == path and r["endpoint"] == 0.99]
        p3_by_iso = {r["iso"]: r["npv_7pct_trillion"]
                     for r in feasible_rows if r["pathway"] == "3" and r["endpoint"] == 0.99}
        if path_rows:
            deltas = [
                r["npv_7pct_trillion"] - p3_by_iso.get(r["iso"], r["npv_7pct_trillion"])
                for r in path_rows if r["iso"] in p3_by_iso
            ]
            pathway_summary[path] = {
                "count": len(path_rows),
                "avg_npv7_delta_vs_p3_trillion": round(sum(deltas) / len(deltas), 4) if deltas else None,
                "total_stranded_billion_sum": round(
                    sum(r["total_stranded_billion_usd"] for r in path_rows), 1
                ),
            }

    # Feasibility counts
    total_runs = len(rows)
    feas_count = sum(1 for r in rows if r.get("feasible_physical"))
    infeas_count = total_runs - feas_count

    return {
        "total_runs": total_runs,
        "feasible_runs": feas_count,
        "infeasible_runs": infeas_count,
        "by_iso_at_ep99": iso_summary,
        "by_pathway_at_ep99": pathway_summary,
    }


def main() -> None:
    rows = []
    for iso in ISOS:
        for pathway in PATHWAYS:
            for ep in ENDPOINTS:
                row = _build_row(iso, pathway, ep)
                rows.append(row)

    agg = _compute_aggregations(rows)

    payload = {
        "columns": COLUMNS,
        "row_count": len(rows),
        "rows": rows,
        "aggregations": agg,
        "meta": {
            "payload_id": "closing_summary_table",
            "note": (
                "Full cross-ISO × pathway × endpoint table. "
                f"{len(rows)} rows (7 ISO × 5 pathway × 5 endpoint). "
                "Pathways: 1a (onshore VRE+storage only, Card R), 1b (+offshore wind), 2a, 2b, 3. "
                "All MANIFEST.json fields plus derived metrics per methodology.md. "
                "Stranding per Card K revised (comparative-to-P3, VRE+storage+gas only). "
                "Retirement per Card L revised (endogenous economic retirement cumulative GW). "
                "Feasibility ceilings per Card J (physical + economic)."
            ),
            "schema_version": "1.0",
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH} ({len(rows)} rows)")

    print("\n--- closing_summary_table quick summary ---")
    print(f"Total rows: {len(rows)}")
    print(f"Feasible: {agg['feasible_runs']} | Infeasible: {agg['infeasible_runs']}")
    print("\nBy ISO at ep99% (P1 vs P3 NPV@7% delta):")
    for iso, d in agg["by_iso_at_ep99"].items():
        print(
            f"  {iso}: P3=${d['p3_npv7_trillion']:.3f}T, P1=${d['p1_npv7_trillion']:.3f}T, "
            f"Δ={d['delta_p1_vs_p3_trillion']:+.3f}T, stranded=${d['p1_total_stranded_billion']:.0f}B"
        )
    print("\nBy pathway at ep99% (avg delta vs P3):")
    for path, d in agg["by_pathway_at_ep99"].items():
        avg_delta = d.get('avg_npv7_delta_vs_p3_trillion') or 0
        print(
            f"  P{path}: avg_delta={avg_delta:+.3f}T, "
            f"total_stranded=${d['total_stranded_billion_sum']:.0f}B"
        )


if __name__ == "__main__":
    main()
