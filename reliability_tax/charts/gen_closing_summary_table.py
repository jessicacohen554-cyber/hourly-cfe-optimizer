"""
gen_closing_summary_table.py — Generate closing_summary_table.json

v2 (Apr 2026). 175 rows = 7 ISO × 5 pathway × 5 endpoint.

Endpoints chosen for the closing table are the five headline thresholds:
  75%, 85%, 90%, 95%, 99% — spanning mid-CFE through the last-mile zone.

All columns surface v2 methodology:
  - Reliability tax ($/MWh) per SPEC §24.4 locked 5-component formula
  - Absolute stranded new-build gas capex per Card K' (from new_gas_fleet)
  - Worst-hour-sized gas fleet MW at 2050 per SPEC §24.5
  - Retirement timeline (Card L')
  - Gross cost only — no capacity-market netting (Card M')

Dropped from v1:
  - total_stranded_billion_usd (old Card K comparative framing)
  - total_stranded_twh
  - net_annual_cost / capacity_rev_netted surfaces
  - Card S CCS-CCGT synthetic stranding proxy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import (  # noqa: E402
    DL_BASE, get_pivot_info,
    ISOS, PATHWAYS, _EP_LABEL,
)

OUT_PATH = Path(__file__).parent / "closing_summary_table.json"
DASH_OUT = REPO_ROOT / "dashboard" / "js" / "reliability-tax" / "closing_summary_table.json"

# Four scheduled endpoints — each solved on its own SBTi-ladder horizon
# (90→2040, 95→2045, 99→2049, 99.9→2050). Must match ENDPOINT_TO_THRESHOLD
# in step_2_3_pathway_optimizer.py.
CLOSING_ENDPOINTS = [0.90, 0.95, 0.99, 0.999]

EXISTING_GAS_GW = {
    "CAISO": 37.0, "ERCOT": 55.0, "PJM": 75.0,
    "NYISO": 18.0, "NEISO": 14.0, "MISO": 68.0, "SPP": 32.0,
}

COLUMNS = [
    {"name": "run_key",            "description": "Unique run identifier", "unit": "string"},
    {"name": "iso",                "description": "ISO/RTO region", "unit": "string"},
    {"name": "pathway",            "description": "Decarbonization pathway (1/1a/2a/2b/3)", "unit": "string"},
    {"name": "endpoint",           "description": "CFE target endpoint (fraction)", "unit": "fraction"},
    {"name": "endpoint_pct",       "description": "CFE target (% label)", "unit": "%"},
    {"name": "feasible_physical",  "description": "Physically achievable (Card J)", "unit": "bool"},
    {"name": "feasible_economic",  "description": "Below $10k/CFE% ceiling (Card J)", "unit": "bool"},
    {"name": "achieved_cfe_pct",   "description": "Achieved hourly CFE at 2050 endpoint year", "unit": "%"},

    # Reliability tax (SPEC §24.4 — primary metric)
    {"name": "reliability_tax_usd_per_mwh",      "description": "PRIMARY: SPEC §24.4 5-component reliability tax divided by 2025-2050 demand MWh", "unit": "$/MWh"},
    {"name": "reliability_tax_total_usd",        "description": "PRIMARY: SPEC §24.4 reliability tax total $ (gross, no cap-market netting)", "unit": "$"},
    {"name": "rtax_new_gas_capex_usd",           "description": "§24.4 component 1: annualized new-gas capex × 26 yrs", "unit": "$"},
    {"name": "rtax_new_gas_fom_usd",             "description": "§24.4 component 2: new-gas fixed O&M", "unit": "$"},
    {"name": "rtax_existing_gas_fom_carried_usd","description": "§24.4 component 3: existing-gas FOM carried for reliability", "unit": "$"},
    {"name": "rtax_priced_vre_curtailment_usd",  "description": "§24.4 component 4: priced VRE curtailment (chart layer)", "unit": "$"},
    {"name": "rtax_vre_storage_overbuild_usd",   "description": "§24.4 component 5: annualized VRE+storage overbuild capex (chart layer)", "unit": "$"},

    # Worst-hour sized gas fleet (SPEC §24.5)
    {"name": "active_new_gas_fleet_mw_2050",     "description": "Worst-hour sized new-build gas fleet at 2050 (SPEC §24.5 99.97 pct residual gap, NOT ELCC)", "unit": "MW"},
    {"name": "residual_gap_p99p97_mw_2050",      "description": "99.97 pct of (demand × RA margin − clean dispatch) at 2050", "unit": "MW"},
    {"name": "existing_gas_carried_mw_2050",     "description": "Existing gas still needed on the worst hour at 2050", "unit": "MW"},

    # Card K' absolute stranded new-build gas
    {"name": "stranded_new_gas_mw",              "description": "Card K' — Σ initial_cap_mw of new-gas vintages with stranded_flag=True (CF<15% for 2 yrs, SPEC Card F')", "unit": "MW"},
    {"name": "stranded_new_gas_capex_billion",   "description": "Card K' — absolute stranded capex on new-build gas at 2050", "unit": "$B"},
    {"name": "stranded_new_gas_vintage_count",   "description": "Number of new-gas vintages that tripped the Card F' stranding trigger", "unit": "count"},

    # Gross cost (Card M')
    {"name": "undiscounted_gross_cost_trillion", "description": "Cumulative 2025-2050 gross ratepayer cost (no cap-market netting)", "unit": "$T"},
    {"name": "npv_at_7pct_gross_trillion",       "description": "Gross ratepayer NPV @ 7% real discount rate", "unit": "$T"},
    {"name": "npv_at_5pct_gross_trillion",       "description": "Gross ratepayer NPV @ 5% real discount rate", "unit": "$T"},
    {"name": "npv_at_9pct_gross_trillion",       "description": "Gross ratepayer NPV @ 9% real discount rate", "unit": "$T"},

    # Pivot (Pathway 2a/2b only)
    {"name": "pivoted",            "description": "Did clean firm unlock during the planning period?", "unit": "bool"},
    {"name": "pivot_year",         "description": "Year clean firm unlocked (null if not pivoted)", "unit": "year"},
    {"name": "pivot_reason",       "description": "Pivot trigger: 90pct_plateau | econ_trigger | null", "unit": "string"},

    # Demand + retirements (Card L')
    {"name": "demand_2050_twh",    "description": "Annual demand at 2050 endpoint year", "unit": "TWh/yr"},
    {"name": "coal_retired_cumulative_gw", "description": "Cumulative coal capacity economically displaced by 2050", "unit": "GW"},
    {"name": "oil_retired_cumulative_gw",  "description": "Cumulative oil capacity economically displaced by 2050", "unit": "GW"},
    {"name": "gas_retired_cumulative_gw",  "description": "Cumulative gas capacity economically displaced by 2050", "unit": "GW"},
    {"name": "gas_remaining_gw",   "description": "Existing gas fleet still active at 2050 (EXISTING_GAS − gas_retired)", "unit": "GW"},

    # Curtailment diagnostic
    {"name": "vre_curtailment_solar_pct", "description": "Solar curtailment rate at endpoint", "unit": "%"},
    {"name": "vre_curtailment_wind_pct",  "description": "Wind curtailment rate at endpoint", "unit": "%"},

    # Feasibility ceilings (Card J)
    {"name": "physical_ceiling_note", "description": "Physical feasibility ceiling note", "unit": "string"},
    {"name": "economic_ceiling_note", "description": "Economic ceiling note ($10k/CFE% trigger)", "unit": "string"},
]

COMPONENT_KEYS = [
    "new_gas_capex_annualized_usd",
    "new_gas_fom_usd",
    "existing_gas_fom_carried_usd",
    "priced_vre_curtailment_usd",
    "vre_storage_overbuild_capex_usd",
]


def _build_row(iso: str, pathway: str, ep: float) -> dict:
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
    feas_notes = feas.get("notes", []) or []

    # Gross cost columns (Card M') — all surface undiscounted_cost_usd and npv_at_Xpct
    # which the solver reports gross already (capacity_rev_netted is 0 in v2).
    undiscounted = float(run.get("undiscounted_cost_usd", 0.0)) / 1e12
    npv5 = float(run.get("npv_at_5pct", 0.0)) / 1e12
    npv7 = float(run.get("npv_at_7pct", 0.0)) / 1e12
    npv9 = float(run.get("npv_at_9pct", 0.0)) / 1e12

    # Reliability tax (SPEC §24.4)
    rt = run.get("reliability_tax", {}) or {}
    rt_comps = rt.get("components_usd", {}) or {}
    rtax_total = float(rt.get("total_usd", 0.0))
    rtax_per_mwh = float(rt.get("usd_per_mwh", 0.0))

    # Worst-hour sized gas fleet from terminal annual_buildout row (SPEC §24.5)
    bo = run.get("tables", {}).get("annual_buildout", []) or []
    if bo:
        terminal_gas = bo[-1].get("gas_sizing", {}) or {}
        active_ng_mw = float(terminal_gas.get("active_new_gas_fleet_mw", 0.0))
        residual_gap = float(terminal_gas.get("residual_gap_p99p97_mw", 0.0))
        existing_carried = float(terminal_gas.get("existing_gas_carried_mw", 0.0))
    else:
        active_ng_mw = residual_gap = existing_carried = 0.0

    # Card K' — absolute stranded new-gas capex
    ngf = run.get("tables", {}).get("new_gas_fleet", []) or []
    stranded_vintages = [v for v in ngf if v.get("stranded_flag")]
    stranded_ng_mw = sum(float(v.get("initial_cap_mw", 0.0)) for v in stranded_vintages)
    stranded_ng_capex = sum(float(v.get("stranded_capex_usd", 0.0)) for v in stranded_vintages)

    # Pivot
    pivot = get_pivot_info(run)

    # Demand
    cost_rows = run.get("tables", {}).get("annual_cost", []) or []
    demand_2050 = cost_rows[-1].get("demand_twh") if cost_rows else None

    # Retirements (Card L')
    rt_tl = run.get("retirement_timeline", []) or []
    if rt_tl:
        rt_2050 = rt_tl[-1]
        coal_retired = float(rt_2050.get("coal_retired_gw", 0.0))
        oil_retired = float(rt_2050.get("oil_retired_gw", 0.0))
        gas_retired = float(rt_2050.get("gas_retired_gw", 0.0))
    else:
        coal_retired = oil_retired = gas_retired = None

    gas_remaining = (
        round(max(0.0, EXISTING_GAS_GW.get(iso, 0.0) - (gas_retired or 0.0)), 1)
        if gas_retired is not None else None
    )

    # Curtailment diagnostic
    curtailment = run.get("vre_curtailment_at_endpoint", {}) or {}
    solar_curt = curtailment.get("solar")
    wind_curt = curtailment.get("wind")
    if solar_curt is None:
        solar_curt = curtailment.get("solar_batt4")
    if wind_curt is None:
        wind_curt = curtailment.get("wind_batt4")

    # Feasibility notes
    phys_note = None
    econ_note = None
    for note in feas_notes:
        if isinstance(note, str):
            low = note.lower()
            if "physical" in low and phys_note is None:
                phys_note = note
            elif ("economic" in low or "10000" in note or "10k" in low) and econ_note is None:
                econ_note = note

    return {
        "run_key": run_key,
        "iso": iso,
        "pathway": pathway,
        "endpoint": ep,
        "endpoint_pct": f"{ep * 100:.1f}%",
        "feasible_physical": phys,
        "feasible_economic": econ,
        "achieved_cfe_pct": round(float(run.get("achieved_cfe_pct", 0.0)), 2),

        # Reliability tax
        "reliability_tax_usd_per_mwh": round(rtax_per_mwh, 2),
        "reliability_tax_total_usd": round(rtax_total, 0),
        "rtax_new_gas_capex_usd": round(float(rt_comps.get("new_gas_capex_annualized_usd", 0.0)), 0),
        "rtax_new_gas_fom_usd": round(float(rt_comps.get("new_gas_fom_usd", 0.0)), 0),
        "rtax_existing_gas_fom_carried_usd": round(float(rt_comps.get("existing_gas_fom_carried_usd", 0.0)), 0),
        "rtax_priced_vre_curtailment_usd": round(float(rt_comps.get("priced_vre_curtailment_usd", 0.0)), 0),
        "rtax_vre_storage_overbuild_usd": round(float(rt_comps.get("vre_storage_overbuild_capex_usd", 0.0)), 0),

        # Worst-hour gas fleet
        "active_new_gas_fleet_mw_2050": round(active_ng_mw, 0),
        "residual_gap_p99p97_mw_2050": round(residual_gap, 0),
        "existing_gas_carried_mw_2050": round(existing_carried, 0),

        # Card K' stranding
        "stranded_new_gas_mw": round(stranded_ng_mw, 0),
        "stranded_new_gas_capex_billion": round(stranded_ng_capex / 1e9, 2),
        "stranded_new_gas_vintage_count": len(stranded_vintages),

        # Gross cost
        "undiscounted_gross_cost_trillion": round(undiscounted, 4),
        "npv_at_7pct_gross_trillion": round(npv7, 4),
        "npv_at_5pct_gross_trillion": round(npv5, 4),
        "npv_at_9pct_gross_trillion": round(npv9, 4),

        # Pivot
        "pivoted": pivot.get("pivoted", False),
        "pivot_year": pivot.get("pivot_year"),
        "pivot_reason": pivot.get("pivot_reason"),

        # Demand + retirements
        "demand_2050_twh": round(demand_2050, 1) if demand_2050 is not None else None,
        "coal_retired_cumulative_gw": round(coal_retired, 2) if coal_retired is not None else None,
        "oil_retired_cumulative_gw": round(oil_retired, 2) if oil_retired is not None else None,
        "gas_retired_cumulative_gw": round(gas_retired, 2) if gas_retired is not None else None,
        "gas_remaining_gw": gas_remaining,

        # Curtailment
        "vre_curtailment_solar_pct": round(solar_curt * 100, 1) if solar_curt is not None else None,
        "vre_curtailment_wind_pct": round(wind_curt * 100, 1) if wind_curt is not None else None,

        # Feasibility ceilings
        "physical_ceiling_note": phys_note,
        "economic_ceiling_note": econ_note,
    }


def _compute_aggregations(rows: list[dict]) -> dict:
    feasible_rows = [r for r in rows if r.get("feasible_physical")]

    # Per-ISO at ep90: P1 vs P3 reliability tax delta. ep90 is the §24.8
    # "central finding" endpoint — ERCOT P1≡P3, PJM P3 saves 66.7%.
    iso_summary: dict[str, dict] = {}
    for iso in ISOS:
        iso_rows = [r for r in feasible_rows if r["iso"] == iso and r["endpoint"] == 0.90]
        p3 = next((r for r in iso_rows if r["pathway"] == "3"), None)
        p1 = next((r for r in iso_rows if r["pathway"] == "1"), None)
        if p3 and p1:
            p1_rtax = p1["reliability_tax_usd_per_mwh"]
            p3_rtax = p3["reliability_tax_usd_per_mwh"]
            p1_cost = p1["undiscounted_gross_cost_trillion"]
            p3_cost = p3["undiscounted_gross_cost_trillion"]
            iso_summary[iso] = {
                "p3_rtax_usd_per_mwh": p3_rtax,
                "p1_rtax_usd_per_mwh": p1_rtax,
                "delta_p1_vs_p3_usd_per_mwh": round(p1_rtax - p3_rtax, 2),
                "p1_stranded_new_gas_billion": p1["stranded_new_gas_capex_billion"],
                "p1_active_new_gas_mw": p1["active_new_gas_fleet_mw_2050"],
                "p1_undiscounted_cost_trillion": p1_cost,
                "p3_undiscounted_cost_trillion": p3_cost,
                "delta_p1_vs_p3_cost_pct": (
                    round((p1_cost - p3_cost) / p3_cost * 100, 1) if p3_cost else None
                ),
            }

    # Per-pathway sum of stranded new-gas capex at ep95 across feasible ISOs
    pathway_summary: dict[str, dict] = {}
    for path in PATHWAYS:
        path_rows = [r for r in feasible_rows if r["pathway"] == path and r["endpoint"] == 0.95]
        if path_rows:
            pathway_summary[path] = {
                "count": len(path_rows),
                "avg_rtax_usd_per_mwh": round(
                    sum(r["reliability_tax_usd_per_mwh"] for r in path_rows) / len(path_rows), 2),
                "total_stranded_new_gas_billion": round(
                    sum(r["stranded_new_gas_capex_billion"] for r in path_rows), 1),
                "total_active_new_gas_gw": round(
                    sum(r["active_new_gas_fleet_mw_2050"] for r in path_rows) / 1e3, 1),
            }

    total_runs = len(rows)
    feas_count = sum(1 for r in rows if r.get("feasible_physical"))
    infeas_count = total_runs - feas_count

    return {
        "total_runs": total_runs,
        "feasible_runs": feas_count,
        "infeasible_runs": infeas_count,
        "by_iso_at_ep90": iso_summary,
        "by_pathway_at_ep95": pathway_summary,
    }


def main() -> None:
    rows = []
    for iso in ISOS:
        for pathway in PATHWAYS:
            for ep in CLOSING_ENDPOINTS:
                rows.append(_build_row(iso, pathway, ep))

    agg = _compute_aggregations(rows)

    payload = {
        "columns": COLUMNS,
        "row_count": len(rows),
        "endpoints": CLOSING_ENDPOINTS,
        "rows": rows,
        "aggregations": agg,
        "meta": {
            "payload_id": "closing_summary_table",
            "schema_version": 2,
            "note": (
                f"Full cross-ISO × pathway × endpoint table. {len(rows)} rows "
                f"(7 ISO × 5 pathway × {len(CLOSING_ENDPOINTS)} endpoint). "
                "Pathways: 1 (VRE + storage + offshore headline), 1a (strict onshore "
                "Card R baseline), 2a (behavioral pivot), 2b (economic pivot), "
                "3 (proactive clean firm per §24.8 NOAK-2035). "
                "Endpoints are the 4 scheduled thresholds (90/95/99/99.9), "
                "each solved on its own SBTi-ladder horizon (2040/2045/2049/2050)."
            ),
            "reliability_tax_formula": (
                "SPEC §24.4 locked 5-component formula: "
                "(annualized_new_gas_capex + new_gas_FOM + existing_gas_FOM_carried "
                "+ priced_VRE_curtailment + annualized_VRE_storage_overbuild_capex) "
                "÷ cumulative_demand_MWh_2025_2050. Gross; no capacity-market netting (Card M')."
            ),
            "gas_sizing_basis": (
                "active_new_gas_fleet_mw is worst-hour sized per SPEC §24.5 "
                "(99.97th percentile of margin-inclusive residual gap). NOT ELCC."
            ),
            "stranding_basis": (
                "stranded_new_gas_* columns are ABSOLUTE per Card K' — "
                "Σ initial_cap_mw / stranded_capex_usd for new-gas vintages where "
                "stranded_flag=True (CF<15% for 2 consecutive years, SPEC Card F')."
            ),
            "dropped_columns_vs_v1": [
                "total_stranded_billion_usd (old Card K comparative framing)",
                "total_stranded_twh (old Card K comparative framing)",
                "net_annual_cost_usd (Card M' — no cap-market netting)",
                "capacity_rev_netted_usd (Card M' — no cap-market netting)",
                "ccs_ccgt synthetic stranding proxy (replaced by Card K' ledger)",
            ],
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    DASH_OUT.parent.mkdir(parents=True, exist_ok=True)
    DASH_OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH} ({len(rows)} rows)")
    print(f"Wrote {DASH_OUT}")

    print("\n--- closing_summary_table quick summary ---")
    print(f"Total rows: {len(rows)}")
    print(f"Feasible: {agg['feasible_runs']} | Infeasible: {agg['infeasible_runs']}")
    print("\nBy ISO at ep90 (P1 vs P3 reliability tax + undisc. cost delta):")
    for iso, d in agg["by_iso_at_ep90"].items():
        print(
            f"  {iso:6s}: P3=${d['p3_rtax_usd_per_mwh']:6.2f}/MWh, "
            f"P1=${d['p1_rtax_usd_per_mwh']:6.2f}/MWh, "
            f"Δ={d['delta_p1_vs_p3_usd_per_mwh']:+6.2f}/MWh, "
            f"Δcost={d['delta_p1_vs_p3_cost_pct']}%, "
            f"P1 stranded=${d['p1_stranded_new_gas_billion']:,.1f}B"
        )
    print("\nBy pathway at ep95 (sum across feasible ISOs):")
    for path, d in agg["by_pathway_at_ep95"].items():
        print(
            f"  P{path}: avg=${d['avg_rtax_usd_per_mwh']:6.2f}/MWh, "
            f"stranded=${d['total_stranded_new_gas_billion']:,.1f}B, "
            f"active gas={d['total_active_new_gas_gw']:,.1f}GW"
        )


if __name__ == "__main__":
    main()
