"""
gen_section3_reliability_tax.py — Generate section3_reliability_tax.json

Data lineage:
  - analysis/reliability-tax/data/MANIFEST.json + per-run JSONs
  - scripts/pipeline_config.py: EXISTING_GAS_CAPACITY_MW, EXISTING_GAS_FOM_KW_YR,
    NEW_CCGT_COST_KW_YR

Section 3 framing: "The Reliability Tax"
As CFE thresholds rise, the existing gas fleet dispatches less but fixed costs
remain. The reliability tax = stacked FOM costs net of capacity payments,
expressed per MWh of total system demand. These are the "insurance costs" of
keeping gas capacity in service even when it rarely runs.

Metrics at each CFE endpoint (Pathway 1):
  - gas_cf: gas fleet utilization = gas_gen_twh / (existing_gas_gw × 8.76)
    where gas_gen_twh = demand_2050 × (1 - achieved_cfe_pct/100)
    (this is fleet-level CF over the full existing fleet, not per-unit)
  - gas_fom_annual_usd: existing_gas_remaining_gw × fom_$/kW-yr × 1e3 × 1e3 ($/yr)
  - capacity_rev_usd: from annual_cost[2050].capacity_rev_netted_usd
  - net_fom_usd: gas_fom - capacity_rev
  - net_fom_per_mwh_demand: net_fom / demand_2050 / 1000 ($/MWh of demand)
  - reliability_tax_usd_per_mwh: net_fom_per_mwh_demand (headline metric)

Also: compare Pathway 1 vs Pathway 3 FOM burden at same endpoint — the clean firm
pathway maintains clean-firm FOM instead of gas FOM; the difference is the "tax savings"
from proactive planning.

Payload structure:
  per_iso: dict[iso ->
    by_endpoint: dict[ep_label ->
      per_pathway: dict[pathway ->
        achieved_cfe_pct, demand_2050_twh,
        gas_gen_twh, gas_fleet_cf,
        gas_remaining_gw,
        gas_fom_annual_million_usd,
        capacity_rev_million_usd,
        net_fom_million_usd,
        net_fom_per_mwh_demand,
        reliability_tax_usd_per_mwh,
      ]
    ]
    gas_cf_series: list[{endpoint, achieved_cfe_pct, gas_fleet_cf}]  — P1 only, for sparkline
    fom_series: list[{endpoint, net_fom_per_mwh_demand}]             — P1 only
    tax_gap_series: list[{endpoint, p1_tax, p3_tax, gap_usd_per_mwh}]
  ]
  global_axis_ranges: dict
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
    ISOS, PATHWAYS, ENDPOINTS,
    _EP_LABEL,
)

OUT_PATH = Path(__file__).parent / "section3_reliability_tax.json"

# From pipeline_config.py
EXISTING_GAS_GW = {
    "CAISO": 37.0, "ERCOT": 55.0, "PJM": 75.0,
    "NYISO": 18.0, "NEISO": 14.0, "MISO": 68.0, "SPP": 32.0,
}
EXISTING_GAS_FOM_KW_YR = {
    "CAISO": 16, "ERCOT": 13, "PJM": 14,
    "NYISO": 17, "NEISO": 15, "MISO": 14, "SPP": 13,
}
# Average gas dispatch CF at baseline (2025) — used for gas_gen implied by fleet
GAS_BASELINE_CF = {
    "CAISO": 0.45, "ERCOT": 0.45, "PJM": 0.45,
    "NYISO": 0.45, "NEISO": 0.45, "MISO": 0.45, "SPP": 0.45,
}


def _compute_tax(iso: str, pathway: str, ep: float) -> dict:
    """Compute reliability tax metrics for one run."""
    try:
        run = DL_BASE.get_run(iso, pathway, ep)
    except (KeyError, FileNotFoundError):
        return {"available": False}

    if not run["feasibility"].get("physical", False):
        return {"available": False, "infeasible": True}

    # Demand and achieved CFE at 2050
    cost_rows = run.get("tables", {}).get("annual_cost", [])
    if not cost_rows:
        return {"available": False, "no_cost_rows": True}
    row_2050 = cost_rows[-1]
    demand_2050 = row_2050["demand_twh"]
    achieved_cfe = row_2050["achieved_cfe_pct"]
    cap_rev_2050 = row_2050["capacity_rev_netted_usd"]

    # Gas generation at 2050: demand × clean gap %
    clean_gap_pct = max(0.0, 1.0 - achieved_cfe / 100.0)
    gas_gen_twh = demand_2050 * clean_gap_pct

    # Gas fleet CF: gas_gen / (full existing fleet × 8.76)
    # Using FULL existing fleet as denominator (not just remaining after retirement).
    # This represents the utilization of the total insurance fleet capacity.
    full_fleet_gw = EXISTING_GAS_GW.get(iso, 0.0)
    gas_fleet_cf = (gas_gen_twh / (full_fleet_gw * 8.76)) if full_fleet_gw > 0 else 0.0

    # Gas remaining (from retirement_timeline)
    rt = run.get("retirement_timeline", [])
    gas_retired_gw = rt[-1]["gas_retired_gw"] if rt else 0.0
    gas_remaining_gw = max(0.0, full_fleet_gw - gas_retired_gw)

    # FOM cost: based on gas_remaining_gw × FOM rate
    # If gas_remaining ≈ 0 (all retired), FOM ≈ 0
    fom_kw_yr = EXISTING_GAS_FOM_KW_YR.get(iso, 14)
    gas_fom_annual_usd = gas_remaining_gw * 1e3 * fom_kw_yr * 1e3  # GW→kW→$

    # Net FOM = FOM - capacity market payments
    net_fom_usd = gas_fom_annual_usd - cap_rev_2050
    net_fom_million = round(net_fom_usd / 1e6, 1)

    # Reliability tax: net FOM per MWh of total demand
    # net_fom_usd / (demand_2050 × 1e6 MWh/TWh) → $/MWh
    reliability_tax_usd_per_mwh = (net_fom_usd / (demand_2050 * 1e6)) if demand_2050 > 0 else 0.0

    # For context: gas FOM as % of total annual system cost
    total_annual_cost = row_2050["net_annual_cost_usd"]
    fom_pct_of_system = (gas_fom_annual_usd / total_annual_cost * 100) if total_annual_cost > 0 else 0.0

    return {
        "available": True,
        "achieved_cfe_pct": round(achieved_cfe, 2),
        "demand_2050_twh": round(demand_2050, 1),
        "clean_gap_pct": round(clean_gap_pct * 100, 2),
        "gas_gen_twh": round(gas_gen_twh, 1),
        "gas_fleet_cf": round(gas_fleet_cf, 4),
        "gas_fleet_cf_pct": round(gas_fleet_cf * 100, 2),
        "gas_remaining_gw": round(gas_remaining_gw, 1),
        "gas_fom_annual_million_usd": round(gas_fom_annual_usd / 1e6, 1),
        "capacity_rev_million_usd": round(cap_rev_2050 / 1e6, 1),
        "net_fom_million_usd": net_fom_million,
        "net_fom_per_mwh_demand": round(reliability_tax_usd_per_mwh, 4),
        "reliability_tax_usd_per_mwh": round(reliability_tax_usd_per_mwh, 4),
        "fom_pct_of_total_system_cost": round(fom_pct_of_system, 2),
        "total_annual_system_cost_2050_million": round(total_annual_cost / 1e6, 1),
    }


def main() -> None:
    per_iso: dict[str, dict] = {}
    all_gas_cfs: list[float] = []
    all_tax_values: list[float] = []

    for iso in ISOS:
        by_endpoint: dict[str, dict] = {}
        for ep in ENDPOINTS:
            ep_label = _EP_LABEL[ep]
            per_pathway: dict[str, dict] = {}
            for pathway in PATHWAYS:
                per_pathway[pathway] = _compute_tax(iso, pathway, ep)
            by_endpoint[ep_label] = per_pathway

        # Build convenience series for Pathway 1 only
        gas_cf_series = []
        fom_series = []
        tax_gap_series = []

        for ep in ENDPOINTS:
            ep_label = _EP_LABEL[ep]
            p1 = by_endpoint[ep_label].get("1", {})
            p3 = by_endpoint[ep_label].get("3", {})

            if p1.get("available"):
                gas_cf_series.append({
                    "endpoint": ep,
                    "endpoint_label": ep_label,
                    "achieved_cfe_pct": p1["achieved_cfe_pct"],
                    "gas_fleet_cf_pct": p1["gas_fleet_cf_pct"],
                })
                fom_series.append({
                    "endpoint": ep,
                    "endpoint_label": ep_label,
                    "net_fom_million_usd": p1["net_fom_million_usd"],
                    "reliability_tax_usd_per_mwh": p1["reliability_tax_usd_per_mwh"],
                })
                all_gas_cfs.append(p1["gas_fleet_cf_pct"])
                all_tax_values.append(p1["reliability_tax_usd_per_mwh"])

            # Tax gap: P1 net FOM vs P3 net FOM (both normalized per MWh)
            p1_tax = p1.get("reliability_tax_usd_per_mwh", 0.0) if p1.get("available") else None
            p3_tax = p3.get("reliability_tax_usd_per_mwh", 0.0) if p3.get("available") else None
            gap = (p1_tax - p3_tax) if (p1_tax is not None and p3_tax is not None) else None
            tax_gap_series.append({
                "endpoint": ep,
                "endpoint_label": ep_label,
                "p1_reliability_tax_usd_per_mwh": p1_tax,
                "p3_reliability_tax_usd_per_mwh": p3_tax,
                "gap_usd_per_mwh": round(gap, 6) if gap is not None else None,
            })

        per_iso[iso] = {
            "existing_gas_gw": EXISTING_GAS_GW.get(iso, 0.0),
            "existing_gas_fom_kw_yr": EXISTING_GAS_FOM_KW_YR.get(iso, 0),
            "by_endpoint": by_endpoint,
            "gas_cf_series_p1": gas_cf_series,
            "fom_series_p1": fom_series,
            "tax_gap_series": tax_gap_series,
        }

    # Global axis ranges
    global_axis_ranges = {
        "gas_cf_pct": {
            "min": 0.0,
            "max": round(max(all_gas_cfs, default=50.0) * 1.05, 1),
        },
        "reliability_tax_usd_per_mwh": {
            "min": 0.0,
            "max": round(max(all_tax_values, default=1.0) * 1.1, 4),
        },
    }

    payload = {
        "per_iso": per_iso,
        "global_axis_ranges": global_axis_ranges,
        "meta": {
            "payload_id": "section3_reliability_tax",
            "note": (
                "Gas fleet capacity factor and stacked FOM cost (net of capacity payments) "
                "by CFE threshold, all 7 ISOs, 4 pathways, 5 endpoints. "
                "Gas fleet CF = gas_gen_twh / (full_existing_fleet_gw × 8.76). "
                "Uses full existing fleet as denominator to show fleet-level underutilization. "
                "FOM computed on gas_remaining_gw (after endogenous retirement). "
                "Reliability tax = net FOM per MWh of total demand."
            ),
            "gas_cf_definition": (
                "gas_fleet_cf = (demand_2050 × (1 − achieved_cfe%)) / (EXISTING_GAS_GW × 8.76). "
                "Numerator = annual gas generation implied by CFE gap. "
                "Denominator = maximum possible generation from full existing fleet at CF=100%."
            ),
            "fom_definition": (
                "gas_fom_annual = gas_remaining_gw × EXISTING_GAS_FOM_KW_YR × 1000. "
                "net_fom = gas_fom_annual − capacity_rev_netted_usd (from annual_cost[2050]). "
                "reliability_tax = net_fom / (demand_2050_twh × 1e6) in $/MWh."
            ),
            "version": "1.0",
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")

    print("\n--- section3_reliability_tax quick summary ---")
    for iso in ISOS:
        print(f"\n  {iso}:")
        for ep in ENDPOINTS:
            ep_label = _EP_LABEL[ep]
            p1 = per_iso[iso]["by_endpoint"][ep_label].get("1", {})
            if not p1.get("available"):
                continue
            print(
                f"    ep{p1['achieved_cfe_pct']:.0f}%: "
                f"gas_CF={p1['gas_fleet_cf_pct']:.1f}%, "
                f"FOM_net=${p1['net_fom_million_usd']:.0f}M/yr, "
                f"tax=${p1['reliability_tax_usd_per_mwh']:.4f}/MWh"
            )


if __name__ == "__main__":
    main()
