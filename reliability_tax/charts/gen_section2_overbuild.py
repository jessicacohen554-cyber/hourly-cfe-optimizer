"""
gen_section2_overbuild.py — Generate section2_overbuild.json

Data lineage:
  - analysis/reliability-tax/data/MANIFEST.json + per-run JSONs (base)
  - analysis/reliability-tax/data-demand-H/ (high demand variant)
  - analysis/reliability-tax/data-demand-L/ (low demand variant)
  - scripts/pipeline_config.py: PEAK_DEMAND_MW, EXISTING_GAS_CAPACITY_MW,
    RESOURCE_CAPACITY_FACTORS

Section 2 framing: "The Overbuild Problem"
As CFE thresholds rise under Pathway 1 (VRE + batteries only), the grid builds
multiples of peak demand in VRE capacity to serve the worst hours. This chart
quantifies the overbuild ratio across thresholds, ISOs, and pathways.

Metrics computed:
  - total_capacity_gw: sum of all installed resource capacity at 2050 endpoint
  - gw_per_gw_peak (overbuild ratio): total_capacity_gw / peak_demand_gw_2050
  - gas_remaining_gw: existing gas minus retired gas at endpoint
  - gas_pct_of_peak: gas_remaining_gw / peak_demand_gw_2050 × 100
  - new_build_clean_firm_gw: clean firm (ccs_ccgt / uprate) built under pathways 2a/2b/3
  - total_new_build_gw_by_resource: breakdown by resource type

Demand sensitivity: L/M/H variants from data-demand-L / base / data-demand-H

Payload structure:
  per_iso: dict[iso ->
    per_endpoint: dict[ep_label ->
      per_pathway: dict[pathway ->
        demand_variant: "M" (medium, base data dir)
        per_demand: dict["M"|"H"|"L" ->
          demand_2050_twh, peak_demand_gw_2050,
          achieved_cfe_pct,
          total_capacity_gw,
          gw_per_gw_peak,
          gas_remaining_gw,
          gas_pct_of_peak,
          new_build_clean_firm_gw,
          total_new_build_gw_by_resource
        ]
      ]
    ]
  ]
  meta: ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import (  # noqa: E402
    DL_BASE, DL_DEMAND_H, DL_DEMAND_L,
    ISOS, PATHWAYS, ENDPOINTS,
    _EP_LABEL,
)

OUT_PATH = Path(__file__).parent / "section2_overbuild.json"

# From pipeline_config.py
PEAK_DEMAND_MW_2025 = {
    "CAISO": 43860, "ERCOT": 83597, "PJM": 160560,
    "NYISO": 31857, "NEISO": 25898, "MISO": 127125, "SPP": 54368,
}
EXISTING_GAS_GW = {
    "CAISO": 37.0, "ERCOT": 55.0, "PJM": 75.0,
    "NYISO": 18.0, "NEISO": 14.0, "MISO": 68.0, "SPP": 32.0,
}
# Card S: new-build gas stranding parameters
CCS_CCGT_CF_STRAND = 0.85
CCS_CCGT_CAPITAL_PER_GW = 2.2e9   # $2,200/kW = $2.2B/GW (NREL ATB 2024 moderate)
CCS_CCGT_USEFUL_LIFE = 25
MODEL_ENDPOINT_YEAR = 2050

SOLAR_CF = {"CAISO":0.28,"ERCOT":0.24,"PJM":0.17,"NYISO":0.15,"NEISO":0.15,"MISO":0.19,"SPP":0.22}
WIND_CF = {"CAISO":0.25,"ERCOT":0.38,"PJM":0.30,"NYISO":0.28,"NEISO":0.30,"MISO":0.36,"SPP":0.42}
OFFSHORE_WIND_CF = 0.40
CLEAN_FIRM_CF = 0.90
CCS_CCGT_CF = 0.85
HYDRO_CF = 0.40
UPRATE_CF = 0.90
BATT4_DISPATCH_H_YR = 1000
BATT8_DISPATCH_H_YR = 1200
LDES_DISPATCH_H_YR = 800


def _twh_to_gw(resource: str, twh: float, iso: str) -> float:
    """Convert annual TWh capacity to installed GW."""
    if twh <= 0:
        return 0.0
    if resource == "solar":
        return twh / (8.76 * SOLAR_CF.get(iso, 0.20))
    elif resource in ("wind", "onshore_wind"):
        return twh / (8.76 * WIND_CF.get(iso, 0.35))
    elif resource == "offshore_wind":
        return twh / (8.76 * OFFSHORE_WIND_CF)
    elif resource == "clean_firm":
        return twh / (8.76 * CLEAN_FIRM_CF)
    elif resource == "ccs_ccgt":
        return twh / (8.76 * CCS_CCGT_CF)
    elif resource in ("hydro",):
        return twh / (8.76 * HYDRO_CF)
    elif resource == "uprate":
        return twh / (8.76 * UPRATE_CF)
    elif resource == "battery4":
        return twh * 1000.0 / BATT4_DISPATCH_H_YR
    elif resource == "battery8":
        return twh * 1000.0 / BATT8_DISPATCH_H_YR
    elif resource in ("ldes", "h2"):
        return twh * 1000.0 / LDES_DISPATCH_H_YR
    else:
        # Unknown resource — skip
        return 0.0


def _new_gas_stranding(run: dict) -> tuple[float, float]:
    """Return (stranded_gw, stranded_usd) for new CCS-CCGT built and not amortized by 2050."""
    bo = run.get("tables", {}).get("annual_buildout", [])
    total_gw = 0.0
    total_usd = 0.0
    for yr_row in bo:
        yr = yr_row.get("year", MODEL_ENDPOINT_YEAR)
        for v in yr_row.get("new_vintages", []):
            if v.get("resource") != "ccs_ccgt":
                continue
            twh = v.get("twh_per_year", 0.0)
            if twh <= 0:
                continue
            gw = twh / (8.76 * CCS_CCGT_CF_STRAND)
            frac = max(0.0, (yr + CCS_CCGT_USEFUL_LIFE - MODEL_ENDPOINT_YEAR) / CCS_CCGT_USEFUL_LIFE)
            total_gw += gw * frac
            total_usd += gw * CCS_CCGT_CAPITAL_PER_GW * frac
    return round(total_gw, 2), round(total_usd, 0)


def _buildout_totals(bo: list[dict]) -> dict[str, float]:
    """Sum cumulative TWh per resource across all buildout years."""
    totals: dict[str, float] = {}
    for yr_row in bo:
        for v in yr_row.get("new_vintages", []):
            r = v["resource"]
            totals[r] = totals.get(r, 0.0) + v.get("twh_per_year", 0.0)
    return totals


def _peak_demand_2050(demand_2050_twh: float, iso: str) -> float:
    """Estimate 2050 peak demand by scaling 2025 peak proportionally to avg demand growth."""
    peak_2025_gw = PEAK_DEMAND_MW_2025[iso] / 1000.0
    avg_2025_gw = peak_2025_gw * 0.55
    demand_2025_twh = avg_2025_gw * 8.76
    avg_2050_gw = demand_2050_twh / 8.76
    peak_2050_gw = peak_2025_gw * (avg_2050_gw / avg_2025_gw)
    return round(peak_2050_gw, 1)


def _compute_overbuild(iso: str, pathway: str, ep: float, dl: Any) -> dict:
    """Compute overbuild metrics for one run."""
    try:
        run = dl.get_run(iso, pathway, ep)
    except (KeyError, FileNotFoundError):
        return {"available": False}

    if not run["feasibility"].get("physical", False):
        return {"available": False, "infeasible": True}

    # Demand
    cost_rows = run.get("tables", {}).get("annual_cost", [])
    demand_2050 = cost_rows[-1]["demand_twh"] if cost_rows else 0.0
    peak_gw = _peak_demand_2050(demand_2050, iso)

    # Gas retirement
    rt = run.get("retirement_timeline", [])
    gas_retired_gw = rt[-1]["gas_retired_gw"] if rt else 0.0
    gas_remaining_gw = max(0.0, EXISTING_GAS_GW.get(iso, 0.0) - gas_retired_gw)

    # Buildout totals
    bo = run.get("tables", {}).get("annual_buildout", [])
    totals_twh = _buildout_totals(bo)

    # Convert to GW
    gw_by_resource: dict[str, float] = {}
    for r, twh in totals_twh.items():
        gw = _twh_to_gw(r, twh, iso)
        if gw > 0:
            gw_by_resource[r] = round(gw, 1)

    total_new_build_gw = sum(gw_by_resource.values())

    # Clean firm GW (ccs_ccgt + uprate)
    clean_firm_gw = (
        gw_by_resource.get("ccs_ccgt", 0.0)
        + gw_by_resource.get("clean_firm", 0.0)
        + gw_by_resource.get("uprate", 0.0)
    )

    # Total installed including gas remaining (this is what the grid relies on)
    total_with_gas_gw = total_new_build_gw + gas_remaining_gw

    overbuild_ratio = round(total_with_gas_gw / peak_gw, 2) if peak_gw > 0 else None
    gas_pct_of_peak = round(gas_remaining_gw / peak_gw * 100, 1) if peak_gw > 0 else None

    # Card S: new-build gas stranding
    ng_stranded_gw, ng_stranded_usd = _new_gas_stranding(run)
    undiscounted_t = run.get("undiscounted_cost_usd", 0.0) / 1e12
    total_actual_cost_t = undiscounted_t + ng_stranded_usd / 1e12

    return {
        "available": True,
        "achieved_cfe_pct": round(run.get("achieved_cfe_pct", 0.0), 1),
        "demand_2050_twh": round(demand_2050, 1),
        "peak_demand_gw_2050": peak_gw,
        "total_new_build_gw": round(total_new_build_gw, 0),
        "total_with_gas_gw": round(total_with_gas_gw, 0),
        "gw_per_gw_peak": overbuild_ratio,
        "gas_remaining_gw": round(gas_remaining_gw, 1),
        "gas_pct_of_peak": gas_pct_of_peak,
        "new_build_clean_firm_gw": round(clean_firm_gw, 1),
        "gw_by_resource": gw_by_resource,
        # Primary cost metric (Card S)
        "undiscounted_cost_trillion_usd": round(undiscounted_t, 3),
        "new_gas_stranded_gw": ng_stranded_gw,
        "new_gas_stranded_billion_usd": round(ng_stranded_usd / 1e9, 2),
        "total_actual_cost_trillion_usd": round(total_actual_cost_t, 3),
        # Secondary cost metric
        "npv_7pct_trillion_usd": round(run.get("npv_at_7pct", 0.0) / 1e12, 3),
    }


def main() -> None:
    # Map demand variant labels to data loaders
    DL_MAP = {"M": DL_BASE, "H": DL_DEMAND_H, "L": DL_DEMAND_L}

    per_iso: dict[str, dict] = {}

    for iso in ISOS:
        per_ep: dict[str, dict] = {}
        for ep in ENDPOINTS:
            ep_label = _EP_LABEL[ep]
            per_pathway: dict[str, dict] = {}
            for pathway in PATHWAYS:
                per_demand: dict[str, dict] = {}
                for demand_label, dl in DL_MAP.items():
                    result = _compute_overbuild(iso, pathway, ep, dl)
                    per_demand[demand_label] = result
                per_pathway[pathway] = {
                    "per_demand_variant": per_demand,
                    # Convenience: M (base) values at top level
                    "base": per_demand.get("M", {}),
                }
            per_ep[ep_label] = per_pathway
        per_iso[iso] = per_ep

    payload = {
        "per_iso": per_iso,
        "meta": {
            "payload_id": "section2_overbuild",
            "note": (
                "GW/load multipliers and overbuild metrics for all 7 ISOs × 4 pathways × 5 endpoints. "
                "Installed GW derived from annual buildout TWh via capacity factors. "
                "Gas remaining = existing_gas_capacity_gw - gas_retired_gw at endpoint year. "
                "L/M/H demand variants from data-demand-L, base, data-demand-H directories."
            ),
            "overbuild_definition": (
                "gw_per_gw_peak = (total_new_build_gw + gas_remaining_gw) / peak_demand_gw_2050. "
                "Includes all new-build resources plus existing gas not yet retired at endpoint."
            ),
            "gas_definition": (
                "gas_remaining_gw = max(0, EXISTING_GAS_CAPACITY_GW - gas_retired_cumulative_gw). "
                "This is legacy fleet capacity, not new-build gas."
            ),
            "clean_firm_definition": (
                "new_build_clean_firm_gw includes ccs_ccgt + uprate resources. "
                "Pathway 1: always 0 (clean firm locked). "
                "Pathways 2a/2b/3: unlocks post-pivot or from year 1."
            ),
            "capacity_derivation": {
                "solar": "twh / (8.76 × CF_solar)",
                "wind_onshore": "twh / (8.76 × CF_wind)",
                "battery4": "twh × 1000 / 1000 h/yr",
                "battery8": "twh × 1000 / 1200 h/yr",
                "ldes": "twh × 1000 / 800 h/yr",
            },
            "version": "1.0",
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")

    # Quick summary: ERCOT and PJM at key endpoints, P1 and P3
    print("\n--- section2_overbuild quick summary (ERCOT + PJM, P1a/P1b vs P3, M demand) ---")
    for iso in ["ERCOT", "PJM"]:
        for ep in ENDPOINTS:
            ep_label = _EP_LABEL[ep]
            for path in ["1a", "1b", "3"]:
                d = per_iso[iso][ep_label][path]["base"]
                if not d.get("available"):
                    continue
                print(
                    f"  {iso} P{path} ep{d['achieved_cfe_pct']:.0f}%: "
                    f"total={d.get('total_with_gas_gw',0):.0f}GW, "
                    f"ratio={d.get('gw_per_gw_peak','N/A')}x, "
                    f"gas={d.get('gas_remaining_gw',0):.1f}GW ({d.get('gas_pct_of_peak',0):.0f}%), "
                    f"clean_firm={d.get('new_build_clean_firm_gw',0):.0f}GW, "
                    f"cost=${d.get('undiscounted_cost_trillion_usd',0):.2f}T"
                )

    # Demand sensitivity comparison: ERCOT P1 ep95
    print("\n--- ERCOT P1 ep95 demand sensitivity (L/M/H) ---")
    ep_label = _EP_LABEL[0.95]
    for d_label in ["L", "M", "H"]:
        d = per_iso["ERCOT"][ep_label]["1a"]["per_demand_variant"][d_label]
        if d.get("available"):
            print(
                f"  {d_label}: demand={d['demand_2050_twh']:.0f}TWh, "
                f"ratio={d.get('gw_per_gw_peak','N/A')}x, "
                f"cost=${d.get('undiscounted_cost_trillion_usd',0):.2f}T"
            )


if __name__ == "__main__":
    main()
