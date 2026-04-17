"""
gen_section2_overbuild.py — Generate section2_overbuild.json

v2 (Apr 2026). Rewritten against the new-gas / worst-hour schema (SPEC §24.4 +
§24.5). Total build-out by resource at 2050 endpoint for each (ISO, pathway,
endpoint), including new-build gas as an explicit resource segment.

Methodology:
  - New-gas GW: cumulative from tables.annual_buildout[].gas_sizing.
    `active_new_gas_fleet_mw` at the terminal row is the total new-build CCGT
    that came online 2025-2050 and remains on the books (stranded vintages stay
    counted as "built" even after the Card F' CF test flags them as stranded).
  - VRE + storage GW: converted from cumulative TWh/yr per vintage using
    resource-specific capacity factors (pipeline_config.RESOURCE_CAPACITY_FACTORS).
  - Existing gas GW: from pipeline_config.EXISTING_GAS_CAPACITY_MW (unchanged
    across pathways — existing fleet is inherited, not a pathway choice).
  - 2050 peak demand: from the terminal row of tables.annual_buildout
    (gas_sizing.peak_demand_mw) — this is the optimizer's scaled 2050 peak,
    not a rescaled estimate.

Worst-hour gas sizing (SPEC §24.5) replaces the legacy ELCC formula; the
active_new_gas_fleet_mw numbers here reflect the 99.97th-percentile residual-gap
sizing. No ELCC conversions anywhere in this script.

Payload structure:
  per_iso[ISO][endpoint_label][pathway] = {
    achieved_cfe_pct,
    demand_2050_twh,
    peak_demand_gw_2050,
    gw_by_resource: {
      existing_gas, new_gas, solar, wind, offshore_wind, hydro,
      clean_firm, ccs_ccgt, battery4, battery8, ldes, h2, ...
    },
    total_new_build_gw,       # includes new_gas
    total_with_existing_gw,   # includes new_gas + existing_gas
    new_gas_built_gw,
    new_gas_stranded_gw,
    new_gas_stranded_capex_billion_usd,
    undiscounted_cost_trillion_usd,
    npv_7pct_trillion_usd,
  }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import (  # noqa: E402
    DL_BASE, ISOS, PATHWAYS, ENDPOINTS, _EP_LABEL,
)

OUT_PATH = Path(__file__).parent / "section2_overbuild.json"
DASH_OUT = REPO_ROOT / "dashboard" / "js" / "reliability-tax" / "section2_overbuild.json"

# ---------------------------------------------------------------------------
# Pipeline config mirrors (avoid runtime import of the full pipeline)
# ---------------------------------------------------------------------------

EXISTING_GAS_MW = {
    "CAISO": 37000, "ERCOT": 55000, "PJM": 75000,
    "NYISO": 18000, "NEISO": 14000, "MISO": 68000, "SPP": 32000,
}

RESOURCE_CF = {
    "solar":         {"CAISO": 0.28, "ERCOT": 0.24, "PJM": 0.17, "NYISO": 0.15, "NEISO": 0.15, "MISO": 0.19, "SPP": 0.22},
    "wind":          {"CAISO": 0.25, "ERCOT": 0.38, "PJM": 0.30, "NYISO": 0.28, "NEISO": 0.30, "MISO": 0.36, "SPP": 0.42},
    "offshore_wind": {"CAISO": 0.42, "ERCOT": 0.00, "PJM": 0.45, "NYISO": 0.45, "NEISO": 0.48, "MISO": 0.00, "SPP": 0.00},
    "clean_firm":    {"CAISO": 0.90, "ERCOT": 0.93, "PJM": 0.93, "NYISO": 0.90, "NEISO": 0.90, "MISO": 0.92, "SPP": 0.92},
    "ccs_ccgt":      {"CAISO": 0.85, "ERCOT": 0.85, "PJM": 0.85, "NYISO": 0.85, "NEISO": 0.85, "MISO": 0.85, "SPP": 0.85},
    "geothermal":    {"CAISO": 0.85, "ERCOT": 0.85, "PJM": 0.85, "NYISO": 0.85, "NEISO": 0.85, "MISO": 0.85, "SPP": 0.85},
    "hydro":         {"CAISO": 0.40, "ERCOT": 0.40, "PJM": 0.40, "NYISO": 0.40, "NEISO": 0.40, "MISO": 0.40, "SPP": 0.40},
    "uprate":        {"CAISO": 0.90, "ERCOT": 0.90, "PJM": 0.90, "NYISO": 0.90, "NEISO": 0.90, "MISO": 0.90, "SPP": 0.90},
    # Hybrid parents use their parent VRE CF; the batt fraction is rolled in at the
    # TWh level so we treat the whole twh_per_year at the parent-CF.
    "solar_batt4":   {"CAISO": 0.32, "ERCOT": 0.28, "PJM": 0.20, "NYISO": 0.18, "NEISO": 0.18, "MISO": 0.22, "SPP": 0.26},
    "solar_batt8":   {"CAISO": 0.34, "ERCOT": 0.30, "PJM": 0.21, "NYISO": 0.19, "NEISO": 0.19, "MISO": 0.23, "SPP": 0.27},
    "wind_batt4":    {"CAISO": 0.28, "ERCOT": 0.40, "PJM": 0.32, "NYISO": 0.30, "NEISO": 0.32, "MISO": 0.38, "SPP": 0.44},
    "wind_batt8":    {"CAISO": 0.29, "ERCOT": 0.41, "PJM": 0.33, "NYISO": 0.31, "NEISO": 0.33, "MISO": 0.39, "SPP": 0.45},
}

# Storage: TWh of DISPATCH → GW of installed power capacity.
# Approximation: battery power ~ dispatch_twh × 1000 / (cycles × duration_h).
# Use representative cycles (battery = 300/yr, ldes = 40/yr, h2 = 15/yr).
STORAGE_POWER_HOURS_PER_YR = {
    "battery":   1200,  # 300 cycles × 4hr
    "battery4":  1200,
    "battery8":  2400,  # 300 cycles × 8hr
    "ldes":      4000,  # 40 cycles × 100hr
    "h2":        15000, # 15 cycles × 1000hr
}


def _twh_to_gw(resource: str, twh: float, iso: str) -> float:
    """Convert annual TWh to installed GW using resource-specific CF or storage hrs."""
    if twh <= 0:
        return 0.0
    if resource in STORAGE_POWER_HOURS_PER_YR:
        return twh * 1000.0 / STORAGE_POWER_HOURS_PER_YR[resource]
    cf_map = RESOURCE_CF.get(resource)
    if cf_map is None:
        return 0.0
    cf = cf_map.get(iso, 0.0)
    if cf <= 0:
        return 0.0
    return twh / (8.76 * cf)


def _compute_one(iso: str, pathway: str, ep: float) -> dict:
    try:
        run = DL_BASE.get_run(iso, pathway, ep)
    except (KeyError, FileNotFoundError):
        return {"available": False}
    if not run["feasibility"].get("physical", False):
        return {"available": False, "infeasible": True,
                "notes": run["feasibility"].get("notes", [])}

    bo = run.get("tables", {}).get("annual_buildout", [])
    cost_rows = run.get("tables", {}).get("annual_cost", [])
    if not bo or not cost_rows:
        return {"available": False, "no_tables": True}

    terminal = bo[-1]
    demand_2050 = cost_rows[-1]["demand_twh"]
    peak_2050_mw = terminal["gas_sizing"].get("peak_demand_mw", 0.0)
    peak_2050_gw = peak_2050_mw / 1000.0
    new_gas_built_mw = terminal["gas_sizing"].get("active_new_gas_fleet_mw", 0.0)

    # Sum cumulative TWh across vintages for each VRE / storage / clean-firm resource.
    vintage_twh: dict[str, float] = {}
    for row in bo:
        for v in row.get("new_vintages", []):
            r = v.get("resource")
            if r == "new_gas_ccgt":
                continue  # gas handled separately via gas_sizing
            twh = float(v.get("twh_per_year", 0.0) or 0.0)
            if twh <= 0:
                continue
            vintage_twh[r] = vintage_twh.get(r, 0.0) + twh

    gw_by_resource: dict[str, float] = {}
    for r, twh in vintage_twh.items():
        gw = _twh_to_gw(r, twh, iso)
        if gw > 0.01:
            gw_by_resource[r] = round(gw, 2)

    # Insert the gas buckets explicitly.
    gw_by_resource["existing_gas"] = round(EXISTING_GAS_MW.get(iso, 0) / 1000.0, 2)
    gw_by_resource["new_gas"] = round(new_gas_built_mw / 1000.0, 2)

    total_new_build_gw = sum(
        gw for r, gw in gw_by_resource.items() if r != "existing_gas"
    )
    total_with_existing_gw = total_new_build_gw + gw_by_resource["existing_gas"]

    # Stranded new-gas from per-vintage ledger
    ngf = run.get("tables", {}).get("new_gas_fleet", []) or []
    stranded_mw = sum(
        float(v.get("initial_cap_mw", 0.0))
        for v in ngf if v.get("stranded_flag")
    )
    stranded_capex_usd = sum(
        float(v.get("stranded_capex_usd", 0.0)) for v in ngf
    )

    return {
        "available": True,
        "achieved_cfe_pct": round(run.get("achieved_cfe_pct", 0.0), 2),
        "demand_2050_twh": round(demand_2050, 1),
        "peak_demand_gw_2050": round(peak_2050_gw, 1),
        "gw_by_resource": gw_by_resource,
        "total_new_build_gw": round(total_new_build_gw, 1),
        "total_with_existing_gw": round(total_with_existing_gw, 1),
        "new_gas_built_gw": round(new_gas_built_mw / 1000.0, 2),
        "new_gas_stranded_gw": round(stranded_mw / 1000.0, 2),
        "new_gas_stranded_capex_billion_usd": round(stranded_capex_usd / 1e9, 2),
        "undiscounted_cost_trillion_usd": round(
            run.get("undiscounted_cost_usd", 0.0) / 1e12, 3
        ),
        "npv_7pct_trillion_usd": round(
            run.get("npv_at_7pct", 0.0) / 1e12, 3
        ),
    }


def main() -> None:
    per_iso: dict[str, dict] = {}
    for iso in ISOS:
        per_ep: dict[str, dict] = {}
        for ep in ENDPOINTS:
            ep_label = _EP_LABEL[ep]
            per_pathway: dict[str, dict] = {}
            for pathway in PATHWAYS:
                per_pathway[pathway] = _compute_one(iso, pathway, ep)
            per_ep[ep_label] = per_pathway
        per_iso[iso] = per_ep

    payload = {
        "per_iso": per_iso,
        "meta": {
            "payload_id": "section2_overbuild",
            "schema_version": 2,
            "note": (
                "Total build-out GW by resource at 2050, 7 ISO × 5 pathway × 10 endpoint. "
                "New-build gas (SPEC Card U) is an explicit resource segment — sized via "
                "worst-hour residual-gap dispatch (SPEC §24.5), not ELCC."
            ),
            "gas_sizing_method": (
                "active_new_gas_fleet_mw read directly from the terminal row of "
                "tables.annual_buildout[].gas_sizing. The optimizer sized this to the "
                "99.97th percentile of the margin-inclusive residual gap "
                "(SPEC §24.5 Option B-i). ELCC is NOT used."
            ),
            "vre_to_gw_method": (
                "GW = cum_TWh / (8.76 × CF_resource) for generation. "
                "Storage GW = cum_dispatch_TWh × 1000 / representative_dispatch_hours_per_year."
            ),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    DASH_OUT.parent.mkdir(parents=True, exist_ok=True)
    DASH_OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {DASH_OUT}")

    # Summary
    for iso in ["ERCOT", "PJM"]:
        print(f"\n  {iso} @ ep95:")
        for pw in PATHWAYS:
            d = per_iso[iso]["ep95"][pw]
            if not d.get("available"):
                continue
            print(
                f"    P{pw}: new_gas={d['new_gas_built_gw']:.0f} GW, "
                f"stranded={d['new_gas_stranded_gw']:.0f} GW, "
                f"existing_gas={d['gw_by_resource']['existing_gas']:.0f} GW, "
                f"total_new_build={d['total_new_build_gw']:.0f} GW"
            )


if __name__ == "__main__":
    main()
