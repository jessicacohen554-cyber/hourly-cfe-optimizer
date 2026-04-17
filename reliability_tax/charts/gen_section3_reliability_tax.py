"""
gen_section3_reliability_tax.py — Generate section3_reliability_tax.json

v2 (Apr 2026). Rewritten against SPEC §24.4 Card M' + the locked 5-component tax
formula. Drops all capacity-market-netting logic. Each component is an
independent field so the dashboard can render a stacked bar.

Locked formula (SPEC §24.4):
  reliability_tax_$_per_MWh[pathway, ISO]
    = [ Σ annualized_new_gas_capex
      + Σ new_gas_FOM
      + Σ existing_gas_FOM_carried_forward
      + Σ priced_VRE_curtailment
      + Σ annualized_VRE_storage_overbuild_capex ]
      ÷ Σ demand_MWh

Provenance of each component in this script:
  1. new_gas_capex_annualized_usd   — from solver (reliability_tax.components_usd)
  2. new_gas_fom_usd                — solver (Lazard bundles FOM into capex, so
                                      this field is 0 for CCGT; retained as an
                                      explicit bar for future pathway variants)
  3. existing_gas_fom_carried_usd   — solver (EXISTING_GAS_FOM_KW_YR × fleet_MW × 25yr)
  4. priced_vre_curtailment_usd     — computed here: mix_excess × weighted_VRE_LCOE
                                      (optimizer doesn't dispatch, so this is a
                                      mix-excess proxy — see curtailment_method below)
  5. vre_storage_overbuild_capex_usd — computed here: storage capacity whose
                                       implied annual cycles are below the
                                       SPEC 15% CF equivalent (underutilized storage)

All components are reported in $/MWh (cumulative $ ÷ cum demand MWh 2025-2050).

Worst-hour sizing note: the optimizer uses SPEC §24.5 99.97th-percentile residual
sizing for gas. This script pulls the resulting components verbatim. ELCC does
not appear anywhere.
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

OUT_PATH = Path(__file__).parent / "section3_reliability_tax.json"
DASH_OUT = REPO_ROOT / "dashboard" / "js" / "reliability-tax" / "section3_reliability_tax.json"

# Weighted-average VRE LCOE at Medium toggle, $/MWh.
# Derived from pipeline_config.LCOE_TABLES['solar'/'wind']['Medium'] + transmission Medium.
# Used to price excess VRE generation as curtailment. Solar/wind blended 50/50.
VRE_LCOE_MEDIUM = {
    "CAISO": (60+3 + 73+8) / 2,    # ≈72
    "ERCOT": (54+3 + 40+6) / 2,    # ≈52
    "PJM":   (65+5 + 62+10) / 2,   # ≈71
    "NYISO": (92+7 + 81+14) / 2,   # ≈97
    "NEISO": (82+6 + 73+12) / 2,   # ≈87
    "MISO":  (62+4 + 43+9) / 2,    # ≈59
    "SPP":   (57+3 + 37+7) / 2,    # ≈52
}

# Storage annualized capacity cost per GW-yr, Medium toggle (bundled $/kW-yr
# derived from pipeline_config battery4 Medium). Applied to overbuilt storage
# power capacity as an annualized capex bar.
STORAGE_CAPEX_PER_GW_YR = {
    "CAISO": 41.61e6, "ERCOT": 37.41e6, "PJM": 39.86e6, "NYISO": 43.98e6,
    "NEISO": 42.66e6, "MISO": 39.07e6, "SPP": 37.93e6,
}

CF_OVERBUILD_THRESHOLD = 0.15    # SPEC Card F' / reliability tax definition
N_YEARS = 26                     # 2025..2050 inclusive


def _mix_excess_pct(run: dict) -> float:
    """Sum of endpoint_mix_pct; excess over 100% is the curtailment/overbuild proxy."""
    mix = run.get("endpoint_mix_pct", {}) or {}
    total = sum(float(v or 0.0) for v in mix.values())
    return max(0.0, total - 100.0)


def _storage_effective_cf(run: dict) -> dict[str, float]:
    """Implied annual capacity-factor equivalent for storage resources.

    Uses endpoint_storage_pct.*_dispatch_pct (fraction of demand dispatched
    through the resource) divided by its share of the capacity mix via the
    endpoint_mix_pct for the same resource bucket. This is a rough proxy —
    the real test lives in the dispatch cache.
    """
    storage = run.get("endpoint_storage_pct", {}) or {}
    mix = run.get("endpoint_mix_pct", {}) or {}
    result = {}
    for key, mix_key in [
        ("battery", "solar_batt4"),   # battery 4hr rolled into hybrid buckets
        ("battery8", "solar_batt8"),
        ("ldes", "clean_firm"),       # ldes doesn't have its own mix % bucket
        ("h2", "clean_firm"),
    ]:
        disp_pct = float(storage.get(f"{key}_dispatch_pct", 0.0) or 0.0)
        capacity_pct = float(mix.get(mix_key, 0.0) or 0.0)
        if capacity_pct > 0:
            # Fraction of "available cycles" actually dispatched
            result[key] = min(1.0, disp_pct / capacity_pct)
        else:
            result[key] = 0.0
    return result


def _compute_overbuild_capex_usd(iso: str, run: dict, demand_mwh_cum: float) -> float:
    """Component 5: annualized VRE+storage overbuild capex.

    Proxy: if the excess-mix fraction is > 0, treat that fraction of storage
    capex as "overbuild" (capacity whose effective utilization falls below the
    15% CF threshold per SPEC §24.4). Applied to the storage resource buckets
    only — VRE overbuild shows up in component 4 as priced curtailment.
    """
    excess_pct = _mix_excess_pct(run) / 100.0
    if excess_pct <= 0:
        return 0.0
    # Total storage capacity (GW) at endpoint, approximated from the annual
    # buildout of battery4/battery8/ldes/h2.
    bo = run.get("tables", {}).get("annual_buildout", []) or []
    storage_twh = 0.0
    for row in bo:
        for v in row.get("new_vintages", []):
            if v.get("resource") in ("battery4", "battery8", "ldes", "h2"):
                storage_twh += float(v.get("twh_per_year", 0.0) or 0.0)
    # Rough TWh → GW: assume 1200 dispatch-hours/yr average (blended 4/8hr battery).
    storage_gw = storage_twh * 1000.0 / 1200.0
    capex_per_gw_yr = STORAGE_CAPEX_PER_GW_YR.get(iso, 40e6)
    # Overbuilt fraction × total storage annualized capex × years
    return excess_pct * storage_gw * capex_per_gw_yr * N_YEARS


def _compute_priced_curtailment_usd(iso: str, run: dict) -> float:
    """Component 4: VRE generation in excess of 100% of demand, priced at VRE LCOE.

    This is a mix-excess proxy. The optimizer doesn't compute hourly dispatch
    or curtailment; the chart layer derives it from:
        curtailed_twh[year] = demand[year] × max(0, (Σ endpoint_mix_pct − 100) / 100)
    summed over the horizon, priced at the regional weighted VRE LCOE.
    """
    excess_pct = _mix_excess_pct(run) / 100.0
    if excess_pct <= 0:
        return 0.0
    cost_rows = run.get("tables", {}).get("annual_cost", []) or []
    cum_demand_twh = sum(float(r.get("demand_twh", 0.0)) for r in cost_rows)
    excess_twh = cum_demand_twh * excess_pct
    lcoe = VRE_LCOE_MEDIUM.get(iso, 70.0)
    return excess_twh * 1e6 * lcoe   # $/MWh × MWh


def _components_for_run(iso: str, pathway: str, ep: float) -> dict:
    try:
        run = DL_BASE.get_run(iso, pathway, ep)
    except (KeyError, FileNotFoundError):
        return {"available": False}
    if not run["feasibility"].get("physical", False):
        return {"available": False, "infeasible": True}

    rt = run.get("reliability_tax", {}) or {}
    solver_comp = rt.get("components_usd", {}) or {}
    demand_mwh_cum = float(rt.get("total_demand_mwh_2025_2050", 0.0))

    comp4 = _compute_priced_curtailment_usd(iso, run)
    comp5 = _compute_overbuild_capex_usd(iso, run, demand_mwh_cum)

    components = {
        "new_gas_capex_annualized_usd":      float(solver_comp.get("new_gas_capex_annualized_usd", 0.0)),
        "new_gas_fom_usd":                   float(solver_comp.get("new_gas_fom_usd", 0.0)),
        "existing_gas_fom_carried_usd":      float(solver_comp.get("existing_gas_fom_carried_usd", 0.0)),
        "priced_vre_curtailment_usd":        round(comp4, 0),
        "vre_storage_overbuild_capex_usd":   round(comp5, 0),
    }
    total_usd = sum(components.values())
    tax_usd_per_mwh = (total_usd / demand_mwh_cum) if demand_mwh_cum > 0 else 0.0

    # Per-MWh breakdown for stacked-bar rendering
    per_mwh = {
        k: round(v / demand_mwh_cum, 4) if demand_mwh_cum > 0 else 0.0
        for k, v in components.items()
    }

    return {
        "available": True,
        "achieved_cfe_pct": round(run.get("achieved_cfe_pct", 0.0), 2),
        "total_demand_mwh_2025_2050": round(demand_mwh_cum, 0),
        "components_usd": components,
        "components_usd_per_mwh": per_mwh,
        "total_reliability_tax_usd": round(total_usd, 0),
        "total_reliability_tax_usd_per_mwh": round(tax_usd_per_mwh, 4),
    }


def main() -> None:
    per_iso: dict[str, dict] = {}
    all_taxes: list[float] = []

    for iso in ISOS:
        per_pw: dict[str, dict] = {}
        for pathway in PATHWAYS:
            per_ep: dict[str, dict] = {}
            for ep in ENDPOINTS:
                ep_label = _EP_LABEL[ep]
                result = _components_for_run(iso, pathway, ep)
                per_ep[ep_label] = result
                if result.get("available"):
                    all_taxes.append(result["total_reliability_tax_usd_per_mwh"])
            per_pw[pathway] = per_ep

        # P1a vs P3 sparkline — the key comparative framing.
        sparkline = []
        for ep in ENDPOINTS:
            ep_label = _EP_LABEL[ep]
            p1a = per_pw["1a"].get(ep_label, {})
            p3  = per_pw["3"].get(ep_label, {})
            sparkline.append({
                "endpoint": ep,
                "endpoint_label": ep_label,
                "achieved_cfe_pct_p1a": p1a.get("achieved_cfe_pct") if p1a.get("available") else None,
                "achieved_cfe_pct_p3":  p3.get("achieved_cfe_pct")  if p3.get("available") else None,
                "p1a_tax_usd_per_mwh":  p1a.get("total_reliability_tax_usd_per_mwh") if p1a.get("available") else None,
                "p3_tax_usd_per_mwh":   p3.get("total_reliability_tax_usd_per_mwh")  if p3.get("available") else None,
                "delta_usd_per_mwh": (
                    round(p1a["total_reliability_tax_usd_per_mwh"] - p3["total_reliability_tax_usd_per_mwh"], 4)
                    if p1a.get("available") and p3.get("available") else None
                ),
            })

        per_iso[iso] = {
            "per_pathway": per_pw,
            "sparkline_p1a_vs_p3": sparkline,
        }

    payload = {
        "per_iso": per_iso,
        "global_axis_ranges": {
            "tax_usd_per_mwh": {
                "min": 0.0,
                "max": round(max(all_taxes, default=1.0) * 1.1, 2),
            },
        },
        "meta": {
            "payload_id": "section3_reliability_tax",
            "schema_version": 2,
            "formula": (
                "reliability_tax_$/MWh = "
                "[Σ new_gas_capex_annualized + Σ new_gas_FOM + "
                " Σ existing_gas_FOM_carried + Σ priced_VRE_curtailment + "
                " Σ VRE_storage_overbuild_capex] ÷ Σ demand_MWh (2025-2050)"
            ),
            "gas_sizing_method": (
                "SPEC §24.5 worst-hour residual-gap sizing (99.97th percentile of "
                "margin-inclusive residual), NOT ELCC. Gas components come from "
                "the solver's reliability_tax.components_usd field verbatim."
            ),
            "curtailment_method": (
                "priced_vre_curtailment_usd = cum_demand_TWh × max(0, (Σ endpoint_mix_pct − 100) / 100) "
                "× regional_weighted_VRE_LCOE (Medium toggle). Proxy for hourly dispatch, which "
                "the pathway optimizer does not compute."
            ),
            "overbuild_method": (
                "vre_storage_overbuild_capex_usd = mix_excess_pct × cum_storage_GW × "
                "STORAGE_CAPEX_PER_GW_YR × 26 years. Applied to storage only; VRE "
                "overbuild is priced via the curtailment component."
            ),
            "card_m_prime": (
                "Card M' — no capacity-market-revenue netting. Reliability tax is gross. "
                "There is no net_fom or capacity_rev_netted field anywhere in this payload."
            ),
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    DASH_OUT.parent.mkdir(parents=True, exist_ok=True)
    DASH_OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {DASH_OUT}")

    # Summary: ERCOT + PJM @ ep95 component breakdown
    for iso in ["ERCOT", "PJM"]:
        print(f"\n  {iso} ep95 tax components ($/MWh):")
        for pw in ["1a", "3"]:
            d = per_iso[iso]["per_pathway"][pw]["ep95"]
            if not d.get("available"):
                continue
            c = d["components_usd_per_mwh"]
            total = d["total_reliability_tax_usd_per_mwh"]
            print(
                f"    P{pw}: total=${total:.3f}  "
                f"new_gas_capex=${c['new_gas_capex_annualized_usd']:.3f}  "
                f"new_gas_fom=${c['new_gas_fom_usd']:.3f}  "
                f"existing_gas_fom=${c['existing_gas_fom_carried_usd']:.3f}  "
                f"curt=${c['priced_vre_curtailment_usd']:.3f}  "
                f"overbuild=${c['vre_storage_overbuild_capex_usd']:.3f}"
            )


if __name__ == "__main__":
    main()
