"""
gen_section3_reliability_tax.py — Generate section3_reliability_tax.json

v2.1 (Apr 2026 — post §24.8 accounting fix). Rewritten against SPEC §24.4 Card M'
+ the locked 5-component tax formula. Drops all capacity-market-netting logic.
Each component is an independent field so the dashboard can render a stacked bar.

Locked formula (SPEC §24.4):
  reliability_tax_$_per_MWh[pathway, ISO]
    = [ Σ annualized_new_gas_capex
      + Σ new_gas_FOM
      + Σ existing_gas_FOM_carried_forward
      + Σ priced_VRE_curtailment
      + Σ annualized_VRE_storage_overbuild_capex ]
      ÷ Σ demand_MWh

Provenance of each component (v2.1 — all five now read from the solver):
  1. new_gas_capex_annualized_usd   — solver (reliability_tax.components_usd)
  2. new_gas_fom_usd                — solver (Lazard bundles FOM into capex, so
                                      this field is 0 for CCGT; retained as an
                                      explicit bar for future pathway variants)
  3. existing_gas_fom_carried_usd   — solver (EXISTING_GAS_FOM_KW_YR × fleet_MW × 25yr)
  4. priced_vre_curtailment_usd     — solver per SPEC §24.7 formula:
                                      Σ_r surplus_frac[r,y] × demand_mwh[y] ×
                                      vintage_lcoe[r,y] summed 2025–2050. This
                                      chart previously recomputed a mix-excess
                                      proxy locally; it now consumes the solver
                                      value directly so VRE-heavy pathways show
                                      the real stranded-capex bar.
  5. vre_storage_overbuild_capex_usd — solver. Currently 0 in the solver (future
                                       work per §24.7 scope note); keep the
                                       pass-through so the bar lights up the
                                       moment the solver starts emitting it.

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

    components = {
        "new_gas_capex_annualized_usd":      float(solver_comp.get("new_gas_capex_annualized_usd", 0.0)),
        "new_gas_fom_usd":                   float(solver_comp.get("new_gas_fom_usd", 0.0)),
        "existing_gas_fom_carried_usd":      float(solver_comp.get("existing_gas_fom_carried_usd", 0.0)),
        "priced_vre_curtailment_usd":        float(solver_comp.get("priced_vre_curtailment_usd", 0.0)),
        "vre_storage_overbuild_capex_usd":   float(solver_comp.get("vre_storage_overbuild_capex_usd", 0.0)),
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

        # P1 vs P3 sparkline — the key comparative framing per §24.8. P1 is
        # the VRE + storage + offshore headline; P1a is surfaced in per_pathway
        # for anyone who wants the strict-onshore baseline.
        sparkline = []
        for ep in ENDPOINTS:
            ep_label = _EP_LABEL[ep]
            p1 = per_pw["1"].get(ep_label, {})
            p3 = per_pw["3"].get(ep_label, {})
            sparkline.append({
                "endpoint": ep,
                "endpoint_label": ep_label,
                "achieved_cfe_pct_p1": p1.get("achieved_cfe_pct") if p1.get("available") else None,
                "achieved_cfe_pct_p3": p3.get("achieved_cfe_pct") if p3.get("available") else None,
                "p1_tax_usd_per_mwh":  p1.get("total_reliability_tax_usd_per_mwh") if p1.get("available") else None,
                "p3_tax_usd_per_mwh":  p3.get("total_reliability_tax_usd_per_mwh") if p3.get("available") else None,
                "delta_usd_per_mwh": (
                    round(p1["total_reliability_tax_usd_per_mwh"] - p3["total_reliability_tax_usd_per_mwh"], 4)
                    if p1.get("available") and p3.get("available") else None
                ),
            })

        per_iso[iso] = {
            "per_pathway": per_pw,
            "sparkline_p1_vs_p3": sparkline,
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
                "priced_vre_curtailment_usd read verbatim from solver per SPEC §24.7: "
                "Σ_r surplus_frac[r,y] × demand_mwh[y] × vintage_lcoe[r,y] summed "
                "2025–2050. Vintage LCOE dilutes with zero-LCOE existing-fleet "
                "vintages seeded at cod_year 2024 (SPEC §24.8). VRE-heavy pathways "
                "dominate the stack at high CFE; clean-firm-heavy P3 runs register 0."
            ),
            "overbuild_method": (
                "vre_storage_overbuild_capex_usd read verbatim from solver. "
                "Currently 0 in every run — future work per SPEC §24.7 scope note. "
                "Kept as a pass-through bar so it lights up the day the solver "
                "emits it."
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

    # Summary: ERCOT + PJM @ ep90 component breakdown (matches §24.8 empirical table)
    for iso in ["ERCOT", "PJM"]:
        print(f"\n  {iso} ep90 tax components ($/MWh):")
        for pw in ["1", "3"]:
            d = per_iso[iso]["per_pathway"][pw]["ep90"]
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
