"""
gen_section6_cost_of_waiting.py — Generate section6_cost_of_waiting.json

v2 (Apr 2026). The "cost of waiting" curve answers: how does the
reliability tax escalate if the system waits N more years before
committing to a clean-firm build-out?

Framing under SPEC §24.4:
  - Pathway 3 (2025 anchor): commit to clean firm immediately. Lowest reliability tax.
  - Pathway 2a (pivot_year anchor): delay then pivot late when VRE-only hits
    its floor. New-build gas accumulates before pivot and ends up stranded
    (SPEC Card F' trigger, CF<15% for 2 yrs).
  - Pathway 1a (2050 upper bound): never commit to clean firm. Accumulates
    the most stranded gas and most residual gap on the worst hour.

No per-year-of-commitment runs exist — curves are LINEARLY INTERPOLATED
between the three anchors above. The penalty vs P3 is decomposed into
the 5 locked reliability-tax components from SPEC §24.4:
  1. annualized_new_gas_capex
  2. new_gas_FOM
  3. existing_gas_FOM_carried
  4. priced_VRE_curtailment
  5. annualized_VRE_storage_overbuild_capex

All values are gross of any capacity-market netting (Card M').

Payload:
  per_iso[ISO] = {
    pivot_year_2a, pivot_year_2b,
    by_endpoint[ep_label] = {
      commitment_curve: [{commit_year, interp_weight, rtax_usd_per_mwh,
                          rtax_total_usd, penalty_vs_p3_usd_per_mwh,
                          components_usd: {...}}],
      anchors: {p3, p2a, p1a} each = {rtax_usd_per_mwh, rtax_total_usd, components_usd}
    }
  }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import (  # noqa: E402
    DL_BASE, get_pivot_info,
    ISOS, ENDPOINTS, _EP_LABEL,
)

OUT_PATH = Path(__file__).parent / "section6_cost_of_waiting.json"
DASH_OUT = REPO_ROOT / "dashboard" / "js" / "reliability-tax" / "section6_cost_of_waiting.json"

COMMITMENT_YEARS = [2026, 2030, 2035, 2040, 2045]

# Canonical component keys (SPEC §24.4)
COMPONENT_KEYS = [
    "new_gas_capex_annualized_usd",
    "new_gas_fom_usd",
    "existing_gas_fom_carried_usd",
    "priced_vre_curtailment_usd",
    "vre_storage_overbuild_capex_usd",
]


def _zero_components() -> dict:
    return {k: 0.0 for k in COMPONENT_KEYS}


def _rtax_snapshot(run: dict) -> dict:
    """Pull the reliability tax snapshot from one run."""
    if not run.get("feasibility", {}).get("physical", False):
        return {
            "feasible": False,
            "rtax_total_usd": 0.0,
            "rtax_usd_per_mwh": 0.0,
            "total_demand_mwh": 0.0,
            "components_usd": _zero_components(),
        }
    rt = run.get("reliability_tax", {}) or {}
    comps = rt.get("components_usd", {}) or {}
    return {
        "feasible": True,
        "rtax_total_usd": float(rt.get("total_usd", 0.0)),
        "rtax_usd_per_mwh": float(rt.get("usd_per_mwh", 0.0)),
        "total_demand_mwh": float(rt.get("total_demand_mwh_2025_2050", 0.0)),
        "components_usd": {k: float(comps.get(k, 0.0)) for k in COMPONENT_KEYS},
    }


def _interp(commit_year: int, pivot_year: int, v3: float, v2a: float, v1a: float) -> float:
    """Linear interpolation between P3 (2025), P2a (pivot_year), P1a (2050)."""
    if commit_year <= 2025:
        return v3
    if commit_year >= 2050:
        return v1a
    if commit_year <= pivot_year:
        t = (commit_year - 2025) / max(pivot_year - 2025, 1)
        return v3 + t * (v2a - v3)
    t = (commit_year - pivot_year) / max(2050 - pivot_year, 1)
    return v2a + t * (v1a - v2a)


def _interp_components(
    commit_year: int, pivot_year: int,
    c3: dict, c2a: dict, c1a: dict,
) -> dict:
    return {
        k: _interp(commit_year, pivot_year, c3.get(k, 0.0), c2a.get(k, 0.0), c1a.get(k, 0.0))
        for k in COMPONENT_KEYS
    }


def _compute_per_ep(iso: str, ep: float) -> dict:
    try:
        r3 = DL_BASE.get_run(iso, "3", ep)
    except (KeyError, FileNotFoundError):
        r3 = {}
    try:
        r2a = DL_BASE.get_run(iso, "2a", ep)
    except (KeyError, FileNotFoundError):
        r2a = {}
    try:
        r1a = DL_BASE.get_run(iso, "1a", ep)
    except (KeyError, FileNotFoundError):
        r1a = {}

    snap3 = _rtax_snapshot(r3)
    snap2a = _rtax_snapshot(r2a)
    snap1a = _rtax_snapshot(r1a)

    pivot_info = get_pivot_info(r2a) if r2a else {}
    pivot_year = pivot_info.get("pivot_year") or 2041

    curve = []
    for cy in COMMITMENT_YEARS:
        interp_weight = min(1.0, max(0.0, (cy - 2025) / max(pivot_year - 2025, 1)))

        rtax_usd_per_mwh = _interp(
            cy, pivot_year,
            snap3["rtax_usd_per_mwh"], snap2a["rtax_usd_per_mwh"], snap1a["rtax_usd_per_mwh"],
        )
        rtax_total_usd = _interp(
            cy, pivot_year,
            snap3["rtax_total_usd"], snap2a["rtax_total_usd"], snap1a["rtax_total_usd"],
        )
        comps = _interp_components(
            cy, pivot_year,
            snap3["components_usd"], snap2a["components_usd"], snap1a["components_usd"],
        )

        curve.append({
            "commit_year": cy,
            "interp_weight": round(interp_weight, 3),
            "rtax_usd_per_mwh": round(rtax_usd_per_mwh, 2),
            "rtax_total_usd": round(rtax_total_usd, 0),
            "penalty_vs_p3_usd_per_mwh": round(
                rtax_usd_per_mwh - snap3["rtax_usd_per_mwh"], 2),
            "penalty_vs_p3_total_usd": round(
                rtax_total_usd - snap3["rtax_total_usd"], 0),
            "components_usd": {k: round(v, 0) for k, v in comps.items()},
        })

    return {
        "commitment_curve": curve,
        "anchors": {
            "p3": {
                "anchor_year": 2025,
                "rtax_usd_per_mwh": round(snap3["rtax_usd_per_mwh"], 2),
                "rtax_total_usd": round(snap3["rtax_total_usd"], 0),
                "components_usd": {k: round(v, 0) for k, v in snap3["components_usd"].items()},
                "feasible": snap3["feasible"],
            },
            "p2a": {
                "anchor_year": pivot_year,
                "rtax_usd_per_mwh": round(snap2a["rtax_usd_per_mwh"], 2),
                "rtax_total_usd": round(snap2a["rtax_total_usd"], 0),
                "components_usd": {k: round(v, 0) for k, v in snap2a["components_usd"].items()},
                "feasible": snap2a["feasible"],
            },
            "p1a": {
                "anchor_year": 2050,
                "rtax_usd_per_mwh": round(snap1a["rtax_usd_per_mwh"], 2),
                "rtax_total_usd": round(snap1a["rtax_total_usd"], 0),
                "components_usd": {k: round(v, 0) for k, v in snap1a["components_usd"].items()},
                "feasible": snap1a["feasible"],
            },
        },
        "pivot_year_2a": pivot_year,
    }


def main() -> None:
    per_iso: dict[str, dict] = {}

    for iso in ISOS:
        try:
            run_2a = DL_BASE.get_run(iso, "2a", 0.99)
            pivot_year_2a = get_pivot_info(run_2a).get("pivot_year") or 2041
        except Exception:
            pivot_year_2a = 2041

        try:
            run_2b = DL_BASE.get_run(iso, "2b", 0.99)
            pivot_year_2b = get_pivot_info(run_2b).get("pivot_year")
        except Exception:
            pivot_year_2b = None

        by_endpoint: dict[str, dict] = {}
        for ep in ENDPOINTS:
            by_endpoint[_EP_LABEL[ep]] = _compute_per_ep(iso, ep)

        per_iso[iso] = {
            "pivot_year_2a": pivot_year_2a,
            "pivot_year_2b": pivot_year_2b,
            "by_endpoint": by_endpoint,
        }

    payload = {
        "commitment_years": COMMITMENT_YEARS,
        "per_iso": per_iso,
        "meta": {
            "payload_id": "section6_cost_of_waiting",
            "schema_version": 2,
            "headline_metric": "reliability_tax_usd_per_mwh",
            "reliability_tax_formula": (
                "SPEC §24.4 locked 5-component formula: "
                "(annualized_new_gas_capex + new_gas_FOM + existing_gas_FOM_carried "
                "+ priced_VRE_curtailment + annualized_VRE_storage_overbuild_capex) "
                "÷ cumulative_demand_MWh_2025_2050. Gross of any capacity-market netting."
            ),
            "anchors": {
                "p3": "Commit to clean firm in 2025 (lowest reliability tax)",
                "p2a": "Delay then pivot at pivot_year (new-gas stranding accumulates)",
                "p1a": "Never commit — 2050 upper bound (max new-gas + max residual gap)",
            },
            "interpolation_method": (
                "Linear between P3 (2025), P2a (pivot_year), P1a (2050). "
                "Per-commit-year optimizer runs do not exist in the sweep."
            ),
            "gas_sizing_basis": "SPEC §24.5 worst-hour (99.97th percentile), not ELCC",
            "left_panel": {
                "x_axis": "Commitment year",
                "y_axis": "Reliability tax ($/MWh, 2025-2050 avg)",
                "reference": "Pathway 3 anchor at 2025",
            },
            "right_panel": {
                "x_axis": "Commitment year",
                "y_axis": "Penalty vs Pathway 3 ($/MWh)",
                "stacked_series": COMPONENT_KEYS,
                "note": "Stacked decomposition of the 5-component reliability tax.",
            },
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    DASH_OUT.parent.mkdir(parents=True, exist_ok=True)
    DASH_OUT.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {DASH_OUT}")

    print("\n--- section6_cost_of_waiting quick summary @ ep95 ---")
    print(f"{'ISO':6s} | pivot | P3 $/MWh | P2a $/MWh | P1a $/MWh | 2040 penalty $/MWh")
    print("-" * 80)
    for iso in ISOS:
        d = per_iso[iso]["by_endpoint"]["ep95"]
        anchors = d["anchors"]
        curve = d["commitment_curve"]
        p2040 = next((c for c in curve if c["commit_year"] == 2040), {})
        print(
            f"{iso:6s} | {d['pivot_year_2a']} | "
            f"{anchors['p3']['rtax_usd_per_mwh']:8.2f} | "
            f"{anchors['p2a']['rtax_usd_per_mwh']:8.2f} | "
            f"{anchors['p1a']['rtax_usd_per_mwh']:8.2f} | "
            f"{p2040.get('penalty_vs_p3_usd_per_mwh', 0):+.2f}"
        )


if __name__ == "__main__":
    main()
