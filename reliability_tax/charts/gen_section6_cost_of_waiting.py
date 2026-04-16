"""
gen_section6_cost_of_waiting.py — Generate section6_cost_of_waiting.json

Data lineage:
  - analysis/reliability-tax/data/MANIFEST.json + per-run JSONs
  - Pathway 3 = "commit immediately" (2025 proactive build)
  - Pathway 2a = "behavioral late pivot" (pivot ~2041 for most ISOs)
  - Pathway 1  = "never commit" (maximum delay / VRE-only through 2050)
  - Commitment years 2026, 2030, 2035, 2040, 2045 (Card Q)

Construction methodology:
  The "cost of waiting" curve maps commitment year → NPV penalty vs. Pathway 3.
  No per-year-of-commitment runs exist. Penalty is INTERPOLATED:
    - Commit 2026 ≈ P3 cost (1 year delay → ~5% of P2a penalty)
    - Commit 2030 → linear interpolation between P3 (2025 anchor) and P2a (pivot_year anchor)
    - Commit 2035 → continuing interpolation
    - Commit 2040 ≈ P2a cost (pivot year ≈ 2041)
    - Commit 2045 → linear interpolation between P2a and P1 (full VRE-only)
  Interpolation basis: years_of_delay / (pivot_year - 2025)
  For ISOs where pivot_year != 2041 (if P2b triggers immediately), use P1 as upper bound.

Left panel (cumulative cost vs commitment year):
  - undiscounted_cumulative_trillion
  - npv_7pct_trillion
  - npv_5pct_trillion (lower bound of band)
  - npv_9pct_trillion (upper bound of band)

Right panel (penalty composition = difference vs P3):
  - stranded_assets_billion: from stranding_ledger.total_stranded_usd
  - foak_premium_billion: estimated as FOAK_fraction × remaining_penalty
  - foregone_learning_billion: residual penalty after stranded + FOAK
  - total_penalty_billion: sum of above = total_penalty vs P3 at NPV@7%

Note: FOAK premium and foregone learning are ESTIMATED using indicative fractions
(30% / 70% split of non-stranding penalty), not independently computed. Flagged in meta.

Payload structure:
  per_iso: dict[iso ->
    pivot_year_2a: int
    pivot_year_2b: int (or None if no pivot)
    by_endpoint: dict[ep_label ->
      commitment_curve: list[5 × {
        commit_year, interpolation_basis, interpolation_weight,
        undiscounted_cumulative_trillion, npv_7pct_trillion, npv_5pct_trillion, npv_9pct_trillion,
        penalty_vs_p3_npv7_trillion,
        penalty_composition: {stranded_assets, foak_premium, foregone_learning}
      }]
      reference_p3: {undiscounted, npv_7pct, npv_5pct, npv_9pct}
      max_penalty_pathway: {pathway, penalty_npv7_trillion, commit_year_equivalent}
    ]
  ]
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
    ISOS, ENDPOINTS, _EP_LABEL,
)

OUT_PATH = Path(__file__).parent / "section6_cost_of_waiting.json"

COMMITMENT_YEARS = [2026, 2030, 2035, 2040, 2045]

# FOAK premium share of non-stranding penalty (indicative fraction)
FOAK_FRACTION = 0.30   # 30% of non-stranded penalty = FOAK premium
LEARNING_FRACTION = 0.70  # remainder = foregone learning benefit

# Card S: new-build gas stranding parameters
CCS_CCGT_CF_STRAND = 0.85
CCS_CCGT_CAPITAL_PER_GW = 2.2e9   # $2,200/kW = $2.2B/GW (NREL ATB 2024 moderate)
CCS_CCGT_USEFUL_LIFE = 25
MODEL_ENDPOINT_YEAR = 2050


def _new_gas_stranding(run: dict) -> float:
    """Return stranded book value (USD) for new CCS-CCGT not fully amortized by 2050."""
    bo = run.get("tables", {}).get("annual_buildout", [])
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
            total_usd += gw * CCS_CCGT_CAPITAL_PER_GW * frac
    return total_usd


def _interpolate_cost(
    commit_year: int,
    pivot_year: int,
    p3_val: float,
    p2a_val: float,
    p1_val: float,
) -> float:
    """
    Linearly interpolate cost at commitment year between P3, P2a, P1 anchors.
    - Before pivot_year: linear from P3 (2025) to P2a (pivot_year)
    - After pivot_year: linear from P2a (pivot_year) to P1 (2050)
    """
    if commit_year <= 2025:
        return p3_val
    if commit_year >= 2050:
        return p1_val

    if commit_year <= pivot_year:
        # Interpolate between P3 and P2a
        t = (commit_year - 2025) / max(pivot_year - 2025, 1)
        return p3_val + t * (p2a_val - p3_val)
    else:
        # Interpolate between P2a and P1
        t = (commit_year - pivot_year) / max(2050 - pivot_year, 1)
        return p2a_val + t * (p1_val - p2a_val)


def _compute_per_ep(iso: str, ep: float) -> dict:
    """Compute cost-of-waiting curve for one ISO × endpoint."""
    # Get all four pathway runs
    runs: dict[str, dict] = {}
    for path in ["1a", "2a", "2b", "3"]:
        try:
            runs[path] = DL_BASE.get_run(iso, path, ep)
        except (KeyError, FileNotFoundError):
            runs[path] = {}

    r3 = runs.get("3", {})
    r2a = runs.get("2a", {})
    r1 = runs.get("1a", {})  # P1a = true "never commit" VRE-only upper bound

    # Reference P3 (baseline, zero penalty)
    p3_undiscounted = r3.get("undiscounted_cost_usd", 0.0) / 1e12
    p3_npv7 = r3.get("npv_at_7pct", 0.0) / 1e12
    p3_npv5 = r3.get("npv_at_5pct", 0.0) / 1e12
    p3_npv9 = r3.get("npv_at_9pct", 0.0) / 1e12
    # Card S: P3 actual cost = undiscounted + new-gas stranding (P3 builds clean firm proactively, no gas stranding)
    p3_total_actual = p3_undiscounted + _new_gas_stranding(r3) / 1e12

    # P2a anchor (late pivot)
    p2a_undiscounted = r2a.get("undiscounted_cost_usd", 0.0) / 1e12
    p2a_npv7 = r2a.get("npv_at_7pct", 0.0) / 1e12
    p2a_npv5 = r2a.get("npv_at_5pct", 0.0) / 1e12
    p2a_npv9 = r2a.get("npv_at_9pct", 0.0) / 1e12
    # Card S: P2a actual cost includes stranded new gas built before pivot
    p2a_ng_stranded_usd = _new_gas_stranding(r2a)
    p2a_total_actual = p2a_undiscounted + p2a_ng_stranded_usd / 1e12
    pivot_info = get_pivot_info(r2a)
    pivot_year = pivot_info.get("pivot_year") or 2041

    # P1 upper bound (maximum delay) — P1a doesn't build new gas, so no gas stranding
    p1_undiscounted = r1.get("undiscounted_cost_usd", 0.0) / 1e12
    p1_npv7 = r1.get("npv_at_7pct", 0.0) / 1e12
    p1_npv5 = r1.get("npv_at_5pct", 0.0) / 1e12
    p1_npv9 = r1.get("npv_at_9pct", 0.0) / 1e12
    p1_total_actual = p1_undiscounted  # P1a: no new gas builds

    # Stranding data for P2a (for penalty decomposition)
    strand_2a = get_pathway_stranding(iso, "2a", ep)
    stranded_2a_usd = strand_2a["total_stranded_usd"]  # comparative (VRE) stranding

    # Card S: total P2a stranding = VRE comparative stranding + new gas book value
    total_p2a_stranding_usd = stranded_2a_usd + p2a_ng_stranded_usd

    # NPV-equivalent stranded for penalty decomposition (50% recovery heuristic)
    stranded_npv_approx = total_p2a_stranding_usd * 0.50 / 1e12  # trillion

    # Build commitment curve for 5 years
    commitment_curve = []
    for commit_year in COMMITMENT_YEARS:
        interp_weight = min(1.0, max(0.0, (commit_year - 2025) / max(pivot_year - 2025, 1)))

        undiscounted = _interpolate_cost(commit_year, pivot_year, p3_undiscounted, p2a_undiscounted, p1_undiscounted)
        # Card S: primary cost = total actual (undiscounted + new gas stranding), interpolated
        total_actual = _interpolate_cost(commit_year, pivot_year, p3_total_actual, p2a_total_actual, p1_total_actual)
        npv7 = _interpolate_cost(commit_year, pivot_year, p3_npv7, p2a_npv7, p1_npv7)
        npv5 = _interpolate_cost(commit_year, pivot_year, p3_npv5, p2a_npv5, p1_npv5)
        npv9 = _interpolate_cost(commit_year, pivot_year, p3_npv9, p2a_npv9, p1_npv9)

        penalty_npv7 = npv7 - p3_npv7  # can be negative if P1/P2a < P3
        penalty_actual = total_actual - p3_total_actual  # Card S: primary penalty

        # Penalty decomposition (indicative, not independently computed)
        # Only decompose if penalty is positive
        if penalty_npv7 > 0:
            # Scale stranding contribution by interpolation weight
            stranded_contrib = min(stranded_npv_approx * interp_weight, penalty_npv7)
            remainder = penalty_npv7 - stranded_contrib
            foak_contrib = remainder * FOAK_FRACTION
            learning_contrib = remainder * LEARNING_FRACTION
        else:
            stranded_contrib = 0.0
            foak_contrib = 0.0
            learning_contrib = 0.0

        commitment_curve.append({
            "commit_year": commit_year,
            "interpolation_basis": "linear between P3 (2025) and P2a (pivot_year) and P1 (2050)",
            "interpolation_weight": round(interp_weight, 3),
            # Primary cost metrics (Card S)
            "total_actual_cost_trillion": round(total_actual, 4),
            "penalty_vs_p3_actual_trillion": round(penalty_actual, 4),
            # Secondary: undiscounted operating cost only
            "undiscounted_cumulative_trillion": round(undiscounted, 4),
            # Secondary: NPV
            "npv_7pct_trillion": round(npv7, 4),
            "npv_5pct_trillion": round(npv5, 4),
            "npv_9pct_trillion": round(npv9, 4),
            "penalty_vs_p3_npv7_trillion": round(penalty_npv7, 4),
            "penalty_composition": {
                "stranded_assets_trillion": round(stranded_contrib, 4),
                "foak_premium_trillion": round(foak_contrib, 4),
                "foregone_learning_trillion": round(learning_contrib, 4),
                "note": (
                    "INDICATIVE decomposition only. Stranded assets from stranding_ledger "
                    "(book value × 50% NPV factor). FOAK premium = 30% of non-stranded remainder. "
                    "Foregone learning = 70% of non-stranded remainder."
                ),
            },
        })

    # Max penalty pathway
    max_penalty = max(
        abs(r2a.get("npv_at_7pct", 0) - r3.get("npv_at_7pct", 0)),
        abs(r1.get("npv_at_7pct", 0) - r3.get("npv_at_7pct", 0)),
    ) / 1e12
    max_penalty_pathway = "1a" if abs(r1.get("npv_at_7pct", 0) - r3.get("npv_at_7pct", 0)) >= abs(r2a.get("npv_at_7pct", 0) - r3.get("npv_at_7pct", 0)) else "2a"

    return {
        "commitment_curve": commitment_curve,
        "reference_p3": {
            # Card S: primary = total actual cost (undiscounted + new gas stranding)
            "total_actual_cost_trillion": round(p3_total_actual, 4),
            "undiscounted_trillion": round(p3_undiscounted, 4),
            "npv_7pct_trillion": round(p3_npv7, 4),
            "npv_5pct_trillion": round(p3_npv5, 4),
            "npv_9pct_trillion": round(p3_npv9, 4),
        },
        "anchor_p2a": {
            "total_actual_cost_trillion": round(p2a_total_actual, 4),
            "new_gas_stranded_billion": round(p2a_ng_stranded_usd / 1e9, 2),
            "undiscounted_trillion": round(p2a_undiscounted, 4),
            "npv_7pct_trillion": round(p2a_npv7, 4),
            "pivot_year": pivot_year,
            "penalty_vs_p3_actual_trillion": round(p2a_total_actual - p3_total_actual, 4),
            "penalty_vs_p3_npv7_trillion": round(p2a_npv7 - p3_npv7, 4),
        },
        "anchor_p1": {
            "total_actual_cost_trillion": round(p1_total_actual, 4),
            "undiscounted_trillion": round(p1_undiscounted, 4),
            "npv_7pct_trillion": round(p1_npv7, 4),
            "penalty_vs_p3_actual_trillion": round(p1_total_actual - p3_total_actual, 4),
            "penalty_vs_p3_npv7_trillion": round(p1_npv7 - p3_npv7, 4),
        },
        "max_penalty_trillion": round(max_penalty, 4),
        "max_penalty_pathway": max_penalty_pathway,
        "p2a_stranded_book_value_trillion": round(total_p2a_stranding_usd / 1e12, 4),
        "p2a_new_gas_stranded_book_value_trillion": round(p2a_ng_stranded_usd / 1e12, 4),
    }


def main() -> None:
    per_iso: dict[str, dict] = {}

    for iso in ISOS:
        # Get pivot year from ep99 P2a (most reliable indicator)
        try:
            run_2a = DL_BASE.get_run(iso, "2a", 0.99)
            pivot_info = get_pivot_info(run_2a)
            pivot_year_2a = pivot_info.get("pivot_year") or 2041
        except Exception:
            pivot_year_2a = 2041

        try:
            run_2b = DL_BASE.get_run(iso, "2b", 0.99)
            pivot_info_2b = get_pivot_info(run_2b)
            pivot_year_2b = pivot_info_2b.get("pivot_year")
        except Exception:
            pivot_year_2b = None

        by_endpoint: dict[str, dict] = {}
        for ep in ENDPOINTS:
            ep_label = _EP_LABEL[ep]
            by_endpoint[ep_label] = _compute_per_ep(iso, ep)

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
            "note": (
                "Cost of waiting curves for 5 commitment years × 7 ISOs × 5 endpoints. "
                "No per-year-of-commitment optimizer runs exist; curves are INTERPOLATED "
                "between P3 (2025 anchor), P2a (pivot_year anchor), P1 (2050 upper bound). "
                "Penalty decomposition is INDICATIVE (not independently computed): "
                "stranded assets from stranding_ledger book value × 50% NPV factor; "
                "FOAK premium and foregone learning estimated as 30%/70% of remainder."
            ),
            "left_panel": {
                "x_axis": "Commitment year (first year clean firm becomes available)",
                "y_axis": "Cumulative system cost (2025–2050)",
                "series": ["undiscounted", "npv_7pct", "npv_5pct_band", "npv_9pct_band"],
                "reference": "Pathway 3 (commit in 2025, proactive clean firm)",
            },
            "right_panel": {
                "x_axis": "Commitment year",
                "y_axis": "Additional cost vs Pathway 3 ($T NPV@7%)",
                "series": ["stranded_assets", "foak_premium", "foregone_learning"],
                "note": "Stacked bar. Only positive-penalty ISOs show meaningful decomposition.",
            },
            "card_q_commitment_years": COMMITMENT_YEARS,
            "interpolation_method": "linear between P3/P2a pivot_year/P1 anchors",
            "version": "1.0",
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")

    print("\n--- section6_cost_of_waiting quick summary ---")
    print("ISO | pivot_2a | commit2026 penalty | commit2040 penalty | commit2045 penalty (NPV@7%)")
    print("-" * 90)
    for iso in ISOS:
        ep_data = per_iso[iso]["by_endpoint"]["ep99"]
        pivot_year = per_iso[iso]["pivot_year_2a"]
        curve = ep_data["commitment_curve"]
        p2026 = next((c for c in curve if c["commit_year"] == 2026), {})
        p2040 = next((c for c in curve if c["commit_year"] == 2040), {})
        p2045 = next((c for c in curve if c["commit_year"] == 2045), {})
        print(
            f"  {iso:6s} | {pivot_year} | "
            f"{p2026.get('penalty_vs_p3_npv7_trillion', 0):+.3f}T | "
            f"{p2040.get('penalty_vs_p3_npv7_trillion', 0):+.3f}T | "
            f"{p2045.get('penalty_vs_p3_npv7_trillion', 0):+.3f}T"
        )


if __name__ == "__main__":
    main()
