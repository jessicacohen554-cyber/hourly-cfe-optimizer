"""
Shared data loader for reliability_tax chart payloads.

Reads MANIFEST.json and per-run JSON files produced by the Prompt-3 optimizer
(scripts/step_2_3_pathway_optimizer.py + scripts/run_pathway_sweep.py).

Data root: analysis/reliability-tax/data/
  MANIFEST.json          — index of all 140 runs (7 ISO × 4 pathway × 5 endpoint)
  <ISO>/pathway<N>_ep<E>.json — per-run detail files

Public API
----------
get_manifest()
get_run(iso, pathway, endpoint)          -> dict (merged MANIFEST entry + per-run detail)
get_pathway_buildout(iso, pathway, endpoint)   -> list[AnnualBuildoutRow]
get_pathway_cost(iso, pathway, endpoint)       -> list[AnnualCostRow]
get_pathway_stranding(iso, pathway, endpoint)  -> dict (stranding ledger + curtailment cross-check)
get_pathway_dispatch(iso, pathway, endpoint)   -> dict or None (hourly dispatch; None if not computed)
get_pathway_retirements(iso, pathway, endpoint)-> list[RetirementRow]

Helpers
-------
list_isos()           -> list[str]
list_pathways()       -> list[str]
list_endpoints()      -> list[float]
is_feasible(run, mode) -> bool   mode = 'physical' | 'economic' | 'both' | 'any'
get_pivot_info(run)    -> dict
marginal_cost_per_cfe_pct(iso, pathway) -> list[dict]  — segment-by-segment $/CFE%

Demand sensitivity variants (demand_growth_level = 'H' or 'L') can be accessed
by passing data_dir explicitly to any function, or via the DL_DEMAND_H / DL_DEMAND_L
convenience aliases defined at module level.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "analysis" / "reliability-tax" / "data"
DEMAND_H_DATA_DIR = REPO_ROOT / "analysis" / "reliability-tax" / "data-demand-H"
DEMAND_L_DATA_DIR = REPO_ROOT / "analysis" / "reliability-tax" / "data-demand-L"

# Endpoint label map — used to build file names
_EP_LABEL: dict[float, str] = {
    0.90:  "ep90",
    0.95:  "ep95",
    0.975: "ep97p5",
    0.99:  "ep99",
    0.999: "ep99p9",
}

# Canonical ordering used for iteration
ENDPOINTS: list[float] = [0.90, 0.95, 0.975, 0.99, 0.999]
PATHWAYS: list[str]    = ["1", "2a", "2b", "3"]
ISOS: list[str]        = ["CAISO", "ERCOT", "MISO", "NEISO", "NYISO", "PJM", "SPP"]

# Fossil capacity factors (from optimizer, _FOSSIL_CAPACITY_FACTORS)
FOSSIL_CF: dict[str, float] = {"coal": 0.55, "oil": 0.20, "gas": 0.45}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_manifest_cache: dict[Path, dict] = {}
_run_cache: dict[Path, dict] = {}


def _load_manifest(data_dir: Path) -> dict:
    if data_dir not in _manifest_cache:
        p = data_dir / "MANIFEST.json"
        if not p.exists():
            raise FileNotFoundError(f"MANIFEST.json not found at {p}")
        with open(p) as fh:
            _manifest_cache[data_dir] = json.load(fh)
    return _manifest_cache[data_dir]


def _load_run_file(path: Path) -> dict:
    if path not in _run_cache:
        if not path.exists():
            raise FileNotFoundError(f"Run file not found: {path}")
        with open(path) as fh:
            _run_cache[path] = json.load(fh)
    return _run_cache[path]


def _run_key(iso: str, pathway: str, endpoint: float) -> str:
    """Build the MANIFEST run key, e.g. ERCOT__pathway1__ep95."""
    ep = _EP_LABEL.get(endpoint)
    if ep is None:
        raise ValueError(
            f"Unknown endpoint {endpoint}. Valid: {list(_EP_LABEL.keys())}"
        )
    return f"{iso}__pathway{pathway}__{ep}"


def _run_file_path(iso: str, pathway: str, endpoint: float, data_dir: Path) -> Path:
    ep = _EP_LABEL.get(endpoint)
    if ep is None:
        raise ValueError(f"Unknown endpoint {endpoint}")
    return data_dir / iso / f"pathway{pathway}_{ep}.json"


# ---------------------------------------------------------------------------
# Public API — discovery
# ---------------------------------------------------------------------------


def list_isos() -> list[str]:
    """Return canonical list of 7 ISOs."""
    return list(ISOS)


def list_pathways() -> list[str]:
    """Return canonical list of pathways: ['1', '2a', '2b', '3']."""
    return list(PATHWAYS)


def list_endpoints() -> list[float]:
    """Return canonical list of 5 CFE endpoints."""
    return list(ENDPOINTS)


# ---------------------------------------------------------------------------
# Public API — run access
# ---------------------------------------------------------------------------


def get_manifest(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """
    Return the full parsed MANIFEST.json as a dict.

    Top-level key: 'runs' → dict keyed by run_key.
    Each entry contains ISO, pathway, endpoint, feasibility flags, headline
    costs, comparative_stranding, vre_curtailment_at_endpoint, output_path.
    """
    return _load_manifest(data_dir)


def get_run(
    iso: str,
    pathway: str,
    endpoint: float,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict:
    """
    Return merged run record: MANIFEST summary fields + full per-run JSON.

    If the per-run file is missing (output_path absent or file not found),
    returns only the MANIFEST summary with a 'detail_missing' flag.

    Fields in returned dict (union of MANIFEST + per-run JSON):
      run_key, iso, pathway, endpoint,
      feasibility {physical, economic, notes},
      achieved_cfe_pct, undiscounted_cost_usd, npv_at_5pct, npv_at_7pct, npv_at_9pct,
      retirement_timeline,
      comparative_stranding {gw_by_resource, usd_by_resource},
      vre_curtailment_at_endpoint,
      headline (from per-run: includes pivot sub-dict),
      tables {annual_buildout, annual_cost, endpoint_hourly_dispatch, stranding_ledger},
      endpoint_mix_pct, endpoint_storage_pct,
      config (from per-run)
    """
    manifest = _load_manifest(data_dir)
    key = _run_key(iso, pathway, endpoint)
    if key not in manifest.get("runs", {}):
        raise KeyError(f"Run {key} not found in MANIFEST at {data_dir}")

    summary = dict(manifest["runs"][key])

    # Try to load per-run detail file
    run_file = _run_file_path(iso, pathway, endpoint, data_dir)
    if not run_file.exists():
        # Fall back to output_path recorded in manifest
        op = summary.get("output_path")
        if op and Path(op).exists():
            run_file = Path(op)

    if run_file.exists():
        detail = _load_run_file(run_file)
        # Merge: per-run fields supplement the manifest summary
        summary["headline"] = detail.get("headline", {})
        summary["tables"] = detail.get("tables", {})
        summary["endpoint_mix_pct"] = detail.get("endpoint_mix_pct", {})
        summary["endpoint_storage_pct"] = detail.get("endpoint_storage_pct", {})
        summary["config"] = detail.get("config", {})
        # Overwrite retirement_timeline from detail (more complete)
        if detail.get("retirement_timeline"):
            summary["retirement_timeline"] = detail["retirement_timeline"]
        if detail.get("vre_curtailment_at_endpoint"):
            summary["vre_curtailment_at_endpoint"] = detail["vre_curtailment_at_endpoint"]
    else:
        summary["detail_missing"] = True

    return summary


# ---------------------------------------------------------------------------
# Public API — typed accessors (return extracted sub-structures)
# ---------------------------------------------------------------------------


def get_pathway_buildout(
    iso: str,
    pathway: str,
    endpoint: float,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[dict]:
    """
    Return annual buildout table for the run.

    Each row:
      year: int
      new_vintages: list[{resource, twh_per_year, locked_lcoe_usd_per_mwh, tx_adder_usd_per_mwh}]

    Returns [] if run is physically infeasible or file is missing.
    """
    run = get_run(iso, pathway, endpoint, data_dir)
    if not run["feasibility"].get("physical", False):
        return []
    return run.get("tables", {}).get("annual_buildout", [])


def get_pathway_cost(
    iso: str,
    pathway: str,
    endpoint: float,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[dict]:
    """
    Return annual cost table for the run.

    Each row:
      year: int
      demand_twh: float
      gross_operating_usd: float
      capacity_rev_netted_usd: float
      net_annual_cost_usd: float
      achieved_cfe_pct: float

    Returns [] if run is physically infeasible.
    """
    run = get_run(iso, pathway, endpoint, data_dir)
    if not run["feasibility"].get("physical", False):
        return []
    return run.get("tables", {}).get("annual_cost", [])


def get_pathway_stranding(
    iso: str,
    pathway: str,
    endpoint: float,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict:
    """
    Return stranding data for the run (comparative-to-Pathway-3 per Card K revised).

    Returns dict with:
      ledger: list[{resource, pathway_twh, pathway3_twh, stranded_twh, stranded_book_value_usd}]
        — from the per-run stranding_ledger table
      comparative_stranding: {gw_by_resource, usd_by_resource}
        — from MANIFEST (pre-aggregated totals)
      vre_curtailment: {solar, wind, solar_batt8, wind_batt4, wind_batt8}
        — per-resource curtailment rates at endpoint
      total_stranded_usd: float  — sum of stranded_book_value_usd across all resources
      total_stranded_gw: float   — sum of stranded_twh across all resources (in TWh, not GW)
      has_stranding: bool

    Note: per methodology Card K, clean_firm is excluded from stranding.
    Clean firm built post-pivot in Pathway 2 is not stranded.
    Transmission excluded (Card I, bubble-node model).
    """
    run = get_run(iso, pathway, endpoint, data_dir)
    ledger = run.get("tables", {}).get("stranding_ledger", []) or []
    comparative = run.get("comparative_stranding", {})
    curtailment = run.get("vre_curtailment_at_endpoint", {})

    total_usd = sum(row.get("stranded_book_value_usd", 0.0) for row in ledger)
    total_twh = sum(row.get("stranded_twh", 0.0) for row in ledger)

    return {
        "ledger": ledger,
        "comparative_stranding": comparative,
        "vre_curtailment": curtailment,
        "total_stranded_usd": total_usd,
        "total_stranded_twh": total_twh,
        "has_stranding": total_usd > 0,
    }


def get_pathway_dispatch(
    iso: str,
    pathway: str,
    endpoint: float,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict | None:
    """
    Return endpoint hourly dispatch summary, or None if not computed.

    The optimizer sets endpoint_hourly_dispatch to None in the current
    schema_version=1 runs. When populated in future runs it will contain
    per-resource 8760-hour dispatch profiles.
    """
    run = get_run(iso, pathway, endpoint, data_dir)
    return run.get("tables", {}).get("endpoint_hourly_dispatch")


def get_pathway_retirements(
    iso: str,
    pathway: str,
    endpoint: float,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[dict]:
    """
    Return existing-fleet retirement timeline for the run (per Card L revised).

    Each row:
      year: int
      clean_pct: float          — achieved CFE% that year
      coal_retired_gw: float    — cumulative GW of coal economically displaced by this year
      oil_retired_gw: float     — cumulative GW of oil economically displaced
      gas_retired_gw: float     — cumulative GW of gas economically displaced
      coal_retired_twh: float   — cumulative TWh baseline generation displaced (coal)
      oil_retired_twh: float    — cumulative TWh displaced (oil)
      gas_retired_twh: float    — cumulative TWh displaced (gas)

    Values are CUMULATIVE — each year shows total displacement from 2025 to that year.
    Fossil capacity factors used for GW conversion: coal=0.55, oil=0.20, gas=0.45.

    Returns [] for physically infeasible runs.
    """
    run = get_run(iso, pathway, endpoint, data_dir)
    if not run["feasibility"].get("physical", False):
        return []
    return run.get("retirement_timeline", [])


# ---------------------------------------------------------------------------
# Public API — derived metrics
# ---------------------------------------------------------------------------


def get_pivot_info(run: dict) -> dict:
    """
    Extract pivot metadata from a merged run record.

    Returns:
      pivoted: bool
      pivot_year: int or None
      pivot_reason: str or None  — '90pct_plateau' | 'econ_trigger' | None
    """
    headline = run.get("headline", {})
    pivot = headline.get("pivot", {})
    if not pivot:
        # Fall back: Pathway 3 never pivots; Pathway 1 can't pivot
        return {"pivoted": False, "pivot_year": None, "pivot_reason": None}
    return {
        "pivoted": bool(pivot.get("pivoted", False)),
        "pivot_year": pivot.get("pivot_year"),
        "pivot_reason": pivot.get("pivot_reason"),
    }


def is_feasible(run: dict, mode: str = "both") -> bool:
    """
    Check feasibility of a run.

    mode:
      'physical' — physically achievable (enough resource potential)
      'economic' — economically below $10k/CFE% ceiling (Card J)
      'both'     — both physical and economic
      'any'      — either physical or economic
    """
    feas = run.get("feasibility", {})
    phys = bool(feas.get("physical", False))
    econ = bool(feas.get("economic", False))
    if mode == "physical":
        return phys
    if mode == "economic":
        return econ
    if mode == "both":
        return phys and econ
    if mode == "any":
        return phys or econ
    raise ValueError(f"Unknown mode {mode!r}. Use 'physical', 'economic', 'both', 'any'.")


def marginal_cost_per_cfe_pct(
    iso: str,
    pathway: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[dict]:
    """
    Compute segment-by-segment marginal cost per CFE percentage point for
    a given ISO and pathway, across all 5 endpoints in order.

    Returns list of dicts:
      from_cfe_pct: float
      to_cfe_pct: float
      from_endpoint: float       — the endpoint label (e.g. 0.90)
      to_endpoint: float
      from_undiscounted_usd: float
      to_undiscounted_usd: float
      delta_cfe_pct: float
      delta_usd: float
      marginal_usd_per_cfe_pct: float | None  — None if delta_cfe_pct <= 0
      from_feasible_physical: bool
      to_feasible_physical: bool

    Segments where either endpoint is physically infeasible are still included
    but flagged so the chart layer can handle them appropriately.
    """
    results = []
    prev: dict | None = None

    for ep in ENDPOINTS:
        try:
            run = get_run(iso, pathway, ep, data_dir)
        except (KeyError, FileNotFoundError):
            prev = None
            continue

        cfe = run.get("achieved_cfe_pct", 0.0)
        cost = run.get("undiscounted_cost_usd", 0.0)
        phys = run["feasibility"].get("physical", False)

        if prev is not None:
            delta_cfe = cfe - prev["cfe"]
            delta_usd = cost - prev["cost"]
            marginal = (delta_usd / delta_cfe) if delta_cfe > 1e-6 else None
            results.append(
                {
                    "from_endpoint": prev["ep"],
                    "to_endpoint": ep,
                    "from_cfe_pct": prev["cfe"],
                    "to_cfe_pct": cfe,
                    "from_undiscounted_usd": prev["cost"],
                    "to_undiscounted_usd": cost,
                    "delta_cfe_pct": delta_cfe,
                    "delta_usd": delta_usd,
                    "marginal_usd_per_cfe_pct": marginal,
                    "from_feasible_physical": prev["phys"],
                    "to_feasible_physical": phys,
                }
            )

        prev = {"ep": ep, "cfe": cfe, "cost": cost, "phys": phys}

    return results


# ---------------------------------------------------------------------------
# Convenience aliases for demand sensitivity variants
# ---------------------------------------------------------------------------


class _DataLoaderVariant:
    """Bound data-dir variant — call .get_run(), .get_pathway_cost(), etc."""

    def __init__(self, data_dir: Path):
        self._d = data_dir

    def get_manifest(self):
        return get_manifest(self._d)

    def get_run(self, iso, pathway, endpoint):
        return get_run(iso, pathway, endpoint, self._d)

    def get_pathway_buildout(self, iso, pathway, endpoint):
        return get_pathway_buildout(iso, pathway, endpoint, self._d)

    def get_pathway_cost(self, iso, pathway, endpoint):
        return get_pathway_cost(iso, pathway, endpoint, self._d)

    def get_pathway_stranding(self, iso, pathway, endpoint):
        return get_pathway_stranding(iso, pathway, endpoint, self._d)

    def get_pathway_dispatch(self, iso, pathway, endpoint):
        return get_pathway_dispatch(iso, pathway, endpoint, self._d)

    def get_pathway_retirements(self, iso, pathway, endpoint):
        return get_pathway_retirements(iso, pathway, endpoint, self._d)

    def marginal_cost_per_cfe_pct(self, iso, pathway):
        return marginal_cost_per_cfe_pct(iso, pathway, self._d)

    def is_feasible(self, run, mode="both"):
        return is_feasible(run, mode)

    def get_pivot_info(self, run):
        return get_pivot_info(run)


DL_BASE = _DataLoaderVariant(DEFAULT_DATA_DIR)
DL_DEMAND_H = _DataLoaderVariant(DEMAND_H_DATA_DIR)
DL_DEMAND_L = _DataLoaderVariant(DEMAND_L_DATA_DIR)


# ---------------------------------------------------------------------------
# Self-test (run with: python data_loader.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== data_loader self-test ===")
    print(f"Data dir: {DEFAULT_DATA_DIR}")

    manifest = get_manifest()
    runs = manifest.get("runs", {})
    print(f"MANIFEST runs loaded: {len(runs)}")

    # Test ERCOT Pathway 1 @ 95%
    run = get_run("ERCOT", "1", 0.95)
    print(f"\nERCOT / Pathway 1 / 95%:")
    print(f"  Achieved CFE: {run['achieved_cfe_pct']:.1f}%")
    print(f"  Undiscounted cost: ${run['undiscounted_cost_usd']/1e12:.3f}T")
    print(f"  Physical feasible: {is_feasible(run, 'physical')}")
    print(f"  Economic feasible: {is_feasible(run, 'economic')}")
    pivot = get_pivot_info(run)
    print(f"  Pivot: {pivot}")

    buildout = get_pathway_buildout("ERCOT", "1", 0.95)
    print(f"  Annual buildout rows: {len(buildout)}")

    cost = get_pathway_cost("ERCOT", "1", 0.95)
    print(f"  Annual cost rows: {len(cost)}")
    if cost:
        print(f"  2050 net cost: ${cost[-1]['net_annual_cost_usd']/1e9:.1f}B")

    retirements = get_pathway_retirements("ERCOT", "1", 0.95)
    print(f"  Retirement timeline rows: {len(retirements)}")
    if retirements:
        print(f"  2050 gas retired cumulative: {retirements[-1]['gas_retired_gw']:.1f} GW")

    stranding = get_pathway_stranding("ERCOT", "1", 0.95)
    print(f"  Stranding: total ${stranding['total_stranded_usd']/1e12:.3f}T")
    print(f"  Ledger rows: {len(stranding['ledger'])}")

    # Marginal cost
    mc = marginal_cost_per_cfe_pct("ERCOT", "1")
    print(f"\nERCOT Pathway 1 marginal $/CFE%:")
    for seg in mc:
        m = seg["marginal_usd_per_cfe_pct"]
        m_str = f"${m/1e9:.1f}B" if m is not None else "N/A"
        print(f"  {seg['from_cfe_pct']:.1f}% → {seg['to_cfe_pct']:.1f}%: {m_str}/CFE%")

    # Test infeasible ISO
    miso_run = get_run("MISO", "1", 0.90)
    print(f"\nMISO / Pathway 1 / 90%: physical={is_feasible(miso_run, 'physical')}, note={miso_run['feasibility'].get('notes', [])}")

    print("\n=== self-test OK ===")
