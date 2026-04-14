"""Step 2.3 — Reliability Tax Pathway Optimizer.

Deterministic NPV-minimization of the 2025–2050 cost to reach a fixed CFE
endpoint under four decarbonization pathways. Not a market prediction.

Parameters
----------
iso      : one of CAISO, ERCOT, PJM, NYISO, NEISO, MISO, SPP
pathway  : one of {1, 2a, 2b, 3}
             1  = VRE + batteries only (no clean firm ever)
             2a = VRE-first until Pathway 1 hits 90% CFE plateau → pivot to clean firm
             2b = VRE-first until marginal $/CFE% > cheapest clean-firm LCOE → pivot
             3  = Clean-firm proactive (available year 1 on Wright's Law curve)
endpoint : one of {0.90, 0.95, 0.975, 0.99, 0.999}

Objective
---------
Minimize NPV@7% real of cumulative 2025–2050 system cost subject to the
endpoint constraint being met at 2050. Also report undiscounted cumulative and
NPV@5% / NPV@9% for sensitivity. Real 2025 dollars; no inflation adjustment.

Run sequence
------------
For each (iso, endpoint) combo, Pathway 3 runs FIRST. The Card K comparative
stranding definition requires Pathway 3's buildout as a reference.

Inheritance (confirmed in Prompt 2A inheritance discovery)
-----------------------------------------------------------
- Wright's Law engine : pipeline_config.learning_fraction / year_adjusted_cost
- FOAK tables         : pipeline_config.FOAK_NUCLEAR_NEWBUILD, FOAK_CCS_45Q_ON/OFF,
                        FOAK_GEOTHERMAL, FOAK_OFFSHORE_WIND, FOAK_LDES, FOAK_H2
- Cost tables         : pipeline_config.LCOE_TABLES, TX_TABLES, NOAK_BATTERY/8,
                        NUCLEAR_NEWBUILD_LCOE, CCS_LCOE_45Q_ON/OFF, GEOTHERMAL_LCOE
- Clean-firm tranches : pipeline_config.compute_clean_firm_tranches (merit order)
- Retirement (Card L) : dispatch_utils.compute_fossil_retirement
                        (+ capacity-revenue netting per Card M to proxy capacity-market
                        clearing; confirmed with user)
- Capacity revenue    : step6_1_smartargets.compute_capacity_revenue (Card M)
- Demand growth       : pipeline_config.DEMAND_GROWTH_RATES (L/M/H from config; no
                        new growth rates invented)
- Existing fossil src : data/eia-930/eia_fossil_mix.parquet (hourly coal/gas/oil
                        share by ISO × year; confirmed with user)

Strip list (NOT inherited from Step 6)
---------------------------------------
Policy scenarios (IRA, RGGI, RPS, SBTi emission trajectories), regulatory
feedback price damping, nuclear policy-floor retirement, Monte-Carlo sweeps,
market-clearing LMP forecasts beyond deterministic merit order, stochastic
scenario trees. This optimizer is pure deterministic cost minimization.

Output
------
analysis/reliability-tax/data/<iso>/<pathway>_<endpoint>.json
  Contains four tables:
    - annual_buildout : year, resource, new_cap_gw, cumulative_cap_gw
    - annual_cost     : year, capex, opex, fuel, transmission, capacity_rev,
                        existing_fleet_cost, total
    - endpoint_hourly_dispatch : 8760-row dispatch snapshot at 2050
    - stranding_ledger : comparative to Pathway 3 (Card K revised)
MANIFEST.json at analysis/reliability-tax/data/MANIFEST.json aggregates all runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow imports from scripts/ when run standalone.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import pipeline_config as pc  # noqa: E402

try:
    import dispatch_utils as du  # noqa: E402
except Exception as _du_err:  # pragma: no cover — defer hard failure to use-site
    du = None
    _DU_IMPORT_ERROR: Exception | None = _du_err
else:
    _DU_IMPORT_ERROR = None

try:
    import step6_1_smartargets as s6  # noqa: E402
except Exception as _s6_err:  # pragma: no cover — defer hard failure to use-site
    s6 = None
    _S6_IMPORT_ERROR: Exception | None = _s6_err
else:
    _S6_IMPORT_ERROR = None


# ============================================================================
# CONFIG CONSTANTS
# ============================================================================

PATHWAYS = ('1', '2a', '2b', '3')
ENDPOINTS = (0.90, 0.95, 0.975, 0.99, 0.999)
ISOS = tuple(pc.ISOS)

BASE_YEAR = 2025
END_YEAR = 2050
YEARS = list(range(BASE_YEAR, END_YEAR + 1))  # 26 year-indices inclusive

# Card D — one discount rate for the objective; 5/9% reported as sensitivity.
OBJECTIVE_DISCOUNT_RATE = 0.07
REPORTING_DISCOUNT_RATES = (0.05, 0.07, 0.09)

# Card J — economic-floor diagnostic threshold.
ECONOMIC_INFEASIBILITY_MARGINAL_USD_PER_CFE_PCT = 10_000.0

# Card Q — Section 6 commitment years (for downstream Cost-of-Waiting work).
COMMITMENT_YEARS = (2026, 2030, 2035, 2040, 2045)

# Repo-relative data roots.
REPO_ROOT = _THIS_DIR.parent
DATA_STEP21 = REPO_ROOT / 'data' / 'step2.1-ef'
DATA_STEP22 = REPO_ROOT / 'data' / 'step2.2-cost'
DATA_STEP3 = REPO_ROOT / 'data' / 'step3-dispatch'
DATA_EIA930 = REPO_ROOT / 'data' / 'eia-930'

# Default output root (prompt specifies analysis/reliability-tax/data).
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'analysis' / 'reliability-tax' / 'data'
MANIFEST_PATH_SUFFIX = 'MANIFEST.json'

# ISO cadence of available EF parquets — filled lazily on first use.
_AVAILABLE_THRESHOLDS_CACHE: dict[str, list[float]] = {}


# ============================================================================
# RUN-LEVEL DATA CLASSES
# ============================================================================


@dataclass
class RunConfig:
    """Parameters for a single (iso, pathway, endpoint) optimizer run."""

    iso: str
    pathway: str
    endpoint: float
    demand_growth_level: str = 'Medium'
    firm_cost_level: str = 'M'
    ccs_cost_level: str = 'M'
    tx_level: str = 'Medium'
    q45: str = '1'  # 45Q on by default
    geo_cost_level: str | None = None  # CAISO only
    output_root: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT)

    def __post_init__(self) -> None:
        if self.iso not in ISOS:
            raise ValueError(f"Unknown ISO {self.iso!r}; expected one of {ISOS}")
        if self.pathway not in PATHWAYS:
            raise ValueError(
                f"Unknown pathway {self.pathway!r}; expected one of {PATHWAYS}"
            )
        if not any(abs(self.endpoint - e) < 1e-9 for e in ENDPOINTS):
            raise ValueError(
                f"Endpoint {self.endpoint!r} not in approved set {ENDPOINTS}"
            )
        if self.iso == 'CAISO' and self.geo_cost_level is None:
            self.geo_cost_level = 'M'

    @property
    def endpoint_pct(self) -> float:
        """Endpoint as a CFE percentage (e.g. 0.999 → 99.9)."""
        return self.endpoint * 100.0

    @property
    def output_path(self) -> Path:
        endpoint_tag = f"{self.endpoint_pct:g}".replace('.', 'p')
        return (
            self.output_root
            / self.iso
            / f"pathway{self.pathway}_ep{endpoint_tag}.json"
        )


@dataclass
class ResourceBook:
    """Running book of deployed capacity by resource, in TWh-per-year supply."""

    clean_firm: float = 0.0  # Tranche winner (uprate/geo/nuclear/CCS summed)
    uprate_twh: float = 0.0
    geothermal_twh: float = 0.0
    nuclear_newbuild_twh: float = 0.0
    ccs_twh: float = 0.0
    solar: float = 0.0
    wind: float = 0.0
    offshore_wind: float = 0.0
    hydro: float = 0.0
    battery4: float = 0.0
    battery8: float = 0.0
    ldes: float = 0.0
    h2: float = 0.0
    new_gas_twh: float = 0.0  # Post-2025 gas added for capacity adequacy

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# ============================================================================
# PARQUET LOADERS
# ============================================================================


def _resolve_ef_paths(iso: str, threshold: float) -> list[Path]:
    """Return every parquet that contains EF rows for (iso, threshold).

    Step 2.1 occasionally splits the 75-threshold file into *_part0/_part1. This
    helper resolves all on-disk pieces deterministically.
    """
    tag = f"{threshold:g}"  # 40 → '40', 87.5 → '87.5'
    pattern_solo = DATA_STEP21 / f"step_2_1_EF_{iso}_{tag}.parquet"
    matches: list[Path] = []
    if pattern_solo.exists():
        matches.append(pattern_solo)
    # Sharded variant (parts).
    matches.extend(sorted(DATA_STEP21.glob(f"step_2_1_EF_{iso}_{tag}_part*.parquet")))
    return matches


def load_ef_mixes(iso: str, threshold: float) -> pd.DataFrame:
    """Load the Step 2.1 efficient-frontier mixes for (iso, threshold).

    Returns a DataFrame with the canonical resource-mix schema, padded to
    include optional geothermal / offshore_wind / ccs_ccgt columns as zeros
    when the ISO does not expose them (EF parquets omit unused dims).
    """
    paths = _resolve_ef_paths(iso, threshold)
    if not paths:
        raise FileNotFoundError(
            f"No EF parquet for iso={iso} threshold={threshold} under {DATA_STEP21}"
        )
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Pad optional dims so downstream code can index them unconditionally.
    optional_cols = {
        'geothermal': np.int16(0),
        'offshore_wind': np.int16(0),
        'ccs_ccgt': np.int16(0),
    }
    for col, fill in optional_cols.items():
        if col not in df.columns:
            df[col] = fill
    return df


def load_dg_costs(iso: str, threshold: float) -> pd.DataFrame:
    """Load the Step 2.2a DG parquet for (iso, threshold).

    This is the pre-computed (scenario × growth_level × target_year) cost cross-eval
    for every EF mix. The pathway optimizer uses these as a fast reference for
    SBTi-calendar costs and as a sanity check on its own cost kernel outputs.
    """
    path = DATA_STEP22 / f"step_2_2a_DG_{iso}_{threshold:g}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No DG parquet at {path}")
    return pd.read_parquet(path)


def load_fossil_mix(iso: str) -> pd.DataFrame:
    """Load EIA-930 hourly fossil-share profiles for ISO.

    Returns a DataFrame with columns: year, hour, coal_share, gas_share, oil_share.
    Averaged across all available years (2021–2025) for a representative baseline.
    """
    path = DATA_EIA930 / 'eia_fossil_mix.parquet'
    if not path.exists():
        raise FileNotFoundError(f"EIA 930 fossil mix parquet missing at {path}")
    df = pd.read_parquet(path)
    mask = df['iso'] == iso
    if not mask.any():
        raise KeyError(f"ISO {iso} not present in eia_fossil_mix.parquet")
    return df.loc[mask].reset_index(drop=True)


def load_annual_manifest(iso: str) -> pd.DataFrame:
    """Load Step 3 annual dispatch manifest for ISO (one row per archetype)."""
    path = DATA_STEP3 / f"{iso}_annual_manifest.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Annual manifest missing for {iso} at {path}")
    return pd.read_parquet(path)


def available_thresholds(iso: str) -> list[float]:
    """Return sorted list of thresholds with EF parquets on disk for an ISO."""
    if iso in _AVAILABLE_THRESHOLDS_CACHE:
        return _AVAILABLE_THRESHOLDS_CACHE[iso]
    found: set[float] = set()
    prefix = f"step_2_1_EF_{iso}_"
    for p in DATA_STEP21.glob(f"{prefix}*.parquet"):
        tail = p.stem[len(prefix):]  # '<tag>' or '<tag>_partN'
        tag = tail.split('_part', 1)[0]
        try:
            found.add(float(tag))
        except ValueError:
            continue
    result = sorted(found)
    _AVAILABLE_THRESHOLDS_CACHE[iso] = result
    return result


# ============================================================================
# NPV + DEMAND HELPERS
# ============================================================================


def npv(annual_values: np.ndarray, base_year: int = BASE_YEAR,
        rate: float = OBJECTIVE_DISCOUNT_RATE) -> float:
    """Discount an array of annual cashflows to ``base_year`` and sum.

    Convention: index 0 corresponds to ``base_year`` (no discount); index k
    corresponds to base_year + k and is discounted by (1 + rate) ** k.
    """
    arr = np.asarray(annual_values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    years_out = np.arange(arr.size, dtype=np.float64)
    discount = np.power(1.0 + rate, years_out)
    return float(np.sum(arr / discount))


def multi_rate_npv(annual_values: np.ndarray,
                    base_year: int = BASE_YEAR) -> dict[str, float]:
    """Return NPV at each of the reporting discount rates (5/7/9%)."""
    return {
        f"npv_at_{int(r * 100)}pct": npv(annual_values, base_year=base_year, rate=r)
        for r in REPORTING_DISCOUNT_RATES
    }


def demand_for_year(iso: str, year: int, growth_level: str = 'Medium') -> float:
    """Annual demand in TWh for ``iso`` at ``year`` using config growth rate."""
    base_twh = pc.REGIONAL_DEMAND_TWH.get(iso)
    if base_twh is None:
        raise KeyError(f"REGIONAL_DEMAND_TWH has no entry for {iso}")
    rate = pc.DEMAND_GROWTH_RATES[iso][growth_level]
    years_out = year - BASE_YEAR
    return float(base_twh) * (1.0 + rate) ** years_out


def growth_factor(iso: str, year: int, growth_level: str = 'Medium') -> float:
    """Multiplicative demand growth factor vs. ``BASE_YEAR``."""
    rate = pc.DEMAND_GROWTH_RATES[iso][growth_level]
    return (1.0 + rate) ** (year - BASE_YEAR)


# ============================================================================
# CLI
# ============================================================================


def _parse_endpoint(raw: str) -> float:
    val = float(raw)
    # Accept both 90 and 0.90 forms for convenience.
    if val > 1.0:
        val = val / 100.0
    return val


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog='step_2_3_pathway_optimizer',
        description='Reliability Tax deterministic pathway optimizer (2025–2050).',
    )
    ap.add_argument('--iso', required=True, choices=ISOS, help='Target ISO.')
    ap.add_argument('--pathway', required=True, choices=PATHWAYS,
                    help='Pathway ID: 1 | 2a | 2b | 3.')
    ap.add_argument('--endpoint', required=True, type=_parse_endpoint,
                    help='CFE endpoint (0.90 | 0.95 | 0.975 | 0.99 | 0.999, or 90/95/...).')
    ap.add_argument('--growth', default='Medium', choices=pc.DEMAND_GROWTH_LEVELS,
                    help='Demand growth level (default Medium).')
    ap.add_argument('--firm', default='M', choices=('L', 'M', 'H'),
                    help='Firm-cost sensitivity level.')
    ap.add_argument('--ccs', default='M', choices=('L', 'M', 'H'),
                    help='CCS-cost sensitivity level.')
    ap.add_argument('--tx', default='Medium', choices=('None', 'Low', 'Medium', 'High'),
                    help='Transmission adder level.')
    ap.add_argument('--q45', default='1', choices=('0', '1'),
                    help='45Q credit toggle (1=on, 0=off).')
    ap.add_argument('--geo', default=None, choices=(None, 'L', 'M', 'H'),
                    help='Geothermal-cost level (CAISO only).')
    ap.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT),
                    help='Root directory for per-run JSON output.')
    return ap


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        iso=args.iso,
        pathway=args.pathway,
        endpoint=args.endpoint,
        demand_growth_level=args.growth,
        firm_cost_level=args.firm,
        ccs_cost_level=args.ccs,
        tx_level=args.tx,
        q45=args.q45,
        geo_cost_level=args.geo,
        output_root=Path(args.output_root),
    )


# ============================================================================
# OPTIMIZER ENTRY POINT (scaffold — filled in later chunks)
# ============================================================================


def run_pathway(config: RunConfig) -> dict[str, Any]:
    """Execute a single (iso, pathway, endpoint) optimization run.

    Chunk 1 scaffold: validates config, pre-loads reference parquets, and
    returns a placeholder result. Chunks 2–5 will add the deterministic cost
    kernel, pathway strategies, endogenous retirement, and the stranding
    ledger.
    """
    if du is None:
        raise RuntimeError(
            f"dispatch_utils import failed: {_DU_IMPORT_ERROR!r}. "
            "This is required for Card L retirement."
        )
    if s6 is None:
        raise RuntimeError(
            f"step6_1_smartargets import failed: {_S6_IMPORT_ERROR!r}. "
            "This is required for Card M capacity-revenue netting."
        )

    # Sanity: confirm the ISO has EF parquets on disk.
    thresholds = available_thresholds(config.iso)
    if not thresholds:
        raise FileNotFoundError(
            f"No Step 2.1 EF parquets found for {config.iso} under {DATA_STEP21}"
        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    placeholder = {
        'iso': config.iso,
        'pathway': config.pathway,
        'endpoint': config.endpoint,
        'status': 'scaffold',
        'note': 'Chunk 1 scaffold — cost kernel and pathway strategies not yet implemented.',
        'available_thresholds': thresholds,
        'output_path': str(config.output_path),
    }
    return placeholder


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    config = config_from_args(args)
    result = run_pathway(config)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
