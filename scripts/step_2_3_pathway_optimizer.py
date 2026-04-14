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
# COST MODEL — Wright's Law-adjusted LCOE per technology per year
# ============================================================================
#
# Design:
#   - Every new-build asset locks its LCOE at commercial operation date
#     (Card N — year-of-commercial-operation).
#   - LCOE at COD is a FOAK→NOAK blend driven by pipeline_config.learning_fraction.
#   - VRE (solar, onshore wind) treated as essentially mature — no FOAK bonus.
#     Cost level (Low/Medium/High) comes from LCOE_TABLES directly.
#   - Clean-firm bucket uses the shared compute_clean_firm_tranches merit order
#     (uprate → geothermal → min(nuclear new-build, CCS)). A year-aware
#     learning-curve callback is passed in so each tranche locks its own LCOE.
#   - Storage costs are handled as annualized $/MW-yr capacity prices pulled
#     from LCOE_TABLES for 2025 start and NOAK_BATTERY/NOAK_BATTERY8 (plus
#     FOAK_LDES / FOAK_H2) for terminal years.
#   - VintageLedger tracks every build so annual operating cost in a later
#     year is a vintage-weighted sum, not a single "current LCOE" snapshot.
#
# Units convention:
#   - TWh for annual energy, MW for peak capacity, USD for annual cost.
#   - 1 TWh = 1e6 MWh; $/MWh × MWh = USD.

LEVEL_LONG = {'L': 'Low', 'M': 'Medium', 'H': 'High'}
LEVEL_SHORT = {'Low': 'L', 'Medium': 'M', 'High': 'H'}


def _as_long_level(level: str) -> str:
    return LEVEL_LONG.get(level, level)


def _as_short_level(level: str) -> str:
    return LEVEL_SHORT.get(level, level)


def _learning_window(tech: str, level_short: str) -> tuple[int, int]:
    """Return (foak_start, noak_year) from pipeline_config.LEARNING_PARAMS."""
    return pc.LEARNING_PARAMS[tech][level_short]


# ---------- Clean firm -------------------------------------------------------


def nuclear_newbuild_lcoe_at_year(iso: str, firm_level: str, year: int) -> float:
    """FOAK→NOAK-adjusted new-build nuclear LCOE ($/MWh) at ``year`` in ``iso``."""
    firm_short = _as_short_level(firm_level)
    foak = float(pc.FOAK_NUCLEAR_NEWBUILD[iso])
    noak = float(pc.NUCLEAR_NEWBUILD_LCOE[firm_short][iso])
    foak_start, noak_year = _learning_window('nuclear', firm_short)
    return pc.year_adjusted_cost(foak, noak, year, foak_start, noak_year)


def ccs_lcoe_at_year(iso: str, ccs_level: str, q45: str, year: int) -> float:
    """FOAK→NOAK-adjusted CCGT+CCS LCOE ($/MWh) at ``year`` with 45Q toggle."""
    ccs_short = _as_short_level(ccs_level)
    foak_table = pc.FOAK_CCS_45Q_ON if q45 == '1' else pc.FOAK_CCS_45Q_OFF
    noak_table = pc.CCS_LCOE_45Q_ON if q45 == '1' else pc.CCS_LCOE_45Q_OFF
    foak = float(foak_table[iso])
    noak = float(noak_table[ccs_short][iso])
    foak_start, noak_year = _learning_window('ccs', ccs_short)
    return pc.year_adjusted_cost(foak, noak, year, foak_start, noak_year)


def geothermal_lcoe_at_year(iso: str, geo_level: str, year: int) -> float:
    """FOAK→NOAK-adjusted geothermal LCOE ($/MWh). CAISO only."""
    if iso not in pc.GEOTHERMAL_ISOS:
        return 0.0
    geo_short = _as_short_level(geo_level)
    foak = float(pc.FOAK_GEOTHERMAL)
    noak = float(pc.GEOTHERMAL_LCOE[geo_short])
    foak_start, noak_year = _learning_window('geo', geo_short)
    return pc.year_adjusted_cost(foak, noak, year, foak_start, noak_year)


def uprate_lcoe_at_year(firm_level: str, year: int) -> float:  # noqa: ARG001
    """Nuclear uprate LCOE ($/MWh). Flat across years — mature tech."""
    return float(pc.UPRATE_LCOE[_as_short_level(firm_level)])


# ---------- VRE --------------------------------------------------------------


def solar_lcoe_at_year(iso: str, level: str, year: int) -> float:  # noqa: ARG001
    """Solar LCOE ($/MWh). Mature, no Wright's Law bonus — config value is final."""
    return float(pc.LCOE_TABLES['solar'][_as_long_level(level)][iso])


def onshore_wind_lcoe_at_year(iso: str, level: str, year: int) -> float:  # noqa: ARG001
    return float(pc.LCOE_TABLES['wind'][_as_long_level(level)][iso])


def offshore_wind_lcoe_at_year(iso: str, level: str, year: int) -> float:
    """FOAK→NOAK-adjusted offshore wind LCOE ($/MWh).

    Uses fixed-bottom learning window for PJM/NYISO/NEISO and floating for CAISO.
    Returns 0 if the ISO is not in ``OFFSHORE_ISOS``.
    """
    if iso not in pc.OFFSHORE_ISOS:
        return 0.0
    is_floating = iso == 'CAISO'
    tech_key = 'offshore_wind_float' if is_floating else 'offshore_wind_fixed'
    level_short = _as_short_level(level)
    foak = float(pc.FOAK_OFFSHORE_WIND.get(iso, 0.0))
    noak = float(pc.NOAK_OFFSHORE_WIND[_as_long_level(level)].get(iso, 0.0))
    foak_start, noak_year = _learning_window(tech_key, level_short)
    return pc.year_adjusted_cost(foak, noak, year, foak_start, noak_year)


def hydro_lcoe_at_year(iso: str, year: int) -> float:  # noqa: ARG001
    """Hydro is always existing-only, wholesale-priced, $0 marginal cost."""
    return 0.0


# ---------- Storage ----------------------------------------------------------
#
# Storage is modelled as annualized $/MW-yr capacity cost. We convert to an
# effective $/MWh at dispatch time by dividing by (effective capacity factor ×
# 8760). For battery4 we assume daily cycle ≈ 0.25 CF, battery8 ≈ 0.35 CF,
# LDES ≈ 0.15 CF (weekly cycle), H2 ≈ 0.05 CF (seasonal/peaker).
#
# The annual cost of storage is then ``delivered_twh * effective_$_per_mwh``.

STORAGE_EFFECTIVE_CF = {
    'battery4': 0.25,
    'battery8': 0.35,
    'ldes': 0.15,
    'h2': 0.05,
}


def _battery_capacity_cost_at_year(iso: str, level: str, year: int,
                                    duration: str) -> float:
    """Annualized $/MW-yr for battery4 or battery8 at ``year``."""
    assert duration in ('battery4', 'battery8')
    level_long = _as_long_level(level)
    level_short = _as_short_level(level)
    tech_key = 'bat4' if duration == 'battery4' else 'bat8'
    start_table_key = 'battery' if duration == 'battery4' else 'battery8'
    start = float(pc.LCOE_TABLES[start_table_key][level_long][iso])
    noak_table = pc.NOAK_BATTERY if duration == 'battery4' else pc.NOAK_BATTERY8
    noak = float(noak_table[level_long][iso])
    foak_start, noak_year = _learning_window(tech_key, level_short)
    return pc.year_adjusted_cost(start, noak, year, foak_start, noak_year)


def _cap_cost_to_per_mwh(cap_cost_per_mw_yr: float, effective_cf: float) -> float:
    """Convert annualized $/MW-yr capacity cost to delivered-MWh $/MWh.

    Dividing by (CF × 8760) converts an annual capacity cost into the per-MWh
    cost of energy actually delivered at the stated effective capacity factor.
    """
    if effective_cf <= 0:
        return 0.0
    return cap_cost_per_mw_yr / (effective_cf * 8760.0)


def battery4_effective_lcoe(iso: str, level: str, year: int) -> float:
    cap_cost = _battery_capacity_cost_at_year(iso, level, year, 'battery4')
    return _cap_cost_to_per_mwh(cap_cost, STORAGE_EFFECTIVE_CF['battery4'])


def battery8_effective_lcoe(iso: str, level: str, year: int) -> float:
    cap_cost = _battery_capacity_cost_at_year(iso, level, year, 'battery8')
    return _cap_cost_to_per_mwh(cap_cost, STORAGE_EFFECTIVE_CF['battery8'])


def ldes_effective_lcoe(iso: str, level: str, year: int) -> float:
    level_long = _as_long_level(level)
    level_short = _as_short_level(level)
    start = float(pc.LCOE_TABLES['ldes'][level_long][iso])
    foak = float(pc.FOAK_LDES[iso])
    foak_start, noak_year = _learning_window('ldes', level_short)
    cap_cost = pc.year_adjusted_cost(foak, start, year, foak_start, noak_year)
    return _cap_cost_to_per_mwh(cap_cost, STORAGE_EFFECTIVE_CF['ldes'])


def h2_effective_lcoe(iso: str, level: str, year: int) -> float:
    level_long = _as_long_level(level)
    level_short = _as_short_level(level)
    start = float(pc.LCOE_TABLES['h2'][level_long][iso])
    foak = float(pc.FOAK_H2[iso])
    foak_start, noak_year = _learning_window('h2', level_short)
    cap_cost = pc.year_adjusted_cost(foak, start, year, foak_start, noak_year)
    return _cap_cost_to_per_mwh(cap_cost, STORAGE_EFFECTIVE_CF['h2'])


# ---------- Transmission -----------------------------------------------------


def transmission_adder(resource: str, iso: str, tx_level: str) -> float:
    """$/MWh transmission adder for ``resource`` in ``iso`` at the given level."""
    table = pc.TX_TABLES.get(resource)
    if table is None:
        return 0.0
    val = table.get(tx_level, 0)
    if isinstance(val, dict):
        return float(val.get(iso, 0))
    return float(val)


# ---------- Clean-firm merit order wrapper (year-aware) ----------------------


def compute_clean_firm_tranches_for_year(
    new_cf_twh: float,
    iso: str,
    config: 'RunConfig',
    year: int,
    uprate_used_twh: float = 0.0,
    ccs_used_twh: float = 0.0,
    geo_used_twh: float = 0.0,
) -> dict[str, float]:
    """Thin wrapper around ``pc.compute_clean_firm_tranches`` that injects a
    year-aware learning-curve function.

    Uses the Wright's Law windows defined in pipeline_config.LEARNING_PARAMS.
    Returns the same dict shape the shared helper produces plus ``year``.
    """

    def _learning_curve_fn(base_cost, foak_cost, noak_cost, tech, level, target_year):
        """Shim matching pc.compute_clean_firm_tranches's expected signature."""
        level_short = _as_short_level(level)
        foak_start, noak_year = _learning_window(tech, level_short)
        return pc.year_adjusted_cost(
            float(foak_cost), float(noak_cost),
            int(target_year), foak_start, noak_year,
        )

    result = pc.compute_clean_firm_tranches(
        new_cf_twh=float(new_cf_twh),
        iso=iso,
        firm_lev=_as_short_level(config.firm_cost_level),
        ccs_lev=_as_short_level(config.ccs_cost_level),
        q45=config.q45,
        tx_name=config.tx_level,
        geo_lev=_as_short_level(config.geo_cost_level) if config.geo_cost_level else None,
        geo_physics_new_twh=0.0,
        ccs_used_twh=ccs_used_twh,
        uprate_used_twh=uprate_used_twh,
        geo_used_twh=geo_used_twh,
        learning_curve_fn=_learning_curve_fn,
        target_year=year,
    )
    result['year'] = year
    return result


# ---------- Vintage ledger ---------------------------------------------------


@dataclass
class Vintage:
    """One build year's worth of a single resource with its locked LCOE."""

    resource: str
    cod_year: int
    twh_per_year: float
    locked_lcoe: float  # $/MWh, applied every year from cod_year onward
    tx_adder: float = 0.0  # $/MWh transmission, locked at COD
    retire_year: int | None = None  # None = does not retire within horizon


@dataclass
class VintageLedger:
    """Append-only record of every asset built during the optimization."""

    vintages: list[Vintage] = field(default_factory=list)

    def add(self, vintage: Vintage) -> None:
        self.vintages.append(vintage)

    def active(self, year: int) -> list[Vintage]:
        return [
            v for v in self.vintages
            if v.cod_year <= year
            and (v.retire_year is None or year < v.retire_year)
        ]

    def operating_cost(self, year: int) -> float:
        """Sum of (twh × $/MWh × 1e6) across all active vintages in ``year``.

        Assumes ``locked_lcoe`` already bundles capex amortization + fixed O&M
        + variable fuel, which is how pipeline_config's LCOE tables are
        published.
        """
        total = 0.0
        for v in self.active(year):
            unit_cost = v.locked_lcoe + v.tx_adder
            total += v.twh_per_year * 1.0e6 * unit_cost
        return total

    def capacity_twh(self, resource: str, year: int) -> float:
        return sum(v.twh_per_year for v in self.active(year)
                   if v.resource == resource)

    def all_resources(self) -> set[str]:
        return {v.resource for v in self.vintages}

    def capex_in_year(self, year: int) -> float:
        """USD of new-build that COMES ONLINE this year (one-shot capex flag).

        We treat ``locked_lcoe`` as already-amortized, so this function is a
        label for reporting only — it returns the first-year annualized cost
        of vintages with ``cod_year == year``. Chunk 4 uses this to split
        annual cost into "incremental build" vs. "operating legacy".
        """
        total = 0.0
        for v in self.vintages:
            if v.cod_year != year:
                continue
            unit_cost = v.locked_lcoe + v.tx_adder
            total += v.twh_per_year * 1.0e6 * unit_cost
        return total


# ---------- Marginal new-build cost helpers (for pivot triggers + sizing) ----


def marginal_lcoe(resource: str, iso: str, year: int, config: 'RunConfig') -> float:
    """Return the $/MWh cost of adding one more MWh of ``resource`` at ``year``.

    Clean-firm resources delegate to the tranche function (caller must supply
    current used-caps). Transmission adder is bundled in.
    """
    tx = transmission_adder(resource, iso, config.tx_level)
    if resource == 'solar':
        return solar_lcoe_at_year(iso, config.firm_cost_level, year) + tx
    if resource == 'wind':
        return onshore_wind_lcoe_at_year(iso, config.firm_cost_level, year) + tx
    if resource == 'offshore_wind':
        return offshore_wind_lcoe_at_year(iso, config.firm_cost_level, year) + tx
    if resource == 'hydro':
        return 0.0
    if resource == 'uprate':
        return uprate_lcoe_at_year(config.firm_cost_level, year) + tx
    if resource == 'nuclear_newbuild':
        return nuclear_newbuild_lcoe_at_year(iso, config.firm_cost_level, year) + tx
    if resource == 'ccs_ccgt':
        return ccs_lcoe_at_year(iso, config.ccs_cost_level, config.q45, year) + tx
    if resource == 'geothermal':
        return geothermal_lcoe_at_year(iso, config.geo_cost_level or 'M', year) + tx
    if resource == 'battery4':
        return battery4_effective_lcoe(iso, config.firm_cost_level, year) + tx
    if resource == 'battery8':
        return battery8_effective_lcoe(iso, config.firm_cost_level, year) + tx
    if resource == 'ldes':
        return ldes_effective_lcoe(iso, config.firm_cost_level, year) + tx
    if resource == 'h2':
        return h2_effective_lcoe(iso, config.firm_cost_level, year) + tx
    raise KeyError(f"marginal_lcoe: unknown resource {resource!r}")


def cheapest_clean_firm_lcoe(iso: str, year: int, config: 'RunConfig',
                              uprate_used_twh: float = 0.0,
                              ccs_used_twh: float = 0.0,
                              geo_used_twh: float = 0.0) -> float:
    """Cheapest available clean-firm marginal LCOE at (iso, year).

    Used by Pathway 2b's economic-pivot trigger (Card C).
    """
    options: list[float] = []
    if uprate_used_twh < pc.UPRATE_CAP_TWH.get(iso, 0.0):
        options.append(uprate_lcoe_at_year(config.firm_cost_level, year))
    if iso in pc.GEOTHERMAL_ISOS and geo_used_twh < pc.GEOTHERMAL_CAP_TWH:
        options.append(
            geothermal_lcoe_at_year(iso, config.geo_cost_level or 'M', year)
            + transmission_adder('clean_firm', iso, config.tx_level)
        )
    if ccs_used_twh < pc.CCS_CAP_TWH.get(iso, 0.0):
        options.append(
            ccs_lcoe_at_year(iso, config.ccs_cost_level, config.q45, year)
            + transmission_adder('ccs_ccgt', iso, config.tx_level)
        )
    # Nuclear new-build has no hard cap (governed by siting in pipeline, but the
    # optimizer treats it as always available if other tranches exhaust).
    options.append(
        nuclear_newbuild_lcoe_at_year(iso, config.firm_cost_level, year)
        + transmission_adder('clean_firm', iso, config.tx_level)
    )
    return min(options)


# ---------- Card M — capacity revenue netting --------------------------------


def capacity_revenue_for_year(iso: str, clean_pct: float,
                               resource_pcts: dict[str, float]) -> float:
    """Return the total $/year capacity-market revenue for the portfolio.

    Wraps ``step6_1_smartargets.compute_capacity_revenue`` (which returns
    $/MWh per resource) and multiplies by portfolio energy to yield an
    absolute $/year figure that can be netted against operating cost.

    NOTE: compute_capacity_revenue returns revenue per MWh of delivered
    energy for each resource — we multiply by that resource's TWh.
    """
    if s6 is None:  # pragma: no cover — validated at run_pathway entry
        return 0.0
    per_res = s6.compute_capacity_revenue(iso, clean_pct, resource_pcts)
    total = 0.0
    # resource_pcts here is "% of demand served"; the caller must supply the
    # per-resource TWh when they want absolute $. For now return a per-MWh map
    # that the annual-cost assembler converts to absolute $.
    for res, rev_per_mwh in per_res.items():
        pct = resource_pcts.get(res, 0.0)
        total += pct  # placeholder — chunk 4 will properly integrate $/MWh × MWh
        del rev_per_mwh  # silence linter; real math lives in chunk 4
    return total


# ============================================================================
# PATHWAY STRATEGIES — target-mix selection under pathway constraints
# ============================================================================
#
# Each pathway is a rule that, given (config, year, endpoint, ledger, state),
# picks the resource mix the optimizer should try to have OPERATING by the end
# of ``year``. Chunk 4 (main solve loop) diffs that target mix against the
# vintage ledger, locks LCOEs at year Y for the delta, and advances.
#
# The EF parquets produced by Step 2.1 already contain the physics-feasible
# efficient frontier per (iso, threshold) — every row is a mix that meets
# that threshold's hourly-match score. Chunk 3 picks the cheapest EF row per
# year subject to pathway availability rules.
#
# Pathways (methodology Cards + README):
#   1  — VRE + batteries only. clean_firm == 0 at all times. May become
#        physically or economically infeasible beyond 90-ish percent CFE;
#        the solver flags and stops advancing (Card J).
#   2a — Behavioural pivot. Before Pathway 1 hits the 90% plateau, uses the
#        same VRE-only logic as Pathway 1. At the plateau year the pivot
#        unlocks clean firm with a FOAK reset (Card P); from then on the
#        post-pivot trajectory is re-optimized freeze-and-forward (Card E).
#   2b — Economic pivot. Same as 2a, but the pivot trigger is marginal
#        $/CFE% of the next VRE+battery addition exceeding the cheapest
#        available clean-firm LCOE for that year and ISO (Card C).
#   3  — Proactive clean firm. Available year 1 on the standard Wright's Law
#        curve. Optimizer uses the full EF from the start.
#
# Run sequence: Pathway 3 runs FIRST for each (iso, endpoint) so Cards K/L
# comparative-stranding and capacity-retirement reference data are ready
# when pathways 1/2a/2b run.

# SBTi calendar used to schedule the endpoint ramp.
_SBTI_TARGETS: tuple[tuple[int, float], ...] = (
    (2025, 0.0),
    (2030, 50.0),
    (2035, 70.0),
    (2040, 90.0),
    (2045, 95.0),
    (2050, 99.9),
)


def sbti_target_for_year(year: int, endpoint_pct: float) -> float:
    """Linear-interpolated CFE% target for year Y, clipped to endpoint.

    Pathways 1/2a/2b follow this ladder exactly — Pathway 3 is allowed to
    front-load (it uses the same ladder but may build clean firm earlier).
    """
    if year <= _SBTI_TARGETS[0][0]:
        return 0.0
    if year >= _SBTI_TARGETS[-1][0]:
        return min(_SBTI_TARGETS[-1][1], endpoint_pct)

    for (y0, t0), (y1, t1) in zip(_SBTI_TARGETS, _SBTI_TARGETS[1:]):
        if y0 <= year <= y1:
            span = y1 - y0
            frac = 0.0 if span == 0 else (year - y0) / span
            interp = t0 + (t1 - t0) * frac
            return min(interp, endpoint_pct)
    return min(_SBTI_TARGETS[-1][1], endpoint_pct)


def first_year_reaching(endpoint_pct: float) -> int:
    """Earliest SBTi-calendar year at which CFE% >= endpoint_pct."""
    for y in range(BASE_YEAR, END_YEAR + 1):
        if sbti_target_for_year(y, endpoint_pct) >= endpoint_pct - 1e-9:
            return y
    return END_YEAR


# ---------- EF mix representation + cost scoring ----------------------------


# Columns in an EF parquet row that are interpretable as "% of demand" per
# resource. Not every ISO exposes every column (e.g. no geothermal / offshore
# wind for ERCOT) — load_ef_mixes pads missing columns with zeros so we can
# index unconditionally.
_EF_RESOURCE_COLS: tuple[str, ...] = (
    'clean_firm', 'solar', 'wind', 'hydro',
    'offshore_wind', 'geothermal', 'ccs_ccgt',
    'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8',
)
_EF_STORAGE_DISPATCH_COLS: tuple[str, ...] = (
    'battery_dispatch_pct', 'battery8_dispatch_pct',
    'ldes_dispatch_pct', 'h2_dispatch_pct',
)


def _row_as_mix_pct(row: pd.Series) -> dict[str, float]:
    """Extract a resource→% dict from an EF parquet row (always padded)."""
    return {col: float(row.get(col, 0.0)) for col in _EF_RESOURCE_COLS}


def _row_as_storage_pct(row: pd.Series) -> dict[str, float]:
    return {col: float(row.get(col, 0.0)) for col in _EF_STORAGE_DISPATCH_COLS}


def score_ef_row_cost(row: pd.Series, iso: str, year: int,
                       config: 'RunConfig') -> float:
    """Return the marginal-LCOE-weighted annual cost for an EF mix row.

    Used to rank candidate EF rows when picking the cheapest mix for a given
    (year, pathway). This is a quick scoring function — the real vintage
    accounting happens in chunk 4's main solve loop. All resources are
    priced at the YEAR's learning-adjusted marginal LCOE regardless of
    vintage (because chunk 3 is comparing candidate end-states, not
    tracking incremental builds).
    """
    demand_twh = demand_for_year(iso, year, config.demand_growth_level)
    mix = _row_as_mix_pct(row)
    storage = _row_as_storage_pct(row)

    total = 0.0

    # VRE (solar, onshore wind, hydro)
    total += mix['solar'] / 100.0 * demand_twh * 1.0e6 * \
        marginal_lcoe('solar', iso, year, config)
    total += mix['wind'] / 100.0 * demand_twh * 1.0e6 * \
        marginal_lcoe('wind', iso, year, config)
    # Hydro is $0 by Card convention — still counted in mix but not costed.

    # Offshore wind
    if mix['offshore_wind'] > 0 and iso in pc.OFFSHORE_ISOS:
        total += mix['offshore_wind'] / 100.0 * demand_twh * 1.0e6 * \
            marginal_lcoe('offshore_wind', iso, year, config)

    # Hybrid batteries — treat the VRE portion as VRE and the battery portion
    # as effective battery LCOE weighted by dispatch pct.
    for hybrid_col, base_res, batt_key in (
        ('solar_batt4', 'solar', 'battery4'),
        ('solar_batt8', 'solar', 'battery8'),
        ('wind_batt4', 'wind', 'battery4'),
        ('wind_batt8', 'wind', 'battery8'),
    ):
        hybrid_pct = mix[hybrid_col]
        if hybrid_pct <= 0:
            continue
        total += hybrid_pct / 100.0 * demand_twh * 1.0e6 * \
            marginal_lcoe(base_res, iso, year, config)
        total += hybrid_pct / 100.0 * demand_twh * 1.0e6 * \
            marginal_lcoe(batt_key, iso, year, config) * 0.5  # ~half-utilization

    # Clean-firm bucket — dispatched through the shared tranche helper so the
    # merit order matches Step 2.2a exactly.
    cf_pct = mix['clean_firm']
    if cf_pct > 0:
        cf_twh = cf_pct / 100.0 * demand_twh
        tranche = compute_clean_firm_tranches_for_year(cf_twh, iso, config, year)
        total += float(tranche.get('total_cost', 0.0)) * 1.0e6  # tranche cost is in $M

    # CCS explicit column (independent of clean-firm bucket) — rare, but
    # preserved for safety.
    if mix['ccs_ccgt'] > 0:
        total += mix['ccs_ccgt'] / 100.0 * demand_twh * 1.0e6 * \
            marginal_lcoe('ccs_ccgt', iso, year, config)

    # Geothermal direct
    if mix['geothermal'] > 0 and iso in pc.GEOTHERMAL_ISOS:
        total += mix['geothermal'] / 100.0 * demand_twh * 1.0e6 * \
            marginal_lcoe('geothermal', iso, year, config)

    # Storage — dispatch pct × demand × effective LCOE
    for col, res in (
        ('battery_dispatch_pct', 'battery4'),
        ('battery8_dispatch_pct', 'battery8'),
        ('ldes_dispatch_pct', 'ldes'),
        ('h2_dispatch_pct', 'h2'),
    ):
        pct = storage[col]
        if pct <= 0:
            continue
        total += pct / 100.0 * demand_twh * 1.0e6 * \
            marginal_lcoe(res, iso, year, config)

    return total


# ---------- Pathway availability filters ------------------------------------


def _filter_pathway_1(df: pd.DataFrame) -> pd.DataFrame:
    """Pathway 1: VRE + batteries only. clean_firm == 0 AND ccs == 0."""
    if df.empty:
        return df
    mask = (df['clean_firm'] == 0) & (df['ccs_ccgt'] == 0)
    if 'geothermal' in df.columns:
        mask &= (df['geothermal'] == 0)
    return df.loc[mask].reset_index(drop=True)


def _filter_pathway_3(df: pd.DataFrame) -> pd.DataFrame:
    """Pathway 3: full EF — no filter."""
    return df


# ---------- Pivot state for pathways 2a / 2b --------------------------------


@dataclass
class PivotState:
    """Tracks whether a 2a/2b run has pivoted to clean-firm yet."""

    pivoted: bool = False
    pivot_year: int | None = None
    pivot_reason: str | None = None  # '90pct_plateau' or 'econ_trigger'

    def trigger(self, year: int, reason: str) -> None:
        if not self.pivoted:
            self.pivoted = True
            self.pivot_year = year
            self.pivot_reason = reason


def should_pivot_2a(current_cfe_pct: float, year: int) -> bool:
    """Card E — Pathway 2a pivots once VRE-only reaches the 90% plateau.

    Implementation: pivot when the achievable VRE-only CFE is at or above the
    SBTi 90% milestone AND year >= 2040 (the SBTi 90% target year). Reading
    both signals avoids pivoting too early if a very VRE-rich ISO could hit
    90% on VRE alone before the methodology's intended pivot window.
    """
    return current_cfe_pct >= 90.0 - 1e-6 and year >= 2040


def should_pivot_2b(marginal_vre_usd_per_cfe_pct: float,
                     cheapest_clean_firm_lcoe_val: float,
                     year: int) -> bool:  # noqa: ARG001
    """Card C — Pathway 2b pivots when marginal $/CFE% of next VRE addition
    exceeds the cheapest available clean-firm LCOE for that year and ISO.

    The comparator is tech-specific LCOE at the margin (Card C). Clean-firm
    LCOE is roughly $/MWh which maps to ~$/CFE% via an ISO's demand_mwh /
    100. The helper accepts the two pre-computed scalars and returns bool.
    """
    return marginal_vre_usd_per_cfe_pct > cheapest_clean_firm_lcoe_val


# ---------- Target-mix selection per pathway --------------------------------


def _closest_available_threshold(iso: str, target_pct: float) -> float | None:
    """Pick the smallest available EF threshold that meets ``target_pct``.

    Returns None if no threshold matches (e.g. Pathway 1 at 99.9% for an ISO
    whose pure-VRE EF caps out at 90%).
    """
    thresholds = available_thresholds(iso)
    if not thresholds:
        return None
    feasible = [t for t in thresholds if t >= target_pct - 1e-6]
    if feasible:
        return min(feasible)
    return max(thresholds)  # best effort when below target — caller checks


def select_target_mix(
    iso: str,
    pathway: str,
    year: int,
    endpoint_pct: float,
    config: 'RunConfig',
    pivot_state: PivotState | None = None,
) -> dict[str, Any]:
    """Pick the cheapest EF mix that meets this pathway's rules at ``year``.

    Returns a dict with keys:
        threshold     : float — the EF band used (may exceed SBTi target)
        mix_pct       : dict[resource -> % of demand]
        storage_pct   : dict[storage_col -> % of demand]
        score         : float — hourly_match_score for the chosen row
        cost_usd      : float — marginal-LCOE-weighted annual cost at ``year``
        infeasible    : bool — True if no row meets the target
        reason        : str | None — populated on infeasibility
    """
    target = sbti_target_for_year(year, endpoint_pct)
    # Beyond the plateau year, enforce the endpoint itself so we don't coast.
    if year >= first_year_reaching(endpoint_pct):
        target = endpoint_pct

    threshold = _closest_available_threshold(iso, target)
    if threshold is None:
        return {'infeasible': True, 'reason': 'no_ef_parquets', 'threshold': None,
                'mix_pct': {}, 'storage_pct': {}, 'score': 0.0, 'cost_usd': 0.0}

    ef = load_ef_mixes(iso, threshold)

    if pathway == '1':
        ef = _filter_pathway_1(ef)
    elif pathway in ('2a', '2b'):
        if pivot_state is None or not pivot_state.pivoted:
            ef = _filter_pathway_1(ef)
        # else: full EF available post-pivot (clean firm unlocked)
    elif pathway == '3':
        ef = _filter_pathway_3(ef)
    else:
        raise ValueError(f"select_target_mix: unknown pathway {pathway!r}")

    if ef.empty:
        return {'infeasible': True, 'reason': f'pathway_{pathway}_filter_empty',
                'threshold': threshold, 'mix_pct': {}, 'storage_pct': {},
                'score': 0.0, 'cost_usd': 0.0}

    # Score every candidate row and pick the min.
    costs = np.fromiter(
        (score_ef_row_cost(row, iso, year, config) for _, row in ef.iterrows()),
        dtype=np.float64, count=len(ef),
    )
    if costs.size == 0 or not np.isfinite(costs).any():
        return {'infeasible': True, 'reason': 'no_finite_cost',
                'threshold': threshold, 'mix_pct': {}, 'storage_pct': {},
                'score': 0.0, 'cost_usd': 0.0}
    best_idx = int(np.nanargmin(costs))
    winner = ef.iloc[best_idx]

    return {
        'infeasible': False,
        'reason': None,
        'threshold': float(threshold),
        'mix_pct': _row_as_mix_pct(winner),
        'storage_pct': _row_as_storage_pct(winner),
        'score': float(winner.get('hourly_match_score', 0.0)),
        'cost_usd': float(costs[best_idx]),
    }


def marginal_vre_usd_per_cfe_pct(iso: str, year: int,
                                   config: 'RunConfig') -> float:
    """Approximate $/CFE% for the next VRE+battery addition at year Y.

    Uses solar + battery4 at the year's marginal LCOE and amortizes over the
    year's demand. The methodology's Card C comparator is tech-specific and
    this captures the dominant term — chunk 4 refines this with an actual
    EF-row delta comparison.
    """
    demand_twh = demand_for_year(iso, year, config.demand_growth_level)
    solar_cost = marginal_lcoe('solar', iso, year, config)
    bat_cost = marginal_lcoe('battery4', iso, year, config)
    # Adding 1% of demand as solar+battery4 shaves ~1% of CFE at the margin.
    # Cost = 1% * demand_mwh * (solar_lcoe + bat_lcoe * 0.5_util)
    return (solar_cost + 0.5 * bat_cost) * (0.01 * demand_twh * 1.0e6)


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
