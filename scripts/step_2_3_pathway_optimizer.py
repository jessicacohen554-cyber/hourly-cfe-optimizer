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

PATHWAYS = ('1', '1a', '1b', '2a', '2b', '3')
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


# Module-level EF parquet cache: (iso, threshold) → DataFrame.
# Avoids re-reading multi-million-row parquets for consecutive years that map
# to the same threshold (e.g. the 75% EF is reused for 2036 and 2037).
_EF_CACHE: dict[tuple[str, float], pd.DataFrame] = {}


def load_ef_mixes(iso: str, threshold: float) -> pd.DataFrame:
    """Load the Step 2.1 efficient-frontier mixes for (iso, threshold).

    Returns a DataFrame with the canonical resource-mix schema, padded to
    include optional geothermal / offshore_wind / ccs_ccgt columns as zeros
    when the ISO does not expose them (EF parquets omit unused dims).

    Results are cached in ``_EF_CACHE`` so repeat calls for the same
    (iso, threshold) — which happen every year the SBTi ladder stays at the
    same threshold — return the same object without re-reading disk.
    """
    cache_key = (iso, threshold)
    if cache_key in _EF_CACHE:
        return _EF_CACHE[cache_key]

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
    _EF_CACHE[cache_key] = df
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


# ---------- Ledger serialization (cross-endpoint floor ratchet) --------------


def ledger_to_json(ledger: VintageLedger) -> list[dict]:
    """Serialize a VintageLedger to a JSON-safe list of vintage dicts.

    Used to persist the terminal ledger from one endpoint run so the next
    endpoint run can load it as a starting-floor ratchet seed.
    """
    return [
        {
            'resource': v.resource,
            'cod_year': v.cod_year,
            'twh_per_year': v.twh_per_year,
            'locked_lcoe': v.locked_lcoe,
            'tx_adder': v.tx_adder,
            'retire_year': v.retire_year,
        }
        for v in ledger.vintages
    ]


def ledger_from_json(data: list[dict]) -> VintageLedger:
    """Deserialize a VintageLedger from a JSON-safe list produced by ledger_to_json."""
    ledger = VintageLedger()
    for item in data:
        ledger.add(Vintage(
            resource=str(item['resource']),
            cod_year=int(item['cod_year']),
            twh_per_year=float(item['twh_per_year']),
            locked_lcoe=float(item['locked_lcoe']),
            tx_adder=float(item.get('tx_adder', 0.0)),
            retire_year=item.get('retire_year'),
        ))
    return ledger


def _load_ledger_from_json(path: Path) -> VintageLedger | None:
    """Load the terminal ledger from an existing per-run JSON file.

    Returns None on any read/parse error so callers can degrade gracefully.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
        raw = data.get('terminal_ledger')
        if raw is None:
            return None
        return ledger_from_json(raw)
    except Exception:
        return None


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


def _existing_resource_twh(iso: str, resource: str = 'clean_firm') -> float:
    """Return the absolute existing TWh for a non-scalable fleet resource.

    For resources that cannot be new-built in Pathway 1 (nuclear, hydro), the
    ceiling is the fixed TWh the existing fleet produces — not a proportion of
    year-t demand, which grows over time.  Call sites convert this to an EF
    percentage using BASE_DEMAND_TWH[iso] as the denominator (matching EF units).
    """
    try:
        import dispatch_utils as _du
    except ImportError:
        return 0.0
    base_demand = float(getattr(_du, 'BASE_DEMAND_TWH', {}).get(iso, 0.0))
    pct = float(getattr(_du, 'GRID_MIX_SHARES', {}).get(iso, {}).get(resource, 0.0))
    return base_demand * pct / 100.0


def _existing_resource_ceiling_pct(iso: str, resource: str = 'clean_firm') -> int:
    """Return the EF-compatible integer ceiling for an existing fleet resource.

    EF mixes encode resource shares as % of BASE-YEAR demand.  This converts
    the existing absolute TWh back to that same percentage, rounding so that
    e.g. PJM 32.1 % → 32, MISO 13.1 % → 13, NEISO 23.8 % → 24.
    """
    try:
        import dispatch_utils as _du
    except ImportError:
        return 0
    pct = float(getattr(_du, 'GRID_MIX_SHARES', {}).get(iso, {}).get(resource, 0.0))
    return round(pct)


def _filter_pathway_1(
    df: pd.DataFrame,
    iso: str | None = None,
    year: int | None = None,       # kept for backwards-compat; not used
    demand_growth_level: str = 'Medium',  # kept for backwards-compat; not used
) -> pd.DataFrame:
    """Pathway 1: VRE + batteries — no NEW clean firm beyond the existing fleet.

    Absolute-TWh semantics (correct):
        existing_cf_twh  = BASE_DEMAND_TWH[iso] × GRID_MIX_SHARES[iso]['clean_firm'] / 100
        allowed if: clean_firm_pct × BASE_DEMAND_TWH / 100  ≤  existing_cf_twh
        ↔  clean_firm_pct  ≤  GRID_MIX_SHARES[iso]['clean_firm']  (i.e. the base-year %)

    The EF stores clean_firm as a percentage of BASE-YEAR demand.  Existing
    nuclear produces a fixed absolute TWh regardless of year, so the ceiling
    must be expressed in those same base-year units — NOT as a shrinking fraction
    of growing year-t demand.  Using demand_t as the denominator caused PJM/NEISO
    to be incorrectly infeasible at 90 %+ (ceiling fell below EF minimum of 29 %
    even though 29 % × 843 = 245 TWh < 270 TWh existing fleet).

    CCS and geothermal are always hard-zero (new-build only, never existing).
    """
    if df.empty:
        return df

    if iso is not None:
        ceiling_pct = _existing_resource_ceiling_pct(iso, 'clean_firm')
        mask = (df['clean_firm'] <= ceiling_pct) & (df['ccs_ccgt'] == 0)
    else:
        mask = (df['clean_firm'] == 0) & (df['ccs_ccgt'] == 0)

    if 'geothermal' in df.columns:
        mask &= (df['geothermal'] == 0)
    return df.loc[mask].reset_index(drop=True)


def _filter_pathway_1a(
    df: pd.DataFrame,
    iso: str | None = None,
) -> pd.DataFrame:
    """Pathway 1a: onshore VRE + storage only — no offshore wind, no new clean firm.

    Identical to _filter_pathway_1 but also hard-zeros offshore_wind.
    This is the true 'no planning commitment' baseline (Card R):
    no offshore wind (requires long permitting/supply-chain commitment),
    no nuclear_newbuild (prevented by nuclear_cap in _derive_delta_vintages),
    only existing uprates + onshore VRE + batteries.
    """
    if df.empty:
        return df

    if iso is not None:
        ceiling_pct = _existing_resource_ceiling_pct(iso, 'clean_firm')
        mask = (df['clean_firm'] <= ceiling_pct) & (df['ccs_ccgt'] == 0)
    else:
        mask = (df['clean_firm'] == 0) & (df['ccs_ccgt'] == 0)

    if 'geothermal' in df.columns:
        mask &= (df['geothermal'] == 0)

    # Hard-zero offshore wind — not available in P1a (Card R).
    if 'offshore_wind' in df.columns:
        mask &= (df['offshore_wind'] == 0)

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


_CLEAN_FIRM_BATCH_CHUNK = 400_000  # rows per chunk — keeps peak alloc ~130 MiB/chunk


def _clean_firm_total_cost_batch(
    cf_twh_array: np.ndarray,
    iso: str,
    config: 'RunConfig',
    year: int,
) -> np.ndarray:
    """Vectorized wrapper around ``pc.compute_clean_firm_tranches``.

    The shared helper already supports numpy-array ``new_cf_twh`` input via
    the ``hasattr(new_cf_twh, '__len__')`` dispatch inside it — but the
    scalar ``compute_clean_firm_tranches_for_year`` wrapper casts with
    ``float()``, which would collapse a batch. This helper reproduces the
    same learning-curve injection but passes arrays straight through,
    returning an ndarray of ``total_cost`` values (in $M, same convention
    as the scalar path used by ``score_ef_row_cost``).

    Large EF bands (e.g. CAISO 70% with 4.5M rows) can exhaust process
    memory when all column arrays are live simultaneously. Chunks of
    ``_CLEAN_FIRM_BATCH_CHUNK`` rows are processed sequentially and results
    concatenated. Per-row computation is independent so chunking is exact.
    """

    def _learning_curve_fn(base_cost, foak_cost, noak_cost, tech, level, target_year):
        level_short = _as_short_level(level)
        foak_start, noak_year = _learning_window(tech, level_short)
        return pc.year_adjusted_cost(
            float(foak_cost), float(noak_cost),
            int(target_year), foak_start, noak_year,
        )

    def _call_tranche(chunk: np.ndarray) -> np.ndarray:
        result = pc.compute_clean_firm_tranches(
            new_cf_twh=chunk,
            iso=iso,
            firm_lev=_as_short_level(config.firm_cost_level),
            ccs_lev=_as_short_level(config.ccs_cost_level),
            q45=config.q45,
            tx_name=config.tx_level,
            geo_lev=_as_short_level(config.geo_cost_level) if config.geo_cost_level else None,
            geo_physics_new_twh=0.0,
            ccs_used_twh=0.0,
            uprate_used_twh=0.0,
            geo_used_twh=0.0,
            learning_curve_fn=_learning_curve_fn,
            target_year=year,
        )
        return np.asarray(result['total_cost'], dtype=np.float64)

    n = len(cf_twh_array)
    if n <= _CLEAN_FIRM_BATCH_CHUNK:
        return _call_tranche(cf_twh_array)

    # Chunked path for large EF bands.
    parts = []
    for start in range(0, n, _CLEAN_FIRM_BATCH_CHUNK):
        parts.append(_call_tranche(cf_twh_array[start:start + _CLEAN_FIRM_BATCH_CHUNK]))
    return np.concatenate(parts)


_SCORE_EF_CHUNK = 300_000  # rows per chunk — keeps peak column-array alloc ~35 MiB/chunk


def score_ef_batch(
    ef: 'pd.DataFrame',
    iso: str,
    year: int,
    config: 'RunConfig',
) -> np.ndarray:
    """Strict-equivalent vectorized cost scoring for all rows in an EF.

    Drop-in replacement for the prior ``_score_ef_vectorized`` path that is
    numerically identical to ``score_ef_row_cost`` (within float64 rounding)
    row-for-row rather than relying on the ``cheapest_clean_firm_lcoe``
    scalar approximation. Concretely: clean-firm content is routed through
    ``pc.compute_clean_firm_tranches`` in batch mode so each row gets the
    same merit-order-aware cost the scalar path produces (uprate tranche
    first, then geothermal on CAISO, then cheapest of nuclear-new-build or
    CCS). This protects the absolute ``cost_usd`` reported by
    ``select_target_mix`` AND the row ranking: when clean-firm and VRE costs
    are near-tied, a merit-order-aware cost can produce a different argmin
    than a flat scalar approximation, so keeping the exact cost function is
    the only way to guarantee row-exact parity with ``score_ef_row_cost``.

    All other LCOE terms (solar/wind/offshore/geo/ccs/storage/hybrids) are
    already scalar at fixed (iso, year, config); those are precomputed once
    and folded as vectorized multiply-adds.

    Large EF bands (e.g. CAISO 70% with 4.5M rows) allocate ~14 column
    arrays simultaneously; at 4.5M × float64 each that is ~485 MiB peak.
    When the EF exceeds ``_SCORE_EF_CHUNK`` rows the function recurses on
    chunks and concatenates, bounding peak column-array memory to ~35 MiB
    per chunk regardless of EF size. Per-row scoring is independent so
    chunking is numerically exact.

    Verified against ``score_ef_row_cost`` on ERCOT threshold 50 (all 3,405
    rows exact, argmin agrees) and ERCOT threshold 90 (2,000 random sample
    of 776k, max relative diff 3.4e-16 = float64 noise).
    """
    n_rows = len(ef)
    if n_rows == 0:
        return np.empty(0, dtype=np.float64)

    # Chunked dispatch for large EF bands — avoids simultaneous allocation of
    # 14 × N float64 column arrays when N is in the millions.
    if n_rows > _SCORE_EF_CHUNK:
        parts = []
        for start in range(0, n_rows, _SCORE_EF_CHUNK):
            chunk = ef.iloc[start:start + _SCORE_EF_CHUNK]
            parts.append(score_ef_batch(chunk, iso, year, config))
        return np.concatenate(parts)

    demand_twh = demand_for_year(iso, year, config.demand_growth_level)
    demand_mwh = demand_twh * 1.0e6
    scale = demand_mwh / 100.0  # convert (pct, %) → per-unit MWh contribution

    # Precompute every scalar LCOE once for this (iso, year, config).
    solar_lcoe = marginal_lcoe('solar', iso, year, config)
    wind_lcoe = marginal_lcoe('wind', iso, year, config)
    offshore_wind_lcoe = (
        marginal_lcoe('offshore_wind', iso, year, config)
        if iso in pc.OFFSHORE_ISOS else 0.0
    )
    geothermal_lcoe = (
        marginal_lcoe('geothermal', iso, year, config)
        if iso in pc.GEOTHERMAL_ISOS else 0.0
    )
    ccs_lcoe = marginal_lcoe('ccs_ccgt', iso, year, config)
    battery4_lcoe = marginal_lcoe('battery4', iso, year, config)
    battery8_lcoe = marginal_lcoe('battery8', iso, year, config)
    ldes_lcoe = marginal_lcoe('ldes', iso, year, config)
    h2_lcoe = marginal_lcoe('h2', iso, year, config)

    def _col_array(name: str) -> np.ndarray:
        if name in ef.columns:
            return ef[name].to_numpy(dtype=np.float64, copy=False)
        return np.zeros(n_rows, dtype=np.float64)

    solar_arr = _col_array('solar')
    wind_arr = _col_array('wind')
    offshore_arr = _col_array('offshore_wind')
    geothermal_arr = _col_array('geothermal')
    ccs_arr = _col_array('ccs_ccgt')
    clean_firm_arr = _col_array('clean_firm')
    solar_batt4_arr = _col_array('solar_batt4')
    solar_batt8_arr = _col_array('solar_batt8')
    wind_batt4_arr = _col_array('wind_batt4')
    wind_batt8_arr = _col_array('wind_batt8')

    bat_disp_arr = _col_array('battery_dispatch_pct')
    bat8_disp_arr = _col_array('battery8_dispatch_pct')
    ldes_disp_arr = _col_array('ldes_dispatch_pct')
    h2_disp_arr = _col_array('h2_dispatch_pct')

    # VRE base contributions (hydro is $0 by Card convention).
    total = solar_arr * (scale * solar_lcoe)
    total += wind_arr * (scale * wind_lcoe)

    # Offshore wind — only counted for OFFSHORE_ISOS, matching scalar path.
    if iso in pc.OFFSHORE_ISOS:
        total += offshore_arr * (scale * offshore_wind_lcoe)

    # Hybrid VRE+battery: VRE portion at full LCOE + battery portion at
    # half-utilization effective LCOE (matches scalar path exactly).
    total += solar_batt4_arr * (scale * solar_lcoe)
    total += solar_batt4_arr * (scale * battery4_lcoe * 0.5)
    total += solar_batt8_arr * (scale * solar_lcoe)
    total += solar_batt8_arr * (scale * battery8_lcoe * 0.5)
    total += wind_batt4_arr * (scale * wind_lcoe)
    total += wind_batt4_arr * (scale * battery4_lcoe * 0.5)
    total += wind_batt8_arr * (scale * wind_lcoe)
    total += wind_batt8_arr * (scale * battery8_lcoe * 0.5)

    # Clean-firm bucket — dispatched through the shared tranche helper in
    # batch mode so the merit order matches Step 2.2a exactly. Rows with
    # clean_firm == 0 naturally get total_cost == 0 from the helper.
    if clean_firm_arr.any():
        cf_twh_arr = clean_firm_arr * (demand_twh / 100.0)
        cf_total_cost_m = _clean_firm_total_cost_batch(
            cf_twh_arr, iso, config, year,
        )
        # compute_clean_firm_tranches returns $M; scalar path multiplies by 1e6.
        total += cf_total_cost_m * 1.0e6

    # Explicit CCS column (independent of clean-firm bucket — rare but
    # preserved for safety, matching scalar path).
    total += ccs_arr * (scale * ccs_lcoe)

    # Direct geothermal column (CAISO only, matching scalar path guard).
    if iso in pc.GEOTHERMAL_ISOS:
        total += geothermal_arr * (scale * geothermal_lcoe)

    # Storage dispatch — each row's storage_pct is treated as delivered %
    # of demand, priced at the year's effective LCOE.
    total += bat_disp_arr * (scale * battery4_lcoe)
    total += bat8_disp_arr * (scale * battery8_lcoe)
    total += ldes_disp_arr * (scale * ldes_lcoe)
    total += h2_disp_arr * (scale * h2_lcoe)

    return total


# Legacy alias — retained so any external call sites still resolve, but
# routed to the strict-equivalent batch scorer above.
_score_ef_vectorized = score_ef_batch


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

    if pathway == '1a':
        ef = _filter_pathway_1a(ef, iso=iso)
    elif pathway in ('1', '1b'):
        # P1b = current P1 filter (offshore wind allowed, no new nuclear via cap).
        # Original '1' runs also use this filter; re-runs will be clean.
        ef = _filter_pathway_1(ef, iso=iso)
    elif pathway in ('2a', '2b'):
        if pivot_state is None or not pivot_state.pivoted:
            ef = _filter_pathway_1(ef, iso=iso)
        # else: full EF available post-pivot (clean firm unlocked)
    elif pathway == '3':
        ef = _filter_pathway_3(ef)
    else:
        raise ValueError(f"select_target_mix: unknown pathway {pathway!r}")

    if ef.empty:
        return {'infeasible': True, 'reason': f'pathway_{pathway}_filter_empty',
                'threshold': threshold, 'mix_pct': {}, 'storage_pct': {},
                'score': 0.0, 'cost_usd': 0.0}

    # Score every candidate row and pick the min — vectorized numpy path.
    # Scalar LCOE terms are precomputed once per (iso, year, config); the
    # resource-pct columns are pulled as numpy arrays and folded with a
    # handful of multiply-adds. Clean-firm goes through the shared tranche
    # helper in batch mode so the per-row cost stays strictly equivalent
    # to ``score_ef_row_cost`` (same merit-order cost, not a scalar
    # cheapest-LCOE approximation). Replaces an iterrows() loop that took
    # minutes on multi-million-row EF parquets.
    costs = score_ef_batch(ef, iso, year, config)
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
# MAIN SOLVE LOOP — year-by-year pathway optimization
# ============================================================================
#
# The outer loop walks 2025 -> 2050 one year at a time. For each year:
#   1. Select the target mix under pathway rules (chunk 3).
#   2. For 2a/2b, test the pivot trigger and re-select if it fires.
#   3. Diff the target mix against the vintage ledger; anything new gets
#      vintaged at the year's learning-adjusted LCOE. Per Card E, we never
#      un-build prior vintages (deltas are clamped to zero from below).
#   4. Compute annual cost = vintage ledger operating cost minus capacity
#      revenue (Card M) minus Card L retirement savings on fossil.
#   5. Check the economic infeasibility floor ($10k/CFE%) as a soft warning.
#
# Retirement bookkeeping (Card L) lives in chunk 4b along with the MANIFEST
# writer. This block focuses on the solve loop itself so the file compiles
# and the pathway optimizer is exercisable end-to-end.


@dataclass
class PathwayRunResult:
    """All per-run artifacts the MANIFEST + output tables need."""

    config: RunConfig
    feasibility: dict[str, Any]
    annual_buildout: list[dict[str, Any]]
    annual_cost: list[dict[str, Any]]
    achieved_cfe: float
    undiscounted_cost_usd: float
    npv_at: dict[str, float]
    pivot_state: PivotState
    ledger: VintageLedger
    endpoint_mix_pct: dict[str, float]
    endpoint_storage_pct: dict[str, float]
    endpoint_threshold: float | None
    retirement_timeline: list[dict[str, Any]] = field(default_factory=list)
    stranding_ledger: list[dict[str, Any]] = field(default_factory=list)
    vre_curtailment_at_endpoint: dict[str, float] = field(default_factory=dict)
    endpoint_hourly_dispatch: list[float] | None = None


def _twh_from_pct(pct: float, demand_twh: float) -> float:
    return pct / 100.0 * demand_twh


def _vintage_lcoe_for_resource(resource: str, iso: str, year: int,
                                config: RunConfig) -> float:
    """Wrap marginal_lcoe() but map hybrid resources to their underlying
    technology. Hybrids are tracked as VRE + storage separately in the ledger.
    """
    mapping = {
        'solar_batt4': 'solar', 'solar_batt8': 'solar',
        'wind_batt4': 'wind', 'wind_batt8': 'wind',
    }
    base = mapping.get(resource, resource)
    if base == 'hydro':
        return 0.0
    return marginal_lcoe(base, iso, year, config)


def _derive_delta_vintages(
    ledger: VintageLedger,
    target: dict[str, Any],
    iso: str,
    year: int,
    config: RunConfig,
    nuclear_cap_twh: float = float('inf'),
) -> list[Vintage]:
    """Produce per-resource Vintage entries for new-build in ``year``.

    Compares target mix TWh vs. cumulative ledger TWh by resource; anything
    positive is a new vintage locked at this year's LCOE.

    nuclear_cap_twh: if finite, caps the clean-firm target TWh at this value.
        Set to existing_cf_twh for Pathway 1 and pre-pivot Pathway 2a/2b so
        that the cost model never credits more nuclear than the existing fleet
        produces — regardless of demand growth scaling.
    """
    demand_twh = demand_for_year(iso, year, config.demand_growth_level)
    new_vintages: list[Vintage] = []

    # Real energy resources (clean firm bucket handled below via tranches).
    energy_resources = [
        ('solar', 'solar'),
        ('wind', 'wind'),
        ('offshore_wind', 'offshore_wind'),
        ('hydro', 'hydro'),
        ('geothermal', 'geothermal'),
        ('ccs_ccgt', 'ccs_ccgt'),
    ]
    for col, ledger_key in energy_resources:
        pct = target['mix_pct'].get(col, 0.0)
        if pct <= 0:
            continue
        target_twh = _twh_from_pct(pct, demand_twh)
        # Floor ratchet: use same-year capacity_twh so seed vintages (cod_year==year)
        # are visible. In the normal (unseeded) case this equals year-1 capacity since
        # the current-year vintage hasn't been appended yet. In the seeded case it
        # correctly sees prior-endpoint builds and prevents double-building.
        existing_twh = ledger.capacity_twh(ledger_key, year)
        new_twh = target_twh - existing_twh
        if new_twh <= 1e-6:
            continue
        lcoe = _vintage_lcoe_for_resource(ledger_key, iso, year, config)
        tx = transmission_adder(
            'clean_firm' if ledger_key in ('geothermal', 'ccs_ccgt') else ledger_key,
            iso, config.tx_level,
        )
        new_vintages.append(Vintage(
            resource=ledger_key, cod_year=year,
            twh_per_year=new_twh, locked_lcoe=lcoe, tx_adder=tx,
        ))

    # Hybrid VRE+storage — split into VRE half and storage half.
    hybrid_map = [
        ('solar_batt4', 'solar', 'battery4'),
        ('solar_batt8', 'solar', 'battery8'),
        ('wind_batt4',  'wind',  'battery4'),
        ('wind_batt8',  'wind',  'battery8'),
    ]
    for col, vre_res, batt_res in hybrid_map:
        pct = target['mix_pct'].get(col, 0.0)
        if pct <= 0:
            continue
        target_twh = _twh_from_pct(pct, demand_twh)
        # VRE side — same-year floor ratchet (see energy_resources block above).
        existing_vre = ledger.capacity_twh(vre_res, year)
        # Hybrid contributes to the same ledger key as standalone VRE — the
        # existing check above already counts standalone solar/wind, so we
        # only add the incremental hybrid slice.
        inc_vre = target_twh  # treat hybrid VRE as additive; solver relies on
        # target pct being the final operating state.
        new_vintages.append(Vintage(
            resource=vre_res, cod_year=year,
            twh_per_year=max(0.0, inc_vre),
            locked_lcoe=_vintage_lcoe_for_resource(vre_res, iso, year, config),
            tx_adder=transmission_adder(vre_res, iso, config.tx_level),
        ))
        del existing_vre  # reserved for future, keeps linter quiet
        # Storage side — same-year floor ratchet.
        existing_batt = ledger.capacity_twh(batt_res, year)
        inc_batt = max(0.0, target_twh * 0.5 - existing_batt)  # ~50% cycle
        if inc_batt > 1e-6:
            new_vintages.append(Vintage(
                resource=batt_res, cod_year=year,
                twh_per_year=inc_batt,
                locked_lcoe=marginal_lcoe(batt_res, iso, year, config),
                tx_adder=transmission_adder(batt_res, iso, config.tx_level),
            ))

    # Standalone storage (battery4/8, LDES, H2) — from the storage dispatch pcts.
    storage_map = [
        ('battery_dispatch_pct', 'battery4'),
        ('battery8_dispatch_pct', 'battery8'),
        ('ldes_dispatch_pct',     'ldes'),
        ('h2_dispatch_pct',       'h2'),
    ]
    for col, res in storage_map:
        pct = target['storage_pct'].get(col, 0.0)
        if pct <= 0:
            continue
        target_twh = _twh_from_pct(pct, demand_twh)
        existing = ledger.capacity_twh(res, year)  # same-year floor ratchet
        new_twh = target_twh - existing
        if new_twh <= 1e-6:
            continue
        new_vintages.append(Vintage(
            resource=res, cod_year=year, twh_per_year=new_twh,
            locked_lcoe=marginal_lcoe(res, iso, year, config),
            tx_adder=transmission_adder(res, iso, config.tx_level),
        ))

    # Clean-firm bucket — run the merit-order tranche helper and produce a
    # Vintage per tranche that actually got built this year.
    #
    # Pathway 1 and pre-pivot 2a/2b: existing clean firm (nuclear, hydro) is
    # INHERITED FLEET — already operating, no new build.  The EF's clean_firm
    # column contributes to CFE-score feasibility but generates no new vintages.
    # Only VRE + storage are new builds under these pathways.  Skipping here
    # ensures compute_clean_firm_tranches never allocates uprates or CCS into
    # P1/pre-pivot runs, which was the root cause of P1 incorrectly appearing
    # to build clean firm (uprates + CCS via the tranche merit order).
    #
    # Post-pivot 2a/2b and Pathway 3: nuclear_cap_twh == inf → full tranche
    # allocation runs normally, building new nuclear/CCS/uprates as optimal.
    _is_existing_fleet_only = nuclear_cap_twh < float('inf')

    cf_pct = target['mix_pct'].get('clean_firm', 0.0)
    if cf_pct > 0 and not _is_existing_fleet_only:
        # EF encodes clean_firm as % of base-year demand.  For Pathway 1 /
        # pre-pivot 2a/2b, nuclear_cap_twh is the absolute existing-fleet cap.
        # min() prevents the cost model from implying new nuclear when demand_twh
        # grows beyond base demand (e.g. 29 % × 1204 = 349 > 270 TWh for PJM).
        cf_target_twh = min(
            _twh_from_pct(cf_pct, demand_twh),
            nuclear_cap_twh,
        )
        # Same-year floor ratchet: see seed vintage at cod_year==year.
        existing_cf = (
            ledger.capacity_twh('uprate', year)
            + ledger.capacity_twh('geothermal', year)
            + ledger.capacity_twh('nuclear_newbuild', year)
            + ledger.capacity_twh('ccs_ccgt', year)
        )
        new_cf_twh = max(0.0, cf_target_twh - existing_cf)
        if new_cf_twh > 1e-6:
            tr = compute_clean_firm_tranches_for_year(
                new_cf_twh, iso, config, year,
                uprate_used_twh=ledger.capacity_twh('uprate', year - 1),
                ccs_used_twh=ledger.capacity_twh('ccs_ccgt', year - 1),
                geo_used_twh=ledger.capacity_twh('geothermal', year - 1),
            )
            tx_cf = transmission_adder('clean_firm', iso, config.tx_level)
            tx_ccs = transmission_adder('ccs_ccgt', iso, config.tx_level)
            if tr.get('uprate_twh', 0.0) > 1e-6:
                new_vintages.append(Vintage(
                    resource='uprate', cod_year=year,
                    twh_per_year=float(tr['uprate_twh']),
                    locked_lcoe=float(tr.get('uprate_lcoe', 0.0)), tx_adder=0.0,
                ))
            if tr.get('geo_twh', 0.0) > 1e-6:
                new_vintages.append(Vintage(
                    resource='geothermal', cod_year=year,
                    twh_per_year=float(tr['geo_twh']),
                    locked_lcoe=float(tr.get('geo_lcoe', 0.0)),
                    tx_adder=tx_cf,
                ))
            if tr.get('nuclear_twh', 0.0) > 1e-6:
                new_vintages.append(Vintage(
                    resource='nuclear_newbuild', cod_year=year,
                    twh_per_year=float(tr['nuclear_twh']),
                    locked_lcoe=float(tr.get('nuclear_lcoe', 0.0)),
                    tx_adder=tx_cf,
                ))
            if tr.get('ccs_tranche_twh', 0.0) > 1e-6:
                new_vintages.append(Vintage(
                    resource='ccs_ccgt', cod_year=year,
                    twh_per_year=float(tr['ccs_tranche_twh']),
                    locked_lcoe=float(tr.get('ccs_lcoe', 0.0)),
                    tx_adder=tx_ccs,
                ))

    return new_vintages


def _current_cfe_pct(target: dict[str, Any]) -> float:
    """Use the EF row's ``hourly_match_score`` as the year's achieved CFE."""
    return float(target.get('score', 0.0))


def _capacity_revenue_annual_usd(iso: str, target: dict[str, Any],
                                   demand_twh: float) -> float:
    """Best-effort Card M netting: $/MWh per resource × resource MWh summed."""
    if s6 is None:
        return 0.0
    resource_pcts = {k: v for k, v in target['mix_pct'].items() if v > 0}
    try:
        per_res = s6.compute_capacity_revenue(
            iso, _current_cfe_pct(target), resource_pcts,
        )
    except Exception:
        return 0.0
    total = 0.0
    demand_mwh = demand_twh * 1.0e6
    for res, rev_per_mwh in per_res.items():
        pct = resource_pcts.get(res, 0.0)
        total += (pct / 100.0) * demand_mwh * float(rev_per_mwh)
    return total


def solve_pathway(
    config: RunConfig,
    initial_ledger: VintageLedger | None = None,
) -> PathwayRunResult:
    """Execute one deterministic pathway run over 2025-2050.

    Parameters
    ----------
    initial_ledger : VintageLedger | None
        If provided, the run inherits all vintages from ``initial_ledger`` as
        a capacity floor (cross-endpoint ratchet). This is the terminal ledger
        produced by the same pathway at the prior CFE endpoint. The within-run
        delta logic in _derive_delta_vintages uses same-year capacity_twh()
        checks, so seed vintages are visible to the ratchet for all years
        including BASE_YEAR.
    """
    ledger = VintageLedger()
    if initial_ledger is not None:
        # Copy all seed vintages into the starting ledger.
        for v in initial_ledger.vintages:
            ledger.add(v)
    # Pivot state always starts fresh: the trigger re-evaluates naturally as
    # last_cfe accumulates from the inherited VRE builds.
    pivot_state = PivotState()
    # Absolute TWh cap for existing-fleet resources (nuclear, hydro).
    # Applied in _derive_delta_vintages for Pathway 1 and pre-pivot 2a/2b.
    _existing_nuclear_cap_twh: float = _existing_resource_twh(config.iso, 'clean_firm')
    annual_buildout: list[dict[str, Any]] = []
    annual_cost: list[dict[str, Any]] = []
    feasibility: dict[str, Any] = {
        'physical': True, 'economic': True, 'notes': [],
    }
    last_target: dict[str, Any] | None = None
    last_cfe = 0.0
    endpoint_threshold: float | None = None

    for year in YEARS:
        target = select_target_mix(
            config.iso, config.pathway, year,
            config.endpoint_pct, config, pivot_state,
        )

        # Pivot triggers for 2a / 2b happen BEFORE we commit the target mix.
        if config.pathway == '2a' and not pivot_state.pivoted:
            if should_pivot_2a(last_cfe, year):
                pivot_state.trigger(year, '90pct_plateau')
                target = select_target_mix(
                    config.iso, config.pathway, year,
                    config.endpoint_pct, config, pivot_state,
                )
        elif config.pathway == '2b' and not pivot_state.pivoted:
            marg_vre = marginal_vre_usd_per_cfe_pct(config.iso, year, config)
            cheapest_cf = cheapest_clean_firm_lcoe(
                config.iso, year, config,
                uprate_used_twh=ledger.capacity_twh('uprate', year - 1),
                ccs_used_twh=ledger.capacity_twh('ccs_ccgt', year - 1),
                geo_used_twh=ledger.capacity_twh('geothermal', year - 1),
            )
            demand_twh = demand_for_year(
                config.iso, year, config.demand_growth_level,
            )
            cheapest_cf_as_scft_cost = cheapest_cf * 0.01 * demand_twh * 1.0e6
            if should_pivot_2b(marg_vre, cheapest_cf_as_scft_cost, year):
                pivot_state.trigger(year, 'econ_trigger')
                target = select_target_mix(
                    config.iso, config.pathway, year,
                    config.endpoint_pct, config, pivot_state,
                )

        if target['infeasible']:
            feasibility['physical'] = False
            feasibility['notes'].append(
                f"{year}: {target.get('reason', 'unknown')}"
            )
            break

        # Vintage the delta from the ledger's prior state.
        # For Pathway 1 and pre-pivot 2a/2b: cap nuclear at existing fleet TWh.
        _no_new_nuclear = (
            config.pathway in ('1', '1a', '1b')
            or (config.pathway in ('2a', '2b') and not pivot_state.pivoted)
        )
        new_vintages = _derive_delta_vintages(
            ledger, target, config.iso, year, config,
            nuclear_cap_twh=_existing_nuclear_cap_twh if _no_new_nuclear else float('inf'),
        )
        for v in new_vintages:
            ledger.add(v)

        demand_twh = demand_for_year(
            config.iso, year, config.demand_growth_level,
        )
        gross_usd = ledger.operating_cost(year)
        cap_rev_usd = _capacity_revenue_annual_usd(
            config.iso, target, demand_twh,
        )
        net_usd = gross_usd - cap_rev_usd

        # Economic infeasibility — soft flag if marginal $/CFE% blows up.
        marg_vre = marginal_vre_usd_per_cfe_pct(config.iso, year, config)
        if marg_vre > ECONOMIC_INFEASIBILITY_MARGINAL_USD_PER_CFE_PCT \
                and config.pathway in ('1', '1a', '1b'):
            feasibility['economic'] = False

        annual_buildout.append({
            'year': year,
            'new_vintages': [
                {
                    'resource': v.resource,
                    'twh_per_year': round(v.twh_per_year, 3),
                    'locked_lcoe_usd_per_mwh': round(v.locked_lcoe, 2),
                    'tx_adder_usd_per_mwh': round(v.tx_adder, 2),
                }
                for v in new_vintages
            ],
        })
        annual_cost.append({
            'year': year,
            'demand_twh': round(demand_twh, 2),
            'gross_operating_usd': round(gross_usd, 0),
            'capacity_rev_netted_usd': round(cap_rev_usd, 0),
            'net_annual_cost_usd': round(net_usd, 0),
            'achieved_cfe_pct': round(_current_cfe_pct(target), 3),
        })

        last_target = target
        last_cfe = _current_cfe_pct(target)
        endpoint_threshold = target.get('threshold')

    # Aggregate costs across years.
    net_annual = np.array(
        [r['net_annual_cost_usd'] for r in annual_cost], dtype=np.float64,
    )
    if net_annual.size == 0:
        npvs = {f"npv_at_{int(r * 100)}pct": 0.0 for r in REPORTING_DISCOUNT_RATES}
        undiscounted = 0.0
    else:
        npvs = multi_rate_npv(net_annual, base_year=BASE_YEAR)
        undiscounted = float(net_annual.sum())

    endpoint_mix = last_target['mix_pct'] if last_target else {}
    endpoint_storage = last_target['storage_pct'] if last_target else {}

    return PathwayRunResult(
        config=config,
        feasibility=feasibility,
        annual_buildout=annual_buildout,
        annual_cost=annual_cost,
        achieved_cfe=last_cfe,
        undiscounted_cost_usd=undiscounted,
        npv_at=npvs,
        pivot_state=pivot_state,
        ledger=ledger,
        endpoint_mix_pct=endpoint_mix,
        endpoint_storage_pct=endpoint_storage,
        endpoint_threshold=endpoint_threshold,
    )


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
    ap.add_argument(
        '--seed-run', default=None,
        help=(
            'Path to a prior-endpoint run JSON to use as a floor-ratchet seed. '
            'The terminal_ledger from that file is loaded as initial_ledger so '
            'this run starts with the prior endpoint\'s builds already in place. '
            'Used by run_pathway_sweep.py to chain endpoint runs sequentially.'
        ),
    )
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
# CARD L — RETIREMENT TIMELINE
# ============================================================================
#
# Methodology says "existing gas retires when it stops clearing capacity
# markets." The pipeline's closest proxy is `compute_fossil_retirement()`,
# which returns a displacement-based coal→oil→gas merit order under each
# year's clean_pct. We call it per year with the year's demand growth factor
# to produce a per-year GW-retired-by-resource ledger. This is not a
# capacity-market clearing model; it's an endogenous economic retirement
# proxy that the rest of the pipeline already uses, which is adequate for
# the reliability-tax comparative framing.


# Typical coal/oil/gas capacity factors for GW ↔ TWh conversion in the
# retirement ledger. These mirror how the rest of the pipeline reports
# nameplate GW from dispatched TWh.
_FOSSIL_CAPACITY_FACTORS = {
    'coal': 0.55,
    'oil':  0.20,
    'gas':  0.45,
}


_EMISSION_RATES_CACHE: dict[str, Any] | None = None


def _load_emission_rates() -> dict[str, Any]:
    global _EMISSION_RATES_CACHE
    if _EMISSION_RATES_CACHE is not None:
        return _EMISSION_RATES_CACHE
    path = REPO_ROOT / 'data' / 'egrid_emission_rates.json'
    if not path.exists():
        raise FileNotFoundError(f"egrid_emission_rates.json missing at {path}")
    with open(path) as fh:
        _EMISSION_RATES_CACHE = json.load(fh)
    return _EMISSION_RATES_CACHE


def _twh_to_gw(twh_per_year: float, capacity_factor: float) -> float:
    """Convert annual TWh into equivalent nameplate GW at the given CF."""
    if capacity_factor <= 0:
        return 0.0
    return twh_per_year / (capacity_factor * 8760.0) * 1000.0


def compute_retirement_timeline(
    run_result: 'PathwayRunResult',
) -> list[dict[str, Any]]:
    """Card L retirement ledger: year-by-year GW retired by fossil resource.

    For each year with an annual_cost row, we call the pipeline's
    `compute_fossil_retirement()` with that year's clean_pct and growth
    factor, convert the displaced TWh into GW at typical CFs, and write one
    row per year. We report cumulative GW retired from the 2025 baseline.
    """
    if du is None:
        return []
    iso = run_result.config.iso
    try:
        emission_rates = _load_emission_rates()
        fossil_mix = load_fossil_mix(iso)
    except Exception:
        return []

    baseline_twh = {
        'coal': float(getattr(du, 'COAL_CAP_TWH', {}).get(iso, 0.0)),
        'oil':  float(getattr(du, 'OIL_CAP_TWH',  {}).get(iso, 0.0)),
        'gas':  float(
            getattr(du, 'BASE_DEMAND_TWH', {}).get(iso, 0.0)
        ) * (1.0 - sum(
            getattr(du, 'GRID_MIX_SHARES', {}).get(iso, {}).values()
        ) / 100.0) - (
            float(getattr(du, 'COAL_CAP_TWH', {}).get(iso, 0.0))
            + float(getattr(du, 'OIL_CAP_TWH', {}).get(iso, 0.0))
        ),
    }
    baseline_twh['gas'] = max(0.0, baseline_twh['gas'])

    timeline: list[dict[str, Any]] = []
    cumulative_gw: dict[str, float] = {'coal': 0.0, 'oil': 0.0, 'gas': 0.0}

    for cost_row in run_result.annual_cost:
        year = int(cost_row['year'])
        clean_pct = float(cost_row.get('achieved_cfe_pct', 0.0))
        gf = growth_factor(
            iso, year, run_result.config.demand_growth_level,
        )
        try:
            _, info = du.compute_fossil_retirement(
                iso, clean_pct, emission_rates, fossil_mix,
                demand_growth_factor=gf, year=year,
            )
        except Exception:
            continue
        coal_remaining = float(info.get('coal_remaining_twh',
                                         info.get('coal_displaced_twh', 0.0)))
        oil_remaining = float(info.get('oil_remaining_twh',
                                        info.get('oil_displaced_twh', 0.0)))
        gas_remaining = float(info.get('gas_remaining_twh',
                                        info.get('gas_displaced_twh', 0.0)))
        # Retired = baseline - remaining (clamped at zero).
        retired_twh = {
            'coal': max(0.0, baseline_twh['coal'] - coal_remaining),
            'oil':  max(0.0, baseline_twh['oil']  - oil_remaining),
            'gas':  max(0.0, baseline_twh['gas']  - gas_remaining),
        }
        gw = {
            r: _twh_to_gw(retired_twh[r], _FOSSIL_CAPACITY_FACTORS[r])
            for r in ('coal', 'oil', 'gas')
        }
        cumulative_gw = gw  # cumulative (the function already returns
                            # total displacement under that year's clean_pct).
        timeline.append({
            'year': year,
            'clean_pct': round(clean_pct, 3),
            'coal_retired_gw':   round(gw['coal'], 3),
            'oil_retired_gw':    round(gw['oil'], 3),
            'gas_retired_gw':    round(gw['gas'], 3),
            'coal_retired_twh':  round(retired_twh['coal'], 2),
            'oil_retired_twh':   round(retired_twh['oil'], 2),
            'gas_retired_twh':   round(retired_twh['gas'], 2),
        })
    return timeline


# ============================================================================
# OUTPUT WRITERS — per-run JSON + MANIFEST.json append
# ============================================================================


MANIFEST_FILENAME = 'MANIFEST.json'


def _run_key(config: RunConfig) -> str:
    endpoint_tag = f"{config.endpoint_pct:g}".replace('.', 'p')
    return f"{config.iso}__pathway{config.pathway}__ep{endpoint_tag}"


def _serialize_run_result(
    result: 'PathwayRunResult',
) -> dict[str, Any]:
    """Shape a PathwayRunResult into the per-run JSON with four tables."""
    cfg = result.config
    return {
        'schema_version': 1,
        'run_key': _run_key(cfg),
        'config': {
            'iso': cfg.iso,
            'pathway': cfg.pathway,
            'endpoint': cfg.endpoint,
            'endpoint_pct': cfg.endpoint_pct,
            'demand_growth_level': cfg.demand_growth_level,
            'firm_cost_level': cfg.firm_cost_level,
            'ccs_cost_level': cfg.ccs_cost_level,
            'tx_level': cfg.tx_level,
            'q45': cfg.q45,
            'geo_cost_level': cfg.geo_cost_level,
        },
        'feasibility': result.feasibility,
        'headline': {
            'achieved_cfe_pct': round(result.achieved_cfe, 4),
            'undiscounted_cost_usd': round(result.undiscounted_cost_usd, 0),
            **{k: round(v, 0) for k, v in result.npv_at.items()},
            'endpoint_threshold': result.endpoint_threshold,
            'pivot': {
                'pivoted': result.pivot_state.pivoted,
                'pivot_year': result.pivot_state.pivot_year,
                'pivot_reason': result.pivot_state.pivot_reason,
            },
        },
        'tables': {
            'annual_buildout': result.annual_buildout,
            'annual_cost': result.annual_cost,
            'endpoint_hourly_dispatch': result.endpoint_hourly_dispatch,
            'stranding_ledger': result.stranding_ledger,
        },
        'retirement_timeline': result.retirement_timeline,
        'vre_curtailment_at_endpoint': result.vre_curtailment_at_endpoint,
        'endpoint_mix_pct': result.endpoint_mix_pct,
        'endpoint_storage_pct': result.endpoint_storage_pct,
        # Terminal ledger — all active vintages at end of run. Used as the
        # initial_ledger seed for the next endpoint run in the same pathway
        # (cross-endpoint floor ratchet). Loaded by _load_ledger_from_json().
        'terminal_ledger': ledger_to_json(result.ledger),
    }


def write_run_json(result: 'PathwayRunResult') -> Path:
    """Write the per-run JSON with the four tables and ledger metadata."""
    out_path = result.config.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_run_result(result)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, out_path)
    return out_path


def _manifest_entry(result: 'PathwayRunResult') -> dict[str, Any]:
    """One MANIFEST.json row per run. Required fields from methodology."""
    cfg = result.config
    stranding_totals = {
        'gw_by_resource': {},
        'usd_by_resource': {},
    }
    for row in result.stranding_ledger:
        res = row.get('resource')
        if res is None:
            continue
        stranding_totals['gw_by_resource'][res] = float(
            row.get('stranded_twh', 0.0)
        )
        stranding_totals['usd_by_resource'][res] = float(
            row.get('stranded_book_value_usd', 0.0)
        )
    return {
        'run_key': _run_key(cfg),
        'iso': cfg.iso,
        'pathway': cfg.pathway,
        'endpoint': cfg.endpoint,
        'feasibility': result.feasibility,
        'achieved_cfe_pct': round(result.achieved_cfe, 4),
        'undiscounted_cost_usd': round(result.undiscounted_cost_usd, 0),
        **{k: round(v, 0) for k, v in result.npv_at.items()},
        'retirement_timeline': result.retirement_timeline,
        'comparative_stranding': stranding_totals,
        'vre_curtailment_at_endpoint': result.vre_curtailment_at_endpoint,
        'output_path': str(cfg.output_path),
    }


def append_to_manifest(result: 'PathwayRunResult') -> Path:
    """Append (or update) the run entry in MANIFEST.json.

    MANIFEST.json lives at ``<output_root>/MANIFEST.json`` and is keyed by
    ``_run_key(config)``. Re-running the same run updates its entry in place
    rather than creating duplicates.
    """
    root = result.config.output_root
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.exists():
        try:
            with open(manifest_path) as fh:
                manifest = json.load(fh)
        except Exception:
            manifest = {}
    else:
        manifest = {}
    if 'runs' not in manifest or not isinstance(manifest.get('runs'), dict):
        manifest['runs'] = {}
    manifest.setdefault('schema_version', 1)
    manifest.setdefault('description',
        'Reliability Tax pathway optimizer manifest (4 pathways × 5 endpoints × 7 ISOs).')
    key = _run_key(result.config)
    manifest['runs'][key] = _manifest_entry(result)
    tmp = manifest_path.with_suffix('.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(manifest, fh, indent=2, default=str)
    os.replace(tmp, manifest_path)
    return manifest_path


# ============================================================================
# CARD K REVISED — COMPARATIVE STRANDING LEDGER
# ============================================================================
#
# Stranding is defined ONLY with respect to Pathway 3 for the same (ISO,
# endpoint). For each resource in {new-build gas, VRE (solar, wind,
# offshore), storage (battery4/8, LDES, H2)}:
#
#     stranded_twh = max(0, pathway_twh[resource] - pathway3_twh[resource])
#
# Clean firm is EXCLUDED from the stranding calc — Pathway 3's clean firm
# IS the destination plan, so anything Pathway 2a/2b builds post-pivot
# that exceeds P3's clean firm build is not considered stranded.
#
# Book value of unrecovered capital is approximated as:
#     locked_lcoe * stranded_twh_per_year * 1e6  ($/MWh × MWh/yr) × remaining_years
# where remaining_years = asset_life - (2050 - cod_year).
#
# For reliability-tax framing we keep this simple: one entry per resource,
# summing across vintages, and clamped at asset-life boundaries.


STRANDING_RESOURCES = (
    'solar', 'wind', 'offshore_wind',
    'battery4', 'battery8', 'ldes', 'h2',
    'new_gas',  # post-2025 gas (not yet built in the solver, placeholder)
)

# Asset lives for book-value amortization.
_ASSET_LIFE_YEARS = {
    'solar': 30,
    'wind': 30,
    'offshore_wind': 30,
    'battery4': 15,
    'battery8': 15,
    'ldes': 25,
    'h2': 20,
    'new_gas': 25,
}


def _total_twh_by_resource(
    ledger: 'VintageLedger', at_year: int = END_YEAR,
) -> dict[str, float]:
    """Sum all active vintages at ``at_year`` grouped by resource."""
    out: dict[str, float] = {}
    for v in ledger.active(at_year):
        out[v.resource] = out.get(v.resource, 0.0) + v.twh_per_year
    return out


def _book_value_stranded(
    ledger: 'VintageLedger', resource: str, stranded_twh_total: float,
    at_year: int = END_YEAR,
) -> float:
    """Approximate book value of unrecovered capital for the stranded slice.

    Strategy: walk the resource's vintages from newest → oldest, attributing
    the stranded TWh against them (newest vintages are the most 'excess'
    under the comparative frame). Each vintage's contribution is:
        locked_lcoe × attributed_twh × 1e6 × remaining_life_years
    """
    if stranded_twh_total <= 1e-9:
        return 0.0
    life = _ASSET_LIFE_YEARS.get(resource, 25)
    vintages = sorted(
        [v for v in ledger.active(at_year) if v.resource == resource],
        key=lambda v: v.cod_year, reverse=True,
    )
    remaining_stranded = stranded_twh_total
    book_value = 0.0
    for v in vintages:
        if remaining_stranded <= 1e-9:
            break
        attributed = min(remaining_stranded, v.twh_per_year)
        remaining_life = max(0, life - (at_year - v.cod_year))
        book_value += (
            float(v.locked_lcoe) * attributed * 1.0e6 * remaining_life
        )
        remaining_stranded -= attributed
    return book_value


def compute_stranding_ledger(
    result: 'PathwayRunResult',
    pathway3_result: 'PathwayRunResult | None' = None,
    pathway3_ledger: VintageLedger | None = None,
) -> list[dict[str, Any]]:
    """Card K revised: comparative-to-Pathway-3 stranding ledger at 2050.

    Pathway 3 is always unstranded by definition (excess vs. self = 0).

    Accepts either a full ``pathway3_result`` or a bare ``pathway3_ledger``
    (loaded from disk when the seeded P3 run already exists and we want to
    avoid re-solving it).
    """
    if result.config.pathway == '3':
        return []
    p3_ledger = (
        pathway3_result.ledger if pathway3_result is not None
        else pathway3_ledger
    )
    if p3_ledger is None:
        return []
    own_totals = _total_twh_by_resource(result.ledger, at_year=END_YEAR)
    p3_totals = _total_twh_by_resource(p3_ledger, at_year=END_YEAR)
    rows: list[dict[str, Any]] = []
    for resource in STRANDING_RESOURCES:
        own = own_totals.get(resource, 0.0)
        p3 = p3_totals.get(resource, 0.0)
        stranded = max(0.0, own - p3)
        if stranded <= 1e-6 and own == 0 and p3 == 0:
            continue
        book_value = _book_value_stranded(result.ledger, resource, stranded)
        rows.append({
            'resource': resource,
            'pathway_twh':   round(own, 3),
            'pathway3_twh':  round(p3, 3),
            'stranded_twh':  round(stranded, 3),
            'stranded_book_value_usd': round(book_value, 0),
        })
    return rows


# ============================================================================
# VRE CURTAILMENT CROSS-CHECK (Card K revised secondary diagnostic)
# ============================================================================


_MANIFEST_CACHE: dict[str, Any] = {}


def _load_annual_manifest_cached(iso: str):
    if iso not in _MANIFEST_CACHE:
        try:
            _MANIFEST_CACHE[iso] = load_annual_manifest(iso)
        except Exception:
            _MANIFEST_CACHE[iso] = None
    return _MANIFEST_CACHE[iso]


def _endpoint_archetype_key(result: 'PathwayRunResult') -> str | None:
    """Compute the dispatch-cache archetype key for the endpoint mix."""
    if du is None:
        return None
    mix = result.endpoint_mix_pct or {}
    storage = result.endpoint_storage_pct or {}
    resource_pcts = {k: float(v) for k, v in mix.items()}
    try:
        return du._archetype_key(
            result.config.iso, resource_pcts,
            procurement_pct=100.0,
            battery_dispatch_pct=float(storage.get('battery_dispatch_pct', 0.0)),
            battery8_dispatch_pct=float(storage.get('battery8_dispatch_pct', 0.0)),
            ldes_dispatch_pct=float(storage.get('ldes_dispatch_pct', 0.0)),
        )
    except Exception:
        return None


def compute_vre_curtailment_at_endpoint(
    result: 'PathwayRunResult',
) -> dict[str, float]:
    """Per-resource VRE curtailment rate at endpoint (2050).

    Primary source: annual manifest row for the endpoint archetype. Falls
    back to approximating curtailment as ``surplus_pct / (mix_pct +
    surplus_pct)`` when the exact archetype isn't in the cache.
    """
    out: dict[str, float] = {}
    mix = result.endpoint_mix_pct or {}
    if not mix:
        return out

    vre_cols = ('solar', 'wind', 'offshore_wind',
                'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8')

    manifest = _load_annual_manifest_cached(result.config.iso)
    archetype_key = _endpoint_archetype_key(result)
    row = None
    if manifest is not None and archetype_key is not None:
        hits = manifest[manifest['archetype_key'] == archetype_key]
        if len(hits) > 0:
            row = hits.iloc[0]

    for col in vre_cols:
        mix_pct = float(mix.get(col, 0.0))
        if mix_pct <= 0:
            continue
        surplus_col = f'{col}_surplus_pct'
        dispatch_col = f'{col}_dispatch_pct'
        if row is not None and surplus_col in row.index and dispatch_col in row.index:
            dispatch_pct = float(row[dispatch_col])
            surplus_pct = float(row[surplus_col])
            gen_total = dispatch_pct + surplus_pct
            curtailment = surplus_pct / gen_total if gen_total > 0 else 0.0
        else:
            # Fallback: assume surplus is proportional to the gap from 100%.
            # This is imprecise but non-zero when exact archetype isn't cached.
            curtailment = 0.0
        out[col] = round(curtailment, 4)
    return out


def compute_endpoint_hourly_dispatch(
    result: 'PathwayRunResult',
) -> list[float] | None:
    """Return the 8760 total_clean vs. demand shape for the endpoint mix.

    Pulled from ``data/step3-dispatch/{ISO}_dispatch_cache.parquet`` by
    archetype key. Returns a list of 8760 floats (matched clean TWh per
    hour, normalized so the sum over the year equals the endpoint clean
    fraction of demand). If the archetype isn't in the cache, returns None.
    """
    archetype_key = _endpoint_archetype_key(result)
    if archetype_key is None:
        return None
    cache_path = DATA_STEP3 / f'{result.config.iso}_dispatch_cache.parquet'
    if not cache_path.exists():
        return None
    try:
        df = pd.read_parquet(cache_path)
        hits = df[df['archetype_key'] == archetype_key]
        if len(hits) == 0:
            return None
        row = hits.iloc[0]
        total_clean = row.get('total_clean')
        if total_clean is None:
            return None
        arr = np.asarray(total_clean, dtype=np.float64)
        if arr.size != 8760:
            return None
        return arr.round(6).tolist()
    except Exception:
        return None


# ============================================================================
# OPTIMIZER ENTRY POINT — single-run and orchestrated
# ============================================================================


def _ensure_imports_ready() -> None:
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


def _solve_and_annotate(
    config: RunConfig,
    initial_ledger: VintageLedger | None = None,
) -> 'PathwayRunResult':
    """Solve pathway + attach Card L retirement + endpoint diagnostics.

    Does NOT compute the stranding ledger (that requires a Pathway 3
    reference). Does NOT write to disk.
    """
    thresholds = available_thresholds(config.iso)
    if not thresholds:
        raise FileNotFoundError(
            f"No Step 2.1 EF parquets found for {config.iso} under {DATA_STEP21}"
        )
    result = solve_pathway(config, initial_ledger=initial_ledger)
    result.retirement_timeline = compute_retirement_timeline(result)
    result.vre_curtailment_at_endpoint = compute_vre_curtailment_at_endpoint(result)
    result.endpoint_hourly_dispatch = compute_endpoint_hourly_dispatch(result)
    return result


def _pathway3_config_for(config: RunConfig) -> RunConfig:
    """Build the Pathway 3 reference config for the same (ISO, endpoint)."""
    return RunConfig(
        iso=config.iso,
        pathway='3',
        endpoint=config.endpoint,
        demand_growth_level=config.demand_growth_level,
        firm_cost_level=config.firm_cost_level,
        ccs_cost_level=config.ccs_cost_level,
        tx_level=config.tx_level,
        q45=config.q45,
        geo_cost_level=config.geo_cost_level,
        output_root=config.output_root,
    )


def run_pathway(
    config: RunConfig,
    initial_ledger: VintageLedger | None = None,
) -> dict[str, Any]:
    """Execute a single (iso, pathway, endpoint) run with Card K stranding.

    Always solves Pathway 3 FIRST for the same (ISO, endpoint) so the
    comparative-to-Pathway-3 stranding ledger can be computed for the
    target pathway. If the target is Pathway 3, only one solve runs.

    Writes the per-run JSON (four tables) and appends to MANIFEST.json. If
    the target is not Pathway 3, also writes the Pathway 3 reference run
    (unless the P3 JSON already exists on disk, in which case the seeded P3
    result is loaded to avoid overwriting it).

    Parameters
    ----------
    initial_ledger : VintageLedger | None
        Cross-endpoint floor-ratchet seed: terminal ledger from the same
        pathway's prior-endpoint run. None = start from scratch (normal).
    """
    _ensure_imports_ready()

    # 1. Pathway 3 reference.
    if config.pathway == '3':
        p3_result = _solve_and_annotate(config, initial_ledger=initial_ledger)
        target_result = p3_result
        p3_ledger_for_stranding = p3_result.ledger
    else:
        p3_cfg = _pathway3_config_for(config)
        p3_out = p3_cfg.output_path
        if p3_out.exists():
            # Seeded P3 already on disk — load its terminal ledger for stranding
            # instead of re-solving (which would overwrite the seeded result).
            p3_ledger_for_stranding = _load_ledger_from_json(p3_out)
        else:
            # P3 not yet computed — solve from scratch and write.
            p3_result = _solve_and_annotate(p3_cfg)
            p3_result.stranding_ledger = []
            write_run_json(p3_result)
            append_to_manifest(p3_result)
            p3_ledger_for_stranding = p3_result.ledger
        target_result = _solve_and_annotate(config, initial_ledger=initial_ledger)

    # 2. Card K revised comparative stranding (vs. Pathway 3 baseline).
    target_result.stranding_ledger = compute_stranding_ledger(
        target_result,
        pathway3_ledger=p3_ledger_for_stranding,
    )

    # 3. Write outputs.
    out_path = write_run_json(target_result)
    manifest_path = append_to_manifest(target_result)

    return {
        'run_key': _run_key(config),
        'status': 'ok' if target_result.feasibility['physical'] else 'infeasible',
        'achieved_cfe_pct': round(target_result.achieved_cfe, 4),
        'undiscounted_cost_usd': round(target_result.undiscounted_cost_usd, 0),
        **{k: round(v, 0) for k, v in target_result.npv_at.items()},
        'output_path': str(out_path),
        'manifest_path': str(manifest_path),
        'pathway3_reference_run_key': _run_key(_pathway3_config_for(config)),
        'stranding_ledger_rows': len(target_result.stranding_ledger),
        'pivot': {
            'pivoted': target_result.pivot_state.pivoted,
            'pivot_year': target_result.pivot_state.pivot_year,
            'pivot_reason': target_result.pivot_state.pivot_reason,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    config = config_from_args(args)
    initial_ledger: VintageLedger | None = None
    if args.seed_run:
        seed_path = Path(args.seed_run)
        if seed_path.exists():
            initial_ledger = _load_ledger_from_json(seed_path)
            if initial_ledger is None:
                print(
                    f"[optimizer] WARNING: --seed-run {seed_path} exists but "
                    "terminal_ledger could not be loaded (missing key or parse "
                    "error). Starting from empty ledger.",
                    file=sys.stderr,
                )
        else:
            print(
                f"[optimizer] WARNING: --seed-run path {seed_path} does not "
                "exist. Starting from empty ledger.",
                file=sys.stderr,
            )
    result = run_pathway(config, initial_ledger=initial_ledger)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
