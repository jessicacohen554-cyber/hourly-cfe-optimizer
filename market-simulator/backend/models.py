"""
Pydantic request/response schemas for the Market Simulator API.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class LevelLMH(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"


class TransmissionLevel(str, Enum):
    none = "None"
    low = "Low"
    medium = "Medium"
    high = "High"


class ConditionType(str, Enum):
    facilitating = "Facilitating"
    challenging = "Challenging"


class SimMode(str, Enum):
    snapshot = "snapshot"
    trajectory = "trajectory"


class SweepStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Nested parameter models
# ─────────────────────────────────────────────────────────────────────────────

class FuelPrices(BaseModel):
    coal: float = 2.25
    gas: float = 3.50
    oil: float = 10.50


class HeatRates(BaseModel):
    coal_steam: float = 10.0
    gas_ccgt: float = 7.0
    gas_ct: float = 10.5
    oil_ct: float = 10.5


class CleanLCOEs(BaseModel):
    solar: float = 65.0
    wind: float = 62.0
    offshore_wind: float = 85.0
    nuclear: float = 90.0
    ccs_ccgt: float = 120.0


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    """Parameters for a single market simulation run."""
    iso: str = "ERCOT"
    fuel_prices: FuelPrices = Field(default_factory=FuelPrices)
    carbon_price: float = 5.50
    heat_rates: HeatRates = Field(default_factory=HeatRates)
    clean_lcoes: CleanLCOEs = Field(default_factory=CleanLCOEs)
    capacity_market_price: Optional[float] = None  # None → use ISO default
    transmission_level: str = "Medium"  # None/Low/Medium/High
    demand_growth: str = "Medium"       # Low/Medium/High
    ppa_level: str = "Medium"           # Low/Medium/High
    gas_friction: str = "Medium"        # Low/Medium/High
    condition: str = "Facilitating"     # Facilitating/Challenging
    nuclear_retirement_threshold: float = 30.0  # $/MWh
    mode: str = "snapshot"              # snapshot / trajectory
    years: List[int] = Field(default_factory=lambda: [2030, 2035, 2040, 2045, 2050])


class SweepRequest(BaseModel):
    """Parameters for a full 270-scenario parametric sweep."""
    isos: List[str] = Field(default_factory=lambda: ["ERCOT"])
    nuclear_retirement_threshold: Optional[float] = 30.0
    snapshot_mode: bool = False
    # Optional overrides for sweep bounds (future use)
    conditions: Optional[List[str]] = None         # e.g. ["Facilitating"]
    demand_levels: Optional[List[str]] = None      # e.g. ["Low", "Medium"]
    price_sens_keys: Optional[List[str]] = None    # e.g. ["all_low", "all_med"]
    ppa_levels: Optional[List[str]] = None
    gas_friction_levels: Optional[List[str]] = None


class SensitivityRequest(BaseModel):
    """Vary a single parameter across a range of values."""
    base_params: SimulationRequest = Field(default_factory=SimulationRequest)
    vary_param: str  # e.g. "gas_price", "carbon_price", "solar_lcoe"
    values: List[float]  # e.g. [2.0, 3.0, 4.0, 5.0, 6.0]


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────

class GeneratorEconomics(BaseModel):
    """Per-unit-type economics from merit-order dispatch."""
    unit_type: str
    capacity_mw: float
    marginal_cost: float          # $/MWh
    dispatch_hours: int           # hours dispatched in year
    capacity_factor: float        # 0–1
    avg_revenue_mwh: float        # $/MWh
    profit_mwh: float             # $/MWh (revenue − marginal cost)
    status: str                   # "profitable", "marginal", "retiring"


class NuclearRevenue(BaseModel):
    """Nuclear revenue breakdown."""
    energy_rev_mwh: float = 0.0
    capacity_rev_mwh: float = 0.0
    ptc_mwh: float = 0.0
    total_mwh: float = 0.0


class CCSBreakeven(BaseModel):
    """CCS-CCGT retrofit breakeven analysis."""
    carbon_price_breakeven: float = 0.0   # $/ton CO2
    lcoe_with_45q: float = 0.0            # $/MWh
    lcoe_without_45q: float = 0.0         # $/MWh


class ZoneDetail(BaseModel):
    """Economics for a single deployment zone (threshold band)."""
    threshold: float
    revenue: float
    cost: float
    profit: float
    new_gw: float
    avg_lmp: float
    energy_rev_mwh: float = 0.0
    capacity_rev_mwh: float = 0.0
    rec_rev_mwh: float = 0.0


class YearResult(BaseModel):
    """Results for a single ISO × year."""
    iso: str
    scenario: str
    year: int
    clean_pct: float
    demand_twh: float
    emissions_mt: float
    emission_rate_tco2_mwh: float
    cost_per_mwh: float
    revenue_per_mwh: float
    energy_rev_mwh: float = 0.0
    capacity_rev_mwh: float = 0.0
    rec_rev_mwh: float = 0.0
    avg_lmp: float
    lmp_p90: float = 0.0
    gas_built_gw: float = 0.0
    total_gas_gw: float = 0.0
    market_stop: bool = False
    resource_mix_twh: Dict[str, float] = Field(default_factory=dict)
    cumulative_gw: Dict[str, float] = Field(default_factory=dict)
    zones_deployed: List[float] = Field(default_factory=list)
    zone_details: List[ZoneDetail] = Field(default_factory=list)
    generator_economics: Dict[str, dict] = Field(default_factory=dict)
    nuclear_revenue: Dict[str, float] = Field(default_factory=dict)
    nuclear_retired: bool = False
    ccs_breakeven: Dict[str, float] = Field(default_factory=dict)


class SimulationResponse(BaseModel):
    """Full simulation response — one or more ISOs, one or more years each."""
    iso: str
    existing_clean_pct: float
    market_outcome_clean_pct: float
    avg_lmp: float
    generator_economics: List[GeneratorEconomics] = Field(default_factory=list)
    nuclear_revenue: NuclearRevenue = Field(default_factory=NuclearRevenue)
    ccs_breakeven_carbon_price: float = 0.0
    emissions_mt: float = 0.0
    demand_twh: float = 0.0
    resource_mix_twh: Dict[str, float] = Field(default_factory=dict)
    year_results: List[YearResult] = Field(default_factory=list)
    zones_deployed: List[ZoneDetail] = Field(default_factory=list)


class SweepJob(BaseModel):
    """Sweep job status and results."""
    job_id: str
    status: SweepStatus = SweepStatus.pending
    progress: float = 0.0           # 0–100
    total_scenarios: int = 0
    completed_scenarios: int = 0
    elapsed_seconds: float = 0.0
    results: Optional[Dict] = None  # Populated on completion
    error: Optional[str] = None


class SensitivityResult(BaseModel):
    """Single point in a sensitivity sweep."""
    param_value: float
    iso: str
    clean_pct: float
    avg_lmp: float
    emissions_mt: float
    cost_per_mwh: float
    revenue_per_mwh: float


class SensitivityResponse(BaseModel):
    """Results of a sensitivity analysis."""
    vary_param: str
    base_iso: str
    results: List[SensitivityResult] = Field(default_factory=list)


class ISOSummary(BaseModel):
    """Summary data for a single ISO."""
    iso: str
    demand_twh: float
    grid_mix: Dict[str, float]
    capacity_market_price: float
    installed_fossil_mw: float
    fossil_capacity_shares: Dict[str, float]


class ISODefaults(BaseModel):
    """Default parameters for a specific ISO."""
    iso: str
    demand_twh: float
    grid_mix: Dict[str, float]
    capacity_market_price: float
    installed_fossil_mw: float
    fossil_capacity_shares: Dict[str, float]
    fuel_prices: Dict[str, Dict[str, float]]
    heat_rates: Dict[str, float]
    vom: Dict[str, float]
    co2_rates: Dict[str, float]


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    status_code: int = 500
