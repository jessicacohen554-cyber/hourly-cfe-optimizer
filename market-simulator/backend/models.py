"""
Pydantic request/response schemas for the Market Simulator API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

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


class VOM(BaseModel):
    """Variable O&M costs ($/MWh) by generator type."""
    coal_steam: float = 5.50
    gas_ccgt: float = 3.50
    gas_ct: float = 5.00
    oil_ct: float = 6.00


class StorageCosts(BaseModel):
    """Installed storage costs ($/kW-yr)."""
    battery: float = 295.0    # 4hr Li-ion
    battery8: float = 456.0   # 8hr Li-ion
    ldes: float = 220.0       # 100hr Iron-Air


class CleanLCOEs(BaseModel):
    solar: float = 65.0
    wind: float = 62.0
    offshore_wind: float = 85.0
    nuclear: float = 90.0
    ccs_ccgt: float = 120.0
    geothermal: Optional[float] = None  # CAISO only


class EmissionPrices(BaseModel):
    """Emission allowance prices ($/ton) for NOx and SOx."""
    nox: float = 0.0       # $/ton NOx
    sox: float = 0.0       # $/ton SOx


class EmissionLimits(BaseModel):
    """Hard emission rate caps (lb/MWh). Generators exceeding these must retire."""
    nox_limit: Optional[float] = None   # lb/MWh — None = no limit
    sox_limit: Optional[float] = None   # lb/MWh — None = no limit


class Incentives(BaseModel):
    """Federal/state incentive inputs."""
    ptc_wind: float = 26.0             # $/MWh Production Tax Credit for wind
    ptc_solar: float = 26.0            # $/MWh PTC for solar (post-IRA)
    ptc_nuclear_new: float = 26.0      # $/MWh 45Y new nuclear PTC
    # 45U existing nuclear — contract-for-difference floor mechanism
    ptc_45u_max: float = 15.0          # $/MWh max credit
    ptc_45u_floor: float = 40.0        # $/MWh revenue floor
    ptc_45u_floor_escalation: float = 0.0  # % annual increase on floor
    ptc_45u_sunset_year: int = 2032    # Sunset year (user can extend)
    itc_pct: float = 30.0             # % Investment Tax Credit (storage, offshore wind)
    rec_price: Optional[float] = None  # $/MWh REC price override — None = model default


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class CustomOverrides(BaseModel):
    """Flags indicating which custom input files to use."""
    fuel: bool = False
    lmp: bool = False
    capacity: bool = False
    rec: bool = False


class SimulationRequest(BaseModel):
    """Parameters for a single market simulation run."""
    iso: str = "ERCOT"
    fuel_prices: FuelPrices = Field(default_factory=FuelPrices)
    carbon_price: float = 5.50
    emission_prices: EmissionPrices = Field(default_factory=EmissionPrices)
    emission_limits: EmissionLimits = Field(default_factory=EmissionLimits)
    heat_rates: HeatRates = Field(default_factory=HeatRates)
    vom: VOM = Field(default_factory=VOM)
    clean_lcoes: CleanLCOEs = Field(default_factory=CleanLCOEs)
    incentives: Incentives = Field(default_factory=Incentives)
    storage_costs: StorageCosts = Field(default_factory=StorageCosts)
    capacity_market_price: Optional[float] = None  # None → use ISO default
    wholesale_price_override: Optional[float] = None  # $/MWh — None = model-derived
    q45: bool = True                    # 45Q tax credit on/off
    ccs_credit_override: Optional[float] = None  # $/MWh CCS credit override (replaces 45Q tables)
    transmission_level: str = "Medium"  # None/Low/Medium/High
    demand_growth: object = "Medium"    # Low/Medium/High OR float for custom %
    ppa_level: str = "Medium"           # Low/Medium/High
    gas_friction: str = "Medium"        # Low/Medium/High
    condition: str = "Facilitating"     # Facilitating/Challenging
    nuclear_retirement_threshold: float = 30.0  # $/MWh
    mode: str = "snapshot"              # snapshot / trajectory
    years: List[int] = Field(default_factory=lambda: [2030, 2035, 2040, 2045, 2050])
    # Annual granularity controls (trajectory/sweep modes)
    start_year: int = 2025              # First simulation year
    end_year: int = 2060                # Last simulation year
    year_step: int = 1                  # 1 = every year, 5 = every 5 years
    custom_overrides: CustomOverrides = Field(default_factory=CustomOverrides)

    # Fossil new-build LCOEs
    fossil_lcoes: Optional[dict] = None  # {gas_ccgt: float, gas_ct: float, coal: float}

    # Per-resource transmission overrides
    tx_overrides: Optional[dict] = None  # {solar: float, wind: float, ...} — blank = use master L/M/H


class SweepRequest(BaseModel):
    """Parameters for a full 270-scenario parametric sweep."""
    isos: List[str] = Field(default_factory=lambda: ["ERCOT"])
    nuclear_retirement_threshold: Optional[float] = 30.0
    snapshot_mode: bool = False
    # Annual granularity controls
    start_year: int = 2025
    end_year: int = 2060
    year_step: int = 5                  # Default to 5yr steps for sweeps (performance)
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


class FuelBinRow(BaseModel):
    """One row in the fuel source / heat rate bin breakdown table."""
    fuel_type: str              # e.g. "coal_steam", "gas_ccgt_efficient"
    heat_rate_bin: str          # e.g. "< 7.2", "7.2-8.0", "> 8.0"
    capacity_gw: float          # Total capacity in GW
    capacity_factor: float      # 0–1
    generation_twh: float       # Annual generation in TWh
    marginal_cost: float        # $/MWh
    avg_revenue: float = 0.0    # $/MWh
    status: str = "operating"   # operating / retiring / retired


class HourlyProfile(BaseModel):
    """Compact hourly profile — 24 representative hours or full 8760."""
    hours: List[float] = Field(default_factory=list)       # hour indices
    values: List[float] = Field(default_factory=list)      # values at each hour
    label: str = ""


class SupplyStackEntry(BaseModel):
    """One slice of the supply stack at a given year."""
    resource: str               # e.g. "solar", "wind", "gas_ccgt", "coal_steam"
    capacity_gw: float
    generation_twh: float


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
    emissions_by_fuel: Dict[str, float] = Field(default_factory=dict)  # Mt CO2 per fuel type
    nuclear_revenue: Dict[str, float] = Field(default_factory=dict)
    nuclear_retired: bool = False
    ccs_breakeven: Dict[str, float] = Field(default_factory=dict)
    # RPS compliance tracking
    rps_mandated_pct: float = 0.0
    rps_eligible_pct: float = 0.0
    rps_shortfall_pct: float = 0.0
    acp_cost_million: float = 0.0
    cumulative_acp_million: float = 0.0
    # New: detailed data for results page
    fuel_bin_table: List[FuelBinRow] = Field(default_factory=list)
    supply_stack: List[SupplyStackEntry] = Field(default_factory=list)
    lmp_hourly_profile: Optional[HourlyProfile] = None
    capacity_revenue_profile: Optional[HourlyProfile] = None


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
    sim_years: List[int] = Field(default_factory=list)  # Years actually simulated
    year_results: List[YearResult] = Field(default_factory=list)
    zones_deployed: List[ZoneDetail] = Field(default_factory=list)
    # Market-wide time series for results page
    lmp_time_series: Optional[HourlyProfile] = None
    capacity_rev_time_series: Optional[HourlyProfile] = None
    supply_stack_summary: List[SupplyStackEntry] = Field(default_factory=list)
    fuel_bin_table: List[FuelBinRow] = Field(default_factory=list)
    # Emissions by fuel type per year (for trajectory emissions chart)
    emissions_by_fuel_by_year: Optional[Dict[str, Any]] = None  # {years: [...], coal_steam: [...], gas_ccgt: [...], ...}
    # Plant-level summary
    plant_level_summary: Optional[dict] = None  # {operating: int, at_risk: int, stranded: int, total: int}
    # Chart data — threshold sweep, deployment, cost ladder, gas shift, sensitivity, CCS
    threshold_sweep: Optional[Dict[str, Any]] = None
    what_gets_built: Optional[Dict[str, float]] = None
    cost_ladder: Optional[List[dict]] = None
    gas_fleet_shift: Optional[List[dict]] = None
    sensitivity_matrix: Optional[dict] = None
    ccs_analysis: Optional[dict] = None
    # Run tracking
    run_id: Optional[str] = None
    narrative: Optional[str] = None


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
    nox_rates: Dict[str, float] = Field(default_factory=dict)
    sox_rates: Dict[str, float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    status_code: int = 500
