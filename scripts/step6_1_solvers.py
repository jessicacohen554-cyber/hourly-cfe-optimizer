"""
step6_1_solvers — Market + Constrained solvers for SMARTargets V3
=================================================================

Two solvers, one ratchet:

  MarketSolver:       Merit-order profit ranking with LMP feedback.
                      Gas competes as a resource. CFE is OUTPUT.
                      Build rate capped per year (GW or % demand).

  ConstrainedSolver:  DE optimizer from step 2.3 architecture.
                      Floor ratchet as lower bounds, CFE ceiling.
                      Finds cheapest portfolio to meet emission cap.

Both share FloorRatchet: per-resource TWh that never decreases.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# --- Step 2.3 imports (scoring kernel) ---
# from step2_3_reliability_tax_adaptive import (
#     score_candidates, _score_mixes, _compute_post_storage_gaps,
#     h2_size_for_target, h2_peaker_capex_kw_yr, h2_peaker_fuel_mwh,
#     N_RESOURCES, RESOURCE_ORDER, RES_IDX, N_STORAGE, STORAGE_COLS,
#     FIRM_PATHWAYS,
# )

# --- Step 6.1 imports (revenue/cost model) ---
# from step6_1_smartargets import (
#     get_resource_lcoe, compute_energy_revenue_by_resource,
#     compute_capacity_revenue, compute_rec_revenue,
#     compute_lmp_at_threshold, estimate_new_gw_from_delta,
#     QUEUE_CAP_GW, SIM_YEARS, RESOURCE_TO_TECH,
#     wright_cost, get_effective_cumulative_gw,
#     NEW_GAS_CCGT_LCOE, NEW_GAS_CT_LCOE,
#     EXISTING_GAS_FOM_KW_YR,
# )

# --- Pipeline config ---
# from pipeline_config import (
#     RESOURCE_CAPACITY_FACTORS, GRID_MIX_SHARES,
#     REGIONAL_DEMAND_TWH, CCS_CAP_TWH, GEOTHERMAL_CAP_TWH,
#     OFFSHORE_ISOS, LCOE_TABLES,
# )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FLOOR RATCHET (shared by both solvers)
# ═══════════════════════════════════════════════════════════════════════════════

# All resources that can appear in the mix, including gas.
ALL_RESOURCES = [
    'solar', 'wind', 'offshore_wind', 'clean_firm', 'ccs_ccgt',
    'geothermal', 'hydro',
    'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8',
    'gas_ccgt', 'gas_ct',
]

STORAGE_TYPES = ['battery4', 'battery8', 'ldes']

# Which resources each pathway allows to be built NEW.
# Existing capacity of any type is always preserved via floor.
# Gas is buildable in all pathways for market trajectory.
PATHWAY_BUILDABLE = {
    'A': {'solar', 'wind', 'gas_ccgt', 'gas_ct',
          'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8'},
    'B': {'solar', 'wind', 'clean_firm', 'ccs_ccgt', 'geothermal',
          'gas_ccgt', 'gas_ct',
          'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8'},
    'C': {'solar', 'wind', 'clean_firm', 'ccs_ccgt', 'geothermal',
          'offshore_wind', 'gas_ccgt', 'gas_ct',
          'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8'},
    'D': {'solar', 'wind', 'clean_firm', 'ccs_ccgt', 'geothermal',
          'offshore_wind', 'gas_ccgt', 'gas_ct',
          'solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8'},
}


@dataclass
class FloorRatchet:
    """Per-resource TWh floor that advances monotonically.

    Can't unbuild. Once 50 TWh of solar is installed, it stays.
    When demand grows, the pct representation falls but TWh holds.
    Gas is tracked too — once built, it stays (but may be stranded).
    """

    iso: str
    pathway: str

    # Core state: TWh per resource, never decreases
    resource_twh: dict = field(default_factory=dict)

    # Storage dispatch %  (not TWh — these are dispatch fractions)
    storage_pct: dict = field(default_factory=dict)

    # Cumulative GW built per tech (for Wright's Law)
    cumulative_gw: dict = field(default_factory=dict)

    # Track gas separately for stranding accounting
    gas_built_gw: float = 0.0
    peak_gas_gw: float = 0.0
    peak_gas_year: int = 0

    @classmethod
    def from_grid_baseline(cls, iso, pathway, cumulative_gw_init=None):
        """Initialize from eGRID baseline + pipeline_config."""
        demand_twh = REGIONAL_DEMAND_TWH[iso]
        resource_twh = {}
        for res, pct in GRID_MIX_SHARES.get(iso, {}).items():
            resource_twh[res] = pct / 100.0 * demand_twh

        storage_pct = {sc: 0.0 for sc in STORAGE_TYPES}
        cumulative_gw = dict(cumulative_gw_init or WRIGHT_CUMULATIVE_GW_2025)

        return cls(
            iso=iso,
            pathway=pathway,
            resource_twh=resource_twh,
            storage_pct=storage_pct,
            cumulative_gw=cumulative_gw,
        )

    def get_floor_pcts(self, demand_twh):
        """Current floors as percentages at given demand level."""
        return {
            res: twh / demand_twh * 100.0
            for res, twh in self.resource_twh.items()
        }

    def get_buildable_resources(self):
        """Resources this pathway allows building new capacity for."""
        return PATHWAY_BUILDABLE.get(self.pathway, set())

    def get_resource_caps(self, demand_twh):
        """Physical caps (CCS geological, geo resource, offshore eligibility)."""
        caps = {}
        if float(CCS_CAP_TWH.get(self.iso, 0)) <= 0:
            caps['ccs_ccgt'] = 0.0
        else:
            caps['ccs_ccgt'] = float(CCS_CAP_TWH[self.iso]) / demand_twh * 100.0

        caps['geothermal'] = float(GEOTHERMAL_CAP_TWH) / demand_twh * 100.0
        if self.iso == 'CAISO':
            pass  # CAISO has geothermal
        else:
            caps['geothermal'] = min(caps['geothermal'], 1.0)  # minimal elsewhere

        if self.iso not in OFFSHORE_ISOS:
            caps['offshore_wind'] = 0.0

        return caps

    def advance(self, built_twh, built_storage_pct=None, built_gw=None):
        """Ratchet forward after deployment.

        Args:
            built_twh: dict {resource: TWh} — total TWh at new level (not delta)
            built_storage_pct: dict {storage: pct} — new dispatch levels
            built_gw: dict {tech: GW} — new GW for Wright's Law
        """
        for res, twh in built_twh.items():
            current = self.resource_twh.get(res, 0.0)
            self.resource_twh[res] = max(current, twh)

        if built_storage_pct:
            for sc, pct in built_storage_pct.items():
                current = self.storage_pct.get(sc, 0.0)
                self.storage_pct[sc] = max(current, pct)

        if built_gw:
            for tech, gw in built_gw.items():
                self.cumulative_gw[tech] = self.cumulative_gw.get(tech, 0) + gw

    def record_gas(self, new_gas_gw, year):
        """Track gas builds for stranding analysis."""
        self.gas_built_gw += new_gas_gw
        total = self.gas_built_gw  # simplified — actual tracks existing + new
        if total > self.peak_gas_gw:
            self.peak_gas_gw = total
            self.peak_gas_year = year


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MARKET SOLVER — merit-order with LMP feedback
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BuildRateConstraint:
    """Limits how much new generation can interconnect per period.

    Two independent caps, both must be satisfied:
      gw_per_year:  Physical interconnection throughput (GW/yr)
      pct_demand:   Prevents unrealistic overnight transformation
                    e.g., 0.08 = can build 8% of annual demand in new
                    capacity per year (% of TWh, converted to GW via CF)

    The binding constraint in practice:
      - Early years: gw_per_year binds (queue bottleneck)
      - Late years: pct_demand may bind (diminishing sites, NIMBYism)
    """
    gw_per_year: float          # from QUEUE_CAP_GW
    pct_demand_per_year: float  # e.g., 0.08 = 8% of demand TWh
    years_in_period: int        # 5 or 7

    @property
    def total_gw(self):
        return self.gw_per_year * self.years_in_period

    def effective_cap_gw(self, demand_twh, blended_cf=0.25):
        """The tighter of the two constraints, in GW."""
        gw_cap = self.total_gw
        # Convert pct_demand to GW: TWh × pct / (CF × 8760h) = GW
        demand_cap_twh = demand_twh * self.pct_demand_per_year * self.years_in_period
        demand_cap_gw = demand_cap_twh / (blended_cf * 8.760)
        return min(gw_cap, demand_cap_gw)


def compute_resource_profit(
    res, iso, year, demand_twh,
    conditions, cumulative_gw,
    hourly_lmp, supply_profiles,
    current_clean_pct,
):
    """Per-resource profit margin: revenue - cost ($/MWh).

    Revenue = profile-weighted LMP + capacity payment + REC credit.
    Cost = Wright's Law LCOE + transmission - PPA discount.

    For gas: revenue = avg LMP (baseload dispatch), cost = fuel LCOE.
    """
    demand_total_mwh = demand_twh * 1e6

    # --- COST ---
    if res in ('gas_ccgt', 'gas_ct'):
        lcoe_table = NEW_GAS_CCGT_LCOE if res == 'gas_ccgt' else NEW_GAS_CT_LCOE
        cost = lcoe_table.get(conditions['lcoe_level'], 55)
        # Gas doesn't get PPA discount or Wright's Law
    else:
        cost = get_resource_lcoe(
            res, iso, conditions['lcoe_level'], cumulative_gw,
            conditions['learning_speed'], year,
            ira_policy=conditions.get('ira_policy', 'off'),
        )
        # Transmission adder
        if res in ('solar', 'wind', 'clean_firm', 'offshore_wind'):
            cost += get_tx(res if res != 'clean_firm' else 'clean_firm',
                           conditions['tx_level'], iso)
        elif res in HYBRID_TYPES:
            cost += get_hybrid_tx(res, conditions['tx_level'], iso)

        # PPA discount
        ppa_level = conditions.get('ppa_level')
        if ppa_level:
            cost *= (1 - _get_ppa_discount(res, ppa_level, iso))

    # --- REVENUE ---
    if res in ('gas_ccgt', 'gas_ct'):
        # Gas earns avg LMP (dispatchable, sets marginal price)
        revenue = float(np.mean(hourly_lmp))
        # Gas also earns capacity payments
        cap_credit = PEAK_CAPACITY_CREDITS.get('gas_ccgt', 0.95)
        cap_price = CAPACITY_MARKET_PRICES.get(iso, 0)
        # Degrade capacity price with clean penetration
        alpha = CAPACITY_DEGRADATION_ALPHA
        cap_degraded = cap_price * max(0, 1 - alpha * current_clean_pct / 100)
        cf_gas = 0.85 if res == 'gas_ccgt' else 0.10
        cap_rev_mwh = cap_degraded * cap_credit / (cf_gas * 8760) * 1000
        revenue += cap_rev_mwh
    else:
        # Clean resource: profile-weighted LMP
        profile = supply_profiles.get(res)
        if profile is not None:
            gen_profile = np.array(profile, dtype=np.float64)
            gen_total = float(np.sum(gen_profile))
            if gen_total > 0:
                energy_rev = float(np.sum(gen_profile * hourly_lmp)) / gen_total
            else:
                energy_rev = 0
        else:
            energy_rev = float(np.mean(hourly_lmp))  # fallback

        # Capacity revenue
        cap_revs = compute_capacity_revenue(iso, current_clean_pct, {res: 1.0})
        cap_rev = cap_revs.get(res, 0)

        # REC revenue
        rec_revs = compute_rec_revenue(iso, {res: 1.0}, current_clean_pct, year)
        rec_rev = rec_revs.get(res, 0)

        revenue = energy_rev + cap_rev + rec_rev

    profit_margin = revenue - cost
    return {
        'resource': res,
        'revenue': round(revenue, 2),
        'cost': round(cost, 2),
        'profit_margin': round(profit_margin, 2),
    }


def find_saturation_gw(
    res, iso, year, demand_twh,
    conditions, cumulative_gw,
    supply_profiles, demand_norm, demand_mw_profile,
    current_mix_pcts,
    max_gw=50.0,
    n_steps=10,
):
    """Find how much of a resource to build before it self-cannibalizes.

    Binary search: add GW, recompute LMP, check if still profitable.
    Returns the GW at which marginal revenue = marginal cost.

    For gas: saturation = unserved demand (gas fills the gap, doesn't
    self-cannibalize because it's the marginal price setter).
    """
    if res in ('gas_ccgt', 'gas_ct'):
        # Gas fills unserved demand — no self-cannibalization
        clean_twh = sum(
            pct / 100.0 * demand_twh
            for r, pct in current_mix_pcts.items()
            if r not in ('gas_ccgt', 'gas_ct')
        )
        unserved_twh = max(0, demand_twh - clean_twh)
        cf = 0.85 if res == 'gas_ccgt' else 0.10
        return unserved_twh / (cf * 8.760)  # GW

    # Clean resources: binary search for saturation
    cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.25)

    lo_gw, hi_gw = 0.0, max_gw
    for _ in range(n_steps):
        mid_gw = (lo_gw + hi_gw) / 2.0
        mid_twh = mid_gw * cf * 8.760
        mid_pct = mid_twh / demand_twh * 100.0

        # Hypothetical mix with this much added
        test_mix = dict(current_mix_pcts)
        test_mix[res] = test_mix.get(res, 0) + mid_pct
        test_clean_pct = sum(
            pct for r, pct in test_mix.items()
            if r not in ('gas_ccgt', 'gas_ct')
        )

        # Recompute LMP at this hypothetical mix
        hourly_lmp, avg_lmp, _ = compute_lmp_at_threshold(
            iso, test_clean_pct, conditions['fuel_level'],
            demand_norm, demand_mw_profile,
            supply_profiles, test_mix,
        )

        # Check profitability at this level
        result = compute_resource_profit(
            res, iso, year, demand_twh,
            conditions, cumulative_gw,
            hourly_lmp, supply_profiles,
            test_clean_pct,
        )

        if result['profit_margin'] > 0:
            lo_gw = mid_gw   # still profitable, try more
        else:
            hi_gw = mid_gw   # unprofitable, try less

    return lo_gw  # last known profitable level


def run_market_year(
    iso, year, demand_twh,
    ratchet: FloorRatchet,
    conditions: dict,
    supply_profiles: dict,
    demand_norm, demand_mw_profile,
    build_cap: BuildRateConstraint,
):
    """One year of market-driven deployment.

    1. Rank all buildable resources by profit margin at current LMP
    2. Build most profitable first, up to saturation or queue cap
    3. Update LMP after each resource build
    4. Advance floor ratchet
    5. Return: what got built, resulting clean%, gas built, costs

    Gas competes with clean resources on equal footing.
    CFE is whatever clean% emerges from profit-driven decisions.
    """
    floor_pcts = ratchet.get_floor_pcts(demand_twh)
    current_mix = dict(floor_pcts)
    buildable = ratchet.get_buildable_resources()
    resource_caps = ratchet.get_resource_caps(demand_twh)

    # Effective build budget in GW
    remaining_gw = build_cap.effective_cap_gw(demand_twh)

    # Compute current LMP
    current_clean_pct = sum(
        pct for res, pct in current_mix.items()
        if res not in ('gas_ccgt', 'gas_ct')
    )
    hourly_lmp, avg_lmp, p90_lmp = compute_lmp_at_threshold(
        iso, current_clean_pct, conditions['fuel_level'],
        demand_norm, demand_mw_profile,
        supply_profiles, current_mix,
    )

    builds = []  # log of what gets built

    # --- ITERATIVE MERIT ORDER ---
    # Re-rank after each build because LMP changes
    max_rounds = 15  # safety valve
    for round_idx in range(max_rounds):
        if remaining_gw <= 0.1:
            break

        # Score all buildable resources at current LMP
        margins = []
        for res in buildable:
            # Skip if capped
            if res in resource_caps and current_mix.get(res, 0) >= resource_caps[res]:
                continue
            # Skip offshore if not in pathway/ISO
            if res == 'offshore_wind' and iso not in OFFSHORE_ISOS:
                continue

            result = compute_resource_profit(
                res, iso, year, demand_twh,
                conditions, ratchet.cumulative_gw,
                hourly_lmp, supply_profiles,
                current_clean_pct,
            )
            if result['profit_margin'] > 0:
                margins.append(result)

        if not margins:
            break  # nothing profitable — market stop

        # Sort by profit margin, highest first
        margins.sort(key=lambda x: x['profit_margin'], reverse=True)
        best = margins[0]
        res = best['resource']

        # Find saturation GW for this resource
        sat_gw = find_saturation_gw(
            res, iso, year, demand_twh,
            conditions, ratchet.cumulative_gw,
            supply_profiles, demand_norm, demand_mw_profile,
            current_mix,
        )

        # Constrain by queue budget and resource cap
        build_gw = min(sat_gw, remaining_gw)
        if res in resource_caps:
            cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.25)
            cap_remaining_twh = (resource_caps[res] - current_mix.get(res, 0)) / 100.0 * demand_twh
            cap_remaining_gw = cap_remaining_twh / (cf * 8.760) if cf > 0 else 0
            build_gw = min(build_gw, cap_remaining_gw)

        if build_gw < 0.01:
            # This resource is saturated/capped, skip to next
            buildable = buildable - {res}
            continue

        # --- DEPLOY ---
        cf = RESOURCE_CAPACITY_FACTORS.get(res, {}).get(iso, 0.25)
        if res == 'gas_ccgt':
            cf = 0.85
        elif res == 'gas_ct':
            cf = 0.10
        build_twh = build_gw * cf * 8.760
        build_pct = build_twh / demand_twh * 100.0

        current_mix[res] = current_mix.get(res, 0) + build_pct
        remaining_gw -= build_gw

        # Track gas
        if res in ('gas_ccgt', 'gas_ct'):
            ratchet.record_gas(build_gw, year)
        else:
            # Update clean% for LMP/revenue recalculation
            current_clean_pct = sum(
                pct for r, pct in current_mix.items()
                if r not in ('gas_ccgt', 'gas_ct')
            )

        builds.append({
            'resource': res,
            'gw': round(build_gw, 2),
            'twh': round(build_twh, 1),
            'profit_margin': best['profit_margin'],
            'revenue': best['revenue'],
            'cost': best['cost'],
        })

        # --- RECOMPUTE LMP at new mix ---
        hourly_lmp, avg_lmp, p90_lmp = compute_lmp_at_threshold(
            iso, current_clean_pct, conditions['fuel_level'],
            demand_norm, demand_mw_profile,
            supply_profiles, current_mix,
        )

    # --- ADVANCE RATCHET ---
    built_twh = {res: pct / 100.0 * demand_twh for res, pct in current_mix.items()}
    built_gw = {}
    for b in builds:
        tech = RESOURCE_TO_TECH.get(b['resource'], b['resource'])
        built_gw[tech] = built_gw.get(tech, 0) + b['gw']
    ratchet.advance(built_twh, built_gw=built_gw)

    # --- RESULTS ---
    clean_pct = sum(
        pct for res, pct in current_mix.items()
        if res not in ('gas_ccgt', 'gas_ct')
    )

    return {
        'year': year,
        'iso': iso,
        'clean_pct': round(clean_pct, 1),
        'mix_pcts': {k: round(v, 2) for k, v in current_mix.items() if v > 0.1},
        'builds': builds,
        'total_new_gw': round(sum(b['gw'] for b in builds), 2),
        'avg_lmp': round(avg_lmp, 1),
        'lmp_p90': round(p90_lmp, 1),
        'queue_used_gw': round(build_cap.effective_cap_gw(demand_twh) - remaining_gw, 1),
        'market_stop_reason': 'queue_exhausted' if remaining_gw < 0.1
                              else 'nothing_profitable' if not builds
                              else 'deployed',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONSTRAINED SOLVER — DE for net-zero trajectories
# ═══════════════════════════════════════════════════════════════════════════════

def run_constrained_year(
    iso, year, demand_twh,
    target_cfe,           # CFE floor from emission cap
    cfe_ceiling,          # from step 2.3 — don't overbuild past next year's target
    ratchet: FloorRatchet,
    conditions: dict,
    supply_profiles: dict,
    demand_norm, demand_mw_profile,
    P32, dm32, dn32,      # 8760 scoring arrays from step 2.3
    build_cap: BuildRateConstraint,
    de_popsize=10,        # × n_dims
    de_maxiter=100,
    de_seed=42,
):
    """One year of emission-constrained deployment via DE.

    Lifts step 2.3 architecture:
      - Floor ratchet as lower bounds (can't unbuild)
      - CFE target as constraint (emission cap → required clean%)
      - CFE ceiling to prevent overbuilding
      - DE searches for cheapest portfolio that meets target

    Objective: minimize total system cost of the incremental build.
    Constraint: CFE(mix, 8760_profiles) >= target_cfe.

    Gas is NOT in the DE search space — constrained trajectories are
    about minimizing fossil, not optimizing a gas/clean portfolio.
    DE only searches over clean resources + storage.
    """
    from scipy.optimize import differential_evolution, NonlinearConstraint

    floor_pcts = ratchet.get_floor_pcts(demand_twh)
    buildable = ratchet.get_buildable_resources() - {'gas_ccgt', 'gas_ct'}
    resource_caps = ratchet.get_resource_caps(demand_twh)

    # Filter to resources actually available in this pathway/ISO
    free_resources = []
    for res in sorted(buildable):
        if res in resource_caps and resource_caps[res] <= 0:
            continue
        if res == 'offshore_wind' and iso not in OFFSHORE_ISOS:
            continue
        free_resources.append(res)

    n_free = len(free_resources)
    n_storage = len(STORAGE_TYPES)
    n_dims = n_free + n_storage

    # --- BOUNDS (floor ratchet → lower, caps → upper) ---
    bounds = []
    for res in free_resources:
        lo = floor_pcts.get(res, 0.0)
        hi = resource_caps.get(res, 200.0)
        hi = max(hi, lo + 0.01)
        bounds.append((lo, hi))

    STORAGE_CAPS = {'battery4': 30.0, 'battery8': 25.0, 'ldes': 20.0}
    for sc in STORAGE_TYPES:
        lo = ratchet.storage_pct.get(sc, 0.0)
        hi = STORAGE_CAPS.get(sc, 30.0)
        bounds.append((lo, hi))

    queue_budget_gw = build_cap.effective_cap_gw(demand_twh)

    # --- OBJECTIVE: minimize incremental cost ---
    def objective(x):
        # Delta from floor → new build
        delta_twh = {}
        for i, res in enumerate(free_resources):
            delta_pct = x[i] - floor_pcts.get(res, 0.0)
            if delta_pct > 0.1:
                delta_twh[res] = delta_pct / 100.0 * demand_twh

        if not delta_twh:
            return 0.0  # nothing to build

        # Queue constraint (hard)
        new_gw = estimate_new_gw_from_delta(delta_twh, iso)
        total_gw = sum(new_gw.values())
        if total_gw > queue_budget_gw:
            return 1e12  # infeasible

        # Cost via Wright's Law
        cost, _ = compute_zone_cost(
            iso, delta_twh,
            conditions['lcoe_level'],
            ratchet.cumulative_gw,
            conditions['learning_speed'],
            year, conditions['tx_level'],
            ppa_level=conditions.get('ppa_level'),
            ira_policy=conditions.get('ira_policy', 'off'),
        )

        # Revenue offset (builds still earn revenue, reducing net cost)
        delta_pcts = {res: x[i] - floor_pcts.get(res, 0.0)
                      for i, res in enumerate(free_resources)
                      if x[i] - floor_pcts.get(res, 0.0) > 0.1}

        if delta_pcts:
            hourly_lmp, _, _ = compute_lmp_at_threshold(
                iso,
                sum(x[i] for i, res in enumerate(free_resources)) + 
                    sum(floor_pcts.get(r, 0) for r in ALL_RESOURCES if r not in free_resources),
                conditions['fuel_level'],
                demand_norm, demand_mw_profile,
                supply_profiles,
                {res: x[i] for i, res in enumerate(free_resources)},
            )
            rev, _, _ = compute_zone_revenue(
                iso, target_cfe, delta_pcts, hourly_lmp,
                supply_profiles, demand_twh * 1e6, year,
            )
            cost -= rev  # net cost = cost - revenue (subsidy needed)

        return max(0.0, cost)  # subsidy can't be negative

    # --- CFE CONSTRAINT ---
    # Uses step 2.3 scoring kernel: score_candidates on the 8760 profiles
    def cfe_constraint(x):
        """Returns CFE% - target. Must be >= 0."""
        # Build W matrix for score_candidates (single candidate)
        W = np.zeros((1, N_RESOURCES), dtype=np.float32)
        for ri, res in enumerate(RESOURCE_ORDER):
            if res in free_resources:
                fi = free_resources.index(res)
                W[0, ri] = np.float32(x[fi] * 0.01)
            else:
                W[0, ri] = np.float32(floor_pcts.get(res, 0.0) * 0.01)

        batt4 = np.array([x[n_free]], dtype=np.float32)
        batt8 = np.array([x[n_free + 1]], dtype=np.float32)
        ldes_arr = np.array([x[n_free + 2]], dtype=np.float32)

        scores, _, _ = score_candidates(W, P32, dm32, dn32, batt4, batt8, ldes_arr)
        achieved_cfe = float(scores[0])

        return achieved_cfe - target_cfe  # must be >= 0

    constraint = NonlinearConstraint(cfe_constraint, lb=0.0, ub=np.inf)

    # --- RUN DE ---
    result = differential_evolution(
        objective,
        bounds=bounds,
        constraints=constraint,
        seed=de_seed,
        maxiter=de_maxiter,
        popsize=de_popsize,       # × n_dims → population size
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
    )

    if not result.success and result.fun >= 1e11:
        return None  # infeasible

    # --- RECONSTRUCT WINNER ---
    x = result.x
    winner_pcts = dict(floor_pcts)
    for i, res in enumerate(free_resources):
        winner_pcts[res] = float(x[i])

    winner_storage = {
        sc: float(x[n_free + j])
        for j, sc in enumerate(STORAGE_TYPES)
    }

    # Score final CFE
    cfe_val = cfe_constraint(x) + target_cfe

    # Compute final delta and costs
    delta_twh = {}
    for i, res in enumerate(free_resources):
        d = x[i] - floor_pcts.get(res, 0.0)
        if d > 0.1:
            delta_twh[res] = d / 100.0 * demand_twh

    new_gw = estimate_new_gw_from_delta(delta_twh, iso)

    cost, per_res_cost = compute_zone_cost(
        iso, delta_twh, conditions['lcoe_level'],
        ratchet.cumulative_gw, conditions['learning_speed'],
        year, conditions['tx_level'],
        ppa_level=conditions.get('ppa_level'),
        ira_policy=conditions.get('ira_policy', 'off'),
    )

    # Advance ratchet
    built_twh = {res: pct / 100.0 * demand_twh for res, pct in winner_pcts.items()}
    ratchet.advance(built_twh, built_storage_pct=winner_storage, built_gw=new_gw)
    ratchet.advance_storage(winner_storage)

    return {
        'year': year,
        'iso': iso,
        'clean_pct': round(cfe_val, 1),
        'target_cfe': target_cfe,
        'mix_pcts': {k: round(v, 2) for k, v in winner_pcts.items() if v > 0.1},
        'storage_pcts': winner_storage,
        'delta_twh': {k: round(v, 1) for k, v in delta_twh.items()},
        'new_gw': new_gw,
        'subsidy_per_mwh': round(result.fun, 2),
        'de_iterations': result.nit,
        'de_converged': result.success,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN LOOP — ties it together
# ═══════════════════════════════════════════════════════════════════════════════

def run_scenario(scenario_id, conditions, isos=None):
    """Run one parametric sweep scenario across all ISOs and years.

    Dispatches to MarketSolver or ConstrainedSolver based on
    conditions['emission_constraint'].
    """
    sweep_type = conditions.get('emission_constraint')  # None, 'power_nz', 'economy_nz'
    is_constrained = sweep_type is not None

    results = {}
    for iso in (isos or ISOS):
        ratchet = FloorRatchet.from_grid_baseline(iso, conditions['pathway'])

        # Load 8760 profiles once per ISO
        demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
        supply_profiles = get_supply_profiles(iso, gen_profiles, include_hybrids=True)
        demand_norm, total_mwh_base = get_demand_profile(iso, demand_data)

        # For constrained: prepare scoring arrays (step 2.3 format)
        if is_constrained:
            P32, dm32, dn32 = prepare_scoring_arrays(iso, supply_profiles, demand_norm)

        iso_results = []
        for year in SIM_YEARS:
            if year == 2023:
                iso_results.append(build_2023_baseline(iso))
                continue

            demand_twh = get_demand_at_year(iso, year, conditions['demand_growth'])
            growth_factor = demand_twh / REGIONAL_DEMAND_TWH[iso]
            demand_mw_profile = np.array(demand_norm) * total_mwh_base * growth_factor

            period_years = 7 if year == 2030 else 5
            build_cap = BuildRateConstraint(
                gw_per_year=QUEUE_CAP_GW[conditions['queue_type']][iso],
                pct_demand_per_year=0.08,  # 8% of demand per year
                years_in_period=period_years,
            )

            if is_constrained:
                # Emission cap → required CFE target
                target_cfe = emission_cap_to_cfe(
                    iso, year, sweep_type,
                    demand_twh, emission_rates, fossil_mix,
                    conditions.get('reduction_target', 1.0),
                )
                # CFE ceiling from next year's target (don't overbuild)
                next_year = SIM_YEARS[SIM_YEARS.index(year) + 1] if year < 2050 else 2050
                cfe_ceiling = emission_cap_to_cfe(
                    iso, next_year, sweep_type,
                    get_demand_at_year(iso, next_year, conditions['demand_growth']),
                    emission_rates, fossil_mix,
                    conditions.get('reduction_target', 1.0),
                )

                year_result = run_constrained_year(
                    iso, year, demand_twh,
                    target_cfe, cfe_ceiling,
                    ratchet, conditions,
                    supply_profiles, demand_norm, demand_mw_profile,
                    P32, dm32, dn32,
                    build_cap,
                )
            else:
                year_result = run_market_year(
                    iso, year, demand_twh,
                    ratchet, conditions,
                    supply_profiles, demand_norm, demand_mw_profile,
                    build_cap,
                )

            iso_results.append(year_result)

        results[iso] = iso_results

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STUBS — functions that need implementation
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_scoring_arrays(iso, supply_profiles, demand_norm):
    """Convert step 6.1 supply profiles to step 2.3 scoring format.

    Returns (P32, dm32, dn32) — the arrays score_candidates needs.
    P32: (N_RESOURCES, 8760) float32 — normalized generation profiles
    dm32: (8760,) float32 — demand profile (for RA margin)
    dn32: (8760,) float32 — demand profile (for CFE scoring)
    """
    # TODO: bridge between step 6.1 profile format and step 2.3 kernel
    raise NotImplementedError


def emission_cap_to_cfe(iso, year, constraint_type, demand_twh,
                         emission_rates, fossil_mix, reduction_target):
    """Convert emission cap (Mt) to required CFE percentage.

    Inverts: emissions = (1 - cfe/100) × demand_twh × emission_rate
    to find: cfe = 100 × (1 - cap / (demand_twh × emission_rate))
    """
    # TODO: uses get_emission_cap_mt + compute_fossil_retirement
    raise NotImplementedError


def build_2023_baseline(iso):
    """2023 eGRID actual data — identical for all scenarios."""
    # TODO: same as current step 6.1 2023 block
    raise NotImplementedError
