"""
Toggle-based repricing for P10/P50/P90 cost envelopes.

Given a fixed resource mix (optimized at Medium costs), recomputes $/MWh
under all step 2.2 toggle combinations. The mix (physics) is fixed —
only the cost assumptions change.

Toggle dimensions (non-CAISO):
  1. Renewable Gen:  Low / Medium / High  → wind, solar LCOE
  2. Firm Gen:       Low / Medium / High  → nuclear new-build LCOE
  3. Storage:        Low / Medium / High  → battery, LDES, H2 capacity cost
  4. Transmission:   None / Low / Medium / High  → TX adders
  5. CCS:            Low / Medium / High  → CCS-CCGT LCOE
  6. 45Q:            On / Off             → which CCS table
  7. Fossil Fuel:    Low / Medium / High  → wholesale baseline (for premium calc)

Total: 3^5 × 4 × 2 = 1,944 combinations (non-CAISO)
CAISO adds Geothermal (L/M/H): 1,944 × 3 = 5,832

Physics-independent: hourly match score doesn't change. Only $/MWh changes.
"""

import sys, os
import numpy as np
from itertools import product

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from config import (
    LCOE_TABLES, TX_TABLES, get_tx, COST_KEY,
    NUCLEAR_NEWBUILD_LCOE, CCS_LCOE_45Q_ON, CCS_LCOE_45Q_OFF,
    GEOTHERMAL_LCOE, STORAGE_PARAMS, WHOLESALE_PRICES,
)

# Wholesale price sensitivity (derived from pipeline's fossil fuel toggle)
# Fossil L = cheap gas → lower wholesale; Fossil H = expensive gas → higher wholesale
WHOLESALE_SENSITIVITY = {
    'Low':    {iso: max(15, p - 8) for iso, p in WHOLESALE_PRICES.items()},
    'Medium': WHOLESALE_PRICES,
    'High':   {iso: p + 10 for iso, p in WHOLESALE_PRICES.items()},
}

LMH = ['Low', 'Medium', 'High']
TX_LEVELS = ['None', 'Low', 'Medium', 'High']
Q45_LEVELS = ['On', 'Off']


def _resource_cost(resource, iso, re_lev, firm_lev, stor_lev, tx_lev, ccs_lev, q45, geo_lev='Medium'):
    """Cost of one resource under a toggle combination.

    For generation: returns $/MWh produced.
    For storage: returns annualized capacity cost per unit of annual demand.
    """
    ck = COST_KEY  # {'Low': 'L', 'Medium': 'M', 'High': 'H'}

    if resource == 'wind':
        return LCOE_TABLES['wind'][re_lev].get(iso, 999) + get_tx('wind', tx_lev, iso)

    elif resource == 'solar':
        return LCOE_TABLES['solar'][re_lev].get(iso, 999) + get_tx('solar', tx_lev, iso)

    elif resource == 'offshore_wind':
        lcoe = LCOE_TABLES['offshore_wind'][re_lev].get(iso, 0)
        if lcoe == 0:
            return float('inf')
        return lcoe + get_tx('offshore_wind', tx_lev, iso)

    elif resource == 'clean_firm':
        return NUCLEAR_NEWBUILD_LCOE[ck[firm_lev]][iso] + get_tx('clean_firm', tx_lev, iso)

    elif resource == 'ccs_ccgt':
        table = CCS_LCOE_45Q_ON if q45 == 'On' else CCS_LCOE_45Q_OFF
        return table[ck[ccs_lev]][iso] + get_tx('ccs_ccgt', tx_lev, iso)

    elif resource == 'hydro':
        return WHOLESALE_PRICES[iso]  # always wholesale-priced

    elif resource == 'geothermal':
        if iso != 'CAISO':
            return float('inf')
        return GEOTHERMAL_LCOE[ck[geo_lev]] + get_tx('clean_firm', tx_lev, iso)

    elif resource in ('battery4', 'battery8', 'ldes', 'h2'):
        cost_key = STORAGE_PARAMS[resource]['cost_key']
        return LCOE_TABLES[cost_key][stor_lev].get(iso, 999)

    return float('inf')


def reprice_mix(gen_resources, gen_allocs, stor_resources, stor_caps, iso):
    """Reprice a fixed mix under all toggle combinations.

    Args:
        gen_resources: list of generation resource names
        gen_allocs: list of allocations (fractions of annual demand)
        stor_resources: list of storage resource names
        stor_caps: list of capacities (fractions of annual demand)
        iso: ISO region

    Returns:
        dict with:
          costs: list of total $/MWh under each combo
          wholesales: list of wholesale baseline under each combo
          premiums: list of premium (cost - wholesale) under each combo
          percentiles: dict with p10, p25, p50, p75, p90 for cost and premium
    """
    geo_levels = LMH if iso == 'CAISO' else ['Medium']

    costs = []
    premiums = []
    wholesales = []

    for re_lev, firm_lev, stor_lev, tx_lev, ccs_lev, q45, fossil_lev, geo_lev in product(
            LMH, LMH, LMH, TX_LEVELS, LMH, Q45_LEVELS, LMH, geo_levels):

        cost = 0.0
        for r, alloc in zip(gen_resources, gen_allocs):
            unit = _resource_cost(r, iso, re_lev, firm_lev, stor_lev,
                                  tx_lev, ccs_lev, q45, geo_lev)
            if unit < float('inf'):
                cost += alloc * unit

        for s, cap in zip(stor_resources, stor_caps):
            unit = _resource_cost(s, iso, re_lev, firm_lev, stor_lev,
                                  tx_lev, ccs_lev, q45, geo_lev)
            cost += cap * unit

        wholesale = WHOLESALE_SENSITIVITY[fossil_lev][iso]

        costs.append(cost)
        wholesales.append(wholesale)
        premiums.append(cost - wholesale)

    costs = np.array(costs)
    premiums = np.array(premiums)

    return {
        'n_scenarios': len(costs),
        'cost': {
            'p10': float(np.percentile(costs, 10)),
            'p25': float(np.percentile(costs, 25)),
            'p50': float(np.percentile(costs, 50)),
            'p75': float(np.percentile(costs, 75)),
            'p90': float(np.percentile(costs, 90)),
            'min': float(costs.min()),
            'max': float(costs.max()),
            'mean': float(costs.mean()),
        },
        'premium': {
            'p10': float(np.percentile(premiums, 10)),
            'p25': float(np.percentile(premiums, 25)),
            'p50': float(np.percentile(premiums, 50)),
            'p75': float(np.percentile(premiums, 75)),
            'p90': float(np.percentile(premiums, 90)),
            'min': float(premiums.min()),
            'max': float(premiums.max()),
        },
    }
