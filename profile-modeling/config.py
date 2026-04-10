"""
Configuration for profile modeling — cost tables, resource definitions, ISO parameters.

Imports from the main pipeline's pipeline_config.py as single source of truth,
then re-exports in a clean interface for the profile-modeling optimizer.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from pipeline_config import (
    LCOE_TABLES, TX_TABLES, get_tx,
    NUCLEAR_NEWBUILD_LCOE, CCS_LCOE_45Q_ON, CCS_LCOE_45Q_OFF,
    GEOTHERMAL_LCOE, GEOTHERMAL_CAP_TWH,
    WHOLESALE_PRICES, HYDRO_CAP_PCT,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    BATTERY8_EFFICIENCY, BATTERY8_DURATION_HOURS,
    LDES_DURATION_HOURS, LDES_EFFICIENCY, LDES_WINDOW_DAYS,
    H2_DURATION_HOURS, H2_EFFICIENCY, H2_WINDOW_DAYS,
    OFFSHORE_ISOS,
)

H = 8760

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Thresholds from 50% through 99.9%
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.875,
              0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999]

COST_LEVELS = ['Low', 'Medium', 'High']

# Map cost level to pipeline_config's L/M/H keys for firm resources
COST_KEY = {'Low': 'L', 'Medium': 'M', 'High': 'H'}


# ── Resource definitions ─────────────────────────────────────────────────────

# Generation resources: have hourly shape profiles, cost = LCOE + TX per MWh
GENERATION_RESOURCES = ['wind', 'solar', 'clean_firm', 'ccs_ccgt', 'hydro',
                        'offshore_wind', 'geothermal']

# Storage resources: have capacity cost + dispatch parameters
STORAGE_RESOURCES = ['battery4', 'battery8', 'ldes', 'h2']

STORAGE_PARAMS = {
    'battery4': {
        'duration': BATTERY_DURATION_HOURS,   # 4
        'window': 24,
        'rte': BATTERY_EFFICIENCY,            # 0.85
        'cost_key': 'battery',                # key in LCOE_TABLES
    },
    'battery8': {
        'duration': BATTERY8_DURATION_HOURS,  # 8
        'window': 48,
        'rte': BATTERY8_EFFICIENCY,           # 0.85
        'cost_key': 'battery8',
    },
    'ldes': {
        'duration': LDES_DURATION_HOURS,      # 100
        'window': LDES_WINDOW_DAYS * 24,      # 168
        'rte': LDES_EFFICIENCY,               # 0.50
        'cost_key': 'ldes',
    },
    'h2': {
        'duration': H2_DURATION_HOURS,        # 1000
        'window': H2_WINDOW_DAYS * 24,        # 720
        'rte': H2_EFFICIENCY,                 # 0.35
        'cost_key': 'h2',
    },
}


def get_generation_cost(resource, cost_level, iso):
    """All-in cost (LCOE + TX) for a generation resource, $/MWh produced.

    For clean_firm: uses nuclear new-build LCOE (the marginal new capacity cost).
    For ccs_ccgt: uses 45Q-ON LCOE (default assumption).
    For hydro: uses wholesale price (existing-only, no new build).
    For geothermal: flat LCOE (CAISO only).
    """
    key = COST_KEY[cost_level]
    tx_key = cost_level  # TX tables use 'Low'/'Medium'/'High'

    if resource == 'clean_firm':
        lcoe = NUCLEAR_NEWBUILD_LCOE[key][iso]
        tx = get_tx('clean_firm', tx_key, iso)
        return lcoe + tx

    elif resource == 'ccs_ccgt':
        lcoe = CCS_LCOE_45Q_ON[key][iso]
        tx = get_tx('ccs_ccgt', tx_key, iso)
        return lcoe + tx

    elif resource == 'hydro':
        return WHOLESALE_PRICES[iso]  # existing only, at wholesale

    elif resource == 'geothermal':
        if iso != 'CAISO':
            return float('inf')
        lcoe = GEOTHERMAL_LCOE[key]
        tx = get_tx('clean_firm', tx_key, iso)  # same TX as clean_firm
        return lcoe + tx

    else:
        # wind, solar, offshore_wind
        lcoe_table = LCOE_TABLES.get(resource)
        if lcoe_table is None:
            return float('inf')
        lcoe_level = lcoe_table.get(cost_level)
        if lcoe_level is None:
            return float('inf')
        lcoe = lcoe_level.get(iso, float('inf'))
        if lcoe == 0:
            return float('inf')  # resource not available in this ISO
        tx = get_tx(resource, tx_key, iso)
        return lcoe + tx


def get_storage_cost(storage_type, cost_level, iso):
    """Annualized capacity cost for storage, per unit of annual demand.

    Returns the table value from LCOE_TABLES. To get $/MWh of load:
      cost_per_mwh = (capacity_fraction) × table_value
    where capacity_fraction = storage_MWh / annual_demand_MWh.
    """
    params = STORAGE_PARAMS[storage_type]
    table = LCOE_TABLES[params['cost_key']]
    level = table.get(cost_level)
    if level is None:
        return float('inf')
    if isinstance(level, dict):
        return level.get(iso, float('inf'))
    return level


def get_available_resources(iso):
    """Return (gen_resources, storage_resources) available for an ISO."""
    gen = []
    for r in GENERATION_RESOURCES:
        cost = get_generation_cost(r, 'Medium', iso)
        if cost < float('inf'):
            gen.append(r)
    # All storage types available everywhere
    stor = list(STORAGE_RESOURCES)
    return gen, stor


def get_hydro_cap_pct(iso):
    """Max hydro allocation as % of annual demand."""
    return HYDRO_CAP_PCT.get(iso, 0.0)
