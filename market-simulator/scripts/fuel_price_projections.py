#!/usr/bin/env python3
"""
Fuel Price Projections — Year-Varying EIA AEO 2025 Fuel Prices
==============================================================
Loads annual fuel price projections (gas, coal, oil) from the parsed
EIA AEO 2025 JSON file and provides a lookup function for any year/level.

Three cases map to the market simulator's L/M/H sensitivity levels:
    Low    → AEO 2025 Low Oil Price case
    Medium → AEO 2025 Reference case
    High   → AEO 2025 High Oil Price case

All prices in real 2024 $/MMBtu.

Usage:
    from fuel_price_projections import get_fuel_prices, FUEL_PRICE_PROJECTIONS

    # Year-specific prices
    prices = get_fuel_prices(2035, 'Medium')
    # → {'gas': 4.43, 'coal': 3.96, 'oil': 13.81}

    # Static fallback (backward compat — uses 2026 prices)
    from fuel_price_projections import FUEL_PRICES
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'eia-natgas-prices')
PROJECTIONS_FILE = os.path.join(DATA_DIR, 'aeo2025_fuel_price_projections.json')

# Static fallback prices (original lmp_engine.py values) — used only if JSON not found
_STATIC_FALLBACK = {
    'Low':    {'coal': 2.00, 'gas': 2.00, 'oil': 8.00},
    'Medium': {'coal': 2.25, 'gas': 3.50, 'oil': 10.50},
    'High':   {'coal': 2.50, 'gas': 6.00, 'oil': 13.00},
}


def _load_projections() -> dict:
    """Load fuel price projections from JSON file.

    Returns:
        dict: {case: {fuel_type: {year_str: price}}} or empty dict if unavailable
    """
    if not os.path.exists(PROJECTIONS_FILE):
        return {}
    with open(PROJECTIONS_FILE, 'r') as f:
        data = json.load(f)
    return data.get('cases', {})


# Load once at import time
FUEL_PRICE_PROJECTIONS = _load_projections()

# Available projection years (sorted int list)
_AVAILABLE_YEARS = set()
for _case_data in FUEL_PRICE_PROJECTIONS.values():
    for _fuel_data in _case_data.values():
        _AVAILABLE_YEARS.update(int(y) for y in _fuel_data.keys())
PROJECTION_YEARS = sorted(_AVAILABLE_YEARS)


def get_fuel_prices(year: int, level: str = 'Medium') -> dict:
    """Get fuel prices for a specific year and sensitivity level.

    Args:
        year: Simulation year (2023-2050)
        level: 'Low', 'Medium', or 'High'

    Returns:
        dict: {'gas': X, 'coal': Y, 'oil': Z} in $/MMBtu
    """
    case_data = FUEL_PRICE_PROJECTIONS.get(level)
    if not case_data:
        # Fall back to static prices if projection data missing for this case
        return dict(_STATIC_FALLBACK.get(level, _STATIC_FALLBACK['Medium']))

    result = {}
    for fuel_type in ('gas', 'coal', 'oil'):
        fuel_ts = case_data.get(fuel_type, {})
        if not fuel_ts:
            result[fuel_type] = _STATIC_FALLBACK.get(level, {}).get(fuel_type, 3.50)
            continue

        year_str = str(year)
        if year_str in fuel_ts:
            result[fuel_type] = fuel_ts[year_str]
        else:
            # Linear interpolation between nearest available years
            avail = sorted(int(y) for y in fuel_ts.keys())
            if year <= avail[0]:
                result[fuel_type] = fuel_ts[str(avail[0])]
            elif year >= avail[-1]:
                result[fuel_type] = fuel_ts[str(avail[-1])]
            else:
                # Find bracketing years
                lo = max(y for y in avail if y <= year)
                hi = min(y for y in avail if y >= year)
                if lo == hi:
                    result[fuel_type] = fuel_ts[str(lo)]
                else:
                    v_lo = fuel_ts[str(lo)]
                    v_hi = fuel_ts[str(hi)]
                    frac = (year - lo) / (hi - lo)
                    result[fuel_type] = round(v_lo + frac * (v_hi - v_lo), 4)

    return result


def get_fuel_prices_all_years(level: str = 'Medium') -> dict:
    """Get fuel prices for all available years at a given sensitivity level.

    Returns:
        dict: {year_int: {'gas': X, 'coal': Y, 'oil': Z}}
    """
    return {yr: get_fuel_prices(yr, level) for yr in PROJECTION_YEARS}


# Backward-compatible static dict — uses 2026 values (first projection year)
# This allows existing code that does FUEL_PRICES['Medium']['gas'] to keep working
FUEL_PRICES = {}
for _lvl in ('Low', 'Medium', 'High'):
    FUEL_PRICES[_lvl] = get_fuel_prices(2026, _lvl)
