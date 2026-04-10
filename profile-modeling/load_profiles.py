"""
Load profile generators — produce 8760-hour demand shapes.

Each generator returns a numpy array of length 8760 normalized so sum = 8760
(equivalent to 1 MW average load = 8760 MWh/yr, 1 MWh per hour on average).
"""

import numpy as np

H = 8760


def flat_load():
    """Perfectly flat load — 1 MWh every hour. Ideal baseload (data center IT)."""
    return np.ones(H, dtype=np.float64)


def datacenter_pue(climate='hot'):
    """Data center with temperature-dependent PUE (cooling overhead).

    Args:
        climate: 'hot' (ERCOT/CAISO/SPP), 'moderate' (PJM/MISO),
                 'cold' (NYISO/NEISO)

    PUE model (well-designed hyperscale):
      T < 10°C:  1.08  (full free cooling)
      10-20°C:   1.08 → 1.15
      20-30°C:   1.15 → 1.25
      30-40°C:   1.25 → 1.35
      > 40°C:    1.35
    """
    climate_params = {
        'hot':      {'avg': 20.0, 'amp': 16.5},   # Texas, Central Valley
        'moderate': {'avg': 13.0, 'amp': 13.0},    # Mid-Atlantic, Midwest
        'cold':     {'avg': 9.0,  'amp': 14.0},    # Northeast, New England
    }
    params = climate_params[climate]

    hours = np.arange(H)
    day_of_year = hours / 24.0

    # Annual cycle: hottest ~Jul 20 (day 201)
    annual = params['avg'] + params['amp'] * np.cos(2 * np.pi * (day_of_year - 201) / 365)

    # Diurnal cycle: peak at ~15:00 local (hour 21 UTC for CST)
    hour_of_day = hours % 24
    diurnal = 5.0 * np.cos(2 * np.pi * (hour_of_day - 21) / 24)

    np.random.seed(42)
    temp = annual + diurnal + np.random.normal(0, 2.0, H)

    pue = np.piecewise(temp,
        [temp < 10, (temp >= 10) & (temp < 20), (temp >= 20) & (temp < 30),
         (temp >= 30) & (temp < 40), temp >= 40],
        [1.08,
         lambda t: 1.08 + (t - 10) * 0.007,
         lambda t: 1.15 + (t - 20) * 0.010,
         lambda t: 1.25 + (t - 30) * 0.010,
         1.35])

    # Normalize: IT load × PUE, scaled so sum = H
    profile = pue / pue.sum() * H
    return profile


def industrial_2shift():
    """Industrial 2-shift load — high weekday, reduced nights/weekends.

    Pattern: 1.0 during 6am-10pm weekdays, 0.4 nights, 0.5 weekends.
    """
    profile = np.zeros(H, dtype=np.float64)
    for h in range(H):
        day = h // 24
        hour_of_day = h % 24
        day_of_week = day % 7  # 0=Monday (Jan 1 2025 is Wednesday → offset)
        # 2025 starts on Wednesday: day 0 = Wed (2), day 1 = Thu (3), ...
        dow = (day + 2) % 7  # 0=Mon, 5=Sat, 6=Sun

        if dow >= 5:  # Weekend
            profile[h] = 0.5
        elif 6 <= hour_of_day <= 21:  # Weekday daytime (6am-9pm UTC, approx)
            profile[h] = 1.0
        else:  # Weekday night
            profile[h] = 0.4

    # Normalize so sum = H
    profile = profile / profile.sum() * H
    return profile


def eia_demand_shape(iso, gen_profiles_loader=None):
    """Actual ISO demand shape from EIA-930 data.

    Uses the normalized demand profile from the EIA data. If not available,
    falls back to flat.
    """
    import sys, os
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from eia_data_io import load_demand_profiles

    demand_data = load_demand_profiles()
    iso_data = demand_data.get(iso, {})

    # Try years in order of preference
    for year in ['2025', '2024', '2023']:
        if year in iso_data:
            norm = iso_data[year].get('normalized')
            if norm and len(norm) >= H:
                arr = np.array(norm[:H], dtype=np.float64)
                # Rescale so sum = H
                arr = arr / arr.sum() * H
                return arr

    # Fallback
    return flat_load()


# Registry of available profiles
LOAD_PROFILES = {
    'flat': flat_load,
    'datacenter_hot': lambda: datacenter_pue('hot'),
    'datacenter_moderate': lambda: datacenter_pue('moderate'),
    'datacenter_cold': lambda: datacenter_pue('cold'),
    'industrial_2shift': industrial_2shift,
}


def get_load_profile(name, iso=None):
    """Get a load profile by name. Returns (8760,) array summing to 8760."""
    if name == 'eia_demand':
        if iso is None:
            raise ValueError("ISO required for eia_demand profile")
        return eia_demand_shape(iso)
    if name in LOAD_PROFILES:
        return LOAD_PROFILES[name]()
    raise ValueError(f"Unknown load profile: {name}. Available: {list(LOAD_PROFILES.keys()) + ['eia_demand']}")
