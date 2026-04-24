"""
Shared test fixtures for the Hourly CFE Optimizer model validation suite.

All fixtures use deterministic synthetic data — no dependency on EIA/eGRID data files.
This ensures tests run in CI without the 100+ MB data directory.
"""
import sys
import os
import numpy as np
import pytest

# Add scripts/ to path so we can import pipeline modules
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

H = 8760  # hours in a non-leap year


# ============================================================================
# SYNTHETIC PROFILES (deterministic, no randomness)
# ============================================================================

@pytest.fixture
def synthetic_demand():
    """Flat normalized demand profile: 1/8760 per hour, sums to 1.0."""
    return np.full(H, 1.0 / H, dtype=np.float64)


@pytest.fixture
def synthetic_solar():
    """Idealized solar profile: zero at night (hours 0-5, 19-23), sinusoidal peak midday.

    Normalized to sum = 1.0 over 8760 hours.
    """
    profile = np.zeros(H, dtype=np.float64)
    for h in range(H):
        hour_of_day = h % 24
        if 6 <= hour_of_day <= 18:
            # Sinusoidal from sunrise (6) to sunset (18)
            profile[h] = np.sin(np.pi * (hour_of_day - 6) / 12)
    # Normalize to sum = 1.0
    total = profile.sum()
    if total > 0:
        profile /= total
    return profile


@pytest.fixture
def synthetic_wind():
    """Idealized wind profile: higher at night, lower midday (anti-correlated with solar).

    Normalized to sum = 1.0 over 8760 hours.
    """
    profile = np.zeros(H, dtype=np.float64)
    for h in range(H):
        hour_of_day = h % 24
        # Base wind + nighttime bonus
        profile[h] = 0.5 + 0.5 * np.cos(np.pi * (hour_of_day - 3) / 12)
    total = profile.sum()
    if total > 0:
        profile /= total
    return profile


@pytest.fixture
def synthetic_clean_firm():
    """Flat baseload profile: constant 1/8760 per hour. Sums to 1.0."""
    return np.full(H, 1.0 / H, dtype=np.float64)


@pytest.fixture
def synthetic_profiles(synthetic_demand, synthetic_solar, synthetic_wind, synthetic_clean_firm):
    """Complete supply profile dict for dispatch_utils functions."""
    return {
        'clean_firm': synthetic_clean_firm.tolist(),
        'solar': synthetic_solar.tolist(),
        'wind': synthetic_wind.tolist(),
        'offshore_wind': [0.0] * H,
        'ccs_ccgt': (np.full(H, 1.0 / H)).tolist(),
        'hydro': (np.full(H, 1.0 / H)).tolist(),
    }


@pytest.fixture
def small_mix():
    """A known deterministic resource mix for spot-checking.

    50% clean_firm, 20% solar, 20% wind, 0% offshore_wind, 0% CCS, 10% hydro = 100% total.
    """
    return {
        'clean_firm': 50.0,
        'solar': 20.0,
        'wind': 20.0,
        'offshore_wind': 0.0,
        'ccs_ccgt': 0.0,
        'hydro': 10.0,
    }


@pytest.fixture
def synthetic_hybrid_profiles(synthetic_profiles, synthetic_solar, synthetic_wind):
    """Supply profiles dict including 4 hybrid co-located resources.

    Hybrid profiles are derived from their parent renewables with battery smoothing:
    solar_batt4/8 extend solar generation into evening hours,
    wind_batt4/8 smooth wind variability into overnight gaps.
    Each profile is normalized to sum = 1.0.
    """
    profiles = dict(synthetic_profiles)  # copy base 6

    # --- Solar+Battery hybrids: solar shape with evening tail ---
    solar_arr = np.array(synthetic_solar, dtype=np.float64)
    for suffix, shift_hours in [('solar_batt4', 3), ('solar_batt8', 5)]:
        hybrid = np.zeros(H, dtype=np.float64)
        for h in range(H):
            hod = h % 24
            if 6 <= hod <= 18:
                # Reduced midday (battery absorbs some), shifted peak later
                hybrid[h] = np.sin(np.pi * (hod - 6) / 12) * 0.7
            elif 18 < hod <= 18 + shift_hours:
                # Evening discharge tail
                hybrid[h] = 0.4 * (1.0 - (hod - 18) / (shift_hours + 1))
        total = hybrid.sum()
        if total > 0:
            hybrid /= total
        profiles[suffix] = hybrid.tolist()

    # --- Wind+Battery hybrids: wind shape with smoothed overnight ---
    for suffix, smooth_factor in [('wind_batt4', 0.3), ('wind_batt8', 0.5)]:
        hybrid = np.zeros(H, dtype=np.float64)
        for h in range(H):
            hod = h % 24
            base_wind = 0.5 + 0.5 * np.cos(np.pi * (hod - 3) / 12)
            # Smooth: reduce peaks, fill troughs
            hybrid[h] = base_wind * (1.0 - smooth_factor) + smooth_factor * 0.5
        total = hybrid.sum()
        if total > 0:
            hybrid /= total
        profiles[suffix] = hybrid.tolist()

    return profiles


@pytest.fixture
def hybrid_mix():
    """A resource mix including hybrid co-located resources.

    30% clean_firm, 15% solar, 15% wind, 0% offshore_wind, 0% CCS,
    10% hydro, 15% solar_batt4, 15% wind_batt4 = 100% total.
    """
    return {
        'clean_firm': 30.0,
        'solar': 15.0,
        'wind': 15.0,
        'offshore_wind': 0.0,
        'ccs_ccgt': 0.0,
        'hydro': 10.0,
        'solar_batt4': 15.0,
        'solar_batt8': 0.0,
        'wind_batt4': 15.0,
        'wind_batt8': 0.0,
    }


@pytest.fixture
def emission_rates():
    """Synthetic emission rates matching eGRID structure (7 ISOs)."""
    return {
        'CAISO': {'coal_co2_lb_per_mwh': 0, 'gas_co2_lb_per_mwh': 900, 'oil_co2_lb_per_mwh': 0,
                  'grid_avg_co2_lb_per_mwh': 450, 'fossil_avg_co2_lb_per_mwh': 900},
        'ERCOT': {'coal_co2_lb_per_mwh': 2100, 'gas_co2_lb_per_mwh': 900, 'oil_co2_lb_per_mwh': 0,
                  'grid_avg_co2_lb_per_mwh': 800, 'fossil_avg_co2_lb_per_mwh': 1200},
        'PJM':   {'coal_co2_lb_per_mwh': 2100, 'gas_co2_lb_per_mwh': 900, 'oil_co2_lb_per_mwh': 1600,
                  'grid_avg_co2_lb_per_mwh': 850, 'fossil_avg_co2_lb_per_mwh': 1300},
        'NYISO': {'coal_co2_lb_per_mwh': 0, 'gas_co2_lb_per_mwh': 900, 'oil_co2_lb_per_mwh': 1600,
                  'grid_avg_co2_lb_per_mwh': 500, 'fossil_avg_co2_lb_per_mwh': 950},
        'NEISO': {'coal_co2_lb_per_mwh': 2100, 'gas_co2_lb_per_mwh': 900, 'oil_co2_lb_per_mwh': 1600,
                  'grid_avg_co2_lb_per_mwh': 600, 'fossil_avg_co2_lb_per_mwh': 1000},
        'MISO':  {'coal_co2_lb_per_mwh': 2100, 'gas_co2_lb_per_mwh': 900, 'oil_co2_lb_per_mwh': 1600,
                  'grid_avg_co2_lb_per_mwh': 900, 'fossil_avg_co2_lb_per_mwh': 1400},
        'SPP':   {'coal_co2_lb_per_mwh': 2100, 'gas_co2_lb_per_mwh': 900, 'oil_co2_lb_per_mwh': 1600,
                  'grid_avg_co2_lb_per_mwh': 750, 'fossil_avg_co2_lb_per_mwh': 1100},
    }


@pytest.fixture
def fossil_mix():
    """Synthetic fossil fuel mix (hourly shares) for CO2 tests."""
    return {iso: {'coal': [0.4] * H, 'gas': [0.5] * H, 'oil': [0.1] * H}
            for iso in ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']}
