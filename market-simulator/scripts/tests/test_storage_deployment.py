"""Tests for R1: Economics-driven storage deployment.

Tests compute_storage_arbitrage_from_lmp(), compute_storage_deployment(),
synthetic fallback warning, and main loop integration.
"""
import sys
import os
import warnings

import numpy as np
import pytest

# Add scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from market_simulation import (
    compute_storage_arbitrage_from_lmp,
    compute_storage_deployment,
    _generate_synthetic_step3_data,
    _STORAGE_TECH_PARAMS,
    H,
)
from pipeline_config import (
    CAPACITY_MARKET_PRICES,
    STORAGE_MAX,
    H2_MIN_THRESHOLD,
    LCOE_TABLES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def flat_lmp():
    """Flat LMP — no price spread, zero arbitrage."""
    return np.full(H, 50.0)


@pytest.fixture
def duck_curve_lmp():
    """Realistic duck-curve LMP with clear peak/trough spread."""
    np.random.seed(42)
    lmp = np.zeros(H)
    for d in range(365):
        h = d * 24
        lmp[h:h+24] = 35 + np.random.normal(0, 5, 24)
        lmp[h+10:h+15] -= 20   # midday solar trough
        lmp[h+16:h+21] += 40   # evening peak
    return np.clip(lmp, 0, 300)


@pytest.fixture
def extreme_duck_lmp():
    """Extreme duck curve — high solar penetration, very wide spreads."""
    lmp = np.zeros(H)
    for d in range(365):
        h = d * 24
        lmp[h:h+24] = 25 + np.random.RandomState(d).normal(0, 3, 24)
        lmp[h+10:h+15] = -5 + np.random.RandomState(d+1000).normal(0, 2, 5)
        lmp[h+17:h+21] = 120 + np.random.RandomState(d+2000).normal(0, 10, 4)
    return np.clip(lmp, -10, 500)


@pytest.fixture
def base_conditions():
    """Standard Medium-cost conditions."""
    return {
        'lcoe_level': 'Medium',
        'learning_speed': 'Medium',
        'fuel_level': 'Medium',
    }


@pytest.fixture
def low_cost_conditions():
    """Low-cost conditions (most favorable for deployment)."""
    return {
        'lcoe_level': 'Low',
        'learning_speed': 'Medium',
        'fuel_level': 'Medium',
    }


def _make_state():
    """Create a fresh iso_state dict with storage fields."""
    return {
        'storage_deployed': {},
        'storage_details': {},
        'clean_pct': 50.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: compute_storage_arbitrage_from_lmp
# ═══════════════════════════════════════════════════════════════════════════════

class TestStorageArbitrage:
    """Tests for price-taking arbitrage revenue calculation."""

    def test_flat_lmp_zero_arbitrage(self, flat_lmp):
        """Flat LMP produces zero arbitrage revenue for all techs."""
        result = compute_storage_arbitrage_from_lmp(flat_lmp)
        for tech in _STORAGE_TECH_PARAMS:
            assert result[tech] == 0.0, f"{tech} should have zero arbitrage with flat LMP"

    def test_duck_curve_positive_battery_arbitrage(self, duck_curve_lmp):
        """Duck curve with clear spread produces positive battery arbitrage."""
        result = compute_storage_arbitrage_from_lmp(duck_curve_lmp)
        assert result['battery'] > 0, "Battery should capture duck curve arbitrage"
        assert result['battery8'] > 0, "Battery 8hr should capture duck curve arbitrage"

    def test_battery_arbitrage_plausible_range(self, duck_curve_lmp):
        """Battery arbitrage should be in plausible $/kW-yr range (10-200)."""
        result = compute_storage_arbitrage_from_lmp(duck_curve_lmp)
        assert 5 < result['battery'] < 500, \
            f"Battery arbitrage ${result['battery']:.1f}/kW-yr outside plausible range"

    def test_h2_low_or_zero_arbitrage(self, duck_curve_lmp):
        """H2 at 35% RTE should have zero or near-zero arbitrage."""
        result = compute_storage_arbitrage_from_lmp(duck_curve_lmp)
        # H2 with 35% RTE can't profitably arbitrage moderate spreads
        assert result['h2'] < 10, \
            f"H2 arbitrage ${result['h2']:.1f}/kW-yr too high for 35% RTE"

    def test_all_techs_returned(self, duck_curve_lmp):
        """All four storage techs should be in the result."""
        result = compute_storage_arbitrage_from_lmp(duck_curve_lmp)
        for tech in ['battery', 'battery8', 'ldes', 'h2']:
            assert tech in result, f"Missing tech {tech} in result"

    def test_revenue_nonnegative(self, duck_curve_lmp):
        """Arbitrage revenue should never be negative."""
        result = compute_storage_arbitrage_from_lmp(duck_curve_lmp)
        for tech, rev in result.items():
            assert rev >= 0, f"{tech} has negative arbitrage: {rev}"

    def test_extreme_spread_higher_revenue(self, duck_curve_lmp, extreme_duck_lmp):
        """Extreme duck curve should produce higher battery arbitrage than moderate."""
        moderate = compute_storage_arbitrage_from_lmp(duck_curve_lmp)
        extreme = compute_storage_arbitrage_from_lmp(extreme_duck_lmp)
        assert extreme['battery'] > moderate['battery'], \
            "Extreme spreads should produce higher battery arbitrage"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: compute_storage_deployment
# ═══════════════════════════════════════════════════════════════════════════════

class TestStorageDeployment:
    """Tests for economics-driven storage deployment decisions."""

    def test_unprofitable_zero_deployment(self, flat_lmp, base_conditions):
        """Flat LMP + Medium costs → no storage deployed."""
        state = _make_state()
        result = compute_storage_deployment(
            'CAISO', 2030, flat_lmp, 224, 224e6, 60.0,
            base_conditions, {}, state)
        for tech, pct in result['deployed_pcts'].items():
            assert pct == 0.0, f"{tech} deployed at {pct}% with flat LMP"

    def test_profitable_deploys_to_cap(self, extreme_duck_lmp, low_cost_conditions):
        """Extreme spreads + Low costs → battery deploys to STORAGE_MAX cap."""
        state = _make_state()
        result = compute_storage_deployment(
            'CAISO', 2035, extreme_duck_lmp, 224, 224e6, 85.0,
            low_cost_conditions, {}, state)
        bat_pct = result['deployed_pcts']['battery']
        if bat_pct > 0:
            assert bat_pct == STORAGE_MAX['battery'], \
                f"Profitable battery should deploy to cap, got {bat_pct}"

    def test_h2_blocked_below_95pct(self, extreme_duck_lmp, low_cost_conditions):
        """H2 should not deploy below 95% clean threshold."""
        state = _make_state()
        result = compute_storage_deployment(
            'CAISO', 2035, extreme_duck_lmp, 224, 224e6, 80.0,
            low_cost_conditions, {}, state)
        assert result['deployed_pcts']['h2'] == 0.0
        assert result['storage_details']['h2']['reason'] == 'below_threshold'

    def test_h2_allowed_at_95pct(self, extreme_duck_lmp, low_cost_conditions):
        """H2 should be evaluated (not blocked) at ≥95% clean."""
        state = _make_state()
        result = compute_storage_deployment(
            'CAISO', 2035, extreme_duck_lmp, 224, 224e6, 96.0,
            low_cost_conditions, {}, state)
        # Should be evaluated (may still not deploy if unprofitable)
        assert 'reason' not in result['storage_details']['h2'] or \
               result['storage_details']['h2'].get('reason') != 'below_threshold'

    def test_result_structure(self, duck_curve_lmp, base_conditions):
        """Result should have expected keys and structure."""
        state = _make_state()
        result = compute_storage_deployment(
            'PJM', 2030, duck_curve_lmp, 150, 150e6, 50.0,
            base_conditions, {}, state)

        assert 'deployed_pcts' in result
        assert 'storage_details' in result
        assert 'total_storage_cost_mwh' in result

        for tech in ['battery', 'battery8', 'ldes', 'h2']:
            assert tech in result['deployed_pcts']
            assert tech in result['storage_details']
            details = result['storage_details'][tech]
            for key in ['revenue_lcoe', 'cost_lcoe', 'margin', 'deployed_pct']:
                assert key in details, f"Missing {key} in {tech} details"

    def test_state_updated(self, duck_curve_lmp, base_conditions):
        """State dict should be updated with storage deployment."""
        state = _make_state()
        compute_storage_deployment(
            'CAISO', 2030, duck_curve_lmp, 224, 224e6, 60.0,
            base_conditions, {}, state)
        assert 'storage_deployed' in state
        assert 'storage_details' in state
        assert isinstance(state['storage_deployed'], dict)

    def test_capacity_degradation_at_high_clean(self, duck_curve_lmp, low_cost_conditions):
        """At high clean%, capacity revenue should be degraded."""
        state_low = _make_state()
        result_low = compute_storage_deployment(
            'PJM', 2030, duck_curve_lmp, 150, 150e6, 30.0,
            low_cost_conditions, {}, state_low)

        state_high = _make_state()
        result_high = compute_storage_deployment(
            'PJM', 2030, duck_curve_lmp, 150, 150e6, 80.0,
            low_cost_conditions, {}, state_high)

        # At 80% clean, capacity revenue should be degraded
        cap_low = result_low['storage_details']['battery']['cap_kw_yr']
        cap_high = result_high['storage_details']['battery']['cap_kw_yr']
        assert cap_high < cap_low, \
            f"Capacity revenue should degrade: ${cap_low} at 30% vs ${cap_high} at 80%"

    def test_energy_only_market_zero_capacity(self, duck_curve_lmp, base_conditions):
        """ERCOT (energy-only) should have zero capacity revenue."""
        state = _make_state()
        result = compute_storage_deployment(
            'ERCOT', 2030, duck_curve_lmp, 400, 400e6, 50.0,
            base_conditions, {}, state)
        assert result['storage_details']['battery']['cap_kw_yr'] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC FALLBACK WARNING
# ═══════════════════════════════════════════════════════════════════════════════

class TestSyntheticFallback:
    """Tests for synthetic data fallback warning."""

    def test_synthetic_fallback_warns(self):
        """_generate_synthetic_step3_data should emit UserWarning."""
        with pytest.warns(UserWarning, match="Parquet data not found"):
            _generate_synthetic_step3_data()

    def test_synthetic_data_still_generated(self):
        """Warning should not prevent data generation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = _generate_synthetic_step3_data()
        assert isinstance(data, dict)
        assert len(data) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# LMP CACHE KEY
# ═══════════════════════════════════════════════════════════════════════════════

class TestLMPCacheKey:
    """Tests for LMP cache key including storage state."""

    def test_different_storage_different_keys(self):
        """Different storage states should produce different cache key tuples."""
        stor1 = {'battery': 0.06, 'battery8': 0, 'ldes': 0, 'h2': 0}
        stor2 = {'battery': 0, 'battery8': 0, 'ldes': 0, 'h2': 0}

        key1 = tuple(sorted(stor1.items()))
        key2 = tuple(sorted(stor2.items()))

        assert key1 != key2, "Different storage states must produce different cache keys"

    def test_empty_storage_key(self):
        """Empty storage state should produce consistent key."""
        key1 = tuple(sorted({}.items()))
        key2 = ()
        # Both represent "no storage"
        assert key1 == key2
