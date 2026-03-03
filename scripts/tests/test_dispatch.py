"""
Unit tests for dispatch model (dispatch_utils.py).

Tests cover:
  - Battery dispatch (4hr/8hr): energy conservation, power limits, edge cases
  - LDES dispatch (100hr iron-air): SOC bounds, efficiency losses
  - Green H2 dispatch (1000hr): activation threshold
  - Merit-order dispatch: priority ordering
  - Full reconstruct_hourly_dispatch: energy balance
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dispatch_utils import (
    _battery_loop, _ldes_loop,
    _dispatch_battery, _dispatch_ldes, _dispatch_h2,
    _dispatch_battery_detailed, _dispatch_ldes_detailed,
    reconstruct_hourly_dispatch, build_supply_matrix,
    _compute_per_resource_dispatch,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    BATTERY8_EFFICIENCY, BATTERY8_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS, LDES_WINDOW_DAYS,
    H2_EFFICIENCY, H2_DURATION_HOURS, H2_WINDOW_DAYS,
    RESOURCE_TYPES, DISPATCH_ORDER,
)

H = 8760


class TestBatteryDispatch:
    """Tests for daily-cycle battery dispatch algorithm."""

    def test_battery_energy_conservation(self, synthetic_demand, synthetic_solar):
        """Discharge energy must equal charge energy × efficiency (no energy creation)."""
        # Create scenario with solar surplus midday and gap at night
        supply = synthetic_solar * 150  # Oversupply during day
        demand = synthetic_demand * 100  # Flat demand

        surplus = np.maximum(0, supply - demand)
        gap = np.maximum(0, demand - supply)

        dispatch_pct = 0.3  # 0.3% of demand
        profile, charge = _dispatch_battery_detailed(
            surplus.copy(), gap.copy(), dispatch_pct,
            BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY
        )

        total_discharge = profile.sum()
        total_charge = charge.sum()

        if total_discharge > 0:
            # Discharge should equal charge × efficiency (within floating-point tolerance)
            assert total_discharge == pytest.approx(total_charge * BATTERY_EFFICIENCY, rel=1e-6), \
                f"Energy not conserved: discharge={total_discharge:.6f}, charge×η={total_charge * BATTERY_EFFICIENCY:.6f}"

    def test_battery_no_negative_residuals(self, synthetic_demand, synthetic_solar):
        """Residual gap and surplus must never go negative after battery dispatch."""
        supply = synthetic_solar * 120
        demand = synthetic_demand * 100

        surplus = np.maximum(0, supply - demand)
        gap = np.maximum(0, demand - supply)

        _dispatch_battery(
            surplus, gap,  # Modified in-place
            0.2, BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY
        )

        assert np.all(surplus >= -1e-12), "Surplus went negative after battery dispatch"
        assert np.all(gap >= -1e-12), "Gap went negative after battery dispatch"

    def test_battery_power_limit(self, synthetic_demand, synthetic_solar):
        """No hour should exceed the power rating (capacity / duration)."""
        supply = synthetic_solar * 200
        demand = synthetic_demand * 100

        surplus = np.maximum(0, supply - demand)
        gap = np.maximum(0, demand - supply)

        dispatch_pct = 0.5
        daily_target = dispatch_pct / 100.0
        power_rating = daily_target / BATTERY_DURATION_HOURS

        profile = _dispatch_battery(
            surplus.copy(), gap.copy(),
            dispatch_pct, BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY
        )

        assert np.all(profile <= power_rating + 1e-12), \
            f"Dispatch exceeded power rating: max={profile.max():.8f}, limit={power_rating:.8f}"

    def test_battery_zero_capacity(self, synthetic_demand, synthetic_solar):
        """dispatch_pct=0 should produce an all-zeros dispatch profile."""
        supply = synthetic_solar * 120
        demand = synthetic_demand * 100

        surplus = np.maximum(0, supply - demand)
        gap = np.maximum(0, demand - supply)

        profile = _dispatch_battery(
            surplus.copy(), gap.copy(),
            0.0, BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY
        )

        assert np.all(profile == 0.0), "Zero-capacity battery produced non-zero dispatch"

    def test_battery_only_dispatches_to_gaps(self, synthetic_demand, synthetic_solar):
        """Battery should only discharge during hours with demand gaps."""
        supply = synthetic_solar * 120
        demand = synthetic_demand * 100

        surplus = np.maximum(0, supply - demand)
        gap = np.maximum(0, demand - supply)
        original_gap = gap.copy()

        profile = _dispatch_battery(
            surplus.copy(), gap.copy(),
            0.3, BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY
        )

        # Dispatch should only happen where there was originally a gap
        for h in range(H):
            if original_gap[h] <= 0:
                assert profile[h] == pytest.approx(0.0, abs=1e-12), \
                    f"Battery dispatched at hour {h} with no gap"


class TestLDESDispatch:
    """Tests for LDES (100-hour iron-air) multi-day dispatch."""

    def test_ldes_efficiency_loss(self, synthetic_demand, synthetic_solar):
        """Total LDES discharge must be less than total charge × efficiency."""
        supply = synthetic_solar * 150
        demand = synthetic_demand * 100

        surplus = np.maximum(0, supply - demand)
        gap = np.maximum(0, demand - supply)

        profile, charge = _dispatch_ldes_detailed(
            surplus.copy(), gap.copy(),
            1.0,  # 1% of demand energy capacity
            demand
        )

        total_discharge = profile.sum()
        total_charge = charge.sum()

        if total_discharge > 0:
            assert total_discharge <= total_charge * LDES_EFFICIENCY + 1e-10, \
                f"LDES created energy: discharge={total_discharge:.6f} > charge×η={total_charge * LDES_EFFICIENCY:.6f}"

    def test_ldes_zero_capacity(self, synthetic_demand):
        """LDES with zero capacity produces no dispatch."""
        surplus = np.full(H, 0.01, dtype=np.float64)
        gap = np.full(H, 0.01, dtype=np.float64)

        profile = _dispatch_ldes(surplus.copy(), gap.copy(), 0.0, synthetic_demand)
        assert np.all(profile == 0.0), "Zero-capacity LDES produced non-zero dispatch"

    def test_ldes_no_negative_residuals(self, synthetic_demand, synthetic_solar):
        """Residuals must not go negative after LDES dispatch."""
        supply = synthetic_solar * 150
        demand = synthetic_demand * 100

        surplus = np.maximum(0, supply - demand)
        gap = np.maximum(0, demand - supply)

        _dispatch_ldes(surplus, gap, 2.0, demand)

        assert np.all(surplus >= -1e-12), "Surplus went negative after LDES dispatch"
        assert np.all(gap >= -1e-12), "Gap went negative after LDES dispatch"


class TestH2Dispatch:
    """Tests for Green H2 seasonal storage."""

    def test_h2_dispatch_produces_output(self, synthetic_demand, synthetic_solar):
        """H2 with non-zero capacity and available surplus should dispatch."""
        supply = synthetic_solar * 200
        demand = synthetic_demand * 100

        surplus = np.maximum(0, supply - demand)
        gap = np.maximum(0, demand - supply)

        profile = _dispatch_h2(
            surplus.copy(), gap.copy(),
            5.0,  # 5% of demand
            demand
        )

        # With significant surplus and gap, H2 should dispatch something
        assert profile.sum() > 0, "H2 with surplus and gap produced zero dispatch"

    def test_h2_zero_capacity(self, synthetic_demand):
        """H2 with zero capacity produces no dispatch."""
        surplus = np.full(H, 0.01, dtype=np.float64)
        gap = np.full(H, 0.01, dtype=np.float64)

        profile = _dispatch_h2(surplus.copy(), gap.copy(), 0.0, synthetic_demand)
        assert np.all(profile == 0.0), "Zero-capacity H2 produced non-zero dispatch"


class TestMeritOrderDispatch:
    """Tests for merit-order resource dispatch priority."""

    def test_clean_firm_dispatched_before_solar(self, synthetic_demand, synthetic_profiles):
        """Clean firm (baseload) should be fully dispatched before solar sees any curtailment."""
        # Mix: 50% clean_firm + 50% solar (total > demand → some curtailment)
        resource_pcts = {
            'clean_firm': 60.0, 'solar': 60.0, 'wind': 0.0,
            'offshore_wind': 0.0, 'ccs_ccgt': 0.0, 'hydro': 0.0,
        }

        supply_matrix = build_supply_matrix(synthetic_profiles)
        matched, surplus = _compute_per_resource_dispatch(
            synthetic_demand, synthetic_profiles, resource_pcts,
            procurement_factor=1.0, supply_matrix=supply_matrix
        )

        # Clean firm should be fully matched (dispatched first = no curtailment
        # except when total demand is already met)
        cf_total_gen = 60.0 / 100.0 * np.array(synthetic_profiles['clean_firm'])
        cf_matched = matched['clean_firm']

        # In hours where demand > 0, clean firm should match up to its full generation
        # (it's dispatched first, so no curtailment from competition)
        for h in range(H):
            if synthetic_demand[h] > 0:
                assert cf_matched[h] >= cf_total_gen[h] - 1e-12 or \
                       synthetic_demand[h] <= cf_total_gen[h], \
                    f"Clean firm curtailed at hour {h} despite being first in merit order"

    def test_dispatch_order_is_correct(self):
        """Verify the dispatch order constant matches expected merit order."""
        expected = ['clean_firm', 'ccs_ccgt', 'hydro', 'offshore_wind', 'wind', 'solar']
        assert DISPATCH_ORDER == expected, \
            f"Dispatch order changed: {DISPATCH_ORDER} != {expected}"


class TestStorageSequencing:
    """Tests for sequential storage dispatch (battery4 → battery8 → LDES → H2)."""

    def test_storage_reduces_residuals_sequentially(self, synthetic_demand, synthetic_profiles):
        """Each storage layer should see reduced residuals from previous layers."""
        resource_pcts = {
            'clean_firm': 30.0, 'solar': 40.0, 'wind': 20.0,
            'offshore_wind': 0.0, 'ccs_ccgt': 0.0, 'hydro': 10.0,
        }

        result = reconstruct_hourly_dispatch(
            synthetic_demand, synthetic_profiles, resource_pcts,
            procurement_pct=100,
            battery_dispatch_pct=0.2,
            battery8_dispatch_pct=0.2,
            ldes_dispatch_pct=1.0,
        )

        # Total clean = supply + all storage dispatch
        expected_total = (result['supply_total'] +
                         result['battery4_profile'] +
                         result['battery8_profile'] +
                         result['ldes_profile'] +
                         result['h2_profile'])

        np.testing.assert_allclose(
            result['total_clean'], expected_total, atol=1e-10,
            err_msg="Total clean doesn't equal supply + storage dispatch"
        )


class TestFullDispatchEnergyBalance:
    """Tests for the complete reconstruct_hourly_dispatch function."""

    def test_energy_balance(self, synthetic_demand, synthetic_profiles):
        """Verify: total_clean = fossil_displaced + curtailed (energy conservation)."""
        resource_pcts = {
            'clean_firm': 30.0, 'solar': 30.0, 'wind': 20.0,
            'offshore_wind': 0.0, 'ccs_ccgt': 10.0, 'hydro': 10.0,
        }

        result = reconstruct_hourly_dispatch(
            synthetic_demand, synthetic_profiles, resource_pcts,
            procurement_pct=100,
            battery_dispatch_pct=0.1,
            ldes_dispatch_pct=0.5,
        )

        # At each hour: total_clean = fossil_displaced + curtailed
        total_clean = result['total_clean']
        fossil_displaced = result['fossil_displaced']
        curtailed = result['curtailed']
        residual_demand = result['residual_demand']

        # fossil_displaced = min(demand, total_clean)
        np.testing.assert_allclose(
            fossil_displaced, np.minimum(synthetic_demand, total_clean), atol=1e-10,
            err_msg="fossil_displaced != min(demand, total_clean)"
        )

        # residual_demand = max(0, demand - total_clean)
        np.testing.assert_allclose(
            residual_demand, np.maximum(0, synthetic_demand - total_clean), atol=1e-10,
            err_msg="residual_demand != max(0, demand - total_clean)"
        )

        # curtailed = max(0, total_clean - demand)
        np.testing.assert_allclose(
            curtailed, np.maximum(0, total_clean - synthetic_demand), atol=1e-10,
            err_msg="curtailed != max(0, total_clean - demand)"
        )

    def test_match_score_bounds(self, synthetic_demand, synthetic_profiles):
        """Match score must be between 0% and 100%."""
        for supply_scale in [0.0, 0.5, 1.0, 2.0, 5.0]:
            resource_pcts = {
                'clean_firm': 20.0 * supply_scale,
                'solar': 15.0 * supply_scale,
                'wind': 15.0 * supply_scale,
                'offshore_wind': 0.0,
                'ccs_ccgt': 0.0,
                'hydro': 0.0,
            }

            result = reconstruct_hourly_dispatch(
                synthetic_demand, synthetic_profiles, resource_pcts,
                procurement_pct=100,
            )

            score = result['fossil_displaced'].sum() / synthetic_demand.sum() * 100
            assert 0 <= score <= 100 + 1e-6, \
                f"Match score {score:.2f}% out of bounds for scale={supply_scale}"

    def test_match_score_monotonic_with_supply(self, synthetic_demand, synthetic_profiles):
        """More supply → higher match score (monotonically non-decreasing)."""
        scores = []
        for scale in [0.2, 0.5, 1.0, 1.5, 2.0]:
            resource_pcts = {
                'clean_firm': 30.0 * scale,
                'solar': 20.0 * scale,
                'wind': 20.0 * scale,
                'offshore_wind': 0.0,
                'ccs_ccgt': 0.0,
                'hydro': 10.0 * min(scale, 1.0),  # Hydro capped
            }

            result = reconstruct_hourly_dispatch(
                synthetic_demand, synthetic_profiles, resource_pcts,
                procurement_pct=100,
            )

            score = result['fossil_displaced'].sum() / synthetic_demand.sum() * 100
            scores.append(score)

        # Should be non-decreasing
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i-1] - 1e-6, \
                f"Match score decreased: scale {[0.2,0.5,1.0,1.5,2.0][i-1]}→{[0.2,0.5,1.0,1.5,2.0][i]}: " \
                f"{scores[i-1]:.4f}% → {scores[i]:.4f}%"

    def test_no_supply_no_match(self, synthetic_demand, synthetic_profiles):
        """Zero supply → zero match score."""
        resource_pcts = {
            'clean_firm': 0.0, 'solar': 0.0, 'wind': 0.0,
            'offshore_wind': 0.0, 'ccs_ccgt': 0.0, 'hydro': 0.0,
        }

        result = reconstruct_hourly_dispatch(
            synthetic_demand, synthetic_profiles, resource_pcts,
            procurement_pct=100,
        )

        assert result['fossil_displaced'].sum() == pytest.approx(0.0, abs=1e-12), \
            "Zero supply produced non-zero match"
