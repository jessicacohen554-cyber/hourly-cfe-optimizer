"""
Physics-level energy conservation tests for the dispatch model.

These tests verify fundamental physical laws:
  - Energy in = energy out + losses (first law of thermodynamics)
  - Battery charge × efficiency ≥ discharge (second law)
  - No negative residuals (conservation of demand)
  - Demand normalization (profile integrity)
  - Match score equals displaced fraction (consistency check)
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dispatch_utils import (
    reconstruct_hourly_dispatch, build_supply_matrix,
    _dispatch_battery_detailed, _dispatch_ldes_detailed, _dispatch_h2_detailed,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    BATTERY8_EFFICIENCY, BATTERY8_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS,
    H2_EFFICIENCY, H2_DURATION_HOURS,
)

H = 8760


class TestDispatchEnergyBalance:
    """Total energy balance across the dispatch system."""

    @pytest.mark.parametrize("battery_pct,ldes_pct", [
        (0.0, 0.0),    # No storage
        (0.2, 0.0),    # Battery only
        (0.0, 1.0),    # LDES only
        (0.2, 1.0),    # Both
        (0.3, 2.0),    # Larger capacities
    ])
    def test_total_clean_equals_components(self, synthetic_demand, synthetic_profiles,
                                           battery_pct, ldes_pct):
        """total_clean = supply_total + battery4 + battery8 + LDES + H2 dispatch."""
        resource_pcts = {
            'clean_firm': 30.0, 'solar': 25.0, 'wind': 20.0,
            'offshore_wind': 0.0, 'ccs_ccgt': 5.0, 'hydro': 10.0,
        }

        result = reconstruct_hourly_dispatch(
            synthetic_demand, synthetic_profiles, resource_pcts,
            procurement_pct=100,
            battery_dispatch_pct=battery_pct,
            battery8_dispatch_pct=battery_pct * 0.5,
            ldes_dispatch_pct=ldes_pct,
        )

        expected = (result['supply_total'] +
                   result['battery4_profile'] +
                   result['battery8_profile'] +
                   result['ldes_profile'] +
                   result['h2_profile'])

        np.testing.assert_allclose(
            result['total_clean'], expected, atol=1e-10,
            err_msg=f"Energy balance violated: battery={battery_pct}%, LDES={ldes_pct}%"
        )

    def test_demand_split_conservation(self, synthetic_demand, synthetic_profiles):
        """At each hour: demand = fossil_displaced + residual_demand."""
        resource_pcts = {
            'clean_firm': 40.0, 'solar': 20.0, 'wind': 15.0,
            'offshore_wind': 0.0, 'ccs_ccgt': 5.0, 'hydro': 10.0,
        }

        result = reconstruct_hourly_dispatch(
            synthetic_demand, synthetic_profiles, resource_pcts,
            procurement_pct=100,
            battery_dispatch_pct=0.2,
            ldes_dispatch_pct=0.5,
        )

        # demand = fossil_displaced + residual_demand
        reconstructed = result['fossil_displaced'] + result['residual_demand']
        np.testing.assert_allclose(
            reconstructed, synthetic_demand, atol=1e-10,
            err_msg="Demand conservation violated: displaced + residual ≠ demand"
        )


class TestBatteryChargeDischargeBalance:
    """Battery must obey thermodynamic losses: discharge ≤ charge × η."""

    def test_battery4_daily_balance(self, synthetic_demand, synthetic_solar):
        """Per day: total discharge ≤ total charge × efficiency."""
        supply = synthetic_solar * 150
        demand = synthetic_demand * 100

        surplus = np.maximum(0, supply - demand)
        gap = np.maximum(0, demand - supply)

        profile, charge = _dispatch_battery_detailed(
            surplus.copy(), gap.copy(), 0.3,
            BATTERY_DURATION_HOURS, BATTERY_EFFICIENCY
        )

        num_days = H // 24
        for day in range(num_days):
            ds, de = day * 24, (day + 1) * 24
            day_discharge = profile[ds:de].sum()
            day_charge = charge[ds:de].sum()

            if day_discharge > 1e-12:
                assert day_discharge <= day_charge * BATTERY_EFFICIENCY + 1e-10, \
                    f"Day {day}: discharge {day_discharge:.8f} > charge×η {day_charge * BATTERY_EFFICIENCY:.8f}"

    def test_ldes_total_balance(self, synthetic_demand, synthetic_solar):
        """Total LDES discharge ≤ total charge × efficiency."""
        supply = synthetic_solar * 150
        demand = synthetic_demand * 100

        surplus = np.maximum(0, supply - demand)
        gap = np.maximum(0, demand - supply)

        profile, charge = _dispatch_ldes_detailed(
            surplus.copy(), gap.copy(), 2.0, demand
        )

        total_discharge = profile.sum()
        total_charge = charge.sum()

        if total_discharge > 1e-12:
            assert total_discharge <= total_charge * LDES_EFFICIENCY + 1e-10, \
                f"LDES violated 2nd law: discharge {total_discharge:.8f} > charge×η {total_charge * LDES_EFFICIENCY:.8f}"


class TestNoNegativeResiduals:
    """Residual demand and curtailed energy must never be negative."""

    @pytest.mark.parametrize("cf_pct,sol_pct", [
        (10, 10),    # Under-supply
        (50, 50),    # Near-match
        (80, 80),    # Over-supply
        (100, 100),  # Heavy over-supply
    ])
    def test_nonnegative_residuals(self, synthetic_demand, synthetic_profiles,
                                    cf_pct, sol_pct):
        """residual_demand ≥ 0 and curtailed ≥ 0 at all 8760 hours."""
        resource_pcts = {
            'clean_firm': cf_pct, 'solar': sol_pct, 'wind': 10.0,
            'offshore_wind': 0.0, 'ccs_ccgt': 0.0, 'hydro': 5.0,
        }

        result = reconstruct_hourly_dispatch(
            synthetic_demand, synthetic_profiles, resource_pcts,
            procurement_pct=100,
            battery_dispatch_pct=0.1,
            ldes_dispatch_pct=0.5,
        )

        assert np.all(result['residual_demand'] >= -1e-12), \
            f"Negative residual demand at cf={cf_pct}%, sol={sol_pct}%: " \
            f"min={result['residual_demand'].min():.10f}"

        assert np.all(result['curtailed'] >= -1e-12), \
            f"Negative curtailment at cf={cf_pct}%, sol={sol_pct}%: " \
            f"min={result['curtailed'].min():.10f}"


class TestDemandNormalization:
    """Normalized demand profiles should sum to 1.0."""

    def test_synthetic_demand_sums_to_one(self, synthetic_demand):
        """Synthetic flat demand profile sums to 1.0."""
        assert synthetic_demand.sum() == pytest.approx(1.0, rel=1e-10), \
            f"Demand profile sum = {synthetic_demand.sum()}, expected 1.0"

    def test_synthetic_solar_sums_to_one(self, synthetic_solar):
        """Synthetic solar profile sums to 1.0."""
        assert synthetic_solar.sum() == pytest.approx(1.0, rel=1e-6), \
            f"Solar profile sum = {synthetic_solar.sum()}, expected 1.0"

    def test_synthetic_wind_sums_to_one(self, synthetic_wind):
        """Synthetic wind profile sums to 1.0."""
        assert synthetic_wind.sum() == pytest.approx(1.0, rel=1e-6), \
            f"Wind profile sum = {synthetic_wind.sum()}, expected 1.0"


class TestMatchScoreConsistency:
    """Match score = fossil_displaced / demand (definition consistency)."""

    @pytest.mark.parametrize("supply_scale", [0.3, 0.6, 1.0, 1.5, 2.5])
    def test_match_score_equals_displaced_fraction(self, synthetic_demand,
                                                     synthetic_profiles, supply_scale):
        """Match score computed from dispatch must equal displaced/demand."""
        resource_pcts = {
            'clean_firm': 30.0 * supply_scale,
            'solar': 20.0 * supply_scale,
            'wind': 15.0 * supply_scale,
            'offshore_wind': 0.0,
            'ccs_ccgt': 5.0 * supply_scale,
            'hydro': 5.0 * min(supply_scale, 1.0),
        }

        result = reconstruct_hourly_dispatch(
            synthetic_demand, synthetic_profiles, resource_pcts,
            procurement_pct=100,
            battery_dispatch_pct=0.1,
            ldes_dispatch_pct=0.3,
        )

        # Score from displaced
        displaced = result['fossil_displaced'].sum()
        demand_total = synthetic_demand.sum()
        score_from_displaced = (displaced / demand_total) * 100

        # Score from 1 - residual
        residual = result['residual_demand'].sum()
        score_from_residual = ((demand_total - residual) / demand_total) * 100

        assert score_from_displaced == pytest.approx(score_from_residual, abs=0.01), \
            f"Score inconsistency: displaced-based={score_from_displaced:.4f}%, " \
            f"residual-based={score_from_residual:.4f}%"
