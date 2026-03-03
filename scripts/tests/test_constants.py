"""
Unit tests for pipeline configuration constants (pipeline_config.py).

Tests cover:
  - All 7 ISOs present in every configuration dict
  - LCOE values are positive
  - Efficiency parameters in valid range (0, 1]
  - Demand totals are reasonable (~4000 TWh US total)
  - Threshold definitions are complete and ordered
  - Cross-module constant consistency (pipeline_config vs dispatch_utils)
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline_config import (
    ISOS, REGIONAL_DEMAND_TWH, GRID_MIX_SHARES,
    CCS_CAP_TWH, HYDRO_CAP_PCT,
    WHOLESALE_PRICES, FUEL_ADJUSTMENTS,
    BATTERY_EFFICIENCY, BATTERY_DURATION_HOURS,
    BATTERY8_EFFICIENCY, BATTERY8_DURATION_HOURS,
    LDES_EFFICIENCY, LDES_DURATION_HOURS, LDES_WINDOW_DAYS,
    H2_EFFICIENCY, H2_DURATION_HOURS, H2_WINDOW_DAYS,
    THRESHOLDS, ACTIVE_THRESHOLDS, COARSE_THRESHOLDS,
    PEAK_DEMAND_MW, GAS_AVAILABILITY_FACTOR,
    PEAK_CAPACITY_CREDITS, RESOURCE_CAPACITY_FACTORS,
    EXISTING_NUCLEAR_GW, UPRATE_CAP_TWH,
    N_SCENARIOS_BASE, N_SCENARIOS_CAISO,
    DISPATCH_ORDER, H,
    OFFSHORE_ISOS, GEOTHERMAL_ISOS,
)


class TestISOCompleteness:
    """All 7 ISOs must be present in every configuration dictionary."""

    EXPECTED_ISOS = {'CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP'}

    @pytest.mark.parametrize("dict_name,config_dict", [
        ("REGIONAL_DEMAND_TWH", REGIONAL_DEMAND_TWH),
        ("GRID_MIX_SHARES", GRID_MIX_SHARES),
        ("CCS_CAP_TWH", CCS_CAP_TWH),
        ("HYDRO_CAP_PCT", HYDRO_CAP_PCT),
        ("WHOLESALE_PRICES", WHOLESALE_PRICES),
        ("FUEL_ADJUSTMENTS", FUEL_ADJUSTMENTS),
        ("PEAK_DEMAND_MW", PEAK_DEMAND_MW),
        ("GAS_AVAILABILITY_FACTOR", GAS_AVAILABILITY_FACTOR),
        ("EXISTING_NUCLEAR_GW", EXISTING_NUCLEAR_GW),
        ("UPRATE_CAP_TWH", UPRATE_CAP_TWH),
    ])
    def test_all_isos_present(self, dict_name, config_dict):
        """Every ISO must have an entry."""
        missing = self.EXPECTED_ISOS - set(config_dict.keys())
        assert not missing, f"{dict_name} missing ISOs: {missing}"

    @pytest.mark.parametrize("dict_name,config_dict", [
        ("RESOURCE_CAPACITY_FACTORS", RESOURCE_CAPACITY_FACTORS),
    ])
    def test_nested_dicts_have_all_isos(self, dict_name, config_dict):
        """Nested dicts (resource → ISO) must have all ISOs."""
        for resource, iso_dict in config_dict.items():
            missing = self.EXPECTED_ISOS - set(iso_dict.keys())
            assert not missing, f"{dict_name}[{resource}] missing ISOs: {missing}"


class TestEfficiencyBounds:
    """Storage efficiencies must be in (0, 1]."""

    @pytest.mark.parametrize("name,value", [
        ("Battery 4hr", BATTERY_EFFICIENCY),
        ("Battery 8hr", BATTERY8_EFFICIENCY),
        ("LDES 100hr", LDES_EFFICIENCY),
        ("Green H2", H2_EFFICIENCY),
    ])
    def test_efficiency_in_range(self, name, value):
        assert 0 < value <= 1.0, f"{name} efficiency {value} outside (0, 1]"

    def test_battery_most_efficient(self):
        """Battery should be most efficient, H2 least."""
        assert BATTERY_EFFICIENCY > LDES_EFFICIENCY > H2_EFFICIENCY, \
            "Efficiency ordering wrong: battery > LDES > H2"

    def test_battery4_equals_battery8_efficiency(self):
        """Both battery durations use same RTE (Li-ion technology)."""
        assert BATTERY_EFFICIENCY == BATTERY8_EFFICIENCY, \
            "4hr and 8hr battery should have same RTE"


class TestStorageDurations:
    """Storage duration parameters must be reasonable."""

    def test_battery_4hr(self):
        assert BATTERY_DURATION_HOURS == 4

    def test_battery_8hr(self):
        assert BATTERY8_DURATION_HOURS == 8

    def test_ldes_100hr(self):
        assert LDES_DURATION_HOURS == 100

    def test_h2_1000hr(self):
        assert H2_DURATION_HOURS == 1000

    def test_ldes_window_7days(self):
        assert LDES_WINDOW_DAYS == 7

    def test_h2_window_30days(self):
        assert H2_WINDOW_DAYS == 30


class TestDemandReasonable:
    """Total US demand across 7 ISOs should be ~2800-4000 TWh."""

    def test_total_demand(self):
        total = sum(REGIONAL_DEMAND_TWH.values())
        assert 2500 <= total <= 4500, \
            f"Total US demand {total:.0f} TWh outside expected range (2500-4500)"

    @pytest.mark.parametrize("iso", ISOS)
    def test_individual_demand_positive(self, iso):
        """Each ISO demand must be positive."""
        assert REGIONAL_DEMAND_TWH[iso] > 0, f"{iso} demand is non-positive"

    def test_pjm_largest(self):
        """PJM should be the largest ISO by demand."""
        pjm = REGIONAL_DEMAND_TWH['PJM']
        for iso in ISOS:
            if iso != 'PJM':
                assert pjm >= REGIONAL_DEMAND_TWH[iso], \
                    f"PJM ({pjm} TWh) not largest: {iso} = {REGIONAL_DEMAND_TWH[iso]} TWh"


class TestThresholds:
    """Threshold definitions must be complete and ordered."""

    def test_21_thresholds(self):
        """Model uses exactly 21 thresholds."""
        assert len(THRESHOLDS) == 21, f"Expected 21 thresholds, got {len(THRESHOLDS)}"

    def test_17_active_thresholds(self):
        """17 active thresholds (≥50%)."""
        assert len(ACTIVE_THRESHOLDS) == 17, f"Expected 17 active, got {len(ACTIVE_THRESHOLDS)}"

    def test_4_coarse_thresholds(self):
        """4 coarse thresholds (<50%)."""
        assert len(COARSE_THRESHOLDS) == 4, f"Expected 4 coarse, got {len(COARSE_THRESHOLDS)}"

    def test_thresholds_sorted(self):
        """Thresholds must be in ascending order."""
        assert THRESHOLDS == sorted(THRESHOLDS), "Thresholds not sorted ascending"

    def test_thresholds_range(self):
        """Thresholds should span 10% to 99.99%."""
        assert THRESHOLDS[0] == 10.0
        assert THRESHOLDS[-1] == 99.99

    def test_active_starts_at_50(self):
        """Active thresholds start at 50%."""
        assert ACTIVE_THRESHOLDS[0] == 50.0

    def test_coarse_below_50(self):
        """All coarse thresholds are below 50%."""
        for t in COARSE_THRESHOLDS:
            assert t < 50.0, f"Coarse threshold {t} is ≥50%"


class TestGridMixShares:
    """Existing grid mix shares must be non-negative and sum to < 100%."""

    @pytest.mark.parametrize("iso", ISOS)
    def test_shares_nonnegative(self, iso):
        """All resource shares must be ≥ 0."""
        for resource, pct in GRID_MIX_SHARES[iso].items():
            assert pct >= 0, f"{iso} {resource} share is negative: {pct}%"

    @pytest.mark.parametrize("iso", ISOS)
    def test_total_below_100(self, iso):
        """Total existing clean share must be < 100% (there's always some fossil)."""
        total = sum(GRID_MIX_SHARES[iso].values())
        assert total < 100.0, f"{iso} total clean share {total}% ≥ 100%"

    @pytest.mark.parametrize("iso", ISOS)
    def test_no_existing_ccs(self, iso):
        """No ISO should have existing CCS-CCGT (technology is new-build only in 2025)."""
        assert GRID_MIX_SHARES[iso].get('ccs_ccgt', 0) == 0, \
            f"{iso} has existing CCS: {GRID_MIX_SHARES[iso]['ccs_ccgt']}%"


class TestScenarioCounts:
    """Scenario count must match toggle combinatorics."""

    def test_base_scenario_count(self):
        """Non-CAISO: 3⁶ × 4 × 2 = 5,832 scenarios."""
        expected = 3**6 * 4 * 2  # Ren×Firm×Batt×LDES×Fuel×CCS (3 each) × Tx(4) × 45Q(2)
        assert N_SCENARIOS_BASE == expected, \
            f"Base scenarios: {N_SCENARIOS_BASE} != {expected}"

    def test_caiso_scenario_count(self):
        """CAISO: 5,832 × 3 = 17,496 (geothermal toggle adds 3x)."""
        assert N_SCENARIOS_CAISO == N_SCENARIOS_BASE * 3, \
            f"CAISO scenarios: {N_SCENARIOS_CAISO} != {N_SCENARIOS_BASE * 3}"


class TestCapacityCredits:
    """Peak capacity credits must be in [0, 1]."""

    @pytest.mark.parametrize("resource,credit", PEAK_CAPACITY_CREDITS.items())
    def test_credit_in_range(self, resource, credit):
        assert 0 <= credit <= 1.0, \
            f"{resource} capacity credit {credit} outside [0, 1]"

    def test_clean_firm_full_credit(self):
        """Nuclear/clean firm should have 100% capacity credit."""
        assert PEAK_CAPACITY_CREDITS['clean_firm'] == 1.0

    def test_solar_partial_credit(self):
        """Solar should have partial capacity credit (low peak contribution)."""
        assert PEAK_CAPACITY_CREDITS['solar'] < 0.5


class TestDispatchOrder:
    """Dispatch order must list all resource types."""

    def test_dispatch_order_complete(self):
        """All resource types should appear in dispatch order."""
        from dispatch_utils import RESOURCE_TYPES
        for r in RESOURCE_TYPES:
            assert r in DISPATCH_ORDER, f"{r} missing from DISPATCH_ORDER"

    def test_clean_firm_first(self):
        """Clean firm (baseload) should be first in merit order."""
        assert DISPATCH_ORDER[0] == 'clean_firm'


class TestHoursConstant:
    """H must be 8760 (non-leap year)."""

    def test_hours_per_year(self):
        assert H == 8760


class TestOffshoreGeothermalISOs:
    """Offshore and geothermal ISO lists."""

    def test_offshore_isos(self):
        """Exactly 4 ISOs have offshore wind."""
        assert set(OFFSHORE_ISOS) == {'NYISO', 'NEISO', 'PJM', 'CAISO'}

    def test_geothermal_caiso_only(self):
        """Only CAISO has geothermal."""
        assert GEOTHERMAL_ISOS == ['CAISO']
