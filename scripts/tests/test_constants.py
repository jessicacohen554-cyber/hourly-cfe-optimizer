"""
Unit tests for pipeline configuration constants (pipeline_config.py).

Tests cover:
  - All 7 ISOs present in every configuration dict
  - LCOE values are positive
  - Efficiency parameters in valid range (0, 1]
  - Demand totals are reasonable (~4000 TWh US total)
  - Threshold definitions are complete and ordered
  - Cross-module constant consistency (pipeline_config vs dispatch_utils vs step1)
  - Wright's Law learning curve parameter validation
  - Dispatch-specific constant completeness (HYDRO_CAPS, COAL_CAP_TWH, etc.)
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
    # Dispatch-specific constants (migrated from dispatch_utils)
    HYDRO_CAPS, COAL_CAP_TWH, OIL_CAP_TWH,
    NUCLEAR_SHARE_OF_CLEAN_FIRM, NUCLEAR_MONTHLY_CF,
    # Wright's Law constants (migrated from procurement_utils)
    WRIGHT_CUMULATIVE_GW_2025, WRIGHT_LEARNING_RATE, WRIGHT_BACKGROUND_GW,
    # Learning curve functions
    learning_fraction, threshold_learning_fraction,
    LEARNING_PARAMS, LEARNING_EXPONENT, THRESHOLD_TARGET_YEARS,
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
        ("HYDRO_CAPS", HYDRO_CAPS),
        ("COAL_CAP_TWH", COAL_CAP_TWH),
        ("OIL_CAP_TWH", OIL_CAP_TWH),
        ("NUCLEAR_SHARE_OF_CLEAN_FIRM", NUCLEAR_SHARE_OF_CLEAN_FIRM),
        ("NUCLEAR_MONTHLY_CF", NUCLEAR_MONTHLY_CF),
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


# ============================================================================
# CROSS-MODULE CONSTANT CONSISTENCY
# ============================================================================
# Step 1 intentionally isolates its constants for physics engine stability
# (multi-hour runtime, can't risk import failures). These tests catch drift
# between step1_pfs_generator.py and pipeline_config.py at test time without
# coupling Step 1's runtime to pipeline_config imports.

class TestStep1CrossValidation:
    """Verify step1_pfs_generator constants match pipeline_config.

    Step 1 intentionally defines its own constants for runtime isolation
    (see CLAUDE.md "Step 1 special treatment"). These tests catch drift.
    """

    @pytest.fixture(autouse=True)
    def load_step1_constants(self):
        """Import step1 constants without running the generator."""
        import importlib.util
        step1_path = os.path.join(os.path.dirname(__file__), '..', 'step1_pfs_generator.py')
        spec = importlib.util.spec_from_file_location("step1_pfs_generator", step1_path)
        self.step1 = importlib.util.module_from_spec(spec)
        # Only load module-level constants, don't execute main()
        try:
            spec.loader.exec_module(self.step1)
        except (SystemExit, Exception):
            pass  # Module may call sys.exit or fail on missing data; constants still loaded

    def test_isos_match(self):
        """Step 1 ISOS must match pipeline_config ISOS."""
        assert self.step1.ISOS == ISOS, \
            f"Step 1 ISOS {self.step1.ISOS} != pipeline_config {ISOS}"

    def test_offshore_isos_match(self):
        """Step 1 OFFSHORE_ISOS must match pipeline_config."""
        assert set(self.step1.OFFSHORE_ISOS) == set(OFFSHORE_ISOS), \
            f"Step 1 OFFSHORE_ISOS {self.step1.OFFSHORE_ISOS} != pipeline_config {OFFSHORE_ISOS}"

    def test_thresholds_match(self):
        """Step 1 THRESHOLDS must match pipeline_config."""
        assert self.step1.THRESHOLDS == THRESHOLDS, \
            f"Step 1 THRESHOLDS differ from pipeline_config"

    def test_battery_efficiency_match(self):
        """Step 1 battery efficiency must match pipeline_config."""
        assert self.step1.BATTERY_EFFICIENCY == BATTERY_EFFICIENCY
        assert self.step1.BATTERY8_EFFICIENCY == BATTERY8_EFFICIENCY

    def test_ldes_efficiency_match(self):
        """Step 1 LDES efficiency must match pipeline_config."""
        assert self.step1.LDES_EFFICIENCY == LDES_EFFICIENCY

    def test_h2_efficiency_match(self):
        """Step 1 H2 efficiency must match pipeline_config."""
        assert self.step1.H2_EFFICIENCY == H2_EFFICIENCY

    def test_battery_duration_match(self):
        """Step 1 battery durations must match pipeline_config."""
        assert self.step1.BATTERY_DURATION_HOURS == BATTERY_DURATION_HOURS
        assert self.step1.BATTERY8_DURATION_HOURS == BATTERY8_DURATION_HOURS

    def test_ldes_duration_match(self):
        """Step 1 LDES duration and window must match pipeline_config."""
        assert self.step1.LDES_DURATION_HOURS == LDES_DURATION_HOURS
        assert self.step1.LDES_WINDOW_DAYS == LDES_WINDOW_DAYS

    def test_h2_duration_match(self):
        """Step 1 H2 duration and window must match pipeline_config."""
        assert self.step1.H2_DURATION_HOURS == H2_DURATION_HOURS
        assert self.step1.H2_WINDOW_DAYS == H2_WINDOW_DAYS

    def test_hours_match(self):
        """Step 1 H constant must match pipeline_config."""
        assert self.step1.H == H


class TestDispatchUtilsConsistency:
    """Verify dispatch_utils re-exports match pipeline_config originals.

    dispatch_utils.py imports constants from pipeline_config. These tests
    confirm the import chain works and values aren't shadowed.
    """

    def test_hydro_caps_match(self):
        """dispatch_utils HYDRO_CAPS must match pipeline_config."""
        from dispatch_utils import HYDRO_CAPS as du_hydro
        assert du_hydro == HYDRO_CAPS

    def test_coal_cap_match(self):
        """dispatch_utils COAL_CAP_TWH must match pipeline_config."""
        from dispatch_utils import COAL_CAP_TWH as du_coal
        assert du_coal == COAL_CAP_TWH

    def test_oil_cap_match(self):
        """dispatch_utils OIL_CAP_TWH must match pipeline_config."""
        from dispatch_utils import OIL_CAP_TWH as du_oil
        assert du_oil == OIL_CAP_TWH

    def test_nuclear_share_match(self):
        """dispatch_utils NUCLEAR_SHARE_OF_CLEAN_FIRM must match pipeline_config."""
        from dispatch_utils import NUCLEAR_SHARE_OF_CLEAN_FIRM as du_nuc
        assert du_nuc == NUCLEAR_SHARE_OF_CLEAN_FIRM

    def test_nuclear_monthly_cf_match(self):
        """dispatch_utils NUCLEAR_MONTHLY_CF must match pipeline_config."""
        from dispatch_utils import NUCLEAR_MONTHLY_CF as du_cf
        assert du_cf == NUCLEAR_MONTHLY_CF

    def test_isos_match(self):
        """dispatch_utils ISOS must match pipeline_config."""
        from dispatch_utils import ISOS as du_isos
        assert du_isos == ISOS


# ============================================================================
# DISPATCH-SPECIFIC CONSTANTS
# ============================================================================

class TestDispatchConstants:
    """Dispatch-specific constants must be complete and reasonable."""

    @pytest.mark.parametrize("iso", ISOS)
    def test_hydro_caps_positive(self, iso):
        """Hydro caps must be non-negative."""
        assert HYDRO_CAPS[iso] >= 0, f"{iso} HYDRO_CAPS is negative"

    @pytest.mark.parametrize("iso", ISOS)
    def test_coal_cap_nonnegative(self, iso):
        """Coal caps must be non-negative."""
        assert COAL_CAP_TWH[iso] >= 0, f"{iso} COAL_CAP_TWH is negative"

    @pytest.mark.parametrize("iso", ISOS)
    def test_oil_cap_nonnegative(self, iso):
        """Oil caps must be non-negative."""
        assert OIL_CAP_TWH[iso] >= 0, f"{iso} OIL_CAP_TWH is negative"

    @pytest.mark.parametrize("iso", ISOS)
    def test_nuclear_share_in_range(self, iso):
        """Nuclear share of clean firm must be in (0, 1]."""
        share = NUCLEAR_SHARE_OF_CLEAN_FIRM[iso]
        assert 0 < share <= 1.0, f"{iso} nuclear share {share} outside (0, 1]"

    def test_caiso_nuclear_share_below_1(self):
        """CAISO nuclear share < 1.0 (has geothermal too)."""
        assert NUCLEAR_SHARE_OF_CLEAN_FIRM['CAISO'] < 1.0

    @pytest.mark.parametrize("iso", ISOS)
    def test_nuclear_monthly_cf_12_months(self, iso):
        """Nuclear monthly CF must have entries for months 1–12."""
        months = set(NUCLEAR_MONTHLY_CF[iso].keys())
        assert months == set(range(1, 13)), f"{iso} missing months: {set(range(1, 13)) - months}"

    @pytest.mark.parametrize("iso", ISOS)
    def test_nuclear_monthly_cf_range(self, iso):
        """Nuclear monthly CF values must be in [0.5, 1.0]."""
        for month, cf in NUCLEAR_MONTHLY_CF[iso].items():
            assert 0.5 <= cf <= 1.0, f"{iso} month {month} CF {cf} outside [0.5, 1.0]"

    def test_caiso_no_coal(self):
        """CAISO should have 0 coal (last CA coal plant closed 2023)."""
        assert COAL_CAP_TWH['CAISO'] == 0.0

    def test_nyiso_no_coal(self):
        """NYISO should have 0 coal (last NY coal plant retired 2020)."""
        assert COAL_CAP_TWH['NYISO'] == 0.0


# ============================================================================
# WRIGHT'S LAW LEARNING CURVE CONSTANTS
# ============================================================================

class TestWrightsLawConstants:
    """Wright's Law learning curve parameters must be complete and reasonable.

    Reference: Wright, T.P. (1936). "Factors Affecting the Cost of Airplanes."
    Journal of the Aeronautical Sciences, 3(4), 122–128.
    """

    EXPECTED_TECHS = {
        'nuclear', 'ccs', 'ldes', 'h2', 'geothermal',
        'battery', 'battery8', 'solar', 'wind', 'offshore_wind',
    }

    def test_cumulative_gw_all_techs(self):
        """Every technology must have a 2025 cumulative GW baseline."""
        missing = self.EXPECTED_TECHS - set(WRIGHT_CUMULATIVE_GW_2025.keys())
        assert not missing, f"WRIGHT_CUMULATIVE_GW_2025 missing: {missing}"

    def test_cumulative_gw_positive(self):
        """All baselines must be positive (some deployment exists)."""
        for tech, gw in WRIGHT_CUMULATIVE_GW_2025.items():
            assert gw > 0, f"{tech} has 0 or negative cumulative GW"

    def test_learning_rate_all_techs(self):
        """Every technology must have Fast and Slow learning rates."""
        missing = self.EXPECTED_TECHS - set(WRIGHT_LEARNING_RATE.keys())
        assert not missing, f"WRIGHT_LEARNING_RATE missing: {missing}"

    @pytest.mark.parametrize("tech", list(WRIGHT_LEARNING_RATE.keys()))
    def test_learning_rate_range(self, tech):
        """Learning rates must be in [0, 0.5] — 0 for mature techs."""
        for speed in ['Fast', 'Slow']:
            rate = WRIGHT_LEARNING_RATE[tech][speed]
            assert 0 <= rate <= 0.5, \
                f"{tech}/{speed} learning rate {rate} outside [0, 0.5]"

    def test_mature_techs_zero_learning(self):
        """Solar and wind should have ~0 learning rate (mature technology)."""
        for tech in ['solar', 'wind']:
            for speed in ['Fast', 'Slow']:
                assert WRIGHT_LEARNING_RATE[tech][speed] == 0.0, \
                    f"{tech}/{speed} should have 0 learning rate"

    def test_background_gw_all_techs(self):
        """Every technology must have background GW projections."""
        missing = self.EXPECTED_TECHS - set(WRIGHT_BACKGROUND_GW.keys())
        assert not missing, f"WRIGHT_BACKGROUND_GW missing: {missing}"

    @pytest.mark.parametrize("tech", list(WRIGHT_BACKGROUND_GW.keys()))
    def test_background_gw_increasing(self, tech):
        """Background GW must increase from 2035 to 2050."""
        for speed in ['Fast', 'Slow']:
            gw_2035, gw_2050 = WRIGHT_BACKGROUND_GW[tech][speed]
            assert gw_2050 >= gw_2035, \
                f"{tech}/{speed} 2050 GW ({gw_2050}) < 2035 GW ({gw_2035})"


# ============================================================================
# LEARNING FRACTION FUNCTIONS
# ============================================================================

class TestLearningFraction:
    """Learning fraction functions must behave correctly at boundaries."""

    def test_before_foak_returns_zero(self):
        """Before FOAK start year, fraction should be 0."""
        assert learning_fraction(2025, 2030, 2040) == 0.0

    def test_at_foak_returns_zero(self):
        """At FOAK start year, fraction should be 0."""
        assert learning_fraction(2030, 2030, 2040) == 0.0

    def test_at_noak_returns_one(self):
        """At NOAK year, fraction should be 1."""
        assert learning_fraction(2040, 2030, 2040) == 1.0

    def test_after_noak_returns_one(self):
        """After NOAK year, fraction should be 1."""
        assert learning_fraction(2050, 2030, 2040) == 1.0

    def test_midpoint_concave(self):
        """Midpoint fraction should be > 0.5 (concave ramp, exponent 0.6)."""
        mid = learning_fraction(2035, 2030, 2040)
        assert mid > 0.5, f"Midpoint fraction {mid} not > 0.5 (concave ramp)"
        assert mid < 1.0

    def test_monotonically_increasing(self):
        """Fraction should increase monotonically with year."""
        fracs = [learning_fraction(y, 2030, 2040) for y in range(2030, 2041)]
        for i in range(1, len(fracs)):
            assert fracs[i] >= fracs[i-1], \
                f"Not monotonic: year {2030+i} frac {fracs[i]} < {fracs[i-1]}"


class TestThresholdLearningFraction:
    """threshold_learning_fraction must correctly delegate to learning_fraction."""

    def test_scenario_b_at_50pct(self):
        """50% threshold → year 2030 → Scenario B FOAK start → fraction 0."""
        frac = threshold_learning_fraction(50, scenario='B')
        assert frac == 0.0

    def test_scenario_b_at_90pct(self):
        """90% threshold → year 2040 → Scenario B NOAK → fraction 1.0."""
        frac = threshold_learning_fraction(90, scenario='B')
        assert frac == 1.0

    def test_scenario_a_no_deployment_returns_zero(self):
        """Scenario A with no deployment year → always FOAK (fraction 0)."""
        frac = threshold_learning_fraction(90, scenario='A', first_deployment_year=None)
        assert frac == 0.0

    def test_scenario_a_with_deployment(self):
        """Scenario A with deployment year → non-zero fraction after deployment."""
        frac = threshold_learning_fraction(90, scenario='A', first_deployment_year=2030)
        assert frac > 0.0  # 2040 - 2030 = 10yr learning window, year=2040 → NOAK

    def test_learning_exponent_is_concave(self):
        """Learning exponent should be < 1 for concave ramp."""
        assert 0 < LEARNING_EXPONENT < 1.0

    def test_threshold_target_years_complete(self):
        """Every threshold should have a target year."""
        for t in THRESHOLDS:
            assert t in THRESHOLD_TARGET_YEARS, \
                f"Threshold {t} missing from THRESHOLD_TARGET_YEARS"

    def test_threshold_target_years_monotonic(self):
        """Higher thresholds should map to later years."""
        sorted_thresholds = sorted(THRESHOLD_TARGET_YEARS.items())
        for i in range(1, len(sorted_thresholds)):
            t1, y1 = sorted_thresholds[i-1]
            t2, y2 = sorted_thresholds[i]
            assert y2 >= y1, \
                f"Threshold {t2}→{y2} earlier than {t1}→{y1}"
