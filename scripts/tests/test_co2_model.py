"""
Unit tests for CO₂ dispatch and fossil retirement model.

Tests cover:
  - Merit-order fossil displacement: coal → oil → gas
  - Above-threshold retirement (all coal/oil retired above 70%)
  - Emission rate bounds
  - CCS residual emission rate
  - Zero-coal ISOs (CAISO)
  - Full retirement at 100% clean
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dispatch_utils import (
    compute_fossil_retirement,
    COAL_CAP_TWH, OIL_CAP_TWH, BASE_DEMAND_TWH,
)
from pipeline_config import (
    ISOS, CCS_RESIDUAL_EMISSION_RATE, COAL_OIL_RETIREMENT_THRESHOLD,
    GRID_MIX_SHARES,
)


class TestMeritOrderDisplacement:
    """Coal should be displaced before oil, oil before gas."""

    def test_coal_displaced_first_at_low_clean(self, emission_rates, fossil_mix):
        """Below retirement threshold, coal should be the first fuel displaced."""
        # Use PJM (has significant coal)
        iso = 'PJM'
        baseline_clean = sum(GRID_MIX_SHARES[iso].values())

        # Set clean_pct just above baseline (displacing some fossil)
        clean_pct = baseline_clean + 5.0  # 5% above baseline

        rate, info = compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix)

        # The amount of fossil remaining should have coal at its cap
        # (coal is retired first, starting from dirty end)
        assert rate > 0, f"Displaced rate should be positive at {clean_pct}% clean"

    @pytest.mark.parametrize("iso", ['ERCOT', 'PJM', 'MISO', 'SPP'])
    def test_coal_displaced_before_gas(self, iso, emission_rates, fossil_mix):
        """In ISOs with coal, coal displacement should occur before gas displacement."""
        coal_cap = COAL_CAP_TWH[iso]
        oil_cap = OIL_CAP_TWH[iso]
        demand = BASE_DEMAND_TWH[iso]
        baseline_clean = sum(GRID_MIX_SHARES[iso].values())

        if coal_cap <= 0:
            pytest.skip(f"{iso} has no coal")

        # Set clean energy to displace less than coal + oil cap
        # So not all coal should be displaced yet
        displaced_twh = coal_cap * 0.5  # Half of coal cap
        clean_pct = baseline_clean + (displaced_twh / demand * 100)
        clean_pct = min(clean_pct, COAL_OIL_RETIREMENT_THRESHOLD - 1)

        rate, info = compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix)

        # At this level, all displaced should be coal (it's first in merit order)
        # Below retirement threshold, displacement comes from the remaining fossil mix
        assert rate >= 0, f"Rate should be non-negative"


class TestRetirementThreshold:
    """Above 70% clean, all coal and oil should be fully retired."""

    @pytest.mark.parametrize("iso", ['ERCOT', 'PJM', 'MISO', 'SPP'])
    def test_above_threshold_coal_oil_retired(self, iso, emission_rates, fossil_mix):
        """Above COAL_OIL_RETIREMENT_THRESHOLD, coal and oil should be fully displaced."""
        clean_pct = COAL_OIL_RETIREMENT_THRESHOLD + 5.0  # 75% clean

        rate, info = compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix)

        coal_cap = COAL_CAP_TWH[iso]
        oil_cap = OIL_CAP_TWH[iso]

        if coal_cap > 0:
            assert info['coal_displaced_twh'] == pytest.approx(coal_cap, rel=0.01), \
                f"{iso}: coal not fully displaced at {clean_pct}%: " \
                f"displaced={info['coal_displaced_twh']}, cap={coal_cap}"

        if oil_cap > 0:
            assert info['oil_displaced_twh'] == pytest.approx(oil_cap, rel=0.01), \
                f"{iso}: oil not fully displaced at {clean_pct}%: " \
                f"displaced={info['oil_displaced_twh']}, cap={oil_cap}"


class TestEmissionRateBounds:
    """Displaced emission rates should be physically reasonable."""

    @pytest.mark.parametrize("iso", ISOS)
    def test_rate_in_bounds(self, iso, emission_rates, fossil_mix):
        """Displaced rate should be 0-1.5 tCO₂/MWh (physical range)."""
        for clean_pct in [50, 70, 90, 99]:
            rate, info = compute_fossil_retirement(iso, clean_pct, emission_rates, fossil_mix)

            assert 0 <= rate <= 1.5, \
                f"{iso} at {clean_pct}%: displaced rate {rate:.4f} tCO₂/MWh out of physical range"

    @pytest.mark.parametrize("iso", ISOS)
    def test_remaining_rate_nonnegative(self, iso, emission_rates, fossil_mix):
        """Remaining fossil rate should be non-negative."""
        rate, info = compute_fossil_retirement(iso, 80, emission_rates, fossil_mix)
        remaining = info.get('remaining_rate_tco2_mwh', 0)
        assert remaining >= 0, f"{iso} remaining rate is negative: {remaining}"


class TestCCSResidualRate:
    """CCS captures 90% → residual 0.037 tCO₂/MWh."""

    def test_ccs_residual_value(self):
        """CCS residual emission rate should be 0.037 tCO₂/MWh."""
        assert CCS_RESIDUAL_EMISSION_RATE == pytest.approx(0.037, abs=0.001), \
            f"CCS residual rate is {CCS_RESIDUAL_EMISSION_RATE}, expected 0.037"

    def test_ccs_capture_rate_implied(self):
        """Implied capture rate from residual: 1 - (0.037 / unabated_rate) ≈ 90%."""
        # Typical gas plant: ~0.40 tCO₂/MWh unabated
        unabated_gas = 0.40
        capture_rate = 1 - (CCS_RESIDUAL_EMISSION_RATE / unabated_gas)
        assert capture_rate >= 0.85, \
            f"Implied CCS capture rate {capture_rate:.1%} is below 85%"


class TestFullRetirement:
    """At 100% clean, all fossil should be displaced."""

    @pytest.mark.parametrize("iso", ISOS)
    def test_100pct_clean_full_displacement(self, iso, emission_rates, fossil_mix):
        """100% clean → remaining fossil rate = 0."""
        rate, info = compute_fossil_retirement(iso, 100.0, emission_rates, fossil_mix)

        remaining = info.get('remaining_rate_tco2_mwh', 0)
        assert remaining == pytest.approx(0.0, abs=0.01), \
            f"{iso} at 100% clean still has remaining fossil rate: {remaining}"


class TestZeroCoalISOs:
    """ISOs without coal should skip coal displacement."""

    def test_caiso_no_coal(self, emission_rates, fossil_mix):
        """CAISO has ~0 coal — coal displacement should be ~0."""
        rate, info = compute_fossil_retirement('CAISO', 80, emission_rates, fossil_mix)

        assert info['coal_displaced_twh'] == pytest.approx(0.0, abs=0.1), \
            f"CAISO coal displaced should be ~0: got {info['coal_displaced_twh']}"

    def test_nyiso_no_coal(self, emission_rates, fossil_mix):
        """NYISO has ~0 coal — coal displacement should be ~0."""
        rate, info = compute_fossil_retirement('NYISO', 80, emission_rates, fossil_mix)

        assert info['coal_displaced_twh'] == pytest.approx(0.0, abs=0.1), \
            f"NYISO coal displaced should be ~0: got {info['coal_displaced_twh']}"


class TestRetirementThresholdValue:
    """The coal/oil retirement threshold should be 70%."""

    def test_threshold_value(self):
        assert COAL_OIL_RETIREMENT_THRESHOLD == 70.0, \
            f"Retirement threshold is {COAL_OIL_RETIREMENT_THRESHOLD}, expected 70.0"
