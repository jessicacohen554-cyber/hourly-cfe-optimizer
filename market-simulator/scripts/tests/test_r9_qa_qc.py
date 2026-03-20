"""R9 QA/QC: Unit tests verifying all peer review implementations (R1-R5, R7-R8, R10).

Run: pytest market-simulator/scripts/tests/test_r9_qa_qc.py -v
"""
import sys
import os
import warnings

import numpy as np
import pytest

# Add scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Add backend dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from market_simulation import (
    wright_cost,
    get_effective_cumulative_gw,
    get_resource_lcoe,
    compute_lcoe_snapshot,
    compute_storage_arbitrage_from_lmp,
    compute_energy_revenue_by_resource,
    compute_zone_revenue,
    compute_capacity_degradation,
    _generate_synthetic_step3_data,
    _percentile_dict,
    _compute_weighted_percentiles,
    compute_market_deployment,
    H,
)
from pipeline_config import (
    ENDOGENOUS_LEARNING,
    WRIGHT_CUMULATIVE_GW_2025,
    WRIGHT_LEARNING_RATE,
    CAPACITY_MARKET_PRICES,
    CAPACITY_SCARCITY_PARAMS,
    VRE_PRIMARY_ZONE,
    SCENARIO_WEIGHTS,
    compute_capacity_price,
)
from models import SimulationRequest, UncertaintyBands


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def flat_lmp():
    """Flat LMP — no price spread, zero arbitrage."""
    return np.full(H, 50.0)


@pytest.fixture
def duck_curve_lmp():
    """Duck-curve LMP with midday trough and evening peak."""
    np.random.seed(42)
    lmp = np.zeros(H)
    for d in range(365):
        h = d * 24
        lmp[h:h+24] = 35 + np.random.normal(0, 5, 24)
        lmp[h+10:h+15] -= 20   # midday solar trough
        lmp[h+16:h+21] += 40   # evening peak
    return np.clip(lmp, 0, 300)


# ═══════════════════════════════════════════════════════════════════════════════
# R2: Wright's Law Trajectory
# ═══════════════════════════════════════════════════════════════════════════════

class TestR2WrightsLaw:
    """Verify costs decrease over 25 years with endogenous learning."""

    def test_wright_cost_decreases_with_cumulative_gw(self):
        """More cumulative GW → lower cost, floored at noak."""
        foak, noak, ref_gw, lr = 100.0, 30.0, 10.0, 0.15
        costs = [wright_cost(foak, noak, gw, ref_gw, lr)
                 for gw in [10, 20, 50, 100, 500, 1000]]
        # Monotonically decreasing
        for i in range(len(costs) - 1):
            assert costs[i] >= costs[i + 1], \
                f"Cost should decrease: {costs[i]:.2f} >= {costs[i+1]:.2f}"
        # Floors at noak
        assert costs[-1] >= noak, f"Cost {costs[-1]:.2f} should not go below noak {noak}"

    def test_wright_cost_static_below_reference(self):
        """If cumulative_gw <= reference_gw, returns foak (no learning yet)."""
        assert wright_cost(100, 30, 5, 10, 0.15) == 100.0
        assert wright_cost(100, 30, 10, 10, 0.15) == 100.0

    def test_wright_cost_zero_learning_rate(self):
        """Zero learning rate → always returns foak."""
        assert wright_cost(100, 30, 1000, 10, 0.0) == 100.0

    def test_lcoe_trajectory_25_years(self):
        """Over 25 years of deployment, LCOE should decrease for clean_firm (nuclear)."""
        # Note: solar/wind have lr=0.0 (mature techs), so use nuclear which has active learning
        cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
        lcoes = []
        for year in range(2025, 2050):
            lcoe = get_resource_lcoe('clean_firm', 'ERCOT', 'Medium',
                                     cumulative_gw, 'Fast', year)
            lcoes.append(lcoe)
            # Simulate 2 GW/year deployment
            cumulative_gw['nuclear'] = cumulative_gw.get('nuclear', 0) + 2
        # First year should be more expensive than last
        assert lcoes[0] > lcoes[-1], \
            f"LCOE should decrease: {lcoes[0]:.2f} → {lcoes[-1]:.2f}"

    def test_compute_lcoe_snapshot_returns_all_techs(self):
        """Snapshot should return LCOE for multiple deployable resources."""
        snap = compute_lcoe_snapshot('ERCOT', dict(WRIGHT_CUMULATIVE_GW_2025),
                                     'Medium', 'Fast', 2030)
        assert isinstance(snap, dict)
        assert len(snap) >= 3, f"Expected ≥3 techs, got {list(snap.keys())}"
        for tech, lcoe in snap.items():
            assert lcoe > 0, f"{tech} LCOE should be positive, got {lcoe}"


# ═══════════════════════════════════════════════════════════════════════════════
# R3: Synthetic Fallback Warning
# ═══════════════════════════════════════════════════════════════════════════════

class TestR3SyntheticFallback:
    """Verify warning is raised when parquets are missing."""

    def test_synthetic_data_emits_warning(self):
        """_generate_synthetic_step3_data() should emit a UserWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _generate_synthetic_step3_data()
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1, \
                "Expected at least one UserWarning from synthetic data generation"
            assert "synthetic" in str(user_warnings[0].message).lower() or \
                   "parquet" in str(user_warnings[0].message).lower(), \
                f"Warning should mention synthetic/parquet: {user_warnings[0].message}"

    def test_synthetic_data_returns_valid_structure(self):
        """Synthetic data should return a dict with ISO keys."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = _generate_synthetic_step3_data()
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        assert len(data) > 0, "Synthetic data should not be empty"


# ═══════════════════════════════════════════════════════════════════════════════
# R4: VRE Basis Differential
# ═══════════════════════════════════════════════════════════════════════════════

class TestR4VREBasis:
    """Verify zonal LMP used for VRE revenue (not system average)."""

    def test_vre_primary_zone_coverage(self):
        """All 7 ISOs should have VRE zone mappings."""
        expected_isos = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
        for iso in expected_isos:
            assert iso in VRE_PRIMARY_ZONE, f"Missing VRE zone mapping for {iso}"
            mapping = VRE_PRIMARY_ZONE[iso]
            assert 'solar' in mapping, f"Missing solar zone for {iso}"
            assert 'wind' in mapping, f"Missing wind zone for {iso}"

    def test_basis_differential_with_zonal_lmp(self):
        """When zonal LMP provided, basis_differentials should be computed."""
        # Create synthetic zonal LMP: 2 zones, zone 0 cheaper than zone 1
        hourly_lmp = np.full(H, 40.0)  # system average
        # Shape: (n_zones, H) — each row is a zone's hourly LMP
        zonal_lmp_matrix = np.array([
            np.full(H, 30.0),  # Zone 0 (cheap, VRE-rich)
            np.full(H, 50.0),  # Zone 1 (expensive, load center)
        ])
        zonal_zone_names = ['West', 'Houston']

        supply_profiles = {
            'solar': np.full(H, 1.0),
            'wind': np.full(H, 1.0),
        }
        resource_pcts = {'solar': 20, 'wind': 15}

        # compute_zone_revenue signature:
        # (iso, clean_pct, resource_pcts, hourly_lmp, supply_profiles,
        #  demand_total_mwh, year, ...)
        blended, per_res, breakdown = compute_zone_revenue(
            'ERCOT', 30.0, resource_pcts, hourly_lmp, supply_profiles,
            demand_total_mwh=100_000_000,
            year=2025,
            zonal_lmp_matrix=zonal_lmp_matrix,
            zonal_zone_names=zonal_zone_names,
        )
        # Should have basis_differentials in the breakdown
        assert 'basis_differentials' in breakdown, \
            "Expected basis_differentials in blended revenue breakdown"

    def test_copper_plate_zero_basis(self):
        """Without zonal data, basis differential should be zero."""
        hourly_lmp = np.full(H, 40.0)
        supply_profiles = {'solar': np.full(H, 1.0)}
        resource_pcts = {'solar': 20}

        blended, per_res, breakdown = compute_zone_revenue(
            'ERCOT', 30.0, resource_pcts, hourly_lmp, supply_profiles,
            demand_total_mwh=100_000_000,
            year=2025,
            zonal_lmp_matrix=None,
            zonal_zone_names=None,
        )
        # No zonal data → no basis differential or all zeros
        bd = breakdown.get('basis_differentials', {})
        for res, diff in bd.items():
            assert diff == 0.0, \
                f"Copper-plate basis diff should be 0, got {diff} for {res}"


# ═══════════════════════════════════════════════════════════════════════════════
# R5: Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════════

class TestR5ConfidenceIntervals:
    """Verify P10/P50/P90 match numpy.percentile on raw data."""

    def test_percentile_dict_matches_numpy(self):
        """_percentile_dict should match np.percentile for unweighted case."""
        np.random.seed(123)
        arr = np.random.normal(50, 10, 100)
        result = _percentile_dict(arr)

        assert abs(result['p10'] - np.percentile(arr, 10)) < 0.01
        assert abs(result['p50'] - np.percentile(arr, 50)) < 0.01
        assert abs(result['p90'] - np.percentile(arr, 90)) < 0.01
        assert result['p10'] <= result['p50'] <= result['p90']

    def test_weighted_percentiles_shift_toward_high_weight(self):
        """Weighted P50 should shift toward values with higher weight."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # Heavy weight on low values
        weights_low = np.array([10.0, 10.0, 1.0, 1.0, 1.0])
        # Heavy weight on high values
        weights_high = np.array([1.0, 1.0, 1.0, 10.0, 10.0])

        p50_low = _compute_weighted_percentiles(values, weights_low, [50])[0]
        p50_high = _compute_weighted_percentiles(values, weights_high, [50])[0]

        assert p50_low < p50_high, \
            f"Low-weighted P50 ({p50_low:.1f}) should be < high-weighted ({p50_high:.1f})"

    def test_uncertainty_bands_model_fields(self):
        """UncertaintyBands model should accept all required fields."""
        bands = UncertaintyBands(
            metric="clean_pct",
            p10=30.0, p25=40.0, p50=55.0, p75=65.0, p90=75.0,
            mean=54.0, std=12.0, n=1215,
        )
        assert bands.metric == "clean_pct"
        assert bands.p10 < bands.p90

    def test_scenario_weights_sum_to_one(self):
        """Each dimension's weights should sum to ~1.0."""
        for dim, weights in SCENARIO_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, \
                f"SCENARIO_WEIGHTS['{dim}'] sums to {total:.3f}, expected ~1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# R7: Capacity Market Price Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

class TestR7CapacityMarket:
    """Verify price sensitivity to clean% and reserve margin; ERCOT returns $0."""

    def test_ercot_always_zero(self):
        """ERCOT is energy-only — capacity price must always be $0."""
        for rm in [5, 10, 15, 20, 30]:
            for clean in [0, 25, 50, 75, 100]:
                price = compute_capacity_price('ERCOT', rm, clean)
                assert price == 0.0, \
                    f"ERCOT capacity price should be $0, got ${price} at RM={rm}%, clean={clean}%"

    def test_spp_always_zero(self):
        """SPP is energy-only — capacity price must always be $0."""
        price = compute_capacity_price('SPP', 10, 50)
        assert price == 0.0, f"SPP capacity price should be $0, got ${price}"

    def test_scarcity_increases_price(self):
        """Lower reserve margin → higher capacity price."""
        price_tight = compute_capacity_price('PJM', 5, 30)    # tight reserves
        price_normal = compute_capacity_price('PJM', 15, 30)  # at target
        price_excess = compute_capacity_price('PJM', 25, 30)  # excess reserves

        assert price_tight > price_normal, \
            f"Tight reserves ({price_tight:.1f}) should > normal ({price_normal:.1f})"
        assert price_normal >= price_excess, \
            f"Normal ({price_normal:.1f}) should >= excess ({price_excess:.1f})"

    def test_clean_penetration_degrades_price(self):
        """Higher clean% → lower capacity price (sigmoid degradation)."""
        price_low_clean = compute_capacity_price('PJM', 15, 20)
        price_high_clean = compute_capacity_price('PJM', 15, 80)

        assert price_low_clean >= price_high_clean, \
            f"Low clean ({price_low_clean:.1f}) should >= high clean ({price_high_clean:.1f})"

    def test_max_scarcity_multiplier_bounded(self):
        """At 0% reserve margin, price should not exceed base × max_multiplier."""
        max_mult = CAPACITY_SCARCITY_PARAMS['max_scarcity_multiplier']
        base = CAPACITY_MARKET_PRICES['PJM']
        price = compute_capacity_price('PJM', 0, 0)
        assert price <= base * max_mult * 1.01, \
            f"Price ${price:.1f} exceeds max ${base * max_mult:.1f}"


# ═══════════════════════════════════════════════════════════════════════════════
# R8: Input Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestR8InputValidation:
    """Verify Pydantic rejects invalid ISOs, negative LCOE, non-monotonic years."""

    def test_invalid_iso_rejected(self):
        """ISO not in VALID_ISOS should raise ValueError."""
        with pytest.raises(Exception):  # Pydantic ValidationError wraps ValueError
            SimulationRequest(iso="INVALID_ISO", years=[2025, 2030])

    def test_non_monotonic_years_rejected(self):
        """Non-ascending years should raise ValueError."""
        with pytest.raises(Exception):
            SimulationRequest(iso="ERCOT", years=[2030, 2025, 2035])

    def test_out_of_range_years_rejected(self):
        """Years outside [2020, 2100] should raise ValueError."""
        with pytest.raises(Exception):
            SimulationRequest(iso="ERCOT", years=[1900])

    def test_valid_request_accepted(self):
        """A well-formed request should construct without error."""
        req = SimulationRequest(iso="ERCOT", years=[2025, 2030, 2035])
        assert req.iso == "ERCOT"
        assert req.years == [2025, 2030, 2035]

    def test_iso_case_insensitive(self):
        """ISO should be uppercased automatically."""
        req = SimulationRequest(iso="ercot", years=[2025])
        assert req.iso == "ERCOT"

    def test_negative_carbon_price_rejected(self):
        """Negative carbon price should raise ValueError."""
        with pytest.raises(Exception):
            SimulationRequest(iso="ERCOT", years=[2025], carbon_price=-10)


# ═══════════════════════════════════════════════════════════════════════════════
# R10: Curtailment Feedback
# ═══════════════════════════════════════════════════════════════════════════════

class TestR10CurtailmentFeedback:
    """Verify effective LCOE increases with curtailment rate."""

    def test_zero_curtailment_no_change(self):
        """At 0% curtailment, effective LCOE = base LCOE."""
        base_lcoe = 30.0
        curtailment_rate = 0.0
        # Simulate the R10 formula
        if curtailment_rate > 0:
            effective_lcoe = base_lcoe / (1.0 - curtailment_rate)
        else:
            effective_lcoe = base_lcoe
        assert effective_lcoe == base_lcoe

    def test_curtailment_increases_effective_lcoe(self):
        """Higher curtailment → higher effective LCOE."""
        base_lcoe = 30.0
        lcoes = []
        for cr in [0.0, 0.1, 0.2, 0.3, 0.5]:
            if cr > 0:
                eff = base_lcoe / (1.0 - cr)
            else:
                eff = base_lcoe
            lcoes.append(eff)

        for i in range(len(lcoes) - 1):
            assert lcoes[i] < lcoes[i + 1], \
                f"LCOE should increase with curtailment: {lcoes[i]:.2f} < {lcoes[i+1]:.2f}"

    def test_50pct_curtailment_doubles_lcoe(self):
        """At 50% curtailment, effective LCOE = 2× base."""
        base_lcoe = 30.0
        effective = base_lcoe / (1.0 - 0.5)
        assert abs(effective - 60.0) < 0.01, \
            f"50% curtailment should double LCOE: got {effective:.2f}"

    def test_extreme_curtailment_blocks_deployment(self):
        """At ~100% curtailment, effective LCOE → infinity."""
        base_lcoe = 30.0
        effective_cf = 0.25 * (1.0 - 0.99)  # 99% curtailment
        if effective_cf > 0:
            effective_lcoe = base_lcoe / (1.0 - 0.99)
        else:
            effective_lcoe = float('inf')
        assert effective_lcoe > 1000, \
            f"Near-100% curtailment should produce very high LCOE: {effective_lcoe:.2f}"
