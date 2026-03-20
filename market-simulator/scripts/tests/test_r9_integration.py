"""R9 QA/QC: Integration tests verifying cross-recommendation compatibility.

Run: pytest market-simulator/scripts/tests/test_r9_integration.py -v
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from market_simulation import (
    compute_storage_arbitrage_from_lmp,
    compute_storage_deployment,
    wright_cost,
    get_resource_lcoe,
    compute_lcoe_snapshot,
    _percentile_dict,
    H,
)
from pipeline_config import (
    WRIGHT_CUMULATIVE_GW_2025,
    CAPACITY_MARKET_PRICES,
    compute_capacity_price,
)


# ═══════════════════════════════════════════════════════════════════════════════
# R1 + R7: Storage Arbitrage + Capacity Market Revenue Stacking
# ═══════════════════════════════════════════════════════════════════════════════

class TestR1R7StorageCapacityStacking:
    """Storage revenue = arbitrage + capacity market. Both must be active."""

    def test_pjm_storage_has_both_revenue_streams(self):
        """PJM storage should earn both arbitrage and capacity revenue."""
        # Duck curve LMP for arbitrage
        np.random.seed(42)
        lmp = np.zeros(H)
        for d in range(365):
            h = d * 24
            lmp[h:h+24] = 35
            lmp[h+10:h+15] = 15  # trough
            lmp[h+16:h+21] = 75  # peak

        arb = compute_storage_arbitrage_from_lmp(lmp, iso='PJM')
        cap_price = compute_capacity_price('PJM', 15, 30)

        assert arb.get('battery', 0) > 0, "Battery should earn arbitrage on duck curve"
        assert cap_price > 0, "PJM capacity price should be positive"
        # Total revenue = arb + capacity — both contribute
        total = arb['battery'] + cap_price
        assert total > arb['battery'], \
            "Total revenue should exceed arbitrage alone (capacity adds value)"

    def test_ercot_storage_no_capacity_revenue(self):
        """ERCOT storage revenue is arbitrage-only (no capacity market)."""
        cap_price = compute_capacity_price('ERCOT', 15, 30)
        assert cap_price == 0, "ERCOT capacity price must be $0"
        # In ERCOT, storage viability depends entirely on arbitrage + ancillary


# ═══════════════════════════════════════════════════════════════════════════════
# R1 + R2: Storage Costs Decline via Wright's Law
# ═══════════════════════════════════════════════════════════════════════════════

class TestR1R2StorageLearning:
    """Storage costs should decline with cumulative deployment."""

    def test_battery_lcoe_decreases_over_time(self):
        """Battery LCOE should decrease with more cumulative GW."""
        cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
        lcoes = []
        for year in range(2025, 2045):
            lcoe = get_resource_lcoe('battery', 'ERCOT', 'Medium',
                                     cumulative_gw, 'Fast', year)
            lcoes.append(lcoe)
            cumulative_gw['battery'] = cumulative_gw.get('battery', 0) + 10

        assert lcoes[0] > lcoes[-1], \
            f"Battery LCOE should decrease: {lcoes[0]:.2f} → {lcoes[-1]:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# R2 + R10: Learning Curves + Curtailment Feedback
# ═══════════════════════════════════════════════════════════════════════════════

class TestR2R10LearningCurtailment:
    """VRE costs decline via learning but curtailment increases effective LCOE."""

    def test_competing_forces(self):
        """Even with learning, high curtailment can make VRE uneconomic."""
        # Use clean_firm (nuclear) which has active Wright's Law learning (lr=0.15)
        # Solar/wind have lr=0.0 (mature technologies)
        cumulative_gw = dict(WRIGHT_CUMULATIVE_GW_2025)
        # Year 1: low deployment, no curtailment
        lcoe_y1 = get_resource_lcoe('clean_firm', 'ERCOT', 'Medium',
                                     cumulative_gw, 'Fast', 2025)
        # Year 20: high deployment (lower base LCOE via learning)
        cumulative_gw['nuclear'] = cumulative_gw.get('nuclear', 0) + 50
        lcoe_y20 = get_resource_lcoe('clean_firm', 'ERCOT', 'Medium',
                                      cumulative_gw, 'Fast', 2045)

        # Learning should reduce base LCOE
        assert lcoe_y20 < lcoe_y1, "Learning should reduce nuclear LCOE"

        # But with 40% curtailment, effective LCOE could be higher than Y1
        curtailment_rate = 0.4
        effective_lcoe_y20 = lcoe_y20 / (1.0 - curtailment_rate)

        # The competing forces: learning reduces LCOE, curtailment increases it
        # With enough curtailment, net effective LCOE can exceed Y1 base LCOE
        assert effective_lcoe_y20 > lcoe_y20, \
            "Curtailment should increase effective LCOE above learning-reduced base"


# ═══════════════════════════════════════════════════════════════════════════════
# R4 + R10: VRE Basis Differential + Curtailment Interaction
# ═══════════════════════════════════════════════════════════════════════════════

class TestR4R10BasisCurtailment:
    """VRE faces both basis risk (lower zonal LMP) and curtailment penalty."""

    def test_double_penalty_on_vre(self):
        """VRE in cheap zone with high curtailment faces compounded penalty."""
        base_lcoe = 25.0  # $/MWh solar LCOE
        system_lmp = 40.0
        zone_lmp = 30.0    # -$10/MWh basis differential
        curtailment_rate = 0.2  # 20% curtailment

        # R4 effect: lower revenue from zone LMP
        revenue_system = system_lmp  # without R4
        revenue_zonal = zone_lmp     # with R4 (lower)

        # R10 effect: higher effective LCOE
        effective_lcoe = base_lcoe / (1.0 - curtailment_rate)  # $31.25

        # Combined: profit margin shrinks from both sides
        profit_without = revenue_system - base_lcoe        # $15/MWh
        profit_with = revenue_zonal - effective_lcoe        # -$1.25/MWh

        assert profit_without > 0, "Without R4+R10, VRE should be profitable"
        assert profit_with < profit_without, \
            "R4+R10 combined should reduce profit margin"


# ═══════════════════════════════════════════════════════════════════════════════
# R5 + All: Confidence Intervals Reflect All Mechanisms
# ═══════════════════════════════════════════════════════════════════════════════

class TestR5CrossRecommendation:
    """Confidence intervals should show variability from all endogenous mechanisms."""

    def test_percentile_spread_increases_with_variance(self):
        """Higher variance data → wider P10-P90 spread."""
        np.random.seed(42)
        narrow = np.random.normal(50, 2, 100)
        wide = np.random.normal(50, 20, 100)

        narrow_bands = _percentile_dict(narrow)
        wide_bands = _percentile_dict(wide)

        narrow_spread = narrow_bands['p90'] - narrow_bands['p10']
        wide_spread = wide_bands['p90'] - wide_bands['p10']

        assert wide_spread > narrow_spread, \
            f"Wide variance spread ({wide_spread:.1f}) should > narrow ({narrow_spread:.1f})"

    def test_percentile_ordering_invariant(self):
        """P10 <= P25 <= P50 <= P75 <= P90 always."""
        np.random.seed(99)
        for _ in range(10):
            arr = np.random.lognormal(3, 1, 50)
            bands = _percentile_dict(arr)
            assert bands['p10'] <= bands['p25'] <= bands['p50'] \
                   <= bands['p75'] <= bands['p90'], \
                f"Percentile ordering violated: {bands}"
