"""G1: Unit tests for plant-level retirement engine.

Tests:
1. High-margin plants survive while low-margin plants retire
2. Reliability floor prevents over-retirement (15% reserve margin)
3. Nuclear plants retire individually based on contract economics
4. Retired plant IDs persist across simulation years

Run: pytest market-simulator/scripts/tests/test_plant_retirement.py -v
"""
import sys
import os
import numpy as np
import pytest

# Add scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from market_simulation import (
    apply_economic_retirement,
    _apply_plant_level_retirement,
    _is_nuclear_plant,
    _evaluate_nuclear_contracts,
    _retire_nuclear_plants,
    _apply_fleet_fraction_retirement,
)
from pipeline_config import NUCLEAR_OFFTAKE_CONTRACTS, CAPACITY_MARKET_PRICES


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_plant(plant_id, unit_type, capacity_mw, profit_per_mwh,
                fuel_type='NG', zone='Z1', revenue_per_mwh=None):
    """Helper to create a plant economics dict."""
    rev = revenue_per_mwh if revenue_per_mwh is not None else max(0, profit_per_mwh + 30)
    return {
        'plant_id': plant_id,
        'generator_id': f'{plant_id}_G1',
        'plant_name': f'Plant {plant_id}',
        'capacity_mw': capacity_mw,
        'unit_type': unit_type,
        'fuel_type': fuel_type,
        'prime_mover': 'ST' if 'coal' in unit_type else 'CT',
        'profit_per_mwh': profit_per_mwh,
        'revenue_per_mwh': rev,
        'zone': zone,
        'status': 'stranded' if profit_per_mwh < -5 else ('at_risk' if profit_per_mwh < 2 else 'operating'),
    }


def _make_nuclear_plant(plant_id, capacity_mw, profit_per_mwh, revenue_per_mwh=None):
    """Helper to create a nuclear plant economics dict."""
    rev = revenue_per_mwh if revenue_per_mwh is not None else max(0, profit_per_mwh + 30)
    return {
        'plant_id': plant_id,
        'generator_id': f'{plant_id}_G1',
        'plant_name': f'Nuclear {plant_id}',
        'capacity_mw': capacity_mw,
        'unit_type': 'nuclear',
        'fuel_type': 'NUC',
        'prime_mover': 'ST',
        'profit_per_mwh': profit_per_mwh,
        'revenue_per_mwh': rev,
        'zone': 'Z1',
        'status': 'stranded' if profit_per_mwh < -5 else 'operating',
    }


def _make_gen_econ(**overrides):
    """Minimal gen_econ dict for the function signature."""
    base = {
        'gas_ccgt': {'capacity_mw': 10000, 'margin_mwh': 5, 'cf': 0.45, 'dispatch_hours': 4000},
        'gas_ct':   {'capacity_mw': 5000,  'margin_mwh': 3, 'cf': 0.15, 'dispatch_hours': 1300},
        'coal_steam': {'capacity_mw': 8000, 'margin_mwh': -8, 'cf': 0.30, 'dispatch_hours': 2600},
    }
    base.update(overrides)
    return base


def _make_state(**overrides):
    """Minimal simulation state dict.

    Default cumulative_gw is large enough (200 GW) to provide ample reserve
    margin so that tests for margin-based retirement aren't blocked by the
    reliability floor unless the test explicitly sets a tight supply scenario.
    """
    state = {
        'economic_retirements': {},
        'retired_plants': [],
        'cumulative_gw': {'solar': 100.0, 'wind': 100.0},  # 200 GW clean → ample reserves
        'nuclear_retired': False,
        'clean_pct': 40.0,
    }
    state.update(overrides)
    return state


# ── Test 1: High-margin survive, low-margin retire ───────────────────────────

class TestPlantMarginRetirement:
    """Verify that plants retire based on individual margin, worst-first."""

    def test_profitable_plants_survive(self):
        """Plants with margin > -$5/MWh should NOT retire."""
        plant_economics = [
            _make_plant('P1', 'gas_ccgt', 500, profit_per_mwh=10.0),
            _make_plant('P2', 'gas_ccgt', 300, profit_per_mwh=2.0),
            _make_plant('P3', 'gas_ct', 200, profit_per_mwh=0.0),
        ]
        gen_econ = _make_gen_econ()
        state = _make_state()

        adjusted, retired_by_type, total_mw, retired_list = apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=300)

        assert total_mw == 0
        assert len(retired_list) == 0

    def test_stranded_plants_retire(self):
        """Plants with margin < -$5/MWh should retire."""
        plant_economics = [
            _make_plant('P1', 'gas_ccgt', 500, profit_per_mwh=10.0),
            _make_plant('P2', 'coal_steam', 400, profit_per_mwh=-12.0),
            _make_plant('P3', 'coal_steam', 300, profit_per_mwh=-20.0),
        ]
        gen_econ = _make_gen_econ()
        state = _make_state()

        adjusted, retired_by_type, total_mw, retired_list = apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=300)

        assert total_mw == 700  # 400 + 300 MW
        assert len(retired_list) == 2
        # Worst margin retires first (P3 at -$20 before P2 at -$12)
        assert retired_list[0]['plant_id'] == 'P3'
        assert retired_list[1]['plant_id'] == 'P2'

    def test_mixed_fleet_selective_retirement(self):
        """Only economically stranded plants retire; profitable ones survive."""
        plant_economics = [
            _make_plant('P1', 'gas_ccgt', 1000, profit_per_mwh=15.0),
            _make_plant('P2', 'gas_ccgt', 800, profit_per_mwh=5.0),
            _make_plant('P3', 'coal_steam', 600, profit_per_mwh=-3.0),  # Above threshold
            _make_plant('P4', 'coal_steam', 400, profit_per_mwh=-8.0),  # Below threshold
            _make_plant('P5', 'oil_ct', 200, profit_per_mwh=-25.0),     # Deep loss
        ]
        gen_econ = _make_gen_econ()
        state = _make_state()

        adjusted, retired_by_type, total_mw, retired_list = apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=300)

        retired_ids = {p['plant_id'] for p in retired_list}
        assert 'P1' not in retired_ids  # Profitable
        assert 'P2' not in retired_ids  # Profitable
        assert 'P3' not in retired_ids  # Above threshold (-3 > -5)
        assert 'P4' in retired_ids      # Below threshold
        assert 'P5' in retired_ids      # Deep loss
        assert total_mw == 600  # 400 + 200

    def test_retirement_reduces_gen_econ_capacity(self):
        """Adjusted gen_econ should have reduced capacity for retired unit types."""
        plant_economics = [
            _make_plant('P1', 'coal_steam', 500, profit_per_mwh=-15.0),
        ]
        gen_econ = _make_gen_econ()
        original_coal_cap = gen_econ['coal_steam']['capacity_mw']
        state = _make_state()

        adjusted, retired_by_type, total_mw, retired_list = apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=300)

        assert adjusted['coal_steam']['capacity_mw'] == original_coal_cap - 500


# ── Test 2: Reliability floor prevents over-retirement ────────────────────────

class TestReliabilityFloor:
    """Verify that 15% zonal reserve margin floor prevents excess retirement."""

    def test_reliability_floor_prevents_full_retirement(self):
        """When retiring all stranded plants would breach reserve margin, stop early."""
        # Create a scenario where retiring all plants would breach reliability.
        # demand_twh=100 → avg_demand = 100e6/8760 ≈ 11,415 MW
        # peak = 11,415 × 1.5 = 17,123 MW
        # min_supply = 17,123 × 1.15 = 19,691 MW
        # clean_mw = 8,000 MW (from cumulative_gw)
        # total_fossil = 15,000 MW → total_supply = 23,000 MW
        # max_retirable = 23,000 - 19,691 = 3,309 MW
        # So we shouldn't be able to retire all 10,000 MW of stranded plants.
        plant_economics = [
            _make_plant('P1', 'coal_steam', 3000, profit_per_mwh=-20.0),
            _make_plant('P2', 'coal_steam', 3000, profit_per_mwh=-15.0),
            _make_plant('P3', 'gas_ccgt', 2000, profit_per_mwh=-10.0),
            _make_plant('P4', 'gas_ccgt', 2000, profit_per_mwh=-8.0),
            _make_plant('P5', 'gas_ccgt', 5000, profit_per_mwh=10.0),  # Profitable
        ]
        gen_econ = _make_gen_econ(
            coal_steam={'capacity_mw': 6000, 'margin_mwh': -15, 'cf': 0.30, 'dispatch_hours': 2600},
            gas_ccgt={'capacity_mw': 9000, 'margin_mwh': 5, 'cf': 0.45, 'dispatch_hours': 4000},
        )
        state = _make_state(cumulative_gw={'solar': 5.0, 'wind': 3.0})  # 8 GW clean

        adjusted, retired_by_type, total_mw, retired_list = apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=100)

        # Not all stranded capacity should retire — reliability floor kicks in
        total_stranded_mw = 3000 + 3000 + 2000 + 2000  # 10,000 MW stranded
        assert total_mw < total_stranded_mw
        assert total_mw > 0  # Some should still retire
        # P5 (profitable) should NOT retire
        retired_ids = {p['plant_id'] for p in retired_list}
        assert 'P5' not in retired_ids

    def test_no_retirement_when_at_reserve_floor(self):
        """If fleet is already at minimum reserve, no plants should retire."""
        # demand_twh=500 → peak ≈ 85,616 MW, min_supply ≈ 98,459 MW
        # total_fossil = 5000 MW + 8000 MW clean = 13,000 MW << 98,459
        # Already below floor — nothing retirable
        plant_economics = [
            _make_plant('P1', 'gas_ccgt', 5000, profit_per_mwh=-15.0),
        ]
        gen_econ = _make_gen_econ(
            gas_ccgt={'capacity_mw': 5000, 'margin_mwh': -15, 'cf': 0.30, 'dispatch_hours': 2600},
        )
        state = _make_state(cumulative_gw={'solar': 5.0, 'wind': 3.0})

        adjusted, retired_by_type, total_mw, retired_list = apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=500)

        assert total_mw == 0
        assert len(retired_list) == 0


# ── Test 3: Nuclear individual contract-based retirement ──────────────────────

class TestNuclearContractRetirement:
    """Verify nuclear plants retire individually, not as a fleet."""

    def test_nuclear_identification(self):
        """_is_nuclear_plant correctly identifies nuclear units."""
        assert _is_nuclear_plant({'fuel_type': 'NUC', 'prime_mover': 'ST', 'unit_type': 'nuclear'})
        assert _is_nuclear_plant({'fuel_type': 'NUCLEAR', 'prime_mover': 'ST', 'unit_type': ''})
        assert _is_nuclear_plant({'fuel_type': 'UR', 'prime_mover': 'ST', 'unit_type': ''})
        assert not _is_nuclear_plant({'fuel_type': 'NG', 'prime_mover': 'CT', 'unit_type': 'gas_ct'})
        assert not _is_nuclear_plant({'fuel_type': 'SUB', 'prime_mover': 'ST', 'unit_type': 'coal_steam'})

    def test_contract_protected_nuclear_survives(self):
        """ERCOT nuclear with active PPA floor should not retire even with low market revenue."""
        # Comanche Peak has contract_floor_mwh = $35/MWh through 2045
        plant_economics = [
            _make_nuclear_plant('NUC1', 2300, profit_per_mwh=-10.0, revenue_per_mwh=25.0),
            _make_plant('P2', 'gas_ccgt', 3000, profit_per_mwh=8.0),
        ]
        gen_econ = _make_gen_econ()
        state = _make_state()

        adjusted, retired_by_type, total_mw, retired_list = apply_economic_retirement(
            gen_econ, 'ERCOT', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=300)

        # Nuclear should NOT retire — PPA floor ($35) > operating cost ($30)
        retired_ids = {p['plant_id'] for p in retired_list}
        assert 'NUC1' not in retired_ids

    def test_unprotected_nuclear_retires_individually(self):
        """Nuclear without contract protection retires based on individual economics."""
        # SPP has no explicit offtake contract and no capacity market
        plant_economics = [
            _make_nuclear_plant('NUC1', 600, profit_per_mwh=-15.0, revenue_per_mwh=10.0),
            _make_nuclear_plant('NUC2', 600, profit_per_mwh=5.0, revenue_per_mwh=40.0),
            _make_plant('P3', 'gas_ccgt', 5000, profit_per_mwh=8.0),
        ]
        gen_econ = _make_gen_econ()
        state = _make_state()

        adjusted, retired_by_type, total_mw, retired_list = apply_economic_retirement(
            gen_econ, 'SPP', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=200)

        retired_ids = {p['plant_id'] for p in retired_list}
        # NUC1 should retire (uneconomic, no protection)
        assert 'NUC1' in retired_ids
        # NUC2 should survive (profitable)
        assert 'NUC2' not in retired_ids
        # This is individual retirement, not fleet-wide
        assert retired_by_type.get('nuclear', 0) == 600  # Only NUC1's capacity

    def test_capacity_market_protects_nuclear(self):
        """Nuclear in capacity market ISOs gets revenue credit from capacity payments."""
        # PJM has capacity market prices that supplement energy revenue.
        # Nuclear OPEX ≈ $30/MWh. If energy_rev ($20) + cap_rev ($15/kW-yr → ~$1.7/MWh)
        # still < $25/MWh threshold, plant may still retire.
        # But if cap market is significant enough, it saves the plant.
        plant_economics = [
            # Energy rev $28 + capacity market → should be saved
            _make_nuclear_plant('NUC_PJM', 1000, profit_per_mwh=-3.0, revenue_per_mwh=28.0),
        ]
        gen_econ = _make_gen_econ()
        state = _make_state()

        adjusted, retired_by_type, total_mw, retired_list = apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=300)

        # With PJM capacity market revenue supplement, margin should be > -$5
        retired_ids = {p['plant_id'] for p in retired_list}
        assert 'NUC_PJM' not in retired_ids


# ── Test 4: Retired plant IDs persist across years ────────────────────────────

class TestRetirementPersistence:
    """Verify that retired plant IDs are tracked in state across simulation years."""

    def test_retired_ids_stored_in_state(self):
        """After retirement, plant IDs should appear in state['retired_plants']."""
        plant_economics = [
            _make_plant('P1', 'coal_steam', 500, profit_per_mwh=-15.0),
            _make_plant('P2', 'gas_ccgt', 300, profit_per_mwh=10.0),
        ]
        gen_econ = _make_gen_econ()
        state = _make_state()

        apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=300)

        assert 'P1' in state['retired_plants']
        assert 'P2' not in state['retired_plants']

    def test_retired_plants_excluded_in_subsequent_year(self):
        """Plants retired in year 1 should not appear as candidates in year 2."""
        plant_economics = [
            _make_plant('P1', 'coal_steam', 500, profit_per_mwh=-15.0),
            _make_plant('P2', 'gas_ccgt', 300, profit_per_mwh=10.0),
        ]
        gen_econ = _make_gen_econ()
        state = _make_state()

        # Year 1: P1 retires
        apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics, demand_twh=300)

        assert 'P1' in state['retired_plants']

        # Year 2: P1 is already in retired_plants, so even if plant_economics
        # still includes it, the function should filter it out
        plant_economics_yr2 = [
            _make_plant('P1', 'coal_steam', 500, profit_per_mwh=-15.0),  # Still in list
            _make_plant('P2', 'gas_ccgt', 300, profit_per_mwh=-8.0),     # Now stranded
        ]

        adjusted2, retired2, total2, retired_list2 = apply_economic_retirement(
            gen_econ, 'PJM', 2035, state, _log=lambda *a: None,
            plant_economics=plant_economics_yr2, demand_twh=300)

        # P1 should NOT appear again in retirement list (already retired)
        retired_ids = {p['plant_id'] for p in retired_list2}
        assert 'P1' not in retired_ids
        # P2 should retire this year
        assert 'P2' in retired_ids
        # Both should be in persistent state
        assert 'P1' in state['retired_plants']
        assert 'P2' in state['retired_plants']

    def test_cumulative_retirement_mw_in_state(self):
        """Legacy economic_retirements dict should accumulate across years."""
        plant_economics_yr1 = [
            _make_plant('P1', 'coal_steam', 500, profit_per_mwh=-15.0),
        ]
        plant_economics_yr2 = [
            _make_plant('P2', 'coal_steam', 300, profit_per_mwh=-10.0),
        ]
        gen_econ = _make_gen_econ()
        state = _make_state()

        # Year 1
        apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None,
            plant_economics=plant_economics_yr1, demand_twh=300)

        assert state['economic_retirements'].get('coal_steam', 0) == 500

        # Year 2
        apply_economic_retirement(
            gen_econ, 'PJM', 2035, state, _log=lambda *a: None,
            plant_economics=plant_economics_yr2, demand_twh=300)

        assert state['economic_retirements'].get('coal_steam', 0) == 800  # 500 + 300


# ── Test 5: Legacy fallback path still works ──────────────────────────────────

class TestLegacyFallback:
    """Verify fleet-fraction retirement works when plant_economics=None."""

    def test_fallback_returns_four_values(self):
        """When plant_economics is None, should return 4 values with empty plant list."""
        gen_econ = _make_gen_econ()
        state = _make_state()

        result = apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None)

        assert len(result) == 4
        adjusted, retired_by_type, total_mw, plant_list = result
        assert isinstance(plant_list, list)
        assert len(plant_list) == 0  # No plant-level data in legacy mode

    def test_fallback_still_retires_stranded_types(self):
        """Legacy path should still retire fleet fractions for negative-margin types."""
        gen_econ = {
            'coal_steam': {'capacity_mw': 8000, 'margin_mwh': -15, 'cf': 0.30, 'dispatch_hours': 2600},
            'gas_ccgt': {'capacity_mw': 10000, 'margin_mwh': 8, 'cf': 0.45, 'dispatch_hours': 4000},
        }
        state = _make_state()

        adjusted, retired_by_type, total_mw, _ = apply_economic_retirement(
            gen_econ, 'PJM', 2030, state, _log=lambda *a: None)

        assert total_mw > 0
        assert 'coal_steam' in retired_by_type
        assert adjusted['coal_steam']['capacity_mw'] < 8000
        # Gas CCGT should be untouched (profitable)
        assert adjusted['gas_ccgt']['capacity_mw'] == 10000


# ── Test 6: Nuclear contract evaluation ───────────────────────────────────────

class TestNuclearContractEvaluation:
    """Test the _evaluate_nuclear_contracts helper."""

    def test_ercot_contract_before_expiry(self):
        """ERCOT nuclear should be contract-protected before 2045."""
        plants = [_make_nuclear_plant('NUC1', 2300, -5)]
        result = _evaluate_nuclear_contracts('ERCOT', 2030, plants)
        assert result['NUC1']['contract_protected'] is True
        assert result['NUC1']['contract_revenue_mwh'] == 35.0

    def test_ercot_contract_after_expiry(self):
        """ERCOT nuclear should lose protection after contract end year."""
        plants = [_make_nuclear_plant('NUC1', 2300, -5)]
        result = _evaluate_nuclear_contracts('ERCOT', 2050, plants)
        assert result['NUC1']['contract_protected'] is False

    def test_pjm_capacity_market(self):
        """PJM nuclear should have capacity market revenue."""
        plants = [_make_nuclear_plant('NUC1', 1000, -5)]
        result = _evaluate_nuclear_contracts('PJM', 2030, plants)
        assert result['NUC1']['contract_protected'] is False
        assert result['NUC1']['has_capacity_market'] is True
        assert result['NUC1']['capacity_market_revenue_kw_yr'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
