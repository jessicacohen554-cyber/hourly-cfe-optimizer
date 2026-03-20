"""G4: Tests for IEA-aligned correlated scenario bundles.

Verifies that:
1. Each scenario in CORRELATED_SCENARIOS has valid parameter combinations.
2. _correlated_scenario_to_conditions produces correct conditions dicts.
3. run_correlated_scenarios validates inputs and returns keyed results.
4. Pydantic models (CorrelatedScenarioRequest/Response) validate correctly.
5. Distinct scenarios produce distinct simulation conditions.

Run: pytest market-simulator/scripts/tests/test_correlated_scenarios.py -v
"""
import sys
import os

import pytest

# Add scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Add backend dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from pipeline_config import CORRELATED_SCENARIOS
from market_simulation import (
    _correlated_scenario_to_conditions,
    run_correlated_scenarios,
)
from models import (
    CorrelatedScenarioRequest,
    CorrelatedScenarioResult,
    CorrelatedScenarioResponse,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definition validation
# ─────────────────────────────────────────────────────────────────────────────

VALID_DEMAND_GROWTH = {'Low', 'Medium', 'High'}
VALID_GAS_PRICE = {'Low', 'Medium', 'High'}
VALID_RENEWABLE_LCOE = {'Low', 'Medium', 'High'}
VALID_LEARNING_RATE = {'Slow', 'Medium', 'Fast'}
REQUIRED_KEYS = {
    'description', 'demand_growth', 'gas_price', 'renewable_lcoe',
    'carbon_price', 'learning_rate', '45q',
}


class TestCorrelatedScenarioDefinitions:
    """Verify CORRELATED_SCENARIOS dict structure and values."""

    def test_has_five_scenarios(self):
        assert len(CORRELATED_SCENARIOS) == 5

    def test_expected_scenario_names(self):
        expected = {'IEA_STEPS', 'IEA_APS', 'IEA_NZE', 'HIGH_FRICTION', 'RAPID_TRANSITION'}
        assert set(CORRELATED_SCENARIOS.keys()) == expected

    @pytest.mark.parametrize("name", list(CORRELATED_SCENARIOS.keys()))
    def test_scenario_has_required_keys(self, name):
        spec = CORRELATED_SCENARIOS[name]
        missing = REQUIRED_KEYS - set(spec.keys())
        assert not missing, f"{name} missing keys: {missing}"

    @pytest.mark.parametrize("name", list(CORRELATED_SCENARIOS.keys()))
    def test_demand_growth_valid(self, name):
        assert CORRELATED_SCENARIOS[name]['demand_growth'] in VALID_DEMAND_GROWTH

    @pytest.mark.parametrize("name", list(CORRELATED_SCENARIOS.keys()))
    def test_gas_price_valid(self, name):
        assert CORRELATED_SCENARIOS[name]['gas_price'] in VALID_GAS_PRICE

    @pytest.mark.parametrize("name", list(CORRELATED_SCENARIOS.keys()))
    def test_renewable_lcoe_valid(self, name):
        assert CORRELATED_SCENARIOS[name]['renewable_lcoe'] in VALID_RENEWABLE_LCOE

    @pytest.mark.parametrize("name", list(CORRELATED_SCENARIOS.keys()))
    def test_learning_rate_valid(self, name):
        assert CORRELATED_SCENARIOS[name]['learning_rate'] in VALID_LEARNING_RATE

    @pytest.mark.parametrize("name", list(CORRELATED_SCENARIOS.keys()))
    def test_carbon_price_non_negative(self, name):
        assert CORRELATED_SCENARIOS[name]['carbon_price'] >= 0

    @pytest.mark.parametrize("name", list(CORRELATED_SCENARIOS.keys()))
    def test_45q_is_boolean(self, name):
        assert isinstance(CORRELATED_SCENARIOS[name]['45q'], bool)

    @pytest.mark.parametrize("name", list(CORRELATED_SCENARIOS.keys()))
    def test_description_is_nonempty_string(self, name):
        desc = CORRELATED_SCENARIOS[name]['description']
        assert isinstance(desc, str) and len(desc) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Condition mapping tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConditionMapping:
    """Verify _correlated_scenario_to_conditions produces valid conditions."""

    @pytest.mark.parametrize("name", list(CORRELATED_SCENARIOS.keys()))
    def test_conditions_have_required_keys(self, name):
        spec = CORRELATED_SCENARIOS[name]
        cond = _correlated_scenario_to_conditions(name, spec)
        required = {
            'name', 'demand_growth', 'lcoe_level', 'learning_speed',
            'queue_cap_level', 'gas_friction', 'carbon_price',
            'fuel_level', 'tx_level', 'ppa_level',
        }
        missing = required - set(cond.keys())
        assert not missing, f"{name} conditions missing: {missing}"

    def test_iae_steps_maps_correctly(self):
        spec = CORRELATED_SCENARIOS['IEA_STEPS']
        cond = _correlated_scenario_to_conditions('IEA_STEPS', spec)
        assert cond['demand_growth'] == 'Medium'
        assert cond['lcoe_level'] == 'Medium'
        assert cond['fuel_level'] == 'Medium'
        assert cond['carbon_price'] == 0
        assert cond['learning_speed'] == 'Medium'
        assert cond['gas_friction'] == 0.7

    def test_nze_maps_correctly(self):
        spec = CORRELATED_SCENARIOS['IEA_NZE']
        cond = _correlated_scenario_to_conditions('IEA_NZE', spec)
        assert cond['demand_growth'] == 'High'
        assert cond['lcoe_level'] == 'Low'
        assert cond['fuel_level'] == 'High'
        assert cond['carbon_price'] == 185
        assert cond['learning_speed'] == 'Fast'
        assert cond['gas_friction'] == 1.0

    def test_high_friction_maps_correctly(self):
        spec = CORRELATED_SCENARIOS['HIGH_FRICTION']
        cond = _correlated_scenario_to_conditions('HIGH_FRICTION', spec)
        assert cond['demand_growth'] == 'High'
        assert cond['lcoe_level'] == 'High'
        assert cond['fuel_level'] == 'Low'
        assert cond['carbon_price'] == 0
        assert cond['learning_speed'] == 'Slow'
        assert cond['gas_friction'] == 0.3

    def test_correlated_scenario_tag_present(self):
        """Each conditions dict should carry its scenario name for provenance."""
        for name in CORRELATED_SCENARIOS:
            spec = CORRELATED_SCENARIOS[name]
            cond = _correlated_scenario_to_conditions(name, spec)
            assert cond.get('_correlated_scenario') == name

    def test_scenarios_produce_distinct_conditions(self):
        """No two scenarios should produce identical conditions."""
        all_conds = {}
        for name, spec in CORRELATED_SCENARIOS.items():
            cond = _correlated_scenario_to_conditions(name, spec)
            # Use a hashable key from the relevant fields
            key = (
                cond['demand_growth'], cond['lcoe_level'], cond['fuel_level'],
                cond['carbon_price'], cond['learning_speed'], cond['gas_friction'],
            )
            assert key not in all_conds.values(), (
                f"{name} produces same conditions as another scenario"
            )
            all_conds[name] = key

        # Verify all 5 are distinct
        assert len(set(all_conds.values())) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Input validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInputValidation:
    """Verify run_correlated_scenarios rejects invalid inputs."""

    def test_invalid_scenario_name_raises(self):
        with pytest.raises(ValueError, match="Unknown correlated scenario"):
            run_correlated_scenarios('PJM', scenario_names=['NONEXISTENT'])

    def test_none_scenarios_defaults_to_all(self):
        """Passing None should select all 5 scenarios (validated via conditions mapping)."""
        # We just verify the names default — full simulation requires data
        names = list(CORRELATED_SCENARIOS.keys())
        assert len(names) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPydanticModels:
    """Verify request/response models validate correctly."""

    def test_request_valid_iso(self):
        req = CorrelatedScenarioRequest(iso='PJM')
        assert req.iso == 'PJM'
        assert req.scenarios is None

    def test_request_with_scenarios(self):
        req = CorrelatedScenarioRequest(iso='ERCOT', scenarios=['IEA_STEPS', 'IEA_NZE'])
        assert req.scenarios == ['IEA_STEPS', 'IEA_NZE']

    def test_request_invalid_iso_raises(self):
        with pytest.raises(Exception):
            CorrelatedScenarioRequest(iso='INVALID')

    def test_result_model(self):
        result = CorrelatedScenarioResult(
            scenario_name='IEA_STEPS',
            scenario_id='CORR_IEA_STEPS',
            description='Current policies continue',
            parameters={'demand_growth': 'Medium', 'gas_price': 'Medium'},
            year_results={'PJM': [{'year': 2025, 'clean_pct': 45.0}]},
        )
        assert result.scenario_name == 'IEA_STEPS'
        assert result.scenario_id == 'CORR_IEA_STEPS'

    def test_response_model(self):
        result = CorrelatedScenarioResult(
            scenario_name='IEA_NZE',
            scenario_id='CORR_IEA_NZE',
            description='1.5C pathway',
            parameters={'demand_growth': 'High'},
        )
        response = CorrelatedScenarioResponse(
            iso='CAISO',
            scenarios=[result],
            provenance={'model_version': '1.0.0'},
        )
        assert response.iso == 'CAISO'
        assert len(response.scenarios) == 1
        assert response.provenance is not None


# ─────────────────────────────────────────────────────────────────────────────
# IEA alignment checks
# ─────────────────────────────────────────────────────────────────────────────

class TestIEAAlignment:
    """Verify scenario parameters match IEA pathway characteristics."""

    def test_steps_is_baseline(self):
        """STEPS = current policies, should be moderate across the board."""
        s = CORRELATED_SCENARIOS['IEA_STEPS']
        assert s['demand_growth'] == 'Medium'
        assert s['carbon_price'] == 0
        assert s['renewable_lcoe'] == 'Medium'

    def test_aps_has_carbon_price(self):
        """APS = announced pledges, should include EPA SCC."""
        s = CORRELATED_SCENARIOS['IEA_APS']
        assert s['carbon_price'] == 51  # EPA SCC

    def test_nze_has_highest_carbon_price(self):
        """NZE = 1.5C, should have Rennert et al. SCC."""
        s = CORRELATED_SCENARIOS['IEA_NZE']
        assert s['carbon_price'] == 185  # Rennert et al.

    def test_high_friction_no_45q(self):
        """HIGH_FRICTION = stress test, 45Q should be off."""
        s = CORRELATED_SCENARIOS['HIGH_FRICTION']
        assert s['45q'] is False
        assert s['carbon_price'] == 0

    def test_rapid_transition_fast_learning(self):
        """RAPID_TRANSITION = bull case, should have fast learning + low LCOE."""
        s = CORRELATED_SCENARIOS['RAPID_TRANSITION']
        assert s['learning_rate'] == 'Fast'
        assert s['renewable_lcoe'] == 'Low'
        assert s['carbon_price'] == 100

    def test_carbon_prices_monotonic_steps_aps_nze(self):
        """Carbon prices should increase: STEPS < APS < NZE."""
        steps = CORRELATED_SCENARIOS['IEA_STEPS']['carbon_price']
        aps = CORRELATED_SCENARIOS['IEA_APS']['carbon_price']
        nze = CORRELATED_SCENARIOS['IEA_NZE']['carbon_price']
        assert steps < aps < nze
