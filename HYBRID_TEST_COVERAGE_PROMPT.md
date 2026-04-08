# Task: Add Hybrid Resource Test Coverage

## Context

The hourly CFE optimizer pipeline supports 10 resource types: 6 base (`clean_firm`, `solar`, `wind`, `offshore_wind`, `ccs_ccgt`, `hydro`) + 4 hybrid co-located (`solar_batt4`, `solar_batt8`, `wind_batt4`, `wind_batt8`). A recent audit (`HYBRID_AUDIT_REPORT.md`) found **zero test coverage** for hybrid dispatch paths. All existing tests in `scripts/tests/` operate exclusively on the 6 base resource types.

## What to Build

Add hybrid resource tests to the existing test suite. **Do not modify existing tests** — add new test classes/functions alongside them. Tests must use **synthetic deterministic data only** (no EIA/eGRID data files), matching the existing pattern in `conftest.py`.

## Specific Test Gaps to Fill

### 1. New Fixtures in `scripts/tests/conftest.py`

Add these fixtures alongside existing ones:

- **`synthetic_hybrid_profiles`**: A `synthetic_profiles`-like dict that also includes 4 hybrid resource profiles (`solar_batt4`, `solar_batt8`, `wind_batt4`, `wind_batt8`). Hybrid profiles should be distinct from base profiles — e.g., `solar_batt4` is solar-shaped but with battery smoothing (evening tail), `wind_batt4` is wind-shaped with similar smoothing. Must be normalized to sum=1.0 each.

- **`hybrid_mix`**: A `small_mix`-like dict that includes hybrid keys. Example: `{'clean_firm': 30, 'solar': 15, 'wind': 15, 'offshore_wind': 0, 'ccs_ccgt': 0, 'hydro': 10, 'solar_batt4': 15, 'wind_batt4': 15}` (sums to 100%).

### 2. New Tests in `scripts/tests/test_dispatch.py`

Add a `TestHybridDispatch` class with these tests:

- **`test_hybrid_build_supply_matrix_shape`**: Verify `build_supply_matrix(profiles, resource_types=RESOURCE_TYPES_HYBRID)` returns shape `(10, 8760)` when given hybrid profiles.

- **`test_hybrid_dispatch_energy_balance`**: Verify `reconstruct_hourly_dispatch` with hybrid `resource_pcts` and `resource_types=RESOURCE_TYPES_HYBRID` conserves energy: `supply_total + battery4 + battery8 + ldes + h2 == total_clean`.

- **`test_hybrid_dispatch_reduces_residual`**: A mix with hybrid resources should have lower residual demand than the same base-only mix (with equivalent total generation %). Hybrids with battery smoothing shift solar/wind to peak hours.

- **`test_hybrid_zero_pcts_matches_base`**: Verify that calling `reconstruct_hourly_dispatch` with `resource_types=RESOURCE_TYPES_HYBRID` but all hybrid keys set to 0 produces identical results to calling with `resource_types=RESOURCE_TYPES` (base only). This tests backward compatibility.

- **`test_hybrid_dispatch_order`**: Verify `_get_dispatch_order_indices(RESOURCE_TYPES_HYBRID)` returns valid indices for all 10 resource types and that hybrid types are dispatched in the correct position (after their base counterpart: solar_batt4 after solar, wind_batt4 after wind).

### 3. New Tests in `scripts/tests/test_energy_conservation.py`

Add a `TestHybridEnergyConservation` class:

- **`test_hybrid_total_clean_equals_components`**: Same as existing `test_total_clean_equals_components` but using hybrid fixtures. Verifies `supply_total + storage = total_clean` identity holds with hybrid resources.

- **`test_hybrid_no_negative_residuals`**: With hybrid resources, residual demand should never go negative.

### 4. New Tests in `scripts/tests/test_constants.py`

- **`test_hybrid_types_in_peak_capacity_credits`**: Verify all 4 `HYBRID_TYPES` exist as keys in `PEAK_CAPACITY_CREDITS` with values between 0 and 1.

- **`test_resource_types_hybrid_length`**: Verify `RESOURCE_TYPES_HYBRID` has exactly 10 elements and includes all `HYBRID_TYPES`.

## Key Constants & Imports

```python
# From pipeline_config
HYBRID_TYPES = ['solar_batt4', 'solar_batt8', 'wind_batt4', 'wind_batt8']

# From dispatch_utils
RESOURCE_TYPES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
RESOURCE_TYPES_HYBRID = RESOURCE_TYPES + HYBRID_TYPES  # 10 types
H = 8760

# Key functions
from dispatch_utils import (
    build_supply_matrix, reconstruct_hourly_dispatch,
    RESOURCE_TYPES, RESOURCE_TYPES_HYBRID, H,
    _get_dispatch_order_indices,
)
from pipeline_config import HYBRID_TYPES, PEAK_CAPACITY_CREDITS
```

## Key Function Signatures

```python
def build_supply_matrix(supply_profiles, resource_types=None)
# resource_types=None → defaults to RESOURCE_TYPES (6). Pass RESOURCE_TYPES_HYBRID for 10.
# Returns: numpy array of shape (len(resource_types), H)

def reconstruct_hourly_dispatch(demand_norm, supply_profiles, resource_pcts,
                                 procurement_pct=100, battery_dispatch_pct=0,
                                 battery8_dispatch_pct=0, ldes_dispatch_pct=0,
                                 supply_matrix=None, detailed=False,
                                 h2_dispatch_pct=0, resource_types=None)
# resource_types=None → defaults to RESOURCE_TYPES (6). Pass RESOURCE_TYPES_HYBRID for 10.
# Returns: dict with 'supply_total', 'total_clean', 'residual_demand', 'curtailed', etc.
```

## Rules

1. **Synthetic data only** — no file I/O, no EIA/eGRID data, no parquets. Match conftest.py patterns.
2. **No modifications to existing tests** — only additive changes.
3. **Use pytest fixtures** — follow the existing fixture pattern in conftest.py.
4. **Assertions must be meaningful** — don't just check "no crash". Verify specific numerical properties (shapes, sums, bounds, relative ordering).
5. **Run the tests** before committing: `cd /home/user/hourly-cfe-optimizer && python -m pytest scripts/tests/ -v`
6. **Commit and push** to branch `claude/hybrid-audit-review-vAlfK` when tests pass.
