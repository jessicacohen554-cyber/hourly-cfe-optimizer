"""
Supply profile loading — wraps eia_data_io + dispatch_utils to provide
normalized 8760-hour generation shapes for each resource type per ISO.

Each shape profile sums to 1.0 (represents 1 MWh of annual generation).
Multiply by allocation_MWh to get hourly generation in MWh.
"""

import sys, os
import numpy as np

_scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from eia_data_io import load_generation_profiles
from dispatch_utils import get_supply_profiles

H = 8760

# Cache loaded profiles across calls
_gen_profiles_cache = None


def _get_gen_profiles():
    global _gen_profiles_cache
    if _gen_profiles_cache is None:
        _gen_profiles_cache = load_generation_profiles()
    return _gen_profiles_cache


def load_supply_shapes(iso):
    """Load normalized generation shape profiles for an ISO.

    Returns:
        dict mapping resource_name → (8760,) numpy array summing to 1.0.
        Resources with zero generation are omitted.
    """
    gen_profiles = _get_gen_profiles()
    raw = get_supply_profiles(iso, gen_profiles)

    shapes = {}
    for resource, profile in raw.items():
        arr = np.array(profile[:H], dtype=np.float64)
        arr = np.maximum(arr, 0.0)
        total = arr.sum()
        if total > 1e-10:
            shapes[resource] = arr / total  # normalize to sum=1.0
        # else: skip — resource not available in this ISO

    # Add geothermal (flat profile, CAISO only)
    if iso == 'CAISO':
        shapes['geothermal'] = np.ones(H, dtype=np.float64) / H

    return shapes


def get_shape(iso, resource):
    """Get a single resource's shape profile. Returns (8760,) summing to 1.0."""
    shapes = load_supply_shapes(iso)
    if resource not in shapes:
        raise ValueError(f"Resource '{resource}' not available for {iso}. "
                         f"Available: {list(shapes.keys())}")
    return shapes[resource]
