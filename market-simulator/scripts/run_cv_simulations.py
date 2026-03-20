import sys, json, time
sys.path.insert(0, 'scripts')
from market_simulation import run_market_simulation, load_egrid_baselines
from dispatch_utils import load_common_data

# Pre-load data once
demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
egrid_baselines = load_egrid_baselines()

# Try loading interchange data (optional)
try:
    from eia_data_io import load_interchange_profiles
    interchange_data = load_interchange_profiles()
except Exception:
    interchange_data = {}

preloaded = {
    'demand_data': demand_data,
    'gen_profiles': gen_profiles,
    'emission_rates': emission_rates,
    'fossil_mix': fossil_mix,
    'egrid_baselines': egrid_baselines,
    'interchange_data': interchange_data,
}

# Reference scenario (AEO-comparable, $0 carbon)
ref_conditions = {
    'name': 'CV_Reference_Zero_Carbon',
    'demand_growth': 'Medium',
    'lcoe_level': 'Medium',
    'fuel_level': 'Medium',
    'carbon_price': 0,
    'learning_rate': 'Medium',
    '45q_enabled': True,
    'queue_cap_level': 'Medium',
}

# EPA scenario ($51/ton carbon)
epa_conditions = dict(ref_conditions)
epa_conditions['name'] = 'CV_Reference_EPA_Carbon'
epa_conditions['carbon_price'] = 51

TARGET_ISOS = ['CAISO', 'PJM', 'ERCOT']

results = {}
for label, conditions in [('reference', ref_conditions), ('epa_carbon', epa_conditions)]:
    print(f"\n{'='*60}")
    print(f"Running {label} scenario...")
    t0 = time.time()
    sim_results = run_market_simulation(
        scenario_id=f'cv_{label}',
        conditions=conditions,
        isos=TARGET_ISOS,
        _preloaded=preloaded,
    )
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    results[label] = sim_results

# Save to JSON cache
output_path = 'data/cv_reference_results.json'

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if hasattr(obj, 'isoformat'): return obj.isoformat()
        if isinstance(obj, set): return list(obj)
        return super().default(obj)

with open(output_path, 'w') as f:
    json.dump(results, f, cls=NumpyEncoder, indent=2)
print(f"\nResults cached to {output_path}")
