# G6 — Cross-Validation Benchmarks: Split Into 3 Prompts

The original Prompt 5 (G6) from Final_Peer_Review_Audit.md times out because it combines
research, simulation execution, analysis, documentation, and test writing in a single session.
Below are 3 focused prompts that accomplish the same goal.

---

## Prompt 5A: Reference Data + Validation Script Skeleton (no simulation)

```
In market-simulator/scripts/tests/validate_cross_model.py, create the cross-validation
script structure WITHOUT running any simulations. This prompt is code-only — no compute.

1. Hard-code reference trajectory data from publicly available sources as Python dicts:

   a. EIA AEO 2025 Reference Case — clean energy share trajectory:
      AEO_REFERENCE = {
          'source': 'EIA Annual Energy Outlook 2025, Table 8',
          'metric': 'clean_share_pct',  # renewables + nuclear as % of generation
          'national': {2025: 42, 2030: 47, 2035: 51, 2040: 54, 2050: 58},
          # Note: AEO is national, not ISO-specific. Use as directional benchmark.
      }

   b. NREL Standard Scenarios 2024 Mid-Case — capacity additions:
      REEDS_MID = {
          'source': 'NREL Standard Scenarios 2024, Mid-Case',
          'metric': 'clean_share_pct',
          'national': {2025: 43, 2030: 52, 2035: 60, 2040: 67, 2050: 78},
          # ReEDS is cost-minimizing with RPS constraints — expect higher clean %
      }

   c. EPA IPM v6 Reference Case — coal retirement:
      EPA_IPM_REF = {
          'source': 'EPA IPM v6 Reference Case (2023)',
          'metric': 'coal_share_pct',
          'national': {2025: 16, 2030: 12, 2035: 8, 2040: 5, 2050: 2},
      }

   IMPORTANT: These are approximate values from public reports. Include source
   citations and note they should be verified against the actual published tables.
   The exact numbers matter less than having the structure in place.

2. Create helper functions (no simulation calls):

   def build_reference_scenario_params():
       """Return the conditions dict for a 'reference-equivalent' scenario.
       Maps to: Medium demand, Medium LCOE, Medium gas, $0 carbon (AEO comparison)."""
       return {
           'name': 'CV_Reference_Zero_Carbon',
           'demand_growth': 'Medium',
           'lcoe_level': 'Medium',
           'fuel_level': 'Medium',
           'carbon_price': 0,
           'learning_rate': 'Medium',
           '45q_enabled': True,
           'queue_cap_level': 'Medium',
       }

   def build_epa_scenario_params():
       """Return conditions for EPA-comparable scenario ($51/ton carbon)."""
       params = build_reference_scenario_params()
       params['name'] = 'CV_Reference_EPA_Carbon'
       params['carbon_price'] = 51
       return params

   def compute_divergence_metrics(model_results, reference_data, milestone_years=[2030, 2035, 2040]):
       """Compare model outputs against reference trajectories.
       Returns dict with divergence at each milestone year."""
       # Extract clean_pct from model_results at each milestone year
       # Compute absolute and relative divergence vs reference
       # Return structured comparison dict
       pass  # Implement the comparison logic

   def format_comparison_table(divergences):
       """Format divergence metrics as a markdown table string."""
       pass

3. Create a run_cross_validation() function that:
   - Accepts pre-computed simulation results as input (dict of {iso: [year_results]})
   - Does NOT call run_market_simulation() — results are passed in
   - Calls compute_divergence_metrics() for each ISO × reference pair
   - Returns structured comparison dict

4. Add a __main__ block that:
   - Checks if cached results exist at market-simulator/data/cv_reference_results.json
   - If yes: loads and runs comparison
   - If no: prints "Run Prompt 5B first to generate simulation results"

5. Add non-blocking test class TestCrossValidationStructure:
   - Test that reference data dicts have required keys
   - Test that build_reference_scenario_params() returns valid conditions
   - Test that compute_divergence_metrics() handles empty inputs gracefully
   - These tests should PASS without any simulation data

Key files: New script only. Import from market_simulation.py only for type
reference, not for execution.
```

---

## Prompt 5B: Run Simulations + Cache Results (compute-only)

```
In market-simulator/, run the cross-validation simulations and cache results to JSON.
This is a compute-only prompt — no documentation, no analysis.

1. Create a small runner script market-simulator/scripts/run_cv_simulations.py:

   import sys, json, time
   sys.path.insert(0, 'scripts')
   from market_simulation import run_market_simulation, load_common_data, load_egrid_baselines

   # Pre-load data once
   demand_data, gen_profiles, emission_rates, fossil_mix = load_common_data()
   egrid_baselines = load_egrid_baselines()
   preloaded = {
       'demand_data': demand_data,
       'gen_profiles': gen_profiles,
       'emission_rates': emission_rates,
       'fossil_mix': fossil_mix,
       'egrid_baselines': egrid_baselines,
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
   # Convert numpy types for JSON serialization
   import numpy as np
   class NumpyEncoder(json.JSONEncoder):
       def default(self, obj):
           if isinstance(obj, (np.integer,)): return int(obj)
           if isinstance(obj, (np.floating,)): return float(obj)
           if isinstance(obj, np.ndarray): return obj.tolist()
           return super().default(obj)

   with open(output_path, 'w') as f:
       json.dump(results, f, cls=NumpyEncoder, indent=2)
   print(f"\nResults cached to {output_path}")

2. Run it: cd market-simulator && python scripts/run_cv_simulations.py

3. Commit the cached results JSON and the runner script.

That's it — no analysis, no documentation. Just run and cache.
```

---

## Prompt 5C: Analysis + Documentation + Test Integration (no simulation)

```
In market-simulator/, use the cached cross-validation results to generate the
comparison analysis and documentation. No simulations — read from cache only.

1. In scripts/tests/validate_cross_model.py (created in Prompt 5A), implement
   the compute_divergence_metrics() function:

   For each ISO × reference model pair, compute:
   - Clean share divergence at 2030, 2035, 2040 (absolute percentage points)
   - If coal data available: coal retirement timing divergence
   - Flag divergences as "expected" or "investigate":
     * Expected: This model is profit-driven → lower clean % than cost-minimizing
       models that assume RPS mandates (ReEDS will be higher)
     * Expected: This model doesn't enforce policy → coal retires slower than EPA IPM
     * Investigate: If this model shows HIGHER clean % than ReEDS (shouldn't happen
       without mandates)
     * Investigate: If divergence exceeds 20 percentage points at any milestone

2. Run the analysis: load data/cv_reference_results.json, call run_cross_validation(),
   generate the comparison table.

3. Create market-simulator/docs/Cross_Validation_Results.md with:
   - Summary table: ISO × Reference Model × Milestone Year → Divergence
   - Expected Divergences section (with explanations)
   - Unexplained Divergences section (if any, with investigation notes)
   - Methodology Note: explain why profit-driven ≠ cost-minimizing and what
     divergence tells us vs. doesn't tell us

4. Add TestCrossValidationResults test class (non-blocking — warnings not failures):
   - Load cached results, run divergence computation
   - Assert divergence is computed for all 3 ISOs × 3 milestone years
   - Warn (don't fail) if any divergence exceeds 20 percentage points
   - Warn if model shows higher clean % than ReEDS Mid-Case (unexpected)

5. Commit everything: updated validate_cross_model.py, Cross_Validation_Results.md.

Key files: scripts/tests/validate_cross_model.py, data/cv_reference_results.json (read-only),
docs/Cross_Validation_Results.md (new)
```

---

## Execution Order

```
Prompt 5A → Prompt 5B → Prompt 5C
```

- **5A** (code structure, no compute): ~15 min session
- **5B** (compute only): ~20 min session depending on simulation runtime
- **5C** (analysis + docs): ~15 min session

5A and 5B have no dependency on G1 (plant-level retirement). If G1 is implemented
later, just re-run 5B to regenerate cached results with plant-level retirement,
then re-run 5C to update the analysis.
