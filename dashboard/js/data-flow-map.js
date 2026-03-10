// Auto-generated pipeline data flow map
const PIPELINE_FLOW = {
  nodes: [
    {id: 'raw_data', label: 'EIA/eGRID Raw Data', type: 'data_input', description: 'EIA-930 hourly generation/demand profiles, eGRID emission factors, fossil fuel mix data', scripts: ['step0_fetch_all_data.py', 'step0_fetch_egrid.py', 'step0_fetch_eia_multiyear.py', 'step0_fetch_lmp_2025.py', 'step0_fetch_offshore_wind.py'], runtime: 'Minutes (API calls)', outputs: ['eia_generation_profiles_multiyear.json', 'eia_demand_profiles_multiyear.json', 'egrid_emission_rates.json', 'eia_fossil_mix_multiyear.json', 'offshore_wind_profiles.parquet']},
    {id: 'config', label: 'Pipeline Config', type: 'config', description: 'Single source of truth: LCOE tables, storage parameters, demand growth rates, emission rates, toggle scenarios (5,832 per ISO, 17,496 for CAISO)', scripts: ['pipeline_config.py'], runtime: 'N/A (imported)'},
    {id: 'step1', label: 'Step 1: Physics Feasible Space', type: 'compute_heavy', description: '4D/5D/6D adaptive grid search over clean_firm \u00d7 solar \u00d7 wind \u00d7 hydro [\u00d7 offshore_wind] [\u00d7 geothermal] + battery (4hr/8hr) + LDES (100hr) + Green H\u2082 (1000hr) dispatch. Hourly 8,760-hour matching score. ~21M feasible mixes across 7 ISOs \u00d7 21 thresholds.', scripts: ['step1_pfs_generator.py', 'step1a_generate_mixes.py', 'step1b_score_mixes.py', 'step1b_zone_search.py', 'step1b2_floor_aware_pfs.py', 'step1b3_fine_grid_pfs.py', 'step1c_storage_refinement.py'], runtime: '3\u20138 hours (full run)', outputs: ['{ISO}_t{T}_raw_pfs.parquet', '{ISO}_t{T}_storage.parquet']},
    {id: 'step2', label: 'Step 2: Efficient Frontier', type: 'compute', description: 'Non-dominated mix extraction: existing generation utilization filter, procurement minimization, strict dominance removal. Reduces ~21M PFS mixes to ~500K EF mixes.', scripts: ['step2_efficient_frontier.py'], runtime: '2\u20135 minutes', outputs: ['step2_ef_{ISO}_t{T}.parquet']},
    {id: 'step3', label: 'Step 3: Cost Optimization', type: 'compute', description: 'Track 1: Vectorized cross-evaluation of EF mixes under 5,832 cost scenarios (17,496 for CAISO). Merit-order tranche pricing. Track 2 (newbuild) + Track 3 (cost-to-replace) with Wright\'s Law FOAK\u2192NOAK learning curves.', scripts: ['step3a_cost_optimization.py', 'step3b_track_nb_ctr.py'], runtime: '10\u201330 minutes', outputs: ['step3_co_{ISO}.parquet']},
    {id: 'step4', label: 'Step 4: Dispatch Cache', type: 'compute', description: 'Pre-computes 8,760-hour dispatch profiles for all unique winning mixes. Versioned NPZ cache (v2) with per-resource matched/surplus + charge profiles.', scripts: ['step4_build_dispatch_cache.py'], runtime: '5\u201315 minutes', outputs: ['{ISO}_dispatch_cache.npz']},
    {id: 'step5', label: 'Step 5: Core Analysis', type: 'analysis', description: '6 parallel scripts: CO\u2082 dispatch-stack model, synthetic LMP pricing, 24-hour representative day profiles, cross-regional deployment queue, track exports, storage utilization analysis.', scripts: ['step5a_compute_co2.py', 'step5b_compute_lmp_prices.py', 'step5c_compress_day_profiles.py', 'step5d_deployment_queue.py', 'step5e_export_tracks.py', 'step5f_analyze_storage.py'], runtime: '5\u201320 minutes (parallel)', outputs: ['co2_{ISO}.parquet', '{ISO}_lmp.parquet', 'day_profiles/', 'mac_queue/', 'track_results.json', 'storage_analysis.json']},
    {id: 'step6', label: 'Step 6: Derived Analytics', type: 'analysis', description: 'MAC statistics (6 metrics with ANOVA decomposition), optimal CFE targets per ISO (DAC crossover via PCHIP spline), track cost envelopes, building blocks, resource density.', scripts: ['step6a_compute_mac_stats.py', 'step6b_compute_optimal_targets.py', 'step6c_analyze_tracks.py', 'step6d_extract_building_blocks.py', 'step6e_extract_resource_density.py'], runtime: '2\u201310 minutes', outputs: ['mac_stats.json', 'optimal_targets.json', 'track_envelopes.json']},
    {id: 'step7', label: 'Step 7: Scenario Comparison', type: 'analysis', description: 'Consequential (MAC-queue) vs. hourly matching procurement strategies. Uses optimal targets from Step 6B.', scripts: ['step7b_scenario_hourly.py', 'step7c_scenario_comparison.py'], runtime: '2\u20135 minutes', outputs: ['scenario_comparison.json']},
    {id: 'step8', label: 'Step 8: Procurement Strategies', type: 'analysis', description: '3 strategy families \u00d7 3\u20134 variants each: (1) Cross-regional consequential, (2) Hourly matching same-ISO, (3) Annual matching 2\u00d72 matrix. Wright\'s Law learning curves. 25-year SBTi timeline.', scripts: ['step8a_strategy_consequential.py', 'step8b_strategy_hourly.py', 'step8c_strategy_annual.py', 'step8d_wrights_law_curves.py'], runtime: '5\u201315 minutes', outputs: ['strategy_comparison.json', 'wrights_law_curves.json']},
    {id: 'step9', label: 'Step 9: Dashboard Data', type: 'output', description: 'Aggregates all upstream results into dashboard-ready JS files. SBTi milestone mapping, DAC trajectory projections, LCOE/transmission tables for client-side repricing.', scripts: ['step9a_generate_shared_data.py', 'step9b_extract_no_regrets.py'], runtime: '1\u20133 minutes', outputs: ['shared-data.js (~5.2MB)', 'optimal-target-data.js (~117KB)', 'procurement-strategy-data.js (~3.8MB)']},
    {id: 'dashboard', label: 'Interactive Dashboard', type: 'output', description: '20+ interactive HTML pages: cost optimizer, abatement analysis, scenario comparison, LMP trends, storage analysis, procurement strategies, methodology, research paper.', scripts: [], runtime: 'N/A (client-side)'}
  ],
  edges: [
    {from: 'raw_data', to: 'step1', label: 'EIA hourly profiles'},
    {from: 'config', to: 'step1', label: 'Storage params, thresholds', style: 'dashed'},
    {from: 'config', to: 'step2', label: 'Filter criteria', style: 'dashed'},
    {from: 'config', to: 'step3', label: 'LCOE tables, toggles', style: 'dashed'},
    {from: 'step1', to: 'step2', label: '~21M PFS mixes'},
    {from: 'step2', to: 'step3', label: '~500K EF mixes'},
    {from: 'step3', to: 'step4', label: 'Winning mixes'},
    {from: 'step4', to: 'step5', label: 'Dispatch cache'},
    {from: 'step3', to: 'step5', label: 'Cost results'},
    {from: 'step3', to: 'step6', label: 'EF mixes + costs'},
    {from: 'step5', to: 'step6', label: 'CO\u2082, MAC queue'},
    {from: 'step5', to: 'step7', label: 'Deployment queue'},
    {from: 'step6', to: 'step7', label: 'Optimal targets'},
    {from: 'step5', to: 'step8', label: 'Dispatch + CO\u2082'},
    {from: 'step6', to: 'step8', label: 'Learning curves'},
    {from: 'step5', to: 'step9', label: 'All Step 5 outputs'},
    {from: 'step6', to: 'step9', label: 'All Step 6 outputs'},
    {from: 'step7', to: 'step9', label: 'Scenario results'},
    {from: 'step8', to: 'step9', label: 'Strategy results'},
    {from: 'step9', to: 'dashboard', label: 'JS data files'}
  ],
  // Data file reference table
  dataFiles: [
    {pattern: 'eia_generation_profiles_multiyear.json', step: 0, format: 'JSON', keyColumns: 'hour, fuel_type, mwh_by_iso', description: 'EIA-930 hourly generation by fuel type, 2019\u20132025', source: 'EIA Open Data API'},
    {pattern: 'eia_demand_profiles_multiyear.json', step: 0, format: 'JSON', keyColumns: 'hour, demand_mwh_by_iso', description: 'EIA-930 hourly demand, 2019\u20132025', source: 'EIA Open Data API'},
    {pattern: 'egrid_emission_rates.json', step: 0, format: 'JSON', keyColumns: 'iso, fuel, lbs_co2_per_mwh', description: 'eGRID 2023 emission rates by fuel and ISO', source: 'EPA eGRID 2023 Rev 2'},
    {pattern: 'eia_fossil_mix_multiyear.json', step: 0, format: 'JSON', keyColumns: 'iso, year, coal_pct, gas_pct, oil_pct', description: 'Regional fossil fuel generation mix', source: 'EIA-923'},
    {pattern: 'offshore_wind_profiles.parquet', step: 0, format: 'Parquet', keyColumns: 'hour, capacity_factor_by_iso', description: 'Synthetic offshore wind profiles from NREL data', source: 'NREL Wind Integration National Dataset'},
    {pattern: '{ISO}_t{T}_raw_pfs.parquet', step: 1, format: 'Parquet', keyColumns: 'clean_firm, solar, wind, hydro, [offshore_wind], [geothermal], bat4_pct, bat8_pct, ldes_pct, h2_pct, score', description: 'Physics feasible mixes meeting hourly matching threshold', approxSize: '~500MB/ISO'},
    {pattern: '{ISO}_t{T}_storage.parquet', step: '1C', format: 'Parquet', keyColumns: 'same as PFS + optimized storage dispatch percentages', description: 'Storage-refined feasible mixes (battery 4hr/8hr + LDES + Green H\u2082)', approxSize: '~200MB/ISO'},
    {pattern: 'step2_ef_{ISO}_t{T}.parquet', step: 2, format: 'Parquet', keyColumns: 'same + iso, hourly_match_score, pareto_type, gen_util_score', description: 'Efficient frontier: non-dominated mixes after dominance + utilization filtering', approxSize: '~50MB/ISO'},
    {pattern: 'step3_co_{ISO}.parquet', step: 3, format: 'Parquet', keyColumns: 'mix_* (composition) + cost_* (5,832 scenarios) + track columns', description: 'Cost-optimized results: every EF mix priced under all sensitivity scenarios', approxSize: '~200MB/ISO'},
    {pattern: '{ISO}_dispatch_cache.npz', step: 4, format: 'NPZ v2', keyColumns: '8,760-hour arrays: matched_*, surplus_*, charge_* per resource', description: 'Pre-computed hourly dispatch for all unique winning mixes', approxSize: '~100MB/ISO'},
    {pattern: 'co2_{ISO}.parquet', step: '5A', format: 'Parquet', keyColumns: 'threshold, fossil_gas_twh, fossil_coal_twh, co2_mt, dispatch_stack', description: 'CO\u2082 dispatch-stack model with merit-order fossil retirement (coal\u2192oil\u2192gas)', approxSize: '~10MB/ISO'},
    {pattern: '{ISO}_lmp.parquet', step: '5B', format: 'Parquet', keyColumns: 'hour, lmp_p10, lmp_p50, lmp_p90, marginal_fuel', description: 'Synthetic hourly LMP from merit-order fossil cost stack', approxSize: '~5MB/ISO'},
    {pattern: 'shared-data.js', step: 9, format: 'JavaScript', keyColumns: 'MAC curves, cost envelopes, resource mixes, benchmarks', description: 'Main dashboard data: all regions, thresholds, cost scenarios', approxSize: '~5.2MB'},
    {pattern: 'optimal-target-data.js', step: 9, format: 'JavaScript', keyColumns: 'optimal_targets, dac_crossover, no_regrets_resources', description: 'Optimal CFE targets per ISO under 3\u00d73 grid-cost \u00d7 DAC matrix', approxSize: '~117KB'},
    {pattern: 'procurement-strategy-data.js', step: 9, format: 'JavaScript', keyColumns: 'strategy_costs, wrights_law, deployment_timelines', description: 'Procurement strategy comparison: 3 families \u00d7 3-4 variants \u00d7 25 years', approxSize: '~3.8MB'}
  ]
};
