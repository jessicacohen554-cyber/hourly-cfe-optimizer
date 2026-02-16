#!/usr/bin/env python3
"""
Generate complete shared-data.js from overprocure_results.json + mac_stats.json.
Replaces the entire file with fresh data from the latest pipeline run.

Pipeline order: Step 1 (physics) → Step 2 (tranche) → postprocess → co2 → mac_stats → THIS

Input:  dashboard/overprocure_results.json  (final pipeline output)
        data/mac_stats.json                 (MAC statistics from compute_mac_stats.py)
Output: dashboard/js/shared-data.js         (complete rewrite)
"""

import json
import os
from datetime import datetime

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
THRESHOLDS = ['75', '80', '85', '87.5', '90', '92.5', '95', '97.5', '99', '100']
THRESHOLDS_NUM = [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99, 100]
RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']
MATCHED_RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro', 'battery', 'ldes']
SCENARIO_KEY = 'MMM_M_M'
MAC_CAP = 1000  # Cap marginal MAC at $1000/ton

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
with open('dashboard/overprocure_results.json') as f:
    data = json.load(f)
with open('data/mac_stats.json') as f:
    mac_stats = json.load(f)

print(f"  Results: {os.path.getsize('dashboard/overprocure_results.json') / 1024:.0f} KB")
print(f"  MAC stats: {os.path.getsize('data/mac_stats.json') / 1024:.0f} KB")

# ============================================================================
# EXTRACT MAC_DATA (average MAC — medium/low/high)
# ============================================================================

print("\nExtracting MAC_DATA...")
mac_data = {'medium': {}, 'low': {}, 'high': {}}
for iso in ISOS:
    env = mac_stats['envelope'][iso]['envelope']
    fan = mac_stats['fan_chart'][iso]

    # Medium = envelope (monotonic running-max of MMM_M_M)
    mac_data['medium'][iso] = [round(v) if v is not None else None for v in env]
    # Low = P10 of fan chart
    mac_data['low'][iso] = [round(v) if v is not None else None for v in fan['p10']]
    # High = P90 of fan chart
    mac_data['high'][iso] = [round(v) if v is not None else None for v in fan['p90']]

    print(f"  {iso} medium: {mac_data['medium'][iso]}")

# ============================================================================
# EXTRACT MARGINAL_MAC_DATA (5-zone stepwise — medium/low/high)
# ============================================================================

print("\nExtracting MARGINAL_MAC_DATA...")
marginal_mac_data = {'medium': {}, 'low': {}, 'high': {}}

for iso in ISOS:
    # Medium: use stepwise_envelope (monotonic)
    sw_env = mac_stats['envelope'][iso]['stepwise_envelope']
    # Steps 1-4 = 75→80, 80→85, 85→87.5, 87.5→90 → Zone 0 (backbone)
    backbone_steps = [sw_env[i] for i in range(1, 5) if sw_env[i] is not None]
    zone0 = round(sum(backbone_steps) / len(backbone_steps)) if backbone_steps else None
    zone0 = min(zone0, MAC_CAP) if zone0 is not None else None

    zones = [zone0]
    # Steps 5-8 = 90→92.5, 92.5→95, 95→97.5, 97.5→99 → Zones 1-4
    for step_idx in range(5, 9):
        v = sw_env[step_idx] if step_idx < len(sw_env) else None
        zones.append(min(round(v), MAC_CAP) if v is not None else None)
    marginal_mac_data['medium'][iso] = zones

    # Low: use stepwise_fan P10
    sw_lo = mac_stats['stepwise_fan'][iso]['p10']
    backbone_lo = [sw_lo[i] for i in range(1, 5) if sw_lo[i] is not None]
    zone0_lo = round(sum(backbone_lo) / len(backbone_lo)) if backbone_lo else None
    zone0_lo = min(zone0_lo, MAC_CAP) if zone0_lo is not None else None
    lo_zones = [zone0_lo]
    for step_idx in range(5, 9):
        v = sw_lo[step_idx] if step_idx < len(sw_lo) else None
        lo_zones.append(min(round(v), MAC_CAP) if v is not None else None)
    marginal_mac_data['low'][iso] = lo_zones

    # High: use stepwise_fan P90
    sw_hi = mac_stats['stepwise_fan'][iso]['p90']
    backbone_hi = [sw_hi[i] for i in range(1, 5) if sw_hi[i] is not None]
    zone0_hi = round(sum(backbone_hi) / len(backbone_hi)) if backbone_hi else None
    zone0_hi = min(zone0_hi, MAC_CAP) if zone0_hi is not None else None
    hi_zones = [zone0_hi]
    for step_idx in range(5, 9):
        v = sw_hi[step_idx] if step_idx < len(sw_hi) else None
        hi_zones.append(min(round(v), MAC_CAP) if v is not None else None)
    marginal_mac_data['high'][iso] = hi_zones

    print(f"  {iso} medium: {marginal_mac_data['medium'][iso]}")

# ============================================================================
# EXTRACT EFFECTIVE_COST_DATA
# ============================================================================

print("\nExtracting EFFECTIVE_COST_DATA...")
effective_cost_data = {}
for iso in ISOS:
    costs = []
    for t in THRESHOLDS:
        sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(SCENARIO_KEY)
        if sc:
            costs.append(round(sc['costs']['effective_cost'], 1))
        else:
            costs.append(None)
    effective_cost_data[iso] = costs
    print(f"  {iso}: {costs}")

# Enforce monotonicity (lower thresholds <= higher thresholds)
for iso in ISOS:
    arr = effective_cost_data[iso]
    # Top-down ceiling enforcement
    for i in range(len(arr) - 2, -1, -1):
        if arr[i] is not None and arr[i + 1] is not None and arr[i] > arr[i + 1]:
            arr[i] = arr[i + 1]

# ============================================================================
# EXTRACT RESOURCE_MIX_DATA
# ============================================================================

print("\nExtracting RESOURCE_MIX_DATA...")
resource_mix_data = {}
for iso in ISOS:
    iso_data = {r: [] for r in RESOURCES}
    iso_data['battery'] = []
    iso_data['ldes'] = []
    iso_data['procurement'] = []

    for t in THRESHOLDS:
        sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(SCENARIO_KEY)
        if sc:
            rm = sc.get('resource_mix', {})
            for res in RESOURCES:
                iso_data[res].append(rm.get(res, 0))
            iso_data['battery'].append(sc.get('battery_dispatch_pct', 0))
            iso_data['ldes'].append(sc.get('ldes_dispatch_pct', 0))
            iso_data['procurement'].append(sc.get('procurement_pct', 100))
        else:
            for res in RESOURCES:
                iso_data[res].append(0)
            iso_data['battery'].append(0)
            iso_data['ldes'].append(0)
            iso_data['procurement'].append(100)

    resource_mix_data[iso] = iso_data
    print(f"  {iso} clean_firm: {iso_data['clean_firm']}")

# ============================================================================
# EXTRACT COMPRESSED_DAY_DATA
# ============================================================================

print("\nExtracting COMPRESSED_DAY_DATA...")
compressed_day_data = {}
for iso in ISOS:
    iso_cd = {
        'demand': [],
        'matched': {r: [] for r in MATCHED_RESOURCES},
        'gap': [],
        'battery_charge': [],
        'ldes_charge': [],
    }

    for t in THRESHOLDS:
        sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(SCENARIO_KEY)
        if sc and 'compressed_day' in sc:
            cd = sc['compressed_day']
            iso_cd['demand'].append([round(v, 5) for v in cd['demand']])
            iso_cd['gap'].append([round(v, 5) for v in cd['gap']])
            iso_cd['battery_charge'].append([round(v, 5) for v in cd.get('battery_charge', [0]*24)])
            iso_cd['ldes_charge'].append([round(v, 5) for v in cd.get('ldes_charge', [0]*24)])
            for res in MATCHED_RESOURCES:
                vals = cd.get('matched', {}).get(res, [0]*24)
                iso_cd['matched'][res].append([round(v, 5) for v in vals])
        else:
            iso_cd['demand'].append([0]*24)
            iso_cd['gap'].append([0]*24)
            iso_cd['battery_charge'].append([0]*24)
            iso_cd['ldes_charge'].append([0]*24)
            for res in MATCHED_RESOURCES:
                iso_cd['matched'][res].append([0]*24)

    compressed_day_data[iso] = iso_cd
    print(f"  {iso}: {len(iso_cd['demand'])} thresholds × 24h")

# ============================================================================
# EXTRACT CF_TRANCHE_DATA
# ============================================================================

print("\nExtracting CF_TRANCHE_DATA...")
cf_tranche_data = {}
for iso in ISOS:
    iso_tr = {k: [] for k in ['new_cf_twh', 'uprate_twh', 'newbuild_twh',
                               'uprate_price', 'newbuild_price', 'effective_cf_lcoe']}
    for t in THRESHOLDS:
        sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(SCENARIO_KEY)
        tc = sc.get('tranche_costs', {}) if sc else {}
        iso_tr['new_cf_twh'].append(round(tc.get('new_cf_twh', 0), 3))
        iso_tr['uprate_twh'].append(round(tc.get('uprate_twh', 0), 4))
        iso_tr['newbuild_twh'].append(round(tc.get('newbuild_twh', 0), 3))
        iso_tr['uprate_price'].append(tc.get('uprate_price', 0))
        iso_tr['newbuild_price'].append(round(tc.get('newbuild_price', 0), 1))
        iso_tr['effective_cf_lcoe'].append(round(tc.get('effective_new_cf_lcoe', 0), 1))
    cf_tranche_data[iso] = iso_tr

# ============================================================================
# EXTRACT WYN_RESOURCE_COSTS
# ============================================================================

print("\nExtracting WYN_RESOURCE_COSTS...")
WYN_RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro', 'battery', 'ldes']
wyn_resource_costs = {}
for iso in ISOS:
    iso_wyn = []
    for t in THRESHOLDS:
        sc = data['results'][iso]['thresholds'].get(t, {}).get('scenarios', {}).get(SCENARIO_KEY)
        rc = sc.get('costs_detail', {}).get('resource_costs', {}) if sc else {}
        entry = {}
        for res in WYN_RESOURCES:
            rd = rc.get(res, {})
            if res in ('battery', 'ldes'):
                entry[res] = {
                    'dispatch_pct': rd.get('dispatch_pct', 0),
                    'cost': round(rd.get('cost_per_demand_mwh', 0), 2),
                }
            else:
                entry[res] = {
                    'existing_pct': round(rd.get('existing_pct', rd.get('existing_share', 0)), 1),
                    'new_pct': round(rd.get('new_pct', rd.get('new_share', 0)), 1),
                    'cost': round(rd.get('cost_per_demand_mwh', 0), 2),
                }
        iso_wyn.append(entry)
    wyn_resource_costs[iso] = iso_wyn

# ============================================================================
# FORMAT AS JAVASCRIPT
# ============================================================================

print("\nGenerating shared-data.js...")

def fmt_array(arr):
    return '[' + ', '.join('null' if v is None else str(v) for v in arr) + ']'

def fmt_24h_array(arr):
    return '[' + ','.join(str(v) for v in arr) + ']'

lines = []

# HEADER
lines.append('// ============================================================================')
lines.append('// SHARED DATA MODULE — Single source of truth for all dashboard pages')
lines.append('// ============================================================================')
lines.append('// RULE: No data constants defined in HTML files. Change here, propagates everywhere.')
lines.append(f'// Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} by generate_shared_data.py')
lines.append('// Source: overprocure_results.json (Step 2 tranche-repriced + postprocess + CO2)')
lines.append('// ============================================================================')
lines.append('')

# THRESHOLDS
lines.append(f'// --- Thresholds (from optimizer) ---')
lines.append(f'const THRESHOLDS = {fmt_array(THRESHOLDS_NUM)};')
lines.append('')

# MAC_DATA
lines.append('// --- Average MAC ($/ton CO2) ---')
lines.append('// Medium = monotonic envelope of MMM_M_M scenario')
lines.append('// Low/High = P10/P90 from 324-scenario factorial experiment')
lines.append('const MAC_DATA = {')
for sens in ['medium', 'low', 'high']:
    lines.append(f'    {sens}: {{')
    for iso_idx, iso in enumerate(ISOS):
        comma = ',' if iso_idx < len(ISOS) - 1 else ''
        lines.append(f'        {iso}:  {fmt_array(mac_data[sens][iso])}{comma}')
    comma = ',' if sens != 'high' else ''
    lines.append(f'    }}{comma}')
lines.append('};')
lines.append('')

# REGION_COLORS
lines.append('// --- Region Colors (used by abatement pages) ---')
lines.append("const REGION_COLORS = {")
lines.append("    CAISO: '#F59E0B',")
lines.append("    ERCOT: '#22C55E',")
lines.append("    PJM:   '#4a90d9',")
lines.append("    NYISO: '#E91E63',")
lines.append("    NEISO: '#9C27B0'")
lines.append("};")
lines.append('')

# MIX_RESOURCES, MIX_LABELS_MAP, MIX_COLORS
lines.append("// --- Resource Colors & Labels (used by dashboard, index, region_deepdive) ---")
lines.append("const MIX_RESOURCES = ['clean_firm', 'ccs_ccgt', 'solar', 'wind', 'hydro', 'battery', 'ldes'];")
lines.append('')
lines.append('const MIX_LABELS_MAP = {')
lines.append("    clean_firm: 'Clean Firm',")
lines.append("    ccs_ccgt:   'CCS-CCGT',")
lines.append("    solar:      'Solar',")
lines.append("    wind:       'Wind',")
lines.append("    hydro:      'Hydro',")
lines.append("    battery:    'Battery (4hr)',")
lines.append("    ldes:       'LDES (100hr)'")
lines.append('};')
lines.append('')
lines.append('const MIX_COLORS = {')
lines.append("    clean_firm: { fill: 'rgba(30,58,95,0.50)',    border: '#1E3A5F' },")
lines.append("    ccs_ccgt:   { fill: 'rgba(13,148,136,0.50)',  border: '#0D9488' },")
lines.append("    solar:      { fill: 'rgba(245,158,11,0.50)',  border: '#F59E0B' },")
lines.append("    wind:       { fill: 'rgba(34,197,94,0.50)',   border: '#22C55E' },")
lines.append("    hydro:      { fill: 'rgba(14,165,233,0.50)',  border: '#0EA5E9' },")
lines.append("    battery:    { fill: 'rgba(139,92,246,0.50)',  border: '#8B5CF6' },")
lines.append("    ldes:       { fill: 'rgba(236,72,153,0.50)',  border: '#EC4899' }")
lines.append('};')
lines.append('')

# BENCHMARKS_STATIC
lines.append('// --- Benchmark Data (static — researched L/M/H with sources) ---')
lines.append('const BENCHMARKS_STATIC = [')
lines.append("    { name: 'Energy Efficiency (Buildings)', short: 'Energy Efficiency', low: -100, mid: 0,   high: 60,   color: '#4CAF50', category: 'demand_reduction', trajectory: 'stable', confidence: 'high',")
lines.append("      sources: 'EDF/Evolved Energy MAC 2.0, World Bank, Gillingham & Stock' },")
lines.append("    { name: 'EU ETS Price',                  short: 'EU ETS', low: 65,   mid: 88,  high: 92,   color: '#2196F3', category: 'benchmark', trajectory: 'rising', confidence: 'high',")
lines.append("      sources: 'Trading Economics, Sandbag, BNEF' },")
lines.append("    { name: 'SCC \\u2014 EPA ($190/ton)',     short: 'SCC (EPA)', low: 140,  mid: 190, high: 380,  color: '#FF9800', category: 'benchmark', trajectory: 'rising', confidence: 'medium',")
lines.append("      sources: 'EPA Dec 2023 Report' },")
lines.append("    { name: 'SCC \\u2014 Rennert et al.',     short: 'SCC (Rennert)', low: 120,  mid: 185, high: 450,  color: '#E65100', category: 'benchmark', trajectory: 'rising', confidence: 'medium',")
lines.append("      sources: 'Rennert et al. (2022) Nature' },")
lines.append("    { name: 'Carbon Credits (Nature)',       short: 'Carbon Credits', low: 3,    mid: 15,  high: 35,   color: '#9E9E9E', category: 'voluntary', trajectory: 'rising', confidence: 'medium',")
lines.append("      sources: 'Sylvera 2026, Regreener, MSCI' }")
lines.append('];')
lines.append('')

# BENCHMARKS_DYNAMIC
lines.append('// --- Benchmark Data (dynamic — shift with user toggles) ---')
lines.append('const BENCHMARKS_DYNAMIC = {')
lines.append("    dac: {")
lines.append("        name: 'Direct Air Capture (DAC)', short: 'DAC', color: '#E91E63', category: 'carbon_removal', confidence: 'low',")
lines.append("        trajectory: 'declining_steep',")
lines.append("        Low:    { low: 65,   mid: 100,  high: 175 },")
lines.append("        Medium: { low: 100,  mid: 175,  high: 300 },")
lines.append("        High:   { low: 175,  mid: 300,  high: 500 },")
lines.append("        sources: 'DOE Liftoff NOAK, IEAGHG 2021, Fasihi et al. (J. Cleaner Prod. 2019), Sievert et al. (Joule 2024), Climeworks Gen 3 roadmap, DOE Carbon Negative Shot, Kanyako & Craig (Earth\\'s Future 2025)'")
lines.append("    },")
lines.append("    industrial: {")
lines.append("        name: 'Industrial Electrification', short: 'Ind. Electrification', color: '#8BC34A', category: 'industrial_decarb', confidence: 'medium',")
lines.append("        trajectory: 'declining',")
lines.append("        Low:    { low: -50, mid: 20,  high: 60 },")
lines.append("        Medium: { low: 0,   mid: 60,  high: 160 },")
lines.append("        High:   { low: 60,  mid: 120, high: 250 },")
lines.append("        sources: 'McKinsey, Thunder Said Energy, Rewiring America'")
lines.append("    },")
lines.append("    removal: {")
lines.append("        name: 'Carbon Removal (BECCS + Enhanced Weathering)', short: 'Carbon Removal\\u00B9', color: '#009688', category: 'carbon_removal', confidence: 'low',")
lines.append("        trajectory: 'declining',")
lines.append("        Low:    { low: 20,  mid: 75,  high: 150 },")
lines.append("        Medium: { low: 50,  mid: 150, high: 300 },")
lines.append("        High:   { low: 100, mid: 200, high: 350 },")
lines.append("        sources: 'ORNL, Nature 2024, Nature Comms 2025'")
lines.append("    }")
lines.append('};')
lines.append('')

# BENCHMARKS_EXTRA
lines.append('// --- Extra Benchmarks (not toggle-controlled) ---')
lines.append('const BENCHMARKS_EXTRA = [')
lines.append("    { name: 'Green Hydrogen (Industrial)',  short: 'Green H\\u2082', low: 150, mid: 500, high: 1250, color: '#00BCD4', category: 'industrial_decarb',")
lines.append("      trajectory: 'declining', confidence: 'low', sources: 'Shafiee & Schrag (Joule 2024), Belfer Center' },")
lines.append("    { name: 'Sustainable Aviation Fuel',    short: 'SAF', low: 136, mid: 300, high: 500,  color: '#FF5722', category: 'transport',")
lines.append("      trajectory: 'declining', confidence: 'medium', sources: 'NREL, RMI 2025, ICCT, WEF' },")
lines.append("    { name: 'CDR Credits (Engineered)',     short: 'CDR Credits', low: 177, mid: 320, high: 600,  color: '#7C4DFF', category: 'voluntary',")
lines.append("      trajectory: 'declining', confidence: 'medium', sources: 'Sylvera, CarbonCredits.com' },")
lines.append("    { name: 'EVs vs ICE (Fleet)',           short: 'EVs vs ICE', low: -50, mid: 250, high: 970,  color: '#FF6F00', category: 'transport',")
lines.append("      trajectory: 'declining_steep', confidence: 'medium', sources: 'Argonne Labs, Penn Wharton, RFF' }")
lines.append('];')
lines.append('')

# MARGINAL_MAC_LABELS + MARGINAL_MAC_DATA
lines.append("// --- Two-Zone Marginal MAC ($/ton CO2) ---")
lines.append("// Zone 0 (75→90%): aggregate backbone MAC")
lines.append("// Zones 1-4 (90→99%): granular steps with monotonicity enforcement")
lines.append("// Cap: $1000/ton (NREL literature max)")
lines.append("const MARGINAL_MAC_LABELS = ['75→90%', '90→92.5%', '92.5→95%', '95→97.5%', '97.5→99%'];")
lines.append('')
lines.append('const MARGINAL_MAC_DATA = {')
for sens in ['medium', 'low', 'high']:
    lines.append(f'    {sens}: {{')
    for iso_idx, iso in enumerate(ISOS):
        comma = ',' if iso_idx < len(ISOS) - 1 else ''
        lines.append(f'        {iso}:  {fmt_array(marginal_mac_data[sens][iso])}{comma}')
    comma = ',' if sens != 'high' else ''
    lines.append(f'    }}{comma}')
lines.append('};')
lines.append('')

# EFFECTIVE_COST_DATA
lines.append('// --- Effective Cost per Useful MWh ($/MWh) ---')
lines.append('// Source: Step 2 tranche-repriced MMM_M_M + postprocess corrections')
lines.append('// Monotonicity enforced (lower thresholds <= higher thresholds)')
thresh_str = ', '.join(str(t) for t in THRESHOLDS_NUM)
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('const EFFECTIVE_COST_DATA = {')
for iso_idx, iso in enumerate(ISOS):
    comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    {iso}:  {fmt_array(effective_cost_data[iso])}{comma}')
lines.append('};')
lines.append('')

# UPRATE_CAPS_TWH
uprate_caps = data['config'].get('tranche_model', {}).get('uprate_caps_twh', {})
if not uprate_caps:
    uprate_caps = {'CAISO': 0.907, 'ERCOT': 1.064, 'PJM': 12.614, 'NYISO': 1.340, 'NEISO': 1.380}
lines.append("// --- Nuclear Uprate Caps (TWh/yr) — 5% of existing nuclear at 90% CF ---")
lines.append("const UPRATE_CAPS_TWH = {")
parts = [f'    {iso}: {uprate_caps.get(iso, 0)}' for iso in ISOS]
lines.append(',\n'.join(parts))
lines.append('};')
lines.append('')

# UTILITY FUNCTIONS
lines.append('// ============================================================================')
lines.append('// SHARED UTILITY FUNCTIONS')
lines.append('// ============================================================================')
lines.append('')
lines.append('function findCrossover(regionData, costLevel) {')
lines.append('    for (let i = 0; i < regionData.length; i++) {')
lines.append('        if (regionData[i] !== null && regionData[i] >= costLevel) return THRESHOLDS[i];')
lines.append('    }')
lines.append("    return '>99';")
lines.append('}')
lines.append('')
lines.append('const MARGINAL_THRESHOLDS = [75, 90, 92.5, 95, 97.5];')
lines.append('function findMarginalCrossover(regionMarginals, costLevel) {')
lines.append('    for (let i = 0; i < regionMarginals.length; i++) {')
lines.append('        if (regionMarginals[i] !== null && regionMarginals[i] > costLevel) {')
lines.append('            return MARGINAL_THRESHOLDS[i];')
lines.append('        }')
lines.append('    }')
lines.append("    return '>99';")
lines.append('}')
lines.append('')
lines.append('function cellClass(val) {')
lines.append("    if (val === '>99' || val === '>100') return 'cell-green';")
lines.append("    if (val >= 97) return 'cell-green';")
lines.append("    if (val >= 95) return 'cell-yellow';")
lines.append("    if (val >= 92) return 'cell-orange';")
lines.append("    return 'cell-red';")
lines.append('}')
lines.append('')
lines.append('function getAllBenchmarks(state) {')
lines.append("    state = state || { dac: 'Medium', industrial: 'Medium', removal: 'Medium' };")
lines.append('    const dynamic = Object.keys(BENCHMARKS_DYNAMIC).map(key => {')
lines.append('        const b = BENCHMARKS_DYNAMIC[key];')
lines.append('        const costs = b[state[key]] || b.Medium;')
lines.append('        return { name: b.name, short: b.short || b.name, low: costs.low, mid: costs.mid, high: costs.high,')
lines.append('                 color: b.color, category: b.category, confidence: b.confidence, trajectory: b.trajectory, sources: b.sources };')
lines.append('    });')
lines.append('    return [...BENCHMARKS_STATIC, ...dynamic, ...BENCHMARKS_EXTRA]')
lines.append('        .filter(b => b.mid >= 0)')
lines.append('        .sort((a, b) => a.mid - b.mid);')
lines.append('}')
lines.append('')

# RESOURCE_MIX_DATA
lines.append('')
lines.append('// --- Resource Mix (% of demand) — MMM_M_M scenario ---')
lines.append('// Source: overprocure_results.json (Step 2 repriced)')
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('// battery/ldes = dispatch % of demand; procurement = over-procurement %')
lines.append('const RESOURCE_MIX_DATA = {')
for iso_idx, iso in enumerate(ISOS):
    d = resource_mix_data[iso]
    lines.append(f'    {iso}: {{')
    for key in RESOURCES + ['battery', 'ldes', 'procurement']:
        comma = ',' if key != 'procurement' else ''
        padding = ' ' * max(0, 12 - len(key))
        lines.append(f'        {key}:{padding}{fmt_array(d[key])}{comma}')
    comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    }}{comma}')
lines.append('};')

# COMPRESSED_DAY_DATA
lines.append('')
lines.append('// --- Compressed Day Hourly Profiles (24h normalized) — MMM_M_M scenario ---')
lines.append('// Source: overprocure_results.json compressed_day field')
lines.append(f'// Each sub-array is [24 hourly values] in UTC, one per threshold (matching THRESHOLDS)')
lines.append('const COMPRESSED_DAY_DATA = {')
for iso_idx, iso in enumerate(ISOS):
    cd = compressed_day_data[iso]
    lines.append(f'    {iso}: {{')

    # demand
    lines.append('        demand: [')
    for i, arr in enumerate(cd['demand']):
        comma = ',' if i < len(cd['demand']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    # matched
    lines.append('        matched: {')
    for res_idx, res in enumerate(MATCHED_RESOURCES):
        res_comma = ',' if res_idx < len(MATCHED_RESOURCES) - 1 else ''
        lines.append(f'            {res}: [')
        for i, arr in enumerate(cd['matched'][res]):
            comma = ',' if i < len(cd['matched'][res]) - 1 else ''
            lines.append(f'                {fmt_24h_array(arr)}{comma}')
        lines.append(f'            ]{res_comma}')
    lines.append('        },')

    # gap
    lines.append('        gap: [')
    for i, arr in enumerate(cd['gap']):
        comma = ',' if i < len(cd['gap']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    # battery_charge
    lines.append('        battery_charge: [')
    for i, arr in enumerate(cd['battery_charge']):
        comma = ',' if i < len(cd['battery_charge']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    # ldes_charge
    lines.append('        ldes_charge: [')
    for i, arr in enumerate(cd['ldes_charge']):
        comma = ',' if i < len(cd['ldes_charge']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ]')

    iso_comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    }}{iso_comma}')
lines.append('};')

# CF_TRANCHE_DATA
lines.append('')
lines.append('// --- CF Tranche Split (uprate vs new-build) — MMM_M_M scenario ---')
lines.append('// Source: overprocure_results.json tranche_costs field')
lines.append(f'// Indices match THRESHOLDS array: [{thresh_str}]')
lines.append('const CF_TRANCHE_DATA = {')
for iso_idx, iso in enumerate(ISOS):
    tr = cf_tranche_data[iso]
    lines.append(f'    {iso}: {{')
    fields = ['new_cf_twh', 'uprate_twh', 'newbuild_twh', 'uprate_price', 'newbuild_price', 'effective_cf_lcoe']
    for fi, field in enumerate(fields):
        comma = ',' if fi < len(fields) - 1 else ''
        padding = ' ' * max(0, 18 - len(field))
        lines.append(f'        {field}:{padding}{fmt_array(tr[field])}{comma}')
    iso_comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    }}{iso_comma}')
lines.append('};')

# WYN_RESOURCE_COSTS
lines.append('')
lines.append('// --- WYN Resource Costs (existing/new/cost per resource) — MMM_M_M scenario ---')
lines.append('// Source: overprocure_results.json costs_detail.resource_costs')
lines.append(f'// Array of {len(THRESHOLDS)} objects per ISO (one per threshold in THRESHOLDS order)')
lines.append('const WYN_RESOURCE_COSTS = {')
for iso_idx, iso in enumerate(ISOS):
    lines.append(f'    {iso}: [')
    for ti, entry in enumerate(wyn_resource_costs[iso]):
        parts = []
        for res in WYN_RESOURCES:
            rd = entry[res]
            if 'dispatch_pct' in rd:
                parts.append(f'{res}:{{d:{rd["dispatch_pct"]},c:{rd["cost"]}}}')
            else:
                parts.append(f'{res}:{{e:{rd["existing_pct"]},n:{rd["new_pct"]},c:{rd["cost"]}}}')
        comma = ',' if ti < len(wyn_resource_costs[iso]) - 1 else ''
        lines.append(f'        {{{", ".join(parts)}}}{comma}')
    iso_comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    ]{iso_comma}')
lines.append('};')

# ============================================================================
# WRITE OUTPUT
# ============================================================================

js_content = '\n'.join(lines) + '\n'
output_path = 'dashboard/js/shared-data.js'
with open(output_path, 'w') as f:
    f.write(js_content)

file_size = os.path.getsize(output_path)
print(f"\nWrote {output_path}")
print(f"  Lines: {len(lines)}")
print(f"  Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

# ============================================================================
# VERIFICATION
# ============================================================================

print("\nVerification:")
print(f"  Thresholds: {len(THRESHOLDS_NUM)} ({THRESHOLDS_NUM[0]}% - {THRESHOLDS_NUM[-1]}%)")

for iso in ISOS:
    rm = resource_mix_data[iso]
    for i in range(len(THRESHOLDS)):
        total = sum(rm[r][i] for r in RESOURCES)
        if total != 100:
            print(f"  WARNING: {iso} threshold {THRESHOLDS[i]} resource mix sums to {total}")

    # Check effective cost monotonicity
    ec = effective_cost_data[iso]
    for i in range(len(ec) - 1):
        if ec[i] is not None and ec[i+1] is not None and ec[i] > ec[i+1]:
            print(f"  WARNING: {iso} effective cost NOT monotonic at {THRESHOLDS[i]}: {ec[i]} > {ec[i+1]}")

    # Check compressed_day data exists for all thresholds
    cd = compressed_day_data[iso]
    zero_count = sum(1 for d in cd['demand'] if all(v == 0 for v in d))
    if zero_count > 0:
        print(f"  WARNING: {iso} has {zero_count} empty compressed_day entries")

print("\nDone.")
