#!/usr/bin/env python3
"""
Extract resource mix and compressed day data from overprocure_results.json
into shared-data.js format.

Adds new constants WITHOUT modifying existing ones.
Uses ISO-aware medium scenario key (8-dim format) — matches EFFECTIVE_COST_DATA convention.
"""

import json
import os

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO']
THRESHOLDS = ['75', '80', '85', '87.5', '90', '92.5', '95', '97.5', '99']
RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro']
MATCHED_RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro', 'battery', 'ldes']


def medium_key(iso):
    """Return the all-Medium scenario key for an ISO."""
    geo = 'M' if iso == 'CAISO' else 'X'
    return f'MMM_M_M_M1_{geo}'


def get_scenario(iso_data, threshold, iso):
    """Get medium scenario with backward compat fallback."""
    scenarios = iso_data.get('thresholds', {}).get(threshold, {}).get('scenarios', {})
    return scenarios.get(medium_key(iso)) or scenarios.get('MMM_M_M')

# Load results
with open('dashboard/overprocure_results.json') as f:
    data = json.load(f)

# ============================================================================
# RESOURCE_MIX_DATA
# ============================================================================
# Structure: { ISO: { resource: [9 values], ..., battery: [...], ldes: [...], procurement: [...] } }
# Indices match THRESHOLDS array

resource_mix_data = {}
for iso in ISOS:
    iso_data = {}
    for res in RESOURCES:
        iso_data[res] = []
    iso_data['battery'] = []
    iso_data['ldes'] = []
    iso_data['procurement'] = []

    for t in THRESHOLDS:
        sc = get_scenario(data['results'][iso], t, iso)
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

# ============================================================================
# COMPRESSED_DAY_DATA
# ============================================================================
# Structure: { ISO: { demand: [9×24], matched: { resource: [9×24] }, gap: [9×24],
#              battery_charge: [9×24], ldes_charge: [9×24] } }

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
        sc = get_scenario(data['results'][iso], t, iso)
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

# ============================================================================
# FORMAT AS JAVASCRIPT
# ============================================================================

def fmt_array(arr, indent=8):
    """Format a flat array of numbers on one line."""
    return '[' + ', '.join(str(v) for v in arr) + ']'

def fmt_24h_array(arr):
    """Format 24-hour array compactly."""
    return '[' + ','.join(str(v) for v in arr) + ']'

lines = []

# --- RESOURCE_MIX_DATA ---
lines.append('')
lines.append('// --- Resource Mix (% of demand) — MMM_M_M scenario ---')
lines.append('// Source: overprocure_results.json (Step 2 repriced)')
lines.append('// Indices match THRESHOLDS array: [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99]')
lines.append('// battery/ldes = dispatch % of demand; procurement = over-procurement %')
lines.append('const RESOURCE_MIX_DATA = {')
for iso in ISOS:
    d = resource_mix_data[iso]
    lines.append(f'    {iso}: {{')
    for key in RESOURCES + ['battery', 'ldes', 'procurement']:
        comma = ',' if key != 'procurement' else ''
        padding = ' ' * max(0, 12 - len(key))
        lines.append(f'        {key}:{padding}{fmt_array(d[key])}{comma}')
    comma = ',' if iso != ISOS[-1] else ''
    lines.append(f'    }}{comma}')
lines.append('};')

# --- COMPRESSED_DAY_DATA ---
lines.append('')
lines.append('// --- Compressed Day Hourly Profiles (24h normalized) — MMM_M_M scenario ---')
lines.append('// Source: overprocure_results.json compressed_day field')
lines.append('// Each sub-array is [24 hourly values] in UTC, one per threshold (matching THRESHOLDS)')
lines.append('// Values are normalized fractions of annual demand (sum across 24h ≈ daily share of annual)')
lines.append('// To convert to MW: value × annual_demand_mwh × 365')
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

# ============================================================================
# CF_TRANCHE_DATA — uprate vs new-build split for WYN panel
# ============================================================================
# Structure: { ISO: { uprate_twh: [9], newbuild_twh: [9], uprate_price: [9],
#              newbuild_price: [9], effective_cf_lcoe: [9], new_cf_twh: [9] } }

cf_tranche_data = {}
for iso in ISOS:
    iso_tr = {
        'new_cf_twh': [],
        'uprate_twh': [],
        'newbuild_twh': [],
        'uprate_price': [],
        'newbuild_price': [],
        'effective_cf_lcoe': [],
    }
    for t in THRESHOLDS:
        sc = get_scenario(data['results'][iso], t, iso)
        tc = sc.get('tranche_costs', {}) if sc else {}
        iso_tr['new_cf_twh'].append(round(tc.get('new_cf_twh', 0), 3))
        iso_tr['uprate_twh'].append(round(tc.get('uprate_twh', 0), 4))
        iso_tr['newbuild_twh'].append(round(tc.get('newbuild_twh', 0), 3))
        iso_tr['uprate_price'].append(tc.get('uprate_price', 0))
        iso_tr['newbuild_price'].append(round(tc.get('newbuild_price', 0), 1))
        iso_tr['effective_cf_lcoe'].append(round(tc.get('effective_new_cf_lcoe', 0), 1))
    cf_tranche_data[iso] = iso_tr

# ============================================================================
# WYN_RESOURCE_COSTS — per-resource existing/new/cost for WYN panel
# ============================================================================
# Structure: { ISO: [ {resource: {existing_pct, new_pct, cost_per_demand_mwh}} × 9 thresholds ] }

wyn_resource_costs = {}
WYN_RESOURCES = ['clean_firm', 'solar', 'wind', 'ccs_ccgt', 'hydro', 'battery', 'ldes']
for iso in ISOS:
    iso_wyn = []
    for t in THRESHOLDS:
        sc = get_scenario(data['results'][iso], t, iso)
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
# FORMAT CF_TRANCHE_DATA
# ============================================================================

lines.append('')
lines.append('// --- CF Tranche Split (uprate vs new-build) — MMM_M_M scenario ---')
lines.append('// Source: overprocure_results.json tranche_costs field')
lines.append('// Indices match THRESHOLDS array: [75, 80, 85, 87.5, 90, 92.5, 95, 97.5, 99]')
lines.append('// uprate_twh capped at 5% of existing nuclear; newbuild = remainder')
lines.append('// Prices: uprate = nuclear uprate LCOE; newbuild = geothermal (CAISO) or SMR + tx')
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

# ============================================================================
# FORMAT WYN_RESOURCE_COSTS
# ============================================================================

lines.append('')
lines.append('// --- WYN Resource Costs (existing/new/cost per resource) — MMM_M_M scenario ---')
lines.append('// Source: overprocure_results.json costs_detail.resource_costs')
lines.append('// Array of 9 objects per ISO (one per threshold in THRESHOLDS order)')
lines.append('// Generation resources: {existing_pct, new_pct, cost} = % of demand + $/MWh-demand')
lines.append('// Storage resources: {dispatch_pct, cost} = dispatch % of demand + $/MWh-demand')
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

js_block = '\n'.join(lines)

# Write to file
output_path = 'dashboard/js/shared-data-new-block.js'
with open(output_path, 'w') as f:
    f.write(js_block)

print(f"Generated {output_path}")
print(f"  Lines: {len(lines)}")
print(f"  Size: {len(js_block):,} bytes ({len(js_block)/1024:.1f} KB)")

# Verification
print("\nVerification:")
for iso in ISOS:
    rm = resource_mix_data[iso]
    # Check resource mix sums to 100 per threshold
    for i in range(9):
        total = sum(rm[r][i] for r in RESOURCES)
        if total != 100:
            print(f"  WARNING: {iso} threshold {THRESHOLDS[i]} resource mix sums to {total} (not 100)")

    # Check compressed day demand sums
    for i in range(9):
        d_sum = sum(compressed_day_data[iso]['demand'][i])
        m_sum = sum(sum(compressed_day_data[iso]['matched'][r][i]) for r in MATCHED_RESOURCES)
        g_sum = sum(compressed_day_data[iso]['gap'][i])
        if abs(d_sum - (m_sum + g_sum)) > 0.01:
            print(f"  WARNING: {iso} threshold {THRESHOLDS[i]} demand={d_sum:.4f} != matched+gap={m_sum+g_sum:.4f}")

print("Done.")
