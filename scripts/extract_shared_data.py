#!/usr/bin/env python3
"""
Extract resource mix and compressed day data from co2 results
into shared-data.js format.

Adds new constants WITHOUT modifying existing ones.
Uses ISO-aware medium scenario key (9-dim format).

Reads from: data/step5-post-processing/co2_results/ (parquets preferred, JSON fallback)
Outputs to: dashboard/js/shared-data-new-block.js
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CO2_DIR = os.path.join(ROOT_DIR, 'data', 'step5-post-processing', 'co2_results')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'dashboard', 'js', 'shared-data-new-block.js')

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
THRESHOLDS = ['50', '55', '60', '65', '70', '75', '80', '85', '87.5', '90', '92.5', '95', '97.5', '99', '99.5', '99.9', '99.99']
RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro']
MATCHED_RESOURCES = ['clean_firm', 'solar', 'wind', 'offshore_wind', 'ccs_ccgt', 'hydro', 'battery', 'battery8', 'ldes', 'h2']


def medium_key(iso):
    """Return the all-Medium scenario key for a given ISO (9-dim format)."""
    return 'MMM_M_M_M_M1_M' if iso == 'CAISO' else 'MMM_M_M_M_M1_X'


def get_scenario(scenarios, iso):
    """Get scenario data using 9-dim key with fallback to old formats."""
    key = medium_key(iso)
    if key in scenarios:
        return scenarios[key]
    for fallback in ['MMM_M_M_M_M1_M', 'MMM_M_M_M_M1_X', 'MMM_M_M']:
        if fallback in scenarios:
            return scenarios[fallback]
    return None


_PARQUET_CACHE = {}


def _load_iso_parquet(iso):
    """Load an ISO's CO2 parquet into the cache (lazy, one-time)."""
    if iso in _PARQUET_CACHE:
        return
    _PARQUET_CACHE[iso] = {}
    for prefix in ['co2_', 'step4_', 'step3_co_']:
        ppath = os.path.join(CO2_DIR, f'{prefix}{iso}.parquet')
        if os.path.exists(ppath):
            try:
                sys.path.insert(0, ROOT_DIR)
                from parquet_io import load_from_parquets
                data = load_from_parquets(CO2_DIR, [iso])
                iso_data = data.get('results', {}).get(iso, {})
                for t_str, t_data in iso_data.get('thresholds', {}).items():
                    _PARQUET_CACHE[iso][t_str] = t_data.get('scenarios', {})
                return
            except Exception:
                pass


def load_scenarios(iso, threshold):
    """Load scenarios from CO2 parquets or batch JSON files."""
    _load_iso_parquet(iso)
    if iso in _PARQUET_CACHE and threshold in _PARQUET_CACHE[iso]:
        return _PARQUET_CACHE[iso][threshold]

    # Fall back to legacy JSON
    fname = f"{iso}_{threshold}_2025.json"
    fpath = os.path.join(CO2_DIR, fname)
    if not os.path.exists(fpath):
        return {}
    with open(fpath) as f:
        data = json.load(f)
    return data.get('scenarios', {})


# ============================================================================
# RESOURCE_MIX_DATA
# ============================================================================
# Structure: { ISO: { resource: [15 values], ..., battery: [...], ldes: [...], h2: [...] } }
# Indices match THRESHOLDS array

resource_mix_data = {}
for iso in ISOS:
    iso_data = {}
    for res in RESOURCES:
        iso_data[res] = []
    iso_data['battery'] = []
    iso_data['battery8'] = []
    iso_data['ldes'] = []
    iso_data['h2'] = []

    for t in THRESHOLDS:
        scenarios = load_scenarios(iso, t)
        sc = get_scenario(scenarios, iso) if scenarios else None
        if sc:
            rm = sc.get('resource_mix', {})
            for res in RESOURCES:
                iso_data[res].append(rm.get(res, 0))
            iso_data['battery'].append(sc.get('battery_dispatch_pct', 0))
            iso_data['battery8'].append(sc.get('battery8_dispatch_pct', 0))
            iso_data['ldes'].append(sc.get('ldes_dispatch_pct', 0))
            iso_data['h2'].append(sc.get('h2_dispatch_pct', 0))
        else:
            for res in RESOURCES:
                iso_data[res].append(0)
            iso_data['battery'].append(0)
            iso_data['battery8'].append(0)
            iso_data['ldes'].append(0)
            iso_data['h2'].append(0)

    resource_mix_data[iso] = iso_data

# ============================================================================
# COMPRESSED_DAY_DATA
# ============================================================================
# Structure: { ISO: { demand: [15×24], matched: { resource: [15×24] }, gap: [15×24],
#              battery_charge: [15×24], battery8_charge: [15×24], ldes_charge: [15×24], h2_charge: [15×24] } }

compressed_day_data = {}
for iso in ISOS:
    iso_cd = {
        'demand': [],
        'matched': {r: [] for r in MATCHED_RESOURCES},
        'gap': [],
        'battery_charge': [],
        'battery8_charge': [],
        'ldes_charge': [],
        'h2_charge': [],
    }

    for t in THRESHOLDS:
        scenarios = load_scenarios(iso, t)
        sc = get_scenario(scenarios, iso) if scenarios else None
        if sc and 'compressed_day' in sc:
            cd = sc['compressed_day']
            iso_cd['demand'].append([round(v, 5) for v in cd['demand']])
            iso_cd['gap'].append([round(v, 5) for v in cd['gap']])
            iso_cd['battery_charge'].append([round(v, 5) for v in cd.get('battery_charge', [0]*24)])
            iso_cd['battery8_charge'].append([round(v, 5) for v in cd.get('battery8_charge', [0]*24)])
            iso_cd['ldes_charge'].append([round(v, 5) for v in cd.get('ldes_charge', [0]*24)])
            iso_cd['h2_charge'].append([round(v, 5) for v in cd.get('h2_charge', [0]*24)])
            for res in MATCHED_RESOURCES:
                vals = cd.get('matched', {}).get(res, [0]*24)
                iso_cd['matched'][res].append([round(v, 5) for v in vals])
        else:
            iso_cd['demand'].append([0]*24)
            iso_cd['gap'].append([0]*24)
            iso_cd['battery_charge'].append([0]*24)
            iso_cd['battery8_charge'].append([0]*24)
            iso_cd['ldes_charge'].append([0]*24)
            iso_cd['h2_charge'].append([0]*24)
            for res in MATCHED_RESOURCES:
                iso_cd['matched'][res].append([0]*24)

    compressed_day_data[iso] = iso_cd

# ============================================================================
# CF_TRANCHE_DATA
# ============================================================================

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
        scenarios = load_scenarios(iso, t)
        sc = get_scenario(scenarios, iso) if scenarios else None
        tc = sc.get('tranche_costs', {}) if sc else {}
        iso_tr['new_cf_twh'].append(round(tc.get('new_cf_twh', 0), 3))
        iso_tr['uprate_twh'].append(round(tc.get('uprate_twh', 0), 4))
        # newbuild = nuclear_newbuild + ccs_tranche + geo
        nb = tc.get('nuclear_newbuild_twh', 0) + tc.get('ccs_tranche_twh', 0) + tc.get('geo_twh', 0)
        iso_tr['newbuild_twh'].append(round(nb, 3))
        iso_tr['uprate_price'].append(tc.get('uprate_price', 0))
        iso_tr['newbuild_price'].append(round(tc.get('newbuild_price', 0), 1))
        eff = tc.get('effective_new_cf_lcoe', 0)
        iso_tr['effective_cf_lcoe'].append(round(eff, 1))
    cf_tranche_data[iso] = iso_tr

# ============================================================================
# FORMAT AS JAVASCRIPT
# ============================================================================

def fmt_array(arr):
    """Format a flat array of numbers on one line."""
    return '[' + ', '.join(str(v) for v in arr) + ']'

def fmt_24h_array(arr):
    """Format 24-hour array compactly."""
    return '[' + ','.join(str(v) for v in arr) + ']'

lines = []

# --- RESOURCE_MIX_DATA ---
lines.append('')
lines.append('// --- Resource Mix (% of demand) — MMM_M_M scenario ---')
lines.append('// Source: co2_results batch JSONs (Step 5)')
lines.append(f'// Indices match THRESHOLDS array: {THRESHOLDS}')
lines.append('// battery/battery8/ldes/h2 = dispatch % of demand')
lines.append('const RESOURCE_MIX_DATA = {')
for iso in ISOS:
    d = resource_mix_data[iso]
    lines.append(f'    {iso}: {{')
    for key in RESOURCES + ['battery', 'battery8', 'ldes', 'h2']:
        comma = ',' if key != 'h2' else ''
        padding = ' ' * max(0, 12 - len(key))
        lines.append(f'        {key}:{padding}{fmt_array(d[key])}{comma}')
    comma = ',' if iso != ISOS[-1] else ''
    lines.append(f'    }}{comma}')
lines.append('};')

# --- COMPRESSED_DAY_DATA ---
lines.append('')
lines.append('// --- Compressed Day Hourly Profiles (24h normalized) — MMM_M_M scenario ---')
lines.append('// Source: co2_results batch JSONs compressed_day field')
lines.append('// Each sub-array is [24 hourly values], one per threshold (matching THRESHOLDS)')
lines.append('const COMPRESSED_DAY_DATA = {')
for iso_idx, iso in enumerate(ISOS):
    cd = compressed_day_data[iso]
    lines.append(f'    {iso}: {{')

    lines.append('        demand: [')
    for i, arr in enumerate(cd['demand']):
        comma = ',' if i < len(cd['demand']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    lines.append('        matched: {')
    for res_idx, res in enumerate(MATCHED_RESOURCES):
        res_comma = ',' if res_idx < len(MATCHED_RESOURCES) - 1 else ''
        lines.append(f'            {res}: [')
        for i, arr in enumerate(cd['matched'][res]):
            comma = ',' if i < len(cd['matched'][res]) - 1 else ''
            lines.append(f'                {fmt_24h_array(arr)}{comma}')
        lines.append(f'            ]{res_comma}')
    lines.append('        },')

    lines.append('        gap: [')
    for i, arr in enumerate(cd['gap']):
        comma = ',' if i < len(cd['gap']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    lines.append('        battery_charge: [')
    for i, arr in enumerate(cd['battery_charge']):
        comma = ',' if i < len(cd['battery_charge']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    lines.append('        battery8_charge: [')
    for i, arr in enumerate(cd['battery8_charge']):
        comma = ',' if i < len(cd['battery8_charge']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    lines.append('        ldes_charge: [')
    for i, arr in enumerate(cd['ldes_charge']):
        comma = ',' if i < len(cd['ldes_charge']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ],')

    lines.append('        h2_charge: [')
    for i, arr in enumerate(cd['h2_charge']):
        comma = ',' if i < len(cd['h2_charge']) - 1 else ''
        lines.append(f'            {fmt_24h_array(arr)}{comma}')
    lines.append('        ]')

    iso_comma = ',' if iso_idx < len(ISOS) - 1 else ''
    lines.append(f'    }}{iso_comma}')
lines.append('};')

# --- CF_TRANCHE_DATA ---
lines.append('')
lines.append('// --- CF Tranche Split (uprate vs new-build) — MMM_M_M scenario ---')
lines.append('// Source: co2_results batch JSONs tranche_costs field')
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

js_block = '\n'.join(lines)

# Write to file
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    f.write(js_block)

print(f"Generated {OUTPUT_PATH}")
print(f"  Lines: {len(lines)}")
print(f"  Size: {len(js_block):,} bytes ({len(js_block)/1024:.1f} KB)")

# Verification
print("\nVerification:")
for iso in ISOS:
    rm = resource_mix_data[iso]
    for i in range(len(THRESHOLDS)):
        total = sum(rm[r][i] for r in RESOURCES)
        if total != 100 and total != 0:
            print(f"  WARNING: {iso} threshold {THRESHOLDS[i]} resource mix sums to {total} (not 100)")

    for i in range(len(THRESHOLDS)):
        d_sum = sum(compressed_day_data[iso]['demand'][i])
        m_sum = sum(sum(compressed_day_data[iso]['matched'][r][i]) for r in MATCHED_RESOURCES)
        g_sum = sum(compressed_day_data[iso]['gap'][i])
        if d_sum > 0 and abs(d_sum - (m_sum + g_sum)) > 0.01:
            print(f"  WARNING: {iso} threshold {THRESHOLDS[i]} demand={d_sum:.4f} != matched+gap={m_sum+g_sum:.4f}")

print("Done.")
