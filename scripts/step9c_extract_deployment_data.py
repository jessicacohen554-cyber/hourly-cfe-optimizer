#!/usr/bin/env python3
"""
Step 9C: Extract compact deployment data for the Procurement Deployment visualization.

Reads strategy1_consequential.json, strategy2_hourly.json, strategy3_annual.json
and produces dashboard/js/deployment-data.js with normalized existing/new/cross-ISO splits.

Format differences handled:
- Strategy 1A/1B/1C: resource_mix is {"{src_iso}_{resource}": scalar_twh} (cross-regional, all new-build)
- Strategy 2A: resource_mix is {resource: {twh, price, cost_m}} (no category, all new-build)
- Strategy 2B: resource_mix is {resource: {twh, price, cost_m}} + "existing_clean_free" (no category)
- Strategy 2C: resource_mix is {source: {twh, price, cost_m, category}} (gold standard)
- Strategy 3A: resource_mix is {resource: {twh, price, cost_m}} (no category, all new-build)
- Strategy 3B: resource_mix is {"{src_iso}_{resource}": {twh, price, cost_m}} (cross-regional, all new-build)
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'step5-post-processing')
JS_DIR = os.path.join(ROOT_DIR, 'dashboard', 'js')

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Canonical resource names for normalization
RESOURCE_ALIASES = {
    'firm': 'clean_firm',
    'nuclear_newbuild': 'clean_firm',
    'nuclear_uprate': 'nuclear_uprate',
    'uprate': 'nuclear_uprate',
    'solar': 'solar',
    'wind': 'wind',
    'offshore_wind': 'offshore_wind',
    'ccs_45q_on': 'ccs',
    'ccs_ccgt': 'ccs',
    'ccs': 'ccs',
    'hydro': 'hydro',
    'battery': 'battery',
    'storage': 'storage',
    'ldes': 'ldes',
    'geothermal': 'geothermal',
    'h2': 'green_h2',
    'green_h2': 'green_h2',
    # 2C tranche names
    'existing_vre_hydro': 'existing_vre',
    'new_build_vre': 'new_vre',
    'new_build_solar': 'solar',
    'new_build_wind': 'wind',
    'new_build_firm': 'clean_firm',
    'new_build_storage': 'storage',
}


def normalize_resource(raw_key):
    """Normalize resource name, stripping ISO prefix if present."""
    # Check for {ISO}_{resource} pattern
    for iso in ISOS:
        if raw_key.startswith(f'{iso}_'):
            resource = raw_key[len(iso) + 1:]
            return iso, RESOURCE_ALIASES.get(resource, resource)
    return None, RESOURCE_ALIASES.get(raw_key, raw_key)


def extract_strategy1(data, buyer_iso):
    """Extract from strategy 1A/1B/1C format: {key: scalar_twh}, all new-build, cross-regional."""
    existing = {}
    new = {}
    cross_iso = {}

    for key, val in data.items():
        if key == 'sss_allocation':
            continue
        # Val is scalar TWh for strategy 1
        twh = val if isinstance(val, (int, float)) else val.get('twh', 0)
        if twh <= 0:
            continue

        source_iso, resource = normalize_resource(key)

        if source_iso and source_iso != buyer_iso:
            # Cross-ISO flow
            if source_iso not in cross_iso:
                cross_iso[source_iso] = {}
            cross_iso[source_iso][resource] = cross_iso[source_iso].get(resource, 0) + twh
        else:
            # Same ISO, all new-build for strategy 1
            new[resource] = new.get(resource, 0) + twh

    return existing, new, cross_iso


def extract_strategy2a(data):
    """Strategy 2A: 100% new-build, same-ISO, no category field."""
    new = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        twh = val.get('twh', 0)
        if twh <= 0:
            continue
        _, resource = normalize_resource(key)
        new[resource] = round(new.get(resource, 0) + twh, 3)
    return {}, new, {}


def extract_strategy2b(data):
    """Strategy 2B: existing_clean_free entry is existing, rest is new-build."""
    existing = {}
    new = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        twh = val.get('twh', 0)
        if twh <= 0:
            continue
        _, resource = normalize_resource(key)

        if key == 'existing_clean_free':
            existing['grid_clean'] = round(twh, 3)
        else:
            new[resource] = round(new.get(resource, 0) + twh, 3)
    return existing, new, {}


def extract_strategy2c(data):
    """Strategy 2C: has category field (existing, uprate, new_build, sss)."""
    existing = {}
    new = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        twh = val.get('twh', 0)
        if twh <= 0:
            continue
        category = val.get('category', 'new_build')
        _, resource = normalize_resource(key)

        if category in ('existing', 'sss', 'uprate'):
            existing[resource] = round(existing.get(resource, 0) + twh, 3)
        else:
            new[resource] = round(new.get(resource, 0) + twh, 3)
    return existing, new, {}


def extract_strategy3a(data):
    """Strategy 3A: SSS + 4-pool tranches (same format as 2C with category field)."""
    existing = {}
    new = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        twh = val.get('twh', 0)
        if twh <= 0:
            continue
        category = val.get('category', 'new_build')
        _, resource = normalize_resource(key)

        if category in ('existing', 'sss', 'uprate'):
            existing[resource] = round(existing.get(resource, 0) + twh, 3)
        else:
            new[resource] = round(new.get(resource, 0) + twh, 3)
    return existing, new, {}


def extract_strategy3b(data, buyer_iso):
    """Strategy 3B: cross-regional new-build, {src_iso}_{resource} keys."""
    new = {}
    cross_iso = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        twh = val.get('twh', 0)
        if twh <= 0:
            continue
        source_iso, resource = normalize_resource(key)

        if source_iso and source_iso != buyer_iso:
            if source_iso not in cross_iso:
                cross_iso[source_iso] = {}
            cross_iso[source_iso][resource] = round(cross_iso[source_iso].get(resource, 0) + twh, 3)
        else:
            new[resource] = round(new.get(resource, 0) + twh, 3)
    return {}, new, cross_iso


EXTRACTORS = {
    '1A': lambda rm, iso: extract_strategy1(rm, iso),
    '1B': lambda rm, iso: extract_strategy1(rm, iso),
    '2A': lambda rm, iso: extract_strategy2a(rm),
    '2B': lambda rm, iso: extract_strategy2b(rm),
    '2C': lambda rm, iso: extract_strategy2c(rm),
    '3A': lambda rm, iso: extract_strategy3a(rm),
    '3B': lambda rm, iso: extract_strategy3b(rm, iso),
}

# Map strategy IDs to their JSON files and keys
STRATEGY_SOURCES = {
    '1A': ('strategy1_consequential.json', 'strategy1A'),
    '1B': ('strategy1_consequential.json', 'strategy1B'),
    '2A': ('strategy2_hourly.json', 'strategy2A'),
    '2B': ('strategy2_hourly.json', 'strategy2B'),
    '2C': ('strategy2_hourly.json', 'strategy2C'),
    '3A': ('strategy3_annual.json', 'strategy3A'),
    '3B': ('strategy3_annual.json', 'strategy3B'),
}


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None
    with open(path) as f:
        return json.load(f)


def remove_zero_entries(d, threshold=1e-6):
    """Remove entries with negligible values to save space."""
    return {k: v for k, v in d.items() if isinstance(v, dict) or (isinstance(v, (int, float)) and v > threshold)}


def main():
    print("=" * 70)
    print("Step 9C: Extract Deployment Data for Procurement Deployment Viz")
    print("=" * 70)

    # Load all strategy JSONs
    loaded = {}
    for filename in set(src[0] for src in STRATEGY_SOURCES.values()):
        print(f"\nLoading {filename}...")
        data = load_json(filename)
        if data:
            loaded[filename] = data
            print(f"  OK — keys: {list(data.keys())}")

    if not loaded:
        print("ERROR: No strategy JSON files found. Run step 8 first.")
        sys.exit(1)

    # Build output
    output = {
        'strategies': list(STRATEGY_SOURCES.keys()),
        'isos': ISOS,
        'data': {},
    }

    # Key participation levels for the visualization (subset for compact output)
    KEY_PARTICIPATION = {'2pct', '5pct', '10pct', '20pct', '30pct', '50pct', '80pct'}
    # Key thresholds — skip fine gradations to reduce size
    KEY_THRESHOLDS = {50, 60, 70, 75, 80, 85, 90, 92.5, 95, 97.5, 99, 99.5, 99.9, 99.99}

    total_entries = 0
    skipped = 0

    for strat_id, (filename, json_key) in STRATEGY_SOURCES.items():
        print(f"\nProcessing {strat_id} ({json_key})...")
        if filename not in loaded:
            print(f"  SKIP — {filename} not loaded")
            continue

        strat_data = loaded[filename].get(json_key, {})
        if not strat_data:
            print(f"  SKIP — {json_key} empty in {filename}")
            continue

        output['data'][strat_id] = {}
        extractor = EXTRACTORS[strat_id]

        for iso in ISOS:
            iso_data = strat_data.get(iso, {})
            if not iso_data:
                continue

            output['data'][strat_id][iso] = {}

            # Use Medium growth only for dashboard (matching procurement-strategy-data.js)
            growth_data = iso_data.get('Medium', {})
            if not growth_data:
                # Fallback to first available
                growth_data = next(iter(iso_data.values()), {})

            for part_key, entries in growth_data.items():
                if not isinstance(entries, list):
                    continue

                # Filter to key participation levels for compact output
                if part_key not in KEY_PARTICIPATION:
                    continue

                # part_key is like "2pct", "10pct" etc
                part_pct = part_key.replace('pct', '')

                part_dict = {}
                for entry in entries:
                    rm = entry.get('resource_mix', {})
                    threshold = entry.get('threshold')
                    if threshold is None or not rm:
                        skipped += 1
                        continue

                    # Filter to key thresholds
                    if threshold not in KEY_THRESHOLDS:
                        skipped += 1
                        continue

                    # Normalize threshold key
                    thr_key = str(threshold) if threshold != int(threshold) else str(int(threshold))

                    existing, new, cross_iso = extractor(rm, iso)

                    # Clean up empty dicts
                    existing = remove_zero_entries(existing)
                    new = remove_zero_entries(new)
                    cross_iso = {k: remove_zero_entries(v) for k, v in cross_iso.items() if v}

                    # Round TWh values — 4 decimals to preserve small flows at low participation
                    existing = {k: round(v, 4) for k, v in existing.items()}
                    new = {k: round(v, 4) for k, v in new.items()}
                    cross_iso = {k: {r: round(t, 4) for r, t in v.items()} for k, v in cross_iso.items()}

                    record = {}
                    if existing:
                        record['e'] = existing
                    if new:
                        record['n'] = new
                    if cross_iso:
                        record['x'] = cross_iso
                    record['c'] = round(entry.get('cost_per_mwh', 0), 0)
                    record['tc'] = round(entry.get('total_cost_m', 0), 0)
                    record['co2'] = round(entry.get('co2_abated_mmt', 0), 3)
                    record['bt'] = round(entry.get('buyer_demand_twh', 0), 1)

                    # MAC ($/tCO2)
                    mac = entry.get('mac_per_ton')
                    if mac is not None and mac > 0:
                        record['mac'] = round(mac, 0)

                    # CO2 reduction framing: baseline + reduction %
                    meta = entry.get('metadata', {})
                    if 'co2_reduction_pct' in meta:
                        record['co2r'] = round(meta['co2_reduction_pct'], 1)
                    if 'baseline_co2_mt' in meta:
                        record['bl'] = round(meta['baseline_co2_mt'], 3)

                    part_dict[thr_key] = record
                    total_entries += 1

                if part_dict:
                    output['data'][strat_id][iso][part_pct] = part_dict

        iso_count = len(output['data'][strat_id])
        print(f"  {strat_id}: {iso_count} ISOs processed")

    print(f"\n{'=' * 70}")
    print(f"Total entries: {total_entries}, skipped: {skipped}")

    # Write JS file
    os.makedirs(JS_DIR, exist_ok=True)
    js_path = os.path.join(JS_DIR, 'deployment-data.js')

    with open(js_path, 'w') as f:
        f.write('// Auto-generated by step9c_extract_deployment_data.py\n')
        f.write(f'// Generated: {__import__("datetime").datetime.now().isoformat()}\n')
        f.write(f'// Entries: {total_entries}\n\n')
        f.write('const DEPLOYMENT_DATA = ')
        json.dump(output, f, separators=(',', ':'))
        f.write(';\n')

    size_kb = os.path.getsize(js_path) / 1024
    print(f"\nWrote {js_path}")
    print(f"Size: {size_kb:.0f} KB")
    print("Done.")


if __name__ == '__main__':
    main()
