#!/usr/bin/env python3
"""CSV-based parameter templates for the market sweep.

Loads/exports sweep parameter definitions from two CSV files:
  1. sweep_params.csv       — 5 simple axes + demand growth rates (per-ISO)
  2. sweep_price_sens.csv   — price sensitivity bundles (9-dim toggle vectors)

This allows users to modify the L/M/H variable space, add custom levels,
or swap in entirely different parameter templates without editing Python code.
"""

import csv
import os
import sys
import warnings

# ISOs in canonical order
ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# Required axes in sweep_params.csv
REQUIRED_AXES = {'demand_growth', 'gas_friction', 'ppa', 'queue', 'new_fossil_cost'}

# Required columns in sweep_price_sens.csv
PRICE_SENS_COLUMNS = ['name', 'ren', 'firm', 'batt', 'ldes_lvl', 'ccs', 'q45', 'fuel', 'tx', 'geo']

# Queue level → learning speed mapping (extensible)
QUEUE_LEARNING_MAP = {
    'High': 'Fast',
    'Medium': 'Medium',
    'Low': 'Slow',
}


def load_sweep_params(csv_path):
    """Load sweep parameter axes from CSV.

    Returns dict with keys:
        demand_growth_levels: list of level names (e.g., ['Low', 'Medium', 'High'])
        demand_growth_rates:  dict {level: {iso: rate}} for per-ISO numeric rates
        gas_friction_levels:  dict {level_name: numeric_value}
        ppa_levels:           list of level names
        queue_levels:         list of level names
        new_fossil_cost_levels: list of level names

    Raises ValueError on validation failures.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Sweep params CSV not found: {csv_path}")

    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Sweep params CSV is empty: {csv_path}")

    # Group rows by axis
    axes = {}
    for row in rows:
        axis = row.get('axis', '').strip()
        if not axis:
            continue
        axes.setdefault(axis, []).append(row)

    # Validate required axes
    missing = REQUIRED_AXES - set(axes.keys())
    if missing:
        raise ValueError(f"Missing required axes in CSV: {sorted(missing)}")

    result = {}

    # --- demand_growth ---
    dg_rows = axes['demand_growth']
    dg_levels = []
    dg_rates = {}
    for row in dg_rows:
        level = row.get('level', '').strip()
        if not level:
            raise ValueError("demand_growth row has empty level name")
        dg_levels.append(level)
        iso_rates = {}
        for iso in ISOS:
            val = row.get(iso, '').strip()
            if not val:
                raise ValueError(
                    f"demand_growth level '{level}' missing rate for {iso}")
            try:
                iso_rates[iso] = float(val)
            except ValueError:
                raise ValueError(
                    f"demand_growth level '{level}', {iso}: "
                    f"'{val}' is not a valid number")
        dg_rates[level] = iso_rates
    result['demand_growth_levels'] = dg_levels
    result['demand_growth_rates'] = dg_rates

    # --- gas_friction ---
    gf_rows = axes['gas_friction']
    gf_levels = {}
    for row in gf_rows:
        level = row.get('level', '').strip()
        val = row.get('value', '').strip()
        if not level:
            raise ValueError("gas_friction row has empty level name")
        if not val:
            raise ValueError(
                f"gas_friction level '{level}' missing numeric value")
        try:
            gf_levels[level] = float(val)
        except ValueError:
            raise ValueError(
                f"gas_friction level '{level}': '{val}' is not a valid number")
    result['gas_friction_levels'] = gf_levels

    # --- ppa ---
    ppa_rows = axes['ppa']
    result['ppa_levels'] = [r.get('level', '').strip() for r in ppa_rows]
    if not all(result['ppa_levels']):
        raise ValueError("ppa has row(s) with empty level name")

    # --- queue ---
    q_rows = axes['queue']
    result['queue_levels'] = [r.get('level', '').strip() for r in q_rows]
    if not all(result['queue_levels']):
        raise ValueError("queue has row(s) with empty level name")

    # --- new_fossil_cost ---
    nfc_rows = axes['new_fossil_cost']
    result['new_fossil_cost_levels'] = [
        r.get('level', '').strip() for r in nfc_rows]
    if not all(result['new_fossil_cost_levels']):
        raise ValueError("new_fossil_cost has row(s) with empty level name")

    return result


def load_price_sensitivities(csv_path):
    """Load price sensitivity bundles from CSV.

    Returns dict matching PRICE_SENSITIVITIES format:
        {name: {ren, firm, batt, ldes_lvl, ccs, q45, fuel, tx, geo}}

    Raises ValueError on validation failures.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Price sensitivities CSV not found: {csv_path}")

    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Price sensitivities CSV is empty: {csv_path}")

    # Validate columns
    missing_cols = set(PRICE_SENS_COLUMNS) - set(headers)
    if missing_cols:
        raise ValueError(
            f"Price sensitivities CSV missing columns: {sorted(missing_cols)}")

    result = {}
    for row in rows:
        name = row.get('name', '').strip()
        if not name:
            raise ValueError("Price sensitivity row has empty name")
        sens = {}
        for col in PRICE_SENS_COLUMNS:
            if col == 'name':
                continue
            val = row.get(col, '').strip()
            if not val:
                raise ValueError(
                    f"Price sensitivity '{name}' missing value for '{col}'")
            sens[col] = val
        result[name] = sens

    return result


def export_default_params(output_dir):
    """Export current hardcoded defaults to CSV template files.

    Creates:
        {output_dir}/sweep_params_default.csv
        {output_dir}/sweep_price_sensitivities_default.csv
    """
    # Import hardcoded defaults
    sys.path.insert(0, os.path.dirname(__file__))
    from pipeline_config import DEMAND_GROWTH_RATES

    os.makedirs(output_dir, exist_ok=True)

    # --- sweep_params_default.csv ---
    params_path = os.path.join(output_dir, 'sweep_params_default.csv')
    with open(params_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['axis', 'level', 'value'] + ISOS)

        # demand_growth
        for level in ['Low', 'Medium', 'High']:
            rates = DEMAND_GROWTH_RATES.get(ISOS[0], {})  # check structure
            iso_vals = [DEMAND_GROWTH_RATES[iso][level] for iso in ISOS]
            writer.writerow(['demand_growth', level, ''] + iso_vals)

        # gas_friction
        for level, val in [('Low', 0.3), ('Medium', 0.7), ('High', 1.0)]:
            writer.writerow(
                ['gas_friction', level, val] + [''] * len(ISOS))

        # ppa
        for level in ['Low', 'Medium', 'High']:
            writer.writerow(['ppa', level, ''] + [''] * len(ISOS))

        # queue
        for level in ['Low', 'Medium', 'High']:
            writer.writerow(['queue', level, ''] + [''] * len(ISOS))

        # new_fossil_cost
        for level in ['Low', 'Medium', 'High']:
            writer.writerow(['new_fossil_cost', level, ''] + [''] * len(ISOS))

    print(f"Exported: {params_path}")

    # --- sweep_price_sensitivities_default.csv ---
    from market_simulation import PRICE_SENSITIVITIES
    ps_path = os.path.join(output_dir, 'sweep_price_sensitivities_default.csv')
    with open(ps_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(PRICE_SENS_COLUMNS)
        for name, sens in PRICE_SENSITIVITIES.items():
            row = [name] + [sens[col] for col in PRICE_SENS_COLUMNS[1:]]
            writer.writerow(row)

    print(f"Exported: {ps_path}")
    return params_path, ps_path


def compute_scenario_count(params, price_sens):
    """Compute total scenario count from loaded params + price sensitivities."""
    n = (len(params['demand_growth_levels'])
         * len(price_sens)
         * len(params['ppa_levels'])
         * len(params['gas_friction_levels'])
         * len(params['queue_levels'])
         * len(params['new_fossil_cost_levels']))
    return n


def _level_code(level_name):
    """Generate a short code from a level name for scenario IDs.

    Standard levels: Low→L, Medium→M, High→H
    Custom levels: first 2 chars uppercase (e.g., Very_High→VH)
    """
    standard = {'Low': 'L', 'Medium': 'M', 'High': 'H'}
    if level_name in standard:
        return standard[level_name]
    # Custom: take first char of each word, or first 2 chars
    parts = level_name.replace('-', '_').split('_')
    if len(parts) >= 2:
        return ''.join(p[0].upper() for p in parts[:2])
    return level_name[:2].upper()
