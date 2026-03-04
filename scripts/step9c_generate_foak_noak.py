#!/usr/bin/env python3
"""
Step 9c — Generate FOAK→NOAK Learning Curve Data for Clean Firm Case Page
==========================================================================
Computes three JS constants appended to dashboard/js/shared-data.js:

1. FOAK_NOAK_YEARS: [2025, 2026, ..., 2050]
2. CLEAN_FIRM_LEARNING: {central, conservative, optimistic} — $/MWh trajectories
3. GAS_TRUE_COST: {base, with_carbon} — gas cost trajectories

Uses the same Wright's Law engine as step3/step8 for consistency.
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pipeline_config import (
    NUCLEAR_NEWBUILD_LCOE, CCS_LCOE_45Q_ON, CCS_LCOE_45Q_OFF,
    FOAK_NUCLEAR_NEWBUILD, FOAK_CCS_45Q_ON, FOAK_CCS_45Q_OFF,
    FOAK_GEOTHERMAL,
    WHOLESALE_PRICES, FUEL_ADJUSTMENTS,
    REGIONAL_DEMAND_TWH,
)

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']

# ═══════════════════════════════════════════════════════════════════════════════
# WRIGHT'S LAW ENGINE (same as step8d)
# ═══════════════════════════════════════════════════════════════════════════════

# Learning curve timing: (foak_start_year, noak_year) per scenario
LEARNING_WINDOWS = {
    'optimistic':   (2028, 2036),   # Low — fast learning
    'central':      (2030, 2040),   # Medium — baseline
    'conservative': (2036, 2048),   # High — slow learning
}
LEARNING_EXPONENT = 0.6

def learning_fraction_vec(years, foak_start, noak_year):
    """Wright's Law S-curve: fraction of FOAK→NOAK transition completed."""
    duration = noak_year - foak_start
    if duration <= 0:
        return np.ones_like(years, dtype=np.float64)
    active = np.clip((years - foak_start) / duration, 0.0, 1.0)
    return np.power(active, LEARNING_EXPONENT)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE DEMAND-WEIGHTED CLEAN FIRM COSTS
# ═══════════════════════════════════════════════════════════════════════════════

years = np.arange(2025, 2051, dtype=np.float64)

# Compute demand-weighted average FOAK and NOAK costs across ISOs
# FOAK = High LCOE (or explicit FOAK tables), NOAK = Low LCOE
demands = np.array([REGIONAL_DEMAND_TWH[iso] for iso in ISOS])
total_demand = demands.sum()
weights = demands / total_demand

# Nuclear: FOAK from FOAK_NUCLEAR_NEWBUILD, NOAK from Low LCOE
nuc_foak = np.array([FOAK_NUCLEAR_NEWBUILD[iso] for iso in ISOS])
nuc_noak = np.array([NUCLEAR_NEWBUILD_LCOE['L'][iso] for iso in ISOS])
nuc_medium = np.array([NUCLEAR_NEWBUILD_LCOE['M'][iso] for iso in ISOS])

# CCS (with 45Q): FOAK from FOAK_CCS_45Q_ON, NOAK from Low LCOE
ccs_foak = np.array([FOAK_CCS_45Q_ON[iso] for iso in ISOS])
ccs_noak = np.array([CCS_LCOE_45Q_ON['L'][iso] for iso in ISOS])
ccs_medium = np.array([CCS_LCOE_45Q_ON['M'][iso] for iso in ISOS])

# Blend nuclear + CCS at 50/50 for "clean firm" composite
# (In practice, the mix varies by ISO, but 50/50 is a reasonable representation)
blend_foak = (nuc_foak + ccs_foak) / 2.0
blend_noak = (nuc_noak + ccs_noak) / 2.0
blend_medium = (nuc_medium + ccs_medium) / 2.0

# Demand-weighted national averages
avg_foak = float(np.dot(weights, blend_foak))
avg_noak = float(np.dot(weights, blend_noak))
avg_medium = float(np.dot(weights, blend_medium))

print(f"Clean firm FOAK (demand-wtd avg): ${avg_foak:.1f}/MWh")
print(f"Clean firm NOAK (demand-wtd avg): ${avg_noak:.1f}/MWh")
print(f"Clean firm Medium 2025:           ${avg_medium:.1f}/MWh")

# Generate trajectories for each learning scenario
clean_firm_learning = {}
for scenario, (foak_start, noak_year) in LEARNING_WINDOWS.items():
    fracs = learning_fraction_vec(years, foak_start, noak_year)
    # Trajectory: starts at FOAK, declines to NOAK over learning window
    trajectory = avg_foak - fracs * (avg_foak - avg_noak)
    clean_firm_learning[scenario] = [round(float(v), 1) for v in trajectory]

# ═══════════════════════════════════════════════════════════════════════════════
# GAS TRUE COST TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════════

# Base gas cost: demand-weighted wholesale + typical CCGT heat rate markup
# CCGT operates at ~$45-55/MWh all-in (fuel + O&M + capacity cost amortized)
gas_base_prices = np.array([WHOLESALE_PRICES[iso] for iso in ISOS], dtype=np.float64)
# Add ~$15/MWh for heat rate premium + O&M above wholesale
gas_all_in = gas_base_prices + 15.0
avg_gas_base = float(np.dot(weights, gas_all_in))

# Gas fuel escalation: ~2% real annual increase (EIA AEO reference case)
gas_escalation = 0.02
gas_base_trajectory = avg_gas_base * np.power(1.0 + gas_escalation, years - 2025)

# Carbon price trajectory: EPA SCC starts ~$51/ton in 2025, grows ~2.5% real/yr
# CO2 intensity of natural gas CCGT: ~0.37 tCO2/MWh
scc_2025 = 51.0
scc_growth = 0.025
co2_intensity = 0.37  # tCO2/MWh for CCGT

carbon_adder = scc_2025 * np.power(1.0 + scc_growth, years - 2025) * co2_intensity

gas_true_cost = {
    'base': [round(float(v), 1) for v in gas_base_trajectory],
    'with_carbon': [round(float(v), 1) for v in gas_base_trajectory + carbon_adder],
}

print(f"Gas base 2025: ${avg_gas_base:.1f}/MWh, 2050: ${gas_base_trajectory[-1]:.1f}/MWh")
print(f"Gas + carbon 2025: ${(avg_gas_base + scc_2025 * co2_intensity):.1f}/MWh, "
      f"2050: ${(gas_base_trajectory[-1] + carbon_adder[-1]):.1f}/MWh")

# ═══════════════════════════════════════════════════════════════════════════════
# WRITE TO shared-data.js
# ═══════════════════════════════════════════════════════════════════════════════

SHARED_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'dashboard', 'js', 'shared-data.js')

years_list = [int(y) for y in years]

lines = []
lines.append('')
lines.append('// ============================================================================')
lines.append('// FOAK → NOAK LEARNING CURVES (Step 9c)')
lines.append('// Clean firm = 50/50 nuclear + CCS-CCGT (45Q ON), demand-weighted across 7 ISOs')
lines.append(f'// FOAK: ${avg_foak:.0f}/MWh → NOAK: ${avg_noak:.0f}/MWh')
lines.append('// Gas: wholesale + $15 heat-rate/O&M premium, 2% real escalation')
lines.append('// Carbon: EPA SCC $51/t (2025), 2.5% real growth, 0.37 tCO2/MWh CCGT')
lines.append('// ============================================================================')
lines.append(f'const FOAK_NOAK_YEARS = {json.dumps(years_list)};')
lines.append('')
lines.append('const CLEAN_FIRM_LEARNING = {')
for scenario in ['central', 'conservative', 'optimistic']:
    vals = ', '.join(str(v) for v in clean_firm_learning[scenario])
    lines.append(f'    {scenario}: [{vals}],')
lines.append('};')
lines.append('')
lines.append('const GAS_TRUE_COST = {')
for key in ['base', 'with_carbon']:
    vals = ', '.join(str(v) for v in gas_true_cost[key])
    lines.append(f'    {key}: [{vals}],')
lines.append('};')
lines.append('')

block = '\n'.join(lines)

# Check if already present, replace if so
with open(SHARED_DATA_PATH, 'r') as f:
    content = f.read()

marker_start = '// FOAK → NOAK LEARNING CURVES (Step 9c)'
marker_end = '};'  # last closing brace of GAS_TRUE_COST

if marker_start in content:
    # Find and replace existing block
    idx_start = content.index(marker_start)
    # Go back to find the preceding separator line
    line_start = content.rfind('\n', 0, idx_start)
    # Find the end of the GAS_TRUE_COST block
    search_from = content.index('const GAS_TRUE_COST', idx_start)
    idx_end = content.index('};', search_from) + 2
    # Check if there's a trailing newline
    if idx_end < len(content) and content[idx_end] == '\n':
        idx_end += 1
    content = content[:line_start] + block + content[idx_end:]
    print(f"\nReplaced existing FOAK-NOAK block in {SHARED_DATA_PATH}")
else:
    content += block
    print(f"\nAppended FOAK-NOAK block to {SHARED_DATA_PATH}")

with open(SHARED_DATA_PATH, 'w') as f:
    f.write(content)

print("Done.")
