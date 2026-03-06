#!/usr/bin/env python3
"""
Step 11: IPP SMARTargets Application
=====================================
Maps Step 10 regional market simulation results onto individual IPP fleets.
For each company × scenario × year, computes per-plant economics, determines
retirement order, and identifies the Qualified Target (QT) breakeven.

Inputs:  data/step10-smartargets/ (R1/R2/AT1-4/QT1-4 parquets)
Outputs: dashboard/js/ipp-smartargets-data.js
"""

import json
import math
import os
import numpy as np
import pandas as pd

# ─── PATHS ───────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
STEP10_DIR = os.path.join(ROOT, 'data', 'step10-smartargets')
OUT_JS = os.path.join(ROOT, 'dashboard', 'js', 'ipp-smartargets-data.js')

ISOS = ['CAISO', 'ERCOT', 'PJM', 'NYISO', 'NEISO', 'MISO', 'SPP']
SIM_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
BASE_SCENARIOS = ['R1', 'R2', 'AT1', 'AT2', 'AT3', 'AT4']
QT_SCENARIOS = ['QT1', 'QT2', 'QT3', 'QT4']

# ─── FIXED O&M ($/kW-yr) — NREL ATB 2024 ────────────────────────
FIXED_OM = {
    'coal': 42,
    'gas_ccgt': 13,
    'gas_peaker': 7,
    'nuclear': 120,
    'solar': 18,
    'wind': 28,
    'battery': 20,
    'hydro': 15,
    'geothermal': 30,
    'oil': 35,
}

# ─── ELCC (fraction of nameplate for capacity revenue) ───────────
ELCC = {
    'coal': 0.85,
    'gas_ccgt': 0.90,
    'gas_peaker': 0.95,
    'nuclear': 0.95,
    'solar': 0.20,
    'wind': 0.15,
    'battery': 0.70,
    'hydro': 0.50,
    'geothermal': 0.90,
    'oil': 0.85,
}

# ─── CAPACITY MARKET PRICES ($/kW-yr) — from pipeline_config ────
CAP_PRICES = {
    'CAISO': 75, 'ERCOT': 0, 'PJM': 120,
    'NYISO': 85, 'NEISO': 55, 'MISO': 22, 'SPP': 0,
}

# Capacity price degrades as clean % rises
CAP_DEGRADE_ALPHA = 0.4  # 40% reduction at 100% clean

# ─── 45U NUCLEAR PTC (through 2032) ─────────────────────────────
NUCLEAR_45U_MWH = 15  # $/MWh for existing nuclear, expires 2032

# ─── DEFAULT HEAT RATES (BTU/kWh) for plants without explicit data
DEFAULT_HEAT_RATES = {
    'coal': 10500,
    'gas_ccgt': 6800,
    'gas_peaker': 9800,
    'oil': 10200,
}

# ─── TYPICAL CAPACITY FACTORS (clean resources, stable) ──────────
CLEAN_CF = {
    'nuclear': 0.90,
    'solar': {'CAISO': 0.27, 'ERCOT': 0.25, 'PJM': 0.20, 'NYISO': 0.18, 'NEISO': 0.17, 'MISO': 0.21, 'SPP': 0.23},
    'wind': {'CAISO': 0.28, 'ERCOT': 0.38, 'PJM': 0.30, 'NYISO': 0.28, 'NEISO': 0.32, 'MISO': 0.40, 'SPP': 0.42},
    'battery': 0.10,
    'hydro': 0.38,
    'geothermal': 0.85,
}

# ─── FLEET DATA — extracted from fleet-survival-data.js ──────────
COMPANIES = [
    {
        'id': 'vistra', 'name': 'Vistra Energy', 'shortName': 'Vistra',
        'co2_2024_mt': 68, 'intensity_kg': 485, 'gen_twh': 154, 'cap_gw': 44,
        'target': 'Net-zero by 2050',
        'plants': [
            {'name': 'Martin Lake Coal (TX)', 'iso': 'ERCOT', 'fuel': 'coal', 'cap_mw': 2250, 'gen_twh': 11.0, 'co2_mt': 11.6, 'hr': 10200, 'retire_by': 2027},
            {'name': 'Coleto Creek Coal (TX)', 'iso': 'ERCOT', 'fuel': 'coal', 'cap_mw': 630, 'gen_twh': 3.1, 'co2_mt': 3.3, 'hr': 10400, 'retire_by': 2027},
            {'name': 'Odessa-Ector CCGT', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 1054, 'gen_twh': 5.5, 'co2_mt': 2.2, 'hr': 6900},
            {'name': 'Midlothian CCGT', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 1600, 'gen_twh': 8.4, 'co2_mt': 3.3, 'hr': 6700},
            {'name': 'Forney CCGT', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 1800, 'gen_twh': 9.4, 'co2_mt': 3.8, 'hr': 6800},
            {'name': 'ERCOT CT/Peaker Fleet', 'iso': 'ERCOT', 'fuel': 'gas_peaker', 'cap_mw': 4600, 'gen_twh': 4.0, 'co2_mt': 2.6, 'hr': 9800},
            {'name': 'Comanche Peak Nuclear', 'iso': 'ERCOT', 'fuel': 'nuclear', 'cap_mw': 2400, 'gen_twh': 20.5, 'co2_mt': 0},
            {'name': 'ERCOT Solar Portfolio', 'iso': 'ERCOT', 'fuel': 'solar', 'cap_mw': 358, 'gen_twh': 0.8, 'co2_mt': 0},
            {'name': 'DeCordova BESS', 'iso': 'ERCOT', 'fuel': 'battery', 'cap_mw': 260, 'gen_twh': 0, 'co2_mt': 0},
            {'name': 'Perry Nuclear (OH)', 'iso': 'PJM', 'fuel': 'nuclear', 'cap_mw': 1268, 'gen_twh': 10.7, 'co2_mt': 0},
            {'name': 'Davis-Besse Nuclear (OH)', 'iso': 'PJM', 'fuel': 'nuclear', 'cap_mw': 908, 'gen_twh': 7.6, 'co2_mt': 0},
            {'name': 'Beaver Valley Nuclear (PA)', 'iso': 'PJM', 'fuel': 'nuclear', 'cap_mw': 1872, 'gen_twh': 15.8, 'co2_mt': 0},
            {'name': 'Kincaid Coal (IL)', 'iso': 'PJM', 'fuel': 'coal', 'cap_mw': 1319, 'gen_twh': 4.0, 'co2_mt': 4.0, 'hr': 10800, 'retire_by': 2027},
            {'name': 'Baldwin Coal (IL)', 'iso': 'PJM', 'fuel': 'coal', 'cap_mw': 1185, 'gen_twh': 3.6, 'co2_mt': 3.6, 'hr': 10600, 'retire_by': 2027},
            {'name': 'Newton Coal (IL)', 'iso': 'PJM', 'fuel': 'coal', 'cap_mw': 615, 'gen_twh': 1.9, 'co2_mt': 1.9, 'hr': 10700, 'retire_by': 2027},
            {'name': 'PJM Gas CCGT Fleet', 'iso': 'PJM', 'fuel': 'gas_ccgt', 'cap_mw': 5200, 'gen_twh': 27.3, 'co2_mt': 10.9, 'hr': 7100},
            {'name': 'PJM CT/Peaker Fleet', 'iso': 'PJM', 'fuel': 'gas_peaker', 'cap_mw': 5000, 'gen_twh': 4.4, 'co2_mt': 2.8, 'hr': 9600},
            {'name': 'Moss Landing CCGT (CA)', 'iso': 'CAISO', 'fuel': 'gas_ccgt', 'cap_mw': 1200, 'gen_twh': 6.3, 'co2_mt': 2.5, 'hr': 6900},
            {'name': 'Newington CCGT (NH)', 'iso': 'NEISO', 'fuel': 'gas_ccgt', 'cap_mw': 624, 'gen_twh': 3.3, 'co2_mt': 1.3, 'hr': 7000},
            {'name': 'Bridgeport CCGT (CT)', 'iso': 'NEISO', 'fuel': 'gas_ccgt', 'cap_mw': 558, 'gen_twh': 2.9, 'co2_mt': 1.2, 'hr': 7100},
            {'name': 'Tiverton/Rumford CCGT', 'iso': 'NEISO', 'fuel': 'gas_ccgt', 'cap_mw': 568, 'gen_twh': 3.0, 'co2_mt': 1.2, 'hr': 7000},
        ]
    },
    {
        'id': 'constellation', 'name': 'Constellation Energy', 'shortName': 'Constellation',
        'co2_2024_mt': 55, 'intensity_kg': 177, 'gen_twh': 310, 'cap_gw': 55,
        'target': '100% carbon-free by 2040',
        'plants': [
            {'name': 'Limerick Nuclear (PA)', 'iso': 'PJM', 'fuel': 'nuclear', 'cap_mw': 2317, 'gen_twh': 19.5, 'co2_mt': 0},
            {'name': 'Peach Bottom Nuclear (PA, 50%)', 'iso': 'PJM', 'fuel': 'nuclear', 'cap_mw': 1385, 'gen_twh': 11.7, 'co2_mt': 0},
            {'name': 'Calvert Cliffs Nuclear (MD)', 'iso': 'PJM', 'fuel': 'nuclear', 'cap_mw': 1790, 'gen_twh': 15.1, 'co2_mt': 0},
            {'name': 'Crane Clean Energy Center (PA)', 'iso': 'PJM', 'fuel': 'nuclear', 'cap_mw': 835, 'gen_twh': 7.0, 'co2_mt': 0},
            {'name': 'Salem Nuclear (NJ, 43%)', 'iso': 'PJM', 'fuel': 'nuclear', 'cap_mw': 987, 'gen_twh': 8.3, 'co2_mt': 0},
            {'name': 'PJM Gas Peaker Fleet', 'iso': 'PJM', 'fuel': 'gas_peaker', 'cap_mw': 786, 'gen_twh': 0.7, 'co2_mt': 0.4, 'hr': 9500},
            {'name': 'Conowingo Hydro (MD)', 'iso': 'PJM', 'fuel': 'hydro', 'cap_mw': 572, 'gen_twh': 1.9, 'co2_mt': 0},
            {'name': 'Nine Mile Point Nuclear (NY)', 'iso': 'NYISO', 'fuel': 'nuclear', 'cap_mw': 1907, 'gen_twh': 16.1, 'co2_mt': 0},
            {'name': 'FitzPatrick Nuclear (NY)', 'iso': 'NYISO', 'fuel': 'nuclear', 'cap_mw': 842, 'gen_twh': 7.1, 'co2_mt': 0},
            {'name': 'Ginna Nuclear (NY)', 'iso': 'NYISO', 'fuel': 'nuclear', 'cap_mw': 576, 'gen_twh': 4.9, 'co2_mt': 0},
            {'name': 'NYISO Calpine Gas Fleet', 'iso': 'NYISO', 'fuel': 'gas_peaker', 'cap_mw': 225, 'gen_twh': 0.5, 'co2_mt': 0.2, 'hr': 9200},
            {'name': 'Fore River CCGT (MA)', 'iso': 'NEISO', 'fuel': 'gas_ccgt', 'cap_mw': 750, 'gen_twh': 3.9, 'co2_mt': 1.5, 'hr': 6800},
            {'name': 'Granite Ridge CCGT (NH)', 'iso': 'NEISO', 'fuel': 'gas_ccgt', 'cap_mw': 745, 'gen_twh': 3.9, 'co2_mt': 1.5, 'hr': 6900},
            {'name': 'Westbrook CCGT (ME)', 'iso': 'NEISO', 'fuel': 'gas_ccgt', 'cap_mw': 552, 'gen_twh': 2.9, 'co2_mt': 1.1, 'hr': 7000},
            {'name': 'South Texas Project (44%)', 'iso': 'ERCOT', 'fuel': 'nuclear', 'cap_mw': 1164, 'gen_twh': 9.8, 'co2_mt': 0},
            {'name': 'Deer Park Energy Center', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 1204, 'gen_twh': 6.3, 'co2_mt': 2.5, 'hr': 6700},
            {'name': 'Guadalupe Energy Center', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 1009, 'gen_twh': 5.3, 'co2_mt': 2.1, 'hr': 6800},
            {'name': 'Baytown/Channel/Freestone CCGT', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 2483, 'gen_twh': 13.0, 'co2_mt': 5.2, 'hr': 7000},
            {'name': 'Bosque/Magic Valley/Pasadena CCGT', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 2083, 'gen_twh': 10.9, 'co2_mt': 4.4, 'hr': 7100},
            {'name': 'Quail Run/Corpus/Lost Pines CCGT', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 1570, 'gen_twh': 8.2, 'co2_mt': 3.3, 'hr': 7200},
            {'name': 'ERCOT Calpine Peaker Fleet', 'iso': 'ERCOT', 'fuel': 'gas_peaker', 'cap_mw': 1500, 'gen_twh': 1.3, 'co2_mt': 0.8, 'hr': 9600},
            {'name': 'Colorado Bend CCGT (TX)', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 550, 'gen_twh': 2.9, 'co2_mt': 1.2, 'hr': 7000},
            {'name': 'Delta/Los Medanos CCGT', 'iso': 'CAISO', 'fuel': 'gas_ccgt', 'cap_mw': 1643, 'gen_twh': 8.6, 'co2_mt': 3.4, 'hr': 6800},
            {'name': 'Russell City/Gateway CCGT', 'iso': 'CAISO', 'fuel': 'gas_ccgt', 'cap_mw': 1239, 'gen_twh': 6.5, 'co2_mt': 2.6, 'hr': 6900},
            {'name': 'Pastoria/Metcalf/Sutter CCGT', 'iso': 'CAISO', 'fuel': 'gas_ccgt', 'cap_mw': 1754, 'gen_twh': 9.2, 'co2_mt': 3.7, 'hr': 7000},
            {'name': 'CAISO Calpine Peaker Fleet', 'iso': 'CAISO', 'fuel': 'gas_peaker', 'cap_mw': 1318, 'gen_twh': 1.2, 'co2_mt': 0.7, 'hr': 9800},
            {'name': 'The Geysers Geothermal (CA)', 'iso': 'CAISO', 'fuel': 'geothermal', 'cap_mw': 725, 'gen_twh': 5.3, 'co2_mt': 0.3},
            {'name': 'Nova BESS Portfolio (CA)', 'iso': 'CAISO', 'fuel': 'battery', 'cap_mw': 798, 'gen_twh': 0, 'co2_mt': 0},
            {'name': 'Wind Portfolio (10 states)', 'iso': 'PJM', 'fuel': 'wind', 'cap_mw': 1400, 'gen_twh': 4.6, 'co2_mt': 0},
            {'name': 'Solar Portfolio', 'iso': 'CAISO', 'fuel': 'solar', 'cap_mw': 360, 'gen_twh': 0.8, 'co2_mt': 0},
        ]
    },
    {
        'id': 'nrg', 'name': 'NRG Energy', 'shortName': 'NRG',
        'co2_2024_mt': 42, 'intensity_kg': 520, 'gen_twh': 80, 'cap_gw': 25,
        'target': 'Net-zero by 2050',
        'plants': [
            {'name': 'W.A. Parish Coal (TX)', 'iso': 'ERCOT', 'fuel': 'coal', 'cap_mw': 2697, 'gen_twh': 13.0, 'co2_mt': 13.7, 'hr': 10500},
            {'name': 'Limestone Coal (TX)', 'iso': 'ERCOT', 'fuel': 'coal', 'cap_mw': 1629, 'gen_twh': 7.8, 'co2_mt': 8.2, 'hr': 10400},
            {'name': 'W.A. Parish Gas CCGT', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 1086, 'gen_twh': 5.7, 'co2_mt': 2.2, 'hr': 7000},
            {'name': 'Cedar Bayou CCGT (TX)', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 1492, 'gen_twh': 7.8, 'co2_mt': 3.1, 'hr': 6900},
            {'name': 'LS Power TX Gas Plants', 'iso': 'ERCOT', 'fuel': 'gas_ccgt', 'cap_mw': 2060, 'gen_twh': 10.8, 'co2_mt': 4.3, 'hr': 6800},
            {'name': 'ERCOT CT/Peaker Fleet', 'iso': 'ERCOT', 'fuel': 'gas_peaker', 'cap_mw': 3489, 'gen_twh': 3.1, 'co2_mt': 2.0, 'hr': 9800},
            {'name': 'Doswell Energy Center (VA)', 'iso': 'PJM', 'fuel': 'gas_ccgt', 'cap_mw': 1313, 'gen_twh': 6.9, 'co2_mt': 2.8, 'hr': 6800},
            {'name': 'Hunterstown CCGT (PA)', 'iso': 'PJM', 'fuel': 'gas_ccgt', 'cap_mw': 810, 'gen_twh': 4.2, 'co2_mt': 1.7, 'hr': 7000},
            {'name': 'Ironwood CCGT (PA)', 'iso': 'PJM', 'fuel': 'gas_ccgt', 'cap_mw': 778, 'gen_twh': 4.1, 'co2_mt': 1.6, 'hr': 7100},
            {'name': 'Springdale/Other PJM CCGT', 'iso': 'PJM', 'fuel': 'gas_ccgt', 'cap_mw': 2623, 'gen_twh': 13.8, 'co2_mt': 5.5, 'hr': 7200},
            {'name': 'PJM CT/Peaker Fleet', 'iso': 'PJM', 'fuel': 'gas_peaker', 'cap_mw': 2000, 'gen_twh': 1.8, 'co2_mt': 1.1, 'hr': 9600},
            {'name': 'Ravenswood Gas/Oil (NY)', 'iso': 'NYISO', 'fuel': 'gas_ccgt', 'cap_mw': 1947, 'gen_twh': 8.5, 'co2_mt': 3.5, 'hr': 7800},
            {'name': 'Arthur Kill Gas (NY)', 'iso': 'NYISO', 'fuel': 'gas_peaker', 'cap_mw': 866, 'gen_twh': 0.8, 'co2_mt': 0.5, 'hr': 10200},
            {'name': 'NE Gas Plants', 'iso': 'NEISO', 'fuel': 'gas_ccgt', 'cap_mw': 940, 'gen_twh': 4.9, 'co2_mt': 1.9, 'hr': 7000},
        ]
    },
    {
        'id': 'talen', 'name': 'Talen Energy', 'shortName': 'Talen',
        'co2_2024_mt': 14, 'intensity_kg': 340, 'gen_twh': 42, 'cap_gw': 10.7,
        'target': 'No formal net-zero target',
        'plants': [
            {'name': 'Susquehanna Nuclear (PA, 90%)', 'iso': 'PJM', 'fuel': 'nuclear', 'cap_mw': 2228, 'gen_twh': 18.8, 'co2_mt': 0},
            {'name': 'Lower Mt. Bethel CCGT (PA)', 'iso': 'PJM', 'fuel': 'gas_ccgt', 'cap_mw': 610, 'gen_twh': 3.2, 'co2_mt': 1.3, 'hr': 6700},
            {'name': 'Martins Creek CT (PA)', 'iso': 'PJM', 'fuel': 'gas_peaker', 'cap_mw': 1710, 'gen_twh': 1.5, 'co2_mt': 1.0, 'hr': 10000},
            {'name': 'Camden CT (NJ)', 'iso': 'PJM', 'fuel': 'gas_peaker', 'cap_mw': 145, 'gen_twh': 0.1, 'co2_mt': 0.1, 'hr': 9800},
            {'name': 'Brandon Shores Coal (MD, RMR)', 'iso': 'PJM', 'fuel': 'coal', 'cap_mw': 1295, 'gen_twh': 3.9, 'co2_mt': 3.9, 'hr': 10600, 'retire_by': 2029},
            {'name': 'Brunner Island (PA)', 'iso': 'PJM', 'fuel': 'coal', 'cap_mw': 1419, 'gen_twh': 4.3, 'co2_mt': 4.3, 'hr': 10500, 'retire_by': 2028},
            {'name': 'Montour (PA, gas conversion)', 'iso': 'PJM', 'fuel': 'gas_ccgt', 'cap_mw': 1508, 'gen_twh': 7.9, 'co2_mt': 3.2, 'hr': 7100},
            {'name': 'Conemaugh (PA, 22%)', 'iso': 'PJM', 'fuel': 'coal', 'cap_mw': 392, 'gen_twh': 1.2, 'co2_mt': 1.2, 'hr': 10700},
            {'name': 'Keystone (PA, 12%)', 'iso': 'PJM', 'fuel': 'coal', 'cap_mw': 214, 'gen_twh': 0.6, 'co2_mt': 0.6, 'hr': 10800},
            {'name': 'H.A. Wagner (MD, oil/gas)', 'iso': 'PJM', 'fuel': 'gas_peaker', 'cap_mw': 838, 'gen_twh': 1.0, 'co2_mt': 0.7, 'hr': 10200, 'retire_by': 2029},
        ]
    },
]

FOSSIL_FUELS = {'coal', 'gas_ccgt', 'gas_peaker', 'oil'}
CLEAN_FUELS = {'nuclear', 'solar', 'wind', 'battery', 'hydro', 'geothermal'}


# ─── PLANT-LEVEL ECONOMICS ───────────────────────────────────────

def get_clean_cf(fuel, iso):
    """Return capacity factor for clean resources."""
    val = CLEAN_CF.get(fuel)
    if isinstance(val, dict):
        return val.get(iso, 0.25)
    return val if val else 0.10


def compute_fossil_cf(plant, iso_clean_pct, year):
    """
    Compute a fossil plant's capacity factor under market conditions.

    Uses merit-order logic: plants with lower heat rates dispatch first.
    As clean_pct rises, fossil demand shrinks and the marginal heat rate
    drops — expensive plants lose dispatch first.

    The "marginal heat rate" curve models the ISO-wide merit order:
    - At 0% clean: all fossil runs, marginal HR ~11,500
    - At 50% clean: marginal HR ~9,000 (coal + peakers retired)
    - At 80% clean: marginal HR ~7,500 (only efficient CCGTs left)
    - At 95% clean: marginal HR ~6,500 (best CCGTs only)
    """
    # Check forced retirement
    retire_by = plant.get('retire_by')
    if retire_by and year >= retire_by:
        return 0.0

    fossil_frac = max(0, 100 - iso_clean_pct) / 100.0
    if fossil_frac <= 0.01:
        return 0.0

    # Marginal heat rate: drops as fossil fraction shrinks
    # Calibrated: at 100% fossil → 11500, at 10% fossil → 6500
    hr_min = 6200  # best modern CCGT
    hr_max = 11500  # worst coal
    marginal_hr = hr_min + (hr_max - hr_min) * (fossil_frac ** 0.6)

    plant_hr = plant.get('hr', DEFAULT_HEAT_RATES.get(plant['fuel'], 7000))

    if plant_hr > marginal_hr + 500:
        # Well above marginal — retired / zero dispatch
        return 0.0
    elif plant_hr > marginal_hr:
        # Near marginal — reduced dispatch (linear ramp to zero over 500 BTU range)
        frac = 1.0 - (plant_hr - marginal_hr) / 500.0
        base_cf = plant['gen_twh'] / (plant['cap_mw'] * 8.760 / 1000) if plant['cap_mw'] > 0 else 0.3
        return max(0.02, base_cf * frac * fossil_frac / 0.5)  # scale down with fossil share
    else:
        # Below marginal — dispatches, but at reduced CF as total fossil demand shrinks
        base_cf = plant['gen_twh'] / (plant['cap_mw'] * 8.760 / 1000) if plant['cap_mw'] > 0 else 0.5
        # CF scales with fossil fraction but doesn't collapse for efficient plants
        scale = min(1.0, fossil_frac / 0.3)  # at 30%+ fossil, full CF
        return max(0.05, base_cf * scale)


def compute_plant_economics(plant, iso_data_row, year):
    """
    Compute a single plant's annual revenue, cost, profit, and emissions.

    Args:
        plant: dict with name, iso, fuel, cap_mw, gen_twh, co2_mt, hr, retire_by
        iso_data_row: dict with avg_lmp, clean_pct, demand_twh from Step 10
        year: simulation year

    Returns: dict with status, cf, gen_twh, co2_mt, revenue_m, cost_m, profit_m, details
    """
    fuel = plant['fuel']
    cap_mw = plant['cap_mw']
    iso = plant['iso']
    clean_pct = iso_data_row.get('clean_pct', 50)
    avg_lmp = iso_data_row.get('avg_lmp', 30)

    # Forced retirement check
    retire_by = plant.get('retire_by')
    if retire_by and year >= retire_by:
        return {
            'status': 'retired', 'reason': f'Scheduled retirement by {retire_by}',
            'cf': 0, 'gen_twh': 0, 'co2_mt': 0,
            'revenue_m': 0, 'cost_m': 0, 'profit_m': 0,
        }

    # Compute capacity factor
    if fuel in FOSSIL_FUELS:
        cf = compute_fossil_cf(plant, clean_pct, year)
    else:
        cf = get_clean_cf(fuel, iso)

    # Generation (TWh)
    gen_twh = cap_mw * cf * 8.760 / 1000.0
    gen_mwh = gen_twh * 1e6

    # CO2 emissions (scale from baseline proportionally to CF change)
    base_cf = plant['gen_twh'] / (cap_mw * 8.760 / 1000) if cap_mw > 0 and plant['gen_twh'] > 0 else cf
    if base_cf > 0 and plant['co2_mt'] > 0:
        co2_mt = plant['co2_mt'] * (cf / base_cf)
    else:
        co2_mt = 0.0

    # ── Revenue ──
    # 1. Energy revenue
    energy_rev_m = gen_mwh * avg_lmp / 1e6

    # 2. Capacity revenue (degrades with clean %)
    cap_price_kw_yr = CAP_PRICES.get(iso, 0) * max(0, 1 - CAP_DEGRADE_ALPHA * clean_pct / 100)
    elcc = ELCC.get(fuel, 0.5)
    capacity_rev_m = cap_mw * elcc * cap_price_kw_yr / 1e3  # $M

    # 3. Nuclear 45U PTC (through 2032)
    ptc_rev_m = 0
    if fuel == 'nuclear' and year <= 2032:
        ptc_rev_m = gen_mwh * NUCLEAR_45U_MWH / 1e6

    total_rev_m = energy_rev_m + capacity_rev_m + ptc_rev_m

    # ── Cost ──
    fom_kw_yr = FIXED_OM.get(fuel, 20)
    fixed_om_m = cap_mw * fom_kw_yr / 1e3  # $M

    # ── Profit ──
    profit_m = total_rev_m - fixed_om_m

    # ── Status ──
    if cf <= 0.01:
        status = 'retired'
        reason = 'Uneconomic dispatch (heat rate above marginal)'
    elif profit_m < -5:  # significantly unprofitable ($5M+ loss)
        status = 'stranded'
        reason = f'Operating at loss: ${profit_m:.0f}M/yr'
    elif profit_m < 0:
        status = 'marginal'
        reason = f'Marginally unprofitable: ${profit_m:.0f}M/yr'
    else:
        status = 'operating'
        reason = f'Profitable: ${profit_m:.0f}M/yr'

    return {
        'status': status, 'reason': reason,
        'cf': round(cf, 3),
        'gen_twh': round(gen_twh, 2),
        'co2_mt': round(co2_mt, 2),
        'revenue_m': round(total_rev_m, 1),
        'cost_m': round(fixed_om_m, 1),
        'profit_m': round(profit_m, 1),
        'energy_rev_m': round(energy_rev_m, 1),
        'capacity_rev_m': round(capacity_rev_m, 1),
    }


# ─── LOAD STEP 10 DATA ──────────────────────────────────────────

def load_step10():
    """Load all Step 10 parquets into a nested dict."""
    data = {'base': {}, 'qt': {}}
    for name in BASE_SCENARIOS:
        path = os.path.join(STEP10_DIR, f'smartargets_{name}.parquet')
        if os.path.exists(path):
            data['base'][name] = pd.read_parquet(path)
    for name in QT_SCENARIOS:
        path = os.path.join(STEP10_DIR, f'smartargets_{name}.parquet')
        if os.path.exists(path):
            data['qt'][name] = pd.read_parquet(path)
    return data


def get_iso_conditions(step10_data, scenario, iso, year, reduction_target=None):
    """Extract market conditions for a specific ISO/year/scenario from Step 10."""
    if scenario in step10_data['base']:
        df = step10_data['base'][scenario]
        row = df[(df.iso == iso) & (df.year == year)]
    elif scenario in step10_data['qt']:
        df = step10_data['qt'][scenario]
        if reduction_target is not None:
            row = df[(df.iso == iso) & (df.year == year) &
                     (df.reduction_target == reduction_target)]
        else:
            row = df[(df.iso == iso) & (df.year == year)]
    else:
        return None

    if row.empty:
        return None

    r = row.iloc[0]
    return {
        'clean_pct': float(r.get('clean_pct', 50)),
        'avg_lmp': float(r.get('avg_lmp', 30)),
        'demand_twh': float(r.get('demand_twh', 200)),
        'emissions_mt': float(r.get('emissions_mt', 50)),
    }


# ─── SIMULATE COMPANY ───────────────────────────────────────────

def simulate_company_scenario(company, step10_data, scenario, reduction_target=None):
    """
    Simulate one company under one scenario across all years.
    Returns fleet-level time series + per-plant detail.
    """
    results = {
        'years': SIM_YEARS,
        'fleet_emissions_mt': [],
        'fleet_revenue_m': [],
        'fleet_cost_m': [],
        'fleet_profit_m': [],
        'operating_mw': [],
        'stranded_mw': [],
        'retired_mw': [],
        'by_iso': {},
        'plant_detail': {},  # plant_name → {years, status, cf, profit, co2}
    }

    for plant in company['plants']:
        results['plant_detail'][plant['name']] = {
            'iso': plant['iso'], 'fuel': plant['fuel'], 'cap_mw': plant['cap_mw'],
            'status': [], 'cf': [], 'profit_m': [], 'co2_mt': [],
            'gen_twh': [], 'reason': [],
        }

    for year in SIM_YEARS:
        fleet_em = 0
        fleet_rev = 0
        fleet_cost = 0
        fleet_profit = 0
        op_mw = 0
        strand_mw = 0
        ret_mw = 0

        for plant in company['plants']:
            iso = plant['iso']
            cond = get_iso_conditions(step10_data, scenario, iso, year, reduction_target)
            if cond is None:
                # Fallback: use default conditions
                cond = {'clean_pct': 50, 'avg_lmp': 30, 'demand_twh': 200, 'emissions_mt': 50}

            econ = compute_plant_economics(plant, cond, year)
            pd_entry = results['plant_detail'][plant['name']]
            pd_entry['status'].append(econ['status'])
            pd_entry['cf'].append(econ['cf'])
            pd_entry['profit_m'].append(econ['profit_m'])
            pd_entry['co2_mt'].append(econ['co2_mt'])
            pd_entry['gen_twh'].append(econ['gen_twh'])
            pd_entry['reason'].append(econ['reason'])

            fleet_em += econ['co2_mt']
            fleet_rev += econ['revenue_m']
            fleet_cost += econ['cost_m']
            fleet_profit += econ['profit_m']

            if econ['status'] == 'operating':
                op_mw += plant['cap_mw']
            elif econ['status'] in ('retired',):
                ret_mw += plant['cap_mw']
            else:
                strand_mw += plant['cap_mw']

        results['fleet_emissions_mt'].append(round(fleet_em, 2))
        results['fleet_revenue_m'].append(round(fleet_rev, 1))
        results['fleet_cost_m'].append(round(fleet_cost, 1))
        results['fleet_profit_m'].append(round(fleet_profit, 1))
        results['operating_mw'].append(round(op_mw))
        results['stranded_mw'].append(round(strand_mw))
        results['retired_mw'].append(round(ret_mw))

    return results


def find_qt_breakeven(company, step10_data, qt_scenario):
    """
    Sweep QT reduction targets to find the breakeven QT.
    The QT = the strictest reduction target where fleet profit at 2050 is ≥ 0.
    Also returns the full emissions fan data.
    """
    df = step10_data['qt'].get(qt_scenario)
    if df is None:
        return None

    targets = sorted(df['reduction_target'].unique())
    fan_emissions = {}  # target_str → [emissions per year]
    fan_profit = {}
    breakeven_qt = 0.0

    for t in targets:
        tkey = f'{t:.2f}'
        res = simulate_company_scenario(company, step10_data, qt_scenario, reduction_target=t)
        fan_emissions[tkey] = res['fleet_emissions_mt']
        fan_profit[tkey] = res['fleet_profit_m']

        # Check if fleet is viable at 2050 (last year profit ≥ 0)
        if res['fleet_profit_m'][-1] >= 0:
            breakeven_qt = max(breakeven_qt, t)

    # Compute fan bands across reduction targets per year
    fan_bands = {'p5': [], 'p25': [], 'p50': [], 'p75': [], 'p95': [], 'min': [], 'max': []}
    for yi in range(len(SIM_YEARS)):
        vals = [fan_emissions[f'{t:.2f}'][yi] for t in targets if f'{t:.2f}' in fan_emissions]
        if vals:
            arr = np.array(vals)
            fan_bands['p5'].append(round(float(np.percentile(arr, 5)), 2))
            fan_bands['p25'].append(round(float(np.percentile(arr, 25)), 2))
            fan_bands['p50'].append(round(float(np.percentile(arr, 50)), 2))
            fan_bands['p75'].append(round(float(np.percentile(arr, 75)), 2))
            fan_bands['p95'].append(round(float(np.percentile(arr, 95)), 2))
            fan_bands['min'].append(round(float(arr.min()), 2))
            fan_bands['max'].append(round(float(arr.max()), 2))

    return {
        'reduction_targets': [round(t, 2) for t in targets],
        'breakeven_qt': round(breakeven_qt, 2),
        'fan': fan_bands,
        'emissions_by_target': fan_emissions,
        'profit_by_target': fan_profit,
    }


def compute_at_qt_gap(company, step10_data):
    """Compute the AT-QT gap for each AT scenario."""
    # Company's 2024 CO2 baseline
    baseline_mt = company['co2_2024_mt']
    # AT trajectory: % remaining at each milestone
    at_trajectory = {2030: 0.43, 2035: 0.18, 2040: 0.12, 2045: 0.06, 2050: 0.0}

    gaps = {}
    for qt_name in QT_SCENARIOS:
        qt_result = find_qt_breakeven(company, step10_data, qt_name)
        if not qt_result:
            continue
        qt_pct = qt_result['breakeven_qt']

        # AT emission targets at each milestone (% reduction from baseline)
        # QT achievable reduction
        gap_data = {}
        for yi, year in enumerate(SIM_YEARS):
            at_remaining = at_trajectory.get(year, 1.0)
            at_target_mt = baseline_mt * at_remaining
            # QT achievable: use median emissions at breakeven QT
            median_em = qt_result['fan']['p50'][yi] if yi < len(qt_result['fan']['p50']) else baseline_mt
            gap_data[str(year)] = {
                'at_target_mt': round(at_target_mt, 2),
                'qt_achievable_mt': round(median_em, 2),
                'gap_mt': round(max(0, median_em - at_target_mt), 2),
            }

        gaps[qt_name] = {
            'breakeven_qt_pct': round(qt_pct * 100, 0),
            'milestones': gap_data,
        }

    return gaps


# ─── MAIN ────────────────────────────────────────────────────────

def main():
    print('Step 11: IPP SMARTargets Application')
    print('=' * 50)

    step10_data = load_step10()
    print(f'Loaded Step 10: {len(step10_data["base"])} base + {len(step10_data["qt"])} QT scenarios')

    output = {'companies': {}}

    for company in COMPANIES:
        cid = company['id']
        cname = company['name']
        print(f'\n  Processing {cname}...')

        co_output = {
            'name': cname,
            'shortName': company['shortName'],
            'co2_2024_mt': company['co2_2024_mt'],
            'intensity_kg': company['intensity_kg'],
            'gen_twh': company['gen_twh'],
            'cap_gw': company['cap_gw'],
            'target': company['target'],
            'fleet_summary': {},
            'scenarios': {},
            'qt_analysis': {},
            'at_qt_gap': {},
        }

        # Fleet summary by ISO and fuel
        iso_summary = {}
        for plant in company['plants']:
            iso = plant['iso']
            fuel = plant['fuel']
            if iso not in iso_summary:
                iso_summary[iso] = {}
            if fuel not in iso_summary[iso]:
                iso_summary[iso][fuel] = {'cap_mw': 0, 'gen_twh': 0, 'co2_mt': 0}
            iso_summary[iso][fuel]['cap_mw'] += plant['cap_mw']
            iso_summary[iso][fuel]['gen_twh'] += plant['gen_twh']
            iso_summary[iso][fuel]['co2_mt'] += plant['co2_mt']
        co_output['fleet_summary'] = iso_summary

        # Base scenarios (R1/R2/AT1-4)
        for scn in BASE_SCENARIOS:
            if scn not in step10_data['base']:
                continue
            print(f'    {scn}...', end=' ', flush=True)
            res = simulate_company_scenario(company, step10_data, scn)
            # Trim plant_detail for JS size: only keep key fields
            trimmed_plants = {}
            for pname, pdata in res['plant_detail'].items():
                trimmed_plants[pname] = {
                    'iso': pdata['iso'], 'fuel': pdata['fuel'], 'cap_mw': pdata['cap_mw'],
                    'status': pdata['status'], 'cf': pdata['cf'],
                    'profit_m': pdata['profit_m'], 'co2_mt': pdata['co2_mt'],
                    'gen_twh': pdata['gen_twh'], 'reason': pdata['reason'],
                }
            co_output['scenarios'][scn] = {
                'years': res['years'],
                'fleet_emissions_mt': res['fleet_emissions_mt'],
                'fleet_revenue_m': res['fleet_revenue_m'],
                'fleet_cost_m': res['fleet_cost_m'],
                'fleet_profit_m': res['fleet_profit_m'],
                'operating_mw': res['operating_mw'],
                'stranded_mw': res['stranded_mw'],
                'retired_mw': res['retired_mw'],
                'plants': trimmed_plants,
            }
            print(f'em={res["fleet_emissions_mt"][-1]:.1f}Mt, profit=${res["fleet_profit_m"][-1]:.0f}M')

        # QT analysis (parametric sweep + breakeven QT)
        for qt_name in QT_SCENARIOS:
            if qt_name not in step10_data['qt']:
                continue
            print(f'    {qt_name} sweep...', end=' ', flush=True)
            qt_res = find_qt_breakeven(company, step10_data, qt_name)
            if qt_res:
                co_output['qt_analysis'][qt_name] = {
                    'breakeven_qt': qt_res['breakeven_qt'],
                    'reduction_targets': qt_res['reduction_targets'],
                    'fan': qt_res['fan'],
                    # Only include select targets to keep JS file size manageable
                    'emissions_at_targets': {
                        k: v for k, v in qt_res['emissions_by_target'].items()
                        if float(k) in (0.05, 0.25, 0.50, 0.75, 0.95)
                    },
                }
                print(f'QT={qt_res["breakeven_qt"]:.0%}')
            else:
                print('no data')

        # AT-QT gap
        print(f'    Computing AT-QT gap...')
        co_output['at_qt_gap'] = compute_at_qt_gap(company, step10_data)

        output['companies'][cid] = co_output

    # Write JS file
    json_str = json.dumps(output, separators=(',', ':'))
    js_content = f'// Auto-generated by step11_ipp_smartargets.py — do not edit manually\nconst IPP_SMARTARGETS = {json_str};\n'

    os.makedirs(os.path.dirname(OUT_JS), exist_ok=True)
    with open(OUT_JS, 'w') as f:
        f.write(js_content)

    size_kb = os.path.getsize(OUT_JS) / 1024
    print(f'\n  Written {OUT_JS} ({size_kb:.0f} KB)')
    print('Done.')


if __name__ == '__main__':
    main()
