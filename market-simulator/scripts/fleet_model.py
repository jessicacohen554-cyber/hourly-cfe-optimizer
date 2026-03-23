#!/usr/bin/env python3
"""
Fleet Model — Real Generator-Level Merit-Order Stacks from EIA 860/923 + EPA CAMPD
==================================================================================
Loads actual generator inventory (EIA 860), generation/fuel data (EIA 923), and
emissions monitoring (EPA CAMPD) from CSV/XLSX/JSON files. Cross-references to
compute revealed heat rates, bins generators by efficiency, and builds real
merit-order stacks that can replace the stylized stack in lmp_engine.py.

Data directory layout:
    market-simulator/data/eia-860/{state}/   — EIA Form 860 generator inventory
    market-simulator/data/eia-923/{state}/   — EIA Form 923 generation + fuel
    market-simulator/data/epa-campd/{state}/ — EPA CAMPD hourly emissions

Usage:
    from fleet_model import FleetModel
    fm = FleetModel(state='TX')
    fm.build_fleet()
    stack, total_mw = fm.build_merit_order_stack(fuel_level='Medium')

    # Or run directly:
    python fleet_model.py          # Loads TX, prints fleet summary
    python fleet_model.py --state CA
"""

import json
import logging
import os
import sys
import glob
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup — allow imports from sibling scripts
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DATA_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')

# ---------------------------------------------------------------------------
# Default constants (from lmp_engine.py) — used as fallback when no EIA data
# ---------------------------------------------------------------------------
DEFAULT_HEAT_RATES = {
    'coal_steam': 10.0,   # MMBtu/MWh
    'gas_ccgt':   7.0,
    'gas_ct':     10.5,
    'oil_ct':     10.5,
}

DEFAULT_VOM = {
    'coal_steam': 5.50,   # $/MWh
    'gas_ccgt':   3.50,
    'gas_ct':     5.00,
    'oil_ct':     6.00,
}

DEFAULT_CO2_RATES = {
    'coal_steam': 0.95,   # tons CO2 / MWh
    'gas_ccgt':   0.37,
    'gas_ct':     0.55,
    'oil_ct':     0.65,
}

# Year-varying fuel prices from EIA AEO 2025 projections
# Falls back to static values if projection data unavailable
try:
    from fuel_price_projections import FUEL_PRICES as DEFAULT_FUEL_PRICES
except ImportError:
    DEFAULT_FUEL_PRICES = {
        'Low':    {'coal': 2.00, 'gas': 2.00, 'oil': 8.00},
        'Medium': {'coal': 2.25, 'gas': 3.50, 'oil': 10.50},
        'High':   {'coal': 2.50, 'gas': 6.00, 'oil': 13.00},
    }

# ---------------------------------------------------------------------------
# Balancing Authority → ISO mapping
# ---------------------------------------------------------------------------
BA_TO_ISO = {
    # CAISO
    'CISO': 'CAISO', 'CAISO': 'CAISO',
    # ERCOT
    'ERCO': 'ERCOT', 'ERCOT': 'ERCOT',
    # PJM
    'PJM': 'PJM', 'PJME': 'PJM', 'PJMW': 'PJM', 'PJMD': 'PJM',
    'PJMC': 'PJM', 'AEP': 'PJM', 'AP': 'PJM', 'ATSI': 'PJM',
    'COMED': 'PJM', 'DAY': 'PJM', 'DEOK': 'PJM', 'DOM': 'PJM',
    'DPL': 'PJM', 'DUQ': 'PJM', 'EKPC': 'PJM', 'JC': 'PJM',
    'ME': 'PJM', 'PE': 'PJM', 'PEP': 'PJM', 'PL': 'PJM',
    'PN': 'PJM', 'PS': 'PJM', 'RECO': 'PJM',
    # NYISO
    'NYIS': 'NYISO', 'NYISO': 'NYISO',
    # NEISO / ISO-NE
    'ISNE': 'NEISO', 'NEISO': 'NEISO', 'ISONE': 'NEISO',
    # MISO
    'MISO': 'MISO', 'MECS': 'MISO', 'AMMO': 'MISO', 'ALTE': 'MISO',
    'ALTW': 'MISO', 'CONS': 'MISO', 'CWEP': 'MISO', 'CWLP': 'MISO',
    'DECI': 'MISO', 'EAI': 'MISO', 'EDE': 'MISO', 'EES': 'MISO',
    'EMBA': 'MISO', 'GRE': 'MISO', 'HE': 'MISO', 'LAFA': 'MISO',
    'LEPA': 'MISO', 'MDU': 'MISO', 'MEC': 'MISO', 'MGE': 'MISO',
    'MPS': 'MISO', 'MPW': 'MISO', 'NIPS': 'MISO', 'NSP': 'MISO',
    'OTP': 'MISO', 'SMP': 'MISO', 'SIPC': 'MISO', 'UPPC': 'MISO',
    'WEC': 'MISO', 'WPS': 'MISO', 'CLEC': 'MISO',
    # SPP
    'SWPP': 'SPP', 'SPP': 'SPP', 'AECI': 'SPP', 'CSWS': 'SPP',
    'EDE_SPP': 'SPP', 'GRDA': 'SPP', 'INDN': 'SPP', 'KACY': 'SPP',
    'KCPL': 'SPP', 'LES': 'SPP', 'MIDW': 'SPP', 'NPPD': 'SPP',
    'OKGE': 'SPP', 'OPPD': 'SPP', 'SPA': 'SPP', 'SECI': 'SPP',
    'SPRM': 'SPP', 'SPS': 'SPP', 'WAUE': 'SPP', 'WFEC': 'SPP',
    'WR': 'SPP',
}

# State → primary ISO (rough mapping for single-state loads)
STATE_TO_ISO = {
    'TX': 'ERCOT', 'CA': 'CAISO', 'NY': 'NYISO',
    'PA': 'PJM', 'OH': 'PJM', 'VA': 'PJM', 'NJ': 'PJM',
    'IL': 'PJM', 'MD': 'PJM', 'WV': 'PJM', 'DE': 'PJM',
    'MA': 'NEISO', 'CT': 'NEISO', 'ME': 'NEISO', 'NH': 'NEISO',
    'VT': 'NEISO', 'RI': 'NEISO',
    'MN': 'MISO', 'WI': 'MISO', 'IN': 'MISO', 'MI': 'MISO',
    'IA': 'MISO', 'MO': 'MISO', 'AR': 'MISO', 'LA': 'MISO', 'MS': 'MISO',
    'AL': 'MISO',
    'OK': 'SPP', 'KS': 'SPP', 'NE': 'SPP', 'ND': 'SPP', 'SD': 'SPP',
    'AZ': 'CAISO', 'OR': 'CAISO',  # WECC, mapped to CAISO as nearest ISO
}

# Prime mover → unit type classification
PRIME_MOVER_MAP = {
    # Steam turbines (coal / oil / gas)
    'ST': None,  # resolved by fuel type
    # Combined cycle
    'CA': 'gas_ccgt', 'CS': 'gas_ccgt', 'CT': 'gas_ccgt', 'CC': 'gas_ccgt',
    # Combustion turbine / gas turbine
    'GT': 'gas_ct', 'IC': 'gas_ct', 'OT': 'gas_ct',
    # Gas engine
    'CE': 'gas_ct',
}

# Fuel type codes → fuel category
FUEL_TYPE_MAP = {
    # Coal
    'BIT': 'coal', 'SUB': 'coal', 'LIG': 'coal', 'ANT': 'coal',
    'RC': 'coal', 'WC': 'coal', 'SC': 'coal', 'COL': 'coal',
    # Natural gas
    'NG': 'gas', 'BFG': 'gas', 'OG': 'gas', 'PG': 'gas', 'SG': 'gas',
    'LFG': 'gas', 'GAS': 'gas',
    # Oil / petroleum
    'DFO': 'oil', 'RFO': 'oil', 'JF': 'oil', 'KER': 'oil',
    'PC': 'oil', 'WO': 'oil', 'OIL': 'oil', 'PET': 'oil',
}

# Default heat-rate bin thresholds for gas CCGT
DEFAULT_HR_BINS_CCGT = {
    'gas_ccgt_efficient': (0.0, 7.2),
    'gas_ccgt_avg':       (7.2, 8.0),
    'gas_ccgt_old':       (8.0, 99.0),
}

DEFAULT_HR_BINS_COAL = {
    'coal_steam_efficient': (0.0, 9.5),
    'coal_steam_avg':       (9.5, 10.5),
    'coal_steam_old':       (10.5, 99.0),
}

DEFAULT_HR_BINS_CT = {
    'gas_ct_efficient': (0.0, 10.0),
    'gas_ct_avg':       (10.0, 11.0),
    'gas_ct_old':       (11.0, 99.0),
}


def _fuzzy_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column from a list of candidate names (case-insensitive).

    Handles EIA's inconsistent column naming across vintages — tries exact match
    first, then case-insensitive, then substring containment.

    Args:
        df: DataFrame to search columns in.
        candidates: Ordered list of candidate column names.

    Returns:
        Matched column name as it appears in df, or None if no match.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        # Exact match
        if cand in df.columns:
            return cand
        # Case-insensitive
        cl = cand.lower().strip()
        if cl in cols_lower:
            return cols_lower[cl]
    # Substring containment (last resort)
    for cand in candidates:
        cl = cand.lower().strip()
        for col_low, col_orig in cols_lower.items():
            if cl in col_low or col_low in cl:
                return col_orig
    return None


def _read_data_files(directory: str) -> Optional[pd.DataFrame]:
    """Read all CSV, XLSX, and JSON files from a directory, concatenate into one DataFrame.

    Args:
        directory: Path to scan for data files.

    Returns:
        Concatenated DataFrame, or None if no files found.
    """
    if not os.path.isdir(directory):
        return None

    frames = []
    for pattern in ['*.csv', '*.CSV', '*.xlsx', '*.XLSX', '*.xls', '*.json', '*.JSON']:
        for fpath in glob.glob(os.path.join(directory, pattern)):
            try:
                if fpath.lower().endswith('.json'):
                    # Handle both array-of-objects and wrapped JSON formats
                    import json as _json
                    with open(fpath) as _f:
                        raw_json = _json.load(_f)
                    # EPA CAMPD API returns array of objects
                    if isinstance(raw_json, list):
                        if not raw_json:
                            logger.debug(f"Empty JSON array in {fpath}, skipping")
                            continue
                        df = pd.DataFrame(raw_json)
                    elif isinstance(raw_json, dict):
                        # Check for common wrapper keys (e.g., {"data": [...], "meta": ...})
                        for key in ('data', 'records', 'results', 'rows'):
                            if key in raw_json and isinstance(raw_json[key], list):
                                df = pd.DataFrame(raw_json[key])
                                break
                        else:
                            # Single-record dict or unknown structure
                            df = pd.DataFrame([raw_json])
                    else:
                        logger.warning(f"Unexpected JSON type in {fpath}: {type(raw_json).__name__}")
                        continue
                elif fpath.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(fpath, engine='openpyxl')
                else:
                    # Try common encodings
                    for enc in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            df = pd.read_csv(fpath, encoding=enc, low_memory=False)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        logger.warning(f"Could not decode {fpath}, skipping")
                        continue
                if len(df) > 0:
                    frames.append(df)
            except Exception as e:
                logger.warning(f"Error reading {fpath}: {e}")
                continue

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _classify_unit(prime_mover: str, fuel_code: str) -> Optional[str]:
    """Classify a generator into a unit type based on prime mover and fuel.

    Args:
        prime_mover: EIA prime mover code (e.g., 'CA', 'GT', 'ST').
        fuel_code: EIA fuel type code (e.g., 'NG', 'BIT', 'DFO').

    Returns:
        Unit type string ('coal_steam', 'gas_ccgt', 'gas_ct', 'oil_ct') or None
        if the generator is not a fossil thermal unit.
    """
    pm = str(prime_mover).strip().upper() if pd.notna(prime_mover) else ''
    fc = str(fuel_code).strip().upper() if pd.notna(fuel_code) else ''

    fuel_cat = FUEL_TYPE_MAP.get(fc)
    if fuel_cat is None:
        return None  # Not a fossil fuel generator (nuclear, hydro, wind, solar, etc.)

    # Steam turbine — resolve by fuel
    if pm == 'ST':
        if fuel_cat == 'coal':
            return 'coal_steam'
        elif fuel_cat == 'oil':
            return 'oil_ct'
        elif fuel_cat == 'gas':
            return 'gas_ccgt'  # Gas-fired steam → treat as CCGT for dispatch
        return None

    # Combined cycle types
    if pm in ('CA', 'CS', 'CT', 'CC'):
        if fuel_cat == 'oil':
            return 'oil_ct'
        return 'gas_ccgt'

    # Combustion turbine / internal combustion
    if pm in ('GT', 'IC', 'OT', 'CE'):
        if fuel_cat == 'coal':
            return 'coal_steam'  # Rare, but some fluidized bed
        if fuel_cat == 'oil':
            return 'oil_ct'
        return 'gas_ct'

    return None


class FleetModel:
    """Loads real generator data and builds heat-rate-binned merit-order stacks.

    Attributes:
        state: Two-letter state abbreviation.
        iso: Mapped ISO name (from BA or state default).
        eia860: Raw EIA 860 DataFrame (generator inventory).
        eia923: Raw EIA 923 DataFrame (generation + fuel).
        campd: Raw EPA CAMPD DataFrame (hourly emissions).
        fleet: Processed fleet DataFrame with revealed heat rates and classifications.
        bins: Dict of bin_name → DataFrame of generators in that bin.
        using_fallback: True if real data was not available and stylized defaults are in use.
    """

    def __init__(self, state: str = 'TX', iso: Optional[str] = None,
                 data_root: Optional[str] = None,
                 hr_bins_ccgt: Optional[Dict] = None,
                 hr_bins_coal: Optional[Dict] = None,
                 hr_bins_ct: Optional[Dict] = None,
                 use_quintiles: bool = False):
        """Initialize FleetModel.

        Args:
            state: Two-letter state abbreviation for data directory lookup.
            iso: Override ISO name. If None, inferred from state.
            data_root: Override root data directory. Defaults to market-simulator/data/.
            hr_bins_ccgt: Custom heat-rate bin thresholds for gas CCGT.
                Dict of bin_name → (hr_low, hr_high).
            hr_bins_coal: Custom heat-rate bin thresholds for coal.
            hr_bins_ct: Custom heat-rate bin thresholds for gas CT.
            use_quintiles: If True, ignore fixed bins and compute quintile boundaries
                from the actual fleet data.
        """
        self.state = state.upper()
        self.iso = iso or STATE_TO_ISO.get(self.state, 'ERCOT')
        self.data_root = data_root or DATA_ROOT

        self.hr_bins_ccgt = hr_bins_ccgt or DEFAULT_HR_BINS_CCGT
        self.hr_bins_coal = hr_bins_coal or DEFAULT_HR_BINS_COAL
        self.hr_bins_ct = hr_bins_ct or DEFAULT_HR_BINS_CT
        self.use_quintiles = use_quintiles

        self.eia860: Optional[pd.DataFrame] = None
        self.eia923: Optional[pd.DataFrame] = None
        self.campd: Optional[pd.DataFrame] = None
        self.fleet: Optional[pd.DataFrame] = None
        self.bins: Dict[str, pd.DataFrame] = {}
        self.using_fallback = False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_eia860(self) -> Optional[pd.DataFrame]:
        """Load EIA Form 860 generator inventory data.

        Reads from market-simulator/data/eia-860/{state}/.
        Extracts: Plant ID, Generator ID, Nameplate Capacity, Heat Rate,
        Fuel Type, Prime Mover, Online Year, Planned Retirement, Balancing Authority.

        Returns:
            DataFrame with standardized column names, or None if no data found.
        """
        path = os.path.join(self.data_root, 'eia-860', self.state)
        raw = _read_data_files(path)
        if raw is None:
            logger.debug(f"No EIA 860 data found in {path} — skipping (optional)")
            return None

        # Map columns to standard names
        # Includes both CSV/XLSX names and EIA API JSON field names
        col_map = {
            'plant_id':       ['plantid', 'Plant Id', 'Plant ID', 'Plant Code',
                               'Utility ID', 'plant_id', 'PLANT_ID', 'Plant.ID'],
            'gen_id':         ['generatorid', 'Generator Id', 'Generator ID', 'gen_id',
                               'GENERATOR_ID', 'Generator.ID', 'Genid'],
            'capacity_mw':    ['nameplate-capacity-mw', 'Nameplate Capacity (MW)',
                               'Nameplate Capacity', 'Capacity (MW)', 'capacity_mw',
                               'NAMEPLATE', 'Summer Capacity (MW)',
                               'Net Summer Capacity (MW)'],
            'heat_rate':      ['Heat Rate (MMBtu/MWh)', 'Heat Rate', 'heat_rate',
                               'HEAT_RATE', 'Design Heat Rate'],
            'fuel_type':      ['energy_source_code', 'Energy Source 1', 'Reported Fuel Type Code',
                               'Energy Source Code', 'fuel_type', 'FUEL_TYPE',
                               'Fuel Type', 'Energy Source', 'Primary Fuel'],
            'prime_mover':    ['prime_mover_code', 'Prime Mover Code', 'Prime Mover',
                               'prime_mover', 'PRIME_MOVER', 'Reported Prime Mover'],
            'online_year':    ['operating-year-month', 'Operating Year', 'Online Year',
                               'online_year', 'ONLINE_YEAR', 'Year Online',
                               'Initial Year of Operation'],
            'retirement_year': ['planned-retirement-year-month', 'Planned Retirement Year',
                                'Retirement Year', 'retirement_year', 'RETIREMENT_YEAR',
                                'Expected Retirement Year', 'Planned Retirement Date'],
            'entity_name':    ['entityName', 'Entity Name', 'Utility Name', 'entity_name',
                               'ENTITY_NAME', 'Owner', 'Operator'],
            'plant_name':     ['plantName', 'Plant Name', 'plant_name', 'PLANT_NAME'],
            'latitude':       ['latitude', 'Latitude', 'LATITUDE', 'Lat'],
            'longitude':      ['longitude', 'Longitude', 'LONGITUDE', 'Lon', 'Long'],
            'county':         ['county', 'County', 'COUNTY'],
            'ba':             ['balancing_authority_code', 'Balancing Authority Code',
                               'Balancing Authority', 'ba', 'BA_CODE', 'BA Code',
                               'Balancing Authority Name'],
            'status':         ['Status', 'Operating Status', 'status', 'STATUS',
                               'Generator Status'],
        }

        df = pd.DataFrame()
        for std_name, candidates in col_map.items():
            col = _fuzzy_col(raw, candidates)
            if col is not None:
                df[std_name] = raw[col]
            else:
                df[std_name] = np.nan

        # Coerce types
        for col in ['capacity_mw', 'heat_rate']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Extract year from YYYY-MM strings (e.g., "1980-07" → 1980) or plain numeric
        for col in ['online_year', 'retirement_year']:
            vals = df[col].astype(str).str.strip()
            # If values look like "YYYY-MM", extract the year portion
            yyyy_mm_mask = vals.str.match(r'^\d{4}-\d{1,2}$', na=False)
            if yyyy_mm_mask.any():
                extracted = vals.str.extract(r'^(\d{4})', expand=False)
                df[col] = pd.to_numeric(extracted, errors='coerce').astype('Int64')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        # Map BA to ISO
        df['iso'] = df['ba'].map(lambda x: BA_TO_ISO.get(str(x).strip().upper(), None)
                                 if pd.notna(x) else None)

        self.eia860 = df
        return df

    def load_eia923(self) -> Optional[pd.DataFrame]:
        """Load EIA Form 923 generation and fuel consumption data.

        Reads from market-simulator/data/eia-923/{state}/.
        Extracts: Plant ID, Generator ID, Net Generation (MWh),
        Fuel Consumed (MMBtu), monthly generation columns.

        Returns:
            DataFrame with standardized column names, or None if no data found.
        """
        path = os.path.join(self.data_root, 'eia-923', self.state)
        raw = _read_data_files(path)
        if raw is None:
            logger.debug(f"No EIA 923 data found in {path} — skipping (optional)")
            return None

        col_map = {
            'plant_id':       ['Plant Id', 'Plant ID', 'Plant Code', 'plant_id',
                               'PLANT_ID', 'Plant.ID', 'plantCode'],
            'gen_id':         ['Generator Id', 'Generator ID', 'gen_id',
                               'GENERATOR_ID', 'Generator.ID', 'Genid',
                               'Nuclear Unit Id'],
            'net_gen_mwh':    ['Net Generation (Megawatthours)',
                               'Net Generation (MWh)', 'Net Generation',
                               'net_gen_mwh', 'NET_GENERATION',
                               'Electricity Net Generation (Thousand Kilowatthours)',
                               'Annual Net Generation (MWh)',
                               'generation'],
            'fuel_mmbtu':     ['Total Fuel Consumption MMBtu',
                               'ELEC_FUEL_CONSUMPTION_MMBTU',
                               'Fuel Consumed (MMBtu)',
                               'Elec Fuel Consumption MMBtu',
                               'fuel_mmbtu', 'FUEL_CONSUMED',
                               'Total Fuel Consumed (MMBtu)',
                               'Electricity Fuel Consumption (MMBtu)',
                               'consumption-for-eg-btu'],
            'fuel_type':      ['Reported Fuel Type Code', 'Energy Source',
                               'Fuel Type', 'fuel_type', 'FUEL_TYPE',
                               'AER Fuel Type Code', 'FUEL_CODE',
                               'fuel2002'],
            'prime_mover':    ['Prime Mover', 'Prime Mover Code', 'prime_mover',
                               'PRIME_MOVER', 'primeMover'],
        }

        # Filter out EIA API "ALL" summary rows (fuel2002='ALL' or primeMover='ALL')
        for all_col in ['fuel2002', 'primeMover']:
            if all_col in raw.columns:
                raw = raw[raw[all_col].astype(str).str.upper() != 'ALL'].copy()

        df = pd.DataFrame()
        for std_name, candidates in col_map.items():
            col = _fuzzy_col(raw, candidates)
            if col is not None:
                df[std_name] = raw[col]
            else:
                df[std_name] = np.nan

        # Look for monthly generation columns (Jan through Dec)
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        for i, month in enumerate(month_names, 1):
            col_name = f'gen_month_{i:02d}'
            candidates = [
                f'Netgen {month}', f'Net Generation {month}',
                f'{month} Net Generation', f'NETGEN_{month.upper()[:3]}',
                f'Net Gen {month[:3]}',
            ]
            col = _fuzzy_col(raw, candidates)
            if col is not None:
                df[col_name] = pd.to_numeric(raw[col], errors='coerce')

        # Coerce numeric
        for col in ['net_gen_mwh', 'fuel_mmbtu']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        self.eia923 = df
        return df

    def load_campd(self) -> Optional[pd.DataFrame]:
        """Load EPA CAMPD (Clean Air Markets Program Data) hourly emissions.

        Reads from market-simulator/data/epa-campd/{state}/.
        Supports JSON (EIA API format) and CSV/XLSX.

        Returns:
            DataFrame with standardized column names, or None if no data found.
        """
        path = os.path.join(self.data_root, 'epa-campd', self.state)
        raw = _read_data_files(path)
        if raw is None:
            logger.debug(f"No EPA CAMPD data files in {path} — skipping (optional)")
            return None

        col_map = {
            'facility_id':  ['facilityId', 'Facility ID', 'ORISPL_CODE', 'facility_id',
                             'FACILITY_ID', 'Plant ID'],
            'facility_name': ['facilityName', 'Facility Name', 'facility_name'],
            'date':         ['date', 'Date', 'OP_DATE', 'DATE', 'Operating Date'],
            'hour':         ['hour', 'Hour', 'OP_HOUR', 'HOUR', 'Operating Hour'],
            'co2_tons':     ['co2Mass', 'CO2 (tons)', 'CO2_MASS', 'co2_tons',
                             'CO2_TONS', 'CO2 Mass (short tons)', 'CO2 Mass'],
            'heat_input':   ['heatInput', 'Heat Input (mmBtu)', 'HEAT_INPUT',
                             'heat_input', 'HEAT_INPUT_MMBTU', 'Heat Input'],
            'gross_load':   ['grossLoad', 'Gross Load (MW)', 'GLOAD', 'gross_load',
                             'GROSS_LOAD', 'Gross Load'],
            'nox_mass':     ['noxMass', 'NOx Mass', 'NOX_MASS', 'nox_mass'],
            'so2_mass':     ['so2Mass', 'SO2 Mass', 'SO2_MASS', 'so2_mass'],
        }

        # Check if the raw DataFrame has any recognizable CAMPD columns
        matched_cols = {}
        for std_name, candidates in col_map.items():
            col = _fuzzy_col(raw, candidates)
            matched_cols[std_name] = col

        # Validate: at minimum we need facility_id and one of gross_load/heat_input/co2_tons
        critical_fields = ['facility_id']
        value_fields = ['gross_load', 'heat_input', 'co2_tons']
        has_critical = all(matched_cols[f] is not None for f in critical_fields)
        has_any_value = any(matched_cols[f] is not None for f in value_fields)

        if not has_critical or not has_any_value:
            found_cols = [f"{k}={v}" for k, v in matched_cols.items() if v is not None]
            missing_cols = [k for k in critical_fields + value_fields if matched_cols[k] is None]
            logger.warning(
                f"EPA CAMPD data in {path} does not have valid columns for fleet modeling. "
                f"Found: {found_cols or 'none'}. Missing required: {missing_cols}. "
                f"Raw columns: {list(raw.columns[:20])}. Skipping CAMPD for {self.state}."
            )
            return None

        df = pd.DataFrame()
        for std_name, candidates in col_map.items():
            col = matched_cols[std_name]
            if col is not None:
                df[std_name] = raw[col]
            else:
                df[std_name] = np.nan

        for col in ['co2_tons', 'heat_input', 'gross_load', 'nox_mass', 'so2_mass']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Log validation summary
        n_valid_load = int((df['gross_load'] > 0).sum())
        n_total = len(df)
        logger.info(f"[{self.state}] EPA CAMPD loaded: {n_total} records, "
                     f"{n_valid_load} with valid gross_load")

        self.campd = df
        return df

    def load_plant_heat_rates(self) -> Dict[str, float]:
        """Load pre-computed plant heat rates from plant_heat_rates.json.

        Returns dict of plant_id (str) -> heat_rate (float).
        Returns empty dict if file not found.
        """
        hr_path = os.path.join(self.data_root, 'plant_heat_rates.json')
        if not os.path.exists(hr_path):
            logger.debug(f"plant_heat_rates.json not found at {hr_path}")
            return {}
        try:
            with open(hr_path) as f:
                raw = json.load(f)
            result = {}
            for k, v in raw.items():
                if isinstance(v, dict) and 'heat_rate' in v and v['heat_rate'] is not None:
                    try:
                        result[str(k)] = float(v['heat_rate'])
                    except (ValueError, TypeError):
                        continue
                elif isinstance(v, (int, float)):
                    result[str(k)] = float(v)
            logger.info(f"Loaded {len(result)} plant heat rates from {hr_path}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load plant_heat_rates.json: {e}")
            return {}

    def validate_emissions(self) -> Optional[pd.DataFrame]:
        """Validate fleet emission rates against EPA CAMPD data.

        Cross-references CAMPD hourly CO2 and heat input data with fleet
        generators to compare modeled vs. measured emission intensities.

        Returns:
            DataFrame with per-unit-type validation metrics, or None if CAMPD unavailable.
        """
        if self.campd is None:
            self.load_campd()
        if self.campd is None or len(self.campd) == 0:
            logger.debug("No CAMPD data available for emission validation — skipping (optional)")
            return None

        campd = self.campd.copy()

        # Aggregate CAMPD: total CO2 and gross load per facility
        if 'facility_id' in campd.columns:
            campd['facility_id'] = campd['facility_id'].astype(str).str.strip()

        agg = campd.groupby('facility_id', as_index=False).agg({
            'co2_tons': 'sum',
            'gross_load': 'sum',
            'heat_input': 'sum',
        })

        # Compute measured emission rate (tons CO2 / MWh)
        load = agg['gross_load'].values.astype(float)
        co2 = agg['co2_tons'].values.astype(float)
        valid = (load > 0) & np.isfinite(load) & np.isfinite(co2)
        measured_co2 = np.full(len(load), np.nan)
        measured_co2[valid] = co2[valid] / load[valid]
        agg['measured_co2_rate'] = measured_co2

        # Compute measured heat rate from CAMPD
        heat = agg['heat_input'].values.astype(float)
        valid_hr = (load > 0) & np.isfinite(load) & np.isfinite(heat)
        measured_hr = np.full(len(load), np.nan)
        measured_hr[valid_hr] = heat[valid_hr] / load[valid_hr]
        agg['measured_heat_rate'] = measured_hr

        # If we have fleet data, aggregate by unit type
        if self.fleet is not None and len(self.fleet) > 0:
            fleet_agg = self.fleet.groupby('unit_type', as_index=False).agg({
                'heat_rate': 'mean',
            })
            fleet_agg['modeled_co2_rate'] = fleet_agg['unit_type'].map(DEFAULT_CO2_RATES)

            overall_measured = float(np.nanmean(agg['measured_co2_rate'].values))
            overall_measured_hr = float(np.nanmean(agg['measured_heat_rate'].values))

            results = []
            for _, row in fleet_agg.iterrows():
                ut = row['unit_type']
                results.append({
                    'unit_type': ut,
                    'modeled_co2_rate': row['modeled_co2_rate'],
                    'fleet_avg_heat_rate': row['heat_rate'],
                    'campd_avg_co2_rate': overall_measured,
                    'campd_avg_heat_rate': overall_measured_hr,
                })
            return pd.DataFrame(results)

        return agg[['facility_id', 'measured_co2_rate', 'measured_heat_rate']].copy()

    # ------------------------------------------------------------------
    # Fleet building
    # ------------------------------------------------------------------

    def build_fleet(self) -> pd.DataFrame:
        """Build the unified fleet DataFrame from EIA 860 + 923 data.

        Loads both data sources (if not already loaded), cross-references by
        Plant ID + Generator ID, computes revealed heat rates, classifies units,
        and filters to fossil thermal generators.

        Falls back to stylized defaults from lmp_engine constants if no EIA data
        is available.

        Returns:
            Fleet DataFrame with columns: plant_id, gen_id, capacity_mw,
            heat_rate_design, heat_rate_revealed, unit_type, fuel_cat,
            online_year, retirement_year, iso, capacity_factor.
        """
        # Load if not already loaded
        if self.eia860 is None:
            self.load_eia860()
        if self.eia923 is None:
            self.load_eia923()
        if self.campd is None:
            self.load_campd()

        # If no real data, fall back to stylized defaults
        if self.eia860 is None or len(self.eia860) == 0:
            return self._build_fallback_fleet()

        df = self.eia860.copy()

        # Classify each generator
        df['unit_type'] = df.apply(
            lambda r: _classify_unit(r.get('prime_mover'), r.get('fuel_type')),
            axis=1
        )

        # Keep only fossil thermal units
        df = df[df['unit_type'].notna()].copy()
        if len(df) == 0:
            return self._build_fallback_fleet()

        # Map fuel category
        df['fuel_cat'] = df['fuel_type'].map(
            lambda x: FUEL_TYPE_MAP.get(str(x).strip().upper()) if pd.notna(x) else None
        )

        # Store design heat rate from 860
        df = df.rename(columns={'heat_rate': 'heat_rate_design'})

        # Cross-reference with 923 for revealed heat rates and capacity factors
        df['heat_rate_revealed'] = np.nan
        df['capacity_factor'] = np.nan
        df['net_gen_mwh'] = np.nan

        if self.eia923 is not None and len(self.eia923) > 0:
            eia923 = self.eia923.copy()

            # Coerce plant_id to string for merge
            df['plant_id'] = df['plant_id'].astype(str).str.strip()
            if 'plant_id' in eia923.columns:
                eia923['plant_id'] = eia923['plant_id'].astype(str).str.strip()

            logger.info(f"[{self.state}] EIA 860 generators: {len(df)}, "
                        f"EIA 923 records: {len(eia923)}")

            # Check if 923 has gen_id — EIA API JSON typically doesn't
            has_gen_id = ('gen_id' in eia923.columns and
                          eia923['gen_id'].notna().any() and
                          (eia923['gen_id'].astype(str).str.strip() != 'nan').any())

            if has_gen_id:
                df['gen_id'] = df['gen_id'].astype(str).str.strip()
                eia923['gen_id'] = eia923['gen_id'].astype(str).str.strip()
                agg_923 = eia923.groupby(['plant_id', 'gen_id'], as_index=False).agg({
                    'net_gen_mwh': 'sum',
                    'fuel_mmbtu': 'sum',
                })
                logger.info(f"[{self.state}] 923 merge: gen-level, "
                            f"{len(agg_923)} agg records")
                df = df.merge(agg_923, on=['plant_id', 'gen_id'], how='left',
                              suffixes=('', '_923'))
            else:
                # Fall back to plant-level merge (aggregate all gens per plant)
                agg_923 = eia923.groupby('plant_id', as_index=False).agg({
                    'net_gen_mwh': 'sum',
                    'fuel_mmbtu': 'sum',
                })
                logger.info(f"[{self.state}] 923 merge: plant-level (no gen_id), "
                            f"{len(agg_923)} unique plants")
                df = df.merge(agg_923, on='plant_id', how='left',
                              suffixes=('', '_923'))

            # Use 923 values where available
            if 'net_gen_mwh_923' in df.columns:
                df['net_gen_mwh'] = df['net_gen_mwh_923'].fillna(df.get('net_gen_mwh', np.nan))
                df.drop(columns=['net_gen_mwh_923'], inplace=True, errors='ignore')
            elif 'net_gen_mwh' not in df.columns:
                df['net_gen_mwh'] = np.nan

            if 'fuel_mmbtu' not in df.columns:
                df['fuel_mmbtu'] = np.nan

            # Log 923 merge success rate
            matched_923 = int(df['net_gen_mwh'].notna().sum())
            logger.info(f"[{self.state}] 923 merge result: {matched_923}/{len(df)} "
                        f"generators matched ({100*matched_923/max(len(df),1):.0f}%)")

            # Compute revealed heat rate: fuel consumed / net generation
            # Vectorized — avoid per-row loops
            gen = df['net_gen_mwh'].values.astype(float)
            fuel = df['fuel_mmbtu'].values.astype(float)
            valid = (gen > 0) & (fuel > 0) & np.isfinite(gen) & np.isfinite(fuel)
            revealed = np.full(len(df), np.nan)
            revealed[valid] = fuel[valid] / gen[valid]
            df['heat_rate_revealed'] = revealed

            # Capacity factor (annual MWh / (capacity_MW * 8760))
            cap = df['capacity_mw'].values.astype(float)
            cf = np.full(len(df), np.nan)
            valid_cf = (cap > 0) & (gen > 0) & np.isfinite(cap) & np.isfinite(gen)
            cf[valid_cf] = gen[valid_cf] / (cap[valid_cf] * 8760.0)
            df['capacity_factor'] = np.clip(cf, 0.0, 1.0)
        else:
            if 'fuel_mmbtu' not in df.columns:
                df['fuel_mmbtu'] = np.nan

        # Merge CAMPD measured heat rates as additional fallback source
        df['heat_rate_campd'] = np.nan
        if self.campd is not None and len(self.campd) > 0:
            campd = self.campd.copy()
            if 'facility_id' in campd.columns:
                campd['facility_id'] = campd['facility_id'].astype(str).str.strip()
                # CAMPD facility_id = EIA plant_id (ORISPL code)
                campd_agg = campd.groupby('facility_id', as_index=False).agg({
                    'heat_input': 'sum',
                    'gross_load': 'sum',
                })
                load_arr = campd_agg['gross_load'].values.astype(float)
                heat_arr = campd_agg['heat_input'].values.astype(float)
                valid_campd = (load_arr > 0) & np.isfinite(load_arr) & np.isfinite(heat_arr)
                campd_hr_arr = np.full(len(load_arr), np.nan)
                campd_hr_arr[valid_campd] = heat_arr[valid_campd] / load_arr[valid_campd]
                campd_agg['measured_heat_rate'] = campd_hr_arr
                campd_hr = campd_agg[['facility_id', 'measured_heat_rate']].rename(
                    columns={'facility_id': 'plant_id', 'measured_heat_rate': 'heat_rate_campd'})
                campd_hr['plant_id'] = campd_hr['plant_id'].astype(str).str.strip()
                df = df.merge(campd_hr, on='plant_id', how='left', suffixes=('', '_campd_merge'))
                if 'heat_rate_campd_campd_merge' in df.columns:
                    df['heat_rate_campd'] = df['heat_rate_campd_campd_merge'].fillna(df['heat_rate_campd'])
                    df.drop(columns=['heat_rate_campd_campd_merge'], inplace=True, errors='ignore')
                matched_campd = int(df['heat_rate_campd'].notna().sum())
                logger.info(f"[{self.state}] CAMPD heat rates: {matched_campd}/{len(df)} "
                            f"generators matched")

        # Load pre-computed plant heat rates (highest priority — best-of EIA 923 > CAMPD)
        precomputed_hr_map = self.load_plant_heat_rates()
        if precomputed_hr_map:
            df['heat_rate_precomputed'] = (
                df['plant_id'].astype(str).str.strip().map(precomputed_hr_map)
            )
            n_precomputed = int(df['heat_rate_precomputed'].notna().sum())
            logger.info(f"[{self.state}] Pre-computed heat rates: {n_precomputed}/{len(df)} "
                        f"generators matched from plant_heat_rates.json")
        else:
            df['heat_rate_precomputed'] = np.nan

        # Best heat rate: precomputed > revealed (923) > CAMPD measured > design (860) > default
        hr_precomputed = df['heat_rate_precomputed'].values.astype(float)
        hr_design = df['heat_rate_design'].values.astype(float)
        hr_revealed = df['heat_rate_revealed'].values.astype(float)
        hr_campd = df['heat_rate_campd'].values.astype(float)
        hr_default = df['unit_type'].map(DEFAULT_HEAT_RATES).values.astype(float)

        valid_range = lambda hr: np.isfinite(hr) & (hr > 3.0) & (hr < 30.0)

        best_hr = np.where(valid_range(hr_precomputed), hr_precomputed,
                  np.where(valid_range(hr_revealed), hr_revealed,
                  np.where(valid_range(hr_campd), hr_campd,
                  np.where(valid_range(hr_design), hr_design,
                           hr_default))))
        df['heat_rate'] = best_hr

        # Track which source was used for each generator
        hr_source = np.where(valid_range(hr_precomputed), 'precomputed',
                   np.where(valid_range(hr_revealed), 'revealed_923',
                   np.where(valid_range(hr_campd), 'campd',
                   np.where(valid_range(hr_design), 'design_860',
                            'default'))))
        df['heat_rate_source'] = hr_source

        # Diagnostic logging
        n_precomputed = int((hr_source == 'precomputed').sum())
        n_revealed = int((hr_source == 'revealed_923').sum())
        n_campd = int((hr_source == 'campd').sum())
        n_design = int((hr_source == 'design_860').sum())
        n_default = int((hr_source == 'default').sum())
        logger.info(f"[{self.state}] Heat rate sources: "
                    f"{n_precomputed} precomputed, {n_revealed} revealed (923), "
                    f"{n_campd} CAMPD, {n_design} design (860), {n_default} default")
        logger.info(f"[{self.state}] Heat rate stats: "
                    f"min={df['heat_rate'].min():.2f}, "
                    f"median={df['heat_rate'].median():.2f}, "
                    f"max={df['heat_rate'].max():.2f}, "
                    f"std={df['heat_rate'].std():.2f}")

        # Filter out unreasonable entries
        df = df[df['capacity_mw'] > 0].copy()

        # Current year for age calculation
        current_year = 2025
        online = df['online_year'].values.astype(float)
        df['age'] = np.where(np.isfinite(online), current_year - online, np.nan)

        self.fleet = df
        self.using_fallback = False
        return df

    def _build_fallback_fleet(self) -> pd.DataFrame:
        """Build a stylized fleet using default constants when no EIA data is available.

        Creates synthetic generators for each unit type with default heat rates,
        matching the structure that lmp_engine.py uses.

        Returns:
            Fleet DataFrame with stylized generator entries.
        """
        logger.debug(f"No EIA data for {self.state} — using stylized defaults")
        self.using_fallback = True

        rows = []
        for unit_type, hr in DEFAULT_HEAT_RATES.items():
            fuel_cat = 'coal' if 'coal' in unit_type else ('oil' if 'oil' in unit_type else 'gas')
            rows.append({
                'plant_id': f'stylized_{unit_type}',
                'gen_id': '1',
                'capacity_mw': 10000.0,  # Placeholder
                'heat_rate_design': hr,
                'heat_rate_revealed': np.nan,
                'heat_rate': hr,
                'unit_type': unit_type,
                'fuel_cat': fuel_cat,
                'online_year': 2000,
                'retirement_year': np.nan,
                'iso': self.iso,
                'capacity_factor': 0.50,
                'net_gen_mwh': np.nan,
                'age': 25.0,
            })

        self.fleet = pd.DataFrame(rows)
        return self.fleet

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------

    def bin_by_heat_rate(self) -> Dict[str, pd.DataFrame]:
        """Bin generators by heat rate into efficiency tiers.

        If use_quintiles=True, computes quintile boundaries from actual fleet data.
        Otherwise uses the configured fixed bin thresholds.

        Returns:
            Dict of bin_name → DataFrame of generators in that bin.
        """
        if self.fleet is None:
            self.build_fleet()

        fleet = self.fleet
        bins = {}

        for unit_type, bin_defs in [('gas_ccgt', self.hr_bins_ccgt),
                                     ('coal_steam', self.hr_bins_coal),
                                     ('gas_ct', self.hr_bins_ct)]:
            mask = fleet['unit_type'] == unit_type
            subset = fleet[mask].copy()

            if len(subset) == 0:
                continue

            if self.use_quintiles:
                # Compute quintile boundaries from actual data
                hr_vals = subset['heat_rate'].dropna().values
                if len(hr_vals) < 5:
                    # Too few units — single bin
                    bins[unit_type] = subset
                    continue
                q20, q40, q60, q80 = np.percentile(hr_vals, [20, 40, 60, 80])
                quintile_bins = {
                    f'{unit_type}_q1': (0.0, q20),
                    f'{unit_type}_q2': (q20, q40),
                    f'{unit_type}_q3': (q40, q60),
                    f'{unit_type}_q4': (q60, q80),
                    f'{unit_type}_q5': (q80, 99.0),
                }
                for bin_name, (lo, hi) in quintile_bins.items():
                    bmask = (subset['heat_rate'] >= lo) & (subset['heat_rate'] < hi)
                    if bmask.any():
                        bins[bin_name] = subset[bmask].copy()
            else:
                for bin_name, (lo, hi) in bin_defs.items():
                    bmask = (subset['heat_rate'] >= lo) & (subset['heat_rate'] < hi)
                    if bmask.any():
                        bins[bin_name] = subset[bmask].copy()

        # Oil CT — typically a single bin (small fleet)
        oil_mask = fleet['unit_type'] == 'oil_ct'
        oil_sub = fleet[oil_mask]
        if len(oil_sub) > 0:
            bins['oil_ct'] = oil_sub.copy()

        self.bins = bins
        return bins

    # ------------------------------------------------------------------
    # Merit-order stack
    # ------------------------------------------------------------------

    def build_merit_order_stack(self, fuel_level: str = 'Medium',
                                co2_price: float = 0.0,
                                custom_fuel_prices: dict = None,
                                ) -> Tuple[List[Tuple[str, float, float]], float]:
        """Build a merit-order stack from the real fleet data.

        Computes marginal cost per generator bin: (heat_rate * fuel_price) + VOM + (CO2_rate * co2_price).
        Orders bins by marginal cost (cheapest dispatched first).

        Compatible with lmp_engine.build_merit_order_stack() return format.

        Args:
            fuel_level: 'Low', 'Medium', or 'High' fuel price scenario.
            co2_price: Carbon price in $/ton CO2.
            custom_fuel_prices: Optional dict {'gas': X, 'coal': Y, 'oil': Z}
                overriding fuel_level defaults (e.g. year-varying AEO prices).

        Returns:
            stack: List of (bin_name, capacity_mw, marginal_cost_per_mwh) tuples,
                ordered by marginal cost ascending.
            total_capacity_mw: Total fossil MW in the stack.
        """
        if not self.bins:
            self.bin_by_heat_rate()

        fp = custom_fuel_prices if custom_fuel_prices else DEFAULT_FUEL_PRICES.get(fuel_level, DEFAULT_FUEL_PRICES['Medium'])

        stack = []
        for bin_name, bin_df in self.bins.items():
            total_mw = bin_df['capacity_mw'].sum()
            avg_hr = bin_df['heat_rate'].mean()

            # Determine fuel price based on bin name
            if 'coal' in bin_name:
                fuel_price = fp['coal']
                base_type = 'coal_steam'
            elif 'oil' in bin_name:
                fuel_price = fp['oil']
                base_type = 'oil_ct'
            elif 'ccgt' in bin_name:
                fuel_price = fp['gas']
                base_type = 'gas_ccgt'
            else:
                fuel_price = fp['gas']
                base_type = 'gas_ct'

            vom = DEFAULT_VOM.get(base_type, 4.0)
            co2_rate = DEFAULT_CO2_RATES.get(base_type, 0.5)

            marginal_cost = avg_hr * fuel_price + vom + co2_rate * co2_price
            stack.append((bin_name, float(total_mw), float(marginal_cost)))

        # Sort by marginal cost ascending (cheapest first)
        stack.sort(key=lambda x: x[2])

        total_mw = sum(s[1] for s in stack)
        return stack, total_mw

    # ------------------------------------------------------------------
    # Zonal merit-order stacks
    # ------------------------------------------------------------------

    def assign_zones(self):
        """Assign each plant in self.fleet to a transmission zone.

        Uses pipeline_config.get_zone_for_plant() with BA → zone mapping,
        lat/lon bounding box fallback, and largest-demand-zone default.
        Adds a 'zone' column to self.fleet.
        """
        from pipeline_config import get_zone_for_plant, ZONE_CONFIG

        iso = self.iso
        if iso not in ZONE_CONFIG or self.fleet is None or self.fleet.empty:
            return

        def _get_zone(row):
            ba = row.get('ba', None)
            lat = row.get('latitude', None)
            lon = row.get('longitude', None)
            if pd.notna(ba):
                ba = str(ba).strip().upper()
            else:
                ba = None
            lat = float(lat) if pd.notna(lat) else None
            lon = float(lon) if pd.notna(lon) else None
            return get_zone_for_plant(iso, ba_code=ba, lat=lat, lon=lon)

        self.fleet['zone'] = self.fleet.apply(_get_zone, axis=1)

    def build_zonal_merit_order_stacks(self, fuel_level: str = 'Medium',
                                        co2_price: float = 0.0,
                                        custom_fuel_prices: dict = None,
                                        ) -> Dict[str, List[Tuple[str, float, float]]]:
        """Build per-zone merit-order stacks.

        Splits the fleet by zone, bins by heat rate within each zone,
        and returns a dict {zone_name: [(bin_name, capacity_mw, mc), ...]}.

        Args:
            fuel_level: 'Low', 'Medium', or 'High' fuel price scenario.
            co2_price: Carbon price in $/ton CO2.
            custom_fuel_prices: Optional dict {'gas': X, 'coal': Y, 'oil': Z}
                overriding fuel_level defaults (e.g. year-varying AEO prices).

        Returns:
            Dict mapping zone name to sorted merit-order stack.
        """
        from pipeline_config import ZONE_CONFIG

        if self.fleet is None or self.fleet.empty:
            return {}

        if 'zone' not in self.fleet.columns:
            self.assign_zones()

        iso = self.iso
        if iso not in ZONE_CONFIG:
            return {}

        fp = custom_fuel_prices if custom_fuel_prices else DEFAULT_FUEL_PRICES.get(fuel_level, DEFAULT_FUEL_PRICES['Medium'])
        zone_stacks = {}

        for zone_name in ZONE_CONFIG[iso]['zones']:
            zone_fleet = self.fleet[self.fleet['zone'] == zone_name]
            if zone_fleet.empty:
                zone_stacks[zone_name] = []
                continue

            stack = []
            # Group by unit_type within this zone
            for unit_type, group in zone_fleet.groupby('unit_type'):
                total_mw = group['capacity_mw'].sum()
                avg_hr = group['heat_rate'].mean()

                if 'coal' in str(unit_type):
                    fuel_price = fp['coal']
                    base_type = 'coal_steam'
                elif 'oil' in str(unit_type):
                    fuel_price = fp['oil']
                    base_type = 'oil_ct'
                elif 'ccgt' in str(unit_type):
                    fuel_price = fp['gas']
                    base_type = 'gas_ccgt'
                else:
                    fuel_price = fp['gas']
                    base_type = 'gas_ct'

                vom = DEFAULT_VOM.get(base_type, 4.0)
                co2_rate = DEFAULT_CO2_RATES.get(base_type, 0.5)
                marginal_cost = avg_hr * fuel_price + vom + co2_rate * co2_price
                stack.append((str(unit_type), float(total_mw), float(marginal_cost)))

            stack.sort(key=lambda x: x[2])
            zone_stacks[zone_name] = stack

        return zone_stacks

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_fleet_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get per-bin summary statistics.

        Returns:
            Dict of bin_name → {count, total_mw, avg_hr, avg_age, avg_cf}.
        """
        if not self.bins:
            self.bin_by_heat_rate()

        summary = {}
        for bin_name, bin_df in self.bins.items():
            summary[bin_name] = {
                'count': len(bin_df),
                'total_mw': float(bin_df['capacity_mw'].sum()),
                'avg_hr': float(bin_df['heat_rate'].mean()),
                'avg_age': float(bin_df['age'].mean()) if 'age' in bin_df.columns else None,
                'avg_cf': float(bin_df['capacity_factor'].mean())
                          if 'capacity_factor' in bin_df.columns
                             and bin_df['capacity_factor'].notna().any()
                          else None,
            }
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Export fleet model as a structured dict for the market simulation engine.

        Returns:
            Dict with keys: state, iso, using_fallback, fleet_summary, merit_order
            (at Low/Medium/High fuel prices), and unit_type_totals.
        """
        summary = self.get_fleet_summary()

        # Build stacks at all three fuel price levels
        stacks = {}
        for level in ['Low', 'Medium', 'High']:
            stack, total_mw = self.build_merit_order_stack(fuel_level=level)
            stacks[level] = {
                'stack': [{'bin': s[0], 'capacity_mw': s[1], 'marginal_cost': s[2]}
                          for s in stack],
                'total_capacity_mw': total_mw,
            }

        # Unit-type totals
        type_totals = {}
        if self.fleet is not None:
            for ut in self.fleet['unit_type'].unique():
                mask = self.fleet['unit_type'] == ut
                type_totals[ut] = {
                    'count': int(mask.sum()),
                    'total_mw': float(self.fleet.loc[mask, 'capacity_mw'].sum()),
                    'avg_hr': float(self.fleet.loc[mask, 'heat_rate'].mean()),
                }

        return {
            'state': self.state,
            'iso': self.iso,
            'using_fallback': self.using_fallback,
            'fleet_summary': summary,
            'merit_order': stacks,
            'unit_type_totals': type_totals,
        }


# Module-level cache for load_iso_fleet() — avoids re-reading all state
# directories from disk on every call during the 1,215-scenario sweep.
# Key: (iso, data_root), Value: DataFrame or None.
_ISO_FLEET_CACHE: Dict[tuple, Optional[pd.DataFrame]] = {}


def load_iso_fleet(iso, data_root=None, use_cache=True):
    """Load all EIA 860 generators for an ISO with real heat rates from 923/CAMPD.

    Uses FleetModel.build_fleet() to get the full heat rate cascade:
    revealed (EIA 923) > measured (EPA CAMPD) > design (EIA 860) > default.

    Loads data from ALL available states and filters by BA_TO_ISO mapping.
    This correctly handles multi-BA states (e.g., TX has ERCOT + SPP).

    Args:
        iso: ISO region identifier (e.g., 'ERCOT', 'PJM').
        data_root: Override data directory. Default: market-simulator/data/.
        use_cache: If True (default), cache results to avoid redundant disk reads
            during sweep runs. Set False to force reload.

    Returns DataFrame of generators belonging to the requested ISO, or None.
    """
    if data_root is None:
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    cache_key = (iso, data_root)
    if use_cache and cache_key in _ISO_FLEET_CACHE:
        cached = _ISO_FLEET_CACHE[cache_key]
        return cached.copy() if cached is not None else None

    eia860_dir = os.path.join(data_root, 'eia-860')
    if not os.path.isdir(eia860_dir):
        _ISO_FLEET_CACHE[cache_key] = None
        return None

    frames = []
    for state_dir in sorted(os.listdir(eia860_dir)):
        state_path = os.path.join(eia860_dir, state_dir)
        if not os.path.isdir(state_path):
            continue
        # Skip empty directories (only .gitkeep)
        files = [f for f in os.listdir(state_path) if not f.startswith('.')]
        if not files:
            continue
        try:
            fm = FleetModel(state_dir, data_root=data_root)
            # Use build_fleet() to get real heat rates from 923/CAMPD cascade,
            # not just raw 860 design rates. Previously only called load_eia860()
            # which gave NaN or unreliable design heat rates for most plants.
            df = fm.build_fleet()
            if df is not None and len(df) > 0:
                if 'state' not in df.columns:
                    df['state'] = state_dir
                frames.append(df)
        except Exception:
            continue

    if not frames:
        _ISO_FLEET_CACHE[cache_key] = None
        return None

    merged = pd.concat(frames, ignore_index=True)
    # Filter to generators whose BA maps to the requested ISO
    # build_fleet() may have already set 'iso' column
    if 'iso' in merged.columns:
        iso_mask = merged['iso'] == iso
    elif 'ba' in merged.columns:
        merged['iso'] = merged['ba'].map(lambda x: BA_TO_ISO.get(str(x).strip().upper(), None)
                                          if pd.notna(x) else None)
        iso_mask = merged['iso'] == iso
    else:
        _ISO_FLEET_CACHE[cache_key] = None
        return None

    result = merged[iso_mask].copy()

    if len(result) == 0:
        _ISO_FLEET_CACHE[cache_key] = None
        return None

    _ISO_FLEET_CACHE[cache_key] = result
    return result.copy()


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Fleet Model — build merit-order stacks from EIA 860/923 + EPA CAMPD'
    )
    parser.add_argument('--state', default='TX',
                        help='Two-letter state abbreviation (default: TX)')
    parser.add_argument('--quintiles', action='store_true',
                        help='Use quintile binning instead of fixed thresholds')
    parser.add_argument('--fuel-level', default='Medium', choices=['Low', 'Medium', 'High'],
                        help='Fuel price level (default: Medium)')
    parser.add_argument('--co2-price', type=float, default=5.50,
                        help='CO2 price in $/ton (default: 5.50)')
    parser.add_argument('--json', action='store_true',
                        help='Output full model dict as JSON')
    args = parser.parse_args()

    fm = FleetModel(state=args.state, use_quintiles=args.quintiles)
    fm.build_fleet()
    fm.bin_by_heat_rate()

    if args.json:
        print(json.dumps(fm.to_dict(), indent=2, default=str))
    else:
        print(f"Fleet Model: {args.state} → {fm.iso}")
        print(f"  Data source: {'stylized defaults' if fm.using_fallback else 'EIA 860/923'}")
        print(f"  Total generators: {len(fm.fleet)}")
        print(f"  Total capacity: {fm.fleet['capacity_mw'].sum():,.0f} MW")
        print()

        summary = fm.get_fleet_summary()
        print("Heat-Rate Bins:")
        print(f"  {'Bin':<28s} {'Count':>6s} {'MW':>10s} {'Avg HR':>8s} {'Avg Age':>8s} {'Avg CF':>8s}")
        print("  " + "-" * 70)
        for bin_name, stats in summary.items():
            age_str = f"{stats['avg_age']:.1f}" if stats['avg_age'] is not None else "N/A"
            cf_str = f"{stats['avg_cf']:.2f}" if stats['avg_cf'] is not None else "N/A"
            print(f"  {bin_name:<28s} {stats['count']:>6d} {stats['total_mw']:>10,.0f} "
                  f"{stats['avg_hr']:>8.2f} {age_str:>8s} {cf_str:>8s}")

        print()
        stack, total_mw = fm.build_merit_order_stack(
            fuel_level=args.fuel_level, co2_price=args.co2_price
        )
        print(f"Merit-Order Stack ({args.fuel_level} fuel, ${args.co2_price:.2f}/ton CO2):")
        print(f"  Total fossil capacity: {total_mw:,.0f} MW")
        print(f"  {'Bin':<28s} {'MW':>10s} {'MC $/MWh':>10s}")
        print("  " + "-" * 50)
        for bin_name, cap_mw, mc in stack:
            print(f"  {bin_name:<28s} {cap_mw:>10,.0f} {mc:>10.2f}")

        # Validate emissions if CAMPD data exists
        val = fm.validate_emissions()
        if val is not None and len(val) > 0:
            print()
            print("CAMPD Emission Validation:")
            print(val.to_string(index=False))
