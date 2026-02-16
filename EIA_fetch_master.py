import os
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests

# Constants
BA_TO_ISO = {'CISO': 'CAISO', 'ERCO': 'ERCOT', 'PJM': 'PJM', 'NYIS': 'NYISO', 'ISNE': 'NEISO'}
ISO_TO_BA = {v: k for k, v in BA_TO_ISO.items()}
ISO_TZS = {'CAISO': 'America/Los_Angeles', 'ERCOT': 'America/Chicago', 'PJM': 'America/New_York', 
           'NYISO': 'America/New_York', 'NEISO': 'America/New_York'}

class EIAClient:
    """Encapsulates the 'being' of the EIA API interaction."""
    BASE_URL = "https://api.eia.gov/v2/electricity/rto/"
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()

    def fetch_data(self, endpoint, ba_code, year, facets=None):
        url = f"{self.BASE_URL}{endpoint}/data/"
        params = {
            'api_key': self.api_key,
            'frequency': 'hourly',
            'data[0]': 'value',
            'facets[respondent][]': ba_code,
            'start': f"{year}-01-01T00",
            'end': f"{year}-12-31T23",
            'sort[0][column]': 'period',
            'length': 5000
        }
        if facets: params.update(facets)
        
        response = self.session.get(url, params=params).json()
        return response.get('response', {}).get('data', [])

def process_to_8760(df, tz_name, year):
    """Refactor: Uses Pandas to handle the 'phenomenology' of time (DST/Gaps)."""
    df['period'] = pd.to_datetime(df['period']).dt.tz_localize('UTC')
    df = df.set_index('period')
    
    # Convert to local and filter for target year
    df.index = df.index.tz_convert(tz_name)
    df = df[df.index.year == year]
    
    # Resample to ensure exactly 8760/8784 hours, then interpolate gaps
    df = df.resample('H').mean().interpolate(method='linear').fillna(0)
    
    # Slice to a standard non-leap 8760 for optimizer consistency
    return df.head(8760)['value'].tolist()

def refactor_iso_pipeline(iso, years, api_key):
    """The categorical imperative: A unified pipeline for each ISO."""
    client = EIAClient(api_key)
    ba = ISO_TO_BA[iso]
    tz = ISO_TZS[iso]
    
    iso_results = {'gen': {}, 'demand': {}}
    
    for year in years:
        # Fetch and Process
        raw_gen = client.fetch_data('fuel-type-data', ba, year)
        raw_demand = client.fetch_data('region-data', ba, year, {'facets[type][]': 'D'})
        
        if raw_demand:
            df_d = pd.DataFrame(raw_demand)
            iso_results['demand'][year] = process_to_8760(df_d, tz, year)
            
    return iso, iso_results

# Main Execution Logic
def run_refactored_fetch(api_key, isos=['CAISO', 'ERCOT'], years=[2023, 2024]):
    final_data = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(refactor_iso_pipeline, iso, years, api_key): iso for iso in isos}
        
        for future in as_completed(futures):
            iso, data = future.result()
            final_data[iso] = data
            print(f"Teleology achieved for {iso}.")

    # Saving logic remains standard JSON for 'Optimizer' compatibility
    with open('data/refactored_profiles.json', 'w') as f:
        json.dump(final_data, f)
