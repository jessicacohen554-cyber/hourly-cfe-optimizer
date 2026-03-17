#!/usr/bin/env python3
"""
CCS Proximity Analysis: Spatial viability scoring for CCUS retrofitting of US fossil plants.

Automatically downloads and cross-references:
  1. EPA eGRID 2023 — fossil power plant locations + CO2 emissions
  2. NETL NATCARB — CO2 pipeline routes (ArcGIS REST API)
  3. NETL NATCARB — geologic saline storage polygons (ArcGIS REST API)
  4. EPA/CATF — Class VI injection well permits (curated reference + EPA FRS)

Produces a ranked viability table scoring each plant on:
  - Annual CO2 emissions volume (40%)
  - Distance to nearest CO2 pipeline (25%)
  - Distance to nearest Class VI well (20%)
  - Overlap with geologic storage formation (15%)

Usage:
    pip install -r requirements.txt
    python ccs_proximity_analysis.py

Output:
    output/ccus_viability_ranking.csv
    output/ccus_summary_by_state.csv
    output/ccus_summary_by_iso.csv
    output/ccus_viability_map.png (if matplotlib available)

Data sources:
    - EPA eGRID 2023: https://www.epa.gov/egrid
    - NETL NATCARB: https://netl.doe.gov/carbon-management/carbon-storage/atlas-data
    - CATF Class VI Wells: https://www.catf.us/classviwellsmap/
    - EPA UIC Class VI: https://www.epa.gov/uic/current-class-vi-projects-under-review-epa
"""

import os
import sys
import json
import time
import warnings
import urllib.request

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, shape
from shapely.ops import nearest_points

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target CRS — NAD83 / Conus Albers Equal Area (meters, optimized for CONUS)
TARGET_CRS = "EPSG:5070"

# eGRID download URLs
EGRID_URLS = [
    "https://www.epa.gov/system/files/documents/2025-03/egrid2023_data.xlsx",
    "https://www.epa.gov/system/files/documents/2025-06/egrid2023_data_rev2.xlsx",
]
EGRID_FILE = os.path.join(DATA_DIR, "egrid2023_data.xlsx")

# NETL ArcGIS REST endpoints
NETL_PIPELINE_URL = (
    "https://arcgis.netl.doe.gov/server/rest/services/Hosted/"
    "Co2_Transportation_Pipeline_wma84/FeatureServer/0/query"
)
NETL_SALINE_URL = (
    "https://arcgis.netl.doe.gov/server/rest/services/Hosted/"
    "service_e5c790678b2f40288e0ef98f4ea63675/FeatureServer"
)

# Fossil fuel categories in eGRID PLFUELCT column
FOSSIL_FUELS = {"COAL", "GAS", "OIL", "OFSL", "OTHF"}

# BA code → ISO mapping (from pipeline_config)
BA_TO_ISO = {
    "CISO": "CAISO", "ERCO": "ERCOT", "PJM": "PJM",
    "NYIS": "NYISO", "ISNE": "NEISO", "MISO": "MISO", "SWPP": "SPP",
}

# Viability scoring weights
W_EMISSIONS = 0.40
W_PIPELINE = 0.25
W_WELL = 0.20
W_GEOLOGY = 0.15

# Distance caps for scoring (meters) — beyond these, score contribution → 0
PIPELINE_CAP_M = 200_000  # 200 km
WELL_CAP_M = 300_000      # 300 km


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════

def _download_file(url, dest, label="file"):
    """Download a file with retry logic."""
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        print(f"  [cached] {label}: {dest}")
        return True
    for attempt in range(3):
        try:
            print(f"  Downloading {label} (attempt {attempt+1})...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(dest, "wb") as f:
                    f.write(resp.read())
            size_mb = os.path.getsize(dest) / 1e6
            print(f"  Downloaded {size_mb:.1f} MB → {dest}")
            return True
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return False


def fetch_egrid_plants():
    """
    Download EPA eGRID 2023 and extract fossil power plant locations + emissions.
    Returns GeoDataFrame with columns: ORISPL, PNAME, PSTATABB, LAT, LON,
    NAMEPCAP, PLCO2AN, PLNGENAN, PLFUELCT, BACODE, ISO, geometry.
    """
    print("\n[1/4] Fetching EPA eGRID 2023 plant data...")

    # Download
    downloaded = False
    for url in EGRID_URLS:
        if _download_file(url, EGRID_FILE, "eGRID 2023"):
            downloaded = True
            break
    if not downloaded:
        print("ERROR: Could not download eGRID. Please manually download from:")
        print("  https://www.epa.gov/egrid/detailed-data")
        print(f"  Save to: {EGRID_FILE}")
        sys.exit(1)

    # Read the plant-level sheet
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        os.system(f"{sys.executable} -m pip install openpyxl -q")

    xls = pd.ExcelFile(EGRID_FILE, engine="openpyxl")
    plant_sheet = None
    for name in xls.sheet_names:
        if "PLNT" in name.upper():
            plant_sheet = name
            break
    if plant_sheet is None:
        print(f"ERROR: No plant sheet found. Sheets: {xls.sheet_names}")
        sys.exit(1)

    # eGRID has a description row (row 0) then header row (row 1)
    df = pd.read_excel(xls, sheet_name=plant_sheet, header=1)
    print(f"  Raw plant records: {len(df):,}")

    # Standardize column names to uppercase
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Required columns
    needed = ["LAT", "LON", "PLCO2AN", "PLFUELCT"]
    # eGRID sometimes uses PLNTLAT/PLNTLON or LAT/LON
    lat_col = next((c for c in df.columns if c in ("LAT", "PLNTLAT", "LATITUD")), None)
    lon_col = next((c for c in df.columns if c in ("LON", "PLNTLON", "LONGITUD")), None)
    if lat_col and lat_col != "LAT":
        df.rename(columns={lat_col: "LAT"}, inplace=True)
    if lon_col and lon_col != "LON":
        df.rename(columns={lon_col: "LON"}, inplace=True)

    for col in needed:
        if col not in df.columns:
            print(f"  WARNING: Column '{col}' not found. Available: {list(df.columns)[:20]}")

    # Filter to fossil plants with valid coordinates
    df = df.dropna(subset=["LAT", "LON"])
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON"])

    # Filter CONUS only (exclude AK, HI, territories)
    df = df[(df["LAT"] > 24) & (df["LAT"] < 50) &
            (df["LON"] > -125) & (df["LON"] < -66)]

    # Filter fossil fuels
    if "PLFUELCT" in df.columns:
        df["PLFUELCT"] = df["PLFUELCT"].astype(str).str.strip().str.upper()
        df = df[df["PLFUELCT"].isin(FOSSIL_FUELS)]

    # Clean numeric columns
    for col in ["PLCO2AN", "PLNGENAN", "NAMEPCAP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Map BA to ISO
    if "BACODE" in df.columns:
        df["ISO"] = df["BACODE"].map(BA_TO_ISO).fillna("Other")
    else:
        df["ISO"] = "Unknown"

    # Build GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(df["LON"], df["LAT"])]
    keep_cols = [c for c in ["ORISPL", "PNAME", "PSTATABB", "LAT", "LON",
                              "NAMEPCAP", "PLCO2AN", "PLNGENAN", "PLFUELCT",
                              "BACODE", "ISO"] if c in df.columns]
    gdf = gpd.GeoDataFrame(df[keep_cols].reset_index(drop=True),
                           geometry=geometry, crs="EPSG:4326")

    print(f"  Fossil plants with coordinates: {len(gdf):,}")
    print(f"  Total CO2: {gdf['PLCO2AN'].sum()/1e6:.1f} million short tons")
    return gdf


def _query_arcgis_features(base_url, params_base, label="features"):
    """
    Page through an ArcGIS REST FeatureServer query, returning a list of
    GeoJSON feature dicts. Handles pagination via resultOffset.
    """
    features = []
    offset = 0
    page_size = 2000
    max_pages = 50  # safety limit

    for page in range(max_pages):
        params = {**params_base, "resultOffset": str(offset), "resultRecordCount": str(page_size)}
        query_str = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{base_url}?{query_str}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  ArcGIS query failed at offset {offset}: {e}")
            break

        # GeoJSON format
        if "features" in data:
            batch = data["features"]
            if not batch:
                break
            features.extend(batch)
            print(f"  Fetched {len(features):,} {label}...", end="\r")
            if len(batch) < page_size:
                break
            offset += page_size
        else:
            # JSON format (non-GeoJSON) — try to parse
            if "error" in data:
                print(f"  ArcGIS error: {data['error']}")
            break

        time.sleep(0.3)  # rate-limit politeness

    print(f"  Fetched {len(features):,} {label} total")
    return features


def fetch_co2_pipelines():
    """
    Fetch CO2 pipeline routes from NETL ArcGIS REST API.
    Returns GeoDataFrame with pipeline polylines.
    Falls back to curated reference data if API is unavailable.
    """
    print("\n[2/4] Fetching NETL CO2 pipeline data...")

    cache_file = os.path.join(DATA_DIR, "co2_pipelines.geojson")
    if os.path.exists(cache_file):
        print(f"  [cached] Loading from {cache_file}")
        gdf = gpd.read_file(cache_file)
        if len(gdf) > 0:
            return gdf

    # Try ArcGIS REST API
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "returnGeometry": "true",
    }
    features = _query_arcgis_features(NETL_PIPELINE_URL, params, "pipeline segments")

    if features:
        geojson = {"type": "FeatureCollection", "features": features}
        gdf = gpd.GeoDataFrame.from_features(geojson, crs="EPSG:4326")
        # Cache for future runs
        gdf.to_file(cache_file, driver="GeoJSON")
        print(f"  Pipeline segments: {len(gdf):,}")
        return gdf

    # Fallback: curated reference data for major US CO2 pipelines
    print("  API unavailable — using curated reference pipeline data")
    return _curated_co2_pipelines()


def _curated_co2_pipelines():
    """
    Curated reference data for major operational US CO2 pipelines.
    Source: PHMSA National Pipeline Mapping System public data + DOE/NETL reports.
    These are approximate centerline routes for the ~5,000 miles of existing CO2 pipelines.
    """
    pipelines = [
        # Cortez Pipeline (McElmo Dome, CO → Denver City, TX) ~500 mi
        {"name": "Cortez Pipeline", "operator": "Kinder Morgan", "miles": 504,
         "coords": [(-108.5, 37.3), (-107.5, 36.8), (-106.0, 35.5),
                     (-104.5, 34.5), (-103.5, 33.5), (-103.0, 32.9)]},
        # Sheep Mountain (Sheep Mtn, CO → Permian Basin, TX) ~410 mi
        {"name": "Sheep Mountain Pipeline", "operator": "Occidental", "miles": 410,
         "coords": [(-106.5, 37.8), (-106.0, 36.5), (-105.0, 35.5),
                     (-104.0, 34.0), (-103.2, 33.0), (-102.5, 32.0)]},
        # Bravo Pipeline (Bravo Dome, NM → Permian Basin, TX) ~220 mi
        {"name": "Bravo Pipeline", "operator": "Occidental", "miles": 218,
         "coords": [(-104.0, 36.2), (-103.5, 35.0), (-103.0, 34.0), (-102.5, 32.5)]},
        # Canyon Reef Carriers (Scurry County, TX area) ~139 mi
        {"name": "Canyon Reef Carriers", "operator": "Kinder Morgan", "miles": 139,
         "coords": [(-101.0, 32.7), (-100.5, 32.3), (-100.0, 31.8), (-99.5, 31.3)]},
        # Val Verde Pipeline (Val Verde Basin, TX) ~85 mi
        {"name": "Val Verde Pipeline", "operator": "Various", "miles": 85,
         "coords": [(-101.5, 30.5), (-101.0, 30.3), (-100.5, 30.0)]},
        # Centerline Pipeline (Permian Basin trunk) ~115 mi
        {"name": "Centerline Pipeline", "operator": "Kinder Morgan", "miles": 115,
         "coords": [(-102.8, 32.8), (-102.5, 32.2), (-102.0, 31.5), (-101.5, 31.0)]},
        # NEJD Pipeline (Jackson Dome, MS → Port Arthur, TX) ~183 mi
        {"name": "NEJD Pipeline", "operator": "Denbury", "miles": 183,
         "coords": [(-90.2, 32.3), (-91.0, 31.5), (-92.0, 31.0),
                     (-93.0, 30.5), (-93.9, 29.9)]},
        # Green Pipeline (Free State, MS → Hastings, TX) ~325 mi
        {"name": "Green Pipeline", "operator": "Denbury", "miles": 325,
         "coords": [(-89.8, 31.5), (-90.5, 31.0), (-91.5, 30.5),
                     (-92.5, 30.2), (-93.5, 30.0), (-94.5, 29.6), (-95.1, 29.5)]},
        # Delta Pipeline (Delhi Field, LA → Tinsley, MS) ~85 mi
        {"name": "Delta Pipeline", "operator": "Denbury", "miles": 85,
         "coords": [(-91.5, 32.4), (-90.8, 32.5), (-90.3, 32.6)]},
        # Weyburn Pipeline (Beulah, ND → Weyburn, SK) ~200 mi (US portion)
        {"name": "Weyburn Pipeline (US)", "operator": "Dakota Gasification", "miles": 200,
         "coords": [(-101.8, 47.3), (-102.5, 47.8), (-103.5, 48.5), (-104.0, 49.0)]},
        # Petro Source Pipeline (West TX) ~60 mi
        {"name": "Petro Source Pipeline", "operator": "Petro Source", "miles": 60,
         "coords": [(-102.0, 31.5), (-101.5, 31.2), (-101.0, 31.0)]},
        # Decatur Pipeline (ADM CCS project, IL) ~1 mi (short but significant)
        {"name": "Decatur Pipeline", "operator": "ADM/Archer Daniels Midland", "miles": 1,
         "coords": [(-88.95, 39.85), (-88.94, 39.84)]},
        # Alberta Carbon Trunk Line proxy — US segment analogy
        # Summit Carbon Solutions (planned, IA/NE/SD/ND) — in permitting
        {"name": "Summit Carbon Solutions (proposed)", "operator": "Summit", "miles": 2000,
         "coords": [(-93.5, 42.0), (-94.0, 42.5), (-95.5, 43.0),
                     (-96.5, 43.5), (-97.5, 44.0), (-99.0, 45.0),
                     (-100.5, 46.0), (-101.0, 47.0)]},
        # Navigator CO2 Heartland Greenway (proposed, IL/IA/NE/SD/MN)
        {"name": "Navigator Heartland Greenway (proposed)", "operator": "Navigator CO2", "miles": 1300,
         "coords": [(-89.5, 40.0), (-90.0, 41.0), (-91.5, 41.5),
                     (-93.0, 42.0), (-95.0, 41.5), (-96.5, 41.0),
                     (-97.5, 41.5), (-98.0, 42.0)]},
    ]

    geometries = []
    names = []
    operators = []
    miles_list = []
    for p in pipelines:
        geometries.append(LineString(p["coords"]))
        names.append(p["name"])
        operators.append(p["operator"])
        miles_list.append(p["miles"])

    gdf = gpd.GeoDataFrame(
        {"name": names, "operator": operators, "miles": miles_list},
        geometry=geometries, crs="EPSG:4326"
    )
    # Cache
    gdf.to_file(os.path.join(DATA_DIR, "co2_pipelines.geojson"), driver="GeoJSON")
    print(f"  Curated pipeline segments: {len(gdf)}")
    return gdf


def fetch_saline_formations():
    """
    Fetch geologic saline storage formation polygons from NETL ArcGIS REST API.
    Falls back to curated major US storage basins.
    """
    print("\n[3/4] Fetching NETL saline formation data...")

    cache_file = os.path.join(DATA_DIR, "saline_formations.geojson")
    if os.path.exists(cache_file):
        print(f"  [cached] Loading from {cache_file}")
        gdf = gpd.read_file(cache_file)
        if len(gdf) > 0:
            return gdf

    # Try multiple NETL ArcGIS layers for saline data
    saline_layer_ids = [0, 1, 2, 3]  # try common layer IDs
    for layer_id in saline_layer_ids:
        url = f"{NETL_SALINE_URL}/{layer_id}/query"
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "returnGeometry": "true",
        }
        features = _query_arcgis_features(url, params, f"saline features (layer {layer_id})")
        if features:
            # Filter for polygon geometries (basins/formations)
            poly_features = [
                f for f in features
                if f.get("geometry", {}).get("type", "") in
                   ("Polygon", "MultiPolygon")
            ]
            if poly_features:
                geojson = {"type": "FeatureCollection", "features": poly_features}
                gdf = gpd.GeoDataFrame.from_features(geojson, crs="EPSG:4326")
                gdf.to_file(cache_file, driver="GeoJSON")
                print(f"  Saline formation polygons: {len(gdf):,}")
                return gdf

    # Fallback: curated major US saline storage basins
    print("  API unavailable — using curated reference formation data")
    return _curated_saline_formations()


def _curated_saline_formations():
    """
    Curated reference polygons for major US geologic CO2 storage basins.
    Source: NETL Carbon Storage Atlas (5th ed.), NATCARB, USGS assessments.
    Polygons are simplified approximations of basin boundaries.
    """
    formations = [
        {
            "name": "Gulf Coast Basin (TX/LA/MS/AL)",
            "type": "Saline + Depleted O&G",
            "storage_gt_co2": 500,  # estimated Gt CO2 capacity
            "coords": [(-98.0, 26.0), (-98.0, 31.0), (-95.0, 33.0),
                        (-91.0, 33.0), (-88.0, 31.5), (-87.5, 30.5),
                        (-88.0, 29.0), (-90.0, 28.5), (-93.0, 28.5),
                        (-97.0, 26.5), (-98.0, 26.0)],
        },
        {
            "name": "Illinois Basin (IL/IN/KY)",
            "type": "Saline (Mt. Simon Sandstone)",
            "storage_gt_co2": 150,
            "coords": [(-90.5, 37.0), (-90.0, 39.5), (-88.5, 40.5),
                        (-87.0, 40.5), (-86.5, 39.5), (-86.5, 38.0),
                        (-87.5, 37.0), (-89.0, 36.8), (-90.5, 37.0)],
        },
        {
            "name": "Williston Basin (ND/MT/SD)",
            "type": "Saline + Depleted O&G (Bakken)",
            "storage_gt_co2": 200,
            "coords": [(-106.0, 46.0), (-106.0, 49.0), (-102.0, 49.0),
                        (-100.0, 48.0), (-99.0, 46.5), (-100.0, 45.5),
                        (-103.0, 45.0), (-106.0, 46.0)],
        },
        {
            "name": "Permian Basin (TX/NM)",
            "type": "Saline + Depleted O&G",
            "storage_gt_co2": 100,
            "coords": [(-105.0, 30.5), (-105.0, 33.5), (-102.5, 34.0),
                        (-100.0, 33.0), (-100.0, 30.5), (-102.0, 30.0),
                        (-105.0, 30.5)],
        },
        {
            "name": "Appalachian Basin (WV/PA/OH/KY)",
            "type": "Saline (Oriskany/Clinton/Medina)",
            "storage_gt_co2": 50,
            "coords": [(-84.0, 36.5), (-82.0, 37.0), (-80.0, 38.0),
                        (-79.0, 40.0), (-78.0, 42.0), (-77.0, 42.5),
                        (-77.5, 41.5), (-79.0, 39.5), (-80.5, 38.0),
                        (-82.5, 37.0), (-84.0, 36.5)],
        },
        {
            "name": "Michigan Basin (MI)",
            "type": "Saline (Mt. Simon)",
            "storage_gt_co2": 40,
            "coords": [(-87.0, 42.0), (-86.5, 44.0), (-85.0, 45.5),
                        (-83.5, 45.0), (-83.0, 43.5), (-83.5, 42.0),
                        (-85.0, 41.5), (-87.0, 42.0)],
        },
        {
            "name": "Denver-Julesburg Basin (CO/WY/NE)",
            "type": "Saline + Depleted O&G",
            "storage_gt_co2": 30,
            "coords": [(-106.0, 39.0), (-105.5, 41.5), (-104.0, 43.0),
                        (-102.0, 42.0), (-101.5, 40.5), (-103.0, 38.5),
                        (-106.0, 39.0)],
        },
        {
            "name": "Powder River Basin (WY/MT)",
            "type": "Saline + Coal seams",
            "storage_gt_co2": 25,
            "coords": [(-108.0, 43.0), (-108.0, 46.0), (-105.5, 46.0),
                        (-105.0, 44.5), (-105.5, 43.0), (-108.0, 43.0)],
        },
        {
            "name": "San Joaquin Basin (CA)",
            "type": "Saline + Depleted O&G",
            "storage_gt_co2": 20,
            "coords": [(-121.5, 35.0), (-121.0, 37.5), (-119.5, 37.8),
                        (-118.5, 36.0), (-119.0, 35.0), (-121.5, 35.0)],
        },
        {
            "name": "Anadarko Basin (OK/TX/KS)",
            "type": "Saline + Depleted O&G",
            "storage_gt_co2": 60,
            "coords": [(-101.0, 34.5), (-100.0, 37.0), (-98.0, 37.5),
                        (-97.0, 36.5), (-97.0, 35.0), (-99.0, 34.5),
                        (-101.0, 34.5)],
        },
    ]

    geometries = []
    names = []
    types = []
    capacities = []
    for f in formations:
        geometries.append(Polygon(f["coords"]))
        names.append(f["name"])
        types.append(f["type"])
        capacities.append(f["storage_gt_co2"])

    gdf = gpd.GeoDataFrame(
        {"name": names, "type": types, "storage_gt_co2": capacities},
        geometry=geometries, crs="EPSG:4326"
    )
    gdf.to_file(os.path.join(DATA_DIR, "saline_formations.geojson"), driver="GeoJSON")
    print(f"  Curated formation polygons: {len(gdf)}")
    return gdf


def fetch_class_vi_wells():
    """
    Fetch Class VI injection well locations.
    Primary: EPA FRS geospatial download filtered for UIC.
    Fallback: Curated reference table from EPA permit documents.
    """
    print("\n[4/4] Fetching Class VI injection well data...")

    cache_file = os.path.join(DATA_DIR, "class_vi_wells.geojson")
    if os.path.exists(cache_file):
        print(f"  [cached] Loading from {cache_file}")
        gdf = gpd.read_file(cache_file)
        if len(gdf) > 0:
            return gdf

    # Try EPA Envirofacts REST API for UIC data
    # https://data.epa.gov/efservice/... (UIC endpoint)
    try:
        uic_url = (
            "https://data.epa.gov/efservice/UIC_WELL/WELL_CLASS/=/6/"
            "JSON/COUNT/"
        )
        req = urllib.request.Request(uic_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            count_data = json.loads(resp.read().decode("utf-8"))
            print(f"  EPA Envirofacts: found {count_data} Class VI records")

        # Fetch actual records (Class 6 = Class VI)
        uic_data_url = (
            "https://data.epa.gov/efservice/UIC_WELL/WELL_CLASS/=/6/"
            "JSON/"
        )
        req = urllib.request.Request(uic_data_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            wells_raw = json.loads(resp.read().decode("utf-8"))
            if wells_raw and isinstance(wells_raw, list) and len(wells_raw) > 0:
                df = pd.DataFrame(wells_raw)
                lat_col = next((c for c in df.columns if "LAT" in c.upper()), None)
                lon_col = next((c for c in df.columns if "LON" in c.upper()), None)
                if lat_col and lon_col:
                    df = df.dropna(subset=[lat_col, lon_col])
                    geometry = [Point(lon, lat) for lon, lat in
                                zip(df[lon_col], df[lat_col])]
                    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                    gdf.to_file(cache_file, driver="GeoJSON")
                    print(f"  EPA Envirofacts Class VI wells: {len(gdf)}")
                    return gdf
    except Exception as e:
        print(f"  EPA Envirofacts query failed: {e}")

    # Fallback: curated Class VI well reference data
    print("  Using curated reference Class VI well data")
    return _curated_class_vi_wells()


def _curated_class_vi_wells():
    """
    Curated Class VI well locations from EPA permit documents and CATF tracker.
    Source: https://www.epa.gov/uic/current-class-vi-projects-under-review-epa
    Source: https://www.catf.us/classviwellsmap/
    Includes approved, pending, and primacy-state wells.
    Last updated: 2025 data.
    """
    wells = [
        # --- APPROVED / OPERATING ---
        {"name": "ADM Decatur (IL-003-6A001)", "state": "IL", "status": "Active",
         "operator": "Archer Daniels Midland", "lat": 39.843, "lon": -88.945,
         "formation": "Mt. Simon Sandstone", "source": "EPA Region 5"},
        {"name": "ADM Decatur Well 2 (IL-003-6A002)", "state": "IL", "status": "Active",
         "operator": "Archer Daniels Midland", "lat": 39.842, "lon": -88.946,
         "formation": "Mt. Simon Sandstone", "source": "EPA Region 5"},

        # --- NORTH DAKOTA (State Primacy) ---
        {"name": "Red Trail Energy", "state": "ND", "status": "Approved",
         "operator": "Red Trail Energy", "lat": 46.876, "lon": -102.789,
         "formation": "Broom Creek", "source": "ND NDIC"},
        {"name": "Project Tundra (Milton R. Young)", "state": "ND", "status": "Pending",
         "operator": "Minnkota Power", "lat": 47.542, "lon": -101.360,
         "formation": "Broom Creek/Deadwood", "source": "ND NDIC"},

        # --- LOUISIANA (State Primacy) ---
        {"name": "Bayou Bend CCS (Hackberry)", "state": "LA", "status": "Pending",
         "operator": "TotalEnergies/Chevron/Equinor", "lat": 30.00, "lon": -93.35,
         "formation": "Hackberry Formation", "source": "LA LDNR"},
        {"name": "Cameron LNG CCS", "state": "LA", "status": "Pending",
         "operator": "Sempra/RWE", "lat": 29.78, "lon": -93.33,
         "formation": "Gulf Coast Miocene", "source": "LA LDNR"},
        {"name": "Calcasieu Pass CCS", "state": "LA", "status": "Pending",
         "operator": "Venture Global", "lat": 29.80, "lon": -93.36,
         "formation": "Gulf Coast Saline", "source": "LA LDNR"},
        {"name": "Lake Charles CCS", "state": "LA", "status": "Pending",
         "operator": "Driftwood LNG", "lat": 30.23, "lon": -93.22,
         "formation": "Gulf Coast Saline", "source": "LA LDNR"},

        # --- WYOMING (State Primacy) ---
        {"name": "Dry Fork Station CCS", "state": "WY", "status": "Pending",
         "operator": "Basin Electric", "lat": 44.77, "lon": -106.38,
         "formation": "Madison Limestone", "source": "WY DEQ"},

        # --- TEXAS (EPA Region 6 + State Primacy pending) ---
        {"name": "ExxonMobil Baytown CCS", "state": "TX", "status": "Pending",
         "operator": "ExxonMobil", "lat": 29.77, "lon": -95.01,
         "formation": "Gulf Coast Miocene", "source": "EPA Region 6"},
        {"name": "Occidental Direct Air Capture (Stratos)", "state": "TX", "status": "Pending",
         "operator": "1PointFive/Occidental", "lat": 31.85, "lon": -103.50,
         "formation": "Permian Basin", "source": "EPA Region 6"},
        {"name": "Port Arthur CCS Hub", "state": "TX", "status": "Pending",
         "operator": "Chevron/Talos", "lat": 29.87, "lon": -93.93,
         "formation": "Gulf Coast Saline", "source": "EPA Region 6"},
        {"name": "Freeport LNG CCS", "state": "TX", "status": "Pending",
         "operator": "Freeport LNG", "lat": 28.95, "lon": -95.31,
         "formation": "Gulf Coast Saline", "source": "EPA Region 6"},

        # --- MISSISSIPPI ---
        {"name": "Kemper County CCS", "state": "MS", "status": "Pending",
         "operator": "Mississippi Power", "lat": 32.60, "lon": -88.80,
         "formation": "Lower Tuscaloosa", "source": "EPA Region 4"},

        # --- CALIFORNIA ---
        {"name": "Elk Hills CCS", "state": "CA", "status": "Pending",
         "operator": "California Resources Corp", "lat": 35.29, "lon": -119.48,
         "formation": "Stevens/Monterey", "source": "EPA Region 9"},

        # --- WEST VIRGINIA ---
        {"name": "Mountaineer CCS", "state": "WV", "status": "Approved (suspended)",
         "operator": "AEP/Battelle", "lat": 38.95, "lon": -81.93,
         "formation": "Rose Run/Copper Ridge", "source": "EPA Region 3"},

        # --- IOWA (via Summit Carbon Solutions hub) ---
        {"name": "Summit Carbon IA Hub", "state": "IA", "status": "Pending",
         "operator": "Summit Carbon Solutions", "lat": 42.50, "lon": -94.00,
         "formation": "Jordan Sandstone", "source": "EPA Region 7"},

        # --- UTAH ---
        {"name": "Intermountain Power CCS", "state": "UT", "status": "Pending",
         "operator": "IPP Renewed", "lat": 38.65, "lon": -112.58,
         "formation": "Navajo Sandstone", "source": "EPA Region 8"},

        # --- ALABAMA ---
        {"name": "Plant Barry CCS", "state": "AL", "status": "Completed (pilot)",
         "operator": "Southern Company/SECARB", "lat": 31.01, "lon": -87.95,
         "formation": "Paluxy Formation", "source": "EPA Region 4"},

        # --- KANSAS ---
        {"name": "Coffeyville CCS", "state": "KS", "status": "Pending",
         "operator": "CVR Partners", "lat": 37.04, "lon": -95.61,
         "formation": "Arbuckle Group", "source": "EPA Region 7"},
    ]

    df = pd.DataFrame(wells)
    geometry = [Point(row["lon"], row["lat"]) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf.to_file(os.path.join(DATA_DIR, "class_vi_wells.geojson"), driver="GeoJSON")
    print(f"  Curated Class VI wells: {len(gdf)}")
    return gdf


# ═══════════════════════════════════════════════════════════════════════════
# 2. CRS HARMONIZATION
# ═══════════════════════════════════════════════════════════════════════════

def harmonize_crs(*gdfs):
    """Reproject all GeoDataFrames to the target CONUS Albers CRS (EPSG:5070)."""
    print(f"\nHarmonizing CRS → {TARGET_CRS}")
    result = []
    for gdf in gdfs:
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        result.append(gdf.to_crs(TARGET_CRS))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. SPATIAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_nearest_distances(plants_gdf, target_gdf, label="target"):
    """
    Compute distance from each plant to the nearest feature in target_gdf.
    Uses spatial index (STRtree) for efficiency.
    Returns array of distances in meters (same CRS units).
    """
    print(f"  Computing distances to nearest {label}...")

    if len(target_gdf) == 0:
        print(f"  WARNING: No {label} features — returning NaN distances")
        return np.full(len(plants_gdf), np.nan)

    # Build spatial index from target geometries
    from shapely import STRtree
    target_geoms = target_gdf.geometry.values
    tree = STRtree(target_geoms)

    distances = np.empty(len(plants_gdf))
    plant_geoms = plants_gdf.geometry.values

    for i, plant_pt in enumerate(plant_geoms):
        nearest_idx = tree.nearest(plant_pt)
        distances[i] = plant_pt.distance(target_geoms[nearest_idx])

    # Convert to km for readability
    distances_km = distances / 1000.0
    print(f"  {label} — median distance: {np.nanmedian(distances_km):.0f} km, "
          f"min: {np.nanmin(distances_km):.0f} km, max: {np.nanmax(distances_km):.0f} km")
    return distances  # return meters for scoring


def compute_formation_overlap(plants_gdf, formations_gdf):
    """
    Determine which plants sit within a geologic storage formation polygon.
    Returns boolean array.
    """
    print("  Computing formation overlap (spatial join)...")

    if len(formations_gdf) == 0:
        return np.zeros(len(plants_gdf), dtype=bool)

    # Spatial join: plants within formations
    joined = gpd.sjoin(plants_gdf, formations_gdf, predicate="within", how="left")

    # Plants that matched at least one formation
    matched_idx = joined.dropna(subset=["index_right"]).index.unique()
    overlap = np.zeros(len(plants_gdf), dtype=bool)
    overlap[plants_gdf.index.isin(matched_idx)] = True

    pct = overlap.sum() / len(overlap) * 100
    print(f"  {overlap.sum():,} of {len(overlap):,} plants ({pct:.1f}%) "
          f"within a storage formation")
    return overlap


# ═══════════════════════════════════════════════════════════════════════════
# 4. VIABILITY SCORING
# ═══════════════════════════════════════════════════════════════════════════

def compute_viability_scores(df):
    """
    Compute composite CCUS viability score (0-100) for each plant.

    Components:
      - CO2 emissions (40%): log-scaled, normalized
      - Pipeline distance (25%): inverse distance, capped at 200km
      - Well distance (20%): inverse distance, capped at 300km
      - Formation overlap (15%): binary 0/1

    Returns DataFrame with score columns added.
    """
    print("\nComputing viability scores...")

    # --- CO2 emissions score (0-1) ---
    # Use log scale because emissions span several orders of magnitude
    co2 = df["PLCO2AN"].clip(lower=1)  # avoid log(0)
    log_co2 = np.log10(co2)
    score_co2 = (log_co2 - log_co2.min()) / (log_co2.max() - log_co2.min())

    # --- Pipeline distance score (0-1) ---
    # Closer = higher score. Beyond cap → 0.
    dist_pipe = df["dist_pipeline_m"].fillna(PIPELINE_CAP_M)
    score_pipe = np.clip(1.0 - dist_pipe / PIPELINE_CAP_M, 0, 1)

    # --- Well distance score (0-1) ---
    dist_well = df["dist_well_m"].fillna(WELL_CAP_M)
    score_well = np.clip(1.0 - dist_well / WELL_CAP_M, 0, 1)

    # --- Formation overlap score (0/1) ---
    score_formation = df["within_formation"].astype(float)

    # --- Composite score ---
    composite = (
        W_EMISSIONS * score_co2 +
        W_PIPELINE * score_pipe +
        W_WELL * score_well +
        W_GEOLOGY * score_formation
    ) * 100  # scale to 0-100

    df["score_emissions"] = (score_co2 * 100).round(1)
    df["score_pipeline"] = (score_pipe * 100).round(1)
    df["score_well"] = (score_well * 100).round(1)
    df["score_formation"] = (score_formation * 100).round(1)
    df["viability_score"] = composite.round(1)

    print(f"  Score range: {composite.min():.1f} – {composite.max():.1f}")
    print(f"  Mean: {composite.mean():.1f}, Median: {composite.median():.1f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. OUTPUT & REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def generate_outputs(df, pipelines_gdf, wells_gdf, formations_gdf):
    """Generate CSV outputs and summary statistics."""
    print("\n" + "=" * 80)
    print("CCUS VIABILITY RANKING — RESULTS")
    print("=" * 80)

    # Sort by viability score
    df = df.sort_values("viability_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # --- Top 25 plants ---
    print(f"\n{'Rank':>5} {'Plant Name':<40} {'State':>5} {'ISO':>7} "
          f"{'CO2 (kT)':>10} {'Pipe km':>9} {'Well km':>9} "
          f"{'Geology':>8} {'Score':>7}")
    print("-" * 110)
    for _, row in df.head(25).iterrows():
        co2_kt = row.get("PLCO2AN", 0) / 1000
        pipe_km = row.get("dist_pipeline_m", 0) / 1000
        well_km = row.get("dist_well_m", 0) / 1000
        geol = "YES" if row.get("within_formation", False) else "no"
        name = str(row.get("PNAME", "Unknown"))[:38]
        print(f"{row['rank']:>5} {name:<40} {row.get('PSTATABB','??'):>5} "
              f"{row.get('ISO',''):>7} {co2_kt:>10,.0f} {pipe_km:>9,.0f} "
              f"{well_km:>9,.0f} {geol:>8} {row['viability_score']:>7.1f}")

    # --- Summary by state ---
    state_summary = df.groupby("PSTATABB").agg(
        plant_count=("viability_score", "size"),
        avg_score=("viability_score", "mean"),
        top_score=("viability_score", "max"),
        total_co2_mt=("PLCO2AN", lambda x: x.sum() / 1e6),
    ).sort_values("avg_score", ascending=False).round(1)

    print(f"\n\nSUMMARY BY STATE (top 15):")
    print(f"{'State':>6} {'Plants':>7} {'Avg Score':>10} {'Top Score':>10} "
          f"{'CO2 (MT)':>10}")
    print("-" * 50)
    for state, row in state_summary.head(15).iterrows():
        print(f"{state:>6} {row['plant_count']:>7.0f} {row['avg_score']:>10.1f} "
              f"{row['top_score']:>10.1f} {row['total_co2_mt']:>10.1f}")

    # --- Summary by ISO ---
    iso_summary = df.groupby("ISO").agg(
        plant_count=("viability_score", "size"),
        avg_score=("viability_score", "mean"),
        top_score=("viability_score", "max"),
        total_co2_mt=("PLCO2AN", lambda x: x.sum() / 1e6),
    ).sort_values("avg_score", ascending=False).round(1)

    print(f"\n\nSUMMARY BY ISO:")
    print(f"{'ISO':>8} {'Plants':>7} {'Avg Score':>10} {'Top Score':>10} "
          f"{'CO2 (MT)':>10}")
    print("-" * 50)
    for iso, row in iso_summary.iterrows():
        print(f"{iso:>8} {row['plant_count']:>7.0f} {row['avg_score']:>10.1f} "
              f"{row['top_score']:>10.1f} {row['total_co2_mt']:>10.1f}")

    # --- Save CSVs ---
    # Drop geometry column for CSV output
    csv_cols = [c for c in df.columns if c != "geometry"]
    out_ranking = os.path.join(OUTPUT_DIR, "ccus_viability_ranking.csv")
    df[csv_cols].to_csv(out_ranking, index=False)
    print(f"\nSaved: {out_ranking} ({len(df):,} plants)")

    out_state = os.path.join(OUTPUT_DIR, "ccus_summary_by_state.csv")
    state_summary.to_csv(out_state)
    print(f"Saved: {out_state}")

    out_iso = os.path.join(OUTPUT_DIR, "ccus_summary_by_iso.csv")
    iso_summary.to_csv(out_iso)
    print(f"Saved: {out_iso}")

    return df


def plot_viability_map(plants_df, pipelines_gdf, wells_gdf, formations_gdf):
    """Generate a CONUS map colored by viability score."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib import cm
    except ImportError:
        print("\nmatplotlib not available — skipping map generation")
        return

    print("\nGenerating viability map...")

    fig, ax = plt.subplots(1, 1, figsize=(18, 11))

    # Reproject everything to EPSG:5070 for plotting
    plants_5070 = gpd.GeoDataFrame(
        plants_df,
        geometry=[Point(xy) for xy in zip(plants_df["LON"], plants_df["LAT"])],
        crs="EPSG:4326"
    ).to_crs(TARGET_CRS)

    # Plot formations (background)
    formations_5070 = formations_gdf.to_crs(TARGET_CRS)
    formations_5070.plot(ax=ax, color="#E8F5E9", edgecolor="#4CAF50",
                         linewidth=0.8, alpha=0.5, label="Storage Formations")

    # Plot pipelines
    pipelines_5070 = pipelines_gdf.to_crs(TARGET_CRS)
    pipelines_5070.plot(ax=ax, color="#64748B", linewidth=1.5,
                        alpha=0.7, label="CO₂ Pipelines")

    # Plot wells
    wells_5070 = wells_gdf.to_crs(TARGET_CRS)
    wells_5070.plot(ax=ax, color="#E91E63", markersize=60, marker="^",
                    edgecolor="white", linewidth=0.5, zorder=5,
                    label="Class VI Wells")

    # Plot plants colored by score
    norm = Normalize(vmin=0, vmax=100)
    cmap = cm.get_cmap("RdYlGn")
    plants_5070.plot(ax=ax, column="viability_score", cmap=cmap, norm=norm,
                     markersize=8, alpha=0.7, edgecolor="none", zorder=4)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("CCUS Viability Score (0-100)", fontsize=12)

    # Style
    ax.set_title("US Fossil Power Plant CCUS Retrofit Viability",
                 fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_axis_off()

    # Tight bounds to CONUS
    ax.set_xlim(-2.5e6, 2.5e6)
    ax.set_ylim(0.2e6, 3.3e6)

    out_map = os.path.join(OUTPUT_DIR, "ccus_viability_map.png")
    fig.savefig(out_map, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_map}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("CCS PROXIMITY ANALYSIS")
    print("Spatial viability scoring for CCUS retrofitting of US fossil power plants")
    print("=" * 80)

    # ── Step 1: Ingest all datasets ──
    plants_gdf = fetch_egrid_plants()
    pipelines_gdf = fetch_co2_pipelines()
    formations_gdf = fetch_saline_formations()
    wells_gdf = fetch_class_vi_wells()

    # ── Step 2: Harmonize CRS ──
    plants_gdf, pipelines_gdf, formations_gdf, wells_gdf = harmonize_crs(
        plants_gdf, pipelines_gdf, formations_gdf, wells_gdf
    )

    # ── Step 3: Spatial analysis ──
    print("\nRunning spatial analysis...")
    plants_gdf["dist_pipeline_m"] = compute_nearest_distances(
        plants_gdf, pipelines_gdf, "CO2 pipeline"
    )
    plants_gdf["dist_well_m"] = compute_nearest_distances(
        plants_gdf, wells_gdf, "Class VI well"
    )
    plants_gdf["within_formation"] = compute_formation_overlap(
        plants_gdf, formations_gdf
    )

    # Convert distances to km columns for output
    plants_gdf["dist_pipeline_km"] = (plants_gdf["dist_pipeline_m"] / 1000).round(1)
    plants_gdf["dist_well_km"] = (plants_gdf["dist_well_m"] / 1000).round(1)

    # ── Step 4: Viability scoring ──
    plants_df = compute_viability_scores(plants_gdf)

    # ── Step 5: Output ──
    # Convert back to WGS84 for output / plotting
    plants_out = plants_df.to_crs("EPSG:4326")
    pipelines_4326 = pipelines_gdf.to_crs("EPSG:4326") if pipelines_gdf.crs else pipelines_gdf
    wells_4326 = wells_gdf.to_crs("EPSG:4326") if wells_gdf.crs else wells_gdf
    formations_4326 = formations_gdf.to_crs("EPSG:4326") if formations_gdf.crs else formations_gdf

    ranked_df = generate_outputs(plants_out, pipelines_4326, wells_4326, formations_4326)
    plot_viability_map(ranked_df, pipelines_4326, wells_4326, formations_4326)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
