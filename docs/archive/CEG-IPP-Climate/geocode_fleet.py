#!/usr/bin/env python3
"""
Cross-reference fleet-rosetta.csv against eGRID 2023 PLNT23 data
to add lat/long coordinates and key eGRID fields to each facility.

Matching strategy (in priority order):
1. Direct ORIS code match: CAMPD Facility ID → eGRID ORISPL
2. Fuzzy name + state match: SP Name / CAMPD Name → eGRID PNAME + PSTATABB
3. Manual overrides for known mismatches

Outputs:
- fleet_geocoded.csv — full fleet with lat/long + eGRID fields appended
- match_report.txt — summary of match quality
"""

import csv
import os
import sys
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EGRID_PATH = os.path.join(REPO_ROOT, "data", "egrid2023_data_rev2 2.xlsx")
FLEET_PATH = os.path.join(SCRIPT_DIR, "fleet-rosetta.csv")
OUT_CSV = os.path.join(SCRIPT_DIR, "fleet_geocoded.csv")
REPORT_PATH = os.path.join(SCRIPT_DIR, "match_report.txt")

# ---------------------------------------------------------------------------
# Manual overrides: SP Name → eGRID ORISPL for plants that can't fuzzy-match
# These were identified from the first-pass match report.
# ---------------------------------------------------------------------------
MANUAL_ORIS = {
    # CEG plants — verified against eGRID 2023 PLNT23
    "Muddy Run Pumped Storage Facility": 3164,
    "LaSalle County Generating Station": 6026,  # eGRID: LaSalle Generating Station
    "Sendero Wind Energy Project": 59654,
    "Salem": 2410,  # Salem Generating Station, NJ
    "Calvert Cliffs": 6011,
    "Harvest I Wind Project": 56635,  # eGRID: Harvest, MI
    "Harvest II Project": 57888,  # eGRID: Harvest 2, MI
    "Calistoga Geothermal": 50066,  # eGRID: Calistoga Power Plant
    "Thumb Wind Park (Michigan Wind 1)": 56416,
    "Wildcat Wind Project (Lovington)": 57887,  # eGRID: Wildcat Wind, NM
    "Whitetail Wind Energy Project": 58021,  # eGRID: Whitetail, TX
    "Beebe Community Wind Farm": 58020,  # eGRID: Beebe 1A, MI
    "Oregon Trail (Energy Vision) (Echo I)": 56971,  # eGRID: Oregon Trail Windfarm LLC, OR
    "Fleshman Solar Project (Kost Road)": 57671,  # eGRID: SMUD at Fleshman, CA
    "Denver International Airport III - Solar": 59462,  # eGRID: Denver Intl Airport IV Solar
    "Vineland Solar One (Landis Sewage)": 57081,  # eGRID: Calpine Vineland Solar LLC
    "Fairless Hills Steam Generating Station": 55298,  # eGRID: Fairless Energy Center
    "Pennsbury Generating Station": 7690,  # eGRID: Pennsbury, PA
    "Newark Power Plant": 58079,  # eGRID: Newark Energy Center, NJ
    "Kennedy International Airport Cogen": None,  # Not in eGRID 2023 — use lat/lon override
    "Constellation New Energy Solar Farm": None,  # Not found in eGRID
}

# Direct lat/lon overrides for plants not in eGRID (international, retired, too new)
MANUAL_LATLON = {
    # Looked up from public sources / EIA data
    "Northern Prairie Power Project (Grande Prairie)": (55.1707, -118.7947),  # Alberta
    "Greenfield Energy Centre": (42.8561, -80.3792),  # Ontario
    "Valladolid III": (20.6896, -89.0939),  # Yucatán
    "Crane": (40.3105, -76.5303),  # PA — Crane Clean Energy Center (planned)
    "Multiple": (39.2904, -76.6122),  # MD Solar aggregate — Baltimore area
    "Menifee Power Bank Storage Project (Nova Power)": (33.7283, -117.1715),  # Menifee, CA
    "Kennedy International Airport Cogen": (40.6413, -73.7781),  # JFK Airport, NY
    "Constellation New Energy Solar Farm": (39.6568, -75.0267),  # NJ
    "Hot Springs Wind (Mountain Home)": (43.133, -115.667),  # Mountain Home, ID
    "Bear Canyon Energy Storage Project": (34.1758, -118.4250),  # CA
    "West Ford Flat Energy Storage": (38.7979, -122.7434),  # CA (near Geysers)
    "Lake View Geothermal (Sonoma)": (38.7575, -122.7475),  # Sonoma County, CA
    "Johanna Battery Storage Center (Santa Ana)": (33.7455, -117.8677),  # Santa Ana, CA
    "Santa Ana Battery Storage 2": (33.7455, -117.8677),  # Santa Ana, CA
    # Retired plants with known locations
    "Cromby 1": (40.1378, -75.5175),  # Phoenixville, PA
    "Cromby 2": (40.1378, -75.5175),
    "Cromby IC": (40.1378, -75.5175),
    "Gould Street": (39.2631, -76.5810),  # Baltimore, MD
    "Inland Empire Energy Center": (33.9625, -117.3475),  # Riverside Co, CA
    "Missouri Avenue": (39.4881, -75.0294),  # Vineland, NJ
    "Riverside (MD)": (39.2083, -76.5272),  # MD
    "Riverside Jet": (39.2083, -76.5272),
    "Westport Jet": (39.2633, -76.5731),  # MD
    "Deepwater CT (NJ)": (39.6917, -75.4833),  # NJ
    "Deepwater ST (NJ) 1": (39.6917, -75.4833),
    "Deepwater ST (NJ) 6": (39.6917, -75.4833),
    "Auburndale Energy Center CT": (28.0650, -81.7886),  # FL
    "Cedar": (39.4500, -75.0333),  # NJ
    "Clear Lake": (29.5708, -95.1103),  # TX
    "Central Wayne Energy Recovery": (42.2808, -83.2458),  # Wayne, MI
    "Edgar": (42.2608, -71.1972),  # MA
    "Madison Street": (39.7478, -75.5472),  # Wilmington, DE
    "Middle": (39.4500, -75.0333),  # NJ
    "Notch Cliff": (39.3611, -76.6378),  # Baltimore, MD
    "O'Brien California Cogen Ltd.": (37.7749, -122.4194),  # San Francisco area, CA
    "Pryor": (36.3086, -95.3172),  # Pryor, OK
    "Pryor ST": (36.3086, -95.3172),
    "PWD Northwest Facility": (40.0178, -75.2250),  # Philadelphia, PA
    "PWD Southwest Facility": (39.9239, -75.2258),  # Philadelphia, PA
    "Vineland Cogeneration": (39.4864, -75.0258),  # Vineland, NJ
    "Watsonville Cogeneration": (36.9103, -121.7569),  # Watsonville, CA
    "South Texas Project": (28.7953, -96.0486),  # Matagorda Co, TX
    "R.E. Ginna/Ontario Sta. 13": (43.2774, -77.3097),  # Ontario, NY
    "Antelope Valley Solar Ranch One": (34.8178, -118.3050),  # Lancaster, CA
    "Fair Wind Generating Facility": (39.6283, -79.0492),  # Garrett Co, MD
}

# Plants that are outside eGRID coverage (now handled by MANUAL_LATLON above)
KNOWN_NO_MATCH = set()  # All formerly-unmatched plants now have manual lat/lon overrides


def load_egrid_plants():
    """Load eGRID PLNT23 sheet into a list of dicts with key fields."""
    try:
        import openpyxl
    except ImportError:
        print("Installing openpyxl...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "-q"])
        import openpyxl

    wb = openpyxl.load_workbook(EGRID_PATH, read_only=True, data_only=True)
    ws = wb["PLNT23"]

    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # Row 0 = long headers, Row 1 = short codes, Row 2+ = data
    short_codes = [str(c) if c else "" for c in rows[1]]

    # Columns we care about
    keep = [
        "ORISPL", "PNAME", "PSTATABB", "LAT", "LON", "ISORTO",
        "NAMEPCAP", "BACODE", "SUBRGN", "CNTYNAME", "FIPSCNTY",
        "PLPRMFL", "PLFUELCT", "CAPFAC",
        "PLNGENAN",  # annual net gen MWh
        "PLCO2AN",   # annual CO2 tons
        "PLCO2RTA",  # CO2 output emission rate lb/MWh
        "PLHTRT",    # heat rate Btu/kWh
    ]
    col_idx = {}
    for col_name in keep:
        if col_name in short_codes:
            col_idx[col_name] = short_codes.index(col_name)

    plants = []
    for row in rows[2:]:  # skip header rows
        if row[col_idx["ORISPL"]] is None:
            continue
        rec = {}
        for col_name, idx in col_idx.items():
            rec[col_name] = row[idx]
        plants.append(rec)

    return plants


def normalize_name(name):
    """Normalize a plant name for fuzzy matching."""
    if not name:
        return ""
    s = str(name).lower().strip()
    # Remove common suffixes
    for suffix in [
        " generating station", " generation station", " energy center",
        " power plant", " power station", " wind farm", " wind project",
        " solar ranch", " solar farm", " cogeneration", " cogen",
        " energy", " center", " station", " facility", " project",
        " plant", " llc", " lp", " inc", " corp",
    ]:
        s = s.replace(suffix, "")
    # Remove punctuation
    s = "".join(c for c in s if c.isalnum() or c == " ")
    return s.strip()


def fuzzy_score(a, b):
    """Return similarity ratio between two normalized names."""
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def match_fleet_to_egrid(fleet_rows, egrid_plants):
    """
    Match each fleet row to an eGRID plant record.
    Returns list of (fleet_row, egrid_record_or_None, match_method).
    """
    # Build ORIS lookup
    oris_lookup = {}
    for p in egrid_plants:
        oris = p["ORISPL"]
        if oris is not None:
            oris_lookup[int(oris)] = p

    # Build state → plants index for fuzzy matching
    state_plants = {}
    for p in egrid_plants:
        st = str(p.get("PSTATABB", "")).upper()
        state_plants.setdefault(st, []).append(p)

    results = []
    for row in fleet_rows:
        campd_id = row.get("CAMPD Facility ID", "").strip()
        state = row.get("State", "").strip().upper()
        sp_name = row.get("SP Name", "")
        campd_name = row.get("CAMPD Name", "")

        matched = None
        method = "unmatched"

        # Strategy 0: Manual lat/lon override (international, retired, not in eGRID)
        if sp_name in MANUAL_LATLON:
            lat, lon = MANUAL_LATLON[sp_name]
            fake_rec = {
                "ORISPL": "", "PNAME": sp_name, "LAT": lat, "LON": lon,
                "CNTYNAME": "", "NAMEPCAP": "", "CAPFAC": "",
                "PLNGENAN": "", "PLCO2AN": "", "PLCO2RTA": "",
                "PLHTRT": "", "PLPRMFL": "", "PLFUELCT": "",
                "BACODE": "", "SUBRGN": "", "ISORTO": "",
            }
            matched = fake_rec
            method = "manual_latlon"
            results.append((row, matched, method))
            continue

        # Strategy 1: Manual ORIS override
        if sp_name in MANUAL_ORIS:
            oris_int = MANUAL_ORIS[sp_name]
            if oris_int is not None and oris_int in oris_lookup:
                matched = oris_lookup[oris_int]
                method = "manual_ORIS"

        # Strategy 2: Direct ORIS match from CAMPD Facility ID
        if matched is None and campd_id and campd_id != "N/A":
            try:
                oris_int = int(campd_id)
                if oris_int in oris_lookup:
                    matched = oris_lookup[oris_int]
                    method = "ORIS_exact"
            except (ValueError, TypeError):
                pass

        # Strategy 3: Fuzzy name match within same state (higher threshold)
        if matched is None and state:
            candidates = state_plants.get(state, [])
            best_score = 0
            best_plant = None

            for name_to_try in [sp_name, campd_name]:
                if not name_to_try or name_to_try == "N/A":
                    continue
                for p in candidates:
                    score = fuzzy_score(name_to_try, p["PNAME"])
                    if score > best_score:
                        best_score = score
                        best_plant = p

            if best_score >= 0.70:
                matched = best_plant
                method = f"fuzzy_{best_score:.2f}"

        results.append((row, matched, method))

    return results


def main():
    print("Loading eGRID PLNT23 data...")
    egrid_plants = load_egrid_plants()
    print(f"  Loaded {len(egrid_plants)} eGRID plant records")

    print("Loading fleet-rosetta.csv...")
    with open(FLEET_PATH, "r") as f:
        reader = csv.DictReader(f)
        fleet_fieldnames = reader.fieldnames
        fleet_rows = list(reader)
    print(f"  Loaded {len(fleet_rows)} fleet rows")

    print("Matching fleet to eGRID...")
    results = match_fleet_to_egrid(fleet_rows, egrid_plants)

    # eGRID columns to append
    egrid_append_cols = [
        "eGRID_ORISPL", "eGRID_PNAME", "eGRID_LAT", "eGRID_LON",
        "eGRID_COUNTY", "eGRID_NAMEPCAP_MW", "eGRID_CAPFAC",
        "eGRID_ANNUAL_GEN_MWh", "eGRID_CO2_TONS", "eGRID_CO2_RATE_LB_MWh",
        "eGRID_HEAT_RATE", "eGRID_PRIMARY_FUEL", "eGRID_FUEL_CATEGORY",
        "eGRID_BA_CODE", "eGRID_SUBREGION", "eGRID_ISO",
        "MATCH_METHOD",
    ]

    out_fieldnames = list(fleet_fieldnames) + egrid_append_cols

    # Write output CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()

        for fleet_row, egrid_rec, method in results:
            out = dict(fleet_row)
            if egrid_rec:
                out["eGRID_ORISPL"] = egrid_rec.get("ORISPL", "")
                out["eGRID_PNAME"] = egrid_rec.get("PNAME", "")
                out["eGRID_LAT"] = egrid_rec.get("LAT", "")
                out["eGRID_LON"] = egrid_rec.get("LON", "")
                out["eGRID_COUNTY"] = egrid_rec.get("CNTYNAME", "")
                out["eGRID_NAMEPCAP_MW"] = egrid_rec.get("NAMEPCAP", "")
                out["eGRID_CAPFAC"] = egrid_rec.get("CAPFAC", "")
                out["eGRID_ANNUAL_GEN_MWh"] = egrid_rec.get("PLNGENAN", "")
                out["eGRID_CO2_TONS"] = egrid_rec.get("PLCO2AN", "")
                out["eGRID_CO2_RATE_LB_MWh"] = egrid_rec.get("PLCO2RTA", "")
                out["eGRID_HEAT_RATE"] = egrid_rec.get("PLHTRT", "")
                out["eGRID_PRIMARY_FUEL"] = egrid_rec.get("PLPRMFL", "")
                out["eGRID_FUEL_CATEGORY"] = egrid_rec.get("PLFUELCT", "")
                out["eGRID_BA_CODE"] = egrid_rec.get("BACODE", "")
                out["eGRID_SUBREGION"] = egrid_rec.get("SUBRGN", "")
                out["eGRID_ISO"] = egrid_rec.get("ISORTO", "")
            else:
                for col in egrid_append_cols[:-1]:
                    out[col] = ""
            out["MATCH_METHOD"] = method
            writer.writerow(out)

    # Generate match report
    total = len(results)
    manual_oris = sum(1 for _, _, m in results if m == "manual_ORIS")
    manual_ll = sum(1 for _, _, m in results if m == "manual_latlon")
    oris_exact = sum(1 for _, _, m in results if m == "ORIS_exact")
    fuzzy = sum(1 for _, _, m in results if m.startswith("fuzzy"))
    no_coverage = sum(1 for _, _, m in results if m == "no_eGRID_coverage")
    unmatched = sum(1 for _, _, m in results if m == "unmatched")
    with_latlon = sum(1 for _, e, _ in results if e and e.get("LAT"))

    report_lines = [
        "=" * 60,
        "FLEET-ROSETTA → eGRID GEOCODING MATCH REPORT",
        "=" * 60,
        f"Total fleet rows:        {total}",
        f"Manual ORIS overrides:   {manual_oris} ({manual_oris/total*100:.1f}%)",
        f"Manual lat/lon:          {manual_ll} ({manual_ll/total*100:.1f}%)",
        f"ORIS exact matches:      {oris_exact} ({oris_exact/total*100:.1f}%)",
        f"Fuzzy name matches:      {fuzzy} ({fuzzy/total*100:.1f}%)",
        f"No eGRID coverage:       {no_coverage} ({no_coverage/total*100:.1f}%)",
        f"Unmatched:               {unmatched} ({unmatched/total*100:.1f}%)",
        f"Rows with lat/lon:       {with_latlon} ({with_latlon/total*100:.1f}%)",
        "",
        "-" * 60,
        "UNMATCHED FACILITIES:",
        "-" * 60,
    ]

    for fleet_row, egrid_rec, method in results:
        if method == "unmatched":
            report_lines.append(
                f"  ID={fleet_row.get('Facility ID', '?'):>4}  "
                f"{fleet_row.get('SP Name', 'N/A'):<45}  "
                f"State={fleet_row.get('State', '?'):<4}  "
                f"CAMPD_ID={fleet_row.get('CAMPD Facility ID', 'N/A')}"
            )

    report_lines.extend([
        "",
        "-" * 60,
        "FUZZY MATCHES (verify these):",
        "-" * 60,
    ])

    for fleet_row, egrid_rec, method in results:
        if method.startswith("fuzzy"):
            report_lines.append(
                f"  Fleet: {fleet_row.get('SP Name', 'N/A'):<40} → "
                f"eGRID: {egrid_rec['PNAME']:<40}  "
                f"({method})"
            )

    report = "\n".join(report_lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nResults written to: {OUT_CSV}")
    print(f"Match report:       {REPORT_PATH}")
    print()
    print(report)


if __name__ == "__main__":
    main()
