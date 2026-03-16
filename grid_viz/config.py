"""
Grid Visualization — Configuration & Constants

Follows the project's canonical color palette from dashboard/js/chart-colors.js
and dashboard/styles/shared.css.
"""

# === Resource Colors (from chart-colors.js) ===
RESOURCE_COLORS = {
    "solar":       "#F59E0B",
    "wind":        "#22C55E",
    "offshoreWind":"#009688",
    "hydro":       "#0EA5E9",
    "nuclear":     "#6366F1",
    "ccs":         "#64748B",
    "cleanFirm":   "#6366F1",
    "battery":     "#06B6D4",
    "battery4":    "#06B6D4",
    "battery8":    "#0891B2",
    "ldes":        "#E91E63",
    "greenH2":     "#10B981",
    "geothermal":  "#D97706",
    "storage":     "#EF4444",
    "gap":         "#D1D5DB",
    "fossilGas":   "#6B7280",
    "fossilCoal":  "#374151",
    "fossilOil":   "#92400E",
}

# === ISO Colors ===
ISO_COLORS = {
    "CAISO": "#F59E0B",
    "ERCOT": "#22C55E",
    "PJM":   "#0EA5E9",
    "NYISO": "#E91E63",
    "NEISO": "#9C27B0",
    "MISO":  "#F97316",
    "SPP":   "#14B8A6",
}

# === Fuel category → display color mapping for grid animation ===
# Maps eGRID PLFUELCT values to our canonical palette
FUEL_COLORS = {
    "GAS":      RESOURCE_COLORS["fossilGas"],
    "COAL":     RESOURCE_COLORS["fossilCoal"],
    "OIL":      RESOURCE_COLORS["fossilOil"],
    "NUCLEAR":  RESOURCE_COLORS["nuclear"],
    "WIND":     RESOURCE_COLORS["wind"],
    "SOLAR":    RESOURCE_COLORS["solar"],
    "HYDRO":    RESOURCE_COLORS["hydro"],
    "BIOMASS":  "#8B5CF6",   # Purple - distinct from other categories
    "OTHF":     "#9CA3AF",   # Gray-400 - other/unknown fuels
    "OFSL":     "#9CA3AF",   # Gray-400 - other fossil
    "GEOTHERMAL": RESOURCE_COLORS["geothermal"],
    "BATTERY":  RESOURCE_COLORS["battery"],
}

# RGB versions for PyDeck (0-255 scale)
FUEL_COLORS_RGB = {
    fuel: [int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)]
    for fuel, hex_color in FUEL_COLORS.items()
}

# === ERCOT-Specific Constants ===
ERCOT_BA_CODE = "ERCO"
ERCOT_STATE = "TX"

# Map center (approximate center of ERCOT territory)
ERCOT_MAP_CENTER = {"latitude": 31.5, "longitude": -99.0}
ERCOT_MAP_ZOOM = 5.5

# === Scenario Days (key dates for animation) ===
SCENARIO_DAYS = {
    "uri_2021": {
        "name": "Winter Storm Uri",
        "date_range": ("2021-02-10", "2021-02-20"),
        "peak_date": "2021-02-15",
        "description": "Record cold snap caused cascading failures across ERCOT grid. "
                       "Over 30 GW of capacity went offline. Rolling blackouts affected 4.5M customers.",
    },
    "summer_2023_heat": {
        "name": "August 2023 Heat Dome",
        "date_range": ("2023-08-20", "2023-08-25"),
        "peak_date": "2023-08-22",
        "description": "Extreme heat drove ERCOT demand to near-record levels. "
                       "Low wind output during peak afternoon hours stressed reserves.",
    },
    "normal_spring": {
        "name": "Normal Spring Day",
        "date_range": ("2023-04-15", "2023-04-16"),
        "peak_date": "2023-04-15",
        "description": "Typical spring day with moderate demand, strong wind/solar, "
                       "comfortable reserves. Shows the grid operating as designed.",
    },
    "battery_2025": {
        "name": "2025 Solar + Battery Peak",
        "date_range": ("2025-03-08", "2025-03-09"),
        "peak_date": "2025-03-09",
        "description": "Peak solar output with record battery dispatch. 25+ GW solar, "
                       "batteries charge midday and discharge into evening peak. "
                       "The grid of the future, today.",
    },
}

# === PyDeck Visual Settings ===
PYDECK_SETTINGS = {
    "map_style": "mapbox://styles/mapbox/dark-v11",
    "pitch": 45,
    "bearing": 0,
    # Plant bubble scaling
    "min_radius_m": 500,        # Minimum bubble radius in meters (offline plant)
    "max_radius_m": 15000,      # Maximum bubble radius for largest generators
    "radius_scale_power": 0.5,  # Square-root scaling for visual balance
    # Glow effect layers
    "glow_layers": 3,           # Number of concentric transparent layers for glow
    "glow_opacity_base": 180,   # Inner layer opacity (0-255)
    "glow_opacity_step": 60,    # Opacity decrease per layer
    "glow_radius_multiplier": 1.5,  # Each glow layer is 1.5x the previous
    # Column (3D bar) settings
    "column_radius_m": 3000,
    "column_elevation_scale": 50,  # MW → meters multiplier
    # Animation
    "frame_interval_ms": 500,   # ms between frames in auto-play
    "transition_ms": 400,       # Smooth transition duration
}

# === CAMD API Settings ===
CAMD_API_BASE = "https://api.epa.gov/easey"
CAMD_EMISSIONS_ENDPOINT = f"{CAMD_API_BASE}/emissions/apportioned/hourly"
# No API key required for public endpoints

# === Data Paths ===
DATA_DIR = "grid_viz/data"
EGRID_DIR = f"{DATA_DIR}/egrid"
CEMS_DIR = f"{DATA_DIR}/cems"
ERCOT_DIR = f"{DATA_DIR}/ercot"
GEO_DIR = f"{DATA_DIR}/geo"
FRAMES_DIR = f"{DATA_DIR}/frames"
