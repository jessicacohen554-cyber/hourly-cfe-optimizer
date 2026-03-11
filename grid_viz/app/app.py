#!/usr/bin/env python3
"""
ERCOT Grid Animation — Dash Application

Synchronized visualization:
1. PyDeck map with animated plant bubbles (radius = MW output, color = fuel type)
2. Plotly stacked area supply stack chart
3. Time slider with auto-play controls
4. Reserve margin gauge with danger state
5. Scrollytelling annotation sidebar

Run: python grid_viz/app/app.py
Visit: http://localhost:8050
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import (
    SCENARIO_DAYS, FUEL_COLORS, FUEL_COLORS_RGB,
    ERCOT_MAP_CENTER, ERCOT_MAP_ZOOM, PYDECK_SETTINGS, FRAMES_DIR,
)

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_scenario_data(scenario_id):
    frames = pd.read_parquet(os.path.join(FRAMES_DIR, f"{scenario_id}_frames.parquet"))
    stack = pd.read_parquet(os.path.join(FRAMES_DIR, f"{scenario_id}_stack.parquet"))
    demand = pd.read_parquet(os.path.join(FRAMES_DIR, f"{scenario_id}_demand.parquet"))
    return frames, stack, demand

def load_annotations(scenario_id):
    path = os.path.join("grid_viz/stories", f"{scenario_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

SCENARIO_DATA = {}
for sid in SCENARIO_DAYS:
    try:
        SCENARIO_DATA[sid] = load_scenario_data(sid)
    except FileNotFoundError:
        print(f"Warning: No frame data for {sid}")

# ── Color/Label Constants ────────────────────────────────────────────────────

STACK_ORDER = ["NUCLEAR", "COAL", "GAS", "OIL", "HYDRO", "WIND", "SOLAR", "BIOMASS", "OTHF", "OFSL"]

FUEL_PLOTLY_COLORS = {
    "NUCLEAR": FUEL_COLORS["NUCLEAR"],
    "COAL": FUEL_COLORS["COAL"],
    "GAS": FUEL_COLORS["GAS"],
    "OIL": FUEL_COLORS["OIL"],
    "HYDRO": FUEL_COLORS["HYDRO"],
    "WIND": FUEL_COLORS["WIND"],
    "SOLAR": FUEL_COLORS["SOLAR"],
    "BIOMASS": "#8B5CF6",
    "OTHF": "#9CA3AF",
    "OFSL": "#9CA3AF",
}

FUEL_LABELS = {
    "NUCLEAR": "Nuclear", "COAL": "Coal", "GAS": "Natural Gas", "OIL": "Oil",
    "HYDRO": "Hydro", "WIND": "Wind", "SOLAR": "Solar", "BIOMASS": "Biomass",
    "OTHF": "Other", "OFSL": "Other Fossil",
}

# ── PyDeck Map ───────────────────────────────────────────────────────────────

def build_map(frame_data, hour_idx):
    hour_data = frame_data[frame_data["hour_index"] == hour_idx].copy()
    if len(hour_data) == 0:
        hour_data = frame_data[frame_data["hour_index"] == 0].copy()

    max_mw = max(frame_data["output_mw"].max(), 1)
    min_r = PYDECK_SETTINGS["min_radius_m"]
    max_r = PYDECK_SETTINGS["max_radius_m"]
    power = PYDECK_SETTINGS["radius_scale_power"]

    hour_data["radius"] = hour_data["output_mw"].apply(
        lambda mw: min_r + (max_r - min_r) * (mw / max_mw) ** power if mw > 0 else min_r * 0.3
    )
    hour_data["opacity"] = hour_data["output_mw"].apply(lambda mw: 200 if mw > 0 else 40)

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=hour_data,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color=["color_r", "color_g", "color_b", "opacity"],
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 80],
        radius_min_pixels=2,
        radius_max_pixels=50,
    )

    # Glow layer for active plants
    active = hour_data[hour_data["output_mw"] > 0].copy()
    layers = []

    if len(active) > 0:
        # Subtle heatmap underlay
        heat_data = active[["lat", "lon", "output_mw"]].copy()
        heat_data = heat_data.rename(columns={"output_mw": "weight"})
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=heat_data,
            get_position=["lon", "lat"],
            get_weight="weight",
            opacity=0.15,
            threshold=0.1,
            radius_pixels=40,
        ))

        # Glow (soft glassmorphic, not neon)
        active["glow_radius"] = active["radius"] * 1.5
        active["glow_opacity"] = 50
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=active,
            get_position=["lon", "lat"],
            get_radius="glow_radius",
            get_fill_color=["color_r", "color_g", "color_b", "glow_opacity"],
            pickable=False,
            radius_min_pixels=3,
            radius_max_pixels=80,
        ))

    layers.append(scatter_layer)

    view_state = pdk.ViewState(
        latitude=ERCOT_MAP_CENTER["latitude"],
        longitude=ERCOT_MAP_CENTER["longitude"],
        zoom=ERCOT_MAP_ZOOM,
        pitch=PYDECK_SETTINGS["pitch"],
        bearing=PYDECK_SETTINGS["bearing"],
    )

    tooltip = {
        "html": "<div style='font-family:DM Sans,sans-serif;padding:8px'>"
                "<b>{plant_name}</b><br/>"
                "<span style='color:#9CA3AF'>Fuel:</span> {fuel_label}<br/>"
                "<span style='color:#9CA3AF'>Output:</span> {output_mw} MW<br/>"
                "<span style='color:#9CA3AF'>Capacity:</span> {capacity_mw} MW<br/>"
                "<span style='color:#9CA3AF'>CO₂:</span> {co2_tons_hr} tons/hr</div>",
        "style": {
            "backgroundColor": "#1B2A4A",
            "color": "white",
            "border": "1px solid rgba(255,255,255,0.1)",
            "borderRadius": "8px",
        },
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=PYDECK_SETTINGS["map_style"],
        tooltip=tooltip,
    )

# ── Supply Stack Chart ───────────────────────────────────────────────────────

def build_supply_stack(stack_data, demand_data, current_hour):
    fig = go.Figure()

    pivot = stack_data.pivot_table(
        index=["hour_index", "datetime"], columns="fuel",
        values="total_mw", fill_value=0,
    ).reset_index()

    x = pivot["datetime"].values

    for fuel in STACK_ORDER:
        if fuel in pivot.columns:
            y = pivot[fuel].values
            if y.sum() > 0:
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines",
                    name=FUEL_LABELS.get(fuel, fuel),
                    line=dict(width=0),
                    fillcolor=FUEL_PLOTLY_COLORS.get(fuel, "#9CA3AF"),
                    fill="tonexty" if len(fig.data) > 0 else "tozeroy",
                    stackgroup="supply",
                    hovertemplate=f"{FUEL_LABELS.get(fuel, fuel)}: %{{y:,.0f}} MW<extra></extra>",
                ))

    if len(demand_data) > 0:
        fig.add_trace(go.Scatter(
            x=demand_data["datetime"].values,
            y=demand_data["demand_mw"].values,
            mode="lines", name="Demand",
            line=dict(color="white", width=2, dash="dot"),
            hovertemplate="Demand: %{y:,.0f} MW<extra></extra>",
        ))

    if current_hour < len(x):
        fig.add_vline(x=x[current_hour], line=dict(color="rgba(255,255,255,0.6)", width=2, dash="dash"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(17,24,39,0.95)",
        plot_bgcolor="rgba(17,24,39,0.95)",
        font=dict(family="DM Sans, sans-serif", color="#E5E7EB"),
        margin=dict(l=60, r=20, t=10, b=40),
        height=250,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=11)),
        xaxis=dict(showgrid=False, tickangle=-45, dtick=24 if len(x) > 48 else 6, tickfont=dict(size=10)),
        yaxis=dict(title="MW", showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickformat=","),
        hovermode="x unified",
    )
    return fig

# ── Reserve Stats ────────────────────────────────────────────────────────────

def compute_reserve_stats(stack_data, demand_data, hour_idx):
    hour_supply = stack_data[stack_data["hour_index"] == hour_idx]["total_mw"].sum()
    demand_row = demand_data[demand_data["hour_index"] == hour_idx]
    hour_demand = demand_row["demand_mw"].values[0] if len(demand_row) > 0 else 0
    renewable = demand_row["renewable_mw"].values[0] if len(demand_row) > 0 and "renewable_mw" in demand_row.columns else 0
    fossil = demand_row["fossil_mw"].values[0] if len(demand_row) > 0 and "fossil_mw" in demand_row.columns else 0

    reserve_mw = hour_supply - hour_demand
    reserve_pct = (reserve_mw / hour_demand * 100) if hour_demand > 0 else 0

    if reserve_pct < 0:
        status, color = "EMERGENCY", "#EF4444"
    elif reserve_pct < 3:
        status, color = "CRITICAL", "#F97316"
    elif reserve_pct < 6:
        status, color = "TIGHT", "#F59E0B"
    else:
        status, color = "NORMAL", "#22C55E"

    return {
        "demand_mw": hour_demand, "supply_mw": hour_supply,
        "reserve_mw": reserve_mw, "reserve_pct": reserve_pct,
        "renewable_mw": renewable, "fossil_mw": fossil,
        "status": status, "status_color": color,
    }

# ── Dash App ─────────────────────────────────────────────────────────────────

app = Dash(__name__, title="ERCOT Grid Animation",
           assets_folder=os.path.join(os.path.dirname(__file__), "assets"))

DEFAULT_SCENARIO = "uri_2021"
default_data = SCENARIO_DATA.get(DEFAULT_SCENARIO)
max_hours = int(default_data[0]["hour_index"].max()) if default_data else 23

def _stat_card(label, value, unit, color):
    return html.Div(
        style={"backgroundColor": "rgba(255,255,255,0.03)", "borderRadius": "8px",
               "padding": "10px 12px", "border": "1px solid rgba(255,255,255,0.05)"},
        children=[
            html.Div(label, style={"fontSize": "10px", "color": "#9CA3AF",
                                    "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "4px", "marginTop": "4px"},
                     children=[
                         html.Span(value, style={"fontSize": "18px", "fontWeight": "700", "color": color}),
                         html.Span(unit, style={"fontSize": "11px", "color": "#6B7280"}),
                     ]),
        ])

app.layout = html.Div(
    className="grid-viz-app",
    style={"backgroundColor": "#111827", "color": "#E5E7EB",
           "fontFamily": "'Plus Jakarta Sans','DM Sans',sans-serif",
           "minHeight": "100vh"},
    children=[
        # Header
        html.Div(
            style={"background": "linear-gradient(135deg,#0F172A 0%,#1E293B 50%,#0F172A 100%)",
                   "padding": "24px 32px", "borderBottom": "1px solid rgba(255,255,255,0.05)"},
            children=[
                html.H1("ERCOT Grid Animation",
                         style={"fontSize": "28px", "fontWeight": "800", "margin": "0",
                                "background": "linear-gradient(90deg,#22C55E,#0EA5E9)",
                                "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent"}),
                html.Div("Spatiotemporal visualization of power generation across the ERCOT grid",
                         style={"color": "#9CA3AF", "fontSize": "14px", "marginTop": "4px"}),
            ]),

        # Controls
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "16px", "padding": "16px 32px",
                   "backgroundColor": "rgba(30,41,59,0.8)", "borderBottom": "1px solid rgba(255,255,255,0.05)",
                   "flexWrap": "wrap"},
            children=[
                html.Div([
                    html.Label("Scenario", style={"fontSize": "11px", "color": "#9CA3AF", "display": "block", "marginBottom": "4px"}),
                    dcc.Dropdown(id="scenario-dropdown",
                                 options=[{"label": info["name"], "value": sid}
                                          for sid, info in SCENARIO_DAYS.items() if sid in SCENARIO_DATA],
                                 value=DEFAULT_SCENARIO, clearable=False,
                                 style={"width": "220px", "backgroundColor": "#1E293B", "color": "#E5E7EB"}),
                ]),
                html.Button("▶ Play", id="play-btn", n_clicks=0,
                            style={"padding": "8px 20px", "borderRadius": "8px",
                                   "border": "1px solid rgba(34,197,94,0.4)",
                                   "backgroundColor": "rgba(34,197,94,0.1)", "color": "#22C55E",
                                   "cursor": "pointer", "fontWeight": "600", "fontSize": "14px", "marginTop": "16px"}),
                html.Div([
                    html.Label("Speed", style={"fontSize": "11px", "color": "#9CA3AF", "display": "block", "marginBottom": "4px"}),
                    dcc.Dropdown(id="speed-dropdown",
                                 options=[{"label": "0.5x", "value": 1000}, {"label": "1x", "value": 500},
                                          {"label": "2x", "value": 250}, {"label": "4x", "value": 125}],
                                 value=500, clearable=False,
                                 style={"width": "90px", "backgroundColor": "#1E293B"}),
                ]),
                html.Div(id="time-display",
                         style={"fontSize": "20px", "fontWeight": "700",
                                "fontFamily": "'JetBrains Mono',monospace", "color": "#F8FAFC",
                                "marginLeft": "auto", "marginTop": "16px"}),
                html.Div(id="reserve-indicator",
                         style={"padding": "8px 16px", "borderRadius": "8px",
                                "fontSize": "13px", "fontWeight": "600", "marginTop": "16px"}),
            ]),

        # Time slider
        html.Div(style={"padding": "8px 32px 0"}, children=[
            dcc.Slider(id="hour-slider", min=0, max=max_hours, value=0, step=1,
                       marks=None, tooltip={"placement": "bottom", "always_visible": False},
                       updatemode="drag"),
        ]),

        # Map + Sidebar
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 320px", "gap": "0",
                   "height": "calc(100vh - 360px)", "minHeight": "500px"},
            children=[
                # Map
                html.Div(id="map-container",
                         style={"position": "relative", "overflow": "hidden",
                                "borderRight": "1px solid rgba(255,255,255,0.05)"},
                         children=[
                             html.Iframe(id="deck-map", style={"width": "100%", "height": "100%", "border": "none"}),
                             # Legend overlay
                             html.Div(
                                 style={"position": "absolute", "bottom": "16px", "left": "16px",
                                        "backgroundColor": "rgba(17,24,39,0.85)", "backdropFilter": "blur(8px)",
                                        "borderRadius": "12px", "padding": "12px 16px",
                                        "border": "1px solid rgba(255,255,255,0.08)"},
                                 children=[html.Div(
                                     style={"display": "flex", "flexWrap": "wrap", "gap": "8px 16px"},
                                     children=[
                                         html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"},
                                                  children=[
                                                      html.Div(style={"width": "10px", "height": "10px",
                                                                       "borderRadius": "50%",
                                                                       "backgroundColor": FUEL_COLORS.get(f, "#9CA3AF")}),
                                                      html.Span(FUEL_LABELS.get(f, f),
                                                                style={"fontSize": "11px", "color": "#D1D5DB"}),
                                                  ])
                                         for f in ["SOLAR", "WIND", "NUCLEAR", "GAS", "COAL", "HYDRO"]
                                     ])]),
                         ]),

                # Sidebar
                html.Div(
                    style={"backgroundColor": "rgba(15,23,42,0.95)", "padding": "20px", "overflowY": "auto"},
                    children=[
                        html.Div(id="stats-panel"),
                        html.Hr(style={"borderColor": "rgba(255,255,255,0.05)", "margin": "16px 0"}),
                        html.Div(id="annotation-panel", style={"fontSize": "13px", "lineHeight": "1.6"}),
                    ]),
            ]),

        # Supply stack chart
        html.Div(style={"padding": "0 32px 24px"}, children=[
            dcc.Graph(id="supply-stack-chart", config={"displayModeBar": False}, style={"height": "250px"}),
        ]),

        # Auto-play interval
        dcc.Interval(id="auto-play-interval", interval=500, disabled=True),
        dcc.Store(id="playing-state", data=False),
    ])

# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("hour-slider", "max"), Output("hour-slider", "value"),
    Input("scenario-dropdown", "value"))
def update_scenario(scenario_id):
    if scenario_id not in SCENARIO_DATA:
        raise PreventUpdate
    frames, _, _ = SCENARIO_DATA[scenario_id]
    return int(frames["hour_index"].max()), 0

@app.callback(
    Output("playing-state", "data"), Output("play-btn", "children"),
    Output("auto-play-interval", "disabled"),
    Input("play-btn", "n_clicks"), State("playing-state", "data"))
def toggle_play(n_clicks, is_playing):
    if n_clicks == 0:
        raise PreventUpdate
    new_state = not is_playing
    return new_state, "⏸ Pause" if new_state else "▶ Play", not new_state

@app.callback(
    Output("auto-play-interval", "interval"),
    Input("speed-dropdown", "value"))
def update_speed(ms):
    return ms

@app.callback(
    Output("hour-slider", "value", allow_duplicate=True),
    Input("auto-play-interval", "n_intervals"),
    State("hour-slider", "value"), State("hour-slider", "max"),
    prevent_initial_call=True)
def auto_advance(_, current, max_h):
    nxt = current + 1
    return 0 if nxt > max_h else nxt

@app.callback(
    Output("deck-map", "srcDoc"), Output("supply-stack-chart", "figure"),
    Output("time-display", "children"), Output("reserve-indicator", "children"),
    Output("reserve-indicator", "style"), Output("stats-panel", "children"),
    Output("annotation-panel", "children"),
    Input("hour-slider", "value"), Input("scenario-dropdown", "value"))
def update_display(hour_idx, scenario_id):
    if scenario_id not in SCENARIO_DATA:
        raise PreventUpdate

    frames, stack, demand = SCENARIO_DATA[scenario_id]

    # Map
    deck = build_map(frames, hour_idx)
    map_html = deck.to_html(as_string=True)

    # Supply stack
    fig = build_supply_stack(stack, demand, hour_idx)

    # Time display
    hour_data = frames[frames["hour_index"] == hour_idx]
    dt_str = hour_data["datetime"].iloc[0] if len(hour_data) > 0 else "—"

    # Reserve stats
    stats = compute_reserve_stats(stack, demand, hour_idx)
    reserve_text = f"Reserve: {stats['reserve_pct']:+.1f}% ({stats['reserve_mw']:+,.0f} MW) — {stats['status']}"
    reserve_style = {
        "padding": "8px 16px", "borderRadius": "8px", "fontSize": "13px",
        "fontWeight": "600", "marginTop": "16px",
        "backgroundColor": f"{stats['status_color']}20",
        "color": stats["status_color"],
        "border": f"1px solid {stats['status_color']}40",
    }

    # Stats panel
    stats_children = [
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
                 children=[
                     _stat_card("Demand", f"{stats['demand_mw']:,.0f}", "MW", "#F8FAFC"),
                     _stat_card("Generation", f"{stats['supply_mw']:,.0f}", "MW", "#22C55E"),
                     _stat_card("Renewables", f"{stats['renewable_mw']:,.0f}", "MW", "#0EA5E9"),
                     _stat_card("Fossil", f"{stats['fossil_mw']:,.0f}", "MW", "#6B7280"),
                 ]),
    ]

    hour_co2 = 0
    hour_stack = stack[stack["hour_index"] == hour_idx]
    if "total_co2_tons" in hour_stack.columns:
        hour_co2 = hour_stack["total_co2_tons"].sum()
    stats_children.append(html.Div(style={"marginTop": "12px"},
                                   children=[_stat_card("CO₂ Emissions", f"{hour_co2:,.0f}", "tons/hr", "#EF4444")]))

    # Annotations
    annotations = load_annotations(scenario_id)
    scenario_info = SCENARIO_DAYS.get(scenario_id, {})
    ann_children = [
        html.H3(scenario_info.get("name", scenario_id),
                 style={"fontSize": "16px", "fontWeight": "700", "marginBottom": "8px"}),
        html.P(scenario_info.get("description", ""),
               style={"color": "#9CA3AF", "fontSize": "12px", "marginBottom": "12px"}),
    ]
    for ann in annotations:
        active = ann.get("hour_start", -1) <= hour_idx <= ann.get("hour_end", -1)
        ann_children.append(html.Div(
            style={"padding": "10px 12px", "marginBottom": "8px", "borderRadius": "8px",
                   "backgroundColor": "rgba(34,197,94,0.1)" if active else "rgba(255,255,255,0.02)",
                   "borderLeft": f"3px solid {'#22C55E' if active else 'rgba(255,255,255,0.1)'}",
                   "transition": "all 0.3s ease"},
            children=[
                html.Div(ann.get("time_label", ""),
                         style={"fontSize": "11px", "color": "#9CA3AF", "fontWeight": "600"}),
                html.Div(ann.get("text", ""), style={"fontSize": "12px", "marginTop": "4px"}),
            ]))

    return map_html, fig, dt_str, reserve_text, reserve_style, stats_children, ann_children

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    print(f"\n{'='*60}")
    print(f"  ERCOT Grid Animation")
    print(f"  Scenarios: {list(SCENARIO_DATA.keys())}")
    print(f"  http://localhost:{port}")
    print(f"{'='*60}\n")
    app.run(debug=debug, port=port, host="0.0.0.0")
