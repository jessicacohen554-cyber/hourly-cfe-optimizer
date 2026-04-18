"""
gen_section1_worst_hours.py — Generate section1_worst_hours.json

Data lineage:
  - analysis/reliability-tax/data/MANIFEST.json
  - per-run JSONs via data_loader.get_run() + get_pathway_retirements()
  - scripts/pipeline_config.py: PEAK_DEMAND_MW, EXISTING_GAS_CAPACITY_MW,
    RESOURCE_CAPACITY_FACTORS

Pathway: 1 (VRE + batteries only). Endpoint: all 5, but worst-hours are most
meaningful at the highest feasible endpoint (99.9%). Payload includes one endpoint
per ISO (highest feasible endpoint in Pathway 1).

IMPORTANT — Schema v1 Note:
  endpoint_hourly_dispatch is None in all schema_version=1 runs. The 100 worst-hour
  dispatch profiles in this payload are SYNTHETIC and DERIVED from resource mix data
  (stranding_ledger TWh, capacity factors, demand profiles). They are NOT extracted
  from a 8760-hour simulation. They are representative modeled profiles for visualization.
  This is clearly flagged in the meta.

Construction methodology:
  1. Installed GW per resource from stranding_ledger.pathway_twh / (8760h × CF)
     Battery dispatch hours approximated as 1000 h/yr for 4hr, 1200 h/yr for 8hr.
  2. Peak demand = avg_demand_gw × peak_to_avg_ratio (ISO-specific, from PEAK_DEMAND_MW
     and demand_2050_twh).
  3. 100 worst hours constructed as: demand sampled from [peak×0.85, peak×1.0],
     solar = 0 (nighttime), wind = wind_installed_gw × [0.02, 0.12] (low-wind period),
     storage dispatches to fill gap up to rated power, gas takes residual.
  4. Sorted descending by residual load (= demand - wind).
  5. Random seed = 42 for reproducibility.

Payload structure:
  per_iso: dict[iso ->
    endpoint: float (highest feasible P1 endpoint for this ISO)
    demand_2050_twh: float
    avg_demand_gw: float
    peak_demand_gw: float
    installed_gw: dict[resource -> GW]
    gas_remaining_gw: float  (existing - retired at endpoint)
    worst_100_hours: list[100 × {rank, demand_gw, solar_gw, wind_gw,
                                  battery_gw, ldes_gw, clean_firm_gw,
                                  gas_gw, residual_load_gw, total_clean_gw}]
    summary_stats: dict (aggregates over worst 100 hours)
  ]
  meta: ...
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reliability_tax" / "charts"))
from data_loader import (  # noqa: E402
    get_run,
    get_pathway_retirements,
    ISOS,
    ENDPOINTS,
)

OUT_PATH = Path(__file__).parent / "section1_worst_hours.json"

# From pipeline_config.py — RESOURCE_CAPACITY_FACTORS
SOLAR_CF = {
    "CAISO": 0.28, "ERCOT": 0.24, "PJM": 0.17,
    "NYISO": 0.15, "NEISO": 0.15, "MISO": 0.19, "SPP": 0.22,
}
WIND_CF = {
    "CAISO": 0.25, "ERCOT": 0.38, "PJM": 0.30,
    "NYISO": 0.28, "NEISO": 0.30, "MISO": 0.36, "SPP": 0.42,
}
OFFSHORE_WIND_CF = {
    "CAISO": 0.40, "ERCOT": 0.38, "PJM": 0.38,
    "NYISO": 0.40, "NEISO": 0.40, "MISO": 0.38, "SPP": 0.38,
}

# Battery rated power = dispatch_twh / dispatch_hours_per_year
BATTERY4_DISPATCH_H_YR = 1000   # ~250 cycles × 4h
BATTERY8_DISPATCH_H_YR = 1200   # ~150 cycles × 8h
LDES_DISPATCH_H_YR = 800        # ~8 cycles × 100h per year

# Existing gas: from pipeline_config.py
EXISTING_GAS_GW = {
    "CAISO": 37.0, "ERCOT": 55.0, "PJM": 75.0,
    "NYISO": 18.0, "NEISO": 14.0, "MISO": 68.0, "SPP": 32.0,
}

# PEAK_DEMAND_MW from pipeline_config.py (2025 base)
PEAK_DEMAND_MW_2025 = {
    "CAISO": 43860, "ERCOT": 83597, "PJM": 160560,
    "NYISO": 31857, "NEISO": 25898, "MISO": 127125, "SPP": 54368,
}

RANDOM_SEED = 42


def _installed_gw_from_buildout(bo: list[dict], iso: str) -> dict[str, float]:
    """
    Convert annual_buildout cumulative TWh → installed GW using capacity factors.

    Sums all vintage TWh across all years (gives total installed capacity at 2050).
    This is the correct source for VRE-only pathways where stranding_ledger
    only contains resources where the pathway exceeds P3 (sparse/empty for
    low-stress ISOs).
    """
    totals: dict[str, float] = {}
    for yr_row in bo:
        for v in yr_row.get("new_vintages", []):
            r = v["resource"]
            totals[r] = totals.get(r, 0.0) + v.get("twh_per_year", 0.0)

    gw: dict[str, float] = {}
    for r, twh in totals.items():
        if twh <= 0:
            continue
        if r == "solar":
            gw["solar"] = twh / (8.76 * SOLAR_CF.get(iso, 0.20))
        elif r in ("wind", "onshore_wind"):
            gw["wind"] = twh / (8.76 * WIND_CF.get(iso, 0.35))
        elif r == "offshore_wind":
            gw["offshore_wind"] = twh / (8.76 * OFFSHORE_WIND_CF.get(iso, 0.40))
        elif r == "battery4":
            gw["battery4"] = twh * 1000.0 / BATTERY4_DISPATCH_H_YR
        elif r == "battery8":
            gw["battery8"] = twh * 1000.0 / BATTERY8_DISPATCH_H_YR
        elif r == "ldes":
            gw["ldes"] = twh * 1000.0 / LDES_DISPATCH_H_YR
    return gw


def _peak_demand_gw(demand_2050_twh: float, iso: str) -> float:
    """
    Estimate 2050 peak demand GW.
    Scale from 2025 peak using same ratio as avg demand growth.
    peak_2050 = PEAK_2025 × (demand_2050_twh / demand_2025_twh)
    """
    # Approximate 2025 demand from PEAK_DEMAND_MW and ~55% load factor
    peak_2025_gw = PEAK_DEMAND_MW_2025[iso] / 1000.0
    avg_2025_gw = peak_2025_gw * 0.55
    demand_2025_twh = avg_2025_gw * 8.76
    avg_2050_gw = demand_2050_twh / 8.76
    peak_2050_gw = peak_2025_gw * (avg_2050_gw / avg_2025_gw)
    return round(peak_2050_gw, 1)


def _worst_100_hours(
    peak_demand_gw: float,
    installed_gw: dict[str, float],
    gas_remaining_gw: float,
    iso: str,
    rng: random.Random,
) -> list[dict]:
    """
    Generate 100 synthetic worst-hour dispatch profiles with battery SOC constraint.

    Worst hours = high demand + minimum VRE (nighttime, low-wind, winter).
    The 100 hours are modeled as sequences within events (not independent hours).
    Battery state-of-charge depletes over consecutive stressed hours and
    recharges when an event resets. This produces realistic gas dispatch during
    prolonged low-VRE, high-demand events.

    Event model: 5 events × 20 hours each. Each event resets battery SOC to full.
    Battery4: 4-hour duration → after 4h at rated power, depleted (proportional to usage).
    Battery8: 8-hour duration → similar.
    LDES: 100-hour duration → rarely depleted over 20-hour event.
    """
    solar_gw = installed_gw.get("solar", 0.0)
    wind_gw = installed_gw.get("wind", 0.0) + installed_gw.get("offshore_wind", 0.0)
    batt4_gw = installed_gw.get("battery4", 0.0)
    batt8_gw = installed_gw.get("battery8", 0.0)
    ldes_gw = installed_gw.get("ldes", 0.0)

    # Clean firm = 0 in Pathway 1
    clean_firm_gw = 0.0

    # Battery energy capacity (GWh): rated_power × duration
    batt4_energy_gwh = batt4_gw * 4.0
    batt8_energy_gwh = batt8_gw * 8.0
    ldes_energy_gwh = ldes_gw * 100.0

    profiles = []
    # Simulate 5 events × 20 hours (= 100 worst hours total)
    NUM_EVENTS = 5
    HOURS_PER_EVENT = 20

    for _event in range(NUM_EVENTS):
        # SOC reset at start of each event (recharges between events)
        soc_batt4 = batt4_energy_gwh
        soc_batt8 = batt8_energy_gwh
        soc_ldes = ldes_energy_gwh

        for h in range(HOURS_PER_EVENT):
            # Demand: slightly higher at event start, tapering toward end
            event_severity = 1.0 - h * 0.005  # 0.5% decay per hour
            demand = peak_demand_gw * rng.uniform(0.88, 1.00) * event_severity

            # Solar: nighttime/winter → near 0 (slight recovery possible late in event)
            solar_frac = rng.uniform(0.00, 0.02) + max(0, h - 15) * 0.005
            solar = solar_gw * solar_frac

            # Wind: calm low-wind (slight recovery possible late in event)
            wind_frac = rng.uniform(0.02, 0.10) + max(0, h - 12) * 0.003
            wind = wind_gw * wind_frac

            # Residual load after VRE
            residual = max(0.0, demand - solar - wind)

            # Battery4 dispatch (limited by available SOC and rated power)
            batt4_can_dispatch = min(batt4_gw, soc_batt4)  # can't dispatch more than SOC
            batt4_dispatch = min(batt4_can_dispatch, residual)
            soc_batt4 = max(0.0, soc_batt4 - batt4_dispatch)
            after_batt4 = max(0.0, residual - batt4_dispatch)

            # Battery8 dispatch
            batt8_can_dispatch = min(batt8_gw, soc_batt8)
            batt8_dispatch = min(batt8_can_dispatch, after_batt4)
            soc_batt8 = max(0.0, soc_batt8 - batt8_dispatch)
            after_batt8 = max(0.0, after_batt4 - batt8_dispatch)

            # LDES dispatch
            ldes_can_dispatch = min(ldes_gw, soc_ldes)
            ldes_dispatch = min(ldes_can_dispatch, after_batt8)
            soc_ldes = max(0.0, soc_ldes - ldes_dispatch)
            after_ldes = max(0.0, after_batt8 - ldes_dispatch)

            # Gas fills remaining gap
            gas = min(gas_remaining_gw, after_ldes)

            total_storage = batt4_dispatch + batt8_dispatch + ldes_dispatch
            total_clean = solar + wind + total_storage + clean_firm_gw

            profiles.append(
                {
                    "demand_gw": round(demand, 2),
                    "solar_gw": round(solar, 2),
                    "wind_gw": round(wind, 2),
                    "battery_gw": round(batt4_dispatch + batt8_dispatch, 2),
                    "ldes_gw": round(ldes_dispatch, 2),
                    "clean_firm_gw": round(clean_firm_gw, 2),
                    "gas_gw": round(gas, 2),
                    "residual_load_gw": round(residual, 2),
                    "total_clean_gw": round(total_clean, 2),
                    "event_id": _event + 1,
                    "event_hour": h + 1,
                    "soc_batt4_remaining_gwh": round(soc_batt4, 1),
                    "soc_batt8_remaining_gwh": round(soc_batt8, 1),
                }
            )

    # Sort descending by residual load (worst = highest residual)
    profiles.sort(key=lambda x: x["residual_load_gw"], reverse=True)
    for i, p in enumerate(profiles):
        p["rank"] = i + 1

    return profiles


def _summary_stats(profiles: list[dict]) -> dict:
    n = len(profiles)
    if n == 0:
        return {}
    fields = [
        "demand_gw", "solar_gw", "wind_gw", "battery_gw",
        "ldes_gw", "clean_firm_gw", "gas_gw", "residual_load_gw", "total_clean_gw",
    ]
    stats: dict = {}
    for f in fields:
        vals = [p[f] for p in profiles]
        stats[f"avg_{f}"] = round(sum(vals) / n, 2)
        stats[f"max_{f}"] = round(max(vals), 2)
        stats[f"min_{f}"] = round(min(vals), 2)
    # Extra
    stats["avg_gas_cf"] = round(
        stats["avg_gas_gw"] / max(profiles[0]["demand_gw"], 1.0), 4
    )
    stats["gas_runs_n_of_100"] = sum(1 for p in profiles if p["gas_gw"] > 0.1)
    return stats


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    per_iso: dict[str, dict] = {}

    for iso in ISOS:
        # Use ep95 (0.95) for worst-hours chart — at this threshold gas still has
        # meaningful dispatch during worst hours (reliability tax most visible).
        # At ep99.9, storage overbuild eclipses gas entirely, obscuring the story.
        TARGET_EP = 0.95
        best_ep = TARGET_EP
        best_run = get_run(iso, "1", TARGET_EP)

        if best_run is None:
            per_iso[iso] = {
                "endpoint": None,
                "note": "No feasible Pathway 1 endpoint found",
                "worst_100_hours": [],
                "summary_stats": {},
            }
            continue

        # Demand and retirement data
        cost_rows = best_run.get("tables", {}).get("annual_cost", [])
        demand_2050 = cost_rows[-1]["demand_twh"] if cost_rows else 0.0
        avg_demand_gw = demand_2050 / 8.76

        # Retirement timeline at endpoint
        rt = best_run.get("retirement_timeline", [])
        gas_retired_gw = rt[-1]["gas_retired_gw"] if rt else 0.0
        gas_remaining_gw = max(0.0, EXISTING_GAS_GW.get(iso, 0.0) - gas_retired_gw)

        # Installed GW from annual_buildout (stranding_ledger is sparse for low-overbuild ISOs)
        bo = best_run.get("tables", {}).get("annual_buildout", [])
        installed_gw = _installed_gw_from_buildout(bo, iso)

        # Peak demand
        peak_gw = _peak_demand_gw(demand_2050, iso)

        # Generate worst 100 hours
        profiles = _worst_100_hours(peak_gw, installed_gw, gas_remaining_gw, iso, rng)
        stats = _summary_stats(profiles)

        per_iso[iso] = {
            "endpoint": best_ep,
            "endpoint_label": f"{best_ep * 100:.1f}%",
            "achieved_cfe_pct": best_run.get("achieved_cfe_pct", 0.0),
            "demand_2050_twh": round(demand_2050, 1),
            "avg_demand_gw": round(avg_demand_gw, 1),
            "peak_demand_gw": peak_gw,
            "installed_gw": {k: round(v, 1) for k, v in installed_gw.items()},
            "gas_remaining_gw": round(gas_remaining_gw, 1),
            "total_storage_rated_gw": round(
                installed_gw.get("battery4", 0)
                + installed_gw.get("battery8", 0)
                + installed_gw.get("ldes", 0),
                1,
            ),
            "worst_100_hours": profiles,
            "summary_stats": stats,
        }

    payload = {
        "per_iso": per_iso,
        "meta": {
            "payload_id": "section1_worst_hours",
            "pathway": "1",
            "note": (
                "SYNTHETIC worst-hour profiles. "
                "endpoint_hourly_dispatch is None in schema_version=1 runs. "
                "These 100 profiles per ISO are DERIVED from resource mix data "
                "(stranding_ledger TWh + capacity factors) and demand statistics. "
                "They are representative modeled profiles for visualization, not "
                "extracted from a 8760-hour simulation. "
                "Random seed = 42 for reproducibility."
            ),
            "construction": {
                "demand_range": "peak×[0.85, 1.00]",
                "solar": "0–2% of installed (nighttime)",
                "wind": "2–12% of installed (calm low-wind conditions)",
                "storage": "fills residual up to rated power (battery4 + battery8 + ldes)",
                "clean_firm_gw": "0 (Pathway 1, locked)",
                "gas": "max(0, residual - storage), capped at gas_remaining_gw",
                "sort": "descending by residual_load = demand - solar - wind",
            },
            "capacity_derivation": {
                "solar_wind": "annual_buildout cumulative TWh / (8760h × CF)",
                "battery4_rated_gw": f"dispatch_twh / {BATTERY4_DISPATCH_H_YR} h/yr",
                "battery8_rated_gw": f"dispatch_twh / {BATTERY8_DISPATCH_H_YR} h/yr",
                "ldes_rated_gw": f"dispatch_twh / {LDES_DISPATCH_H_YR} h/yr",
            },
            "version": "1.0",
        },
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH}")

    print("\n--- section1_worst_hours quick summary ---")
    for iso, d in per_iso.items():
        if not d.get("worst_100_hours"):
            print(f"  {iso}: {d.get('note', 'no data')}")
            continue
        st = d["summary_stats"]
        print(
            f"  {iso} @ ep{d['endpoint_label']}: "
            f"peak={d['peak_demand_gw']:.0f}GW, "
            f"gas_remaining={d['gas_remaining_gw']:.1f}GW, "
            f"avg_wind={st.get('avg_wind_gw',0):.0f}GW, "
            f"avg_storage={st.get('avg_battery_gw',0)+st.get('avg_ldes_gw',0):.0f}GW, "
            f"avg_gas={st.get('avg_gas_gw',0):.1f}GW ({st.get('gas_runs_n_of_100',0)}/100 hrs)"
        )
        installed = d["installed_gw"]
        print(
            f"    installed: solar={installed.get('solar',0):.0f}GW, "
            f"wind={installed.get('wind',0):.0f}GW, "
            f"batt4={installed.get('battery4',0):.0f}GW, "
            f"batt8={installed.get('battery8',0):.0f}GW, "
            f"ldes={installed.get('ldes',0):.0f}GW"
        )


if __name__ == "__main__":
    main()
