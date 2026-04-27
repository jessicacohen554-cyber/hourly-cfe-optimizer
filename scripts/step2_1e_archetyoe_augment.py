#!/usr/bin/env python3
"""
step2_1e_archetype_augment.py
Generate EF mixes to fill archetype gaps at the 2030 waypoint band.

For each ISO, ensures ≥5,000 mixes per archetype (balanced, solar-led,
wind-led, nuclear-heavy) exist in the Step 2.1 EF parquets. Generates
constrained random mixes, scores them against ISO generation/demand
profiles, and keeps those hitting the target CFE band.

Archetypes (classification thresholds):
  solar-led:     solar_family / total_clean > 0.5
  wind-led:      wind_family / total_clean > 0.5
  nuclear-heavy: clean_firm / total_clean > 0.4  (Pathway B only)
  balanced:      everything else

Output: step_2_1_EF_{iso}_{band}_augment.parquet per ISO.
Stage 1 of Step 2.3 loads these alongside original parquets via filename
prefix matching.

Usage:
    python scripts/step2_1e_archetype_augment.py                 # all ISOs
    python scripts/step2_1e_archetype_augment.py --iso NYISO     # single ISO
    python scripts/step2_1e_archetype_augment.py --target 10000  # custom threshold
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import pipeline_config as pc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H = 8760
TARGET_PER_ARCHETYPE = 5000

RESOURCE_ORDER = [
    "clean_firm", "solar", "wind", "hydro", "offshore_wind",
    "ccs_ccgt", "geothermal",
    "solar_batt4", "solar_batt8", "wind_batt4", "wind_batt8",
]
NR = len(RESOURCE_ORDER)
RES_IDX = {r: i for i, r in enumerate(RESOURCE_ORDER)}

STORAGE_COLS = [
    "battery_dispatch_pct", "battery8_dispatch_pct",
    "ldes_dispatch_pct", "h2_dispatch_pct",
]

EF_DIR = PROJECT_ROOT / "data" / "step2.1-ef"
GEN_PROF_PATH = PROJECT_ROOT / "data" / "eia-930" / "eia_generation_profiles.parquet"
DEMAND_PROF_PATH = PROJECT_ROOT / "data" / "eia-930" / "eia_demand_profiles.parquet"
HYBRID_DIR = PROJECT_ROOT / "data" / "hybrid_profiles"

HIGH_BASELINE_ISOS = {"CAISO", "SPP", "ERCOT"}

# Pathway A archetypes (no nuclear-heavy — clean_firm is frozen)
PA_ARCHETYPES = ["balanced", "solar-led", "wind-led"]
# Pathway B adds nuclear-heavy
PB_ARCHETYPES = ["balanced", "solar-led", "wind-led", "nuclear-heavy"]

# Batch size for generate-and-filter (per round)
BATCH_SIZE = 50_000
MAX_ROUNDS = 20


# ---------------------------------------------------------------------------
# Archetype classification (matches step2_3 classify_archetype exactly)
# ---------------------------------------------------------------------------
def classify_archetype(mix_pcts: dict[str, float], pathway: str) -> str:
    solar_family = sum(mix_pcts.get(r, 0) for r in
                       ["solar", "solar_batt4", "solar_batt8"])
    wind_family = sum(mix_pcts.get(r, 0) for r in
                      ["wind", "wind_batt4", "wind_batt8"])
    total_clean = sum(mix_pcts.get(r, 0) for r in RESOURCE_ORDER)
    if total_clean < 1e-6:
        return "balanced"
    sf = solar_family / total_clean
    wf = wind_family / total_clean
    if pathway == "B":
        nuke = mix_pcts.get("clean_firm", 0) / total_clean
        off = mix_pcts.get("offshore_wind", 0) / total_clean
        if nuke > 0.4:
            return "nuclear-heavy"
        if off > 0.3:
            return "offshore-heavy"
    if sf > 0.5:
        return "solar-led"
    if wf > 0.5:
        return "wind-led"
    return "balanced"


def classify_batch(W_pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized classification for PA and PB. W_pct is (n, NR) in percent."""
    n = W_pct.shape[0]
    solar_fam = (W_pct[:, RES_IDX["solar"]]
                 + W_pct[:, RES_IDX["solar_batt4"]]
                 + W_pct[:, RES_IDX["solar_batt8"]])
    wind_fam = (W_pct[:, RES_IDX["wind"]]
                + W_pct[:, RES_IDX["wind_batt4"]]
                + W_pct[:, RES_IDX["wind_batt8"]])
    total = W_pct.sum(axis=1)
    total_safe = np.maximum(total, 1e-6)
    sf_r = solar_fam / total_safe
    wf_r = wind_fam / total_safe
    nk_r = W_pct[:, RES_IDX["clean_firm"]] / total_safe

    # PA classification: solar-led=1, wind-led=2, balanced=0
    pa = np.zeros(n, dtype=np.int8)
    pa[sf_r > 0.5] = 1  # solar-led
    pa[wf_r > 0.5] = 2  # wind-led

    # PB classification: nuclear-heavy=3, then same as PA
    pb = pa.copy()
    pb[nk_r > 0.4] = 3  # nuclear-heavy overrides

    return pa, pb


ARCH_LABELS = {0: "balanced", 1: "solar-led", 2: "wind-led", 3: "nuclear-heavy"}
ARCH_CODES = {v: k for k, v in ARCH_LABELS.items()}


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------
def load_profiles(iso: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (P32, dn32): profile matrix and demand norm, both float32."""
    flat = np.full(H, 1.0 / H, dtype=np.float64)
    zero = np.zeros(H, dtype=np.float64)

    gen = {}
    if GEN_PROF_PATH.exists():
        t = pq.read_table(GEN_PROF_PATH, filters=[("iso", "=", iso)])
        if t.num_rows > 0:
            max_yr = pac.max(t["year"]).as_py()
            t = t.filter(pac.equal(t["year"], max_yr))
            hrs = t.column("hour").to_numpy().astype(np.int64)
            vals = t.column("value").to_numpy().astype(np.float64)
            fc = t.column("fuel")
            for fuel in fc.unique().to_pylist():
                m = pac.equal(fc, fuel).to_numpy()
                p = np.zeros(H, dtype=np.float64)
                p[hrs[m]] = vals[m]
                gen[str(fuel)] = p
        del t

    hyb = {}
    npz = HYBRID_DIR / f"{iso}_hybrid_profiles.npz"
    if npz.exists():
        z = np.load(npz)
        for k in ("solar_batt4", "solar_batt8", "wind_batt4", "wind_batt8"):
            hyb[k] = z[k].astype(np.float64) if k in z else zero.copy()

    defaults = {
        "clean_firm": ("nuclear", flat), "solar": ("solar", zero),
        "wind": ("wind", zero), "hydro": ("hydro", flat),
        "offshore_wind": ("offshore_wind", zero),
        "ccs_ccgt": (None, flat), "geothermal": ("geothermal", flat),
    }
    rows = []
    for res in RESOURCE_ORDER:
        if res in hyb:
            rows.append(hyb[res])
        elif res in defaults:
            gk, df = defaults[res]
            rows.append(gen.get(gk, df) if gk else df)
        else:
            rows.append(gen.get(res, zero))
    P = np.stack(rows, axis=0)

    dn = flat.copy()
    if DEMAND_PROF_PATH.exists():
        dt = pq.read_table(DEMAND_PROF_PATH,
                           columns=["iso", "year", "hour", "normalized"],
                           filters=[("iso", "=", iso)])
        if dt.num_rows > 0:
            max_yr = pac.max(dt["year"]).as_py()
            dt = dt.filter(pac.equal(dt["year"], max_yr))
            dn2 = np.zeros(H, dtype=np.float64)
            dn2[dt.column("hour").to_numpy().astype(np.int64)] = (
                dt.column("normalized").to_numpy().astype(np.float64))
            s = dn2.sum()
            if s > 0:
                dn = dn2 / s
        del dt

    return P.astype(np.float32), dn.astype(np.float32)


def score_batch(W_frac: np.ndarray, P32: np.ndarray,
                dn32: np.ndarray) -> np.ndarray:
    """Score mixes. W_frac is (n, NR) in fractional form (0-1).
    Returns CFE scores as percentages.

    Processes in chunks to avoid OOM on large batches.
    """
    n = W_frac.shape[0]
    chunk = 5000  # process 5k at a time to stay under memory
    scores = np.zeros(n, dtype=np.float32)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        supply = W_frac[i:end] @ P32  # (chunk, H)
        scores[i:end] = np.minimum(supply, dn32[None, :]).sum(axis=1) * 100.0
        del supply
    return scores


# ---------------------------------------------------------------------------
# Existing EF inventory
# ---------------------------------------------------------------------------
def load_existing_counts(iso: str, band: int) -> dict[str, int]:
    """Count mixes per archetype in existing EF parquets for this ISO/band."""
    pat = f"step_2_1_EF_{iso}_{band}"
    paths = sorted(p for p in EF_DIR.iterdir()
                   if p.name.startswith(pat)
                   and "peakclean" not in p.name
                   and "interp" not in p.name
                   and p.suffix == ".parquet")
    if not paths:
        return {}

    # Read only resource columns
    schema = pq.read_schema(paths[0])
    cols = [r for r in RESOURCE_ORDER if r in schema.names]
    tbls = [pq.read_table(p, columns=cols) for p in paths]
    ef = pa.concat_tables(tbls) if len(tbls) > 1 else tbls[0]
    n = ef.num_rows

    W_pct = np.zeros((n, NR), dtype=np.float64)
    names = set(ef.schema.names)
    for i, res in enumerate(RESOURCE_ORDER):
        if res in names:
            W_pct[:, i] = ef.column(res).to_numpy().astype(np.float64)
    del ef

    pa_cls, pb_cls = classify_batch(W_pct)
    del W_pct

    # Count using the max of PA and PB (a mix needs to satisfy both)
    # But archetypes are different per pathway, so count the union
    counts = {}
    for code, label in ARCH_LABELS.items():
        # For nuclear-heavy, only count PB
        if label == "nuclear-heavy":
            counts[label] = int((pb_cls == code).sum())
        else:
            # Use PA classification (nuclear-heavy only exists on PB)
            pa_count = int((pa_cls == code).sum())
            pb_count = int((pb_cls == code).sum())
            counts[label] = min(pa_count, pb_count)  # conservative
    return counts


# ---------------------------------------------------------------------------
# Mix generation with archetype constraints
# ---------------------------------------------------------------------------
def generate_archetype_mixes(
    iso: str,
    archetype: str,
    n_target: int,
    band: int,
    P32: np.ndarray,
    dn32: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate mixes classified as `archetype` scoring in [band-1.5, band+6).

    Returns (n_found, NR) integer percentage array matching EF schema.
    """
    gm = pc.GRID_MIX_SHARES[iso]
    sol_cap = pc.SOLAR_FAMILY_CAP[iso]
    wnd_cap = pc.WIND_FAMILY_CAP[iso]
    hyb_max = pc.HYBRID_MAX_PER_TYPE  # 40
    has_offshore = iso in pc.OFFSHORE_ISOS

    # Fixed baseline resources
    hydro_base = gm.get("hydro", 0)
    cf_base = gm.get("clean_firm", 0)

    band_lo = band - 1.5
    band_hi = band + 6.0

    collected = []
    total_generated = 0
    total_scored = 0

    for round_num in range(MAX_ROUNDS):
        if sum(len(c) for c in collected) >= n_target:
            break

        n_gen = BATCH_SIZE
        W_pct = np.zeros((n_gen, NR), dtype=np.float64)

        # Hydro: fixed at baseline
        W_pct[:, RES_IDX["hydro"]] = hydro_base

        if archetype == "solar-led":
            # Solar family must be >50% of total.
            # Strategy: solar high, solar hybrids moderate, wind low, nuke at baseline
            W_pct[:, RES_IDX["clean_firm"]] = cf_base
            W_pct[:, RES_IDX["solar"]] = rng.integers(10, min(70, sol_cap), n_gen)
            W_pct[:, RES_IDX["solar_batt4"]] = rng.integers(0, min(30, hyb_max), n_gen)
            W_pct[:, RES_IDX["solar_batt8"]] = rng.integers(0, min(20, hyb_max), n_gen)
            W_pct[:, RES_IDX["wind"]] = rng.integers(0, 15, n_gen)
            W_pct[:, RES_IDX["wind_batt4"]] = rng.integers(0, 5, n_gen)
            W_pct[:, RES_IDX["wind_batt8"]] = rng.integers(0, 5, n_gen)
            if has_offshore:
                W_pct[:, RES_IDX["offshore_wind"]] = rng.integers(0, 5, n_gen)

        elif archetype == "wind-led":
            # Wind family must be >50% of total.
            W_pct[:, RES_IDX["clean_firm"]] = cf_base
            W_pct[:, RES_IDX["wind"]] = rng.integers(10, min(70, wnd_cap), n_gen)
            W_pct[:, RES_IDX["wind_batt4"]] = rng.integers(0, min(30, hyb_max), n_gen)
            W_pct[:, RES_IDX["wind_batt8"]] = rng.integers(0, min(20, hyb_max), n_gen)
            W_pct[:, RES_IDX["solar"]] = rng.integers(0, 15, n_gen)
            W_pct[:, RES_IDX["solar_batt4"]] = rng.integers(0, 5, n_gen)
            W_pct[:, RES_IDX["solar_batt8"]] = rng.integers(0, 5, n_gen)
            if has_offshore:
                W_pct[:, RES_IDX["offshore_wind"]] = rng.integers(0, 5, n_gen)

        elif archetype == "nuclear-heavy":
            # clean_firm / total > 0.4. Boost nuclear well above baseline.
            W_pct[:, RES_IDX["clean_firm"]] = rng.integers(
                max(int(cf_base), 15), min(60, 80), n_gen)
            W_pct[:, RES_IDX["solar"]] = rng.integers(0, 20, n_gen)
            W_pct[:, RES_IDX["solar_batt4"]] = rng.integers(0, 10, n_gen)
            W_pct[:, RES_IDX["wind"]] = rng.integers(0, 20, n_gen)
            W_pct[:, RES_IDX["wind_batt4"]] = rng.integers(0, 10, n_gen)
            if has_offshore:
                W_pct[:, RES_IDX["offshore_wind"]] = rng.integers(0, 10, n_gen)

        elif archetype == "balanced":
            # Neither solar nor wind >50%, nuclear ≤40%.
            W_pct[:, RES_IDX["clean_firm"]] = rng.integers(
                max(0, int(cf_base) - 2), int(cf_base) + 15, n_gen)
            W_pct[:, RES_IDX["solar"]] = rng.integers(0, 30, n_gen)
            W_pct[:, RES_IDX["solar_batt4"]] = rng.integers(0, 15, n_gen)
            W_pct[:, RES_IDX["solar_batt8"]] = rng.integers(0, 10, n_gen)
            W_pct[:, RES_IDX["wind"]] = rng.integers(0, 30, n_gen)
            W_pct[:, RES_IDX["wind_batt4"]] = rng.integers(0, 15, n_gen)
            W_pct[:, RES_IDX["wind_batt8"]] = rng.integers(0, 10, n_gen)
            if has_offshore:
                W_pct[:, RES_IDX["offshore_wind"]] = rng.integers(0, 10, n_gen)

        # Enforce family caps
        sol_fam = (W_pct[:, RES_IDX["solar"]]
                   + W_pct[:, RES_IDX["solar_batt4"]]
                   + W_pct[:, RES_IDX["solar_batt8"]])
        wnd_fam = (W_pct[:, RES_IDX["wind"]]
                   + W_pct[:, RES_IDX["wind_batt4"]]
                   + W_pct[:, RES_IDX["wind_batt8"]])
        cap_mask = (sol_fam <= sol_cap) & (wnd_fam <= wnd_cap)
        W_pct = W_pct[cap_mask]

        if len(W_pct) == 0:
            continue

        total_generated += len(W_pct)

        # Classify and filter to target archetype
        pa_cls, pb_cls = classify_batch(W_pct)

        if archetype == "nuclear-heavy":
            arch_mask = pb_cls == ARCH_CODES[archetype]
        else:
            arch_code = ARCH_CODES[archetype]
            # Must classify as target on BOTH pathways
            # (nuclear-heavy on PB overrides, so for balanced/solar/wind
            # we need PA match AND PB match or PB=nuclear-heavy is ok
            # since the mix serves PB nuclear-heavy separately)
            arch_mask = pa_cls == arch_code

        W_pct = W_pct[arch_mask]
        if len(W_pct) == 0:
            continue

        # Score against profiles
        W_frac = (W_pct * 0.01).astype(np.float32)
        scores = score_batch(W_frac, P32, dn32)
        del W_frac

        band_mask = (scores >= band_lo) & (scores < band_hi)
        W_valid = W_pct[band_mask]
        scores_valid = scores[band_mask]
        total_scored += len(W_valid)

        if len(W_valid) > 0:
            collected.append(np.column_stack([
                W_valid, scores_valid[:, None]
            ]))

        n_so_far = sum(len(c) for c in collected)
        hit_rate = total_scored / max(total_generated, 1) * 100
        print(f"    round {round_num}: generated {total_generated:,}, "
              f"in-band {total_scored:,} ({hit_rate:.1f}%), "
              f"collected {n_so_far:,}/{n_target:,}")

        if n_so_far >= n_target:
            break

    if not collected:
        return np.zeros((0, NR + 1), dtype=np.float64)

    result = np.concatenate(collected, axis=0)
    # Trim to target
    if len(result) > n_target:
        result = result[:n_target]
    return result  # (n, NR+1) — last col is score


# ---------------------------------------------------------------------------
# Write augmented parquet
# ---------------------------------------------------------------------------
def write_augment_parquet(iso: str, band: int,
                          all_mixes: np.ndarray) -> Path:
    """Write augmented mixes to parquet matching Step 2.1 schema.

    all_mixes: (n, NR+1) — resource pcts as int, last col is score.
    """
    if len(all_mixes) == 0:
        return None

    n = len(all_mixes)
    data = {
        "iso": pa.array([iso] * n, type=pa.string()),
    }
    for ri, res in enumerate(RESOURCE_ORDER):
        data[res] = pa.array(all_mixes[:, ri].astype(np.int16), type=pa.int16())
    for sc in STORAGE_COLS:
        data[sc] = pa.array(np.zeros(n), type=pa.float64())
    data["hourly_match_score"] = pa.array(
        all_mixes[:, NR].astype(np.float64), type=pa.float64())
    data["pareto_type"] = pa.array(["augment"] * n, type=pa.string())

    table = pa.table(data)
    out_path = EF_DIR / f"step_2_1_EF_{iso}_{band}_augment.parquet"
    pq.write_table(table, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_iso(iso: str, target: int):
    band = 60 if iso in HIGH_BASELINE_ISOS else 50
    base_pct = sum(pc.GRID_MIX_SHARES[iso].values())

    print(f"\n{'='*60}")
    print(f"  {iso} (baseline={base_pct:.1f}%, band={band}%)")
    print(f"{'='*60}")

    # Count existing mixes per archetype
    existing = load_existing_counts(iso, band)
    print(f"  Existing counts: {existing}")

    # Determine deficits — use the full PB archetype set
    # (PA archetypes are a subset; nuclear-heavy is PB-only)
    all_archetypes = PB_ARCHETYPES
    deficits = {}
    for arch in all_archetypes:
        have = existing.get(arch, 0)
        need = max(0, target - have)
        if need > 0:
            deficits[arch] = need
            print(f"  {arch:>15}: have {have:,}, need +{need:,}")
        else:
            print(f"  {arch:>15}: have {have:,} ✓")

    if not deficits:
        print(f"  All archetypes have ≥{target:,} mixes. Nothing to do.")
        return

    total_needed = sum(deficits.values())
    print(f"\n  Total mixes to generate: {total_needed:,}")

    # Load profiles once
    print(f"  Loading profiles...")
    P32, dn32 = load_profiles(iso)

    rng = np.random.default_rng(seed=42 + hash(iso) % 1000)

    all_generated = []
    for arch, need in sorted(deficits.items(), key=lambda x: -x[1]):
        print(f"\n  Generating {need:,} {arch} mixes:")
        result = generate_archetype_mixes(
            iso, arch, need, band, P32, dn32, rng)
        n_got = len(result)
        if n_got < need:
            print(f"  ⚠️  Only generated {n_got:,}/{need:,} {arch} mixes")
        else:
            print(f"  ✓ Generated {n_got:,} {arch} mixes")
        if n_got > 0:
            all_generated.append(result)

    if not all_generated:
        print(f"  No mixes generated. Skipping write.")
        return

    combined = np.concatenate(all_generated, axis=0)
    out_path = write_augment_parquet(iso, band, combined)
    print(f"\n  Wrote {len(combined):,} augmented mixes to {out_path.name}")

    # Verify
    pa_cls, pb_cls = classify_batch(combined[:, :NR])
    for code, label in ARCH_LABELS.items():
        pa_n = int((pa_cls == code).sum())
        pb_n = int((pb_cls == code).sum())
        print(f"    {label:>15}: PA={pa_n:,}  PB={pb_n:,}")


def main():
    ap = argparse.ArgumentParser(
        description="Step 2.1e — Archetype augmentation for EF parquets")
    ap.add_argument("--iso", type=str, default=None,
                    help="Single ISO to process (default: all)")
    ap.add_argument("--target", type=int, default=TARGET_PER_ARCHETYPE,
                    help=f"Target mixes per archetype (default: {TARGET_PER_ARCHETYPE})")
    args = ap.parse_args()

    isos = [args.iso.upper()] if args.iso else list(pc.ISOS)
    print(f"Step 2.1e Archetype Augment")
    print(f"Target: {args.target:,} mixes per archetype")
    print(f"ISOs: {isos}")

    t0 = time.time()
    for iso in isos:
        process_iso(iso, args.target)

    print(f"\n{'='*60}")
    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
