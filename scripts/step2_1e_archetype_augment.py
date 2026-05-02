#!/usr/bin/env python3
"""
step2_1e_archetype_augment.py
Generate EF mixes to fill archetype gaps at the 2030 seed band.

For each ISO, ensures ≥5,000 mixes per archetype exist in the Step 2.1
EF parquets at the seed band used by the Step 2.3 pathway optimizer.

Key design principle: archetypes are defined by what's ADDED above the
existing grid mix, not by total resource composition. A PJM mix with 32%
existing nuclear + 20% added solar + 10% added wind is "solar-led" because
the incremental solar family (20%) exceeds 50% of total incremental (30%).

Seed bands (from Step 2.3 SEED_BAND_BY_ISO):
  Band 60: CAISO, ERCOT, SPP  (baseline ≥ 45%)
  Band 50: MISO, NEISO, NYISO, PJM

Archetypes (incremental classification):
  solar-led:     incremental solar family / total incremental > 0.5
  wind-led:      incremental wind family / total incremental > 0.5
  nuclear-heavy: incremental clean_firm / total incremental > 0.4  (B/C/D only)
  balanced:      everything else

Output: step_2_1_EF_{iso}_{band}_augment.parquet per ISO.
All resource columns are present in the output with 0 for unused resources —
no nulls that would produce NaN after concat with base parquets.

Usage:
    python scripts/step2_1e_archetype_augment.py                 # all ISOs
    python scripts/step2_1e_archetype_augment.py --iso NYISO     # single ISO
    python scripts/step2_1e_archetype_augment.py --target 10000  # custom target
"""
from __future__ import annotations

import argparse
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

SOLAR_FAMILY = ["solar", "solar_batt4", "solar_batt8"]
WIND_FAMILY = ["wind", "wind_batt4", "wind_batt8"]

EF_DIR = PROJECT_ROOT / "data" / "step2.1-ef"
GEN_PROF_PATH = PROJECT_ROOT / "data" / "eia-930" / "eia_generation_profiles.parquet"
DEMAND_PROF_PATH = PROJECT_ROOT / "data" / "eia-930" / "eia_demand_profiles.parquet"
HYBRID_DIR = PROJECT_ROOT / "data" / "hybrid_profiles"

# Matches Step 2.3 solver's SEED_BAND_BY_ISO
SEED_BAND_BY_ISO = {
    "CAISO": 60, "ERCOT": 60, "SPP": 60,
    "PJM": 50, "NYISO": 50, "NEISO": 50, "MISO": 50,
}

# Pathway A archetypes (no nuclear-heavy — clean_firm expansion frozen)
PA_ARCHETYPES = ["balanced", "solar-led", "wind-led"]
# Pathway B/C/D add nuclear-heavy
PB_ARCHETYPES = ["balanced", "solar-led", "wind-led", "nuclear-heavy"]

BATCH_SIZE = 50_000
MAX_ROUNDS = 20

ARCH_LABELS = {0: "balanced", 1: "solar-led", 2: "wind-led", 3: "nuclear-heavy"}
ARCH_CODES = {v: k for k, v in ARCH_LABELS.items()}


# ---------------------------------------------------------------------------
# Incremental archetype classification (matches solver's classify_archetype)
# ---------------------------------------------------------------------------
def classify_archetype_incremental(
    mix_pcts: dict[str, float], pathway: str, iso: str,
) -> str:
    """Classify by dominant INCREMENTAL resource above grid baseline."""
    baseline = pc.GRID_MIX_SHARES.get(iso, {})

    solar_delta = sum(
        max(mix_pcts.get(r, 0) - baseline.get(r, 0), 0) for r in SOLAR_FAMILY
    )
    wind_delta = sum(
        max(mix_pcts.get(r, 0) - baseline.get(r, 0), 0) for r in WIND_FAMILY
    )
    total_delta = sum(
        max(mix_pcts.get(r, 0) - baseline.get(r, 0), 0) for r in RESOURCE_ORDER
    )
    if total_delta < 1e-6:
        return "balanced"

    sf = solar_delta / total_delta
    wf = wind_delta / total_delta

    if pathway in ("B", "C", "D"):
        nuke_delta = max(
            mix_pcts.get("clean_firm", 0) - baseline.get("clean_firm", 0), 0
        )
        off_delta = max(
            mix_pcts.get("offshore_wind", 0) - baseline.get("offshore_wind", 0), 0
        )
        if nuke_delta / total_delta > 0.4:
            return "nuclear-heavy"
        if off_delta / total_delta > 0.3:
            return "offshore-heavy"

    if sf > 0.5:
        return "solar-led"
    if wf > 0.5:
        return "wind-led"
    return "balanced"


def classify_batch_incremental(
    W_pct: np.ndarray, iso: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized incremental classification for PA and PB.

    W_pct is (n, NR) in percent. Subtracts grid baseline before classifying.
    Returns (pa_cls, pb_cls) integer arrays.
    """
    n = W_pct.shape[0]
    baseline = pc.GRID_MIX_SHARES.get(iso, {})

    # Build baseline vector aligned to RESOURCE_ORDER
    base_vec = np.array(
        [baseline.get(r, 0) for r in RESOURCE_ORDER], dtype=np.float64
    )

    # Incremental = max(mix - baseline, 0) per resource
    delta = np.maximum(W_pct - base_vec[None, :], 0)

    sol_idx = [RES_IDX[r] for r in SOLAR_FAMILY]
    wnd_idx = [RES_IDX[r] for r in WIND_FAMILY]

    solar_delta = delta[:, sol_idx].sum(axis=1)
    wind_delta = delta[:, wnd_idx].sum(axis=1)
    total_delta = delta.sum(axis=1)
    total_safe = np.maximum(total_delta, 1e-6)

    sf_r = solar_delta / total_safe
    wf_r = wind_delta / total_safe
    nk_r = delta[:, RES_IDX["clean_firm"]] / total_safe
    off_r = delta[:, RES_IDX["offshore_wind"]] / total_safe

    # PA classification: solar-led=1, wind-led=2, balanced=0
    pa = np.zeros(n, dtype=np.int8)
    pa[sf_r > 0.5] = 1
    pa[wf_r > 0.5] = 2

    # PB classification: nuclear-heavy=3 overrides, then same
    pb = pa.copy()
    pb[nk_r > 0.4] = 3

    # Zero-delta mixes → balanced
    zero_mask = total_delta < 1e-6
    pa[zero_mask] = 0
    pb[zero_mask] = 0

    return pa, pb


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
        dt = pq.read_table(
            DEMAND_PROF_PATH,
            columns=["iso", "year", "hour", "normalized"],
            filters=[("iso", "=", iso)],
        )
        if dt.num_rows > 0:
            max_yr = pac.max(dt["year"]).as_py()
            dt = dt.filter(pac.equal(dt["year"], max_yr))
            dn2 = np.zeros(H, dtype=np.float64)
            dn2[dt.column("hour").to_numpy().astype(np.int64)] = (
                dt.column("normalized").to_numpy().astype(np.float64)
            )
            s = dn2.sum()
            if s > 0:
                dn = dn2 / s
        del dt

    return P.astype(np.float32), dn.astype(np.float32)


def score_batch(
    W_frac: np.ndarray, P32: np.ndarray, dn32: np.ndarray,
) -> np.ndarray:
    """Score mixes. W_frac is (n, NR) in 0-1 scale.
    Returns CFE scores as percentages. Chunked to avoid OOM.
    """
    n = W_frac.shape[0]
    chunk = 5000
    scores = np.zeros(n, dtype=np.float32)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        supply = W_frac[i:end] @ P32
        scores[i:end] = np.minimum(supply, dn32[None, :]).sum(axis=1) * 100.0
        del supply
    return scores


# ---------------------------------------------------------------------------
# Existing EF inventory (with NaN zero-fill)
# ---------------------------------------------------------------------------
def load_existing_counts(iso: str, band: int) -> dict[str, int]:
    """Count mixes per archetype in existing EF parquets for this ISO/band.

    Uses incremental classification against grid baseline.
    Zero-fills NaN from concat of parquets with mismatched column sets.
    """
    pat = f"step_2_1_EF_{iso}_{band}"
    paths = sorted(
        p for p in EF_DIR.iterdir()
        if p.name.startswith(pat)
        and "peakclean" not in p.name
        and "interp" not in p.name
        and p.suffix == ".parquet"
    )
    if not paths:
        return {}

    cols_available = set()
    for p in paths:
        cols_available.update(pq.read_schema(p).names)
    cols = [r for r in RESOURCE_ORDER if r in cols_available]

    tbls = [pq.read_table(p, columns=[c for c in cols if c in pq.read_schema(p).names])
            for p in paths]
    ef = pa.concat_tables(tbls, promote_options="default") if len(tbls) > 1 else tbls[0]
    n = ef.num_rows

    W_pct = np.zeros((n, NR), dtype=np.float64)
    names = set(ef.schema.names)
    for i, res in enumerate(RESOURCE_ORDER):
        if res in names:
            raw = ef.column(res).to_numpy(zero_copy_only=False).astype(np.float64)
            W_pct[:, i] = np.nan_to_num(raw, nan=0.0)
    del ef

    pa_cls, pb_cls = classify_batch_incremental(W_pct, iso)
    del W_pct

    counts = {}
    for code, label in ARCH_LABELS.items():
        if label == "nuclear-heavy":
            counts[label] = int((pb_cls == code).sum())
        else:
            pa_count = int((pa_cls == code).sum())
            pb_count = int((pb_cls == code).sum())
            counts[label] = min(pa_count, pb_count)
    return counts


# ---------------------------------------------------------------------------
# Mix generation with incremental archetype constraints
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
    """Generate mixes classified as `archetype` (incremental) in target band.

    Strategy: set baseline resources at grid mix levels, then sample
    INCREMENTAL resources with constraints that drive the desired
    archetype classification.

    Returns (n_found, NR+1) array — resource pcts + score.
    """
    gm = pc.GRID_MIX_SHARES[iso]
    sol_cap = pc.SOLAR_FAMILY_CAP[iso]
    wnd_cap = pc.WIND_FAMILY_CAP[iso]
    hyb_max = pc.HYBRID_MAX_PER_TYPE
    has_offshore = iso in pc.OFFSHORE_ISOS

    # Baseline levels (cannot go below existing grid)
    cf_base = gm.get("clean_firm", 0)
    hydro_base = gm.get("hydro", 0)
    solar_base = gm.get("solar", 0)
    wind_base = gm.get("wind", 0)

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

        # Fixed baseline resources
        W_pct[:, RES_IDX["hydro"]] = hydro_base

        if archetype == "solar-led":
            # Incremental solar family must be >50% of total incremental.
            # Keep clean_firm at baseline (Pathway A) or near baseline.
            # Add significant solar, moderate solar hybrids, minimal wind.
            W_pct[:, RES_IDX["clean_firm"]] = cf_base
            # Incremental solar: substantial above baseline
            W_pct[:, RES_IDX["solar"]] = rng.integers(
                max(0, int(solar_base)), min(70, sol_cap), n_gen)
            W_pct[:, RES_IDX["solar_batt4"]] = rng.integers(0, min(30, hyb_max), n_gen)
            W_pct[:, RES_IDX["solar_batt8"]] = rng.integers(0, min(20, hyb_max), n_gen)
            # Incremental wind: small so solar dominates
            W_pct[:, RES_IDX["wind"]] = rng.integers(
                int(wind_base), int(wind_base) + 12, n_gen)
            W_pct[:, RES_IDX["wind_batt4"]] = rng.integers(0, 5, n_gen)
            W_pct[:, RES_IDX["wind_batt8"]] = rng.integers(0, 5, n_gen)
            if has_offshore:
                W_pct[:, RES_IDX["offshore_wind"]] = rng.integers(0, 5, n_gen)

        elif archetype == "wind-led":
            # Incremental wind family must be >50% of total incremental.
            W_pct[:, RES_IDX["clean_firm"]] = cf_base
            W_pct[:, RES_IDX["wind"]] = rng.integers(
                max(0, int(wind_base)), min(70, wnd_cap), n_gen)
            W_pct[:, RES_IDX["wind_batt4"]] = rng.integers(0, min(30, hyb_max), n_gen)
            W_pct[:, RES_IDX["wind_batt8"]] = rng.integers(0, min(20, hyb_max), n_gen)
            # Incremental solar: small so wind dominates
            W_pct[:, RES_IDX["solar"]] = rng.integers(
                int(solar_base), int(solar_base) + 12, n_gen)
            W_pct[:, RES_IDX["solar_batt4"]] = rng.integers(0, 5, n_gen)
            W_pct[:, RES_IDX["solar_batt8"]] = rng.integers(0, 5, n_gen)
            if has_offshore:
                W_pct[:, RES_IDX["offshore_wind"]] = rng.integers(0, 5, n_gen)

        elif archetype == "nuclear-heavy":
            # Incremental clean_firm / total incremental > 0.4.
            # Must ADD nuclear well above baseline.
            W_pct[:, RES_IDX["clean_firm"]] = rng.integers(
                int(cf_base) + 10, int(cf_base) + 50, n_gen)
            W_pct[:, RES_IDX["solar"]] = rng.integers(
                int(solar_base), int(solar_base) + 15, n_gen)
            W_pct[:, RES_IDX["solar_batt4"]] = rng.integers(0, 10, n_gen)
            W_pct[:, RES_IDX["wind"]] = rng.integers(
                int(wind_base), int(wind_base) + 15, n_gen)
            W_pct[:, RES_IDX["wind_batt4"]] = rng.integers(0, 10, n_gen)
            if has_offshore:
                W_pct[:, RES_IDX["offshore_wind"]] = rng.integers(0, 10, n_gen)

        elif archetype == "balanced":
            # Neither solar nor wind family >50% incremental,
            # nuclear ≤40% incremental.
            W_pct[:, RES_IDX["clean_firm"]] = rng.integers(
                max(0, int(cf_base) - 2), int(cf_base) + 15, n_gen)
            W_pct[:, RES_IDX["solar"]] = rng.integers(
                int(solar_base), int(solar_base) + 25, n_gen)
            W_pct[:, RES_IDX["solar_batt4"]] = rng.integers(0, 15, n_gen)
            W_pct[:, RES_IDX["solar_batt8"]] = rng.integers(0, 10, n_gen)
            W_pct[:, RES_IDX["wind"]] = rng.integers(
                int(wind_base), int(wind_base) + 25, n_gen)
            W_pct[:, RES_IDX["wind_batt4"]] = rng.integers(0, 15, n_gen)
            W_pct[:, RES_IDX["wind_batt8"]] = rng.integers(0, 10, n_gen)
            if has_offshore:
                W_pct[:, RES_IDX["offshore_wind"]] = rng.integers(0, 10, n_gen)

        # Enforce family caps
        sol_fam = sum(W_pct[:, RES_IDX[r]] for r in SOLAR_FAMILY)
        wnd_fam = sum(W_pct[:, RES_IDX[r]] for r in WIND_FAMILY)
        cap_mask = (sol_fam <= sol_cap) & (wnd_fam <= wnd_cap)
        W_pct = W_pct[cap_mask]

        if len(W_pct) == 0:
            continue

        total_generated += len(W_pct)

        # Classify incrementally and filter to target archetype
        pa_cls, pb_cls = classify_batch_incremental(W_pct, iso)

        if archetype == "nuclear-heavy":
            arch_mask = pb_cls == ARCH_CODES[archetype]
        else:
            arch_code = ARCH_CODES[archetype]
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
            collected.append(np.column_stack([W_valid, scores_valid[:, None]]))

        n_so_far = sum(len(c) for c in collected)
        hit_rate = total_scored / max(total_generated, 1) * 100
        print(
            f"    round {round_num}: generated {total_generated:,}, "
            f"in-band {total_scored:,} ({hit_rate:.1f}%), "
            f"collected {n_so_far:,}/{n_target:,}"
        )

        if n_so_far >= n_target:
            break

    if not collected:
        return np.zeros((0, NR + 1), dtype=np.float64)

    result = np.concatenate(collected, axis=0)
    if len(result) > n_target:
        result = result[:n_target]
    return result


# ---------------------------------------------------------------------------
# Write augmented parquet (all resource columns, zero-filled, no NaN)
# ---------------------------------------------------------------------------
def write_augment_parquet(iso: str, band: int, all_mixes: np.ndarray) -> Path:
    """Write augmented mixes to parquet matching Step 2.1 schema.

    all_mixes: (n, NR+1) — resource pcts, last col is score.
    All resource columns are written explicitly with 0 defaults so concat
    with base parquets never introduces null → NaN.
    """
    if len(all_mixes) == 0:
        return None

    n = len(all_mixes)
    data = {"iso": pa.array([iso] * n, type=pa.string())}

    # Write ALL resource columns — even those that are zero
    for ri, res in enumerate(RESOURCE_ORDER):
        data[res] = pa.array(all_mixes[:, ri].astype(np.int16), type=pa.int16())

    # Storage columns: zero (augment mixes have no pre-computed dispatch)
    for sc in STORAGE_COLS:
        data[sc] = pa.array(np.zeros(n, dtype=np.float64), type=pa.float64())

    data["hourly_match_score"] = pa.array(
        all_mixes[:, NR].astype(np.float64), type=pa.float64()
    )
    data["pareto_type"] = pa.array(["augment"] * n, type=pa.string())

    table = pa.table(data)
    out_path = EF_DIR / f"step_2_1_EF_{iso}_{band}_augment.parquet"
    pq.write_table(table, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_iso(iso: str, target: int):
    band = SEED_BAND_BY_ISO[iso]
    base_pct = sum(pc.GRID_MIX_SHARES[iso].values())
    baseline = pc.GRID_MIX_SHARES[iso]

    print(f"\n{'='*60}")
    print(f"  {iso} (baseline={base_pct:.1f}%, band={band}%)")
    print(f"  Grid mix: {baseline}")
    print(f"{'='*60}")

    # Count existing mixes per archetype (incremental classification)
    existing = load_existing_counts(iso, band)
    print(f"  Existing counts (incremental classification): {existing}")

    # Determine deficits — use full PB archetype set
    all_archetypes = PB_ARCHETYPES
    deficits = {}
    for arch in all_archetypes:
        have = existing.get(arch, 0)
        need = max(0, target - have)
        if need > 0:
            deficits[arch] = need
            print(f"  {arch:>15}: have {have:,}, need +{need:,}")
        else:
            print(f"  {arch:>15}: have {have:,} OK")

    if not deficits:
        print(f"  All archetypes have >= {target:,} mixes. Nothing to do.")
        return

    total_needed = sum(deficits.values())
    print(f"\n  Total mixes to generate: {total_needed:,}")

    print("  Loading profiles...")
    P32, dn32 = load_profiles(iso)

    rng = np.random.default_rng(seed=42 + hash(iso) % 1000)

    all_generated = []
    for arch, need in sorted(deficits.items(), key=lambda x: -x[1]):
        print(f"\n  Generating {need:,} {arch} mixes:")
        result = generate_archetype_mixes(iso, arch, need, band, P32, dn32, rng)
        n_got = len(result)
        if n_got < need:
            print(f"  WARNING: Only generated {n_got:,}/{need:,} {arch} mixes")
        else:
            print(f"  OK: Generated {n_got:,} {arch} mixes")
        if n_got > 0:
            all_generated.append(result)

    if not all_generated:
        print("  No mixes generated. Skipping write.")
        return

    combined = np.concatenate(all_generated, axis=0)
    out_path = write_augment_parquet(iso, band, combined)
    print(f"\n  Wrote {len(combined):,} augmented mixes to {out_path.name}")

    # Verify incremental classification
    pa_cls, pb_cls = classify_batch_incremental(combined[:, :NR], iso)
    for code, label in ARCH_LABELS.items():
        pa_n = int((pa_cls == code).sum())
        pb_n = int((pb_cls == code).sum())
        print(f"    {label:>15}: PA={pa_n:,}  PB={pb_n:,}")


def main():
    ap = argparse.ArgumentParser(
        description="Step 2.1e — Archetype augmentation for seed-band EF parquets"
    )
    ap.add_argument("--iso", type=str, default=None,
                    help="Single ISO to process (default: all)")
    ap.add_argument("--target", type=int, default=TARGET_PER_ARCHETYPE,
                    help=f"Target mixes per archetype (default: {TARGET_PER_ARCHETYPE})")
    args = ap.parse_args()

    isos = [args.iso.upper()] if args.iso else list(pc.ISOS)
    print("Step 2.1e Archetype Augment (v2.0 — incremental classification)")
    print(f"Target: {args.target:,} mixes per archetype")
    print(f"ISOs: {isos}")
    print(f"Seed bands: { {iso: SEED_BAND_BY_ISO[iso] for iso in isos} }")

    t0 = time.time()
    for iso in isos:
        process_iso(iso, args.target)

    print(f"\n{'='*60}")
    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
