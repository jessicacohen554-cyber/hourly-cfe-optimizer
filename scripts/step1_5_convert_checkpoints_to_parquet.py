#!/usr/bin/env python3
"""Step 1.5 utility: convert Step 1 JSON checkpoints into raw parquet files.

Input:  data/checkpoints/*_cache_*.json
Output: data/step1_raw_pfs_parquets/*_raw_pfs.parquet
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

FILENAME_RE = re.compile(r"^(?P<iso>[A-Z0-9]+)_cache_(?P<threshold>[0-9.]+)\.json$")
KEY_RE = re.compile(
    r"^\('(?P<dispatch_mode>[a-z]+)',\s*(?P<mix>\([^)]*\)|\[[^\]]*\]),\s*(?P<threshold>[^,\)]+)(?:,\s*(?P<arg3>[^,\)]+))?(?:,\s*(?P<arg4>[^,\)]+))?\)$"
)


def _parse_number(value: str):
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _parse_mix(mix_blob: str):
    trimmed = mix_blob.strip()
    if len(trimmed) < 2:
        return None

    inner = trimmed[1:-1].strip()
    if not inner:
        return []

    mix_values = []
    for part in inner.split(","):
        parsed = _parse_number(part)
        if parsed is None:
            return None
        mix_values.append(parsed)
    return mix_values


def parse_key(key: str) -> dict:
    """Parse stringified tuple keys from checkpoint JSONs.

    Known key shapes:
      ('h', mix, threshold)
      ('b', mix, threshold, battery_hours)
      ('bl', mix, threshold, ldes_hours, battery_hours)
    """
    parsed = {
        "raw_key": key,
        "dispatch_mode": None,
        "mix": None,
        "threshold": None,
        "ldes_hours": None,
        "battery_hours": None,
    }

    m = KEY_RE.match(key)
    if m:
        parsed_mix = _parse_mix(m.group("mix"))
        threshold = _parse_number(m.group("threshold"))
        if parsed_mix is not None and threshold is not None:
            parsed["dispatch_mode"] = m.group("dispatch_mode")
            parsed["mix"] = parsed_mix
            parsed["threshold"] = threshold

            arg3 = m.group("arg3")
            arg4 = m.group("arg4")
            if arg4 is None and arg3 is not None:
                parsed["battery_hours"] = _parse_number(arg3)
            elif arg4 is not None:
                parsed["ldes_hours"] = _parse_number(arg3)
                parsed["battery_hours"] = _parse_number(arg4)
            return parsed

    # Fallback parser for unexpected key shapes.
    try:
        key_tuple = ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return parsed

    if not isinstance(key_tuple, tuple) or len(key_tuple) < 3:
        return parsed

    parsed["dispatch_mode"] = key_tuple[0]

    mix = key_tuple[1]
    if isinstance(mix, tuple):
        parsed["mix"] = list(mix)
    elif isinstance(mix, list):
        parsed["mix"] = mix

    parsed["threshold"] = key_tuple[2]

    if len(key_tuple) == 4:
        parsed["battery_hours"] = key_tuple[3]
    elif len(key_tuple) >= 5:
        parsed["ldes_hours"] = key_tuple[3]
        parsed["battery_hours"] = key_tuple[4]

    return parsed


def convert_file(path: Path, output_dir: Path) -> Path:
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected checkpoint filename format: {path.name}")

    iso = m.group("iso")
    threshold = float(m.group("threshold"))

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for k, v in data.items():
        row = parse_key(k)
        row["iso"] = iso
        row["source_threshold"] = threshold
        row["lcoe"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[
        [
            "iso",
            "source_threshold",
            "dispatch_mode",
            "mix",
            "threshold",
            "ldes_hours",
            "battery_hours",
            "lcoe",
            "raw_key",
        ]
    ]

    output_path = output_dir / f"{iso}_t{threshold:g}_raw_pfs.parquet"
    df.to_parquet(output_path, index=False, compression="zstd")
    return output_path


def convert_files(checkpoint_files: list[Path], output_dir: Path, workers: int) -> list[Path]:
    if workers <= 1:
        return [convert_file(fp, output_dir) for fp in checkpoint_files]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(convert_file, fp, output_dir) for fp in checkpoint_files]
        return [future.result() for future in futures]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Step 1 JSON checkpoints to parquet")
    parser.add_argument("--input-dir", default="data/checkpoints", help="Directory with *_cache_*.json files")
    parser.add_argument(
        "--output-dir",
        default="data/step1_raw_pfs_parquets",
        help="Directory where parquet outputs are written",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes to use across files (0 = auto, 1 = sequential)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = sorted(input_dir.glob("*_cache_*.json"))
    if not checkpoint_files:
        raise SystemExit(f"No checkpoint JSON files found in: {input_dir}")

    auto_workers = max(1, math.ceil((os.cpu_count() or 1) * 0.75))
    max_workers = args.workers or min(len(checkpoint_files), auto_workers)
    max_workers = max(1, min(max_workers, len(checkpoint_files)))

    converted = convert_files(checkpoint_files, output_dir, workers=max_workers)
    for fp, out in zip(checkpoint_files, converted):
        print(f"Converted {fp} -> {out}")

    print(f"Done. Converted {len(converted)} files into {output_dir} using {max_workers} worker(s)")


if __name__ == "__main__":
    main()
