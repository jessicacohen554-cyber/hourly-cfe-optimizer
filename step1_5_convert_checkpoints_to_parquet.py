#!/usr/bin/env python3
"""Step 1.5 utility: convert Step 1 JSON checkpoints into raw parquet files.

Input:  data/checkpoints/*_cache_*.json
Output: data/step1_raw_pfs_parquets/*_raw_pfs.parquet
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

FILENAME_RE = re.compile(r"^(?P<iso>[A-Z0-9]+)_cache_(?P<threshold>[0-9.]+)\.json$")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Step 1 JSON checkpoints to parquet")
    parser.add_argument("--input-dir", default="data/checkpoints", help="Directory with *_cache_*.json files")
    parser.add_argument(
        "--output-dir",
        default="data/step1_raw_pfs_parquets",
        help="Directory where parquet outputs are written",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = sorted(input_dir.glob("*_cache_*.json"))
    if not checkpoint_files:
        raise SystemExit(f"No checkpoint JSON files found in: {input_dir}")

    converted = []
    for fp in checkpoint_files:
        out = convert_file(fp, output_dir)
        converted.append(out)
        print(f"Converted {fp} -> {out}")

    print(f"Done. Converted {len(converted)} files into {output_dir}")


if __name__ == "__main__":
    main()
