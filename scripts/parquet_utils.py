"""Shared parquet I/O utilities for the step-1 pipeline.

The main export is ``write_parquet_chunked`` which transparently splits a
PyArrow table into multiple part-files when the serialized size would exceed
a configurable threshold (default 45 MB).  This keeps every committed file
under GitHub's 50 MB warning limit.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Union

import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet_chunked(
    table: pa.Table,
    path: Union[str, Path],
    max_mb: int = 45,
    compression: str = "snappy",
) -> List[str]:
    """Write *table* to parquet, splitting into ``_part{NNN}`` files if needed.

    Parameters
    ----------
    table : pa.Table
        The data to write.
    path : str | Path
        Target file path (e.g. ``ERCOT_t90_raw_pfs.parquet``).
        If splitting is needed the stem is reused with ``_part001``, etc.
    max_mb : int
        Maximum file size in megabytes.  Files exceeding this are split.
        Set to 0 to disable splitting.
    compression : str
        Parquet compression codec (default ``snappy``).

    Returns
    -------
    list[str]
        Paths of all files written (length 1 when no split was needed).
    """
    path = str(path)
    max_bytes = max_mb * 1024 * 1024 if max_mb > 0 else float("inf")

    if table.num_rows == 0:
        pq.write_table(table, path, compression=compression)
        return [path]

    # ── Fast path: write once, check size ────────────────────────────────
    pq.write_table(table, path, compression=compression)
    file_size = os.path.getsize(path)

    if file_size <= max_bytes:
        return [path]

    # ── Need to split ────────────────────────────────────────────────────
    os.remove(path)
    # Use 80% of max_bytes as target to leave margin for compression
    # variation across row subsets (different rows compress differently).
    target_bytes = int(max_bytes * 0.80)
    n_parts = math.ceil(file_size / target_bytes)
    n_parts = max(n_parts, 2)
    rows_per_part = math.ceil(table.num_rows / n_parts)

    base = path.replace(".parquet", "")
    written: List[str] = []

    for i in range(n_parts):
        start = i * rows_per_part
        length = min(rows_per_part, table.num_rows - start)
        if length <= 0:
            break
        part_path = f"{base}_part{i + 1:03d}.parquet"
        pq.write_table(table.slice(start, length), part_path, compression=compression)
        part_size = os.path.getsize(part_path)

        # If a part still exceeds max_bytes, re-split it recursively
        if part_size > max_bytes:
            os.remove(part_path)
            sub_table = table.slice(start, length)
            sub_parts = _split_oversized(
                sub_table, base, i + 1, n_parts, max_bytes, compression
            )
            written.extend(sub_parts)
        else:
            written.append(part_path)
            print(f"  wrote {os.path.basename(part_path)}: {part_size / 1024 / 1024:.1f} MB ({length:,} rows)")

    print(f"  split {os.path.basename(path)} → {len(written)} parts (original {file_size / 1024 / 1024:.1f} MB)")
    return written


def _split_oversized(
    table: pa.Table,
    base: str,
    part_idx: int,
    total_parts: int,
    max_bytes: int,
    compression: str,
) -> List[str]:
    """Re-split a part that still exceeds max_bytes after the first split.

    Uses sub-part naming: ``_part{NNN}a``, ``_part{NNN}b``, etc.
    """
    # Write to temp to measure actual size
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    pq.write_table(table, tmp_path, compression=compression)
    actual_size = os.path.getsize(tmp_path)
    os.remove(tmp_path)

    # Compute sub-parts with generous margin
    n_sub = math.ceil(actual_size / (max_bytes * 0.75))
    n_sub = max(n_sub, 2)
    rows_per_sub = math.ceil(table.num_rows / n_sub)

    written: List[str] = []
    for j in range(n_sub):
        start = j * rows_per_sub
        length = min(rows_per_sub, table.num_rows - start)
        if length <= 0:
            break
        suffix = chr(ord('a') + j)
        sub_path = f"{base}_part{part_idx:03d}{suffix}.parquet"
        pq.write_table(table.slice(start, length), sub_path, compression=compression)
        sub_size = os.path.getsize(sub_path)
        written.append(sub_path)
        print(f"  wrote {os.path.basename(sub_path)}: {sub_size / 1024 / 1024:.1f} MB ({length:,} rows) [re-split]")

    return written


def read_parquet_parts(path: Union[str, Path]) -> pa.Table:
    """Read a parquet file that may have been split into ``_part{NNN}`` files.

    If *path* exists as-is, reads it directly.  Otherwise looks for
    ``{stem}_part*.parquet`` files and concatenates them.

    Parameters
    ----------
    path : str | Path
        The canonical path (e.g. ``ERCOT_t90_raw_pfs.parquet``).

    Returns
    -------
    pa.Table or None
        Combined table, or None if no files found.
    """
    import glob as _glob

    path = str(path)
    if os.path.exists(path):
        return pq.read_table(path)

    # Look for part files
    base = path.replace(".parquet", "")
    pattern = f"{base}_part*.parquet"
    parts = sorted(_glob.glob(pattern))
    if not parts:
        return None

    tables = [pq.read_table(p) for p in parts]
    return pa.concat_tables(tables)
