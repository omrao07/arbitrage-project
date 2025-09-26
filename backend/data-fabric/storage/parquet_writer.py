# storage/parquet_writer.py
"""
Partitioned Parquet writer with schema coercion and atomic writes.

Features
- fsspec paths (local, s3://, gcs://) with atomic temp->final renames
- Partitioning by one or more columns (e.g., date=..., ticker=...)
- Optional schema enforcement (dtype map)
- File rotation by approx. max rows or target bytes
- Compression / encoding options
- Lightweight manifest + stats (row counts, files) per write session
- Idempotent-friendly: unique file names via content hash or uuid

Dependencies
  pip install pyarrow fsspec s3fs gcsfs

Example
-------
from storage.parquet_writer import ParquetWriter, WriteOptions

w = ParquetWriter()
opts = WriteOptions(
    root="s3://hyper-lakehouse/equities/prices/bronze",
    partition_cols=["date","ticker"],
    compression="zstd",
    file_row_limit=1_000_000,
)
stats = w.write(df, opts)
print(stats.files, stats.rows)
"""

from __future__ import annotations

import io
import os
import uuid
import json
import time
import hashlib
import typing as T
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PA = True
except Exception:
    HAVE_PA = False

try:
    import fsspec
    HAVE_FSSPEC = True
except Exception:
    HAVE_FSSPEC = False


# --------------------------- Dataclasses ---------------------------

@dataclass
class WriteOptions:
    root: str                                    # e.g., ./data/... or s3://bucket/prefix
    partition_cols: list[str] = field(default_factory=list)
    compression: str = "zstd"                    # "zstd" | "snappy" | "gzip" | None
    coerce_schema: dict[str, str] | None = None  # pandas dtype strings, e.g. {"ticker":"string"}
    file_row_limit: int | None = None            # hard rotate by rows
    target_file_bytes: int | None = None         # soft rotate by size (approx using Arrow size)
    filename_prefix: str = "part"
    filename_suffix: str = ".parquet"
    use_content_hash_in_name: bool = True        # add sha1 for idempotency/uniqueness
    write_manifest: bool = True                  # write a small JSON manifest at session end
    manifest_name: str = "_manifest.json"
    logical_dataset: str | None = None           # optional dataset id to embed in metadata
    filesystem_kwargs: dict[str, T.Any] = field(default_factory=dict)  # e.g., anon creds for s3/gcs


@dataclass
class WriteStats:
    files: int = 0
    rows: int = 0
    partitions: int = 0
    bytes: int = 0
    details: list[dict] = field(default_factory=list)


# --------------------------- Helper utils --------------------------

def _ensure_deps():
    if not HAVE_PA:
        raise RuntimeError("pyarrow is required. Install with: pip install pyarrow")
    if not HAVE_FSSPEC:
        raise RuntimeError("fsspec is required. Install with: pip install fsspec (and s3fs/gcsfs for cloud)")

def _is_cloud(path: str) -> bool:
    return path.startswith(("s3://", "gcs://", "gs://"))

def _fs_for(path: str, **kwargs):
    if not _is_cloud(path):
        return fsspec.filesystem("file")
    if path.startswith("s3://"):
        return fsspec.filesystem("s3", **kwargs)
    if path.startswith(("gcs://", "gs://")):
        return fsspec.filesystem("gcs", **kwargs)
    # fallback: let fsspec infer
    return fsspec.open(path).fs

def _norm_part(value: T.Any) -> str:
    # ensure safe partition values (no slashes etc.)
    s = str(value)
    return s.replace("/", "_").replace(" ", "_")

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _hash_df(df: pd.DataFrame, cols: list[str] | None = None, limit_rows: int = 10_000) -> str:
    """
    Quick content hash combining first N rows + schema.
    """
    head = df.head(limit_rows)
    buf = io.BytesIO()
    head.to_parquet(buf, index=False)  # pandas+pyarrow
    meta = f"{list(df.columns)}|{str(df.dtypes)}".encode("utf-8")
    return _hash_bytes(buf.getvalue() + meta)

def _coerce_pandas_types(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, dtype in schema.items():
        if col not in out.columns:
            continue
        try:
            if dtype == "datetime64[ns, UTC]":
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
            else:
                out[col] = out[col].astype(dtype) # type: ignore
        except Exception:
            # best-effort: use pandas to_numeric/to_datetime fallbacks
            if dtype in ("float64", "float32"):
                out[col] = pd.to_numeric(out[col], errors="coerce")
            elif dtype.startswith("datetime"):
                out[col] = pd.to_datetime(out[col], errors="coerce", utc="UTC" in dtype)
            else:
                out[col] = out[col].astype("string")
    return out


# --------------------------- Writer -------------------------------

class ParquetWriter:
    def __init__(self):
        _ensure_deps()

    def write(self, df: pd.DataFrame, opts: WriteOptions) -> WriteStats:
        if df is None or len(df) == 0:
            return WriteStats(files=0, rows=0, partitions=0, bytes=0, details=[])

        # schema coercion (pandas-level)
        if opts.coerce_schema:
            df = _coerce_pandas_types(df, opts.coerce_schema)

        # validate partition columns
        for p in opts.partition_cols:
            if p not in df.columns:
                raise ValueError(f"Partition column '{p}' not in DataFrame")

        # create filesystem
        fs = _fs_for(opts.root, **opts.filesystem_kwargs)

        # prepare stats
        stats = WriteStats()
        session_id = uuid.uuid4().hex[:8]
        session_ts = datetime.now(timezone.utc).isoformat()

        # group by partitions (or treat whole df as one partition)
        group_iter = df.groupby(opts.partition_cols, dropna=False, sort=True) if opts.partition_cols else [((), df)]

        for key, part in group_iter:
            part = part.reset_index(drop=True)
            part_rows = len(part)
            if part_rows == 0:
                continue

            # build partition path
            part_path = opts.root.rstrip("/")
            if opts.partition_cols:
                # key is scalar for single col, tuple for multi
                key_tuple = key if isinstance(key, tuple) else (key,)
                for col, val in zip(opts.partition_cols, key_tuple):
                    part_path += f"/{col}={_norm_part(val)}"

            # rotate by row or estimated size if requested
            batches: list[pd.DataFrame] = [part]
            if opts.file_row_limit and part_rows > opts.file_row_limit:
                batches = [part.iloc[i:i + opts.file_row_limit] for i in range(0, part_rows, opts.file_row_limit)]
            elif opts.target_file_bytes:
                # rough cut using Arrow in-memory size
                batches = self._split_by_target_bytes(part, opts.target_file_bytes)

            for idx, batch in enumerate(batches):
                # file name
                content_hash = _hash_df(batch) if opts.use_content_hash_in_name else uuid.uuid4().hex
                fname = f"{opts.filename_prefix}-{session_id}-{idx:04d}-{content_hash}{opts.filename_suffix}"
                final_path = f"{part_path}/{fname}"
                tmp_path = f"{part_path}/._tmp_{session_id}_{uuid.uuid4().hex}{opts.filename_suffix}"

                # ensure parent exists (local FS only; cloud is implicit)
                if not _is_cloud(final_path):
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)

                # convert to Arrow table with metadata
                table = pa.Table.from_pandas(batch, preserve_index=False)
                md = {
                    "writer": "ParquetWriter",
                    "session_id": session_id,
                    "session_ts": session_ts,
                    "logical_dataset": opts.logical_dataset or "",
                }
                # merge metadata
                existing_md = table.schema.metadata or {}
                combined = {**existing_md, **{k.encode(): str(v).encode() for k, v in md.items()}}
                table = table.replace_schema_metadata(combined)

                # write to temp then rename
                with fs.open(tmp_path, "wb") as f: # type: ignore
                    pq.write_table(
                        table,
                        f,
                        compression=opts.compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )
                # obtain size
                size = fs.size(tmp_path) # type: ignore
                # atomic rename (same filesystem/prefix)
                fs.rename(tmp_path, final_path) # type: ignore

                stats.files += 1
                stats.rows += len(batch)
                stats.bytes += int(size)
                stats.details.append({
                    "path": final_path,
                    "rows": int(len(batch)),
                    "bytes": int(size),
                })

            stats.partitions += 1

        # optional manifest
        if opts.write_manifest:
            manifest = {
                "session_id": session_id,
                "session_ts": session_ts,
                "root": opts.root,
                "partitions": stats.partitions,
                "files": stats.files,
                "rows": stats.rows,
                "bytes": stats.bytes,
                "details": stats.details[-50:],  # keep last N to avoid giant file
                "partition_cols": opts.partition_cols,
                "compression": opts.compression,
                "dataset": opts.logical_dataset,
            }
            mpath = opts.root.rstrip("/") + "/" + opts.manifest_name
            # append or overwrite with latest session
            existing = []
            try:
                with fs.open(mpath, "rb") as f: # type: ignore
                    existing = json.loads(f.read().decode("utf-8"))
                    if not isinstance(existing, list):
                        existing = [existing]
            except Exception:
                existing = []
            existing.append(manifest)
            tmp_manifest = opts.root.rstrip("/") + f"/._tmp_manifest_{session_id}.json"
            with fs.open(tmp_manifest, "wb") as f: # type: ignore
                f.write(json.dumps(existing, ensure_ascii=False, indent=2).encode("utf-8"))
            fs.rename(tmp_manifest, mpath) # type: ignore

        return stats

    # --------------------- internals ---------------------

    def _split_by_target_bytes(self, df: pd.DataFrame, target_bytes: int) -> list[pd.DataFrame]:
        """
        Greedy split by estimating Arrow size chunk-by-chunk.
        """
        if len(df) <= 1:
            return [df]
        chunks: list[pd.DataFrame] = []
        start = 0
        approx = 0
        # probe a small sample to estimate per-row bytes
        probe_n = min(len(df), 5000)
        probe_tbl = pa.Table.from_pandas(df.head(probe_n), preserve_index=False)
        per_row = max(1, int(probe_tbl.nbytes / max(1, probe_n)))
        rows_per_file = max(1, int(target_bytes / per_row))
        if rows_per_file >= len(df):
            return [df]

        while start < len(df):
            end = min(len(df), start + rows_per_file)
            chunks.append(df.iloc[start:end])
            start = end
        return chunks


# --------------------------- CLI (optional) ------------------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser("parquet_writer")
    ap.add_argument("--input-csv", required=True, help="Path to a CSV to convert into partitioned parquet")
    ap.add_argument("--root", required=True, help="Output root (local or s3:// / gcs://)")
    ap.add_argument("--partitions", nargs="*", default=[], help="Columns to partition by")
    ap.add_argument("--compression", default="zstd")
    ap.add_argument("--row-limit", type=int, default=None)
    ap.add_argument("--target-bytes", type=int, default=None)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    w = ParquetWriter()
    stats = w.write(
        df,
        WriteOptions(
            root=args.root,
            partition_cols=args.partitions,
            compression=args.compression,
            file_row_limit=args.row_limit,
            target_file_bytes=args.target_bytes,
            logical_dataset=args.dataset,
        ),
    )
    print(json.dumps(stats.__dict__, indent=2))

if __name__ == "__main__":
    _cli()