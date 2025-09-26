# utils/storage.py
"""
Unified storage utilities for local, S3, and GCS backends.

Features
--------
- Storage class abstracts local, s3://, and gs:// URIs
- get/put/delete/list operations
- JSON, CSV, and Parquet convenience helpers
- Optional local cache to speed repeated remote reads
- Dependency soft-fails (boto3 / google-cloud-storage / pyarrow optional)

Example
-------
from utils.storage import Storage

store = Storage(cache_dir=".cache")

# Local
store.put("data/local.json", {"hello": "world"})

# S3
df = store.get_parquet("s3://my-bucket/data.parquet")

# GCS
txt = store.get("gs://my-bucket/file.txt").decode()
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any, List, Optional

# Optional deps
try:
    import boto3
    _HAS_BOTO3 = True
except Exception:
    _HAS_BOTO3 = False

try:
    from google.cloud import storage as gcs # type: ignore
    _HAS_GCS = True
except Exception:
    _HAS_GCS = False

try:
    import pandas as pd
    import pyarrow.parquet as pq
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


# ---------------------- Helpers ----------------------

def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")

def _is_gs(path: str) -> bool:
    return str(path).startswith("gs://")

def _strip_scheme(uri: str) -> tuple[str, str]:
    """Return (bucket, key) for s3://bucket/key or gs://bucket/key."""
    no_scheme = uri.split("://", 1)[1]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


# ---------------------- Storage Abstraction ----------------------

class Storage:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Core methods ----
    def get(self, path: str) -> bytes:
        """Fetch object contents (bytes) from local/S3/GCS."""
        if _is_s3(path):
            if not _HAS_BOTO3:
                raise ImportError("boto3 required for S3 access")
            bucket, key = _strip_scheme(path)
            s3 = boto3.client("s3")
            buf = io.BytesIO()
            s3.download_fileobj(bucket, key, buf)
            return buf.getvalue()
        elif _is_gs(path):
            if not _HAS_GCS:
                raise ImportError("google-cloud-storage required for GCS access")
            bucket, key = _strip_scheme(path)
            client = gcs.Client()
            blob = client.bucket(bucket).blob(key)
            return blob.download_as_bytes()
        else:
            with open(path, "rb") as f:
                return f.read()

    def put(self, path: str, data: bytes | str | dict | list, overwrite: bool = True):
        """Upload/store object."""
        if isinstance(data, (dict, list)):
            data = json.dumps(data).encode()
        elif isinstance(data, str):
            data = data.encode()

        if _is_s3(path):
            if not _HAS_BOTO3:
                raise ImportError("boto3 required for S3 access")
            bucket, key = _strip_scheme(path)
            boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=data)
        elif _is_gs(path):
            if not _HAS_GCS:
                raise ImportError("google-cloud-storage required for GCS access")
            bucket, key = _strip_scheme(path)
            blob = gcs.Client().bucket(bucket).blob(key)
            blob.upload_from_string(data)
        else:
            p = pathlib.Path(path)
            if p.exists() and not overwrite:
                raise FileExistsError(f"{path} exists and overwrite=False")
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)

    def delete(self, path: str):
        """Delete object."""
        if _is_s3(path):
            bucket, key = _strip_scheme(path)
            boto3.client("s3").delete_object(Bucket=bucket, Key=key)
        elif _is_gs(path):
            bucket, key = _strip_scheme(path)
            gcs.Client().bucket(bucket).blob(key).delete()
        else:
            pathlib.Path(path).unlink(missing_ok=True)

    def list(self, prefix: str) -> List[str]:
        """List keys/files under a given prefix."""
        if _is_s3(prefix):
            bucket, key = _strip_scheme(prefix)
            resp = boto3.client("s3").list_objects_v2(Bucket=bucket, Prefix=key)
            return [f"s3://{bucket}/{c['Key']}" for c in resp.get("Contents", [])]
        elif _is_gs(prefix):
            bucket, key = _strip_scheme(prefix)
            blobs = gcs.Client().list_blobs(bucket, prefix=key)
            return [f"gs://{bucket}/{b.name}" for b in blobs]
        else:
            p = pathlib.Path(prefix)
            if p.is_file():
                return [str(p)]
            return [str(x) for x in p.glob("**/*") if x.is_file()]

    # ---- Format helpers ----
    def get_json(self, path: str) -> Any:
        return json.loads(self.get(path).decode())

    def put_json(self, path: str, obj: Any, overwrite: bool = True):
        self.put(path, json.dumps(obj, indent=2), overwrite=overwrite)

    def get_parquet(self, path: str):
        if not _HAS_PANDAS:
            raise ImportError("pandas+pyarrow required for parquet")
        data = self.get(path)
        buf = io.BytesIO(data)
        return pd.read_parquet(buf)

    def put_parquet(self, path: str, df, overwrite: bool = True):
        if not _HAS_PANDAS:
            raise ImportError("pandas+pyarrow required for parquet")
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.put(path, buf.getvalue(), overwrite=overwrite)

    # ---- Local cache ----
    def cached_path(self, path: str) -> str:
        """Ensure remote object is cached locally; return local path."""
        if not self.cache_dir:
            raise ValueError("cache_dir not set")
        local_path = self.cache_dir / path.replace("://", "_").replace("/", "_")
        if not local_path.exists():
            data = self.get(path)
            local_path.write_bytes(data)
        return str(local_path)