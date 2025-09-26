# simulation-farm/utils/serializer.py
"""
serializer.py
=============

Unified serialization helpers for Simulation Farm.

Features
--------
- JSON dump/load with support for numpy, pandas, dataclasses, enums, Path, datetime
- YAML dump/load (PyYAML)
- Optional gzip compression for any text/binary
- MessagePack (optional) for compact binary blobs
- DataFrame/Series CSV & Parquet helpers (optional deps)
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import gzip
import io
import json
import pathlib
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

# Optional deps
try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None

try:
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover
    _pd = None

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("serializer.py requires PyYAML. Install with: pip install pyyaml") from e

try:
    import msgpack  # type: ignore
except Exception:
    msgpack = None  # optional

Jsonish = Union[dict, list, str, int, float, bool, None]


# -------------------------------------------------------------------
# JSON support
# -------------------------------------------------------------------

class _EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy/pandas/dataclasses/enums/datetime/Path."""
    def default(self, o: Any) -> Any:  # noqa: N802
        # dataclasses
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o) # type: ignore

        # pathlib paths
        if isinstance(o, (Path, pathlib.PurePath)):
            return str(o)

        # datetime / date / time
        if isinstance(o, (_dt.datetime, _dt.date, _dt.time)):
            # Use ISO format; ensure timezone-aware datetimes stay ISO 8601
            try:
                return o.isoformat()
            except Exception:
                return str(o)

        # enums
        if isinstance(o, enum.Enum):
            return o.value

        # numpy scalars / arrays
        if _np is not None:
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()

        # pandas Series / DataFrame / Timestamp
        if _pd is not None:
            if isinstance(o, _pd.DataFrame):
                # Use records to keep columns visible & order stable
                return {
                    "__kind__": "DataFrame",
                    "columns": list(map(str, o.columns)),
                    "data": o.to_dict(orient="records"),
                }
            if isinstance(o, _pd.Series):
                return {"__kind__": "Series", "name": str(o.name), "data": o.to_dict()}
            if isinstance(o, _pd.Timestamp):
                return o.isoformat()

        # fallback
        try:
            return super().default(o)
        except TypeError:
            return repr(o)


def to_json(obj: Any, *, indent: int | None = 2, sort_keys: bool = False) -> str:
    """Serialize object to a JSON string with enhanced encoder."""
    return json.dumps(obj, cls=_EnhancedJSONEncoder, indent=indent, sort_keys=sort_keys, ensure_ascii=False)


def dump_json(obj: Any, path: Union[str, Path], *, indent: int | None = 2, sort_keys: bool = False, gzip_compress: bool = False) -> Path:
    """Write JSON to file. If gzip_compress=True, writes .gz."""
    p = _ensure_path(path, auto_suffix=".json.gz" if gzip_compress else ".json")
    data = to_json(obj, indent=indent, sort_keys=sort_keys).encode("utf-8")
    if gzip_compress:
        with gzip.open(p, "wb") as f:
            f.write(data)
    else:
        p.write_bytes(data)
    return p


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON from file (supports .gz)."""
    p = Path(path)
    if str(p).endswith(".gz"):
        with gzip.open(p, "rb") as f:
            return json.loads(f.read().decode("utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))


# -------------------------------------------------------------------
# YAML support
# -------------------------------------------------------------------

def dump_yaml(obj: Any, path: Union[str, Path], *, gzip_compress: bool = False) -> Path:
    """Write YAML to file. If gzip_compress=True, writes .gz."""
    p = _ensure_path(path, auto_suffix=".yaml.gz" if gzip_compress else ".yaml")
    text = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    if gzip_compress:
        with gzip.open(p, "wb") as f:
            f.write(text.encode("utf-8"))
    else:
        p.write_text(text, encoding="utf-8")
    return p


def load_yaml(path: Union[str, Path]) -> Any:
    """Load YAML from file (supports .gz)."""
    p = Path(path)
    if str(p).endswith(".gz"):
        with gzip.open(p, "rb") as f:
            return yaml.safe_load(f.read().decode("utf-8"))
    return yaml.safe_load(p.read_text(encoding="utf-8"))


# -------------------------------------------------------------------
# MessagePack (optional)
# -------------------------------------------------------------------

def dump_msgpack(obj: Any, path: Union[str, Path]) -> Optional[Path]:
    """Write object as MessagePack (requires msgpack). Returns None if not available."""
    if msgpack is None:
        return None
    p = _ensure_path(path, auto_suffix=".msgpack")
    with open(p, "wb") as f:
        msgpack.pack(obj, f, use_bin_type=True, default=_msgpack_default)
    return p


def load_msgpack(path: Union[str, Path]) -> Any:
    """Load MessagePack object. Raises if msgpack not installed."""
    if msgpack is None:
        raise RuntimeError("msgpack is not installed. `pip install msgpack`")
    with open(path, "rb") as f:
        return msgpack.unpack(f, raw=False)


def _msgpack_default(o: Any) -> Any:
    # Provide basic numpy/pandas support for msgpack
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)# type: ignore
    if isinstance(o, (Path, pathlib.PurePath)):
        return str(o)
    if isinstance(o, (_dt.datetime, _dt.date, _dt.time)):
        return o.isoformat()
    if isinstance(o, enum.Enum):
        return o.value
    if _np is not None:
        if isinstance(o, _np.generic):
            return o.item()
        if isinstance(o, _np.ndarray):
            return o.tolist()
    if _pd is not None:
        if isinstance(o, _pd.Timestamp):
            return o.isoformat()
        if isinstance(o, _pd.Series):
            return {"__kind__": "Series", "name": str(o.name), "data": o.to_dict()}
        if isinstance(o, _pd.DataFrame):
            return {"__kind__": "DataFrame", "columns": list(map(str, o.columns)), "data": o.to_dict(orient="records")}
    raise TypeError(f"Cannot serialize type {type(o)} to msgpack")


# -------------------------------------------------------------------
# CSV / Parquet helpers (optional)
# -------------------------------------------------------------------

def df_to_csv(df: "pd.DataFrame", path: Union[str, Path], *, index: bool = False) -> Path:# type: ignore
    """Save DataFrame to CSV (requires pandas)."""
    if _pd is None:
        raise RuntimeError("pandas is not installed. `pip install pandas`")
    p = _ensure_path(path, auto_suffix=".csv")
    df.to_csv(p, index=index)
    return p


def df_to_parquet(df: "pd.DataFrame", path: Union[str, Path], *, compression: str = "snappy") -> Path:# type: ignore
    """Save DataFrame to Parquet (requires pandas + pyarrow/fastparquet)."""# type: ignore
    if _pd is None:
        raise RuntimeError("pandas is not installed. `pip install pandas`")
    p = _ensure_path(path, auto_suffix=".parquet")
    df.to_parquet(p, compression=compression)
    return p


def load_csv(path: Union[str, Path]) -> "._pd.DataFrame":# type: ignore
    if _pd is None:
        raise RuntimeError("pandas is not installed. `pip install pandas`")
    return _pd.read_csv(path)


def load_parquet(path: Union[str, Path]) -> "pd.DataFrame":# type: ignore
    if _pd is None:
        raise RuntimeError("pandas is not installed. `pip install pandas`")
    return _pd.read_parquet(path)


# -------------------------------------------------------------------
# Bytes helpers
# -------------------------------------------------------------------

def write_bytes(data: bytes, path: Union[str, Path], *, gzip_compress: bool = False) -> Path:
    p = _ensure_path(path, auto_suffix=".gz" if gzip_compress else None)
    if gzip_compress:
        with gzip.open(p, "wb") as f:
            f.write(data)
    else:
        p.write_bytes(data)
    return p


def read_bytes(path: Union[str, Path]) -> bytes:
    p = Path(path)
    if str(p).endswith(".gz"):
        with gzip.open(p, "rb") as f:
            return f.read()
    return p.read_bytes()


# -------------------------------------------------------------------
# Path utilities
# -------------------------------------------------------------------

def _ensure_path(path: Union[str, Path], *, auto_suffix: Optional[str] = None) -> Path:
    """Ensure parent dirs exist and return a Path, optionally appending a suffix if not present."""
    p = Path(path)
    if auto_suffix and not str(p).endswith(auto_suffix):
        # only add if no known suffix present
        if not p.suffix:
            p = p.with_suffix(auto_suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------------------------------------------------
# Convenience round-trippers for specs / configs
# -------------------------------------------------------------------

def save_spec_json(spec_obj: Any, path: Union[str, Path], *, gzip_compress: bool = False) -> Path:
    """Serialize a dataclass (or dict-like) spec to JSON."""
    return dump_json(spec_obj, path, indent=2, sort_keys=False, gzip_compress=gzip_compress)


def load_spec_json(path: Union[str, Path]) -> Any:
    """Load a spec JSON and return dict."""
    return load_json(path)


def save_config_yaml(cfg: Dict[str, Any], path: Union[str, Path], *, gzip_compress: bool = False) -> Path:
    """Save YAML config dict to disk."""
    return dump_yaml(cfg, path, gzip_compress=gzip_compress)


def load_config_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML config dict from disk."""
    obj = load_yaml(path)
    return obj or {}