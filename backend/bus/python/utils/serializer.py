# bus/python/utils/serializer.py
from __future__ import annotations

import json
import os
import time
import uuid
import pathlib
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union

# ---- Optional deps -----------------------------------------------------------
try:
    import msgpack  # pip install msgpack
except Exception:
    msgpack = None  # graceful fallback

try:
    from fastavro import schemaless_writer, schemaless_reader, parse_schema  # type: ignore # pip install fastavro
except Exception:
    schemaless_writer = None
    schemaless_reader = None
    parse_schema = None

# -----------------------------------------------------------------------------
# Content-type constants (used in message headers across Kafka/NATS/Redis)
# -----------------------------------------------------------------------------
CT_JSON = "application/json"
CT_MSGPACK = "application/msgpack"
CT_AVRO = "avro/binary"

# Header keys we’ll use (case-insensitive on most buses, but keep lowercase)
H_CONTENT_TYPE = "content-type"
H_SCHEMA_NAME = "x-schema-name"      # logical name, e.g., "market/tick"
H_SCHEMA_PATH = "x-schema-path"      # explicit file path to .avsc (optional)
H_EVENT_ID = "x-event-id"
H_ENV = "x-env"
H_TS = "x-ts"

# Default root for your .avsc files (aligns with your repo layout)
DEFAULT_SCHEMA_ROOT = os.getenv("SCHEMA_ROOT", "schemas")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def encode_message(
    obj: Any,
    headers: Optional[Dict[str, str]] = None,
    *,
    content_type: Optional[str] = None,
    schema_name: Optional[str] = None,
    schema_path: Optional[str] = None,
    ensure_ascii: bool = False,
) -> bytes:
    """
    Serialize a Python object into bytes and (optionally) tag headers.

    - JSON (default): no extra deps; smallest friction.
    - MsgPack: if 'msgpack' is installed OR headers request it.
    - Avro (binary): if fastavro is installed AND a schema is provided (by name or path).

    Args:
        obj: dataclass | dict | any JSON-serializable object
        headers: mutable dict; we add content-type, env, ts, event-id (if missing)
        content_type: override ("application/json", "application/msgpack", "avro/binary")
        schema_name: for Avro, e.g., "market/tick" (resolved under SCHEMA_ROOT with ".avsc")
        schema_path: explicit path to .avsc (overrides schema_name)
        ensure_ascii: json.dumps flag

    Returns:
        bytes payload
    """
    h = headers if headers is not None else {}

    # Normalize/convert obj → dict if needed
    payload_obj = _to_plain(obj)

    # Decide content-type
    ct = (content_type or h.get(H_CONTENT_TYPE) or _infer_content_type(h)) or CT_JSON # type: ignore

    # Fill standard envelope headers if absent
    h.setdefault(H_CONTENT_TYPE, ct)
    h.setdefault(H_ENV, os.getenv("APP_ENV", "prod"))
    h.setdefault(H_TS, str(int(time.time() * 1000)))
    h.setdefault(H_EVENT_ID, str(uuid.uuid4()))

    # ---------- AVRO ----------
    if ct == CT_AVRO:
        assert schemaless_writer is not None and parse_schema is not None, (
            "Avro selected but 'fastavro' not installed. `pip install fastavro`"
        )
        schema = _resolve_avro_schema(schema_name=schema_name or h.get(H_SCHEMA_NAME),
                                      schema_path=schema_path or h.get(H_SCHEMA_PATH))
        # Remember what we used so consumers can mirror if needed
        if schema_name and H_SCHEMA_NAME not in h:
            h[H_SCHEMA_NAME] = schema_name
        if schema_path and H_SCHEMA_PATH not in h:
            h[H_SCHEMA_PATH] = schema_path
        # Encode
        import io
        buff = io.BytesIO()
        schemaless_writer(buff, schema, payload_obj)
        return buff.getvalue()

    # ---------- MSGPACK ----------
    if ct == CT_MSGPACK:
        if msgpack is None:
            # Fallback silently to JSON if msgpack not available
            h[H_CONTENT_TYPE] = CT_JSON
        else:
            return msgpack.dumps(payload_obj, use_bin_type=True) # type: ignore

    # ---------- JSON (default) ----------
    return json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=ensure_ascii).encode("utf-8")


def decode_message(
    payload: Optional[bytes],
    headers: Optional[Dict[str, Union[str, bytes]]] = None,
) -> Any:
    """
    Deserialize message bytes according to headers.
    Falls back to JSON, then raw bytes if unknown.

    Args:
        payload: raw bytes (can be None for control messages)
        headers: bus headers dict; may include content-type and schema hints

    Returns:
        Python object (dict/list/primitive), or raw bytes if undecodable.
    """
    if payload is None:
        return None

    # Normalize header keys to lowercase str for lookups
    hdr = _normalize_headers(headers)

    ct = (hdr.get(H_CONTENT_TYPE) or _infer_content_type(hdr) or CT_JSON).lower() # type: ignore

    # ---------- AVRO ----------
    if ct == CT_AVRO:
        assert schemaless_reader is not None and parse_schema is not None, (
            "Avro payload but 'fastavro' not installed. `pip install fastavro`"
        )
        schema = _resolve_avro_schema(schema_name=hdr.get(H_SCHEMA_NAME), schema_path=hdr.get(H_SCHEMA_PATH))
        import io
        buff = io.BytesIO(payload)
        return schemaless_reader(buff, schema)

    # ---------- MSGPACK ----------
    if ct == CT_MSGPACK and msgpack is not None:
        try:
            return msgpack.loads(payload, raw=False)
        except Exception:
            pass  # fall through to JSON

    # ---------- JSON (default) ----------
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception:
        # As a last resort, give raw bytes back to the caller
        return payload


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_plain(obj: Any) -> Any:
    """Convert dataclasses and objects to plain dicts; leave dict/list/primitive as-is."""
    if is_dataclass(obj):
        return asdict(obj) # type: ignore
    if hasattr(obj, "to_dict") and callable(obj.to_dict):  # our event classes
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__") and not isinstance(obj, dict):
        # Avoid encoding large objects with non-serializable fields; best-effort
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj


def _infer_content_type(headers: Optional[Dict[str, Union[str, bytes]]]) -> Optional[str]:
    """Try to infer content type from existing headers (if any)."""
    if not headers:
        return None
    # Accept common alternatives
    for k, v in headers.items():
        key = k.decode().lower() if isinstance(k, (bytes, bytearray)) else str(k).lower()
        if key in (H_CONTENT_TYPE, "content_type", "ct"):
            val = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
            return val.lower()
    return None


def _normalize_headers(headers: Optional[Dict[str, Union[str, bytes]]]) -> Dict[str, str]:
    """Lowercase keys; decode bytes; keep string values."""
    if not headers:
        return {}
    norm: Dict[str, str] = {}
    for k, v in headers.items():
        kk = k.decode().lower() if isinstance(k, (bytes, bytearray)) else str(k).lower()
        if isinstance(v, (bytes, bytearray)):
            try:
                norm[kk] = v.decode("utf-8")
            except Exception:
                norm[kk] = repr(v)
        else:
            norm[kk] = str(v)
    return norm


# ---------- Avro schema loading (local) ----------
@lru_cache(maxsize=256)
def _resolve_avro_schema(schema_name: Optional[str] = None, schema_path: Optional[str] = None) -> dict:
    """
    Load and parse a local .avsc schema for fastavro.
    Priority:
        1) explicit schema_path
        2) schema_name resolved under SCHEMA_ROOT (e.g., 'market/tick' -> schemas/market/tick.avsc)
    """
    if parse_schema is None:
        raise RuntimeError("Avro requested but 'fastavro' is not installed.")

    if schema_path:
        path = pathlib.Path(schema_path)
        if not path.suffix:
            path = path.with_suffix(".avsc")
        if not path.exists():
            raise FileNotFoundError(f"Avro schema not found at path: {path}")
        with path.open("r", encoding="utf-8") as f:
            return parse_schema(json.load(f))

    if schema_name:
        root = pathlib.Path(DEFAULT_SCHEMA_ROOT)
        path = (root / f"{schema_name}.avsc") if not schema_name.endswith(".avsc") else (root / schema_name)
        if not path.exists():
            # try without joining root if schema_name was actually a full path
            alt = pathlib.Path(schema_name)
            if not alt.suffix:
                alt = alt.with_suffix(".avsc")
            if alt.exists():
                with alt.open("r", encoding="utf-8") as f:
                    return parse_schema(json.load(f))
            raise FileNotFoundError(f"Avro schema '{schema_name}' not found under {root} (looked for {path})")
        with path.open("r", encoding="utf-8") as f:
            return parse_schema(json.load(f))

    raise ValueError("Avro content-type selected but neither 'schema_name' nor 'schema_path' provided.")


# -----------------------------------------------------------------------------
# Convenience utilities
# -----------------------------------------------------------------------------
def wants_avro(headers: Optional[Dict[str, str]] = None) -> bool:
    """Return True if the caller asked for Avro via headers or env default."""
    hct = (headers or {}).get(H_CONTENT_TYPE, "").lower()
    if hct == CT_AVRO:
        return True
    env_ct = os.getenv("DEFAULT_CONTENT_TYPE", "").lower()
    return env_ct == CT_AVRO


def wants_msgpack(headers: Optional[Dict[str, str]] = None) -> bool:
    """Return True if MsgPack is requested and installed."""
    hct = (headers or {}).get(H_CONTENT_TYPE, "").lower()
    if hct == CT_MSGPACK and msgpack is not None:
        return True
    env_ct = os.getenv("DEFAULT_CONTENT_TYPE", "").lower()
    return env_ct == CT_MSGPACK and msgpack is not None


def default_headers(
    content_type: Optional[str] = None,
    schema_name: Optional[str] = None,
    schema_path: Optional[str] = None,
) -> Dict[str, str]:
    """Produce a sane headers baseline to attach to outgoing messages."""
    h = {
        H_CONTENT_TYPE: content_type or os.getenv("DEFAULT_CONTENT_TYPE", CT_JSON),
        H_ENV: os.getenv("APP_ENV", "prod"),
        H_TS: str(int(time.time() * 1000)),
        H_EVENT_ID: str(uuid.uuid4()),
    }
    if schema_name:
        h[H_SCHEMA_NAME] = schema_name
    if schema_path:
        h[H_SCHEMA_PATH] = schema_path
    return h


# -----------------------------------------------------------------------------
# Example CLI for quick manual test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {"event_type": "tick", "symbol": "AAPL", "price": 190.25}

    # JSON round-trip
    h = default_headers()
    b = encode_message(sample, headers=h)
    print("JSON bytes:", b[:60], "...", "headers:", h)
    print("Decoded:", decode_message(b, h)) # type: ignore

    # MsgPack round-trip (if available)
    if msgpack:
        h2 = default_headers(content_type=CT_MSGPACK)
        b2 = encode_message(sample, headers=h2)
        print("MSGPACK bytes:", b2[:10], "...", "headers:", h2)
        print("Decoded:", decode_message(b2, h2)) # type: ignore

    # Avro round-trip (if fastavro + schema)
    # Expect schema at: schemas/market/tick.avsc (or provide SCHEMA_ROOT)
    if schemaless_writer and pathlib.Path(DEFAULT_SCHEMA_ROOT, "market", "tick.avsc").exists():
        h3 = default_headers(content_type=CT_AVRO, schema_name="market/tick")
        b3 = encode_message(sample, headers=h3, schema_name="market/tick")
        print("AVRO bytes:", b3[:10], "...", "headers:", h3)
        print("Decoded:", decode_message(b3, h3)) # type: ignore