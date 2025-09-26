# utils/serializer.py
"""
Serialization / Deserialization Utilities
-----------------------------------------
Central helpers to save/load dict-like objects and lists
to/from JSON, YAML (if available), CSV, or line-delimited JSONL.

Features:
- dumps/loads for in-memory objects (JSON by default)
- dump_file/load_file for persistent storage
- safe roundtrips: ensure_ascii=False, sort_keys=True, UTF-8 everywhere
- fallback to JSON if YAML isn't installed
- helpers for JSONL streaming

Stdlib-only, optional PyYAML support if installed.
"""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any, Dict, Iterable, List, Optional

# Optional YAML
try:
    import yaml
except ImportError:  # fallback
    yaml = None


# ----------------------------- core JSON ---------------------------------

def dumps(obj: Any, fmt: str = "json", **kwargs) -> str:
    """
    Serialize object to string in chosen format (json|yaml).
    """
    fmt = fmt.lower()
    if fmt == "json":
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), **kwargs)
    elif fmt == "yaml":
        if yaml is None:
            raise RuntimeError("PyYAML not available")
        return yaml.safe_dump(obj, sort_keys=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def loads(s: str, fmt: str = "json") -> Any:
    fmt = fmt.lower()
    if fmt == "json":
        return json.loads(s)
    elif fmt == "yaml":
        if yaml is None:
            raise RuntimeError("PyYAML not available")
        return yaml.safe_load(s)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


# ----------------------------- file helpers -------------------------------

def dump_file(obj: Any, path: str, fmt: Optional[str] = None) -> None:
    """
    Write obj to file, format inferred from extension if not given.
    """
    fmt = fmt or _infer_fmt(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps(obj, fmt))

def load_file(path: str, fmt: Optional[str] = None) -> Any:
    fmt = fmt or _infer_fmt(path)
    with open(path, "r", encoding="utf-8") as f:
        return loads(f.read(), fmt)


def _infer_fmt(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        return "yaml"
    return "json"


# ----------------------------- CSV helpers --------------------------------

def dump_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ----------------------------- JSONL helpers -------------------------------

def append_jsonl(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


# ----------------------------- demo ---------------------------------------

if __name__ == "__main__":
    obj = {"b": 2, "a": 1}
    s = dumps(obj)
    print("JSON:", s)
    roundtrip = loads(s)
    print("Roundtrip ok:", roundtrip == obj)

    tmp = "./logs/test.json"
    dump_file(obj, tmp)
    print("Loaded from file:", load_file(tmp))

    rows = [{"id": 1, "val": "x"}, {"id": 2, "val": "y"}]
    dump_csv(rows, "./logs/test.csv")
    print("CSV rows:", load_csv("./logs/test.csv"))

    append_jsonl({"t": 1, "msg": "hi"}, "./logs/test.jsonl")
    print("JSONL rows:", list(iter_jsonl("./logs/test.jsonl")))