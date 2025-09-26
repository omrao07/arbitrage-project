"""
io.py
---------
Centralized I/O utilities for loading and saving data files
across the quant research platform (CSV, JSON, YAML).

Ensures consistency and handles common errors (missing dirs, bad encodings).
"""

import os
import json
import yaml
import pandas as pd
from typing import Any, Dict, Optional


# ---------- Path Helpers ----------

def ensure_dir(path: str) -> None:
    """Ensure directory exists before writing a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------- CSV Helpers ----------

def load_csv(path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """
    Load a CSV into a DataFrame.
    Args:
        path: file path
        parse_dates: list of columns to parse as datetime
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, parse_dates=parse_dates) # type: ignore


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Save DataFrame to CSV.
    Args:
        df: dataframe
        path: file path
        index: include row index
    """
    ensure_dir(path)
    df.to_csv(path, index=index)


# ---------- JSON Helpers ----------

def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file into dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save dict to JSON."""
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


# ---------- YAML Helpers ----------

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file into dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str) -> None:
    """Save dict to YAML file."""
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# ---------- Unified Loader ----------

def smart_load(path: str) -> Any:
    """
    Smart loader that infers file type from extension.
    Supports: .csv, .json, .yaml/.yml
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        return load_csv(path)
    elif ext == ".json":
        return load_json(path)
    elif ext in [".yaml", ".yml"]:
        return load_yaml(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def smart_save(obj: Any, path: str) -> None:
    """
    Smart saver that infers file type from extension.
    Supports: DataFrame -> CSV, dict -> JSON/YAML.
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv" and isinstance(obj, pd.DataFrame):
        save_csv(obj, path)
    elif ext == ".json" and isinstance(obj, dict):
        save_json(obj, path)
    elif ext in [".yaml", ".yml"] and isinstance(obj, dict):
        save_yaml(obj, path)
    else:
        raise ValueError(f"Unsupported save for {ext} with type {type(obj)}")


# ---------- Example Usage ----------
if __name__ == "__main__":
    # Demo: load and save CSV
    df = pd.DataFrame({"ticker": ["AAPL", "TSLA"], "PE": [30, 80]})
    save_csv(df, "data/tmp/pe_demo.csv")
    reloaded = load_csv("data/tmp/pe_demo.csv")
    print("✅ CSV roundtrip:\n", reloaded)

    # Demo: JSON
    obj = {"hello": "world", "value": 42}
    save_json(obj, "data/tmp/demo.json")
    print("✅ JSON reload:", load_json("data/tmp/demo.json"))

    # Demo: YAML
    save_yaml(obj, "data/tmp/demo.yaml")
    print("✅ YAML reload:", load_yaml("data/tmp/demo.yaml"))