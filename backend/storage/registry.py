#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
storage/registry.py
-------------------
Persistence layer for strategy registry.

Responsibilities
----------------
- Load/save strategy registry from YAML or JSON
- Atomic save (write temp then rename)
- Simple version stamping and metadata
- In-memory cache to avoid re-loading each time
- Read-only or read/write modes (optional)

The actual registry logic (register, update, query) lives in registry.py.
This module is just the *storage backend*.

Typical usage
-------------
from storage.registry import RegistryStorage
from registry import StrategyRegistry

storage = RegistryStorage("configs/registry.yaml")
reg = storage.load_registry()      # -> StrategyRegistry
# modify registry
reg.register("yen_carry", module="strategies.macro.yen_carry", class_name="YenCarry")
storage.save_registry(reg)
"""

from __future__ import annotations
import os
import json
import yaml
import time
import tempfile
import shutil
from typing import Optional

from registry import StrategyRegistry


class RegistryStorage:
    def __init__(self, path: str, read_only: bool = False):
        self.path = os.path.abspath(path)
        self.read_only = read_only
        self._cache: Optional[StrategyRegistry] = None#type:ignore
        self._last_loaded: Optional[float] = None

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def load_registry(self, force: bool = False) -> StrategyRegistry:#type:ignore
        """
        Load registry into memory. Cached unless force=True.
        """
        if self._cache is not None and not force:
            return self._cache

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Registry file not found: {self.path}")

        ext = os.path.splitext(self.path)[1].lower()
        with open(self.path, "r", encoding="utf-8") as f:
            if ext in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            elif ext == ".json":
                data = json.load(f)
            else:
                raise ValueError("Unsupported registry format (use .yaml/.yml/.json)")

        if not isinstance(data, dict) or "strategies" not in data:
            raise ValueError("Registry file must contain a 'strategies' dict.")

        reg = StrategyRegistry()#type:ignore
        reg._strategies = data["strategies"] or {}
        self._cache = reg
        self._last_loaded = time.time()
        return reg

    def save_registry(self, reg: StrategyRegistry, path: Optional[str] = None) -> None:#type:ignore
        """
        Save registry atomically. Default path = self.path.
        """
        if self.read_only:
            raise PermissionError("RegistryStorage is read-only.")

        out_path = path or self.path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        payload = {
            "strategies": reg.list(),#type:ignore
            "_meta": {
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(reg.list()),#type:ignore
            },
        }

        ext = os.path.splitext(out_path)[1].lower()
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(out_path))
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            if ext in (".yaml", ".yml"):
                yaml.safe_dump(payload, tmp, sort_keys=True, default_flow_style=False)
            elif ext == ".json":
                json.dump(payload, tmp, indent=2)
            else:
                raise ValueError("Unsupported registry format (use .yaml/.yml/.json)")
        shutil.move(tmp_path, out_path)

        # Update cache
        self._cache = reg
        self._last_loaded = time.time()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def last_loaded(self) -> Optional[float]:
        return self._last_loaded

    def path_info(self) -> dict:
        return {
            "path": self.path,
            "exists": os.path.exists(self.path),
            "size": os.path.getsize(self.path) if os.path.exists(self.path) else None,
            "last_loaded": self._last_loaded,
        }


# ----------------------------------------------------------------------
# CLI helper
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Registry storage utility")
    ap.add_argument("--path", required=True, help="Path to registry file")
    ap.add_argument("--info", action="store_true", help="Show file info")
    ap.add_argument("--list", action="store_true", help="List all strategies")
    args = ap.parse_args()

    store = RegistryStorage(args.path)

    if args.info:
        print(store.path_info())
    elif args.list:
        reg = store.load_registry()
        for name, meta in reg.list().items():#type:ignore
            print(f"{name}: {meta}")
    else:
        print("Use --info or --list.")