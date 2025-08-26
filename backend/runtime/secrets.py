# backend/utils/secrets.py
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("secrets")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ============================= loader ==============================

class Secrets:
    """
    Centralized secrets/config loader.

    Precedence:
        1. Runtime env vars (os.environ)
        2. Loaded .env file(s) (backend/.env, project root, etc.)
        3. JSON/YAML overrides (if explicitly set)

    Usage:
        from backend.utils.secrets import secrets
        api_key = secrets.get("BROKER_API_KEY")
        db_url  = secrets.get("DB_URL", "sqlite:///runtime.db")
    """

    def __init__(self):
        self._loaded_env: Dict[str, str] = {}
        self._loaded_files: Dict[str, Dict[str, Any]] = {}
        self._load_dotenv()

    # ------------------- core API -------------------

    def get(self, key: str, default: Optional[Any] = None, required: bool = False) -> Any:
        # 1. system env
        if key in os.environ:
            return os.environ[key]

        # 2. loaded .env
        if key in self._loaded_env:
            return self._loaded_env[key]

        # 3. config files (JSON/YAML)
        for _, m in self._loaded_files.items():
            if key in m:
                return m[key]

        if required and default is None:
            raise KeyError(f"Missing required secret: {key}")
        return default

    def all(self) -> Dict[str, Any]:
        out = {}
        out.update(self._loaded_env)
        for m in self._loaded_files.values():
            out.update(m)
        return out

    # ------------------- loaders --------------------

    def _load_dotenv(self, paths: Optional[list[str]] = None) -> None:
        """
        Naive .env loader (no dependency on python-dotenv).
        Looks for .env in cwd and backend/.
        """
        if paths is None:
            paths = [
                Path(os.getcwd()) / ".env",
                Path(__file__).resolve().parent.parent / ".env",
            ] # type: ignore
        for path in paths: # type: ignore
            if path.exists(): # type: ignore
                try:
                    with open(path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" not in line:
                                continue
                            k, v = line.split("=", 1)
                            self._loaded_env[k.strip()] = v.strip().strip('"').strip("'")
                    log.info("Loaded .env from %s", path)
                except Exception as e:
                    log.warning("Failed to read %s: %s", path, e)

    def load_json(self, path: str, name: Optional[str] = None) -> None:
        try:
            with open(path, "r") as f:
                obj = json.load(f)
            self._loaded_files[name or path] = obj
            log.info("Loaded secrets from %s", path)
        except Exception as e:
            log.warning("Could not load json secrets file %s: %s", path, e)

    # YAML optional
    def load_yaml(self, path: str, name: Optional[str] = None) -> None:
        try:
            import yaml  # pip install pyyaml if you want this
        except ImportError:
            raise RuntimeError("pyyaml not installed; cannot load YAML secrets")
        try:
            with open(path, "r") as f:
                obj = yaml.safe_load(f) or {}
            self._loaded_files[name or path] = obj
            log.info("Loaded secrets from %s", path)
        except Exception as e:
            log.warning("Could not load yaml secrets file %s: %s", path, e)


# ------------------- singleton --------------------
secrets = Secrets()