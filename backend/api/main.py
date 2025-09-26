#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api/main.py
-----------
FastAPI service for running backtests, parameter calibration, and utilities.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Env:
    RUNS_DIR=./runs
    LOG_LEVEL=INFO
"""

from __future__ import annotations

import io
import os
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ---- optional modules (graceful fallbacks) -----------------------------------
try:
    from orchestrator import Orchestrator, load_yaml_or_json, ensure_dir, setup_logging#type:ignore
except Exception as e:
    raise RuntimeError("api requires orchestrator.py on PYTHONPATH") from e

try:
    from calibrate import Calibrator, ParamSpace, TimeSeriesCV#type:ignore
    _HAS_CALIBRATE = True
except Exception:
    _HAS_CALIBRATE = False

try:
    import prompts  # for explain endpoint (optional)#type:ignore
    _HAS_PROMPTS = True
except Exception:
    _HAS_PROMPTS = False

# ------------------------------------------------------------------------------
# App & logging
# ------------------------------------------------------------------------------

APP_VERSION = "0.2.0"
RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs")).resolve()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

RUNS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
logger = logging.getLogger("api")

app = FastAPI(title="Damodar Orchestrator API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------

class InlineData(BaseModel):
    """Inline time series data for backtest."""
    # minimal: either provide `records` or `columns + data`
    price_col: str = Field(default="price", description="Column to use as price")
    timestamp_col: Optional[str] = Field(default=None, description="Timestamp column name (if None, index order used)")
    records: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    data: Optional[List[List[Any]]] = None

    def to_dataframe(self) -> pd.DataFrame:
        if self.records is not None:
            df = pd.DataFrame(self.records)
        elif self.columns is not None and self.data is not None:
            df = pd.DataFrame(self.data, columns=self.columns)
        else:
            raise ValueError("InlineData requires either `records` or (`columns` and `data`).")

        # ensure price present
        if self.price_col not in df.columns:
            # try common names
            for c in ["close", "px", "last"]:
                if c in df.columns:
                    df = df.rename(columns={c: self.price_col})
                    break
        if self.price_col not in df.columns:
            raise ValueError(f"Missing '{self.price_col}' column.")

        # index by timestamp if provided
        if self.timestamp_col and self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df = df.set_index(self.timestamp_col).sort_index()
        else:
            df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
        return df


class BacktestRequest(BaseModel):
    # Provide config/registry either as file paths or inline dicts
    config_path: Optional[str] = None
    registry_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    registry: Optional[Dict[str, Any]] = None

    # Data: use one of csv_path, inline_data, or upload via /backtest/upload
    csv_path: Optional[str] = None
    inline_data: Optional[InlineData] = None

    out_dir: Optional[str] = None

    @validator("config", "registry", pre=True)
    def _empty_to_none(cls, v):
        return None if v in ("", {}, []) else v


class BacktestResponse(BaseModel):
    run_id: str
    history_path: str
    manifest: Dict[str, Any]


class CalibrateRequest(BaseModel):
    # Minimal calibration wrapper; requires calibrate.py
    df_spec: InlineData
    objective_module: Optional[str] = Field(
        default=None,
        description="Python module path with `objective(train_df, val_df, params)`"
    )
    method: str = Field(default="random", description="grid|random|bayes")
    n_trials: int = 100
    maximize: bool = True

    # Param space (simple)
    int_params: Dict[str, List[int]] = Field(default_factory=dict, description="e.g., {'lookback':[5,120]}")
    float_params: Dict[str, List[float]] = Field(default_factory=dict)
    log_float_params: Dict[str, List[float]] = Field(default_factory=dict)
    categorical_params: Dict[str, List[Any]] = Field(default_factory=dict)

    # CV
    n_splits: int = 5
    mode: str = "expanding"
    min_train: int = 252
    step: int = 63
    val_horizon: int = 63


class CalibrateResponse(BaseModel):
    best_params: Dict[str, Any]
    best_score: float
    trials_path: Optional[str] = None
    summary_path: Optional[str] = None


class StrategiesRequest(BaseModel):
    registry_path: Optional[str] = None
    registry: Optional[Dict[str, Any]] = None


class ExplainRequest(BaseModel):
    kind: str = Field(default="strategy", description="strategy|dcf|relative|risk")
    name: str
    context: Dict[str, Any] = Field(default_factory=dict)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _resolve_config_and_registry(cfg_path: Optional[str], reg_path: Optional[str],
                                 cfg_obj: Optional[Dict[str, Any]], reg_obj: Optional[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg_obj is None and cfg_path:
        cfg = load_yaml_or_json(cfg_path)
    else:
        cfg = cfg_obj or {}
    if reg_obj is None and reg_path:
        reg = load_yaml_or_json(reg_path)
    else:
        reg = reg_obj or {}
    if not cfg:
        raise HTTPException(status_code=400, detail="Config missing (provide `config_path` or inline `config`).")
    if not reg:
        raise HTTPException(status_code=400, detail="Registry missing (provide `registry_path` or inline `registry`).")
    return cfg, reg

def _resolve_data(csv_path: Optional[str], inline: Optional[InlineData], upload_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if upload_df is not None:
        return upload_df
    if inline is not None:
        return inline.to_dataframe()
    if csv_path:
        df = pd.read_csv(csv_path)
        # try timestamp column
        for c in ["timestamp", "date", "time", "datetime"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c).sort_index()
                break
        if "price" not in df.columns:
            for c in ["close", "px", "last"]:
                if c in df.columns:
                    df = df.rename(columns={c: "price"})
                    break
        if "price" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'price' (or close/px/last).")
        return df
    raise HTTPException(status_code=400, detail="Provide data via `csv_path`, `inline_data`, or file upload.")

def _new_run_dir(prefix: str = "bt") -> Path:
    rid = uuid.uuid4().hex[:10]
    d = RUNS_DIR / f"{prefix}_{rid}"
    d.mkdir(parents=True, exist_ok=True)
    # local log file for orchestrator internals
    setup_logging(d, LOG_LEVEL)
    return d

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION, "runs_dir": str(RUNS_DIR)}

@app.post("/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest):
    """
    Run a backtest using orchestrator.Orchestrator.
    Supply config/registry via paths or inline, and data via CSV path or inline JSON.
    """
    cfg, reg = _resolve_config_and_registry(req.config_path, req.registry_path, req.config, req.registry)
    run_dir = Path(req.out_dir).resolve() if req.out_dir else _new_run_dir("bt")
    # Prepare data (no uploads on this endpoint)
    df = _resolve_data(req.csv_path, req.inline_data, upload_df=None)

    # Orchestrate
    orch = Orchestrator(config=cfg, registry=reg, out_dir=run_dir, mode="backtest", paper=True)
    res = orch.run_backtest(df)

    return BacktestResponse(run_id=res["manifest"]["run_id"],
                            history_path=res["history_path"],
                            manifest=res["manifest"])

@app.post("/backtest/upload", response_model=BacktestResponse)
async def run_backtest_with_upload(
    file: UploadFile = File(...),
    config_path: Optional[str] = None,
    registry_path: Optional[str] = None,
    out_dir: Optional[str] = None
):
    """
    Upload a CSV file directly. Query params can pass config/registry paths.
    """
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    # timestamp heuristics
    for c in ["timestamp", "date", "time", "datetime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.set_index(c).sort_index()
            break
    if "price" not in df.columns:
        for c in ["close", "px", "last"]:
            if c in df.columns:
                df = df.rename(columns={c: "price"})
                break
    if "price" not in df.columns:
        raise HTTPException(status_code=400, detail="Uploaded CSV must contain 'price' (or close/px/last).")

    if not config_path or not registry_path:
        raise HTTPException(status_code=400, detail="Provide `config_path` and `registry_path` query params.")

    cfg, reg = _resolve_config_and_registry(config_path, registry_path, None, None)
    run_dir = Path(out_dir).resolve() if out_dir else _new_run_dir("bt")
    orch = Orchestrator(config=cfg, registry=reg, out_dir=run_dir, mode="backtest", paper=True)
    res = orch.run_backtest(df)

    return BacktestResponse(run_id=res["manifest"]["run_id"],
                            history_path=res["history_path"],
                            manifest=res["manifest"])

@app.post("/calibrate", response_model=CalibrateResponse)
def calibrate_params(req: CalibrateRequest):
    """
    Parameter search with calibrate.Calibrator (if present).
    """
    if not _HAS_CALIBRATE:
        raise HTTPException(status_code=400, detail="Calibration module not available. Install or include calibrate.py.")

    df = req.df_spec.to_dataframe()

    # Resolve objective
    if not req.objective_module:
        raise HTTPException(status_code=400, detail="Provide `objective_module` path with objective(train_df, val_df, params).")
    try:
        modpath = req.objective_module
        mod, fn = modpath.rsplit(".", 1)
        m = __import__(mod, fromlist=[fn])
        objective_fn = getattr(m, fn)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to import objective: {e}")

    # Build param space & CV
    ispan = {k: (int(v[0]), int(v[1])) for k, v in req.int_params.items()}
    fspan = {k: (float(v[0]), float(v[1])) for k, v in req.float_params.items()}
    lspan = {k: (float(v[0]), float(v[1])) for k, v in req.log_float_params.items()}
    cat   = {k: list(v) for k, v in req.categorical_params.items()}

    space = ParamSpace(int_params=ispan, float_params=fspan, log_float_params=lspan, categorical_params=cat)
    cv = TimeSeriesCV(n_splits=req.n_splits, mode=req.mode, min_train=req.min_train, step=req.step, val_horizon=req.val_horizon)
    cal = Calibrator(objective_fn=objective_fn, space=space, cv=cv, maximize=req.maximize, n_jobs=4, seed=42)

    res = cal.fit(df, method=req.method, n_trials=req.n_trials)
    # persist trials
    out_dir = _new_run_dir("cal")
    cal.save_artifacts(str(out_dir))

    return CalibrateResponse(best_params=res["best_params"], best_score=float(res["best_score"]),
                             trials_path=str(out_dir / "trials.csv"),
                             summary_path=str(out_dir / "summary.json"))

@app.get("/strategies")
def list_strategies(req: StrategiesRequest = Body(default=None)):
    """
    Returns the registry strategies dictionary (names and metadata).
    """
    registry = req.registry if req and req.registry else (
        load_yaml_or_json(req.registry_path) if req and req.registry_path else None
    )
    if not registry:
        raise HTTPException(status_code=400, detail="Provide 'registry' or 'registry_path' in body.")
    reg = registry.get("strategies") or registry
    return {"count": len(reg), "strategies": reg}

@app.post("/explain")
def explain(req: ExplainRequest):
    """
    Lightweight explainer that leverages prompts.py (if present).
    """
    if not _HAS_PROMPTS:
        raise HTTPException(status_code=400, detail="prompts.py not found; remove this endpoint or add the module.")
    kind = req.kind.lower()
    if kind == "strategy":
        text = prompts.build_strategy_prompt(req.name, req.context)
    elif kind in ("dcf", "relative"):
        ticker = req.context.get("ticker", req.name)
        text = prompts.build_valuation_prompt(kind, ticker, req.context)
    elif kind == "risk":
        text = prompts.build_risk_prompt(req.context)
    else:
        raise HTTPException(status_code=400, detail="kind must be one of: strategy|dcf|relative|risk")
    return {"prompt": text}

@app.post("/echo")
def echo(payload: Dict[str, Any]):
    """Debug helper to mirror payloads during integration."""
    return {"received": payload, "ts": time.time()}

# ------------------------------------------------------------------------------
# File upload helper (e.g., OHLC CSV for quick tests)
# ------------------------------------------------------------------------------

@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    path = RUNS_DIR / f"upload_{uuid.uuid4().hex[:8]}.csv"
    df.to_csv(path, index=False)
    return {"path": str(path), "rows": int(len(df))}

# ------------------------------------------------------------------------------
# Entrypoint (for `python api/main.py`)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)