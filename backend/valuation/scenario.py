#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario.py
-----------
Scenario engine for deterministic stresses, parameter sweeps, and Monte Carlo.

You provide:
  - market data (CSV or preloaded DataFrame)
  - a strategy/portfolio callback:  fn(state: MarketState, params: dict) -> dict
    returning at least {"pnl": float} (you can add custom metrics).
  - a scenario spec (dict / YAML / JSON)

Outputs:
  - Per-scenario metrics, aggregated table (CSV/JSON optional)
  - Optional per-path Monte Carlo summaries

Example (Python):
    from scenario import ScenarioEngine, MarketState

    def my_portfolio(state: MarketState, params: dict) -> dict:
        # toy example: DV01 on US10s + equity beta to SPX
        pnl_rates = params.get("dv01_us10", 0.0) * state.shocks.get("rates.us10y_bp", 0.0)
        pnl_eq    = params.get("beta_spx", 0.0)  * state.shocks.get("equity.spx_pct", 0.0)
        return {"pnl": pnl_rates + pnl_eq}

    engine = ScenarioEngine(portfolio_fn=my_portfolio,
                            base_params={"dv01_us10": -120000, "beta_spx": 50000})
    # quick shock
    res = engine.run({"name": "Rates +50bp, SPX -2%", "shocks": {"rates.us10y_bp": 50, "equity.spx_pct": -0.02}})
    print(res)

CLI:
    python scenario.py run --spec scenarios.yaml --out results.csv
"""

from __future__ import annotations
import os
import sys
import json
import math
import time
import argparse
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class MarketData:
    """
    Container for historical/spot market data used by scenarios.
    Columns can include: 'equity.spx', 'fx.usdjpy', 'rates.us10y', 'vol.vix', etc.
    """
    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    @staticmethod
    def from_csv(path: str, parse_dates: bool = True) -> "MarketData":
        df = pd.read_csv(path)
        # try to parse first column as date if looks like a timestamp
        if parse_dates:
            first = df.columns[0]
            try:
                df[first] = pd.to_datetime(df[first])
                df.set_index(first, inplace=True, drop=True)
            except Exception:
                pass
        return MarketData(df=df)

    def latest(self) -> Dict[str, float]:
        if self.df.empty:
            return {}
        row = self.df.iloc[-1].dropna()
        return {str(k): float(v) for k, v in row.items()}


@dataclass
class MarketState:
    """
    The state passed to the portfolio callback.
    """
    spot: Dict[str, float]                      # current spot levels (from MarketData.latest or provided)
    shocks: Dict[str, float]                    # shock dictionary (bp, pct, absolute, you decide by key name)
    path: Optional[pd.DataFrame] = None         # for Monte Carlo: simulated path of levels
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """
    Scenario specification.
    name: display name
    shocks: dict of key -> value (e.g., {"equity.spx_pct": -0.03, "rates.us10y_bp": 100})
    params_override: optional strategy param overrides for this scenario
    mc: Monte Carlo config if present
    sweep: parameter sweep config if present
    """
    name: str
    shocks: Dict[str, float] = field(default_factory=dict)
    params_override: Dict[str, Any] = field(default_factory=dict)
    # Monte Carlo (optional)
    mc: Optional[Dict[str, Any]] = None
    # Parameter sweep (optional)
    sweep: Optional[Dict[str, Any]] = None


# =============================================================================
# Scenario Engine
# =============================================================================

class ScenarioEngine:
    def __init__(
        self,
        portfolio_fn: Callable[[MarketState, Dict[str, Any]], Dict[str, Any]],
        base_spot: Optional[Dict[str, float]] = None,
        base_params: Optional[Dict[str, Any]] = None,
        market: Optional[MarketData] = None,
        random_seed: Optional[int] = 42,
    ):
        """
        portfolio_fn: fn(state, params) -> dict, should include {"pnl": float}
        base_spot: fallback spot dictionary if market not provided
        base_params: default strategy/portfolio params
        market: optional MarketData container
        """
        self.portfolio_fn = portfolio_fn
        self.market = market or MarketData()
        self.base_spot = base_spot or self.market.latest() or {}
        self.base_params = base_params or {}
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

    # --------------------- Deterministic scenario ---------------------

    def run(self, scenario: Union[Scenario, Dict[str, Any]]) -> Dict[str, Any]:
        sc = self._normalize_scenario(scenario)
        params = {**self.base_params, **(sc.params_override or {})}
        state = MarketState(spot=dict(self.base_spot), shocks=dict(sc.shocks), path=None, meta={"scenario": sc.name})
        out = self.portfolio_fn(state, params) or {}
        if "pnl" not in out:
            raise ValueError("portfolio_fn must return a dict with at least {'pnl': float}")
        return {"name": sc.name, "pnl": float(out["pnl"]), **{k: v for k, v in out.items() if k != "pnl"}}

    # --------------------- Parameter sweep ----------------------------

    def run_sweep(self, scenario: Union[Scenario, Dict[str, Any]]) -> pd.DataFrame:
        sc = self._normalize_scenario(scenario)
        if not sc.sweep:
            # run single scenario
            res = self.run(sc)
            return pd.DataFrame([res])
        grid = self._build_grid(sc.sweep)
        rows: List[Dict[str, Any]] = []
        for point in grid:
            params = {**self.base_params, **(sc.params_override or {}), **point}
            state = MarketState(spot=dict(self.base_spot), shocks=dict(sc.shocks), path=None, meta={"scenario": sc.name, "sweep": point})
            out = self.portfolio_fn(state, params) or {}
            if "pnl" not in out:
                raise ValueError("portfolio_fn must return {'pnl': float} in sweep")
            row = {"name": sc.name, "pnl": float(out["pnl"]), **point}
            # include extra metrics
            for k, v in out.items():
                if k != "pnl":
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)

    # --------------------- Monte Carlo --------------------------------

    def run_mc(self, scenario: Union[Scenario, Dict[str, Any]]) -> Dict[str, Any]:
        sc = self._normalize_scenario(scenario)
        if not sc.mc:
            # fall back to deterministic
            r = self.run(sc)
            r.update({"mc": False})
            return r

        cfg = self._norm_mc(sc.mc)
        params = {**self.base_params, **(sc.params_override or {})}
        # build simulated path(s)
        res_pnls: List[float] = []
        extra_metrics: List[Dict[str, Any]] = []
        for i in range(cfg["n_paths"]):
            path = self._simulate_path(cfg, sc.shocks)
            state = MarketState(spot=dict(self.base_spot), shocks=dict(sc.shocks), path=path, meta={"scenario": sc.name, "mc_path": i})
            out = self.portfolio_fn(state, params) or {}
            if "pnl" not in out:
                raise ValueError("portfolio_fn must return {'pnl': float} in MC")
            res_pnls.append(float(out["pnl"]))
            extra_metrics.append({k: v for k, v in out.items() if k != "pnl"})

        pnl_arr = np.array(res_pnls, dtype=float)
        summary = {
            "name": sc.name,
            "mc": True,
            "n_paths": int(cfg["n_paths"]),
            "horizon_days": int(cfg["horizon_days"]),
            "pnl_mean": float(np.mean(pnl_arr)),
            "pnl_std": float(np.std(pnl_arr, ddof=1)) if len(pnl_arr) > 1 else 0.0,
            "pnl_p01": float(np.percentile(pnl_arr, 1)),
            "pnl_p05": float(np.percentile(pnl_arr, 5)),
            "pnl_p50": float(np.percentile(pnl_arr, 50)),
            "pnl_p95": float(np.percentile(pnl_arr, 95)),
            "pnl_p99": float(np.percentile(pnl_arr, 99)),
        }
        # attach optional aggregated extra metrics (means)
        if extra_metrics and any(extra_metrics[0].keys()):
            keys = sorted(set().union(*[m.keys() for m in extra_metrics]))
            for k in keys:
                vals = [m.get(k) for m in extra_metrics if isinstance(m.get(k), (int, float, np.floating))]
                if vals:
                    summary[f"{k}_mean"] = float(np.mean(vals)) # type: ignore
        return summary

    # =============================================================================
    # Helpers
    # =============================================================================

    def _normalize_scenario(self, s: Union[Scenario, Dict[str, Any]]) -> Scenario:
        if isinstance(s, Scenario):
            return s
        if not isinstance(s, dict):
            raise TypeError("scenario must be Scenario or dict")
        return Scenario(
            name=str(s.get("name", "unnamed")),
            shocks=dict(s.get("shocks", {})),
            params_override=dict(s.get("params_override", {})),
            mc=s.get("mc"),
            sweep=s.get("sweep"),
        )

    @staticmethod
    def _build_grid(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        sweep format examples:
          {"beta_spx": {"start": -100000, "stop": 100000, "step": 25000},
           "dv01_us10": [-200000, -150000, -100000, -50000]}
        """
        axes: List[List[Tuple[str, Any]]] = []
        for k, v in sweep.items():
            if isinstance(v, dict) and all(x in v for x in ("start", "stop", "step")):
                vals = np.arange(v["start"], v["stop"] + 1e-12, v["step"]).tolist()
            elif isinstance(v, (list, tuple)):
                vals = list(v)
            else:
                vals = [v]
            axes.append([(k, val) for val in vals])

        # cartesian product
        grid: List[Dict[str, Any]] = [dict()]
        for axis in axes:
            new: List[Dict[str, Any]] = []
            for g in grid:
                for k, val in axis:
                    gg = dict(g); gg[k] = val; new.append(gg)
            grid = new
        return grid

    @staticmethod
    def _norm_mc(mc: Dict[str, Any]) -> Dict[str, Any]:
        """
        mc config:
          {
            "n_paths": 5000,
            "horizon_days": 10,
            "dt": 1.0,
            "drivers": {
              "equity.spx":  {"mu": 0.0, "sigma": 0.02, "kind": "gbm", "shock_key": "equity.spx_pct"},
              "fx.usdjpy":  {"mu": 0.0, "sigma": 0.015, "kind": "gbm", "shock_key": "fx.usdjpy_pct"},
              "rates.us10y": {"mu": 0.0, "sigma": 0.10, "kind": "normal", "shock_key": "rates.us10y_bp", "scale_bp": 100}
            },
            "corr": [[1.0, 0.2, -0.3],
                     [0.2, 1.0, -0.1],
                     [-0.3, -0.1, 1.0]]
          }
        """
        out = dict(mc)
        out.setdefault("n_paths", 2000)
        out.setdefault("horizon_days", 10)
        out.setdefault("dt", 1.0)
        out.setdefault("drivers", {})
        # If corr missing or mis-shaped, use identity
        d = out["drivers"]
        n = len(d)
        C = np.array(out.get("corr") or np.eye(n), dtype=float)
        if C.shape != (n, n):
            C = np.eye(n)
        # Make symmetric PSD via nearest PSD (simple clip)
        C = (C + C.T) / 2.0
        # add a tiny ridge to avoid numerical issues
        C = C + 1e-10 * np.eye(n)
        out["corr"] = C
        return out

    def _simulate_path(self, mc: Dict[str, Any], base_shocks: Dict[str, float]) -> pd.DataFrame:
        """
        Simulate daily shocks for each driver, returning DataFrame with columns == shock_key.
        GBM drivers produce % returns; normal kind can be used for bp shocks (with scale_bp).
        """
        drivers = list(mc["drivers"].items())  # [(name, cfg), ...]
        n = len(drivers)
        T = int(mc["horizon_days"])
        dt = float(mc["dt"])
        mu = np.array([float(cfg.get("mu", 0.0)) for _, cfg in drivers], dtype=float) * dt
        sig = np.array([float(cfg.get("sigma", 0.0)) for _, cfg in drivers], dtype=float) * math.sqrt(dt)

        # Correlated normals
        L = np.linalg.cholesky(mc["corr"])
        Z = np.random.normal(size=(T, n))
        W = Z @ L.T  # T x n

        # Build shocks per driver per day
        data = {}
        for j, (name, cfg) in enumerate(drivers):
            kind = str(cfg.get("kind", "gbm")).lower()
            shock_key = str(cfg.get("shock_key", f"{name}_pct"))
            if kind == "gbm":
                # simple discrete GBM log-return
                r = mu[j] + sig[j] * W[:, j]
                data[shock_key] = r  # daily % change approximation
            else:
                # normal shocks, optionally scaled to bp
                scale_bp = float(cfg.get("scale_bp", 1.0))
                r = mu[j] + sig[j] * W[:, j]
                data[shock_key] = r * scale_bp  # e.g., in bp
        idx = pd.RangeIndex(1, T + 1, name="day")
        df = pd.DataFrame(data, index=idx)

        # Add any deterministic base shocks (applied on day 1)
        if base_shocks:
            first = df.iloc[0].copy()
            for k, v in base_shocks.items():
                if k in df.columns:
                    first[k] += float(v)
                else:
                    # add new column with deterministic shock on day 1; zeros elsewhere
                    df[k] = 0.0
                    first[k] = float(v)
            df.iloc[0] = first

        return df


# =============================================================================
# Spec loading (YAML/JSON)
# =============================================================================

def load_spec(path: str) -> List[Scenario]:
    if path.lower().endswith((".yaml", ".yml")):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML not installed. `pip install pyyaml`")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

    # file can be a dict {"scenarios":[...]} or a list
    items = raw.get("scenarios") if isinstance(raw, dict) else raw
    out: List[Scenario] = []
    for s in items: # type: ignore
        out.append(Scenario(
            name=str(s.get("name", "unnamed")),
            shocks=dict(s.get("shocks", {})),
            params_override=dict(s.get("params_override", {})),
            mc=s.get("mc"),
            sweep=s.get("sweep"),
        ))
    return out


# =============================================================================
# Built-in demo portfolio (optional)
# =============================================================================

def demo_portfolio(state: MarketState, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    A minimal linear risk model to sanity-check the engine.
    Uses keys if present:
      shocks: 'rates.us10y_bp' (bp), 'equity.spx_pct' (%), 'fx.usdjpy_pct' (%), 'vol.vix_abs' (points)
      params: 'dv01_us10', 'beta_spx', 'usd_jpy_exposure', 'vix_vega'
    """
    dv01    = float(params.get("dv01_us10", 0.0))
    beta    = float(params.get("beta_spx", 0.0))
    fx_exp  = float(params.get("usd_jpy_exposure", 0.0))
    vega    = float(params.get("vix_vega", 0.0))

    s = state.shocks
    pnl = 0.0
    pnl += dv01 * float(s.get("rates.us10y_bp", 0.0))            # $/bp
    pnl += beta * float(s.get("equity.spx_pct", 0.0))            # $ per 1.00 == 100%
    pnl += fx_exp * float(s.get("fx.usdjpy_pct", 0.0))           # $ per 1.00 == 100%
    pnl += vega * float(s.get("vol.vix_abs", 0.0))               # $ per 1 vol pt

    # MC path example: sum shocks across days as linear impact
    if state.path is not None and not state.path.empty:
        pnl_path = 0.0
        pnl_path += dv01 * float(state.path["rates.us10y_bp"].sum()) if "rates.us10y_bp" in state.path else 0.0
        pnl_path += beta * float(state.path["equity.spx_pct"].sum()) if "equity.spx_pct" in state.path else 0.0
        pnl_path += fx_exp * float(state.path["fx.usdjpy_pct"].sum()) if "fx.usdjpy_pct" in state.path else 0.0
        pnl += pnl_path

    return {"pnl": pnl, "exposures": {"dv01_us10": dv01, "beta_spx": beta, "usd_jpy_exposure": fx_exp, "vix_vega": vega}}


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Scenario runner (deterministic, sweeps, Monte Carlo)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run scenarios from a YAML/JSON spec")
    p_run.add_argument("--spec", required=True, help="Path to scenarios.yaml/json")
    p_run.add_argument("--out", default=None, help="Write results to CSV/JSON (by extension)")
    p_run.add_argument("--demo", action="store_true", help="Use built-in demo portfolio instead of importing your fn")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed")

    p_quick = sub.add_parser("quick", help="Run a single inline scenario (CLI)")
    p_quick.add_argument("--name", default="inline")
    p_quick.add_argument("--shock", action="append", default=[], help="key=val (e.g., rates.us10y_bp=50, equity.spx_pct=-0.02)")
    p_quick.add_argument("--param", action="append", default=[], help="key=val param override")
    p_quick.add_argument("--mc", action="store_true", help="Enable simple MC with defaults")
    p_quick.add_argument("--out", default=None)

    args = ap.parse_args()

    if args.cmd == "run":
        scenarios = load_spec(args.spec)
        engine = ScenarioEngine(
            portfolio_fn=demo_portfolio if args.demo else demo_portfolio,  # replace with your import
            base_params={}, market=MarketData(), random_seed=args.seed
        )
        rows: List[Dict[str, Any]] = []
        for sc in scenarios:
            if sc.sweep:
                df = engine.run_sweep(sc)
                rows.extend(df.to_dict(orient="records")) # type: ignore
            elif sc.mc:
                rows.append(engine.run_mc(sc))
            else:
                rows.append(engine.run(sc))
        _write_out(rows, args.out)
        _print_preview(rows)
        return

    if args.cmd == "quick":
        shocks = _kv_list_to_dict(args.shock)
        params = _kv_list_to_dict(args.param)
        sc = {"name": args.name, "shocks": shocks}
        engine = ScenarioEngine(portfolio_fn=demo_portfolio, base_params=params)
        res = engine.run_mc(sc) if args.mc else engine.run(sc)
        _write_out([res], args.out)
        _print_preview([res])
        return


def _kv_list_to_dict(items: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for it in items:
        if "=" not in it:
            continue
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except Exception:
            # try int or leave as string (for non-numeric params)
            try:
                out[k.strip()] = int(v)
            except Exception:
                out[k.strip()] = v # type: ignore
    return out

def _write_out(rows: List[Dict[str, Any]], path: Optional[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)

def _print_preview(rows: List[Dict[str, Any]], n: int = 8) -> None:
    import itertools
    for r in itertools.islice(rows, 0, n):
        print(r)


if __name__ == "__main__":
    main()