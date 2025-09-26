#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orchestrator.py
---------------
Single entry-point to run strategies in backtest or live/paper modes.

Responsibilities
- Load registry + config (YAML/JSON)
- Build StrategyManager, StrategySelector, RiskManager, OrderRouter, Alerting, PnL Attribution
- Orchestrate data feed ticks/batches
- Route signals -> orders -> fills, apply slippage/fees
- Track risk limits & policies, trigger alerts
- Persist artifacts: logs, metrics, run manifest

CLI
----
Backtest:
  python orchestrator.py backtest --config configs/portfolio.yaml --registry configs/registry.yaml --out runs/bt_2024Q4

Live/Paper:
  python orchestrator.py live --config configs/portfolio.yaml --registry configs/registry.yaml --paper --broker paper

Scenario (optional; if scenario.py present):
  python orchestrator.py scenario --spec scenarios.yaml --demo

Notes
- Pure Python, no async requirements.
- PyYAML optional (falls back to JSON).
- Stubs activated if optional modules are missing.
"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import math
import shutil
import signal
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

# -----------------------------------------------------------------------------
# Optional deps
# -----------------------------------------------------------------------------
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Dynamic imports with safe fallbacks
# -----------------------------------------------------------------------------

def _try_import(modname: str, fallback: Any) -> Any:
    try:
        return __import__(modname, fromlist=["*"])
    except Exception:
        return fallback

# --- Fallback stubs (used only if your project modules aren't found) ---------

class _StubStrategyManager:
    """Minimal manager: calls .generate_signals(data, params) on each strategy object."""
    def __init__(self, strategies: Dict[str, Any], params: Dict[str, Dict[str, Any]]):
        self.strategies = strategies
        self.params = params

    def on_bar(self, bar: Dict[str, Any]) -> Dict[str, float]:
        signals = {}
        for name, strat in self.strategies.items():
            p = self.params.get(name, {})
            try:
                signals[name] = float(strat.generate_signal(bar, p))
            except Exception:
                # default: no position change
                signals[name] = 0.0
        return signals

class _StubStrategySelector:
    """Chooses weights from signals; simplistic risk-parity normalization."""
    def choose(self, signals: Dict[str, float], meta: Dict[str, Any]) -> Dict[str, float]:
        x = np.array(list(signals.values()), float)
        if x.size == 0 or np.allclose(x, 0):
            return {k: 0.0 for k in signals}
        w = x / (np.sum(np.abs(x)) + 1e-12)
        return {k: float(v) for k, v in zip(signals.keys(), w)}

class _StubRiskManager:
    """Applies gross/net, per-strategy caps, and volatility targeting."""
    def __init__(self, limits: Dict[str, Any]):
        self.limits = limits or {}

    def apply(self, weights: Dict[str, float], risk_state: Dict[str, Any]) -> Dict[str, float]:
        gross_cap = float(self.limits.get("gross", 1.0))
        net_cap   = float(self.limits.get("net", 1.0))
        vol_tgt   = self.limits.get("vol_target", None)

        # gross/net
        w = dict(weights)
        gross = sum(abs(v) for v in w.values())
        if gross > gross_cap and gross > 0:
            scale = gross_cap / gross
            w = {k: v * scale for k, v in w.items()}
        net = sum(w.values())
        if abs(net) > net_cap and net != 0:
            adj = net_cap / net
            w = {k: v * adj for k, v in w.items()}

        # vol targeting (scalar)
        if isinstance(vol_tgt, (int, float)) and risk_state.get("port_vol"):
            scale = float(vol_tgt) / max(1e-9, float(risk_state["port_vol"]))
            w = {k: v * scale for k, v in w.items()}

        # per-strategy caps
        caps = self.limits.get("per_strategy", {})
        for k, cap in caps.items():
            if k in w:
                w[k] = float(np.clip(w[k], -abs(cap), abs(cap)))
        return w

class _StubOrderRouter:
    """Converts target weights to notional orders; simple fill model."""
    def __init__(self, slippage_bps: float = 0.5, fee_bps: float = 0.1):
        self.slip = slippage_bps / 1e4
        self.fee  = fee_bps / 1e4

    def rebalance(self, px: float, current_pos: float, target_w: float, equity: float) -> Tuple[float, float]:
        target_shares = (target_w * equity) / max(1e-12, px)
        qty = target_shares - current_pos
        # simple fill with slip/fee included in PnL externally
        return float(qty), float(target_shares)

class _StubAlerts:
    def notify(self, level: str, msg: str, **kw):  # noqa
        logging.getLogger("alerts").warning("[%s] %s | %s", level.upper(), msg, kw)

class _StubPnLAttribution:
    def step(self, ret_by_strat: Dict[str, float]) -> Dict[str, float]:
        return ret_by_strat

# Load user modules if available
StrategyManager = _try_import("strategy_manager", _StubStrategyManager)
StrategySelector = _try_import("strategy_selector", _StubStrategySelector)
RiskLimits = _try_import("risk_limits", _StubRiskManager)
RiskPolicies = _try_import("risk_policies", object)  # policies applied inside RiskManager if you have it
OrderRouter = _try_import("order_router", _StubOrderRouter)
Alerts = _try_import("alerts", _StubAlerts)
PnLAttr = _try_import("pnl_attribution", _StubPnLAttribution)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def load_yaml_or_json(path: str | Path) -> Dict[str, Any]:
    path = str(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yaml", ".yml")) and _HAS_YAML:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except Exception:
        if not _HAS_YAML:
            raise RuntimeError("Install PyYAML or provide JSON config.")
        return yaml.safe_load(text)

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def setup_logging(out_dir: Path, level: str = "INFO") -> None:
    log_path = out_dir / "run.log"
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

@dataclass
class Orchestrator:
    config: Dict[str, Any]
    registry: Dict[str, Any]
    out_dir: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    mode: str = "backtest"   # 'backtest' | 'live'
    paper: bool = True
    broker: Optional[str] = None
    _alive: bool = field(default=True, init=False)

    def __post_init__(self):
        self.logger = logging.getLogger("orchestrator")
        self.alerts = Alerts if isinstance(Alerts, _StubAlerts.__class__) else Alerts.Alerts() if hasattr(Alerts, "Alerts") else Alerts#type:ignore
        self.attr = PnLAttr if isinstance(PnLAttr, _StubPnLAttribution.__class__) else PnLAttr.PnLAttribution() if hasattr(PnLAttr, "PnLAttribution") else PnLAttr#type:ignore

        # Build strategies from registry
        self.strategies, self.params = self._build_strategies()

        # Managers
        self.manager = StrategyManager.StrategyManager(self.strategies, self.params) if hasattr(StrategyManager, "StrategyManager") else StrategyManager(self.strategies, self.params)
        self.selector = StrategySelector.StrategySelector() if hasattr(StrategySelector, "StrategySelector") else StrategySelector()
        self.risk = RiskLimits.RiskManager(self.config.get("risk_limits", {})) if hasattr(RiskLimits, "RiskManager") else RiskLimits(self.config.get("risk_limits", {}))
        self.router = OrderRouter.OrderRouter(**(self.config.get("execution", {}) or {})) if hasattr(OrderRouter, "OrderRouter") else OrderRouter(**(self.config.get("execution", {}) or {}))

        # Portfolio state
        self.equity = float(self.config.get("starting_equity", 1_000_000))
        self.positions: Dict[str, float] = {k: 0.0 for k in self.strategies}  # shares per strategy
        self.weights: Dict[str, float] = {k: 0.0 for k in self.strategies}
        self.history: List[Dict[str, Any]] = []

        # Handle SIGINT/SIGTERM for live mode
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        self.logger.info("Initialized Orchestrator run_id=%s mode=%s paper=%s", self.run_id, self.mode, self.paper)

    # ----- Build strategies from registry/config --------------------------------

    def _build_strategies(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Expects registry like:
          strategies:
            momentum_1m: { module: "strategies.momentum", class: "Momentum", params: {...}, universe: "SP500" }
            value_ls:    { module: "strategies.value",    class: "Value",    params: {...} }
        """
        reg = self.registry.get("strategies") or self.registry
        strategies: Dict[str, Any] = {}
        params: Dict[str, Dict[str, Any]] = {}
        for name, meta in reg.items():
            modname = meta.get("module")
            clsname = meta.get("class")
            strat_params = meta.get("params", {})
            if not modname or not clsname:
                # fallback: treat as function-based strategy with `.generate_signal(bar, params)`
                strategies[name] = _FunctionStrategy(meta)
                params[name] = strat_params
                continue
            try:
                mod = __import__(modname, fromlist=[clsname])
                cls = getattr(mod, clsname)
                strategies[name] = cls(**strat_params)
                params[name] = strat_params
            except Exception as e:
                logging.getLogger("orchestrator").exception("Failed to load %s.%s: %s", modname, clsname, e)
                strategies[name] = _NoopStrategy()
                params[name] = {}
        return strategies, params

    # ----- Backtest loop --------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        data columns: must contain 'price' (portfolio ref) and can include feature/news columns your strategies use.
        Index: datetime-like.
        """
        self.logger.info("Backtest start: %d bars", len(data))
        px = data["price"].astype(float)
        equity_curve = [self.equity]

        for ts, row in data.iterrows():
            bar = row.to_dict()
            signals = self.manager.on_bar(bar)  # per-strategy raw signal (e.g., desired direction/score)
            raw_weights = self.selector.choose(signals, meta={"t": ts, "bar": bar})
            # risk state estimate (simple realized vol over window)
            ret_win = pd.Series(equity_curve).pct_change().dropna().tail(63)
            risk_state = {"port_vol": float(ret_win.std(ddof=1) * math.sqrt(252)) if not ret_win.empty else None}
            tgt_weights = self.risk.apply(raw_weights, risk_state=risk_state)

            # rebalance per strategy on reference price
            fills: Dict[str, float] = {}
            for strat, w in tgt_weights.items():
                qty, tgt_pos = self.router.rebalance(px.loc[ts], self.positions.get(strat, 0.0), w, self.equity)#type:ignore
                self.positions[strat] = tgt_pos
                fills[strat] = qty

            # mark-to-market: sum strategy exposures * daily return of reference price
            if len(equity_curve) >= 1:
                r = px.loc[ts] / px.iloc[max(0, px.index.get_loc(ts)-1)] - 1.0 if px.index.get_loc(ts) > 0 else 0.0#type:ignore
            else:
                r = 0.0
            # portfolio return approximated by weight sum * ref return
            port_w = sum(tgt_weights.values())
            pnl = self.equity * port_w * r
            self.equity += pnl

            # bookkeeping
            self.weights = tgt_weights
            rec = {
                "t": ts,
                "price": float(px.loc[ts]),#type:ignore
                "signals": signals,
                "weights": tgt_weights,
                "fills": fills,
                "pnl": float(pnl),
                "equity": float(self.equity),
            }
            self.history.append(rec)

            # alerts: simple drawdown check
            if len(equity_curve) > 50:
                peak = max(equity_curve)
                dd = (self.equity - peak) / peak
                if dd < -float(self.config.get("risk_limits", {}).get("max_drawdown", 0.2)):
                    self.alerts.notify("error", "Max drawdown breached", drawdown=dd, t=str(ts))
                    break

            equity_curve.append(self.equity)

        # results
        hist_df = pd.DataFrame(self.history).set_index("t")
        out = self._finalize_run(hist_df, mode="backtest")
        return out

    # ----- Live/paper loop ------------------------------------------------------

    def run_live(self, feed_iter) -> None:
        """
        feed_iter should yield dict bars with at least {'timestamp', 'price'}.
        """
        self.logger.info("Live start (paper=%s broker=%s)", self.paper, self.broker)
        for bar in feed_iter:
            if not self._alive:
                break
            ts = bar.get("timestamp") or pd.Timestamp.utcnow()
            px = float(bar["price"])
            signals = self.manager.on_bar(bar)
            raw_weights = self.selector.choose(signals, meta={"t": ts, "bar": bar})
            risk_state = {"port_vol": None}  # left to your live risk estimator
            tgt_weights = self.risk.apply(raw_weights, risk_state=risk_state)
            fills = {}
            for strat, w in tgt_weights.items():
                qty, tgt_pos = self.router.rebalance(px, self.positions.get(strat, 0.0), w, self.equity)
                self.positions[strat] = tgt_pos
                fills[strat] = qty
            rec = {
                "t": ts, "price": px, "signals": signals, "weights": tgt_weights, "fills": fills, "equity": float(self.equity)
            }
            self.history.append(rec)
            self.logger.info("Live tick %s | px=%.4f | gross=%0.3f", ts, px, sum(abs(v) for v in tgt_weights.values()))
        # persist on exit
        hist_df = pd.DataFrame(self.history).set_index("t")
        self._finalize_run(hist_df, mode="live")

    # ----- Shared helpers -------------------------------------------------------

    def _finalize_run(self, hist_df: pd.DataFrame, mode: str) -> Dict[str, Any]:
        ensure_dir(self.out_dir)
        # Persist history
        hist_path = self.out_dir / f"history_{self.run_id}.parquet"
        hist_df.to_parquet(hist_path)
        # Manifest
        manifest = {
            "run_id": self.run_id,
            "mode": mode,
            "paper": self.paper,
            "broker": self.broker,
            "start_equity": float(self.config.get("starting_equity", 1_000_000)),
            "end_equity": float(hist_df["equity"].iloc[-1]) if not hist_df.empty else float(self.equity),
            "n_steps": int(len(hist_df)),
            "config": self.config,
            "registry": self.registry,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.logger.info("Finished %s | equity=%.2f | steps=%d | saved: %s", mode, manifest["end_equity"], manifest["n_steps"], hist_path)
        return {"history_path": str(hist_path), "manifest": manifest}

    def _handle_stop(self, *_):
        self._alive = False
        self.logger.info("Received stop signal; will exit after current step.")

# -----------------------------------------------------------------------------
# Minimal function/noop strategy fallbacks
# -----------------------------------------------------------------------------

class _FunctionStrategy:
    """
    Registry entry can be:
      { 'module': 'mymod', 'function': 'my_signal_fn', ... }
    Or:
      { 'function': 'path.to.fn' }
    The function must be fn(bar_dict, params_dict) -> float signal.
    """
    def __init__(self, meta: Dict[str, Any]):
        fn = meta.get("function")
        if fn and "." in fn:
            modname, fname = fn.rsplit(".", 1)
            mod = __import__(modname, fromlist=[fname])
            self._fn = getattr(mod, fname)
        else:
            self._fn = lambda bar, p: 0.0

    def generate_signal(self, bar: Dict[str, Any], params: Dict[str, Any]) -> float:
        try:
            return float(self._fn(bar, params))
        except Exception:
            return 0.0

class _NoopStrategy:
    def generate_signal(self, bar: Dict[str, Any], params: Dict[str, Any]) -> float:
        return 0.0

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _load_data_for_backtest(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Expect either:
      cfg['data']['csv'] with columns [timestamp, price, ...]
    or client-provided loader (extend here if needed).
    """
    data_cfg = cfg.get("data", {})
    if "csv" in data_cfg:
        df = pd.read_csv(data_cfg["csv"])
        # try parsing timestamp
        for c in ["timestamp", "date", "time", "datetime"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c).sort_index()
                break
        if "price" not in df.columns:
            # try common names
            for c in ["close", "px", "last"]:
                if c in df.columns:
                    df = df.rename(columns={c: "price"})
                    break
        if "price" not in df.columns:
            raise ValueError("Backtest CSV must have 'price' (or close/px/last).")
        return df
    raise ValueError("Provide data path in config.data.csv")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Strategy Orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_bt = sub.add_parser("backtest", help="Run backtest")
    p_bt.add_argument("--config", required=True)
    p_bt.add_argument("--registry", required=True)
    p_bt.add_argument("--out", required=True)
    p_bt.add_argument("--log", default="INFO")

    p_live = sub.add_parser("live", help="Run live/paper loop from stdin JSON lines or a feeder module")
    p_live.add_argument("--config", required=True)
    p_live.add_argument("--registry", required=True)
    p_live.add_argument("--out", required=True)
    p_live.add_argument("--paper", action="store_true")
    p_live.add_argument("--broker", default=None)
    p_live.add_argument("--log", default="INFO")

    p_scn = sub.add_parser("scenario", help="Run scenarios via scenario.py (if present)")
    p_scn.add_argument("--spec", required=True)
    p_scn.add_argument("--demo", action="store_true")
    p_scn.add_argument("--out", default=None)

    args = ap.parse_args()

    if args.cmd in {"backtest", "live"}:
        cfg = load_yaml_or_json(args.config)
        reg = load_yaml_or_json(args.registry)
        out_dir = ensure_dir(args.out)
        setup_logging(out_dir, args.log)

        orch = Orchestrator(config=cfg, registry=reg, out_dir=out_dir, mode=args.cmd, paper=getattr(args, "paper", True), broker=getattr(args, "broker", None))

        if args.cmd == "backtest":
            data = _load_data_for_backtest(cfg)
            res = orch.run_backtest(data)
            print(json.dumps(res["manifest"], indent=2, default=str))
        else:
            # Live mode: read bars from stdin as JSON lines if no feeder provided
            def stdin_feed():
                for line in sys.stdin:
                    try:
                        bar = json.loads(line)
                        yield bar
                    except Exception:
                        continue
            orch.run_live(stdin_feed())
            print("Live session finished. Output:", str(orch.out_dir))

    elif args.cmd == "scenario":
        # delegate to scenario.py if available
        try:
            from scenario import ScenarioEngine, load_spec, demo_portfolio, MarketData  # type: ignore
        except Exception:
            print("scenario.py not available in PYTHONPATH.", file=sys.stderr)
            sys.exit(2)
        specs = load_spec(args.spec)
        eng = ScenarioEngine(portfolio_fn=demo_portfolio if args.demo else demo_portfolio)
        rows = []
        for sc in specs:
            if sc.sweep:
                rows.extend(eng.run_sweep(sc).to_dict(orient="records"))
            elif sc.mc:
                rows.append(eng.run_mc(sc))
            else:
                rows.append(eng.run(sc))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            if args.out.lower().endswith(".json"):
                Path(args.out).write_text(json.dumps(rows, indent=2))
            else:
                pd.DataFrame(rows).to_csv(args.out, index=False)
        print(json.dumps(rows[:3], indent=2))  # preview

if __name__ == "__main__":
    main()