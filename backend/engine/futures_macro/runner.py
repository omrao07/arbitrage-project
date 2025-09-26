#!/usr/bin/env python3
"""
runner.py
----------
Main orchestrator for strategy execution:
- Loads registry/configs (YAML/CSV).
- Runs signal generation for each strategy family.
- Allocates capital based on weights.
- Routes orders (simulator / broker).
- Logs PnL, risk, diagnostics.

Usage:
    python runner.py --mode auto
    python runner.py --mode manual --strategy momentum

"""

import argparse
import logging
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import engines
from engines.equity_ls.signals import momentum, value, quality, sector_rotation # type: ignore
from engines.stat_arb.signals import pairs, dispersion, overnight_reversal # type: ignore
from engines.futures_macro.signals import breakevens, commodity_spreads # type: ignore
from engines.fx.signals import fx_carry # type: ignore
from engines.rates.signals import yield_curve # type: ignore
from engines.macro.signals import macro_quadrants # type: ignore

from engines.equity_ls.execution.allocator import allocate_from_scores # type: ignore
from engines.equity_ls.execution.order_router import default_router # type: ignore
from engines.futures_macro.execution.allocator import allocate_from_weights as futs_allocator # type: ignore

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_PATH = Path("config/runner_config.yaml")
REGISTRY_PATH = Path("config/strategies_registry.csv")

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {"mode": "manual", "capital": 1_000_000, "families": ["equity_ls", "fx", "rates"]}

def load_registry():
    if REGISTRY_PATH.exists():
        return pd.read_csv(REGISTRY_PATH)
    return pd.DataFrame(columns=["id","name","family","status","params"])

# ---------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------

def run_strategy(strategy_id: str, family: str, cfg: dict, prices: pd.DataFrame, macro: dict):
    """
    Dispatch to correct family signal builder and return weights/orders.
    """
    if family == "equity_ls":
        if strategy_id == "momentum":
            scores = momentum.build_signal(prices, cfg)
        elif strategy_id == "value":
            scores = value.build_signal(prices, cfg)
        elif strategy_id == "quality":
            scores = quality.build_signal(prices, cfg)
        elif strategy_id == "sector_rotation":
            scores = sector_rotation.build_signal(prices, cfg)
        else:
            return None
        w, dollars, shares = allocate_from_scores(scores, prices.iloc[-1], nav=cfg.get("capital",1_000_000))
        return {"weights": w, "orders": shares}

    elif family == "stat_arb":
        if strategy_id == "pairs":
            snap = pairs.build_pairs_snapshot(prices, cfg)
        elif strategy_id == "dispersion":
            snap = dispersion.build_dispersion_snapshot(prices, cfg)
        elif strategy_id == "overnight_reversal":
            snap = overnight_reversal.build_signal(prices, cfg)
        else:
            return None
        return {"weights": snap.get("weights", pd.Series(dtype=float))}

    elif family == "futures_macro":
        if strategy_id == "breakevens":
            snap = breakevens.build_snapshot(prices, cfg)
        elif strategy_id == "commodity_spreads":
            snap = commodity_spreads.build_spread_snapshot(prices=prices, specs=cfg["specs"], spreads=cfg["spreads"], mode=cfg.get("mode","mean_revert"))
        else:
            return None
        return {"weights": snap.get("weights_symbol", pd.Series(dtype=float))}

    elif family == "fx":
        snap = fx_carry.build_fx_carry_weights(
            spot_usd=prices,
            carry_source=cfg.get("carry_source","rates"),
            usd_short_rate=macro.get("usd_rate"),
            ccy_short_rates=macro.get("ccy_rates"),
            forward_points=macro.get("forward_points"),
            cfg=cfg
        )
        return {"weights": snap[0], "diag": snap[1]}

    elif family == "rates":
        snap = yield_curve.build_yield_curve_trades(yields=prices, cfg=cfg, map_cfg=cfg.get("map_cfg"))
        return {"weights": snap.get("contracts_steepener"), "diag": snap["diag"]}

    elif family == "macro":
        snap = macro_quadrants.build_macro_quadrants_snapshot(
            growth=macro["growth"], inflation=macro["inflation"], asset_prices=prices
        )
        return {"weights": snap["weights"], "regime": snap["current_regime"]}

    return None

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["manual","auto"], default="manual")
    parser.add_argument("--strategy", type=str, help="Run only this strategy id")
    args = parser.parse_args()

    config = load_config()
    registry = load_registry()

    mode = args.mode or config.get("mode","manual")
    logging.info(f"Runner starting in mode={mode}")

    # Dummy placeholders: replace with real market data loaders
    prices = pd.DataFrame()    # inject with your daily/hourly OHLC
    macro = {}                 # macro series (rates, inflation, growth, etc.)

    results = {}

    for _, row in registry.iterrows():
        sid, fam, status = row["id"], row["family"], row.get("status","active")
        if status != "active": continue
        if args.strategy and sid != args.strategy: continue

        try:
            logging.info(f"Running {fam}/{sid}...")
            res = run_strategy(sid, fam, config, prices, macro)
            if res: results[sid] = res
        except Exception as e:
            logging.error(f"Strategy {sid} failed: {e}")

    logging.info("Run complete.")
    print("Summary keys:", list(results.keys()))

if __name__ == "__main__":
    main()