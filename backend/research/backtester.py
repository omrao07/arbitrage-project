#!/usr/bin/env python3
"""
backtester.py — clean, production-ready single-asset backtester with:
- CSV or yfinance loader
- Strategy registry (SMA crossover, RSI mean-revert, Donchian breakout, Buy&Hold)
- Simple execution model (next-bar close), commissions + slippage in bps
- Position sizing via leverage on {-1,0,1} signals
- Walk-forward grid search (optional)
- CLI that saves ledger.csv, trades.csv, metrics.json, config.json

Usage (CSV):
  python backtester.py --source ./data.csv --strategy sma_cross --params '{"fast":10,"slow":50}'

Usage (yfinance):
  python backtester.py --source yfinance --symbol SPY --start 2010-01-01 --strategy rsi_revert --params '{"period":14}'

Walk-forward example:
  python backtester.py --source yfinance --symbol SPY --start 2010-01-01 \
    --strategy sma_cross \
    --walkforward '{"train_window":756,"test_window":126,"param_grid":{"fast":[5,10,20],"slow":[50,100,150]}}'
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Strategy Interfaces
# =========================
class Strategy:
    """Base Strategy interface."""

    def name(self) -> str:
        return self.__class__.__name__

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """
        Return a signal Series aligned to df.index with values in {-1, 0, +1}.
        Must not peek into the future.
        """
        raise NotImplementedError


class BuyHold(Strategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        s = pd.Series(1, index=df.index, name="signal")
        s.iloc[:1] = 0
        return s


class SMACross(Strategy):
    def generate_signals(self, df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
        c = df["Close"].astype(float)
        f = c.rolling(fast, min_periods=1).mean()
        s = c.rolling(slow, min_periods=1).mean()
        raw = np.where(f > s, 1, -1)
        sig = pd.Series(raw, index=df.index, name="signal")
        # warmup: ensure we don't trade before slow MA is formed
        sig.iloc[: max(1, slow)] = 0
        return sig


class RSIRevert(Strategy):
    def generate_signals(
        self,
        df: pd.DataFrame,
        period: int = 14,
        low: int = 30,
        high: int = 70,
    ) -> pd.Series:
        c = df["Close"].astype(float)
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        raw = np.where(rsi < low, 1, np.where(rsi > high, -1, 0))
        sig = pd.Series(raw, index=df.index, name="signal")
        sig.iloc[: period] = 0
        return sig


class DonchianBreakout(Strategy):
    def generate_signals(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        high_n = df["High"].rolling(lookback, min_periods=lookback).max()
        low_n = df["Low"].rolling(lookback, min_periods=lookback).min()
        c = df["Close"]
        raw = np.where(c > high_n.shift(1), 1, np.where(c < low_n.shift(1), -1, 0))
        sig = pd.Series(raw, index=df.index, name="signal")
        sig.iloc[: lookback] = 0
        return sig


STRATEGY_REGISTRY: Dict[str, Strategy] = {
    "buy_hold": BuyHold(),
    "sma_cross": SMACross(),
    "rsi_revert": RSIRevert(),
    "donchian": DonchianBreakout(),
}


# =========================
# Utilities
# =========================
def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            raise ValueError("Data must have a DatetimeIndex or a 'Date' column.")
    return df.sort_index()


def load_data(source: str, symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    Load OHLCV data.
    - If source endswith .csv: read CSV (expects Date, Open, High, Low, Close, Volume).
    - Else if source == 'yfinance': fetch via yfinance (if installed).
    """
    if source.lower().endswith(".csv"):
        df = pd.read_csv(source)
        df = ensure_datetime_index(df)
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        return df[cols].astype(float)

    if source.lower() == "yfinance":
        try:
            import yfinance as yf  # type: ignore
        except Exception as e:
            raise SystemExit("yfinance not installed. Install with: pip install yfinance") from e
        if not symbol:
            raise ValueError("When --source yfinance, you must pass --symbol")
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty: # type: ignore
            raise ValueError("No data returned—check symbol/dates.")
        # yfinance returns columns like 'Open','High','Low','Close','Adj Close','Volume'
        df = df.rename(columns=str.title) # type: ignore
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    raise ValueError("Unsupported source. Use a CSV file path or 'yfinance'.")


def position_sizer(signal: pd.Series, max_leverage: float = 1.0) -> pd.Series:
    """Map {-1,0,1} to target position in [-lev, +lev]."""
    return (signal.clip(-1, 1) * float(max_leverage)).rename("target_pos")


@dataclass
class Fees:
    commission_bps: float = 1.0  # per trade notional
    slippage_bps: float = 0.0    # per trade notional


def apply_execution(
    df: pd.DataFrame,
    target_pos: pd.Series,
    fees: Fees,
    allow_short: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple daily execution:
    - Trades are changes in target position.
    - Assume fill at same-day close (no look-ahead in signal creation).
    - Commission + slippage cost applied on notional traded.
    Returns (ledger, trades).
    """
    px = df["Close"].astype(float)
    target_pos = target_pos.reindex(px.index).fillna(0.0).clip(-1.0 if allow_short else 0.0, 1.0)

    # Daily return
    ret = px.pct_change().fillna(0.0)

    # Hold yesterday's target across today's return
    pos = target_pos.shift(1).fillna(0.0)
    trade = target_pos.diff().fillna(target_pos.iloc[0])

    # Costs
    notional_traded = (trade.abs() * px)
    trans_cost = notional_traded * ((fees.commission_bps + fees.slippage_bps) / 1e4)

    gross_pnl = pos * ret  # equity fraction P&L
    cost_pnl = trans_cost / px  # convert cost to equity fraction
    net_pnl = gross_pnl - cost_pnl

    equity = (1.0 + net_pnl).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    ledger = pd.DataFrame(
        {
            "price": px,
            "ret": ret,
            "pos": pos,
            "trade": trade,
            "gross_pnl": gross_pnl,
            "cost_pnl": -cost_pnl,
            "net_pnl": net_pnl,
            "equity": equity,
            "drawdown": drawdown,
        }
    )

    trades = pd.DataFrame(
        {
            "trade": trade[trade != 0.0],
            "price": px[trade != 0.0],
            "notional": notional_traded[trade != 0.0],
            "cost": trans_cost[trade != 0.0],
        }
    )
    return ledger, trades


def metrics(equity: pd.Series, ret: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> Dict[str, float]:
    eq = equity.dropna().astype(float)
    rets = ret.dropna().astype(float)

    total_return = float(eq.iloc[-1] - 1.0)
    span_days = max((eq.index[-1] - eq.index[0]).days, 1) # type: ignore
    years = span_days / 365.25
    cagr = float((eq.iloc[-1]) ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0

    vol = float(rets.std() * np.sqrt(periods_per_year))
    sharpe = float((rets.mean() * periods_per_year - rf) / (vol + 1e-12))
    mdd = float((eq / eq.cummax() - 1.0).min())
    calmar = float(cagr / abs(mdd)) if mdd != 0 else float("nan")
    win_rate = float((rets > 0).mean())
    avg_abs_turnover = float(rets.size and (np.abs(np.diff(equity.values)).mean())) # type: ignore

    return {
        "TOTAL_RETURN": total_return,
        "CAGR": cagr,
        "VOL": vol,
        "SHARPE": sharpe,
        "MAX_DRAWDOWN": mdd,
        "CALMAR": calmar,
        "WIN_RATE": win_rate,
        "AVG_ABS_EQUITY_MOVE": avg_abs_turnover,
    }


# =========================
# Walk-Forward (optional)
# =========================
def param_grid_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = [{}]
    for k, vals in grid.items():
        combos = [dict(c, **{k: v}) for c in combos for v in vals]
    return combos


def walk_forward(
    df: pd.DataFrame,
    strategy: Strategy,
    param_grid: Dict[str, List[Any]],
    train_window: int,
    test_window: int,
    fees: Fees,
    allow_short: bool,
    max_leverage: float,
) -> Dict[str, Any]:
    results: List[pd.DataFrame] = []
    params_used: List[Dict[str, Any]] = []

    start = train_window
    while start + test_window <= len(df):
        train = df.iloc[start - train_window:start]
        test = df.iloc[start : start + test_window]

        # grid search on train by Sharpe
        best_params: Dict[str, Any] = {}
        best_score = -np.inf
        for p in param_grid_combinations(param_grid):
            sig = strategy.generate_signals(train, **p)
            tgt = position_sizer(sig, max_leverage=max_leverage)
            led, _ = apply_execution(train, tgt, fees=fees, allow_short=allow_short)
            m = metrics(led["equity"], led["net_pnl"])
            if m["SHARPE"] > best_score:
                best_score = m["SHARPE"]
                best_params = p

        # apply to test
        sig_t = strategy.generate_signals(test, **best_params)
        tgt_t = position_sizer(sig_t, max_leverage=max_leverage)
        led_t, _ = apply_execution(test, tgt_t, fees=fees, allow_short=allow_short)
        led_t["best_params"] = json.dumps(best_params)
        results.append(led_t)
        params_used.append(best_params)

        start += test_window

    if not results:
        raise ValueError("Walk-forward produced no splits—check window sizes vs data length.")

    joined = pd.concat(results)
    return {
        "ledger": joined,
        "params_used": params_used,
        "wf_metrics": metrics(joined["equity"], joined["net_pnl"]),
    }


# =========================
# Runner
# =========================
def run_backtest(
    data: pd.DataFrame,
    strategy_name: str,
    strategy_params: Dict[str, Any],
    fees: Fees,
    allow_short: bool = True,
    max_leverage: float = 1.0,
    walkforward: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = ensure_datetime_index(data)

    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {list(STRATEGY_REGISTRY.keys())}")

    strat = STRATEGY_REGISTRY[strategy_name]

    if walkforward:
        wf = walk_forward(
            data,
            strategy=strat,
            param_grid=walkforward["param_grid"],
            train_window=int(walkforward["train_window"]),
            test_window=int(walkforward["test_window"]),
            fees=fees,
            allow_short=allow_short,
            max_leverage=max_leverage,
        )
        ledger = wf["ledger"]
        ret = ledger["net_pnl"]
        meta = wf["wf_metrics"]
        trades = pd.DataFrame()
    else:
        sig = strat.generate_signals(data, **strategy_params)
        tgt = position_sizer(sig, max_leverage=max_leverage)
        ledger, trades = apply_execution(data, tgt, fees=fees, allow_short=allow_short)
        ret = ledger["net_pnl"]
        meta = metrics(ledger["equity"], ret)

    return {
        "ledger": ledger,
        "trades": trades,
        "metrics": meta,
        "config": {
            "strategy": strategy_name,
            "strategy_params": strategy_params,
            "fees": asdict(fees),
            "allow_short": allow_short,
            "max_leverage": max_leverage,
            "walkforward": bool(walkforward),
        },
    }


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple, reliable backtester.py")
    p.add_argument("--source", required=True, help="CSV path or 'yfinance'")
    p.add_argument("--symbol", default="", help="Ticker if source is yfinance")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p.add_argument("--outdir", default="bt_results", help="Output directory")
    p.add_argument("--strategy", default="sma_cross", choices=list(STRATEGY_REGISTRY.keys()))
    p.add_argument("--params", default="{}", help='JSON dict, e.g. {"fast":10,"slow":50}')
    p.add_argument("--fees", default='{"commission_bps":1.0,"slippage_bps":0.0}', help="JSON dict of bps costs")
    p.add_argument("--no-short", action="store_true", help="Disable shorting")
    p.add_argument("--leverage", type=float, default=1.0, help="Max leverage (abs position)")
    p.add_argument(
        "--walkforward",
        default="",
        help='JSON like {"train_window":504,"test_window":126,"param_grid":{"fast":[5,10],"slow":[50,100]}}',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.source, args.symbol, args.start, args.end)

    fees = Fees(**json.loads(args.fees))
    wf = json.loads(args.walkforward) if args.walkforward else None

    res = run_backtest(
        df,
        strategy_name=args.strategy,
        strategy_params=json.loads(args.params),
        fees=fees,
        allow_short=not args.no_short,
        max_leverage=args.leverage,
        walkforward=wf,
    )

    ledger = res["ledger"]
    trades = res["trades"]
    met = res["metrics"]

    ledger_path = outdir / "ledger.csv"
    trades_path = outdir / "trades.csv"
    metrics_path = outdir / "metrics.json"
    config_path = outdir / "config.json"

    ledger.to_csv(ledger_path)
    if not trades.empty:
        trades.to_csv(trades_path)
    with open(metrics_path, "w") as f:
        json.dump(met, f, indent=2)
    with open(config_path, "w") as f:
        json.dump(res["config"], f, indent=2)

    print(
        json.dumps(
            {
                "metrics": met,
                "outputs": {
                    "ledger": str(ledger_path),
                    "trades": str(trades_path) if trades_path.exists() else None,
                    "metrics": str(metrics_path),
                    "config": str(config_path),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
