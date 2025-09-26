#!/usr/bin/env python3
from pathlib import Path
import sys
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    raise SystemExit(f"Required packages missing: {e}")

# --------- Strategy Interfaces ---------
class Strategy:
    """Base Strategy interface. Implement generate_signals and name."""
    def name(self) -> str:
        return self.__class__.__name__

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        """Return signal series aligned to df.index with values in {-1, 0, +1}."""
        raise NotImplementedError


class SMACross(Strategy):
    def generate_signals(self, df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
        close = df["Close"].astype(float)
        fast_ma = close.rolling(fast, min_periods=1).mean()
        slow_ma = close.rolling(slow, min_periods=1).mean()
        sig = np.where(fast_ma > slow_ma, 1, -1)
        s = pd.Series(sig, index=df.index, name="signal")
        s.iloc[: slow] = 0  # warmup
        return s


class RSIRevert(Strategy):
    def generate_signals(self, df: pd.DataFrame, period: int = 14, low: int = 30, high: int = 70) -> pd.Series:
        close = df["Close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        sig = np.where(rsi < low, 1, np.where(rsi > high, -1, 0))
        s = pd.Series(sig, index=df.index, name="signal")
        s.iloc[: period] = 0
        return s


STRATEGY_REGISTRY: Dict[str, Strategy] = {
    "sma_cross": SMACross(),
    "rsi_revert": RSIRevert(),
}


# --------- Utilities ---------
def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column.")
    return df.sort_index()


def load_data(source: str, symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """
    Load OHLCV data.
    - If source endswith .csv: read CSV (expects Date, Open, High, Low, Close, Volume).
    - Else if source == 'yfinance': fetch via yfinance (if available).
    """
    if source.lower().endswith(".csv"):
        df = pd.read_csv(source)
        df = ensure_datetime_index(df)
        if start: df = df[df.index >= pd.to_datetime(start)]
        if end: df = df[df.index <= pd.to_datetime(end)]
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()

    if source.lower() == "yfinance":
        try:
            import yfinance as yf  # type: ignore
        except Exception as e:
            raise SystemExit("yfinance not installed. Install with: pip install yfinance") from e
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        df = df.rename(columns=str.title) # type: ignore
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    raise ValueError("Unsupported source. Use a CSV path or 'yfinance'.")


def position_sizer(signal: pd.Series, max_leverage: float = 1.0) -> pd.Series:
    """Convert discrete signal {-1,0,1} to target position in [-max_leverage, +max_leverage]."""
    return (signal.clip(-1, 1) * max_leverage).rename("target_pos")


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
    Simple execution simulator:
    - Convert target positions into trades (delta in position).
    - Assume fills at next close with slippage.
    - Apply commissions/slippage in bps.
    Returns: (ledger, trades)
    """
    px = df["Close"].astype(float)
    target_pos = target_pos.reindex(px.index).fillna(0.0).clip(-1.0 if allow_short else 0.0, 1.0)

    # Compute daily returns
    ret = px.pct_change().fillna(0.0)

    # Position changes -> trades at next bar open/close (we'll use next close for simplicity)
    pos = target_pos.shift(1).fillna(0.0)  # hold yesterday's target from today's open to close
    trade = target_pos.diff().fillna(target_pos.iloc[0])

    # Slippage and commission cost computed on notional traded
    notional_traded = (trade.abs() * px)
    trans_cost = notional_traded * ((fees.commission_bps + fees.slippage_bps) / 1e4)

    # PnL = position * daily return * price (assuming 1 unit capital)
    # We interpret position as fraction of equity invested (leverage)
    gross_pnl = pos * ret
    cost_pnl = trans_cost / px  # convert cost to equity fraction
    net_pnl = gross_pnl - cost_pnl

    equity = (1.0 + net_pnl).cumprod()
    dd = equity / equity.cummax() - 1.0

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
            "drawdown": dd,
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
    total_return = eq.iloc[-1] - 1.0
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9) # type: ignore
    cagr = (eq.iloc[-1]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = (rets.mean() * periods_per_year - rf) / (vol + 1e-12)
    mdd = (eq / eq.cummax() - 1.0).min()
    calmar = (cagr / abs(mdd)) if mdd != 0 else np.nan
    win_rate = (rets > 0).mean()
    avg_trade = rets[rets != 0].mean() if (rets != 0).any() else 0.0

    return {
        "TOTAL_RETURN": float(total_return),
        "CAGR": float(cagr),
        "VOL": float(vol),
        "SHARPE": float(sharpe),
        "MAX_DRAWDOWN": float(mdd),
        "CALMAR": float(calmar),
        "WIN_RATE": float(win_rate),
        "AVG_EX_RET": float(avg_trade),
    }


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
    """
    Expanding walk-forward with grid search on each train split selecting by Sharpe.
    """
    results: List[pd.DataFrame] = []
    params_used: List[Dict[str, Any]] = []
    idx = df.index
    start = train_window
    while start + test_window <= len(df):
        train = df.iloc[start - train_window:start]
        test = df.iloc[start : start + test_window]

        # Grid search on train
        best_sharpe = -np.inf
        best_params = {}

        # generate param combinations
        keys = list(param_grid.keys())
        grids: List[Dict[str, Any]] = [{}]
        for k in keys:
            grids = [dict(g, **{k: v}) for g in grids for v in param_grid[k]]

        for params in grids:
            sig = strategy.generate_signals(train, **params)
            pos = position_sizer(sig, max_leverage=max_leverage)
            led, _ = apply_execution(train, pos, fees=fees, allow_short=allow_short)
            m = metrics(led["equity"], led["net_pnl"])
            if m["SHARPE"] > best_sharpe:
                best_sharpe = m["SHARPE"]
                best_params = params

        # Apply best to test
        sig_t = strategy.generate_signals(test, **best_params)
        pos_t = position_sizer(sig_t, max_leverage=max_leverage)
        led_t, _ = apply_execution(test, pos_t, fees=fees, allow_short=allow_short)
        led_t["best_params"] = json.dumps(best_params)
        results.append(led_t)
        params_used.append(best_params)

        start += test_window

    if not results:
        raise ValueError("Walk-forward produced no splits—check window sizes.")

    wf_equity = pd.concat(results)["equity"]
    wf_returns = pd.concat(results)["net_pnl"]

    return {
        "ledger": pd.concat(results),
        "params_used": params_used,
        "wf_metrics": metrics(wf_equity, wf_returns),
    }


# --------- Main Backtest Runner ---------
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
            train_window=walkforward["train_window"],
            test_window=walkforward["test_window"],
            fees=fees,
            allow_short=allow_short,
            max_leverage=max_leverage,
        )
        ledger = wf["ledger"]
        ret = ledger["net_pnl"]
        meta_metrics = wf["wf_metrics"]
        trades = pd.DataFrame()  # aggregated trades omitted for WF
    else:
        signal = strat.generate_signals(data, **strategy_params)
        target = position_sizer(signal, max_leverage=max_leverage)
        ledger, trades = apply_execution(data, target, fees=fees, allow_short=allow_short)
        ret = ledger["net_pnl"]
        meta_metrics = metrics(ledger["equity"], ret)

    return {
        "ledger": ledger,
        "trades": trades,
        "metrics": meta_metrics,
        "config": {
            "strategy": strategy_name,
            "strategy_params": strategy_params,
            "fees": asdict(fees),
            "allow_short": allow_short,
            "max_leverage": max_leverage,
            "walkforward": walkforward is not None,
        },
    }


# --------- CLI ---------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto Backtester")
    p.add_argument("--source", type=str, required=True, help="CSV path or 'yfinance'")
    p.add_argument("--symbol", type=str, default="", help="Ticker (if using yfinance)")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--outdir", type=str, default="backtest_results", help="Output directory")
    p.add_argument("--strategy", type=str, default="sma_cross", choices=list(STRATEGY_REGISTRY.keys()))
    p.add_argument("--params", type=str, default="{}", help='JSON dict of strategy params, e.g. \'{"fast":10,"slow":50}\'')
    p.add_argument("--fees", type=str, default='{"commission_bps":1.0,"slippage_bps":0.0}', help="JSON dict of fees")
    p.add_argument("--no-short", action="store_true", help="Disable shorting")
    p.add_argument("--leverage", type=float, default=1.0, help="Max leverage (abs position)")
    p.add_argument("--walkforward", type=str, default="", help='JSON like {"train_window":252,"test_window":63,"param_grid":{"fast":[5,10],"slow":[30,50]}}')
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(args.source, args.symbol, args.start, args.end)
    # Backtest
    fees = Fees(**json.loads(args.fees))
    walkforward = json.loads(args.walkforward) if args.walkforward else None
    res = run_backtest(
        df,
        strategy_name=args.strategy,
        strategy_params=json.loads(args.params),
        fees=fees,
        allow_short=not args.no_short,
        max_leverage=args.leverage,
        walkforward=walkforward,
    )

    # Save outputs
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

    # Print a concise summary
    print(json.dumps({"metrics": met, "outputs": {
        "ledger": str(ledger_path),
        "trades": str(trades_path) if trades_path.exists() else None,
        "metrics": str(metrics_path),
        "config": str(config_path),
    }}, indent=2))


if __name__ == "__main__":
    main()
