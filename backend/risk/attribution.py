#!/usr/bin/env python3
"""
attribution.py — PnL / return attribution and Brinson-Fachler analysis.

Works with either:
A) Single-asset backtester ledger (from backtester.py / auto_backtester.py)
   -> pass --ledger path/to/ledger.csv --asset TICKER
B) Multi-asset inputs (wide CSVs with Date index or 'Date' column):
   -> --positions positions.csv  (weights/target_pos per asset, per date)
   -> --returns   returns.csv    (per-period asset returns)
   -> optional: --trades trades.csv (per-period delta position)
   -> optional: --prices prices.csv (for turnover and cost allocation)
   -> optional: --map asset_map.json (maps each asset to groups like sector/region/strategy)

Also supports Brinson-Fachler attribution with benchmark weights & returns:
   -> --benchmark-weights bench_w.csv  (wide CSV like positions)
   -> --benchmark-returns bench_r.csv  (wide CSV like returns)
   -> --brinson-by sector               (one grouping key present in asset_map.json)

Outputs (in --outdir):
  - asset_contrib.csv            (asset-level per-period contributions)
  - group_contrib_<key>.csv      (group-level per-period contributions per key)
  - group_summary_<key>.csv      (totals and shares per group)
  - waterfall.csv                (gross vs costs vs net)
  - brinson_<key>.csv            (if benchmark provided)
  - summary.json                 (topline totals)

Conventions:
- contributions = position_{t-1} * return_{t}
- If trades/prices provided, costs are allocated by |trade|*price share; otherwise equal across active names.
- CSVs should be "wide": columns=assets/tickers; rows=time; Date index or "Date" column.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# I/O helpers
# =========================
def read_wide_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{path} must have a Date column or a DatetimeIndex.")
    return df.sort_index()


def read_map_json(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    # Normalize values to str
    out: Dict[str, Dict[str, str]] = {}
    for k, v in data.items():
        out[k] = {kk: (str(vv) if vv is not None else "") for kk, vv in v.items()}
    return out


def ensure_align(*dfs: pd.DataFrame) -> Tuple[pd.DatetimeIndex, List[pd.DataFrame]]:
    """Intersect indices and columns, return aligned copies."""
    idx = dfs[0].index
    cols = set(dfs[0].columns)
    for d in dfs[1:]:
        idx = idx.intersection(d.index)
        cols = cols.intersection(set(d.columns))
    if not cols:
        raise ValueError("No common assets across provided matrices.")
    cols = sorted(cols)
    out = []
    for d in dfs:
        out.append(d.loc[idx, cols].sort_index())
    return idx, out # type: ignore


# =========================
# Core computations
# =========================
def contributions_from_positions_returns(positions: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    contributions_{i,t} = positions_{i,t-1} * returns_{i,t}
    """
    # Align and compute
    _, (pos, ret) = ensure_align(positions, returns)
    contrib = pos.shift(1).fillna(0.0) * ret.fillna(0.0)
    contrib = contrib.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return contrib


def allocate_costs(trades: Optional[pd.DataFrame], prices: Optional[pd.DataFrame], total_cost: pd.Series) -> pd.DataFrame:
    """
    Allocate a per-period total cost (Series on index) across assets.
    - If trades & prices given: weight by |trade|*price per asset per period
    - Else: equal-split among assets that traded (if trades) or among all assets
    Returns a DataFrame same shape as trades/prices with negative cost allocations summing to total_cost per row.
    """
    if total_cost is None or total_cost.empty:
        return pd.DataFrame()

    if trades is not None:
        trades = trades.copy()
    if prices is not None:
        prices = prices.copy()

    # Determine base weights per row
    if trades is not None and prices is not None:
        idx, (tr, px) = ensure_align(trades, prices)
        notional = tr.abs() * px
        w = notional.div(notional.sum(axis=1).replace(0.0, np.nan), axis=0)
    elif trades is not None:
        tr = trades.copy()
        w = tr.abs().div(tr.abs().sum(axis=1).replace(0.0, np.nan), axis=0)
    else:
        # no trades: equal split across all live columns
        cols = prices.columns if prices is not None else None
        if cols is None:
            # Cannot determine assets -> return empty
            return pd.DataFrame()
        w = pd.DataFrame(1.0 / len(cols), index=prices.index, columns=cols) # type: ignore

    w = w.fillna(0.0)
    # Broadcast total_cost across columns
    alloc = -w.mul(total_cost.reindex(w.index).fillna(0.0), axis=0)
    # Clean inf/nan
    alloc = alloc.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return alloc


def regroup(df: pd.DataFrame, asset_map: Dict[str, Dict[str, str]], key: str) -> pd.DataFrame:
    """
    Sum columns (assets) into groups according to mapping[asset][key].
    Unknown assets go to group='UNKNOWN'.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)
    groups = {}
    for asset in df.columns:
        grp = asset_map.get(asset, {}).get(key, "UNKNOWN")
        groups.setdefault(grp, []).append(asset)
    agg = pd.DataFrame(index=df.index)
    for grp, cols in groups.items():
        agg[grp] = df[cols].sum(axis=1)
    return agg


def summarize_groups(group_df: pd.DataFrame) -> pd.DataFrame:
    """Return totals per group with share of grand total."""
    totals = group_df.sum(axis=0).sort_values(ascending=False)
    grand = totals.sum()
    share = totals / (grand if grand != 0 else 1.0)
    out = pd.DataFrame({"TOTAL_CONTRIB": totals, "SHARE": share})
    return out


# =========================
# Brinson-Fachler Attribution
# =========================
def _weights_from_positions(pos: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw positions (can be leveraged) to normalized weights per row.
    If all-zero => keep zeros (avoid div by zero).
    """
    gross = pos.shift(1).abs().sum(axis=1)
    w = pos.shift(1).div(pos.shift(1).sum(axis=1).replace(0.0, np.nan), axis=0)
    # Fallback: if sums to zero, keep zeros
    return w.fillna(0.0)


def brinson_fachler(
    port_pos: pd.DataFrame,
    port_ret: pd.DataFrame,
    bench_w: pd.DataFrame,
    bench_r: pd.DataFrame,
    asset_map: Dict[str, Dict[str, str]],
    by_key: str,
) -> pd.DataFrame:
    """
    Compute Brinson-Fachler attribution by group (by_key).
    Formulas per period t:
      Allocation_g = (W_p_g - W_b_g) * (R_b_g - R_b_total)
      Selection_g  = W_b_g * (R_p_g - R_b_g)
      Interaction_g= (W_p_g - W_b_g) * (R_p_g - R_b_g)
    """
    # Align all matrices
    idx, (Ppos, Pret, Bw, Br) = ensure_align(port_pos, port_ret, bench_w, bench_r)

    # Group membership
    assets = Ppos.columns
    grp_cols = {}
    for a in assets:
        grp = asset_map.get(a, {}).get(by_key, "UNKNOWN")
        grp_cols.setdefault(grp, []).append(a)

    # Pre-allocate result store
    frames = []

    # Precompute group weights & returns per date
    Wp = _weights_from_positions(Ppos)
    Wb = Bw.copy()
    Wb = Wb.div(Wb.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)  # normalize benchmark weights

    for grp, cols in grp_cols.items():
        # group-weight
        Wp_g = Wp[cols].sum(axis=1)
        Wb_g = Wb[cols].sum(axis=1)

        # group return (portfolio): weight within group using portfolio weights
        Wp_cols_norm = Wp[cols].div(Wp[cols].sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        Rp_g = (Wp_cols_norm * Pret[cols]).sum(axis=1)

        # group return (benchmark): weight within group using benchmark weights
        Wb_cols_norm = Wb[cols].div(Wb[cols].sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        Rb_g = (Wb_cols_norm * Br[cols]).sum(axis=1)

        # market total (benchmark)
        Rb_total = (Wb * Br).sum(axis=1)

        alloc = (Wp_g - Wb_g) * (Rb_g - Rb_total)
        selec = Wb_g * (Rp_g - Rb_g)
        inter = (Wp_g - Wb_g) * (Rp_g - Rb_g)

        df_g = pd.DataFrame(
            {
                "group": grp,
                "allocation": alloc,
                "selection": selec,
                "interaction": inter,
            },
            index=idx,
        )
        frames.append(df_g)

    out = pd.concat(frames, axis=0).sort_index()
    # Also add totals per date (sum across groups)
    totals = out.groupby(level=0)[["allocation", "selection", "interaction"]].sum()
    totals["group"] = "__TOTAL__"
    totals = totals.reset_index().set_index("index")
    totals.index.name = None
    totals = totals[["group", "allocation", "selection", "interaction"]]
    # Merge totals back
    out = pd.concat([out, totals])
    return out


# =========================
# Single-asset helpers (ledger)
# =========================
@dataclass
class LedgerBundle:
    positions: pd.DataFrame
    returns: pd.DataFrame
    trades: Optional[pd.DataFrame]
    prices: Optional[pd.DataFrame]
    costs: Optional[pd.Series]  # negative values per period


def load_from_ledger(ledger_path: str, asset_name: str) -> LedgerBundle:
    led = pd.read_csv(ledger_path)
    if "Date" in led.columns:
        led["Date"] = pd.to_datetime(led["Date"])
        led = led.set_index("Date").sort_index()
    elif "date" in led.columns:
        led["date"] = pd.to_datetime(led["date"])
        led = led.set_index("date").sort_index()

    for col in ["pos", "ret", "trade", "price", "cost_pnl", "net_pnl", "gross_pnl", "equity", "drawdown"]:
        if col in led.columns:
            led[col] = pd.to_numeric(led[col], errors="coerce")

    pos = led["pos"].rename(asset_name).to_frame()
    ret = led["net_pnl"].rename(asset_name).to_frame() if "net_pnl" in led.columns else led["ret"].rename(asset_name).to_frame()
    trd = led["trade"].rename(asset_name).to_frame() if "trade" in led.columns else None
    px = led["price"].rename(asset_name).to_frame() if "price" in led.columns else None
    costs = led["cost_pnl"] if "cost_pnl" in led.columns else None  # already negative in our backtester
    return LedgerBundle(pos, ret, trd, px, costs)


# =========================
# CLI Orchestration
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PnL / Return attribution and Brinson analysis.")
    # Mode A (ledger)
    p.add_argument("--ledger", default="", help="Path to ledger.csv from backtester.py")
    p.add_argument("--asset", default="ASSET", help="Asset name for single-asset ledger mode")

    # Mode B (multi-asset)
    p.add_argument("--positions", default="", help="CSV of positions/weights per asset (wide)")
    p.add_argument("--returns", default="", help="CSV of returns per asset (wide)")
    p.add_argument("--trades", default="", help="CSV of trades (delta positions) per asset (wide)")
    p.add_argument("--prices", default="", help="CSV of prices per asset (wide)")

    # Mapping & grouping
    p.add_argument("--map", default="", help="JSON mapping: {asset:{sector:...,region:...,strategy:...}}")
    p.add_argument("--group-by", nargs="*", default=["sector", "region", "strategy"], help="Keys in --map to aggregate by")

    # Costs (only when no ledger costs): fees/slip in bps to estimate costs from trades*price
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)

    # Benchmark (Brinson)
    p.add_argument("--benchmark-weights", default="", help="CSV of benchmark weights (wide)")
    p.add_argument("--benchmark-returns", default="", help="CSV of benchmark returns (wide)")
    p.add_argument("--brinson-by", default="", help="Grouping key (e.g., 'sector') for Brinson-Fachler")

    # Output
    p.add_argument("--outdir", default="attrib_out", help="Output directory")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load mapping
    asset_map = read_map_json(args.map)

    # Choose mode
    if args.ledger:
        bundle = load_from_ledger(args.ledger, args.asset)
        positions = bundle.positions
        returns = bundle.returns
        trades = bundle.trades
        prices = bundle.prices
        costs_series = bundle.costs  # already negative
        assets = [args.asset]
        # If no mapping provided, synthesize a minimal map
        if not asset_map:
            asset_map = {args.asset: {"sector": "SINGLE", "region": "NA", "strategy": "MODEL"}}
    else:
        if not args.positions or not args.returns:
            raise SystemExit("Provide either --ledger or both --positions and --returns.")
        positions = read_wide_csv(args.positions)
        returns = read_wide_csv(args.returns)
        trades = read_wide_csv(args.trades) if args.trades else None
        prices = read_wide_csv(args.prices) if args.prices else None
        costs_series = None
        assets = list(set(positions.columns).intersection(returns.columns))

        # Trim to common set everywhere
        if trades is not None:
            assets = list(set(assets).intersection(trades.columns))
        if prices is not None:
            assets = list(set(assets).intersection(prices.columns))
        assets = sorted(assets)
        positions = positions[assets]
        returns = returns[assets]
        if trades is not None:
            trades = trades[assets]
        if prices is not None:
            prices = prices[assets]

    # Compute contributions
    contrib = contributions_from_positions_returns(positions, returns)
    contrib.to_csv(outdir / "asset_contrib.csv")

    # If no costs series and we have trades/prices + bps, estimate per-period total cost as:
    # sum_i (|trade_i| * price_i * bps / 1e4) then allocate per asset by notional traded
    if costs_series is None and trades is not None and prices is not None and (args.commission_bps + args.slippage_bps) > 0:
        idx, (tr, px) = ensure_align(trades, prices)
        notional = (tr.abs() * px).fillna(0.0)
        total_cost = (notional.sum(axis=1) * ((args.commission_bps + args.slippage_bps) / 1e4)).reindex(contrib.index).fillna(0.0)
        cost_alloc = allocate_costs(trades, prices, total_cost)
    else:
        # If we have costs_series from ledger (single asset), broadcast to that asset
        if costs_series is not None:
            cost_alloc = pd.DataFrame(index=contrib.index, columns=contrib.columns).fillna(0.0)
            # ledger cost_pnl is negative; assign to the single column
            cost_alloc.iloc[:, 0] = costs_series.reindex(contrib.index).fillna(0.0)
        else:
            cost_alloc = pd.DataFrame(index=contrib.index, columns=contrib.columns).fillna(0.0)

    cost_alloc.to_csv(outdir / "asset_costs.csv")

    # Net contributions per asset (gross + costs)
    net_contrib = contrib.add(cost_alloc, fill_value=0.0)
    net_contrib.to_csv(outdir / "asset_net_contrib.csv")

    # Group-level aggregation for requested keys
    for key in args.group_by:
        g = regroup(contrib, asset_map, key)
        g.to_csv(outdir / f"group_contrib_{key}.csv")
        gs = summarize_groups(g)
        gs.to_csv(outdir / f"group_summary_{key}.csv")

        gnet = regroup(net_contrib, asset_map, key)
        gnet.to_csv(outdir / f"group_net_contrib_{key}.csv")
        gsn = summarize_groups(gnet)
        gsn.to_csv(outdir / f"group_net_summary_{key}.csv")

    # Waterfall
    gross_total = float(contrib.sum().sum())
    cost_total = float(cost_alloc.sum().sum())
    net_total = float(net_contrib.sum().sum())
    waterfall = pd.DataFrame(
        {
            "component": ["GROSS_CONTRIB", "COSTS", "NET_CONTRIB"],
            "value": [gross_total, cost_total, net_total],
        }
    )
    waterfall.to_csv(outdir / "waterfall.csv", index=False)

    # Optional: Brinson-Fachler
    if args.benchmark_weights and args.benchmark_returns and args.brinson_by:
        bench_w = read_wide_csv(args.benchmark_weights)
        bench_r = read_wide_csv(args.benchmark_returns)
        # Align assets to intersection with portfolio matrices
        common = sorted(set(positions.columns) & set(returns.columns) & set(bench_w.columns) & set(bench_r.columns))
        if not common:
            raise SystemExit("No common assets across portfolio and benchmark for Brinson.")
        Ppos = positions[common]
        Pret = returns[common]
        Bw = bench_w[common]
        Br = bench_r[common]
        bf = brinson_fachler(Ppos, Pret, Bw, Br, asset_map, args.brinson_by)
        bf.to_csv(outdir / f"brinson_{args.brinson_by}.csv")

    # Topline summary
    summary = {
        "assets": len(contrib.columns),
        "periods": len(contrib.index),
        "gross_total": gross_total,
        "cost_total": cost_total,
        "net_total": net_total,
        "outdir": str(outdir.resolve()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
