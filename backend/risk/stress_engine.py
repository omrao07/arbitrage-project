#!/usr/bin/env python3
"""
stress_engine.py — Flexible scenario stress-testing engine

What it does
------------
- Applies *deterministic scenarios* (asset return shocks, vol shifts, rate moves) to a portfolio
- Supports *generated scenarios* from history: worst-N portfolio days, chosen dates, and z·σ moves
- Supports *factor-based* shocks via factor loadings (asset = B * factor_shock + idio_shock)
- Handles linear assets, options (Δ/Γ/ν approximation), and rates (DV01/convexity)
- Adds liquidation costs (slippage) and liquidity haircuts
- Emits detailed per-asset P&L and scenario-level summaries

Inputs
------
1) Positions CSV (required) — columns (case-insensitive; extras ignored):
   - asset                (str)  : unique asset identifier (e.g., AAPL, UST10Y)
   - qty                  (float): position size (positive = long, negative = short)
   - price                (float): unit price or notional PER UNIT (default 1.0 if omitted)
   - asset_type           (str)  : one of {linear, equity, fx, commodity, crypto, option, rate}
   - underlier            (str)  : for options/rates, the risk driver key (e.g., AAPL or UST10Y)
   - multiplier           (float): contract multiplier for options/futures (default 1.0)
   - delta, gamma, vega   (float): option Greeks (per unit) for Δ/Γ/ν P&L (optional)
   - dv01, convexity      (float): rate risk in currency per 1bp and per (1bp)^2 (per unit) (optional)

2) Returns CSV (optional but recommended)
   - Wide matrix Date x Asset of historical *periodic returns* in decimals (e.g., 0.01 = +1%).
   - Used to compute σ for z·σ scenarios and to locate historic worst-N portfolio days.

3) Factor Loadings (optional) — either format:
   - Wide CSV with index=factor names, columns=assets, values=betas
   - Long CSV with columns: factor, asset, beta

4) Scenarios (choose any combination)
   - JSON file: --scenario-file scenario.json
       [
         {
           "name": "covid_like",
           "asset_shocks": {"AAPL": -0.12, "MSFT": -0.10},
           "vol_shocks":   {"AAPL": 0.10},          # absolute Δσ (e.g., +10 points)
           "rate_shocks":  {"UST10Y": 0.01},        # +100 bps move
           "factor_shocks":{"MKT": -0.08, "SMB": 0.02}
         },
         ...
       ]
   - Inline JSON string via --scenario-json "<...same structure...>"
   - Generated:
       * --z-sigma "2.0,3.0" [--direction down|up]  -> per-asset shock = sign*z*σ_asset
       * --hist-dates "2020-03-16,2008-10-10"       -> use those rows from returns
       * --worst-n 3                                 -> find 3 worst historical *portfolio* P&L days

Costs & Haircuts (optional)
---------------------------
- --slippage-bps 10          : liquidation slippage applied on |notional| (default 0)
- --liquidation-frac 1.0     : fraction of book assumed liquidated for cost/haircut
- --haircuts haircuts.csv    : CSV with columns {asset, haircut} (fraction e.g., 0.15)

Outputs (in --outdir)
---------------------
- scenario_<NAME>_by_asset.csv      : per-asset P&L breakdown for scenario NAME
- summary.csv                        : table of total P&L by scenario + components
- contributions_<NAME>.csv           : asset contributions sorted (largest loss first)
- config.json                        : run configuration

Usage examples
--------------
# 1) 3σ-down move + worst-2 historical portfolio days; include option and rates
python stress_engine.py \
  --positions positions.csv --returns returns.csv \
  --z-sigma "3" --direction down --worst-n 2 \
  --slippage-bps 10 --liquidation-frac 0.5 \
  --outdir stress_out

# 2) Factor shock + explicit asset and rate shocks from JSON
python stress_engine.py \
  --positions positions.csv --factor-loadings betas.csv \
  --scenario-file scenario.json --outdir stress_out

# 3) Use a given crash date from your returns file
python stress_engine.py --positions book.csv --returns ret.csv \
  --hist-dates "2020-03-16" --outdir stress_out
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------
# I/O helpers
# --------------------
def read_positions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    def get(name, default=None):
        return df[cols[name]] if name in cols else default

    out = pd.DataFrame({#type:ignore
        "asset": get("asset", df.columns[0]).astype(str),#type:ignore
        "qty": pd.to_numeric(get("qty", 0.0), errors="coerce").fillna(0.0),#type:ignore
        "price": pd.to_numeric(get("price", 1.0), errors="coerce").fillna(1.0),#type:ignore
        "asset_type": get("asset_type", "linear").fillna("linear").astype(str).str.lower(),#type:ignore
        "underlier": get("underlier", pd.Series([""], index=df.index)).astype(str),
        "multiplier": pd.to_numeric(get("multiplier", 1.0), errors="coerce").fillna(1.0),#type:ignore
        "delta": pd.to_numeric(get("delta", np.nan), errors="coerce"),
        "gamma": pd.to_numeric(get("gamma", np.nan), errors="coerce"),
        "vega": pd.to_numeric(get("vega", np.nan), errors="coerce"),
        "dv01": pd.to_numeric(get("dv01", np.nan), errors="coerce"),
        "convexity": pd.to_numeric(get("convexity", np.nan), errors="coerce"),
    })
    out["asset_type"] = out["asset_type"].replace(
        {"equity":"linear","fx":"linear","commodity":"linear","crypto":"linear"}
    )
    return out


def read_wide_returns(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    c0 = df.columns[0].lower()
    if c0 in ("date","time","t"):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df = df.set_index(df.columns[0])
    return df.sort_index()


def read_factor_loadings(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if set(["factor","asset","beta"]).issubset(set(cols)):
        # long format
        c = {c.lower(): c for c in df.columns}
        long = df[[c["factor"], c["asset"], c["beta"]]].copy()
        pivot = long.pivot(index=c["factor"], columns=c["asset"], values=c["beta"]).fillna(0.0)
        return pivot
    else:
        # assume wide with index=factor; ensure first col isn't Date
        if df.columns[0].lower() in ("factor","name","id"):
            df = df.set_index(df.columns[0])
        return df.set_index(df.columns[0]) if df.index.name is None else df


def read_haircuts(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    a = cols.get("asset", df.columns[0])
    h = cols.get("haircut", df.columns[-1])
    return dict(zip(df[a].astype(str), pd.to_numeric(df[h], errors="coerce").fillna(0.0)))


# --------------------
# Scenario parsing/generation
# --------------------
@dataclass
class Scenario:
    name: str
    asset_shocks: Dict[str, float]       # returns (decimal) by asset
    vol_shocks: Dict[str, float]         # absolute Δσ by underlier/asset
    rate_shocks: Dict[str, float]        # Δy in *rate units* (e.g., 0.01 for +100 bps)
    factor_shocks: Dict[str, float]      # factor shock values


def load_scenarios_json(json_str: Optional[str], json_file: Optional[str]) -> List[Scenario]:
    if not json_str and not json_file:
        return []
    if json_file:
        data = json.loads(Path(json_file).read_text())
    else:
        data = json.loads(json_str)  # may be a dict or list#type:ignore
    if isinstance(data, dict):
        data = [data]
    out = []
    for s in data:
        out.append(Scenario(
            name=str(s.get("name", "scenario")),
            asset_shocks={k: float(v) for k, v in (s.get("asset_shocks", {}) or {}).items()},
            vol_shocks={k: float(v) for k, v in (s.get("vol_shocks", {}) or {}).items()},
            rate_shocks={k: float(v) for k, v in (s.get("rate_shocks", {}) or {}).items()},
            factor_shocks={k: float(v) for k, v in (s.get("factor_shocks", {}) or {}).items()},
        ))
    return out


def generate_zsigma_scenarios(returns: Optional[pd.DataFrame], zs: List[float], direction: str) -> List[Scenario]:
    if returns is None or returns.empty:
        return []
    vol = returns.std(ddof=1)
    sign = -1.0 if direction.lower().startswith("down") else 1.0
    scens = []
    for z in zs:
        asset_shocks = (sign * float(z) * vol).to_dict()
        scens.append(Scenario(
            name=f"{int(z) if z.is_integer() else z}sigma_{'down' if sign<0 else 'up'}",
            asset_shocks=asset_shocks, vol_shocks={}, rate_shocks={}, factor_shocks={}
        ))
    return scens


def generate_hist_date_scenarios(returns: Optional[pd.DataFrame], dates: List[str]) -> List[Scenario]:
    if returns is None or returns.empty:
        return []
    scens = []
    for d in dates:
        try:
            idx = pd.to_datetime(d)
            row = returns.loc[idx]
        except Exception:
            continue
        scens.append(Scenario(
            name=f"hist_{idx.date()}",
            asset_shocks=row.to_dict(), vol_shocks={}, rate_shocks={}, factor_shocks={}#type:ignore
        ))
    return scens


def generate_worstn_scenarios(returns: Optional[pd.DataFrame], positions: pd.DataFrame, n: int) -> List[Scenario]:
    if returns is None or returns.empty or n <= 0:
        return []
    # Linear approximation P&L by day: sum_i (qty*price*ret_i)
    weights = (positions["qty"] * positions["price"]).groupby(positions["asset"]).sum()
    aligned = returns.reindex(columns=weights.index, fill_value=0.0)
    pnl = (aligned * weights.values).sum(axis=1)
    worst_days = pnl.nsmallest(min(n, len(pnl))).index
    scens = []
    for dt in worst_days:
        scens.append(Scenario(
            name=f"worst_{pd.to_datetime(dt).date()}",
            asset_shocks=returns.loc[dt].to_dict(), vol_shocks={}, rate_shocks={}, factor_shocks={}
        ))
    return scens


# --------------------
# Shock assembly
# --------------------
def assemble_asset_shocks(
    scen: Scenario,
    assets_in_book: List[str],
    betas: Optional[pd.DataFrame],
) -> pd.Series:
    """
    Combine explicit asset_shocks with factor_shocks via loadings: r = B'f + asset_override
    """
    r = pd.Series(0.0, index=pd.Index(assets_in_book, name="asset"))
    # factor contribution
    if betas is not None and scen.factor_shocks:
        f = pd.Series(scen.factor_shocks)
        common_f = f.index.intersection(betas.index)#type:ignore
        if not common_f.empty:#type:ignore
            contrib = (betas.loc[common_f].T @ f.loc[common_f]).reindex(r.index).fillna(0.0)
            r = r.add(contrib, fill_value=0.0)
    # explicit per-asset overrides/additions
    if scen.asset_shocks:
        r = r.add(pd.Series(scen.asset_shocks), fill_value=0.0)
    return r


# --------------------
# P&L engine
# --------------------
@dataclass
class Costs:
    slippage_bps: float = 0.0
    liquidation_frac: float = 1.0
    haircuts: Dict[str, float] = None#type:ignore


def pnl_for_scenario(
    scen: Scenario,
    positions: pd.DataFrame,
    betas: Optional[pd.DataFrame],
    costs: Costs,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Returns:
      - per-asset breakdown DataFrame
      - totals dict
    """
    assets = positions["asset"].astype(str).tolist()
    r_asset = assemble_asset_shocks(scen, assets, betas)  # decimal returns
    vol_shock = pd.Series(scen.vol_shocks) if scen.vol_shocks else pd.Series(dtype=float)
    rate_shock = pd.Series(scen.rate_shocks) if scen.rate_shocks else pd.Series(dtype=float)

    recs = []
    total = {
        "pnl_linear": 0.0,
        "pnl_option": 0.0,
        "pnl_rate": 0.0,
        "haircuts": 0.0,
        "slippage": 0.0,
    }

    for i, row in positions.iterrows():
        asset = row["asset"]
        a_type = row["asset_type"]
        qty = float(row["qty"])
        px = float(row["price"])
        mult = float(row["multiplier"])
        base_notional = qty * px * mult

        pnl_lin = pnl_opt = pnl_rate = h_cut = slip = 0.0

        if a_type == "linear":
            ret = float(r_asset.get(asset, 0.0))
            pnl_lin = base_notional * ret

        elif a_type == "option":
            und = row["underlier"] if isinstance(row["underlier"], str) and row["underlier"] else asset
            dS = float(r_asset.get(und, 0.0)) * px  # using price as proxy for S if S not separately provided
            dvol = float(vol_shock.get(und, 0.0))
            delta = 0.0 if pd.isna(row["delta"]) else float(row["delta"])
            gamma = 0.0 if pd.isna(row["gamma"]) else float(row["gamma"])
            vega  = 0.0 if pd.isna(row["vega"])  else float(row["vega"])
            # Δ/Γ assume ∂P/∂S and ∂²P/∂S²; vega assumes ∂P/∂σ (σ in absolute units)
            pnl_unit = delta * dS + 0.5 * gamma * (dS ** 2) + vega * dvol
            pnl_opt = qty * mult * pnl_unit

        elif a_type == "rate":
            key = row["underlier"] if isinstance(row["underlier"], str) and row["underlier"] else asset
            dy = float(rate_shock.get(key, 0.0))              # rate units (e.g., 0.01 = +100 bps)
            dbp = dy * 10000.0
            dv01 = 0.0 if pd.isna(row["dv01"]) else float(row["dv01"])
            conv = 0.0 if pd.isna(row["convexity"]) else float(row["convexity"])
            # DV01 is currency per 1bp (per unit). Price change ≈ -DV01*Δbp + 0.5*Conv*(Δbp)^2
            pnl_rate = qty * (-dv01 * dbp + 0.5 * conv * (dbp ** 2))

        # Haircut on liquidation notionals (always applied if provided)
        hc = float(costs.haircuts.get(asset, 0.0)) if (costs.haircuts is not None) else 0.0
        h_cut = -hc * abs(base_notional) * float(costs.liquidation_frac)

        # Slippage/TC on liquidation
        slip = -abs(base_notional) * (costs.slippage_bps / 1e4) * float(costs.liquidation_frac)

        recs.append({
            "asset": asset,
            "type": a_type,
            "qty": qty,
            "price": px,
            "notional": base_notional,
            "shock_return": float(r_asset.get(asset, 0.0)),
            "underlier": row["underlier"],
            "pnl_linear": pnl_lin,
            "pnl_option": pnl_opt,
            "pnl_rate": pnl_rate,
            "haircut": h_cut,
            "slippage": slip,
            "pnl_total": pnl_lin + pnl_opt + pnl_rate + h_cut + slip,
        })

        total["pnl_linear"] += pnl_lin
        total["pnl_option"] += pnl_opt
        total["pnl_rate"] += pnl_rate
        total["haircuts"]  += h_cut
        total["slippage"]  += slip

    df = pd.DataFrame(recs)
    totals = {
        "scenario": scen.name,
        **{k: float(v) for k, v in total.items()},
    }
    totals["pnl_total"] = float(sum(v for k, v in total.items()))
    return df, totals


# --------------------
# CLI
# --------------------
@dataclass
class Config:
    positions: str
    returns: Optional[str]
    factor_loadings: Optional[str]
    scenario_file: Optional[str]
    scenario_json: Optional[str]
    z_sigma: List[float]
    direction: str
    hist_dates: List[str]
    worst_n: int
    slippage_bps: float
    liquidation_frac: float
    haircuts: Optional[str]
    outdir: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scenario stress-testing engine")
    p.add_argument("--positions", required=True, help="Positions CSV (see header for expected columns).")
    p.add_argument("--returns", default="", help="Wide returns CSV (Date x Assets, decimals).")
    p.add_argument("--factor-loadings", default="", help="Factor loadings CSV (wide or long).")
    p.add_argument("--scenario-file", default="", help="JSON file with scenarios.")
    p.add_argument("--scenario-json", default="", help="Inline JSON string with scenarios.")
    p.add_argument("--z-sigma", default="", help='Comma list of z for z·σ scenarios, e.g. "2,3".')
    p.add_argument("--direction", default="down", choices=["down","up"], help="Direction for z·σ scenarios.")
    p.add_argument("--hist-dates", default="", help='Comma dates present in returns (e.g., "2020-03-16").')
    p.add_argument("--worst-n", type=int, default=0, help="Generate worst-N historical portfolio days.")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Liquidation slippage on |notional| in bps.")
    p.add_argument("--liquidation-frac", type=float, default=1.0, help="Fraction liquidated (for slippage & haircuts).")
    p.add_argument("--haircuts", default="", help="CSV with columns {asset, haircut} (fraction).")
    p.add_argument("--outdir", default="stress_out", help="Output directory.")
    return p.parse_args()


# --------------------
# Main
# --------------------
def main():
    args = parse_args()
    cfg = Config(
        positions=args.positions,
        returns=args.returns or None,
        factor_loadings=args.factor_loadings or None,
        scenario_file=args.scenario_file or None,
        scenario_json=args.scenario_json or None,
        z_sigma=[float(x) for x in args.z_sigma.split(",") if x.strip()] if args.z_sigma else [],
        direction=args.direction,
        hist_dates=[s.strip() for s in args.hist_dates.split(",") if s.strip()],
        worst_n=int(args.worst_n),
        slippage_bps=float(args.slippage_bps),
        liquidation_frac=float(args.liquidation_frac),
        haircuts=args.haircuts or None,
        outdir=args.outdir,
    )

    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    positions = read_positions(cfg.positions)
    returns = read_wide_returns(cfg.returns) if cfg.returns else None
    betas = read_factor_loadings(cfg.factor_loadings)
    haircuts_map = read_haircuts(cfg.haircuts)

    # Build scenarios
    scens: List[Scenario] = []
    scens += load_scenarios_json(cfg.scenario_json, cfg.scenario_file)
    scens += generate_zsigma_scenarios(returns, cfg.z_sigma, cfg.direction)
    scens += generate_hist_date_scenarios(returns, cfg.hist_dates)
    scens += generate_worstn_scenarios(returns, positions, cfg.worst_n)

    if not scens:
        raise SystemExit("No scenarios provided. Use --scenario-file / --scenario-json / --z-sigma / --hist-dates / --worst-n.")

    # Run
    totals_list: List[Dict[str, float]] = []
    for scen in scens:
        df_asset, totals = pnl_for_scenario(
            scen,
            positions,
            betas,
            Costs(slippage_bps=cfg.slippage_bps, liquidation_frac=cfg.liquidation_frac, haircuts=haircuts_map),
        )
        # Save per-scenario asset breakdown
        out_asset = outdir / f"scenario_{scen.name}_by_asset.csv"
        df_asset.sort_values("pnl_total").to_csv(out_asset, index=False)

        # Save contributions sorted
        contrib = df_asset[["asset","type","pnl_total"]].copy().sort_values("pnl_total")
        contrib.to_csv(outdir / f"contributions_{scen.name}.csv", index=False)

        totals_list.append(totals)

    # Summary table
    summary = pd.DataFrame(totals_list).set_index("scenario")
    summary = summary[["pnl_linear","pnl_option","pnl_rate","haircuts","slippage","pnl_total"]].sort_values("pnl_total")
    summary.to_csv(outdir / "summary.csv")

    # Save config
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Console print
    print("== Stress test complete ==")
    print(summary.round(2))


if __name__ == "__main__":
    main()
