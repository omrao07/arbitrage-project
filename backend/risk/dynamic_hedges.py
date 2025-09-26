#!/usr/bin/env python3
"""
dynamic_hedges.py — Discrete-time dynamic hedging engine with multiple schemes

Implements:
- Black–Scholes prices & Greeks (call/put)
- Path generation (GBM) or load paths from CSV
- Hedging modes:
    1) delta        : classic delta replication with cash + underlying
    2) dg           : delta–gamma neutralization using 1 extra hedge option
    3) dgv          : delta–gamma–vega neutralization using 2 extra hedge options
    4) leland       : delta hedging with Leland volatility adjustment for proportional costs
- Rebalance rules: fixed time, or threshold on delta change
- Transaction costs: proportional (bps or fraction), fixed per trade, and slippage (bps)
- Outputs per-path PnL and summary stats; saves trades & equity if requested

Usage examples
--------------
# 1) Simulate GBM and delta-hedge a call
python dynamic_hedges.py --mode delta --simulate --S0 100 --sigma 0.2 --r 0.01 --q 0.0 \
  --T 1.0 --steps 252 --npaths 200 --option call --K 100 --outdir out_delta

# 2) Delta–Gamma neutralization with a hedge option (same maturity)
python dynamic_hedges.py --mode dg --simulate --S0 100 --sigma 0.25 --r 0.02 --q 0.0 \
  --T 0.5 --steps 126 --npaths 100 --option call --K 100 \
  --hedge-K 110 --outdir out_dg

# 3) Delta–Gamma–Vega neutralization with two hedge options
python dynamic_hedges.py --mode dgv --simulate --S0 100 --sigma 0.3 --r 0.0 --q 0.0 \
  --T 1.0 --steps 252 --npaths 50 --option put --K 95 \
  --hedge-K 90,110 --outdir out_dgv

# 4) Leland adjusted volatility (proportional TC k=10 bps, daily hedge)
python dynamic_hedges.py --mode leland --simulate --S0 100 --sigma 0.2 --r 0.00 --q 0.0 \
  --T 1.0 --steps 252 --npaths 1000 --option call --K 100 \
  --tc-prop 0.001 --outdir out_leland

Notes
-----
- CSV input: pass --paths path.csv where columns are paths (one column per path) and index are dates or steps.
- Transaction costs:
    * --tc-prop can be a fraction (e.g., 0.001 = 10 bps). If you prefer bps, pass e.g. --tc-bps 10.
    * Slippage applies to trade price via S*(1 +/- slippage), where slippage is a fraction or set via --slippage-bps.
- Rebalance control:
    * default: hedge at every step
    * delta threshold: --delta-thresh 0.02 (rebalance when |Δ_new - Δ_old| > 0.02)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from math import log, sqrt, exp, erf, pi


# =========================
# Black–Scholes Pricer/Greeks
# =========================
def _phi(x: float | np.ndarray) -> float | np.ndarray:
    # standard normal CDF via erf (no scipy)
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _pdf(x: float | np.ndarray) -> float | np.ndarray:
    return (1.0 / sqrt(2.0 * pi)) * np.exp(-0.5 * (np.asarray(x) ** 2))


def _d1_d2(S, K, r, q, sigma, tau):
    # avoid divide-by-zero
    eps = 1e-16
    sigma = np.maximum(sigma, eps)
    tau = np.maximum(tau, eps)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return d1, d2


def bs_price(kind: str, S, K, r, q, sigma, tau):
    d1, d2 = _d1_d2(S, K, r, q, sigma, tau)
    if kind.lower() == "call":
        return S * np.exp(-q * tau) * _phi(d1) - K * np.exp(-r * tau) * _phi(d2)
    else:
        return K * np.exp(-r * tau) * _phi(-d2) - S * np.exp(-q * tau) * _phi(-d1)


def bs_delta(kind: str, S, K, r, q, sigma, tau):
    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    if kind.lower() == "call":
        return np.exp(-q * tau) * _phi(d1)
    else:
        return np.exp(-q * tau) * (_phi(d1) - 1.0)


def bs_gamma(S, K, r, q, sigma, tau):
    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    return np.exp(-q * tau) * _pdf(d1) / (S * sigma * np.sqrt(np.maximum(tau, 1e-16)))


def bs_vega(S, K, r, q, sigma, tau):
    d1, _ = _d1_d2(S, K, r, q, sigma, tau)
    return S * np.exp(-q * tau) * _pdf(d1) * np.sqrt(np.maximum(tau, 1e-16))


@dataclass
class EuropeanOption:
    kind: str  # "call" or "put"
    K: float
    T: float  # years


# =========================
# Path generation / loading
# =========================
def simulate_gbm(S0: float, mu: float, sigma: float, T: float, steps: int, npaths: int, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dt = T / steps
    # paths shape: (npaths, steps+1)
    paths = np.empty((npaths, steps + 1), dtype=float)
    paths[:, 0] = S0
    drift = (mu - 0.5 * sigma * sigma) * dt
    shock = sigma * np.sqrt(dt)
    for t in range(1, steps + 1):
        z = rng.standard_normal(npaths)
        paths[:, t] = paths[:, t - 1] * np.exp(drift + shock * z)
    times = np.linspace(0.0, T, steps + 1)
    return paths, times


def load_paths_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if df.columns[0].lower() in ("date", "time", "t"):
        df = df.set_index(df.columns[0])
    # Each column is a path
    arr = df.values.T  # shape (npaths, steps)
    # assume uniform steps; time unit arbitrary -> set tau in fractions of length-1
    steps = arr.shape[1] - 1
    times = np.linspace(0.0, 1.0, steps + 1)
    return arr, times


# =========================
# Transaction Costs
# =========================
@dataclass
class Costs:
    prop: float = 0.0        # proportional fraction (e.g., 0.001 = 10 bps)
    fixed: float = 0.0       # fixed per trade (in currency units of underlying)
    slippage: float = 0.0    # price slippage fraction (apply to trade price)


def trade_cost(notional: np.ndarray, n_trades: np.ndarray, costs: Costs) -> np.ndarray:
    # cost = prop*|notional| + fixed * 1_{trade!=0}
    return costs.prop * np.abs(notional) + costs.fixed * (n_trades != 0)


def effective_trade_price(S: np.ndarray, qty: np.ndarray, costs: Costs) -> np.ndarray:
    # buy (qty>0) -> worse price S*(1+slip); sell (qty<0) -> S*(1-slip)
    return S * (1.0 + costs.slippage * np.sign(qty))


# =========================
# Leland volatility adjustment (for proportional costs)
# =========================
def leland_sigma(sigma: float, k: float, dt: float) -> float:
    # Leland (1985) adjustment: sigma_hat^2 = sigma^2 * (1 + (2/pi) * (k / (sigma)) * sqrt(1/dt))
    if sigma <= 0 or dt <= 0 or k <= 0:
        return sigma
    adj = 1.0 + (2.0 / pi) * (k / max(sigma, 1e-12)) * np.sqrt(1.0 / dt)
    return sigma * np.sqrt(adj)


# =========================
# Hedging cores
# =========================
def delta_hedge_path(
    S: np.ndarray, times: np.ndarray, opt: EuropeanOption,
    r: float, q: float, sigma: float, costs: Costs,
    delta_thresh: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
    """
    Classic delta replication (short 1 option liability; long Δ shares + cash).
    Returns per-path PnL, summary, and (optional) trade/ledger DataFrame for the first path.
    """
    nsteps = len(times) - 1
    dt = times[1] - times[0]
    npaths = S.shape[0]

    # storage
    pnl = np.zeros(npaths)
    # For path 0, collect a ledger
    rec = []

    for p in range(npaths):
        s = S[p]
        cash = 0.0
        delta_old = 0.0
        pos = 0.0  # shares held

        # initial option value (liability)
        tau0 = opt.T - times[0]
        V0 = bs_price(opt.kind, s[0], opt.K, r, q, sigma, tau0)
        # replicate from t=0: set Δ and buy shares
        delta_new = bs_delta(opt.kind, s[0], opt.K, r, q, sigma, tau0)
        do_trade = True
        if delta_thresh is not None and abs(delta_new - delta_old) <= delta_thresh:
            do_trade = False
            delta_new = delta_old

        if do_trade:
            qty = delta_new - delta_old
            px = effective_trade_price(np.array([s[0]]), np.array([qty]), costs)[0]
            cost = trade_cost(np.array([qty * px]), np.array([qty]), costs)[0]
            cash -= qty * px + cost
            pos += qty
        delta_old = delta_new

        if p == 0:
            rec.append({"step": 0, "S": s[0], "delta": delta_old, "pos": pos, "cash": cash, "action": "init"})

        # iterate steps
        for t in range(1, nsteps + 1):
            tau = max(opt.T - times[t], 0.0)

            # accrue risk-free on cash
            cash *= exp(r * (dt))

            # Decide rebalance?
            delta_new = bs_delta(opt.kind, s[t], opt.K, r, q, sigma, tau) if tau > 0 else 0.0
            do_trade = True
            if delta_thresh is not None and abs(delta_new - delta_old) <= delta_thresh:
                do_trade = False
                delta_new = delta_old

            if do_trade:
                qty = delta_new - delta_old
                px = effective_trade_price(np.array([s[t]]), np.array([qty]), costs)[0]
                cost = trade_cost(np.array([qty * px]), np.array([qty]), costs)[0]
                cash -= qty * px + cost
                pos += qty
                delta_old = delta_new

                if p == 0:
                    rec.append({"step": t, "S": s[t], "delta": delta_old, "pos": pos, "cash": cash, "action": f"rebalance({qty:.6f})"})
            else:
                if p == 0:
                    rec.append({"step": t, "S": s[t], "delta": delta_old, "pos": pos, "cash": cash, "action": "hold"})

        # settle at maturity
        payoff = max(0.0, (s[-1] - opt.K)) if opt.kind == "call" else max(0.0, (opt.K - s[-1]))
        # close share position
        if abs(pos) > 0:
            qty = -pos
            px = effective_trade_price(np.array([s[-1]]), np.array([qty]), costs)[0]
            cost = trade_cost(np.array([qty * px]), np.array([qty]), costs)[0]
            cash -= qty * px + cost  # note qty may be negative/positive; -qty*px subtracts proceeds for sells etc.
            pos = 0.0
            if p == 0:
                rec.append({"step": nsteps, "S": s[-1], "delta": 0.0, "pos": pos, "cash": cash, "action": f"close({qty:.6f})"})

        pnl[p] = cash - payoff  # short 1 option liability replicated -> hedger PnL

        if p == 0:
            rec.append({"step": nsteps, "S": s[-1], "delta": 0.0, "pos": pos, "cash": cash, "action": f"settle(payoff={payoff:.6f})"})

    summary = {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        "min_pnl": float(np.min(pnl)),
        "max_pnl": float(np.max(pnl)),
        "hit_rate": float(np.mean(pnl > 0)),
        "npaths": int(npaths),
    }
    ledger = pd.DataFrame(rec)
    return pnl, summary, ledger


def solve_dg_positions(
    S: float, t: float, opt_L: EuropeanOption, K2: float,
    r: float, q: float, sigma: float, T: float,
) -> Tuple[float, float]:
    """Delta–Gamma neutralization: short 1 liability (opt_L); choose x (underlying), y (hedge option @K2) s.t. net Δ=0 and Γ=0."""
    tau = max(opt_L.T - t, 0.0)
    if tau <= 0:
        return 0.0, 0.0
    dL = bs_delta(opt_L.kind, S, opt_L.K, r, q, sigma, tau)
    gL = bs_gamma(S, opt_L.K, r, q, sigma, tau)

    # Hedge option Greeks
    d2 = bs_delta("call", S, K2, r, q, sigma, tau)  # hedge option kind doesn't matter for gamma; use call for stability
    g2 = bs_gamma(S, K2, r, q, sigma, tau)

    # Solve:
    # x*1 + y*d2 - dL = 0
    # 0*x + y*g2 - gL = 0  => y = gL/g2
    y = gL / max(g2, 1e-16)
    x = dL - y * d2
    return float(x), float(y)


def solve_dgv_positions(
    S: float, t: float, opt_L: EuropeanOption, K2: float, K3: float,
    r: float, q: float, sigma: float, T: float,
) -> Tuple[float, float, float]:
    """Delta–Gamma–Vega neutralization using underlying + two hedge options (calls @K2, K3)."""
    tau = max(opt_L.T - t, 0.0)
    if tau <= 0:
        return 0.0, 0.0, 0.0
    dL = bs_delta(opt_L.kind, S, opt_L.K, r, q, sigma, tau)
    gL = bs_gamma(S, opt_L.K, r, q, sigma, tau)
    vL = bs_vega(S, opt_L.K, r, q, sigma, tau)

    d2 = bs_delta("call", S, K2, r, q, sigma, tau)
    g2 = bs_gamma(S, K2, r, q, sigma, tau)
    v2 = bs_vega(S, K2, r, q, sigma, tau)

    d3 = bs_delta("call", S, K3, r, q, sigma, tau)
    g3 = bs_gamma(S, K3, r, q, sigma, tau)
    v3 = bs_vega(S, K3, r, q, sigma, tau)

    # Solve linear system:
    # [1  d2  d3][x]   [dL]
    # [0  g2  g3][y] = [gL]
    # [0  v2  v3][z]   [vL]
    A = np.array([[1.0, d2, d3],
                  [0.0, g2, g3],
                  [0.0, v2, v3]], dtype=float)
    b = np.array([dL, gL, vL], dtype=float)
    # robust solve (least-squares if near singular)
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    x, y, z = sol
    return float(x), float(y), float(z)


def multi_greek_hedge_path(
    S: np.ndarray, times: np.ndarray, opt: EuropeanOption,
    mode: str, r: float, q: float, sigma: float, costs: Costs,
    hedge_Ks: List[float], delta_thresh: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Hedge a short 1-liability option with:
      - dg: underlying + 1 hedge option (neutralize Δ & Γ)
      - dgv: underlying + 2 hedge options (neutralize Δ,Γ,ν)
    Rebalance at each step (or on delta threshold for the aggregate delta).
    PnL = (x shares + y/z hedge options + cash) - liability payoff at T.
    """
    assert mode in ("dg", "dgv")
    nsteps = len(times) - 1
    dt = times[1] - times[0]
    npaths = S.shape[0]
    pnl = np.zeros(npaths)

    for p in range(npaths):
        s = S[p]
        # Positions
        x = 0.0  # underlying
        y = 0.0  # hedge option K2
        z = 0.0  # hedge option K3 (if dgv)
        cash = 0.0

        for t in range(nsteps + 1):
            tau = max(opt.T - times[t], 0.0)

            # accrue cash
            if t > 0:
                cash *= exp(r * dt)

            # compute target positions
            if mode == "dg":
                K2 = hedge_Ks[0]
                xt, yt = solve_dg_positions(s[t], times[t], opt, K2, r, q, sigma, opt.T)
                # optional delta threshold on aggregate delta change
                agg_delta_old = x + y * bs_delta("call", s[t], K2, r, q, sigma, tau if tau > 0 else 1e-9) - bs_delta(opt.kind, s[t], opt.K, r, q, sigma, tau if tau > 0 else 1e-9)
                agg_delta_new = xt + yt * bs_delta("call", s[t], K2, r, q, sigma, tau if tau > 0 else 1e-9) - bs_delta(opt.kind, s[t], opt.K, r, q, sigma, tau if tau > 0 else 1e-9)

                do_trade = True
                if delta_thresh is not None and abs(agg_delta_new - agg_delta_old) <= delta_thresh:
                    do_trade = False
                    xt, yt = x, y

                if do_trade:
                    # trades
                    dq_x = xt - x
                    dq_y = yt - y
                    # underlyer trade
                    if abs(dq_x) > 0:
                        px = effective_trade_price(np.array([s[t]]), np.array([dq_x]), costs)[0]
                        c = trade_cost(np.array([dq_x * px]), np.array([dq_x]), costs)[0]
                        cash -= dq_x * px + c
                        x = xt
                    # hedge option approximated by Black–Scholes price (call @K2)
                    if abs(dq_y) > 0 and tau > 0:
                        P2 = bs_price("call", s[t], K2, r, q, sigma, tau)
                        px2 = P2 * (1.0 + costs.slippage * np.sign(dq_y))
                        c2 = trade_cost(np.array([dq_y * px2]), np.array([dq_y]), costs)[0]
                        cash -= dq_y * px2 + c2
                        y = yt

            else:
                K2, K3 = hedge_Ks
                xt, yt, zt = solve_dgv_positions(s[t], times[t], opt, K2, K3, r, q, sigma, opt.T)

                # threshold on aggregate delta change
                d2 = bs_delta("call", s[t], K2, r, q, sigma, max(tau, 1e-9))
                d3 = bs_delta("call", s[t], K3, r, q, sigma, max(tau, 1e-9))
                dL = bs_delta(opt.kind, s[t], opt.K, r, q, sigma, max(tau, 1e-9))
                agg_delta_old = x + y * d2 + z * d3 - dL
                agg_delta_new = xt + yt * d2 + zt * d3 - dL

                do_trade = True
                if delta_thresh is not None and abs(agg_delta_new - agg_delta_old) <= delta_thresh:
                    do_trade = False
                    xt, yt, zt = x, y, z

                if do_trade:
                    dq_x = xt - x
                    dq_y = yt - y
                    dq_z = zt - z
                    if abs(dq_x) > 0:
                        px = effective_trade_price(np.array([s[t]]), np.array([dq_x]), costs)[0]
                        c = trade_cost(np.array([dq_x * px]), np.array([dq_x]), costs)[0]
                        cash -= dq_x * px + c
                        x = xt
                    if tau > 0:
                        if abs(dq_y) > 0:
                            P2 = bs_price("call", s[t], K2, r, q, sigma, tau)
                            px2 = P2 * (1.0 + costs.slippage * np.sign(dq_y))
                            c2 = trade_cost(np.array([dq_y * px2]), np.array([dq_y]), costs)[0]
                            cash -= dq_y * px2 + c2
                            y = yt
                        if abs(dq_z) > 0:
                            P3 = bs_price("call", s[t], K3, r, q, sigma, tau)
                            px3 = P3 * (1.0 + costs.slippage * np.sign(dq_z))
                            c3 = trade_cost(np.array([dq_z * px3]), np.array([dq_z]), costs)[0]
                            cash -= dq_z * px3 + c3
                            z = zt

        # settle
        payoff = max(0.0, (s[-1] - opt.K)) if opt.kind == "call" else max(0.0, (opt.K - s[-1]))
        # close positions at final price
        if abs(x) > 0:
            dq = -x
            px = effective_trade_price(np.array([s[-1]]), np.array([dq]), costs)[0]
            c = trade_cost(np.array([dq * px]), np.array([dq]), costs)[0]
            cash -= dq * px + c
            x = 0.0

        # hedge options expire worthless or intrinsic at T
        # since hedge options are same maturity, their settlement is intrinsic
        # but we maintained neutrality; simply mark to intrinsic and close:
        if mode == "dg":
            K2 = hedge_Ks[0]
            payoff2 = max(0.0, s[-1] - K2)
            cash += y * payoff2
            y = 0.0
        else:
            K2, K3 = hedge_Ks
            payoff2 = max(0.0, s[-1] - K2)
            payoff3 = max(0.0, s[-1] - K3)
            cash += y * payoff2 + z * payoff3
            y = z = 0.0

        pnl[p] = cash - payoff

    summary = {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)),
        "min_pnl": float(np.min(pnl)),
        "max_pnl": float(np.max(pnl)),
        "hit_rate": float(np.mean(pnl > 0)),
        "npaths": int(npaths),
    }
    return pnl, summary


# =========================
# CLI / Orchestration
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic hedging engine with delta / DG / DGV / Leland modes")
    # Data
    p.add_argument("--paths", default="", help="CSV of paths (columns are paths). If omitted, --simulate is used.")
    p.add_argument("--simulate", action="store_true", help="Simulate GBM paths if no CSV provided.")
    p.add_argument("--S0", type=float, default=100.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--r", type=float, default=0.0)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=252)
    p.add_argument("--npaths", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)

    # Option liability
    p.add_argument("--option", choices=["call", "put"], default="call")
    p.add_argument("--K", type=float, default=100.0)

    # Mode & hedge options
    p.add_argument("--mode", choices=["delta", "dg", "dgv", "leland"], default="delta")
    p.add_argument("--hedge-K", default="", help="Hedge strike(s): one strike for dg; two comma-separated for dgv.")

    # Rebalance control
    p.add_argument("--delta-thresh", type=float, default=None, help="Optional |Δ change| threshold to trigger re-hedge.")
    # Costs
    p.add_argument("--tc-prop", type=float, default=0.0, help="Proportional cost as fraction (e.g., 0.001=10 bps).")
    p.add_argument("--tc-bps", type=float, default=None, help="Alternative: proportional cost in bps (e.g., 10).")
    p.add_argument("--tc-fixed", type=float, default=0.0, help="Fixed cost per trade (currency).")
    p.add_argument("--slippage", type=float, default=0.0, help="Slippage fraction.")
    p.add_argument("--slippage-bps", type=float, default=None, help="Alternative slippage in bps.")

    # Output
    p.add_argument("--outdir", default="hedge_out")
    p.add_argument("--save-ledger", action="store_true", help="Save ledger for first path (delta mode).")
    p.add_argument("--save-pnl", action="store_true", help="Save per-path PnL.")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Costs
    prop = args.tc_prop if args.tc_bps is None else args.tc_bps / 1e4
    slip = args.slippage if args.slippage_bps is None else args.slippage_bps / 1e4
    costs = Costs(prop=float(prop), fixed=float(args.tc_fixed), slippage=float(slip))

    # Paths
    if args.paths:
        S, times = load_paths_csv(args.paths)
        # scale maturity T to provided T
        times = np.linspace(0.0, args.T, len(times))
    else:
        if not args.simulate:
            # default to simulate if no CSV
            args.simulate = True
        S, times = simulate_gbm(args.S0, args.mu, args.sigma, args.T, args.steps, args.npaths, args.seed)

    opt = EuropeanOption(kind=args.option, K=args.K, T=args.T)

    if args.mode == "delta":
        pnl, summary, ledger = delta_hedge_path(
            S, times, opt, r=args.r, q=args.q, sigma=args.sigma, costs=costs, delta_thresh=args.delta_thresh
        )
        if args.save_ledger:
            ledger.to_csv(outdir / "ledger_first_path.csv", index=False)
    elif args.mode == "leland":
        dt = times[1] - times[0]
        sigma_hat = leland_sigma(args.sigma, costs.prop, dt)
        pnl, summary, ledger = delta_hedge_path(
            S, times, opt, r=args.r, q=args.q, sigma=sigma_hat, costs=costs, delta_thresh=args.delta_thresh
        )
        if args.save_ledger:
            ledger.to_csv(outdir / "ledger_first_path.csv", index=False)
        summary["sigma_leland"] = float(sigma_hat)
    elif args.mode in ("dg", "dgv"):
        if args.mode == "dg":
            if not args.hedge_K:
                raise SystemExit("--hedge-K (single strike) required for mode=dg")
            hedge_Ks = [float(args.hedge_K)]
        else:
            parts = [x.strip() for x in args.hedge_K.split(",") if x.strip()]
            if len(parts) != 2:
                raise SystemExit("--hedge-K requires two comma-separated strikes for mode=dgv")
            hedge_Ks = [float(parts[0]), float(parts[1])]
        pnl, summary = multi_greek_hedge_path(
            S, times, opt, args.mode, r=args.r, q=args.q, sigma=args.sigma, costs=costs, hedge_Ks=hedge_Ks, delta_thresh=args.delta_thresh
        )
    else:
        raise SystemExit("Unknown mode")

    # Save & print
    if args.save_pnl:
        pd.DataFrame({"pnl": pnl}).to_csv(outdir / "pnl.csv", index=False)
    pd.Series(summary).to_json(outdir / "summary.json", indent=2)

    # Console summary
    print(pd.Series(summary))


if __name__ == "__main__":
    main()
