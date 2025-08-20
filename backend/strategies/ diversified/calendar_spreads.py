"""
Calendar Spreads Strategy
-------------------------
Finds mispricings between near- and far-dated options of the same strike/type.

Long calendar:  BUY far-dated option, SELL near-dated option  (seek cheap time-value, positive theta carry)
Short calendar: SELL far-dated option, BUY near-dated option (opposite scenario)

Inputs
------
A pandas DataFrame `options` with columns:
  - date:            timestamp (string or datetime)
  - underlying:      str, e.g., "AAPL"
  - type:            "call" | "put"
  - strike:          float
  - expiry:          timestamp (string or datetime)
  - price:           mid price (float)
  - iv:              (optional) implied vol as decimal, per-leg; if missing, falls back to `fallback_iv`
  - r:               (optional) risk-free rate as decimal (per row); otherwise use `risk_free_rate` arg
  - q:               (optional) dividend yield as decimal (per row); default 0
  - spot:            (optional) underlying spot at row; or pass `spot_overrides` map/single float

Outputs
-------
- generate_signals(): per (strike,type,near,far) row with model debit, market debit, edge bps, theta edge, and signal
- backtest(): naive PnL attribution using mark-to-model drift & realized re-mark of market debit changes

Notes
-----
This module is analytics/backtest-ready. If you want *live OMS* integration,
wrap `place_orders()` calls via your Strategy base (like we did for FX basis).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ------------- Black–Scholes helpers -------------
SQRT_2PI = math.sqrt(2 * math.pi)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def _bs_d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def _bs_price_theta(S: float, K: float, T: float, r: float, q: float, sigma: float, opt_type: str) -> Tuple[float, float]:
    """
    Returns (price, theta_per_year). Theta sign: negative for long options.
    """
    opt_type = opt_type.lower()
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # intrinsic approximation when degenerate
        if opt_type == "call":
            price = max(0.0, S - K) * math.exp(-q * T)  # rough
        else:
            price = max(0.0, K - S) * math.exp(-r * T)
        return price, 0.0

    d1 = _bs_d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * math.sqrt(T)

    if opt_type == "call":
        price = S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        theta = (
            - (S * math.exp(-q * T) * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * _norm_cdf(d2)
            + q * S * math.exp(-q * T) * _norm_cdf(d1)
        )
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)
        theta = (
            - (S * math.exp(-q * T) * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * _norm_cdf(-d2)
            - q * S * math.exp(-q * T) * _norm_cdf(-d1)
        )

    # theta returned in price units per year
    return price, theta


# ------------- Core strategy -------------
@dataclass
class CalendarConfig:
    fallback_iv: float = 0.30     # if iv column missing/NaN, use this
    risk_free_rate: float = 0.03  # used if per-row r not provided
    dividend_yield: float = 0.0   # used if per-row q not provided
    min_days_near: int = 7        # skip weeklies shorter than this
    max_days_near: int = 45       # typical near leg window
    min_days_far: int = 25        # far leg must be at least this far
    edge_bps_entry: float = 25.0  # enter when |edge_bps| >= this
    theta_pref: str = "positive"  # "positive" (prefer theta>0), "any"
    price_floor: float = 0.05     # ignore penny options


class CalendarSpreadsStrategy:
    """
    Pair the same (strike, type) across near/far expiries.
    Model debit = BS(far) - BS(near)
    Market debit = mkt_far - mkt_near

    edge_bps = 10,000 * (model_debit - market_debit) / spot
      > 0  -> market cheaper than model -> LONG calendar
      < 0  -> market richer than model  -> SHORT calendar
    """

    def __init__(self, options: pd.DataFrame, config: Optional[CalendarConfig] = None, spot_overrides: Optional[float | dict] = None):
        self.df = options.copy()
        self.cfg = config or CalendarConfig()
        self.spot_overrides = spot_overrides

        # normalize types
        self.df["type"] = self.df["type"].str.lower()
        # parse dates
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["expiry"] = pd.to_datetime(self.df["expiry"])
        # precompute days to expiry & T in years (ACT/365)
        self.df["days_to_expiry"] = (self.df["expiry"] - self.df["date"]).dt.days.clip(lower=0)
        self.df["T"] = self.df["days_to_expiry"] / 365.0

    def _spot_for_row(self, row: pd.Series) -> Optional[float]:
        # precedence: row.spot > overrides map/single > NaN
        if pd.notna(row.get("spot", np.nan)):
            return float(row["spot"])
        if isinstance(self.spot_overrides, (int, float)):
            return float(self.spot_overrides)
        if isinstance(self.spot_overrides, dict):
            key = str(row.get("underlying"))
            if key in self.spot_overrides:
                return float(self.spot_overrides[key])
        return None

    def _rate_for_row(self, row: pd.Series) -> float:
        if pd.notna(row.get("r", np.nan)):
            return float(row["r"])
        return float(self.cfg.risk_free_rate)

    def _yield_for_row(self, row: pd.Series) -> float:
        if pd.notna(row.get("q", np.nan)):
            return float(row["q"])
        return float(self.cfg.dividend_yield)

    def _iv_for_row(self, row: pd.Series) -> float:
        v = row.get("iv", np.nan)
        if pd.notna(v) and v > 0:
            return float(v)
        return float(self.cfg.fallback_iv)

    def _pair_near_far(self, grp: pd.DataFrame) -> pd.DataFrame:
        """
        For one (date, underlying, type, strike) group, return best near/far pairs within windows.
        """
        g = grp.sort_values("expiry")
        # near candidates window
        near = g[(g["days_to_expiry"] >= self.cfg.min_days_near) & (g["days_to_expiry"] <= self.cfg.max_days_near)]
        if near.empty:
            return pd.DataFrame()
        # far candidates window
        far  = g[g["days_to_expiry"] >= self.cfg.min_days_far]
        if far.empty:
            return pd.DataFrame()

        pairs = []
        # choose the *closest* near, and the *next available* far for that date/strike/type
        n = near.iloc[0]
        for _, f in far.iterrows():
            if f["expiry"] <= n["expiry"]:
                continue
            pairs.append((n, f))
            break

        if not pairs:
            return pd.DataFrame()

        rows = []
        for n, f in pairs:
            rows.append({
                "date": n["date"],
                "underlying": n["underlying"],
                "type": n["type"],
                "strike": float(n["strike"]),
                "near_expiry": n["expiry"],
                "far_expiry": f["expiry"],
                "near_price": float(n["price"]),
                "far_price": float(f["price"]),
                "near_T": float(n["T"]),
                "far_T": float(f["T"]),
                "near_iv": self._iv_for_row(n),
                "far_iv": self._iv_for_row(f),
                "r": self._rate_for_row(n),
                "q": self._yield_for_row(n),
                "spot": self._spot_for_row(n),
            })
        return pd.DataFrame(rows)

    def _build_pairs(self) -> pd.DataFrame:
        cols = ["date", "underlying", "type", "strike"]
        pairs = (
            self.df
            .sort_values(["date", "underlying", "type", "strike", "expiry"])
            .groupby(cols, group_keys=False)
            .apply(self._pair_near_far)
            .reset_index(drop=True)
        )
        # drop invalid
        pairs = pairs.dropna(subset=["spot"])
        pairs = pairs[(pairs["near_price"] >= self.cfg.price_floor) & (pairs["far_price"] >= self.cfg.price_floor)]
        return pairs

    def generate_signals(self) -> pd.DataFrame:
        """
        Returns rows with:
          model_debit, market_debit, edge_bps, theta_near, theta_far, theta_edge, signal
        signal ∈ {"LONG_CAL", "SHORT_CAL", "HOLD"}
        """
        pairs = self._build_pairs()
        if pairs.empty:
            return pairs

        out = []
        for _, row in pairs.iterrows():
            S = float(row["spot"])
            K = float(row["strike"])
            r = float(row["r"])
            q = float(row["q"])
            typ = str(row["type"]).lower()

            # model prices via BS (per-leg IVs)
            m_near, theta_near = _bs_price_theta(S, K, row["near_T"], r, q, row["near_iv"], typ)
            m_far,  theta_far  = _bs_price_theta(S, K, row["far_T"],  r, q, row["far_iv"],  typ)

            model_debit  = m_far - m_near
            market_debit = float(row["far_price"] - row["near_price"])

            edge = model_debit - market_debit
            edge_bps = 1e4 * edge / max(S, 1e-9)

            theta_edge = theta_far - theta_near  # per year (price units)

            # signal logic
            sig = "HOLD"
            if abs(edge_bps) >= self.cfg.edge_bps_entry:
                if edge > 0:
                    # market cheaper than model -> long calendar (buy far, sell near)
                    if self.cfg.theta_pref != "positive" or theta_edge > 0:
                        sig = "LONG_CAL"
                else:
                    # market richer than model -> short calendar
                    if self.cfg.theta_pref != "positive" or theta_edge < 0:
                        sig = "SHORT_CAL"

            out.append({
                "date": row["date"],
                "underlying": row["underlying"],
                "type": typ,
                "strike": K,
                "near_expiry": row["near_expiry"],
                "far_expiry": row["far_expiry"],
                "spot": S,
                "near_price": row["near_price"],
                "far_price": row["far_price"],
                "near_iv": row["near_iv"],
                "far_iv": row["far_iv"],
                "model_debit": model_debit,
                "market_debit": market_debit,
                "edge": edge,
                "edge_bps": edge_bps,
                "theta_near": theta_near,
                "theta_far": theta_far,
                "theta_edge": theta_edge,
                "signal": sig,
            })

        res = pd.DataFrame(out).sort_values(["date", "underlying", "type", "strike"])
        return res

    def backtest(self, entry_bps: Optional[float] = None, hold_to_near_expiry: bool = True) -> pd.DataFrame:
        """
        Naive backtest:
          - enter when signal fires and |edge_bps| >= threshold
          - exit at near expiry (or when edge flips sign)
          - PnL approximated by change in *market_debit* over holding window

        Returns trade log with entry/exit & PnL (per 1 contract notionally).
        """
        sigs = self.generate_signals()
        if sigs.empty:
            return sigs

        thr = float(entry_bps if entry_bps is not None else self.cfg.edge_bps_entry)

        trades = []
        open_map = {}  # key=(und,type,strike,near,far) -> (entry_debit, side, entry_date)

        for _, row in sigs.iterrows():
            key = (row["underlying"], row["type"], row["strike"], row["near_expiry"], row["far_expiry"])
            side = None
            if abs(row["edge_bps"]) >= thr:
                side = 1 if row["signal"] == "LONG_CAL" else (-1 if row["signal"] == "SHORT_CAL" else 0)

            # open if edge qualifies and no open position
            if side and key not in open_map:
                open_map[key] = (row["market_debit"], side, row["date"])
                continue

            # manage exits
            if key in open_map:
                entry_debit, sgn, entry_dt = open_map[key]
                done = False
                reason = ""
                if not hold_to_near_expiry:
                    # flip exit when edge changes sign across zero
                    if (sgn > 0 and row["edge"] < 0) or (sgn < 0 and row["edge"] > 0):
                        done = True
                        reason = "edge_flip"
                else:
                    # exit on or after near_expiry date
                    if pd.to_datetime(row["date"]).date() >= pd.to_datetime(row["near_expiry"]).date():
                        done = True
                        reason = "near_expiry"

                if done:
                    exit_debit = row["market_debit"]
                    # PnL sign: long calendar profits when market_debit moves UP
                    pnl = (exit_debit - entry_debit) * sgn
                    trades.append({
                        "underlying": row["underlying"],
                        "type": row["type"],
                        "strike": row["strike"],
                        "near_expiry": row["near_expiry"],
                        "far_expiry": row["far_expiry"],
                        "entry_date": entry_dt,
                        "exit_date": row["date"],
                        "entry_debit": entry_debit,
                        "exit_debit": exit_debit,
                        "side": "LONG_CAL" if sgn > 0 else "SHORT_CAL",
                        "reason": reason,
                        "pnl": pnl
                    })
                    del open_map[key]

        return pd.DataFrame(trades)


# ---------------- Quick demo ----------------
if __name__ == "__main__":
    # Tiny synthetic chain
    dates = pd.date_range("2024-01-02", periods=10, freq="D")
    chain = []
    for d in dates:
        for k in [95, 100, 105]:
            for t in ["call", "put"]:
                # two expiries: ~20d (near) and ~50d (far)
                for dd in [20, 50]:
                    chain.append({
                        "date": d,
                        "underlying": "ACME",
                        "type": t,
                        "strike": float(k),
                        "expiry": d + pd.Timedelta(days=dd),
                        "price": 2.0 + 0.01 * (k == 100) + (0.002 if dd == 50 else 0.0),  # far slightly richer
                        "iv": 0.30 + (0.02 if dd == 50 else 0.0),  # term structure
                        "r": 0.03,
                        "q": 0.00,
                        "spot": 100.0
                    })
    df = pd.DataFrame(chain)

    strat = CalendarSpreadsStrategy(df)
    sigs = strat.generate_signals()
    print(sigs.head())

    trades = strat.backtest()
    print(trades.head())