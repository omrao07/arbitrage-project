# backend/execution_plus/pricer.py
"""
Lightweight pricing utilities for arbitrage, routing, and risk.

Features
--------
- Spot & forward/fair value (carry model with rates/dividend yield)
- Black-Scholes(-Merton) European calls/puts (continuous div yield)
- Full greeks (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility solver (robust bisection + optional Newton)
- Basket and spread pricing helpers
- Futures basis / carry analytics
- Perpetual swap (perps) fair funding helpers
- Tiny piecewise-constant yield curve

No third-party deps. Uses math.erf for normal CDF.

Usage (quick):
--------------
from backend.execution_plus.pricer import Pricer, BSResult, YieldCurve
px = Pricer()
res = px.bs_price(S=100, K=95, T=0.5, r=0.03, q=0.0, vol=0.35, call=True)
print(res.price, res.delta, res.vega)

# With adapter quotes:
from backend.execution_plus.adapters import AdapterRegistry
q = AdapterRegistry.get("BINANCE").get_quote("BTCUSDT")
fwd = px.forward(S=q.mid, T=7/365, r=0.05)  # 1-week fair fwd
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any


# ----------------------------- Yield curve -----------------------------

@dataclass
class YieldCurve:
    """
    Very small piecewise-constant curve: list of (time_years, rate) sorted by time.
    Interprets rate as continuously compounded annualized.
    """
    points: List[Tuple[float, float]]  # [(t0, r0), (t1, r1), ...] t>0

    def r(self, T: float, default: float = 0.0) -> float:
        if not self.points or T <= 0:
            return default
        pts = sorted(self.points, key=lambda x: x[0])
        rlast = pts[-1][1]
        for t, r in pts:
            if T <= t:
                return r
        return rlast


# ----------------------------- Results ---------------------------------

@dataclass
class BSResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    d1: float
    d2: float
    inputs: Dict[str, Any]


# ----------------------------- Math utils ------------------------------

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)

def _phi(x: float) -> float:
    """Standard normal PDF."""
    return _INV_SQRT2PI * math.exp(-0.5 * x * x)

def _Phi(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))

def _safe(x: Optional[float], fallback: float = 0.0) -> float:
    try:
        return float(x) # type: ignore
    except Exception:
        return fallback

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------- Pricer ----------------------------------

class Pricer:
    # ---------- basics ----------

    @staticmethod
    def forward(S: float, T: float, r: float, q: float = 0.0) -> float:
        """
        Fair forward price under continuous compounding:
        F = S * exp((r - q) * T)
        """
        S, T, r, q = map(float, (S, T, r, q))
        if T <= 0:
            return S
        return S * math.exp((r - q) * max(0.0, T))

    @staticmethod
    def carry_basis(F: float, S: float, T: float) -> float:
        """
        Implied carry (r - q) from F, S, T:
        r - q = ln(F/S)/T
        """
        F, S, T = map(float, (F, S, T))
        if T <= 0 or S <= 0:
            return 0.0
        return math.log(max(1e-12, F / S)) / T

    # ---------- Black-Scholes-Merton ----------

    @staticmethod
    def _d1_d2(S: float, K: float, T: float, r: float, q: float, vol: float) -> Tuple[float, float]:
        if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
            return float("nan"), float("nan")
        vt = vol * math.sqrt(T)
        m = math.log(S / K)
        d1 = (m + (r - q + 0.5 * vol * vol) * T) / vt
        d2 = d1 - vt
        return d1, d2

    def bs_price(
        self, *,
        S: float, K: float, T: float,
        r: float, q: float = 0.0, vol: float,
        call: bool = True
    ) -> BSResult:
        """
        Black‑Scholes price and greeks (European).
        q: continuous dividend yield (or foreign rate in FX)
        """
        S, K, T, r, q, vol = map(float, (S, K, T, r, q, vol))
        disc_r = math.exp(-r * max(0.0, T))
        disc_q = math.exp(-q * max(0.0, T))

        if T <= 0 or vol <= 0:
            # intrinsic at expiry / no variance
            intrinsic = max(0.0, S - K) if call else max(0.0, K - S)
            price = intrinsic
            # degenerate greeks
            return BSResult(price, 0.0, 0.0, 0.0, 0.0, disc_r * (T>0), float("nan"), float("nan"),
                            {"S": S, "K": K, "T": T, "r": r, "q": q, "vol": vol, "call": call})

        d1, d2 = self._d1_d2(S, K, T, r, q, vol)
        if call:
            price = disc_q * S * _Phi(d1) - disc_r * K * _Phi(d2)
            delta = disc_q * _Phi(d1)
            rho = K * T * disc_r * _Phi(d2)  # ∂C/∂r
        else:
            price = disc_r * K * _Phi(-d2) - disc_q * S * _Phi(-d1)
            delta = -disc_q * _Phi(-d1)
            rho = -K * T * disc_r * _Phi(-d2)  # ∂P/∂r

        gamma = (disc_q * _phi(d1)) / (S * vol * math.sqrt(T))
        vega = S * disc_q * _phi(d1) * math.sqrt(T)

        # Black-Scholes Theta (calendar decay, per year)
        if call:
            theta = (-S * disc_q * _phi(d1) * vol / (2 * math.sqrt(T))
                     - r * K * disc_r * _Phi(d2)
                     + q * S * disc_q * _Phi(d1))
        else:
            theta = (-S * disc_q * _phi(d1) * vol / (2 * math.sqrt(T))
                     + r * K * disc_r * _Phi(-d2)
                     - q * S * disc_q * _Phi(-d1))

        return BSResult(price, delta, gamma, vega, theta, rho, d1, d2,
                        {"S": S, "K": K, "T": T, "r": r, "q": q, "vol": vol, "call": call})

    # ---------- Implied volatility ----------

    def implied_vol(
        self, *,
        price: float, S: float, K: float, T: float, r: float, q: float = 0.0, call: bool = True,
        vol_lo: float = 1e-4, vol_hi: float = 5.0, tol: float = 1e-6, max_iter: int = 100
    ) -> float:
        """
        Robust bisection on Black-Scholes price. Returns vol in [vol_lo, vol_hi].
        """
        target = float(price)
        lo, hi = float(vol_lo), float(vol_hi)
        # clamp to arbitrage bounds (intrinsic <= price <= forward bound)
        intrinsic = max(0.0, S - K) if call else max(0.0, K - S)
        if target <= intrinsic + 1e-12:
            return 0.0

        def f(sig: float) -> float:
            return self.bs_price(S=S, K=K, T=T, r=r, q=q, vol=sig, call=call).price - target

        flo, fhi = f(lo), f(hi)
        # If bounds wrong, expand hi
        it_expand = 0
        while flo * fhi > 0 and it_expand < 10:
            hi *= 2.0
            fhi = f(hi)
            it_expand += 1

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if abs(fm) < tol or (hi - lo) < tol:
                return _clip(mid, vol_lo, hi)
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return _clip(0.5 * (lo + hi), vol_lo, hi)

    # ---------- Parity & sanity ----------

    @staticmethod
    def put_call_parity(C: float, P: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
        """
        Returns parity residual: C - P - (S*e^{-qT} - K*e^{-rT}).
        ~0 if parity holds.
        """
        disc_r = math.exp(-r * max(0.0, T))
        disc_q = math.exp(-q * max(0.0, T))
        return float(C) - float(P) - (float(S) * disc_q - float(K) * disc_r)

    # ---------- Baskets & spreads ----------

    @staticmethod
    def basket_spot(weights: List[float], spots: List[float]) -> float:
        """
        Linear basket spot: sum_i w_i * S_i
        """
        if len(weights) != len(spots):
            raise ValueError("weights and spots length mismatch")
        return sum(float(w) * float(s) for w, s in zip(weights, spots))

    def basket_forward(self, weights: List[float], spots: List[float], T: float, r: float, q: float = 0.0) -> float:
        if len(weights) != len(spots):
            raise ValueError("weights and spots length mismatch")
        fwd = [self.forward(S=s, T=T, r=r, q=q) for s in spots]
        return self.basket_spot(weights, fwd)

    @staticmethod
    def spread_price(a_price: float, b_price: float, a_weight: float = 1.0, b_weight: float = -1.0) -> float:
        """
        Simple linear spread value: a_weight*A + b_weight*B.
        """
        return float(a_weight) * float(a_price) + float(b_weight) * float(b_price)

    # ---------- Futures & perps analytics ----------

    @staticmethod
    def futures_fair_from_basis(S: float, basis_annual: float, T: float) -> float:
        """
        If you observe annualized basis (F/S - 1)/T -> basis_annual, back out F.
        Uses continuous approx: F = S * exp(basis_annual * T)
        """
        return float(S) * math.exp(float(basis_annual) * max(0.0, float(T)))

    @staticmethod
    def annualized_basis(S: float, F: float, T: float) -> float:
        if T <= 0 or S <= 0:
            return 0.0
        return math.log(float(F) / float(S)) / float(T)

    @staticmethod
    def perps_fair_rate(r: float, q: float = 0.0, fee_bps: float = 0.0) -> float:
        """
        Perps fair funding (very rough): r - q - fees
        (continuous compounding approximation → per-year rate)
        """
        return float(r) - float(q) - float(fee_bps) / 10_000.0

    @staticmethod
    def perps_payment(notional: float, funding_rate_8h: float) -> float:
        """
        Payment per funding window (8h): notional * rate_8h.
        If given an annual rate R, convert outside: rate_8h = R * (8/24) / 365
        """
        return float(notional) * float(funding_rate_8h)


# ----------------------------- Tiny CLI --------------------------------

if __name__ == "__main__":
    px = Pricer()

    # Example: BS call
    res = px.bs_price(S=100, K=95, T=0.5, r=0.03, q=0.01, vol=0.35, call=True)
    print("Call:", round(res.price, 4), "Δ", round(res.delta, 4), "Γ", round(res.gamma, 6),
          "ν", round(res.vega, 4), "Θ/yr", round(res.theta, 4), "ρ", round(res.rho, 4))

    # Implied vol from price
    imp = px.implied_vol(price=res.price, S=100, K=95, T=0.5, r=0.03, q=0.01, call=True)
    print("Implied vol:", round(imp, 6))

    # Basket forward & spread
    bfw = px.basket_forward([0.6, 0.4], [200, 50], T=30/365, r=0.04, q=0.0)
    spr = px.spread_price(100.0, 101.2)
    print("Basket fwd:", round(bfw, 4), "Spread:", round(spr, 4))

    # Basis
    basis = px.annualized_basis(100, 101, 30/365)
    print("Ann. basis:", round(basis, 6))