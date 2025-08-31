# backend/quant/greeks.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

# ===================== Utilities =====================

OptionType = Literal["call", "put"]

_EPS = 1e-12
_IV_MIN = 1e-6
_IV_MAX = 6.0

def _sign(is_call: bool) -> float:
    return 1.0 if is_call else -1.0

def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _ncdf(x: float) -> float:
    """Standard normal CDF (erf-based; good to ~1e-7)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _year_fraction(days: float, basis: Literal["ACT/365","ACT/252","ACT/360"]="ACT/365") -> float:
    if basis == "ACT/252": return max(_EPS, days / 252.0)
    if basis == "ACT/360": return max(_EPS, days / 360.0)
    return max(_EPS, days / 365.0)

# ===================== Data classes =====================

@dataclass
class BSMResult:
    price: float
    delta: float
    gamma: float
    vega: float         # per 1.00 vol (not 1%)
    theta: float        # per year
    rho: float

@dataclass
class B76Result:
    price: float
    delta_f: float      # w.r.t. forward F
    gamma_f: float
    vega: float         # per 1.00 vol
    theta: float        # per year (holding F constant)
    rho: float          # ∂price/∂r (via discounting)

# ===================== Black–Scholes–Merton =====================

def bsm_price(
    s: float, k: float, t: float, r: float, q: float, vol: float, typ: OptionType
) -> float:
    """
    BSM price for spot-based option with cont. dividend yield q.
    s: spot, k: strike, t: years to expiry, r: risk-free, q: dividend yield, vol: sigma
    """
    if t <= 0.0 or vol <= 0.0:
        intrinsic = max(0.0, _sign(typ=="call") * (s*math.exp(-q*t) - k*math.exp(-r*t)))
        # For t->0 or vol->0 use discounted intrinsic approx
        return intrinsic

    st = s
    is_call = (typ == "call")
    sqrt_t = math.sqrt(t)
    d1 = (math.log((st + _EPS)/k) + (r - q + 0.5*vol*vol)*t) / (vol*sqrt_t)
    d2 = d1 - vol*sqrt_t

    disc_r = math.exp(-r*t)
    disc_q = math.exp(-q*t)

    if is_call:
        return disc_q*st*_ncdf(d1) - disc_r*k*_ncdf(d2)
    else:
        return disc_r*k*_ncdf(-d2) - disc_q*st*_ncdf(-d1)

def bsm_greeks(
    s: float, k: float, t: float, r: float, q: float, vol: float, typ: OptionType
) -> BSMResult:
    is_call = (typ == "call")
    if t <= 0.0 or vol <= 0.0 or s <= 0.0 or k <= 0.0:
        # Degenerate: use robust finite approximations around zero
        price = bsm_price(s,k,t,r,q,vol,typ)
        return BSMResult(price=price, delta= (1.0 if (is_call and s>k) else (-1.0 if (not is_call and s<k) else 0.0)),
                         gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s/k) + (r - q + 0.5*vol*vol)*t) / (vol*sqrt_t)
    d2 = d1 - vol*sqrt_t
    disc_r = math.exp(-r*t)
    disc_q = math.exp(-q*t)

    # Price
    if is_call:
        price = disc_q*s*_ncdf(d1) - disc_r*k*_ncdf(d2)
    else:
        price = disc_r*k*_ncdf(-d2) - disc_q*s*_ncdf(-d1)

    # Greeks
    delta = disc_q * (_ncdf(d1) if is_call else (_ncdf(d1)-1.0))
    gamma = disc_q * _phi(d1) / (s * vol * sqrt_t)
    vega  = s * disc_q * _phi(d1) * sqrt_t
    # Theta (per year)
    theta_time = - (s * disc_q * _phi(d1) * vol) / (2.0 * sqrt_t)
    theta_carry = q * s * disc_q * (_ncdf(d1) if is_call else (_ncdf(d1)-1.0))
    theta_rate  = - r * k * disc_r * (_ncdf(d2) if is_call else _ncdf(-d2))
    theta = theta_time - theta_carry + theta_rate
    # Rho
    rho = k * t * disc_r * (_ncdf(d2) if is_call else (_ncdf(d2)-1.0))

    return BSMResult(price, delta, gamma, vega, theta, rho)

# ===================== Black '76 (futures options) =====================

def b76_price(
    f: float, k: float, t: float, r: float, vol: float, typ: OptionType
) -> float:
    """
    Black '76 for options on futures (forward F).
    Returns discounted expectation: exp(-r t) * [F N(d1) - K N(d2)] for calls.
    """
    if t <= 0.0 or vol <= 0.0:
        intrinsic = max(0.0, _sign(typ=="call") * (f - k)) * math.exp(-r*t)
        return intrinsic

    sqrt_t = math.sqrt(t)
    d1 = (math.log((f + _EPS)/k) + 0.5*vol*vol*t) / (vol*sqrt_t)
    d2 = d1 - vol*sqrt_t
    disc = math.exp(-r*t)
    if typ == "call":
        return disc * (f*_ncdf(d1) - k*_ncdf(d2))
    else:
        return disc * (k*_ncdf(-d2) - f*_ncdf(-d1))

def b76_greeks(
    f: float, k: float, t: float, r: float, vol: float, typ: OptionType
) -> B76Result:
    if t <= 0.0 or vol <= 0.0 or f <= 0.0 or k <= 0.0:
        price = b76_price(f,k,t,r,vol,typ)
        return B76Result(price=price, delta_f=(1.0 if (typ=="call" and f>k) else (-1.0 if (typ=="put" and f<k) else 0.0)),
                         gamma_f=0.0, vega=0.0, theta=0.0, rho=0.0)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(f/k) + 0.5*vol*vol*t) / (vol*sqrt_t)
    d2 = d1 - vol*sqrt_t
    disc = math.exp(-r*t)
    is_call = (typ == "call")

    price = b76_price(f,k,t,r,vol,typ)
    delta_f = disc * (_ncdf(d1) if is_call else (_ncdf(d1)-1.0))
    gamma_f = disc * _phi(d1) / (f * vol * sqrt_t)
    vega    = disc * f * _phi(d1) * sqrt_t
    theta   = - (disc * f * _phi(d1) * vol) / (2.0 * sqrt_t) - r * price
    rho     = -t * price  # ∂/∂r of discount factor component
    return B76Result(price, delta_f, gamma_f, vega, theta, rho)

# ===================== Implied Vol =====================

def implied_vol_bsm(
    price: float, s: float, k: float, t: float, r: float, q: float, typ: OptionType,
    *, tol: float = 1e-7, max_iter: int = 100
) -> Optional[float]:
    """Implied vol for BSM using bracketed Newton (safe). Returns None if no solution."""
    # Quick no-arbitrage bounds: intrinsic <= price <= forward-bound
    intrinsic = max(0.0, _sign(typ=="call") * (s*math.exp(-q*t) - k*math.exp(-r*t)))
    if price < intrinsic - 1e-12:
        return None

    lo, hi = _IV_MIN, _IV_MAX
    v = 0.5 * (lo + hi)
    for _ in range(max_iter):
        res = bsm_greeks(s,k,t,r,q,v,typ)
        diff = res.price - price
        if abs(diff) < tol:
            return max(_IV_MIN, min(_IV_MAX, v))
        # Newton step with vega safeguard
        if res.vega > 1e-10:
            v_new = v - diff / res.vega
        else:
            v_new = v
        # Keep within bracket, else bisect
        if v_new <= lo or v_new >= hi or not math.isfinite(v_new):
            v_new = 0.5 * (lo + hi)
        # Update bracket
        if diff > 0:  # model too high -> vol too high
            hi = v_new
        else:
            lo = v_new
        v = v_new
    return max(_IV_MIN, min(_IV_MAX, v))

def implied_vol_b76(
    price: float, f: float, k: float, t: float, r: float, typ: OptionType,
    *, tol: float = 1e-7, max_iter: int = 100
) -> Optional[float]:
    lo, hi = _IV_MIN, _IV_MAX
    v = 0.5 * (lo + hi)
    for _ in range(max_iter):
        res = b76_greeks(f,k,t,r,v,typ)
        diff = res.price - price
        if abs(diff) < tol:
            return max(_IV_MIN, min(_IV_MAX, v))
        if res.vega > 1e-10:
            v_new = v - diff / res.vega
        else:
            v_new = v
        if v_new <= lo or v_new >= hi or not math.isfinite(v_new):
            v_new = 0.5 * (lo + hi)
        if diff > 0:
            hi = v_new
        else:
            lo = v_new
        v = v_new
    return max(_IV_MIN, min(_IV_MAX, v))

# ===================== Parity & Helpers =====================

def put_from_call_bsm(call: float, s: float, k: float, t: float, r: float, q: float) -> float:
    """Put via put–call parity under BSM."""
    return call - s*math.exp(-q*t) + k*math.exp(-r*t)

def call_from_put_bsm(put: float, s: float, k: float, t: float, r: float, q: float) -> float:
    return put + s*math.exp(-q*t) - k*math.exp(-r*t)

def days_to_years(days: float, basis: Literal["ACT/365","ACT/252","ACT/360"]="ACT/365") -> float:
    return _year_fraction(days, basis)

# ===================== Extras (optional) =====================

def charm(s: float, k: float, t: float, r: float, q: float, vol: float, typ: OptionType) -> float:
    """
    ∂Δ/∂t (per year). Useful for dynamics of delta hedges.
    """
    if t <= 0.0 or vol <= 0.0 or s <= 0.0 or k <= 0.0:
        return 0.0
    is_call = (typ == "call")
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s/k) + (r - q + 0.5*vol*vol)*t) / (vol*sqrt_t)
    d2 = d1 - vol*sqrt_t
    disc_q = math.exp(-q*t)
    term1 = -disc_q * _phi(d1) * (2*(r - q) * t - d2 * vol * sqrt_t) / (2 * t * vol * sqrt_t)
    return term1 if is_call else term1 * -1.0

def vanna(s: float, k: float, t: float, r: float, q: float, vol: float) -> float:
    """
    ∂^2 Price / ∂S ∂σ (approx). Often used for volatility surface risk.
    """
    if t <= 0.0 or vol <= 0.0 or s <= 0.0 or k <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s/k) + (r - q + 0.5*vol*vol)*t) / (vol*sqrt_t)
    disc_q = math.exp(-q*t)
    return disc_q * _phi(d1) * sqrt_t * (1.0 - d1 / (vol*sqrt_t))

# ===================== Smoke test =====================

if __name__ == "__main__":  # pragma: no cover
    s, k, r, q, t, sig = 100.0, 100.0, 0.02, 0.01, 30/365.0, 0.25
    for typ in ("call","put"):
        res = bsm_greeks(s,k,t,r,q,sig,typ)  # type: ignore
        print(typ.upper(), "BSM price=", round(res.price,4), "delta=", round(res.delta,4),
              "gamma=", round(res.gamma,6), "vega=", round(res.vega,4), "theta/yr=", round(res.theta,4), "rho=", round(res.rho,4))
        iv = implied_vol_bsm(res.price, s,k,t,r,q,typ)  # type: ignore
        print(" recovered IV:", round(iv or 0,6))

    f = 100.0
    res76 = b76_greeks(f,k,t,r,sig,"call")  # type: ignore
    print("B76 CALL price=", round(res76.price,4), "delta_f=", round(res76.delta_f,4),
          "gamma_f=", round(res76.gamma_f,6), "vega=", round(res76.vega,4))