# backend/data/options_adapter.py
"""
Options Market-Data Adapter
---------------------------
Provider: Yahoo Finance via `yfinance` (lightweight, broad coverage).

Capabilities
  • list_expiries(underlying)
  • fetch_chain(underlying, expiry, [strikes], [side])
  • fetch_quote(underlying, expiry, strike, right)
  • surface_snapshot(underlying, [expiry], [grid])  -> IV/greeks grid
  • greeks(S,K,T,r,q,vol, right) & price()          -> Black-Scholes
  • iv_from_price(price, S,K,T,r,q, right)          -> implied vol (brent/bisection)
  • publish(envelope_or_list)                        -> bus stream

Envelopes (audit-friendly, hashed)
{
  "ts": <ms>, "adapter":"options", "provider":"yfinance",
  "kind":"chain|quote|surface|calc", "symbol": "AAPL",
  "payload": {...}, "version":1, "hash":"sha256(...)"
}

Dependencies
  • yfinance (pip install yfinance)
  • numpy
  • Optional: scipy (for fast Brent solver). Fallback to robust bisection included.

Notes
  • Rates/dividends: defaults r=0.0, q=0.0 unless provided.
  • Right: "C" (call), "P" (put).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------- optional deps ----------
try:
    import yfinance as yf  # type: ignore
    _HAS_YF = True
except Exception:
    _HAS_YF = False

import numpy as np

try:
    from scipy.optimize import brentq  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------- bus hook ----------
try:
    from backend.bus.streams import publish_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload):
        head = payload if isinstance(payload, dict) else (payload[0] if payload else {})
        print(f"[stub publish_stream] {stream} <- {json.dumps(head, separators=(',',':'))[:220]}{'...' if isinstance(payload, list) and len(payload)>1 else ''}")

# ---------- optional ledger ----------
def _ledger_append(payload, ledger_path: Optional[str]):
    if not ledger_path:
        return
    try:
        from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
        MerkleLedger(ledger_path).append({"type": "options_md", "payload": payload})
    except Exception:
        pass

# ---------- config & models ----------

@dataclass
class OptConfig:
    provider: str = "yfinance"
    stream: str = "STREAM_OPTIONS_MD"
    ledger_path: Optional[str] = None
    default_rate: float = 0.0     # risk-free r (continuously compounded)
    default_div_yield: float = 0.0  # dividend yield q

@dataclass
class QuoteKey:
    underlying: str
    expiry: str          # "YYYY-MM-DD"
    strike: float
    right: str           # "C" or "P"

@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    iv: Optional[float] = None

# ---------- adapter ----------

class OptionsAdapter:
    def __init__(self, cfg: OptConfig) -> None:
        self.cfg = cfg
        if self.cfg.provider.lower() != "yfinance":
            raise RuntimeError(f"Unsupported provider '{cfg.provider}' (only 'yfinance' implemented).")
        if not _HAS_YF:
            raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    # ----- public: expiries & chains -----

    def list_expiries(self, underlying: str) -> List[str]:
        tk = yf.Ticker(underlying.upper().strip())
        exps = getattr(tk, "options", []) or []
        if not exps:
            raise RuntimeError(f"No option expiries found for {underlying}")
        env = self._envelope(kind="expiries", symbol=underlying.upper(), payload={"expiries": list(exps)})
        _ledger_append(env, self.cfg.ledger_path)
        return exps

    def fetch_chain(
        self,
        underlying: str,
        expiry: str,
        strikes: Optional[Sequence[float]] = None,
        side: Optional[str] = None,  # "C" | "P" | None (both)
    ) -> Dict[str, Any]:
        """
        Return a normalized chain snapshot for a given expiry.
        """
        sym = underlying.upper().strip()
        tk = yf.Ticker(sym)
        try:
            chain = tk.option_chain(expiry)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch chain for {sym} {expiry}: {e}")

        calls = chain.calls.reset_index(drop=True)
        puts = chain.puts.reset_index(drop=True)

        def _norm(df, right: str) -> List[Dict[str, Any]]:
            out = []
            for _, row in df.iterrows():
                k = float(row["strike"])
                if strikes and (k not in strikes):
                    continue
                mid = _mid(row.get("bid"), row.get("ask"), row.get("lastPrice"))
                out.append({
                    "strike": k,
                    "right": right,
                    "last": _f(row.get("lastPrice")),
                    "bid": _f(row.get("bid")),
                    "ask": _f(row.get("ask")),
                    "mid": mid,
                    "volume": _f(row.get("volume")),
                    "openInterest": _f(row.get("openInterest")),
                    "impliedVol": _f(row.get("impliedVolatility")),  # as fraction (e.g., 0.24)
                    "inTheMoney": bool(row.get("inTheMoney", False)),
                    "contractSymbol": row.get("contractSymbol"),
                })
            return out

        out_calls = _norm(calls, "C") if side in (None, "C") else []
        out_puts  = _norm(puts,  "P") if side in (None, "P") else []

        payload = {
            "underlying": sym,
            "expiry": expiry,
            "calls": out_calls,
            "puts": out_puts,
        }
        env = self._envelope(kind="chain", symbol=sym, payload=payload)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    # ----- public: single quote -----

    def fetch_quote(self, key: QuoteKey, *, spot: Optional[float] = None, r: Optional[float] = None, q: Optional[float] = None) -> Dict[str, Any]:
        """
        Fetch a single option quote by (underlying, expiry, strike, right).
        Also computes IV (from mid/last) and greeks with supplied or default r/q.
        """
        chain_env = self.fetch_chain(key.underlying, key.expiry, strikes=[key.strike], side=key.right)
        leg_list = chain_env["payload"]["calls" if key.right == "C" else "puts"]
        if not leg_list:
            raise RuntimeError(f"Option not found: {key}")

        leg = leg_list[0]
        # Spot from yfinance if not provided
        S = float(spot) if spot is not None else _yf_spot(key.underlying)
        K = float(key.strike)
        T = _year_fraction_to_expiry(key.expiry)
        r = float(self.cfg.default_rate if r is None else r)
        q = float(self.cfg.default_div_yield if q is None else q)

        price_ref = leg.get("mid") or leg.get("last")
        iv = None
        if price_ref and S and K and T > 0:
            try:
                iv = iv_from_price(float(price_ref), S, K, T, r, q, key.right)
            except Exception:
                iv = None

        vol_for_greeks = iv if iv is not None else (leg.get("impliedVol") if leg.get("impliedVol") not in (None, float("nan")) else 0.20)

        g = greeks(S, K, T, r, q, float(vol_for_greeks), key.right)
        payload = {
            "key": key.__dict__,
            "spot": S,
            "r": r,
            "q": q,
            "market": {
                "bid": leg.get("bid"),
                "ask": leg.get("ask"),
                "last": leg.get("last"),
                "mid": leg.get("mid"),
                "volume": leg.get("volume"),
                "openInterest": leg.get("openInterest"),
            },
            "iv": iv,
            "greeks": g.__dict__,
        }
        env = self._envelope(kind="quote", symbol=key.underlying.upper(), payload=payload)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    # ----- public: surface snapshot -----

    def surface_snapshot(
        self,
        underlying: str,
        expiry: Optional[str] = None,
        *,
        strikes: Optional[Sequence[float]] = None,
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build an IV/greeks surface for a single expiry (or nearest if None).
        """
        sym = underlying.upper().strip()
        exps = self.list_expiries(sym)
        exp = expiry or exps[0]
        ch = self.fetch_chain(sym, exp, strikes=strikes)

        S = _yf_spot(sym)
        T = _year_fraction_to_expiry(exp)
        r = float(self.cfg.default_rate if r is None else r)
        q = float(self.cfg.default_div_yield if q is None else q)

        def _calc(leg: Dict[str, Any]) -> Dict[str, Any]:
            K = float(leg["strike"])
            right = str(leg["right"])
            mkt = leg.get("mid") or leg.get("last")
            iv_mkt = None
            if mkt and S and T > 0:
                try:
                    iv_mkt = iv_from_price(float(mkt), S, K, T, r, q, right)
                except Exception:
                    iv_mkt = None
            vol = iv_mkt if iv_mkt is not None else (leg.get("impliedVol") if leg.get("impliedVol") not in (None, float("nan")) else 0.20)
            g = greeks(S, K, T, r, q, float(vol), right) # type: ignore
            return {"strike": K, "right": right, "iv": iv_mkt, "greeks": g.__dict__, "mid_or_last": mkt}

        rows: List[Dict[str, Any]] = []
        for leg in ch["payload"]["calls"]:
            rows.append(_calc(leg))
        for leg in ch["payload"]["puts"]:
            rows.append(_calc(leg))

        payload = {
            "underlying": sym,
            "expiry": exp,
            "spot": S,
            "r": r, "q": q, "T": T,
            "rows": rows,
        }
        env = self._envelope(kind="surface", symbol=sym, payload=payload)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    # ----- publish -----

    def publish(self, env_or_envs):
        if isinstance(env_or_envs, list):
            for e in env_or_envs:
                publish_stream(self.cfg.stream, e)
        else:
            publish_stream(self.cfg.stream, env_or_envs)

    # ----- envelope -----

    def _envelope(self, *, kind: str, symbol: str, payload: Dict[str, Any], ts: Optional[int] = None) -> Dict[str, Any]:
        env = {
            "ts": int(ts or time.time() * 1000),
            "adapter": "options",
            "provider": "yfinance",
            "kind": kind,  # "expiries" | "chain" | "quote" | "surface"
            "symbol": symbol,
            "payload": payload,
            "version": 1,
        }
        env["hash"] = hashlib.sha256(json.dumps(env, separators=(",", ":"), sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
        return env


# ======================= Black–Scholes toolkit =======================

SQRT_2PI = math.sqrt(2.0 * math.pi)

def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def _d1_d2(S: float, K: float, T: float, r: float, q: float, vol: float) -> Tuple[float, float]:
    if S <= 0 or K <= 0 or T <= 0 or vol <= 0:
        return float("nan"), float("nan")
    vsqrt = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2

def bs_price(S: float, K: float, T: float, r: float, q: float, vol: float, right: str) -> float:
    """Black-Scholes price for European call/put with continuous dividend yield q."""
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        # intrinsic for T≈0 as a sane fallback
        if right.upper() == "C":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    d1, d2 = _d1_d2(S, K, T, r, q, vol)
    if right.upper() == "C":
        return math.exp(-q * T) * S * _cdf(d1) - math.exp(-r * T) * K * _cdf(d2)
    else:
        return math.exp(-r * T) * K * _cdf(-d2) - math.exp(-q * T) * S * _cdf(-d1)

def greeks(S: float, K: float, T: float, r: float, q: float, vol: float, right: str) -> Greeks:
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        # degenerate case: zeroed greeks with intrinsic price
        price = bs_price(S, K, T, r, q, max(vol, 1e-8), right)
        return Greeks(price=price, delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0, iv=None)

    d1, d2 = _d1_d2(S, K, T, r, q, vol)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    nd1 = _pdf(d1)
    sign = 1.0 if right.upper() == "C" else -1.0

    price = bs_price(S, K, T, r, q, vol, right)
    delta = sign * disc_q * _cdf(sign * d1)
    gamma = disc_q * nd1 / (S * vol * math.sqrt(T))
    vega = S * disc_q * nd1 * math.sqrt(T) * 0.01  # per 1% vol
    theta = (
        -(S * disc_q * nd1 * vol) / (2 * math.sqrt(T))
        - sign * (r * K * disc_r * _cdf(sign * d2) - q * S * disc_q * _cdf(sign * d1))
    ) / 365.0
    rho = sign * K * T * disc_r * _cdf(sign * d2) * 0.01  # per 1% rate

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, iv=None)

def iv_from_price(price: float, S: float, K: float, T: float, r: float, q: float, right: str,
                  *, tol: float = 1e-6, max_iter: int = 100, v_lo: float = 1e-4, v_hi: float = 5.0) -> float:
    """
    Solve for implied volatility from option price using Brent (if available) or robust bisection.
    Returns vol as a fraction (e.g., 0.24 = 24%).
    """
    target = float(price)

    def f(sig: float) -> float:
        return bs_price(S, K, T, r, q, sig, right) - target

    # Ensure bracket contains root
    lo, hi = v_lo, v_hi
    f_lo, f_hi = f(lo), f(hi)
    # Expand if both on same side (try limited times)
    attempts = 0
    while f_lo * f_hi > 0 and attempts < 10:
        lo *= 0.5
        hi *= 2.0
        f_lo, f_hi = f(lo), f(hi)
        attempts += 1
    if f_lo * f_hi > 0:
        # No root in bracket; best-effort: return NaN
        raise ValueError("IV solve failed: no bracket")

    if _HAS_SCIPY:
        return float(brentq(f, lo, hi, xtol=tol, maxiter=max_iter)) # type: ignore

    # Bisection fallback
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


# ======================= small helpers =======================

def _yf_spot(underlying: str) -> float:
    tk = yf.Ticker(underlying.upper().strip())
    fi = getattr(tk, "fast_info", {}) or {}
    last = _f(fi.get("last_price"))
    if last is not None:
        return last
    try:
        h = tk.history(period="1d").tail(1)
        if len(h) > 0:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    raise RuntimeError(f"Cannot determine spot for {underlying}")

def _year_fraction_to_expiry(expiry_yyyy_mm_dd: str) -> float:
    import datetime as dt
    # Use ACT/365f
    today = dt.datetime.utcnow().date()
    y, m, d = [int(x) for x in expiry_yyyy_mm_dd.split("-")]
    exp = dt.date(y, m, d)
    days = (exp - today).days
    return max(0.0, days / 365.0)

def _mid(bid: Any, ask: Any, last: Any) -> Optional[float]:
    b = _f(bid); a = _f(ask); l = _f(last)
    if b is not None and a is not None and b > 0 and a > 0:
        return 0.5 * (b + a)
    return l

def _f(x) -> Optional[float]:
    try:
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
    except Exception:
        return None


# ======================= script demo =======================

if __name__ == "__main__":
    cfg = OptConfig()
    oa = OptionsAdapter(cfg)
    sym = "AAPL"
    exps = oa.list_expiries(sym)
    chain = oa.fetch_chain(sym, exps[0])
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in chain["payload"].items()}, indent=2))
    # One leg
    k0 = float(chain["payload"]["calls"][len(chain["payload"]["calls"])//3]["strike"])
    q = oa.fetch_quote(QuoteKey(sym, exps[0], k0, "C"))
    print(json.dumps(q["payload"]["greeks"], indent=2))
    surf = oa.surface_snapshot(sym, exps[0])
    print(json.dumps({"rows": len(surf["payload"]["rows"])}, indent=2))