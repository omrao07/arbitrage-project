# backend/data/option_chain.py
"""
Option Chain Fetcher / Normalizer / Cache
-----------------------------------------
Providers (auto-graceful):
  - Yahoo Finance (via yfinance, optional)
  - NSE India (via nsepython or nsetools, optional)
  - Custom raw -> normalizer

Emits (if bus is available):
  - derivs.option_chain (one snapshot envelope per fetch)
  - ai.insight (compact note on extreme IV/skew)

Caches:
  - SQLite at runtime/options.db (schema: options, fetch_log)

CLI:
  python -m backend.data.option_chain --symbol AAPL --provider yahoo --save --publish
  python -m backend.data.option_chain --symbol RELIANCE.NS --provider nse --save --export chain.csv
  python -m backend.data.option_chain --from-json raw.json --save

Notes:
- This is a fetcher/normalizer; wire it into your dashboards (smart option chain, F&O panels).
- For NSE, install one of: `pip install nsepython` or `pip install nsetools`.
- For Yahoo, install: `pip install yfinance`.
"""

from __future__ import annotations

import math
import os
import json
import time
import sqlite3
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Optional providers
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

try:
    import nsepython as nsepy  # type: ignore
except Exception:
    nsepy = None  # type: ignore

try:
    from nsetools import Nse  # type: ignore
except Exception:
    Nse = None  # type: ignore

# Optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore


# ------------------------ math: BS greeks / IV ------------------------

def _norm_cdf(x: float) -> float:
    # Abramowitz & Stegun erf-based
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def _bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, right: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if right == "C" else (K - S))
    d1 = _bs_d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    if right == "C":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)

def _bs_vega(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = _bs_d1(S, K, r, q, sigma, T)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return S * math.exp(-q * T) * pdf * math.sqrt(T)

def implied_vol(
    S: float, K: float, r: float, q: float, T: float, right: str, px: float, *,
    guess: float = 0.3, tol: float = 1e-4, iters: int = 60
) -> Optional[float]:
    """Newton-Raphson IV; returns None if fails."""
    sigma = max(1e-4, float(guess))
    for _ in range(iters):
        model = _bs_price(S, K, r, q, sigma, T, right)
        diff = model - px
        if abs(diff) < tol:
            return max(1e-4, min(5.0, sigma))
        v = _bs_vega(S, K, r, q, sigma, T)
        if v <= 1e-10:
            break
        sigma -= diff / v
        sigma = max(1e-4, min(5.0, sigma))
    return None

def greeks(S: float, K: float, r: float, q: float, T: float, sigma: float, right: str) -> Dict[str, float]:
    if T <= 0 or sigma is None or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    d1 = _bs_d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    if right == "C":
        delta = math.exp(-q * T) * _norm_cdf(d1)
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2)
        theta = (
            -(S * math.exp(-q*T) * pdf * sigma) / (2*math.sqrt(T))
            - r * K * math.exp(-r*T) * _norm_cdf(d2)
            + q * S * math.exp(-q*T) * _norm_cdf(d1)
        )
    else:
        delta = -math.exp(-q * T) * _norm_cdf(-d1)
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2)
        theta = (
            -(S * math.exp(-q*T) * pdf * sigma) / (2*math.sqrt(T))
            + r * K * math.exp(-r*T) * _norm_cdf(-d2)
            - q * S * math.exp(-q*T) * _norm_cdf(-d1)
        )
    gamma = math.exp(-q * T) * pdf / (S * sigma * math.sqrt(T))
    vega = _bs_vega(S, K, r, q, sigma, T) / 100.0  # per 1 vol pt if desired
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


# ------------------------ data models ------------------------

@dataclass
class OptionQuote:
    underlying: str
    expiry: str           # ISO date (YYYY-MM-DD)
    strike: float
    right: str            # 'C' or 'P'
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    volume: Optional[float]
    open_interest: Optional[float]
    iv: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    rho: Optional[float]
    ts_ms: int
    provider: str
    mid: Optional[float] = None

@dataclass
class ChainSnapshot:
    underlying: str
    spot: Optional[float]
    r: float
    q: float
    ts_ms: int
    provider: str
    rows: List[OptionQuote]


# ------------------------ normalizer / helpers ------------------------

def _ms_now() -> int:
    return int(time.time() * 1000)

def _mid(b: Optional[float], a: Optional[float]) -> Optional[float]:
    if b is None or a is None or b <= 0 or a <= 0: return None
    if a < b: return (a + b) / 2.0
    return (b + a) / 2.0

def _days_to_years(days: float) -> float:
    return max(0.0, days) / 365.0

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

def _safe_float(x) -> Optional[float]:
    try:
        if x is None: return None
        v = float(x)
        return v if not (math.isnan(v) or math.isinf(v)) else None
    except Exception:
        return None


# ------------------------ providers ------------------------

def fetch_yahoo_chain(symbol: str, expiry: Optional[str] = None) -> ChainSnapshot:
    assert yf is not None, "yfinance not installed. `pip install yfinance`"
    t = yf.Ticker(symbol)
    info = t.info if hasattr(t, "info") else {}
    spot = _safe_float(info.get("regularMarketPrice") or info.get("lastPrice") or info.get("previousClose"))
    r = 0.0  # leave 0 unless you pipe a curve here
    q = 0.0
    ts = _ms_now()
    expiries = t.options or []
    rows: List[OptionQuote] = []
    targets = expiries if (expiry in (None, "ALL")) else [expiry] if expiry in expiries else []
    for exp in targets:
        oc = t.option_chain(exp)
        for side, df in (("C", oc.calls), ("P", oc.puts)):
            for _, row in df.iterrows():
                K = _safe_float(row.get("strike"))
                bid = _safe_float(row.get("bid"))
                ask = _safe_float(row.get("ask"))
                last = _safe_float(row.get("lastPrice"))
                oi = _safe_float(row.get("openInterest"))
                vol = _safe_float(row.get("volume"))
                iv = _safe_float(row.get("impliedVolatility"))
                mid = _mid(bid, ask)
                # time to expiry
                # yfinance gives exp as 'YYYY-MM-DD'
                T = max(0.0, (time.mktime(time.strptime(exp, "%Y-%m-%d")) - time.time()) / (365.0*24*3600))
                # compute IV if missing and we have spot + price
                ref_px = mid or last
                if iv is None and ref_px and spot and K:
                    iv = implied_vol(spot, K, r, q, T, side, ref_px) or None
                grek = greeks(spot or 0.0, K or 0.0, r, q, T, iv or 0.0, side) if (spot and K and iv) else {"delta":None,"gamma":None,"theta":None,"vega":None,"rho":None}
                rows.append(OptionQuote(
                    underlying=symbol, expiry=exp, strike=K or 0.0, right=side,
                    bid=bid, ask=ask, last=last, volume=vol, open_interest=oi, iv=iv,
                    delta=grek["delta"], gamma=grek["gamma"], theta=grek["theta"], vega=grek["vega"], rho=grek["rho"],
                    ts_ms=ts, provider="yahoo", mid=mid
                ))
    return ChainSnapshot(underlying=symbol, spot=spot, r=r, q=q, ts_ms=ts, provider="yahoo", rows=rows)

def fetch_nse_chain(symbol: str, expiry: Optional[str] = None) -> ChainSnapshot:
    """
    Tries nsepython first, then nsetools. EXPIRY string format depends on lib; we normalize to YYYY-MM-DD.
    """
    ts = _ms_now()
    r = 0.0; q = 0.0
    rows: List[OptionQuote] = []
    spot = None

    if nsepy is not None:
        # nsepython: nse_opt_chain_ltp, etc. API shapes can vary by version; best-effort parse
        try:
            raw = nsepy.nse_optionchain_scrapper(symbol)  # returns dict with 'records'/'filtered'
            spot = _safe_float(raw.get("records", {}).get("underlyingValue"))
            # Build a map expiry->rows
            all_data = raw.get("records", {}).get("data", [])
            for rec in all_data:
                exp_raw = rec.get("expiryDate") or rec.get("expirydate")
                # NSE often uses '25-Sep-2025' -> normalize:
                exp_norm = _nse_exp_to_iso(exp_raw)
                for side_key, side in (("CE","C"),("PE","P")):
                    leg = rec.get(side_key)
                    if not leg: continue
                    K = _safe_float(leg.get("strikePrice"))
                    bid = _safe_float(leg.get("bidprice") or leg.get("bidPrice"))
                    ask = _safe_float(leg.get("askPrice") or leg.get("askprice"))
                    last = _safe_float(leg.get("lastPrice") or leg.get("lastprice"))
                    oi = _safe_float(leg.get("openInterest"))
                    vol = _safe_float(leg.get("totalTradedVolume") or leg.get("totalTradedVolume"))
                    iv = _safe_float(leg.get("impliedVolatility"))
                    mid = _mid(bid, ask)
                    T = _tenor_years_from_exp(exp_norm)
                    ref_px = mid or last
                    if iv is None and ref_px and spot and K:
                        iv = implied_vol(spot, K, r, q, T, side, ref_px) or None
                    grek = greeks(spot or 0.0, K or 0.0, r, q, T, iv or 0.0, side) if (spot and K and iv) else {"delta":None,"gamma":None,"theta":None,"vega":None,"rho":None}
                    if expiry in (None, "ALL") or exp_norm == expiry:
                        rows.append(OptionQuote(
                            underlying=symbol, expiry=exp_norm, strike=K or 0.0, right=side,
                            bid=bid, ask=ask, last=last, volume=vol, open_interest=oi, iv=iv,
                            delta=grek["delta"], gamma=grek["gamma"], theta=grek["theta"], vega=grek["vega"], rho=grek["rho"],
                            ts_ms=ts, provider="nse", mid=mid
                        ))
            return ChainSnapshot(underlying=symbol, spot=spot, r=r, q=q, ts_ms=ts, provider="nse", rows=rows)
        except Exception:
            pass

    if Nse is not None:
        try:
            n = Nse()
            raw = n.get_option_chain(symbol)
            spot = _safe_float(raw.get("records", {}).get("underlyingValue"))
            data = raw.get("records", {}).get("data", [])
            for rec in data:
                exp_norm = _nse_exp_to_iso(rec.get("expiryDate") or rec.get("expirydate"))
                for side_key, side in (("CE","C"),("PE","P")):
                    leg = rec.get(side_key)
                    if not leg: continue
                    K = _safe_float(leg.get("strikePrice"))
                    bid = _safe_float(leg.get("bidprice") or leg.get("bidPrice"))
                    ask = _safe_float(leg.get("askPrice") or leg.get("askprice"))
                    last = _safe_float(leg.get("lastPrice") or leg.get("lastprice"))
                    oi = _safe_float(leg.get("openInterest"))
                    vol = _safe_float(leg.get("totalTradedVolume"))
                    iv = _safe_float(leg.get("impliedVolatility"))
                    mid = _mid(bid, ask)
                    T = _tenor_years_from_exp(exp_norm)
                    ref_px = mid or last
                    if iv is None and ref_px and spot and K:
                        iv = implied_vol(spot, K, 0.0, 0.0, T, side, ref_px) or None
                    grek = greeks(spot or 0.0, K or 0.0, 0.0, 0.0, T, iv or 0.0, side) if (spot and K and iv) else {"delta":None,"gamma":None,"theta":None,"vega":None,"rho":None}
                    if expiry in (None, "ALL") or exp_norm == expiry:
                        rows.append(OptionQuote(
                            underlying=symbol, expiry=exp_norm, strike=K or 0.0, right=side,
                            bid=bid, ask=ask, last=last, volume=vol, open_interest=oi, iv=iv,
                            delta=grek["delta"], gamma=grek["gamma"], theta=grek["theta"], vega=grek["vega"], rho=grek["rho"],
                            ts_ms=_ms_now(), provider="nse", mid=mid
                        ))
            return ChainSnapshot(underlying=symbol, spot=spot, r=0.0, q=0.0, ts_ms=_ms_now(), provider="nse", rows=rows)
        except Exception:
            pass

    raise RuntimeError("No NSE provider available. Install `nsepython` or `nsetools`.")

def _nse_exp_to_iso(s: Optional[str]) -> str:
    """
    Convert common NSE expiries like '26-Sep-2025' or '26-SEP-2025' to '2025-09-26'.
    """
    if not s: return ""
    s = str(s).strip()
    try:
        # handle '26-Sep-2025' or uppercase variants
        d, m, y = s.split("-")
        m_map = {"JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06",
                 "JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12"}
        mm = m_map[m.upper()]
        return f"{y}-{mm}-{int(d):02d}"
    except Exception:
        # last resort: return as-is if already ISO
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return s
        return s

def _tenor_years_from_exp(exp_iso: str) -> float:
    try:
        tm = time.mktime(time.strptime(exp_iso, "%Y-%m-%d"))
        return max(0.0, (tm - time.time()) / (365.0*24*3600))
    except Exception:
        return 0.0


# ------------------------ SQLite cache ------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS options (
  underlying TEXT, expiry TEXT, strike REAL, right TEXT,
  bid REAL, ask REAL, last REAL, volume REAL, open_interest REAL,
  iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL,
  mid REAL, ts_ms INTEGER, provider TEXT,
  PRIMARY KEY (underlying, expiry, strike, right, ts_ms, provider)
);
CREATE TABLE IF NOT EXISTS fetch_log (
  ts_ms INTEGER,
  underlying TEXT,
  provider TEXT,
  expiries TEXT,
  spot REAL
);
"""

class OptionCache:
    def __init__(self, db_path: str = "runtime/options.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        with self._cx() as cx:
            cx.executescript(_SCHEMA)

    def _cx(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self.db_path, timeout=30.0)
        cx.row_factory = sqlite3.Row
        return cx

    def write_snapshot(self, snap: ChainSnapshot) -> None:
        with self._cx() as cx:
            for r in snap.rows:
                cx.execute("""
                INSERT OR REPLACE INTO options
                (underlying, expiry, strike, right, bid, ask, last, volume, open_interest,
                 iv, delta, gamma, theta, vega, rho, mid, ts_ms, provider)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (r.underlying, r.expiry, float(r.strike), r.right,
                      _n(r.bid), _n(r.ask), _n(r.last), _n(r.volume), _n(r.open_interest),
                      _n(r.iv), _n(r.delta), _n(r.gamma), _n(r.theta), _n(r.vega), _n(r.rho),
                      _n(r.mid), r.ts_ms, snap.provider))
            cx.execute("INSERT INTO fetch_log(ts_ms, underlying, provider, expiries, spot) VALUES(?,?,?,?,?)",
                       (snap.ts_ms, snap.underlying, snap.provider, json.dumps(sorted(list({r.expiry for r in snap.rows}))), _n(snap.spot)))
            cx.commit()

    def latest_chain(self, underlying: str, *, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM options WHERE underlying=?"
        args = [underlying]
        if provider:
            q += " AND provider=?"
            args.append(provider)
        q += " ORDER BY ts_ms DESC, expiry, strike, right"
        with self._cx() as cx:
            rows = [dict(r) for r in cx.execute(q, args).fetchall()]
        return rows

def _n(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x)


# ------------------------ publishing & insights ------------------------

def publish_snapshot(snap: ChainSnapshot) -> None:
    if not publish_stream:
        return
    payload = {
        "ts_ms": snap.ts_ms,
        "underlying": snap.underlying,
        "provider": snap.provider,
        "spot": snap.spot,
        "rows": [asdict(r) for r in snap.rows]
    }
    publish_stream("derivs.option_chain", payload)
    # Simple skew/iv insight
    try:
        ivs_c = [r.iv for r in snap.rows if r.right == "C" and r.iv]
        ivs_p = [r.iv for r in snap.rows if r.right == "P" and r.iv]
        if (ivs_c or ivs_p) and publish_stream:
            ivc = sum(ivs_c)/len(ivs_c) if ivs_c else None
            ivp = sum(ivs_p)/len(ivs_p) if ivs_p else None
            if ivc and ivp and abs(ivc-ivp) > 0.05:
                publish_stream("ai.insight", {
                    "ts_ms": snap.ts_ms, "kind":"options",
                    "summary": f"{snap.underlying}: call/put IV skew {ivc-ivp:+.2%}",
                    "details": [f"avgC={ivc:.2%} avgP={ivp:.2%}", f"rows={len(snap.rows)}"],
                    "tags": ["options","skew", snap.underlying]
                })
    except Exception:
        pass


# ------------------------ public API ------------------------

def get_chain(
    symbol: str,
    *,
    provider: str = "yahoo",
    expiry: Optional[str] = None,
    r: float = 0.0,
    q: float = 0.0,
    compute_if_missing: bool = True,
) -> ChainSnapshot:
    """
    Fetch a normalized chain snapshot from a provider.
    `expiry`: 'YYYY-MM-DD' or 'ALL' or None (-> ALL).
    """
    provider = provider.lower()
    if provider == "yahoo":
        snap = fetch_yahoo_chain(symbol, expiry or "ALL")
    elif provider == "nse":
        snap = fetch_nse_chain(symbol, expiry or "ALL")
    else:
        raise ValueError("provider must be 'yahoo' or 'nse'")
    # If caller supplies r/q, override in greeks (optional recompute)
    if compute_if_missing and (r != 0.0 or q != 0.0):
        for r0 in snap.rows:
            T = _tenor_years_from_exp(r0.expiry)
            if r0.iv and snap.spot and r0.strike:
                g = greeks(snap.spot, r0.strike, r, q, T, r0.iv, r0.right)
                r0.delta, r0.gamma, r0.theta, r0.vega, r0.rho = g["delta"], g["gamma"], g["theta"], g["vega"], g["rho"]
        snap.r, snap.q = r, q
    return snap

def from_raw(
    underlying: str,
    rows: List[Dict[str, Any]],
    *,
    spot: Optional[float] = None,
    provider: str = "custom",
    r: float = 0.0,
    q: float = 0.0,
) -> ChainSnapshot:
    """
    Normalize a list of raw option dicts into ChainSnapshot. Expected keys per row:
      - expiry ('YYYY-MM-DD'), strike, right ('C'/'P'), bid?, ask?, last?, iv?, oi?, volume?
    If iv missing and we have spot + price, we compute IV/greeks.
    """
    ts = _ms_now()
    out_rows: List[OptionQuote] = []
    for m in rows:
        exp = str(m.get("expiry"))
        K   = _safe_float(m.get("strike"))
        side = str(m.get("right") or m.get("type") or m.get("optionType") or "C").upper()[0]
        bid = _safe_float(m.get("bid")); ask = _safe_float(m.get("ask"))
        last= _safe_float(m.get("last")); oi  = _safe_float(m.get("oi") or m.get("open_interest"))
        vol = _safe_float(m.get("volume"))
        iv  = _safe_float(m.get("iv"))
        mid = _mid(bid, ask)
        T = _tenor_years_from_exp(exp)
        ref_px = mid or last
        if iv is None and ref_px and spot and K:
            iv = implied_vol(spot, K, r, q, T, side, ref_px) or None
        g = greeks(spot or 0.0, K or 0.0, r, q, T, iv or 0.0, side) if (spot and K and iv) else {"delta":None,"gamma":None,"theta":None,"vega":None,"rho":None}
        out_rows.append(OptionQuote(
            underlying=underlying, expiry=exp, strike=K or 0.0, right=side,
            bid=bid, ask=ask, last=last, volume=vol, open_interest=oi, iv=iv,
            delta=g["delta"], gamma=g["gamma"], theta=g["theta"], vega=g["vega"], rho=g["rho"],
            ts_ms=ts, provider=provider, mid=mid
        ))
    return ChainSnapshot(underlying=underlying, spot=spot, r=r, q=q, ts_ms=ts, provider=provider, rows=out_rows)


# ------------------------ CLI ------------------------

def _write_csv(rows: List[Dict[str, Any]], path: str) -> str:
    import csv
    _ensure_dir(path)
    cols = ["underlying","expiry","strike","right","bid","ask","mid","last","volume","open_interest","iv","delta","gamma","theta","vega","rho","ts_ms","provider"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    return path

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Option chain fetcher/normalizer")
    ap.add_argument("--symbol", type=str, help="Underlying symbol (e.g., AAPL or RELIANCE.NS)")
    ap.add_argument("--provider", type=str, default="yahoo", choices=["yahoo","nse"])
    ap.add_argument("--expiry", type=str, default="ALL", help="YYYY-MM-DD | ALL")
    ap.add_argument("--save", action="store_true", help="Persist to SQLite runtime/options.db")
    ap.add_argument("--publish", action="store_true", help="Publish to bus derivs.option_chain")
    ap.add_argument("--export", type=str, help="Export CSV path")
    ap.add_argument("--json", action="store_true", help="Print JSON to stdout")
    ap.add_argument("--from-json", type=str, help="Normalize from a local raw JSON file")
    ap.add_argument("--spot", type=float, help="Spot override (for from-json or greeks recompute)")
    ap.add_argument("--r", type=float, default=0.0, help="Risk-free rate (annualized, e.g., 0.07)")
    ap.add_argument("--q", type=float, default=0.0, help="Dividend/borrow yield (annualized)")
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rows = raw if isinstance(raw, list) else raw.get("rows") or raw.get("data") or []
        sym = raw.get("underlying") or args.symbol or "UNKNOWN" # type: ignore
        snap = from_raw(sym, rows, spot=args.spot, provider="custom", r=args.r, q=args.q)
    else:
        if not args.symbol:
            ap.error("--symbol is required unless --from-json is used")
        snap = get_chain(args.symbol, provider=args.provider, expiry=args.expiry, r=args.r, q=args.q)

    if args.save:
        OptionCache().write_snapshot(snap)

    if args.publish:
        publish_snapshot(snap)

    rows_dicts = [asdict(r) for r in snap.rows]

    if args.export:
        path = _write_csv(rows_dicts, args.export)
        print(f"Wrote {path}")

    if args.json:
        print(json.dumps({
            "underlying": snap.underlying,
            "provider": snap.provider,
            "spot": snap.spot,
            "r": snap.r, "q": snap.q,
            "ts_ms": snap.ts_ms,
            "rows": rows_dicts
        }, indent=2))

if __name__ == "__main__":
    main()