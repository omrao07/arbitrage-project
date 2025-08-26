# backend/analytics/vol_surface.py
"""
Volatility Surface Builder
--------------------------
Builds an IV surface from option-chain snapshots (Yahoo/NSE/custom) and provides
smooth interpolation (SABR or splines when available; linear fallback otherwise).

Typical use:
    from backend.data.option_chain import get_chain
    from backend.analytics.vol_surface import SurfaceBuilder

    snap = get_chain("AAPL", provider="yahoo", expiry="ALL")
    surf = SurfaceBuilder().from_snapshot(snap, method="auto", beta=1.0)  # lognormal SABR if possible
    iv = surf.iv(T=0.25, K=200.0)
    grid = surf.grid()   # dense surface grid for plotting/dashboards

CLI:
    python -m backend.analytics.vol_surface --symbol AAPL --provider yahoo --publish --save
    python -m backend.analytics.vol_surface --from-json chain.json --json --export surface.csv

Notes:
- Requires only stdlib, but will auto-use numpy/scipy if present for better smoothing.
- Calibrates *per expiry* smile in moneyness; then time-interpolates across expiries.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Optional numeric stack
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import scipy.optimize as spo  # type: ignore
    import scipy.interpolate as spi  # type: ignore
except Exception:
    spo = spi = None  # type: ignore

# Option data model (shared with option_chain)
try:
    from backend.data.option_chain import ChainSnapshot, OptionQuote, _tenor_years_from_exp # type: ignore
except Exception:
    ChainSnapshot = Any  # type: ignore
    OptionQuote = Any  # type: ignore
    def _tenor_years_from_exp(exp_iso: str) -> float:
        try:
            tm = time.mktime(time.strptime(exp_iso, "%Y-%m-%d"))
            return max(0.0, (tm - time.time()) / (365.0*24*3600))
        except Exception:
            return 0.0

# Optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore


# --------------------- SABR (Hagan) helpers ---------------------

def _sabr_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Hagan 2002 lognormal SABR implied vol approximation (beta∈[0,1]).
    """
    if F <= 0 or K <= 0 or T <= 0:
        return float("nan")
    if F == K:
        num = alpha
        den = (F ** (1 - beta))
        zeta = (nu/alpha) * (F ** (1 - beta)) * math.log(F/K if K>0 else 1.0)
        xz = 1.0  # at-the-money limit; avoid 0/0
        term1 = ((1 - beta) ** 2 / 24) * (alpha ** 2) / (F ** (2 - 2*beta))
        term2 = (rho * beta * nu * alpha) / (4 * (F ** (1 - beta)))
        term3 = (2 - 3 * rho * rho) * (nu ** 2) / 24
        return (num/den) * (1 + (term1 + term2 + term3) * T)
    lnFK = math.log(F / K)
    FK_beta = (F * K) ** ((1 - beta) / 2.0)
    z = (nu / alpha) * FK_beta * lnFK
    xz = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    denom = FK_beta * (1 + ((1 - beta) ** 2 / 24) * (lnFK ** 2) + ((1 - beta) ** 4 / 1920) * (lnFK ** 4))
    if abs(xz) < 1e-12:
        return float("nan")
    A = alpha / denom
    B = 1 + (((1 - beta) ** 2 / 24) * (alpha ** 2 / (FK_beta ** 2))
             + (rho * beta * nu * alpha) / (4 * FK_beta)
             + ((2 - 3 * rho ** 2) * (nu ** 2) / 24)) * T
    return A * (z / xz) * B


# --------------------- storage (SQLite) ---------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS vol_surface (
  ts_ms INTEGER,
  underlying TEXT,
  provider TEXT,
  r REAL,
  q REAL,
  expiries TEXT,          -- JSON: list of expiries ISO
  F REAL,                 -- spot
  knot_T JSON,            -- list of maturities in years
  # values below are JSON per expiry
  smiles JSON,            -- {expiry_iso: [[m, iv], ...]}
  meta JSON               -- calibration diagnostics per expiry
);
"""

class SurfaceCache:
    def __init__(self, db_path: str = "runtime/options.db"):
        self.db_path = db_path
        d = os.path.dirname(db_path) or "."
        os.makedirs(d, exist_ok=True)
        with self._cx() as cx:
            # Keep in same db as option_chain for convenience
            try:
                cx.executescript(_SCHEMA)
            except sqlite3.OperationalError:
                # Fallback: create a simple key-value table if comments break old SQLite
                cx.execute("""CREATE TABLE IF NOT EXISTS vol_surface (
                    ts_ms INTEGER, underlying TEXT, provider TEXT, r REAL, q REAL,
                    expiries TEXT, F REAL, knot_T TEXT, smiles TEXT, meta TEXT
                )""")

    def _cx(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self.db_path, timeout=30.0)
        cx.row_factory = sqlite3.Row
        return cx

    def write(self, payload: Dict[str, Any]) -> None:
        with self._cx() as cx:
            cx.execute("INSERT INTO vol_surface(ts_ms, underlying, provider, r, q, expiries, F, knot_T, smiles, meta) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (payload["ts_ms"], payload["underlying"], payload["provider"], payload.get("r",0.0), payload.get("q",0.0),
                        json.dumps(payload["expiries"]), float(payload.get("F") or 0.0),
                        json.dumps(payload["knot_T"]), json.dumps(payload["smiles"]), json.dumps(payload.get("meta", {}))))
            cx.commit()

    def latest(self, underlying: str, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
        q = "SELECT * FROM vol_surface WHERE underlying=?"
        args = [underlying]
        if provider:
            q += " AND provider=?"
            args.append(provider)
        q += " ORDER BY ts_ms DESC LIMIT 1"
        with self._cx() as cx:
            row = cx.execute(q, args).fetchone()
        if not row:
            return None
        return {**dict(row),
                "expiries": json.loads(row["expiries"]),
                "knot_T": json.loads(row["knot_T"]),
                "smiles": json.loads(row["smiles"]),
                "meta": json.loads(row["meta"]) if row["meta"] else {}}


# --------------------- surface model ---------------------

@dataclass
class VolSurface:
    underlying: str
    F: float
    expiries: List[str]                   # ISO dates
    knot_T: List[float]                   # years to each expiry
    smiles: Dict[str, List[Tuple[float, float]]]  # expiry -> [(moneyness m=K/F, iv)]
    r: float = 0.0
    q: float = 0.0
    ts_ms: int = 0
    provider: str = "custom"
    meta: Dict[str, Any] = None # type: ignore

    # ---- query ----
    def iv_m(self, T: float, m: float) -> float:
        """
        Interpolate IV at maturity T (years) and moneyness m=K/F.
        """
        T = max(1e-8, float(T))
        # 1) interpolate within each expiry smile to get IV at m -> IV(Ti, m)
        Ti, ivi = [], []
        for exp in self.expiries:
            Ti.append(_tenor_years_from_exp(exp))
            ivi.append(_interp_smile(self.smiles[exp], m))
        # 2) interpolate across maturities
        return _interp_1d(Ti, ivi, T)

    def iv(self, T: float, K: float) -> float:
        m = max(1e-6, float(K)/max(1e-9, float(self.F)))
        return self.iv_m(T, m)

    def grid(self, *, T_points: int = 15, m_min: float = 0.7, m_max: float = 1.3, m_points: int = 31) -> Dict[str, Any]:
        Ts = _linspace(min(self.knot_T) if self.knot_T else 0.05,
                       max(self.knot_T) if self.knot_T else 1.0, T_points)
        ms = _linspace(m_min, m_max, m_points)
        Z = [[self.iv_m(T, m) for m in ms] for T in Ts]
        return {"F": self.F, "T": Ts, "m": ms, "iv": Z}

    def to_payload(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms or int(time.time()*1000),
            "underlying": self.underlying,
            "provider": self.provider,
            "F": self.F,
            "r": self.r, "q": self.q,
            "expiries": self.expiries,
            "knot_T": self.knot_T,
            "smiles": self.smiles,
            "meta": self.meta or {}
        }


# --------------------- builder ---------------------

class SurfaceBuilder:
    def __init__(self, *, min_points_per_expiry: int = 6, use_put_call_merge: bool = True):
        self.min_pts = int(min_points_per_expiry)
        self.merge = bool(use_put_call_merge)

    def from_snapshot(self, snap: ChainSnapshot, *, method: str = "auto", beta: float = 1.0) -> VolSurface: # type: ignore
        """
        method: 'auto' -> sabr if SciPy; else 'spline' if SciPy; else 'linear'
                'sabr' | 'spline' | 'linear'
        beta: SABR beta (0=normal, 1=lognormal)
        """
        F = float(snap.spot or 0.0)
        assert F > 0, "Snapshot needs a positive spot/underlying price"
        group: Dict[str, List[Tuple[float, float]]] = {}
        for r in snap.rows:
            if not (r.iv and r.strike):
                continue
            # merge calls/puts by moneyness; pick mid IV if duplicates
            m = float(r.strike) / F
            group.setdefault(r.expiry, []).append((m, float(r.iv)))

        # Clean per expiry & fit smile
        expiries = sorted(group.keys(), key=lambda e: _tenor_years_from_exp(e))
        smiles: Dict[str, List[Tuple[float, float]]] = {}
        meta: Dict[str, Any] = {}

        for exp in expiries:
            pts = _aggregate_by_bin(group[exp], bins=50)  # collapse duplicates/noise
            if len(pts) < self.min_pts:
                continue
            if method == "auto":
                method_eff = "sabr" if spo is not None else ("spline" if spi is not None else "linear")
            else:
                method_eff = method

            if method_eff == "sabr" and spo is not None:
                T = max(1e-6, _tenor_years_from_exp(exp))
                fit, diag = _fit_sabr_smile(pts, T, beta=beta)
                ms = _linspace(min(p[0] for p in pts), max(p[0] for p in pts), 41)
                ivs = [fit(m) for m in ms]
                smiles[exp] = list(zip(ms, ivs))
                meta[exp] = diag
            elif method_eff == "spline" and spi is not None:
                ms, ys = zip(*sorted(pts))
                f = spi.CubicSpline(ms, ys, bc_type="natural")
                grid = _linspace(min(ms), max(ms), 41)
                smiles[exp] = [(m, float(max(1e-4, min(5.0, f(m))))) for m in grid] # type: ignore
                meta[exp] = {"method": "spline", "n": len(pts)}
            else:
                # Linear fallback
                ms, ys = zip(*sorted(pts))
                grid = _linspace(min(ms), max(ms), 41)
                ivs = [_interp_1d(ms, ys, m) for m in grid]
                smiles[exp] = list(zip(grid, ivs))
                meta[exp] = {"method": "linear", "n": len(pts)}

        knot_T = [_tenor_years_from_exp(e) for e in smiles.keys()]
        expiries_final = list(smiles.keys())
        assert expiries_final, "No valid expiries to build a surface (need IV data)."
        return VolSurface(
            underlying=snap.underlying, F=F,
            expiries=expiries_final, knot_T=knot_T,
            smiles=smiles, r=getattr(snap, "r", 0.0), q=getattr(snap, "q", 0.0),
            ts_ms=int(time.time()*1000), provider=snap.provider, meta=meta
        )


# --------------------- fitting & interpolation utils ---------------------

def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1: return [a]
    step = (b - a) / (n - 1)
    return [a + i*step for i in range(n)]

def _aggregate_by_bin(pts: List[Tuple[float, float]], bins: int = 50) -> List[Tuple[float, float]]:
    """
    Collapse duplicate/nearby moneyness points using simple binning average.
    """
    pts = [(float(m), float(iv)) for m, iv in pts if math.isfinite(m) and math.isfinite(iv) and iv > 0]
    if not pts: return []
    mmin, mmax = min(p[0] for p in pts), max(p[0] for p in pts)
    if mmax <= mmin: return pts
    width = (mmax - mmin) / bins
    buckets: Dict[int, List[float]] = {}
    centers: Dict[int, float] = {}
    for m, iv in pts:
        k = int((m - mmin) / max(width, 1e-12))
        k = min(bins-1, max(0, k))
        buckets.setdefault(k, []).append(iv)
        centers[k] = mmin + (k + 0.5) * width
    out = []
    for k, ar in sorted(buckets.items()):
        out.append((centers[k], sum(ar)/len(ar)))
    return out

def _interp_1d(xs: List[float] | Tuple[float, ...], ys: List[float] | Tuple[float, ...], x: float) -> float:
    # Pure-python linear interpolation + flat extrapolation
    X, Y = list(xs), list(ys)
    if x <= X[0]: return float(Y[0])
    if x >= X[-1]: return float(Y[-1])
    # binary search
    lo, hi = 0, len(X)-1
    while hi - lo > 1:
        mid = (lo + hi)//2
        if x < X[mid]: hi = mid
        else: lo = mid
    x0, x1 = X[lo], X[hi]
    y0, y1 = Y[lo], Y[hi]
    w = (x - x0)/(x1 - x0)
    return float((1-w)*y0 + w*y1)

def _interp_smile(smile: List[Tuple[float, float]], m: float) -> float:
    ms, ivs = zip(*sorted(smile))
    return max(1e-4, min(5.0, _interp_1d(ms, ivs, m)))

def _fit_sabr_smile(pts: List[Tuple[float, float]], T: float, *, beta: float = 1.0):
    """
    Calibrate SABR (alpha, rho, nu) for fixed beta on (m, iv) where m=K/F.
    Uses SciPy least squares if available; falls back to simple heuristic otherwise.
    """
    assert spo is not None, "SciPy is required for SABR fitting"
    # Rebuild K from m; assume F=1 in smile-space
    F = 1.0
    ms, ivs = zip(*sorted(pts))
    Ks = [F * m for m in ms]

    # initial guesses
    alpha0 = max(0.05, min(0.6, float(sum(ivs)/len(ivs))))
    rho0, nu0 = 0.0, 0.5

    def model(params):
        a, r, n = params
        return [max(1e-4, min(5.0, _sabr_vol(F, K, T, a, beta, r, n))) for K in Ks]

    def resid(params):
        y = model(params)
        return [(y[i] - ivs[i]) for i in range(len(ivs))]

    bounds = ([1e-4, -0.999, 1e-4], [3.0, 0.999, 5.0])
    res = spo.least_squares(resid, x0=[alpha0, rho0, nu0], bounds=bounds, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=5000)

    a, r, n = res.x.tolist()
    def f(m: float) -> float:
        K = max(1e-9, F * m)
        return max(1e-4, min(5.0, _sabr_vol(F, K, T, a, beta, r, n)))

    diag = {"method": "sabr", "beta": beta, "alpha": a, "rho": r, "nu": n, "cost": float(res.cost), "n": len(pts)}
    return f, diag


# --------------------- publishing ---------------------

def publish_surface(surface: VolSurface) -> None:
    if not publish_stream:
        return
    payload = surface.to_payload()
    publish_stream("derivs.vol_surface", payload)
    # quick insight: ATM term structure slope
    try:
        if len(surface.knot_T) >= 2:
            atms = [surface.iv_m(T, 1.0) for T in surface.knot_T]
            slope = (atms[-1] - atms[0]) / max(1e-6, (surface.knot_T[-1] - surface.knot_T[0]))
            publish_stream("ai.insight", {
                "ts_ms": payload["ts_ms"],
                "kind": "vol_surface",
                "summary": f"{surface.underlying}: ATM term slope {slope:+.2%}/yr",
                "details": [f"ATM first={atms[0]:.2%} last={atms[-1]:.2%}", f"knots={len(surface.knot_T)}"],
                "tags": ["options","vol","term-structure", surface.underlying]
            })
    except Exception:
        pass


# --------------------- CLI ---------------------

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

def _write_csv(surface: VolSurface, path: str) -> str:
    import csv
    _ensure_dir(path)
    grid = surface.grid()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # header
        w.writerow(["F", surface.F])
        w.writerow(["T_points"] + grid["T"])
        w.writerow(["m_points"] + grid["m"])
        w.writerow(["matrix_rows", len(grid["T"])])
        # rows: T, then ivs across m
        for i, T in enumerate(grid["T"]):
            w.writerow([T] + [grid["iv"][i][j] for j in range(len(grid["m"]))])
    return path

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Volatility surface builder")
    ap.add_argument("--symbol", type=str, help="Underlying symbol (e.g., AAPL or RELIANCE.NS)")
    ap.add_argument("--provider", type=str, default="yahoo", choices=["yahoo","nse","custom"])
    ap.add_argument("--from-json", type=str, help="Load a ChainSnapshot-like JSON")
    ap.add_argument("--method", type=str, default="auto", choices=["auto","sabr","spline","linear"])
    ap.add_argument("--beta", type=float, default=1.0, help="SABR beta (0..1)")
    ap.add_argument("--save", action="store_true", help="Save surface to runtime/options.db")
    ap.add_argument("--publish", action="store_true", help="Publish to bus derivs.vol_surface")
    ap.add_argument("--export", type=str, help="Export grid CSV")
    ap.add_argument("--json", action="store_true", help="Print surface JSON")
    args = ap.parse_args()

    # Load chain snapshot
    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # minimal normalization to ChainSnapshot-like dict
        snap = type("Snap", (), {})()
        snap.underlying = raw.get("underlying","UNKNOWN") # type: ignore
        snap.provider = raw.get("provider","custom") # type: ignore
        snap.spot = float(raw.get("spot") or 0.0) # type: ignore
        rows = raw.get("rows") or []
        class Row: pass
        snap.rows = [] # type: ignore
        for r in rows:
            o = Row()
            for k,v in r.items(): setattr(o, k, v)
            snap.rows.append(o) # type: ignore
    else:
        if not args.symbol:
            ap.error("--symbol required unless --from-json is used")
        # Lazy import to avoid hard dependency
        try:
            from backend.data.option_chain import get_chain # type: ignore
        except Exception as e:
            raise RuntimeError("backend.data.option_chain.get_chain not available") from e
        snap = get_chain(args.symbol, provider=args.provider, expiry="ALL")

    # Build surface
    surf = SurfaceBuilder().from_snapshot(snap, method=args.method, beta=args.beta)

    if args.save:
        SurfaceCache().write(surf.to_payload())

    if args.publish:
        publish_surface(surf)

    if args.export:
        p = _write_csv(surf, args.export)
        print(f"Wrote {p}")

    if args.json:
        print(json.dumps(surf.to_payload(), indent=2))

if __name__ == "__main__":
    main()