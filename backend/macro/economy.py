# backend/macro/economy.py
from __future__ import annotations

import csv
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# -------------------- Optional deps kept soft --------------------
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

try:
    from backend.utils.secrets import secrets # type: ignore
except Exception:
    class _S:
        def get(self, k, default=None, required=False):
            v = default
            return v
    secrets = _S()  # type: ignore


# =====================================================================
# Core models
# =====================================================================

@dataclass
class EconPoint:
    ts: datetime
    value: float
    revision: Optional[int] = None  # 0=flash/advance,1=2nd,2=final, etc.
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EconSeries:
    """
    Container for a single economic series (e.g., CPI YoY, GDP QoQ SAAR).
    Frequencies: 'D','W','M','Q','Y'
    """
    code: str
    name: str
    freq: str = "M"
    unit: str = ""
    region: str = ""     # e.g. "US", "IN", "EU"
    points: List[EconPoint] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, *pts: EconPoint) -> None:
        self.points.extend(pts)
        self.points.sort(key=lambda p: (p.ts, p.revision or 0))

    def last(self) -> Optional[EconPoint]:
        return self.points[-1] if self.points else None

    def to_pairs(self) -> List[Tuple[datetime, float]]:
        return [(p.ts, float(p.value)) for p in self.points]

    def to_pandas(self):
        if pd is None:
            raise RuntimeError("pandas not installed")
        if not self.points:
            return pd.Series(dtype=float)
        idx = [p.ts for p in self.points]
        vals = [p.value for p in self.points]
        return pd.Series(vals, index=pd.to_datetime(idx), name=self.code).sort_index()

    # --- transforms (pure python fallbacks if pandas missing) ---
    def pct_change(self, periods: int = 1) -> "EconSeries":
        if len(self.points) < (periods + 1):
            return EconSeries(self.code + f"_pct{periods}", self.name + " %Δ", self.freq, self.unit, self.region, [])
        out: List[EconPoint] = []
        # naive: assume points are ordered end-to-end at native frequency
        for i in range(periods, len(self.points)):
            a = self.points[i - periods].value
            b = self.points[i].value
            if a == 0:
                continue
            out.append(EconPoint(ts=self.points[i].ts, value=(b - a) / a))
        return EconSeries(self.code + f"_pct{periods}", self.name + " %Δ", self.freq, self.unit, self.region, out)

    def yoy(self) -> "EconSeries":
        # year-over-year: 12 for monthly, 4 for quarterly
        k = 12 if self.freq.upper() == "M" else 4 if self.freq.upper() == "Q" else 1
        s = self.pct_change(k)
        s.code = self.code + "_YoY"
        s.name = self.name + " YoY"
        return s

    def qoq_annualized(self) -> "EconSeries":
        # quarter over quarter, annualized (SAAR): ((1+qoq)^4 - 1)
        if self.freq.upper() != "Q":
            raise ValueError("qoq_annualized requires quarterly frequency")
        qoq = self.pct_change(1)
        out = []
        for p in qoq.points:
            out.append(EconPoint(ts=p.ts, value=(1 + p.value) ** 4 - 1))
        return EconSeries(self.code + "_QoQ_SAAR", self.name + " QoQ SAAR", "Q", self.unit, self.region, out)

    def zscore(self, lookback: int = 36) -> "EconSeries":
        pts = self.points[-lookback:]
        if len(pts) < 3:
            return EconSeries(self.code + "_Z", self.name + " z-score", self.freq, self.unit, self.region, [])
        vals = [p.value for p in pts]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(1, (len(vals) - 1))
        sd = math.sqrt(var)
        out = []
        for p in self.points:
            z = 0.0 if sd == 0 else (p.value - mu) / sd
            out.append(EconPoint(ts=p.ts, value=z))
        return EconSeries(self.code + "_Z", self.name + " z-score", self.freq, self.unit, self.region, out)

    def align_to_month_end(self) -> "EconSeries":
        """Map timestamps to last day of month (helps joins)."""
        def _month_end(dt: datetime) -> datetime:
            y, m = dt.year, dt.month
            if m == 12:
                nxt = datetime(y + 1, 1, 1)
            else:
                nxt = datetime(y, m + 1, 1)
            return (nxt - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        out = [EconPoint(ts=_month_end(p.ts), value=p.value, revision=p.revision, meta=p.meta) for p in self.points]
        return EconSeries(self.code, self.name, self.freq, self.unit, self.region, out)


# =====================================================================
# Surprise & calendar
# =====================================================================

@dataclass
class EconEvent:
    """
    Calendar event (release).
    consensus and previous are optional; surprise = actual - consensus.
    """
    code: str           # e.g., "US_CPI_HEADLINE"
    name: str
    ts: datetime        # release timestamp (UTC)
    actual: Optional[float] = None
    consensus: Optional[float] = None
    previous: Optional[float] = None
    region: str = ""
    unit: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def surprise(self) -> Optional[float]:
        if self.actual is None or self.consensus is None:
            return None
        return float(self.actual - self.consensus)

    def standardized_surprise(self, std_dev: Optional[float]) -> Optional[float]:
        s = self.surprise()
        if s is None or not std_dev or std_dev <= 0:
            return None
        return s / std_dev

@dataclass
class EconCalendar:
    events: List[EconEvent] = field(default_factory=list)

    def add(self, *evs: EconEvent) -> None:
        self.events.extend(evs)
        self.events.sort(key=lambda e: e.ts)

    def filter(self, *, code: Optional[str] = None, region: Optional[str] = None, since: Optional[datetime] = None, until: Optional[datetime] = None) -> List[EconEvent]:
        out = []
        for e in self.events:
            if code and e.code != code: continue
            if region and e.region != region: continue
            if since and e.ts < since: continue
            if until and e.ts > until: continue
            out.append(e)
        return out

    def to_json(self) -> str:
        return json.dumps([asdict(e) | {"ts": e.ts.isoformat()} for e in self.events], indent=2)

    @staticmethod
    def from_json(s: str) -> "EconCalendar":
        arr = json.loads(s)
        cal = EconCalendar()
        for o in arr:
            o = dict(o)
            o["ts"] = datetime.fromisoformat(o["ts"].replace("Z", "+00:00"))
            cal.add(EconEvent(**o))
        return cal


# =====================================================================
# Storage (in-memory + optional SQLite)
# =====================================================================

class EconStore:
    """
    Tiny data store.
    - In-memory dict of series
    - Optional SQLite persistence (ts,value,revision) per `code`
    """

    def __init__(self, *, db_path: Optional[str] = None):
        self._series: Dict[str, EconSeries] = {}
        self._db_path = db_path
        if db_path:
            self._init_db()

    def _init_db(self) -> None:
        con = sqlite3.connect(self._db_path) # type: ignore
        cur = con.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS econ_series (
            code TEXT NOT NULL,
            ts TEXT NOT NULL,
            value REAL NOT NULL,
            revision INTEGER,
            meta_json TEXT,
            PRIMARY KEY(code, ts, revision)
        )""")
        cur.execute("""CREATE INDEX IF NOT EXISTS idx_econ_series_code ON econ_series(code)""")
        con.commit()
        con.close()

    def put(self, s: EconSeries, *, persist: bool = True) -> None:
        self._series[s.code] = s
        if self._db_path and persist and s.points:
            con = sqlite3.connect(self._db_path)
            cur = con.cursor()
            for p in s.points:
                cur.execute(
                    "INSERT OR REPLACE INTO econ_series(code,ts,value,revision,meta_json) VALUES(?,?,?,?,?)",
                    (s.code, p.ts.isoformat(), float(p.value), p.revision if p.revision is not None else 0, json.dumps(p.meta or {}))
                )
            con.commit()
            con.close()

    def get(self, code: str) -> Optional[EconSeries]:
        if code in self._series:
            return self._series[code]
        if not self._db_path:
            return None
        # lazy-load from DB
        con = sqlite3.connect(self._db_path)
        cur = con.cursor()
        cur.execute("SELECT ts,value,revision,meta_json FROM econ_series WHERE code=? ORDER BY ts ASC, revision ASC", (code,))
        rows = cur.fetchall()
        con.close()
        if not rows:
            return None
        pts = [EconPoint(ts=datetime.fromisoformat(r[0]), value=float(r[1]), revision=int(r[2] or 0), meta=json.loads(r[3] or "{}")) for r in rows]
        s = EconSeries(code=code, name=code, points=pts)
        self._series[code] = s
        return s

    def list_codes(self) -> List[str]:
        out = list(self._series.keys())
        if self._db_path:
            con = sqlite3.connect(self._db_path)
            cur = con.cursor()
            cur.execute("SELECT DISTINCT code FROM econ_series")
            out += [r[0] for r in cur.fetchall() if r and r[0] not in out]
            con.close()
        return sorted(out)


# =====================================================================
# Source adapters (HTTP optional; kept soft)
# =====================================================================

class SourceBase:
    """Abstract-ish source. Return EconSeries."""
    def fetch_series(self, code: str, **kwargs) -> EconSeries:  # pragma: no cover
        raise NotImplementedError

class FREDSource(SourceBase):
    """
    Fetch from FRED (US). Requires an API key (FRED_API_KEY).
    code examples: "CPIAUCSL" (CPI), "GDPC1" (Real GDP)
    """
    BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or secrets.get("FRED_API_KEY", None)

    def fetch_series(self, code: str, **kwargs) -> EconSeries:
        if requests is None:
            raise RuntimeError("requests not installed")
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY missing")
        params = {
            "series_id": code,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": kwargs.get("start", "2000-01-01"),
        }
        r = requests.get(self.BASE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        pts: List[EconPoint] = []
        for obs in data.get("observations", []):
            val = obs.get("value")
            if val in (".", None):
                continue
            ts = datetime.fromisoformat(obs["date"])
            pts.append(EconPoint(ts=ts, value=float(val)))
        return EconSeries(code=code, name=code, points=pts)

class MOSPISource(SourceBase):
    """
    Placeholder for India MOSPI (CPI/IIP/GDP releases).
    This is a skeleton: you can wire their CSV/portal dumps or an internal mirror.
    """
    def fetch_series(self, code: str, **kwargs) -> EconSeries:
        raise NotImplementedError("Wire MOSPI fetch here")

class RBISource(SourceBase):
    """
    Placeholder for RBI time series (money supply, policy rate, FX reserves).
    If you maintain CSV dumps, point to them via path=...
    """
    def fetch_series(self, code: str, **kwargs) -> EconSeries:
        # Example: load from a CSV with columns: date,value
        path = kwargs.get("path")
        if not path:
            raise RuntimeError("RBISource requires path=<csv>")
        pts: List[EconPoint] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                ts = datetime.fromisoformat(row["date"])
                pts.append(EconPoint(ts=ts, value=float(row["value"])))
        return EconSeries(code=code, name=code, region="IN", points=pts)


# =====================================================================
# Lightweight nowcasting / factor utilities
# =====================================================================

class Nowcaster:
    """
    Simple linear nowcast/regression:
      target_t ~ β0 + Σ β_i * factor_i_t
    Uses numpy if present; otherwise uses a naive OLS in pure Python.
    """

    def __init__(self, target_code: str):
        self.target_code = target_code
        self.beta: Dict[str, float] = {}
        self.bias: float = 0.0
        self.factors: List[str] = []

    @staticmethod
    def _align(series_map: Dict[str, EconSeries]) -> List[Tuple[datetime, Dict[str, float]]]:
        # align by timestamp intersection
        keys = list(series_map.keys())
        stamp_sets = [set(p.ts for p in series_map[k].points) for k in keys]
        inter = set.intersection(*stamp_sets) if stamp_sets else set()
        rows: List[Tuple[datetime, Dict[str, float]]] = []
        for ts in sorted(inter):
            row = {k: float(next(p.value for p in series_map[k].points if p.ts == ts)) for k in keys}
            rows.append((ts, row))
        return rows

    def fit(self, target: EconSeries, factors: Dict[str, EconSeries]) -> None:
        aligned = self._align({target.code: target} | factors)
        if len(aligned) < 5:
            # not enough overlap; fall back to zero beta
            self.beta = {k: 0.0 for k in factors.keys()}
            self.bias = target.last().value if target.last() else 0.0 # type: ignore
            self.factors = list(factors.keys())
            return

        # build matrices
        y = []
        X = []
        for _, row in aligned:
            y.append(row[target.code])
            X.append([row[k] for k in factors.keys()])

        if np is not None:
            Xmat = np.column_stack([np.ones(len(X)), np.array(X, dtype=float)])
            yvec = np.array(y, dtype=float)
            # OLS: (X'X)^-1 X'y
            beta_all, *_ = np.linalg.lstsq(Xmat, yvec, rcond=None)
            self.bias = float(beta_all[0])
            self.beta = {k: float(b) for k, b in zip(factors.keys(), beta_all[1:])}
        else:
            # Pure python normal equations with small ridge
            m = len(X); k = len(X[0])
            lam = 1e-6
            # build augmented with intercept
            XA = [[1.0] + xi for xi in X]
            # compute (X'X + lam I) and X'y
            XtX = [[0.0]*(k+1) for _ in range(k+1)]
            Xty = [0.0]*(k+1)
            for i in range(m):
                for a in range(k+1):
                    Xty[a] += XA[i][a] * y[i]
                    for b in range(k+1):
                        XtX[a][b] += XA[i][a] * XA[i][b]
            for d in range(k+1):
                XtX[d][d] += lam
            # solve via Gauss-Jordan
            beta_all = _solve_linear(XtX, Xty)
            self.bias = beta_all[0]
            self.beta = {k: beta_all[i+1] for i, k in enumerate(factors.keys())}
        self.factors = list(factors.keys())

    def predict(self, latest_factors: Dict[str, float]) -> float:
        x = self.bias
        for k, b in self.beta.items():
            x += b * float(latest_factors.get(k, 0.0))
        return float(x)


def _solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """Gauss-Jordan elimination (tiny helper for OLS without numpy)."""
    n = len(b)
    # augment
    for i in range(n):
        A[i].append(b[i])
    # elimination
    for i in range(n):
        # pivot
        piv = A[i][i] if A[i][i] != 0 else 1e-12
        for j in range(i, n+1):
            A[i][j] /= piv
        # eliminate others
        for r in range(n):
            if r == i: continue
            fac = A[r][i]
            for c in range(i, n+1):
                A[r][c] -= fac * A[i][c]
    return [A[i][-1] for i in range(n)]


# =====================================================================
# Convenience helpers
# =====================================================================

def correlate(series_a: EconSeries, series_b: EconSeries) -> Optional[float]:
    """
    Pearson correlation using overlapping timestamps.
    """
    A = {p.ts: p.value for p in series_a.points}
    B = {p.ts: p.value for p in series_b.points}
    keys = sorted(set(A.keys()) & set(B.keys()))
    if len(keys) < 3:
        return None
    xs = [A[k] for k in keys]
    ys = [B[k] for k in keys]
    mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / (len(xs)-1)
    vx = sum((x-mx)**2 for x in xs) / (len(xs)-1)
    vy = sum((y-my)**2 for y in ys) / (len(ys)-1)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)

def diffusion_index(series_list: List[EconSeries], *, last_n: int = 1) -> float:
    """
    Simple breadth metric: share of series with positive change over last_n periods.
    """
    ups = 0; tot = 0
    for s in series_list:
        if len(s.points) > last_n:
            a = s.points[-last_n-1].value
            b = s.points[-1].value
            if b > a:
                ups += 1
            tot += 1
    return (ups / tot) if tot else 0.0

def rolling_std(series: EconSeries, window: int = 12) -> Optional[float]:
    vals = [p.value for p in series.points[-window:]]
    if len(vals) < 2:
        return None
    mu = sum(vals)/len(vals)
    var = sum((v-mu)**2 for v in vals) / (len(vals)-1)
    return math.sqrt(var)


# =====================================================================
# Tiny example / smoke test
# =====================================================================

if __name__ == "__main__":
    # Build a fake CPI index (monthly)
    idx = EconSeries(code="US_CPI", name="US CPI (Index)", freq="M", unit="idx", region="US")
    base = 260.0
    start = datetime(2023, 1, 31)
    for i in range(0, 24):
        ts = (start + timedelta(days=31*i)).replace(day=28)  # rough month end
        base *= (1 + 0.0025 + (0.001 if i % 6 == 0 else 0.0))  # slow trend
        idx.add(EconPoint(ts=ts, value=base))

    yoy = idx.yoy()
    z = yoy.zscore(lookback=12)

    print("Last CPI:", round(idx.last().value, 3)) # type: ignore
    print("YoY last:", round(yoy.last().value * 100, 2), "%") # type: ignore
    print("YoY z-score last:", round(z.last().value, 2)) # type: ignore

    # Nowcasting US_GDP from PMI + Payrolls (toy values)
    gdp = EconSeries("US_GDP_Q", "US GDP QoQ", "Q")
    pmi = EconSeries("US_PMI", "US PMI", "M")
    pay = EconSeries("US_NFP", "US Payrolls", "M")

    # fabricate aligned quarterly stamps for demo
    qbase = datetime(2024, 3, 31)
    for k in range(6):
        gdp.add(EconPoint(ts=qbase + timedelta(days=90*k), value=0.01 + 0.003*k))
    mbase = datetime(2024, 1, 31)
    for k in range(9):
        pmi.add(EconPoint(ts=mbase + timedelta(days=31*k), value=50 + k*0.3))
        pay.add(EconPoint(ts=mbase + timedelta(days=31*k), value=200 + k*10))

    # align by constructing quarterly PMI/NFP via last-in-quarter for toy fit
    # (in practice you’d build a proper alignment or use pandas resample)
    pmiQ = EconSeries("US_PMI_Q", "US PMI (Q)", "Q", points=[EconPoint(ts=gdp.points[i].ts, value=pmi.points[i*1+1].value) for i in range(6)])
    payQ = EconSeries("US_NFP_Q", "US NFP (Q)", "Q", points=[EconPoint(ts=gdp.points[i].ts, value=pay.points[i*1+1].value) for i in range(6)])

    nc = Nowcaster(target_code="US_GDP_Q")
    nc.fit(gdp, {"US_PMI_Q": pmiQ, "US_NFP_Q": payQ})
    pred = nc.predict({"US_PMI_Q": 55.0, "US_NFP_Q": 260.0})
    print("Nowcast GDP:", round(pred, 4))