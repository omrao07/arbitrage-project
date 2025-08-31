# backend/risk/volatility_cones.py
from __future__ import annotations
"""
Volatility Cones
----------------
Build historical "vol cones" from prices and optionally overlay implied vols.

Inputs
------
Prices CSV/DataFrame/list[dict]:
  columns: date, close [, symbol]
  - date: ISO date or timestamp
  - close: adjusted close (float)
  - symbol: optional; use --symbol to filter

Optional implied vols CSV/DataFrame/list[dict]:
  columns: date, tenor_days, iv
  - iv as decimal (e.g., 0.24 for 24%)

Outputs
-------
- cones table: per window (days) → p10, p25, p50, p75, p90, today_rv, today_pct
- rolling realized vol timeseries per window (optional return)
- implied overlay matched by nearest tenor (optional)
- optional Plotly HTML chart (cones + today's points)

CLI
---
python -m backend.risk.volatility_cones \
  --prices data/close.csv --symbol AAPL \
  --windows 5,10,20,60,120,252 \
  --implied data/iv.csv \
  --out_csv cones.csv --out_json cones.json \
  --plot cones.html

Env (optional):
  VOL_CONES_STREAM  (Redis stream name; only used if backend.bus.streams.publish_stream exists)
  TRADING_DAYS      (default 252)

Notes
-----
- Realized vol = sqrt(annualization) * stdev( log returns ) over each window.
- Percentile rank for today's RV is computed vs the historical distribution for that window.
"""

import csv, json, math, os, sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

TRADING_DAYS = float(os.getenv("TRADING_DAYS", "252"))

# Optional deps
try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None
try:
    import plotly.graph_objects as _go  # type: ignore
except Exception:
    _go = None

# Optional bus
try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass

VOL_STREAM = os.getenv("VOL_CONES_STREAM", "risk.vol_cones")

# ---------- Helpers ----------
def _parse_date(x: Any) -> datetime:
    if isinstance(x, (int, float)):
        # assume epoch seconds if < 1e12, else ms
        if x > 1e12: x = x / 1000.0
        return datetime.utcfromtimestamp(float(x))
    s = str(x).strip().replace("Z", "")
    # Try ISO first
    try:
        return datetime.fromisoformat(s[:19]) if "T" in s else datetime.strptime(s[:10], "%Y-%m-%d")
    except Exception:
        # last resort
        try:
            return datetime.utcfromtimestamp(float(s))
        except Exception:
            raise ValueError(f"Unparseable date: {x}")

def _as_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def _pct_rank(sample: List[float], x: float) -> float:
    """Percentile rank of x within sample (0..100)."""
    if not sample:
        return float("nan")
    if _np is not None:
        return float((_np.sum(_np.array(sample) <= x) / len(sample)) * 100.0)
    # pure python
    s = sorted(sample)
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi) // 2
        if s[mid] <= x: lo = mid + 1
        else: hi = mid
    return (lo / len(s)) * 100.0

def _nanpercentiles(vals: List[float], qs=(10,25,50,75,90)) -> Dict[str, float]:
    vv = [v for v in vals if v is not None and math.isfinite(v)]
    if not vv:
        return {f"p{q}": float("nan") for q in qs}
    if _np is not None:
        pr = _np.percentile(vv, qs).tolist()
    else:
        vv.sort()
        pr = []
        for q in qs:
            k = (q/100)*(len(vv)-1)
            f = int(math.floor(k)); c = min(int(math.ceil(k)), len(vv)-1)
            if f == c: pr.append(vv[f]); continue
            pr.append(vv[f] + (vv[c]-vv[f])*(k-f))
    return {f"p{q}": float(pr[i]) for i, q in enumerate(qs)}

# ---------- Core ----------
@dataclass
class ConeRow:
    window: int
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    today_rv: Optional[float]
    today_pct: Optional[float]
    implied_iv: Optional[float] = None
    implied_minus_p50: Optional[float] = None
    symbol: Optional[str] = None

class VolatilityCones:
    def __init__(self, *, trading_days: float = TRADING_DAYS):
        self.ann = float(trading_days)

    # -- public API --
    def build(
        self,
        prices: Union[str, List[Dict[str, Any]], Any],
        *,
        symbol: Optional[str] = None,
        windows: List[int] = (5,10,20,60,120,252), # type: ignore
        implied: Optional[Union[str, List[Dict[str, Any]], Any]] = None,
        match_implied: bool = True,
        return_timeseries: bool = False
    ) -> Tuple[List[ConeRow], Optional[Dict[int, List[Tuple[datetime, float]]]]]:
        """
        Returns (cone_rows, rv_timeseries_by_window?)
        """
        dates, closes = self._load_prices(prices, symbol=symbol)
        if len(closes) < max(windows) + 3:
            raise ValueError("Not enough price history for the largest window.")

        # log returns
        rets = []
        for i in range(1, len(closes)):
            if closes[i-1] and closes[i] and closes[i-1] > 0 and closes[i] > 0:
                rets.append(math.log(closes[i]/closes[i-1]))
            else:
                rets.append(float("nan"))
        r_dates = dates[1:]

        # rolling realized vol per window
        rv_ts: Dict[int, List[Tuple[datetime, float]]] = {}
        for w in windows:
            series: List[Tuple[datetime, float]] = []
            window_vals: List[float] = []
            for i in range(len(rets)):
                window_vals.append(rets[i])
                if len(window_vals) > w:
                    window_vals.pop(0)
                if len(window_vals) == w and all(math.isfinite(v) for v in window_vals):
                    mu = sum(window_vals)/w
                    var = sum((v-mu)**2 for v in window_vals)/(w-1 if w>1 else 1)
                    rv = math.sqrt(self.ann * max(0.0, var))
                    series.append((r_dates[i], rv))
                else:
                    series.append((r_dates[i], float("nan")))
            rv_ts[w] = series

        # cones and today's stats
        cones: List[ConeRow] = []
        for w in windows:
            vals = [rv for _, rv in rv_ts[w] if math.isfinite(rv)]
            p = _nanpercentiles(vals)
            today_rv = None
            if rv_ts[w]:
                last = rv_ts[w][-1][1]
                today_rv = float(last) if math.isfinite(last) else None
            today_pct = (_pct_rank(vals, today_rv) if (today_rv is not None) else None)
            cones.append(ConeRow(
                window=w, p10=p["p10"], p25=p["p25"], p50=p["p50"], p75=p["p75"], p90=p["p90"],
                today_rv=today_rv, today_pct=today_pct, symbol=symbol
            ))

        # attach implied overlay if provided
        if implied is not None:
            iv_map = self._load_implied(implied)
            for row in cones:
                # match nearest tenor_days to window
                if match_implied and iv_map:
                    nearest = min(iv_map.keys(), key=lambda t: abs(int(t) - int(row.window)))
                    row.implied_iv = float(iv_map.get(nearest)) if iv_map.get(nearest) is not None else None # type: ignore
                else:
                    row.implied_iv = float(iv_map.get(row.window)) if iv_map.get(row.window) is not None else None # type: ignore
                row.implied_minus_p50 = (row.implied_iv - row.p50) if (row.implied_iv is not None and math.isfinite(row.p50)) else None

        # (optional) publish a tiny snapshot
        try:
            publish_stream(VOL_STREAM, {
                "ts_ms": int(datetime.utcnow().timestamp()*1000),
                "symbol": symbol,
                "windows": [r.window for r in cones],
                "today_rv": [r.today_rv for r in cones]
            })
        except Exception:
            pass

        return cones, (rv_ts if return_timeseries else None)

    # -- plotting (optional) --
    def plot(self, cones: List[ConeRow], *, title: str = "Volatility Cones"):
        if _go is None:
            return None
        x = [c.window for c in cones]
        p10 = [c.p10 for c in cones]
        p25 = [c.p25 for c in cones]
        p50 = [c.p50 for c in cones]
        p75 = [c.p75 for c in cones]
        p90 = [c.p90 for c in cones]
        today = [c.today_rv if c.today_rv is not None else None for c in cones]
        impl = [c.implied_iv for c in cones]

        fig = _go.Figure()
        fig.add_trace(_go.Scatter(x=x, y=p10, name="p10", mode="lines"))
        fig.add_trace(_go.Scatter(x=x, y=p25, name="p25", mode="lines"))
        fig.add_trace(_go.Scatter(x=x, y=p50, name="p50 (median)", mode="lines"))
        fig.add_trace(_go.Scatter(x=x, y=p75, name="p75", mode="lines"))
        fig.add_trace(_go.Scatter(x=x, y=p90, name="p90", mode="lines"))
        fig.add_trace(_go.Scatter(x=x, y=today, name="today RV", mode="markers", marker=dict(size=9, symbol="diamond")))
        if any(v is not None for v in impl):
            fig.add_trace(_go.Scatter(x=x, y=impl, name="implied IV", mode="markers", marker=dict(size=10, symbol="x")))
        fig.update_layout(title=title, xaxis_title="Window (trading days)", yaxis_title="Vol (annualized, σ)")
        return fig

    # ---------- loaders ----------
    def _load_prices(self, src: Union[str, List[Dict[str, Any]], Any], *, symbol: Optional[str]) -> Tuple[List[datetime], List[float]]:
        if isinstance(src, str):
            rows = []
            with open(src, "r", newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    if symbol and r.get("symbol") and str(r.get("symbol")) != symbol:
                        continue
                    d = _parse_date(r.get("date"))
                    c = _as_float(r.get("close"))
                    if c is None: continue
                    rows.append((d, c))
            rows.sort(key=lambda x: x[0])
            dates, closes = [r[0] for r in rows], [r[1] for r in rows]
            return dates, closes
        # pandas DataFrame
        if _pd is not None and hasattr(src, "to_dict"):
            df = src.copy()
            if symbol and "symbol" in df.columns: # type: ignore
                df = df[df["symbol"].astype(str) == str(symbol)] # type: ignore
            df = df[["date","close"]].copy() # type: ignore
            df["date"] = df["date"].apply(_parse_date)
            df = df.sort_values("date")
            return df["date"].tolist(), df["close"].astype(float).tolist()
        # list of dicts
        rows: List[Tuple[datetime, float]] = []
        for r in src:  # type: ignore
            if symbol and r.get("symbol") and str(r.get("symbol")) != symbol:
                continue
            d = _parse_date(r.get("date"))
            c = _as_float(r.get("close"))
            if c is None: continue
            rows.append((d, c))
        rows.sort(key=lambda x: x[0])
        return [r[0] for r in rows], [r[1] for r in rows]

    def _load_implied(self, src: Union[str, List[Dict[str, Any]], Any]) -> Dict[int, float]:
        """Return tenor_days -> latest iv for each tenor (use last date per tenor)."""
        rows: List[Dict[str, Any]] = []
        if isinstance(src, str):
            with open(src, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        elif _pd is not None and hasattr(src, "to_dict"):
            rows = src.to_dict(orient="records")  # type: ignore
        else:
            rows = src  # type: ignore
        # group by tenor_days and pick the last by date
        bucket: Dict[int, Tuple[datetime, float]] = {}
        for r in rows:
            try:
                d = _parse_date(r.get("date"))
                t = int(float(r.get("tenor_days"))) # type: ignore
                iv = float(r.get("iv")) # type: ignore
            except Exception:
                continue
            if t not in bucket or d > bucket[t][0]:
                bucket[t] = (d, iv)
        return {t: iv for t, (_, iv) in bucket.items()}

# ---------- serialization helpers ----------
def cones_to_dicts(cones: List[ConeRow]) -> List[Dict[str, Any]]:
    return [asdict(c) for c in cones]

def save_csv(path: str, cones: List[ConeRow]) -> None:
    rows = cones_to_dicts(cones)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    cols = ["symbol","window","p10","p25","p50","p75","p90","today_rv","today_pct","implied_iv","implied_minus_p50"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            if "symbol" not in r: r["symbol"] = None
            wr.writerow({k: r.get(k) for k in cols})

def save_json(path: str, cones: List[ConeRow]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cones_to_dicts(cones), f, indent=2, ensure_ascii=False)

# ---------- CLI ----------
def _parse_windows(s: Optional[str]) -> List[int]:
    if not s: return [5,10,20,60,120,252]
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok: continue
        out.append(int(tok))
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build volatility cones from prices")
    ap.add_argument("--prices", required=True, help="CSV with date,close[,symbol]")
    ap.add_argument("--symbol", default=None, help="optional symbol filter")
    ap.add_argument("--windows", default="5,10,20,60,120,252", help="comma-separated windows (days)")
    ap.add_argument("--implied", default=None, help="optional implied vols CSV (date,tenor_days,iv)")
    ap.add_argument("--out_csv", default=None, help="write cones table as CSV")
    ap.add_argument("--out_json", default=None, help="write cones table as JSON")
    ap.add_argument("--plot", default=None, help="write Plotly HTML chart")
    args = ap.parse_args()

    eng = VolatilityCones()
    windows = _parse_windows(args.windows)
    cones, _ = eng.build(args.prices, symbol=args.symbol, windows=windows, implied=args.implied)

    if args.out_csv:
        save_csv(args.out_csv, cones)
    if args.out_json:
        save_json(args.out_json, cones)
    if args.plot:
        fig = eng.plot(cones, title=f"Volatility Cones — {args.symbol or ''}".strip())
        if fig is None:
            print("plotly not installed; skipping plot.", file=sys.stderr)
        else:
            fig.write_html(args.plot, include_plotlyjs="cdn")

    # stdout summary
    for r in cones:
        print(f"W={r.window:>3}d  p10={r.p10:.3f}  p50={r.p50:.3f}  p90={r.p90:.3f}  today={None if r.today_rv is None else round(r.today_rv,3)}  pct={None if r.today_pct is None else round(r.today_pct,1)}  iv={None if r.implied_iv is None else round(r.implied_iv,3)}")

if __name__ == "__main__":  # pragma: no cover
    main()