# backend/altdata/internet_usage.py
from __future__ import annotations
"""
Internet Usage Index (Alt-Data)
--------------------------------
Fuse multiple telemetry sources into a 0..100 index:
  • CDN bytes served (bytes/day)
  • Web/app hits (requests/day)
  • ISP utilization (% or bytes)
  • Optional Google Trends (pytrends)

Outputs per (date, region):
  value_0_100, components, smoothed, z, yoy, anomaly_flag

Inputs (CSV/DataFrame/list-of-dicts accepted):
  - CDN:        date,region,bytes
  - HITS:       date,region,hits
  - ISP:        date,region,utilization   (0..1 or 0..100; bytes also ok)
  - TRENDS:     date,region,score         (0..100 from Google Trends or any normalized series)

CLI:
  python -m backend.altdata.internet_usage \
    --cdn data/cdn.csv --hits data/hits.csv --isp data/isp.csv --trends data/trends.csv \
    --out data/internet_usage_index.csv --smoothing 7 --weights 0.4 0.3 0.2 0.1

Optional live pull (if pytrends installed):
  from backend.altdata.internet_usage import TrendsFetcher
  tf = TrendsFetcher(); tf.fetch(["internet outage","whatsapp"], geo="US", timeframe="today 3-m")
"""

import csv, json, math, os, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# ---------- Optional deps (graceful fallbacks) ----------
try:
    import pandas as _pd
except Exception:
    _pd = None
try:
    import numpy as _np
except Exception:
    _np = None
try:
    from pytrends.request import TrendReq as _TrendReq  # type: ignore
    _has_pytrends = True
except Exception:
    _has_pytrends = False
try:
    import redis as _redis  # type: ignore
except Exception:
    _redis = None
try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
OUT_STREAM = os.getenv("INTERNET_USAGE_STREAM", "alt.internet_usage")

# ==================== Helpers ====================

def _to_float(x) -> Optional[float]:
    try:
        f = float(x)
        if math.isnan(f): return None
        return f
    except Exception:
        return None

def _parse_date(s) -> str:
    return str(s)[:10]

def _movavg(xs: List[float], w: int) -> List[float]:
    if w <= 1 or not xs: return xs
    out, s = [], 0.0
    for i, v in enumerate(xs):
        s += v
        if i >= w: s -= xs[i - w]
        out.append(s / min(i + 1, w))
    return out

def _zscore(xs: List[float]) -> List[float]:
    if not xs: return xs
    if _np is not None:
        mu = float(_np.mean(xs)); sd = float(_np.std(xs)) or 1.0
    else:
        mu = sum(xs)/len(xs)
        sd = (sum((x-mu)**2 for x in xs)/max(1, len(xs)-1))**0.5 or 1.0
    return [(x - mu) / sd for x in xs]

def _yoy(dates: List[str], vals: List[float]) -> List[Optional[float]]:
    out = [None]*len(vals)
    if not dates: return out # type: ignore
    idx = {d: i for i, d in enumerate(dates)}
    from datetime import datetime, timedelta
    for i, d in enumerate(dates):
        try:
            dt = datetime.fromisoformat(d[:10])
            prior = (dt - timedelta(days=365)).date().isoformat()
            j = idx.get(prior)
            if j is None: continue
            base = vals[j]
            if base is None or base == 0: continue
            out[i] = (vals[i] - base) / base # type: ignore
        except Exception:
            continue
    return out # type: ignore

def _minmax(xs: List[float]) -> Tuple[float, float]:
    if not xs: return 0.0, 1.0
    lo, hi = min(xs), max(xs)
    if hi <= lo: hi = lo + 1e-9
    return float(lo), float(hi)

def _scale_0_100(x: float, lo: float, hi: float) -> float:
    return 100.0 * (x - lo) / (hi - lo + 1e-12)

# ==================== Data Models ====================

@dataclass
class ComponentWeights:
    cdn: float = 0.4
    hits: float = 0.3
    isp: float = 0.2
    trends: float = 0.1

    def normalize(self):
        s = max(1e-9, self.cdn + self.hits + self.isp + self.trends)
        self.cdn /= s; self.hits /= s; self.isp /= s; self.trends /= s
        return self

@dataclass
class RowOut:
    date: str
    region: str
    value_0_100: float
    smoothed: float
    z: float
    yoy: Optional[float]
    anomaly: bool
    c_cdn: Optional[float] = None
    c_hits: Optional[float] = None
    c_isp: Optional[float] = None
    c_trends: Optional[float] = None

# ==================== Core Engine ====================

class InternetUsageIndex:
    def __init__(
        self,
        *,
        smoothing_days: int = 7,
        anomaly_z: float = 3.0,
        weights: ComponentWeights = ComponentWeights(),
        emit_stream: str = OUT_STREAM
    ):
        self.smoothing_days = max(1, int(smoothing_days))
        self.anomaly_z = float(anomaly_z)
        self.w = weights.normalize()
        self.emit_stream = emit_stream
        self._r = None
        if _redis is not None:
            try:
                self._r = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            except Exception:
                self._r = None

    # ---- public ----

    def compute(
        self,
        *,
        cdn: Union[str, List[Dict[str, Any]], Any, None] = None,
        hits: Union[str, List[Dict[str, Any]], Any, None] = None,
        isp: Union[str, List[Dict[str, Any]], Any, None] = None,
        trends: Union[str, List[Dict[str, Any]], Any, None] = None,
        emit: bool = False,
        return_frame: bool = True
    ):
        """
        Accepts CSV path / DataFrame / list-of-dicts per source.
        Returns a unified index per (date, region).
        """
        # load
        df_cdn  = self._read(cdn, cols=("date","region","bytes"))
        df_hits = self._read(hits, cols=("date","region","hits"))
        df_isp  = self._read(isp, cols=("date","region","utilization"))
        df_tr   = self._read(trends, cols=("date","region","score"))

        # union of all (date, region)
        if _pd is None:
            # pure Python path: coalesce keys
            keys = set()
            for df, dkey, rkey in [(df_cdn,"date","region"),(df_hits,"date","region"),
                                   (df_isp,"date","region"),(df_tr,"date","region")]:
                for r in df: # type: ignore
                    keys.add((_parse_date(r[dkey]), str(r[rkey]))) # type: ignore
            rows = []
            for (d, rgn) in sorted(keys):
                c = self._get(df_cdn, d, rgn, "bytes")
                h = self._get(df_hits, d, rgn, "hits")
                u = self._get(df_isp,  d, rgn, "utilization")
                t = self._get(df_tr,   d, rgn, "score")
                rows.append(self._blend_series(rgn, d, c, h, u, t))
            # fill smoothed/z/yoy/anomaly per region
            out = self._post_process(rows)
            if emit:
                self._emit_tail(out)
            return out
        else:
            # pandas path
            frames = []
            if df_cdn is not None:  frames.append(df_cdn.rename(columns={"bytes":"cdn"})) # type: ignore
            if df_hits is not None: frames.append(df_hits.rename(columns={"hits":"hits"})) # type: ignore
            if df_isp is not None:  frames.append(df_isp.rename(columns={"utilization":"isp"})) # type: ignore
            if df_tr is not None:   frames.append(df_tr.rename(columns={"score":"trends"})) # type: ignore
            if not frames:
                return _pd.DataFrame(columns=["date","region","value_0_100","smoothed","z","yoy","anomaly"])
            # outer join
            df = frames[0]
            for f in frames[1:]:
                df = df.merge(f, on=["date","region"], how="outer")
            # normalize per-component to 0..100 within each region (robust to scale)
            df["date"] = _pd.to_datetime(df["date"]).dt.date.astype(str)
            reg_groups = df.groupby("region", dropna=False)
            for comp in ["cdn","hits","isp","trends"]:
                if comp in df.columns:
                    lo = reg_groups[comp].transform("min")
                    hi = reg_groups[comp].transform("max")
                    df[f"c_{comp}"] = 100.0 * (df[comp] - lo) / (hi - lo + 1e-12)
            # weighted blend
            df["value_0_100"] = (
                (df.get("c_cdn", 0)*self.w.cdn).fillna(0) +
                (df.get("c_hits",0)*self.w.hits).fillna(0) +
                (df.get("c_isp", 0)*self.w.isp).fillna(0) +
                (df.get("c_trends",0)*self.w.trends).fillna(0)
            )
            # per-region smoothing/z/yoy/anomaly
            outs = []
            for rgn, g in df.sort_values("date").groupby("region", dropna=False):
                vals = g["value_0_100"].fillna(method="ffill").fillna(0).tolist()
                sm = _movavg(vals, self.smoothing_days)
                zz = _zscore(sm)
                yy = _yoy(g["date"].tolist(), sm)
                anom = [abs(z) >= self.anomaly_z for z in zz]
                gg = g.copy()
                gg["smoothed"] = sm
                gg["z"] = zz
                gg["yoy"] = yy
                gg["anomaly"] = anom
                outs.append(gg)
            out = _pd.concat(outs, axis=0).sort_values(["region","date"])
            cols = ["date","region","value_0_100","smoothed","z","yoy","anomaly",
                    "c_cdn","c_hits","c_isp","c_trends"]
            out = out[ [c for c in cols if c in out.columns] ]
            if emit:
                self._emit_tail(out.tail(1))
            return out

    # ================= internal =================

    def _read(self, src, cols: Tuple[str, ...]):
        if src is None:
            return None if _pd is not None else []
        if isinstance(src, str):
            # CSV
            if _pd is None:
                out = []
                with open(src, "r", newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        d = {k: r.get(k) for k in cols}
                        if None in d.values(): continue
                        d = dict(d)
                        d["date"] = _parse_date(d["date"])
                        return_key = [x for x in d.keys() if x not in ("date","region")][0]
                        d[return_key] = _to_float(d[return_key])
                        out.append(d)
                return out
            else:
                df = _pd.read_csv(src)
                have = [c for c in cols if c in df.columns]
                if len(have) < 3:
                    raise ValueError(f"CSV missing required columns {cols}")
                df = df[list(cols)].copy()
                # parse types
                df["date"] = _pd.to_datetime(df["date"]).dt.date.astype(str)
                val_col = [c for c in cols if c not in ("date","region")][0]
                df[val_col] = _pd.to_numeric(df[val_col], errors="coerce")
                return df
        # DataFrame
        if _pd is not None and hasattr(src, "to_dict"):
            df = src.copy()
            miss = [c for c in cols if c not in df.columns]
            if miss:
                raise ValueError(f"DataFrame missing {miss}")
            df["date"] = _pd.to_datetime(df["date"]).dt.date.astype(str)
            val_col = [c for c in cols if c not in ("date","region")][0]
            df[val_col] = _pd.to_numeric(df[val_col], errors="coerce")
            return df
        # list of dicts
        if isinstance(src, list):
            out = []
            for r in src:
                d = {k: r.get(k) for k in cols}
                if None in d.values(): continue
                d = dict(d)
                d["date"] = _parse_date(d["date"])
                val_col = [c for c in cols if c not in ("date","region")][0]
                d[val_col] = _to_float(d[val_col])
                out.append(d)
            return out
        raise ValueError("Unsupported source type")

    def _get(self, rows, d: str, region: str, col: str) -> Optional[float]:
        if not rows: return None
        for r in rows:
            if _parse_date(r["date"]) == d and str(r["region"]) == region:
                return _to_float(r.get(col))
        return None

    def _blend_series(self, rgn: str, d: str, c: Optional[float], h: Optional[float], u: Optional[float], t: Optional[float]) -> Dict[str, Any]:
        comps = {"c_cdn": None, "c_hits": None, "c_isp": None, "c_trends": None}
        # In pure-Python path, we’ll do min-max per region after collection in _post_process.
        return {"date": d, "region": rgn, "cdn": c, "hits": h, "isp": u, "trends": t, **comps}

    def _post_process(self, rows: List[Dict[str, Any]]):
        # compute per-region min-max for each comp, then blend
        by_reg: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_reg.setdefault(r["region"], []).append(r)
        out: List[Dict[str, Any]] = []
        for rgn, lst in by_reg.items():
            # min-max per comp
            comps = ["cdn","hits","isp","trends"]
            bounds: Dict[str, Tuple[float,float]] = {}
            for c in comps:
                vals = [v[c] for v in lst if v.get(c) is not None]
                lo, hi = _minmax(vals) if vals else (0.0, 1.0)
                bounds[c] = (lo, hi)
            # scale and blend
            vals_series: List[float] = []
            for r in sorted(lst, key=lambda x: x["date"]):
                cc = {}
                v_acc = 0.0
                for c, w in [("cdn", self.w.cdn), ("hits", self.w.hits), ("isp", self.w.isp), ("trends", self.w.trends)]:
                    v = r.get(c)
                    if v is None:
                        cc[f"c_{c}"] = None
                        continue
                    lo, hi = bounds[c]
                    cc[f"c_{c}"] = _scale_0_100(v, lo, hi)
                    v_acc += (cc[f"c_{c}"] or 0.0) * w
                vals_series.append(v_acc)
                out.append({"date": r["date"], "region": rgn, "value_0_100": v_acc, **cc})
            # backfill smoothed/z/yoy/anomaly
            sm = _movavg(vals_series, self.smoothing_days)
            zz = _zscore(sm)
            yy = _yoy([x["date"] for x in sorted(lst, key=lambda x: x["date"])], sm)
            for i, r in enumerate([x for x in out if x["region"] == rgn]):
                r["smoothed"] = sm[i]; r["z"] = zz[i]; r["yoy"] = yy[i]; r["anomaly"] = (abs(zz[i]) >= self.anomaly_z)
        return out

    def _emit_tail(self, tail):
        try:
            if _pd is not None and hasattr(tail, "to_dict"):
                payload = tail.tail(1).to_dict(orient="records")[0]
            else:
                payload = tail[-1]
            payload = {"ts_ms": int(time.time()*1000), "internet_usage": payload}
            publish_stream(self.emit_stream, payload)
        except Exception:
            pass

    # ------------- Serialization -------------
    @staticmethod
    def to_json(obj: Any, limit: Optional[int] = None) -> str:
        if _pd is not None and hasattr(obj, "to_dict"):
            df = obj.tail(limit) if (limit and len(obj) > limit) else obj
            return json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2)
        return json.dumps(obj[-limit:] if limit else obj, ensure_ascii=False, indent=2)

# ==================== Optional Trends fetcher ====================

class TrendsFetcher:
    """
    Thin wrapper around pytrends to build a region/date/score DataFrame
    you can pass as `trends` to InternetUsageIndex.compute().
    """
    def __init__(self, *, hl: str = "en-US", tz: int = 0):
        if not _has_pytrends:
            raise RuntimeError("pytrends is not installed. pip install pytrends")
        self.pt = _TrendReq(hl=hl, tz=tz)

    def fetch(self, terms: List[str], *, geo: str = "", timeframe: str = "today 3-m", cat: int = 0, gprop: str = ""):
        """
        terms: list of search terms
        geo: region code (e.g., "US", "IN", "GB", "US-CA")
        timeframe: "today 3-m", "now 7-d", "2023-01-01 2024-01-01", etc.
        Returns pandas DataFrame: date,region,score
        """
        self.pt.build_payload(terms, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
        df = self.pt.interest_over_time().reset_index()
        if df.empty:
            return _pd.DataFrame(columns=["date","region","score"]) # type: ignore
        df = df.rename(columns={"date":"date"})
        # combine terms (mean)
        score = df[terms].mean(axis=1)
        out = _pd.DataFrame({"date": _pd.to_datetime(df["date"]).dt.date.astype(str), # type: ignore
                             "region": geo or "GLOBAL",
                             "score": _pd.to_numeric(score, errors="coerce")}) # type: ignore
        return out

# ==================== CLI ====================

def _read_any(path: Optional[str]):
    if path is None: return None
    if _pd is None:
        rows = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        return rows
    return _pd.read_csv(path)

def _write_any(path: str, obj: Any):
    if _pd is not None and hasattr(obj, "to_csv"):
        obj.to_csv(path, index=False); return
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not obj: f.write(""); return
        wr = csv.DictWriter(f, fieldnames=list(obj[0].keys()))
        wr.writeheader()
        for r in obj: wr.writerow(r)

def _main():
    import argparse
    p = argparse.ArgumentParser(description="Internet Usage Index")
    p.add_argument("--cdn", type=str, default=None)
    p.add_argument("--hits", type=str, default=None)
    p.add_argument("--isp", type=str, default=None)
    p.add_argument("--trends", type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--smoothing", type=int, default=7)
    p.add_argument("--weights", type=float, nargs=4, default=(0.4,0.3,0.2,0.1))
    p.add_argument("--emit", action="store_true")
    args = p.parse_args()

    w = ComponentWeights(cdn=args.weights[0], hits=args.weights[1], isp=args.weights[2], trends=args.weights[3]).normalize()
    ix = InternetUsageIndex(smoothing_days=args.smoothing, weights=w)
    out = ix.compute(
        cdn=_read_any(args.cdn),
        hits=_read_any(args.hits),
        isp=_read_any(args.isp),
        trends=_read_any(args.trends),
        emit=args.emit,
        return_frame=True
    )
    _write_any(args.out, out)

if __name__ == "__main__":  # pragma: no cover
    _main()