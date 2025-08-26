# backend/altdata/corp_exhaust.py
"""
Corporate Exhaust (Alt-Data) Ingestion & Signal Engine

- Normalizes heterogeneous “exhaust” data (web traffic, jobs, reviews, social, shipping).
- Builds features (growth, acceleration, z-scores, EMA slope).
- Produces composite alpha signals per ticker/day with pluggable weights.
- Pure-Python, pandas optional.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


# =============================== Data model ===============================

@dataclass
class ExhaustEvent:
    symbol: str
    source: str            # "web" | "jobs" | "reviews" | "social" | "ship" | ...
    metric: str            # "visits" | "postings" | "rating" | "mentions" | "parcels" | ...
    ts: str                # 'YYYY-MM-DD'
    value: float
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================== Adapters =================================

class BaseAdapter:
    source: str = "base"
    def parse(self, rows: Iterable[Dict[str, Any]]) -> List[ExhaustEvent]:
        out: List[ExhaustEvent] = []
        for r in rows:
            out.append(ExhaustEvent(
                symbol=str(r["symbol"]).upper(),
                source=self.source,
                metric=str(r.get("metric","value")),
                ts=_as_iso_date(r.get("ts")),
                value=float(r.get("value", 0.0)),
                meta=dict(r.get("meta", {})),
            ))
        return out

class WebTrafficAdapter(BaseAdapter):
    source = "web"
    def parse(self, rows):
        return [ExhaustEvent(str(r["symbol"]).upper(), self.source, "visits",
                             _as_iso_date(r.get("ts")), float(r.get("visits", r.get("value", 0.0))),
                             dict(r.get("meta", {}))) for r in rows]

class JobPostingsAdapter(BaseAdapter):
    source = "jobs"
    def parse(self, rows):
        return [ExhaustEvent(str(r["symbol"]).upper(), self.source, "postings",
                             _as_iso_date(r.get("ts")), float(r.get("postings", r.get("value", 0.0))),
                             dict(r.get("meta", {}))) for r in rows]

class AppReviewsAdapter(BaseAdapter):
    source = "reviews"
    def parse(self, rows):
        return [ExhaustEvent(str(r["symbol"]).upper(), self.source, "rating",
                             _as_iso_date(r.get("ts")), float(r.get("rating", r.get("value", 0.0))),
                             dict(r.get("meta", {}))) for r in rows]

class SocialMentionsAdapter(BaseAdapter):
    source = "social"
    def parse(self, rows):
        return [ExhaustEvent(str(r["symbol"]).upper(), self.source, "mentions",
                             _as_iso_date(r.get("ts")), float(r.get("mentions", r.get("value", 0.0))),
                             dict(r.get("meta", {}))) for r in rows]

class ShippingAdapter(BaseAdapter):
    source = "ship"
    def parse(self, rows):
        return [ExhaustEvent(str(r["symbol"]).upper(), self.source, "parcels",
                             _as_iso_date(r.get("ts")), float(r.get("parcels", r.get("value", 0.0))),
                             dict(r.get("meta", {}))) for r in rows]


# =============================== Store =====================================

class ExhaustStore:
    def __init__(self):
        self._events: List[ExhaustEvent] = []

    def add(self, events: Iterable[ExhaustEvent]) -> None:
        self._events.extend(events)

    def query(self, *, symbols: Optional[Sequence[str]] = None, sources: Optional[Sequence[str]] = None) -> List[ExhaustEvent]:
        out = self._events
        if symbols:
            S = {s.upper() for s in symbols}
            out = [e for e in out if e.symbol.upper() in S]
        if sources:
            R = {s.lower() for s in sources}
            out = [e for e in out if e.source.lower() in R]
        out.sort(key=lambda e: (e.symbol, e.source, e.metric, e.ts))
        return out

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self._events], indent=2)

    @staticmethod
    def from_json(s: str) -> "ExhaustStore":
        o = json.loads(s)
        st = ExhaustStore()
        st.add(ExhaustEvent(**e) for e in o)
        return st


# =============================== Features ===================================

@dataclass
class FeatureRow:
    symbol: str
    ts: str
    feats: Dict[str, float]
    raw: Dict[str, float]

class FeatureEngine:
    def __init__(self, *, min_history_days: int = 30):
        self.min_hist = int(min_history_days)

    def build(self, events: Iterable[ExhaustEvent], *, calendar: Optional[Sequence[str]] = None) -> List[FeatureRow]:
        panel: Dict[Tuple[str, str], Dict[str, float]] = {}
        for e in events:
            panel.setdefault((e.symbol.upper(), f"{e.source}_{e.metric}"), {})[e.ts] = float(e.value)

        dates = list(calendar) if calendar is not None else sorted({ts for s in panel.values() for ts in s})
        per_symbol: Dict[str, List[FeatureRow]] = {}

        for (sym, _), _ in sorted(panel.items(), key=lambda kv: kv[0][0]):
            keys = sorted({k for (s, k) in panel.keys() if s == sym})
            for ts in dates:
                raw_vals = {k: panel.get((sym, k), {}).get(ts) for k in keys if panel.get((sym, k), {}).get(ts) is not None}
                feats = self._compute_feats(sym, ts, panel, keys, dates)
                if feats:
                    per_symbol.setdefault(sym, []).append(FeatureRow(sym, ts, feats, raw_vals)) # type: ignore

        out: List[FeatureRow] = []
        for arr in per_symbol.values():
            out.extend(arr)
        return out

    def _compute_feats(self, sym: str, ts: str, panel: Dict[Tuple[str, str], Dict[str, float]], keys: List[str], dates: List[str]) -> Dict[str, float]:
        idx = dates.index(ts)
        feats: Dict[str, float] = {}

        def win(k: str, n: int, include_today: bool = False) -> List[float]:
            end = idx + 1 if include_today else idx
            start = max(0, end - n)
            series = panel.get((sym, k), {})
            vals = [series.get(d) for d in dates[start:end]]
            return [float(v) for v in vals if v is not None]

        for k in keys:
            series = panel.get((sym, k), {})
            x_t = series.get(ts)
            lag_7  = _lag_value(series, dates, idx, 7)
            lag_28 = _lag_value(series, dates, idx, 28)

            if x_t is not None and lag_7 not in (None, 0):
                feats[f"{k}_gw"] = _safe_ratio(x_t, lag_7) - 1.0
            if x_t is not None and lag_28 not in (None, 0):
                feats[f"{k}_gm"] = _safe_ratio(x_t, lag_28) - 1.0

            if lag_7 is not None:
                prev_gw = None
                lag_14 = _lag_value(series, dates, idx, 14)
                if lag_14 not in (None, 0):
                    prev_gw = _safe_ratio(lag_7, lag_14) - 1.0
                cur_gw = feats.get(f"{k}_gw")
                if cur_gw is not None and prev_gw is not None:
                    feats[f"{k}_accel"] = cur_gw - prev_gw

            for N in (28, 56, 84):
                w = win(k, N, include_today=True)
                if len(w) >= max(10, N // 2):
                    z = _zscore(w[-1], w[:-1])
                    if z is not None:
                        feats[f"{k}_z{N}"] = z
                        if abs(z) >= 2.5:
                            feats[f"{k}_anomaly"] = 1.0 if z > 0 else -1.0

            wema = win(k, 28, include_today=True)
            if len(wema) >= 14:
                feats[f"{k}_ema14_slope"] = _ema_slope(wema, span=14)

        return feats


# =============================== Signals ====================================

@dataclass
class SignalRow:
    symbol: str
    ts: str
    score: float
    components: Dict[str, float]
    features: Dict[str, float]

class SignalComposer:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = dict(weights or {"web": 0.4, "jobs": 0.2, "reviews": 0.2, "social": 0.2})

    def score_rows(self, feats: List[FeatureRow]) -> List[SignalRow]:
        out: List[SignalRow] = []
        for r in feats:
            comp = {
                "web":    _combine_source(r.feats, prefix="web_",    keys=("visits_gw","visits_gm","visits_accel","visits_z28")),
                "jobs":   _combine_source(r.feats, prefix="jobs_",   keys=("postings_gw","postings_gm","postings_accel","postings_z28")),
                "reviews":_combine_source(r.feats, prefix="reviews_",keys=("rating_gw","rating_accel","rating_z28")),
                "social": _combine_source(r.feats, prefix="social_", keys=("mentions_gw","mentions_accel","mentions_z28")),
                "ship":   _combine_source(r.feats, prefix="ship_",   keys=("parcels_gw","parcels_accel","parcels_z28")),
            }
            raw = 0.0; wsum = 0.0
            for k, v in comp.items():
                w = self.weights.get(k, 0.0)
                if v is None or w == 0.0:
                    continue
                raw += w * max(-3.0, min(3.0, v))
                wsum += abs(w)
            score = 0.0 if wsum == 0.0 else math.tanh(raw / max(1e-9, wsum))
            out.append(SignalRow(r.symbol, r.ts, score, {k:(v if v is not None else 0.0) for k,v in comp.items()}, r.feats))
        return out


# =============================== Utils ======================================

def _as_iso_date(d: Any) -> str:
    if isinstance(d, str) and len(d) >= 8: return d[:10]
    if isinstance(d, (date, datetime)):    return d.date().isoformat() if isinstance(d, datetime) else d.isoformat()
    if isinstance(d, (int, float)):        return datetime.utcfromtimestamp(d).date().isoformat()
    return datetime.utcnow().date().isoformat()

def _lag_value(series: Dict[str, float], dates: List[str], idx: int, lag_days: int) -> Optional[float]:
    j = idx - lag_days
    if j < 0: return None
    return series.get(dates[j])

def _safe_ratio(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a) / float(b)

def _zscore(x_t: float, history: List[float]) -> Optional[float]:
    hist = [float(x) for x in history if x is not None]
    if len(hist) < 8: return None
    mu = statistics.fmean(hist)
    try: sd = statistics.pstdev(hist)
    except Exception: sd = 0.0
    sd = max(sd, 1e-9)
    return (float(x_t) - mu) / sd

def _ema_slope(vals: List[float], span: int = 14) -> float:
    alpha = 2.0 / (span + 1.0)
    ema = None
    for x in vals:
        ema = (x if ema is None else alpha * x + (1 - alpha) * ema)
    if len(vals) < 2 or ema is None: return 0.0
    return (vals[-1] - vals[-2]) / max(1e-9, abs(ema))

def _combine_source(feats: Dict[str, float], *, prefix: str, keys: Sequence[str]) -> Optional[float]:
    vals: List[float] = []
    for k in keys:
        v = feats.get(prefix + k)
        if v is not None: vals.append(float(v))
    return None if not vals else sum(vals) / len(vals)


# =============================== JSON / DF ==================================

def features_to_json(rows: List[FeatureRow]) -> str:
    return json.dumps([{"symbol": r.symbol, "ts": r.ts, "feats": r.feats, "raw": r.raw} for r in rows], indent=2)

def features_from_json(s: str) -> List[FeatureRow]:
    arr = json.loads(s)
    return [FeatureRow(symbol=o["symbol"], ts=o["ts"], feats=o["feats"], raw=o.get("raw", {})) for o in arr]

def signals_to_json(rows: List[SignalRow]) -> str:
    return json.dumps([{"symbol": r.symbol, "ts": r.ts, "score": r.score, "components": r.components, "features": r.features} for r in rows], indent=2)

def signals_to_dataframe(rows: List[SignalRow]):
    if pd is None:
        raise RuntimeError("pandas not installed")
    return pd.DataFrame([{"symbol": r.symbol, "ts": r.ts, "score": r.score, **{f"c_{k}": v for k, v in r.components.items()}} for r in rows])


# =============================== Tiny demo ==================================

if __name__ == "__main__":
    base = datetime.utcnow().date() - timedelta(days=45)
    days = [(base + timedelta(days=i)).isoformat() for i in range(46)]

    def mk_series(symbol: str, key: str, amp: float, noise: float = 0.05):
        rows = []
        for i, d in enumerate(days):
            val = (1.0 + amp * math.sin(i/6.0)) * (1.0 + 0.1 * (i/30.0))
            val *= (1.0 + noise * math.cos(i/3.5))
            rows.append({"symbol": symbol, "ts": d, key: max(0.0, 100.0 * val)})
        return rows

    web = WebTrafficAdapter().parse(mk_series("AAPL", "visits", 0.3) + mk_series("MSFT", "visits", 0.2))
    jobs = JobPostingsAdapter().parse(mk_series("AAPL", "postings", 0.15) + mk_series("MSFT", "postings", 0.1))
    reviews = AppReviewsAdapter().parse([{"symbol":"AAPL","ts": d,"rating": 4.2 + 0.05*math.sin(i/8)} for i,d in enumerate(days)])
    social = SocialMentionsAdapter().parse(mk_series("AAPL", "mentions", 0.4, 0.15))

    store = ExhaustStore()
    store.add(web); store.add(jobs); store.add(reviews); store.add(social)

    feats = FeatureEngine(min_history_days=28).build(store.query(symbols=["AAPL","MSFT"]))
    sigs  = SignalComposer(weights={"web":0.5,"jobs":0.2,"reviews":0.2,"social":0.1}).score_rows(feats)

    print("Features sample:", json.dumps(asdict(feats[-1]) if feats else {}, indent=2))
    print("Signal sample:",   json.dumps(asdict(sigs[-1]) if sigs else {}, indent=2))

    if pd:
        df = signals_to_dataframe(sigs[-20:])
        print(df.tail())