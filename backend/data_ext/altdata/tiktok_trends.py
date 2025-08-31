# backend/altdata/tiktok_trends.py
from __future__ import annotations
"""
TikTok Trends (Alt-Data)
------------------------
Compute trending scores for hashtags, sounds, creators from raw post telemetry.

Inputs (CSV / DataFrame / list[dict]):
  required:
    - id            (post/video id)
    - ts            (timestamp ISO or epoch secs)
  recommended:
    - author        (creator handle/id)
    - caption       (text)
    - hashtags      (semicolon/space/comma-separated if you have it; else auto-extract from caption)
    - sound_id      (music/sound id or name)
    - views         (int)
    - likes         (int)
    - comments      (int)
    - shares        (int)
    - duration_s    (float)
    - country       (optional ISO country)
    - topic         (optional coarse label you add upstream)

Outputs:
  Per-window (e.g., last 24h) aggregates for:
   - hashtags: trend_score, velocity, growth, virality, novelty, sentiment
   - sounds:   trend_score, velocity, virality
   - creators: trend_score, posting_rate, engagement_rate
  Plus a unified "top_trends" table.

No hard deps:
 - With pandas/numpy -> faster & richer stats
 - Without -> pure Python fallbacks

Optional:
 - If backend.ai.nlp.sentiment_ai.SentimentAI is available, sentiment is scored automatically.
 - Emits snapshots to Redis stream "alt.tiktok_trends" if backend.bus.streams.publish_stream exists.

CLI:
  python -m backend.altdata.tiktok_trends --in data/tiktok_posts.csv --out data/tiktok_trends.csv --hours 24

NOTE: This module does not scrape TikTok. Point your own collector to produce a CSV/JSON feed.
"""

import csv, json, math, os, re, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# -------- Optional deps (graceful) --------
try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

# optional bus
try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass

OUT_STREAM = os.getenv("TIKTOK_TRENDS_STREAM", "alt.tiktok_trends")

# optional Sentiment
try:
    from backend.ai.nlp.sentiment_ai import SentimentAI  # type: ignore
    _has_sa = True
except Exception:
    _has_sa = False

# -------- Helpers --------
_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
_CASHTAG_RE = re.compile(r"\$[A-Z]{1,5}(?:\.[A-Z])?")

def _now_utc() -> datetime:
    return datetime.utcfromtimestamp(time.time())

def _parse_ts(x: Any) -> datetime:
    if isinstance(x, (int, float)):
        return datetime.utcfromtimestamp(float(x))
    s = str(x)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
        try: return datetime.strptime(s[:len(fmt)], fmt)
        except Exception: pass
    try: return datetime.fromisoformat(s.replace("Z","")[:19])
    except Exception: return _now_utc()

def _as_int(v: Any) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0

def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _split_tags(s: Any) -> List[str]:
    if not s: return []
    if isinstance(s, list): return [t.strip().lower() for t in s if t]
    txt = str(s)
    # accept "#a #b", "a;b", "a,b"
    if "#" in txt:
        return [m.group(1).lower() for m in _HASHTAG_RE.finditer(txt)]
    sep = ";" if ";" in txt else ("," if "," in txt else " ")
    return [t.strip().lower().lstrip("#") for t in txt.split(sep) if t.strip()]

def _extract_cashtags(txt: str) -> List[str]:
    return list({m.group(0).upper().lstrip("$") for m in _CASHTAG_RE.finditer(txt or "")})

def _safe_div(a: float, b: float) -> float:
    return a / b if (b not in (0, None) and b != 0) else 0.0

def _exp_decay(age_hours: float, half_life_h: float = 24.0) -> float:
    lam = math.log(2) / max(1e-6, half_life_h)
    return math.exp(-lam * max(0.0, age_hours))

# -------- Data classes --------
@dataclass
class TrendRow:
    kind: str         # "hashtag"|"sound"|"creator"
    key: str
    window_h: int
    posts: int
    views: int
    likes: int
    comments: int
    shares: int
    velocity: float         # decayed posts/hour
    virality: float         # engagement per view (likes+comments+shares)/views
    growth: float           # last 6h vs prior 6h ratio (if available)
    novelty: float          # fraction of first-time posters for this key (0..1)
    sentiment: Optional[float]  # [-1,1] avg if SentimentAI available
    score: float            # composite 0..100
    meta: Dict[str, Any]

# -------- Core engine --------
class TikTokTrends:
    def __init__(
        self,
        *,
        window_hours: int = 24,
        half_life_h: float = 24.0,
        score_weights: Tuple[float, float, float, float, float] = (0.40, 0.25, 0.15, 0.10, 0.10),  # vel, virality, growth, novelty, sentiment
        min_posts: int = 3,
        use_sentiment: bool = True,
        emit_stream: str = OUT_STREAM
    ):
        self.win_h = int(window_hours)
        self.hlh = float(half_life_h)
        self.w_vel, self.w_vir, self.w_gro, self.w_nov, self.w_sent = score_weights
        self.min_posts = min_posts
        self.emit_stream = emit_stream
        self.sa = SentimentAI(prefer="hf") if (_has_sa and use_sentiment) else None

    # ---- public API ----
    def compute(self, posts: Union[str, List[Dict[str, Any]], Any], *, return_frame: bool = True, emit: bool = False) -> Any:
        rows = self._load_rows(posts)
        if not rows: return _pd.DataFrame() if (_pd is not None and return_frame) else []

        t_end = max(r["ts_dt"] for r in rows)
        t_start = t_end - timedelta(hours=self.win_h)

        # filter window
        win = [r for r in rows if r["ts_dt"] >= t_start]

        # split early/late halves for growth (last 6h vs prior 6h default; cap by window)
        h6 = min(6, self.win_h // 2)
        cut = t_end - timedelta(hours=h6)
        early = [r for r in win if r["ts_dt"] < cut]
        late  = [r for r in win if r["ts_dt"] >= cut]

        # build aggregations
        trends: List[TrendRow] = []
        trends += self._aggregate(win, early, late, kind="hashtag", keyer=lambda r: r["hashtags"], explode=True)
        trends += self._aggregate(win, early, late, kind="sound",   keyer=lambda r: [r["sound_id"]] if r["sound_id"] else [], explode=True)
        trends += self._aggregate(win, early, late, kind="creator", keyer=lambda r: [r["author"]] if r["author"] else [], explode=True)

        # to frame or list
        if _pd is not None and return_frame:
            import pandas as pd
            df = pd.DataFrame([asdict(t) for t in trends])
            # rank & top_trends view
            if len(df):
                df["rank"] = df.groupby("kind")["score"].rank(ascending=False, method="first")
            if emit and len(df):
                tail = df.sort_values(["kind","score"], ascending=[True, False]).groupby("kind").head(20)
                publish_stream(self.emit_stream, {
                    "ts_ms": int(time.time()*1000),
                    "window_h": self.win_h,
                    "kinds": tail["kind"].nunique(),
                    "rows": tail.to_dict(orient="records")
                })
            return df.sort_values(["kind","score"], ascending=[True, False]).reset_index(drop=True)
        else:
            out = [asdict(t) for t in trends]
            if emit and out:
                publish_stream(self.emit_stream, {"ts_ms": int(time.time()*1000), "window_h": self.win_h, "rows": out[:50]})
            return out

    # ---- loaders/normalizers ----
    def _load_rows(self, posts: Union[str, List[Dict[str, Any]], Any]) -> List[Dict[str, Any]]:
        if isinstance(posts, str):
            # CSV
            out = []
            with open(posts, "r", newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    out.append(self._norm_row(r))
            return out
        # pandas DataFrame
        if _pd is not None and hasattr(posts, "to_dict"):
            return [self._norm_row(r) for r in posts.to_dict(orient="records")]  # type: ignore
        # list of dicts
        return [self._norm_row(r) for r in posts]

    def _norm_row(self, r: Dict[str, Any]) -> Dict[str, Any]:
        cap = str(r.get("caption") or "")
        hashtags = _split_tags(r.get("hashtags") or cap)
        return {
            "id": str(r.get("id") or r.get("video_id") or ""),
            "ts_dt": _parse_ts(r.get("ts") or r.get("timestamp") or r.get("time")),
            "author": str(r.get("author") or r.get("username") or r.get("creator") or ""),
            "caption": cap,
            "hashtags": hashtags,
            "tickers": _extract_cashtags(cap),
            "sound_id": str(r.get("sound_id") or r.get("music") or r.get("sound") or ""),
            "views": _as_int(r.get("views")),
            "likes": _as_int(r.get("likes")),
            "comments": _as_int(r.get("comments")),
            "shares": _as_int(r.get("shares")),
            "duration_s": _as_float(r.get("duration_s"), 0.0),
            "country": (str(r.get("country")) if r.get("country") else None),
            "topic": (str(r.get("topic")) if r.get("topic") else None),
        }

    # ---- aggregation ----
    def _aggregate(
        self,
        win: List[Dict[str, Any]],
        early: List[Dict[str, Any]],
        late: List[Dict[str, Any]],
        *,
        kind: str,
        keyer,
        explode: bool = True
    ) -> List[TrendRow]:
        # collect by key
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        def _add(key: str, row: Dict[str, Any]):
            if not key: return
            buckets.setdefault(key, []).append(row)

        for r in win:
            keys = keyer(r) if explode else [keyer(r)]
            for k in (keys or []):
                _add(str(k).lower(), r)

        out: List[TrendRow] = []
        if not buckets: return out

        # early/late for growth
        early_b: Dict[str, int] = {}
        late_b: Dict[str, int]  = {}
        for r in early:
            for k in (keyer(r) if explode else [keyer(r)]):
                if not k: continue
                kk = str(k).lower()
                early_b[kk] = early_b.get(kk, 0) + 1
        for r in late:
            for k in (keyer(r) if explode else [keyer(r)]):
                if not k: continue
                kk = str(k).lower()
                late_b[kk] = late_b.get(kk, 0) + 1

        # sentiment (avg over captions) if available
        def _avg_sent(rows: List[Dict[str, Any]]) -> Optional[float]:
            if not self.sa: return None
            scs = []
            for rr in rows:
                try:
                    res = self.sa.score(rr.get("caption",""))
                    scs.append(float(res.score))
                except Exception:
                    continue
            return (sum(scs) / len(scs)) if scs else None

        t_end = max(r["ts_dt"] for r in win)
        out_min_posts = max(1, self.min_posts)

        for key, rows in buckets.items():
            n = len(rows)
            if n < out_min_posts:  # minimum mass
                continue
            # sums
            v = sum(r["views"] for r in rows)
            lk = sum(r["likes"] for r in rows)
            cm = sum(r["comments"] for r in rows)
            sh = sum(r["shares"] for r in rows)

            # velocity: time-decayed posts/hour
            vel = 0.0
            for r in rows:
                age_h = (t_end - r["ts_dt"]).total_seconds() / 3600.0
                vel += _exp_decay(age_h, self.hlh)
            vel = vel * (1.0 / max(1.0, self.win_h)) * 24.0  # scale to daily-ish

            # virality: engagement per view (guard small-denom)
            vir = _safe_div(lk + cm + sh, max(v, n * 100.0))

            # growth: late vs early posts ratio
            g = _safe_div(late_b.get(key, 0) - early_b.get(key, 0), max(1.0, early_b.get(key, 0)))

            # novelty: fraction of *first-time* authors using this key in window
            seen_authors: Dict[str, int] = {}
            for r in rows: seen_authors[r["author"]] = seen_authors.get(r["author"], 0) + 1
            first_timers = sum(1 for _, c in seen_authors.items() if c == 1)
            nov = _safe_div(first_timers, len(seen_authors))  # 0..1

            # sentiment
            sent = _avg_sent(rows)

            # composite score → 0..100 (cross-sectional scaling is done later in DataFrame; here, local min-max)
            # Build raw score
            s_raw = (
                self.w_vel  * vel +
                self.w_vir  * vir +
                self.w_gro  * g   +
                self.w_nov  * nov +
                (self.w_sent * (sent or 0.0))
            )

            meta = {
                "views_per_post": _safe_div(v, n),
                "eng_per_post": _safe_div(lk + cm + sh, n),
                "authors": len(seen_authors),
                "country_top": rows[0].get("country"),
            }

            out.append(TrendRow(
                kind=kind, key=key, window_h=self.win_h,
                posts=n, views=v, likes=lk, comments=cm, shares=sh,
                velocity=float(vel), virality=float(vir), growth=float(g), novelty=float(nov),
                sentiment=(None if sent is None else float(sent)),
                score=float(s_raw),
                meta=meta
            ))

        # Cross-sectional 0..100 scaling per kind
        if not out: return out
        # compute min-max on raw score
        raw = [t.score for t in out]
        lo, hi = (min(raw), max(raw))
        span = (hi - lo) if (hi > lo) else 1.0
        for i in range(len(out)):
            out[i].score = 100.0 * ((out[i].score - lo) / span)
        # sort top
        out.sort(key=lambda r: r.score, reverse=True)
        return out

# ------------- CLI -------------
def _read_any(path: str):
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # csv
    if _pd is not None:
        return _pd.read_csv(path)
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def _write_any(path: str, obj: Any):
    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        return
    if _pd is not None and hasattr(obj, "to_csv"):
        obj.to_csv(path, index=False)
        return
    # list of dicts
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(obj[0].keys()))
            wr.writeheader()
            for r in obj: wr.writerow(r)
        return
    # fallback JSON
    _write_any(path + ".json", obj)

def _main():
    import argparse
    p = argparse.ArgumentParser(description="TikTok Trends (no scraping; compute scores from your feed)")
    p.add_argument("--in", dest="inp", required=True, help="CSV/JSON with posts")
    p.add_argument("--out", dest="out", required=True, help="Output CSV/JSON with trends")
    p.add_argument("--hours", dest="hours", type=int, default=24)
    p.add_argument("--emit", action="store_true")
    args = p.parse_args()

    ttt = TikTokTrends(window_hours=args.hours)
    posts = _read_any(args.inp)
    trends = ttt.compute(posts, return_frame=(_pd is not None), emit=args.emit)
    _write_any(args.out, trends)

if __name__ == "__main__":  # pragma: no cover
    _main()