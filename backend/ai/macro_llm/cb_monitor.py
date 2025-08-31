# backend/ai/monitors/cb_monitor.py
from __future__ import annotations

"""
Central Bank Monitor
- Optional RSS/Atom polling (feedparser if installed)
- Manual event ingest (push from your pipelines)
- Lightweight hawkish/dovish scorer (finance lexicon)
- Policy metadata extraction (rate/hike/cut/hold; surprise vs expected)
- Emits normalized events to Redis stream or returns dicts

Env (optional):
  REDIS_HOST / REDIS_PORT
  CB_OUT_STREAM         (default: "news.cb")
  CB_SEEN_SET           (default: "cb:seen")
  CB_POLL_INTERVAL_S    (default: 120)

Usage (polling):
  python -m backend.ai.monitors.cb_monitor --poll

Usage (push):
  from backend.ai.monitors.cb_monitor import CBMonitor
  mon = CBMonitor()
  mon.process_text(source="FOMC", title="Statement", text="The Committee decided ...", meta={"expected_rate": 5.25, "actual_rate": 5.50})
"""

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

# ---------- Optional Redis / Bus glue (graceful fallback) ----------
try:
    import redis
    _has_redis = True
except Exception:
    redis = None
    _has_redis = False

try:
    # Your bus helpers if present
    from backend.bus.streams import publish_stream, hset  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass
    def hset(key: str, field: str, value: Any) -> None:
        pass

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CB_OUT_STREAM = os.getenv("CB_OUT_STREAM", "news.cb")
CB_SEEN_SET = os.getenv("CB_SEEN_SET", "cb:seen")
CB_POLL_INTERVAL_S = int(os.getenv("CB_POLL_INTERVAL_S", "120"))

_r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True) if _has_redis else None#type:ignore

# ---------- Optional feed parser ----------
try:
    import feedparser  # pip install feedparser
    _has_feed = True
except Exception:
    _has_feed = False

# ---------- Data models ----------
@dataclass
class CBEvent:
    ts: int
    source: str              # "FOMC" | "ECB" | "BoE" | "BoJ" | "RBI" | "SNB" | ...
    title: str
    text: str
    url: Optional[str] = None
    region: Optional[str] = None
    policy_rate: Optional[float] = None      # parsed/actual
    expected_rate: Optional[float] = None
    action: Optional[str] = None             # "hike" | "cut" | "hold"
    move_bp: Optional[int] = None            # +25 / -50 etc.
    surprise_bp: Optional[int] = None        # (actual-expected)*100
    stance_score: float = 0.0                # [-1, +1] dovish ↔ hawkish
    stance_label: str = "neutral"            # "hawkish" | "neutral" | "dovish"
    risk_flag: Optional[str] = None          # "live", "hint", "qt", "qe" etc.
    raw_meta: Dict[str, Any] = None#type:ignore


_HAWK = {
    "tightening","restrictive","inflation remains elevated","further tightening",
    "rate increases","higher for longer","balance sheet reduction","qt",
    "elevated inflation","labor market remains tight","upside risks","firming","raise rates"
}
_DOVE = {
    "easing","accommodative","rate cuts","cut rates","lower for longer",
    "slowdown in growth","downside risks","q e","asset purchases","support the economy",
    "deterioration","softening","weakening"
}
_HINT = {"will assess", "data dependent", "closely monitor", "prepared to adjust", "proceed carefully"}


_CB_ALIASES = {
    "fomc": "FOMC", "federal reserve": "FOMC", "fed": "FOMC",
    "ecb": "ECB", "european central bank": "ECB",
    "boe": "BoE", "bank of england": "BoE",
    "boj": "BoJ", "bank of japan": "BoJ",
    "rbi": "RBI", "reserve bank of india": "RBI",
    "snb": "SNB", "swiss national bank": "SNB",
    "rba": "RBA", "reserve bank of australia": "RBA",
}

_REGION = {
    "FOMC":"US", "ECB":"EU", "BoE":"UK", "BoJ":"JP", "RBI":"IN", "SNB":"CH", "RBA":"AU"
}

_RATE_RE = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_MOVE_RE = re.compile(r"(?:\b(hike|cut|raise|lower|increase|decrease)\b).*?(\d+)\s?(?:bps|bp|basis points?)", re.I)

def _norm_source(s: str) -> str:
    x = (s or "").strip().lower()
    return _CB_ALIASES.get(x, s.strip() or "UNKNOWN")

def _stance(text: str) -> (float, str, Optional[str]):#type:ignore
    t = text.lower()
    score = 0.0
    for w in _HAWK:
        if w in t: score += 1.0
    for w in _DOVE:
        if w in t: score -= 1.0
    risk = None
    if any(h in t for h in _HINT):
        risk = "hint"
    if "qt" in t or "balance sheet reduction" in t:
        risk = (risk or "qt")
    if "qe" in t or "asset purchases" in t:
        risk = (risk or "qe")
    # squashing to [-1,1]
    score = max(-3.0, min(3.0, score)) / 3.0
    label = "hawkish" if score > 0.15 else ("dovish" if score < -0.15 else "neutral")
    return score, label, risk

def _parse_policy_meta(text: str, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Actual rate
    m = _RATE_RE.search(text)
    if m:
        try:
            out["policy_rate"] = float(m.group(1))
        except Exception:
            pass
    # Verb + move size
    mm = _MOVE_RE.search(text)
    if mm:
        verb = mm.group(1).lower()
        bps = int(mm.group(2))
        if verb in ("hike","raise","increase"):
            out["action"] = "hike"
            out["move_bp"] = +bps
        elif verb in ("cut","lower","decrease"):
            out["action"] = "cut"
            out["move_bp"] = -bps
    # expected vs actual if provided
    if meta:
        exp = meta.get("expected_rate")
        act = meta.get("actual_rate") or out.get("policy_rate")
        if isinstance(exp,(int,float)) and isinstance(act,(int,float)):
            out["expected_rate"] = float(exp)
            out["policy_rate"] = float(act)
            out["surprise_bp"] = int(round((act - exp) * 100))
        # If action missing, infer from exp/act
        if out.get("action") is None and isinstance(exp,(int,float)) and isinstance(act,(int,float)):
            if abs(act-exp) < 1e-9: out["action"]="hold"
            elif act > exp: out["action"]="hike"
            else: out["action"]="cut"
            out["move_bp"] = int(round((act-exp)*100))
    # default action if still unknown
    if out.get("action") is None:
        out["action"] = "hold"
    return out

def _seen_key(uid: str) -> str:
    return f"{uid}"


class CBMonitor:
    def __init__(self, out_stream: str = CB_OUT_STREAM):
        self.out_stream = out_stream
        self.r = _r

   
    def process_text(self, *, source: str, title: str, text: str, url: str | None = None, meta: Dict[str, Any] | None = None) -> CBEvent:
        src = _norm_source(source)
        region = _REGION.get(src)
        score, label, risk = _stance(text)
        fields = _parse_policy_meta(text, meta or {})
        ev = CBEvent(
            ts=int(time.time()*1000),
            source=src,
            title=title.strip(),
            text=text.strip(),
            url=url,
            region=region,
            stance_score=score,
            stance_label=label,
            risk_flag=risk,
            raw_meta=meta or {},
            **fields
        )
        self._emit(ev)
        return ev

  
    def poll(self, feeds: Dict[str, str], *, once: bool = False, interval_s: int = CB_POLL_INTERVAL_S) -> None:
        if not _has_feed:
            raise RuntimeError("feedparser not installed. pip install feedparser")
        while True:
            for source, url in feeds.items():
                try:
                    self._poll_one(source, url)
                except Exception as e:
                    # minimal error reporting to Redis list (if available)
                    if self.r is not None:
                        self.r.lpush("cb:errors", json.dumps({"ts": int(time.time()*1000), "src": source, "err": str(e)}))
            if once:
                break
            time.sleep(max(10, interval_s))

    def _poll_one(self, source: str, url: str) -> None:
        d = feedparser.parse(url)  # type: ignore
        for entry in d.entries:
            uid = entry.get("id") or entry.get("link") or f"{source}:{entry.get('title','')}"
            if self._is_seen(uid):#type:ignore
                continue
            title = entry.get("title","").strip()#type:ignore
            text = (entry.get("summary","") or entry.get("description","") or "").strip()#type:ignore
            link = entry.get("link")
            self.process_text(source=source, title=title, text=text, url=link, meta={})#type:ignore
            self._mark_seen(uid)#type:ignore

    # ---- emit & dedupe ----
    def _emit(self, ev: CBEvent) -> None:
        payload = asdict(ev)
        publish_stream(self.out_stream, payload)  # to risk/news bus
        if self.r is not None:
            # also store last stance per source for dashboards
            hset("cb:last_stance", ev.source, {"score": ev.stance_score, "label": ev.stance_label})

    def _is_seen(self, uid: str) -> bool:
        if self.r is None:
            return False
        return bool(self.r.sismember(CB_SEEN_SET, _seen_key(uid)))

    def _mark_seen(self, uid: str) -> None:
        if self.r is None:
            return
        self.r.sadd(CB_SEEN_SET, _seen_key(uid))


def _main():
    import argparse
    p = argparse.ArgumentParser(description="Central Bank Monitor")
    p.add_argument("--poll", action="store_true", help="Poll default central bank feeds (requires feedparser)")
    p.add_argument("--once", action="store_true", help="Run one poll cycle and exit")
    p.add_argument("--sample", action="store_true", help="Emit example events and exit")
    args = p.parse_args()

    mon = CBMonitor()

    if args.sample:
        samples = [
            dict(source="FOMC", title="FOMC Statement",
                 text="The Committee decided to raise the target range by 25 bps to 5.50%. Inflation remains elevated and the labor market is tight."),
            dict(source="ECB", title="ECB Press Conference",
                 text="The Governing Council decided to keep the key ECB interest rates unchanged at 4.00%. We will continue to monitor the inflation outlook and are prepared to adjust all instruments as necessary."),
            dict(source="RBI", title="MPC Resolution",
                 text="The MPC voted 5-1 to reduce the policy repo rate by 25 bps to 6.25% amid signs of growth slowdown.")
        ]
        for s in samples:
            ev = mon.process_text(**s)#type:ignore
            print(json.dumps(asdict(ev), indent=2))
        return

    if args.poll:
        if not _has_feed:
            raise RuntimeError("feedparser not installed. pip install feedparser")
        # Example public feeds (replace with your curated ones)
        feeds = {
            "FOMC": "https://www.federalreserve.gov/feeds/press_all.xml",
            "ECB":  "https://www.ecb.europa.eu/rss/press.html",
            "BoE":  "https://www.bankofengland.co.uk/boeapps/rss/feeds.aspx?feed=NewsRelease",
            "RBI":  "https://www.rbi.org.in/pressreleases_rss.xml",
        }
        mon.poll(feeds, once=args.once)
        return

    # default: show help
    p.print_help()

if __name__ == "__main__":
    _main()