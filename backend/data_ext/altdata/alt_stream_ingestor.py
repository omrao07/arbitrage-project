# backend/altdata/alt_stream_ingestor.py
from __future__ import annotations

import os, asyncio, time, json, hashlib, math, re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable, Tuple, Iterable

# ---------------- optional deps (graceful) -----------------------------------
HAVE_FEEDPARSER = True
try:
    import feedparser  # RSS/Atom
except Exception:
    HAVE_FEEDPARSER = False

HAVE_AIOHTTP = True
try:
    import aiohttp  # type: ignore # async HTTP
except Exception:
    HAVE_AIOHTTP = False
    aiohttp = None  # type: ignore

HAVE_WS = True
try:
    import websockets  # simple ws client
except Exception:
    HAVE_WS = False

HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

HAVE_YAML = True
try:
    import yaml  # config loader (optional)
except Exception:
    HAVE_YAML = False

# ---------------- env / streams ---------------------------------------------
REDIS_URL            = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ALT_OUT_STREAM       = os.getenv("ALTDATA_OUT_STREAM", "altdata.events")
ALT_FEATURE_PREFIX   = os.getenv("ALTDATA_FEATURE_PREFIX", "features.alt")  # features.alt.news, features.alt.cardspend,...
ALTDATA_MAXLEN       = int(os.getenv("ALTDATA_MAXLEN", "5000"))
DEFAULT_POLL_SEC     = int(os.getenv("ALTDATA_POLL_SEC", "60"))

YAHOO_RSS_URL        = os.getenv("YAHOO_RSS_URL", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,MSFT,GOOGL&region=US&lang=en-US")
MONEYC_RSS_URL       = os.getenv("MONEYC_RSS_URL", "https://www.moneycontrol.com/rss/marketreports.xml")

# ---------------- small utils ------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()

def json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(obj))

# Minimal JSONPath-lite extractor: dotted paths + [idx] + //fallback
_PATH_RE = re.compile(r"""(?x)
    (?P<key>[^.\[\]/]+)        # key
  | \[(?P<idx>-?\d+)\]         # [index]
  | \/\/                       # // fallback separator (OR)
  | \.                         # dot
""")

def jget(obj: Any, path: str, default: Any = None) -> Any:
    """
    Extract value using a very small JSONPath-like syntax:
      "a.b[0].c" or "entry/id // id // guid"
    Returns `default` if nothing found.
    """
    # support // meaning "try next path if None"
    for alt in [p.strip() for p in path.split("//")]:
        cur = obj
        ok = True
        tokens = [m for m in _PATH_RE.finditer(alt)]
        i = 0
        # assemble segments: keys or [idx]
        segs: List[Tuple[str, Optional[int]]] = []
        buf = ""
        for m in tokens:
            if m.group("key"):
                buf = (buf + "." + m.group("key")) if buf else m.group("key")
            elif m.group("idx"):
                segs.append((buf, int(m.group("idx"))))
                buf = ""
        if buf:
            segs.append((buf, None))
        try:
            for key, idx in segs:
                if key:
                    for k in key.split("."):
                        if k == "":
                            continue
                        if isinstance(cur, dict) and k in cur:
                            cur = cur[k]
                        else:
                            ok = False; break
                    if not ok: break
                if idx is not None:
                    if isinstance(cur, (list, tuple)) and len(cur) > 0:
                        cur = cur[idx]
                    else:
                        ok = False; break
            if ok:
                return cur
        except Exception:
            pass
    return default

# ---------------- schemas ----------------------------------------------------
@dataclass
class NewsEvent:
    ts_ms: int
    source: str
    id: str
    title: str
    url: str
    summary: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    lang: Optional[str] = None
    quality: float = 1.0
    type: str = "news"

@dataclass
class TimeseriesEvent:
    ts_ms: int
    source: str
    feature: str              # e.g., "card_spend_yoy"
    entity: str               # e.g., "US" or "AAPL"
    value: float
    unit: Optional[str] = None
    quality: float = 1.0
    tags: List[str] = field(default_factory=list)
    type: str = "timeseries"

# ---------------- deduper ----------------------------------------------------
class Deduper:
    def __init__(self, max_size: int = 10000):
        self.max = max_size
        self._seen: Dict[str, int] = {}  # hash -> ts_ms

    def seen(self, key: str) -> bool:
        if key in self._seen:
            return True
        if len(self._seen) >= self.max:
            # drop oldest half
            keys = sorted(self._seen.items(), key=lambda kv: kv[1])[: self.max // 2]
            for k, _ in keys:
                self._seen.pop(k, None)
        self._seen[key] = now_ms()
        return False

# ---------------- publisher --------------------------------------------------
class Publisher:
    def __init__(self, url: str = REDIS_URL):
        self.url = url
        self.r: Optional[AsyncRedis] = None # type: ignore

    async def connect(self):
        if not HAVE_REDIS:
            return
        try:
            self.r = AsyncRedis.from_url(self.url, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    async def publish_raw(self, obj: Dict[str, Any]):
        if not self.r:
            print("[altdata.raw]", obj); return
        try:
            await self.r.xadd(ALT_OUT_STREAM, {"json": json_dumps(obj)}, maxlen=ALTDATA_MAXLEN, approximate=True)  # type: ignore
        except Exception:
            pass

    async def publish_feature(self, feature: str, obj: Dict[str, Any]):
        if not self.r:
            print(f"[altdata.feature:{feature}]", obj); return
        try:
            stream = f"{ALT_FEATURE_PREFIX}.{feature}"
            await self.r.xadd(stream, {"json": json_dumps(obj)}, maxlen=ALTDATA_MAXLEN, approximate=True)  # type: ignore
        except Exception:
            pass

# ---------------- base connector --------------------------------------------
class Connector:
    name: str
    async def run(self, publisher: Publisher, deduper: Deduper):
        raise NotImplementedError

# ---------------- RSS connector (Yahoo, Moneycontrol, generic) ---------------
class RssConnector(Connector):
    def __init__(self, name: str, url: str, symbols_hint: List[str] | None = None, lang: Optional[str] = None, poll_sec: int = DEFAULT_POLL_SEC):
        self.name = name
        self.url = url
        self.lang = lang
        self.poll_sec = int(max(15, poll_sec))
        self.symbols_hint = symbols_hint or []
        self._last_ids: set[str] = set()

    async def _fetch(self) -> List[Dict[str, Any]]:
        if not HAVE_FEEDPARSER:
            # graceful: try HTTP GET and parse minimal RSS tags
            if not HAVE_AIOHTTP:
                return []
            async with aiohttp.ClientSession() as sess:  # type: ignore
                async with sess.get(self.url, timeout=20) as resp:
                    txt = await resp.text()
            # crude parse for <item> ... </item>
            items = []
            for m in re.finditer(r"<item>(.*?)</item>", txt, flags=re.S | re.I):
                item = m.group(1)
                def _tag(t): 
                    m2 = re.search(rf"<{t}[^>]*>(.*?)</{t}>", item, flags=re.S|re.I); 
                    return (m2.group(1).strip() if m2 else None)
                items.append({
                    "id": _tag("guid") or sha256((_tag("link") or "")+(_tag("title") or "")),
                    "title": _tag("title") or "",
                    "link": _tag("link") or "",
                    "summary": _tag("description") or "",
                    "published": _tag("pubDate"),
                })
            return items
        # feedparser path
        feed = feedparser.parse(self.url)  # type: ignore
        out = []
        for e in feed.entries:
            eid = (getattr(e, "id", None) or getattr(e, "guid", None) or getattr(e, "link", None) or getattr(e, "title", ""))
            t = str(getattr(e, "title", "") or "")
            link = str(getattr(e, "link", "") or "")
            summ = str(getattr(e, "summary", "") or getattr(e, "description", "") or "")
            # published parse
            ts = now_ms()
            try:
                if getattr(e, "published_parsed", None):
                    import calendar
                    ts = int(calendar.timegm(e.published_parsed) * 1000) # type: ignore
            except Exception:
                pass
            out.append({
                "id": eid, "title": t, "link": link, "summary": summ, "ts_ms": ts
            })
        return out

    def _symbols_from_text(self, text: str) -> List[str]:
        syms = set()
        # naive ticker extraction like (AAPL), NSE:RELIANCE, BSE:500325
        for m in re.finditer(r"\b([A-Z]{2,6})\b", text):
            s = m.group(1)
            if s in {"CEO","CFO","USD","EPS","RSS"}: 
                continue
            syms.add(s)
        for s in self.symbols_hint:
            if s in text:
                syms.add(s)
        return sorted(syms)

    async def run(self, publisher: Publisher, deduper: Deduper):
        while True:
            try:
                items = await self._fetch()
                for it in items:
                    nid = str(it.get("id") or sha256((it.get("title") or "") + (it.get("link") or "")))
                    if nid in self._last_ids: 
                        continue
                    self._last_ids.add(nid)
                    ev = NewsEvent(
                        ts_ms=int(it.get("ts_ms") or now_ms()),
                        source=self.name,
                        id=nid,
                        title=str(it.get("title") or ""),
                        url=str(it.get("link") or ""),
                        summary=str(it.get("summary") or ""),
                        symbols=self._symbols_from_text((it.get("title") or "") + " " + (it.get("summary") or "")),
                        tags=[],
                        lang=self.lang or "en",
                        quality=1.0
                    )
                    h = sha256(ev.id + ev.title + ev.url)
                    if deduper.seen(h):
                        continue
                    await publisher.publish_raw({"type":"news","source":self.name, "payload": asdict(ev)})
                    await publisher.publish_feature("news", asdict(ev))
            except Exception as e:
                await publisher.publish_raw({"type":"error","source":self.name,"error":str(e),"ts_ms":now_ms()})
            await asyncio.sleep(self.poll_sec)

# ---------------- HTTP JSON connector (generic timeseries) -------------------
@dataclass
class JsonFieldMap:
    ts_path: str
    entity_path: str
    value_path: str
    feature: str
    unit: Optional[str] = None
    tags_path: Optional[str] = None

class HttpJsonConnector(Connector):
    def __init__(self, name: str, url: str, field: JsonFieldMap, headers: Optional[Dict[str,str]] = None, poll_sec: int = DEFAULT_POLL_SEC, list_path: Optional[str] = None):
        self.name = name
        self.url = url
        self.headers = headers or {}
        self.poll_sec = int(max(10, poll_sec))
        self.field = field
        self.list_path = list_path  # path to array of rows

    async def _fetch(self) -> Any:
        if not HAVE_AIOHTTP:
            return None
        async with aiohttp.ClientSession() as sess:  # type: ignore
            async with sess.get(self.url, headers=self.headers, timeout=25) as resp:
                return await resp.json(content_type=None)

    async def run(self, publisher: Publisher, deduper: Deduper):
        while True:
            try:
                data = await self._fetch()
                if data is None:
                    await asyncio.sleep(self.poll_sec); continue
                rows = jget(data, self.list_path, data) if self.list_path else data
                if not isinstance(rows, list):
                    rows = [rows]
                for row in rows:
                    ts = jget(row, self.field.ts_path)
                    # ts normalization: accept epoch seconds/ms or ISO string
                    ts_ms = _coerce_ts_ms(ts)
                    ent = str(jget(row, self.field.entity_path, "UNK"))
                    val = float(jget(row, self.field.value_path, float("nan")))
                    if math.isnan(val): 
                        continue
                    tags = jget(row, self.field.tags_path, []) if self.field.tags_path else []
                    ev = TimeseriesEvent(
                        ts_ms=ts_ms, source=self.name, feature=self.field.feature,
                        entity=ent, value=val, unit=self.field.unit, tags=tags, quality=1.0
                    )
                    key = sha256(f"{self.name}|{ev.feature}|{ev.entity}|{ev.ts_ms}|{ev.value}")
                    if deduper.seen(key):
                        continue
                    await publisher.publish_raw({"type":"timeseries","source":self.name,"payload":asdict(ev)})
                    await publisher.publish_feature(self.field.feature, asdict(ev))
            except Exception as e:
                await publisher.publish_raw({"type":"error","source":self.name,"error":str(e),"ts_ms":now_ms()})
            await asyncio.sleep(self.poll_sec)

def _coerce_ts_ms(v: Any) -> int:
    try:
        if v is None:
            return now_ms()
        if isinstance(v, (int, float)):
            x = int(v)
            # if seconds, scale to ms
            return x if x > 10_000_000_000 else x * 1000
        if isinstance(v, str):
            # try ISO
            from datetime import datetime
            try:
                return int(datetime.fromisoformat(v.replace("Z","+00:00")).timestamp() * 1000)
            except Exception:
                return now_ms()
        return now_ms()
    except Exception:
        return now_ms()

# ---------------- WebSocket connector (optional) -----------------------------
class WebSocketConnector(Connector):
    def __init__(self, name: str, url: str, feature: str, entity_key: str = "symbol", value_key: str = "value", ts_key: Optional[str] = None):
        self.name = name
        self.url = url
        self.feature = feature
        self.entity_key = entity_key
        self.value_key = value_key
        self.ts_key = ts_key

    async def run(self, publisher: Publisher, deduper: Deduper):
        if not HAVE_WS:
            await publisher.publish_raw({"type":"error","source":self.name,"error":"websockets not installed","ts_ms":now_ms()})
            return
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:  # type: ignore
                    await publisher.publish_raw({"type":"info","source":self.name,"msg":"ws connected","ts_ms":now_ms()})
                    async for msg in ws:
                        try:
                            j = json.loads(msg)
                        except Exception:
                            continue
                        ent = str(j.get(self.entity_key, "UNK"))
                        val = j.get(self.value_key)
                        if val is None:
                            continue
                        ts_ms = _coerce_ts_ms(j.get(self.ts_key)) if self.ts_key else now_ms()
                        ev = TimeseriesEvent(ts_ms=ts_ms, source=self.name, feature=self.feature, entity=ent, value=float(val))
                        key = sha256(f"{self.name}|{self.feature}|{ent}|{ts_ms}|{val}")
                        if not deduper.seen(key):
                            await publisher.publish_raw({"type":"timeseries","source":self.name,"payload":asdict(ev)})
                            await publisher.publish_feature(self.feature, asdict(ev))
            except Exception as e:
                await publisher.publish_raw({"type":"error","source":self.name,"error":str(e),"ts_ms":now_ms()})
                await asyncio.sleep(5)

# ---------------- Orchestrator ----------------------------------------------
class AltStreamIngestor:
    def __init__(self, connectors: List[Connector]):
        self.connectors = connectors
        self.pub = Publisher()
        self.deduper = Deduper()

    async def run(self):
        await self.pub.connect()
        tasks = [asyncio.create_task(c.run(self.pub, self.deduper)) for c in self.connectors]
        await asyncio.gather(*tasks)

# ---------------- YAML loader (optional) -------------------------------------
def load_from_yaml(path: str) -> List[Connector]:
    if not HAVE_YAML:
        raise RuntimeError("PyYAML not installed; cannot read YAML config.")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    conns: List[Connector] = []
    for s in cfg.get("sources", []):
        typ = s.get("type")
        name = s.get("name")
        if typ == "rss":
            conns.append(RssConnector(
                name=name, url=s["url"],
                symbols_hint=s.get("symbols_hint"), lang=s.get("lang"),
                poll_sec=int(s.get("poll_sec", DEFAULT_POLL_SEC))
            ))
        elif typ == "http_json":
            fmap = JsonFieldMap(
                ts_path=s["field"]["ts_path"],
                entity_path=s["field"]["entity_path"],
                value_path=s["field"]["value_path"],
                feature=s["field"]["feature"],
                unit=s["field"].get("unit"),
                tags_path=s["field"].get("tags_path"),
            )
            conns.append(HttpJsonConnector(
                name=name, url=s["url"], headers=s.get("headers"),
                poll_sec=int(s.get("poll_sec", DEFAULT_POLL_SEC)),
                list_path=s.get("list_path"),
                field=fmap
            ))
        elif typ == "ws":
            conns.append(WebSocketConnector(
                name=name, url=s["url"],
                feature=s["feature"],
                entity_key=s.get("entity_key","symbol"),
                value_key=s.get("value_key","value"),
                ts_key=s.get("ts_key")
            ))
    return conns

# ---------------- CLI --------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("alt_stream_ingestor")
    ap.add_argument("--yaml", type=str, default=None, help="configs/altdata.yaml")
    ap.add_argument("--yahoo", action="store_true", help="Enable Yahoo Finance RSS connector (env YAHOO_RSS_URL)")
    ap.add_argument("--moneycontrol", action="store_true", help="Enable Moneycontrol RSS connector (env MONEYC_RSS_URL)")
    ap.add_argument("--poll", type=int, default=DEFAULT_POLL_SEC, help="Polling seconds for RSS/HTTP")
    args = ap.parse_args()

    conns: List[Connector] = []
    if args.yaml:
        conns = load_from_yaml(args.yaml)
    else:
        if args.yahoo:
            conns.append(RssConnector("yahoo_rss", YAHOO_RSS_URL, poll_sec=args.poll))
        if args.moneycontrol:
            conns.append(RssConnector("moneycontrol_rss", MONEYC_RSS_URL, poll_sec=args.poll))

    if not conns:
        # sensible default: both RSS feeds
        conns = [
            RssConnector("yahoo_rss", YAHOO_RSS_URL, poll_sec=args.poll),
            RssConnector("moneycontrol_rss", MONEYC_RSS_URL, poll_sec=args.poll),
        ]

    ing = AltStreamIngestor(conns)
    try:
        asyncio.run(ing.run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    _cli()