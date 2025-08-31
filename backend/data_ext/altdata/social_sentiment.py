# backend/altdata/social_sentiment.py
from __future__ import annotations
import os, re, time, json, math, asyncio, random, hashlib, statistics
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# ---------- Optional deps (graceful) -----------------------------------------
HAVE_AIOHTTP = True
try:
    import aiohttp # type: ignore
except Exception:
    HAVE_AIOHTTP = False
    aiohttp = None  # type: ignore

HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

HAVE_TRANSFORMERS = True
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    import torch  # type: ignore
except Exception:
    HAVE_TRANSFORMERS = False

HAVE_VADER = True
try:
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    _vader_ok = True
except Exception:
    HAVE_VADER = False
    _vader_ok = False

# ---------- Env / Streams ----------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OUT_RAW   = os.getenv("ALTDATA_OUT_STREAM", "altdata.events")
OUT_POST  = os.getenv("SOCIAL_OUT_STREAM", "features.alt.social_sentiment")
OUT_AGG   = os.getenv("SOCIAL_AGG_STREAM",  "features.alt.social_sentiment_agg")
MAXLEN    = int(os.getenv("ALTDATA_MAXLEN", "8000"))

# Twitter/X
X_BEARER       = os.getenv("X_BEARER", "")
X_QUERY        = os.getenv("X_QUERY", "(($AAPL OR $MSFT OR $TSLA) OR (NSE OR BSE)) lang:en -is:retweet")
X_POLL_SEC     = int(os.getenv("X_POLL_SEC", "45"))

# Reddit
REDDIT_SR      = os.getenv("REDDIT_SR", "stocks+wallstreetbets+IndianStockMarket")
REDDIT_POLL    = int(os.getenv("REDDIT_POLL_SEC", "60"))

# Stocktwits
STWITS_SYMBOLS = os.getenv("STWITS_SYMBOLS", "AAPL,MSFT,TSLA,RELIANCE,TCS,INFY")
STWITS_POLL    = int(os.getenv("STWITS_POLL_SEC", "45"))

# ---------- Utilities --------------------------------------------------------
def now_ms() -> int: return int(time.time() * 1000)
def sha(s: str) -> str: return hashlib.sha256(s.encode("utf-8","ignore")).hexdigest()

CASHTAG = re.compile(r"(?:(?<!\w)\$|(?:NSE:|BSE:))([A-Z]{1,10})")
HASHTAG = re.compile(r"(?<!\w)#([A-Za-z0-9_]{2,50})")

def extract_symbols(text: str) -> List[str]:
    syms = set([m.group(1).upper() for m in CASHTAG.finditer(text or "")])
    # quick Indian hints
    for m in re.finditer(r"\b(RELIANCE|TCS|INFY|HDFCBANK|ICICIBANK|SBIN|ITC|HINDUNILVR|BAJFINANCE|LT)\b", text or "", re.I):
        syms.add(m.group(1).upper())
    return sorted(syms)

def extract_hashtags(text: str) -> List[str]:
    return sorted({m.group(1).lower() for m in HASHTAG.finditer(text or "")})

# ---------- Publisher & Deduper ---------------------------------------------
class Publisher:
    def __init__(self, url: str = REDIS_URL):
        self.url = url; self.r: Optional[AsyncRedis] = None # type: ignore
    async def connect(self):
        if not HAVE_REDIS: return
        try:
            self.r = AsyncRedis.from_url(self.url, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None
    async def xadd(self, stream: str, obj: Dict[str, Any]):
        if not self.r:
            print(f"[{stream}]", obj); return
        try:
            await self.r.xadd(stream, {"json": json.dumps(obj, ensure_ascii=False)}, maxlen=MAXLEN, approximate=True)  # type: ignore
        except Exception:
            pass

class Deduper:
    def __init__(self, cap=20000):
        self.cap = cap; self._seen: Dict[str,int] = {}
    def seen(self, key: str) -> bool:
        if key in self._seen: return True
        if len(self._seen) >= self.cap:
            # drop oldest half
            for k,_ in sorted(self._seen.items(), key=lambda kv: kv[1])[: self.cap//2]:
                self._seen.pop(k, None)
        self._seen[key] = now_ms()
        return False

# ---------- Canonical events -------------------------------------------------
@dataclass
class SocialEvent:
    ts_ms: int
    source: str   # 'x','reddit','stocktwits'
    id: str
    user: Optional[str]
    text: str
    url: Optional[str]
    symbols: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    lang: Optional[str] = "en"
    likes: Optional[int] = 0
    replies: Optional[int] = 0
    reposts: Optional[int] = 0
    score: Optional[float] = None     # raw model score in [-1,1]
    label: Optional[str] = None       # 'pos','neg','neu'
    conf: Optional[float] = None      # confidence [0,1]
    type: str = "social"

# ---------- Sentiment engine -------------------------------------------------
class SentimentEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vader = None
        self.device = "cpu"
        # Try transformers first (FinBERT or any 3-class)
        if HAVE_TRANSFORMERS:
            mdl = os.getenv("SENT_MODEL", "ProsusAI/finbert")  # change if needed
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(mdl)
                self.model = AutoModelForSequenceClassification.from_pretrained(mdl)
                self.model.eval()
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.model.to(self.device)
            except Exception:
                self.model = None; self.tokenizer = None
        # Then VADER
        if self.model is None and HAVE_VADER:
            try:
                # VADER may require nltk data; attempt lazy init
                self.vader = SentimentIntensityAnalyzer()
            except Exception:
                self.vader = None

    def score(self, text: str) -> Tuple[float, str, float]:
        """
        Returns (score in [-1,1], label, confidence)
        """
        txt = (text or "").strip()
        if not txt:
            return 0.0, "neu", 0.0

        # transformers path
        if self.model is not None and self.tokenizer is not None:
            with torch.no_grad():
                toks = self.tokenizer(txt[:512], truncation=True, return_tensors="pt").to(self.device)  # type: ignore
                logits = self.model(**toks).logits  # type: ignore
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]  # type: ignore
            # Heuristic mapping for FinBERT (neg, neu, pos)
            if probs.shape[-1] == 3:
                neg, neu, pos = float(probs[0]), float(probs[1]), float(probs[2])
                score = pos - neg
                label = "pos" if score > 0.15 else ("neg" if score < -0.15 else "neu")
                conf = max(neg, neu, pos)
                return float(max(-1.0, min(1.0, score))), label, float(conf)
            # Otherwise fall back to argmax ±
            idx = int(probs.argmax())
            label = ["neg","neu","pos"][idx] if len(probs)==3 else str(idx)
            score = (2 * (float(probs[idx])) - 1.0)
            return float(max(-1.0, min(1.0, score))), label, float(probs[idx])

        # VADER path
        if self.vader is not None:
            s = self.vader.polarity_scores(txt)
            score = float(s.get("compound", 0.0))
            label = "pos" if score > 0.25 else ("neg" if score < -0.25 else "neu")
            conf = float(max(abs(score), s.get("pos",0.0), s.get("neg",0.0)))
            return float(score), label, conf

        # Tiny lexicon fallback
        POS = {"moon","soar","beat","bull","rally","upgrade","strong","profit","surprise","buy","breakout","up"}
        NEG = {"dump","crash","bear","downgrade","weak","loss","miss","sell","fraud","scam","down"}
        t = re.findall(r"[a-z]+", txt.lower())
        p = sum(w in POS for w in t); n = sum(w in NEG for w in t)
        score = (p - n) / max(1, (p + n))
        label = "pos" if score > 0.15 else ("neg" if score < -0.15 else "neu")
        conf = min(1.0, abs(score))
        return float(score), label, float(conf)

# ---------- Connectors -------------------------------------------------------
class Connector:
    name: str
    async def run(self, pub: Publisher, dedupe: Deduper, se: SentimentEngine): ...

class TwitterConnector(Connector):
    def __init__(self, query: str = X_QUERY, poll_sec: int = X_POLL_SEC):
        self.name = "x"
        self.query = query
        self.poll_sec = poll_sec
        self._since_id: Optional[str] = None

    async def run(self, pub: Publisher, dedupe: Deduper, se: SentimentEngine):
        if not HAVE_AIOHTTP or not X_BEARER:
            return
        url = "https://api.twitter.com/2/tweets/search/recent"
        params_base = {
            "max_results": "50",
            "tweet.fields": "created_at,public_metrics,lang,entities",
            "expansions": "author_id",
            "user.fields": "username",
            "query": self.query,
        }
        headers = {"Authorization": f"Bearer {X_BEARER}"}
        while True:
            try:
                params = dict(params_base)
                if self._since_id:
                    params["since_id"] = self._since_id
                async with aiohttp.ClientSession() as sess:  # type: ignore
                    async with sess.get(url, headers=headers, params=params, timeout=20) as resp:
                        j = await resp.json(content_type=None)
                data = j.get("data", []) or []
                users = {u["id"]: u for u in j.get("includes", {}).get("users", [])}
                if data:
                    self._since_id = data[0]["id"]
                for t in data:
                    txt = t.get("text","")
                    syms = extract_symbols(txt)
                    tags = extract_hashtags(txt)
                    metrics = t.get("public_metrics") or {}
                    user = users.get(t.get("author_id") or "", {})
                    urlp = f"https://x.com/{user.get('username','i')}/status/{t['id']}"
                    score, label, conf = se.score(txt)
                    ev = SocialEvent(
                        ts_ms=now_ms(), source="x", id=str(t["id"]), user=user.get("username"),
                        text=txt, url=urlp, symbols=syms, hashtags=tags, lang=t.get("lang","en"),
                        likes=metrics.get("like_count"), replies=metrics.get("reply_count"),
                        reposts=metrics.get("retweet_count"), score=score, label=label, conf=conf
                    )
                    key = sha(f"x|{ev.id}")
                    if dedupe.seen(key): 
                        continue
                    await pub.xadd(OUT_RAW, {"type":"social","source":"x","payload":asdict(ev)})
                    await pub.xadd(OUT_POST, asdict(ev))
            except Exception as e:
                await pub.xadd(OUT_RAW, {"type":"error","source":"x","error":str(e),"ts_ms":now_ms()})
            await asyncio.sleep(self.poll_sec)

class RedditConnector(Connector):
    def __init__(self, subreddits: str = REDDIT_SR, poll_sec: int = REDDIT_POLL):
        self.name = "reddit"
        self.sr = subreddits
        self.poll_sec = poll_sec
        self._seen_ids: set[str] = set()

    async def run(self, pub: Publisher, dedupe: Deduper, se: SentimentEngine):
        if not HAVE_AIOHTTP:
            return
        # simple JSON endpoints: https://www.reddit.com/r/<sub>/new.json
        subs = [s.strip() for s in self.sr.split("+") if s.strip()]
        while True:
            try:
                async with aiohttp.ClientSession(headers={"User-Agent":"sent-bot/0.1"}) as sess:  # type: ignore
                    for sub in subs:
                        url = f"https://www.reddit.com/r/{sub}/new.json?limit=50"
                        async with sess.get(url, timeout=20) as resp:
                            j = await resp.json(content_type=None)
                        for c in (j.get("data",{}).get("children") or []):
                            d = c.get("data",{})
                            rid = d.get("id")
                            if not rid or rid in self._seen_ids:
                                continue
                            self._seen_ids.add(rid)
                            txt = (d.get("title","") + " " + (d.get("selftext","") or "")).strip()
                            syms = extract_symbols(txt)
                            tags = extract_hashtags(txt)
                            score, label, conf = se.score(txt)
                            ev = SocialEvent(
                                ts_ms=now_ms(), source="reddit", id=str(rid), user=d.get("author"),
                                text=txt, url=f"https://www.reddit.com{d.get('permalink','')}",
                                symbols=syms, hashtags=tags, lang="en", likes=d.get("score") or 0,
                                replies=d.get("num_comments") or 0, reposts=0,
                                score=score, label=label, conf=conf
                            )
                            key = sha(f"reddit|{ev.id}")
                            if dedupe.seen(key): 
                                continue
                            await pub.xadd(OUT_RAW, {"type":"social","source":"reddit","payload":asdict(ev)})
                            await pub.xadd(OUT_POST, asdict(ev))
            except Exception as e:
                await pub.xadd(OUT_RAW, {"type":"error","source":"reddit","error":str(e),"ts_ms":now_ms()})
            await asyncio.sleep(self.poll_sec)

class StocktwitsConnector(Connector):
    def __init__(self, symbols_csv: str = STWITS_SYMBOLS, poll_sec: int = STWITS_POLL):
        self.name = "stocktwits"
        self.syms = [s.strip().upper() for s in symbols_csv.split(",") if s.strip()]
        self.poll_sec = poll_sec
        self._last_id: Dict[str, int] = {}

    async def run(self, pub: Publisher, dedupe: Deduper, se: SentimentEngine):
        if not HAVE_AIOHTTP:
            return
        base = "https://api.stocktwits.com/api/2/streams/symbol/{}.json?limit=30"
        while True:
            try:
                async with aiohttp.ClientSession() as sess:  # type: ignore
                    for sym in self.syms:
                        url = base.format(sym)
                        if sym in self._last_id:
                            url += f"&since={self._last_id[sym]}"
                        async with sess.get(url, timeout=20) as resp:
                            j = await resp.json(content_type=None)
                        msgs = j.get("messages") or []
                        if msgs:
                            self._last_id[sym] = int(msgs[0].get("id", self._last_id.get(sym, 0)))
                        for m in msgs:
                            mid = str(m.get("id"))
                            txt = m.get("body","")
                            syms = sorted({sym} | set(extract_symbols(txt)))
                            tags = extract_hashtags(txt)
                            user = (m.get("user") or {}).get("username")
                            score, label, conf = se.score(txt)
                            ev = SocialEvent(
                                ts_ms=now_ms(), source="stocktwits", id=mid, user=user,
                                text=txt, url=f"https://stocktwits.com/message/{mid}",
                                symbols=syms, hashtags=tags, lang="en",
                                likes=(m.get("likes") or {}).get("total",0), replies=0, reposts=0,
                                score=score, label=label, conf=conf
                            )
                            key = sha(f"stwits|{ev.id}")
                            if dedupe.seen(key): 
                                continue
                            await pub.xadd(OUT_RAW, {"type":"social","source":"stocktwits","payload":asdict(ev)})
                            await pub.xadd(OUT_POST, asdict(ev))
            except Exception as e:
                await pub.xadd(OUT_RAW, {"type":"error","source":"stocktwits","error":str(e),"ts_ms":now_ms()})
            await asyncio.sleep(self.poll_sec)

# ---------- Aggregator -------------------------------------------------------
class MinuteAggregator:
    """
    Maintains 1-minute aggregates per symbol:
      - mean score, volatility of score, message count, pos/neg fraction
    Emits to features.alt.social_sentiment_agg.
    """
    def __init__(self, pub: Publisher):
        self.pub = pub
        self.bucket: Dict[Tuple[str,int], List[float]] = {}
        self.cnts: Dict[Tuple[str,int], Dict[str,int]] = {}

    @staticmethod
    def _bucket_min(ts_ms: int) -> int:
        return (ts_ms // 60000) * 60000

    async def ingest(self, ev: SocialEvent):
        if not ev.symbols:
            # still aggregate under a NONE bucket if you want — skipping here
            return
        b = self._bucket_min(ev.ts_ms)
        for sym in ev.symbols:
            key = (sym, b)
            self.bucket.setdefault(key, []).append(float(ev.score or 0.0))
            d = self.cnts.setdefault(key, {"n":0, "pos":0, "neg":0, "neu":0})
            d["n"] += 1; d[ev.label or "neu"] += 1

    async def flush_old(self):
        """
        Periodically flush buckets older than the current minute.
        """
        nowb = self._bucket_min(now_ms())
        keys = [k for k in list(self.bucket.keys()) if k[1] < nowb]
        for k in keys:
            sym, tsb = k
            vals = self.bucket.pop(k, [])
            d = self.cnts.pop(k, {"n":0, "pos":0, "neg":0, "neu":0})
            if not vals: 
                continue
            mean = statistics.fmean(vals)
            vol  = statistics.pstdev(vals) if len(vals)>1 else 0.0
            agg = {
                "ts_ms": tsb,
                "symbol": sym,
                "n": d["n"],
                "pos_frac": d["pos"]/max(1,d["n"]),
                "neg_frac": d["neg"]/max(1,d["n"]),
                "neu_frac": d["neu"]/max(1,d["n"]),
                "score_mean": round(mean,6),
                "score_vol": round(vol,6),
                "source": "social_agg"
            }
            await self.pub.xadd(OUT_AGG, agg)

# ---------- Orchestrator -----------------------------------------------------
class SocialSentimentService:
    def __init__(self):
        self.pub = Publisher()
        self.dedupe = Deduper()
        self.se = SentimentEngine()
        self.agg = MinuteAggregator(self.pub)
        self.connectors: List[Connector] = []
        # enable defaults
        if X_BEARER and HAVE_AIOHTTP:
            self.connectors.append(TwitterConnector())
        if HAVE_AIOHTTP:
            self.connectors.append(RedditConnector())
            self.connectors.append(StocktwitsConnector())

    async def run(self):
        await self.pub.connect()
        tasks = [asyncio.create_task(self._wrap_connector(c)) for c in self.connectors]
        tasks.append(asyncio.create_task(self._flusher()))
        await asyncio.gather(*tasks)

    async def _wrap_connector(self, c: Connector):
        while True:
            try:
                await c.run(self.pub, self.dedupe, self.se)
            except Exception as e:
                await self.pub.xadd(OUT_RAW, {"type":"error","source":getattr(c, 'name','?'),"error":str(e),"ts_ms":now_ms()})
            await asyncio.sleep(3)

    async def _flusher(self):
        while True:
            try:
                await self.agg.flush_old()
            except Exception:
                pass
            await asyncio.sleep(5)

# ---------- CLI --------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("social_sentiment")
    ap.add_argument("--no-x", action="store_true")
    ap.add_argument("--no-reddit", action="store_true")
    ap.add_argument("--no-stwits", action="store_true")
    ap.add_argument("--query", type=str, default=X_QUERY)
    ap.add_argument("--stwits", type=str, default=STWITS_SYMBOLS)
    args = ap.parse_args()

    svc = SocialSentimentService()
    # override connectors
    svc.connectors = []
    if not args.no_x and X_BEARER and HAVE_AIOHTTP:
        svc.connectors.append(TwitterConnector(query=args.query))
    if not args.no_reddit and HAVE_AIOHTTP:
        svc.connectors.append(RedditConnector())
    if not args.no_stwits and HAVE_AIOHTTP:
        svc.connectors.append(StocktwitsConnector(symbols_csv=args.stwits))

    async def _run():
        await svc.run()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    _cli()