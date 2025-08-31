# backend/ai/analyst_agent.py
from __future__ import annotations

"""
Analyst Agent
-------------
Consumes normalized news/events, produces trade ideas with confidence,
and publishes them to ai.signals / feature store.

- Input (Redis Stream: ANALYST_INBOX, default: "news.events")
  Event JSON (minimal):
  {
    "ts_ms": 1724726400123,
    "symbol": "TSLA" | ["TSLA","NVDA"] | null,
    "title": "Tesla beats earnings ...",
    "body": "Longer text...",
    "source": "moneycontrol" | "yahoo" | "rss",
    "url": "https://...",
    "region": "US" | "IN" | null,
    "meta": {...}
  }

- Output (Redis Stream: AI_SIGNALS, default: "ai.signals")
  TradeIdea JSON:
  {
    "id": "sha1(...)", "ts_ms": 1724726400456,
    "symbol": "TSLA",
    "direction": "long" | "short" | "none",
    "horizon": "intraday" | "swing" | "event",
    "confidence": 0.0..1.0,
    "score": float,                 # signed score (-1..+1)
    "rationale": "…",
    "features": {"sentiment": 0.63, "topic:earnings": 0.88, ...},
    "source": "analyst_agent",
    "links": [{"title": "...", "url": "..."}],
    "raw_ref": {"event_id": "..."}  # optional
  }

Config (optional): config/analyst.yaml
  model:
    use_transformers: true
    sentiment_model: "ProsusAI/finbert"
    zero_shot_model: "facebook/bart-large-mnli"
  thresholds:
    min_conf: 0.55
    bullish_bias_bp: 0
  topics:
    candidates: ["earnings beat","earnings miss","guidance raise","guidance cut","regulatory","merger","downgrade","upgrade","litigation","macro","supply chain","product launch"]
"""

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# -------- Optional deps ------------------------------------------------------
USE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    AsyncRedis = None  # type: ignore
    USE_REDIS = False

try:
    import yaml  # PyYAML
except Exception:
    yaml = None  # type: ignore

# Transformers are optional
try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None  # type: ignore

# -------- Env / defaults -----------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ANALYST_INBOX = os.getenv("ANALYST_INBOX", "news.events")
AI_SIGNALS = os.getenv("AI_SIGNALS", "ai.signals")
FEATURES_STREAM = os.getenv("FEATURES_STREAM", "features.signals")
MAXLEN = int(os.getenv("ANALYST_MAXLEN", "20000"))
CFG_PATH = os.getenv("ANALYST_CONFIG", "config/analyst.yaml")

# -------- Small utils --------------------------------------------------------
def now_ms() -> int:
    return int(time.time() * 1000)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()

def as_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(i) for i in x]

TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")

# -------- Config -------------------------------------------------------------
DEFAULT_CFG = {
    "model": {
        "use_transformers": True,
        "sentiment_model": "ProsusAI/finbert",
        "zero_shot_model": "facebook/bart-large-mnli",
    },
    "thresholds": {
        "min_conf": 0.55,
        "bullish_bias_bp": 0  # optional tiny bias if you want
    },
    "topics": {
        "candidates": [
            "earnings beat","earnings miss","guidance raise","guidance cut",
            "regulatory","merger","downgrade","upgrade","litigation",
            "macro","supply chain","product launch"
        ]
    },
    "mapping": {
        # optional keyword → symbol hints (for Indian market brand names, etc.)
        # "Tata Motors": "TATAMOTORS"
    }
}

def load_cfg(path: str = CFG_PATH) -> Dict[str, Any]:
    if yaml and os.path.exists(path):
        try:
            with open(path, "r") as f:
                doc = yaml.safe_load(f) or {}
            # shallow merge
            cfg = DEFAULT_CFG.copy()
            for k, v in (doc or {}).items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
            return cfg
        except Exception:
            return DEFAULT_CFG
    return DEFAULT_CFG

# -------- Models -------------------------------------------------------------
@dataclass
class NewsEvent:
    ts_ms: int
    title: str
    body: str
    symbol: Optional[Union[str, List[str]]]
    source: Optional[str] = None
    url: Optional[str] = None
    region: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    event_id: Optional[str] = None

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "NewsEvent":
        return NewsEvent(
            ts_ms=int(obj.get("ts_ms") or obj.get("ts") or now_ms()),
            title=str(obj.get("title") or ""),
            body=str(obj.get("body") or obj.get("text") or ""),
            symbol=obj.get("symbol"),
            source=obj.get("source"),
            url=obj.get("url"),
            region=obj.get("region"),
            meta=obj.get("meta"),
            event_id=obj.get("id") or obj.get("event_id")
        )

@dataclass
class TradeIdea:
    id: str
    ts_ms: int
    symbol: str
    direction: str           # long|short|none
    horizon: str             # intraday|swing|event
    confidence: float        # 0..1
    score: float             # -1..+1
    rationale: str
    features: Dict[str, Any]
    links: List[Dict[str, str]]
    raw_ref: Dict[str, Any]

# -------- LLM / NLP adapters -------------------------------------------------
class SentimentAdapter:
    def __init__(self, model_name: str | None, enabled: bool):
        self.enabled = enabled and pipeline is not None
        self.model_name = model_name or "ProsusAI/finbert"
        self.pipe = None
        if self.enabled:
            try:
                self.pipe = pipeline("sentiment-analysis", model=self.model_name) # type: ignore
            except Exception:
                self.pipe = None
                self.enabled = False

    def score(self, text: str) -> float:
        """
        Returns sentiment in [-1, +1]. If disabled/unavailable, uses heuristic.
        """
        if self.pipe is not None:
            try:
                res = self.pipe(text[:512])[0]
                label = (res["label"] or "").lower()
                s = float(res.get("score", 0.0))
                if "neg" in label: return -s
                if "pos" in label: return +s
                # neutral → 0
                return 0.0
            except Exception:
                pass
        # Heuristic fallback
        pos = sum(1 for w in ("beat","surge","upgrade","raise","strong","record","approve") if w in text.lower())
        neg = sum(1 for w in ("miss","downgrade","cut","probe","delay","ban","recall") if w in text.lower())
        if pos == neg == 0: return 0.0
        return (pos - neg) / max(3.0, pos + neg)

class ZeroShotTopics:
    def __init__(self, model_name: str | None, enabled: bool, candidates: List[str]):
        self.enabled = enabled and pipeline is not None
        self.model_name = model_name or "facebook/bart-large-mnli"
        self.candidates = candidates
        self.pipe = None
        if self.enabled:
            try:
                self.pipe = pipeline("zero-shot-classification", model=self.model_name) # type: ignore
            except Exception:
                self.pipe = None
                self.enabled = False

    def classify(self, text: str) -> Dict[str, float]:
        if self.pipe is not None:
            try:
                res = self.pipe(text[:512], candidate_labels=self.candidates, multi_label=True)
                labels = res.get("labels", [])
                scores = res.get("scores", [])
                return {str(l): float(s) for l, s in zip(labels, scores)}
            except Exception:
                pass
        # Heuristic fallback: keyword hit ratio
        text_l = text.lower()
        out: Dict[str, float] = {}
        for c in self.candidates:
            toks = [t for t in re.split(r"\W+", c.lower()) if t]
            if not toks:
                out[c] = 0.0
                continue
            hits = sum(1 for t in toks if t in text_l)
            out[c] = hits / len(toks)
        return out

# -------- Feature sink (optional) -------------------------------------------
class FeatureSink:
    def __init__(self, r: Optional[AsyncRedis], stream: str = FEATURES_STREAM): # type: ignore
        self.r = r
        self.stream = stream

    async def emit(self, symbol: str, features: Dict[str, Any]):
        payload = {
            "ts_ms": now_ms(),
            "symbol": symbol,
            "features": features,
            "source": "analyst_agent",
        }
        if self.r:
            try:
                await self.r.xadd(self.stream, {"json": json.dumps(payload)}, maxlen=MAXLEN, approximate=True)
                return
            except Exception:
                pass
        # fallback: print (or hook into your local file logger)
        print("[feature_sink]", json.dumps(payload)[:500])

# -------- Core agent ---------------------------------------------------------
class AnalystAgent:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or load_cfg()
        mcfg = self.cfg.get("model", {})
        tcfg = self.cfg.get("topics", {})
        self.min_conf = float(self.cfg.get("thresholds", {}).get("min_conf", 0.55))

        self.sent = SentimentAdapter(mcfg.get("sentiment_model"), enabled=bool(mcfg.get("use_transformers", True)))
        self.zs = ZeroShotTopics(mcfg.get("zero_shot_model"), enabled=bool(mcfg.get("use_transformers", True)),
                                 candidates=tcfg.get("candidates", []))
        self.mapping = self.cfg.get("mapping", {})

        self.r: Optional[AsyncRedis] = None # type: ignore
        self.feature_sink: Optional[FeatureSink] = None

    async def connect(self):
        if USE_REDIS:
            try:
                self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
                await self.r.ping() # type: ignore
            except Exception:
                self.r = None
        self.feature_sink = FeatureSink(self.r)

    def _extract_symbols(self, e: NewsEvent) -> List[str]:
        # explicit
        sym = as_list(e.symbol)
        out: List[str] = []
        out.extend(sym)
        # mapping hints
        for k, v in self.mapping.items():
            if k.lower() in (e.title + " " + e.body).lower():
                out.append(str(v))
        # regex uppercase tickers in title
        for m in TICKER_RE.findall(e.title):
            out.append(m)
        return sorted(set(s.upper() for s in out if s))

    def _idea_from_scores(self, symbol: str, s_sent: float, topics: Dict[str, float], e: NewsEvent) -> TradeIdea:
        # simple signed score: sentiment plus topic influence
        topic_boost = 0.0
        if topics.get("earnings beat", 0) > 0.5 or topics.get("upgrade", 0) > 0.5 or topics.get("guidance raise", 0) > 0.5:
            topic_boost += 0.25
        if topics.get("earnings miss", 0) > 0.5 or topics.get("downgrade", 0) > 0.5 or topics.get("guidance cut", 0) > 0.5:
            topic_boost -= 0.25
        if topics.get("merger", 0) > 0.5 or topics.get("product launch", 0) > 0.5:
            topic_boost += 0.10
        if topics.get("litigation", 0) > 0.5 or topics.get("regulatory", 0) > 0.5:
            topic_boost -= 0.10

        score = max(-1.0, min(1.0, s_sent + topic_boost))
        direction = "long" if score > 0.05 else "short" if score < -0.05 else "none"
        # horizon heuristic
        horizon = "event" if (topics.get("earnings beat",0) > 0.5 or topics.get("earnings miss",0) > 0.5) else "swing"
        conf = min(1.0, abs(score))

        rationale = []
        if score >= 0.2: rationale.append("Positive tone and favorable event topics.")
        elif score <= -0.2: rationale.append("Negative tone and adverse event topics.")
        else: rationale.append("Mixed signals; low conviction.")
        top_topics = [k for k, v in sorted(topics.items(), key=lambda kv: kv[1], reverse=True)[:3] if v > 0.25]
        if top_topics: rationale.append("Top topics: " + ", ".join(top_topics))
        rationale.append(f"Sentiment={s_sent:+.2f}, TopicBoost={topic_boost:+.2f}, Score={score:+.2f}")

        idea = TradeIdea(
            id=sha1(f"{symbol}|{e.ts_ms}|{e.title[:64]}"),
            ts_ms=e.ts_ms,
            symbol=symbol,
            direction=direction,
            horizon=horizon,
            confidence=conf,
            score=score,
            rationale=" ".join(rationale),
            features={"sentiment": s_sent, **{f"topic:{k}": float(v) for k, v in topics.items()}},
            links=([{"title": e.title[:120], "url": e.url}] if e.url else [{"title": e.title[:120], "url": ""}]),
            raw_ref={"event_id": e.event_id or "", "source": e.source or "", "region": e.region or ""},
        )
        return idea

    async def analyze_event(self, e: NewsEvent) -> List[TradeIdea]:
        text = (e.title or "") + "\n" + (e.body or "")
        s_sent = self.sent.score(text)
        topics = self.zs.classify(text)
        syms = self._extract_symbols(e) or as_list(e.symbol)
        ideas: List[TradeIdea] = []
        for sym in syms:
            idea = self._idea_from_scores(sym, s_sent, topics, e)
            ideas.append(idea)
        return ideas

    async def publish_idea(self, idea: TradeIdea):
        payload = {
            "id": idea.id,
            "ts_ms": idea.ts_ms or now_ms(),
            "symbol": idea.symbol,
            "direction": idea.direction,
            "horizon": idea.horizon,
            "confidence": round(float(idea.confidence), 4),
            "score": round(float(idea.score), 4),
            "rationale": idea.rationale,
            "features": idea.features,
            "links": idea.links,
            "source": "analyst_agent",
            "raw_ref": idea.raw_ref,
        }
        # emit to feature sink for backtests
        if self.feature_sink:
            await self.feature_sink.emit(idea.symbol, idea.features)
        # stream
        if self.r:
            try:
                await self.r.xadd(AI_SIGNALS, {"json": json.dumps(payload)}, maxlen=MAXLEN, approximate=True)
            except Exception:
                pass
        else:
            print("[ai.signals]", json.dumps(payload)[:800])

    async def run_forever(self):
        if self.r is None:
            await self.connect()
        assert self.r is not None or not USE_REDIS, "Redis not available and required for worker mode."
        last_id = "$"
        # Warm read to keep cache rolling if publisher already pushed before we connected
        try:
            if self.r:
                await self.r.xread({ANALYST_INBOX: last_id}, count=1, block=10)
        except Exception:
            pass

        while True:
            if self.r:
                try:
                    resp = await self.r.xread({ANALYST_INBOX: last_id}, count=200, block=5000)
                    if not resp:
                        await asyncio.sleep(0.2)
                        continue
                    _, entries = resp[0]
                    for _id, fields in entries:
                        last_id = _id
                        try:
                            obj = json.loads(fields.get("json", "{}"))
                            ev = NewsEvent.from_json(obj)
                        except Exception:
                            continue
                        ideas = await self.analyze_event(ev)
                        for idea in ideas:
                            # only push actionable ones or keep all? keep all; UI can filter on confidence
                            await self.publish_idea(idea)
                except Exception as e:
                    # backoff on transient errors
                    print("[analyst_agent] error:", str(e))
                    await asyncio.sleep(1.0)
            else:
                # no redis: idle loop (you can call analyze_event() programmatically)
                await asyncio.sleep(1.0)

# -------- CLI entry ----------------------------------------------------------
async def _amain():
    agent = AnalystAgent()
    await agent.connect()
    print("[analyst_agent] started. inbox=", ANALYST_INBOX, " out=", AI_SIGNALS, " redis=", bool(agent.r))
    await agent.run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass