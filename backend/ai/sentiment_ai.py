# backend/ai/nlp/sentiment_ai.py
from __future__ import annotations

"""
Sentiment AI
------------
Unified sentiment scorer for finance/news/feeds.

Priority of engines (auto-detect):
1) HuggingFace pipeline (finance model preferred, else generic)
2) VADER (vaderSentiment)
3) Lexicon + rules fallback (ships with this file)

Outputs: score in [-1,+1], label, confidence [0,1], entities, hashtags/tickers.

Env (optional bus):
  REDIS_HOST, REDIS_PORT, SENTIMENT_OUT_STREAM (default "nlp.sentiment")

Example:
    sa = SentimentAI()
    out = sa.score("AAPL beats on EPS but guides lower; supply issues linger.")
    print(out)

Batch:
    rows = [{"id":"n1","text":"...","source":"yahoo"}, ...]
    scored = sa.batch(rows, emit=True)

CLI:
    python -m backend.ai.nlp.sentiment_ai --in texts.json --out scored.json
"""

import os, re, json, math, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# -------- Optional libs (graceful fallbacks) --------
try:
    from transformers import pipeline  # type: ignore
    _has_hf = True
except Exception:
    _has_hf = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _has_vader = True
except Exception:
    _has_vader = False

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
OUT_STREAM = os.getenv("SENTIMENT_OUT_STREAM", "nlp.sentiment")


# ---------------- Models & utils ----------------

@dataclass
class SentimentResult:
    text: str
    score: float            # [-1, +1]
    label: str              # "neg"|"neu"|"pos"
    confidence: float       # [0,1]
    entities: List[str]
    tickers: List[str]
    meta: Dict[str, Any]

_FIN_POS = {
    "beat", "beats", "outperform", "raise", "raised", "upgrade", "bullish", "acceleration",
    "record", "surge", "rally", "strong", "resilient", "expansion", "buyback", "dividend hike",
    "guides higher", "exceeds", "above consensus", "surprise to upside", "profit", "profitable",
}
_FIN_NEG = {
    "miss", "cuts", "cut", "downgrade", "bearish", "slowdown", "weak", "decline", "slump",
    "warns", "warning", "layoffs", "lawsuit", "probe", "guides lower", "shortfall", "deficit",
    "default", "bankruptcy", "impairment", "write-down", "fraud", "restatement"
}
_NEGATORS = {"no", "not", "isn't", "wasn't", "aren't", "never", "without", "hardly"}
_INTENS = {"very": 1.2, "extremely": 1.5, "significantly": 1.3, "slightly": 0.6, "mildly": 0.7}
_TICKER_RE = re.compile(r"(?<![A-Z0-9])\$?[A-Z]{1,5}(?:\.[A-Z])?(?![a-z])")
_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")

def _clean(txt: str) -> str:
    txt = _URL_RE.sub("", txt or "")
    return _WS_RE.sub(" ", txt).strip()

def _clip(x: float, lo=-1.0, hi=1.0) -> float:
    return max(lo, min(hi, x))

def _to_label(score: float) -> str:
    if score > 0.15: return "pos"
    if score < -0.15: return "neg"
    return "neu"

def _confidence_from_margin(score: float) -> float:
    # farther from 0 => higher confidence
    return round(min(1.0, 0.5 + 0.5 * abs(score)), 3)

def _extract_entities(text: str) -> Tuple[List[str], List[str]]:
    # very light entity/ticker capture
    tickers = []
    for m in _TICKER_RE.finditer(text):
        t = m.group(0)
        if t.startswith("$"): t = t[1:]
        if 1 <= len(t) <= 6:
            tickers.append(t)
    hashtags = [m.group(1) for m in _HASHTAG_RE.finditer(text)]
    ents = list(set(hashtags))
    return list(sorted(set(tickers))), ents

# ---------------- Engines ----------------

class _HFEngine:
    def __init__(self):
        # Prefer finance-specific model when available; fall back to SST-2
        self.pipe = None
        if _has_hf:
            try:
                self.pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
            except Exception:
                self.pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True)

    def ok(self) -> bool:
        return self.pipe is not None

    def score(self, text: str) -> Tuple[float, float]:
        # returns (score in [-1,1], confidence)
        out = self.pipe(_clean(text)) # type: ignore
        # unify output
        if isinstance(out, list) and out and isinstance(out[0], dict) and "label" in out[0]:
            lab = out[0]["label"].upper()
            sc = float(out[0].get("score", 0.5))
            sgn = +1.0 if ("POS" in lab or "bull" in lab.lower()) else -1.0 if ("NEG" in lab or "bear" in lab.lower()) else 0.0
            score = _clip((2 * sc - 1.0) * sgn)
            conf = _confidence_from_margin(score)
            return score, conf
        # some HF pipelines return list of dicts probs for all labels
        try:
            probs = {d["label"].upper(): float(d["score"]) for d in out}
            pos = max(probs.get("POSITIVE", 0.0), probs.get("POS", 0.0), probs.get("LABEL_1", 0.0))
            neg = max(probs.get("NEGATIVE", 0.0), probs.get("NEG", 0.0), probs.get("LABEL_0", 0.0))
            score = _clip(pos - neg)
            conf = _confidence_from_margin(score)
            return score, conf
        except Exception:
            return 0.0, 0.0

class _VaderEngine:
    def __init__(self):
        self.v = SentimentIntensityAnalyzer() if _has_vader else None

    def ok(self) -> bool:
        return self.v is not None

    def score(self, text: str) -> Tuple[float, float]:
        s = self.v.polarity_scores(_clean(text)) # type: ignore
        comp = _clip(s.get("compound", 0.0))
        # VADER compound already in [-1,1]; confidence ~ |compound|
        return comp, _confidence_from_margin(comp)

class _LexEngine:
    def __init__(self):
        self.pos = _FIN_POS
        self.neg = _FIN_NEG
        self.intens = _INTENS
        self.negators = _NEGATORS

    def ok(self) -> bool:
        return True

    def score(self, text: str) -> Tuple[float, float]:
        toks = re.findall(r"[A-Za-z']+", _clean(text).lower())
        score = 0.0
        i = 0
        while i < len(toks):
            w = toks[i]
            mult = 1.0
            # lookback for intensifiers and negators within 2 tokens
            for j in range(max(0, i - 2), i):
                if toks[j] in self.intens:
                    mult *= self.intens[toks[j]]
                if toks[j] in self.negators:
                    mult *= -1.0
            if w in self.pos: score += 1.0 * mult
            if w in self.neg: score -= 1.0 * mult
            i += 1
        if len(toks) > 0:
            score = score / math.sqrt(len(toks))  # length dampening
        score = _clip(math.tanh(score / 3.0))
        return score, _confidence_from_margin(score)

# ---------------- Main API ----------------

class SentimentAI:
    def __init__(self, prefer: Optional[str] = None):
        """
        prefer: "hf" | "vader" | "lex" | None (auto)
        """
        self.hf = _HFEngine() if _has_hf else None
        self.vd = _VaderEngine() if _has_vader else None
        self.lx = _LexEngine()

        order = []
        if prefer == "hf" and self.hf and self.hf.ok(): order = ["hf", "vader", "lex"]
        elif prefer == "vader" and self.vd and self.vd.ok(): order = ["vader", "hf", "lex"]
        elif prefer == "lex": order = ["lex", "vader", "hf"]
        else:
            # auto: HF if available, else VADER, else lex
            if self.hf and self.hf.ok(): order.append("hf")
            if self.vd and self.vd.ok(): order.append("vader")
            order.append("lex")
        self.order = order

        # optional Redis
        self._r = None
        if _redis is not None:
            try:
                self._r = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            except Exception:
                self._r = None

    def score(self, text: str, *, source: str = "", meta: Optional[Dict[str, Any]] = None) -> SentimentResult:
        text = text or ""
        tickers, ents = _extract_entities(text)
        score, conf, engine = 0.0, 0.0, "none"
        for name in self.order:
            try:
                if name == "hf" and self.hf and self.hf.ok():
                    score, conf = self.hf.score(text); engine = "hf"
                    break
                if name == "vader" and self.vd and self.vd.ok():
                    score, conf = self.vd.score(text); engine = "vader"
                    break
                if name == "lex":
                    score, conf = self.lx.score(text); engine = "lex"
                    break
            except Exception:
                continue
        label = _to_label(score)
        res = SentimentResult(
            text=text,
            score=float(score),
            label=label,
            confidence=float(conf),
            entities=ents,
            tickers=tickers,
            meta={"source": source, "engine": engine, **(meta or {})}
        )
        return res

    def batch(self, rows: List[Dict[str, Any]], *, emit: bool = False, stream: str = OUT_STREAM) -> List[Dict[str, Any]]:
        """
        rows: [{"id":..., "text":..., "source":"yahoo"|... , ...}, ...]
        Returns list of results dicts. If emit=True, publishes to Redis stream/bus.
        """
        out: List[Dict[str, Any]] = []
        ts = int(time.time() * 1000)
        for r in rows:
            txt = r.get("text") or r.get("headline") or ""
            res = self.score(txt, source=r.get("source",""), meta={"id": r.get("id"), "ts": r.get("ts")})
            payload = {
                "id": r.get("id"),
                "ts_ms": r.get("ts") or ts,
                "text": res.text,
                "score": res.score,
                "label": res.label,
                "confidence": res.confidence,
                "tickers": res.tickers,
                "entities": res.entities,
                "meta": res.meta
            }
            out.append(payload)
            if emit:
                try:
                    publish_stream(stream, payload)
                except Exception:
                    pass
        return out

# ---------------- CLI ----------------

def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _main():
    import argparse
    p = argparse.ArgumentParser(description="Sentiment AI (finance/news aware)")
    p.add_argument("--in", dest="inp", required=True, help="Input JSON: [{'id':..., 'text':..., 'source':...}, ...]")
    p.add_argument("--out", dest="out", required=False, help="Output JSON")
    p.add_argument("--prefer", dest="prefer", default=None, choices=[None,"hf","vader","lex"])
    p.add_argument("--emit", dest="emit", action="store_true")
    args = p.parse_args()

    sa = SentimentAI(prefer=args.prefer)
    rows = _load_json(args.inp)
    out = sa.batch(rows, emit=args.emit)
    if args.out:
        _save_json(args.out, out)
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":  # pragma: no cover
    _main()