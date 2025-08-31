# backend/ai/audio_nlp/earnings_calls_sentiment.py
from __future__ import annotations

"""
Earnings Calls Sentiment
------------------------
Lightweight, dependency-optional sentiment analysis tailored for earnings call transcripts.

Backends (in this order):
  1) FinBERT via transformers (ProsusAI/finbert)
  2) VADER (if installed)
  3) Built-in finance lexicon (fallback)

Features
- Sentence segmentation & speaker detection ("Name: …")
- Section tagging: "prepared_remarks" vs "qa"
- Per-sentence scores + document aggregates
- Top positive/negative snippets for UI
- CLI:
    python -m backend.ai.audio_nlp.earnings_calls_sentiment --file transcript.txt --json out.json

Optional installs for better accuracy:
    pip install transformers torch --upgrade
    pip install vaderSentiment
"""

import json
import math
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------- Data Models ----------------------------

@dataclass
class SentSpan:
    start: int
    end: int
    text: str
    score: float
    label: str
    speaker: Optional[str] = None
    section: Optional[str] = None     # "prepared_remarks" | "qa" | None
    prob: Optional[float] = None      # backend confidence if available

@dataclass
class DocSentiment:
    method: str
    avg: float
    pos: float
    neg: float
    neu: float
    count: int
    prepared_avg: Optional[float] = None
    qa_avg: Optional[float] = None
    top_pos: List[SentSpan] = None # type: ignore
    top_neg: List[SentSpan] = None # type: ignore

# ------------------------- Backend Selection -------------------------

class _Backend:
    name = "fallback"
    def score_sentence(self, text: str) -> Tuple[float, str, Optional[float]]:
        """Return (score in [-1,1], label, prob)."""
        raise NotImplementedError

class _FinBERT(_Backend):
    name = "finbert"
    def __init__(self):
        from transformers import pipeline # type: ignore
        self.pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    def score_sentence(self, text: str) -> Tuple[float, str, Optional[float]]:
        out = self.pipe(text[:512])[0]  # truncate long sentences for speed
        label = out["label"].lower()    # "positive" | "neutral" | "negative"
        prob = float(out.get("score", 0.0))
        if label == "positive":
            score = prob
        elif label == "negative":
            score = -prob
        else:
            score = 0.0
        return float(max(-1.0, min(1.0, score))), label, prob

class _Vader(_Backend):
    name = "vader"
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore
        self.vader = SentimentIntensityAnalyzer()
    def score_sentence(self, text: str) -> Tuple[float, str, Optional[float]]:
        s = self.vader.polarity_scores(text)
        score = float(s.get("compound", 0.0))
        label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
        return max(-1.0, min(1.0, score)), label, None

class _Lexicon(_Backend):
    name = "lexicon"

    POS = {
        "record","strong","strength","exceeded","beat","beats",
        "growth","improve","improving","improved","upbeat",
        "resilient","tailwind","robust","optimistic","opportunity",
        "profitable","profitability","accelerate","acceleration",
        "guidance_raised","raise","raised","increase","increased"
    }
    NEG = {
        "headwind","soft","softer","decline","declined","miss",
        "missed","challenge","challenging","uncertain","uncertainty",
        "inflationary","pressure","pressures","loss","impairment",
        "lower","decrease","decreased","suspend","suspended",
        "guidance_cut","cut","cuts","restructuring","charges",
        "shortfall","slowdown","slow","weak","weaker"
    }
    BOOST_POS = {"very","significantly","materially","meaningfully","solidly"}
    BOOST_NEG = {"severely","sharply","materially","significantly","meaningfully"}
    NEGATORS = {"not","no","never","without","lack","lacking","less"}

    def _tokenize(self, text: str) -> List[str]:
        text = unicodedata.normalize("NFKC", text.lower())
        text = re.sub(r"[^a-z0-9\s\-\_\.]", " ", text)
        return re.findall(r"[a-z0-9_]+", text)

    def score_sentence(self, text: str) -> Tuple[float, str, Optional[float]]:
        toks = self._tokenize(text)
        if not toks:
            return 0.0, "neutral", None
        score = 0.0
        i = 0
        while i < len(toks):
            t = toks[i]
            val = 0.0
            if t in self.POS:
                val = +1.0
                if i+1 < len(toks) and f"{t}_{toks[i+1]}" in self.POS:
                    val = +1.2; i += 1
            elif t in self.NEG:
                val = -1.0
                if i+1 < len(toks) and f"{t}_{toks[i+1]}" in self.NEG:
                    val = -1.2; i += 1
            if val != 0.0:
                window = toks[max(0, i-2):i]
                if any(w in self.NEGATORS for w in window):
                    val = -val
                if any(w in self.BOOST_POS for w in window) and val > 0:
                    val *= 1.25
                if any(w in self.BOOST_NEG for w in window) and val < 0:
                    val *= 1.25
                score += val
            i += 1
        norm = math.sqrt(len(toks))
        score = score / (norm if norm > 0 else 1.0)
        score = max(-2.0, min(2.0, score)) / 2.0  # to [-1,1]
        label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
        return float(score), label, None

# --------------------------- Main Analyzer ---------------------------

_SECTION_QA_RE = re.compile(r"\b(q&amp;a|q and a|question\s*and\s*answer|qa session)\b", re.I)
_SECTION_PREP_RE = re.compile(r"\bprepared\s+remarks?\b", re.I)
_SPEAKER_RE = re.compile(r"^\s*([A-Z][A-Za-z\.\s\-]{1,40}):\s+", re.M)

def _sent_tokenize(text: str) -> List[Tuple[int, int, str]]:
    """Simple sentence splitter robust to all-caps speaker lines."""
    spans: List[Tuple[int, int, str]] = []
    normalized = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    for m in re.finditer(r"[^.!?]+[.!?]+|[^.!?]+$", normalized):
        s, e = m.start(), m.end()
        chunk = normalized[s:e].strip()
        if chunk:
            spans.append((s, e, chunk))
    return spans

def _guess_section(text_before: str) -> Optional[str]:
    tail = text_before[-2000:].lower()
    if _SECTION_QA_RE.search(tail): return "qa"
    if _SECTION_PREP_RE.search(tail): return "prepared_remarks"
    return None

def _guess_speaker(prev_text: str) -> Optional[str]:
    tail = prev_text[-400:]
    matches = list(_SPEAKER_RE.finditer(tail))
    return matches[-1].group(1).strip() if matches else None

class EarningsCallSentiment:
    def __init__(self, prefer: Optional[str] = None):
        """
        prefer: "finbert" | "vader" | "lexicon" | None(auto)
        """
        self.backend: _Backend = self._auto_backend(prefer)

    def _auto_backend(self, prefer: Optional[str]) -> _Backend:
        order = []
        if prefer in ("finbert","vader","lexicon"):
            order = [prefer]
        order += ["finbert","vader","lexicon"]
        for name in order:
            try:
                if name == "finbert": return _FinBERT()
                if name == "vader":   return _Vader()
                return _Lexicon()
            except Exception:
                continue
        return _Lexicon()

    def analyze(self, text: str, *, top_k: int = 5) -> Tuple[DocSentiment, List[SentSpan]]:
        """Analyze a full transcript. Returns (DocSentiment, spans)."""
        if not text or not text.strip():
            return DocSentiment(method=self.backend.name, avg=0.0, pos=0.0, neg=0.0, neu=1.0, count=0), []
        sents = _sent_tokenize(text)
        spans: List[SentSpan] = []
        prepared_scores, qa_scores, all_scores = [], [], []

        for (s, e, sent) in sents:
            prev = text[:s]
            section = _guess_section(prev)
            speaker = _guess_speaker(prev)
            score, label, prob = self.backend.score_sentence(sent)
            span = SentSpan(start=s, end=e, text=sent, score=score, label=label, prob=prob,
                            speaker=speaker, section=section)
            spans.append(span)
            all_scores.append(score)
            if section == "qa": qa_scores.append(score)
            elif section == "prepared_remarks": prepared_scores.append(score)

        agg = _aggregate(all_scores)
        prep_avg = _safe_mean(prepared_scores)
        qa_avg = _safe_mean(qa_scores)

        top_pos = sorted([sp for sp in spans if sp.score > 0], key=lambda x: x.score, reverse=True)[:top_k]
        top_neg = sorted([sp for sp in spans if sp.score < 0], key=lambda x: x.score)[:top_k]

        doc = DocSentiment(
            method=self.backend.name,
            avg=agg["avg"], pos=agg["pos"], neg=agg["neg"], neu=agg["neu"],
            count=len(all_scores),
            prepared_avg=prep_avg, qa_avg=qa_avg,
            top_pos=top_pos, top_neg=top_neg
        )
        return doc, spans

    def to_json(self, doc: DocSentiment, spans: List[SentSpan]) -> str:
        payload = {
            "doc": {
                **asdict(doc),
                "top_pos": [asdict(s) for s in (doc.top_pos or [])],
                "top_neg": [asdict(s) for s in (doc.top_neg or [])],
            },
            "spans": [asdict(s) for s in spans],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

# --------------------------- Aggregation -----------------------------

def _safe_mean(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    if not xs: return None
    return float(sum(xs) / len(xs))

def _aggregate(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"avg": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
    avg = float(sum(scores) / len(scores))
    pos = sum(1 for s in scores if s > 0.05) / len(scores)
    neg = sum(1 for s in scores if s < -0.05) / len(scores)
    neu = 1.0 - pos - neg
    return {"avg": avg, "pos": pos, "neg": neg, "neu": neu}

# ------------------------------ CLI ---------------------------------

def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _main():
    import argparse
    p = argparse.ArgumentParser(description="Earnings calls sentiment (FinBERT/VADER/lexicon)")
    p.add_argument("--file", type=str, required=True, help="Transcript .txt file")
    p.add_argument("--json", type=str, default=None, help="Write JSON output to this path")
    p.add_argument("--backend", type=str, choices=["finbert","vader","lexicon"], default=None)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    text = _read_file(args.file)
    eng = EarningsCallSentiment(prefer=args.backend)
    doc, spans = eng.analyze(text, top_k=args.topk)
    out = eng.to_json(doc, spans)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)

if __name__ == "__main__":
    _main()