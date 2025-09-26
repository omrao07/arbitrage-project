# news-intel/models/ner_model.py
"""
NER Model (lightweight wrapper)

Goals
- Simple API:
    ner = NERModel.prefer_spacy()  # or .auto()
    ents = ner.extract("Microsoft CEO Satya Nadella ...")
- Optional deps:
    * spaCy if installed (en_core_web_sm or any loaded nlp)
    * HuggingFace transformers pipeline('ner', grouped_entities=True) if installed
    * Fallback: fast regex-based extractor (capitalized spans + tickers)
- Friendly outputs:
    Entity(type, text, start, end, score, meta)

Integration
- Use `to_mentions(ents)` to adapt into {text, span, label} for actor_linker.py.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import importlib
import re
import unicodedata

# --------------------------- Data model ---------------------------

@dataclass
class Entity:
    text: str
    start: int
    end: int
    type: str          # "person" | "org" | "ticker" | "loc" | "misc"
    score: float = 1.0
    meta: Dict[str, Any] = None # type: ignore

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["meta"] = d.get("meta") or {}
        return d


# --------------------------- Utilities ----------------------------

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
# Greedy title-cased chunks: sequences like "International Business Machines"
_TITLE_CHUNK_RE = re.compile(r"\b([A-Z][a-zA-Z&.'\-]+(?:\s+[A-Z][a-zA-Z&.'\-]+){0,4})\b")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s


def _likely_org(chunk: str) -> bool:
    key = chunk.lower()
    return any(k in key for k in (
        "inc", "corp", "corporation", "company", "co.", "ltd", "llc", "plc", "bank",
        "university", "institute", "ministry", "department", "technologies", "systems",
        "holdings", "group", "ag", "sa", "nv"
    ))


# --------------------------- Backends -----------------------------

class _SpaCyBackend:
    def __init__(self, model: Optional[str] = None):
        spacy = importlib.import_module("spacy")
        if model is None:
            # try default small English
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                # allow user to load externally and pass in via .from_spacy()
                raise RuntimeError("spaCy present but model 'en_core_web_sm' not installed. "
                                   "Install: python -m spacy download en_core_web_sm")
        else:
            self.nlp = spacy.load(model)

    def extract(self, text: str) -> List[Entity]:
        doc = self.nlp(text)
        out: List[Entity] = []
        for ent in doc.ents:
            label = ent.label_.upper()
            typ = {
                "PERSON": "person",
                "PER": "person",
                "ORG": "org",
                "GPE": "loc",
                "LOC": "loc",
                "FAC": "loc",
                "PRODUCT": "misc",
                "EVENT": "misc",
                "LAW": "misc",
                "WORK_OF_ART": "misc",
            }.get(label, "misc")
            out.append(Entity(ent.text, ent.start_char, ent.end_char, typ, score=getattr(ent, "kb_id_", None) and 0.9 or 0.85, meta={"spacy": label}))
        # tickers with regex (spaCy often marks them as ORG/MISC)
        for m in _TICKER_RE.finditer(text):
            out.append(Entity(m.group(0), m.start(), m.end(), "ticker", 0.9, {"source": "regex"}))
        return _dedupe(out)


class _HFBackend:
    def __init__(self, model: Optional[str] = None):
        transformers = importlib.import_module("transformers")
        kwargs = {"aggregation_strategy": "simple"}
        if model:
            kwargs["model"] = model
        self.pipe = transformers.pipeline("ner", grouped_entities=True, **kwargs)

    def extract(self, text: str) -> List[Entity]:
        groups = self.pipe(text)
        ents: List[Entity] = []
        for g in groups:
            label = (g.get("entity_group") or g.get("entity") or "").upper()
            typ = "misc"
            if "PER" in label or "PERSON" in label: typ = "person"
            elif "ORG" in label: typ = "org"
            elif "LOC" in label or "GPE" in label: typ = "loc"
            start, end = int(g["start"]), int(g["end"])
            ents.append(Entity(g["word"], start, end, typ, float(g.get("score", 0.8)), {"hf": label}))
        for m in _TICKER_RE.finditer(text):
            ents.append(Entity(m.group(0), m.start(), m.end(), "ticker", 0.9, {"source": "regex"}))
        return _dedupe(ents)


class _RegexBackend:
    def extract(self, text: str) -> List[Entity]:
        out: List[Entity] = []
        # Tickers (ALLCAP up to 5)
        for m in _TICKER_RE.finditer(text):
            tok = m.group(0)
            if tok in {"AND", "THE", "FOR", "WITH", "THIS"}:
                continue
            out.append(Entity(tok, m.start(), m.end(), "ticker", 0.85, {"source": "regex"}))
        # Title-cased chunks; heuristics to split org vs person
        for m in _TITLE_CHUNK_RE.finditer(text):
            chunk = m.group(1)
            # skip if it's just the start of a sentence like "The"
            if chunk in {"The", "A", "An"}:
                continue
            typ = "org" if _likely_org(chunk) else "person"
            out.append(Entity(chunk, m.start(), m.end(), typ, 0.7, {"source": "regex"}))
        return _dedupe(out)


def _dedupe(ents: List[Entity]) -> List[Entity]:
    # Keep highest score per (span, type, text)
    seen: Dict[Tuple[int, int, str, str], Entity] = {}
    for e in ents:
        k = (e.start, e.end, e.type, e.text)
        if k not in seen or e.score > seen[k].score:
            seen[k] = e
    # Merge overlapping identical text with same type (pick best score)
    return sorted(seen.values(), key=lambda x: (x.start, -x.score))


# --------------------------- Public API ----------------------------

class NERModel:
    """
    Front-end wrapper around the chosen backend.
    Use:
        ner = NERModel.auto()         # spaCy -> HF -> regex
        ner = NERModel.prefer_spacy()
        ner = NERModel.prefer_hf()
        ner = NERModel.regex_only()
    """

    def __init__(self, backend: Any):
        self.backend = backend

    @classmethod
    def auto(cls) -> "NERModel":
        """Try spaCy, then HuggingFace, then fallback regex."""
        try:
            return cls(_SpaCyBackend())
        except Exception:
            try:
                return cls(_HFBackend())
            except Exception:
                return cls(_RegexBackend())

    @classmethod
    def prefer_spacy(cls, model: Optional[str] = None) -> "NERModel":
        return cls(_SpaCyBackend(model))

    @classmethod
    def prefer_hf(cls, model: Optional[str] = None) -> "NERModel":
        return cls(_HFBackend(model))

    @classmethod
    def regex_only(cls) -> "NERModel":
        return cls(_RegexBackend())

    def extract(self, text: str) -> List[Entity]:
        text = _norm(text or "")
        if not text:
            return []
        return self.backend.extract(text)

    # ---------- Integration helper for actor_linker ----------

    def to_mentions(self, ents: Iterable[Entity]) -> List[Dict[str, Any]]:
        """
        Convert Entity -> actor_linker Mention dicts:
            {"text": e.text, "span": (start, end), "label": mapped_type}
        Only "person", "org", "ticker" are kept; others labeled "unknown".
        """
        out: List[Dict[str, Any]] = []
        for e in ents:
            lbl = e.type
            if lbl not in {"person", "org", "ticker"}:
                lbl = "unknown"
            out.append({"text": e.text, "span": (e.start, e.end), "label": lbl})
        return out


# --------------------------- Self-test ----------------------------

if __name__ == "__main__":
    text = ("MSFT rallied after Microsoft Corporation announced new AI features. "
            "CEO Satya Nadella met officials in New York.")
    ner = NERModel.auto()
    ents = ner.extract(text)
    for e in ents:
        print(e.asdict())
    print("--- as mentions ---")
    print(NERModel.regex_only().to_mentions(ents))