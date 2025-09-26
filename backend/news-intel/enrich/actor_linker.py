# news-intel/enrich/actor_linker.py
"""
Actor Linker
------------
Resolve actor mentions in articles (persons, orgs, tickers) to a knowledge base.

Design goals:
- Zero hard deps (stdlib only). Optional fuzzy support if `rapidfuzz` is installed.
- Deterministic scoring with clear provenance.
- Small, testable surface: `link_actors(article, kb)`.

Typical flow:
    1) Extract candidate mentions (regex/lightweight heuristics).
    2) Normalize (casefold, strip punctuation, expand common aliases).
    3) Candidate generation from KB (symbol map, aliases, prefix index).
    4) Score candidates (exact > alias > fuzzy).
    5) Return links with confidence + evidence.

You can swap step (1) with a heavier NER upstream and feed us mentions directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
import re
import unicodedata
import math

# Optional fuzzy matcher (pip install rapidfuzz)
try:
    from rapidfuzz import fuzz
except Exception:  # noqa: BLE001 - optional dep
    fuzz = None  # type: ignore[assignment]


# ----------------------------- Utilities -------------------------------------


_WORD_RE = re.compile(r"[A-Z][a-zA-Z&.'\-]{1,}|\b[A-Z]{2,}\b")
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")


def _norm(s: str) -> str:
    """Aggressive normalization for matching keys."""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"[\u2010-\u2015\-‐–—]", "-", s)  # unify hyphens
    s = re.sub(r"[^\w&\-. ]+", "", s)           # drop punctuation except -, ., &, _
    s = re.sub(r"\s+", " ", s)
    return s


def _token_set(s: str) -> Tuple[str, frozenset]:
    n = _norm(s)
    toks = frozenset(t for t in n.split(" ") if t)
    return n, toks


# ----------------------------- Data Model ------------------------------------


@dataclass
class KBEntity:
    id: str
    name: str
    type: str  # "person" | "org" | "ticker" | "place" ...
    aliases: List[str] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)

    def keys(self) -> List[str]:
        return [self.name, *self.aliases]


class KnowledgeBase:
    """
    Tiny in-memory KB with fast lookups by:
    - normalized name/alias
    - ticker symbol (uppercase)
    - prefix index for cheap candidate gen
    """

    def __init__(self, entities: Iterable[KBEntity]):
        self.entities: Dict[str, KBEntity] = {}
        self.by_key: Dict[str, str] = {}      # norm_name -> id
        self.by_ticker: Dict[str, List[str]] = {}  # TICKER -> [id]
        self.prefix: Dict[str, List[str]] = {}     # first 3 letters -> [id]

        for e in entities:
            self.entities[e.id] = e
            for k in e.keys():
                self.by_key[_norm(k)] = e.id
                p = _norm(k)[:3]
                if p:
                    self.prefix.setdefault(p, []).append(e.id)
            for t in e.tickers:
                self.by_ticker.setdefault(t.upper(), []).append(e.id)

    def get(self, id_: str) -> Optional[KBEntity]:
        return self.entities.get(id_)

    def candidates(self, mention: str) -> List[KBEntity]:
        """Return a small candidate set using exact/alias/prefix heuristics."""
        n = _norm(mention)
        exact = self.by_key.get(n)
        if exact:
            return [self.entities[exact]]

        pkey = n[:3]
        ids = self.prefix.get(pkey, [])
        # Deduplicate while preserving order
        seen, out = set(), []
        for id_ in ids:
            if id_ not in seen:
                out.append(self.entities[id_])
                seen.add(id_)
        return out

    def candidates_by_ticker(self, ticker: str) -> List[KBEntity]:
        return [self.entities[i] for i in self.by_ticker.get(ticker.upper(), [])]


@dataclass
class Mention:
    text: str
    span: Tuple[int, int]  # (start, end) in the article text
    label: str             # "person" | "org" | "ticker" | "unknown"


@dataclass
class Link:
    mention: Mention
    entity_id: Optional[str]
    confidence: float      # 0..1
    method: str            # "exact" | "alias" | "ticker" | "fuzzy" | "none"
    evidence: Dict[str, str] = field(default_factory=dict)


# ----------------------------- Extraction ------------------------------------


def extract_mentions(text: str) -> List[Mention]:
    """
    Extremely light extractor:
    - Upper-cased tokens of length 2-5 → possible tickers
    - Capitalized word chunks → actor candidates (person/org)
    (Replace with your NER upstream if available.)
    """
    mentions: List[Mention] = []

    # Tickers
    for m in _TICKER_RE.finditer(text):
        tok = m.group(0)
        # Heuristic: ignore common stop "AND", "THE", etc.
        if tok in {"AND", "THE", "FOR", "BUT", "WITH", "THIS"}:
            continue
        mentions.append(Mention(tok, (m.start(), m.end()), "ticker"))

    # Actor-like phrases (very rough)
    for m in _WORD_RE.finditer(text):
        w = m.group(0)
        if w.isupper():  # already captured as ticker likely
            continue
        # Extend to multi-word span greedily (e.g., "International Business Machines")
        # For simplicity, we just capture single capitalized tokens here.
        mentions.append(Mention(w, (m.start(), m.end()), "unknown"))

    return _dedupe_mentions(mentions)


def _dedupe_mentions(mentions: List[Mention]) -> List[Mention]:
    seen = set()
    out = []
    for m in mentions:
        key = (m.text, m.span)
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


# ----------------------------- Scoring ---------------------------------------


def _score_exact(mention: str, ent: KBEntity) -> Optional[Tuple[float, str]]:
    nm = _norm(mention)
    if nm == _norm(ent.name):
        return 1.0, "exact"
    for a in ent.aliases:
        if nm == _norm(a):
            return 0.95, "alias"
    return None


def _score_ticker(mention: str, ent: KBEntity) -> Optional[Tuple[float, str]]:
    m = mention.upper()
    if m in ent.tickers:
        return 0.98, "ticker"
    return None


def _score_fuzzy(mention: str, ent: KBEntity) -> Optional[Tuple[float, str]]:
    if fuzz is None:
        return None
    # token_set_ratio is resilient to order/extra tokens
    s = max(
        fuzz.token_set_ratio(mention, ent.name),
        *[fuzz.token_set_ratio(mention, a) for a in ent.aliases] or [0],
    )
    if s >= 90:
        # map (90..100) to (0.80..0.92)
        conf = 0.80 + (s - 90) * (0.12 / 10.0)
        return conf, "fuzzy"
    return None


def _best_candidate(mention: Mention, cands: List[KBEntity]) -> Link:
    best: Tuple[float, str, Optional[str], Dict[str, str]] = (0.0, "none", None, {})
    for e in cands:
        for scorer in (_score_exact, _score_ticker, _score_fuzzy):
            res = scorer(mention.text, e) if scorer is not _score_ticker or mention.label == "ticker" else None
            if res:
                conf, method = res
                if conf > best[0]:
                    best = (conf, method, e.id, {"name": e.name})
    return Link(mention, best[2], best[0], best[1], best[3])


# ----------------------------- Public API ------------------------------------


def link_actors(
    article_text: str,
    kb: KnowledgeBase,
    mentions: Optional[List[Mention]] = None,
    min_confidence: float = 0.80,
) -> List[Link]:
    """
    Link actor mentions in `article_text` to KB.
    If `mentions` is provided (e.g., from NER), we’ll use them; otherwise we self-extract.

    Returns a list of Link with entity_id or None if below threshold.
    """
    if mentions is None:
        mentions = extract_mentions(article_text)

    links: List[Link] = []
    for m in mentions:
        # Fast path: ticker mentions
        cand_list: List[KBEntity]
        if m.label == "ticker":
            cand_list = kb.candidates_by_ticker(m.text)
            if not cand_list:  # fall back to name lookup with same token
                cand_list = kb.candidates(m.text)
        else:
            cand_list = kb.candidates(m.text)

        if not cand_list:
            links.append(Link(m, None, 0.0, "none", {}))
            continue

        link = _best_candidate(m, cand_list)
        if link.confidence >= min_confidence:
            links.append(link)
        else:
            links.append(Link(m, None, link.confidence, link.method, link.evidence))
    return _merge_overlaps(links)


def _merge_overlaps(links: List[Link]) -> List[Link]:
    """
    If multiple identical mentions map to the same entity nearby, keep the highest-confidence one.
    Simple O(n log n) sweep.
    """
    links_sorted = sorted(links, key=lambda l: (l.mention.span[0], -l.confidence))
    out: List[Link] = []
    last_span = (-1, -1)
    last_entity = None
    for l in links_sorted:
        if l.entity_id is not None and l.mention.span[0] <= last_span[1] and l.entity_id == last_entity:
            # overlap with same entity; keep the one already in out (higher conf due to sort)
            continue
        out.append(l)
        last_span = l.mention.span
        last_entity = l.entity_id
    return out


# ----------------------------- Example / Self-test ----------------------------

if __name__ == "__main__":
    kb = KnowledgeBase([
        KBEntity(id="org:msft", name="Microsoft Corporation", type="org",
                 aliases=["Microsoft", "MSFT"], tickers=["MSFT"], meta={"country":"US"}),
        KBEntity(id="org:ibm", name="International Business Machines", type="org",
                 aliases=["IBM"], tickers=["IBM"]),
        KBEntity(id="person:nadella", name="Satya Nadella", type="person",
                 aliases=["Mr. Nadella", "Satya N."], tickers=[]),
    ])

    text = ("MSFT surged after Microsoft announced new AI features. "
            "Chief Executive Satya Nadella said IBM would be a key partner.")

    results = link_actors(text, kb)
    for r in results:
        print(f"Mention='{r.mention.text}' -> {r.entity_id} "
              f"conf={r.confidence:.2f} via {r.method} span={r.mention.span} ev={r.evidence}")