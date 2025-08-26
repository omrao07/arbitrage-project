# backend/news/news_moneycontrol.py
"""
Moneycontrol News Ingestor
--------------------------
- Fetches headlines via RSS (top news, business, markets, results, IPO, etc.)
- Normalizes to your event schema:
    {
      "ts_ms": <epoch ms>,
      "source": "moneycontrol",
      "title": "...",
      "url": "https://...",
      "summary": "...",            # best-effort
      "tickers": ["RELIANCE.NS"],  # best-effort mapping
      "symbols": ["NIFTY"],        # indices/sectors if detected
      "category": "markets",
      "lang": "en",
    }
- Optional full-text expansion if `newspaper3k` or `bs4`+`requests` present.
- Can publish on bus topic: news.moneycontrol

CLI:
  python -m backend.news.news_moneycontrol --limit 30 --publish
  python -m backend.news.news_moneycontrol --search "HDFC merger" --limit 20
  python -m backend.news.news_moneycontrol --probe
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---- optional deps (graceful fallbacks) ----------------
try:
    import feedparser  # pip install feedparser
except Exception:
    feedparser = None  # type: ignore

try:
    import requests  # pip install requests
except Exception:
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception:
    BeautifulSoup = None  # type: ignore

try:
    from newspaper import Article  # type: ignore # pip install newspaper3k
except Exception:
    Article = None  # type: ignore

# ---- optional project glue ---------------------------------------
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore

try:
    # if you added a base class
    from backend.news.news_base import NewsSource # type: ignore
except Exception:
    class NewsSource:  # minimal stand-in
        pass

# ---- config -------------------------------------------------------

DEFAULT_FEEDS: Dict[str, str] = {
    # These are commonly used Moneycontrol RSS endpoints; if any change, override via env.
    # You can add or remove feeds safely.
    "top":      os.getenv("MC_FEED_TOP",      "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    "business": os.getenv("MC_FEED_BUSINESS", "https://www.moneycontrol.com/rss/business.xml"),
    "markets":  os.getenv("MC_FEED_MARKETS",  "https://www.moneycontrol.com/rss/marketreports.xml"),
    "ipo":      os.getenv("MC_FEED_IPO",      "https://www.moneycontrol.com/rss/iponews.xml"),
    "results":  os.getenv("MC_FEED_RESULTS",  "https://www.moneycontrol.com/rss/results.xml"),
    "mutual":   os.getenv("MC_FEED_MF",       "https://www.moneycontrol.com/rss/mf-news.xml"),
}

BUS_TOPIC = os.getenv("NEWS_MONEYCONTROL_TOPIC", "news.moneycontrol")
UA = os.getenv("HTTP_UA", "Mozilla/5.0 (compatible; MCNewsBot/1.0)")

# crude NSE/BSE ticker hints; expand in config if you like
TICKER_HINTS = [
    r"\bRELIANCE\b", r"\bTCS\b", r"\bHDFC\b", r"\bHDFC BANK\b", r"\bINFY\b", r"\bINFOSYS\b",
    r"\bICICI\b", r"\bSBIN\b", r"\bITC\b", r"\bLT\b", r"\bBHARTI\b", r"\bSUN PHARMA\b",
    r"\bADANI\b", r"\bKOTAK\b", r"\bMARUTI\b",
]
INDEX_HINTS = [r"\bNIFTY\b", r"\bSENSEX\b", r"\bBANK NIFTY\b", r"\bFINNIFTY\b"]

# Optional mapping for better symbols (attach .NS by default)
CANON_MAP = {
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFC": "HDFC.NS",
    "ICICI": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "MARUTI": "MARUTI.NS",
}

# ---- helpers ------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)

def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _extract_tickers(title: str, summary: str = "") -> Tuple[List[str], List[str]]:
    text = f"{title} {summary}".upper()
    tickers: List[str] = []
    for pat in TICKER_HINTS:
        if re.search(pat, text):
            key = pat.strip(r"\b").replace("\\b", "").upper().strip()
            # attempt canon
            canon = None
            for k, v in CANON_MAP.items():
                if k in key or key in k:
                    canon = v; break
            tickers.append(canon or key)
    # indices
    symbols: List[str] = []
    for pat in INDEX_HINTS:
        if re.search(pat, text):
            sym = pat.strip(r"\b").replace("\\b", "").upper().strip().replace(" ", "_")
            symbols.append(sym)
    # dedupe
    tickers = sorted(set(tickers))
    symbols = sorted(set(symbols))
    return tickers, symbols

def _fetch_article_text(url: str, timeout: float = 6.0) -> str:
    """
    Best-effort: use newspaper3k if present; else bs4+requests; else return "".
    """
    try:
        if Article is not None:
            art = Article(url)
            art.download()
            art.parse()
            return _clean_text(art.text)
    except Exception:
        pass
    if requests is None or BeautifulSoup is None:
        return ""
    try:
        hdrs = {"User-Agent": UA}
        resp = requests.get(url, timeout=timeout, headers=hdrs)
        if not (200 <= resp.status_code < 300):
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        # heuristic: pick paragraphs inside article tag
        arts = soup.find_all(["article"])
        if not arts:
            arts = [soup]
        paras: List[str] = []
        for a in arts:
            for p in a.find_all("p"): # type: ignore
                txt = _clean_text(p.get_text(" "))
                if len(txt) > 50:  # skip tiny crumbs
                    paras.append(txt)
        return _clean_text(" ".join(paras)[:12000])  # cap to 12k chars
    except Exception:
        return ""

# ---- main class ---------------------------------------------------

@dataclass
class MoneycontrolNews(NewsSource): # type: ignore
    feeds: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FEEDS))
    expand_text: bool = True     # try to fetch full text (optional)
    default_category: str = "markets"

    def _load_feed(self, url: str) -> List[Dict[str, Any]]:
        if feedparser is None:
            raise RuntimeError("feedparser not installed. Run: pip install feedparser")
        d = feedparser.parse(url)
        out: List[Dict[str, Any]] = []
        for e in d.entries:
            title = _clean_text(getattr(e, "title", ""))
            link = getattr(e, "link", "")
            summary = _clean_text(getattr(e, "summary", getattr(e, "description", "")))
            # timestamps: feedparser normalizes 'published_parsed'
            ts_ms = _now_ms()
            try:
                if getattr(e, "published_parsed", None):
                    ts_ms = int(time.mktime(e.published_parsed) * 1000) # type: ignore
                elif getattr(e, "updated_parsed", None):
                    ts_ms = int(time.mktime(e.updated_parsed) * 1000) # type: ignore
            except Exception:
                pass
            out.append({
                "ts_ms": ts_ms,
                "title": title,
                "url": link,
                "summary": summary,
            })
        return out

    def fetch(self, *, category: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch merged & de-duped entries across chosen RSS feeds.
        """
        cats = [category] if category else list(self.feeds.keys())
        items: List[Dict[str, Any]] = []
        for c in cats:
            url = self.feeds.get(c)
            if not url:
                continue
            try:
                rows = self._load_feed(url)
                for r in rows:
                    r["category"] = c
                    items.append(r)
            except Exception:
                # skip broken feeds
                continue
        # de-dupe by URL/title hash
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for r in sorted(items, key=lambda x: x["ts_ms"], reverse=True):
            key = _hash((r.get("url") or "") + "|" + (r.get("title") or ""))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
            if len(uniq) >= limit:
                break
        # optional expand
        if self.expand_text:
            for r in uniq:
                if not r.get("summary"):
                    r["summary"] = _fetch_article_text(r.get("url") or "")
        return uniq

    def to_events(self, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for r in rows:
            title = r.get("title") or ""
            summary = r.get("summary") or ""
            tickers, symbols = _extract_tickers(title, summary)
            ev = {
                "ts_ms": int(r.get("ts_ms") or _now_ms()),
                "source": "moneycontrol",
                "title": title,
                "url": r.get("url") or "",
                "summary": summary,
                "tickers": tickers,
                "symbols": symbols,
                "category": r.get("category") or self.default_category,
                "lang": "en",
            }
            events.append(ev)
        return events

    def publish(self, events: Iterable[Dict[str, Any]]) -> None:
        if not publish_stream:
            return
        for ev in events:
            try:
                publish_stream(BUS_TOPIC, ev)
            except Exception:
                # ignore bus errors; caller can log
                pass

# ---- CLI ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Moneycontrol News Ingestor")
    ap.add_argument("--category", type=str, help="Which feed (default: all)")
    ap.add_argument("--limit", type=int, default=50, help="Max items")
    ap.add_argument("--no-expand", action="store_true", help="Disable article text expansion")
    ap.add_argument("--publish", action="store_true", help="Publish to bus")
    ap.add_argument("--search", type=str, help="Keyword filter after fetch")
    ap.add_argument("--json", action="store_true", help="Print events JSON to stdout")
    ap.add_argument("--probe", action="store_true", help="Print 5 latest titles for quick sanity")
    args = ap.parse_args()

    mc = MoneycontrolNews(expand_text=(not args.no_expand))
    rows = mc.fetch(category=args.category, limit=args.limit)
    if args.search:
        q = args.search.lower()
        rows = [r for r in rows if q in (r.get("title","").lower() + " " + r.get("summary","").lower())]
    events = mc.to_events(rows)

    if args.probe:
        for ev in events[:5]:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ev['ts_ms']/1000))}] {ev['title']} ({ev['url']})")
        return

    if args.json:
        import json as _json
        print(_json.dumps(events, indent=2, ensure_ascii=False))

    if args.publish:
        mc.publish(events)
        print(f"Published {len(events)} events to {BUS_TOPIC}")

if __name__ == "__main__":
    main()