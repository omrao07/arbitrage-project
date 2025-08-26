# backend/data_ext/sentiment/sources/x.py
"""
X (Twitter) source loader for social sentiment.

Reads config from sentiment.yaml -> sources.x and returns raw posts
for social_scraper.py to score (via sentiment_model.py) and normalize.

Two modes:
1) Real mode (requires tweepy or equivalent client + API keys):
   - pip install tweepy
   - Config:
       bearer_token: "${X_BEARER_TOKEN}"
       queries: ["TSLA", "AAPL"]
       max_results: 50
       lang: "en"
   - Note: X API has strict rate limits; for production use, consider paid tiers or data partners.

2) Demo mode (no tweepy or no creds):
   - Emits plausible fake tweets for pipeline testing.

Returned record schema (raw):
{
  "source": "x",
  "query": "TSLA",
  "text": "TSLA earnings beat, stock up 🚀",
  "timestamp": "2025-08-22T00:15:00Z",
  "symbol": "TSLA",           # best-effort extraction
  "meta": {
      "author": "@elonmusk",
      "id": "123456",
      "retweets": 321,
      "likes": 2100,
      "link": "https://x.com/elonmusk/status/123456"
  }
}
"""

from __future__ import annotations

import os
import re
import time
import random
import datetime as dt
from typing import Any, Dict, List, Optional

# Try real client
_HAVE_TWEEPY = False
try:
    import tweepy  # type: ignore
    _HAVE_TWEEPY = True
except Exception:
    _HAVE_TWEEPY = False

SYMBOL_DOLLAR = re.compile(r"\$([A-Za-z]{1,6})")
SYMBOL_UPPER  = re.compile(r"\b([A-Z]{2,5})\b")


def _iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _extract_symbol(text: str, whitelist: Optional[List[str]] = None) -> Optional[str]:
    if not text:
        return None
    m = SYMBOL_DOLLAR.search(text)
    if m:
        cand = m.group(1).upper()
        if not whitelist or cand in whitelist:
            return cand
    m2 = SYMBOL_UPPER.search(text.upper())
    if m2:
        cand = m2.group(1).upper()
        if not whitelist or cand in whitelist:
            return cand
    return None


# ------------------------------------------------------------------------------
# Real fetch (Tweepy, bearer_token auth)
# ------------------------------------------------------------------------------

def _fetch_x_real(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    token = (cfg.get("bearer_token") or os.getenv("X_BEARER_TOKEN") or "").strip()
    if not (_HAVE_TWEEPY and token):
        return []

    client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)

    queries = cfg.get("queries") or ["TSLA"]
    max_results = int(cfg.get("max_results", 50))
    lang = cfg.get("lang", "en")
    symbols_whitelist: Optional[List[str]] = cfg.get("symbols")

    out: List[Dict[str, Any]] = []
    now = _iso_now()

    for q in queries:
        try:
            # Search recent tweets
            resp = client.search_recent_tweets(
                query=f"{q} lang:{lang} -is:retweet",
                tweet_fields=["id", "text", "created_at", "public_metrics"],
                max_results=min(max_results, 100),
            )
            if not resp or not resp.data:
                continue

            for tw in resp.data:
                text = tw.text or ""
                symbol = _extract_symbol(text, symbols_whitelist) or q.upper()
                ts_iso = (tw.created_at or dt.datetime.utcnow()).replace(microsecond=0).isoformat() + "Z"
                metrics = getattr(tw, "public_metrics", {}) or {}

                out.append(
                    {
                        "source": "x",
                        "query": q,
                        "text": text,
                        "timestamp": ts_iso,
                        "symbol": symbol,
                        "meta": {
                            "author": "@unknown",  # Basic search API doesn’t return usernames without expansions
                            "id": str(tw.id),
                            "retweets": metrics.get("retweet_count", 0),
                            "likes": metrics.get("like_count", 0),
                            "link": f"https://x.com/i/web/status/{tw.id}",
                        },
                    }
                )
        except Exception:
            continue

    return out


# ------------------------------------------------------------------------------
# Fallback generator (demo)
# ------------------------------------------------------------------------------

_FAKE_TWEETS = [
    ("TSLA", "TSLA earnings beat, stock up 🚀", "TSLA"),
    ("AAPL", "AAPL services momentum strong, margins expanding", "AAPL"),
    ("BTC", "BTC ETF flows bullish, funding flipping positive", "BTC"),
    ("NVDA", "NVDA AI capex cycle intact, demand still huge", "NVDA"),
    ("OIL", "Oil prices spiking on supply cuts, XOM benefits", "XOM"),
]

def _fetch_x_fallback(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    now = _iso_now()
    out: List[Dict[str, Any]] = []
    for query, text, sym in _FAKE_TWEETS:
        out.append(
            {
                "source": "x",
                "query": query,
                "text": text,
                "timestamp": now,
                "symbol": sym,
                "meta": {
                    "author": "@demo_user",
                    "id": f"demo_{query}_{int(time.time())}",
                    "retweets": random.randint(0, 500),
                    "likes": random.randint(0, 2000),
                    "link": f"https://x.com/search?q={query}",
                },
            }
        )
    return out


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def fetch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Entry for social_scraper.py.

    sentiment.yaml example:
    -----------------------
    sources:
      x:
        enabled: true
        bearer_token: "${X_BEARER_TOKEN}"
        queries: ["TSLA","AAPL","NVDA","BTC"]
        max_results: 30
        lang: "en"
        symbols: ["TSLA","AAPL","NVDA","BTC"]
    """
    if not cfg.get("enabled", False):
        return []

    real = _fetch_x_real(cfg)
    if real:
        return real

    return _fetch_x_fallback(cfg)


# ------------------------------------------------------------------------------
# Demo CLI
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    demo_cfg = {
        "enabled": True,
        "queries": ["TSLA", "AAPL", "NVDA", "BTC"],
        "max_results": 5,
        "symbols": ["TSLA", "AAPL", "NVDA", "BTC"],
    }
    posts = fetch(demo_cfg)
    for p in posts:
        print(p)