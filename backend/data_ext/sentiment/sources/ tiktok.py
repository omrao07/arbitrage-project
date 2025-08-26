# backend/data_ext/sentiment/sources/tiktok.py
"""
TikTok source loader for social sentiment.

Reads config from sentiment.yaml -> sources.tiktok and returns raw short-form
posts for your social_scraper.py to score (via sentiment_model.py) and normalize.

Two modes:
1) Real mode (best-effort; community libs are brittle and may require cookies):
   - Optional libs: TikTokApi (pip install TikTokApi) / tiktokapipy
   - Config: hashtags list, max_results per hashtag
   - NOTE: Respect TikTok ToS. Prefer first-party approved data partners for production.

2) Demo mode (no client / errors):
   - Emits plausible fake posts for pipeline testing.

Returned record schema (raw):
{
  "source": "tiktok",
  "hashtag": "stockmarket",
  "text": "TSLA breakout incoming",
  "timestamp": "2025-08-22T00:15:00Z",
  "symbol": "TSLA",                 # optional, best-effort extraction
  "meta": {
      "likes": 1234,
      "views": 54321,
      "shares": 120,
      "author": "@creator",
      "id": "vid123",
      "link": "https://www.tiktok.com/@creator/video/vid123"
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

# Try optional community client (often fragile; safe-guarded)
_HAVE_TIKTOK = False
try:
    # from TikTokApi import TikTokApi  # uncomment if you actually wire it
    # _HAVE_TIKTOK = True
    pass
except Exception:
    _HAVE_TIKTOK = False

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
# Real fetch (placeholder; keep disabled unless you wire credentials/cookies)
# ------------------------------------------------------------------------------

def _fetch_tiktok_real(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Placeholder for real TikTok pulls.
    In practice you will:
      - Initialize a client (requires cookies or device params)
      - Iterate over hashtag feeds
      - Extract caption text, metrics, author, link
    Return [] to fall back if not wired.
    """
    if not _HAVE_TIKTOK:
        return []

    # Example (pseudo):
    # hashtags = cfg.get("hashtags", ["stockmarket"])
    # max_results = int(cfg.get("max_results", 30))
    # out = []
    # with TikTokApi() as api:
    #     for tag in hashtags:
    #         for post in api.hashtag(name=tag).videos(count=max_results):
    #             text = post.caption or ""
    #             ts_iso = dt.datetime.utcfromtimestamp(post.create_time).replace(microsecond=0).isoformat()+"Z"
    #             out.append({
    #                 "source": "tiktok",
    #                 "hashtag": tag,
    #                 "text": text,
    #                 "timestamp": ts_iso,
    #                 "symbol": _extract_symbol(text, cfg.get("symbols")),
    #                 "meta": {
    #                     "likes": int(getattr(post.stats, "digg_count", 0)),
    #                     "views": int(getattr(post.stats, "play_count", 0)),
    #                     "shares": int(getattr(post.stats, "share_count", 0)),
    #                     "author": f"@{getattr(post.author, 'unique_id', 'unknown')}",
    #                     "id": getattr(post, "id", ""),
    #                     "link": f"https://www.tiktok.com/@{getattr(post.author, 'unique_id','')}/video/{getattr(post,'id','')}",
    #                 }
    #             })
    # return out
    return []


# ------------------------------------------------------------------------------
# Fallback generator (demo)
# ------------------------------------------------------------------------------

_FAKE_LINES = [
    ("stockmarket", "TSLA breakout incoming, earnings momentum still strong 🚀", "TSLA"),
    ("trading", "AAPL services growth > hardware; margin lift case 📈", "AAPL"),
    ("crypto", "BTC funding positive; watch for short squeeze 🔥", "BTC"),
    ("ai", "NVDA cycle durable? hyperscaler capex says yes", "NVDA"),
    ("energy", "Oil volatility up; refining margins improving", None),
]

def _fetch_tiktok_fallback(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    now = _iso_now()
    out: List[Dict[str, Any]] = []
    symbols_whitelist: Optional[List[str]] = cfg.get("symbols")

    for tag, text, sym in _FAKE_LINES:
        sym = sym or _extract_symbol(text, symbols_whitelist)
        out.append(
            {
                "source": "tiktok",
                "hashtag": tag,
                "text": text,
                "timestamp": now,
                "symbol": sym,
                "meta": {
                    "likes": random.randint(200, 5000),
                    "views": random.randint(5_000, 200_000),
                    "shares": random.randint(10, 1000),
                    "author": "@demo_creator",
                    "id": f"vid_{tag}_{int(time.time())}",
                    "link": f"https://www.tiktok.com/tag/{tag}",
                },
            }
        )
    return out


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def fetch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Entry point for social_scraper.py.

    sentiment.yaml example:
    -----------------------
    sources:
      tiktok:
        enabled: false
        api_key: "${TIKTOK_API_KEY}"      # if using an approved partner
        hashtags: ["stockmarket", "trading", "crypto"]
        max_results: 30
        symbols: ["TSLA","AAPL","NVDA","BTC"]  # optional whitelist

    Returns list[dict] raw posts (schema in module docstring).
    """
    if not cfg.get("enabled", False):
        return []

    # Attempt real path (likely returns [] unless you wire a client)
    real = _fetch_tiktok_real(cfg)
    if real:
        return real

    # Fallback demo
    return _fetch_tiktok_fallback(cfg)


# ------------------------------------------------------------------------------
# Demo CLI
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    demo_cfg = {
        "enabled": True,
        "hashtags": ["stockmarket", "trading", "crypto"],
        "max_results": 10,
        "symbols": ["TSLA", "AAPL", "NVDA", "BTC"],
    }
    posts = fetch(demo_cfg)
    for p in posts:
        print(p)