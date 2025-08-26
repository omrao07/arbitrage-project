# backend/data_ext/sentiment/sources/reddit.py
"""
Reddit source loader for social sentiment.

Reads config from sentiment.yaml -> sources.reddit and returns raw posts
that your social_scraper.py can score and normalize.

Two modes:
1) Real mode (requires PRAW and credentials):
   - env / cfg: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, user_agent
   - subreddits: ["wallstreetbets", "stocks", ...]
   - limit: number of posts per subreddit (default 100)

2) Demo mode (no PRAW or no creds):
   - Emits plausible fake posts so the pipeline can be tested.

Returned record schema (raw):
{
  "source": "reddit",
  "subreddit": "wallstreetbets",
  "text": "TSLA to the moon!",
  "timestamp": "2025-08-22T00:15:00Z",
  "symbol": "TSLA",              # optional, best-effort extraction
  "meta": {
      "score": 123,
      "num_comments": 45,
      "permalink": "/r/...",
      "author": "u/someuser",
      "id": "postid"
  }
}

Note: social_scraper.py should call this fetch() and then pass each record to
sentiment_model.py to compute {'sentiment': float, 'confidence': float}, then
normalize/publish to STREAM_ALT_SIGNALS.
"""

from __future__ import annotations

import os
import re
import time
import datetime as dt
from typing import Any, Dict, List, Optional

# Optional real backend: PRAW
try:
    import praw  # type: ignore
    _HAVE_PRAW = True
except Exception:
    _HAVE_PRAW = False

SYMBOL_REGEX = re.compile(r"\b([A-Z]{2,5})(?:\b|\$)", re.IGNORECASE)
DOLLAR_TICKER = re.compile(r"\$([A-Za-z]{1,6})")


def _iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _extract_symbol(text: str, symbols_whitelist: Optional[List[str]] = None) -> Optional[str]:
    """
    Best-effort symbol extraction:
      - $TSLA style
      - UPPERCASE tickers (2-5 chars)
      - Optional whitelist to reduce false positives
    """
    if not text:
        return None

    m = DOLLAR_TICKER.search(text)
    if m:
        cand = m.group(1).upper()
        if (not symbols_whitelist) or (cand in symbols_whitelist):
            return cand

    m2 = SYMBOL_REGEX.search(text.upper())
    if m2:
        cand = m2.group(1).upper()
        if (not symbols_whitelist) or (cand in symbols_whitelist):
            return cand

    return None


# ------------------------------------------------------------------------------
# Real fetch via PRAW
# ------------------------------------------------------------------------------

def _fetch_reddit_real(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch recent submissions from configured subreddits using PRAW."""
    client_id = (cfg.get("client_id") or os.getenv("REDDIT_CLIENT_ID") or "").strip()
    client_secret = (cfg.get("client_secret") or os.getenv("REDDIT_CLIENT_SECRET") or "").strip()
    user_agent = (cfg.get("user_agent") or "hedgefundx-bot/0.1").strip()

    if not (_HAVE_PRAW and client_id and client_secret):
        return []

    subs = cfg.get("subreddits") or ["wallstreetbets"]
    limit = int(cfg.get("limit", 100))
    symbols_whitelist: Optional[List[str]] = cfg.get("symbols")  # optional ticker list

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )

    out: List[Dict[str, Any]] = []
    now_iso = _iso_now()

    for sub in subs:
        try:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.new(limit=limit):
                # Compose a text blob for scoring (title + selftext)
                text = f"{post.title or ''}\n{post.selftext or ''}".strip()
                if not text:
                    continue

                symbol = _extract_symbol(text, symbols_whitelist)
                ts = dt.datetime.utcfromtimestamp(getattr(post, "created_utc", time.time()))
                ts_iso = ts.replace(microsecond=0).isoformat() + "Z"

                out.append(
                    {
                        "source": "reddit",
                        "subreddit": str(sub),
                        "text": text,
                        "timestamp": ts_iso,
                        "symbol": symbol,
                        "meta": {
                            "score": int(getattr(post, "score", 0)),
                            "num_comments": int(getattr(post, "num_comments", 0)),
                            "permalink": f"https://www.reddit.com{getattr(post, 'permalink', '')}",
                            "author": f"u/{getattr(post, 'author', '')}",
                            "id": getattr(post, "id", ""),
                        },
                    }
                )
        except Exception:
            # If one subreddit fails (rate limits, auth), continue with others
            continue

    return out


# ------------------------------------------------------------------------------
# Fallback generator (no PRAW or no creds)
# ------------------------------------------------------------------------------

_FAKE_POSTS = [
    ("wallstreetbets", "TSLA to the moon, QQQ strong momentum into close", "TSLA"),
    ("stocks", "AAPL earnings whisper looks bullish, services growing fast", "AAPL"),
    ("cryptocurrency", "BTC funding flipping positive, watch out for squeeze", "BTC"),
    ("wallstreetbets", "NVDA AI cycle still insane, semis ripping", "NVDA"),
    ("stocks", "Oil supply shocks again? XOM looks interesting", "XOM"),
]

def _fetch_reddit_fallback(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    now_iso = _iso_now()
    out: List[Dict[str, Any]] = []
    for sub, text, sym in _FAKE_POSTS:
        out.append(
            {
                "source": "reddit",
                "subreddit": sub,
                "text": text,
                "timestamp": now_iso,
                "symbol": sym,
                "meta": {
                    "score": 100,
                    "num_comments": 25,
                    "permalink": f"https://www.reddit.com/r/{sub}/",
                    "author": "u/demo",
                    "id": f"demo_{sub}_{sym}_{int(time.time())}",
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

    cfg example (from sentiment.yaml):
    ----------------------------------
    sources:
      reddit:
        enabled: true
        client_id: "${REDDIT_CLIENT_ID}"
        client_secret: "${REDDIT_CLIENT_SECRET}"
        user_agent: "hedgefundx-bot/0.1"
        subreddits: ["wallstreetbets", "stocks", "cryptocurrency"]
        limit: 100
        symbols: ["TSLA", "AAPL", "NVDA", "BTC"]   # optional whitelist

    Returns list[dict] raw posts (see schema in module docstring).
    """
    if not cfg.get("enabled", False):
        return []

    # Try real PRAW path first
    real = _fetch_reddit_real(cfg)
    if real:
        return real

    # Fall back to demo posts
    return _fetch_reddit_fallback(cfg)



if __name__ == "__main__":
    demo_cfg = {
        "enabled": True,
        "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
        "user_agent": "hedgefundx-bot/0.1",
        "subreddits": ["wallstreetbets", "stocks"],
        "limit": 10,
        "symbols": ["TSLA", "AAPL", "NVDA", "BTC"],
    }
    posts = fetch(demo_cfg)
    for p in posts:
        print(p)