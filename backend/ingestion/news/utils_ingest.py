# backend/ingestion/news/utils_ingest.py
from __future__ import annotations

import hashlib
import html
import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import request, error as urlerror
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

# -------- HTTP (stdlib) --------

_DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

def http_get(
    url: str,
    timeout: float = 6.0,
    max_retries: int = 3,
    backoff: float = 0.4,
    headers: Optional[Dict[str, str]] = None,
) -> bytes:
    """
    Minimal HTTP GET with retry + backoff.
    Returns raw bytes (caller decides how to parse).
    """
    hdrs = {"User-Agent": _DEFAULT_UA}
    if headers:
        hdrs.update(headers)

    last_exc: Optional[Exception] = None
    delay = backoff
    for _ in range(max_retries):
        try:
            req = request.Request(url, headers=hdrs, method="GET")
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_exc = e
            time.sleep(delay)
            delay *= 2
    raise last_exc or RuntimeError("http_get failed without exception")

def http_get_json(url: str, **kw) -> Any:
    raw = http_get(url, **kw)
    return json.loads(raw.decode("utf-8", "ignore"))


# -------- RSS parsing --------
try:
    import feedparser  # pip install feedparser
except Exception:  # soft dependency
    feedparser = None  # type: ignore

def parse_rss(url: str, use_fetch: bool = False):
    """
    Returns a feedparser result. If use_fetch=True, we fetch bytes ourselves.
    """
    if feedparser is None:
        raise ImportError("feedparser is required. pip install feedparser")
    if use_fetch:
        raw = http_get(url)
        return feedparser.parse(raw)
    return feedparser.parse(url)


# -------- Time helpers --------

def to_unix(ts: Any) -> float:
    """
    Convert various timestamp representations to unix seconds.
    Accepts:
      - int/float (seconds or ms)
      - RFC2822 strings (e.g., 'Wed, 21 Aug 2024 12:34:56 GMT')
      - ISO strings (e.g., '2024-08-21T12:34:56Z')
    Fallback: now()
    """
    if ts is None:
        return time.time()

    # numeric
    if isinstance(ts, (int, float)):
        v = float(ts)
        return v / 1000.0 if v > 10_000_000_000 else v

    s = str(ts).strip()
    # RFC2822
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        pass

    # ISO 8601
    try:
        # normalize 'Z'
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time()


# -------- Text helpers --------

def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    # collapse whitespace + unescape HTML
    return html.unescape(" ".join(s.split()))

def truncate(s: str, n: int = 500) -> str:
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)] + "…"

def headline_hash(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8", "ignore")).hexdigest()


# -------- Symbol extraction --------

# NSE/BSE like RELIANCE.NS / SBIN.BO
_PAT_NSE_BSE = re.compile(r"\b([A-Z][A-Z0-9]{0,11}\.(?:NS|BO))\b")

# simple US tickers (best-effort; avoids matching words)
_PAT_US = re.compile(r"\b([A-Z]{1,5})(?=\s|\W|$)")

# basic crypto symbols (BTC, ETH, SOL, etc.)
_PAT_CRYPTO = re.compile(r"\b(BTC|ETH|SOL|XRP|ADA|DOGE|MATIC|BNB|DOT|LTC)\b", re.I)

def extract_nse_bse(text: str) -> Optional[str]:
    m = _PAT_NSE_BSE.search(text.upper())
    return m.group(1) if m else None

def extract_us(text: str) -> Optional[str]:
    """
    Very naive extractor — only use as fallback.
    Prefer provider-supplied ticker fields whenever possible.
    """
    m = _PAT_US.search(text.upper())
    return m.group(1) if m else None

def extract_crypto(text: str) -> Optional[str]:
    m = _PAT_CRYPTO.search(text)
    return m.group(1).upper() if m else None

def first_symbol(text: str) -> Optional[str]:
    """
    Try NSE/BSE, then crypto, then US ticker best-effort.
    """
    t = text or ""
    return extract_nse_bse(t) or extract_crypto(t) or extract_us(t)