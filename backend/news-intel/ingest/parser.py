# news-intel/ingest/parser.py
"""
Parser / Normalizer for news inputs.

Goals
- Accept multiple raw shapes (RSS item dicts, REST JSON records, raw HTML).
- Produce a consistent Article schema:
    {
      "id": str, "title": str, "body": str, "url": str,
      "published_at": str, "source": str, "lang": str, "raw": dict
    }
- Be dependency-light; optionally use BeautifulSoup/langdetect if available.

Public entrypoints
- normalize_article(raw: dict, *, source="unknown") -> dict
- parse_html(html: str, url: str | None = None, title_hint: str | None = None) -> dict
- clean_text(text: str) -> str
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import datetime as dt
import hashlib
import html
import json
import re
import unicodedata

# Optional helpers
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

try:
    from langdetect import detect as _ld_detect  # type: ignore
except Exception:
    _ld_detect = None  # type: ignore


# ----------------------------- utils -----------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")  # fallback when BS4 is unavailable


def _norm_ws(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = html.unescape(s)
    s = s.replace("\u00A0", " ")  # nbsp
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def _strip_html(html_text: str) -> str:
    if not html_text:
        return ""
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html_text, "html.parser")
        # remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return _norm_ws(text)
    # fallback: crude tag strip
    return _norm_ws(_TAG_RE.sub(" ", html_text))


def _detect_lang(text: str) -> str:
    text = text[:2000]  # keep it short for detectors
    if _ld_detect is not None:
        try:
            return _ld_detect(text) or "und"
        except Exception:
            return "und"
    # ultra-simple heuristic fallback (latin vs cyrillic vs han)
    if not text:
        return "und"
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "zh"
    if re.search(r"[\u0400-\u04FF]", text):
        return "ru"
    return "en"


def _iso8601(ts: Any) -> str:
    """Best-effort to return ISO-8601 string."""
    if ts is None:
        return ""
    if isinstance(ts, (int, float)):
        try:
            return dt.datetime.utcfromtimestamp(float(ts)).isoformat() + "Z"
        except Exception:
            return ""
    if isinstance(ts, (dt.datetime, )):
        t = ts
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t.astimezone(dt.timezone.utc).isoformat()
    if isinstance(ts, str):
        s = ts.strip()
        # already iso?
        if re.match(r"^\d{4}-\d{2}-\d{2}T", s):
            return s
        # common formats
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%d %b %Y %H:%M:%S %Z"):
            try:
                return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc).isoformat()
            except Exception:
                continue
    return ""


def _mk_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        if p:
            h.update(p.encode("utf-8", errors="ignore"))
            h.update(b"\x00")
    return h.hexdigest()


def clean_text(text: str) -> str:
    """Public helper: normalize whitespace, unescape HTML entities, strip control chars."""
    return _norm_ws(text)


# ----------------------------- schema -----------------------------

@dataclass
class Article:
    id: str
    title: str
    body: str
    url: str
    published_at: str
    source: str
    lang: str
    raw: Dict[str, Any]


# ----------------------------- normalizers -----------------------------

def normalize_article(raw: Dict[str, Any], *, source: str = "unknown") -> Dict[str, Any]:
    """
    Normalize a heterogeneous input dict into Article schema.
    Handles common RSS keys (title, summary, link, published) and generic API keys.
    """
    # try common keys first
    title = raw.get("title") or raw.get("headline") or raw.get("name") or ""
    title = _strip_html(str(title))

    body = (
        raw.get("body")
        or raw.get("summary")
        or raw.get("description")
        or raw.get("content")
        or ""
    )
    # body may be array/obj (e.g., {rendered: ..} or [{"type":"text","data":".."}])
    if isinstance(body, (list, tuple)):
        body = " ".join(_strip_html(str(x)) for x in body)
    elif isinstance(body, dict):
        # common WP/ghost shape
        body = body.get("rendered") or body.get("text") or body.get("data") or body
    body = _strip_html(str(body))

    url = raw.get("url") or raw.get("link") or raw.get("permalink") or ""

    published = (
        raw.get("published_at")
        or raw.get("pubDate")
        or raw.get("date")
        or raw.get("created_at")
        or ""
    )
    published_iso = _iso8601(published)

    # language
    lang = raw.get("language") or raw.get("lang") or _detect_lang(f"{title} {body}")

    # id preference: explicit id > url > hash(title+published)
    raw_id = str(raw.get("id") or raw.get("_id") or "")
    art_id = raw_id or (url if url else _mk_id(title, published_iso))

    art = Article(
        id=art_id,
        title=title[:500],
        body=body,
        url=str(url),
        published_at=published_iso,
        source=source or str(raw.get("source") or "unknown"),
        lang=lang,
        raw=raw,
    )
    return asdict(art)


def parse_html(html_text: str, url: Optional[str] = None, title_hint: Optional[str] = None,
               *, source: str = "web") -> Dict[str, Any]:
    """
    Parse raw HTML into an Article (best-effort).
    - If BeautifulSoup is available, use it to extract <title> and main text.
    - Fallback: strip tags and use first ~12 words as title.
    """
    title = title_hint or ""
    body = ""

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html_text, "html.parser")
        # try meta title first
        mt = soup.find("meta", attrs={"property": "og:title"}) or soup.find("meta", attrs={"name": "twitter:title"})
        if mt and not title:
            title = mt.get("content") or "" # type: ignore
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)

        # crude main text: remove nav/header/footer/aside
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # choose largest <p> cluster
        ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        ps = [t for t in ps if len(t.split()) >= 3]
        if ps:
            body = _norm_ws(" ".join(ps))
        else:
            body = _norm_ws(soup.get_text(" "))

    else:
        body = _strip_html(html_text)
        if not title:
            title = " ".join(body.split()[:12])

    title = clean_text(title) # type: ignore
    body = clean_text(body)
    lang = _detect_lang(f"{title} {body}")

    art = Article(
        id=_mk_id(url or "", title, body[:128]),
        title=title[:500],
        body=body,
        url=url or "",
        published_at="",
        source=source,
        lang=lang,
        raw={"url": url or "", "html_len": len(html_text)},
    )
    return asdict(art)


# ----------------------------- self-test -----------------------------

if __name__ == "__main__":
    # RSS-like example
    rss_item = {
        "title": "Microsoft earnings beat expectations",
        "description": "<p>Revenue rose 12% driven by cloud.</p>",
        "link": "https://example.com/msft-earnings",
        "pubDate": "Tue, 16 Jan 2024 21:00:00 GMT",
        "source": "ExampleWire",
    }
    print(json.dumps(normalize_article(rss_item, source="ExampleWire"), indent=2)[:600])

    # Raw HTML example (fallback if BS4 missing)
    html_doc = "<html><head><title>Fed signals rate cut</title></head><body><p>Stocks rallied...</p></body></html>"
    print(json.dumps(parse_html(html_doc, url="https://example.com/fed"), indent=2)[:600])