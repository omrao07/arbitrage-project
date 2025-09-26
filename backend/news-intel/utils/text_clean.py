# news-intel/utils/text_clean.py
"""
Text cleaning utilities for news-intel pipeline.

Features
--------
- Unicode normalization (NFKC).
- HTML tag stripping (BeautifulSoup if available, fallback regex).
- Whitespace normalization.
- Optional lowercasing, stopword removal, stemming.
- Boilerplate phrase removal.

Use
---
from news_intel.utils.text_clean import clean_text, strip_html

txt = clean_text("<p>Microsoft earnings beat expectations!</p>")
"""

from __future__ import annotations

import re
import unicodedata
import html
from typing import List

# optional deps
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

try:
    import nltk
    from nltk.corpus import stopwords  # type: ignore
    from nltk.stem import PorterStemmer  # type: ignore
    _HAVE_NLTK = True
except Exception:
    _HAVE_NLTK = False


# ------------------ regexes ------------------

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")
_BOILERPLATE = [
    "click here to read more",
    "all rights reserved",
    "subscribe to our newsletter",
    "follow us on twitter",
    "©",
]


# ------------------ basics ------------------

def normalize_unicode(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def normalize_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def strip_html(s: str) -> str:
    if not s:
        return ""
    if BeautifulSoup is not None:
        soup = BeautifulSoup(s, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return normalize_ws(soup.get_text(" "))
    return normalize_ws(_TAG_RE.sub(" ", s))


# ------------------ main cleaning ------------------

def clean_text(
    s: str,
    *,
    lowercase: bool = False,
    remove_boilerplate: bool = True,
    remove_stopwords: bool = False,
    stem: bool = False,
) -> str:
    """Full pipeline: normalize unicode, unescape HTML entities, strip tags, clean ws."""
    if not s:
        return ""
    s = normalize_unicode(s)
    s = html.unescape(s)
    s = strip_html(s)
    if lowercase:
        s = s.lower()
    if remove_boilerplate:
        for bp in _BOILERPLATE:
            s = re.sub(bp, "", s, flags=re.I)
    if remove_stopwords and _HAVE_NLTK:
        try:
            sw = set(stopwords.words("english"))
        except Exception:
            sw = set()
        tokens = [t for t in s.split() if t.lower() not in sw]
        s = " ".join(tokens)
    if stem and _HAVE_NLTK:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in s.split()]
        s = " ".join(tokens)
    return normalize_ws(s)


def clean_batch(texts: List[str], **kwargs) -> List[str]:
    return [clean_text(t, **kwargs) for t in texts]


# ------------------ self-test ------------------

if __name__ == "__main__":
    raw = "<html><body><h1>Breaking News</h1><p>Click here to read more about Microsoft earnings beat expectations!</p></body></html>"
    print(clean_text(raw))
    print(clean_text(raw, lowercase=True, remove_stopwords=True, stem=True))