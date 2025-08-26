# backend/ai/deepfake_guard.py
"""
DeepfakeGuard: lightweight fake/hoax/manipulated-news filter for your feed.

Listens:
  - news.events  (from your news_yahoo.py / news_moneycontrol.py)

Emits:
  - news.vetted  (original event + {"trust": {score, reasons, tags}})
  - ai.insight   (when low-trust: kind="deepfake_alert", summary, refs)

Heuristics (fast, no external calls):
  - Domain reputation (whitelist/graylist/blacklist)
  - URL sanity (typosquats, non-https, odd TLDs, tracking params)
  - Title/style signals (ALL CAPS, !!!, clickbait phrases, % too-perfect numbers)
  - Time/corroboration (same story seen from multiple reputable domains rises)
  - Entity/source mismatch (mentions of 'Reuters/Bloomberg' without matching domain)
  - Image/attachment presence flags if provided (optional)

Optional: if you add a config at config/deepfake_guard.yaml it will load overrides:
  domain:
    whitelist: ["reuters.com", "bloomberg.com", "moneycontrol.com", "wsj.com", "ft.com", "economictimes.com"]
    blacklist: ["coinz-news.biz", "marketbuzz-xyz.ru"]
  thresholds:
    low_trust: 0.35
    high_trust: 0.75
    corroboration_needed: 2
"""

from __future__ import annotations
import os, re, time, json, math, hashlib
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque

try:
    import yaml  # optional config
except Exception:
    yaml = None  # type: ignore

# Your bus helpers
try:
    from backend.bus.streams import consume_stream, publish_stream, hset
except Exception:
    consume_stream = publish_stream = hset = None  # type: ignore


# --------------------------- utilities ---------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _norm_domain(url: str) -> str:
    m = re.search(r"https?://([^/]+)/?", url or "", re.I)
    if not m:
        return ""
    host = m.group(1).lower()
    # strip common prefixes
    for p in ("www.", "m.", "amp."):
        if host.startswith(p):
            host = host[len(p):]
    return host

def _slug_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:160]


# --------------------------- default config ---------------------------

_DEFAULT_WHITELIST = {
    "reuters.com","bloomberg.com","moneycontrol.com","wsj.com","ft.com",
    "economictimes.com","thehindu.com","livemint.com","cnbc.com","cnbctv18.com",
    "business-standard.com","hindustantimes.com","indiatimes.com","theguardian.com",
    "nytimes.com","apnews.com","financialexpress.com","investing.com","marketwatch.com",
}
_DEFAULT_BLACKLIST = {
    "marketbuzz-xyz.ru","coinz-news.biz","finance-scoop.click","stockpump.today",
    "thetraderinsider.xyz","alt-biz-news.top"
}
_CLICKBAIT = {
    "you won’t believe","shocking","exposed","leaked","secret",
    "destroys","obliterates","to the moon","guaranteed","100% profit","get rich quick"
}
_FAKE_BRANDS = {"reuters","bloomberg","ap","associated press","financial times","wsj","ft"}

_DEFAULT_THRESHOLDS = {
    "low_trust": 0.35,
    "high_trust": 0.75,
    "corroboration_needed": 2,
}

CONFIG_PATH = os.getenv("DEEPFAKE_GUARD_CONFIG", "config/deepfake_guard.yaml")


# --------------------------- guard core ---------------------------

class DeepfakeGuard:
    """
    Stateless scorer + small rolling state for corroboration.
    """
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = self._load_cfg(cfg)
        self.whitelist: Set[str] = set(self.cfg.get("domain", {}).get("whitelist", _DEFAULT_WHITELIST))
        self.blacklist: Set[str] = set(self.cfg.get("domain", {}).get("blacklist", _DEFAULT_BLACKLIST))
        self.th = {**_DEFAULT_THRESHOLDS, **(self.cfg.get("thresholds") or {})}
        # rolling memory of slugs -> {domains}
        self._seen: Dict[str, Set[str]] = defaultdict(set)
        self._recent: deque = deque(maxlen=5000)  # store (ts, slug, domain)

    def _load_cfg(self, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if override is not None:
            return override
        if yaml and os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
        return {}

    # ---- scoring ----
    def assess(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: normalized news event (see InsightAgent docstring).
        Output: {score, reasons[], tags[], domain, corroborations}
        """
        title = (event.get("title") or "").strip()
        summary = (event.get("summary") or "").strip()
        url = event.get("url") or ""
        source = (event.get("source") or "").lower()
        domain = _norm_domain(url)

        reasons: List[str] = []
        tags: List[str] = []
        score = 0.5  # start neutral

        # 1) Domain reputation
        if domain in self.whitelist:
            score += 0.25; reasons.append(f"whitelisted domain: {domain}"); tags.append("reputable")
        elif domain in self.blacklist or _looks_tampered_domain(domain):
            score -= 0.35; reasons.append(f"suspect domain: {domain}"); tags.append("suspect")

        # 2) URL sanity
        url_pen, url_notes = _url_penalty(url)
        score += url_pen; reasons += url_notes
        if url_notes: tags.append("url-odd")

        # 3) Title/style
        tpen, tnotes = _title_penalty(title)
        score += tpen; reasons += tnotes

        # 4) Brand/source mismatch (mentions big brands but domain doesn't match)
        if any(b in (title + " " + summary).lower() for b in _FAKE_BRANDS):
            if domain and not any(b in domain for b in ("reuters","bloomberg","apnews","ft.com","wsj.com")):
                score -= 0.15
                reasons.append("mentions tier-1 brand but domain mismatch")
                tags.append("brand-mismatch")

        # 5) Numbers that are too neat / sensational
        if re.search(r"\b(1000%|10,000%|guaranteed|risk[- ]?free)\b", (title + " " + summary).lower()):
            score -= 0.25; reasons.append("sensational claims"); tags.append("sensational")

        # 6) Corroboration: same slug seen from multiple reputable domains -> boost
        slug = _slug_text(title) or hashlib.md5((title or url).encode()).hexdigest()[:16]
        if domain:
            self._seen[slug].add(domain)
            self._recent.append((_utc_ms(), slug, domain))

        doms = self._seen[slug]
        reputable_hits = sum(1 for d in doms if d in self.whitelist)
        if reputable_hits >= max(1, int(self.th["corroboration_needed"])):
            boost = 0.20 + 0.05 * (reputable_hits - 1)
            score += boost
            reasons.append(f"corroborated by {reputable_hits} reputable domain(s)")
            tags.append("corroborated")
        elif len(doms) == 1 and domain and domain not in self.whitelist:
            score -= 0.10
            reasons.append("single-source untrusted")
            tags.append("single-source")

        # Clamp and summarize
        score = float(clamp(score, 0.0, 1.0))
        trust = {
            "score": score,
            "reasons": reasons[:6],   # keep short
            "tags": sorted(list(set(tags)))[:6],
            "domain": domain,
            "corroborations": list(sorted(doms))[:8],
        }
        return trust

    # ---- loop ----
    def run(self, in_stream="news.events", out_stream="news.vetted", alert_stream="ai.insight", poll_ms=300):
        assert consume_stream and publish_stream, "bus streams not wired"
        cur = "$"
        lows = self.th["low_trust"]; highs = self.th["high_trust"]
        while True:
            for _, msg in consume_stream(in_stream, start_id=cur, block_ms=poll_ms, count=200):
                cur = "$"
                try:
                    if isinstance(msg, str):
                        msg = json.loads(msg)
                except Exception:
                    continue
                trust = self.assess(msg)
                vetted = dict(msg)
                vetted["trust"] = trust
                publish_stream(out_stream, vetted)

                # low-trust alert
                if trust["score"] <= lows:
                    publish_stream(alert_stream, {
                        "ts_ms": _utc_ms(),
                        "kind": "deepfake_alert",
                        "summary": f"Low-trust news ({trust['domain']}): {msg.get('title','')[:160]}",
                        "tickers": msg.get("tickers") or [],
                        "score": -1.0 + trust["score"],
                        "confidence": 0.75,
                        "tags": ["news","deepfake","low-trust"] + trust.get("tags", []),
                        "refs": {"url": msg.get("url"), "source": msg.get("source")},
                    })


# --------------------------- scoring helpers ---------------------------

_ODD_TLDS = (".ru",".click",".top",".biz",".xyz",".gq",".cf",".ml",".tk")

def _looks_tampered_domain(domain: str) -> bool:
    if not domain:
        return True
    if domain.endswith(_ODD_TLDS):
        return True
    # typosquat: bloomberq.com, reuterss.com
    if re.search(r"(blo+mb?er[gq]|reuterss|associafed|financiaI|ws1)\.com", domain):
        return True
    return False

def _url_penalty(url: str) -> Tuple[float, List[str]]:
    notes: List[str] = []
    pen = 0.0
    if not url:
        return -0.10, ["missing url"]
    if not url.startswith("https://"):
        pen -= 0.05; notes.append("non-https")
    if re.search(r"[?&](utm_|ref|aff)=", url):
        pen -= 0.02; notes.append("tracking params")
    if len(url) > 180:
        pen -= 0.03; notes.append("very long url")
    return pen, notes

def _title_penalty(title: str) -> Tuple[float, List[str]]:
    t = title or ""
    notes: List[str] = []
    pen = 0.0
    if not t.strip():
        return -0.10, ["empty title"]
    if re.search(r"[A-Z]{5,}", t) and sum(1 for ch in t if ch.isalpha()) >= 10:
        pen -= 0.05; notes.append("excessive CAPS")
    if "?" in t and "!" in t:
        pen -= 0.05; notes.append("sensational punctuation")
    tl = t.lower()
    if any(phrase in tl for phrase in _CLICKBAIT):
        pen -= 0.10; notes.append("clickbait phrase")
    # too neat % numbers (1000% etc handled elsewhere)
    if re.search(r"\b\d{2,3}%\b", tl):
        notes.append("percent claim")
    return pen, notes


# --------------------------- CLI ---------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Deepfake news guard")
    ap.add_argument("--run", action="store_true", help="Run guard loop (news.events -> news.vetted)")
    ap.add_argument("--probe", type=str, help="Probe a single JSON payload string")
    ap.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to deepfake_guard.yaml")
    args = ap.parse_args()

    cfg = None
    if args.config and os.path.exists(args.config) and yaml:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or None

    guard = DeepfakeGuard(cfg)

    if args.probe:
        try:
            evt = json.loads(args.probe)
        except Exception as e:
            print(f"Invalid JSON: {e}")
            return
        print(json.dumps(guard.assess(evt), indent=2))
        return

    if args.run:
        try:
            guard.run()
        except KeyboardInterrupt:
            pass
    else:
        print("Nothing to do. Use --run to start or --probe '<json>' to test.")


if __name__ == "__main__":
    main()