# backend/data/equities_adapters.py
"""
Equities Market-Data Adapters
-----------------------------
Normalized envelopes for equities data from pluggable providers:
  • Quotes (NBBO-ish best effort), OHLCV bars, corporate actions, news.
  • Providers: yfinance (default), Polygon.io (if API key), AlphaVantage (if API key).

Envelope schema (all variants):
{
  "ts": <epoch_ms>,
  "adapter": "equities",
  "provider": "yfinance|polygon|alphavantage",
  "kind": "quote|ohlcv|corp_action|news",
  "symbol": "AAPL",
  "payload": {...},
  "version": 1,
  "hash": "sha256(...)"
}

Usage
-----
from backend.data.equities_adapters import EquitiesAdapter, EqConfig, OHLCV

cfg = EqConfig(provider="yfinance", stream="STREAM_EQUITIES_MD")
eq = EquitiesAdapter(cfg)

# Quote
env = eq.fetch_quote("AAPL")
eq.publish(env)

# OHLCV
envs = eq.fetch_ohlcv("AAPL", interval="1d", lookback=60)
eq.publish(envs)

# Corporate actions
acts = eq.fetch_corporate_actions("AAPL")
eq.publish(acts)

# Headlines
news = eq.fetch_news("AAPL", limit=20)
eq.publish(news)
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------- Optional deps ----------------
try:
    import yfinance as yf  # pip install yfinance
    _HAS_YF = True
except Exception:
    _HAS_YF = False

try:
    import requests  # used for Polygon/AlphaVantage
    _HAS_REQ = True
except Exception:
    _HAS_REQ = False

# ---------------- Bus hook ----------------
try:
    from backend.bus.streams import publish_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload):
        # Safe stub
        head = payload if isinstance(payload, dict) else (payload[0] if payload else {})
        print(f"[stub publish_stream] {stream} <- {json.dumps(head, separators=(',',':'))[:220]}{'...' if isinstance(payload, list) and len(payload)>1 else ''}")

# ---------------- Optional ledger ----------------
def _ledger_append(payload, ledger_path: Optional[str]):
    if not ledger_path:
        return
    try:
        from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
        MerkleLedger(ledger_path).append({"type": "equities_md", "payload": payload})
    except Exception:
        pass

# ---------------- Config & Models ----------------

@dataclass
class EqConfig:
    provider: str = "yfinance"           # "yfinance" | "polygon" | "alphavantage"
    stream: str = "STREAM_EQUITIES_MD"
    ledger_path: Optional[str] = None

    # HTTP (Polygon/AV)
    polygon_api_key: Optional[str] = None
    alphavantage_api_key: Optional[str] = None
    timeout_s: int = 15
    max_retries: int = 5
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 10.0
    user_agent: str = "HF-OS/equities-adapter/1.0"

@dataclass
class OHLCV:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

# ---------------- Adapter ----------------

class EquitiesAdapter:
    def __init__(self, cfg: EqConfig) -> None:
        self.cfg = cfg
        p = cfg.provider.lower().strip()
        if p not in ("yfinance", "polygon", "alphavantage"):
            raise RuntimeError(f"Unsupported provider '{cfg.provider}'")
        if p in ("polygon", "alphavantage") and not _HAS_REQ:
            raise RuntimeError("requests not installed. Run: pip install requests")
        if p == "yfinance" and not _HAS_YF:
            raise RuntimeError("yfinance not installed. Run: pip install yfinance")
        if p == "polygon" and not cfg.polygon_api_key:
            raise RuntimeError("Polygon provider requires polygon_api_key")
        if p == "alphavantage" and not cfg.alphavantage_api_key:
            raise RuntimeError("AlphaVantage provider requires alphavantage_api_key")
        self.provider = p

    # ---------- Public: Quotes ----------

    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.upper().strip()
        if self.provider == "yfinance":
            data = self._yf_quote(sym)
        elif self.provider == "polygon":
            data = self._poly_quote(sym)
        else:
            data = self._av_quote(sym)
        env = self._envelope(kind="quote", symbol=sym, payload=data)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    # ---------- Public: OHLCV ----------

    def fetch_ohlcv(self, symbol: str, *, interval: str = "1d", lookback: int = 60, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        interval: "1m","5m","15m","30m","60m","1d","1wk","1mo"
        lookback: number of bars if start/end not provided.
        """
        sym = symbol.upper().strip()
        bars: List[OHLCV]
        if self.provider == "yfinance":
            bars = self._yf_ohlcv(sym, interval=interval, lookback=lookback, start=start, end=end)
        elif self.provider == "polygon":
            bars = self._poly_ohlcv(sym, interval=interval, lookback=lookback, start=start, end=end)
        else:
            bars = self._av_ohlcv(sym, interval=interval, lookback=lookback)

        envs = [self._envelope(kind="ohlcv", symbol=sym, payload=bar.__dict__) for bar in bars]
        for e in envs:
            _ledger_append(e, self.cfg.ledger_path)
        return envs

    # ---------- Public: Corporate Actions ----------

    def fetch_corporate_actions(self, symbol: str) -> List[Dict[str, Any]]:
        sym = symbol.upper().strip()
        if self.provider == "polygon":
            acts = self._poly_corp_actions(sym)
        elif self.provider == "yfinance":
            acts = self._yf_corp_actions(sym)
        else:
            acts = self._av_corp_actions(sym)  # limited, dividends/splits
        envs = [self._envelope(kind="corp_action", symbol=sym, payload=a) for a in acts]
        for e in envs:
            _ledger_append(e, self.cfg.ledger_path)
        return envs

    # ---------- Public: News ----------

    def fetch_news(self, symbol: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        sym = symbol.upper().strip()
        if self.provider == "polygon":
            news = self._poly_news(sym, limit=limit)
        elif self.provider == "yfinance":
            news = self._yf_news(sym, limit=limit)
        else:
            news = self._av_news(sym, limit=limit)
        envs = [self._envelope(kind="news", symbol=sym, payload=n) for n in news]
        for e in envs:
            _ledger_append(e, self.cfg.ledger_path)
        return envs

    # ---------- Publish ----------

    def publish(self, env_or_envs):
        if isinstance(env_or_envs, list):
            for e in env_or_envs:
                publish_stream(self.cfg.stream, e)
        else:
            publish_stream(self.cfg.stream, env_or_envs)

    # ---------- yfinance backend ----------

    def _yf_quote(self, symbol: str) -> Dict[str, Any]:
        tk = yf.Ticker(symbol)
        q = tk.fast_info if hasattr(tk, "fast_info") else {}
        # Fallback to info if needed (slower)
        bid = float(q.get("last_bid", float("nan"))) if q else float("nan") # type: ignore
        ask = float(q.get("last_ask", float("nan"))) if q else float("nan") # type: ignore
        last = float(q.get("last_price", float("nan"))) if q else float("nan") # type: ignore
        if math.isnan(last):
            try:
                last = float(getattr(tk.history(period="1d").tail(1)["Close"], "iloc", lambda x=0: float("nan"))())
            except Exception:
                pass
        return {
            "bid": None if math.isnan(bid) else bid,
            "ask": None if math.isnan(ask) else ask,
            "last": None if math.isnan(last) else last,
            "exchange": getattr(tk, "info", {}).get("exchange", None),
        }

    def _yf_ohlcv(self, symbol: str, *, interval: str, lookback: int, start: Optional[str], end: Optional[str]) -> List[OHLCV]:
        # yfinance interval mapping is mostly 1m,2m,5m,15m,30m,60m,90m,1d,5d,1wk,1mo,3mo
        hist = yf.download(symbol, period=None if start else f"{lookback}d", interval=interval, start=start, end=end, progress=False, auto_adjust=False)
        if hist is None or len(hist) == 0:
            return []
        out: List[OHLCV] = []
        for ts, row in hist.iterrows():
            ts_ms = int(getattr(ts, "value", int(time.time()*1000)) // 10**6)
            out.append(OHLCV(
                ts_ms=ts_ms,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0.0)),
            ))
        return out

    def _yf_corp_actions(self, symbol: str) -> List[Dict[str, Any]]:
        tk = yf.Ticker(symbol)
        acts: List[Dict[str, Any]] = []
        try:
            cal = tk.actions  # dividends & splits
            if cal is not None and len(cal) > 0:
                for ts, row in cal.iterrows():
                    item = {"ts": int(getattr(ts, "value", 0) // 10**6)}
                    if "Dividends" in row and not math.isnan(row["Dividends"]):
                        item |= {"type": "dividend", "amount": float(row["Dividends"])}
                    if "Stock Splits" in row and not math.isnan(row["Stock Splits"]) and row["Stock Splits"] > 0:
                        item |= {"type": "split", "ratio": float(row["Stock Splits"])}
                    if len(item) > 1:
                        acts.append(item)
        except Exception:
            pass
        return acts

    def _yf_news(self, symbol: str, *, limit: int) -> List[Dict[str, Any]]:
        try:
            tk = yf.Ticker(symbol)
            items = getattr(tk, "news", []) or []
        except Exception:
            items = []
        out: List[Dict[str, Any]] = []
        for it in items[:limit]:
            out.append({
                "id": it.get("uuid") or it.get("id"),
                "title": it.get("title"),
                "publisher": it.get("publisher"),
                "link": it.get("link") or it.get("url"),
                "providerPublishTime": int(it.get("providerPublishTime", 0)) * 1000 if isinstance(it.get("providerPublishTime", 0), (int, float)) else None,
            })
        return out

    # ---------- Polygon backend ----------

    def _poly_quote(self, symbol: str) -> Dict[str, Any]:
        # Use /v2/last/nbbo/{stocksTicker} for NBBO, fallback to /v2/last/trade
        api = self.cfg.polygon_api_key or ""
        base = "https://api.polygon.io"
        # Try NBBO
        url_nbbo = f"{base}/v2/last/nbbo/{symbol}?apiKey={api}"
        j = self._http_get(url_nbbo)
        if j and j.get("status") == "success":
            res = j.get("results", {})
            return {
                "bid": _f(res.get("bid", {}).get("price")),
                "ask": _f(res.get("ask", {}).get("price")),
                "last": _f(res.get("last", {}).get("price")) or _f(res.get("bid", {}).get("price")) or _f(res.get("ask", {}).get("price")),
                "exchange": "NBBO",
            }
        # Fallback trade
        url_trade = f"{base}/v2/last/trade/{symbol}?apiKey={api}"
        j = self._http_get(url_trade)
        res = (j or {}).get("results", {})
        return {"bid": None, "ask": None, "last": _f(res.get("p")), "exchange": res.get("e")}

    def _poly_ohlcv(self, symbol: str, *, interval: str, lookback: int, start: Optional[str], end: Optional[str]) -> List[OHLCV]:
        api = self.cfg.polygon_api_key or ""
        base = "https://api.polygon.io"
        # Map interval to polygon aggregates
        ivmap = {"1m":"minute","5m":"minute","15m":"minute","30m":"minute","60m":"minute",
                 "1d":"day","1wk":"week","1mo":"month"}
        unit = ivmap.get(interval, "day")
        mult = {"1m":1,"5m":5,"15m":15,"30m":30,"60m":60,"1d":1,"1wk":1,"1mo":1}[interval if interval in ivmap else "1d"]
        if start and end:
            url = f"{base}/v2/aggs/ticker/{symbol}/range/{mult}/{unit}/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={api}"
        else:
            url = f"{base}/v2/aggs/ticker/{symbol}/range/{mult}/{unit}/now-{lookback}day/now?adjusted=true&sort=asc&limit=50000&apiKey={api}"
        j = self._http_get(url)
        out: List[OHLCV] = []
        for r in (j or {}).get("results", []) or []:
            out.append(OHLCV(
                ts_ms=int(r.get("t", 0)),
                open=_f(r.get("o")) or 0.0,
                high=_f(r.get("h")) or 0.0,
                low=_f(r.get("l")) or 0.0,
                close=_f(r.get("c")) or 0.0,
                volume=float(r.get("v") or 0.0),
            ))
        return out

    def _poly_corp_actions(self, symbol: str) -> List[Dict[str, Any]]:
        api = self.cfg.polygon_api_key or ""
        base = "https://api.polygon.io"
        url = f"{base}/vX/reference/corporate-actions?ticker={symbol}&limit=1000&apiKey={api}"
        j = self._http_get(url)
        out: List[Dict[str, Any]] = []
        for r in (j or {}).get("results", []) or []:
            out.append({
                "ts": self._parse_poly_date(r.get("declaration_date")) or self._parse_poly_date(r.get("execution_date")) or None,
                "type": r.get("action_type"),
                "details": r,
            })
        return out

    def _poly_news(self, symbol: str, *, limit: int) -> List[Dict[str, Any]]:
        api = self.cfg.polygon_api_key or ""
        url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit={int(limit)}&apiKey={api}"
        j = self._http_get(url)
        out: List[Dict[str, Any]] = []
        for r in (j or {}).get("results", []) or []:
            out.append({
                "id": r.get("id"),
                "title": r.get("title"),
                "publisher": (r.get("publisher") or {}).get("name"),
                "link": r.get("article_url") or r.get("url"),
                "providerPublishTime": self._parse_poly_date(r.get("published_utc")),
            })
        return out

    @staticmethod
    def _parse_poly_date(s: Optional[str]) -> Optional[int]:
        # Polygon timestamps are ISO8601; we convert to ms (no timezone parsing to keep deps zero; rely on "Z")
        if not s:
            return None
        try:
            # yyyy-mm-dd or full iso
            if len(s) == 10 and s[4] == "-" and s[7] == "-":
                import datetime as dt
                d = dt.datetime.strptime(s, "%Y-%m-%d")
                return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
            if s.endswith("Z"):
                import datetime as dt
                d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
                return int(d.timestamp() * 1000)
        except Exception:
            return None
        return None

    # ---------- AlphaVantage backend (quotes/ohlcv/news-lite) ----------

    def _av_quote(self, symbol: str) -> Dict[str, Any]:
        api = self.cfg.alphavantage_api_key or ""
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api}"
        j = self._http_get(url)
        gq = (j or {}).get("Global Quote", {})
        return {
            "bid": None,
            "ask": None,
            "last": _f(gq.get("05. price")),
            "exchange": None,
        }

    def _av_ohlcv(self, symbol: str, *, interval: str, lookback: int) -> List[OHLCV]:
        api = self.cfg.alphavantage_api_key or ""
        if interval == "1d":
            fn = "TIME_SERIES_DAILY_ADJUSTED"
            url = f"https://www.alphavantage.co/query?function={fn}&symbol={symbol}&outputsize=full&apikey={api}"
            j = self._http_get(url)
            ts = j.get("Time Series (Daily)", {}) if j else {}
            rows = list(ts.items())[:lookback][::-1]  # most recent first; we reverse to asc
            out: List[OHLCV] = []
            for day, row in rows:
                out.append(OHLCV(
                    ts_ms=_date_to_ms(day), # type: ignore
                    open=_f(row.get("1. open")) or 0.0,
                    high=_f(row.get("2. high")) or 0.0,
                    low=_f(row.get("3. low")) or 0.0,
                    close=_f(row.get("4. close")) or 0.0,
                    volume=float(row.get("6. volume") or row.get("5. volume") or 0.0),
                ))
            return out
        else:
            # AlphaVantage intraday
            step = {"1m":"1min","5m":"5min","15m":"15min","30m":"30min","60m":"60min"}.get(interval, "5min")
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={step}&outputsize=full&apikey={api}"
            j = self._http_get(url)
            ts = j.get(f"Time Series ({step})", {}) if j else {}
            rows = list(ts.items())[:lookback][::-1]
            out: List[OHLCV] = []
            for tstamp, row in rows:
                out.append(OHLCV(
                    ts_ms=_datetime_to_ms(tstamp), # type: ignore
                    open=_f(row.get("1. open")) or 0.0,
                    high=_f(row.get("2. high")) or 0.0,
                    low=_f(row.get("3. low")) or 0.0,
                    close=_f(row.get("4. close")) or 0.0,
                    volume=float(row.get("5. volume") or 0.0),
                ))
            return out

    def _av_corp_actions(self, symbol: str) -> List[Dict[str, Any]]:
        # AV doesn't have a single corp-actions endpoint; we compose from dividends + splits
        api = self.cfg.alphavantage_api_key or ""
        out: List[Dict[str, Any]] = []
        # Dividends
        url_div = f"https://www.alphavantage.co/query?function=DIVIDENDS&symbol={symbol}&apikey={api}"
        j = self._http_get(url_div)
        for r in (j or {}).get("data", []) or []:
            out.append({"ts": _date_to_ms(r.get("paymentDate")), "type": "dividend", "amount": _f(r.get("dividend"))})
        # Splits
        url_split = f"https://www.alphavantage.co/query?function=STOCK_SPLITS&symbol={symbol}&apikey={api}"
        j2 = self._http_get(url_split)
        for r in (j2 or {}).get("data", []) or []:
            out.append({"ts": _date_to_ms(r.get("declaredDate")), "type": "split", "ratio": _ratio(r)})
        return out

    def _av_news(self, symbol: str, *, limit: int) -> List[Dict[str, Any]]:
        # AV news endpoint
        api = self.cfg.alphavantage_api_key or ""
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit={int(limit)}&apikey={api}"
        j = self._http_get(url)
        out: List[Dict[str, Any]] = []
        for it in (j or {}).get("feed", []) or []:
            out.append({
                "id": it.get("url"),
                "title": it.get("title"),
                "publisher": it.get("source"),
                "link": it.get("url"),
                "providerPublishTime": _datetime_to_ms(it.get("time_published")),
            })
        return out

    # ---------- HTTP + Retry ----------

    def _http_get(self, url: str) -> Dict[str, Any]:
        delay = self.cfg.backoff_base_s
        headers = {"User-Agent": self.cfg.user_agent}
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                r = requests.get(url, headers=headers, timeout=self.cfg.timeout_s)  # type: ignore
                if r.status_code == 429:
                    # rate limited; backoff harder
                    time.sleep(min(self.cfg.backoff_cap_s, delay + 0.5))
                    delay = min(self.cfg.backoff_cap_s, delay * 2)
                    continue
                r.raise_for_status()
                j = r.json()
                # Handle AV throttling w/ Note
                if isinstance(j, dict) and ("Note" in j or "Information" in j):
                    time.sleep(min(self.cfg.backoff_cap_s, delay + 0.5))
                    delay = min(self.cfg.backoff_cap_s, delay * 2)
                    continue
                return j
            except Exception:
                if attempt >= self.cfg.max_retries:
                    raise
                time.sleep(min(self.cfg.backoff_cap_s, delay + 0.5))
                delay = min(self.cfg.backoff_cap_s, delay * 2)
        return {}

    # ---------- Envelope ----------

    def _envelope(self, *, kind: str, symbol: str, payload: Dict[str, Any], ts: Optional[int] = None) -> Dict[str, Any]:
        env = {
            "ts": int(ts or time.time() * 1000),
            "adapter": "equities",
            "provider": self.provider,
            "kind": kind,           # "quote" | "ohlcv" | "corp_action" | "news"
            "symbol": symbol,
            "payload": payload,
            "version": 1,
        }
        env["hash"] = hashlib.sha256(json.dumps(env, separators=(",", ":"), sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
        return env

# ---------------- Small utils ----------------

def _f(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def _date_to_ms(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        import datetime as dt
        d = dt.datetime.strptime(s[:10], "%Y-%m-%d")
        return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    except Exception:
        return None

def _datetime_to_ms(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        import datetime as dt
        # Accept "YYYYMMDDTHHMMSS" (AlphaVantage) or ISO8601
        if "T" in s and len(s) >= 15 and s[8] == "T":
            # AlphaVantage format: 20240131T133000
            d = dt.datetime.strptime(s[:15], "%Y%m%dT%H%M%S")
            return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        if s.endswith("Z"):
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return int(d.timestamp() * 1000)
        # Fallback: try full ISO
        d = dt.datetime.fromisoformat(s)
        return int(d.timestamp() * 1000)
    except Exception:
        return None

def _ratio(r: Dict[str, Any]) -> Optional[float]:
    try:
        a = float((r.get("numerator") or r.get("to") or 0.0))
        b = float((r.get("denominator") or r.get("for") or 1.0))
        if b == 0:
            return None
        return a / b
    except Exception:
        return None

# ---------------- Script demo ----------------
if __name__ == "__main__":
    # Minimal quick test (yfinance)
    cfg = EqConfig(provider="yfinance")
    eq = EquitiesAdapter(cfg)
    print(json.dumps(eq.fetch_quote("AAPL"), indent=2))
    bars = eq.fetch_ohlcv("AAPL", interval="1d", lookback=5)
    print(json.dumps([b["payload"] for b in bars], indent=2))
    print(json.dumps(eq.fetch_news("AAPL")[:2], indent=2))