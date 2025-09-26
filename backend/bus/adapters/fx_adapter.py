# backend/data/fx_adapters.py
"""
FX Market-Data Adapters
-----------------------
Providers:
  • yfinance (default): Yahoo symbols like "EURUSD=X", "USDJPY=X".
  • AlphaVantage (optional): free tier API (throttled) for spot & daily bars.

Capabilities
  • fetch_quote(pair)                    # spot (best-effort bid/ask/last)
  • fetch_ohlcv(pair, interval, ...)     # bars
  • fetch_cross(base, quote, via)        # derive cross via triangulation
  • convert(amount, from_ccy, to_ccy)    # convert using live quote
  • publish(envelope_or_list)            # emit to internal bus

Envelopes are SHA-256 hashed & optionally appended to Merkle ledger.

Symbols & Pairs
  • Pair format is "BASE/QUOTE" (e.g., "EUR/USD"), case-insensitive.
  • yfinance symbol mapping: "EURUSD=X", "USDJPY=X", etc.
  • AlphaVantage uses separate 'from_currency' & 'to_currency'.

Dependencies
  • yfinance (for provider="yfinance")  -> pip install yfinance
  • requests (for provider="alphavantage") -> pip install requests

Usage
-----
from backend.data.fx_adapters import FXAdapter, FXConfig

fx = FXAdapter(FXConfig(provider="yfinance", stream="STREAM_FX_MD"))
q = fx.fetch_quote("EUR/USD"); fx.publish(q)
bars = fx.fetch_ohlcv("EUR/USD", interval="1d", lookback=60); fx.publish(bars)
cross = fx.fetch_cross(base="EUR", quote="JPY", via="USD"); fx.publish(cross)
amt = fx.convert(1_000_000, "EUR", "USD")  # -> dict with converted amount & envelope
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------- optional deps ----------
try:
    import yfinance as yf  # type: ignore
    _HAS_YF = True
except Exception:
    _HAS_YF = False

try:
    import requests  # type: ignore
    _HAS_REQ = True
except Exception:
    _HAS_REQ = False

# ---------- bus hook ----------
try:
    from backend.bus.streams import publish_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload):
        head = payload if isinstance(payload, dict) else (payload[0] if payload else {})
        print(f"[stub publish_stream] {stream} <- {json.dumps(head, separators=(',',':'))[:220]}{'...' if isinstance(payload, list) and len(payload)>1 else ''}")

# ---------- optional ledger ----------
def _ledger_append(payload, ledger_path: Optional[str]):
    if not ledger_path:
        return
    try:
        from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
        MerkleLedger(ledger_path).append({"type": "fx_md", "payload": payload})
    except Exception:
        pass

# ---------- config & models ----------

@dataclass
class FXConfig:
    provider: str = "yfinance"               # "yfinance" | "alphavantage"
    stream: str = "STREAM_FX_MD"
    ledger_path: Optional[str] = None
    # AlphaVantage HTTP
    alphavantage_api_key: Optional[str] = None
    timeout_s: int = 15
    max_retries: int = 5
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 10.0
    user_agent: str = "HF-OS/fx-adapter/1.0"

@dataclass
class OHLCV:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

# ---------- adapter ----------

class FXAdapter:
    def __init__(self, cfg: FXConfig) -> None:
        self.cfg = cfg
        p = cfg.provider.lower().strip()
        if p not in ("yfinance", "alphavantage"):
            raise RuntimeError(f"Unsupported provider '{cfg.provider}'")
        if p == "yfinance" and not _HAS_YF:
            raise RuntimeError("yfinance not installed. Run: pip install yfinance")
        if p == "alphavantage" and (not _HAS_REQ or not cfg.alphavantage_api_key):
            raise RuntimeError("AlphaVantage requires requests + alphavantage_api_key")
        self.provider = p

    # ----------- Public: Quotes -----------

    def fetch_quote(self, pair: str) -> Dict[str, Any]:
        base, quote = _parse_pair(pair)
        if self.provider == "yfinance":
            sym = _yf_symbol(base, quote)
            tk = yf.Ticker(sym)
            fi = getattr(tk, "fast_info", {}) or {}
            last = _f(fi.get("last_price"))
            if last is None:
                try:
                    hist = tk.history(period="1d").tail(1)
                    if len(hist) > 0:
                        last = float(hist["Close"].iloc[-1])
                except Exception:
                    last = None
            payload = {
                "bid": _f(fi.get("last_bid")),
                "ask": _f(fi.get("last_ask")),
                "last": last,
                "provider_symbol": sym,
            }
            env = self._envelope(kind="quote", pair=f"{base}/{quote}", payload=payload)
            _ledger_append(env, self.cfg.ledger_path)
            return env

        # AlphaVantage
        data = self._av_exchange_rate(base, quote)
        payload = {
            "bid": None,  # AV endpoint does not provide bid/ask
            "ask": None,
            "last": _f(data.get("5. Exchange Rate")),
            "provider_symbol": f"{base}/{quote}",
        }
        env = self._envelope(kind="quote", pair=f"{base}/{quote}", payload=payload,
                             ts=_maybe_ms(data.get("6. Last Refreshed")))
        _ledger_append(env, self.cfg.ledger_path)
        return env

    # ----------- Public: OHLCV -----------

    def fetch_ohlcv(self, pair: str, *, interval: str = "1d", lookback: int = 120,
                    start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        interval (yfinance): "1m","2m","5m","15m","30m","60m","90m","1d","5d","1wk","1mo","3mo"
        AlphaVantage: supports "1d" via FX_DAILY (intraday has strict limits; not exposed here).
        """
        base, quote = _parse_pair(pair)
        if self.provider == "yfinance":
            sym = _yf_symbol(base, quote)
            hist = yf.download(sym, period=None if start else f"{lookback}d",
                               interval=interval, start=start, end=end,
                               progress=False, auto_adjust=False)
            if hist is None or len(hist) == 0:
                return []
            envs: List[Dict[str, Any]] = []
            for ts, row in hist.iterrows():
                ts_ms = int(getattr(ts, "value", int(time.time()*1000)) // 10**6)
                bar = OHLCV(
                    ts_ms=ts_ms,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume")) if "Volume" in row else None, # type: ignore
                )
                env = self._envelope(kind="ohlcv", pair=f"{base}/{quote}", payload=bar.__dict__, ts=ts_ms)
                _ledger_append(env, self.cfg.ledger_path)
                envs.append(env)
            return envs

        # AlphaVantage (daily only)
        bars = self._av_fx_daily(base, quote)
        envs = [self._envelope(kind="ohlcv", pair=f"{base}/{quote}", payload=b.__dict__, ts=b.ts_ms) for b in bars]
        for e in envs:
            _ledger_append(e, self.cfg.ledger_path)
        return envs

    # ----------- Public: Cross & Conversion -----------

    def fetch_cross(self, *, base: str, quote: str, via: str = "USD") -> Dict[str, Any]:
        """
        Cross-rate: base/quote = (base/via) * (via/quote).
        Uses two live quotes from the same provider.
        """
        b, q, v = base.upper(), quote.upper(), via.upper()
        if b == q:
            payload = {"base": b, "quote": q, "via": v, "rate": 1.0, "legs": []}
            env = self._envelope(kind="cross", pair=f"{b}/{q}", payload=payload)
            _ledger_append(env, self.cfg.ledger_path)
            return env

        q1 = self.fetch_quote(f"{b}/{v}")
        q2 = self.fetch_quote(f"{v}/{q}")

        r1 = _pick_rate(q1["payload"])
        r2 = _pick_rate(q2["payload"])
        rate = None if (r1 is None or r2 is None) else (r1 * r2)

        payload = {
            "base": b, "quote": q, "via": v, "rate": rate,
            "legs": [
                {"pair": f"{b}/{v}", "rate": r1, "hash": q1["hash"]},
                {"pair": f"{v}/{q}", "rate": r2, "hash": q2["hash"]},
            ],
        }
        env = self._envelope(kind="cross", pair=f"{b}/{q}", payload=payload)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    def convert(self, amount: float, from_ccy: str, to_ccy: str) -> Dict[str, Any]:
        """
        Convert 'amount' from_ccy → to_ccy using current spot (best-effort).
        """
        f, t = from_ccy.upper(), to_ccy.upper()
        if f == t:
            payload = {"from": f, "to": t, "amount": amount, "rate": 1.0, "converted": amount}
            env = self._envelope(kind="convert", pair=f"{f}/{t}", payload=payload)
            _ledger_append(env, self.cfg.ledger_path)
            return env

        q = self.fetch_quote(f"{f}/{t}")
        rate = _pick_rate(q["payload"])
        if rate is None:
            # Try cross via USD
            cross = self.fetch_cross(base=f, quote=t, via="USD")
            rate = cross["payload"]["rate"]

        converted = None if rate is None else amount * rate
        payload = {"from": f, "to": t, "amount": amount, "rate": rate, "converted": converted}
        env = self._envelope(kind="convert", pair=f"{f}/{t}", payload=payload)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    # ----------- Publish -----------

    def publish(self, env_or_envs):
        if isinstance(env_or_envs, list):
            for e in env_or_envs:
                publish_stream(self.cfg.stream, e)
        else:
            publish_stream(self.cfg.stream, env_or_envs)

    # ----------- AlphaVantage backend -----------

    def _http_get(self, url: str) -> Dict[str, Any]:
        delay = self.cfg.backoff_base_s
        headers = {"User-Agent": self.cfg.user_agent}
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                r = requests.get(url, headers=headers, timeout=self.cfg.timeout_s)  # type: ignore
                if r.status_code == 429:
                    time.sleep(min(self.cfg.backoff_cap_s, delay + 0.5))
                    delay = min(self.cfg.backoff_cap_s, delay * 2)
                    continue
                r.raise_for_status()
                j = r.json()
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

    def _av_exchange_rate(self, base: str, quote: str) -> Dict[str, Any]:
        api = self.cfg.alphavantage_api_key or ""
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base}&to_currency={quote}&apikey={api}"
        j = self._http_get(url)
        return (j or {}).get("Realtime Currency Exchange Rate", {}) or {}

    def _av_fx_daily(self, base: str, quote: str, lookback: int = 300) -> List[OHLCV]:
        api = self.cfg.alphavantage_api_key or ""
        url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={base}&to_symbol={quote}&outputsize=full&apikey={api}"
        j = self._http_get(url)
        ts = (j or {}).get("Time Series FX (Daily)", {}) or {}
        rows = list(ts.items())[:lookback][::-1]  # newest first → reverse
        out: List[OHLCV] = []
        for day, row in rows:
            out.append(OHLCV(
                ts_ms=_date_to_ms(day), # type: ignore
                open=float(row.get("1. open", 0.0)),
                high=float(row.get("2. high", 0.0)),
                low=float(row.get("3. low", 0.0)),
                close=float(row.get("4. close", 0.0)),
                volume=None,
            ))
        return out

    # ----------- Envelope -----------

    def _envelope(self, *, kind: str, pair: str, payload: Dict[str, Any], ts: Optional[int] = None) -> Dict[str, Any]:
        env = {
            "ts": int(ts or time.time() * 1000),
            "adapter": "fx",
            "provider": self.provider,
            "kind": kind,               # "quote" | "ohlcv" | "cross" | "convert"
            "pair": pair.upper(),
            "payload": payload,
            "version": 1,
        }
        env["hash"] = hashlib.sha256(json.dumps(env, separators=(",",":"), sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
        return env

# ---------- helpers ----------

def _parse_pair(pair: str) -> Tuple[str, str]:
    s = pair.replace("-", "/").replace("\\", "/").strip().upper()
    if "/" not in s:
        if len(s) == 6:
            return s[:3], s[3:]
        raise ValueError(f"Invalid FX pair: {pair}")
    a, b = s.split("/", 1)
    if len(a) != 3 or len(b) != 3:
        raise ValueError(f"Invalid FX pair: {pair}")
    return a, b

def _yf_symbol(base: str, quote: str) -> str:
    return f"{base}{quote}=X"

def _f(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def _maybe_ms(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    # Try ISO date "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
    try:
        import datetime as dt
        if len(s) >= 19 and s[4] == "-" and s[7] == "-":
            # with time
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return int(d.timestamp() * 1000)
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            d = dt.datetime.strptime(s[:10], "%Y-%m-%d")
            return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    except Exception:
        return None
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

def _pick_rate(payload: Dict[str, Any]) -> Optional[float]:
    """
    Choose a sensible rate from payload (bid/ask/last). If both bid/ask present, mid.
    """
    bid = _f(payload.get("bid"))
    ask = _f(payload.get("ask"))
    last = _f(payload.get("last"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    if last is not None and last > 0:
        return last
    return bid or ask

# ---------- script demo ----------
if __name__ == "__main__":
    cfg = FXConfig(provider="yfinance")
    fx = FXAdapter(cfg)
    print(json.dumps(fx.fetch_quote("EUR/USD"), indent=2))
    bars = fx.fetch_ohlcv("EUR/USD", interval="1d", lookback=5)
    print(json.dumps([b["payload"] for b in bars], indent=2))
    print(json.dumps(fx.fetch_cross(base="EUR", quote="JPY", via="USD"), indent=2))
    print(json.dumps(fx.convert(1000000, "EUR", "USD"), indent=2))