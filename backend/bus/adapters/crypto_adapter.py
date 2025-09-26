# backend/data/crypto_adapter.py
"""
Crypto Market-Data Adapter
--------------------------
Purpose
  • Pull or stream PUBLIC crypto market data (quotes, order books, trades).
  • Normalize into audit-friendly envelopes with deterministic hashes.
  • Publish onto the internal data bus for strategies / dashboards.

Features
  • REST polling via CCXT (ticker, orderbook, recent trades).
  • Optional WebSocket streaming (binance, bybit, okx) if `websockets` is installed.
  • Robust retries with exponential backoff + jitter.
  • Strict symbol normalization (BTC/USDT style).
  • Tamper-evident envelopes (SHA-256 over canonical JSON).
  • Optional ledger append using backend.audit.merkle_ledger.MerkleLedger.

Dependencies
  • ccxt  (pip install ccxt)
  • Optional streaming: websockets (pip install websockets)

Bus Hook
  • publish_stream(stream, payload) is imported from backend.bus.streams
    (falls back to a simple print stub if not available).

Usage
-----
from backend.data.crypto_adapter import CryptoAdapter, CryptoConfig

cfg = CryptoConfig(exchange="binance", stream="STREAM_CRYPTO_MD")
adapter = CryptoAdapter(cfg)

env = adapter.fetch_ticker("BTC/USDT")    # REST ticker
adapter.publish(env)

# Orderbook (L2, default 50):
env = adapter.fetch_orderbook("BTC/USDT", depth=50)
adapter.publish(env)

# Trades (recent):
envs = adapter.fetch_trades("BTC/USDT", limit=100)
for e in envs: adapter.publish(e)

# Stream trades (WebSocket) — runs until cancelled:
# for env in adapter.stream_trades("BTC/USDT"):
#     adapter.publish(env)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterator, List, Optional, Sequence

# ---------------- Optional dependencies ----------------
try:
    import ccxt  # type: ignore
    _HAS_CCXT = True
except Exception:
    _HAS_CCXT = False

try:
    import websockets  # type: ignore
    _HAS_WS = True
except Exception:
    _HAS_WS = False

# ---------------- Bus hook ----------------
try:
    from backend.bus.streams import publish_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        print(f"[stub publish_stream] {stream} <- {json.dumps(payload, separators=(',',':'))[:200]}...")

# ---------------- Optional ledger ----------------
def _ledger_append(payload: Dict[str, Any], ledger_path: Optional[str]) -> None:
    if not ledger_path:
        return
    try:
        from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
        MerkleLedger(ledger_path).append({"type": "crypto_md", "payload": payload})
    except Exception:
        pass

# ---------------- Config ----------------

@dataclass
class CryptoConfig:
    exchange: str = "binance"         # binance | okx | bybit | coinbase | kraken | ...
    stream: str = "STREAM_CRYPTO_MD"  # bus stream name
    user_agent: str = "HF-OS/crypto-adapter/1.0"
    timeout_ms: int = 15000           # ccxt request timeout
    api_key: Optional[str] = None     # not used (public data), left for future
    api_secret: Optional[str] = None
    sandbox: bool = False             # some exchanges support sandbox
    rate_limit: bool = True           # ccxt enableRateLimit
    max_retries: int = 5
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 10.0
    ledger_path: Optional[str] = None # optional audit ledger append
    ws_trades_enabled: bool = True    # allow WS path when available

# ---------------- Adapter ----------------

class CryptoAdapter:
    def __init__(self, cfg: CryptoConfig) -> None:
        if not _HAS_CCXT:
            raise RuntimeError("ccxt not installed. Run: pip install ccxt")
        self.cfg = cfg
        self._ex = self._mk_exchange(cfg)

    # ---------- Public REST methods ----------

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        sym = self._normalize_symbol(symbol)
        data = self._retry(lambda: self._ex.fetch_ticker(sym))
        # ccxt ticker fields vary; normalize essential ones
        ts = int(self._now_ms())
        env = self._envelope(
            kind="ticker",
            exchange=self._ex.id,
            symbol=sym,
            payload={
                "bid": _get_num(data, "bid"), # type: ignore
                "ask": _get_num(data, "ask"), # type: ignore
                "last": _get_num(data, "last"), # type: ignore
                "quoteVolume": _get_num(data, "quoteVolume"), # type: ignore
                "baseVolume": _get_num(data, "baseVolume"), # type: ignore
                "info": data.get("info", {}), # type: ignore
            },
            ts=data.get("timestamp", ts) or ts, # type: ignore
        )
        _ledger_append(env, self.cfg.ledger_path)
        return env

    def fetch_orderbook(self, symbol: str, depth: int = 50) -> Dict[str, Any]:
        sym = self._normalize_symbol(symbol)
        depth = int(depth)
        data = self._retry(lambda: self._ex.fetch_order_book(sym, limit=depth))
        ts = int(data.get("timestamp") or self._now_ms()) # type: ignore
        # Trim to requested depth (some exchs return more)
        bids = data.get("bids", [])[:depth] # type: ignore
        asks = data.get("asks", [])[:depth] # type: ignore
        env = self._envelope(
            kind="orderbook",
            exchange=self._ex.id,
            symbol=sym,
            payload={"bids": bids, "asks": asks},
            ts=ts,
        )
        _ledger_append(env, self.cfg.ledger_path)
        return env

    def fetch_trades(self, symbol: str, since_ms: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        sym = self._normalize_symbol(symbol)
        limit = int(limit)
        data = self._retry(lambda: self._ex.fetch_trades(sym, since=since_ms, limit=limit))
        out: List[Dict[str, Any]] = []
        for t in data or []:
            env = self._envelope(
                kind="trade",
                exchange=self._ex.id,
                symbol=sym,
                payload={
                    "id": t.get("id"),
                    "side": t.get("side"),
                    "price": _get_num(t, "price"),
                    "amount": _get_num(t, "amount"),
                    "cost": _get_num(t, "cost"),
                },
                ts=t.get("timestamp") or self._now_ms(),
            )
            _ledger_append(env, self.cfg.ledger_path)
            out.append(env)
        return out

    # ---------- Optional WebSocket trades stream ----------
    # Yields normalized trade envelopes indefinitely (until CancelledError)

    def stream_trades(self, symbol: str) -> Iterator[Dict[str, Any]]:
        """
        Synchronous generator wrapper around an async WS client for common exchanges.
        Requires `websockets`. Supports: binance, okx, bybit.
        """
        if not self.cfg.ws_trades_enabled:
            raise RuntimeError("WebSocket streaming disabled in config.")
        if not _HAS_WS:
            raise RuntimeError("websockets not installed. Run: pip install websockets")
        sym = self._normalize_symbol(symbol)
        exid = self._ex.id

        # Map ccxt symbol → exchange channel symbol
        if exid == "binance":
            chan = sym.replace("/", "").lower() + "@trade"  # e.g., btcusdt@trade
            url = f"wss://stream.binance.com:9443/ws/{chan}"
            parse = _parse_binance_trade
        elif exid == "okx":
            # OKX: args=[{"channel":"trades","instId":"BTC-USDT"}]
            inst = sym.replace("/", "-")
            url = "wss://ws.okx.com:8443/ws/v5/public"
            parse = _parse_okx_trade(inst)
        elif exid == "bybit":
            inst = sym.replace("/", "")
            url = f"wss://stream.bybit.com/v5/public/spot"
            parse = _parse_bybit_trade(inst)
        else:
            raise RuntimeError(f"WebSocket trades not implemented for exchange: {exid}")

        # Run the async consumer and yield events
        loop = _ensure_group() # type: ignore
        queue: asyncio.Queue = asyncio.Queue()

        async def runner():
            backoff = self.cfg.backoff_base_s
            while True:
                try:
                    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                        if exid == "okx":
                            sub = {"op": "subscribe", "args": [{"channel": "trades", "instId": inst}]}
                            await ws.send(json.dumps(sub))
                        elif exid == "bybit":
                            # bybit v5 public subscriptions use a JSON op
                            req = {"op": "subscribe", "args": [f"publicTrade.{inst}"]}
                            await ws.send(json.dumps(req))

                        backoff = self.cfg.backoff_base_s  # reset on connect
                        async for msg in ws:
                            try:
                                evt = parse(msg)
                                if evt:
                                    env = self._envelope(kind="trade", exchange=exid, symbol=sym, payload=evt, ts=evt.get("ts", self._now_ms()))
                                    _ledger_append(env, self.cfg.ledger_path)
                                    await queue.put(env)
                            except Exception:
                                # swallow parse error, continue
                                continue
                except asyncio.CancelledError:
                    break
                except Exception:
                    # exponential backoff with jitter
                    await asyncio.sleep(min(self.cfg.backoff_cap_s, backoff + random.random()))
                    backoff = min(self.cfg.backoff_cap_s, backoff * 2)

        task = loop.create_task(runner())

        try:
            while True:
                env = loop.run_until_complete(queue.get())
                yield env
        except KeyboardInterrupt:
            pass
        finally:
            task.cancel()
            try:
                loop.run_until_complete(task)
            except Exception:
                pass

    # ---------- Publish ----------

    def publish(self, env_or_envs):
        if isinstance(env_or_envs, list):
            for e in env_or_envs:
                publish_stream(self.cfg.stream, e)
        else:
            publish_stream(self.cfg.stream, env_or_envs)

    # ---------- Internals ----------

    def _mk_exchange(self, cfg: CryptoConfig):
        exid = cfg.exchange.lower()
        if not hasattr(ccxt, exid):
            raise RuntimeError(f"Exchange '{cfg.exchange}' not supported by ccxt.")
        klass = getattr(ccxt, exid)
        ex = klass({
            "timeout": cfg.timeout_ms,
            "enableRateLimit": cfg.rate_limit,
            "userAgent": cfg.user_agent,
            "apiKey": cfg.api_key or "",
            "secret": cfg.api_secret or "",
        })
        if cfg.sandbox and hasattr(ex, "set_sandbox_mode"):
            try:
                ex.set_sandbox_mode(True)
            except Exception:
                pass
        return ex

    def _retry(self, fn):
        """
        Generic retry with exponential backoff + jitter for REST calls.
        """
        delay = self.cfg.backoff_base_s
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return fn()
            except Exception as e:
                if attempt >= self.cfg.max_retries:
                    raise
                time.sleep(min(self.cfg.backoff_cap_s, delay + random.random()))
                delay = min(self.cfg.backoff_cap_s, delay * 2)

    def _normalize_symbol(self, symbol: str) -> str:
        s = symbol.strip().upper()
        # Common CCXT symbols are already "BASE/QUOTE" e.g. BTC/USDT.
        if "/" not in s and len(s) >= 6:
            # Try to split common endings
            for q in ("USDT", "USD", "USDC", "BTC", "ETH", "EUR", "JPY"):
                if s.endswith(q):
                    base = s[:-len(q)]
                    return f"{base}/{q}"
        return s

    def _envelope(self, *, kind: str, exchange: str, symbol: str, payload: Dict[str, Any], ts: Optional[int] = None) -> Dict[str, Any]:
        base, quote = (symbol.split("/") + [""])[:2]
        env = {
            "ts": int(ts or self._now_ms()),
            "adapter": "crypto",
            "kind": kind,                 # "ticker" | "orderbook" | "trade"
            "exchange": exchange,
            "symbol": symbol,
            "base": base,
            "quote": quote,
            "payload": payload,
            "version": 1,
        }
        env["hash"] = hashlib.sha256(_canon(env)).hexdigest()
        return env

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

# ---------------- Utils ----------------

def _get_num(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    try:
        return None if v is None else float(v)
    except Exception:
        return None

def _canon(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False, default=str).encode()

# ---------------- WS Parsers ----------------

def _parse_binance_trade(msg: Any) -> Optional[Dict[str, Any]]:
    try:
        if isinstance(msg, (bytes, bytearray)):
            msg = msg.decode("utf-8")
        obj = json.loads(msg)
        # Expected schema: { "e":"trade","E":..., "s":"BTCUSDT", "t":12345, "p":"0.001", "q":"100", "m":true }
        if "e" in obj and obj.get("e") == "trade":
            return {
                "id": obj.get("t"),
                "price": float(obj.get("p")),
                "amount": float(obj.get("q")),
                "side": "sell" if obj.get("m") else "buy",  # m=true means buyer is market maker => sell
                "ts": int(obj.get("T") or obj.get("E") or time.time() * 1000),
            }
        # Raw stream payload (without keys) sometimes wraps as {"stream": "...", "data": {...}}
        if "data" in obj and obj.get("data", {}).get("e") == "trade":
            d = obj["data"]
            return {
                "id": d.get("t"),
                "price": float(d.get("p")),
                "amount": float(d.get("q")),
                "side": "sell" if d.get("m") else "buy",
                "ts": int(d.get("T") or d.get("E") or time.time() * 1000),
            }
    except Exception:
        return None
    return None

def _parse_okx_trade(inst: str):
    def _inner(msg: Any) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(msg, (bytes, bytearray)):
                msg = msg.decode("utf-8")
            obj = json.loads(msg)
            # OKX trades push: {"arg":{"channel":"trades","instId":"BTC-USDT"},"data":[{"instId":"BTC-USDT","px":"...","sz":"...","side":"buy","ts":"..."}]}
            if obj.get("arg", {}).get("channel") == "trades" and obj.get("arg", {}).get("instId") == inst:
                data = obj.get("data", [])
                if not data:
                    return None
                d0 = data[0]
                return {
                    "id": d0.get("tradeId"),
                    "price": float(d0.get("px")),
                    "amount": float(d0.get("sz")),
                    "side": d0.get("side"),
                    "ts": int(d0.get("ts")),
                }
        except Exception:
            return None
        return None
    return _inner

def _parse_bybit_trade(inst: str):
    def _inner(msg: Any) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(msg, (bytes, bytearray)):
                msg = msg.decode("utf-8")
            obj = json.loads(msg)
            # bybit v5: {"topic":"publicTrade.BTCUSDT","type":"snapshot","data":[{"T":...,"p":"...","v":"...","S":"Buy","i":"..."}]}
            if obj.get("topic", "") == f"publicTrade.{inst}":
                data = obj.get("data", [])
                if not data:
                    return None
                d0 = data[0]
                return {
                    "id": d0.get("i"),
                    "price": float(d0.get("p")),
                    "amount": float(d0.get("v")),
                    "side": d0.get("S").lower(),
                    "ts": int(d0.get("T")),
                }
        except Exception:
            return None
        return None
    return _inner

# ---------------- Script entry (manual test) ----------------

if __name__ == "__main__":
    cfg = CryptoConfig(exchange="binance")
    ca = CryptoAdapter(cfg)
    print(json.dumps(ca.fetch_ticker("BTC/USDT"), indent=2))
    print(json.dumps(ca.fetch_orderbook("BTC/USDT", depth=10), indent=2))
    trades = ca.fetch_trades("BTC/USDT", limit=5)
    print(json.dumps(trades[:2], indent=2))