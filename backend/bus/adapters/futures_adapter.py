# backend/data/futures_adapter.py
"""
Futures Market-Data Adapter
---------------------------
Provider(s): currently Yahoo Finance via `yfinance` (works well for many CME/NYMEX/CBOT/COMEX/ICE symbols).
Examples of Yahoo symbols:
  • Continuous/front: ES=F, NQ=F, YM=F, RTY=F, CL=F, NG=F, GC=F, SI=F, ZC=F, ZS=F, ZW=F
  • Specific expiry (when available): CLX25.NYM, NGF26.NYM, GCZ25.CMX, ZCH26.CBT, ESZ25.CME

Capabilities
  • fetch_quote(symbol)
  • fetch_ohlcv(symbol, interval="1d", lookback=120, start=None, end=None)
  • fetch_open_interest(symbol, ...)  # pulls OI column when available
  • build_continuous(root="CL", rule="front")  # front-month continuous from Yahoo front symbol
  • term_structure(root="CL", months=["X25","Z25","F26"])  # prices/last for chosen expiries
  • calendar_spread(root="CL", near="X25", far="Z25")  # near - far price spread

All outputs are normalized "envelopes" with a deterministic SHA-256 hash and optional Merkle-append.

Dependencies
  • yfinance  (pip install yfinance)
  • pandas, numpy (pulled by yfinance)

Bus Hook
  • publish_stream(stream, payload) from backend.bus.streams (falls back to a safe print stub).

Notes
  • Exchange official feeds (CME MDP3, ICE, etc) are licensed—this adapter stays on public data.
  • You can add more providers later (e.g., Nasdaq Data Link, dxFeed) without changing downstream consumers.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# -------- optional deps --------
try:
    import yfinance as yf  # type: ignore
    _HAS_YF = True
except Exception:
    _HAS_YF = False

# -------- bus hook --------
try:
    from backend.bus.streams import publish_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload):
        head = payload if isinstance(payload, dict) else (payload[0] if payload else {})
        print(f"[stub publish_stream] {stream} <- {json.dumps(head, separators=(',',':'))[:220]}{'...' if isinstance(payload, list) and len(payload)>1 else ''}")

# -------- optional ledger --------
def _ledger_append(payload, ledger_path: Optional[str]):
    if not ledger_path:
        return
    try:
        from backend.audit.merkle_ledger import MerkleLedger  # type: ignore
        MerkleLedger(ledger_path).append({"type": "futures_md", "payload": payload})
    except Exception:
        pass

# -------- config & models --------

@dataclass
class FutConfig:
    provider: str = "yfinance"            # currently only yfinance
    stream: str = "STREAM_FUTURES_MD"
    ledger_path: Optional[str] = None
    user_agent: str = "HF-OS/futures-adapter/1.0"

@dataclass
class OHLCV:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: Optional[float] = None

# -------- adapter --------

class FuturesAdapter:
    def __init__(self, cfg: FutConfig) -> None:
        self.cfg = cfg
        p = cfg.provider.lower().strip()
        if p != "yfinance":
            raise RuntimeError(f"Unsupported provider '{cfg.provider}' (only 'yfinance' is implemented).")
        if not _HAS_YF:
            raise RuntimeError("yfinance not installed. Run: pip install yfinance")
        self.provider = p

    # ---------- Quotes ----------

    def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.strip().upper()
        tk = yf.Ticker(sym)
        # Try fast_info; fallback to last close
        q = getattr(tk, "fast_info", {}) or {}
        last = _f(q.get("last_price"))
        if last is None:
            try:
                hist = tk.history(period="1d").tail(1)
                if len(hist) > 0:
                    last = float(hist["Close"].iloc[-1])
            except Exception:
                last = None
        payload = {
            "bid": _f(q.get("last_bid")),
            "ask": _f(q.get("last_ask")),
            "last": last,
            "exchange": getattr(tk, "info", {}).get("exchange", None),
            "contract": sym,
        }
        env = self._envelope(kind="quote", symbol=sym, payload=payload)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    # ---------- OHLCV / OI ----------

    def fetch_ohlcv(self, symbol: str, *, interval: str = "1d", lookback: int = 120,
                    start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
        sym = symbol.strip().upper()
        tk = yf.Ticker(sym)
        hist = yf.download(sym, period=None if start else f"{lookback}d",
                           interval=interval, start=start, end=end, progress=False, auto_adjust=False)
        if hist is None or len(hist) == 0:
            return []
        # Try to join Open Interest if yfinance returns it separately
        oi_series = None
        try:
            oi_df = tk.history(period="max")  # sometimes includes "Open Interest"
            if "Open Interest" in oi_df.columns:
                oi_series = oi_df["Open Interest"]
        except Exception:
            pass

        envs: List[Dict[str, Any]] = []
        for ts, row in hist.iterrows():
            ts_ms = int(getattr(ts, "value", int(time.time() * 1000)) // 10**6)
            oi_val = None
            if oi_series is not None:
                try:
                    v = oi_series.loc[ts] # type: ignore
                    oi_val = None if _isna(v) else float(v)
                except Exception:
                    oi_val = None
            bar = OHLCV(
                ts_ms=ts_ms,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0.0)),
                open_interest=oi_val,
            )
            env = self._envelope(kind="ohlcv", symbol=sym, payload=bar.__dict__, ts=ts_ms)
            _ledger_append(env, self.cfg.ledger_path)
            envs.append(env)
        return envs

    def fetch_open_interest(self, symbol: str, *, lookback: int = 180) -> List[Dict[str, Any]]:
        """Return a time series of Open Interest if available."""
        sym = symbol.strip().upper()
        tk = yf.Ticker(sym)
        try:
            hist = tk.history(period="max").tail(lookback)
        except Exception:
            return []
        if "Open Interest" not in hist.columns:
            return []
        envs: List[Dict[str, Any]] = []
        for ts, v in hist["Open Interest"].iterrows() if isinstance(hist["Open Interest"], dict) else hist["Open Interest"].items():
            try:
                val = float(v)
            except Exception:
                continue
            ts_ms = int(getattr(ts, "value", int(time.time() * 1000)) // 10**6)
            payload = {"ts_ms": ts_ms, "open_interest": val}
            env = self._envelope(kind="open_interest", symbol=sym, payload=payload, ts=ts_ms)
            _ledger_append(env, self.cfg.ledger_path)
            envs.append(env)
        return envs

    # ---------- Continuous & Curve ----------

    def build_continuous(self, *, root: str) -> Dict[str, Any]:
        """
        Use Yahoo's front-month continuous (root+'=F') if present.
        Example: root='CL' -> 'CL=F'
        """
        sym = f"{root.strip().upper()}=F"
        q = self.fetch_quote(sym)
        # Also return last 60 daily closes for the continuous
        bars = self.fetch_ohlcv(sym, interval="1d", lookback=60)
        payload = {
            "contract": sym,
            "last": q["payload"].get("last"),
            "recent_close": [b["payload"]["close"] for b in bars],
            "recent_ts": [b["payload"]["ts_ms"] for b in bars],
        }
        env = self._envelope(kind="continuous", symbol=sym, payload=payload)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    def term_structure(self, *, root: str, months: Sequence[str]) -> Dict[str, Any]:
        """
        Get last prices for specified expiries (e.g., months=["X25","Z25","F26"]).
        Returns a single envelope with a curve snapshot.

        Month codes: F G H J K M N Q U V X Z  (Jan..Dec)
        Yahoo symbol template: <ROOT><MONTH><YY>.<EXCH> (often .NYM, .CBT, .CMX, .CME)
        If explicit contract not found, we try common exchange suffixes; if all fail,
        we fall back to continuous for that node.
        """
        root = root.strip().upper()
        suffixes = [".NYM", ".CME", ".CBT", ".CMX", ".CBOE"]  # try in order
        curve: List[Tuple[str, Optional[float], Optional[str]]] = []
        tried: List[str] = []

        for m in months:
            m = m.strip().upper()
            sym = None
            last = None
            used = None
            # Try explicit exchange suffixes
            for sfx in suffixes:
                cand = f"{root}{m}{sfx}"
                tried.append(cand)
                last = self._try_last_price(cand)
                if last is not None:
                    sym = cand
                    used = "explicit"
                    break
            if last is None:
                # fallback to continuous if available
                cont = f"{root}=F"
                last = self._try_last_price(cont)
                sym = cont
                used = "fallback_continuous"
            curve.append((sym, last, used)) # type: ignore

        payload = {
            "root": root,
            "months": list(months),
            "nodes": [{"contract": c, "last": p, "source": src} for c, p, src in curve],
            "tried": tried[:],
        }
        env = self._envelope(kind="term_structure", symbol=root, payload=payload)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    def calendar_spread(self, *, root: str, near: str, far: str) -> Dict[str, Any]:
        """
        Compute near - far price spread for given expiries (e.g., CLX25 - CLZ25).
        Tries explicit contracts first; falls back to continuous for whichever fails.
        """
        root = root.strip().upper()
        n_last, n_sym, n_src = self._pick_contract_last(root, near)
        f_last, f_sym, f_src = self._pick_contract_last(root, far)
        spread = None if (n_last is None or f_last is None) else (n_last - f_last)
        payload = {
            "near": {"contract": n_sym, "last": n_last, "source": n_src},
            "far":  {"contract": f_sym, "last": f_last, "source": f_src},
            "spread": spread,
        }
        env = self._envelope(kind="calendar_spread", symbol=root, payload=payload)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    # ---------- helpers ----------

    def _pick_contract_last(self, root: str, month_code: str) -> Tuple[Optional[float], str, str]:
        suffixes = [".NYM", ".CME", ".CBT", ".CMX", ".CBOE"]
        for sfx in suffixes:
            cand = f"{root}{month_code}{sfx}"
            last = self._try_last_price(cand)
            if last is not None:
                return last, cand, "explicit"
        cont = f"{root}=F"
        return self._try_last_price(cont), cont, "fallback_continuous"

    def _try_last_price(self, symbol: str) -> Optional[float]:
        try:
            tk = yf.Ticker(symbol)
            fi = getattr(tk, "fast_info", {}) or {}
            last = _f(fi.get("last_price"))
            if last is None:
                h = tk.history(period="1d").tail(1)
                if len(h) > 0:
                    last = float(h["Close"].iloc[-1])
            return last
        except Exception:
            return None

    def _envelope(self, *, kind: str, symbol: str, payload: Dict[str, Any], ts: Optional[int] = None) -> Dict[str, Any]:
        env = {
            "ts": int(ts or time.time() * 1000),
            "adapter": "futures",
            "provider": self.provider,
            "kind": kind,           # "quote" | "ohlcv" | "open_interest" | "continuous" | "term_structure" | "calendar_spread"
            "symbol": symbol,
            "payload": payload,
            "version": 1,
        }
        env["hash"] = hashlib.sha256(json.dumps(env, separators=(",", ":"), sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
        return env

    # ---------- publish ----------

    def publish(self, env_or_envs):
        if isinstance(env_or_envs, list):
            for e in env_or_envs:
                publish_stream(self.cfg.stream, e)
        else:
            publish_stream(self.cfg.stream, env_or_envs)


# -------- utils --------

_MONTH_MAP = {"F":"Jan","G":"Feb","H":"Mar","J":"Apr","K":"May","M":"Jun","N":"Jul","Q":"Aug","U":"Sep","V":"Oct","X":"Nov","Z":"Dec"}

def format_contract_human(root: str, month_code: str) -> str:
    """
    Pretty label, e.g., format_contract_human('CL','X25') -> 'CL Nov-2025'
    """
    root = root.strip().upper()
    m = month_code.strip().upper()
    mon = _MONTH_MAP.get(m[0], m[0])
    yy = m[1:]
    year = int("20" + yy) if len(yy) == 2 else int(yy)
    return f"{root} {mon}-{year}"

def _f(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def _isna(v) -> bool:
    try:
        import math
        return v is None or (isinstance(v, float) and math.isnan(v))
    except Exception:
        return v is None


# -------- script demo --------
if __name__ == "__main__":
    cfg = FutConfig()
    fa = FuturesAdapter(cfg)
    print(json.dumps(fa.fetch_quote("CL=F"), indent=2))
    bars = fa.fetch_ohlcv("CL=F", interval="1d", lookback=5)
    print(json.dumps([b["payload"] for b in bars], indent=2))
    print(json.dumps(fa.term_structure(root="CL", months=["X25","Z25","F26"]), indent=2))
    print(json.dumps(fa.calendar_spread(root="CL", near="X25", far="Z25"), indent=2))