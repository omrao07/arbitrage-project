# backend/strategies/diversified/weather_derivative_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Weather Derivative Arbitrage — paper
------------------------------------
Indices:
  • HDD = Σ max(0, 65°F - T_avg_d)  over a contract window
  • CDD = Σ max(0, T_avg_d - 65°F)  over a contract window
(Use °C and base-18 if you prefer; just keep the feed consistent.)

Modes:
  1) SWAP_QVF:
       Fair swap value = NOTIONAL_USD_PER_DEG × E[index].
       If Street_Quote >> Fair → SELL swap; if << → BUY swap.

       Redis feeds you publish elsewhere:
         # Forecast ensemble for the contract window (deg-day index)
         HSET wx:ensemble "<LOC>|<SEASON>|<INDEX>" '{"samples":[310, 295, 333, ...]}'
         # Street quotes (USD total for the period; per-deg notional given separately)
         HSET wx:swap:quote "<LOC>|<SEASON>|<INDEX>" 315.0
         # Contract meta
         HSET wx:meta "<LOC>|<SEASON>|<INDEX>" '{"notional_per_deg": 20.0, "start_ms": <epoch>, "end_ms": <epoch>}'

  2) OPTION_QVF:
       Fit a simple distribution to ensemble (Normal with μ,σ; clipped at 0), then price:
         Call(K) = E[max(Index - K, 0)] ; Put(K) = E[max(K - Index, 0)].
       Compare mid vs fair; buy cheap / sell rich.

       Redis feeds:
         HSET wx:opt:quote "<LOC>|<SEASON>|<INDEX>|CALL|<K>" 12.4
         HSET wx:opt:quote "<LOC>|<SEASON>|<INDEX>|PUT|<K>"  10.1
         (Units = USD total option premium for the contract, already notionalized.)

Paper routing (your adapters will map these symbols later):
  • Swaps  : "WX_SWAP:<LOC>|<SEASON>|<INDEX>"
  • Options: "WX_OPT_<RIGHT>:<LOC>|<SEASON>|<INDEX>|K:<K>"
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("WX_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("WX_REDIS_PORT", "6379"))

MODE   = os.getenv("WX_MODE", "SWAP_QVF").upper()        # SWAP_QVF | OPTION_QVF
LOC    = os.getenv("WX_LOC", "ORD").upper()              # e.g., ORD (Chicago O'Hare), LHR, DEL
SEASON = os.getenv("WX_SEASON", "2025-12").upper()       # contract window label
INDEX  = os.getenv("WX_INDEX", "HDD").upper()            # HDD | CDD

# Thresholds / gates
ENTRY_USD   = float(os.getenv("WX_ENTRY_USD", "150"))    # min mispricing (USD) to enter
EXIT_USD    = float(os.getenv("WX_EXIT_USD",  "50"))
ENTRY_Z     = float(os.getenv("WX_ENTRY_Z", "1.1"))
EXIT_Z      = float(os.getenv("WX_EXIT_Z",  "0.5"))

# Sizing / risk
USD_BUDGET     = float(os.getenv("WX_USD_BUDGET", "50000"))
MIN_TICKET_USD = float(os.getenv("WX_MIN_TICKET_USD", "500"))
MAX_CONCURRENT = int(os.getenv("WX_MAX_CONCURRENT", "1"))

# Cadence / stats
RECHECK_SECS = float(os.getenv("WX_RECHECK_SECS", "1.0"))
EWMA_ALPHA   = float(os.getenv("WX_EWMA_ALPHA", "0.08"))

# Redis keys
HALT_KEY   = os.getenv("WX_HALT_KEY", "risk:halt")
ENSEMBLE_HK= os.getenv("WX_ENSEMBLE_HK", "wx:ensemble")
SWAP_Q_HK  = os.getenv("WX_SWAP_Q_HK",  "wx:swap:quote")
OPT_Q_HK   = os.getenv("WX_OPT_Q_HK",   "wx:opt:quote")
META_HK    = os.getenv("WX_META_HK",    "wx:meta")
FEES_HK    = os.getenv("WX_FEES_HK",    "fees:wx")      # HSET fees:wx OTC 8 (bps)

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw)
        return j if isinstance(j, dict) else None
    except Exception:
        return None

def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try:
            j = json.loads(v)
            return float(j) if isinstance(j, (int,float)) else None
        except Exception:
            return None

def _fees_bps() -> float:
    v = r.hget(FEES_HK, "OTC")
    try: return float(v) if v is not None else 8.0
    except Exception: return 8.0

def _now_ms() -> int: return int(time.time() * 1000)

def _key_root() -> str:
    return f"{LOC}|{SEASON}|{INDEX}"

# Basic stats from ensemble
def _ensemble_stats(samples: List[float]) -> Tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    n = float(len(samples))
    m = sum(samples)/n
    v = sum((x - m)*(x - m) for x in samples) / max(1.0, n-1.0)
    return m, max(0.0, v)

# Option expectation under Normal(μ,σ) clipped at 0 (quick closed-form approx)
def _call_put_fair_from_normal(mu: float, sigma: float, k: float, notional_per_deg: float) -> Tuple[float, float]:
    """
    Approx E[(X-K)+] & E[(K-X)+] for X~N(mu, sigma^2), X>=0 (clipped).
    We ignore the small mass below 0 when mu >> 0; adequate for HDD/CDD.
    """
    if sigma <= 0:
        call = max(0.0, mu - k)
        put  = max(0.0, k - mu)
        return notional_per_deg * call, notional_per_deg * put
    import math
    z = (mu - k) / sigma
    # Standard normal cdf/pdf
    Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    call_dd = (mu - k) * Phi + sigma * phi
    put_dd  = (k - mu) * (1.0 - Phi) + sigma * phi
    return notional_per_deg * max(0.0, call_dd), notional_per_deg * max(0.0, put_dd)

# ============================ EWMA ============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float,float]:
        m0 = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var  = max(1e-12, (1 - self.alpha) * (self.var + (x - m0)*(x - self.mean)))
        return self.mean, self.var

def _ewma_key(tag: str) -> str: return f"wx:ewma:{tag}"

def _load_ewma(tag: str) -> EwmaMV:
    raw = r.get(_ewma_key(tag))
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(tag: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(tag), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ state ============================
@dataclass
class OpenState:
    mode: str
    side: str         # swaps: "sell_swap" | "buy_swap"; options: "sell_opt" | "buy_opt"
    qty: float        # integer units (swaps) or contracts (options); adapter defines multipliers
    entry_edge_usd: float
    entry_z: float
    tag: str
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"wx:open:{name}:{tag}"

# ============================ Strategy ============================
class WeatherDerivativeArbitrage(Strategy):
    """
    Swap vs fair and option vs fair mispricings on HDD/CDD indices (paper).
    """
    def __init__(self, name: str = "weather_derivative_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "OPTION_QVF":
            self._eval_option_qvf()
        else:
            self._eval_swap_qvf()

    # --------------- SWAP_QVF ---------------
    def _eval_swap_qvf(self) -> None:
        root = _key_root()
        tag = f"SWAP:{root}"

        es = _hget_json(ENSEMBLE_HK, root) or {}
        meta = _hget_json(META_HK, root) or {}
        quote = _hgetf(SWAP_Q_HK, root)
        if not es or not meta or quote is None: return

        samples = [float(x) for x in (es.get("samples") or []) if x is not None]
        if len(samples) < 5: return
        mu, var = _ensemble_stats(samples)

        notional = float(meta.get("notional_per_deg", 20.0))
        fair = notional * mu  # USD
        edge = (quote - fair)  # +ve ⇒ street rich (receive $ too high vs fair)

        fee = _fees_bps() * 1e-4
        edge_adj = edge - abs(fair) * fee

        ew = _load_ewma(tag); m,v = ew.update(edge_adj); _save_ewma(tag, ew)
        z = (edge_adj - m)/math.sqrt(max(v,1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_adj / max(1.0, ENTRY_USD))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_adj) <= EXIT_USD) or (abs(z) <= EXIT_Z):
                self._close(tag, st, root, is_option=False)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_adj) >= ENTRY_USD and abs(z) >= ENTRY_Z): return

        # Sizing: 1 "unit" = notional_per_deg exposure; budget by fair magnitude
        unit_value = max(1.0, abs(fair))
        qty = max(1.0, math.floor(USD_BUDGET / max(unit_value, MIN_TICKET_USD)))
        if qty * unit_value < MIN_TICKET_USD: return

        if edge_adj > 0:
            # Rich street → SELL swap (receive high)
            self.order(f"WX_SWAP:{root}", "sell", qty=qty, order_type="market", venue="OTC")
            side = "sell_swap"
        else:
            self.order(f"WX_SWAP:{root}", "buy", qty=qty, order_type="market", venue="OTC")
            side = "buy_swap"

        self._save_state(tag, OpenState(mode="SWAP_QVF", side=side, qty=qty,
                                        entry_edge_usd=edge_adj, entry_z=z, tag=root, ts_ms=_now_ms()))

    # --------------- OPTION_QVF ---------------
    def _eval_option_qvf(self) -> None:
        root = _key_root()

        # scan a few strikes with quotes
        # Key format: "<LOC>|<SEASON>|<INDEX>|<RIGHT>|<K>"
        fields = r.hkeys(OPT_Q_HK) or []
        candidates = [f for f in fields if isinstance(f, str) and f.startswith(f"{root}|")]
        if not candidates: return

        es = _hget_json(ENSEMBLE_HK, root) or {}
        meta = _hget_json(META_HK, root) or {}
        samples = [float(x) for x in (es.get("samples") or []) if x is not None]
        if len(samples) < 5 or not meta: return

        mu, var = _ensemble_stats(samples)
        sigma = math.sqrt(max(1e-9, var))
        notional = float(meta.get("notional_per_deg", 20.0))

        best = None  # (edge_abs, tag, side, qty)
        for key in candidates:
            # parse
            try:
                _,_,_, right, k_str = key.split("|")
                K = float(k_str)
                right = right.upper()
            except Exception:
                continue

            q = _hgetf(OPT_Q_HK, key)
            if q is None: continue

            fair_call, fair_put = _call_put_fair_from_normal(mu, sigma, K, notional)
            fair = fair_call if right == "CALL" else fair_put
            edge = q - fair   # +ve ⇒ street rich
            edge_adj = edge - abs(fair) * (_fees_bps() * 1e-4)

            tag = f"OPT:{root}|{right}|K:{int(round(K))}"
            ew = _load_ewma(tag); m,v = ew.update(edge_adj); _save_ewma(tag, ew)
            z = (edge_adj - m)/math.sqrt(max(v,1e-12))
            self.emit_signal(max(-1.0, min(1.0, edge_adj / max(1.0, ENTRY_USD))))

            # Decide & place only for the single best edge to avoid overtrading
            if abs(edge_adj) >= ENTRY_USD and abs(z) >= ENTRY_Z and r.get(_poskey(self.ctx.name, tag)) is None:
                # Sizing: spend fraction of budget per trade
                unit = max(1.0, q)  # one contract premium proxy
                qty = max(1.0, math.floor(USD_BUDGET / (5.0 * max(unit, MIN_TICKET_USD))))
                if qty * unit < MIN_TICKET_USD: continue

                side = "sell_opt" if edge_adj > 0 else "buy_opt"
                best = (abs(edge_adj), tag, side, qty, right, K, q)
                # keep searching to find max edge

        if not best: return
        _, tag, side, qty, right, K, q = best
        sym = f"WX_OPT_{right}:{root}|K:{int(round(K))}"
        act = "sell" if side == "sell_opt" else "buy"
        self.order(sym, act, qty=qty, order_type="market", venue="OTC")
        self._save_state(tag, OpenState(mode="OPTION_QVF", side=side, qty=qty,
                                        entry_edge_usd=q, entry_z=0.0, tag=tag, ts_ms=_now_ms()))

    # --------------- close / unwind ---------------
    def _close(self, tag: str, st: OpenState, root: str, is_option: bool) -> None:
        if st.mode == "SWAP_QVF":
            rev = "buy" if st.side == "sell_swap" else "sell"
            self.order(f"WX_SWAP:{root}", rev, qty=st.qty, order_type="market", venue="OTC")
        else:
            # For options we reverse same series
            parts = st.tag.split("|")
            # st.tag like "OPT:<root>|<RIGHT>|K:<K>"
            right = parts[-2]
            k = parts[-1].split(":")[1]
            sym = f"WX_OPT_{right}:{root}|K:{k}"
            rev = "buy" if st.side == "sell_opt" else "sell"
            self.order(sym, rev, qty=st.qty, order_type="market", venue="OTC")
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- state I/O ---------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(mode=str(o["mode"]), side=str(o["side"]), qty=float(o["qty"]),
                             entry_edge_usd=float(o["entry_edge_usd"]), entry_z=float(o.get("entry_z", 0.0)),
                             tag=str(o.get("tag", "")), ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side, "qty": st.qty,
            "entry_edge_usd": st.entry_edge_usd, "entry_z": st.entry_z,
            "tag": st.tag, "ts_ms": st.ts_ms
        }))