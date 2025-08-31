# backend/strategies/diversified/fx_crypto_bridge.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import redis
from backend.engine.strategy_base import Strategy

"""
FX ⇄ Crypto Bridge — paper
--------------------------
Idea:
  Compare official FX spot (e.g., USD/INR) with the *implied FX* you get by
  going via crypto rails (USDT/INR, USDT/USD, BTC/INR vs BTC/USD, etc.).
  When spread > fees + slippage + buffer, go long cheap currency on the cheap leg
  and short expensive on the rich leg (paper legs across venues).

Modes:
  1) STABLE_ARBITRAGE
     implied_fx = (USDT/INR) / (USDT/USD)
     Edge bps   = 1e4 * (implied_fx / fx_spot - 1)
     Trade: if implied > spot ⇒ INR rich vs USD → SELL INR leg (sell USDT for INR), BUY USD leg (buy USDT for USD).
            if implied < spot ⇒ INR cheap → BUY INR leg, SELL USD leg.

     Redis you publish elsewhere:
       HSET fx:spot "USDINR" 83.20         # official FX (bank/forex reference)
       HSET last_price "CEX:BINANCE:USDTINR"  '{"price": 84.10}'
       HSET last_price "CEX:KRAKEN:USDTUSD"   '{"price": 1.0007}'

  2) BTC_TRI
     implied_fx = (BTCINR / BTCUSD)
     Same logic, but routes BTC on both sides, then flat BTC via offsetting.

     Redis:
       HSET last_price "CEX:WAZIRX:BTCINR" '{"price": 5600000.0}'
       HSET last_price "CEX:COINBASE:BTCUSD" '{"price": 67000.0}'

Optional:
  • Funding hedge on perps to keep crypto delta near zero (not included here; paper spot only).
  • Regional gates (KYC/AML, capital controls) ⇒ we expose allowlists and caps.

Routing (paper; adapters wire later):
  order("<VEN>:<SYMBOL>", side, qty, order_type="market", venue="<VEN>")
    e.g., order("CEX:BINANCE:USDTINR", "sell", 1000)  # sell 1000 USDT for INR (paper)
          order("CEX:KRAKEN:USDTUSD",  "buy",  1000)  # buy 1000 USDT for USD (paper)

Compliance note:
  This is **paper‑only**. Real cross‑border settlement faces KYC/AML, taxes,
  LRS/ODI caps (e.g., India), exchange T&Cs, banking rails timing, and counterparty risk.
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("FXCB_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("FXCB_REDIS_PORT", "6379"))

MODE = os.getenv("FXCB_MODE", "STABLE_ARBITRAGE").upper()   # STABLE_ARBITRAGE | BTC_TRI

# Pairs / venues (override via env)
FX_PAIR       = os.getenv("FXCB_FX_PAIR", "USDINR").upper()
USDT_INR_SYM  = os.getenv("FXCB_USDT_INR", "CEX:BINANCE:USDTINR").upper()
USDT_USD_SYM  = os.getenv("FXCB_USDT_USD", "CEX:KRAKEN:USDTUSD").upper()
BTC_INR_SYM   = os.getenv("FXCB_BTC_INR",  "CEX:WAZIRX:BTCINR").upper()
BTC_USD_SYM   = os.getenv("FXCB_BTC_USD",  "CEX:COINBASE:BTCUSD").upper()

# Thresholds / gates
ENTRY_BPS     = float(os.getenv("FXCB_ENTRY_BPS", "35"))   # enter if |edge| >= 35 bps
EXIT_BPS      = float(os.getenv("FXCB_EXIT_BPS",  "12"))   # exit when |edge| <= 12 bps
ENTRY_Z       = float(os.getenv("FXCB_ENTRY_Z", "1.0"))
EXIT_Z        = float(os.getenv("FXCB_EXIT_Z",  "0.4"))
RECHECK_SECS  = float(os.getenv("FXCB_RECHECK_SECS", "0.8"))

# Sizing / risk
USD_BUDGET       = float(os.getenv("FXCB_USD_BUDGET", "15000"))
MIN_TICKET_USD   = float(os.getenv("FXCB_MIN_TICKET_USD", "200"))
MAX_CONCURRENT   = int(os.getenv("FXCB_MAX_CONCURRENT", "1"))

# Regional/venue caps
ALLOW_USD   = int(os.getenv("FXCB_ALLOW_USD", "1"))   # 0/1
ALLOW_INR   = int(os.getenv("FXCB_ALLOW_INR", "1"))
CAP_INR_USD = float(os.getenv("FXCB_CAP_INR_USD", "20000"))   # cap notional routed via INR venues (USD eq)

# Fees/Slippage (bps on notional each leg)
FEE_USDT_INR_BPS = float(os.getenv("FXCB_FEE_USDT_INR_BPS", "20"))
FEE_USDT_USD_BPS = float(os.getenv("FXCB_FEE_USDT_USD_BPS", "6"))
FEE_BTC_INR_BPS  = float(os.getenv("FXCB_FEE_BTC_INR_BPS",  "25"))
FEE_BTC_USD_BPS  = float(os.getenv("FXCB_FEE_BTC_USD_BPS",  "6"))
FEE_FIAT_MOVE_BPS= float(os.getenv("FXCB_FEE_FIAT_MOVE_BPS","10"))  # off‑exchange rails buffer
SLIPPAGE_BPS     = float(os.getenv("FXCB_SLIPPAGE_BPS",     "10"))  # generic cushion

# Redis keys
HALT_KEY = os.getenv("FXCB_HALT_KEY", "risk:halt")
LAST_HK  = os.getenv("FXCB_LAST_HK",  "last_price")
FX_HK    = os.getenv("FXCB_FX_HK",    "fx:spot")
STATE_HK = os.getenv("FXCB_STATE_HK", "fxcb:state")
EWMA_HK  = os.getenv("FXCB_EWMA_HK",  "fxcb:ewma")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hgetf(hk: str, field: str) -> Optional[float]:
    raw = r.hget(hk, field)
    if raw is None: return None
    try: return float(raw) # type: ignore
    except Exception:
        try:
            j = json.loads(raw) # type: ignore
            if isinstance(j, dict) and "price" in j: return float(j["price"])
            if isinstance(j, (int,float)): return float(j)
        except Exception:
            return None
    return None

def _px(symkey: str) -> Optional[float]:
    # symkey is already the field for LAST_HK (e.g., "CEX:BINANCE:USDTINR")
    return _hgetf(LAST_HK, symkey)

def _fx(pair: str) -> Optional[float]:
    # fx:spot stores e.g., HSET fx:spot "USDINR" 83.2
    return _hgetf(FX_HK, pair)

def _now_ms() -> int: return int(time.time()*1000)

# ====== simple EWMA of edge for z‑score ======
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float,float]:
        m0 = self.mean
        self.mean = (1 - self.alpha)*self.mean + self.alpha*x
        self.var  = max(1e-12, (1 - self.alpha)*(self.var + (x - m0)*(x - self.mean)))
        return self.mean, self.var

def _ewma_key(tag: str) -> str: return f"{EWMA_HK}:{tag}"
def _load_ewma(tag: str, alpha: float=0.08) -> EwmaMV:
    raw = r.get(_ewma_key(tag))
    if raw:
        try:
            o = json.loads(raw); return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", alpha))) # type: ignore
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)
def _save_ewma(tag: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(tag), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ state ============================
@dataclass
class OpenState:
    mode: str
    side: str          # "INR_rich" or "INR_cheap" (sign of edge)
    qty_usdt: float    # USDT bridged notionally (for STABLE)
    qty_btc: float     # BTC bridged (for BTC_TRI)
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"fxcb:open:{name}:{tag}"

# ============================ Strategy ============================
class FxCryptoBridge(Strategy):
    """
    Bridge arbitrage between fiat FX and crypto-implied FX (paper).
    """
    def __init__(self, name: str = "fx_crypto_bridge", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "BTC_TRI":
            self._run_btc_tri()
        else:
            self._run_stable()

    # --------------- STABLE (USDT) ---------------
    def _run_stable(self) -> None:
        tag = f"STABLE:{FX_PAIR}"

        fx = _fx(FX_PAIR)
        p_inr = _px(USDT_INR_SYM)
        p_usd = _px(USDT_USD_SYM)
        if fx is None or p_inr is None or p_usd is None or fx <= 0 or p_inr <= 0 or p_usd <= 0:
            self.emit_signal(0.0); return

        implied = p_inr / p_usd  # INR per USD via USDT legs
        edge_bps = 1e4 * (implied / fx - 1.0)

        # total fees/slippage buffer in bps (both legs + fiat move)
        buf = (FEE_USDT_INR_BPS + FEE_USDT_USD_BPS + FEE_FIAT_MOVE_BPS + SLIPPAGE_BPS)
        net_edge_bps = (edge_bps - (buf if edge_bps > 0 else -buf))

        ew = _load_ewma(tag); m,v = ew.update(net_edge_bps); _save_ewma(tag, ew)
        z = (net_edge_bps - m)/math.sqrt(max(v,1e-12))
        self.emit_signal(max(-1.0, min(1.0, net_edge_bps/100.0)))  # scale for UI

        st = self._load_state(tag)
        # Exit conditions
        if st and (abs(net_edge_bps) <= EXIT_BPS or abs(z) <= EXIT_Z):
            self._close_stable(tag, st)
            return

        # Entry
        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(net_edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return
        if (ALLOW_INR == 0): return

        # Size in USDT notionally
        # Make each side notional coherent; keep within USD_BUDGET and regional caps.
        usdt_qty = max(1.0, math.floor(USD_BUDGET / max(1.0, p_usd)))
        # Skip if ticket too small
        if usdt_qty * p_usd < MIN_TICKET_USD: return

        if net_edge_bps > 0:
            # implied > spot ⇒ INR rich: SELL INR leg (sell USDT for INR), BUY USD leg (buy USDT for USD)
            self.order(USDT_INR_SYM, "sell", qty=usdt_qty, order_type="market", venue=USDT_INR_SYM.split(":")[1])
            self.order(USDT_USD_SYM, "buy",  qty=usdt_qty, order_type="market", venue=USDT_USD_SYM.split(":")[1])
            side = "INR_rich"
        else:
            # implied < spot ⇒ INR cheap: BUY INR leg, SELL USD leg
            self.order(USDT_INR_SYM, "buy",  qty=usdt_qty, order_type="market", venue=USDT_INR_SYM.split(":")[1])
            self.order(USDT_USD_SYM, "sell", qty=usdt_qty, order_type="market", venue=USDT_USD_SYM.split(":")[1])
            side = "INR_cheap"

        self._save_state(tag, OpenState(mode="STABLE_ARBITRAGE", side=side,
                                        qty_usdt=usdt_qty, qty_btc=0.0,
                                        entry_bps=net_edge_bps, entry_z=z, ts_ms=_now_ms()))

    def _close_stable(self, tag: str, st: OpenState) -> None:
        # Reverse legs
        if st.side == "INR_rich":
            self.order(USDT_INR_SYM, "buy",  qty=st.qty_usdt, order_type="market", venue=USDT_INR_SYM.split(":")[1])
            self.order(USDT_USD_SYM, "sell", qty=st.qty_usdt, order_type="market", venue=USDT_USD_SYM.split(":")[1])
        else:
            self.order(USDT_INR_SYM, "sell", qty=st.qty_usdt, order_type="market", venue=USDT_INR_SYM.split(":")[1])
            self.order(USDT_USD_SYM, "buy",  qty=st.qty_usdt, order_type="market", venue=USDT_USD_SYM.split(":")[1])
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- BTC TRI ---------------
    def _run_btc_tri(self) -> None:
        tag = f"BTC_TRI:{FX_PAIR}"

        fx = _fx(FX_PAIR)
        p_inr = _px(BTC_INR_SYM)
        p_usd = _px(BTC_USD_SYM)
        if fx is None or p_inr is None or p_usd is None or fx <= 0 or p_inr <= 0 or p_usd <= 0:
            self.emit_signal(0.0); return

        implied = p_inr / p_usd  # INR per USD via BTC
        edge_bps = 1e4 * (implied / fx - 1.0)

        buf = (FEE_BTC_INR_BPS + FEE_BTC_USD_BPS + FEE_FIAT_MOVE_BPS + SLIPPAGE_BPS)
        net_edge_bps = (edge_bps - (buf if edge_bps > 0 else -buf))

        ew = _load_ewma(tag); m,v = ew.update(net_edge_bps); _save_ewma(tag, ew)
        z = (net_edge_bps - m)/math.sqrt(max(v,1e-12))
        self.emit_signal(max(-1.0, min(1.0, net_edge_bps/100.0)))

        st = self._load_state(tag)
        if st and (abs(net_edge_bps) <= EXIT_BPS or abs(z) <= EXIT_Z):
            self._close_btc(tag, st); return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(net_edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return
        if (ALLOW_INR == 0): return

        # Size in BTC based on USD_BUDGET on USD side
        btc_qty = max(0.0001, USD_BUDGET / max(1.0, p_usd))
        if btc_qty * p_usd < MIN_TICKET_USD: return

        if net_edge_bps > 0:
            # INR rich ⇒ SELL BTC/INR, BUY BTC/USD (keeps BTC roughly flat notionally)
            self.order(BTC_INR_SYM, "sell", qty=btc_qty, order_type="market", venue=BTC_INR_SYM.split(":")[1])
            self.order(BTC_USD_SYM, "buy",  qty=btc_qty, order_type="market", venue=BTC_USD_SYM.split(":")[1])
            side = "INR_rich"
        else:
            # INR cheap ⇒ BUY BTC/INR, SELL BTC/USD
            self.order(BTC_INR_SYM, "buy",  qty=btc_qty, order_type="market", venue=BTC_INR_SYM.split(":")[1])
            self.order(BTC_USD_SYM, "sell", qty=btc_qty, order_type="market", venue=BTC_USD_SYM.split(":")[1])
            side = "INR_cheap"

        self._save_state(tag, OpenState(mode="BTC_TRI", side=side,
                                        qty_usdt=0.0, qty_btc=btc_qty,
                                        entry_bps=net_edge_bps, entry_z=z, ts_ms=_now_ms()))

    def _close_btc(self, tag: str, st: OpenState) -> None:
        if st.side == "INR_rich":
            self.order(BTC_INR_SYM, "buy",  qty=st.qty_btc, order_type="market", venue=BTC_INR_SYM.split(":")[1])
            self.order(BTC_USD_SYM, "sell", qty=st.qty_btc, order_type="market", venue=BTC_USD_SYM.split(":")[1])
        else:
            self.order(BTC_INR_SYM, "sell", qty=st.qty_btc, order_type="market", venue=BTC_INR_SYM.split(":")[1])
            self.order(BTC_USD_SYM, "buy",  qty=st.qty_btc, order_type="market", venue=BTC_USD_SYM.split(":")[1])
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- state I/O ---------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(mode=str(o["mode"]), side=str(o["side"]),
                             qty_usdt=float(o.get("qty_usdt", 0.0)), qty_btc=float(o.get("qty_btc", 0.0)),
                             entry_bps=float(o["entry_bps"]), entry_z=float(o["entry_z"]), ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side,
            "qty_usdt": st.qty_usdt, "qty_btc": st.qty_btc,
            "entry_bps": st.entry_bps, "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))