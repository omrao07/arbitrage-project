# backend/strategies/diversified/water_rights_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Water Rights Arbitrage — paper
------------------------------
Modes:
  1) SPOT_SWAP:
     • Compare basin A vs basin B spot prices for the *same usable class* (e.g., transferable ag allocation).
     • Net spread accounts for conveyance loss %, transfer fee, and lead-time gating.
     • If (Price_B * (1 - loss) - Price_A) - fees > ENTRY → BUY in A, SELL/SWAP in B; reverse if negative.

     Redis you publish elsewhere (paper feed):
       HSET water:spot "BASIN:<ID>|CLASS:<C>" '{"price_per_af": 320.0, "ccy":"USD"}'
       HSET water:transfer_rules "<A>-><B>|<CLASS>" '{"loss_pct":0.07,"fee_per_af":15.0,"min_days":10,"eligible":1}'
       HSET cal:today ms_epoch     # optional gate by calendar
       HSET fees:water OTC 8       # bps guard on notional
       SET  risk:halt 0|1

     Paper routing:
       "WR:BUY:<BASIN>|<CLASS>"  qty in acre‑feet (AF)
       "WR:SELL:<BASIN>|<CLASS>"

  2) CARRY_STORAGE:
     • Compare spot vs forward/next‑season allocation in same basin:
         F_theo = S * (1 + finance_rate * T) + storage_fee_per_af + evap_loss_pct * S
       Edge per AF = F_mkt - F_theo  (positive ⇒ forward rich → SELL forward / BUY spot & store)
     • Includes storage capacity check and capex/permit gate.

     Redis (paper feed):
       HSET water:spot "BASIN:<ID>|CLASS:<C>"  '{"price_per_af": 320.0}'
       HSET water:forward "BASIN:<ID>|CLASS:<C>|<YYYY-SEASON>" '{"price_per_af": 415.0, "start_ms": <epoch>, "end_ms": <epoch>}'
       HSET water:storage "<BASIN>|<FACILITY>" '{"cap_af": 10000, "free_af": 8000, "evap_loss_pct":0.05,"fee_per_af":12.0}'
       HSET water:finance "<BASIN>" 0.06   # APR
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("WRA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("WRA_REDIS_PORT", "6379"))

MODE      = os.getenv("WRA_MODE", "SPOT_SWAP").upper()     # SPOT_SWAP | CARRY_STORAGE

# SPOT_SWAP params
SRC_BASIN = os.getenv("WRA_SRC_BASIN", "SAC").upper()
DST_BASIN = os.getenv("WRA_DST_BASIN", "SJR").upper()
W_CLASS   = os.getenv("WRA_CLASS", "AG_XFER").upper()      # your right/allocation class label

# CARRY_STORAGE params
CS_BASIN   = os.getenv("WRA_CS_BASIN", "SAC").upper()
CS_CLASS   = os.getenv("WRA_CS_CLASS", "AG_XFER").upper()
CS_FWD_KEY = os.getenv("WRA_CS_FWDKEY", "2026-WET").upper()   # e.g., season label
CS_FAC     = os.getenv("WRA_CS_FACILITY", "SITEA").upper()    # storage facility key

# Thresholds / gates
ENTRY_USD_AF = float(os.getenv("WRA_ENTRY_USD_AF", "10"))   # per‑AF edge to enter
EXIT_USD_AF  = float(os.getenv("WRA_EXIT_USD_AF",  "3"))
ENTRY_Z      = float(os.getenv("WRA_ENTRY_Z", "1.1"))
EXIT_Z       = float(os.getenv("WRA_EXIT_Z",  "0.5"))

# Sizing / risk
USD_BUDGET     = float(os.getenv("WRA_USD_BUDGET", "50000"))
AF_MIN_TICKET  = float(os.getenv("WRA_AF_MIN_TICKET", "10"))      # minimum trade size (AF)
AF_MAX_CONC    = float(os.getenv("WRA_AF_MAX_CONC", "1000"))      # cap per opportunity

# Cadence / stats
RECHECK_SECS   = float(os.getenv("WRA_RECHECK_SECS", "1.0"))
EWMA_ALPHA     = float(os.getenv("WRA_EWMA_ALPHA", "0.06"))

# Redis keys
HALT_KEY   = os.getenv("WRA_HALT_KEY", "risk:halt")
SPOT_HK    = os.getenv("WRA_SPOT_HK", "water:spot")
RULES_HK   = os.getenv("WRA_RULES_HK", "water:transfer_rules")
FWD_HK     = os.getenv("WRA_FWD_HK",  "water:forward")
STOR_HK    = os.getenv("WRA_STOR_HK", "water:storage")
FIN_HK     = os.getenv("WRA_FIN_HK",  "water:finance")
FEES_HK    = os.getenv("WRA_FEES_HK", "fees:water")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw) # type: ignore
        return j if isinstance(j, dict) else None
    except Exception:
        return None

def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try:
            j = json.loads(v) # type: ignore
            return float(j) if isinstance(j, (int,float)) else None
        except Exception: return None

def _fees_bps(venue: str="OTC") -> float:
    v = r.hget(FEES_HK, venue)
    try: return float(v) if v is not None else 8.0 # type: ignore
    except Exception: return 8.0

def _now_ms() -> int: return int(time.time() * 1000)

# ============================ EWMA ============================
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

def _ewma_key(tag: str) -> str: return f"wra:ewma:{tag}"
def _load_ewma(tag: str) -> EwmaMV:
    raw = r.get(_ewma_key(tag))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(tag: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(tag), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ state ============================
@dataclass
class OpenState:
    mode: str
    side: str     # "buy_src_sell_dst" | "sell_src_buy_dst" | "sell_fwd_buy_spot" | "buy_fwd_sell_spot"
    qty_af: float
    entry_edge_usd_af: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"wra:open:{name}:{tag}"

# ============================ Strategy ============================
class WaterRightsArbitrage(Strategy):
    """
    Paper trading for water-right spreads and carry/storage parity.
    """
    def __init__(self, name: str = "water_rights_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "CARRY_STORAGE":
            self._eval_carry_storage()
        else:
            self._eval_spot_swap()

    # --------------- SPOT_SWAP ---------------
    def _eval_spot_swap(self) -> None:
        tag = f"SWAP:{SRC_BASIN}->{DST_BASIN}|{W_CLASS}"

        ja = _hget_json(SPOT_HK, f"BASIN:{SRC_BASIN}|CLASS:{W_CLASS}")
        jb = _hget_json(SPOT_HK, f"BASIN:{DST_BASIN}|CLASS:{W_CLASS}")
        if not (ja and jb): return
        pa = float(ja.get("price_per_af", 0.0))
        pb = float(jb.get("price_per_af", 0.0))
        if pa <= 0 or pb <= 0: return

        rule = _hget_json(RULES_HK, f"{SRC_BASIN}->{DST_BASIN}|{W_CLASS}") or {}
        eligible = int(rule.get("eligible", 0)) == 1
        if not eligible: return

        loss_pct = float(rule.get("loss_pct", 0.0))
        fee_per_af = float(rule.get("fee_per_af", 0.0))

        # Net spread per AF if moving from SRC to DST
        net_ab = (pb * (1.0 - loss_pct)) - pa - fee_per_af
        # If negative, consider reverse path (if rule exists); else we’ll just act based on sign.
        # Risk guard: add fee bps
        fee_bps = _fees_bps("OTC") * 1e-4
        net_ab_adj = net_ab - (pa * fee_bps)

        ew = _load_ewma(tag); m,v = ew.update(net_ab_adj); _save_ewma(tag, ew)
        z = (net_ab_adj - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, net_ab_adj / max(1.0, ENTRY_USD_AF))))

        st = self._load_state(tag)
        if st:
            if (abs(net_ab_adj) <= EXIT_USD_AF) or (abs(z) <= EXIT_Z):
                self._close(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(net_ab_adj) >= ENTRY_USD_AF and abs(z) >= ENTRY_Z): return

        # Size in AF subject to USD budget at source price and AF caps
        max_af_by_budget = USD_BUDGET / pa
        qty_af = max(AF_MIN_TICKET, math.floor(min(max_af_by_budget, AF_MAX_CONC)))
        if qty_af < AF_MIN_TICKET: return

        if net_ab_adj > 0:
            # Move A→B: BUY in A, SELL/SWAP in B
            self.order(f"WR:BUY:{SRC_BASIN}|{W_CLASS}",  "buy",  qty=qty_af, order_type="market", venue="OTC")
            self.order(f"WR:SELL:{DST_BASIN}|{W_CLASS}", "sell", qty=qty_af*(1.0 - loss_pct), order_type="market", venue="OTC")
            side = "buy_src_sell_dst"
        else:
            # Reverse: SELL in A (if you hold), BUY in B — for paper we mirror the legs
            self.order(f"WR:SELL:{SRC_BASIN}|{W_CLASS}", "sell", qty=qty_af*(1.0 - loss_pct), order_type="market", venue="OTC")
            self.order(f"WR:BUY:{DST_BASIN}|{W_CLASS}",  "buy",  qty=qty_af, order_type="market", venue="OTC")
            side = "sell_src_buy_dst"

        self._save_state(tag, OpenState(mode="SPOT_SWAP", side=side, qty_af=qty_af,
                                        entry_edge_usd_af=net_ab_adj, entry_z=z, ts_ms=_now_ms()))

    # --------------- CARRY_STORAGE ---------------
    def _eval_carry_storage(self) -> None:
        tag = f"CARRY:{CS_BASIN}|{CS_CLASS}|{CS_FWD_KEY}"

        jspot = _hget_json(SPOT_HK, f"BASIN:{CS_BASIN}|CLASS:{CS_CLASS}")
        jfwd  = _hget_json(FWD_HK,  f"BASIN:{CS_BASIN}|CLASS:{CS_CLASS}|{CS_FWD_KEY}")
        jstor = _hget_json(STOR_HK, f"{CS_BASIN}|{CS_FAC}")
        rfin  = _hgetf(FIN_HK, CS_BASIN) or 0.0
        if not (jspot and jfwd and jstor): return

        S  = float(jspot.get("price_per_af", 0.0))
        Fm = float(jfwd.get("price_per_af", 0.0))
        evap = float(jstor.get("evap_loss_pct", 0.0))
        stor_fee = float(jstor.get("fee_per_af", 0.0))
        free_af = float(jstor.get("free_af", 0.0))
        if S<=0 or Fm<=0 or free_af < AF_MIN_TICKET: return

        # Tenor T (years) from forward window mid (approx)
        end_ms = int((jfwd.get("end_ms") or jfwd.get("start_ms") or 0))
        start_ms = int(time.time()*1000)
        T = max(1.0/365.0, (end_ms - start_ms) / 86400000.0) / 365.0 if end_ms>start_ms else 0.25

        Ftheo = S * (1.0 + rfin * T) + stor_fee + (evap * S)
        edge = Fm - Ftheo  # per AF

        fee_bps = _fees_bps("OTC") * 1e-4
        edge_adj = edge - (S * fee_bps)

        ew = _load_ewma(tag); m,v = ew.update(edge_adj); _save_ewma(tag, ew)
        z = (edge_adj - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_adj / max(1.0, ENTRY_USD_AF))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_adj) <= EXIT_USD_AF) or (abs(z) <= EXIT_Z):
                self._close(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_adj) >= ENTRY_USD_AF and abs(z) >= ENTRY_Z): return

        # Size: by budget & storage capacity
        max_af_by_budget = USD_BUDGET / S
        qty_af = max(AF_MIN_TICKET, math.floor(min(max_af_by_budget, free_af, AF_MAX_CONC)))
        if qty_af < AF_MIN_TICKET: return

        if edge_adj > 0:
            # Forward rich → SELL forward, BUY spot & STORE
            self.order(f"WRFWD:{CS_BASIN}|{CS_CLASS}|{CS_FWD_KEY}", "sell", qty=qty_af*(1.0-evap), order_type="market", venue="OTC")
            self.order(f"WR:BUY:{CS_BASIN}|{CS_CLASS}", "buy", qty=qty_af, order_type="market", venue="OTC")
            side = "sell_fwd_buy_spot"
        else:
            # Forward cheap → BUY forward, SELL spot (release storage) — paper mirror
            self.order(f"WRFWD:{CS_BASIN}|{CS_CLASS}|{CS_FWD_KEY}", "buy", qty=qty_af*(1.0-evap), order_type="market", venue="OTC")
            self.order(f"WR:SELL:{CS_BASIN}|{CS_CLASS}", "sell", qty=qty_af, order_type="market", venue="OTC")
            side = "buy_fwd_sell_spot"

        self._save_state(tag, OpenState(mode="CARRY_STORAGE", side=side, qty_af=qty_af,
                                        entry_edge_usd_af=edge_adj, entry_z=z, ts_ms=_now_ms()))

    # --------------- close / unwind ---------------
    def _close(self, tag: str, st: OpenState) -> None:
        if st.mode == "SPOT_SWAP":
            if st.side == "buy_src_sell_dst":
                self.order(f"WR:SELL:{SRC_BASIN}|{W_CLASS}", "sell", qty=st.qty_af, order_type="market", venue="OTC")
                self.order(f"WR:BUY:{DST_BASIN}|{W_CLASS}",  "buy",  qty=st.qty_af, order_type="market", venue="OTC")
            else:
                self.order(f"WR:BUY:{SRC_BASIN}|{W_CLASS}",  "buy",  qty=st.qty_af, order_type="market", venue="OTC")
                self.order(f"WR:SELL:{DST_BASIN}|{W_CLASS}", "sell", qty=st.qty_af, order_type="market", venue="OTC")
        else:
            if st.side == "sell_fwd_buy_spot":
                self.order(f"WRFWD:{CS_BASIN}|{CS_CLASS}|{CS_FWD_KEY}", "buy",  qty=st.qty_af, order_type="market", venue="OTC")
                self.order(f"WR:SELL:{CS_BASIN}|{CS_CLASS}",               "sell", qty=st.qty_af, order_type="market", venue="OTC")
            else:
                self.order(f"WRFWD:{CS_BASIN}|{CS_CLASS}|{CS_FWD_KEY}", "sell", qty=st.qty_af, order_type="market", venue="OTC")
                self.order(f"WR:BUY:{CS_BASIN}|{CS_CLASS}",             "buy",  qty=st.qty_af, order_type="market", venue="OTC")
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- state I/O ---------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(mode=str(o["mode"]), side=str(o["side"]),
                             qty_af=float(o["qty_af"]), entry_edge_usd_af=float(o["entry_edge_usd_af"]),
                             entry_z=float(o["entry_z"]), ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side, "qty_af": st.qty_af,
            "entry_edge_usd_af": st.entry_edge_usd_af, "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))