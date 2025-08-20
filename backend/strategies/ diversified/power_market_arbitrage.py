# backend/strategies/diversified/power_market_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Power Market Arbitrage (DA ↔ RT LMP + optional BESS overlay) — paper
--------------------------------------------------------------------
Core play (virtual bidding):
  For node N at delivery hour H:
    spread = E[ RT_LMP(N,H) ] - DA_LMP(N,H)
    If spread > ENTRY_$ → place INC (buy DA, sell RT)
    If spread < -ENTRY_$ → place DEC (sell DA, buy RT)
  Exits auto-resolve at settlement; this module manages entry gating & risk.

Optional overlay (BESS time-shift):
  Use BESS to buy energy at low RT price windows and discharge at high windows,
  respecting SoC, power, and efficiency. Intended for paper/sim; wire to real
  EMS/DERMS in adapters.

Redis feeds you publish elsewhere:
  # Prices / forecasts per node-hour (currency $/MWh)
  HSET lmp:da <NODE|H:YYYYMMDDHH> <da_lmp>                      # e.g., HSET lmp:da "PJM-A|H:2025081309" 24.31
  HSET lmp:rt_nowcast <NODE|H:YYYYMMDDHH> <rt_expected_lmp>     # your forecast (stat/ML); module EWMA’s errors
  # Uncertainty / costs
  HSET lmp:vol_bps <NODE|H:YYYYMMDDHH> <bps_equiv>              # optional: converts to $ gates dynamically
  HSET vb:fees <ISO> <bps_equiv>                                # virtual bid fees/slippage proxy
  # Risk caps
  HSET vb:caps:mw <NODE> <MW_cap>                               # per-node cap
  HSET vb:caps:notional <ISO> <USD_cap>                         # portfolio cap
  # Operational flags
  SET risk:halt 0|1

BESS (optional) — publish state & limits:
  HSET bess:state <ASSET_ID> '{"soc_mwh":..., "p_max_mw":..., "e_max_mwh":..., "eff_rt":0.94, "eff_ch":0.94}'
  HSET bess:price_bands <ASSET_ID> '{"buy": <$/MWh>, "sell": <$/MWh>}'  # static bands or update each tick
  HSET lmp:rt_now <NODE> <latest_rt_lmp>                                # for intra-hour BESS tick

Paper routing (map in adapters):
  • INC submit  : "VBID:INC:<ISO>:<NODE>:<YYYYMMDDHH>"
  • DEC submit  : "VBID:DEC:<ISO>:<NODE>:<YYYYMMDDHH>"
  • BESS charge : "BESS:<ASSET_ID>:CHARGE"
  • BESS discharge : "BESS:<ASSET_ID>:DISCHARGE"
"""

# ===================== CONFIG (env) =====================
REDIS_HOST = os.getenv("POWER_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("POWER_REDIS_PORT", "6379"))

ISO   = os.getenv("POWER_ISO", "PJM").upper()           # e.g., PJM, NYISO, CAISO, ERCOT, MISO, ISONE
NODE  = os.getenv("POWER_NODE", "PJM-A").upper()        # node/zone/ptid
HOUR  = os.getenv("POWER_HOUR", "H:2025081309")         # delivery hour key "H:YYYYMMDDHH"
KEY    = f"{NODE}|{HOUR}"

MODE  = os.getenv("POWER_MODE", "VBID").upper()         # "VBID", "BESS", or "BOTH"

# Entry/exit thresholds (absolute $/MWh unless using vol_bps)
ENTRY_USD = float(os.getenv("POWER_ENTRY_USD", "3.0"))
EXIT_USD  = float(os.getenv("POWER_EXIT_USD",  "1.0"))
ENTRY_Z   = float(os.getenv("POWER_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("POWER_EXIT_Z",    "0.5"))

# Sizing / risk
MW_BASE            = float(os.getenv("POWER_MW_BASE", "10"))         # default MW per ticket
MW_MIN_TICKET      = float(os.getenv("POWER_MW_MIN_TICKET", "1"))
USD_PORTF_CAP      = float(os.getenv("POWER_USD_PORTF_CAP", "500000"))
PER_NODE_MW_CAP    = float(os.getenv("POWER_PER_NODE_MW_CAP", "50")) # fallback if vb:caps:mw missing

# Cadence
RECHECK_SECS       = float(os.getenv("POWER_RECHECK_SECS", "5"))
EWMA_ALPHA         = float(os.getenv("POWER_EWMA_ALPHA", "0.06"))

# Redis keys
DA_HKEY            = os.getenv("POWER_DA_HKEY", "lmp:da")
RTF_HKEY           = os.getenv("POWER_RTF_HKEY", "lmp:rt_nowcast")
VOL_HKEY           = os.getenv("POWER_VOL_HKEY", "lmp:vol_bps")
FEES_HKEY          = os.getenv("POWER_FEES_HKEY", "vb:fees")
CAPS_MW_HKEY       = os.getenv("POWER_CAPS_MW_HKEY", "vb:caps:mw")
CAPS_USD_HKEY      = os.getenv("POWER_CAPS_USD_HKEY", "vb:caps:notional")
HALT_KEY           = os.getenv("POWER_HALT_KEY", "risk:halt")

# BESS keys
BESS_ID            = os.getenv("POWER_BESS_ID", "BESS-01").upper()
BESS_STATE_HK      = os.getenv("POWER_BESS_STATE_HK", "bess:state")
BESS_BANDS_HK      = os.getenv("POWER_BESS_BANDS_HK", "bess:price_bands")
RT_NOW_HK          = os.getenv("POWER_RT_NOW_HK", "lmp:rt_now")

# ===================== Redis =====================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ===================== helpers =====================
def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw);  return j if isinstance(j, dict) else None
    except Exception:
        return None

def _now_ms() -> int: return int(time.time()*1000)

def _mw_cap_for(node: str) -> float:
    v = _hgetf(CAPS_MW_HKEY, node)
    return v if v is not None else PER_NODE_MW_CAP

def _fees_bps() -> float:
    v = _hgetf(FEES_HKEY, ISO)
    return float(v) if v is not None else 5.0  # rough default

# ===================== EWMA =====================
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

def _ewma_key() -> str:
    return f"power:ewma:{ISO}:{NODE}:{HOUR}"

def _load_ewma() -> EwmaMV:
    raw = r.get(_ewma_key())
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(ew: EwmaMV) -> None:
    r.set(_ewma_key(), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ===================== state =====================
@dataclass
class OpenState:
    side: str      # "INC" or "DEC"
    mw: float
    entry_spread: float
    entry_z: float
    ts_ms: int

def _poskey(name: str) -> str:
    return f"power:open:{name}:{ISO}:{NODE}:{HOUR}"

# ===================== strategy =====================
class PowerMarketArbitrage(Strategy):
    """
    DA↔RT virtual bidding with EWMA+z gating; optional BESS time-shift overlay.
    """
    def __init__(self, name: str = "power_market_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "iso": ISO, "node": NODE, "hour": HOUR, "mode": MODE, "ts": _now_ms()
        }))

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1":
            return
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now

        if MODE in ("VBID", "BOTH"):
            self._eval_vbid()

        if MODE in ("BESS", "BOTH"):
            self._eval_bess()

    # ---------------- VBID (DA vs RT) ----------------
    def _eval_vbid(self) -> None:
        da = _hgetf(DA_HKEY, KEY)
        rt = _hgetf(RTF_HKEY, KEY)
        if da is None or rt is None: return

        # net expected edge after fees (bps convert to $/MWh on DA)
        fees_bps = _fees_bps()
        fees_abs = da * (fees_bps * 1e-4)
        spread = (rt - da) - fees_abs  # $/MWh

        # optional dynamic threshold using vol_bps
        vol_bps = _hgetf(VOL_HKEY, KEY)
        dyn_entry = da * (vol_bps * 1e-4) if vol_bps is not None else ENTRY_USD
        dyn_exit  = da * (0.5 * (vol_bps * 1e-4)) if vol_bps is not None else EXIT_USD

        ew = _load_ewma(); m, v = ew.update(spread); _save_ewma(ew)
        z = (spread - m) / math.sqrt(max(v, 1e-12))

        # Monitoring signal (scaled to entry)
        self.emit_signal(max(-1.0, min(1.0, spread / max(0.01, dyn_entry))))

        st = self._load_state()

        # Exits: if a position exists, close administratively when edge mean-reverts pre‑submission window.
        if st:
            if (abs(spread) <= dyn_exit) or (abs(z) <= EXIT_Z):
                self._cancel(st)
            return

        # Entries: simple one‑ticket per node-hour
        if r.get(_poskey(self.ctx.name)) is not None:
            return
        if not (abs(spread) >= dyn_entry and abs(z) >= ENTRY_Z):
            return

        # Risk caps
        node_cap_mw = _mw_cap_for(NODE)
        mw = min(MW_BASE, node_cap_mw)
        if mw < MW_MIN_TICKET:
            return

        # Portfolio notional check (rough): cap by USD_PORTF_CAP using DA price
        usd_guess = mw * da
        portf_cap = _hgetf(CAPS_USD_HKEY, ISO) or USD_PORTF_CAP
        # (Optional) track aggregated usage in r.get("power:used_usd:ISO"), omitted here for brevity.

        if usd_guess > portf_cap:
            mw = max(MW_MIN_TICKET, portf_cap / max(0.01, da))

        if spread > 0:
            # Expect RT > DA ⇒ INC (buy DA, sell RT)
            self.order(f"VBID:INC:{ISO}:{NODE}:{HOUR}", "buy", qty=mw, order_type="market", venue=ISO)
            side = "INC"
        else:
            # Expect RT < DA ⇒ DEC (sell DA, buy RT)
            self.order(f"VBID:DEC:{ISO}:{NODE}:{HOUR}", "sell", qty=mw, order_type="market", venue=ISO)
            side = "DEC"

        self._save_state(OpenState(side=side, mw=mw, entry_spread=spread, entry_z=z, ts_ms=_now_ms()))

    def _cancel(self, st: OpenState) -> None:
        # Paper cancel for visual; real ISOs have submission/cancel windows – adapter should enforce windows
        self.order(f"VBID:{st.side}:{ISO}:{NODE}:{HOUR}", "cancel", qty=st.mw, order_type="cancel", venue=ISO)
        r.delete(_poskey(self.ctx.name))

    # ---------------- BESS (time‑shift) ----------------
    def _eval_bess(self) -> None:
        state = _hget_json(BESS_STATE_HK, BESS_ID)
        bands = _hget_json(BESS_BANDS_HK, BESS_ID)
        if not (state and bands): return

        soc = float(state.get("soc_mwh", 0.0))
        emax = float(state.get("e_max_mwh", 0.0))
        pmax = float(state.get("p_max_mw", 0.0))
        eff_ch = float(state.get("eff_ch", 0.94))
        eff_rt = float(state.get("eff_rt", 0.94))

        rt_now = _hgetf(RT_NOW_HK, NODE)
        if rt_now is None: return

        buy_band = float(bands.get("buy", -1e9))
        sell_band = float(bands.get("sell", 1e9))

        # Simple band logic (one step per tick, 1h granularity assumed; adapters can sub‑hourly dispatch)
        if rt_now <= buy_band and soc < emax:
            charge_mw = min(pmax, max(0.0, (emax - soc)))  # 1h step
            if charge_mw > 0.0:
                self.order(f"BESS:{BESS_ID}:CHARGE", "buy", qty=charge_mw, order_type="market", venue=ISO,
                           flags={"price": rt_now, "eff": eff_ch})
        elif rt_now >= sell_band and soc > 0:
            discharge_mw = min(pmax, soc)  # 1h step
            if discharge_mw > 0.0:
                self.order(f"BESS:{BESS_ID}:DISCHARGE", "sell", qty=discharge_mw, order_type="market", venue=ISO,
                           flags={"price": rt_now, "eff": eff_rt})

    # ---------------- state I/O ----------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(side=str(o["side"]), mw=float(o["mw"]),
                             entry_spread=float(o["entry_spread"]), entry_z=float(o["entry_z"]),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name), json.dumps({
            "side": st.side, "mw": st.mw, "entry_spread": st.entry_spread,
            "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))