# backend/strategies/diversified/variance_swap_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Variance Swap Arbitrage — paper
-------------------------------
Modes:
  1) QUOTE_VS_FAIR
     • Build fair variance strike σ_var^2 from listed options using a discrete replication:
         σ_var^2 ≈ (2/T) * Σ w_i * OTM_price(K_i)/K_i^2  − (1/T) * adj_forward_term
       where w_i comes from trapezoidal rule over strikes and OTM_price(K) is the put for K<F
       and call for K>F. (We omit tiny discounting nuances in paper mode.)
     • Compare to quoted var level VAR_QUOTE (variance, not volatility).
       If VAR_QUOTE >> FAIR → SELL variance (short var swap / sell strip).
       If VAR_QUOTE << FAIR → BUY variance.

  2) INDEX_VS_COMPONENTS
     • Compute index fair variance (as above) and a cap‑weighted sum of component variances.
       Trade short index variance vs long components (or the reverse) when the spread exceeds gates.

Paper routing:
  • Synthetic OTC leg: "VAR_SWAP:<SYM>:<DTE>" (qty in variance notional units, USD per variance point)
  • Listed options replication (optional): "OPT_CALL:<SYM>:<K>:<DTE>", "OPT_PUT:<SYM>:<K>:<DTE>"
    (We default to the VAR_SWAP synthetic fill for clarity; flip a flag to place the strip instead.)

Redis feeds you publish elsewhere:

  # Prices (spot/forward and option mids)
  HSET last_price "EQ:<SYM>" '{"price": <spot>}'
  HSET fwd:level "<SYM>:<DTE>" <forward_level>                  # optional; else infer F≈spot
  HSET opt:surface "<SYM>:<DTE>" '{"strikes":[...],"call":[...],"put":[...]}'  # mids

  # Quoted variance swap (variance, not vol): e.g., 0.04 for 20% vol (since 0.2^2)
  HSET var:quote "<SYM>:<DTE>" 0.040

  # Variance notional (USD per variance point, i.e., per 0.01 in variance)
  HSET var:notional "<SYM>:<DTE>" 10000

  # Components (only for INDEX_VS_COMPONENTS)
  HSET var:weights "<SYM>:<DTE>" '{"weights":{"AAPL":0.07,"MSFT":0.06,...}}'
  HSET opt:surface "AAPL:<DTE>" '{"strikes":[...],"call":[...],"put":[...]}'
  HSET last_price "EQ:AAPL" '{"price": <spot>}'
  HSET var:quote "AAPL:<DTE>" 0.045     # optional; if missing, we compute fair only

  # Fees / guards
  HSET fees:var OTC 5          # bps guard on notional
  SET  risk:halt 0|1
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("VAR_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("VAR_REDIS_PORT", "6379"))

MODE      = os.getenv("VAR_MODE", "QUOTE_VS_FAIR").upper()   # QUOTE_VS_FAIR | INDEX_VS_COMPONENTS
SYM       = os.getenv("VAR_SYM", "SPX").upper()
DTE       = int(os.getenv("VAR_DTE", "30"))

# Thresholds / gating
ENTRY_VOL_BPS = float(os.getenv("VAR_ENTRY_VOL_BPS", "60"))  # edge threshold in **vol basis points** (100 bps = 1 vol pt)
EXIT_VOL_BPS  = float(os.getenv("VAR_EXIT_VOL_BPS",  "25"))
ENTRY_Z       = float(os.getenv("VAR_ENTRY_Z", "1.1"))
EXIT_Z        = float(os.getenv("VAR_EXIT_Z",  "0.5"))

# Sizing / risk
USD_BUDGET        = float(os.getenv("VAR_USD_BUDGET", "50000"))
MIN_TICKET_USD    = float(os.getenv("VAR_MIN_TICKET_USD", "500"))
USE_STRIP_ORDERS  = os.getenv("VAR_USE_STRIP", "0") == "1"   # if true, place listed option legs instead of VAR_SWAP synthetic

# Cadence / stats
RECHECK_SECS = float(os.getenv("VAR_RECHECK_SECS", "1.1"))
EWMA_ALPHA   = float(os.getenv("VAR_EWMA_ALPHA", "0.08"))

# Redis keys
HALT_KEY    = os.getenv("VAR_HALT_KEY", "risk:halt")
LAST_HK     = os.getenv("VAR_LAST_HK", "last_price")
FWD_HK      = os.getenv("VAR_FWD_HK", "fwd:level")
SURF_HK     = os.getenv("VAR_SURF_HK", "opt:surface")
QUOTE_HK    = os.getenv("VAR_QUOTE_HK", "var:quote")
NOTIONAL_HK = os.getenv("VAR_NOTIONAL_HK", "var:notional")
WEIGHTS_HK  = os.getenv("VAR_WEIGHTS_HK", "var:weights")
FEES_HK     = os.getenv("VAR_FEES_HK", "fees:var")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw); return j if isinstance(j, dict) else None
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
        except Exception: return None

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, sym)
    if not raw: return None
    try:
        j = json.loads(raw); return float(j.get("price", 0.0))
    except Exception:
        try: return float(raw)
        except Exception: return None

def _fees_bps(venue: str) -> float:
    v = r.hget(FEES_HK, venue)
    try: return float(v) if v is not None else 5.0
    except Exception: return 5.0

def _T_years(dte: int) -> float:
    return max(1.0/365.0, dte/365.0)

# ============================ surface & fair variance ============================
def _load_surface(sym: str, dte: int) -> Optional[Tuple[List[float], List[float], List[float]]]:
    j = _hget_json(SURF_HK, f"{sym}:{dte}")
    if not j: return None
    K = [float(x) for x in (j.get("strikes") or [])]
    C = [float(x) for x in (j.get("call") or [])]
    P = [float(x) for x in (j.get("put") or [])]
    if len(K) >= 8 and len(K) == len(C) == len(P): return K, C, P
    return None

def _forward(sym: str, dte: int) -> Optional[float]:
    f = _hgetf(FWD_HK, f"{sym}:{dte}")
    if f is not None: return f
    S = _px(f"EQ:{sym}")
    return S

def _fair_variance_from_surface(sym: str, dte: int) -> Optional[float]:
    """
    Discrete variant of the well-known replication:
      σ_var^2 ≈ (2/T) * Σ ΔK_i * OTM(K_i) / K_i^2  − (1/T) * ((F/K0 - 1)**2)   [approx]
    where ΔK_i is the strike step (trapezoidal weights). We pick K0≈F nearest strike.
    Returns variance (not vol): e.g., 0.04 for 20% vol.
    """
    surf = _load_surface(sym, dte);  F = _forward(sym, dte)
    if not surf or F is None or F <= 0: return None
    K, C, P = surf
    pairs = sorted(zip(K, C, P), key=lambda x: x[0])
    K = [k for (k,_,__) in pairs]
    C = [c for (_,c,__) in pairs]
    P = [p for (_,__,p) in pairs]

    # Choose K0 (closest strike to F)
    idx0 = min(range(len(K)), key=lambda i: abs(K[i] - F))
    K0 = K[idx0]

    # Build OTM payoffs: for K<=K0 use PUT price; for K>=K0 use CALL price
    O = [ (P[i] if K[i] <= K0 else C[i]) for i in range(len(K)) ]

    # Trapezoidal integration over strikes (ΔK_i)
    var_sum = 0.0
    for i in range(1, len(K)):
        dK = K[i] - K[i-1]
        if dK <= 0: continue
        Obar = 0.5 * (O[i] + O[i-1])
        Kbar = 0.5 * (K[i] + K[i-1])
        var_sum += (Obar / max(1e-12, Kbar*Kbar)) * dK

    T = _T_years(dte)
    fair_var = (2.0 / max(1e-9, T)) * var_sum
    # Small forward adjustment term (optional; improves center): (F/K0 - 1)^2 / T
    fair_var -= (1.0 / max(1e-9, T)) * ((F / max(1e-9, K0) - 1.0) ** 2)
    return max(1e-6, fair_var)

# ============================ EWMA ============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float, float]:
        m0 = self.mean
        self.mean = (1 - self.alpha)*self.mean + self.alpha*x
        self.var  = max(1e-12, (1 - self.alpha)*(self.var + (x - m0)*(x - self.mean)))
        return self.mean, self.var

def _ewma_key(tag: str) -> str: return f"var:ewma:{tag}"

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
    side: str          # "sell_var" | "buy_var" | "short_index_long_components" | "long_index_short_components"
    qty: float         # variance notional (units of USD per variance point)
    entry_vol_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str: return f"var:open:{name}:{tag}"

# ============================ Strategy ============================
class VarianceSwapArbitrage(Strategy):
    """
    Quote-vs-fair and index-vs-components variance dislocations (paper).
    """
    def __init__(self, name: str = "variance_swap_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "INDEX_VS_COMPONENTS":
            self._eval_index_vs_components()
        else:
            self._eval_quote_vs_fair()

    # --------------- QUOTE_VS_FAIR ---------------
    def _eval_quote_vs_fair(self) -> None:
        tag = f"QVF:{SYM}:{DTE}"

        fair = _fair_variance_from_surface(SYM, DTE)
        quote = _hgetf(QUOTE_HK, f"{SYM}:{DTE}")
        notional = _hgetf(NOTIONAL_HK, f"{SYM}:{DTE}") or 10000.0
        if fair is None or quote is None: return

        # Convert variance gap to **vol basis points** (so it's tangible)
        # Let vol_fair = sqrt(fair), vol_q = sqrt(quote); edge_vol_bps = 1e4*(vol_q - vol_fair)
        vol_fair = math.sqrt(max(1e-12, fair))
        vol_q = math.sqrt(max(1e-12, quote))
        edge_vol_bps = 1e4 * (vol_q - vol_fair)  # positive ⇒ quoted vol higher (rich)

        ew = _load_ewma(tag); m,v = ew.update(edge_vol_bps); _save_ewma(tag, ew)
        z = (edge_vol_bps - m) / math.sqrt(max(v, 1e-12))

        # dashboard signal
        self.emit_signal(max(-1.0, min(1.0, edge_vol_bps / max(1.0, ENTRY_VOL_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_vol_bps) <= EXIT_VOL_BPS) or (abs(z) <= EXIT_Z):
                self._close_qvf(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_vol_bps) >= ENTRY_VOL_BPS and abs(z) >= ENTRY_Z): return

        # Sizing: spend a slice of USD_BUDGET on variance notional
        fee = _fees_bps("OTC") * 1e-4
        qty = max(1.0, math.floor(USD_BUDGET / max(1.0, notional)))  # number of var notional units
        if qty * notional < MIN_TICKET_USD: return

        if edge_vol_bps > 0:
            # Quoted variance rich → SELL variance (receive VAR_SWAP at quote)
            self.order(f"VAR_SWAP:{SYM}:{DTE}", "sell", qty=qty, order_type="market", venue="OTC")
            side = "sell_var"
            if USE_STRIP_ORDERS:
                self._optional_place_strip(SYM, DTE, qty, sell_strip=True)
        else:
            # Quoted variance cheap → BUY variance
            self.order(f"VAR_SWAP:{SYM}:{DTE}", "buy", qty=qty, order_type="market", venue="OTC")
            side = "buy_var"
            if USE_STRIP_ORDERS:
                self._optional_place_strip(SYM, DTE, qty, sell_strip=False)

        self._save_state(tag, OpenState(mode="QUOTE_VS_FAIR", side=side, qty=qty,
                                        entry_vol_bps=edge_vol_bps, entry_z=z, ts_ms=int(time.time()*1000)))

    def _close_qvf(self, tag: str, st: OpenState) -> None:
        # Reverse the synthetic var swap
        side = "buy" if st.side == "sell_var" else "sell"
        self.order(f"VAR_SWAP:{SYM}:{DTE}", side, qty=st.qty, order_type="market", venue="OTC")
        r.delete(_poskey(self.ctx.name, tag))

    def _optional_place_strip(self, sym: str, dte: int, qty_units: float, sell_strip: bool) -> None:
        """
        Optional: place a coarse OTM strip (10 wings) proportional to ΔK spacing.
        This is *illustrative*; real replication needs delta‑hedging.
        """
        surf = _load_surface(sym, dte); F = _forward(sym, dte)
        if not surf or F is None: return
        K, C, P = surf
        pairs = sorted(zip(K, C, P), key=lambda x: x[0])
        K = [k for (k,_,__) in pairs]; C = [c for (_,c,__) in pairs]; P = [p for (_,__,p) in pairs]
        idx0 = min(range(len(K)), key=lambda i: abs(K[i]-F))

        # pick 5 puts below and 5 calls above
        put_idx = list(range(max(1, idx0-5), idx0+1))
        call_idx= list(range(idx0, min(len(K)-1, idx0+5)))
        legs = []
        for i in put_idx:
            dK = K[i] - K[i-1] if i>0 else (K[i+1] - K[i])  # local spacing proxy
            contracts = max(1.0, math.floor( (qty_units * dK / max(1.0, K[i])) ))
            legs.append(("OPT_PUT", K[i], contracts))
        for i in call_idx:
            dK = K[i+1] - K[i] if i < len(K)-1 else (K[i] - K[i-1])
            contracts = max(1.0, math.floor( (qty_units * dK / max(1.0, K[i])) ))
            legs.append(("OPT_CALL", K[i], contracts))

        for right, k, q in legs:
            side = "sell" if sell_strip else "buy"
            self.order(f"{right}:{sym}:{int(round(k))}:{dte}", side, qty=q, order_type="market", venue="OPT")

    # --------------- INDEX_VS_COMPONENTS ---------------
    def _eval_index_vs_components(self) -> None:
        tag = f"IDXCOMP:{SYM}:{DTE}"

        fair_idx = _fair_variance_from_surface(SYM, DTE)
        if fair_idx is None: return

        wj = _hget_json(WEIGHTS_HK, f"{SYM}:{DTE}") or {}
        weights: Dict[str, float] = (wj.get("weights") or {})
        if not weights: return

        # Compute cap-weighted component variances (fair if quote missing)
        var_comp = 0.0
        for ticker, w in weights.items():
            w = float(w)
            fv = _fair_variance_from_surface(ticker.upper(), DTE)
            if fv is None:
                qv = _hgetf(QUOTE_HK, f"{ticker.upper()}:{DTE}")
                fv = qv if qv is not None else None
            if fv is None: return
            var_comp += w * fv

        # Translate to vol‑bps edge: vol_idx - sqrt(var_comp)
        vol_idx = math.sqrt(max(1e-12, fair_idx))
        vol_basket = math.sqrt(max(1e-12, var_comp))
        edge_vol_bps = 1e4 * (vol_idx - vol_basket)  # positive ⇒ index vol richer than comp basket

        ew = _load_ewma(tag); m,v = ew.update(edge_vol_bps); _save_ewma(tag, ew)
        z = (edge_vol_bps - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_vol_bps / max(1.0, ENTRY_VOL_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_vol_bps) <= EXIT_VOL_BPS) or (abs(z) <= EXIT_Z):
                self._close_disp(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_vol_bps) >= ENTRY_VOL_BPS and abs(z) >= ENTRY_Z): return

        notional = _hgetf(NOTIONAL_HK, f"{SYM}:{DTE}") or 10000.0
        qty = max(1.0, math.floor(USD_BUDGET / max(1.0, notional)))
        if qty * notional < MIN_TICKET_USD: return

        if edge_vol_bps > 0:
            # Index variance rich → SHORT index var, LONG component vars
            self.order(f"VAR_SWAP:{SYM}:{DTE}", "sell", qty=qty, order_type="market", venue="OTC")
            for ticker, w in weights.items():
                q = max(1.0, math.floor(qty * float(w)))
                self.order(f"VAR_SWAP:{ticker.upper()}:{DTE}", "buy", qty=q, order_type="market", venue="OTC")
            side = "short_index_long_components"
        else:
            # Index variance cheap → LONG index, SHORT components
            self.order(f"VAR_SWAP:{SYM}:{DTE}", "buy", qty=qty, order_type="market", venue="OTC")
            for ticker, w in weights.items():
                q = max(1.0, math.floor(qty * float(w)))
                self.order(f"VAR_SWAP:{ticker.upper()}:{DTE}", "sell", qty=q, order_type="market", venue="OTC")
            side = "long_index_short_components"

        self._save_state(tag, OpenState(mode="INDEX_VS_COMPONENTS", side=side, qty=qty,
                                        entry_vol_bps=edge_vol_bps, entry_z=z, ts_ms=int(time.time()*1000)))

    def _close_disp(self, tag: str, st: OpenState) -> None:
        # In a real adapter we’d track per-leg sizes; here we just reverse the index leg.
        rev = "buy" if st.side == "short_index_long_components" else "sell"
        self.order(f"VAR_SWAP:{SYM}:{DTE}", rev, qty=st.qty, order_type="market", venue="OTC")
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- state I/O ---------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(mode=str(o["mode"]), side=str(o["side"]), qty=float(o["qty"]),
                             entry_vol_bps=float(o["entry_vol_bps"]), entry_z=float(o["entry_z"]),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side, "qty": st.qty,
            "entry_vol_bps": st.entry_vol_bps, "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))