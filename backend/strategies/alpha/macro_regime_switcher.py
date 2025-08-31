# backend/strategies/diversified/macro_regime_switcher.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import redis
from backend.engine.strategy_base import Strategy

"""
Macro Regime Switcher — paper
-----------------------------
Purpose:
  • Ingest macro factors (inflation, growth, unemployment, PMI, yield curve, credit spreads, VIX, USD, commodities).
  • Produce regime probabilities (Risk‑On, Risk‑Off, Inflationary, Stagflation, Deflation‑Scare).
  • Map regime → asset tilts and place/adjust paper positions (ETF/Index proxies) with hysteresis.

Redis you publish elsewhere (examples; update daily/weekly/intraday as you like):

  # Macro factors (unitless or % as decimals). Use latest prints or nowcasts.
  HSET macro:factors "CPI_YOY"         0.032      # 3.2% YoY
  HSET macro:factors "CORE_CPI_YOY"    0.034
  HSET macro:factors "PMI_MFG"         49.2
  HSET macro:factors "PMI_SERV"        51.4
  HSET macro:factors "UNEMP"           0.040      # 4.0% unemployment
  HSET macro:factors "YC_10Y_2Y"      -0.60       # 10y-2y in %
  HSET macro:factors "CREDIT_IG_OAS"   0.013      # 130 bps
  HSET macro:factors "CREDIT_HY_OAS"   0.040      # 400 bps
  HSET macro:factors "VIX"            17.5
  HSET macro:factors "USD_TWI"       103.2
  HSET macro:factors "OIL"            82.0
  HSET macro:factors "BREAKEVEN_5Y"   0.024
  HSET macro:factors "GDP_NOWCAST"     0.020      # QoQ SAAR ~ 2.0% -> express as decimal

  # Prices (routing sanity for sizing)
  HSET last_price "ETF:SPY"   '{"price": 530.0}'
  HSET last_price "ETF:IEF"   '{"price": 96.5}'
  HSET last_price "ETF:TLT"   '{"price": 95.2}'
  HSET last_price "ETF:LQD"   '{"price": 112.3}'
  HSET last_price "ETF:HYG"   '{"price": 78.1}'
  HSET last_price "ETF:GLD"   '{"price": 215.0}'
  HSET last_price "ETF:DBC"   '{"price": 24.7}'     # broad commodities
  HSET last_price "ETF:UUP"   '{"price": 31.2}'     # USD proxy
  HSET last_price "CEX:COINBASE:BTCUSD" '{"price": 67000.0}'

  # Ops / fees
  HSET fees:etf EXCH 2       # bps
  HSET fees:crypto CEX 6     # bps
  SET  risk:halt 0|1

Routing (paper; adapters wire later):
  order("ETF:<TICKER>", side, qty, order_type="market", venue="EXCH")
  order("CEX:COINBASE:BTCUSD", side, qty, order_type="market", venue="COINBASE")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("MRS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("MRS_REDIS_PORT", "6379"))

RECHECK_SECS    = float(os.getenv("MRS_RECHECK_SECS", "5.0"))
ENTRY_CONF      = float(os.getenv("MRS_ENTRY_CONF",   "0.55"))  # min regime probability to act
EXIT_CONF       = float(os.getenv("MRS_EXIT_CONF",    "0.48"))  # hysteresis to avoid churn
COOLDOWN_SECS   = float(os.getenv("MRS_COOLDOWN_SECS","900"))   # min 15 min between rebalances
MAX_GROSS_USD   = float(os.getenv("MRS_MAX_GROSS_USD","50000"))
MIN_TICKET_USD  = float(os.getenv("MRS_MIN_TICKET_USD","200"))
LOT             = float(os.getenv("MRS_LOT", "1"))

# Redis keys
HALT_KEY   = os.getenv("MRS_HALT_KEY",   "risk:halt")
FACT_HK    = os.getenv("MRS_FACT_HK",    "macro:factors")
LAST_HK    = os.getenv("MRS_LAST_HK",    "last_price")
FEES_ETF   = os.getenv("MRS_FEES_ETF",   "fees:etf")
FEES_CRYP  = os.getenv("MRS_FEES_CRYP",  "fees:crypto")

STATE_HK   = os.getenv("MRS_STATE_HK",   "mrs:state")    # per strategy state blob

# Universe / proxies (override via env if you prefer different tickers)
PROXIES = {
    "EQUITIES":    "ETF:SPY",
    "UST_7_10Y":   "ETF:IEF",
    "UST_20Y":     "ETF:TLT",
    "IG_CREDIT":   "ETF:LQD",
    "HY_CREDIT":   "ETF:HYG",
    "GOLD":        "ETF:GLD",
    "COMMODS":     "ETF:DBC",
    "USD":         "ETF:UUP",
    "BTCUSD":      "CEX:COINBASE:BTCUSD",
}

# Regimes
REGIMES = ["RISK_ON", "RISK_OFF", "INFLATIONARY", "STAGFLATION", "DEF_SCARE"]

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

def _px(sym: str) -> Optional[float]:
    return _hgetf(LAST_HK, sym)

def _fees_bps(hk: str, venue: str) -> float:
    v = r.hget(hk, venue)
    try: return float(v) if v is not None else 5.0 # type: ignore
    except Exception: return 5.0

def _now_s() -> float: return time.time()

# ============================ regime model ============================
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _std(x: float, mean: float, sd: float) -> float:
    return (x - mean) / max(1e-6, sd)

def _probabilities(f: Dict[str, float]) -> Dict[str, float]:
    """
    Lightweight logistic blend. We z the inputs around rough priors and run 5 one‑vs‑rest logits.
    Tunables are simple and can be calibrated later.
    """
    # Priors / scales (rough defaults; adjust as you ingest real series)
    pri = dict(
        CPI_YOY=(0.025, 0.01), CORE_CPI_YOY=(0.025, 0.01),
        PMI_MFG=(50.0, 2.5), PMI_SERV=(50.0, 2.5),
        UNEMP=(0.045, 0.01),
        YC_10Y_2Y=(0.00, 0.5),
        CREDIT_IG_OAS=(0.015, 0.005), CREDIT_HY_OAS=(0.045, 0.015),
        VIX=(18.0, 6.0),
        USD_TWI=(100.0, 5.0),
        OIL=(75.0, 15.0),
        BREAKEVEN_5Y=(0.022, 0.006),
        GDP_NOWCAST=(0.02, 0.01),
    )
    z = {k: _std(float(f.get(k, pri[k][0])), pri[k][0], pri[k][1]) for k in pri.keys()}

    # Score functions (simple linear forms)
    s_risk_on = +0.9*z["PMI_MFG"] + 0.6*z["PMI_SERV"] - 0.4*z["UNEMP"] - 0.6*z["VIX"] - 0.4*z["CREDIT_HY_OAS"] \
                + 0.3*z["GDP_NOWCAST"] + 0.2*z["YC_10Y_2Y"] - 0.2*z["USD_TWI"]
    s_risk_off= -0.8*z["PMI_MFG"] - 0.6*z["PMI_SERV"] + 0.6*z["UNEMP"] + 0.8*z["VIX"] + 0.6*z["CREDIT_HY_OAS"] \
                - 0.3*z["GDP_NOWCAST"] - 0.2*z["YC_10Y_2Y"] + 0.2*z["USD_TWI"]
    s_infl    = +0.9*z["CPI_YOY"] + 0.8*z["CORE_CPI_YOY"] + 0.6*z["BREAKEVEN_5Y"] \
                + 0.4*z["OIL"] - 0.3*z["USD_TWI"]
    s_stag    = +0.7*z["CPI_YOY"] + 0.6*z["CORE_CPI_YOY"] - 0.8*z["PMI_MFG"] - 0.6*z["PMI_SERV"] \
                + 0.4*z["UNEMP"] + 0.3*z["BREAKEVEN_5Y"]
    s_def     = -0.7*z["CPI_YOY"] - 0.6*z["BREAKEVEN_5Y"] - 0.5*z["OIL"] - 0.3*z["GDP_NOWCAST"] \
                + 0.6*z["USD_TWI"] - 0.2*z["YC_10Y_2Y"]

    raw = {
        "RISK_ON":      _sigmoid(s_risk_on),
        "RISK_OFF":     _sigmoid(s_risk_off),
        "INFLATIONARY": _sigmoid(s_infl),
        "STAGFLATION":  _sigmoid(s_stag),
        "DEF_SCARE":    _sigmoid(s_def),
    }
    # Normalize to sum=1 (softmax-ish but simple)
    tot = sum(raw.values())
    if tot <= 0: return {k: 1.0/len(raw) for k in raw}
    return {k: v/tot for k,v in raw.items()}

# ============================ regime → tilts ============================
# Target weights by regime (sum roughly to 1 gross long; we allow short via negatives)
REGIME_WEIGHTS = {
    "RISK_ON": {
        "EQUITIES":  +0.60, "HY_CREDIT": +0.15, "IG_CREDIT": +0.10,
        "UST_7_10Y": -0.10, "GOLD": -0.05, "COMMODS": +0.10, "USD": -0.10, "BTCUSD": +0.10
    },
    "RISK_OFF": {
        "EQUITIES":  -0.40, "HY_CREDIT": -0.15, "IG_CREDIT": +0.20,
        "UST_7_10Y": +0.20, "UST_20Y": +0.15, "GOLD": +0.10, "COMMODS": -0.05, "USD": +0.10
    },
    "INFLATIONARY": {
        "EQUITIES":  +0.15, "IG_CREDIT": -0.10, "HY_CREDIT": 0.00,
        "UST_7_10Y": -0.20, "UST_20Y": -0.10, "GOLD": +0.20, "COMMODS": +0.25, "USD": -0.05
    },
    "STAGFLATION": {
        "EQUITIES":  -0.20, "IG_CREDIT": 0.00, "HY_CREDIT": -0.10,
        "UST_7_10Y": +0.05, "UST_20Y": 0.00, "GOLD": +0.25, "COMMODS": +0.10, "USD": +0.05
    },
    "DEF_SCARE": {
        "EQUITIES":  -0.25, "HY_CREDIT": -0.10, "IG_CREDIT": +0.15,
        "UST_7_10Y": +0.25, "UST_20Y": +0.20, "GOLD": +0.05, "COMMODS": -0.10, "USD": +0.10
    }
}

# ============================ state ============================
@dataclass
class PositionState:
    regime: str
    weights: Dict[str, float]
    last_rebalance_s: float

def _state_key(ctx_name: str) -> str:
    return f"{STATE_HK}:{ctx_name}"

def _load_state(ctx_name: str) -> Optional[PositionState]:
    raw = r.get(_state_key(ctx_name))
    if not raw: return None
    try:
        o = json.loads(raw) # type: ignore
        return PositionState(regime=str(o["regime"]),
                             weights={k: float(v) for k,v in (o.get("weights") or {}).items()},
                             last_rebalance_s=float(o.get("last_rebalance_s", 0.0)))
    except Exception:
        return None

def _save_state(ctx_name: str, st: PositionState) -> None:
    r.set(_state_key(ctx_name), json.dumps({
        "regime": st.regime, "weights": st.weights, "last_rebalance_s": st.last_rebalance_s
    }))

# ============================ Strategy ============================
class MacroRegimeSwitcher(Strategy):
    """
    Macro regime → asset tilts with confidence gates, hysteresis and cooldown (paper).
    """
    def __init__(self, name: str = "macro_regime_switcher", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        # 1) Read factors
        f = {k: float(v) for k,v in (r.hgetall(FACT_HK) or {}).items() if v is not None} # type: ignore
        if not f:
            self.emit_signal(0.0); return

        # 2) Compute regime probabilities
        probs = _probabilities(f)
        # For UI: emit signed bias (+risk_on - risk_off, +infl - def)
        bias = (probs["RISK_ON"] - probs["RISK_OFF"]) + 0.5*(probs["INFLATIONARY"] - probs["DEF_SCARE"])
        self.emit_signal(max(-1.0, min(1.0, bias)))

        # 3) Choose winning regime with hysteresis
        winner, p = max(probs.items(), key=lambda kv: kv[1])
        st = _load_state(self.ctx.name)

        # hysteresis: if we already have a regime and its prob > EXIT_CONF, keep it; else switch when new>ENTRY_CONF
        current = st.regime if st else None
        if current and probs.get(current, 0.0) >= EXIT_CONF:
            winner = current
            p = probs[current]
        elif p < ENTRY_CONF:
            return  # no strong regime yet

        # cooldown
        if st and (now - st.last_rebalance_s < COOLDOWN_SECS):
            return

        # 4) Translate to target weights
        tgt = REGIME_WEIGHTS[winner].copy()

        # 5) Convert weights → orders vs current inventory (we assume flat every rebalance for paper simplicity)
        self._rebalance_to_weights(tgt)

        # 6) Save state
        _save_state(self.ctx.name, PositionState(regime=winner, weights=tgt, last_rebalance_s=now))

    # ---------------- rebalance helper ----------------
    def _rebalance_to_weights(self, tgt_w: Dict[str, float]) -> None:
        # Spend MAX_GROSS_USD on absolute weights; compute notional per sleeve.
        # For simplicity we flatten then open new target each rebalance (paper).
        budget = MAX_GROSS_USD
        for sleeve, w in tgt_w.items():
            if sleeve not in PROXIES: continue
            sym = PROXIES[sleeve]
            px = _px(sym)
            if not px or px <= 0: continue

            notional = abs(w) * budget
            if notional < MIN_TICKET_USD: continue
            qty = math.floor((notional / px) / max(1.0, LOT)) * LOT
            if qty <= 0: continue

            side = "buy" if w > 0 else "sell"
            venue = sym.split(":")[1] if ":" in sym else "EXCH"
            self.order(sym, side, qty=qty, order_type="market", venue=venue)