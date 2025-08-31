# backend/strategies/diversified/sentiment_policy_indicator.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional

import redis
from backend.engine.strategy_base import Strategy

"""
Sentiment Policy Indicator — paper
----------------------------------
Goal:
  • Convert live policy tone (hawkish↔dovish) into positions: duration, USD, equities, gold.
Inputs you publish to Redis (examples; update around speeches/minutes/meetings and intraday):

# NLP + event surprises (all in normalized ranges)
HSET policy:agg "FED" '{
  "tone_nlp": -0.35,            # [-1,+1] dovish↔hawkish from speech/statement/minutes
  "tone_conf": 0.72,            # 0..1 confidence in NLP reading (speech quality, length)
  "dots_surprise_bps": +18,     # current year dot minus OIS (bps); + = hawkish surprise
  "term_path_surprise_bps": +10,# avg of next-4-quarters path vs OIS (bps)
  "minutes_bias": +0.20,        # [-1,+1] hawkishness from minutes summary
  "qt_qe_shift": +0.10,         # [-1,+1] +QT / -QE hawkishness
  "statement_shift": +0.15,     # [-1,+1] redlines delta → hawkish
  "speaker_weight": 0.9,        # 0..1 (Chair ~1.0, voter 0.7, non-voter 0.4)
  "event_type": "SPEECH",       # SPEECH | MINUTES | FOMC | TESTIMONY
  "updated_ms": 1765400000000
}'

# Market reaction (fast tape) — optional but helpful
HSET policy:tape "FED" '{
  "ust2y_move_bps_5m": +6.5,    # + = yields up → hawkish
  "ust5y_move_bps_5m": +4.2,
  "dxy_move_bps_5m": +8.0,      # + = USD up
  "gold_move_bps_5m": -5.0,     # + = gold up (often dovish)
  "eq_fut_move_bps_5m": -9.0,   # + = ES up (often dovish)
  "liq_score": 0.8,             # 0..1 microstructure/liquidity confidence
  "updated_ms": 1765400000000
}'

# Pricing for proxies (for sizing)
HSET last_price "ETF:SPY"  '{"price":530.0}'
HSET last_price "ETF:IEF"  '{"price":96.5}'
HSET last_price "ETF:TLT"  '{"price":95.2}'
HSET last_price "ETF:GLD"  '{"price":215.0}'
HSET last_price "ETF:UUP"  '{"price":31.2}'

# Fees & ops
HSET fees:etf EXCH 2
SET  risk:halt 0|1

Routing (paper; adapters later):
  order("ETF:<TICKER>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("SPI_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("SPI_REDIS_PORT", "6379"))

AGG_HK     = os.getenv("SPI_AGG_HK", "policy:agg")
TAPE_HK    = os.getenv("SPI_TAPE_HK", "policy:tape")
LAST_HK    = os.getenv("SPI_LAST_HK", "last_price")
FEES_HK    = os.getenv("SPI_FEES_HK", "fees:etf")
HALT_KEY   = os.getenv("SPI_HALT_KEY", "risk:halt")
STATE_KEY  = os.getenv("SPI_STATE_KEY", "spi:state")

# cadence / gates
RECHECK_SECS  = float(os.getenv("SPI_RECHECK_SECS", "1.5"))
ENTRY_TONE    = float(os.getenv("SPI_ENTRY_TONE",   "0.55"))  # |tone| threshold to act
EXIT_TONE     = float(os.getenv("SPI_EXIT_TONE",    "0.45"))  # hysteresis
COOLDOWN_SECS = float(os.getenv("SPI_COOLDOWN_SECS","600"))   # 10 min

# risk / sizing
MAX_GROSS_USD   = float(os.getenv("SPI_MAX_GROSS_USD", "50000"))
MIN_TICKET_USD  = float(os.getenv("SPI_MIN_TICKET_USD","200"))
LOT             = float(os.getenv("SPI_LOT", "1"))

# universe proxies
PROXIES = {
    "EQUITIES": "ETF:SPY",
    "UST_7_10": "ETF:IEF",
    "UST_20Y":  "ETF:TLT",
    "GOLD":     "ETF:GLD",
    "USD":      "ETF:UUP",
}

# target weights given hawkish (+1) vs dovish (‑1) tone
# Positive tone → hawkish: short duration/equities, long USD; negative tone → dovish opposite.
BASE_WEIGHTS_HAWK = {"EQUITIES": -0.20, "UST_7_10": -0.25, "UST_20Y": -0.20, "GOLD": -0.05, "USD": +0.20}
BASE_WEIGHTS_DOVE = {"EQUITIES": +0.20, "UST_7_10": +0.25, "UST_20Y": +0.20, "GOLD": +0.15, "USD": -0.20}

# blend strength caps
MAX_TILT_SCALE = float(os.getenv("SPI_MAX_TILT_SCALE", "1.0"))

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ utils ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try: 
        j = json.loads(raw);  # type: ignore
        return j if isinstance(j, dict) else None
    except Exception: 
        return None

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, sym)
    if raw is None: return None
    try:
        j = json.loads(raw) # type: ignore
        if isinstance(j, dict) and "price" in j: return float(j["price"])
        return float(raw) # type: ignore
    except Exception:
        return None

def _now() -> float: return time.time()

# ============================ model ============================
def _tone_score(agg: dict, tape: dict) -> float:
    """
    Return signed tone in [-1,+1]; + hawkish, - dovish.
    """
    tone = float(agg.get("tone_nlp", 0.0))
    conf = float(agg.get("tone_conf", 0.0))
    spk  = float(agg.get("speaker_weight", 0.5))

    # surprise from dots/path (bps → ~[-1,+1] via softsign)
    dots   = float(agg.get("dots_surprise_bps", 0.0))
    path   = float(agg.get("term_path_surprise_bps", 0.0))
    dot_s  = dots / (35.0 + abs(dots))      # 35 bps scale
    path_s = path / (35.0 + abs(path))

    minutes = float(agg.get("minutes_bias", 0.0))
    stmt    = float(agg.get("statement_shift", 0.0))
    qtqe    = float(agg.get("qt_qe_shift", 0.0))

    # tape confirmation (scaled to [-1,+1])
    u2 = float(tape.get("ust2y_move_bps_5m", 0.0))
    dxy = float(tape.get("dxy_move_bps_5m", 0.0))
    gold = float(tape.get("gold_move_bps_5m", 0.0))
    es = float(tape.get("eq_fut_move_bps_5m", 0.0))
    liq = float(tape.get("liq_score", 0.7))

    tape_hawk = 0.35*(u2/(10.0+abs(u2))) + 0.25*(dxy/(12.0+abs(dxy))) \
                + 0.20*(-gold/(12.0+abs(gold))) + 0.20*(-es/(12.0+abs(es)))  # equities up = dovish

    # core composite
    core = (0.38 * tone) \
         + (0.20 * dot_s) + (0.15 * path_s) \
         + (0.10 * minutes) + (0.07 * stmt) + (0.05 * qtqe) \
         + (0.20 * tape_hawk * liq)

    # confidence & speaker weight scale (0..1)
    scale = max(0.0, min(1.0, 0.5*conf + 0.5*spk))
    signed = max(-1.0, min(1.0, core * scale))
    return float(signed)

# ============================ state ============================
@dataclass
class SPIState:
    active: bool     # whether we're currently tilted (non‑flat)
    tone: float      # last tone score
    last_rebalance_s: float

def _state_key(ctx: str) -> str: return f"{STATE_KEY}:{ctx}"

def _load_state(ctx: str) -> Optional[SPIState]:
    raw = r.get(_state_key(ctx))
    if not raw: return None
    try:
        o = json.loads(raw) # type: ignore
        return SPIState(active=bool(o.get("active", False)),
                        tone=float(o.get("tone", 0.0)),
                        last_rebalance_s=float(o.get("last_rebalance_s", 0.0)))
    except Exception:
        return None

def _save_state(ctx: str, st: SPIState) -> None:
    r.set(_state_key(ctx), json.dumps({"active": st.active, "tone": st.tone, "last_rebalance_s": st.last_rebalance_s}))

# ============================ Strategy ============================
class SentimentPolicyIndicator(Strategy):
    """
    Reads policy tone and applies hawkish/dovish tilts with hysteresis and cooldown (paper).
    """
    def __init__(self, name: str = "sentiment_policy_indicator", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = _now()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        # Read current aggregates
        agg = _hget_json(AGG_HK, "FED")
        if not agg:
            self.emit_signal(0.0); return
        tape = _hget_json(TAPE_HK, "FED") or {}

        tone = _tone_score(agg, tape)  # [-1,+1]
        self.emit_signal(max(-1.0, min(1.0, tone)))  # UI heartbeat

        st = _load_state(self.ctx.name) or SPIState(active=False, tone=0.0, last_rebalance_s=0.0)

        # hysteresis
        abs_tone = abs(tone)
        want_active = st.active
        if st.active:
            if abs_tone < EXIT_TONE:
                want_active = False
        else:
            if abs_tone >= ENTRY_TONE:
                want_active = True

        # cooldown
        if now - st.last_rebalance_s < COOLDOWN_SECS:
            _save_state(self.ctx.name, SPIState(active=want_active, tone=tone, last_rebalance_s=st.last_rebalance_s))
            return

        # build targets
        if not want_active:
            tgt = {k: 0.0 for k in BASE_WEIGHTS_HAWK.keys()}
        else:
            if tone >= 0:
                base = BASE_WEIGHTS_HAWK
                strength = min(MAX_TILT_SCALE, 0.5 + 0.5*abs_tone)  # 0.5..1.0
            else:
                base = BASE_WEIGHTS_DOVE
                strength = min(MAX_TILT_SCALE, 0.5 + 0.5*abs_tone)
            tgt = {k: v*strength for k,v in base.items()}

        # rebalance (simple open/flatten per sleeve; delta-only is an easy future upgrade)
        self._rebalance_to(tgt)

        _save_state(self.ctx.name, SPIState(active=want_active, tone=tone, last_rebalance_s=now))

    # --------------- helpers ---------------
    def _rebalance_to(self, weights: Dict[str, float]) -> None:
        budget = MAX_GROSS_USD
        for sleeve, w in weights.items():
            sym = PROXIES.get(sleeve)
            if not sym: continue
            px = _px(sym)
            if not px or px <= 0: continue

            notional = abs(w) * budget
            if notional < MIN_TICKET_USD: continue

            qty = math.floor((notional / px) / max(1.0, LOT)) * LOT
            if qty <= 0: continue

            # For paper simplicity we just place target-direction orders each rebalance.
            side = "buy" if w > 0 else "sell"
            if w == 0.0:
                # Flatten: send both ways once (paper)
                self.order(sym, "buy",  qty=qty, order_type="market", venue="EXCH")
                self.order(sym, "sell", qty=qty, order_type="market", venue="EXCH")
            else:
                self.order(sym, side, qty=qty, order_type="market", venue="EXCH")