# backend/strategies/diversified/onchain_settlement_arbitrage.py
from __future__ import annotations

import json, math, os, time, uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
On-Chain Settlement Arbitrage (CEX ↔ DEX with perp hedge) — paper
-----------------------------------------------------------------
Playbook (example: spot cheaper on DEX, richer on CEX):
  1) BUY token on DEX (on-chain)  → simultaneously SHORT perp (hedge 100%)
  2) Bridge/withdraw token to CEX wallet; once received, SELL spot on CEX
  3) CLOSE perp hedge → realize basis - costs

Reverse direction mirrors the steps.

We only trigger when **net executable edge** clears:
  edge_net = price_rich - price_cheap
             - (gas + bridge + withdrawal + taker fees + est slippage) per token
             - carry_guard (hedge funding/financing over T_settle)

We also enforce **latency/MEV guards** and **inventory caps**.

Redis feeds (publish these from your adapters/routers):
  # Top-of-book / quotes
  HSET ob:cex:<SYM> '{"bid":..., "ask":..., "taker_bps": ...}'
  HSET quote:dex:<CHAIN>:<SYM> '{"buy_px":..., "sell_px":..., "slip_bps": ...}'  # buy_px = what you pay to buy 1 unit now

  # Gas & bridge/fees (absolute per *trade unit*, i.e., per 1 token; adapters can convert)
  HSET fees:gas:<CHAIN> gwei <gwei>
  HSET fees:gas:<CHAIN> usd_per_unit <usd>
  HSET fees:bridge <SYM> <usd_per_unit>
  HSET fees:withdraw:<CEX>:<SYM> <usd_per_unit>
  HSET fees:taker:<CEX> <bps>   # fallback if not in ob:cex

  # Funding/carry guard (per hour in decimals; used for perp hedging carry)
  HSET funding:<PERP_EXCH> <SYM> <per_hour_decimal>

  # Wallet balances & transfer status (adapters update)
  HSET bal:cex:<CEX> <SYM> <units>
  HSET bal:dex:<CHAIN> <SYM> <units>
  HSET xfer:status:<TXID> '{"status":"pending|confirmed|failed","rx_chain":"...","ts":...}'

  # Kill switch (global)
  SET risk:halt 0|1

Paper symbols you’ll map in adapters:
  • DEX trade: "DEX:<CHAIN>:<SYM>"
  • CEX spot : "CEX:<CEX>:<SYM>"
  • PERP     : "PERP:<EXCH>:<SYM>"

State machine:
  IDLE → ENTER (trade+hedge) → INFLIGHT (waiting transfer) → SETTLE (offload) → DONE/ERROR
"""

# ====================== ENV / CONFIG ======================
REDIS_HOST = os.getenv("OSA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("OSA_REDIS_PORT", "6379"))

SYM        = os.getenv("OSA_SYMBOL", "ETHUSDT").upper()
CHAIN      = os.getenv("OSA_CHAIN", "ARBITRUM").upper()
CEX        = os.getenv("OSA_CEX", "BINANCE").upper()
PERP_EXCH  = os.getenv("OSA_PERP_EXCH", "BINANCE_PERP").upper()

# thresholds
ENTRY_USD_MIN   = float(os.getenv("OSA_ENTRY_USD_MIN", "15"))     # min net edge (USD) for the whole package
ENTRY_BPS_MIN   = float(os.getenv("OSA_ENTRY_BPS_MIN", "8"))      # min bps over rich leg
EXIT_BPS_BACKOUT= float(os.getenv("OSA_EXIT_BPS_BACKOUT", "3"))   # cancel if edge collapses below this before commit
MAX_SETTLE_MINS = float(os.getenv("OSA_MAX_SETTLE_MINS", "20"))   # give up if transfer exceeds this

# sizing / risk
USD_NOTIONAL    = float(os.getenv("OSA_USD_NOTIONAL", "1000"))
MIN_TICKET_USD  = float(os.getenv("OSA_MIN_TICKET_USD", "100"))
POS_CAP_UNITS   = float(os.getenv("OSA_POS_CAP_UNITS", "10"))     # max inventory token units
HEDGE_RATIO     = float(os.getenv("OSA_HEDGE_RATIO", "1.0"))      # 0.95..1.05 typical

# guards
MEV_SLIP_BPS_GUARD = float(os.getenv("OSA_MEV_SLIP_BPS", "5.0"))  # extra bps added to DEX side
CARRY_GUARD_BPS_HR = float(os.getenv("OSA_CARRY_BPS_PER_HR", "1.0"))  # perp funding cushion per hour
CONFIRM_WAIT_SECS  = float(os.getenv("OSA_CONFIRM_WAIT_SECS", "2.0"))  # poll cadence

# venues (advisory)
VENUE_DEX   = os.getenv("OSA_VENUE_DEX", f"DEX:{CHAIN}").upper()
VENUE_CEX   = os.getenv("OSA_VENUE_CEX", f"CEX:{CEX}").upper()
VENUE_PERP  = os.getenv("OSA_VENUE_PERP", f"PERP:{PERP_EXCH}").upper()

# redis keys
OB_CEX_HKEY   = os.getenv("OSA_OB_CEX", f"ob:cex:{SYM}")
Q_DEX_HKEY    = os.getenv("OSA_Q_DEX",  f"quote:dex:{CHAIN}:{SYM}")
GAS_HKEY      = os.getenv("OSA_GAS_HK", f"fees:gas:{CHAIN}")
BRIDGE_HKEY   = os.getenv("OSA_BRIDGE_HK", "fees:bridge")
WDRAW_HKEY    = os.getenv("OSA_WDRAW_HK", f"fees:withdraw:{CEX}:{SYM}")
TAKER_CEX_HK  = os.getenv("OSA_TAKER_CEX", f"fees:taker:{CEX}")
FUNDING_HK    = os.getenv("OSA_FUNDING_HK", f"funding:{PERP_EXCH}")
BAL_CEX_HK    = os.getenv("OSA_BAL_CEX", f"bal:cex:{CEX}")
BAL_DEX_HK    = os.getenv("OSA_BAL_DEX", f"bal:dex:{CHAIN}")
XFER_STATUS_PFX= os.getenv("OSA_XFER_PFX", "xfer:status:")

HALT_KEY      = os.getenv("OSA_HALT_KEY", "risk:halt")

# ====================== Redis ======================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ====================== helpers ======================
def _now_ms() -> int: return int(time.time() * 1000)

def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try: return json.loads(raw) # type: ignore
    except Exception: return None

def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _ob_cex(sym: str) -> Optional[Tuple[float,float,float]]:
    o = _hget_json(OB_CEX_HKEY, sym)
    if not o: return None
    bid, ask = float(o.get("bid",0)), float(o.get("ask",0))
    taker_bps = float(o.get("taker_bps", _hgetf(TAKER_CEX_HK, CEX) or 4.0))
    if bid<=0 or ask<=0: return None
    return bid, ask, taker_bps

def _q_dex(chain: str, sym: str) -> Optional[Tuple[float,float,float]]:
    # buy_px: cost to buy 1 unit; sell_px: proceeds if selling 1 unit now
    o = _hget_json(Q_DEX_HKEY, sym)
    if not o: return None
    buy_px, sell_px = float(o.get("buy_px",0)), float(o.get("sell_px",0))
    slip_bps = float(o.get("slip_bps", 0))
    if buy_px<=0 or sell_px<=0: return None
    return buy_px, sell_px, slip_bps

def _fees_per_unit_usd(sym: str) -> float:
    gas = _hgetf(GAS_HKEY, "usd_per_unit") or 0.0
    bridge = _hgetf(BRIDGE_HKEY, sym) or 0.0
    withdraw = _hgetf(WDRAW_HKEY, sym) or 0.0
    return gas + bridge + withdraw

def _funding_per_hr(sym: str) -> float:
    return _hgetf(FUNDING_HK, sym) or 0.0  # decimal per hr (e.g., 0.0005 = 5 bps per hr)

# ====================== state ======================
@dataclass
class OpenState:
    state: str         # IDLE|ENTERED|INFLIGHT|SETTLE
    direction: str     # "DEX_TO_CEX" | "CEX_TO_DEX"
    qty: float
    avg_px_cheap: float
    avg_px_rich: float
    perp_qty: float
    txid: str
    start_ms: int

def _poskey(name: str) -> str:
    return f"osa:open:{name}:{CHAIN}:{CEX}:{SYM}"

# ====================== strategy ======================
class OnchainSettlementArbitrage(Strategy):
    def __init__(self, name: str = "onchain_settlement_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1":
            return
        now = time.time()
        if now - self._last < 0.5:
            return
        self._last = now

        st = self._load_state()
        if not st:
            self._maybe_enter()
        else:
            if st.state in ("ENTERED","INFLIGHT"):
                self._poll_transfer_and_maybe_settle(st)
            elif st.state == "SETTLE":
                self._finalize(st)

    # ------------ signals & entry ------------
    def _edge_and_paths(self) -> Optional[Tuple[str, float, float, float, float, float, float]]:
        ob = _ob_cex(SYM); qd = _q_dex(CHAIN, SYM)
        if not ob or not qd: return None
        c_bid, c_ask, c_taker_bps = ob
        d_buy, d_sell, d_slip_bps = qd

        fees_unit = _fees_per_unit_usd(SYM)
        carry_bps_hr = max(CARRY_GUARD_BPS_HR, 1e4 * _funding_per_hr(SYM))  # convert decimal→bps if using funding
        est_hours = max(0.1, (MAX_SETTLE_MINS / 60.0))
        carry_guard = (carry_bps_hr * est_hours) * 1e-4  # as fraction of price

        # Add conservative slippage guards
        dex_buy = d_buy * (1 + (d_slip_bps + MEV_SLIP_BPS_GUARD) * 1e-4)
        dex_sell= d_sell * (1 - (d_slip_bps + MEV_SLIP_BPS_GUARD) * 1e-4)
        cex_buy = c_ask * (1 + c_taker_bps * 1e-4)
        cex_sell= c_bid * (1 - c_taker_bps * 1e-4)

        # Path A: Buy DEX → Sell CEX
        edge_a_unit = (cex_sell - dex_buy) - fees_unit - (carry_guard * cex_sell)
        edge_a_bps  = 1e4 * (edge_a_unit / cex_sell) if cex_sell>0 else -1e9

        # Path B: Buy CEX → Sell DEX
        edge_b_unit = (dex_sell - cex_buy) - fees_unit - (carry_guard * dex_sell)
        edge_b_bps  = 1e4 * (edge_b_unit / dex_sell) if dex_sell>0 else -1e9

        if edge_a_unit >= edge_b_unit and edge_a_unit > 0:
            return ("DEX_TO_CEX", edge_a_unit, edge_a_bps, dex_buy, cex_sell, fees_unit, est_hours)
        if edge_b_unit > 0:
            return ("CEX_TO_DEX", edge_b_unit, edge_b_bps, cex_buy, dex_sell, fees_unit, est_hours)
        return None

    def _maybe_enter(self) -> None:
        sig = self._edge_and_paths()
        if not sig: return
        direction, edge_usd_unit, edge_bps, cheap_px, rich_px, fees_unit, est_hours = sig

        # size
        qty = max(0.0, USD_NOTIONAL / max(cheap_px, 1e-9))
        if qty * cheap_px < MIN_TICKET_USD: return
        # inventory cap (simple check on DEX balance for buy-on-DEX direction etc. if you want)
        if qty > POS_CAP_UNITS: qty = POS_CAP_UNITS

        total_edge_usd = edge_usd_unit * qty
        if not (edge_bps >= ENTRY_BPS_MIN and total_edge_usd >= ENTRY_USD_MIN):
            return

        # Hedge notional (perp) sized to rich leg price
        perp_qty = HEDGE_RATIO * qty

        # Sanity re-check right before commit
        sig2 = self._edge_and_paths()
        if not sig2 or sig2[0] != direction or sig2[2] < EXIT_BPS_BACKOUT:
            return

        txid = f"osa-{uuid.uuid4().hex[:12]}"
        if direction == "DEX_TO_CEX":
            # Enter: BUY on DEX, SHORT perp; then start transfer to CEX
            self.order(f"DEX:{CHAIN}:{SYM}", "buy",  qty=qty, price=cheap_px, order_type="market", venue=VENUE_DEX, flags={"slippage_bps": MEV_SLIP_BPS_GUARD}) # type: ignore
            self.order(f"PERP:{PERP_EXCH}:{SYM}", "sell", qty=perp_qty, order_type="market", venue=VENUE_PERP)
            # initiate transfer off-chain (adapter will process and publish xfer status)
            self.order(f"DEX:{CHAIN}:{SYM}", "withdraw_to_cex", qty=qty, order_type="transfer", venue=VENUE_DEX, flags={"txid": txid, "dest": CEX}) # type: ignore
        else:
            # Enter: BUY on CEX, SHORT perp; then withdraw to DEX
            self.order(f"CEX:{CEX}:{SYM}", "buy",  qty=qty, price=cheap_px, order_type="market", venue=VENUE_CEX) # type: ignore
            self.order(f"PERP:{PERP_EXCH}:{SYM}", "sell", qty=perp_qty, order_type="market", venue=VENUE_PERP)
            self.order(f"CEX:{CEX}:{SYM}", "withdraw_to_chain", qty=qty, order_type="transfer", venue=VENUE_CEX, flags={"txid": txid, "dest_chain": CHAIN}) # type: ignore

        st = OpenState(state="ENTERED", direction=direction, qty=qty,
                       avg_px_cheap=cheap_px, avg_px_rich=rich_px,
                       perp_qty=perp_qty, txid=txid, start_ms=_now_ms())
        self._save_state(st)
        # Monitor signal scaled by bps
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS_MIN))))

    # ------------ transfer & settlement ------------
    def _poll_transfer_and_maybe_settle(self, st: OpenState) -> None:
        # time guard
        if (_now_ms() - st.start_ms) > MAX_SETTLE_MINS * 60_000:
            self._abort_and_flatten(st, reason="timeout")
            return

        stat = _hget_json(f"{XFER_STATUS_PFX}{st.txid}", "status")  # convention: field 'status' on same key
        # Some adapters will store entire JSON in the key (not hash), handle that:
        if stat is None:
            raw = r.get(f"{XFER_STATUS_PFX}{st.txid}")
            if raw:
                try:
                    o = json.loads(raw); stat = o.get("status") # type: ignore
                except Exception: stat = None

        if (stat or "").lower() == "failed": # type: ignore
            self._abort_and_flatten(st, reason="xfer_failed")
            return

        if (stat or "").lower() != "confirmed": # type: ignore
            # still in flight
            time.sleep(CONFIRM_WAIT_SECS)
            # move to INFLIGHT so we don’t re-enter anywhere
            if st.state != "INFLIGHT":
                st.state = "INFLIGHT"; self._save_state(st)
            return

        # Received funds → sell rich spot and move to SETTLE
        if st.direction == "DEX_TO_CEX":
            self.order(f"CEX:{CEX}:{SYM}", "sell", qty=st.qty, price=st.avg_px_rich, order_type="market", venue=VENUE_CEX) # type: ignore
        else:
            self.order(f"DEX:{CHAIN}:{SYM}", "sell", qty=st.qty, price=st.avg_px_rich, order_type="market", venue=VENUE_DEX, flags={"slippage_bps": MEV_SLIP_BPS_GUARD}) # type: ignore

        st.state = "SETTLE"
        self._save_state(st)

    def _finalize(self, st: OpenState) -> None:
        # Close hedge
        self.order(f"PERP:{PERP_EXCH}:{SYM}", "buy", qty=st.perp_qty, order_type="market", venue=VENUE_PERP)
        # Done
        r.delete(_poskey(self.ctx.name))

    # ------------ error handling ------------
    def _abort_and_flatten(self, st: OpenState, reason: str) -> None:
        # Undo spot leg on the venue we entered; close hedge
        if st.direction == "DEX_TO_CEX":
            # we hold tokens on-chain (or transfer failed) → sell back on DEX
            self.order(f"DEX:{CHAIN}:{SYM}", "sell", qty=st.qty, order_type="market", venue=VENUE_DEX, flags={"slippage_bps": MEV_SLIP_BPS_GUARD}) # type: ignore
        else:
            self.order(f"CEX:{CEX}:{SYM}", "sell", qty=st.qty, order_type="market", venue=VENUE_CEX)
        self.order(f"PERP:{PERP_EXCH}:{SYM}", "buy", qty=st.perp_qty, order_type="market", venue=VENUE_PERP)
        r.delete(_poskey(self.ctx.name))

    # ------------ state I/O ------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw: return None
        try: return OpenState(**json.loads(raw)) # type: ignore
        except Exception: return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name), json.dumps(st.__dict__))