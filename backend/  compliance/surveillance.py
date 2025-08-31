# backend/surveillance/surveillance.py
from __future__ import annotations

import os, json, time, asyncio, math, collections, statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Deque, DefaultDict

# -------- Optional Redis (graceful) ------------------------------------------
HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

def now_ms() -> int: return int(time.time() * 1000)

# -------- Env / Streams ------------------------------------------------------
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
S_ORD_IN         = os.getenv("ORDERS_INCOMING", "orders.incoming")
S_ORD_FILLED     = os.getenv("ORDERS_FILLED", "orders.filled")
S_ORD_REJ        = os.getenv("ORDERS_REJECTED", "orders.rejected")
S_BARS           = os.getenv("PRICES_STREAM", "prices.bars")
S_ORDERBOOK      = os.getenv("WS_ORDERBOOK", "ws.orderbook")    # optional L2/L1 updates
S_ALERTS         = os.getenv("SURV_ALERTS", "surv.alerts")
MAXLEN           = int(os.getenv("SURV_MAXLEN", "12000"))
ART_DIR          = os.getenv("SURV_ART_DIR", "artifacts/surveillance")

os.makedirs(ART_DIR, exist_ok=True)
ALERTS_PATH = os.path.join(ART_DIR, "alerts.jsonl")

# -------- Utilities ----------------------------------------------------------
def parse_json_or_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    raw = fields.get("json")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    # fallback: best-effort numeric coerce
    out: Dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, str):
            try:
                if v.strip().isdigit():
                    out[k] = int(v)
                else:
                    out[k] = float(v)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out

def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------- Config -------------------------------------------------------------
@dataclass
class RuleConfig:
    # Spoofing/Layering
    spoof_near_bps: float = 3.0         # how close to mid to be "near touch"
    spoof_min_notional: float = 50_000  # min order notional to track
    spoof_cancel_ms: int = 2000         # canceled within this window and no fill
    spoof_min_ratio: float = 0.9        # canceled qty / placed qty threshold

    # OTR (Order-to-Trade) / Quote stuffing
    otr_window_ms: int = 60_000
    otr_ratio_warn: float = 20.0
    otr_ratio_crit: float = 100.0
    msg_rate_sigma: float = 5.0         # spike threshold over rolling mean

    # Momentum ignition
    ignite_window_ms: int = 10_000
    ignite_min_msgs: int = 30           # number of msgs by account/symbol in window
    ignite_px_jump_bps: float = 15.0    # subsequent move threshold

    # Marking the close
    mkt_close_ms: int = 15 * 60_000     # last N ms of session
    mkt_close_jump_bps: float = 15.0
    mkt_close_min_notional: float = 25_000

    # Self-trade / Wash-like
    self_trade_window_ms: int = 1500    # same account both sides within this
    wash_cross_min_qty: float = 1.0

    # General
    keep_minutes: int = 30              # state retention (rolling)

# -------- Alert shape --------------------------------------------------------
@dataclass
class Alert:
    ts_ms: int
    rule: str
    severity: str       # 'info' | 'warn' | 'crit'
    account: Optional[str]
    symbol: Optional[str]
    details: Dict[str, Any]

# -------- Surveillance core --------------------------------------------------
class Surveillance:
    def __init__(self, cfg: Optional[RuleConfig] = None):
        self.cfg = cfg or RuleConfig()
        self.r: Optional[AsyncRedis] = None # type: ignore
        self.last_ids: Dict[str, str] = {S_ORD_IN: "$", S_ORD_FILLED: "$", S_ORD_REJ: "$", S_BARS: "$", S_ORDERBOOK: "$"}
        # state
        self.mid: Dict[str, float] = {}                           # symbol -> mid/close
        self.l1: Dict[str, Tuple[float,float]] = {}               # symbol -> (bid, ask)
        self.placed: Dict[str, Dict[str, Any]] = {}               # order_id -> order payload
        self.canceled: DefaultDict[str, int] = collections.defaultdict(int)
        self.account_msgs: DefaultDict[Tuple[str,str], Deque[int]] = collections.defaultdict(collections.deque)  # (acct, sym) -> ts list
        self.account_orders: DefaultDict[str, Deque[int]] = collections.defaultdict(collections.deque)           # acct -> ts list
        self.account_trades: DefaultDict[str, Deque[int]] = collections.defaultdict(collections.deque)           # acct -> ts list
        self.sided_msgs: DefaultDict[Tuple[str,str,str], Deque[int]] = collections.defaultdict(collections.deque)# (acct,sym,side) -> ts
        self.last_trade_side: DefaultDict[Tuple[str,str], Tuple[int,str,float,float]] = collections.defaultdict(tuple) # (acct,sym)->(ts,side,qty,notional)

        # message rate stats
        self.msg_rate_hist: DefaultDict[str, Deque[int]] = collections.defaultdict(collections.deque)  # sym -> ts list

    # ---- wiring -------------------------------------------------------------
    async def connect(self):
        if not HAVE_REDIS:
            return
        try:
            self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    async def publish_alert(self, a: Alert):
        obj = asdict(a)
        jsonl_append(ALERTS_PATH, obj)
        if not self.r:
            # keep quiet in non-Redis mode
            return
        try:
            await self.r.xadd(S_ALERTS, {"json": json.dumps(obj, ensure_ascii=False)}, maxlen=MAXLEN, approximate=True)  # type: ignore
        except Exception:
            pass

    # ---- processing helpers -------------------------------------------------
    def _prune(self, now_ts: int):
        horizon = now_ts - self.cfg.keep_minutes * 60_000
        def _drop_old(dq: Deque[int]):
            while dq and dq[0] < horizon:
                dq.popleft()
        for dq in self.account_orders.values(): _drop_old(dq)
        for dq in self.account_trades.values(): _drop_old(dq)
        for dq in self.msg_rate_hist.values(): _drop_old(dq)
        for dq in self.account_msgs.values(): _drop_old(dq)
        for dq in self.sided_msgs.values(): _drop_old(dq)

    # ---- price updates ------------------------------------------------------
    def on_bar(self, ev: Dict[str, Any]):
        sym = str(ev.get("symbol", "")).upper()
        if not sym: return
        close = float(ev.get("close") or ev.get("price") or 0.0)
        if close > 0:
            self.mid[sym] = close
        self.msg_rate_hist[sym].append(int(ev.get("ts_ms") or now_ms()))

    def on_l1(self, ev: Dict[str, Any]):
        sym = str(ev.get("symbol", "")).upper()
        if not sym: return
        bid = float(ev.get("bid") or 0.0)
        ask = float(ev.get("ask") or 0.0)
        if bid > 0 and ask > 0:
            self.l1[sym] = (bid, ask)
            self.mid[sym] = (bid + ask) / 2.0

    # ---- order lifecycle ----------------------------------------------------
    async def on_order_in(self, ev: Dict[str, Any]):
        oid = str(ev.get("id") or ev.get("order_id") or f"in-{now_ms()}")
        self.placed[oid] = ev
        acct = str(ev.get("account") or ev.get("strategy") or "unknown")
        sym  = str(ev.get("symbol") or "").upper()
        side = str(ev.get("side") or "").lower()
        ts   = int(ev.get("ts_ms") or now_ms())
        self.account_orders[acct].append(ts)
        self.account_msgs[(acct, sym)].append(ts)
        self.sided_msgs[(acct, sym, side or "?")].append(ts)

        # quote stuffing / message spike
        await self._rule_quote_stuffing(acct, sym, ts)

    async def on_order_rej(self, ev: Dict[str, Any]):
        oid = str(ev.get("order_id") or "")
        if oid:
            self.canceled[oid] += 1

    async def on_fill(self, ev: Dict[str, Any]):
        oid = str(ev.get("order_id") or "")
        if oid and oid in self.placed:
            self.placed.pop(oid, None)
        acct = str(ev.get("account") or ev.get("strategy") or "unknown")
        sym  = str(ev.get("symbol") or "").upper()
        side = str(ev.get("side") or "").lower()
        qty  = float(ev.get("qty") or ev.get("quantity") or 0.0)
        px   = float(ev.get("price") or 0.0)
        notional = abs(qty * px)
        ts   = int(ev.get("ts_ms") or now_ms())
        self.account_trades[acct].append(ts)
        self.last_trade_side[(acct, sym)] = (ts, side, qty, notional)

        # Marking the close
        await self._rule_marking_close(acct, sym, ts, px, notional)

        # Self-trade / wash-like (same account, opposite side within short window)
        await self._rule_self_trade(acct, sym, ts, side, qty)

        # OTR check (needs trade count vs orders)
        await self._rule_otr(acct, sym, ts)

    # ---- RULES --------------------------------------------------------------
    async def _rule_quote_stuffing(self, acct: str, sym: str, ts: int):
        # spike in (acct,sym) message rate and symbol-wide message rate
        win = self.cfg.otr_window_ms
        # per-symbol rate stats
        hist = self.msg_rate_hist[sym]
        # compute mean/std over last window
        cut = ts - win
        while hist and hist[0] < cut: hist.popleft()
        n = len(hist)
        if n < 10:
            return
        # approximate rate as msgs per second
        span_s = max(1.0, (hist[-1] - hist[0]) / 1000.0)
        rate = n / span_s
        # simple rolling stats using small history of rates
        # (keep last 60 seconds)
        # if rate extremely high -> warn
        threshold = 200.0  # msgs/sec heuristic; tune per venue
        if rate > threshold:
            await self.publish_alert(Alert(
                ts_ms=ts, rule="QUOTE_STUFFING", severity="warn",
                account=acct, symbol=sym,
                details={"rate_msgs_per_s": round(rate, 2), "window_ms": win}
            ))

    async def _rule_otr(self, acct: str, sym: str, ts: int):
        win = self.cfg.otr_window_ms
        cut = ts - win

        def _prune(dq: Deque[int]):
            while dq and dq[0] < cut: dq.popleft()

        o = self.account_orders[acct]; _prune(o)
        t = self.account_trades[acct]; _prune(t)
        orders = len(o); trades = max(1, len(t))
        ratio = orders / trades
        if ratio >= self.cfg.otr_ratio_crit:
            sev = "crit"
        elif ratio >= self.cfg.otr_ratio_warn:
            sev = "warn"
        else:
            return
        await self.publish_alert(Alert(
            ts_ms=ts, rule="OTR_EXCESSIVE", severity=sev,
            account=acct, symbol=sym,
            details={"orders": orders, "trades": len(t), "ratio": round(ratio,2), "window_ms": win}
        ))

    async def _rule_marking_close(self, acct: str, sym: str, ts: int, px: float, notional: float):
        # if within last N ms of session (assume bars provide session end via ts modulo 24h; we use env override)
        # heuristic: use local time-of-day in ms; real systems use calendar
        mkt_close_ms = self.cfg.mkt_close_ms
        local = time.localtime(ts/1000.0)
        ms_tod = (local.tm_hour*3600 + local.tm_min*60 + local.tm_sec)*1000
        if (24*3600*1000 - ms_tod) > mkt_close_ms:
            return
        if notional < self.cfg.mkt_close_min_notional:
            return
        mid = self.mid.get(sym)
        if not mid or mid <= 0:
            return
        move_bps = abs((px - mid)/mid)*1e4
        if move_bps >= self.cfg.mkt_close_jump_bps:
            await self.publish_alert(Alert(
                ts_ms=ts, rule="MARKING_CLOSE", severity="warn",
                account=acct, symbol=sym,
                details={"notional": round(notional,2), "px": px, "mid": mid, "move_bps": round(move_bps,2)}
            ))

    async def _rule_self_trade(self, acct: str, sym: str, ts: int, side: str, qty: float):
        last = self.last_trade_side.get((acct, sym))
        if not last:
            return
        ts0, side0, qty0, _ = last
        if ts - ts0 <= self.cfg.self_trade_window_ms and side0 != side and min(abs(qty0), abs(qty)) >= self.cfg.wash_cross_min_qty:
            await self.publish_alert(Alert(
                ts_ms=ts, rule="SELF_TRADE_LIKE", severity="info",
                account=acct, symbol=sym,
                details={"prev_side": side0, "cur_side": side, "Δms": ts - ts0, "min_qty": min(abs(qty0), abs(qty))}
            ))

    async def check_spoofing(self, now_ts: int):
        """
        Iterate open orders; if canceled fast without fills and near touch -> alert.
        Trigger via periodic call.
        """
        for oid, o in list(self.placed.items()):
            ts = int(o.get("ts_ms") or now_ms())
            if now_ts - ts > self.cfg.spoof_cancel_ms:
                # saw no fill; if later we observe a cancel/reject count -> consider spoof
                if self.canceled.get(oid, 0) <= 0:
                    continue
                sym = str(o.get("symbol") or "").upper()
                side = str(o.get("side") or "").lower()
                px   = float(o.get("limit_price") or o.get("price") or 0.0)
                qty  = float(o.get("qty") or o.get("quantity") or 0.0)
                mid  = float(self.mid.get(sym) or 0.0)
                notional = abs(qty * (px or mid))
                if notional < self.cfg.spoof_min_notional or mid <= 0 or px <= 0:
                    continue
                bps = abs((px - mid)/mid) * 1e4
                if bps <= self.cfg.spoof_near_bps:
                    await self.publish_alert(Alert(
                        ts_ms=now_ts, rule="SPOOFING_LIKE", severity="crit",
                        account=str(o.get("account") or o.get("strategy") or "unknown"),
                        symbol=sym,
                        details={"order_id": oid, "side": side, "px": px, "mid": mid, "near_bps": round(bps,2), "notional": round(notional,2), "age_ms": now_ts - ts}
                    ))
                # whether spoof or not, drop old tracked orders
                self.placed.pop(oid, None)

    async def check_momentum_ignition(self, acct: str, sym: str, now_ts: int):
        win = self.cfg.ignite_window_ms
        dq = self.account_msgs[(acct, sym)]
        cut = now_ts - win
        while dq and dq[0] < cut: dq.popleft()
        if len(dq) < self.cfg.ignite_min_msgs:
            return
        # price jump since window start
        mid0 = self.mid.get(sym)
        if not mid0: 
            return
        # approximate: compare current mid vs average of first/last bar midpoint in window (we only have last)
        mid1 = float(self.mid.get(sym)) # type: ignore
        move_bps = abs((mid1 - mid0)/mid0) * 1e4
        if move_bps >= self.cfg.ignite_px_jump_bps:
            await self.publish_alert(Alert(
                ts_ms=now_ts, rule="MOMENTUM_IGNITION_LIKE", severity="warn",
                account=acct, symbol=sym,
                details={"msgs_in_window": len(dq), "move_bps": round(move_bps,2), "window_ms": win}
            ))

    # ---- main loop ----------------------------------------------------------
    async def run(self):
        await self.connect()
        if not self.r:
            # No Redis: idle loop so you can still write alerts via API
            while True:
                await asyncio.sleep(1)

        # main pump
        while True:
            try:
                resp = await self.r.xread(self.last_ids, count=500, block=1000)  # type: ignore
            except Exception:
                resp = []
            if not resp:
                # periodic sweeps
                await self.check_spoofing(now_ms())
                continue

            for stream, entries in resp:
                self.last_ids[stream] = entries[-1][0]
                for _id, fields in entries:
                    ev = parse_json_or_fields(fields)
                    ts = int(ev.get("ts_ms") or now_ms())
                    if stream == S_BARS:
                        self.on_bar(ev)
                    elif stream == S_ORDERBOOK:
                        self.on_l1(ev)
                    elif stream == S_ORD_IN:
                        await self.on_order_in(ev)
                        # ignition check is per account/symbol
                        acct = str(ev.get("account") or ev.get("strategy") or "unknown")
                        sym  = str(ev.get("symbol") or "").upper()
                        await self.check_momentum_ignition(acct, sym, ts)
                    elif stream == S_ORD_REJ:
                        await self.on_order_rej(ev)
                    elif stream == S_ORD_FILLED:
                        await self.on_fill(ev)

            # housekeeping
            self._prune(now_ms())
            await self.check_spoofing(now_ms())

# -------- CLI ----------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("surveillance")
    ap.add_argument("--redis", type=str, default=REDIS_URL)
    ap.add_argument("--spoof-near-bps", type=float, default=RuleConfig().spoof_near_bps)
    ap.add_argument("--spoof-min-notional", type=float, default=RuleConfig().spoof_min_notional)
    ap.add_argument("--otr-warn", type=float, default=RuleConfig().otr_ratio_warn)
    ap.add_argument("--otr-crit", type=float, default=RuleConfig().otr_ratio_crit)
    ap.add_argument("--ignite-msgs", type=int, default=RuleConfig().ignite_min_msgs)
    ap.add_argument("--ignite-bps", type=float, default=RuleConfig().ignite_px_jump_bps)
    ap.add_argument("--close-bps", type=float, default=RuleConfig().mkt_close_jump_bps)
    ap.add_argument("--close-notional", type=float, default=RuleConfig().mkt_close_min_notional)
    args = ap.parse_args()

    cfg = RuleConfig(
        spoof_near_bps=args.spoof_near_bps,
        spoof_min_notional=args.spoof_min_notional,
        otr_ratio_warn=args.otr_warn,
        otr_ratio_crit=args.otr_crit,
        ignite_min_msgs=args.ignite_msgs,
        ignite_px_jump_bps=args.ignite_bps,
        mkt_close_jump_bps=args.close_bps,
        mkt_close_min_notional=args.close_notional,
    )

    async def _run():
        s = Surveillance(cfg)
        await s.run()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    _cli()