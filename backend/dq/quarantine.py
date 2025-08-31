# backend/risk/quarantine.py
from __future__ import annotations

import os, json, time, asyncio, signal, uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Tuple

# ---------- Optional Redis (graceful) ----------------------------------------
HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

# ---------- Streams / Paths --------------------------------------------------
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")
S_SURV_ALERTS   = os.getenv("SURV_ALERTS", "surv.alerts")
S_RISK_EVENTS   = os.getenv("RISK_EVENTS", "risk.events")           # optional: your risk manager can publish here
S_HEALTH        = os.getenv("HEALTH_STREAM", "ops.health")          # optional health pings/errors
S_Q_EVENTS      = os.getenv("QUAR_STREAM", "risk.quarantine")
MAXLEN          = int(os.getenv("QUAR_MAXLEN", "12000"))

STATE_DIR       = os.getenv("QUAR_STATE_DIR", "artifacts/quarantine")
STATE_PATH      = os.path.join(STATE_DIR, "quarantine_state.json")

os.makedirs(STATE_DIR, exist_ok=True)

def now_ms() -> int: return int(time.time() * 1000)
def ms(n: float) -> int: return int(n)

# ---------- Models -----------------------------------------------------------
@dataclass
class QuarItem:
    key: str                  # "symbol:RELIANCE" | "strategy:mm_us" | "account:ABCD..." | "venue:XNAS"
    reason: str               # free text / source rule id
    created_ms: int
    ttl_ms: Optional[int]     # None = until cleared
    meta: Dict[str, Any]

    def expires_at(self) -> Optional[int]:
        return (self.created_ms + self.ttl_ms) if self.ttl_ms else None

    def live(self, t: Optional[int] = None) -> bool:
        t = t or now_ms()
        ex = self.expires_at()
        return True if ex is None else (t < ex)

# ---------- Storage & API ----------------------------------------------------
class QuarantineDB:
    """
    Simple in-memory + JSON file + optional Redis mirror for quarantine decisions.
    Use it inside risk manager / OMS to block orders:

        qdb = QuarantineDB(redis_mirror=True)
        if qdb.is_blocked(symbol="AAPL", strategy="meanrev", account="ACC1", venue="XNAS"):
            # reject order upstream

    Keys are normalized as: 'symbol:<SYM>', 'strategy:<NAME>', 'account:<ID>', 'venue:<MIC>'.
    """
    def __init__(self, persist_path: str = STATE_PATH, redis_mirror: bool = True):
        self.persist_path = persist_path
        self.items: Dict[str, QuarItem] = {}
        self.r: Optional[AsyncRedis] = None # type: ignore
        self.redis_mirror = redis_mirror

    async def connect(self):
        if not (self.redis_mirror and HAVE_REDIS):
            return
        try:
            self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    # ---------- persistence -------------
    def load(self):
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            now = now_ms()
            for k, v in data.items():
                qi = QuarItem(**v)
                if qi.live(now):
                    self.items[k] = qi
        except Exception:
            self.items = {}

    def save(self):
        try:
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump({k: asdict(v) for k, v in self.items.items()}, f, indent=2)
        except Exception:
            pass

    # ---------- key helpers -------------
    @staticmethod
    def k_symbol(sym: str) -> str:   return f"symbol:{str(sym).upper()}"
    @staticmethod
    def k_strategy(s: str) -> str:   return f"strategy:{str(s)}"
    @staticmethod
    def k_account(a: str) -> str:    return f"account:{str(a)}"
    @staticmethod
    def k_venue(m: str) -> str:      return f"venue:{str(m).upper()}"

    def _publish(self, action: str, item: QuarItem):
        if not self.r:
            return
        try:
            obj = {"ts_ms": now_ms(), "action": action, **asdict(item)}
            asyncio.create_task(self.r.xadd(S_Q_EVENTS, {"json": json.dumps(obj, ensure_ascii=False)}, maxlen=MAXLEN, approximate=True))  # type: ignore
        except Exception:
            pass

    # ---------- CRUD --------------------
    def add(self, key: str, *, reason: str, ttl_ms: Optional[int], meta: Optional[Dict[str, Any]] = None) -> QuarItem:
        qi = QuarItem(key=key, reason=reason, created_ms=now_ms(), ttl_ms=ttl_ms, meta=meta or {})
        self.items[key] = qi
        self.save()
        self._publish("add", qi)
        return qi

    def remove(self, key: str) -> bool:
        ok = key in self.items
        if ok:
            qi = self.items.pop(key)
            self.save()
            self._publish("remove", qi)
        return ok

    def clear_expired(self) -> int:
        t = now_ms()
        gone = [k for k, v in self.items.items() if not v.live(t)]
        for k in gone:
            qi = self.items.pop(k)
            self._publish("expire", qi)
        if gone:
            self.save()
        return len(gone)

    def list(self) -> List[QuarItem]:
        self.clear_expired()
        return list(self.items.values())

    # ---------- checks ------------------
    def is_blocked(self, *, symbol: Optional[str] = None, strategy: Optional[str] = None, account: Optional[str] = None, venue: Optional[str] = None) -> Tuple[bool, Optional[QuarItem]]:
        """
        Evaluate most-specific to least-specific. First hit wins.
        """
        self.clear_expired()
        # exacts
        if symbol:
            qi = self.items.get(self.k_symbol(symbol))
            if qi and qi.live(): return True, qi
        if strategy:
            qi = self.items.get(self.k_strategy(strategy))
            if qi and qi.live(): return True, qi
        if account:
            qi = self.items.get(self.k_account(account))
            if qi and qi.live(): return True, qi
        if venue:
            qi = self.items.get(self.k_venue(venue))
            if qi and qi.live(): return True, qi
        return False, None

# ---------- Auto-quarantine daemon ------------------------------------------
@dataclass
class AutoRules:
    # map surveillance rules → default TTLs & severities that trigger quarantines
    spoofing_ttl_ms: int = 30 * 60_000
    ignition_ttl_ms: int = 10 * 60_000
    close_mark_ttl_ms: int = 30 * 60_000
    self_trade_ttl_ms: int = 5 * 60_000
    otr_crit_ttl_ms: int = 15 * 60_000

    # gates
    quarantine_on_spoofing: bool = True
    quarantine_on_ignition: bool = True
    quarantine_on_mark_close: bool = True
    quarantine_on_self_trade: bool = False     # often informational
    quarantine_on_otr_crit: bool = True

class QuarantineDaemon:
    """
    Listens to `surv.alerts` (and optionally `risk.events` / `ops.health`) and applies AutoRules.
    Publishes state changes to `risk.quarantine`.
    """
    def __init__(self, qdb: QuarantineDB, rules: Optional[AutoRules] = None):
        self.qdb = qdb
        self.rules = rules or AutoRules()
        self.r: Optional[AsyncRedis] = None # type: ignore
        self.last_ids: Dict[str, str] = {S_SURV_ALERTS: "$", S_RISK_EVENTS: "$", S_HEALTH: "$"}
        self._running = False

    async def connect(self):
        await self.qdb.connect()
        if not HAVE_REDIS: return
        try:
            self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    async def run(self):
        self.qdb.load()
        await self.connect()
        self._running = True

        # signal handling
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.stop(f"signal:{s.name}")))
        except Exception:
            pass

        if not self.r:
            # Idle mode without Redis, still usable via qdb API.
            while self._running:
                await asyncio.sleep(1.0)
            return

        while self._running:
            try:
                resp = await self.r.xread(self.last_ids, count=500, block=1000)  # type: ignore
            except Exception:
                resp = []
            if not resp:
                self.qdb.clear_expired()
                continue

            for stream, entries in resp:
                self.last_ids[stream] = entries[-1][0]
                for _id, fields in entries:
                    ev = _parse_json_or_fields(fields)
                    if stream == S_SURV_ALERTS:
                        await self._handle_surv_alert(ev)
                    elif stream == S_RISK_EVENTS:
                        await self._handle_risk_event(ev)
                    elif stream == S_HEALTH:
                        await self._handle_health(ev)

            self.qdb.clear_expired()

    async def stop(self, reason: str = "requested"):
        self._running = False

    # ---------- Handlers ----------------
    async def _handle_surv_alert(self, a: Dict[str, Any]):
        rule = str(a.get("rule") or "").upper()
        sev  = str(a.get("severity") or "info").lower()
        sym  = (a.get("symbol") or "").upper() or None
        acct = a.get("account") or None

        # Map rules → quarantine keys/TTLs
        if rule in ("SPOOFING_LIKE","SPOOFING","LAYERING"):
            if not self.rules.quarantine_on_spoofing: return
            if sev in ("warn","crit") and sym:
                self.qdb.add(self.qdb.k_symbol(sym), reason=f"{rule}:{sev}", ttl_ms=self.rules.spoofing_ttl_ms, meta=a)
        elif rule in ("MOMENTUM_IGNITION_LIKE","IGNITION"):
            if not self.rules.quarantine_on_ignition: return
            if sev in ("warn","crit") and sym:
                self.qdb.add(self.qdb.k_symbol(sym), reason=f"{rule}:{sev}", ttl_ms=self.rules.ignition_ttl_ms, meta=a)
        elif rule in ("MARKING_CLOSE","MARKING_THE_CLOSE"):
            if not self.rules.quarantine_on_mark_close: return
            if sym:
                self.qdb.add(self.qdb.k_symbol(sym), reason=f"{rule}:{sev}", ttl_ms=self.rules.close_mark_ttl_ms, meta=a)
        elif rule in ("SELF_TRADE_LIKE","WASH_LIKE"):
            if not self.rules.quarantine_on_self_trade: return
            if acct:
                self.qdb.add(self.qdb.k_account(acct), reason=f"{rule}:{sev}", ttl_ms=self.rules.self_trade_ttl_ms, meta=a)
        elif rule in ("OTR_EXCESSIVE",):
            if not self.rules.quarantine_on_otr_crit: return
            if str(a.get("details", {}).get("ratio", "")) not in ("", "nan"):
                try:
                    ratio = float(a["details"]["ratio"])
                except Exception:
                    ratio = 0.0
            else:
                ratio = 0.0
            if ratio >= 100.0:  # critical
                if acct:
                    self.qdb.add(self.qdb.k_account(acct), reason=f"{rule}:{sev}", ttl_ms=self.rules.otr_crit_ttl_ms, meta=a)

    async def _handle_risk_event(self, e: Dict[str, Any]):
        """
        Optional: respond to risk manager events, e.g. VAR_LIMIT_BREACH, INTRADAY_DD, CAPACITY_HALT.
        Expect shape: {"type": "...", "severity": "warn|crit", "symbol": "...", "strategy": "...", "account": "..."}
        """
        typ = str(e.get("type") or "").upper()
        sev = str(e.get("severity") or "info").lower()
        sym = (e.get("symbol") or "").upper() or None
        strat = e.get("strategy") or None
        acct = e.get("account") or None

        ttl = 30 * 60_000
        if typ in ("VAR_LIMIT_BREACH","INTRADAY_DRAWDOWN","CAPACITY_HALT"):
            if sym:
                self.qdb.add(self.qdb.k_symbol(sym), reason=f"{typ}:{sev}", ttl_ms=ttl, meta=e)
            if strat:
                self.qdb.add(self.qdb.k_strategy(strat), reason=f"{typ}:{sev}", ttl_ms=ttl, meta=e)
            if acct:
                self.qdb.add(self.qdb.k_account(acct), reason=f"{typ}:{sev}", ttl_ms=ttl, meta=e)

    async def _handle_health(self, h: Dict[str, Any]):
        """
        Optional: “poison pill” on broken data sources / venues.
        Example: {"component":"venue:XNAS", "status":"down"} -> quarantine venue
        """
        comp = str(h.get("component") or "")
        status = str(h.get("status") or "").lower()
        if comp.startswith("venue:") and status in ("down","degraded"):
            ven = comp.split(":",1)[1].upper()
            self.qdb.add(self.qdb.k_venue(ven), reason=f"HEALTH:{status}", ttl_ms=10*60_000, meta=h)

# ---------- Helpers ----------------------------------------------------------
def _parse_json_or_fields(fields: Dict[str,Any]) -> Dict[str,Any]:
    raw = fields.get("json")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {k: _coerce(v) for k, v in fields.items()}

def _coerce(v: Any) -> Any:
    try:
        if isinstance(v, str) and v.strip().isdigit():
            return int(v)
        return float(v)
    except Exception:
        return v

# ---------- CLI --------------------------------------------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser("quarantine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # daemon
    d = sub.add_parser("daemon", help="Run auto-quarantine daemon (listens to surv.alerts)")
    # manual add
    a = sub.add_parser("add", help="Manually quarantine")
    a.add_argument("--scope", required=True, choices=["symbol","strategy","account","venue"])
    a.add_argument("--value", required=True)
    a.add_argument("--reason", default="manual")
    a.add_argument("--ttl-sec", type=int, default=None)

    # remove
    r = sub.add_parser("remove", help="Remove a quarantine")
    r.add_argument("--scope", required=True, choices=["symbol","strategy","account","venue"])
    r.add_argument("--value", required=True)

    # list
    l = sub.add_parser("list", help="List current quarantines")

    args = ap.parse_args()

    async def _run():
        qdb = QuarantineDB()
        qdb.load()
        await qdb.connect()

        if args.cmd == "daemon":
            dmn = QuarantineDaemon(qdb)
            await dmn.run()
        elif args.cmd == "add":
            key = {
                "symbol": qdb.k_symbol(args.value),
                "strategy": qdb.k_strategy(args.value),
                "account": qdb.k_account(args.value),
                "venue": qdb.k_venue(args.value)
            }[args.scope]
            ttl_ms = args.ttl_sec * 1000 if args.ttl_sec else None
            qi = qdb.add(key, reason=args.reason, ttl_ms=ttl_ms, meta={"source":"cli"})
            print(json.dumps(asdict(qi), indent=2))
        elif args.cmd == "remove":
            key = {
                "symbol": qdb.k_symbol(args.value),
                "strategy": qdb.k_strategy(args.value),
                "account": qdb.k_account(args.value),
                "venue": qdb.k_venue(args.value)
            }[args.scope]
            ok = qdb.remove(key)
            print("removed" if ok else "not_found")
        elif args.cmd == "list":
            out = [asdict(x) for x in qdb.list()]
            print(json.dumps(out, indent=2))

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    _cli()