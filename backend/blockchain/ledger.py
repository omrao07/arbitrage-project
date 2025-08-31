# backend/accounting/ledger.py
from __future__ import annotations

import os, json, time, csv, asyncio
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

# ---------- Optional Redis ---------------------------------------------------
HAVE_REDIS = True
try:
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

# ---------- Env / Streams ----------------------------------------------------
REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379/0")
S_LEDGER     = os.getenv("LEDGER_STREAM", "accounting.ledger")
MAXLEN       = int(os.getenv("LEDGER_MAXLEN", "20000"))

def now_ms() -> int: return int(time.time() * 1000)

# ---------- Entries ----------------------------------------------------------
@dataclass
class LedgerEntry:
    ts_ms: int
    account: str
    typ: str       # 'trade' | 'fee' | 'cash' | 'dividend' | 'adjustment'
    symbol: Optional[str] = None
    qty: float = 0.0
    price: float = 0.0
    value: float = 0.0
    description: Optional[str] = None
    extra: Dict[str,Any] = None # type: ignore

    def to_dict(self) -> Dict[str,Any]:
        d = asdict(self)
        d["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.ts_ms/1000))
        return d

# ---------- Ledger -----------------------------------------------------------
class Ledger:
    """
    Persistent, double-entry ledger. Can publish to Redis + write JSONL/CSV.
    """
    def __init__(self, account: str = "default", out_dir: str = "artifacts/ledger", publish: bool = True):
        self.account = account
        self.out_dir = out_dir
        self.publish = publish
        os.makedirs(out_dir, exist_ok=True)
        self.path_jsonl = os.path.join(out_dir, f"{account}.jsonl")
        self.path_csv   = os.path.join(out_dir, f"{account}.csv")
        self.r: Optional[AsyncRedis] = None # type: ignore
        self._rows: List[LedgerEntry] = []

    async def connect(self):
        if not HAVE_REDIS or not self.publish: return
        try:
            self.r = AsyncRedis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
            await self.r.ping() # type: ignore
        except Exception:
            self.r = None

    async def record(self, entry: LedgerEntry):
        """
        Record an entry in memory, persist to file, and publish to Redis.
        """
        self._rows.append(entry)
        d = entry.to_dict()
        # append to jsonl
        with open(self.path_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        # append to csv
        write_header = not os.path.exists(self.path_csv)
        with open(self.path_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(d.keys()))
            if write_header:
                w.writeheader()
            w.writerow(d)

        if self.r:
            try:
                await self.r.xadd(S_LEDGER, {"json": json.dumps(d)}, maxlen=MAXLEN, approximate=True)  # type: ignore
            except Exception:
                pass

    # Convenience wrappers
    async def trade(self, symbol: str, side: str, qty: float, price: float, fee: float = 0.0, desc: str = ""):
        val = qty * price * (1 if side.lower()=="buy" else -1)
        await self.record(LedgerEntry(ts_ms=now_ms(), account=self.account, typ="trade", symbol=symbol,
                                      qty=qty if side=="buy" else -qty, price=price, value=val,
                                      description=desc or f"{side} {qty} {symbol} @ {price}",
                                      extra={"side": side, "fee": fee}))

        if fee:
            await self.fee(symbol=symbol, amount=fee, desc="commission")

    async def cash(self, amount: float, desc: str = ""):
        await self.record(LedgerEntry(ts_ms=now_ms(), account=self.account, typ="cash", value=amount, description=desc))

    async def fee(self, symbol: Optional[str], amount: float, desc: str = ""):
        await self.record(LedgerEntry(ts_ms=now_ms(), account=self.account, typ="fee", symbol=symbol,
                                      value=-abs(amount), description=desc))

    async def dividend(self, symbol: str, amount: float, desc: str = ""):
        await self.record(LedgerEntry(ts_ms=now_ms(), account=self.account, typ="dividend", symbol=symbol,
                                      value=amount, description=desc))

    async def adjustment(self, amount: float, desc: str = ""):
        await self.record(LedgerEntry(ts_ms=now_ms(), account=self.account, typ="adjustment", value=amount, description=desc))

    def all_entries(self) -> List[LedgerEntry]:
        return list(self._rows)

# ---------- CLI --------------------------------------------------------------
def _cli():
    import argparse, asyncio
    ap = argparse.ArgumentParser("ledger")
    ap.add_argument("--account", type=str, default="default")
    ap.add_argument("--cash", type=float, default=None)
    ap.add_argument("--trade", type=str, help="symbol,side,qty,price")
    ap.add_argument("--fee", type=float, default=None)
    args = ap.parse_args()

    async def _run():
        l = Ledger(account=args.account)
        await l.connect()
        if args.cash:
            await l.cash(args.cash, "manual cash entry")
        if args.trade:
            sym, side, qty, px = args.trade.split(",")
            await l.trade(sym, side, float(qty), float(px), fee=(args.fee or 0.0))
    asyncio.run(_run())

if __name__ == "__main__":
    _cli()