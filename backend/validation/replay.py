"""
In-memory Redis replay harness for backtesting Redis-coupled strategies.

WHY THIS IS NECESSARY
---------------------
An audit of the 82 engine-contract strategies found they do not receive their
inputs — they *fetch* them, from a module-level Redis handle:

    218  r.get()      171  r.hget()      10  r.smembers()      9  r.hgetall()

The keys they read are dominated by a small set:

     51  last_price     32  risk:halt     11  fees:eq
      9  ref:sector      8  universe:eq   ... (179 distinct in total)

So a bare price series makes every one of them no-op: they look up `last_price`,
find nothing, and return without signalling. They are not "flat", they are
*unfed*. This harness feeds them, bar by bar, from a real price series, with no
live Redis anywhere.

WHAT IT REFUSES TO DO
---------------------
It seeds only structural scaffolding (halt flags, fee constants, a universe) —
never prices, fundamentals, or alt-data. Anything that carries alpha must come
from the caller's real series. Inventing a sector score or an ESG point here
would fabricate the very signal the gauntlet is supposed to measure.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

log = logging.getLogger(__name__)


class FakeRedis:
    """
    Minimal in-memory stand-in for the read/write surface strategies actually use.

    Mirrors `decode_responses=True` clients: everything in and out is `str`.
    Unimplemented commands raise rather than silently returning None, so an
    unsupported access shows up as a failed strategy run instead of a quiet
    zero that would look like a real (flat) result.
    """

    def __init__(self) -> None:
        self.kv: Dict[str, str] = {}
        self.hashes: Dict[str, Dict[str, str]] = {}
        self.sets: Dict[str, Set[str]] = {}
        self.lists: Dict[str, List[str]] = {}
        self.access_log: Dict[str, int] = {}

    # ── strings ─────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[str]:
        self.access_log[key] = self.access_log.get(key, 0) + 1
        return self.kv.get(key)

    def set(self, key: str, value: Any) -> bool:
        self.kv[key] = str(value)
        return True

    def mget(self, keys: Iterable[str]) -> List[Optional[str]]:
        return [self.get(k) for k in keys]

    def exists(self, key: str) -> int:
        return int(key in self.kv or key in self.hashes or key in self.sets or key in self.lists)

    def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            for store in (self.kv, self.hashes, self.sets, self.lists):
                if k in store:
                    del store[k]
                    n += 1
        return n

    # ── hashes ──────────────────────────────────────────────────────────────

    def hget(self, name: str, key: str) -> Optional[str]:
        self.access_log[f"{name}[{key}]"] = self.access_log.get(f"{name}[{key}]", 0) + 1
        return self.hashes.get(name, {}).get(key)

    def hset(self, name: str, key: Any = None, value: Any = None,
             mapping: Optional[Dict[str, Any]] = None) -> int:
        h = self.hashes.setdefault(name, {})
        if mapping:
            h.update({str(k): str(v) for k, v in mapping.items()})
            return len(mapping)
        if key is not None:
            h[str(key)] = str(value)
            return 1
        return 0

    def hgetall(self, name: str) -> Dict[str, str]:
        self.access_log[name] = self.access_log.get(name, 0) + 1
        return dict(self.hashes.get(name, {}))

    def hmget(self, name: str, keys: Iterable[str]) -> List[Optional[str]]:
        return [self.hget(name, k) for k in keys]

    def hdel(self, name: str, *keys: str) -> int:
        h = self.hashes.get(name, {})
        return sum(1 for k in keys if h.pop(k, None) is not None)

    # ── sets ────────────────────────────────────────────────────────────────

    def smembers(self, name: str) -> Set[str]:
        self.access_log[name] = self.access_log.get(name, 0) + 1
        return set(self.sets.get(name, set()))

    def sadd(self, name: str, *values: Any) -> int:
        s = self.sets.setdefault(name, set())
        before = len(s)
        s.update(str(v) for v in values)
        return len(s) - before

    def sismember(self, name: str, value: Any) -> bool:
        return str(value) in self.sets.get(name, set())

    # ── lists ───────────────────────────────────────────────────────────────

    def lrange(self, name: str, start: int, end: int) -> List[str]:
        self.access_log[name] = self.access_log.get(name, 0) + 1
        lst = self.lists.get(name, [])
        return lst[start:] if end == -1 else lst[start : end + 1]

    def rpush(self, name: str, *values: Any) -> int:
        lst = self.lists.setdefault(name, [])
        lst.extend(str(v) for v in values)
        return len(lst)

    def lpush(self, name: str, *values: Any) -> int:
        lst = self.lists.setdefault(name, [])
        for v in values:
            lst.insert(0, str(v))
        return len(lst)

    def ltrim(self, name: str, start: int, end: int) -> bool:
        lst = self.lists.get(name, [])
        self.lists[name] = lst[start:] if end == -1 else lst[start : end + 1]
        return True

    def llen(self, name: str) -> int:
        return len(self.lists.get(name, []))

    # ── misc ────────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        return True

    def keys(self, pattern: str = "*") -> List[str]:
        everything = list(self.kv) + list(self.hashes) + list(self.sets) + list(self.lists)
        return [k for k in everything if fnmatch.fnmatch(k, pattern)]

    def pipeline(self, *_a: Any, **_k: Any) -> "FakeRedis":
        return self

    def execute(self) -> List[Any]:
        return []

    def expire(self, *_a: Any, **_k: Any) -> bool:
        return True

    def publish(self, *_a: Any, **_k: Any) -> int:
        return 0

    def xadd(self, *_a: Any, **_k: Any) -> str:
        return "0-0"


@dataclass
class ReplayHarness:
    """
    Seeds a FakeRedis with structural scaffolding and advances `last_price`
    bar by bar.

    `symbol` is the single instrument the price series represents. Multi-asset
    strategies will still find only one name in the universe — they should fail
    the run rather than be handed invented cross-sectional data.
    """

    symbol: str = "SYNTH"
    universe: List[str] = field(default_factory=list)
    redis: FakeRedis = field(default_factory=FakeRedis)

    def __post_init__(self) -> None:
        if not self.universe:
            self.universe = [self.symbol]
        self._seed_scaffolding()

    def _seed_scaffolding(self) -> None:
        """
        Structural constants only — nothing that could carry alpha.

        Fee and rate constants are the same values a live deployment would
        publish; they are cost inputs, not predictive ones.
        """
        r = self.redis
        r.set("risk:halt", "0")
        r.set("engine:halt", "0")
        for key in ("universe:eq", "universe", "universe:all"):
            r.sadd(key, *self.universe)
        for sym in self.universe:
            r.hset("ref:sector", sym, "UNKNOWN")
            r.hset("borrow:ok", sym, "1")
            r.hset("borrow:fee", sym, "0.005")
        r.set("fees:eq", "0.0003")
        r.set("fees:etf", "0.0003")
        r.set("fees:trading", "0.0003")
        r.set("rate:risk_free:USD", "0.05")
        r.set("rate:risk_free:INR", "0.065")
        r.set("portfolio:nlv", "1000000")
        r.set("funding:cash", "1000000")
        r.hset("fx:spot", "USDINR", "83.0")

    def advance(self, price: float, bar_index: int) -> None:
        """Publish one bar's price into every key strategies read it from."""
        r = self.redis
        p = str(float(price))
        r.hset("last_price", self.symbol, p)
        r.hset("price:last", self.symbol, p)
        r.set(f"last_price:{self.symbol}", p)
        r.hset("mark", self.symbol, p)
        r.hset(
            "orderbook:best",
            self.symbol,
            json.dumps({"bid": float(price) * 0.9995, "ask": float(price) * 1.0005}),
        )
        r.rpush(f"prices:{self.symbol}", p)
        r.ltrim(f"prices:{self.symbol}", -500, -1)
        r.set("bar:index", str(bar_index))

    def unread_keys(self) -> List[str]:
        """Seeded keys nothing ever looked at — useful for pruning the harness."""
        seeded = set(self.redis.keys("*"))
        return sorted(seeded - set(self.redis.access_log))

    def missing_keys(self) -> List[str]:
        """
        Keys a strategy asked for that were never seeded.

        This is the actionable output: it names exactly which real data feed
        must exist before a given strategy can be honestly evaluated.
        """
        return sorted(k for k, _ in self.redis.access_log.items() if self.redis.get(k) is None
                      and k not in self.redis.hashes
                      and k not in self.redis.sets
                      and k not in self.redis.lists
                      and "[" not in k)


def install(module: Any, harness: ReplayHarness) -> bool:
    """
    Swap a strategy module's Redis handle for the harness.

    Strategies bind `r = redis.Redis(...)` at module scope, so replacing that
    attribute after import is sufficient and avoids patching library internals.
    Returns False when the module has no such handle to replace.
    """
    replaced = False
    for attr in ("r", "R", "_r", "redis_client", "rds"):
        if hasattr(module, attr):
            setattr(module, attr, harness.redis)
            replaced = True
    return replaced
