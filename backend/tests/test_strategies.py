# test_strategies.py
# Self-contained test suite + tiny reference "strategies" implementation.
#
# It mirrors the fields/ideas from your strategies.tsx component and tests:
#  - CRUD (upsert/delete) idempotency
#  - Search/filter/sort by name/desk/status/tags
#  - Aggregates: weight sum, gross/net (weight-weighted), live count
#  - Guardrails: maxGross, maxNetAbs, maxWeightSum helpers
#  - Rebalance: normalize weights, compute target $ allocs, lot/price rounding
#  - CSV import/export round-trip
#  - Sparkline helper: last delta sign roughly matches pnlD sign
#
# Run:
#   pytest -q test_strategies.py

from __future__ import annotations

import csv
import io
import math
import unittest
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

UTC = timezone.utc
StrategyStatus = Literal["live", "paper", "paused"]


# =========================
# Reference implementation
# =========================

@dataclass
class Strategy:
    id: str
    name: str
    desk: str = ""
    tags: List[str] = field(default_factory=list)
    status: StrategyStatus = "paper"
    capital: Optional[float] = None  # informational
    target_weight: float = 0.0       # 0..100
    gross: Optional[float] = None    # %
    net: Optional[float] = None      # %
    sharpe: Optional[float] = None
    vol: Optional[float] = None
    pnl_ytd: Optional[float] = None
    pnl_d: Optional[float] = None
    positions: Optional[int] = None
    last_updated: Optional[datetime] = field(default_factory=lambda: datetime.now(tz=UTC))
    spark: List[float] = field(default_factory=list)  # recent daily P&L %

def _now() -> datetime:
    return datetime.now(tz=UTC)

class Strategies:
    def __init__(self):
        self._by_id: Dict[str, Strategy] = {}

    # ---- CRUD ----
    def upsert(self, s: Strategy) -> Strategy:
        prev = self._by_id.get(s.id)
        if prev:
            merged = Strategy(
                id=s.id,
                name=s.name or prev.name,
                desk=s.desk or prev.desk,
                tags=(s.tags or prev.tags),
                status=s.status or prev.status,
                capital=s.capital if s.capital is not None else prev.capital,
                target_weight=float(s.target_weight if s.target_weight is not None else prev.target_weight),
                gross=s.gross if s.gross is not None else prev.gross,
                net=s.net if s.net is not None else prev.net,
                sharpe=s.sharpe if s.sharpe is not None else prev.sharpe,
                vol=s.vol if s.vol is not None else prev.vol,
                pnl_ytd=s.pnl_ytd if s.pnl_ytd is not None else prev.pnl_ytd,
                pnl_d=s.pnl_d if s.pnl_d is not None else prev.pnl_d,
                positions=s.positions if s.positions is not None else prev.positions,
                last_updated=_now(),
                spark=s.spark or prev.spark,
            )
            self._by_id[s.id] = merged
        else:
            s.last_updated = s.last_updated or _now()
            self._by_id[s.id] = s
        return self._by_id[s.id]

    def delete(self, sid: str) -> bool:
        return self._by_id.pop(sid, None) is not None

    def get(self, sid: str) -> Optional[Strategy]:
        return self._by_id.get(sid)

    def list(self) -> List[Strategy]:
        return list(self._by_id.values())

    # ---- Search / filter / sort ----
    def search(
        self,
        q: str = "",
        desk: Optional[str] = None,
        status: Optional[StrategyStatus] = None,
        tag: Optional[str] = None,
        sort_key: str = "sharpe",
        asc: bool = False,
    ) -> List[Strategy]:
        qq = (q or "").strip().lower()
        def match(s: Strategy) -> bool:
            if qq and not any(
                part for part in [
                    s.name.lower(),
                    s.desk.lower(),
                    " ".join(s.tags).lower(),
                ] if qq in part
            ):
                return False
            if desk and s.desk != desk:
                return False
            if status and s.status != status:
                return False
            if tag and tag not in (s.tags or []):
                return False
            return True

        out = [s for s in self._by_id.values() if match(s)]

        def keyfunc(s: Strategy):
            v = getattr(s, sort_key, None)
            if isinstance(v, str):
                return v
            return float(v if v is not None else -math.inf)

        out.sort(key=keyfunc, reverse=not asc)
        return out

    # ---- Aggregates / guardrails ----
    def aggregates(self) -> Dict[str, float]:
        ss = self.list()
        sumW = sum(float(s.target_weight or 0.0) for s in ss)
        gross = sum(((s.gross or 0.0) * (s.target_weight or 0.0) / 100.0) for s in ss)
        net = sum(((s.net or 0.0) * (s.target_weight or 0.0) / 100.0) for s in ss)
        live_count = sum(1 for s in ss if s.status == "live")
        return {"sum_weight": sumW, "gross": gross, "net": net, "live_count": live_count}

    @staticmethod
    def guardrails_ok(agg: Dict[str, float], *, max_gross: float, max_net_abs: float, max_weight_sum: float) -> bool:
        return (
            agg["sum_weight"] <= max_weight_sum + 1e-9 and
            agg["gross"] <= max_gross + 1e-9 and
            abs(agg["net"]) <= max_net_abs + 1e-9
        )

    # ---- Rebalance ----
    def normalize_weights(self) -> None:
        ss = self.list()
        tot = sum(max(0.0, s.target_weight) for s in ss)
        if tot <= 0:
            return
        for s in ss:
            s.target_weight = 100.0 * max(0.0, s.target_weight) / tot

    def rebalance_dollars(
        self,
        total_capital: float,
        prices: Dict[str, float],
        lot_sizes: Optional[Dict[str, int]] = None,
        min_trade_dollars: float = 0.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Return {id: {target_dollars, lots, notional}} respecting lot size & min trade.
        If price or lot missing -> assume 1.0 and 1.
        """
        self.normalize_weights()
        out: Dict[str, Dict[str, float]] = {}
        for s in self.list():
            w = max(0.0, float(s.target_weight))
            tgt = total_capital * (w / 100.0)
            px = float(prices.get(s.id, 1.0))
            lot = int((lot_sizes or {}).get(s.id, 1))
            # compute lots rounding to nearest lot; enforce min_trade
            lots = 0 if px <= 0 else int(round(tgt / (px * lot)))
            notional = lots * lot * px
            if notional < min_trade_dollars:
                lots = 0
                notional = 0.0
            out[s.id] = {"target_dollars": tgt, "lots": float(lots), "notional": notional}
        return out

    # ---- CSV IO ----
    def export_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id","name","desk","tags","status","capital","target_weight","gross","net","sharpe","vol","pnl_ytd","pnl_d","positions","last_updated"])
        for s in self.list():
            w.writerow([
                s.id, s.name, s.desk, "|".join(s.tags or []), s.status,
                s.capital if s.capital is not None else "",
                s.target_weight,
                s.gross if s.gross is not None else "",
                s.net if s.net is not None else "",
                s.sharpe if s.sharpe is not None else "",
                s.vol if s.vol is not None else "",
                s.pnl_ytd if s.pnl_ytd is not None else "",
                s.pnl_d if s.pnl_d is not None else "",
                s.positions if s.positions is not None else "",
                (s.last_updated.isoformat() if s.last_updated else ""),
            ])
        return buf.getvalue()

    def import_csv(self, text: str) -> None:
        rdr = csv.DictReader(io.StringIO(text))
        for row in rdr:
            s = Strategy(
                id=row["id"],
                name=row["name"],
                desk=row.get("desk",""),
                tags=[t for t in (row.get("tags","").split("|")) if t],
                status=(row.get("status") or "paper"),  # type: ignore[assignment]
                capital=float(row["capital"]) if row.get("capital") else None,
                target_weight=float(row.get("target_weight") or 0.0),
                gross=float(row["gross"]) if row.get("gross") else None,
                net=float(row["net"]) if row.get("net") else None,
                sharpe=float(row["sharpe"]) if row.get("sharpe") else None,
                vol=float(row["vol"]) if row.get("vol") else None,
                pnl_ytd=float(row["pnl_ytd"]) if row.get("pnl_ytd") else None,
                pnl_d=float(row["pnl_d"]) if row.get("pnl_d") else None,
                positions=int(row["positions"]) if row.get("positions") else None,
                last_updated=datetime.fromisoformat(row["last_updated"]) if row.get("last_updated") else _now(),
            )
            self.upsert(s)

    # ---- Spark helpers ----
    @staticmethod
    def spark_delta(data: Sequence[float]) -> float:
        if not data:
            return 0.0
        return float(data[-1] - data[0])


# =========================
# Tests
# =========================

class TestStrategiesBasics(unittest.TestCase):
    def setUp(self):
        self.s = Strategies()
        self.s1 = Strategy(id="s1", name="US Tech Mean Reversion", desk="Equities", tags=["mean reversion","intraday"], status="live", target_weight=20, gross=250, net=10, sharpe=1.4, vol=12, pnl_ytd=420000, pnl_d=18000, positions=63, spark=[-0.1,0.05,0.08,0.02,0.1])
        self.s2 = Strategy(id="s2", name="FX Carry G10", desk="Macro", tags=["carry","swing"], status="paper", target_weight=10, gross=120, net=35, sharpe=0.9, vol=9, pnl_ytd=120000, pnl_d=-6000, positions=12, spark=[0.0,0.01,-0.02,0.0,-0.01])
        self.s3 = Strategy(id="s3", name="CTA Trend", desk="Macro", tags=["trend","futures"], status="live", target_weight=30, gross=180, net=-5, sharpe=1.2, vol=14, pnl_ytd=680000, pnl_d=4200, positions=45, spark=[-0.02,0.03,0.05,0.04,0.02])
        for x in [self.s1, self.s2, self.s3]:
            self.s.upsert(x)

    def test_crud_and_idempotency(self):
        self.assertEqual(len(self.s.list()), 3)
        got = self.s.get("s1")
        self.assertEqual(got.name, "US Tech Mean Reversion")
        # update keeps id, changes weight and status
        upd = Strategy(id="s1", name="US Tech MR", target_weight=22, status="paused")
        out = self.s.upsert(upd)
        self.assertEqual(out.name, "US Tech MR")
        self.assertEqual(out.target_weight, 22)
        self.assertEqual(out.status, "paused")
        # delete
        self.assertTrue(self.s.delete("s2"))
        self.assertFalse(self.s.delete("s2"))
        self.assertEqual(len(self.s.list()), 2)

    def test_search_filters_and_sort(self):
        # restore s2
        self.s.upsert(self.s2)
        by_desk = self.s.search(desk="Macro")
        self.assertEqual(set(x.id for x in by_desk), {"s2","s3"})
        carry = self.s.search(q="carry")
        self.assertEqual(len(carry), 1)
        self.assertEqual(carry[0].id, "s2")
        live = self.s.search(status="live")
        self.assertTrue(all(x.status=="live" for x in live))
        tagpick = self.s.search(tag="trend")
        self.assertEqual([x.id for x in tagpick], ["s3"])
        # sort by sharpe desc (default), s1(1.4) should appear before s3(1.2)
        out = self.s.search()
        self.assertEqual(out[0].id, "s1")

    def test_aggregates_and_guardrails(self):
        agg = self.s.aggregates()
        self.assertAlmostEqual(agg["sum_weight"], self.s1.target_weight + self.s2.target_weight + self.s3.target_weight, places=6)
        self.assertGreaterEqual(agg["gross"], 0.0)
        self.assertEqual(agg["live_count"], 2)
        ok = Strategies.guardrails_ok(agg, max_gross=300, max_net_abs=50, max_weight_sum=100)
        self.assertTrue(ok)
        # Break a guardrail
        self.s.upsert(Strategy(id="s4", name="Leveraged Beta", target_weight=200, gross=400, net=0, status="live"))
        agg2 = self.s.aggregates()
        self.assertFalse(Strategies.guardrails_ok(agg2, max_gross=300, max_net_abs=50, max_weight_sum=100))

    def test_normalize_weights(self):
        # messy weights
        self.s.upsert(Strategy(id="w1", name="X", target_weight=0))
        self.s.upsert(Strategy(id="w2", name="Y", target_weight=30))
        self.s.upsert(Strategy(id="w3", name="Z", target_weight=70))
        self.s.normalize_weights()
        total = sum(x.target_weight for x in self.s.list())
        self.assertAlmostEqual(total, 100.0, places=6)

    def test_rebalance_allocation_with_lots(self):
        prices = {"s1": 50.0, "s2": 100.0, "s3": 25.0}
        lots = {"s1": 10, "s2": 5, "s3": 20}
        self.s.normalize_weights()
        alloc = self.s.rebalance_dollars(1_000_000, prices, lot_sizes=lots, min_trade_dollars=1000.0)
        # sanity: keys exist
        self.assertTrue(all(k in alloc for k in ["s1","s2","s3"]))
        # notional should be close to target dollars, within 1 lot difference
        for sid, row in alloc.items():
            tgt = row["target_dollars"]
            notional = row["notional"]
            px = prices[sid]
            lot = lots[sid]
            self.assertLessEqual(abs(notional - tgt), px * lot * 1.0 + 1e-6)

    def test_csv_roundtrip(self):
        txt = self.s.export_csv()
        s2 = Strategies()
        s2.import_csv(txt)
        self.assertEqual(set(x.id for x in s2.list()), set(x.id for x in self.s.list()))
        # modify and ensure upsert updates vs duplicates
        s2.upsert(Strategy(id="s3", name="CTA Trend +", target_weight=25))
        self.assertEqual(len(s2.list()), len(self.s.list()))
        self.assertEqual(s2.get("s3").name, "CTA Trend +")

    def test_sparkline_delta_direction_matches_pnld(self):
        # sign of last - first aligns with pnl_d sign most of the time
        d1 = Strategies.spark_delta(self.s1.spark)
        d2 = Strategies.spark_delta(self.s2.spark)
        self.assertGreaterEqual(d1, 0.0)
        self.assertLessEqual(d2, 0.0)

class TestEdgeCases(unittest.TestCase):
    def test_empty_book_aggregates(self):
        book = Strategies()
        agg = book.aggregates()
        self.assertEqual(agg["sum_weight"], 0.0)
        self.assertEqual(agg["gross"], 0.0)
        self.assertEqual(agg["net"], 0.0)
        self.assertEqual(agg["live_count"], 0)

    def test_missing_prices_and_lots_defaults(self):
        book = Strategies()
        book.upsert(Strategy(id="a", name="A", target_weight=60))
        book.upsert(Strategy(id="b", name="B", target_weight=40))
        alloc = book.rebalance_dollars(100_000, prices={})  # defaults px=1, lot=1
        self.assertAlmostEqual(alloc["a"]["target_dollars"], 60_000, places=6)
        self.assertAlmostEqual(alloc["b"]["target_dollars"], 40_000, places=6)
        self.assertGreater(alloc["a"]["lots"], 0)
        self.assertGreater(alloc["b"]["lots"], 0)

    def test_search_no_matches(self):
        book = Strategies()
        book.upsert(Strategy(id="x", name="Alpha", desk="EQ"))
        out = book.search(q="nonexistent")
        self.assertEqual(out, [])

# PyTest bridge
def test_pytest_bridge():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestStrategiesBasics)
    suite2 = unittest.defaultTestLoader.loadTestsFromTestCase(TestEdgeCases)
    allsuite = unittest.TestSuite([suite, suite2])
    res = unittest.TextTestRunner(verbosity=0).run(allsuite)
    assert res.wasSuccessful(), "strategies tests failed"