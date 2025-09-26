# test_risk.py
# A self-contained test suite + tiny reference implementation for a
# risk register / scoring library used by dashboards like risk.tsx / risk-matrix.tsx.
#
# It covers:
#  - Severity score (likelihood × impact), bucket & color mapping monotonicity
#  - Add / upsert / delete semantics and idempotency
#  - Search / filter by text, tag, owner, status
#  - Matrix bucketing into (likelihood, impact) grid
#  - Aggregations by tag/owner and portfolio-level KPIs
#  - SLA / review cadence derived from severity, overdue detection
#  - CSV import/export round-trip
#
# Run:
#   pytest -q test_risk.py

from __future__ import annotations

import csv
import io
import json
import math
import unittest
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

UTC = timezone.utc

RiskStatus = Literal["open", "mitigating", "closed"]


# =========================
# Reference implementation
# =========================

@dataclass
class Risk:
    id: str
    title: str
    owner: str = ""
    tag: str = ""
    likelihood: int = 1  # 1..L
    impact: int = 1      # 1..C
    status: RiskStatus = "open"
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    target_date: Optional[datetime] = None  # e.g., mitigation due date

    @property
    def score(self) -> int:
        return int(self.likelihood) * int(self.impact)


def severity_bucket(score: int, max_score: int) -> str:
    """
    Map score -> bucket label. Tuned for 1..25 default grid.
    """
    t = score / max(1, max_score)
    if t < 0.2:
        return "low"
    if t < 0.44:
        return "moderate"
    if t < 0.68:
        return "high"
    return "critical"


def severity_color_t(score: int, max_score: int) -> float:
    """
    Return a 0..1 severity "t" used for color ramps (green->yellow->red).
    Monotonic in score.
    """
    if max_score <= 1:
        return 1.0
    # Smooth S-curve helps visually separate mid-range risks.
    x = score / max_score
    # logistic centered at 0.5
    k = 8.0
    t = 1 / (1 + math.exp(-k * (x - 0.5)))
    return float(t)


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


class RiskRegister:
    def __init__(self, L: int = 5, C: int = 5):
        self.L = int(L)
        self.C = int(C)
        self._by_id: Dict[str, Risk] = {}

    # ---- CRUD ----
    def upsert(self, r: Risk) -> Risk:
        r.likelihood = clamp(r.likelihood, 1, self.L)
        r.impact = clamp(r.impact, 1, self.C)
        now = datetime.now(tz=UTC)
        if r.id in self._by_id:
            base = self._by_id[r.id]
            merged = Risk(
                id=r.id,
                title=r.title or base.title,
                owner=r.owner or base.owner,
                tag=r.tag or base.tag,
                likelihood=r.likelihood,
                impact=r.impact,
                status=r.status or base.status,
                notes=r.notes or base.notes,
                created_at=base.created_at,
                updated_at=now,
                target_date=r.target_date or base.target_date,
            )
            self._by_id[r.id] = merged
        else:
            r.created_at = now
            r.updated_at = now
            self._by_id[r.id] = r
        return self._by_id[r.id]

    def delete(self, rid: str) -> bool:
        return self._by_id.pop(rid, None) is not None

    def get(self, rid: str) -> Optional[Risk]:
        return self._by_id.get(rid)

    def list(self) -> List[Risk]:
        return list(self._by_id.values())

    # ---- Search / filter ----
    def search(
        self,
        q: str = "",
        tag: Optional[str] = None,
        owner: Optional[str] = None,
        status: Optional[RiskStatus] = None,
    ) -> List[Risk]:
        qq = (q or "").strip().lower()
        out = []
        for r in self._by_id.values():
            if qq and not any(
                s for s in [
                    r.title.lower(),
                    r.owner.lower(),
                    r.tag.lower(),
                    r.notes.lower(),
                ] if qq in s
            ):
                continue
            if tag and r.tag != tag:
                continue
            if owner and r.owner != owner:
                continue
            if status and r.status != status:
                continue
            out.append(r)
        return out

    # ---- Matrix bucketing ----
    def to_matrix(self) -> Dict[Tuple[int, int], List[Risk]]:
        grid: Dict[Tuple[int, int], List[Risk]] = {}
        for r in self._by_id.values():
            key = (r.likelihood, r.impact)
            grid.setdefault(key, []).append(r)
        return grid

    # ---- Aggregations ----
    def aggregate_by(self, key: Literal["tag", "owner"]) -> Dict[str, Dict[str, Any]]:
        res: Dict[str, Dict[str, Any]] = {}
        for r in self._by_id.values():
            k = getattr(r, key) or "—"
            b = res.setdefault(k, {"count": 0, "avg_score": 0.0, "open": 0, "critical": 0})
            b["count"] += 1
            b["avg_score"] += r.score
            if r.status != "closed":
                b["open"] += 1
            if severity_bucket(r.score, self.L * self.C) == "critical":
                b["critical"] += 1
        for k, b in res.items():
            if b["count"]:
                b["avg_score"] = b["avg_score"] / b["count"]
        return res

    def portfolio_kpis(self) -> Dict[str, Any]:
        risks = self.list()
        max_score = self.L * self.C
        total = len(risks)
        open_ = sum(1 for r in risks if r.status != "closed")
        critical = sum(1 for r in risks if severity_bucket(r.score, max_score) == "critical")
        avg_score = (sum(r.score for r in risks) / total) if total else 0.0
        return {"count": total, "open": open_, "critical": critical, "avg_score": avg_score}

    # ---- SLA / overdue ----
    @staticmethod
    def review_interval_days(score: int, max_score: int) -> int:
        """
        Example SLA:
          1..20% -> 90d, 20..44% -> 60d, 44..68% -> 30d, >68% -> 7d
        """
        t = score / max(1, max_score)
        if t <= 0.20:
            return 90
        if t <= 0.44:
            return 60
        if t <= 0.68:
            return 30
        return 7

    def overdue(self, as_of: datetime) -> List[Risk]:
        out = []
        max_s = self.L * self.C
        for r in self._by_id.values():
            if r.status == "closed":
                continue
            days = self.review_interval_days(r.score, max_s)
            next_due = (r.updated_at or r.created_at) + timedelta(days=days)
            if next_due < as_of:
                out.append(r)
        return out

    # ---- CSV IO ----
    def export_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([
            "id", "title", "owner", "tag", "likelihood", "impact", "status", "notes",
            "created_at", "updated_at", "target_date"
        ])
        for r in self.list():
            w.writerow([
                r.id, r.title, r.owner, r.tag, r.likelihood, r.impact, r.status, r.notes,
                r.created_at.isoformat(), r.updated_at.isoformat(),
                r.target_date.isoformat() if r.target_date else "",
            ])
        return buf.getvalue()

    def import_csv(self, s: str) -> None:
        rdr = csv.DictReader(io.StringIO(s))
        for row in rdr:
            r = Risk(
                id=row["id"] or row["title"],  # fallback
                title=row["title"],
                owner=row.get("owner", ""),
                tag=row.get("tag", ""),
                likelihood=int(row.get("likelihood") or 1),
                impact=int(row.get("impact") or 1),
                status=(row.get("status") or "open"),  # type: ignore[assignment]
                notes=row.get("notes", ""),
                created_at=datetime.fromisoformat(row["created_at"]).astimezone(UTC) if row.get("created_at") else datetime.now(tz=UTC),
                updated_at=datetime.fromisoformat(row["updated_at"]).astimezone(UTC) if row.get("updated_at") else datetime.now(tz=UTC),
                target_date=datetime.fromisoformat(row["target_date"]).astimezone(UTC) if row.get("target_date") else None,
            )
            self.upsert(r)


# =========================
# Tests
# =========================

class TestRiskBasics(unittest.TestCase):
    def setUp(self):
        self.reg = RiskRegister(L=5, C=5)
        now = datetime(2025, 1, 1, tzinfo=UTC)
        self.r1 = Risk(id="r1", title="Data breach", owner="Security", tag="Security", likelihood=2, impact=5, status="mitigating", created_at=now, updated_at=now)
        self.r2 = Risk(id="r2", title="Cloud outage", owner="Infra", tag="Ops", likelihood=3, impact=4, status="open", created_at=now, updated_at=now)
        self.r3 = Risk(id="r3", title="Regulatory delay", owner="Legal", tag="Reg", likelihood=2, impact=4, status="open", created_at=now, updated_at=now)
        for r in [self.r1, self.r2, self.r3]:
            self.reg.upsert(r)

    def test_score_and_bucket_monotonic(self):
        max_s = self.reg.L * self.reg.C
        scores = [1, 2, 4, 6, 9, 12, 16, 20, 25]
        ts = [severity_color_t(s, max_s) for s in scores]
        # strictly non-decreasing
        self.assertTrue(all(ts[i] <= ts[i+1] + 1e-12 for i in range(len(ts)-1)))
        # buckets ordered
        buckets = [severity_bucket(s, max_s) for s in scores]
        order = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
        self.assertTrue(all(order[buckets[i]] <= order[buckets[i+1]] for i in range(len(buckets)-1)))

    def test_upsert_and_delete(self):
        r = Risk(id="rX", title="Vendor breach", owner="Security", tag="ThirdParty", likelihood=5, impact=5)
        self.reg.upsert(r)
        self.assertIsNotNone(self.reg.get("rX"))
        # Update
        r2 = Risk(id="rX", title="Vendor breach - updated", owner="Security", tag="ThirdParty", likelihood=4, impact=5, status="mitigating")
        updated = self.reg.upsert(r2)
        self.assertEqual(updated.title, "Vendor breach - updated")
        self.assertEqual(updated.likelihood, 4)
        self.assertEqual(updated.status, "mitigating")
        # Delete
        self.assertTrue(self.reg.delete("rX"))
        self.assertFalse(self.reg.delete("rX"))  # already gone

    def test_search_and_filters(self):
        out = self.reg.search(q="cloud")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].id, "r2")
        self.assertEqual(len(self.reg.search(tag="Security")), 1)
        self.assertEqual(len(self.reg.search(owner="Legal")), 1)
        self.assertEqual(len(self.reg.search(status="open")), 2)

    def test_matrix_bucket_counts(self):
        mat = self.reg.to_matrix()
        self.assertIn((3, 4), mat)  # r2
        self.assertEqual(len(mat[(3, 4)]), 1)

    def test_aggregations_and_kpis(self):
        by_tag = self.reg.aggregate_by("tag")
        self.assertIn("Security", by_tag)
        self.assertGreaterEqual(by_tag["Security"]["count"], 1)
        kpis = self.reg.portfolio_kpis()
        self.assertEqual(kpis["count"], 3)
        self.assertEqual(kpis["open"], 2)
        self.assertGreater(kpis["avg_score"], 0.0)

    def test_sla_and_overdue(self):
        # r1 score 10 -> bucket "high" => 30d interval by our rule
        as_of = self.r1.updated_at + timedelta(days=31)
        overdue = self.reg.overdue(as_of)
        ids = {r.id for r in overdue}
        self.assertIn("r1", ids)
        # Closed risks are never overdue
        self.reg.upsert(Risk(id="r4", title="Closed", owner="x", tag="x", likelihood=5, impact=5, status="closed"))
        self.assertNotIn("r4", {r.id for r in self.reg.overdue(datetime.now(tz=UTC) + timedelta(days=999))})

    def test_clamps_grid_bounds(self):
        # Likelihood/impact are clamped to the grid
        r = Risk(id="clamp", title="Clamp", likelihood=99, impact=0)
        self.reg.upsert(r)
        got = self.reg.get("clamp")
        self.assertEqual(got.likelihood, 5)
        self.assertEqual(got.impact, 1)

    def test_csv_roundtrip(self):
        # Export
        csv_text = self.reg.export_csv()
        # New register, import, contents should be equivalent (ids preserved)
        reg2 = RiskRegister(L=5, C=5)
        reg2.import_csv(csv_text)
        got_ids = sorted(r.id for r in reg2.list())
        exp_ids = sorted(r.id for r in self.reg.list())
        self.assertEqual(got_ids, exp_ids)
        # Modify reg2 and ensure upsert updates instead of duplicating
        reg2.upsert(Risk(id="r2", title="Cloud outage mitigations", likelihood=2, impact=3))
        self.assertEqual(len(reg2.list()), len(self.reg.list()))
        self.assertEqual(reg2.get("r2").title, "Cloud outage mitigations")


class TestEdgeCases(unittest.TestCase):
    def test_empty_register_kpis(self):
        reg = RiskRegister()
        k = reg.portfolio_kpis()
        self.assertEqual(k["count"], 0)
        self.assertEqual(k["open"], 0)
        self.assertEqual(k["critical"], 0)
        self.assertEqual(k["avg_score"], 0.0)

    def test_bucket_thresholds(self):
        # Make sure edges fall into expected buckets for 5x5 grid (max=25)
        max_s = 25
        self.assertEqual(severity_bucket(1, max_s), "low")
        self.assertEqual(severity_bucket(5, max_s), "moderate")   # 5/25 = 0.2 -> moderate edge
        self.assertEqual(severity_bucket(11, max_s), "high")      # ~0.44+ epsilon
        self.assertEqual(severity_bucket(25, max_s), "critical")

    def test_severity_color_is_stable(self):
        max_s = 25
        t1 = severity_color_t(12, max_s)
        t2 = severity_color_t(12, max_s)  # deterministic
        self.assertAlmostEqual(t1, t2, places=12)
        # Ends approach 0 and 1
        self.assertLess(severity_color_t(1, max_s), 0.2)
        self.assertGreater(severity_color_t(25, max_s), 0.8)

    def test_import_missing_dates_defaults_now(self):
        reg = RiskRegister()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id","title","owner","tag","likelihood","impact","status","notes","created_at","updated_at","target_date"])
        w.writerow(["x1","X","O","T",3,4,"open","", "", "", ""])
        reg.import_csv(buf.getvalue())
        self.assertEqual(len(reg.list()), 1)
        self.assertEqual(reg.get("x1").score, 12)

# PyTest bridge
def test_pytest_bridge():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRiskBasics)
    suite2 = unittest.defaultTestLoader.loadTestsFromTestCase(TestEdgeCases)
    allsuite = unittest.TestSuite([suite, suite2])
    res = unittest.TextTestRunner(verbosity=0).run(allsuite)
    assert res.wasSuccessful(), "risk tests failed"