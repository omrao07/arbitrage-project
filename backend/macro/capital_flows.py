# backend/treasury/capital_flows.py
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --------- Optional adapters (kept soft so this file stands alone) ----------
try:
    from backend.treasury.bank_adapters import BankAdapterBase, Transaction as BankTx # type: ignore
except Exception:
    class BankAdapterBase:  # minimal stub
        def list_transactions(self, *_a, **_k): return []
    @dataclass
    class BankTx:
        tx_id: str
        account_id: str
        amount: float
        currency: str
        description: str
        booked_at: str
        counterparty: Optional[str] = None
        meta: Dict[str, Any] = field(default_factory=dict)


# ============================== Data models ===============================

@dataclass
class FlowEvent:
    """
    Generic capital flow. Positive = cash in (subscription, realized PnL, interest),
    Negative = cash out (redemption, fees, funding transfers).
    """
    ts: datetime
    amount: float
    currency: str = "USD"
    kind: str = "other"              # 'subscription','redemption','fee','interest','dividend','transfer','pnl','tax','other'
    source: str = "manual"           # 'bank','broker','ops','pnl','manual', etc.
    account: Optional[str] = None    # account id or name
    region: Optional[str] = None     # 'US','IN','EU',...
    strategy: Optional[str] = None   # strategy tag
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_base(self, fx: Dict[str, float], base_ccy: str) -> float:
        """
        Convert to base currency with a simple spot table (pair like 'USDINR' or direct rates map).
        If fx is {"USD":1, "INR":83} treat as 1 base_ccy unit pricing per other ccy:
          base_amount = amount / fx[currency]   (assuming fx[base_ccy] == 1)
        """
        if self.currency.upper() == base_ccy.upper():
            return float(self.amount)
        # allow two formats:
        # 1) table keyed by currency -> units per base (fx["INR"]=83 means 1 USD = 83 INR if base=USD)
        if self.currency.upper() in fx and base_ccy.upper() in fx:
            # interpret as "units per base" (e.g., fx["USD"]=1, fx["INR"]=83)
            base_per_self = fx[base_ccy.upper()] / fx[self.currency.upper()]
            return float(self.amount) * base_per_self
        # 2) direct pairs like "USDINR": price of pair (base=USD)
        pair = f"{base_ccy.upper()}{self.currency.upper()}"
        if pair in fx and fx[pair] != 0:
            return float(self.amount) / float(fx[pair])
        # fallback no conversion
        return float(self.amount)


@dataclass
class FlowBook:
    """
    In-memory store of flows with helpers for rollups and KPIs.
    """
    base_ccy: str = "USD"
    events: List[FlowEvent] = field(default_factory=list)

    # ------------ ingest ------------
    def add(self, *events: FlowEvent) -> None:
        self.events.extend(events)

    def add_bank_transactions(
        self,
        bank: BankAdapterBase,
        account_id: str,
        *,
        sign: Optional[int] = None,
        currency_hint: Optional[str] = None,
        mapper: Optional[callable] = None, # type: ignore
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 1000,
    ) -> int:
        """
        Pull transactions from a bank adapter and map them to FlowEvents.
        - 'sign' lets you flip conventions if your adapter uses opposite signs.
        - 'mapper' can classify kinds from descriptions.
        Returns number of events ingested.
        """
        txs: List[BankTx] = bank.list_transactions(account_id, since=since, until=until, limit=limit)  # type: ignore[arg-type]
        n0 = len(self.events)

        for t in txs:
            amt = float(t.amount) * (sign if sign else 1)
            kind = "other"
            src = "bank"
            if mapper:
                try:
                    k = mapper(t)
                    if isinstance(k, tuple):
                        kind, src = k[0], k[1] if len(k) > 1 else src
                    elif isinstance(k, str):
                        kind = k
                except Exception:
                    pass
            ev = FlowEvent(
                ts=_parse_ts(getattr(t, "booked_at", None)),
                amount=amt,
                currency=(t.currency or currency_hint or self.base_ccy),
                kind=kind,
                source=src,
                account=getattr(t, "account_id", None),
                meta={"tx_id": getattr(t, "tx_id", None), "desc": getattr(t, "description", "")},
            )
            self.add(ev)
        return len(self.events) - n0

    # ------------ transforms ------------
    def to_base_series(self, fx: Dict[str, float]) -> List[Tuple[datetime, float]]:
        return [(e.ts, e.to_base(fx, self.base_ccy)) for e in self.events]

    def filtered(self, *, kind: Optional[str] = None, source: Optional[str] = None, strategy: Optional[str] = None, region: Optional[str] = None) -> "FlowBook":
        es = [e for e in self.events
              if (kind is None or e.kind == kind)
              and (source is None or e.source == source)
              and (strategy is None or e.strategy == strategy)
              and (region is None or e.region == region)]
        return FlowBook(base_ccy=self.base_ccy, events=es)

    # ------------ rollups ------------
    def rollup_daily(self, fx: Dict[str, float]) -> List[Tuple[date, float]]:
        daymap: Dict[date, float] = {}
        for ts, amt in self.to_base_series(fx):
            d = ts.date()
            daymap[d] = daymap.get(d, 0.0) + amt
        return sorted(daymap.items(), key=lambda x: x[0])

    def rollup_by(self, key: str, fx: Dict[str, float]) -> Dict[str, float]:
        """
        key in {'kind','source','strategy','region','account','currency'}
        """
        agg: Dict[str, float] = {}
        for e in self.events:
            k = getattr(e, key) if key != "currency" else e.currency
            k = k or "NA"
            agg[str(k)] = agg.get(str(k), 0.0) + e.to_base(fx, self.base_ccy)
        return dict(sorted(agg.items(), key=lambda kv: -abs(kv[1])))

    def cumulative(self, fx: Dict[str, float]) -> List[Tuple[date, float]]:
        cum = 0.0
        out: List[Tuple[date, float]] = []
        for d, x in self.rollup_daily(fx):
            cum += x
            out.append((d, cum))
        return out

    # ------------ KPIs ------------
    def net_subscriptions(self, fx: Dict[str, float]) -> float:
        sub = sum(e.to_base(fx, self.base_ccy) for e in self.events if e.kind == "subscription")
        red = sum(e.to_base(fx, self.base_ccy) for e in self.events if e.kind == "redemption")
        return sub - red

    def fees_paid(self, fx: Dict[str, float]) -> float:
        return sum(e.to_base(fx, self.base_ccy) for e in self.events if e.kind == "fee")

    def realized_pnl(self, fx: Dict[str, float]) -> float:
        return sum(e.to_base(fx, self.base_ccy) for e in self.events if e.kind == "pnl")

    def moic(self, fx: Dict[str, float]) -> Optional[float]:
        """
        MOIC = (Total Distributions) / (Total Paid-In).
        Treat positive flows as distributions; negative as paid-in (subscriptions).
        """
        paid_in = sum(-min(0.0, e.to_base(fx, self.base_ccy)) for e in self.events)
        distributed = sum(max(0.0, e.to_base(fx, self.base_ccy)) for e in self.events)
        if paid_in <= 0:
            return None
        return distributed / paid_in

    def xirr(self, fx: Dict[str, float], *, guess: float = 0.15) -> Optional[float]:
        """
        Compute XIRR using Newton’s method on day counts.
        Convention: investments (subs) negative, distributions positive.
        """
        if not self.events:
            return None
        # Build cash flows at daily granularity
        base = min(e.ts for e in self.events).date()
        flows: List[Tuple[float, float]] = []  # (years, amount)
        for e in self.events:
            days = (e.ts.date() - base).days
            flows.append((days / 365.0, e.to_base(fx, self.base_ccy)))
        # Need at least one positive and one negative
        if not any(v > 0 for _, v in flows) or not any(v < 0 for _, v in flows):
            return None

        def npv(r: float) -> float:
            return sum(v / ((1 + r) ** t) for t, v in flows)

        def dnpv(r: float) -> float:
            return sum(-t * v / ((1 + r) ** (t + 1)) for t, v in flows)

        r = guess
        for _ in range(100):
            f = npv(r)
            df = dnpv(r)
            if abs(f) < 1e-9:
                return r
            step = f / (df if df != 0 else 1e-12)
            r -= step
            if r <= -0.9999:  # avoid blowups
                r = -0.9
        return r

    # ------------ I/O ------------
    def to_json(self) -> str:
        return json.dumps([_ev_to_dict(e) for e in self.events], default=str, indent=2)

    @staticmethod
    def from_json(s: str, *, base_ccy: str = "USD") -> "FlowBook":
        arr = json.loads(s)
        fb = FlowBook(base_ccy=base_ccy)
        for o in arr:
            fb.add(_ev_from_dict(o))
        return fb

    def write_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ts","amount","currency","kind","source","account","region","strategy","meta_json"
            ])
            w.writeheader()
            for e in self.events:
                w.writerow({
                    "ts": e.ts.isoformat(),
                    "amount": e.amount,
                    "currency": e.currency,
                    "kind": e.kind,
                    "source": e.source,
                    "account": e.account or "",
                    "region": e.region or "",
                    "strategy": e.strategy or "",
                    "meta_json": json.dumps(e.meta or {}),
                })

    @staticmethod
    def read_csv(path: str, *, base_ccy: str = "USD") -> "FlowBook":
        fb = FlowBook(base_ccy=base_ccy)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                fb.add(FlowEvent(
                    ts=_parse_ts(row.get("ts")),
                    amount=float(row.get("amount") or 0.0),
                    currency=(row.get("currency") or base_ccy),
                    kind=row.get("kind") or "other",
                    source=row.get("source") or "manual",
                    account=row.get("account") or None,
                    region=row.get("region") or None,
                    strategy=row.get("strategy") or None,
                    meta=json.loads(row.get("meta_json") or "{}"),
                ))
        return fb


# ============================== Utilities ===============================

def _parse_ts(x: Any) -> datetime:
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime.combine(x, datetime.min.time())
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x.replace("Z", "+00:00"))
        except Exception:
            pass
    # fallback now
    return datetime.utcnow()

def _ev_to_dict(e: FlowEvent) -> Dict[str, Any]:
    d = asdict(e)
    d["ts"] = e.ts.isoformat()
    return d

def _ev_from_dict(d: Dict[str, Any]) -> FlowEvent:
    dd = dict(d)
    dd["ts"] = _parse_ts(d.get("ts"))
    dd["meta"] = d.get("meta") or {}
    return FlowEvent(**dd)


# ============================== Tiny demo / CLI ===============================

def _demo() -> None:
    # Example FX table: treat base USD=1, INR=83 => 1 USD = 83 INR
    fx = {"USD": 1.0, "INR": 83.0, "EUR": 0.92}

    fb = FlowBook(base_ccy="USD")
    fb.add(
        FlowEvent(ts=datetime(2025, 1, 5), amount=-250_000, currency="USD", kind="subscription", source="ops"),
        FlowEvent(ts=datetime(2025, 2, 1), amount=-100_000, currency="USD", kind="subscription", source="ops"),
        FlowEvent(ts=datetime(2025, 4, 15), amount=+6_500, currency="USD", kind="interest", source="bank"),
        FlowEvent(ts=datetime(2025, 6, 30), amount=+40_000, currency="USD", kind="pnl", source="pnl"),
        FlowEvent(ts=datetime(2025, 7, 15), amount=-4_000, currency="USD", kind="fee", source="ops"),
        FlowEvent(ts=datetime(2025, 8, 10), amount=+90_000, currency="USD", kind="redemption", source="ops"),
    )

    print("Net subs (USD):", round(fb.net_subscriptions(fx), 2))
    print("Fees paid (USD):", round(fb.fees_paid(fx), 2))
    print("MOIC:", round(fb.moic(fx) or 0.0, 4))
    print("XIRR:", round(100 * (fb.xirr(fx) or 0.0), 3), "%")

    print("By kind:", fb.rollup_by("kind", fx))
    print("Cumulative:", fb.cumulative(fx)[-3:])

if __name__ == "__main__":
    _demo()