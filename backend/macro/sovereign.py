# backend/treasury/soverign.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Tuple

# Reuse your adapter (we wrote this earlier)
from backend.treasury.soverign_adapter import ( # type: ignore
    YieldCurve, CurvePoint, BondSpec, SovereignAdapter,
    price_from_curve, ytm_from_price, duration_convexity, dv01,
)

# ------------------------------ Facade ---------------------------------

@dataclass
class Bucket:
    """Key-rate bucket definition (tenor midpoint in years, width in years)."""
    mid_yrs: float
    width_yrs: float

class SovereignService:
    """
    Convenience wrapper around SovereignAdapter:
      • set/load curves
      • price/ytm
      • carry/roll & risk
      • bucketed DV01 (key-rate)
    """

    def __init__(self):
        self.sov = SovereignAdapter()

    # ---- Curve management ----
    def set_curve(self, currency: str, curve_date: date | str, points: List[Tuple[float, float]]) -> YieldCurve:
        c = YieldCurve(
            curve_date=_d(curve_date),
            currency=currency.upper(),
            points=[CurvePoint(float(t), float(y if y < 1.5 else y/100.0)) for t, y in points],
        )
        self.sov.set_curve(c)
        return c

    def load_curve_csv(self, currency: str, curve_date: date | str, path: str) -> YieldCurve:
        return self.sov.load_csv(path, currency, curve_date)

    def get_curve(self, currency: str, curve_date: date | str) -> YieldCurve:
        return self.sov.get_curve(currency, _d(curve_date))

    # ---- Pricing ----
    def price_bond(self, spec: BondSpec, asof: date | str, *, clean: bool = True) -> Dict:
        pr = price_from_curve(spec, self.get_curve(spec.currency, asof), _d(asof), clean=clean)
        return {"clean": pr.clean, "dirty": pr.dirty, "accrued": pr.accrued}

    def ytm_from_clean(self, spec: BondSpec, asof: date | str, price_clean: float, guess: float = 0.07) -> Dict:
        res = ytm_from_price(spec, _d(asof), price_clean, guess=guess)
        return {
            "ytm": res.ytm, "clean": res.clean, "dirty": res.dirty, "accrued": res.accrued,
            "mod_duration": res.mod_duration, "macaulay_duration": res.macaulay_duration,
            "convexity": res.convexity, "dv01": res.dv01,
        }

    # ---- Risk ----
    def risk_measures(self, spec: BondSpec, asof: date | str, ytm: float) -> Dict[str, float]:
        mod, mac, conv = duration_convexity(spec, _d(asof), ytm)
        return {"mod_duration": mod, "macaulay_duration": mac, "convexity": conv, "dv01": dv01(spec, _d(asof), ytm)}

    def bucketed_dv01(
        self,
        spec: BondSpec,
        asof: date | str,
        buckets: Iterable[Bucket] = (Bucket(0.5, 0.5), Bucket(2, 1), Bucket(5, 2), Bucket(10, 3)),
        bump_bp: float = 1.0,
    ) -> Dict[str, float]:
        """
        Key-rate DV01 by locally bumping the curve around each bucket.
        """
        asof_d = _d(asof)
        base_curve = self.get_curve(spec.currency, asof_d)

        def bump(curve: YieldCurve, mid: float, width: float, bp: float) -> YieldCurve:
            shift = float(bp) / 1e4
            pts = []
            lo, hi = mid - width/2.0, mid + width/2.0
            for p in curve.sorted_points():
                y = p.yld + (shift if lo <= p.tenor_yrs <= hi else 0.0)
                pts.append(CurvePoint(p.tenor_yrs, max(0.0, y)))
            return YieldCurve(curve_date=curve.curve_date, currency=curve.currency, points=pts)

        base_px = price_from_curve(spec, base_curve, asof_d, clean=False).dirty
        out: Dict[str, float] = {}
        for b in buckets:
            up = price_from_curve(spec, bump(base_curve, b.mid_yrs, b.width_yrs, +bump_bp), asof_d, clean=False).dirty
            dn = price_from_curve(spec, bump(base_curve, b.mid_yrs, b.width_yrs, -bump_bp), asof_d, clean=False).dirty
            # DV01 for the bucket: symmetric diff per 1bp (dollar price change)
            out[f"{b.mid_yrs:g}y"] = (dn - up) / 2.0
        out["_total_dv01"] = sum(out.values())
        out["_base_dirty"] = base_px
        return out

    def carry_roll(self, spec: BondSpec, asof: date | str, horizon_days: int = 30) -> Dict[str, float]:
        # Reuse adapter’s carry/roll
        curve = self.get_curve(spec.currency, asof)
        from backend.treasury.soverign_adapter import carry_roll_down # type: ignore
        return carry_roll_down(spec, curve, _d(asof), horizon_days=horizon_days)


# ------------------------------ CLI ------------------------------------

def _d(x: date | str | datetime) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    return datetime.fromisoformat(str(x)).date()

def _bond_from_args(args: argparse.Namespace) -> BondSpec:
    return BondSpec(
        issuer=args.issuer, currency=args.ccy, face=args.face, coupon=args.coupon,
        freq=args.freq, issue_date=_d(args.issue), maturity=_d(args.maturity),
        daycount=args.daycount,
    )

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sovereign curve & bond tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # set curve from CSV
    c = sub.add_parser("load-curve", help="Load curve from CSV (tenor_yrs,yield)")
    c.add_argument("--ccy", required=True)
    c.add_argument("--date", required=True, help="YYYY-MM-DD")
    c.add_argument("--csv", required=True)
    c.set_defaults(action="load_curve")

    # set curve from list
    s = sub.add_parser("set-curve", help="Set curve with inline points")
    s.add_argument("--ccy", required=True)
    s.add_argument("--date", required=True)
    s.add_argument("--points", required=True, help='JSON: [[0.25,0.068],[1,0.07],[5,0.073],[10,0.0745]]')
    s.set_defaults(action="set_curve")

    # price
    p = sub.add_parser("price", help="Price a fixed-coupon bond off curve")
    _bond_args(p)
    p.add_argument("--asof", required=True)
    p.add_argument("--dirty", action="store_true")
    p.set_defaults(action="price")

    # ytm from clean
    y = sub.add_parser("ytm", help="Solve YTM from clean price")
    _bond_args(y)
    y.add_argument("--asof", required=True)
    y.add_argument("--clean", type=float, required=True)
    y.add_argument("--guess", type=float, default=0.07)
    y.set_defaults(action="ytm")

    # bucketed dv01
    b = sub.add_parser("kr-dv01", help="Key-rate DV01 by buckets")
    _bond_args(b)
    b.add_argument("--asof", required=True)
    b.add_argument("--buckets", default='[["0.5",0.5],["2",1],["5",2],["10",3]]', help='JSON [[mid, width], ...]')
    b.add_argument("--bump", type=float, default=1.0, help="bp")
    b.set_defaults(action="krdv01")

    # carry/roll
    cr = sub.add_parser("carry", help="Carry & roll-down estimate")
    _bond_args(cr)
    cr.add_argument("--asof", required=True)
    cr.add_argument("--days", type=int, default=30)
    cr.set_defaults(action="carry")

    return ap

def _bond_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--issuer", required=True)
    p.add_argument("--ccy", required=True)
    p.add_argument("--face", type=float, default=100.0)
    p.add_argument("--coupon", type=float, required=True, help="decimal, e.g., 0.0718 for 7.18%")
    p.add_argument("--freq", type=int, default=2, help="1=annual,2=semi,4=quarterly")
    p.add_argument("--issue", required=True)
    p.add_argument("--maturity", required=True)
    p.add_argument("--daycount", default="ACT/365")

def main(argv: Optional[List[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    svc = SovereignService()

    if args.action == "load_curve":
        curve = svc.load_curve_csv(args.ccy, args.date, args.csv)
        print(json.dumps({"curve": {"ccy": curve.currency, "date": str(curve.curve_date),
                                    "points": [asdict(p) for p in curve.sorted_points()] }}, indent=2))
        return

    if args.action == "set_curve":
        points = json.loads(args.points)
        curve = svc.set_curve(args.ccy, args.date, [(float(t), float(y)) for t, y in points])
        print(json.dumps({"curve": {"ccy": curve.currency, "date": str(curve.curve_date),
                                    "points": [asdict(p) for p in curve.sorted_points()] }}, indent=2))
        return

    spec = _bond_from_args(args)

    if args.action == "price":
        out = svc.price_bond(spec, args.asof, clean=not args.dirty)
        print(json.dumps(out, indent=2)); return

    if args.action == "ytm":
        out = svc.ytm_from_clean(spec, args.asof, args.clean, guess=args.guess)
        print(json.dumps(out, indent=2)); return

    if args.action == "krdv01":
        raw = json.loads(args.buckets)
        bks = [Bucket(float(m), float(w)) for m, w in raw]
        out = svc.bucketed_dv01(spec, args.asof, bks, bump_bp=args.bump)
        print(json.dumps(out, indent=2)); return

    if args.action == "carry":
        out = svc.carry_roll(spec, args.asof, horizon_days=args.days)
        print(json.dumps(out, indent=2)); return


if __name__ == "__main__":
    main()