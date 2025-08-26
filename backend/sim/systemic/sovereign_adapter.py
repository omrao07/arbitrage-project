# backend/treasury/soverign_adapter.py
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Iterable, Tuple

# ============================== Time & Daycount ==============================

def _to_date(d: date | datetime | str) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    # try ISO
    return datetime.fromisoformat(str(d)).date()

def yearfrac(d1: date | datetime | str, d2: date | datetime | str, basis: str = "ACT/365") -> float:
    a = _to_date(d1)
    b = _to_date(d2)
    days = (b - a).days
    if basis.upper() in ("ACT/365", "ACT/365F", "ACT365F"):
        return days / 365.0
    if basis.upper() in ("ACT/360", "ACT360"):
        return days / 360.0
    if basis.upper() in ("30/360", "30E/360", "30U/360"):
        # simple 30/360 US (not distinguishing EU/ISDA variants for brevity)
        y1, m1, d1 = a.year, a.month, min(30, a.day) # type: ignore
        y2, m2, d2 = b.year, b.month, min(30, b.day) # type: ignore
        return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)) / 360.0 # type: ignore
    # default
    return days / 365.0

# ============================== Curve Models ==============================

@dataclass
class CurvePoint:
    tenor_yrs: float
    yld: float  # simple annualized yield as decimal (e.g., 0.0725 == 7.25%)

@dataclass
class YieldCurve:
    curve_date: date
    currency: str = "USD"
    label: str = "govt"
    method: str = "loglinear_zero"  # interpolation mode
    points: List[CurvePoint] = field(default_factory=list)

    def sorted_points(self) -> List[CurvePoint]:
        return sorted(self.points, key=lambda p: p.tenor_yrs)

    # --- simple bootstrap helpers (par to zero using log-linear between nodes) ---
    def zero_yield(self, t: float) -> float:
        """
        Interpolate zero yield at year fraction t using log-linear on discount factors.
        Assumes 'points' are zero yields by tenor. If you feed par points, pre-convert.
        """
        pts = self.sorted_points()
        if not pts:
            raise ValueError("Curve has no points")
        if t <= pts[0].tenor_yrs:
            return pts[0].yld
        if t >= pts[-1].tenor_yrs:
            return pts[-1].yld

        # find bracketing nodes
        for i in range(1, len(pts)):
            left = pts[i - 1]
            right = pts[i]
            if left.tenor_yrs <= t <= right.tenor_yrs:
                r1, r2 = left.yld, right.yld
                # log-linear on DFs: D = exp(-r * t)  => ln D = -r t
                lnD1 = -r1 * left.tenor_yrs
                lnD2 = -r2 * right.tenor_yrs
                w = (t - left.tenor_yrs) / (right.tenor_yrs - left.tenor_yrs)
                lnDt = lnD1 * (1 - w) + lnD2 * w
                rt = -lnDt / max(t, 1e-9)
                return rt
        # fallback (shouldn’t hit)
        return pts[-1].yld

    def discount_factor(self, t: float) -> float:
        r = self.zero_yield(t)
        return math.exp(-r * t)

# ============================== Bond Models ==============================

@dataclass
class BondSpec:
    """
    Vanilla fixed-coupon bullet.
    coupon: annual rate as decimal (0.07 = 7%)
    freq: 1=annual, 2=semi, 4=quarterly
    daycount: used for accrual and YTM conventions
    """
    issuer: str
    currency: str
    face: float
    coupon: float
    freq: int
    issue_date: date
    maturity: date
    daycount: str = "ACT/365"
    street_count: str = "30/360"  # for schedule if you want fixed spacing

@dataclass
class PriceResult:
    clean: float
    dirty: float
    accrued: float
    ytm: Optional[float] = None
    mod_duration: Optional[float] = None
    macaulay_duration: Optional[float] = None
    dv01: Optional[float] = None
    convexity: Optional[float] = None

# ---------- coupon schedule & accrual ----------

def _coupon_dates(spec: BondSpec) -> List[date]:
    # approximation: roll back by exact frequency in months
    months = {1: 12, 2: 6, 4: 3}.get(spec.freq, 6)
    out: List[date] = []
    d = spec.maturity
    while d > spec.issue_date:
        out.append(d)
        # naive month subtraction
        m = d.month - months
        y = d.year
        while m <= 0:
            m += 12
            y -= 1
        day = min(d.day, _month_end_day(y, m))
        d = date(y, m, day)
    out.sort()
    return out

def _month_end_day(y: int, m: int) -> int:
    if m in (1,3,5,7,9,10,12):
        return 31 if m != 2 else 28
    if m == 2:
        # leap?
        if (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0):
            return 29
        return 28
    return 30

def _last_coupon_on_or_before(spec: BondSpec, asof: date) -> Tuple[Optional[date], Optional[date]]:
    cds = _coupon_dates(spec)
    prev = None
    nextd = None
    for d in cds:
        if d <= asof:
            prev = d
        elif d > asof and nextd is None:
            nextd = d
    return prev, nextd

def accrued_interest(spec: BondSpec, asof: date) -> float:
    prev, nxt = _last_coupon_on_or_before(spec, asof)
    if prev is None:
        # before first coupon: accrue from issue_date
        start = spec.issue_date
        end = nxt or spec.maturity
    else:
        start = prev
        end = nxt or spec.maturity
    frac = max(0.0, min(1.0, yearfrac(start, asof, spec.daycount) / max(1e-9, yearfrac(start, end, spec.daycount))))
    coupon_amt = spec.face * spec.coupon / spec.freq
    return coupon_amt * frac

# ---------- pricing & YTM ----------

def price_from_curve(spec: BondSpec, curve: YieldCurve, asof: date, *, clean: bool = True) -> PriceResult:
    asof = _to_date(asof)
    cds = [d for d in _coupon_dates(spec) if d > asof]
    cf = spec.face * spec.coupon / spec.freq
    pv = 0.0
    for d in cds:
        t = yearfrac(asof, d, "ACT/365")  # using ACT/365 to align with zero comp here
        pv += cf * curve.discount_factor(t)
    # redemption
    t_red = yearfrac(asof, spec.maturity, "ACT/365")
    pv += spec.face * curve.discount_factor(t_red)

    ai = accrued_interest(spec, asof)
    dirty = pv
    clean_px = dirty - ai if clean else dirty
    return PriceResult(clean=clean_px, dirty=dirty, accrued=ai)

def ytm_from_price(spec: BondSpec, asof: date, price_clean: float, guess: float = 0.07) -> PriceResult:
    asof = _to_date(asof)
    cds = [d for d in _coupon_dates(spec) if d > asof]
    cf = spec.face * spec.coupon / spec.freq
    ai = accrued_interest(spec, asof)
    target_dirty = price_clean + ai

    # Newton-Raphson on YTM with payments at freq
    y = max(1e-6, guess)
    for _ in range(50):
        pv = 0.0
        dv = 0.0
        for d in cds:
            t = yearfrac(asof, d, spec.daycount)
            n = t * spec.freq
            disc = (1 + y / spec.freq) ** (-n)
            pv += cf * disc
            dv += -cf * (n / (spec.freq + y)) * disc  # derivative w.r.t y (approx)
        # redemption
        t_red = yearfrac(asof, spec.maturity, spec.daycount)
        n_red = t_red * spec.freq
        disc_red = (1 + y / spec.freq) ** (-n_red)
        pv += spec.face * disc_red
        dv += -spec.face * (n_red / (spec.freq + y)) * disc_red

        diff = pv - target_dirty
        if abs(diff) < 1e-8:
            break
        step = diff / max(dv, 1e-12)
        y -= step
        if y < 0:
            y = 1e-6
    res = PriceResult(clean=price_clean, dirty=target_dirty, accrued=ai, ytm=float(y))
    # duration / conv / dv01 at solved y
    dur_mod, dur_mac, conv = duration_convexity(spec, asof, y)
    res.mod_duration = dur_mod
    res.macaulay_duration = dur_mac
    res.convexity = conv
    res.dv01 = dv01(spec, asof, y)
    return res

def price_from_ytm(spec: BondSpec, asof: date, ytm: float, clean: bool = True) -> PriceResult:
    asof = _to_date(asof)
    cds = [d for d in _coupon_dates(spec) if d > asof]
    cf = spec.face * spec.coupon / spec.freq
    pv = 0.0
    for d in cds:
        t = yearfrac(asof, d, spec.daycount)
        n = t * spec.freq
        disc = (1 + ytm / spec.freq) ** (-n)
        pv += cf * disc
    t_red = yearfrac(asof, spec.maturity, spec.daycount)
    n_red = t_red * spec.freq
    pv += spec.face * (1 + ytm / spec.freq) ** (-n_red)
    ai = accrued_interest(spec, asof)
    dirty = pv
    return PriceResult(clean=(dirty - ai) if clean else dirty, dirty=dirty, accrued=ai, ytm=ytm)

# ---------- risk measures ----------

def duration_convexity(spec: BondSpec, asof: date, ytm: float) -> Tuple[float, float, float]:
    """
    Returns (modified_duration, macaulay_duration, convexity).
    """
    asof = _to_date(asof)
    cds = [d for d in _coupon_dates(spec) if d > asof]
    cf = spec.face * spec.coupon / spec.freq

    pv = 0.0
    sum_t_pv = 0.0
    sum_t2_pv = 0.0
    for d in cds:
        t = yearfrac(asof, d, spec.daycount)
        n = t * spec.freq
        disc = (1 + ytm / spec.freq) ** (-n)
        c = cf
        pv_leg = c * disc
        pv += pv_leg
        sum_t_pv += t * pv_leg
        sum_t2_pv += t * t * pv_leg

    t_red = yearfrac(asof, spec.maturity, spec.daycount)
    n_red = t_red * spec.freq
    disc_red = (1 + ytm / spec.freq) ** (-n_red)
    pv_red = spec.face * disc_red
    pv += pv_red
    sum_t_pv += t_red * pv_red
    sum_t2_pv += t_red * t_red * pv_red

    macaulay = sum_t_pv / max(pv, 1e-12)
    mod = macaulay / (1 + ytm / spec.freq)
    # convexity (approx continuous-time variant)
    conv = sum_t2_pv / max(pv, 1e-12)
    return float(mod), float(macaulay), float(conv)

def dv01(spec: BondSpec, asof: date, ytm: float) -> float:
    bump = 0.0001  # 1bp
    p_up = price_from_ytm(spec, asof, ytm + bump, clean=False).dirty
    p_dn = price_from_ytm(spec, asof, ytm - bump, clean=False).dirty
    return (p_dn - p_up) / 2.0  # dollars per 1bp

# ---------- carry & roll-down ----------

def carry_roll_down(spec: BondSpec, curve: YieldCurve, asof: date, horizon_days: int = 30) -> Dict[str, float]:
    """
    Estimate carry + roll-down over horizon:
      carry ≈ coupon accrual over horizon - financing (ignored here)
      roll-down ≈ price(s(t+h)) under shifted maturity along current curve
    """
    asof = _to_date(asof)
    h = asof + timedelta(days=int(horizon_days))
    # current price
    now_pr = price_from_curve(spec, curve, asof, clean=False).dirty
    # horizon price: approximate by aging bond and re-pricing with same curve
    pr_h = price_from_curve(spec, curve, h, clean=False).dirty
    # accrued change
    ai_now = accrued_interest(spec, asof)
    ai_h = accrued_interest(spec, h)
    carry = (ai_h - ai_now)
    roll = (pr_h - now_pr) - carry
    return {"carry": carry, "roll_down": roll, "total": carry + roll}

# ============================== Adapter ==============================

class SovereignAdapter:
    """
    In-memory curves + helpers.
    - set_curve / load_csv
    - price bonds using curve zeros
    - compute DV01/duration/carry/roll
    """

    def __init__(self):
        self._curves: Dict[Tuple[str, date], YieldCurve] = {}  # (currency, curve_date) -> curve

    # ---- curve mgmt ----

    def set_curve(self, curve: YieldCurve) -> None:
        self._curves[(curve.currency.upper(), curve.curve_date)] = curve

    def get_curve(self, currency: str, curve_date: date | str | datetime) -> YieldCurve:
        key = (currency.upper(), _to_date(curve_date))
        if key not in self._curves:
            raise KeyError(f"curve not found for {key}")
        return self._curves[key]

    def load_csv(self, path: str, currency: str, curve_date: date | str | datetime) -> YieldCurve:
        """
        CSV with columns: tenor_yrs,yield (decimal or percent like 7.25)
        """
        pts: List[CurvePoint] = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                t = float(row["tenor_yrs"])
                y = float(row["yield"])
                if y > 1.0:  # assume percentage
                    y = y / 100.0
                pts.append(CurvePoint(tenor_yrs=t, yld=y))
        curve = YieldCurve(curve_date=_to_date(curve_date), currency=currency.upper(), points=pts)
        self.set_curve(curve)
        return curve

    # ---- pricing wrappers ----

    def price_bond(self, spec: BondSpec, curve_date: date | str | datetime, clean: bool = True) -> PriceResult:
        ccy = spec.currency.upper()
        curve = self.get_curve(ccy, curve_date)
        return price_from_curve(spec, curve, _to_date(curve_date), clean=clean)

    def price_from_ytm(self, spec: BondSpec, asof: date | str | datetime, ytm: float, clean: bool = True) -> PriceResult:
        return price_from_ytm(spec, _to_date(asof), ytm, clean=clean)

    def ytm_from_price(self, spec: BondSpec, asof: date | str | datetime, price_clean: float, guess: float = 0.07) -> PriceResult:
        return ytm_from_price(spec, _to_date(asof), price_clean, guess=guess)

    def risk_measures(self, spec: BondSpec, asof: date | str | datetime, ytm: float) -> Dict[str, float]:
        mod, mac, conv = duration_convexity(spec, _to_date(asof), ytm)
        return {"mod_duration": mod, "macaulay_duration": mac, "convexity": conv, "dv01": dv01(spec, _to_date(asof), ytm)}

    def carry_roll(self, spec: BondSpec, curve_date: date | str | datetime, horizon_days: int = 30) -> Dict[str, float]:
        curve = self.get_curve(spec.currency, curve_date)
        return carry_roll_down(spec, curve, _to_date(curve_date), horizon_days=horizon_days)

# ============================== Simple Smoke Test ==============================

if __name__ == "__main__":
    # Build a tiny INR curve (tenors in years, yields in %)
    inr_curve = YieldCurve(
        curve_date=date.today(),
        currency="INR",
        points=[
            CurvePoint(0.25, 0.068),
            CurvePoint(1.0,  0.070),
            CurvePoint(3.0,  0.0715),
            CurvePoint(5.0,  0.073),
            CurvePoint(10.0, 0.0745),
        ],
    )

    adapter = SovereignAdapter()
    adapter.set_curve(inr_curve)

    # 5Y G-Sec: 7.18% semi-annual coupon, face 100
    spec = BondSpec(
        issuer="GOI", currency="INR", face=100.0, coupon=0.0718, freq=2,
        issue_date=date(2023, 8, 1), maturity=date(2028, 8, 1), daycount="ACT/365"
    )

    # Price off today's curve
    pr = adapter.price_bond(spec, date.today(), clean=True)
    print("Price (clean):", round(pr.clean, 4), "Accrued:", round(pr.accrued, 4))

    # Back out YTM from the clean price
    yres = adapter.ytm_from_price(spec, date.today(), pr.clean)
    print("Implied YTM:", round(100 * (yres.ytm or 0.0), 4), "%", "DV01:", round(yres.dv01 or 0.0, 5))

    # 1M carry/roll
    cr = adapter.carry_roll(spec, date.today(), horizon_days=30)
    print("Carry/Roll:", {k: round(v, 4) for k, v in cr.items()})