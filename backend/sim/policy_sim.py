# backend/risk/policy_sim.py
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---- optional imports from your codebase (kept loose) -----------------
try:
    from backend.execution.pricer import ( # type: ignore
        Position as EqPosition,
        Quote,
        quotes_from_last,
        mark_portfolio,
        mtm_dicts,
        bs_price,
        bs_greeks,
    )
except Exception:
    # Minimal fallbacks so the module can import without your pricer
    @dataclass
    class EqPosition:
        symbol: str
        qty: float
        avg_price: float
        currency: str = "USD"

    @dataclass
    class Quote:
        symbol: str
        bid: float | None = None
        ask: float | None = None
        last: float | None = None
        def mid(self) -> Optional[float]:
            if self.bid and self.ask:
                return (self.bid + self.ask) / 2.0
            return self.last

    def quotes_from_last(d: Dict[str, float]) -> Dict[str, Quote]:
        return {k: Quote(symbol=k, last=v) for k, v in d.items()}

    def mark_portfolio(positions, quotes, **_k) -> Dict[str, Any]:
        tot_mv = 0.0; tot_cv = 0.0; items = []
        for p in positions:
            px = quotes.get(p.symbol).mid() if quotes.get(p.symbol) else None
            if not px: continue
            mv = px * p.qty
            cv = p.avg_price * p.qty
            tot_mv += mv; tot_cv += cv
            items.append(dict(symbol=p.symbol, qty=p.qty, avg_price=p.avg_price, mark=px, market_value=mv, cost_value=cv, unrealized=mv-cv, pnl_bps=((mv-cv)/max(cv,1e-9))*1e4))
        return {"items": items, "totals": {"market_value":tot_mv, "cost_value":tot_cv, "unrealized":tot_mv-tot_cv, "pnl_bps":((tot_mv-tot_cv)/max(tot_cv,1e-9))*1e4}}

    def mtm_dicts(result: Dict[str, Any]) -> Dict[str, Any]:
        return result

try:
    from backend.treasury.soverign_adapter import ( # type: ignore
        YieldCurve, CurvePoint, BondSpec, SovereignAdapter, price_from_curve,
    )
except Exception:
    # Tiny stand-in types to keep imports happy if treasury module not present
    @dataclass
    class CurvePoint:
        tenor_yrs: float
        yld: float
    @dataclass
    class YieldCurve:
        curve_date: Any
        currency: str
        label: str = "govt"
        points: List[CurvePoint] = field(default_factory=list)
        def sorted_points(self) -> List[CurvePoint]:
            return sorted(self.points, key=lambda p: p.tenor_yrs)
        def zero_yield(self, t: float) -> float:
            pts = self.sorted_points()
            if t <= pts[0].tenor_yrs: return pts[0].yld
            if t >= pts[-1].tenor_yrs: return pts[-1].yld
            for i in range(1, len(pts)):
                l, r = pts[i-1], pts[i]
                if l.tenor_yrs <= t <= r.tenor_yrs:
                    w = (t - l.tenor_yrs)/(r.tenor_yrs-l.tenor_yrs)
                    lnD1 = -l.yld * l.tenor_yrs; lnD2 = -r.yld * r.tenor_yrs
                    lnDt = lnD1*(1-w) + lnD2*w
                    return -lnDt/max(t,1e-9)
            return pts[-1].yld
        def discount_factor(self, t: float) -> float:
            return math.exp(-self.zero_yield(t)*t)
    @dataclass
    class BondSpec:
        issuer: str; currency: str; face: float; coupon: float; freq: int; issue_date: Any; maturity: Any; daycount: str="ACT/365"
    class SovereignAdapter:
        def __init__(self): self._c = {}
        def set_curve(self, c): self._c[(c.currency, c.curve_date)] = c
        def get_curve(self, currency, curve_date): return self._c[(currency, curve_date)]
    def price_from_curve(spec, curve, asof, clean=True):
        # primitive: discount level (no accrual nuance)
        # assume equal coupon intervals; good enough for delta comparison
        from datetime import datetime
        def yf(a,b): return (b-a).days/365.0
        if hasattr(asof,"date"): asof = asof
        cds = []  # not building full schedule here (sim is delta-oriented)
        T = max(0.001, (spec.maturity - asof).days/365.0)
        pv_red = spec.face * curve.discount_factor(T)
        # crude coupon PV: level annuity with duration T and freq F
        n = max(1, int(T*spec.freq))
        cf = spec.face*spec.coupon/spec.freq
        pv_c = 0.0
        for i in range(1, n+1):
            t = T * (i/n)
            pv_c += cf * curve.discount_factor(t)
        dirty = pv_c + pv_red
        return type("PR", (), {"clean": dirty, "dirty": dirty, "accrued": 0.0})

# ============================ Scenario Models ============================

@dataclass
class RateShock:
    """BP shifts (0.0001 = 1bp). Positive bp increases yields."""
    parallel_bp: float = 0.0                # e.g., +50 -> 50 bps up across curve
    steepen_bp: float = 0.0                 # + steepens (long up / short down)
    twist_pivot_yrs: float = 5.0            # pivot for steepen (yrs)
    butterfly_bp: float = 0.0               # + moves wings vs belly

    custom_tenor_bp: Dict[float, float] = field(default_factory=dict)  # {tenor_yrs: bp}

@dataclass
class EquityShock:
    """Percent returns applied to prices: +5.0 = +5%."""
    pct_by_symbol: Dict[str, float] = field(default_factory=dict)
    default_pct: float = 0.0

@dataclass
class FXShock:
    """
    Percent move for FX pairs keyed as 'USDINR', 'EURUSD' etc.
    Convention: +% means the *pair* rises (base strengthens vs quote).
    """
    pct_by_pair: Dict[str, float] = field(default_factory=dict)

@dataclass
class VolShock:
    """Absolute vol points (e.g., +3.0 adds 3 vol points = +0.03 sigma)."""
    vol_pts_by_symbol: Dict[str, float] = field(default_factory=dict)

@dataclass
class PolicyShock:
    name: str
    rates: Optional[RateShock] = None
    equities: Optional[EquityShock] = None
    fx: Optional[FXShock] = None
    vol: Optional[VolShock] = None
    notes: List[str] = field(default_factory=list)

# ============================ Bond Holdings ============================

@dataclass
class BondHolding:
    spec: BondSpec
    qty_face: float  # in face units (e.g., 100.0 face per bond * number of bonds)

# ============================ Helpers ============================

def _apply_rate_shock(curve: YieldCurve, shock: RateShock) -> YieldCurve:
    """
    Returns a *new* curve with shocked zero yields (log-linear DF interp preserved).
    """
    pts = curve.sorted_points()
    if not pts:
        return curve

    def bp_to_dec(bp: float) -> float:
        return float(bp) / 1e4

    # compute shift per tenor
    def shift_for_t(t: float) -> float:
        s = bp_to_dec(shock.parallel_bp)
        # steepener: below pivot go -x, above pivot go +x (scaled by distance)
        if shock.steepen_bp:
            k = bp_to_dec(shock.steepen_bp)
            s += k * (1.0 if t >= shock.twist_pivot_yrs else -1.0)
        # butterfly: wings up, belly down (simple triangular around pivot)
        if shock.butterfly_bp:
            k = bp_to_dec(shock.butterfly_bp)
            # belly around pivot, reduce at extremes
            belly = max(0.0, 1.0 - abs(t - shock.twist_pivot_yrs) / max(shock.twist_pivot_yrs, 1e-6))
            s += -k * belly  # down in belly
        # custom points override/add
        if shock.custom_tenor_bp:
            # nearest tenor add
            nearest = min(shock.custom_tenor_bp.keys(), key=lambda x: abs(x - t))
            s += bp_to_dec(shock.custom_tenor_bp[nearest])
        return s

    new_pts = [CurvePoint(p.tenor_yrs, max(0.0, p.yld + shift_for_t(p.tenor_yrs))) for p in pts]
    return YieldCurve(curve_date=curve.curve_date, currency=curve.currency, label=curve.label, points=new_pts)

def _shock_equity_quotes(quotes: Dict[str, Quote], eq_shock: EquityShock) -> Dict[str, Quote]:
    out: Dict[str, Quote] = {}
    for sym, q in quotes.items():
        pct = eq_shock.pct_by_symbol.get(sym, eq_shock.default_pct)
        mult = 1.0 + pct / 100.0
        last = q.mid() or q.last or 0.0
        new_last = last * mult
        out[sym] = Quote(symbol=sym, last=new_last)
    # also include symbols present only in pct_by_symbol
    for sym, pct in eq_shock.pct_by_symbol.items():
        if sym not in out:
            out[sym] = Quote(symbol=sym, last=(1.0 + pct / 100.0))  # synthetic base 1.0
    return out

def _shock_fx_table(fx_rates: Dict[str, float], fx_shock: FXShock) -> Dict[str, float]:
    out = dict(fx_rates)
    for pair, pct in fx_shock.pct_by_pair.items():
        if pair in out and out[pair] > 0:
            out[pair] = out[pair] * (1.0 + pct / 100.0)
    return out

# ============================ Simulator ============================

@dataclass
class SimResult:
    shock_name: str
    notes: List[str]
    equities: Dict[str, Any]
    bonds: Dict[str, Any]
    totals: Dict[str, float]

class PolicySimulator:
    """
    Runs policy/market scenarios and returns MTM deltas for:
      • Equities (uses pricer.mark_portfolio)
      • Bonds (reprice with shocked YieldCurve via SovereignAdapter)
      • FX (only affects reporting currency via your own FX app if you pass it)
      • Options (optional: via vol shifts & bs_price — left as extension hook)
    """

    def __init__(self, *, sov: Optional[SovereignAdapter] = None):
        self.sov = sov or SovereignAdapter()

    # ---------- Equities ----------

    def simulate_equities(
        self,
        positions: List[EqPosition],
        base_quotes: Dict[str, Quote],
        shock: Optional[EquityShock],
        *,
        base_ccy: str = "USD",
        fx_rates: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Returns {'base':..., 'shocked':..., 'delta':...} with mtm_dicts."""
        base_res = mtm_dicts(mark_portfolio(positions, base_quotes, base_ccy=base_ccy, fx_rates=fx_rates))
        if not shock:
            return {"base": base_res, "shocked": base_res, "delta": {"unrealized": 0.0, "pnl_bps": 0.0}}

        shock_quotes = _shock_equity_quotes(base_quotes, shock)
        sh_res = mtm_dicts(mark_portfolio(positions, shock_quotes, base_ccy=base_ccy, fx_rates=fx_rates))
        delta_unreal = float(sh_res["totals"]["unrealized"] - base_res["totals"]["unrealized"])
        delta_bps = float(sh_res["totals"]["pnl_bps"] - base_res["totals"]["pnl_bps"])
        return {"base": base_res, "shocked": sh_res, "delta": {"unrealized": delta_unreal, "pnl_bps": delta_bps}}

    # ---------- Bonds (sovereign) ----------

    def simulate_bonds(
        self,
        holdings: List[BondHolding],
        curve: YieldCurve,
        rate_shock: Optional[RateShock],
        *,
        asof,  # date
    ) -> Dict[str, Any]:
        """
        Returns dict with per-bond price changes and total PnL (face * price_delta/100).
        """
        base_curve = curve
        sh_curve = _apply_rate_shock(base_curve, rate_shock) if rate_shock else base_curve

        items = []
        total_pnl = 0.0
        for h in holdings:
            pr0 = price_from_curve(h.spec, base_curve, asof, clean=True).clean # type: ignore
            pr1 = price_from_curve(h.spec, sh_curve, asof, clean=True).clean # type: ignore
            dpx = pr1 - pr0  # per 100 face
            pnl = (dpx / 100.0) * float(h.qty_face)
            total_pnl += pnl
            items.append({
                "issuer": h.spec.issuer,
                "symbol": f"{h.spec.issuer}-{h.spec.maturity}",
                "face_qty": h.qty_face,
                "px_base": pr0,
                "px_shocked": pr1,
                "dpx": dpx,
                "pnl": pnl,
            })

        return {
            "curve_base": {"points": [asdict(p) for p in base_curve.sorted_points()]},
            "curve_shocked": {"points": [asdict(p) for p in sh_curve.sorted_points()]},
            "items": items,
            "totals": {"pnl": total_pnl},
        }

    # ---------- Whole book ----------

    def run(
        self,
        *,
        shock: PolicyShock,
        eq_positions: List[EqPosition],
        eq_last_prices: Dict[str, float],
        bond_holdings: List[BondHolding],
        base_curve: YieldCurve,
        asof,
        base_ccy: str = "USD",
        fx_rates: Optional[Dict[str, float]] = None,
    ) -> SimResult:
        quotes = quotes_from_last(eq_last_prices)

        # FX shock (if given) -> adjust fx table passed to MTM
        fx_after = _shock_fx_table(fx_rates or {}, shock.fx) if shock.fx else (fx_rates or {})

        eq_res = self.simulate_equities(
            positions=eq_positions,
            base_quotes=quotes,
            shock=shock.equities,
            base_ccy=base_ccy,
            fx_rates=fx_after,
        )
        bond_res = self.simulate_bonds(
            holdings=bond_holdings,
            curve=base_curve,
            rate_shock=shock.rates,
            asof=asof,
        )

        totals = {
            "equities_unrealized_delta": float(eq_res["delta"]["unrealized"]),
            "bonds_pnl": float(bond_res["totals"]["pnl"]),
            "book_delta": float(eq_res["delta"]["unrealized"] + bond_res["totals"]["pnl"]),
        }
        return SimResult(
            shock_name=shock.name,
            notes=list(shock.notes),
            equities=eq_res,
            bonds=bond_res,
            totals=totals,
        )

# ============================ Preset Scenarios ============================

def preset_scenarios() -> Dict[str, PolicyShock]:
    """
    Handy presets you can tweak/extend.
    """
    return {
        "fed_hawkish_50bp": PolicyShock(
            name="Fed +50bp hawkish",
            rates=RateShock(parallel_bp=50.0, steepen_bp=10.0, twist_pivot_yrs=5.0),
            equities=EquityShock(default_pct=-3.5),
            notes=["Parallel +50bp; slight bear steepener; equities -3.5%"],
        ),
        "rbi_dovish_25bp": PolicyShock(
            name="RBI -25bp dovish",
            rates=RateShock(parallel_bp=-25.0, steepen_bp=-5.0, twist_pivot_yrs=4.0),
            equities=EquityShock(default_pct=1.2),
            fx=FXShock(pct_by_pair={"USDINR": -0.8}),  # INR strengthens (USDINR down 0.8%)
            notes=["Parallel -25bp; mild bull steepener; INR +0.8% vs USD; equities +1.2%"],
        ),
        "oil_spike_riskoff": PolicyShock(
            name="Oil spike risk-off",
            rates=RateShock(steepen_bp=20.0, butterfly_bp=10.0, twist_pivot_yrs=7.0),
            equities=EquityShock(default_pct=-2.0),
            fx=FXShock(pct_by_pair={"USDINR": 1.0}),   # USD up 1% vs INR
            notes=["Curve bear steepens and wings up; equities -2%; USD +1% vs INR"],
        ),
    }

# ============================ CLI Example (optional) ============================

if __name__ == "__main__":
    from datetime import date

    # Tiny demo book
    eq_positions = [
        EqPosition(symbol="RELIANCE.NS", qty=100, avg_price=2900.0, currency="INR"),
        EqPosition(symbol="AAPL", qty=10, avg_price=180.0, currency="USD"),
    ]
    eq_last = {"RELIANCE.NS": 3000.0, "AAPL": 192.0}
    fx = {"USDINR": 83.0}

    # Simple INR curve
    inr_curve = YieldCurve(
        curve_date=date.today(), currency="INR",
        points=[CurvePoint(0.25, 0.068), CurvePoint(1.0, 0.070), CurvePoint(5.0, 0.073), CurvePoint(10.0, 0.0745)]
    )

    # One 5Y GOI holding (face 1,000,000)
    bh = BondHolding(
        spec=BondSpec(issuer="GOI5Y", currency="INR", face=100.0, coupon=0.0718, freq=2,
                      issue_date=date(2023,8,1), maturity=date(2028,8,1)),
        qty_face=1_000_000.0
    )

    sim = PolicySimulator()
    shock = preset_scenarios()["rbi_dovish_25bp"]
    res = sim.run(
        shock=shock,
        eq_positions=eq_positions,
        eq_last_prices=eq_last,
        bond_holdings=[bh],
        base_curve=inr_curve,
        asof=date.today(),
        base_ccy="USD",
        fx_rates=fx,
    )

    import json
    print(json.dumps({
        "shock": res.shock_name,
        "totals": res.totals,
        "eq_delta": res.equities["delta"],
        "bond_totals": res.bonds["totals"],
    }, indent=2))