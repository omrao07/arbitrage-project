# backend/risk/bank_stress.py
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---- Soft imports from your project (fallback stubs keep this importable) ----
try:
    from backend.treasury.bank_adapters import ( # type: ignore
        BankAdapterBase, MockBankAdapter, BankAccount, Balance, Transaction,
        to_dicts,
    )
except Exception:
    # minimal fallbacks
    @dataclass
    class BankAccount:
        bank_name: str; account_id: str; account_number_masked: str; currency: str="USD"; meta: Dict[str,Any]=field(default_factory=dict)
    @dataclass
    class Balance:
        available: float; ledger: float; currency: str; ts: str=""
    class BankAdapterBase:
        def list_accounts(self) -> List[BankAccount]: return []
        def get_balance(self, account_id: str) -> Balance: raise NotImplementedError
        def list_transactions(self, account_id: str, *, since: Optional[str]=None, until: Optional[str]=None, limit:int=200)->List[Dict[str,Any]]: return []

try:
    from backend.treasury.soverign_adapter import YieldCurve, CurvePoint, BondSpec, SovereignAdapter, price_from_curve # type: ignore
except Exception:
    @dataclass
    class CurvePoint: tenor_yrs: float; yld: float
    @dataclass
    class YieldCurve:
        curve_date: Any; currency: str; points: List[CurvePoint]=field(default_factory=list)
        def sorted_points(self): return sorted(self.points, key=lambda p:p.tenor_yrs)
        def zero_yield(self, t: float)->float:
            pts=self.sorted_points()
            if not pts: return 0.0
            if t<=pts[0].tenor_yrs: return pts[0].yld
            if t>=pts[-1].tenor_yrs: return pts[-1].yld
            for i in range(1,len(pts)):
                l,r=pts[i-1],pts[i]
                if l.tenor_yrs<=t<=r.tenor_yrs:
                    w=(t-l.tenor_yrs)/(r.tenor_yrs-l.tenor_yrs)
                    lnD1=-l.yld*l.tenor_yrs; lnD2=-r.yld*r.tenor_yrs
                    lnDt=lnD1*(1-w)+lnD2*w
                    return -lnDt/max(t,1e-9)
            return pts[-1].yld
        def discount_factor(self, t: float)->float:
            return math.exp(-self.zero_yield(t)*t)
    @dataclass
    class BondSpec:
        issuer: str; currency: str; face: float; coupon: float; freq: int; issue_date: Any; maturity: Any; daycount: str="ACT/365"
    class SovereignAdapter: ...

    def price_from_curve(spec: BondSpec, curve: YieldCurve, asof: Any, *, clean: bool=True):
        T = max(0.001, (spec.maturity - asof).days/365.0)
        disc = curve.discount_factor(T)
        # very crude PV: coupon annuity + redemption at par
        n = max(1, int(T*spec.freq))
        cf = spec.face*spec.coupon/spec.freq
        pv_c = sum(cf * curve.discount_factor(T * (i/n)) for i in range(1, n+1))
        dirty = pv_c + spec.face*disc
        return type("PR", (), {"clean": dirty, "dirty": dirty, "accrued": 0.0})

# Optional shock types (parallel/steepen etc.)
try:
    from backend.risk.policy_sim import RateShock, FXShock # type: ignore
except Exception:
    @dataclass
    class RateShock:
        parallel_bp: float=0.0; steepen_bp: float=0.0; butterfly_bp: float=0.0; twist_pivot_yrs: float=5.0
        custom_tenor_bp: Dict[float,float]=field(default_factory=dict)
    @dataclass
    class FXShock:
        pct_by_pair: Dict[str,float]=field(default_factory=dict)

# =============================================================================
# Models
# =============================================================================

@dataclass
class SecurityHolding:
    """
    Simple holding recorded at face for bonds; haircut + MTM will be applied.
    For HQLA classification (LCR), set hqla_level = 1, 2A, 2B, or None.
    """
    symbol: str
    spec: BondSpec
    qty_face: float                # e.g., total face value held
    hqla_level: Optional[str] = "1"
    market_liquidity_discount_pct: float = 0.0  # addt'l liquidation discount

@dataclass
class DepositBucket:
    """
    Behavioral runoff assumptions for deposits.
    """
    name: str
    balance: float
    runoff_30d_pct: float  # percent outflow over 30 days
    stable: bool = False

@dataclass
class CreditLine:
    """
    Undrawn committed facilities that may get drawn in stress.
    """
    name: str
    undrawn: float
    draw_30d_pct: float

@dataclass
class StressAssumptions:
    """
    Core assumptions to run LCR-like and cash runway stress.
    """
    horizon_days: int = 30
    fx_pairs: Dict[str, float] = field(default_factory=dict)  # e.g., {"USDINR": 83.0}
    sellable_fraction_bonds: float = 0.6     # fraction of bond book you can liquidate inside horizon
    lcr_weight_level1: float = 1.0           # HQLA efficiency
    lcr_weight_level2a: float = 0.85
    lcr_weight_level2b: float = 0.5
    non_hqla_haircut_pct: float = 25.0       # non-HQLA haircut when selling
    intraday_buffer_pct: float = 5.0         # extra buffer for ops frictions
    min_operating_cash: float = 0.0          # cash you never deploy

@dataclass
class LCRResult:
    total_hqla: float
    net_outflows_30d: float
    lcr_ratio: float
    details: Dict[str, Any]

@dataclass
class LiquidityResult:
    start_cash: float
    projected_min_cash: float
    end_cash: float
    days_until_zero: Optional[int]
    notes: List[str] = field(default_factory=list)

@dataclass
class SecuritiesMTMResult:
    base_value: float
    shocked_value: float
    pnl: float
    per_security: List[Dict[str, Any]]

@dataclass
class BankStressReport:
    asof: date
    currency: str
    lcr: LCRResult
    liquidity: LiquidityResult
    bonds_mtm: SecuritiesMTMResult
    params: Dict[str, Any]

# =============================================================================
# Helpers
# =============================================================================

def _apply_rate_shock_to_curve(curve: YieldCurve, shock: Optional[RateShock]) -> YieldCurve:
    if not shock:
        return curve
    pts = curve.sorted_points()
    def bp2dec(x: float) -> float: return x/1e4
    def shift(t: float) -> float:
        s = bp2dec(shock.parallel_bp)
        if getattr(shock, "steepen_bp", 0.0):
            k = bp2dec(shock.steepen_bp)
            s += k * (1.0 if t >= getattr(shock, "twist_pivot_yrs", 5.0) else -1.0)
        if getattr(shock, "butterfly_bp", 0.0):
            k = bp2dec(shock.butterfly_bp)
            pivot = getattr(shock, "twist_pivot_yrs", 5.0)
            belly = max(0.0, 1.0 - abs(t - pivot) / max(pivot, 1e-6))
            s += -k * belly
        for kr, v in (getattr(shock, "custom_tenor_bp", {}) or {}).items():
            if abs(kr - t) < 1e-6:
                s += bp2dec(v)
        return s
    new_pts = [CurvePoint(p.tenor_yrs, max(0.0, p.yld + shift(p.tenor_yrs))) for p in pts]
    return YieldCurve(curve_date=curve.curve_date, currency=curve.currency, points=new_pts)

def _hqla_weight(level: Optional[str], sa: StressAssumptions) -> float:
    if level == "1":   return sa.lcr_weight_level1
    if level == "2A":  return sa.lcr_weight_level2a
    if level == "2B":  return sa.lcr_weight_level2b
    return 0.0

# =============================================================================
# Core calculators
# =============================================================================

def lcr_like(
    *,
    deposits: List[DepositBucket],
    credit_lines: List[CreditLine],
    bonds: List[SecurityHolding],
    curve: YieldCurve,
    asof: date,
    sa: StressAssumptions,
    rate_shock: Optional[RateShock] = None,
) -> LCRResult:
    """
    Approximate LCR:
      LHS = HQLA after effective weights/haircuts
      RHS = 30d net cash outflows = deposit runoff + CL drawdowns - (secured funding capacity ignored here)
    """
    sh_curve = _apply_rate_shock_to_curve(curve, rate_shock)

    # Value HQLA: price bonds clean and apply market liquidity discount + HQLA weight
    total_hqla = 0.0
    details: Dict[str, Any] = {"bonds": []}

    for h in bonds:
        pr = price_from_curve(h.spec, sh_curve, asof, clean=True).clean  # type: ignore # per 100 face
        fair = (pr / 100.0) * h.qty_face
        # liquidity discount
        fair_after_liq = fair * (1.0 - max(0.0, h.market_liquidity_discount_pct)/100.0)
        weight = _hqla_weight(h.hqla_level, sa)
        eff = fair_after_liq * weight
        total_hqla += eff
        details["bonds"].append({
            "symbol": h.symbol, "hqla_level": h.hqla_level, "price": pr, "face": h.qty_face,
            "fair": fair, "after_liq": fair_after_liq, "weight": weight, "hqla_value": eff
        })

    # 30d outflows: deposit runoff + CL draws
    out_dep = sum(max(0.0, d.balance) * max(0.0, d.runoff_30d_pct) / 100.0 for d in deposits)
    out_cl  = sum(max(0.0, c.undrawn) * max(0.0, c.draw_30d_pct) / 100.0 for c in credit_lines)
    net_out = out_dep + out_cl

    # add operating frictions buffer
    net_out *= (1.0 + max(0.0, sa.intraday_buffer_pct)/100.0)

    lcr = 9999.0 if net_out <= 0 else (total_hqla / net_out) * 100.0
    return LCRResult(total_hqla=total_hqla, net_outflows_30d=net_out, lcr_ratio=lcr, details=details)

def liquidity_runway(
    *,
    starting_cash: float,
    deposits: List[DepositBucket],
    credit_lines: List[CreditLine],
    bonds: List[SecurityHolding],
    curve: YieldCurve,
    asof: date,
    sa: StressAssumptions,
    rate_shock: Optional[RateShock] = None,
) -> LiquidityResult:
    """
    Simulate daily cash over the 30-day horizon:
      cash_{t+1} = cash_t - deposit_outflow(t) - cl_draw(t) + liquidation_proceeds(t)
    Liquidation proceeds are capped by sellable_fraction_bonds over the full horizon.
    """
    days = max(1, int(sa.horizon_days))
    sh_curve = _apply_rate_shock_to_curve(curve, rate_shock)

    # Total liquidatable amount (face-valued to fair, apply liquidity discount)
    total_saleable = 0.0
    for h in bonds:
        pr = price_from_curve(h.spec, sh_curve, asof, clean=True).clean # type: ignore
        fair = (pr / 100.0) * h.qty_face
        fair *= (1.0 - max(0.0, h.market_liquidity_discount_pct)/100.0)
        total_saleable += fair
    total_saleable *= max(0.0, min(1.0, sa.sellable_fraction_bonds))

    # Evenly distribute liquidation across horizon (could be front-loaded if needed)
    daily_liq = total_saleable / days if days > 0 else 0.0

    # Convert 30d runoff/draw to per-day (linear proxy)
    dep_daily = sum(max(0.0, d.balance) * max(0.0, d.runoff_30d_pct)/100.0 for d in deposits) / days
    cl_daily  = sum(max(0.0, c.undrawn) * max(0.0, c.draw_30d_pct)/100.0 for c in credit_lines) / days

    # Intraday buffer on outflows
    dep_daily *= (1.0 + max(0.0, sa.intraday_buffer_pct)/100.0)
    cl_daily  *= (1.0 + max(0.0, sa.intraday_buffer_pct)/100.0)

    cash = max(0.0, starting_cash - max(0.0, sa.min_operating_cash))
    min_cash = cash
    days_to_zero: Optional[int] = None

    for t in range(1, days+1):
        cash = cash - dep_daily - cl_daily + daily_liq
        if cash < min_cash:
            min_cash = cash
        if days_to_zero is None and cash <= 0:
            days_to_zero = t

    end_cash = cash
    notes = []
    if days_to_zero is not None:
        notes.append(f"Cash reaches zero on day {days_to_zero} of {days}.")
    else:
        notes.append("Cash stays positive across horizon.")
    return LiquidityResult(
        start_cash=starting_cash, projected_min_cash=min_cash, end_cash=end_cash, days_until_zero=days_to_zero, notes=notes
    )

def bonds_mtm(
    *,
    bonds: List[SecurityHolding],
    curve: YieldCurve,
    asof: date,
    rate_shock: Optional[RateShock] = None,
) -> SecuritiesMTMResult:
    sh_curve = _apply_rate_shock_to_curve(curve, rate_shock)
    per: List[Dict[str, Any]] = []
    base = 0.0; shocked = 0.0
    for h in bonds:
        p0 = price_from_curve(h.spec, curve, asof, clean=True).clean # type: ignore
        p1 = price_from_curve(h.spec, sh_curve, asof, clean=True).clean # type: ignore
        v0 = (p0/100.0) * h.qty_face
        v1 = (p1/100.0) * h.qty_face
        base += v0; shocked += v1
        per.append({"symbol": h.symbol, "px_base": p0, "px_shocked": p1, "value_base": v0, "value_shocked": v1, "pnl": v1 - v0})
    return SecuritiesMTMResult(base_value=base, shocked_value=shocked, pnl=shocked-base, per_security=per)

# =============================================================================
# High-level runner
# =============================================================================

def run_bank_stress(
    *,
    bank: BankAdapterBase,
    currency: str,
    deposits: List[DepositBucket],
    credit_lines: List[CreditLine],
    bonds: List[SecurityHolding],
    curve: YieldCurve,
    asof: date,
    assumptions: Optional[StressAssumptions] = None,
    rate_shock: Optional[RateShock] = None,
) -> BankStressReport:
    sa = assumptions or StressAssumptions()
    # Sum cash across bank accounts (same currency bucket)
    cash = 0.0
    for acct in bank.list_accounts():
        try:
            if acct.currency.upper() != currency.upper():
                continue
        except Exception:
            pass
        try:
            bal = bank.get_balance(acct.account_id)
            if bal.currency.upper() == currency.upper():
                cash += float(bal.available)
        except Exception:
            continue

    lcr_res = lcr_like(
        deposits=deposits, credit_lines=credit_lines, bonds=bonds,
        curve=curve, asof=asof, sa=sa, rate_shock=rate_shock
    )
    liq_res = liquidity_runway(
        starting_cash=cash, deposits=deposits, credit_lines=credit_lines, bonds=bonds,
        curve=curve, asof=asof, sa=sa, rate_shock=rate_shock
    )
    mtm_res = bonds_mtm(bonds=bonds, curve=curve, asof=asof, rate_shock=rate_shock)

    return BankStressReport(
        asof=asof,
        currency=currency.upper(),
        lcr=lcr_res,
        liquidity=liq_res,
        bonds_mtm=mtm_res,
        params={
            "assumptions": asdict(sa),
            "rate_shock": asdict(rate_shock) if rate_shock else None,
        },
    )

# =============================================================================
# Example CLI smoke test
# =============================================================================

if __name__ == "__main__":
    from datetime import date

    # Mock bank with INR cash
    try:
        from backend.treasury.bank_adapters import MockBankAdapter # type: ignore
        bank = MockBankAdapter(currency="INR", start_cash=25_00_000.0)  # ₹2.5m
    except Exception:
        class _B(BankAdapterBase): # type: ignore
            def list_accounts(self): return [BankAccount("Mock","acct1","****", "INR", {})]
            def get_balance(self, _): return Balance(available=25_00_000.0, ledger=25_00_000.0, currency="INR")
        bank = _B()

    # Simple INR curve
    inr_curve = YieldCurve(
        curve_date=date.today(), currency="INR",
        points=[CurvePoint(0.25, 0.068), CurvePoint(1.0, 0.070), CurvePoint(5.0, 0.073), CurvePoint(10.0, 0.0745)]
    )

    # GOI 5Y holding
    bspec = BondSpec(issuer="GOI5Y", currency="INR", face=100.0, coupon=0.0718, freq=2, issue_date=date(2023,8,1), maturity=date(2028,8,1))
    bonds = [
        SecurityHolding(symbol="GOI5Y", spec=bspec, qty_face=50_00_000.0, hqla_level="1", market_liquidity_discount_pct=1.0)
    ]

    deposits = [
        DepositBucket(name="Retail stable", balance=15_00_000.0, runoff_30d_pct=5.0, stable=True),
        DepositBucket(name="Uninsured corp", balance=30_00_000.0, runoff_30d_pct=25.0),
    ]

    credit_lines = [
        CreditLine(name="Undrawn WC lines", undrawn=10_00_000.0, draw_30d_pct=30.0),
    ]

    # 25 bp parallel up shock
    rshock = RateShock(parallel_bp=25.0)

    rep = run_bank_stress(
        bank=bank, currency="INR", deposits=deposits, credit_lines=credit_lines, bonds=bonds,
        curve=inr_curve, asof=date.today(), assumptions=StressAssumptions(horizon_days=30, sellable_fraction_bonds=0.6), rate_shock=rshock
    )

    import json
    print(json.dumps({
        "asof": str(rep.asof),
        "currency": rep.currency,
        "lcr": {
            "total_hqla": rep.lcr.total_hqla,
            "net_outflows_30d": rep.lcr.net_outflows_30d,
            "lcr_ratio_pct": rep.lcr.lcr_ratio,
        },
        "liquidity": asdict(rep.liquidity),
        "bonds_mtm_pnl": rep.bonds_mtm.pnl,
    }, indent=2, default=str))