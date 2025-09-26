# engines/structured_credit/clo_tranches.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import pandas as pd

MONTHS = 12

# =============================================================================
# Tranche & deal configuration
# =============================================================================

@dataclass
class Tranche:
    name: str
    notional: float                  # initial tranche notional
    spread_bps: float                # over ref rate (per annum)
    seniority: int                   # 1 = most senior
    # Optional OC diversion guardrails (simple)
    oc_trigger: Optional[float] = None  # if OC < trigger, divert residual to paydown
    oc_weight: float = 1.0              # OC contribution weight (usually 1.0)
    # Results placeholders (filled after run)
    cashflows: Optional[pd.DataFrame] = None

@dataclass
class PoolConfig:
    collateral_notional: float
    asset_spread_bps: float             # over ref rate (per annum)
    ref_curve: Callable[[int], float]   # monthly index rate function (m -> decimal per annum)
    deal_term_years: float = 8.0        # legal final (years)
    reinvest_years: float = 4.0         # reinvestment period (years)
    cdr_annual: float = 0.04            # constant default rate (annual, dec)
    cpr_annual: float = 0.06            # constant prepayment rate (annual, dec)
    recovery_rate: float = 0.4
    recovery_lag_months: int = 3
    mgmt_fee_bps: float = 40.0          # on performing collateral (per annum)
    senior_fee_bps: float = 15.0        # trustee/admin (per annum)
    oc_target: float = 1.0              # OC ratio target (asset/tranche princ)
    use_oc_diversion: bool = True

@dataclass
class PricingConfig:
    disc_curve: Callable[[int], float]  # monthly discount rate function (m -> decimal per annum)
    price_to: str = "clean"             # 'clean' or 'dirty' (dirty includes accrued)
    day_count: str = "ACT/360"

# =============================================================================
# Helpers
# =============================================================================

def ann_to_month(q: float) -> float:
    # convert annualized continuous-ish rate to monthly simple prob/ratio
    return 1 - (1 - q) ** (1 / MONTHS)

def accrual_factor(day_count: str = "ACT/360") -> float:
    return 1.0 / 12.0 if day_count.upper() in ("30/360", "30E/360", "ACT/360") else 1.0 / 12.0

def pay_rate_annual(ref: float, spread_bps: float) -> float:
    return ref + spread_bps * 1e-4

def npv_from_leg(df: pd.DataFrame, disc_curve: Callable[[int], float], price_to: str = "clean") -> float:
    """Discount monthly cashflows using per-period annual rate from curve(m)."""
    pv = 0.0
    for i, (t, row) in enumerate(df.iterrows(), start=1):
        r = disc_curve(i)  # annual dec
        mfac = 1.0 / (1.0 + r / 12.0) ** i
        amt = float(row["interest"] + row["principal"])
        if price_to == "dirty":
            amt += float(row.get("accrual", 0.0))
        pv += amt * mfac
    return pv

# =============================================================================
# Engine
# =============================================================================

class CLOEngine:
    def __init__(self, pool: PoolConfig, tranches: List[Tranche], pricing: PricingConfig):
        # Sort tranches by seniority (1 = most senior)
        self.tranches = sorted(tranches, key=lambda x: x.seniority)
        self.pool = pool
        self.pricing = pricing

        # State
        self.months = int(round(pool.deal_term_years * MONTHS))
        self.reinv_end = int(round(pool.reinvest_years * MONTHS))
        self.af = accrual_factor(pricing.day_count)

        # Probabilities
        self.cdr_m = ann_to_month(pool.cdr_annual)
        self.cpr_m = ann_to_month(pool.cpr_annual)

        # Schedules
        self.timeline = pd.date_range("2000-01-31", periods=self.months, freq="M")

    def run(self, seed: Optional[int] = 7) -> Dict[str, pd.DataFrame]:
        rng = np.random.default_rng(seed)

        # Collateral state
        coll_bal = self.pool.collateral_notional
        performing = coll_bal
        default_queue: List[Tuple[int, float]] = []  # (pay_month, recovery$)

        # Tranche states
        tr_bal = np.array([t.notional for t in self.tranches], dtype=float)

        # Result collectors
        deal_rows = []
        tr_cf: List[List[Dict]] = [[] for _ in self.tranches]

        for m, dt in enumerate(self.timeline, start=1):
            # ----- Collateral dynamics -----
            ref = self.pool.ref_curve(m)                # ref rate (annual)
            asset_coupon = pay_rate_annual(ref, self.pool.asset_spread_bps)
            mgmt_fee = self.pool.mgmt_fee_bps * 1e-4
            senior_fee = self.pool.senior_fee_bps * 1e-4

            # Defaults on performing collateral
            defaults = performing * self.cdr_m
            recover_pay_m = m + max(1, self.pool.recovery_lag_months)
            recovery = defaults * self.pool.recovery_rate
            if defaults > 0:
                default_queue.append((recover_pay_m, recovery))
            performing -= defaults

            # Prepayments (on surviving performing only)
            prepays = performing * self.cpr_m
            performing -= prepays

            # Interest income from performing collateral (pre-fees)
            int_coll = performing * asset_coupon * self.af
            # Reinvestment in reinvest window: keep collateral balance roughly flat by adding prepays back
            if m <= self.reinv_end:
                performing += prepays
                int_coll  # type: ignore # interest unaffected (assume reinvest at start next month)

            # Recoveries arriving this month
            rec_this = 0.0
            if default_queue and default_queue[0][0] == m:
                rec_this = default_queue[0][1]
                performing += rec_this
                default_queue.pop(0)

            # Fees (on performing)
            fees = performing * (mgmt_fee + senior_fee) * self.af

            # Available interest & principal
            avail_int= max(int_coll - fees, 0.0)
            avail_prin = prepays + rec_this

            # ----- Coverage & diversion (simple OC test) -----
            total_tranche_prin = float(tr_bal.sum())
            oc_ratio = (performing / max(total_tranche_prin, 1e-9)) if total_tranche_prin > 0 else np.inf
            divert = self.pool.use_oc_diversion and (oc_ratio < self.pool.oc_target)

            # ----- Waterfall: Interest then Principal (sequential pay) -----
            tr_int_due = []
            for i, t in enumerate(self.tranches):
                coup = pay_rate_annual(ref, t.spread_bps)
                due = tr_bal[i] * coup * self.af
                tr_int_due.append(due)

            int_paid = np.zeros(len(self.tranches))
            rem_int = avail_int
            for i in range(len(self.tranches)):
                pay = min(rem_int, tr_int_due[i])
                int_paid[i] = pay
                rem_int -= pay

            # Any unpaid interest is not deferred here (could add PIK logic)
            residual = rem_int

            # Principal waterfall: sequential amortization
            prin_paid = np.zeros(len(self.tranches))
            rem_prin = avail_prin + (residual if divert else 0.0)  # divert residual interest to principal if OC breached
            for i in range(len(self.tranches)):
                if tr_bal[i] <= 0 or rem_prin <= 0:
                    continue
                pay = min(rem_prin, tr_bal[i])
                prin_paid[i] = pay
                tr_bal[i] -= pay
                rem_prin -= pay

            # Record tranche legs
            for i, t in enumerate(self.tranches):
                tr_cf[i].append({
                    "date": dt,
                    "balance": tr_bal[i],
                    "interest": float(int_paid[i]),
                    "principal": float(prin_paid[i]),
                    "oc_ratio": float(oc_ratio),
                })

            # Deal wide
            deal_rows.append({
                "date": dt,
                "performing_collateral$": performing,
                "int_income$": int_coll,
                "fees$": fees,
                "avail_int$": avail_int,
                "avail_prin$": avail_prin,
                "oc_ratio": oc_ratio,
                "divert": divert,
            })

        # Assemble DataFrames and compute measures
        deal_df = pd.DataFrame(deal_rows).set_index("date")
        for i, t in enumerate(self.tranches):
            leg = pd.DataFrame(tr_cf[i]).set_index("date")
            t.cashflows = leg

        return {"deal": deal_df, "tranches": {t.name: t.cashflows.copy() for t in self.tranches}} # type: ignore

    # ---------- Metrics ----------
    def price_tranche(self, t: Tranche) -> Dict[str, float]:
        if t.cashflows is None or t.cashflows.empty:
            raise ValueError("Run the engine first (t.cashflows is empty).")
        pv = npv_from_leg(t.cashflows, self.pricing.disc_curve, self.pricing.price_to)
        irr = self._xirr_monthly(t, t.cashflows)
        wal = self._wal_months(t.cashflows) / MONTHS
        tot_int = float(t.cashflows["interest"].sum())
        tot_prin = float(t.cashflows["principal"].sum())
        loss = max(0.0, t.notional - tot_prin)
        return {
            "PV$": pv,
            "IRR_ann": irr,
            "WAL_years": wal,
            "TotalInt$": tot_int,
            "ReturnedPrin$": tot_prin,
            "PrincipalLoss$": loss,
        }

    @staticmethod
    def _wal_months(cf: pd.DataFrame) -> float:
        weights = cf["principal"].values
        months = np.arange(1, len(cf) + 1, dtype=float)
        denom = weights.sum() if weights.sum() > 0 else 1.0 # type: ignore
        return float((weights * months).sum() / denom)

    def _xirr_monthly(self, t: Tranche, cf: pd.DataFrame) -> float:
        # Construct cashflow vector: initial -notional then monthly receipts
        c = [-t.notional] + (cf["interest"] + cf["principal"]).tolist()
        if all(abs(x) < 1e-9 for x in c[1:]):
            return 0.0
        # Newton method on monthly IRR
        def npv(rate_m):
            return sum(c[k] / (1 + rate_m) ** k for k in range(len(c)))
        def d_npv(rate_m):
            return sum(-k * c[k] / (1 + rate_m) ** (k + 1) for k in range(1, len(c)))
        r = 0.005  # 0.5% per month initial guess
        for _ in range(50):
            f = npv(r); df = d_npv(r)
            if abs(df) < 1e-12: break
            r2 = r - f / df
            if abs(r2 - r) < 1e-8: r = r2; break
            r = max(-0.95, r2)
        ann = (1 + r) ** 12 - 1
        return float(ann)

# =============================================================================
# Example usage (synthetic)
# =============================================================================

if __name__ == "__main__":
    # Curves: flat 3m SOFR 5% & flat discount 7% (annual decimals)
    ref_curve = lambda m: 0.05
    disc_curve = lambda m: 0.07

    pool = PoolConfig(
        collateral_notional=400_000_000,
        asset_spread_bps=350,
        ref_curve=ref_curve,
        deal_term_years=8.0,
        reinvest_years=4.0,
        cdr_annual=0.04,
        cpr_annual=0.08,
        recovery_rate=0.40,
        recovery_lag_months=3,
        mgmt_fee_bps=40,
        senior_fee_bps=15,
        oc_target=1.0,
        use_oc_diversion=True,
    )

    tranches = [
        Tranche("AAA", 240_000_000, 120, seniority=1, oc_trigger=1.0),
        Tranche("AA",   60_000_000, 180, seniority=2, oc_trigger=1.0),
        Tranche("A",    40_000_000, 240, seniority=3, oc_trigger=1.0),
        Tranche("BBB",  30_000_000, 350, seniority=4, oc_trigger=1.0),
        Tranche("BB",   15_000_000, 600, seniority=5, oc_trigger=1.0),
        Tranche("Equity", 15_000_000, 0, seniority=6, oc_trigger=None),
    ]

    pricing = PricingConfig(disc_curve=disc_curve, price_to="clean")

    eng = CLOEngine(pool, tranches, pricing)
    out = eng.run(seed=42)

    # Print tranche metrics
    for t in eng.tranches:
        metrics = eng.price_tranche(t)
        print(f"{t.name}: {metrics}")

    # Deal-level snapshot
    print(out["deal"].tail(3))  # last 3 months of deal cashflows