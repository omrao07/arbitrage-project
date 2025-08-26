# backend/risk/contagian_graph.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Optional NumPy for faster linear ops; falls back to pure Python if absent.
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None  # type: ignore


# =============================================================================
# Data model
# =============================================================================

@dataclass
class Bank:
    """
    Simple bank balance sheet (single currency).
    - equity = assets - liabilities (updated as losses realized)
    - liquid_assets: cash + HQLA that can be sold with haircuts
    - illiquid_assets: loans, HTM, etc. subject to fire-sale discounts
    - liabilities: total outside liabilities (ex interbank payables)
    - interbank_assets/payables are tracked on the graph edges.
    """
    id: str
    name: str = ""
    equity: float = 0.0
    liquid_assets: float = 0.0
    illiquid_assets: float = 0.0
    liabilities: float = 0.0
    risk_weight: float = 0.5      # for simple capital ratio calc
    defaulted: bool = False
    resolution_cost: float = 0.0  # fixed cost when defaulting (legal/admin)

    def capital_ratio(self) -> float:
        rwa = self.risk_weight * max(0.0, self.liquid_assets + self.illiquid_assets)
        return 0.0 if rwa <= 0 else max(0.0, self.equity) / rwa


@dataclass
class Exposure:
    """
    Directed interbank exposure: lender -> borrower (who owes to lender).
    amount: gross exposure before recovery.
    """
    lender: str
    borrower: str
    amount: float
    recovery_rate: float = 0.4     # LGD = 1 - recovery_rate


@dataclass
class ShockParams:
    """
    Global knobs for propagation & fire-sale dynamics.
    """
    # Payment clearing (Eisenberg–Noe style)
    max_iter: int = 200
    tol: float = 1e-8

    # Fire-sale price impact: Δp/p ≈ -k * (sold / market_depth)
    firesale_k: float = 0.15
    market_depth: float = 1e9

    # Liquidity spiral (haircuts widen as prices drop): haircut = base + alpha * drawdown
    base_haircut: float = 0.05
    haircut_alpha: float = 0.8

    # Resolution (default) parameters
    default_threshold: float = 0.0     # default when equity <= threshold
    min_capital_ratio: float = 0.04    # non-default breach triggers deleveraging

    # Dampening on iterative rounds
    round_cap: int = 10


# =============================================================================
# Graph engine
# =============================================================================

class ContagionGraph:
    """
    Interbank network with:
      • Eisenberg–Noe clearing of interbank payments
      • Default cascade & recoveries
      • Fire-sale liquidations with market impact
      • Liquidity spiral via haircut widening
    """
    def __init__(self, params: Optional[ShockParams] = None):
        self.params = params or ShockParams()
        self.banks: Dict[str, Bank] = {}
        self.out_edges: Dict[str, List[Exposure]] = {}  # by lender
        self.in_edges: Dict[str, List[Exposure]] = {}   # by borrower

    # ----------------------- Construction ---------------------------------

    def add_bank(self, bank: Bank) -> None:
        self.banks[bank.id] = bank

    def add_exposure(self, lender: str, borrower: str, amount: float, recovery_rate: float = 0.4) -> None:
        e = Exposure(lender=lender, borrower=borrower, amount=float(amount), recovery_rate=float(recovery_rate))
        self.out_edges.setdefault(lender, []).append(e)
        self.in_edges.setdefault(borrower, []).append(e)

    # ----------------------- Utilities ------------------------------------

    def _exposure_matrix(self) -> Tuple[List[str], List[List[float]], List[float]]:
        """
        Returns (ids, L, e) where:
          L[i][j] = liability of i to j  (i owes j), i.e., exposure j->i
          e[i] = external assets of i (excluding interbank assets)
        """
        ids = sorted(self.banks.keys())
        n = len(ids)
        idx = {bid: i for i, bid in enumerate(ids)}

        # Build L (who owes whom): i owes j the amount that j lent to i
        L = [[0.0 for _ in range(n)] for __ in range(n)]
        for b in ids:
            for e in self.in_edges.get(b, []):  # exposures where b is borrower (owes)
                i = idx[e.borrower]
                j = idx[e.lender]
                L[i][j] += e.amount

        # External assets (non-interbank)
        e = []
        for bid in ids:
            bk = self.banks[bid]
            e.append(max(0.0, bk.liquid_assets + bk.illiquid_assets - bk.liabilities))
        return ids, L, e

    def _total_owed(self, L: List[List[float]]) -> List[float]:
        return [sum(row) for row in L]

    # ----------------------- Clearing (Eisenberg–Noe) ----------------------

    def clearing_vector(self) -> Dict[str, float]:
        """
        Compute the fixed point of payments p satisfying:
          0 <= p_i <= \bar{p}_i
          p_i = min( \bar{p}_i, e_i + sum_j P_{ji} p_j )
        where P is the matrix of liability shares.
        Returns dict bank_id -> paid amount (to all creditors in aggregate).
        """
        ids, L, ext = self._exposure_matrix()
        n = len(ids)
        barp = self._total_owed(L)
        if n == 0:
            return {}

        # Build P (shares) with row i normalized liabilities
        P = [[0.0]*n for _ in range(n)]
        for i in range(n):
            tot = barp[i]
            if tot > 0:
                for j in range(n):
                    P[i][j] = L[i][j] / tot

        # Iterate to fixed point
        p = [min(barp[i], ext[i]) for i in range(n)]  # init: pay what ext assets can cover
        for _ in range(self.params.max_iter):
            # rhs = min(barp, ext + P^T p)
            rhs = [ext[i] for i in range(n)]
            # add interbank receipts: sum_j P_{j,i} p_j
            for j in range(n):
                pj = p[j]
                if pj <= 0:
                    continue
                row = P[j]
                for i in range(n):
                    rhs[i] += row[i] * pj
            newp = [min(barp[i], max(0.0, rhs[i])) for i in range(n)]
            # convergence check
            diff = sum(abs(newp[i] - p[i]) for i in range(n))
            p = newp
            if diff < self.params.tol:
                break

        return {ids[i]: p[i] for i in range(n)}

    # ----------------------- Default cascade --------------------------------

    def apply_payments(self, payments: Dict[str, float]) -> None:
        """
        Given clearing payments (total each bank pays), allocate to lenders
        proportionally and update lenders' interbank asset value → equity.
        """
        # Compute each borrower's proportions to each lender
        ids, L, _ext = self._exposure_matrix()
        idx = {bid: i for i, bid in enumerate(ids)}
        barp = self._total_owed(L)

        # How much each lender receives from each borrower
        recv_by_lender: Dict[str, float] = {bid: 0.0 for bid in ids}
        for borrower in ids:
            i = idx[borrower]
            pay_i = payments.get(borrower, 0.0)
            owed_i = barp[i]
            if owed_i <= 0 or pay_i <= 0:
                continue
            for lender in ids:
                j = idx[lender]
                if L[i][j] <= 0:
                    continue
                share = L[i][j] / owed_i
                recv = share * pay_i
                recv_by_lender[lender] += recv

        # Update equity: lenders recognize received interbank cash; borrowers reduce assets by paid cash.
        for bid, amt in recv_by_lender.items():
            self.banks[bid].equity += amt  # cash in increases equity

        for borrower, amt in payments.items():
            self.banks[borrower].equity -= amt  # paying out reduces equity

        # Any unpaid portion becomes loss to lenders (LGD)
        for borrower in ids:
            i = idx[borrower]
            paid = payments.get(borrower, 0.0)
            owed = barp[i]
            shortfall = max(0.0, owed - paid)
            if shortfall <= 0:
                continue
            # allocate loss to lenders pro-rata, net of recovery
            for e in self.in_edges.get(borrower, []):
                lender = self.banks[e.lender]
                if owed > 0 and e.amount > 0:
                    share = e.amount / owed
                    gross_loss = share * shortfall
                    loss = gross_loss * (1.0 - e.recovery_rate)
                    lender.equity -= loss

    def mark_defaults(self) -> List[str]:
        """
        Mark banks with equity <= default_threshold as defaulted and apply resolution cost.
        Returns list of newly defaulted IDs.
        """
        newly = []
        th = self.params.default_threshold
        for b in self.banks.values():
            if not b.defaulted and b.equity <= th:
                b.defaulted = True
                b.equity -= b.resolution_cost
                newly.append(b.id)
        return newly

    # ----------------------- Fire-sale mechanics -----------------------------

    def fire_sale_round(self, stress_ids: List[str], price_drawdown: float = 0.0) -> float:
        """
        Banks undercapitalized/non-defaulted sell illiquid assets to lift capital ratios.
        Returns aggregate notional sold (for price impact).
        """
        sold_total = 0.0
        for bid in stress_ids:
            b = self.banks[bid]
            if b.defaulted:
                continue
            # target to restore min capital ratio
            min_cr = self.params.min_capital_ratio
            cr = b.capital_ratio()
            if cr >= min_cr:
                continue

            # haircut widens with drawdown
            haircut = self.params.base_haircut + self.params.haircut_alpha * max(0.0, price_drawdown)
            haircut = max(0.0, min(0.95, haircut))

            # compute required equity increase: ΔE ≈ target*RWA - current E
            rwa = b.risk_weight * max(0.0, b.liquid_assets + b.illiquid_assets)
            target_e = min_cr * rwa
            need_e = max(0.0, target_e - max(0.0, b.equity))

            if need_e <= 0:
                continue

            # selling S notional raises cash S*(1-haircut), but reduces assets S
            # equity change = (cash_in - asset_reduction) = S*(1-h) - S = -S*h  (i.e., selling actually lowers equity!)
            # Practically, banks sell to raise LIQUIDITY to meet outflows; here we lift ratio by reducing RWA.
            # We approximate: sell S so that equity / (rwa - rw*S) >= min_cr
            rw = b.risk_weight
            max_sell = max(0.0, b.illiquid_assets)
            if rw > 0 and max_sell > 0:
                # Solve for S: E / (rw*(A - S)) = min_cr  => S = A - E/(rw*min_cr)
                A = b.liquid_assets + b.illiquid_assets
                S_needed = max(0.0, A - (max(0.0, b.equity) / (rw * min_cr)))
                S = min(max_sell, S_needed)
            else:
                S = 0.0

            if S <= 0:
                continue

            # Execute sale
            b.illiquid_assets -= S
            b.liquid_assets += S * (1.0 - haircut)
            # equity impact immediate: mark-to-market of sold block at haircut
            b.equity -= S * haircut
            sold_total += S

        return sold_total

    def price_impact(self, sold_notional: float) -> float:
        """
        Returns price drawdown (fraction) from aggregate sales.
        Δp/p ≈ -k * (sold / market_depth)
        """
        if sold_notional <= 0 or self.params.market_depth <= 0:
            return 0.0
        dd = - self.params.firesale_k * (sold_notional / self.params.market_depth)
        return dd

    # ----------------------- End-to-end propagation -------------------------

    def propagate(self, *, rounds: Optional[int] = None) -> Dict[str, any]: # type: ignore
        """
        Orchestrates:
          1) Compute clearing payments → apply → defaults
          2) Fire-sale from undercapitalized survivors → price impact → MTM losses
          3) Repeat until convergence or max rounds.
        Returns a summary dict.
        """
        max_r = rounds or self.params.round_cap
        price_dd_cum = 0.0

        history = []

        for r in range(max_r):
            step = {"round": r+1}

            # 1) Clearing
            pvec = self.clearing_vector()
            self.apply_payments(pvec)
            newly = self.mark_defaults()
            step["clearing_paid"] = pvec # type: ignore
            step["new_defaults"] = newly # type: ignore

            # 2) Fire-sale among stressed survivors (non-defaulted with CR < min)
            stressed = [bid for bid, b in self.banks.items() if (not b.defaulted and b.capital_ratio() < self.params.min_capital_ratio)]
            sold = self.fire_sale_round(stressed, price_drawdown=abs(price_dd_cum))
            dd = self.price_impact(sold)
            price_dd_cum += dd
            step["firesale_sold"] = sold # type: ignore
            step["price_drawdown"] = dd # type: ignore

            # 3) Apply MTM losses from market drawdown on remaining illiquid assets
            if dd != 0.0:
                for b in self.banks.values():
                    if b.defaulted:
                        continue
                    loss = max(0.0, -dd) * b.illiquid_assets
                    b.illiquid_assets *= (1.0 + dd)
                    b.equity -= loss

            # defaults after MTM
            newly2 = self.mark_defaults()
            if newly2:
                step["new_defaults_after_mtm"] = newly2 # type: ignore

            history.append(step)

            # Stop if system stable: no new defaults, tiny sales/impact, and clearing converged (pvec stable)
            stable = (not newly and not newly2 and abs(sold) < 1e-6 and abs(dd) < 1e-9)
            if stable:
                break

        # Final metrics
        out = {
            "rounds": len(history),
            "history": history,
            "system_defaults": [b.id for b in self.banks.values() if b.defaulted],
            "total_equity": sum(max(0.0, b.equity) for b in self.banks.values()),
            "price_drawdown_cum": price_dd_cum,
            "banks": {bid: asdict(b) for bid, b in self.banks.items()},
        }
        return out

    # ----------------------- Shocks & I/O -----------------------------------

    def apply_exogenous_shock(self, *, id: str, equity_loss: float = 0.0, illiquid_haircut: float = 0.0) -> None:
        """
        Apply direct hit to a bank:
          - equity_loss: subtract from equity
          - illiquid_haircut: fraction, e.g., 0.1 = 10% mark-down on illiquid assets
        """
        b = self.banks[id]
        if equity_loss:
            b.equity -= max(0.0, equity_loss)
        if illiquid_haircut:
            dd = max(0.0, illiquid_haircut)
            b.illiquid_assets *= (1.0 - dd)
            b.equity -= b.illiquid_assets * dd / max(1e-12, 1.0 - dd)  # mark loss on the written-down amount

    def to_json(self) -> str:
        payload = {
            "params": asdict(self.params),
            "banks": {bid: asdict(b) for bid, b in self.banks.items()},
            "edges": [
                asdict(e) for outs in self.out_edges.values() for e in outs
            ],
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def from_json(s: str) -> "ContagionGraph":
        o = json.loads(s)
        g = ContagionGraph(ShockParams(**o["params"]))
        for bid, raw in o["banks"].items():
            g.add_bank(Bank(**raw))
        for e in o.get("edges", []):
            g.add_exposure(e["lender"], e["borrower"], e["amount"], recovery_rate=e.get("recovery_rate", 0.4))
        return g


# =============================================================================
# Tiny demo
# =============================================================================

if __name__ == "__main__":
    # Build a toy 3-bank system
    g = ContagionGraph()
    g.add_bank(Bank(id="A", name="Alpha", equity=100.0, liquid_assets=300.0, illiquid_assets=700.0, liabilities=800.0))
    g.add_bank(Bank(id="B", name="Beta",  equity=80.0,  liquid_assets=200.0, illiquid_assets=500.0, liabilities=620.0))
    g.add_bank(Bank(id="C", name="Gamma", equity=60.0,  liquid_assets=150.0, illiquid_assets=450.0, liabilities=540.0))

    # Interbank: lender->borrower (borrower owes lender)
    g.add_exposure("A", "B", 120.0, recovery_rate=0.5)  # B owes A 120
    g.add_exposure("B", "C", 100.0, recovery_rate=0.4)  # C owes B 100
    g.add_exposure("C", "A", 90.0,  recovery_rate=0.3)  # A owes C 90

    # Exogenous hit to B's book
    g.apply_exogenous_shock(id="B", equity_loss=30.0, illiquid_haircut=0.05)

    res = g.propagate()
    print(json.dumps(res, indent=2))