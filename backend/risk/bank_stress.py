# backend/risk/bank_stress.py
"""
Bank Stress Engine
------------------
Balance-sheet & liquidity stress testing per bank entity.

Features
- Capital: CET1%, RWA, leverage ratio, OCI/MTM impacts
- Credit: NPL uplift, LGD, provisions, RWA inflation
- Market: IR shock -> AFS/Trading MTM; HTM OCI buffer
- Liquidity: LCR, NSFR under deposit run-off & wholesale freeze
- Funding: spread shock lifts interest expense
- Coupling: map sovereign spread shocks to securities MTM
- Emissions: per-step snapshot to Redis stream (optional)

CLI
  python -m backend.risk.bank_stress --probe
  python -m backend.risk.bank_stress --yaml config/bank_stress.yaml --bank BANK_A --scenario "ir:+100,credit:+2,runoff:0.1"

YAML (example: config/bank_stress.yaml)
---------------------------------------
banks:
  BANK_A:
    # Balance sheet (currency units, e.g., USD)
    assets_total: 500e9
    loans_gross: 300e9
    securities_afs: 80e9
    securities_htm: 70e9
    trading_book: 20e9
    cash_hqlA: 30e9
    interbank_assets: 10e9
    # Liabilities & capital
    deposits: 350e9
    wholesale_funding: 90e9
    long_term_debt: 30e9
    equity_tier1: 20e9
    rwa: 250e9
    # Liquidity metrics (baseline)
    lcr_hqla: 90e9
    lcr_net_outflows_30d: 60e9
    nsfr_stable_funding: 320e9
    nsfr_required: 300e9
    # Credit quality
    npl_ratio: 0.025
    coverage_ratio: 0.55
    lgd: 0.45
    # IR/Duration (effective)
    dur_afs: 3.5
    dur_htm: 5.5
    dur_trading: 1.5
    # Mix / mapping
    sovereign_share_afs: 0.35       # share of AFS tied to sovereign risk bucket
    # Run-off tiers (optional overrides)
    runoff_core: 0.03
    runoff_non_core: 0.12
defaults:
  tax_rate: 0.25
  cet1_min: 0.085
  lcr_min: 1.0
  nsfr_min: 1.0
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List

# optional deps
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None  # type: ignore

# optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore

_NOW_MS = lambda: int(time.time() * 1000)


# ---------------------- Data Model ----------------------

@dataclass
class BankState:
    name: str

    # Balance sheet (currency)
    assets_total: float
    loans_gross: float
    securities_afs: float
    securities_htm: float
    trading_book: float
    cash_hqlA: float
    interbank_assets: float

    deposits: float
    wholesale_funding: float
    long_term_debt: float
    equity_tier1: float
    rwa: float

    # Liquidity baseline
    lcr_hqla: float
    lcr_net_outflows_30d: float
    nsfr_stable_funding: float
    nsfr_required: float

    # Credit quality / parameters
    npl_ratio: float = 0.02
    coverage_ratio: float = 0.50
    lgd: float = 0.45

    # Durations for IR shocks
    dur_afs: float = 3.0
    dur_htm: float = 5.0
    dur_trading: float = 1.0

    # Mix / mapping
    sovereign_share_afs: float = 0.30

    # Derived / evolving
    provision_stock: float = 0.0
    oci_afs: float = 0.0
    mtm_trading: float = 0.0
    pnl_pre_tax: float = 0.0
    pnl_after_tax: float = 0.0
    cet1_ratio: float = 0.0
    leverage_ratio: float = 0.0
    lcr: float = 0.0
    nsfr: float = 0.0
    breaches: List[str] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "BankState":
        c = BankState(**{k: v for k, v in asdict(self).items() if k not in ("notes","breaches")})
        c.notes = list(self.notes)
        c.breaches = list(self.breaches)
        return c


@dataclass
class StressDefaults:
    tax_rate: float = 0.25
    cet1_min: float = 0.085
    lcr_min: float = 1.0
    nsfr_min: float = 1.0


@dataclass
class ShockSpec:
    ir_bp: float = 0.0              # parallel IR shock in basis points (+100 = +1%)
    credit_pd_up: float = 0.0       # absolute PD uplift in percentage points (e.g., +2 = +2pp)
    credit_lgd_up: float = 0.0      # absolute LGD uplift (e.g., +5 = +5pp)
    runoff: float = 0.0             # deposit run-off fraction (0.1 = 10% out in 30d)
    wholesale_freeze: float = 0.0   # fraction of wholesale that can't roll (0.5 = half frozen)
    funding_spread_bp: float = 0.0  # increase in funding spreads (bps)
    sovereign_spread_bp: float = 0.0# sovereign spread widening (bps) for AFS sovereign share
    trading_var_mult: float = 1.0   # scale trading MTM by VAR multiplier
    rwa_inflation: float = 0.0      # % increase in RWA (0.1 = +10%)
    opex_hit: float = 0.0           # exogenous operating loss (currency)
    fx_depr: float = 0.0            # FX depreciation fraction (if relevant to books)


# ---------------------- Engine ----------------------

class BankStressEngine:
    def __init__(self, state: BankState, defaults: Optional[StressDefaults] = None):
        self.s = state
        self.defs = defaults or StressDefaults()

    # ---- main step ----
    def step(self, shocks: ShockSpec, publish: bool = False) -> BankState:
        s = self.s
        D = self.defs

        # Reset rolling P&L
        s.pnl_pre_tax = 0.0
        s.pnl_after_tax = 0.0
        s.breaches = []

        # 1) MARKET: interest rate shock -> AFS OCI + Trading MTM; HTM below-the-line (buffered)
        d_y = shocks.ir_bp / 1e4  # convert bps to decimal (100bp=0.01)
        # AFS OCI loss (approx) + sovereign spread add-on on sovereign share
        base_afs_loss = - s.dur_afs * d_y * s.securities_afs
        sov_add = - (shocks.sovereign_spread_bp / 1e4) * s.dur_afs * (s.securities_afs * s.sovereign_share_afs)
        s.oci_afs += (base_afs_loss + sov_add)
        # Trading immediate MTM
        s.mtm_trading += - s.dur_trading * d_y * s.trading_book * max(1.0, shocks.trading_var_mult)

        # HTM: track an "economic OCI buffer" but do not hit CET1 immediately (reg relief)
        htm_economic_oci = - s.dur_htm * d_y * s.securities_htm  # informational

        # 2) CREDIT: PD/LGD uplifts -> expected loss; provisions add to P&L and CET1 deduction
        base_npl = s.npl_ratio * s.loans_gross
        pd_uplift = (shocks.credit_pd_up / 100.0) * s.loans_gross
        lgd_now = min(1.0, max(0.0, s.lgd + (shocks.credit_lgd_up / 100.0)))
        expected_loss = pd_uplift * lgd_now
        # Provision increment relative to coverage on increased NPL base
        desired_provisions = (s.coverage_ratio * (base_npl + pd_uplift))
        delta_provisions = max(0.0, desired_provisions - s.provision_stock)
        s.provision_stock += delta_provisions
        s.pnl_pre_tax -= (expected_loss + delta_provisions)

        # 3) FUNDING: spread + wholesale freeze -> interest expense and liquidity draw
        funding_spread = shocks.funding_spread_bp / 1e4
        funding_cost_hit = funding_spread * (s.wholesale_funding + s.long_term_debt)
        s.pnl_pre_tax -= funding_cost_hit

        # 4) LIQUIDITY: deposit run-off and wholesale freeze -> LCR, NSFR
        runoff_amt = shocks.runoff * s.deposits
        frozen_wholesale = shocks.wholesale_freeze * s.wholesale_funding
        # LCR HQLA used to meet outflows; assume 100% of runoff and 100% of frozen roll don’t return in 30d
        net_outflows = s.lcr_net_outflows_30d + runoff_amt + frozen_wholesale
        lcr_hqla_next = max(0.0, s.lcr_hqla - (runoff_amt * 0.9))  # assume 90% of run-off met by HQLA liquidation
        s.lcr = (lcr_hqla_next / max(1e-9, net_outflows)) if net_outflows > 0 else float("inf")

        # NSFR: required rises with wholesale stress; stable funding declines with runoff
        nsfr_stable_next = max(0.0, s.nsfr_stable_funding - runoff_amt * 0.5)  # some deposits were stable
        nsfr_req_next = s.nsfr_required * (1.0 + 0.2 * shocks.wholesale_freeze)  # stress add-on
        s.nsfr = nsfr_stable_next / max(1e-9, nsfr_req_next)

        # 5) RWA & CET1: RWA inflation + OCI loss (AFS) + P&L after tax
        rwa_next = s.rwa * (1.0 + shocks.rwa_inflation)
        # CET1 bridge
        pre_cet1 = s.equity_tier1
        # OCI from AFS reduces CET1 directly; trading MTM & credit go through P&L => after-tax hit reduces CET1
        taxable = s.pnl_pre_tax + s.mtm_trading  # trading P&L passes P&L; credit already included
        s.pnl_after_tax = taxable * (1.0 - D.tax_rate)
        cet1_next = pre_cet1 + s.pnl_after_tax + s.oci_afs  # OCI cumulative
        s.equity_tier1 = max(0.0, cet1_next)
        s.rwa = max(0.0, rwa_next)

        # 6) Ratios & breaches
        s.cet1_ratio = (s.equity_tier1 / max(1e-9, s.rwa))
        s.leverage_ratio = (s.equity_tier1 / max(1e-9, s.assets_total))
        if s.cet1_ratio < D.cet1_min:
            s.breaches.append(f"CET1<{D.cet1_min:.3f}")
        if s.lcr < D.lcr_min:
            s.breaches.append("LCR<100%")
        if s.nsfr < D.nsfr_min:
            s.breaches.append("NSFR<100%")

        # 7) Bookkeeping & note
        s.notes.append({
            "ts_ms": _NOW_MS(),
            "ir_bp": shocks.ir_bp,
            "sov_bp": shocks.sovereign_spread_bp,
            "credit_pd_up": shocks.credit_pd_up,
            "credit_lgd_up": shocks.credit_lgd_up,
            "runoff": shocks.runoff,
            "wholesale_freeze": shocks.wholesale_freeze,
            "funding_spread_bp": shocks.funding_spread_bp,
            "rwa_inflation": shocks.rwa_inflation,
            "pnl_pre_tax": s.pnl_pre_tax,
            "pnl_after_tax": s.pnl_after_tax,
            "oci_afs": s.oci_afs,
            "mtm_trading": s.mtm_trading,
            "cet1_ratio": s.cet1_ratio,
            "lcr": s.lcr,
            "nsfr": s.nsfr,
            "breaches": list(s.breaches),
            "htm_economic_oci": htm_economic_oci
        })

        if publish and publish_stream:
            try:
                publish_stream("risk.bank_stress", {
                    "ts_ms": _NOW_MS(), "bank": s.name,
                    "cet1": s.cet1_ratio, "lcr": s.lcr, "nsfr": s.nsfr,
                    "pnl_after_tax": s.pnl_after_tax, "oci_afs": s.oci_afs,
                    "breaches": s.breaches
                })
            except Exception:
                pass

        return s

    # Convenience mapping from sovereign spread change to AFS loss (for contagion coupling)
    def sovereign_spread_to_mtm(self, spread_bp: float) -> float:
        s = self.s
        return - (spread_bp / 1e4) * s.dur_afs * (s.securities_afs * s.sovereign_share_afs)

    # Produce an approximate “equity loss” to feed your contagion graph edges
    def to_contagion_equity_loss(self) -> float:
        s = self.s
        # Use last note delta on CET1 (since start) as loss proxy
        if not s.notes:
            return 0.0
        # Sum P&L after tax + OCI to date (negative => equity loss)
        loss = - (s.pnl_after_tax + s.oci_afs)
        return max(0.0, loss)


# ---------------------- YAML Loader ----------------------

def load_from_yaml(path: str, bank: str) -> tuple[BankState, StressDefaults]:
    if yaml is None:
        raise RuntimeError("pyyaml not installed. Run: pip install pyyaml")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    b = (doc.get("banks") or {}).get(bank)
    if not b:
        raise KeyError(f"{bank} not found in {path}")

    defs = doc.get("defaults") or {}
    s = BankState(
        name=bank,
        assets_total=float(b["assets_total"]),
        loans_gross=float(b["loans_gross"]),
        securities_afs=float(b["securities_afs"]),
        securities_htm=float(b["securities_htm"]),
        trading_book=float(b["trading_book"]),
        cash_hqlA=float(b["cash_hqlA"]),
        interbank_assets=float(b["interbank_assets"]),
        deposits=float(b["deposits"]),
        wholesale_funding=float(b["wholesale_funding"]),
        long_term_debt=float(b["long_term_debt"]),
        equity_tier1=float(b["equity_tier1"]),
        rwa=float(b["rwa"]),
        lcr_hqla=float(b["lcr_hqla"]),
        lcr_net_outflows_30d=float(b["lcr_net_outflows_30d"]),
        nsfr_stable_funding=float(b["nsfr_stable_funding"]),
        nsfr_required=float(b["nsfr_required"]),
        npl_ratio=float(b.get("npl_ratio", 0.02)),
        coverage_ratio=float(b.get("coverage_ratio", 0.5)),
        lgd=float(b.get("lgd", 0.45)),
        dur_afs=float(b.get("dur_afs", 3.0)),
        dur_htm=float(b.get("dur_htm", 5.0)),
        dur_trading=float(b.get("dur_trading", 1.0)),
        sovereign_share_afs=float(b.get("sovereign_share_afs", 0.30)),
        runoff_core=float(b.get("runoff_core", 0.03)) if "runoff_core" in b else 0.03,  # optional, kept for compatibility # type: ignore
        runoff_non_core=float(b.get("runoff_non_core", 0.12)) if "runoff_non_core" in b else 0.12 # type: ignore
    )
    d = StressDefaults(
        tax_rate=float(defs.get("tax_rate", 0.25)),
        cet1_min=float(defs.get("cet1_min", 0.085)),
        lcr_min=float(defs.get("lcr_min", 1.0)),
        nsfr_min=float(defs.get("nsfr_min", 1.0)),
    )
    return s, d


# ---------------------- CLI & Helpers ----------------------

def _parse_scenario(s: str) -> ShockSpec:
    """
    "ir:+100,credit:+2,lgd:+5,runoff:0.1,freeze:0.5,fund:+50,sov:+150,rwa:+0.1"
    """
    spec = ShockSpec()
    if not s:
        return spec
    for tok in [t.strip() for t in s.split(",") if t.strip()]:
        if ":" not in tok:
            continue
        k, v = tok.split(":", 1)
        k = k.strip().lower(); v = v.strip()
        def fnum(x):
            try: return float(x)
            except: return 0.0
        if k in ("ir","rate","rates"): spec.ir_bp = fnum(v.replace("+",""))
        elif k in ("credit","pd"): spec.credit_pd_up = fnum(v.replace("+",""))
        elif k in ("lgd",): spec.credit_lgd_up = fnum(v.replace("+",""))
        elif k in ("runoff","run"): spec.runoff = fnum(v)
        elif k in ("freeze","wholesale","wholesale_freeze"): spec.wholesale_freeze = fnum(v)
        elif k in ("fund","funding","spread"): spec.funding_spread_bp = fnum(v.replace("+",""))
        elif k in ("sov","sovereign"): spec.sovereign_spread_bp = fnum(v.replace("+",""))
        elif k in ("var","trading_var"): spec.trading_var_mult = fnum(v)
        elif k in ("rwa", "rwa_infl"): spec.rwa_inflation = fnum(v)
        elif k in ("opex","loss"): spec.opex_hit = fnum(v)
        elif k in ("fx","fx_dep"): spec.fx_depr = fnum(v)
    return spec


def _probe():
    # Minimal example with toy numbers
    s = BankState(
        name="BANK_A",
        assets_total=500e9, loans_gross=300e9, securities_afs=80e9, securities_htm=70e9,
        trading_book=20e9, cash_hqlA=30e9, interbank_assets=10e9,
        deposits=350e9, wholesale_funding=90e9, long_term_debt=30e9,
        equity_tier1=20e9, rwa=250e9,
        lcr_hqla=90e9, lcr_net_outflows_30d=60e9, nsfr_stable_funding=320e9, nsfr_required=300e9,
        npl_ratio=0.025, coverage_ratio=0.55, lgd=0.45,
        dur_afs=3.5, dur_htm=5.5, dur_trading=1.5, sovereign_share_afs=0.35
    )
    eng = BankStressEngine(s)
    shock = _parse_scenario("ir:+100,credit:+2,lgd:+5,runoff:0.08,freeze:0.3,fund:+40,sov:+120,rwa:+0.08")
    eng.step(shock, publish=False)
    print(f"CET1={s.cet1_ratio:.3%}  LCR={s.lcr:.2f}  NSFR={s.nsfr:.2f}  Breaches={s.breaches}")

def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Bank Stress Engine")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--yaml", type=str, help="config/bank_stress.yaml")
    ap.add_argument("--bank", type=str, help="bank key in YAML")
    ap.add_argument("--scenario", type=str, help='e.g. "ir:+100,credit:+2,lgd:+5,runoff:0.1"')
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    if args.probe:
        _probe(); return

    if args.yaml and args.bank:
        s, d = load_from_yaml(args.yaml, args.bank)
        eng = BankStressEngine(s, d)
        shock = _parse_scenario(args.scenario or "")
        eng.step(shock, publish=args.publish)
        print(json.dumps(asdict(eng.s), indent=2, default=float))
    else:
        _probe()

if __name__ == "__main__":
    main()