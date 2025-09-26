#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# healthcare_reg_arbitrage.py
#
# Scenario engine to quantify **healthcare regulatory arbitrage** across jurisdictions
# (states/countries) and care models (hospital, ambulatory, telehealth, biotech, medtech).
#
# What it does
# ------------
# - Ingests a jurisdiction parameters CSV (rates, payer mix, wage/rent indices, legal flags)
# - Computes unit economics (rev / visit-procedure; cost / visit-procedure) by care model
# - Scores “arbitrage” vs a chosen base jurisdiction (EBITDA delta, risk-adjusted)
# - Flags key legal/regulatory blockers (CPOM, CoN, telehealth parity, SOP limits, trial rules)
# - Optional: sensitivity bands & plots; exports tidy CSVs
#
# Inputs
# ------
# --jurisdictions FILE (CSV, required). Columns (names are flexible; see template below):
#   jurisdiction                    : str  e.g., "Texas", "Bavaria"
#   region                          : str  (optional)
#   model                           : str  one of {'hospital','ambulatory','telehealth','biotech','medtech'}
#   medicare_rate_idx               : float (=1.00 at national baseline; use DRG/RVU equivalent scaling)
#   medicaid_rate_idx               : float
#   commercial_rate_idx             : float
#   payer_mix_medicaid_pct          : float 0..100
#   payer_mix_medicare_pct          : float 0..100
#   payer_mix_commercial_pct        : float 0..100
#   wage_idx                        : float (labor cost multiplier)
#   rent_idx                        : float (facility cost multiplier)
#   malpractice_premium_usd         : float (per FTE physician/year or per 10k procedures; see --scale)
#   cpom_flag                       : {0,1} corporate practice of medicine restriction (1 = restrictive)
#   con_flag                        : {0,1} certificate-of-need required
#   telehealth_parity_flag          : {0,1} payment parity for telehealth (1 = parity)
#   sop_expansion_idx               : float (scope-of-practice breadth; 1 baseline, >1 broader)
#   compliance_burden_idx           : float (paperwork/inspection burden; >1 = heavier)
#   approval_timeline_months        : float (for biotech/medtech trials/market access)
#   drug_price_controls_flag        : {0,1} (relevant for biotech)
#   trial_incentive_idx             : float (credits/grants for trials; >1 better)
#   corporate_tax_pct               : float
#   payroll_tax_pct                 : float
#
# Optional knobs (business model):
# --base-jurisdiction NAME        : compare everyone against this (default: first row)
# --asp_base_usd FLOAT            : baseline ASP (per visit/proc/kit) at national 1.0 index
# --cost_base_labor_usd FLOAT     : baseline labor cost per unit at wage_idx=1 (ex FTE mix)
# --cost_base_rent_usd FLOAT      : baseline facility cost per unit at rent_idx=1
# --cost_other_pct FLOAT          : other opex % of revenue (billing/it/supply)
# --malpractice_scale {'perFTE','per10k'} : treats malpractice_premium_usd scaling
# --volume_per_FTE INT            : used if malpractice_scale=perFTE to get per-unit cost
# --units_per_year INT            : denominator for rent/labor unitization if you have totals
# --risk_aversion FLOAT           : penalize risk flags (default 0.5); higher = stronger penalty
# --sensitivity-pct FLOAT         : build ± bands on key indices (default 0 for off)
# --plot                          : save simple PNG charts
# --outdir PATH                   : default ./artifacts
#
# Outputs
# -------
# outdir/
#   inputs_clean.csv
#   unit_economics.csv            (rev, cost, EBITDA/unit)
#   arbitrage_scores.csv          (vs base, with risk penalties)
#   blockers_flags.csv            (CPOM/CoN/telehealth etc.)
#   top_pairs.csv                 (best source→target moves by score)
#   plots/*.png                   (optional)
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    jurisdictions_file: str
    base_jur: Optional[str]
    asp_base_usd: float
    cost_labor_base: float
    cost_rent_base: float
    cost_other_pct: float
    malpractice_scale: str
    volume_per_fte: int
    units_per_year: int
    risk_aversion: float
    sensitivity_pct: float
    plot: bool
    outdir: str


# ----------------------------- IO helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "healthcare_reg_arbitrage_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def read_inputs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = [
        "jurisdiction", "model",
        "medicare_rate_idx","medicaid_rate_idx","commercial_rate_idx",
        "payer_mix_medicaid_pct","payer_mix_medicare_pct","payer_mix_commercial_pct",
        "wage_idx","rent_idx","malpractice_premium_usd",
        "cpom_flag","con_flag","telehealth_parity_flag",
        "sop_expansion_idx","compliance_burden_idx",
        "approval_timeline_months","drug_price_controls_flag",
        "trial_incentive_idx","corporate_tax_pct","payroll_tax_pct"
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"jurisdictions CSV missing columns: {missing}")
    # Types
    df["jurisdiction"] = df["jurisdiction"].astype(str).str.strip()
    df["model"] = df["model"].str.lower().str.strip()
    for c in [c for c in need if c not in ["jurisdiction","model"]]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Normalize payer mix to 100 if needed
    pm = df[["payer_mix_medicaid_pct","payer_mix_medicare_pct","payer_mix_commercial_pct"]].fillna(0)
    tot = pm.sum(axis=1).replace(0, np.nan)
    for col in pm.columns:
        df[col] = 100 * pm[col] / tot
    return df


# ----------------------------- Economics -----------------------------

def blended_rate(row: pd.Series, model: str) -> float:
    """
    Blended reimbursement index vs national ASP=1.0
    Telehealth parity & SOP breadth can uplift/downlift depending on model.
    """
    # Base blend
    mix_mcaid = row["payer_mix_medicaid_pct"]/100.0
    mix_mcare = row["payer_mix_medicare_pct"]/100.0
    mix_comm  = row["payer_mix_commercial_pct"]/100.0
    base = (mix_mcaid*row["medicaid_rate_idx"] +
            mix_mcare*row["medicare_rate_idx"] +
            mix_comm *row["commercial_rate_idx"])
    # Telehealth parity (only for telehealth model)
    if model == "telehealth":
        parity = 1.0 if row["telehealth_parity_flag"] >= 1 else 0.9
        base *= parity
    # SOP expansion (ambulatory care can substitute MD with APPs → capacity ↑ → effective margin)
    sop_uplift = 1.0 + 0.05*(row["sop_expansion_idx"] - 1.0) if model in ("ambulatory","telehealth") else 1.0
    return float(base * sop_uplift)


def unit_costs(row: pd.Series, cfg: Config) -> Tuple[float, float, float]:
    """
    Returns (labor_cost, facility_cost, malpractice_cost) per unit.
    """
    labor = cfg.cost_labor_base * float(row["wage_idx"])
    rent  = cfg.cost_rent_base * float(row["rent_idx"])
    # Malpractice scaling
    if cfg.malpractice_scale.lower() == "perfte":
        per_unit_malp = float(row["malpractice_premium_usd"]) / max(1, cfg.volume_per_fte)
    else:
        # per10k procedures/visits
        per_unit_malp = float(row["malpractice_premium_usd"]) / 10000.0
    return float(labor), float(rent), float(per_unit_malp)


def regulatory_risk_penalty(row: pd.Series, cfg: Config) -> float:
    """
    Convert legal flags to a multiplicative penalty factor in [0,1],
    higher penalty for higher risk_aversion.
    """
    hits = 0.0
    # CPOM and CoN are big frictions for ambulatory/telehealth/hospital
    hits += 0.6 if row["cpom_flag"] >= 1 else 0.0
    hits += 0.5 if row["con_flag"] >= 1 and row["model"] in ("hospital","ambulatory") else 0.0
    # Compliance burden scales
    hits += 0.3 * max(0.0, row["compliance_burden_idx"] - 1.0)
    # Drug price controls hit biotech/medtech ASP
    if row["model"] in ("biotech","medtech"):
        hits += 0.4 if row["drug_price_controls_flag"] >= 1 else 0.0
        # Long timelines reduce NPV; approximate as penalty
        hits += 0.02 * max(0.0, row["approval_timeline_months"] - 12.0) / 12.0
        hits -= 0.2 * max(0.0, row["trial_incentive_idx"] - 1.0)  # incentives offset
    # Telehealth parity absence also a hit for telehealth
    hits += 0.3 if (row["model"] == "telehealth" and row["telehealth_parity_flag"] < 1) else 0.0
    # Clamp
    hits = max(0.0, hits)
    # Map to multiplicative penalty
    return float(np.exp(-cfg.risk_aversion * hits))


def tax_effect(row: pd.Series) -> float:
    """
    Simple tax drag on EBITDA (corporate + payroll share applied to labor portion proxy).
    We apply taxes as a reduction factor on EBITDA.
    """
    corp = max(0.0, float(row["corporate_tax_pct"]))/100.0
    payr = max(0.0, float(row["payroll_tax_pct"]))/100.0
    # Treat half of EBITDA as labor-linked for payroll proxy; crude but directional.
    return float(1.0 - corp - 0.5*payr)


def compute_economics(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        model = r["model"]
        price_idx = blended_rate(r, model)
        asp = cfg.asp_base_usd * price_idx
        labor, rent, malp = unit_costs(r, cfg)
        other = cfg.cost_other_pct/100.0 * asp
        # Model-specific tweaks
        if model == "telehealth":
            # lower facility cost, higher IT (folded in 'other')
            rent *= 0.25
            other *= 1.10
        if model == "hospital":
            # heavier compliance → higher other opex
            other *= max(1.0, r["compliance_burden_idx"])
        if model in ("biotech","medtech"):
            # interpret ASP as gross margin per unit-equivalent (kit/therapy); elevate R&D share into other
            other *= 1.15

        rev = asp
        cash_cost = labor + rent + malp + other
        ebitda = rev - cash_cost
        margin = ebitda / rev if rev > 0 else np.nan

        # Risk & tax adjustments
        risk_mult = regulatory_risk_penalty(r, cfg)
        tax_mult  = tax_effect(r)
        ebitda_risk_tax = ebitda * risk_mult * tax_mult

        rows.append({
            "jurisdiction": r["jurisdiction"], "region": r.get("region",""),
            "model": model,
            "price_idx": price_idx, "asp_usd": asp,
            "cost_labor": labor, "cost_rent": rent, "cost_malpractice": malp, "cost_other": other,
            "revenue_usd": rev, "cost_total_usd": cash_cost,
            "ebitda_usd": ebitda, "ebitda_margin": margin,
            "risk_multiplier": risk_mult, "tax_multiplier": tax_mult,
            "ebitda_risk_tax_usd": ebitda_risk_tax,
            "cpom_flag": r["cpom_flag"], "con_flag": r["con_flag"],
            "telehealth_parity_flag": r["telehealth_parity_flag"],
            "sop_expansion_idx": r["sop_expansion_idx"],
            "compliance_burden_idx": r["compliance_burden_idx"],
            "approval_timeline_months": r["approval_timeline_months"],
            "drug_price_controls_flag": r["drug_price_controls_flag"],
            "trial_incentive_idx": r["trial_incentive_idx"],
            "corporate_tax_pct": r["corporate_tax_pct"], "payroll_tax_pct": r["payroll_tax_pct"],
        })
    out = pd.DataFrame(rows)
    return out


def arbitrage_vs_base(econ: pd.DataFrame, base_name: Optional[str]) -> pd.DataFrame:
    econ = econ.copy()
    # Base per model
    base = (
        econ if base_name is None else econ[econ["jurisdiction"].str.lower() == base_name.lower()]
    )
    if base.empty:
        base = econ.groupby("model", as_index=False).head(1)  # first per model
    # Map base ebitda per model
    bmap = base.groupby("model")["ebitda_risk_tax_usd"].mean().to_dict()
    econ["base_jur_ebitda_risk_tax"] = econ["model"].map(bmap)
    econ["arb_delta_usd"] = econ["ebitda_risk_tax_usd"] - econ["base_jur_ebitda_risk_tax"]
    econ["arb_delta_pct_of_rev"] = econ["arb_delta_usd"] / econ["revenue_usd"].replace(0, np.nan)
    # Score: delta normalized by volatility proxy (compliance burden + flags)
    vol_proxy = 0.25 + 0.25*econ["compliance_burden_idx"].clip(lower=0) \
                + 0.25*econ["cpom_flag"] + 0.25*econ["con_flag"]
    econ["arb_score"] = econ["arb_delta_usd"] / vol_proxy.replace(0, np.nan)
    return econ


def blockers_table(econ: pd.DataFrame) -> pd.DataFrame:
    blk = econ[[
        "jurisdiction","model","cpom_flag","con_flag","telehealth_parity_flag",
        "drug_price_controls_flag","approval_timeline_months","sop_expansion_idx","compliance_burden_idx"
    ]].copy()
    # Add text flags
    blk["blockers_summary"] = (
        blk.apply(lambda r: "; ".join(
            [txt for txt in [
                "CPOM" if r["cpom_flag"]==1 else "",
                "CoN" if r["con_flag"]==1 else "",
                "No parity" if r["telehealth_parity_flag"]==0 and r["model"]=="telehealth" else "",
                "Drug price ctrl" if r["drug_price_controls_flag"]==1 and r["model"] in ("biotech","medtech") else "",
                f"Timeline>{int(r['approval_timeline_months'])}m" if r["approval_timeline_months"]>18 and r["model"] in ("biotech","medtech") else "",
                "Narrow SOP" if r["sop_expansion_idx"]<1.0 and r["model"] in ("ambulatory","telehealth") else "",
                "High burden" if r["compliance_burden_idx"]>1.2 else ""
            ] if txt]),
            axis=1)
    )
    return blk


def top_pairs(econ: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """
    Return top source→target moves: for each model, pair every jurisdiction with every other,
    compute target_score - source_score.
    """
    rows = []
    for m, sub in econ.groupby("model"):
        sub = sub[["jurisdiction","arb_score","ebitda_risk_tax_usd"]].copy()
        for i in range(len(sub)):
            for j in range(len(sub)):
                if i == j: continue
                a = sub.iloc[i]; b = sub.iloc[j]
                rows.append({
                    "model": m,
                    "source": a["jurisdiction"],
                    "target": b["jurisdiction"],
                    "score_uplift": b["arb_score"] - a["arb_score"],
                    "ebitda_uplift_usd": b["ebitda_risk_tax_usd"] - a["ebitda_risk_tax_usd"]
                })
    df = pd.DataFrame(rows)
    return df.sort_values(["model","score_uplift","ebitda_uplift_usd"], ascending=[True, False, False]).groupby("model").head(k)


def sensitivity(econ: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Simple ± sensitivity on price_idx and wage_idx.
    """
    if cfg.sensitivity_pct <= 0:
        return pd.DataFrame()
    sens = []
    pct = cfg.sensitivity_pct/100.0
    for _, r in econ.iterrows():
        for sign in (-1, +1):
            price_idx_adj = r["price_idx"] * (1 + sign*pct)
            wage_idx_adj = (r["cost_labor"]/cfg.cost_labor_base) * (1 + sign*pct)
            asp = cfg.asp_base_usd * price_idx_adj
            labor = cfg.cost_labor_base * wage_idx_adj
            new_ebitda = asp - (labor + r["cost_rent"]*(1 + 0) + r["cost_malpractice"] + r["cost_other"])
            sens.append({
                "jurisdiction": r["jurisdiction"], "model": r["model"],
                "case": f"{'down' if sign<0 else 'up'}_{int(cfg.sensitivity_pct)}pct",
                "ebitda_delta_usd": new_ebitda - r["ebitda_usd"]
            })
    return pd.DataFrame(sens)


# ----------------------------- Plotting -----------------------------

def make_plots(econ: pd.DataFrame, outdir: str):
    if plt is None or econ.empty:
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    # EBITDA/unit by jurisdiction for each model (top 12)
    for m, sub in econ.groupby("model"):
        s = sub.sort_values("ebitda_risk_tax_usd", ascending=False).head(12)
        fig = plt.figure(figsize=(10,5)); ax = plt.gca()
        ax.barh(s["jurisdiction"], s["ebitda_risk_tax_usd"])
        ax.invert_yaxis()
        ax.set_title(f"{m}: EBITDA (risk & tax adj) per unit")
        ax.set_xlabel("USD / unit")
        plt.tight_layout(); fig.savefig(os.path.join(outdir, "plots", f"{m}_ebitda_rank.png"), dpi=140); plt.close(fig)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Healthcare regulatory arbitrage analyzer")
    ap.add_argument("--jurisdictions", required=True, help="CSV of jurisdictions & parameters")
    ap.add_argument("--base-jurisdiction", type=str, default=None)
    ap.add_argument("--asp-base-usd", type=float, default=220.0, help="Baseline ASP at index=1.0")
    ap.add_argument("--cost-base-labor-usd", type=float, default=80.0)
    ap.add_argument("--cost-base-rent-usd", type=float, default=15.0)
    ap.add_argument("--cost-other-pct", type=float, default=18.0)
    ap.add_argument("--malpractice-scale", choices=["perFTE","per10k"], default="per10k")
    ap.add_argument("--volume-per-FTE", type=int, default=2500)
    ap.add_argument("--units-per-year", type=int, default=25000)
    ap.add_argument("--risk-aversion", type=float, default=0.5)
    ap.add_argument("--sensitivity-pct", type=float, default=0.0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        jurisdictions_file=args.jurisdictions,
        base_jur=args.base_jurisdiction,
        asp_base_usd=float(args.asp_base_usd),
        cost_labor_base=float(args.cost_base_labor_usd),
        cost_rent_base=float(args.cost_base_rent_usd),
        cost_other_pct=float(args.cost_other_pct),
        malpractice_scale=args.malpractice_scale,
        volume_per_fte=int(args.volume_per_FTE),
        units_per_year=int(args.units_per_year),
        risk_aversion=float(args.risk_aversion),
        sensitivity_pct=float(args.sensitivity_pct),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    print(f"[INFO] Writing to: {cfg.outdir}")

    df = read_inputs(cfg.jurisdictions_file)
    df.to_csv(os.path.join(cfg.outdir, "inputs_clean.csv"), index=False)

    econ = compute_economics(df, cfg)
    econ = arbitrage_vs_base(econ, cfg.base_jur)

    econ.to_csv(os.path.join(cfg.outdir, "unit_economics.csv"), index=False)

    blk = blockers_table(econ)
    blk.to_csv(os.path.join(cfg.outdir, "blockers_flags.csv"), index=False)

    pairs = top_pairs(econ)
    pairs.to_csv(os.path.join(cfg.outdir, "top_pairs.csv"), index=False)

    if cfg.sensitivity_pct > 0:
        sens = sensitivity(econ, cfg)
        sens.to_csv(os.path.join(cfg.outdir, "sensitivity.csv"), index=False)

    # Export arbitrage ranking
    rank = (
        econ.sort_values(["model","arb_score"], ascending=[True, False])
            .loc[:, ["jurisdiction","model","revenue_usd","cost_total_usd","ebitda_usd",
                     "risk_multiplier","tax_multiplier","ebitda_risk_tax_usd",
                     "arb_delta_usd","arb_delta_pct_of_rev","arb_score"]]
    )
    rank.to_csv(os.path.join(cfg.outdir, "arbitrage_scores.csv"), index=False)

    if cfg.plot:
        make_plots(econ, cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    print("\n=== Top arbitrage by model (risk & tax adjusted EBITDA/unit) ===")
    snap = rank.groupby("model").head(5)
    with pd.option_context("display.width", 140):
        print(snap.round(2).to_string(index=False))

    print("\nFiles written to:", cfg.outdir)


if __name__ == "__main__":
    main()