#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# border_trade_policy.py
#
# Compact scenario engine to quantify the impact of **border-trade policy
# shocks** (tariffs, quotas, export bans, NTBs, FX moves, logistics frictions,
# VAT rebate changes) on trade volumes, prices, CPI, and fiscal revenue.
#
# Key features
# ------------
# - Reads a baseline trade panel (by sector × partner) and a list of policy
#   actions (dated). Applies them sequentially to build counterfactuals.
# - Models ad-valorem wedges on import/export prices (tariff & NTB AVEs),
#   quota rationing, simple FX pass-through, and logistics time-cost adder.
# - Uses isoelastic demand/supply to update volumes; reports ΔCPI from
#   consumption weights; approximates Δ consumer surplus (triangle rule);
#   tabulates tariff revenue and export tax-equivalent revenue.
# - Produces tidy CSVs and optional PNG plots.
#
# Inputs
# ------
# --baseline FILE (CSV, required) columns:
#   sector                : str
#   partner               : str
#   imports_usd           : float  (annual baseline import value)
#   exports_usd           : float  (annual baseline export value)
#   cif_price_in          : float  (baseline import unit price, same currency)
#   fob_price_out         : float  (baseline export unit price)
#   import_elasticity     : float  (|ε_m| > 0, isoelastic)
#   export_elasticity     : float  (|ε_x| > 0, isoelastic)
#   cpi_weight            : float  (share of CPI basket, 0..1; can be 0)
#   va_share              : float  (domestic value-add share of sectoral output, 0..1)
#
# --policies FILE (CSV, required) columns:
#   date                  : YYYY-MM-DD
#   measure               : {'tariff_in','tariff_out','ntb_in','ntb_out',
#                            'quota_in','quota_out','export_ban',
#                            'fx_depreciation','logistics_days','vat_rebate_out'}
#   sector                : str or '' for All
#   partner               : str or '' for All
#   value                 : float (percentage points for rates; absolute days for logistics)
#   notes                 : str (optional)
#
# Interpretation
# --------------
# - Rates in 'value' are in **percent** (e.g., 5 = +5% ad-valorem tariff).
# - 'fx_depreciation' is % change in **domestic currency per USD** (e.g., 10 means -10% appreciation of home currency).
#   Pass-through to border prices controlled by --fx-pass-through (default 0.5).
# - 'logistics_days' increases lead time (days). Cost adder uses --time-value-bps (basis points/day)
#   applied to price as an AVE: ave = days * bps/10000.
# - 'quota_*' is a binding cap expressed as **% of baseline volume** remaining permissible (e.g., 60 means cap at 60%).
# - 'export_ban' with value > 0 sets exports to zero for matched rows.
# - 'vat_rebate_out' reduces export price received by producers by given % (e.g., -5 lowers rebate → worsens net price).
#
# Outputs
# -------
# outdir/
#   run_params.json
#   policy_timeline.csv
#   scenario_panel.csv         (sector×partner with base vs counterfactual)
#   cpi_decomposition.csv
#   fiscal_revenue.csv
#   plots/*.png (optional)
#
# Usage
# -----
# python border_trade_policy.py \
#   --baseline baseline.csv \
#   --policies policies.csv \
#   --year-scale 1.0 \
#   --fx-pass-through 0.5 \
#   --time-value-bps 8 \
#   --start 2025-01-01 --end 2025-12-31 \
#   --plot
#
# Dependencies: pip install pandas numpy matplotlib python-dateutil orjson

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import orjson as _json
    def dumps(obj): return _json.dumps(obj)
except Exception:
    def dumps(obj): return json.dumps(obj).encode("utf-8")


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    baseline: str
    policies: str
    start: Optional[str]
    end: Optional[str]
    fx_pass: float
    time_bps: float
    year_scale: float
    plot: bool
    outdir: str


# ----------------------------- IO helpers -----------------------------

def ensure_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, f"border_trade_policy_{ts}")
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    return outdir


def read_baseline(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = ["sector","partner","imports_usd","exports_usd","cif_price_in","fob_price_out",
           "import_elasticity","export_elasticity","cpi_weight","va_share"]
    for c in req:
        if c not in df.columns:
            raise SystemExit(f"baseline missing column: {c}")
    # Fill/clean
    for c in ["imports_usd","exports_usd","cif_price_in","fob_price_out",
              "import_elasticity","export_elasticity","cpi_weight","va_share"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # guardrails
    df["import_elasticity"] = df["import_elasticity"].abs().replace(0, 1.0)
    df["export_elasticity"] = df["export_elasticity"].abs().replace(0, 1.0)
    df["cpi_weight"] = df["cpi_weight"].clip(0, 1)
    df["va_share"] = df["va_share"].clip(0, 1)
    return df


def read_policies(path: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    pol = pd.read_csv(path)
    req = ["date","measure","sector","partner","value"]
    for c in req:
        if c not in pol.columns:
            raise SystemExit(f"policies missing column: {c}")
    pol["date"] = pd.to_datetime(pol["date"])
    if start:
        pol = pol[pol["date"] >= pd.to_datetime(start)]
    if end:
        pol = pol[pol["date"] <= pd.to_datetime(end)]
    pol = pol.sort_values("date").reset_index(drop=True)
    # normalize blanks
    pol["sector"] = pol["sector"].fillna("").astype(str)
    pol["partner"] = pol["partner"].fillna("").astype(str)
    pol["measure"] = pol["measure"].str.strip()
    pol["value"] = pd.to_numeric(pol["value"], errors="coerce").fillna(0.0)
    return pol


# ----------------------------- Economics core -----------------------------

def apply_ave(
    df: pd.DataFrame,
    pol: pd.DataFrame,
    fx_pass: float,
    time_bps: float
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Build ad-valorem equivalents (AVEs) for import and export sides per row.
    Returns a copy of df with added columns and a timeline list for auditing.
    """
    out = df.copy()
    out["ave_in"] = 0.0     # % applied to CIF price (imports)
    out["ave_out"] = 0.0    # % applied to FOB price (exports)
    out["quota_in_cap"] = np.inf
    out["quota_out_cap"] = np.inf
    out["export_ban"] = False
    out["vat_rebate_out"] = 0.0  # % change to producer net price

    timeline = []
    for _, row in pol.iterrows():
        m = row["measure"]
        sector = row["sector"].strip()
        partner = row["partner"].strip()
        val = float(row["value"])
        mask = np.ones(len(out), dtype=bool)
        if sector:
            mask &= (out["sector"] == sector)
        if partner:
            mask &= (out["partner"] == partner)

        if m == "tariff_in":
            out.loc[mask, "ave_in"] += val
        elif m == "tariff_out":
            out.loc[mask, "ave_out"] += val
        elif m == "ntb_in":
            out.loc[mask, "ave_in"] += val
        elif m == "ntb_out":
            out.loc[mask, "ave_out"] += val
        elif m == "quota_in":
            # cap at X% of baseline imports
            cap = (val / 100.0) * out.loc[mask, "imports_usd"]
            out.loc[mask, "quota_in_cap"] = np.minimum(out.loc[mask, "quota_in_cap"].values, cap.values)
        elif m == "quota_out":
            cap = (val / 100.0) * out.loc[mask, "exports_usd"]
            out.loc[mask, "quota_out_cap"] = np.minimum(out.loc[mask, "quota_out_cap"].values, cap.values)
        elif m == "export_ban":
            out.loc[mask, "export_ban"] = out.loc[mask, "export_ban"] | (val > 0)
        elif m == "fx_depreciation":
            # pass-through fraction to border prices
            out.loc[mask, "ave_in"] += fx_pass * val
            out.loc[mask, "ave_out"] -= fx_pass * val  # depreciation makes exports cheaper to foreigners
        elif m == "logistics_days":
            add = (val * time_bps) / 100.0  # convert bps/day to %
            out.loc[mask, "ave_in"] += add
            out.loc[mask, "ave_out"] += add
        elif m == "vat_rebate_out":
            out.loc[mask, "vat_rebate_out"] += val
        else:
            # ignore unknown measure
            pass

        timeline.append({
            "date": str(row["date"].date()),
            "measure": m, "sector": sector or "All", "partner": partner or "All",
            "value": val
        })

    return out, timeline


def isoelastic_update(base_val: float, ave_pct: float, elastic: float, cap: float, ban: bool) -> Tuple[float, float]:
    """
    Given baseline value (USD), ad-valorem price change ave_pct (%), and |elastic|,
    compute new value using isoelastic demand: Q' = Q * (1 + ave)^{-ε}
    Value V = P*Q; with ad-valorem on price, V' = V * (1 + ave)^{1-ε}.
    Apply quota cap and bans. Returns (new_value_usd, price_change_pct_effective).
    """
    if base_val <= 0 or ban:
        return 0.0, ave_pct
    ave = ave_pct / 100.0
    eps = max(0.05, float(elastic))
    v_prime = base_val * (1.0 + ave) ** (1.0 - eps)
    if np.isfinite(cap):
        v_prime = min(v_prime, cap)
    return float(max(v_prime, 0.0)), ave_pct


def consumer_surplus_change(base_val: float, ave_pct: float, elastic: float) -> float:
    """
    Triangle approximation of ΔCS for an ad-valorem increase on imports.
    ΔCS ≈ -0.5 * ε * V * (ΔP/P)^2   (on value basis), negative for price increases.
    """
    if base_val <= 0:
        return 0.0
    eps = max(0.05, float(elastic))
    dp = ave_pct / 100.0
    return -0.5 * eps * base_val * (dp ** 2)


def tariff_revenue(base_val: float, new_val: float, ave_tariff_pct: float) -> float:
    """
    Approximate tariff revenue = effective ad-valorem tariff * new import value / (1 + total AVE).
    Here we use the tariff component only on the counterfactual value.
    """
    if new_val <= 0:
        return 0.0
    t = max(0.0, ave_tariff_pct) / 100.0
    return new_val * t / (1.0 + t)


# ----------------------------- Scenario runner -----------------------------

def run_scenario(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[dict]]:
    base = read_baseline(cfg.baseline)
    pol = read_policies(cfg.policies, cfg.start, cfg.end)
    enriched, timeline = apply_ave(base, pol, cfg.fx_pass, cfg.time_bps)

    # Decompose AVEs into tariff vs. other for revenue purposes
    # We approximate tariff components as portions of ave_in/out coming from explicit tariff_* measures
    pol_tar_in = pol[pol["measure"] == "tariff_in"].groupby(["sector","partner"])["value"].sum().rename("tariff_in").reset_index()
    pol_tar_out = pol[pol["measure"] == "tariff_out"].groupby(["sector","partner"])["value"].sum().rename("tariff_out").reset_index()
    enriched = enriched.merge(pol_tar_in, on=["sector","partner"], how="left").merge(pol_tar_out, on=["sector","partner"], how="left")
    enriched["tariff_in"] = enriched["tariff_in"].fillna(0.0)
    enriched["tariff_out"] = enriched["tariff_out"].fillna(0.0)

    rows = []
    for _, r in enriched.iterrows():
        # IMPORT side
        new_imp_val, _ = isoelastic_update(
            base_val=r["imports_usd"] * cfg.year_scale,
            ave_pct=r["ave_in"],
            elastic=r["import_elasticity"],
            cap=r["quota_in_cap"] if np.isfinite(r["quota_in_cap"]) else np.inf,
            ban=False
        )
        # EXPORT side
        new_exp_val, _ = isoelastic_update(
            base_val=r["exports_usd"] * cfg.year_scale,
            ave_pct=(r["ave_out"] - r["vat_rebate_out"]),  # export rebates affect producer price
            elastic=r["export_elasticity"],
            cap=r["quota_out_cap"] if np.isfinite(r["quota_out_cap"]) else np.inf,
            ban=bool(r["export_ban"])
        )

        # Fiscal: tariff revenue only from tariff_in component on imports
        rev = tariff_revenue(
            base_val=r["imports_usd"],
            new_val=new_imp_val,
            ave_tariff_pct=r["tariff_in"]
        )

        # Welfare: ΔCS from import-side price change
        dcs = consumer_surplus_change(
            base_val=r["imports_usd"] * cfg.year_scale,
            ave_pct=r["ave_in"],
            elastic=r["import_elasticity"]
        )

        # CPI: pass through %Δ import price * CPI weight (very stylized)
        cpi_contrib = (r["ave_in"] / 100.0) * r["cpi_weight"]

        rows.append({
            "sector": r["sector"],
            "partner": r["partner"],
            "imports_base_usd": r["imports_usd"] * cfg.year_scale,
            "exports_base_usd": r["exports_usd"] * cfg.year_scale,
            "imports_new_usd": new_imp_val,
            "exports_new_usd": new_exp_val,
            "ave_in_pct": r["ave_in"],
            "ave_out_pct": r["ave_out"],
            "quota_in_cap_usd": r["quota_in_cap"] if np.isfinite(r["quota_in_cap"]) else np.nan,
            "quota_out_cap_usd": r["quota_out_cap"] if np.isfinite(r["quota_out_cap"]) else np.nan,
            "export_ban": bool(r["export_ban"]),
            "tariff_in_pct": r["tariff_in"],
            "tariff_revenue_usd": rev,
            "delta_consumer_surplus_usd": dcs,
            "cpi_weight": r["cpi_weight"],
            "cpi_contribution": cpi_contrib,
            "va_share": r["va_share"]
        })

    panel = pd.DataFrame(rows)
    # Aggregations
    cpi = panel.groupby("sector", as_index=False)[["cpi_weight","cpi_contribution"]].sum()
    cpi["sector_cpi_impact_pct"] = 100.0 * cpi["cpi_contribution"]  # convert share-weighted to percentage points
    cpi_total = float(100.0 * panel["cpi_contribution"].sum())

    fiscal = panel.groupby(["sector"], as_index=False)["tariff_revenue_usd"].sum()
    fiscal_total = float(panel["tariff_revenue_usd"].sum())

    # Attach totals rows
    tot_row = {
        "sector": "__TOTAL__", "partner": "",
        "imports_base_usd": panel["imports_base_usd"].sum(),
        "exports_base_usd": panel["exports_base_usd"].sum(),
        "imports_new_usd": panel["imports_new_usd"].sum(),
        "exports_new_usd": panel["exports_new_usd"].sum(),
        "ave_in_pct": np.nan, "ave_out_pct": np.nan,
        "quota_in_cap_usd": np.nan, "quota_out_cap_usd": np.nan,
        "export_ban": False, "tariff_in_pct": np.nan,
        "tariff_revenue_usd": fiscal_total,
        "delta_consumer_surplus_usd": panel["delta_consumer_surplus_usd"].sum(),
        "cpi_weight": panel["cpi_weight"].sum(),
        "cpi_contribution": panel["cpi_contribution"].sum(),
        "va_share": np.nan
    }
    panel_total = pd.concat([panel, pd.DataFrame([tot_row])], ignore_index=True)

    # Format secondary tables
    cpi_tbl = cpi[["sector","cpi_weight","sector_cpi_impact_pct"]].copy()
    cpi_tbl = pd.concat([cpi_tbl, pd.DataFrame([{
        "sector": "__TOTAL__", "cpi_weight": cpi_tbl["cpi_weight"].sum(),
        "sector_cpi_impact_pct": cpi_total
    }])], ignore_index=True)

    fiscal_tbl = fiscal.copy()
    fiscal_tbl = pd.concat([fiscal_tbl, pd.DataFrame([{
        "sector": "__TOTAL__", "tariff_revenue_usd": fiscal_total
    }])], ignore_index=True)

    return panel_total, cpi_tbl, fiscal_tbl, timeline


# ----------------------------- Plotting -----------------------------

def make_plots(panel: pd.DataFrame, cpi: pd.DataFrame, outdir: str):
    if plt is None:
        return
    # Trade bars by sector (imports base vs new)
    sec = panel[panel["sector"] != "__TOTAL__"].groupby("sector", as_index=False)[
        ["imports_base_usd","imports_new_usd","exports_base_usd","exports_new_usd"]
    ].sum().sort_values("imports_base_usd", ascending=False).head(20)

    fig1 = plt.figure(figsize=(10,6))
    ax = plt.gca()
    idx = np.arange(len(sec))
    w = 0.35
    ax.bar(idx - w/2, sec["imports_base_usd"]/1e9, width=w, label="Imports (base)")
    ax.bar(idx + w/2, sec["imports_new_usd"]/1e9, width=w, label="Imports (new)")
    ax.set_xticks(idx); ax.set_xticklabels(sec["sector"], rotation=40, ha="right")
    ax.set_ylabel("USD bn"); ax.set_title("Imports: base vs. policy scenario")
    ax.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(outdir, "plots", "imports_base_vs_new.png"), dpi=140)
    plt.close(fig1)

    # CPI contributions
    cpi2 = cpi[cpi["sector"] != "__TOTAL__"].sort_values("sector_cpi_impact_pct", ascending=False).head(20)
    fig2 = plt.figure(figsize=(8,6))
    ax2 = plt.gca()
    ax2.barh(cpi2["sector"], cpi2["sector_cpi_impact_pct"])
    ax2.set_xlabel("CPI impact (percentage points)"); ax2.set_title("Sector CPI contributions")
    plt.tight_layout()
    fig2.savefig(os.path.join(outdir, "plots", "cpi_contributions.png"), dpi=140)
    plt.close(fig2)


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Border trade policy scenario engine")
    ap.add_argument("--baseline", required=True, help="CSV with baseline sector×partner trade panel")
    ap.add_argument("--policies", required=True, help="CSV with dated policy actions")
    ap.add_argument("--start", type=str, default=None, help="Earliest policy date to include (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="Latest policy date to include (YYYY-MM-DD)")
    ap.add_argument("--fx-pass-through", type=float, default=0.5, dest="fx_pass",
                    help="FX pass-through to border prices (0..1), default 0.5")
    ap.add_argument("--time-value-bps", type=float, default=8.0, dest="time_bps",
                    help="AVE basis points per extra logistics day (default 8 bps/day)")
    ap.add_argument("--year-scale", type=float, default=1.0,
                    help="Scale baseline annual values (e.g., 0.5 for half-year)")
    ap.add_argument("--plot", action="store_true", help="Write PNG plots")
    ap.add_argument("--outdir", type=str, default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        baseline=args.baseline,
        policies=args.policies,
        start=args.start,
        end=args.end,
        fx_pass=float(max(0.0, min(1.0, args.fx_pass))),
        time_bps=float(max(0.0, args.time_bps)),
        year_scale=float(max(0.0, args.year_scale)),
        plot=bool(args.plot),
        outdir=args.outdir
    )

    outdir = ensure_outdir(cfg.outdir)
    print(f"[INFO] Writing artifacts to: {outdir}")

    panel, cpi_tbl, fiscal_tbl, timeline = run_scenario(cfg)

    # Write outputs
    panel.to_csv(os.path.join(outdir, "scenario_panel.csv"), index=False)
    cpi_tbl.to_csv(os.path.join(outdir, "cpi_decomposition.csv"), index=False)
    fiscal_tbl.to_csv(os.path.join(outdir, "fiscal_revenue.csv"), index=False)
    with open(os.path.join(outdir, "policy_timeline.csv"), "w", encoding="utf-8") as f:
        pd.DataFrame(timeline).to_csv(f, index=False)
    with open(os.path.join(outdir, "run_params.json"), "wb") as f:
        f.write(dumps(cfg.__dict__))

    if cfg.plot:
        make_plots(panel, cpi_tbl, outdir)
        print("[OK] Plots saved to:", os.path.join(outdir, "plots"))

    # Console summary
    tot = panel[panel["sector"] == "__TOTAL__"].iloc[0]
    print("\n=== Scenario summary ===")
    print(f"ΔImports (USD bn): {(tot['imports_new_usd'] - tot['imports_base_usd'])/1e9:,.2f}")
    print(f"ΔExports (USD bn): {(tot['exports_new_usd'] - tot['exports_base_usd'])/1e9:,.2f}")
    print(f"Tariff revenue (USD bn): {panel['tariff_revenue_usd'].sum()/1e9:,.2f}")
    print(f"ΔConsumer surplus (USD bn): {panel['delta_consumer_surplus_usd'].sum()/1e9:,.2f}")
    print(f"Headline CPI impact (pp): {100*panel['cpi_contribution'].sum():.2f}")


if __name__ == "__main__":
    main()