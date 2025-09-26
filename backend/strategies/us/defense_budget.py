#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# defense_budget.py
#
# Analyze & project **defense budgets** by country:
# - Cleans country-year panels (local currency → USD → real terms)
# - Computes % of GDP, per-capita, category shares (Personnel/Procurement/R&D/O&M)
# - Flags NATO 2% & equipment 20% guidelines (if data available)
# - Regional & alliance aggregates (optional mapping)
# - 1–10y baseline forecasts via rolling CAGR + optional scenario shocks
# - Exports tidy CSVs and optional PNG plots
#
# Inputs
# ------
# --budgets FILE (CSV, required). Accepts wide or long categories.
#   Required cols:
#     country, year, defense_local,
#     fx_usd (LCU per 1 USD, or USD per 1 LCU if you set --fx-invert),
#     gdp_usd, population
#   Optional cols (used when present):
#     cpi_index (base=100), deflator (base=1), ppp_rate (LCU per PPP-$)
#     cat_personnel, cat_procurement, cat_rnd, cat_operations (all in local currency)
#   Alternative (long) breakdown accepted if you pass --long-cats and provide:
#     country, year, category, amount_local   # category in {'personnel','procurement','rnd','operations',...}
#
# --map FILE (CSV, optional) for region/alliance grouping:
#   country, region, alliance
#
# --scenario FILE (CSV, optional) with shocks (percent) to baseline in given years:
#   country, year, shock_pct   # e.g., +10 means +10% to that year's baseline
#   country can be 'ALL'; year can be 'ALL' (applies broadly)
#
# Key flags
# ---------
# --fx-invert            : if fx_usd is USD per 1 LCU rather than LCU per USD
# --price-base YEAR      : real terms base year (default: last year in data)
# --forecast-years N     : years to project (default 5)
# --forecast-window N    : lookback years for CAGR (default 5)
# --plot                 : write PNG charts
# --outdir PATH          : default ./artifacts
#
# Outputs
# -------
# outdir/
#   cleaned.csv                 (core panel with USD, real, shares)
#   regional_aggregates.csv     (by region/alliance if map provided)
#   compliance.csv              (%GDP, equipment>20% flags, NATO 2% flags)
#   forecasts_baseline.csv      (country-year baseline projections)
#   forecasts_scenario.csv      (with shocks if provided)
#   plots/*.png                 (optional)
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ----------------------------- Config -----------------------------

@dataclass
class Config:
    budgets_file: str
    map_file: Optional[str]
    scenario_file: Optional[str]
    long_cats: bool
    fx_invert: bool
    price_base: Optional[int]
    forecast_years: int
    forecast_window: int
    plot: bool
    outdir: str


# ----------------------------- IO helpers -----------------------------

def ensure_outdir(base: str) -> str:
    out = os.path.join(base, "defense_budget_artifacts")
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def read_budgets(path: str, long_cats: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"country", "year", "defense_local", "fx_usd", "gdp_usd", "population"}
    if not req.issubset(df.columns):
        raise SystemExit(f"budgets CSV missing required columns: {sorted(list(req - set(df.columns)))}")
    df["country"] = df["country"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    # numeric coercion
    for c in ["defense_local","fx_usd","gdp_usd","population","cpi_index","deflator","ppp_rate",
              "cat_personnel","cat_procurement","cat_rnd","cat_operations"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Long categories -> wide
    if long_cats:
        long_req = {"country","year","category","amount_local"}
        if not long_req.issubset(df.columns):
            raise SystemExit("When --long-cats is set, you must include columns: country, year, category, amount_local")
        base_cols = [c for c in df.columns if c not in ("category","amount_local")]
        cats = (df.pivot_table(index=base_cols, columns="category", values="amount_local", aggfunc="sum")
                  .reset_index().rename_axis(None, axis=1))
        # Normalize known names
        rename_map = {
            "personnel": "cat_personnel",
            "procurement": "cat_procurement",
            "rnd": "cat_rnd",
            "research_and_development": "cat_rnd",
            "operations": "cat_operations",
            "o&m": "cat_operations",
            "operations_and_maintenance": "cat_operations",
        }
        cats.columns = [rename_map.get(c, c) for c in cats.columns]
        df = cats

    return df


def read_map(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    m = pd.read_csv(path)
    if "country" not in m.columns:
        raise SystemExit("map CSV must include 'country' and optionally 'region','alliance'")
    for c in ["region","alliance"]:
        if c not in m.columns: m[c] = ""
    m["country"] = m["country"].astype(str).str.strip()
    return m[["country","region","alliance"]]


def read_scenario(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    s = pd.read_csv(path)
    if not {"country","year","shock_pct"}.issubset(s.columns):
        raise SystemExit("scenario CSV must include: country, year, shock_pct")
    s["country"] = s["country"].astype(str).str.strip()
    s["year"] = s["year"].astype(str).str.strip()
    s["shock_pct"] = pd.to_numeric(s["shock_pct"], errors="coerce").fillna(0.0)
    return s


# ----------------------------- Core transforms -----------------------------

def compute_usd(df: pd.DataFrame, fx_invert: bool) -> pd.DataFrame:
    """Convert local currency to USD using fx_usd column."""
    d = df.copy()
    if fx_invert:
        # fx_usd is USD per 1 LCU → USD = LCU * fx
        d["defense_usd_n"] = d["defense_local"] * d["fx_usd"]
        for c in ["cat_personnel","cat_procurement","cat_rnd","cat_operations"]:
            if c in d.columns:
                d[c.replace("cat_", "usd_")] = d[c] * d["fx_usd"]
    else:
        # fx_usd is LCU per 1 USD → USD = LCU / fx
        d["defense_usd_n"] = d["defense_local"] / d["fx_usd"].replace(0, np.nan)
        for c in ["cat_personnel","cat_procurement","cat_rnd","cat_operations"]:
            if c in d.columns:
                d[c.replace("cat_", "usd_")] = d[c] / d["fx_usd"].replace(0, np.nan)
    return d


def deflate_to_real(df: pd.DataFrame, price_base: Optional[int]) -> pd.DataFrame:
    """Create real (constant-price) USD series from nominal using CPI/deflator if available."""
    d = df.copy()
    # Build price index where possible
    if "deflator" in d.columns and d["deflator"].notna().any():
        idx = d["deflator"].copy()
    elif "cpi_index" in d.columns and d["cpi_index"].notna().any():
        # cpi_index given base=100 → normalize to 1
        idx = d["cpi_index"] / 100.0
    else:
        # No price series; treat nominal as real
        d["defense_usd_r"] = d["defense_usd_n"]
        for c in ["usd_personnel","usd_procurement","usd_rnd","usd_operations"]:
            if c in d.columns:
                d[c.replace("usd_", "real_")] = d[c]
        return d

    if price_base is None:
        price_base = int(d["year"].max())

    # Compute deflator multiplier to convert to base-year dollars
    base_idx = d.loc[d["year"] == price_base, ["country", idx.name]].rename(columns={idx.name: "base_idx"})
    d = d.merge(base_idx, on="country", how="left")
    d["deflator_mult"] = d["base_idx"] / idx.replace(0, np.nan)

    d["defense_usd_r"] = d["defense_usd_n"] * d["deflator_mult"]
    for c in ["usd_personnel","usd_procurement","usd_rnd","usd_operations"]:
        if c in d.columns:
            d[c.replace("usd_", "real_")] = d[c] * d["deflator_mult"]
    return d


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["defense_gdp_pct"] = 100.0 * d["defense_usd_n"] / d["gdp_usd"].replace(0, np.nan)
    d["defense_pc_usd"] = d["defense_usd_n"] / d["population"].replace(0, np.nan)
    # Category shares (nominal)
    tot = d["defense_usd_n"].replace(0, np.nan)
    for c in [("usd_personnel","share_personnel_pct"),
              ("usd_procurement","share_procurement_pct"),
              ("usd_rnd","share_rnd_pct"),
              ("usd_operations","share_operations_pct")]:
        if c[0] in d.columns:
            d[c[1]] = 100.0 * d[c[0]] / tot
    # NATO “equipment >= 20%” heuristic: procurement share >= 20%
    if "share_procurement_pct" in d.columns:
        d["equip20_flag"] = (d["share_procurement_pct"] >= 20.0).astype(int)
    else:
        d["equip20_flag"] = np.nan
    # NATO 2% GDP flag
    d["nato2_flag"] = (d["defense_gdp_pct"] >= 2.0).astype(int)
    return d


def aggregates(df: pd.DataFrame, mapping: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if mapping is None:
        return None
    d = df.merge(mapping, on="country", how="left")
    grp = d.groupby(["year","region","alliance"], dropna=False).agg({
        "defense_usd_n":"sum","defense_usd_r":"sum","gdp_usd":"sum","population":"sum"
    }).reset_index()
    grp["defense_gdp_pct"] = 100.0 * grp["defense_usd_n"] / grp["gdp_usd"].replace(0, np.nan)
    grp["defense_pc_usd"] = grp["defense_usd_n"] / grp["population"].replace(0, np.nan)
    return grp


def yoy(series: pd.Series) -> pd.Series:
    return series.pct_change()


def rolling_cagr(values: pd.Series, window: int) -> float:
    v = values.dropna()
    if len(v) < max(2, window):
        return np.nan
    v = v.iloc[-window:]
    if v.iloc[0] <= 0 or v.iloc[-1] <= 0:
        return np.nan
    years = len(v) - 1
    return float((v.iloc[-1] / v.iloc[0]) ** (1/years) - 1)


def make_forecast(df: pd.DataFrame, years_ahead: int, window: int) -> pd.DataFrame:
    rows = []
    for c, sub in df.sort_values("year").groupby("country"):
        last_year = int(sub["year"].max())
        last_nom = float(sub.loc[sub["year"] == last_year, "defense_usd_n"].iloc[0])
        last_real = float(sub.loc[sub["year"] == last_year, "defense_usd_r"].iloc[0])
        cagr_nom = rolling_cagr(sub["defense_usd_n"], window)
        cagr_real = rolling_cagr(sub["defense_usd_r"], window)
        # fallbacks to simple YoY
        if np.isnan(cagr_nom):
            cagr_nom = yoy(sub.set_index("year")["defense_usd_n"]).iloc[-1]
        if np.isnan(cagr_real):
            cagr_real = yoy(sub.set_index("year")["defense_usd_r"]).iloc[-1]
        for k in range(1, years_ahead+1):
            y = last_year + k
            nom = last_nom * ((1 + (0 if np.isnan(cagr_nom) else cagr_nom)) ** k)
            real = last_real * ((1 + (0 if np.isnan(cagr_real) else cagr_real)) ** k)
            rows.append({"country": c, "year": y,
                         "defense_usd_n": nom, "defense_usd_r": real,
                         "method": f"CAGR{window}"})
    return pd.DataFrame(rows)


def apply_scenario(base_fc: pd.DataFrame, scen: Optional[pd.DataFrame]) -> pd.DataFrame:
    if scen is None or base_fc.empty:
        return base_fc.assign(scenario="baseline")
    out = base_fc.copy()
    out["shock_pct"] = 0.0
    for i, r in scen.iterrows():
        c = r["country"].upper()
        y = str(r["year"]).upper()
        mask_c = (out["country"].str.upper() == c) if c != "ALL" else np.ones(len(out), dtype=bool)
        if y == "ALL":
            mask_y = np.ones(len(out), dtype=bool)
        else:
            try:
                yy = int(y)
                mask_y = (out["year"] == yy)
            except Exception:
                mask_y = np.ones(len(out), dtype=bool)
        out.loc[mask_c & mask_y, "shock_pct"] += float(r["shock_pct"])
    out["defense_usd_n"] = out["defense_usd_n"] * (1.0 + out["shock_pct"]/100.0)
    out["defense_usd_r"] = out["defense_usd_r"] * (1.0 + out["shock_pct"]/100.0)
    out["scenario"] = "with_shocks"
    return out


# ----------------------------- Plotting -----------------------------

def make_plots(clean: pd.DataFrame, fc_b: pd.DataFrame, fc_s: pd.DataFrame, outdir: str):
    if plt is None:
        return
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    # Top 10 by latest nominal USD
    latest = (clean.sort_values("year").groupby("country").tail(1)
              .sort_values("defense_usd_n", ascending=False).head(10)["country"].tolist())
    for c in latest:
        d = clean[clean["country"] == c].sort_values("year")
        fig = plt.figure(figsize=(9, 5)); ax = plt.gca()
        ax.plot(d["year"], d["defense_usd_n"]/1e9, label="Nominal USD")
        ax.plot(d["year"], d["defense_usd_r"]/1e9, linestyle="--", label="Real USD")
        # add baseline forecast
        if not fc_b.empty:
            fb = fc_b[fc_b["country"] == c].sort_values("year")
            if not fb.empty:
                ax.plot(fb["year"], fb["defense_usd_n"]/1e9, label="Forecast (nominal)", alpha=0.8)
        if not fc_s.empty:
            fs = fc_s[fc_s["country"] == c].sort_values("year")
            if not fs.empty:
                ax.plot(fs["year"], fs["defense_usd_n"]/1e9, label="Forecast (scenario)", alpha=0.8)
        ax.set_title(f"{c}: Defense outlays"); ax.set_ylabel("USD bn")
        ax.legend(); plt.tight_layout()
        fig.savefig(os.path.join(outdir, "plots", f"{c}_defense_trend.png"), dpi=140); plt.close(fig)

    # NATO 2% compliance distribution (latest year)
    try:
        latest_year = int(clean["year"].max())
        lat = clean[clean["year"] == latest_year]
        if not lat.empty:
            fig2 = plt.figure(figsize=(9,5)); ax2 = plt.gca()
            lat.sort_values("defense_gdp_pct", ascending=True).plot.barh(
                x="country", y="defense_gdp_pct", ax=ax2, legend=False)
            ax2.axvline(2.0, linestyle="--", color="k")
            ax2.set_title(f"Defense %GDP in {latest_year} (2% line)"); ax2.set_xlabel("% of GDP")
            plt.tight_layout(); fig2.savefig(os.path.join(outdir, "plots", f"nato2pct_{latest_year}.png"), dpi=140); plt.close(fig2)
    except Exception:
        pass


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Defense budget analyzer & forecaster")
    ap.add_argument("--budgets", required=True, help="CSV with country-year defense data")
    ap.add_argument("--map", default=None, help="Optional country→region/alliance mapping CSV")
    ap.add_argument("--scenario", default=None, help="Optional scenario shocks CSV")
    ap.add_argument("--long-cats", action="store_true", help="Treat budgets CSV as long-form categories")
    ap.add_argument("--fx-invert", action="store_true", help="fx_usd is USD per 1 LCU (default LCU per USD)")
    ap.add_argument("--price-base", type=int, default=None, help="Base year for real terms (default: latest)")
    ap.add_argument("--forecast-years", type=int, default=5)
    ap.add_argument("--forecast-window", type=int, default=5)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default="./artifacts")
    args = ap.parse_args()

    cfg = Config(
        budgets_file=args.budgets,
        map_file=args.map,
        scenario_file=args.scenario,
        long_cats=bool(args.long_cats),
        fx_invert=bool(args.fx_invert),
        price_base=args.price_base,
        forecast_years=int(max(0, args.forecast_years)),
        forecast_window=int(max(2, args.forecast_window)),
        plot=bool(args.plot),
        outdir=ensure_outdir(args.outdir)
    )

    print(f"[INFO] Writing artifacts to: {cfg.outdir}")

    df0 = read_budgets(cfg.budgets_file, cfg.long_cats)
    mapping = read_map(cfg.map_file)
    scenario = read_scenario(cfg.scenario_file)

    # Pipeline
    d1 = compute_usd(df0, cfg.fx_invert)
    d2 = deflate_to_real(d1, cfg.price_base)
    clean = compute_ratios(d2)

    # Growth stats
    clean = clean.sort_values(["country","year"])
    clean["yoy_nominal"] = clean.groupby("country")["defense_usd_n"].pct_change()
    clean["yoy_real"] = clean.groupby("country")["defense_usd_r"].pct_change()

    # Save cleaned
    clean.to_csv(os.path.join(cfg.outdir, "cleaned.csv"), index=False)

    # Aggregates
    reg = aggregates(clean, mapping) if mapping is not None else None
    if reg is not None:
        reg.to_csv(os.path.join(cfg.outdir, "regional_aggregates.csv"), index=False)

    # Compliance table (latest year per country)
    latest = clean.sort_values("year").groupby("country").tail(1).copy()
    latest["nato2_flag"] = (latest["defense_gdp_pct"] >= 2.0).astype(int)
    if "share_procurement_pct" in latest.columns:
        latest["equip20_flag"] = (latest["share_procurement_pct"] >= 20.0).astype(int)
    latest[["country","year","defense_usd_n","defense_gdp_pct","defense_pc_usd","nato2_flag","equip20_flag"]].to_csv(
        os.path.join(cfg.outdir, "compliance.csv"), index=False
    )

    # Forecasts
    fc_base = make_forecast(clean[["country","year","defense_usd_n","defense_usd_r"]], cfg.forecast_years, cfg.forecast_window)
    fc_scen = apply_scenario(fc_base, scenario)

    fc_base.to_csv(os.path.join(cfg.outdir, "forecasts_baseline.csv"), index=False)
    fc_scen.to_csv(os.path.join(cfg.outdir, "forecasts_scenario.csv"), index=False)

    if cfg.plot:
        make_plots(clean, fc_base, fc_scen, cfg.outdir)
        print("[OK] Plots saved to:", os.path.join(cfg.outdir, "plots"))

    # Console snapshot
    top = latest.sort_values("defense_usd_n", ascending=False).head(12)
    print("\n=== Latest snapshot (top 12 by nominal USD) ===")
    with pd.option_context("display.width", 120):
        cols = ["country","year","defense_usd_n","defense_gdp_pct","defense_pc_usd"]
        print(top[cols].assign(
            defense_usd_n=lambda x: (x["defense_usd_n"]/1e9).round(2)
        ).rename(columns={"defense_usd_n":"defense_usd_bn"}).to_string(index=False))

    print("\nDone. Files written to:", cfg.outdir)


if __name__ == "__main__":
    main()