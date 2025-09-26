#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deglobalization.py — Trade fragmentation / reshoring impact engine
------------------------------------------------------------------

What this does
==============
Given bilateral trade flows, tariffs / shipping frictions, simple cost proxies
(wage, energy), and a portfolio map (issuer→country/sector), this script:

1) Builds per country/sector *exposure* metrics
   - Trade Openness (Exports+Imports)/GDP
   - Import Concentration (HHI) and Fragmentation Index
   - Domestic vs External input dependence (from import_shares or IO proxy)
2) Applies *deglobalization scenarios* (tariff hikes, shipping/lead-time shocks,
   export controls, re/friend-shoring shares) to produce:
   - Δ import prices and volumes (Armington-style)
   - CPI impulse via import share × pass-through
   - Sector margin compression from higher input costs & reshoring cost gap
   - GDP impact via trade elasticity & demand/price effects
3) Maps to portfolio issuers and estimates ΔEBITDA, ΔValue (DCF-style), ΔCredit spread
4) Saves tidy CSVs + a concise JSON summary

Inputs (CSV; headers are case-insensitive & flexible)
-----------------------------------------------------
--trade trade.csv         REQUIRED (annual or quarterly)
  Columns: year, exporter, importer, sector, value_usd

--tariffs tariffs.csv     OPTIONAL
  Columns: exporter, importer, sector, tariff_pct      (ad valorem, %)

--import_shares imports.csv OPTIONAL (by country/sector)
  Columns: country, sector, import_share_pct           (of input costs or sales)

--costs costs.csv         OPTIONAL (country cost proxies)
  Columns: country, wage_idx, energy_idx               (indexes ~100=base)

--logistics logistics.csv OPTIONAL (route/region costs)
  Columns: origin, dest, shipping_cost_idx, lead_time_days

--gdp gdp.csv             OPTIONAL
  Columns: year, country, gdp_usd

--alliances allies.csv    OPTIONAL (friend-shoring blocs)
  Columns: country, bloc

--portfolio portfolio.csv OPTIONAL (to map finance impacts)
  Columns: issuer_id, name, country, sector, weight (or position_usd),
           [revenue_usd, ebitda_margin_pct, wacc_pct, growth_pct]

--scenarios scenarios.csv OPTIONAL
  Columns: scenario, key, value
  Key examples:
    tariff.change.ALL = +5                         (pp increase)
    tariff.change.CN->US = +20
    nontariff.cost.multiplier = 1.15               (global multiplicative)
    shipping.multiplier.ASIA->EUROPE = 1.5
    leadtime.multiplier = 1.2
    reshoring.share.US.MANUFACTURING = 0.10        (10% of imports reshore)
    friendshoring.share.US.ELECTRONICS = 0.20      (to same-bloc partners)
    export_controls.block.CN.SEMIS = 0.15          (cut of exports share)
    fx_passthrough = 0.25                          (CPI pass-through)
    trade_elasticity = -1.5                        (Armington)
    cpi_passthrough = 0.6                          (% of import price to CPI)
    value_wacc = 8                                 (portfolio default WACC %)
    value_g = 2                                    (portfolio default growth %)

CLI
---
--start 2015 --end 2025 --scenario BASE --outdir out_deglob
(Use `--scenario` to pick a scenario name from scenarios.csv; if absent, BASELINE.)

Outputs
-------
- exposures_country.csv        Openness, HHI, Fragmentation, import share
- exposures_sector.csv         Same, by country×sector
- impacts_country.csv          ΔCPI (pp), ΔGDP (%), ΔToT proxy, price/volume shocks
- impacts_sector.csv           Sector margin Δpp, cost drivers, volume changes
- portfolio_impacts.csv        Issuer-level ΔEBITDA, ΔValue, ΔSpread (if portfolio provided)
- scenario_applied.csv         Tidy list of effective shocks (resolved)
- summary.json                 Headline metrics + top/bottom countries & sectors
- config.json                  Reproducibility dump

DISCLAIMER: Research tool with simplifying assumptions; not investment advice.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- utilities -----------------------------

def ncol(df: pd.DataFrame, target: str) -> Optional[str]:
    t = target.lower()
    for c in df.columns:
        if c.lower() == t: return c
    for c in df.columns:
        if t in c.lower(): return c
    return None

def ensure_dir(d: str) -> Path:
    p = Path(d); p.mkdir(parents=True, exist_ok=True); return p

def to_year(s: pd.Series) -> pd.Series:
    return pd.to_numeric(pd.to_datetime(s, errors="coerce").dt.year.fillna(pd.to_numeric(s, errors="coerce")), errors="coerce").astype("Int64")

def pct_to_mult(x: float) -> float:
    return 1.0 + (x / 100.0)

def safe_num(x):
    return pd.to_numeric(x, errors="coerce")

def hhi(shares: pd.Series) -> float:
    s = shares.fillna(0.0).astype(float)
    if s.sum() <= 0: return np.nan
    p = s / s.sum()
    return float((p**2).sum())

def theil_index(shares: pd.Series) -> float:
    s = shares.fillna(0.0).astype(float)
    if s.sum() <= 0: return np.nan
    p = s / s.sum()
    p = p[p>0]
    return float((p * np.log(p * len(p))).sum())

def zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or not np.isfinite(sd): return s*0
    return (s - m)/sd


# ----------------------------- loaders -----------------------------

def load_trade(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ren = {(ncol(df,"year") or "year"):"year",
           (ncol(df,"exporter") or "exporter"):"exporter",
           (ncol(df,"importer") or "importer"):"importer",
           (ncol(df,"sector") or "sector"):"sector",
           (ncol(df,"value_usd") or ncol(df,"trade_usd") or "value"):"value"}
    df = df.rename(columns=ren)
    df["year"] = to_year(df["year"])
    for c in ["exporter","importer","sector"]:
        df[c] = df[c].astype(str).str.upper().str.strip()
    df["value"] = safe_num(df["value"]).fillna(0.0).clip(lower=0.0)
    return df

def load_tariffs(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["exporter","importer","sector","tariff_pct"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"exporter") or "exporter"):"exporter",
           (ncol(df,"importer") or "importer"):"importer",
           (ncol(df,"sector") or "sector"):"sector",
           (ncol(df,"tariff_pct") or "tariff_pct"):"tariff_pct"}
    df = df.rename(columns=ren)
    for c in ["exporter","importer","sector"]:
        df[c] = df[c].astype(str).str.upper().str.strip()
    df["tariff_pct"] = safe_num(df["tariff_pct"]).fillna(0.0)
    return df

def load_import_shares(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["country","sector","import_share_pct"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"country") or "country"):"country",
           (ncol(df,"sector") or "sector"):"sector",
           (ncol(df,"import_share_pct") or "import_share_pct"):"import_share_pct"}
    df = df.rename(columns=ren)
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    df["sector"] = df["sector"].astype(str).str.upper().str.strip()
    df["import_share_pct"] = safe_num(df["import_share_pct"]).fillna(0.0)
    return df

def load_costs(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["country","wage_idx","energy_idx"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"country") or "country"):"country",
           (ncol(df,"wage_idx") or "wage_idx"):"wage_idx",
           (ncol(df,"energy_idx") or "energy_idx"):"energy_idx"}
    df = df.rename(columns=ren)
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    for c in ["wage_idx","energy_idx"]:
        df[c] = safe_num(df[c]).fillna(100.0)
    return df

def load_logistics(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["origin","dest","shipping_cost_idx","lead_time_days"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"origin") or "origin"):"origin",
           (ncol(df,"dest") or "dest"):"dest",
           (ncol(df,"shipping_cost_idx") or "shipping_cost_idx"):"shipping_cost_idx",
           (ncol(df,"lead_time_days") or "lead_time_days"):"lead_time_days"}
    df = df.rename(columns=ren)
    for c in ["origin","dest"]:
        df[c] = df[c].astype(str).str.upper().str.strip()
    df["shipping_cost_idx"] = safe_num(df["shipping_cost_idx"]).fillna(100.0)
    df["lead_time_days"] = safe_num(df["lead_time_days"]).fillna(np.nan)
    return df

def load_gdp(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["year","country","gdp_usd"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"year") or "year"):"year",
           (ncol(df,"country") or "country"):"country",
           (ncol(df,"gdp_usd") or "gdp_usd"):"gdp_usd"}
    df = df.rename(columns=ren)
    df["year"] = to_year(df["year"])
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    df["gdp_usd"] = safe_num(df["gdp_usd"]).fillna(np.nan)
    return df

def load_allies(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["country","bloc"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"country") or "country"):"country",
           (ncol(df,"bloc") or "bloc"):"bloc"}
    df = df.rename(columns=ren)
    df["country"] = df["country"].astype(str).str.upper().str.strip()
    df["bloc"] = df["bloc"].astype(str).str.upper().str.strip()
    return df

def load_portfolio(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame()
    df = pd.read_csv(path)
    ren = {(ncol(df,"issuer_id") or "issuer_id"):"issuer_id",
           (ncol(df,"name") or "name"):"name",
           (ncol(df,"country") or "country"):"country",
           (ncol(df,"sector") or "sector"):"sector",
           (ncol(df,"weight") or "weight"):"weight",
           (ncol(df,"position_usd") or "position_usd"):"mv_usd",
           (ncol(df,"revenue_usd") or "revenue_usd"):"revenue",
           (ncol(df,"ebitda_margin_pct") or "ebitda_margin_pct"):"margin",
           (ncol(df,"wacc_pct") or "wacc_pct"):"wacc",
           (ncol(df,"growth_pct") or "growth_pct"):"g"}
    df = df.rename(columns=ren)
    for c in ["issuer_id","name","country","sector"]:
        if c in df.columns: df[c] = df[c].astype(str).str.upper().str.strip()
    for c in ["weight","mv_usd","revenue","margin","wacc","g"]:
        if c in df.columns: df[c] = safe_num(df[c])
    if "weight" not in df.columns or df["weight"].isna().all():
        if "mv_usd" in df.columns and df["mv_usd"].sum()>0:
            df["weight"] = df["mv_usd"]/df["mv_usd"].sum()
        else:
            df["weight"] = 1.0/len(df)
    return df

def load_scenarios(path: Optional[str]) -> pd.DataFrame:
    if not path: return pd.DataFrame(columns=["scenario","key","value"])
    df = pd.read_csv(path)
    ren = {(ncol(df,"scenario") or "scenario"):"scenario",
           (ncol(df,"key") or "key"):"key",
           (ncol(df,"value") or "value"):"value"}
    df = df.rename(columns=ren)
    return df


# ----------------------------- exposure metrics -----------------------------

def build_exposures(trade: pd.DataFrame, gdp: pd.DataFrame, import_shares: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Country Openness
    T = trade.copy()
    last_year = int(T["year"].dropna().max()) if not T["year"].dropna().empty else None
    country_exports = T.groupby(["year","exporter"], as_index=False)["value"].sum().rename(columns={"exporter":"country","value":"exports"})
    country_imports = T.groupby(["year","importer"], as_index=False)["value"].sum().rename(columns={"importer":"country","value":"imports"})
    CO = pd.merge(country_exports, country_imports, on=["year","country"], how="outer").fillna(0.0)
    if not gdp.empty:
        CO = CO.merge(gdp, on=["year","country"], how="left")
        CO["openness"] = (CO["exports"] + CO["imports"]) / CO["gdp_usd"].replace(0,np.nan)
    else:
        CO["openness"] = np.nan
    # Partner concentration (HHI) at last_year
    last = T[T["year"]==last_year] if last_year else T
    hhi_imp = last.groupby(["importer","exporter"], as_index=False)["value"].sum()
    hhi_list = []
    for c, g in hhi_imp.groupby("importer"):
        h = hhi(g.set_index("exporter")["value"])
        th = theil_index(g.set_index("exporter")["value"])
        hhi_list.append({"country": c, "import_hhi": h, "import_fragmentation": th})
    H = pd.DataFrame(hhi_list)
    country_exp = CO[CO["year"]==last_year].merge(H, on="country", how="left") if last_year else CO.merge(H, on="country", how="left")
    country_exp["import_share_inputs"] = np.nan
    if not import_shares.empty:
        s = (import_shares.groupby("country")["import_share_pct"].mean() / (100.0 if import_shares["import_share_pct"].max()>1 else 1.0))
        country_exp["import_share_inputs"] = country_exp["country"].map(s)

    # Sector exposures (by country×sector, imports & HHI of sources)
    sec_imp = last.groupby(["importer","sector","exporter"], as_index=False)["value"].sum()
    rows = []
    for (c, s), g in sec_imp.groupby(["importer","sector"]):
        rows.append({
            "country": c, "sector": s,
            "imports_usd": float(g["value"].sum()),
            "import_hhi": hhi(g.set_index("exporter")["value"]),
            "import_fragmentation": theil_index(g.set_index("exporter")["value"])
        })
    SE = pd.DataFrame(rows)
    # Merge import share for sector if available
    if not import_shares.empty:
        ms = import_shares.copy()
        ms["import_share"] = ms["import_share_pct"] / (100.0 if ms["import_share_pct"].max()>1 else 1.0)
        SE = SE.merge(ms[["country","sector","import_share"]], on=["country","sector"], how="left")
    else:
        SE["import_share"] = np.nan
    return country_exp.sort_values("country"), SE.sort_values(["country","sector"])


# ----------------------------- scenario parsing -----------------------------

@dataclass
class Params:
    trade_elasticity: float = -1.5      # Armington ε
    cpi_passthrough: float = 0.6        # share of import price shock into CPI
    fx_passthrough: float = 0.25        # unused hook
    value_wacc: float = 8.0
    value_g: float = 2.0

def parse_scenario(scen_df: pd.DataFrame, pick: Optional[str]) -> Tuple[Dict, Params, pd.DataFrame]:
    if scen_df.empty:
        return {}, Params(), pd.DataFrame(columns=["key","value","note"])
    if pick:
        S = scen_df[scen_df["scenario"].astype(str)==pick]
        if S.empty:
            S = scen_df.copy()
    else:
        S = scen_df.copy()
    shocks = {}
    meta_rows = []
    p = Params()
    for _, r in S.iterrows():
        k = str(r["key"]).strip()
        v_str = str(r["value"]).strip()
        try:
            v = float(v_str)
        except Exception:
            v = v_str
        shocks[k] = v
        meta_rows.append({"key": k, "value": v, "note": ""})
        kl = k.lower()
        if kl == "trade_elasticity":
            p.trade_elasticity = float(v)
        elif kl == "cpi_passthrough":
            p.cpi_passthrough = float(v)
        elif kl == "fx_passthrough":
            p.fx_passthrough = float(v)
        elif kl == "value_wacc":
            p.value_wacc = float(v)
        elif kl == "value_g":
            p.value_g = float(v)
    return shocks, p, pd.DataFrame(meta_rows)


# ----------------------------- shock building -----------------------------

def apply_tariff_shocks(base_tariffs: pd.DataFrame, shocks: Dict) -> pd.DataFrame:
    T = base_tariffs.copy()
    if T.empty:
        T = pd.DataFrame(columns=["exporter","importer","sector","tariff_pct"])
    # global change
    for k, v in shocks.items():
        kl = k.lower()
        if kl == "tariff.change.all":
            if T.empty:
                # create wildcard table with 0 baseline → apply global pp
                pass
            else:
                T["tariff_pct"] = T["tariff_pct"] + float(v)
        elif kl.startswith("tariff.change.") and "->" in k:
            pair = k.split(".", 2)[2]
            frm, to = pair.split("->")
            mask = (T["exporter"].astype(str).str.upper()==frm.upper()) & (T["importer"].astype(str).str.upper()==to.upper())
            if T[mask].empty:
                # add row wildcard sector
                T = pd.concat([T, pd.DataFrame([{"exporter": frm.upper(), "importer": to.upper(), "sector":"*", "tariff_pct": float(v)}])], ignore_index=True)
            else:
                T.loc[mask, "tariff_pct"] = T.loc[mask, "tariff_pct"] + float(v)
    return T

def logistics_multipliers(shocks: Dict) -> Tuple[float, Dict[Tuple[str,str], float], float]:
    mult_global = 1.0
    lead_mult = 1.0
    by_route = {}
    for k, v in shocks.items():
        kl = k.lower()
        if kl == "nontariff.cost.multiplier":
            mult_global = float(v)
        elif kl.startswith("shipping.multiplier.") and "->" in k:
            route = k.split(".",2)[2]
            o, d = route.split("->")
            by_route[(o.upper(), d.upper())] = float(v)
        elif kl == "leadtime.multiplier":
            lead_mult = float(v)
    return mult_global, by_route, lead_mult

def reshoring_friendshoring_rules(shocks: Dict) -> List[Tuple[str,str,float,str]]:
    """
    Returns list of (country, sector, share, kind) with kind ∈ {"RESHORE","FRIEND"}.
    """
    out = []
    for k, v in shocks.items():
        kl = k.lower()
        if kl.startswith("reshoring.share."):
            _, _, tail = k.partition(".share.")
            country, sector = tail.split(".", 1)
            out.append((country.upper(), sector.upper(), float(v), "RESHORE"))
        elif kl.startswith("friendshoring.share."):
            _, _, tail = k.partition(".share.")
            country, sector = tail.split(".", 1)
            out.append((country.upper(), sector.upper(), float(v), "FRIEND"))
    return out

def export_controls_rules(shocks: Dict) -> List[Tuple[str,str,float]]:
    """
    Returns list of (block_country, sector, share_cut_of_exports).
    """
    out = []
    for k, v in shocks.items():
        kl = k.lower()
        if kl.startswith("export_controls.block."):
            _, _, tail = k.partition("block.")
            ctry, sector = tail.split(".", 1)
            out.append((ctry.upper(), sector.upper(), float(v)))
    return out


# ----------------------------- impacts engine -----------------------------

def effective_import_price_change(
    exporter: str, importer: str, sector: str,
    tariffs: pd.DataFrame, ship_cost_mult: float, route_mult: Dict[Tuple[str,str], float],
    base_ship_idx: float, global_mult: float
) -> float:
    # Δ price ≈ Δ tariff (pp) + Δ shipping multiplicative deviation − 1
    # If tariff table has wildcard sector "*", apply.
    dtar_pp = 0.0
    if not tariffs.empty:
        rows = tariffs[
            ((tariffs["exporter"]==exporter) & (tariffs["importer"]==importer) & (tariffs["sector"]==sector)) |
            ((tariffs["exporter"]==exporter) & (tariffs["importer"]==importer) & (tariffs["sector"]=="*"))
        ]
        if not rows.empty:
            dtar_pp = float(rows["tariff_pct"].mean())  # we treat numbers in shocks as *changes* vs baseline
    route = (exporter, importer)
    rmult = route_mult.get(route, 1.0)
    # shipping idx base→shock: (global_mult * ship_cost_mult * rmult) - 1
    dship = (global_mult * ship_cost_mult * rmult) - 1.0
    return (dtar_pp/100.0) + dship

def run_country_sector_impacts(
    trade: pd.DataFrame, tariffs_applied: pd.DataFrame, logistics: pd.DataFrame, costs: pd.DataFrame,
    import_shares: pd.DataFrame, allies: pd.DataFrame, rules_rf: List[Tuple[str,str,float,str]],
    rules_exports: List[Tuple[str,str,float]], params: Params
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (country_impacts, sector_impacts).
    """
    # Prepare logistics maps
    base_ship_idx = 100.0
    global_mult, route_mult, lead_mult = logistics_multipliers(shocks={})
    # Note: the scenario-specific multipliers will be passed as price_change argument (we'll call again with shocks later)
    # For internal calc, we take actual multipliers from caller; here we just rely on params at call site.

    # Import share table
    IS = import_shares.copy()
    IS["import_share"] = IS["import_share_pct"] / (100.0 if (IS["import_share_pct"].max() or 0)>1 else 1.0)

    # Start with last available year
    lastY = int(trade["year"].dropna().max()) if not trade.empty else None
    TR = trade[trade["year"]==lastY] if lastY else trade

    # Build shipping maps if provided
    ship_idx = {}
    if not logistics.empty:
        for _, r in logistics.iterrows():
            ship_idx[(str(r["origin"]).upper(), str(r["dest"]).upper())] = float(r["shipping_cost_idx"]) if pd.notna(r["shipping_cost_idx"]) else 100.0

    # For friend-shoring, map countries to bloc
    bloc = allies.set_index("country")["bloc"] if not allies.empty else pd.Series(dtype=str)

    # Aggregate structures
    imp_rows = []
    sec_rows = []

    # Pre-aggregate imports by importer×sector×exporter
    G = TR.groupby(["importer","sector","exporter"], as_index=False)["value"].sum()

    # Precompute country-level imports and import shares (inputs)
    imp_by_country = G.groupby(["importer"])["value"].sum().to_dict()

    # Iterate import relationships
    for (imp, sec), g in G.groupby(["importer","sector"]):
        total_imp_usd = float(g["value"].sum())
        # baseline import share for sector
        base_imp_share = np.nan
        if not IS.empty:
            row = IS[(IS["country"]==imp) & (IS["sector"]==sec)]
            if not row.empty:
                base_imp_share = float(row["import_share"].iloc[0])
        # sector cost uplift accumulator
        sector_cost_uplift = 0.0
        sector_volume_change = 0.0
        reshored_cost_uplift = 0.0

        for _, r in g.iterrows():
            exp = r["exporter"]
            val = float(r["value"])
            # Price change from tariffs + shipping (we assume the passed tariffs already encode Δpp from baseline)
            # We'll fill in multipliers later from outer call; for now assume they are attributes on tariffs_applied
            # We stored only tariff deltas; for shipping we approximate global + route multipliers via attributes:
            ship_base = ship_idx.get((exp, imp), base_ship_idx)
            global_mult = getattr(tariffs_applied, "_global_mult", 1.0)   # injected by caller
            route_mult = getattr(tariffs_applied, "_route_mult", {})
            price_chg = effective_import_price_change(exp, imp, sec, tariffs_applied, 1.0, route_mult, ship_base, global_mult)

            # Export controls: if exporter blocked in sector, cut volume share
            cut = 0.0
            for (blk_ctry, blk_sec, share_cut) in rules_exports:
                if blk_ctry == exp and (blk_sec == sec or blk_sec=="*"):
                    cut = max(cut, share_cut)
            vol_after_controls = val * (1.0 - cut)

            # Friend/reshoring: portion shifts away
            resh_share = 0.0
            frnd_share = 0.0
            for (ctry, sct, share, kind) in rules_rf:
                if ctry == imp and (sct == sec or sct=="*"):
                    if kind == "RESHORE":
                        resh_share = max(resh_share, share)
                    elif kind == "FRIEND":
                        frnd_share = max(frnd_share, share)

            # Apply Armington volume response to price_chg (ΔQ/Q ≈ ε * (−ΔP/P))
            dQ_Q = params.trade_elasticity * price_chg
            vol_after = vol_after_controls * (1.0 + dQ_Q)
            # Reshoring portion: remove from imports → produce domestically with cost differential
            resh_vol = vol_after * resh_share
            friend_vol = vol_after * frnd_share

            # Domestic production cost premium proxy from country wage/energy vs exporter
            # If costs.csv given, use wage_idx/energy_idx; else assume domestic=110 vs foreign=100
            def cost_idx(country):
                if costs.empty: return 110.0
                row = costs[costs["country"]==country]
                if row.empty: return 110.0
                return float(0.6*row["wage_idx"].iloc[0] + 0.4*row["energy_idx"].iloc[0])
            domestic_idx = cost_idx(imp)
            foreign_idx = cost_idx(exp)
            cost_gap = (domestic_idx / max(1e-9, foreign_idx)) - 1.0  # >0 means domestic costlier

            # Uplift contribution = reshored share × cost_gap + remaining imports price_chg weighted by share
            base_share = val / max(1e-9, total_imp_usd)
            # Remaining (non-reshored, non-friendshored) imports still face price_chg; assume friend-shored imports face 50% of price_chg
            eff_price_chg = (price_chg * (1.0 - resh_share - frnd_share) + price_chg * 0.5 * frnd_share)
            sector_cost_uplift += base_share * eff_price_chg
            reshored_cost_uplift += base_share * (resh_share * max(0.0, cost_gap))
            # Volume change (as share of sector inputs): approximate by base_share * dQ/Q minus reshored share not counted as imports
            sector_volume_change += base_share * (dQ_Q - resh_share)  # friendshored stays in imports, counted above with lower price

        # Total input cost uplift for sector (imports pass-through proportionally to import share)
        import_cost_share = base_imp_share if base_imp_share==base_imp_share else 0.3  # default 30% if unknown
        total_cost_uplift = import_cost_share * (sector_cost_uplift + reshored_cost_uplift)
        # Sector margin delta (pp): assume COGS share = 70%; Δmargin ≈ − COGS_share × total_cost_uplift × 100
        cogs_share = 0.70
        delta_margin_pp = - cogs_share * total_cost_uplift * 100.0

        sec_rows.append({
            "country": imp, "sector": sec,
            "imports_usd": total_imp_usd,
            "import_cost_uplift_pct": (sector_cost_uplift+reshored_cost_uplift)*100.0,
            "import_price_component_pct": sector_cost_uplift*100.0,
            "reshoring_cost_component_pct": reshored_cost_uplift*100.0,
            "volume_change_share_inputs_pct": sector_volume_change*100.0,
            "import_share_inputs": import_cost_share,
            "delta_margin_pp": delta_margin_pp
        })

    SEC = pd.DataFrame(sec_rows)

    # Country CPI & GDP impacts (aggregate sectors)
    # CPI Δ ≈ cpi_passthrough × Σ_sector (import_share × price_uplift)
    CPI = (SEC.assign(w=lambda d: d["import_share_inputs"].fillna(d["import_share_inputs"].median()))
              .groupby("country")
              .apply(lambda g: params.cpi_passthrough * float((g["w"] * (g["import_cost_uplift_pct"]/100.0)).sum()))
              .reset_index(name="delta_cpi_pp"))

    # GDP Δ: rough: ΔGDP/GDP ≈ 0.5 * openness * (weighted volume change)  (sign negative if imports fall without perfect substitution)
    # Use exposures openness if available outside call; approximate with total imports as proxy
    vol_w = SEC.groupby("country")["volume_change_share_inputs_pct"].mean().fillna(0.0) / 100.0
    gdp_rows = []
    for c in vol_w.index:
        # openness proxy: imports / (imports + exports) ~ 0.5 if symmetric; fallback 0.3
        openness_proxy = 0.3
        gdp_rows.append({"country": c, "delta_gdp_pct": float(0.5 * openness_proxy * vol_w.loc[c] * 100.0)})
    GDP_IMP = pd.DataFrame(gdp_rows)

    COUNTRY = pd.merge(CPI, GDP_IMP, on="country", how="outer")
    return COUNTRY.sort_values("country"), SEC.sort_values(["country","sector"])


# ----------------------------- portfolio mapping -----------------------------

def value_delta_dcf(delta_ebitda: float, wacc_pct: float, g_pct: float) -> float:
    w = max(0.1, wacc_pct/100.0)  # avoid tiny denom
    g = max(0.0, g_pct/100.0)
    denom = max(1e-6, w - g)
    return float(delta_ebitda / denom)

def spread_delta_from_margin(delta_margin_pp: float, beta: float=15.0) -> float:
    return float(beta * delta_margin_pp)  # bps (negative margin → positive bps)

def map_portfolio_impacts(port: pd.DataFrame, sec_imp: pd.DataFrame, params: Params) -> pd.DataFrame:
    if port.empty or sec_imp.empty:
        return pd.DataFrame()
    M = port.copy()
    M = M.merge(sec_imp[["country","sector","delta_margin_pp","import_cost_uplift_pct"]], on=["country","sector"], how="left")
    # Fill reasonable defaults
    M["margin"] = M["margin"].fillna(15.0)
    M["revenue"] = M["revenue"].fillna(1e6)
    M["wacc"] = M["wacc"].fillna(params.value_wacc)
    M["g"] = M["g"].fillna(params.value_g)
    # EBITDA baseline
    base_mgn = M["margin"]/100.0
    base_ebitda = M["revenue"] * base_mgn
    # Shocked margin and EBITDA (apply delta_margin_pp)
    shocked_mgn = base_mgn + (M["delta_margin_pp"].fillna(0.0)/100.0)
    shocked_ebitda = M["revenue"] * shocked_mgn
    delta_ebitda = shocked_ebitda - base_ebitda
    # Value delta
    dV = [value_delta_dcf(de, w, g) for de, w, g in zip(delta_ebitda, M["wacc"], M["g"])]
    dSpr = [spread_delta_from_margin(dm) for dm in M["delta_margin_pp"].fillna(0.0)]
    out = M.copy()
    out["baseline_ebitda_usd"] = base_ebitda
    out["shocked_ebitda_usd"] = shocked_ebitda
    out["delta_ebitda_usd"] = delta_ebitda
    out["delta_value_usd"] = dV
    out["delta_spread_bps"] = dSpr
    return out


# ----------------------------- orchestration -----------------------------

@dataclass
class Config:
    trade: str
    tariffs: Optional[str]
    import_shares: Optional[str]
    costs: Optional[str]
    logistics: Optional[str]
    gdp: Optional[str]
    allies: Optional[str]
    portfolio: Optional[str]
    scenarios: Optional[str]
    scenario: Optional[str]
    start: Optional[int]
    end: Optional[int]
    outdir: str

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deglobalization engine — exposures, scenarios, and portfolio impacts")
    ap.add_argument("--trade", required=True)
    ap.add_argument("--tariffs", default="")
    ap.add_argument("--import_shares", default="")
    ap.add_argument("--costs", default="")
    ap.add_argument("--logistics", default="")
    ap.add_argument("--gdp", default="")
    ap.add_argument("--allies", default="")
    ap.add_argument("--portfolio", default="")
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--scenario", default="")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--outdir", default="out_deglob")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    # Load
    TRADE = load_trade(args.trade)
    if args.start:
        TRADE = TRADE[TRADE["year"] >= args.start]
    if args.end:
        TRADE = TRADE[TRADE["year"] <= args.end]
    TARIFFS = load_tariffs(args.tariffs)
    IMPORTS = load_import_shares(args.import_shares)
    COSTS   = load_costs(args.costs)
    LOGI    = load_logistics(args.logistics)
    GDP     = load_gdp(args.gdp)
    ALLIES  = load_allies(args.allies)
    PORT    = load_portfolio(args.portfolio) if args.portfolio else pd.DataFrame()
    SCENDF  = load_scenarios(args.scenarios) if args.scenarios else pd.DataFrame()

    # Exposures
    EXP_C, EXP_S = build_exposures(TRADE, GDP, IMPORTS)
    EXP_C.to_csv(outdir / "exposures_country.csv", index=False)
    EXP_S.to_csv(outdir / "exposures_sector.csv", index=False)

    # Scenario parsing
    shocks, params, scen_applied = parse_scenario(SCENDF, args.scenario or None)

    # Tariff application
    TAR_SHOCK = apply_tariff_shocks(TARIFFS, shocks)

    # Logistics multipliers
    global_mult, route_mult, lead_mult = logistics_multipliers(shocks)
    # attach as attributes to tariff frame for downstream use
    TAR_SHOCK._global_mult = global_mult
    TAR_SHOCK._route_mult = route_mult

    # RF rules and export controls
    RF = reshoring_friendshoring_rules(shocks)
    EXC = export_controls_rules(shocks)

    # Impacts engine
    IMP_C, IMP_S = run_country_sector_impacts(
        trade=TRADE, tariffs_applied=TAR_SHOCK, logistics=LOGI, costs=COSTS,
        import_shares=IMPORTS, allies=ALLIES, rules_rf=RF, rules_exports=EXC, params=params
    )
    IMP_C.to_csv(outdir / "impacts_country.csv", index=False)
    IMP_S.to_csv(outdir / "impacts_sector.csv", index=False)

    # Portfolio mapping
    PORT_IMP = map_portfolio_impacts(PORT, IMP_S, params) if not PORT.empty else pd.DataFrame()
    if not PORT_IMP.empty:
        PORT_IMP.to_csv(outdir / "portfolio_impacts.csv", index=False)

    # Scenario table dump
    if not scen_applied.empty:
        scen_applied.to_csv(outdir / "scenario_applied.csv", index=False)

    # Summary
    top_cpi = IMP_C.nlargest(5, "delta_cpi_pp", keep="all")[["country","delta_cpi_pp"]].to_dict(orient="records") if not IMP_C.empty else []
    bot_cpi = IMP_C.nsmallest(5, "delta_cpi_pp", keep="all")[["country","delta_cpi_pp"]].to_dict(orient="records") if not IMP_C.empty else []
    top_marg = IMP_S.nsmallest(5, "delta_margin_pp", keep="all")[["country","sector","delta_margin_pp"]].to_dict(orient="records") if not IMP_S.empty else []
    sumy = {
        "scenario": args.scenario or "BASELINE",
        "trade_years": {"min": int(TRADE["year"].min()) if not TRADE.empty else None,
                        "max": int(TRADE["year"].max()) if not TRADE.empty else None},
        "top_cpi_up": top_cpi,
        "top_margin_pressure": top_marg,
        "portfolio_delta_value_total_usd": float(PORT_IMP["delta_value_usd"].sum()) if not PORT_IMP.empty else None
    }
    (outdir / "summary.json").write_text(json.dumps(sumy, indent=2))

    # Config dump
    cfg = asdict(Config(
        trade=args.trade, tariffs=(args.tariffs or None), import_shares=(args.import_shares or None),
        costs=(args.costs or None), logistics=(args.logistics or None), gdp=(args.gdp or None),
        allies=(args.allies or None), portfolio=(args.portfolio or None),
        scenarios=(args.scenarios or None), scenario=(args.scenario or None),
        start=(args.start or None), end=(args.end or None), outdir=args.outdir
    ))
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Console
    print("== Deglobalization Engine ==")
    print(f"Scenario: {args.scenario or 'BASELINE'} | Countries: {EXP_C['country'].nunique()} | Country CPI median Δ: {IMP_C['delta_cpi_pp'].median() if not IMP_C.empty else np.nan:+.2f} pp")
    if not PORT_IMP.empty:
        dv = PORT_IMP["delta_value_usd"].sum()
        print(f"Portfolio ΔValue: {dv:,.0f} USD")
    print("Outputs in:", outdir.resolve())


if __name__ == "__main__":
    main()
