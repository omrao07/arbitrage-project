#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tourism_visa_flows.py

Event study for tourism visa policy shocks:
- e-visa launches, visa-on-arrival (VOA), fee changes, bans/restrictions, multi-entry relaxations,
  biometric/appointment rules, processing-time shifts, reciprocity changes.

Inputs
------
--events-file (CSV, required) with at least 'date'.
Optional columns to strengthen analysis:
  'country'            : the destination issuing the policy (e.g., Japan)
  'flow'               : 'inbound' (tourists coming into country) or 'outbound' (nationals going out)
  'type'               : e.g., 'evisa_launch','voa','ban','restriction','fee_hike','fee_cut','multi_entry'
  'tickers'            : directly impacted equities (semicolon/comma separated)
  'roles'              : optional roles aligned to tickers (same order), e.g., 'Airline;Hotel;Airport;OTA'
  'markets'            : semicolon/comma list of source markets targeted (e.g., 'China;India;US')
  'market_weights'     : same-length weights (percent shares) for the listed markets (e.g., '35;25;15')
  'shock_points'       : numeric size of shock (e.g., +1 liberalization to -1 restriction; or bps of fee change)
  'expected_uplift_pct': expected inbound change in arrivals (%), if you have it
  'season'             : optional season/month label (e.g., 'GoldenWeek','Diwali','PeakSummer')
  'notes'              : free text

--map-file (CSV, optional) to auto-attach tickers by geography/flow/type/role/market:
  required: 'tickers'
  optional filters: 'country','flow','type','role','market'
  Blank values act as wildcards; 'Global' applies globally. Multiple tickers allowed per row.

What it does
------------
1) Loads events; resolves tickers per event (explicit ∪ mapped by country/flow/type/market/role).
2) Downloads daily prices for all unique tickers + chosen benchmark (yfinance).
3) Computes abnormal returns (benchmark-adjusted, or market-model OLS with --market-model).
4) Builds event windows [-tpre, +tpost]; exports:
   - prices.csv, daily_returns.csv
   - event_window_{pre}_{post}.csv (panel of AR)
   - AAR.csv, CAAR.csv, per_event_CAR.csv, summary_stats.csv
   - roles_by_event.csv (if roles provided)
5) Optional cross-section OLS of CAR on shock size, expected uplift, weighted market exposure, and dummies.
6) Saves PNG plots (AAR/CAAR, heatmap) with --plot.

Usage
-----
python tourism_visa_flows.py \
  --events-file events.csv \
  --map-file mapping.csv \
  --benchmark ^N225 \
  --tpre 5 --tpost 10 \
  --market-model --plot

Dependencies
------------
pip install yfinance pandas numpy matplotlib statsmodels
"""

import argparse
import os
import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Set

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install deps: pip install yfinance pandas numpy matplotlib statsmodels")

# Optional libs
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False


# ---------------------------- Config ----------------------------

@dataclass
class Config:
    events_file: str
    map_file: Optional[str]
    benchmark: str
    start: str
    end: Optional[str]
    tpre: int
    tpost: int
    use_market_model: bool
    plot: bool
    outdir: str


# ---------------------------- IO / helpers ----------------------------

def ensure_outdir(base: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, f"tourism_visa_flows_{ts}")
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    return outdir


def _split_list(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise SystemExit("events-file must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Normalize expected columns
    for col in [
        "country","flow","type","tickers","roles","markets","market_weights",
        "shock_points","expected_uplift_pct","season","notes"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce numerics
    def _to_float(x):
        if pd.isna(x): return np.nan
        try:
            s = str(x).strip().replace("%","")
            return float(s)
        except Exception:
            return np.nan

    df["shock_points"]        = df["shock_points"].apply(_to_float)
    df["expected_uplift_pct"] = df["expected_uplift_pct"].apply(_to_float)

    # Parse markets/weights to a canonical form
    df["markets_list"] = df["markets"].apply(_split_list)
    def _weights_list(s):
        arr = _split_list(s)
        out = []
        for a in arr:
            try:
                out.append(float(str(a).replace("%","")))
            except Exception:
                out.append(np.nan)
        return out
    df["market_weights_list"] = df["market_weights"].apply(_weights_list)

    return df.sort_values("date").reset_index(drop=True)


def load_map(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    m = pd.read_csv(path)
    if "tickers" not in m.columns:
        raise SystemExit("map-file must include a 'tickers' column; optional: 'country','flow','type','role','market'")
    for col in ["country","flow","type","role","market"]:
        if col not in m.columns:
            m[col] = np.nan
    return m


def resolve_event_tickers(ev: pd.Series, mapper: Optional[pd.DataFrame]) -> List[str]:
    """Union of explicit tickers in events.csv and inferred via map-file by country/flow/type/market/role."""
    explicit = set(_split_list(ev.get("tickers","")))
    if mapper is None:
        return sorted(list(explicit))

    cand = mapper.copy()

    # Country (exact or Global/blank)
    ctry = str(ev.get("country","")).strip().lower()
    cand = cand[
        (cand["country"].fillna("").str.lower().isin(["","global"])) |
        (cand["country"].fillna("").str.lower() == ctry)
    ]
    # Flow (inbound/outbound) filter if provided
    flw = str(ev.get("flow","")).strip().lower()
    if flw:
        cand = cand[(cand["flow"].fillna("").str.lower().isin([""])) |
                    (cand["flow"].fillna("").str.lower() == flw)]
    # Type filter
    typ = str(ev.get("type","")).strip().lower()
    if typ:
        cand = cand[(cand["type"].fillna("").str.lower().isin([""])) |
                    (cand["type"].fillna("").str.lower() == typ)]
    # Market filter: if event targets specific source markets, include rows whose 'market' is in that set or blank
    mkts = set([m.lower() for m in ev.get("markets_list", [])])
    if mkts:
        cand = cand[(cand["market"].fillna("") == "") | (cand["market"].str.lower().isin(mkts))]

    mapped: Set[str] = set()
    for _, row in cand.iterrows():
        mapped |= set(_split_list(row.get("tickers","")))
    return sorted(list(explicit | mapped))


def roles_table(events: pd.DataFrame, per_event_tickers: Dict[int, List[str]]) -> pd.DataFrame:
    """
    Build (event_id, date, ticker, role) table using optional inline 'roles' aligned with 'tickers'.
    If roles missing/misaligned, role = NA.
    """
    rows = []
    for i in range(len(events)):
        date = events.loc[i, "date"]
        tickers = per_event_tickers.get(i, [])
        roles = _split_list(events.loc[i, "roles"]) if isinstance(events.loc[i, "roles"], str) else []
        for j, t in enumerate(tickers):
            r = roles[j] if j < len(roles) else np.nan
            rows.append({"event_id": i+1, "date": date, "ticker": t, "role": r})
    return pd.DataFrame(rows)


# ---------------------------- Prices / Returns ----------------------------

def fetch_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in data.columns:
                out[t] = data[(t, "Close")]
            elif (t, "Adj Close") in data.columns:
                out[t] = data[(t, "Adj Close")]
    else:
        t = tickers[0]
        out[t] = data.get("Close", data.get("Adj Close"))
    px = pd.DataFrame(out).sort_index().ffill()
    missing = [t for t in tickers if t not in px.columns]
    for t in missing:
        print(f"[WARN] Missing prices for {t}; skipping.")
    return px.dropna(how="all", axis=1)


def daily_returns(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna(how="all")


def market_model_abnormal(ri: pd.Series, rb: pd.Series, est_win: int = 120) -> pd.Series:
    idx = ri.index.intersection(rb.index)
    ri, rb = ri.reindex(idx), rb.reindex(idx)
    ar = pd.Series(index=idx, dtype=float)
    if HAVE_SM:
        for i in range(len(idx)):
            if i < est_win:
                ar.iloc[i] = np.nan
                continue
            y = ri.iloc[i - est_win:i].values
            X = sm.add_constant(rb.iloc[i - est_win:i].values)
            res = sm.OLS(y, X).fit()
            alpha, beta = res.params[0], res.params[1]
            ar.iloc[i] = ri.iloc[i] - (alpha + beta * rb.iloc[i])
    else:
        for i in range(len(idx)):
            if i < est_win:
                ar.iloc[i] = np.nan
                continue
            r_i = ri.iloc[i - est_win:i].values
            r_b = rb.iloc[i - est_win:i].values
            var = np.var(r_b, ddof=0)
            beta = (np.cov(r_b, r_i, ddof=0)[0,1] / var) if var > 0 else 0.0
            alpha = r_i.mean() - beta * r_b.mean()
            ar.iloc[i] = ri.iloc[i] - (alpha + beta * rb.iloc[i])
    ar.name = "ar"
    return ar


def benchmark_adjusted(ri: pd.Series, rb: pd.Series) -> pd.Series:
    s = ri.align(rb, join="inner")
    return s[0] - s[1]


# ---------------------------- Event windows & Aggregation ----------------------------

def build_event_windows(ar_df: pd.DataFrame,
                        events: pd.DataFrame,
                        per_event_tickers: Dict[int, List[str]],
                        tpre: int, tpost: int) -> pd.DataFrame:
    """
    Panel of abnormal returns around events.
    Index = relative day [-tpre..tpost]
    Columns = MultiIndex (event_id, ticker)
    """
    idx_prices = ar_df.index
    panels = []
    for eid in range(len(events)):
        edate = events.loc[eid, "date"]
        if edate not in idx_prices:
            after = idx_prices[idx_prices >= edate]
            if after.empty:
                continue
            ed_trade = after[0]
        else:
            ed_trade = edate
        center = idx_prices.get_indexer([ed_trade])[0]
        start = max(0, center - tpre)
        end = min(len(idx_prices) - 1, center + tpost)
        tickers = [t for t in per_event_tickers.get(eid, []) if t in ar_df.columns]
        if not tickers:
            continue
        win = ar_df.loc[idx_prices[start]:idx_prices[end], tickers].copy()
        rel_idx = np.arange(start - center, end - center + 1)
        win.index = rel_idx
        win = win.reindex(np.arange(-tpre, tpost + 1))
        win.columns = pd.MultiIndex.from_product([[f"event_{eid+1}_{str(ed_trade.date())}"], tickers])
        panels.append(win)
    if not panels:
        raise RuntimeError("No event windows constructed (check dates/tickers).")
    return pd.concat(panels, axis=1)


def aggregate_aar_caar(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    AAR = panel.mean(axis=1).to_frame("AAR")
    CAAR = AAR.cumsum().rename(columns={"AAR": "CAAR"})
    return AAR, CAAR


def per_event_car(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in panel.columns:
        ev, tic = col
        series = panel[col].dropna()
        rows.append({"event": ev, "ticker": tic, "CAR": float(series.sum())})
    return pd.DataFrame(rows)


def ttest_zero(x: pd.Series) -> Tuple[float, float]:
    x = x.dropna()
    if len(x) < 3:
        return np.nan, np.nan
    m = x.mean()
    s = x.std(ddof=1)
    t = m / (s / math.sqrt(len(x))) if (s and not np.isnan(s) and s != 0) else np.nan
    try:
        from scipy.stats import t as student_t
        p = 2 * (1 - student_t.cdf(abs(t), df=len(x) - 1))
    except Exception:
        p = np.nan
    return t, p


# ---------------------------- Cross-sectional (optional) ----------------------------

def _weighted_market_exposure(markets: List[str], weights: List[float]) -> float:
    """Simple scalar exposure: sum of provided weights. If empty, returns NaN."""
    if not markets or not weights:
        return np.nan
    w = np.array([w for w in weights if not pd.isna(w)], dtype=float)
    if w.size == 0:
        return np.nan
    return float(w.sum())  # you can replace with a more nuanced score later


def cross_sectional_ols(per_event_df: pd.DataFrame, events: pd.DataFrame, roles_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Regress CAR on visa shock size, expected uplift, and weighted market exposure,
    plus dummies for country/flow/type/role/season.
    Returns coefficients (t, p) if statsmodels available.
    """
    if not HAVE_SM or per_event_df.empty:
        return None

    # Extract event_id from key "event_#_YYYY-MM-DD"
    def _eid(s):
        try:
            return int(str(s).split("_")[1])
        except Exception:
            return np.nan

    df = per_event_df.copy()
    df["eid"] = df["event"].apply(_eid)

    ev = events.copy()
    ev = ev.assign(eid=np.arange(1, len(ev)+1))
    # Precompute exposure scalar
    ev["market_exposure"] = [
        _weighted_market_exposure(mkts if isinstance(mkts, list) else [], wts if isinstance(wts, list) else [])
        for mkts, wts in zip(ev.get("markets_list", []), ev.get("market_weights_list", []))
    ]

    df = df.merge(
        ev[["eid","country","flow","type","season","shock_points","expected_uplift_pct","market_exposure"]],
        on="eid", how="left"
    )

    # Merge roles if available
    if not roles_df.empty:
        df = df.merge(roles_df[["event_id","ticker","role"]], left_on=["eid","ticker"], right_on=["event_id","ticker"], how="left")
        df.drop(columns=["event_id"], inplace=True, errors="ignore")

    X = pd.DataFrame(index=df.index)
    X["shock_points"]        = df["shock_points"].fillna(0.0).astype(float)
    X["expected_uplift_pct"] = df["expected_uplift_pct"].fillna(0.0).astype(float)
    X["market_exposure"]     = df["market_exposure"].fillna(0.0).astype(float)
    # Interaction (bigger shock × higher exposure)
    X["shock_x_exposure"]    = X["shock_points"] * X["market_exposure"]

    for col, pref in [("country","cty"),("flow","flow"),("type","typ"),("season","ssn"),("role","role")]:
        if col in df.columns:
            X = pd.concat([X, pd.get_dummies(df[col].fillna("NA"), prefix=pref, drop_first=True)], axis=1)

    X = sm.add_constant(X)
    y = df["CAR"].astype(float)
    model = sm.OLS(y, X, missing="drop").fit()
    return pd.DataFrame({"coef": model.params, "t": model.tvalues, "p": model.pvalues})


# ---------------------------- Plotting ----------------------------

def plot_aar_caar(AAR: pd.DataFrame, CAAR: pd.DataFrame, outdir: str, title_suffix: str = ""):
    if plt is None:
        print("[INFO] matplotlib not available; skipping plots.")
        return
    fig1 = plt.figure(figsize=(9, 5))
    AAR["AAR"].plot(ax=plt.gca(), linewidth=1.2)
    plt.axvline(0, linestyle="--")
    plt.title(f"AAR around event day {title_suffix}".strip())
    plt.xlabel("Event time (days)")
    plt.ylabel("Average abnormal return")
    plt.tight_layout()
    fig1.savefig(os.path.join(outdir, "plots", "AAR.png"), dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(9, 5))
    CAAR["CAAR"].plot(ax=plt.gca(), linewidth=1.2)
    plt.axvline(0, linestyle="--")
    plt.title(f"CAAR around event day {title_suffix}".strip())
    plt.xlabel("Event time (days)")
    plt.ylabel("Cumulative AAR")
    plt.tight_layout()
    fig2.savefig(os.path.join(outdir, "plots", "CAAR.png"), dpi=150)
    plt.close(fig2)


def plot_heatmap(panel: pd.DataFrame, outdir: str):
    if plt is None:
        return
    M = panel.copy()
    M.columns = [f"{ev}|{tic}" for ev, tic in M.columns]
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    im = ax.imshow(M.T.values, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(M.columns)))
    ax.set_yticklabels(M.columns, fontsize=7)
    ax.set_xticks(range(len(M.index)))
    ax.set_xticklabels(M.index, fontsize=7)
    ax.set_title("Abnormal returns heatmap (events × tickers)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "plots", "heatmap_abnormal_returns.png"), dpi=160)
    plt.close(fig)


# ---------------------------- Main ----------------------------

def main():
    p = argparse.ArgumentParser(description="Event study for tourism visa policy shocks.")
    p.add_argument("--events-file", type=str, required=True,
                   help="CSV with 'date' (+ optional 'country','flow','type','tickers','roles','markets','market_weights','shock_points','expected_uplift_pct','season')")
    p.add_argument("--map-file", type=str, default=None,
                   help="CSV mapping to tickers; required 'tickers'; optional 'country','flow','type','role','market'")
    p.add_argument("--benchmark", type=str, default="^N225",
                   help="Benchmark (e.g., ^N225, 1306.T, ^GSPC, ^STOXX50E, ^NSEI)")
    p.add_argument("--start", type=str, default="2010-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--tpre", type=int, default=5)
    p.add_argument("--tpost", type=int, default=10)
    p.add_argument("--market-model", action="store_true",
                   help="Use market-model (rolling OLS) abnormal returns; else benchmark-adjusted")
    p.add_argument("--plot", action="store_true", help="Write PNG plots")
    p.add_argument("--outdir", type=str, default="./artifacts")
    args = p.parse_args()

    cfg = Config(
        events_file=args.events_file,
        map_file=args.map_file,
        benchmark=args.benchmark,
        start=args.start,
        end=args.end,
        tpre=args.tpre,
        tpost=args.tpost,
        use_market_model=bool(args.market_model),
        plot=bool(args.plot),
        outdir=args.outdir,
    )

    outdir = ensure_outdir(cfg.outdir)
    print(f"[INFO] Writing artifacts to: {outdir}")

    events = load_events(cfg.events_file)
    mapper = load_map(cfg.map_file)

    # Resolve tickers per event & gather universe
    per_event_tickers: Dict[int, List[str]] = {}
    all_tickers: Set[str] = set()
    for i in range(len(events)):
        tickers = resolve_event_tickers(events.loc[i], mapper)
        if not tickers:
            print(f"[WARN] Event {i+1} {events.loc[i,'date'].date()} has no tickers; may be skipped.")
        per_event_tickers[i] = tickers
        all_tickers |= set(tickers)

    # Include benchmark
    if cfg.benchmark:
        all_tickers |= {cfg.benchmark}
    if not all_tickers:
        raise SystemExit("No tickers to analyze. Provide tickers in events or a map-file.")

    # Extend start for estimation if using market model
    est_pad_days = 200 if cfg.use_market_model else 30
    start_ext = (pd.to_datetime(cfg.start) - pd.Timedelta(days=cfg.tpre + est_pad_days)).date().isoformat()

    print(f"[INFO] Downloading prices for {len(all_tickers)} tickers...")
    px = fetch_prices(sorted(all_tickers), start_ext, cfg.end)
    px.to_csv(os.path.join(outdir, "prices.csv"))

    r = daily_returns(px)
    r.to_csv(os.path.join(outdir, "daily_returns.csv"))

    if cfg.benchmark not in r.columns:
        raise SystemExit(f"Benchmark {cfg.benchmark} not found in returns. Columns: {list(r.columns)}")

    rb = r[cfg.benchmark].dropna()
    ar_df = pd.DataFrame(index=r.index)
    for t in sorted(all_tickers):
        if t == cfg.benchmark or t not in r.columns:
            continue
        ar = market_model_abnormal(r[t].dropna(), rb) if cfg.use_market_model else benchmark_adjusted(r[t].dropna(), rb)
        ar_df[t] = ar

    # Build event windows
    panel = build_event_windows(ar_df, events, per_event_tickers, cfg.tpre, cfg.tpost)
    panel.to_csv(os.path.join(outdir, f"event_window_{cfg.tpre}_{cfg.tpost}.csv"))

    # AAR/CAAR
    AAR, CAAR = aggregate_aar_caar(panel)
    AAR.to_csv(os.path.join(outdir, "AAR.csv"))
    CAAR.to_csv(os.path.join(outdir, "CAAR.csv"))

    # Per-event CARs
    per_event = per_event_car(panel)
    per_event.to_csv(os.path.join(outdir, "per_event_CAR.csv"), index=False)

    # Roles (if provided)
    roles_df = roles_table(events, per_event_tickers)
    if not roles_df.empty:
        roles_df.to_csv(os.path.join(outdir, "roles_by_event.csv"), index=False)

    # Summary stats
    aar0 = AAR.loc[0, "AAR"] if 0 in AAR.index else np.nan
    caar_end = CAAR.iloc[-1, 0] if not CAAR.empty else np.nan
    tstat, pval = ttest_zero(per_event["CAR"]) if not per_event.empty else (np.nan, np.nan)
    summary = pd.DataFrame([{
        "AAR_t0": aar0,
        "CAAR_end": caar_end,
        "CAR_mean": per_event["CAR"].mean() if not per_event.empty else np.nan,
        "CAR_sd": per_event["CAR"].std(ddof=1) if not per_event.empty else np.nan,
        "t_stat": tstat,
        "p_value": pval,
        "n_events": per_event["event"].nunique() if not per_event.empty else 0
    }])
    summary.to_csv(os.path.join(outdir, "summary_stats.csv"), index=False)

    print("\n=== Summary ===")
    print(summary.round(4).to_string(index=False))

    # Cross-sectional OLS (optional)
    xsec = cross_sectional_ols(per_event, events, roles_df)
    if xsec is not None:
        xsec.to_csv(os.path.join(outdir, "cross_sectional_ols.csv"))
        print("\n=== Cross-sectional OLS (CAR drivers) ===")
        print(xsec.round(4).to_string())

    # Plots
    if cfg.plot:
        plot_aar_caar(AAR, CAAR, outdir, title_suffix=f"(t=[-{cfg.tpre}, +{cfg.tpost}])")
        plot_heatmap(panel, outdir)
        print("[OK] Plots saved to:", os.path.join(outdir, "plots"))

    print("\nDone.")


if __name__ == "__main__":
    main()