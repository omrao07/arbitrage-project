# backend/altdata/patent_momentum.py
from __future__ import annotations
"""
Patent Momentum
---------------
Compute per-assignee monthly "patent momentum" from raw patent rows.

Inputs (CSV/DataFrame/list-of-dicts supported):
  required columns:
    - date              (YYYY-MM-DD or timestamp)
    - assignee          (company / org string)
  optional columns:
    - patent_id         (str)
    - fwd_citations     (int) forward citations (total to date or 3y window)
    - cpc               (str) pipe/semicolon/comma-separated CPC/IPC codes
    - abstract,title    (str) free text (for lightweight novelty if numpy/pandas available)
    - jurisdiction      (str) "US"/"EP"/"WO"/…
  optional ticker map:
    - assignee -> ticker CSV to attach public tickers

Outputs:
  Monthly table with:
    filings_m12          rolling 12M filing count
    yoy_filings          YoY change of 12M filings
    accel_filings        acceleration (Δ YoY)
    cite_intensity       citations per filing (normalized)
    novelty_code         fraction of new CPC/IPC codes vs 36M history
    novelty_text         (optional) text n-gram novelty score
    pm_score             composite momentum in [0,100]
    z_*                  z-scored components (for debugging/attribution)

No hard dependencies:
  - With pandas/numpy: full set of features + faster pipeline
  - Without: computes filings_m12 / yoy / accel and a simpler composite

Bus:
  Emits to Redis stream "alt.patent_momentum" if backend.bus.streams.publish_stream is available.

CLI:
  python -m backend.altdata.patent_momentum --patents data/patents.csv --out data/patent_momentum.csv \
      --assignee_col assignee --date_col date --ticker_map data/assignee_ticker.csv

"""

import csv, json, math, os, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict
from datetime import datetime, timedelta

# -------- optional deps (graceful fallbacks) --------
try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

# -------- optional bus --------
try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass

OUT_STREAM = os.getenv("PATENT_MOMENTUM_STREAM", "alt.patent_momentum")

# ----------------- helpers -----------------

def _parse_date(s: Any) -> datetime:
    if isinstance(s, datetime):
        return s
    ss = str(s)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ss[:len(fmt)], fmt)
        except Exception:
            continue
    # last resort
    return datetime.fromtimestamp(float(ss)) if ss.isdigit() else datetime.fromisoformat(ss[:10])

def _month_floor(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1)

def _roll_count(months: List[datetime], current: datetime, window_m: int = 12) -> int:
    lo = current - timedelta(days=int(window_m * 30.5))
    return sum(1 for d in months if (lo < d <= current))

def _z(vals: List[float]) -> List[float]:
    if not vals:
        return []
    if _np is not None:
        mu = float(_np.nanmean(vals))
        sd = float(_np.nanstd(vals)) or 1.0
        return [0.0 if (v is None) else (v - mu) / sd for v in vals]
    mu = sum(v for v in vals if v is not None) / max(1, sum(1 for v in vals if v is not None))
    sd = (sum((v - mu) ** 2 for v in vals if v is not None) / max(1, sum(1 for v in vals if v is not None) - 1)) ** 0.5 or 1.0
    return [0.0 if (v is None) else (v - mu) / sd for v in vals]

def _safe_div(a: float, b: float) -> float:
    return a / b if b not in (0, 0.0, None) else 0.0

def _tokenize_codes(s: str) -> List[str]:
    if not s: return []
    s = str(s).replace("|", ";").replace(",", ";")
    return [t.strip().upper() for t in s.split(";") if t.strip()]

def _ngram_shingle(text: str, n: int = 3) -> List[str]:
    t = "".join(ch.lower() if ch.isalnum() or ch == " " else " " for ch in (text or ""))
    toks = [w for w in t.split() if w]
    if len(toks) < n: return toks
    return [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]

# ----------------- dataclasses -----------------

@dataclass
class PMRow:
    date: str
    assignee: str
    ticker: Optional[str]
    filings_m12: float
    yoy_filings: float
    accel_filings: float
    cite_intensity: float
    novelty_code: float
    novelty_text: Optional[float]
    z_filings: float
    z_citations: float
    z_nov_code: float
    z_nov_text: float
    pm_score: float  # 0..100 composite

# ----------------- core engine -----------------

class PatentMomentum:
    def __init__(
        self,
        *,
        min_history_months: int = 18,
        code_window_m: int = 36,
        cite_norm: str = "per_filing",     # or "raw"
        weights: Tuple[float, float, float, float] = (0.45, 0.30, 0.20, 0.05),  # filings, citations, code novelty, text novelty
        emit_stream: str = OUT_STREAM
    ):
        self.min_hist = min_history_months
        self.code_win = code_window_m
        self.cite_norm = cite_norm
        self.w_filings, self.w_cites, self.w_code, self.w_text = weights
        self.emit_stream = emit_stream

    # ---------- public API ----------

    def compute(
        self,
        patents: Union[str, List[Dict[str, Any]], Any],
        *,
        assignee_col: str = "assignee",
        date_col: str = "date",
        fwd_cite_col: str = "fwd_citations",
        cpc_col: str = "cpc",
        abstract_col: str = "abstract",
        ticker_map: Optional[Union[str, Dict[str, str], Any]] = None,
        return_frame: bool = True,
        emit_tail: bool = False
    ) -> Any:
        """
        Return a pandas DataFrame if pandas is available (and return_frame=True). Otherwise returns a list of dicts.
        """
        tickers = self._load_ticker_map(ticker_map)

        if _pd is not None:
            df = self._to_df(patents)
            # normalize
            df["__date"] = _pd.to_datetime(df[date_col].apply(_parse_date)).dt.to_period("M").dt.to_timestamp() # type: ignore
            df["__assignee"] = df[assignee_col].astype(str) # type: ignore
            if fwd_cite_col in df.columns: # type: ignore
                df["__fwd"] = _pd.to_numeric(df[fwd_cite_col], errors="coerce").fillna(0.0) # type: ignore
            else:
                df["__fwd"] = 0.0 # type: ignore
            df["__cpc_list"] = df[cpc_col].fillna("").apply(_tokenize_codes) if cpc_col in df.columns else [[]] # type: ignore
            df["__text"] = df[abstract_col].fillna("") if abstract_col in df.columns else "" # type: ignore

            # monthly filings
            mfil = df.groupby(["__assignee","__date"]).size().rename("filings").reset_index() # type: ignore
            # rolling 12m filings per assignee
            mfil["filings_m12"] = mfil.groupby("__assignee")["filings"].transform(
                lambda s: s.rolling(window=12, min_periods=1).sum()
            )
            # YoY & acceleration
            mfil["yoy_filings"] = (mfil["filings_m12"] - mfil.groupby("__assignee")["filings_m12"].shift(12)) / \
                                  (mfil.groupby("__assignee")["filings_m12"].shift(12) + 1e-9)
            mfil["accel_filings"] = mfil["yoy_filings"] - mfil.groupby("__assignee")["yoy_filings"].shift(1)

            # citations intensity (sum fwd citations for filings in last N months / filings count)
            ctab = df.groupby(["__assignee","__date"])["__fwd"].sum().rename("fwd_sum").reset_index() # type: ignore
            ctab["fwd_m12"] = ctab.groupby("__assignee")["fwd_sum"].transform(lambda s: s.rolling(12, min_periods=1).sum())
            joined = mfil.merge(ctab[["__assignee","__date","fwd_m12"]], on=["__assignee","__date"], how="left")
            joined["cite_intensity"] = _pd.Series(_safe_div(a, b) for a, b in zip(joined["fwd_m12"].fillna(0.0), joined["filings_m12"].replace(0, _pd.NA))) # type: ignore
            joined["cite_intensity"] = joined["cite_intensity"].fillna(0.0)

            # code novelty: fraction of CPC codes in this month that were NOT seen in trailing 36m for the assignee
            code_hist = {}
            nov_vals = []
            for (assg, dt), grp in df.groupby(["__assignee","__date"]): # type: ignore
                seen = code_hist.get(assg, [])
                window_start = dt - _pd.offsets.DateOffset(months=self.code_win)
                # build set of historical codes in window
                hist_codes = set()
                for d0, codes in seen:
                    if d0 > window_start:
                        hist_codes.update(codes)
                now_codes = set(c for row in grp["__cpc_list"] for c in row)
                frac_new = 0.0
                if now_codes:
                    frac_new = len([c for c in now_codes if c not in hist_codes]) / max(1, len(now_codes))
                nov_vals.append((assg, dt, frac_new))
                # append to history
                seen.append((dt, list(now_codes)))
                # prune old
                code_hist[assg] = [(d0, cs) for (d0, cs) in seen if d0 > window_start]
            nov = _pd.DataFrame(nov_vals, columns=["__assignee","__date","novelty_code"])
            joined = joined.merge(nov, on=["__assignee","__date"], how="left")

            # text novelty (optional): shingle overlap vs trailing 36m abstracts
            if abstract_col in df.columns and _np is not None: # type: ignore
                text_hist = {}
                tnv = []
                for (assg, dt), grp in df.groupby(["__assignee","__date"]): # type: ignore
                    window_start = dt - _pd.offsets.DateOffset(months=self.code_win)
                    past = text_hist.get(assg, [])
                    past_ngrams = set()
                    for d0, grams in past:
                        if d0 > window_start:
                            past_ngrams.update(grams)
                    cur = set()
                    for txt in grp["__text"]:
                        cur.update(_ngram_shingle(txt, 3))
                    nov_text = 0.0
                    if cur:
                        inter = len(cur & past_ngrams)
                        nov_text = 1.0 - (inter / float(len(cur)))
                    tnv.append((assg, dt, nov_text))
                    past.append((dt, list(cur)))
                    text_hist[assg] = [(d0, gs) for (d0, gs) in past if d0 > window_start]
                tnv = _pd.DataFrame(tnv, columns=["__assignee","__date","novelty_text"])
                joined = joined.merge(tnv, on=["__assignee","__date"], how="left")
            else:
                joined["novelty_text"] = _pd.NA

            # z-scores & composite
            def _colz(name: str) -> _pd.Series: # type: ignore
                x = joined[name].astype(float)
                return (x - x.mean()) / (x.std(ddof=1) + 1e-9)

            joined["z_filings"]   = _colz("yoy_filings").fillna(0.0) + 0.5 * _colz("accel_filings").fillna(0.0)
            joined["z_citations"] = _colz("cite_intensity").fillna(0.0)
            joined["z_nov_code"]  = _colz("novelty_code").fillna(0.0)
            joined["z_nov_text"]  = _colz("novelty_text").fillna(0.0) if "novelty_text" in joined.columns else 0.0

            joined["pm_raw"] = (
                self.w_filings  * joined["z_filings"]   +
                self.w_cites    * joined["z_citations"] +
                self.w_code     * joined["z_nov_code"]  +
                self.w_text     * joined["z_nov_text"]
            )
            # scale to 0..100 per month (cross-sectional)
            joined["pm_score"] = joined.groupby("__date")["pm_raw"].transform(
                lambda s: 100.0 * ( (s - s.min()) / ( (s.max() - s.min()) + 1e-9 ) )
            )

            # attach tickers & tidy
            joined["ticker"] = joined["__assignee"].map(tickers) if tickers else None
            out = joined.rename(columns={
                "__assignee":"assignee", "__date":"date"
            })[
                ["date","assignee","ticker","filings_m12","yoy_filings","accel_filings",
                 "cite_intensity","novelty_code","novelty_text","z_filings","z_citations",
                 "z_nov_code","z_nov_text","pm_score"]
            ].sort_values(["assignee","date"])

            # min history gate
            out = out.groupby("assignee").apply(
                lambda g: g.iloc[self.min_hist-1 :] if len(g) >= self.min_hist else _pd.DataFrame(columns=g.columns) # type: ignore
            ).reset_index(drop=True)

            if emit_tail and len(out):
                tail = out.groupby("assignee").tail(1).to_dict(orient="records")
                publish_stream(self.emit_stream, {"ts_ms": int(time.time()*1000), "n": len(tail), "rows": tail})

            return out if return_frame else out.to_dict(orient="records")

        # -------- no-pandas fallback (simpler metrics) --------
        rows = self._to_rows(patents)
        # group months per assignee
        by_assg_dates: Dict[str, List[datetime]] = defaultdict(list)
        by_assg_month: Dict[Tuple[str, datetime], Dict[str, Any]] = defaultdict(dict)
        for r in rows:
            assg = str(r.get(assignee_col, "")).strip()
            if not assg: continue
            dt = _month_floor(_parse_date(r.get(date_col)))
            by_assg_dates[assg].append(dt)
            key = (assg, dt)
            by_assg_month[key]["filings"] = by_assg_month.get(key, {}).get("filings", 0) + 1
            by_assg_month[key]["fwd"] = by_assg_month.get(key, {}).get("fwd", 0) + float(r.get("fwd_citations", 0.0))

        # assemble per-month features
        out_rows: List[Dict[str, Any]] = []
        for assg, dates in by_assg_dates.items():
            uniq_months = sorted(set(dates))
            hist_m12 = []
            fwd_sum = 0.0
            for m in uniq_months:
                filings = by_assg_month[(assg, m)].get("filings", 0)
                fwd_sum += by_assg_month[(assg, m)].get("fwd", 0.0)
                hist_m12.append((m, filings))
                # rolling 12m filings
                f12 = sum(v for (d,v) in hist_m12 if (m - timedelta(days=365)) < d <= m)
                # YoY and accel
                m_ago = m - timedelta(days=365)
                f12_prev = sum(v for (d,v) in hist_m12 if (m - timedelta(days=730)) < d <= m_ago)
                yoy = _safe_div((f12 - f12_prev), (f12_prev + 1e-9))
                # previous month yoy
                m_prev = uniq_months[max(0, uniq_months.index(m)-1)]
                f12_prevprev = sum(v for (d,v) in hist_m12 if (m_prev - timedelta(days=730)) < d <= (m_prev - timedelta(days=365)))
                f12_prev_cur = sum(v for (d,v) in hist_m12 if (m_prev - timedelta(days=365)) < d <= m_prev)
                yoy_prev = _safe_div((f12_prev_cur - f12_prevprev), (f12_prevprev + 1e-9))
                accel = yoy - yoy_prev
                cite_int = _safe_div(fwd_sum, f12)
                out_rows.append({
                    "date": m.strftime("%Y-%m-%d"),
                    "assignee": assg,
                    "ticker": (tickers.get(assg) if tickers else None),
                    "filings_m12": float(f12),
                    "yoy_filings": float(yoy),
                    "accel_filings": float(accel),
                    "cite_intensity": float(cite_int),
                    "novelty_code": 0.0,
                    "novelty_text": None
                })

        # z & composite per month (cross-sectional)
        # collect by month
        by_m: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in out_rows:
            by_m[r["date"]].append(r)
        final_rows: List[Dict[str, Any]] = []
        for d, recs in by_m.items():
            zf = _z([r["yoy_filings"] for r in recs])
            zc = _z([r["cite_intensity"] for r in recs])
            for i, r in enumerate(recs):
                pm_raw = self.w_filings * (zf[i] + 0.5 * r["accel_filings"]) + self.w_cites * zc[i]
                # min-max to 0..100
                # collect raw list
                recs[i]["z_filings"] = float(zf[i])
                recs[i]["z_citations"] = float(zc[i])
            raws = [self.w_filings*(rr["z_filings"] + 0.5*rr["accel_filings"]) + self.w_cites*rr["z_citations"] for rr in recs]
            lo, hi = (min(raws), max(raws)) if raws else (0.0, 1.0)
            for i, r in enumerate(recs):
                r["z_nov_code"] = 0.0
                r["z_nov_text"] = 0.0
                r["pm_score"] = 100.0 * ((raws[i] - lo) / ( (hi - lo) + 1e-9 ))
                final_rows.append(r)

        # min-history filter
        # (approximate: keep if assignee has >= min_hist months)
        keep = []
        cnt = defaultdict(int)
        for r in sorted(final_rows, key=lambda x: (x["assignee"], x["date"])):
            cnt[r["assignee"]] += 1
            if cnt[r["assignee"]] >= self.min_hist:
                keep.append(r)

        if emit_tail and keep:
            tails = {}
            for r in keep:
                tails[r["assignee"]] = r
            publish_stream(self.emit_stream, {"ts_ms": int(time.time()*1000), "n": len(tails), "rows": list(tails.values())})
        return keep

    # ---------- utilities ----------

    def _to_df(self, patents: Union[str, List[Dict[str, Any]], Any]):
        if isinstance(patents, str):
            return _pd.read_csv(patents) # type: ignore
        if _pd is not None and hasattr(patents, "to_dict"):
            return patents.copy()
        # list of dicts
        return _pd.DataFrame(patents) # type: ignore

    def _to_rows(self, patents: Union[str, List[Dict[str, Any]], Any]) -> List[Dict[str, Any]]:
        if isinstance(patents, str):
            out = []
            with open(patents, "r", newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    out.append(r)
            return out
        if _pd is not None and hasattr(patents, "to_dict"):
            return patents.to_dict(orient="records")  # type: ignore
        return list(patents)

    def _load_ticker_map(self, ticker_map: Optional[Union[str, Dict[str,str], Any]]) -> Dict[str, str]:
        if ticker_map is None:
            return {}
        if isinstance(ticker_map, dict):
            return {str(k): str(v) for k, v in ticker_map.items()}
        if isinstance(ticker_map, str):
            table = {}
            with open(ticker_map, "r", newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    a = str(r.get("assignee") or r.get("name") or r.get("Assignee") or "").strip()
                    t = str(r.get("ticker") or r.get("Ticker") or "").strip()
                    if a and t:
                        table[a] = t
            return table
        if _pd is not None and hasattr(ticker_map, "to_dict"):
            m = {}
            df = ticker_map
            ac = "assignee" if "assignee" in df.columns else df.columns[0]
            tc = "ticker" if "ticker" in df.columns else df.columns[1]
            for _, r in df[[ac, tc]].dropna().iterrows():
                m[str(r[ac]).strip()] = str(r[tc]).strip()
            return m
        return {}

# ----------------- CLI -----------------

def _main():
    import argparse
    p = argparse.ArgumentParser(description="Patent Momentum (per-assignee monthly)")
    p.add_argument("--patents", required=True, help="CSV with at least [date,assignee]")
    p.add_argument("--out", required=True, help="Output CSV")
    p.add_argument("--assignee_col", default="assignee")
    p.add_argument("--date_col", default="date")
    p.add_argument("--fwd_cite_col", default="fwd_citations")
    p.add_argument("--cpc_col", default="cpc")
    p.add_argument("--abstract_col", default="abstract")
    p.add_argument("--ticker_map", default=None, help="CSV with [assignee,ticker] (optional)")
    args = p.parse_args()

    pm = PatentMomentum()
    if _pd is None:
        # no pandas — fallback: read CSV as rows and write CSV manually
        rows = pm.compute(args.patents, assignee_col=args.assignee_col, date_col=args.date_col,
                          fwd_cite_col=args.fwd_cite_col, cpc_col=args.cpc_col, abstract_col=args.abstract_col,
                          ticker_map=(args.ticker_map if args.ticker_map else None), return_frame=False)
        # write
        if not rows:
            open(args.out, "w").close()
            return
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
        return

    df_out = pm.compute(args.patents, assignee_col=args.assignee_col, date_col=args.date_col,
                        fwd_cite_col=args.fwd_cite_col, cpc_col=args.cpc_col, abstract_col=args.abstract_col,
                        ticker_map=( _pd.read_csv(args.ticker_map) if args.ticker_map else None ))
    df_out.to_csv(args.out, index=False)

if __name__ == "__main__":  # pragma: no cover
    _main()