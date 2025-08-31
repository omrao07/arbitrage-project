# backend/altdata/satellite_ndvi.py
from __future__ import annotations
"""
Satellite NDVI (Alt-Data)
-------------------------
Compute daily per-AOI NDVI from Red/NIR pixel samples.

Inputs
------
1) AOIs (CSV / DataFrame / list[dict]) with:
   id,name,lat,lon[,radius_m,city,region,polygon_wkt]
   - If shapely is available, you may pass `polygon_wkt` (WKT POLYGON/MULTIPOLYGON) to use exact shapes.
   - Otherwise circles are used (haversine) with radius_m (default 500m).

2) Pixels (CSV / DataFrame / list[dict]) with:
   date,lat,lon,red,nir[,cloud_flag,quality]
   - date: ISO YYYY-MM-DD (or timestamp)
   - red/nir: reflectance (0..1) or radiance (scaled OK; NDVI normalizes)
   - cloud_flag: truthy value masks the pixel
   - quality (float): optional; filter with `quality_min`

Outputs
-------
Per-AOI daily table with:
  mean_ndvi, median_ndvi, p10, p90, smoothed, z, anomaly_doy, yoy, vigor_idx(0..100)
Plus optional city/region rollups.

No hard deps:
- Uses pure Python + haversine by default.
- Will auto-use pandas/numpy/shapely when available for speed/geometry.

CLI
---
python -m backend.altdata.satellite_ndvi \
  --aoi data/aois.csv \
  --pixels data/red_nir_pixels.csv \
  --out data/ndvi_per_aoi.csv \
  --city_out data/ndvi_city.csv --region_out data/ndvi_region.csv \
  --smoothing 7 --minpixels 10

Env (optional bus):
  REDIS_HOST, REDIS_PORT, NDVI_STREAM (default "alt.ndvi")

Notes
-----
- For Sentinel-2: RED=B4, NIR=B8; Landsat 8/9: RED=Band4, NIR=Band5.
- Pre-extract/cloud-mask upstream (e.g., QA60/SCL/Fmask) or pass cloud_flag here.
"""

import csv, math, os, json, time, re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# ---------- Optional deps ----------
try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None
try:
    from shapely import wkt as _wkt  # type: ignore
    from shapely.geometry import Point as _Point  # type: ignore
    _has_shapely = True
except Exception:
    _has_shapely = False

# ---------- Optional bus ----------
try:
    import redis as _redis  # type: ignore
except Exception:
    _redis = None
try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
OUT_STREAM = os.getenv("NDVI_STREAM", "alt.ndvi")

# ---------- Helpers ----------
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _clip(x: float, lo=-1.0, hi=1.0) -> float:
    return max(lo, min(hi, x))

def _parse_date(s: Any) -> str:
    if isinstance(s, datetime):
        return s.date().isoformat()
    ss = str(s)
    # try common forms
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ss[:len(fmt)], fmt).date().isoformat()
        except Exception:
            continue
    try:
        return datetime.fromtimestamp(float(ss)).date().isoformat()
    except Exception:
        return ss[:10]

def _movavg(xs: List[float], w: int) -> List[float]:
    if w <= 1 or not xs: return xs
    out, s = [], 0.0
    for i, v in enumerate(xs):
        s += v
        if i >= w: s -= xs[i-w]
        out.append(s / min(i+1, w))
    return out

def _zscore(xs: List[float]) -> List[float]:
    if not xs: return xs
    if _np is not None:
        mu = float(_np.mean(xs)); sd = float(_np.std(xs)) or 1.0
    else:
        mu = sum(xs)/len(xs); sd = (sum((x-mu)**2 for x in xs)/max(1, len(xs)-1))**0.5 or 1.0
    return [(x - mu) / sd for x in xs]

def _doy(s: str) -> int:
    try:
        dt = datetime.fromisoformat(s[:10])
        return int(dt.timetuple().tm_yday)
    except Exception:
        return 1

def _winsor(vals: List[float], lo_q=0.05, hi_q=0.95) -> List[float]:
    if not vals: return vals
    if _np is not None:
        lo, hi = _np.quantile(vals, [lo_q, hi_q]).tolist()
    else:
        s = sorted(vals); n = len(s)
        lo, hi = s[int(lo_q*(n-1))], s[int(hi_q*(n-1))]
    return [max(lo, min(hi, v)) for v in vals]

# ---------- Data models ----------
@dataclass
class AOI:
    id: str
    name: str
    lat: float
    lon: float
    radius_m: float = 500.0
    city: str = ""
    region: str = ""
    polygon_wkt: Optional[str] = None

@dataclass
class NDVIStat:
    date: str
    aoi_id: str
    mean_ndvi: float
    median_ndvi: float
    p10: float
    p90: float
    smoothed: float
    z: float
    anomaly_doy: Optional[float]
    yoy: Optional[float]
    vigor_idx: float      # 0..100 scaled
    n_px: int

# ---------- Core engine ----------
class NDVIEngine:
    def __init__(
        self,
        *,
        default_radius_m: float = 500.0,
        smoothing_days: int = 7,
        winsor_lo: float = 0.02,
        winsor_hi: float = 0.98,
        min_pixels: int = 10,
        mask_cloud: bool = True,
        cloud_col: str = "cloud_flag",
        quality_col: str = "quality",
        quality_min: Optional[float] = None,
        emit_stream: str = OUT_STREAM,
    ):
        self.default_radius_m = default_radius_m
        self.smoothing_days = max(1, int(smoothing_days))
        self.winsor_lo = winsor_lo
        self.winsor_hi = winsor_hi
        self.min_pixels = min_pixels
        self.mask_cloud = mask_cloud
        self.cloud_col = cloud_col
        self.quality_col = quality_col
        self.quality_min = quality_min
        self.emit_stream = emit_stream

        self.aois: Dict[str, AOI] = {}
        self._r = None
        if _redis is not None:
            try:
                self._r = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            except Exception:
                self._r = None

        # cache compiled polygons if shapely is available
        self._polys: Dict[str, Any] = {}

    # ----- load AOIs -----
    def load_aois(self, aois: Union[str, List[Dict[str, Any]], Any]) -> None:
        rows: List[Dict[str, Any]] = []
        if isinstance(aois, str):
            with open(aois, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        elif _pd is not None and hasattr(aois, "to_dict"):
            rows = aois.to_dict(orient="records")  # type: ignore
        elif isinstance(aois, list):
            rows = aois
        else:
            raise ValueError("Unsupported AOIs input")

        for r in rows:
            aid = str(r.get("id") or r.get("aoi_id") or "").strip()
            if not aid: continue
            lat = float(r.get("lat", "0") or 0.0)
            lon = float(r.get("lon", "0") or 0.0)
            rad = float(r.get("radius_m", self.default_radius_m) or self.default_radius_m)
            a = AOI(
                id=aid,
                name=str(r.get("name") or aid),
                lat=lat, lon=lon, radius_m=rad,
                city=str(r.get("city") or ""),
                region=str(r.get("region") or ""),
                polygon_wkt=(str(r.get("polygon_wkt")) if r.get("polygon_wkt") else None),
            )
            self.aois[aid] = a
            # pre-parse polygon
            if _has_shapely and a.polygon_wkt:
                try:
                    self._polys[aid] = _wkt.loads(a.polygon_wkt)
                except Exception:
                    self._polys[aid] = None

    # ----- compute -----
    def compute(
        self,
        pixels: Union[str, List[Dict[str, Any]], Any],
        *,
        return_frames: bool = True,
        emit: bool = False
    ) -> Tuple[Any, Any, Any]:
        """
        Compute NDVI per AOI per day; returns (per_aoi_df, city_df, region_df)
        """
        if not self.aois:
            raise RuntimeError("No AOIs loaded. Call load_aois() first.")

        rows: List[Dict[str, Any]] = []
        if isinstance(pixels, str):
            with open(pixels, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        elif _pd is not None and hasattr(pixels, "to_dict"):
            rows = pixels.to_dict(orient="records")  # type: ignore
        elif isinstance(pixels, list):
            rows = pixels
        else:
            raise ValueError("Unsupported pixels input")

        # date -> list[pixels]
        by_date: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            d = _parse_date(r.get("date"))
            lat, lon = r.get("lat"), r.get("lon")
            red, nir = r.get("red"), r.get("nir")
            try:
                lat = float(lat); lon = float(lon); red = float(red); nir = float(nir) # type: ignore
            except Exception:
                continue
            # cloud/quality mask
            if self.mask_cloud and self.cloud_col in r:
                if str(r[self.cloud_col]).strip() not in ("0","False","false","","None"):
                    continue
            if (self.quality_min is not None) and (self.quality_col in r):
                try:
                    if float(r[self.quality_col]) < float(self.quality_min):
                        continue
                except Exception:
                    pass
            # NDVI
            ndvi = _clip((nir - red) / (nir + red + 1e-9), -1.0, 1.0)
            by_date.setdefault(d, []).append({"lat": lat, "lon": lon, "ndvi": ndvi})

        # Per-date, assign pixels to AOIs
        per_aoi_series: Dict[str, Dict[str, List[float]]] = {aid: {} for aid in self.aois.keys()}

        for d, px in sorted(by_date.items()):
            if not px:
                continue
            # Geo cache
            shapely_pts = [_Point(p["lon"], p["lat"]) for p in px] if _has_shapely else None
            for aid, a in self.aois.items():
                vals: List[float] = []
                if _has_shapely and a.polygon_wkt and self._polys.get(aid) is not None:
                    poly = self._polys[aid]
                    for i, pt in enumerate(shapely_pts): # type: ignore
                        if poly.contains(pt):
                            vals.append(px[i]["ndvi"])
                else:
                    # circle haversine
                    for p in px:
                        if _haversine_m(a.lat, a.lon, p["lat"], p["lon"]) <= a.radius_m:
                            vals.append(p["ndvi"])

                if len(vals) >= self.min_pixels:
                    vals = _winsor(vals, self.winsor_lo, self.winsor_hi)
                    per_aoi_series[aid].setdefault("date", []).append(d) # type: ignore
                    per_aoi_series[aid].setdefault("ndvi", []).append(float(sum(vals) / len(vals)))
                    # store quantiles for that day (approximate)
                    if _np is not None:
                        per_aoi_series[aid].setdefault("p10", []).append(float(_np.quantile(vals, 0.10)))
                        per_aoi_series[aid].setdefault("p90", []).append(float(_np.quantile(vals, 0.90)))
                        per_aoi_series[aid].setdefault("median", []).append(float(_np.median(vals)))
                    else:
                        s = sorted(vals); n = len(s)
                        per_aoi_series[aid].setdefault("p10", []).append(s[int(0.10*(n-1))])
                        per_aoi_series[aid].setdefault("p90", []).append(s[int(0.90*(n-1))])
                        per_aoi_series[aid].setdefault("median", []).append(s[n//2])

        # Build rows
        out_rows: List[NDVIStat] = []
        # DOY climatology store per AOI
        clim: Dict[str, Dict[int, List[float]]] = {aid: {} for aid in self.aois.keys()}

        for aid, store in per_aoi_series.items():
            dates = store.get("date", []); nd = store.get("ndvi", [])
            med = store.get("median", []); p10 = store.get("p10", []); p90 = store.get("p90", [])
            if not dates:
                continue
            # sort by date
            order = sorted(range(len(dates)), key=lambda i: dates[i])
            dates = [dates[i] for i in order]
            nd    = [nd[i]    for i in order]
            med   = [med[i]   for i in order]
            p10   = [p10[i]   for i in order]
            p90   = [p90[i]   for i in order]

            sm = _movavg(nd, self.smoothing_days)
            zz = _zscore(sm)

            # build DOY climatology from smoothed series
            for i, d in enumerate(dates):
                doy = _doy(d) # type: ignore
                clim[aid].setdefault(doy, []).append(sm[i])

            # compute anomaly vs climatology (same DOY)
            anom = []
            for i, d in enumerate(dates):
                doy = _doy(d) # type: ignore
                base = clim[aid].get(doy, [])
                if base:
                    mu = sum(base)/len(base)
                    anom.append(sm[i] - mu)
                else:
                    anom.append(None)

            # YoY change (exact calendar day match if present)
            idx = {d: i for i, d in enumerate(dates)}
            yoy = [None]*len(dates)
            for i, d in enumerate(dates):
                try:
                    prev = (datetime.fromisoformat(d) - timedelta(days=365)).date().isoformat() # type: ignore
                except Exception:
                    continue
                j = idx.get(prev) # type: ignore
                if j is not None and nd[j] is not None:
                    base = nd[j]
                    if base not in (None, 0.0):
                        yoy[i] = (nd[i] - base) / (abs(base) + 1e-9) # type: ignore

            # vigor index (0..100) per AOI from smoothed NDVI min-max range
            lo = min(sm); hi = max(sm) if max(sm) > lo else lo + 1e-9
            vigor = [100.0 * (x - lo) / (hi - lo) for x in sm]

            for i in range(len(dates)):
                out_rows.append(NDVIStat(
                    date=dates[i], # type: ignore
                    aoi_id=aid,
                    mean_ndvi=float(nd[i]),
                    median_ndvi=float(med[i]),
                    p10=float(p10[i]),
                    p90=float(p90[i]),
                    smoothed=float(sm[i]),
                    z=float(zz[i]),
                    anomaly_doy=(None if anom[i] is None else float(anom[i])),
                    yoy=(None if yoy[i] is None else float(yoy[i])), # type: ignore
                    vigor_idx=float(vigor[i]),
                    n_px=0  # per-day pixel count not retained post-aggregation
                ))

        # Convert to frames & rollups
        if _pd is not None and return_frames:
            per = _pd.DataFrame([asdict(r) for r in out_rows])
            # attach AOI meta
            meta = _pd.DataFrame([asdict(a) for a in self.aois.values()]).rename(columns={"id": "aoi_id"})
            per = per.merge(meta, on="aoi_id", how="left")

            city = per.groupby(["date","city"], dropna=False).agg(
                mean_ndvi=("mean_ndvi","mean"),
                smoothed=("smoothed","mean"),
                z=("z","mean"),
                anomaly_doy=("anomaly_doy","mean"),
                yoy=("yoy","mean"),
                vigor_idx=("vigor_idx","mean"),
                n_aois=("aoi_id","nunique"),
            ).reset_index()

            region = per.groupby(["date","region"], dropna=False).agg(
                mean_ndvi=("mean_ndvi","mean"),
                smoothed=("smoothed","mean"),
                z=("z","mean"),
                anomaly_doy=("anomaly_doy","mean"),
                yoy=("yoy","mean"),
                vigor_idx=("vigor_idx","mean"),
                n_aois=("aoi_id","nunique"),
            ).reset_index()
        else:
            # lists fallback
            per = [asdict(r) for r in out_rows]
            # naive rollups
            def _roll(key):
                agg: Dict[Tuple[str,str], List[float]] = {}
                for r in out_rows:
                    a = self.aois.get(r.aoi_id)
                    if not a: continue
                    k = (r.date, getattr(a, key))
                    agg.setdefault(k, []).append(r.smoothed)
                out = []
                for (d, label), vals in agg.items():
                    out.append({"date": d, key: label, "smoothed": sum(vals)/len(vals), "n_aois": len(vals)})
                return out
            city = _roll("city")
            region = _roll("region")

        if emit:
            try:
                last_date = per["date"].max() if (_pd is not None and hasattr(per, "max")) else None # type: ignore
                publish_stream(self.emit_stream, {"ts_ms": int(time.time()*1000), "last_date": last_date, "n_aois": len(self.aois)})
            except Exception:
                pass

        return per, city, region

    # ---------- Utils ----------
    @staticmethod
    def to_json(df: Any, limit: Optional[int] = None) -> str:
        if _pd is not None and hasattr(df, "to_dict"):
            recs = df.tail(limit).to_dict(orient="records") if (limit and len(df) > limit) else df.to_dict(orient="records")
        else:
            recs = df[-limit:] if limit else df
        return json.dumps(recs, ensure_ascii=False, indent=2)

# ---------- CLI ----------
def _read(path: str):
    if _pd is None:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return _pd.read_csv(path)

def _write(path: str, df: Any):
    if _pd is not None and hasattr(df, "to_csv"):
        df.to_csv(path, index=False); return
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not df:
            f.write(""); return
        wr = csv.DictWriter(f, fieldnames=list(df[0].keys()))
        wr.writeheader()
        for r in df: wr.writerow(r)

def _main():
    import argparse
    p = argparse.ArgumentParser(description="Compute NDVI per AOI per day from Red/NIR pixels")
    p.add_argument("--aoi", required=True, help="CSV with AOIs (id,name,lat,lon[,radius_m,polygon_wkt,city,region])")
    p.add_argument("--pixels", required=True, help="CSV with pixels (date,lat,lon,red,nir[,cloud_flag,quality])")
    p.add_argument("--out", required=True, help="Output CSV for per-AOI NDVI")
    p.add_argument("--city_out", type=str, default=None)
    p.add_argument("--region_out", type=str, default=None)
    p.add_argument("--smoothing", type=int, default=7)
    p.add_argument("--winsor", type=float, nargs=2, default=(0.02, 0.98))
    p.add_argument("--minpixels", type=int, default=10)
    p.add_argument("--emit", action="store_true")
    args = p.parse_args()

    eng = NDVIEngine(
        smoothing_days=args.smoothing,
        winsor_lo=args.winsor[0], winsor_hi=args.winsor[1],
        min_pixels=args.minpixels
    )
    eng.load_aois(_read(args.aoi))
    per, city, region = eng.compute(_read(args.pixels), emit=args.emit)

    _write(args.out, per)
    if args.city_out: _write(args.city_out, city)
    if args.region_out: _write(args.region_out, region)

if __name__ == "__main__":  # pragma: no cover
    _main()