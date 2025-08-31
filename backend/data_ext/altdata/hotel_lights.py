# backend/altdata/hotel_lights.py
from __future__ import annotations

"""
Hotel Lights (Alt-Data)
-----------------------
Compute hotel activity indices from nighttime lights (e.g., VIIRS DNB).

Inputs
------
1) Hotels metadata CSV (or DataFrame) with at least:
   id,name,lat,lon[,radius_m,city,brand]
   - radius_m: optional (default 300m per hotel)
2) Night-lights CSV (or DataFrame) with either per-pixel or pre-aggregated rows:
   date,lat,lon,radiance[,cloud_flag,quality]
   - date: ISO 'YYYY-MM-DD' or timestamp
   - radiance: brightness (e.g., nW/cm^2/sr)
   - Optional flags will be used to mask bad pixels.

Outputs
-------
Per-hotel daily series with:
  mean_rad, smoothed, zscore, yoy, occupancy_idx (0..100),
  plus city/brand aggregates.

No hard dependencies:
- If available, uses pandas/numpy/shapely for speed & geometry.
- Otherwise falls back to pure-Python Haversine + lists.

CLI
---
Compute & save CSV:
  python -m backend.altdata.hotel_lights \
    --hotels data/hotels.csv \
    --ntl data/viirs_pixels_2024.csv \
    --out data/hotel_lights_index.csv \
    --smoothing 7

Optionally calibrate to ground-truth occupancy:
  python -m backend.altdata.hotel_lights \
    --hotels hotels.csv --ntl viirs.csv --occup occupancy.csv --out index.csv

Environment (optional bus):
  REDIS_HOST, REDIS_PORT, HOTEL_LIGHTS_STREAM (default "alt.hotel_lights")

"""

import csv
import math
import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Optional deps
try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None
try:
    from shapely.geometry import Point  # type: ignore
    from shapely.geometry.polygon import Polygon as _Polygon # type: ignore
    _has_shapely = True
except Exception:
    _has_shapely = False

# Optional bus
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
OUT_STREAM = os.getenv("HOTEL_LIGHTS_STREAM", "alt.hotel_lights")


# --------------------------- Helpers ---------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _winsor(vals: List[float], lo_q=0.05, hi_q=0.95) -> List[float]:
    if not vals:
        return vals
    if _np is not None:
        lo, hi = _np.quantile(vals, [lo_q, hi_q]).tolist()
    else:
        s = sorted(vals); n = len(s)
        lo_i = max(0, int(lo_q * (n - 1))); hi_i = min(n - 1, int(hi_q * (n - 1)))
        lo, hi = s[lo_i], s[hi_i]
    return [max(lo, min(hi, v)) for v in vals]

def _movavg(xs: List[float], w: int) -> List[float]:
    if w <= 1 or not xs:
        return xs
    out, s = [], 0.0
    for i, v in enumerate(xs):
        s += v
        if i >= w: s -= xs[i - w]
        out.append(s / min(i + 1, w))
    return out

def _zscore(xs: List[float]) -> List[float]:
    if not xs:
        return xs
    if _np is not None:
        mu = float(_np.mean(xs))
        sd = float(_np.std(xs)) or 1.0
    else:
        mu = sum(xs)/len(xs)
        sd = (sum((x-mu)**2 for x in xs)/max(1, len(xs)-1))**0.5 or 1.0
    return [(x - mu) / sd for x in xs]

def _yoy(dates: List[str], vals: List[float]) -> List[Optional[float]]:
    # naive 365-day lookback match
    out = [None]*len(vals)
    if not dates: return out # type: ignore
    # build date->index
    idx = {d: i for i, d in enumerate(dates)}
    from datetime import datetime, timedelta
    for i, d in enumerate(dates):
        try:
            dt = datetime.fromisoformat(str(d)[:10])
            prior = (dt.replace(microsecond=0) - timedelta(days=365)).date().isoformat()
            j = idx.get(prior)
            if j is None: continue
            base = vals[j]
            if base is None or base == 0: continue
            out[i] = (vals[i] - base) / base # type: ignore
        except Exception:
            continue
    return out # type: ignore

def _parse_date(s: Any) -> str:
    ss = str(s)
    return ss[:10]  # normalize to YYYY-MM-DD

def _safe_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
        if math.isnan(f): return None
        return f
    except Exception:
        return None


# --------------------------- Data Models ---------------------------

@dataclass
class Hotel:
    id: str
    name: str
    lat: float
    lon: float
    radius_m: float = 300.0
    city: str = ""
    brand: str = ""

@dataclass
class HotelLightsRow:
    date: str
    hotel_id: str
    mean_rad: float
    smoothed: float
    z: float
    yoy: Optional[float]
    occupancy_idx: float  # 0..100 (proxy)
    n_pixels: int

# -------------------------- Core Engine ---------------------------

class HotelLightsIndex:
    def __init__(
        self,
        *,
        default_radius_m: float = 300.0,
        smoothing_days: int = 7,
        winsor_lo: float = 0.05,
        winsor_hi: float = 0.95,
        min_pixels: int = 5,
        mask_cloud: bool = True,
        cloud_col: str = "cloud_flag",
        quality_col: str = "quality",
        quality_min: Optional[float] = None,
        emit_stream: str = OUT_STREAM
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
        # hotels registry
        self.hotels: Dict[str, Hotel] = {}
        # optional occupancy calibration per hotel: y = a*x + b
        self._calib: Dict[str, Tuple[float, float]] = {}
        # bus
        self.emit_stream = emit_stream
        self._r = None
        if _redis is not None:
            try:
                self._r = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            except Exception:
                self._r = None

    # -------- Hotels --------

    def load_hotels(self, hotels: Union[str, List[Dict[str, Any]], Any]) -> None:
        """
        hotels: path to CSV OR list[dict] OR pandas DataFrame
        """
        rows: List[Dict[str, Any]] = []
        if isinstance(hotels, str):
            with open(hotels, "r", newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    rows.append(r)
        elif _pd is not None and hasattr(hotels, "to_dict"):
            rows = hotels.to_dict(orient="records")  # type: ignore
        elif isinstance(hotels, list):
            rows = hotels
        else:
            raise ValueError("Unsupported hotels input")

        for r in rows:
            hid = str(r.get("id") or r.get("hotel_id"))
            if not hid: continue
            lat = _safe_float(r.get("lat")); lon = _safe_float(r.get("lon"))
            if lat is None or lon is None: continue
            rad = _safe_float(r.get("radius_m")) or self.default_radius_m
            h = Hotel(
                id=hid,
                name=str(r.get("name") or hid),
                lat=float(lat), lon=float(lon),
                radius_m=float(rad),
                city=str(r.get("city") or ""),
                brand=str(r.get("brand") or "")
            )
            self.hotels[h.id] = h

    # -------- Occupancy calibration (optional) --------

    def fit_occupancy_calibration(self, occupancy: Union[str, Any]) -> None:
        """
        Fit simple per-hotel linear map: occ ~ a * smoothed_radiance + b
        occupancy CSV/DataFrame needs: date,hotel_id,occupancy (0..1 or 0..100)
        Call after you have run compute() once and stored smoothed radiance per hotel.
        To keep it minimal, we'll compute on the fly from an internal cache you pass into calibrate().
        """
        # This function is a stub to indicate workflow; actual calibration is done in `calibrate_from_frames`
        pass

    def calibrate_from_frames(self, per_hotel_df: Any, occupancy_df: Any) -> None:
        """
        per_hotel_df: DataFrame with columns [date,hotel_id,smoothed]
        occupancy_df: DataFrame with columns [date,hotel_id,occupancy]
        Sets self._calib[hotel_id] = (a,b)
        """
        if _pd is None:
            # Fallback: skip, use min-max later
            return
        ph = per_hotel_df.copy()
        oc = occupancy_df.copy()
        ph["date"] = _pd.to_datetime(ph["date"]).dt.date.astype(str)
        oc["date"] = _pd.to_datetime(oc["date"]).dt.date.astype(str)
        j = ph.merge(oc, on=["date", "hotel_id"], how="inner")
        if j.empty:
            return
        for hid, g in j.groupby("hotel_id"):
            x = _np.asarray(g["smoothed"].values, dtype=float) if _np is not None else list(g["smoothed"].values)
            y = _np.asarray(g["occupancy"].values, dtype=float) if _np is not None else list(g["occupancy"].values)
            if _np is not None and len(x) >= 5:
                X = _np.column_stack([x, _np.ones(len(x))])
                beta, _, _, _ = _np.linalg.lstsq(X, y, rcond=None)
                a, b = float(beta[0]), float(beta[1])
            else:
                # crude two-point scaling
                x0, x1 = min(x), max(x); y0, y1 = min(y), max(y)
                a = (y1 - y0) / (x1 - x0 + 1e-9); b = y0 - a * x0
            self._calib[hid] = (a, b)

    # -------- Core compute --------

    def compute(
        self,
        ntl: Union[str, List[Dict[str, Any]], Any],
        *,
        return_frames: bool = True,
        emit: bool = False
    ) -> Tuple[Any, Any, Any]:
        """
        Aggregate pixels into per-hotel daily metrics and (city/brand) aggregates.

        Returns: (per_hotel_df, city_df, brand_df)
        """
        # Load NTL pixels
        px: List[Dict[str, Any]] = []
        if isinstance(ntl, str):
            with open(ntl, "r", newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    px.append(r)
        elif _pd is not None and hasattr(ntl, "to_dict"):
            px = ntl.to_dict(orient="records")  # type: ignore
        elif isinstance(ntl, list):
            px = ntl
        else:
            raise ValueError("Unsupported night-lights input")

        # Index by date for streaming aggregation
        by_date: Dict[str, List[Dict[str, Any]]] = {}
        for r in px:
            d = _parse_date(r.get("date"))
            rad = _safe_float(r.get("radiance"))
            lat = _safe_float(r.get("lat"))
            lon = _safe_float(r.get("lon"))
            if d is None or rad is None or lat is None or lon is None:  # skip bad rows
                continue
            if self.mask_cloud and (self.cloud_col in r) and str(r[self.cloud_col]).strip() not in ("0", "False", "false", "", "None"):
                # clouded pixel
                continue
            if (self.quality_min is not None) and (self.quality_col in r):
                try:
                    if float(r[self.quality_col]) < float(self.quality_min):
                        continue
                except Exception:
                    pass
            row = {"date": d, "lat": float(lat), "lon": float(lon), "radiance": float(rad)}
            by_date.setdefault(d, []).append(row)

        # For each date, map pixels to hotels by radius
        hotel_ids = list(self.hotels.keys())
        if not hotel_ids:
            raise RuntimeError("No hotels loaded. Call load_hotels() first.")

        # Prepare geometry if shapely present (circles as buffer)
        geoms = {}
        if _has_shapely:
            for h in self.hotels.values():
                # Approximate meters->degrees at this latitude for small buffers:
                # 1 deg lat ~ 111,111 m; lon scaled by cos(lat)
                dy = h.radius_m / 111_111.0
                dx = dy / max(1e-6, math.cos(math.radians(h.lat)))
                # Build rough square polygon around hotel center (fast); circle buffer would need projection
                poly = _Polygon([
                    (h.lon - dx, h.lat - dy),
                    (h.lon - dx, h.lat + dy),
                    (h.lon + dx, h.lat + dy),
                    (h.lon + dx, h.lat - dy),
                ])
                geoms[h.id] = poly

        per_hotel_rows: List[HotelLightsRow] = []
        # also keep raw series store per hotel to compute smoothing/z/yoy
        series_store: Dict[str, Dict[str, List[Any]]] = {hid: {"date": [], "rad": []} for hid in hotel_ids}

        for d, rows in sorted(by_date.items(), key=lambda kv: kv[0]):
            # group pixels per hotel
            buckets: Dict[str, List[float]] = {hid: [] for hid in hotel_ids}
            for p in rows:
                plat, plon, prad = p["lat"], p["lon"], p["radiance"]
                if _has_shapely:
                    pt = Point(plon, plat)
                    for hid, poly in geoms.items():
                        if poly.contains(pt):
                            buckets[hid].append(prad)
                else:
                    # Haversine radius match
                    for h in self.hotels.values():
                        if _haversine_m(h.lat, h.lon, plat, plon) <= h.radius_m:
                            buckets[h.id].append(prad)
            # reduce per hotel
            for hid, vals in buckets.items():
                if len(vals) < self.min_pixels:
                    continue
                vals = _winsor(vals, self.winsor_lo, self.winsor_hi)
                mean_rad = sum(vals) / len(vals)
                series_store[hid]["date"].append(d)
                series_store[hid]["rad"].append(mean_rad)

        # Build per-hotel metrics
        for hid, store in series_store.items():
            dates = store["date"]; rads = store["rad"]
            if not dates:
                continue
            sm = _movavg(rads, self.smoothing_days)
            zz = _zscore(sm)
            yy = _yoy(dates, sm)
            # occupancy proxy: calibrated if available else per-hotel min-max on smoothed
            occ = []
            a_b = self._calib.get(hid)
            if a_b is not None:
                a, b = a_b
                for x in sm:
                    y = a * x + b
                    if y <= 1.5:  # likely 0..1 scale
                        y = max(0.0, min(1.0, y)) * 100.0
                    occ.append(float(max(0.0, min(100.0, y))))
            else:
                lo = min(sm); hi = max(sm) if max(sm) > lo else lo + 1e-9
                for x in sm:
                    occ.append(100.0 * (x - lo) / (hi - lo))
            for i in range(len(dates)):
                per_hotel_rows.append(HotelLightsRow(
                    date=dates[i], hotel_id=hid, mean_rad=rads[i], smoothed=sm[i], z=zz[i],
                    yoy=yy[i], occupancy_idx=occ[i], n_pixels=0  # n_pixels is not kept post-agg here
                ))

        # Frame outputs
        if _pd is not None and return_frames:
            ph = _pd.DataFrame([asdict(r) for r in per_hotel_rows])
            # join hotel meta
            meta = _pd.DataFrame([asdict(h) for h in self.hotels.values()])
            meta = meta.rename(columns={"id": "hotel_id"})
            ph = ph.merge(meta, on="hotel_id", how="left")

            # Aggregates
            city = ph.groupby(["date", "city"], dropna=False).agg(
                mean_rad=("mean_rad", "mean"),
                smoothed=("smoothed", "mean"),
                z=("z", "mean"),
                yoy=("yoy", "mean"),
                occupancy_idx=("occupancy_idx", "mean"),
                n_hotels=("hotel_id", "nunique")
            ).reset_index()

            brand = ph.groupby(["date", "brand"], dropna=False).agg(
                mean_rad=("mean_rad", "mean"),
                smoothed=("smoothed", "mean"),
                z=("z", "mean"),
                yoy=("yoy", "mean"),
                occupancy_idx=("occupancy_idx", "mean"),
                n_hotels=("hotel_id", "nunique")
            ).reset_index()
        else:
            # pure-Python lists
            ph = [asdict(r) for r in per_hotel_rows]
            # simple city/brand aggregates (means)
            city_map: Dict[Tuple[str, str], List[float]] = {}
            brand_map: Dict[Tuple[str, str], List[float]] = {}
            for r in per_hotel_rows:
                h = self.hotels.get(r.hotel_id)
                if not h: continue
                key_c = (r.date, h.city)
                key_b = (r.date, h.brand)
                city_map.setdefault(key_c, []).append(r.smoothed)
                brand_map.setdefault(key_b, []).append(r.smoothed)
            city = [{"date": d, "city": c, "smoothed": sum(v)/len(v), "n_hotels": len(v)} for (d,c), v in city_map.items()]
            brand = [{"date": d, "brand": b, "smoothed": sum(v)/len(v), "n_hotels": len(v)} for (d,b), v in brand_map.items()]

        # emit latest snapshot (optional)
        if emit:
            payload = {
                "ts_ms": int(time.time()*1000),
                "hotels": len(self.hotels),
                "last_date": (ph["date"].max() if (_pd is not None and hasattr(ph, "max")) else None), # type: ignore
            }
            publish_stream(self.emit_stream, payload)

        return ph, city, brand

    # ----------------- Serialization helpers -----------------

    @staticmethod
    def to_json(df: Any, limit: Optional[int] = None) -> str:
        if _pd is not None and hasattr(df, "to_dict"):
            recs = df.tail(limit).to_dict(orient="records") if (limit and len(df) > limit) else df.to_dict(orient="records")
        else:
            recs = df[-limit:] if limit else df
        return json.dumps(recs, ensure_ascii=False, indent=2)


# ------------------------------- CLI --------------------------------

def _read_df(path: str):
    if _pd is None:
        # parse CSV into list of dicts
        out = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.append(r)
        return out
    return _pd.read_csv(path)

def _write_df(path: str, df: Any):
    if _pd is not None and hasattr(df, "to_csv"):
        df.to_csv(path, index=False)
        return
    # list of dicts
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not df:
            f.write("")
            return
        wr = csv.DictWriter(f, fieldnames=list(df[0].keys()))
        wr.writeheader()
        for r in df:
            wr.writerow(r)

def _main():
    import argparse
    p = argparse.ArgumentParser(description="Hotel Lights Index from VIIRS pixels")
    p.add_argument("--hotels", required=True, help="Hotels CSV (id,name,lat,lon[,radius_m,city,brand])")
    p.add_argument("--ntl", required=True, help="Night-lights CSV (date,lat,lon,radiance[,cloud_flag,quality])")
    p.add_argument("--out", required=True, help="Output CSV for per-hotel index")
    p.add_argument("--smoothing", type=int, default=7)
    p.add_argument("--winsor", type=float, nargs=2, default=(0.05, 0.95))
    p.add_argument("--minpixels", type=int, default=5)
    p.add_argument("--emit", action="store_true")
    p.add_argument("--city_out", type=str, default=None)
    p.add_argument("--brand_out", type=str, default=None)
    args = p.parse_args()

    hx = HotelLightsIndex(
        smoothing_days=args.smoothing,
        winsor_lo=args.winsor[0], winsor_hi=args.winsor[1],
        min_pixels=args.minpixels
    )
    hx.load_hotels(_read_df(args.hotels))
    ph, city, brand = hx.compute(_read_df(args.ntl), emit=args.emit)

    _write_df(args.out, ph)
    if args.city_out: _write_df(args.city_out, city)
    if args.brand_out: _write_df(args.brand_out, brand)

if __name__ == "__main__":  # pragma: no cover
    _main()