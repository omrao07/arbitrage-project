# backend/geo/geo_spatial.py
"""
Geospatial utilities for your trading/alt-data stack.

What you get (pure-Python, optional speedups):
- Great-circle distance (Haversine) & bearings
- Point-in-polygon (ray casting) and polygon area (spherical approx)
- Simple geofences (circles/polygons) with batch filters
- Fast-ish spatial lookups via a uniform grid index (no heavy deps)
- Radius queries / nearest neighbors (approx)
- Heatmap binning (lat/lon → tiles or fixed-size grids)
- Optional shapely/geopandas adapters if installed

Typical uses:
- Map news/exhaust signals to store/branch lat-lons, deduplicate by geofence
- Venue proximity filters (e.g., distance to exchange colos)
- Region attribution (aggregate by city/state/custom polygon)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional deps (silently skipped)
try:
    from shapely.geometry import Point as _ShPoint, Polygon as _ShPoly  # type: ignore
except Exception:
    _ShPoint = _ShPoly = None  # type: ignore

EARTH_RADIUS_KM = 6371.0088
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float

    def to_tuple(self) -> Tuple[float, float]:
        return (float(self.lat), float(self.lon))


@dataclass
class GeoPolygon:
    """
    Simple polygon (no holes). Coordinates in (lat, lon) order.
    """
    coords: List[Tuple[float, float]]

    def contains(self, p: GeoPoint) -> bool:
        # Ray casting in lon axis
        x, y = p.lon, p.lat
        inside = False
        n = len(self.coords)
        if n < 3:
            return False
        for i in range(n):
            y1, x1 = self.coords[i][0], self.coords[i][1]
            y2, x2 = self.coords[(i + 1) % n][0], self.coords[(i + 1) % n][1]
            # Check edge straddling y and compute x-intersect
            if ((y1 > y) != (y2 > y)):
                xin = (x2 - x1) * (y - y1) / max(1e-12, (y2 - y1)) + x1
                if xin > x:
                    inside = not inside
        return inside

    def area_km2(self) -> float:
        """
        Approx polygon area on sphere via l'Huilier’s theorem (triangulation).
        Good enough for small/medium polygons (< few 1000 km^2).
        """
        if len(self.coords) < 3:
            return 0.0
        rad = [(lat * DEG2RAD, lon * DEG2RAD) for lat, lon in self.coords]
        # Fan triangulation around first vertex
        A = 0.0
        a0 = rad[0]
        for i in range(1, len(rad) - 1):
            A += _spherical_triangle_area(a0, rad[i], rad[i + 1]) # type: ignore
        return abs(A) * (EARTH_RADIUS_KM ** 2)


# ---------------------------------------------------------------------------
# Distance, bearings, boxes
# ---------------------------------------------------------------------------

def haversine_km(a: GeoPoint | Tuple[float, float], b: GeoPoint | Tuple[float, float]) -> float:
    la1, lo1 = (a.lat, a.lon) if isinstance(a, GeoPoint) else (a[0], a[1])
    la2, lo2 = (b.lat, b.lon) if isinstance(b, GeoPoint) else (b[0], b[1])
    dlat = (la2 - la1) * DEG2RAD
    dlon = (lo2 - lo1) * DEG2RAD
    lat1 = la1 * DEG2RAD
    lat2 = la2 * DEG2RAD
    h = (math.sin(dlat / 2) ** 2) + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))

def bearing_deg(a: GeoPoint, b: GeoPoint) -> float:
    lat1, lat2 = a.lat * DEG2RAD, b.lat * DEG2RAD
    dlon = (b.lon - a.lon) * DEG2RAD
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.atan2(y, x) * RAD2DEG
    return (brng + 360.0) % 360.0

def destination_point(start: GeoPoint, bearing_deg_: float, distance_km: float) -> GeoPoint:
    br = bearing_deg_ * DEG2RAD
    lat1 = start.lat * DEG2RAD
    lon1 = start.lon * DEG2RAD
    dr = distance_km / EARTH_RADIUS_KM
    lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(dr) * math.cos(lat1), math.cos(dr) - math.sin(lat1) * math.sin(lat2))
    return GeoPoint(lat2 * RAD2DEG, ((lon2 * RAD2DEG + 540) % 360) - 180)

def bbox_from_point_radius(center: GeoPoint, radius_km: float) -> Tuple[float, float, float, float]:
    """
    Returns (min_lat, min_lon, max_lat, max_lon) bounding box for a circle radius_km.
    """
    lat_delta = (radius_km / EARTH_RADIUS_KM) * RAD2DEG
    lon_delta = lat_delta / max(1e-12, math.cos(center.lat * DEG2RAD))
    return (center.lat - lat_delta, center.lon - lon_delta, center.lat + lat_delta, center.lon + lon_delta)


# ---------------------------------------------------------------------------
# Geofences
# ---------------------------------------------------------------------------

@dataclass
class CircleFence:
    center: GeoPoint
    radius_km: float
    def contains(self, p: GeoPoint) -> bool:
        return haversine_km(self.center, p) <= self.radius_km + 1e-9

@dataclass
class PolygonFence:
    polygon: GeoPolygon
    def contains(self, p: GeoPoint) -> bool:
        if _ShPoly and isinstance(self.polygon, GeoPolygon):
            # If shapely available and polygon is big/complex, we could accelerate
            pass
        return self.polygon.contains(p)

def filter_points_in_fences(points: Iterable[GeoPoint], fences: Sequence[CircleFence | PolygonFence]) -> List[GeoPoint]:
    out: List[GeoPoint] = []
    for p in points:
        for f in fences:
            if f.contains(p):
                out.append(p)
                break
    return out


# ---------------------------------------------------------------------------
# Uniform grid index (fast approximate spatial lookups)
# ---------------------------------------------------------------------------

class GridIndex:
    """
    Uniform grid over lat/lon with user-specified cell size (in degrees).
    Ideal for quick radius/NN queries without heavy spatial libs.

    cell_dlat ~ 0.01 ≈ 1.11 km in latitude. Longitude varies with cos(lat).
    """
    def __init__(self, cell_dlat: float = 0.02, cell_dlon: Optional[float] = None):
        self.dlat = float(cell_dlat)
        self.dlon = float(cell_dlon) if cell_dlon else float(cell_dlat)
        self._cells: Dict[Tuple[int, int], List[Tuple[GeoPoint, Any]]] = {}

    def _key(self, lat: float, lon: float) -> Tuple[int, int]:
        i = int(math.floor((lat + 90.0) / self.dlat))
        j = int(math.floor((lon + 180.0) / self.dlon))
        return (i, j)

    def insert(self, p: GeoPoint, payload: Any = None) -> None:
        k = self._key(p.lat, p.lon)
        self._cells.setdefault(k, []).append((p, payload))

    def bulk_insert(self, pts: Iterable[Tuple[GeoPoint, Any]]) -> None:
        for p, pay in pts:
            self.insert(p, pay)

    def _neighbor_keys(self, lat: float, lon: float, radius_km: float) -> List[Tuple[int, int]]:
        # Expand by enough cells to cover radius (conservative)
        lat_deg = radius_km / EARTH_RADIUS_KM * RAD2DEG
        lon_deg = lat_deg / max(1e-12, math.cos(lat * DEG2RAD))
        di = max(1, int(math.ceil(lat_deg / self.dlat)))
        dj = max(1, int(math.ceil(lon_deg / self.dlon)))
        i0, j0 = self._key(lat, lon)
        return [(i, j) for i in range(i0 - di, i0 + di + 1) for j in range(j0 - dj, j0 + dj + 1)]

    def radius_query(self, center: GeoPoint, radius_km: float) -> List[Tuple[GeoPoint, Any, float]]:
        out: List[Tuple[GeoPoint, Any, float]] = []
        for k in self._neighbor_keys(center.lat, center.lon, radius_km):
            for p, pay in self._cells.get(k, []):
                d = haversine_km(center, p)
                if d <= radius_km:
                    out.append((p, pay, d))
        out.sort(key=lambda x: x[2])
        return out

    def nearest(self, p: GeoPoint, k: int = 1, search_km: float = 50.0) -> List[Tuple[GeoPoint, Any, float]]:
        """
        Return up to k nearest points within search_km (expand if empty).
        """
        r = max(0.5, float(search_km))
        for _ in range(4):  # expand a few times if dense area
            cand = self.radius_query(p, r)
            if cand:
                return cand[:k]
            r *= 2.0
        return []


# ---------------------------------------------------------------------------
# Heatmap binning (grid or slippy tiles)
# ---------------------------------------------------------------------------

def grid_bin(points: Iterable[GeoPoint], dlat: float = 0.05, dlon: Optional[float] = None) -> Dict[Tuple[int, int], int]:
    """
    Count points per grid cell. Returns {(i,j): count}.
    """
    dlon = dlon if dlon is not None else dlat
    def key(pt: GeoPoint) -> Tuple[int, int]:
        i = int(math.floor((pt.lat + 90.0) / dlat))
        j = int(math.floor((pt.lon + 180.0) / dlon))
        return (i, j)
    counts: Dict[Tuple[int, int], int] = {}
    for p in points:
        counts[key(p)] = counts.get(key(p), 0) + 1
    return counts

def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Web Mercator slippy map tiling (x,y) for a given zoom.
    """
    lat_rad = lat * DEG2RAD
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / max(1e-12, math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (max(0, min(int(n) - 1, xtile)), max(0, min(int(n) - 1, ytile)))

def tile_to_bounds(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    """
    Returns lat/lon bounds for a tile: (min_lat, min_lon, max_lat, max_lon)
    """
    n = 2.0 ** zoom
    lon1 = x / n * 360.0 - 180.0
    lon2 = (x + 1) / n * 360.0 - 180.0
    lat1 = _merc_to_lat(math.pi * (1 - 2 * y / n))
    lat2 = _merc_to_lat(math.pi * (1 - 2 * (y + 1) / n))
    return (lat2, lon1, lat1, lon2)

def _merc_to_lat(merc_y: float) -> float:
    return (math.atan(math.sinh(merc_y)) * RAD2DEG)


# ---------------------------------------------------------------------------
# Optional shapely helpers
# ---------------------------------------------------------------------------

def to_shapely_polygon(poly: GeoPolygon):
    if _ShPoly is None:
        raise RuntimeError("shapely not installed")
    # shapely expects (lon, lat) order
    return _ShPoly([(lon, lat) for lat, lon in poly.coords])

def shapely_contains(poly: GeoPolygon, p: GeoPoint) -> bool:
    if _ShPoly is None:
        raise RuntimeError("shapely not installed")
    return to_shapely_polygon(poly).contains(_ShPoint(p.lon, p.lat)) # type: ignore


# ---------------------------------------------------------------------------
# Tiny demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Basics
    a = GeoPoint(19.0760, 72.8777)  # Mumbai
    b = GeoPoint(28.6139, 77.2090)  # Delhi
    print("Mumbai–Delhi km:", round(haversine_km(a, b), 2))
    print("Bearing:", round(bearing_deg(a, b), 1))

    # Geofence
    circle = CircleFence(center=a, radius_km=5.0)
    print("Circle contains center?", circle.contains(a))

    # Polygon (rough Mumbai bounding polygon)
    poly = GeoPolygon(coords=[(19.2, 72.77), (19.2, 72.98), (18.9, 72.98), (18.9, 72.77)])
    print("Polygon area km^2:", round(poly.area_km2(), 2))
    print("Polygon contains a?", PolygonFence(poly).contains(a))

    # Grid index NN
    idx = GridIndex(cell_dlat=0.02)
    pts = [GeoPoint(19.07 + i*0.01, 72.88 + i*0.01) for i in range(20)]
    idx.bulk_insert([(p, {"id": i}) for i, p in enumerate(pts)])
    print("Nearest to center:", idx.nearest(a, k=3))

    # Heatmap tiles
    z = 10
    tx, ty = latlon_to_tile(a.lat, a.lon, z)
    print("Tile:", (tx, ty), "bounds:", tile_to_bounds(tx, ty, z))