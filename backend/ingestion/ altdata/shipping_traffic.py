# backend/altdata/shipping_traffic.py
"""
AIS Shipping Traffic -> Port/Route Activity Index
-------------------------------------------------
Features
- Ingest AIS pings (CSV/JSON/parquet*): ts, mmsi/imo, lat, lon, sog (kn), cog, nav_status
- Register Ports/Areas (GeoJSON polygons) & Waypoints (Points/Lines for channels & chokepoints)
- Detect arrivals/departures, dwell time, queue length (vessels near port but not docked),
  average speed, daily/weekly traffic counts, and congestion score
- Aggregate by port, waterway (e.g., 'SUEZ', 'MALACCA'), route (port->port), vessel class
- Optional mapping MMSI/IMO -> company/flag/commodity/ticker
- Persist to SQLite (runtime/altdata.db), export CSV, and publish "movers" to bus

* Parquet input uses pyarrow if installed.

Expected AIS columns (case-insensitive, aliases allowed):
  ts|timestamp (epoch s/ms/us/ns or ISO), mmsi|imo, lat|latitude, lon|longitude,
  sog|speed|speed_over_ground, cog|course, nav_status|status, type|vessel_type, draught|draft

CLI quickstart
  # Register geometry
  python -m backend.altdata.shipping_traffic --ports config/ports.geojson
  python -m backend.altdata.shipping_traffic --chokepoints config/chokepoints.geojson

  # Ingest AIS & build index
  python -m backend.altdata.shipping_traffic --load data/ais_2025-08-01.csv --index

  # Export + publish top movers
  python -m backend.altdata.shipping_traffic --export runtime/shipping_index.csv --publish
"""

from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional deps
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pq = None  # type: ignore

try:
    from shapely.geometry import shape, Point, LineString  # type: ignore
    from shapely.ops import nearest_points  # type: ignore
except Exception:
    shape = Point = LineString = nearest_points = None  # type: ignore

# Optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore

DB_PATH = "runtime/altdata.db"

# ------------------------ utils ------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _parse_ts(x) -> int:
    if x is None:
        return _utc_ms()
    try:
        if isinstance(x, (int, float)):
            v = float(x)
            # ns/us/ms/s
            if v > 1e16: return int(v / 1e6)
            if v > 1e14: return int(v / 1e3)
            if v > 1e12: return int(v)
            return int(v * 1000)
        s = str(x).strip().replace("Z", "+00:00")
        from datetime import datetime
        try:
            return int(datetime.fromisoformat(s).timestamp() * 1000)
        except Exception:
            # date-only
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
            return int(time.mktime(dt.timetuple()) * 1000)
    except Exception:
        return _utc_ms()

def _day(ms: int) -> str:
    t = time.gmtime(ms/1000)
    return f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"

def _lower(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}

def _get(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

# ------------------------ storage ------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS ports (
  id TEXT PRIMARY KEY,
  name TEXT,
  geom_geojson TEXT,  -- polygon or point buffer
  props TEXT
);
CREATE TABLE IF NOT EXISTS chokepoints (
  id TEXT PRIMARY KEY,
  name TEXT,
  geom_geojson TEXT,  -- line or polygon
  props TEXT
);
CREATE TABLE IF NOT EXISTS ais (
  ts_ms INTEGER,
  day TEXT,
  mmsi TEXT,
  imo TEXT,
  lat REAL,
  lon REAL,
  sog REAL,
  cog REAL,
  nav_status TEXT,
  vtype TEXT
);
CREATE INDEX IF NOT EXISTS ix_ais_day ON ais(day);
CREATE INDEX IF NOT EXISTS ix_ais_mmsi ON ais(mmsi);

CREATE TABLE IF NOT EXISTS vessel_map (
  mmsi TEXT PRIMARY KEY,
  imo TEXT,
  company TEXT,
  flag TEXT,
  commodity TEXT,
  ticker TEXT
);

-- daily metrics by port / chokepoint
CREATE TABLE IF NOT EXISTS shipping_index (
  level TEXT,          -- 'port' | 'choke'
  key TEXT,            -- port_id or chokepoint_id
  day TEXT,
  arrivals INTEGER,
  departures INTEGER,
  unique_vessels INTEGER,
  avg_speed_kn REAL,
  dwell_hours REAL,        -- avg dwell among vessels that stopped
  queue_len REAL,          -- vessels within 5-20km moving slowly
  congestion REAL,         -- composite score (0-100)
  meta TEXT,
  PRIMARY KEY(level,key,day)
);
CREATE INDEX IF NOT EXISTS ix_ship_day ON shipping_index(day);
"""

class ShipStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        _ensure_dir(db_path)
        with self._cx() as cx:
            cx.executescript(_SCHEMA)

    def _cx(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self.db_path, timeout=60.0)
        cx.row_factory = sqlite3.Row
        return cx

    # ---- geometry registration ----
    def upsert_ports_geojson(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        feats = gj.get("features") or []
        n = 0
        with self._cx() as cx:
            for ft in feats:
                props = ft.get("properties") or {}
                pid = str(props.get("id") or props.get("ID") or props.get("name")).strip()
                if not pid: continue
                cx.execute("INSERT OR REPLACE INTO ports(id,name,geom_geojson,props) VALUES(?,?,?,?)",
                           (pid, props.get("name", pid), json.dumps(ft.get("geometry")), json.dumps(props)))
                n += 1
            cx.commit()
        return n

    def upsert_chokepoints_geojson(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        feats = gj.get("features") or []
        n = 0
        with self._cx() as cx:
            for ft in feats:
                props = ft.get("properties") or {}
                cid = str(props.get("id") or props.get("name")).strip()
                if not cid: continue
                cx.execute("INSERT OR REPLACE INTO chokepoints(id,name,geom_geojson,props) VALUES(?,?,?,?)",
                           (cid, props.get("name", cid), json.dumps(ft.get("geometry")), json.dumps(props)))
                n += 1
            cx.commit()
        return n

    # ---- vessel mapping ----
    def upsert_vessel_map(self, mapping_rows: Iterable[Dict[str, Any]]) -> int:
        n = 0
        with self._cx() as cx:
            for r in mapping_rows:
                rr = _lower(r)
                cx.execute("""INSERT OR REPLACE INTO vessel_map(mmsi,imo,company,flag,commodity,ticker)
                              VALUES(?,?,?,?,?,?)""",
                           (str(_get(rr,"mmsi")), str(_get(rr,"imo") or ""),
                            _get(rr,"company") or "", _get(rr,"flag") or "",
                            _get(rr,"commodity") or "", _get(rr,"ticker") or ""))
                n += 1
            cx.commit()
        return n

    # ---- ingest AIS ----
    def ingest_rows(self, rows: Iterable[Dict[str, Any]]) -> int:
        n = 0
        with self._cx() as cx:
            for r in rows:
                rr = _lower(r)
                ts = _parse_ts(_get(rr,"ts","timestamp","time"))
                lat = _get(rr,"lat","latitude"); lon = _get(rr,"lon","longitude")
                if lat is None or lon is None: continue
                mmsi = str(_get(rr,"mmsi","m")) if _get(rr,"mmsi","m") else None
                imo  = str(_get(rr,"imo")) if _get(rr,"imo") else ""
                sog  = float(_get(rr,"sog","speed","speed_over_ground") or 0.0)
                cog  = float(_get(rr,"cog","course") or 0.0)
                nav  = str(_get(rr,"nav_status","status") or "")
                vtyp = str(_get(rr,"type","vessel_type") or "")
                cx.execute("""INSERT INTO ais(ts_ms,day,mmsi,imo,lat,lon,sog,cog,nav_status,vtype)
                              VALUES(?,?,?,?,?,?,?,?,?,?)""",
                           (ts, _day(ts), mmsi, imo, float(lat), float(lon), sog, cog, nav, vtyp))
                n += 1
            cx.commit()
        return n

    def load_csv(self, path: str, delimiter: str = ",") -> int:
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter=delimiter)
            return self.ingest_rows(rdr)

    def load_json(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("rows", [])
        return self.ingest_rows(rows)

    def load_parquet(self, path: str) -> int:
        assert pq is not None, "pyarrow not installed"
        table = pq.read_table(path)
        df = table.to_pandas()
        return self.ingest_rows(df.to_dict(orient="records"))

    # ---- queries for geometry ----
    def list_ports(self) -> List[Dict[str, Any]]:
        with self._cx() as cx:
            return [dict(r) for r in cx.execute("SELECT * FROM ports").fetchall()]

    def list_chokes(self) -> List[Dict[str, Any]]:
        with self._cx() as cx:
            return [dict(r) for r in cx.execute("SELECT * FROM chokepoints").fetchall()]

# ------------------------ analytics ------------------------

def _contains_point(geom_geojson: Dict[str, Any], lat: float, lon: float) -> bool:
    if shape is None:
        # fallback: simple bbox test if polygon with bbox in properties not present -> approximate 0
        g = geom_geojson
        try:
            xs = [p[0] for ring in g.get("coordinates", [[]]) for p in ring]
            ys = [p[1] for ring in g.get("coordinates", [[]]) for p in ring]
            if not xs or not ys: return False
            return (min(ys) <= lat <= max(ys)) and (min(xs) <= lon <= max(xs))
        except Exception:
            return False
    return shape(geom_geojson).contains(Point(lon, lat)) # type: ignore

def _distance_to_geom_km(geom_geojson: Dict[str, Any], lat: float, lon: float) -> float:
    if shape is None:
        # fallback to distance from centroid
        try:
            coords = geom_geojson.get("coordinates", [[]])[0]
            cx = sum(x for x, _ in coords)/max(1,len(coords))
            cy = sum(y for _, y in coords)/max(1,len(coords))
            return _haversine_km(lat, lon, cy, cx)
        except Exception:
            return 1e9
    g = shape(geom_geojson)
    p = Point(lon, lat) # type: ignore
    try:
        return float(g.distance(p)) * 111.0  # degrees->km rough
    except Exception:
        # nearest point fallback
        if nearest_points:
            q = nearest_points(g, p)[0]
            return _haversine_km(lat, lon, q.y, q.x)
        return 1e9

@dataclass
class IndexParams:
    stop_kn: float = 0.5          # sog<= kn → considered stopped
    near_km: float = 5.0          # within port polygon OR <= near_km from port centroid
    queue_band_km: Tuple[float,float] = (5.0, 20.0)  # ring around port for queue estimate
    choke_buffer_km: float = 10.0 # distance from chokepoint line/polygon to count passing
    min_pings_for_dwell: int = 3  # require N low-speed pings to consider dwell event

def build_daily_index(store: ShipStore, params: IndexParams = IndexParams()) -> int:
    """
    Compute per-day metrics for each port & chokepoint:
      arrivals, departures, unique_vessels, avg_speed, dwell_hours, queue_len, congestion
    """
    ports = store.list_ports()
    chokes = store.list_chokes()
    if not ports and not chokes:
        return 0

    written = 0
    with store._cx() as cx:
        days = [r["day"] for r in cx.execute("SELECT DISTINCT day FROM ais ORDER BY day").fetchall()]
        for day in days:
            # load ais of that day
            rows = cx.execute("SELECT * FROM ais WHERE day=?", (day,)).fetchall()
            pings = [dict(r) for r in rows]
            # group by vessel
            by_vessel: Dict[str, List[Dict[str, Any]]] = {}
            for p in pings:
                k = p["mmsi"] or p["imo"] or ""
                if not k: continue
                by_vessel.setdefault(k, []).append(p)
            for k in by_vessel.keys():
                by_vessel[k].sort(key=lambda r: r["ts_ms"])

            # ---- PORTS ----
            for port in ports:
                gid = port["id"]; geom = json.loads(port["geom_geojson"])
                # per vessel states
                arrivals = departures = 0
                stopped_count = 0
                total_stop_ms = 0
                seen_vessels = set()
                queue_vessels = set()
                speed_sum = 0.0; speed_n = 0

                for vid, seq in by_vessel.items():
                    in_port_prev = False
                    low_speed_streak = 0
                    stop_start_ms: Optional[int] = None
                    last_ping = None
                    near_for_queue_prev = False

                    for ping in seq:
                        lat, lon = ping["lat"], ping["lon"]
                        sog = float(ping["sog"] or 0.0)
                        ts  = int(ping["ts_ms"])

                        inside = _contains_point(geom, lat, lon)
                        dist_km = 0.0 if inside else _distance_to_geom_km(geom, lat, lon)
                        near_for_queue = (params.queue_band_km[0] <= dist_km <= params.queue_band_km[1] and sog <= 2.0)

                        # arrivals/departures transition
                        if inside and not in_port_prev:
                            arrivals += 1
                            seen_vessels.add(vid)
                        if (not inside) and in_port_prev:
                            departures += 1

                        # dwell detection (stopped inside or very close)
                        if (inside or dist_km <= params.near_km) and sog <= params.stop_kn:
                            low_speed_streak += 1
                            if stop_start_ms is None:
                                stop_start_ms = ts
                        else:
                            if stop_start_ms is not None and low_speed_streak >= params.min_pings_for_dwell:
                                total_stop_ms += (ts - stop_start_ms)
                                stopped_count += 1
                            stop_start_ms = None
                            low_speed_streak = 0

                        # queue estimate
                        if near_for_queue and not near_for_queue_prev:
                            queue_vessels.add(vid)

                        # averages
                        if inside:
                            speed_sum += sog; speed_n += 1

                        in_port_prev = inside
                        near_for_queue_prev = near_for_queue
                        last_ping = ping

                    # finalize lingering stop at day end
                    if stop_start_ms is not None and low_speed_streak >= params.min_pings_for_dwell:
                        last_ts = seq[-1]["ts_ms"]
                        total_stop_ms += (last_ts - stop_start_ms)
                        stopped_count += 1

                avg_speed = (speed_sum / speed_n) if speed_n else None
                dwell_hours = (total_stop_ms/3600_000.0 / max(1, stopped_count)) if stopped_count else None
                queue_len = float(len(queue_vessels))
                unique_vessels = len(seen_vessels)

                # simple congestion score (0-100)
                congestion = 0.0
                if unique_vessels:
                    # normalize via heuristic scalers
                    q_term = min(100.0, queue_len * 2.0)           # each queued vessel ~2 pts
                    d_term = min(100.0, (dwell_hours or 0)*10.0)   # 6h dwell ~60
                    s_term = 0.0 if not avg_speed else max(0.0, 25 - min(25.0, avg_speed)) * 2.0  # slower -> more congested
                    congestion = min(100.0, q_term*0.5 + d_term*0.3 + s_term*0.2)

                meta = {"calc":"heuristic_v1","stopped_count":stopped_count}
                cx.execute("""
                    INSERT OR REPLACE INTO shipping_index(level,key,day,arrivals,departures,unique_vessels,
                                                          avg_speed_kn,dwell_hours,queue_len,congestion,meta)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """, ("port", gid, day, arrivals, departures, unique_vessels,
                      avg_speed, dwell_hours, queue_len, congestion, json.dumps(meta)))
                written += 1

            # ---- CHOKEPOINTS ----
            for chok in chokes:
                cid = chok["id"]; geom = json.loads(chok["geom_geojson"])
                passes = 0
                avg_speed = 0.0; n_spd = 0
                unique_vessels = set()

                for vid, seq in by_vessel.items():
                    near_prev = False
                    for ping in seq:
                        lat, lon = ping["lat"], ping["lon"]
                        sog = float(ping["sog"] or 0.0)
                        dist = _distance_to_geom_km(geom, lat, lon)
                        near = dist <= IndexParams().choke_buffer_km
                        if near: unique_vessels.add(vid)
                        if near:
                            avg_speed += sog; n_spd += 1
                        if near and not near_prev:
                            passes += 1
                        near_prev = near

                avg_speed_kn = (avg_speed/n_spd) if n_spd else None
                congestion = None
                cx.execute("""
                    INSERT OR REPLACE INTO shipping_index(level,key,day,arrivals,departures,unique_vessels,
                                                          avg_speed_kn,dwell_hours,queue_len,congestion,meta)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """, ("choke", cid, day, passes, None, len(unique_vessels),
                      avg_speed_kn, None, None, congestion, json.dumps({"calc":"pass_count"})))
                written += 1

        cx.commit()
    return written

# ------------------------ publish & export ------------------------

def publish_movers(store: ShipStore, *, top_k: int = 10) -> None:
    if not publish_stream:
        return
    with store._cx() as cx:
        row = cx.execute("SELECT day FROM shipping_index ORDER BY day DESC LIMIT 1").fetchone()
        if not row: return
        day = row["day"]
        movers = cx.execute("""
            SELECT * FROM shipping_index
            WHERE day=? AND level='port'
            ORDER BY congestion DESC, queue_len DESC
            LIMIT ?
        """, (day, top_k)).fetchall()
        payload = {
            "ts_ms": _utc_ms(),
            "day": day,
            "top_ports": [dict(r) for r in movers]
        }
        publish_stream("altdata.shipping_traffic", payload)
        if movers:
            m0 = dict(movers[0])
            publish_stream("ai.insight", {
                "ts_ms": payload["ts_ms"],
                "kind": "shipping",
                "summary": f"Port {m0['key']} congestion {m0.get('congestion') or 0:.1f}",
                "details": [f"arrivals={m0.get('arrivals')}, queue~{m0.get('queue_len')}, dwell={m0.get('dwell_hours') or 0:.1f}h"],
                "tags": ["shipping","port", m0["key"]]
            })

def export_index_csv(store: ShipStore, path: str) -> str:
    _ensure_dir(path)
    with store._cx() as cx, open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = [c[1] for c in cx.execute("PRAGMA table_info(shipping_index)").fetchall()]
        w.writerow(cols)
        for r in cx.execute("SELECT * FROM shipping_index ORDER BY day, level, key").fetchall():
            w.writerow([r[c] for c in cols])
    return path

# ------------------------ CLI ------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="AIS Shipping Traffic -> Port/Route Activity Index")
    ap.add_argument("--db", type=str, default=DB_PATH)

    ap.add_argument("--ports", type=str, help="GeoJSON ports polygons/points")
    ap.add_argument("--chokepoints", type=str, help="GeoJSON chokepoints (lines/polygons)")

    ap.add_argument("--load", type=str, help="AIS file (.csv/.json/.parquet)")
    ap.add_argument("--delimiter", type=str, default=",")
    ap.add_argument("--index", action="store_true", help="Rebuild daily index after ingest")

    ap.add_argument("--export", type=str, help="Export shipping_index to CSV")
    ap.add_argument("--publish", action="store_true", help="Publish latest movers to bus")

    ap.add_argument("--probe", action="store_true", help="Synthetic demo with fake AIS")
    args = ap.parse_args()

    store = ShipStore(db_path=args.db)

    if args.ports:
        n = store.upsert_ports_geojson(args.ports)
        print(f"Registered {n} ports from {args.ports}")

    if args.chokepoints:
        n = store.upsert_chokepoints_geojson(args.chokepoints)
        print(f"Registered {n} chokepoints from {args.chokepoints}")

    if args.probe:
        # Minimal synthetic sample: 3 vessels loitering near PORT_A and passing CHOKE_X
        if not store.list_ports():
            demo_ports = {
                "PORT_A": {"type":"Polygon","coordinates":[[[72.80,18.90],[72.90,18.90],[72.90,19.00],[72.80,19.00],[72.80,18.90]]]},
                "PORT_B": {"type":"Polygon","coordinates":[[[103.95,1.20],[104.05,1.20],[104.05,1.30],[103.95,1.30],[103.95,1.20]]]},
            }
            with store._cx() as cx:
                for pid, geom in demo_ports.items():
                    cx.execute("INSERT OR REPLACE INTO ports(id,name,geom_geojson,props) VALUES(?,?,?,?)",
                               (pid, pid, json.dumps(geom), json.dumps({})))
                cx.commit()
        if not store.list_chokes():
            demo_ck = {"SUEZ_LINE":{"type":"LineString","coordinates":[[32.3,30.0],[32.6,30.6]]}}
            with store._cx() as cx:
                for cid, geom in demo_ck.items():
                    cx.execute("INSERT OR REPLACE INTO chokepoints(id,name,geom_geojson,props) VALUES(?,?,?,?)",
                               (cid, cid, json.dumps(geom), json.dumps({})))
                cx.commit()
        # Fake AIS
        now = _utc_ms()
        rows = []
        for i in range(40):
            t = now - (39-i)*15*60*1000
            # Vessel 1 loiter near PORT_A
            rows.append({"ts": t, "mmsi": "100001", "lat": 18.955, "lon": 72.86, "sog": 0.2, "cog": 10})
            # Vessel 2 approaches and leaves
            rows.append({"ts": t, "mmsi": "100002", "lat": 18.905 + 0.002*i, "lon": 72.805 + 0.002*i, "sog": 6.0, "cog": 45})
            # Vessel 3 passes SUEZ
            rows.append({"ts": t, "mmsi": "100003", "lat": 30.05 + 0.001*i, "lon": 32.35 + 0.001*i, "sog": 12.0, "cog": 30})
        store.ingest_rows(rows)
        build_daily_index(store)
        if args.publish:
            publish_movers(store)
        if args.export:
            p = export_index_csv(store, args.export)
            print(f"Wrote {p}")
        return

    if args.load:
        ext = os.path.splitext(args.load)[1].lower()
        if ext == ".csv":
            n = store.load_csv(args.load, delimiter=args.delimiter)
        elif ext in (".json", ".ndjson"):
            n = store.load_json(args.load)
        elif ext in (".parquet", ".pq"):
            n = store.load_parquet(args.load)
        else:
            raise SystemExit("Unsupported --load type (csv/json/parquet)")
        print(f"Ingested {n} AIS rows from {args.load}")

    if args.index:
        rows = build_daily_index(store)
        print(f"Indexed {rows} daily rows")

    if args.publish:
        publish_movers(store)
        print("Published latest shipping movers")

    if args.export:
        p = export_index_csv(store, args.export)
        print(f"Wrote {p}")

if __name__ == "__main__":
    main()