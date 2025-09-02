# platform/calendars.py
from __future__ import annotations

import dataclasses
import datetime as dt
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import yaml

try:  # Python 3.9+
    from zoneinfo import ZoneInfo  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Python 3.9+ with zoneinfo is required") from e


# ---------- Data structures ----------

@dataclasses.dataclass(frozen=True)
class EarlyClose:
    date: dt.date
    close: Optional[dt.time] = None
    note: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class RegionCalendar:
    region: str
    tz: ZoneInfo
    # Trading hours map: weekday (0=Mon ... 6=Sun) -> list of (start, end) times (local)
    weekday_sessions: Dict[int, List[Tuple[dt.time, dt.time]]]
    holidays: set
    early_closes: Dict[dt.date, EarlyClose]

    def sessions_for_date(self, d_local: dt.date) -> List[Tuple[dt.time, dt.time]]:
        """Return list of (start,end) times for a given local date, adjusted for early close."""
        wd = d_local.weekday()
        sessions = list(self.weekday_sessions.get(wd, []))
        if not sessions:
            return []

        # Apply early close if configured for this date
        ec = self.early_closes.get(d_local)
        if ec and ec.close:
            # Adjust last session's end to early close
            start, _ = sessions[-1]
            sessions[-1] = (start, ec.close)
        return sessions

    def session_bounds(self, d_local: dt.date) -> Optional[Tuple[dt.datetime, dt.datetime]]:
        """Return (session_start, session_end) as timezone-aware datetimes for the full day."""
        sessions = self.sessions_for_date(d_local)
        if not sessions:
            return None
        start_dt = dt.datetime.combine(d_local, sessions[0][0], self.tz)
        end_dt = dt.datetime.combine(d_local, sessions[-1][1], self.tz)
        return (start_dt, end_dt)

    def is_holiday(self, d_local: dt.date) -> bool:
        return d_local in self.holidays

    def is_open_at(self, ts: dt.datetime) -> bool:
        """Check if market is open at timezone-aware timestamp ts (any tz)."""
        ts_local = ts.astimezone(self.tz)
        d = ts_local.date()

        if self.is_holiday(d):
            return False

        sessions = self.sessions_for_date(d)
        for start_t, end_t in sessions:
            start = dt.datetime.combine(d, start_t, self.tz)
            end = dt.datetime.combine(d, end_t, self.tz)
            if start <= ts_local <= end:
                return True
        return False

    def next_open_after(self, ts: dt.datetime) -> Optional[dt.datetime]:
        """Next opening time (>= ts). Returns tz-aware datetime or None if not found in 365 days."""
        ts_local = ts.astimezone(self.tz)
        d = ts_local.date()
        # If still before today's first session start, return today's open
        sessions_today = self.sessions_for_date(d)
        if not self.is_holiday(d) and sessions_today:
            first_open = dt.datetime.combine(d, sessions_today[0][0], self.tz)
            last_close = dt.datetime.combine(d, sessions_today[-1][1], self.tz)
            if ts_local <= first_open:
                return first_open
            if first_open <= ts_local <= last_close:
                # Already open; return ts itself (market is open now)
                return ts_local

        # Otherwise, scan forward up to 365 days
        for i in range(1, 366):
            d2 = d + dt.timedelta(days=i)
            if self.is_holiday(d2):
                continue
            sessions = self.sessions_for_date(d2)
            if sessions:
                return dt.datetime.combine(d2, sessions[0][0], self.tz)
        return None

    def previous_close_before(self, ts: dt.datetime) -> Optional[dt.datetime]:
        """Previous closing time (< ts). Returns tz-aware datetime or None if not found in 365 days."""
        ts_local = ts.astimezone(self.tz)
        d = ts_local.date()

        # If after today's last close, return today's last close
        if not self.is_holiday(d):
            sessions_today = self.sessions_for_date(d)
            if sessions_today:
                last_close = dt.datetime.combine(d, sessions_today[-1][1], self.tz)
                if ts_local > last_close:
                    return last_close
                # If in-session, previous close is the previous day's last close
                first_open = dt.datetime.combine(d, sessions_today[0][0], self.tz)
                if first_open <= ts_local <= last_close:
                    # previous trading day's close
                    pass
                else:
                    # before today's open → yesterday's close
                    pass

        # Scan backward up to 365 days
        for i in range(1, 366):
            d2 = d - dt.timedelta(days=i)
            if self.is_holiday(d2):
                continue
            sessions = self.sessions_for_date(d2)
            if sessions:
                return dt.datetime.combine(d2, sessions[-1][1], self.tz)
        return None

    def trading_minutes_on(self, d_local: dt.date) -> int:
        """Total scheduled trading minutes for the given local date (after early close)."""
        mins = 0
        for start_t, end_t in self.sessions_for_date(d_local):
            start_dt = dt.datetime.combine(d_local, start_t, self.tz)
            end_dt = dt.datetime.combine(d_local, end_t, self.tz)
            mins += int((end_dt - start_dt).total_seconds() // 60)
        return mins


# ---------- Loader / Registry ----------

def _parse_hours_str(hours_str: str) -> List[Tuple[dt.time, dt.time]]:
    """
    Parse hours like:
      "09:30-16:00"
      "09:00-11:30,12:30-15:00" (split session)
      "" (closed)
    into list[(start_time, end_time)]
    """
    if not hours_str:
        return []
    segments = []
    for seg in hours_str.split(","):
        seg = seg.strip()
        if not seg:
            continue
        try:
            left, right = seg.split("-")
            start = dt.time.fromisoformat(left.strip())
            end = dt.time.fromisoformat(right.strip())
            if end <= start:
                raise ValueError(f"End time {end} must be after start {start}")
            segments.append((start, end))
        except Exception as e:
            raise ValueError(f"Invalid hours segment '{seg}': {e}") from e
    return segments


_WEEKDAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_KEY_TO_WD = {k: i for i, k in enumerate(_WEEKDAY_KEYS)}


def _weekday_sessions_from_yaml(hours_map: Dict[str, str]) -> Dict[int, List[Tuple[dt.time, dt.time]]]:
    out: Dict[int, List[Tuple[dt.time, dt.time]]] = {}
    for key, val in (hours_map or {}).items():
        k = key.strip().lower()
        if k not in _KEY_TO_WD:
            raise ValueError(f"Unknown weekday '{key}' in hours map")
        out[_KEY_TO_WD[k]] = _parse_hours_str(val.strip())
    # ensure missing days default to closed
    for wd in range(7):
        out.setdefault(wd, [])
    return out


def _parse_early_closes(objs: Optional[List[dict]]) -> Dict[dt.date, EarlyClose]:
    out: Dict[dt.date, EarlyClose] = {}
    if not objs:
        return out
    for item in objs:
        try:
            d = dt.date.fromisoformat(str(item.get("date")))
            close_str = item.get("close")
            close = dt.time.fromisoformat(close_str) if close_str else None
            out[d] = EarlyClose(date=d, close=close, note=item.get("note"))
        except Exception as e:
            raise ValueError(f"Invalid early_closes entry: {item!r} ({e})")
    return out


class CalendarRegistry:
    """
    Loads and serves RegionCalendar objects from a directory of YAML files.
    Each file shape:
      region: US
      timezone: America/New_York
      hours: { mon: "09:30-16:00", ... }
      holidays: ["2025-01-01", ...]
      early_closes: [{date: 2025-12-24, close: "13:00", note: "..."}]
    """

    def __init__(self, dir_path: str = "configs/calendars") -> None:
        self.dir = dir_path
        self._cache: Dict[str, Tuple[RegionCalendar, float]] = {}  # region -> (cal, mtime)

    def _file_for_region(self, region: str) -> str:
        return os.path.join(self.dir, f"{region.lower()}.yml")

    def _load_file(self, path: str) -> RegionCalendar:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        region = str(cfg["region"]).strip()
        tzname = str(cfg.get("timezone") or cfg.get("tz") or "").strip()
        if not tzname:
            raise ValueError(f"Missing timezone in {path}")
        tz = ZoneInfo(tzname)

        hours_map = cfg.get("hours") or {}
        weekday_sessions = _weekday_sessions_from_yaml(hours_map)

        holidays = set()
        for h in (cfg.get("holidays") or []):
            try:
                holidays.add(dt.date.fromisoformat(str(h)))
            except Exception as e:
                raise ValueError(f"Invalid holiday date '{h}' in {path}: {e}")

        early_closes = _parse_early_closes(cfg.get("early_closes"))

        return RegionCalendar(
            region=region,
            tz=tz,
            weekday_sessions=weekday_sessions,
            holidays=holidays,
            early_closes=early_closes,
        )

    def get(self, region: str) -> RegionCalendar:
        """Get RegionCalendar for region (US/EU/IN/JP). Hot-reloads on file mtime change."""
        path = self._file_for_region(region)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calendar file not found for region '{region}': {path}")
        mtime = os.path.getmtime(path)
        cached = self._cache.get(region)
        if cached and cached[1] == mtime:
            return cached[0]
        cal = self._load_file(path)
        self._cache[region] = (cal, mtime)
        return cal

    # Convenience pass-throughs

    def is_open(self, region: str, ts: dt.datetime) -> bool:
        return self.get(region).is_open_at(ts)

    def session_bounds(self, region: str, d_local: dt.date) -> Optional[Tuple[dt.datetime, dt.datetime]]:
        return self.get(region).session_bounds(d_local)

    def next_open(self, region: str, ts: dt.datetime) -> Optional[dt.datetime]:
        return self.get(region).next_open_after(ts)

    def previous_close(self, region: str, ts: dt.datetime) -> Optional[dt.datetime]:
        return self.get(region).previous_close_before(ts)

    def trading_minutes_on(self, region: str, d_local: dt.date) -> int:
        return self.get(region).trading_minutes_on(d_local)


# ---------- Module-level helpers (singleton) ----------

@lru_cache(maxsize=1)
def _registry(default_dir: str = "configs/calendars") -> CalendarRegistry:
    return CalendarRegistry(default_dir)


def is_open(region: str, ts: dt.datetime) -> bool:
    """True if market for region is open at timestamp ts (tz-aware or naive UTC)."""
    ts = _ensure_aware(ts)
    return _registry().is_open(region, ts)


def next_open(region: str, ts: dt.datetime) -> Optional[dt.datetime]:
    ts = _ensure_aware(ts)
    return _registry().next_open(region, ts)


def previous_close(region: str, ts: dt.datetime) -> Optional[dt.datetime]:
    ts = _ensure_aware(ts)
    return _registry().previous_close(region, ts)


def session_bounds(region: str, d_local: dt.date) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    return _registry().session_bounds(region, d_local)


def trading_minutes_on(region: str, d_local: dt.date) -> int:
    return _registry().trading_minutes_on(region, d_local)


def set_calendars_dir(path: str) -> None:
    """Override default calendars directory (e.g., in tests)."""
    _registry.cache_clear()  # type: ignore
    _registry(path)  # re-seed cache


# ---------- Utilities ----------

def _ensure_aware(ts: dt.datetime) -> dt.datetime:
    """If naive, assume UTC; else pass-through."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=ZoneInfo("UTC"))
    return ts