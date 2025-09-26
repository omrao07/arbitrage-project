"""
dates.py
---------
Date and time utilities for the quant platform.
Handles trading calendars, safe parsing, ranges, and formatting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Optional, Union


# ---------- Parsing & Formatting ----------

def to_datetime(obj: Union[str, date, datetime, pd.Timestamp]) -> pd.Timestamp:
    """Convert string/date/datetime to pandas Timestamp."""
    return pd.to_datetime(obj)


def today_str(fmt: str = "%Y-%m-%d") -> str:
    """Return today's date as string."""
    return datetime.today().strftime(fmt)


def format_date(dt: Union[str, datetime, pd.Timestamp], fmt: str = "%Y-%m-%d") -> str:
    """Format datetime-like into string."""
    return pd.to_datetime(dt).strftime(fmt)


# ---------- Date Ranges ----------

def date_range(start: str, end: str, freq: str = "D") -> pd.DatetimeIndex:
    """Generate range of dates."""
    return pd.date_range(start=start, end=end, freq=freq)


def last_n_days(n: int, end: Optional[str] = None) -> pd.DatetimeIndex:
    """Return last N calendar days up to end (default today)."""
    end = pd.to_datetime(end) if end else pd.Timestamp.today() # type: ignore
    start = end - pd.Timedelta(days=n-1) # type: ignore
    return pd.date_range(start, end, freq="D")


def last_n_years(n: int, end: Optional[str] = None) -> pd.DatetimeIndex:
    """Return last N years of daily dates up to end (default today)."""
    end = pd.to_datetime(end) if end else pd.Timestamp.today() # type: ignore
    start = end - pd.DateOffset(years=n) # type: ignore
    return pd.date_range(start, end, freq="D")


# ---------- Trading Calendar Helpers ----------

def is_weekend(dt: Union[str, datetime, pd.Timestamp]) -> bool:
    """Check if date is a weekend."""
    return pd.to_datetime(dt).weekday() >= 5


def next_business_day(dt: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """Get next business day (Mon–Fri)."""
    d = pd.to_datetime(dt)
    while d.weekday() >= 5:  # Sat/Sun
        d += pd.Timedelta(days=1)
    return d


def prev_business_day(dt: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """Get previous business day (Mon–Fri)."""
    d = pd.to_datetime(dt)
    while d.weekday() >= 5:  # Sat/Sun
        d -= pd.Timedelta(days=1)
    return d


def business_days_between(start: str, end: str) -> pd.DatetimeIndex:
    """List business days between two dates (Mon–Fri)."""
    days = pd.date_range(start, end, freq="D")
    return days[days.weekday < 5]


# ---------- Rolling / Alignment ----------

def align_to_month_end(dates: Union[pd.Series, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Align a series of dates to month-end."""
    return pd.to_datetime(dates) + pd.offsets.MonthEnd(0) # type: ignore


def align_to_quarter_end(dates: Union[pd.Series, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Align a series of dates to quarter-end."""
    return pd.to_datetime(dates) + pd.offsets.QuarterEnd(0) # type: ignore


def align_to_year_end(dates: Union[pd.Series, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Align a series of dates to year-end."""
    return pd.to_datetime(dates) + pd.offsets.YearEnd(0) # type: ignore


# ---------- Example Usage ----------

if __name__ == "__main__":
    print("Today:", today_str())
    print("Next business day:", next_business_day("2025-09-13"))
    print("Prev business day:", prev_business_day("2025-09-13"))
    print("Business days last 10:", business_days_between("2025-09-01", "2025-09-13"))
    print("Align to Q-end:", align_to_quarter_end(["2025-02-11", "2025-05-13"])) # type: ignore