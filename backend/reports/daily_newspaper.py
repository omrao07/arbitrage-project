#!/usr/bin/env python3
# tools/daily_newspaper.py
"""
Daily Newspaper Generator
-------------------------
Builds a Markdown "daily newspaper" with:
  • Headlines from RSS feeds
  • Market summary for equities/ETFs/indices (yfinance)
  • Crypto snapshot (yfinance tickers like BTC-USD, ETH-USD)
  • Weather (Open-Meteo; no API key)

Output
  • Markdown file saved to: <out_dir>/daily_<YYYY-MM-DD>.md
  • Optional stdout print

Dependencies (pip):
  pip install feedparser yfinance requests pandas pytz python-dateutil tabulate

Examples
  python tools/daily_newspaper.py \
      --rss "https://feeds.a.dj.com/rss/RSSMarketsMain.xml" \
      --rss "https://www.ft.com/?format=rss" \
      --tickers "^GSPC,NDX,SPY,AAPL,MSFT,TSLA" \
      --crypto "BTC-USD,ETH-USD" \
      --city "Mumbai" \
      --country "India" \
      --out-dir reports \
      --top-n 5 --print

Notes
  • yfinance returns recent OHLC; we compute %Δ vs prior close.
  • Weather uses Open-Meteo geocoding + forecast (metric units).
  • Everything is best-effort; failures are logged and skipped so the paper still prints.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

import requests
import feedparser
import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False

from dateutil import tz
from tabulate import tabulate

# --------- constants ---------

USER_AGENT = "DailyNewspaper/1.0 (+https://example.local)"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# --------- utils ---------

def today_str(tz_name: Optional[str] = None) -> str:
    z = tz.gettz(tz_name) if tz_name else tz.tzlocal()
    return dt.datetime.now(tz=z).date().isoformat()

def hms_now(tz_name: Optional[str] = None) -> str:
    z = tz.gettz(tz_name) if tz_name else tz.tzlocal()
    return dt.datetime.now(tz=z).strftime("%Y-%m-%d %H:%M:%S %Z")

def safe_get(url: str, params: Dict[str, Any] | None = None, timeout: int = 15) -> Optional[requests.Response]:
    try:
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200:
            return r
        return None
    except Exception:
        return None

def pct_change(curr: float, prev: float) -> Optional[float]:
    try:
        if prev == 0:
            return None
        return (curr - prev) / prev * 100.0
    except Exception:
        return None

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

# --------- weather ---------

def geocode_city(city: str, country: Optional[str] = None) -> Optional[Tuple[float, float, str]]:
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    if country:
        params["country"] = country
    r = safe_get(OPEN_METEO_GEOCODE, params=params)
    if not r:
        return None
    try:
        data = r.json()
        if not data.get("results"):
            return None
        res = data["results"][0]
        return float(res["latitude"]), float(res["longitude"]), res.get("timezone", "UTC")
    except Exception:
        return None

def fetch_weather(lat: float, lon: float, tz_name: str) -> Optional[Dict[str, Any]]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "forecast_days": 3,
        "timezone": tz_name,
    }
    r = safe_get(OPEN_METEO_FORECAST, params=params)
    if not r:
        return None
    try:
        return r.json()
    except Exception:
        return None

def render_weather(city: str, country: Optional[str]) -> str:
    g = geocode_city(city, country)
    if not g:
        return f"### Weather — {city}{', ' + country if country else ''}\n\n_(unavailable)_\n"
    lat, lon, tz_name = g
    data = fetch_weather(lat, lon, tz_name)
    if not data:
        return f"### Weather — {city}, {country or ''}\n\n_(unavailable)_\n"

    daily = data.get("daily", {})
    dates = daily.get("time", [])[:3]
    tmax = daily.get("temperature_2m_max", [])[:3]
    tmin = daily.get("temperature_2m_min", [])[:3]
    rain = daily.get("precipitation_sum", [])[:3]
    wind = daily.get("wind_speed_10m_max", [])[:3]

    rows = []
    for i in range(min(3, len(dates))):
        rows.append([dates[i], f"{tmin[i]:.1f}°C", f"{tmax[i]:.1f}°C", f"{rain[i]:.1f} mm", f"{wind[i]:.1f} m/s"])
    table = tabulate(rows, headers=["Date", "Low", "High", "Rain", "Wind"], tablefmt="github")
    return f"""### Weather — {city}{', ' + country if country else ''}

{table}

"""

# --------- markets ---------

def fetch_quotes(tickers: List[str]) -> pd.DataFrame:
    if not _HAS_YF or not tickers:
        return pd.DataFrame(columns=["ticker","price","prev_close","chg","chg_pct"])
    try:
        df = []
        for t in tickers:
            t = t.strip()
            if not t:
                continue
            tk = yf.Ticker(t)
            info = getattr(tk, "fast_info", {}) or {}
            price = info.get("last_price")
            prev = info.get("previous_close")
            if price is None:
                # fallback: history
                hist = tk.history(period="2d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
            chg = price - prev if (price is not None and prev is not None) else None
            chg_pct = pct_change(price, prev) if (price is not None and prev is not None) else None
            df.append({"ticker": t, "price": price, "prev_close": prev, "chg": chg, "chg_pct": chg_pct})
        return pd.DataFrame(df)
    except Exception:
        return pd.DataFrame(columns=["ticker","price","prev_close","chg","chg_pct"])

def render_markets(tickers: List[str], title: str) -> str:
    if not tickers:
        return ""
    df = fetch_quotes(tickers)
    if df.empty:
        return f"### {title}\n\n_(unavailable)_\n"
    rows = []
    for _, r in df.iterrows():
        p = f"{r['price']:.2f}" if pd.notnull(r["price"]) else "—"
        c = f"{r['chg']:+.2f}" if pd.notnull(r["chg"]) else "—"
        cp = f"{r['chg_pct']:+.2f}%" if pd.notnull(r["chg_pct"]) else "—"
        rows.append([r["ticker"], p, c, cp])
    table = tabulate(rows, headers=["Ticker", "Price", "Δ", "Δ%"], tablefmt="github")
    return f"""### {title}

{table}

"""

# --------- news ---------

def fetch_rss_items(url: str, top_n: int) -> List[Dict[str, Any]]:
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:top_n]:
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()
            pub = getattr(e, "published", getattr(e, "updated", "")) or ""
            items.append({"title": title, "link": link, "published": pub})
        return items
    except Exception:
        return []

def render_headlines(rss_urls: List[str], top_n: int) -> str:
    if not rss_urls:
        return ""
    out = ["### Top Headlines"]
    for u in rss_urls:
        items = fetch_rss_items(u, top_n)
        src = u.split("/")[2] if "://" in u else u
        out.append(f"\n**{src}**")
        if not items:
            out.append("\n- _(no items)_")
            continue
        for it in items:
            title = it["title"] or "(untitled)"
            link = it["link"]
            pub = it["published"]
            out.append(f"- {title} — {pub}\n  <{link}>")
    out.append("")  # trailing newline
    return "\n".join(out)

# --------- compose ---------

def compose_paper(
    *,
    title: str,
    tz_name: Optional[str],
    rss_urls: List[str],
    top_n: int,
    tickers: List[str],
    crypto: List[str],
    city: Optional[str],
    country: Optional[str],
) -> str:
    datestamp = today_str(tz_name)
    timestamp = hms_now(tz_name)

    parts = [f"# {title} — {datestamp}\n", f"_Generated at {timestamp}_\n"]

    # Markets
    if tickers:
        parts.append(render_markets(tickers, "Market Snapshot"))
    if crypto:
        parts.append(render_markets(crypto, "Crypto Snapshot"))

    # Weather
    if city:
        parts.append(render_weather(city, country))

    # Headlines
    if rss_urls:
        parts.append(render_headlines(rss_urls, top_n))

    return "\n".join([p for p in parts if p.strip()])

# --------- main ---------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a daily newspaper (Markdown).")
    p.add_argument("--title", default="Daily Brief", help="Paper title header.")
    p.add_argument("--tz", default=None, help="Timezone name (e.g., Asia/Kolkata). Defaults to local.")
    p.add_argument("--rss", action="append", default=[], help="RSS feed URL. Pass multiple --rss flags.")
    p.add_argument("--top-n", type=int, default=5, help="Top N headlines per feed.")
    p.add_argument("--tickers", default="", help="Comma-separated tickers (^GSPC,NDX,SPY,AAPL).")
    p.add_argument("--crypto", default="BTC-USD,ETH-USD", help="Comma-separated crypto tickers.")
    p.add_argument("--city", default=None, help="City for weather (uses Open-Meteo geocoder).")
    p.add_argument("--country", default=None, help="Country filter for geocoding (optional).")
    p.add_argument("--out-dir", default="reports", help="Directory to write the Markdown file.")
    p.add_argument("--filename", default=None, help="Override output file name (e.g., my_brief.md).")
    p.add_argument("--print", dest="do_print", action="store_true", help="Also print to stdout.")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    rss_urls = [u.strip() for u in args.rss if u.strip()]
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    crypto = [t.strip() for t in args.crypto.split(",") if t.strip()]

    md = compose_paper(
        title=args.title,
        tz_name=args.tz,
        rss_urls=rss_urls,
        top_n=clamp(args.top_n, 1, 20),
        tickers=tickers,
        crypto=crypto,
        city=args.city,
        country=args.country,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    if args.filename:
        out_path = os.path.join(args.out_dir, args.filename)
    else:
        out_path = os.path.join(args.out_dir, f"daily_{today_str(args.tz)}.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    if args.do_print:
        print(md)

    print(f"[daily_newspaper] wrote {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())