"""
Index price history ingester — Nifty 50, Bank Nifty, India VIX.

Supplies the one input the whole gauntlet is blocked on: a real, long price
series. `GAUNTLET_PRICES_FILE` points at the CSV this writes.

Source is the public Yahoo chart endpoint, which serves daily OHLCV for Indian
indices without a key. That is deliberate — the plan's cost model shows a
₹10,000/month vendor feed is a 24% annual hurdle on a ₹5 lakh account, so every
input that can be free should be free until something survives validation.

WHAT THIS IS NOT
----------------
Daily index closes. Not an option chain, not intraday, not bid/ask. Options
strategies still cannot be honestly backtested from this — that needs a real
point-in-time chain with quotes, which is a paid dataset. This unblocks the
price-and-flow strategies, which is where the plan says the durable edge is
anyway.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json,*/*",
}

# Yahoo tickers for the instruments the India strategies reference.
SYMBOLS: Dict[str, str] = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
    "SENSEX": "^BSESN",
    "USDINR": "INR=X",
}

DEFAULT_STORE = Path("backend/data/nse/prices")


class PriceIngestError(RuntimeError):
    """Raised when a price series could not be fetched or is unusable."""


def fetch_series(
    symbol: str,
    years: int = 5,
    interval: str = "1d",
    timeout: int = 25,
    retries: int = 2,
) -> List[Dict[str, Any]]:
    """
    Download daily OHLCV. Rows with a null close are DROPPED, never filled —
    forward-filling a price is the synthetic-fabrication bug in a nicer costume.
    """
    ticker = SYMBOLS.get(symbol.upper(), symbol)
    end = int(time.time())
    start = end - int(years * 365.25 * 86400)
    qs = urllib.parse.urlencode(
        {"period1": start, "period2": end, "interval": interval, "events": "div,split"}
    )
    url = f"{CHART_URL.format(symbol=urllib.parse.quote(ticker))}?{qs}"

    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return _parse_chart(payload, symbol)
        except Exception as exc:  # noqa: BLE001 - network is unreliable; retry
            last = exc
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    raise PriceIngestError(f"{symbol}: fetch failed after {retries + 1} attempts: {last}")


def _parse_chart(payload: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
    chart = (payload or {}).get("chart") or {}
    if chart.get("error"):
        raise PriceIngestError(f"{symbol}: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise PriceIngestError(f"{symbol}: empty chart result")

    res = results[0]
    stamps = res.get("timestamp") or []
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    if not stamps or not closes:
        raise PriceIngestError(f"{symbol}: no timestamps/closes in response")

    opens, highs, lows, vols = (
        quote.get("open") or [],
        quote.get("high") or [],
        quote.get("low") or [],
        quote.get("volume") or [],
    )

    def at(seq, i, fallback):
        v = seq[i] if i < len(seq) else None
        return float(v) if v is not None else fallback

    rows: List[Dict[str, Any]] = []
    dropped = 0
    for i, ts in enumerate(stamps):
        c = closes[i] if i < len(closes) else None
        if c is None:
            dropped += 1  # a gap we know about; never invented over
            continue
        c = float(c)
        if c <= 0:
            dropped += 1
            continue
        d = dt.datetime.utcfromtimestamp(int(ts)).date()
        rows.append(
            {
                "date": d.isoformat(),
                "open": at(opens, i, c),
                "high": at(highs, i, c),
                "low": at(lows, i, c),
                "close": c,
                "volume": at(vols, i, 0.0),
                "ticker": symbol.upper(),
            }
        )

    if dropped:
        log.warning("%s: dropped %d rows with no usable close", symbol, dropped)
    if len(rows) < 100:
        raise PriceIngestError(
            f"{symbol}: only {len(rows)} usable rows — too short to validate anything"
        )
    return rows


def save(rows: List[Dict[str, Any]], symbol: str, store: Path = DEFAULT_STORE) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    out = store / f"{symbol.upper()}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out


def ingest_all(
    symbols: Optional[List[str]] = None,
    years: int = 5,
    store: Path = DEFAULT_STORE,
) -> Dict[str, Any]:
    """Fetch every configured index. One failure does not abort the rest."""
    names = symbols or list(SYMBOLS)
    written: Dict[str, str] = {}
    failed: Dict[str, str] = {}
    for name in names:
        try:
            rows = fetch_series(name, years=years)
            path = save(rows, name, store)
            written[name] = f"{len(rows)} rows -> {path}"
            log.info("%s: %d rows", name, len(rows))
        except PriceIngestError as exc:
            failed[name] = str(exc)[:140]
            log.error("%s: %s", name, exc)
        time.sleep(0.5)
    return {"written": written, "failed": failed, "store": str(store)}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Ingest index price history.")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = ingest_all(args.symbols, args.years, args.store)
    for k, v in result["written"].items():
        print(f"  OK   {k:10s} {v}")
    for k, v in result["failed"].items():
        print(f"  FAIL {k:10s} {v}")
    return 0 if result["written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
