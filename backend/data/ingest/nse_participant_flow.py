"""
NSE participant-wise OI ingester — FII / DII / Client / Pro positioning.

WHY THIS FILE MATTERS MORE THAN THE OTHER INGESTERS
---------------------------------------------------
This is the dataset the plan identifies as the structural edge: NSE publishes
participant-wise open interest daily, for free, and almost no global systematic
fund ingests it. It is public, underexploited, and specific to a market you
understand. Everything else in the repo is a commodity data problem; this one
is the moat.

POINT-IN-TIME CORRECTNESS — THE TRAP THAT MANUFACTURES FAKE ALPHA
------------------------------------------------------------------
The file for trade date T is published AFTER the close of T. A backtest that
reads T's participant flow while trading T's session is using information that
did not exist yet, and it will show beautiful, entirely fictional alpha.

So every row carries two timestamps:

    trade_date  — the session the positioning describes
    avail_ts    — when a strategy could first legitimately act on it,
                  set to the NEXT trading day's pre-open (09:00 IST)

Feature code must join on `avail_ts`, never on `trade_date`. `load_point_in_time()`
enforces this: it will not return a row whose `avail_ts` is in the future
relative to the `as_of` you pass.

Source: https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)

OI_URL = "https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{ddmmyyyy}.csv"
VOL_URL = "https://nsearchives.nseindia.com/content/nsccl/fao_participant_vol_{ddmmyyyy}.csv"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/csv,*/*",
}

DEFAULT_STORE = Path("backend/data/nse/participant_flow")

# NSE publishes after close; a strategy can first act at the next session's open.
PUBLISH_LAG_HOURS = 17  # ~16:00 IST close -> 09:00 IST next morning


class IngestError(RuntimeError):
    """Raised when data could not be fetched or is structurally wrong."""


@dataclass(frozen=True, slots=True)
class ParticipantRow:
    """One participant class's positioning for one trade date."""

    trade_date: dt.date
    avail_ts: dt.datetime
    client_type: str
    future_index_long: float
    future_index_short: float
    future_stock_long: float
    future_stock_short: float
    option_index_call_long: float
    option_index_put_long: float
    option_index_call_short: float
    option_index_put_short: float

    @property
    def net_index_futures(self) -> float:
        """Net directional index-futures position — the headline FII signal."""
        return self.future_index_long - self.future_index_short

    @property
    def index_futures_ls_ratio(self) -> float:
        """Long/short ratio; guarded so a zero short side does not explode."""
        return self.future_index_long / max(self.future_index_short, 1.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trade_date": self.trade_date.isoformat(),
            "avail_ts": self.avail_ts.isoformat(),
            "client_type": self.client_type,
            "future_index_long": self.future_index_long,
            "future_index_short": self.future_index_short,
            "future_stock_long": self.future_stock_long,
            "future_stock_short": self.future_stock_short,
            "option_index_call_long": self.option_index_call_long,
            "option_index_put_long": self.option_index_put_long,
            "option_index_call_short": self.option_index_call_short,
            "option_index_put_short": self.option_index_put_short,
            "net_index_futures": self.net_index_futures,
            "index_futures_ls_ratio": self.index_futures_ls_ratio,
        }


def _to_float(raw: str) -> float:
    """NSE emits '1,234.00', '-', and blanks. A blank is 0 contracts, not NaN."""
    s = (raw or "").strip().replace(",", "").replace('"', "")
    if s in ("", "-"):
        return 0.0
    try:
        return float(s)
    except ValueError as exc:
        raise IngestError(f"non-numeric participant value: {raw!r}") from exc


def _availability(trade_date: dt.date) -> dt.datetime:
    """
    First moment a strategy may legitimately use this row.

    Next calendar day at 09:00 IST, skipping weekends. Exchange holidays are not
    modelled here, which errs toward availability being EARLIER than reality —
    the conservative direction is later, so treat this as a floor and gate on
    the trading calendar upstream if you need exactness.
    """
    nxt = trade_date + dt.timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += dt.timedelta(days=1)
    return dt.datetime.combine(nxt, dt.time(9, 0))


def fetch_day(trade_date: dt.date, timeout: int = 20, retries: int = 2) -> List[ParticipantRow]:
    """
    Download and parse one trading day. Raises rather than returning [] so a
    missing day is visibly missing instead of quietly becoming a gap.
    """
    url = OI_URL.format(ddmmyyyy=trade_date.strftime("%d%m%Y"))
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    raise IngestError(f"HTTP {resp.status} for {trade_date}")
                raw = resp.read().decode("utf-8", errors="replace")
            return _parse(raw, trade_date)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise IngestError(
                    f"no participant file for {trade_date} (holiday or not yet published)"
                ) from exc
            last = exc
        except Exception as exc:  # noqa: BLE001 - network is unreliable; retry
            last = exc
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise IngestError(f"failed to fetch {trade_date} after {retries + 1} attempts: {last}")


def _parse(raw: str, trade_date: dt.date) -> List[ParticipantRow]:
    """
    Parse the NSE layout: a title line, then a header, then one row per
    participant class (Client / DII / FII / Pro / TOTAL).
    """
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.lower().startswith("client type")), None
    )
    if header_idx is None:
        raise IngestError(f"{trade_date}: no 'Client Type' header — layout changed?")

    reader = csv.reader(io.StringIO("\n".join(lines[header_idx:])))
    header = [h.strip().strip('"') for h in next(reader)]
    if len(header) < 9:
        raise IngestError(f"{trade_date}: expected >= 9 columns, got {len(header)}")

    avail = _availability(trade_date)
    rows: List[ParticipantRow] = []
    for parts in reader:
        if not parts or not parts[0].strip():
            continue
        ctype = parts[0].strip().strip('"').upper()
        if ctype.startswith("TOTAL"):
            continue  # aggregate, not a participant class
        vals = [_to_float(p) for p in parts[1:11]] + [0.0] * 10
        rows.append(
            ParticipantRow(
                trade_date=trade_date,
                avail_ts=avail,
                client_type=ctype,
                future_index_long=vals[0],
                future_index_short=vals[1],
                future_stock_long=vals[2],
                future_stock_short=vals[3],
                option_index_call_long=vals[4],
                option_index_put_long=vals[5],
                option_index_call_short=vals[6],
                option_index_put_short=vals[7],
            )
        )
    if not rows:
        raise IngestError(f"{trade_date}: header present but no participant rows")
    return rows


def _trading_days(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def backfill(
    start: dt.date,
    end: dt.date,
    store: Path = DEFAULT_STORE,
    polite_delay: float = 0.4,
) -> Dict[str, Any]:
    """
    Download a date range to CSV, one file per day.

    Missing days are recorded, not silently skipped: a gap you know about is
    fine, a gap you invented over is not.
    """
    store.mkdir(parents=True, exist_ok=True)
    got, missing, failed = [], [], {}

    for day in _trading_days(start, end):
        out = store / f"{day.isoformat()}.csv"
        if out.exists():
            got.append(day.isoformat())
            continue
        try:
            rows = fetch_day(day)
        except IngestError as exc:
            (missing if "no participant file" in str(exc) else failed.setdefault("x", []))
            if "no participant file" in str(exc):
                missing.append(day.isoformat())
            else:
                failed[day.isoformat()] = str(exc)[:120]
            continue

        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].as_dict()))
            w.writeheader()
            for r in rows:
                w.writerow(r.as_dict())
        got.append(day.isoformat())
        time.sleep(polite_delay)  # be a good citizen of a free public endpoint

    return {
        "downloaded": len(got),
        "missing_holidays": len(missing),
        "failed": failed,
        "store": str(store),
    }


def load_point_in_time(
    as_of: dt.datetime,
    store: Path = DEFAULT_STORE,
    client_type: str = "FII",
) -> List[Dict[str, Any]]:
    """
    Load every row a strategy could legitimately know at `as_of`.

    This is the lookahead guard: rows whose `avail_ts` is later than `as_of` are
    excluded, so a same-day-flow backtest is impossible by construction rather
    than by discipline.
    """
    if not store.exists():
        raise IngestError(f"no participant store at {store} — run backfill() first")

    out: List[Dict[str, Any]] = []
    for f in sorted(store.glob("*.csv")):
        with f.open() as fh:
            for row in csv.DictReader(fh):
                if row.get("client_type", "").upper() != client_type.upper():
                    continue
                if dt.datetime.fromisoformat(row["avail_ts"]) > as_of:
                    continue  # not knowable yet
                out.append(row)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Backfill NSE participant-wise OI.")
    ap.add_argument("--days", type=int, default=90, help="calendar days back from today")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    end = dt.date.today()
    start = end - dt.timedelta(days=args.days)
    result = backfill(start, end, args.store)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
