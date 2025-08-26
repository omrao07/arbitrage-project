#!/usr/bin/env python3
"""
MiFID II / MiFIR RTS 22 Transaction Reporter
--------------------------------------------

Purpose
=======
Convert internal trade/fill records into a MiFID II (RTS 22-style) transaction reporting
CSV suitable for uploading to an ARM or for internal reconciliation.

This module is **opinionated but pluggable**:
- Accepts input as CSV or JSON (array of trades)
- Optionally consumes live fills from a Redis Stream (e.g., STREAM_FILLS) and writes rolling CSVs
- Validates key identifiers (ISIN, LEI, MIC, currency, timestamp)
- Normalizes timestamps to UTC (ISO 8601 with "Z")
- Generates a UTI if not provided (hash of firm LEI + trade id + date)
- Emits a compact RTS22-style CSV with sane defaults

⚠️ Compliance note
------------------
RTS 22 has 65 fields with jurisdiction- and venue-specific nuances. This script outputs a
useful, commonly required subset that many ARMs accept as CSV inputs for equities/ETDs.
You MUST review with your compliance team, your ARM specs, and your NCA technical standards
before using in production. Extend the `RTS22_FIELDS` and `map_trade_to_rts22` as needed.

Quick start
===========
1) Validate only (no output):
   python mifid_reporter.py --input fills.csv --input-format fills_csv --validate-only \
       --firm-lei 5493001KJTIIGC8Y1R12 --branch-country GB

2) Convert to RTS22 CSV:
   python mifid_reporter.py --input fills.csv --input-format fills_csv \
       --output mifid_report_2025-08-25.csv --firm-lei 5493001KJTIIGC8Y1R12 \
       --branch-country GB --default-currency EUR

3) Generate an input template you can fill in:
   python mifid_reporter.py --emit-template fills_template.csv --template-format fills_csv

4) Stream from Redis (optional):
   python mifid_reporter.py --from-redis --redis-stream STREAM_FILLS --firm-lei 5493... \
       --output mifid_stream_report.csv

Input schemas
=============
A) fills_csv (wide, human-friendly) — headers expected:
   trade_id, exec_time, symbol, isin, price, qty, currency, side, buyer_lei, seller_lei,
   mic, short_sale, algo, client_id, underlying_isin, maturity_date, delivery_type,
   notional_ccy1, notional_ccy2, price_notation

B) json (machine-friendly): list of objects with keys similar to above (snake_case).

What we output (default columns)
================================
See `RTS22_FIELDS` below. These include: transaction_reference_number, trading_date,
execution_timestamp_utc, isin, price, quantity, price_currency, buyer_lei, seller_lei,
short_sale_indicator, algorithmic_indicator, execution_venue_mic, client_id,
investment_firm_lei, branch_country, trading_capacity, liquidity_provider,
commodity_derivative_indicator, emission_allowance_indicator, derivative_underlying_isin,
derivative_maturity_date, derivative_delivery_type, notional_currency_1,
notional_currency_2, price_notation, publication_mode, uti

Extend/override mapping rules in `map_trade_to_rts22()` as required.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional
    redis = None  # optional dependency

# ---------------------------
# Configuration & Constants
# ---------------------------

RTS22_FIELDS: List[str] = [
    # A compact, common subset of the full RTS22 fields
    "transaction_reference_number",
    "trading_date",
    "execution_timestamp_utc",
    "isin",
    "price",
    "quantity",
    "price_currency",
    "buyer_lei",
    "seller_lei",
    "short_sale_indicator",
    "algorithmic_indicator",
    "execution_venue_mic",
    "client_id",
    "investment_firm_lei",
    "branch_country",
    "trading_capacity",  # DEAL, MTCH, AOTC
    "liquidity_provider",  # Y/N
    "commodity_derivative_indicator",  # Y/N
    "emission_allowance_indicator",  # Y/N
    "derivative_underlying_isin",
    "derivative_maturity_date",
    "derivative_delivery_type",  # CASH/PHYS
    "notional_currency_1",
    "notional_currency_2",
    "price_notation",
    "publication_mode",  # APA/ARM/Internal, set for your workflow
    "uti",
]

FALLBACK_TRADING_CAPACITY = os.getenv("MIFID_TRADING_CAPACITY", "DEAL")
FALLBACK_LIQUIDITY_PROVIDER = os.getenv("MIFID_LIQUIDITY_PROVIDER", "N")
FALLBACK_PUBLICATION_MODE = os.getenv("MIFID_PUBLICATION_MODE", "ARM")
FALLBACK_BRANCH_COUNTRY = os.getenv("MIFID_BRANCH_COUNTRY", "GB")
FALLBACK_PRICE_CCY = os.getenv("MIFID_DEFAULT_CURRENCY", "EUR")

# Regexes
_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_LEI_RE = re.compile(r"^[A-Z0-9]{20}$")
_MIC_RE = re.compile(r"^[A-Z0-9]{4}$")
_CCY_RE = re.compile(r"^[A-Z]{3}$")


# ---------------------------
# Utilities
# ---------------------------

def to_utc_z(ts: str) -> str:
    """Parse ISO-like timestamp string to UTC ISO8601 with trailing 'Z'.

    Accepts inputs like:
      - "2025-08-25T10:22:33.123456+05:30"
      - "2025-08-25T04:52:33Z"
      - "2025-08-25 04:52:33" (treated as UTC if naive)
    """
    ts = ts.strip()
    # Replace space with T for fromisoformat compatibility
    ts_norm = ts.replace(" ", "T")
    try:
        dt_obj = dt.datetime.fromisoformat(ts_norm)
    except Exception:
        # Last-ditch: try without microseconds
        try:
            dt_obj = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            raise ValueError(f"Unparseable timestamp: {ts}") from e

    if dt_obj.tzinfo is None:
        # Assume UTC if naive
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    dt_utc = dt_obj.astimezone(dt.timezone.utc)
    # Normalize to milliseconds for CSV friendliness
    return dt_utc.replace(microsecond=(dt_utc.microsecond // 1000) * 1000).isoformat().replace("+00:00", "Z")


def trading_date_from_ts(ts_utc_z: str) -> str:
    """Extract YYYY-MM-DD trading date from a UTC ISO8601 timestamp string."""
    # ts_utc_z like '2025-08-25T10:22:33.123Z'
    date_part = ts_utc_z.split("T", 1)[0]
    return date_part


def luhn_check_isin(isin: str) -> bool:
    """Validate ISIN via Luhn algorithm (ISO 6166)."""
    isin = isin.strip().upper()
    if not _ISIN_RE.match(isin):
        return False
    # Expand letters to numbers (A=10 ... Z=35)
    digits = ""
    for ch in isin[:-1]:
        if ch.isdigit():
            digits += ch
        else:
            digits += str(ord(ch) - 55)  # 'A'->10
    # Double every second digit from right
    total = 0
    reverse = digits[::-1]
    for i, d in enumerate(reverse):
        n = int(d)
        if i % 2 == 0:
            n *= 2
        total += n // 10 + n % 10
    check = (10 - (total % 10)) % 10
    return check == int(isin[-1])


def is_valid_lei(lei: str) -> bool:
    return bool(_LEI_RE.match(lei or ""))


def is_valid_mic(mic: str) -> bool:
    return bool(_MIC_RE.match((mic or "").upper()))


def is_valid_ccy(ccy: str) -> bool:
    return bool(_CCY_RE.match((ccy or "").upper()))


# ---------------------------
# Data classes
# ---------------------------

@dataclass
class Defaults:
    investment_firm_lei: str
    branch_country: str = FALLBACK_BRANCH_COUNTRY
    trading_capacity: str = FALLBACK_TRADING_CAPACITY
    liquidity_provider: str = FALLBACK_LIQUIDITY_PROVIDER
    publication_mode: str = FALLBACK_PUBLICATION_MODE
    price_currency: str = FALLBACK_PRICE_CCY


# ---------------------------
# Mapping & Validation
# ---------------------------

def validate_trade(trade: Dict[str, Any], d: Defaults) -> List[str]:
    """Return a list of validation error messages for a single trade."""
    errs: List[str] = []

    # Required minimal fields
    if not trade.get("exec_time"):
        errs.append("Missing exec_time")
    else:
        try:
            to_utc_z(str(trade["exec_time"]))
        except Exception as e:
            errs.append(f"Bad exec_time: {e}")

    isin = (trade.get("isin") or "").upper()
    if isin and not luhn_check_isin(isin):
        errs.append("Invalid ISIN (Luhn)")

    mic = (trade.get("mic") or "").upper()
    if mic and not is_valid_mic(mic):
        errs.append("Invalid MIC")

    ccy = (trade.get("currency") or d.price_currency).upper()
    if not is_valid_ccy(ccy):
        errs.append("Invalid currency (price_currency)")

    for key in ("price", "qty"):
        try:
            val = float(trade.get(key)) # type: ignore
            if val <= 0:
                errs.append(f"{key} must be > 0")
        except Exception:
            errs.append(f"{key} missing or not a number")

    for key in ("buyer_lei", "seller_lei"):
        lei = trade.get(key)
        if lei and not is_valid_lei(lei):
            errs.append(f"{key} not a valid LEI")

    if not is_valid_lei(d.investment_firm_lei):
        errs.append("Defaults.investment_firm_lei (firm LEI) is invalid")

    return errs


def normalize_side(side: Optional[str]) -> str:
    s = (side or "").strip().upper()
    if s in {"BUY", "B", "BOT"}:
        return "BUY"
    if s in {"SELL", "S", "SLD"}:
        return "SELL"
    return ""


def derive_uti(firm_lei: str, trade_id: str, exec_date: str) -> str:
    base = f"{firm_lei}:{trade_id}:{exec_date}"
    h = hashlib.sha256(base.encode()).hexdigest()[:20].upper()
    return f"{firm_lei}-{exec_date.replace('-', '')}-{h}"


def map_trade_to_rts22(trade: Dict[str, Any], d: Defaults) -> Dict[str, Any]:
    """Map one internal trade dict to RTS22 CSV row dict.

    Unknown/NA fields are filled with '' (empty) unless a default is specified.
    Extend this function for your firm/ARM specifics.
    """
    ts_utc = to_utc_z(str(trade.get("exec_time")))
    tdate = trading_date_from_ts(ts_utc)

    # Reference
    trn = str(trade.get("trade_id") or f"{tdate}-{int(time.time()*1000)}")

    isin = (trade.get("isin") or "").upper()
    mic = (trade.get("mic") or "").upper()

    # Short sale indicator conventions vary by ARM. We pass through if looks sane.
    short_sale = str(trade.get("short_sale") or "0")
    algo_ind = str(trade.get("algo") or trade.get("algorithmic_indicator") or "N").upper()
    if algo_ind in {"TRUE", "T", "YES", "Y", "1"}:
        algo_ind = "Y"
    elif algo_ind in {"FALSE", "F", "NO", "N", "0"}:
        algo_ind = "N"

    row = {
        "transaction_reference_number": trn,
        "trading_date": tdate,
        "execution_timestamp_utc": ts_utc,
        "isin": isin,
        "price": f"{float(trade.get('price')):.10g}", # type: ignore
        "quantity": f"{float(trade.get('qty')):.10g}", # type: ignore
        "price_currency": (trade.get("currency") or d.price_currency).upper(),
        "buyer_lei": (trade.get("buyer_lei") or ""),
        "seller_lei": (trade.get("seller_lei") or ""),
        "short_sale_indicator": short_sale,
        "algorithmic_indicator": algo_ind,
        "execution_venue_mic": mic,
        "client_id": str(trade.get("client_id") or ""),
        "investment_firm_lei": d.investment_firm_lei,
        "branch_country": (trade.get("branch_country") or d.branch_country).upper(),
        "trading_capacity": (trade.get("trading_capacity") or d.trading_capacity).upper(),
        "liquidity_provider": (trade.get("liquidity_provider") or d.liquidity_provider).upper(),
        "commodity_derivative_indicator": str(trade.get("commodity_derivative_indicator") or "N").upper(),
        "emission_allowance_indicator": str(trade.get("emission_allowance_indicator") or "N").upper(),
        "derivative_underlying_isin": (trade.get("underlying_isin") or "").upper(),
        "derivative_maturity_date": (trade.get("maturity_date") or ""),  # YYYY-MM-DD
        "derivative_delivery_type": (trade.get("delivery_type") or ""),   # CASH/PHYS
        "notional_currency_1": (trade.get("notional_ccy1") or ""),
        "notional_currency_2": (trade.get("notional_ccy2") or ""),
        "price_notation": (trade.get("price_notation") or ""),           # e.g., CUR, PCT, YLD
        "publication_mode": (trade.get("publication_mode") or d.publication_mode),
        "uti": (trade.get("uti")
                 or derive_uti(d.investment_firm_lei, trn, tdate)),
    }

    # Optionally enrich with side if your ARM expects it (non-standard in RTS22 CSVs)
    side = normalize_side(trade.get("side"))
    if side:
        row["side"] = side  # non-standard, kept as extra column if you add to RTS22_FIELDS

    return row


# ---------------------------
# I/O helpers
# ---------------------------

def read_fills_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def read_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON input must be an array of trade objects")
    return [dict(x) for x in data]


def write_csv(rows: List[Dict[str, Any]], path: str, headers: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def emit_template(path: str, template_format: str) -> None:
    if template_format == "fills_csv":
        headers = [
            "trade_id","exec_time","symbol","isin","price","qty","currency","side",
            "buyer_lei","seller_lei","mic","short_sale","algo","client_id",
            "underlying_isin","maturity_date","delivery_type","notional_ccy1","notional_ccy2",
            "price_notation","publication_mode","uti"
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Template written to {path}")
    else:
        raise ValueError("Unsupported template format. Use fills_csv")


# ---------------------------
# Redis stream consumer (optional)
# ---------------------------

def consume_redis_stream(
    stream: str,
    start_id: str = "$",
    host: str = os.getenv("REDIS_HOST", "localhost"),
    port: int = int(os.getenv("REDIS_PORT", "6379")),
    block_ms: int = 5000,
) -> Iterable[Dict[str, Any]]:
    if redis is None:
        raise RuntimeError("redis package not installed. pip install redis")
    r = redis.Redis(host=host, port=port, decode_responses=True)
    last_id = start_id
    while True:
        resp = r.xread({stream: last_id}, block=block_ms, count=100)
        if not resp:
            continue
        # resp: [(stream_name, [(id, {field:value, ...}), ...])]
        _, entries = resp[0] # type: ignore
        for msg_id, fields in entries:
            last_id = msg_id
            # Expect JSON payload in field 'data' or key-value pairs resembling a trade
            if "data" in fields:
                try:
                    trade = json.loads(fields["data"])  # your engine publishes JSON
                except Exception:
                    trade = fields
            else:
                trade = fields
            yield trade


# ---------------------------
# Main workflow
# ---------------------------

def process_trades(
    trades: List[Dict[str, Any]],
    defaults: Defaults,
    validate_only: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, List[str]]]]:
    rows: List[Dict[str, Any]] = []
    errors: List[Tuple[int, List[str]]] = []
    for i, t in enumerate(trades):
        errs = validate_trade(t, defaults)
        if errs:
            errors.append((i, errs))
            continue
        if not validate_only:
            rows.append(map_trade_to_rts22(t, defaults))
    return rows, errors


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiFID II (RTS 22) transaction reporter")
    g_in = p.add_argument_group("Input")
    g_in.add_argument("--input", help="Input file (CSV or JSON)")
    g_in.add_argument("--input-format", choices=["fills_csv", "json"], default="fills_csv")
    g_in.add_argument("--emit-template", help="Write an input template and exit")
    g_in.add_argument("--template-format", choices=["fills_csv"], default="fills_csv")

    g_out = p.add_argument_group("Output")
    g_out.add_argument("--output", help="Output CSV path (RTS22 subset)")
    g_out.add_argument("--validate-only", action="store_true", help="Validate inputs without writing CSV")

    g_def = p.add_argument_group("Defaults")
    g_def.add_argument("--firm-lei", required=False, help="Investment firm LEI (also from env FIRM_LEI)")
    g_def.add_argument("--branch-country", default=FALLBACK_BRANCH_COUNTRY)
    g_def.add_argument("--trading-capacity", default=FALLBACK_TRADING_CAPACITY)
    g_def.add_argument("--liquidity-provider", default=FALLBACK_LIQUIDITY_PROVIDER)
    g_def.add_argument("--publication-mode", default=FALLBACK_PUBLICATION_MODE)
    g_def.add_argument("--default-currency", default=FALLBACK_PRICE_CCY)

    g_r = p.add_argument_group("Redis (optional)")
    g_r.add_argument("--from-redis", action="store_true", help="Consume live fills from Redis stream")
    g_r.add_argument("--redis-stream", default=os.getenv("STREAM_FILLS", "STREAM_FILLS"))
    g_r.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    g_r.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.emit_template:
        emit_template(args.emit_template, args.template_format)
        return 0

    firm_lei = args.firm_lei or os.getenv("FIRM_LEI") or ""
    if not firm_lei:
        print("ERROR: --firm-lei or env FIRM_LEI is required", file=sys.stderr)
        return 2

    defaults = Defaults(
        investment_firm_lei=firm_lei,
        branch_country=args.branch_country,
        trading_capacity=args.trading_capacity,
        liquidity_provider=args.liquidity_provider,
        publication_mode=args.publication_mode,
        price_currency=args.default_currency,
    )

    headers = list(RTS22_FIELDS)  # copy

    all_rows: List[Dict[str, Any]] = []
    all_errors: List[Tuple[int, List[str]]] = []

    if args-redis: # type: ignore
        # Stream mode: write rows incrementally to --output (required) and print errors to stderr
        if not args.output:
            print("ERROR: --output is required in --from-redis mode", file=sys.stderr)
            return 2
        try:
            for trade in consume_redis_stream(args.redis_stream, host=args.redis_host, port=args.redis_port):
                errs = validate_trade(trade, defaults)
                if errs:
                    print(f"VALIDATION ERROR for trade {trade.get('trade_id')}: {errs}", file=sys.stderr)
                    continue
                row = map_trade_to_rts22(trade, defaults)
                # Append to CSV file incrementally
                file_exists = os.path.exists(args.output)
                with open(args.output, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
        except KeyboardInterrupt:
            print("Stopped by user.")
        return 0

    # Batch mode: read from file
    if not args.input:
        print("ERROR: --input required (or use --from-redis)", file=sys.stderr)
        return 2

    if args.input_format == "fills_csv":
        trades = read_fills_csv(args.input)
    elif args.input_format == "json":
        trades = read_json_array(args.input)
    else:
        print("Unsupported --input-format", file=sys.stderr)
        return 2

    rows, errors = process_trades(trades, defaults, validate_only=args.validate_only)
    all_rows.extend(rows)
    all_errors.extend(errors)

    # Report validation issues
    if all_errors:
        print("Validation errors (index: [issues]):", file=sys.stderr)
        for idx, errs in all_errors:
            print(f"  {idx}: {errs}", file=sys.stderr)

    if args.validate_only:
        print(f"Checked {len(trades)} trades; {len(all_errors)} with errors; {len(all_rows)} ready.")
        return 0 if not all_errors else 1

    if not args.output:
        print("ERROR: --output path required to write CSV", file=sys.stderr)
        return 2

    write_csv(all_rows, args.output, headers)
    print(f"Wrote {len(all_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
