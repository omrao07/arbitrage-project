# observability/logs/log_parser.py
"""
Structured log parser & normalizer.

Usage
-----
from observability.logs.log_parser import parse_file, tail, Normalizer

norm = Normalizer(service="news-intel", env="dev")
for rec in parse_file("app.log", normalizer=norm):
    ...  # dict records ready for exporter.write(rec)

# or tail with rotation
for rec in tail("app.log", normalizer=norm):
    ...

Design
------
- Prefer JSON lines; fallback to regex parsers for common text formats.
- Normalize timestamps to ISO-8601 UTC; levels to DEBUG/INFO/WARNING/ERROR/CRITICAL.
- Extract trace ids from fields or text (traceparent, x-datadog-*, otel.*).
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Optional

# -------------------------- timestamp parsing --------------------------

_ISO_RE = re.compile(
    r"(?P<iso>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+\-]\d{2}:\d{2}))"
)
_SYSLOG_RE = re.compile(
    r"(?P<sys>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"  # e.g., "Jan  2 15:04:05"
)
_EPOCH_RE = re.compile(r"\b(?P<epoch>\d{10}(?:\.\d{3})?)\b")

def _to_iso_utc(ts: str) -> Optional[str]:
    """Best-effort convert various timestamp strings to ISO8601 UTC."""
    import datetime as dt
    s = ts.strip()
    try:
        if s.isdigit() or re.match(r"^\d{10}\.\d{3}$", s):
            secs = float(s)
            return dt.datetime.utcfromtimestamp(secs).replace(tzinfo=dt.timezone.utc).isoformat()
        # already RFC3339/ISO
        if s.endswith("Z") or re.search(r"[+\-]\d{2}:\d{2}$", s):
            t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            return t.astimezone(dt.timezone.utc).isoformat()
        # try fractional/no tz
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                t = dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc)
                return t.isoformat()
            except Exception:
                pass
        # syslog-like (assume current year, UTC)
        m = _SYSLOG_RE.search(s)
        if m:
            year = dt.datetime.utcnow().year
            t = dt.datetime.strptime(f"{year} {m.group('sys')}", "%Y %b %d %H:%M:%S")
            return t.replace(tzinfo=dt.timezone.utc).isoformat()
    except Exception:
        return None
    return None

def _extract_any_timestamp(line: str) -> Optional[str]:
    m = _ISO_RE.search(line)
    if m:
        return _to_iso_utc(m.group("iso"))
    m = _EPOCH_RE.search(line)
    if m:
        return _to_iso_utc(m.group("epoch"))
    m = _SYSLOG_RE.search(line)
    if m:
        return _to_iso_utc(m.group("sys"))
    return None

# -------------------------- level normalization --------------------------

_LEVEL_MAP = {
    "debug": "DEBUG",
    "info": "INFO",
    "warn": "WARNING",
    "warning": "WARNING",
    "error": "ERROR",
    "err": "ERROR",
    "critical": "CRITICAL",
    "fatal": "CRITICAL",
}

def _norm_level(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    return _LEVEL_MAP.get(val.strip().lower()) or val.upper()

# -------------------------- trace extraction --------------------------

# W3C traceparent: 00-<trace_id>-<span_id>-<flags>
_TRACEPARENT_RE = re.compile(r"\b00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})\b", re.I)
# Datadog ids
_DD_TRACE_RE = re.compile(r"\bdd\.?trace_id[=:]\s*([0-9a-fA-F]+)\b")
_DD_SPAN_RE = re.compile(r"\bdd\.?span_id[=:]\s*([0-9a-fA-F]+)\b")
# Generic otel keys inside JSON
_OTEL_KEYS = ("trace_id", "span_id", "traceid", "spanid")

def _add_trace_fields(rec: Dict) -> None:
    # If already present (JSON)
    for k in _OTEL_KEYS:
        if k in rec:
            if "trace_id" not in rec and k != "trace_id":
                rec["trace_id"] = rec[k]
            if "span_id" not in rec and ("span" in k):
                rec["span_id"] = rec[k]
    msg = str(rec.get("msg", ""))

    if "traceparent" in rec and isinstance(rec["traceparent"], str):
        tp = rec["traceparent"]
        m = _TRACEPARENT_RE.search(tp)
        if m:
            rec.setdefault("trace_id", m.group(1))
            rec.setdefault("span_id", m.group(2))
            return

    m = _TRACEPARENT_RE.search(msg)
    if m:
        rec.setdefault("trace_id", m.group(1))
        rec.setdefault("span_id", m.group(2))

    if "dd.trace_id" in rec:
        rec.setdefault("trace_id", str(rec["dd.trace_id"]))
    if "dd.span_id" in rec:
        rec.setdefault("span_id", str(rec["dd.span_id"]))

    if "trace_id" not in rec:
        m = _DD_TRACE_RE.search(msg)
        if m:
            rec["trace_id"] = m.group(1)
    if "span_id" not in rec:
        m = _DD_SPAN_RE.search(msg)
        if m:
            rec["span_id"] = m.group(1)

# -------------------------- format-specific regexes --------------------------

# nginx/access: '127.0.0.1 - - [16/Jan/2025:15:20:10 +0000] "GET / HTTP/1.1" 200 612 "-" "UA"'
_NGINX_RE = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] '
    r'"(?P<method>[A-Z]+) (?P<path>[^"]+) HTTP/\d\.\d" '
    r'(?P<status>\d{3}) (?P<size>\d+|-) "([^"]*)" "([^"]*)"'
)

# gunicorn/uvicorn: 'INFO:     127.0.0.1:54062 - "GET /health" 200 OK'
_UVI_RE = re.compile(
    r'(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL):\s+.+?"(?P<method>[A-Z]+) (?P<path>[^"]+)" (?P<status>\d{3})'
)

# -------------------------- normalizer --------------------------

@dataclass
class Normalizer:
    service: Optional[str] = None
    env: Optional[str] = None
    host: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def apply(self, rec: Dict) -> Dict:
        # timestamp
        ts = rec.get("ts") or rec.get("timestamp")
        if isinstance(ts, (int, float)):
            # seconds since epoch
            rec["ts"] = _to_iso_utc(str(ts)) or rec.get("ts")
        elif isinstance(ts, str):
            iso = _to_iso_utc(ts)
            if iso:
                rec["ts"] = iso
        if "ts" not in rec:
            # try to extract from message if missing
            iso = _extract_any_timestamp(rec.get("msg", ""))
            if iso:
                rec["ts"] = iso
        # level
        if "level" in rec:
            rec["level"] = _norm_level(str(rec["level"])) or rec["level"]
        # add context
        if self.service: rec.setdefault("service", self.service)
        if self.env: rec.setdefault("env", self.env)
        if self.host: rec.setdefault("host", self.host)
        if self.extra: 
            for k,v in self.extra.items():
                rec.setdefault(k, v)
        # traces
        _add_trace_fields(rec)
        return rec

# -------------------------- parsers --------------------------

def parse_line(line: str, *, normalizer: Optional[Normalizer] = None) -> Optional[Dict]:
    """
    Parse a single log line into a structured dict.
    Returns None if the line is empty or cannot be parsed (and no fallback).
    """
    line = line.rstrip("\r\n")
    if not line:
        return None

    # 1) JSON first
    if line.lstrip().startswith("{") and line.rstrip().endswith("}"):
        try:
            rec = json.loads(line)
            rec.setdefault("msg", rec.get("message", ""))  # harmonize
            if normalizer:
                rec = normalizer.apply(rec)
            return rec
        except Exception:
            pass

    # 2) Nginx access log
    m = _NGINX_RE.match(line)
    if m:
        rec = {
            "logger": "nginx",
            "level": "INFO",
            "ip": m.group("ip"),
            "method": m.group("method"),
            "path": m.group("path").split()[0],
            "status": int(m.group("status")),
            "bytes": None if m.group("size") == "-" else int(m.group("size")),
            "msg": line,
        }
        # ts like 16/Jan/2025:15:20:10 +0000
        ts = m.group("ts")
        rec["ts"] = _to_iso_utc(_nginx_ts_to_iso(ts)) or _extract_any_timestamp(line)
        if normalizer:
            rec = normalizer.apply(rec)
        return rec

    # 3) Uvicorn/Gunicorn summary
    m = _UVI_RE.search(line)
    if m:
        rec = {
            "logger": "uvicorn",
            "level": m.group("level"),
            "method": m.group("method"),
            "path": m.group("path"),
            "status": int(m.group("status")),
            "msg": line,
        }
        rec["ts"] = _extract_any_timestamp(line)
        if normalizer:
            rec = normalizer.apply(rec)
        return rec

    # 4) plain text fallback
    rec = {
        "level": "INFO",
        "msg": line,
        "ts": _extract_any_timestamp(line) or _to_iso_utc(str(time.time())),
    }
    if normalizer:
        rec = normalizer.apply(rec)
    return rec

def parse_stream(stream: Iterable[str], *, normalizer: Optional[Normalizer] = None) -> Iterator[Dict]:
    for line in stream:
        rec = parse_line(line, normalizer=normalizer)
        if rec:
            yield rec

def parse_file(path: str, *, normalizer: Optional[Normalizer] = None, encoding: str = "utf-8") -> Iterator[Dict]:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        for rec in parse_stream(f, normalizer=normalizer):
            yield rec

# -------------------------- tail -F --------------------------

def tail(path: str, *, normalizer: Optional[Normalizer] = None, poll_s: float = 0.5, encoding: str = "utf-8") -> Iterator[Dict]:
    """
    Follow a file like `tail -F` (handles rotation/truncation).
    """
    fh = _open_follow(path, encoding)
    try:
        while True:
            where = fh.tell()
            line = fh.readline()
            if line:
                rec = parse_line(line, normalizer=normalizer)
                if rec:
                    yield rec
            else:
                time.sleep(poll_s)
                # handle rotation/truncation
                if not os.path.exists(path):
                    continue
                if os.stat(path).st_size < where:
                    fh.close()
                    fh = _open_follow(path, encoding)
    finally:
        try:
            fh.close()
        except Exception:
            pass

def _open_follow(path: str, encoding: str):
    # open and seek to end
    f = open(path, "r", encoding=encoding, errors="replace")
    f.seek(0, io.SEEK_END)
    return f

# -------------------------- helpers --------------------------

def _nginx_ts_to_iso(ts: str) -> str:
    """
    Convert Nginx time '16/Jan/2025:15:20:10 +0000' -> '2025-01-16T15:20:10+00:00'
    """
    import datetime as dt
    try:
        t = dt.datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S %z")
        return t.isoformat()
    except Exception:
        return ts

# -------------------------- self-test --------------------------

if __name__ == "__main__":
    norm = Normalizer(service="news-intel", env="dev")
    samples = [
        '{"ts":"2025-01-16T15:20:10Z","level":"info","msg":"hello","traceparent":"00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}',
        '127.0.0.1 - - [16/Jan/2025:15:20:10 +0000] "GET / HTTP/1.1" 200 612 "-" "curl/8.0"',
        'INFO:     127.0.0.1:54062 - "GET /health" 200 OK',
        'Jan 16 15:20:10 api[123]: dd.trace_id=42 dd.span_id=7 processing request',
        'random text without ts',
    ]
    for s in samples:
        print(parse_line(s, normalizer=norm))