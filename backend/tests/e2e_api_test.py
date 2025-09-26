# e2e_api_test.py
"""
End-to-end API tests for your quant/valuation service.

Usage:
  pip install pytest requests
  export API_BASE_URL="http://localhost:8000"         # or your deployed URL (no trailing slash)
  export API_TOKEN="..."                               # optional Bearer token
  pytest -q e2e_api_test.py

Environment:
  - API_BASE_URL (required) : Base URL (e.g., http://localhost:8000)
  - API_TOKEN    (optional) : Bearer token for auth
  - API_TIMEOUT  (optional) : seconds (default 20)
  - API_RETRIES  (optional) : attempts per request (default 2)
  - API_VERIFY_TLS (optional): "0" to disable TLS verify (default verify on)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import pytest # type: ignore
import requests

# ----------------------------
# Config / Helpers
# ----------------------------

API_BASE = os.environ.get("API_BASE_URL")
if API_BASE:
    API_BASE = API_BASE.rstrip("/")
API_TOKEN = os.environ.get("API_TOKEN")
API_TIMEOUT = float(os.environ.get("API_TIMEOUT", "20"))
API_RETRIES = int(os.environ.get("API_RETRIES", "2"))
VERIFY_TLS = os.environ.get("API_VERIFY_TLS", "1") not in ("0", "false", "False", "NO", "no")

pytestmark = pytest.mark.skipif(not API_BASE, reason="Set API_BASE_URL to run E2E tests.")


def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def request_json(method: str, path: str, expected_status: int = 200, json_body: Optional[dict] = None) -> Any:
    """
    Request helper with small retry + helpful skips:
    - If endpoint returns 404, we skip the test (not implemented).
    - If status != expected, we assert with body context.
    """
    url = f"{API_BASE}{path}"
    last_exc = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                headers=_headers(),
                json=json_body,
                timeout=API_TIMEOUT,
                verify=VERIFY_TLS,
            )
            if resp.status_code == 404:
                pytest.skip(f"Endpoint {path} not implemented (404).")
            if resp.status_code != expected_status:
                # Include response preview for debugging
                snippet = resp.text[:500]
                raise AssertionError(
                    f"{method} {path} expected {expected_status} got {resp.status_code}\n"
                    f"Response: {snippet}"
                )
            if "application/json" in resp.headers.get("Content-Type", ""):
                return resp.json()
            # Fallback: try json parse; else raw text
            try:
                return resp.json()
            except Exception:
                return resp.text
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < API_RETRIES:
                time.sleep(0.5 * attempt)
                continue
            raise
    if last_exc:
        raise last_exc


# ----------------------------
# Tests
# ----------------------------

def test_health_ok():
    """
    Accept any of these common health paths:
      /health, /v1/health, /api/health
    """
    for p in ("/health", "/v1/health", "/api/health"):
        try:
            data = request_json("GET", p, 200)
            # Accept flexible shapes; assert minimal truthy signal
            if isinstance(data, dict):
                assert any(k.lower() in ("ok", "status", "healthy") for k in data.keys()) or data != {}
            else:
                assert data  # non-empty
            return
        except pytest.skip.Exception:
            continue
    pytest.skip("No health endpoint found among tried paths.")


def test_valuation_dcf_perpetuity():
    """
    Posts a realistic DCF input. Expects a JSON with fair value fields.
    Accepted paths: /valuation, /v1/valuation, /api/valuation
    """
    payload = {
        "snap": {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "price": 210.0,
            "sharesOut": 15.6e9,
            "netDebt": -60e9,
            "fcf0": 100e9,
            "revenue0": 383e9,
            "ebitda0": 130e9,
            "eps0": 6.4,
        },
        "peers": {"evEbitda": 22.0, "pe": 28.0, "ps": 7.5},
        "dcf": {
            "years": 5,
            "wacc": 0.09,
            "growth": 0.08,
            "fadeTo": 0.03,
            "exitMode": "perpetuity",
            "exitG": 0.025,
            "ebitdaMargin": 0.28,
        },
    }
    for p in ("/valuation", "/v1/valuation", "/api/valuation"):
        try:
            data = request_json("POST", p, 200, payload)
            # Minimal schema expectations
            assert isinstance(data, dict)
            # Accept any naming like fair, fair_value, price_fair
            joined = json.dumps(data).lower()
            assert any(k in joined for k in ("fair", "fair_value", "fairvalue", "price_fair"))
            return
        except pytest.skip.Exception:
            continue
    pytest.skip("No valuation endpoint found among tried paths.")


def test_scenarios_run_basic():
    """
    Sends a scenario with shocks and expects computed metrics back.
    Accepted paths: /scenarios/run, /v1/scenarios/run, /api/scenarios/run
    """
    scenario = {
        "id": "e2e-upside",
        "name": "E2E Upside",
        "tag": "Macro",
        "date": "2025-09-01",
        "base": {"Revenue": 100.0, "EBITDA": 25.0, "EPS": 5.0, "Margin": 0.20, "EV/EBITDA": 12.0},
        "shocks": [
            {"variable": "Revenue", "type": "rel", "value": 0.05},
            {"variable": "EBITDA", "type": "rel", "value": 0.08},
            {"variable": "Margin", "type": "abs", "value": 0.01},
        ],
    }
    for p in ("/scenarios/run", "/v1/scenarios/run", "/api/scenarios/run"):
        try:
            data = request_json("POST", p, 200, {"scenario": scenario})
            assert isinstance(data, dict)
            # Expect a "result" or "metrics" object containing updated values
            result = data.get("result") or data.get("metrics") or data
            assert all(k in result for k in ("Revenue", "EBITDA", "EPS", "Margin"))
            assert result["Revenue"] >= 100.0  # shocked up
            return
        except pytest.skip.Exception:
            continue
    pytest.skip("No scenarios/run endpoint found among tried paths.")


def test_regimes_query_range():
    """
    Queries regimes over a date range; expects an array of points with date+regime.
    Accepted paths: /regimes, /v1/regimes, /api/regimes
    """
    params = "?from=2024-01-01&to=2025-09-01"
    for base in ("/regimes", "/v1/regimes", "/api/regimes"):
        path = f"{base}{params}"
        try:
            data = request_json("GET", path, 200)
            assert isinstance(data, (list, dict))
            # If dict wraps the list (e.g., {"series": [...]})
            rows = data.get("series") if isinstance(data, dict) else data
            assert isinstance(rows, list)
            if rows:
                row0 = rows[0]
                assert any(k in row0 for k in ("date", "ts"))
                assert any(k.lower() == "regime" for k in row0.keys())
            return
        except pytest.skip.Exception:
            continue
    pytest.skip("No regimes endpoint found among tried paths.")


def test_comps_query_ticker():
    """
    Queries comps for a ticker; expects at least the ticker and a few multiples.
    Accepted paths: /comps, /v1/comps, /api/comps
    """
    params = "?ticker=AAPL"
    for base in ("/comps", "/v1/comps", "/api/comps"):
        path = f"{base}{params}"
        try:
            data = request_json("GET", path, 200)
            # Accept list or {data:[...]}
            rows = data.get("data") if isinstance(data, dict) else data
            assert isinstance(rows, list)
            if rows:
                first = rows[0]
                # Expect basic fields
                assert any(k.lower() in ("ticker", "symbol") for k in first.keys())
                assert any(k.replace("_", "").lower() in ("evebitda", "pe", "ps") for k in first.keys())
            return
        except pytest.skip.Exception:
            continue
    pytest.skip("No comps endpoint found among tried paths.")