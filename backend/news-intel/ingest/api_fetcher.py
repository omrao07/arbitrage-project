# news-intel/ingest/api_fetcher.py
"""
Generic API fetcher for news sources.

Features
- stdlib-only HTTP client (urllib). If 'httpx' is available, it will be used.
- Retries with exponential backoff + jitter.
- Simple rate limiting (requests per second) across the fetcher.
- Pagination helpers (cursor/next-link, page+size, offset+limit).
- Normalization into a common Article schema:
    {
      "id": str, "title": str, "body": str, "url": str,
      "published_at": str, "source": str, "raw": dict
    }

Usage
-----
fetcher = ApiFetcher(base_url="https://api.example.com",
                     default_headers={"Authorization": "Bearer X"},
                     rps=3.0, timeout=10.0)

# Single call
data = fetcher.get_json("/v1/news", params={"q":"ai"})

# Pagination (cursor in 'next' or 'next_page' field)
for page in fetcher.paginate_json("/v1/news", params={"q":"ai"}):
    ...

# Normalize a page of items to articles using a mapping
articles = fetcher.map_to_articles(
    items=data["results"],
    mapping={
        "id": "id",
        "title": "headline",
        "body": "summary",
        "url": "link",
        "published_at": "published_at",
        "source": lambda item: item.get("source", {}).get("name", "unknown"),
    },
    default_source="example"
)
"""

from __future__ import annotations

import json
import time
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union

# Optional fast client
try:
    import httpx  # type: ignore
except Exception:  # noqa: BLE001
    httpx = None  # type: ignore[assignment]


@dataclass
class HttpResponse:
    status: int
    headers: Dict[str, str]
    text: str

    def json(self) -> Any:
        return json.loads(self.text)


class RateLimiter:
    """Simple token bucket for requests-per-second."""
    def __init__(self, rps: float):
        self.capacity = max(1.0, rps)
        self.tokens = self.capacity
        self.fill_rate = rps
        self.timestamp = time.monotonic()

    def acquire(self) -> None:
        now = time.monotonic()
        elapsed = now - self.timestamp
        self.timestamp = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        if self.tokens < 1.0:
            sleep_s = (1.0 - self.tokens) / self.fill_rate
            time.sleep(max(0.0, sleep_s))
            # refill after sleeping
            now2 = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now2 - now) * self.fill_rate)
            self.timestamp = now2
        self.tokens -= 1.0


def _join_url(base: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return urllib.parse.urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _encode_params(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return ""
    # Flatten lists: {"a": [1,2]} -> a=1&a=2
    return urllib.parse.urlencode([(k, v) for k, vv in params.items()
                                   for v in (vv if isinstance(vv, (list, tuple)) else [vv])])


class ApiFetcher:
    def __init__(
        self,
        base_url: str,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        rps: float = 5.0,
        max_retries: int = 4,
        backoff_base: float = 0.25,  # seconds
    ):
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.timeout = timeout
        self.rps = rps
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._rl = RateLimiter(rps) if rps > 0 else None

        # prefer httpx if present
        self._use_httpx = httpx is not None

    # --------------------- low-level HTTP ---------------------

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[Dict[str, Any], str, bytes]] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        url = _join_url(self.base_url, path)
        if params:
            qs = _encode_params(params)
            url = f"{url}?{qs}" if qs else url

        hdrs = dict(self.default_headers)
        if headers:
            hdrs.update(headers)

        tmo = timeout if timeout is not None else self.timeout

        # basic retry loop
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if self._rl:
                self._rl.acquire()

            try:
                if self._use_httpx:
                    assert httpx is not None
                    data = body
                    if isinstance(body, (dict, list)):
                        data = json.dumps(body).encode("utf-8")
                        hdrs.setdefault("Content-Type", "application/json")
                    with httpx.Client(timeout=tmo) as client:
                        resp = client.request(method.upper(), url, headers=hdrs, content=data) # type: ignore
                        text = resp.text
                        status = resp.status_code
                        # retry on 429/5xx
                        if status in (429, 500, 502, 503, 504):
                            raise _RetryableHTTP(f"status {status}", status)
                        return HttpResponse(status=status, headers=dict(resp.headers), text=text)

                # stdlib path
                req = urllib.request.Request(url=url, method=method.upper(), headers=hdrs)
                data = None
                if isinstance(body, (dict, list)):
                    data = json.dumps(body).encode("utf-8")
                    req.add_header("Content-Type", "application/json")
                elif isinstance(body, str):
                    data = body.encode("utf-8")
                elif isinstance(body, bytes):
                    data = body

                with urllib.request.urlopen(req, data=data, timeout=tmo) as resp:  # nosec B310
                    status = getattr(resp, "status", 200)
                    text = resp.read().decode("utf-8", errors="replace")
                    if status in (429, 500, 502, 503, 504):
                        raise _RetryableHTTP(f"status {status}", status)
                    headers_out = {k.lower(): v for k, v in resp.headers.items()}
                    return HttpResponse(status=status, headers=headers_out, text=text)

            except _RetryableHTTP as e:
                last_exc = e
                self._sleep_backoff(attempt, e.status)
                continue
            except Exception as e:  # network errors etc.
                last_exc = e
                # consider network errors retryable for early attempts
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt, None)
                    continue
                raise

        # exhausted retries
        assert last_exc is not None
        raise last_exc

    def get_json(self, path: str, **kwargs) -> Any:
        resp = self.request("GET", path, **kwargs)
        return resp.json()

    # --------------------- pagination helpers ---------------------

    def paginate_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        next_key: Optional[str] = None,      # e.g. "next" or "next_page"
        items_key: Optional[str] = None,     # e.g. "results" or "data"
        page_param: str = "page",
        per_page_param: str = "per_page",
        start_page: int = 1,
        per_page: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield JSON pages. Supports two common patterns:
        1) Cursor/URL continuation: response[next_key] contains a full URL or cursor token.
        2) Page numbers: ?page=N&per_page=M

        If both next_key and items_key are provided, we still yield the full page dict
        so the caller can extract items.
        """
        p = dict(params or {})
        h = dict(headers or {})

        # Strategy A: page numbers (if next_key is None)
        if next_key is None:
            page = start_page
            if per_page is not None:
                p[per_page_param] = per_page
            while True:
                p[page_param] = page
                data = self.get_json(path, params=p, headers=h)
                yield data
                page += 1
                if max_pages and page > max_pages:
                    break
                # stop when no items or identical page
                if items_key:
                    items = data.get(items_key, [])
                    if not items:
                        break
                else:
                    # if server returns a 'next' anyway, honor it
                    if isinstance(data, dict) and "next" in data and not data["next"]:
                        break
        else:
            # Strategy B: cursor or next URL
            next_val: Optional[str] = None
            first_url = path
            while True:
                if next_val:
                    # next_val may be full URL or cursor token
                    if next_val.startswith("http"):
                        url = next_val
                        # When full URL provided, drop base/path handling
                        data = self.get_json(url)
                    else:
                        p[next_key] = next_val
                        data = self.get_json(first_url, params=p, headers=h)
                else:
                    data = self.get_json(first_url, params=p, headers=h)

                yield data
                # compute next
                nv = None
                if isinstance(data, dict):
                    nv = data.get(next_key)
                    # some APIs put pagination under data['pagination']['next']
                    if nv is None:
                        pg = data.get("pagination") or {}
                        nv = pg.get("next") if isinstance(pg, dict) else None
                next_val = nv
                if not next_val:
                    break

    # --------------------- normalization ---------------------

    def map_to_articles(
        self,
        items: Iterable[Dict[str, Any]],
        mapping: Dict[str, Union[str, Callable[[Dict[str, Any]], Any]]],
        default_source: str = "unknown",
    ) -> List[Dict[str, Any]]:
        """
        Map a list of item dicts to our Article schema using a field mapping.
        `mapping` keys: id, title, body, url, published_at, source (optional).
        Values may be a key string or a callable item -> value.
        """
        def get(item: Dict[str, Any], key_or_fn: Union[str, Callable[[Dict[str, Any]], Any]]) -> Any:
            if callable(key_or_fn):
                return key_or_fn(item)
            # support dotted paths, e.g., "source.name"
            cur: Any = item
            for part in key_or_fn.split("."):
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(part)
                if cur is None:
                    return None
            return cur

        out: List[Dict[str, Any]] = []
        for it in items:
            art = {
                "id": str(get(it, mapping.get("id", "id")) or ""),
                "title": (get(it, mapping.get("title", "title")) or "")[:500],
                "body": get(it, mapping.get("body", "body")) or "",
                "url": get(it, mapping.get("url", "url")) or "",
                "published_at": get(it, mapping.get("published_at", "published_at")) or "",
                "source": get(it, mapping.get("source", lambda x: x.get("source", "unknown"))) or default_source,
                "raw": it,
            }
            # skip empty rows (require at least id or url or title)
            if art["id"] or art["url"] or art["title"]:
                out.append(art)
        return out

    # --------------------- helpers ---------------------

    def _sleep_backoff(self, attempt: int, status: Optional[int]) -> None:
        # exponential backoff with jitter; honor Retry-After if present in 429/503 (httpx path already handled)
        base = self.backoff_base * (2 ** attempt)
        jitter = random.uniform(0, base * 0.25)
        time.sleep(base + jitter)


class _RetryableHTTP(Exception):
    def __init__(self, msg: str, status: Optional[int] = None):
        super().__init__(msg)
        self.status = status


# --------------------- quick self-test ---------------------
if __name__ == "__main__":
    # This block shows shape only; replace with a real endpoint.
    fetcher = ApiFetcher("https://example.com/api", rps=2.0)
    print("This is a scaffold. Point to a real endpoint for actual results.")