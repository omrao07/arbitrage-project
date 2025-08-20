# backend/ingestion/news/news_yahoo.py
from __future__ import annotations

import time
from typing import List, Sequence, Optional

try:
    import yfinance as yf  # pip install yfinance
except Exception as e:
    raise ImportError("yfinance is required for Yahoo news. pip install yfinance") from e

from .news_base import (
    NewsSource,
    NewsEvent,
    hash_key,
    clean_text,
    to_unix,
)


class YahooNews(NewsSource):
    """
    Polls Yahoo Finance news per ticker using yfinance.Ticker(t).news.

    Example:
        yn = YahooNews(
            tickers=["AAPL", "MSFT", "^NSEI", "RELIANCE.NS"],
            poll_seconds=60,
            dedupe_ttl_seconds=7200,
        )
        await yn.run(sink=queue.put_nowait, interval=yn.poll_seconds, stop=stop)
    """

    source_name = "yahoo"

    def __init__(
        self,
        tickers: Sequence[str],
        poll_seconds: int = 60,
        dedupe_ttl_seconds: int = 7200,
    ):
        super().__init__(dedupe_ttl_seconds=dedupe_ttl_seconds)
        if not tickers:
            raise ValueError("YahooNews requires at least one ticker symbol.")
        self.tickers = list(tickers)
        self.poll_seconds = int(max(5, poll_seconds))

    # sync poll (base will run this in a thread)
    def poll_sync(self) -> List[NewsEvent]:
        events: List[NewsEvent] = []
        now = time.time()

        for t in self.tickers:
            try:
                items = yf.Ticker(t).news or []  # list[dict]
            except Exception:
                # network/library error; skip this ticker for this round
                continue

            for it in items:
                # Fields commonly present: title, link, publisher, providerPublishTime, type, relatedTickers, content
                headline = clean_text(str(it.get("title", "") or ""))
                if not headline:
                    continue

                url = str(it.get("link", "") or "")
                ts = to_unix(it.get("providerPublishTime", None)) or now
                summary = clean_text(
                    str(it.get("summary") or it.get("content") or "")[:600]
                )

                # Prefer the exact ticker we queried; fall back to related tickers if present
                symbol: Optional[str] = t
                rel = it.get("relatedTickers")
                if not symbol and isinstance(rel, list) and rel:
                    symbol = str(rel[0])

                # Stable id for dedupe (minute granularity)
                eid = hash_key("yahoo", t, headline, str(int(ts // 60)))

                events.append(
                    NewsEvent(
                        id=eid,
                        source=self.source_name,
                        headline=headline,
                        url=url,
                        published_at=ts,
                        summary=summary,
                        symbol=symbol,
                        raw=it,
                    )
                )

        return events