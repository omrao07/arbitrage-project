# backend/literacy/demos.py
"""
Learning-Mode Demos
-------------------
Simple demos you can run live for classrooms/Rotaract:
- Streams Yahoo + Moneycontrol headlines
- Uses SentimentModel (FinBERT -> VADER fallback)
- Emits plain-English "bullish/bearish/neutral" takes with a tiny risk explainer

Run:
  python -m backend.literacy.demos --tickers RELIANCE.NS ^NSEI --moneycontrol https://www.moneycontrol.com/rss/marketreports.xml
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, List
from collections import deque

# sentiment + news sources (use your existing modules)
from backend.analytics.sentiment_ai import SentimentModel, sentiment_weight # type: ignore
from backend.ingestion.news.news_base import NewsEvent
from backend.ingestion.news.news_yahoo import YahooNews
from backend.ingestion.news.news_moneycontrol import MoneycontrolNews # type: ignore


# ---------- Pretty printing helpers ----------

def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def _explain_weight(w: float) -> str:
    # map [-1,1] to simple text
    if w >= 0.25:
        return "Bullish"
    if w <= -0.25:
        return "Bearish"
    return "Neutral"

def _risk_tip(label: str) -> str:
    tips = {
        "Bullish": "Reminder: even good news can be priced in. Size small; use stop-loss.",
        "Bearish": "Negative headlines can overreact. Watch for reversals and liquidity.",
        "Neutral": "No clear edge. Patience often beats forced trades.",
    }
    return tips.get(label, "")

def _one_liner(ev: NewsEvent, label: str, score: float) -> str:
    base = f"[{_fmt_ts(ev.published_at)}] {label}: {ev.headline}"
    if ev.symbol:
        base += f" (symbol: {ev.symbol})"
    return f"{base}  | sentiment={score:+.2f}  source={ev.source}"

# ---------- Ring buffer so the console stays tidy ----------

@dataclass
class Clip:
    when: float
    text: str

class RollingConsole:
    def __init__(self, max_lines: int = 12):
        self.buf: deque[Clip] = deque(maxlen=max_lines)

    def push(self, text: str):
        self.buf.append(Clip(time.time(), text))
        self.render()

    def render(self):
        print("\033[2J\033[H", end="")  # clear screen, move home
        print("=== Learning Mode: Live News → Sentiment → Plain-English Signals ===")
        for c in list(self.buf):
            print(c.text)
        print("\n(Press Ctrl+C to stop)")

# ---------- Demo core ----------

async def _run_sources(tickers: List[str], feeds: List[str], sink, stop: asyncio.Event, log=print):
    tasks = []
    if tickers:
        yn = YahooNews(tickers=tickers, poll_seconds=60)
        tasks.append(asyncio.create_task(yn.run(sink=sink, interval=yn.poll_seconds, stop=stop, log=log)))
    if feeds:
        mc = MoneycontrolNews(feeds=feeds, poll_seconds=60)
        tasks.append(asyncio.create_task(mc.run(sink=sink, interval=mc.poll_seconds, stop=stop, log=log)))
    if not tasks:
        log("No sources configured. Provide --tickers and/or --moneycontrol.")
        return
    await asyncio.gather(*tasks)

async def classroom_demo(tickers: List[str], moneycontrol_feeds: List[str], min_conf: float = 0.3):
    """
    Streams headlines and prints simple signals + risk reminders.
    """
    queue: asyncio.Queue[NewsEvent] = asyncio.Queue()
    stop = asyncio.Event()
    console = RollingConsole(max_lines=14)
    sent = SentimentModel(prefer_finbert=True)

    async def producer():
        await _run_sources(tickers, moneycontrol_feeds, sink=queue.put_nowait, stop=stop, log=lambda *_: None)

    async def consumer():
        while not stop.is_set():
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            sr = sent.analyze_event(ev)
            w = sentiment_weight(sr, min_conf=min_conf)
            label = _explain_weight(w)
            tip = _risk_tip(label)
            console.push(_one_liner(ev, label, sr.score) + (f"\n   ⚠ {tip}" if tip else ""))

    try:
        await asyncio.gather(producer(), consumer())
    except asyncio.CancelledError:
        pass
    finally:
        stop.set()

async def rotaract_demo():
    """
    Minimal canned demo if you just want to show the UX without live internet.
    """
    console = RollingConsole(max_lines=10)
    samples = [
        NewsEvent(id="1", source="yahoo", headline="RBI holds policy rate; hints at inflation risks", url="", published_at=time.time(), summary="", symbol="^NSEI"),
        NewsEvent(id="2", source="moneycontrol", headline="Reliance Q1 profit beats estimates on refining margins", url="", published_at=time.time(), summary="", symbol="RELIANCE.NS"),
        NewsEvent(id="3", source="yahoo", headline="Tech stocks slump as US yields rise", url="", published_at=time.time(), summary="", symbol="^NDX"),
    ]
    sent = SentimentModel(prefer_finbert=False)  # quick start with VADER if offline
    for ev in samples:
        sr = sent.analyze_event(ev)
        label = _explain_weight(sr.score)
        tip = _risk_tip(label)
        console.push(_one_liner(ev, label, sr.score) + (f"\n   ⚠ {tip}" if tip else ""))
        await asyncio.sleep(1.2)

# ---------- CLI ----------

def _parse_args():
    p = argparse.ArgumentParser(description="Learning Mode demos")
    p.add_argument("--tickers", nargs="*", default=["RELIANCE.NS", "^NSEI"], help="Yahoo tickers to follow")
    p.add_argument("--moneycontrol", nargs="*", default=[], help="Moneycontrol RSS URLs")
    p.add_argument("--mode", choices=["classroom", "rotaract"], default="classroom")
    p.add_argument("--min-conf", type=float, default=0.3, help="Confidence floor for signals (0..1)")
    return p.parse_args()

def main():
    args = _parse_args()
    try:
        if args.mode == "classroom":
            asyncio.run(classroom_demo(args.tickers, args.moneycontrol, min_conf=args.min_conf))
        else:
            asyncio.run(rotaract_demo())
    except KeyboardInterrupt:
        print("\nStopping…")

if __name__ == "__main__":
    main()