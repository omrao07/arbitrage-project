# backend/data_ext/sentiment/sources/discord.py
"""
Discord source loader for social sentiment.

Reads config from sentiment.yaml -> sources.discord and returns raw messages
for social_scraper.py to score (via sentiment_model.py) and normalize.

Two modes:
1) Real mode (requires discord.py and a bot token; bot must be in the target servers):
   - pip install discord.py
   - Config: bot_token, servers[{id, channels[]}], max_messages
   - NOTE: Only read channels your bot has permission to read. Respect server rules.

2) Demo mode (no client / no token):
   - Emits plausible fake messages for pipeline testing.

Returned record schema (raw):
{
  "source": "discord",
  "server": "MyTradingHub",
  "channel": "markets",
  "text": "AAPL call flow looks heavy today",
  "timestamp": "2025-08-22T00:15:00Z",
  "symbol": "AAPL",                 # optional, best-effort extraction
  "meta": {
      "author": "TraderJoe#1234",
      "author_id": "1234567890",
      "message_id": "9876543210",
      "reactions": 12,
      "attachments": 0
  }
}
"""

from __future__ import annotations

import os
import re
import time
import asyncio
import datetime as dt
from typing import Any, Dict, List, Optional

# Try real client
_HAVE_DISCORD = False
try:
    import discord  # type: ignore
    _HAVE_DISCORD = True
except Exception:
    _HAVE_DISCORD = False

SYMBOL_DOLLAR = re.compile(r"\$([A-Za-z]{1,6})")
SYMBOL_UPPER  = re.compile(r"\b([A-Z]{2,5})\b")


def _iso(dtobj: dt.datetime) -> str:
    return dtobj.replace(tzinfo=None, microsecond=0).isoformat() + "Z"


def _extract_symbol(text: str, whitelist: Optional[List[str]] = None) -> Optional[str]:
    if not text:
        return None
    m = SYMBOL_DOLLAR.search(text)
    if m:
        cand = m.group(1).upper()
        if not whitelist or cand in whitelist:
            return cand
    m2 = SYMBOL_UPPER.search(text.upper())
    if m2:
        cand = m2.group(1).upper()
        if not whitelist or cand in whitelist:
            return cand
    return None


# ------------------------------------------------------------------------------
# Real fetch (discord.py)
# ------------------------------------------------------------------------------

async def _fetch_discord_async(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Connects with a bot and fetches recent messages from specified channels.
    cfg:
      bot_token: "...",
      servers: [
        { "id": "123", "channels": ["markets", "crypto"] },
        { "id": "456", "channels": ["flow", "earnings"] }
      ],
      max_messages: 50,
      symbols: ["TSLA","AAPL","NVDA","BTC"]  # optional
    """
    token = (cfg.get("bot_token") or os.getenv("DISCORD_BOT_TOKEN") or "").strip()
    if not (_HAVE_DISCORD and token):
        return []

    intents = discord.Intents.none() # type: ignore
    intents.guilds = True
    intents.messages = True
    intents.message_content = True  # must be enabled in bot settings too

    client = discord.Client(intents=intents) # type: ignore

    out: List[Dict[str, Any]] = []
    max_messages = int(cfg.get("max_messages", 50))
    servers_cfg = cfg.get("servers") or []
    symbols_whitelist: Optional[List[str]] = cfg.get("symbols")

    ready_event = asyncio.Event()

    @client.event
    async def on_ready():
        try:
            for srv_cfg in servers_cfg:
                guild_id = int(srv_cfg.get("id"))
                guild = client.get_guild(guild_id)
                if guild is None:
                    continue

                # Build a lookup of desired channels by name
                wanted = set(map(str.lower, srv_cfg.get("channels") or []))

                for ch in guild.text_channels:
                    if wanted and ch.name.lower() not in wanted:
                        continue

                    # Fetch recent messages
                    async for msg in ch.history(limit=max_messages):
                        if msg.author.bot:
                            continue
                        text = msg.content or ""
                        if not text.strip() and not msg.embeds:
                            continue
                        # Simple text aggregation: content + embed titles
                        if msg.embeds:
                            et = " ".join([e.title or "" for e in msg.embeds if hasattr(e, "title") and e.title])
                            text = (text + " " + et).strip()

                        rec = {
                            "source": "discord",
                            "server": guild.name,
                            "channel": ch.name,
                            "text": text,
                            "timestamp": _iso(msg.created_at.replace(tzinfo=None)),
                            "symbol": _extract_symbol(text, symbols_whitelist),
                            "meta": {
                                "author": f"{getattr(msg.author, 'name', 'user')}#{getattr(msg.author, 'discriminator', '')}",
                                "author_id": str(getattr(msg.author, 'id', '')),
                                "message_id": str(getattr(msg, 'id', '')),
                                "reactions": sum(r.count for r in msg.reactions),
                                "attachments": len(msg.attachments or []),
                            },
                        }
                        out.append(rec)
        finally:
            ready_event.set()
            await client.close()

    # Run the client until on_ready completes the harvest
    await client.start(token)
    await ready_event.wait()
    return out


def _fetch_discord_real(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not (_HAVE_DISCORD and (cfg.get("bot_token") or os.getenv("DISCORD_BOT_TOKEN"))):
        return []
    try:
        return asyncio.run(_fetch_discord_async(cfg))
    except Exception:
        return []


# ------------------------------------------------------------------------------
# Fallback generator (demo)
# ------------------------------------------------------------------------------

_FAKE = [
    ("MyTradingHub", "markets", "AAPL call flow looks heavy today, watch 200DMA", "AAPL"),
    ("CryptoAlpha", "crypto", "BTC funding flipping positive, risky squeeze setup", "BTC"),
    ("MacroLab", "macro", "NVDA capex cycle still intact, hyperscalers ramp", "NVDA"),
    ("EnergyTalk", "oil", "Crack spreads widening; refiners could beat", None),
]

def _fetch_discord_fallback(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    now = _iso(dt.datetime.utcnow())
    out: List[Dict[str, Any]] = []
    for server, channel, text, sym in _FAKE:
        out.append(
            {
                "source": "discord",
                "server": server,
                "channel": channel,
                "text": text,
                "timestamp": now,
                "symbol": sym or _extract_symbol(text, cfg.get("symbols")),
                "meta": {
                    "author": "TraderBot#0001",
                    "author_id": "0",
                    "message_id": f"demo_{int(time.time())}",
                    "reactions": 5,
                    "attachments": 0,
                },
            }
        )
    return out


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def fetch(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Entry for social_scraper.py.

    sentiment.yaml example:
    -----------------------
    sources:
      discord:
        enabled: false
        bot_token: "${DISCORD_BOT_TOKEN}"
        servers:
          - id: "123456789012345678"
            channels: ["markets", "crypto"]
        max_messages: 50
        symbols: ["TSLA","AAPL","NVDA","BTC"]  # optional whitelist
    """
    if not cfg.get("enabled", False):
        return []

    real = _fetch_discord_real(cfg)
    if real:
        return real

    return _fetch_discord_fallback(cfg)


# ------------------------------------------------------------------------------
# Demo CLI
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    demo_cfg = {
        "enabled": True,
        "bot_token": os.getenv("DISCORD_BOT_TOKEN", ""),
        "servers": [{"id": "123456789012345678", "channels": ["markets", "crypto"]}],
        "max_messages": 20,
        "symbols": ["TSLA", "AAPL", "NVDA", "BTC"],
    }
    msgs = fetch(demo_cfg)
    for m in msgs:
        print(m)