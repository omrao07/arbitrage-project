# feature-store/entities.py
"""
Central entity registry for the feature store.

Entities defined
---------------
- equity           : join on ticker (e.g., "AAPL")
- fx_pair          : join on pair   (e.g., "EURUSD")
- macro_country    : join on country (e.g., "US","EA","UK","JP")
- credit_issuer    : join on issuer_id (CUSIP/Ticker/UUID)
- credit_index     : join on index_id (e.g., "CDXHY","CDXIG","ITRXEUR")
- underlier        : join on symbol (options underlier: "AAPL","SPY","ES","BTC")

Keep these join keys consistent in your Parquet/streaming payloads and FeatureViews.
"""

from feast import Entity # type: ignore

# ----------------------------- Core Entities -----------------------------

equity = Entity(
    name="equity",
    join_keys=["ticker"],
    description="Equity ticker symbol (e.g., AAPL, TSLA, SPY).",
    tags={"domain": "equities"},
)

fx_pair = Entity(
    name="fx_pair",
    join_keys=["pair"],  # 6-letter pair like EURUSD, USDJPY
    description="FX currency pair (BASE+QUOTE).",
    tags={"domain": "fx"},
)

macro_country = Entity(
    name="macro_country",
    join_keys=["country"],  # ISO-like country/region code: US, EA, UK, JP, ...
    description="Country/region code for macro features.",
    tags={"domain": "macro"},
)

credit_issuer = Entity(
    name="credit_issuer",
    join_keys=["issuer_id"],  # issuer identifier (CUSIP/Ticker/UUID—use one consistently)
    description="Issuer identifier for single-name credit features.",
    tags={"domain": "credit"},
)

credit_index = Entity(
    name="credit_index",
    join_keys=["index_id"],  # e.g., CDXHY, CDXIG, ITRXEUR
    description="Credit index identifier (HY/IG/ITRX/etc).",
    tags={"domain": "credit"},
)

underlier = Entity(
    name="underlier",
    join_keys=["symbol"],  # options underlier (AAPL, SPY, ES, BTC, ...)
    description="Options underlier symbol used for vol-surface features.",
    tags={"domain": "options"},
)

# ----------------------------- Exports -----------------------------------

__all__ = [
    "equity",
    "fx_pair",
    "macro_country",
    "credit_issuer",
    "credit_index",
    "underlier",
]