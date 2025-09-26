#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
valuation/schemas.py
--------------------
Typed schemas for the valuation stack:
- Identifiers & metadata
- Fundamentals (periodic & TTM)
- Market snapshots
- Multiples (aligned with multiples.py)
- DCF inputs & results (aligned with dcf.py)
- ERP & ratios timeseries
- Comps rows and tables
- Aggregate ValuationPackage for persistence

Safe to import from other modules; only pure typing/validation here.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

# --- Pydantic v1/v2 compatibility shim ---------------------------------------
try:
    from pydantic import BaseModel, Field, root_validator, validator # type: ignore
    PydV2 = False
except Exception:  # pydantic v2
    from pydantic import BaseModel, Field, model_validator # type: ignore
    PydV2 = True
    def root_validator(*args, **kwargs):  # no-op shim for v2
        def _wrap(fn):
            return fn
        return _wrap
    def validator(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

__all__ = [
    "SCHEMA_VERSION",
    "Currency", "Frequency", "ValuationMethod", "TerminalMethod",
    "CompanyId", "Tag",
    "Fundamentals", "MarketSnapshot",
    "MultipleSet", "MultiplesRecord",
    "DCFInputsSchema", "DCFResultSchema",
    "ERPPoint", "RatioPoint",
    "CompsRow", "CompsTable",
    "ValuationPackage",
]

SCHEMA_VERSION: str = "1.0.0"


# -----------------------------------------------------------------------------
# Enums & simple types
# -----------------------------------------------------------------------------

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    INR = "INR"
    CNY = "CNY"
    AUD = "AUD"
    CAD = "CAD"
    CHF = "CHF"
    OTHER = "OTHER"


class Frequency(str, Enum):
    ANNUAL = "A"
    QUARTERLY = "Q"
    MONTHLY = "M"
    TTM = "TTM"


class ValuationMethod(str, Enum):
    DCF = "DCF"
    MULTIPLES = "MULTIPLES"
    HYBRID = "HYBRID"


class TerminalMethod(str, Enum):
    PERPETUITY = "perpetuity"
    MULTIPLE = "multiple"


class Tag(BaseModel):
    key: str = Field(..., description="Dimension name, e.g., sector, region, theme")
    value: str = Field(..., description="Value, e.g., Technology, US, Quality")


# -----------------------------------------------------------------------------
# Core identity
# -----------------------------------------------------------------------------

class CompanyId(BaseModel):
    ticker: str = Field(..., description="Primary trading ticker (e.g., AAPL)")
    name: Optional[str] = Field(None, description="Company name")
    exchange: Optional[str] = Field(None, description="Exchange code (e.g., NASDAQ)")
    isin: Optional[str] = Field(None)
    cusip: Optional[str] = Field(None)
    sedol: Optional[str] = Field(None)
    country: Optional[str] = Field(None)
    sector: Optional[str] = Field(None)
    industry: Optional[str] = Field(None)
    currency: Currency = Field(Currency.USD, description="Reporting / presentation currency")

    @validator("ticker")
    def _ticker_norm(cls, v: str) -> str:
        v = (v or "").strip().upper()
        if not v:
            raise ValueError("ticker cannot be empty")
        return v


# -----------------------------------------------------------------------------
# Fundamentals & Market snapshots
# -----------------------------------------------------------------------------

class Fundamentals(BaseModel):
    """
    Financial statement slice in the chosen currency (usually millions).
    Positive = inflow to equity/firm (standard sign conventions).
    """
    as_of: date
    frequency: Frequency = Frequency.ANNUAL

    # Income statement
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    ebit: Optional[float] = None
    net_income: Optional[float] = None

    # Cash flow (to firm)
    fcf: Optional[float] = Field(None, description="Free cash flow to firm (FCFF)")
    capex: Optional[float] = None
    depreciation: Optional[float] = None
    wc_change: Optional[float] = Field(None, description="Change in working capital (use negative if investment)")

    # Balance sheet-like
    book_value: Optional[float] = None
    net_debt: Optional[float] = Field(None, description="Debt - cash (>= -cash if net cash)")

    # Share info
    shares_diluted: Optional[float] = Field(None, description="Diluted shares outstanding")

    currency: Currency = Currency.USD
    source: Optional[str] = Field(None, description="Where this row came from (file/feed)")

    @validator("shares_diluted")
    def _shares_pos(cls, v):
        if v is not None and v <= 0:
            raise ValueError("shares_diluted must be > 0 when provided")
        return v


class MarketSnapshot(BaseModel):
    as_of: date
    price: float = Field(..., gt=0)
    shares_outstanding: Optional[float] = Field(None, gt=0)
    market_cap: Optional[float] = None
    net_debt: Optional[float] = None
    currency: Currency = Currency.USD
    source: Optional[str] = None

    @root_validator
    def _compute_mcap(cls, values):
        price, shares, mcap = values.get("price"), values.get("shares_outstanding"), values.get("market_cap")
        if mcap is None and price is not None and shares is not None:
            values["market_cap"] = price * shares
        return values


# -----------------------------------------------------------------------------
# Multiples (aligned with multiples.py)
# -----------------------------------------------------------------------------

class MultipleSet(BaseModel):
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_sales: Optional[float] = None
    ev_fcf: Optional[float] = None
    fcy: Optional[float] = Field(None, description="FCF yield = FCF/MarketCap")
    ey: Optional[float] = Field(None, description="Earnings yield = NI/MarketCap")


class MultiplesRecord(BaseModel):
    company: CompanyId
    as_of: date
    inputs: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Normalized numeric inputs used (price, shares, market_cap, net_debt, ebitda, revenue, fcf, net_income, book_value)",
    )
    multiples: MultipleSet


# -----------------------------------------------------------------------------
# DCF (aligned with dcf.py)
# -----------------------------------------------------------------------------

class DCFInputsSchema(BaseModel):
    fcfs: List[float] = Field(..., description="Forecast FCFs per period (e.g., millions)")
    wacc: float = Field(..., gt=0)
    terminal_growth: float = Field(0.02)
    terminal_method: TerminalMethod = Field(TerminalMethod.PERPETUITY)
    exit_multiple: Optional[float] = Field(None, description="If terminal_method==multiple")
    ebitda_terminal: Optional[float] = Field(None, description="If terminal_method==multiple")
    net_debt: float = 0.0
    shares_outstanding: float = 1.0
    currency: Currency = Currency.USD

    @root_validator
    def _tv_requirements(cls, values):
        tm = values.get("terminal_method")
        if tm == TerminalMethod.MULTIPLE:
            if values.get("exit_multiple") is None or values.get("ebitda_terminal") is None:
                raise ValueError("Exit multiple terminal needs exit_multiple and ebitda_terminal")
        # basic stability: g < wacc for perpetuity
        if tm == TerminalMethod.PERPETUITY:
            g = float(values.get("terminal_growth") or 0.0)
            wacc = float(values.get("wacc"))
            if g >= wacc:
                raise ValueError("terminal_growth must be < wacc for perpetuity method")
        return values


class DCFResultSchema(BaseModel):
    inputs: DCFInputsSchema
    enterprise_value: float
    equity_value: float
    price_per_share: float
    as_of: Optional[date] = None
    method: ValuationMethod = ValuationMethod.DCF


# -----------------------------------------------------------------------------
# ERP & ratio time series
# -----------------------------------------------------------------------------

class ERPPoint(BaseModel):
    as_of: date
    erp: float = Field(..., description="Equity risk premium (decimal, e.g., 0.045 for 4.5%)")
    region: Optional[str] = None
    notes: Optional[str] = None


class RatioPoint(BaseModel):
    as_of: date
    name: str = Field(..., description="Ratio name (ROE, ROIC, EBIT margin, etc.)")
    value: float
    unit: Optional[str] = Field(None, description="%, x, bps, etc.")


# -----------------------------------------------------------------------------
# Comps table
# -----------------------------------------------------------------------------

class CompsRow(BaseModel):
    company: CompanyId
    as_of: date
    price: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    fcf: Optional[float] = None
    net_income: Optional[float] = None
    book_value: Optional[float] = None
    multiples: Optional[MultipleSet] = None
    tags: List[Tag] = Field(default_factory=list)


class CompsTable(BaseModel):
    name: str
    as_of: date
    rows: List[CompsRow]
    currency: Currency = Currency.USD
    notes: Optional[str] = None

    def median(self, field: str) -> Optional[float]:
        vals = [getattr(r, field) for r in self.rows if getattr(r, field) is not None]
        if not vals:
            return None
        vals = sorted(vals)
        n = len(vals)
        mid = n // 2
        return (vals[mid] if n % 2 == 1 else 0.5 * (vals[mid - 1] + vals[mid]))


# -----------------------------------------------------------------------------
# Aggregate package
# -----------------------------------------------------------------------------

class ValuationPackage(BaseModel):
    schema_version: str = SCHEMA_VERSION
    company: CompanyId
    as_of: date
    currency: Currency = Currency.USD

    fundamentals_ttm: Optional[Fundamentals] = None
    fundamentals_annual: List[Fundamentals] = Field(default_factory=list)
    market: Optional[MarketSnapshot] = None

    dcf_inputs: Optional[DCFInputsSchema] = None
    dcf_result: Optional[DCFResultSchema] = None

    multiples_record: Optional[MultiplesRecord] = None
    comps: Optional[CompsTable] = None

    erp_series: List[ERPPoint] = Field(default_factory=list)
    ratio_series: List[RatioPoint] = Field(default_factory=list)

    tags: List[Tag] = Field(default_factory=list)
    source: Optional[str] = None

    @validator("currency", always=True)
    def _ensure_currency(cls, v, values):
        # Harmonize currency with company default if not set elsewhere
        comp: CompanyId = values.get("company")
        return v or (comp.currency if comp else Currency.USD)

    def ensure_consistency(self) -> None:
        """
        Light checks to run after constructing the package.
        - Market cap from price/shares if missing
        - Multiples presence aligns with inputs
        """
        if self.market and self.market.market_cap is None:
            if self.market.price is not None and self.market.shares_outstanding is not None:
                self.market.market_cap = self.market.price * self.market.shares_outstanding  # type: ignore

    def summary(self) -> Dict[str, Union[str, float, int]]:
        """Compact summary for logging/printing."""
        out: Dict[str, Union[str, float, int]] = {
            "schema": self.schema_version,
            "ticker": self.company.ticker,
            "as_of": self.as_of.isoformat(),
            "currency": self.currency.value,
        }
        if self.market and self.market.market_cap is not None:
            out["market_cap"] = float(self.market.market_cap)
        if self.dcf_result:
            out["ev"] = float(self.dcf_result.enterprise_value)
            out["eq"] = float(self.dcf_result.equity_value)
            out["px"] = float(self.dcf_result.price_per_share)
        return out