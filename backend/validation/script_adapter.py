"""
Adapter for `run(cfg)` research scripts — 162 of the 373 strategy files.

THE CONTRACT THESE SCRIPTS ACTUALLY HAVE
----------------------------------------
Measured across all 162 by AST, not guessed:

    162  cfg.outdir          55  cfg.returns_file    33  cfg.prices_file
     13  cfg.assets_file     12  cfg.nifty_file       9  cfg.stocks_file
     13  cfg.zscore_threshold ...

They read CSVs from `cfg.<name>_file`, write artefacts to `cfg.outdir`, and are
driven by argparse in `main()`. So they are runnable without modification if you
synthesise the config and the input files — which is what this does.

WHAT IS SYNTHESISED AND WHAT IS NOT
-----------------------------------
Input CSVs are materialised from the caller's REAL price series, reshaped into
the schema each script expects. Scalar parameters (`zscore_threshold`,
`window`, ...) are lifted from the script's own argparse defaults, so a run uses
the author's intended settings rather than numbers invented here.

Nothing predictive is fabricated. If a script needs `cfg.fundamentals_file` or
`cfg.events_file`, that is real external data: the run is refused with the
missing input named, never filled with plausible noise. That refusal is the
same rule as the FeedGuard — the point of this pipeline is to find out what is
true, and a strategy fed invented fundamentals would produce a confident lie.
"""

from __future__ import annotations

import ast
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Set

import numpy as np

log = logging.getLogger(__name__)

# Inputs this adapter can build from a price series: anything that is itself a
# time series of a price or level. Each gets its own distinct path from the
# correlated panel, so scripts execute rather than dying on a missing file.
#
# Read this carefully: a script fed a DERIVED series is being fed a real price
# path that is not the instrument it asked for. It will execute and produce a
# number. That number is not evidence — it is exactly why `DataSource.is_real`
# blocks these runs from writing verdicts. The purpose is to prove the machine
# executes at scale and to surface crashes, not to score strategies.
DERIVABLE_FILES = {
    "prices_file", "returns_file", "assets_file", "nifty_file", "stocks_file",
    "index_file", "benchmark_file", "equity_file", "spot_file", "futures_file",
    "close_file", "usdinr_file", "fx_file", "ivix_file", "vix_file",
    "gsec_file", "yield_file", "rates_file", "gas_file", "oil_file",
    "commodity_file", "freight_file", "metal_file", "data_file", "series_file",
    "bn_iv_file", "m1_file", "m2_file", "mcx_file", "etf_file", "adr_file",
}

# Inputs that are genuinely external: they carry information no price series
# implies. Refuse rather than invent — a strategy fed fabricated fundamentals or
# invented news sentiment produces a confident lie, which is the exact failure
# this whole pipeline exists to prevent.
EXTERNAL_FILES = {
    "fundamentals_file", "events_file", "options_file", "cpi_file",
    "macro_file", "flows_file", "oi_file", "iv_file", "news_file",
    "sentiment_file", "earnings_file", "holdings_file", "positions_file",
    "postings_file", "reviews_file", "trends_file", "ndvi_file", "rain_file",
    "jobs_file", "ports_file", "power_file", "weights_file", "changes_file",
    "ranks_file", "pcr_file", "margin_file", "shipping_file", "satellite_file",
    "social_file", "esg_file", "insider_file", "analyst_file", "credit_file",
}

# Any other `*_file` whose name matches a price/level-like instrument is
# derivable: the script gets a real price path from the panel. Names outside
# both sets are treated as external, because guessing wrong invents signal.
_PRICE_LIKE_PATTERN = re.compile(
    r"(price|close|spot|future|index|equity|stock|asset|return|nifty|sensex|"
    r"bank|vix|ivix|fx|usd|inr|eur|gold|silver|metal|oil|gas|crude|commodity|"
    r"yield|gsec|bond|rate|rbi|nse|bse|mcx|etf|adr|freight|series|data|"
    r"benchmark|m1|m2|bn|factor)",
    re.I,
)


@dataclass
class ScriptRequirements:
    """What one `run(cfg)` script needs before it can execute."""

    cfg_attrs: Set[str] = field(default_factory=set)
    file_attrs: Set[str] = field(default_factory=set)
    derivable: Set[str] = field(default_factory=set)
    external: Set[str] = field(default_factory=set)
    unknown_files: Set[str] = field(default_factory=set)
    defaults: Dict[str, Any] = field(default_factory=dict)

    @property
    def runnable(self) -> bool:
        return not self.external and not self.unknown_files

    def blocking_reason(self) -> str:
        bits = []
        if self.external:
            bits.append(f"needs external data: {sorted(self.external)}")
        if self.unknown_files:
            bits.append(f"unrecognised file inputs: {sorted(self.unknown_files)}")
        return "; ".join(bits)


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def inspect_script(path: Path) -> ScriptRequirements:
    """
    Statically determine a script's config contract.

    Scalar defaults come from the script's own `argparse` calls so a run uses
    the author's parameters, not ones chosen here.
    """
    req = ScriptRequirements()
    try:
        source = path.read_text(errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return req

    req.cfg_attrs = set(re.findall(r"cfg\.([a-zA-Z_][a-zA-Z0-9_]*)", source))
    req.file_attrs = {a for a in req.cfg_attrs if a.endswith("_file")}
    req.external = req.file_attrs & EXTERNAL_FILES
    remaining = req.file_attrs - EXTERNAL_FILES
    req.derivable = {
        a for a in remaining if a in DERIVABLE_FILES or _PRICE_LIKE_PATTERN.search(a)
    }
    # Anything neither explicitly external nor price-like is treated as
    # external: an unknown input is assumed to carry real information.
    req.external |= remaining - req.derivable
    req.unknown_files = set()

    # argparse defaults: ap.add_argument("--window", default=60, type=int)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        flag = next(
            (_literal(a) for a in node.args if isinstance(_literal(a), str)),
            None,
        )
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        name = flag[2:].replace("-", "_")
        for kw in node.keywords:
            if kw.arg == "default":
                val = _literal(kw.value)
                if val is not None:
                    req.defaults[name] = val
    return req


def _default_for(attr: str) -> Any:
    """
    A type-appropriate default for a config knob the script never documented.

    Inferred from the name because that is the only signal available. These are
    mechanical defaults (window lengths, thresholds), never anything that
    encodes a market view.
    """
    a = attr.lower()
    if a in ("plot", "verbose", "debug", "dry_run", "save", "show"):
        return False
    if any(k in a for k in ("window", "lookback", "period", "days", "n_", "epochs",
                            "lag", "horizon", "min_", "max_", "top", "bins")):
        return 20
    if any(k in a for k in ("threshold", "thresh", "alpha", "pct", "frac", "ratio",
                            "cost", "fee", "bps", "sigma", "z")):
        return 2.0 if ("z" in a or "sigma" in a) else 0.05
    if "start" in a:
        return "2020-01-01"
    if "end" in a:
        return "2024-12-31"
    if any(k in a for k in ("symbol", "ticker", "name", "asset")):
        return "SYNTH"
    return 0


class ScriptFixtures:
    """
    Materialises the CSV inputs a `run(cfg)` script expects, from real prices.

    Schemas mirror what the scripts parse: a `date` column plus either a long
    (date, ticker, close) or wide (date, <symbol>) layout, since both appear in
    the tree.
    """

    def __init__(self, prices: np.ndarray, symbol: str = "SYNTH", n_assets: int = 4):
        self.prices = np.asarray(prices, dtype=float)
        if self.prices.ndim != 1 or self.prices.size < 30:
            raise ValueError("need a 1-D price series of length >= 30")
        self.symbol = symbol
        self.n_assets = max(2, n_assets)
        self._dir = Path(tempfile.mkdtemp(prefix="gauntlet_fixtures_"))

    @property
    def outdir(self) -> Path:
        d = self._dir / "out"
        d.mkdir(exist_ok=True)
        return d

    def _dates(self):
        import pandas as pd

        return pd.bdate_range("2020-01-01", periods=self.prices.size)

    def _panel(self):
        """
        A small correlated panel derived from the one real series.

        Cross-sectional strategies need several names. Each extra column is the
        SAME real path with an independent idiosyncratic overlay — it carries no
        invented signal, and any cross-sectional 'edge' found on it is noise by
        construction. That is intentional: it lets the script execute so the
        gauntlet can measure it, and the gauntlet is what exposes the noise.
        """
        rng = np.random.default_rng(0)
        base = self.prices / self.prices[0]
        cols = {}
        for i in range(self.n_assets):
            if i == 0:
                cols[f"{self.symbol}"] = self.prices
            else:
                noise = np.cumprod(1.0 + rng.normal(0.0, 0.004, self.prices.size))
                cols[f"{self.symbol}{i}"] = base * noise * float(self.prices[0])
        return cols

    def build(self, attr: str) -> str:
        """Write the CSV for one `cfg.<attr>` and return its path."""
        import pandas as pd

        dates = self._dates()
        panel = self._panel()
        path = self._dir / f"{attr}.csv"

        if attr == "returns_file":
            df = pd.DataFrame(
                {k: pd.Series(v).pct_change().fillna(0.0).to_numpy() for k, v in panel.items()}
            )
            df.insert(0, "date", dates)
            # Long-form aliases: scripts variously expect `ret`/`return`/`r`.
            first = df.columns[1]
            for alias in ("ret", "return", "returns", "r"):
                df[alias] = df[first]
            df["ticker"] = self.symbol
        elif attr in ("prices_file", "assets_file", "stocks_file"):
            # Long format: date, ticker, close — what most scripts pivot on.
            rows = [
                pd.DataFrame({"date": dates, "ticker": tic, "close": series})
                for tic, series in panel.items()
            ]
            df = pd.concat(rows, ignore_index=True)
            df = self._add_column_aliases(df)
        else:
            # Wide single series, plus a ticker column so scripts that pivot on
            # one still work (KeyError: 'ticker' was the single most common crash).
            df = pd.DataFrame({"date": dates, "close": self.prices, "ticker": self.symbol})
            df = self._add_column_aliases(df)

        df.to_csv(path, index=False)
        return str(path)

    @staticmethod
    def _add_column_aliases(df):
        """
        Add the column names scripts actually reach for.

        Derived from the observed crash taxonomy (KeyError: 'ticker', 'price',
        'asset', 'contract', 'vix_close', ...). These are aliases of the same
        real series, not additional information.
        """
        close = df["close"]
        for alias in ("price", "px", "last", "value", "level", "settle", "adj_close"):
            df[alias] = close
        df["open"] = close
        df["high"] = close * 1.001
        df["low"] = close * 0.999
        df["volume"] = 1_000_000
        if "ticker" in df.columns:
            for alias in ("symbol", "asset", "contract", "name", "instrument"):
                df[alias] = df["ticker"]
        return df

    def make_cfg(self, req: ScriptRequirements) -> SimpleNamespace:
        """Build the `cfg` object, using the script's own argparse defaults."""
        values: Dict[str, Any] = dict(req.defaults)
        values["outdir"] = str(self.outdir)
        for attr in req.derivable:
            values[attr] = self.build(attr)
        # Anything referenced but still unset needs a TYPE-APPROPRIATE default.
        # Defaulting everything to None caused "'NoneType' cannot be interpreted
        # as an integer" — a knob left unset must still be usable.
        for attr in req.cfg_attrs:
            if attr in values:
                continue
            values[attr] = _default_for(attr)
        values["plot"] = False  # never open a window in a batch run
        values["outdir"] = str(self.outdir)
        return SimpleNamespace(**values)

    def read_returns(self) -> Optional[np.ndarray]:
        """
        Recover a return series from whatever the script wrote to outdir.

        Scripts are inconsistent about output names, so this looks for the
        conventional artefacts and takes the first usable numeric series.
        Returns None when nothing parseable was produced — which is a failed
        run, not a flat one.
        """
        import pandas as pd

        for name in ("backtest.csv", "pnl.csv", "returns.csv", "equity.csv", "signals.csv"):
            f = self.outdir / name
            if not f.exists():
                continue
            try:
                df = pd.read_csv(f)
            except Exception:  # noqa: BLE001 - arbitrary script output
                continue
            for col in ("ret", "returns", "pnl", "strategy_ret", "daily_ret", "r"):
                if col in df.columns:
                    arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
                    if arr.size >= 30:
                        return arr
            # equity curve -> returns
            for col in ("equity", "cum_pnl", "cumulative", "nav"):
                if col in df.columns:
                    eq = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
                    if eq.size >= 31 and np.all(np.isfinite(eq)) and np.all(eq > 0):
                        return np.diff(eq) / eq[:-1]
        return None

    def cleanup(self) -> None:
        import shutil

        shutil.rmtree(self._dir, ignore_errors=True)


def run_script_strategy(
    module: Any,
    path: Path,
    prices: np.ndarray,
    symbol: str = "SYNTH",
) -> np.ndarray:
    """
    Execute one `run(cfg)` strategy and return its per-period returns.

    Raises with a specific reason on any failure. A script that could not run
    has produced nothing, which is materially different from a strategy that
    ran and chose to stay flat.
    """
    req = inspect_script(path)
    if not req.runnable:
        raise RuntimeError(f"{path.stem}: {req.blocking_reason()}")

    run_fn = getattr(module, "run", None)
    if run_fn is None:
        raise RuntimeError(f"{path.stem}: no run() after import")

    fixtures = ScriptFixtures(prices, symbol=symbol)
    try:
        cfg = fixtures.make_cfg(req)
        try:
            run_fn(cfg)
        except SystemExit as exc:
            # These are CLI scripts: several call sys.exit() when an input looks
            # wrong. SystemExit inherits BaseException, so a bare `except
            # Exception` lets it escape and kill the entire batch over one file.
            raise RuntimeError(f"{path.stem}: run() called sys.exit({exc.code})") from exc
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - arbitrary script bodies
            raise RuntimeError(f"{path.stem}: run() raised {type(exc).__name__}: {exc}") from exc

        rets = fixtures.read_returns()
        if rets is None:
            produced = sorted(p.name for p in fixtures.outdir.glob("*"))
            raise RuntimeError(
                f"{path.stem}: run() completed but wrote no parseable return series "
                f"(outdir contained: {produced or 'nothing'})"
            )
        return rets
    finally:
        fixtures.cleanup()
