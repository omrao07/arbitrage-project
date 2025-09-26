#!/usr/bin/env python3
# orchestrator/live_job.py
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import pandas as pd

# ---- Repo paths ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / "runtime" / "state"
LOG_DIR = REPO_ROOT / "runtime" / "logs"
CACHE_DIR = REPO_ROOT / "data" / "cache"

# ---- Router (reuse your CDS router; others can plug in similarly) ------------
try:
    from engines.credit_cds.order_router import ( # type: ignore
        build_default_router, OrderRouter, RouteResult
    )
except Exception:  # optional import so equity-only books still run
    OrderRouter = object  # type: ignore
    RouteResult = object  # type: ignore
    def build_default_router():  # type: ignore
        raise RuntimeError("Order router not available; install credit_cds module.")

# ---- Logging -----------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "live_job.log", mode="a")
    ],
)
log = logging.getLogger("live_job")

# =============================================================================
# Config models
# =============================================================================

@dataclass
class LiveJobConfig:
    # General
    sid: str = "DEMO-0000"                    # strategy id
    mode: str = "paper"                       # 'paper' or 'live'
    poll_secs: int = 15                       # main loop sleep interval
    rebalance_secs: int = 60                  # how often to reconsider trades
    tz: str = "UTC"
    venue: str = "MOCK"                       # router venue key
    # Risk limits
    max_gross_usd: float = 25_000_000.0
    max_name_usd: float = 5_000_000.0
    max_daily_turnover_usd: float = 10_000_000.0
    allow_short: bool = True
    # Files
    state_path: Path = STATE_DIR / "live_state.json"
    fills_path: Path = STATE_DIR / "fills.csv"
    pnl_path: Path = STATE_DIR / "pnl.csv"
    # Adapters (callables / import strings)
    adapter_path: str = "orchestrator.liveadapters.generic:Adapter"  # class with required interface
    # Optional extra knobs passed to the adapter
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)

# Persistent portfolio state (simple & robust)
@dataclass
class PortfolioState:
    ts_epoch: int = 0
    nav_usd: float = 1_000_000.0
    cash_usd: float = 1_000_000.0
    # current signed notionals per symbol (e.g., CDS protection, shares, contracts)
    positions: Dict[str, float] = field(default_factory=dict)
    # rolling turnover for the day
    traded_today_usd: float = 0.0
    # book PnL components
    pnl_day_usd: float = 0.0
    pnl_cum_usd: float = 0.0
    # book-keeping
    last_rebalance_ts: int = 0

# =============================================================================
# Adapter interface (strategies plug in here)
# =============================================================================
"""
Your Adapter must provide:

class Adapter:
    def __init__(self, **kwargs): ...
    def warmup(self) -> None:
        # Fetch/prepare data; called once on start
    def generate_signals(self, now_ts: int) -> Dict[str, Any]:
        # Return arbitrary signal diagnostics (logged)
    def propose_trades(self, state: PortfolioState, now_ts: int) -> pd.DataFrame:
        # Return DataFrame with columns:
        # ['ticker','trade_notional','side','tenor_years','currency','client_order_id']
        # 'side' is a domain-specific string (e.g., BUY_PROTECTION/SELL_PROTECTION or BUY/SELL)
    def mark_to_market(self, state: PortfolioState, now_ts: int) -> Dict[str, float]:
        # Return dict with 'pnl_usd' incremental, plus optional components
"""

def load_adapter(adapter_path: str, **kwargs) -> Any:
    if ":" in adapter_path:
        mod, cls = adapter_path.split(":", 1)
    else:
        *parts, cls = adapter_path.split(".")
        mod = ".".join(parts)
    mod_obj = __import__(mod, fromlist=[cls])
    klass = getattr(mod_obj, cls)
    return klass(**kwargs)

# =============================================================================
# Live Job
# =============================================================================

class LiveJob:
    def __init__(self, cfg: LiveJobConfig):
        self.cfg = cfg
        self.state = PortfolioState()
        self.router: Optional[OrderRouter] = None # type: ignore
        self.adapter = load_adapter(cfg.adapter_path, **(cfg.adapter_kwargs or {}))
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        # attach signal handlers
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)
        self._stop = False

    # ---- lifecycle -----------------------------------------------------------

    def start(self):
        log.info(f"Starting live job for SID={self.cfg.sid} mode={self.cfg.mode}")
        self._load_state()
        self._init_router()
        self.adapter.warmup()

        last_loop = 0
        while not self._stop:
            now = int(time.time())
            # throttle
            if now - last_loop < max(1, self.cfg.poll_secs):
                time.sleep(0.25)
                continue
            last_loop = now

            # 1) Mark-to-market & record PnL
            mtm = self.adapter.mark_to_market(self.state, now)
            inc = float(mtm.get("pnl_usd", 0.0))
            self.state.pnl_day_usd += inc
            self.state.pnl_cum_usd += inc
            self.state.ts_epoch = now
            self._append_pnl(now, inc, mtm)

            # 2) Signals (optional diagnostics)
            try:
                sigs = self.adapter.generate_signals(now)
                if sigs:
                    log.debug(f"signals: {sigs}")
            except Exception as e:
                log.warning(f"generate_signals failed: {e}")

            # 3) Rebalance?
            if now - self.state.last_rebalance_ts >= self.cfg.rebalance_secs:
                self._rebalance(now)

            # 4) Persist lightweight state
            self._save_state()

        log.info("Stopped.")

    # ---- core steps ----------------------------------------------------------

    def _rebalance(self, now: int):
        log.info("Rebalance tick")
        # Ask strategy for proposed trades
        try:
            proposed = self.adapter.propose_trades(self.state, now)
        except Exception as e:
            log.exception(f"propose_trades failed: {e}")
            return
        if proposed is None or len(proposed) == 0:
            self.state.last_rebalance_ts = now
            log.info("No trades proposed.")
            return

        # Risk checks (pre-trade)
        proposed = self._pretrade_filter(proposed)

        # Paper vs live execution
        if self.cfg.mode == "paper":
            fills = self._paper_fill(proposed)
        else:
            fills = self._route_orders(proposed)

        # Apply fills to state
        self._apply_fills(fills)
        self._append_fills(fills)
        self.state.last_rebalance_ts = now

    def _pretrade_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        req = {"ticker", "trade_notional", "side"}
        if not req.issubset(df.columns):
            raise ValueError(f"propose_trades must contain {req}")
        df = df.copy()
        # Cap per-name
        capped: List[int] = []
        for i, row in df.iterrows():
            name = str(row["ticker"]).split("_")[0]
            cur = float(self.state.positions.get(name, 0.0))
            signed = float(row["trade_notional"]) if "BUY" in row["side"] else -float(row["trade_notional"])
            if abs(cur + signed) > self.cfg.max_name_usd + 1e-6:
                capped.append(i) # type: ignore
        if capped:
            log.warning(f"Per-name caps triggered; dropping {len(capped)} orders.")
            df = df.drop(index=capped)

        # Cap daily turnover
        gross_add = float(df["trade_notional"].abs().sum())
        if self.state.traded_today_usd + gross_add > self.cfg.max_daily_turnover_usd:
            allow = max(0.0, self.cfg.max_daily_turnover_usd - self.state.traded_today_usd)
            if allow <= 0:
                log.warning("Daily turnover limit reached; no trades sent.")
                return df.iloc[0:0]
            # scale down proportionally
            scale = allow / max(1e-9, gross_add)
            df["trade_notional"] = df["trade_notional"] * scale
            log.warning(f"Turnover scaled by {scale:.2%} to fit daily cap.")
        return df

    def _paper_fill(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        # Instant fills at requested size; you can add slippage model here
        fills: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            fills.append({
                "parent_id": str(r.get("client_order_id", f"paper-{int(time.time()*1000)}")),
                "ticker": str(r["ticker"]),
                "side": str(r["side"]),
                "filled_usd": float(r["trade_notional"]),
                "avg_price_bps": float(r.get("px_hint_bps", 0.0)),
                "status": "FILLED",
                "ts": int(time.time()),
            })
        log.info(f"[paper] filled {len(fills)} orders.")
        return fills

    def _route_orders(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if self.router is None:
            raise RuntimeError("Router not initialized.")
        results = self.router.route_from_df(df, venue_name=self.cfg.venue, px_hint_bps=float(df.get("px_hint_bps", pd.Series([0])).iloc[-1]))
        fills: List[Dict[str, Any]] = []
        for res in results:
            if not isinstance(res, RouteResult):
                continue
            fills.append({
                "parent_id": res.parent_id,
                "ticker": getattr(res.children[0], "ticker", "NA") if res.children else "NA",
                "side": getattr(res.children[0], "side", "NA") if res.children else "NA",
                "filled_usd": float(res.filled_usd),
                "avg_price_bps": float(res.avg_price_bps) if res.avg_price_bps is not None else None,
                "status": res.status,
                "ts": int(time.time()),
            })
        return fills

    def _apply_fills(self, fills: List[Dict[str, Any]]):
        for f in fills:
            if f["status"] not in ("FILLED", "PARTIAL"):
                continue
            signed = f["filled_usd"] if "BUY" in f["side"] else -f["filled_usd"]
            name = f["ticker"].split("_")[0]
            self.state.positions[name] = float(self.state.positions.get(name, 0.0) + signed)
            self.state.traded_today_usd += abs(f["filled_usd"])

    # ---- persistence ---------------------------------------------------------

    def _init_router(self):
        try:
            self.router = build_default_router()
            # sync limits from cfg
            lim = self.router.limits # type: ignore
            lim.max_gross_notional_usd = self.cfg.max_gross_usd
            lim.max_notional_per_name_usd = self.cfg.max_name_usd
            lim.max_daily_trade_usd = self.cfg.max_daily_turnover_usd
            lim.allow_short_protection = self.cfg.allow_short
            log.info("Router initialized.")
        except Exception as e:
            if self.cfg.mode == "live":
                raise
            log.warning(f"No router available; running paper mode only. ({e})")

    def _save_state(self):
        try:
            d = asdict(self.state)
            self.cfg.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.state_path.write_text(json.dumps(d))
        except Exception as e:
            log.warning(f"state save failed: {e}")

    def _load_state(self):
        try:
            if self.cfg.state_path.exists():
                d = json.loads(self.cfg.state_path.read_text())
                self.state = PortfolioState(**d)
                log.info(f"Loaded state from {self.cfg.state_path}")
        except Exception as e:
            log.warning(f"state load failed: {e}")

    def _append_fills(self, fills: List[Dict[str, Any]]):
        if not fills:
            return
        df = pd.DataFrame(fills)
        self.cfg.fills_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg.fills_path.exists():
            old = pd.read_csv(self.cfg.fills_path)
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(self.cfg.fills_path, index=False)

    def _append_pnl(self, now: int, inc: float, components: Dict[str, float]):
        row = {"ts": now, "pnl_inc$": inc, **components}
        self.cfg.pnl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg.pnl_path.exists():
            pd.concat([pd.read_csv(self.cfg.pnl_path), pd.DataFrame([row])]).to_csv(self.cfg.pnl_path, index=False)
        else:
            pd.DataFrame([row]).to_csv(self.cfg.pnl_path, index=False)

    # ---- shutdown ------------------------------------------------------------

    def _graceful_shutdown(self, *_):
        log.info("Signal received; shutting down...")
        self._stop = True

# =============================================================================
# CLI
# =============================================================================

def parse_args() -> LiveJobConfig:
    p = argparse.ArgumentParser(description="Live trading loop")
    p.add_argument("--sid", default="DEMO-0000")
    p.add_argument("--mode", default="paper", choices=["paper", "live"])
    p.add_argument("--poll", type=int, default=15)
    p.add_argument("--rebalance", type=int, default=60)
    p.add_argument("--venue", default="MOCK")
    p.add_argument("--adapter", default="orchestrator.liveadapters.generic:Adapter")
    p.add_argument("--adapter-kwargs", default="{}",
                   help="JSON dict passed to adapter constructor")
    p.add_argument("--state", default=str(STATE_DIR / "live_state.json"))
    p.add_argument("--fills", default=str(STATE_DIR / "fills.csv"))
    p.add_argument("--pnl", default=str(STATE_DIR / "pnl.csv"))
    p.add_argument("--max-gross", type=float, default=25_000_000)
    p.add_argument("--max-name", type=float, default=5_000_000)
    p.add_argument("--max-turnover", type=float, default=10_000_000)
    p.add_argument("--allow-short", action="store_true")
    args = p.parse_args()

    cfg = LiveJobConfig(
        sid=args.sid,
        mode=args.mode,
        poll_secs=args.poll,
        rebalance_secs=args.rebalance,
        venue=args.venue,
        adapter_path=args.adapter,
        adapter_kwargs=json.loads(args.adapter_kwargs),
        state_path=Path(args.state),
        fills_path=Path(args.fills),
        pnl_path=Path(args.pnl),
        max_gross_usd=args.max_gross,
        max_name_usd=args.max_name,
        max_daily_turnover_usd=args.max_turnover,
        allow_short=args.allow_short,
    )
    return cfg

def main():
    cfg = parse_args()
    job = LiveJob(cfg)
    job.start()

if __name__ == "__main__":
    main()