# backend/engine/manager.py
from __future__ import annotations

import os
import sys
import time
import json
import signal
import queue
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

# ---- logging ----
log = logging.getLogger("manager")
if not log.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# ---- bus helpers (your existing wrappers) ----
from backend.bus.streams import (
    consume_stream,
    publish_stream,
    hset,
)

# ---- core components ----
from backend.engine.strategy_base import Strategy, ExampleBuyTheDip
from backend.engine.risk_manager import RiskManager # type: ignore
from backend.execution.broker_base import BrokerBase, PaperBroker, Order, Side, OrderType, TIF # type: ignore

# Optional strategies (import if present)
try:
    from backend.engine.strategies.market_maker import MarketMakerStrategy  # type: ignore # noqa: F401
except Exception:
    MarketMakerStrategy = None  # type: ignore
try:
    from backend.engine.strategies.predictor_strategy import PredictorStrategy  # type: ignore # noqa: F401
except Exception:
    PredictorStrategy = None  # type: ignore

# -------------------- env / streams --------------------
REDIS_TICKS_STREAM   = os.getenv("TICKS_STREAM", "ticks")                 # normalized ticks: {"symbol","price","ts_ms",...}
REDIS_NEWS_STREAM    = os.getenv("NEWS_STREAM", "news")                   # NewsEvent dicts (symbol, score, headline,...)
ORDERS_IN            = os.getenv("RISK_INCOMING_STREAM", "orders.incoming")
ORDERS_OK            = os.getenv("RISK_APPROVED_STREAM", "orders.approved")
ORDERS_NOK           = os.getenv("RISK_REJECTED_STREAM", "orders.rejected")
FILLS_STREAM         = os.getenv("FILLS_STREAM", "fills")                 # if your OMS/broker publishes fills here
PRICES_HASH          = os.getenv("PRICES_HASH", "px:last")                # HSET symbol -> last
HEALTH_KEY           = os.getenv("HEALTH_HEARTBEAT_KEY", "engine:hb")     # HSET keys updated by manager

BROKER_NAME          = os.getenv("BROKER", "paper")                       # "paper" | "zerodha" | "ibkr" (your adapters)
BASE_CCY             = os.getenv("BASE_CCY", "USD")

# -------------------- small utils --------------------
def now_ms() -> int:
    return int(time.time() * 1000)

def safe_json(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except Exception:
        return {}

@dataclass
class ManagerConfig:
    symbols: List[str] = None  # type: ignore # subscribe/filter list (optional)
    region: Optional[str] = None
    mode: str = "paper"        # "paper" | "live"
    throttle_ms_ticks: int = 0 # per-symbol throttle (0 = off)
    throttle_ms_news: int = 0

# -------------------- Manager --------------------
class EngineManager:
    """
    Orchestrates:
      • Tick & News ingestion from Redis Streams
      • Dispatch to strategies (on_tick / on_news)
      • RiskManager gate on orders.incoming -> orders.approved/rejected
      • OMS: ship approved orders to broker adapter and publish fills
      • Heartbeats & lightweight metrics (HSET)
    """

    def __init__(
        self,
        *,
        cfg: ManagerConfig,
        strategies: List[Strategy],
        risk: Optional[RiskManager] = None,
        broker: Optional[BrokerBase] = None,
    ):
        self.cfg = cfg
        self.strategies = strategies
        self.risk = risk or RiskManager()
        self.broker = broker or PaperBroker(currency=BASE_CCY)
        self._stop = threading.Event()

        # local state
        self._last_tick_ts: Dict[str, int] = {}
        self._last_news_ts: int = 0

        # queues (optional: used by broker worker)
        self._approved_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=10_000)

        # threads
        self.threads: List[threading.Thread] = []

    # ------------- lifecycle -------------
    def start(self) -> None:
        log.info("Connecting broker ...")
        self.broker.connect()

        log.info("Starting threads ...")
        self._spawn(self._tick_loop, name="tick_loop")
        self._spawn(self._news_loop, name="news_loop")
        self._spawn(self._risk_loop, name="risk_loop")
        self._spawn(self._oms_loop, name="oms_loop")

        self._install_signals()
        log.info("EngineManager started.")

    def _install_signals(self) -> None:
        def _sig(_s, _f):
            log.warning("Signal received; shutting down ...")
            self.stop()
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        # give threads a moment to exit
        for t in self.threads:
            t.join(timeout=1.5)
        try:
            self.broker.close()
        except Exception:
            pass
        log.info("EngineManager stopped.")

    def _spawn(self, target: Callable, *, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self.threads.append(t)

    # ------------- loops -----------------
    def _tick_loop(self) -> None:
        """
        Consume ticks from Redis, update last prices, call strategies.on_tick().
        Expect stream entries shaped like {"symbol": "...", "price": 123.45, "ts_ms": ...}
        """
        log.info(f"[tick_loop] consuming '{REDIS_TICKS_STREAM}'")
        for _id, msg in consume_stream(REDIS_TICKS_STREAM, start_id="$", block_ms=1000, count=500):
            if self._stop.is_set():
                break
            tick = safe_json(msg)
            sym = (tick.get("symbol") or tick.get("s") or "").upper()
            px = float(tick.get("price") or tick.get("p") or 0.0)
            if not sym or px <= 0:
                continue

            # optional symbol filter
            if self.cfg.symbols and sym not in self.cfg.symbols:
                continue

            # throttle per symbol if configured
            if self.cfg.throttle_ms_ticks:
                last_ts = self._last_tick_ts.get(sym, 0)
                if (now_ms() - last_ts) < self.cfg.throttle_ms_ticks:
                    continue
                self._last_tick_ts[sym] = now_ms()

            # update last price for risk / dashboards
            hset(PRICES_HASH, sym, px)

            # forward last to paper broker so IOC/LIMIT can simulate
            try:
                if isinstance(self.broker, PaperBroker):
                    self.broker.set_price(sym, px)
            except Exception:
                pass

            # call strategies
            for s in self.strategies:
                try:
                    s.on_tick(tick)
                except Exception as e:
                    log.exception("strategy on_tick error: %s", e)

            # heartbeat
            hset(HEALTH_KEY, "last_tick_ms", now_ms())

    def _news_loop(self) -> None:
        """
        Consume news stream and call strategies.on_news if present.
        News event example:
          {"source":"yahoo","symbol":"RELIANCE.NS","score":0.42,"headline":"...","url":"...", "published_at": 169...}
        """
        log.info(f"[news_loop] consuming '{REDIS_NEWS_STREAM}'")
        for _id, msg in consume_stream(REDIS_NEWS_STREAM, start_id="$", block_ms=1500, count=200):
            if self._stop.is_set():
                break
            ev = safe_json(msg)

            # throttle global news dispatch
            if self.cfg.throttle_ms_news:
                if (now_ms() - self._last_news_ts) < self.cfg.throttle_ms_news:
                    continue
                self._last_news_ts = now_ms()

            for s in self.strategies:
                try:
                    if hasattr(s, "on_news"):
                        # type: ignore[attr-defined]
                        s.on_news(ev)  # type: ignore # optional hook
                except Exception as e:
                    log.exception("strategy on_news error: %s", e)

            hset(HEALTH_KEY, "last_news_ms", now_ms())

    def _risk_loop(self) -> None:
        """
        Consume orders.incoming -> validate via RiskManager -> publish approved/rejected.
        NOTE: If you already run RiskManager in its own process, you can disable this loop.
        """
        log.info(f"[risk_loop] consuming '{ORDERS_IN}', producing OK/NOK")
        for _id, order in consume_stream(ORDERS_IN, start_id="$", block_ms=1000, count=200):
            if self._stop.is_set():
                break
            try:
                ok, reason, adj = self.risk.validate(order)
                if ok:
                    publish_stream(ORDERS_OK, dict(adj, risk="pass", reason=reason, ts_ms=now_ms()))
                    # also push to local queue for OMS worker
                    try:
                        self._approved_q.put_nowait(adj)
                    except queue.Full:
                        log.warning("approved queue full, dropping order")
                else:
                    publish_stream(ORDERS_NOK, dict(order, risk="fail", reason=reason, ts_ms=now_ms()))
            except Exception as e:
                publish_stream(ORDERS_NOK, dict(order, risk="error", reason=str(e), ts_ms=now_ms()))
            hset(HEALTH_KEY, "last_risk_ms", now_ms())

    def _oms_loop(self) -> None:
        """
        Pull approved orders and send to broker adapter. Publish lightweight fill echoes if available.
        """
        log.info("[oms_loop] waiting for approved orders")
        while not self._stop.is_set():
            try:
                order = self._approved_q.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                sym = order["symbol"]
                side = Side.BUY if str(order.get("side","")).lower() == "buy" else Side.SELL
                qty  = float(order.get("qty", 0.0))
                typ  = OrderType.LIMIT if str(order.get("typ","")).lower() == "limit" else OrderType.MARKET
                limit= order.get("limit_price")
                tif  = TIF.IOC if str(order.get("tif","")).lower() == "ioc" else TIF.DAY
                coid = order.get("client_order_id")
                strat= order.get("strategy")

                ack = self.broker.place_order(Order(
                    symbol=sym, side=side, qty=qty,
                    order_type=typ, limit_price=limit, tif=tif,
                    client_order_id=coid, strategy=strat, meta={"source":"oms_loop"}
                ))
                # Publish a minimal echo so TCA/UI can react instantly
                publish_stream(
                    "orders.acks",
                    {"ts_ms": now_ms(), "symbol": sym, "ok": ack.ok, "order_id": ack.order_id, "reason": ack.reason, "strategy": strat}
                )

                # If the paper broker includes a fill in ack.raw, emit it to FILLS_STREAM
                raw = ack.raw or {}
                fill = raw.get("fill")
                if fill:
                    publish_stream(FILLS_STREAM, fill)

            except Exception as e:
                log.exception("broker.place_order error: %s", e)
            hset(HEALTH_KEY, "last_oms_ms", now_ms())

# -------------------- bootstrap --------------------
def default_strategies(region: Optional[str] = None) -> List[Strategy]:
    """
    Build a small default set so manager runs out-of-the-box.
    Replace with your registry if you have one.
    """
    strats: List[Strategy] = [ExampleBuyTheDip(region=region, default_qty=1.0, bps=10.0)]
    # Optionally include market maker / predictor if available
    if MarketMakerStrategy:
        try:
            strats.append(MarketMakerStrategy(symbols=["RELIANCE.NS"], region=region, default_qty=1.0, min_tick=0.05))
        except Exception:
            pass
    if PredictorStrategy:
        try:
            # you would load your trained predictor artifact here and pass it in
            # from backend.alpha.predictors.predictor_base import PricePredictorBase
            # pred = PricePredictorBase.load("backend/alpha/artifacts/price_predictor.joblib")
            # strats.append(PredictorStrategy(predictor=pred, region=region, default_qty=1.0))
            pass
        except Exception:
            pass
    return strats

def make_broker() -> BrokerBase:
    # Plug real adapters here later (Zerodha/IBKR)
    name = (BROKER_NAME or "paper").lower()
    if name == "paper":
        return PaperBroker(currency=BASE_CCY, start_cash=1_000_000.0)
    # from backend.execution.zerodha_broker import ZerodhaBroker
    # from backend.execution.ibkr_broker import IBKRBroker
    # if name == "zerodha": return ZerodhaBroker(...)
    # if name == "ibkr": return IBKRBroker(...)
    return PaperBroker(currency=BASE_CCY)

def main():
    cfg = ManagerConfig(
        symbols=None,                 # or e.g., ["RELIANCE.NS","AAPL"] # type: ignore
        region=os.getenv("REGION"),   # "india" / "us" / ...
        mode=os.getenv("MODE", "paper"),
        throttle_ms_ticks=int(os.getenv("THROTTLE_MS_TICKS", "0")),
        throttle_ms_news=int(os.getenv("THROTTLE_MS_NEWS", "0")),
    )
    mgr = EngineManager(
        cfg=cfg,
        strategies=default_strategies(region=cfg.region),
        risk=RiskManager(),
        broker=make_broker(),
    )
    mgr.start()

    # Keep foreground alive until signal
    try:
        while True:
            time.sleep(1.0)
            # heartbeat
            hset(HEALTH_KEY, "manager_alive_ms", now_ms())
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()

if __name__ == "__main__":
    main()