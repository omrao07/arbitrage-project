# backend/pipelines.py
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# ---- logging -------------------------------------------------------------
log = logging.getLogger("pipelines")
if not log.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# ---- bus / stores (your existing modules) --------------------------------
# These imports are optional; keep them inside functions if your tree differs.
try:
    from backend.bus.streams import publish_stream, consume_stream, hset
except Exception:
    publish_stream = lambda *a, **k: None   # type: ignore
    consume_stream = lambda *a, **k: []     # type: ignore
    hset = lambda *a, **k: None             # type: ignore

# Optional components
# from backend.ingestion.news.news_yahoo import YahooNews
# from backend.ingestion.news.news_moneycontrol import MoneycontrolNews
# from backend.ingestion.news.news_base import NewsEvent
# from backend.analytics.sentiment_ai import SentimentModel
# from backend.execution.order_store import OrderStore
# from backend.alpha.predictors.predictor_base import PricePredictorBase
# from backend.alpha.predictors.price_predictor import RidgePricePredictor

# =============================================================================
#                             CORE PIPELINE PRIMITIVES
# =============================================================================

@dataclass
class Context:
    """Mutable context passed across tasks in a pipeline run."""
    vars: Dict[str, Any]

    def get(self, k: str, default=None):
        return self.vars.get(k, default)

    def set(self, k: str, v: Any) -> None:
        self.vars[k] = v


@dataclass
class Task:
    name: str
    fn: Callable[[Context], Any]
    retries: int = 1
    backoff_sec: float = 0.5

    def run(self, ctx: Context) -> Any:
        last_err = None
        for i in range(self.retries + 1):
            try:
                return self.fn(ctx)
            except Exception as e:
                last_err = e
                if i >= self.retries:
                    break
                wait = self.backoff_sec * (2 ** i)
                log.warning("Task %s failed (%s). retrying in %.2fs...", self.name, e, wait)
                time.sleep(wait)
        # if here: failed
        trace = traceback.format_exc(limit=8)
        raise RuntimeError(f"Task '{self.name}' failed after {self.retries+1} attempts: {last_err}\n{trace}") from last_err


class Pipeline:
    """
    Minimal pipeline runner.
    - Add tasks (functions taking & returning via Context)
    - run_once(ctx) or run_forever(interval_sec)
    """
    def __init__(self, name: str):
        self.name = name
        self.tasks: List[Task] = []

    def add(self, name: str, fn: Callable[[Context], Any], *, retries: int = 1, backoff_sec: float = 0.5) -> "Pipeline":
        self.tasks.append(Task(name=name, fn=fn, retries=retries, backoff_sec=backoff_sec))
        return self

    def run_once(self, initial_vars: Optional[Dict[str, Any]] = None) -> Context:
        ctx = Context(vars=dict(initial_vars or {}))
        hset("pipeline:hb", self.name, int(time.time() * 1000))
        for t in self.tasks:
            t.run(ctx)
            hset(f"pipeline:last:{self.name}", t.name, int(time.time() * 1000))
        return ctx

    def run_forever(self, interval_sec: float, initial_vars: Optional[Dict[str, Any]] = None) -> None:
        log.info("Starting pipeline '%s' with interval=%.2fs", self.name, interval_sec)
        while True:
            t0 = time.time()
            try:
                self.run_once(initial_vars)
            except Exception as e:
                log.error("Pipeline '%s' run failed: %s", self.name, e)
            # sleep remainder
            dt = time.time() - t0
            sleep = max(0.0, float(interval_sec) - dt)
            time.sleep(sleep)

# =============================================================================
#                           REUSABLE TASK BUILDERS
# =============================================================================

# ---------- NEWS INGESTION + SENTIMENT ENRICHMENT -----------------------------

def task_fetch_yahoo(limit: int = 50):
    def _fn(ctx: Context):
        from backend.ingestion.news.news_yahoo import YahooNews
        items = YahooNews().fetch(limit=limit) # type: ignore
        ctx.set("news_items", items)
        log.info("yahoo: fetched %d items", len(items))
    return _fn

def task_fetch_moneycontrol(limit: int = 50):
    def _fn(ctx: Context):
        from backend.ingestion.news.news_moneycontrol import MoneycontrolNews # type: ignore
        items = MoneycontrolNews().fetch(limit=limit)
        prev = ctx.get("news_items", [])
        ctx.set("news_items", (prev + items)) # type: ignore
        log.info("moneycontrol: fetched %d items (total now %d)", len(items), len(prev) + len(items)) # type: ignore
    return _fn

def task_dedupe_news(key: str = "headline"):
    def _fn(ctx: Context):
        items = ctx.get("news_items", []) or []
        seen = set()
        out = []
        for it in items:
            k = (it.get(key) or "").strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(it)
        ctx.set("news_items", out)
        log.info("dedupe: %d -> %d", len(items), len(out))
    return _fn

def task_sentiment():
    def _fn(ctx: Context):
        from backend.analytics.sentiment_ai import SentimentModel # type: ignore
        sm = ctx.get("sent_model")
        if sm is None:
            sm = SentimentModel.load_or_default()
            ctx.set("sent_model", sm)
        items = ctx.get("news_items", []) or []
        for it in items:
            if it.get("score") is None:
                it["score"] = float(sm.score(it.get("headline","")))
        ctx.set("news_items", items)
        log.info("sentiment: scored %d items", len(items))
    return _fn

def task_publish_news(stream: str = os.getenv("NEWS_STREAM", "news")):
    def _fn(ctx: Context):
        items = ctx.get("news_items", []) or []
        for it in items:
            publish_stream(stream, it)
        log.info("published %d news to stream '%s'", len(items), stream)
    return _fn

# ---------- TICKS → LAST PRICE CACHE -----------------------------------------

def task_update_last_prices(stream_in: str = os.getenv("TICKS_STREAM", "ticks"),
                            hash_key: str = os.getenv("PRICES_HASH", "px:last")):
    """
    Pulls a batch of ticks and updates last-price hash (for risk/OMS/paper broker).
    Use run_forever with a small interval (e.g., 0.5s).
    """
    def _fn(ctx: Context):
        cnt = 0
        for _id, tick in consume_stream(stream_in, start_id="$", block_ms=250, count=500):
            sym = (tick.get("symbol") or tick.get("s") or "").upper()
            px  = float(tick.get("price") or tick.get("p") or 0.0)
            if sym and px > 0:
                hset(hash_key, sym, px)
                cnt += 1
        if cnt:
            log.info("updated %d last prices into '%s'", cnt, hash_key)
    return _fn

# ---------- ORDER/FILL CAPTURE TO SQLITE -------------------------------------

def task_capture_acks_and_fills(db_path: str = "runtime/order_store.db"):
    """
    Consume 'orders.acks' and 'fills' streams and persist to SQLite OrderStore.
    """
    def _fn(ctx: Context):
        from backend.execution.order_store import OrderStore, AckIn, FillIn # type: ignore
        store = ctx.get("order_store")
        if store is None:
            store = OrderStore(db_path=db_path)
            ctx.set("order_store", store)

        # acks
        for _id, ack in consume_stream("orders.acks", start_id="$", block_ms=200, count=200):
            a = AckIn(
                client_order_id=ack.get("client_order_id"),
                broker_order_id=ack.get("order_id"),
                ok=bool(ack.get("ok", False)),
                reason=ack.get("reason"),
                ts_ms=int(ack.get("ts_ms") or int(time.time()*1000)),
            )
            store.record_ack(a)

        # fills
        for _id, fill in consume_stream("fills", start_id="$", block_ms=200, count=200):
            f = FillIn(
                order_id=fill.get("order_id_int"),
                order_coid=fill.get("client_order_id"),
                order_oid=fill.get("order_id"),
                symbol=str(fill.get("symbol")),
                side=str(fill.get("side")).lower(),
                qty=float(fill.get("qty", 0.0)),
                price=float(fill.get("price", 0.0)),
                strategy=fill.get("strategy"),
                ts_ms=int(fill.get("ts_ms") or int(time.time()*1000)),
                raw=fill,
            )
            store.record_fill(f)

        log.debug("capture pass done")
    return _fn

# ---------- MODEL TRAINING / REFRESH -----------------------------------------

def task_train_price_predictor(symbol: str, csv_path: str, out_path: str,
                               alpha: float = 1.0):
    def _fn(ctx: Context):
        from backend.cli.train_price_predictor import load_csv # type: ignore
        from backend.alpha.predictors.price_predictor import RidgePricePredictor # type: ignore
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = load_csv(csv_path)
        model = RidgePricePredictor(alpha=alpha)
        model.fit_symbol(symbol, df)
        model.save(out_path)
        ctx.set("predictor_path", out_path)
        log.info("trained predictor for %s -> %s", symbol, out_path)
    return _fn

# ---------- BACKTESTER HOOK (SHELL) ------------------------------------------

def task_backtest(strategy_name: str, start: str, end: str, symbols: List[str]):
    """
    Shell task to call your backtester (you can wire in later).
    """
    def _fn(ctx: Context):
        log.info("[backtest] %s %s→%s on %s", strategy_name, start, end, symbols)
        # TODO: import and run your real backtester here.
        ctx.set("backtest_result", {"strategy": strategy_name, "start": start, "end": end, "symbols": symbols})
    return _fn

# =============================================================================
#                             READY-MADE PIPELINES
# =============================================================================

def make_news_pipeline(name: str = "news_ingest", *, limit: int = 50, with_sentiment: bool = True) -> Pipeline:
    """
    Yahoo + Moneycontrol → dedupe → (sentiment) → publish to Redis 'news'
    """
    p = Pipeline(name)
    p.add("fetch_yahoo", task_fetch_yahoo(limit=limit), retries=2, backoff_sec=0.5)
    p.add("fetch_moneycontrol", task_fetch_moneycontrol(limit=limit), retries=2, backoff_sec=0.5)
    p.add("dedupe", task_dedupe_news(), retries=0)
    if with_sentiment:
        p.add("sentiment", task_sentiment(), retries=1)
    p.add("publish_news", task_publish_news(), retries=1)
    return p

def make_tick_cache_pipeline(name: str = "tick_cache", *, stream_in: Optional[str] = None) -> Pipeline:
    """
    Reads TICKS stream and updates the last-price hash used by risk/OMS/paper broker.
    """
    p = Pipeline(name)
    p.add("update_last_prices", task_update_last_prices(stream_in or os.getenv("TICKS_STREAM","ticks")))
    return p

def make_capture_pipeline(name: str = "capture_bus", *, db_path: str = "runtime/order_store.db") -> Pipeline:
    """
    Saves orders.acks & fills into SQLite for dashboards and analytics.
    """
    p = Pipeline(name)
    p.add("capture", task_capture_acks_and_fills(db_path=db_path))
    return p

def make_train_predictor_pipeline(symbol: str, csv_path: str, out_path: str,
                                  name: str = "train_predictor", alpha: float = 1.0) -> Pipeline:
    p = Pipeline(name)
    p.add("train_predictor", task_train_price_predictor(symbol, csv_path, out_path, alpha=alpha))
    return p

def make_backtest_pipeline(strategy_name: str, start: str, end: str, symbols: List[str],
                           name: str = "backtest") -> Pipeline:
    p = Pipeline(name)
    p.add("backtest", task_backtest(strategy_name, start, end, symbols))
    return p

# =============================================================================
#                                   CLI
# =============================================================================

def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Pipelines runner")
    ap.add_argument("--name", required=True, help="pipeline name preset",
                    choices=["news", "tick_cache", "capture", "train_predictor", "backtest"])
    ap.add_argument("--interval", type=float, default=0.0, help="if >0, run forever with this seconds between runs")
    # train params
    ap.add_argument("--symbol", help="symbol for predictor/backtest")
    ap.add_argument("--csv", help="csv for training")
    ap.add_argument("--out", default="backend/alpha/artifacts/price_predictor.joblib", help="artifact path")
    ap.add_argument("--alpha", type=float, default=1.0)
    # backtest params
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--symbols", nargs="*", default=[])
    return ap.parse_args()

def main():
    args = _parse_args()

    if args.name == "news":
        pipe = make_news_pipeline(limit=50, with_sentiment=True)
    elif args.name == "tick_cache":
        pipe = make_tick_cache_pipeline()
    elif args.name == "capture":
        pipe = make_capture_pipeline()
    elif args.name == "train_predictor":
        if not (args.symbol and args.csv):
            raise SystemExit("--symbol and --csv required for train_predictor")
        pipe = make_train_predictor_pipeline(args.symbol, args.csv, args.out, alpha=args.alpha)
    elif args.name == "backtest":
        if not (args.symbols and args.start and args.end and args.symbol):
            # 'symbol' kept for symmetry; you may not need it.
            raise SystemExit("--symbols, --start, --end (and optionally --symbol) required for backtest")
        pipe = make_backtest_pipeline(args.name, args.start, args.end, args.symbols)
    else:
        raise SystemExit("unknown pipeline name")

    if args.interval and args.interval > 0:
        pipe.run_forever(args.interval)
    else:
        ctx = pipe.run_once()
        log.info("done. ctx: %s", ctx.vars)

if __name__ == "__main__":
    main()