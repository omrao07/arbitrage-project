# backend/cli/runner.py
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from multiprocessing import Process
from typing import Any, Dict, Optional

# ---- logging -------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("runner")

# ---- optional deps (kept soft) ------------------------------------------
def _try_import(path: str):
    try:
        return importlib.import_module(path)
    except Exception as e:
        raise SystemExit(f"Failed to import {path}: {e}")

# ---- secrets (loads .env if you used backend/utils/secrets.py) ----------
try:
    from backend.utils.secrets import secrets  # type: ignore # noqa
except Exception:
    class _S:
        def get(self, k, default=None, required=False):
            v = os.getenv(k, default)
            if required and v is None:
                raise KeyError(f"Missing required secret {k}")
            return v
    secrets = _S()  # type: ignore

# ---- graceful shutdown ---------------------------------------------------
@dataclass
class _Proc:
    name: str
    proc: Process

_children: list[_Proc] = []

def _spawn(name: str, target, *args, **kwargs) -> Process:
    p = Process(target=target, args=args, kwargs=kwargs, daemon=False, name=name)
    p.start()
    _children.append(_Proc(name, p))
    log.info("spawned %s [pid=%s]", name, p.pid)
    return p

def _on_signal(sig, frame):
    log.warning("signal %s received; terminating children ...", sig)
    for ch in list(_children):
        if ch.proc.is_alive():
            ch.proc.terminate()
    # give a moment, then join
    t0 = time.time()
    for ch in list(_children):
        ch.proc.join(timeout=2.0)
        if ch.proc.is_alive():
            log.warning("forcing kill of %s [pid=%s]", ch.name, ch.proc.pid)
            ch.proc.kill()
    dt = time.time() - t0
    log.info("shutdown complete in %.2fs", dt)
    sys.exit(0)

signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

# ---- helpers -------------------------------------------------------------
def import_symbol(dotted: str):
    """
    Import a symbol from a dotted path like:
      backend.engine.strategy_base:ExampleBuyTheDip
    """
    if ":" in dotted:
        mod, sym = dotted.split(":", 1)
    else:
        # if only module given, expect it to expose 'main' or 'run'
        mod, sym = dotted, ""
    m = _try_import(mod)
    if sym:
        if not hasattr(m, sym):
            raise SystemExit(f"{dotted} not found")
        return getattr(m, sym)
    return m

def _ensure_uvicorn():
    try:
        import uvicorn  # noqa
    except Exception:
        raise SystemExit("uvicorn not installed. Try: pip install uvicorn fastapi")

# ---- subcommands ---------------------------------------------------------
def run_strategy(args: argparse.Namespace) -> None:
    """
    Launch a Strategy subclass and attach to a Redis stream.
    """
    StrategyClass = import_symbol(args.class_path)
    strat = StrategyClass(name=args.name, region=args.region, default_qty=args.default_qty) # type: ignore
    stream = args.stream or os.getenv("TICKS_STREAM", "ticks")
    log.info("Starting strategy '%s' on stream '%s' ...", args.name, stream)
    strat.run(stream=stream)

def run_pipeline_once(args: argparse.Namespace) -> None:
    from backend.pipelines import ( # type: ignore
        make_news_pipeline, make_tick_cache_pipeline, make_capture_pipeline,
        make_train_predictor_pipeline, make_backtest_pipeline,
    )
    if args.which == "news":
        pipe = make_news_pipeline(limit=args.limit, with_sentiment=not args.no_sentiment)
    elif args.which == "tick_cache":
        pipe = make_tick_cache_pipeline()
    elif args.which == "capture":
        pipe = make_capture_pipeline(db_path=args.db)
    elif args.which == "train_predictor":
        if not (args.symbol and args.csv):
            raise SystemExit("--symbol and --csv required")
        pipe = make_train_predictor_pipeline(args.symbol, args.csv, args.out, alpha=args.alpha)
    elif args.which == "backtest":
        if not (args.symbols and args.start and args.end):
            raise SystemExit("--symbols --start --end required")
        pipe = make_backtest_pipeline(args.which, args.start, args.end, args.symbols)
    else:
        raise SystemExit("unknown pipeline")
    ctx = pipe.run_once()
    print(json.dumps(ctx.vars, indent=2, default=str))

def run_pipeline_loop(args: argparse.Namespace) -> None:
    from backend.pipelines import ( # type: ignore
        make_news_pipeline, make_tick_cache_pipeline, make_capture_pipeline,
    )
    if args.which == "news":
        pipe = make_news_pipeline(limit=args.limit, with_sentiment=not args.no_sentiment)
    elif args.which == "tick_cache":
        pipe = make_tick_cache_pipeline()
    elif args.which == "capture":
        pipe = make_capture_pipeline(db_path=args.db)
    else:
        raise SystemExit("only news/tick_cache/capture are supported for --loop")
    interval = float(args.interval)
    log.info("Running pipeline '%s' forever every %.2fs", args.which, interval)
    pipe.run_forever(interval)

def run_api(args: argparse.Namespace) -> None:
    _ensure_uvicorn()
    import uvicorn
    app_path = args.app or "backend.api.app:app"
    host = args.host
    port = int(args.port)
    log.info("Starting API at http://%s:%s ... (%s)", host, port, app_path)
    uvicorn.run(app_path, host=host, port=port, reload=args.reload)

def run_supervisor(args: argparse.Namespace) -> None:
    """
    Start multiple services (pipelines + API) as child processes.
    """
    # news
    if not args.no_news:
        _spawn(
            "pipe-news",
            run_pipeline_loop,
            argparse.Namespace(which="news", limit=args.news_limit, no_sentiment=args.no_sentiment, interval=args.news_interval, db=None),
        )
    # tick cache
    if not args.no_ticks:
        _spawn(
            "pipe-tickcache",
            run_pipeline_loop,
            argparse.Namespace(which="tick_cache", limit=None, no_sentiment=False, interval=args.tick_interval, db=None),
        )
    # capture
    if not args.no_capture:
        _spawn(
            "pipe-capture",
            run_pipeline_loop,
            argparse.Namespace(which="capture", limit=None, no_sentiment=False, interval=args.capture_interval, db=args.db),
        )
    # API
    if not args.no_api:
        _spawn("api", run_api, argparse.Namespace(app=args.app, host=args.host, port=args.port, reload=args.reload))

    # wait forever; signals will stop children
    log.info("Supervisor started. Press Ctrl+C to stop.")
    while True:
        time.sleep(1.0)

def run_reconciler(args: argparse.Namespace) -> None:
    from backend.execution.reconciler import Reconciler # type: ignore
    rec = Reconciler(
        db_path=args.db,
        use_sqlite=not args.no_sqlite,
        use_redis=not args.no_redis,
        base_ccy=args.base_ccy,
        qty_tol=args.qty_tol,
        avg_tol_bps=args.avg_tol_bps,
        cash_tol=args.cash_tol,
        equity_tol=args.equity_tol,
    )
    try:
        out = {
            "positions": rec.reconcile_positions(),
            "cash": rec.reconcile_cash(),
        }
        if args.write_redis:
            rec.write_redis_positions()
        print(json.dumps(out if args.pretty is False else out, indent=2 if args.pretty else None))
    finally:
        rec.close()

def run_latency_dump(_args: argparse.Namespace) -> None:
    try:
        from backend.utils.latency_adapter import all_stats # type: ignore
    except Exception:
        raise SystemExit("latency_adapter not found")
    print(json.dumps(all_stats(), indent=2))

# ---- parser -----------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Hedge Fund Box Runner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # strategy
    s = sub.add_parser("strategy", help="Run a Strategy subclass on a stream")
    s.add_argument("--class-path", required=True, help="e.g. backend.engine.strategy_base:ExampleBuyTheDip")
    s.add_argument("--name", required=True)
    s.add_argument("--region", default=None)
    s.add_argument("--default-qty", type=float, default=1.0)
    s.add_argument("--stream", default=None, help="ticks stream (defaults env TICKS_STREAM or 'ticks')")
    s.set_defaults(func=run_strategy)

    # pipeline once
    p1 = sub.add_parser("pipeline", help="Run a pipeline once")
    p1.add_argument("--which", choices=["news", "tick_cache", "capture", "train_predictor", "backtest"], required=True)
    p1.add_argument("--limit", type=int, default=50)
    p1.add_argument("--no-sentiment", action="store_true")
    p1.add_argument("--db", default="runtime/order_store.db")
    p1.add_argument("--symbol")
    p1.add_argument("--csv")
    p1.add_argument("--out", default="backend/alpha/artifacts/price_predictor.joblib")
    p1.add_argument("--alpha", type=float, default=1.0)
    p1.add_argument("--start")
    p1.add_argument("--end")
    p1.add_argument("--symbols", nargs="*")
    p1.set_defaults(func=run_pipeline_once)

    # pipeline loop
    p2 = sub.add_parser("loop", help="Run a pipeline forever")
    p2.add_argument("--which", choices=["news", "tick_cache", "capture"], required=True)
    p2.add_argument("--interval", type=float, default=30.0)
    p2.add_argument("--limit", type=int, default=50)
    p2.add_argument("--no-sentiment", action="store_true")
    p2.add_argument("--db", default="runtime/order_store.db")
    p2.set_defaults(func=run_pipeline_loop)

    # api
    a = sub.add_parser("api", help="Run FastAPI app")
    a.add_argument("--app", default="backend.api.app:app", help="ASGI app path")
    a.add_argument("--host", default="0.0.0.0")
    a.add_argument("--port", type=int, default=8080)
    a.add_argument("--reload", action="store_true")
    a.set_defaults(func=run_api)

    # supervisor
    sv = sub.add_parser("supervisor", help="Run news/ticks/capture and API together")
    sv.add_argument("--no-news", action="store_true")
    sv.add_argument("--no-ticks", action="store_true")
    sv.add_argument("--no-capture", action="store_true")
    sv.add_argument("--no-api", action="store_true")
    sv.add_argument("--news-interval", type=float, default=30.0)
    sv.add_argument("--tick-interval", type=float, default=0.5)
    sv.add_argument("--capture-interval", type=float, default=1.0)
    sv.add_argument("--news-limit", type=int, default=50)
    sv.add_argument("--no-sentiment", action="store_true")
    sv.add_argument("--db", default="runtime/order_store.db")
    sv.add_argument("--app", default="backend.api.app:app")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=run_supervisor)

    # reconciler
    r = sub.add_parser("reconcile", help="Compare broker vs internal positions/cash")
    r.add_argument("--db", default="runtime/order_store.db")
    r.add_argument("--no-sqlite", action="store_true")
    r.add_argument("--no-redis", action="store_true")
    r.add_argument("--base-ccy", default="USD")
    r.add_argument("--qty-tol", type=float, default=1e-6)
    r.add_argument("--avg-tol-bps", type=float, default=5.0)
    r.add_argument("--cash-tol", type=float, default=1.0)
    r.add_argument("--equity-tol", type=float, default=5.0)
    r.add_argument("--write-redis", action="store_true")
    r.add_argument("--pretty", action="store_true")
    r.set_defaults(func=run_reconciler)

    # latency dump
    l = sub.add_parser("latency", help="Print latency stats gathered by latency_adapter")
    l.set_defaults(func=run_latency_dump)

    return ap

# ---- entry ------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()