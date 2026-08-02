# backend/live_engine/scheduler.py
"""
Live Engine Scheduler — orchestrates all automated jobs using APScheduler.

IST cron schedule:
  9:10 AM Mon-Fri   pre_market_job       — VIX, F&O ban, arm risk limits
  9:15 AM Mon-Fri   intraday_loop start  — 60s tick loop begins
  Every 60s         intraday_tick        — new bars → strategies → risk → orders → PnL
  Every 5 min       health_monitor       — ping all services, Telegram if down
  Every 1 hour      intraday_risk_recalc — VaR, CVaR, beta, sector exposure
  3:30 PM Mon-Fri   post_market_job      — reconcile, PnL calc, Telegram EOD
  10:00 PM Mon-Fri  nightly_pipeline     — OHLCV download, features, re-rank
  6:00 AM Tue-Sat   nightly_early        — lighter 6 AM features + allocation pass
  Sunday 8:00 AM    weekly_pipeline      — ML retrain, WF re-run, alpha decay
  1st Sunday 9 AM   monthly_pipeline     — full rebalance, tax opt, full backtest
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)

try:
    import pytz
    _IST = pytz.timezone("Asia/Kolkata")
except ImportError:
    _IST = None

try:
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    _HAS_APScheduler = True
except ImportError:
    _HAS_APScheduler = False
    log.warning("APScheduler not installed — install with: pip install apscheduler")


class LiveEngineScheduler:
    """
    Manages all automated live trading jobs.
    Gracefully handles job failures and sends Telegram alerts.
    """

    def __init__(self):
        self._scheduler: Optional[Any] = None
        self._start_time: Optional[float] = None
        self._job_stats: Dict[str, dict] = {}
        self._lock = threading.Lock()

        if _HAS_APScheduler:
            tz = _IST if _IST else "UTC"
            self._scheduler = BackgroundScheduler(timezone=tz)
            self._scheduler.add_listener(self._on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
            self._register_jobs()
        else:
            log.error("APScheduler unavailable — scheduler will not run")

        # Graceful shutdown (only register if not already handled by __main__)
        if not getattr(signal.getsignal(signal.SIGTERM), "__module__", "").startswith("backend.live_engine.__main__"):
            try:
                signal.signal(signal.SIGTERM, self._handle_shutdown)
                signal.signal(signal.SIGINT, self._handle_shutdown)
            except ValueError:
                pass  # Not the main thread — signal registration not allowed

    # ── Job registry ──────────────────────────────────────────────────────────

    def _safe_run(self, job_name: str, fn: Callable) -> None:
        """Wrap a job function with error handling and Telegram alerts."""
        def wrapper():
            t0 = time.perf_counter()
            log.info("[JOB START] %s", job_name)
            try:
                fn()
                elapsed = time.perf_counter() - t0
                log.info("[JOB DONE] %s in %.1fs", job_name, elapsed)
                with self._lock:
                    self._job_stats[job_name] = {
                        "last_run": time.time(),
                        "last_status": "ok",
                        "last_elapsed_s": round(elapsed, 2),
                        "error": None,
                    }
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                log.error("[JOB FAILED] %s after %.1fs: %s", job_name, elapsed, exc, exc_info=True)
                with self._lock:
                    self._job_stats[job_name] = {
                        "last_run": time.time(),
                        "last_status": "error",
                        "last_elapsed_s": round(elapsed, 2),
                        "error": str(exc),
                    }
                self._alert_job_failure(job_name, str(exc))
        return wrapper

    def _register_jobs(self) -> None:
        if not self._scheduler:
            return
        tz = _IST if _IST else "UTC"

        # ── Pre-market: 9:10 AM Mon-Fri ───────────────────────────────────────
        from backend.live_engine.jobs.pre_market_job import run as pre_market_run
        self._scheduler.add_job(
            self._safe_run("pre_market_job", pre_market_run),
            CronTrigger(hour=9, minute=10, day_of_week="mon-fri", timezone=tz),
            id="pre_market_job",
            name="Pre-Market Setup",
            replace_existing=True,
            max_instances=1,
        )

        # ── Intraday tick: every 60s during market hours ───────────────────────
        from backend.live_engine.jobs.intraday_loop import run_tick
        def _guarded_tick():
            from backend.live_engine.config import is_market_open
            if is_market_open():
                run_tick()
        self._scheduler.add_job(
            self._safe_run("intraday_tick", _guarded_tick),
            IntervalTrigger(seconds=60, timezone=tz),
            id="intraday_tick",
            name="Intraday 60s Tick",
            replace_existing=True,
            max_instances=1,
        )

        # ── Health monitor: every 5 minutes ───────────────────────────────────
        from backend.live_engine.jobs.health_monitor import run as health_run
        self._scheduler.add_job(
            self._safe_run("health_monitor", health_run),
            IntervalTrigger(seconds=300, timezone=tz),
            id="health_monitor",
            name="Health Monitor",
            replace_existing=True,
            max_instances=1,
        )

        # ── Intraday risk recalculation: every 1 hour ─────────────────────────
        self._scheduler.add_job(
            self._safe_run("intraday_risk_recalc", self._run_risk_recalc),
            IntervalTrigger(hours=1, timezone=tz),
            id="intraday_risk_recalc",
            name="Intraday Risk Recalc",
            replace_existing=True,
            max_instances=1,
        )

        # ── Post-market: 3:30 PM Mon-Fri ──────────────────────────────────────
        from backend.live_engine.jobs.post_market_job import run as post_market_run
        self._scheduler.add_job(
            self._safe_run("post_market_job", post_market_run),
            CronTrigger(hour=15, minute=30, day_of_week="mon-fri", timezone=tz),
            id="post_market_job",
            name="Post-Market Reconciliation",
            replace_existing=True,
            max_instances=1,
        )

        # ── Nightly pipeline: 10:00 PM Mon-Fri ────────────────────────────────
        from backend.live_engine.jobs.nightly_pipeline import run as nightly_run
        self._scheduler.add_job(
            self._safe_run("nightly_pipeline", nightly_run),
            CronTrigger(hour=22, minute=0, day_of_week="mon-fri", timezone=tz),
            id="nightly_pipeline",
            name="Nightly Pipeline (10 PM)",
            replace_existing=True,
            max_instances=1,
        )

        # ── Validation gauntlet: 11:00 PM Mon-Fri (after nightly data) ────────
        # Re-scores every registered strategy on real history and demotes
        # anything whose DSR/PBO has decayed. Runs after nightly_pipeline so it
        # sees the day's data.
        from backend.live_engine.jobs.gauntlet_job import run as gauntlet_run
        self._scheduler.add_job(
            self._safe_run("validation_gauntlet", gauntlet_run),
            CronTrigger(hour=23, minute=0, day_of_week="mon-fri", timezone=tz),
            id="validation_gauntlet",
            name="Validation Gauntlet (DSR/PBO re-score + auto-demote)",
            replace_existing=True,
            max_instances=1,
        )

        # ── Early morning pass: 6:00 AM Tue-Sat ───────────────────────────────
        from backend.live_engine.jobs.nightly_pipeline import run_lite
        self._scheduler.add_job(
            self._safe_run("nightly_early", run_lite),
            CronTrigger(hour=6, minute=0, day_of_week="tue-sat", timezone=tz),
            id="nightly_early",
            name="Early Morning Data Refresh (6 AM)",
            replace_existing=True,
            max_instances=1,
        )

        # ── Weekly pipeline: Sunday 8:00 AM ───────────────────────────────────
        from backend.live_engine.jobs.weekly_pipeline import run as weekly_run
        self._scheduler.add_job(
            self._safe_run("weekly_pipeline", weekly_run),
            CronTrigger(hour=8, minute=0, day_of_week="sun", timezone=tz),
            id="weekly_pipeline",
            name="Weekly Pipeline (Sunday)",
            replace_existing=True,
            max_instances=1,
        )

        # ── Monthly pipeline: 1st Sunday of month 9:00 AM ─────────────────────
        from backend.live_engine.jobs.monthly_pipeline import run as monthly_run
        self._scheduler.add_job(
            self._safe_run("monthly_pipeline", monthly_run),
            CronTrigger(hour=9, minute=0, day_of_week="sun", day="1-7", timezone=tz),
            id="monthly_pipeline",
            name="Monthly Portfolio Rebalance (1st Sunday)",
            replace_existing=True,
            max_instances=1,
        )

        log.info("All %d live engine jobs registered", len(self._scheduler.get_jobs()))

    # ── Hourly risk recalc ────────────────────────────────────────────────────

    def _run_risk_recalc(self) -> None:
        """Recompute VaR, CVaR, beta, sector exposure every hour."""
        try:
            import numpy as np
            import redis as _r_mod
            r = _r_mod.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True,
            )
            from backend.risk.institutional_risk_engine import (
                InsufficientDataError,
                PortfolioRiskEngine,
                VaREngine,
                get_risk_config_from_redis,
            )

            raw_returns = r.lrange("portfolio:daily_returns", 0, -1)
            returns = np.array([float(x) for x in raw_returns])
            var_engine = VaREngine()
            port_engine = PortfolioRiskEngine()

            # Risk metrics are best-effort; the kill-switch check below is NOT.
            # An unavailable metric is published as "unavailable", never as 0.0 —
            # a zero VaR renders green on the dashboard and is divided into an
            # equity budget by CVaR sizing.
            try:
                hist_var = var_engine.historical_var(returns, 0.99, 1)
                cvar = var_engine.historical_cvar(returns, 0.975)
                sharpe = port_engine.sharpe_ratio(returns)
                max_dd = port_engine.max_drawdown(np.cumprod(1 + returns))
                snapshot = {
                    "ts": str(int(time.time())),
                    "status": "ok",
                    "var_99_1d": str(round(hist_var, 6)),
                    "cvar_975": str(round(cvar, 6)),
                    "sharpe_rolling": str(round(sharpe, 4)),
                    "max_drawdown": str(round(max_dd, 4)),
                    "n_obs": str(len(returns)),
                }
                r.hset("risk:hourly_metrics", mapping=snapshot)
                log.info("Risk recalc: VaR=%.4f, CVaR=%.4f, Sharpe=%.3f, MDD=%.3f",
                         hist_var, cvar, sharpe, max_dd)
            except InsufficientDataError as exc:
                r.hset("risk:hourly_metrics", mapping={
                    "ts": str(int(time.time())),
                    "status": "insufficient_data",
                    "reason": str(exc),
                    "n_obs": str(len(returns)),
                })
                r.hdel("risk:hourly_metrics", "var_99_1d", "cvar_975",
                       "sharpe_rolling", "max_drawdown")
                log.warning("Risk metrics unavailable: %s", exc)

            # Kill switch check
            config = get_risk_config_from_redis(r)
            from backend.live_engine.pnl_tracker import PnLTracker
            tracker = PnLTracker()
            drawdown = tracker.get_drawdown()
            daily_pnl = tracker.get_daily_pnl()
            nav = tracker.get_total_equity()

            if drawdown > config.drawdown_kill_switch_pct:
                r.set("risk:kill_switch_active", "1")
                self._alert(f"🚨 DRAWDOWN KILL SWITCH: {drawdown*100:.1f}% > {config.drawdown_kill_switch_pct*100:.1f}%")
                log.critical("KILL SWITCH activated: drawdown %.2f%%", drawdown * 100)

            if nav > 0 and daily_pnl / nav < -config.daily_loss_limit_pct:
                r.set("risk:daily_trading_halted", "1")
                self._alert(f"🚨 DAILY LOSS LIMIT: {daily_pnl/nav*100:.2f}% < -{config.daily_loss_limit_pct*100:.1f}%")
                log.critical("DAILY HALT: daily loss %.2f%%", daily_pnl / nav * 100)

        except Exception as exc:
            log.error("Risk recalc failed: %s", exc)

    # ── Scheduler lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        if not self._scheduler:
            log.error("Scheduler not initialized (APScheduler missing)")
            return
        if not self._scheduler.running:
            self._scheduler.start()
            self._start_time = time.time()
            log.info("LiveEngineScheduler started. %d jobs active.", len(self._scheduler.get_jobs()))

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=True)
            log.info("LiveEngineScheduler stopped gracefully")

    def is_running(self) -> bool:
        return bool(self._scheduler and self._scheduler.running)

    def has_job(self, job_id: str) -> bool:
        if not self._scheduler:
            return False
        return self._scheduler.get_job(job_id) is not None

    def trigger_job(self, job_id: str) -> None:
        """Manually fire a job immediately (for /live/trigger API)."""
        if not self._scheduler:
            raise RuntimeError("Scheduler not running")
        job = self._scheduler.get_job(job_id)
        if not job:
            raise ValueError(f"No job: {job_id}")
        job.modify(next_run_time=time.time())
        log.info("Manually triggered job: %s", job_id)

    def status(self) -> dict:
        if not self._scheduler:
            return {"running": False, "jobs": [], "uptime_s": 0.0}

        jobs = []
        for job in self._scheduler.get_jobs():
            stats = self._job_stats.get(job.id, {})
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None,
                "last_status": stats.get("last_status"),
                "last_run": stats.get("last_run"),
                "last_elapsed_s": stats.get("last_elapsed_s"),
                "error": stats.get("error"),
            })

        return {
            "running": self._scheduler.running,
            "jobs": jobs,
            "uptime_s": round(time.time() - self._start_time, 1) if self._start_time else 0.0,
        }

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_job_event(self, event) -> None:
        if hasattr(event, "exception") and event.exception:
            log.error("APScheduler job error [%s]: %s", event.job_id, event.exception)

    def _alert_job_failure(self, job_name: str, error: str) -> None:
        self._alert(f"⚠️ Job FAILED: {job_name}\n{error[:200]}")

    def _alert(self, msg: str) -> None:
        try:
            from backend.live_engine.telegram_alerts import TelegramAlerter
            TelegramAlerter().send_sync(msg)
        except Exception:
            pass

    def _handle_shutdown(self, signum, frame) -> None:
        log.info("Signal %s received — stopping scheduler", signum)
        self.stop()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    sched = LiveEngineScheduler()
    sched.start()
    log.info("Live engine running. Press Ctrl+C to stop.")
    try:
        while sched.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        sched.stop()


if __name__ == "__main__":
    main()
