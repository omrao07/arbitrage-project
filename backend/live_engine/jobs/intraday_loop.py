# backend/live_engine/jobs/intraday_loop.py
"""
Intraday 60-second tick loop — the heartbeat of the live engine.

Called every 60s by the scheduler during market hours (9:15 AM – 3:30 PM IST).

Uses pre-initialised singletons from engine_state so nothing is re-created
between ticks — strategies stay warm, positions stay live, the router stays
connected.

Full tick pipeline:
  1.  Kill-switch / halt check (fast exit)
  2.  Fetch current-minute bars for every tracked symbol
  3.  Run ALL active strategies on each bar (parallel via ThreadPoolExecutor)
  4.  Run each resulting OrderRequest through InstitutionalRiskEngine (11 gates)
  5.  Route approved orders to Zerodha → fills → PnL tracker
  6.  Mark portfolio to market with latest quotes
  7.  Drawdown + daily-loss kill-switch re-check (fire if breached this tick)
  8.  Publish portfolio state → Redis + WebSocket
  9.  Record hourly VaR/CVaR snapshot (every ~60 ticks)
  10. Log concise tick summary
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

import numpy as np

log = logging.getLogger(__name__)

# How often (in ticks) to recompute VaR snapshot
_VAR_RECALC_EVERY = 60


def run_tick() -> Dict[str, Any]:
    """
    Execute one 60-second tick.  Returns a summary dict.
    All heavy objects come from the shared engine_state singleton.
    """
    from backend.live_engine.engine_state import state

    t0 = time.perf_counter()

    summary: Dict[str, Any] = {
        "ts": int(time.time()),
        "n_bars": 0,
        "n_strategies": 0,
        "n_orders_raw": 0,
        "n_orders_approved": 0,
        "n_orders_submitted": 0,
        "n_orders_rejected": 0,
        "daily_pnl": 0.0,
        "equity": 0.0,
        "drawdown": 0.0,
        "kill_switch": False,
        "elapsed_ms": 0,
    }

    # ── 0. Ensure state is ready ──────────────────────────────────────────────
    if not state.is_ready():
        log.warning("EngineState not initialized — calling initialize()")
        state.initialize()

    # ── 1. Kill-switch fast exit ──────────────────────────────────────────────
    if state.is_halted():
        log.warning("Tick skipped — kill switch or daily halt is active")
        summary["kill_switch"] = True
        state.publish_portfolio_state(summary)
        return summary

    # ── 2. Fetch bars ─────────────────────────────────────────────────────────
    bars: Dict[str, dict] = {}
    prices: Dict[str, float] = {}

    if state.market_svc:
        try:
            from backend.live_engine.config import NIFTY50_SYMBOLS
            quotes = state.market_svc.get_multi_quote(NIFTY50_SYMBOLS)
            for sym, q in (quotes or {}).items():
                if not q:
                    continue
                ltp = float(q.get("last_price") or q.get("close") or 0)
                if ltp <= 0:
                    continue
                ohlc = q.get("ohlc", {}) or {}
                bars[sym] = {
                    "symbol": sym,
                    "open":   float(ohlc.get("open", ltp)),
                    "high":   float(ohlc.get("high", ltp)),
                    "low":    float(ohlc.get("low",  ltp)),
                    "close":  ltp,
                    "volume": float(q.get("volume", 0)),
                    "ts":     int(time.time()),
                }
                prices[sym] = ltp
            summary["n_bars"] = len(bars)
        except Exception as exc:
            log.error("Bar fetch failed: %s", exc)

    if not bars:
        log.warning("No bars received — skipping strategy run")
        return summary

    # ── 3. Run strategies (all 337, parallel) ─────────────────────────────────
    raw_orders = []
    if state.runner:
        try:
            raw_orders = state.runner.run_all_bars(bars)
            summary["n_strategies"] = state.runner.strategy_count()
            summary["n_orders_raw"] = len(raw_orders)
        except Exception as exc:
            log.error("StrategyRunner.run_all_bars failed: %s", exc)

    # ── 4. Institutional risk gate + route ────────────────────────────────────
    n_approved = n_submitted = n_rejected = 0

    if raw_orders and state.router:
        # Build shared portfolio/market state once per tick (not per order)
        portfolio_state = _build_portfolio_state(state, prices)
        market_state    = _build_market_state(state)

        for order_req in raw_orders:
            try:
                # 4a. InstitutionalRiskEngine pre-trade check (11 gates)
                approved, modified_qty = _institutional_gate(
                    state, order_req, portfolio_state, market_state
                )
                if not approved:
                    n_rejected += 1
                    continue
                n_approved += 1

                # Apply scaled qty from risk gates
                if modified_qty is not None and modified_qty > 0:
                    order_req.qty = modified_qty

                # 4b. OrderRouter → Zerodha → PnL tracker
                order_id = state.router.route(order_req)
                if order_id:
                    n_submitted += 1
                else:
                    n_rejected += 1

            except Exception as exc:
                log.debug("Order pipeline error for %s: %s", order_req.symbol, exc)
                n_rejected += 1

    summary["n_orders_approved"] = n_approved
    summary["n_orders_submitted"] = n_submitted
    summary["n_orders_rejected"] = n_rejected

    # ── 5. Mark-to-market ─────────────────────────────────────────────────────
    if state.tracker and prices:
        try:
            state.tracker.mark_to_market(prices)
            summary["daily_pnl"]  = round(state.tracker.get_daily_pnl(), 2)
            summary["equity"]     = round(state.tracker.get_total_equity(), 2)
            summary["drawdown"]   = round(state.tracker.get_drawdown(), 6)
        except Exception as exc:
            log.error("mark_to_market failed: %s", exc)

    # ── 6. Kill-switch re-check after fills ───────────────────────────────────
    if state.risk_engine and state.tracker and not state.is_halted():
        try:
            _check_and_fire_kill_switch(state, summary)
        except Exception as exc:
            log.debug("Kill-switch check error: %s", exc)

    # ── 7. Publish state ──────────────────────────────────────────────────────
    state.publish_portfolio_state(summary)

    # ── 8. Periodic VaR snapshot ──────────────────────────────────────────────
    if state.ticks_processed % _VAR_RECALC_EVERY == 0:
        _async_var_snapshot(state)

    # ── 9. Record stats and log ───────────────────────────────────────────────
    elapsed = (time.perf_counter() - t0) * 1000
    summary["elapsed_ms"] = round(elapsed, 1)
    state.record_tick_stats(n_submitted, n_rejected, summary["daily_pnl"])

    log.info(
        "TICK #%d | %dms | bars=%d strats=%d raw=%d ✓=%d →broker=%d ✗=%d | "
        "PnL=₹%.0f eq=₹%.0f DD=%.2f%%",
        state.ticks_processed,
        round(elapsed),
        summary["n_bars"],
        summary["n_strategies"],
        summary["n_orders_raw"],
        n_approved,
        n_submitted,
        n_rejected,
        summary["daily_pnl"],
        summary["equity"],
        summary["drawdown"] * 100,
    )
    return summary


# ── Risk helpers ──────────────────────────────────────────────────────────────

def _build_portfolio_state(state, prices: Dict[str, float]) -> dict:
    """Build the portfolio_state dict required by InstitutionalRiskEngine."""
    nav = peak = 0.0
    daily_pnl = drawdown = gross_exposure = 0.0
    positions: dict = {}
    sector_weights: dict = {}

    if state.tracker:
        nav          = state.tracker.get_total_equity()
        peak         = state.tracker.get_peak_equity()
        daily_pnl    = state.tracker.get_daily_pnl()
        drawdown     = state.tracker.get_drawdown()
        positions    = state.tracker.get_all_positions()

    if state.redis:
        try:
            raw = state.redis.hget("portfolio:sector_weights", "json")
            if raw:
                sector_weights = json.loads(raw)
            ge_raw = state.redis.get("portfolio:gross_usd")
            if ge_raw:
                gross_exposure = float(json.loads(ge_raw).get("usd", 0))
        except Exception:
            pass

    return {
        "nav": nav,
        "peak_equity": peak,
        "daily_pnl": daily_pnl,
        "drawdown": drawdown,
        "positions": positions,
        "sector_weights": sector_weights,
        "gross_exposure": gross_exposure,
    }


def _build_market_state(state) -> dict:
    """Build the market_state dict required by InstitutionalRiskEngine."""
    india_vix = 15.0
    fo_ban_list: list = []
    circuit_breakers: dict = {}

    if state.market_svc:
        try:
            india_vix = state.market_svc.get_india_vix()
        except Exception:
            pass
        try:
            fo_ban_list = state.market_svc.get_fo_ban_list()
        except Exception:
            pass

    if state.redis:
        try:
            raw = state.redis.get("market:circuit_breakers")
            if raw:
                circuit_breakers = json.loads(raw)
        except Exception:
            pass

    # Portfolio returns for beta calculation (last 60 bars)
    portfolio_returns: list = []
    benchmark_returns: list = []
    if state.redis:
        try:
            raw_p = state.redis.lrange("portfolio:daily_returns", -60, -1)
            portfolio_returns = [float(x) for x in raw_p]
            raw_b = state.redis.lrange("market:nifty_returns", -60, -1)
            benchmark_returns = [float(x) for x in raw_b]
        except Exception:
            pass

    return {
        "india_vix": india_vix,
        "cboe_vix": 20.0,
        "fo_ban_list": fo_ban_list,
        "circuit_breakers": circuit_breakers,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
    }


def _institutional_gate(state, order_req, portfolio_state: dict, market_state: dict):
    """
    Run the InstitutionalRiskEngine's pre_trade_check() on one order.
    Returns (approved: bool, scaled_qty: float|None).
    Falls back to approved=True if the risk engine isn't available.
    """
    if not state.risk_engine:
        return True, None

    order_dict = {
        "symbol": order_req.symbol,
        "side": order_req.side,
        "qty": order_req.qty,
        "price": order_req.limit_price or 0.0,
        "order_type": order_req.order_type,
        "strategy": order_req.strategy,
        "sector": getattr(order_req, "sector", ""),
        "fo_type": getattr(order_req, "fo_type", ""),
    }

    # Add existing position qty to portfolio state for position-size gate
    portfolio_state = {
        **portfolio_state,
        "existing_qty": float(
            portfolio_state.get("positions", {})
            .get(order_req.symbol, {})
            .get("qty", 0.0)
        ),
    }

    try:
        approved, gate_results, modified_order = state.risk_engine.pre_trade_check(
            order_dict, portfolio_state, market_state
        )
        scaled_qty = float(modified_order.get("qty", order_req.qty)) if modified_order else None
        return approved, scaled_qty
    except Exception as exc:
        log.error("InstitutionalRiskEngine.pre_trade_check raised: %s", exc)
        return True, None   # fail open — log and continue


def _check_and_fire_kill_switch(state, summary: dict) -> None:
    """Fire kill switch if drawdown or daily loss limit is breached this tick."""
    from backend.risk.institutional_risk_engine import get_risk_config_from_redis
    config = get_risk_config_from_redis(state.redis)

    dd     = summary.get("drawdown", 0.0)
    nav    = summary.get("equity", 0.0)
    dpnl   = summary.get("daily_pnl", 0.0)

    if dd > config.drawdown_kill_switch_pct:
        if state.redis:
            state.redis.set("risk:kill_switch_active", "1")
        state.alert(
            f"🚨 DRAWDOWN KILL SWITCH FIRED\n"
            f"Drawdown {dd*100:.2f}% > limit {config.drawdown_kill_switch_pct*100:.1f}%\n"
            f"All orders cancelled."
        )
        if state.router:
            state.router.cancel_all_orders()
        summary["kill_switch"] = True
        log.critical("KILL SWITCH: drawdown %.2f%% exceeded %.2f%%",
                     dd * 100, config.drawdown_kill_switch_pct * 100)

    elif nav > 0 and (dpnl / nav) < -config.daily_loss_limit_pct:
        if state.redis:
            state.redis.set("risk:daily_trading_halted", "1")
        state.alert(
            f"🚨 DAILY LOSS LIMIT HIT\n"
            f"Daily PnL {dpnl/nav*100:.2f}% < -{config.daily_loss_limit_pct*100:.1f}%\n"
            f"Trading halted for today."
        )
        summary["kill_switch"] = True
        log.critical("DAILY HALT: daily loss %.2f%% exceeded %.2f%%",
                     dpnl / nav * 100, config.daily_loss_limit_pct * 100)


def _async_var_snapshot(state) -> None:
    """Recompute VaR/CVaR in a background thread so it doesn't block the tick."""
    import threading
    def _compute():
        try:
            if not state.redis:
                return
            raw = state.redis.lrange("portfolio:daily_returns", -252, -1)
            rets = np.array([float(x) for x in raw])
            from backend.risk.institutional_risk_engine import (
                InsufficientDataError,
                PortfolioRiskEngine,
                VaREngine,
            )
            var_e = VaREngine()
            port_e = PortfolioRiskEngine()
            try:
                snap = {
                    "ts":             str(int(time.time())),
                    "status":         "ok",
                    "var_99":         str(round(var_e.historical_var(rets, 0.99, 1), 6)),
                    "cvar_975":       str(round(var_e.historical_cvar(rets, 0.975), 6)),
                    "sharpe_rolling": str(round(port_e.sharpe_ratio(rets), 4)),
                    "max_drawdown":   str(round(port_e.max_drawdown(np.cumprod(1 + rets)), 4)),
                    "n_obs":          str(len(rets)),
                }
                state.redis.hset("risk:intraday_snapshot", mapping=snap)
            except InsufficientDataError as exc:
                # Publish the absence explicitly; stale/absent numbers must not
                # be mistaken for a live reading.
                state.redis.hset("risk:intraday_snapshot", mapping={
                    "ts": str(int(time.time())),
                    "status": "insufficient_data",
                    "reason": str(exc),
                    "n_obs": str(len(rets)),
                })
                state.redis.hdel("risk:intraday_snapshot", "var_99", "cvar_975",
                                 "sharpe_rolling", "max_drawdown")
                log.warning("Intraday VaR snapshot unavailable: %s", exc)
        except Exception as exc:
            log.warning("VaR snapshot error: %s", exc)
    threading.Thread(target=_compute, daemon=True).start()
