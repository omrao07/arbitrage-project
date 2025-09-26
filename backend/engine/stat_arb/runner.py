# engines/stat_arb/runner.py
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Literal

from engines.stat_arb.signals import pairs, dispersion # type: ignore
from engines.stat_arb.execution.allocator import allocate_from_units, AllocConfig # type: ignore
from engines.stat_arb.execution.order_router import Order, default_router, ExecutionReport # type: ignore
from engines.stat_arb.backtest.pnl import compute_portfolio_pnl, PairSpec # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SignalType = Literal["pairs", "dispersion"]

# -------------------------------------------------------------------
# Build signals
# -------------------------------------------------------------------

def build_signal(
    signal_type: SignalType,
    prices: pd.DataFrame,
    *,
    group_map: Optional[Dict[str, str]] = None,
    candidates: Optional[List[tuple[str, str]]] = None,
    **kwargs,
):
    """
    Dispatch to the correct stat-arb signal builder.
    Returns (units_by_pair, betas_by_pair, legs_by_pair, diag DataFrame).
    """
    if signal_type == "pairs":
        if not candidates:
            raise ValueError("Pairs signal requires 'candidates'.")
        selected = pairs.select_pairs(prices, candidates, **{k: kwargs.get(k) for k in ["corr_lb","coint_lb","signif","min_corr"] if k in kwargs})
        units_by_pair, betas_by_pair, legs_by_pair, rows = {}, {}, {}, []
        for y, x in selected:
            u, b, z = pairs.generate_pair_signal(
                prices[y], prices[x],
                entry_z=kwargs.get("entry_z", 1.0),
                exit_z=kwargs.get("exit_z", 0.25),
                beta_method=kwargs.get("beta_method","rolling_ols"),
                beta_lookback=kwargs.get("beta_lookback",120),
                z_lookback=kwargs.get("z_lookback",60),
                max_units=kwargs.get("max_units",1.0),
            )
            name = f"{y}/{x}"
            units_by_pair[name] = u
            betas_by_pair[name] = b
            legs_by_pair[name] = (y,x)
            rows.append({"pair": name,"units":u,"beta":b,"z":z})
        diag = pd.DataFrame(rows).set_index("pair") if rows else pd.DataFrame()
        return units_by_pair, betas_by_pair, legs_by_pair, diag

    elif signal_type == "dispersion":
        if not group_map:
            raise ValueError("Dispersion signal requires 'group_map'.")
        snap = dispersion.build_dispersion_snapshot(prices, group_map, **kwargs)
        return snap["units_by_pair"], snap["betas_by_pair"], snap["legs_by_pair"], snap["diag"]

    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

# -------------------------------------------------------------------
# Run full pipeline
# -------------------------------------------------------------------

def run_stat_arb_strategy(
    signal_type: SignalType,
    prices: pd.DataFrame,
    *,
    group_map: Optional[Dict[str, str]] = None,
    candidates: Optional[List[tuple[str, str]]] = None,
    current_pos: Optional[pd.Series] = None,
    adv_map: Optional[Dict[str,float]] = None,
    nav: float = 1_000_000.0,
    alloc_cfg: AllocConfig = AllocConfig(),
    **kwargs,
) -> Dict[str,object]:
    """
    Full pipeline:
      1. Build signal (pairs/dispersion)
      2. Allocate → target shares & generate orders
      3. Route orders via router
      4. Compute PnL via compute_portfolio_pnl
    Returns dict with weights/units, orders, reports, pnl, diagnostics.
    """
    logger.info(f"Running stat_arb strategy: {signal_type}")

    # --- Step 1: signal
    units, betas, legs, diag = build_signal(signal_type, prices, group_map=group_map, candidates=candidates, **kwargs)

    if not units:
        return {"units":units,"orders":pd.DataFrame(),"reports":[],"pnl":{},"diag":diag}

    # --- Step 2: allocate
    last_px = prices.iloc[-1]
    cur = current_pos if current_pos is not None else pd.Series(0.0, index=last_px.index)
    orders = allocate_from_units(
        units_by_pair=units,
        betas_by_pair=betas,
        legs_by_pair=legs,
        current_pos_shares=cur,
        last_prices=last_px,
        adv_usd=adv_map,
        cfg=alloc_cfg,
    )

    # --- Step 3: route
    router = default_router()
    parent_orders = [
        Order(
            ticker=t,
            side=("BUY" if row.order_shares>0 else "SELL"),
            qty=abs(float(row.order_shares)),
            price=float(row.price),
            order_type="MKT",
            meta={"strategy":signal_type},
        )
        for t,row in orders.iterrows()
    ]
    reports: List[ExecutionReport] = router.route_batch(parent_orders, last_px.to_dict(), adv_map or {})

    # --- Step 4: PnL backtest
    pair_specs: Dict[str,PairSpec] = {}
    for name,(y,x) in legs.items():
        u = units[name]
        b = betas[name]
        pair_specs[name] = PairSpec(
            price_y=prices[y], price_x=prices[x],
            spread_units=pd.DataFrame(u, index=prices.index, columns=["u"]),
            fee_bps=0.3, slippage_bps=1.5,
            borrow_bps_y=50.0, borrow_bps_x=50.0,
            div_yield_bps_y=150.0, div_yield_bps_x=150.0,
        )
    pnl = compute_portfolio_pnl(pair_specs)

    return {"units":units,"orders":orders,"reports":reports,"pnl":pnl,"diag":diag}

# -------------------------------------------------------------------
# Demo
# -------------------------------------------------------------------

if __name__=="__main__":
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame({
        "AAA":100*np.exp(np.cumsum(0.0002+0.015*rng.standard_normal(len(idx)))),
        "BBB":100*np.exp(np.cumsum(0.00021+0.015*rng.standard_normal(len(idx)))),
        "CCC":50*np.exp(np.cumsum(0.00025+0.017*rng.standard_normal(len(idx)))),
        "DDD":50*np.exp(np.cumsum(0.00018+0.016*rng.standard_normal(len(idx)))),
    }, index=idx)

    out = run_stat_arb_strategy(
        signal_type="pairs",
        prices=prices,
        candidates=[("AAA","BBB"),("CCC","DDD")],
        entry_z=1.0, exit_z=0.25,
    )
    print("Orders:\n", out["orders"].head()) # type: ignore
    print("PnL summary:\n", out["pnl"]["summary"].tail()) # type: ignore