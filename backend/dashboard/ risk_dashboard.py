# backend/dashboard/risk_dashboard.py
"""
Risk Dashboard (Dash)
---------------------
Interactive dashboard for portfolio risk, PnL attribution, exposures, and execution quality (TCA).

Run:
  python -m backend.dashboard.risk_dashboard
Requires:
  pip install dash plotly requests pandas
Env (optional):
  DATA_API_URL=http://localhost:8000
  DATA_API_KEY=dev123
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, List

import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import dash # type: ignore
from dash import Dash, dcc, html, dash_table # type: ignore
from dash.dependencies import Input, Output # type: ignore

API_URL = os.getenv("DATA_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("DATA_API_KEY", "").strip()

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# -------- helpers to fetch from data_api --------

def _get(path: str, default: Any):
    url = f"{API_URL}{path}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default

def fetch_snapshots():
    pnl = _get("/pnl", {})
    risk = _get("/risk", {})
    tca = _get("/tca", {})
    acct = _get("/account", {"equity": 0.0, "cash": 0.0, "buying_power": 0.0, "currency": "USD"})
    pos = _get("/positions", {})
    prices = _get("/prices", {})
    return pnl, risk, tca, acct, pos, prices

def _totals(d: Dict[str, Any]) -> Dict[str, float]:
    t = d.get("totals") or {}
    return {
        "realized": float(t.get("realized", 0.0) or 0.0),
        "unrealized": float(t.get("unrealized", 0.0) or 0.0),
        "fees": float(t.get("fees", 0.0) or 0.0),
        "pnl": float(t.get("pnl", 0.0) or 0.0),
    }

def _bucket_to_df(d: Dict[str, Any], label: str) -> pd.DataFrame:
    rows = []
    for k, v in (d or {}).items():
        rows.append({
            label: str(k),
            "realized": float(v.get("realized", 0.0) or 0.0),
            "unrealized": float(v.get("unrealized", 0.0) or 0.0),
            "fees": float(v.get("fees", 0.0) or 0.0),
            "pnl": float(v.get("pnl", 0.0) or 0.0),
        })
    if not rows:
        return pd.DataFrame(columns=[label, "realized", "unrealized", "fees", "pnl"])
    df = pd.DataFrame(rows).sort_values("pnl", ascending=False)
    return df

def _risk_to_df(risk: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for strat, met in (risk or {}).items():
        if isinstance(met, dict):
            rows.append({
                "strategy": str(strat),
                "sharpe": float(met.get("sharpe", 0.0) or 0.0),
                "sortino": float(met.get("sortino", 0.0) or 0.0),
                "vol": float(met.get("vol", 0.0) or 0.0),
                "max_drawdown": float(met.get("max_drawdown", 0.0) or 0.0),
                "var_95": float(met.get("var_95", 0.0) or 0.0),
            })
    return pd.DataFrame(rows)

def _pos_to_df(pos: Dict[str, Dict[str, float]], prices: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for sym, p in (pos or {}).items():
        qty = float(p.get("qty", 0.0) or 0.0)
        avg = float(p.get("avg_price", 0.0) or 0.0)
        px  = float(prices.get(sym, avg) or 0.0)
        mtm = qty * px
        rows.append({"symbol": sym, "qty": qty, "avg_price": avg, "last_price": px, "mtm": mtm})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mtm", ascending=False)
    return df

def _tca_totals(tca: Dict[str, Any]) -> Dict[str, Any]:
    tot = (tca or {}).get("totals") or {}
    return {
        "orders": int(tot.get("orders", 0) or 0),
        "qty_filled": float(tot.get("qty_filled", 0.0) or 0.0),
        "is_bps_wavg": tot.get("is_bps_wavg", None),
        "fill_ratio_wavg": tot.get("fill_ratio_wavg", None),
        "slippage_bps_vs_mid_avg": tot.get("slippage_bps_vs_mid_avg", None),
        "ttf_first_med_s": tot.get("ttf_first_med_s", None),
        "ttf_last_med_s": tot.get("ttf_last_med_s", None),
    }

# -------- Dash app --------

app: Dash = dash.Dash(__name__)
app.title = "Risk Dashboard"

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, sans-serif", "padding": "18px"},
    children=[
        html.H2("Risk Dashboard"),
        html.Div(id="top-cards", style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginTop": "16px"},
            children=[
                dcc.Graph(id="heatmap-strategy"),
                dcc.Graph(id="heatmap-region"),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "marginTop": "16px"},
            children=[
                dcc.Graph(id="pnl-bar"),
                dcc.Graph(id="risk-table"),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "10px", "marginTop": "16px"},
            children=[
                html.H4("Positions"),
                dash_table.DataTable(
                    id="positions-table",
                    columns=[
                        {"name": "Symbol", "id": "symbol"},
                        {"name": "Qty", "id": "qty", "type": "numeric", "format": {"specifier": ".4f"}},
                        {"name": "Avg Price", "id": "avg_price", "type": "numeric", "format": {"specifier": ".4f"}},
                        {"name": "Last Price", "id": "last_price", "type": "numeric", "format": {"specifier": ".4f"}},
                        {"name": "MTM", "id": "mtm", "type": "numeric", "format": {"specifier": ".2f"}},
                    ],
                    data=[],
                    page_size=8,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "6px", "fontSize": "14px"},
                    style_header={"fontWeight": "600"},
                ),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px", "marginTop": "16px"},
            children=[
                dcc.Graph(id="tca-by-strategy"),
                dcc.Graph(id="tca-by-region"),
                dcc.Graph(id="tca-scatter"),
            ],
        ),
        dcc.Interval(id="poll", interval=3000, n_intervals=0),
        html.Div(style={"marginTop": "8px", "color": "#888"}, children=[
            "Source: ", html.Code(API_URL), " (set DATA_API_URL / DATA_API_KEY env if needed)"
        ]),
    ],
)

# -------- Callbacks --------

@app.callback(
    [
        Output("top-cards", "children"),
        Output("heatmap-strategy", "figure"),
        Output("heatmap-region", "figure"),
        Output("pnl-bar", "figure"),
        Output("risk-table", "figure"),
        Output("positions-table", "data"),
        Output("tca-by-strategy", "figure"),
        Output("tca-by-region", "figure"),
        Output("tca-scatter", "figure"),
    ],
    [Input("poll", "n_intervals")],
)
def refresh(_n):
    pnl, risk, tca, acct, pos, prices = fetch_snapshots()

    # cards
    totals = _totals(pnl)
    eq = float(acct.get("equity", 0.0))
    cash = float(acct.get("cash", 0.0))
    bp = float(acct.get("buying_power", 0.0))
    ccy = acct.get("currency", "USD")

    def _card(title, value, sub=""):
        return html.Div(
            style={"padding": "12px", "border": "1px solid #eee", "borderRadius": "12px", "boxShadow": "0 1px 3px rgba(0,0,0,.06)"},
            children=[
                html.Div(title, style={"color": "#666", "fontSize": "13px"}),
                html.Div(value, style={"fontWeight": "700", "fontSize": "22px"}),
                html.Div(sub, style={"color": "#888", "fontSize": "12px"}),
            ],
        )

    cards = [
        _card("Equity", f"{eq:,.2f} {ccy}", f"Cash {cash:,.2f} • BP {bp:,.2f}"),
        _card("PnL (Total)", f"{totals['pnl']:,.2f}", f"Real {totals['realized']:,.2f} • Unrl {totals['unrealized']:,.2f} • Fees {totals['fees']:,.2f}"),
        _card("Orders (TCA)", str(_tca_totals(tca).get("orders", 0)), "weighted IS bps ~ " + (f"{_tca_totals(tca).get('is_bps_wavg'):.2f}" if _tca_totals(tca).get("is_bps_wavg") is not None else "n/a")),
        _card("Last Update", time.strftime("%H:%M:%S"), "Local time"),
    ]

    # heatmaps (strategy / region)
    df_strat = _bucket_to_df(pnl.get("by_strategy", {}), "strategy")
    df_region = _bucket_to_df(pnl.get("by_region", {}), "region")
    fig_strat = px.imshow(
        df_strat.set_index("strategy")[["pnl"]].T if not df_strat.empty else np.zeros((1,1)),
        labels=dict(color="PnL"),
        aspect="auto",
        title="PnL by Strategy (latest)",
    )
    if not df_strat.empty:
        fig_strat.update_yaxes(showticklabels=False)
        fig_strat.update_xaxes(tickangle=45)

    fig_region = px.imshow(
        df_region.set_index("region")[["pnl"]].T if not df_region.empty else np.zeros((1,1)),
        labels=dict(color="PnL"),
        aspect="auto",
        title="PnL by Region (latest)",
    )
    if not df_region.empty:
        fig_region.update_yaxes(showticklabels=False)
        fig_region.update_xaxes(tickangle=45)

    # PnL bars (top symbols)
    df_sym = _bucket_to_df(pnl.get("by_symbol", {}), "symbol").head(12)
    fig_pnl = px.bar(df_sym, x="symbol", y="pnl", title="Top Symbol Contributors", text="pnl")
    fig_pnl.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_pnl.update_layout(xaxis_tickangle=45, uniformtext_minsize=8, uniformtext_mode="hide")

    # risk table (as heatmap-like bar)
    df_risk = _risk_to_df(risk)
    if df_risk.empty:
        df_risk = pd.DataFrame([{"strategy": "total", "sharpe": 0, "sortino": 0, "vol": 0, "max_drawdown": 0, "var_95": 0}])
    fig_risk = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(df_risk.columns), fill_color="#111", font=dict(color="white")),
                cells=dict(values=[df_risk[c] for c in df_risk.columns]),
            )
        ]
    )
    fig_risk.update_layout(title="Risk Metrics")

    # positions table
    df_pos = _pos_to_df(pos, prices)
    pos_data = df_pos.to_dict("records")

    # TCA by strategy / region
    t_by_strat = tca.get("by_strategy", {})
    df_t_s = []
    for k, v in t_by_strat.items():
        df_t_s.append({"strategy": k, "is_bps_wavg": v.get("is_bps_wavg"), "fill_ratio_wavg": v.get("fill_ratio_wavg")})
    df_t_s = pd.DataFrame(df_t_s)
    fig_t_s = px.bar(df_t_s, x="strategy", y="is_bps_wavg", title="TCA: IS bps (weighted) by Strategy") if not df_t_s.empty else px.bar(title="TCA: IS bps (weighted) by Strategy")

    t_by_reg = tca.get("by_region", {})
    df_t_r = []
    for k, v in t_by_reg.items():
        df_t_r.append({"region": k, "is_bps_wavg": v.get("is_bps_wavg"), "fill_ratio_wavg": v.get("fill_ratio_wavg")})
    df_t_r = pd.DataFrame(df_t_r)
    fig_t_r = px.bar(df_t_r, x="region", y="is_bps_wavg", title="TCA: IS bps (weighted) by Region") if not df_t_r.empty else px.bar(title="TCA: IS bps (weighted) by Region")

    # TCA scatter: fill ratio vs IS bps
    per_orders = tca.get("per_order", []) or []
    df_sc = pd.DataFrame(per_orders)
    if not df_sc.empty and "fill_ratio" in df_sc and "IS_bps" in df_sc:
        fig_sc = px.scatter(df_sc, x="fill_ratio", y="IS_bps", color="strategy", hover_data=["symbol", "qty", "vwap_fill", "decision_px"], title="Per‑Order: Fill Ratio vs IS (bps)")
    else:
        fig_sc = px.scatter(title="Per‑Order: Fill Ratio vs IS (bps)")

    return cards, fig_strat, fig_region, fig_pnl, fig_risk, pos_data, fig_t_s, fig_t_r, fig_sc


# ---- entrypoint ----
def main():
    app.run_server(debug=True, host="127.0.0.1", port=8050)

if __name__ == "__main__":
    main()