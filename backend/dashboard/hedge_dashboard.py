# backend/dashboard/hedge_dashboard.py
"""
Hedge Dashboard (Pro View)
--------------------------
Run:
  python -m backend.dashboard.hedge_dashboard
Deps:
  pip install dash plotly requests pandas

Env (optional):
  DATA_API_URL=http://localhost:8000
  DATA_API_KEY=dev123
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import dash # type: ignore
from dash import Dash, dcc, html, dash_table # type: ignore
from dash.dependencies import Input, Output, State # type: ignore

API_URL = os.getenv("DATA_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("DATA_API_KEY", "").strip()
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# --------------- helpers ---------------

def _get(path: str, default: Any):
    try:
        r = requests.get(f"{API_URL}{path}", headers=HEADERS, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default

def fetch_all():
    pnl = _get("/pnl", {})
    risk = _get("/risk", {})
    tca = _get("/tca", {})
    acct = _get("/account", {"equity": 0.0, "cash": 0.0, "buying_power": 0.0, "currency": "USD"})
    pos  = _get("/positions", {})
    pxs  = _get("/prices", {})
    news = _get("/news?limit=50", [])
    return pnl, risk, tca, acct, pos, pxs, news

def _totals(d: Dict[str, Any]) -> Dict[str, float]:
    t = (d or {}).get("totals") or {}
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
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pnl", ascending=False)
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
        "is_bps_wavg": tot.get("is_bps_wavg"),
        "fill_ratio_wavg": tot.get("fill_ratio_wavg"),
        "slip_bps_avg": tot.get("slippage_bps_vs_mid_avg"),
    }

def _news_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for e in items or []:
        score = e.get("score")
        if score is None: label = "NEU"
        elif score >= 0.25: label = "POS"
        elif score <= -0.25: label = "NEG"
        else: label = "NEU"
        rows.append({
            "time": time.strftime("%H:%M:%S", time.localtime(float(e.get("published_at", time.time())))),
            "source": e.get("source", ""),
            "symbol": e.get("symbol", ""),
            "headline": e.get("headline", ""),
            "score": None if score is None else round(float(score), 2),
            "label": label,
            "url": e.get("url", ""),
        })
    return pd.DataFrame(rows)

# --------------- app ---------------

app: Dash = dash.Dash(__name__)
app.title = "Hedge Dashboard"

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, sans-serif", "padding": "18px"},
    children=[
        html.H2("Hedge Dashboard"),
        dcc.Store(id="equity-store", data={"points": []}),  # client-side equity history
        html.Div(id="kpi-row", style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "marginTop": "16px"},
            children=[
                dcc.Graph(id="equity-curve"),
                dcc.Graph(id="risk-quick"),
            ],
        ),

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
                dcc.Graph(id="top-symbols"),
                dcc.Graph(id="tca-cards"),
            ],
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "marginTop": "16px"},
            children=[
                dcc.Graph(id="tca-scatter"),
                html.Div(children=[
                    html.H4("Live News"),
                    html.Div(id="news-tape", style={"height": "360px", "overflowY": "auto", "border": "1px solid #eee", "borderRadius": "10px", "padding": "8px"}),
                ]),
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

        dcc.Interval(id="poll", interval=3000, n_intervals=0),
        html.Div(style={"marginTop": "8px", "color": "#888"}, children=[
            "Source: ", html.Code(API_URL), " (set DATA_API_URL / DATA_API_KEY to change)"
        ]),
    ],
)

# --------------- callbacks ---------------

@app.callback(
    [
        Output("kpi-row", "children"),
        Output("equity-store", "data"),
        Output("equity-curve", "figure"),
        Output("risk-quick", "figure"),
        Output("heatmap-strategy", "figure"),
        Output("heatmap-region", "figure"),
        Output("top-symbols", "figure"),
        Output("positions-table", "data"),
        Output("tca-cards", "figure"),
        Output("tca-scatter", "figure"),
        Output("news-tape", "children"),
    ],
    [Input("poll", "n_intervals")],
    [State("equity-store", "data")],
)
def refresh(_n, eq_store):
    pnl, risk, tca, acct, pos, prices, news = fetch_all()

    # KPIs
    totals = _totals(pnl)
    eq = float(acct.get("equity", 0.0)); cash = float(acct.get("cash", 0.0)); bp = float(acct.get("buying_power", 0.0))
    ccy = acct.get("currency", "USD")
    def _card(title, value, sub=""):
        return html.Div(style={"padding": "12px", "border": "1px solid #eee", "borderRadius": "12px",
                               "boxShadow": "0 1px 3px rgba(0,0,0,.06)"},
                        children=[
                            html.Div(title, style={"color": "#666", "fontSize": "13px"}),
                            html.Div(value, style={"fontWeight": "700", "fontSize": "22px"}),
                            html.Div(sub, style={"color": "#888", "fontSize": "12px"}),
                        ])
    kpis = [
        _card("Equity", f"{eq:,.2f} {ccy}", f"Cash {cash:,.2f} • BP {bp:,.2f}"),
        _card("PnL (Total)", f"{totals['pnl']:,.2f}", f"Real {totals['realized']:,.2f} • Unrl {totals['unrealized']:,.2f} • Fees {totals['fees']:,.2f}"),
        _card("Risk", f"VaR {_safe_risk(risk):.2%}" if _safe_risk(risk) is not None else "VaR n/a",
              "Sharpe {:.2f}".format(_safe_sharpe(risk)) if _safe_sharpe(risk) is not None else "Sharpe n/a"),
        _card("Orders", str(_tca_totals(tca).get("orders", 0)),
              "IS bps ~ " + (f"{_tca_totals(tca).get('is_bps_wavg'):.2f}" if _tca_totals(tca).get("is_bps_wavg") is not None else "n/a")),
    ]

    # Equity curve (client-side accumulation)
    points = (eq_store or {}).get("points", [])
    points.append({"t": time.time(), "eq": eq})
    if len(points) > 800: points = points[-800:]
    eq_store = {"points": points}
    df_eq = pd.DataFrame(points)
    fig_eq = px.line(df_eq, x="t", y="eq", title="Equity (rolling session)") if not df_eq.empty else px.line(title="Equity")
    fig_eq.update_layout(xaxis_title="Time", yaxis_title=f"Equity ({ccy})")

    # Risk quick table
    df_risk = _risk_to_df(risk)
    if df_risk.empty:
        df_risk = pd.DataFrame([{"strategy": "total", "sharpe": 0, "sortino": 0, "vol": 0, "max_drawdown": 0, "var_95": 0}])
    fig_risk = go.Figure(data=[go.Table(
        header=dict(values=list(df_risk.columns), fill_color="#111", font=dict(color="white")),
        cells=dict(values=[df_risk[c] for c in df_risk.columns]),
    )])
    fig_risk.update_layout(title="Risk Metrics")

    # Heatmaps
    df_s = _bucket_to_df(pnl.get("by_strategy", {}), "strategy")
    df_r = _bucket_to_df(pnl.get("by_region", {}), "region")
    fig_s = px.imshow(df_s.set_index("strategy")[["pnl"]].T if not df_s.empty else np.zeros((1,1)),
                      labels=dict(color="PnL"), aspect="auto", title="PnL by Strategy")
    if not df_s.empty: fig_s.update_yaxes(showticklabels=False); fig_s.update_xaxes(tickangle=45)
    fig_reg = px.imshow(df_r.set_index("region")[["pnl"]].T if not df_r.empty else np.zeros((1,1)),
                        labels=dict(color="PnL"), aspect="auto", title="PnL by Region")
    if not df_r.empty: fig_reg.update_yaxes(showticklabels=False); fig_reg.update_xaxes(tickangle=45)

    # Top symbols bar
    df_sym = _bucket_to_df(pnl.get("by_symbol", {}), "symbol").head(12)
    fig_sym = px.bar(df_sym, x="symbol", y="pnl", title="Top Symbol Contributors", text="pnl")
    fig_sym.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_sym.update_layout(xaxis_tickangle=45, uniformtext_minsize=8, uniformtext_mode="hide")

    # Positions
    pos_df = _pos_to_df(pos, prices)
    pos_data = pos_df.to_dict("records")

    # TCA quick cards + scatter
    ttot = _tca_totals(tca)
    fig_tcac = go.Figure()
    fig_tcac.add_trace(go.Indicator(mode="number+delta", value=(ttot.get("is_bps_wavg") or 0),
                                    title={"text": "IS (bps, wavg)"}, delta={"reference": 0}))
    fig_tcac.add_trace(go.Indicator(mode="number", value=(ttot.get("fill_ratio_wavg") or 0),
                                    title={"text": "Fill Ratio (wavg)"}))
    fig_tcac.add_trace(go.Indicator(mode="number", value=(ttot.get("slip_bps_avg") or 0),
                                    title={"text": "Slippage vs Mid (bps)"}))
    fig_tcac.update_layout(grid={"rows": 1, "columns": 3}, title="Execution Quality")

    per = (tca or {}).get("per_order", []) or []
    df_sc = pd.DataFrame(per)
    if not df_sc.empty and "fill_ratio" in df_sc and "IS_bps" in df_sc:
        fig_sc = px.scatter(df_sc, x="fill_ratio", y="IS_bps", color="strategy",
                            hover_data=["symbol", "qty", "vwap_fill", "decision_px"],
                            title="Per‑Order: Fill Ratio vs IS (bps)")
    else:
        fig_sc = px.scatter(title="Per‑Order: Fill Ratio vs IS (bps)")

    # News tape
    df_news = _news_df(news)
    tape_children = []
    if df_news.empty:
        tape_children = [html.Div("No news yet.", style={"color": "#777"})]
    else:
        for _, r in df_news.head(25).iterrows():
            badge = {"POS": "#16a34a", "NEG": "#dc2626", "NEU": "#6b7280"}[r["label"]]
            tape_children.append(
                html.Div(style={"padding": "6px 8px", "borderBottom": "1px solid #eee"},
                         children=[
                            html.Span(r["time"], style={"color": "#888", "fontSize": "12px", "marginRight": "8px"}),
                            html.Span(r["source"], style={"color": "#555", "fontSize": "12px", "marginRight": "8px"}),
                            html.Span(r["symbol"] or "", style={"color": "#0ea5e9", "fontSize": "12px", "marginRight": "8px"}),
                            html.Span(r["headline"], style={"fontWeight": 600}),
                            html.Span(f"  ({'%.2f' % r['score'] if r['score'] is not None else '—'})",
                                      style={"color": badge, "fontWeight": 700, "marginLeft": "6px"}),
                         ])
            )

    return kpis, eq_store, fig_eq, fig_risk, fig_s, fig_reg, fig_sym, pos_data, fig_tcac, fig_sc, tape_children


# ---- small risk helpers ----
def _safe_risk(risk: Dict[str, Any]):
    # look for a 'total' block or average var_95
    if "total" in (risk or {}):
        v = risk["total"].get("var_95")
        return None if v is None else float(v)
    vals = [v.get("var_95") for v in (risk or {}).values() if isinstance(v, dict) and v.get("var_95") is not None]
    return float(np.mean(vals)) if vals else None # pyright: ignore[reportCallIssue, reportArgumentType]

def _safe_sharpe(risk: Dict[str, Any]):
    if "total" in (risk or {}):
        v = risk["total"].get("sharpe")
        return None if v is None else float(v)
    vals = [v.get("sharpe") for v in (risk or {}).values() if isinstance(v, dict) and v.get("sharpe") is not None]
    return float(np.mean(vals)) if vals else None # type: ignore


# ---- entrypoint ----
def main():
    app.run_server(debug=True, host="127.0.0.1", port=8051)

if __name__ == "__main__":
    main()