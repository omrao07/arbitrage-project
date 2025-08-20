# backend/dashboard/literacy_dashboard.py
"""
Financial Literacy Dashboard
----------------------------
Purpose: To make finance concepts approachable for students / communities.

Run:
  python -m backend.dashboard.literacy_dashboard

Deps:
  pip install dash plotly requests pandas
"""

from __future__ import annotations
import os, time
import requests, pandas as pd, numpy as np
import dash # type: ignore
from dash import Dash, dcc, html, dash_table # type: ignore
from dash.dependencies import Input, Output # type: ignore

API_URL = os.getenv("DATA_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("DATA_API_KEY", "").strip()
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# ----------------- helpers -----------------
def _get(path: str, default):
    try:
        r = requests.get(f"{API_URL}{path}", headers=HEADERS, timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default

def fetch_literacy():
    acct = _get("/account", {"equity": 0.0, "cash": 0.0, "currency": "USD"})
    pnl  = _get("/pnl", {})
    news = _get("/news?limit=20", [])
    return acct, pnl, news

# ----------------- app -----------------
app: Dash = dash.Dash(__name__)
app.title = "Financial Literacy Dashboard"

app.layout = html.Div(
    style={"fontFamily": "Inter, sans-serif", "padding": "20px"},
    children=[
        html.H2("📊 Financial Literacy Dashboard"),
        html.P("A simplified view to teach finance concepts to students and communities."),

        html.Div(id="acct-cards", style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"}),

        html.Div(style={"marginTop": "20px"}, children=[
            html.H4("What is happening with our portfolio?"),
            dcc.Graph(id="pnl-pie"),
        ]),

        html.Div(style={"marginTop": "20px"}, children=[
            html.H4("Learning Corner"),
            html.Ul([
                html.Li("💰 Equity: Total value of your investments + cash."),
                html.Li("📈 PnL: Profit & Loss — how much you’ve gained or lost."),
                html.Li("📰 News sentiment shows how markets might react."),
                html.Li("🎯 Diversification: Don’t put all eggs in one basket!"),
            ]),
        ]),

        html.Div(style={"marginTop": "20px"}, children=[
            html.H4("Latest Market News"),
            html.Div(id="news-list", style={"border": "1px solid #eee", "borderRadius": "10px", "padding": "10px", "maxHeight": "300px", "overflowY": "auto"}),
        ]),

        dcc.Interval(id="poll", interval=5000, n_intervals=0),
        html.Div(style={"marginTop": "8px", "color": "#777"}, children=[
            "Source: ", html.Code(API_URL)
        ]),
    ]
)

# ----------------- callbacks -----------------
@app.callback(
    [Output("acct-cards", "children"),
     Output("pnl-pie", "figure"),
     Output("news-list", "children")],
    [Input("poll", "n_intervals")]
)
def refresh(_n):
    acct, pnl, news = fetch_literacy()
    eq = float(acct.get("equity", 0.0)); cash = float(acct.get("cash", 0.0)); ccy = acct.get("currency", "USD")
    totals = (pnl or {}).get("totals", {})
    realized = float(totals.get("realized", 0.0) or 0.0)
    unrealized = float(totals.get("unrealized", 0.0) or 0.0)
    fees = float(totals.get("fees", 0.0) or 0.0)
    total = float(totals.get("pnl", 0.0) or 0.0)

    def card(title, val, sub=""):
        return html.Div(style={"padding": "12px", "border": "1px solid #eee", "borderRadius": "12px",
                               "boxShadow": "0 1px 2px rgba(0,0,0,.05)"}, children=[
            html.Div(title, style={"color": "#666", "fontSize": "13px"}),
            html.Div(val, style={"fontWeight": "700", "fontSize": "20px"}),
            html.Div(sub, style={"color": "#888", "fontSize": "12px"}),
        ])

    cards = [
        card("Equity", f"{eq:,.2f} {ccy}", f"Cash {cash:,.2f}"),
        card("Profit / Loss", f"{total:,.2f}", f"Real {realized:,.2f} • Unreal {unrealized:,.2f}"),
        card("Fees", f"{fees:,.2f}", "Cost of trading"),
    ]

    import plotly.express as px
    labels = ["Realized", "Unrealized", "Fees"]
    values = [realized, unrealized, fees]
    fig = px.pie(names=labels, values=values, title="PnL Breakdown")

    news_children = []
    for n in news or []:
        label = "NEU"
        score = n.get("score")
        if score is not None:
            if score >= 0.25: label = "POS"
            elif score <= -0.25: label = "NEG"
        badge_color = {"POS": "#16a34a", "NEG": "#dc2626", "NEU": "#6b7280"}[label]
        news_children.append(
            html.Div(style={"padding": "6px 8px", "borderBottom": "1px solid #eee"}, children=[
                html.Span(n.get("source",""), style={"color":"#888","marginRight":"8px"}),
                html.Span(n.get("headline",""), style={"fontWeight":600}),
                html.Span(f" ({label})", style={"color":badge_color, "fontWeight":600, "marginLeft":"6px"}),
            ])
        )

    if not news_children:
        news_children = [html.Div("No news available.", style={"color":"#777"})]

    return cards, fig, news_children

# ----------------- entrypoint -----------------
def main():
    app.run_server(debug=True, host="127.0.0.1", port=8052)

if __name__ == "__main__":
    main()