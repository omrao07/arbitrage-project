from flask import Blueprint, request, jsonify, send_file
from datetime import datetime, timedelta
import pandas as pd
import io
import random

pnl_bp = Blueprint('pnl', __name__)

# Dummy list of strategies and regions — adjust to match your project
REGIONS = ["us", "india", "china", "europe", "japan"]
STRATEGIES = [
    "interest_rate_differential", "currency_carry_trade", "recession_indicator_alpha",
    "earnings_surprise_momentum", "insider_transaction_alpha", "short_interest_alpha",
    "macro_regime_switching", "esg_alpha", "nlp_social_alpha", "commodity_supply_demand",
]

def generate_dummy_pnl_data():
    """Simulate 30 days of P&L for each region + strategy."""
    data = []
    for region in REGIONS:
        for strategy in STRATEGIES:
            for i in range(30):
                date = datetime.now() - timedelta(days=i)
                pnl = round(random.uniform(-1000, 1200), 2)
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "region": region,
                    "strategy": strategy,
                    "pnl": pnl,
                    "color": "green" if pnl >= 0 else "red"
                })
    df = pd.DataFrame(data)
    return df

# In-memory cache
pnl_df = generate_dummy_pnl_data()

@pnl_bp.route('/pnl', methods=['GET'])
def get_pnl():
    """Return filtered or full P&L data."""
    region = request.args.get("region")
    strategy = request.args.get("strategy")
    start = request.args.get("start_date")
    end = request.args.get("end_date")

    df = pnl_df.copy()

    if region:
        df = df[df["region"] == region]
    if strategy:
        df = df[df["strategy"] == strategy]
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]

    df = df.sort_values("date", ascending=False)

    return jsonify(df.to_dict(orient="records"))

@pnl_bp.route('/pnl/download', methods=['GET'])
def download_pnl():
    """Download P&L data as CSV."""
    df = pnl_df.copy()
    region = request.args.get("region")
    strategy = request.args.get("strategy")

    if region:
        df = df[df["region"] == region]
    if strategy:
        df = df[df["strategy"] == strategy]

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='pnl_export.csv'
    )