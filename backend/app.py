
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import random
import traceback
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, jsonify, request, render_template
import threading
import os

app = Flask(__name__)

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
lookback_days = 90
threshold_z = 2.0
log_file = "arbitrage_log.csv"
capital = 100000  # starting capital in USD
portfolio_value = capital
risk_tolerance = 0.03

# === TOGGLES ===
ENABLE_ML = True
ENABLE_EXECUTION = True
ENABLE_CRYPTO = True
ENABLE_VAR = True
ENABLE_BETA = True

# === DATA COLLECTION ===
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", progress=False, auto_adjust=True)
        df = df[df.index <= pd.Timestamp.today()]
        df = df[~df['Close'].isna()]
        if df.empty or df.shape[0] < 5:
            return pd.Series(dtype=float)
        return df['Close']
    except Exception:
        return pd.Series(dtype=float)

def fetch_all_data():
    data = {}
    for t in tickers:
        close = get_stock_data(t)
        if isinstance(close, pd.Series) and not close.empty:
            data[t] = close
    if not data:
        raise ValueError("❌ No valid stock data found. Exiting.")
    return pd.DataFrame(data).dropna()

# === CORE STRATEGIES ===
def check_cross_stock_arbitrage():
    df = fetch_all_data()
    trades = []
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            spread = df.iloc[:, i] - df.iloc[:, j]
            z = (spread.iloc[-1] - spread.mean()) / spread.std()
            if abs(z) > threshold_z:
                trades.append((df.columns[i], df.columns[j], z))
    return trades

def check_statistical_arbitrage():
    df = fetch_all_data()
    returns = df.pct_change().dropna()
    corr_mean = returns.corr().mean().mean()
    return corr_mean

# === ML MODEL FORECASTING ===
def run_ml_models():
    if not ENABLE_ML:
        return []
    df = fetch_all_data()
    target = df[tickers[0]]
    features = df.drop(columns=[tickers[0]])
    X = features.values
    y = target.values

    models = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(alpha=0.01),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X, y)
        score = model.score(X, y)
        results.append((name, score))
    return results

# === RISK + VAR ===
def calculate_var():
    if not ENABLE_VAR:
        return 0
    df = fetch_all_data()
    returns = df.pct_change().dropna().mean(axis=1)
    var = np.percentile(returns, 5)
    return var

# === BETA CALC ===
def calculate_beta():
    if not ENABLE_BETA:
        return 0
    portfolio = fetch_all_data().pct_change().dropna().mean(axis=1)
    market = yf.download('^GSPC', period='90d', progress=False, auto_adjust=True)
    market_returns = market['Close'].pct_change().dropna() if 'Close' in market.columns else market['Adj Close'].pct_change().dropna()
    combined = pd.concat([portfolio, market_returns], axis=1).dropna()
    combined.columns = ['portfolio', 'market']
    beta = np.cov(combined['portfolio'], combined['market'])[0][1] / combined['market'].var()
    return beta

# === API ROUTES ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run')
def run_engine():
    try:
        cross_trades = check_cross_stock_arbitrage()
        corr = check_statistical_arbitrage()
        ml_results = run_ml_models()
        var = calculate_var()
        beta = calculate_beta()

        return jsonify({
            'cross_arbitrage': cross_trades,
            'correlation': corr,
            'ml_models': ml_results,
            'value_at_risk': var,
            'beta': beta
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
