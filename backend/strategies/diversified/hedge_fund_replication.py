import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class HedgeFundReplication:
    """
    Strategy that attempts to replicate hedge fund returns using factor models
    such as market, size, value, and momentum (Fama-French + optional factors).
    """

    def __init__(self, fund_returns: pd.DataFrame, factor_data: pd.DataFrame):
        """
        Parameters:
        - fund_returns: DataFrame with ['Date', 'Return'] of hedge fund index
        - factor_data: DataFrame with ['Date', 'MKT', 'SMB', 'HML', 'MOM', ...]
        """
        self.fund_returns = fund_returns.set_index('Date')
        self.factor_data = factor_data.set_index('Date')
        self.model = LinearRegression()
        self.coefficients = None
        self.replication_weights = None
        self.predicted_returns = None

    def fit_model(self):
        """Fit linear factor model to hedge fund index returns"""
        df = self.fund_returns.join(self.factor_data, how='inner')
        X = df.drop(columns=['Return'])
        y = df['Return']

        self.model.fit(X, y)
        self.coefficients = dict(zip(X.columns, self.model.coef_))
        self.replication_weights = pd.Series(self.model.coef_, index=X.columns)
        return self.replication_weights

    def generate_signals(self):
        """Generate expected hedge fund return using factor exposure"""
        if self.replication_weights is None:
            self.fit_model()

        self.predicted_returns = self.factor_data.dot(self.replication_weights)
        signal_df = pd.DataFrame({
            'Date': self.factor_data.index,
            'PredictedReturn': self.predicted_returns
        }).reset_index(drop=True)

        signal_df['Signal'] = np.where(signal_df['PredictedReturn'] > 0, 1, -1)
        return signal_df

    def backtest(self):
        """Backtest signals using actual hedge fund index returns"""
        signals = self.generate_signals()
        merged = pd.merge(signals, self.fund_returns.reset_index(), on='Date', how='inner')
        merged['StrategyPnL'] = merged['Signal'] * merged['Return']
        return merged[['Date', 'Signal', 'Return', 'StrategyPnL']]

    def latest_signal(self):
        """Return latest signal"""
        if self.predicted_returns is None:
            self.generate_signals()
        return {
            "PredictedReturn": self.predicted_returns.iloc[-1],
            "Signal": 1 if self.predicted_returns.iloc[-1] > 0 else -1
        }


# Example usage
if __name__ == "__main__":
    dates = pd.date_range(end=pd.Timestamp.today(), periods=250)
    fund_returns = pd.DataFrame({
        'Date': dates,
        'Return': np.random.normal(0.001, 0.01, size=250)
    })

    factor_data = pd.DataFrame({
        'Date': dates,
        'MKT': np.random.normal(0, 0.01, size=250),
        'SMB': np.random.normal(0, 0.01, size=250),
        'HML': np.random.normal(0, 0.01, size=250),
        'MOM': np.random.normal(0, 0.01, size=250),
    })

    strategy = HedgeFundReplication(fund_returns, factor_data)
    weights = strategy.fit_model()
    signals = strategy.generate_signals()
    results = strategy.backtest()
    print(strategy.latest_signal())
    print(results.tail())