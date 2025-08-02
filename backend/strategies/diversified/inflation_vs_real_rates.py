import yfinance as yf
import pandas as pd
import numpy as np

class InflationVsRealRates:
    """
    Long TIPS when real yields are falling (inflation protection),
    short TIPS when real yields are rising (deflationary regime).
    """
    def __init__(self, tips_symbol="TIP", bond_symbol="IEF"):
        self.tips_symbol = tips_symbol  # iShares TIPS ETF
        self.bond_symbol = bond_symbol  # 7–10 Year Treasury Bond ETF
        self.data = None
        self.signals = None

    def load_data(self):
        """Download data for TIPS and nominal bond ETF"""
        tips = yf.download(self.tips_symbol, period="1y", interval="1d")['Adj Close']
        bonds = yf.download(self.bond_symbol, period="1y", interval="1d")['Adj Close']

        df = pd.DataFrame({
            'TIPS': tips,
            'Bonds': bonds
        })

        df['TIPS_Return'] = df['TIPS'].pct_change()
        df['Bonds_Return'] = df['Bonds'].pct_change()

        # Real yield proxy: Nominal yield - Inflation expectation
        # Here: bond ETF return - TIPS return = real yield change
        df['Real_Yield_Proxy'] = df['Bonds_Return'] - df['TIPS_Return']
        self.data = df.dropna()

    def generate_signals(self):
        """Signal: long TIPS if real yields are falling"""
        if self.data is None:
            self.load_data()

        df = self.data.copy()

        # If real yield falling → long TIP (pro-inflation hedge)
        df['Signal'] = np.where(df['Real_Yield_Proxy'] < -0.001, 1,
                         np.where(df['Real_Yield_Proxy'] > 0.001, -1, 0))

        self.signals = df
        return df[['TIPS', 'Bonds', 'Real_Yield_Proxy', 'Signal']]

    def backtest(self):
        """Simple backtest using signals"""
        if self.signals is None:
            self.generate_signals()

        df = self.signals.copy()
        df['Strategy_Return'] = df['Signal'].shift(1) * df['TIPS_Return']
        df.dropna(inplace=True)
        return df[['TIPS', 'Bonds', 'Signal', 'Strategy_Return']]

    def latest_signal(self):
        """Returns most recent signal for use in live trading"""
        if self.signals is None:
            self.generate_signals()
        latest = self.signals.iloc[-1]
        return {
            "TIPS": latest["TIPS"],
            "Bonds": latest["Bonds"],
            "Real_Yield_Proxy": latest["Real_Yield_Proxy"],
            "Signal": latest["Signal"]
        }

# Debug usage
if __name__ == "__main__":
    strat = InflationVsRealRates()
    strat.load_data()
    strat.generate_signals()
    backtest_df = strat.backtest()
    print(strat.latest_signal())
    print(backtest_df.tail())