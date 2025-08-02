import pandas as pd
import numpy as np
from datetime import datetime
from backend.utils.loader import load_macro_data
from backend.utils.logger import log

class MacroRegimeSwitching:
    """
    Strategy that switches portfolio allocation based on macroeconomic regime:
    - Inflation rising + growth slowing → Defensive
    - Growth rising + inflation stable → Pro-cyclical
    - Inflation falling + growth falling → Bonds
    - Growth rising + inflation rising → Commodities
    """

    def __init__(self, region="us"):
        self.region = region.lower()
        self.macro_data = None
        self.signal = None

    def load_data(self):
        """Loads region-specific macro data"""
        try:
            self.macro_data = load_macro_data(self.region)
        except Exception as e:
            log(f"[MacroRegimeSwitching] Error loading macro data: {e}")
            self.macro_data = pd.DataFrame()

    def classify_regime(self):
        """Assign macroeconomic regime labels"""
        df = self.macro_data.copy()
        df['Growth_Change'] = df['GDP_Growth'].diff()
        df['Inflation_Change'] = df['Inflation'].diff()

        latest = df.iloc[-1]
        g = latest['Growth_Change']
        i = latest['Inflation_Change']

        if g > 0 and i > 0:
            regime = 'Stagflation Risk / Commodities Tilt'
        elif g > 0 and i <= 0:
            regime = 'Expansion / Pro-Cyclicals Tilt'
        elif g <= 0 and i > 0:
            regime = 'Late Cycle / Defensive Tilt'
        else:
            regime = 'Recession / Bonds Tilt'

        self.signal = {
            "Regime": regime,
            "GDP_Growth": round(latest['GDP_Growth'], 2),
            "Inflation": round(latest['Inflation'], 2),
            "Signal_Date": latest['Date'].strftime("%Y-%m-%d")
        }
        return self.signal

    def get_allocation_signal(self):
        """Returns asset allocation suggestion based on macro regime"""
        if self.signal is None:
            self.load_data()
            self.classify_regime()

        regime = self.signal["Regime"]
        if "Expansion" in regime:
            return {"Equities": 0.6, "Commodities": 0.2, "Bonds": 0.2}
        elif "Stagflation" in regime:
            return {"Equities": 0.2, "Commodities": 0.5, "Bonds": 0.3}
        elif "Late Cycle" in regime:
            return {"Equities": 0.3, "Commodities": 0.2, "Bonds": 0.5}
        else:  # Recession
            return {"Equities": 0.1, "Commodities": 0.1, "Bonds": 0.8}

# Example Usage
if __name__ == "__main__":
    strat = MacroRegimeSwitching(region="us")
    strat.load_data()
    print(strat.classify_regime())
    print(strat.get_allocation_signal())