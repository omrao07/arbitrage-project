import pandas as pd
from backend.utils.loader import load_macro_data
from backend.utils.logger import log

class MacroSurpriseIndex:
    """
    Strategy that trades based on economic data surprises:
    - Long equities if surprises are mostly positive
    - Long bonds or defensive assets if mostly negative
    """

    def __init__(self, region="us", surprise_threshold=0.0):
        self.region = region.lower()
        self.surprise_threshold = surprise_threshold
        self.data = None
        self.signal = None

    def load_data(self):
        """Loads macroeconomic forecast vs. actual surprise data"""
        try:
            self.data = load_macro_data(self.region)
        except Exception as e:
            log(f"[MacroSurpriseIndex] Failed to load macro data: {e}")
            self.data = pd.DataFrame()

    def compute_surprise_index(self):
        """Calculate the average surprise index from recent macro indicators"""
        df = self.data.copy()
        if "Surprise" not in df.columns:
            log("[MacroSurpriseIndex] Missing 'Surprise' column in macro data.")
            return None

        # Consider last 6 months of data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values("Date")
        recent_data = df[df['Date'] > df['Date'].max() - pd.Timedelta(days=180)]
        
        if recent_data.empty:
            log("[MacroSurpriseIndex] No recent macro surprise data available.")
            return None

        avg_surprise = recent_data["Surprise"].mean()
        self.signal = {
            "Avg_Surprise_Index": round(avg_surprise, 4),
            "Region": self.region,
            "Signal": "LONG_RISK" if avg_surprise > self.surprise_threshold else "DEFENSIVE_TILT"
        }
        return self.signal

    def get_trade_recommendation(self):
        """Returns a portfolio tilt suggestion"""
        if self.signal is None:
            self.load_data()
            self.compute_surprise_index()
        if not self.signal:
            return {}

        if self.signal["Signal"] == "LONG_RISK":
            return {"Equities": 0.6, "Commodities": 0.2, "Bonds": 0.2}
        else:
            return {"Equities": 0.3, "Commodities": 0.1, "Bonds": 0.6}

# Example Usage
if __name__ == "__main__":
    strategy = MacroSurpriseIndex(region="us")
    strategy.load_data()
    print(strategy.compute_surprise_index())
    print(strategy.get_trade_recommendation())