import pandas as pd
from backend.utils.logger import log

class MergerArbitrage:
    """
    Merger Arbitrage Strategy:
    - Long the target company
    - Short the acquirer (if stock deal)
    - Profit from the spread between current price and expected deal close price
    """

    def __init__(self, deals=None):
        """
        deals: List of dicts with fields:
            {
                'target': 'XYZ',
                'acquirer': 'ABC',
                'deal_price': 45.0,
                'current_price': 42.0,
                'deal_type': 'cash' or 'stock'
            }
        """
        self.deals = deals or []
        self.signals = []

    def evaluate_deals(self):
        """Analyze each deal's spread and return potential"""
        results = []
        for deal in self.deals:
            try:
                spread = deal["deal_price"] - deal["current_price"]
                spread_pct = (spread / deal["current_price"]) * 100

                signal = {
                    "Target": deal["target"],
                    "Acquirer": deal["acquirer"],
                    "Current Price": deal["current_price"],
                    "Deal Price": deal["deal_price"],
                    "Spread (%)": round(spread_pct, 2),
                    "Deal Type": deal["deal_type"],
                    "Signal": "LONG_TARGET" if spread_pct > 2 else "NO_ACTION"
                }

                if deal["deal_type"] == "stock" and spread_pct > 2:
                    signal["Acquirer Action"] = "SHORT"
                else:
                    signal["Acquirer Action"] = "NONE"

                results.append(signal)
            except Exception as e:
                log(f"[MergerArbitrage] Error processing deal: {e}")
        self.signals = results
        return results

    def get_trade_recommendations(self):
        """Allocates 5% per active merger deal"""
        allocations = {}
        for signal in self.signals:
            if signal["Signal"] == "LONG_TARGET":
                allocations[signal["Target"]] = 0.05
            if signal["Acquirer Action"] == "SHORT":
                allocations[f"SHORT_{signal['Acquirer']}"] = 0.05
        return allocations

# Example usage
if __name__ == "__main__":
    example_deals = [
        {"target": "XYZ", "acquirer": "ABC", "deal_price": 45.0, "current_price": 42.0, "deal_type": "cash"},
        {"target": "LMN", "acquirer": "PQR", "deal_price": 120.0, "current_price": 110.0, "deal_type": "stock"},
    ]

    strategy = MergerArbitrage(deals=example_deals)
    print(pd.DataFrame(strategy.evaluate_deals()))
    print(strategy.get_trade_recommendations())