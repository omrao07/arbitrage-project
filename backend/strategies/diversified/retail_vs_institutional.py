import pandas as pd
from backend.utils.logger import log

class RetailVsInstitutionalDivergence:
    """
    Strategy:
    - Monitors flow divergence between retail and institutional investors.
    - If retail flows are high but institutions are exiting → SHORT (potential bubble).
    - If institutions are accumulating while retail is exiting → LONG.
    """

    def __init__(self, flow_data_path="backend/data/flow_data.csv", divergence_threshold=0.25):
        self.flow_data_path = flow_data_path
        self.divergence_threshold = divergence_threshold
        self.signals = {}

    def load_flow_data(self):
        try:
            df = pd.read_csv(self.flow_data_path, parse_dates=["date"])
            df.set_index("date", inplace=True)
            self.data = df
        except Exception as e:
            log(f"[RetailVsInstitutional] Failed to load flow data: {e}")
            self.data = pd.DataFrame()

    def compute_signals(self):
        self.signals = {}
        if self.data.empty:
            return self.signals

        for date, row in self.data.iterrows():
            retail = row.get("retail_inflow", 0)
            institutional = row.get("institutional_inflow", 0)

            # Calculate normalized divergence
            total_flow = abs(retail) + abs(institutional)
            if total_flow == 0:
                continue
            divergence = (retail - institutional) / total_flow

            if divergence > self.divergence_threshold:
                signal = "SHORT"
            elif divergence < -self.divergence_threshold:
                signal = "LONG"
            else:
                signal = "NEUTRAL"

            self.signals[date.strftime("%Y-%m-%d")] = {
                "RetailFlow": retail,
                "InstitutionalFlow": institutional,
                "Divergence": round(divergence, 3),
                "Signal": signal
            }

        return self.signals

    def latest_signal(self):
        if not self.signals:
            return None
        latest_date = max(self.signals.keys())
        return {latest_date: self.signals[latest_date]}

# Example usage
if __name__ == "__main__":
    strat = RetailVsInstitutionalDivergence()
    strat.load_flow_data()
    strat.compute_signals()
    print("Latest signal:", strat.latest_signal())