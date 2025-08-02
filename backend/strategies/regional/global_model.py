from ...models import india_model, china_model, europe_model, japan_model, us_model
from ...engine.aggregator import aggregate_signals
from ...engine.risk_manager import RiskManager
from ...utils.weight_allocator import allocate_weights

class GlobalMacroModel:
    def __init__(self, capital=1_000_000):
        self.capital = capital
        self.risk_manager = RiskManager()
        self.regions = {
            "US": us_model.USMacroModel(),
            "India": india_model.IndiaMacroModel(),
            "China": china_model.ChinaMacroModel(),
            "Europe": europe_model.EuropeMacroModel(),
            "Japan": japan_model.JapanMacroModel()
        }

    def run(self, data_dict):
        """
        Run all regional models and compile a global signal portfolio.
        :param data_dict: Dictionary with keys 'US', 'India', etc. each mapping to region-specific data
        :return: Global weighted portfolio
        """
        region_signals = {}

        for region_name, model in self.regions.items():
            region_data = data_dict.get(region_name, {})
            signals = model.generate_signals(region_data)
            region_signals[region_name] = signals

        # Step 1: Aggregate all regional signals
        combined_signals = aggregate_signals(region_signals)

        # Step 2: Allocate capital across signals based on region Sharpe ratios
        region_sharpes = {
            region: self.risk_manager.estimate_sharpe(signals)
            for region, signals in region_signals.items()
        }
        weights = allocate_weights(region_sharpes)

        # Step 3: Apply risk constraints and get final portfolio
        final_portfolio = self.risk_manager.apply_risk_budget(combined_signals, weights, self.capital)

        return final_portfolio