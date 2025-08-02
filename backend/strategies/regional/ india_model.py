from backend.strategies.alpha import (
    news_sentiment_alpha,
    recession_indicator_alpha,
    earnings_surprise_momentum,
    insider_transaction_alpha,
    post_ipo_drift_alpha,
    esg_alpha,
    analyst_revision_momentum,
    short_interest_alpha,
    alternative_credit_scoring,
    web_traffic_alpha,
    sentiment_policy_alpha,
    nlp_social_alpha,
    volatility_event_driven
)

from backend.strategies.diversified import (
    interest_rate_differential,
    currency_carry_trade,
    central_bank_policy_divergence,
    inflation_vs_real_rates,
    global_liquidity_cycle,
    geopolitical_hedging,
    commodity_supply_demand,
    twin_deficits_arbitrage,
    capital_flow_rotation,
    bond_term_premium,
    factor_investing,
    mean_reversion,
    volatility_arbitrage,
    dispersion_trades,
    skew_arbitrage,
    convertible_bond_arbitrage,
    credit_spread_trading,
    calendar_spreads,
    cross_asset_correlation,
    retail_vs_institutional,
    crowded_trade_unwind,
    dark_pool_sentiment,
    earnings_drift_post_reaction,
    etf_flow_alpha,
    hedge_fund_replication,
    ipo_lockup_short,
    merger_arbitrage,
    macro_surprise_index,
    metals_demand_cycle,
    oil_inventory_surprise,
    gold_vs_real_yield,
    shipping_cost_arbitrage,
    green_transition_commodities,
    livestock_spread_trade,
    seasonal_commodity_patterns,
    energy_roll_yield,
    crack_spread_trading,
    implied_volatility_premium,
    policy_sentiment_model,
    macro_regime_switching,
    multi_asset_trend_following
)

from backend.engine.aggregator import SignalAggregator
from backend.engine.risk_manager import RiskManager
from backend.engine.execution_engine import ExecutionEngine
from backend.utils.loader import load_macro_data

class IndiaMacroModel:
    def __init__(self):
        self.alpha_strategies = [
            news_sentiment_alpha,
            recession_indicator_alpha,
            earnings_surprise_momentum,
            insider_transaction_alpha,
            post_ipo_drift_alpha,
            esg_alpha,
            analyst_revision_momentum,
            short_interest_alpha,
            alternative_credit_scoring,
            web_traffic_alpha,
            sentiment_policy_alpha,
            nlp_social_alpha,
            volatility_event_driven
        ]

        self.diversified_strategies = [
            interest_rate_differential,
            currency_carry_trade,
            central_bank_policy_divergence,
            inflation_vs_real_rates,
            global_liquidity_cycle,
            geopolitical_hedging,
            commodity_supply_demand,
            twin_deficits_arbitrage,
            capital_flow_rotation,
            bond_term_premium,
            factor_investing,
            mean_reversion,
            volatility_arbitrage,
            dispersion_trades,
            skew_arbitrage,
            convertible_bond_arbitrage,
            credit_spread_trading,
            calendar_spreads,
            cross_asset_correlation,
            retail_vs_institutional,
            crowded_trade_unwind,
            dark_pool_sentiment,
            earnings_drift_post_reaction,
            etf_flow_alpha,
            hedge_fund_replication,
            ipo_lockup_short,
            merger_arbitrage,
            macro_surprise_index,
            metals_demand_cycle,
            oil_inventory_surprise,
            gold_vs_real_yield,
            shipping_cost_arbitrage,
            green_transition_commodities,
            livestock_spread_trade,
            seasonal_commodity_patterns,
            energy_roll_yield,
            crack_spread_trading,
            implied_volatility_premium,
            policy_sentiment_model,
            macro_regime_switching,
            multi_asset_trend_following
        ]

        self.aggregator = SignalAggregator()
        self.risk_manager = RiskManager()
        self.executor = ExecutionEngine()

    def run_model(self):
        # Load macroeconomic indicators for India
        macro_data = load_macro_data("india_macro.csv")

        # Step 1: Run all strategies and collect signals
        strategy_signals = {}

        for strategy in self.alpha_strategies + self.diversified_strategies:
            try:
                signal = strategy.generate_signal(macro_data)
                strategy_signals[strategy.__name__] = signal
            except Exception as e:
                print(f"[{strategy.__name__}] failed: {e}")

        # Step 2: Aggregate signals into weighted exposures
        combined_signal = self.aggregator.combine(strategy_signals)

        # Step 3: Adjust for risk (e.g., correlation, VaR, stress tests)
        adjusted_weights = self.risk_manager.adjust(combined_signal, macro_data)

        # Step 4: Execute and return positions
        positions = self.executor.execute(adjusted_weights)
        return positions

# Optional run for testing
if __name__ == "__main__":
    model = IndiaMacroModel()
    positions = model.run_model()
    print("Final Allocations:", positions)