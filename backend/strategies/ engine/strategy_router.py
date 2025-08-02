# strategy_router.py

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
    volatility_event_driven,
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
    multi_asset_trend_following,
)

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Regions supported
REGIONS = ["us", "india", "china", "europe", "japan"]

class StrategyRouter:
    def __init__(self, region="global", mode="all"):
        """
        region: one of ["us", "india", "china", "europe", "japan", "global"]
        mode: "alpha", "diversified", or "all"
        """
        self.region = region.lower()
        self.mode = mode.lower()

    def get_strategies(self):
        alpha_strategies = [
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
            volatility_event_driven,
        ]

        diversified_strategies = [
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
            multi_asset_trend_following,
        ]

        if self.mode == "alpha":
            return alpha_strategies
        elif self.mode == "diversified":
            return diversified_strategies
        else:
            return alpha_strategies + diversified_strategies

    def route(self, market_data):
        """
        Executes all active strategies and combines signals.

        Args:
            market_data: dict of current prices, fundamentals, news, etc.
        Returns:
            combined_signals: dict of {symbol: signal_weight}
        """
        signals = {}
        for strategy in self.get_strategies():
            try:
                s = strategy.run(market_data, region=self.region)
                for k, v in s.items():
                    signals[k] = signals.get(k, 0) + v
            except Exception as e:
                logger.error(f"Strategy {strategy.__name__} failed: {e}")
        return signals