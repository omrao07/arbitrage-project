# engine/regions.py

from strategies.alpha import (
    news_sentiment_alpha, recession_indicator_alpha, earnings_surprise_momentum,
    insider_transaction_alpha, post_ipo_drift_alpha, esg_alpha, analyst_revision_momentum,
    short_interest_alpha, alternative_credit_scoring, web_traffic_alpha,
    sentiment_policy_alpha, nlp_social_alpha, volatility_event_driven
)

from strategies.diversified import (
    interest_rate_differential, currency_carry_trade, central_bank_policy_divergence,
    inflation_vs_real_rates, global_liquidity_cycle, geopolitical_hedging,
    commodity_supply_demand, twin_deficits_arbitrage, capital_flow_rotation,
    bond_term_premium, factor_investing, mean_reversion, volatility_arbitrage,
    dispersion_trades, skew_arbitrage, convertible_bond_arbitrage, credit_spread_trading,
    calendar_spreads, cross_asset_correlation, retail_vs_institutional, crowded_trade_unwind,
    dark_pool_sentiment, earnings_drift_post_reaction, etf_flow_alpha, hedge_fund_replication,
    ipo_lockup_short, merger_arbitrage, macro_surprise_index, metals_demand_cycle,
    oil_inventory_surprise, gold_vs_real_yield, shipping_cost_arbitrage,
    green_transition_commodities, livestock_spread_trade, seasonal_commodity_patterns,
    energy_roll_yield, crack_spread_trading, implied_volatility_premium,
    policy_sentiment_model, macro_regime_switching, multi_asset_trend_following
)

# Common pool of strategy modules (can be class-based later)
ALPHA_STRATEGIES = [
    news_sentiment_alpha, recession_indicator_alpha, earnings_surprise_momentum,
    insider_transaction_alpha, post_ipo_drift_alpha, esg_alpha, analyst_revision_momentum,
    short_interest_alpha, alternative_credit_scoring, web_traffic_alpha,
    sentiment_policy_alpha, nlp_social_alpha, volatility_event_driven
]

DIVERSIFIED_STRATEGIES = [
    interest_rate_differential, currency_carry_trade, central_bank_policy_divergence,
    inflation_vs_real_rates, global_liquidity_cycle, geopolitical_hedging,
    commodity_supply_demand, twin_deficits_arbitrage, capital_flow_rotation,
    bond_term_premium, factor_investing, mean_reversion, volatility_arbitrage,
    dispersion_trades, skew_arbitrage, convertible_bond_arbitrage, credit_spread_trading,
    calendar_spreads, cross_asset_correlation, retail_vs_institutional,
    crowded_trade_unwind, dark_pool_sentiment, earnings_drift_post_reaction,
    etf_flow_alpha, hedge_fund_replication, ipo_lockup_short, merger_arbitrage,
    macro_surprise_index, metals_demand_cycle, oil_inventory_surprise, gold_vs_real_yield,
    shipping_cost_arbitrage, green_transition_commodities, livestock_spread_trade,
    seasonal_commodity_patterns, energy_roll_yield, crack_spread_trading,
    implied_volatility_premium, policy_sentiment_model, macro_regime_switching,
    multi_asset_trend_following
]

# Region-based grouping
def get_region_strategies(region):
    """Return both alpha + diversified strategies for the given region"""
    return {
        "region": region,
        "alpha": ALPHA_STRATEGIES,
        "diversified": DIVERSIFIED_STRATEGIES
    }

# Global combination
def get_global_model():
    """Return global model (all regions aggregated)"""
    return {
        "region": "global",
        "alpha": ALPHA_STRATEGIES,
        "diversified": DIVERSIFIED_STRATEGIES
    }

# Helper function to list supported regions
def get_supported_regions():
    return ["us", "india", "china", "europe", "japan"]