import pandas as pd
import numpy as np

def compute_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Compute annualized Sharpe ratio from daily returns.
    """
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-8)

def allocate_weights(strategy_metrics: pd.DataFrame, method="sharpe") -> pd.Series:
    """
    Allocates weights to strategies based on selected method.

    Parameters:
    - strategy_metrics (pd.DataFrame): Must contain columns ['strategy', 'returns', 'sharpe', 'volatility'].
    - method (str): 'sharpe', 'inverse_volatility', 'equal', 'historical_return'

    Returns:
    - pd.Series: Weights indexed by strategy.
    """
    if method == "sharpe":
        scores = strategy_metrics['sharpe'].clip(lower=0)
    elif method == "inverse_volatility":
        scores = 1 / (strategy_metrics['volatility'] + 1e-8)
    elif method == "historical_return":
        scores = strategy_metrics['returns'].clip(lower=0)
    elif method == "equal":
        return pd.Series(1 / len(strategy_metrics), index=strategy_metrics['strategy'])
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    normalized_weights = scores / scores.sum()
    return pd.Series(normalized_weights.values, index=strategy_metrics['strategy'])

def rebalance_weights(strategy_returns: dict, method="sharpe"):
    """
    Compute weights from raw return series dictionary.

    Parameters:
    - strategy_returns (dict): {strategy_name: pd.Series of returns}

    Returns:
    - pd.Series: Weights
    """
    metrics = []
    for name, returns in strategy_returns.items():
        returns = pd.Series(returns).dropna()
        sharpe = compute_sharpe_ratio(returns)
        vol = returns.std()
        ret = returns.mean() * 252
        metrics.append({'strategy': name, 'sharpe': sharpe, 'volatility': vol, 'returns': ret})

    metrics_df = pd.DataFrame(metrics)
    return allocate_weights(metrics_df, method=method)