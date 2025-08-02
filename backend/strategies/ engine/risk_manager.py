# risk_manager.py

import numpy as np
from backend.utils.logger import get_logger

logger = get_logger(__name__)

class RiskManager:
    def __init__(self,
                 max_exposure=1.0,
                 max_position_size=0.1,
                 volatility_target=0.15,
                 stop_loss_threshold=-0.10,
                 drawdown_limit=-0.15):
        """
        Parameters:
            max_exposure: Total exposure limit across portfolio.
            max_position_size: Max capital % allowed per trade.
            volatility_target: Target portfolio volatility.
            stop_loss_threshold: Max loss % allowed per position.
            drawdown_limit: Portfolio-level drawdown stop.
        """
        self.max_exposure = max_exposure
        self.max_position_size = max_position_size
        self.volatility_target = volatility_target
        self.stop_loss_threshold = stop_loss_threshold
        self.drawdown_limit = drawdown_limit
        self.current_drawdown = 0.0

    def apply_risk_filters(self, signals, portfolio, volatility_estimates, pnl_tracker):
        """
        Filter trade signals based on risk constraints.
        
        Args:
            signals: Dict of {symbol: signal weight}
            portfolio: Dict of current positions
            volatility_estimates: Dict of {symbol: volatility}
            pnl_tracker: Dict of {symbol: PnL %}
        Returns:
            filtered_signals: Dict of {symbol: adjusted weight}
        """
        filtered = {}
        total_weight = sum(abs(v) for v in signals.values())
        logger.info(f"Raw signals total weight: {total_weight:.2f}")

        for symbol, weight in signals.items():
            weight = np.clip(weight, -self.max_position_size, self.max_position_size)

            # Adjust based on volatility
            vol = volatility_estimates.get(symbol, 0.2)  # default fallback
            vol_adjustment = self.volatility_target / max(vol, 1e-3)
            adjusted_weight = weight * vol_adjustment

            # Stop-loss check
            pnl = pnl_tracker.get(symbol, 0)
            if pnl < self.stop_loss_threshold:
                logger.info(f"STOP LOSS hit for {symbol} ({pnl:.2%})")
                continue

            filtered[symbol] = np.clip(adjusted_weight, -self.max_position_size, self.max_position_size)

        # Normalize to max exposure
        if sum(abs(w) for w in filtered.values()) > self.max_exposure:
            scale = self.max_exposure / sum(abs(w) for w in filtered.values())
            filtered = {k: v * scale for k, v in filtered.items()}
            logger.info(f"Scaling weights down to fit max exposure: {self.max_exposure:.2f}")

        return filtered

    def update_drawdown(self, current_return, peak_return):
        """
        Updates portfolio drawdown and returns True if limit breached.
        """
        drawdown = (current_return - peak_return) / max(peak_return, 1e-3)
        self.current_drawdown = drawdown
        if drawdown < self.drawdown_limit:
            logger.warning(f"PORTFOLIO DRAWDOWN LIMIT BREACHED: {drawdown:.2%}")
            return True
        return False